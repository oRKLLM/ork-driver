// domain_probe — can the RK3588 NPU reside >4 GiB by splitting weights across multiple IOMMU domains?
//
// dma_probe.c showed a SINGLE iommu domain EFAULTs at ~4 GiB (the rk_iommu v2 32-bit IOVA cap).
// librkllmrt is observed to reside a 6.6+ GiB model across 2 domains. The rknpu uABI exposes the
// selector: struct rknpu_mem_create.iommu_domain_id (per-buffer) and RKNPU_SET_IOMMU_DOMAIN_ID
// (action 25). This probe asks the decisive question EMPIRICALLY:
//
//   Does allocating ~half the buffers in domain 0 and ~half in domain 1 let the TOTAL resident
//   exceed 4 GiB (target 6-7 GiB), where a single domain caps at ~4 GiB?
//
// Modes (argv[1]):
//   single   : alloc all buffers in domain 0 -> baseline cap (should EFAULT ~4 GiB), sanity check.
//   create N : MEM_CREATE buffers with iommu_domain_id alternating across N domains (default N=2).
//              Tests whether the FIELD alone places buffers in distinct domains that stay resident.
//   switch N : like create, but call RKNPU_SET_IOMMU_DOMAIN_ID(action 25) to make domain `d` active
//              before allocating each buffer in domain d. Mirrors how the driver dmesg shows
//              "switch iommu domain from 0 to 1". This is the mechanism librkllmrt likely uses.
//
//   build:  gcc -O2 -I src tools/domain_probe.c -o domain_probe   (on the board)
//   run:    sudo timeout -s INT 120 ./domain_probe [mode=switch] [ndom=2] [chunkMB=64] [card]
//
// EXPECTED: a per-domain EFAULT at ~4 GiB is NORMAL and frees on exit; it is NOT a wedge.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include "rknpu_ioctl.h"

#define MAXDOM 8
#define MAXBUF 4096

static int open_card(const char **card_out) {
    const char *card = getenv("ORK_NPU_CARD");
    if (card) { int fd = open(card, O_RDWR); if (fd >= 0) { *card_out = card; return fd; } }
    static char found[32];
    for (int i = 0; i < 8; i++) {
        snprintf(found, sizeof found, "/dev/dri/card%d", i);
        int t = open(found, O_RDWR); if (t < 0) continue;
        struct rknpu_mem_create c; memset(&c, 0, sizeof c);
        c.size = 4096; c.flags = 0x403; c.core_mask = RKNPU_CORE0_MASK;
        if (ioctl(t, DRM_IOCTL_RKNPU_MEM_CREATE, &c) == 0) { *card_out = found; return t; }
        close(t);
    }
    return -1;
}

// Set the active IOMMU domain via action 25. Returns 0 on success.
static int set_active_domain(int fd, int dom) {
    struct rknpu_action a; memset(&a, 0, sizeof a);
    a.flags = RKNPU_SET_IOMMU_DOMAIN_ID; a.value = (uint32_t) dom;
    if (ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &a)) return -errno;
    return 0;
}
static int get_active_domain(int fd) {
    struct rknpu_action a; memset(&a, 0, sizeof a);
    a.flags = RKNPU_GET_IOMMU_DOMAIN_ID; a.value = 0;
    if (ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &a)) return -errno;
    return (int) a.value;
}

// Allocate one buffer in domain `dom` (field on mem_create). dma_out gets the IOVA.
static int create_one(int fd, size_t size, int dom, uint64_t *dma_out, uint32_t *handle_out) {
    struct rknpu_mem_create c; memset(&c, 0, sizeof c);
    c.size = (size + 4095) & ~((size_t) 4095);
    c.flags = 0x403; // CONTIGUOUS-ish per dma_probe (CACHEABLE|IOMMU|KERNEL_MAPPING bits used there)
    c.core_mask = RKNPU_CORE0_MASK;
    c.iommu_domain_id = dom;
    if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &c)) return -errno;
    if (dma_out) *dma_out = c.dma_addr;
    if (handle_out) *handle_out = c.handle;
    return 0;
}

int main(int argc, char **argv) {
    const char *mode = (argc > 1) ? argv[1] : "switch";
    int ndom = (argc > 2) ? atoi(argv[2]) : 2;
    size_t chunk = ((argc > 3) ? (size_t) atoll(argv[3]) : 64) * 1024UL * 1024UL;
    const char *card = (argc > 4) ? argv[4] : NULL;
    if (card) setenv("ORK_NPU_CARD", card, 1);
    if (ndom < 1) ndom = 1; if (ndom > MAXDOM) ndom = MAXDOM;
    if (!strcmp(mode, "single")) ndom = 1;

    const char *cname = "?";
    int fd = open_card(&cname);
    if (fd < 0) { fprintf(stderr, "domain_probe: no rknpu card\n"); return 1; }

    int use_switch = !strcmp(mode, "switch");
    int active0 = get_active_domain(fd);
    printf("domain_probe: card=%s mode=%s ndom=%d chunk=%zuMB  (initial active domain query=%d%s)\n",
           cname, mode, ndom, chunk / (1024 * 1024), active0,
           active0 < 0 ? " [GET unsupported]" : "");

    // Probe whether SET works at all up front (for clear reporting).
    if (use_switch) {
        int r = set_active_domain(fd, 1);
        printf("  SET_IOMMU_DOMAIN_ID(1) -> %s\n", r ? strerror(-r) : "OK");
        if (r) printf("  (action 25 not supported; falling back to field-only placement)\n"), use_switch = 0;
        set_active_domain(fd, 0);
    }

    size_t per_dom[MAXDOM]; memset(per_dom, 0, sizeof per_dom);
    int per_dom_n[MAXDOM]; memset(per_dom_n, 0, sizeof per_dom_n);
    int dead[MAXDOM]; memset(dead, 0, sizeof dead);  // domain hit its cap
    size_t total = 0; int nbuf = 0;
    uint64_t first_dma[MAXDOM]; for (int i=0;i<MAXDOM;i++) first_dma[i]=0;

    for (;;) {
        // pick a domain round-robin among those not yet dead
        int dom = -1;
        for (int t = 0; t < ndom; t++) {
            int d = (nbuf + t) % ndom;
            if (!dead[d]) { dom = d; break; }
        }
        if (dom < 0) { printf("all %d domain(s) hit their cap.\n", ndom); break; }

        if (use_switch) {
            int sr = set_active_domain(fd, dom);
            if (sr) { printf("  SET(%d) failed mid-run: %s\n", dom, strerror(-sr)); }
        }
        uint64_t dma = 0; uint32_t h = 0;
        int r = create_one(fd, chunk, dom, &dma, &h);
        if (r) {
            printf("  dom%d FAIL at its %d-th buf (dom total %.2f GB): errno=%d %s -> marking dom%d full\n",
                   dom, per_dom_n[dom], per_dom[dom] / 1e9, -r, strerror(-r), dom);
            dead[dom] = 1;
            continue;
        }
        if (per_dom_n[dom] == 0) first_dma[dom] = dma;
        per_dom[dom] += chunk; per_dom_n[dom]++;
        total += chunk; nbuf++;
        if (nbuf % 16 == 0) {
            printf("  ... %4d bufs, TOTAL %6.2f GB | ", nbuf, total / 1e9);
            for (int d = 0; d < ndom; d++) printf("dom%d=%.2fGB%s ", d, per_dom[d]/1e9, dead[d]?"(full)":"");
            printf("\n");
        }
        if (total > 30UL * 1024 * 1024 * 1024 || nbuf >= MAXBUF) { printf("reached cap/limit, stopping.\n"); break; }
    }

    printf("\n==== RESULT (mode=%s) ====\n", mode);
    printf("TOTAL resident: %.2f GB across %d buffers, %d domain(s)\n", total / 1e9, nbuf, ndom);
    for (int d = 0; d < ndom; d++)
        if (per_dom_n[d]) printf("  domain %d: %.2f GB (%d bufs)  first IOVA=0x%llx\n",
                                 d, per_dom[d]/1e9, per_dom_n[d], (unsigned long long) first_dma[d]);
    printf("VERDICT: %s 4 GiB total %s multi-domain.\n",
           total > 4.0e9 ? ">" : "<=", total > 4.0e9 ? "EXCEEDED via" : "NOT exceeded with");
    return 0;
}
