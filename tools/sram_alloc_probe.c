/* sram_alloc_probe — prove the NPU on-chip SRAM is actually ALLOCATABLE (not just reported).
 * Allocates a buffer with RKNPU_MEM_TRY_ALLOC_SRAM and verifies it landed in SRAM two ways:
 *   (1) the driver's RKNPU_GET_FREE_SRAM_SIZE drops by the alloc size while it is live, and
 *   (2) the mem_create.sram_size out-field reports how many bytes came from SRAM.
 * Then destroys it and checks the free pool is restored. Read-only w.r.t. the NPU (no submit).
 *   cc -O2 -Isrc -o sram_alloc_probe tools/sram_alloc_probe.c && sudo ./sram_alloc_probe [KB]
 */
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include "rknpu_ioctl.h"

static unsigned q(int fd, unsigned action) {
    struct rknpu_action a; memset(&a, 0, sizeof a); a.flags = action;
    if (ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &a) != 0) return 0xffffffffu;
    return a.value;
}

int main(int argc, char **argv) {
    size_t kb = argc > 1 ? (size_t)atoi(argv[1]) : 256;
    size_t sz = kb * 1024;
    int fd = open("/dev/dri/card1", O_RDWR);
    if (fd < 0) { perror("open card1"); return 1; }

    unsigned tot = q(fd, RKNPU_GET_TOTAL_SRAM_SIZE), fre0 = q(fd, RKNPU_GET_FREE_SRAM_SIZE);
    printf("SRAM total=%u (%.1f KB)  free=%u (%.1f KB)\n", tot, tot/1024.0, fre0, fre0/1024.0);
    if (tot == 0 || tot == 0xffffffffu) { fprintf(stderr, "no SRAM exposed — nothing to test\n"); return 1; }

    /* base flags used by ork-driver's normal resident buffers (0x403), plus the SRAM bit;
     * also try +KERNEL_MAPPING (0x40b) in case the SRAM path wants a kernel mapping. */
    uint32_t cands[] = {
        RKNPU_MEM_NON_CONTIGUOUS|RKNPU_MEM_CACHEABLE|RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT|RKNPU_MEM_TRY_ALLOC_SRAM,           /* 0x503 */
        RKNPU_MEM_NON_CONTIGUOUS|RKNPU_MEM_CACHEABLE|RKNPU_MEM_KERNEL_MAPPING|RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT|RKNPU_MEM_TRY_ALLOC_SRAM, /* 0x50b */
        RKNPU_MEM_CONTIGUOUS|RKNPU_MEM_KERNEL_MAPPING|RKNPU_MEM_TRY_ALLOC_SRAM,                                                /* 0x108 */
        RKNPU_MEM_TRY_ALLOC_SRAM,                                                                                              /* 0x100 */
    };
    for (unsigned i = 0; i < sizeof cands/sizeof *cands; i++) {
        struct rknpu_mem_create c; memset(&c, 0, sizeof c);
        c.size = sz; c.flags = cands[i]; c.core_mask = RKNPU_CORE0_MASK; c.iommu_domain_id = 0;
        printf("\n[flags=0x%03x] MEM_CREATE %zu KB ... ", cands[i], kb);
        if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &c) != 0) { printf("FAIL errno=%d (%s)\n", errno, strerror(errno)); continue; }
        unsigned fre1 = q(fd, RKNPU_GET_FREE_SRAM_SIZE);
        long drop = (long)fre0 - (long)fre1;
        printf("ok handle=%u dma=0x%llx sram_size=%llu (%.1f KB)  free %u->%u (drop=%ld B, %.1f KB)\n",
               c.handle, (unsigned long long)c.dma_addr, (unsigned long long)c.sram_size, c.sram_size/1024.0,
               fre0, fre1, drop, drop/1024.0);
        int in_sram = (c.sram_size > 0) || (drop >= (long)sz);
        printf("   -> %s\n", in_sram ? "IN SRAM ✔ (weight tile can live on-chip)" : "fell back to DRAM (not SRAM)");
        struct rknpu_mem_destroy d; memset(&d, 0, sizeof d); d.handle = c.handle; d.obj_addr = c.obj_addr;
        if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &d) != 0) printf("   (destroy errno=%d)\n", errno);
        unsigned fre2 = q(fd, RKNPU_GET_FREE_SRAM_SIZE);
        printf("   after destroy: free=%u (%s)\n", fre2, fre2 == fre0 ? "restored" : "NOT fully restored");
        if (in_sram) { close(fd); return 0; }
    }
    close(fd);
    return 2;
}
