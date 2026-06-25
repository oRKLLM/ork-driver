// dma_probe — measure the NPU's DMA-mappable capacity in isolation (no model, no ggml).
//
// Allocates DMA buffers (rknpu MEM_CREATE, like ork-driver's bcreate) in a loop until the kernel
// refuses, and reports the total. This isolates the *hardware* DMA/IOVA window from system RAM and
// from any model-load path — the single number that decides whether a full int4 27B (~13.5 GB) or a
// full MoE expert set (~17.5 GB) can ever be NPU-resident, or whether partial offload is the ceiling.
//
//   build:  gcc -O2 -I src tools/dma_probe.c -o dma_probe
//   run:    sudo ./dma_probe [card] [chunkMB=64] [map=1]
//             card    : /dev/dri/cardN (default: autodetect the rknpu card)
//             chunkMB : per-allocation size (default 64)
//             map     : 1 = CREATE+MAP+mmap (real-use path, also consumes host RAM)
//                       0 = CREATE only (isolates the IOVA/DMA window from host address space)
//
// Run with NO model loaded to measure the full window; with a model loaded it reports the remainder.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include "rknpu_ioctl.h"

static int create_one(int fd, size_t size, int do_map, uint32_t flags) {
    struct rknpu_mem_create c; memset(&c, 0, sizeof c);
    c.size = (size + 4095) & ~((size_t) 4095);
    c.flags = flags; c.core_mask = RKNPU_CORE0_MASK;
    if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &c)) return -errno;
    if (do_map) {
        struct rknpu_mem_map m; memset(&m, 0, sizeof m); m.handle = c.handle;
        if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &m)) return -errno;
        void *p = mmap(NULL, c.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, m.offset);
        if (p == MAP_FAILED) return -errno;
        // touch one byte per page-ish so the mapping is actually backed (real-use parity)
        ((volatile char *) p)[0] = 0;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *card = (argc > 1) ? argv[1] : getenv("ORK_NPU_CARD");
    size_t chunk = ((argc > 2) ? (size_t) atoll(argv[2]) : 64) * 1024UL * 1024UL;
    int do_map = (argc > 3) ? atoi(argv[3]) : 1;

    int fd = -1; char found[32] = {0};
    if (card) { fd = open(card, O_RDWR); }
    else {
        for (int i = 0; i < 8 && fd < 0; i++) {
            snprintf(found, sizeof found, "/dev/dri/card%d", i);
            int t = open(found, O_RDWR); if (t < 0) continue;
            struct rknpu_mem_create c; memset(&c, 0, sizeof c);
            c.size = 4096; c.flags = 0x403; c.core_mask = RKNPU_CORE0_MASK;
            if (ioctl(t, DRM_IOCTL_RKNPU_MEM_CREATE, &c) == 0) { fd = t; card = found; }
            else close(t);
        }
    }
    if (fd < 0) { fprintf(stderr, "dma_probe: no rknpu card found (try passing /dev/dri/cardN)\n"); return 1; }

    printf("dma_probe: card=%s chunk=%zuMB mode=%s\n", card, chunk / (1024 * 1024), do_map ? "CREATE+mmap" : "CREATE-only");
    size_t total = 0; int n = 0;
    for (;;) {
        int r = create_one(fd, chunk, do_map, 0x403);
        if (r) { printf("FAIL at buffer #%d after %.2f GB (errno=%d %s)\n", n, total / 1e9, -r, strerror(-r)); break; }
        n++; total += chunk;
        if (n % 16 == 0) printf("  ... %4d buffers, %6.2f GB\n", n, total / 1e9);
        if (total > 40UL * 1024 * 1024 * 1024) { printf("reached 40 GB without failure; stopping\n"); break; }
    }
    printf("RESULT: NPU DMA window ~= %.2f GB  (%d x %zuMB, %s)\n",
           total / 1e9, n, chunk / (1024 * 1024), do_map ? "create+mmap" : "create-only");
    return 0;
}
