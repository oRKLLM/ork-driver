// stream_probe — prove that ork_mm_free actually reclaims NPU DMA/IOVA, the prerequisite for
// layer-streaming (running models bigger than the 4 GiB IOVA window by cycling weights through it).
//
// Packs a large int8 weight, frees it with ork_mm_free, and repeats N times. The *total cycled*
// (N x weight) far exceeds 4 GiB, but only one weight is resident at a time. If free reclaims IOVA,
// every pack succeeds; if it leaks (the old ork_w_free behaviour), pack fails once ~4 GiB has been
// allocated (~16 iters at 256 MB) — the dma_probe wall.
//
//   build:  gcc -O2 -Iinclude -Isrc -pthread tools/stream_probe.c \
//             src/npu.c src/soc.c src/soc/rk3588.c src/soc/rk3576.c -lm -o stream_probe
//   run:    sudo ./stream_probe [iters=100] [K=8192] [N=32768]   (8192x32768 int8 = 256 MB/weight)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 100;
    int K     = argc > 2 ? atoi(argv[2]) : 8192;
    int N     = argc > 3 ? atoi(argv[3]) : 32768;   // K*N int8 = 256 MB per weight
    ork_npu *c = ork_npu_init();
    if (!c) { fprintf(stderr, "ork_npu_init failed\n"); return 1; }

    size_t wbytes = (size_t) K * N;
    int8_t *B = malloc(wbytes);
    if (!B) { fprintf(stderr, "host alloc %.0f MB failed\n", wbytes / 1e6); return 1; }
    memset(B, 1, wbytes);
    double mb = wbytes / 1e6;
    printf("stream_probe: %d x pack/free of %dx%d int8 (%.0f MB each) = %.1f GB cycled, <=1 resident\n",
           iters, K, N, mb, iters * mb / 1000.0);

    for (int i = 0; i < iters; i++) {
        ork_w *w = ork_i8_mm_pack(c, K, N, B);
        if (!w) {
            printf("FAIL: pack #%d failed after %.1f GB cycled — IOVA NOT reclaimed (leak)\n", i, i * mb / 1000.0);
            free(B); ork_npu_free(c); return 2;
        }
        ork_mm_free(c, w);
        if ((i + 1) % 16 == 0) printf("  ... %3d packs ok (%.1f GB cycled)\n", i + 1, (i + 1) * mb / 1000.0);
    }
    printf("RESULT: %d x %.0f MB = %.1f GB cycled through the 4 GiB window with <=256 MB resident — RECLAIM WORKS\n",
           iters, mb, iters * mb / 1000.0);
    free(B); ork_npu_free(c);
    return 0;
}
