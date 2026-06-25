// persist_probe — validate the .orkpack persist primitive: pack a weight, dump its tile bytes, reload
// them with ork_mm_load_i8 (no dequant/quant/tile), and confirm the reloaded weight is byte-identical
// NPU input (dump(load(dump(pack))) == dump(pack)). Also times pack (tile) vs load (DMA copy).
//
// Byte-equality is decisive: the NPU reads Bb verbatim, so identical Bb bytes ⇒ identical matmul.
//
//   build:  gcc -O2 -Iinclude -Isrc -pthread tools/persist_probe.c \
//             src/npu.c src/soc.c src/soc/rk3588.c src/soc/rk3576.c -lm -o persist_probe
//   run:    sudo ./persist_probe [K=4096] [N=4096]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "ork_npu.h"

static double ms_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e3 + t.tv_nsec/1e6; }

int main(int argc, char **argv) {
    int K = argc > 1 ? atoi(argv[1]) : 4096;
    int N = argc > 2 ? atoi(argv[2]) : 4096;
    ork_npu *c = ork_npu_init();
    if (!c) { fprintf(stderr, "ork_npu_init failed\n"); return 1; }

    int8_t *B = malloc((size_t) K * N);
    if (!B) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (size_t i = 0; i < (size_t) K * N; i++) B[i] = (int8_t)((i * 2654435761u) >> 24);  // deterministic pseudo-random

    double t0 = ms_now();
    ork_w *w1 = ork_mm_pack_i8(c, K, N, B);
    double t_pack = ms_now() - t0;
    if (!w1) { fprintf(stderr, "pack failed\n"); return 1; }

    size_t blobn = ork_w_dump(w1, NULL, 0);
    void *blob1 = malloc(blobn), *blob2 = malloc(blobn);
    ork_w_dump(w1, blob1, blobn);

    t0 = ms_now();
    ork_w *w2 = ork_mm_load_i8(c, K, N, blob1, blobn);
    double t_load = ms_now() - t0;
    if (!w2) { fprintf(stderr, "FAIL: ork_mm_load_i8 returned NULL (shape/size mismatch)\n"); return 2; }

    size_t n2 = ork_w_dump(w2, blob2, blobn);
    int match = (n2 == blobn) && (memcmp(blob1, blob2, blobn) == 0);

    printf("persist_probe: %dx%d int8, blob=%.1f MB\n", K, N, blobn / 1e6);
    printf("  roundtrip byte-equal: %s\n", match ? "YES — reloaded weight is identical NPU input" : "NO — MISMATCH");
    printf("  pack(tile)=%.1f ms   load(DMA copy)=%.1f ms   (load %.1fx %s)\n",
           t_pack, t_load, t_pack > t_load ? t_pack / t_load : t_load / t_pack, t_pack > t_load ? "faster" : "slower");
    free(B); free(blob1); free(blob2);
    return match ? 0 : 2;
}
