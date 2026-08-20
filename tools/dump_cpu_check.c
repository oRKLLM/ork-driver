// Validate ork_i8_w_dump_cpu produces bytes identical to ork_i8_mm_pack() + ork_w_dump()
// (the NPU-IOVA path). If they match, the CPU-only dump is a correct .orkpack producer with no NPU.
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(ork_npu *c, int K, int N) {
    int8_t *B = malloc((size_t)K * N);
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = (int8_t)((i * 2654435761u) >> 24);  // deterministic pseudo-random

    ork_w *w = ork_i8_mm_pack(c, K, N, B);
    if (!w) { printf("K=%d N=%d: pack_i8 FAILED\n", K, N); free(B); return 1; }
    size_t na = ork_w_dump(w, NULL, 0);
    size_t nb = ork_i8_w_dump_cpu(c, K, N, B, NULL, 0);
    if (na != nb) { printf("K=%d N=%d: SIZE mismatch npu=%zu cpu=%zu\n", K, N, na, nb); free(B); return 1; }
    char *a = malloc(na), *b = malloc(nb);
    ork_w_dump(w, a, na);
    ork_i8_w_dump_cpu(c, K, N, B, b, nb);
    int mism = memcmp(a, b, na);
    printf("K=%d N=%d: %zu bytes, %s\n", K, N, na, mism == 0 ? "BYTE-IDENTICAL OK" : "MISMATCH");
    free(a); free(b); free(B); ork_mm_free(c, w);
    return mism != 0;
}

int main(void) {
    ork_npu *c = ork_npu_init();
    if (!c) { printf("no NPU\n"); return 1; }
    int bad = 0;
    bad |= check(c, 512, 512);
    bad |= check(c, 2048, 2048);
    bad |= check(c, 2048, 6144);
    bad |= check(c, 6144, 2048);
    bad |= check(c, 3584, 18944);
    ork_npu_free(c);
    printf(bad ? "FAIL\n" : "ALL BYTE-IDENTICAL — CPU dump is a correct NPU-free .orkpack producer\n");
    return bad;
}
