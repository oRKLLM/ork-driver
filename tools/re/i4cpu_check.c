/* i4cpu_check.c — validate ork_pack_i4a8_cpu_blob is BYTE-IDENTICAL to the NPU path
 * (ork_mm_pack_i4a8_im + ork_w_dump_i4a8). Packs the same random f32 [N][K] weight both ways and memcmps.
 * Exercises a few shapes. Exit 0 = bit-exact, nonzero = mismatch.  ./i4cpu_check
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(ork_npu *c, int K, int N) {
    size_t nel = (size_t) N * K;
    float *f32 = malloc(nel * sizeof(float));
    srand(1234 + K + N);
    for (size_t i = 0; i < nel; i++) f32[i] = ((rand() % 4001) - 2000) / 2000.0f;  /* [-1,1] */

    /* NPU path: pack (bcreate+tile) then serialize the compact int4 blob */
    ork_npu_set_pack_domain(c, 0);
    ork_w *w = ork_mm_pack_i4a8_im(c, K, N, f32, NULL, NULL);
    if (!w) { printf("K=%d N=%d: NPU pack FAILED\n", K, N); free(f32); return 1; }
    size_t nb = ork_w_dump_i4a8(w, NULL, 0);
    void *bnpu = malloc(nb); ork_w_dump_i4a8(w, bnpu, nb);
    ork_mm_free(c, w);

    /* CPU path: pack straight to the same blob, no NPU */
    size_t cb = ork_pack_i4a8_cpu_blob(c, K, N, f32, NULL, NULL, 0);
    void *bcpu = malloc(cb); size_t got = ork_pack_i4a8_cpu_blob(c, K, N, f32, NULL, bcpu, cb);

    int ok = (nb == cb) && (got == cb) && (memcmp(bnpu, bcpu, nb) == 0);
    if (!ok) {
        size_t diff = 0, first = (size_t)-1;
        size_t m = nb < cb ? nb : cb;
        for (size_t i = 0; i < m; i++) if (((char*)bnpu)[i] != ((char*)bcpu)[i]) { if (first==(size_t)-1) first=i; diff++; }
        printf("K=%d N=%d: MISMATCH npu=%zu cpu=%zu(%zu) diffbytes=%zu first@%zu\n", K, N, nb, cb, got, diff, first);
    } else {
        printf("K=%d N=%d: OK (%zu bytes bit-identical)\n", K, N, nb);
    }
    free(f32); free(bnpu); free(bcpu);
    return ok ? 0 : 1;
}

int main(void) {
    ork_npu *c = ork_npu_init(); if (!c) { printf("no board\n"); return 0; }
    int rc = 0;
    rc |= check(c, 1024, 512);
    rc |= check(c, 2048, 256);
    rc |= check(c, 3584, 128);
    ork_npu_free(c);
    printf(rc ? "FAIL\n" : "ALL BIT-EXACT\n");
    return rc;
}
