/* domain1_import_test.c — isolate whether an IMPORTED weight faults when submitted in a NON-zero IOMMU
 * domain, vs a NATIVE (bcreate) weight in the same domain. Packs one tiny int8 matmul three ways and runs
 * it: native-into-domain-1, import-into-domain-1, import-into-domain-0 (control). A=all-1s[M,K], B=all-1s
 * [K,N] => correct C[m][n] == K. rc!=0 or C[0]!=K on the import-domain-1 case = the multi-domain import bug.
 *   gcc -O2 -Iinclude -Isrc -pthread domain1_import_test.c <CORE> -lm -o d1t && sudo ./d1t   (board) */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void run_case(ork_npu *c, const char *tag, ork_w *w, int M, int K, int N,
                     const int8_t *A, int32_t *C) {
    memset(C, 0, (size_t)M*N*4);
    if (!w) { printf("%-22s WEIGHT NULL (pack/load failed)\n", tag); return; }
    int rc = ork_mm_run_i8(c, w, M, A, C);
    printf("%-22s rc=%d  C[0]=%d  (expect %d)  -> %s\n",
           tag, rc, C[0], K, (rc==0 && C[0]==K) ? "OK" : "*** FAULT/WRONG ***");
}

int main(void) {
    ork_npu *c = ork_npu_init();
    if (!c) { printf("no board\n"); return 0; }
    const int K = 1024, N = 64, M = 8;
    int8_t *B = calloc((size_t)K*N, 1); for (int i=0;i<K*N;i++) B[i]=1;
    int8_t *A = calloc((size_t)M*K, 1); for (int i=0;i<M*K;i++) A[i]=1;
    int32_t *C = calloc((size_t)M*N, 4);

    /* WARM DOMAIN 0 first — the real model flow always establishes domain 0 (init warmup + layer-0 weights)
     * before the residence manager fills domain 1+. Skipping this and jumping cold to domain 1 is not a valid
     * test of the sliding window. */
    ork_npu_set_pack_domain(c, 0);
    ork_w *w0 = ork_mm_pack_i8(c, K, N, B);
    run_case(c, "NATIVE  domain0(warm)", w0, M, K, N, A, C);

    /* NATIVE (bcreate) weight in DOMAIN 1 — control: does the sliding-window submit path itself work? */
    ork_npu_set_pack_domain(c, 1);
    ork_w *wn = ork_mm_pack_i8(c, K, N, B);
    run_case(c, "NATIVE  domain1", wn, M, K, N, A, C);

    /* Build the pre-tiled blob (what a .orkpack holds) from the native weight, then IMPORT it. */
    size_t need = ork_w_dump(wn, NULL, 0);
    void *blob = malloc(need);
    if (ork_w_dump(wn, blob, need) != need) { printf("dump size mismatch\n"); return 1; }
    printf("(blob=%zu bytes)\n", need);

    /* IMPORT (dma-buf PRIME) weight in DOMAIN 1, accessed while domain 1 is already active (no switch). */
    ork_npu_set_pack_domain(c, 1);
    ork_w *wi1 = ork_mm_load_i8_import(c, K, N, blob, need);
    run_case(c, "IMPORT  dom1 in-domain", wi1, M, K, N, A, C);

    /* Switch AWAY to domain 0 (native), then... */
    run_case(c, "NATIVE  dom0 (switch away)", w0, M, K, N, A, C);

    /* ...switch BACK to domain 1 and re-run the IMPORTED weight. THIS is the real sliding-window case:
     * an imported weight in a non-0 domain, accessed after a dom_activate SWITCH *to* its domain. */
    run_case(c, "IMPORT  dom1 AFTER switch", wi1, M, K, N, A, C);

    return 0;
}
