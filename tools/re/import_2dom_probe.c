/* import_2dom_probe.c — the last isolable variable for the 7B import scale-fault: switching between TWO
 * domains that are BOTH heavily import-loaded. Fill dom0 and dom1 each with W imported 16MB weights, then
 * alternately submit a dom0 weight and a dom1 weight (forcing a dom_activate switch each time) while both
 * domains are full of imports. If a submit faults here (but the single-full-domain fill did not), the 7B
 * fault is "switch between multiple import-heavy domains." ~2.4GB/domain, guard-capped, bounded.
 *   ./i2d [W_per_domain=150] */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    int W = argc > 1 ? atoi(argv[1]) : 150;
    ork_npu *c = ork_npu_init();
    if (!c) { printf("no board\n"); return 0; }
    const int K = 1024, N = 8192, M = 8;   /* 8MB Bb + 8MB Bf = 16MB/weight, 2 imports */
    int8_t *B = calloc((size_t)K*N,1); for (int i=0;i<K*N;i++) B[i]=1;
    int8_t *A = calloc((size_t)M*K,1); for (int i=0;i<M*K;i++) A[i]=1;
    int32_t *C = calloc((size_t)M*N,4);

    ork_npu_set_pack_domain(c, 0);
    ork_w *wt = ork_mm_pack_i8(c, K, N, B);
    size_t need = ork_w_dump(wt, NULL, 0);
    void *blob = malloc(need); ork_w_dump(wt, blob, need); ork_mm_free(c, wt);

    ork_w *first0 = NULL, *first1 = NULL;
    for (int dom = 0; dom <= 1; dom++) {
        int got = 0;
        for (int i = 0; i < W; i++) {
            ork_npu_set_pack_domain(c, dom);
            ork_w *wi = ork_mm_load_i8_import(c, K, N, blob, need);
            if (!wi) break;
            if (dom == 0 && !first0) first0 = wi;
            if (dom == 1 && !first1) first1 = wi;
            if (ork_mm_run_i8(c, wi, M, A, C)) { printf("run fail during fill dom%d #%d\n", dom, i); break; }
            got++;
        }
        printf("filled domain %d with %d imports (~%dMB)\n", dom, got, got*16);
    }
    if (!first0 || !first1) { printf("fill incomplete\n"); return 1; }

    printf("=== now ALTERNATING submits across two import-heavy domains (switch each time) ===\n");
    for (int j = 0; j < 50; j++) {
        memset(C, 0, (size_t)M*N*4);
        int r0 = ork_mm_run_i8(c, first0, M, A, C);            /* switch -> dom0 (full) */
        int c0 = C[0];
        memset(C, 0, (size_t)M*N*4);
        int r1 = ork_mm_run_i8(c, first1, M, A, C);            /* switch -> dom1 (full) */
        int c1 = C[0];
        if (r0 || r1 || c0 != K || c1 != K) {
            printf("*** FAULT at alternation #%d: dom0 rc=%d C=%d | dom1 rc=%d C=%d (expect %d)\n",
                   j, r0, c0, r1, c1, K);
            return 0;
        }
        if (j % 8 == 0) printf("  alternation #%d OK (dom0 C=%d dom1 C=%d)\n", j, c0, c1);
    }
    printf("=== 50 alternations across two full import-domains: ALL OK ===\n");
    return 0;
}
