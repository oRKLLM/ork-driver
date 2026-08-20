/* import_count_probe.c — isolate COUNT vs BYTES for the multi-domain import scale-fault. Accumulate many
 * TINY imported dma-buf weights (K=1024,N=64 => Bb+Bf ~128KB each, 2 foreign mappings each) in domain 1,
 * running each after a dom_activate switch (dom0->dom1). Total bytes stay small (~100-200MB) so IOVA can't
 * fill — if a submit faults at some COUNT of foreign mappings while bytes are trivial, the limit is COUNT
 * (# of PRIME imports / fragmentation), not size => the fix is domain-consolidation (one big dma-buf). If it
 * never faults up to ~3000 mappings (~the 7B's count) at low bytes, the fault is byte/size-driven instead.
 *   gcc -O2 -Iinclude -Isrc -pthread import_count_probe.c <CORE> -lm -o icp && sudo env ... ./icp [maxWeights]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    int maxW = argc > 1 ? atoi(argv[1]) : 1500;
    ork_npu *c = ork_npu_init();
    if (!c) { printf("no board\n"); return 0; }
    const int K = 1024, N = argc > 2 ? atoi(argv[2]) : 64, M = 8;   /* N large => big tiles (fill toward cap) */
    int8_t *B = calloc((size_t)K*N,1); for (int i=0;i<K*N;i++) B[i]=1;
    int8_t *A = calloc((size_t)M*K,1); for (int i=0;i<M*K;i++) A[i]=1;
    int32_t *C = calloc((size_t)M*N,4);

    /* warm domain 0 (native) — the weight we switch AWAY to each iteration */
    ork_npu_set_pack_domain(c, 0);
    ork_w *w0 = ork_i8_mm_pack(c, K, N, B);
    if (!w0 || ork_i8_mm_run(c, w0, M, A, C)) { printf("warm dom0 failed\n"); return 1; }

    /* pre-tiled blob to import (from a native pack) */
    ork_npu_set_pack_domain(c, 1);
    ork_w *wt = ork_i8_mm_pack(c, K, N, B);
    size_t need = ork_w_dump(wt, NULL, 0);
    void *blob = malloc(need); ork_w_dump(wt, blob, need);
    ork_mm_free(c, wt);
    int per = 2;   /* Bb + Bf imports per weight (K<=4096 builds Bf) */

    printf("accumulating imported weights in domain 1 (each ~%zuKB, %d foreign maps)...\n", need>>10, per);
    for (int i = 0; i < maxW; i++) {
        ork_npu_set_pack_domain(c, 1);
        ork_w *wi = ork_i8_mm_load_import(c, K, N, blob, need);
        if (!wi) { printf("*** import #%d: LOAD/MEM_CREATE FAILED (foreign maps ~%d, ~%zuMB)\n",
                          i, i*per, ((size_t)i*need)>>20); break; }
        memset(C, 0, (size_t)M*N*4);
        ork_i8_mm_run(c, w0, M, A, C);            /* switch -> domain 0 */
        int rc = ork_i8_mm_run(c, wi, M, A, C);   /* switch -> domain 1, submit against import #i */
        if (rc != 0 || C[0] != K) {
            printf("*** SUBMIT FAULT at import weight #%d: rc=%d C[0]=%d | foreign maps ~%d | total ~%zuMB\n",
                   i, rc, C[0], (i+1)*per, ((size_t)(i+1)*need)>>20);
            break;
        }
        if (i % 32 == 0) printf("  #%-4d OK | foreign maps ~%-5d | total ~%zuMB\n",
                                i, (i+1)*per, ((size_t)(i+1)*need)>>20);
        /* intentionally do NOT free wi — accumulate the foreign mappings */
    }
    printf("done\n");
    return 0;
}
