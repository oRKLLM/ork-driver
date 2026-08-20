/* chain_import_probe.c — the 7B's ACTUAL submit path: ork_i8_mm_run_chain (PC-chained, "i8 chain tasks=1"),
 * not the plain ork_i8_mm_run my earlier probes used. Test an IMPORTED weight run via the chain path after a
 * dom_activate switch, at a multi-core N. If this faults where run_i8 didn't, the regression is the chained/
 * multi-core regcmd path x imported buffers x domain switch. N=8192 => multi-core; also try K=18944 (K-slice).
 *   ./cip  (K=1024,N=8192)   |   ./cip 18944 3584  (K-sliced down-proj shape) */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int run_chain1(ork_npu *c, ork_w *w, int M, const int8_t *A, int32_t *C) {
    ork_mm_task_i8 t; t.w = w; t.M = M; t.A = A; t.C = C;
    return ork_i8_mm_run_chain(c, 1, &t);
}
static void rc(ork_npu *c, const char *tag, ork_w *w, int M, int K, int N, const int8_t *A, int32_t *C) {
    memset(C, 0, (size_t)M*N*4);
    if (!w) { printf("%-26s WEIGHT NULL\n", tag); return; }
    int r = run_chain1(c, w, M, A, C);
    printf("%-26s rc=%d C[0]=%d (expect %d) -> %s\n", tag, r, C[0], K,
           (r==0 && C[0]==K) ? "OK" : "*** FAULT ***");
}

int main(int argc, char **argv) {
    ork_npu *c = ork_npu_init();
    if (!c) { printf("no board\n"); return 0; }
    const int K = argc > 1 ? atoi(argv[1]) : 1024;
    const int N = argc > 2 ? atoi(argv[2]) : 8192;
    const int M = 128;   /* multi-core, 7B-like */
    int8_t *B = calloc((size_t)K*N,1); for (size_t i=0;i<(size_t)K*N;i++) B[i]=1;
    int8_t *A = calloc((size_t)M*K,1); for (size_t i=0;i<(size_t)M*K;i++) A[i]=1;
    int32_t *C = calloc((size_t)M*N,4);
    printf("K=%d N=%d M=%d (chain path)\n", K, N, M);

    ork_npu_set_pack_domain(c, 0);
    ork_w *w0 = ork_i8_mm_pack(c, K, N, B);
    rc(c, "NATIVE dom0 warm", w0, M, K, N, A, C);

    ork_npu_set_pack_domain(c, 1);
    ork_w *wn1 = ork_i8_mm_pack(c, K, N, B);
    rc(c, "NATIVE dom1 (switch)", wn1, M, K, N, A, C);

    size_t need = ork_w_dump(wn1, NULL, 0);
    void *blob = malloc(need); ork_w_dump(wn1, blob, need);
    ork_npu_set_pack_domain(c, 1);
    ork_w *wi1 = ork_i8_mm_load_import(c, K, N, blob, need);
    rc(c, "IMPORT dom1 in-domain", wi1, M, K, N, A, C);

    rc(c, "NATIVE dom0 (switch away)", w0, M, K, N, A, C);
    rc(c, "IMPORT dom1 AFTER switch", wi1, M, K, N, A, C);   /* <-- the 7B case, chain path */
    return 0;
}
