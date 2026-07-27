/* slice_dbrun_probe — execution proof + test for the slice-and-dice library primitive
 * (ork_mm_pack_i8_sliced / ork_mm_run_i8_sliced). Runs an arbitrary int8 matmul ENTIRELY on the doorbell by
 * decomposing it into c_base tiles (K-slice int32-accumulate + N-tile scatter + M), and compares BIT-EXACT
 * vs the blocking full reference (ork_mm_run_i8). Wide-K (ffn_down) wedges as one submit; sliced it runs
 * wedge-free. `make slice_dbrun_probe && sudo ./slice_dbrun_probe [K=6144] [N=2048] [M=256] [nc=0(all)]`
 * (nc=1 sidesteps the non-even-N colsplit driver bug #36 when validating wide-N correctness).
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static uint32_t g = 2463534242u;
static int8_t r8(void) { g ^= g << 13; g ^= g >> 17; g ^= g << 5; return (int8_t)(((int)(g & 0x3f)) - 31); }

int main(int argc, char **argv) {
    int K = argc > 1 ? atoi(argv[1]) : 6144, N = argc > 2 ? atoi(argv[2]) : 2048, M = argc > 3 ? atoi(argv[3]) : 256;
    int nc = argc > 4 ? atoi(argv[4]) : 0;
    setvbuf(stdout, 0, _IONBF, 0);
    ork_npu *c = ork_npu_init(); if (!c) { printf("init fail\n"); return 1; }
    printf("slice_dbrun: K=%d N=%d M=%d nc=%d\n", K, N, M, nc);

    int8_t *A = (int8_t *) malloc((size_t) M*K), *B = (int8_t *) malloc((size_t) K*N);
    for (size_t i = 0; i < (size_t) M*K; i++) A[i] = r8();
    for (size_t i = 0; i < (size_t) K*N; i++) B[i] = r8();

    /* reference: blocking full matmul */
    ork_w *wf = ork_mm_pack_i8(c, K, N, B); if (!wf) { printf("ref pack fail\n"); return 1; }
    int32_t *Cref = (int32_t *) malloc((size_t) M*N*4);
    if (ork_mm_run_i8(c, wf, M, A, Cref)) { printf("ref run FAIL\n"); return 1; }
    ork_mm_free(c, wf);

    /* slice-and-dice library primitive: pack once (c_base tiles), run on the doorbell (general dtype API) */
    ork_w_sliced *ws = ork_mm_pack_sliced(c, K, N, B, ORK_DT_I8);
    if (!ws) { printf("pack_sliced FAIL (alignment? K%%512=%d N%%16=%d)\n", K%512, N%16); return 1; }
    int32_t *Cslc = (int32_t *) malloc((size_t) M*N*4);
    if (ork_mm_run_sliced(c, ws, M, A, Cslc, nc)) { printf("run_sliced FAIL\n"); return 1; }
    ork_mm_free_sliced(c, ws);

    long bad = 0, first = -1;
    for (size_t i = 0; i < (size_t) M*N; i++) if (Cslc[i] != Cref[i]) { if (first < 0) first = (long) i; bad++; }
    printf("mismatches = %ld / %d", bad, M*N);
    if (bad) printf("  first@%ld: sliced %d vs ref %d", first, Cslc[first], Cref[first]);
    printf(" -> %s\n", bad ? "FAIL" : "BIT-EXACT PASS");
    return bad ? 1 : 0;
}
