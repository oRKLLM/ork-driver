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
#include <time.h>
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

    /* reference: blocking full matmul (ork_mm_run_i8 -> mcworker at M>1 wide-K/wide-N) */
    ork_w *wf = ork_mm_pack_i8(c, K, N, B); if (!wf) { printf("ref pack fail\n"); return 1; }
    int32_t *Cref = (int32_t *) malloc((size_t) M*N*4);
    if (ork_mm_run_i8(c, wf, M, A, Cref)) { printf("ref run FAIL\n"); return 1; }

    /* slice-and-dice library primitive: pack once (c_base tiles), run on the doorbell (general dtype API) */
    ork_w_sliced *ws = ork_mm_pack_sliced(c, K, N, B, ORK_DT_I8);
    if (!ws) { printf("pack_sliced FAIL (alignment? K%%512=%d N%%16=%d)\n", K%512, N%16); return 1; }
    int32_t *Cslc = (int32_t *) malloc((size_t) M*N*4);
    if (ork_mm_run_sliced(c, ws, M, A, Cslc, nc)) { printf("run_sliced FAIL\n"); return 1; }

    long bad = 0, first = -1;
    for (size_t i = 0; i < (size_t) M*N; i++) if (Cslc[i] != Cref[i]) { if (first < 0) first = (long) i; bad++; }
    printf("mismatches (sliced vs run_i8) = %ld / %d", bad, M*N);
    if (bad) printf("  first@%ld: sliced %d vs ref %d", first, Cslc[first], Cref[first]);
    printf(" -> %s\n", bad ? "DIFFER" : "identical");

    /* CPU int32 reference (CPUREF=1) — the ONLY trustworthy truth when the driver path itself is suspect
     * (task #36: N=3584 colsplit can break ork_mm_run_i8, so "sliced vs run_i8" above would be meaningless).
     * O(M*N*K): use a SMALL M. Reports run_i8 AND sliced EACH vs CPU, so we see which path is correct. */
    if (getenv("CPUREF")) {
        int32_t *Ccpu = (int32_t *) malloc((size_t) M*N*4);
        for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) { int32_t s = 0; const int8_t *ar = A + (size_t) m*K;
            for (int k = 0; k < K; k++) s += (int32_t) ar[k] * (int32_t) B[(size_t) k*N + n];
            Ccpu[(size_t) m*N + n] = s; }
        long br = 0, bs = 0, fr = -1, fs = -1;
        for (size_t i = 0; i < (size_t) M*N; i++) { if (Cref[i] != Ccpu[i]) { if (fr<0) fr=(long)i; br++; }
                                                    if (Cslc[i] != Ccpu[i]) { if (fs<0) fs=(long)i; bs++; } }
        printf("vs CPU: run_i8 mism=%ld/%d%s | sliced mism=%ld/%d%s\n",
               br, M*N, br?" <-- run_i8 BROKEN for this shape":" OK", bs, M*N, bs?" <-- sliced FAIL":" OK");
        if (br) printf("   run_i8 first@%ld: %d vs cpu %d\n", fr, Cref[fr], Ccpu[fr]);
        if (bs) printf("   sliced first@%ld: %d vs cpu %d\n", fs, Cslc[fs], Ccpu[fs]);
        free(Ccpu);
    }

    /* A/B throughput: blocking mcworker (run_i8) vs sliced doorbell (run_sliced), same shape. REPS env (default 40). */
    int reps = getenv("REPS") ? atoi(getenv("REPS")) : 40; if (reps < 1) reps = 1;
    for (int w = 0; w < 3; w++) { ork_mm_run_i8(c, wf, M, A, Cref); ork_mm_run_sliced(c, ws, M, A, Cslc, nc); }   /* warmup */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < reps; r++) ork_mm_run_i8(c, wf, M, A, Cref);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double us_block = ((t1.tv_sec-t0.tv_sec)*1e6 + (t1.tv_nsec-t0.tv_nsec)*1e-3) / reps;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < reps; r++) ork_mm_run_sliced(c, ws, M, A, Cslc, nc);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double us_slice = ((t1.tv_sec-t0.tv_sec)*1e6 + (t1.tv_nsec-t0.tv_nsec)*1e-3) / reps;
    printf("A/B (%d reps): blocking run_i8 = %.1f us/call | sliced doorbell = %.1f us/call | sliced/blocking = %.2fx\n",
           reps, us_block, us_slice, us_slice / us_block);

    ork_mm_free(c, wf); ork_mm_free_sliced(c, ws);
    return bad ? 1 : 0;
}
