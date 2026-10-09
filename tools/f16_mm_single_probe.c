/* f16_mm_single_probe — run ONE fp16 matmul shape through the production API
 * (ork_f16_mm_pack + ork_f16_mm_run), alone, and check it against fp64.
 *
 * Why: r102's first full-size training step wedged the NPU on the GDN in_proj_b forward
 * (fp16, M=1024, K=1024, N=16): rk_iommu read page fault at an unmapped IOVA, then core 0 held by a
 * job that never retired. N=16 was the only shape class r101 never sent to the NPU. This isolates a
 * single shape so a hang points at the shape and nothing else. (tools/f16_shape_probe.c is a
 * different tool: it sweeps the bare synth() path with register overrides.)
 *
 *   f16_mm_single_probe M K N [reps]      e.g.  f16_mm_single_probe 1024 1024 16
 *
 * Operands are fp16 values from a continuous distribution (NOT multiples of a power of two, which
 * accumulate exactly and make the check vacuous). Reference = fp64 matmul of the SAME fp16 operands.
 * Pass if max |C - ref| / max|ref| <= 1e-3. Positive control: the same check against a reference built
 * from a wrong B (each column swapped for the next) MUST fail, or the verdict is VOID.
 * Run only via tools/util/npu_guard.sh with timeout; SIGTERM only.                                 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ork_npu.h>

static unsigned long long rs = 0x9E3779B97F4A7C15ull;
static double urand(void){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }

static double relerr(const float *C, const double *R, size_t n){
    double mx = 0, md = 0;
    for(size_t i = 0; i < n; i++){ double d = fabs((double)C[i] - R[i]); if(d > md) md = d; if(fabs(R[i]) > mx) mx = fabs(R[i]); }
    return md / (mx > 0 ? mx : 1);
}

int main(int argc, char **argv){
    if(argc < 4){ fprintf(stderr, "usage: %s M K N [reps]\n", argv[0]); return 2; }
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]), reps = argc > 4 ? atoi(argv[4]) : 3;
    printf("f16_mm_single_probe M=%d K=%d N=%d reps=%d  (K%%32=%d N%%16=%d)\n", M, K, N, reps, K % 32, N % 16);
    fflush(stdout);
    ork_f16 *A = malloc(sizeof(ork_f16) * (size_t)M * K), *B = malloc(sizeof(ork_f16) * (size_t)K * N);
    float *C = calloc((size_t)M * N, sizeof(float));
    double *R = calloc((size_t)M * N, sizeof(double)), *Rbad = calloc((size_t)M * N, sizeof(double));
    if(!A || !B || !C || !R || !Rbad){ fprintf(stderr, "alloc failed\n"); return 2; }
    for(size_t i = 0; i < (size_t)M * K; i++) A[i] = (ork_f16)((urand() * 2 - 1) * 0.73);
    for(size_t i = 0; i < (size_t)K * N; i++) B[i] = (ork_f16)((urand() * 2 - 1) * 0.37);
    for(int m = 0; m < M; m++) for(int n = 0; n < N; n++){
        double s = 0, sb = 0;
        for(int k = 0; k < K; k++){
            s  += (double)A[(size_t)m * K + k] * (double)B[(size_t)k * N + n];
            sb += (double)A[(size_t)m * K + k] * (double)B[(size_t)k * N + (n + 1) % N];   /* wrong column */
        }
        R[(size_t)m * N + n] = s; Rbad[(size_t)m * N + n] = sb;
    }
    ork_npu *ctx = ork_npu_init();
    if(!ctx){ fprintf(stderr, "ork_npu_init failed\n"); return 3; }
    printf("packing...\n"); fflush(stdout);
    ork_w *w = ork_f16_mm_pack(ctx, K, N, B);
    if(!w){ printf("RESULT pack REFUSED (K%%32 or N%%16)\n"); ork_npu_free(ctx); return 4; }
    int fail = 0;
    for(int r = 0; r < reps; r++){
        printf("run %d submitting...\n", r); fflush(stdout);
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        int rc = ork_f16_mm_run(ctx, w, M, A, C);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double us = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
        if(rc){ printf("RESULT run %d rc=%d -- STOPPING (no retry)\n", r, rc); fail = 1; break; }
        double e = relerr(C, R, (size_t)M * N), ebad = relerr(C, Rbad, (size_t)M * N);
        int ok = e <= 1e-3, ctrl = ebad > 1e-3;
        printf("run %d  %.0f us  %.2f GFLOP/s  relerr %.3e  control %.3e  %s%s\n", r, us, 2.0 * M * K * N / us / 1e3,
               e, ebad, ok ? "PASS" : "FAIL", ctrl ? "" : "  [control did not fail: VOID]");
        fflush(stdout);
        if(!ok || !ctrl) fail = 1;
    }
    ork_mm_free(ctx, w);
    ork_npu_free(ctx);
    printf("RESULT %s\n", fail ? "FAIL" : "PASS");
    return fail;
}
