/* f16_mm_rows_probe -- r104: tools/f16_mm_single_probe.c with the fp64 reference on SAMPLED ROWS.
 * The NPU runs the FULL M x K x N fp16 matmul (ork_f16_mm_pack + ork_f16_mm_run, unchanged); only the
 * check is sampled: rows 0..63, the last 64, and 64 strided across the middle (so the first, last and
 * interior M-chunks are all read). Same operands, same pass bar (1e-3 relative) and the same positive
 * control (reference from a wrong B column, must fail) as the original. The full-row fp64 reference
 * of the 248,320-wide head shapes would take ~1.5 h single-threaded; this takes seconds.
 *   f16_mm_rows_probe M K N [reps]         run only via tools/util/npu_guard.sh with timeout      */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ork_npu.h>

static unsigned long long rs = 0x9E3779B97F4A7C15ull;
static double urand(void){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }

int main(int argc, char **argv){
    if(argc < 4){ fprintf(stderr, "usage: %s M K N [reps]\n", argv[0]); return 2; }
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]), reps = argc > 4 ? atoi(argv[4]) : 3;
    int rows[192], nr = 0;
    for(int i = 0; i < 64 && i < M; i++) rows[nr++] = i;
    for(int i = 1; i <= 64 && M > 128; i++){ int r = (int)((long long)i * (M - 128) / 65) + 64; rows[nr++] = r; }
    for(int i = (M > 64 ? M - 64 : 64); i < M; i++) if(i >= 64) rows[nr++] = i;
    printf("f16_mm_rows_probe M=%d K=%d N=%d reps=%d  (K%%32=%d N%%16=%d)  check rows %d\n", M, K, N, reps, K % 32, N % 16, nr);
    fflush(stdout);
    ork_f16 *A = malloc(sizeof(ork_f16) * (size_t)M * K), *B = malloc(sizeof(ork_f16) * (size_t)K * N);
    float *C = calloc((size_t)M * N, sizeof(float));
    double *R = calloc((size_t)nr * N, sizeof(double)), *Rbad = calloc((size_t)nr * N, sizeof(double));
    if(!A || !B || !C || !R || !Rbad){ fprintf(stderr, "alloc failed\n"); return 2; }
    for(size_t i = 0; i < (size_t)M * K; i++) A[i] = (ork_f16)((urand() * 2 - 1) * 0.73);
    for(size_t i = 0; i < (size_t)K * N; i++) B[i] = (ork_f16)((urand() * 2 - 1) * 0.37);
    struct timespec r0, r1; clock_gettime(CLOCK_MONOTONIC, &r0);
    #pragma omp parallel for schedule(dynamic)
    for(int j = 0; j < nr; j++){
        double *s = R + (size_t)j * N;
        const ork_f16 *a = A + (size_t)rows[j] * K;
        for(int k = 0; k < K; k++){ double av = (double)a[k]; const ork_f16 *b = B + (size_t)k * N;
            for(int n = 0; n < N; n++) s[n] += av * (double)b[n]; }
        for(int n = 0; n < N; n++) Rbad[(size_t)j * N + n] = s[(n + 1) % N];   /* wrong column, as the original */
    }
    clock_gettime(CLOCK_MONOTONIC, &r1);
    printf("reference %.1f s\n", (r1.tv_sec - r0.tv_sec) + (r1.tv_nsec - r0.tv_nsec) / 1e9); fflush(stdout);
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
        double mx = 0, md = 0, mdb = 0;
        for(int j = 0; j < nr; j++) for(int n = 0; n < N; n++){
            double c = C[(size_t)rows[j] * N + n], ref = R[(size_t)j * N + n];
            if(fabs(c - ref) > md) md = fabs(c - ref);
            if(fabs(c - Rbad[(size_t)j * N + n]) > mdb) mdb = fabs(c - Rbad[(size_t)j * N + n]);
            if(fabs(ref) > mx) mx = fabs(ref); }
        double e = md / (mx > 0 ? mx : 1), ebad = mdb / (mx > 0 ? mx : 1);
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
