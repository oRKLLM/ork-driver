/* chain_bench — time ork_i8_mm_run_chain for a batch of S independent int8 matmuls, so the cross-core
 * fan-out (ORK_CHAIN_MC) can be measured against single-core. This is the workload shape the fan-out
 * targets: many independent matmuls sharing one submit (EAGLE-3 verification, dense batches). Uses
 * malloc'd A/C (no per-task DMA) so multi-core actually engages (DMA tasks fall back to single-core).
 *
 *   make chain_bench && sudo ./chain_bench [S] [K] [N] [M] [iters]
 *   sudo ./chain_bench               # single-core
 *   sudo ORK_CHAIN_MC=1 ./chain_bench   # 3-core fan-out — compare us/chain
 *
 * NOTE: dummy random data — times throughput only, does NOT check correctness (test_matmul does that).
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e6 + t.tv_nsec*1e-3; }
static unsigned int rs = 0x9e3779b9u;
static int rnd8(void) { rs = rs*1103515245u + 12345u; return (int)((rs >> 16) & 0xff) - 128; }

int main(int argc, char **argv) {
    int S     = argc > 1 ? atoi(argv[1]) : 24;
    int K     = argc > 2 ? atoi(argv[2]) : 2048;
    int N     = argc > 3 ? atoi(argv[3]) : 2048;
    int M     = argc > 4 ? atoi(argv[4]) : 16;
    int iters = argc > 5 ? atoi(argv[5]) : 50;

    ork_npu *c = ork_npu_init();
    if (!c) { fprintf(stderr, "ork_npu_init failed\n"); return 1; }

    ork_mm_task_i8 *tasks = calloc((size_t)S, sizeof *tasks);
    for (int i = 0; i < S; i++) {
        int8_t *B = malloc((size_t)K * N);
        for (size_t j = 0; j < (size_t)K * N; j++) B[j] = (int8_t)rnd8();
        ork_w *w = ork_i8_mm_pack(c, K, N, B);
        free(B);
        if (!w) { fprintf(stderr, "pack_i8 failed at %d (K=%d N=%d)\n", i, K, N); return 1; }
        /* A/C in DMA buffers (ork_dma_alloc) so run_chain_i8 AND run_i8 use them IN-PLACE (dma_find hits)
         * — this removes run_chain's per-call bcreate/bdestroy of staging buffers, the confound that made
         * the first measurement time buffer-churn instead of the submit mechanism. Fallback to malloc if
         * the dma-heap is full. Set ORK_NO_DMA=1 to force the old malloc path (re-introduces the confound). */
        int use_dma = !getenv("ORK_NO_DMA");
        int8_t  *A  = use_dma ? ork_dma_alloc(c, (size_t)M * K)     : NULL; if (!A)  A  = malloc((size_t)M * K);
        for (size_t j = 0; j < (size_t)M * K; j++) A[j] = (int8_t)rnd8();
        int32_t *Cc = use_dma ? ork_dma_alloc(c, (size_t)M * N * 4) : NULL; if (!Cc) Cc = malloc((size_t)M * N * 4);
        tasks[i].w = w; tasks[i].M = M; tasks[i].A = A; tasks[i].C = Cc;
    }

    int stream = getenv("ORK_STREAM") != NULL;
    int rc = stream ? ork_i8_mm_run_stream(c, S, tasks) : ork_i8_mm_run_chain(c, S, tasks);  /* warmup */
    if (rc) { fprintf(stderr, "run %s warmup failed rc=%d\n", stream ? "stream" : "chain", rc); return 1; }

    /* correctness: verify task 0 row 0 against the int32 CPU reference (the stream path is new) */
    { int bad = 0; const ork_mm_task_i8 *t = &tasks[0]; const int8_t *A = t->A;
      int8_t *Bref = malloc((size_t)K * N);   /* regenerate task 0's B with the same seed prefix */
      rs = 0x9e3779b9u; for (size_t j = 0; j < (size_t)K*N; j++) Bref[j] = (int8_t)rnd8();
      for (int n = 0; n < N && bad < 3; n++) {
          int32_t ref = 0; for (int kk = 0; kk < K; kk++) ref += (int)A[kk] * (int)Bref[(size_t)kk*N+n];
          if (t->C[n] != ref) { printf("  CORRECTNESS MISMATCH col %d: got %d exp %d\n", n, t->C[n], ref); bad++; }
      }
      free(Bref); if (!bad) printf("  correctness OK (task0 row0 vs CPU)\n"); }

    double t0 = now_us();
    for (int it = 0; it < iters; it++) {
        rc = stream ? ork_i8_mm_run_stream(c, S, tasks) : ork_i8_mm_run_chain(c, S, tasks);
        if (rc) { fprintf(stderr, "run failed rc=%d (iter %d)\n", rc, it); return 1; }
    }
    double dt = now_us() - t0;

    /* re-check correctness AFTER the warm loop (rules out a cold-buffer warmup issue) */
    { int bad = 0; const ork_mm_task_i8 *t = &tasks[0]; const int8_t *A = t->A;
      int8_t *Bref = malloc((size_t)K * N); rs = 0x9e3779b9u;
      for (size_t j = 0; j < (size_t)K*N; j++) Bref[j] = (int8_t)rnd8();
      for (int n = 0; n < N && bad < 3; n++) { int32_t ref = 0; for (int kk=0;kk<K;kk++) ref += (int)A[kk]*(int)Bref[(size_t)kk*N+n];
          if (t->C[n] != ref) { printf("  WARM-CHECK MISMATCH col %d: got %d exp %d\n", n, t->C[n], ref); bad++; } }
      free(Bref); if (!bad) printf("  warm-check OK (task0 row0)\n"); }

    double chain_us_mm = dt / iters / S;
    printf("chain S=%d K=%d N=%d M=%d iters=%d  [%s]  %.1f us/chain  (%.1f us/matmul)\n",
           S, K, N, M, iters, stream ? "STREAM-mc" : (getenv("ORK_CHAIN_MC") ? "MULTI-core" : "single-core"),
           dt / iters, chain_us_mm);

    /* BASELINE: the SAME S matmuls as S SEPARATE submits (the current decode / per-expert-MoE path).
     * The delta vs the chain above = the multi-task-submit amortization of the per-submit floor — the
     * whole point of packing many GEMVs into one submit (proven possible by the vendor's task_number=96).
     * Skip with ORK_NO_SEP=1. */
    if (!getenv("ORK_NO_SEP")) {
        for (int i = 0; i < S; i++)   /* warm each weight on the single-submit path */
            ork_i8_mm_run(c, tasks[i].w, tasks[i].M, tasks[i].A, tasks[i].C);
        double s0 = now_us();
        for (int it = 0; it < iters; it++)
            for (int i = 0; i < S; i++) {
                rc = ork_i8_mm_run(c, tasks[i].w, tasks[i].M, tasks[i].A, tasks[i].C);
                if (rc) { fprintf(stderr, "separate run failed rc=%d (i=%d)\n", rc, i); return 1; }
            }
        double sdt = now_us() - s0;
        double sep_us_mm = sdt / iters / S;
        printf("separate  S=%d (S submits/iter)  %.1f us/matmul\n", S, sep_us_mm);
        printf("MULTI-TASK SUBMIT AMORTIZATION: %.2fx  (separate %.1f -> chain %.1f us/matmul; floor saved %.1f us)\n",
               sep_us_mm / chain_us_mm, sep_us_mm, chain_us_mm, sep_us_mm - chain_us_mm);
    }

    ork_npu_free(c);
    return 0;
}
