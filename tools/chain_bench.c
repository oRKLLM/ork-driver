/* chain_bench — time ork_mm_run_chain_i8 for a batch of S independent int8 matmuls, so the cross-core
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
        ork_w *w = ork_mm_pack_i8(c, K, N, B);
        free(B);
        if (!w) { fprintf(stderr, "pack_i8 failed at %d (K=%d N=%d)\n", i, K, N); return 1; }
        int8_t  *A  = malloc((size_t)M * K);
        for (size_t j = 0; j < (size_t)M * K; j++) A[j] = (int8_t)rnd8();
        int32_t *Cc = malloc((size_t)M * N * 4);
        tasks[i].w = w; tasks[i].M = M; tasks[i].A = A; tasks[i].C = Cc;
    }

    int rc = ork_mm_run_chain_i8(c, S, tasks);          /* warmup (also primes per-core warm state) */
    if (rc) { fprintf(stderr, "run_chain_i8 warmup failed rc=%d\n", rc); return 1; }

    double t0 = now_us();
    for (int it = 0; it < iters; it++) {
        rc = ork_mm_run_chain_i8(c, S, tasks);
        if (rc) { fprintf(stderr, "run_chain_i8 failed rc=%d (iter %d)\n", rc, it); return 1; }
    }
    double dt = now_us() - t0;

    printf("chain S=%d K=%d N=%d M=%d iters=%d  [%s]  %.1f us/chain  (%.1f us/matmul)\n",
           S, K, N, M, iters, getenv("ORK_CHAIN_MC") ? "MULTI-core" : "single-core",
           dt / iters, dt / iters / S);

    ork_npu_free(c);
    return 0;
}
