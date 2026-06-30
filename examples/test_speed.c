/* examples/test_speed.c — validates that multi-core prefill speed has not regressed.
 * Checks scaling of large-M matmuls across cores and absolute latency thresholds.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

static double run(ork_npu*c,ork_w*w,int M,int K,int8_t*A,int32_t*C,int iters){
    ork_mm_run_i8(c,w,M,A,C);                          /* warm */
    double t0=now_us(); for(int i=0;i<iters;i++) ork_mm_run_i8(c,w,M,A,C);
    return (now_us()-t0)/iters;
}

int main(void){
    int M=512, K=2048, N=2048, iters=10;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int cores=ork_npu_cores(c);
    
    int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    int8_t*A=malloc((size_t)M*K); memset(A,1,(size_t)M*K);
    int32_t*C=malloc((size_t)M*N*4);
    
    ork_w*w=ork_mm_pack_i8(c,K,N,B); if(!w){printf("pack failed\n");return 1;}
    
    ork_npu_set_core_budget(c,1);
    double t1=run(c,w,M,K,A,C,iters);
    printf("1-core: %.1f us/matmul\n", t1);
    
    ork_npu_set_core_budget(c,cores);
    double tN=run(c,w,M,K,A,C,iters);
    printf("%d-core: %.1f us/matmul (scaling %.2fx)\n", cores, tN, t1/tN);
    
    ork_w_free(w); free(A); free(B); free(C); ork_npu_free(c);
    
    int fail = 0;
    /* Thresholds ratcheted to current RK3588 perf (2026-06-30): M=512 K=N=2048 int8 measures
     * 1-core ~11.7ms, 3-core ~4.55ms (scaling ~2.55-2.62x) across runs, ~3% spread. Set with margin
     * below best-observed so a real regression (lost core, little-core pinning, kernel slowdown) trips
     * it but normal variance does not. Was 1.4x / 9000us — far too loose for the current kernel. */
    if (cores > 1 && (t1 / tN) < 2.2) {
        printf("FAIL: Multi-core scaling (%.2fx) is below 2.2x threshold. Possible thread pinning regression.\n", t1/tN);
        fail = 1;
    }
    /* Absolute latency guards (RK3588; RK3576 may differ — loosen there if validated). */
    if (tN > 6000.0) {
        printf("FAIL: Multi-core latency (%.1f us) exceeds 6000 us limit.\n", tN);
        fail = 1;
    }
    if (t1 > 14000.0) {
        printf("FAIL: Single-core latency (%.1f us) exceeds 14000 us limit (per-core kernel regression).\n", t1);
        fail = 1;
    }

    if (!fail) printf("SPEED OK\n");
    return fail;
}
