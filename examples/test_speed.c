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
    
    /* Best-of-3 (min) for each: the guards below are all about the BEST-ACHIEVABLE latency/scaling, so a
     * single jittery window (a big core momentarily stolen by a background thread — orkllm, a kworker, the
     * prior test's residue) must not trip them. A genuinely lost/parked core collapses ALL reps, so the
     * parked-core scaling guard is preserved. Fixed a real false-fail: in the full `make test` run the
     * 3-core scaling landed ~1.49x (vs ~1.53x standalone) from post-`model` residue — just under the 1.5x
     * floor — while standalone it clears reliably. */
    ork_npu_set_core_budget(c,1);
    double t1=run(c,w,M,K,A,C,iters);
    for(int r=1;r<3;r++){ double x=run(c,w,M,K,A,C,iters); if(x<t1) t1=x; }
    printf("1-core: %.1f us/matmul\n", t1);

    ork_npu_set_core_budget(c,cores);
    double tN=run(c,w,M,K,A,C,iters);
    for(int r=1;r<3;r++){ double x=run(c,w,M,K,A,C,iters); if(x<tN) tN=x; }
    printf("%d-core: %.1f us/matmul (scaling %.2fx)\n", cores, tN, t1/tN);
    
    ork_w_free(w); free(A); free(B); free(C); ork_npu_free(c);
    
    int fail = 0;
    /* Thresholds re-ratcheted 2026-06-30 after the WEIGHT-DMA AMORTIZATION fix (M-tile cap raised from
     * R-1 to the 0x1040 schedule max mg_max*64 — see AGENTS.md). M=512 K=N=2048 int8 now measures
     * 1-core ~5.1ms (was ~11.7ms; ~2.1x faster), 3-core ~2.8ms (was ~4.55ms; ~1.6x). NOTE: the SCALING
     * RATIO DROPPED (~2.55x -> ~1.82x) BECAUSE single-core sped up MORE than multi-core — that is the
     * intended win, not a regression. So the absolute-latency guards (below) are now the real regression
     * detectors; the scaling floor only catches a lost/parked core (scaling collapses toward 1.0). */
    if (cores > 1 && (t1 / tN) < 1.5) {
        printf("FAIL: Multi-core scaling (%.2fx) is below 1.5x floor — likely a lost/parked core.\n", t1/tN);
        fail = 1;
    }
    /* Absolute latency guards (RK3588; RK3576 may differ — loosen there if validated). Margin ~25%
     * over best-observed (1-core ~5.1ms, 3-core ~2.8ms) so normal ~3-7% spread does not trip them. */
    if (tN > 3600.0) {
        printf("FAIL: Multi-core latency (%.1f us) exceeds 3600 us limit.\n", tN);
        fail = 1;
    }
    if (t1 > 6500.0) {
        printf("FAIL: Single-core latency (%.1f us) exceeds 6500 us limit (per-core kernel regression).\n", t1);
        fail = 1;
    }

    if (!fail) printf("SPEED OK\n");
    return fail;
}
