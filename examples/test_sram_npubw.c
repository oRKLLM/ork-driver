/* examples/test_sram_npubw.c — NPU read bandwidth: SRAM-resident vs DRAM-resident weight, 1-core vs all-core.
 *
 * Tests whether on-chip NPU SRAM is a SEPARATE memory port that adds AGGREGATE bandwidth beyond the shared
 * ~28 GB/s DRAM wall. Method (same as test_npubw): a weight-DMA-bound matmul (tiny M, so compute is negligible
 * and time ~= weight-read time); effective GB/s = K*N useful weight bytes / matmul time. The weight is sized to
 * FIT the ~700 KiB usable SRAM, packed once in DRAM and once in SRAM (ORK_WEIGHT_SRAM, toggled via setenv at
 * pack time), and swept 1-core vs all-core.
 *
 * Read:
 *   - DRAM 1-core vs all-core: does NPU DRAM read scale with cores or hit the shared-DRAM wall (expect wall).
 *   - SRAM 1-core: is a single core's SRAM read even FASTER than DRAM (gating question — if not, SRAM adds nothing).
 *   - SRAM all-core >> DRAM all-core  ==>  SRAM is a separate/higher-BW port => the "2 cores DRAM || 1 core SRAM"
 *     aggregate win is real. SRAM all-core ~= DRAM all-core ==> no aggregate win, idea is dead.
 *
 * PASS = runs + prints (probe, not a correctness gate). Run: make test_sram_npubw && sudo ./test_sram_npubw
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

static double run(ork_npu*c,ork_w*w,int M,int8_t*A,int32_t*C,int iters){
    ork_mm_run_i8(c,w,M,A,C);                               /* warm */
    double best=1e18;
    for(int r=0;r<3;r++){ double t0=now_us(); for(int i=0;i<iters;i++) ork_mm_run_i8(c,w,M,A,C);
        double us=(now_us()-t0)/iters; if(us<best) best=us; }
    return best;                                            /* us/matmul */
}

/* pack in DRAM (sram=0) or SRAM (sram=1, via ORK_WEIGHT_SRAM), then time single- and all-core reads. */
static void measure(ork_npu*c,int sram,int K,int N,int M,int cores){
    int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    int8_t*A=malloc((size_t)M*K); memset(A,1,(size_t)M*K);
    int32_t*C=malloc((size_t)M*N*4);
    if(sram) setenv("ORK_WEIGHT_SRAM","1",1); else unsetenv("ORK_WEIGHT_SRAM");
    size_t f0=ork_npu_sram_free(c);
    ork_w*w=ork_mm_pack_i8(c,K,N,B);
    size_t f1=ork_npu_sram_free(c);
    if(!w){ printf("  [%s] pack failed\n", sram?"SRAM":"DRAM"); free(A);free(B);free(C); return; }
    size_t wbytes=(size_t)K*N;
    int placed = (f0-f1) >= wbytes;                         /* SRAM actually used vs failed-over to DRAM */
    int iters=200;
    for(int nc=1;nc<=cores;nc+= (cores-1>0?cores-1:1)){      /* 1-core, then all-core */
        ork_npu_set_core_budget(c,nc);
        double us=run(c,w,M,A,C,iters);
        printf("  %-4s M=%d %d-core: K=%d N=%-5d %5.0f KiB  %8.1f us  %7.1f GB/s%s\n",
               sram?"SRAM":"DRAM", M, nc, K, N, wbytes/1024.0, us, (double)wbytes/1e3/us,
               (sram && !placed) ? "  (SRAM alloc FAILED-OVER to DRAM!)" : "");
        if(cores==1) break;
    }
    ork_w_free(w); free(A); free(B); free(C);
}

int main(void){
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    int cores=ork_npu_cores(c);
    size_t total=ork_npu_sram_total(c);
    printf("NPU read BW: SRAM vs DRAM resident weight; %d cores; SRAM total=%zu KiB\n", cores, total>>10);
    if(total==0){ printf("  no NPU SRAM on this kernel/DTB — SRAM path will fail over; test is moot\n"); }
    /* weight sized to fit ~700 KiB usable SRAM: K=4096 x N=128 = 512 KiB. Tiny M so time ~= weight read. */
    int K=4096, N=128;
    for(int mi=0; mi<2; mi++){ int M = mi?8:1;
        printf("-- M=%d --\n", M);
        measure(c,0,K,N,M,cores);   /* DRAM: 1-core + all-core */
        measure(c,1,K,N,M,cores);   /* SRAM: 1-core + all-core */
    }
    ork_npu_free(c);
    return 0;
}
