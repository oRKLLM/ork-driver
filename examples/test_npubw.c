/* examples/test_npubw.c — NPU effective DRAM read bandwidth.
 * Weight-DMA-bound matmul: small M (compute negligible), large K*N weights (per-submit overhead amortized).
 * effective GB/s = (K*N useful weight bytes) / time_per_matmul. Sweeps N to separate fixed overhead
 * (intercept) from the asymptotic streaming rate (slope), at M=1 (decode point) and a small M. Reports
 * 1-core and all-core so we can see whether NPU DMA scales with cores or hits the shared-DRAM wall like the CPU.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

static double run(ork_npu*c,ork_w*w,int M,int8_t*A,int32_t*C,int iters){
    ork_mm_run_i8(c,w,M,A,C);                              /* warm */
    double t0=now_us(); for(int i=0;i<iters;i++) ork_mm_run_i8(c,w,M,A,C);
    return (now_us()-t0)/iters;                            /* us/matmul */
}

static void sweep(ork_npu*c,int cores,int M){
    int K=4096; int Ns[]={2048,8192,16384,32768}; int nN=sizeof(Ns)/sizeof(Ns[0]);
    ork_npu_set_core_budget(c,cores);
    printf("  M=%d  %d-core:\n", M, cores);
    for(int i=0;i<nN;i++){ int N=Ns[i];
        int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
        int8_t*A=malloc((size_t)M*K); memset(A,1,(size_t)M*K);
        int32_t*C=malloc((size_t)M*N*4);
        ork_w*w=ork_mm_pack_i8(c,K,N,B); if(!w){printf("    pack failed N=%d\n",N);free(A);free(B);free(C);continue;}
        int iters = (N>=16384)?5:10;
        double us=run(c,w,M,A,C,iters);
        for(int r=1;r<3;r++){ double x=run(c,w,M,A,C,iters); if(x<us) us=x; }
        double bytes=(double)K*N;                          /* useful int8 weight bytes */
        printf("    K=%d N=%-6d  %6.1f MB  %8.1f us  %6.1f GB/s\n", K, N, bytes/1e6, us, bytes/1e3/us);
        ork_w_free(w); free(A); free(B); free(C);
    }
}

int main(int argc,char**argv){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int cores=ork_npu_cores(c);
    printf("NPU effective read bandwidth (useful weight bytes / matmul time); %d cores available\n", cores);
    sweep(c,1,1);
    sweep(c,cores,1);
    sweep(c,cores,8);
    ork_npu_free(c);
    return 0;
}
