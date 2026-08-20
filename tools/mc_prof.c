/* tools/mc_prof.c — diagnose why large-M (prefill) multi-core barely scales.
 * Runs the SAME int8 matmul at 1 core vs N cores, and (multi-core) prints the per-core
 * phase split (copy / submit / acc) from ork_npu_mc_timing. If 'submit' (NPU ioctl+wait)
 * doesn't shrink with more cores, the cores aren't overlapping on the NPU; if 'copy'/'acc'
 * dominate, the CPU-side per-submit work is the bottleneck.
 *   make mc_prof && sudo ./mc_prof [M] [K] [N] [iters]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

static double run(ork_npu*c,ork_w*w,int M,int K,int8_t*A,int32_t*C,int iters){
    ork_i8_mm_run(c,w,M,A,C);                          /* warm */
    double t0=now_us(); for(int i=0;i<iters;i++) ork_i8_mm_run(c,w,M,A,C);
    return (now_us()-t0)/iters;
}

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):512, K=argc>2?atoi(argv[2]):2048, N=argc>3?atoi(argv[3]):2048;
    int iters=argc>4?atoi(argv[4]):20;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int cores=ork_npu_cores(c);
    int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    int dma=getenv("ORK_TEST_DMA")!=NULL;   /* allocate A in a zero-copy DMA buffer to exercise the no-gather path */
    int8_t*A=dma?ork_dma_alloc(c,(size_t)M*K):malloc((size_t)M*K); if(A)memset(A,1,(size_t)M*K);
    int32_t*C=malloc((size_t)M*N*4);
    if(dma) printf("[A in zero-copy DMA buffer]\n");
    ork_w*w=ork_i8_mm_pack(c,K,N,B); if(!w){printf("pack failed\n");return 1;}
    printf("int8 matmul %dx%dx%d, %d warm iters. us/matmul:\n",M,K,N,iters);

    ork_npu_set_core_budget(c,1);
    ork_npu_mc_reset();
    double t1=run(c,w,M,K,A,C,iters);
    printf("  1-core: %.1f us/matmul\n", t1);
    { double cp,su,ac; long n; ork_npu_mc_timing(0,&cp,&su,&ac,&n);
      if(n) printf("    1-core phases over %d iters: submits=%ld  copy=%.0f (%.1f/sub)  submit=%.0f (%.1f/sub)  acc=%.0f (%.1f/sub)\n",
                   iters,n,cp,cp/n,su,su/n,ac,ac/n); }

    ork_npu_set_core_budget(c,cores);
    ork_npu_mc_reset();
    double tN=run(c,w,M,K,A,C,iters);
    printf("  %d-core: %.1f us/matmul   (scaling %.2fx)\n", cores, tN, t1/tN);
    printf("  per-core phase totals over %d warm iters (us; copy=act host-copy+bsync, submit=regcmd+ioctl+result bsync, acc=host accumulate):\n", iters);
    for(int i=0;i<cores;i++){
        double cp,su,ac; long n; ork_npu_mc_timing(i,&cp,&su,&ac,&n);
        double sy=ork_npu_mc_synth(i), io=su-sy;   /* synth = overlappable host inside submit; io = ioctl/NPU (unhideable) */
        printf("    core %d: submits=%ld  copy=%.0f  submit=%.0f (synth=%.0f ioctl=%.0f)  acc=%.0f  | overlappable host (copy+synth+acc)=%.0f vs NPU ioctl=%.0f -> pipeline ceiling %.3fx\n",
               i, n, cp, su, sy, io, ac, cp+sy+ac, io, io>0?(cp+su+ac)/io:0);
    }
    ork_w_free(w); if(!dma)free(A); ork_npu_free(c); free(B);free(C); return 0;   /* dma A freed by ork_npu_free */
}
