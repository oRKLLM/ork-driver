/* tools/batch_verify_probe.c — does ork's existing multi-core matmul already make M>1 batched
 * verify efficient? (Outstanding API lead #1: rknn_set_batch_core_num / spec-decode batched verify.)
 *
 * Speculative decode verifies M candidate tokens in ONE forward pass: the per-layer projections
 * become M-row GEMMs instead of M=1 GEMVs. The lever only pays if the per-VERIFIED-TOKEN NPU cost
 * DROPS as M grows (the ~hundreds-of-us submit/dispatch floor amortizes over M rows) AND multi-core
 * scaling kicks in. This probe measures exactly that on the real Qwen2.5-7B decode projections
 * (Q,K,V,O,gate,up,down), sweeping M in {1,2,4,8,16,32}, 1-core vs 3-core:
 *   - t_layer    = sum of the 7 projection matmuls for a layer (us)
 *   - us/token   = t_layer(3-core) / M   <- the amortized cost per verified token (the key curve)
 *   - scaling    = t1/t3                  <- does multi-core engage at this M?
 * If us/token at M=8/16 is well below M=1, batched verify is the decode lever and ork ALREADY has
 * the matmul capability (set_batch_core_num would add nothing) — the work is the drafter integration.
 * ORK_DECODE_MC is forced on so M=1 also tries multi-core (apples-to-apples curve).
 *   make batch_verify_probe && sudo ./batch_verify_probe [iters]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

int main(int argc,char**argv){
    setenv("ORK_DECODE_MC","1",1);   /* let M=1 also use multi-core, for a clean full-M-sweep curve */
    int iters=argc>1?atoi(argv[1]):20;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int cores=ork_npu_cores(c);
    printf("INIT OK: soc=%s cores=%d\n", ork_npu_soc(c), cores);

    int KN[7][2]={ {3584,3584},{3584,512},{3584,512},{3584,3584},{3584,18944},{3584,18944},{18944,3584} };
    const char*nm[7]={"Q","K","V","O","gate","up","down"};

    ork_w *w[7];
    for(int m=0;m<7;m++){
        int K=KN[m][0],N=KN[m][1];
        int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
        ork_npu_set_pack_domain(c,0);
        w[m]=ork_mm_pack_i8(c,K,N,B); free(B);
        if(!w[m]){printf("pack %s failed\n",nm[m]);return 1;}
    }
    int maxK=18944, maxN=18944, maxM=32;
    int8_t*A=malloc((size_t)maxM*maxK); memset(A,1,(size_t)maxM*maxK);
    int32_t*C=malloc((size_t)maxM*maxN*4);

    int Ms[]={1,2,4,8,16,32};
    printf("\n%-3s %11s %11s %8s %12s %12s\n","M","1core(us)","3core(us)","scaling","us/tok(3c)","vs M=1");
    double base_per_tok=0;
    for(unsigned mi=0; mi<sizeof(Ms)/sizeof(int); mi++){
        int M=Ms[mi];
        /* warm */
        ork_npu_set_core_budget(c,1);     for(int m=0;m<7;m++) ork_mm_run_i8(c,w[m],M,A,C);
        ork_npu_set_core_budget(c,cores); for(int m=0;m<7;m++) ork_mm_run_i8(c,w[m],M,A,C);

        ork_npu_set_core_budget(c,1);
        double t0=now_us();
        for(int it=0;it<iters;it++) for(int m=0;m<7;m++) ork_mm_run_i8(c,w[m],M,A,C);
        double t1=(now_us()-t0)/iters;

        ork_npu_set_core_budget(c,cores);
        t0=now_us();
        for(int it=0;it<iters;it++) for(int m=0;m<7;m++) ork_mm_run_i8(c,w[m],M,A,C);
        double t3=(now_us()-t0)/iters;

        double per_tok=t3/M;
        if(M==1) base_per_tok=per_tok;
        printf("%-3d %11.1f %11.1f %7.2fx %12.1f %11.2fx\n",
               M, t1, t3, t1/t3, per_tok, base_per_tok/per_tok);
    }
    printf("\nus/tok = per-verified-token NPU cost for the 7 projections (3-core). 'vs M=1' = amortization\n");
    printf("(excludes attention + CPU ops; projections are the bulk of per-token NPU work)\n");
    ork_npu_free(c); free(A); free(C); return 0;
}
