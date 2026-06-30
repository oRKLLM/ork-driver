/* tools/fp16_check.c — reproduce the layer/model FP16 failure at cbuf=57344. layer.c uses ork_mm_run
 * (fp16), not int8 — so the cbuf-raise breaks the fp16 M-tiling (down @ K=2048 grows 8->14 rows at
 * M=16). Multi-core fp16 matmul + fp32 CPU reference (fp16 inputs, fp32 accumulate = the NPU's math).
 *   make fp16_check && sudo ./fp16_check [M] [K] [N]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ork_npu.h"

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):16, K=argc>2?atoi(argv[2]):2048, N=argc>3?atoi(argv[3]):512;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int cores=ork_npu_cores(c); ork_npu_set_core_budget(c,cores);
    ork_f16*B=malloc((size_t)K*N*sizeof(ork_f16)), *A=malloc((size_t)M*K*sizeof(ork_f16));
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=(ork_f16)(0.05f*((i%7)+1));   /* positive: no cancellation */
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(ork_f16)(0.05f*((i%5)+1));
    float*C=malloc((size_t)M*N*4);
    ork_w*w=ork_mm_pack(c,K,N,B); if(!w){printf("pack failed\n");return 1;}
    ork_mm_run(c,w,M,A,C);
    double maxe=0, refmax=0; int nan=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){
        float acc=0; for(int k=0;k<K;k++) acc+=(float)A[(size_t)m*K+k]*(float)B[(size_t)k*N+n];
        float d=C[(size_t)m*N+n]-acc; if(d<0)d=-d; if(d>maxe)maxe=d;
        float r=acc<0?-acc:acc; if(r>refmax)refmax=r; if(isnan(C[(size_t)m*N+n]))nan++;
    }
    int fail = (maxe > 0.05*refmax + 1e-3) || nan;
    printf("fp16 M=%d K=%d N=%d cores=%d: maxabs=%.4f |ref|=%.3f nan=%d %s\n",
           M,K,N,cores,maxe,refmax,nan, fail?"FAIL":"OK");
    ork_w_free(w); ork_npu_free(c); return fail;
}
