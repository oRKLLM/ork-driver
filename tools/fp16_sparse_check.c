/* fp16_sparse_check.c — fp16 matmul correctness at a chosen (M,K,N,cores), sparse-sampled (full O(MNK)
 * ref is slow). Validates the claim "fp16 miscomputes M-tiles > 8 at K=2048": single-core run() uses
 * sched=1 mc=64 at K=2048, multi-core mcworker uses sched=0 mc=8 — compare both vs a fp32 reference.
 *   fp16_sparse_check M K N [cores] */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ork_npu.h"
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):512, K=argc>2?atoi(argv[2]):2048, N=argc>3?atoi(argv[3]):2048, cores=argc>4?atoi(argv[4]):1;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    ork_npu_set_core_budget(c,cores);
    ork_f16*B=malloc((size_t)K*N*sizeof(ork_f16)), *A=malloc((size_t)M*K*sizeof(ork_f16));
    /* positive, non-cancelling fill so |ref| is large (a cancelling fill hides errors) */
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=(ork_f16)(0.02f*((i%11)+1));
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(ork_f16)(0.02f*((i%7)+1));
    float*C=malloc((size_t)M*N*4);
    ork_w*w=ork_f16_mm_pack(c,K,N,B); if(!w){printf("pack failed\n");return 1;}
    if(ork_f16_mm_run(c,w,M,A,C)){printf("run failed\n");return 1;}
    int rows[]={0,1,7,8,9,15,16,63,64,65,127,200,255,M-1}; int nr=sizeof(rows)/sizeof(rows[0]);
    int cols[]={0,1,63,N/2,N-1}; int ncl=sizeof(cols)/sizeof(cols[0]);
    double maxrel=0; int bad=0, nan=0, ns=0;
    for(int ri=0;ri<nr;ri++){int i=rows[ri]; if(i<0||i>=M)continue;
        for(int ci=0;ci<ncl;ci++){int n=cols[ci]; if(n<0||n>=N)continue; ns++;
            float acc=0; for(int k=0;k<K;k++) acc+=(float)A[(size_t)i*K+k]*(float)B[(size_t)k*N+n];
            float got=C[(size_t)i*N+n]; if(isnan(got))nan++;
            float r=fabsf(acc), d=fabsf(got-acc), rel=r>1e-3f?d/r:d;
            if(rel>maxrel)maxrel=rel;
            if(rel>0.03f){bad++; if(bad<=4)printf("  MISM (%d,%d): got %.4f ref %.4f rel %.3f\n",i,n,got,acc,rel);}
        }}
    int fail=bad||nan;
    printf("fp16 M=%d K=%d N=%d cores=%d: samples=%d maxrel=%.4f nan=%d %s\n",M,K,N,cores,ns,maxrel,nan,fail?"FAIL":"OK");
    ork_w_free(w); ork_npu_free(c); return fail;
}
