/* tools/ork_trace_mm.c — run ONE single-core int8 ork matmul at argv M,K,N. ORK_TRACE=1 dumps the
 * regcmd; otherwise it CPU-reference-checks the output (varied data → catches per-row M-tile errors)
 * and optionally loops `iters` for utilization sampling. For the single-core M-tile-size experiment
 * (ORK_MCAP/ORK_RCAP/ORK_R1040): validates bit-exactness before trusting a speed change.
 *   make ork_trace_mm && sudo ./ork_trace_mm 92 3584 512          # correctness
 *   sudo ORK_MCAP=46 ORK_RCAP=30 ORK_R1040=117 ./ork_trace_mm 92 3584 512   # rknn-tile, checked
 *   sudo ORK_TRACE=1 ./ork_trace_mm 256 3584 3584 2>rc.log        # regcmd dump
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ork_npu.h"
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):256, K=argc>2?atoi(argv[2]):3584, N=argc>3?atoi(argv[3]):3584;
    int iters=argc>4?atoi(argv[4]):0;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int nc=1; { const char*e=getenv("ORK_TMM_CORES"); if(e) nc=atoi(e); }
    ork_npu_set_core_budget(c,nc);   /* ORK_TMM_CORES=3 to exercise the multi-core path (layer/model) */
    int8_t*B=malloc((size_t)K*N), *A=malloc((size_t)M*K);
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)((i%5)-2);
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)((i%7)-3);
    int32_t*C=malloc((size_t)M*N*4);
    ork_w*w=ork_i8_mm_pack(c,K,N,B); if(!w){printf("pack failed\n");return 1;}
    ork_i8_mm_run(c,w,M,A,C);                      /* ORK_TRACE prints regcmd here */
    int rc=0;
    if(!getenv("ORK_TRACE") && (long)M*N<=4000000){  /* CPU int8 reference check */
        int mism=0; long maxe=0;
        for(int m=0;m<M;m++)for(int n=0;n<N;n++){
            long acc=0; for(int k=0;k<K;k++) acc+=(long)A[(size_t)m*K+k]*B[(size_t)k*N+n];
            long d=(long)C[(size_t)m*N+n]-acc; if(d<0)d=-d; if(d>maxe)maxe=d; if(C[(size_t)m*N+n]!=acc)mism++;
        }
        printf("CPU-ref check M=%d K=%d N=%d: mism=%d maxerr=%ld %s\n",M,K,N,mism,maxe,mism?"FAIL":"OK");
        rc = mism?1:0;
    }
    for(int i=0;i<iters;i++) ork_i8_mm_run(c,w,M,A,C);
    if(iters) printf("ork ran M=%d K=%d N=%d single-core (%d iters)\n",M,K,N,iters);
    ork_w_free(w); ork_npu_free(c); free(A);free(B);free(C); return rc;
}
