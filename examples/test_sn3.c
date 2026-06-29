/* Focused gate for LEVER #1: Sn>1 chain-prefill (ffn_gate/up shape K=3584 N=18944 -> Sn=3).
 * This is the path lever1 restructures (one submit per N-slice). Validates bit-exact vs an
 * OpenMP CPU int32 reference and surfaces any errno-110 / cdma-wild from run_i8. */
#include <stddef.h>
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
static unsigned sd=12345; static int rnd(void){sd=sd*1103515245u+12345u;return (int)((sd>>16)&3);}
static int one(ork_npu*ctx,int M,int K,int N){
    printf("Sn3-gate: M=%d K=%d N=%d (Sn=%d)\n",M,K,N,(N+8191)/8192); fflush(stdout);
    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t*C=malloc((size_t)M*N*4);
    if(!A||!B||!C){printf("  OOM\n");return 1;}
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(int8_t)(rnd()-1);
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=(int8_t)(rnd()-1);
    ork_w*w=ork_mm_pack_i8(ctx,K,N,B);
    if(!w){printf("  pack_i8 FAILED (Sn=%d alloc?)\n",(N+8191)/8192);return 1;}
    errno=0;
    int rc=ork_mm_run_i8(ctx,w,M,A,C);
    if(rc){printf("  run_i8 FAILED rc=%d errno=%d (%s)\n",rc,errno,strerror(errno));ork_w_free(w);return 1;}
    /* Verify a subset of rows fully across ALL N (catches per-N-slice boundary bugs). */
    int rows = M<8?M:8;
    long bad=0;
    #pragma omp parallel for reduction(+:bad)
    for(int i=0;i<rows;i++){
        for(int n=0;n<N;n++){int32_t ref=0;for(int k=0;k<K;k++)ref+=(int)A[(size_t)i*K+k]*(int)B[(size_t)k*N+n];
            if(C[(size_t)i*N+n]!=ref){ if(bad<3)printf("    mism @ (%d,%d): got %d ref %d\n",i,n,C[(size_t)i*N+n],ref); bad++; }}
    }
    /* Also spot-check the last row + last column (the tail N-slice / Sn-3 edge). */
    {int i=M-1; for(int n=N-64;n<N;n++){int32_t ref=0;for(int k=0;k<K;k++)ref+=(int)A[(size_t)i*K+k]*(int)B[(size_t)k*N+n];
        if(C[(size_t)i*N+n]!=ref){ if(bad<6)printf("    tail mism @ (%d,%d): got %d ref %d\n",i,n,C[(size_t)i*N+n],ref); bad++; }}}
    printf("  %s (checked %d full rows + tail; mism=%ld)\n",bad?"WRONG":"ok",rows,bad); fflush(stdout);
    ork_w_free(w);free(A);free(B);free(C);
    return bad?1:0;
}
int main(void){
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed\n");return 1;}
    int fail=0;
    fail|=one(ctx,256,3584,18944);   /* ffn gate/up prefill, Sn=3 */
    fail|=one(ctx,512,3584,18944);   /* larger M, more M-tiles chained per N-slice */
    ork_npu_free(ctx);
    printf("%s\n",fail?"SN3 GATE: FAIL":"SN3 GATE: PASS");
    return fail;
}
