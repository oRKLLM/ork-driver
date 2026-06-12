/* examples/test_matmul.c — validates the ork_npu library: one handle, resident weights reused across
 * many matmuls of varying M (the forward-pass access pattern). Builds vs CPU reference.
 *   cc -O2 -I. -o test_mm test_mm.c ork_npu.c && sudo ./test_mm */
#include <stdio.h>
#include <stdlib.h>
#include "ork_npu.h"
typedef ork_f16 f16;
static unsigned sd=12345; static int rnd(){sd=sd*1103515245+12345;return (sd>>16)%4;}

static int check(ork_npu*ctx,int M,int K,int N){
    f16*A=malloc((size_t)M*K*2),*B=malloc((size_t)K*N*2); float*C=malloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(f16)rnd();
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=(f16)rnd();
    ork_w*w=ork_mm_pack(ctx,K,N,B);
    if(!w){printf("pack failed %d,%d\n",K,N);return 1;}
    int bad=0;
    /* run the SAME resident weights for several M (decode then prefill), validate each */
    int Ms[]={1,1,4,M}; for(int t=0;t<4;t++){int m=Ms[t]; if(m>M)m=M;
        if(ork_mm_run(ctx,w,m,A,C)){printf("run failed\n");return 1;}
        for(int i=0;i<m;i++)for(int n=0;n<N;n++){float ref=0;for(int k=0;k<K;k++)ref+=(float)A[(size_t)i*K+k]*(float)B[(size_t)k*N+n]; if(C[(size_t)i*N+n]!=ref)bad++;}
    }
    printf("  %s MKN=%d,%d,%d (reused weights x4 runs) mism=%d\n",bad?"WRONG":"ok  ",M,K,N,bad);
    ork_w_free(w); free(A);free(B);free(C); return bad?1:0;
}
int main(void){
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    int fail=0;
    fail|=check(ctx,128,512,128);
    fail|=check(ctx,256,4096,512);
    fail|=check(ctx,512,8192,128);
    fail|=check(ctx,64,11008,64);     /* non-power-of-2 K */
    fail|=check(ctx,32,2048,256);
    fail|=check(ctx,8,512,16384);     /* N-tiling: N>8192 (NPU output-width cap) */
    fail|=check(ctx,1,288,32000);     /* LM-head shape: non-pow2 K + N tiled into 4 slices */
    ork_npu_free(ctx);
    printf("%s\n",fail?"FAIL":"ALL OK");
    return fail?1:0;
}
