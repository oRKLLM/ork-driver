/* examples/i4.c — W4A4 (int4 A x int4 B -> int32 C) via the public ork_mm_pack_i4 / ork_mm_run_i4.
 * Self-validates the NPU result against a CPU int4xint4 reference across shapes that exercise the
 * API's tiling: M>1 (M-tiling), N>64 (N-tiling at 64), and K>10752 (K-split with int32 accumulate).
 *   make i4 && sudo ./i4
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "ork_npu.h"
static double ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }

static int test(ork_npu*c,int M,int K,int N){
    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t*C=malloc((size_t)M*N*4);
    unsigned sd=99+M*7+K*13+N*17;
    for(size_t i=0;i<(size_t)M*K;i++){ sd=sd*1103515245+12345; A[i]=(int8_t)((int)((sd>>17)%15)-7); }
    for(size_t i=0;i<(size_t)K*N;i++){ sd=sd*1103515245+12345; B[i]=(int8_t)((int)((sd>>17)%15)-7); }
    ork_w*w=ork_mm_pack_i4(c,K,N,B); if(!w){ printf("  M=%d K=%d N=%d: pack failed\n",M,K,N); free(A);free(B);free(C); return 1; }
    int rc=ork_mm_run_i4(c,w,M,A,C); ork_w_free(w);
    if(rc){ printf("  M=%d K=%d N=%d: run rc=%d\n",M,K,N,rc); free(A);free(B);free(C); return 1; }
    long maxe=0; int bad=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[m*K+k]*B[k*N+n];
        long e=C[m*N+n]-s; if(e<0)e=-e; if(e>maxe)maxe=e; if(e)bad++; }
    printf("  M=%-3d K=%-6d N=%-5d  maxerr=%-4ld %s  (Sk=%d Sn=%d)\n",M,K,N,maxe,
           maxe==0?"OK":"FAIL",(K+10751)/10752,(N+8191)/8192);
    free(A);free(B);free(C); return maxe!=0;
}
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed (NPU?)\n");return 1;}
    printf("W4A4 public API (ork_mm_pack_i4/ork_mm_run_i4) vs CPU int4 reference:\n");
    int fail=0;
    fail|=test(c,1,64,64);        /* baseline (decode)            */
    fail|=test(c,4,128,128);      /* M-tiling + N-tiling          */
    fail|=test(c,2,256,64);       /* K within one slice           */
    fail|=test(c,1,12288,64);     /* K-split (>10752) + accumulate */
    fail|=test(c,3,2048,256);     /* M + N tiling, mid K          */
    printf("%s\n", fail?"SOME TESTS FAILED":"ALL W4A4 API TESTS PASSED");

    if(getenv("ORK_I4_BENCH")){   /* decode-shape throughput: M=1, time R runs (set ORK_NPU_MC to compare) */
        int K=2048,N=2048,R=100; signed char*A=malloc(K),*B=malloc((size_t)K*N); int32_t*C=malloc((size_t)N*4);
        for(int i=0;i<K;i++) A[i]=(i%15)-7;
        for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int)(i%15)-7;
        ork_w*w=ork_mm_pack_i4(c,K,N,B);
        if(w){ ork_mm_run_i4(c,w,1,A,C); ork_mm_run_i4(c,w,1,A,C);   /* warm */
            double t0=ms(); for(int r=0;r<R;r++) ork_mm_run_i4(c,w,1,A,C); double dt=(ms()-t0)/R;
            const char*mc=getenv("ORK_NPU_MC");
            printf("decode bench M=1 K=%d N=%d: %.3f ms/matmul (%.0f matmul/s)  cores=%s\n",K,N,dt,1000.0/dt,mc?mc:"auto");
            ork_w_free(w); }
        free(A);free(B);free(C);
    }
    ork_npu_free(c); return fail;
}
