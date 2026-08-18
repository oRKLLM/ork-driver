/* examples/test_i4_gemm.c — validate + benchmark the BATCHED int4 NEON GEMM (ork_cpu_gemm_i4) against the
 * M=1 gemv it replaces. Pure CPU kernel probe: no model, no NPU, <1s. Two checks:
 *   (1) BIT-EXACT vs M separate ork_cpu_gemv_m1 calls (the reference it must match).
 *   (2) the M>1 speedup (weight/unpack amortized across rows) — the prefill lever the M=1 gemv lacks.
 * A/B the win across M: ./test_i4_gemm [M]   (default sweeps M=1,4,8,16,32,64).
 *   make test_i4_gemm && sudo ./test_i4_gemm
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "ork_native_cpu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static void fillf (float *p,size_t n,unsigned s){ for(size_t i=0;i<n;i++){ s=s*1103515245u+12345u; p[i]=((int)((s>>16)%1000)-500)/500.0f; } }
static void filli8(int8_t*p,size_t n,unsigned s){ for(size_t i=0;i<n;i++){ s=s*1103515245u+12345u; p[i]=(int8_t)((int)((s>>17)%255)-127); } }

static int run_M(ork_cpu_fmt fmt,int K,int N,int M){
    float  *W  = malloc((size_t)N*K*sizeof(float)); fillf(W,(size_t)N*K,7);
    uint8_t*nib= malloc((size_t)N*K/2);  float *bsc = malloc((size_t)N*sizeof(float));
    int8_t lut16[16];
    ork_cpu_pack(fmt,K,N,W,nib,NULL,NULL,NULL,NULL,bsc,lut16);
    ork_cpu_w w; memset(&w,0,sizeof w); w.fmt=fmt; w.nibble=nib; w.bscale=bsc; w.K=K; w.N=N;
    if(fmt==ORK_CPU_NF4) w.nf4_lut=vld1q_s8(lut16);

    int8_t *A  = malloc((size_t)M*K); filli8(A,(size_t)M*K,23);
    float  *asc= malloc((size_t)M*sizeof(float)); for(int m=0;m<M;m++) asc[m]=0.01f+0.0007f*m;
    float  *og = malloc((size_t)M*N*sizeof(float));   /* batched */
    float  *or_= malloc((size_t)M*N*sizeof(float));   /* reference (M x gemv) */
    #define GEMM(o) do{ if(fmt==ORK_CPU_NF4) ork_cpu_gemm_nf4(&w,A,K,asc,o,N,M,0,N); else ork_cpu_gemm_i4(&w,A,K,asc,o,N,M,0,N); }while(0)

    for(int m=0;m<M;m++) ork_cpu_gemv_m1(&w, A+(size_t)m*K, asc[m], or_+(size_t)m*N, 0, N);
    GEMM(og);
    double maxd=0; for(size_t i=0;i<(size_t)M*N;i++){ double d=fabs((double)og[i]-(double)or_[i]); if(d>maxd)maxd=d; }

    double bg=1e18,bb=1e18;
    for(int r=0;r<5;r++){ double t=now_us(); for(int it=0;it<3;it++) for(int m=0;m<M;m++) ork_cpu_gemv_m1(&w,A+(size_t)m*K,asc[m],or_+(size_t)m*N,0,N); double u=(now_us()-t)/3; if(u<bg)bg=u; }
    for(int r=0;r<5;r++){ double t=now_us(); for(int it=0;it<3;it++) GEMM(og); double u=(now_us()-t)/3; if(u<bb)bb=u; }
    #undef GEMM

    printf("  %s M=%-3d K=%d N=%d: %-10s | gemv(Mx1) %8.1f us  gemm(batched) %8.1f us  speedup %.2fx\n",
           fmt==ORK_CPU_NF4?"NF4":"I4 ", M,K,N, maxd==0?"BIT-EXACT":"*MISMATCH*", bg, bb, bg/bb);
    free(W);free(nib);free(bsc);free(A);free(asc);free(og);free(or_);
    return maxd==0?0:1;
}

int main(int argc,char**argv){
    const int K=2048, N=512;   /* qwen35moe gate/up expert shape */
    printf("batched int4/NF4 NEON GEMM (ork_cpu_gemm_i4/nf4) vs M=1 gemv:\n");
    int bad=0;
    int Ms[]={1,4,8,16,32,64}; int nM = argc>1?1:6; int Mone = argc>1?atoi(argv[1]):0;
    for(int i=0;i<nM;i++){ int M = argc>1?Mone:Ms[i]; bad|=run_M(ORK_CPU_I4,K,N,M); bad|=run_M(ORK_CPU_NF4,K,N,M); }
    printf("  ---- %s ----\n", bad?"FAIL (mismatch)":"OK (all bit-exact)");
    return bad;
}
