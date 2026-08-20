/* tools/silu32_check.c — validate the INT32-output fused SiLU (ork_i8_mm_run_silu32): the silu value is
 * emitted at int32 precision instead of int8, so it should match true silu far better than the int8 path
 * (ablation: int8 silu output = the whole FFN-chain PPL gap). Build a fine-scale LUT, run, compare to CPU silu.
 *   make silu32_check && sudo ./silu32_check [M] [in_scale]      (board only)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double silu(double x){ return x/(1.0+exp(-x)); }

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):64; const int K=512,N=64;
    double in=argc>2?atof(argv[2]):4.88e-4;     /* gate preact scale so the acc sweep spans x in [-8,8] (LUT range) */
    ork_npu*c=ork_npu_init(); if(!c){printf("no board\n");return 0;}
    int8_t*B=malloc((size_t)K*N),*A=malloc((size_t)M*K);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=1;                       /* A=1 -> acc[n]=sum_k B[k,n] = K*b[n] */
    int bcol[64];
    for(int n=0;n<N;n++){ int b=(n-32); if(b>127)b=127; if(b<-128)b=-128; bcol[n]=b;  /* x = acc*in in ~[-8,8] */
        for(int k=0;k<K;k++)B[(size_t)k*N+n]=(int8_t)b; }
    ork_w*w=ork_i8_mm_pack(c,K,N,B); if(!w){printf("pack fail\n");return 2;}

    double xmax=(double)K*32*in; double smax=silu(xmax); double out=smax/30000.0;   /* fine out_scale: ~15-bit (int16 range) */
    const int RM=0x4000,RS=0x10; const unsigned OB=0,IO=0xffffc000u,C4=0x56391300u;
    int16_t lut[1030];
    if(ork_mm_silu_build_lut(c,in,out,RM,RS,C4,lut)){ printf("lut build fail\n"); return 1; }

    int32_t*C32=malloc((size_t)M*N*4);
    if(ork_i8_mm_run_silu32(c,w,M,A,C32,RM,RS,OB,IO,C4,lut,1030)){ printf("silu32 run fail\n"); return 1; }
    /* also run the int8 path for comparison (same LUT is fine — resolution differs at the OUTPUT) */
    int8_t*C8=malloc((size_t)M*N); double out8=smax/127.0; int16_t lut8[1030];
    ork_mm_silu_build_lut(c,in,out8,RM,RS,C4,lut8);
    ork_i8_mm_run_silu(c,w,M,A,C8,RM,RS,OB,IO,C4,lut8,1030);

    double se32=0,mx32=0,se8=0,mx8=0; int cnt=0;
    for(int n=0;n<N;n++){ double acc=(double)K*bcol[n]; double ref=silu(acc*in);
        double g32=C32[n]*out, g8=(double)C8[n]*out8;
        double e32=fabs(g32-ref), e8=fabs(g8-ref); se32+=e32; if(e32>mx32)mx32=e32; se8+=e8; if(e8>mx8)mx8=e8; cnt++; }
    printf("in=%.2e silu_range~%.2f  |  INT32: mean|err|=%.5f max=%.5f  |  int8: mean=%.5f max=%.5f\n",
           in, smax, se32/cnt, mx32, se8/cnt, mx8);
    printf("  -> INT32 is %.1fx more accurate than int8 (mean)\n", (se8>1e-9)? (se8/se32) : 0.0);
    ork_mm_free(c,w); ork_npu_free(c);
    return 0;
}
