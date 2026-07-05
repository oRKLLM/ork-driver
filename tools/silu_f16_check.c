/* tools/silu_f16_check.c — smoke-test the fp16 fused gate+SiLU primitive (ork_mm_run_f16_silu): does it RUN
 * (not wedge) and produce finite, silu-shaped fp16->fp32 output? Calibration is a follow-up; this validates
 * the pipeline structure (fp16 matmul + grafted silu output stage, fp16 output CVT kept).
 *   make silu_f16_check && sudo ./silu_f16_check [M]      (board only)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double silu(double x){ return x/(1.0+exp(-x)); }

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):16; const int K=512,N=64;
    ork_npu*c=ork_npu_init(); if(!c){printf("no board\n");return 0;}
    ork_f16*B=malloc((size_t)K*N*2),*A=malloc((size_t)M*K*2);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(ork_f16)1.0f;                 /* A=1 -> gate[n]=sum_k B[k,n] */
    double bcol[64];
    for(int n=0;n<N;n++){ double b=0.0005*(n-32); bcol[n]=b; for(int k=0;k<K;k++)B[(size_t)k*N+n]=(ork_f16)(b); } /* gate=K*b in ~[-8,8] */
    ork_w*w=ork_mm_pack(c,K,N,B); if(!w){printf("fp16 pack fail\n");return 2;}
    float*Cf=malloc((size_t)M*N*4);
    /* NULL lut = built-in default silu curve (calibration TBD); we just check it runs + is silu-shaped */
    int r=ork_mm_run_f16_silu(c,w,M,A,Cf,0,0xffffc000u,0x56391100u,NULL,0);
    if(r){ printf("ork_mm_run_f16_silu rc=%d (wedge/shape/soc)\n",r); return 1; }
    int finite=1,mono=1; double prev=-1e9;
    printf("  n   gate      npu_out    cpu_silu(gate)\n");
    for(int n=0;n<N;n++){ double gate=(double)K*bcol[n]; double got=Cf[n]; double ref=silu(gate);
        if(!isfinite(got))finite=0; if(got<prev-0.5)mono=0; prev=got;
        if(n%8==0) printf("  %2d  %8.3f  %9.4f  %9.4f\n",n,gate,got,ref); }
    printf("RESULT: ran OK, output %s, %s (silu is monotonic-increasing)\n",
           finite?"finite":"has NaN/inf", mono?"monotonic":"NON-monotonic");
    ork_mm_free(c,w); ork_npu_free(c);
    return 0;
}
