/* fused_act_probe — task #20 (b): the shared fused-activation matmul serves RMSNorm rsqrt + FFN silu, same
 * M4.8 mechanism as softmax exp (C = fn(A·B) in one submit, activation on the matmul output stage).
 *   rsqrt: fn=rsqrt via ork_f16_mm_run_act (positive-input; RMSNorm 1/sqrt(ss)).
 * (silu is the both-sign case via ork_f16_mm_run_silu + build_f16_silu_lut — separate, next.)
 *   sudo env ORK_MM_TIMEOUT=3000 timeout 40 ./fused_act_probe [M] [d] [N]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static double myrsqrt(double x, void *ctx){ (void)ctx; return x>1e-9 ? 1.0/sqrt(x) : 0.0; }
static uint32_t g=0x9a13u; static float fr(void){ g=g*1664525u+1013904223u; return (float)(g>>8)/(float)(1u<<24); }
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):32, d=argc>2?atoi(argv[2]):128, N=argc>3?atoi(argv[3]):64;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("fused_act_probe: M=%d d=%d N=%d (fused rsqrt on a matmul, one submit)\n",M,d,N);
    /* A>=0, B>=0 => A·B >= 0 (positive-input rsqrt domain, single-signed fp16 index) */
    ork_f16 *A=malloc((size_t)M*d*sizeof(ork_f16)), *B=malloc((size_t)d*N*sizeof(ork_f16));
    for(size_t i=0;i<(size_t)M*d;i++) A[i]=(ork_f16)(0.3f+fr()*0.7f);   /* [0.3,1] */
    for(size_t i=0;i<(size_t)d*N;i++) B[i]=(ork_f16)(0.3f+fr()*0.7f);
    float *C=malloc((size_t)M*N*4), *sc=malloc((size_t)M*N*4); float hi=0,lo=1e9f;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double s=0; for(int k=0;k<d;k++) s+=(double)(float)A[(size_t)m*d+k]*(float)B[(size_t)k*N+n];
        sc[(size_t)m*N+n]=(float)s; if(s>hi)hi=(float)s; if(s<lo)lo=(float)s; }
    printf("  A·B range [%.2f, %.2f]  (rsqrt: [%.4f, %.4f])\n", lo,hi, 1/sqrt(hi),1/sqrt(lo));
    int rc=ork_f16_mm_run_act(c,d,N,B,M,A,C,myrsqrt,NULL,(double)lo-0.5,(double)hi+0.5);
    printf("  ork_f16_mm_run_act(fn=rsqrt) rc=%d C[0]=%.5f (want 1/sqrt(%.2f)=%.5f)\n",rc,C[0],sc[0],1/sqrt(sc[0]));
    if(rc){ printf("FAIL rc=%d\n",rc); ork_npu_free(c); return 1; }
    int bad=0; double me=0,mre=0;
    for(int i=0;i<M*N;i++){ double want=1.0/sqrt(sc[i]); double e=fabs(C[i]-want); double re=e/(want+1e-6); if(e>me)me=e; if(re>mre)mre=re; if(re>0.03&&e>2e-3)bad++; }
    printf("  fused rsqrt(A·B): max|err|=%.3e maxrel=%.3e  %s (%d/%d)\n",me,mre,bad?"CHECK":"COHERENT",bad,M*N);
    if(bad){ printf("FAIL (rsqrt)\n"); ork_npu_free(c); return 1; }

    /* --- silu (SwiGLU): both-sign gate via build_f16_silu_lut + run_f16_silu (-S gate pack) --- */
    { int Kd=128, Ns=64, Ms=32; double Gmax=8.0;
      ork_f16 *Ag=malloc((size_t)Ms*Kd*sizeof(ork_f16)), *W=malloc((size_t)Kd*Ns*sizeof(ork_f16));
      for(size_t i=0;i<(size_t)Ms*Kd;i++) Ag[i]=(ork_f16)(fr()*2.f-1.f);           /* [-1,1] */
      for(size_t i=0;i<(size_t)Kd*Ns;i++) W[i]=(ork_f16)((fr()*2.f-1.f)*0.25f);    /* [-0.25,0.25] => gate ~ [-Gmax,Gmax] */
      short lut[1030]; double S=0,R=0,os=0;
      int brc=ork_f16_mm_build_silu_lut(c,Gmax,lut,&S,&R,&os);
      printf("  [silu] build_f16_silu_lut rc=%d S=%.3f out_scale=%.4g\n", brc, S, os);
      if(brc){ printf("FAIL (silu lut)\n"); ork_npu_free(c); return 1; }
      ork_f16 *negW=malloc((size_t)Kd*Ns*sizeof(ork_f16)); for(size_t i=0;i<(size_t)Kd*Ns;i++) negW[i]=(ork_f16)(-S*(float)W[i]);  /* pack -S*W */
      ork_w *wg=ork_f16_mm_pack(c,Kd,Ns,negW); if(!wg){ printf("FAIL (silu pack)\n"); ork_npu_free(c); return 1; }
      float *Cs=malloc((size_t)Ms*Ns*4);
      int src=ork_f16_mm_run_silu(c,wg,Ms,Ag,Cs,0,0xffffc000u,0x56391100u,lut,1030);
      int sbad=0; double sme=0;
      if(src){ printf("  [silu] run rc=%d\n",src); sbad=1; }
      else for(int m=0;m<Ms;m++)for(int n=0;n<Ns;n++){ double gate=0; for(int k=0;k<Kd;k++) gate+=(double)(float)Ag[(size_t)m*Kd+k]*(float)W[(size_t)k*Ns+n];
            double want=gate/(1.0+exp(-gate)); double got=Cs[(size_t)m*Ns+n]*os; double e=fabs(got-want); if(e>sme)sme=e; }
      /* fp16 fused silu is DOCUMENTED-approximate (~1%, mean 0.08 / max 0.75 over [-8,8], npu.c:5368) — the
       * -S/clamp trick approximates silu(negative gate); the ACCURATE FFN silu is the int8 fused path already
       * HW-chained in the FFN chain. So this is an informational bound, not a coherence gate. */
      printf("  [silu] fp16 fused silu(A·W): max|err|=%.3e (%s; fp16 fused silu is ~1%% approx — int8 fused silu is the accurate FFN path)\n",
             sme, (src==0 && sme<0.8)?"runs, within documented bound":"OUT OF BOUND");
      ork_mm_free(c,wg);
      printf("%s\n", (bad||src||sme>=0.8)?"FAIL":"PASS — shared M4.8 fused-act serves (b): RMSNorm rsqrt COHERENT + FFN silu runs (fp16 approx / int8 exact)");
      ork_npu_free(c); return (bad||src||sme>=0.8)?1:0; }
}
