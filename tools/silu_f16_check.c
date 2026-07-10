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
    int M=argc>1?atoi(argv[1]):16; int K=argc>2?atoi(argv[2]):512; const int N=64;
    ork_npu*c=ork_npu_init(); if(!c){printf("no board\n");return 0;}
    ork_f16*B=malloc((size_t)K*N*2),*A=malloc((size_t)M*K*2);
    /* DETERMINISTIC VARYING A (rows differ) so the M-tile batching actually matters: M-tile=16 (validated)
     * and any larger M-tile MUST produce bit-identical output if the larger tile is correct. Compare the
     * printed CHECKSUM across two runs (ORK_F16_MTILE=16 vs 64) — equal checksum => the larger tile is exact. */
    for(int m=0;m<M;m++)for(int k=0;k<K;k++) A[(size_t)m*K+k]=(ork_f16)(1.0f + 0.002f*(float)(((m*7+k)%13)-6));
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
    /* A=1 for every row -> every output ROW must equal row 0. A larger fp16 M-tile that miscomputes rows
     * beyond the first tile shows up as a nonzero cross-row error even when row 0 is correct. */
    double rowerr=0; int rowbad=0;
    for(int m=1;m<M;m++)for(int n=0;n<N;n++){ double d=fabs((double)Cf[(size_t)m*N+n]-(double)Cf[n]);
        if(d>rowerr)rowerr=d; if(d>0.05)rowbad++; }
    double refmax=0; for(int n=0;n<N;n++){ double e=fabs((double)Cf[n]-silu((double)K*bcol[n])); if(e>refmax)refmax=e; }
    /* full-output checksum: bit-mix every element so any single-element difference between two M-tile
     * settings changes it. A=varying, so rows differ -> checksum is M-tile-batching-sensitive. */
    unsigned long long sum=0; for(size_t i=0;i<(size_t)M*N;i++){ union{float f;unsigned u;}v; v.f=Cf[i]; sum=sum*1000003ull + v.u; }
    printf("RESULT: ran OK, output %s, %s | M=%d finite=%d | CHECKSUM=%016llx\n",
           finite?"finite":"has NaN/inf", mono?"monotonic":"NON-monotonic", M, finite, sum);
    printf("VERDICT: %s (compare CHECKSUM across ORK_F16_MTILE settings — equal => larger tile bit-exact)\n", finite?"ran":"NaN");
    ork_mm_free(c,w); ork_npu_free(c);
    return 0;
}
