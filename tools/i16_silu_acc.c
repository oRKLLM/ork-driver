/* tools/i16_silu_acc.c — is the on-NPU int16 SiLU (ork_i16_npu_silu) accurate enough to fix the gmax-gate
 * coherence? Compares its silu(gate) error vs CPU fp32 silu over the gate range [-gmax,gmax], in silu units
 * — directly comparable to f16_gate_acc's numbers (int8 fused LUT err = 92 @gmax132; fp16+CPU-silu = 0.03).
 * If int16-silu err is small at high gmax, the UN-FUSED int8-matmul -> int16-silu gate is coherent (and
 * stays on the integer datapath, avoiding the fp16 -1/wedge). Standalone op, no matmul -> low wedge risk.
 *   make i16_silu_acc && sudo ./i16_silu_acc     (board only)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static double siluf(double x){ return x/(1.0+exp(-x)); }

static void run_gmax(ork_npu*c,double gmax){
    const int M=8,N=64; static short in[512],out[512];
    double is=gmax/32000.0;                     /* gate = in*is; gate in [-gmax,gmax] -> in in [-32000,32000] */
    double smax=siluf(gmax); double os=smax/32000.0; if(os<=0)os=1e-9;  /* silu(gate)=out*os */
    double gate[64];
    for(int n=0;n<N;n++){ gate[n]=-gmax + 2.0*gmax*n/(double)(N-1);
        for(int m=0;m<M;m++){ long q=lround(gate[n]/is); if(q>32767)q=32767; if(q<-32768)q=-32768; in[m*N+n]=(short)q; } }
    double us=0; int r=ork_i16_npu_silu(c,in,M,N,is,os,out,&us);
    if(r){ printf("  gmax=%6.1f: rc=%d\n",gmax,r); return; }
    double emax=0,esum=0;
    for(int n=0;n<N;n++){ double got=(double)out[n]*os, ref=siluf(gate[n]); double d=fabs(got-ref);
        if(d>emax)emax=d; esum+=d; }
    printf("  gmax=%6.1f | int16-silu err max=%.4g mean=%.4g   (%.1f us)   [int8 fused LUT was 92@132; fp16+cpu 0.03]\n",
           gmax, emax, esum/N, us);
}
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("no board\n");return 0;}
    printf("i16_silu_acc: on-NPU int16 SiLU error vs CPU fp32 silu, over gate range\n");
    double g[]={8,16,30,64,132}; for(int i=0;i<5;i++) run_gmax(c,g[i]);
    printf("VERDICT: small err at gmax132 => int16-silu fixes the gmax-gate coherence (un-fused int8-mm + int16-silu).\n");
    ork_npu_free(c); return 0;
}
