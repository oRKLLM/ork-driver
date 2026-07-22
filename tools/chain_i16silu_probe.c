/* chain_i16silu_probe — validate the FIXED set_i16_out feeds the int16 SiLU (the matmul-i16-out -> int16-silu
 * data bridge, the gating step for the fp16/int16-quality coalesced FFN). ork_npu_chain_gatesilu_i16 does
 * gate=A*B (int8 matmul, set_i16_out -> int16 G) then silu(G) -> O, with the silu reading G's buffer directly.
 * ORK_GS_SEQ=1 isolates the DATA bridge (two submits, no PC-chain walk). Coherence: O ~= silu(gate). Board only.
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static double siluf(double x){ return x/(1.0+exp(-x)); }
static int clampi16(long q){ if(q>32767)q=32767; if(q<-32768)q=-32768; return (int)q; }
int main(void){
    ork_npu *c=ork_npu_init(); if(!c){ printf("no board\n"); return 0; }
    enum { M=8, K=512, N=64 };
    const int mult=1, shift=0;   /* identity requant: gate_i16 = clamp_i16(acc), small acc */
    static signed char A[M*K], B[K*N];
    for(int i=0;i<M*K;i++) A[i]=(signed char)(((i*7+3)%5)-2);   /* small -> |acc|<=~128 stays in int16 */
    for(int i=0;i<K*N;i++) B[i]=(signed char)(((i*3+1)%5)-2);
    int gate16[M*N]; double gmax=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long acc=0; for(int k=0;k<K;k++) acc+=(long)A[m*K+k]*B[k*N+n];
        gate16[m*N+n]=clampi16((shift>=0)?((acc*mult)>>shift):(acc*mult)); double a=fabs((double)gate16[m*N+n]); if(a>gmax)gmax=a; }
    if(gmax<1)gmax=1; double is=gmax/32000.0, os=1.0/32000.0;
    short gate_hw[M*N], out_hw[M*N]; double us=0;
    int rc=ork_npu_chain_gatesilu_i16(c,M,K,N,A,B,mult,shift,is,os,gate_hw,out_hw,&us);
    printf("chain_gatesilu_i16 rc=%d (0=ran,-1=wedge) %.0fus  M=%d K=%d N=%d  mode=%s\n",
           rc,us,M,K,N, getenv("ORK_GS_SEQ")?"GS_SEQ(2 submits)":"chain_progs");
    if(rc!=0){ printf("  VERDICT: %s\n", rc==-1?"WEDGE/stall":"err"); ork_npu_free(c); return 1; }
    /* end-to-end: did the silu correctly read the matmul's int16 output? O ~= silu(gate) */
    int sbad=0; double serr=0;
    for(int i=0;i<M*N;i++){ double ref=siluf(gate16[i]*is)/os; int refi=clampi16(lround(ref));
        double d=fabs((double)out_hw[i]-refi); if(d>serr)serr=d; if(d>3.0)sbad++; }
    printf("  [end-to-end] O vs silu(cpu_gate): maxerr=%.1f  mismatch=%d/%d\n", serr, sbad, M*N);
    printf("  VERDICT: %s\n", sbad==0 ? "BRIDGE WORKS — fixed set_i16_out feeds the int16 SiLU coherently"
                                      : "silu misread the matmul int16 output — WRITE!=READ layout (reconcile silu 0x5018 geom to the fp16-surface stride)");
    ork_npu_free(c);
    return sbad==0?0:2;
}
