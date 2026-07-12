/* tools/chain_gatesilu_probe.c — CHAIN ASSEMBLER increment 1: DATA-CONNECTED matmul(int16-out) -> silu.
 * Phase-0 (chain_probe) only proved the 2-task chain WALKS (matmul out and silu in were separate buffers).
 * This validates the INTERMEDIATE-BUFFER BRIDGE: the gate matmul's int16 output IS the silu's input.
 * The crux question -> does the matmul set_i16_out layout match the silu 0x5018 EWCUBEH input layout?
 * Diagnostic: reads back BOTH G (matmul int16 out, via EWCUBEH) and O (silu out) so a mismatch localizes:
 *   - G matches CPU gate_i16  => matmul writes EWCUBEH cube (bridge input layout OK)
 *   - O matches silu(G)        => silu read G correctly (bridge fully works)
 *   make chain_gatesilu_probe && sudo ./chain_gatesilu_probe      (board only)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static double siluf(double x){ return x/(1.0+exp(-x)); }
static int clampi16(long q){ if(q>32767)q=32767; if(q<-32768)q=-32768; return (int)q; }

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    ork_npu *c=ork_npu_init(); if(!c){ printf("no board\n"); return 0; }
    if(getenv("ORK_GS_SELFTEST")){
        int t0=-1,t1=-1; int r=ork_npu_chain_selftest(c,&t0,&t1);
        printf("chain_selftest: 2 plain int8 matmuls via ork_npu_chain_progs (all-ones, MN=%d)\n", 8*64);
        printf("  rc=%d  task0 slots==K: %d/512  task1 slots==2K: %d/512\n", r, t0, t1);
        printf("  VERDICT: %s\n", (r==0&&t0>0&&t1>0) ? "CHAIN CORE WORKS -- both tasks execute + produce output"
                                : (r==0&&t1>0&&t0==0) ? "task1 ran but task0 EMPTY (the Phase-0 problem persists in chain_progs)"
                                : (r==-1) ? "WEDGED" : "check counts");
        ork_npu_free(c); return (r==0&&t0>0&&t1>0)?0:2;
    }
    const int M=8, K=32, N=64;
    int mult=1, shift=0;                         /* identity requant: gate_i16 = clamp_i16(acc) */
    signed char A[M*K], B[K*N];
    /* small values so the int32 accumulator stays well inside int16 (K=32, |A|,|B|<=2 -> |acc|<=128) */
    for(int i=0;i<M*K;i++) A[i]=(signed char)(((i*7+3)%5)-2);      /* in [-2,2] */
    for(int i=0;i<K*N;i++) B[i]=(signed char)(((i*3+1)%5)-2);      /* in [-2,2] */

    /* CPU reference: gate_i32 -> gate_i16 (requant) -> silu */
    int gate16[M*N]; double gmax=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){
        long acc=0; for(int k=0;k<K;k++) acc += (long)A[m*K+k]*B[k*N+n];
        long q = (shift>=0) ? ((acc*mult) >> shift) : (acc*mult);
        gate16[m*N+n]=clampi16(q);
        double a=fabs((double)gate16[m*N+n]); if(a>gmax)gmax=a;
    }
    if(gmax<1) gmax=1;
    double is = gmax/32000.0, os = 1.0/32000.0;   /* in_scale maps gate_i16 -> real; out_scale for silu range */

    short gate_hw[M*N], out_hw[M*N]; double us=0;
    int rc=ork_npu_chain_gatesilu_i16(c,M,K,N,A,B,mult,shift,is,os,gate_hw,out_hw,&us);
    printf("chain_gatesilu: int8 matmul(int16-out) -> int16 silu, ONE submit  (M=%d K=%d N=%d)\n",M,K,N);
    printf("  rc=%d (0=ran, -1=wedge)  %.1f us\n", rc, us);
    if(rc!=0){ printf("  VERDICT: chain WEDGED/err rc=%d\n", rc); ork_npu_free(c); return rc==-1?1:0; }

    /* stage A: did the matmul write gate_i16 in the EWCUBEH layout the silu expects to read? */
    int gate_bad=0; double gerr=0;
    for(int i=0;i<M*N;i++){ double d=fabs((double)gate_hw[i]-gate16[i]); if(d>gerr)gerr=d; if(d>1.5)gate_bad++; }
    printf("  [stage A] matmul int16-out vs CPU gate_i16 (via EWCUBEH): maxerr=%.1f  mismatch=%d/%d  %s\n",
           gerr, gate_bad, M*N, gate_bad==0?"MATCH (matmul writes EWCUBEH cube)":"MISMATCH (layout differs)");

    /* stage B: did the silu correctly transform its input (using the ACTUAL hw gate it read)? */
    double serr=0; int sbad=0;
    for(int i=0;i<M*N;i++){ double ref=siluf(gate_hw[i]*is)/os; int refi=clampi16(lround(ref));
        double d=fabs((double)out_hw[i]-refi); if(d>serr)serr=d; if(d>3.0)sbad++; }
    printf("  [stage B] silu(out) vs silu(hw_gate): maxerr=%.1f  mismatch=%d/%d  %s\n",
           serr, sbad, M*N, sbad==0?"MATCH (silu read the bridge buffer correctly)":"MISMATCH");

    /* end-to-end: silu of the TRUE gate */
    double eerr=0; for(int i=0;i<M*N;i++){ double ref=siluf(gate16[i]*is)/os; int refi=clampi16(lround(ref));
        double d=fabs((double)out_hw[i]-refi); if(d>eerr)eerr=d; }
    printf("  [end-to-end] out vs silu(cpu_gate): maxerr=%.1f\n", eerr);
    printf("  VERDICT: %s\n",
        (gate_bad==0&&sbad==0) ? "BRIDGE WORKS — data-connected matmul->silu coherent in one submit" :
        (gate_bad!=0)          ? "matmul int16-out NOT in EWCUBEH layout — bridge needs a layout match (RE set_i16_out strides vs silu 0x5018 cube)" :
                                 "matmul out OK but silu misread the bridge buffer — RE silu input stride");
    ork_npu_free(c);
    return (gate_bad==0&&sbad==0)?0:2;
}
