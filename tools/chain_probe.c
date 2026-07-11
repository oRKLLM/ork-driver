/* tools/chain_probe.c — PHASE 0 of the chained-regcmd FFN: does the NPU walk a HETEROGENEOUS 2-task
 * PC-chain (int8 matmul [0xd] -> int16 silu [0x18]) in ONE submit? The FFN-chain-critical CNA/DPU->
 * pure-SDP transition. Verifies (a) rc=0 (no wedge), (b) silu output ~ CPU silu (task1 ran), (c) mm_ran
 * (task0 matmul ran). All three => heterogeneous chain walked in one submit. See CHAINED_FFN_DESIGN.md.
 *   make chain_probe && sudo ./chain_probe        (board only)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static double siluf(double x){ return x/(1.0+exp(-x)); }
int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("no board\n"); return 0; }
    const int M=8,N=64; double gmax=30.0; double is=gmax/32000.0, os=gmax/32000.0;
    short in[M*N], out[M*N]; double gate[N];
    for(int n=0;n<N;n++){ gate[n]=-gmax + 2.0*gmax*n/(double)(N-1);
        for(int m=0;m<M;m++){ long q=lround(gate[n]/is); if(q>32767)q=32767; if(q<-32768)q=-32768; in[m*N+n]=(short)q; } }
    int mm_ran=-1; double us=0;
    int rc=ork_npu_chain_mm_silu_i16(c,in,M,N,is,os,out,&mm_ran,&us);
    printf("chain_probe: matmul(0xd) -> int16 silu(0x18), ONE submit\n");
    printf("  rc=%d (0=ran, -1=wedge)  %.1f us  mm_ran(task0)=%d\n", rc, us, mm_ran);
    if(rc==0){
        double emax=0; for(int n=0;n<N;n++){ double got=(double)out[n]*os, ref=siluf(gate[n]); double d=fabs(got-ref); if(d>emax)emax=d; }
        printf("  silu(task1) err max=%.4g  %s\n", emax, emax<1.0?"CORRECT":"WRONG");
        printf("  VERDICT: %s\n", (emax<1.0&&mm_ran==1)?"HETEROGENEOUS CHAIN WALKS — both tasks in one submit":
                                  (emax<1.0)?"silu ran; matmul task0 output empty":"silu output wrong");
    } else printf("  VERDICT: chain WEDGED (rc=%d) — 0xd->0x18 walk is the blocker\n", rc);
    ork_npu_free(c); return 0;
}
