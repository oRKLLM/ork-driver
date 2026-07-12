/* tools/chain_gu_silu_probe.c — CHAIN ASSEMBLER increment-1 (corrected): [gate*silu -> up] in ONE submit via
 * ork_mm_run_chain_i8_gsilu (built on the PROVEN run_chain_i8 + set_i8_silu fused output stage). Validates the
 * first real one-submit FFN unit: task0 = int8 gate matmul with a FUSED SiLU output stage (int8 silu(gate)),
 * task1 = plain int8 up matmul (int32). All-ones inputs -> gate=up=K everywhere (layout-agnostic):
 *   - up (task1, int32) == K       => the chain executed task1
 *   - gate*silu (task0, int8) ~ silu(K*in_scale)/out_scale != 0  => task0 ran AND the fused silu applied
 *   make chain_gu_silu_probe && sudo ./chain_gu_silu_probe      (board only)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static double siluf(double x){ return x/(1.0+exp(-x)); }

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    ork_npu *c=ork_npu_init(); if(!c){ printf("no board\n"); return 0; }
    const int M=8, K=512, N=64;   /* run_chain_i8 envelope: K%512==0 && K<=4096 */
    static signed char A[8*512], Wg[512*64], Wu[512*64];
    for(int i=0;i<M*K;i++) A[i]=1;
    for(int i=0;i<K*N;i++){ Wg[i]=1; Wu[i]=1; }   /* all-ones -> gate=up=K everywhere */

    ork_w *wg=ork_mm_pack_i8(c,K,N,Wg), *wu=ork_mm_pack_i8(c,K,N,Wu);
    if(!wg||!wu){ printf("pack failed\n"); ork_npu_free(c); return 1; }

    if(getenv("ORK_GSILU_SDP")){   /* OPTION B: [gate matmul(int8-out) -> silu-SDP] one submit (vendor matmul->SDP) */
        int Cg[8*64], Cs[8*64]; for(int i=0;i<M*N;i++){ Cg[i]=0; Cs[i]=0; }
        ork_mm_task_i8 t[2] = { { wg, M, A, Cg }, { wg, M, A, Cs } };   /* task0=gate(int8-out); task1=silu-SDP reads Cg */
        double is = 3.0/32.0, os = siluf(3.0)/60.0;   /* gate_i8~32 (gate R=0x4000>>18=1/16 * K=512), silu(3)/os~60 */
        int r=ork_mm_run_chain_i8_sdpsilu(c,2,t,1, 0x4000,18, is,os);
        printf("chain_gu_silu[SDP-silu]: [gate(int8-out) -> silu-SDP(int8)] ONE submit  (M=%d K=%d N=%d)\n",M,K,N);
        printf("  rc=%d\n", r);
        if(r==0){ signed char *cg=(signed char*)Cg,*cs=(signed char*)Cs; int wg8=32, ws=(int)lround(siluf(32*is)/os);
            int ng=0,gmn=127,gmx=-128,ns=0,smn=127,smx=-128;
            for(int i=0;i<M*N;i++){ int g=cg[i],s=cs[i]; if(abs(g-wg8)<=3)ng++; if(g<gmn)gmn=g; if(g>gmx)gmx=g;
                if(abs(s-ws)<=4)ns++; if(s<smn)smn=s; if(s>smx)smx=s; }
            printf("  [task0 gate int8~%d] %d/%d range[%d,%d]\n", wg8,ng,M*N,gmn,gmx);
            printf("  [task1 silu int8~%d] %d/%d range[%d,%d]\n", ws,ns,M*N,smn,smx);
            printf("  VERDICT: %s\n", (ng>0&&ns>0)?"ONE-SUBMIT [gate -> silu-SDP] WORKS (vendor matmul->SDP pattern)":
                                      (ng>0)?"gate ran but silu-SDP empty/wrong":"chain did not execute"); }
        else printf("  VERDICT: %s\n", r==-1?"WEDGED":"error");
        ork_npu_free(c); return (r==0)?0:1;
    }

    /* fused-SiLU LUT for (in_scale,out_scale): real_gate = acc*in_scale; int8_out = silu(real_gate)/out_scale.
     * acc=K; pick in_scale so real_gate~3.0, out_scale so int8_out~60 (well inside int8, avoids saturation). */
    const double is = 3.0/(double)K, os = siluf(3.0)/60.0;
    short lut[1030];
    if(ork_mm_silu_build_lut(c,is,os,0x4000,0x10,0x56391300u,lut)){ printf("build_lut failed\n"); ork_npu_free(c); return 1; }

    int Cg[M*N]; int Cu[M*N];              /* Cg holds int8 silu in its first M*N bytes; Cu = int32 up */
    for(int i=0;i<M*N;i++){ Cg[i]=0; Cu[i]=0; }
    ork_mm_task_i8 tasks[2] = { { wg, M, A, Cg }, { wu, M, A, Cu } };
    int r;
    if(getenv("ORK_GSILU_LAST")){     /* fused gate*silu as the LAST task: [up(plain,task0) -> gate*silu(task1)] */
        ork_mm_task_i8 t[2] = { { wu, M, A, Cu }, { wg, M, A, Cg } };   /* up first, gate*silu last (gate_task=1) */
        r=ork_mm_run_chain_i8_gsilu(c,2,t,1, 0x4000,0x10,0,0xffffc000u,0x56391300u, lut,1030);
        printf("chain_gu_silu[SILU-LAST]: [up(int32) -> gate*silu(int8)] ONE submit  (M=%d K=%d N=%d)\n",M,K,N);
        printf("  rc=%d\n", r);
        if(r==0){ int nup=0,upmax=0; for(int i=0;i<M*N;i++){ if(Cu[i]==K)nup++; if(Cu[i]>upmax)upmax=Cu[i]; }
            signed char *g8=(signed char*)Cg; int want=(int)lround(siluf(K*is)/os); int ng=0,gmn=127,gmx=-128;
            for(int i=0;i<M*N;i++){ int v=g8[i]; if(abs(v-want)<=3)ng++; if(v<gmn)gmn=v; if(v>gmx)gmx=v; }
            printf("  [task0 up] int32==K: %d/%d (max=%d)\n", nup,M*N,upmax);
            printf("  [task1 gate*silu] int8~%d: %d/%d (range [%d,%d])\n", want,ng,M*N,gmn,gmx);
            printf("  VERDICT: %s\n", (nup>0&&ng>0)?"ONE-SUBMIT [up -> gate*silu] WORKS (fused silu as LAST task)":"still wrong"); }
        else printf("  VERDICT: %s\n", r==-1?"WEDGED":"error");
        ork_npu_free(c); return (r==0)?0:1;
    }
    if(getenv("ORK_GSILU_PLAIN")){    /* bisection: ss=NULL plain chain (both int32) -- verify refactor didn't break run_chain_i8 */
        r=ork_mm_run_chain_i8(c,2,tasks);
        printf("chain_gu_silu[PLAIN ss=NULL]: [gate -> up] both int32, ONE submit  (M=%d K=%d N=%d)\n",M,K,N);
        printf("  rc=%d\n",r);
        if(r==0){ int ng=0,nu=0; for(int i=0;i<M*N;i++){ if(Cg[i]==K)ng++; if(Cu[i]==K)nu++; }
            printf("  gate int32==K: %d/%d  up int32==K: %d/%d  VERDICT: %s\n", ng,M*N, nu,M*N, (ng>0&&nu>0)?"PLAIN PATH OK":"PLAIN PATH BROKEN"); }
        ork_npu_free(c); return 0;
    }
    r=ork_mm_run_chain_i8_gsilu(c,2,tasks,0, 0x4000,0x10,0,0xffffc000u,0x56391300u, lut,1030);
    printf("chain_gu_silu: [gate*silu(int8) -> up(int32)] ONE submit  (M=%d K=%d N=%d)\n",M,K,N);
    printf("  rc=%d (0=ran, -1=wedge, -2=dims)\n", r);
    if(r){ printf("  VERDICT: %s\n", r==-1?"WEDGED":"error"); ork_npu_free(c); return r==-1?1:0; }

    /* task1: up int32 == K everywhere? */
    int nup=0, upmax=0; for(int i=0;i<M*N;i++){ if(Cu[i]==K)nup++; if(Cu[i]>upmax)upmax=Cu[i]; }
    /* task0: int8 silu output (read Cg's bytes) ~ silu(K*is)/os */
    signed char *g8=(signed char*)Cg; int want=(int)lround(siluf(K*is)/os);
    int ngate=0, gmin=127,gmax=-128; for(int i=0;i<M*N;i++){ int v=g8[i]; if(abs(v-want)<=3)ngate++; if(v<gmin)gmin=v; if(v>gmax)gmax=v; }
    printf("  [task1 up]   int32==K(%d): %d/%d  (max=%d)\n", K, nup, M*N, upmax);
    printf("  [task0 gate*silu] int8~%d: %d/%d  (range [%d,%d])\n", want, ngate, M*N, gmin, gmax);
    int ok = (nup>0 && ngate>0);
    printf("  VERDICT: %s\n", ok ? "ONE-SUBMIT [gate*silu -> up] WORKS — fused silu task0 + up task1 both executed"
                                 : (nup>0) ? "up ran but gate*silu task0 empty/wrong" : "chain did not execute");
    ork_npu_free(c);
    return ok?0:2;
}
