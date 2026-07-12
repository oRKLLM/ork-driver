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

    if(getenv("ORK_GSILU_FFN6K")){   /* REAL-WIDTH FFN inner: Nff=6144 -> down K-splits (Sk=6) IN the chain */
        enum { Mf=8, Kf=512, Nff=6144, Kd=512 };
        static signed char A6[Mf*Kf], Wg6[Kf*Nff], Wd6[Nff*Kd];
        for(int i=0;i<Mf*Kf;i++) A6[i]=1; for(int i=0;i<Kf*Nff;i++) Wg6[i]=1; for(int i=0;i<Nff*Kd;i++) Wd6[i]=1;
        ork_w *wg6=ork_mm_pack_i8(c,Kf,Nff,Wg6);   /* gate/up [512,6144] */
        ork_w *wd6=ork_mm_pack_i8(c,Nff,Kd,Wd6);   /* down [6144,512], Sk=6 */
        if(!wg6||!wd6){ printf("pack6k failed\n"); ork_npu_free(c); return 1; }
        static int Cg[Mf*Nff],Cs[Mf*Nff],Cu[Mf*Nff],Ch[Mf*Nff],Cd[Mf*Kd];
        for(int i=0;i<Mf*Nff;i++){ Cg[i]=Cs[i]=Cu[i]=Ch[i]=0; } for(int i=0;i<Mf*Kd;i++) Cd[i]=0;
        ork_mm_task_i8 t[5] = { {wg6,Mf,A6,Cg}, {wg6,Mf,A6,Cs}, {wg6,Mf,A6,Cu}, {wg6,Mf,A6,Ch}, {wd6,Mf,A6,Cd} };
        ork_chain_op ops[5] = {
            { 1, -1,0, 0x4000,18 },  /* gate MM8 (reads A) 512->32 */
            { 2, 0,0, 0,0 },         /* silu(gate) */
            { 1, -1,0, 0x4000,18 },  /* up MM8 (reads A) */
            { 3, 1,2, 0x4000,19 },   /* glu = silu*up -> ~61 */
            { 0, 3,0, 0,0 },         /* down MM32 reads glu(t3), K-split Sk=6: out = 61*6144 = 374784 */
        };
        double is = 3.0/32.0, os = siluf(3.0)/60.0;
        int r=ork_mm_run_chain_i8_ffn(c,5,t,ops,is,os);
        printf("chain_gu_silu[FFN6K]: [gate->silu->up->glu->down(K-split Sk=6)] ONE submit  (M=%d K=%d Nff=%d Kd=%d)\n",Mf,Kf,Nff,Kd);
        printf("  rc=%d\n", r);
        if(r==0){ signed char *cg=(signed char*)Cg,*cs=(signed char*)Cs,*cu=(signed char*)Cu,*ch=(signed char*)Ch;
            int g=cg[0],s=cs[0],u=cu[0],h=ch[0],d=Cd[0]; int glu=61, wantd=glu*Nff;
            int nd=0; for(int i=0;i<Mf*Kd;i++) if(Cd[i]==d)nd++;
            printf("  gate=%d silu=%d up=%d glu=%d | down=%d (%d/%d) [want glu*%d=%d]\n", g,s,u,h,d,nd,Mf*Kd,Nff,wantd);
            int ok = (g==32 && s>0 && u==32 && h!=0 && d==wantd);
            printf("  VERDICT: %s\n", ok?"REAL-WIDTH FFN INNER (down K-split IN chain) WORKS, ONE submit, EXACT":
                                        "check values (down K-split may need layout fix)"); }
        else printf("  VERDICT: %s\n", r==-1?"WEDGED":"error");
        ork_npu_free(c); return (r==0)?0:1;
    }
    if(getenv("ORK_GSILU_FFN5")){   /* FULL FFN inner: [gate -> silu -> up -> glu -> down] 5-task one submit */
        enum { Mf=8, Kf=512, Nf=512 };   /* Nf=512 so down's contraction (=glu width) satisfies K%512, <=4096 */
        static signed char A5[Mf*Kf], W5[Kf*Nf];
        for(int i=0;i<Mf*Kf;i++) A5[i]=1; for(int i=0;i<Kf*Nf;i++) W5[i]=1;
        ork_w *w5=ork_mm_pack_i8(c,Kf,Nf,W5);   /* one all-ones [512,512] weight reused for gate/up/down */
        if(!w5){ printf("pack5 failed\n"); ork_npu_free(c); return 1; }
        static int Cg[Mf*Nf],Cs[Mf*Nf],Cu[Mf*Nf],Ch[Mf*Nf],Cd[Mf*Nf];
        for(int i=0;i<Mf*Nf;i++){ Cg[i]=Cs[i]=Cu[i]=Ch[i]=Cd[i]=0; }
        ork_mm_task_i8 t[5] = { {w5,Mf,A5,Cg}, {w5,Mf,A5,Cs}, {w5,Mf,A5,Cu}, {w5,Mf,A5,Ch}, {w5,Mf,A5,Cd} };
        ork_chain_op ops[5] = {
            { 1, -1,0, 0x4000,18 },  /* gate MM8 (reads A) 512->32 */
            { 2, 0,0, 0,0 },         /* silu(gate) */
            { 1, -1,0, 0x4000,18 },  /* up MM8 (reads A) 512->32 */
            { 3, 1,2, 0x4000,19 },   /* glu = silu*up (1/32) -> ~61 */
            { 0, 3,0, 0,0 },         /* down MM32, reads glu(t3): out = 61*512 = 31232 (int32) */
        };
        double is = 3.0/32.0, os = siluf(3.0)/60.0;
        int r=ork_mm_run_chain_i8_ffn(c,5,t,ops,is,os);
        printf("chain_gu_silu[FFN5]: [gate -> silu -> up -> glu -> down] ONE submit  (M=%d K=%d N=%d)\n",Mf,Kf,Nf);
        printf("  rc=%d\n", r);
        if(r==0){ signed char *cg=(signed char*)Cg,*cs=(signed char*)Cs,*cu=(signed char*)Cu,*ch=(signed char*)Ch;
            int g=cg[0],s=cs[0],u=cu[0],h=ch[0],d=Cd[0], glu=61, wantd=glu*Kf;   /* down = glu*512 */
            int ng=0,ns=0,nu=0,nh=0,nd=0; for(int i=0;i<Mf*Nf;i++){ if(cg[i]==g)ng++; if(cs[i]==s)ns++; if(cu[i]==u)nu++; if(ch[i]==h)nh++; if(Cd[i]==d)nd++; }
            printf("  gate=%d(%d) silu=%d(%d) up=%d(%d) glu=%d(%d) down=%d(%d)/%d  [down want ~glu*512=%d]\n",
                   g,ng,s,ns,u,nu,h,nh,d,nd,Mf*Nf,wantd);
            int ok = (g==32 && s>0 && u==32 && h!=0 && d!=0);
            printf("  VERDICT: %s\n", ok?"FULL FFN INNER CHAIN WORKS (gate,silu,up,glu,down all ran, ONE submit)":
                                        "some task empty/wrong -- see values"); }
        else printf("  VERDICT: %s\n", r==-1?"WEDGED":"error");
        ork_npu_free(c); return (r==0)?0:1;
    }
    if(getenv("ORK_GSILU_FFN4")){   /* [gate -> silu -> up -> glu] 4-task one submit (matmul/SDP mix, aliased) */
        static int Cg[8*64], Cs[8*64], Cu[8*64], Ch[8*64];
        for(int i=0;i<M*N;i++){ Cg[i]=Cs[i]=Cu[i]=Ch[i]=0; }
        ork_mm_task_i8 t[4] = { {wg,M,A,Cg}, {wg,M,A,Cs}, {wu,M,A,Cu}, {wg,M,A,Ch} };   /* SDP tasks reuse wg for N sizing */
        ork_chain_op ops[4] = {   /* kind: 1=matmul int8-out, 2=silu-SDP, 3=ewmul-SDP; matmul in0=-1 -> reads A */
            { 1, -1,0, 0x4000,18 },  /* t0 gate: int8 out, requant 1/16 (512->32), reads A */
            { 2, 0,0, 0,0 },         /* t1 silu(in0=gate) */
            { 1, -1,0, 0x4000,18 },  /* t2 up: int8 out (512->32), reads A */
            { 3, 1,2, 0x4000,19 },   /* t3 glu = silu(t1) * up(t2), gain 1/32 */
        };
        double is = 3.0/32.0, os = siluf(3.0)/60.0;
        int r=ork_mm_run_chain_i8_ffn(c,4,t,ops,is,os);
        printf("chain_gu_silu[FFN4]: [gate -> silu -> up -> glu] ONE submit  (M=%d K=%d N=%d)\n",M,K,N);
        printf("  rc=%d\n", r);
        if(r==0){ signed char *cg=(signed char*)Cg,*cs=(signed char*)Cs,*cu=(signed char*)Cu,*ch=(signed char*)Ch;
            int g=cg[0],s=cs[0],u=cu[0],h=ch[0];
            int ng=0,ns=0,nu=0,nh=0; for(int i=0;i<M*N;i++){ if(cg[i]==g)ng++; if(cs[i]==s)ns++; if(cu[i]==u)nu++; if(ch[i]==h)nh++; }
            printf("  t0 gate=%d (%d/%d)  t1 silu=%d (%d/%d)  t2 up=%d (%d/%d)  t3 glu=%d (%d/%d)\n",
                   g,ng,M*N, s,ns,M*N, u,nu,M*N, h,nh,M*N);
            int ok = (g==32 && s>0 && u==32 && h!=0);
            printf("  VERDICT: %s\n", ok?"FULL FFN-INNER-MINUS-DOWN CHAIN WORKS (gate,silu,up,glu all ran, one submit)":
                                        "some task empty/wrong -- see values"); }
        else printf("  VERDICT: %s\n", r==-1?"WEDGED":"error");
        ork_npu_free(c); return (r==0)?0:1;
    }
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
