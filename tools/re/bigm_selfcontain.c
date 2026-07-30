/* bigm_selfcontain — #39 Path-1: run a big-M (M=36) fold tile STANDALONE by making it fully self-contained.
 *
 * ★★ RESULT (2026-07-30, board): HARD-WEDGES. ★★  Giving M=36 its complete register set — its own 108-pair body
 * (84 unique + same-value dup writes; compaction to unique is VALID, regcmd_decode shows 0 {CHANGED} so the dups
 * are idempotent, NOT a per-position value sequence), the 20 output-stage regs via sdp_stamp, AND
 * PC_OPERATION_ENABLE=0xd — makes the op actually START (vs the earlier clean errno-110 of an un-kicked tile), and
 * it then HARD-WEDGES mid-execution (SSH-dead; needs a HA "Rock 5B Plug" power-cycle). This CONFIRMS the genuine,
 * long-documented CBUF/12-bank structural wall: a SINGLE submit cannot hold full-K weight + the M=36 feature tile
 * at once. It is NOT a weight-retention/reuse problem (M=36 streams its own weight; there is no residency to
 * inherit) — the earlier "big-M needs resident weight" framing was WRONG. rkllm runs M=36 only inside its
 * task_number=3 submit that SPLITS the work so each task fits CBUF (the multi-task tiling-planner RE, still open).
 * DO NOT RE-RUN casually: it hard-wedges the board every time. Kept as the RE proof artifact for the wall.
 *
 * The full-sequence decode showed rkllm's M=36 tile omits exactly 23 registers and inherits them: 20 output-stage
 * regs (block 0x1001), the chain descriptor (0x101:0x0014), and — critically — PC_OPERATION_ENABLE (0x81:0x0008
 * = 0xd), the doorbell that kicks the op. It omits them because in rkllm's submit it is a LATER task the kernel
 * re-arms; run standalone with them unset, the op is never configured nor kicked -> errno-110. NONE of the 23 is
 * weight/CNA-related (M=36 streams its own weight), so there is no residency dependency. This harness rebuilds
 * M=36 as a COMPLETE self-contained tile: its own captured regs + the 20 output-stage regs (canonical values via
 * ork_npu_sdp_stamp) + PC_OPERATION_ENABLE=0xd + BLK41=0, then runs it P=1 and verifies bit-exact vs a CPU ref.
 *   sudo env ORK_MM_TIMEOUT=500 ./bigm_selfcontain [M] [K] [N] [tile.txt]      (default 36 3584 1216 m36)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"
extern int ork_npu_mfold_chain_v(ork_npu*,int,const int*,int,int,const unsigned*,int,const int8_t*,const int8_t*,int32_t*,int,int,double*);

static uint32_t rng=0x9e37u; static int r7(void){ rng=rng*1664525u+1013904223u; return (int)((rng>>25)%7)-3; }
static size_t nc16(int m,int c,int w){ return (size_t)(c/16)*((size_t)w*16)+(size_t)m*16+(c%16); }
static size_t woff(int n,int k,int K){ int KT=(K+31)/32; return ((size_t)(n/32)*KT+(k/32))*1024+(size_t)(n%32)*32+(k%32); }
static size_t c4(int m,int n,int w){ return (size_t)(n/4)*((size_t)w*4)+(size_t)m*4+(n%4); }
static int loadtile(const char*p, uint32_t*rc){ FILE*f=fopen(p,"r"); if(!f){perror(p);return 0;} int n=0; while(n<232&&fscanf(f,"%x",&rc[n])==1)n++; fclose(f); return n; }

/* the 20 output-stage (block 0x1001) registers a big-M tile omits + inherits */
static const unsigned DPU_MISSING[20]={0x4098,0x409c,0x40a0,0x40a4,0x40a8,0x40ac,0x40c0,0x40c4,
                                       0x4100,0x4104,0x4108,0x410c,0x4110,0x4114,0x4118,0x411c,0x4120,0x4124,0x4128,0x412c};

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int M=argc>1?atoi(argv[1]):36, K=argc>2?atoi(argv[2]):3584, N=argc>3?atoi(argv[3]):1216;
    const char*tf=argc>4?argv[4]:"/tmp/mm_regcmd_m36.txt";
    static uint32_t cap[232]; if(loadtile(tf,cap)!=232){ printf("load %s fail\n",tf); return 2; }

    /* rebuild a self-contained 232-word tile. M=36's 108 pairs include ~24 DUPLICATE (blk,off) writes (its
     * per-sub-block DMA position writes); all regcmd writes are pre-op config (last write per register wins), so
     * COMPACT to unique registers keeping the last value — then APPEND the 20 omitted output-stage regs + the
     * doorbell. (A bit-exact result also proves the duplicates are idempotent config, not a consumed sequence.) */
    static uint32_t kb[160], kr[160], kw0[160], kw1[160]; int nu=0, seen=0;
    for(int k=0;k+1<216;k+=2){ uint32_t w0=cap[k],w1=cap[k+1]; unsigned reg=w0&0xffff, blk=w1>>16;
        if(w0==0&&w1==0) continue;                                   /* padding */
        if(blk==0x101 && (reg==0x0010||reg==0x0014)) continue;       /* chain descriptor (rewritten below) */
        seen++;
        int f=-1; for(int j=0;j<nu;j++) if(kb[j]==blk&&kr[j]==reg){ f=j; break; }
        if(f<0){ kb[nu]=blk; kr[nu]=reg; kw0[nu]=w0; kw1[nu]=w1; nu++; }
        else { kw0[f]=w0; kw1[f]=w1; } }                             /* last write wins */
    static uint32_t sc[232]; memset(sc,0,sizeof sc);
    int np=0;
    for(int j=0;j<nu;j++){ sc[2*np]=kw0[j]; sc[2*np+1]=kw1[j]; np++; }
    printf("M=%d body: %d reg-writes -> %d unique (collapsed %d duplicate position-writes)\n", M, seen, nu, seen-nu);
    int base_regs=np;
    for(int i=0;i<20;i++){ sc[2*np]=DPU_MISSING[i]; sc[2*np+1]=(0x1001u<<16); np++; }   /* append DPU regs (val 0; sdp_stamp fixes) */
    sc[2*np]=(0x000du<<16)|0x0008; sc[2*np+1]=(0x0081u<<16); np++;                       /* PC_OPERATION_ENABLE = 0xd */
    sc[2*np]=0x0000; sc[2*np+1]=(0x0041u<<16); np++;                                     /* BLK41_R0000 = 0 */
    /* now stamp the canonical output-stage values (0x1001 + 0x801) for this M,N; surfadd = 0x3000 (M=36 dominant) */
    int nset=ork_npu_sdp_stamp(sc,232,M,N, M==36?0x3000u:(uint32_t)(128*M));
    printf("self-contained M=%d: %d base regs + 22 appended = %d pairs (%d words); sdp_stamp set %d output-stage regs\n",
           M, base_regs, np, 2*np, nset);
    if(2*np>224){ printf("ERROR: %d words exceeds REGCMD budget (224)\n",2*np); return 3; }

    /* operands */
    int8_t *A=malloc((size_t)M*K),*W=malloc((size_t)K*N); int32_t*ref=calloc((size_t)M*N,4);
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)r7();
    for(size_t i=0;i<(size_t)K*N;i++) W[i]=(int8_t)r7();
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[(size_t)m*K+k]*W[(size_t)k*N+n]; ref[(size_t)m*N+n]=(int32_t)s; }
    int8_t *Ap=calloc((size_t)M*K,1); for(int m=0;m<M;m++)for(int k=0;k<K;k++) Ap[nc16(m,k,M)]=A[(size_t)m*K+k];
    int8_t *Wp=calloc((size_t)K*N,1); for(int k=0;k<K;k++)for(int n=0;n<N;n++) Wp[woff(n,k,K)]=W[(size_t)k*N+n];
    int32_t *Craw=calloc((size_t)M*N,4);

    ork_npu*c=ork_npu_init(); if(!c){ printf("init (board only)\n"); return 77; }
    ork_npu_set_core_budget(c,1);
    { ork_w*ww=ork_mm_pack_i8(c,K,N,W); if(ww){int32_t*t=calloc((size_t)8*N,4);int8_t*a8=calloc((size_t)8*K,1);ork_mm_run_i8(c,ww,8,a8,t);free(t);free(a8);ork_mm_free(c,ww);} }
    int ws[1]={M}; double us=0;
    int r=ork_npu_mfold_chain_v(c,1,ws,K,N,sc,232,Ap,Wp,Craw,0,3,&us);
    if(r){ printf("RESULT M=%d: chain_v rc=%d (STALL/err) — still not self-runnable\n",M,r); ork_npu_free(c); return 1; }
    long mm=0,mx=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int32_t got=Craw[c4(m,n,M)],rf=ref[(size_t)m*N+n]; long e=labs((long)got-rf); if(e)mm++; if(e>mx)mx=e; }
    printf("RESULT M=%d: %ld/%d mismatch maxerr=%ld  %.1f us  %s\n", M, mm, M*N, mx, us,
           mm?"MISMATCH":"*** BIT-EXACT — big-M runs STANDALONE, self-contained! ***");
    free(A);free(W);free(ref);free(Ap);free(Wp);free(Craw); ork_npu_free(c); return mm?1:0;
}
