/* i4_tp_precision_probe — E3: what does DPU_WDMA_SIZE_0.TP_PRECISION (0x4058 bit 27) do?
 *
 * The int4 BCHAIN output slot is 4x the cells actually written: 2 from the cube height (synth_i4
 * programs mc_phys = 2*mc, so logical row m lands at physical row 2m) and 2 more that have no
 * identified register — int16 data sitting at what looks like an int32 row pitch. 0x4058 carries a
 * one-bit precision selector on the WRITE DMA that both vendor captures leave 0 and our synth
 * silently clears (it writes plain N-1). Its companion is DPU_FEATURE_MODE_CFG.TP_EN (0x400c bit
 * 30), also 0 in both captures. If TP_PRECISION is the output element-size selector, it should
 * account for that second factor.
 *
 * HOW A HALVED PITCH IS VISIBLE FROM THE CALLER. bch_db_cells_off de-tiles with
 *     base = (4*j + 4*Hg*b) * 64        (int16 elements)
 * With NBc == 1 (one 64-column block, i.e. Wb = 64) that is base = 4*j*64. If the hardware began
 * writing at HALF that pitch, real row r would sit at 2*r*64, so the unchanged de-tile reading
 * offset 4*j*64 would pick up real row 2*j. So the signature of "TP_PRECISION halves the write
 * pitch" is exact and checkable without touching the de-tile:
 *     C[g*H + j] == reference row (g*H + 2*j)   for every j < H/2,
 * with the upper half of each group left unwritten (still at the 0x7fff seed, or stale).
 * Anything else — output unchanged, or wrong without that structure — falsifies it.
 *
 * The POSITIVE CONTROL is not optional: a fuzz value that must change the output proves the
 * override reaches the hardware at all. Without it, "no change" proves nothing.
 *
 *   make i4_tp_precision_probe
 *   sudo tools/util/npu_guard.sh -- timeout -s TERM 300 env ORK_I4_NOLADDER=1 ORK_I4_H=32 \
 *        ORK_I4_WB=64 ORK_MM_TIMEOUT=400 ./i4_tp_precision_probe
 * Board only. Diagnostic; not in `all`/`test`. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"

#define FEATURE_MODE_BASE 0x000001e4u   /* REGCMD_I4's own 0x400c value (block 0x1001) */

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static unsigned s_=2024u; static int rnd4(void){ s_=s_*1103515245u+12345u; return -8+(int)((s_>>16)%16u); }
static int32_t sat16(long v){ return v>32767?32767:(v<-32768?-32768:(int32_t)v); }

int main(int argc,char**argv){
    int M = argc>1?atoi(argv[1]):64;
    int K = argc>2?atoi(argv[2]):512;
    int N = argc>3?atoi(argv[3]):64;
    int H = argc>4?atoi(argv[4]):32;     /* must match ORK_I4_H; the signature test needs it */

    setvbuf(stdout,NULL,_IOLBF,0);
    ork_npu *c=ork_npu_init(); if(!c){ printf("no board\n"); return 1; }
    printf("i4_tp_precision_probe  M=%d K=%d N=%d H=%d  (ork %s)\n",M,K,N,H,ork_npu_version());
    printf("  env: ORK_I4_H=%s ORK_I4_WB=%s ORK_I4_NOLADDER=%s  (Wb MUST be 64 so NBc==1)\n",
           getenv("ORK_I4_H")?:"-", getenv("ORK_I4_WB")?:"-", getenv("ORK_I4_NOLADDER")?:"-");

    int8_t *A=malloc((size_t)M*K), *B=malloc((size_t)K*N);
    int32_t *R=malloc((size_t)M*N*4), *C0=malloc((size_t)M*N*4), *Cx=malloc((size_t)M*N*4);
    if(!A||!B||!R||!C0||!Cx){ printf("OOM\n"); return 2; }
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)rnd4();
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)rnd4();
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ long acc=0;
        for(int k=0;k<K;k++) acc+=(long)A[(size_t)m*K+k]*B[(size_t)k*N+n];
        R[(size_t)m*N+n]=sat16(acc); }

    ork_w *W=ork_i4_mm_pack(c,K,N,B); if(!W){ printf("pack failed\n"); return 2; }

    /* baseline, unfuzzed */
    ork_i4_fuzz_clear();
    memset(C0,0,(size_t)M*N*4);
    double t0=now_us(); int rc0=ork_i4_mm_run(c,W,M,A,C0); double e0=now_us()-t0;
    long bad0=0; for(size_t i=0;i<(size_t)M*N;i++) if(C0[i]!=R[i]) bad0++;
    printf("  BASELINE           rc=%d %8.0f us  cells!=ref %ld/%d  %s\n",
           rc0,e0,bad0,M*N, (rc0==0&&bad0==0)?"EXACT":"*** baseline is not exact — stop here ***");
    if(rc0||bad0){ printf("  (the arms below are uninterpretable against a broken baseline)\n"); }

    /* TP_EN (0x400c bit 30) is deliberately LAST and run once: it commits a job that writes nothing,
     * so it costs a doorbell miss, a multi-second timeout and a recover round. It is also the arm that
     * proves the override mechanism is live — both arms patch block 0x1001 through the same orki_setr. */
    struct { const char *name; unsigned blk, reg, val; const char *why; } arm[] = {
        { "CTL 0x403c/2",  0x1001, 0x403c, (unsigned)(((N/2-1)<<16)|(N/2-1)), "MECHANISM CONTROL: halve DST_N_DIMS — must CHANGE the output" },
        { "0x4058 = 0",    0x1001, 0x4058, 0u,                            "SCOPE CONTROL: is the whole register inert on this path?" },
        { "TP_PRECISION",  0x1001, 0x4058, (unsigned)(N-1)|(1u<<27),      "0x4058 = (N-1) | bit27" },
        { "TP_PREC alone", 0x1001, 0x4058, (1u<<27),                      "bit27 with the width field cleared" },
        { "TP_EN",         0x1001, 0x400c, FEATURE_MODE_BASE|(1u<<30),    "0x400c |= bit30" },
    };
    for(int a=0;a<5;a++){
        ork_i4_fuzz_clear();
        ork_i4_fuzz_add(arm[a].blk,arm[a].reg,arm[a].val);
        memset(Cx,0,(size_t)M*N*4);
        double t1=now_us(); int rc=ork_i4_mm_run(c,W,M,A,Cx); double e1=now_us()-t1;
        long same=0,badr=0,sent=0;
        for(size_t i=0;i<(size_t)M*N;i++){ if(Cx[i]==C0[i]) same++; if(Cx[i]!=R[i]) badr++; if(Cx[i]==32767) sent++; }
        /* the halved-pitch signature: de-tiled row j of each group carries real row 2j */
        long hit=0,tot=0;
        for(int g=0; g*H<M; g++) for(int j=0;j<H/2;j++){ int lo=g*H+j, hi=g*H+2*j;
            if(lo>=M||hi>=M) continue;
            int eq=1; for(int n=0;n<N;n++) if(Cx[(size_t)lo*N+n]!=R[(size_t)hi*N+n]){ eq=0; break; }
            tot++; hit+=eq; }
        printf("  %-14s rc=%-3d %8.0f us  ==baseline %ld/%d  !=ref %ld  ==0x7fff %ld  half-pitch rows %ld/%ld   (%s)\n",
               arm[a].name,rc,e1,same,M*N,badr,sent,hit,tot,arm[a].why);
    }
    ork_i4_fuzz_clear();
    ork_mm_free(c,W); free(A);free(B);free(R);free(C0);free(Cx); ork_npu_free(c);
    return 0;
}
