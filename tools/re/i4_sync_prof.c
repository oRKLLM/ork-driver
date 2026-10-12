/* tools/re/i4_sync_prof.c — what sets the clock on an int4 BCHAIN prefill matmul, and what the
 * grow-only scratch buffers' WHOLE-BUFFER cache maintenance costs on a MULTI-SHAPE run.
 *
 * The four per-call scratch buffers (maf/mrc/mtk/mcc) only ever grow, so a whole-buffer orki_bsync
 * charges every later call maintenance over the WIDEST shape the process ever ran. A single-shape
 * microbenchmark cannot see this (high-water == live), which is why this probe interleaves several
 * shapes in ONE process, the way a model's layer does.
 *
 *   modes:
 *     seq    one shape only (high-water == live)        -> the clean per-shape baseline
 *     multi  all shapes interleaved in one process      -> the real high-water case
 *     dom    the same shapes split across N IOMMU domains -> domain-swap cost
 *
 * Data is FULL-RANDOM (a patterned RNG hides corruption from a checksum; see the Wb-frontier trap),
 * and every point is both checksummed against a per-shape reference AND timed, because the driver
 * self-heals a bad config into a correct-but-800x-slow answer.
 *
 *   make i4_sync_prof
 *   sudo tools/util/npu_guard.sh -- timeout -s TERM 300 env ORK_MM_TIMEOUT=2000 ./i4_sync_prof multi
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static unsigned long long fnv64(const void *p,size_t n){
    const unsigned char *b=p; unsigned long long h=1469598103934665603ULL;
    for(size_t i=0;i<n;i++){ h^=b[i]; h*=1099511628211ULL; } return h; }
/* full-random, fixed seed: patterned data has been measured to hide W4A4 corruption from a checksum */
static unsigned long long rs=0x243f6a8885a308d3ULL;
static inline unsigned rnd(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (unsigned)(rs>>32); }

#define NSHAPE 3
static const struct { int K,N; const char *name; } SH[NSHAPE] = {
    {2048,2048,"attn-o   K2048 N2048"},
    {2048,6144,"ffn-up   K2048 N6144"},
    {6144,2048,"ffn-down K6144 N2048"},
};

int main(int argc,char**argv){
    const char *mode = argc>1?argv[1]:"multi";
    int M = argc>2?atoi(argv[2]):128;
    int only = argc>3?atoi(argv[3]):-1;          /* seq mode: which shape */
    int iters = getenv("ORK_ITERS")?atoi(getenv("ORK_ITERS")):8;
    int skew  = getenv("ORK_SKEW")?atoi(getenv("ORK_SKEW")):0;   /* pre-run one WIDE shape to inflate the grow-only high-water, as a real model does */
    int ndom  = getenv("ORK_NDOM")?atoi(getenv("ORK_NDOM")):2;

    ork_npu *c = ork_npu_init(); if(!c){ printf("no board\n"); return 0; }
    if(!strcmp(mode,"dom")) ork_npu_set_ndomains(c,ndom);

    int8_t *A = malloc((size_t)M*8192);
    for(size_t i=0;i<(size_t)M*8192;i++) A[i]=(int8_t)(rnd()%15)-7;

    ork_w   *W[NSHAPE]={0}; int32_t *C[NSHAPE]={0}; unsigned long long ref[NSHAPE]={0};
    double  base[NSHAPE]={0};

    for(int s=0;s<NSHAPE;s++){
        if(!strcmp(mode,"seq") && only>=0 && s!=only) continue;
        int K=SH[s].K,N=SH[s].N;
        int8_t *B=malloc((size_t)K*N);
        for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)(rnd()%15)-7;
        int dom = !strcmp(mode,"dom") ? (s%ndom) : 0;
        ork_npu_set_pack_domain(c,dom);
        W[s] = ork_i4_mm_pack(c,K,N,B);
        free(B);
        if(!W[s]){ printf("pack fail shape %d\n",s); return 2; }
        C[s] = calloc((size_t)M*N,4);
    }

    printf("mode=%s M=%d iters=%d ndom=%d\n",mode,M,iters,!strcmp(mode,"dom")?ndom:1);

    /* ---- optional high-water skew: one wide shape, run once, then freed. It is never timed; its
     * only effect is to leave maf/mrc/mtk/mcc sized for a shape the timed loop never uses again. ---- */
    if(skew){
        int K=2048,N=18944; int8_t *B=malloc((size_t)K*N);
        for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)(rnd()%15)-7;
        ork_npu_set_pack_domain(c,0);
        ork_w *Wk=ork_i4_mm_pack(c,K,N,B); free(B);
        int32_t *Ck=calloc((size_t)M*N,4);
        if(Wk&&Ck){ ork_i4_mm_run(c,Wk,M,A,Ck); printf("skew: ran K=%d N=%d once (high-water now set by it)\n",K,N); }
        if(Wk) ork_mm_free(c,Wk); free(Ck);
    }

    /* ---- warm + reference, per shape, in the order a model would touch them ---- */
    for(int s=0;s<NSHAPE;s++){ if(!W[s]) continue;
        if(ork_i4_mm_run(c,W[s],M,A,C[s])){ printf("run fail shape %d\n",s); return 2; }
        ref[s]=fnv64(C[s],(size_t)M*SH[s].N*4); }

    /* second reference pass: catches a shape whose answer CHANGES once another shape has run
     * (a wrong live extent corrupts rather than slows, and it shows up exactly here) */
    for(int s=0;s<NSHAPE;s++){ if(!W[s]) continue;
        memset(C[s],0,(size_t)M*SH[s].N*4);
        ork_i4_mm_run(c,W[s],M,A,C[s]);
        if(fnv64(C[s],(size_t)M*SH[s].N*4)!=ref[s]) printf("  !! shape %d NOT reproducible across an interleave\n",s); }

    /* ---- timed interleave ---- */
    double tot=0; int bad=0;
    for(int it=0; it<iters; it++){
        for(int s=0;s<NSHAPE;s++){ if(!W[s]) continue;
            memset(C[s],0,(size_t)M*SH[s].N*4);
            double t0=now_us();
            int rc=ork_i4_mm_run(c,W[s],M,A,C[s]);
            double dt=now_us()-t0;
            if(rc){ printf("  rc=%d shape %d\n",rc,s); bad=1; continue; }
            if(fnv64(C[s],(size_t)M*SH[s].N*4)!=ref[s]){ printf("  WRONG shape %d iter %d\n",s,it); bad=1; }
            base[s]+=dt; tot+=dt; } }

    printf("\n%-22s %10s %12s  %s\n","shape","us/call","GB/s weight","fnv64 (compare ACROSS arms)");
    for(int s=0;s<NSHAPE;s++){ if(!W[s]) continue;
        double us=base[s]/iters;
        printf("%-22s %10.1f %12.2f  %016llx\n",SH[s].name,us,(double)SH[s].K*SH[s].N/2.0/us/1000.0,ref[s]); }
    printf("%-22s %10.1f  (one pass over every shape)\n","TOTAL",tot/iters);
    printf("checksum: %s\n", bad?"MISMATCH -- DO NOT TRUST THE TIMING":"all shapes bit-identical to reference");
    for(int s=0;s<NSHAPE;s++) if(W[s]) ork_mm_free(c,W[s]);
    ork_npu_free(c);
    return bad?1:0;
}
