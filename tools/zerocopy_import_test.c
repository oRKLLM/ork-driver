/* tools/zerocopy_import_test.c — validate + measure zero-copy weight IMPORT and a fixed-slot
 * streaming pool built on it. Board only (needs /dev/dri + /dev/dma_heap). NOT in `make test`.
 *
 * Piece 1: ork_dma_import as a matmul B (zero-copy, NPU reads user dma-buf pages directly) — vs CPU.
 *          ork_i8_mm_load_import round-trip vs ork_i8_mm_load vs CPU; load-time import vs alloc+copy.
 * Piece 2: fixed-size-slot streaming pool: cycle a working set LARGER than the IOVA window through a
 *          small set of uniform slots via import + MEM_DESTROY, asserting no fragmentation crash and
 *          correct output; per-swap cost vs MEM_CREATE+copy+DESTROY.
 *
 *   make zerocopy_import_test && sudo ./zerocopy_import_test
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static unsigned sd=999; static int r4(){ sd=sd*1103515245+12345; return (sd>>16)%4; }

/* CPU int8 reference C[M,N] = A[M,K] x B[K,N] */
static void ref_i8(int M,int K,int N,const int8_t*A,const int8_t*B,int32_t*C){
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int32_t s=0; for(int k=0;k<K;k++) s+=(int)A[(size_t)m*K+k]*(int)B[(size_t)k*N+n]; C[(size_t)m*N+n]=s; }
}

/* ---- Piece 1a: imported matmul B via ork_dma_import (raw, no ork_w) is out of scope here because the
 * weight path tiles B internally; instead we validate import end-to-end through ork_i8_mm_load_import,
 * which packs+imports the resident weight. That exercises the same primitive on the weight path. ---- */

static int test_load_import(ork_npu*ctx,int K,int N,int M){
    printf("\n[load-import] K=%d N=%d M=%d\n",K,N,M);
    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N);
    int32_t*Cref=malloc((size_t)M*N*4),*Cimp=malloc((size_t)M*N*4),*Cstd=malloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(int8_t)(r4()-1);
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=(int8_t)(r4()-1);
    ref_i8(M,K,N,A,B,Cref);

    /* pack normally to produce the canonical tile blob (ork_w_dump) */
    ork_w*wp=ork_i8_mm_pack(ctx,K,N,B);
    if(!wp){ printf("  pack_i8 failed\n"); return 1; }
    size_t blobsz=ork_w_dump(wp,NULL,0); void*blob=malloc(blobsz); ork_w_dump(wp,blob,blobsz);
    ork_mm_free(ctx,wp);

    /* standard alloc+copy load (baseline), timed */
    double t0=now_us(); ork_w*ws=ork_i8_mm_load(ctx,K,N,blob,blobsz); double t_std=now_us()-t0;
    if(!ws){ printf("  load_i8 failed\n"); return 1; }
    if(ork_i8_mm_run(ctx,ws,M,A,Cstd)){ printf("  run std failed\n"); return 1; }
    ork_mm_free(ctx,ws);

    /* zero-copy import load, timed */
    t0=now_us(); ork_w*wi=ork_i8_mm_load_import(ctx,K,N,blob,blobsz); double t_imp=now_us()-t0;
    if(!wi){ printf("  load_i8_import returned NULL (import unavailable?)\n"); free(blob); return 2; }
    if(ork_i8_mm_run(ctx,wi,M,A,Cimp)){ printf("  run import failed\n"); return 1; }
    ork_mm_free(ctx,wi);

    int bad_ref=0,bad_std=0;
    for(size_t i=0;i<(size_t)M*N;i++){ if(Cimp[i]!=Cref[i])bad_ref++; if(Cimp[i]!=Cstd[i])bad_std++; }
    printf("  vs CPU ref: %s (mism=%d)   vs alloc-load: %s (mism=%d)\n",
           bad_ref?"WRONG":"ok",bad_ref, bad_std?"DIFFER":"identical",bad_std);
    printf("  load time: alloc+copy %.0f us   import %.0f us   (%.2fx)\n",
           t_std,t_imp, t_std/(t_imp>0?t_imp:1));
    free(A);free(B);free(Cref);free(Cimp);free(Cstd);free(blob);
    return bad_ref?1:0;
}

/* ---- Piece 2: fixed-slot streaming pool. Reserve NSLOT uniform slots. A "working set" of NEXP
 * distinct weights (each (K,N), pre-dumped to a blob) is cycled through the slots: load -> run ->
 * free, round-robin, ITERS times. Asserts every output is correct and no import ever fails after a
 * free (the fragmentation symptom). Per-swap cost = (load_import + free) / swap. Compared to the
 * alloc+copy churn (load_i8 + free) which is what fragmented before. ---- */
static int test_fixed_slot_stream(ork_npu*ctx,int K,int N,int NEXP,int ITERS,int use_import){
    printf("\n[stream %s] K=%d N=%d experts=%d iters=%d  (working set = %.1f MB)\n",
           use_import?"IMPORT":"alloc+copy", K,N,NEXP, (double)NEXP*K*N/1e6);
    int M=1;
    int8_t*A=malloc((size_t)M*K);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(int8_t)(r4()-1);
    /* build per-expert weight + dumped blob + CPU ref */
    int8_t**B=malloc(NEXP*sizeof*B); void**blob=malloc(NEXP*sizeof*blob); size_t blobsz=0;
    int32_t**Cref=malloc(NEXP*sizeof*Cref);
    for(int e=0;e<NEXP;e++){
        B[e]=malloc((size_t)K*N); for(size_t i=0;i<(size_t)K*N;i++)B[e][i]=(int8_t)(r4()-1);
        ork_w*wp=ork_i8_mm_pack(ctx,K,N,B[e]);
        if(!wp){ printf("  pack expert %d failed\n",e); return 1; }
        if(!blobsz) blobsz=ork_w_dump(wp,NULL,0);
        blob[e]=malloc(blobsz); ork_w_dump(wp,blob[e],blobsz);
        ork_mm_free(ctx,wp);
        Cref[e]=malloc((size_t)M*N*4); ref_i8(M,K,N,A,B[e],Cref[e]);
    }
    int bad=0,fail=0; double t_load=0,t_free=0; int swaps=0;
    int32_t*C=malloc((size_t)M*N*4);
    double t0_all=now_us();
    for(int it=0;it<ITERS;it++){
        for(int e=0;e<NEXP;e++){
            double t0=now_us();
            ork_w*w = use_import ? ork_i8_mm_load_import(ctx,K,N,blob[e],blobsz)
                                 : ork_i8_mm_load(ctx,K,N,blob[e],blobsz);
            t_load+=now_us()-t0;
            if(!w){ fail++; printf("  !! load failed it=%d e=%d (FRAGMENTATION/OOM)\n",it,e); continue; }
            if(ork_i8_mm_run(ctx,w,M,A,C)){ printf("  run failed it=%d e=%d\n",it,e); fail++; }
            else { for(int n=0;n<N;n++) if(C[n]!=Cref[e][n]){bad++;break;} }
            t0=now_us(); ork_mm_free(ctx,w); t_free+=now_us()-t0;
            swaps++;
        }
    }
    double t_all=now_us()-t0_all;
    printf("  swaps=%d  load-fails=%d  output-mism=%d\n",swaps,fail,bad);
    printf("  per-swap: load %.0f us  free %.0f us  total wall %.0f us/swap\n",
           t_load/swaps, t_free/swaps, t_all/swaps);
    for(int e=0;e<NEXP;e++){ free(B[e]); free(blob[e]); free(Cref[e]); }
    free(B);free(blob);free(Cref);free(A);free(C);
    return (bad||fail)?1:0;
}

int main(int argc,char**argv){
    ork_npu*ctx=ork_npu_init(); if(!ctx){ printf("init failed\n"); return 77; }
    printf("ork %s  soc=%s cores=%d\n",ork_npu_version(),ork_npu_soc(ctx),ork_npu_cores(ctx));

    /* quick standalone import smoke: alloc a dma-buf, fill, free */
    void*p=ork_dma_import(ctx,1<<20);
    if(!p){ printf("ork_dma_import smoke: NULL (dma-heap unavailable) — import tests will skip\n"); }
    else { memset(p,0xAB,1<<20); ork_dma_import_sync(ctx,p,1<<20); printf("ork_dma_import smoke: ok (1MB)\n"); ork_dma_import_free(ctx,p); }

    int rc=0;
    rc |= test_load_import(ctx,512,256,8);
    rc |= test_load_import(ctx,2048,512,4);   /* in Bf full-K envelope (K<=4096, K%512) */
    rc |= test_load_import(ctx,4096,1024,2);

    /* #36 REPRO: imported-Bf through the M>1 colsplit at N=3584 (7B shapes), vs CPU. The direct-JIT
     * (natively-packed Bf) path is already CPU-verified correct for these; this isolates whether the
     * IMPORTED-Bf layout mis-dispatches the non-even 3-core N=3584 colsplit. */
    rc |= test_load_import(ctx,3584,3584,228);    /* base q/o_proj: colsplit N=3584 across 3 cores */
    rc |= test_load_import(ctx,3584,3584,32);     /* same, small M */
    rc |= test_load_import(ctx,18944,3584,64);    /* ffn_down wide-K, N=3584 */
    if (getenv("SKIP_STREAM")) { printf("\n%s (load-import only)\n", rc?"FAIL":"ALL OK"); ork_npu_free(ctx); return rc; }

    /* fixed-slot streaming: working set bigger than what fits at once would normally fragment.
     * default: many experts cycled. Sizes chosen so the total working set is multi-GB if held, but
     * the pool only holds 1 at a time. */
    int K=4096,N=4096;                 /* one expert = 16 MB int8 */
    int NEXP = argc>1?atoi(argv[1]):64; /* 64 * 16MB = 1 GB cycled; bump to exceed 4GB total */
    int ITERS= argc>2?atoi(argv[2]):4;
    rc |= test_fixed_slot_stream(ctx,K,N,NEXP,ITERS,1);   /* IMPORT */
    rc |= test_fixed_slot_stream(ctx,K,N,NEXP,ITERS,0);   /* alloc+copy baseline */

    printf("\n%s\n", rc?"FAIL":"ALL OK");
    ork_npu_free(ctx);
    return rc;
}
