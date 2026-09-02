/* examples/test_matmul.c — validates the ork_npu library: one handle, resident weights reused across
 * many matmuls of varying M (the forward-pass access pattern). Builds vs CPU reference.
 *   cc -O2 -I. -o test_mm test_mm.c ork_npu.c && sudo ./test_mm */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "ork_npu.h"
typedef ork_f16 f16;
static unsigned sd=12345; static int rnd(){sd=sd*1103515245+12345;return (sd>>16)%4;}
/* FNV-1a 64-bit checksum. rnd() is a fixed-seed LCG, so the inputs — and therefore the NPU output —
 * are DETERMINISTIC across runs; a correctness check need only compare the O(M*N) output checksum
 * against a static golden, NOT recompute the O(M*N*K) CPU reference every run. The reference is kept
 * (below) and runs ONLY to regenerate a golden (ORK_REGEN=1) or diagnose a mismatch (ORK_FULL_REF=1). */
static uint64_t fnv64u(uint64_t h,const void*p,size_t n){ const uint8_t*b=(const uint8_t*)p;
    for(size_t i=0;i<n;i++){ h^=b[i]; h*=1099511628211ULL; } return h; }
static uint64_t fnv64(const void*p,size_t n){ return fnv64u(1469598103934665603ULL,p,n); }

static int check(ork_npu*ctx,int M,int K,int N,uint64_t gold){
    printf("FP16 check start: M=%d, K=%d, N=%d\n", M, K, N); fflush(stdout);
    f16*A=malloc((size_t)M*K*2),*B=malloc((size_t)K*N*2); float*C=malloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(f16)rnd();
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=(f16)rnd();
    ork_w*w=ork_f16_mm_pack(ctx,K,N,B);
    if(!w){printf("pack failed %d,%d\n",K,N);free(A);free(B);free(C);return 1;}
    /* run the SAME resident weights for several M (decode then prefill); hash each run's output */
    int Ms[]={1,1,4,M}; uint64_t got=1469598103934665603ULL;
    for(int t=0;t<4;t++){int m=Ms[t]; if(m>M)m=M;
        if(ork_f16_mm_run(ctx,w,m,A,C)){printf("run failed\n");ork_w_free(w);free(A);free(B);free(C);return 1;}
        got=fnv64u(got,C,(size_t)m*N*4); }
    int bad=0,ret,regen=getenv("ORK_REGEN")!=NULL;
    if(gold && got==gold && !regen && !getenv("ORK_FULL_REF")){
        printf("  ok   MKN=%d,%d,%d fp16 (golden 0x%016llx)\n",M,K,N,(unsigned long long)got); ret=0;   /* fast: no O(M*N*K) ref */
    } else {   /* preserved fp32 reference (re-run the M-tiles — NPU cheap; the CPU ref is the cost) */
        for(int t=0;t<4;t++){int m=Ms[t]; if(m>M)m=M; ork_f16_mm_run(ctx,w,m,A,C);
            for(int i=0;i<m;i++)for(int n=0;n<N;n++){float ref=0;for(int k=0;k<K;k++)ref+=(float)A[(size_t)i*K+k]*(float)B[(size_t)k*N+n]; if(C[(size_t)i*N+n]!=ref)bad++;} }
        if(regen||!gold) printf("  REGEN check GOLD {%d,%d,%d} = 0x%016llxULL (mism=%d)\n",M,K,N,(unsigned long long)got,bad);
        else if(got!=gold) printf("  GOLDEN MISMATCH fp16 {%d,%d,%d} (mism=%d) — regen if intended\n",M,K,N,bad);
        printf("  %s MKN=%d,%d,%d (reused weights x4 runs) mism=%d\n",bad?"WRONG":"ok  ",M,K,N,bad); fflush(stdout);
        ret=(bad||(gold&&got!=gold&&!regen))?1:0;
    }
    ork_w_free(w); free(A);free(B);free(C); return ret;
}
/* int8/w8a8: A,B int8 -> C int32 (exact integer reference). K%32, N%32. */
static int check_i8(ork_npu*ctx,int M,int K,int N,uint64_t gold){
    printf("Int8 check start: M=%d, K=%d, N=%d\n", M, K, N); fflush(stdout);
    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t*C=malloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(int8_t)(rnd()-1);
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=(int8_t)(rnd()-1);
    ork_w*w=ork_i8_mm_pack(ctx,K,N,B);
    if(!w){printf("pack_i8 failed %d,%d\n",K,N);free(A);free(B);free(C);return 1;}
    int Ms[]={1,1,4,M}; uint64_t got=1469598103934665603ULL;
    for(int t=0;t<4;t++){int m=Ms[t]; if(m>M)m=M;
        if(ork_i8_mm_run(ctx,w,m,A,C)){printf("run_i8 failed\n");ork_w_free(w);free(A);free(B);free(C);return 1;}
        got=fnv64u(got,C,(size_t)m*N*4); }
    int bad=0,ret,regen=getenv("ORK_REGEN")!=NULL;
    if(gold && got==gold && !regen && !getenv("ORK_FULL_REF")){
        printf("  ok   MKN=%d,%d,%d int8 (golden 0x%016llx)\n",M,K,N,(unsigned long long)got); ret=0;   /* fast: no O(M*N*K) ref */
    } else {   /* preserved exact int32 reference */
        for(int t=0;t<4;t++){int m=Ms[t]; if(m>M)m=M; ork_i8_mm_run(ctx,w,m,A,C);
            for(int i=0;i<m;i++)for(int n=0;n<N;n++){int32_t ref=0;for(int k=0;k<K;k++)ref+=(int)A[(size_t)i*K+k]*(int)B[(size_t)k*N+n]; if(C[(size_t)i*N+n]!=ref)bad++;} }
        if(regen||!gold) printf("  REGEN check_i8 GOLD {%d,%d,%d} = 0x%016llxULL (mism=%d)\n",M,K,N,(unsigned long long)got,bad);
        else if(got!=gold) printf("  GOLDEN MISMATCH int8 {%d,%d,%d} (mism=%d) — regen if intended\n",M,K,N,bad);
        printf("  %s MKN=%d,%d,%d int8 (reused weights x4 runs) mism=%d\n",bad?"WRONG":"ok  ",M,K,N,bad); fflush(stdout);
        ret=(bad||(gold&&got!=gold&&!regen))?1:0;
    }
    ork_w_free(w); free(A);free(B);free(C); return ret;
}
/* CHAIN-PREFILL correctness: exercise the int8 M>1 full-K prefill multi-core path with shapes that
 * force BOTH multiple M-tiles (M >> chunk) AND multiple N-tiles across cores (N spanning several
 * N-tiles so the auto-tuner picks nc>1). Validates the exact integer product. Run the binary with
 * ORK_CHAIN_PREFILL=1 (default, chained) and =0 (per-tile) — both MUST match this exact reference. */
static int check_chain_prefill(ork_npu*ctx){
    /* {256,18944,3584} is the wide-K ffn_down shape that exercises CHAIN-KSPLIT (K>4096, K-split
     * passes PC-chained into one submit/core then host-accumulated). MUST match the CPU ref both
     * with ORK_CHAIN_KSPLIT=1 (chained) and =0 (per-tile K-split). */
    int shapes[][3] = { {256,18944,3584}, {256,3584,3584}, {256,2048,2048}, {128,512,1536}, {200,1024,2048},
        {128,2048,16384}, {96,3584,18944} };   /* last two: WIDE-N int8 (N>nmax=8192 => Sn=2, Sn=3) — exercise r_wideN (default, Bf) AND the per-N-slice K-split (ORK_NO_BF, no Bf); ref-checked vs CPU (#48) */
    int ns = (int)(sizeof(shapes)/sizeof(shapes[0]));
    /* Static golden output checksums (fnv64 of the int32 C) for the fixed-seed inputs above. Regenerate
     * with `ORK_REGEN=1 sudo ./test_matmul` and paste the printed values. A 0 entry auto-triggers regen. */
    static const uint64_t GOLD[] = {  /* regenerated 2026-07-14 (RK3588); ref-verified bad=0. Last two (2026-08-06): WIDE-N int8 (Sn=2,Sn=3), default Bf path (r_wideN), ref-verified bad=0. */
        0xb4d325e573e7037cULL, 0x6084624fae4ea1cdULL, 0x1fc8a45628241e4bULL, 0xa752ea8100ba7d54ULL, 0x21dc86bea3c1015fULL,
        0x5b98ab74d6e8947aULL, 0x3fcd5ced1fed0f53ULL,
    };
    int fail=0; unsigned sd_save=sd;
    for(int s=0;s<ns;s++){
        if(s==5) sd_save=sd;   /* capture the shared fixed-seed PRNG position just before the appended WIDE-N shapes (idx 5,6) — restored after the loop so DOWNSTREAM golden'd tests (check_i8, ...) see the same rnd() inputs they were baked with */
        int M=shapes[s][0],K=shapes[s][1],N=shapes[s][2];
        printf("ChainPrefill: M=%d K=%d N=%d\n",M,K,N); fflush(stdout);
        int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t*C=malloc((size_t)M*N*4);
        for(size_t i=0;i<(size_t)M*K;i++)A[i]=(int8_t)(rnd()-1);
        for(size_t i=0;i<(size_t)K*N;i++)B[i]=(int8_t)(rnd()-1);
        ork_w*w=ork_i8_mm_pack(ctx,K,N,B);
        if(!w){printf("  pack_i8 failed\n");free(A);free(B);free(C);return 1;}
        if(ork_i8_mm_run(ctx,w,M,A,C)){printf("  run_i8 failed\n");ork_w_free(w);free(A);free(B);free(C);return 1;}
        uint64_t got=fnv64(C,(size_t)M*N*4);
        uint64_t gold = s<(int)(sizeof GOLD/sizeof*GOLD) ? GOLD[s] : 0;
        int regen=getenv("ORK_REGEN")!=NULL;
        if(gold && got==gold && !regen && !getenv("ORK_FULL_REF")){
            printf("  ok (golden 0x%016llx)\n",(unsigned long long)got); fflush(stdout);   /* fast: no O(M*N*K) recompute */
        } else {
            long bad=0;   /* preserved CPU reference: regen a golden or diagnose a mismatch */
            for(int i=0;i<M && bad<5;i++)for(int n=0;n<N;n++){int32_t ref=0;for(int k=0;k<K;k++)ref+=(int)A[(size_t)i*K+k]*(int)B[(size_t)k*N+n]; if(C[(size_t)i*N+n]!=ref){bad++;if(bad<=3)printf("    mism @ (%d,%d): got %d ref %d\n",i,n,C[(size_t)i*N+n],ref);}}
            if(regen||!gold) printf("  REGEN GOLD[%d]=0x%016llxULL  {%d,%d,%d} ref-bad=%ld\n",s,(unsigned long long)got,M,K,N,bad);
            else if(got!=gold) printf("  GOLDEN MISMATCH {%d,%d,%d}: output changed (ref-bad=%ld) — regen if intended\n",M,K,N,bad);
            if(bad || (gold && got!=gold && !regen)) fail=1;
            printf("  %s (ref-checked)\n",bad?"WRONG":"ok"); fflush(stdout);
        }
        ork_w_free(w);free(A);free(B);free(C);
    }
    if(ns>5) sd=sd_save;   /* the WIDE-N shapes consumed rnd(); undo that so later tests' fixed-seed inputs (and their static goldens) are unaffected by adding these shapes */
    return fail;
}
static int check_chain_i8(ork_npu*ctx) {
    int S = 4;
    int Ms[4] = {1, 1, 1, 1};
    int K = 512;        /* in the chain's full-K envelope (K%512==0, K<=4096) */
    int N = 64;
    
    int8_t *A[4] = {NULL};
    int8_t *B[4] = {NULL};
    int32_t *C[4] = {NULL};
    ork_w *w[4] = {NULL};
    ork_mm_task_i8 tasks[4];

    for (int i = 0; i < S; i++) {
        /* plain host buffers — DMA-resident A/C in a chain hits the ZC-output cold-coherency issue
         * (intermittent stale output), which is tracked separately (wiki Tier 7); keep this correctness
         * test deterministic. */
        A[i] = malloc((size_t)Ms[i] * K);
        C[i] = malloc((size_t)Ms[i] * N * 4);
        B[i] = malloc((size_t)K * N);
        
        for (size_t j = 0; j < (size_t)Ms[i] * K; j++) A[i][j] = (int8_t)(rnd() - 1);
        for (size_t j = 0; j < (size_t)K * N; j++) B[i][j] = (int8_t)(rnd() - 1);
        
        w[i] = ork_i8_mm_pack(ctx, K, N, B[i]);
        if (!w[i]) { printf("pack_chain_i8 failed %d\n", i); return 1; }
        
        tasks[i].w = w[i];
        tasks[i].M = Ms[i];
        tasks[i].A = A[i];
        tasks[i].C = C[i];
    }
    
    int bad = 0;
    printf("Running ork_i8_mm_run_chain...\n");
    if (ork_i8_mm_run_chain(ctx, S, tasks)) {
        printf("run_chain_i8 failed\n");
        bad = 1;
    } else {
        for (int i = 0; i < S; i++) {
            int m = Ms[i];
            int task_bad = 0;
            for (int r = 0; r < m; r++) {
                for (int n = 0; n < N; n++) {
                    int32_t ref = 0;
                    for (int k = 0; k < K; k++) {
                        ref += (int)A[i][(size_t)r * K + k] * (int)B[i][(size_t)k * N + n];
                    }
                    if (C[i][(size_t)r * N + n] != ref) {
                        if (task_bad < 3) printf("  mism at task %d, row %d, col %d: expected %d, got %d\n", i, r, n, ref, C[i][(size_t)r * N + n]);
                        bad++;
                        task_bad++;
                    }
                }
            }
            if (task_bad) printf("  task %d had %d mismatches (M=%d)\n", i, task_bad, m);
        }
    }

    printf("  %s chained S=%d (varying M, independent matmuls) mism=%d\n", bad ? "WRONG" : "ok  ", S, bad);

    for (int i = 0; i < S; i++) {
        if (w[i]) ork_w_free(w[i]);
        free(A[i]); free(C[i]); free(B[i]);
    }
    return bad ? 1 : 0;
}

/* Chain experts whose K forces a 2-slice pack (Sk=2 at K=2048) but which carry a Bf full-K buffer.
 * This is the MoE-prefill chaining path: the chain uses Bf so each expert is one PC-chained submit,
 * even though pack_i8 K-splits at 1024. Varies M (decode M=1 + small prefill M>1, all <= the
 * single-submit row cap). Different weights per expert; validate each vs the int32 CPU reference. */
static int check_chain_i8_bf(ork_npu *ctx) {
    enum { S = 4 };
    int Ms[S] = {1, 8, 40, 96};        // 1/8 single-submit; 40/96 force M-tiling (mcap=31 at K=2048)
    int K = 2048, N = 768;             // K=2048 -> Sk=2 + Bf; N=768 -> Sn=1 (N<=nmax)
    int8_t *A[S] = {0}, *B[S] = {0}; int32_t *C[S] = {0}; ork_w *w[S] = {0};
    ork_mm_task_i8 tasks[S];
    for (int i = 0; i < S; i++) {
        A[i] = malloc((size_t)Ms[i] * K); B[i] = malloc((size_t)K * N); C[i] = malloc((size_t)Ms[i] * N * 4);
        for (size_t j = 0; j < (size_t)Ms[i] * K; j++) A[i][j] = (int8_t)(rnd() - 1);
        for (size_t j = 0; j < (size_t)K * N; j++) B[i][j] = (int8_t)(rnd() - 1);
        w[i] = ork_i8_mm_pack(ctx, K, N, B[i]);
        if (!w[i]) { printf("pack_chain_i8_bf failed %d\n", i); return 1; }
        tasks[i].w = w[i]; tasks[i].M = Ms[i]; tasks[i].A = A[i]; tasks[i].C = C[i];
    }
    int bad = 0, rc = ork_i8_mm_run_chain(ctx, S, tasks);
    if (rc) { printf("run_chain_i8 (Bf) failed rc=%d\n", rc); bad = 1; }
    else for (int i = 0; i < S; i++) for (int r = 0; r < Ms[i]; r++) for (int n = 0; n < N; n++) {
        int32_t ref = 0; for (int k = 0; k < K; k++) ref += (int)A[i][(size_t)r*K+k] * (int)B[i][(size_t)k*N+n];
        if (C[i][(size_t)r*N+n] != ref) { if (bad < 3) printf("  Bf mism task %d row %d col %d: exp %d got %d\n", i, r, n, ref, C[i][(size_t)r*N+n]); bad++; }
    }
    printf("  %s chained S=%d Bf K=%d N=%d (Sk=2, M=1/8/40/96 incl. M-tiling) mism=%d\n", bad ? "WRONG" : "ok  ", S, K, N, bad);
    for (int i = 0; i < S; i++) { if (w[i]) ork_w_free(w[i]); free(A[i]); free(B[i]); free(C[i]); }
    return bad ? 1 : 0;
}

/* Async round-robin stream: S independent matmuls dispatched dynamically across cores. Different weights,
 * mixed K/N/M, validated EVERY output vs the int32 CPU reference (the stream path is new). */
static int check_stream_i8(ork_npu *ctx) {
    enum { S = 6 };
    int Ks[S] = {2048, 512, 4096, 1536, 1024, 2048};  /* full-K Bf envelope: K%512==0, K<=4096 */
    int Ns[S] = {512, 256, 2048, 512, 512, 768};
    int Ms[S] = {64, 8, 1, 40, 128, 32};
    int8_t *A[S]={0}, *B[S]={0}; int32_t *C[S]={0}; ork_w *w[S]={0};
    ork_mm_task_i8 tasks[S];
    for (int i = 0; i < S; i++) {
        int K=Ks[i], N=Ns[i], M=Ms[i];
        A[i]=malloc((size_t)M*K); B[i]=malloc((size_t)K*N); C[i]=malloc((size_t)M*N*4);
        for (size_t j=0;j<(size_t)M*K;j++) A[i][j]=(int8_t)(rnd()-1);
        for (size_t j=0;j<(size_t)K*N;j++) B[i][j]=(int8_t)(rnd()-1);
        w[i]=ork_i8_mm_pack(ctx,K,N,B[i]); if(!w[i]){printf("pack_stream failed %d\n",i);return 1;}
        tasks[i].w=w[i]; tasks[i].M=M; tasks[i].A=A[i]; tasks[i].C=C[i];
    }
    int bad=0, rc=ork_i8_mm_run_stream(ctx,S,tasks);
    if(rc){printf("run_stream_i8 failed rc=%d\n",rc); bad=1;}
    else for(int i=0;i<S;i++) for(int r=0;r<Ms[i];r++) for(int n=0;n<Ns[i];n++){
        int32_t ref=0; for(int k=0;k<Ks[i];k++) ref+=(int)A[i][(size_t)r*Ks[i]+k]*(int)B[i][(size_t)k*Ns[i]+n];
        if(C[i][(size_t)r*Ns[i]+n]!=ref){ if(bad<3) printf("  stream mism task %d r%d c%d: exp %d got %d\n",i,r,n,ref,C[i][(size_t)r*Ns[i]+n]); bad++; }
    }
    printf("  %s stream S=%d (mixed K/N/M, async round-robin) mism=%d\n", bad?"WRONG":"ok  ", S, bad);
    for(int i=0;i<S;i++){ if(w[i])ork_w_free(w[i]); free(A[i]); free(B[i]); free(C[i]); }
    return bad?1:0;
}

/* Regression for the full-K Bf schedule envelope (K%512==0 && K<=4096). A full-K single submit with
 * sched=1 silently mis-computes for out-of-envelope K (e.g. 768) when M>1 — so run_chain_i8 /
 * run_stream_i8 must REJECT such tasks (rc=-3) instead of returning wrong output. Also confirms an
 * in-envelope K next to it still computes correctly (so the guard isn't over-broad). */
static int check_chain_envelope(ork_npu *ctx) {
    int bad = 0;
    /* (a) out-of-envelope K=768, M>1 must be rejected by BOTH chain and stream (not computed wrong) */
    { int K = 768, N = 512, M = 8;
      int8_t *B = malloc((size_t)K*N), *A = malloc((size_t)M*K); int32_t *C = malloc((size_t)M*N*4);
      for (size_t j=0;j<(size_t)K*N;j++) B[j]=(int8_t)(rnd()-1);
      for (size_t j=0;j<(size_t)M*K;j++) A[j]=(int8_t)(rnd()-1);
      ork_w *w = ork_i8_mm_pack(ctx,K,N,B);
      ork_mm_task_i8 t = { w, M, A, C };
      int rc_chain = w ? ork_i8_mm_run_chain(ctx, 1, &t) : 0;   /* S=1 still goes through validation... */
      ork_mm_task_i8 ts[2] = { {w,M,A,C}, {w,M,A,C} };          /* S=2 to exercise the chain path proper */
      int rc_chain2 = w ? ork_i8_mm_run_chain(ctx, 2, ts) : 0;
      int rc_stream = w ? ork_i8_mm_run_stream(ctx, 2, ts) : 0;
      /* chain S=1 delegates to run_i8 (handles K-split) so it may succeed; the chain proper (S>=2) and
       * the stream must reject K=768 M>1 with -3. */
      if (rc_chain2 != -3) { printf("  ENVELOPE: run_chain_i8 K=768 M=8 expected -3, got %d\n", rc_chain2); bad++; }
      if (rc_stream != -3) { printf("  ENVELOPE: run_stream_i8 K=768 M=8 expected -3, got %d\n", rc_stream); bad++; }
      (void)rc_chain;
      if (w) ork_w_free(w); free(A); free(B); free(C); }
    /* (b) in-envelope K=1536 (Sk=2, %512, <=4096), M>1 chained must be CORRECT vs CPU */
    { int K = 1536, N = 256, M = 33;
      int8_t *B = malloc((size_t)K*N), *A = malloc((size_t)M*K); int32_t *C = malloc((size_t)M*N*4);
      for (size_t j=0;j<(size_t)K*N;j++) B[j]=(int8_t)(rnd()-1);
      for (size_t j=0;j<(size_t)M*K;j++) A[j]=(int8_t)(rnd()-1);
      ork_w *w = ork_i8_mm_pack(ctx,K,N,B);
      ork_mm_task_i8 ts[2] = { {w,M,A,C}, {w,M,A,C} };
      int rc = w ? ork_i8_mm_run_chain(ctx, 2, ts) : -1;
      if (rc) { printf("  ENVELOPE: in-envelope K=1536 chain failed rc=%d\n", rc); bad++; }
      else for (int r=0;r<M && bad<6;r++) for (int n=0;n<N;n++) {
          int32_t ref=0; for(int k=0;k<K;k++) ref+=(int)A[(size_t)r*K+k]*(int)B[(size_t)k*N+n];
          if (C[(size_t)r*N+n]!=ref) { printf("  ENVELOPE: K=1536 mism r%d c%d exp %d got %d\n",r,n,ref,C[(size_t)r*N+n]); bad++; } }
      if (w) ork_w_free(w); free(A); free(B); free(C); }
    printf("  %s chain/stream full-K envelope guard (reject K%%512!=0; K=1536 correct)\n", bad?"WRONG":"ok  ");
    return bad ? 1 : 0;
}

/* Validate the NEON f32->int8 pack (ork_i8_mm_pack_f32): pack f32 weights, run_i8, dequant with the
 * returned per-channel bscale, compare to the f32 CPU reference (within int8 W8A8 tolerance). */
static int check_pack_i8_f32(ork_npu *ctx) {
    int M = 8, K = 2048, N = 512;   /* K=2048 -> Sk=2 + Bf, exercises the full-K tile path too */
    float *Bf = malloc((size_t)N*K*sizeof(float)), *Af = malloc((size_t)M*K*sizeof(float)), *bsc = malloc((size_t)N*sizeof(float));
    for (size_t j=0;j<(size_t)N*K;j++) Bf[j] = ((int)rnd()-128)/64.0f;     /* [N][K] n-major */
    for (size_t j=0;j<(size_t)M*K;j++) Af[j] = ((int)rnd()-128)/64.0f;
    ork_w *w = ork_i8_mm_pack_f32(ctx, K, N, Bf, bsc);
    if (!w) { printf("  pack_i8_f32 failed\n"); free(Bf);free(Af);free(bsc); return 1; }
    int8_t *Ai = malloc((size_t)M*K); float *asc = malloc((size_t)M*sizeof(float)); int32_t *Ci = malloc((size_t)M*N*4);
    for (int m=0;m<M;m++){ float mx=1e-9f; for(int k=0;k<K;k++){float v=fabsf(Af[(size_t)m*K+k]); if(v>mx)mx=v;}
        asc[m]=mx/127.0f; float iv=127.0f/mx; for(int k=0;k<K;k++){int q=(int)lrintf(Af[(size_t)m*K+k]*iv); Ai[(size_t)m*K+k]=(int8_t)(q>127?127:q<-127?-127:q);} }
    int rc = ork_i8_mm_run(ctx, w, M, Ai, Ci);
    double se=0, sref=0;
    if (!rc) for (int m=0;m<M;m++) for (int n=0;n<N;n++){
        double ref=0; for(int k=0;k<K;k++) ref += (double)Af[(size_t)m*K+k]*Bf[(size_t)n*K+k];
        double got = (double)asc[m]*bsc[n]*Ci[(size_t)m*N+n];
        se += (got-ref)*(got-ref); sref += ref*ref; }
    double rms = sqrt(se/((double)M*N)) / (sqrt(sref/((double)M*N))+1e-9);
    int ok = (!rc) && rms < 0.03;
    printf("  %s pack_i8_f32 (NEON f32->int8 quant+tile) M=%d K=%d N=%d  rc=%d RMS rel err=%.3f%%\n", ok?"ok  ":"WRONG", M,K,N, rc, rms*100);
    ork_w_free(w); free(Bf);free(Af);free(bsc);free(Ai);free(asc);free(Ci);
    return ok?0:1;
}

/* Validate the "effective w4a8" pack (ork_i4a8_mm_pack): int4-precision weights, int8 compute, int4
 * storage. Packs f32 weights to int4 (round-to-nearest, ORK_SR unset for determinism), runs run_i8,
 * dequants with the returned per-channel bscale, and compares to a CPU reference that computes the SAME
 * int4-quantized weights (dequantized by bscale) x the int8 activations. This isolates COMPUTE
 * correctness given int4 weights (not int4-vs-fp32 accuracy). Tolerance: int8-activation W8 quant noise. */
static int check_pack_i4a8(ork_npu *ctx) {
    int M = 8, K = 2048, N = 512;   /* K=2048 -> Sk=2 + Bf, exercises the full-K tile path too */
    float *Bf = malloc((size_t)N*K*sizeof(float)), *Af = malloc((size_t)M*K*sizeof(float)), *bsc = malloc((size_t)N*sizeof(float));
    for (size_t j=0;j<(size_t)N*K;j++) Bf[j] = ((int)rnd()-128)/64.0f;     /* [N][K] n-major */
    for (size_t j=0;j<(size_t)M*K;j++) Af[j] = ((int)rnd()-128)/64.0f;
    ork_w *w = ork_i4a8_mm_pack(ctx, K, N, Bf, bsc);
    if (!w) { printf("  pack_i4a8 failed\n"); free(Bf);free(Af);free(bsc); return 1; }
    /* int8-quantize A (per-row symmetric, same as the W8A8 path) */
    int8_t *Ai = malloc((size_t)M*K); float *asc = malloc((size_t)M*sizeof(float)); int32_t *Ci = malloc((size_t)M*N*4);
    for (int m=0;m<M;m++){ float mx=1e-9f; for(int k=0;k<K;k++){float v=fabsf(Af[(size_t)m*K+k]); if(v>mx)mx=v;}
        asc[m]=mx/127.0f; float iv=127.0f/mx; for(int k=0;k<K;k++){int q=(int)lrintf(Af[(size_t)m*K+k]*iv); Ai[(size_t)m*K+k]=(int8_t)(q>127?127:q<-127?-127:q);} }
    int rc = ork_i8_mm_run(ctx, w, M, Ai, Ci);
    /* CPU reference: re-quantize the weights to int4 the SAME way (scale=max/7, RN, clamp [-7,7]),
     * dequant by bsc[n], and dot against the dequantized int8 activations. */
    double se=0, sref=0;
    if (!rc) for (int n=0;n<N;n++){
        const float *wr = Bf + (size_t)n*K; float bs = bsc[n], biv = bs>0?1.0f/bs:0.0f;
        for (int m=0;m<M;m++){
            double ref=0;
            for (int k=0;k<K;k++){ int q=(int)lrintf(wr[k]*biv); if(q>7)q=7; else if(q<-7)q=-7;
                double wdq = (double)q*bs; double adq = (double)Ai[(size_t)m*K+k]*asc[m];
                ref += adq*wdq; }
            double got = (double)asc[m]*bs*Ci[(size_t)m*N+n];
            se += (got-ref)*(got-ref); sref += ref*ref; }
    }
    double rms = sqrt(se/((double)M*N)) / (sqrt(sref/((double)M*N))+1e-9);
    int ok = (!rc) && rms < 0.03;
    printf("  %s pack_i4a8 (int4 wt / int8 compute / int4 storage) M=%d K=%d N=%d  rc=%d RMS rel err=%.3f%%\n", ok?"ok  ":"WRONG", M,K,N, rc, rms*100);
    ork_w_free(w); free(Bf);free(Af);free(bsc);free(Ai);free(asc);free(Ci);
    return ok?0:1;
}

/* The fixed bitsandbytes NF4 levels (index 0..15) — must match ORK_NF4_LEVELS in src/npu.c. */
static const float NF4_LEVELS[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f };
/* nearest NF4 index for normalized w in [-1,1] (round-to-nearest, matches quant_chan_nf4 sr=0) */
static int nf4_nearest(float wn) {
    if (wn > 1.0f) wn = 1.0f; else if (wn < -1.0f) wn = -1.0f;
    int hi = 0; while (hi < 15 && NF4_LEVELS[hi] < wn) hi++;
    int lo = hi > 0 ? hi-1 : 0;
    if (lo == hi) return hi;
    float t = (wn - NF4_LEVELS[lo]) / (NF4_LEVELS[hi]-NF4_LEVELS[lo]);
    return (t >= 0.5f) ? hi : lo;
}
/* Box-Muller standard-normal sample (own LCG so it's deterministic + independent of rnd()) */
static unsigned gsd = 99173; static double gunif(void){ gsd = gsd*1103515245u+12345u; return ((gsd>>8)&0xffffff)/16777216.0 + 1e-9; }
static float gauss(void){ double u1=gunif(), u2=gunif(); return (float)(sqrt(-2.0*log(u1))*cos(6.283185307179586*u2)); }

/* (1) NF4 COMPUTE correctness: pack random f32 weights with the NF4 codebook (ORK_NF4 set), run run_i8,
 * compare to a CPU reference that uses the SAME nearest-NF4 indices (level*bscale x dequant'd int8 acts).
 * Isolates compute correctness given the NF4-quantized weights. Tolerance: int8-activation quant noise. */
static int check_pack_nf4_correct(ork_npu *ctx) {
    int M = 8, K = 2048, N = 512;
    float *Bf = malloc((size_t)N*K*sizeof(float)), *Af = malloc((size_t)M*K*sizeof(float)), *bsc = malloc((size_t)N*sizeof(float));
    for (size_t j=0;j<(size_t)N*K;j++) Bf[j] = gauss();                    /* Gaussian weights: NF4's design point */
    for (size_t j=0;j<(size_t)M*K;j++) Af[j] = gauss();
    setenv("ORK_NF4", "1", 1);
    ork_w *w = ork_i4a8_mm_pack(ctx, K, N, Bf, bsc);
    unsetenv("ORK_NF4");
    if (!w) { printf("  pack_i4a8(NF4) failed\n"); free(Bf);free(Af);free(bsc); return 1; }
    int qk = ork_w_quant_kind(w); int qk_ok = (qk == ORK_QK_CODEBOOK_NF4);
    int8_t *Ai = malloc((size_t)M*K); float *asc = malloc((size_t)M*sizeof(float)); int32_t *Ci = malloc((size_t)M*N*4);
    for (int m=0;m<M;m++){ float mx=1e-9f; for(int k=0;k<K;k++){float v=fabsf(Af[(size_t)m*K+k]); if(v>mx)mx=v;}
        asc[m]=mx/127.0f; float iv=127.0f/mx; for(int k=0;k<K;k++){int q=(int)lrintf(Af[(size_t)m*K+k]*iv); Ai[(size_t)m*K+k]=(int8_t)(q>127?127:q<-127?-127:q);} }
    int rc = ork_i8_mm_run(ctx, w, M, Ai, Ci);
    /* CPU reference: per channel n, absmax=max|w|, bscale=absmax/127; each weight -> nearest NF4 level,
     * dequant = level*absmax = level*127*bscale. Compare against asc[m]*bsc[n]*Ci. */
    double se=0, sref=0;
    if (!rc) for (int n=0;n<N;n++){
        const float *wr = Bf + (size_t)n*K; float bs = bsc[n]; float absmax = bs*127.0f; float ainv = absmax>0?1.0f/absmax:0.0f;
        for (int m=0;m<M;m++){
            double ref=0;
            for (int k=0;k<K;k++){ int idx = nf4_nearest(wr[k]*ainv);
                double wdq = (double)NF4_LEVELS[idx]*absmax; double adq = (double)Ai[(size_t)m*K+k]*asc[m];
                ref += adq*wdq; }
            double got = (double)asc[m]*bs*Ci[(size_t)m*N+n];
            se += (got-ref)*(got-ref); sref += ref*ref; }
    }
    double rms = sqrt(se/((double)M*N)) / (sqrt(sref/((double)M*N))+1e-9);
    int ok = (!rc) && qk_ok && rms < 0.03;
    printf("  %s pack_i4a8 NF4 correctness M=%d K=%d N=%d  rc=%d quant_kind=%d RMS rel err=%.3f%%\n",
           ok?"ok  ":"WRONG", M,K,N, rc, qk, rms*100);
    ork_w_free(w); free(Bf);free(Af);free(bsc);free(Ai);free(asc);free(Ci);
    return ok?0:1;
}

/* (2) ACCURACY GATE (Phase-1): on Gaussian N(0,1) weights, NF4's reconstruction error must beat uniform
 * int4. Quantize the SAME weights both ways (CPU-side, matching the pack math) and compare RMS of
 * (dequant(q) - w) vs the original f32. Asserts NF4_err < uniform_err. */
static int check_nf4_accuracy_gate(void) {
    int K = 4096, N = 256;
    float *Bf = malloc((size_t)N*K*sizeof(float));
    for (size_t j=0;j<(size_t)N*K;j++) Bf[j] = gauss();
    double se_u=0, se_n=0, sw=0;
    for (int n=0;n<N;n++){
        const float *wr = Bf + (size_t)n*K; float mx=1e-9f;
        for (int k=0;k<K;k++){ float v=fabsf(wr[k]); if(v>mx)mx=v; }
        float uscale = mx/7.0f;                          /* uniform int4: scale=max/7, RN, clamp [-7,7] */
        float ainv = mx>0?1.0f/mx:0.0f;                  /* NF4: normalize by absmax */
        for (int k=0;k<K;k++){
            int q=(int)lrintf(wr[k]/uscale); if(q>7)q=7; else if(q<-7)q=-7;
            double udq = (double)q*uscale;
            int idx = nf4_nearest(wr[k]*ainv); double ndq = (double)NF4_LEVELS[idx]*mx;
            double d_u = udq - wr[k], d_n = ndq - wr[k];
            se_u += d_u*d_u; se_n += d_n*d_n; sw += (double)wr[k]*wr[k];
        }
    }
    double n_tot=(double)N*K, wr_rms=sqrt(sw/n_tot)+1e-12;
    double err_u = sqrt(se_u/n_tot), err_n = sqrt(se_n/n_tot);
    int ok = err_n < err_u;
    printf("  %s NF4 accuracy gate (Gaussian N(0,1), K=%d N=%d): uniform err=%.4f  NF4 err=%.4f  ratio NF4/uniform=%.3f  (w_rms=%.3f)\n",
           ok?"ok  ":"WRONG", K, N, err_u/wr_rms, err_n/wr_rms, err_n/err_u, wr_rms);
    free(Bf);
    return ok?0:1;
}

/* (3) IMATRIX weighted scale selection (ork_i4a8_mm_pack_im, Phase 1.3): build channels where a few
 * INPUT columns are "important" (large imatrix[k]) and hold mid-range, non-outlier values, while a few
 * UNIMPORTANT columns hold large outliers. Plain absmax then wastes the int4 grid resolving outliers
 * that don't matter; imatrix-weighted clip selection picks a tighter scale that resolves the important
 * bulk better. We measure the IMPORTANCE-WEIGHTED reconstruction error Sum_k im[k]*(w-dequant)^2 for the
 * absmax (imatrix=NULL) pack and the imatrix pack, and assert imatrix < absmax. We also assert the NULL
 * path is byte-identical to the prior absmax behavior (bscale[n] == absmax_n/7), and that a real matmul
 * on the imatrix-packed weights stays finite/bounded vs a CPU reference. */
static double i4a8_chan_werr(const float *wr, int K, float bscale, const float *im) {
    /* dequant the channel the SAME way the uniform int4 pack does (scale=bscale, RN, clamp [-7,7]) and
     * accumulate the importance-weighted squared error. bscale == chosen scale (absmax/7 or r*absmax/7). */
    float biv = bscale > 0 ? 1.0f/bscale : 0.0f; double e = 0;
    for (int k=0;k<K;k++){ int q=(int)lrintf(wr[k]*biv); if(q>7)q=7; else if(q<-7)q=-7;
        double dq=(double)q*bscale, d=(double)wr[k]-dq; e += (double)im[k]*d*d; }
    return e;
}
static int check_pack_i4a8_imatrix(ork_npu *ctx) {
    int M = 8, K = 2048, N = 512;
    float *Bf = malloc((size_t)N*K*sizeof(float)), *Af = malloc((size_t)M*K*sizeof(float));
    float *bsc0 = malloc((size_t)N*sizeof(float)), *bsc1 = malloc((size_t)N*sizeof(float));
    float *im = malloc((size_t)K*sizeof(float));
    /* importance: first 64 input columns important (weight 50), the rest unimportant (weight ~0.01) */
    for (int k=0;k<K;k++) im[k] = (k < 64) ? 50.0f : 0.01f;
    /* weights: important columns hold mid-range values; a handful of unimportant columns carry big
     * outliers (which inflate absmax and waste the int4 grid). */
    for (int n=0;n<N;n++){ float *wr = Bf + (size_t)n*K;
        for (int k=0;k<K;k++){
            if (k < 64)            wr[k] = ((int)rnd()-128)/256.0f;        /* important: ~[-0.5,0.5] mid-range */
            else if ((k % 401)==0) wr[k] = ((rnd()&1)?+1:-1)*(6.0f + (rnd()&7)); /* unimportant outliers ~+-6..13 */
            else                   wr[k] = ((int)rnd()-128)/4096.0f;       /* unimportant: tiny */
        }
    }
    for (size_t j=0;j<(size_t)M*K;j++) Af[j] = ((int)rnd()-128)/64.0f;
    /* absmax pack (imatrix=NULL) and imatrix pack */
    ork_w *w0 = ork_i4a8_mm_pack_im(ctx, K, N, Bf, NULL, bsc0);
    ork_w *w1 = ork_i4a8_mm_pack_im(ctx, K, N, Bf, im,   bsc1);
    if (!w0 || !w1) { printf("  pack_i4a8_im failed\n"); free(Bf);free(Af);free(bsc0);free(bsc1);free(im);
        if(w0)ork_w_free(w0); if(w1)ork_w_free(w1); return 1; }
    /* (a) NULL path byte-identical: bsc0[n] must equal absmax_n/7 exactly (the prior absmax behavior). */
    int null_ok = 1;
    for (int n=0;n<N;n++){ const float *wr=Bf+(size_t)n*K; float mx=1e-9f;
        for (int k=0;k<K;k++){float v=fabsf(wr[k]); if(v>mx)mx=v;}
        if (bsc0[n] != mx/7.0f) { null_ok = 0; break; } }
    /* (b) weighted-error comparison: imatrix pack must beat absmax pack on the importance-weighted error */
    double e0=0, e1=0;
    for (int n=0;n<N;n++){ const float *wr=Bf+(size_t)n*K;
        e0 += i4a8_chan_werr(wr, K, bsc0[n], im);
        e1 += i4a8_chan_werr(wr, K, bsc1[n], im); }
    /* (c) correctness: matmul on the imatrix-packed weights stays finite + bounded vs CPU reference */
    int8_t *Ai = malloc((size_t)M*K); float *asc = malloc((size_t)M*sizeof(float)); int32_t *Ci = malloc((size_t)M*N*4);
    for (int m=0;m<M;m++){ float mx=1e-9f; for(int k=0;k<K;k++){float v=fabsf(Af[(size_t)m*K+k]); if(v>mx)mx=v;}
        asc[m]=mx/127.0f; float iv=127.0f/mx; for(int k=0;k<K;k++){int q=(int)lrintf(Af[(size_t)m*K+k]*iv); Ai[(size_t)m*K+k]=(int8_t)(q>127?127:q<-127?-127:q);} }
    int rc = ork_i8_mm_run(ctx, w1, M, Ai, Ci);
    double se=0, sref=0; int finite=1;
    if (!rc) for (int n=0;n<N;n++){
        const float *wr = Bf + (size_t)n*K; float bs = bsc1[n], biv = bs>0?1.0f/bs:0.0f;
        for (int m=0;m<M;m++){
            double ref=0;
            for (int k=0;k<K;k++){ int q=(int)lrintf(wr[k]*biv); if(q>7)q=7; else if(q<-7)q=-7;
                ref += ((double)Ai[(size_t)m*K+k]*asc[m]) * ((double)q*bs); }
            double got = (double)asc[m]*bs*Ci[(size_t)m*N+n];
            if (!isfinite(got)) finite=0;
            se += (got-ref)*(got-ref); sref += ref*ref; }
    }
    double rms = sqrt(se/((double)M*N)) / (sqrt(sref/((double)M*N))+1e-9);
    int ok = (!rc) && null_ok && finite && (e1 < e0) && rms < 0.03;
    printf("  %s pack_i4a8 imatrix M=%d K=%d N=%d  rc=%d NULL-identical=%d  weighted err: absmax=%.4g imatrix=%.4g (ratio im/absmax=%.3f)  matmul RMS=%.3f%%\n",
           ok?"ok  ":"WRONG", M,K,N, rc, null_ok, e0, e1, e0>0?e1/e0:0.0, rms*100);
    ork_w_free(w0); ork_w_free(w1); free(Bf);free(Af);free(bsc0);free(bsc1);free(im);free(Ai);free(asc);free(Ci);
    return ok?0:1;
}

/* COMPACT int4 persist/load round-trip (Phase 2.1): pack random f32 with int4 (NF4 or UNIFORM) via
 * ork_i4a8_mm_pack and run it (C_resident). Dump the COMPACT int4 form (ork_i4a8_w_dump, size-query then
 * fill), reload it (ork_i4a8_mm_load) and run again (C_streamed). Asserts the Phase-2 streaming gate:
 *   (a) C_streamed == C_resident EXACTLY (same nibbles/scales/LUT => bit-identical NPU output),
 *   (b) re-dumping the loaded weight yields byte-identical bytes (full self-contained round-trip),
 *   (c) the per-channel bscale survives the round-trip. */
static int check_dump_load_i4a8(ork_npu *ctx, int nf4) {
    int M = 8, K = 2048, N = 512;
    const char *tag = nf4 ? "NF4" : "UNIFORM";
    float *Bf = malloc((size_t)N*K*sizeof(float)), *Af = malloc((size_t)M*K*sizeof(float)), *bsc = malloc((size_t)N*sizeof(float));
    for (size_t j=0;j<(size_t)N*K;j++) Bf[j] = gauss();
    for (size_t j=0;j<(size_t)M*K;j++) Af[j] = gauss();
    if (nf4) setenv("ORK_NF4", "1", 1);
    ork_w *w = ork_i4a8_mm_pack(ctx, K, N, Bf, bsc);
    if (nf4) unsetenv("ORK_NF4");
    if (!w) { printf("  dump_load_i4a8(%s): pack failed\n", tag); free(Bf);free(Af);free(bsc); return 1; }

    /* int8-quantize A (matches the run_i8 activation path) */
    int8_t *Ai = malloc((size_t)M*K); int32_t *C_res = malloc((size_t)M*N*4), *C_str = malloc((size_t)M*N*4);
    for (int m=0;m<M;m++){ float mx=1e-9f; for(int k=0;k<K;k++){float v=fabsf(Af[(size_t)m*K+k]); if(v>mx)mx=v;}
        float iv=127.0f/mx; for(int k=0;k<K;k++){int q=(int)lrintf(Af[(size_t)m*K+k]*iv); Ai[(size_t)m*K+k]=(int8_t)(q>127?127:q<-127?-127:q);} }
    int rc1 = ork_i8_mm_run(ctx, w, M, Ai, C_res);

    /* dump compact int4 form: size-query then fill */
    size_t need = ork_i4a8_w_dump(w, NULL, 0);
    int bad = 0;
    if (need == 0) { printf("  dump_load_i4a8(%s): size-query returned 0\n", tag); bad = 1; }
    size_t expect = 5*4 /*hdr: 2 u32 + 2 i32 + 1 u32*/ + (size_t)N*sizeof(float) + (size_t)K*N/2;
    if (!bad && need != expect) { printf("  dump_load_i4a8(%s): size %zu != expected %zu\n", tag, need, expect); bad = 1; }
    void *blob = malloc(need);
    size_t got = ork_i4a8_w_dump(w, blob, need);
    if (!bad && got != need) { printf("  dump_load_i4a8(%s): fill returned %zu != %zu\n", tag, got, need); bad = 1; }

    /* reload + run */
    ork_w *wl = bad ? NULL : ork_i4a8_mm_load(ctx, K, N, blob, need);
    if (!bad && !wl) { printf("  dump_load_i4a8(%s): load failed\n", tag); bad = 1; }
    int rc2 = wl ? ork_i8_mm_run(ctx, wl, M, Ai, C_str) : -1;

    /* (a) bit-identical output */
    int out_exact = (!bad && rc1==0 && rc2==0) ? (memcmp(C_res, C_str, (size_t)M*N*4)==0) : 0;
    /* (b) byte-identical re-dump */
    int rt_exact = 0;
    if (wl) { void *blob2 = malloc(need); size_t got2 = ork_i4a8_w_dump(wl, blob2, need);
              rt_exact = (got2==need && memcmp(blob, blob2, need)==0); free(blob2); }
    /* (c) bscale carried + quant_kind */
    int bsc_ok = 0;
    if (wl) { const float *bl = ork_w_bscale(wl); bsc_ok = (bl != NULL);
              if (bl) for (int n=0;n<N;n++) if (bl[n] != bsc[n]) { bsc_ok = 0; break; } }
    int qk_ok = wl ? (ork_w_quant_kind(wl) == (nf4?ORK_QK_CODEBOOK_NF4:ORK_QK_UNIFORM)) : 0;

    int ok = !bad && out_exact && rt_exact && bsc_ok && qk_ok;
    printf("  %s dump_load_i4a8 %s M=%d K=%d N=%d  blob=%zuB (int8 dump=%zuB)  rc=%d/%d  out_exact=%d roundtrip=%d bscale=%d qk=%d\n",
           ok?"ok  ":"WRONG", tag, M,K,N, need, (size_t)K*N, rc1, rc2, out_exact, rt_exact, bsc_ok, qk_ok);
    ork_w_free(w); if (wl) ork_mm_free(ctx, wl);
    free(Bf);free(Af);free(bsc);free(Ai);free(C_res);free(C_str);free(blob);
    return ok?0:1;
}

static int test_overlap_guards(ork_npu *ctx) {
    printf("Testing memory overlap safety guards...\n");
    int M = 1, K = 32, N = 32;
    int8_t *B = malloc((size_t)K * N);
    for (int i = 0; i < K * N; i++) B[i] = 1;
    ork_w *w = ork_i8_mm_pack(ctx, K, N, B);
    if (!w) {
        printf("  test_overlap_guards failed: pack_i8 failed\n");
        free(B);
        return 1;
    }

    size_t size = 256;
    int8_t *shared = malloc(size);
    int8_t *A = shared;
    int32_t *C = (int32_t *)(shared + 16); // 16 bytes offset, overlapping under K=32

    int ret = ork_i8_mm_run(ctx, w, M, A, C);
    int bad = 0;
    if (ret != -1) {
        printf("  test_overlap_guards failed: run_i8 did not reject overlapping buffers! (ret=%d)\n", ret);
        bad = 1;
    } else {
        printf("  test_overlap_guards passed: run_i8 correctly rejected overlapping buffers.\n");
    }

    ork_w_free(w);
    free(B);
    free(shared);
    return bad;
}

int main(void){
    int fail=0;
    /* fp16 and int8 in SEPARATE contexts: a model is one precision, and switching regcmd mode
     * (fp16<->int8) on a live context wedges the first submit in the new mode for ~6s (the NPU
     * mode is stateful — see the wiki). Each precision gets a fresh context here, which also
     * mirrors real usage. (ork_i8_mm_run on fp16 weights still safely returns an error.) */
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    ork_npu_set_core_budget(ctx, 3);
    //
    //256,4096,512);
    //512,8192,128);
    //64,11008,64);     /* non-power-of-2 K */
    //32,2048,256);
    //8,512,16384);     /* N-tiling: N>8192 (NPU output-width cap) */
    //1,288,32000);     /* LM-head shape: non-pow2 K + N tiled into 4 slices */
    //4,6144,2048);     /* non-pow2 K<=8192 — decode (M=1) single-submit boundary */
    /* SINGLE-CORE fp16 regression (budget=1): the run() M-scheduler (sched=1) miscomputes >8 rows at
     * Kp>=2048 (validated mc<=8 OK / mc>=9 garbage); it is now gated to sched=0 there. The layer/model
     * path is multi-core (always tiles to mc=8) so this single-core M>8 path was previously UNTESTED. */
    ork_npu_set_core_budget(ctx, 1);
    fail|=check(ctx, 64, 2048, 256,  0x63da26feb5488eddULL);    /* fp16 1-core K=2048 M=64>8 — pre-fix: garbage */
    fail|=check(ctx, 256, 3584, 256, 0xd659053d19504e43ULL);   /* fp16 1-core K=3584 M=256       */
    ork_npu_free(ctx);
    ork_npu*c8=ork_npu_init(); if(!c8){printf("init failed (NPU?)\n");return 1;}
    ork_npu_set_core_budget(c8, 3);
    //128,512,128);
    //256,4096,512);
    //64,11008,32);   /* non-power-of-2 K (768 remainder fallback) */
    //1,8192,512);    /* decode */
    //8,1280,64);     /* 256 remainder slice */
    //4,6144,2048);   /* non-pow2 K<=8192 — decode (M=1) single-submit boundary */
    fail|=check_chain_prefill(c8);    /* verify CHAIN-PREFILL: int8 M>1 multi-core M-tile chaining (bit-exact) */
    fail|=check_chain_i8(c8);         /* verify chained matmuls / MoE API */
    fail|=check_chain_i8_bf(c8);      /* verify Bf-based chaining (Sk=2 experts, MoE-prefill path) */
    fail|=check_stream_i8(c8);        /* verify async round-robin stream (cross-core, mixed shapes) */
    fail|=check_chain_envelope(c8);   /* verify full-K envelope guard (reject K%512!=0 vs silent-wrong) */
    fail|=check_pack_i8_f32(c8);      /* verify NEON f32->int8 pack (used by the MoE repack) */
    fail|=check_pack_i4a8(c8);        /* verify "effective w4a8": int4 wt / int8 compute / int4 storage */
    fail|=check_pack_nf4_correct(c8); /* verify NF4 codebook compute correctness (ORK_NF4 path) */
    fail|=check_nf4_accuracy_gate();  /* Phase-1 gate: NF4 beats uniform int4 on Gaussian weights (CPU-only) */
    fail|=check_pack_i4a8_imatrix(c8);/* Phase-1.3: imatrix weighted scale selection beats absmax on important cols */
    fail|=check_dump_load_i4a8(c8, 1);/* Phase-2.1: compact int4 persist/load round-trip (NF4) — streamed==resident */
    fail|=check_dump_load_i4a8(c8, 0);/* Phase-2.1: compact int4 persist/load round-trip (UNIFORM) */
    fail|=test_overlap_guards(c8);    /* verify memory overlap guards */
    /* SINGLE-CORE int8 regression (budget=1, run LAST so it doesn't disturb the multi-core checks
     * above): exercises the weight-DMA M-tile = mg_max*64 (128 @K2048) on the single-core full-K path —
     * the lever that was throttled to R-1=31. Bit-exact integer ref guards both the size and the fix. */
    ork_npu_set_core_budget(c8, 1);
    fail|=check_i8(c8, 512, 2048, 256, 0xda6b68f190453353ULL);   /* int8 1-core K=2048 M=512 (mg_max*64 tile) */
    fail|=check_i8(c8, 256, 3584, 256, 0x454860dd91b46165ULL);   /* int8 1-core K=3584 M=256 (mg_max*64=64)   */
    fail|=check_i8(c8, 256, 256,  32, 0xf23ebd149849d390ULL);    /* int8 1-core K=256 (K%512!=0 -> run_loop) M=256>32768/Kp — guards the sched=0 M-tile cap (rows past the tile were garbage pre-fix) */
    ork_npu_free(c8);

    /* No ioctl may fail UNEXPECTEDLY over a clean run. MEM_SYNC/MEM_DESTROY/ACTION returns used to be
     * discarded everywhere, and on this vendor driver those failures do not surface at the call site --
     * they surface later as silent wrong numbers (a stale buffer) or as a wedge (leaked IOVA). Asserting
     * here turns that class into a test failure instead of a mystery two hours later. Expected failures
     * (recovery paths poking a device already believed stuck) are counted separately and NOT asserted on,
     * so a healthy run cannot go red -- a gate that cries wolf gets switched off. */
    const long iofail = ork_io_failures(), ioexp = ork_io_expected_failures();
    if (iofail) { fprintf(stderr, "[test_matmul] %ld UNEXPECTED ioctl failure(s) — see the [ork-io] lines above\n", iofail); fail = 1; }
    else if (ioexp) printf("test_matmul: %ld expected (recovery-path) ioctl failure(s), 0 unexpected\n", ioexp);

    printf("%s\n",fail?"FAIL":"ALL OK");
    return fail?1:0;
}
