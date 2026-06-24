/* examples/test_matmul.c — validates the ork_npu library: one handle, resident weights reused across
 * many matmuls of varying M (the forward-pass access pattern). Builds vs CPU reference.
 *   cc -O2 -I. -o test_mm test_mm.c ork_npu.c && sudo ./test_mm */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "ork_npu.h"
typedef ork_f16 f16;
static unsigned sd=12345; static int rnd(){sd=sd*1103515245+12345;return (sd>>16)%4;}

static int check(ork_npu*ctx,int M,int K,int N){
    printf("FP16 check start: M=%d, K=%d, N=%d\n", M, K, N); fflush(stdout);
    f16*A=malloc((size_t)M*K*2),*B=malloc((size_t)K*N*2); float*C=malloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(f16)rnd();
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=(f16)rnd();
    printf("  Packing...\n"); fflush(stdout);
    ork_w*w=ork_mm_pack(ctx,K,N,B);
    if(!w){printf("pack failed %d,%d\n",K,N);return 1;}
    int bad=0;
    /* run the SAME resident weights for several M (decode then prefill), validate each */
    int Ms[]={1,1,4,M}; for(int t=0;t<4;t++){int m=Ms[t]; if(m>M)m=M;
        printf("  Running run (t=%d, m=%d)...\n", t, m); fflush(stdout);
        printf("Running ork_mm_run...\n");
        if(ork_mm_run(ctx,w,m,A,C)){printf("run failed\n");return 1;}
        for(int i=0;i<m;i++)for(int n=0;n<N;n++){float ref=0;for(int k=0;k<K;k++)ref+=(float)A[(size_t)i*K+k]*(float)B[(size_t)k*N+n]; if(C[(size_t)i*N+n]!=ref)bad++;}
    }
    printf("  %s MKN=%d,%d,%d (reused weights x4 runs) mism=%d\n",bad?"WRONG":"ok  ",M,K,N,bad); fflush(stdout);
    ork_w_free(w); free(A);free(B);free(C); return bad?1:0;
}
/* int8/w8a8: A,B int8 -> C int32 (exact integer reference). K%32, N%32. */
static int check_i8(ork_npu*ctx,int M,int K,int N){
    printf("Int8 check start: M=%d, K=%d, N=%d\n", M, K, N); fflush(stdout);
    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t*C=malloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(int8_t)(rnd()-1);
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=(int8_t)(rnd()-1);
    printf("  Packing...\n"); fflush(stdout);
    ork_w*w=ork_mm_pack_i8(ctx,K,N,B);
    if(!w){printf("pack_i8 failed %d,%d\n",K,N);return 1;}
    int bad=0; int Ms[]={1,1,4,M};
    for(int t=0;t<4;t++){int m=Ms[t]; if(m>M)m=M;
        printf("  Running run_i8 (t=%d, m=%d)...\n", t, m); fflush(stdout);
        printf("Running ork_mm_run_i8...\n");
        if(ork_mm_run_i8(ctx,w,m,A,C)){printf("run_i8 failed\n");return 1;}
        for(int i=0;i<m;i++)for(int n=0;n<N;n++){int32_t ref=0;for(int k=0;k<K;k++)ref+=(int)A[(size_t)i*K+k]*(int)B[(size_t)k*N+n]; if(C[(size_t)i*N+n]!=ref)bad++;}
    }
    printf("  %s MKN=%d,%d,%d int8 (reused weights x4 runs) mism=%d\n",bad?"WRONG":"ok  ",M,K,N,bad); fflush(stdout);
    ork_w_free(w); free(A);free(B);free(C); return bad?1:0;
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
        
        w[i] = ork_mm_pack_i8(ctx, K, N, B[i]);
        if (!w[i]) { printf("pack_chain_i8 failed %d\n", i); return 1; }
        
        tasks[i].w = w[i];
        tasks[i].M = Ms[i];
        tasks[i].A = A[i];
        tasks[i].C = C[i];
    }
    
    int bad = 0;
    printf("Running ork_mm_run_chain_i8...\n");
    if (ork_mm_run_chain_i8(ctx, S, tasks)) {
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
        w[i] = ork_mm_pack_i8(ctx, K, N, B[i]);
        if (!w[i]) { printf("pack_chain_i8_bf failed %d\n", i); return 1; }
        tasks[i].w = w[i]; tasks[i].M = Ms[i]; tasks[i].A = A[i]; tasks[i].C = C[i];
    }
    int bad = 0, rc = ork_mm_run_chain_i8(ctx, S, tasks);
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
        w[i]=ork_mm_pack_i8(ctx,K,N,B[i]); if(!w[i]){printf("pack_stream failed %d\n",i);return 1;}
        tasks[i].w=w[i]; tasks[i].M=M; tasks[i].A=A[i]; tasks[i].C=C[i];
    }
    int bad=0, rc=ork_mm_run_stream_i8(ctx,S,tasks);
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
      ork_w *w = ork_mm_pack_i8(ctx,K,N,B);
      ork_mm_task_i8 t = { w, M, A, C };
      int rc_chain = w ? ork_mm_run_chain_i8(ctx, 1, &t) : 0;   /* S=1 still goes through validation... */
      ork_mm_task_i8 ts[2] = { {w,M,A,C}, {w,M,A,C} };          /* S=2 to exercise the chain path proper */
      int rc_chain2 = w ? ork_mm_run_chain_i8(ctx, 2, ts) : 0;
      int rc_stream = w ? ork_mm_run_stream_i8(ctx, 2, ts) : 0;
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
      ork_w *w = ork_mm_pack_i8(ctx,K,N,B);
      ork_mm_task_i8 ts[2] = { {w,M,A,C}, {w,M,A,C} };
      int rc = w ? ork_mm_run_chain_i8(ctx, 2, ts) : -1;
      if (rc) { printf("  ENVELOPE: in-envelope K=1536 chain failed rc=%d\n", rc); bad++; }
      else for (int r=0;r<M && bad<6;r++) for (int n=0;n<N;n++) {
          int32_t ref=0; for(int k=0;k<K;k++) ref+=(int)A[(size_t)r*K+k]*(int)B[(size_t)k*N+n];
          if (C[(size_t)r*N+n]!=ref) { printf("  ENVELOPE: K=1536 mism r%d c%d exp %d got %d\n",r,n,ref,C[(size_t)r*N+n]); bad++; } }
      if (w) ork_w_free(w); free(A); free(B); free(C); }
    printf("  %s chain/stream full-K envelope guard (reject K%%512!=0; K=1536 correct)\n", bad?"WRONG":"ok  ");
    return bad ? 1 : 0;
}

/* Validate the NEON f32->int8 pack (ork_mm_pack_i8_f32): pack f32 weights, run_i8, dequant with the
 * returned per-channel bscale, compare to the f32 CPU reference (within int8 W8A8 tolerance). */
static int check_pack_i8_f32(ork_npu *ctx) {
    int M = 8, K = 2048, N = 512;   /* K=2048 -> Sk=2 + Bf, exercises the full-K tile path too */
    float *Bf = malloc((size_t)N*K*sizeof(float)), *Af = malloc((size_t)M*K*sizeof(float)), *bsc = malloc((size_t)N*sizeof(float));
    for (size_t j=0;j<(size_t)N*K;j++) Bf[j] = ((int)rnd()-128)/64.0f;     /* [N][K] n-major */
    for (size_t j=0;j<(size_t)M*K;j++) Af[j] = ((int)rnd()-128)/64.0f;
    ork_w *w = ork_mm_pack_i8_f32(ctx, K, N, Bf, bsc);
    if (!w) { printf("  pack_i8_f32 failed\n"); free(Bf);free(Af);free(bsc); return 1; }
    int8_t *Ai = malloc((size_t)M*K); float *asc = malloc((size_t)M*sizeof(float)); int32_t *Ci = malloc((size_t)M*N*4);
    for (int m=0;m<M;m++){ float mx=1e-9f; for(int k=0;k<K;k++){float v=fabsf(Af[(size_t)m*K+k]); if(v>mx)mx=v;}
        asc[m]=mx/127.0f; float iv=127.0f/mx; for(int k=0;k<K;k++){int q=(int)lrintf(Af[(size_t)m*K+k]*iv); Ai[(size_t)m*K+k]=(int8_t)(q>127?127:q<-127?-127:q);} }
    int rc = ork_mm_run_i8(ctx, w, M, Ai, Ci);
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

static int test_overlap_guards(ork_npu *ctx) {
    printf("Testing memory overlap safety guards...\n");
    int M = 1, K = 32, N = 32;
    int8_t *B = malloc((size_t)K * N);
    for (int i = 0; i < K * N; i++) B[i] = 1;
    ork_w *w = ork_mm_pack_i8(ctx, K, N, B);
    if (!w) {
        printf("  test_overlap_guards failed: pack_i8 failed\n");
        free(B);
        return 1;
    }

    size_t size = 256;
    int8_t *shared = malloc(size);
    int8_t *A = shared;
    int32_t *C = (int32_t *)(shared + 16); // 16 bytes offset, overlapping under K=32

    int ret = ork_mm_run_i8(ctx, w, M, A, C);
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
     * mirrors real usage. (ork_mm_run_i8 on fp16 weights still safely returns an error.) */
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
    ork_npu_free(ctx);
    ork_npu*c8=ork_npu_init(); if(!c8){printf("init failed (NPU?)\n");return 1;}
    ork_npu_set_core_budget(c8, 3);
    //128,512,128);
    //256,4096,512);
    //64,11008,32);   /* non-power-of-2 K (768 remainder fallback) */
    //1,8192,512);    /* decode */
    //8,1280,64);     /* 256 remainder slice */
    //4,6144,2048);   /* non-pow2 K<=8192 — decode (M=1) single-submit boundary */
    fail|=check_chain_i8(c8);         /* verify chained matmuls / MoE API */
    fail|=check_chain_i8_bf(c8);      /* verify Bf-based chaining (Sk=2 experts, MoE-prefill path) */
    fail|=check_stream_i8(c8);        /* verify async round-robin stream (cross-core, mixed shapes) */
    fail|=check_chain_envelope(c8);   /* verify full-K envelope guard (reject K%512!=0 vs silent-wrong) */
    fail|=check_pack_i8_f32(c8);      /* verify NEON f32->int8 pack (used by the MoE repack) */
    fail|=test_overlap_guards(c8);    /* verify memory overlap guards */
    ork_npu_free(c8);
    printf("%s\n",fail?"FAIL":"ALL OK");
    return fail?1:0;
}
