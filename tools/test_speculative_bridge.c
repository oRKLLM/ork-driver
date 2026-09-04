/* tools/test_speculative_bridge.c — speculative-decoding ECONOMICS harness (mock, measurement only).
 *
 * The pivot: amortize the M=1 GEMV submit floor (~3234 us/token, see fence_probe) by verifying N
 * draft tokens in ONE batched M=N target forward. This harness measures the two things that decide
 * whether the pivot pays off on THIS hardware:
 *
 *   (1) M-batch amortization curve: ork_f16_mm_run at M = 1,2,4,8,16 on a target-shaped matmul. The
 *       correct primitive is a single M=N matmul (the M-scheduler / prefill path) — NOT a multi-task
 *       submit (STALE: "task_number>1 is broken" is REFUTED -- see OPS_REGISTRY chain assemblers). If M=N collapses the
 *       per-token cost, the floor is amortized.
 *   (2) Accept-gate cost: the CPU NEON logit-argmax + accept/reject sweep over the N-token tree, in
 *       place (no tensor re-allocation). Must stay a tiny fraction of one verify forward or it leaks
 *       the gains. Mocked draft tokens + logits — this measures STRUCTURAL overhead, not model quality.
 *
 * The actual draft->target speculative loop + hidden-state plumbing lives in the llama.cpp fork
 * (common_speculative_impl_draft_eagle3); this harness measures ork-driver's verify-matmul economics.
 *
 *   make test_speculative_bridge && sudo ./test_speculative_bridge
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#define K 2048            /* target hidden dim (Qwen3-1.7B-ish) */
#define NOUT 2048         /* a representative projection width */
#define VOCAB 151936      /* Qwen3 vocab — the accept-gate argmax span */
#define NDRAFT 8          /* speculative tokens proposed per step */
#define ITERS 30

static double us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW,&t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }

/* NEON argmax over V floats (the per-token verify decision: predicted token = argmax(logits)). */
static int argmax_f32(const float *x, int n){
    int best=0; float bv=x[0]; int i=0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t vm=vdupq_n_f32(x[0]);
    for(i=0;i+4<=n;i+=4) vm=vmaxq_f32(vm, vld1q_f32(x+i));
    bv=vmaxvq_f32(vm);                          /* max value (NEON horizontal) */
    /* locate first index hitting the max */
    for(i=0;i<n;i++) if(x[i]==bv){best=i;break;}
    return best;
#else
    for(i=1;i<n;i++) if(x[i]>bv){bv=x[i];best=i;}
    return best;
#endif
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    ork_npu *c = ork_npu_init(); if(!c){ fprintf(stderr,"init failed\n"); return 1; }

    /* resident target weight (one projection, reused across all M) */
    ork_f16 *B = malloc((size_t)K*NOUT*sizeof(ork_f16));
    for(size_t i=0;i<(size_t)K*NOUT;i++) B[i]=(ork_f16)(((int)(i%17)-8)*0.02f);
    ork_w *w = ork_f16_mm_pack(c,K,NOUT,B); if(!w){ fprintf(stderr,"pack failed\n"); return 1; }

    int Mset[5]={1,2,4,8,16}; int MMAX=16;
    ork_f16 *A = malloc((size_t)MMAX*K*sizeof(ork_f16));
    for(size_t i=0;i<(size_t)MMAX*K;i++) A[i]=(ork_f16)(((i%13)-6)*0.04f);
    float *C = malloc((size_t)MMAX*NOUT*sizeof(float));

    printf("\n=== (1) M-batch amortization of the GEMV submit floor (K=%d N=%d) ===\n", K, NOUT);
    printf("  %-4s %-14s %-16s %s\n","M","us/forward","us/token","vs M=1 /token");
    double t_m1_per_tok=0;
    for(int mi=0; mi<5; mi++){
        int M=Mset[mi];
        for(int it=0; it<5; it++) ork_f16_mm_run(c,w,M,A,C);           /* warm */
        double t0=us(); for(int it=0; it<ITERS; it++) ork_f16_mm_run(c,w,M,A,C); double per=(us()-t0)/ITERS;
        double per_tok=per/M;
        if(M==1) t_m1_per_tok=per_tok;
        printf("  %-4d %-14.1f %-16.1f %.2fx\n", M, per, per_tok, t_m1_per_tok/per_tok);
    }

    /* (2) accept-gate: NDRAFT mock logit rows over VOCAB; argmax-compare vs proposed draft tokens. */
    printf("\n=== (2) CPU accept/reject gate (NEON argmax over vocab=%d, N=%d draft tokens) ===\n", VOCAB, NDRAFT);
    float *logits = malloc((size_t)NDRAFT*VOCAB*sizeof(float));
    unsigned sd=12345; for(size_t i=0;i<(size_t)NDRAFT*VOCAB;i++){ sd=sd*1103515245u+12345u; logits[i]=((int)((sd>>9)&0xffff)-32768)/4096.0f; }
    int draft[NDRAFT]; for(int i=0;i<NDRAFT;i++){ sd=sd*1103515245u+12345u; draft[i]=(int)(sd%VOCAB); }

    for(int it=0; it<20; it++){ volatile int z=0; for(int n=0;n<NDRAFT;n++) z+=argmax_f32(logits+(size_t)n*VOCAB,VOCAB); (void)z; }
    double t0=us();
    int accepted=0;
    for(int it=0; it<ITERS; it++){
        accepted=0;
        for(int n=0;n<NDRAFT;n++){ int pred=argmax_f32(logits+(size_t)n*VOCAB,VOCAB);
            if(pred==draft[n]) accepted++; else break; }   /* accept longest matching prefix */
    }
    double gate_us=(us()-t0)/ITERS;
    printf("  accept-gate sweep           : %8.3f us  (%d/%d tokens accepted, mock data)\n", gate_us, accepted, NDRAFT);
    printf("  per draft token             : %8.3f us\n", gate_us/NDRAFT);

    /* (3) structural verdict */
    printf("\n=== (3) structural verdict ===\n");
    double verify_fwd_proxy = 0; /* one M=NDRAFT forward, single projection as a proxy */
    for(int it=0; it<5; it++) ork_f16_mm_run(c,w,NDRAFT,A,C);
    { double t=us(); for(int it=0; it<ITERS; it++) ork_f16_mm_run(c,w,NDRAFT,A,C); verify_fwd_proxy=(us()-t)/ITERS; }
    printf("  M=%d verify matmul (proxy)   : %8.1f us\n", NDRAFT, verify_fwd_proxy);
    printf("  accept-gate / verify        : %.2f%%  %s\n", 100.0*gate_us/verify_fwd_proxy,
           gate_us < 0.05*verify_fwd_proxy ? "(under 5% budget — OK)" : "(EXCEEDS 5% budget — investigate)");
    printf("\n  NOTE: mock data, single-projection proxy — measures STRUCTURAL overhead, not model quality.\n");
    printf("  The real per-forward cost is multiple matmuls; the gate fraction shrinks further there.\n");
    return 0;
}
