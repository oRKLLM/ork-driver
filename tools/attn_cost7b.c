/* tools/attn_cost7b.c — Phase 1 gating measurement for attention-on-NPU (lever b).
 *
 * Qwen2.5-7B: hidden 3584, NH=28 q-heads, NKV=4 kv-heads, HD=128, NL=28 layers.
 * Attention core per head = two activation×activation matmuls:
 *   QK^T : Q[M,HD] x K^T[HD,L2] -> S[M,L2]
 *   PV   : P[M,L2] x V[L2,HD]   -> O[M,HD]
 * softmax(S)->P stays on CPU (Phase 1 accepts the bounce).
 *
 * CRUCIAL difference from a weight matmul: K and V are PER-TOKEN ACTIVATIONS, so unlike a
 * resident weight they must be (re)packed/tiled into NPU layout EVERY forward pass. This probe
 * measures BOTH:
 *   (A) idealized "pack once" floor (submit cost only) — optimistic lower bound.
 *   (B) realistic "repack K/V each layer" cost — what attention-on-NPU would actually pay.
 * int8 (w8a8), matching the production path.
 *
 * Compare against the measured CPU attention cost for this shape (printed; fill from llama-bench
 * ORK_PROFILE attn bucket). If even (A) exceeds the CPU attention ms, NPU attention loses.
 *   make attn_cost7b && sudo ./attn_cost7b [M]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"
#define NH 28
#define NKV 4
#define HD 128
#define NL 28
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):256, L2=M, grp=NH/NKV;
    /* int8 needs N%32==0; L2 (=M) and HD must satisfy. HD=128 ok. L2=M must be %32. */
    if(M%32){printf("M must be %%32 for int8 (got %d)\n",M);return 2;}
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int8_t*Kt=malloc((size_t)HD*L2),*V=malloc((size_t)L2*HD),*Q=malloc((size_t)M*HD),*P=malloc((size_t)M*L2);
    memset(Kt,1,(size_t)HD*L2); memset(V,1,(size_t)L2*HD); memset(Q,1,(size_t)M*HD); memset(P,1,(size_t)M*L2);
    int32_t*S=malloc((size_t)M*L2*4),*O=malloc((size_t)M*HD*4);
    /* pack K^T (HD x L2) and V (L2 x HD) per kv-head */
    ork_w*wK[NKV],*wV[NKV];
    for(int kv=0;kv<NKV;kv++){ wK[kv]=ork_i8_mm_pack(c,HD,L2,Kt); wV[kv]=ork_i8_mm_pack(c,L2,HD,V);
        if(!wK[kv]||!wV[kv]){printf("pack failed (M=%d L2=%d: HD%%32=%d L2%%32=%d)\n",M,L2,HD%32,L2%32);return 1;} }
    ork_i8_mm_run(c,wK[0],M,Q,S); ork_i8_mm_run(c,wV[0],M,P,O); /* warm */

    /* (A) idealized floor: submit cost only, K/V already packed */
    double t0=now();
    for(int l=0;l<NL;l++) for(int h=0;h<NH;h++){int kv=h/grp;
        ork_i8_mm_run(c,wK[kv],M,Q,S);
        ork_i8_mm_run(c,wV[kv],M,P,O); }
    double dtA=now()-t0;

    /* (B) realistic: repack K and V (per-token activations) each layer before its heads run.
     * GQA: NKV distinct K/V per layer. repack_i8 reuses the DMA (no alloc churn). */
    t0=now();
    for(int l=0;l<NL;l++){
        for(int kv=0;kv<NKV;kv++){ ork_i8_mm_repack(c,wK[kv],HD,L2,Kt); ork_i8_mm_repack(c,wV[kv],L2,HD,V); }
        for(int h=0;h<NH;h++){int kv=h/grp;
            ork_i8_mm_run(c,wK[kv],M,Q,S);
            ork_i8_mm_run(c,wV[kv],M,P,O); }
    }
    double dtB=now()-t0;

    int nmm=NL*NH*2;
    printf("Qwen2.5-7B attention-on-NPU, M=L2=%d, %d matmuls (%d layers x %d heads x 2):\n",M,nmm,NL,NH);
    printf("  (A) pack-once FLOOR (submit only) : %.1f ms  (%.1f us/matmul)\n",dtA*1e3,dtA*1e6/nmm);
    printf("  (B) realistic (repack K/V/layer)  : %.1f ms  (repack adds %.1f ms)\n",dtB*1e3,(dtB-dtA)*1e3);
    printf("  >>> compare to the CPU attention bucket for this shape (llama-bench ORK_PROFILE).\n");
    ork_npu_free(c); return 0;
}
