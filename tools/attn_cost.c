/* tools/attn_cost.c — is attention-on-NPU worth it vs the CPU flash attention?
 *
 * Attention per head = two matmuls: QK^T (Q[M,HD] x K^T[HD,L2] -> S[M,L2]) and PV (P[M,L2] x
 * V[L2,HD] -> O[M,HD]); softmax(S)->P on CPU between them. For a Qwen3-1.7B prefill that's
 * NL*NH of each = ~896 *small* NPU matmuls. The NPU's weakness is per-submit overhead, so this
 * measures the floor: pack representative K/V ONCE, run all the QK^T+PV matmuls, time it. If this
 * floor already exceeds the CPU attention ms (M=128 ~264ms, M=256 ~1252ms pooled), NPU attention
 * loses (per-layer K/V packing would only add more). fp16 (attention precision).
 *   make attn_cost && sudo ./attn_cost [M]
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ork_npu.h"
typedef ork_f16 f16;
#define NH 16
#define NKV 8
#define HD 128
#define NL 28
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):128, L2=M, grp=NH/NKV;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    /* representative fp16 operands (values irrelevant — measuring time) */
    f16*Kt=malloc((size_t)HD*L2*2),*V=malloc((size_t)L2*HD*2),*Q=malloc((size_t)M*HD*2),*P=malloc((size_t)M*L2*2);
    for(size_t i=0;i<(size_t)HD*L2;i++)Kt[i]=(f16)0.01f;
    for(size_t i=0;i<(size_t)L2*HD;i++)V[i]=(f16)0.01f;
    for(size_t i=0;i<(size_t)M*HD;i++)Q[i]=(f16)0.01f;
    for(size_t i=0;i<(size_t)M*L2;i++)P[i]=(f16)0.01f;
    float*S=malloc((size_t)M*L2*4),*O=malloc((size_t)M*HD*4);
    /* pack K^T (HD x L2) and V (L2 x HD) once per kv-head (GQA shares across grp q-heads) */
    ork_w*wK[NKV],*wV[NKV];
    for(int kv=0;kv<NKV;kv++){ wK[kv]=ork_mm_pack(c,HD,L2,Kt); wV[kv]=ork_mm_pack(c,L2,HD,V);
        if(!wK[kv]||!wV[kv]){printf("pack failed (M=%d L2=%d: HD%%32=%d L2%%16=%d L2%%32=%d)\n",M,L2,HD%32,L2%16,L2%32);return 1;} }
    /* warm */
    ork_mm_run(c,wK[0],M,Q,S); ork_mm_run(c,wV[0],M,P,O);
    double t0=now();
    for(int l=0;l<NL;l++)
        for(int h=0;h<NH;h++){int kv=h/grp;
            ork_mm_run(c,wK[kv],M,Q,S);   /* QK^T -> S[M,L2]  (softmax on CPU, omitted: cheap) */
            ork_mm_run(c,wV[kv],M,P,O); }  /* PV   -> O[M,HD] */
    double dt=now()-t0;
    printf("NPU attention floor: M=%d  %d matmuls (%d layers x %d heads x 2) in %.0f ms\n",
           M,NL*NH*2,NL,NH,dt*1e3);
    printf("  vs CPU flash attention: ~264 ms (M=128) / ~1252 ms (M=256, pooled). %s\n",
           dt*1e3 < (M>=256?1252:264) ? "NPU floor WINS -> worth building" : "NPU floor LOSES (submit overhead) -> keep CPU");
    ork_npu_free(c); return 0;
}
