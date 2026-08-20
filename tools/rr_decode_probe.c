// Decode RR payoff probe: for a group of INDEPENDENT int8 matmuls (like QKV, gate/up) at a given M,
// compare running them sequentially via ork_i8_mm_run (each a single-core submit — the current backbone
// path) vs one ork_i8_mm_run_stream (round-robin across the NPU cores). Answers: is decode core-bound
// (RR wins ~Ncores) or submit-floor/work bound (RR ~flat or loses)? Single domain (default).
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

static ork_w *mkw(ork_npu *c, int K, int N){
    int8_t *B = malloc((size_t)K*N);
    for (size_t i=0;i<(size_t)K*N;i++) B[i] = (int8_t)((i*131u+7u)&0x7f);
    ork_w *w = ork_i8_mm_pack(c, K, N, B); free(B); return w;
}

static void bench(ork_npu *c, const char *label, int K, const int *Ns, int ng, int M, int iters){
    int8_t *A = calloc((size_t)M*K, 1);
    for (size_t i=0;i<(size_t)M*K;i++) A[i] = (int8_t)((i*7u+3u)&0x3f);
    ork_w *w[8]; int32_t *C[8]; ork_mm_task_i8 tk[8];
    for (int i=0;i<ng;i++){ w[i]=mkw(c,K,Ns[i]); C[i]=calloc((size_t)M*Ns[i],4); tk[i]=(ork_mm_task_i8){w[i],M,A,C[i]}; }
    for (int i=0;i<ng;i++) ork_i8_mm_run(c,w[i],M,A,C[i]);   // warm
    ork_i8_mm_run_stream(c,ng,tk);
    double t0=now_us(); for(int it=0;it<iters;it++) for(int i=0;i<ng;i++) ork_i8_mm_run(c,w[i],M,A,C[i]);
    double t_seq=(now_us()-t0)/iters;
    t0=now_us(); for(int it=0;it<iters;it++) ork_i8_mm_run_stream(c,ng,tk);
    double t_rr=(now_us()-t0)/iters;
    printf("%-14s M=%-3d ng=%d: seq(single-core %dx run_i8)=%7.1fus  RR(run_stream)=%7.1fus  speedup=%.2fx\n",
           label, M, ng, ng, t_seq, t_rr, t_seq/t_rr);
    for (int i=0;i<ng;i++){ ork_mm_free(c,w[i]); free(C[i]); } free(A);
}

int main(void){
    ork_npu *c = ork_npu_init(); if(!c){ printf("no NPU\n"); return 1; }
    printf("cores=%d\n", ork_npu_cores(c));
    int qkv[3] = {2048,1024,1024};    // Qwen3-1.7B GQA: attn_q [2048,2048]->N=2048? use hidden=2048 head split
    int qkv2[3]= {2048,2048,2048};
    int gu[2]  = {6144,6144};         // ffn gate/up (N=6144, exceeds the 4096 chain gate but run_stream handles it)
    for (int M=1; M<=8; M*=8){        // M=1 (decode) and M=8 (small batch / verify)
        bench(c,"QKV(gqa)",2048,qkv,3,M,2000);
        bench(c,"QKV(mha)",2048,qkv2,3,M,2000);
        bench(c,"gate/up",2048,gu,2,M,1500);
    }
    ork_npu_free(c); return 0;
}
