/* ork_pc_bench — regime A: precompiled-program cache vs synth-every-call.
 * A fixed chain of S int8 matmuls (M=1, pinned A/C buffers). Compares per-token wall time:
 *   T_synth  : ork_dyn_begin+end  — re-synthesizes all S programs every call (the current build cost)
 *   T_pc     : ork_pc_run         — compiled once, re-run with only the activation contents refreshed
 * The delta is the host-build floor (~synth+validate+memcpy) that the precompiled cache eliminates.
 *   make ork_pc_bench && sudo ./ork_pc_bench [S=16] [iters=50]
 * (NPU op; run alone.)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static int check(const int32_t*O,int N,int Nn,int K){ int ok=0; for(int i=0;i<N;i++) if(O[(size_t)i*Nn+(Nn-1)]==K) ok++; return ok; }

int main(int argc,char**argv){
    int S=argc>1?atoi(argv[1]):16, iters=argc>2?atoi(argv[2]):50, K=512, N=512;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int8_t*A=(int8_t*)malloc(K); memset(A,1,K);              /* fixed A source (contents would change per token) */
    int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    ork_w*w=ork_i8_mm_pack(c,K,N,B); if(!w){printf("pack fail\n");return 1;}
    int32_t*O=(int32_t*)ork_dma_alloc(c,(size_t)S*N*sizeof(int32_t)); if(!O){printf("dma fail\n");return 1;}
    ork_mm_task_i8*tk=malloc(sizeof(*tk)*S);
    for(int i=0;i<S;i++){ tk[i].w=w; tk[i].M=1; tk[i].A=A; tk[i].C=O+(size_t)i*N; }
    printf("ork_pc_bench: S=%d int8 matmuls (M=1,K=%d,N=%d), iters=%d\n",S,K,N,iters);

    /* --- synth-every-call (ork_dyn_begin), single-core --- */
    { ork_dyn_chain*h=ork_dyn_begin(c,S,tk); ork_dyn_end(h); }  /* warm */
    double t=now_us();
    for(int it=0;it<iters;it++){ ork_dyn_chain*h=ork_dyn_begin(c,S,tk); ork_dyn_end(h); }
    double T_synth=(now_us()-t)/iters; int ok_s=check(O,S,N,K);

    /* --- precompiled (compile once, run many) --- */
    double tc=now_us(); ork_pc_chain*pc=ork_pc_compile(c,S,tk); double T_compile=now_us()-tc;
    if(!pc){printf("  compile failed\n"); ork_npu_free(c); return 1;}
    ork_pc_run(pc);  /* warm */
    t=now_us();
    for(int it=0;it<iters;it++) ork_pc_run(pc);
    double T_pc=(now_us()-t)/iters; int ok_pc=check(O,S,N,K);
    ork_pc_free(pc);

    printf("  T_synth (build+run every call) = %.1fus/token (%s)\n", T_synth, ok_s==S?"ok":"WRONG");
    printf("  T_compile (once)               = %.1fus\n", T_compile);
    printf("  T_pc (run precompiled)         = %.1fus/token (%s)\n", T_pc, ok_pc==S?"ok":"WRONG");
    printf("  ★ precompiled speedup = %.2fx ; per-token build cost removed = %.1fus (amortized after %.0f tokens)\n",
           T_synth/T_pc, T_synth-T_pc, (T_synth>T_pc)?T_compile/(T_synth-T_pc):0.0);
    ork_npu_free(c); return (ok_s==S&&ok_pc==S)?0:2;
}
