/* tools/f16_gate_bench.c — decide (safely, with validated ops) whether a FUSED fp16 gate+SiLU beats the
 * baseline's int8 gate + CPU SiLU. Fused fp16-gate cost ~= the fp16 matmul time (silu rides the output stage
 * ~free). Baseline gate cost = int8 matmul + CPU transcendental silu over M*N. If t_fp16 < t_i8 + t_cpu_silu,
 * the fused fp16-gate wins (recovering the int8-silu PPL loss); else it can't. No new/untested regcmd (uses
 * ork_mm_run / ork_mm_run_i8), so no wedge risk.
 *   make f16_gate_bench && sudo ./f16_gate_bench [M] [K] [N] [iters]     (board only)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):128, K=argc>2?atoi(argv[2]):2048, N=argc>3?atoi(argv[3]):6144, it=argc>4?atoi(argv[4]):50;
    ork_npu*c=ork_npu_init(); if(!c){printf("no board\n");return 0;}
    /* fp16 weight + activation */
    ork_f16*Bf=malloc((size_t)K*N*2),*Af=malloc((size_t)M*K*2);
    for(size_t i=0;i<(size_t)K*N;i++)Bf[i]=(ork_f16)(((i*7)%13-6)*0.01f);
    for(size_t i=0;i<(size_t)M*K;i++)Af[i]=(ork_f16)(((i*11)%17-8)*0.01f);
    /* int8 weight + activation */
    int8_t*Bi=malloc((size_t)K*N),*Ai=malloc((size_t)M*K);
    for(size_t i=0;i<(size_t)K*N;i++)Bi[i]=(int8_t)((i*7)%13-6);
    for(size_t i=0;i<(size_t)M*K;i++)Ai[i]=(int8_t)((i*11)%17-8);

    ork_w*wf=ork_mm_pack(c,K,N,Bf), *wi=ork_mm_pack_i8(c,K,N,Bi);
    if(!wf||!wi){ printf("pack failed (wf=%p wi=%p)\n",(void*)wf,(void*)wi); return 2; }
    float*Cf=malloc((size_t)M*N*4); int32_t*Ci=malloc((size_t)M*N*4);

    /* warm */
    for(int w=0;w<3;w++){ ork_mm_run(c,wf,M,Af,Cf); ork_mm_run_i8(c,wi,M,Ai,Ci); }
    double t0=now_us(); for(int i=0;i<it;i++) ork_mm_run(c,wf,M,Af,Cf);   double t_f16=(now_us()-t0)/it;
    t0=now_us();        for(int i=0;i<it;i++) ork_mm_run_i8(c,wi,M,Ai,Ci); double t_i8 =(now_us()-t0)/it;
    /* CPU transcendental silu over M*N (what baseline pays for the gate) — 4 threads via OpenMP if built with it */
    float*g=malloc((size_t)M*N*4); for(size_t i=0;i<(size_t)M*N;i++)g[i]=(float)(Ci[i%((size_t)M*N)]*1e-4);
    volatile float sink=0; t0=now_us();
    for(int r=0;r<it;r++){ float acc=0; for(size_t i=0;i<(size_t)M*N;i++) acc+=g[i]/(1.0f+expf(-g[i])); sink=acc; }
    double t_silu=(now_us()-t0)/it; (void)sink;

    printf("=== gate M=%d K=%d N=%d (%d iters) ===\n",M,K,N,it);
    printf("  int8 matmul      : %8.1f us\n", t_i8);
    printf("  fp16 matmul      : %8.1f us   (%.2fx int8)\n", t_f16, t_f16/t_i8);
    printf("  CPU silu (M*N)   : %8.1f us\n", t_silu);
    printf("  BASELINE gate (int8 mm + CPU silu) : %8.1f us\n", t_i8 + t_silu);
    printf("  FUSED fp16 gate  (~fp16 mm, silu free): %8.1f us\n", t_f16);
    printf("  -> fp16-gate is %.2fx the baseline gate  (%s)\n",
           t_f16/(t_i8+t_silu), t_f16 < (t_i8+t_silu) ? "WINS" : "loses");
    ork_mm_free(c,wf); ork_mm_free(c,wi); ork_npu_free(c);
    return 0;
}
