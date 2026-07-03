/* tools/silu_bench.c — matmul-level benchmark of the fused-SiLU output stage.
 *
 * Question: does folding SiLU into the matmul's output stage cost anything, vs a plain matmul plus the
 * CPU-silu handoff it replaces? Measures (warm, LUT amortized — the LUT-load is a one-time prologue):
 *   - fused-silu compute-submit time  (SiLU applied on-chip in the output stage)
 *   - plain int8-output matmul submit  (linear requant, the matmul baseline)
 *   - CPU silu over the M*N outputs    (the handoff the fused stage removes)
 *
 * Finding (RK3588, 2026-07-03): on-NPU SiLU overhead ~= 0 (fused submit == matmul submit within noise) —
 * SiLU is FREE in the output stage — and it eliminates the CPU-silu pass (~5us at M=1 up to ~158us at
 * M=32/16k-out) plus the dequant/requant roundtrip. Win scales with M (largest in prefill), and is the
 * enabler for keeping a whole FFN NPU-resident (chaining synergy: no CPU handoff between links).
 *
 *   make silu_bench && sudo ./silu_bench            (board only)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static double silu(double x){ return x/(1.0+exp(-x)); }
static signed char *A,*B; static int8_t *C8; static int16_t lut[1030];

static void bench(ork_npu*c,int M,int K,int N){
    A=malloc((size_t)M*K); B=calloc(1,(size_t)K*N); C8=malloc((size_t)M*N);
    for(int i=0;i<M*K;i++)A[i]=(signed char)((i*7)%5-2);
    for(int i=0;i<K*N;i++)B[i]=(signed char)((i*3)%5-2);
    double usf=0,usl=0;
    ork_npu_probe_i8_silu_cfg(c,M,K,N,A,B,0x4000,0x10,0u,0xffffc000u,0x56391300u,lut,1030,C8,&usf);
    ork_npu_probe_i8_out8(c,M,K,N,A,B,0x4000,14,C8,&usl);
    static float tmp[64*512]; int MN=M*N; if(MN>64*512)MN=64*512;
    double t0=now_us(); volatile float s=0;
    for(int r=0;r<50;r++){ for(int i=0;i<MN;i++) tmp[i]=(float)silu((C8[i])*0.02); s+=tmp[0]; }
    double uscpu=(now_us()-t0)/50;
    printf("  M=%-3d K=%d N=%d: fused-silu=%.1fus  matmul=%.1fus  (silu overhead=%.1fus)  CPU-silu(%d out)=%.1fus\n",
           M,K,N,usf,usl,usf-usl,MN,uscpu);
    free(A);free(B);free(C8);
}
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("no board\n");return 0;}
    printf("fused-SiLU matmul-level benchmark (warm, LUT amortized):\n");
    bench(c,1,2048,512); bench(c,32,2048,512); bench(c,1,4096,512); bench(c,32,4096,512);
    ork_npu_free(c); return 0;
}
