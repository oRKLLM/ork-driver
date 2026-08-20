/* jit_i4_i8_probe — Slice 1 of the tiered int4-park/JIT-int8 residence: validate the JIT int4->int8
 * materialization. Park a weight as int4 values, JIT-inflate to an int8 ork_w (ork_i4_mm_pack_to_i8, the
 * free CPU unpack + tile), run int8 on the NPU, verify correct + time the inflate vs a direct int8 pack.
 * All-ones int4 (value 1) -> int8 matmul == K everywhere (bit-exact gate).
 *   make jit_i4_i8_probe && sudo env ORK_MM_TIMEOUT=4000 ./jit_i4_i8_probe [iters]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

int main(int argc,char**argv){
    int iters=argc>1?atoi(argv[1]):20;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int K=3584, N=18944, M=1;                               /* 7B FFN gate (real width) */
    int8_t*A=malloc((size_t)M*K); memset(A,1,(size_t)M*K);
    int8_t*Bi4=malloc((size_t)K*N); memset(Bi4,1,(size_t)K*N);   /* int4 values [-8,7], all 1 (parked form) */
    int8_t*Bi8=malloc((size_t)K*N); memset(Bi8,1,(size_t)K*N);   /* int8 direct */
    int32_t*C=malloc((size_t)M*N*4);
    printf("jit_i4_i8_probe: K=%d N=%d M=%d (parked int4 -> JIT int8)\n",K,N,M);

    /* ---- direct int8 (baseline: pre-inflated) ---- */
    ork_npu_set_pack_domain(c,0); ork_w*w8=ork_i8_mm_pack(c,K,N,Bi8); if(!w8){printf("pack_i8 fail\n");return 1;}
    ork_i8_mm_run(c,w8,M,A,C);
    double t0=now_us(); for(int i=0;i<iters;i++) ork_i8_mm_run(c,w8,M,A,C); double run8=(now_us()-t0)/iters;
    int ok8=1; for(int i=0;i<M*N;i++)if(C[i]!=K){ok8=0;break;}
    printf("  int8 direct: run %8.1f us  bit-exact=%s\n", run8, ok8?"YES":"NO");

    /* ---- JIT: inflate parked int4 -> int8 ork_w, then run ---- */
    ork_npu_set_pack_domain(c,0);
    t0=now_us(); ork_w*wj=0; for(int i=0;i<iters;i++){ if(wj)ork_w_free(wj); wj=ork_i4_mm_pack_to_i8(c,K,N,Bi4); }
    double inflate=(now_us()-t0)/iters;
    if(!wj){printf("pack_i4_to_i8 fail\n");return 1;}
    memset(C,0,(size_t)M*N*4); ork_i8_mm_run(c,wj,M,A,C);
    t0=now_us(); for(int i=0;i<iters;i++) ork_i8_mm_run(c,wj,M,A,C); double runj=(now_us()-t0)/iters;
    int okj=1; for(int i=0;i<M*N;i++)if(C[i]!=K){okj=0;break;}
    printf("  JIT int4->int8: inflate %8.1f us + run %8.1f us  bit-exact=%s\n", inflate, runj, okj?"YES":"NO");
    printf("  ★ parked=int4 (half RAM); run matches int8 (%s); inflate/run ratio %.2f\n",
           (ok8&&okj&&run8>0&&runj>0)?"COHERENT":"CHECK", runj>0?inflate/runj:0);
    printf("  (Slice 1: JIT materialization %s. next: in-place inflate into reused domain buffer.)\n",
           (ok8&&okj)?"CORRECT":"FAILED");
    ork_npu_free(c); return (ok8&&okj)?0:2;
}
