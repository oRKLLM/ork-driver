/* tools/mode_switch_probe.c — reproduce the int8<->fp16 mode-switch instability (blocker b) in ISOLATION.
 *
 * The selective/mixed FFN (int8 layers + fp16 layers) STALLS/WEDGES at scale, while uniform all-fp16 is
 * stable. Hypothesis: entering int8 from fp16 does RKNPU_ACT_RESET + re-warm EVERY transition (NOTHRASH only
 * covers int<->int, not fp16), and that per-boundary reset thrash wedges at large M. This alternates an int8
 * matmul and an fp16 matmul for `iters` iterations at a given M, printing progress each step so the LAST
 * printed line tells you exactly where it wedges. Run up an M ladder to find the smallest wedging M:
 *   make mode_switch_probe && for M in 8 64 128 256 512; do echo "== M=$M =="; sudo timeout 60 ./mode_switch_probe 30 $M; done
 * (board only; a wedge => board hard-hangs regardless of timeout — power-cycle to recover.)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>

typedef ork_f16 f16;

/* mode: 0 = int8 <-> fp16(f16_silu) alternate (basic mode-switch)
 *       1 = PLAIN fp16 ork_mm_run only (the CPUSILU gate's matmul call) — no mode switch
 *       2 = PLAIN fp16 ork_mm_run with core-budget SWITCH each iter (1 -> cores), mimicking CPUSILU gate/up */
int main(int argc,char**argv){
    int iters=argc>1?atoi(argv[1]):30, M=argc>2?atoi(argv[2]):512, mode=argc>3?atoi(argv[3]):0;
    const int K=2048, Ni=2048, Nf=3072;   /* fp16 single-tile: K<=2048 (Sk=1), Nf<=nmax (Sn=1) */
    ork_npu*c=ork_npu_init(); if(!c){printf("no board\n");return 0;}
    printf("mode_switch_probe iters=%d M=%d (int8[%dx%d] <-> fp16[%dx%d] alternating)\n",iters,M,K,Ni,K,Nf);

    int8_t*Bi=malloc((size_t)K*Ni); for(size_t i=0;i<(size_t)K*Ni;i++)Bi[i]=(int8_t)((i%7)-3);
    if(mode==4) ork_npu_set_pack_domain(c,0);   /* mode 4: int8 weight in domain 0 ... */
    ork_w*wi=ork_mm_pack_i8(c,K,Ni,Bi); if(!wi){printf("int8 pack fail\n");return 2;}
    int8_t*Ai=malloc((size_t)M*K); for(size_t i=0;i<(size_t)M*K;i++)Ai[i]=(int8_t)((i%5)-2);
    int32_t*Ci=malloc((size_t)M*Ni*4);

    f16*Bf=malloc((size_t)K*Nf*2); for(size_t i=0;i<(size_t)K*Nf;i++)Bf[i]=(f16)(0.001f*(float)((i%9)-4));
    if(mode==4) ork_npu_set_pack_domain(c,1);   /* ... fp16 weight in domain 1 -> runs trigger dom_activate */
    ork_w*wf=ork_mm_pack(c,K,Nf,Bf); if(!wf){printf("fp16 pack fail\n");return 2;}
    if(mode==4) ork_npu_set_pack_domain(c,-1);
    f16*Af=malloc((size_t)M*K*2); for(size_t i=0;i<(size_t)M*K;i++)Af[i]=(f16)(0.01f*(float)((i%11)-5));
    float*Cf=malloc((size_t)M*Nf*4);

    int ncore=ork_npu_cores(c);
    printf("  mode=%d (%s) ncore=%d\n", mode,
        mode==0?"int8<->f16_silu":mode==1?"plain-fp16 ork_mm_run":"plain-fp16 + core-budget switch", ncore);
    for(int it=0; it<iters; it++){
        int r1=0,r2=0;
        if(mode==0 || mode==4){   /* mode 4 = same calls, but weights in 2 domains -> dom_activate each run */
            printf("  iter %2d int8...",it); fflush(stdout); r1=ork_mm_run_i8(c,wi,M,Ai,Ci);
            printf(" f16silu...");          fflush(stdout); r2=ork_mm_run_f16_silu(c,wf,M,Af,Cf,0,0xffffc000u,0x56391100u,NULL,0);
        } else if(mode==1){
            printf("  iter %2d f16run...",it); fflush(stdout); r2=ork_mm_run(c,wf,M,Af,Cf);
        } else if(mode==2){
            printf("  iter %2d cb1 f16run...",it); fflush(stdout);
            ork_npu_set_core_budget(c,1);     r1=ork_mm_run(c,wf,M,Af,Cf);
            printf(" cbN f16run...");          fflush(stdout);
            ork_npu_set_core_budget(c,ncore); r2=ork_mm_run(c,wf,M,Af,Cf);
        } else {   /* mode 3: int8 CHAIN <-> fp16 — the actual model mixed path (chain "always resets") */
            ork_mm_task_i8 t={wi,M,Ai,Ci};
            printf("  iter %2d chain_i8...",it); fflush(stdout); r1=ork_mm_run_chain_i8(c,1,&t);
            printf(" f16silu...");              fflush(stdout); r2=ork_mm_run_f16_silu(c,wf,M,Af,Cf,0,0xffffc000u,0x56391100u,NULL,0);
        }
        printf(" ok (r1=%d r2=%d)\n",r1,r2); fflush(stdout);
        if(r1||r2){ printf("  -> FAIL at iter %d (r1=%d r2=%d)\n",it,r1,r2); break; }
    }
    printf("DONE (survived without wedge)\n");
    ork_mm_free(c,wi); ork_mm_free(c,wf); ork_npu_free(c);
    return 0;
}
