/* chainrr_bench_probe — throughput: N attention-core chains via ork_mm_run_chains_rr (concurrent, all cores)
 * vs the same N run sequentially on core 0 (ork_i8_mm_run_chain_ffn_exp). Reports chains/s + speedup. Steady
 * state: warm cores + one warmup pass to populate the per-core LUT/task caches, then R timed repeats. Run with
 * ORK_CHAIN_LUT_STICKY=1 so the ~148us/core LUT-load is amortized (the prefill-queue case).
 *   sudo env ORK_CHAIN_LUT_STICKY=1 ORK_MM_TIMEOUT=3000 ./chainrr_bench_probe [nchains] [repeats] [Nq]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3 + t.tv_nsec/1e6; }
int main(int argc,char**argv){
    int NCH=argc>1?atoi(argv[1]):24, REP=argc>2?atoi(argv[2]):5, Nq=argc>3?atoi(argv[3]):32;
    int d=128, Nk=512, Kp=512, dv=128;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("chainrr_bench_probe: %d chains x %d repeats, Nq=%d Nk=%d dv=%d\n",NCH,REP,Nq,Nk,dv);
    int r_mult=0x4000, r_shift=16; double in_scale=0.0625, out_scale=1.0/127.0;
    ork_chain_op ops[4] = { {1,-1,0,r_mult,r_shift}, {2,0,0,0,0}, {0,1,0,0,0}, {0,1,0,0,0} };
    uint32_t rng=0x1234u;
    #define R3 (rng=rng*1664525u+1013904223u, (int)((rng>>27)%3))
    #define RV (rng=rng*1664525u+1013904223u, (int)((rng>>27)%5)-2)
    /* shared read-only weights (a bench cares about dispatch, not distinct data) */
    int8_t *Q=malloc((size_t)Nq*d), *K=malloc((size_t)Nk*d), *V=malloc((size_t)Nk*dv);
    for(size_t i=0;i<(size_t)Nq*d;i++)Q[i]=(int8_t)R3;
    for(size_t i=0;i<(size_t)Nk*d;i++)K[i]=(int8_t)(-R3);
    for(size_t i=0;i<(size_t)Nk*dv;i++)V[i]=(int8_t)RV;
    int8_t *Qp=calloc((size_t)Nq*Kp,1), *KTp=calloc((size_t)Kp*Nk,1);
    for(int i=0;i<Nq;i++)for(int k=0;k<d;k++) Qp[(size_t)i*Kp+k]=Q[(size_t)i*d+k];
    for(int k=0;k<d;k++)for(int j=0;j<Nk;j++) KTp[(size_t)k*Nk+j]=K[(size_t)j*d+k];
    ork_w *w_kt=ork_i8_mm_pack(c,Kp,Nk,KTp), *w_v=ork_i8_mm_pack(c,Nk,dv,V), *w_ones;
    { int8_t *o=malloc((size_t)Nk*32); memset(o,1,(size_t)Nk*32); w_ones=ork_i8_mm_pack(c,Nk,32,o); free(o); }
    if(!w_kt||!w_v||!w_ones){ printf("pack fail\n"); return 2; }
    /* per-chain IO buffers (intermediates MUST be per-chain so concurrent chains don't collide) */
    ork_mm_task_i8 **chains=calloc(NCH,sizeof*chains); int *S=calloc(NCH,sizeof*S);
    for(int ch=0;ch<NCH;ch++){
        int32_t *scb=calloc((size_t)Nq*Nk,4), *eb=calloc((size_t)Nq*Nk,4), *ss=calloc((size_t)Nq*32,4), *avb=calloc((size_t)Nq*dv,4);
        chains[ch]=malloc(4*sizeof(ork_mm_task_i8)); S[ch]=4;
        chains[ch][0]=(ork_mm_task_i8){ w_kt,Nq,Qp,scb };
        chains[ch][1]=(ork_mm_task_i8){ w_kt,Nq,(int8_t*)scb,eb };
        chains[ch][2]=(ork_mm_task_i8){ w_ones,Nq,(int8_t*)eb,ss };
        chains[ch][3]=(ork_mm_task_i8){ w_v,Nq,(int8_t*)eb,avb };
    }
    /* warm all cores (prefill precondition) + one warmup pass each path to populate per-core caches */
    { int32_t *wc=calloc((size_t)Nq*Nk,4); ork_mm_task_i8 wt={ w_kt, Nq, Qp, wc }; ork_i8_mm_run_chain(c,1,&wt); free(wc); }
    ork_mm_run_chains_rr(c,NCH,(const ork_mm_task_i8*const*)chains,S,ops,in_scale,out_scale);
    for(int ch=0;ch<NCH;ch++) ork_i8_mm_run_chain_ffn_exp(c,S[ch],chains[ch],ops,in_scale,out_scale);

    /* timed: concurrent round-robin */
    double t0=now_ms();
    for(int r=0;r<REP;r++){ int rc=ork_mm_run_chains_rr(c,NCH,(const ork_mm_task_i8*const*)chains,S,ops,in_scale,out_scale); if(rc){printf("rr rc=%d\n",rc);return 1;} }
    double rr_ms=(now_ms()-t0)/REP;
    /* timed: single-core sequential (core 0) */
    t0=now_ms();
    for(int r=0;r<REP;r++){ for(int ch=0;ch<NCH;ch++){ int rc=ork_i8_mm_run_chain_ffn_exp(c,S[ch],chains[ch],ops,in_scale,out_scale); if(rc){printf("sc rc=%d\n",rc);return 1;} } }
    double sc_ms=(now_ms()-t0)/REP;

    printf("  single-core (core0 seq): %8.2f ms/pass  %8.1f chains/s\n", sc_ms, NCH*1e3/sc_ms);
    printf("  round-robin (3-core rr): %8.2f ms/pass  %8.1f chains/s\n", rr_ms, NCH*1e3/rr_ms);
    printf("  speedup: %.2fx\n", sc_ms/rr_ms);
    printf("PASS — benchmark done\n");
    ork_mm_free(c,w_kt); ork_mm_free(c,w_v); ork_mm_free(c,w_ones); ork_npu_free(c);
    return 0;
}
