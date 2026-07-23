/* chainbench_probe — benchmark the current iteration: the HW-chained attention core [QK^T->exp->reduce,e.V]
 * as ONE coalesced submit vs the per-submit floor. The 4-op core runs in 1 submit; the separate-op baseline
 * pays ~4x the submit overhead. Times the chain and a single int8 matmul submit (the floor) to quantify the
 * amortization.  sudo env ORK_MM_TIMEOUT=3000 timeout 90 ./chainbench_probe [Nq] [iters]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
static uint32_t g=0x9e10u; static int q2(void){ g=g*1664525u+1013904223u; return (int)((g>>27)%3); }
static int vv(void){ g=g*1664525u+1013904223u; return (int)((g>>27)%5)-2; }
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
int main(int argc,char**argv){
    int Nq=argc>1?atoi(argv[1]):32, iters=argc>2?atoi(argv[2]):200, d=128, Nk=512, Kp=512, dv=128;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("chainbench_probe: Nq=%d Nk=%d dv=%d iters=%d\n",Nq,Nk,dv,iters);
    int8_t *Q=malloc((size_t)Nq*d), *K=malloc((size_t)Nk*d), *V=malloc((size_t)Nk*dv);
    for(size_t i=0;i<(size_t)Nq*d;i++) Q[i]=(int8_t)q2();
    for(size_t i=0;i<(size_t)Nk*d;i++) K[i]=(int8_t)(-q2());
    for(size_t i=0;i<(size_t)Nk*dv;i++) V[i]=(int8_t)vv();
    int8_t *Qp=calloc((size_t)Nq*Kp,1), *KTp=calloc((size_t)Kp*Nk,1);
    for(int i=0;i<Nq;i++)for(int k=0;k<d;k++) Qp[(size_t)i*Kp+k]=Q[(size_t)i*d+k];
    for(int k=0;k<d;k++)for(int j=0;j<Nk;j++) KTp[(size_t)k*Nk+j]=K[(size_t)j*d+k];
    int r_mult=0x4000, r_shift=16; double in_scale=0.0625, out_scale=1.0/127.0;
    ork_w *w_kt=ork_mm_pack_i8(c,Kp,Nk,KTp), *w_ones, *w_v;
    { int8_t *ones=malloc((size_t)Nk*32); memset(ones,1,(size_t)Nk*32); w_ones=ork_mm_pack_i8(c,Nk,32,ones); free(ones); }
    w_v=ork_mm_pack_i8(c,Nk,dv,V);
    if(!w_kt||!w_ones||!w_v){ printf("pack fail\n"); return 2; }
    int32_t *scb=calloc((size_t)Nq*Nk,4), *eb=calloc((size_t)Nq*Nk,4), *ss=calloc((size_t)Nq*32,4), *avb=calloc((size_t)Nq*dv,4);
    ork_mm_task_i8 tasks[4] = { { w_kt,Nq,Qp,scb }, { w_kt,Nq,(int8_t*)scb,eb }, { w_ones,Nq,(int8_t*)eb,ss }, { w_v,Nq,(int8_t*)eb,avb } };
    ork_chain_op ops[4] = { {1,-1,0,r_mult,r_shift}, {2,0,0,0,0}, {0,1,0,0,0}, {0,1,0,0,0} };

    /* warm */
    if(ork_mm_run_chain_i8_ffn_exp(c,4,tasks,ops,in_scale,out_scale)){ printf("chain warm FAIL\n"); return 1; }
    /* time the coalesced 4-op HW chain (ONE submit) */
    double t0=now_us(); for(int i=0;i<iters;i++) ork_mm_run_chain_i8_ffn_exp(c,4,tasks,ops,in_scale,out_scale);
    double chain_us=(now_us()-t0)/iters;
    /* per-submit floor: a single int8 matmul submit (QK^T alone) via run_chain S=1 -> run_i8 */
    ork_mm_task_i8 one[1] = { { w_kt, Nq, Qp, scb } };
    ork_mm_run_chain_i8(c,1,one);   /* warm */
    t0=now_us(); for(int i=0;i<iters;i++) ork_mm_run_chain_i8(c,1,one);
    double one_us=(now_us()-t0)/iters;

    printf("  HW chain [QK^T->exp->reduce,e.V] (4 ops, ONE submit): %.1f us/iter (%.0f iter/s)\n", chain_us, 1e6/chain_us);
    printf("  single int8 matmul submit (per-submit floor):          %.1f us/iter\n", one_us);
    printf("  separate-op estimate (~4 submits):                     %.1f us  => chain speedup ~%.2fx\n", 4*one_us, (4*one_us)/chain_us);
    printf("PASS — benchmark done\n");
    ork_mm_free(c,w_kt); ork_mm_free(c,w_ones); ork_mm_free(c,w_v); ork_npu_free(c);
    return 0;
}
