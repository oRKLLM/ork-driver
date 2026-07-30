/* ws_xition_probe — measure the per-task HW-chain TRANSITION cost vs the per-INDIVIDUAL-submit cost
 * for uniform int8 matmul tiles. This is the decisive empirical input to ws_model.c: it decides whether
 * weight-stationary (which needs ~100x more tiles than the fold) can escape the submit-bound wall by
 * HW-chaining the tiles into one submit (task_number=S, HW walks the 0x0010/0x0014 descriptor).
 *
 *   model "individual" per-tile cost  == single-submit floor  (this probe: t_single)
 *   model "chained"    per-tile cost  == chain marginal/task   (this probe: slope of chain(S))
 *
 * Method: tiny task (K=512 N=64 M=1 -> ~32KB weight, near-zero DMA) so the measured slope is the fixed
 * overhead, not bandwidth. All-ones operands => C[m][n]==K (self-validating). Also times a REALISTIC
 * WS tile (K=256 N=192) single vs chained to see if the transition hides under real weight-DMA (overlap).
 * ork_npu_dma_rw() confirms the DMA bytes/submit so we know which regime we're in.
 *
 * BOARD: sudo env ORK_MM_TIMEOUT=3000 timeout 180 ./ws_xition_probe [iters]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }

/* build S tasks all sharing weight w (K,N), each M rows of all-ones A, own C. returns 0 ok */
static int build_tasks(ork_npu *c, int K, int N, int M, int S, ork_w **wo, ork_mm_task_i8 *tasks, int8_t **Ao, int32_t **Cs){
    int8_t *B = malloc((size_t)K*N); memset(B, 1, (size_t)K*N);
    ork_w *w = ork_mm_pack_i8(c, K, N, B); free(B);
    if(!w) return -1;
    int8_t *A = malloc((size_t)M*K); memset(A, 1, (size_t)M*K);
    *wo = w; *Ao = A;
    for(int i=0;i<S;i++){ Cs[i] = calloc((size_t)M*N, 4); tasks[i].w=w; tasks[i].M=M; tasks[i].A=A; tasks[i].C=Cs[i]; }
    return 0;
}
static int verify(int32_t **Cs, int S, int M, int N, int K){
    for(int i=0;i<S;i++) for(size_t e=0;e<(size_t)M*N;e++) if(Cs[i][e]!=K) return 1;
    return 0;
}

int main(int argc, char**argv){
    int iters = argc>1?atoi(argv[1]):100;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu *c = ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    printf("ws_xition_probe: iters=%d  BW/overhead separation via tiny (K512 N64 M1) tasks\n", iters);

    /* ---- (1) single-submit floor: tiny task, one ork_mm_run_i8 ---- */
    { int K=512,N=64,M=1,S=1; ork_w*w; int8_t*A; int32_t*Cs[1]; ork_mm_task_i8 tk[1];
      if(build_tasks(c,K,N,M,S,&w,tk,&A,Cs)){ printf("pack fail (tiny)\n"); return 2; }
      ork_mm_run_i8(c,w,M,A,Cs[0]);                                     /* warm */
      uint64_t d0=ork_npu_dma_rw(c);
      double t0=now_us(); for(int i=0;i<iters;i++) ork_mm_run_i8(c,w,M,A,Cs[0]); double t=(now_us()-t0)/iters;
      uint64_t dma=(ork_npu_dma_rw(c)-d0)/iters;
      printf("\n[single submit ] tiny K512N64M1: %.2f us/submit   dma=%llu B/submit  C==K:%s\n",
             t,(unsigned long long)dma, verify(Cs,S,M,N,K)?"NO":"yes");
    }

    /* ---- (2) HW chain: same tiny task chained S ways, slope over S = per-task transition ---- */
    printf("\n[HW chain      ] tiny K512N64M1, task_number=S:\n");
    int Ss[]={1,2,4,8,16,32}; double tS[8]={0}; int nS=sizeof Ss/sizeof*Ss;
    for(int j=0;j<nS;j++){ int K=512,N=64,M=1,S=Ss[j]; ork_w*w; int8_t*A; int32_t*Cs[64]; ork_mm_task_i8 tk[64];
      if(build_tasks(c,K,N,M,S,&w,tk,&A,Cs)){ printf("  S=%d pack fail\n",S); continue; }
      int rc=ork_mm_run_chain_i8(c,S,tk);                              /* warm */
      if(rc){ printf("  S=%-2d chain rc=%d (rejected — max task_number?)\n",S,rc); for(int i=0;i<S;i++)free(Cs[i]); free(A); continue; }
      uint64_t d0=ork_npu_dma_rw(c);
      double t0=now_us(); for(int i=0;i<iters;i++) ork_mm_run_chain_i8(c,S,tk); double t=(now_us()-t0)/iters;
      uint64_t dma=(ork_npu_dma_rw(c)-d0)/iters; tS[j]=t;
      printf("  S=%-2d  %8.2f us/chain  = %6.2f us/task   dma=%7llu B/chain   C==K:%s\n",
             S, t, t/S, (unsigned long long)dma, verify(Cs,S,M,N,K)?"NO":"yes");
      for(int i=0;i<S;i++){ free(Cs[i]); } free(A);
    }
    if(tS[0]>0 && tS[nS-1]>0){
      double slope=(tS[nS-1]-tS[0])/(Ss[nS-1]-Ss[0]);
      printf("  --> chain per-task TRANSITION slope = %.2f us/task (intercept ~%.2f us fixed submit)\n",
             slope, tS[0]-slope*Ss[0]);
    }

    /* ---- (3) realistic WS tile (K256 N186) single vs chained: does transition hide under weight-DMA? ---- */
    printf("\n[realistic tile] K256 N192 M36 (ws_model best chip config):\n");
    { int K=256,N=192,M=36; ork_w*w; int8_t*A; int32_t*Cs[64]; ork_mm_task_i8 tk[64];
      if(!build_tasks(c,K,N,M,1,&w,tk,&A,Cs)){
        ork_mm_run_i8(c,w,M,A,Cs[0]);
        uint64_t d0=ork_npu_dma_rw(c); double t0=now_us(); for(int i=0;i<iters;i++) ork_mm_run_i8(c,w,M,A,Cs[0]);
        double ts=(now_us()-t0)/iters; uint64_t dma=(ork_npu_dma_rw(c)-d0)/iters;
        printf("  single: %.2f us  dma=%llu B  (weight=%dKB, DMA-floor=%.2fus @11GB/s)\n",
               ts,(unsigned long long)dma, K*N/1024, (K*N)/11000.0);
        free(Cs[0]);
        int S=16; for(int i=0;i<S;i++){ Cs[i]=calloc((size_t)M*N,4); tk[i].w=w; tk[i].M=M; tk[i].A=A; tk[i].C=Cs[i]; }
        int rc=ork_mm_run_chain_i8(c,S,tk);
        if(!rc){ uint64_t e0=ork_npu_dma_rw(c); double u0=now_us(); for(int i=0;i<iters;i++) ork_mm_run_chain_i8(c,S,tk);
          double tc=(now_us()-u0)/iters; uint64_t dma2=(ork_npu_dma_rw(c)-e0)/iters;
          printf("  chain S=16: %.2f us = %.2f us/task  dma=%llu B  C==K:%s\n",
                 tc, tc/S, (unsigned long long)dma2, verify(Cs,S,M,N,K)?"NO":"yes");
          printf("  --> per-task chained %.2fus vs single-submit %.2fus  (transition %s the %0.2fus weight-DMA)\n",
                 tc/S, ts, (tc/S < ts)?"amortizes under":"exceeds", (K*N)/11000.0);
        } else printf("  chain S=16 rc=%d\n",rc);
        for(int i=0;i<S;i++){ free(Cs[i]); } free(A);
      }
    }
    printf("\nDONE\n");
    return 0;
}
