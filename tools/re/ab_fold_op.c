/* ab_fold_op — #39 FULL-OP A/B: rkllm-style N-split fold (ork_i8_npu_fold_op) vs ork-normal (ork_i8_mm_run), for a
 * real wide attention-projection shape K=3584 N=3584, both 3-core. This is the honest end-to-end-representative
 * test: the fold N-splits N across the 3 cores (each core single-core-folds its ~1216 slice), ork-normal N-splits
 * the same op across 3 cores its own way. Decides whether the fold's compute-hiding beats its extra weight
 * re-streams under 3-core DRAM-BW contention. Verifies bit-exact output; reports avg time each.
 *   sudo env ORK_MM_TIMEOUT=800 ./ab_fold_op [reps] [N]      (default 5 3584)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "ork_npu.h"
extern int ork_i8_npu_fold_op(ork_npu*,int,int,const int8_t*,int,const int8_t*,int32_t*,int,double*);
static uint32_t rng=0x1234u; static int r7(void){ rng=rng*1664525u+1013904223u; return (int)((rng>>25)%7)-3; }
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec*1e-3; }

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int reps=argc>1?atoi(argv[1]):5;
    const int K=3584, N=argc>2?atoi(argv[2]):3584, Mmax=128;
    int Ms[]={36,72,100,128}; int nM=(int)(sizeof Ms/sizeof*Ms);
    int8_t *W=malloc((size_t)K*N),*A=malloc((size_t)Mmax*K);
    for(size_t i=0;i<(size_t)K*N;i++) W[i]=(int8_t)r7();
    for(size_t i=0;i<(size_t)Mmax*K;i++) A[i]=(int8_t)r7();
    int32_t *Cf=malloc((size_t)Mmax*N*4),*Cb=malloc((size_t)Mmax*N*4);

    ork_npu*c=ork_npu_init(); if(!c){ printf("init (board only)\n"); return 77; }
    ork_npu_set_core_budget(c,3);                       /* 3-core baseline (matches fold's N-split cores) */
    ork_w*ww=ork_i8_mm_pack(c,K,N,W); if(!ww){ printf("pack fail\n"); return 1; }
    { int8_t*a8=calloc((size_t)8*K,1); int32_t*z=calloc((size_t)8*N,4); ork_i8_mm_run(c,ww,8,a8,z); free(a8); free(z); }  /* prewarm */

    printf("FULL-OP A/B  K=%d N=%d (%.1f slices) 3-core reps=%d\n", K, N, N/1216.0, reps);
    printf("   M   base us    fold us    ratio   match\n");
    for(int i=0;i<nM;i++){ int M=Ms[i]; double ub=0,uf=0;
        int frc=ork_i8_npu_fold_op(c,K,N,W,M,A,Cf,reps,&uf);
        if(frc){ printf("  %3d  fold rc=%d\n",M,frc); continue; }
        ork_i8_mm_run(c,ww,M,A,Cb); double t0=now_us(); for(int r=0;r<reps;r++) ork_i8_mm_run(c,ww,M,A,Cb); ub=(now_us()-t0)/reps;
        long mm=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++) if(Cf[(size_t)m*N+n]!=Cb[(size_t)m*N+n]) mm++;
        printf("  %3d  %8.1f   %8.1f   %5.2fx   %s\n", M, ub, uf, (uf>0?ub/uf:0), mm?"MISMATCH!":"ok");
    }
    ork_mm_free(c,ww); free(W);free(A);free(Cf);free(Cb); ork_npu_free(c); return 0;
}
