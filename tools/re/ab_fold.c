/* ab_fold — #39 A/B: rkllm token-fold (ork_npu_fold_run_i8) vs ork-normal int8 matmul (ork_mm_run_i8), K=3584
 * N=1216. Verifies the fold output matches the baseline (identical C checksum) and reports avg time each.
 *   ./ab_fold both [reps] [ncore]   baseline + fold in ONE context (tests the shared-context fix; A/B in-process)
 *   ./ab_fold fold [reps] [ncore]   fold only  (separate-process fallback)
 *   ./ab_fold base [reps] [ncore]   baseline only
 * ncore (1 or 3) sets the fold's round-robin cores AND the baseline's core budget for a fair compare.
 * fold us = avg submit (pure NPU); base us = full ork_mm_run_i8 call. sudo env ORK_MM_TIMEOUT=800 ./ab_fold both 5 3
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "ork_npu.h"
extern int ork_npu_fold_run_i8(ork_npu*,int,int,const int8_t*,int,const int8_t*,int32_t*,int,int,double*);
static uint32_t rng=0x1234u; static int r7(void){ rng=rng*1664525u+1013904223u; return (int)((rng>>25)%7)-3; }
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static uint64_t fnv(const int32_t*p,size_t n){ uint64_t h=1469598103934665603ull; const uint8_t*b=(const uint8_t*)p;
    for(size_t i=0;i<n*4;i++){ h^=b[i]; h*=1099511628211ull; } return h; }

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    const char*mode=argc>1?argv[1]:"both";
    int reps=argc>2?atoi(argv[2]):5;
    int ncore=argc>3?atoi(argv[3]):1;
    int dofold = strcmp(mode,"base")!=0, dobase = strcmp(mode,"fold")!=0;
    const int K=3584,N=1216, Mmax=128;
    int Ms[]={36,72,100,128}; int nM=(int)(sizeof Ms/sizeof*Ms);
    int8_t *W=malloc((size_t)K*N),*A=malloc((size_t)Mmax*K);
    for(size_t i=0;i<(size_t)K*N;i++) W[i]=(int8_t)r7();
    for(size_t i=0;i<(size_t)Mmax*K;i++) A[i]=(int8_t)r7();
    int32_t *Cf=malloc((size_t)Mmax*N*4),*Cb=malloc((size_t)Mmax*N*4);

    ork_npu*c=ork_npu_init(); if(!c){ printf("init (board only)\n"); return 77; }
    ork_npu_set_core_budget(c, ncore>=3?3:1);
    ork_w*ww=ork_mm_pack_i8(c,K,N,W); if(!ww){ printf("pack fail\n"); return 1; }
    { int8_t*a8=calloc((size_t)8*K,1); int32_t*z=calloc((size_t)8*N,4); ork_mm_run_i8(c,ww,8,a8,z); free(a8); free(z); }  /* prewarm: activate domain + warm */

    printf("A/B mode=%s ncore=%d K=%d N=%d reps=%d\n", mode, ncore, K, N, reps);
    printf("   M   base us    fold us    ratio   match\n");
    for(int i=0;i<nM;i++){ int M=Ms[i]; double ub=0,uf=0; uint64_t hb=0,hf=0; int frc=0;
        if(dobase){ ork_mm_run_i8(c,ww,M,A,Cb); double t0=now_us(); for(int r=0;r<reps;r++) ork_mm_run_i8(c,ww,M,A,Cb); ub=(now_us()-t0)/reps; hb=fnv(Cb,(size_t)M*N); }
        if(dofold){ frc=ork_npu_fold_run_i8(c,K,N,W,M,A,Cf,ncore,reps,&uf); if(!frc) hf=fnv(Cf,(size_t)M*N); }
        if(dofold&&frc){ printf("  %3d   fold rc=%d\n",M,frc); continue; }
        const char*mt = (dofold&&dobase)?(hf==hb?"ok":"MISMATCH!"):"-";
        printf("  %3d  %8.1f   %8.1f   %5.2fx   %s\n", M, ub, uf, (uf>0?ub/uf:0), mt);
    }
    ork_mm_free(c,ww); free(W);free(A);free(Cf);free(Cb); ork_npu_free(c); return 0;
}
