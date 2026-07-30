/* ab_fold — #39 A/B: rkllm token-fold vs ork-normal int8 matmul, K=3584 N=1216, single-core.
 * Run in TWO SEPARATE PROCESSES (separate NPU contexts) so the fold's multi-task submit never shares a context
 * with ork_mm_run_i8 (which currently EINVALs it — a run-path integration bug to fix; fold_tiler proves the fold
 * is correct standalone). Each mode prints, per M, the avg time and an FNV checksum of C; compare checksums across
 * the two runs for correctness and the times for the ratio.
 *   sudo env ORK_MM_TIMEOUT=800 ./ab_fold fold [reps]      # token-fold (ork_npu_fold_run_i8), submit-avg us
 *   sudo env ORK_MM_TIMEOUT=800 ./ab_fold base [reps]      # ork-normal (ork_mm_run_i8), full-call wall us
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "ork_npu.h"
extern int ork_npu_fold_run_i8(ork_npu*,int,int,const int8_t*,int,const int8_t*,int32_t*,int,double*);
static uint32_t rng=0x1234u; static int r7(void){ rng=rng*1664525u+1013904223u; return (int)((rng>>25)%7)-3; }
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static uint64_t fnv(const int32_t*p,size_t n){ uint64_t h=1469598103934665603ull; const uint8_t*b=(const uint8_t*)p;
    for(size_t i=0;i<n*4;i++){ h^=b[i]; h*=1099511628211ull; } return h; }

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    const char*mode=argc>1?argv[1]:"fold";
    int reps=argc>2?atoi(argv[2]):8;
    const int K=3584,N=1216, Mmax=128;
    int Ms[]={36,72,100,128}; int nM=(int)(sizeof Ms/sizeof*Ms);
    int8_t *W=malloc((size_t)K*N),*A=malloc((size_t)Mmax*K);
    for(size_t i=0;i<(size_t)K*N;i++) W[i]=(int8_t)r7();
    for(size_t i=0;i<(size_t)Mmax*K;i++) A[i]=(int8_t)r7();
    int32_t *C=malloc((size_t)Mmax*N*4);

    ork_npu*c=ork_npu_init(); if(!c){ printf("init (board only)\n"); return 77; }
    ork_npu_set_core_budget(c,1);
    /* prewarm: activate IOMMU domain + warm (fold_batch needs an active domain), then FREE — no resident weight */
    { ork_w*t=ork_mm_pack_i8(c,K,N,W); if(t){int32_t*z=calloc((size_t)8*N,4);int8_t*a8=calloc((size_t)8*K,1);ork_mm_run_i8(c,t,8,a8,z);free(z);free(a8);ork_mm_free(c,t);} }

    int isfold = !strcmp(mode,"fold");
    printf("A/B mode=%s single-core K=%d N=%d reps=%d\n", mode, K, N, reps);
    printf("   M     us       C-checksum\n");
    ork_w*ww=NULL;
    if(!isfold){ ww=ork_mm_pack_i8(c,K,N,W); if(!ww){ printf("pack fail\n"); return 1; } }
    for(int i=0;i<nM;i++){ int M=Ms[i]; double us=0;
        if(isfold){ int rc=ork_npu_fold_run_i8(c,K,N,W,M,A,C,reps,&us); if(rc){ printf("  %3d  fold rc=%d\n",M,rc); continue; } }
        else { ork_mm_run_i8(c,ww,M,A,C); double t0=now_us(); for(int r=0;r<reps;r++) ork_mm_run_i8(c,ww,M,A,C); us=(now_us()-t0)/reps; }
        printf("  %3d  %8.1f   0x%016llx\n", M, us, (unsigned long long)fnv(C,(size_t)M*N));
    }
    if(ww) ork_mm_free(c,ww);
    free(W);free(A);free(C); ork_npu_free(c); return 0;
}
