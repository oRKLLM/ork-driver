/* examples/test_moe_smoke.c — <1 min MoE int4 perf+correctness smoke validator.
 *
 * The 35B llama-bench is ~15 min (dominated by the 22GB load+pack). For iterating on the int4 MoE matmul
 * path (weight-reuse, tiling, submit mode, ...) this skips the model entirely: it packs the REAL qwen35moe
 * expert shapes (gate/up K=2048 N=512, down K=512 N=2048), (1) VERIFIES bit-exact vs a CPU int4 reference —
 * catches miscompute regressions (blocking, bad reuse) — and (2) times the gate+up+down "expert triple" at
 * MoE-realistic M, printing ONE headline number to A/B a change. A/B: ./test_moe_smoke  vs  ENV=1 ./test_moe_smoke.
 *   make test_moe_smoke && sudo ./test_moe_smoke [M]      (default M=32, the padded per-expert prefill M)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static void fill_i4(int8_t*p,size_t n,unsigned s){ for(size_t i=0;i<n;i++){ s=s*1103515245u+12345u; p[i]=(int8_t)((int)((s>>17)%15)-7); } }
static long verify(const int8_t*A,const int8_t*B,const int32_t*C,int M,int K,int N){ long e=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[(size_t)m*K+k]*B[(size_t)k*N+n];
        long d=C[(size_t)m*N+n]-s; if(d<0)d=-d; if(d>e)e=d; } return e; }

/* pack+run one expert matmul; verify bit-exact (once) + return best us/matmul. */
static double one(ork_npu*c,const char*tag,int K,int N,int M,int*bad){
    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t*C=malloc((size_t)M*N*4);
    fill_i4(A,(size_t)M*K,7); fill_i4(B,(size_t)K*N,23);
    ork_w*w=ork_mm_pack_i4(c,K,N,B); if(!w){ printf("  %-8s PACK FAIL\n",tag); *bad=1; free(A);free(B);free(C); return 0; }
    if(ork_mm_run_i4(c,w,M,A,C)){ printf("  %-8s RUN FAIL\n",tag); *bad=1; ork_mm_free(c,w); free(A);free(B);free(C); return 0; }
    long e=verify(A,B,C,M,K,N);
    double best=1e18; for(int r=0;r<5;r++){ double t0=now_us(); for(int i=0;i<20;i++) ork_mm_run_i4(c,w,M,A,C); double u=(now_us()-t0)/20; if(u<best)best=u; }
    printf("  %-8s K=%-4d N=%-4d M=%d: %7.1f us/matmul  %s\n", tag,K,N,M,best, e==0?"bit-exact":"*** MISCOMPUTE ***");
    if(e) *bad=1;
    ork_mm_free(c,w); free(A);free(B);free(C); return best;
}

int main(int argc,char**argv){
    int M = (argc>1)?atoi(argv[1]):32;
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    printf("MoE int4 smoke (qwen35moe expert shapes, M=%d):\n", M);
    double t0=now_us(); int bad=0;
    double g=one(c,"gate",2048,512,M,&bad);
    double u=one(c,"up",  2048,512,M,&bad);
    double d=one(c,"down",512,2048,M,&bad);
    double triple=g+u+d;
    printf("  ---- EXPERT-TRIPLE (gate+up+down) = %.1f us  [wall %.1fs] ----  %s\n",
           triple,(now_us()-t0)/1e6, bad?"FAIL (miscompute/regression)":"OK");
    ork_npu_free(c);
    return bad?1:0;
}
