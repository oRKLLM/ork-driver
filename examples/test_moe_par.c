/* examples/test_moe_par.c — does the int4 MoE dispatch floor PARALLELIZE across the 3 NPU cores?
 *
 * The wall is ~18us/program fixed dispatch (reconfigure). If that floor is PER-CORE, running whole experts
 * one-per-core concurrently (expert-parallel) earns ~3x; if it's a SHARED front-end, 3 cores buy nothing and
 * NPU-experts are capped (=> tilt the hybrid to NEON). Prior tests only showed COLSPLIT of one tiny matmul
 * (3-core==1-core); this is the real expert-parallel test: N whole experts via ork_mm_run_i4_experts, nc=1 vs
 * nc=3. Verdict from the ratio. Also verifies bit-exact (coalesce path + default weight-reuse).
 *   make test_moe_par && sudo ./test_moe_par
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

int main(void){
    int NE=24, K=2048, N=512, M=32;    /* NE gate-shape experts, MoE-realistic M */
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    ork_mm_task_i4 *ex=calloc(NE,sizeof*ex);
    int8_t **Aa=calloc(NE,sizeof*Aa),**Bb=calloc(NE,sizeof*Bb);
    for(int e=0;e<NE;e++){
        Aa[e]=malloc((size_t)M*K); Bb[e]=malloc((size_t)K*N); int32_t*C=malloc((size_t)M*N*4);
        fill_i4(Aa[e],(size_t)M*K,7+e); fill_i4(Bb[e],(size_t)K*N,23+e);
        ork_w*w=ork_mm_pack_i4(c,K,N,Bb[e]); if(!w){ printf("pack %d fail\n",e); return 1; }
        ex[e]=(ork_mm_task_i4){w,M,Aa[e],C};
    }
    printf("expert-parallel: %d experts K=%d N=%d M=%d\n",NE,K,N,M);
    double best[4]={1e18,0,0,1e18};
    for(int pass=0;pass<2;pass++){ int nc=pass?3:1;
        if(ork_mm_run_i4_experts(c,ex,NE,nc)){ printf("run nc=%d FAIL\n",nc); return 1; }   /* warm */
        double b=1e18; for(int r=0;r<4;r++){ double t0=now_us(); for(int it=0;it<8;it++) ork_mm_run_i4_experts(c,ex,NE,nc);
            double u=(now_us()-t0)/8; if(u<b)b=u; }
        best[nc]=b;
        printf("  nc=%d: %8.1f us for %d experts  (%.1f us/expert)\n", nc, b, NE, b/NE);
    }
    long e0=verify(Aa[0],Bb[0],ex[0].C,M,K,N);
    double ratio=best[1]/best[3];
    printf("  ---- nc=1/nc=3 speedup = %.2fx  ----  %s | expert0 %s\n", ratio,
           ratio>=2.2?"PER-CORE front-end: expert-parallel EARNS ~3x":
           (ratio>=1.3?"PARTIAL parallel":"SHARED front-end: 3 cores buy little => tilt hybrid to NEON"),
           e0==0?"bit-exact":"MISCOMPUTE");
    ork_npu_free(c);
    return 0;
}
