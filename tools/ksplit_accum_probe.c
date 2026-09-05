/* ksplit_accum_probe — how much is on-chip accumulator retention actually worth?
 *
 * THE CHAIN OF REASONING. The NPU's matmul is weight-DMA-bound: each M-tile submit re-streams the whole
 * K*N weight, and the M-tile cap is CBUF-capacity-driven, so cap*K is roughly constant (measured
 * 704@K512, 320@K1024, 128@K2048, 64@K4096 -- all ~260-360k elements). Weight traffic for M rows is
 * therefore (M/cap)*K*N = M*K^2*N/const: it scales with K SQUARED. Splitting K into S slices should cut
 * total weight traffic by S.
 *
 * The catch is the accumulator. ork today writes S full M*N int32 partials to DRAM and HOST-SUMS them
 * (npu.c:123, chain.c:400/614), which costs S*M*N*4 bytes of write plus the read-back -- and that is what
 * made the weight-stationary model come out at break-even (Optimization-Roadmap Tier 13, x0.98-1.03).
 * If the NPU could retain the accumulator across K-slices, the partials would vanish and the full 1/S
 * weight saving would land.
 *
 * Rather than reverse-engineer the output stage first, MEASURE THE CEILING: run the same matmul as one
 * K=4096 task and as S K-slices with host summing, and time the pieces separately. That gives
 *   - the real weight-traffic saving (NPU time, sliced vs single), and
 *   - exactly what accumulator retention would have to buy back (the partial-write + host-sum time).
 * If the NPU-only sliced time is not much better than single, no accumulator mechanism can rescue it and
 * the RE is not worth doing. Correctness is checked first: the summed slices must equal the single-task
 * result bit-exactly, or the comparison is meaningless.
 *
 *   make ksplit_accum_probe && sudo tools/util/npu_guard.sh -- ./ksplit_accum_probe [M] [K] [N] [S]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):704, K=argc>2?atoi(argv[2]):4096, N=argc>3?atoi(argv[3]):1024, S=argc>4?atoi(argv[4]):8;
    int ITER=8;
    setvbuf(stdout,0,_IONBF,0);
    if(K%S){ printf("K must divide by S\n"); return 2; }
    int Ks=K/S;
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    printf("ksplit_accum_probe: M=%d K=%d N=%d  S=%d (Ks=%d)\n",M,K,N,S,Ks);
    printf("  M-tile cap is ~const/K, so the single-task path re-streams the %.1f MB weight ~%d times;\n",
           (double)K*N/1e6, (M + 63)/64);
    printf("  each slice's weight is %.2f MB and its cap is ~%dx higher.\n\n",(double)Ks*N/1e6,S);

    int8_t *B=malloc((size_t)K*N); for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)((i*7+3)&0x3f)-32;
    int8_t *A=malloc((size_t)M*K); for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)((i*5+1)&0x3f)-32;

    /* --- single task, K wide --- */
    ork_w *w1=ork_i8_mm_pack(c,K,N,B); if(!w1){ printf("pack fail\n"); return 2; }
    int32_t *C1=malloc((size_t)M*N*4);
    ork_i8_mm_run(c,w1,M,A,C1);
    double t0=now_us(); for(int i=0;i<ITER;i++) ork_i8_mm_run(c,w1,M,A,C1); double t_single=(now_us()-t0)/ITER;

    /* --- S slices at Ks, host-summed --- */
    ork_w **ws=calloc(S,sizeof *ws); int8_t **As=calloc(S,sizeof *As); int32_t **Cs=calloc(S,sizeof *Cs);
    int8_t *Bs=malloc((size_t)Ks*N);
    for(int s=0;s<S;s++){
        for(int k=0;k<Ks;k++) memcpy(Bs+(size_t)k*N, B+(size_t)(s*Ks+k)*N, N);
        ws[s]=ork_i8_mm_pack(c,Ks,N,Bs); if(!ws[s]){ printf("slice pack fail\n"); return 2; }
        As[s]=malloc((size_t)M*Ks);
        for(int m=0;m<M;m++) memcpy(As[s]+(size_t)m*Ks, A+(size_t)m*K+s*Ks, Ks);
        Cs[s]=malloc((size_t)M*N*4);
    }
    for(int s=0;s<S;s++) ork_i8_mm_run(c,ws[s],M,As[s],Cs[s]);      /* warm */
    t0=now_us();
    for(int i=0;i<ITER;i++) for(int s=0;s<S;s++) ork_i8_mm_run(c,ws[s],M,As[s],Cs[s]);
    double t_npu_sliced=(now_us()-t0)/ITER;

    int32_t *Csum=malloc((size_t)M*N*4);
    t0=now_us();
    for(int i=0;i<ITER;i++){
        memcpy(Csum,Cs[0],(size_t)M*N*4);
        for(int s=1;s<S;s++){ const int32_t*p=Cs[s]; for(size_t j=0;j<(size_t)M*N;j++) Csum[j]+=p[j]; }
    }
    double t_hostsum=(now_us()-t0)/ITER;

    long bad=0; for(size_t j=0;j<(size_t)M*N;j++) if(Csum[j]!=C1[j]) bad++;
    printf("  correctness: summed slices vs single task -> %s (%ld/%zu differ)\n",
           bad?"*** MISMATCH ***":"bit-exact", bad, (size_t)M*N);
    if(bad){ printf("  refusing to compare timings for a wrong result.\n"); return 3; }

    double gmac=(double)M*(double)K*N/1e9;
    printf("\n  single task K=%d      %8.2f ms   %6.1f GMAC/s\n", K, t_single/1e3, gmac/(t_single/1e6));
    printf("  %d slices, NPU only  %8.2f ms   %6.1f GMAC/s   -> weight-traffic saving %.2fx\n",
           S, t_npu_sliced/1e3, gmac/(t_npu_sliced/1e6), t_single/t_npu_sliced);
    printf("  host sum of partials %8.2f ms\n", t_hostsum/1e3);
    printf("  sliced TOTAL         %8.2f ms   %6.1f GMAC/s   -> net %.2fx vs single\n",
           (t_npu_sliced+t_hostsum)/1e3, gmac/((t_npu_sliced+t_hostsum)/1e6), t_single/(t_npu_sliced+t_hostsum));
    printf("\n  What accumulator retention would buy: it removes the host sum (%.2f ms) and the S-1 extra\n"
           "  partial writes. Ceiling = the NPU-only number above, i.e. %.2fx.\n", t_hostsum/1e3, t_single/t_npu_sliced);
    printf("  VERDICT: %s\n", (t_single/t_npu_sliced > 1.3)
        ? "the weight-traffic saving is REAL and large -- accumulator retention is worth reverse-engineering"
        : "the weight-traffic saving is small; no accumulator mechanism could rescue K-slicing here");
    return 0;
}
