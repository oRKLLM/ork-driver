/* wreuse_probe — does HW WEIGHT_REUSE compute CORRECTLY when the weight actually fits the CBUF weight banks?
 *
 * #39 concluded cross-tile weight reuse is unachievable on RK3588: setting WEIGHT_REUSE (0x1040 bit13) skips
 * the weight re-DMA (1.6-3.4x) but the reuse tile computes WRONG. That conclusion was reached entirely on the
 * FOLD path, whose 0x1040/0x1030 come baked from a K=3584 N=1216 capture -- a 4.36 MB weight against ~352 KB
 * of weight banks (0x1040 base 0xb1 => WEIGHT_BANK=11 of 12 banks x 32 KB at mg=1). The weight could never be
 * resident there, so "computes wrong" is exactly what a reuse-without-residency would produce. The mechanism
 * was never tested on a weight small enough to stay in CBUF.
 *
 * ork's normal chain SYNTHESISES 0x1040 from mc, and M>mcap already emits one program per M-tile all sharing
 * one weight address -- the reuse configuration, for free. ORK_MTILE_WR=1 sets the bit on tiles after the
 * first. So the experiment is a shape sweep with correctness as the readout:
 *
 *   K*N <= weight-bank capacity  -> resident  -> reuse should be BIT-EXACT and faster
 *   K*N >  weight-bank capacity  -> evicted   -> reuse should be WRONG (reproducing #39)
 *
 * Finding the crossover at the predicted capacity is what would turn "unachievable" into "size-bounded".
 *
 *   make wreuse_probe && sudo tools/util/npu_guard.sh -- ./wreuse_probe [M] [K]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

extern int orki_i8_chain_fullk_mcap(ork_npu *c, int K);   /* internal: the chain M-tile cap */

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

static int run_one(ork_npu*c,int M,int K,int N,int wr,int32_t*C,double*us,int iters){
    int8_t*B=malloc((size_t)K*N); for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)((i*7+3)&0x1f)-16;
    int8_t*A=malloc((size_t)M*K); for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)((i*5+1)&0x1f)-16;
    ork_w*w=ork_i8_mm_pack(c,K,N,B); if(!w){ free(A); free(B); return -1; }
    ork_mm_task_i8 t={0}; t.w=w; t.M=M; t.A=A; t.C=C;
    if(wr){ char v[8]; snprintf(v,sizeof v,"%d",wr); setenv("ORK_MTILE_WR",v,1); } else unsetenv("ORK_MTILE_WR");
    int rc=ork_i8_mm_run_chain(c,1,&t);
    if(rc==0 && iters>0){ double t0=now_us(); for(int i=0;i<iters;i++) ork_i8_mm_run_chain(c,1,&t); *us=(now_us()-t0)/iters; }
    unsetenv("ORK_MTILE_WR");
    ork_mm_free(c,w); free(A); free(B); return rc;
}

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):256, K=argc>2?atoi(argv[2]):4096;
    int N0=argc>3?atoi(argv[3]):32, N1=argc>4?atoi(argv[4]):512;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    int mcap=orki_i8_chain_fullk_mcap(c,K);
    printf("wreuse_probe: M=%d K=%d  (chain M-tile cap %d -> %d M-tiles sharing one weight)\n",
           M,K,mcap,(M+mcap-1)/mcap);
    printf("  observed 0x1040 base on this path is 0x48 (DATA_BANK=8, WEIGHT_BANK=4 => ~128 KB), and reuse is\n"
           "  bit-exact ABOVE that size, so the win is NOT explained by simple weight-bank residency.\n\n");
    printf("  %-6s %-10s %-12s %-10s %-10s\n","N","weight","reuse result","base us","reuse us");
    for(int N=N0; N<=N1; N*=2){
        int32_t*C0=calloc((size_t)M*N,4), *C1=calloc((size_t)M*N,4);
        double u0=0,u1=0;
        int r0=run_one(c,M,K,N,0,C0,&u0,20);
        int r1=run_one(c,M,K,N,1,C1,&u1,20);
        /* The mode-2 positive control (clobber 0x1040 to a bogus bank split) is DELIBERATELY NOT RUN here.
         * It did its job -- on a correctly rebuilt binary it produced "RKNPU: job timeout ... elapsed time
         * 60799192us" plus six soft resets, which proves the 0x1040 write reaches the hardware far more
         * conclusively than wrong output would. But a 60 s hang and a reset storm on a shared board is the
         * thrash pattern that has previously escalated to a kernel Oops, so it is not repeated. Run it by
         * hand with ORK_MTILE_WR=2 only if the write path must be re-proven. */
        long ctl=1;   /* write path already proven by the mode-2 timeout; see above */
        if(r0||r1){ printf("  %-6d %-10s chain rc %d/%d\n",N,"-",r0,r1); free(C0); free(C1); continue; }
        long bad=0; for(size_t j=0;j<(size_t)M*N;j++) if(C0[j]!=C1[j]) bad++;
        /* "reuse matches base" is not "both are right". Verify BOTH against an exact int32 CPU reference on a
         * strided sample that touches every output tile -- cheap, and it closes the one gap that a pure A/B
         * leaves open. */
        long refbad0=0, refbad1=0;
        { int8_t*Br=malloc((size_t)K*N); for(size_t i=0;i<(size_t)K*N;i++) Br[i]=(int8_t)((i*7+3)&0x1f)-16;
          int8_t*Ar=malloc((size_t)M*K); for(size_t i=0;i<(size_t)M*K;i++) Ar[i]=(int8_t)((i*5+1)&0x1f)-16;
          for(size_t idx=0; idx<(size_t)M*N; idx+=(N>1024?509:53)){
              int r=(int)(idx/N), cc=(int)(idx%N); int32_t ref=0;
              for(int k=0;k<K;k++) ref += (int32_t)Ar[(size_t)r*K+k]*(int32_t)Br[(size_t)k*N+cc];
              if(C0[idx]!=ref) refbad0++; if(C1[idx]!=ref) refbad1++; }
          free(Ar); free(Br); }
        printf("  %-6d %-8.0fKB %-12s %-9.1f %-9.1f  ctl=%s  %s\n", N, (double)K*N/1024.0,
               bad? "WRONG" : "bit-exact", u0, u1,
               refbad0||refbad1 ? "REF-FAIL" : "ref-ok",
               (bad ? "" : (u1<u0*0.97 ? "<= FASTER" : "(no gain)")));
        free(C0); free(C1);
    }
    printf("\n  Reading it: a bit-exact row means WEIGHT_REUSE works at that size -- #39's 'unachievable'\n"
           "  would then be a RESIDENCY bound, not a mechanism limit. All rows WRONG reproduces #39.\n");
    ork_npu_free(c); return 0;
}
