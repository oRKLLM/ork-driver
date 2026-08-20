/* examples/i4.c — W4A4 (int4 A x int4 B -> int32 C) via the public ork_i4_mm_pack / ork_i4_mm_run.
 * Self-validates the NPU result against a CPU int4xint4 reference across shapes that exercise the
 * API's tiling: M>1 (M-tiling), N>64 (N-tiling at 64), and K>10752 (K-split with int32 accumulate).
 *   make i4 && sudo ./i4
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "ork_npu.h"
static double ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }

static int test(ork_npu*c,int M,int K,int N){
    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t*C=malloc((size_t)M*N*4);
    unsigned sd=99+M*7+K*13+N*17;
    for(size_t i=0;i<(size_t)M*K;i++){ sd=sd*1103515245+12345; A[i]=(int8_t)((int)((sd>>17)%15)-7); }
    for(size_t i=0;i<(size_t)K*N;i++){ sd=sd*1103515245+12345; B[i]=(int8_t)((int)((sd>>17)%15)-7); }
    ork_w*w=ork_i4_mm_pack(c,K,N,B); if(!w){ printf("  M=%d K=%d N=%d: pack failed\n",M,K,N); free(A);free(B);free(C); return 1; }
    int rc=ork_i4_mm_run(c,w,M,A,C); ork_w_free(w);
    if(rc){ printf("  M=%d K=%d N=%d: run rc=%d\n",M,K,N,rc); free(A);free(B);free(C); return 1; }
    long maxe=0; int bad=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[m*K+k]*B[k*N+n];
        long e=C[m*N+n]-s; if(e<0)e=-e; if(e>maxe)maxe=e; if(e)bad++; }
    printf("  M=%-3d K=%-6d N=%-5d  maxerr=%-4ld %s  (Sk=%d Sn=%d)\n",M,K,N,maxe,
           maxe==0?"OK":"FAIL",(K+10751)/10752,(N+8191)/8192);
    free(A);free(B);free(C); return maxe!=0;
}
/* per-group W4A4: fp32 -> int4 (per-group scales along K) -> NPU dequant -> RMS vs fp32 */
static int gtest(ork_npu*c,int M,int K,int N,int G){
    int Sk=K/G;
    float*Af=malloc((size_t)M*K*4),*Bf=malloc((size_t)K*N*4);
    float*aS=malloc((size_t)M*Sk*4),*bS=malloc((size_t)Sk*N*4),*C=malloc((size_t)M*N*4);
    signed char*Ai=malloc((size_t)M*K),*Bi=malloc((size_t)K*N);
    unsigned sd=7+M+K+N+G;
    for(size_t i=0;i<(size_t)M*K;i++){sd=sd*1103515245+12345;Af[i]=((int)(sd>>9)%2001-1000)/1000.0f;}
    for(size_t i=0;i<(size_t)K*N;i++){sd=sd*1103515245+12345;Bf[i]=((int)(sd>>9)%2001-1000)/1000.0f;}
    for(int m=0;m<M;m++)for(int g=0;g<Sk;g++){ float mx=1e-9f; for(int j=0;j<G;j++){float a=Af[m*K+g*G+j];if(a<0)a=-a;if(a>mx)mx=a;}
        aS[m*Sk+g]=mx/7; for(int j=0;j<G;j++){int q=(int)(Af[m*K+g*G+j]/aS[m*Sk+g]+(Af[m*K+g*G+j]>=0?.5f:-.5f));if(q>7)q=7;if(q<-8)q=-8;Ai[m*K+g*G+j]=(signed char)q;} }
    for(int g=0;g<Sk;g++)for(int n=0;n<N;n++){ float mx=1e-9f; for(int j=0;j<G;j++){float b=Bf[(g*G+j)*N+n];if(b<0)b=-b;if(b>mx)mx=b;}
        bS[g*N+n]=mx/7; for(int j=0;j<G;j++){int q=(int)(Bf[(g*G+j)*N+n]/bS[g*N+n]+(Bf[(g*G+j)*N+n]>=0?.5f:-.5f));if(q>7)q=7;if(q<-8)q=-8;Bi[(g*G+j)*N+n]=(signed char)q;} }
    ork_w*w=ork_i4_mm_pack_grouped(c,K,N,Bi,G);
    if(!w){printf("  grouped M=%d K=%d N=%d G=%d: pack failed\n",M,K,N,G);return 1;}
    int rc=ork_i4_mm_run_grouped(c,w,M,Ai,aS,bS,C); ork_w_free(w);
    if(rc){printf("  grouped M=%d K=%d N=%d G=%d: run rc=%d\n",M,K,N,G,rc);return 1;}
    double maxe=0,se=0,sr=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){
        double exact=0,f32=0;                          /* exact = same int4 vals + per-group dequant */
        for(int g=0;g<Sk;g++){ long p=0; for(int j=0;j<G;j++)p+=(long)Ai[m*K+g*G+j]*Bi[(g*G+j)*N+n];
            exact+=(double)aS[m*Sk+g]*bS[g*N+n]*p; }
        for(int k=0;k<K;k++)f32+=(double)Af[m*K+k]*Bf[k*N+n];
        double e=C[m*N+n]-exact; if(e<0)e=-e; if(e>maxe)maxe=e;   /* NPU vs exact dequant: mechanism */
        double q=C[m*N+n]-f32; se+=q*q; sr+=f32*f32;              /* vs fp32: the quant error */
    }
    int ok=maxe<0.05;                                  /* mechanism must be exact (fp32 rounding only) */
    printf("  grouped M=%-2d K=%-5d N=%-4d G=%-3d: dequant maxerr=%.4f %s | quant RMS vs fp32 %.1f%% (Sk=%d)\n",
           M,K,N,G,maxe,ok?"EXACT":"WRONG",100.0*sqrt(se/sr),Sk);
    free(Af);free(Bf);free(aS);free(bS);free(C);free(Ai);free(Bi); return !ok;
}
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed (NPU?)\n");return 1;}
    printf("W4A4 public API (ork_i4_mm_pack/ork_i4_mm_run) vs CPU int4 reference:\n");
    int fail=0;
    fail|=test(c,1,64,64);        /* baseline (decode)            */
    fail|=test(c,4,128,128);      /* M-tiling + N-tiling          */
    fail|=test(c,2,256,64);       /* K within one slice           */
    fail|=test(c,1,12288,64);     /* K-split (>10752) + accumulate */
    fail|=test(c,3,2048,256);     /* M + N tiling, mid K          */
    fail|=test(c,8,512,256);      /* regression: prefill M=8 multi-core */
    /* native multi-M (ORK_I4_MSCHED) coverage: larger-M batches exercise the 0x107c=K/16 batch scheduler.
     * These pass on the default per-row path today; `make test` also runs this example under
     * ORK_I4_MSCHED=1 (once the wide-N tile-budget law is wired into i4_mcworker) to validate the native
     * multi-M submit path bit-exact vs the CPU reference — see the wiki INT4 Multi-M RE log. */
    fail|=test(c,16,512,64);      /* multi-M batch, single 64-wide N-block (the proven sweet spot) */
    fail|=test(c,16,2048,64);     /* multi-M batch at production K */
    fail|=test(c,16,2048,256);    /* multi-M batch, wide N (multi-block tile budget) */
    /* #52: BCHAIN batch-chain on the NONBLOCK doorbell (run_i4_bchain_db) — the DEFAULT int4 M>1 prefill path.
     * Large-M (M=256) guards the shape the per-row doorbell REFUSES; the N=512 case exercises the bank-width
     * (Wb=131072/K) N-tiling + de-tile across >1 chunk. Bit-exact vs the CPU int4 reference (K/N kept small so
     * the O(M*N*K) CPU ref stays cheap). */
    fail|=test(c,128,512,256);    /* batch doorbell: large-M prefill (single bank chunk) */
    fail|=test(c,256,512,256);    /* batch doorbell: M=256 — the per-row doorbell refuses this; BCHAIN serves it */
    fail|=test(c,128,512,512);    /* batch doorbell: multi bank-chunk N-tiling (Wb=256, NC=2) */
    /* SINGLE-CORE int4 (budget=1): W4A4 is physically single-row (mc=1, captured regcmd) + PC-chained
     * for M>1 — no sched=1 M-scheduler/mg_max*64 tile (so the fp16/int8 large-tile bugs don't apply),
     * but this guards the 1-core M>1 chain path that the multi-core cases above don't exercise. */
    { int cores=ork_npu_cores(c);
      ork_npu_set_core_budget(c,1);
      fail|=test(c,8,512,256);    /* int4 1-core M=8 (chained rows) */
      fail|=test(c,4,2048,256);   /* int4 1-core M=4, K-split + accumulate */
      /* BUDGET-COUPLING probe (single-core so Ncore=N): both the activation budget (rows*K<=16384) and the
       * weight budget (Ncore*K<=131072) are AT their limits simultaneously. If these pass bit-exact, the two
       * CBUF budgets are independent (separate weight/data banks) — the N-subslice loop can set H and Nsub
       * freely. If they miscompute, the budgets are coupled and the msched weight-fit guard must be tightened. */
      fail|=test(c,32,1024,128);  /* K=1024: H=16 act=16384 (=limit) x Ncore=128 wt=131072 (=limit) */
      fail|=test(c,64,512,256);   /* K=512:  H=16 x Ncore=256 wt=131072 (=limit) */
      ork_npu_set_core_budget(c,cores); }
    printf("per-group W4A4 (fp32 -> int4 group-quant -> NPU dequant) vs fp32:\n");
    fail|=gtest(c,1,2048,256,128);    /* decode, group_size 128 (16 groups) */
    fail|=gtest(c,1,4096,512,128);    /* more groups + N-tiling                */
    fail|=gtest(c,4,2048,128,64);     /* M-tiling + finer groups                */
    fail|=gtest(c,8,1024,256,64);     /* regression: grouped prefill M=8 multi-core */
    printf("%s\n", fail?"SOME TESTS FAILED":"ALL W4A4 API TESTS PASSED");

    if(getenv("ORK_I4_BENCH")){   /* throughput: time R runs at (M,K,N). ORK_BENCH_M>1 = prefill (msched). */
        int K=2048,N=2048,R=100,M=1;
        if(getenv("ORK_BENCH_K")) K=atoi(getenv("ORK_BENCH_K"));
        if(getenv("ORK_BENCH_N")) N=atoi(getenv("ORK_BENCH_N"));
        if(getenv("ORK_BENCH_M")) M=atoi(getenv("ORK_BENCH_M"));
        signed char*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t*C=malloc((size_t)M*N*4);
        for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int)(i%15)-7;
        for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int)(i%15)-7;
        ork_w*w=ork_i4_mm_pack(c,K,N,B);
        if(w){ ork_i4_mm_run(c,w,M,A,C); ork_i4_mm_run(c,w,M,A,C);   /* warm */
            double t0=ms(); for(int r=0;r<R;r++) ork_i4_mm_run(c,w,M,A,C); double dt=(ms()-t0)/R;
            printf("bench M=%d K=%d N=%d: %.3f ms/matmul (%.1f Mrow/s, %.0f matmul/s)  msched=%s\n",
                   M,K,N,dt, (M*1e-3)/dt, 1000.0/dt, getenv("ORK_I4_MSCHED")?getenv("ORK_I4_MSCHED"):"default");
            ork_w_free(w); }
        free(A);free(B);free(C);
    }
    ork_npu_free(c); return fail;
}
