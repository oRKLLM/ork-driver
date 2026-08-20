/* tools/int4_bench.c — Tier 4b: does PER-CHANNEL int4 (one scale/output channel, full-K SINGLE submit)
 * kill the grouped int4 submit explosion and reach ~int8 speed?
 *
 * The grouped W4A4 path (ork_i4_mm_run_grouped) needs K/G submits/matmul (~16 for K=2048,G=128) — on a
 * submit-floor-bound NPU that's the ~30x prefill slowdown vs int8. The per-channel path (ork_i4_mm_run,
 * already in the driver, full-K single submit at the 10752 ceiling like int8) should run at int8 speed.
 * This times all three on the same shapes (warm), and sanity-checks per-channel int4 correctness vs the
 * fp32 reference (the MAC is exact; the only loss is the coarse per-channel int4 quant — that's the 4a
 * accuracy question, handled by Hadamard; here we care about SPEED + that the primitive is wired right).
 *
 *   make int4_bench && sudo ./int4_bench [iters]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"

static unsigned sd=1; static float frand(void){sd=sd*1103515245+12345;return ((int)((sd>>9)&0x3fff)-8192)/16384.0f;}
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; } /* ms */
static int8_t qc(float x,int lim){int q=(int)lrintf(x);return (int8_t)(q<-lim?-lim:q>lim?lim:q);}

/* per-channel int4 quant of B[K,N] (col scale) and A[M,K] (row scale); run ork_i4_mm_run; dequant;
 * return RMS rel err vs fp ref and write the warm per-call ms into *ms. */
static double pc_i4(ork_npu*ctx,int M,int K,int N,int iters,double*ms){
    float*A=malloc((size_t)M*K*4),*B=malloc((size_t)K*N*4),*Cf=malloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=frand();
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=frand();
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){float s=0;for(int k=0;k<K;k++)s+=A[(size_t)m*K+k]*B[(size_t)k*N+n];Cf[(size_t)m*N+n]=s;}
    float*ws=malloc((size_t)N*4); int8_t*Bq=malloc((size_t)K*N);
    for(int n=0;n<N;n++){float mx=1e-9f;for(int k=0;k<K;k++){float a=fabsf(B[(size_t)k*N+n]);if(a>mx)mx=a;}
        ws[n]=mx/7.0f; for(int k=0;k<K;k++)Bq[(size_t)k*N+n]=qc(B[(size_t)k*N+n]/ws[n],7);}
    float*as=malloc((size_t)M*4); int8_t*Aq=malloc((size_t)M*K);
    for(int m=0;m<M;m++){float mx=1e-9f;for(int k=0;k<K;k++){float a=fabsf(A[(size_t)m*K+k]);if(a>mx)mx=a;}
        as[m]=mx/7.0f; for(int k=0;k<K;k++)Aq[(size_t)m*K+k]=qc(A[(size_t)m*K+k]/as[m],7);}
    ork_w*w=ork_i4_mm_pack(ctx,K,N,Bq); if(!w){printf("pack_i4 failed\n");exit(1);}
    /* ORK_TEST_DMA: put A + C in ork_dma_alloc buffers to exercise the int4 zero-copy CHAINING path
     * (coherency bsync of DMA-resident A/C in run_i4_mc). Result must stay correct, not garbage. */
    int dma=getenv("ORK_TEST_DMA")!=NULL; int8_t*Aqd=Aq; int32_t*Ci;
    if(dma){ Aqd=ork_dma_alloc(ctx,(size_t)M*K); if(Aqd)memcpy(Aqd,Aq,(size_t)M*K); else Aqd=Aq;
             Ci=ork_dma_alloc(ctx,(size_t)M*N*4); if(!Ci)Ci=malloc((size_t)M*N*4); }
    else Ci=malloc((size_t)M*N*4);
    if(ork_i4_mm_run(ctx,w,M,Aqd,Ci)){printf("run_i4 failed\n");exit(1);}        /* warm */
    double t0=now(); for(int it=0;it<iters;it++) ork_i4_mm_run(ctx,w,M,Aqd,Ci); *ms=(now()-t0)/iters;
    double num=0,den=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){double c=(double)Ci[(size_t)m*N+n]*as[m]*ws[n],r=Cf[(size_t)m*N+n];num+=(c-r)*(c-r);den+=r*r;}
    ork_w_free(w); free(A);free(B);free(Cf);free(ws);free(Bq);free(as);free(Aq);
    if(dma){ if(Aqd!=Aq)ork_dma_free(ctx,Aqd); ork_dma_free(ctx,Ci); } else free(Ci);
    return den>0?sqrt(num/den):0;
}
/* grouped int4 timing (same shapes, group G) — just the warm per-call ms (correctness is in Int4-W4A4). */
static void grp_i4(ork_npu*ctx,int M,int K,int N,int G,int iters,double*ms){
    int8_t*Bq=calloc((size_t)K*N,1),*Aq=calloc((size_t)M*K,1);
    int NG=K/G; float*bS=calloc((size_t)NG*N,4),*aS=calloc((size_t)M*NG,4),*Cf=malloc((size_t)M*N*4);
    for(int i=0;i<NG*N;i++)bS[i]=1; for(int i=0;i<M*NG;i++)aS[i]=1;
    ork_w*w=ork_i4_mm_pack_grouped(ctx,K,N,Bq,G); if(!w){*ms=-1;free(Bq);free(Aq);free(bS);free(aS);free(Cf);return;}
    ork_i4_mm_run_grouped(ctx,w,M,Aq,aS,bS,Cf);
    double t0=now(); for(int it=0;it<iters;it++) ork_i4_mm_run_grouped(ctx,w,M,Aq,aS,bS,Cf); *ms=(now()-t0)/iters;
    ork_w_free(w); free(Bq);free(Aq);free(bS);free(aS);free(Cf);
}
/* int8 timing baseline (same shapes). */
static void pc_i8(ork_npu*ctx,int M,int K,int N,int iters,double*ms){
    int8_t*Bq=calloc((size_t)K*N,1),*Aq=calloc((size_t)M*K,1); int32_t*Ci=malloc((size_t)M*N*4);
    ork_w*w=ork_i8_mm_pack(ctx,K,N,Bq); if(!w){*ms=-1;free(Bq);free(Aq);free(Ci);return;}
    ork_i8_mm_run(ctx,w,M,Aq,Ci);
    double t0=now(); for(int it=0;it<iters;it++) ork_i8_mm_run(ctx,w,M,Aq,Ci); *ms=(now()-t0)/iters;
    ork_w_free(w); free(Bq);free(Aq);free(Ci);
}

/* GUARDED ISOLATION: run ONE path on ONE shape so a wedge can be attributed + recovered.
 *   ./int4_bench <iters> <mode> [M K N G]   mode: 0=int8  1=int4-perchan  2=int4-grouped */
int main(int argc,char**argv){
    int iters=argc>1?atoi(argv[1]):5;
    int mode =argc>2?atoi(argv[2]):0;
    int M=argc>3?atoi(argv[3]):1, K=argc>4?atoi(argv[4]):2048, N=argc>5?atoi(argv[5]):512, G=argc>6?atoi(argv[6]):128;
    setvbuf(stdout,0,_IONBF,0);
    printf("int4_bench mode=%d (%s)  M=%d K=%d N=%d G=%d iters=%d\n",
           mode, mode==0?"int8":mode==1?"int4-perchan":"int4-grouped", M,K,N,G,iters);
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    printf("INIT OK\n");
    double ms=-1,rms=-1;
    if(mode==0){ pc_i8(ctx,M,K,N,iters,&ms); printf("  int8: %.3f ms\n",ms); }
    else if(mode==1){ rms=pc_i4(ctx,M,K,N,iters,&ms); printf("  int4-perchan: %.3f ms  RMS=%.1f%%\n",ms,rms*100); }
    else { grp_i4(ctx,M,K,N,G,iters,&ms); printf("  int4-grouped: %.3f ms\n",ms); }
    printf("DONE mode=%d ms=%.3f\n",mode,ms);
    ork_npu_free(ctx);
    return 0;
}
