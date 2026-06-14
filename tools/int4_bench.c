/* tools/int4_bench.c — Tier 4b: does PER-CHANNEL int4 (one scale/output channel, full-K SINGLE submit)
 * kill the grouped int4 submit explosion and reach ~int8 speed?
 *
 * The grouped W4A4 path (ork_mm_run_i4_grouped) needs K/G submits/matmul (~16 for K=2048,G=128) — on a
 * submit-floor-bound NPU that's the ~30x prefill slowdown vs int8. The per-channel path (ork_mm_run_i4,
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

/* per-channel int4 quant of B[K,N] (col scale) and A[M,K] (row scale); run ork_mm_run_i4; dequant;
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
    ork_w*w=ork_mm_pack_i4(ctx,K,N,Bq); if(!w){printf("pack_i4 failed\n");exit(1);}
    /* ORK_TEST_DMA: put A + C in ork_dma_alloc buffers to exercise the int4 zero-copy CHAINING path
     * (coherency bsync of DMA-resident A/C in run_i4_mc). Result must stay correct, not garbage. */
    int dma=getenv("ORK_TEST_DMA")!=NULL; int8_t*Aqd=Aq; int32_t*Ci;
    if(dma){ Aqd=ork_dma_alloc(ctx,(size_t)M*K); if(Aqd)memcpy(Aqd,Aq,(size_t)M*K); else Aqd=Aq;
             Ci=ork_dma_alloc(ctx,(size_t)M*N*4); if(!Ci)Ci=malloc((size_t)M*N*4); }
    else Ci=malloc((size_t)M*N*4);
    if(ork_mm_run_i4(ctx,w,M,Aqd,Ci)){printf("run_i4 failed\n");exit(1);}        /* warm */
    double t0=now(); for(int it=0;it<iters;it++) ork_mm_run_i4(ctx,w,M,Aqd,Ci); *ms=(now()-t0)/iters;
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
    ork_w*w=ork_mm_pack_i4_grouped(ctx,K,N,Bq,G); if(!w){*ms=-1;free(Bq);free(Aq);free(bS);free(aS);free(Cf);return;}
    ork_mm_run_i4_grouped(ctx,w,M,Aq,aS,bS,Cf);
    double t0=now(); for(int it=0;it<iters;it++) ork_mm_run_i4_grouped(ctx,w,M,Aq,aS,bS,Cf); *ms=(now()-t0)/iters;
    ork_w_free(w); free(Bq);free(Aq);free(bS);free(aS);free(Cf);
}
/* int8 timing baseline (same shapes). */
static void pc_i8(ork_npu*ctx,int M,int K,int N,int iters,double*ms){
    int8_t*Bq=calloc((size_t)K*N,1),*Aq=calloc((size_t)M*K,1); int32_t*Ci=malloc((size_t)M*N*4);
    ork_w*w=ork_mm_pack_i8(ctx,K,N,Bq); if(!w){*ms=-1;free(Bq);free(Aq);free(Ci);return;}
    ork_mm_run_i8(ctx,w,M,Aq,Ci);
    double t0=now(); for(int it=0;it<iters;it++) ork_mm_run_i8(ctx,w,M,Aq,Ci); *ms=(now()-t0)/iters;
    ork_w_free(w); free(Bq);free(Aq);free(Ci);
}

int main(int argc,char**argv){
    int iters=argc>1?atoi(argv[1]):30;
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    int G=128;
    struct{int M,K,N;}shp[]={ {512,2048,512}, {128,2048,512}, {16,2048,512}, {1,2048,512} };
    printf("int4 per-channel (1 submit) vs grouped G=%d (K/G submits) vs int8 — warm, %d iters\n",G,iters);
    printf("  %-14s %10s %10s %10s   %8s %8s\n","MKN","pc-i4 ms","grp-i4 ms","i8 ms","i4/i8","grp/i4");
    for(unsigned s=0;s<sizeof shp/sizeof*shp;s++){
        int M=shp[s].M,K=shp[s].K,N=shp[s].N; double m_pc,m_g,m_i8,rms;
        rms=pc_i4(ctx,M,K,N,iters,&m_pc);
        grp_i4(ctx,M,K,N,G,iters,&m_g);
        pc_i8(ctx,M,K,N,iters,&m_i8);
        char mkn[32]; snprintf(mkn,sizeof mkn,"%d,%d,%d",M,K,N);
        printf("  %-14s %10.3f %10.3f %10.3f   %8.2f %8.2f   pc-i4 RMS=%.1f%%\n",
               mkn,m_pc,m_g,m_i8, m_i8>0?m_pc/m_i8:0, m_pc>0?m_g/m_pc:0, rms*100);
    }
    ork_npu_free(ctx);
    return 0;
}
