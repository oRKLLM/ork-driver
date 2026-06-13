/* examples/quant.c — validate REAL per-channel int8 quantization end-to-end.
 *
 * The int8 matmul itself is exact (int8·int8→int32 — see test_matmul). This checks the *lossy*
 * quant/dequant pattern a real engine (e.g. a llama.cpp-rockchip backend) uses around it:
 *   weights  B[K,N]:  per-OUTPUT-CHANNEL scale  ws[n] = max|B[:,n]|/127,  Bq = round(B/ws)
 *   activ.   A[M,K]:  per-ROW (per-token) scale as[m] = max|A[m,:]|/127, Aq = round(A/as)
 *   matmul   Ci = Aq · Bq      (ork_mm_run_i8, exact int32)
 *   dequant  C[m,n] = Ci[m,n] · as[m] · ws[n]
 * and compares C to the fp32 reference A·B. Per-channel int8 keeps RMS relative error ~<1%.
 *   make quant && sudo ./quant
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "ork_npu.h"
static unsigned sd=1; static float frand(void){sd=sd*1103515245+12345;return ((int)((sd>>9)&0x3fff)-8192)/16384.0f;} /* ~[-0.5,0.5] */
static int8_t q8(float x){int q=(int)lrintf(x);return (int8_t)(q<-127?-127:q>127?127:q);}
static int check(ork_npu*ctx,int M,int K,int N){
    float*A=malloc((size_t)M*K*4),*B=malloc((size_t)K*N*4),*Cf=malloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=frand();
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=frand();
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){float s=0;for(int k=0;k<K;k++)s+=A[(size_t)m*K+k]*B[(size_t)k*N+n];Cf[(size_t)m*N+n]=s;}
    /* per-output-channel weight scales + quantize */
    float*ws=malloc((size_t)N*4); int8_t*Bq=malloc((size_t)K*N);
    for(int n=0;n<N;n++){float mx=0;for(int k=0;k<K;k++){float a=fabsf(B[(size_t)k*N+n]);if(a>mx)mx=a;}
        ws[n]=mx>0?mx/127.0f:1.0f; for(int k=0;k<K;k++)Bq[(size_t)k*N+n]=q8(B[(size_t)k*N+n]/ws[n]);}
    /* per-row activation scales + quantize */
    float*as=malloc((size_t)M*4); int8_t*Aq=malloc((size_t)M*K);
    for(int m=0;m<M;m++){float mx=0;for(int k=0;k<K;k++){float a=fabsf(A[(size_t)m*K+k]);if(a>mx)mx=a;}
        as[m]=mx>0?mx/127.0f:1.0f; for(int k=0;k<K;k++)Aq[(size_t)m*K+k]=q8(A[(size_t)m*K+k]/as[m]);}
    ork_w*w=ork_mm_pack_i8(ctx,K,N,Bq); if(!w){printf("pack failed\n");return 1;}
    /* ORK_TEST_DMA: put the activation + output in zero-copy DMA buffers so ork_mm_run_i8 takes the
     * no-host-copy path — validates that path against the same fp32 reference. */
    int dma=getenv("ORK_TEST_DMA")!=NULL; int8_t*Aqd=Aq; int32_t*Ci;
    if(dma){ Aqd=ork_dma_alloc(ctx,(size_t)M*K); if(Aqd)memcpy(Aqd,Aq,(size_t)M*K); else Aqd=Aq;
             Ci=ork_dma_alloc(ctx,(size_t)M*N*4); if(!Ci)Ci=malloc((size_t)M*N*4); }
    else Ci=malloc((size_t)M*N*4);
    if(ork_mm_run_i8(ctx,w,M,Aqd,Ci)){printf("run failed\n");return 1;}
    double num=0,den=0,mx=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){double c=(double)Ci[(size_t)m*N+n]*as[m]*ws[n],r=Cf[(size_t)m*N+n];
        double e=c-r; num+=e*e; den+=r*r; if(fabs(e)>mx)mx=fabs(e);}
    double rms=den>0?sqrt(num/den):0;
    int ok=rms<0.03;   /* int8 per-channel: RMS rel err ~<1%; 3% is a generous pass bar */
    printf("  %s MKN=%d,%d,%d  RMS rel err=%.3f%%  max abs err=%.4f\n",ok?"ok  ":"WRONG",M,K,N,rms*100,mx);
    ork_w_free(w); free(A);free(B);free(Cf);free(ws);free(Bq);free(as);free(Aq);
    if(dma){ if(Aqd!=Aq)ork_dma_free(ctx,Aqd); ork_dma_free(ctx,Ci); } else free(Ci);
    return ok?0:1;
}
int main(void){
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    int fail=0;
    fail|=check(ctx,64,2048,512);   /* attention/FFN-ish shapes */
    fail|=check(ctx,16,4096,512);
    fail|=check(ctx,8,6144,256);
    fail|=check(ctx,1,2048,2048);   /* decode (M=1) */
    /* Tier 1c-ii: single-core full-K prefill path (set ORK_FULLK_PREFILL=1 to exercise it). K<=10752. */
    ork_npu_set_core_budget(ctx,1);
    printf("  -- single-core (budget=1; full-K prefill if ORK_FULLK_PREFILL=1) --\n");
    fail|=check(ctx,512,2048,2048);
    fail|=check(ctx,128,2048,2048);
    fail|=check(ctx,64,4096,512);
    ork_npu_free(ctx);
    printf("%s\n",fail?"FAIL":"ALL OK");
    return fail;
}
