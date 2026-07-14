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
/* FNV-1a 64-bit. frand() is a fixed-seed LCG (sd=1) so inputs — and the NPU int32 output Ci — are
 * deterministic; the pass-path asserts an O(M*N) checksum of Ci against a static golden instead of
 * recomputing the O(M*N*K) fp32 reference matmul + RMS every run. The reference + RMS are kept and run
 * ONLY to regenerate a golden (ORK_REGEN=1 -> paste the printed values) or diagnose (ORK_FULL_REF=1). */
static uint64_t fnv64(const void*p,size_t n){ const uint8_t*b=(const uint8_t*)p; uint64_t h=1469598103934665603ULL;
    for(size_t i=0;i<n;i++){ h^=b[i]; h*=1099511628211ULL; } return h; }
static int check(ork_npu*ctx,int M,int K,int N,uint64_t gold){
    float*A=malloc((size_t)M*K*4),*B=malloc((size_t)K*N*4);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=frand();
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=frand();
    /* per-output-channel weight scales + quantize */
    float*ws=malloc((size_t)N*4); int8_t*Bq=malloc((size_t)K*N);
    for(int n=0;n<N;n++){float mx=0;for(int k=0;k<K;k++){float a=fabsf(B[(size_t)k*N+n]);if(a>mx)mx=a;}
        ws[n]=mx>0?mx/127.0f:1.0f; for(int k=0;k<K;k++)Bq[(size_t)k*N+n]=q8(B[(size_t)k*N+n]/ws[n]);}
    /* per-row activation scales + quantize */
    float*as=malloc((size_t)M*4); int8_t*Aq=malloc((size_t)M*K);
    for(int m=0;m<M;m++){float mx=0;for(int k=0;k<K;k++){float a=fabsf(A[(size_t)m*K+k]);if(a>mx)mx=a;}
        as[m]=mx>0?mx/127.0f:1.0f; for(int k=0;k<K;k++)Aq[(size_t)m*K+k]=q8(A[(size_t)m*K+k]/as[m]);}
    ork_w*w=ork_mm_pack_i8(ctx,K,N,Bq); if(!w){printf("pack failed\n");free(A);free(B);free(ws);free(Bq);free(as);free(Aq);return 1;}
    /* ORK_TEST_DMA: put the activation + output in zero-copy DMA buffers so ork_mm_run_i8 takes the
     * no-host-copy path — validates that path against the same fp32 reference. */
    int dma=getenv("ORK_TEST_DMA")!=NULL; int8_t*Aqd=Aq; int32_t*Ci;
    if(dma){ Aqd=ork_dma_alloc(ctx,(size_t)M*K); if(Aqd)memcpy(Aqd,Aq,(size_t)M*K); else Aqd=Aq;
             Ci=ork_dma_alloc(ctx,(size_t)M*N*4); if(!Ci)Ci=malloc((size_t)M*N*4); }
    else Ci=malloc((size_t)M*N*4);
    if(ork_mm_run_i8(ctx,w,M,Aqd,Ci)){printf("run failed\n");return 1;}
    uint64_t got=fnv64(Ci,(size_t)M*N*4);
    int regen=getenv("ORK_REGEN")!=NULL, ret;
    if(gold && got==gold && !regen && !getenv("ORK_FULL_REF")){
        printf("  ok   MKN=%d,%d,%d (golden 0x%016llx)\n",M,K,N,(unsigned long long)got); ret=0;  /* fast: no fp32 ref */
    } else {   /* preserved fp32 reference + RMS: regen a golden or diagnose a mismatch */
        float*Cf=malloc((size_t)M*N*4);
        for(int m=0;m<M;m++)for(int n=0;n<N;n++){float s=0;for(int k=0;k<K;k++)s+=A[(size_t)m*K+k]*B[(size_t)k*N+n];Cf[(size_t)m*N+n]=s;}
        double num=0,den=0,mx=0;
        for(int m=0;m<M;m++)for(int n=0;n<N;n++){double c=(double)Ci[(size_t)m*N+n]*as[m]*ws[n],r=Cf[(size_t)m*N+n];
            double e=c-r; num+=e*e; den+=r*r; if(fabs(e)>mx)mx=fabs(e);}
        double rms=den>0?sqrt(num/den):0; int ok=rms<0.03;   /* int8 per-channel: RMS rel err ~<1%; 3% generous */
        if(regen||!gold) printf("  REGEN quant GOLD {%d,%d,%d} = 0x%016llxULL  (RMS %.3f%% %s)\n",M,K,N,(unsigned long long)got,rms*100,ok?"ok":"BAD");
        else if(got!=gold) printf("  GOLDEN MISMATCH {%d,%d,%d}: Ci changed (RMS %.3f%%) — regen if intended\n",M,K,N,rms*100);
        printf("  %s MKN=%d,%d,%d  RMS rel err=%.3f%%  max abs err=%.4f\n",ok?"ok  ":"WRONG",M,K,N,rms*100,mx);
        ret = (ok && !(gold && got!=gold && !regen)) ? 0 : 1;
        free(Cf);
    }
    ork_w_free(w); free(A);free(B);free(ws);free(Bq);free(as);free(Aq);
    if(dma){ if(Aqd!=Aq)ork_dma_free(ctx,Aqd); ork_dma_free(ctx,Ci); } else free(Ci);
    return ret;
}
int main(void){
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    int fail=0;
    /* Static goldens (fnv64 of the NPU int32 output) for the fixed-seed inputs; regen with
     * `sudo env ORK_REGEN=1 ./quant` and paste. A 0 auto-triggers the fp32-reference regen path. */
    fail|=check(ctx,64,2048,512,  0x44e94f9c68925002ULL);   /* attention/FFN-ish shapes */
    fail|=check(ctx,16,4096,512,  0x70b268248f6790abULL);
    fail|=check(ctx,8,6144,256,   0xe0fb450b023ba66eULL);
    fail|=check(ctx,1,2048,2048,  0x7313f30cba5d9aa7ULL);   /* decode (M=1) */
    /* Tier 1c-ii: single-core full-K prefill path (set ORK_FULLK_PREFILL=1 to exercise it). K<=10752. */
    ork_npu_set_core_budget(ctx,1);
    printf("  -- single-core (budget=1; full-K prefill if ORK_FULLK_PREFILL=1) --\n");
    fail|=check(ctx,512,2048,2048, 0xc1bfa757281554ecULL);
    fail|=check(ctx,128,2048,2048, 0xe4bd381d52fa0d8cULL);
    fail|=check(ctx,64,4096,512,   0x8ee9e7f5d99078cdULL);
    ork_npu_free(ctx);
    printf("%s\n",fail?"FAIL":"ALL OK");
    return fail;
}
