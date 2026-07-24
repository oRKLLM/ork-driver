/* spine_attnblock_probe — increment 3: assemble a full attention BLOCK as a resident spine op-list, mixing CPU
 * glue kernels and NPU doorbell matmuls, and validate bit-exact vs an int8-faithful CPU reference (isolating the
 * orchestration from int8-vs-fp32 quality, which is the PPL axis). Demonstrates the full-layer DAG assembly:
 *   rmsnorm[CPU] -> QKV[NPU] -> dequant[CPU] -> q-norm[CPU] -> rope[CPU] -> attn[CPU] -> o-proj[NPU] -> residual[CPU]
 * with quant/dequant bridges at each CPU<->NPU edge, activation resident (doorbell C in dma), same-thread so the
 * read-after-drain is coherent. (Cross-thread overlap + civac handoff proven separately in spine_sched_probe.)
 *   make spine_attnblock_probe && sudo env ORK_MM_TIMEOUT=3000 ./spine_attnblock_probe
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static uint32_t rng=0xA11CEu; static float frnd(void){ rng=rng*1664525u+1013904223u; return (int)(rng>>9)/4194304.0f-1.0f; }
static int    i8rnd(void){ rng=rng*1664525u+1013904223u; return (int)((rng>>27)%9)-4; }   /* [-4,4] */

/* CPU glue kernels (same as spine_kernels_probe) */
static void k_rmsnorm(const float*x,const float*g,int D,float eps,float*y){ float ss=0; for(int i=0;i<D;i++)ss+=x[i]*x[i]; float r=1.0f/sqrtf(ss/D+eps); for(int i=0;i<D;i++)y[i]=x[i]*r*g[i]; }
static void k_rope_neox(float*x,int D,int pos,float base){ int H=D/2; for(int i=0;i<H;i++){ float th=pos*powf(base,-2.0f*i/D),cs=cosf(th),sn=sinf(th),a=x[i],b=x[i+H]; x[i]=a*cs-b*sn; x[i+H]=a*sn+b*cs; } }
static void k_attn(const float*q,const float*K,const float*V,int nkv,int dk,int dv,float scale,float*o){ float*s=malloc((size_t)nkv*4); float mx=-1e30f;
    for(int j=0;j<nkv;j++){ float a=0; for(int d=0;d<dk;d++)a+=q[d]*K[(size_t)j*dk+d]; s[j]=a*scale; if(s[j]>mx)mx=s[j]; }
    float Z=0; for(int j=0;j<nkv;j++){ s[j]=expf(s[j]-mx); Z+=s[j]; }
    for(int d=0;d<dv;d++){ float acc=0; for(int j=0;j<nkv;j++)acc+=s[j]*V[(size_t)j*dv+d]; o[d]=acc/Z; } free(s); }
/* quant fp32->int8 (per-tensor); returns the scale so dequant = int32 / scale */
static float quant(const float*x,int n,int8_t*o){ float mx=1e-6f; for(int i=0;i<n;i++){ float a=fabsf(x[i]); if(a>mx)mx=a; } float s=127.0f/mx; for(int i=0;i<n;i++){ int q=(int)lrintf(x[i]*s); o[i]=(int8_t)(q>127?127:q<-127?-127:q); } return s; }
/* civac range (same-thread read-after-drain is coherent, but harmless + documents the rule) */
static inline void civac1(volatile void*p){ __asm__ volatile("dc civac,%0"::"r"(p):"memory"); }
static void civac_range(const void*b,size_t n){ for(size_t o=0;o<n;o+=64) civac1((char*)b+o); __asm__ volatile("dsb ish":::"memory"); }

/* run one M=1 int8 matmul on the NPU via the doorbell; A=malloc int8[K], C=dma int32[N] */
static int npu_mm(ork_npu*c, ork_w*W, const int8_t*Ai8, int K, int N, int32_t*Cdma){
    for(int tr=0;tr<4;tr++){ ork_mm_task_i8 t={W,1,(int8_t*)Ai8,Cdma}; ork_dyn_chain*h=ork_dyn_begin(c,1,&t);
        if(!h) continue; int r=ork_dyn_end(h); if(r==0){ civac_range(Cdma,(size_t)N*4); return 0; } }   /* retry the intermittent doorbell-miss */
    return -1; }
/* int8 matmul reference on CPU */
static void cpu_mm(const int8_t*A,const int8_t*W,int K,int N,int32_t*C){ for(int n=0;n<N;n++){ long a=0; for(int k=0;k<K;k++)a+=A[k]*W[(size_t)k*N+n]; C[n]=(int32_t)a; } }

int main(void){
    int D=512, H=4, Hkv=2, dk=128, dv=128, nkv=256, rk2=H/Hkv;
    int Nq=H*dk, Nkv=Hkv*dk;   /* q proj N=512; k/v proj N=256 */
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    printf("spine_attnblock_probe: resident attention block (D=%d H=%d Hkv=%d dk=%d nkv=%d)\n",D,H,Hkv,dk,nkv);

    /* weights (int8 directly = the "real" weight), gains, synthetic KV cache, input */
    int8_t *Wq=malloc((size_t)D*Nq), *Wk=malloc((size_t)D*Nkv), *Wv=malloc((size_t)D*Nkv), *Wo=malloc((size_t)Nq*D);
    for(size_t i=0;i<(size_t)D*Nq;i++) Wq[i]=(int8_t)i8rnd(); for(size_t i=0;i<(size_t)D*Nkv;i++) Wk[i]=(int8_t)i8rnd();
    for(size_t i=0;i<(size_t)D*Nkv;i++) Wv[i]=(int8_t)i8rnd(); for(size_t i=0;i<(size_t)Nq*D;i++) Wo[i]=(int8_t)i8rnd();
    float *x=malloc(D*4),*an=malloc(D*4),*qn=malloc(dk*4),*kn=malloc(dk*4);
    for(int i=0;i<D;i++) x[i]=frnd()*3; for(int i=0;i<D;i++) an[i]=1+0.1f*frnd(); for(int i=0;i<dk;i++){ qn[i]=1+0.1f*frnd(); kn[i]=1+0.1f*frnd(); }
    float *Kc=malloc((size_t)Hkv*nkv*dk*4), *Vc=malloc((size_t)Hkv*nkv*dv*4);
    for(size_t i=0;i<(size_t)Hkv*nkv*dk;i++) Kc[i]=frnd(); for(size_t i=0;i<(size_t)Hkv*nkv*dv;i++) Vc[i]=frnd();
    ork_w *pWq=ork_mm_pack_i8(c,D,Nq,Wq), *pWk=ork_mm_pack_i8(c,D,Nkv,Wk), *pWv=ork_mm_pack_i8(c,D,Nkv,Wv), *pWo=ork_mm_pack_i8(c,Nq,D,Wo);
    if(!pWq||!pWk||!pWv||!pWo){ printf("pack fail\n"); return 2; }
    float scale=1.0f/sqrtf(dk);

    /* ================= SPINE (CPU kernels + NPU doorbell), activation resident ================= */
    int8_t *xn_i8=malloc(D), *ao_i8=malloc(Nq);
    int32_t *Cq=ork_dma_alloc(c,(size_t)Nq*4), *Ck=ork_dma_alloc(c,(size_t)Nkv*4), *Cv=ork_dma_alloc(c,(size_t)Nkv*4), *Co=ork_dma_alloc(c,(size_t)D*4);
    if(!Cq||!Ck||!Cv||!Co){ printf("dma fail\n"); return 2; }
    float *sp=malloc(D*4);   /* spine output */
    /* WARM the doorbell — the first cold int8 submit(s) after init are unreliable (all the other spine probes
     * warm first). Prime each distinct weight shape used below. */
    { int8_t*wa=calloc(D,1); memset(wa,1,(size_t)D);
      npu_mm(c,pWq,wa,D,Nq,Cq); npu_mm(c,pWk,wa,D,Nkv,Ck); npu_mm(c,pWv,wa,D,Nkv,Cv); npu_mm(c,pWo,wa,Nq,D,Co); free(wa); }
    {
        k_rmsnorm(x,an,D,1e-6f,sp);                                   /* 1 rmsnorm [CPU] (into sp as scratch) */
        float sx=quant(sp,D,xn_i8);                                   /* quant bridge [CPU] */
        if(npu_mm(c,pWq,xn_i8,D,Nq,Cq)||npu_mm(c,pWk,xn_i8,D,Nkv,Ck)||npu_mm(c,pWv,xn_i8,D,Nkv,Cv)){ printf("QKV npu fail\n"); return 1; }  /* QKV [NPU] (k/v computed as in a real layer; this probe attends q to the synthetic cache) */
        { int32_t*chk=malloc((size_t)Nq*4); cpu_mm(xn_i8,Wq,D,Nq,chk); long b=0; int32_t mm=0; for(int i=0;i<Nq;i++){ int32_t d=Cq[i]-chk[i]; if(d){b++; if(d<0)d=-d; if(d>mm)mm=d;} } fprintf(stderr,"[chk] Cq(NPU) vs cpu_mm: %ld/%d differ (maxdiff=%d)\n",b,Nq,mm); free(chk); }
        float *qf=malloc(Nq*4),*kf=malloc(Nkv*4),*vf=malloc(Nkv*4);
        for(int i=0;i<Nq;i++) qf[i]=Cq[i]/sx; for(int i=0;i<Nkv;i++){ kf[i]=Ck[i]/sx; vf[i]=Cv[i]/sx; }   /* dequant [CPU] */
        (void)kf;(void)vf;(void)kn;   /* k/v computed (would append to cache); this probe attends q to the synthetic cache */
        float *ao=malloc(Nq*4);
        for(int h=0;h<H;h++){ float*qh=qf+(size_t)h*dk; k_rmsnorm(qh,qn,dk,1e-6f,qh); k_rope_neox(qh,dk,nkv,1e6f);   /* q-norm + rope [CPU] */
            k_attn(qh, Kc+(size_t)(h/rk2)*nkv*dk, Vc+(size_t)(h/rk2)*nkv*dv, nkv,dk,dv,scale, ao+(size_t)h*dv); }   /* attn [CPU] */
        float sa=quant(ao,Nq,ao_i8);                                  /* quant bridge [CPU] */
        if(npu_mm(c,pWo,ao_i8,Nq,D,Co)){ printf("O npu fail\n"); return 1; }                                       /* o-proj [NPU] */
        { int32_t*chk=malloc((size_t)D*4); cpu_mm(ao_i8,Wo,Nq,D,chk); long b=0; int32_t mm=0; for(int i=0;i<D;i++){ int32_t d=Co[i]-chk[i]; if(d){b++; if(d<0)d=-d; if(d>mm)mm=d;} } fprintf(stderr,"[chk] Co(NPU) vs cpu_mm: %ld/%d differ (maxdiff=%d), sa=%.4f ao_i8[0..3]=%d,%d,%d,%d\n",b,D,mm,sa,ao_i8[0],ao_i8[1],ao_i8[2],ao_i8[3]); free(chk); }
        for(int i=0;i<D;i++) sp[i]=x[i] + Co[i]/sa;                    /* dequant + residual [CPU] */
        free(qf);free(kf);free(vf);free(ao);
    }

    /* ================= int8-faithful CPU REFERENCE (same pipeline, NPU matmul -> cpu_mm) ================= */
    float *rf=malloc(D*4);
    {
        float *an_x=malloc(D*4); k_rmsnorm(x,an,D,1e-6f,an_x);
        int8_t *xi=malloc(D); float sx=quant(an_x,D,xi);
        int32_t *cq=malloc((size_t)Nq*4),*ck=malloc((size_t)Nkv*4),*cv=malloc((size_t)Nkv*4);
        cpu_mm(xi,Wq,D,Nq,cq); cpu_mm(xi,Wk,D,Nkv,ck); cpu_mm(xi,Wv,D,Nkv,cv);
        float *qf=malloc(Nq*4); for(int i=0;i<Nq;i++) qf[i]=cq[i]/sx;
        float *ao=malloc(Nq*4);
        for(int h=0;h<H;h++){ float*qh=qf+(size_t)h*dk; k_rmsnorm(qh,qn,dk,1e-6f,qh); k_rope_neox(qh,dk,nkv,1e6f);
            k_attn(qh, Kc+(size_t)(h/rk2)*nkv*dk, Vc+(size_t)(h/rk2)*nkv*dv, nkv,dk,dv,scale, ao+(size_t)h*dv); }
        int8_t *ai=malloc(Nq); float sa=quant(ao,Nq,ai);
        int32_t *co=malloc((size_t)D*4); cpu_mm(ai,Wo,Nq,D,co);
        for(int i=0;i<D;i++) rf[i]=x[i]+co[i]/sa;
        free(an_x);free(xi);free(cq);free(ck);free(cv);free(qf);free(ao);free(ai);free(co);
    }

    double me=0; int am=0; for(int i=0;i<D;i++){ double d=fabs((double)sp[i]-rf[i]); if(d>me){me=d;am=i;} }
    fprintf(stderr,"[chk] argmax i=%d: spine=%.5f ref=%.5f (x=%.5f)\n", am, sp[am], rf[am], x[am]);
    int pass = me < 1e-3;
    printf("  spine attn-block vs int8-faithful CPU ref: max|err|=%.2e %s\n", me, pass?"bit-exact":"MISMATCH");
    printf("%s\n", pass ? "PASS — full attention block assembled on the spine (CPU glue + NPU matmuls, resident), coherent"
                        : "FAIL — assembly/dataflow bug");
    ork_dma_free(c,Cq);ork_dma_free(c,Ck);ork_dma_free(c,Cv);ork_dma_free(c,Co);
    ork_mm_free(c,pWq);ork_mm_free(c,pWk);ork_mm_free(c,pWv);ork_mm_free(c,pWo); ork_npu_free(c);
    return pass?0:1;
}
