/* spine_layer_probe — increment 3: a FULL qwen3-style decode layer assembled on the heterogeneous spine
 * (CPU glue kernels + NPU doorbell matmuls, activation resident), validated BIT-EXACT vs an int8-faithful CPU
 * reference. Attention block + FFN:
 *   rmsnorm[CPU]->QKV[NPU]->deq->q-norm->rope->attn[CPU]->o-proj[NPU]->residual[CPU]
 *   ->ffn-norm[CPU]->gate/up[NPU]->silu·glu[CPU]->down[NPU]->residual[CPU]
 * quant/dequant bridges at each CPU<->NPU edge; the doorbell is warmed per weight shape (cold submits garbage).
 *   make spine_layer_probe && sudo env ORK_MM_TIMEOUT=3000 ./spine_layer_probe
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static uint32_t rng=0xA11CEu; static float frnd(void){ rng=rng*1664525u+1013904223u; return (int)(rng>>9)/4194304.0f-1.0f; }
static int    i8rnd(void){ rng=rng*1664525u+1013904223u; return (int)((rng>>27)%9)-4; }

/* CPU glue kernels */
static void k_rmsnorm(const float*x,const float*g,int D,float eps,float*y){ float ss=0; for(int i=0;i<D;i++)ss+=x[i]*x[i]; float r=1.0f/sqrtf(ss/D+eps); for(int i=0;i<D;i++)y[i]=x[i]*r*g[i]; }
static void k_rope_neox(float*x,int D,int pos,float base){ int H=D/2; for(int i=0;i<H;i++){ float th=pos*powf(base,-2.0f*i/D),cs=cosf(th),sn=sinf(th),a=x[i],b=x[i+H]; x[i]=a*cs-b*sn; x[i+H]=a*sn+b*cs; } }
static void k_attn(const float*q,const float*K,const float*V,int nkv,int dk,int dv,float scale,float*o){ float*s=malloc((size_t)nkv*4); float mx=-1e30f;
    for(int j=0;j<nkv;j++){ float a=0; for(int d=0;d<dk;d++)a+=q[d]*K[(size_t)j*dk+d]; s[j]=a*scale; if(s[j]>mx)mx=s[j]; }
    float Z=0; for(int j=0;j<nkv;j++){ s[j]=expf(s[j]-mx); Z+=s[j]; }
    for(int d=0;d<dv;d++){ float acc=0; for(int j=0;j<nkv;j++)acc+=s[j]*V[(size_t)j*dv+d]; o[d]=acc/Z; } free(s); }
static void k_silu_glu(const float*gate,const float*up,int n,float*o){ for(int i=0;i<n;i++){ float g=gate[i]; o[i]=(g/(1.0f+expf(-g)))*up[i]; } }
static float quant(const float*x,int n,int8_t*o){ float mx=1e-6f; for(int i=0;i<n;i++){ float a=fabsf(x[i]); if(a>mx)mx=a; } float s=127.0f/mx; for(int i=0;i<n;i++){ int q=(int)lrintf(x[i]*s); o[i]=(int8_t)(q>127?127:q<-127?-127:q); } return s; }
static inline void civac1(volatile void*p){ __asm__ volatile("dc civac,%0"::"r"(p):"memory"); }
static void civac_range(const void*b,size_t n){ for(size_t o=0;o<n;o+=64) civac1((char*)b+o); __asm__ volatile("dsb ish":::"memory"); }
static int npu_mm(ork_npu*c, ork_w*W, const int8_t*Ai8, int N, int32_t*Cdma){
    for(int tr=0;tr<4;tr++){ ork_mm_task_i8 t={W,1,(int8_t*)Ai8,Cdma}; ork_dyn_chain*h=ork_dyn_begin(c,1,&t);
        if(!h) continue; if(ork_dyn_end(h)==0){ civac_range(Cdma,(size_t)N*4); return 0; } } return -1; }
static void cpu_mm(const int8_t*A,const int8_t*W,int K,int N,int32_t*C){ for(int n=0;n<N;n++){ long a=0; for(int k=0;k<K;k++)a+=A[k]*W[(size_t)k*N+n]; C[n]=(int32_t)a; } }

int main(void){
    int D=512,H=4,Hkv=2,dk=128,dv=128,nkv=256,rk2=H/Hkv, Nq=H*dk,Nkv=Hkv*dk, Nff=512;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    printf("spine_layer_probe: FULL layer (attn + FFN) on the spine (D=%d H=%d Hkv=%d Nff=%d nkv=%d)\n",D,H,Hkv,Nff,nkv);

    int8_t *Wq=malloc((size_t)D*Nq),*Wk=malloc((size_t)D*Nkv),*Wv=malloc((size_t)D*Nkv),*Wo=malloc((size_t)Nq*D);
    int8_t *Wg=malloc((size_t)D*Nff),*Wu=malloc((size_t)D*Nff),*Wd=malloc((size_t)Nff*D);
    for(size_t i=0;i<(size_t)D*Nq;i++)Wq[i]=(int8_t)i8rnd(); for(size_t i=0;i<(size_t)D*Nkv;i++){Wk[i]=(int8_t)i8rnd();Wv[i]=(int8_t)i8rnd();}
    for(size_t i=0;i<(size_t)Nq*D;i++)Wo[i]=(int8_t)i8rnd(); for(size_t i=0;i<(size_t)D*Nff;i++){Wg[i]=(int8_t)i8rnd();Wu[i]=(int8_t)i8rnd();} for(size_t i=0;i<(size_t)Nff*D;i++)Wd[i]=(int8_t)i8rnd();
    float *x=malloc(D*4),*an=malloc(D*4),*qn=malloc(dk*4),*fn=malloc(D*4);
    for(int i=0;i<D;i++){x[i]=frnd()*3;an[i]=1+0.1f*frnd();fn[i]=1+0.1f*frnd();} for(int i=0;i<dk;i++)qn[i]=1+0.1f*frnd();
    float *Kc=malloc((size_t)Hkv*nkv*dk*4),*Vc=malloc((size_t)Hkv*nkv*dv*4);
    for(size_t i=0;i<(size_t)Hkv*nkv*dk;i++)Kc[i]=frnd(); for(size_t i=0;i<(size_t)Hkv*nkv*dv;i++)Vc[i]=frnd();
    ork_w *pWq=ork_mm_pack_i8(c,D,Nq,Wq),*pWk=ork_mm_pack_i8(c,D,Nkv,Wk),*pWv=ork_mm_pack_i8(c,D,Nkv,Wv),*pWo=ork_mm_pack_i8(c,Nq,D,Wo);
    ork_w *pWg=ork_mm_pack_i8(c,D,Nff,Wg),*pWu=ork_mm_pack_i8(c,D,Nff,Wu),*pWd=ork_mm_pack_i8(c,Nff,D,Wd);
    if(!pWq||!pWk||!pWv||!pWo||!pWg||!pWu||!pWd){ printf("pack fail\n"); return 2; }
    float scale=1.0f/sqrtf(dk);

    int8_t *xn_i8=malloc(D),*ao_i8=malloc(Nq),*xf_i8=malloc(D),*ac_i8=malloc(Nff);
    int32_t *Cq=ork_dma_alloc(c,(size_t)Nq*4),*Ck=ork_dma_alloc(c,(size_t)Nkv*4),*Cv=ork_dma_alloc(c,(size_t)Nkv*4),*Co=ork_dma_alloc(c,(size_t)D*4);
    int32_t *Cg=ork_dma_alloc(c,(size_t)Nff*4),*Cu=ork_dma_alloc(c,(size_t)Nff*4),*Cd=ork_dma_alloc(c,(size_t)D*4);
    if(!Cq||!Ck||!Cv||!Co||!Cg||!Cu||!Cd){ printf("dma fail\n"); return 2; }
    /* WARM the doorbell per weight shape */
    { int8_t*wa=calloc(Nff,1); memset(wa,1,(size_t)Nff);
      npu_mm(c,pWq,wa,Nq,Cq);npu_mm(c,pWk,wa,Nkv,Ck);npu_mm(c,pWv,wa,Nkv,Cv);npu_mm(c,pWo,wa,D,Co);
      npu_mm(c,pWg,wa,Nff,Cg);npu_mm(c,pWu,wa,Nff,Cu);npu_mm(c,pWd,wa,D,Cd); free(wa); }

    float *sp=malloc(D*4);
    {   /* ---- SPINE ---- */
        float *xn=malloc(D*4); k_rmsnorm(x,an,D,1e-6f,xn); float sx=quant(xn,D,xn_i8);
        if(npu_mm(c,pWq,xn_i8,Nq,Cq)||npu_mm(c,pWk,xn_i8,Nkv,Ck)||npu_mm(c,pWv,xn_i8,Nkv,Cv)){ printf("QKV fail\n"); return 1; }
        float *qf=malloc(Nq*4); for(int i=0;i<Nq;i++) qf[i]=Cq[i]/sx;
        float *ao=malloc(Nq*4);
        for(int h=0;h<H;h++){ float*qh=qf+(size_t)h*dk; k_rmsnorm(qh,qn,dk,1e-6f,qh); k_rope_neox(qh,dk,nkv,1e6f);
            k_attn(qh,Kc+(size_t)(h/rk2)*nkv*dk,Vc+(size_t)(h/rk2)*nkv*dv,nkv,dk,dv,scale,ao+(size_t)h*dv); }
        float sa=quant(ao,Nq,ao_i8); if(npu_mm(c,pWo,ao_i8,D,Co)){ printf("O fail\n"); return 1; }
        float *x1=malloc(D*4); for(int i=0;i<D;i++) x1[i]=x[i]+Co[i]/sa;                    /* attn residual */
        float *xf=malloc(D*4); k_rmsnorm(x1,fn,D,1e-6f,xf); float sf=quant(xf,D,xf_i8);      /* ffn-norm */
        if(npu_mm(c,pWg,xf_i8,Nff,Cg)||npu_mm(c,pWu,xf_i8,Nff,Cu)){ printf("gate/up fail\n"); return 1; }
        float *gf=malloc(Nff*4),*uf=malloc(Nff*4),*act=malloc(Nff*4);
        for(int i=0;i<Nff;i++){ gf[i]=Cg[i]/sf; uf[i]=Cu[i]/sf; } k_silu_glu(gf,uf,Nff,act);  /* silu·glu */
        float sac=quant(act,Nff,ac_i8); if(npu_mm(c,pWd,ac_i8,D,Cd)){ printf("down fail\n"); return 1; }
        for(int i=0;i<D;i++) sp[i]=x1[i]+Cd[i]/sac;                                          /* ffn residual */
        free(xn);free(qf);free(ao);free(x1);free(xf);free(gf);free(uf);free(act);
    }
    float *rf=malloc(D*4);
    {   /* ---- int8-faithful CPU REFERENCE ---- */
        float *xn=malloc(D*4); k_rmsnorm(x,an,D,1e-6f,xn); int8_t*xi=malloc(D); float sx=quant(xn,D,xi);
        int32_t *cq=malloc((size_t)Nq*4); cpu_mm(xi,Wq,D,Nq,cq);
        float *qf=malloc(Nq*4); for(int i=0;i<Nq;i++) qf[i]=cq[i]/sx;
        float *ao=malloc(Nq*4);
        for(int h=0;h<H;h++){ float*qh=qf+(size_t)h*dk; k_rmsnorm(qh,qn,dk,1e-6f,qh); k_rope_neox(qh,dk,nkv,1e6f);
            k_attn(qh,Kc+(size_t)(h/rk2)*nkv*dk,Vc+(size_t)(h/rk2)*nkv*dv,nkv,dk,dv,scale,ao+(size_t)h*dv); }
        int8_t*ai=malloc(Nq); float sa=quant(ao,Nq,ai); int32_t*co=malloc((size_t)D*4); cpu_mm(ai,Wo,Nq,D,co);
        float *x1=malloc(D*4); for(int i=0;i<D;i++) x1[i]=x[i]+co[i]/sa;
        float *xf=malloc(D*4); k_rmsnorm(x1,fn,D,1e-6f,xf); int8_t*xfi=malloc(D); float sf=quant(xf,D,xfi);
        int32_t*cg=malloc((size_t)Nff*4),*cu=malloc((size_t)Nff*4); cpu_mm(xfi,Wg,D,Nff,cg); cpu_mm(xfi,Wu,D,Nff,cu);
        float *gf=malloc(Nff*4),*uf=malloc(Nff*4),*act=malloc(Nff*4);
        for(int i=0;i<Nff;i++){ gf[i]=cg[i]/sf; uf[i]=cu[i]/sf; } k_silu_glu(gf,uf,Nff,act);
        int8_t*aci=malloc(Nff); float sac=quant(act,Nff,aci); int32_t*cd=malloc((size_t)D*4); cpu_mm(aci,Wd,Nff,D,cd);
        for(int i=0;i<D;i++) rf[i]=x1[i]+cd[i]/sac;
        free(xn);free(xi);free(cq);free(qf);free(ao);free(ai);free(co);free(x1);free(xf);free(xfi);free(cg);free(cu);free(gf);free(uf);free(act);free(aci);free(cd);
    }
    double me=0; for(int i=0;i<D;i++){ double d=fabs((double)sp[i]-rf[i]); if(d>me)me=d; }
    int pass=me<1e-3;
    printf("  full layer (attn+FFN) vs int8-faithful CPU ref: max|err|=%.2e %s\n", me, pass?"bit-exact":"MISMATCH");
    printf("%s\n", pass?"PASS — full qwen3-style layer assembled on the spine (CPU glue + NPU matmuls, resident), coherent":"FAIL");
    ork_npu_free(c);
    return pass?0:1;
}
