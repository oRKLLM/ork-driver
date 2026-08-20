/* examples/model.c — multi-layer transformer body (N stacked decoder layers) on the NPU stack,
 * with grouped-query attention + arbitrary head_dim. Each layer has its own projection + norm
 * weights; prefills a sequence through all layers. Validates the NPU-hybrid forward against a
 * pure-CPU reference (identical ops, CPU matmul) — confirms layer stacking under GQA.
 *   make model && sudo ./model [nlayers]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ork_npu.h"
#include "neon_activations.h"
typedef ork_f16 f16;
#define SEQ 16
#define MAXL 32
#define EPS 1e-5f
/* GQA config: hidden H, q heads NH, kv heads NKV, head dim HD (q-dim=NH*HD), mlp FFN */
#define H 512
#define NH 8
#define NKV 2
#define HD 64
#define FFN 2048
#define QD (NH*HD)
#define KVD (NKV*HD)

static void rmsnorm(float*o,const float*x,const float*w,int n){ ork_rmsnorm_f32(o,x,w,n,EPS); }
static void rope(float*x,int seq,int nh,int hd){
    for(int t=0;t<seq;t++)for(int h=0;h<nh;h++){float*v=x+((size_t)t*nh+h)*hd;
        for(int i=0;i<hd/2;i++){float fr=powf(10000.0f,-2.0f*i/hd),ang=t*fr,c=cosf(ang),s=sinf(ang);
            float a=v[i],b=v[i+hd/2]; v[i]=a*c-b*s; v[i+hd/2]=a*s+b*c;}}
}
static void softmax(float*x,int n){ ork_softmax_f32(x,n); }

typedef struct { int K,N; f16 *Brow; ork_w *w; } weight;
static weight mkw(ork_npu*ctx,int K,int N,unsigned seed){weight wt={K,N,malloc((size_t)K*N*2),NULL};
    float sc=0.5f/sqrtf((float)K);   /* 1/sqrt(K) init so stacked-layer activations stay fp16-safe */
    unsigned s=seed; for(size_t i=0;i<(size_t)K*N;i++){s=s*1103515245+12345;wt.Brow[i]=(f16)((((int)((s>>16)%9))-4)*sc);}
    wt.w=ork_f16_mm_pack(ctx,K,N,wt.Brow); return wt;}
static void mm(ork_npu*ctx,weight*wt,int M,const float*Af32,float*C,int useNPU){
    int K=wt->K,N=wt->N; f16*A=malloc((size_t)M*K*2);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(f16)Af32[i];
    if(useNPU) ork_f16_mm_run(ctx,wt->w,M,A,C);
    else for(int m=0;m<M;m++)for(int n=0;n<N;n++){float acc=0;for(int k=0;k<K;k++)acc+=(float)A[(size_t)m*K+k]*(float)wt->Brow[(size_t)k*N+n];C[(size_t)m*N+n]=acc;}
    free(A);
}
typedef struct { float n1[H],n2[H]; weight Wq,Wk,Wv,Wo,Wg,Wu,Wd; } tlayer;
static void mklayer(ork_npu*ctx,tlayer*L,unsigned seed){
    for(int i=0;i<H;i++){L->n1[i]=1.0f+((int)((i+seed)%5)-2)*0.02f;L->n2[i]=1.0f+((int)((i+seed)%7)-3)*0.02f;}
    L->Wq=mkw(ctx,H,QD,seed+1);L->Wk=mkw(ctx,H,KVD,seed+2);L->Wv=mkw(ctx,H,KVD,seed+3);L->Wo=mkw(ctx,QD,H,seed+4);
    L->Wg=mkw(ctx,H,FFN,seed+5);L->Wu=mkw(ctx,H,FFN,seed+6);L->Wd=mkw(ctx,FFN,H,seed+7);
}
/* one GQA layer, in-place on x[SEQ*H] */
static void layer(ork_npu*ctx,tlayer*L,float*x,int useNPU){
    static float xn[SEQ*H],q[SEQ*QD],k[SEQ*KVD],v[SEQ*KVD],att[SEQ*QD],o[SEQ*H],h[SEQ*H];
    static float hn[SEQ*H],g[SEQ*FFN],u[SEQ*FFN],a[SEQ*FFN],d[SEQ*H];
    int grp=NH/NKV;
    for(int t=0;t<SEQ;t++) rmsnorm(xn+t*H,x+t*H,L->n1,H);
    mm(ctx,&L->Wq,SEQ,xn,q,useNPU);mm(ctx,&L->Wk,SEQ,xn,k,useNPU);mm(ctx,&L->Wv,SEQ,xn,v,useNPU);
    rope(q,SEQ,NH,HD);rope(k,SEQ,NKV,HD); float scale=1.0f/sqrtf((float)HD);
    for(int hh=0;hh<NH;hh++){int kvh=hh/grp; for(int i=0;i<SEQ;i++){float sc[SEQ];
        for(int j=0;j<=i;j++){float dt=0;for(int e=0;e<HD;e++)dt+=q[((size_t)i*NH+hh)*HD+e]*k[((size_t)j*NKV+kvh)*HD+e];sc[j]=dt*scale;}
        softmax(sc,i+1);
        for(int e=0;e<HD;e++){float ac=0;for(int j=0;j<=i;j++)ac+=sc[j]*v[((size_t)j*NKV+kvh)*HD+e];att[((size_t)i*NH+hh)*HD+e]=ac;}}}
    mm(ctx,&L->Wo,SEQ,att,o,useNPU);
    for(int i=0;i<SEQ*H;i++)h[i]=x[i]+o[i];
    for(int t=0;t<SEQ;t++) rmsnorm(hn+t*H,h+t*H,L->n2,H);
    mm(ctx,&L->Wg,SEQ,hn,g,useNPU);mm(ctx,&L->Wu,SEQ,hn,u,useNPU);
    ork_silu_mul_to_f32(a,g,u,SEQ*FFN);   /* SwiGLU (NEON) */
    mm(ctx,&L->Wd,SEQ,a,d,useNPU);
    for(int i=0;i<SEQ*H;i++)x[i]=h[i]+d[i];
}
/* FNV-1a 64. The seed (s=42) is fixed, so the NPU forward output xn_ is deterministic — assert its
 * checksum against a static golden (keyed by layer count) instead of recomputing the CPU-reference
 * forward every run (12 layers of CPU matmuls dominate `model 12`). The CPU reference + tolerance
 * compare are kept, and run only to regenerate a golden (ORK_REGEN=1) or diagnose (ORK_FULL_REF=1). */
static uint64_t fnv64(const void*p,size_t n){ const uint8_t*b=(const uint8_t*)p; uint64_t h=1469598103934665603ULL;
    for(size_t i=0;i<n;i++){ h^=b[i]; h*=1099511628211ULL; } return h; }
static uint64_t model_gold(int nl){ switch(nl){ case 1: return 0xf7130c221a756578ULL; case 12: return 0x046bcfec25cd6248ULL; default: return 0; } }
int main(int argc,char**argv){
    int NL=argc>1?atoi(argv[1]):6; if(NL>MAXL)NL=MAXL;
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    static float x0[SEQ*H],xn_[SEQ*H],xc[SEQ*H]; unsigned s=42;
    for(int i=0;i<SEQ*H;i++){s=s*1103515245+12345;x0[i]=(((int)((s>>16)%17))-8)*0.1f;}
    static tlayer L[MAXL]; for(int l=0;l<NL;l++) mklayer(ctx,&L[l],1000u+l*100u);
    memcpy(xn_,x0,sizeof xn_); for(int l=0;l<NL;l++) layer(ctx,&L[l],xn_,1);   /* NPU */
    uint64_t got=fnv64(xn_,sizeof xn_), gold=model_gold(NL);
    int regen=getenv("ORK_REGEN")!=NULL, ok;
    if(gold && got==gold && !regen && !getenv("ORK_FULL_REF")){
        printf("MODEL %d layers : NPU forward golden 0x%016llx OK\n",NL,(unsigned long long)got); ok=1;   /* fast: no CPU forward */
    } else {   /* preserved CPU-reference forward + tolerance compare */
        memcpy(xc,x0,sizeof xc);   for(int l=0;l<NL;l++) layer(ctx,&L[l],xc,0);    /* CPU */
        float maxabs=0,refmax=0; int nans=0;
        for(int i=0;i<SEQ*H;i++){if(!isfinite(xn_[i])||!isfinite(xc[i])){nans++;continue;}
            float e=fabsf(xn_[i]-xc[i]);if(e>maxabs)maxabs=e;if(fabsf(xc[i])>refmax)refmax=fabsf(xc[i]);}
        ok=nans==0 && refmax>1e-3f && maxabs<0.05f*refmax+1e-2f;
        if(regen||!gold) printf("REGEN model GOLD NL=%d = 0x%016llxULL  (maxabs=%.4g ref=%.3g %s)\n",NL,(unsigned long long)got,maxabs,refmax,ok?"ok":"BAD");
        else if(got!=gold) printf("MODEL %d GOLDEN MISMATCH: NPU output changed (maxabs=%.4g) — regen if intended\n",NL,maxabs);
        printf("MODEL %d layers H=%d NH=%d NKV=%d HD=%d FFN=%d SEQ=%d : NPU vs CPU  maxabs=%.4g (ref|max|=%.3g) nans=%d : %s\n",
            NL,H,NH,NKV,HD,FFN,SEQ,maxabs,refmax,nans,ok?"OK":"MISMATCH");
        if(gold && got!=gold && !regen) ok=0;
    }
    ork_npu_free(ctx);
    return ok?0:2;
}
