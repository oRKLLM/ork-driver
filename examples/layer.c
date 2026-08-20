/* examples/layer.c — a transformer decoder layer (Llama/Qwen style) with the big projections
 * on the NPU (ork_npu) and the non-matmul ops (RMSNorm, RoPE, softmax, SwiGLU) on the CPU.
 * Supports grouped-query attention (n_kv_heads <= n_heads) and an arbitrary head_dim
 * (n_heads*head_dim need not equal hidden) — what real models (Qwen3 etc.) need. Validates
 * the NPU-hybrid forward against a pure-CPU reference across MHA / GQA / arbitrary-HD configs.
 *   make layer && sudo ./layer
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ork_npu.h"
typedef ork_f16 f16;
#define EPS 1e-5f
#define SEQ 16

/* config: hidden H, query heads NH, kv heads NKV (GQA when NKV<NH), head dim HD
 * (NH*HD = q-dim, may differ from H), mlp FFN. NPU dims must satisfy K%32, N%16. */
typedef struct { const char*name; int H,NH,NKV,HD,FFN; } Cfg;

static void rmsnorm(float*o,const float*x,const float*w,int n){
    float ss=0; for(int i=0;i<n;i++)ss+=x[i]*x[i]; float s=1.0f/sqrtf(ss/n+EPS);
    for(int i=0;i<n;i++)o[i]=x[i]*s*w[i];
}
static void rope(float*x,int seq,int nh,int hd){           /* NeoX rotate-halves, per head */
    for(int t=0;t<seq;t++)for(int h=0;h<nh;h++){float*v=x+((size_t)t*nh+h)*hd;
        for(int i=0;i<hd/2;i++){float fr=powf(10000.0f,-2.0f*i/hd),ang=t*fr,c=cosf(ang),s=sinf(ang);
            float a=v[i],b=v[i+hd/2]; v[i]=a*c-b*s; v[i+hd/2]=a*s+b*c;}}
}
static void softmax(float*x,int n){float m=x[0];for(int i=1;i<n;i++)if(x[i]>m)m=x[i];
    float s=0;for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];} for(int i=0;i<n;i++)x[i]/=s;}
static float silu(float x){return x/(1.0f+expf(-x));}

typedef struct { int K,N; f16 *Brow; ork_w *w; } weight;
static weight mkw(ork_npu*ctx,int K,int N,unsigned seed){weight wt={K,N,malloc((size_t)K*N*2),NULL};
    unsigned s=seed; for(size_t i=0;i<(size_t)K*N;i++){s=s*1103515245+12345;wt.Brow[i]=(f16)((((int)((s>>16)%9))-4)*(0.5f/sqrtf((float)K)));}
    wt.w=ork_f16_mm_pack(ctx,K,N,wt.Brow); return wt;}
static void mm(ork_npu*ctx,weight*wt,int M,const float*Af32,float*C,int useNPU){
    int K=wt->K,N=wt->N; f16*A=malloc((size_t)M*K*2);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(f16)Af32[i];
    if(useNPU) ork_f16_mm_run(ctx,wt->w,M,A,C);
    else for(int m=0;m<M;m++)for(int n=0;n<N;n++){float acc=0;for(int k=0;k<K;k++)acc+=(float)A[(size_t)m*K+k]*(float)wt->Brow[(size_t)k*N+n];C[(size_t)m*N+n]=acc;}
    free(A);
}
/* GQA decoder layer; writes y[SEQ*H]. q-dim=NH*HD, kv-dim=NKV*HD, group=NH/NKV. */
static void layer(ork_npu*ctx,const Cfg*c,const float*x,float*y,const float*wn1,const float*wn2,
        weight*Wq,weight*Wk,weight*Wv,weight*Wo,weight*Wg,weight*Wu,weight*Wd,int useNPU){
    int H=c->H,NH=c->NH,NKV=c->NKV,HD=c->HD,FFN=c->FFN, Qd=NH*HD,KVd=NKV*HD,grp=NH/NKV;
    float*xn=malloc(SEQ*H*4),*q=malloc((size_t)SEQ*Qd*4),*k=malloc((size_t)SEQ*KVd*4),*v=malloc((size_t)SEQ*KVd*4);
    float*att=malloc((size_t)SEQ*Qd*4),*o=malloc(SEQ*H*4),*h=malloc(SEQ*H*4),*hn=malloc(SEQ*H*4);
    float*g=malloc((size_t)SEQ*FFN*4),*u=malloc((size_t)SEQ*FFN*4),*a=malloc((size_t)SEQ*FFN*4),*d=malloc(SEQ*H*4);
    for(int t=0;t<SEQ;t++) rmsnorm(xn+t*H,x+t*H,wn1,H);
    mm(ctx,Wq,SEQ,xn,q,useNPU); mm(ctx,Wk,SEQ,xn,k,useNPU); mm(ctx,Wv,SEQ,xn,v,useNPU);
    rope(q,SEQ,NH,HD); rope(k,SEQ,NKV,HD);
    float scale=1.0f/sqrtf((float)HD);
    for(int hh=0;hh<NH;hh++){int kvh=hh/grp;          /* GQA: query head -> shared kv head */
      for(int i=0;i<SEQ;i++){float sc[SEQ];
        for(int j=0;j<=i;j++){float dt=0;for(int e=0;e<HD;e++)dt+=q[((size_t)i*NH+hh)*HD+e]*k[((size_t)j*NKV+kvh)*HD+e];sc[j]=dt*scale;}
        softmax(sc,i+1);
        for(int e=0;e<HD;e++){float ac=0;for(int j=0;j<=i;j++)ac+=sc[j]*v[((size_t)j*NKV+kvh)*HD+e];att[((size_t)i*NH+hh)*HD+e]=ac;}
      }}
    mm(ctx,Wo,SEQ,att,o,useNPU);
    for(int i=0;i<SEQ*H;i++)h[i]=x[i]+o[i];
    for(int t=0;t<SEQ;t++) rmsnorm(hn+t*H,h+t*H,wn2,H);
    mm(ctx,Wg,SEQ,hn,g,useNPU); mm(ctx,Wu,SEQ,hn,u,useNPU);
    for(int i=0;i<SEQ*FFN;i++)a[i]=silu(g[i])*u[i];
    mm(ctx,Wd,SEQ,a,d,useNPU);
    for(int i=0;i<SEQ*H;i++)y[i]=h[i]+d[i];
    free(xn);free(q);free(k);free(v);free(att);free(o);free(h);free(hn);free(g);free(u);free(a);free(d);
}
static int run_cfg(ork_npu*ctx,const Cfg*c){
    int H=c->H,NH=c->NH,NKV=c->NKV,HD=c->HD,FFN=c->FFN, Qd=NH*HD,KVd=NKV*HD;
    float *x=malloc(SEQ*H*4),*yn=malloc(SEQ*H*4),*yc=malloc(SEQ*H*4),*wn1=malloc(H*4),*wn2=malloc(H*4);
    unsigned s=42; for(int i=0;i<SEQ*H;i++){s=s*1103515245+12345;x[i]=(((int)((s>>16)%17))-8)*0.1f;}
    for(int i=0;i<H;i++){wn1[i]=1.0f+((int)((i+1u)%5)-2)*0.02f;wn2[i]=1.0f+((int)((i+1u)%7)-3)*0.02f;}
    weight Wq=mkw(ctx,H,Qd,11),Wk=mkw(ctx,H,KVd,22),Wv=mkw(ctx,H,KVd,33),Wo=mkw(ctx,Qd,H,44);
    weight Wg=mkw(ctx,H,FFN,55),Wu=mkw(ctx,H,FFN,66),Wd=mkw(ctx,FFN,H,77);
    layer(ctx,c,x,yn,wn1,wn2,&Wq,&Wk,&Wv,&Wo,&Wg,&Wu,&Wd,1);
    layer(ctx,c,x,yc,wn1,wn2,&Wq,&Wk,&Wv,&Wo,&Wg,&Wu,&Wd,0);
    float maxabs=0,refmax=0; int nans=0;
    for(int i=0;i<SEQ*H;i++){if(!isfinite(yn[i])||!isfinite(yc[i])){nans++;continue;}
        float e=fabsf(yn[i]-yc[i]);if(e>maxabs)maxabs=e;if(fabsf(yc[i])>refmax)refmax=fabsf(yc[i]);}
    int ok=nans==0 && refmax>1e-3f && maxabs<0.05f*refmax+1e-2f;
    printf("  %-10s H=%d NH=%d NKV=%d HD=%d (q=%d kv=%d) FFN=%d : maxabs=%.4g |ref|=%.3g : %s\n",
        c->name,H,NH,NKV,HD,Qd,KVd,FFN,maxabs,refmax,ok?"OK":"MISMATCH");
    ork_w_free(Wq.w);ork_w_free(Wk.w);ork_w_free(Wv.w);ork_w_free(Wo.w);ork_w_free(Wg.w);ork_w_free(Wu.w);ork_w_free(Wd.w);
    free(Wq.Brow);free(Wk.Brow);free(Wv.Brow);free(Wo.Brow);free(Wg.Brow);free(Wu.Brow);free(Wd.Brow);
    free(x);free(yn);free(yc);free(wn1);free(wn2);
    return ok?0:1;
}
int main(void){
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    Cfg cfgs[]={
        {"MHA",       512,8,8,64,2048},   /* baseline: kv heads == q heads */
        {"GQA-4x",    512,8,2,64,2048},   /* 8 q heads share 2 kv heads (group 4) */
        {"GQA-arbHD", 512,4,2,96,2048},   /* arbitrary head_dim: NH*HD=384 != H=512 */
    };
    int fail=0; printf("decoder layer (NPU-hybrid vs CPU ref):\n");
    for(unsigned i=0;i<sizeof cfgs/sizeof*cfgs;i++) fail|=run_cfg(ctx,&cfgs[i]);
    ork_npu_free(ctx);
    printf("%s\n",fail?"FAIL":"ALL OK"); return fail?2:0;
}
