/* examples/layer.c — M4.2/M4.3: a full transformer decoder layer (Llama/Qwen style) with the
 * big projections on the NPU (ork_npu) and the non-matmul ops (RMSNorm, RoPE, softmax,
 * SwiGLU) on the CPU — the pragmatic hybrid graph. Validates the NPU-hybrid forward pass
 * against a pure-CPU reference (identical ops, CPU matmul) so the only variable is the
 * matmul engine; expects a tight relative match (fp16 matmul rounding only).
 *   cc -O2 -I. -o rknpu_layer rknpu_layer.c ork_npu.c -lm && sudo ./rknpu_layer
 *
 * Layer:  h = x + Wo·Attn(RoPE(Q,K),V) over heads, Q/K/V = Wqkv·RMSNorm(x)
 *         y = h + Wdown·( SiLU(Wgate·RMSNorm(h)) ⊙ (Wup·RMSNorm(h)) )
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ork_npu.h"
typedef ork_f16 f16;

/* ---- config (NPU constraints: K%32==0, N%16==0) ---- */
#define H     512      /* hidden */
#define NH    8        /* heads */
#define HD    64       /* head dim (NH*HD==H) */
#define FFN   2048     /* mlp intermediate */
#define SEQ   16       /* tokens */
#define EPS   1e-5f

/* ---- CPU ops (fp32) ---- */
static void rmsnorm(float*o,const float*x,const float*w,int n){
    float ss=0; for(int i=0;i<n;i++)ss+=x[i]*x[i]; float s=1.0f/sqrtf(ss/n+EPS);
    for(int i=0;i<n;i++)o[i]=x[i]*s*w[i];
}
static void rope(float*x,int seq,int nh,int hd){           /* GPT-NeoX rotate-halves */
    for(int t=0;t<seq;t++)for(int h=0;h<nh;h++){float*v=x+((size_t)t*nh+h)*hd;
        for(int i=0;i<hd/2;i++){float freq=powf(10000.0f,-2.0f*i/hd),ang=t*freq,c=cosf(ang),s=sinf(ang);
            float a=v[i],b=v[i+hd/2]; v[i]=a*c-b*s; v[i+hd/2]=a*s+b*c;}}
}
static void softmax(float*x,int n){float m=x[0];for(int i=1;i<n;i++)if(x[i]>m)m=x[i];
    float s=0;for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];} for(int i=0;i<n;i++)x[i]/=s;}
static float silu(float x){return x/(1.0f+expf(-x));}

/* ---- matmul: NPU (resident weights) or CPU fp16 reference, same fp16 inputs ---- */
typedef struct { int K,N; f16 *Brow; ork_w *w; } weight;
static weight mkw(ork_npu*ctx,int K,int N){weight wt={K,N,malloc((size_t)K*N*2),NULL};
    unsigned s=K*131+N*17+7; for(size_t i=0;i<(size_t)K*N;i++){s=s*1103515245+12345;wt.Brow[i]=(f16)(((int)((s>>16)%9))-4)*0.05f;}
    wt.w=ork_mm_pack(ctx,K,N,wt.Brow); return wt;}
/* C[M,N] fp32 = A[M,K] fp32 (cast to fp16) x weight; useNPU picks engine */
static void mm(ork_npu*ctx,weight*wt,int M,const float*Af32,float*C,int useNPU){
    int K=wt->K,N=wt->N; f16*A=malloc((size_t)M*K*2);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(f16)Af32[i];
    if(useNPU) ork_mm_run(ctx,wt->w,M,A,C);
    else for(int m=0;m<M;m++)for(int n=0;n<N;n++){float acc=0;for(int k=0;k<K;k++)acc+=(float)A[(size_t)m*K+k]*(float)wt->Brow[(size_t)k*N+n];C[(size_t)m*N+n]=acc;}
    free(A);
}

/* one decoder layer; writes y[SEQ*H]. useNPU selects matmul engine (ops always CPU). */
static void layer(ork_npu*ctx,const float*x,float*y,
        const float*wn1,const float*wn2,weight*Wq,weight*Wk,weight*Wv,weight*Wo,
        weight*Wg,weight*Wu,weight*Wd,int useNPU){
    static float xn[SEQ*H],q[SEQ*H],k[SEQ*H],v[SEQ*H],att[SEQ*H],o[SEQ*H],h[SEQ*H];
    static float hn[SEQ*H],g[SEQ*FFN],u[SEQ*FFN],a[SEQ*FFN],d[SEQ*H];
    for(int t=0;t<SEQ;t++) rmsnorm(xn+t*H,x+t*H,wn1,H);
    mm(ctx,Wq,SEQ,xn,q,useNPU); mm(ctx,Wk,SEQ,xn,k,useNPU); mm(ctx,Wv,SEQ,xn,v,useNPU);
    rope(q,SEQ,NH,HD); rope(k,SEQ,NH,HD);
    float scale=1.0f/sqrtf((float)HD);
    for(int hh=0;hh<NH;hh++)for(int i=0;i<SEQ;i++){
        float sc[SEQ];
        for(int j=0;j<=i;j++){float dt=0;for(int e=0;e<HD;e++)dt+=q[((size_t)i*NH+hh)*HD+e]*k[((size_t)j*NH+hh)*HD+e];sc[j]=dt*scale;}
        softmax(sc,i+1);
        for(int e=0;e<HD;e++){float acc=0;for(int j=0;j<=i;j++)acc+=sc[j]*v[((size_t)j*NH+hh)*HD+e];att[((size_t)i*NH+hh)*HD+e]=acc;}
    }
    mm(ctx,Wo,SEQ,att,o,useNPU);
    for(int i=0;i<SEQ*H;i++)h[i]=x[i]+o[i];
    for(int t=0;t<SEQ;t++) rmsnorm(hn+t*H,h+t*H,wn2,H);
    mm(ctx,Wg,SEQ,hn,g,useNPU); mm(ctx,Wu,SEQ,hn,u,useNPU);
    for(int i=0;i<SEQ*FFN;i++)a[i]=silu(g[i])*u[i];
    mm(ctx,Wd,SEQ,a,d,useNPU);
    for(int i=0;i<SEQ*H;i++)y[i]=h[i]+d[i];
}
int main(void){
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    float *x=malloc(SEQ*H*4),*yn=malloc(SEQ*H*4),*yc=malloc(SEQ*H*4);
    float wn1[H],wn2[H]; unsigned s=42;
    for(int i=0;i<SEQ*H;i++){s=s*1103515245+12345;x[i]=(((int)((s>>16)%17))-8)*0.1f;}
    for(int i=0;i<H;i++){wn1[i]=1.0f+((i%5)-2)*0.02f;wn2[i]=1.0f+((i%7)-3)*0.02f;}
    weight Wq=mkw(ctx,H,H),Wk=mkw(ctx,H,H),Wv=mkw(ctx,H,H),Wo=mkw(ctx,H,H);
    weight Wg=mkw(ctx,H,FFN),Wu=mkw(ctx,H,FFN),Wd=mkw(ctx,FFN,H);
    layer(ctx,x,yn,wn1,wn2,&Wq,&Wk,&Wv,&Wo,&Wg,&Wu,&Wd,1);   /* NPU hybrid */
    layer(ctx,x,yc,wn1,wn2,&Wq,&Wk,&Wv,&Wo,&Wg,&Wu,&Wd,0);   /* CPU reference */
    float maxabs=0,maxrel=0,refmax=0;
    for(int i=0;i<SEQ*H;i++){float e=fabsf(yn[i]-yc[i]),ra=fabsf(yc[i]); if(e>maxabs)maxabs=e; if(ra>refmax)refmax=ra;
        float rel=e/(ra+1e-6f); if(rel>maxrel)maxrel=rel;}
    int ok = maxabs < 0.05f*refmax + 1e-3f;     /* fp16 matmul tolerance */
    printf("LAYER H=%d NH=%d HD=%d FFN=%d SEQ=%d : NPU-hybrid vs CPU-ref  maxabs=%.4g (ref|max|=%.3g) maxrel=%.3g : %s\n",
        H,NH,HD,FFN,SEQ,maxabs,refmax,maxrel,ok?"OK":"MISMATCH");
    ork_npu_free(ctx);
    return ok?0:2;
}
