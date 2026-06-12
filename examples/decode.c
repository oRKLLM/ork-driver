/* examples/decode.c — M4.4: incremental decode with a KV cache (the token-generation path).
 * Same decoder layer as rknpu_layer.c, but processes ONE token at a time: each step does
 * M=1 NPU projections, RoPE at the running position, appends K/V to a per-layer cache, and
 * attends over the cached keys/values. Validates the NPU-hybrid decode against a pure-CPU
 * reference (identical ops + KV cache, CPU matmul) over a full sequence.
 *   cc -O2 -I. -o rknpu_decode rknpu_decode.c ork_npu.c -lm && sudo ./rknpu_decode
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ork_npu.h"
typedef ork_f16 f16;
#define H 512
#define NH 8
#define HD 64
#define FFN 2048
#define SEQ 16
#define EPS 1e-5f

static void rmsnorm(float*o,const float*x,const float*w,int n){
    float ss=0; for(int i=0;i<n;i++)ss+=x[i]*x[i]; float s=1.0f/sqrtf(ss/n+EPS);
    for(int i=0;i<n;i++)o[i]=x[i]*s*w[i];
}
/* RoPE one row [nh,hd] at absolute position pos */
static void rope_pos(float*x,int nh,int hd,int pos){
    for(int h=0;h<nh;h++){float*v=x+(size_t)h*hd;
        for(int i=0;i<hd/2;i++){float freq=powf(10000.0f,-2.0f*i/hd),ang=pos*freq,c=cosf(ang),s=sinf(ang);
            float a=v[i],b=v[i+hd/2]; v[i]=a*c-b*s; v[i+hd/2]=a*s+b*c;}}
}
static void softmax(float*x,int n){float m=x[0];for(int i=1;i<n;i++)if(x[i]>m)m=x[i];
    float s=0;for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];} for(int i=0;i<n;i++)x[i]/=s;}
static float silu(float x){return x/(1.0f+expf(-x));}

typedef struct { int K,N; f16 *Brow; ork_w *w; } weight;
static weight mkw(ork_npu*ctx,int K,int N){weight wt={K,N,malloc((size_t)K*N*2),NULL};
    unsigned s=K*131+N*17+7; for(size_t i=0;i<(size_t)K*N;i++){s=s*1103515245+12345;wt.Brow[i]=(f16)(((int)((s>>16)%9))-4)*0.05f;}
    wt.w=ork_mm_pack(ctx,K,N,wt.Brow); return wt;}
static void mm1(ork_npu*ctx,weight*wt,const float*Af32,float*C,int useNPU){   /* M=1 */
    int K=wt->K,N=wt->N; f16*A=malloc((size_t)K*2);
    for(int i=0;i<K;i++)A[i]=(f16)Af32[i];
    if(useNPU) ork_mm_run(ctx,wt->w,1,A,C);
    else for(int n=0;n<N;n++){float acc=0;for(int k=0;k<K;k++)acc+=(float)A[k]*(float)wt->Brow[(size_t)k*N+n];C[n]=acc;}
    free(A);
}
/* per-layer KV cache */
typedef struct { float Kc[SEQ*H], Vc[SEQ*H]; int len; } kv_t;

/* one decode step for token x1[H] at position kv->len; writes y1[H] */
static void step(ork_npu*ctx,const float*x1,float*y1,kv_t*kv,const float*wn1,const float*wn2,
        weight*Wq,weight*Wk,weight*Wv,weight*Wo,weight*Wg,weight*Wu,weight*Wd,int useNPU){
    float xn[H],q[H],k[H],v[H],att[H],o[H],h[H],hn[H],g[FFN],u[FFN],a[FFN],d[H];
    int pos=kv->len;
    rmsnorm(xn,x1,wn1,H);
    mm1(ctx,Wq,xn,q,useNPU); mm1(ctx,Wk,xn,k,useNPU); mm1(ctx,Wv,xn,v,useNPU);
    rope_pos(q,NH,HD,pos); rope_pos(k,NH,HD,pos);
    memcpy(kv->Kc+(size_t)pos*H,k,H*4); memcpy(kv->Vc+(size_t)pos*H,v,H*4); kv->len++;
    float scale=1.0f/sqrtf((float)HD);
    for(int hh=0;hh<NH;hh++){
        float sc[SEQ];
        for(int j=0;j<=pos;j++){float dt=0;for(int e=0;e<HD;e++)dt+=q[hh*HD+e]*kv->Kc[(size_t)j*H+hh*HD+e];sc[j]=dt*scale;}
        softmax(sc,pos+1);
        for(int e=0;e<HD;e++){float acc=0;for(int j=0;j<=pos;j++)acc+=sc[j]*kv->Vc[(size_t)j*H+hh*HD+e];att[hh*HD+e]=acc;}
    }
    mm1(ctx,Wo,att,o,useNPU);
    for(int i=0;i<H;i++)h[i]=x1[i]+o[i];
    rmsnorm(hn,h,wn2,H);
    mm1(ctx,Wg,hn,g,useNPU); mm1(ctx,Wu,hn,u,useNPU);
    for(int i=0;i<FFN;i++)a[i]=silu(g[i])*u[i];
    mm1(ctx,Wd,a,d,useNPU);
    for(int i=0;i<H;i++)y1[i]=h[i]+d[i];
}
int main(void){
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    static float x[SEQ*H],yn[SEQ*H],yc[SEQ*H]; float wn1[H],wn2[H]; unsigned s=42;
    for(int i=0;i<SEQ*H;i++){s=s*1103515245+12345;x[i]=(((int)((s>>16)%17))-8)*0.1f;}
    for(int i=0;i<H;i++){wn1[i]=1.0f+((i%5)-2)*0.02f;wn2[i]=1.0f+((i%7)-3)*0.02f;}
    weight Wq=mkw(ctx,H,H),Wk=mkw(ctx,H,H),Wv=mkw(ctx,H,H),Wo=mkw(ctx,H,H);
    weight Wg=mkw(ctx,H,FFN),Wu=mkw(ctx,H,FFN),Wd=mkw(ctx,FFN,H);
    kv_t kvn={0},kvc={0};
    for(int t=0;t<SEQ;t++) step(ctx,x+t*H,yn+t*H,&kvn,wn1,wn2,&Wq,&Wk,&Wv,&Wo,&Wg,&Wu,&Wd,1);
    for(int t=0;t<SEQ;t++) step(ctx,x+t*H,yc+t*H,&kvc,wn1,wn2,&Wq,&Wk,&Wv,&Wo,&Wg,&Wu,&Wd,0);
    float maxabs=0,refmax=0;
    for(int i=0;i<SEQ*H;i++){float e=fabsf(yn[i]-yc[i]);if(e>maxabs)maxabs=e;if(fabsf(yc[i])>refmax)refmax=fabsf(yc[i]);}
    int ok=maxabs<0.05f*refmax+1e-3f;
    printf("DECODE+KVcache H=%d NH=%d FFN=%d steps=%d : NPU vs CPU  maxabs=%.4g (ref|max|=%.3g) : %s\n",
        H,NH,FFN,SEQ,maxabs,refmax,ok?"OK":"MISMATCH");
    ork_npu_free(ctx);
    return ok?0:2;
}
