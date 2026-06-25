/* examples/decode.c — incremental decode with a KV cache (the token-generation path), with
 * grouped-query attention + arbitrary head_dim. Each step does M=1 NPU projections, RoPE at
 * the running position, appends K/V (only n_kv_heads worth) to a per-layer cache, and attends
 * over the cache with query heads sharing kv heads. Validates NPU-hybrid decode vs a pure-CPU
 * reference across MHA / GQA / arbitrary-HD configs.
 *   make decode && sudo ./decode
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ork_npu.h"
#include "neon_activations.h"
typedef ork_f16 f16;
#define EPS 1e-5f
#define SEQ 16

typedef struct { const char*name; int H,NH,NKV,HD,FFN; } Cfg;

static void rmsnorm(float*o,const float*x,const float*w,int n){ ork_rmsnorm_f32(o,x,w,n,EPS); }
static void rope_pos(float*x,int nh,int hd,int pos){       /* per head, at absolute position */
    for(int h=0;h<nh;h++){float*v=x+(size_t)h*hd;
        for(int i=0;i<hd/2;i++){float fr=powf(10000.0f,-2.0f*i/hd),ang=pos*fr,c=cosf(ang),s=sinf(ang);
            float a=v[i],b=v[i+hd/2]; v[i]=a*c-b*s; v[i+hd/2]=a*s+b*c;}}
}
static void softmax(float*x,int n){ ork_softmax_f32(x,n); }

typedef struct { int K,N; f16 *Brow; ork_w *w; } weight;
static weight mkw(ork_npu*ctx,int K,int N,unsigned seed){weight wt={K,N,malloc((size_t)K*N*2),NULL};
    unsigned s=seed; for(size_t i=0;i<(size_t)K*N;i++){s=s*1103515245+12345;wt.Brow[i]=(f16)((((int)((s>>16)%9))-4)*(0.5f/sqrtf((float)K)));}
    wt.w=ork_mm_pack(ctx,K,N,wt.Brow); return wt;}
static void mm1(ork_npu*ctx,weight*wt,const float*Af32,float*C,int useNPU){   /* M=1 */
    int K=wt->K,N=wt->N; f16*A=malloc((size_t)K*2);
    for(int i=0;i<K;i++)A[i]=(f16)Af32[i];
    if(useNPU) ork_mm_run(ctx,wt->w,1,A,C);
    else for(int n=0;n<N;n++){float acc=0;for(int k=0;k<K;k++)acc+=(float)A[k]*(float)wt->Brow[(size_t)k*N+n];C[n]=acc;}
    free(A);
}
typedef struct { float*Kc,*Vc; int len; } kv_t;   /* cache holds KVd = NKV*HD per position */

static void step(ork_npu*ctx,const Cfg*c,const float*x1,float*y1,kv_t*kv,const float*wn1,const float*wn2,
        weight*Wq,weight*Wk,weight*Wv,weight*Wo,weight*Wg,weight*Wu,weight*Wd,int useNPU){
    int H=c->H,NH=c->NH,NKV=c->NKV,HD=c->HD,FFN=c->FFN, Qd=NH*HD,KVd=NKV*HD,grp=NH/NKV, pos=kv->len;
    float*xn=malloc(H*4),*q=malloc(Qd*4),*k=malloc(KVd*4),*v=malloc(KVd*4),*att=malloc(Qd*4),*o=malloc(H*4),*h=malloc(H*4),*hn=malloc(H*4),*g=malloc(FFN*4),*u=malloc(FFN*4),*a=malloc(FFN*4),*d=malloc(H*4);
    rmsnorm(xn,x1,wn1,H);
    mm1(ctx,Wq,xn,q,useNPU); mm1(ctx,Wk,xn,k,useNPU); mm1(ctx,Wv,xn,v,useNPU);
    rope_pos(q,NH,HD,pos); rope_pos(k,NKV,HD,pos);
    memcpy(kv->Kc+(size_t)pos*KVd,k,KVd*4); memcpy(kv->Vc+(size_t)pos*KVd,v,KVd*4); kv->len++;
    float scale=1.0f/sqrtf((float)HD);
    for(int hh=0;hh<NH;hh++){int kvh=hh/grp; float sc[SEQ];
        for(int j=0;j<=pos;j++){float dt=0;for(int e=0;e<HD;e++)dt+=q[hh*HD+e]*kv->Kc[(size_t)j*KVd+kvh*HD+e];sc[j]=dt*scale;}
        softmax(sc,pos+1);
        for(int e=0;e<HD;e++){float ac=0;for(int j=0;j<=pos;j++)ac+=sc[j]*kv->Vc[(size_t)j*KVd+kvh*HD+e];att[hh*HD+e]=ac;}
    }
    mm1(ctx,Wo,att,o,useNPU);
    for(int i=0;i<H;i++)h[i]=x1[i]+o[i];
    rmsnorm(hn,h,wn2,H);
    mm1(ctx,Wg,hn,g,useNPU); mm1(ctx,Wu,hn,u,useNPU);
    ork_silu_mul_to_f32(a,g,u,FFN);   /* SwiGLU: a = silu(g)*u (NEON) */
    mm1(ctx,Wd,a,d,useNPU);
    for(int i=0;i<H;i++)y1[i]=h[i]+d[i];
    free(xn);free(q);free(k);free(v);free(att);free(o);free(h);free(hn);free(g);free(u);free(a);free(d);
}
static int run_cfg(ork_npu*ctx,const Cfg*c){
    int H=c->H,NH=c->NH,NKV=c->NKV,HD=c->HD,FFN=c->FFN, Qd=NH*HD,KVd=NKV*HD;
    float*x=malloc(SEQ*H*4),*yn=malloc(SEQ*H*4),*yc=malloc(SEQ*H*4),*wn1=malloc(H*4),*wn2=malloc(H*4);
    unsigned s=42; for(int i=0;i<SEQ*H;i++){s=s*1103515245+12345;x[i]=(((int)((s>>16)%17))-8)*0.1f;}
    for(int i=0;i<H;i++){wn1[i]=1.0f+((int)((i+1u)%5)-2)*0.02f;wn2[i]=1.0f+((int)((i+1u)%7)-3)*0.02f;}
    weight Wq=mkw(ctx,H,Qd,11),Wk=mkw(ctx,H,KVd,22),Wv=mkw(ctx,H,KVd,33),Wo=mkw(ctx,Qd,H,44);
    weight Wg=mkw(ctx,H,FFN,55),Wu=mkw(ctx,H,FFN,66),Wd=mkw(ctx,FFN,H,77);
    kv_t kvn={malloc((size_t)SEQ*KVd*4),malloc((size_t)SEQ*KVd*4),0};
    kv_t kvc={malloc((size_t)SEQ*KVd*4),malloc((size_t)SEQ*KVd*4),0};
    for(int t=0;t<SEQ;t++) step(ctx,c,x+t*H,yn+t*H,&kvn,wn1,wn2,&Wq,&Wk,&Wv,&Wo,&Wg,&Wu,&Wd,1);
    for(int t=0;t<SEQ;t++) step(ctx,c,x+t*H,yc+t*H,&kvc,wn1,wn2,&Wq,&Wk,&Wv,&Wo,&Wg,&Wu,&Wd,0);
    float maxabs=0,refmax=0; int nans=0;
    for(int i=0;i<SEQ*H;i++){if(!isfinite(yn[i])||!isfinite(yc[i])){nans++;continue;}
        float e=fabsf(yn[i]-yc[i]);if(e>maxabs)maxabs=e;if(fabsf(yc[i])>refmax)refmax=fabsf(yc[i]);}
    int ok=nans==0 && refmax>1e-3f && maxabs<0.05f*refmax+1e-2f;
    printf("  %-10s H=%d NH=%d NKV=%d HD=%d (kv-cache %d/pos) : maxabs=%.4g |ref|=%.3g : %s\n",
        c->name,H,NH,NKV,HD,KVd,maxabs,refmax,ok?"OK":"MISMATCH");
    ork_w_free(Wq.w);ork_w_free(Wk.w);ork_w_free(Wv.w);ork_w_free(Wo.w);ork_w_free(Wg.w);ork_w_free(Wu.w);ork_w_free(Wd.w);
    free(Wq.Brow);free(Wk.Brow);free(Wv.Brow);free(Wo.Brow);free(Wg.Brow);free(Wu.Brow);free(Wd.Brow);
    free(kvn.Kc);free(kvn.Vc);free(kvc.Kc);free(kvc.Vc); free(x);free(yn);free(yc);free(wn1);free(wn2);
    return ok?0:1;
}
int main(void){
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    Cfg cfgs[]={{"MHA",512,8,8,64,2048},{"GQA-4x",512,8,2,64,2048},{"GQA-arbHD",512,4,2,96,2048}};
    int fail=0; printf("KV-cache decode (NPU vs CPU ref):\n");
    for(unsigned i=0;i<sizeof cfgs/sizeof*cfgs;i++) fail|=run_cfg(ctx,&cfgs[i]);
    ork_npu_free(ctx);
    printf("%s\n",fail?"FAIL":"ALL OK"); return fail?2:0;
}
