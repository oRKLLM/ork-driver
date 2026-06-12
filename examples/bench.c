/* examples/bench.c — end-to-end throughput benchmark: decode + prefill tok/s for a real-size
 * GQA transformer (matmul on NPU, ops on CPU). Synthetic weights (measuring speed, not output).
 * fp16 or int8/w8a8 weights. Reports per-token latency vs the closed runtime.
 *   make bench && sudo ./bench [layers] [decode_tok] [prefill_tok] [dtype: f16|i8]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include "ork_npu.h"
typedef ork_f16 f16;
/* Qwen3-1.7B config (for apples-to-apples vs librkllmrt's Qwen3-1.7B w8a8). NPU dims: K%32, N%16. */
#define H 2048
#define NH 16
#define NKV 8
#define HD 128
#define FFN 6144
#define VOCAB 151936
#define QD (NH*HD)
#define KVD (NKV*HD)
#define MAXSEQ 256
#define EPS 1e-5f

static int g_i8=0;          /* 0 = fp16 weights, 1 = int8/w8a8 */
static double g_mm=0;       /* accumulated time inside ork_mm_run* (NPU + library) */
static double g_att=0;      /* accumulated time in the (CPU) attention block */
/* big.LITTLE: attention threads must run on the PERFORMANCE cores (A55 stragglers gate the join).
 * Detect the big cores (highest cpufreq max) once; pin attention there, oversubscribed 2x (hides
 * KV-cache read latency). RK3588: cpu4-7 A76 @2.3GHz vs cpu0-3 A55 @1.8GHz. */
static int g_big[16], g_nbig=0;
static void att_init(void){
    if(g_nbig) return;
    int nc=(int)sysconf(_SC_NPROCESSORS_ONLN); if(nc>16)nc=16; long mx=0,f[16]={0};
    for(int c=0;c<nc;c++){char p[96];snprintf(p,sizeof p,"/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",c);
        FILE*fp=fopen(p,"r"); if(fp){if(fscanf(fp,"%ld",&f[c])!=1)f[c]=0;fclose(fp);} if(f[c]>mx)mx=f[c];}
    for(int c=0;c<nc;c++) if(mx>0 && f[c]*10>=mx*9) g_big[g_nbig++]=c;  /* within 90% of top = a perf core (A76 clusters differ slightly: 2304 vs 2352 MHz) */
    if(!g_nbig){ for(int c=0;c<nc&&c<16;c++)g_big[g_nbig++]=c; }   /* fallback: all cores */
}
static int att_threads(void){ att_init(); const char*e=getenv("ORK_ATT_THREADS"); int v=e?atoi(e):2*g_nbig; if(v<1)v=1; return v; }
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
static void rmsnorm(float*o,const float*x,const float*w,int n){float ss=0;for(int i=0;i<n;i++)ss+=x[i]*x[i];float s=1.0f/sqrtf(ss/n+EPS);for(int i=0;i<n;i++)o[i]=x[i]*s*w[i];}
static void rope(float*x,int seq,int nh,int hd,int p0){for(int t=0;t<seq;t++)for(int h=0;h<nh;h++){float*v=x+((size_t)t*nh+h)*hd;for(int i=0;i<hd/2;i++){float fr=powf(10000.0f,-2.0f*i/hd),an=(p0+t)*fr,c=cosf(an),s=sinf(an);float a=v[i],b=v[i+hd/2];v[i]=a*c-b*s;v[i+hd/2]=a*s+b*c;}}}
static void softmax(float*x,int n){float m=x[0];for(int i=1;i<n;i++)if(x[i]>m)m=x[i];float s=0;for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];}for(int i=0;i<n;i++)x[i]/=s;}
static float silu(float x){return x/(1.0f+expf(-x));}

typedef struct { ork_w*Wq,*Wk,*Wv,*Wo,*Wg,*Wu,*Wd; float n1[H],n2[H]; } Layer;
static ork_w* mkw(ork_npu*ctx,int K,int N){
    if(g_i8){ int8_t*B=malloc((size_t)K*N);unsigned s=K*7+N*13+1;for(size_t i=0;i<(size_t)K*N;i++){s=s*1103515245+12345;B[i]=(int8_t)(((int)((s>>16)%7))-3);}ork_w*w=ork_mm_pack_i8(ctx,K,N,B);free(B);return w; }
    f16*B=malloc((size_t)K*N*2);unsigned s=K*7+N*13+1;float sc=0.4f/sqrtf((float)K);
    for(size_t i=0;i<(size_t)K*N;i++){s=s*1103515245+12345;B[i]=(f16)((((int)((s>>16)%9))-4)*sc);} ork_w*w=ork_mm_pack(ctx,K,N,B);free(B);return w;}
static void mkl(ork_npu*ctx,Layer*L){for(int i=0;i<H;i++){L->n1[i]=1.0f;L->n2[i]=1.0f;}
    L->Wq=mkw(ctx,H,QD);L->Wk=mkw(ctx,H,KVD);L->Wv=mkw(ctx,H,KVD);L->Wo=mkw(ctx,QD,H);L->Wg=mkw(ctx,H,FFN);L->Wu=mkw(ctx,H,FFN);L->Wd=mkw(ctx,FFN,H);}
/* C[M,N] = A[M,K] x packed weights. fp16: cast A->fp16. int8: quantize A->int8, run, dequant
 * (dummy per-tensor scale — bench measures matmul throughput, not numerics). */
static void mm(ork_npu*ctx,ork_w*w,int K,int N,int M,const float*Af,float*C){
    if(g_i8){ int8_t*A=malloc((size_t)M*K);int32_t*Ci=malloc((size_t)M*N*4);
        for(size_t i=0;i<(size_t)M*K;i++){int q=(int)lrintf(Af[i]*8.0f);A[i]=(int8_t)(q<-127?-127:q>127?127:q);}
        double t=now(); ork_mm_run_i8(ctx,w,M,A,Ci); g_mm+=now()-t;
        for(size_t i=0;i<(size_t)M*N;i++)C[i]=Ci[i]*(1.0f/8.0f);
        free(A);free(Ci); return; }
    f16*A=malloc((size_t)M*K*2);for(size_t i=0;i<(size_t)M*K;i++)A[i]=(f16)Af[i];
    double t=now(); ork_mm_run(ctx,w,M,A,C); g_mm+=now()-t; free(A);
}
/* Attention is the prefill bottleneck (O(M^2 * heads * HD), and it's ~65-72% of prefill time on
 * the CPU). The NPU is idle during it, so parallelize across heads (independent; each writes its
 * own att columns) over the CPU cores. */
#define ATT_THREADS 16
struct attw { const float *q,*Kc,*Vc; float *att; int M,p0,grp,h0,h1,core; };
static void *attworker(void *vp){
    struct attw *a=vp; float scale=1.0f/sqrtf((float)HD);
    if(a->core>=0){ cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(a->core,&cs); sched_setaffinity(0,sizeof cs,&cs); }
    for(int hh=a->h0;hh<a->h1;hh++){int kvh=hh/a->grp;
        for(int t=0;t<a->M;t++){int L2=a->p0+t+1; float sc[MAXSEQ];
            for(int j=0;j<L2;j++){float dt=0;for(int e=0;e<HD;e++)dt+=a->q[((size_t)t*NH+hh)*HD+e]*a->Kc[(size_t)j*KVD+kvh*HD+e];sc[j]=dt*scale;}
            softmax(sc,L2);
            for(int e=0;e<HD;e++){float ac=0;for(int j=0;j<L2;j++)ac+=sc[j]*a->Vc[(size_t)j*KVD+kvh*HD+e];a->att[((size_t)t*NH+hh)*HD+e]=ac;}}}
    return NULL;
}
static void layer(ork_npu*ctx,Layer*L,int M,int p0,float*x,float*Kc,float*Vc){
    int grp=NH/NKV; float scale=1.0f/sqrtf((float)HD); (void)scale;
    float*xn=malloc((size_t)M*H*4),*q=malloc((size_t)M*QD*4),*k=malloc((size_t)M*KVD*4),*v=malloc((size_t)M*KVD*4),*att=malloc((size_t)M*QD*4),*o=malloc((size_t)M*H*4),*hn=malloc((size_t)M*H*4),*g=malloc((size_t)M*FFN*4),*u=malloc((size_t)M*FFN*4),*a=malloc((size_t)M*FFN*4),*d=malloc((size_t)M*H*4);
    for(int t=0;t<M;t++)rmsnorm(xn+(size_t)t*H,x+(size_t)t*H,L->n1,H);
    mm(ctx,L->Wq,H,QD,M,xn,q);mm(ctx,L->Wk,H,KVD,M,xn,k);mm(ctx,L->Wv,H,KVD,M,xn,v);
    rope(q,M,NH,HD,p0);rope(k,M,NKV,HD,p0);
    for(int t=0;t<M;t++){memcpy(Kc+(size_t)(p0+t)*KVD,k+(size_t)t*KVD,KVD*4);memcpy(Vc+(size_t)(p0+t)*KVD,v+(size_t)t*KVD,KVD*4);}
    double _ta=now();
    int nth=att_threads(); if(nth>NH)nth=NH; if(nth>ATT_THREADS)nth=ATT_THREADS;
    pthread_t th[ATT_THREADS]; struct attw aw[ATT_THREADS];   /* pin each to a big (perf) core */
    for(int i=0;i<nth;i++) aw[i]=(struct attw){q,Kc,Vc,att,M,p0,grp,i*NH/nth,(i+1)*NH/nth, g_big[i%g_nbig]};
    for(int i=1;i<nth;i++) pthread_create(&th[i],NULL,attworker,&aw[i]);
    attworker(&aw[0]);
    for(int i=1;i<nth;i++) pthread_join(th[i],NULL);
    g_att+=now()-_ta;
    mm(ctx,L->Wo,QD,H,M,att,o);
    for(size_t i=0;i<(size_t)M*H;i++)x[i]+=o[i];
    for(int t=0;t<M;t++)rmsnorm(hn+(size_t)t*H,x+(size_t)t*H,L->n2,H);
    mm(ctx,L->Wg,H,FFN,M,hn,g);mm(ctx,L->Wu,H,FFN,M,hn,u);
    for(size_t i=0;i<(size_t)M*FFN;i++)a[i]=silu(g[i])*u[i];
    mm(ctx,L->Wd,FFN,H,M,a,d);
    for(size_t i=0;i<(size_t)M*H;i++)x[i]+=d[i];
    free(xn);free(q);free(k);free(v);free(att);free(o);free(hn);free(g);free(u);free(a);free(d);
}
int main(int argc,char**argv){
    int NL=argc>1?atoi(argv[1]):24, TDEC=argc>2?atoi(argv[2]):32, TPRE=argc>3?atoi(argv[3]):64;
    if(argc>4 && (argv[4][0]=='i'||argv[4][0]=='I')) g_i8=1;
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    int wsz=g_i8?1:2;
    double wmb=((double)NL*(2.0*H*QD+2.0*H*KVD+(double)H*FFN*3)+(double)H*VOCAB)*wsz/1e6;
    printf("bench: %s  H=%d NL=%d NH=%d NKV=%d HD=%d FFN=%d vocab=%d  dtype=%s (~%.0f MB weights)\n",
        ork_npu_soc(ctx),H,NL,NH,NKV,HD,FFN,VOCAB,g_i8?"int8":"fp16",wmb);
    printf("packing weights...\n"); fflush(stdout);
    Layer*Ls=malloc(NL*sizeof(Layer)); for(int l=0;l<NL;l++)mkl(ctx,&Ls[l]);
    ork_w*Wlm=mkw(ctx,H,VOCAB);
    float*Kc=calloc((size_t)NL*MAXSEQ*KVD,4),*Vc=calloc((size_t)NL*MAXSEQ*KVD,4);
    float*x=malloc((size_t)MAXSEQ*H*4),*fn=malloc(H*4),*logits=malloc((size_t)VOCAB*4); unsigned s=1;
    for(int i=0;i<H;i++){s=s*1103515245+12345;x[i]=(((int)((s>>16)%17))-8)*0.05f;}
    for(int warm=0;warm<2;warm++){ float t1[H];memcpy(t1,x,H*4);
        for(int l=0;l<NL;l++)layer(ctx,&Ls[l],1,0,t1,Kc+(size_t)l*MAXSEQ*KVD,Vc+(size_t)l*MAXSEQ*KVD); }
    memset(Kc,0,(size_t)NL*MAXSEQ*KVD*4);memset(Vc,0,(size_t)NL*MAXSEQ*KVD*4);
    double t0=now();
    for(int p=0;p<TDEC;p++){ float t1[H];memcpy(t1,x,H*4);
        for(int l=0;l<NL;l++)layer(ctx,&Ls[l],1,p,t1,Kc+(size_t)l*MAXSEQ*KVD,Vc+(size_t)l*MAXSEQ*KVD);
        rmsnorm(fn,t1,Ls[0].n1,H); mm(ctx,Wlm,H,VOCAB,1,fn,logits); }
    double dt=now()-t0;
    printf("DECODE : %d tok in %.2fs = %.2f tok/s  (%.1f ms/tok)\n",TDEC,dt,TDEC/dt,dt/TDEC*1e3);
    printf("         of which ork_mm_run (NPU+lib): %.1f ms/tok (%.0f%%);  rest (quant/ops/alloc on CPU): %.1f ms/tok (%.0f%%)\n",
        g_mm/TDEC*1e3, g_mm/dt*100, (dt-g_mm)/TDEC*1e3, (dt-g_mm)/dt*100);
    for(size_t i=0;i<(size_t)TPRE*H;i++){s=s*1103515245+12345;x[i]=(((int)((s>>16)%17))-8)*0.05f;}
    memset(Kc,0,(size_t)NL*MAXSEQ*KVD*4);memset(Vc,0,(size_t)NL*MAXSEQ*KVD*4);
    g_mm=0; g_att=0;                       /* measure prefill: NPU(matmul) vs attention vs other */
    t0=now();
    for(int l=0;l<NL;l++)layer(ctx,&Ls[l],TPRE,0,x,Kc+(size_t)l*MAXSEQ*KVD,Vc+(size_t)l*MAXSEQ*KVD);
    double pt=now()-t0;
    printf("PREFILL: %d tok in %.2fs = %.1f tok/s  (%d att-threads)\n",TPRE,pt,TPRE/pt,att_threads());
    printf("         NPU matmul: %.0f ms (%.0f%%);  attention: %.0f ms (%.0f%%);  other CPU (norm/rope/silu/malloc): %.0f ms (%.0f%%)\n",
        g_mm*1e3, g_mm/pt*100, g_att*1e3, g_att/pt*100, (pt-g_mm-g_att)*1e3, (pt-g_mm-g_att)/pt*100);
    ork_npu_free(ctx);
    return 0;
}
