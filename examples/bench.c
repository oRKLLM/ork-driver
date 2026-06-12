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

/* Persistent thread pool pinned to the perf cores. Per-op pthread_create/join drowns in spawn
 * overhead (the regression we saw), so spawn workers ONCE and dispatch parallel-for jobs to them.
 * Used for every parallelizable CPU op (attention, quant/dequant, silu, rmsnorm). */
typedef void (*pf_fn)(int lo,int hi,void *ctx);
static pthread_t P_th[16]; static int P_nth=0;
static pthread_mutex_t P_mu=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t P_go=PTHREAD_COND_INITIALIZER, P_dn=PTHREAD_COND_INITIALIZER;
static pf_fn P_fn; static void *P_ctx; static int P_total, P_gen=0, P_done=0, P_stop=0;
static void p_chunk(int id,int nth,int total,int*lo,int*hi){*lo=(int)((long)id*total/nth);*hi=(int)((long)(id+1)*total/nth);}
static void *p_worker(void *arg){
    int id=(int)(long)arg; cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(g_big[id%g_nbig],&cs);sched_setaffinity(0,sizeof cs,&cs);
    int mygen=0;
    for(;;){
        pthread_mutex_lock(&P_mu);
        while(P_gen==mygen && !P_stop) pthread_cond_wait(&P_go,&P_mu);
        if(P_stop){pthread_mutex_unlock(&P_mu);return NULL;}
        mygen=P_gen; pf_fn fn=P_fn; void*ctx=P_ctx; int total=P_total,nth=P_nth;
        pthread_mutex_unlock(&P_mu);
        int lo,hi; p_chunk(id,nth,total,&lo,&hi); if(hi>lo) fn(lo,hi,ctx);
        pthread_mutex_lock(&P_mu); if(++P_done==nth-1) pthread_cond_signal(&P_dn); pthread_mutex_unlock(&P_mu);
    }
}
static void pool_init(void){ att_init(); P_nth=att_threads(); if(P_nth>16)P_nth=16; if(P_nth<1)P_nth=1;
    for(int i=1;i<P_nth;i++) pthread_create(&P_th[i],NULL,p_worker,(void*)(long)i); }
/* run fn over [0,total) split across the pool; main runs chunk 0, workers the rest, then barrier */
static void parallel_for(int total,pf_fn fn,void*ctx){
    if(P_nth<=1||total<=0){ if(total>0)fn(0,total,ctx); return; }
    pthread_mutex_lock(&P_mu); P_fn=fn;P_ctx=ctx;P_total=total;P_done=0;P_gen++; pthread_cond_broadcast(&P_go); pthread_mutex_unlock(&P_mu);
    int lo,hi; p_chunk(0,P_nth,total,&lo,&hi); if(hi>lo) fn(lo,hi,ctx);
    pthread_mutex_lock(&P_mu); while(P_done<P_nth-1) pthread_cond_wait(&P_dn,&P_mu); pthread_mutex_unlock(&P_mu);
}
/* pool only for prefill-sized work (big=M>=8). Decode (M=1) runs inline — 450 tiny pool barriers
 * per token would regress decode. */
static void pfor(int big,int total,pf_fn fn,void*ctx){ if(big&&P_nth>1)parallel_for(total,fn,ctx); else if(total>0)fn(0,total,ctx); }
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
static void rmsnorm(float*o,const float*x,const float*w,int n){float ss=0;for(int i=0;i<n;i++)ss+=x[i]*x[i];float s=1.0f/sqrtf(ss/n+EPS);for(int i=0;i<n;i++)o[i]=x[i]*s*w[i];}
static void rope(float*x,int seq,int nh,int hd,int p0){for(int t=0;t<seq;t++)for(int h=0;h<nh;h++){float*v=x+((size_t)t*nh+h)*hd;for(int i=0;i<hd/2;i++){float fr=powf(10000.0f,-2.0f*i/hd),an=(p0+t)*fr,c=cosf(an),s=sinf(an);float a=v[i],b=v[i+hd/2];v[i]=a*c-b*s;v[i+hd/2]=a*s+b*c;}}}
static float silu(float x){return x/(1.0f+expf(-x));}

typedef struct { ork_w*Wq,*Wk,*Wv,*Wo,*Wg,*Wu,*Wd; float n1[H],n2[H]; } Layer;
static ork_w* mkw(ork_npu*ctx,int K,int N){
    if(g_i8){ int8_t*B=malloc((size_t)K*N);unsigned s=K*7+N*13+1;for(size_t i=0;i<(size_t)K*N;i++){s=s*1103515245+12345;B[i]=(int8_t)(((int)((s>>16)%7))-3);}ork_w*w=ork_mm_pack_i8(ctx,K,N,B);free(B);return w; }
    f16*B=malloc((size_t)K*N*2);unsigned s=K*7+N*13+1;float sc=0.4f/sqrtf((float)K);
    for(size_t i=0;i<(size_t)K*N;i++){s=s*1103515245+12345;B[i]=(f16)((((int)((s>>16)%9))-4)*sc);} ork_w*w=ork_mm_pack(ctx,K,N,B);free(B);return w;}
static void mkl(ork_npu*ctx,Layer*L){for(int i=0;i<H;i++){L->n1[i]=1.0f;L->n2[i]=1.0f;}
    L->Wq=mkw(ctx,H,QD);L->Wk=mkw(ctx,H,KVD);L->Wv=mkw(ctx,H,KVD);L->Wo=mkw(ctx,QD,H);L->Wg=mkw(ctx,H,FFN);L->Wu=mkw(ctx,H,FFN);L->Wd=mkw(ctx,FFN,H);}
/* pooled CPU ops (#1: cut the prefill "other-CPU" — quant/dequant/silu — across the perf cores) */
struct siluctx{const float*g,*u;float*a;};
static void silu_pf(int lo,int hi,void*vp){struct siluctx*c=vp;for(int i=lo;i<hi;i++)c->a[i]=silu(c->g[i])*c->u[i];}
struct qctx{const float*src;int8_t*dst;};
static void quant_pf(int lo,int hi,void*vp){struct qctx*c=vp;for(int i=lo;i<hi;i++){int q=(int)lrintf(c->src[i]*8.0f);c->dst[i]=(int8_t)(q<-127?-127:q>127?127:q);}}
struct dqctx{const int32_t*src;float*dst;};
static void dequant_pf(int lo,int hi,void*vp){struct dqctx*c=vp;for(int i=lo;i<hi;i++)c->dst[i]=c->src[i]*(1.0f/8.0f);}
/* C[M,N] = A[M,K] x packed weights. fp16: cast A->fp16. int8: quantize A->int8, run, dequant
 * (dummy per-tensor scale — bench measures matmul throughput, not numerics). The fp32<->int8
 * round-trip is the bench's; a real engine keeps activations int8 on-device. Pooled here. */
static void mm(ork_npu*ctx,ork_w*w,int K,int N,int M,const float*Af,float*C){
    if(g_i8){ int8_t*A=malloc((size_t)M*K);int32_t*Ci=malloc((size_t)M*N*4);
        struct qctx qc={Af,A}; pfor(M>=8,(int)((size_t)M*K),quant_pf,&qc);
        double t=now(); ork_mm_run_i8(ctx,w,M,A,Ci); g_mm+=now()-t;
        struct dqctx dc={Ci,C}; pfor(M>=8,(int)((size_t)M*N),dequant_pf,&dc);
        free(A);free(Ci); return; }
    f16*A=malloc((size_t)M*K*2);for(size_t i=0;i<(size_t)M*K;i++)A[i]=(f16)Af[i];
    double t=now(); ork_mm_run(ctx,w,M,A,C); g_mm+=now()-t; free(A);
}
/* Attention is the prefill bottleneck (O(M^2 * heads * HD), ~65-72% of prefill CPU). Flash-style:
 * online softmax in a SINGLE pass over keys, O(HD) running state (no sc[L2] buffer, one read of
 * K/V each). Parallel over heads via the pool (independent; each head writes its own att cols).
 * Causal (j<=t). */
struct attctx { const float *q,*Kc,*Vc; float *att; int M,p0,grp; };
static void att_pf(int lo,int hi,void *vp){
    struct attctx *c=vp; float scale=1.0f/sqrtf((float)HD);
    for(int hh=lo;hh<hi;hh++){int kvh=hh/c->grp;
        for(int t=0;t<c->M;t++){int L2=c->p0+t+1; const float*qp=c->q+((size_t)t*NH+hh)*HD;
            float m=-1e30f,l=0,acc[HD]; for(int e=0;e<HD;e++)acc[e]=0;
            for(int j=0;j<L2;j++){const float*kp=c->Kc+(size_t)j*KVD+kvh*HD; float s=0;for(int e=0;e<HD;e++)s+=qp[e]*kp[e]; s*=scale;
                float nm=s>m?s:m, cc=expf(m-nm), p=expf(s-nm); l=l*cc+p;
                const float*vp2=c->Vc+(size_t)j*KVD+kvh*HD; for(int e=0;e<HD;e++)acc[e]=acc[e]*cc+p*vp2[e]; m=nm; }
            float inv=l>0?1.0f/l:0,*ap=c->att+((size_t)t*NH+hh)*HD; for(int e=0;e<HD;e++)ap[e]=acc[e]*inv; }}
}
static void layer(ork_npu*ctx,Layer*L,int M,int p0,float*x,float*Kc,float*Vc){
    int grp=NH/NKV; float scale=1.0f/sqrtf((float)HD); (void)scale;
    float*xn=malloc((size_t)M*H*4),*q=malloc((size_t)M*QD*4),*k=malloc((size_t)M*KVD*4),*v=malloc((size_t)M*KVD*4),*att=malloc((size_t)M*QD*4),*o=malloc((size_t)M*H*4),*hn=malloc((size_t)M*H*4),*g=malloc((size_t)M*FFN*4),*u=malloc((size_t)M*FFN*4),*a=malloc((size_t)M*FFN*4),*d=malloc((size_t)M*H*4);
    for(int t=0;t<M;t++)rmsnorm(xn+(size_t)t*H,x+(size_t)t*H,L->n1,H);
    mm(ctx,L->Wq,H,QD,M,xn,q);mm(ctx,L->Wk,H,KVD,M,xn,k);mm(ctx,L->Wv,H,KVD,M,xn,v);
    rope(q,M,NH,HD,p0);rope(k,M,NKV,HD,p0);
    for(int t=0;t<M;t++){memcpy(Kc+(size_t)(p0+t)*KVD,k+(size_t)t*KVD,KVD*4);memcpy(Vc+(size_t)(p0+t)*KVD,v+(size_t)t*KVD,KVD*4);}
    double _ta=now();
    struct attctx ac={q,Kc,Vc,att,M,p0,grp}; pfor(M>=8,NH,att_pf,&ac);
    g_att+=now()-_ta;
    mm(ctx,L->Wo,QD,H,M,att,o);
    for(size_t i=0;i<(size_t)M*H;i++)x[i]+=o[i];
    for(int t=0;t<M;t++)rmsnorm(hn+(size_t)t*H,x+(size_t)t*H,L->n2,H);
    mm(ctx,L->Wg,H,FFN,M,hn,g);mm(ctx,L->Wu,H,FFN,M,hn,u);
    struct siluctx sc2={g,u,a}; pfor(M>=8,(int)((size_t)M*FFN),silu_pf,&sc2);
    mm(ctx,L->Wd,FFN,H,M,a,d);
    for(size_t i=0;i<(size_t)M*H;i++)x[i]+=d[i];
    free(xn);free(q);free(k);free(v);free(att);free(o);free(hn);free(g);free(u);free(a);free(d);
}
int main(int argc,char**argv){
    int NL=argc>1?atoi(argv[1]):24, TDEC=argc>2?atoi(argv[2]):32, TPRE=argc>3?atoi(argv[3]):64;
    if(argc>4 && (argv[4][0]=='i'||argv[4][0]=='I')) g_i8=1;
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    pool_init();                           /* persistent CPU pool on the perf cores */
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
