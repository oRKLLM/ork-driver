/* tools/overlap_probe.c — measure the CEILING of within-backend CPU/NPU overlap.
 * The ggml-ork prefill is synchronous per matmul: quant(A) -> run_i8(NPU) -> dequant(C).
 * This probe times each phase, then runs the loop SERIAL vs PIPELINED (a background thread
 * quantizes the NEXT iter's A and dequantizes the PREV iter's C while the NPU runs the CURRENT
 * matmul). The speedup is the absolute ceiling of overlapping per-matmul CPU work with NPU compute
 * WITHOUT cross-node async (ggml-sched). If it's small, within-backend overlap isn't the lever.
 *   make overlap_probe && sudo ./overlap_probe [M] [K] [N] [iters]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

/* per-row int8 quant: A_f32[M*K] -> A_i8[M*K] (absmax/127). Mirrors ggml-ork's act-quant. */
static void quant(const float*Af,int8_t*Ai,int M,int K){
    for(int m=0;m<M;m++){ const float*r=Af+(size_t)m*K; int8_t*o=Ai+(size_t)m*K;
        float amax=0; for(int k=0;k<K;k++){ float a=fabsf(r[k]); if(a>amax)amax=a; }
        float s=amax>0?127.0f/amax:0; for(int k=0;k<K;k++){ int v=(int)lrintf(r[k]*s); if(v>127)v=127; if(v<-127)v=-127; o[k]=(int8_t)v; }
    }
}
/* dequant: C_i32[M*N] -> C_f32[M*N] (aScale[m]*bScale[n]). Mirrors ggml-ork's output dequant. */
static void dequant(const int32_t*Ci,float*Cf,const float*as,const float*bs,int M,int N){
    for(int m=0;m<M;m++){ const int32_t*r=Ci+(size_t)m*N; float*o=Cf+(size_t)m*N; float am=as[m];
        for(int n=0;n<N;n++) o[n]=r[n]*am*bs[n];
    }
}

struct bgargs { const float*Af; int8_t*Ai; const int32_t*Ci; float*Cf; const float*as,*bs; int M,K,N; int do_deq; };
static void*bg(void*vp){ struct bgargs*a=vp; quant(a->Af,a->Ai,a->M,a->K); if(a->do_deq) dequant(a->Ci,a->Cf,a->as,a->bs,a->M,a->N); return NULL; }

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):512, K=argc>2?atoi(argv[2]):3584, N=argc>3?atoi(argv[3]):18944;
    int iters=argc>4?atoi(argv[4]):30;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    ork_npu_set_core_budget(c, ork_npu_cores(c));
    int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    ork_w*w=ork_i8_mm_pack(c,K,N,B); if(!w){printf("pack failed\n");return 1;}
    float*Af=malloc((size_t)M*K*4); for(size_t i=0;i<(size_t)M*K;i++)Af[i]=((i*1103515245u)%255)/255.0f-0.5f;
    int8_t*Ai0=malloc((size_t)M*K),*Ai1=malloc((size_t)M*K);
    int32_t*Ci=malloc((size_t)M*N*4); float*Cf=malloc((size_t)M*N*4);
    float*as=malloc(M*4),*bs=malloc(N*4); for(int i=0;i<M;i++)as[i]=0.01f; for(int i=0;i<N;i++)bs[i]=0.01f;

    printf("overlap_probe int8 %dx%dx%d, %d iters, %d cores\n",M,K,N,iters,ork_npu_cores(c));
    /* phase timings (isolated) */
    quant(Af,Ai0,M,K); ork_i8_mm_run(c,w,M,Ai0,Ci);  /* warm */
    double t0=now_us(); for(int i=0;i<iters;i++) quant(Af,Ai0,M,K);  double tq=(now_us()-t0)/iters;
    t0=now_us(); for(int i=0;i<iters;i++) ork_i8_mm_run(c,w,M,Ai0,Ci); double tr=(now_us()-t0)/iters;
    t0=now_us(); for(int i=0;i<iters;i++) dequant(Ci,Cf,as,bs,M,N);   double td=(now_us()-t0)/iters;
    printf("  phases (us/iter):  quant=%.0f  run_i8(NPU)=%.0f  dequant=%.0f   quant+deq=%.0f (%.1f%% of run)\n",
           tq,tr,td,tq+td,100.0*(tq+td)/tr);

    /* SERIAL: quant -> run -> deq */
    t0=now_us(); for(int i=0;i<iters;i++){ quant(Af,Ai0,M,K); ork_i8_mm_run(c,w,M,Ai0,Ci); dequant(Ci,Cf,as,bs,M,N); }
    double tser=(now_us()-t0)/iters;
    /* PIPELINED: bg quants next A + deqs prev C while NPU runs current */
    quant(Af,Ai0,M,K);
    t0=now_us();
    for(int i=0;i<iters;i++){ int cur=i&1; int8_t*Acur=cur?Ai1:Ai0, *Anext=cur?Ai0:Ai1;
        struct bgargs a={Af,Anext,Ci,Cf,as,bs,M,K,N,i>0};
        pthread_t th; pthread_create(&th,NULL,bg,&a);
        ork_i8_mm_run(c,w,M,Acur,Ci);
        pthread_join(th,NULL);
    }
    double tpipe=(now_us()-t0)/iters;
    printf("  SERIAL    quant+run+deq = %.0f us/iter\n", tser);
    printf("  PIPELINED (overlap)     = %.0f us/iter   -> speedup %.3fx  (ceiling: serial/run = %.3fx)\n",
           tpipe, tser/tpipe, tser/tr);
    ork_w_free(w); ork_npu_free(c); return 0;
}
