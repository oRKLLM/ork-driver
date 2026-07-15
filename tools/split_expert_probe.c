/* split_expert_probe — MoE expert-split CPU+NPU co-work (the right granularity for MoE decode).
 * E independent expert matmuls (M=1, int8). NPU computes experts [0:k] (single-core dispatch, worker pinned to
 * a LITTLE core so big cores stay free), CPU computes [k:E] (NEON sdot, big cores), CONCURRENTLY. The experts
 * are independent, so unlike within-matmul tensor-split (each matmul ~saturates DRAM) this fills the between-
 * expert gaps that leave MoE decode at ~19% BW. Q: does routing some experts to the idle NPU beat all-CPU /
 * all-NPU (aggregate)? Sweeps k. All-ones -> out==K (bit-exact gate).
 *   make split_expert_probe && sudo env ORK_MM_TIMEOUT=4000 ./split_expert_probe [iters]
 * Knobs: ORK_SE_E ORK_SE_K ORK_SE_N | ORK_SE_NPU_CORES (NPU core budget, def 1) ORK_SE_NPU_CORE (worker core,
 * def 0=little) | ORK_SE_CPU_THREADS (def 4) ORK_SE_CPU_CORE0 (def 4=big). */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <arm_neon.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static int32_t dot_i8(const int8_t*a,const int8_t*b,int K){
    int32x4_t acc=vdupq_n_s32(0); int k=0;
    for(;k+16<=K;k+=16) acc=vdotq_s32(acc, vld1q_s8(a+k), vld1q_s8(b+k));
    int32_t s=vaddvq_s32(acc); for(;k<K;k++) s+=(int32_t)a[k]*b[k]; return s;
}
static int gK,gN;
static ork_w **gW; static int8_t **gBt, *gA; static int32_t **gC;

/* CPU: experts [lo,hi), thread tid strides by nt, pinned to c0+tid */
typedef struct { int lo,hi,nt,tid,core; } cej;
static void* cpu_ew(void*p){ cej*j=p;
    if(j->core>=0){ cpu_set_t s; CPU_ZERO(&s); CPU_SET(j->core,&s); pthread_setaffinity_np(pthread_self(),sizeof s,&s); }
    for(int e=j->lo+j->tid; e<j->hi; e+=j->nt){ const int8_t*Bt=gBt[e]; int32_t*C=gC[e];
        for(int n=0;n<gN;n++) C[n]=dot_i8(gA, Bt+(size_t)n*gK, gK); }
    return NULL; }
static void cpu_experts(int lo,int hi,int nt,int c0){ if(lo>=hi)return;
    pthread_t th[8]; cej jb[8]; if(nt>8)nt=8;
    for(int t=0;t<nt;t++){ jb[t]=(cej){lo,hi,nt,t,c0+t}; pthread_create(&th[t],0,cpu_ew,&jb[t]); }
    for(int t=0;t<nt;t++) pthread_join(th[t],0); }

/* NPU persistent worker (pinned to a chosen core) — does experts [lo,hi) serially via ork_mm_run_i8 */
typedef struct { ork_npu*c; int lo,hi,core; pthread_mutex_t mu; pthread_cond_t go,dn; int has,done,stop; } npw;
static void* npu_w(void*p){ npw*w=p;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(w->core,&s); pthread_setaffinity_np(pthread_self(),sizeof s,&s);
    for(;;){ pthread_mutex_lock(&w->mu); while(!w->has&&!w->stop) pthread_cond_wait(&w->go,&w->mu);
        if(w->stop){ pthread_mutex_unlock(&w->mu); return NULL; } w->has=0; int lo=w->lo,hi=w->hi; pthread_mutex_unlock(&w->mu);
        for(int e=lo;e<hi;e++) ork_mm_run_i8(w->c,gW[e],1,gA,gC[e]);
        pthread_mutex_lock(&w->mu); w->done=1; pthread_cond_signal(&w->dn); pthread_mutex_unlock(&w->mu); } }
static void np_submit(npw*w,int lo,int hi){ pthread_mutex_lock(&w->mu); w->lo=lo; w->hi=hi; w->done=0; w->has=1; pthread_cond_signal(&w->go); pthread_mutex_unlock(&w->mu); }
static void np_wait(npw*w){ pthread_mutex_lock(&w->mu); while(!w->done) pthread_cond_wait(&w->dn,&w->mu); pthread_mutex_unlock(&w->mu); }

static int env_i(const char*k,int d){ const char*e=getenv(k); return e?atoi(e):d; }

int main(int argc,char**argv){
    int iters=argc>1?atoi(argv[1]):40;
    int E=env_i("ORK_SE_E",8), K=env_i("ORK_SE_K",2048), N=env_i("ORK_SE_N",1536);
    int npu_cores=env_i("ORK_SE_NPU_CORES",1), npu_core=env_i("ORK_SE_NPU_CORE",0);
    int nt=env_i("ORK_SE_CPU_THREADS",4), c0=env_i("ORK_SE_CPU_CORE0",4);
    gK=K; gN=N;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    ork_npu_set_core_budget(c,npu_cores);
    printf("split_expert_probe: E=%d experts K=%d N=%d (%.1f MB each)  NPU=%d-core @cpu%d  CPU=%d thr @cpu%d+\n",
           E,K,N,(double)K*N/1e6,npu_cores,npu_core,nt,c0);

    gW=malloc(E*sizeof*gW); gBt=malloc(E*sizeof*gBt); gC=malloc(E*sizeof*gC);
    gA=malloc(K); memset(gA,1,K);
    int8_t*Bpk=malloc((size_t)K*N); memset(Bpk,1,(size_t)K*N);
    for(int e=0;e<E;e++){ ork_npu_set_pack_domain(c,0); gW[e]=ork_mm_pack_i8(c,K,N,Bpk);
        if(!gW[e]){printf("pack e%d failed\n",e);return 1;}
        gBt[e]=malloc((size_t)N*K); memset(gBt[e],1,(size_t)N*K); gC[e]=malloc((size_t)N*4); }
    free(Bpk);

    npw w; memset(&w,0,sizeof w); w.c=c; w.core=npu_core;
    pthread_mutex_init(&w.mu,0); pthread_cond_init(&w.go,0); pthread_cond_init(&w.dn,0);
    pthread_t wth; pthread_create(&wth,0,npu_w,&w);

    np_submit(&w,0,E); np_wait(&w); cpu_experts(0,E,nt,c0);              /* warm both */

    double t0=now_us(); for(int i=0;i<iters;i++){ np_submit(&w,0,E); np_wait(&w); } double t_npu=(now_us()-t0)/iters;
    int okn=1; for(int e=0;e<E;e++)for(int n=0;n<N;n++) if(gC[e][n]!=K){okn=0;break;}
    t0=now_us(); for(int i=0;i<iters;i++) cpu_experts(0,E,nt,c0); double t_cpu=(now_us()-t0)/iters;
    int okc=1; for(int e=0;e<E;e++)for(int n=0;n<N;n++) if(gC[e][n]!=K){okc=0;break;}
    printf("  ALL-NPU (%d experts serial): %8.1f us  %s\n", E, t_npu, okn?"ok":"BAD");
    printf("  ALL-CPU (%d experts,%d thr): %8.1f us  %s\n", E, nt, t_cpu, okc?"ok":"BAD");
    double best=t_npu<t_cpu?t_npu:t_cpu;

    printf("\n  k(NPU) | E-k(CPU) | split us | vs best-single | ok\n");
    for(int k=1;k<E;k++){
        t0=now_us();
        for(int i=0;i<iters;i++){ np_submit(&w,0,k); cpu_experts(k,E,nt,c0); np_wait(&w); }
        double t_split=(now_us()-t0)/iters;
        int ok=1; for(int e=0;e<E;e++)for(int n=0;n<N&&ok;n++) if(gC[e][n]!=K)ok=0;
        printf("  %5d  | %6d   | %8.1f | %6.2fx        | %s\n", k, E-k, t_split, best/t_split, ok?"ok":"BAD");
    }
    printf("\n  (>1.0x => routing experts across NPU+CPU AGGREGATES; the MoE idle-NPU headroom is exploitable)\n");
    pthread_mutex_lock(&w.mu); w.stop=1; pthread_cond_signal(&w.go); pthread_mutex_unlock(&w.mu); pthread_join(wth,0);
    ork_npu_free(c); return 0;
}
