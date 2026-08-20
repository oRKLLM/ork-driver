/* split_matmul_probe — split-tensor CPU+NPU co-working for a decode-scale matmul.
 * One M=1 int8 matmul C[N] = A[K]·B[K,N], split by N: NPU computes [0:Nn] (async), CPU computes [Nn:N]
 * (NEON sdot, big cores) CONCURRENTLY, then concatenate. Answers the load-bearing question: does concurrent
 * CPU+NPU DRAM access AGGREGATE bandwidth (split faster than either alone) or CONTEND (split ~= worse)?
 * All-ones data -> every output == K (bit-exact gate). Sweeps the NPU fraction to find the balance point.
 *   make split_matmul_probe && sudo env ORK_MM_TIMEOUT=4000 ./split_matmul_probe [iters]
 */
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

/* NEON sdot int8 GEMV: C[n] = Σ_k A[k]·Bt[n*K+k], Bt = weight transposed [N][K] (contiguous per output). */
static int32_t dot_i8(const int8_t*a,const int8_t*b,int K){
    int32x4_t acc=vdupq_n_s32(0); int k=0;
    for(;k+16<=K;k+=16){ acc=vdotq_s32(acc, vld1q_s8(a+k), vld1q_s8(b+k)); }
    int32_t s=vaddvq_s32(acc); for(;k<K;k++) s+=(int32_t)a[k]*b[k]; return s;
}
typedef struct { const int8_t*A,*Bt; int32_t*C; int K,n0,n1,core; } cjob;
static void* cworker(void*p){ cjob*j=p;
    if(j->core>=0){ cpu_set_t s; CPU_ZERO(&s); CPU_SET(j->core,&s); pthread_setaffinity_np(pthread_self(),sizeof s,&s); }
    for(int n=j->n0;n<j->n1;n++) j->C[n]=dot_i8(j->A, j->Bt+(size_t)n*j->K, j->K);
    return NULL; }
/* CPU GEMV over [n0,n1) with nt threads pinned to cores [c0..c0+nt-1] */
static void cpu_gemv(const int8_t*A,const int8_t*Bt,int32_t*C,int K,int n0,int n1,int nt,int c0){
    pthread_t th[8]; cjob jb[8]; int span=n1-n0, per=(span+nt-1)/nt;
    for(int t=0;t<nt;t++){ int a=n0+t*per, b=a+per; if(b>n1)b=n1;
        jb[t]=(cjob){A,Bt,C,K,a,b,c0+t}; th[t]=0; if(a<b) pthread_create(&th[t],0,cworker,&jb[t]); }
    for(int t=0;t<nt;t++) if(th[t]) pthread_join(th[t],0);
}

int main(int argc,char**argv){
    int iters=argc>1?atoi(argv[1]):30;
    int K=3584, N=18944;                                   /* gate/up decode projection: big + memory-bound */
    int nt=4; { const char*e=getenv("SPLIT_CPU_THREADS"); if(e)nt=atoi(e); }
    int c0=4; { const char*e=getenv("SPLIT_CPU_CORE0"); if(e)c0=atoi(e); }   /* big cores 4-7 on RK3588 */
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    printf("split_matmul_probe: M=1 K=%d N=%d  weight=%.1f MB  CPU=%d threads @cpu%d+\n",
           K,N,(double)K*N/1e6,nt,c0);

    int8_t*A=malloc(K); memset(A,1,K);
    int8_t*Bpk=malloc((size_t)K*N); memset(Bpk,1,(size_t)K*N);     /* for ork_i8_mm_pack (K*N row-major) */
    int8_t*Bt=malloc((size_t)N*K); memset(Bt,1,(size_t)N*K);       /* transposed [N][K] for the CPU GEMV */
    int32_t*C=malloc((size_t)N*4);

    ork_npu_set_pack_domain(c,0);
    ork_w*wfull=ork_i8_mm_pack(c,K,N,Bpk); if(!wfull){printf("pack full failed\n");return 1;}

    /* ---- NPU-alone (full N) ---- */
    for(int i=0;i<3;i++) ork_i8_mm_run(c,wfull,1,A,C);
    double t0=now_us(); for(int i=0;i<iters;i++) ork_i8_mm_run(c,wfull,1,A,C); double t_npu=(now_us()-t0)/iters;
    int okn=1; for(int n=0;n<N;n++) if(C[n]!=K){okn=0;break;}
    printf("  NPU-alone : %8.1f us  %6.1f GB/s  %s\n", t_npu, (double)K*N/t_npu/1e3, okn?"ok":"BAD");

    /* ---- CPU-alone (full N, NEON sdot) ---- */
    memset(C,0,(size_t)N*4);
    cpu_gemv(A,Bt,C,K,0,N,nt,c0);                                  /* warm */
    t0=now_us(); for(int i=0;i<iters;i++) cpu_gemv(A,Bt,C,K,0,N,nt,c0); double t_cpu=(now_us()-t0)/iters;
    int okc=1; for(int n=0;n<N;n++) if(C[n]!=K){okc=0;break;}
    printf("  CPU-alone : %8.1f us  %6.1f GB/s  %s\n", t_cpu, (double)K*N/t_cpu/1e3, okc?"ok":"BAD");

    /* ---- SPLIT: NPU [0:Nn] async  ‖  CPU [Nn:N]  ---- */
    printf("\n  NPU-frac |  Nn   | split us | agg GB/s | vs best-single | ok\n");
    double best_single = t_npu<t_cpu?t_npu:t_cpu;
    double fr[]={0.2,0.3,0.4,0.5,0.6};
    for(unsigned fi=0; fi<sizeof(fr)/sizeof(double); fi++){
        int Nn=((int)(fr[fi]*N))/16*16; if(Nn<16)Nn=16; if(Nn>=N)Nn=N-16;   /* NPU N-tile %16 */
        ork_w*wnpu=ork_i8_mm_pack(c,K,Nn,Bpk); if(!wnpu){printf("  pack Nn=%d failed\n",Nn);continue;}
        int32_t*Cn=malloc((size_t)Nn*4);
        ork_i8_mm_run(c,wnpu,1,A,Cn);                              /* warm */
        double s0=now_us();
        for(int i=0;i<iters;i++){
            ork_async*h=ork_i8_mm_run_async(c,wnpu,1,A,Cn);        /* NPU half async */
            cpu_gemv(A,Bt,C,K,Nn,N,nt,c0);                         /* CPU half concurrent (big cores) */
            ork_async_wait(h);
        }
        double t_split=(now_us()-s0)/iters;
        int ok=1; for(int n=0;n<Nn;n++) if(Cn[n]!=K){ok=0;break;} for(int n=Nn;n<N&&ok;n++) if(C[n]!=K){ok=0;break;}
        printf("  %6.2f   | %5d | %8.1f | %7.1f  | %6.2fx        | %s\n",
               fr[fi], Nn, t_split, (double)K*N/t_split/1e3, best_single/t_split, ok?"ok":"BAD");
        ork_mm_free(c,wnpu); free(Cn);
    }
    printf("\n  (>1.0x vs best-single => CPU+NPU DRAM access AGGREGATES; ~<=1.0x => contends/core-bound)\n");
    ork_npu_free(c); return 0;
}
