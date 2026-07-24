/* spine_worker_probe — increment 2 engineering: the CPU worker UNIT of the heterogeneous doorbell spine.
 * A pre-warmed, big-core-pinned worker thread parked on a condvar, with a uniform async interface:
 *     cpu_dispatch(u, fn, arg) -> generation "handle";  cpu_poll(u, gen) -> done?  cpu_wait(u, gen)
 * mirroring the doorbell's begin/progress/end. Validates that a CPU glue op dispatched to the worker overlaps
 * the async doorbell NPU op "for free" (the domain-free-CPU parallelism axis), and stays coherent.
 * This is the reusable CPU execution unit the spine scheduler will drive alongside the doorbell NPU unit.
 *   make spine_worker_probe && sudo env ORK_MM_TIMEOUT=3000 ./spine_worker_probe
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
static uint32_t rng=0x51edu; static int s3(void){ rng=rng*1664525u+1013904223u; return (int)((rng>>28)%3)-1; }
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

/* ---- the CPU worker unit ---- */
struct cpu_unit { pthread_t th; pthread_mutex_t mu; pthread_cond_t go, dn; int gen, done_gen, stop; long(*fn)(void*); void*arg; long ret; };
static void* cpu_loop(void*p){ struct cpu_unit*u=p;
    pthread_mutex_lock(&u->mu);
    for(;;){ while(u->gen==u->done_gen && !u->stop) pthread_cond_wait(&u->go,&u->mu);
        if(u->stop){ pthread_mutex_unlock(&u->mu); return NULL; }
        long(*fn)(void*)=u->fn; void*arg=u->arg; int g=u->gen;
        pthread_mutex_unlock(&u->mu);
        long r=fn(arg);                                  /* run the glue op OUTSIDE the lock (concurrent with NPU) */
        pthread_mutex_lock(&u->mu); u->ret=r; u->done_gen=g; pthread_cond_signal(&u->dn); }
}
static void cpu_unit_start(struct cpu_unit*u, int core){ memset(u,0,sizeof *u);
    pthread_mutex_init(&u->mu,0); pthread_cond_init(&u->go,0); pthread_cond_init(&u->dn,0);
    pthread_create(&u->th,0,cpu_loop,u);
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core,&cs); pthread_setaffinity_np(u->th,sizeof cs,&cs);   /* pin to a big core */
}
static int  cpu_dispatch(struct cpu_unit*u, long(*fn)(void*), void*arg){ pthread_mutex_lock(&u->mu);
    u->fn=fn; u->arg=arg; int g=++u->gen; pthread_cond_signal(&u->go); pthread_mutex_unlock(&u->mu); return g; }
static int  cpu_poll(struct cpu_unit*u, int gen){ pthread_mutex_lock(&u->mu); int d=u->done_gen>=gen; pthread_mutex_unlock(&u->mu); return d; }
static long cpu_wait(struct cpu_unit*u, int gen){ pthread_mutex_lock(&u->mu);
    while(u->done_gen<gen) pthread_cond_wait(&u->dn,&u->mu); long r=u->ret; pthread_mutex_unlock(&u->mu); return r; }
static void cpu_unit_stop(struct cpu_unit*u){ pthread_mutex_lock(&u->mu); u->stop=1; pthread_cond_signal(&u->go); pthread_mutex_unlock(&u->mu); pthread_join(u->th,0); }

/* the domain-free glue kernel (a stand-in norm/requant), on host scratch */
struct glue_arg { const int8_t*x; size_t n; long out; };
static long glue_fn(void*p){ struct glue_arg*a=p; long acc=0; for(size_t i=0;i<a->n;i++){ long q=(long)a->x[i]>>1; if(q>63)q=63; if(q<-63)q=-63; acc+=q; } a->out=acc; return acc; }

int main(void){
    int M=1, K=512, N=512, S=8;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    int cap=ork_dyn_max_steps(); if(S>cap)S=cap;
    printf("spine_worker_probe: CPU worker unit ∥ doorbell NPU unit (S=%d M=%d K=%d N=%d)\n",S,M,K,N);

    int8_t *A=malloc((size_t)M*K); int32_t *C=ork_dma_alloc(c,(size_t)S*N*4);
    if(!A||!C){ printf("alloc FAILED\n"); return 2; }
    int8_t *Wb=malloc((size_t)K*N);
    for(int i=0;i<M*K;i++) A[i]=(int8_t)s3();
    for(size_t i=0;i<(size_t)K*N;i++) Wb[i]=(int8_t)s3();
    ork_w *W=ork_mm_pack_i8(c,K,N,Wb); if(!W){ printf("pack failed\n"); return 2; }
    ork_mm_task_i8 *tk=malloc((size_t)S*sizeof *tk);
    for(int s=0;s<S;s++) tk[s]=(ork_mm_task_i8){W,M,A,C+(size_t)s*N};
    int32_t *Cr=malloc((size_t)N*4);
    for(int n=0;n<N;n++){ long a=0; for(int k=0;k<K;k++) a+=A[k]*Wb[(size_t)k*N+n]; Cr[n]=(int32_t)a; }

    size_t SN=(size_t)K*N; int8_t*scr=malloc(SN); for(size_t i=0;i<SN;i++) scr[i]=(int8_t)(i*7);
    struct cpu_unit U; cpu_unit_start(&U, 6);                 /* pin to big core 6 (rk3588 A76 = cpu4..7) */
    struct glue_arg ga={scr,SN,0};
    /* warm both units */
    { ork_dyn_chain*h=ork_dyn_begin(c,S,tk); if(h) ork_dyn_end(h); }
    cpu_wait(&U, cpu_dispatch(&U,glue_fn,&ga));

    /* HETEROGENEOUS OVERLAP: fire the CPU glue on the worker, run the doorbell NPU on main, then join */
    double na=now_us(); for(int r=0;r<10;r++){ ork_dyn_chain*h=ork_dyn_begin(c,S,tk); ork_dyn_end(h); } double t_npu=(now_us()-na)/10;
    double ca=now_us(); for(int r=0;r<10;r++){ int g=cpu_dispatch(&U,glue_fn,&ga); cpu_wait(&U,g); } double t_cpu=(now_us()-ca)/10;
    double oa=now_us();
    for(int r=0;r<10;r++){ int g=cpu_dispatch(&U,glue_fn,&ga);   /* CPU worker starts */
        ork_dyn_chain*h=ork_dyn_begin(c,S,tk); ork_dyn_end(h);    /* NPU on main, concurrent with the worker */
        cpu_wait(&U,g); }                                        /* join the worker */
    double t_over=(now_us()-oa)/10;

    long bad=0; int32_t mx=0;
    for(int s=0;s<S;s++) for(int n=0;n<N;n++){ int32_t d=C[(size_t)s*N+n]-Cr[n]; if(d){bad++; if(d<0)d=-d; if(d>mx)mx=d;} }
    double eff=t_over>0?(t_npu+t_cpu)/t_over:0, sm=t_cpu<t_npu?t_cpu:t_npu;
    double hidden=(t_npu+t_cpu)>t_over?100.0*(t_npu+t_cpu-t_over)/sm:0.0;
    printf("  NPU coherent: %s (bad=%ld/%d)\n", bad?"MISMATCH":"bit-exact", bad, S*N);
    printf("  worker-unit overlap: NPU %.0fus | CPU %.0fus | overlapped %.0fus -> %.2fx (%.0f%% hidden)\n",
           t_npu, t_cpu, t_over, eff, hidden>100?100:hidden);
    printf("%s\n", (bad==0 && eff>1.3) ? "PASS — CPU worker unit overlaps the doorbell NPU unit; coherent"
                 : (bad==0 ? "COHERENT but overlap weak" : "FAIL — incoherent"));
    cpu_unit_stop(&U);
    free(A); ork_dma_free(c,C); ork_mm_free(c,W); ork_npu_free(c);
    return bad?1:0;
}
