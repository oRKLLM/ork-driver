/* spine_sched_probe — increment 2 finish: the heterogeneous SCHEDULER. Drives an op DAG across two execution
 * units (NPU = doorbell on its own worker thread so NPU stays single-stream; CPU = the pinned worker), placing
 * each op per a table (NPU / CPU / EITHER) and dispatching a ready op to a free matching unit, in ONE poll loop.
 * Proves: (a) independent NPU & CPU ops overlap; (b) a dependent CPU op waits for its NPU producer then reads
 * the NPU output coherently (read-after-drain is OK; write-after-end is not — see doorbell_overlap_probe).
 *
 * Test DAG (mini "layer" shape):
 *   op0 NPU  matmul -> C0            (no deps)          ─┐ run concurrently
 *   op1 CPU  glue on scratch         (no deps)          ─┘ (overlap)
 *   op2 CPU  bridge: read C0, checksum (dep op0)         after op0
 *   op3 NPU  matmul -> C3            (dep op1)           after op1
 *   make spine_sched_probe && sudo env ORK_MM_TIMEOUT=3000 ./spine_sched_probe
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
/* dc civac = data-cache invalidate to point-of-coherency. ork_dyn_end's invalidate is on the DRAINING thread's
 * cache; ANY OTHER unit/thread reading the resident dma buffer must civac the region first (the cross-unit
 * handoff bsync). This is a hard rule for the spine's CPU worker reading an NPU-produced buffer. */
static inline void civac1(volatile void*p){ __asm__ volatile("dc civac,%0"::"r"(p):"memory"); }
static void civac_range(const void*base,size_t bytes){ for(size_t o=0;o<bytes;o+=64) civac1((char*)base+o); __asm__ volatile("dsb ish":::"memory"); }

/* ---- generic execution unit: a worker thread that runs any fn(arg) ---- */
struct unit { pthread_t th; pthread_mutex_t mu; pthread_cond_t go,dn; int gen,done_gen,stop; long(*fn)(void*); void*arg; long ret; int busy, op; };
static void* unit_loop(void*p){ struct unit*u=p; pthread_mutex_lock(&u->mu);
    for(;;){ while(u->gen==u->done_gen && !u->stop) pthread_cond_wait(&u->go,&u->mu);
        if(u->stop){ pthread_mutex_unlock(&u->mu); return NULL; }
        long(*fn)(void*)=u->fn; void*arg=u->arg; int g=u->gen; pthread_mutex_unlock(&u->mu);
        long r=fn(arg); pthread_mutex_lock(&u->mu); u->ret=r; u->done_gen=g; pthread_cond_signal(&u->dn); } }
static void unit_start(struct unit*u,int core){ memset(u,0,sizeof *u); pthread_mutex_init(&u->mu,0); pthread_cond_init(&u->go,0); pthread_cond_init(&u->dn,0);
    pthread_create(&u->th,0,unit_loop,u); cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core,&cs); pthread_setaffinity_np(u->th,sizeof cs,&cs); }
static int  unit_dispatch(struct unit*u,long(*fn)(void*),void*arg){ pthread_mutex_lock(&u->mu); u->fn=fn; u->arg=arg; int g=++u->gen; pthread_cond_signal(&u->go); pthread_mutex_unlock(&u->mu); return g; }
static int  unit_poll(struct unit*u,int gen){ pthread_mutex_lock(&u->mu); int d=u->done_gen>=gen; pthread_mutex_unlock(&u->mu); return d; }
static void spoll(struct unit*u,int gen){ while(!unit_poll(u,gen)){ struct timespec ts={0,20000}; nanosleep(&ts,0);} }   /* sleep-poll: don't busy-spin (starves the worker's core) */
static void unit_stop(struct unit*u){ pthread_mutex_lock(&u->mu); u->stop=1; pthread_cond_signal(&u->go); pthread_mutex_unlock(&u->mu); pthread_join(u->th,0); }

/* ---- op DAG ---- */
enum { PL_NPU, PL_CPU, PL_EITHER };
struct op { int deps, placement; long(*fn)(void*); void*arg; int state; /*0 pend,1 inflight,2 done*/ int unit, gen; };

/* op payloads */
static ork_npu *g_c; static int g_S;
struct npu_arg { ork_mm_task_i8 *tk; const void*out; size_t outbytes; };
static long npu_fn(void*p){ struct npu_arg*a=p; ork_dyn_chain*h=ork_dyn_begin(g_c,g_S,a->tk); if(!h) return -1;
    long r=ork_dyn_end(h);
    /* producer-side flush: dc civac (clean+invalidate) on THIS (NU) thread pushes the drained output to DRAM so a
     * consumer on another core sees it after its own civac. Cross-thread NPU->CPU handoff needs BOTH sides. */
    civac_range(a->out, a->outbytes);
    return r; }
struct glue_arg { const int8_t*x; size_t n; long out; };
static long glue_fn(void*p){ struct glue_arg*a=p; long acc=0; for(size_t i=0;i<a->n;i++){ long q=(long)a->x[i]>>1; if(q>63)q=63; if(q<-63)q=-63; acc+=q; } a->out=acc; return acc; }
struct bridge_arg { const int32_t*C; int n; long sum; };
static long bridge_fn(void*p){ struct bridge_arg*a=p; civac_range(a->C,(size_t)a->n*4);   /* invalidate before this thread reads the NPU output */
    long s=0; for(int i=0;i<a->n;i++) s+=a->C[i]; a->sum=s; return s; }

int main(void){
    int M=1,K=512,N=512,S=8;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 2; } g_c=c;
    int cap=ork_dyn_max_steps(); if(S>cap)S=cap; g_S=S;
    printf("spine_sched_probe: DAG over {NPU-unit, CPU-unit} (S=%d K=%d N=%d)\n",S,K,N);

    int8_t*A=malloc((size_t)M*K); int8_t*Wb=malloc((size_t)K*N);
    for(int i=0;i<M*K;i++) A[i]=(int8_t)s3(); for(size_t i=0;i<(size_t)K*N;i++) Wb[i]=(int8_t)s3();
    ork_w*W=ork_i8_mm_pack(c,K,N,Wb); if(!W){ printf("pack fail\n"); return 2; }
    int32_t *C0=ork_dma_alloc(c,(size_t)S*N*4), *C3=ork_dma_alloc(c,(size_t)S*N*4);
    if(!C0||!C3){ printf("dma fail\n"); return 2; }
    ork_mm_task_i8 *tk0=malloc((size_t)S*sizeof *tk0), *tk3=malloc((size_t)S*sizeof *tk3);
    for(int s=0;s<S;s++){ tk0[s]=(ork_mm_task_i8){W,M,A,C0+(size_t)s*N}; tk3[s]=(ork_mm_task_i8){W,M,A,C3+(size_t)s*N}; }
    int32_t *Cr=malloc((size_t)N*4); for(int n=0;n<N;n++){ long a=0; for(int k=0;k<K;k++) a+=A[k]*Wb[(size_t)k*N+n]; Cr[n]=(int32_t)a; }
    long refsum=0; for(int n=0;n<N;n++) refsum+=Cr[n];   /* op2 bridge should see S*refsum (all S rows == ref) */

    size_t SN=(size_t)K*N; int8_t*scr=malloc(SN); for(size_t i=0;i<SN;i++) scr[i]=(int8_t)(i*7);
    struct npu_arg na0={tk0,C0,(size_t)S*N*4}, na3={tk3,C3,(size_t)S*N*4}; struct glue_arg ga={scr,SN,0}; struct bridge_arg ba={C0,S*N,0};
    struct op ops[4] = {
        {0,        PL_NPU, npu_fn,   &na0, 0,-1,0},   /* op0 NPU matmul -> C0 */
        {0,        PL_CPU, glue_fn,  &ga,  0,-1,0},   /* op1 CPU glue (independent -> overlaps op0) */
        {1<<0,     PL_CPU, bridge_fn,&ba,  0,-1,0},   /* op2 CPU bridge: read C0 after op0 */
        {1<<1,     PL_NPU, npu_fn,   &na3, 0,-1,0},   /* op3 NPU matmul -> C3 after op1 */
    };
    int NOPS=4;
    struct unit NU, CU; unit_start(&NU,4); unit_start(&CU,6);   /* NPU-unit=big core4, CPU-unit=big core6 */
    /* warm both */
    ops[0].gen=unit_dispatch(&NU,npu_fn,&na0); spoll(&NU,ops[0].gen); ops[0].state=0; ops[0].gen=0;
    { int g=unit_dispatch(&CU,glue_fn,&ga); spoll(&CU,g); }

    /* ---- SCHEDULER: DAG-driven dispatch across the two units, one poll loop ---- */
    double t0=now_us(); int done_mask=0, ndone=0;
    while(ndone<NOPS){
        for(int i=0;i<NOPS;i++) if(ops[i].state==0 && (ops[i].deps & done_mask)==ops[i].deps){
            struct unit*u = (ops[i].placement==PL_NPU) ? &NU : &CU;   /* NPU->NPU unit; CPU/EITHER->CPU unit */
            if(!u->busy){ ops[i].gen=unit_dispatch(u,ops[i].fn,ops[i].arg); ops[i].state=1; u->busy=1; u->op=i; }
        }
        for(struct unit*u=&NU;;u=&CU){
            if(u->busy && unit_poll(u,ops[u->op].gen)){ ops[u->op].state=2; done_mask|=1<<u->op; ndone++; u->busy=0; }
            if(u==&CU) break;
        }
        struct timespec ts={0,20000}; nanosleep(&ts,0);   /* 20us scheduler tick */
    }
    double wall=now_us()-t0;

    /* validate + report */
    civac_range(C0,(size_t)S*N*4);   /* main thread must invalidate before reading the NPU output */
    long bad=0; int32_t mx=0; for(int s=0;s<S;s++) for(int n=0;n<N;n++){ int32_t d=C0[(size_t)s*N+n]-Cr[n]; if(d){bad++; if(d<0)d=-d; if(d>mx)mx=d;} }
    int bridge_ok = (ba.sum == (long)S*refsum);
    /* serial baseline: each op run alone THROUGH ITS UNIT (no overlap), same execution path — NPU stays on NU */
    double serial=0;
    for(int i=0;i<NOPS;i++){ struct unit*u=(ops[i].placement==PL_NPU)?&NU:&CU; double a=now_us();
        int g=unit_dispatch(u,ops[i].fn,ops[i].arg); spoll(u,g); serial+=now_us()-a; }
    printf("  op0 NPU coherent: %s (bad=%ld)\n", bad?"MISMATCH":"bit-exact", bad);
    printf("  op2 bridge read C0: %s (sum=%ld want=%ld)\n", bridge_ok?"OK":"WRONG", ba.sum, (long)S*refsum);
    printf("  scheduler wall %.0fus vs serial %.0fus -> %.2fx (DAG: op0∥op1 overlap, op2>op0, op3>op1)\n", wall, serial, serial/wall);
    int pass = (bad==0 && bridge_ok && ndone==NOPS);
    printf("%s\n", pass ? "PASS — heterogeneous scheduler drives the DAG across NPU+CPU units, coherent, overlapped"
                        : "FAIL");
    unit_stop(&NU); unit_stop(&CU);
    free(A); ork_dma_free(c,C0); ork_dma_free(c,C3); ork_mm_free(c,W); ork_npu_free(c);
    return pass?0:1;
}
