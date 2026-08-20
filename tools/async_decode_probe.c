/* tools/async_decode_probe.c — validate the decode async-overlap hypothesis on real matmuls.
 *
 * The decode wall (Exp-2026-06-30): with ORK_DECODE_MC=1 the NPU runs ~33% busy and cpu7 ~62% —
 * the synchronous backend pays wall ≈ t_cpu + t_npu (they take turns). The hypothesis is that
 * overlapping the CPU prep behind the NPU compute (async dispatch) collapses that toward
 * wall ≈ max(t_cpu, t_npu). This probe TESTS that directly, end-to-end, on the real decode matmuls.
 *
 * Method: pack the 7 per-layer Qwen2.5-7B-Q8_0 projections (Q,K,V,O,gate,up,down) resident, M=1,
 * 3-core (the ORK_DECODE_MC config). A "token" = run all 7. Between matmuls run a REAL representative
 * CPU-prep workload (activation requant of the K-row + an RMSNorm pass — genuine float work, not a
 * sleep), sized by a sweepable multiplier. Compare:
 *   SYNC  : for each m: cpu_prep(m); ork_i8_mm_run(m)            -> wall ≈ Σ(t_cpu + t_npu)
 *   ASYNC : pipeline — launch run(m) async; cpu_prep(m+1) overlaps; wait(m) -> wall ≈ Σ max(t_cpu,t_npu)
 * Sweep the CPU/NPU ratio (multiplier) and print sync/async/speedup vs the sum→max ceiling, so the
 * curve shows what async buys at decode's measured ratio (~1.9). Bit-exact gate: async C == sync C.
 *   make async_decode_probe && sudo ./async_decode_probe [iters]
 *
 * NOTE on dependencies: a real decode layer is partly a dependency chain (matmul m+1's input needs
 * m's output + a norm), so the pipelined number here is the ACHIEVABLE CEILING for the overlap
 * mechanism — the independent groups {Q,K,V} and {gate,up} can overlap; the chain edges can't. The
 * point is to validate the mechanism and quantify the ceiling, not to claim a full decode is this fast.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sched.h>
#include <pthread.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

/* representative between-matmul CPU prep: requant a K-row f32->int8 (RMSNorm + scale), `reps` passes.
 * volatile sink prevents the optimizer from deleting it. Returns nothing; cost ~ reps*K. */
static volatile int32_t g_sink;
static void cpu_prep(const float *x, int8_t *out, int K, int reps){
    for(int r=0;r<reps;r++){
        float ss=0.f; for(int i=0;i<K;i++) ss+=x[i]*x[i];
        float inv=1.f/sqrtf(ss/(float)K + 1e-6f);
        float amax=1e-9f; for(int i=0;i<K;i++){ float v=x[i]*inv; float a=fabsf(v); if(a>amax)amax=a; }
        float s=127.f/amax; int32_t acc=0;
        for(int i=0;i<K;i++){ int q=(int)lrintf(x[i]*inv*s); if(q>127)q=127; if(q<-128)q=-128; out[i]=(int8_t)q; acc+=q; }
        g_sink=acc;
    }
}

/* Prototype of the REAL async fix: a PERSISTENT worker created once and pinned to a chosen big core
 * (or a big-core SET) — not per-matmul pthread_create (spawn cost), not inheriting the caller's pin
 * (collision), not unpinned (A55 drift). Mirrors ork's own run_multicore pool (condvar feed/done). */
typedef struct {
    ork_npu*c; ork_w*w; int M; const int8_t*A; int32_t*C;   /* the job */
    int core_lo, core_hi;                                   /* bind to cores [lo..hi] (big-core set) */
    pthread_mutex_t mu; pthread_cond_t go, dn; int has_job, done, stop;
} awork;
static void* aworker(void*p){ awork*a=(awork*)p;
    cpu_set_t s; CPU_ZERO(&s); for(int k=a->core_lo;k<=a->core_hi;k++) CPU_SET(k,&s);
    pthread_setaffinity_np(pthread_self(),sizeof s,&s);     /* bound to the big-core set, never A55 */
    for(;;){ pthread_mutex_lock(&a->mu);
        while(!a->has_job && !a->stop) pthread_cond_wait(&a->go,&a->mu);
        if(a->stop){ pthread_mutex_unlock(&a->mu); return NULL; }
        a->has_job=0; pthread_mutex_unlock(&a->mu);
        ork_i8_mm_run(a->c,a->w,a->M,a->A,a->C);
        pthread_mutex_lock(&a->mu); a->done=1; pthread_cond_signal(&a->dn); pthread_mutex_unlock(&a->mu); }
}
static void aw_submit(awork*a, ork_w*w, int M, const int8_t*A, int32_t*C){
    pthread_mutex_lock(&a->mu); a->w=w; a->M=M; a->A=A; a->C=C; a->done=0; a->has_job=1;
    pthread_cond_signal(&a->go); pthread_mutex_unlock(&a->mu); }
static void aw_wait(awork*a){
    pthread_mutex_lock(&a->mu); while(!a->done) pthread_cond_wait(&a->dn,&a->mu); pthread_mutex_unlock(&a->mu); }

int main(int argc,char**argv){
    int iters=argc>1?atoi(argv[1]):60;
    /* ORK_PROBE_PIN=<lo> [ORK_PROBE_PIN_HI=<hi>] → use a PERSISTENT worker bound to big cores [lo..hi]
     * (e.g. PIN=5 → {5}; PIN=4 HI=6 → big-core set {4,5,6}); caller pinned to ORK_PROBE_CALLER (default 7).
     * All big (cpu4-7 on RK3588) — overlap WITHOUT little-core drift, no per-matmul spawn. Unset → stock
     * ork async API (per-call pthread_create, inherits caller pin unless ORK_NO_AFFINITY=1). */
    int pin_lo=-1; { const char*e=getenv("ORK_PROBE_PIN"); if(e) pin_lo=atoi(e); }
    int pin_hi=pin_lo; { const char*e=getenv("ORK_PROBE_PIN_HI"); if(e) pin_hi=atoi(e); }
    int caller_core=7; { const char*e=getenv("ORK_PROBE_CALLER"); if(e) caller_core=atoi(e); }
    if(pin_lo>=0){ cpu_set_t s; CPU_ZERO(&s); CPU_SET(caller_core,&s); pthread_setaffinity_np(pthread_self(),sizeof s,&s); }
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int cores=ork_npu_cores(c);
    int budget=cores; { const char*e=getenv("ORK_PROBE_CORES"); if(e) budget=atoi(e); if(budget<1)budget=1; if(budget>cores)budget=cores; }
    printf("INIT OK: soc=%s cores=%d  submit-budget=%d  async=%s\n",
           ork_npu_soc(c), cores, budget,
           pin_lo>=0 ? "PERSISTENT worker (pinned big-core set, no per-call spawn)" : "stock ork API (per-call pthread_create)");
    if(pin_lo>=0) printf("  caller pinned cpu%d ; worker bound cpu%d..%d\n", caller_core, pin_lo, pin_hi);
    ork_npu_set_core_budget(c, budget);

    /* start the persistent async worker (production-shape fix) */
    awork aw; memset(&aw,0,sizeof aw);
    pthread_t awth=0;
    if(pin_lo>=0){ aw.c=c; aw.core_lo=pin_lo; aw.core_hi=pin_hi;
        pthread_mutex_init(&aw.mu,NULL); pthread_cond_init(&aw.go,NULL); pthread_cond_init(&aw.dn,NULL);
        pthread_create(&awth,NULL,aworker,&aw); }

    /* Qwen2.5-7B decode projections, M=1: {K,N} for Q,K,V,O,gate,up,down */
    int KN[7][2]={ {3584,3584},{3584,512},{3584,512},{3584,3584},{3584,18944},{3584,18944},{18944,3584} };
    const char*nm[7]={"Q","K","V","O","gate","up","down"};
    int M=1;

    ork_w *w[7]; int8_t *A[7]; int32_t *Cs[7], *Ca[7]; float *X[7]; int8_t *prep[7];
    for(int m=0;m<7;m++){
        int K=KN[m][0],N=KN[m][1];
        int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
        ork_npu_set_pack_domain(c,0);
        w[m]=ork_i8_mm_pack(c,K,N,B); free(B);
        if(!w[m]){printf("pack %s (K%d N%d) failed\n",nm[m],K,N);return 1;}
        A[m]=malloc((size_t)M*K); memset(A[m],1,(size_t)M*K);
        Cs[m]=malloc((size_t)M*N*4); Ca[m]=malloc((size_t)M*N*4);
        X[m]=malloc((size_t)K*sizeof(float)); for(int i=0;i<K;i++)X[m][i]=0.01f*((i%17)-8);
        prep[m]=malloc((size_t)K);
    }

    /* warm + per-matmul NPU time (sync) */
    for(int m=0;m<7;m++) ork_i8_mm_run(c,w[m],M,A[m],Cs[m]);
    double t_npu_tok=0;
    for(int m=0;m<7;m++){
        double t0=now_us(); for(int it=0;it<iters;it++) ork_i8_mm_run(c,w[m],M,A[m],Cs[m]);
        double tm=(now_us()-t0)/iters; t_npu_tok+=tm;
        printf("  NPU %-4s K%-5d N%-5d : %7.1f us\n", nm[m],KN[m][0],KN[m][1],tm);
    }
    printf("  -> NPU per-token (7 matmuls): %.1f us\n\n", t_npu_tok);

    /* bit-exact gate: async result == sync result (Q) */
    ork_async*h=ork_i8_mm_run_async(c,w[0],M,A[0],Ca[0]); cpu_prep(X[0],prep[0],KN[0][0],4); ork_async_wait(h);
    int mism=0; for(int j=0;j<KN[0][1];j++) if(Cs[0][j]!=Ca[0][j]) mism++;
    printf("async-vs-sync bit-exact: mism=%d %s\n\n", mism, mism?"FAIL":"OK");

    printf("%-8s %8s %9s %9s %8s %9s %8s\n","cpu/npu","t_cpu","sync(us)","async(us)","speedup","ceil(us)","of-ceil");
    int reps_set[]={2,8,18,40,70,110,170}; /* sweep CPU-prep size -> CPU/NPU ratio (reach decode ~1.9) */
    for(unsigned ri=0; ri<sizeof(reps_set)/sizeof(int); ri++){
        int reps=reps_set[ri];
        /* measure t_cpu per token at this reps */
        double tc0=now_us(); for(int m=0;m<7;m++) cpu_prep(X[m],prep[m],KN[m][0],reps); double t_cpu_tok=now_us()-tc0;

        /* SYNC: cpu_prep then blocking run, summed */
        double s0=now_us();
        for(int it=0;it<iters;it++) for(int m=0;m<7;m++){ cpu_prep(X[m],prep[m],KN[m][0],reps); ork_i8_mm_run(c,w[m],M,A[m],Cs[m]); }
        double t_sync=(now_us()-s0)/iters;

        /* ASYNC: pipeline cpu_prep(m+1) behind NPU compute(m). pin_lo>=0 → persistent pinned worker
         * (production fix); else → stock per-call ork async API. */
        double a0=now_us();
        for(int it=0;it<iters;it++){
            cpu_prep(X[0],prep[0],KN[0][0],reps);
            for(int m=0;m<7;m++){
                if(pin_lo>=0){
                    aw_submit(&aw,w[m],M,A[m],Ca[m]);
                    if(m<6) cpu_prep(X[m+1],prep[m+1],KN[m+1][0],reps);
                    aw_wait(&aw);
                } else {
                    ork_async*hh=ork_i8_mm_run_async(c,w[m],M,A[m],Ca[m]);
                    if(m<6) cpu_prep(X[m+1],prep[m+1],KN[m+1][0],reps);
                    ork_async_wait(hh);
                }
            }
        }
        double t_async=(now_us()-a0)/iters;

        double ceil_us = (t_cpu_tok>t_npu_tok?t_cpu_tok:t_npu_tok); /* Σ max ~ max(Σ) here since per-m dominated similarly */
        printf("%-8.2f %8.1f %9.1f %9.1f %7.2fx %9.1f %7.0f%%\n",
               t_cpu_tok/t_npu_tok, t_cpu_tok, t_sync, t_async, t_sync/t_async, ceil_us, 100.0*ceil_us/t_async);
    }
    printf("\n(decode operating point ~ cpu/npu 1.9 from measured NPU 33%%/cpu7 62%%)\n");
    if(pin_lo>=0){ pthread_mutex_lock(&aw.mu); aw.stop=1; pthread_cond_signal(&aw.go); pthread_mutex_unlock(&aw.mu); pthread_join(awth,NULL); }
    ork_npu_free(c); return 0;
}
