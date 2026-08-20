/* moe_concurrent_probe — FEASIBILITY: can an NPU expert submit OVERLAP with CPU expert compute at
 * M=1 decode, and does the per-layer activation crossing let a concurrent split BEAT the CPU-fused MoE?
 *
 * Prior MoE-on-NPU tests ran NPU experts SERIALLY with the CPU -> more NPU work = more total time ->
 * decode always lost. The ONLY untested lever: fire j experts on the NPU AND k-j on the CPU CONCURRENTLY,
 * so per-token MoE latency becomes max(NPU_part, CPU_part) + crossing, NOT the sum. The win (if any) is
 * overlap/saturation, not the NPU being faster per-expert.
 *
 * ork_i8_mm_run_stream is BLOCKING (runs core0 on the calling thread + joins its pool); the rknpu submit
 * ioctl is a blocking kernel wait. We overlap by running the WHOLE stream call on a DEDICATED thread while
 * the remaining k-j experts run on a CPU threadpool on the COMPLEMENTARY big cores. The NPU stream pins
 * its workers top-down (cpu = ncpu-1-id => 7,6,5..); we pin the CPU experts BOTTOM big core(s) (4,5..) so
 * the two clusters don't fight for the same A76.
 *
 * Measures per "decode layer" (M=1, k active experts), median of REPS:
 *   T_cpu_all    all k experts on CPU, threaded (the baseline to beat).
 *   T_npu_serial j on NPU (run_stream_i8) THEN k-j on CPU, SERIAL + crossing.
 *   T_concurrent j NPU on a dedicated thread || k-j CPU threadpool, OVERLAPPED, then join+combine.
 *   T_crossing   quantize token act + copy into NPU buf + copy result back + combine, ALONE.
 * Sweeps the split j in 0..k.
 *
 * Board-only (needs /dev/dri + rknpu). NOT in all/test.
 *   build: make moe_concurrent_probe   run: sudo taskset -c 0-7 ./moe_concurrent_probe [model]
 *   model = lfm (default) | qwen
 * dummy data (TIMING probe, not correctness — but every NPU stream output is sanity-checked vs CPU once).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sched.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y?1:0; }
static double median(double*s,int n){ qsort(s,n,sizeof(double),cmp_d); return n&1?s[n/2]:0.5*(s[n/2-1]+s[n/2]); }

/* M=1 int8 GEMV: C[N] = A[K] . B[n][K]. NEON SDOT-ish (vmull/vpadal). */
static void cpu_gemv_i8(int K, int N, const int8_t *A, const int8_t *B8, int32_t *C){
#ifdef __ARM_NEON
    for(int n=0;n<N;n++){ const int8_t *b=B8+(size_t)n*K;
        int32x4_t acc=vdupq_n_s32(0); int k=0;
        for(;k+16<=K;k+=16){
            int8x16_t va=vld1q_s8(A+k), vb=vld1q_s8(b+k);
            int16x8_t lo=vmull_s8(vget_low_s8(va),vget_low_s8(vb));
            int16x8_t hi=vmull_s8(vget_high_s8(va),vget_high_s8(vb));
            acc=vpadalq_s16(acc,lo); acc=vpadalq_s16(acc,hi);
        }
        int32_t s=vaddvq_s32(acc);
        for(;k<K;k++) s += (int)A[k]*(int)b[k];
        C[n]=s;
    }
#else
    for(int n=0;n<N;n++){ long s=0; const int8_t*b=B8+(size_t)n*K; for(int k=0;k<K;k++) s+=(int)A[k]*(int)b[k]; C[n]=(int32_t)s; }
#endif
}

/* ---- CPU expert threadpool: a fixed set of workers, each pinned to a chosen big core, that compute
 * a contiguous slice of the [lo,hi) active experts. Persistent (spun up once) so spawn cost isn't timed. */
struct cpu_pool {
    int nthreads; int pinbase;     /* pin to cores pinbase, pinbase+1, ... (bottom-up big cores) */
    int K, N; const int8_t *A; int8_t *const*B8; int32_t *const*C;  /* job */
    int lo, hi;                    /* experts [lo,hi) to do this job */
    pthread_t th[8]; pthread_mutex_t mu; pthread_cond_t go, dn;
    int gen, done, stop;
};
static void *cpu_worker(void *vp){
    struct cpu_pool *p = vp;
    /* derive this worker's id by spin: we encode id in the low bits via a per-call ticket is overkill;
     * instead store id in a thread-local set at create. We pass id via a tiny wrapper below. */
    (void)p; return NULL;
}
struct cpu_worker_arg { struct cpu_pool *p; int id; };
static void *cpu_worker2(void *vp){
    struct cpu_worker_arg *wa = vp; struct cpu_pool *p = wa->p; int id = wa->id;
    if(!getenv("ORK_NO_AFFINITY")){
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(p->pinbase + id, &s);
        pthread_setaffinity_np(pthread_self(), sizeof s, &s);
    }
    int mygen = 0;
    for(;;){
        pthread_mutex_lock(&p->mu);
        while(p->gen==mygen && !p->stop) pthread_cond_wait(&p->go, &p->mu);
        if(p->stop){ pthread_mutex_unlock(&p->mu); return NULL; }
        mygen = p->gen;
        pthread_mutex_unlock(&p->mu);
        /* split [lo,hi) experts round-robin across nthreads workers */
        for(int e = p->lo + id; e < p->hi; e += p->nthreads)
            cpu_gemv_i8(p->K, p->N, p->A, p->B8[e], p->C[e]);
        pthread_mutex_lock(&p->mu);
        if(++p->done == p->nthreads) pthread_cond_signal(&p->dn);
        pthread_mutex_unlock(&p->mu);
    }
}
static struct cpu_worker_arg g_wa[8];
static void cpu_pool_start(struct cpu_pool *p, int nthreads, int pinbase){
    memset(p,0,sizeof *p); p->nthreads=nthreads; p->pinbase=pinbase;
    pthread_mutex_init(&p->mu,0); pthread_cond_init(&p->go,0); pthread_cond_init(&p->dn,0);
    p->gen=0; p->done=0; p->stop=0;
    (void)cpu_worker;
    for(int i=0;i<nthreads;i++){ g_wa[i].p=p; g_wa[i].id=i; pthread_create(&p->th[i],0,cpu_worker2,&g_wa[i]); }
}
/* dispatch [lo,hi) experts; if block!=0 wait for completion */
static void cpu_pool_dispatch(struct cpu_pool *p, int lo, int hi, const int8_t*A, int8_t*const*B8, int32_t*const*C, int K,int N){
    pthread_mutex_lock(&p->mu);
    p->lo=lo; p->hi=hi; p->A=A; p->B8=B8; p->C=C; p->K=K; p->N=N; p->done=0; p->gen++;
    pthread_cond_broadcast(&p->go);
    pthread_mutex_unlock(&p->mu);
}
static void cpu_pool_wait(struct cpu_pool *p){
    pthread_mutex_lock(&p->mu); while(p->done < p->nthreads) pthread_cond_wait(&p->dn,&p->mu); pthread_mutex_unlock(&p->mu);
}
static void cpu_pool_stop(struct cpu_pool *p){
    pthread_mutex_lock(&p->mu); p->stop=1; pthread_cond_broadcast(&p->go); pthread_mutex_unlock(&p->mu);
    for(int i=0;i<p->nthreads;i++) pthread_join(p->th[i],0);
}

/* ---- NPU stream on a dedicated thread (so its blocking submit overlaps the CPU pool) ---- */
struct npu_arg { ork_npu *c; int S; ork_mm_task_i8 *tasks; int rc; };
static void *npu_thread(void *vp){
    struct npu_arg *a = vp;
    a->rc = ork_i8_mm_run_stream(a->c, a->S, a->tasks);
    if(a->rc){ a->rc=0; for(int x=0;x<a->S && a->rc==0;x++) a->rc=ork_i8_mm_run(a->c,a->tasks[x].w,a->tasks[x].M,a->tasks[x].A,a->tasks[x].C); }
    return NULL;
}

static uint32_t g_s = 0x2468acef;
static int8_t r8(void){ g_s^=g_s<<13; g_s^=g_s>>17; g_s^=g_s<<5; return (int8_t)((int)(g_s&0xff)-128); }
static float rf(void){ g_s^=g_s<<13; g_s^=g_s>>17; g_s^=g_s<<5; return ((int)(g_s&0xffff)-32768)/9000.0f; }

struct model { const char *name; int K, N; int topk; };

int main(int argc, char**argv){
    const char *which = argc>1 ? argv[1] : "lfm";
    enum { WARM=8, REPS=40 };
    ork_npu *c = ork_npu_init();
    if(!c){ fprintf(stderr,"init FAIL (need /dev/dri + rknpu)\n"); return 1; }
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    printf("# ork-driver %s soc=%s cores=%d ncpu=%ld\n", ork_npu_version(), ork_npu_soc(c), ork_npu_cores(c), ncpu);

    /* one decode layer = gate/up shape (the dominant, conforming one). We probe gate/up (stream-eligible)
     * as the headline; down is non-conforming (falls to K-split run_i8) and reported separately if asked. */
    struct model models_lfm[]  = { {"LFM2.5 gate/up", 2048, 1792, 4}, {"LFM2.5 down", 1792, 2048, 4} };
    struct model models_qwen[] = { {"Qwen3.6 gate/up", 2048, 512, 8}, {"Qwen3.6 down", 512, 2048, 8} };
    struct model *ms; int nms;
    if(!strcmp(which,"qwen")){ ms=models_qwen; nms=2; } else { ms=models_lfm; nms=2; }

    int big_cores = 4;                                 /* RK3588 A76 count */
    /* NPU stream pins top-down (7,6,5). CPU experts pin bottom-up from core (big_lo). On RK3588 big = 4..7.
     * Give CPU the bottom big cores not used by the NPU stream this split. */
    int big_lo = (int)ncpu - big_cores;                /* 4 */

    for(int mi=0; mi<nms; mi++){
        int K = ms[mi].K, N = ms[mi].N, topk = ms[mi].topk;
        int conforming = (K%512==0 && K<=4096);
        printf("\n=== %s  K=%d N=%d  top-k=%d  (stream-eligible: %s) ===\n",
               ms[mi].name, K, N, topk, conforming?"yes":"NO (K-split fallback)");

        /* build topk distinct resident int8 experts */
        float **f32 = malloc(topk*sizeof(float*));
        int8_t **B8 = malloc(topk*sizeof(int8_t*));
        ork_w **w   = malloc(topk*sizeof(ork_w*));
        float **bsc = malloc(topk*sizeof(float*));
        int32_t **C = malloc(topk*sizeof(int32_t*));
        for(int e=0;e<topk;e++){
            f32[e]=malloc((size_t)N*K*sizeof(float)); for(size_t i=0;i<(size_t)N*K;i++) f32[e][i]=rf();
            bsc[e]=malloc((size_t)N*sizeof(float));
            w[e]=ork_i8_mm_pack_f32(c,K,N,f32[e],bsc[e]);
            if(!w[e]){ printf("  pack FAIL e=%d\n",e); return 1; }
            B8[e]=malloc((size_t)N*K);
            for(int n=0;n<N;n++){ const float*wn=f32[e]+(size_t)n*K; float amax=0; for(int k=0;k<K;k++){float a=fabsf(wn[k]); if(a>amax)amax=a;}
                float sc=amax>0?amax/127.0f:1.0f;
                for(int k=0;k<K;k++){ int q=(int)lrintf(wn[k]/sc); if(q>127)q=127; if(q<-128)q=-128; B8[e][(size_t)n*K+k]=(int8_t)q; } }
            C[e]=malloc((size_t)N*4);
        }
        int8_t *A = malloc((size_t)K); for(int i=0;i<K;i++) A[i]=r8();
        int32_t *Cref = malloc((size_t)N*4);
        float   *Aout = malloc((size_t)K*sizeof(float)); for(int i=0;i<K;i++) Aout[i]=rf(); /* fp32 token act (pre-quant) */

        /* ---- sanity: NPU stream output matches CPU for expert 0 (M=1) ---- */
        ork_mm_task_i8 chk = { .w=w[0], .M=1, .A=A, .C=C[0] };
        int crc = ork_i8_mm_run_stream(c,1,&chk);
        if(crc==0){
            cpu_gemv_i8(K,N,A,B8[0],Cref);
            long maxd=0; for(int i=0;i<N;i++){ long d=labs((long)C[0][i]-(long)Cref[i]); if(d>maxd)maxd=d; }
            printf("  [sanity] stream vs CPU expert0 maxd=%ld (tol ~%d) %s\n", maxd, K, maxd<=(long)K?"OK":"SUSPECT");
        } else printf("  [sanity] stream rc=%d (non-conforming -> per-task fallback used in probe)\n", crc);

        /* CPU pool sized to remaining big cores (worst case k-j experts). Use up to big_cores threads. */
        struct cpu_pool pool; cpu_pool_start(&pool, big_cores, big_lo);

        double sm[REPS];

        /* ---- T_crossing: quantize fp32 token act -> int8 A (per-token amax), copy in, run+copy out, dequant+combine.
         * Isolate the per-layer crossing alone (no expert GEMM compute attributed; we just exercise the data path). */
        {
            int8_t *Aq = malloc(K); int32_t *Cq = malloc((size_t)N*4); float *acc = malloc((size_t)N*sizeof(float));
            for(int i=0;i<WARM;i++){
                float amax=0; for(int k=0;k<K;k++){ float a=fabsf(Aout[k]); if(a>amax)amax=a; }
                float sc = amax>0?amax/127.0f:1.0f, inv=1.0f/sc;
                for(int k=0;k<K;k++){ int q=(int)lrintf(Aout[k]*inv); if(q>127)q=127; if(q<-128)q=-128; Aq[k]=(int8_t)q; }
                for(int n=0;n<N;n++) acc[n] += (float)Cq[n]*sc;
            }
            for(int r=0;r<REPS;r++){ double t=now_us();
                float amax=0; for(int k=0;k<K;k++){ float a=fabsf(Aout[k]); if(a>amax)amax=a; }
                float sc = amax>0?amax/127.0f:1.0f, inv=1.0f/sc;
                for(int k=0;k<K;k++){ int q=(int)lrintf(Aout[k]*inv); if(q>127)q=127; if(q<-128)q=-128; Aq[k]=(int8_t)q; }
                for(int n=0;n<N;n++) acc[n] += (float)Cq[n]*sc;
                sm[r]=now_us()-t; }
            double tx = median(sm,REPS);
            printf("  T_crossing (quant+dequant+combine, per layer) = %.2f us\n", tx);
            free(Aq); free(Cq); free(acc);
        }

        /* ---- T_cpu_all: all topk experts on CPU pool ---- */
        for(int i=0;i<WARM;i++){ cpu_pool_dispatch(&pool,0,topk,A,B8,C,K,N); cpu_pool_wait(&pool); }
        for(int r=0;r<REPS;r++){ double t=now_us(); cpu_pool_dispatch(&pool,0,topk,A,B8,C,K,N); cpu_pool_wait(&pool); sm[r]=now_us()-t; }
        double t_cpu_all = median(sm,REPS);
        printf("  T_cpu_all (%d experts, %d threads) = %.2f us\n", topk, big_cores, t_cpu_all);

        /* anchor: NPU submit floor for ONE expert via stream (j=1) */
        {
            ork_mm_task_i8 t1 = { .w=w[0], .M=1, .A=A, .C=C[0] };
            for(int i=0;i<WARM;i++) ork_i8_mm_run_stream(c,1,&t1);
            for(int r=0;r<REPS;r++){ double t=now_us(); ork_i8_mm_run_stream(c,1,&t1); sm[r]=now_us()-t; }
            printf("  [anchor] NPU stream 1-expert floor = %.2f us\n", median(sm,REPS));
        }

        printf("  %-3s | %-18s | %-18s | %-12s | overlap_eff | beats_cpu_all\n",
               "j", "T_npu_serial(us)", "T_concurrent(us)", "T_npu_part");
        /* sweep split j = experts on NPU, k-j on CPU */
        for(int j=0; j<=topk; j++){
            int kj = topk - j;   /* CPU experts */
            ork_mm_task_i8 ntasks[8];
            for(int e=0;e<j;e++){ ntasks[e].w=w[e]; ntasks[e].M=1; ntasks[e].A=A; ntasks[e].C=C[e]; }

            /* measure NPU-only part (j experts) for the overlap-efficiency denominator */
            double t_npu_part = 0;
            if(j>0){
                struct npu_arg na = {c,j,ntasks,0};
                for(int i=0;i<WARM;i++) npu_thread(&na);
                for(int r=0;r<REPS;r++){ double t=now_us(); npu_thread(&na); sm[r]=now_us()-t; }
                t_npu_part = median(sm,REPS);
            }
            /* CPU-only part (kj experts) */
            double t_cpu_part = 0;
            if(kj>0){
                for(int i=0;i<WARM;i++){ cpu_pool_dispatch(&pool,j,topk,A,B8,C,K,N); cpu_pool_wait(&pool); }
                for(int r=0;r<REPS;r++){ double t=now_us(); cpu_pool_dispatch(&pool,j,topk,A,B8,C,K,N); cpu_pool_wait(&pool); sm[r]=now_us()-t; }
                t_cpu_part = median(sm,REPS);
            }

            /* T_npu_serial = NPU part THEN CPU part, sequential */
            double t_serial = 0;
            {
                struct npu_arg na = {c,j,ntasks,0};
                for(int i=0;i<WARM;i++){ if(j>0) npu_thread(&na); if(kj>0){ cpu_pool_dispatch(&pool,j,topk,A,B8,C,K,N); cpu_pool_wait(&pool);} }
                for(int r=0;r<REPS;r++){ double t=now_us();
                    if(j>0) npu_thread(&na);
                    if(kj>0){ cpu_pool_dispatch(&pool,j,topk,A,B8,C,K,N); cpu_pool_wait(&pool);}
                    sm[r]=now_us()-t; }
                t_serial = median(sm,REPS);
            }

            /* T_concurrent = NPU part on dedicated thread || CPU pool, overlapped, then join */
            double t_conc = 0;
            {
                for(int i=0;i<WARM;i++){
                    struct npu_arg na = {c,j,ntasks,0}; pthread_t th;
                    if(j>0) pthread_create(&th,0,npu_thread,&na);
                    if(kj>0){ cpu_pool_dispatch(&pool,j,topk,A,B8,C,K,N); cpu_pool_wait(&pool);}
                    if(j>0) pthread_join(th,0);
                }
                for(int r=0;r<REPS;r++){ double t=now_us();
                    struct npu_arg na = {c,j,ntasks,0}; pthread_t th;
                    if(j>0) pthread_create(&th,0,npu_thread,&na);
                    if(kj>0){ cpu_pool_dispatch(&pool,j,topk,A,B8,C,K,N); cpu_pool_wait(&pool);}
                    if(j>0) pthread_join(th,0);
                    sm[r]=now_us()-t; }
                t_conc = median(sm,REPS);
            }

            double maxpart = t_npu_part>t_cpu_part? t_npu_part : t_cpu_part;
            double eff = (j>0 && kj>0 && t_conc>0) ? (t_serial / t_conc) : 0; /* 1=no overlap, ->2 = full overlap of equal halves */
            char effbuf[24]; if(j>0&&kj>0) snprintf(effbuf,sizeof effbuf,"%.2fx (max=%.0f)",eff,maxpart); else snprintf(effbuf,sizeof effbuf,"n/a");
            printf("  %-3d | %-18.2f | %-18.2f | %-12.2f | %-11s | %s\n",
                   j, t_serial, t_conc, t_npu_part, effbuf, (t_conc>0 && t_conc < t_cpu_all)?"YES":"no");
        }

        cpu_pool_stop(&pool);
        for(int e=0;e<topk;e++){ ork_mm_free(c,w[e]); free(f32[e]); free(B8[e]); free(bsc[e]); free(C[e]); }
        free(f32);free(B8);free(w);free(bsc);free(C);free(A);free(Cref);free(Aout);
    }

    ork_npu_free(c);
    return 0;
}
