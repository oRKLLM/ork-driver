/* moe_batched_probe — BATCHED (M>1) extension of moe_concurrent_probe (commit 999b7e8).
 *
 * The M=1 decode kill was M=1-specific: the NPU submit floor (~174-407us) >= the ENTIRE CPU-fused MoE
 * job, so there was nothing to overlap behind. At M>1 two things change in our favor:
 *   (a) the CPU's fused MoE job grows ~linearly with M -> finally enough CPU work to hide the NPU behind;
 *   (b) the NPU GEMM amortizes its fixed submit floor over M rows.
 * M=1 was tested two ways (B = serial M>1, concurrency = overlap M=1) but NEVER M>1 + CONCURRENT. This is
 * the throughput regime: prefill, speculative-verify batches, continuous batching.
 *
 * Per "decode/verify layer" (top-k active experts, M rows each), median of REPS:
 *   T_cpu_all(M)     : ALL k active experts on the CPU int8/NEON threaded GEMM (the baseline).
 *   T_distributed(M) : j experts on NPU (resident int8, run_stream on a dedicated thread, M>1 rows) ||
 *                      k-j experts on the CPU threadpool, OVERLAPPED + join + crossing. Sweep j, report BEST.
 *   T_crossing(M)    : per-layer activation quant(MxK) + dequant/combine(MxN) cost ALONE (grows with M).
 *
 * Real dims: LFM2.5-8B-A1B (top-4; gate/up K=2048 N=1792, down K=1792 N=2048) AND
 *            Qwen3.6-35B-A3B (top-8; 2048x512, 512x2048). down uses run_i8 (K-split) per task.
 * Dummy data (TIMING probe); NPU vs CPU output sanity-checked once per shape at M>1.
 *
 * Board-only (needs /dev/dri + rknpu). NOT in all/test.
 *   build: make moe_batched_probe   run: sudo taskset -c 0-7 ./moe_batched_probe [lfm|qwen]
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

/* M-row int8 GEMM: C[M,N] = A[M,K] . B[N,K]^T (B row-major [n][K]).
 * Uses ARM SDOT (vdotq_s32, asimddp) — the same instruction llama.cpp's int8 vec_dot uses on RK3588,
 * so T_cpu_all is a FAIR baseline (not the slower vmull/vpadal path). Falls back if no dotprod. */
static void cpu_gemm_i8(int M, int K, int N, const int8_t *A, const int8_t *B8, int32_t *C){
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    for(int m=0;m<M;m++){ const int8_t *a=A+(size_t)m*K; int32_t *cr=C+(size_t)m*N;
        int n=0;
        for(;n+4<=N;n+=4){
            const int8_t *b0=B8+(size_t)n*K,*b1=b0+K,*b2=b1+K,*b3=b2+K;
            int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0),a2=vdupq_n_s32(0),a3=vdupq_n_s32(0);
            int k=0;
            for(;k+16<=K;k+=16){ int8x16_t va=vld1q_s8(a+k);
                a0=vdotq_s32(a0,va,vld1q_s8(b0+k)); a1=vdotq_s32(a1,va,vld1q_s8(b1+k));
                a2=vdotq_s32(a2,va,vld1q_s8(b2+k)); a3=vdotq_s32(a3,va,vld1q_s8(b3+k)); }
            int32_t s0=vaddvq_s32(a0),s1=vaddvq_s32(a1),s2=vaddvq_s32(a2),s3=vaddvq_s32(a3);
            for(;k<K;k++){ s0+=(int)a[k]*(int)b0[k]; s1+=(int)a[k]*(int)b1[k]; s2+=(int)a[k]*(int)b2[k]; s3+=(int)a[k]*(int)b3[k]; }
            cr[n]=s0; cr[n+1]=s1; cr[n+2]=s2; cr[n+3]=s3;
        }
        for(;n<N;n++){ const int8_t *b=B8+(size_t)n*K; int32x4_t acc=vdupq_n_s32(0); int k=0;
            for(;k+16<=K;k+=16) acc=vdotq_s32(acc,vld1q_s8(a+k),vld1q_s8(b+k));
            int32_t s=vaddvq_s32(acc); for(;k<K;k++) s+=(int)a[k]*(int)b[k]; cr[n]=s; }
    }
#elif defined(__ARM_NEON)
    for(int m=0;m<M;m++){ const int8_t *a=A+(size_t)m*K; int32_t *cr=C+(size_t)m*N;
        for(int n=0;n<N;n++){ const int8_t *b=B8+(size_t)n*K;
            int32x4_t acc=vdupq_n_s32(0); int k=0;
            for(;k+16<=K;k+=16){
                int8x16_t va=vld1q_s8(a+k), vb=vld1q_s8(b+k);
                int16x8_t lo=vmull_s8(vget_low_s8(va),vget_low_s8(vb));
                int16x8_t hi=vmull_s8(vget_high_s8(va),vget_high_s8(vb));
                acc=vpadalq_s16(acc,lo); acc=vpadalq_s16(acc,hi);
            }
            int32_t s=vaddvq_s32(acc); for(;k<K;k++) s += (int)a[k]*(int)b[k]; cr[n]=s;
        }
    }
#else
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ long s=0; const int8_t*a=A+(size_t)m*K,*b=B8+(size_t)n*K;
        for(int k=0;k<K;k++) s+=(int)a[k]*(int)b[k]; C[(size_t)m*N+n]=(int32_t)s; }
#endif
}

/* ---- CPU expert threadpool: persistent workers pinned to chosen big cores; each does a round-robin
 * slice of the [lo,hi) active experts. Each expert is a full M-row GEMM. */
struct cpu_pool {
    int nthreads; int pinbase;
    int M, K, N; const int8_t *A; int8_t *const*B8; int32_t *const*C;
    int lo, hi;
    pthread_t th[8]; pthread_mutex_t mu; pthread_cond_t go, dn;
    int gen, done, stop;
};
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
        for(int e = p->lo + id; e < p->hi; e += p->nthreads)
            cpu_gemm_i8(p->M, p->K, p->N, p->A, p->B8[e], p->C[e]);
        pthread_mutex_lock(&p->mu);
        if(++p->done == p->nthreads) pthread_cond_signal(&p->dn);
        pthread_mutex_unlock(&p->mu);
    }
}
static struct cpu_worker_arg g_wa[8];
static void cpu_pool_start(struct cpu_pool *p, int nthreads, int pinbase){
    memset(p,0,sizeof *p); p->nthreads=nthreads; p->pinbase=pinbase;
    pthread_mutex_init(&p->mu,0); pthread_cond_init(&p->go,0); pthread_cond_init(&p->dn,0);
    for(int i=0;i<nthreads;i++){ g_wa[i].p=p; g_wa[i].id=i; pthread_create(&p->th[i],0,cpu_worker2,&g_wa[i]); }
}
static void cpu_pool_dispatch(struct cpu_pool *p, int lo, int hi, int M, const int8_t*A, int8_t*const*B8, int32_t*const*C, int K,int N){
    pthread_mutex_lock(&p->mu);
    p->lo=lo; p->hi=hi; p->M=M; p->A=A; p->B8=B8; p->C=C; p->K=K; p->N=N; p->done=0; p->gen++;
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

/* ---- NPU stream on a dedicated thread. j experts as a single run_stream_i8 (M>1 rows each). If the
 * stream rejects the shape (non-conforming K, rc=-3), fall back to per-task run_i8 (K-split). ---- */
struct npu_arg { ork_npu *c; int S; ork_mm_task_i8 *tasks; int rc; };
static void *npu_thread(void *vp){
    struct npu_arg *a = vp;
    a->rc = a->S>0 ? ork_mm_run_stream_i8(a->c, a->S, a->tasks) : 0;
    if(a->rc){ a->rc=0; for(int x=0;x<a->S && a->rc==0;x++) a->rc=ork_mm_run_i8(a->c,a->tasks[x].w,a->tasks[x].M,a->tasks[x].A,a->tasks[x].C); }
    return NULL;
}

static uint32_t g_s = 0x2468acef;
static int8_t r8(void){ g_s^=g_s<<13; g_s^=g_s>>17; g_s^=g_s<<5; return (int8_t)((int)(g_s&0xff)-128); }
static float rf(void){ g_s^=g_s<<13; g_s^=g_s>>17; g_s^=g_s<<5; return ((int)(g_s&0xffff)-32768)/9000.0f; }

struct model { const char *name; int K, N; int topk; };

int main(int argc, char**argv){
    const char *which = argc>1 ? argv[1] : "lfm";
    enum { WARM=4, REPS=20 };
    int Ms[] = {1, 8, 16, 32, 64, 128};
    int nM = (int)(sizeof Ms/sizeof Ms[0]);
    int Mmax = 128;

    ork_npu *c = ork_npu_init();
    if(!c){ fprintf(stderr,"init FAIL (need /dev/dri + rknpu)\n"); return 1; }
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    printf("# ork-driver %s soc=%s cores=%d ncpu=%ld  M-sweep batched concurrent MoE\n",
           ork_npu_version(), ork_npu_soc(c), ork_npu_cores(c), ncpu);

    struct model models_lfm[]  = { {"LFM2.5 gate/up", 2048, 1792, 4}, {"LFM2.5 down", 1792, 2048, 4} };
    struct model models_qwen[] = { {"Qwen3.6 gate/up", 2048, 512, 8}, {"Qwen3.6 down", 512, 2048, 8} };
    struct model *ms; int nms;
    if(!strcmp(which,"qwen")){ ms=models_qwen; nms=2; } else { ms=models_lfm; nms=2; }

    int big_cores = 4;
    int big_lo = (int)ncpu - big_cores;   /* RK3588 big = 4..7 */

    for(int mi=0; mi<nms; mi++){
        int K = ms[mi].K, N = ms[mi].N, topk = ms[mi].topk;
        int conforming = (K%512==0 && K<=4096);
        printf("\n=== %s  K=%d N=%d  top-k=%d  (stream-eligible: %s) ===\n",
               ms[mi].name, K, N, topk, conforming?"yes":"NO (per-task run_i8 K-split fallback)");

        /* build topk distinct resident int8 experts */
        float **f32 = malloc(topk*sizeof(float*));
        int8_t **B8 = malloc(topk*sizeof(int8_t*));
        ork_w **w   = malloc(topk*sizeof(ork_w*));
        float **bsc = malloc(topk*sizeof(float*));
        int32_t **C = malloc(topk*sizeof(int32_t*));   /* each [Mmax,N] */
        for(int e=0;e<topk;e++){
            f32[e]=malloc((size_t)N*K*sizeof(float)); for(size_t i=0;i<(size_t)N*K;i++) f32[e][i]=rf();
            bsc[e]=malloc((size_t)N*sizeof(float));
            w[e]=ork_mm_pack_i8_f32(c,K,N,f32[e],bsc[e]);
            if(!w[e]){ printf("  pack FAIL e=%d\n",e); return 1; }
            B8[e]=malloc((size_t)N*K);
            for(int n=0;n<N;n++){ const float*wn=f32[e]+(size_t)n*K; float amax=0; for(int k=0;k<K;k++){float a=fabsf(wn[k]); if(a>amax)amax=a;}
                float sc=amax>0?amax/127.0f:1.0f;
                for(int k=0;k<K;k++){ int q=(int)lrintf(wn[k]/sc); if(q>127)q=127; if(q<-128)q=-128; B8[e][(size_t)n*K+k]=(int8_t)q; } }
            C[e]=malloc((size_t)Mmax*N*4);
        }
        int8_t *A = malloc((size_t)Mmax*K); for(size_t i=0;i<(size_t)Mmax*K;i++) A[i]=r8();   /* [Mmax,K] */
        int32_t *Cref = malloc((size_t)Mmax*N*4);
        float   *Aout = malloc((size_t)Mmax*K*sizeof(float)); for(size_t i=0;i<(size_t)Mmax*K;i++) Aout[i]=rf();

        /* ---- sanity: NPU stream output matches CPU for expert 0 at M=16 (or conforming->M=1 fallback) ---- */
        {
            int Ms_chk = conforming ? 16 : 8;
            ork_mm_task_i8 chk = { .w=w[0], .M=Ms_chk, .A=A, .C=C[0] };
            int crc = ork_mm_run_stream_i8(c,1,&chk);
            if(crc){ crc = ork_mm_run_i8(c,w[0],Ms_chk,A,C[0]); }
            if(crc==0){
                cpu_gemm_i8(Ms_chk,K,N,A,B8[0],Cref);
                long maxd=0; for(size_t i=0;i<(size_t)Ms_chk*N;i++){ long d=labs((long)C[0][i]-(long)Cref[i]); if(d>maxd)maxd=d; }
                printf("  [sanity] NPU vs CPU expert0 M=%d maxd=%ld (tol ~%d) %s\n", Ms_chk, maxd, K, maxd<=(long)K?"OK":"SUSPECT");
            } else printf("  [sanity] NPU rc=%d\n", crc);
        }

        struct cpu_pool pool; cpu_pool_start(&pool, big_cores, big_lo);
        double sm[REPS];

        printf("  %-4s | %-11s | %-12s | %-10s | %-9s | %-7s | %-8s | %s\n",
               "M","T_cpu_all","T_distrib","best_j","T_cross","cross%","speedup","scales");
        double prev_speed = 0;
        for(int xi=0; xi<nM; xi++){
            int M = Ms[xi];

            /* ---- T_crossing(M): quant fp32 act[M,K] -> int8 (per-row amax) + dequant/combine outputs[M,N] ---- */
            double t_cross;
            {
                int8_t *Aq = malloc((size_t)M*K); float *acc = malloc((size_t)M*N*sizeof(float));
                int32_t *Cq = C[0];   /* reuse a populated output buffer as the "to dequant" source */
                for(int it=0; it<WARM+REPS; it++){
                    double t = now_us();
                    for(int m=0;m<M;m++){ const float *ar=Aout+(size_t)m*K; int8_t *aq=Aq+(size_t)m*K;
                        float amax=0; for(int k=0;k<K;k++){ float a=fabsf(ar[k]); if(a>amax)amax=a; }
                        float sc=amax>0?amax/127.0f:1.0f, inv=1.0f/sc;
                        for(int k=0;k<K;k++){ int q=(int)lrintf(ar[k]*inv); if(q>127)q=127; if(q<-128)q=-128; aq[k]=(int8_t)q; }
                    }
                    for(size_t i=0;i<(size_t)M*N;i++) acc[i] += (float)Cq[i]*0.001f;
                    if(it>=WARM) sm[it-WARM]=now_us()-t;
                }
                t_cross = median(sm,REPS);
                free(Aq); free(acc);
            }

            /* ---- T_cpu_all(M): all topk experts on CPU pool ---- */
            for(int i=0;i<WARM;i++){ cpu_pool_dispatch(&pool,0,topk,M,A,B8,C,K,N); cpu_pool_wait(&pool); }
            for(int r=0;r<REPS;r++){ double t=now_us(); cpu_pool_dispatch(&pool,0,topk,M,A,B8,C,K,N); cpu_pool_wait(&pool); sm[r]=now_us()-t; }
            double t_cpu_all = median(sm,REPS);

            /* ---- T_distributed(M): sweep split j (NPU experts) || k-j (CPU), best ---- */
            int verbose = getenv("ORK_VERBOSE")!=NULL;
            double best_conc = 1e30; int best_j = -1;
            double jcurve[9];
            for(int j=0; j<=topk; j++){
                int kj = topk - j;
                ork_mm_task_i8 ntasks[8];
                for(int e=0;e<j;e++){ ntasks[e].w=w[e]; ntasks[e].M=M; ntasks[e].A=A; ntasks[e].C=C[e]; }
                /* overlap: NPU run_stream on dedicated thread || CPU pool, then join. + crossing added once. */
                for(int i=0;i<WARM;i++){
                    struct npu_arg na = {c,j,ntasks,0}; pthread_t th;
                    if(j>0) pthread_create(&th,0,npu_thread,&na);
                    if(kj>0){ cpu_pool_dispatch(&pool,j,topk,M,A,B8,C,K,N); cpu_pool_wait(&pool);}
                    if(j>0) pthread_join(th,0);
                }
                double tj[REPS];
                for(int r=0;r<REPS;r++){ double t=now_us();
                    struct npu_arg na = {c,j,ntasks,0}; pthread_t th;
                    if(j>0) pthread_create(&th,0,npu_thread,&na);
                    if(kj>0){ cpu_pool_dispatch(&pool,j,topk,M,A,B8,C,K,N); cpu_pool_wait(&pool);}
                    if(j>0) pthread_join(th,0);
                    tj[r]=now_us()-t; }
                double m = median(tj,REPS);
                /* the distributed path also pays the per-layer crossing once (quant in + dequant/combine out) */
                double tot = m + t_cross;
                jcurve[j] = tot;
                if(tot < best_conc){ best_conc = tot; best_j = j; }
            }
            if(verbose){ printf("    M=%d j-curve (NPU experts j, T=concurrent+cross us):", M);
                for(int j=0;j<=topk;j++) printf(" j%d=%.0f", j, jcurve[j]); printf("\n"); }
            double speed = t_cpu_all / best_conc;
            char scbuf[16];
            if(prev_speed>0) snprintf(scbuf,sizeof scbuf, speed>prev_speed?"up":"down");
            else snprintf(scbuf,sizeof scbuf,"-");
            prev_speed = speed;
            printf("  %-4d | %-11.1f | %-12.1f | %-10d | %-9.1f | %-6.1f | %-8.2fx| %s\n",
                   M, t_cpu_all, best_conc, best_j, t_cross, 100.0*t_cross/best_conc, speed, scbuf);
        }

        cpu_pool_stop(&pool);
        for(int e=0;e<topk;e++){ ork_mm_free(c,w[e]); free(f32[e]); free(B8[e]); free(bsc[e]); free(C[e]); }
        free(f32);free(B8);free(w);free(bsc);free(C);free(A);free(Cref);free(Aout);
    }

    ork_npu_free(c);
    return 0;
}
