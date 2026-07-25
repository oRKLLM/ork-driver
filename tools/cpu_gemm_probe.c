/* cpu_gemm_probe — the engine question for the ork_spine prefill hook.
 *
 * The prefill overlap win requires relocating/splitting matmul COMPUTE onto the idle CPU (prefill is
 * latency-bound: NPU ~0.7%, CPU ~22%). That only pays off if the CPU can do int8 GEMM at a meaningful fraction
 * of the NPU. This probe measures a NEON-dotprod int8 GEMM (C[M,N] int32 = A[M,K] i8 · B[K,N] i8) at prefill
 * dims and prints its throughput next to ork_mm_run_i8 (NPU) on the SAME shape, so the achievable column-split
 * ceiling (npu_rate + cpu_rate) is sized before any hot-path wiring. Correctness checked vs a scalar reference.
 *
 *   make cpu_gemm_probe && sudo env ORK_MM_TIMEOUT=3000 ./cpu_gemm_probe [M] [K] [N]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <arm_neon.h>
#include <pthread.h>

static uint32_t rng = 0x1234u;
static int8_t r8(void){ rng = rng*1664525u+1013904223u; return (int8_t)((int)((rng>>25)%255)-127); }
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

/* NEON int8 GEMM: C[M,N] = A[M,K] (row-major) x Bt[N,K] (B TRANSPOSED so each output col's K-vector is
 * contiguous — the layout a dot-product GEMM wants). dotprod (vdotq_s32) over K, 4 output cols at a time. */
static void cpu_gemm_i8(const int8_t *A, const int8_t *Bt, int M, int K, int N, int32_t *C){
    for (int m = 0; m < M; m++){
        const int8_t *a = A + (size_t)m*K;
        for (int n = 0; n < N; n++){
            const int8_t *b = Bt + (size_t)n*K;
            int32x4_t acc = vdupq_n_s32(0);
            int k = 0;
            for (; k <= K-16; k += 16) acc = vdotq_s32(acc, vld1q_s8(a+k), vld1q_s8(b+k));
            int32_t s = vaddvq_s32(acc);
            for (; k < K; k++) s += (int32_t)a[k]*(int32_t)b[k];
            C[(size_t)m*N + n] = s;
        }
    }
}

/* N-split multi-thread: each thread computes a contiguous column band [n0,n1) — the exact shape of a
 * CPU/NPU column-split, and how a 4-core NEON GEMM would be dispatched under ork_spine. */
struct gw { const int8_t *A,*Bt; int M,K,N,n0,n1; int32_t *C; };
static void *gworker(void *p){ struct gw *w=p;
    for (int m=0; m<w->M; m++){ const int8_t *a=w->A+(size_t)m*w->K;
        for (int n=w->n0; n<w->n1; n++){ const int8_t *b=w->Bt+(size_t)n*w->K;
            int32x4_t acc=vdupq_n_s32(0); int k=0;
            for (; k<=w->K-16; k+=16) acc=vdotq_s32(acc, vld1q_s8(a+k), vld1q_s8(b+k));
            int32_t s=vaddvq_s32(acc); for(; k<w->K; k++) s+=(int32_t)a[k]*(int32_t)b[k];
            w->C[(size_t)m*w->N + n]=s; } } return 0; }
static void cpu_gemm_i8_mt(const int8_t *A, const int8_t *Bt, int M,int K,int N, int32_t *C, int nt){
    if(nt<1)nt=1; pthread_t th[16]; struct gw w[16]; if(nt>16)nt=16;
    for(int t=0;t<nt;t++){ w[t]=(struct gw){A,Bt,M,K,N,(int)((long)N*t/nt),(int)((long)N*(t+1)/nt),C};
        if(t) pthread_create(&th[t],0,gworker,&w[t]); }
    gworker(&w[0]); for(int t=1;t<nt;t++) pthread_join(th[t],0);
}

int main(int argc, char**argv){
    int M = argc>1?atoi(argv[1]):228, K = argc>2?atoi(argv[2]):2048, N = argc>3?atoi(argv[3]):2048;
    int NT = argc>4?atoi(argv[4]):4;
    setvbuf(stdout,0,_IONBF,0);
    printf("cpu_gemm_probe: M=%d K=%d N=%d (int8 GEMM, %.2f GMAC)\n", M,K,N, (double)M*K*N/1e9);
    int8_t *A = malloc((size_t)M*K), *B = malloc((size_t)K*N), *Bt = malloc((size_t)N*K);
    int32_t *C = malloc((size_t)M*N*4), *Cref = malloc((size_t)M*N*4);
    for (size_t i=0;i<(size_t)M*K;i++) A[i]=r8();
    for (size_t i=0;i<(size_t)K*N;i++) B[i]=r8();
    for (int k=0;k<K;k++) for (int n=0;n<N;n++) Bt[(size_t)n*K+k]=B[(size_t)k*N+n];   /* transpose B -> Bt[N,K] */

    /* correctness vs scalar ref on a small corner */
    cpu_gemm_i8(A,Bt,M,K,N,C);
    long bad=0; for (int m=0;m<(M<4?M:4);m++) for (int n=0;n<(N<4?N:4);n++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[(size_t)m*K+k]*B[(size_t)k*N+n]; if(C[(size_t)m*N+n]!=s) bad++; }
    printf("  correctness (4x4 corner vs scalar): %s\n", bad?"MISMATCH":"OK");

    /* CPU GEMM timing (warm + timed) */
    cpu_gemm_i8(A,Bt,M,K,N,C);
    int it=5; double t0=now_us(); for(int i=0;i<it;i++) cpu_gemm_i8(A,Bt,M,K,N,C); double cpu_us=(now_us()-t0)/it;
    double gmac = (double)M*K*N/1e9;
    printf("  CPU  NEON-dotprod: %.2f ms/gemm  (%.1f GMAC/s, single-thread)\n", cpu_us/1e3, gmac/(cpu_us/1e6));

    /* multi-thread (N-split) — the real 4-core NEON number, and how a CPU column-split would run */
    cpu_gemm_i8_mt(A,Bt,M,K,N,C,NT);   /* warm */
    double m0=now_us(); for(int i=0;i<it;i++) cpu_gemm_i8_mt(A,Bt,M,K,N,C,NT); double mt_us=(now_us()-m0)/it;
    printf("  CPU  NEON-dotprod: %.2f ms/gemm  (%.1f GMAC/s, %d threads = %.2fx scaling)\n",
           mt_us/1e3, gmac/(mt_us/1e6), NT, cpu_us/mt_us);

    /* NPU same shape via ork_mm_run_i8 */
    ork_npu *c = ork_npu_init(); if(!c){ printf("  (no NPU — CPU-only run)\n"); return bad?1:0; }
    ork_w *W = ork_mm_pack_i8(c,K,N,B);
    if (W){ int32_t *Cn=ork_dma_alloc(c,(size_t)M*N*4); if(Cn){
        ork_mm_run_i8(c,W,M,A,Cn);   /* warm */
        double n0=now_us(); for(int i=0;i<it;i++) ork_mm_run_i8(c,W,M,A,Cn); double npu_us=(now_us()-n0)/it;
        printf("  NPU  ork_mm_run_i8: %.2f ms/gemm  (%.1f GMAC/s)\n", npu_us/1e3, gmac/(npu_us/1e6));
        double frac1 = (cpu_us>0)? (npu_us/cpu_us) : 0;   /* single-thread CPU fraction of NPU rate */
        double fracN = (mt_us>0)? (npu_us/mt_us) : 0;      /* NT-thread CPU fraction of NPU rate */
        printf("  => CPU 1-thread is %.0f%% of NPU (split ceiling ~%.2fx); %d-thread is %.0f%% of NPU (split ceiling ~%.2fx over NPU-alone)\n",
               100.0*frac1, 1.0+frac1, NT, 100.0*fracN, 1.0+fracN);
        ork_dma_free(c,Cn); }
        ork_mm_free(c,W); }
    ork_npu_free(c);
    free(A);free(B);free(Bt);free(C);free(Cref);
    return bad?1:0;
}
