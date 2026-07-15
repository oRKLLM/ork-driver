/* cpu_i4_vs_i8 — CPU-only: does int4 beat int8 for a memory-bound decode GEMV? (the CPU side of the
 * "CPU=int4 / NPU=int8" split). int8 reads K bytes/output-row; int4 reads K/2 packed nibble bytes + unpacks
 * (sign-extend nibble->int8) in NEON registers, then sdot. Big weight (68MB int8 / 34MB int4) so it's
 * DRAM-bound, not cached. 4 threads on the A76 big cores. Reports GB/s + speedup (>1 => int4 faster on CPU).
 *   make cpu_i4_vs_i8 && ./cpu_i4_vs_i8 [iters] [threads]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <arm_neon.h>
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

static int gK,gN; static int8_t *gA,*gB8,*gB4; static int32_t *gC; static volatile int32_t sink;

/* int8: C[n] = sum_k A[k]*B8[n*K+k]  (reads K bytes) */
static int32_t dot_i8(const int8_t*a,const int8_t*b,int K){
    int32x4_t ac=vdupq_n_s32(0); int k=0;
    for(;k+16<=K;k+=16) ac=vdotq_s32(ac,vld1q_s8(a+k),vld1q_s8(b+k));
    int32_t s=vaddvq_s32(ac); for(;k<K;k++)s+=(int32_t)a[k]*b[k]; return s;
}
/* int4: B4[n*(K/2)..] packed 2 nibbles/byte; unpack sign-extended nibble->int8, sdot with A (reads K/2 bytes) */
static int32_t dot_i4(const int8_t*a,const int8_t*b4,int K){
    int32x4_t ac=vdupq_n_s32(0); int k=0,kb=0;
    for(;k+32<=K;k+=32,kb+=16){
        int8x16_t p=vld1q_s8(b4+kb);
        int8x16_t lo=vshrq_n_s8(vshlq_n_s8(p,4),4);   /* low nibble  -> sign-extended int8 */
        int8x16_t hi=vshrq_n_s8(p,4);                 /* high nibble -> sign-extended int8 */
        ac=vdotq_s32(ac, lo, vld1q_s8(a+k));
        ac=vdotq_s32(ac, hi, vld1q_s8(a+k+16));
    }
    return vaddvq_s32(ac);
}
typedef struct { int lo,hi,core,mode; } job;
static void* wk(void*p){ job*j=p; cpu_set_t s;CPU_ZERO(&s);CPU_SET(j->core,&s);pthread_setaffinity_np(pthread_self(),sizeof s,&s);
    int32_t acc=0;
    if(j->mode==8) for(int n=j->lo;n<j->hi;n++) gC[n]=dot_i8(gA,gB8+(size_t)n*gK,gK);
    else           for(int n=j->lo;n<j->hi;n++) gC[n]=dot_i4(gA,gB4+(size_t)n*(gK/2),gK);
    (void)acc; return NULL; }
static double run(int mode,int nt,int c0){
    pthread_t th[8]; job jb[8]; int per=(gN+nt-1)/nt;
    for(int t=0;t<nt;t++){ jb[t]=(job){t*per,(t+1)*per<gN?(t+1)*per:gN,c0+t,mode}; pthread_create(&th[t],0,wk,&jb[t]); }
    for(int t=0;t<nt;t++) pthread_join(th[t],0);
    sink+=gC[0]; return 0;
}
int main(int argc,char**argv){
    int iters=argc>1?atoi(argv[1]):20, nt=argc>2?atoi(argv[2]):4, c0=4;
    int K=3584, N=18944;   /* 7B FFN gate: 68MB int8 / 34MB int4, DRAM-bound */
    gK=K; gN=N;
    gA=malloc(K); memset(gA,1,K);
    gB8=malloc((size_t)N*K); memset(gB8,1,(size_t)N*K);
    gB4=malloc((size_t)N*K/2); memset(gB4,0x11,(size_t)N*K/2);   /* nibbles = 1 */
    gC=malloc((size_t)N*4);
    printf("cpu_i4_vs_i8: M=1 K=%d N=%d  int8=%.0fMB int4=%.0fMB  %d threads @cpu%d\n",
           K,N,(double)N*K/1e6,(double)N*K/2/1e6,nt,c0);
    run(8,nt,c0); run(0,nt,c0);   /* warm both */
    double t0=now_us(); for(int i=0;i<iters;i++) run(8,nt,c0); double i8=(now_us()-t0)/iters;
    t0=now_us(); for(int i=0;i<iters;i++) run(0,nt,c0); double i4=(now_us()-t0)/iters;
    printf("  int8: %8.1f us  %6.1f GB/s\n", i8, (double)N*K/i8/1e3);
    printf("  int4: %8.1f us  %6.1f GB/s (of the 34MB it reads)\n", i4, (double)N*K/2/i4/1e3);
    printf("  ★ int4/int8 speedup: %.2fx  (>1 => CPU int4 faster — confirms CPU=int4/NPU=int8 split)\n", i8/i4);
    return 0;
}
