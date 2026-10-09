/* tiny_e2e_mm.h — the one matmul entry point of the tiny end-to-end trainer (train_tiny_e2e.c).
 *
 * Every matmul in the model, forward and backward, linear layers, attention and the GDN chunk, goes
 * through mm_nn / mm_nt / mm_tn below. Each call names a CLASS (linear, attention, GDN chunk), and each
 * class is routed to one of three back ends at run time:
 *
 *   MM_CPU   fp32 operands, fp32 accumulate, on the CPU          -- the build that must match MLX fp32
 *   MM_EMU   operands rounded to fp16 exactly as the NPU path rounds them, fp32 accumulate, on the CPU
 *   MM_NPU   ork_f16_mm_repack + ork_f16_mm_run: fp16 in, fp32 out, on the NPU
 *
 * MM_EMU exists to JUSTIFY the NPU bound instead of inventing one: it applies the same operand
 * quantisation the NPU sees (same scaling, same cast), so MLX-vs-EMU is the error fp16 operands alone
 * should cost, and NPU-vs-EMU isolates whatever the hardware adds on top (accumulation order, any
 * flush-to-zero). An NPU result far from EMU is an NPU/harness problem; an EMU result far from MLX is
 * the fp16 operand question itself.
 *
 * OPERAND SCALING (g_mm_scale). With 0 each fp32 operand is cast to fp16 as is -- what the scoping
 * probes did. With 1 each operand is first multiplied by a power of two chosen so its max |x| is 2^10,
 * and the fp32 output is divided by the product of the two scales. Power-of-two scaling is exact, so
 * the only effect is WHERE in fp16's range the operand sits: gradients in this model are ~1e-5..1e-7
 * per element, which is in or below fp16's subnormal range (normals stop at 6.1e-5), so unscaled
 * gradient operands lose most of their bits. This is per-operand dynamic loss scaling, applied at the
 * one place precision is lost. Whether it is NEEDED is the measurement (scope report section 6 left
 * loss scaling explicitly untested).
 *
 * Pack rules: ork_f16_mm_pack/scratch refuse K%32 or N%16. A shape that cannot go to the NPU falls back
 * to the CPU (fp32 in MM_NPU class mode) and is COUNTED, so the report can say which matmuls missed.
 * Weights for the NPU are pooled scratch handles keyed by (K,N): allocated once, ork_f16_mm_repack'd
 * on every call, never packed-and-freed in the loop.
 */
#ifndef TINY_E2E_MM_H
#define TINY_E2E_MM_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef WITH_NPU
#include <ork_npu.h>
typedef ork_f16 mm_f16;
#else
typedef _Float16 mm_f16;
#endif

enum { MM_CPU = 0, MM_EMU = 1, MM_NPU = 2 };
enum { MMC_LIN = 0, MMC_ATT = 1, MMC_GDN = 2, MMC_N = 3 };
static const char *MMC_NAME[MMC_N] = { "linear", "attention", "gdn-chunk" };

static int    g_mm_mode[MMC_N] = { MM_CPU, MM_CPU, MM_CPU };
static int    g_mm_scale = 0;
static int    g_mm_class = MMC_LIN;            /* set by the caller before a group of matmuls */
static double g_mm_sec[MMC_N], g_mm_flop[MMC_N], g_mm_npu_sec[MMC_N];   /* npu = repack+run only */
static long   g_mm_calls[MMC_N], g_mm_fallback[MMC_N], g_mm_slowpath[MMC_N];

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }

/* ---------------------------------------------------------------- CPU fp32 kernels */
/* Row-parallel with OpenMP when built with -fopenmp; each C row is produced by one thread in a fixed
 * order, so the result does not depend on the thread count. */
static void cpu_nn(int M,int K,int N,const float*A,const float*B,float*C){
    #pragma omp parallel for schedule(static)
    for(int i=0;i<M;i++){
        float *c=C+(size_t)i*N; memset(c,0,(size_t)N*4);
        const float *a=A+(size_t)i*K;
        for(int k=0;k<K;k++){ float av=a[k]; if(av==0.f) continue;
            const float *b=B+(size_t)k*N;
            for(int j=0;j<N;j++) c[j]+=av*b[j]; }
    }
}
static int g_cpu_fast = 0;   /* r102: cpu_nt transposes B and uses the vectorisable nn kernel (different
                               * accumulation order from r101's dot-product loop, so OFF by default) */
static void cpu_nn(int M,int K,int N,const float*A,const float*B,float*C);
static void cpu_nt(int M,int K,int N,const float*A,const float*B,float*C){   /* B is [N,K] */
    if(g_cpu_fast){
        float *Bt=malloc((size_t)K*N*4); if(!Bt){fprintf(stderr,"oom\n");exit(3);}
        #pragma omp parallel for schedule(static)
        for(int k=0;k<K;k++) for(int j=0;j<N;j++) Bt[(size_t)k*N+j]=B[(size_t)j*K+k];
        cpu_nn(M,K,N,A,Bt,C); free(Bt); return;
    }
    #pragma omp parallel for schedule(static)
    for(int i=0;i<M;i++){
        const float *a=A+(size_t)i*K;
        for(int j=0;j<N;j++){ const float *b=B+(size_t)j*K; float s=0;
            for(int k=0;k<K;k++) s+=a[k]*b[k];
            C[(size_t)i*N+j]=s; }
    }
}
static void cpu_tn(int M,int K,int N,const float*A,const float*B,float*C){   /* A is [K,M] */
    #pragma omp parallel for schedule(static)
    for(int i=0;i<M;i++){
        float *c=C+(size_t)i*N; memset(c,0,(size_t)N*4);
        for(int k=0;k<K;k++){ float av=A[(size_t)k*M+i]; if(av==0.f) continue;
            const float *b=B+(size_t)k*N;
            for(int j=0;j<N;j++) c[j]+=av*b[j]; }
    }
}

/* ---------------------------------------------------------------- fp16 operand preparation */
static float mm_pow2_scale(const float *x,size_t n){
    if(!g_mm_scale) return 1.0f;
    float mx=0; for(size_t i=0;i<n;i++){ float a=fabsf(x[i]); if(a>mx) mx=a; }
    if(!(mx>0) || !isfinite(mx)) return 1.0f;
    int e; frexpf(mx,&e);                      /* mx = f * 2^e, f in [0.5,1) */
    return ldexpf(1.0f, 10-e);                 /* max|x| * s in [2^9, 2^10) */
}
/* STOCHASTIC ROUNDING (g_mm_sr). Round-to-nearest makes the fp16 operand error a fixed function of
 * the operands: re-running with a 1-ulp fp32 perturbation re-rounds almost nothing (an fp16 ulp is
 * 2^13 fp32 ulps), so averaging such runs cannot shrink it. Stochastic rounding picks the upper
 * neighbour with probability (v - lo)/(hi - lo): unbiased, and independent across seeds, so the mean
 * of N gradients carries 1/sqrt(N) of the operand noise while a WRONG gradient term does not shrink.
 * That is what lets a positive control smaller than one draw's fp16 noise be detected at all. Used
 * only by the gradient check (--draws); training runs round to nearest. Host-side, before the pack. */
static int g_mm_sr = 0; static unsigned long long g_sr_state = 0x9E3779B97F4A7C15ull;
static inline unsigned sr_u32(void){ unsigned long long x=g_sr_state; x^=x<<13; x^=x>>7; x^=x<<17; g_sr_state=x; return (unsigned)(x>>32); }
static inline mm_f16 sr_cast(float v){
    mm_f16 h=(mm_f16)v; float hf=(float)h, d=v-hf;
    if(d==0.f || !isfinite(hf)) return h;
    unsigned short b, nb; memcpy(&b,&h,2);
    int away = (d>0) == !(b&0x8000);                 /* moving away from zero? */
    if(away) nb=(unsigned short)(b+1);
    else if((b&0x7fff)==0) nb=(unsigned short)((b^0x8000)+1);   /* +-0 toward the other sign */
    else nb=(unsigned short)(b-1);
    mm_f16 n; memcpy(&n,&nb,2); float nf=(float)n;
    if(!isfinite(nf)) return h;
    float p=d/(nf-hf);                               /* in (0, 0.5] */
    return (sr_u32()*(1.0f/4294967296.0f) < p) ? n : h;
}
/* dst[r,c] = f16(src[r,c]*s)  (row-major, same layout) */
static void to_f16(mm_f16 *dst,const float *src,size_t n,float s){
    if(g_mm_sr){ for(size_t i=0;i<n;i++) dst[i]=sr_cast(src[i]*s); return; }
    for(size_t i=0;i<n;i++) dst[i]=(mm_f16)(src[i]*s);
}
/* dst[c,r] = f16(src[r,c]*s): transpose + narrow */
static float *g_tmpf; static size_t g_tmpf_n;
static float *tmpf(size_t n){ if(n>g_tmpf_n){ free(g_tmpf); g_tmpf=malloc(n*4); g_tmpf_n=n; if(!g_tmpf){fprintf(stderr,"oom\n");exit(3);} } return g_tmpf; }
static mm_f16 *g_sr16; static size_t g_sr16_n;
static void to_f16_t(mm_f16 *dst,const float *src,int rows,int cols,float s){
    if(g_mm_sr){
        size_t n=(size_t)rows*cols;
        if(n>g_sr16_n){ free(g_sr16); g_sr16=malloc(n*2); g_sr16_n=n; if(!g_sr16){fprintf(stderr,"oom\n");exit(3);} }
        for(size_t i=0;i<n;i++) g_sr16[i]=sr_cast(src[i]*s);
#ifdef WITH_NPU
        if(ork_f16_transpose(dst,g_sr16,rows,cols)){ fprintf(stderr,"transpose failed\n"); exit(3); }
#else
        for(int r=0;r<rows;r++) for(int c=0;c<cols;c++) dst[(size_t)c*rows+r]=g_sr16[(size_t)r*cols+c];
#endif
        return;
    }
#ifdef WITH_NPU
    const float *p=src;
    if(s!=1.0f){ float*t=tmpf((size_t)rows*cols); for(size_t i=0;i<(size_t)rows*cols;i++) t[i]=src[i]*s; p=t; }
    if(ork_f32_transpose_to_f16(dst,p,rows,cols)){ fprintf(stderr,"transpose failed\n"); exit(3); }
#else
    for(int r=0;r<rows;r++) for(int c=0;c<cols;c++) dst[(size_t)c*rows+r]=(mm_f16)(src[(size_t)r*cols+c]*s);
#endif
}
static mm_f16 *g_ha,*g_hb; static size_t g_ha_n,g_hb_n;
static mm_f16 *buf16(mm_f16 **p,size_t *cap,size_t n){ if(n>*cap){ free(*p); *p=malloc(n*2); *cap=n; if(!*p){fprintf(stderr,"oom\n");exit(3);} } return *p; }

/* EMU kernel: C = A16 * B16 with A16 [M,K], B16 [K,N] already fp16, fp32 accumulate. */
static void emu_nn16(int M,int K,int N,const mm_f16*A,const mm_f16*B,float*C){
    float *Bf=malloc((size_t)K*N*4); if(!Bf){fprintf(stderr,"oom\n");exit(3);}
    for(size_t i=0;i<(size_t)K*N;i++) Bf[i]=(float)B[i];
    #pragma omp parallel for schedule(static)
    for(int i=0;i<M;i++){
        float *c=C+(size_t)i*N; memset(c,0,(size_t)N*4);
        for(int k=0;k<K;k++){ float av=(float)A[(size_t)i*K+k]; if(av==0.f) continue;
            const float *b=Bf+(size_t)k*N;
            for(int j=0;j<N;j++) c[j]+=av*b[j]; }
    }
    free(Bf);
}

#ifdef WITH_NPU
static ork_npu *g_npu;
typedef struct { int K,N; ork_w *w; } mm_slot;
static mm_slot g_slots[128]; static int g_nslots;
static ork_w *mm_scratch(int K,int N){
    for(int i=0;i<g_nslots;i++) if(g_slots[i].K==K && g_slots[i].N==N) return g_slots[i].w;
    if(g_nslots==128){ fprintf(stderr,"scratch pool full\n"); exit(3); }
    ork_w *w=ork_f16_mm_scratch(g_npu,K,N);
    if(!w){ fprintf(stderr,"ork_f16_mm_scratch(%d,%d) failed\n",K,N); exit(3); }
    g_slots[g_nslots].K=K; g_slots[g_nslots].N=N; g_slots[g_nslots].w=w; g_nslots++;
    return w;
}
static double mm_pool_bytes(int *nshapes){   /* resident fp16 scratch = what this harness pins in IOVA */
    double b=0; for(int i=0;i<g_nslots;i++) b+=2.0*g_slots[i].K*g_slots[i].N; if(nshapes)*nshapes=g_nslots; return b;
}
static void mm_npu_free_all(void){
    for(int i=0;i<g_nslots;i++) ork_mm_free(g_npu,g_slots[i].w);
    g_nslots=0;
}
#endif

static int pow2_fast(int K){ return K>=128 && K<2048 && (K&(K-1))==0; }

/* The fp16 paths share one operand preparation: A16 [M,K], B16 [K,N], scales sa, sb. */
static void mm_f16_core(int M,int K,int N,const mm_f16*A16,const mm_f16*B16,float sa,float sb,float*C,int mode){
    if(mode==MM_NPU){
#ifdef WITH_NPU
        double tn=now_s();
        ork_w *w=mm_scratch(K,N);
        if(ork_f16_mm_repack(g_npu,w,K,N,B16)<0){ fprintf(stderr,"repack(%d,%d) failed\n",K,N); exit(4); }
        int rc=ork_f16_mm_run(g_npu,w,M,A16,C);
        if(rc){ fprintf(stderr,"ork_f16_mm_run M=%d K=%d N=%d rc=%d -- STOPPING (no retry)\n",M,K,N,rc); exit(4); }
        g_mm_npu_sec[g_mm_class]+=now_s()-tn;
#else
        fprintf(stderr,"MM_NPU requested in a CPU-only build\n"); exit(3);
#endif
    } else emu_nn16(M,K,N,A16,B16,C);
    float inv=1.0f/(sa*sb);
    if(inv!=1.0f) for(size_t i=0;i<(size_t)M*N;i++) C[i]*=inv;
}

/* which: 0 = nn (A[M,K] B[K,N]), 1 = nt (B is [N,K]), 2 = tn (A is [K,M]). */
static void mm_any(int which,int M,int K,int N,const float*A,const float*B,float*C){
    int cls=g_mm_class, mode=g_mm_mode[cls];
    double t0=now_s();
    g_mm_calls[cls]++; g_mm_flop[cls]+=2.0*M*K*N;
    if(mode!=MM_CPU && (K%32 || N%16)){ g_mm_fallback[cls]++; mode=MM_CPU; }
    if(mode==MM_CPU){
        if(which==0) cpu_nn(M,K,N,A,B,C); else if(which==1) cpu_nt(M,K,N,A,B,C); else cpu_tn(M,K,N,A,B,C);
    } else {
        if(!pow2_fast(K)) g_mm_slowpath[cls]++;
        float sa=mm_pow2_scale(A,(size_t)M*K), sb=mm_pow2_scale(B,(size_t)K*N);
        mm_f16 *A16=buf16(&g_ha,&g_ha_n,(size_t)M*K), *B16=buf16(&g_hb,&g_hb_n,(size_t)K*N);
        if(which==2) to_f16_t(A16,A,K,M,sa); else to_f16(A16,A,(size_t)M*K,sa);
        if(which==1) to_f16_t(B16,B,N,K,sb); else to_f16(B16,B,(size_t)K*N,sb);
        mm_f16_core(M,K,N,A16,B16,sa,sb,C,mode);
    }
    g_mm_sec[cls]+=now_s()-t0;
}
static inline void mm_nn(int M,int K,int N,const float*A,const float*B,float*C){ mm_any(0,M,K,N,A,B,C); }
static inline void mm_nt(int M,int K,int N,const float*A,const float*B,float*C){ mm_any(1,M,K,N,A,B,C); }
static inline void mm_tn(int M,int K,int N,const float*A,const float*B,float*C){ mm_any(2,M,K,N,A,B,C); }

static void mm_stats_reset(void){
    memset(g_mm_sec,0,sizeof g_mm_sec); memset(g_mm_npu_sec,0,sizeof g_mm_npu_sec); memset(g_mm_flop,0,sizeof g_mm_flop);
    memset(g_mm_calls,0,sizeof g_mm_calls); memset(g_mm_fallback,0,sizeof g_mm_fallback);
    memset(g_mm_slowpath,0,sizeof g_mm_slowpath);
}
#endif /* TINY_E2E_MM_H */
