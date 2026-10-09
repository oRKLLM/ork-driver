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
 * to the CPU (fp32 in MM_NPU class mode) and is COUNTED, so the report can say which matmuls missed
 * AND what they cost (g_mm_fb_sec) — the answer to "is it worth padding K to 32" is a number, not an
 * argument. Weights for the NPU are pooled scratch handles keyed by (K,N): allocated once,
 * ork_f16_mm_repack'd on every call, never packed-and-freed in the loop.
 *
 * HOST WORK IS NOT FREE HERE. Preparing the operands (absmax scan, fp32->fp16 cast, transpose, output
 * rescale) touches gigabytes a step on the 248,320-wide head shapes, so every one of those passes is
 * NEON + OpenMP with a scalar fallback, and the per-step report splits them out (absmax / cast /
 * transpose / rescale, and the NPU bucket into weight re-tile vs submit). The CPU matmul kernels are
 * register-blocked too, because the GDN chunk path calls them ~10^5 times a step.
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
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MM_NEON 1
#endif
#ifdef _OPENMP
#include <omp.h>
#define MM_OUTER (!omp_in_parallel())   /* don't nest: the GDN head loop is itself parallel */
#else
#define MM_OUTER 1
#endif

enum { MM_CPU = 0, MM_EMU = 1, MM_NPU = 2 };
enum { MMC_LIN = 0, MMC_ATT = 1, MMC_GDN = 2, MMC_N = 3 };
static const char *MMC_NAME[MMC_N] = { "linear", "attention", "gdn-chunk" };

static int    g_mm_mode[MMC_N] = { MM_CPU, MM_CPU, MM_CPU };
static int    g_mm_scale = 0;
static int    g_mm_class = MMC_LIN;            /* set by the caller before a group of matmuls */
static double g_mm_sec[MMC_N], g_mm_flop[MMC_N], g_mm_npu_sec[MMC_N];   /* npu = repack+run only */
static long   g_mm_calls[MMC_N], g_mm_fallback[MMC_N], g_mm_slowpath[MMC_N];
/* finer split of the two big buckets, so a change can be attributed: prep = absmax scan / cast /
 * transpose; npu = weight re-tile (ork_f16_mm_repack) vs the submit itself (ork_f16_mm_run). */
static double g_mm_scan_sec[MMC_N], g_mm_cast_sec[MMC_N], g_mm_tr_sec[MMC_N];
static double g_mm_repack_sec[MMC_N], g_mm_run_sec[MMC_N];
static double g_mm_fb_sec[MMC_N];   /* time in CPU FALLBACKS (shape the pack refuses) — is padding K worth it? */

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }

/* ---------------------------------------------------------------- CPU fp32 kernels */
/* Row-parallel with OpenMP; each C row is produced by one thread, so the result does not depend on
 * the thread count. nn/tn keep the k-ascending accumulation order per output element (register
 * blocking only changes which registers hold the running sums), so they differ from the scalar loop
 * by at most fused-vs-separate multiply-add. nt sums four k-partials at the end — a reassociation,
 * like the pre-existing transpose route. Blocking is 4 rows x 16 columns for nn/tn (16 accumulators
 * + 4 B vectors, and one B load feeds four FMAs) and 4x4 k-vectorised dots for nt. */
#ifdef MM_NEON
/* C[i0..i0+4, j0..j0+16) = sum_k AVAL(r) * B[k, j0..]; AVAL supplies the caller's A addressing. */
#define MM_KERN_4x16(AVAL)                                                                       \
    do {                                                                                         \
        float32x4_t c00=vdupq_n_f32(0),c01=vdupq_n_f32(0),c02=vdupq_n_f32(0),c03=vdupq_n_f32(0); \
        float32x4_t c10=vdupq_n_f32(0),c11=vdupq_n_f32(0),c12=vdupq_n_f32(0),c13=vdupq_n_f32(0); \
        float32x4_t c20=vdupq_n_f32(0),c21=vdupq_n_f32(0),c22=vdupq_n_f32(0),c23=vdupq_n_f32(0); \
        float32x4_t c30=vdupq_n_f32(0),c31=vdupq_n_f32(0),c32=vdupq_n_f32(0),c33=vdupq_n_f32(0); \
        for(int k=0;k<K;k++){                                                                    \
            const float *bp=B+(size_t)k*N+j0;                                                    \
            float32x4_t b0=vld1q_f32(bp),b1=vld1q_f32(bp+4),b2=vld1q_f32(bp+8),b3=vld1q_f32(bp+12);\
            float av0=AVAL(0),av1=AVAL(1),av2=AVAL(2),av3=AVAL(3);                               \
            c00=vfmaq_n_f32(c00,b0,av0); c01=vfmaq_n_f32(c01,b1,av0);                            \
            c02=vfmaq_n_f32(c02,b2,av0); c03=vfmaq_n_f32(c03,b3,av0);                            \
            c10=vfmaq_n_f32(c10,b0,av1); c11=vfmaq_n_f32(c11,b1,av1);                            \
            c12=vfmaq_n_f32(c12,b2,av1); c13=vfmaq_n_f32(c13,b3,av1);                            \
            c20=vfmaq_n_f32(c20,b0,av2); c21=vfmaq_n_f32(c21,b1,av2);                            \
            c22=vfmaq_n_f32(c22,b2,av2); c23=vfmaq_n_f32(c23,b3,av2);                            \
            c30=vfmaq_n_f32(c30,b0,av3); c31=vfmaq_n_f32(c31,b1,av3);                            \
            c32=vfmaq_n_f32(c32,b2,av3); c33=vfmaq_n_f32(c33,b3,av3);                            \
        }                                                                                        \
        float *q0=C+(size_t)(i0+0)*N+j0,*q1=C+(size_t)(i0+1)*N+j0;                               \
        float *q2=C+(size_t)(i0+2)*N+j0,*q3=C+(size_t)(i0+3)*N+j0;                               \
        vst1q_f32(q0,c00);vst1q_f32(q0+4,c01);vst1q_f32(q0+8,c02);vst1q_f32(q0+12,c03);          \
        vst1q_f32(q1,c10);vst1q_f32(q1+4,c11);vst1q_f32(q1+8,c12);vst1q_f32(q1+12,c13);          \
        vst1q_f32(q2,c20);vst1q_f32(q2+4,c21);vst1q_f32(q2+8,c22);vst1q_f32(q2+12,c23);          \
        vst1q_f32(q3,c30);vst1q_f32(q3+4,c31);vst1q_f32(q3+8,c32);vst1q_f32(q3+12,c33);          \
    } while(0)
#endif

static void cpu_nn(int M,int K,int N,const float*A,const float*B,float*C){      /* C = A[M,K] B[K,N] */
#ifdef MM_NEON
    int MB=M&~3, NB=N&~15;
    #pragma omp parallel for schedule(static) if(MM_OUTER && M>=8)
    for(int i0=0;i0<MB;i0+=4){
        const float *a0=A+(size_t)(i0+0)*K,*a1=A+(size_t)(i0+1)*K,*a2=A+(size_t)(i0+2)*K,*a3=A+(size_t)(i0+3)*K;
        (void)a0;(void)a1;(void)a2;(void)a3;   /* unused when N < 16 (whole row runs the scalar tail) */
        for(int j0=0;j0<NB;j0+=16){
#define MM_A_NN(r) (a##r[k])
            MM_KERN_4x16(MM_A_NN);
#undef MM_A_NN
        }
        for(int r=0;r<4;r++){ float *c=C+(size_t)(i0+r)*N; const float *a=A+(size_t)(i0+r)*K;
            for(int j=NB;j<N;j++) c[j]=0.f;
            for(int k=0;k<K;k++){ float av=a[k]; if(av==0.f) continue; const float *b=B+(size_t)k*N;
                for(int j=NB;j<N;j++) c[j]+=av*b[j]; } }
    }
    for(int i=MB;i<M;i++){
        float *c=C+(size_t)i*N; memset(c,0,(size_t)N*4); const float *a=A+(size_t)i*K;
        for(int k=0;k<K;k++){ float av=a[k]; if(av==0.f) continue; const float *b=B+(size_t)k*N;
            for(int j=0;j<N;j++) c[j]+=av*b[j]; }
    }
#else
    #pragma omp parallel for schedule(static) if(MM_OUTER)
    for(int i=0;i<M;i++){
        float *c=C+(size_t)i*N; memset(c,0,(size_t)N*4);
        const float *a=A+(size_t)i*K;
        for(int k=0;k<K;k++){ float av=a[k]; if(av==0.f) continue;
            const float *b=B+(size_t)k*N;
            for(int j=0;j<N;j++) c[j]+=av*b[j]; }
    }
#endif
}
/* --cpufast. Three accumulation orders exist for nt and all three are reachable, because the
 * gradient table is sensitive enough that "which order did this run use" has to be answerable:
 *   g_cpu_fast=0            r101's scalar dot product, the reference order
 *   g_cpu_fast=1            the NEON 4x4 dot below — four k-partials summed at the end
 *   ORK_NT_TRANSPOSE=1      r102's route: pre-transpose B and reuse the nn kernel (nn's order)
 * r105 changed what =1 MEANS (it used to select the transpose route) because the direct kernel is
 * now the fast one AND allocates nothing, which matters: the GDN chunk path calls nt ~10^5 times a
 * step and r102's route malloc'd a scratch B on every one. */
static int g_cpu_fast = 0;
static void cpu_nt(int M,int K,int N,const float*A,const float*B,float*C){   /* B is [N,K] */
    static int viat=-1; if(viat<0) viat=getenv("ORK_NT_TRANSPOSE")?1:0;
    if(viat){
        float *Bt=malloc((size_t)K*N*4); if(!Bt){fprintf(stderr,"oom\n");exit(3);}
        #pragma omp parallel for schedule(static) if(MM_OUTER)
        for(int k=0;k<K;k++) for(int j=0;j<N;j++) Bt[(size_t)k*N+j]=B[(size_t)j*K+k];
        cpu_nn(M,K,N,A,Bt,C); free(Bt); return;
    }
#ifdef MM_NEON
    if(g_cpu_fast)
    { int KB=K&~3, MB=M&~3, NB=N&~3;
    #pragma omp parallel for schedule(static) if(MM_OUTER && M>=8)
    for(int i0=0;i0<MB;i0+=4){
        for(int j0=0;j0<NB;j0+=4){
            float32x4_t s[4][4];
            for(int r=0;r<4;r++) for(int c=0;c<4;c++) s[r][c]=vdupq_n_f32(0);
            for(int k=0;k<KB;k+=4){
                float32x4_t a0=vld1q_f32(A+(size_t)(i0+0)*K+k),a1=vld1q_f32(A+(size_t)(i0+1)*K+k),
                            a2=vld1q_f32(A+(size_t)(i0+2)*K+k),a3=vld1q_f32(A+(size_t)(i0+3)*K+k);
                float32x4_t b0=vld1q_f32(B+(size_t)(j0+0)*K+k),b1=vld1q_f32(B+(size_t)(j0+1)*K+k),
                            b2=vld1q_f32(B+(size_t)(j0+2)*K+k),b3=vld1q_f32(B+(size_t)(j0+3)*K+k);
                s[0][0]=vfmaq_f32(s[0][0],a0,b0); s[0][1]=vfmaq_f32(s[0][1],a0,b1);
                s[0][2]=vfmaq_f32(s[0][2],a0,b2); s[0][3]=vfmaq_f32(s[0][3],a0,b3);
                s[1][0]=vfmaq_f32(s[1][0],a1,b0); s[1][1]=vfmaq_f32(s[1][1],a1,b1);
                s[1][2]=vfmaq_f32(s[1][2],a1,b2); s[1][3]=vfmaq_f32(s[1][3],a1,b3);
                s[2][0]=vfmaq_f32(s[2][0],a2,b0); s[2][1]=vfmaq_f32(s[2][1],a2,b1);
                s[2][2]=vfmaq_f32(s[2][2],a2,b2); s[2][3]=vfmaq_f32(s[2][3],a2,b3);
                s[3][0]=vfmaq_f32(s[3][0],a3,b0); s[3][1]=vfmaq_f32(s[3][1],a3,b1);
                s[3][2]=vfmaq_f32(s[3][2],a3,b2); s[3][3]=vfmaq_f32(s[3][3],a3,b3);
            }
            for(int r=0;r<4;r++) for(int c=0;c<4;c++){
                float v=vaddvq_f32(s[r][c]);
                for(int k=KB;k<K;k++) v+=A[(size_t)(i0+r)*K+k]*B[(size_t)(j0+c)*K+k];
                C[(size_t)(i0+r)*N+j0+c]=v; }
        }
        for(int r=0;r<4;r++) for(int j=NB;j<N;j++){ const float*a=A+(size_t)(i0+r)*K,*b=B+(size_t)j*K;
            float v=0; for(int k=0;k<K;k++) v+=a[k]*b[k]; C[(size_t)(i0+r)*N+j]=v; }
    }
    for(int i=MB;i<M;i++) for(int j=0;j<N;j++){ const float*a=A+(size_t)i*K,*b=B+(size_t)j*K;
        float v=0; for(int k=0;k<K;k++) v+=a[k]*b[k]; C[(size_t)i*N+j]=v; }
      return; }
#endif
    #pragma omp parallel for schedule(static) if(MM_OUTER)            /* r101 scalar dot, the reference order */
    for(int i=0;i<M;i++){
        const float *a=A+(size_t)i*K;
        for(int j=0;j<N;j++){ const float *b=B+(size_t)j*K; float s=0;
            for(int k=0;k<K;k++) s+=a[k]*b[k];
            C[(size_t)i*N+j]=s; }
    }
}
static void cpu_tn(int M,int K,int N,const float*A,const float*B,float*C){   /* C = A^T B, A is [K,M] */
#ifdef MM_NEON
    int MB=M&~3, NB=N&~15;
    #pragma omp parallel for schedule(static) if(MM_OUTER && M>=8)
    for(int i0=0;i0<MB;i0+=4){
        for(int j0=0;j0<NB;j0+=16){
#define MM_A_TN(r) (A[(size_t)k*M+i0+r])
            MM_KERN_4x16(MM_A_TN);
#undef MM_A_TN
        }
        for(int r=0;r<4;r++){ float *c=C+(size_t)(i0+r)*N;
            for(int j=NB;j<N;j++) c[j]=0.f;
            for(int k=0;k<K;k++){ float av=A[(size_t)k*M+i0+r]; if(av==0.f) continue; const float *b=B+(size_t)k*N;
                for(int j=NB;j<N;j++) c[j]+=av*b[j]; } }
    }
    for(int i=MB;i<M;i++){
        float *c=C+(size_t)i*N; memset(c,0,(size_t)N*4);
        for(int k=0;k<K;k++){ float av=A[(size_t)k*M+i]; if(av==0.f) continue; const float *b=B+(size_t)k*N;
            for(int j=0;j<N;j++) c[j]+=av*b[j]; }
    }
#else
    #pragma omp parallel for schedule(static) if(MM_OUTER)
    for(int i=0;i<M;i++){
        float *c=C+(size_t)i*N; memset(c,0,(size_t)N*4);
        for(int k=0;k<K;k++){ float av=A[(size_t)k*M+i]; if(av==0.f) continue;
            const float *b=B+(size_t)k*N;
            for(int j=0;j<N;j++) c[j]+=av*b[j]; }
    }
#endif
}

/* ---------------------------------------------------------------- fp16 operand preparation */
/* absmax over n floats. NEON + OpenMP; max is associative and exact, so the split is value-identical
 * to the scalar scan whatever the thread count. */
static float mm_absmax(const float *x,size_t n){
    float mx=0;
#ifdef MM_NEON
    size_t nb=n&~15u;
    #pragma omp parallel for schedule(static) reduction(max:mx) if(MM_OUTER && n>(1u<<16))
    for(size_t i=0;i<nb;i+=16){
        float32x4_t v0=vabsq_f32(vld1q_f32(x+i)),   v1=vabsq_f32(vld1q_f32(x+i+4));
        float32x4_t v2=vabsq_f32(vld1q_f32(x+i+8)), v3=vabsq_f32(vld1q_f32(x+i+12));
        float m=vmaxvq_f32(vmaxq_f32(vmaxq_f32(v0,v1),vmaxq_f32(v2,v3)));
        if(m>mx) mx=m;
    }
    for(size_t i=nb;i<n;i++){ float a=fabsf(x[i]); if(a>mx) mx=a; }
#else
    for(size_t i=0;i<n;i++){ float a=fabsf(x[i]); if(a>mx) mx=a; }
#endif
    return mx;
}
static float mm_pow2_scale(const float *x,size_t n){
    if(!g_mm_scale) return 1.0f;
    float mx=mm_absmax(x,n);
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
/* dst[r,c] = f16(src[r,c]*s)  (row-major, same layout).
 * NEON + OpenMP. s is an exact power of two (mm_pow2_scale), so vmulq_f32 by it is EXACT — it is the
 * exponent add the scaling wants, expressed as the one instruction that also gets zero, subnormal and
 * overflow right (a raw exponent add on the bit pattern does not). vcvt_f16_f32 rounds to nearest-even,
 * the same as the scalar (mm_f16) cast, so this is bit-identical to the loop it replaces. */
static void to_f16(mm_f16 *dst,const float *src,size_t n,float s){
    if(g_mm_sr){ for(size_t i=0;i<n;i++) dst[i]=sr_cast(src[i]*s); return; }
#ifdef MM_NEON
    size_t nb=n&~7u; float32x4_t vs=vdupq_n_f32(s);
    #pragma omp parallel for schedule(static) if(MM_OUTER && n>(1u<<16))
    for(size_t i=0;i<nb;i+=8){
        float16x4_t h0=vcvt_f16_f32(vmulq_f32(vld1q_f32(src+i),vs));
        float16x4_t h1=vcvt_f16_f32(vmulq_f32(vld1q_f32(src+i+4),vs));
        vst1_u16((uint16_t*)dst+i,  vreinterpret_u16_f16(h0));
        vst1_u16((uint16_t*)dst+i+4,vreinterpret_u16_f16(h1));
    }
    for(size_t i=nb;i<n;i++) dst[i]=(mm_f16)(src[i]*s);
#else
    for(size_t i=0;i<n;i++) dst[i]=(mm_f16)(src[i]*s);
#endif
}
/* dst[c,r] = f16(src[r,c]*s): transpose + narrow */
static float *g_tmpf; static size_t g_tmpf_n;
__attribute__((unused)) static float *tmpf(size_t n){ if(n>g_tmpf_n){ free(g_tmpf); g_tmpf=malloc(n*4); g_tmpf_n=n; if(!g_tmpf){fprintf(stderr,"oom\n");exit(3);} } return g_tmpf; }
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
#ifdef MM_NEON
    /* FUSED scale + transpose + narrow, cache-blocked and threaded. The library's
     * ork_f32_transpose_to_f16 is the same 4x4 register transpose but takes no scale and is
     * deliberately single-threaded (see f16/transpose.c), so the old route needed a whole extra fp32
     * pass to pre-scale — on the 1 GB head operands that pass cost more than the transpose. Folding
     * the exact power-of-two scale into the block's four multiplies removes it. */
    { const int R=rows, C=cols, TB=64; int rv=R&~3, cv=C&~3; float32x4_t vs=vdupq_n_f32(s);
    #pragma omp parallel for schedule(static) if(MM_OUTER && (size_t)rows*cols>(1u<<16))
    for(int r0=0;r0<rv;r0+=TB){
        int rm=r0+TB<rv?r0+TB:rv;
        for(int c0=0;c0<cv;c0+=TB){
            int cm=c0+TB<cv?c0+TB:cv;
            for(int r=r0;r<rm;r+=4) for(int c=c0;c<cm;c+=4){
                float32x4_t a0=vmulq_f32(vld1q_f32(src+(size_t)(r+0)*C+c),vs);
                float32x4_t a1=vmulq_f32(vld1q_f32(src+(size_t)(r+1)*C+c),vs);
                float32x4_t a2=vmulq_f32(vld1q_f32(src+(size_t)(r+2)*C+c),vs);
                float32x4_t a3=vmulq_f32(vld1q_f32(src+(size_t)(r+3)*C+c),vs);
                float32x4x2_t t0=vtrnq_f32(a0,a1), t1=vtrnq_f32(a2,a3);
                float32x4_t b0=vcombine_f32(vget_low_f32 (t0.val[0]),vget_low_f32 (t1.val[0]));
                float32x4_t b1=vcombine_f32(vget_low_f32 (t0.val[1]),vget_low_f32 (t1.val[1]));
                float32x4_t b2=vcombine_f32(vget_high_f32(t0.val[0]),vget_high_f32(t1.val[0]));
                float32x4_t b3=vcombine_f32(vget_high_f32(t0.val[1]),vget_high_f32(t1.val[1]));
                vst1_u16((uint16_t*)dst+(size_t)(c+0)*R+r, vreinterpret_u16_f16(vcvt_f16_f32(b0)));
                vst1_u16((uint16_t*)dst+(size_t)(c+1)*R+r, vreinterpret_u16_f16(vcvt_f16_f32(b1)));
                vst1_u16((uint16_t*)dst+(size_t)(c+2)*R+r, vreinterpret_u16_f16(vcvt_f16_f32(b2)));
                vst1_u16((uint16_t*)dst+(size_t)(c+3)*R+r, vreinterpret_u16_f16(vcvt_f16_f32(b3)));
            }
        }
    }
    for(int r=rv;r<R;r++) for(int c=0;c<C;c++) dst[(size_t)c*R+r]=(mm_f16)(src[(size_t)r*C+c]*s);
    for(int r=0;r<rv;r++) for(int c=cv;c<C;c++) dst[(size_t)c*R+r]=(mm_f16)(src[(size_t)r*C+c]*s); }
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

/* Does the WHOLE contraction sit inside orki_f16_sched's window? Once it did imply a slow path: the
 * library sliced K on the SoC's 2048, which is OUTSIDE the window, so a non-conforming K ran at the
 * one-bank 8-row M-cap. Since orki_f16_ks the slice is 1024 and every FULL slice is inside, so this
 * counter now only flags shapes whose TAIL slice (K mod 1024) may be non-conforming. Kept because it
 * still names the shapes worth probing, not because the count is a cost. */
static int pow2_fast(int K){ return K>=128 && K<2048 && (K&(K-1))==0; }

/* The fp16 paths share one operand preparation: A16 [M,K], B16 [K,N], scales sa, sb. */
static void mm_f16_core(int M,int K,int N,const mm_f16*A16,const mm_f16*B16,float sa,float sb,float*C,int mode){
    if(mode==MM_NPU){
#ifdef WITH_NPU
        double tn=now_s();
        ork_w *w=mm_scratch(K,N);
        if(ork_f16_mm_repack(g_npu,w,K,N,B16)<0){ fprintf(stderr,"repack(%d,%d) failed\n",K,N); exit(4); }
        double tr=now_s(); g_mm_repack_sec[g_mm_class]+=tr-tn;
        int rc=ork_f16_mm_run(g_npu,w,M,A16,C);
        if(rc){ fprintf(stderr,"ork_f16_mm_run M=%d K=%d N=%d rc=%d -- STOPPING (no retry)\n",M,K,N,rc); exit(4); }
        g_mm_run_sec[g_mm_class]+=now_s()-tr;
        g_mm_npu_sec[g_mm_class]+=now_s()-tn;
#else
        fprintf(stderr,"MM_NPU requested in a CPU-only build\n"); exit(3);
#endif
    } else emu_nn16(M,K,N,A16,B16,C);
    /* undo the two operand scales. Both are exact powers of two, so the product and its reciprocal
     * are exact and the multiply is lossless; NEON + OpenMP because on the head shapes this touches
     * 254M floats. */
    float inv=1.0f/(sa*sb);
    if(inv!=1.0f){
        size_t n=(size_t)M*N;
#ifdef MM_NEON
        size_t nb=n&~15u; float32x4_t vi=vdupq_n_f32(inv);
        #pragma omp parallel for schedule(static) if(MM_OUTER && n>(1u<<16))
        for(size_t i=0;i<nb;i+=16){
            vst1q_f32(C+i,   vmulq_f32(vld1q_f32(C+i),vi));
            vst1q_f32(C+i+4, vmulq_f32(vld1q_f32(C+i+4),vi));
            vst1q_f32(C+i+8, vmulq_f32(vld1q_f32(C+i+8),vi));
            vst1q_f32(C+i+12,vmulq_f32(vld1q_f32(C+i+12),vi));
        }
        for(size_t i=nb;i<n;i++) C[i]*=inv;
#else
        for(size_t i=0;i<n;i++) C[i]*=inv;
#endif
    }
}

/* which: 0 = nn (A[M,K] B[K,N]), 1 = nt (B is [N,K]), 2 = tn (A is [K,M]). */
/* The GDN head loop runs these concurrently when the chunk matmuls are on the CPU, so every counter
 * update is atomic. Uncontended atomics cost a few ns against a matmul; losing counts would make the
 * report silently wrong. In that mode the SECONDS are CPU time summed over threads, not wall — the
 * wall figure for the GDN block is T_GDN, which the harness times around the whole parallel loop. */
#ifdef _OPENMP
#define MM_ADD(x,v) do { _Pragma("omp atomic") (x) += (v); } while(0)
#else
#define MM_ADD(x,v) do { (x) += (v); } while(0)
#endif
static void mm_any(int which,int M,int K,int N,const float*A,const float*B,float*C){
    int cls=g_mm_class, mode=g_mm_mode[cls];
    double t0=now_s();
    MM_ADD(g_mm_calls[cls],1); MM_ADD(g_mm_flop[cls],2.0*M*K*N);
    int fb=0;
    if(mode!=MM_CPU && (K%32 || N%16)){ MM_ADD(g_mm_fallback[cls],1); mode=MM_CPU; fb=1; }
    if(mode==MM_CPU){
        if(which==0) cpu_nn(M,K,N,A,B,C); else if(which==1) cpu_nt(M,K,N,A,B,C); else cpu_tn(M,K,N,A,B,C);
        if(fb) MM_ADD(g_mm_fb_sec[cls],now_s()-t0);
    } else {
        if(!pow2_fast(K)) g_mm_slowpath[cls]++;
        double ts=now_s();
        float sa=mm_pow2_scale(A,(size_t)M*K), sb=mm_pow2_scale(B,(size_t)K*N);
        double tc=now_s(); g_mm_scan_sec[cls]+=tc-ts;
        mm_f16 *A16=buf16(&g_ha,&g_ha_n,(size_t)M*K), *B16=buf16(&g_hb,&g_hb_n,(size_t)K*N);
        if(which==2){ to_f16_t(A16,A,K,M,sa); g_mm_tr_sec[cls]+=now_s()-tc; }
        else        { to_f16  (A16,A,(size_t)M*K,sa); g_mm_cast_sec[cls]+=now_s()-tc; }
        tc=now_s();
        if(which==1){ to_f16_t(B16,B,N,K,sb); g_mm_tr_sec[cls]+=now_s()-tc; }
        else        { to_f16  (B16,B,(size_t)K*N,sb); g_mm_cast_sec[cls]+=now_s()-tc; }
        mm_f16_core(M,K,N,A16,B16,sa,sb,C,mode);
    }
    MM_ADD(g_mm_sec[cls],now_s()-t0);
}
static inline void mm_nn(int M,int K,int N,const float*A,const float*B,float*C){ mm_any(0,M,K,N,A,B,C); }
static inline void mm_nt(int M,int K,int N,const float*A,const float*B,float*C){ mm_any(1,M,K,N,A,B,C); }
static inline void mm_tn(int M,int K,int N,const float*A,const float*B,float*C){ mm_any(2,M,K,N,A,B,C); }

static void mm_stats_reset(void){
    memset(g_mm_sec,0,sizeof g_mm_sec); memset(g_mm_npu_sec,0,sizeof g_mm_npu_sec); memset(g_mm_flop,0,sizeof g_mm_flop);
    memset(g_mm_calls,0,sizeof g_mm_calls); memset(g_mm_fallback,0,sizeof g_mm_fallback);
    memset(g_mm_slowpath,0,sizeof g_mm_slowpath);
    memset(g_mm_scan_sec,0,sizeof g_mm_scan_sec); memset(g_mm_cast_sec,0,sizeof g_mm_cast_sec);
    memset(g_mm_tr_sec,0,sizeof g_mm_tr_sec);
    memset(g_mm_repack_sec,0,sizeof g_mm_repack_sec); memset(g_mm_run_sec,0,sizeof g_mm_run_sec);
    memset(g_mm_fb_sec,0,sizeof g_mm_fb_sec);
}
#endif /* TINY_E2E_MM_H */
