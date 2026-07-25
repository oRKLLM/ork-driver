/* cpu_op_sweep — per-op NPU-vs-CPU cost sweep to POPULATE the op-capability table's optimal placement.
 *
 * cpu_gemm_probe settled matmul (NPU by a mile). This sweeps the OTHER ops — the SDP/activation kind where the
 * NPU pays a LUT-load + SDP-stage + submit + completion-wait for what CPU does in one pass ("5 NPU ops -> 1 NEON
 * op"). For each op it times the real on-NPU path (ork_npu_*, which INCLUDES submit/LUT) vs a straightforward CPU
 * loop, at decode (M=1) and prefill (M>>1) sizes, and prints the winner -> that is the op's optimal `place`.
 * CPU baseline is scalar libm (conservative: ggml's SIMD activations are faster, so a CPU win here is decisive).
 *
 *   make cpu_op_sweep && sudo env ORK_MM_TIMEOUT=3000 ./cpu_op_sweep
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

static uint32_t rng = 0x77u;
static int8_t r8(void){ rng = rng*1664525u+1013904223u; return (int8_t)((int)((rng>>25)%255)-127); }
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static inline int8_t cl(float v){ int q=(int)lrintf(v); return (int8_t)(q>127?127:q<-127?-127:q); }

/* scalar CPU reference implementations (dequant -> f -> requant), the realistic "1 pass over the data" cost */
static void cpu_silu (const int8_t*in,int n,double is,double os,int8_t*o){ for(int i=0;i<n;i++){float x=in[i]*is; o[i]=cl((x/(1.f+expf(-x)))/os);} }
static void cpu_gelu (const int8_t*in,int n,double is,double os,int8_t*o){ for(int i=0;i<n;i++){float x=in[i]*is; o[i]=cl((0.5f*x*(1.f+erff(x*0.70710678f)))/os);} }
static void cpu_rsqrt(const int8_t*in,int n,double is,double os,int8_t*o){ for(int i=0;i<n;i++){float x=in[i]*is; if(x<1e-6f)x=1e-6f; o[i]=cl((1.f/sqrtf(x))/os);} }
static void cpu_exp  (const int8_t*in,int n,double is,double os,int8_t*o){ for(int i=0;i<n;i++){float x=in[i]*is; o[i]=cl(expf(x)/os);} }
static void cpu_add  (const int8_t*a,const int8_t*b,int n,double as,double bs,double os,int8_t*o){ for(int i=0;i<n;i++) o[i]=cl((a[i]*as+b[i]*bs)/os); }

int main(void){
    setvbuf(stdout,0,_IONBF,0);
    ork_npu *c = ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    const double is=0.02, os=0.02;
    int sizes[][2] = { {1,2048}, {228,2048} };   /* decode (M=1) and prefill (M>>1) */
    printf("%-8s %-12s %10s %10s   %-6s\n","op","shape","NPU us","CPU us","optimal");
    for (int s=0;s<2;s++){
        int M=sizes[s][0], N=sizes[s][1], n=M*N; const char*tag = M==1?"decode":"prefill";
        int8_t *in=malloc(n),*b=malloc(n),*o=malloc(n); double us=0;
        for(int i=0;i<n;i++){ in[i]=r8(); b[i]=r8(); }
        #define SWEEP1(NAME, NPUCALL, CPUCALL) do{ \
            NPUCALL; /* warm (LUT calibrate once) */ \
            double t0=now_us(); for(int it=0;it<3;it++){ NPUCALL; } double npu=(now_us()-t0)/3; \
            double t1=now_us(); for(int it=0;it<3;it++){ CPUCALL; } double cpu=(now_us()-t1)/3; \
            printf("%-8s %-12s %10.1f %10.1f   %-6s\n", NAME, tag, npu, cpu, cpu<npu?"CPU":"NPU"); \
        }while(0)
        char shp[16]; snprintf(shp,sizeof shp,"M%dxN%d",M,N); (void)shp;
        SWEEP1("silu",  ork_npu_silu_i8 (c,in,M,N,is,os,o,&us),        cpu_silu (in,n,is,os,o));
        SWEEP1("gelu",  ork_npu_gelu_i8 (c,in,M,N,is,os,o,&us),        cpu_gelu (in,n,is,os,o));
        SWEEP1("rsqrt", ork_npu_rsqrt_i8(c,in,M,N,is,os,o,&us),        cpu_rsqrt(in,n,is,os,o));
        SWEEP1("exp",   ork_npu_exp_i8  (c,in,M,N,is,os,o,&us),        cpu_exp  (in,n,is,os,o));
        SWEEP1("add",   ork_npu_add_i8  (c,in,b,M,N,is,is,os,o,&us),   cpu_add  (in,b,n,is,is,os,o));
        free(in);free(b);free(o);
        #undef SWEEP1
    }
    ork_npu_free(c);
    return 0;
}
