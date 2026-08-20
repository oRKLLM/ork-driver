/* op_profile — per-op NPU vs CPU timing across the transformer hot path, decode (M=1) + prefill (M=64).
 * For each op: warm NPU time vs a CPU reference (NEON where trivial; scalar expf/tanhf for transcendental
 * activations — note ggml's CPU uses vectorized approximations, so the real CPU is faster than this naive
 * ref for silu/gelu/exp). Ratio = CPU/NPU (>1 => NPU faster => an offload candidate). Reveals where the
 * NPU genuinely wins vs where the CPU does. Shapes ~7B (hidden 3584, ff 18944, nkv 512).
 *   make op_profile && sudo env ORK_MM_TIMEOUT=4000 ./op_profile [iters]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static volatile int32_t g_sink;

/* ---- CPU references (int8 in/out, dequant->func->requant, same work the NPU SDP op does) ---- */
static int8_t qc(float x){ int q=(int)lrintf(x); return (int8_t)(q<-128?-128:q>127?127:q); }
static void cpu_silu(const int8_t*in,int M,int N,double is,double os,int8_t*out){
    for(size_t i=0;i<(size_t)M*N;i++){ float x=in[i]*is; float s=x/(1.f+expf(-x)); out[i]=qc(s/os); } }
static void cpu_gelu(const int8_t*in,int M,int N,double is,double os,int8_t*out){
    for(size_t i=0;i<(size_t)M*N;i++){ float x=in[i]*is; float g=0.5f*x*(1.f+tanhf(0.7978845608f*(x+0.044715f*x*x*x))); out[i]=qc(g/os); } }
static void cpu_exp(const int8_t*in,int M,int N,double is,double os,int8_t*out){
    for(size_t i=0;i<(size_t)M*N;i++){ out[i]=qc(expf(in[i]*is)/os); } }
static void cpu_rsqrt(const int8_t*in,int M,int N,double is,double os,int8_t*out){
    for(size_t i=0;i<(size_t)M*N;i++){ float x=in[i]*is; if(x<1e-6f)x=1e-6f; out[i]=qc((1.f/sqrtf(x))/os); } }
static void cpu_ewmul(const int8_t*u,const int8_t*s,int M,int N,int mult,int shift,int8_t*out){
    for(size_t i=0;i<(size_t)M*N;i++){ int v=((int)u[i]*(int)s[i]*mult)>>shift; out[i]=(int8_t)(v<-128?-128:v>127?127:v); } }
static void cpu_add(const int8_t*a,const int8_t*b,int M,int N,double as,double bs,double os,int8_t*out){
    for(size_t i=0;i<(size_t)M*N;i++){ out[i]=qc((a[i]*as+b[i]*bs)/os); } }
/* NEON sdot int8 GEMV for the CPU matmul ref (M rows) */
static void cpu_mm(const int8_t*A,const int8_t*Bt,int M,int K,int N,int32_t*C){
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ const int8_t*a=A+(size_t)m*K,*b=Bt+(size_t)n*K;
        int32x4_t ac=vdupq_n_s32(0); int k=0; for(;k+16<=K;k+=16) ac=vdotq_s32(ac,vld1q_s8(a+k),vld1q_s8(b+k));
        int32_t sacc=vaddvq_s32(ac); for(;k<K;k++)sacc+=(int32_t)a[k]*b[k]; C[(size_t)m*N+n]=sacc; } }

static double tcpu(void(*f)(void),int iters){ double t0=now_us(); for(int i=0;i<iters;i++)f(); return (now_us()-t0)/iters; }

/* globals for the thunks */
static ork_npu*C_; static int M_,N_,K_; static int8_t *I1,*I2,*O8; static int32_t *Oi; static ork_w*W_; static int8_t*A_,*Bt_;
static void t_silu(void){ cpu_silu(I1,M_,N_,0.05,0.05,O8); }
static void t_gelu(void){ cpu_gelu(I1,M_,N_,0.05,0.05,O8); }
static void t_exp(void){ cpu_exp(I1,M_,N_,0.02,0.02,O8); }
static void t_rsqrt(void){ cpu_rsqrt(I1,M_,N_,0.05,0.05,O8); }
static void t_ewmul(void){ cpu_ewmul(I1,I2,M_,N_,1,7,O8); }
static void t_add(void){ cpu_add(I1,I2,M_,N_,1,1,1,O8); }
static void t_mm(void){ cpu_mm(A_,Bt_,M_,K_,N_,Oi); }

int main(int argc,char**argv){
    int iters=argc>1?atoi(argv[1]):20;
    int HID=3584, FF=18944, NKV=512;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;} C_=c;
    int Ms[2]={1,64};
    printf("per-op NPU vs CPU (7B shapes; ratio=CPU/NPU, >1 => NPU faster)\n");
    printf("  %-10s %-8s %10s %10s %8s\n","op","M","NPU us","CPU us","ratio");
    size_t big=(size_t)64*FF; int8_t*a=malloc(big),*b=malloc(big); int8_t*o=malloc(big);
    for(size_t i=0;i<big;i++){a[i]=(int8_t)((i*2654435761u)>>25);b[i]=(int8_t)((i*40503u)>>7);}
    I1=a;I2=b;O8=o;
    double us;
    for(int mi=0;mi<2;mi++){ int M=Ms[mi]; M_=M;
        #define ROW(name,N,npu_call,cpu_thunk) do{ N_=(N); \
            int rc=(npu_call); double t0=now_us(); for(int i=0;i<iters;i++){ (npu_call); } double np=(now_us()-t0)/iters; \
            double cp=tcpu(cpu_thunk,iters); \
            printf("  %-10s %-8d %10.1f %10.1f %7.2fx  rc=%d\n",name,M,np,cp,cp>0&&np>0?cp/np:0,rc);}while(0)
        ROW("silu",  FF,  ork_i8_npu_silu (c,I1,M,N_,0.05,0.05,O8,&us), t_silu);
        ROW("gelu",  FF,  ork_i8_npu_gelu (c,I1,M,N_,0.05,0.05,O8,&us), t_gelu);
        ROW("exp",   NKV, ork_i8_npu_exp  (c,I1,M,N_,0.02,0.02,O8,&us), t_exp);
        ROW("rsqrt", HID, ork_i8_npu_rsqrt(c,I1,M,N_,0.05,0.05,O8,&us), t_rsqrt);
        ROW("ewmul", FF,  ork_i8_npu_ewmul(c,I1,I2,M,N_,1,7,O8,&us),    t_ewmul);
        ROW("add",   HID, ork_i8_npu_add  (c,I1,I2,M,N_,1,1,1,O8,&us),  t_add);
        #undef ROW
    }
    /* matmul: pack once per shape, NPU run vs NEON sdot CPU */
    int8_t*Bpk=malloc((size_t)FF*HID); memset(Bpk,1,(size_t)FF*HID);
    struct{const char*nm;int K,N;}mm[3]={{"mm-proj",HID,HID},{"mm-gate/up",HID,FF},{"mm-down",FF,HID}};
    A_=malloc((size_t)64*FF); memset(A_,1,(size_t)64*FF); Oi=malloc((size_t)64*FF*4);
    for(int s=0;s<3;s++){ int K=mm[s].K,N=mm[s].N; K_=K;
        ork_npu_set_pack_domain(c,0); W_=ork_i8_mm_pack(c,K,N,Bpk); if(!W_){printf("pack %s fail\n",mm[s].nm);continue;}
        Bt_=malloc((size_t)N*K); memset(Bt_,1,(size_t)N*K);
        for(int mi=0;mi<2;mi++){ int M=Ms[mi]; M_=M; N_=N;
            ork_i8_mm_run(c,W_,M,A_,Oi); double t0=now_us(); for(int i=0;i<iters;i++)ork_i8_mm_run(c,W_,M,A_,Oi); double np=(now_us()-t0)/iters;
            double cp=tcpu(t_mm,iters<5?iters:5);
            printf("  %-10s %-8d %10.1f %10.1f %7.2fx  (K%d N%d)\n",mm[s].nm,M,np,cp,cp>0&&np>0?cp/np:0,K,N); }
        ork_w_free(W_); free(Bt_);
    }
    (void)g_sink; ork_npu_free(c); return 0;
}
