/* ork_native_roundtrip — validate ork_native_cpu.h pack->gemv against an f32 reference (accuracy) and
 * confirm the pack layout matches the dot layout (self-consistency). Gaussian weights (real-LLM-like).
 *   make ork_native_roundtrip && ./ork_native_roundtrip   (CPU-only, board-safe)
 */
#define _GNU_SOURCE
#include "ork_native_cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float relrmse(const float*x,const float*ref,int N){ double e=0,r=0; for(int n=0;n<N;n++){double d=x[n]-ref[n];e+=d*d;r+=(double)ref[n]*ref[n];} return (float)sqrt(e/(r+1e-12)); }

int main(void){
    int K=2048,N=1024;
    float*W=malloc((size_t)N*K*4),*Av=malloc(K*4),*ref=malloc(N*4),*out=malloc(N*4);
    uint64_t s=0x2545F4914F6CDD1DULL; float (*u)(void); (void)u;
    #define U() ({ s=s*6364136223846793005ULL+1442695040888963407ULL; ((s>>33)&0x7fffffff)/2147483647.0f; })
    for(int k=0;k<K;k++) Av[k]=0.5f*(U()-0.5f);
    for(size_t i=0;i<(size_t)N*K;i++){ float a=U(); if(a<1e-7f)a=1e-7f; if(a>0.9999999f)a=0.9999999f; W[i]=0.08f*sqrtf(-2.0f*logf(a))*cosf(6.2831853f*U()); }
    /* int8 activation */
    float amx=1e-9f; for(int k=0;k<K;k++){float v=fabsf(Av[k]); if(v>amx)amx=v;} float asc=amx/127.0f, ainv=127.0f/amx;
    int8_t*A=malloc(K); for(int k=0;k<K;k++){int q=(int)lrintf(Av[k]*ainv); A[k]=(int8_t)(q>127?127:q<-127?-127:q);}
    /* f32 reference: out_ref[n] = sum_k W[n,k]*Av[k] */
    for(int n=0;n<N;n++){ double d=0; const float*fr=W+(size_t)n*K; for(int k=0;k<K;k++) d+=(double)fr[k]*Av[k]; ref[n]=(float)d; }

    uint8_t*nib=malloc((size_t)N*K/2),*b4=malloc((size_t)N*K/8),*b5=malloc((size_t)N*K/8);
    int8_t*i8=malloc((size_t)N*K); float*bs=malloc(N*4); int8_t lut[16];
    const char*names[]={"int4","NF4 ","int5","int6","int8"};
    ork_cpu_fmt fmts[]={ORK_CPU_I4,ORK_CPU_NF4,ORK_CPU_I5,ORK_CPU_I6,ORK_CPU_I8};
    printf("ork_native_roundtrip: K=%d N=%d Gaussian; pack->gemv vs f32 ref (rel-RMSE, incl activation-int8 err)\n",K,N);
    for(int f=0;f<5;f++){ ork_cpu_fmt fmt=fmts[f];
        memset(nib,0,(size_t)N*K/2); memset(b4,0,(size_t)N*K/8); memset(b5,0,(size_t)N*K/8);
        ork_cpu_pack(fmt,K,N,W,nib,b4,b5,i8,bs,lut);
        ork_cpu_w w={0}; w.fmt=fmt; w.nibble=nib; w.bit4=b4; w.bit5=b5; w.i8=i8; w.bscale=bs; w.K=K; w.N=N;
        if(fmt==ORK_CPU_NF4) w.nf4_lut=vld1q_s8(lut);
        ork_cpu_gemv_m1(&w,A,asc,out,0,N);
        printf("  %s : rel-RMSE %.4f\n", names[f], relrmse(out,ref,N));
    }
    return 0;
}
