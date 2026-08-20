/* tools/silu_native.c — ork-NATIVE fused SiLU: build ork's own silu LUT for its own matmul program.
 *
 * The RKNN-captured LUT + registers are for RKNN's 126-reg fused op and do NOT transfer to ork's 108-reg
 * matmul program (REGCMD_I8). But ork's fused-output path DOES apply a Q6-PWL LUT — we just have to build
 * OUR OWN LUT matched to ork's acc->index mapping. Since we control both the LUT (probe lut override) and
 * the registers, that's a clean 3-pass construction that yields CORRECT silu on the NPU (validated ~1 int8):
 *
 *   pass 1  MEASURE ork's index(acc): run the fused path with a RAMP LUT (LUT[i]=(i-512)*8) + chosen
 *           registers (R = mult/2^shift = 0.25 so a reachable acc range spans silu's transition band,
 *           cfg4068 fixed, bias=0). out = R*LUT[idx] = 2*(idx-512)  =>  idx = out/2 + 512.
 *   pass 2  BUILD ork's LUT: for each calibration acc, LUT[idx(acc)] = correct_silu_i8(acc)/R
 *           = 4 * clamp(round(silu(acc*in_scale)/out_scale)); interpolate gaps, hold at ends.
 *   pass 3  VALIDATE: run with the built LUT; compare to correct silu.
 *
 * index(acc) depends only on (R, cfg4068) — measure ONCE, then generate the LUT for ANY (in_scale,out_scale).
 * Measured: mean|err| ~1 int8, max 3-4, across in 2.5e-4..1.5e-3 / out 0.04..0.20 (near the int8 rounding floor).
 *
 *   make silu_native && sudo ./silu_native            (board only)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ork_npu.h"

#define M 1
#define K 512
#define N 64
/* chosen register config (index(acc) is measured for these) */
#define R_MULT   0x4000
#define R_SHIFT  0x10        /* R = 0x4000/2^16 = 0.25 */
#define CFG4068  0x56391300u
#define OUT_BIAS 0x0u
#define IDX_OFF  0xffffc000u

static signed char A[K], B[K*N], C[N];
static short lut[1030];
static int acc[N], idx[N];
static ork_npu *c;

static double silu(double x){ return x/(1.0+exp(-x)); }
static int tgt(int a, double in_s, double out_s){
    double v = silu(a*in_s)/out_s; long r = lround(v);
    if(r>127)r=127; if(r<-128)r=-128; return (int)r;
}
static int run(void){ return ork_i8_npu_probe_silu_cfg(c,M,K,N,A,B,R_MULT,R_SHIFT,OUT_BIAS,IDX_OFF,CFG4068,lut,1030,C,0); }

/* build ork's LUT for (in_scale,out_scale) from the measured idx(acc), run, return mean/max err */
static void build_and_run(double in_s, double out_s){
    int set[1030]; for(int i=0;i<1030;i++){ set[i]=0; lut[i]=0; }
    for(int n=0;n<N;n++){ int i=idx[n]; if(i<0||i>1029) continue; lut[i]=(short)(tgt(acc[n],in_s,out_s)*4); set[i]=1; }
    int lo=-1,hi=-1; for(int i=0;i<1030;i++) if(set[i]){lo=i;break;} for(int i=1029;i>=0;i--) if(set[i]){hi=i;break;}
    for(int i=0;i<lo;i++) lut[i]=lut[lo]; for(int i=hi+1;i<1030;i++) lut[i]=lut[hi];
    for(int i=lo;i<=hi;i++){ if(set[i])continue; int a=i,b=i; while(a>lo&&!set[a])a--; while(b<hi&&!set[b])b++;
        lut[i]=(short)(lut[a]+(lut[b]-lut[a])*(i-a)/(b-a)); }
    if(run()){ printf("  in=%.3e out=%.3e : RUN FAILED\n",in_s,out_s); return; }
    int mx=0; long s=0; for(int n=0;n<N;n++){ int e=abs(C[n]-tgt(acc[n],in_s,out_s)); s+=e; if(e>mx)mx=e; }
    printf("  in=%.3e out=%.3e : mean|err|=%.2f max=%d\n", in_s,out_s,(double)s/N,mx);
}

int main(void){
    c=ork_npu_init(); if(!c){ printf("no board\n"); return 0; }
    for(int k=0;k<K;k++) A[k]=1;
    for(int n=0;n<N;n++){ int T=(n-32)*500; int b=T/K; for(int k=0;k<K;k++) B[k*N+n]=(signed char)(b+(k<(T-b*K)?1:0)); }
    for(int n=0;n<N;n++){ int a=0; for(int k=0;k<K;k++) a+=A[k]*B[k*N+n]; acc[n]=a; }
    /* pass 1: measure idx(acc) with a ramp LUT */
    for(int i=0;i<1030;i++){ int v=(i-512)*8; if(v>32767)v=32767; if(v<-32768)v=-32768; lut[i]=(short)v; }
    if(run()){ printf("ramp run failed\n"); return 1; }
    for(int n=0;n<N;n++) idx[n]=C[n]/2+512;
    printf("ork-native fused SiLU (one idx(acc) measure, LUT rebuilt per scale):\n");
    build_and_run(3.75e-4,0.05); build_and_run(7.5e-4,0.10); build_and_run(1.5e-3,0.20);
    build_and_run(2.5e-4,0.04); build_and_run(5.0e-4,0.08);
    ork_npu_free(c); return 0;
}
