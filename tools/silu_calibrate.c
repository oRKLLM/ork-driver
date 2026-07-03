/* tools/silu_calibrate.c — pack-time calibrator for the fused-SiLU output stage.
 *
 * The fused-SiLU pipeline is fully decoded (acc -> R-requant -> Q6-PWL silu LUT -> R*V16 + bias -> int8;
 * see PPU_FUSED_ACTIVATION_WIP.md), but the exact register-generation formula is a multi-register,
 * multi-variable fit that doesn't close analytically. Since the mechanism IS known and the fixed LUT is
 * on-chip, the robust way to get bit-exact registers for a given (in_scale, out_scale) is to CALIBRATE:
 * search the small register space (R = mult/2^shift, cfg4068 index mantissa, out_bias) with the on-board
 * fused path as the oracle, minimizing error vs a CPU silu reference over a controlled accumulator sweep.
 * The winning register set is then cached with the packed weight.
 *
 * This is the scaffold: coordinate descent from the fitted centers (R ~ 2.9e-4/out_scale,
 * bias ~ -128 + 0.5/out_scale, cfg4068 ~ 0x5300_1100). Reports the best registers + residual error.
 *
 *   make silu_calibrate && sudo ./silu_calibrate <in_scale> <out_scale>   (board only)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ork_npu.h"

#define K 512
#define N 64

static double silu(double x){ return x/(1.0+exp(-x)); }

/* CPU reference (3-scale model): the LUT is indexed by the int8-QUANTIZED PRE-ACTIVATION, so silu is applied
 * to the quantized->dequantized preact, then requantized to out_scale:
 *   preact_q = clamp_i8(round(acc*in_scale / preact_scale)); out = clamp_i8(round(silu(preact_q*preact_scale)/out_scale)) */
static double g_preact_scale = 0;
static void cpu_ref(const int *acc, double in_scale, double out_scale, int8_t *ref){
    for(int n=0;n<N;n++){
        double preact = acc[n]*in_scale;
        long pq = lround(preact/g_preact_scale); if(pq>127)pq=127; if(pq<-128)pq=-128;
        double v = silu(pq*g_preact_scale)/out_scale;
        long r = lround(v); if(r>127)r=127; if(r<-128)r=-128; ref[n]=(int8_t)r;
    }
}

/* run the fused path with a candidate register set; return sum|err| vs ref (and max). */
static long eval(ork_npu*c, const int8_t*A, const int8_t*B, const int *acc,
                 int r_mult,int r_shift,uint32_t out_bias,uint32_t cfg4068,
                 const int8_t*ref, int *maxerr){
    static int8_t C[N];
    if(ork_npu_probe_i8_silu_cfg(c,1,K,N,A,B,r_mult,r_shift,out_bias,0xffffc000u,cfg4068,NULL,0,C,0))
        return -1;
    long s=0; int mx=0;
    for(int n=0;n<N;n++){ int e=abs((int)C[n]-(int)ref[n]); s+=e; if(e>mx)mx=e; }
    if(maxerr)*maxerr=mx;
    return s;
}

int main(int argc,char**argv){
    double in_scale  = argc>1?atof(argv[1]):7.26e-5;
    double out_scale = argc>2?atof(argv[2]):2.02e-2;
    g_preact_scale   = argc>3?atof(argv[3]):2.155e-2;
    /* optional: check a captured register set (argv 4..7 = mult shift bias cfg4068) against the 3-scale ref */
    int chk_mult = argc>4?(int)strtol(argv[4],0,0):0;
    ork_npu*c=ork_npu_init(); if(!c){printf("no board\n");return 0;}
    printf("calibrate fused-SiLU: in=%.4e preact=%.4e out=%.4e (3-scale ref, K=%d N=%d)\n",
           in_scale,g_preact_scale,out_scale,K,N);

    /* build a calibration matmul whose acc[n] sweeps silu's active range: x=acc*in_scale in ~[-6,6]. */
    static int8_t A[K], B[K*N]; static int acc[N];
    for(int k=0;k<K;k++)A[k]=1;
    double xspan=6.0; int accmax=(int)(xspan/in_scale); int step=(2*accmax)/(N-1);
    for(int n=0;n<N;n++){ int T=-accmax + n*step; int b=T/K; if(b>127)b=127; if(b<-128)b=-128;
        for(int k=0;k<K;k++)B[k*N+n]=(int8_t)b; }
    for(int n=0;n<N;n++){ int a=0; for(int k=0;k<K;k++)a+=A[k]*B[k*N+n]; acc[n]=a; }
    static int8_t ref[N]; cpu_ref(acc,in_scale,out_scale,ref);

    /* optional: evaluate a captured register set against the 3-scale ref (validates the model) */
    if(chk_mult){
        int cm=(int)strtol(argv[4],0,0), cs=(int)strtol(argv[5],0,0);
        uint32_t cb=(uint32_t)strtoul(argv[6],0,0), cc=(uint32_t)strtoul(argv[7],0,0);
        int mx; long e=eval(c,A,B,acc,cm,cs,cb,cc,ref,&mx);
        printf("  [captured regs] mult=0x%x shift=%d bias=0x%x cfg4068=0x%x -> sum|err|=%ld max=%d mean=%.2f\n",
               cm,cs,cb,cc,e,mx,(double)e/N);
    }

    /* fitted centers */
    double R0 = 2.9e-4/out_scale;
    int best_shift=0, best_mult=0; uint32_t best_bias=0xffffff9f, best_4068=0x53001100u;
    long best=-1; int bestmax=0;

    /* coordinate descent: (1) shift+mult from R sweep, (2) bias, (3) cfg4068 mantissa */
    for(int shift=12; shift<=24; shift++){
        long m = lround(R0*pow(2.0,shift));
        if(m<0x3000 || m>0x7fff) continue;                 /* keep mantissa in a sane fixed-point range */
        int mx; long e=eval(c,A,B,acc,(int)m,shift,best_bias,best_4068,ref,&mx);
        if(e>=0 && (best<0||e<best)){ best=e; bestmax=mx; best_shift=shift; best_mult=(int)m; }
    }
    printf("  [1] R sweep: mult=0x%x shift=%d  sum|err|=%ld max=%d\n",best_mult,best_shift,best,bestmax);
    /* refine mult around the winner */
    for(long m=best_mult-0x600; m<=best_mult+0x600; m+=0x80){ if(m<0x3000||m>0x7fff)continue;
        int mx; long e=eval(c,A,B,acc,(int)m,best_shift,best_bias,best_4068,ref,&mx);
        if(e>=0&&e<best){best=e;bestmax=mx;best_mult=(int)m;} }
    /* bias sweep */
    for(int bz=-128; bz<=0; bz+=4){ uint32_t bias=(uint32_t)(bz<0?(0xffffff00u|(bz&0xff)):bz);
        int mx; long e=eval(c,A,B,acc,best_mult,best_shift,bias,best_4068,ref,&mx);
        if(e>=0&&e<best){best=e;bestmax=mx;best_bias=bias;} }
    /* cfg4068 mantissa sweep (high 16 bits, low fixed 0x1100) */
    for(uint32_t mant=0x4c00; mant<=0x6400; mant+=0x100){ uint32_t cfg=(mant<<16)|0x1100u;
        int mx; long e=eval(c,A,B,acc,best_mult,best_shift,best_bias,cfg,ref,&mx);
        if(e>=0&&e<best){best=e;bestmax=mx;best_4068=cfg;} }

    printf("BEST: mult=0x%x shift=%d bias=0x%08x cfg4068=0x%08x  sum|err|=%ld  max|err|=%d  mean=%.2f\n",
           best_mult,best_shift,best_bias,best_4068,best,bestmax,(double)best/N);
    ork_npu_free(c);
    return 0;
}
