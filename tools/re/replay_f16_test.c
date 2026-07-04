/* identify what the fp16 task1 LUT op computes: replay RKNN's fp16 LUT + params, compare to sigmoid AND silu. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ork_npu.h"
#include "regcmd_sig_f16.h"   /* LUT_SIG_F16[] */
static double sigf(double x){ return 1.0/(1.0+exp(-x)); }
static double siluf(double x){ return x/(1.0+exp(-x)); }
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("SKIP\n");return 0;}
    if(!ork_ppu_fuse_enabled(c)){printf("SKIP\n");ork_npu_free(c);return 0;}
    const int M=8,N=64; static ork_f16 in[512],out[512];
    for(int i=0;i<M*N;i++){ double x=-8.0+16.0*i/(M*N-1); in[i]=(ork_f16)x; }
    double us=0;
    int r=ork_npu_probe_silu_std_f16(c,in,M,N,0xffffc000u,0x80000000u,0x69840000u,LUT_SIG_F16,LUT_SIG_F16_N,out,&us);
    printf("replay fp16 rc=%d (%.1f us)\n",r,us);
    if(!r) for(int i=0;i<M*N;i+=8){ double x=(float)in[i],o=(float)out[i];
        printf("  x=%7.3f out=%9.4f sig=%.4f silu=%.4f x*sig=%.4f\n",x,o,sigf(x),siluf(x),x*sigf(x)); }
    ork_npu_free(c); return 0;
}
