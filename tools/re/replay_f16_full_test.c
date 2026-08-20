#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ork_npu.h"
#include "regcmd_silu_lut_f16.h"   /* REGCMD_SILU_LUT_F16[] — fp16 LE-table loader (RKNN curve baked) */
static double sigf(double x){ return 1.0/(1.0+exp(-x)); }
static double siluf(double x){ return x/(1.0+exp(-x)); }
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("SKIP\n");return 0;}
    if(!ork_ppu_fuse_enabled(c)){printf("SKIP\n");ork_npu_free(c);return 0;}
    const int M=8,N=64; static ork_f16 in[512],out[512];
    for(int i=0;i<M*N;i++){ double x=-8.0+16.0*i/(M*N-1); in[i]=(ork_f16)x; }
    double us=0;
    int r=ork_f16_npu_replay_full(c, REGCMD_SILU_LUT_F16, REGCMD_SILU_LUT_F16_N, in, M, N, out, &us);
    printf("replay_full_f16 rc=%d (%.1f us)\n",r,us);
    if(!r) for(int i=0;i<M*N;i+=6){ double x=(float)in[i],o=(float)out[i];
        printf("  x=%7.3f out=%9.4f sig=%.4f silu=%.4f\n",x,o,sigf(x),siluf(x)); }
    ork_npu_free(c); return 0;
}
