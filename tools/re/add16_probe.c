/* add16_probe.c — decode the int16 element-wise ADD operand-scale registers by driving each operand alone.
 * Uses ork_npu_add_i16 with env reg overrides (ORK_ADD16_R48/R84/R88/R78). Feeds (a=ramp,b=0) then (a=0,b=ramp)
 * so the a-path and b-path transfer are isolated -> reveals which reg scales which SDP operand (X1 vs X2).
 *   sudo env ORK_ADD16_R48=.. ORK_ADD16_R84=.. ORK_ADD16_R88=.. ORK_ADD16_R78=.. ./add16_probe
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include "ork_npu.h"
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("SKIP\n");return 0;}
    if(!ork_ppu_fuse_enabled(c)){printf("SKIP\n");ork_npu_free(c);return 0;}
    const int M=8,N=64; static short a[512],b[512],out[512]; double us=0;
    /* a-path: a=100*ramp, b=0. equal scales => out should track a if a-coeff==1 */
    for(int i=0;i<M*N;i++){ a[i]=(short)(-(i%40)*100); b[i]=0; }
    if(ork_npu_add_i16(c,a,b,M,N,0.001,0.001,0.001,out,&us)){printf("wedge\n");ork_npu_free(c);return 1;}
    printf("A-path (b=0):  ");
    for(int i=0;i<6;i++) printf("a=%d->%d  ", a[i*3], out[i*3]); printf("\n");
    for(int i=0;i<M*N;i++){ a[i]=0; b[i]=(short)(-(i%40)*100); }
    if(ork_npu_add_i16(c,a,b,M,N,0.001,0.001,0.001,out,&us)){printf("wedge\n");ork_npu_free(c);return 1;}
    printf("B-path (a=0):  ");
    for(int i=0;i<6;i++) printf("b=%d->%d  ", b[i*3], out[i*3]); printf("\n");
    /* both nonzero: out should = a+b */
    for(int i=0;i<M*N;i++){ a[i]=(short)((i%40)*100); b[i]=(short)((i%25)*80); }
    if(ork_npu_add_i16(c,a,b,M,N,0.001,0.001,0.001,out,&us)){printf("wedge\n");ork_npu_free(c);return 1;}
    printf("A+B:  ");
    for(int i=0;i<8;i++) printf("%d+%d->%d(exp%d)  ", a[i*3], b[i*3], out[i*3], a[i*3]+b[i*3]); printf("\n");
    /* transfer sweep: a=b=v across magnitudes -> out vs 2v (find the scaling) */
    for(int i=0;i<M*N;i++){ int v=-((i%40)*400); a[i]=(short)v; b[i]=(short)v; }
    if(ork_npu_add_i16(c,a,b,M,N,0.001,0.001,0.001,out,&us)){printf("wedge\n");ork_npu_free(c);return 1;}
    printf("a=b sweep:  ");
    for(int i=0;i<40;i++){ printf("v=%d->%d(2v=%d) ", a[i], out[i], 2*a[i]); if(i%4==3)printf("\n"); }
    ork_npu_free(c); return 0;
}
