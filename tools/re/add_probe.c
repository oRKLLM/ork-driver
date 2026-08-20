/* add_probe.c — decode the standalone int8 element-wise ADD scale structure on silicon. Sweeps the out-scale
 * (mult/shift) and b-operand scale (0x4078) with za=zb=zo=0 to find the config giving plain residual add
 * out=clamp_i8(a+b). Then validates. Usage: add_probe [mult] [shift] [bscale_hex]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include "ork_npu.h"
static int clampi8(long v){ if(v>127)v=127; if(v<-128)v=-128; return (int)v; }
int main(int argc,char**argv){
    int mult=argc>1?atoi(argv[1]):0x4000, shift=argc>2?atoi(argv[2]):14;
    unsigned bscale=argc>3?(unsigned)strtoul(argv[3],0,16):0x00004000;
    ork_npu*c=ork_npu_init(); if(!c){printf("SKIP\n");return 0;}
    if(!ork_ppu_fuse_enabled(c)){printf("SKIP\n");ork_npu_free(c);return 0;}
    const int M=8,N=64; static signed char a[512],b[512],out[512];
    for(int i=0;i<M*N;i++){ a[i]=(signed char)((i%61)-30); b[i]=(signed char)((i%37)-18); }
    double us=0;
    int r=ork_i8_npu_probe_add(c,a,b,M,N,mult,shift,bscale,0,0,0,out,&us);
    printf("add mult=0x%x shift=%d bscale=0x%x rc=%d (%.1f us)\n",mult,shift,bscale,r,us);
    if(r){ ork_npu_free(c); return 1; }
    int mism=0,mx=0;
    for(int i=0;i<M*N;i++){ int ref=clampi8((long)a[i]+b[i]); int d=abs((int)out[i]-ref); if(d>mx)mx=d; if(d>1)mism++;
        if(i<12) printf("  a=%4d b=%4d out=%4d a+b=%4d\n",a[i],b[i],out[i],ref); }
    printf("vs clamp(a+b): mism(|e|>1)=%d/%d max|e|=%d\n",mism,M*N,mx);
    ork_npu_free(c); return 0;
}
