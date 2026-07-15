/* requant_i32_probe — LOOPBACK Pass-2 crux test: can the standalone SDP read INT32 (a matmul accumulator)
 * from DRAM, apply a per-channel scale, and requant to int16 — self-completing on the enable=0x18 SDP path
 * (a DIFFERENT physical block than the wedged CNA->DPU requant-WDMA)?  If yes, the loopback architecture
 * (int32 matmul -> int32 DRAM -> this SDP requant -> int16) is a pure-NPU O(M.N) route that routes around
 * the narrow-output-matmul stall.  Integer-valued inputs keep every value exact so a match proves the path.
 * BOARD: sudo env ORK_EW_TIMEOUT=1500 ORK_RQ_DUMP=1 ./requant_i32_probe
 *        sweep precision:  sudo env ... ORK_RQ_4010=0x30000001 ORK_RQ_MSTRIDE=<hex> ./requant_i32_probe */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int one(ork_npu*c,int M,int N,int mult,int shift){
    int *a=malloc((size_t)M*N*4); short *b=malloc((size_t)N*2); short *out=malloc((size_t)M*N*2);
    unsigned s=7;
    int big=getenv("ORK_RQ_BIG")!=0;
    if(big){ for(int i=0;i<M*N;i++) a[i]=(i%N+1)*20000;   /* LARGE int32 (>2^16 for n>=4): expose low/high split */
             for(int n=0;n<N;n++) b[n]=1; mult=16384; shift=14; }               /* b=1, x1.0 -> out should == a */
    else { for(int i=0;i<M*N;i++){ s=s*1103515245+12345; a[i]=(int)((s>>18)%64)-16; }  /* small int32 accumulators */
           for(int n=0;n<N;n++){ s=s*1103515245+12345; b[n]=(short)((s>>20)%5); } }     /* per-channel scale 0..4 */
    double us=0; int rc=ork_npu_requant_perchan_i32(c,a,b,M,N,mult,shift,out,&us);
    if(big){ printf("  BIG raw out[0][0..15] (expect even=low16(a), odd=high16(a) if int16-lane-split):\n   ");
        for(int n=0;n<16;n++) printf(" %d",(unsigned short)out[n]);
        printf("\n   a[0][0..7]="); for(int n=0;n<8;n++) printf(" %d(lo=%d,hi=%d)",a[n],a[n]&0xffff,(a[n]>>16)&0xffff); printf("\n"); }
    int bad=0,nz=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){
        long prod=(long)a[(size_t)m*N+n]*(long)b[n]*(long)mult; long ref=prod>>shift;
        if(ref>32767)ref=32767; if(ref<-32768)ref=-32768;
        int got=out[(size_t)m*N+n]; if(got)nz++;
        if(got!=(int)ref){ if(bad<4)printf("  [%d][%d] NPU=%d ref=%ld (a=%d b=%d)\n",m,n,got,ref,a[(size_t)m*N+n],b[n]); bad++; } }
    printf("  M=%d N=%d mult=%d shift=%d: rc=%d %d/%d exact (%d nz) %.0fus  %s\n",
           M,N,mult,shift,rc,M*N-bad,M*N,nz,us,(rc==0&&!bad)?"OK":(rc?"SUBMIT-FAIL":"LAYOUT"));
    free(a);free(b);free(out);
    return (rc==0&&!bad)?0:1;
}
int main(int argc,char**argv){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    if(!ork_ppu_fuse_enabled(c)){printf("PPU fuse not enabled — SKIP\n");ork_npu_free(c);return 0;}
    int bad=0;
    if(argc>=3){ int M=atoi(argv[1]),N=atoi(argv[2]); bad|=one(c,M,N,16384,14); ork_npu_free(c); return bad; }
    bad|=one(c,8,64,16384,14);      /* mult/shift = *1.0 (scale carries the value) */
    bad|=one(c,16,128,16384,14);
    bad|=one(c,32,256,16384,14);
    printf("requant_i32 (loopback Pass-2): %s\n",bad?"CHECK — int32-input SDP not (yet) a clean datapath":"ALL OK — SDP requants int32->int16, loopback viable");
    ork_npu_free(c);
    return bad;
}
