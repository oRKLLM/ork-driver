/* tools/slice_probe.c — hunt the weight stride register for in-place K-slicing of a full-K buffer.
 *
 * Decode wants a full-K weight layout (one submit, K<=10752); prefill wants K-split. Keeping both
 * resident doubles NPU memory and exhausts the IOVA range (see Performance wiki). The fix: ONE
 * full-K layout that prefill slices in place. Problem: when an op processes a Kp-slice of a full-K
 * buffer, the engine derives the per-N-tile stride from the op's K (=Kp), so N-tile 0 reads right
 * but N-tiles 1+ land in the wrong place. This probe packs B[Kfull,N] full-K, runs a Kp-slice, and
 * classifies each N-tile's output as PARTIAL (== sum k<Kp, correct slice), FULL (== sum k<Kfull,
 * the override made the op compute all K), or WRONG. Goal: an override making ALL N-tiles PARTIAL.
 *
 *   make slice_probe && sudo ./slice_probe [reg val]...
 * e.g.  sudo ./slice_probe                 # baseline: expect N-tile 0 PARTIAL, 1+ WRONG
 *       sudo ./slice_probe 0x1044 128       # set K-passes reg to full Kfull/32
 *       sudo ./slice_probe 0x1034 8192      # set 0x1034 to Kfull*2
 *       sudo ./slice_probe 0x1030 524288    # set 0x1030 to Kfull*N*2
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ork_npu.h"
typedef ork_f16 f16;
static unsigned sd=7; static float r1(void){sd=sd*1103515245+12345;return (float)((int)((sd>>16)%5)-2);}
int main(int argc,char**argv){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed (NPU?)\n");return 1;}
    int Kfull=4096,N=64,Kp=2048;
    f16*A=malloc((size_t)Kfull*2),*B=malloc((size_t)Kfull*N*2); float*C=malloc((size_t)N*4);
    float*part=malloc((size_t)N*4),*full=malloc((size_t)N*4);
    for(int i=0;i<Kfull;i++)A[i]=(f16)r1();
    for(size_t i=0;i<(size_t)Kfull*N;i++)B[i]=(f16)r1();
    for(int n=0;n<N;n++){float p=0,f=0;
        for(int k=0;k<Kfull;k++){float t=(float)A[k]*(float)B[(size_t)k*N+n];f+=t;if(k<Kp)p+=t;}
        part[n]=p;full[n]=f;}
    uint32_t regs[4],vals[4]; int nov=0;
    for(int i=1;i+1<argc && nov<4;i+=2){regs[nov]=(uint32_t)strtoul(argv[i],0,0);vals[nov]=(uint32_t)strtoul(argv[i+1],0,0);nov++;}
    printf("Kfull=%d N=%d Kp=%d (KTfull=%d op-KT=%d, %d N-tiles)  overrides:",Kfull,N,Kp,Kfull/32,Kp/32,N/16);
    for(int i=0;i<nov;i++)printf(" 0x%x=0x%x",regs[i],vals[i]); printf("\n");
    int rc=ork_f16_npu_probe_slice(c,Kfull,N,Kp,nov,regs,vals,A,B,C);
    if(rc){printf("submit failed/wedged (rc=%d)\n",rc);ork_npu_free(c);return 1;}
    int allpart=1;
    for(int nt=0;nt<N/16;nt++){int p=0,f=0;
        for(int n=nt*16;n<nt*16+16;n++){ if(C[n]==part[n])p++; else if(C[n]==full[n])f++; }
        const char*cls=p==16?"PARTIAL (correct slice)":f==16?"FULL (op computed all K)":"WRONG";
        if(p!=16)allpart=0;
        printf("  N-tile %d (cols %3d-%3d): %-24s C[%d]=%.0f  part=%.0f full=%.0f\n",
               nt,nt*16,nt*16+15,cls,nt*16,C[nt*16],part[nt*16],full[nt*16]);
    }
    printf("%s\n",allpart?"ALL PARTIAL — this override slices the full-K buffer correctly":"not yet");
    free(A);free(B);free(C);free(part);free(full); ork_npu_free(c); return 0;
}
