/* mul_perchan_f16_contig_probe — validate the vendor-derived CONTIGUOUS-read fp16 per-channel MUL SDP
 * (task13: FLYING_MODE + NOTCH). Reads a CONTIGUOUS [M][N] fp16 input (the native fp16 matmul output),
 * applies per-channel scale, so it can feed directly off ork_npu_probe_f16_mm_f16out with no repack.
 * out[m][n] = in[m][n] * scale[n]. BOARD: sudo env ORK_EW_TIMEOUT=1500 ./mul_perchan_f16_contig_probe */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    if(!ork_ppu_fuse_enabled(c)){printf("PPU fuse not enabled — SKIP\n");ork_npu_free(c);return 0;}
    int M=8, N=64;
    ork_f16*A=malloc((size_t)M*N*2),*scale=malloc((size_t)N*2),*out=malloc((size_t)M*N*2);
    unsigned s=5;
    for(int i=0;i<M*N;i++){ s=s*1103515245+12345; A[i]=(ork_f16)((s>>16)%7); }   /* 0..6 */
    for(int n=0;n<N;n++) scale[n]=(ork_f16)(n%3);                                  /* 0..2 per channel */
    double us=0;
    int rc=ork_npu_mul_perchan_f16_contig(c,A,scale,M,N,out,&us);
    int bad=0,nz=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ float ref=(float)A[(size_t)m*N+n]*(float)scale[n]; float got=(float)out[(size_t)m*N+n];
        if(got!=0)nz++; if(got!=ref){ if(bad<6)printf("  [%d][%d] NPU=%g ref=%g\n",m,n,got,ref); bad++; } }
    printf("contiguous-read fp16 per-channel mul: rc=%d  %d/%d exact  (%d nonzero)  %.0f us  %s\n",rc,M*N-bad,M*N,nz,us,
           (rc==0&&!bad)?"OK — SDP reads contiguous matmul output":"CHECK");
    ork_npu_free(c);
    return (rc==0&&!bad)?0:1;
}
