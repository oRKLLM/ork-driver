/* mm_perchan_f16_fused_probe — the SINGLE-SUBMIT close: fp16 matmul with per-channel scale FUSED into the
 * output stage (one task, no SDP, no layout bridge). out[m][n] = (Σ_k A[m][k]B[k][n]) * scale[n], all on-NPU.
 * Integer-valued fp16 inputs keep every value exact so an exact match proves the fused value path.
 * BOARD: sudo env ORK_EW_TIMEOUT=1500 ./mm_perchan_f16_fused_probe */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    if(!ork_ppu_fuse_enabled(c)){printf("PPU fuse not enabled — SKIP\n");ork_npu_free(c);return 0;}
    int M=8, K=32, N=64;
    ork_f16*A=malloc((size_t)M*K*2),*B=malloc((size_t)K*N*2),*scale=malloc((size_t)N*2),*out=malloc((size_t)M*N*2);
    unsigned s=5;
    for(int i=0;i<M*K;i++){ s=s*1103515245+12345; A[i]=(ork_f16)((s>>16)&1); }
    for(int i=0;i<K*N;i++){ s=s*1103515245+12345; B[i]=(ork_f16)((s>>16)&1); }
    for(int n=0;n<N;n++) scale[n]=(ork_f16)(n%3);
    int rc=ork_f16_npu_mm_perchan_fused(c,M,K,N,(unsigned short*)A,(unsigned short*)B,(unsigned short*)scale,(unsigned short*)out);
    int bad=0,nz=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ float acc=0; for(int k=0;k<K;k++) acc+=(float)A[(size_t)m*K+k]*(float)B[(size_t)k*N+n];
        float ref=acc*(float)scale[n]; float got=(float)out[(size_t)m*N+n]; if(got!=0)nz++;
        if(got!=ref){ if(bad<4)printf("  [%d][%d] NPU=%g ref=%g (acc=%g scale=%g)\n",m,n,got,ref,acc,(float)scale[n]); bad++; } }
    printf("fp16 fused matmul*perchan (1 submit): rc=%d  %d/%d exact  (%d nonzero)  %s\n",rc,M*N-bad,M*N,nz,
           (rc==0&&!bad)?"OK — single-submit per-channel-scaled matmul on NPU":"CHECK");
    ork_npu_free(c);
    return (rc==0&&!bad)?0:1;
}
