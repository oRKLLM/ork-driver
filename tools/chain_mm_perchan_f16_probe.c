/* chain_mm_perchan_f16_probe — CLOSE the attention A·V sub-chain with a fp16-IN matmul: ONE PC-chain submit
 * of fp16-matmul(fp16-out) -> per-channel-scale, all-fp16 (matches the vendor conv->mul dtype path that fixed
 * the chained 2-input-SDP hang). out[m][n] = (Σ_k A[m][k]B[k][n]) * scale[n], computed on-NPU end-to-end.
 * Inputs 0/1 + K=32 + scale 0..2 keep every value exact in fp16, so an exact match proves the value path.
 * BOARD: sudo env ORK_EW_TIMEOUT=1500 ./chain_mm_perchan_f16_probe */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    if(!ork_ppu_fuse_enabled(c)){printf("PPU fuse not enabled — SKIP\n");ork_npu_free(c);return 0;}
    int M=8, K=32, N=64;
    ork_f16*A=malloc((size_t)M*K*2),*B=malloc((size_t)K*N*2),*scale=malloc((size_t)N*2),*out=malloc((size_t)M*N*2);
    unsigned s=5;
    for(int i=0;i<M*K;i++){ s=s*1103515245+12345; A[i]=(ork_f16)((s>>16)&1); }        /* 0/1 */
    for(int i=0;i<K*N;i++){ s=s*1103515245+12345; B[i]=(ork_f16)((s>>16)&1); }        /* 0/1 */
    for(int n=0;n<N;n++) scale[n]=(ork_f16)(n%3);                                       /* 0..2 per channel */
    double us=0;
    int rc=ork_npu_chain_mm_perchan_f16(c,M,K,N,(unsigned short*)A,(unsigned short*)B,(unsigned short*)scale,(unsigned short*)out,&us);
    int bad=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ float acc=0; for(int k=0;k<K;k++) acc+=(float)A[(size_t)m*K+k]*(float)B[(size_t)k*N+n];
        float ref=acc*(float)scale[n]; float got=(float)out[(size_t)m*N+n];
        if(got!=ref){ if(bad<4)printf("  [%d][%d] NPU=%g ref=%g (acc=%g scale=%g)\n",m,n,got,ref,acc,(float)scale[n]); bad++; } }
    printf("fp16-in chain matmul->per-channel: rc=%d  %d/%d exact  %.0f us  %s\n",rc,M*N-bad,M*N,us,
           (rc==0&&!bad)?"OK — attention A·V normalize in ONE all-fp16 submit":"CHECK");
    ork_npu_free(c);
    return (rc==0&&!bad)?0:1;
}
