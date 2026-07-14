/* chain_mm_perchan_probe — validate the FIRST attention sub-chain: int8-matmul(int16-out) ->
 * per-channel-scale, ONE PC-chain submit (A·V->normalize pattern). out[m][n] = (Σ_k A[m][k]B[k][n]) * scale[n].
 * Small values (no clamp), gain=1 (m=0x4000,s=14). Validates the intermediate-buffer bridge on-device.
 * BOARD: sudo ./chain_mm_perchan_probe */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    if(!ork_ppu_fuse_enabled(c)){printf("PPU fuse not enabled — SKIP\n");ork_npu_free(c);return 0;}
    int M=8, K=32, N=64;
    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); short*scale=malloc(N*2),*out=malloc((size_t)M*N*2);
    unsigned s=5;
    for(int i=0;i<M*K;i++){ s=s*1103515245+12345; A[i]=(int8_t)((s>>16)&1); }        /* 0/1 */
    for(int i=0;i<K*N;i++){ s=s*1103515245+12345; B[i]=(int8_t)((s>>16)&1); }        /* 0/1 */
    for(int n=0;n<N;n++) scale[n]=(short)(n%3);                                        /* 0..2 per channel */
    double us=0;
    int rc=ork_npu_chain_mm_perchan_i16(c,M,K,N,A,B,scale,0x4000,14,0x4000,14,out,&us);
    int bad=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int acc=0; for(int k=0;k<K;k++) acc+=(int)A[(size_t)m*K+k]*(int)B[(size_t)k*N+n];
        int ref=acc*(int)scale[n]; if(out[(size_t)m*N+n]!=ref){ if(bad<4)printf("  [%d][%d] NPU=%d ref=%d (acc=%d scale=%d)\n",m,n,out[(size_t)m*N+n],ref,acc,scale[n]); bad++; } }
    printf("chain matmul(int16)->per-channel: rc=%d  %d/%d exact  %.0f us  %s\n",rc,M*N-bad,M*N,us,(rc==0&&!bad)?"OK — first attention sub-chain in ONE submit":"CHECK");
    ork_npu_free(c);
    return (rc==0&&!bad)?0:1;
}
