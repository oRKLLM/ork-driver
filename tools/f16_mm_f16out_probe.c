/* f16_mm_f16out_probe — isolate the fp16-matmul->fp16-output bridge (NO chain, task_number=1). Confirms the
 * fp16 matmul can emit correct fp16 G in the EWCUBEH atom-8 layout the chained SDP reads. If this passes but
 * the chain hangs, the fault is the handoff; if this zeros/hangs, the fault is the fp16 output stage.
 * BOARD: sudo env ORK_EW_TIMEOUT=2000 ./f16_mm_f16out_probe */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    if(!ork_ppu_fuse_enabled(c)){printf("PPU fuse not enabled — SKIP\n");ork_npu_free(c);return 0;}
    int M=8, K=32, N=64;
    ork_f16*A=malloc((size_t)M*K*2),*B=malloc((size_t)K*N*2),*out=malloc((size_t)M*N*2);
    unsigned s=5;
    for(int i=0;i<M*K;i++){ s=s*1103515245+12345; A[i]=(ork_f16)((s>>16)&1); }
    for(int i=0;i<K*N;i++){ s=s*1103515245+12345; B[i]=(ork_f16)((s>>16)&1); }
    int rc=ork_npu_probe_f16_mm_f16out(c,M,K,N,(unsigned short*)A,(unsigned short*)B,(unsigned short*)out);
    int bad=0,nz=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ float acc=0; for(int k=0;k<K;k++) acc+=(float)A[(size_t)m*K+k]*(float)B[(size_t)k*N+n];
        float got=(float)out[(size_t)m*N+n]; if(got!=0)nz++;
        if(got!=acc){ if(bad<4)printf("  [%d][%d] NPU=%g ref=%g\n",m,n,got,acc); bad++; } }
    printf("fp16 matmul->fp16-out (standalone): rc=%d  %d/%d exact  (%d nonzero)  %s\n",rc,M*N-bad,M*N,nz,
           (rc==0&&!bad)?"OK — fp16-out bridge correct":"CHECK");
    ork_npu_free(c);
    return (rc==0&&!bad)?0:1;
}
