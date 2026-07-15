/* stridedA_bmm_probe — the densify-drop primitive: fp16 matmul reads a ZERO-COPY strided activation (A stored
 * at row pitch>K in a DMA buffer, read via CNA LINE_STRIDE, no CPU gather) — the permuted-Q/KV-cache-view case.
 * out[m][n]=Σ_k A[m][k]·B[k][n], validated vs contiguous ref. BOARD: sudo env ORK_EW_TIMEOUT=1500 ./stridedA_bmm_probe */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    if(!ork_ppu_fuse_enabled(c)){printf("SKIP\n");ork_npu_free(c);return 0;}
    /* attention-shaped: M=queries, K=head_dim, N=out; apitch = K strided into a wider tensor (e.g. GQA/packed heads) */
    struct { int M,K,N,pitch; } t[]={{8,32,64,64},{16,64,128,256},{32,128,256,512}};
    int fail=0;
    for(int i=0;i<3;i++){ int M=t[i].M,K=t[i].K,N=t[i].N,P=t[i].pitch;
        ork_f16*A=malloc((size_t)M*K*2),*B=malloc((size_t)K*N*2),*out=malloc((size_t)M*N*2);
        unsigned s=7+i;
        for(int j=0;j<M*K;j++){s=s*1103515245+12345;A[j]=(ork_f16)((s>>16)&1);}
        for(int j=0;j<K*N;j++){s=s*1103515245+12345;B[j]=(ork_f16)((s>>16)&1);}
        int rc=ork_npu_probe_f16_stridedA(c,M,K,N,(unsigned short*)A,P,(unsigned short*)B,(unsigned short*)out);
        int bad=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++){ float acc=0; for(int k=0;k<K;k++) acc+=(float)A[(size_t)m*K+k]*(float)B[(size_t)k*N+n];
            if((float)out[(size_t)m*N+n]!=acc) bad++; }
        printf("  M=%d K=%d N=%d apitch=%d: rc=%d %d/%d exact %s\n",M,K,N,P,rc,M*N-bad,M*N,(rc==0&&!bad)?"OK":"CHECK"); if(rc||bad)fail=1;
        free(A);free(B);free(out); }
    printf("zero-copy strided-A fp16 matmul (densify-drop primitive): %s\n",fail?"CHECK":"ALL OK — reads strided view in place, no gather");
    ork_npu_free(c); return fail;
}
