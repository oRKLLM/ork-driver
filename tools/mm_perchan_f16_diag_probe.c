/* mm_perchan_f16_diag_probe — PURE-NPU per-channel-scaled fp16 matmul via a diagonal 2nd matmul (no SDP, no
 * reshape, no CPU repack; the 2nd matmul reads G contiguous natively). out[m][n]=(Σ_k A·B)*scale[n].
 * BOARD: sudo env ORK_EW_TIMEOUT=1500 ./mm_perchan_f16_diag_probe */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    if(!ork_ppu_fuse_enabled(c)){printf("SKIP\n");ork_npu_free(c);return 0;}
    int sh[][3]={{8,32,64},{16,64,128},{32,128,256}}; int fail=0;
    for(int s=0;s<3;s++){ int M=sh[s][0],K=sh[s][1],N=sh[s][2];
        ork_f16*A=malloc((size_t)M*K*2),*B=malloc((size_t)K*N*2),*sc=malloc((size_t)N*2),*out=malloc((size_t)M*N*2);
        unsigned st=5+s;
        for(int i=0;i<M*K;i++){st=st*1103515245+12345;A[i]=(ork_f16)((st>>16)&1);}
        for(int i=0;i<K*N;i++){st=st*1103515245+12345;B[i]=(ork_f16)((st>>16)&1);}
        for(int n=0;n<N;n++) sc[n]=(ork_f16)(n%3);
        double us=0; int rc=ork_f16_npu_mm_perchan_diag(c,M,K,N,(unsigned short*)A,(unsigned short*)B,(unsigned short*)sc,(unsigned short*)out,&us);
        int bad=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++){ float acc=0; for(int k=0;k<K;k++) acc+=(float)A[(size_t)m*K+k]*(float)B[(size_t)k*N+n];
            float ref=acc*(float)sc[n]; if((float)out[(size_t)m*N+n]!=ref) bad++; }
        printf("  MKN=%d,%d,%d: rc=%d %d/%d exact %s\n",M,K,N,rc,M*N-bad,M*N,(rc==0&&!bad)?"OK":"CHECK"); if(rc||bad)fail=1;
        free(A);free(B);free(sc);free(out); }
    printf("diagonal-matmul per-channel scale (pure NPU): %s\n",fail?"CHECK":"ALL OK — pure-NPU, bit-exact");
    ork_npu_free(c); return fail;
}
