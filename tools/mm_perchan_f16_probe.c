/* mm_perchan_f16_probe — the CLOSE: per-channel-scaled fp16 matmul fully on NPU (bit-exact), composing the
 * proven fp16 matmul (contiguous fp16 out) + the atom-8 per-channel EW-mul SDP. out[m][n]=(Σ_k A B)*scale[n].
 * The vendor's own structure (plain fp16 matmul -> separate fp16 per-channel SDP). Integer-valued fp16 inputs
 * keep every value exact. BOARD: sudo env ORK_EW_TIMEOUT=1500 ./mm_perchan_f16_probe */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    if(!ork_ppu_fuse_enabled(c)){printf("PPU fuse not enabled — SKIP\n");ork_npu_free(c);return 0;}
    int shapes[][3]={{8,32,64},{16,64,128},{32,128,256}};
    int fail=0;
    for(int s=0;s<3;s++){
        int M=shapes[s][0],K=shapes[s][1],N=shapes[s][2];
        ork_f16*A=malloc((size_t)M*K*2),*B=malloc((size_t)K*N*2),*scale=malloc((size_t)N*2),*out=malloc((size_t)M*N*2);
        unsigned st=5+s;
        for(int i=0;i<M*K;i++){ st=st*1103515245+12345; A[i]=(ork_f16)((st>>16)&1); }
        for(int i=0;i<K*N;i++){ st=st*1103515245+12345; B[i]=(ork_f16)((st>>16)&1); }
        for(int n=0;n<N;n++) scale[n]=(ork_f16)(n%3);
        double us=0;
        int rc=ork_f16_npu_mm_perchan(c,M,K,N,(unsigned short*)A,(unsigned short*)B,(unsigned short*)scale,(unsigned short*)out,&us);
        int bad=0;
        for(int m=0;m<M;m++)for(int n=0;n<N;n++){ float acc=0; for(int k=0;k<K;k++) acc+=(float)A[(size_t)m*K+k]*(float)B[(size_t)k*N+n];
            float ref=acc*(float)scale[n]; if((float)out[(size_t)m*N+n]!=ref) bad++; }
        printf("  MKN=%d,%d,%d: rc=%d  %d/%d exact  %.0f us  %s\n",M,K,N,rc,M*N-bad,M*N,us,(rc==0&&!bad)?"OK":"CHECK");
        if(rc||bad)fail=1;
        free(A);free(B);free(scale);free(out);
    }
    printf("per-channel-scaled fp16 matmul on NPU: %s\n",fail?"CHECK":"ALL OK — bit-exact, on-NPU");
    ork_npu_free(c);
    return fail;
}
