/* perchan_bench — head-to-head: SDP-path (matmul + CPU repack + atom-8 SDP, O(M.N)) vs DIAGONAL (matmul +
 * diag 2nd matmul, pure-NPU O(M.N^2)) for the per-channel-scaled fp16 matmul (attention A.V-normalize).
 * Fixes M=head_dim, K=seq; sweeps N (=queries in the transposed normalize) to find the crossover.
 * Both timed (us) + bit-exact-checked vs CPU (0/1 inputs keep fp16 exact). BOARD:
 *   sudo env ORK_EW_TIMEOUT=3000 ./perchan_bench [M] [K]        (defaults M=64 K=512) */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char**argv){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    if(!ork_ppu_fuse_enabled(c)){printf("PPU fuse not enabled — SKIP\n");ork_npu_free(c);return 0;}
    int M=argc>1?atoi(argv[1]):64, K=argc>2?atoi(argv[2]):512;
    int Ns[]={64,128,256,512,1024,2048}; int nN=sizeof(Ns)/sizeof(Ns[0]);
    printf("per-channel-scaled fp16 matmul: SDP-path vs DIAGONAL  (M=%d head_dim, K=%d seq)\n",M,K);
    printf("   N(queries) | SDP-path us | diagonal us | winner\n");
    for(int s=0;s<nN;s++){ int N=Ns[s];
        ork_f16*A=malloc((size_t)M*K*2),*B=malloc((size_t)K*N*2),*sc=malloc((size_t)N*2);
        ork_f16*o1=malloc((size_t)M*N*2),*o2=malloc((size_t)M*N*2);
        unsigned st=5+s;
        for(int i=0;i<M*K;i++){st=st*1103515245+12345;A[i]=(ork_f16)((st>>16)&1);}
        for(int i=0;i<K*N;i++){st=st*1103515245+12345;B[i]=(ork_f16)((st>>16)&1);}
        for(int n=0;n<N;n++) sc[n]=(ork_f16)(n%3);
        double us1=0,us2=0;
        int rc1=ork_npu_mm_perchan_f16(c,M,K,N,(unsigned short*)A,(unsigned short*)B,(unsigned short*)sc,(unsigned short*)o1,&us1);
        int rc2=ork_npu_mm_perchan_f16_diag(c,M,K,N,(unsigned short*)A,(unsigned short*)B,(unsigned short*)sc,(unsigned short*)o2,&us2);
        /* bit-exact vs CPU (verify a few rows to bound cost at large N*K) */
        int bad1=0,bad2=0; int MR=M<8?M:8;
        for(int m=0;m<MR;m++)for(int n=0;n<N;n++){ float acc=0; for(int k=0;k<K;k++) acc+=(float)A[(size_t)m*K+k]*(float)B[(size_t)k*N+n];
            float ref=acc*(float)sc[n];
            if((float)o1[(size_t)m*N+n]!=ref)bad1++; if((float)o2[(size_t)m*N+n]!=ref)bad2++; }
        const char*w = (rc1||rc2)?"(err)":(us1<us2?"SDP":"diag");
        printf("   %9d | %8.0f%s | %8.0f%s | %s\n", N,
               us1, (rc1||bad1)?"!":" ", us2, (rc2||bad2)?"!":" ", w);
        free(A);free(B);free(sc);free(o1);free(o2);
    }
    printf("   (! = error or mismatch; times are single-shot warm)\n");
    ork_npu_free(c); return 0;
}
