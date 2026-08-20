/* tnorm_probe — validate the TRANSPOSED-NORMALIZE attention math on-NPU (the piece that puts the softmax
 * normalize on the NPU as a native per-channel scale). Reference: out[m][d] = (Σ_j e[m][j] V[j][d]) / Σ[m],
 * Σ[m]=Σ_j e[m][j]. NPU path (queries m become the output channel):
 *   Ô[d][m] = Σ_j V[j][d] e[m][j]  = matmul( A=V^T[DV][K],  W=e^T[K][N] )   (ork_mm, fp16)
 *   out[m][d] = Ô[d][m] * (1/Σ)[m]  via ork_f16_npu_mul_perchan (1/Σ per-query = per-channel)
 * Compares to the CPU reference. BOARD: sudo ./tnorm_probe */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    int N=16, K=64, DV=64;                       /* N queries, K keys, DV head-dim */
    float *e=malloc((size_t)N*K*4), *V=malloc((size_t)K*DV*4);
    unsigned s=13; for(int i=0;i<N*K;i++){ s=s*1103515245+12345; e[i]=((s>>16)&0xff)/255.0f; }   /* e in [0,1] */
    for(int i=0;i<K*DV;i++){ s=s*1103515245+12345; V[i]=(((int)((s>>16)&0xff))-128)/128.0f; }      /* V in [-1,1] */
    double *Sig=malloc(N*sizeof(double));
    for(int m=0;m<N;m++){ double sm=0; for(int j=0;j<K;j++) sm+=e[(size_t)m*K+j]; Sig[m]=sm; }
    float *ref=malloc((size_t)N*DV*4);
    for(int m=0;m<N;m++)for(int d=0;d<DV;d++){ double a=0; for(int j=0;j<K;j++) a+=(double)e[(size_t)m*K+j]*V[(size_t)j*DV+d]; ref[(size_t)m*DV+d]=(float)(a/Sig[m]); }

    /* NPU: W=e^T[K][N] (fp16 pack), A=V^T[DV][K] (fp16). C=Ô[DV][N] fp32. */
    ork_f16 *eT=malloc((size_t)K*N*2), *VT=malloc((size_t)DV*K*2);
    for(int j=0;j<K;j++)for(int m=0;m<N;m++) eT[(size_t)j*N+m]=(ork_f16)e[(size_t)m*K+j];
    for(int d=0;d<DV;d++)for(int j=0;j<K;j++) VT[(size_t)d*K+j]=(ork_f16)V[(size_t)j*DV+d];
    ork_w *w=ork_f16_mm_pack(c,K,N,eT); if(!w){printf("pack failed\n");return 2;}
    float *Ohat=malloc((size_t)DV*N*4);
    if(ork_f16_mm_run(c,w,DV,VT,Ohat)){printf("mm_run failed\n");return 2;}
    /* per-channel 1/Σ over Ô[DV][N] (N=queries=channels) */
    ork_f16 *Oh16=malloc((size_t)DV*N*2), *inv=malloc(N*2), *Onorm=malloc((size_t)DV*N*2);
    for(int i=0;i<DV*N;i++) Oh16[i]=(ork_f16)Ohat[i];
    for(int m=0;m<N;m++) inv[m]=(ork_f16)(1.0/Sig[m]);
    int rc=ork_f16_npu_mul_perchan(c,Oh16,inv,DV,N,Onorm,NULL);
    double maxerr=0; int bad=0;
    for(int m=0;m<N;m++)for(int d=0;d<DV;d++){ float g=(float)Onorm[(size_t)d*N+m]; float r=ref[(size_t)m*DV+d]; double er=fabs(g-r); if(er>maxerr)maxerr=er; if(er>0.02)bad++; }
    printf("transposed-normalize: rc=%d  max|err|=%.4f  bad=%d/%d  %s\n",rc,maxerr,bad,N*DV,(rc==0&&bad==0)?"OK":"CHECK");
    ork_npu_free(c);
    return (rc==0&&bad==0)?0:1;
}
