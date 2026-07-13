/* ssd_fusedchain_probe — validate + time the multi-matmul real-operand fused chain (the SSD scan's
 * per-stage H-batch): nb fp16 matmuls (distinct data) in ONE PC-chained submit via
 * ork_bmm_fp16_fused, vs a CPU reference. Also times fused (1 submit) vs nb separate submits
 * (ork_ssd_probe_fusedmm_f16) to show the ~48us/submit floor amortization. Board only.
 *   make ssd_fusedchain_probe && sudo ./ssd_fusedchain_probe [nb] [M] [K] [N]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec*1e-3; }

int main(int argc,char**argv){
    int nb=argc>1?atoi(argv[1]):8, M=argc>2?atoi(argv[2]):64, K=argc>3?atoi(argv[3]):128, N=argc>4?atoi(argv[4]):64;
    ork_npu*c=ork_npu_init(); if(!c){ fprintf(stderr,"no NPU — skip\n"); return 0; }
    ork_f16 *A=malloc((size_t)nb*M*K*sizeof(ork_f16)), *B=malloc((size_t)nb*K*N*sizeof(ork_f16));
    float *C=malloc((size_t)nb*M*N*sizeof(float)), *ref=malloc((size_t)nb*M*N*sizeof(float));
    srand(20260712);
    for(size_t i=0;i<(size_t)nb*M*K;i++) A[i]=(ork_f16)(((double)rand()/RAND_MAX)*2-1);
    for(size_t i=0;i<(size_t)nb*K*N;i++) B[i]=(ork_f16)(((double)rand()/RAND_MAX)*2-1);
    for(int b=0;b<nb;b++) for(int m=0;m<M;m++) for(int n=0;n<N;n++){ double a=0;
        for(int k=0;k<K;k++) a+=(double)A[((size_t)b*M+m)*K+k]*(double)B[((size_t)b*K+k)*N+n];
        ref[((size_t)b*M+m)*N+n]=(float)a; }
    int rc=ork_bmm_fp16_fused(c,nb,M,K,N,A,B,C);
    int fail;
    if(rc){ fprintf(stderr,"fusedchain rc=%d\n",rc); fail=1; }
    else {
        double num=0,den=0; int worstb=0; double worst=0;
        for(int b=0;b<nb;b++){ double bn=0,bd=0; for(size_t i=0;i<(size_t)M*N;i++){ size_t j=(size_t)b*M*N+i; double e=C[j]-ref[j]; bn+=e*e; bd+=ref[j]*ref[j]; }
            double br=bd>0?sqrt(bn/bd):0; if(br>worst){worst=br;worstb=b;} num+=bn; den+=bd; }
        double rl2=den>0?sqrt(num/den):0;
        fail=(rl2>3e-2);
        fprintf(stderr,"[fusedchain] nb=%d [%d,%d]x[%d,%d] ONE submit: rel-L2=%.3e (worst batch %d=%.3e)  %s\n",
                nb,M,K,K,N,rl2,worstb,worst, fail?"FAIL":"OK(all matmuls correct in one chained submit)");
        /* timing: fused chain vs nb separate submits */
        double f0=now_us(); for(int r=0;r<20;r++) ork_bmm_fp16_fused(c,nb,M,K,N,A,B,C); double fus=(now_us()-f0)/20;
        double p0=now_us(); for(int r=0;r<5;r++) for(int b=0;b<nb;b++){ float cc[64*64]; (void)cc;
            ork_ssd_probe_fusedmm_f16(c,M,K,N,A+(size_t)b*M*K,B+(size_t)b*K*N,C+(size_t)b*M*N); } double per=(now_us()-p0)/5;
        fprintf(stderr,"  timing: fused(1 submit, %d mm)=%.1fus | per-op(%d submits)=%.1fus | amortization %.2fx\n",
                nb,fus,nb,per, per>0?per/fus:0);
    }
    free(A);free(B);free(C);free(ref); ork_npu_free(c);
    fprintf(stderr, fail?"\nSSD_FUSEDCHAIN_PROBE: FAIL\n":"\nSSD_FUSEDCHAIN_PROBE: PASS\n");
    return fail?1:0;
}
