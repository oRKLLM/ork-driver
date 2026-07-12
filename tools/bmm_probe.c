/* 2a: the wedge is at M=1 (DECODE A·V), not prefill. Confirm the fp16 bmm wedges at M=1 across N/K. */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void){
    ork_npu*c=ork_npu_init(); if(!c) return 2;
    struct { int M,K,N; } cs[] = { {1,512,128}, {1,256,128}, {1,512,64}, {1,512,256}, {2,512,128}, {8,512,128} };
    for(unsigned i=0;i<sizeof(cs)/sizeof(cs[0]);i++){
        int M=cs[i].M,K=cs[i].K,N=cs[i].N;
        ork_f16 *A=malloc((size_t)M*K*sizeof(ork_f16)), *B=malloc((size_t)K*N*sizeof(ork_f16)); float *C=malloc((size_t)M*N*4);
        for(size_t j=0;j<(size_t)M*K;j++)A[j]=(ork_f16)0.01f; for(size_t j=0;j<(size_t)K*N;j++)B[j]=(ork_f16)0.01f;
        memset(C,0,(size_t)M*N*4); int r=ork_bmm_fp16(c,1,M,K,N,A,B,C);
        printf("M=%d K=%d N=%d : rc=%d C[0]=%.4f %s\n", M,K,N,r,C[0], r?"WEDGE":"OK");
        free(A);free(B);free(C);
    }
    ork_npu_free(c); return 0;
}
