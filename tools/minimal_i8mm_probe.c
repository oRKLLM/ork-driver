/* minimal_i8mm_probe — reproduce the reduce bug in the SIMPLEST safe form: a plain int8 matmul (MM_I8 via
 * ork_submit_seq, the exact path the softmax reduce uses) with an all-ones weight, at varying K. No LUT, no
 * exp, no requant -> should NOT wedge. C[m,n] = sum_k A[m,k]*1 = row-sum (identical across n). Compares vs CPU
 * to pinpoint which K (and M/N) miscomputes -- the softmax reduce is exactly this (e . ones[N_keys,32]).
 *   direct: sudo env ORK_MM_TIMEOUT=3000 timeout 120 ./minimal_i8mm_probe [M] [N]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static uint32_t g=0x13579u; static int r8(void){ g=g*1664525u+1013904223u; return (int)((g>>26)%50); }  /* [0,49] like exp out */
static int test_K(ork_npu*c,int M,int K,int N){
    int8_t *A=malloc((size_t)M*K), *W=malloc((size_t)K*N); memset(W,1,(size_t)K*N);   /* ones weight */
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)r8();
    ork_w *w=ork_mm_pack_i8(c,K,N,W); if(!w){ printf("  K=%-4d pack FAIL\n",K); free(A);free(W); return 1; }
    int32_t *C=malloc((size_t)M*N*4); for(size_t i=0;i<(size_t)M*N;i++)C[i]=0x7f7f7f7f;
    ork_seq_op op={ .kind=ORK_OP_MM_I8, .w=w, .M=M, .A=A, .C=C };
    int rc=ork_submit_seq(c,&op,1);
    int bad=0; long me=0, sample_npu=0, sample_cpu=0;
    if(rc){ printf("  K=%-4d N=%d rc=%d (submit fail)\n",K,N,rc); }
    else { for(int m=0;m<M;m++){ long S=0; for(int k=0;k<K;k++)S+=A[(size_t)m*K+k];   /* CPU row-sum */
             for(int n=0;n<N;n++){ long er=labs((long)C[(size_t)m*N+n]-S); if(er>me)me=er; if(er>0)bad++; }
             if(m==0){ sample_npu=C[0]; sample_cpu=S; } }
           printf("  K=%-4d N=%d rc=0 max|err|=%ld %s (%d/%d)  C[0]=%ld want %ld\n",K,N,me,bad?"MISMATCH":"OK",bad,M*N,sample_npu,sample_cpu); }
    ork_mm_free(c,w); free(A);free(W);free(C);
    return rc||bad;
}
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):64, N=argc>2?atoi(argv[2]):32;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("minimal_i8mm_probe: M=%d N=%d, ones weight, plain MM_I8 across K (softmax-reduce shape)\n",M,N);
    int fail=0, Ks[]={32,64,128,256,512,1024};
    for(unsigned i=0;i<sizeof(Ks)/sizeof(Ks[0]);i++) fail |= test_K(c,M,Ks[i],N);
    printf("%s\n", fail?"FAIL — some K miscomputes (reduce bug reproduced, plain matmul)":"PASS — plain int8 matmul correct across all K (reduce bug is NOT plain-matmul)");
    ork_npu_free(c);
    return fail;
}
