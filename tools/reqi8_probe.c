/* reqi8_probe — task #20 (A1): validate MATMUL_REQUANT_I8 (int8 matmul -> int8 requant out) as a seq op.
 * The int8 x-max broadcast + any int8 matmul->SDP feed for the all-int8 softmax island. C=clamp_i8((A·W)*mult>>shift).
 *   direct: sudo env ORK_MM_TIMEOUT=3000 ./reqi8_probe ; orkd: + ORK_USE_ORKD=1 ORKD_BIN=./orkd
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static uint32_t g=0x1bd3u; static int r8(void){ g=g*1664525u+1013904223u; return (int)((g>>24)%15)-7; }
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):32, K=argc>2?atoi(argv[2]):64, N=argc>3?atoi(argv[3]):64;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    int viaorkd=getenv("ORK_USE_ORKD")?1:0;
    printf("reqi8_probe: M=%d K=%d N=%d path=%s\n",M,K,N,viaorkd?"ORKD":"direct");
    int8_t *A=malloc((size_t)M*K), *B=malloc((size_t)K*N), *C=malloc((size_t)M*N);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(int8_t)r8();
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=(int8_t)r8();
    for(size_t i=0;i<(size_t)M*N;i++)C[i]=-128;
    int mult=0x4000, shift=18;   /* coeff = mult/2^shift = 2^14/2^18 = 1/16 (keep A·W in int8 range) */
    ork_w *w=ork_mm_pack_i8(c,K,N,B); if(!w){printf("pack fail\n");return 2;}
    ork_seq_op op={ .kind=ORK_OP_MATMUL_REQUANT_I8, .w=w, .M=M, .A=A, .C=C, .mult=mult, .shift=shift };
    int rc=ork_submit_seq(c,&op,1);
    printf("  rc=%d C[0]=%d\n",rc,C[0]);
    if(rc){printf("FAIL rc=%d\n",rc);ork_npu_free(c);return 1;}
    int bad=0,me=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long acc=0; for(int k=0;k<K;k++)acc+=(long)A[(size_t)m*K+k]*B[(size_t)k*N+n];
        long q=(acc*mult)>>shift; if(q>127)q=127; if(q<-128)q=-128; int e=abs((int)C[(size_t)m*N+n]-(int)q); if(e>me)me=e; if(e>1)bad++; }
    printf("  MATMUL_REQUANT_I8: max|err|=%d LSB  %s (%d/%d)\n",me,bad?"MISMATCH":"OK",bad,M*N);
    int fail=bad!=0;
    printf("%s\n",fail?"FAIL":"PASS — MATMUL_REQUANT_I8 seq op (int8 matmul->int8 out) coherent");
    ork_mm_free(c,w); ork_npu_free(c);
    return fail;
}
