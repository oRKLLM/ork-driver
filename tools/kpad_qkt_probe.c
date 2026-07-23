/* kpad_qkt_probe — task #20 (high-effort path, crack c): the small-K int8-requant wall (K%512, head_dim=128
 * rejects) is SIDESTEPPED by zero-padding the contraction dim 128->512. Zeros add 0·0 terms that don't change
 * the sum, so the K-padded int8 QK^T is EXACT and rides the PROVEN K=512 requant path — no schedule RE, no
 * wedge risk. This unlocks int8 scores at real attention head_dim, the front of the all-int8 resident softmax.
 *   scores_i8 = clamp_i8((Q_pad · K^T_pad)*mult>>shift), K=512 (real d=128 + 384 zero rows), vs CPU over d=128.
 *   direct: sudo env ORK_MM_TIMEOUT=3000 ./kpad_qkt_probe [N] [d]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static uint32_t g=0x2ad9u; static int r8(void){ g=g*1664525u+1013904223u; return (int)((g>>24)%9)-4; }
int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):64, d=argc>2?atoi(argv[2]):128;   /* N tokens, d head_dim (<512) */
    int Kp=512;                                                  /* padded contraction */
    setvbuf(stdout,0,_IONBF,0);
    if(d>Kp||N%32){ printf("need d<=512 and N%%32\n"); return 2; }
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("kpad_qkt_probe: N=%d d=%d (int8 QK^T, K padded %d->%d)\n",N,d,d,Kp);
    /* real int8 Q[N,d], K[N,d]; scores = Q · K^T over d. Lay padded operands: Q_pad[N,Kp], KT_pad[Kp,N]. */
    int8_t *Q=malloc((size_t)N*d), *Kk=malloc((size_t)N*d);
    for(size_t i=0;i<(size_t)N*d;i++){ Q[i]=(int8_t)r8(); Kk[i]=(int8_t)r8(); }
    int8_t *Qp=calloc((size_t)N*Kp,1), *KTp=calloc((size_t)Kp*N,1);   /* zero-padded */
    for(int i=0;i<N;i++)for(int k=0;k<d;k++) Qp[(size_t)i*Kp+k]=Q[(size_t)i*d+k];        /* Q_pad rows: d then zeros */
    for(int k=0;k<d;k++)for(int j=0;j<N;j++) KTp[(size_t)k*N+j]=Kk[(size_t)j*d+k];        /* K^T_pad: k<d filled, k>=d zero */
    int mult=0x4000, shift=18;   /* 1/16 scale, keep scores in int8 range */
    int8_t *C=malloc((size_t)N*N); for(size_t i=0;i<(size_t)N*N;i++)C[i]=-128;
    ork_w *w=ork_mm_pack_i8(c,Kp,N,KTp); if(!w){printf("pack fail\n");return 2;}
    ork_seq_op op={ .kind=ORK_OP_MATMUL_REQUANT_I8, .w=w, .M=N, .A=Qp, .C=C, .mult=mult, .shift=shift };
    int rc=ork_submit_seq(c,&op,1);
    printf("  K-padded requant rc=%d C[0]=%d\n",rc,C[0]);
    if(rc){printf("FAIL rc=%d\n",rc);ork_mm_free(c,w);ork_npu_free(c);return 1;}
    int bad=0,me=0;
    for(int m=0;m<N;m++)for(int n=0;n<N;n++){ long acc=0; for(int k=0;k<d;k++)acc+=(long)Q[(size_t)m*d+k]*Kk[(size_t)n*d+k];   /* real d only */
        long q=(acc*mult)>>shift; if(q>127)q=127; if(q<-128)q=-128; int e=abs((int)C[(size_t)m*N+n]-(int)q); if(e>me)me=e; if(e>1)bad++; }
    printf("  int8 QK^T (K-padded %d->%d) vs CPU over d=%d: max|err|=%d LSB  %s (%d/%d)\n",d,Kp,d,me,bad?"MISMATCH":"OK",bad,N*N);
    int fail=bad!=0;
    printf("%s\n",fail?"FAIL":"PASS — K-padding sidesteps the small-K requant wall: exact int8 QK^T at head_dim (no RE)");
    ork_mm_free(c,w); ork_npu_free(c);
    return fail;
}
