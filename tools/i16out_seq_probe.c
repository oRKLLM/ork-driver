/* i16out_seq_probe — task #20 (#3): int8 matmul -> INT16 out (MM_I8_OUT16) feeds int16 exp RESIDENT, proving
 * A2 resident-forwarding sidesteps the set_i16_out PC-chain issue (LINEAR int16 out IS the contiguous host
 * layout the int16 SDP reads — no cube bridge, no hardware chain).
 *   S1: MM_I8_OUT16 alone (scores_i16 = clamp_i16((A·W)*mult>>shift)) vs CPU.
 *   S2: [MM_I8_OUT16 -> EXP_I16] one seq, scores_i16 ALIASED resident -> validate exp vs CPU.
 * K=512 (int8 requant needs K>=512; also runs K=128 to test softmax head_dim viability).
 *   direct: sudo env ORK_MM_TIMEOUT=3000 ./i16out_seq_probe [M] [K] [N]  (+ORK_USE_ORKD=1 ORKD_BIN=./orkd)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static uint32_t g=0x71cd3u; static int r8(void){ g=g*1664525u+1013904223u; return (int)((g>>25)%11)-5; }
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):32, K=argc>2?atoi(argv[2]):512, N=argc>3?atoi(argv[3]):64;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    int viaorkd=getenv("ORK_USE_ORKD")?1:0;
    printf("i16out_seq_probe: M=%d K=%d N=%d path=%s\n",M,K,N,viaorkd?"ORKD":"direct");
    int mult=0x4000, shift=14;                    /* coeff 1 -> scores_i16 = clamp_i16(A·W) */
    double in_scale=1.0/512.0, out_scale=1.0/3200.0;
    /* A>=0, W<=0 => scores = A·W <= 0 (softmax x-max domain: exp of <=0 in (0,1], where exp_i16 is accurate) */
    int8_t *A=malloc((size_t)M*K), *B=malloc((size_t)K*N);
    for(size_t i=0;i<(size_t)M*K;i++){ g=g*1664525u+1013904223u; A[i]=(int8_t)((g>>26)%4); }        /* 0..3 */
    for(size_t i=0;i<(size_t)K*N;i++){ g=g*1664525u+1013904223u; B[i]=(int8_t)(-(int)((g>>26)%4)); }   /* -3..0 */
    short *scores=malloc((size_t)M*N*2), *e=malloc((size_t)M*N*2);
    for(size_t i=0;i<(size_t)M*N;i++){ scores[i]=-1; e[i]=-1; }
    ork_w *w=ork_mm_pack_i8(c,K,N,B); if(!w){printf("pack fail\n");return 2;}
    int fail=0;
    /* pre-calibrate the int16 LUT in a CLEAN context (before any matmul) — the documented fix for the
     * "int16-LUT calibration wedges as first op after a multi-core matmul" trap (orkd does this at startup). */
    { short wi[32],wo[32]; for(int i=0;i<32;i++) wi[i]=(short)(100+i); ork_npu_exp_i16(c,wi,1,32,in_scale,out_scale,wo,NULL); }

    /* S0: exp_i16 STANDALONE on CPU-computed int16 scores (isolates exp-on-these-values from the chain handoff) */
    { short *cs=malloc((size_t)M*N*2), *ce=malloc((size_t)M*N*2);
      for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long acc=0; for(int k=0;k<K;k++)acc+=(long)A[(size_t)m*K+k]*B[(size_t)k*N+n];
          long q=(acc*mult)>>shift; if(q>32767)q=32767; if(q<-32768)q=-32768; cs[(size_t)m*N+n]=(short)q; }
      int rc=ork_npu_exp_i16(c,cs,M,N,in_scale,out_scale,ce,NULL); int bad=0; double me=0;
      if(!rc) for(int i=0;i<M*N;i++){ double want=exp((double)cs[i]*in_scale)/out_scale; if(want>32767)want=32767;
          double er=fabs((double)ce[i]-want); if(er>me)me=er; if(er>200+0.05*want)bad++; }
      printf("  [S0] exp_i16 standalone on int16 scores: rc=%d max|err|=%.0f LSB %s (%d/%d)\n",rc,me,bad?"MISMATCH":"OK",bad,M*N);
      free(cs);free(ce); }

    /* S1: MM_I8_OUT16 alone */
    { ork_seq_op o={ .kind=ORK_OP_MATMUL_I16OUT_I8, .w=w, .M=M, .A=A, .C=scores, .mult=mult, .shift=shift };
      int rc=ork_submit_seq(c,&o,1); int bad=0,me=0;
      if(rc){ printf("  [S1] MM_I8_OUT16 rc=%d %s\n",rc,rc==-1?"(wedge/small-K?)":"(dims)"); fail=1; }
      else { for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long acc=0; for(int k=0;k<K;k++)acc+=(long)A[(size_t)m*K+k]*B[(size_t)k*N+n];
              long q=(acc*mult)>>shift; if(q>32767)q=32767; if(q<-32768)q=-32768; int er=abs((int)scores[(size_t)m*N+n]-(int)q); if(er>me)me=er; if(er>1)bad++; }
            printf("  [S1] MM_I8_OUT16 alone: max|err|=%d LSB %s (%d/%d)  scores[0]=%d\n",me,bad?"MISMATCH":"OK",bad,M*N,(int)scores[0]); if(bad)fail=1; } }

    /* S2: [MM_I8_OUT16 -> EXP_I16] resident (scores_i16 aliased into exp) */
    short *scores2=malloc((size_t)M*N*2); for(size_t i=0;i<(size_t)M*N;i++){ scores2[i]=-1; e[i]=-1; }
    { ork_seq_op ops[2]={ { .kind=ORK_OP_MATMUL_I16OUT_I8, .w=w, .M=M, .A=A, .C=scores2, .mult=mult, .shift=shift },
                          { .kind=ORK_OP_EXP_I16, .M=M, .N=N, .A=scores2, .C=e, .in_scale=in_scale, .out_scale=out_scale } };  /* A==op0.C resident */
      int rc=ork_submit_seq(c,ops,2);
      if(rc){ printf("  [S2] chain rc=%d\n",rc); fail=1; }
      else { int bad=0; double me=0;
        for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long acc=0; for(int k=0;k<K;k++)acc+=(long)A[(size_t)m*K+k]*B[(size_t)k*N+n];
              long q=(acc*mult)>>shift; if(q>32767)q=32767; if(q<-32768)q=-32768;
              double want=exp((double)q*in_scale)/out_scale; if(want>32767)want=32767;
              double er=fabs((double)e[(size_t)m*N+n]-want); if(er>me)me=er; if(er>200+0.05*want)bad++; }
        printf("  [S2] MM_I8_OUT16->exp_i16 (scores resident): max|err|=%.0f LSB %s (%d/%d)\n",me,bad?"MISMATCH":"COHERENT",bad,M*N); if(bad)fail=1; } }

    printf("%s\n", fail? "FAIL" : "PASS — int8-mm->int16-out feeds int16 exp resident (A2 sidesteps the set_i16_out chain)");
    ork_mm_free(c,w); ork_npu_free(c);
    return fail;
}
