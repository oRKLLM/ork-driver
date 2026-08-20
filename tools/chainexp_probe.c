/* chainexp_probe — task #20 HW CHAIN: the softmax numerator [QK^T -> exp -> reduce] in ONE coalesced submit
 * via run_chain_i8_ffn_exp (data-dependent chain; SDP/matmul tasks read prior outputs ON-CHIP by index).
 *   t0 (kind1 matmul): scores_i8 = requant(Q_pad . K^T_pad)   (K padded 128->512)
 *   t1 (kind2 exp-SDP): e_i8 = exp(scores*in_scale)/out_scale  (reads t0 on-chip, in0=0)
 *   t2 (kind0 matmul): Sigma = e_i8 . ones[Nk,32]              (reads t1 on-chip, reduce, int32 out)
 * ONE submit, scores+e never leave the NPU. This is the SW-chain -> HW-chain step for softmax. Validates
 * Sigma vs the CPU int8 exp-sum. Direct (chain is direct-only).  sudo env ORK_MM_TIMEOUT=3000 ./chainexp_probe [Nq]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static uint32_t g=0x9e10u; static int q2(void){ g=g*1664525u+1013904223u; return (int)((g>>27)%3); }
int main(int argc,char**argv){
    int Nq=argc>1?atoi(argv[1]):32, d=128, Nk=512, Kp=512;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("chainexp_probe: Nq=%d d=%d Nk=%d (HW chain: QK^T->exp->reduce, ONE submit)\n",Nq,d,Nk);
    int8_t *Q=malloc((size_t)Nq*d), *K=malloc((size_t)Nk*d);
    for(size_t i=0;i<(size_t)Nq*d;i++) Q[i]=(int8_t)q2();        /* >=0 */
    for(size_t i=0;i<(size_t)Nk*d;i++) K[i]=(int8_t)(-q2());     /* <=0 => scores<=0 */
    int8_t *Qp=calloc((size_t)Nq*Kp,1), *KTp=calloc((size_t)Kp*Nk,1);
    for(int i=0;i<Nq;i++)for(int k=0;k<d;k++) Qp[(size_t)i*Kp+k]=Q[(size_t)i*d+k];
    for(int k=0;k<d;k++)for(int j=0;j<Nk;j++) KTp[(size_t)k*Nk+j]=K[(size_t)j*d+k];
    int r_mult=0x4000, r_shift=16; double in_scale=0.0625, out_scale=1.0/127.0;   /* scores_i8=acc>>~2 in [-128,0]; exp arg [-8,0] */
    /* CPU reference: scores_i8 -> e_i8 -> Sigma */
    int8_t *csc=malloc((size_t)Nq*Nk); double *cS=malloc((size_t)Nq*sizeof(double));
    for(int i=0;i<Nq;i++){ double S=0; for(int j=0;j<Nk;j++){ long a=0; for(int k=0;k<d;k++)a+=Q[(size_t)i*d+k]*K[(size_t)j*d+k];
        long s=(a*r_mult)>>r_shift; if(s>127)s=127; if(s<-128)s=-128; csc[(size_t)i*Nk+j]=(int8_t)s;
        double e=exp((double)s*in_scale)/out_scale; if(e>127)e=127; S+=lround(e); } cS[i]=S; }

    ork_w *w_kt=ork_i8_mm_pack(c,Kp,Nk,KTp); ork_w *w_ones;
    { int8_t *ones=malloc((size_t)Nk*32); memset(ones,1,(size_t)Nk*32); w_ones=ork_i8_mm_pack(c,Nk,32,ones); free(ones); }
    if(!w_kt||!w_ones){ printf("pack fail\n"); return 2; }
    /* generously-sized buffers (int32 span) to be safe about the task.C dtype */
    int32_t *scb=calloc((size_t)Nq*Nk,4), *eb=calloc((size_t)Nq*Nk,4), *ss=calloc((size_t)Nq*32,4);
    for(size_t i=0;i<(size_t)Nq*32;i++)ss[i]=0x7f7f7f7f;
    ork_mm_task_i8 tasks[3] = { { w_kt,   Nq, Qp,             scb },        /* t0: QK^T -> scores */
                               { w_kt,   Nq, (int8_t*)scb,   eb  },        /* t1: exp(scores) -> e  (reads t0; reuses w_kt for N-sizing) */
                               { w_ones, Nq, (int8_t*)eb,    ss  } };      /* t2: e . ones -> Sigma (reads t1) */
    ork_chain_op ops[3] = { { 1, -1, 0, r_mult, r_shift },   /* t0 matmul int8 requant */
                            { 2,  0, 0, 0, 0 },               /* t1 exp-SDP, in0=t0 */
                            { 0,  1, 0, 0, 0 } };             /* t2 matmul int32, in0=t1 (reduce) */
    int rc=ork_i8_mm_run_chain_ffn_exp(c,3,tasks,ops,in_scale,out_scale);
    printf("  run_chain_i8_ffn_exp([QK^T->exp->reduce]) rc=%d  scores[0]=%d e[0]=%d ss[0]=%d\n",
           rc,(int)((int8_t*)scb)[0],(int)((int8_t*)eb)[0],ss[0]);
    if(rc){ printf("FAIL rc=%d\n",rc); return 1; }
    int bad=0; double me=0;
    for(int i=0;i<Nq;i++){ double Sn=(double)ss[(size_t)i*32]; double rel=fabs(Sn-cS[i])/(cS[i]>0?cS[i]:1); if(rel>me)me=rel; if(rel>0.05)bad++; }
    printf("  Sigma vs CPU exp-sum: max rel-err=%.3e %s (%d/%d rows)  (CPU row0 Sigma=%.0f)\n", me, bad?"CHECK":"COHERENT", bad, Nq, cS[0]);
    printf("%s\n", bad?"FAIL":"PASS — HW chain: QK^T->exp->reduce in ONE submit (scores+e on-chip), coherent");
    ork_mm_free(c,w_kt); ork_mm_free(c,w_ones); ork_npu_free(c);
    return bad;
}
