/* chainav_probe — task #20 HW CHAIN, full attention core: [QK^T -> exp -> reduce, e.V] in ONE coalesced submit.
 *   t0 (kind1 matmul): scores_i8 = requant(Q_pad . K^T_pad)        (K padded 128->512)
 *   t1 (kind2 exp-SDP): e_i8 = exp(scores)                          (reads t0 on-chip)
 *   t2 (kind0 matmul): Sigma  = e_i8 . ones[Nk,32]                  (reads t1; softmax denom)
 *   t3 (kind0 matmul): av     = e_i8 . V[Nk,dv]                     (reads t1; UNNORMALIZED attention)
 * attn = av / Sigma (deferred normalize; out_scale cancels). e never leaves the NPU and feeds BOTH the reduce
 * and A.V on-chip. Validates attn vs CPU (same int8 e). Direct.  sudo env ORK_MM_TIMEOUT=3000 ./chainav_probe [Nq]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static uint32_t g=0x9e10u; static int q2(void){ g=g*1664525u+1013904223u; return (int)((g>>27)%3); }
static int vv(void){ g=g*1664525u+1013904223u; return (int)((g>>27)%5)-2; }   /* [-2,2] */
int main(int argc,char**argv){
    int Nq=argc>1?atoi(argv[1]):32, d=128, Nk=512, Kp=512, dv=128;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("chainav_probe: Nq=%d d=%d Nk=%d dv=%d (HW chain: QK^T->exp->reduce,e.V  ONE submit)\n",Nq,d,Nk,dv);
    int8_t *Q=malloc((size_t)Nq*d), *K=malloc((size_t)Nk*d), *V=malloc((size_t)Nk*dv);
    for(size_t i=0;i<(size_t)Nq*d;i++) Q[i]=(int8_t)q2();
    for(size_t i=0;i<(size_t)Nk*d;i++) K[i]=(int8_t)(-q2());
    for(size_t i=0;i<(size_t)Nk*dv;i++) V[i]=(int8_t)vv();
    int8_t *Qp=calloc((size_t)Nq*Kp,1), *KTp=calloc((size_t)Kp*Nk,1);
    for(int i=0;i<Nq;i++)for(int k=0;k<d;k++) Qp[(size_t)i*Kp+k]=Q[(size_t)i*d+k];
    for(int k=0;k<d;k++)for(int j=0;j<Nk;j++) KTp[(size_t)k*Nk+j]=K[(size_t)j*d+k];
    int r_mult=0x4000, r_shift=16; double in_scale=0.0625, out_scale=1.0/127.0;
    /* CPU ref: scores_i8 -> e_i8 -> Sigma, av, attn */
    int8_t *ce=malloc((size_t)Nq*Nk); double *cS=malloc((size_t)Nq*8), *cav=malloc((size_t)Nq*dv*sizeof(double));
    for(int i=0;i<Nq;i++){ double S=0; for(int j=0;j<Nk;j++){ long a=0; for(int k=0;k<d;k++)a+=Q[(size_t)i*d+k]*K[(size_t)j*d+k];
        long s=(a*r_mult)>>r_shift; if(s>127)s=127; if(s<-128)s=-128; double e=exp((double)s*in_scale)/out_scale; if(e>127)e=127;
        int ei=(int)lround(e); ce[(size_t)i*Nk+j]=(int8_t)ei; S+=ei; }
      cS[i]=S; for(int x=0;x<dv;x++){ double av=0; for(int j=0;j<Nk;j++) av+=(double)ce[(size_t)i*Nk+j]*V[(size_t)j*dv+x]; cav[(size_t)i*dv+x]=av; } }

    ork_w *w_kt=ork_i8_mm_pack(c,Kp,Nk,KTp), *w_ones, *w_v;
    { int8_t *ones=malloc((size_t)Nk*32); memset(ones,1,(size_t)Nk*32); w_ones=ork_i8_mm_pack(c,Nk,32,ones); free(ones); }
    w_v=ork_i8_mm_pack(c,Nk,dv,V);
    if(!w_kt||!w_ones||!w_v){ printf("pack fail\n"); return 2; }
    int32_t *scb=calloc((size_t)Nq*Nk,4), *eb=calloc((size_t)Nq*Nk,4), *ss=calloc((size_t)Nq*32,4), *avb=calloc((size_t)Nq*dv,4);
    ork_mm_task_i8 tasks[4] = { { w_kt,   Nq, Qp,           scb },
                               { w_kt,   Nq, (int8_t*)scb, eb  },
                               { w_ones, Nq, (int8_t*)eb,  ss  },
                               { w_v,    Nq, (int8_t*)eb,  avb } };
    ork_chain_op ops[4] = { { 1, -1, 0, r_mult, r_shift },   /* QK^T */
                            { 2,  0, 0, 0, 0 },               /* exp(t0) */
                            { 0,  1, 0, 0, 0 },               /* reduce e (t1) -> Sigma */
                            { 0,  1, 0, 0, 0 } };             /* e.V (t1) -> av */
    int rc=ork_i8_mm_run_chain_ffn_exp(c,4,tasks,ops,in_scale,out_scale);
    printf("  run_chain_i8_ffn_exp([QK^T->exp->reduce,e.V]) rc=%d  ss[0]=%d av[0]=%d\n",rc,ss[0],avb[0]);
    if(rc){ printf("FAIL rc=%d\n",rc); return 1; }
    /* attn = av/Sigma vs CPU */
    int bad=0; double me=0, sae=0;
    for(int i=0;i<Nq;i++){ double Sn=(double)ss[(size_t)i*32]; if(Sn<=0)Sn=1; double Sc=cS[i]>0?cS[i]:1;
        for(int x=0;x<dv;x++){ double an=(double)avb[(size_t)i*dv+x]/Sn; double ac=cav[(size_t)i*dv+x]/Sc;
            double e=fabs(an-ac); sae+=e; if(e>me)me=e; if(e>0.05&&fabs(ac)>1e-3&&e/fabs(ac)>0.05)bad++; } }
    printf("  attn=av/Sigma vs CPU: max|err|=%.4f mae=%.4f %s (%d/%d)\n", me, sae/(Nq*dv), bad?"CHECK":"COHERENT", bad, Nq*dv);
    printf("%s\n", bad?"FAIL":"PASS — full attention core [QK^T->exp->reduce,e.V] in ONE HW-chain submit, coherent");
    ork_mm_free(c,w_kt); ork_mm_free(c,w_ones); ork_mm_free(c,w_v); ork_npu_free(c);
    return bad;
}
