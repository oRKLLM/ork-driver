/* fused_exp_probe — task #20: the HW-CHAINABLE softmax exp — exp fused onto the score matmul's DPU output
 * stage (ork_f16_mm_run_act, fn=exp) in ONE submit (no separate exp op, no matmul->SDP crossing). This is
 * the M4.8 "no-crossing chain" applied to softmax: C = exp(Q·K^T) fused. Scores constructed <=0 (the
 * post-max-subtract softmax domain; fp16 SDP index spreads for a single sign). Validates vs CPU exp.
 *   sudo env ORK_MM_TIMEOUT=3000 timeout 40 ./fused_exp_probe [M] [d] [N]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static double myexp(double x, void *ctx){ (void)ctx; return exp(x); }
static uint32_t g=0x3ef1u; static float fr(void){ g=g*1664525u+1013904223u; return (float)(g>>8)/(float)(1u<<24); }
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):32, d=argc>2?atoi(argv[2]):128, N=argc>3?atoi(argv[3]):64;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("fused_exp_probe: M=%d d=%d N=%d (exp fused on the score matmul, one submit)\n",M,d,N);
    /* Q>=0, K^T<=0 => scores = Q.K^T <= 0 (softmax post-max domain, single-signed for the fp16 index) */
    ork_f16 *Q=malloc((size_t)M*d*sizeof(ork_f16)), *KT=malloc((size_t)d*N*sizeof(ork_f16));
    for(size_t i=0;i<(size_t)M*d;i++) Q[i]=(ork_f16)(fr()*0.5f);          /* [0,0.5] */
    for(size_t i=0;i<(size_t)d*N;i++) KT[i]=(ork_f16)(-fr()*0.5f);        /* [-0.5,0] */
    float *C=malloc((size_t)M*N*4);
    /* CPU scores + band */
    float *sc=malloc((size_t)M*N*4); float lo=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double s=0; for(int k=0;k<d;k++) s+=(double)(float)Q[(size_t)m*d+k]*(float)KT[(size_t)k*N+n];
        sc[(size_t)m*N+n]=(float)s; if(s<lo)lo=(float)s; }
    printf("  score range [%.3f, 0]\n", lo);
    double us_est=0; int rc=ork_f16_mm_run_act(c,d,N,KT,M,Q,C,myexp,NULL,(double)lo-0.01,0.0);
    printf("  ork_f16_mm_run_act(fn=exp) rc=%d C[0]=%.4f (want exp(%.3f)=%.4f)\n",rc,C[0],sc[0],exp(sc[0]));
    if(rc){ printf("FAIL rc=%d\n",rc); ork_npu_free(c); return 1; }
    int bad=0; double me=0,mre=0;
    for(int i=0;i<M*N;i++){ double want=exp((double)sc[i]); double e=fabs(C[i]-want); double re=e/(want+1e-3); if(e>me)me=e; if(re>mre)mre=re; if(re>0.03&&e>4e-3)bad++; }
    printf("  fused exp(scores): max|err|=%.3e maxrel=%.3e  %s (%d/%d)\n",me,mre,bad?"CHECK":"COHERENT",bad,M*N);
    printf("%s\n", bad?"FAIL":"PASS — HW-chained fused exp on the score matmul (one submit, no separate exp op / no crossing)");
    ork_npu_free(c);
    return bad?1:0;
}
