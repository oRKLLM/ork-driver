/* fused_softmax_probe — task #20 (a): the full softmax built on the HW-chained fused exp (M4.8).
 *   e   = exp(Q·K^T)         ork_mm_run_f16_act(fn=exp)  — exp fused on the score matmul, ONE submit (free exp)
 *   Sig = e . ones[n,16]     MM_F16                       — Σ reduce on-NPU
 *   P   = e / Σ                                            — softmax
 * Scores constructed ≤0 (Q≥0, K^T≤0), the post-max softmax domain (exp≤1, no overflow, single-signed index).
 * Validates P vs CPU softmax(scores). This is the fused-exp softmax island — the expensive exp is HW-chained
 * onto the matmul (no separate exp op, no quantize, no crossing).
 *   sudo env ORK_MM_TIMEOUT=3000 timeout 60 ./fused_softmax_probe [M] [d] [n]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static double myexp(double x, void *ctx){ (void)ctx; return exp(x); }
static uint32_t g=0x5c0ffeeu; static float fr(void){ g=g*1664525u+1013904223u; return (float)(g>>8)/(float)(1u<<24); }
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):32, d=argc>2?atoi(argv[2]):128, n=argc>3?atoi(argv[3]):64;  /* n keys, %16 */
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("fused_softmax_probe: M=%d d=%d n=%d (fused exp on QK^T + Sigma-reduce)\n",M,d,n);
    ork_f16 *Q=malloc((size_t)M*d*sizeof(ork_f16)), *KT=malloc((size_t)d*n*sizeof(ork_f16));
    for(size_t i=0;i<(size_t)M*d;i++) Q[i]=(ork_f16)(fr()*0.5f);       /* >=0 */
    for(size_t i=0;i<(size_t)d*n;i++) KT[i]=(ork_f16)(-fr()*0.5f);     /* <=0 => scores<=0 */
    /* CPU scores + reference softmax + band */
    float *sc=malloc((size_t)M*n*4), *ref=malloc((size_t)M*n*4); float lo=0;
    for(int m=0;m<M;m++){ for(int j=0;j<n;j++){ double s=0; for(int k=0;k<d;k++) s+=(double)(float)Q[(size_t)m*d+k]*(float)KT[(size_t)k*n+j]; sc[(size_t)m*n+j]=(float)s; if(s<lo)lo=(float)s; }
        double sum=0; for(int j=0;j<n;j++) sum+=exp((double)sc[(size_t)m*n+j]); for(int j=0;j<n;j++) ref[(size_t)m*n+j]=(float)(exp((double)sc[(size_t)m*n+j])/sum); }
    /* fused exp(Q·K^T) — one submit */
    float *e=malloc((size_t)M*n*4);
    int rc=ork_mm_run_f16_act(c,d,n,KT,M,Q,e,myexp,NULL,(double)lo-0.01,0.0);
    printf("  fused exp(QK^T) rc=%d e[0]=%.5f (want %.5f)\n",rc,e[0],exp(sc[0]));
    if(rc){ printf("FAIL rc=%d\n",rc); ork_npu_free(c); return 1; }
    /* Sigma = e . ones[n,16] on-NPU (narrow e->f16 first) */
    ork_f16 *ef=malloc((size_t)M*n*sizeof(ork_f16)); for(size_t i=0;i<(size_t)M*n;i++) ef[i]=(ork_f16)e[i];
    ork_f16 *ones=malloc((size_t)n*16*sizeof(ork_f16)); for(size_t i=0;i<(size_t)n*16;i++) ones[i]=(ork_f16)1.0f;
    ork_w *w=ork_mm_pack(c,n,16,ones); float *ss=malloc((size_t)M*16*4);
    int rrc=-1; if(w){ ork_mm_task_f16 t={w,M,ef,ss}; rrc=ork_mm_run_stream_f16(c,1,&t); }
    /* normalize P = e / Sigma */
    float *P=malloc((size_t)M*n*4);
    for(int m=0;m<M;m++){ double S=(rrc==0)?ss[(size_t)m*16]:0; if(rrc!=0){ for(int j=0;j<n;j++) S+=e[(size_t)m*n+j]; } if(S<=0)S=1;
        for(int j=0;j<n;j++) P[(size_t)m*n+j]=(float)(e[(size_t)m*n+j]/S); }
    int bad=0; double me=0,sae=0;
    for(int i=0;i<M*n;i++){ double er=fabs((double)P[i]-ref[i]); sae+=er; if(er>me)me=er; if(er>2e-2)bad++; }
    printf("  softmax P vs CPU: reduce_rc=%d max|err|=%.3e mae=%.3e  %s (%d/%d)\n",rrc,me,sae/(M*n),bad?"CHECK":"COHERENT",bad,M*n);
    printf("%s\n", bad?"FAIL":"PASS — fused-exp softmax coherent: exp HW-chained on QK^T (free, one submit) + on-NPU Sigma-reduce");
    if(w) ork_mm_free(c,w); ork_npu_free(c);
    return bad?1:0;
}
