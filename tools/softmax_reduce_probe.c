/* softmax_reduce_probe — the gating experiment for a fused on-NPU softmax chain: does the
 * exp(NPU activation/SDP) -> Sigma(NPU reduce-matmul) sequence work POST-REFACTOR, or does it still
 * ETIMEDOUT on the activation->matmul mode-switch (the reason ork_npu_softmax_f16 keeps the sum on CPU)?
 * If ork_npu_enter now carries this transition, the full softmax (max+exp+sum+norm) can run on-NPU and
 * a single-submit QK^T->softmax->A.V chain becomes buildable. Reduce = e . ones[n,16] (sum in col 0).
 * Self-validating: sum_j exp(x-max) vs CPU. BOARD: sudo ./softmax_reduce_probe */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    int M=256,n=256;
    /* ones[n,16] fp16 reduce weight */
    ork_f16*ones=malloc((size_t)n*16*2); for(size_t i=0;i<(size_t)n*16;i++)ones[i]=(ork_f16)1.0f;
    ork_w*ow=ork_mm_pack(c,n,16,ones); free(ones); if(!ow){printf("pack ones failed\n");return 2;}
    ork_f16*x=malloc((size_t)M*n*2); float*mx=malloc((size_t)M*4);
    int16_t*xi=malloc((size_t)M*n*2),*ei=malloc((size_t)M*n*2);
    ork_f16*e=malloc((size_t)M*n*2); float*ss=malloc((size_t)M*16*4);
    unsigned s=7; for(int i=0;i<M*n;i++){s=s*1103515245+12345; x[i]=(ork_f16)(((int)((s>>16)&0xff)-128)/48.0f);}
    for(int m=0;m<M;m++){float mv=(float)x[(size_t)m*n];for(int j=1;j<n;j++){float v=(float)x[(size_t)m*n+j];if(v>mv)mv=v;}mx[m]=mv;}
    double lo=0; for(int m=0;m<M;m++)for(int j=0;j<n;j++){float d=(float)x[(size_t)m*n+j]-mx[m];if(d<lo)lo=d;}
    double in_scale=(-lo)/32000.0; if(in_scale<=0)in_scale=1e-6; double out_scale=1.0/32000.0;

    int bad=0;
    for(int it=0; it<4; it++){
        for(int m=0;m<M;m++)for(int j=0;j<n;j++){long q=lround(((double)((float)x[(size_t)m*n+j]-mx[m]))/in_scale); if(q<-32768)q=-32768; if(q>32767)q=32767; xi[(size_t)m*n+j]=(int16_t)q;}
        double t0=now();
        int er=ork_npu_exp_i16(c,xi,M,n,in_scale,out_scale,ei,NULL);          /* exp on NPU (SDP) */
        double t_exp=now()-t0;
        for(int i=0;i<M*n;i++)e[i]=(ork_f16)((double)ei[i]*out_scale);
        t0=now();
        int rr=ork_mm_run(c,ow,M,e,ss);                                       /* Sigma on NPU (reduce-matmul) -- the mode-switch */
        double t_red=now()-t0;
        /* coherence: NPU sum vs CPU sum of exp(x-max) */
        double maxrel=0;
        for(int m=0;m<M;m++){ double cpu=0; for(int j=0;j<n;j++)cpu+=exp((double)((float)x[(size_t)m*n+j]-mx[m]));
            double npu=ss[(size_t)m*16]; double rel=fabs(npu-cpu)/(cpu>0?cpu:1); if(rel>maxrel)maxrel=rel; if(rel>0.05)bad++; }
        printf("it%d: exp rc=%d (%.0fus)  reduce rc=%d (%.0fus)  max rel-err=%.4f  %s\n",
               it,er,t_exp,rr,t_red,maxrel,(er==0&&rr==0&&maxrel<0.05)?"OK":(rr!=0?"REDUCE FAILED (mode-switch?)":"CHECK"));
    }
    ork_npu_free(c);
    printf(bad?"\nRESULT: exp->reduce on-NPU NOT clean (%d bad)\n":"\nRESULT: exp->reduce on-NPU WORKS post-refactor — full on-NPU softmax buildable\n",bad);
    return bad?1:0;
}
