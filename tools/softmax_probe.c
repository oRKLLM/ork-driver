/* 2b: validate ork_f16_npu_softmax (exp-on-NPU) coherence vs CPU softmax, at attention-ish shapes. */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
static float h2f(ork_f16 h){ return (float)h; }
int main(void){
    ork_npu*c=ork_npu_init(); if(!c) return 2;
    struct { int M,n; } cs[] = { {256,256}, {64,512}, {16,128}, {256,64} };
    for(unsigned i=0;i<sizeof(cs)/sizeof(cs[0]);i++){
        int M=cs[i].M,n=cs[i].n;
        ork_f16 *x=malloc((size_t)M*n*sizeof(ork_f16)), *o=malloc((size_t)M*n*sizeof(ork_f16));
        unsigned seed=1234+i;
        for(int j=0;j<M*n;j++){ seed=seed*1103515245+12345; x[j]=(ork_f16)(((int)((seed>>16)&0xff)-128)/32.0f); }
        int r=ork_f16_npu_softmax(c,M,n,x,o);
        /* CPU ref + max-abs-err + row-sum check */
        double maxerr=0, worstsum=0;
        for(int m=0;m<M;m++){ float mx=h2f(x[m*n]); for(int j=1;j<n;j++){float v=h2f(x[m*n+j]); if(v>mx)mx=v;}
            double sm=0; for(int j=0;j<n;j++) sm+=exp((double)h2f(x[m*n+j])-mx);
            double rowsum=0; for(int j=0;j<n;j++){ double ref=exp((double)h2f(x[m*n+j])-mx)/sm; double got=h2f(o[m*n+j]); rowsum+=got;
                double e=fabs(got-ref); if(e>maxerr)maxerr=e; }
            if(fabs(rowsum-1.0)>fabs(worstsum-1.0)) worstsum=rowsum; }
        printf("M=%d n=%d : rc=%d  max|err|=%.5f  worst row-sum=%.4f  %s\n",
               M,n,r,maxerr,worstsum,(r==0&&maxerr<0.02)?"OK":"CHECK");
        free(x);free(o);
    }
    ork_npu_free(c); return 0;
}
