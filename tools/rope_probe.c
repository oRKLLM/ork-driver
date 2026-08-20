/* Validate ork_f16_npu_rope_neox vs a CPU NEOX-RoPE reference at attention shapes. */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(void){
    ork_npu*c=ork_npu_init(); if(!c) return 2;
    double fb=1000000.0;
    struct { int hd,nh,nt; } cs[] = { {128,16,64}, {128,8,256}, {128,16,256} };
    for(unsigned t=0;t<sizeof(cs)/sizeof(cs[0]);t++){
        int hd=cs[t].hd, nh=cs[t].nh, nt=cs[t].nt, nrow=nh*nt, hd2=hd/2;
        ork_f16 *x=malloc((size_t)nrow*hd*2), *o=malloc((size_t)nrow*hd*2);
        int *pos=malloc(nrow*sizeof(int));
        unsigned s=99+t;
        for(int r=0;r<nrow;r++){ pos[r]=(r/nh); for(int i=0;i<hd;i++){ s=s*1103515245+12345; x[(size_t)r*hd+i]=(ork_f16)(((int)((s>>16)&0xff)-128)/64.0f); } }
        int rc=ork_f16_npu_rope_neox(c,x,hd,nrow,pos,fb,o);
        double maxerr=0;
        for(int r=0;r<nrow;r++){ double p=pos[r];
            for(int i=0;i<hd2;i++){ double th=p*pow(fb,-2.0*i/hd); double cc=cos(th),ss=sin(th);
                double x0=(float)x[(size_t)r*hd+i], x1=(float)x[(size_t)r*hd+i+hd2];
                double r0=x0*cc-x1*ss, r1=x0*ss+x1*cc;
                double e0=fabs((float)o[(size_t)r*hd+i]-r0), e1=fabs((float)o[(size_t)r*hd+i+hd2]-r1);
                if(e0>maxerr)maxerr=e0; if(e1>maxerr)maxerr=e1; } }
        printf("hd=%d nh=%d nt=%d nrow=%d : rc=%d max|err|=%.5f  %s\n", hd,nh,nt,nrow,rc,maxerr,(rc==0&&maxerr<0.02)?"OK":"CHECK");
        free(x);free(o);free(pos);
    }
    ork_npu_free(c); return 0;
}
