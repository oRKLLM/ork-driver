/* tools/f16_gmax_sweep.c — validate the bounded-Gmax fp16 fused-SiLU builder across the in-model gate range.
 * For each Gmax, build the LUT via ork_f16_mm_build_silu_lut (which caps the effective Gmax, ORK_F16_GCAP),
 * pack a fresh -S*W probe whose gates span [-Gmax,Gmax], run the fp16 gate, and compare to CPU silu. Reports
 * BULK error (|gate|<=8, the mass of the distribution) separately from FULL (incl. the rare large tail that
 * clamps by design). This is the decisive test for picking the cap: bulk error must stay small at large Gmax.
 *   make f16_gmax_sweep && sudo ./f16_gmax_sweep        (board only; ORK_F16_GCAP to try a ceiling)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double silu(double x){ return x/(1.0+exp(-x)); }
#define K 512
#define N 64

int main(void){
    ork_npu *c=ork_npu_init(); if(!c){printf("no board\n");return 0;}
    ork_f16 *A=malloc((size_t)8*K*2), *B=malloc((size_t)K*N*2); float *C=malloc((size_t)8*N*4);
    for(int i=0;i<8*K;i++)A[i]=(ork_f16)1.0f;
    double gmaxes[]={8,12,20,40,60,91,130};
    printf("Gmax    S       bulk|err|(|g|<=8)   full|err|   bulk-rel%%\n");
    for(unsigned gi=0; gi<sizeof gmaxes/sizeof*gmaxes; gi++){
        double Gmax=gmaxes[gi];
        int16_t lut[1030]; double S=0,R=0,out=0;
        if(ork_f16_mm_build_silu_lut(c,Gmax,lut,&S,&R,&out)){ printf("%-6.1f build FAIL\n",Gmax); continue; }
        /* fresh probe: gates span [-Gmax,Gmax]; weight is -S*gate/K so acc=-S*gate (matches the builder's bake) */
        double tru[N]; for(int n=0;n<N;n++){ tru[n]=Gmax*(n-32)/32.0; double b=(-S*tru[n])/(double)K;
            for(int k=0;k<K;k++)B[(size_t)k*N+n]=(ork_f16)b; }
        ork_w *w=ork_f16_mm_pack(c,K,N,B); if(!w){ printf("%-6.1f pack FAIL\n",Gmax); continue; }
        if(ork_f16_mm_run_silu(c,w,8,A,C,0,0xffffc000u,0x56391100u,lut,1030)){ printf("%-6.1f run FAIL\n",Gmax); ork_mm_free(c,w); continue; }
        double bse=0,fse=0,brel=0; int bn=0;
        for(int n=0;n<N;n++){ double ref=silu(tru[n]), got=C[n]*out, e=fabs(got-ref); fse+=e;
            if(fabs(tru[n])<=8.0){ bse+=e; bn++; if(fabs(ref)>0.1) brel+=e/fabs(ref); } }
        printf("%-6.1f  %-7.2f %-18.4f  %-10.4f  %.1f\n", Gmax, S, bn?bse/bn:0, fse/N, bn?100.0*brel/bn:0);
        ork_mm_free(c,w);
    }
    free(A);free(B);free(C); ork_npu_free(c); return 0;
}
