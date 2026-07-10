/* tools/f16_gate_acc.c — ACCURACY of the fp16 gate+SiLU, two ways, vs a CPU fp32 reference.
 *
 * The all-fp16 FFN chain gives garbage PPL (9072 vs int8 9.15). Localized to the fp16 GATE + fused-silu-LUT
 * path. This isolates WHY, off-model (no wedge risk — just matmuls):
 *   REF   : CPU fp32  silu(gate),  gate = A.Wg
 *   FUSED : ork_mm_run_f16_silu (gate packed -S*Wg, per-gmax LUT) -> C*out_scale     [what the model does]
 *   PLAIN : ork_mm_run (gate packed raw Wg) -> fp16 gate -> CPU silu                 [the proposed fix]
 * A=1 (every row identical) so gate_n = sum_k Wg[k,n]; sweep gate_n across [-gmax,gmax] over N columns.
 * Reports max/mean abs error of FUSED and PLAIN vs REF, at several gmax (incl blk.2's ~132) and M.
 *   make f16_gate_acc && sudo ./f16_gate_acc [K] [M]     (board only)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef ork_f16 f16;
static double siluf(double x){ return x/(1.0+exp(-x)); }

static void run_gmax(ork_npu*c,int K,int M,double gmax){
    const int N=64;
    double gate[64]; for(int n=0;n<N;n++) gate[n]=-gmax + 2.0*gmax*n/(double)(N-1);   /* sweep [-gmax,gmax] */
    f16*A=malloc((size_t)M*K*2); for(size_t i=0;i<(size_t)M*K;i++)A[i]=(f16)1.0f;      /* A=1 */
    f16*Bp=malloc((size_t)K*N*2), *Bf=malloc((size_t)K*N*2);
    float*Cp=malloc((size_t)M*N*4), *Cf=malloc((size_t)M*N*4);

    /* ---- build the model's LUT for this gmax (also yields S, out_scale) ---- */
    int16_t lut[1030]; double S=0,R=0,os=0;
    if(ork_mm_build_f16_silu_lut(c,gmax,lut,&S,&R,&os)){ printf("  gmax=%.0f: LUT build FAIL\n",gmax); return; }

    /* PLAIN: raw Wg = gate_n/K ; FUSED: -S*Wg */
    for(int k=0;k<K;k++)for(int n=0;n<N;n++){ double w=gate[n]/(double)K; Bp[(size_t)k*N+n]=(f16)w; Bf[(size_t)k*N+n]=(f16)(-S*w); }

    ork_w*wp=ork_mm_pack(c,K,N,Bp), *wf=ork_mm_pack(c,K,N,Bf);
    if(!wp||!wf){ printf("  pack fail\n"); return; }
    int rp=ork_mm_run(c,wp,M,A,Cp);                                                    /* plain gate matmul */
    int rf=ork_mm_run_f16_silu(c,wf,M,A,Cf,0,0xffffc000u,0x56391100u,lut,1030);         /* fused silu */
    if(rp||rf){ printf("  gmax=%.0f: run rc plain=%d fused=%d\n",gmax,rp,rf); ork_mm_free(c,wp);ork_mm_free(c,wf); return; }

    double ef_max=0,ef_sum=0, ep_max=0,ep_sum=0, gerr=0;
    for(int n=0;n<N;n++){
        double ref=siluf(gate[n]);
        double got_f=(double)Cf[n]*os;                         /* FUSED silu output */
        double got_p=siluf((double)Cp[n]);                     /* PLAIN: cpu silu on the fp16 gate */
        double df=fabs(got_f-ref), dp=fabs(got_p-ref);
        ef_max=df>ef_max?df:ef_max; ef_sum+=df; ep_max=dp>ep_max?dp:ep_max; ep_sum+=dp;
        double ge=fabs((double)Cp[n]-gate[n]); gerr=ge>gerr?ge:gerr;   /* plain matmul gate accuracy */
    }
    printf("  gmax=%6.1f | FUSED err max=%.4g mean=%.4g | PLAIN err max=%.4g mean=%.4g | plain-gate maxerr=%.4g\n",
           gmax, ef_max, ef_sum/N, ep_max, ep_sum/N, gerr);
    ork_mm_free(c,wp); ork_mm_free(c,wf); free(A);free(Bp);free(Bf);free(Cp);free(Cf);
}

int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):2048, M=argc>2?atoi(argv[2]):8;
    ork_npu*c=ork_npu_init(); if(!c){printf("no board\n");return 0;}
    printf("f16_gate_acc K=%d M=%d  (silu(gate) error vs CPU fp32 ref)\n",K,M);
    double gmaxes[]={8,16,30,64,132}; for(int i=0;i<5;i++) run_gmax(c,K,M,gmaxes[i]);
    printf("VERDICT: PLAIN (fp16 gate + CPU silu) should be ~fp16-precision; FUSED (per-tensor LUT) error grows with gmax.\n");
    ork_npu_free(c); return 0;
}
