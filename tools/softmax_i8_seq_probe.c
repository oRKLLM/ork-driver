/* softmax_i8_seq_probe — task #20: the RESIDENT int8 softmax island as ONE ork_submit_seq, composing the three
 * proven pieces with intermediates A2-resident (aliased, no re-upload between ops):
 *   op0  scores_i8 = clamp_i8((Q_pad·K^T_pad)*mult>>shift)   MATMUL_REQUANT_I8   (K padded 128->512, kpad_qkt_probe)
 *   op1  e_i8      = clamp_i8(exp((scores-gmax)*in_scale)/out_scale)  EXP_I8 (b_scale=gmax, exp_biased_probe)
 *   op2  Sigma     = e_i8 · ones_i8[N,32]                     MM_I8 reduce        (int8reduce_probe)
 *   [normalize] P = e_i8/Sigma (deferred; the gmax constant + out_scale cancel)
 * scores_i8 aliased into op1.A, e_i8 aliased into op2.A => the softmax MID as a resident chain (the piece the
 * layer's resident FRONT/BACK already had). gmax = a STATIC calibrated bound (here: the CPU-reference max).
 * Validates the seq's scores vs CPU and the softmax P=e/Sigma vs CPU softmax.
 *   direct: sudo env ORK_MM_TIMEOUT=3000 timeout 90 ./softmax_i8_seq_probe [N] [d]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static uint32_t g=0x50f7a5u; static int r8lo(void){ g=g*1664525u+1013904223u; return (int)((g>>26)%9)-4; }  /* [-4,4] */
int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):64, d=argc>2?atoi(argv[2]):128;   /* N tokens (queries==keys), d head_dim; N%32 */
    int Kp=512;
    setvbuf(stdout,0,_IONBF,0);
    if(d>Kp||N%32){ printf("need d<=512, N%%32\n"); return 2; }
    int viaorkd=getenv("ORK_USE_ORKD")?1:0;
    /* DIRECT mode hits the int8-LUT-after-multicore-matmul wedge (EXP_I8 right after MATMUL_REQUANT_I8) and
     * HANGS the NPU (needs sudo reboot). Only orkd's hardened path runs it wedge-free. Refuse direct unless
     * explicitly overridden, so this can't casually wedge the board. */
    if(!viaorkd && !getenv("ORK_ALLOW_DIRECT_WEDGE")){
        printf("REFUSING direct mode: this seq wedges the NPU (int8-LUT-after-matmul). Run with ORK_USE_ORKD=1 ORKD_BIN=./orkd, or set ORK_ALLOW_DIRECT_WEDGE=1 to force.\n");
        return 2;
    }
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("softmax_i8_seq_probe: N=%d d=%d path=%s (resident int8 softmax: reqQK^T -> biased exp -> reduce)\n",N,d,viaorkd?"ORKD (hardened)":"direct");
    int mult=0x4000, shift=18; double in_scale=0.05, out_scale=1.0/127.0;
    int8_t *Q=malloc((size_t)N*d), *Kk=malloc((size_t)N*d);
    for(size_t i=0;i<(size_t)N*d;i++){ Q[i]=(int8_t)r8lo(); Kk[i]=(int8_t)r8lo(); }
    int8_t *Qp=calloc((size_t)N*Kp,1), *KTp=calloc((size_t)Kp*N,1);
    for(int i=0;i<N;i++)for(int k=0;k<d;k++) Qp[(size_t)i*Kp+k]=Q[(size_t)i*d+k];
    for(int k=0;k<d;k++)for(int j=0;j<N;j++) KTp[(size_t)k*N+j]=Kk[(size_t)j*d+k];
    /* CPU reference: int8 scores, global max, softmax over dequantized scores */
    int8_t *csc=malloc((size_t)N*N); int gmax=-128;
    for(int m=0;m<N;m++)for(int j=0;j<N;j++){ long acc=0; for(int k=0;k<d;k++)acc+=(long)Q[(size_t)m*d+k]*Kk[(size_t)j*d+k];
        long q=(acc*mult)>>shift; if(q>127)q=127; if(q<-128)q=-128; csc[(size_t)m*N+j]=(int8_t)q; if(q>gmax)gmax=(int)q; }
    float *ref=malloc((size_t)N*N*4);
    for(int m=0;m<N;m++){ double S=0; for(int j=0;j<N;j++) S+=exp((double)csc[(size_t)m*N+j]*in_scale);
        for(int j=0;j<N;j++) ref[(size_t)m*N+j]=(float)(exp((double)csc[(size_t)m*N+j]*in_scale)/S); }
    printf("  int8 score global max (static bias) = %d\n", gmax);

    /* pre-calibrate the int-LUT idx in a clean context (first-op-after-matmul trap guard) */
    { int8_t wi[32],wo[32]; for(int i=0;i<32;i++) wi[i]=(int8_t)(i-16); ork_i8_npu_exp(c,wi,1,32,in_scale,out_scale,wo,NULL); }

    ork_w *w_kt=ork_i8_mm_pack(c,Kp,N,KTp); if(!w_kt){printf("pack KT fail\n");return 2;}
    int8_t *ones=malloc((size_t)N*32); memset(ones,1,(size_t)N*32);
    ork_w *w_ones=ork_i8_mm_pack(c,N,32,ones); if(!w_ones){printf("pack ones fail\n");return 2;}
    int8_t *scores=malloc((size_t)N*N), *e=malloc((size_t)N*N); int32_t *ss=malloc((size_t)N*32*4);
    for(size_t i=0;i<(size_t)N*N;i++){ scores[i]=-128; e[i]=-128; }

    /* ONE seq: reqQK^T -> biased exp -> reduce; scores/e aliased resident */
    ork_seq_op ops[3] = {
        { .kind=ORK_OP_MATMUL_REQUANT_I8, .w=w_kt,   .M=N, .A=Qp,     .C=scores, .mult=mult, .shift=shift },
        { .kind=ORK_OP_EXP_I8,            .M=N, .N=N, .A=scores, .C=e, .in_scale=in_scale, .out_scale=out_scale, .b_scale=(double)gmax },  /* A==op0.C */
        { .kind=ORK_OP_MM_I8,             .w=w_ones, .M=N, .A=e,      .C=ss },  /* A==op1.C */
    };
    int rc=ork_submit_seq(c,ops,3);
    printf("  ork_submit_seq([reqQK^T, biased exp, reduce], scores/e aliased) rc=%d\n", rc);
    if(rc){ printf("FAIL rc=%d\n", rc); ork_mm_free(c,w_kt); ork_mm_free(c,w_ones); ork_npu_free(c); return 1; }

    int fail=0;
    /* CPU reference for the reduced Sigma of the biased exp: Sigma_m = sum_j clamp_i8(exp((scores-gmax)*in_scale)/out_scale) */
    double *cpuS=malloc((size_t)N*sizeof(double));
    for(int m=0;m<N;m++){ double S=0; for(int j=0;j<N;j++){ double v=exp(((double)csc[(size_t)m*N+j]-gmax)*in_scale)/out_scale; long q=lround(v); if(q>127)q=127; if(q<-128)q=-128; S+=q; } cpuS[m]=S; }
    if(viaorkd){
        /* ORKD: scores & e are A2-resident (c_keep, NOT shipped back) — only the reduce output Sigma returns.
         * Validate Sigma vs the CPU biased-exp sum (the whole resident chain ran: scores -> biased exp -> reduce). */
        double me=0; int bad=0;
        for(int m=0;m<N;m++){ double Sn=(double)ss[(size_t)m*32]; double rel=fabs(Sn-cpuS[m])/(cpuS[m]>0?cpuS[m]:1); if(rel>me)me=rel; if(rel>0.03)bad++; }
        printf("  [ORKD] Sigma (resident chain output) vs CPU biased-exp sum: max rel-err=%.3e %s (%d/%d rows)\n",me,bad?"CHECK":"COHERENT",bad,N);
        if(bad)fail=1;
    } else {
        int emin=127,emax=-128; for(size_t i=0;i<(size_t)N*N;i++){ if(e[i]<emin)emin=e[i]; if(e[i]>emax)emax=e[i]; }
        printf("  [dbg] e_i8 range [%d,%d]  ss[0]=%d (CPU ~ %.0f)\n",emin,emax,ss[0],cpuS[0]);
        /* scores vs CPU */
        { int bad=0,mev=0; for(int m=0;m<N;m++)for(int j=0;j<N;j++){ int er=abs((int)scores[(size_t)m*N+j]-(int)csc[(size_t)m*N+j]); if(er>mev)mev=er; if(er>1)bad++; }
          printf("  scores_i8 vs CPU: max|err|=%d LSB %s (%d/%d)\n",mev,bad?"MISMATCH":"OK",bad,N*N); if(bad)fail=1; }
        /* softmax P=e/Sigma vs CPU (gmax + out_scale cancel) */
        { double me=0; int bad=0;
          for(int m=0;m<N;m++){ double S=(double)ss[(size_t)m*32]; if(S<=0)S=1;
            for(int j=0;j<N;j++){ double P=(double)e[(size_t)m*N+j]/S; double er=fabs(P-ref[(size_t)m*N+j]); if(er>me)me=er; if(er>2e-2)bad++; } }
          printf("  softmax P=e/Sigma vs CPU: max|err|=%.3e %s (%d/%d)\n",me,bad?"CHECK":"COHERENT",bad,N*N); if(bad)fail=1; }
    }
    printf("%s\n", fail?"FAIL":"PASS — resident int8 softmax island as one seq (K-pad QK^T -> biased exp -> reduce, A2-resident)");
    ork_mm_free(c,w_kt); ork_mm_free(c,w_ones); ork_npu_free(c);
    return fail;
}
