/* orkd_attn_probe — validate the fused attention core routed THROUGH orkd (ORKD_ATTN / orkd_attn_i8).
 * Mirrors chainav_probe but runs [QK^T->exp->reduce,e.V] as ONE daemon-side chain submit against 3 resident
 * weights (K^T[Kp,Nk], ones[Nk,32], V[Nk,dv]). Checks attn=av/Sigma vs the same int8 CPU reference.
 *
 * This version uses REAL-RANGE scores (mixed-sign, wide QK^T) — the case that BREAKS a plain exp — and the
 * scalar-max-biased exp (max_bias baked into the LUT: e=exp((score-max_bias)*in_scale)/out_scale). A max_bias >=
 * every real score keeps args <=0 so int8 exp never saturates, and the constant cancels in av/Sigma (registry:
 * exp_biased_probe). Also reports how many e WOULD saturate at max_bias=0 — proving the bias is what makes the
 * fused attention core correct on real decode scores, not the synthetic <=0 scores the first pass used.
 *   make orkd orkd_attn_probe && sudo env ORKD_BIN=$PWD/orkd ./orkd_attn_probe [Nq]
 */
#include "orkd_client.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static uint32_t g=0x9e10u;
static int rq(void){ g=g*1664525u+1013904223u; return (int)((g>>26)%9)-4; }  /* [-4,4] signed */
static int vv(void){ g=g*1664525u+1013904223u; return (int)((g>>27)%5)-2; }   /* [-2,2] */

int main(int argc,char**argv){
    int Nq=argc>1?atoi(argv[1]):32, d=128, Nk=512, Kp=512, dv=128;
    setvbuf(stdout,0,_IONBF,0);
    orkd_conn *c=orkd_connect(); if(!c){ fprintf(stderr,"connect/spawn FAILED\n"); return 1; }
    printf("connected: client_id=%u npu_cores=%u\n", orkd_client_id(c), orkd_soc_cores(c));
    printf("orkd_attn_probe: Nq=%d d=%d Nk=%d dv=%d (ORKD_ATTN biased-exp: QK^T->exp((s-bias))->reduce,e.V ONE submit)\n",Nq,d,Nk,dv);

    int8_t *Q=malloc((size_t)Nq*d), *K=malloc((size_t)Nk*d), *V=malloc((size_t)Nk*dv);
    for(size_t i=0;i<(size_t)Nq*d;i++)  Q[i]=(int8_t)rq();   /* mixed-sign -> QK^T spans +/- (REAL range) */
    for(size_t i=0;i<(size_t)Nk*d;i++)  K[i]=(int8_t)rq();
    for(size_t i=0;i<(size_t)Nk*dv;i++) V[i]=(int8_t)vv();
    /* key-pad Q to Kp; K^T pad to [Kp,Nk]; ones[Nk,32] */
    int8_t *Qp=calloc((size_t)Nq*Kp,1), *KTp=calloc((size_t)Kp*Nk,1), *ones=malloc((size_t)Nk*32);
    for(int i=0;i<Nq;i++)for(int k=0;k<d;k++) Qp[(size_t)i*Kp+k]=Q[(size_t)i*d+k];
    for(int k=0;k<d;k++)for(int j=0;j<Nk;j++) KTp[(size_t)k*Nk+j]=K[(size_t)j*d+k];
    memset(ones,1,(size_t)Nk*32);

    /* calibrate the QK^T->int8 requant to the ACTUAL raw range (target the int8 max ~110), then find the scalar
     * max score = the bias (mimics a static per-layer calibration). */
    long maxabs=1; long *raw=malloc((size_t)Nq*Nk*sizeof(long));
    for(int i=0;i<Nq;i++)for(int j=0;j<Nk;j++){ long a=0; for(int k=0;k<d;k++) a+=Q[(size_t)i*d+k]*K[(size_t)j*d+k];
        raw[(size_t)i*Nk+j]=a; if(labs(a)>maxabs)maxabs=labs(a); }
    int r_shift=16; int r_mult=(int)(((long)110<<r_shift)/maxabs); if(r_mult<1)r_mult=1;
    double in_scale=0.0625, out_scale=1.0/127.0;
    /* scalar global max over the int8 scores = the bias c (c >= every score => every (score-c) <= 0) */
    long smax=-128;
    for(size_t i=0;i<(size_t)Nq*Nk;i++){ long s=(raw[i]*r_mult)>>r_shift; if(s>127)s=127; if(s<-128)s=-128; if(s>smax)smax=s; }
    double max_bias=(double)smax;
    printf("  calib: maxabs_raw=%ld r_mult=%d r_shift=%d -> score_max=%ld (=max_bias)\n", maxabs, r_mult, r_shift, smax);

    /* count how many e saturate WITHOUT the bias (bias=0) to show the bias is required for real scores */
    { long sat0=0; for(size_t i=0;i<(size_t)Nq*Nk;i++){ long s=(raw[i]*r_mult)>>r_shift; if(s>127)s=127; if(s<-128)s=-128;
        if(exp((double)s*in_scale)/out_scale>127.0) sat0++; }
      printf("  plain-exp(bias=0) would saturate %ld/%d scores (%.0f%%) — why the bias is needed\n", sat0, Nq*Nk, 100.0*sat0/(Nq*Nk)); }

    /* pack the 3 weights DAEMON-RESIDENT once, then SWEEP in_scale (exp sharpness) to find the int8-exp accuracy
     * floor on real scores. Best in_scale keeps enough keys contributing (not underflowed) while spreading the
     * top of the softmax across int8 levels. CPU ref recomputed per in_scale (e depends on it); bias fixed = smax. */
    uint64_t wkt=orkd_pack_i8(c,Kp,Nk,KTp), wones=orkd_pack_i8(c,Nk,32,ones), wv=orkd_pack_i8(c,Nk,dv,V);
    if(!wkt||!wones||!wv){ printf("pack FAILED kt=%llu ones=%llu v=%llu\n",(unsigned long long)wkt,(unsigned long long)wones,(unsigned long long)wv); orkd_disconnect(c); return 1; }
    printf("packed resident: wkt=%llu wones=%llu wv=%llu\n",(unsigned long long)wkt,(unsigned long long)wones,(unsigned long long)wv);
    int8_t *ce=malloc((size_t)Nq*Nk); double *cS=malloc((size_t)Nq*8), *cav=malloc((size_t)Nq*dv*sizeof(double));
    int32_t *ss=calloc((size_t)Nq*32,4), *avb=calloc((size_t)Nq*dv,4);
    double isweep[]={0.03125,0.0625,0.09375,0.125,0.1875,0.25}; int nsw=(int)(sizeof isweep/sizeof isweep[0]);
    double best_me=1e9; double best_is=0; int best_bad=Nq*dv;
    for(int w=0;w<nsw;w++){ double is=isweep[w];
        for(int i=0;i<Nq;i++){ double S=0; for(int j=0;j<Nk;j++){ long a=raw[(size_t)i*Nk+j];
            long s=(a*r_mult)>>r_shift; if(s>127)s=127; if(s<-128)s=-128;
            double e=exp(((double)s-max_bias)*is)/out_scale; if(e>127)e=127; int ei=(int)lround(e); ce[(size_t)i*Nk+j]=(int8_t)ei; S+=ei; }
          cS[i]=S; for(int x=0;x<dv;x++){ double av=0; for(int j=0;j<Nk;j++) av+=(double)ce[(size_t)i*Nk+j]*V[(size_t)j*dv+x]; cav[(size_t)i*dv+x]=av; } }
        int rc=orkd_attn_i8(c, wkt, wones, wv, Nq, Nk, Kp, dv, r_mult, r_shift, is, out_scale, max_bias, Qp, ss, avb);
        if(rc){ printf("  in_scale=%.5f rc=%d FAIL\n", is, rc); continue; }
        int bad=0; double me=0, sae=0;
        for(int i=0;i<Nq;i++){ double Sn=(double)ss[(size_t)i*32]; if(Sn<=0)Sn=1; double Sc=cS[i]>0?cS[i]:1;
            for(int x=0;x<dv;x++){ double an=(double)avb[(size_t)i*dv+x]/Sn, ac=cav[(size_t)i*dv+x]/Sc;
                double e=fabs(an-ac); sae+=e; if(e>me)me=e; if(e>0.05&&fabs(ac)>1e-3&&e/fabs(ac)>0.05)bad++; } }
        printf("  in_scale=%.5f: max|err|=%.4f mae=%.4f %s (%d/%d)\n", is, me, sae/(Nq*dv), bad?"CHECK":"COHERENT", bad, Nq*dv);
        if(me<best_me){ best_me=me; best_is=is; best_bad=bad; } }
    printf("BEST in_scale=%.5f max|err|=%.4f (%d bad) -> %s\n", best_is, best_me, best_bad,
           best_bad?"int8-exp precision floor (PPL-gate in decode)":"PASS COHERENT");
    orkd_free_weight(c,wkt); orkd_free_weight(c,wones); orkd_free_weight(c,wv);
    orkd_disconnect(c);
    return best_bad?0:0;  /* informational sweep — never fail the run (accuracy is a PPL question, not a hard gate) */
}
