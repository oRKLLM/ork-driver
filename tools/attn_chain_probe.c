/* attn_chain_probe — task #20: assemble a FULL single-head attention layer as an ork_submit_seq pipeline
 * and validate coherence vs a CPU reference. Pre-norm transformer attention sublayer, fp16, full (non-causal).
 *
 * Assembled dataflow (each NPU op via ork_submit_seq; [CPU] = host bridge/marshalling between ops):
 *   xn   = RMSNorm(x, wn)         RMSNORM_F16   (eps via in_scale, gain via B)
 *   Q    = xn . Wq               MM_F16
 *   K    = xn . Wk               MM_F16
 *   V    = xn . Wv               MM_F16
 *   Q    = rope(Q, pos)          ROPE_NEOX_F16  (pos via B, freq_base via in_scale)
 *   K    = rope(K, pos)          ROPE_NEOX_F16
 *   [CPU] transpose+pack K^T, V as runtime weights (computed activations -> the densify/repack cost)
 *   scores = Q . K^T             MM_F16
 *   [CPU] scale 1/sqrt(d), quantize -> int8
 *   max  = rowmax(scores)        REDUCEMAX_I8
 *   [CPU] x-max, quantize -> int16
 *   e    = exp(x-max)            EXP_I16
 *   [CPU] int16 -> f16
 *   Sig  = e . ones[Nk,16]       MM_F16
 *   [CPU] 1/Sigma, normalize -> P
 *   av   = P . V                 MM_F16
 *   O    = av . Wo               MM_F16
 *   y    = x + O                 ADD_F16
 *
 * Proves the whole layer composes coherently through the seq adapters. The [CPU] bridges (norm/quant/x-max/
 * requant/1-over-Sigma + the runtime K^T/V densify+pack) are the residency + marshalling gaps for a later
 * single-submit resident layer.
 *
 * BOARD:  make attn_chain_probe && sudo env ORK_MM_TIMEOUT=3000 timeout 300 ./attn_chain_probe [N] [d]
 * Exit 0 = coherent; nonzero = a stage failed / miscomputed.
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static uint32_t g_rng=0x2ab4c1u;
static float frand(void){ g_rng=g_rng*1664525u+1013904223u; return (float)(g_rng>>8)/(float)(1u<<24); }
static float rn(void){ return frand()*2.f-1.f; }

int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):64;    /* tokens (queries==keys, self-attention) — n%32 (exp), %16 (f16) */
    int d=argc>2?atoi(argv[2]):128;   /* head dim (%16 f16, %8 rope, %8 rmsnorm)                         */
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    printf("attn_chain_probe: N=%d d=%d (full single-head pre-norm attention layer)\n", N,d);
    int fail=0;
    float invsqrt=1.0f/sqrtf((float)d), eps=1e-5f, freq_base=10000.f;

    /* --- inputs + weights --- */
    ork_f16 *x=malloc((size_t)N*d*2), *wn=malloc((size_t)d*2);
    ork_f16 *Wq=malloc((size_t)d*d*2), *Wk=malloc((size_t)d*d*2), *Wv=malloc((size_t)d*d*2), *Wo=malloc((size_t)d*d*2);
    int *pos=malloc((size_t)N*sizeof(int));
    for(size_t i=0;i<(size_t)N*d;i++) x[i]=(ork_f16)rn();
    for(int i=0;i<d;i++) wn[i]=(ork_f16)(0.5f+frand());
    for(size_t i=0;i<(size_t)d*d;i++){ Wq[i]=(ork_f16)(rn()*0.1f); Wk[i]=(ork_f16)(rn()*0.1f); Wv[i]=(ork_f16)(rn()*0.1f); Wo[i]=(ork_f16)(rn()*0.1f); }
    for(int i=0;i<N;i++) pos[i]=i;

    /* ===================== CPU REFERENCE (full layer) ===================== */
    float *ref=malloc((size_t)N*d*sizeof(float));
    {
        float *xn=malloc((size_t)N*d*4), *Q=malloc((size_t)N*d*4), *K=malloc((size_t)N*d*4), *Vv=malloc((size_t)N*d*4);
        for(int i=0;i<N;i++){ double ms=0; for(int k=0;k<d;k++){ float v=(float)x[(size_t)i*d+k]; ms+=(double)v*v; } float r=1.0f/sqrtf((float)(ms/d)+eps);
            for(int k=0;k<d;k++) xn[(size_t)i*d+k]=(float)x[(size_t)i*d+k]*r*(float)wn[k]; }
        for(int i=0;i<N;i++) for(int o=0;o<d;o++){ double q=0,kk=0,vv=0; for(int k=0;k<d;k++){ float a=xn[(size_t)i*d+k];
            q+=(double)a*(float)Wq[(size_t)k*d+o]; kk+=(double)a*(float)Wk[(size_t)k*d+o]; vv+=(double)a*(float)Wv[(size_t)k*d+o]; }
            Q[(size_t)i*d+o]=(float)q; K[(size_t)i*d+o]=(float)kk; Vv[(size_t)i*d+o]=(float)vv; }
        /* rope Q,K */
        int hd2=d/2; for(int r=0;r<N;r++){ double p=(double)pos[r]; for(int i=0;i<hd2;i++){ double th=p*pow((double)freq_base,-2.0*(double)i/(double)d);
            float cc=(float)cos(th),ss=(float)sin(th);
            float qa=Q[(size_t)r*d+i],qb=Q[(size_t)r*d+i+hd2]; Q[(size_t)r*d+i]=qa*cc-qb*ss; Q[(size_t)r*d+i+hd2]=qb*cc+qa*ss;
            float ka=K[(size_t)r*d+i],kb=K[(size_t)r*d+i+hd2]; K[(size_t)r*d+i]=ka*cc-kb*ss; K[(size_t)r*d+i+hd2]=kb*cc+ka*ss; } }
        float *sc=malloc((size_t)N*4);
        float *av=malloc((size_t)N*d*4);
        for(int i=0;i<N;i++){ float mx=-1e30f; for(int j=0;j<N;j++){ double s=0; for(int k=0;k<d;k++) s+=(double)Q[(size_t)i*d+k]*K[(size_t)j*d+k];
                sc[j]=(float)s*invsqrt; if(sc[j]>mx)mx=sc[j]; }
            double sum=0; for(int j=0;j<N;j++){ sc[j]=expf(sc[j]-mx); sum+=sc[j]; }
            for(int k=0;k<d;k++){ double o=0; for(int j=0;j<N;j++) o+=(double)sc[j]*Vv[(size_t)j*d+k]; av[(size_t)i*d+k]=(float)(o/sum); } }
        /* O = av . Wo ; y = x + O */
        for(int i=0;i<N;i++) for(int o=0;o<d;o++){ double s=0; for(int k=0;k<d;k++) s+=(double)av[(size_t)i*d+k]*(float)Wo[(size_t)k*d+o];
            ref[(size_t)i*d+o]=(float)x[(size_t)i*d+o]+(float)s; }
        free(xn);free(Q);free(K);free(Vv);free(sc);free(av);
    }

    /* ===================== NPU SEQ PIPELINE ===================== */
    ork_w *w_q=ork_f16_mm_pack(c,d,d,Wq), *w_k=ork_f16_mm_pack(c,d,d,Wk), *w_v=ork_f16_mm_pack(c,d,d,Wv), *w_o=ork_f16_mm_pack(c,d,d,Wo);
    ork_f16 *ones=malloc((size_t)N*16*2); for(size_t i=0;i<(size_t)N*16;i++) ones[i]=(ork_f16)1.0f;
    ork_w *w_ones=ork_f16_mm_pack(c,N,16,ones);
    if(!w_q||!w_k||!w_v||!w_o||!w_ones){ printf("pack fail\n"); return 2; }
    int rc[16]; int ri=0;
    ork_f16 *xn=malloc((size_t)N*d*2), *Q=malloc((size_t)N*d*2), *K=malloc((size_t)N*d*2), *Vv=malloc((size_t)N*d*2);

    /* 1: RMSNorm */
    { ork_seq_op o={ .kind=ORK_OP_RMSNORM_F16, .M=N, .N=d, .A=x, .B=wn, .C=xn, .in_scale=eps }; rc[ri++]=ork_submit_seq(c,&o,1); }
    /* 2-4: Q/K/V = xn . W*  (MM_F16 outputs f32 -> narrow to f16 for rope/attention) */
    float *Qf=malloc((size_t)N*d*4), *Kf=malloc((size_t)N*d*4), *Vf=malloc((size_t)N*d*4);
    { ork_seq_op o={ .kind=ORK_OP_MM_F16, .w=w_q, .M=N, .A=xn, .C=Qf }; rc[ri++]=ork_submit_seq(c,&o,1); }
    { ork_seq_op o={ .kind=ORK_OP_MM_F16, .w=w_k, .M=N, .A=xn, .C=Kf }; rc[ri++]=ork_submit_seq(c,&o,1); }
    { ork_seq_op o={ .kind=ORK_OP_MM_F16, .w=w_v, .M=N, .A=xn, .C=Vf }; rc[ri++]=ork_submit_seq(c,&o,1); }
    for(size_t i=0;i<(size_t)N*d;i++){ Q[i]=(ork_f16)Qf[i]; K[i]=(ork_f16)Kf[i]; Vv[i]=(ork_f16)Vf[i]; }
    /* 5-6: rope Q,K */
    ork_f16 *Qr=malloc((size_t)N*d*2), *Kr=malloc((size_t)N*d*2);
    { ork_seq_op o={ .kind=ORK_OP_ROPE_NEOX_F16, .M=N, .N=d, .A=Q, .B=pos, .C=Qr, .in_scale=freq_base }; rc[ri++]=ork_submit_seq(c,&o,1); }
    { ork_seq_op o={ .kind=ORK_OP_ROPE_NEOX_F16, .M=N, .N=d, .A=K, .B=pos, .C=Kr, .in_scale=freq_base }; rc[ri++]=ork_submit_seq(c,&o,1); }

    /* [CPU] transpose+pack K^T[d,N] and V[N,d] as runtime weights */
    ork_f16 *KrT=malloc((size_t)d*N*2); for(int j=0;j<N;j++) for(int k=0;k<d;k++) KrT[(size_t)k*N+j]=Kr[(size_t)j*d+k];
    ork_w *w_kt=ork_f16_mm_pack(c,d,N,KrT);
    ork_w *w_vv=ork_f16_mm_pack(c,N,d,Vv);
    if(!w_kt||!w_vv){ printf("runtime pack fail\n"); return 2; }

    /* 7: scores = Qr . K^T */
    float *scores=malloc((size_t)N*N*4);
    { ork_seq_op o={ .kind=ORK_OP_MM_F16, .w=w_kt, .M=N, .A=Qr, .C=scores }; rc[ri++]=ork_submit_seq(c,&o,1); }
    /* [CPU] scale + int8 quant */
    for(size_t i=0;i<(size_t)N*N;i++) scores[i]*=invsqrt;
    float amax=0; for(size_t i=0;i<(size_t)N*N;i++){ float a=fabsf(scores[i]); if(a>amax)amax=a; } if(amax<=0)amax=1; double sq=amax/127.0;
    int8_t *q8=malloc((size_t)N*N); for(size_t i=0;i<(size_t)N*N;i++){ long v=lround(scores[i]/sq); if(v<-127)v=-127; if(v>127)v=127; q8[i]=(int8_t)v; }
    /* 8: row-max */
    int8_t *maxq=malloc((size_t)N); for(int i=0;i<N;i++) maxq[i]=-128;
    { ork_seq_op o={ .kind=ORK_OP_REDUCEMAX_I8, .M=N, .N=N, .A=q8, .C=maxq }; rc[ri++]=ork_submit_seq(c,&o,1); }
    /* [CPU] x-max, int16 quant */
    float lo=0; for(int i=0;i<N;i++){ float mf=maxq[i]*(float)sq; for(int j=0;j<N;j++){ float dd=scores[(size_t)i*N+j]-mf; if(dd<lo)lo=dd; } }
    double in_scale=(-lo)/32000.0; if(in_scale<=0)in_scale=1e-6; double out_scale=1.0/32000.0;
    int16_t *xi=malloc((size_t)N*N*2), *ei=malloc((size_t)N*N*2);
    for(int i=0;i<N;i++){ float mf=maxq[i]*(float)sq; for(int j=0;j<N;j++){ long v=lround((double)(scores[(size_t)i*N+j]-mf)/in_scale); if(v<-32768)v=-32768; if(v>32767)v=32767; xi[(size_t)i*N+j]=(int16_t)v; } }
    /* 9: exp */
    { ork_seq_op o={ .kind=ORK_OP_EXP_I16, .M=N, .N=N, .A=xi, .C=ei, .in_scale=in_scale, .out_scale=out_scale }; rc[ri++]=ork_submit_seq(c,&o,1); }
    /* [CPU] int16 -> f16 */
    ork_f16 *ef=malloc((size_t)N*N*2); for(size_t i=0;i<(size_t)N*N;i++) ef[i]=(ork_f16)((double)ei[i]*out_scale);
    /* 10: Sigma */
    float *sig=malloc((size_t)N*16*4);
    { ork_seq_op o={ .kind=ORK_OP_MM_F16, .w=w_ones, .M=N, .A=ef, .C=sig }; rc[ri++]=ork_submit_seq(c,&o,1); }
    /* [CPU] normalize -> P */
    ork_f16 *P=malloc((size_t)N*N*2);
    for(int i=0;i<N;i++){ float S=sig[(size_t)i*16]; float inv=S>0?1.0f/S:0.f; for(int j=0;j<N;j++) P[(size_t)i*N+j]=(ork_f16)((float)ef[(size_t)i*N+j]*inv); }
    /* 11: av = P . V */
    float *avf=malloc((size_t)N*d*4);
    { ork_seq_op o={ .kind=ORK_OP_MM_F16, .w=w_vv, .M=N, .A=P, .C=avf }; rc[ri++]=ork_submit_seq(c,&o,1); }
    ork_f16 *av=malloc((size_t)N*d*2); for(size_t i=0;i<(size_t)N*d;i++) av[i]=(ork_f16)avf[i];
    /* 12: O = av . Wo */
    float *Of=malloc((size_t)N*d*4);
    { ork_seq_op o={ .kind=ORK_OP_MM_F16, .w=w_o, .M=N, .A=av, .C=Of }; rc[ri++]=ork_submit_seq(c,&o,1); }
    /* 13: residual y = x + O  (ADD_F16) */
    ork_f16 *Oh=malloc((size_t)N*d*2); for(size_t i=0;i<(size_t)N*d;i++) Oh[i]=(ork_f16)Of[i];
    ork_f16 *y=malloc((size_t)N*d*2);
    { ork_seq_op o={ .kind=ORK_OP_ADD_F16, .M=N, .N=d, .A=x, .B=Oh, .C=y }; rc[ri++]=ork_submit_seq(c,&o,1); }

    /* coherence */
    int anyrc=0; for(int i=0;i<ri;i++) if(rc[i]) anyrc=1;
    { static const char*L[]={"rmsnorm","Q=xn.Wq","K=xn.Wk","V=xn.Wv","rope(Q)","rope(K)","scores=Q.K^T","rowmax","exp","Sigma=e.ones","av=P.V","O=av.Wo","y=x+O"};
      for(int i=0;i<ri;i++) if(rc[i]||getenv("ORK_DUMP_RC")) printf("    op[%2d] %-14s rc=%d\n", i, i<(int)(sizeof L/sizeof*L)?L[i]:"?", rc[i]); }
    double me=0,sae=0; int bad=0;
    for(size_t i=0;i<(size_t)N*d;i++){ double e=fabs((double)y[i]-(double)ref[i]); sae+=e; if(e>me)me=e; if(e>3e-2*(fabs((double)ref[i])+1e-2)) bad++; }
    printf("  %d seq ops, all rc==0: %s | max|err|=%.3e mae=%.3e  %s (%d/%d off)\n",
           ri, anyrc?"NO":"yes", me, sae/((double)N*d), (!anyrc&&!bad)?"COHERENT":"CHECK", bad, N*d);
    if(anyrc||bad) fail=1;

    printf("%s\n", fail? "FAIL — full attention layer seq miscomputed" : "PASS — full single-head attention layer assembled + coherent via ork_submit_seq");
    ork_mm_free(c,w_q);ork_mm_free(c,w_k);ork_mm_free(c,w_v);ork_mm_free(c,w_o);ork_mm_free(c,w_ones);ork_mm_free(c,w_kt);ork_mm_free(c,w_vv);
    ork_npu_free(c);
    return fail;
}
