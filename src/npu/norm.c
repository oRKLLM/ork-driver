/* npu/norm.c — on-NPU normalization and softmax primitives (RMSNorm, L2-norm, softmax, RoPE, the
 * fast Walsh-Hadamard normaliser) plus the reduce/rsqrt helpers they share.
 *
 * An OP FAMILY, not a precision: these are fp16-typed but their identity is the operation, so they sit
 * beside sdp.c and ssm.c rather than inside f16/. Gated behind ORK_NORM_NPU / ORK_SOFTMAX_NPU — the
 * default is still CPU. Lifted from npu.c by the round-1 sweep. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ork_regs.h"
#include "regcmd_array_4x32x16.h"
#include "regcmd_i8.h"
#include "npu/internal.h"
#include "npu/core.h"

/* module-owned state: the resident all-ones reduce weight and the rsqrt LUT cache */
ork_w *orki_ones_w=NULL; static int orki_ones_n=0; static ork_npu *orki_ones_ctx=NULL;
struct { ork_npu *c; int nf; double eps, lo, hi, osc; ork_w *wS; int16_t lut[1030]; int valid; } orki_rs = {0};
#define ORK_RSQRT_KD 512

static ork_w *norm_reduce_w(ork_npu *c,int n){
    if(orki_ones_w && orki_ones_n==n && orki_ones_ctx==c) return orki_ones_w;
    if(orki_ones_w){ ork_mm_free(orki_ones_ctx,orki_ones_w); orki_ones_w=NULL; }
    f16 *ones=malloc((size_t)n*16*sizeof(f16)); if(!ones) return NULL;
    for(size_t i=0;i<(size_t)n*16;i++) ones[i]=(f16)1.0f;
    orki_ones_w=ork_f16_mm_pack(c,n,16,ones); free(ones);
    if(orki_ones_w){ orki_ones_n=n; orki_ones_ctx=c; }
    return orki_ones_w;
}

/* rmsnorm/l2norm: reduction sum(x^2) on the NPU (ork_norm_reduce_npu, any n via K-split) when ORK_NORM_NPU
 * is set; rsqrt + scale on CPU. Validated 0.0005 vs the CPU ref. (The fully-fused single-submit reduce+rsqrt
 * — ork_f16_mm_build_rsqrt_lut, validated 0.0012 standalone for n<=2048 in test_bmm's rsqrt test — is the
 * building block for a one-submit on-NPU norm; wiring it into the general path needs LUT pre-calibration, so
 * the shipped norm keeps rsqrt on CPU for robustness across all n.) The ork_norm_rsqrt_npu helper above is
 * the decoupled K=32 rsqrt op, kept for that follow-up. */
int ork_norm_reduce_npu(ork_npu *c,int M,int n,const f16 *x,float *ss_out){
    if(!ork_norm_npu_enabled() || n%32) return -1;      /* K%32 for the matmul */
    ork_w *ow=norm_reduce_w(c,n); if(!ow) return -1;
    f16 *sq=malloc((size_t)M*n*sizeof(f16)); float *ss16=malloc((size_t)M*16*sizeof(float));
    int rc=-1;
    if(sq&&ss16){
        for(size_t i=0;i<(size_t)M*n;i++){ float v=(float)x[i]; sq[i]=(f16)(v*v); }
        if(ork_f16_mm_run(c,ow,M,sq,ss16)==0){ for(int m=0;m<M;m++) ss_out[m]=ss16[(size_t)m*16]; rc=0; }
    }
    free(sq); free(ss16); return rc;
}

static int rsqrt_lut_ensure(ork_npu *c,int nf,double eps,double ss_lo,double ss_hi){
    if(orki_rs.valid && orki_rs.c==c && orki_rs.nf==nf && orki_rs.eps==eps && ss_lo>=orki_rs.lo && ss_hi<=orki_rs.hi) return 0;
    /* TIGHT padding: the 64-point PWL LUT must keep resolution IN the actual ss band (a wide range spreads the
     * probe points thin -> big rsqrt error). ~1.3x span keeps ~0.1% accuracy; drift outside triggers a rebuild. */
    double lo=ss_lo*0.9, hi=ss_hi*1.15; if(hi<=0) return -1;
    double fl=(double)nf*eps; if(lo<fl) lo=fl; if(lo>=hi) lo=hi*0.5;
    double S,R,osc; int16_t lut[1030];
    if(ork_f16_mm_build_rsqrt_lut(c,nf,eps,lo,hi,lut,&S,&R,&osc)) return -1;
    f16 *B=malloc((size_t)ORK_RSQRT_KD*16*sizeof(f16)); if(!B) return -1;
    for(int i=0;i<ORK_RSQRT_KD*16;i++) B[i]=(f16)(-S*hi/(double)ORK_RSQRT_KD);   /* acc = sum_k (ss/hi)*(-S*hi/Kd) = -S*ss */
    ork_w *wS=ork_f16_mm_pack(c,ORK_RSQRT_KD,16,B); free(B); if(!wS) return -1;
    if(orki_rs.wS) ork_mm_free(orki_rs.c,orki_rs.wS);
    memcpy(orki_rs.lut,lut,sizeof lut); orki_rs.wS=wS; orki_rs.c=c; orki_rs.nf=nf; orki_rs.eps=eps; orki_rs.lo=lo; orki_rs.hi=hi; orki_rs.osc=osc; orki_rs.valid=1;
    return 0;
}

int ork_norm_rsqrt_npu(ork_npu *c,int M,int nf,double eps,const float *ss,float *scale){
    double lo=1e30,hi=0; for(int m=0;m<M;m++){ if(ss[m]<lo)lo=ss[m]; if(ss[m]>hi)hi=ss[m]; }
    if(hi<=0 || rsqrt_lut_ensure(c,nf,eps,lo,hi)) return -1;
    double G=orki_rs.hi;
    f16 *A=malloc((size_t)M*ORK_RSQRT_KD*sizeof(f16)); float *C=malloc((size_t)M*16*sizeof(float));
    int rc=-1;
    if(A&&C){ for(int m=0;m<M;m++){ f16 v=(f16)(ss[m]/G); for(int k=0;k<ORK_RSQRT_KD;k++) A[(size_t)m*ORK_RSQRT_KD+k]=v; } /* dense ss/G */
        if(ork_f16_mm_run_silu(c,orki_rs.wS,M,A,C,0,0xffffc000u,0x56391100u,orki_rs.lut,1030)==0){
            for(int m=0;m<M;m++) scale[m]=(float)((double)C[(size_t)m*16]*orki_rs.osc); rc=0; } }
    free(A); free(C); return rc;
}

void ork_fwht_norm(float *v, int n){
    for(int len=1; len<n; len<<=1)
        for(int i=0;i<n;i+=len<<1)
            for(int j=i;j<i+len;j++){ float a=v[j], b=v[j+len]; v[j]=a+b; v[j+len]=a-b; }
    float s=1.0f/sqrtf((float)n);
    for(int i=0;i<n;i++) v[i]*=s;
}
int ork_f16_npu_rmsnorm(ork_npu *c,int M,int n,const f16 *x,const f16 *w,float eps,f16 *out){
    if(!c||!x||!w||!out||M<1||n<1) return -2;
    float *ss=malloc((size_t)M*sizeof(float)), *sc=malloc((size_t)M*sizeof(float)); int have_ss=0, have_sc=0;
    if(ss && ork_norm_reduce_npu(c,M,n,x,ss)==0) have_ss=1;                 /* sum(x^2) on NPU (any n) */
    if(have_ss && sc && ork_norm_rsqrt_npu(c,M,n,(double)eps,ss,sc)==0) have_sc=1; /* rsqrt on NPU (K=512) */
    for(int m=0;m<M;m++){ const f16 *xr=x+(size_t)m*n; f16 *o=out+(size_t)m*n; float s;
        if(have_sc) s=sc[m];
        else { double sumsq; if(have_ss) sumsq=(double)ss[m]; else { sumsq=0; for(int i=0;i<n;i++){ double v=(double)xr[i]; sumsq+=v*v; } }
               s=(float)(1.0/sqrt(sumsq/(double)n+(double)eps)); }
        for(int i=0;i<n;i++) o[i]=(f16)((float)xr[i]*s*(float)w[i]); }
    free(ss); free(sc); return 0;
}

int ork_f16_npu_l2norm(ork_npu *c,int M,int n,const f16 *x,float eps,f16 *out){
    if(!c||!x||!out||M<1||n<1) return -2;
    float *ss=malloc((size_t)M*sizeof(float)), *sc=malloc((size_t)M*sizeof(float)); int have_ss=0, have_sc=0;
    if(ss && ork_norm_reduce_npu(c,M,n,x,ss)==0) have_ss=1;
    if(have_ss && sc && ork_norm_rsqrt_npu(c,M,1,(double)eps,ss,sc)==0) have_sc=1; /* nf=1: 1/sqrt(ss+eps) */
    for(int m=0;m<M;m++){ const f16 *xr=x+(size_t)m*n; f16 *o=out+(size_t)m*n; float s;
        if(have_sc) s=sc[m];
        else { double sumsq; if(have_ss) sumsq=(double)ss[m]; else { sumsq=0; for(int i=0;i<n;i++){ double v=(double)xr[i]; sumsq+=v*v; } }
               s=(float)(1.0/sqrt(sumsq+(double)eps)); }
        for(int i=0;i<n;i++) o[i]=(f16)((float)xr[i]*s); }
    free(ss); free(sc); return 0;
}
int ork_f16_npu_rope_neox(ork_npu *c, const ork_f16 *x, int hd, int nrow, const int *pos, double freq_base, ork_f16 *out){
    if(!c||!x||!pos||!out||hd<2||(hd&7)||nrow<1) return -2;
    int hd2=hd/2; size_t sz=(size_t)nrow*hd*sizeof(ork_f16);
    ork_f16 *cosT=malloc(sz),*sinT=malloc(sz),*xr=malloc(sz),*t1=malloc(sz),*t2=malloc(sz);
    if(!cosT||!sinT||!xr||!t1||!t2){ free(cosT);free(sinT);free(xr);free(t1);free(t2); return -1; }
    for(int r=0;r<nrow;r++){ double p=(double)pos[r];
        for(int i=0;i<hd2;i++){ double th=p*pow(freq_base,-2.0*(double)i/(double)hd); float cc=(float)cos(th), ss=(float)sin(th);
            cosT[(size_t)r*hd+i]=(ork_f16)cc; cosT[(size_t)r*hd+i+hd2]=(ork_f16)cc;
            sinT[(size_t)r*hd+i]=(ork_f16)(-ss); sinT[(size_t)r*hd+i+hd2]=(ork_f16)ss; }
        for(int i=0;i<hd2;i++){ xr[(size_t)r*hd+i]=x[(size_t)r*hd+i+hd2]; xr[(size_t)r*hd+i+hd2]=x[(size_t)r*hd+i]; } }
    int rc=0;
    if(ork_f16_npu_ewmul(c,x,cosT,nrow,hd,t1,NULL)) rc=-1;
    else if(ork_f16_npu_ewmul(c,xr,sinT,nrow,hd,t2,NULL)) rc=-1;
    else if(ork_f16_npu_add(c,t1,t2,nrow,hd,out,NULL)) rc=-1;
    free(cosT);free(sinT);free(xr);free(t1);free(t2);
    return rc;
}

/* On-NPU composed softmax over each row of [M][n]: y = exp(x-max)/Σexp(x-max). Gated ORK_SOFTMAX_NPU.
 * The per-row max and the final normalize (÷Σ) are CPU (cheap per-row scalars); the heavy parts run on the
 * NPU: exp via ork_i16_npu_exp (int16 SDP LUT; x-max quantized to a shared in_scale so exp maps in*in_scale
 * -> exp(x-max)), Σ via the reduction-as-matmul (e·ones[n,16], reusing the norm reduce weight). Like the
 * norm this is submit-floor-bound standalone (gated off; the win is fusing into the attention chain). Any
 * NPU-path failure (PPU-fuse off, n%32!=0, exp/reduce error) falls back to the full CPU softmax. 0/ok,-2. */
int ork_f16_npu_softmax(ork_npu *c,int M,int n,const f16 *x,f16 *out){
    if(!c||!x||!out||M<1||n<1) return -2;
    float *mx=malloc((size_t)M*sizeof(float)), *e=malloc((size_t)M*n*sizeof(float)), *s=malloc((size_t)M*sizeof(float));
    if(!mx||!e||!s){ free(mx);free(e);free(s); return -1; }
    for(int m=0;m<M;m++){ float mv=(float)x[(size_t)m*n]; for(int j=1;j<n;j++){ float v=(float)x[(size_t)m*n+j]; if(v>mv)mv=v; } mx[m]=mv; }
    int have_npu=0;
    /* Composition: max (CPU) -> exp(x-max) on the NPU (SDP act-LUT, int16) -> Sum + scale on CPU.
     * The Sum is intentionally NOT a reduce-matmul here: an activation(exp)->matmul(reduce) submit
     * reliably ETIMEDOUTs on the stateful activation->matmul mode-switch (the reverse order,
     * matmul->activation, is fine — cf. the rsqrt path), and self-healing per row-batch just discards
     * the good NPU exp. So exp rides the NPU (the transcendental win) and the cheap Sigma stays on CPU. */
    if(ork_softmax_npu_enabled() && n%32==0){
        float lo=0; for(int m=0;m<M;m++){ float mv=mx[m]; for(int j=0;j<n;j++){ float d=(float)x[(size_t)m*n+j]-mv; if(d<lo)lo=d; } }
        double in_scale=(-lo)/32000.0; if(in_scale<=0) in_scale=1e-6; double out_scale=1.0/32000.0;
        int16_t *xi=malloc((size_t)M*n*2), *ei=malloc((size_t)M*n*2);
        if(xi&&ei){
            for(int m=0;m<M;m++) for(int j=0;j<n;j++){ long q=lround(((double)((float)x[(size_t)m*n+j]-mx[m]))/in_scale); if(q<-32768)q=-32768; if(q>32767)q=32767; xi[(size_t)m*n+j]=(int16_t)q; }
            if(ork_i16_npu_exp(c,xi,M,n,in_scale,out_scale,ei,NULL)==0){                 /* exp(x-max) on NPU */
                for(int m=0;m<M;m++){ double sm=0; for(int j=0;j<n;j++){ double d=(double)ei[(size_t)m*n+j]*out_scale; e[(size_t)m*n+j]=(float)d; sm+=d; } s[m]=(float)sm; }
                have_npu=1;
            }
        }
        free(xi);free(ei);
    }
    if(!have_npu){ for(int m=0;m<M;m++){ double sm=0; for(int j=0;j<n;j++){ float d=expf((float)x[(size_t)m*n+j]-mx[m]); e[(size_t)m*n+j]=d; sm+=d; } s[m]=(float)sm; } }
    for(int m=0;m<M;m++){ float inv=1.0f/s[m]; for(int j=0;j<n;j++) out[(size_t)m*n+j]=(f16)(e[(size_t)m*n+j]*inv); }
    free(mx);free(e);free(s); return 0;
}
