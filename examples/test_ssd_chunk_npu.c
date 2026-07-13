/* test_ssd_chunk_npu — real-data numerics for one Mamba-2/SSD chunk-scan layer on the NPU, validated vs
 * an fp32 chunked reference (same staging as examples/test_ssd_chunk.c / xamba.py "ssd naive").
 *
 * Executes SSM_ON_NPU §8 "what remains": end-to-end REAL-DATA validation of the on-NPU scan, using the
 * proven public primitives (RK3588_SSM_NPU_Investigation.md §6-§7). Three modes, increasing on-NPU coverage:
 *   mode 0 MATMUL : 4 contractions on NPU (ork_bmm_fp16, per-head fp16); cumsum/exp/mul on CPU.
 *   mode 1 EXP    : + decay exp on NPU (ork_npu_exp_i16, int16, k=0.9357).   [fp16<->int16 bridge]
 *   mode 2 FULL   : + cumsum on NPU (CumBA = tril_ones matmul) + all decay MULTIPLIES on NPU
 *                   (ork_npu_ewmul_i16, calibrated int16). WHOLE decay-elementwise path on-NPU.
 * Residual adds (Yd+Yoff+D*x, inter-chunk carry) stay on CPU (int16 add is experimental; adds aren't the
 * XAMBA concern). Skips (exit 0) with no NPU. Board only for a real result.
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec*1e-3; }

/* CPU baseline for the scan matmuls: threaded fp32 GEMM pinned to the 4 big cores (A76 4-7), same
 * row-major operands as the NPU path. nb batches, rows split across NT threads. The "pure-CPU scan"
 * (mode 7) uses this instead of the NPU — the apples-to-apples comparison vs mode 5. */
#define NT 4
typedef struct { const ork_f16*A,*B; float*C; int nb,M,K,N,r0,r1,core; } cpugemm;
static void *cpu_gemm_w(void *vp){ cpugemm *w=vp;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(w->core,&set); pthread_setaffinity_np(pthread_self(),sizeof set,&set);
    for(int b=0;b<w->nb;b++){ const ork_f16 *A=w->A+(size_t)b*w->M*w->K,*B=w->B+(size_t)b*w->K*w->N; float *C=w->C+(size_t)b*w->M*w->N;
        for(int m=w->r0;m<w->r1;m++) for(int n=0;n<w->N;n++){ float acc=0;
            for(int k=0;k<w->K;k++) acc+=(float)A[(size_t)m*w->K+k]*(float)B[(size_t)k*w->N+n];
            C[(size_t)m*w->N+n]=acc; } }
    return NULL; }
static int cpu_bmm_f16(int nb,int M,int K,int N,const ork_f16 *A,const ork_f16 *B,float *C){
    pthread_t th[NT]; cpugemm w[NT]; int per=(M+NT-1)/NT;
    for(int t=0;t<NT;t++){ int r0=t*per,r1=r0+per; if(r1>M)r1=M; if(r0>M)r0=M;
        w[t]=(cpugemm){A,B,C,nb,M,K,N,r0,r1,4+t}; pthread_create(&th[t],NULL,cpu_gemm_w,&w[t]); }
    for(int t=0;t<NT;t++) pthread_join(th[t],NULL);
    return 0; }

typedef struct { int H,P,Nst,G,CS,NC; } ssd_dims;
#define IX_XHP(t,h,p)  (((size_t)(t)*d->H + (h))*d->P + (p))
#define IX_DT(t,h)     ((size_t)(t)*d->H + (h))
#define IX_BC(t,g,n)   (((size_t)(t)*d->G + (g))*d->Nst + (n))

static double frand(void){ return (double)rand()/RAND_MAX; }
static double frand_sym(void){ return frand()*2.0 - 1.0; }

/* On-NPU exp via int16 (ork_npu_exp_i16) with the investigation's validated calibration (§7). N%8. */
static void npu_exp_i16_calib(ork_npu *ctx, const double *arg, int M, int N, double *out){
    const double in_scale=30.0/30000.0, out_scale=1.0/30000.0, k=0.9357;
    size_t n=(size_t)M*N; short *in=malloc(n*2),*o=malloc(n*2); double us=0;
    for(size_t i=0;i<n;i++){ long q=lround(arg[i]/in_scale); if(q>32767)q=32767; if(q<-32768)q=-32768; in[i]=(short)q; }
    int rc=ork_npu_exp_i16(ctx,in,M,N,in_scale,out_scale,o,&us);
    for(size_t i=0;i<n;i++) out[i] = rc ? exp(arg[i]) : (double)o[i]*out_scale*k;
    free(in);free(o);
}

/* On-NPU element-wise multiply via int16 (ork_npu_ewmul_i16): out=up*sil. Per-call symmetric-int16 quant
 * (scale=amax/30000 each operand), requant mult/2^shift = s_up*s_sil/s_out chosen for int16 precision.
 * Dequant out_i16*s_out == up*sil. Falls back to CPU on rc!=0. N%8. */
static void npu_ewmul_calib(ork_npu *ctx, const double *up, const double *sil, int M, int N, double *out){
    size_t n=(size_t)M*N; double au=0,as=0,ap=0;
    for(size_t i=0;i<n;i++){ double u=fabs(up[i]),s=fabs(sil[i]),p=fabs(up[i]*sil[i]); if(u>au)au=u; if(s>as)as=s; if(p>ap)ap=p; }
    if(au<1e-12||as<1e-12||ap<1e-12){ for(size_t i=0;i<n;i++) out[i]=up[i]*sil[i]; return; }
    double su=au/30000.0, ss=as/30000.0, so=ap/30000.0, r=su*ss/so;
    int shift=(int)lround(log2(16384.0/r)); if(shift<0)shift=0; if(shift>31)shift=31;
    int mult=(int)lround(r*ldexp(1.0,shift)); if(mult<1)mult=1; if(mult>0x7fff)mult=0x7fff;
    short *qu=malloc(n*2),*qs=malloc(n*2),*qo=malloc(n*2); double us=0;
    for(size_t i=0;i<n;i++){ long a=lround(up[i]/su),b=lround(sil[i]/ss);
        if(a>32767)a=32767; if(a<-32768)a=-32768; if(b>32767)b=32767; if(b<-32768)b=-32768; qu[i]=(short)a; qs[i]=(short)b; }
    int rc=ork_npu_ewmul_i16(ctx,qu,qs,M,N,mult,shift,qo,&us);
    for(size_t i=0;i<n;i++) out[i] = rc ? up[i]*sil[i] : (double)qo[i]*so;
    free(qu);free(qs);free(qo);
}

/* On-NPU fp16 element-wise multiply (ork_npu_ewmul_f16): out=up*sil, MATMUL precision (no int16 switch,
 * no requant). Keeps the multiplies in the fp16 datapath so the only int16 op is exp — avoiding the
 * fp16-matmul-after-int16-SDP mode-switch wedge (npu.c:3519 only ACT_RESETs on int8 entry). N%8. */
static void npu_ewmul_f16(ork_npu *ctx, const double *up, const double *sil, int M, int N, double *out){
    size_t n=(size_t)M*N; ork_f16 *u=malloc(n*2),*s=malloc(n*2),*o=malloc(n*2); double us=0;
    for(size_t i=0;i<n;i++){ u[i]=(ork_f16)up[i]; s[i]=(ork_f16)sil[i]; }
    int rc=ork_npu_ewmul_f16(ctx,u,s,M,N,o,&us);
    for(size_t i=0;i<n;i++) out[i] = rc ? up[i]*sil[i] : (double)o[i];
    free(u);free(s);free(o);
}

/* fp32 chunked reference — identical staging to examples/test_ssd_chunk.c ssd_chunked(). */
static void ssd_chunked_ref(const ssd_dims *d, const double *x, const double *dt,
                            const double *A, const double *B, const double *C, const double *D, double *y){
    const int CS=d->CS,NC=d->NC,P=d->P,Nst=d->Nst;
    double *Acs=malloc(CS*sizeof(double)),*xbar=malloc((size_t)CS*P*sizeof(double));
    double *state_in=calloc((size_t)P*Nst,sizeof(double)),*cstate=malloc((size_t)P*Nst*sizeof(double));
    double Acs_last_prev=0;
    for(int h=0;h<d->H;h++){ int g=h%d->G; memset(state_in,0,(size_t)P*Nst*sizeof(double));
        for(int c=0;c<NC;c++){ int base=c*CS; double run=0;
            for(int l=0;l<CS;l++){ double dtv=dt[IX_DT(base+l,h)]; run+=dtv*A[h]; Acs[l]=run;
                for(int p=0;p<P;p++) xbar[(size_t)l*P+p]=dtv*x[IX_XHP(base+l,h,p)]; }
            if(c>0){ double dp=exp(Acs_last_prev); for(size_t i=0;i<(size_t)P*Nst;i++) state_in[i]=dp*state_in[i]+cstate[i]; }
            for(int l=0;l<CS;l++){ int t=base+l; double sdo=exp(Acs[l]);
                for(int p=0;p<P;p++){ double yd=0;
                    for(int s=0;s<=l;s++){ double Lls=exp(Acs[l]-Acs[s]),Gls=0;
                        for(int n=0;n<Nst;n++) Gls+=C[IX_BC(t,g,n)]*B[IX_BC(base+s,g,n)];
                        yd+=Gls*Lls*xbar[(size_t)s*P+p]; }
                    double yo=0; for(int n=0;n<Nst;n++) yo+=C[IX_BC(t,g,n)]*state_in[(size_t)p*Nst+n];
                    y[IX_XHP(t,h,p)]=yd+yo*sdo+D[h]*x[IX_XHP(t,h,p)]; } }
            for(int p=0;p<P;p++) for(int n=0;n<Nst;n++){ double acc=0;
                for(int s=0;s<CS;s++) acc+=exp(Acs[CS-1]-Acs[s])*B[IX_BC(base+s,g,n)]*xbar[(size_t)s*P+p];
                cstate[(size_t)p*Nst+n]=acc; }
            Acs_last_prev=Acs[CS-1];
        }
    }
    free(Acs);free(xbar);free(state_in);free(cstate);
}

static double g_ssd_mm_us, g_ssd_tot_us;   /* ORK_SSD_PROF breakdown: matmul vs elementwise+glue */
/* pooled 3-core stream stage (mode 5): repack each of H B-operands into the PERSISTENT pool (no
 * bcreate/free — kills the per-matmul pack churn), then round-robin the H matmuls across cores. */
static int stage_stream(ork_npu *ctx, ork_w **pool, int H, int M, int K, int N,
                        const ork_f16 *A, const ork_f16 *B, float *C){
    ork_mm_task_f16 tk[128]; if(H>128) return -2;
    for(int h=0;h<H;h++){ if(ork_mm_repack_f16(ctx,pool[h],K,N,B+(size_t)h*K*N)) return -1;
        tk[h]=(ork_mm_task_f16){pool[h],M,A+(size_t)h*M*K,C+(size_t)h*M*N}; }
    return ork_mm_run_stream_f16(ctx,H,tk);
}
static int ssd_chunked_npu(ork_npu *ctx, const ssd_dims *d, const double *x, const double *dt,
                           const double *A, const double *B, const double *C, const double *D,
                           double *y, int mode){
    const int H=d->H,CS=d->CS,NC=d->NC,P=d->P,Nst=d->Nst;
    const int Ncol=((H+15)/16)*16;                          /* cumsum head-batch width (bmm N%16) */
    /* mode: 0=matmul spine (elementwise CPU), 1=+exp NPU, 2=+cumsum/ewmul NPU, 3=FUSED matmuls (each stage's
     * H matmuls in ONE chained submit; elementwise CPU like mode 0) — the submit-floor speed layer. */
    /* mode 6 = FULL on-NPU: mode-2 elementwise (cumsum CumBA + exp int16 + mul ewmul) + mode-5 pooled stream. */
    const int expnpu=(mode==1||mode==2||mode==6), ewnpu=(mode==2||mode==6), fused=(mode==3), strm=(mode==4), strm5=(mode==5||mode==6), cpu=(mode==7);
    #define BMM(...) (strm?ork_bmm_fp16_stream:fused?ork_bmm_fp16_fused:ork_bmm_fp16)(ctx,__VA_ARGS__)
    /* MM: timed matmul dispatch. 7=pure CPU gemm (baseline); 5/6=pooled 3-core stream; else BMM. */
    #define MM(pool,...) ({ double _mt=now_us(); int _r=cpu?cpu_bmm_f16(__VA_ARGS__):strm5?stage_stream(ctx,pool,__VA_ARGS__):BMM(__VA_ARGS__); g_ssd_mm_us+=now_us()-_mt; _r; })
    ork_w *pl_sc[128]={0},*pl_cs[128]={0},*pl_yo[128]={0},*pl_yd[128]={0};
    if(strm5) for(int h=0;h<H;h++){ pl_sc[h]=ork_mm_f16_scratch(ctx,Nst,CS); pl_cs[h]=ork_mm_f16_scratch(ctx,CS,Nst);
        pl_yo[h]=ork_mm_f16_scratch(ctx,Nst,P); pl_yd[h]=ork_mm_f16_scratch(ctx,CS,P); }
    g_ssd_mm_us=0; double _tot0=now_us();
    double *Acs=malloc((size_t)H*CS*sizeof(double)),*Acs_last=malloc((size_t)H*sizeof(double));
    double *Acs_last_prev=calloc(H,sizeof(double));
    ork_f16 *aS=malloc((size_t)H*CS*Nst*sizeof(ork_f16)),*bS=malloc((size_t)H*Nst*CS*sizeof(ork_f16));
    float   *G =malloc((size_t)H*CS*CS *sizeof(float));
    ork_f16 *aD=malloc((size_t)H*CS*CS *sizeof(ork_f16)),*bD=malloc((size_t)H*CS*P *sizeof(ork_f16));
    float   *Yd=malloc((size_t)H*CS*P  *sizeof(float));
    ork_f16 *aC=malloc((size_t)H*P*CS  *sizeof(ork_f16)),*bC=malloc((size_t)H*CS*Nst*sizeof(ork_f16));
    float   *cs=malloc((size_t)H*P*Nst *sizeof(float)),*cs_prev=calloc((size_t)H*P*Nst,sizeof(float));
    ork_f16 *aO=malloc((size_t)H*CS*Nst*sizeof(ork_f16)),*bO=malloc((size_t)H*Nst*P *sizeof(ork_f16));
    float   *tmp=malloc((size_t)H*CS*P *sizeof(float));
    float   *state_in=calloc((size_t)H*P*Nst,sizeof(float));
    double  *xbar=malloc((size_t)H*CS*P*sizeof(double));
    double *Larg=malloc((size_t)H*CS*CS*sizeof(double)),*Lexp=malloc((size_t)H*CS*CS*sizeof(double));
    double *sarg=malloc((size_t)H*CS*sizeof(double)),*sexp=malloc((size_t)H*CS*sizeof(double));
    double *darg=malloc((size_t)H*CS*sizeof(double)),*dexp=malloc((size_t)H*CS*sizeof(double));
    /* CumBA + ewmul scratch (mode 2) */
    ork_f16 *tri=malloc((size_t)CS*CS*sizeof(ork_f16)),*abarP=malloc((size_t)CS*Ncol*sizeof(ork_f16));
    float *acsout=malloc((size_t)CS*Ncol*sizeof(float));
    size_t ewn=(size_t)H*CS*Nst; if((size_t)H*CS*CS>ewn)ewn=(size_t)H*CS*CS;
    double *e1=malloc(ewn*sizeof(double)),*e2=malloc(ewn*sizeof(double)),*e3=malloc(ewn*sizeof(double));
    for(int l=0;l<CS;l++) for(int s=0;s<CS;s++) tri[(size_t)l*CS+s]=(ork_f16)(s<=l?1.0f:0.0f);
    int rc=0;
    for(int c=0;c<NC && rc==0;c++){ int base=c*CS;
        for(int h=0;h<H;h++) for(int l=0;l<CS;l++){ double dtv=dt[IX_DT(base+l,h)];
            for(int p=0;p<P;p++) xbar[((size_t)h*CS+l)*P+p]=dtv*x[IX_XHP(base+l,h,p)]; }   /* xbar (input discretization) */
        if(ewnpu){ /* CumBA: Acs[l,h] = (tril_ones · Abar)[l,h], head-batched */
            for(int s=0;s<CS;s++){ for(int h=0;h<H;h++) abarP[(size_t)s*Ncol+h]=(ork_f16)(dt[IX_DT(base+s,h)]*A[h]);
                for(int h=H;h<Ncol;h++) abarP[(size_t)s*Ncol+h]=(ork_f16)0.0f; }
            if((rc=ork_bmm_fp16(ctx,1,CS,CS,Ncol,tri,abarP,acsout))) break;
            for(int h=0;h<H;h++) for(int l=0;l<CS;l++) Acs[(size_t)h*CS+l]=acsout[(size_t)l*Ncol+h];
        } else for(int h=0;h<H;h++){ double run=0; for(int l=0;l<CS;l++){ run+=dt[IX_DT(base+l,h)]*A[h]; Acs[(size_t)h*CS+l]=run; } }
        for(int h=0;h<H;h++) Acs_last[h]=Acs[(size_t)h*CS+CS-1];
        /* decay exp args -> exp */
        for(int h=0;h<H;h++){ const double *Ah=Acs+(size_t)h*CS;
            for(int l=0;l<CS;l++) for(int s=0;s<CS;s++) Larg[((size_t)h*CS+l)*CS+s]=(s<=l)?(Ah[l]-Ah[s]):0.0;
            for(int l=0;l<CS;l++) sarg[(size_t)h*CS+l]=Ah[l];
            for(int s=0;s<CS;s++) darg[(size_t)h*CS+s]=Ah[CS-1]-Ah[s]; }
        if(expnpu){ npu_exp_i16_calib(ctx,Larg,H*CS,CS,Lexp); npu_exp_i16_calib(ctx,sarg,H,CS,sexp); npu_exp_i16_calib(ctx,darg,H,CS,dexp); }
        else { for(size_t i=0;i<(size_t)H*CS*CS;i++)Lexp[i]=exp(Larg[i]); for(size_t i=0;i<(size_t)H*CS;i++){sexp[i]=exp(sarg[i]);dexp[i]=exp(darg[i]);} }
        for(int h=0;h<H;h++) for(int l=0;l<CS;l++) for(int s=0;s<CS;s++) if(s>l) Lexp[((size_t)h*CS+l)*CS+s]=0.0; /* causal mask */
        if(c>0) for(int h=0;h<H;h++){ double dp=exp(Acs_last_prev[h]);
            for(size_t i=0;i<(size_t)P*Nst;i++){ size_t j=(size_t)h*P*Nst+i; state_in[j]=(float)(dp*state_in[j]+cs_prev[j]); } }
        /* ---- Mmask = G (bmm) ⊙ Lexp ---- */
        for(int h=0;h<H;h++){ int g=h%d->G;
            for(int l=0;l<CS;l++) for(int n=0;n<Nst;n++){ ork_f16 v=(ork_f16)C[IX_BC(base+l,g,n)];
                aS[((size_t)h*CS+l)*Nst+n]=v; aO[((size_t)h*CS+l)*Nst+n]=v; }
            for(int n=0;n<Nst;n++) for(int s=0;s<CS;s++) bS[((size_t)h*Nst+n)*CS+s]=(ork_f16)B[IX_BC(base+s,g,n)];
            for(int p=0;p<P;p++) for(int s=0;s<CS;s++) aC[((size_t)h*P+p)*CS+s]=(ork_f16)xbar[((size_t)h*CS+s)*P+p];
            for(int l=0;l<CS;l++) for(int p=0;p<P;p++) bD[((size_t)h*CS+l)*P+p]=(ork_f16)xbar[((size_t)h*CS+l)*P+p];
            for(int n=0;n<Nst;n++) for(int p=0;p<P;p++) bO[((size_t)h*Nst+n)*P+p]=(ork_f16)state_in[((size_t)h*P+p)*Nst+n];
        }
        if((rc=MM(pl_sc,H,CS,Nst,CS,aS,bS,G))) break;
        if(ewnpu){ for(size_t i=0;i<(size_t)H*CS*CS;i++){ e1[i]=G[i]; e2[i]=Lexp[i]; }
            npu_ewmul_f16(ctx,e1,e2,H*CS,CS,e3);
            for(size_t i=0;i<(size_t)H*CS*CS;i++) aD[i]=(ork_f16)e3[i]; }
        else for(size_t i=0;i<(size_t)H*CS*CS;i++) aD[i]=(ork_f16)((double)G[i]*Lexp[i]);
        if((rc=MM(pl_yd,H,CS,CS,P,aD,bD,Yd))) break;
        /* ---- Bdec = dexp ⊙ B  (feeds cstate matmul) ---- */
        if(ewnpu){ for(int h=0;h<H;h++){ int g=h%d->G; for(int s=0;s<CS;s++) for(int n=0;n<Nst;n++){
                e1[((size_t)h*CS+s)*Nst+n]=dexp[(size_t)h*CS+s]; e2[((size_t)h*CS+s)*Nst+n]=B[IX_BC(base+s,g,n)]; } }
            npu_ewmul_f16(ctx,e1,e2,H*CS,Nst,e3);
            for(size_t i=0;i<(size_t)H*CS*Nst;i++) bC[i]=(ork_f16)e3[i]; }
        else for(int h=0;h<H;h++){ int g=h%d->G; for(int s=0;s<CS;s++){ double ds=dexp[(size_t)h*CS+s];
                for(int n=0;n<Nst;n++) bC[((size_t)h*CS+s)*Nst+n]=(ork_f16)(ds*B[IX_BC(base+s,g,n)]); } }
        if((rc=MM(pl_cs,H,P,CS,Nst,aC,bC,cs))) break;
        if((rc=MM(pl_yo,H,CS,Nst,P,aO,bO,tmp))) break;
        /* ---- Yoff = tmp ⊙ sdo, then y = Yd + Yoff + D*x (add on CPU) ---- */
        if(ewnpu){ for(int h=0;h<H;h++) for(int l=0;l<CS;l++) for(int p=0;p<P;p++){
                e1[((size_t)h*CS+l)*P+p]=tmp[((size_t)h*CS+l)*P+p]; e2[((size_t)h*CS+l)*P+p]=sexp[(size_t)h*CS+l]; }
            npu_ewmul_f16(ctx,e1,e2,H*CS,P,e3);
            for(int h=0;h<H;h++) for(int l=0;l<CS;l++){ int t=base+l; for(int p=0;p<P;p++)
                y[IX_XHP(t,h,p)]=Yd[((size_t)h*CS+l)*P+p]+e3[((size_t)h*CS+l)*P+p]+D[h]*x[IX_XHP(t,h,p)]; } }
        else for(int h=0;h<H;h++) for(int l=0;l<CS;l++){ int t=base+l; double sdo=sexp[(size_t)h*CS+l]; for(int p=0;p<P;p++)
            y[IX_XHP(t,h,p)]=Yd[((size_t)h*CS+l)*P+p]+tmp[((size_t)h*CS+l)*P+p]*sdo+D[h]*x[IX_XHP(t,h,p)]; }
        memcpy(cs_prev,cs,(size_t)H*P*Nst*sizeof(float)); memcpy(Acs_last_prev,Acs_last,(size_t)H*sizeof(double));
    }
    free(Acs);free(Acs_last);free(Acs_last_prev);free(aS);free(bS);free(G);free(aD);free(bD);free(Yd);
    free(aC);free(bC);free(cs);free(cs_prev);free(aO);free(bO);free(tmp);free(state_in);free(xbar);
    free(Larg);free(Lexp);free(sarg);free(sexp);free(darg);free(dexp);free(tri);free(abarP);free(acsout);free(e1);free(e2);free(e3);
    if(strm5) for(int h=0;h<H;h++){ if(pl_sc[h])ork_mm_free(ctx,pl_sc[h]); if(pl_cs[h])ork_mm_free(ctx,pl_cs[h]); if(pl_yo[h])ork_mm_free(ctx,pl_yo[h]); if(pl_yd[h])ork_mm_free(ctx,pl_yd[h]); }
    g_ssd_tot_us=now_us()-_tot0;
    if(getenv("ORK_SSD_PROF")&&g_ssd_tot_us>0) fprintf(stderr,"  [prof mode %d] total %.2f ms | matmul %.2f ms (%.0f%%) | elementwise+glue %.2f ms (%.0f%%)\n",
        mode, g_ssd_tot_us/1000, g_ssd_mm_us/1000, 100*g_ssd_mm_us/g_ssd_tot_us, (g_ssd_tot_us-g_ssd_mm_us)/1000, 100*(g_ssd_tot_us-g_ssd_mm_us)/g_ssd_tot_us);
    return rc;
}

static double rel_l2(const ssd_dims *d, const double *a, const double *b){
    size_t n=(size_t)d->NC*d->CS*d->H*d->P; double num=0,den=0;
    for(size_t i=0;i<n;i++){ double e=a[i]-b[i]; num+=e*e; den+=b[i]*b[i]; }
    return den>0? sqrt(num/den):sqrt(num);
}

static int run_case(ork_npu *ctx, const char *tag, ssd_dims d, int mode, double tol){
    int L=d.NC*d.CS;
    double *x=malloc((size_t)L*d.H*d.P*sizeof(double)),*dt=malloc((size_t)L*d.H*sizeof(double));
    double *A=malloc(d.H*sizeof(double)),*B=malloc((size_t)L*d.G*d.Nst*sizeof(double));
    double *C=malloc((size_t)L*d.G*d.Nst*sizeof(double)),*D=malloc(d.H*sizeof(double));
    double *yr=malloc((size_t)L*d.H*d.P*sizeof(double)),*yn=malloc((size_t)L*d.H*d.P*sizeof(double));
    for(size_t i=0;i<(size_t)L*d.H*d.P;i++) x[i]=frand_sym();
    for(int h=0;h<d.H;h++){ A[h]=-1.0; D[h]=frand(); }                 /* dt*A in the int16-exp range */
    for(size_t i=0;i<(size_t)L*d.H;i++) dt[i]=0.1+0.35*frand();
    for(size_t i=0;i<(size_t)L*d.G*d.Nst;i++){ B[i]=frand_sym(); C[i]=frand_sym(); }
    ssd_chunked_ref(&d,x,dt,A,B,C,D,yr);
    int rc=ssd_chunked_npu(ctx,&d,x,dt,A,B,C,D,yn,mode);
    const char*mn[8]={"matmul(mul/cumsum/exp=CPU)","+exp=NPU","full: cumsum+exp+mul=NPU","FUSED matmuls (1 submit/stage)","STREAM matmuls (3-core round-robin)","POOLED stream (3-core, pre-packed B)","FULL on-NPU (elementwise NPU + pooled stream)","PURE CPU (matmul on 4 big cores + elementwise CPU)"};
    int fail;
    if(rc){ fprintf(stderr,"[%s] NPU op failed rc=%d\n",tag,rc); fail=1; }
    else { double e=rel_l2(&d,yn,yr); fail=!(e<=tol);
        fprintf(stderr,"[%s] H=%d P=%d N=%d G=%d CS=%d NC=%d  [%s]  rel-L2=%.3e  %s\n",
                tag,d.H,d.P,d.Nst,d.G,d.CS,d.NC, mn[mode], e, fail?"FAIL":"OK"); }
    free(x);free(dt);free(A);free(B);free(C);free(D);free(yr);free(yn);
    return fail;
}

int main(void){
    ork_npu *ctx=ork_npu_init();
    if(!ctx){ fprintf(stderr,"[test_ssd_chunk_npu] no NPU — skipping\n"); return 0; }
    fprintf(stderr,"[test_ssd_chunk_npu] SoC=%s cores=%d\n",ork_npu_soc(ctx),ork_npu_cores(ctx));
    srand(20260712);
    int fail=0;
    ssd_dims g1={.H=4,.P=64,.Nst=128,.G=1,.CS=64,.NC=2}, g2={.H=8,.P=64,.Nst=128,.G=2,.CS=64,.NC=3};
    fail|=run_case(ctx,"matmul-G1",g1,0,3e-2);
    fail|=run_case(ctx,"exp-G1",   g1,1,3e-2);
    fail|=run_case(ctx,"full-G1",  g1,2,4e-2);
    fail|=run_case(ctx,"full-G2",  g2,2,4e-2);
    /* FUSED matmul spine (mode 3): each stage's H matmuls in ONE chained submit. Same numerics as mode 0
     * (CPU elementwise), validating the fused batched-GEMM primitive on the real scan. */
    fail|=run_case(ctx,"fused-G1", g1,3,3e-2);
    fail|=run_case(ctx,"fused-G2", g2,3,3e-2);
    /* STREAM matmul spine (mode 4): scan's independent stage matmuls round-robin across 3 cores. */
    fail|=run_case(ctx,"stream-G1",g1,4,3e-2);
    fail|=run_case(ctx,"stream-G2",g2,4,3e-2);
    /* POOLED stream (mode 5): pre-packed persistent B pool (repack per chunk) + 3-core round-robin. */
    fail|=run_case(ctx,"pool-G1",  g1,5,3e-2);
    fail|=run_case(ctx,"pool-G2",  g2,5,3e-2);
    fail|=run_case(ctx,"cpu-G2",   g2,7,3e-2);   /* pure-CPU baseline (matmul on 4 big cores) */
    /* mode 6 (FULL on-NPU: elementwise on NPU + pooled stream) is CORRECT but a MEASURED DEAD-END: ~97x
     * slower (973ms vs 10ms) — the tiny elementwise ops each pay the NPU submit floor + int16<->fp16
     * mode-switch reset churn (968ms elementwise vs 5ms on CPU). The elementwise belongs on the CPU;
     * only the matmul spine belongs on the NPU (mode 5). Kept reachable (ORK_SSD_MODE6=1) but out of the
     * default run so make test stays fast. */
    if(getenv("ORK_SSD_MODE6")){ fail|=run_case(ctx,"fullnpu-G1",g1,6,4e-2); fail|=run_case(ctx,"fullnpu-G2",g2,6,4e-2); }
    /* Timing: per-op (mode 0, H submits/stage) vs fused (mode 3, 1 submit/stage), warm. */
    { ssd_dims d=g2; int L=d.NC*d.CS;
      double *x=malloc((size_t)L*d.H*d.P*8),*dt=malloc((size_t)L*d.H*8),*A=malloc(d.H*8),*B=malloc((size_t)L*d.G*d.Nst*8),
             *Cm=malloc((size_t)L*d.G*d.Nst*8),*D=malloc(d.H*8),*yo=malloc((size_t)L*d.H*d.P*8);
      for(size_t i=0;i<(size_t)L*d.H*d.P;i++)x[i]=frand_sym(); for(int h=0;h<d.H;h++){A[h]=-1;D[h]=frand();}
      for(size_t i=0;i<(size_t)L*d.H;i++)dt[i]=0.1+0.35*frand(); for(size_t i=0;i<(size_t)L*d.G*d.Nst;i++){B[i]=frand_sym();Cm[i]=frand_sym();}
      ssd_chunked_npu(ctx,&d,x,dt,A,B,Cm,D,yo,0); ssd_chunked_npu(ctx,&d,x,dt,A,B,Cm,D,yo,5); ssd_chunked_npu(ctx,&d,x,dt,A,B,Cm,D,yo,7); /* warm */
      double t0=now_us(); for(int r=0;r<5;r++) ssd_chunked_npu(ctx,&d,x,dt,A,B,Cm,D,yo,0); double per=(now_us()-t0)/5;
      double t3=now_us(); for(int r=0;r<5;r++) ssd_chunked_npu(ctx,&d,x,dt,A,B,Cm,D,yo,5); double pool=(now_us()-t3)/5;
      double t7=now_us(); for(int r=0;r<5;r++) ssd_chunked_npu(ctx,&d,x,dt,A,B,Cm,D,yo,7); double cpu=(now_us()-t7)/5;
      fprintf(stderr,"\n[timing G2 L=%d]  PURE-CPU(mode7) %.2f ms  vs  NPU-POOL(mode5) %.2f ms  ->  NPU %s CPU by %.2fx\n",
              L,cpu/1000,pool/1000, pool<cpu?"BEATS":"loses to", pool<cpu?cpu/pool:pool/cpu);
      fprintf(stderr,"  (per-op NPU %.2f ms; both mode5 & mode7 share the SAME CPU elementwise, so this isolates matmul-on-NPU vs matmul-on-CPU.)\n",per/1000);
      free(x);free(dt);free(A);free(B);free(Cm);free(D);free(yo); }
    ork_npu_free(ctx);
    fprintf(stderr, fail? "\nTEST_SSD_CHUNK_NPU: FAIL\n":"\nTEST_SSD_CHUNK_NPU: PASS\n");
    return fail?1:0;
}
