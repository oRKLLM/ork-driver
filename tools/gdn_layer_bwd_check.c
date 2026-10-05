/* gdn_layer_bwd_check — the GDN LAYER around the recurrence: fp64 gradient checks.
 *
 * gdn_bwd_fp64_ref.c and gdn_chunk_bwd.c prove the state recurrence. The recurrence is not the
 * layer. Qwen3.5's linear-attention block wraps it in four more pieces, each of which needs its own
 * backward, and none of which the recurrence checks touch. Transcribed from the real forward
 * (mlx_lm/models/qwen3_5.py Qwen3_5GatedDeltaNet.__call__, gated_delta.compute_g, and
 * qwen3_next.Qwen3NextRMSNormGated), not from memory:
 *
 *   1. CONV      conv_out = silu(depthwise_causal_conv1d(concat(conv_state, qkv), w, bias))
 *                kernel 4, one filter per channel. q, k, v are slices of conv_out.
 *   2. QKNORM    s = Dk^-0.5;  q = s^2 * rms_norm(q, no weight, 1e-6);  k = s * rms_norm(k, ...)
 *                NOTE: rms_norm, not l2_norm -- they differ by a factor sqrt(D), which is absorbed
 *                into the scale here but would be a silent 1/sqrt(D) error in a gradient check.
 *   3. DECAY     beta = sigmoid(b);  g = exp(-exp(A_log) * softplus(a + dt_bias))
 *                A_log and dt_bias are per-head PARAMETERS and get gradients too.
 *   4. OUTGATE   out = silu(z) * rms_norm(h, weight, eps)        (the gated RMSNorm)
 *
 * All four are elementwise or near-elementwise and small next to the recurrence, which is exactly
 * why they are easy to leave unchecked. Each is verified here against central finite differences in
 * fp64, and each carries a positive control -- a backward with one term removed that MUST fail.
 *
 * What this does NOT establish: numerics at real width. These run at D=6..8 in fp64 to test the
 * FORMULAE. d=128, T=1024 in fp16 is a separate question, and answering it needs fp32 fixtures from
 * the MLX implementation rather than more of this.
 *
 * CPU-only, no NPU, no board.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned s_=20261005;
static double rnd(void){ s_=s_*1103515245u+12345u; return (double)((int)((s_>>13)&0x7fff)-16384)/16384.0; }

static double sigm(double x){ return 1.0/(1.0+exp(-x)); }
static double silu(double x){ return x*sigm(x); }
static double dsilu(double x){ double s=sigm(x); return s*(1.0+x*(1.0-s)); }
static double softplus(double x){ return x>20.0? x : log1p(exp(x)); }

static int BRK=0;                    /* which term to omit; 0 = none. Set per op. */
static int fails=0, ctl_caught=0, ctl_total=0;

/* ---- generic central-difference check against a scalar loss ---------------------------------- */
static double (*LOSS)(void);
static int chk(const char*nm,double*p,const double*g,int n,double eps,int quiet){
    double L0=LOSS();
    double atol=16.0*fabs(L0)*2.220446049250313e-16/eps; if(atol<1e-14) atol=1e-14;
    double worst=0,wd=0; int bad=0, ns=n<8?n:8;
    for(int s=0;s<ns;s++){
        int i=(s*n)/ns; if(i>=n) i=n-1;
        double sv=p[i];
        p[i]=sv+eps; double Lp=LOSS();
        p[i]=sv-eps; double Lm=LOSS();
        p[i]=sv;
        double fd=(Lp-Lm)/(2*eps), an=g[i], ad=fabs(an-fd);
        double rel=fabs(fd)>0?ad/fabs(fd):(fabs(an)>0?1e9:0);
        if(!(ad<=atol || rel<=1e-6)) bad++;
        if(rel>worst){ worst=rel; wd=ad; }
    }
    if(!quiet)
        printf("  %-18s rel %.2e  |d| %.2e  (floor %.1e)  %s\n", nm, worst, wd, atol,
               bad?"FAIL":(worst>1e-6?"PASS (within fd floor)":"PASS"));
    return bad!=0;
}

/* ============================ 1. depthwise causal conv1d + silu =============================== */
/* conv_input = [state (K-1 rows); x (T rows)], channels C.  pre[t,c] = sum_r w[c,r] in[t+r,c] + bs[c]
 * y[t,c] = silu(pre[t,c]).   Depthwise: one length-K filter per channel, no cross-channel mixing. */
#define CT 6
#define CC 5
#define CK 4
static double cx[CT*CC], cst[(CK-1)*CC], cw[CC*CK], cb[CC], cdo_[CT*CC];
static double g_cx[CT*CC], g_cst[(CK-1)*CC], g_cw[CC*CK], g_cb[CC];

static void conv_fwd(double*y){
    for(int t=0;t<CT;t++) for(int c=0;c<CC;c++){
        double p=cb[c];
        for(int r=0;r<CK;r++){
            int u=t+r;                                        /* index into the concatenated input */
            double in = u<CK-1 ? cst[u*CC+c] : cx[(u-(CK-1))*CC+c];
            p += cw[c*CK+r]*in;
        }
        y[t*CC+c]=silu(p);
    }
}
static double conv_loss(void){
    double y[CT*CC]; conv_fwd(y);
    double L=0; for(int i=0;i<CT*CC;i++) L+=cdo_[i]*y[i];
    return L;
}
static void conv_bwd(void){
    memset(g_cx,0,sizeof g_cx); memset(g_cst,0,sizeof g_cst);
    memset(g_cw,0,sizeof g_cw); memset(g_cb,0,sizeof g_cb);
    for(int t=0;t<CT;t++) for(int c=0;c<CC;c++){
        double p=cb[c];
        for(int r=0;r<CK;r++){
            int u=t+r;
            double in = u<CK-1 ? cst[u*CC+c] : cx[(u-(CK-1))*CC+c];
            p += cw[c*CK+r]*in;
        }
        /* BRK 1: forget that the conv output passes through silu. */
        double dp = cdo_[t*CC+c] * (BRK==1 ? 1.0 : dsilu(p));
        g_cb[c]+=dp;
        for(int r=0;r<CK;r++){
            int u=t+r;
            double in = u<CK-1 ? cst[u*CC+c] : cx[(u-(CK-1))*CC+c];
            g_cw[c*CK+r] += dp*in;
            if(u<CK-1) g_cst[u*CC+c]           += dp*cw[c*CK+r];
            else       g_cx[(u-(CK-1))*CC+c]   += dp*cw[c*CK+r];
        }
    }
}

/* ============================ 2. scaled RMSNorm on q and k ==================================== */
/* y = scale * x / r,  r = sqrt(mean(x^2) + eps), over the last axis of length D, per (t,h). */
#define NT 4
#define NH 2
#define ND 6
static double nx[NT*NH*ND], ndo_[NT*NH*ND], g_nx[NT*NH*ND];
static double nscale, neps=1e-6;

static void qkn_fwd(double*y){
    for(int b=0;b<NT*NH;b++){
        const double*x=nx+(size_t)b*ND; double*o=y+(size_t)b*ND;
        double ss=0; for(int j=0;j<ND;j++) ss+=x[j]*x[j];
        double r=sqrt(ss/ND+neps);
        for(int j=0;j<ND;j++) o[j]=nscale*x[j]/r;
    }
}
static double qkn_loss(void){
    double y[NT*NH*ND]; qkn_fwd(y);
    double L=0; for(int i=0;i<NT*NH*ND;i++) L+=ndo_[i]*y[i];
    return L;
}
static void qkn_bwd(void){
    for(int b=0;b<NT*NH;b++){
        const double*x=nx+(size_t)b*ND, *dy=ndo_+(size_t)b*ND; double*dx=g_nx+(size_t)b*ND;
        double ss=0,dot=0;
        for(int j=0;j<ND;j++){ ss+=x[j]*x[j]; dot+=dy[j]*x[j]; }
        double r=sqrt(ss/ND+neps);
        /* BRK 2: drop the coupling through r -- the classic RMSNorm backward slip. */
        for(int j=0;j<ND;j++)
            dx[j]=nscale*( dy[j]/r - (BRK==2?0.0: x[j]*dot/((double)ND*r*r*r)) );
    }
}

/* ============================ 3. decay parameterisation ======================================= */
/* beta = sigmoid(bb);  g = exp(-exp(A_log) * softplus(a + dt_bias)).  A_log, dt_bias per head. */
#define DT 5
#define DH 3
static double da_[DT*DH], dbb[DT*DH], Alog[DH], dtb[DH], ddg[DT*DH], ddb[DT*DH];
static double g_da[DT*DH], g_dbb[DT*DH], g_Alog[DH], g_dtb[DH];

static double dec_loss(void){
    double L=0;
    for(int t=0;t<DT;t++) for(int h=0;h<DH;h++){
        double sp=softplus(da_[t*DH+h]+dtb[h]);
        double gg=exp(-exp(Alog[h])*sp);
        L += ddg[t*DH+h]*gg + ddb[t*DH+h]*sigm(dbb[t*DH+h]);
    }
    return L;
}
static void dec_bwd(void){
    memset(g_Alog,0,sizeof g_Alog); memset(g_dtb,0,sizeof g_dtb);
    for(int t=0;t<DT;t++) for(int h=0;h<DH;h++){
        double u=da_[t*DH+h]+dtb[h], sp=softplus(u), ea=exp(Alog[h]);
        double gg=exp(-ea*sp), dg=ddg[t*DH+h];
        /* BRK 3: forget the chain through g itself when differentiating the exponent. */
        double core = (BRK==3?1.0:gg);
        g_Alog[h]      += dg*core*(-ea*sp);          /* d/dA_log: exp(A_log) is itself exp'd */
        double dsp      = dg*core*(-ea);
        g_da[t*DH+h]    = dsp*sigm(u);               /* d softplus = sigmoid */
        g_dtb[h]       += dsp*sigm(u);
        double be=sigm(dbb[t*DH+h]);
        g_dbb[t*DH+h]   = ddb[t*DH+h]*be*(1.0-be);
    }
}

/* ============================ 4. gated RMSNorm on the output ================================== */
/* out = silu(z) * (weight * h / r),  r = sqrt(mean(h^2) + eps), over D. */
#define GT 4
#define GH 2
#define GD 6
static double gh[GT*GH*GD], gz[GT*GH*GD], gw[GD], gdo_[GT*GH*GD];
static double g_gh[GT*GH*GD], g_gz[GT*GH*GD], g_gw[GD];
static double geps=1e-6;

static double og_loss(void){
    double L=0;
    for(int b=0;b<GT*GH;b++){
        const double*h=gh+(size_t)b*GD, *z=gz+(size_t)b*GD, *dO=gdo_+(size_t)b*GD;
        double ss=0; for(int j=0;j<GD;j++) ss+=h[j]*h[j];
        double r=sqrt(ss/GD+geps);
        for(int j=0;j<GD;j++) L += dO[j]*silu(z[j])*gw[j]*h[j]/r;
    }
    return L;
}
static void og_bwd(void){
    memset(g_gw,0,sizeof g_gw);
    for(int b=0;b<GT*GH;b++){
        const double*h=gh+(size_t)b*GD, *z=gz+(size_t)b*GD, *dO=gdo_+(size_t)b*GD;
        double *dh=g_gh+(size_t)b*GD, *dz=g_gz+(size_t)b*GD;
        double ss=0; for(int j=0;j<GD;j++) ss+=h[j]*h[j];
        double r=sqrt(ss/GD+geps);
        double dot=0;
        for(int j=0;j<GD;j++){
            double xn=gw[j]*h[j]/r;                  /* the normalised, weighted value */
            dz[j]=dO[j]*xn*dsilu(z[j]);
            double dxn=dO[j]*silu(z[j]);             /* cotangent of the rms_norm output */
            g_gw[j]+=dxn*h[j]/r;
            dot+=dxn*gw[j]*h[j];
        }
        /* BRK 4: drop the r-coupling, as in BRK 2 but with the weight in the way. */
        for(int j=0;j<GD;j++){
            double dxn=dO[j]*silu(z[j]);
            dh[j]=dxn*gw[j]/r - (BRK==4?0.0: h[j]*dot/((double)GD*r*r*r));
        }
    }
}

/* ============================================================================================== */
static void control(const char*nm,int brk,double*p,const double*g,int n,double eps,void(*bwd)(void)){
    ctl_total++;
    BRK=brk; bwd();
    int bad=chk(nm,p,g,n,eps,1);
    BRK=0;   bwd();
    printf("  %-34s %s\n", nm, bad?"FAIL (as required)":"PASSED -- the check is vacuous");
    if(bad) ctl_caught++;
}

int main(void){
    double eps=1e-6;
    printf("GDN LAYER backward -- the four pieces around the recurrence   (fp64, CPU only)\n\n");

    /* 1 */
    for(int i=0;i<CT*CC;i++){ cx[i]=rnd(); cdo_[i]=rnd(); }
    for(int i=0;i<(CK-1)*CC;i++) cst[i]=rnd();
    for(int i=0;i<CC*CK;i++) cw[i]=rnd();
    for(int i=0;i<CC;i++) cb[i]=0.3*rnd();
    LOSS=conv_loss; conv_bwd();
    printf("1. depthwise causal conv1d + silu   (T=%d C=%d K=%d)\n",CT,CC,CK);
    fails += chk("d/dx",      cx,  g_cx,  CT*CC,     eps,0);
    fails += chk("d/dstate",  cst, g_cst, (CK-1)*CC, eps,0);
    fails += chk("d/dweight", cw,  g_cw,  CC*CK,     eps,0);
    fails += chk("d/dbias",   cb,  g_cb,  CC,        eps,0);

    /* 2 */
    nscale=pow((double)ND,-0.5);
    for(int i=0;i<NT*NH*ND;i++){ nx[i]=rnd(); ndo_[i]=rnd(); }
    LOSS=qkn_loss; qkn_bwd();
    printf("\n2. scaled RMSNorm on q/k           (T=%d H=%d D=%d, scale=D^-0.5)\n",NT,NH,ND);
    fails += chk("d/dx", nx, g_nx, NT*NH*ND, eps,0);

    /* 3 */
    for(int i=0;i<DT*DH;i++){ da_[i]=rnd(); dbb[i]=rnd(); ddg[i]=rnd(); ddb[i]=rnd(); }
    for(int h=0;h<DH;h++){ Alog[h]=0.5*rnd(); dtb[h]=0.5*rnd(); }
    LOSS=dec_loss; dec_bwd();
    printf("\n3. decay parameterisation          g=exp(-exp(A_log)softplus(a+dt)), beta=sigmoid(b)\n");
    fails += chk("d/da",       da_,  g_da,  DT*DH, eps,0);
    fails += chk("d/db",       dbb,  g_dbb, DT*DH, eps,0);
    fails += chk("d/dA_log",   Alog, g_Alog, DH,   eps,0);
    fails += chk("d/ddt_bias", dtb,  g_dtb,  DH,   eps,0);

    /* 4 */
    for(int i=0;i<GT*GH*GD;i++){ gh[i]=rnd(); gz[i]=rnd(); gdo_[i]=rnd(); }
    for(int j=0;j<GD;j++) gw[j]=1.0+0.3*rnd();
    LOSS=og_loss; og_bwd();
    printf("\n4. gated RMSNorm on the output     out = silu(z) * rms_norm(h, weight)\n");
    fails += chk("d/dh",      gh, g_gh, GT*GH*GD, eps,0);
    fails += chk("d/dz",      gz, g_gz, GT*GH*GD, eps,0);
    fails += chk("d/dweight", gw, g_gw, GD,       eps,0);

    /* controls */
    printf("\npositive controls (each MUST fail):\n");
    LOSS=conv_loss; control("conv: silu' dropped",       1, cx, g_cx, CT*CC,    eps, conv_bwd);
    LOSS=qkn_loss;  control("qk-norm: r-coupling dropped",2, nx, g_nx, NT*NH*ND,eps, qkn_bwd);
    LOSS=dec_loss;  control("decay: chain through g lost",3, da_,g_da, DT*DH,   eps, dec_bwd);
    LOSS=og_loss;   control("out-gate: r-coupling dropped",4,gh, g_gh, GT*GH*GD,eps, og_bwd);
    if(ctl_caught<ctl_total){
        printf("  %d of %d controls did not fail; the checks prove nothing.\n",
               ctl_total-ctl_caught, ctl_total);
        fails++;
    } else printf("  all %d controls failed, as required.\n", ctl_total);

    printf("\n%s\n", fails==0
      ? "ALL PASS -- conv1d, q/k RMSNorm, the decay parameterisation and the output gate are correct."
      : "SOME FAILED -- see above.");
    return fails!=0;
}
