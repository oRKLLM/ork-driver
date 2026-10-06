/* bwd_ops_fp64_check — reference implementations + fp64 gradient checks for the non-matmul backward ops
 * a training step needs: RMSNorm, SwiGLU, cross-entropy, and the KL distillation term.
 *
 * These four were listed as "write" in the scope report with no evidence behind them. They are all CPU
 * fp32/fp64 work in the real loop (elementwise and reductions — arithmetic intensity under 1, so the NPU
 * would only add DRAM round-trips), which means they can be implemented and VALIDATED with no board at
 * all. That is what this does: each op gets an analytic backward and a central-finite-difference check
 * of the actual scalar loss, entirely in fp64.
 *
 * Why fp64 end to end: the point is to validate the FORMULAE, independent of any precision question. A
 * check that differentiates a rounded forward cannot distinguish a wrong gradient from rounding noise —
 * that mistake cost four detours on the attention path, where a correct dK read as 38x wrong. With both
 * sides in fp64 and eps=1e-6, agreement to ~1e-8 means the maths is right, full stop. Precision effects
 * are a separate, later question and belong in the op that actually runs in fp16.
 *
 * Each check reports the raw analytic and finite-difference values alongside the relative error, and
 * does NOT floor the denominator — flooring is what inflated a correct gradient into a 38x failure.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned s_=90210;
static double rnd(void){ s_=s_*1103515245u+12345u; return (double)((int)((s_>>13)&0x7fff)-16384)/16384.0; }
static double sig(double x){ return 1.0/(1.0+exp(-x)); }

static int fails = 0;
/* Central difference of loss() wrt p[idx], compared to analytic g[idx].
 *
 * THE TOLERANCE KNOWS ITS OWN NOISE FLOOR, and that is the whole point of this function. A central
 * difference of a loss of magnitude |L| computed in fp64 carries absolute noise of roughly
 * |L|·ε_machine/eps — it cannot resolve a derivative smaller than that, no matter how correct the
 * analytic gradient is. Judging such an element on RELATIVE error reports a perfect gradient as broken:
 * the first version of this check did exactly that, flagging cross-entropy and KL at 1.1e-05 / 3.8e-05
 * relative while analytic and finite difference agreed to six significant figures. Their gradients are
 * (p−y)/B, so non-target entries are legitimately ~1e-6 and sit under the floor.
 *
 * The same mistake, with the same shape, made a correct attention dK read as 38x wrong earlier in this
 * work. So: pass when |an − fd| is within the finite-difference noise floor OR within a relative
 * tolerance — never on relative error alone. Both are printed so a reader can see which one carried. */
static void cmp(const char *op,const char *wrt,double *p,const double *g,int n,double (*loss)(void),double eps){
    double L0 = loss();
    double atol = 16.0 * fabs(L0) * 2.220446049250313e-16 / eps;   /* fd roundoff floor */
    if(atol < 1e-14) atol = 1e-14;
    double worst=0, wabs=0; int bad=0;
    for(int s=0;s<6;s++){
        int i=(s*n)/6+s; if(i>=n) i=n-1;
        double sv=p[i];
        p[i]=sv+eps; double Lp=loss();
        p[i]=sv-eps; double Lm=loss();
        p[i]=sv;     loss();
        double fd=(Lp-Lm)/(2*eps), an=g[i], ad=fabs(an-fd);
        double rel = fabs(fd)>0 ? ad/fabs(fd) : (fabs(an)>0?1e9:0);
        if(!(ad<=atol || rel<=1e-6)) bad++;
        if(rel>worst){ worst=rel; wabs=ad; }
    }
    if(bad) fails++;
    printf("  %-12s d/d%-8s rel %.2e  |Δ| %.2e  (floor %.1e)  %s\n", op, wrt, worst, wabs, atol,
           bad ? "FAIL" : (worst>1e-6 ? "PASS (within fd noise floor)" : "PASS"));
}

/* ───────────────────────── RMSNorm ─────────────────────────
 * y_i = g_i · x_i / r,   r = sqrt(mean(x²)+eps),  L = ½‖y‖², dy = y
 *   dg_k = dy_k · x_k / r
 *   dx_k = g_k·dy_k/r − (x_k /(N·r³))·Σ_i dy_i·g_i·x_i        (the r-coupling term is the easy one to miss)
 */
static int RN; static double *rx,*rg,*ry,*rdx,*rdg; static double reps=1e-6;
static double rms_loss(void){
    double ms=0; for(int i=0;i<RN;i++) ms+=rx[i]*rx[i]; ms/=RN;
    double r=sqrt(ms+reps), L=0;
    for(int i=0;i<RN;i++){ ry[i]=rg[i]*rx[i]/r; L+=ry[i]*ry[i]; }
    return 0.5*L;
}
static void rms_bwd(void){
    double ms=0; for(int i=0;i<RN;i++) ms+=rx[i]*rx[i]; ms/=RN;
    double r=sqrt(ms+reps);
    for(int i=0;i<RN;i++) ry[i]=rg[i]*rx[i]/r;              /* dy = y */
    double dot=0; for(int i=0;i<RN;i++) dot += ry[i]*rg[i]*rx[i];
    for(int k=0;k<RN;k++){
        rdg[k] = ry[k]*rx[k]/r;
        rdx[k] = rg[k]*ry[k]/r - rx[k]*dot/((double)RN*r*r*r);
    }
}

/* ───────────────────────── SwiGLU ─────────────────────────
 * y = silu(a) ⊙ b,  silu(x)=x·σ(x),  L = ½‖y‖², dy = y
 *   db_k = silu(a_k)·dy_k
 *   da_k = b_k·dy_k·silu'(a_k),   silu'(x) = σ(x)·(1 + x·(1−σ(x)))
 */
static int SN; static double *sa,*sb,*sy,*sda,*sdb;
static double sw_loss(void){
    double L=0; for(int i=0;i<SN;i++){ sy[i]=sa[i]*sig(sa[i])*sb[i]; L+=sy[i]*sy[i]; } return 0.5*L;
}
static void sw_bwd(void){
    for(int i=0;i<SN;i++) sy[i]=sa[i]*sig(sa[i])*sb[i];     /* dy = y */
    for(int k=0;k<SN;k++){
        double s=sig(sa[k]), silu=sa[k]*s, dsilu=s*(1.0+sa[k]*(1.0-s));
        sdb[k]=silu*sy[k];
        sda[k]=sb[k]*sy[k]*dsilu;
    }
}

/* ──────────────────── cross-entropy ────────────────────
 * p = softmax(z) per row, L = −(1/B)·Σ_b log p[b,t_b]
 *   dL/dz[b,v] = (p[b,v] − [v==t_b]) / B
 */
static int CB,CV; static double *cz,*cdz; static int *ct;
static double ce_loss(void){
    double L=0;
    for(int b=0;b<CB;b++){
        double mx=-1e300; for(int v=0;v<CV;v++) if(cz[b*CV+v]>mx) mx=cz[b*CV+v];
        double sum=0; for(int v=0;v<CV;v++) sum+=exp(cz[b*CV+v]-mx);
        L += -( cz[b*CV+ct[b]] - mx - log(sum) );
    }
    return L/CB;
}
static void ce_bwd(void){
    for(int b=0;b<CB;b++){
        double mx=-1e300; for(int v=0;v<CV;v++) if(cz[b*CV+v]>mx) mx=cz[b*CV+v];
        double sum=0; for(int v=0;v<CV;v++) sum+=exp(cz[b*CV+v]-mx);
        for(int v=0;v<CV;v++){
            double p=exp(cz[b*CV+v]-mx)/sum;
            cdz[b*CV+v] = (p - (v==ct[b]?1.0:0.0))/CB;
        }
    }
}

/* ──────────────────────── KL ────────────────────────
 * q = softmax(zref) frozen, p = softmax(z), L = (1/B)·Σ KL(q‖p)
 *   dL/dz[b,v] = (p[b,v] − q[b,v]) / B
 */
static double *kz,*kzr,*kdz;
static double kl_loss(void){
    double L=0;
    for(int b=0;b<CB;b++){
        double mp=-1e300,mq=-1e300;
        for(int v=0;v<CV;v++){ if(kz[b*CV+v]>mp)mp=kz[b*CV+v]; if(kzr[b*CV+v]>mq)mq=kzr[b*CV+v]; }
        double sp=0,sq=0;
        for(int v=0;v<CV;v++){ sp+=exp(kz[b*CV+v]-mp); sq+=exp(kzr[b*CV+v]-mq); }
        for(int v=0;v<CV;v++){
            double q=exp(kzr[b*CV+v]-mq)/sq;
            double lp=kz[b*CV+v]-mp-log(sp), lq=kzr[b*CV+v]-mq-log(sq);
            L += q*(lq-lp);
        }
    }
    return L/CB;
}
static void kl_bwd(void){
    for(int b=0;b<CB;b++){
        double mp=-1e300,mq=-1e300;
        for(int v=0;v<CV;v++){ if(kz[b*CV+v]>mp)mp=kz[b*CV+v]; if(kzr[b*CV+v]>mq)mq=kzr[b*CV+v]; }
        double sp=0,sq=0;
        for(int v=0;v<CV;v++){ sp+=exp(kz[b*CV+v]-mp); sq+=exp(kzr[b*CV+v]-mq); }
        for(int v=0;v<CV;v++){
            double p=exp(kz[b*CV+v]-mp)/sp, q=exp(kzr[b*CV+v]-mq)/sq;
            kdz[b*CV+v] = (p-q)/CB;
        }
    }
}

/* POSITIVE CONTROLS — this harness's own rule (a numerics check with no configuration that MUST fail is
 * vacuous) applied to itself. Each is a plausible real mistake:
 *   rms_bwd_broken : drops the r-coupling term in dx — the single easiest RMSNorm error, and it still
 *                    gives a descent direction, so only a gradient check catches it.
 *   sw_bwd_broken  : uses silu(a) where silu'(a) belongs in da — a wrong-derivative slip.
 * If either PASSES the check, the check cannot see a wrong gradient and every other row is meaningless. */
static void rms_bwd_broken(void){
    double ms=0; for(int i=0;i<RN;i++) ms+=rx[i]*rx[i]; ms/=RN;
    double r=sqrt(ms+reps);
    for(int i=0;i<RN;i++) ry[i]=rg[i]*rx[i]/r;
    for(int k=0;k<RN;k++){ rdg[k]=ry[k]*rx[k]/r; rdx[k]=rg[k]*ry[k]/r; }   /* r-coupling term MISSING */
}
static void sw_bwd_broken(void){
    for(int i=0;i<SN;i++) sy[i]=sa[i]*sig(sa[i])*sb[i];
    for(int k=0;k<SN;k++){ double s=sig(sa[k]), silu=sa[k]*s;
        sdb[k]=silu*sy[k]; sda[k]=sb[k]*sy[k]*silu; }                      /* silu where silu' belongs */
}

int main(void){
    const double eps=1e-6;
    printf("fp64 gradient checks for the non-matmul backward ops   (CPU only, no board)\n");
    printf("eps=%g, relative error UNFLOORED, threshold 1e-6\n\n", eps);

    RN=256; rx=malloc(RN*8); rg=malloc(RN*8); ry=malloc(RN*8); rdx=malloc(RN*8); rdg=malloc(RN*8);
    for(int i=0;i<RN;i++){ rx[i]=rnd()*2.0; rg[i]=1.0+rnd()*0.3; }
    rms_bwd(); cmp("RMSNorm","x",rx,rdx,RN,rms_loss,eps); cmp("RMSNorm","g",rg,rdg,RN,rms_loss,eps);

    SN=256; sa=malloc(SN*8); sb=malloc(SN*8); sy=malloc(SN*8); sda=malloc(SN*8); sdb=malloc(SN*8);
    for(int i=0;i<SN;i++){ sa[i]=rnd()*3.0; sb[i]=rnd()*2.0; }
    sw_bwd(); cmp("SwiGLU","a",sa,sda,SN,sw_loss,eps); cmp("SwiGLU","b",sb,sdb,SN,sw_loss,eps);

    CB=8; CV=64; cz=malloc(CB*CV*8); cdz=malloc(CB*CV*8); ct=malloc(CB*sizeof(int));
    for(int i=0;i<CB*CV;i++) cz[i]=rnd()*3.0;
    for(int b=0;b<CB;b++) ct[b]=((unsigned)(rnd()*1000+1000))%CV;
    ce_bwd(); cmp("CrossEntropy","z",cz,cdz,CB*CV,ce_loss,eps);

    kz=malloc(CB*CV*8); kzr=malloc(CB*CV*8); kdz=malloc(CB*CV*8);
    for(int i=0;i<CB*CV;i++){ kz[i]=rnd()*3.0; kzr[i]=rnd()*3.0; }
    kl_bwd(); cmp("KL(q||p)","z",kz,kdz,CB*CV,kl_loss,eps);

    /* ---- positive controls: these MUST fail, or the checker is blind ---- */
    printf("\npositive controls (each MUST fail — otherwise this harness proves nothing):\n");
    int before=fails;
    rms_bwd_broken(); cmp("RMSNorm-BAD","x",rx,rdx,RN,rms_loss,eps);
    sw_bwd_broken();  cmp("SwiGLU-BAD","a",sa,sda,SN,sw_loss,eps);
    int controls_fired = fails-before;

    if(controls_fired<2){
        printf("\nVOID — %d of 2 positive controls did NOT fail. The check cannot detect a wrong\n"
               "       gradient, so the PASSes above carry no information.\n", 2-controls_fired);
        return 3;
    }
    printf("  (both controls failed, as required — the checker can see a wrong gradient)\n");
    printf("\n%s\n", before==0
        ? "ALL PASS — RMSNorm, SwiGLU, cross-entropy and KL backward formulae are correct."
        : "SOME REAL CHECKS FAILED — see above.");
    return before!=0;
}
