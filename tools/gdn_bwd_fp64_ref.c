/* gdn_bwd_fp64_ref — Gated DeltaNet backward: the last unvalidated piece of mathematics.
 *
 * Everything else a training step needs is now proven. This one is different in kind: the GDN state
 * update is a SEQUENTIAL RECURRENCE, not a matmul, so there is no NPU path to validate — the on-NPU
 * scan is measured to LOSE (1.9x @0.8B, 4.8x @9B, gap widening, because the state fits in cache and the
 * CPU recurrence has no submit overhead). It belongs on the CPU. What has to be proven is the gradient
 * maths, and that is what this does, in fp64 against central finite differences.
 *
 * FORWARD (per timestep t; state S ∈ R^{dv×dk}, S_0 = 0):
 *     Ŝ_t = α_t · S_{t−1}                        decay gate
 *     e_t = v_t − Ŝ_t k_t                        delta: how wrong the current state is about v_t
 *     S_t = Ŝ_t + β_t · e_t k_tᵀ                 write, strength β_t
 *     o_t = S_t q_t                              read
 *     L   = ½ Σ_t ‖o_t‖²   ⇒   do_t = o_t
 *
 * BACKWARD, reverse time, carrying dS from the future. Each line is one forward line differentiated:
 *     from o_t = S_t q_t      dq_t  = S_tᵀ do_t                    dS_t += do_t q_tᵀ
 *     from S_t = Ŝ + β e kᵀ   dβ_t  = ⟨dS_t , e_t k_tᵀ⟩            de_t  = β_t · dS_t k_t
 *                             dk_t += β_t · dS_tᵀ e_t              dŜ   += dS_t
 *     from e_t = v − Ŝ k      dv_t  = de_t                         dŜ   += −de_t k_tᵀ
 *                             dk_t += −Ŝ_tᵀ de_t
 *     from Ŝ_t = α S_{t−1}    dα_t  = ⟨dŜ , S_{t−1}⟩               dS_{t−1} += α_t · dŜ
 *
 * The two easy errors are both covered by the checks: k_t appears TWICE (once through the write, once
 * through the delta), so dk needs both terms; and dS must be accumulated from the future BEFORE dq is
 * taken, not after.
 *
 * This is the recurrent form. The chunked form used in production — the one carrying a UT-solve at ~38%
 * of forward scan time — is an algebraic rearrangement of exactly this, so this file is the oracle the
 * chunked backward must match. Validating the chunked version against a reference is far cheaper than
 * deriving it twice independently.
 *
 * CPU-only, no NPU, no board.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int T,DK,DV;
static double *q,*k,*v,*al,*be;                 /* inputs:  q,k [T,DK]  v [T,DV]  al,be [T] */
static double *dq,*dk,*dv,*dal,*dbe;            /* grads, same shapes */
static double *Sh,*Sall,*o;                     /* Ŝ per t, S_t per t (both [T,DV,DK]), o [T,DV] */
static unsigned s_=271828;
static double rnd(void){ s_=s_*1103515245u+12345u; return (double)((int)((s_>>13)&0x7fff)-16384)/16384.0; }

/* forward; stores Ŝ_t and S_t for every t (the backward needs both) and returns L */
static double fwd(void){
    size_t sz=(size_t)DV*DK;
    double *S=calloc(sz,8), *e=malloc(DV*8);
    double L=0;
    for(int t=0;t<T;t++){
        double a=al[t], b=be[t];
        double *Sh_t=Sh+(size_t)t*sz, *S_t=Sall+(size_t)t*sz;
        for(size_t i=0;i<sz;i++) Sh_t[i]=a*S[i];                      /* Ŝ = α S_{t−1} */
        for(int i=0;i<DV;i++){ double acc=0;
            for(int j=0;j<DK;j++) acc+=Sh_t[(size_t)i*DK+j]*k[(size_t)t*DK+j];
            e[i]=v[(size_t)t*DV+i]-acc; }                             /* e = v − Ŝ k */
        for(int i=0;i<DV;i++) for(int j=0;j<DK;j++)
            S_t[(size_t)i*DK+j]=Sh_t[(size_t)i*DK+j]+b*e[i]*k[(size_t)t*DK+j];   /* S = Ŝ + β e kᵀ */
        for(int i=0;i<DV;i++){ double acc=0;
            for(int j=0;j<DK;j++) acc+=S_t[(size_t)i*DK+j]*q[(size_t)t*DK+j];
            o[(size_t)t*DV+i]=acc; L+=acc*acc; }                      /* o = S q */
        memcpy(S,S_t,sz*8);
    }
    free(S); free(e);
    return 0.5*L;
}

/* When set, bwd() omits the delta-path dk term — the "k_t appears twice" trap the header names.
 * The checks below must FAIL on it; if they pass, the checks prove nothing. */
static int broken=0;

static void bwd(void){
    size_t sz=(size_t)DV*DK;
    double *dS=calloc(sz,8), *dSh=malloc(sz*8), *e=malloc(DV*8), *de=malloc(DV*8);
    memset(dq,0,(size_t)T*DK*8); memset(dk,0,(size_t)T*DK*8);
    memset(dv,0,(size_t)T*DV*8); memset(dal,0,T*8); memset(dbe,0,T*8);
    for(int t=T-1;t>=0;t--){
        double a=al[t], b=be[t];
        double *Sh_t=Sh+(size_t)t*sz, *S_t=Sall+(size_t)t*sz;
        const double *Sprev = t? Sall+(size_t)(t-1)*sz : NULL;        /* S_{t−1}; zero at t=0 */

        for(int i=0;i<DV;i++){ double acc=0;                          /* recompute e_t */
            for(int j=0;j<DK;j++) acc+=Sh_t[(size_t)i*DK+j]*k[(size_t)t*DK+j];
            e[i]=v[(size_t)t*DV+i]-acc; }

        /* o_t = S_t q_t  — dS_t gets the future's dS plus this step's contribution */
        for(int j=0;j<DK;j++){ double acc=0;
            for(int i=0;i<DV;i++) acc+=S_t[(size_t)i*DK+j]*o[(size_t)t*DV+i];
            dq[(size_t)t*DK+j]=acc; }                                 /* dq = S_tᵀ do,  do = o */
        for(int i=0;i<DV;i++) for(int j=0;j<DK;j++)
            dS[(size_t)i*DK+j] += o[(size_t)t*DV+i]*q[(size_t)t*DK+j];/* dS += do qᵀ */

        /* S_t = Ŝ + β e kᵀ */
        double dbeta=0;
        for(int i=0;i<DV;i++) for(int j=0;j<DK;j++)
            dbeta += dS[(size_t)i*DK+j]*e[i]*k[(size_t)t*DK+j];
        dbe[t]=dbeta;
        for(int i=0;i<DV;i++){ double acc=0;
            for(int j=0;j<DK;j++) acc+=dS[(size_t)i*DK+j]*k[(size_t)t*DK+j];
            de[i]=b*acc; }                                            /* de = β dS k */
        for(int j=0;j<DK;j++){ double acc=0;
            for(int i=0;i<DV;i++) acc+=dS[(size_t)i*DK+j]*e[i];
            dk[(size_t)t*DK+j] += b*acc; }                            /* dk += β dSᵀ e  (write path) */
        memcpy(dSh,dS,sz*8);                                          /* dŜ += dS */

        /* e_t = v_t − Ŝ k_t */
        for(int i=0;i<DV;i++) dv[(size_t)t*DV+i]=de[i];
        for(int i=0;i<DV;i++) for(int j=0;j<DK;j++)
            dSh[(size_t)i*DK+j] -= de[i]*k[(size_t)t*DK+j];           /* dŜ += −de kᵀ */
        if(!broken) for(int j=0;j<DK;j++){ double acc=0;
            for(int i=0;i<DV;i++) acc+=Sh_t[(size_t)i*DK+j]*de[i];
            dk[(size_t)t*DK+j] -= acc; }                              /* dk += −Ŝᵀ de  (delta path) */

        /* Ŝ_t = α_t S_{t−1} */
        double dalpha=0;
        if(Sprev) for(size_t i=0;i<sz;i++) dalpha += dSh[i]*Sprev[i];
        dal[t]=dalpha;
        for(size_t i=0;i<sz;i++) dS[i]=a*dSh[i];                      /* dS_{t−1} = α dŜ */
    }
    free(dS); free(dSh); free(e); free(de);
}

static int fails=0;
static int cmp(const char*nm,double*p,const double*g,int n,double eps){
    double L0=fwd();
    double atol=16.0*fabs(L0)*2.220446049250313e-16/eps; if(atol<1e-14) atol=1e-14;
    double worst=0,wa=0,wf=0,wd=0; int bad=0;
    for(int s=0;s<8;s++){
        int i=(s*n)/8+s; if(i>=n) i=n-1;
        double sv=p[i];
        p[i]=sv+eps; double Lp=fwd();
        p[i]=sv-eps; double Lm=fwd();
        p[i]=sv;     fwd();
        double fd=(Lp-Lm)/(2*eps), an=g[i], ad=fabs(an-fd);
        double rel=fabs(fd)>0?ad/fabs(fd):(fabs(an)>0?1e9:0);
        if(!(ad<=atol || rel<=1e-6)) bad++;
        if(rel>worst){worst=rel;wa=an;wf=fd;wd=ad;}
    }
    printf("  d%-6s rel %.2e  |Δ| %.2e  (floor %.1e)  analytic %+.6e fd %+.6e  %s\n",
           nm,worst,wd,atol,wa,wf, bad?"FAIL":(worst>1e-6?"PASS (within fd floor)":"PASS"));
    return bad;
}
static void check(const char*nm,double*p,const double*g,int n,double eps){
    if(cmp(nm,p,g,n,eps)) fails++;
}

int main(int argc,char**argv){
    T  = argc>1?atoi(argv[1]):24;
    DK = argc>2?atoi(argv[2]):8;
    DV = argc>3?atoi(argv[3]):8;
    double eps = argc>4?atof(argv[4]):1e-6;
    size_t sz=(size_t)DV*DK;
    q=malloc((size_t)T*DK*8); k=malloc((size_t)T*DK*8); v=malloc((size_t)T*DV*8);
    al=malloc(T*8); be=malloc(T*8);
    dq=malloc((size_t)T*DK*8); dk=malloc((size_t)T*DK*8); dv=malloc((size_t)T*DV*8);
    dal=malloc(T*8); dbe=malloc(T*8);
    Sh=malloc((size_t)T*sz*8); Sall=malloc((size_t)T*sz*8); o=malloc((size_t)T*DV*8);
    if(!q||!Sall||!dbe){ printf("alloc failed\n"); return 1; }
    for(int t=0;t<T;t++){
        for(int j=0;j<DK;j++){ q[(size_t)t*DK+j]=rnd(); k[(size_t)t*DK+j]=rnd(); }
        for(int i=0;i<DV;i++) v[(size_t)t*DV+i]=rnd();
        al[t]=0.5+0.4*rnd();            /* decay in (0.1, 0.9) */
        be[t]=0.5+0.4*rnd();            /* write strength      */
    }
    printf("Gated DeltaNet backward (recurrent form)   T=%d dk=%d dv=%d  eps=%g   CPU only\n",T,DK,DV,eps);
    printf("  S_t = a_t S_{t-1};  e_t = v_t - S_t^ k_t;  S_t += b_t e_t k_t^T;  o_t = S_t q_t\n\n");
    double L=fwd(); bwd();
    printf("loss %.9f  over %d timesteps\n\n", L, T);
    check("q", q, dq,  T*DK, eps);
    check("k", k, dk,  T*DK, eps);
    check("v", v, dv,  T*DV, eps);
    check("alpha", al, dal, T,  eps);
    check("beta",  be, dbe, T,  eps);
    /* positive control: the same checks against a backward with one term removed must FAIL.
     * Without this the PASSes above could mean "the checks are too loose" rather than "it is right". */
    printf("\npositive control — delta-path dk term removed, these MUST fail:\n");
    broken=1; bwd();
    int caught = cmp("k[BAD]", k, dk, T*DK, eps) ? 1 : 0;
    broken=0; bwd();
    if(!caught){ printf("  CONTROL DID NOT FAIL — the check is vacuous.\n"); fails++; }
    else printf("  control failed as required: the check has teeth.\n");

    printf("\n%s\n", fails==0
      ? "ALL PASS — GDN backward is correct. This is the oracle the chunked/UT-solve form must match."
      : "SOME FAILED — see above.");
    return fails!=0;
}
