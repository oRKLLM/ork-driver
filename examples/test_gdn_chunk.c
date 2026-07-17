/* test_gdn_chunk — Phase-0 reference harness for on-NPU Gated-DeltaNet (SSM_ON_NPU_PLAN.md).
 *
 * The GDN twin of test_ssd_chunk.c: an fp32/double CPU reference for one Gated-DeltaNet (GDA)
 * chunk-scan layer, decomposed into exactly the op-stages the NPU kernel will implement, so every
 * later NPU op has a bit-exact target to validate against. PURE-CPU self-test (no NPU calls yet).
 *
 * It proves the *chunked reformulation* equals the definitional per-token recurrence BEFORE any
 * hardware work. The recurrence (mirrors the fork's delta-net-base.cpp build_delta_net_autoregressive,
 * GDA = scalar gate per token; state S indexed [key,val], head_dim d = S_k = S_v):
 *     S      <- exp(g_t)·S                         (gate decays the whole state)
 *     pred   =  Sᵀ k_t                             (prediction from the decayed state)
 *     w_t    =  beta_t·(v_t - pred)                (delta write, value-space)
 *     S      <- S + k_t · w_tᵀ                     (rank-1 outer-product update)
 *     o_t    =  Sᵀ q_t          (q pre-scaled 1/√d)
 *
 * The chunked form linearizes the intra-chunk dependency into a unit-lower-triangular solve
 * (the UT / WY transform = ggml solve_tri). With cumulative gate G_l = Σ_{i<=l} g_i, a_l = exp(G_l):
 *     A[l,s] = beta_l·(a_l/a_s)·(k_l·k_s)   for s<l   (strictly lower)   ; T = (I+A)^{-1}
 *     rhs_l  = beta_l·v_l - beta_l·a_l·(S0ᵀ k_l)      ; W = T·rhs   (forward substitution)
 *     o_l    = a_l·(S0ᵀ q_l) + Σ_{s<=l} (a_l/a_s)(k_s·q_l)·w_s        (inter + intra chunk)
 *     S_end  = a_{CS-1}·S0 + Σ_s (a_{CS-1}/a_s)·k_s·w_sᵀ             (inter-chunk carry)
 * These stages map 1:1 onto NPU matmuls (the gram k·kᵀ, the WY matmuls, the carry matmuls) + one
 * chunk-local triangular solve (CPU-first; the only non-matmul piece).
 *
 * Exits 0 on all-pass, 1 on any mismatch. Part of `make test` (the examples ARE the tests).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double frand(void){ return (double)rand()/RAND_MAX; }          /* [0,1) */
static double frand_sym(void){ return frand()*2.0 - 1.0; }            /* [-1,1) */

/* GDN layer dims (one sequence, batch=1 — the chunk kernel is per (head, sequence) slice).
 * GDA: gate is a scalar per (head,token). State is square: S_k == S_v == d (head_dim). */
typedef struct {
    int H;    /* num value heads (each an independent scan)  */
    int d;    /* head_dim = S_k = S_v (square state d×d)      */
    int CS;   /* chunk_size                                   */
    int NC;   /* num_chunks   → L = NC*CS                     */
} gdn_dims;

/* indexing: q,k,v,o [L,H,d]; g,beta [L,H]; state S[d,d] = [key,val] per head */
#define IX_QKV(t,h,e)  (((size_t)(t)*dm->H + (h))*dm->d + (e))
#define IX_GB(t,h)     ((size_t)(t)*dm->H + (h))
#define IX_S(key,val)  ((size_t)(key)*dm->d + (val))

/* ============================================================================================
 * Ground truth: the definitional gated-delta-rule recurrence, per head, per token.
 * ============================================================================================ */
static void gdn_sequential(const gdn_dims *dm, const double *q, const double *k, const double *v,
                           const double *g, const double *beta, double *o)
{
    const int L = dm->NC*dm->CS, d = dm->d;
    const double qscale = 1.0/sqrt((double)d);
    double *S    = malloc((size_t)d*d*sizeof(double));
    double *pred = malloc((size_t)d*sizeof(double));
    for (int h=0; h<dm->H; h++){
        memset(S, 0, (size_t)d*d*sizeof(double));
        for (int t=0; t<L; t++){
            double gate = exp(g[IX_GB(t,h)]);
            /* decay */
            for (size_t i=0;i<(size_t)d*d;i++) S[i]*=gate;
            /* pred = Sᵀ k  (contract key dim) */
            for (int val=0; val<d; val++){
                double acc=0;
                for (int key=0; key<d; key++) acc += S[IX_S(key,val)]*k[IX_QKV(t,h,key)];
                pred[val]=acc;
            }
            /* write w = beta*(v - pred); S += k · wᵀ */
            double bt = beta[IX_GB(t,h)];
            for (int key=0; key<d; key++){
                double kk = k[IX_QKV(t,h,key)];
                for (int val=0; val<d; val++){
                    double w = bt*(v[IX_QKV(t,h,val)] - pred[val]);
                    S[IX_S(key,val)] += kk*w;
                }
            }
            /* o = Sᵀ q  (q pre-scaled) */
            for (int val=0; val<d; val++){
                double acc=0;
                for (int key=0; key<d; key++) acc += S[IX_S(key,val)]*q[IX_QKV(t,h,key)]*qscale;
                o[IX_QKV(t,h,val)]=acc;
            }
        }
    }
    free(S); free(pred);
}

/* ============================================================================================
 * Chunked GDN (fp64), staged like the fork's build_delta_net_chunking (GDA path). Per head, per
 * chunk: cumsum(g) → gram+decay A → forward-subst UT-transform (T=(I+A)^-1) for the writes W →
 * intra+inter output → inter-chunk state carry.  Mirrors the NPU op-graph 1:1.
 * ============================================================================================ */
static void gdn_chunked(const gdn_dims *dm, const double *q, const double *k, const double *v,
                        const double *g, const double *beta, double *o)
{
    const int CS=dm->CS, NC=dm->NC, d=dm->d;
    const double qscale = 1.0/sqrt((double)d);
    double *S   = calloc((size_t)d*d, sizeof(double));  /* state entering the chunk (S0)   */
    double *acs = malloc((size_t)CS*sizeof(double));    /* a_l = exp(cumsum g) within chunk*/
    double *W   = malloc((size_t)CS*d*sizeof(double));  /* per-token writes w_l[val]       */
    double *Sk  = malloc((size_t)CS*d*sizeof(double));  /* a_l·(S0ᵀ k_l)[val]  (RHS carry) */
    double *Sq  = malloc((size_t)CS*d*sizeof(double));  /* a_l·(S0ᵀ q_l)[val]  (inter out) */

    for (int h=0; h<dm->H; h++){
        memset(S, 0, (size_t)d*d*sizeof(double));
        for (int c=0; c<NC; c++){
            int base=c*CS;
            /* cumulative gate a_l = exp(Σ_{i<=l} g); and S0-projections Sk, Sq (inter-chunk terms) */
            double run=0;
            for (int l=0;l<CS;l++){
                run += g[IX_GB(base+l,h)];
                acs[l]=exp(run);
                for (int val=0; val<d; val++){
                    double sk=0, sq=0;
                    for (int key=0; key<d; key++){
                        double s=S[IX_S(key,val)];
                        sk += s*k[IX_QKV(base+l,h,key)];
                        sq += s*q[IX_QKV(base+l,h,key)]*qscale;
                    }
                    Sk[(size_t)l*d+val]=acs[l]*sk;
                    Sq[(size_t)l*d+val]=acs[l]*sq;
                }
            }
            /* UT-transform: W = (I+A)^{-1} rhs by forward substitution, row l depends on s<l.
             *   A[l,s] = beta_l·(a_l/a_s)·(k_l·k_s)    (strictly lower)
             *   rhs_l  = beta_l·v_l - Sk_l   (= beta_l v_l - beta_l a_l S0ᵀk_l ... beta folded below) */
            for (int l=0;l<CS;l++){
                double bl = beta[IX_GB(base+l,h)];
                for (int val=0; val<d; val++)
                    W[(size_t)l*d+val] = bl*(v[IX_QKV(base+l,h,val)] - Sk[(size_t)l*d+val]);
                for (int s=0;s<l;s++){
                    double kk=0;
                    for (int e=0;e<d;e++) kk += k[IX_QKV(base+l,h,e)]*k[IX_QKV(base+s,h,e)];
                    double a = bl*(acs[l]/acs[s])*kk;              /* A[l,s] */
                    for (int val=0; val<d; val++) W[(size_t)l*d+val] -= a*W[(size_t)s*d+val];
                }
            }
            /* output: o_l = Sq_l (inter) + Σ_{s<=l} (a_l/a_s)(k_s·q_l) w_s (intra) */
            for (int l=0;l<CS;l++){
                for (int val=0; val<d; val++) o[IX_QKV(base+l,h,val)] = Sq[(size_t)l*d+val];
                for (int s=0;s<=l;s++){
                    double kq=0;
                    for (int e=0;e<d;e++) kq += k[IX_QKV(base+s,h,e)]*q[IX_QKV(base+l,h,e)]*qscale;
                    double coef = (acs[l]/acs[s])*kq;
                    for (int val=0; val<d; val++) o[IX_QKV(base+l,h,val)] += coef*W[(size_t)s*d+val];
                }
            }
            /* inter-chunk carry: S_end = a_{CS-1} S0 + Σ_s (a_{CS-1}/a_s) k_s · w_sᵀ */
            double alast=acs[CS-1];
            for (size_t i=0;i<(size_t)d*d;i++) S[i]*=alast;
            for (int s=0;s<CS;s++){
                double dec=alast/acs[s];
                for (int key=0; key<d; key++){
                    double kk=dec*k[IX_QKV(base+s,h,key)];
                    for (int val=0; val<d; val++) S[IX_S(key,val)] += kk*W[(size_t)s*d+val];
                }
            }
        }
    }
    free(S);free(acs);free(W);free(Sk);free(Sq);
}

static double maxrel(const gdn_dims *dm, const double *a, const double *b){
    int L=dm->NC*dm->CS; double mx=0;
    for (size_t i=0;i<(size_t)L*dm->H*dm->d;i++){
        double e=fabs(a[i]-b[i])/(fabs(b[i])+1e-6);
        if(e>mx) mx=e;
    }
    return mx;
}

static int run_case(const char *tag, gdn_dims dm, double tol){
    int L=dm.NC*dm.CS;
    double *q=malloc((size_t)L*dm.H*dm.d*sizeof(double));
    double *k=malloc((size_t)L*dm.H*dm.d*sizeof(double));
    double *v=malloc((size_t)L*dm.H*dm.d*sizeof(double));
    double *g=malloc((size_t)L*dm.H*sizeof(double));
    double *bt=malloc((size_t)L*dm.H*sizeof(double));
    double *o_seq=malloc((size_t)L*dm.H*dm.d*sizeof(double));
    double *o_chk=malloc((size_t)L*dm.H*dm.d*sizeof(double));

    for (size_t i=0;i<(size_t)L*dm.H*dm.d;i++){ q[i]=frand_sym(); k[i]=frand_sym(); v[i]=frand_sym(); }
    /* gate g<=0 (log-decay, exp(g)∈(0,1]); beta∈(0,1) the delta write strength. */
    for (size_t i=0;i<(size_t)L*dm.H;i++){ g[i]=-(frand()*0.6+0.02); bt[i]=frand()*0.9+0.05; }

    gdn_sequential(&dm,q,k,v,g,bt,o_seq);
    gdn_chunked   (&dm,q,k,v,g,bt,o_chk);
    double mr=maxrel(&dm,o_chk,o_seq);

    int fail = !(mr<=tol);
    fprintf(stderr,"[%s] H=%d d=%d CS=%d NC=%d  chunked-vs-sequential maxrel=%.3e  %s\n",
            tag,dm.H,dm.d,dm.CS,dm.NC,mr, fail?"FAIL":"OK");
    free(q);free(k);free(v);free(g);free(bt);free(o_seq);free(o_chk);
    return fail;
}

int main(void){
    srand(20260713);
    int fail=0;

    /* Chunked GDN == sequential recurrence, across chunk layouts and head_dims.
     * fp64 accumulation differs only by summation order → rel err ~1e-12; tol 1e-6 is ample.
     * Last case is the Qwen3-Next GDA shape (head_dim 128, CS 64). */
    fail |= run_case("gdn-1chunk",   (gdn_dims){.H=2,.d=8,  .CS=8, .NC=1}, 1e-6);
    fail |= run_case("gdn-multi",    (gdn_dims){.H=2,.d=8,  .CS=8, .NC=4}, 1e-6);
    fail |= run_case("gdn-d16",      (gdn_dims){.H=4,.d=16, .CS=16,.NC=3}, 1e-6);
    fail |= run_case("gdn-cs32",     (gdn_dims){.H=2,.d=32, .CS=32,.NC=2}, 1e-6);
    fail |= run_case("gdn-qwen3next",(gdn_dims){.H=2,.d=128,.CS=64,.NC=3}, 1e-6);

    fprintf(stderr, fail ? "\nTEST_GDN_CHUNK: FAIL\n" : "\nTEST_GDN_CHUNK: PASS\n");
    return fail?1:0;
}
