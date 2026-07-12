/* test_ssd_chunk — Phase-0 reference harness for on-NPU Mamba-2 / SSD (SSM_ON_NPU_PLAN.md).
 *
 * This is the *spine* of the SSM-on-NPU work: an fp32 CPU reference for one Mamba-2 SSD
 * chunk-scan layer, decomposed into exactly the op-stages the NPU kernel will implement
 * (XAMBA CumBA/ReduBA/ActiBA), so every later NPU op has a bit-exact target to validate against.
 *
 * It is a PURE-CPU self-test (no NPU calls yet — Phases 1-2 add NPU-backed variants). It proves the
 * *reformulation* is correct before any hardware work:
 *   1. cumsum-as-matmul (CumBA): cumsum(v) == tril_ones[CS,CS] · v, bit-close to a direct cumsum.
 *   2. the chunked SSD-naive scan (xamba.py:torch_forward "ssd naive", the XAMBA reference) equals
 *      the definitional per-token sequential recurrence  h_t = exp(dt_t·A)·h_{t-1} + dt_t·B_t·x_t,
 *      y_t = C_t·h_t + D·x_t  — across single- and multi-chunk sequences and several group counts.
 *
 * The staged reference (segsum decay L, scores G=C·Bᵀ, masked M, Y_diag, per-chunk state, the
 * O(n_chunks) inter-chunk carry, Y_off) mirrors the tensor ops in xamba.py 1:1 so a future NPU
 * implementation can be diffed stage-by-stage.  Config follows the plan's dtype/shape map (SSD CS=64).
 *
 * Exits 0 on all-pass, 1 on any mismatch. Part of `make test` (the examples ARE the tests).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- tiny fp32 helpers ---------------------------------------------------------------------- */
static double frand(void){ return (double)rand()/RAND_MAX; }          /* [0,1) */
static double frand_sym(void){ return frand()*2.0 - 1.0; }            /* [-1,1) */

/* SSD layer dims (one sequence, batch=1 — the chunk kernel is per (head, sequence) slice). */
typedef struct {
    int H;    /* num_heads                */
    int P;    /* head_dim                 */
    int Nst;  /* ssm_state_size           */
    int G;    /* n_groups (B/C shared across H/G heads; head h uses group h%G, per xamba .repeat) */
    int CS;   /* chunk_size               */
    int NC;   /* num_chunks   → L = NC*CS */
} ssd_dims;

/* indexing */
#define IX_XHP(t,h,p)  (((size_t)(t)*d->H + (h))*d->P + (p))          /* x, y   [L,H,P] */
#define IX_DT(t,h)     ((size_t)(t)*d->H + (h))                       /* dt     [L,H]   */
#define IX_BC(t,g,n)   (((size_t)(t)*d->G + (g))*d->Nst + (n))        /* B,C    [L,G,N] */

/* ============================================================================================
 * Ground truth: the definitional Mamba-2 SSM sequential recurrence, per head, per token.
 *   state[p][n] = exp(dt_t·A_h)·state[p][n] + (dt_t·B_{t,n})·x_{t,p}
 *   y_{t,p}     = Σ_n C_{t,n}·state[p][n] + D_h·x_{t,p}
 * (x here is the post-conv value; dt = relu(dt_raw+dt_bias) clamped, precomputed by the caller.)
 * ============================================================================================ */
static void ssd_sequential(const ssd_dims *d, const double *x, const double *dt,
                           const double *A, const double *B, const double *C, const double *D,
                           double *y)
{
    int L = d->NC * d->CS;
    double *state = calloc((size_t)d->P*d->Nst, sizeof(double));
    for (int h=0; h<d->H; h++){
        int g = h % d->G;
        memset(state, 0, (size_t)d->P*d->Nst*sizeof(double));
        for (int t=0; t<L; t++){
            double dtv = dt[IX_DT(t,h)];
            double dA  = exp(dtv * A[h]);                 /* A[h] < 0 → dA ∈ (0,1] */
            for (int p=0; p<d->P; p++){
                double xv = x[IX_XHP(t,h,p)];
                double dtx = dtv * xv;                    /* dt·x  (discretized input scale) */
                for (int n=0; n<d->Nst; n++)
                    state[(size_t)p*d->Nst+n] = state[(size_t)p*d->Nst+n]*dA
                                              + dtx * B[IX_BC(t,g,n)];
                double acc = 0;
                for (int n=0; n<d->Nst; n++) acc += C[IX_BC(t,g,n)] * state[(size_t)p*d->Nst+n];
                y[IX_XHP(t,h,p)] = acc + D[h]*xv;
            }
        }
    }
    free(state);
}

/* ============================================================================================
 * cumsum-as-matmul (XAMBA CumBA):  out[i] = Σ_{j<=i} v[j]  == (tril_ones[CS,CS] · v)[i].
 * The NPU computes the chunk cumsum this exact way (a tiny lower-triangular-ones GEMM), so we
 * validate the identity here in fp32.  Returns max abs error vs a direct running-sum cumsum.
 * ============================================================================================ */
static double cumba_check(int CS)
{
    double *v = malloc((size_t)CS*sizeof(double));
    double *direct = malloc((size_t)CS*sizeof(double));
    double *viamm  = malloc((size_t)CS*sizeof(double));
    for (int i=0;i<CS;i++) v[i]=frand_sym();
    double run=0; for (int i=0;i<CS;i++){ run+=v[i]; direct[i]=run; }        /* direct cumsum */
    for (int i=0;i<CS;i++){ double s=0; for (int j=0;j<=i;j++) s+=v[j]; viamm[i]=s; } /* tril·v */
    double maxerr=0; for (int i=0;i<CS;i++){ double e=fabs(viamm[i]-direct[i]); if(e>maxerr)maxerr=e; }
    free(v);free(direct);free(viamm);
    return maxerr;
}

/* ============================================================================================
 * Chunked SSD-naive (fp32), staged exactly like xamba.py:torch_forward "ssd naive".
 * Stages (per head h, group g=h%G):
 *   Abar[t]      = dt_t·A_h                                    (discretized decay)
 *   xbar[t,p]    = dt_t·x_{t,p}                                (discretized input)
 *   A_cumsum[c,l]= Σ_{i<=l} Abar[c·CS+i]                       (CumBA over the chunk)
 *   L[l,s]       = exp(A_cumsum[l]-A_cumsum[s]) for s<=l else 0  (segsum decay mask, intra-chunk)
 *   G[l,s]       = Σ_n C_{cl,n}·B_{cs,n}                        (scores, C·Bᵀ contraction)
 *   M[l,s]       = G[l,s]·L[l,s]                               (masked scores)
 *   Y_diag[c,l,p]= Σ_{s<=l} M[l,s]·xbar[cs,p]                  (intra-chunk output)
 *   cstate[c,p,n]= Σ_s exp(A_cumsum[CS-1]-A_cumsum[s])·B_{cs,n}·xbar[cs,p]  (this chunk's state)
 *   state_in[c]  = exp(A_cumsum[c-1,CS-1])·state_in[c-1] + cstate[c-1]      (O(NC) inter-chunk carry)
 *   Y_off[c,l,p] = (Σ_n C_{cl,n}·state_in[c,p,n])·exp(A_cumsum[c,l])        (inter-chunk output)
 *   y[t,p]       = Y_diag + Y_off + D_h·x_{t,p}
 * ============================================================================================ */
static void ssd_chunked(const ssd_dims *d, const double *x, const double *dt,
                        const double *A, const double *B, const double *C, const double *D,
                        double *y)
{
    const int CS=d->CS, NC=d->NC, P=d->P, Nst=d->Nst;
    double *Abar     = malloc((size_t)CS*sizeof(double));
    double *Acs      = malloc((size_t)CS*sizeof(double));                 /* A_cumsum for one chunk */
    double *xbar     = malloc((size_t)CS*P*sizeof(double));               /* xbar[s,p] for one chunk */
    double *state_in = calloc((size_t)P*Nst, sizeof(double));             /* state entering chunk c */
    double *cstate   = malloc((size_t)P*Nst*sizeof(double));             /* this chunk's own state */
    double *Acs_last = malloc((size_t)NC*sizeof(double));                /* per-chunk total decay a_c */

    for (int h=0; h<d->H; h++){
        int g = h % d->G;
        memset(state_in, 0, (size_t)P*Nst*sizeof(double));

        for (int c=0; c<NC; c++){
            int base = c*CS;
            /* Abar, A_cumsum (CumBA), xbar */
            double run=0;
            for (int l=0;l<CS;l++){
                double dtv = dt[IX_DT(base+l,h)];
                Abar[l] = dtv * A[h];
                run += Abar[l]; Acs[l] = run;
                for (int p=0;p<P;p++) xbar[(size_t)l*P+p] = dtv * x[IX_XHP(base+l,h,p)];
            }
            Acs_last[c] = Acs[CS-1];

            /* inter-chunk carry: state entering THIS chunk (Y_off uses state_in for chunk c) */
            if (c>0){
                double decay_prev = exp(Acs_last[c-1]);
                for (size_t i=0;i<(size_t)P*Nst;i++) state_in[i] = decay_prev*state_in[i] + cstate[i];
            }
            /* NB: cstate below is THIS chunk's contribution, folded into state_in for the NEXT chunk. */

            /* ---- Y_diag (intra-chunk) + Y_off (inter-chunk) ---- */
            for (int l=0; l<CS; l++){
                int t = base+l;
                double sdo = exp(Acs[l]);                       /* state_decay_out */
                for (int p=0;p<P;p++){
                    /* Y_diag: Σ_{s<=l} G[l,s]·L[l,s]·xbar[s,p] */
                    double yd = 0;
                    for (int s=0;s<=l;s++){
                        double Lls = exp(Acs[l]-Acs[s]);        /* segsum decay, s<=l */
                        double Gls = 0;
                        for (int n=0;n<Nst;n++) Gls += C[IX_BC(t,g,n)]*B[IX_BC(base+s,g,n)];
                        yd += Gls*Lls*xbar[(size_t)s*P+p];
                    }
                    /* Y_off: (Σ_n C[t,n]·state_in[p,n])·exp(A_cumsum[l]) */
                    double yo = 0;
                    for (int n=0;n<Nst;n++) yo += C[IX_BC(t,g,n)]*state_in[(size_t)p*Nst+n];
                    yo *= sdo;
                    y[IX_XHP(t,h,p)] = yd + yo + D[h]*x[IX_XHP(t,h,p)];
                }
            }

            /* ---- this chunk's own state contribution (for the next chunk's carry) ---- */
            for (int p=0;p<P;p++) for (int n=0;n<Nst;n++){
                double acc=0;
                for (int s=0;s<CS;s++){
                    double ds = exp(Acs[CS-1]-Acs[s]);          /* decay from s to chunk end */
                    acc += ds * B[IX_BC(base+s,g,n)] * xbar[(size_t)s*P+p];
                }
                cstate[(size_t)p*Nst+n] = acc;
            }
        }
    }
    free(Abar);free(Acs);free(xbar);free(state_in);free(cstate);free(Acs_last);
}

/* max relative error between two [L,H,P] tensors */
static double maxrel_xhp(const ssd_dims *d, const double *a, const double *b){
    int L=d->NC*d->CS; double mx=0;
    for (size_t i=0;i<(size_t)L*d->H*d->P;i++){
        double e=fabs(a[i]-b[i])/(fabs(b[i])+1e-6);
        if(e>mx) mx=e;
    }
    return mx;
}

/* Run one SSD config: build random inputs, compute both references, compare. */
static int run_case(const char *tag, ssd_dims d, double tol){
    int L = d.NC*d.CS;
    double *x  = malloc((size_t)L*d.H*d.P*sizeof(double));
    double *dt = malloc((size_t)L*d.H*sizeof(double));
    double *A  = malloc((size_t)d.H*sizeof(double));
    double *B  = malloc((size_t)L*d.G*d.Nst*sizeof(double));
    double *C  = malloc((size_t)L*d.G*d.Nst*sizeof(double));
    double *D  = malloc((size_t)d.H*sizeof(double));
    double *y_seq = malloc((size_t)L*d.H*d.P*sizeof(double));
    double *y_chk = malloc((size_t)L*d.H*d.P*sizeof(double));

    for (size_t i=0;i<(size_t)L*d.H*d.P;i++) x[i]=frand_sym();
    /* dt = relu(dt_raw + dt_bias) clamped to a small positive floor (ActiBA: softplus→relu).
     * Keep dt modest so exp(dt·A) doesn't underflow to a constant — realistic decode range. */
    for (size_t i=0;i<(size_t)L*d.H;i++){ double v=frand_sym()*0.5+0.3; dt[i]= v>1e-4? v:1e-4; }
    for (int h=0;h<d.H;h++){ double a_log=frand()*1.5; A[h]=-exp(a_log); D[h]=frand(); } /* A<0 */
    for (size_t i=0;i<(size_t)L*d.G*d.Nst;i++){ B[i]=frand_sym(); C[i]=frand_sym(); }

    ssd_sequential(&d,x,dt,A,B,C,D,y_seq);
    ssd_chunked   (&d,x,dt,A,B,C,D,y_chk);
    double mr = maxrel_xhp(&d,y_chk,y_seq);

    int fail = !(mr<=tol);
    fprintf(stderr,"[%s] H=%d P=%d N=%d G=%d CS=%d NC=%d  chunked-vs-sequential maxrel=%.3e  %s\n",
            tag,d.H,d.P,d.Nst,d.G,d.CS,d.NC,mr, fail?"FAIL":"OK");
    free(x);free(dt);free(A);free(B);free(C);free(D);free(y_seq);free(y_chk);
    return fail;
}

int main(void){
    srand(20260712);
    int fail = 0;

    /* CumBA identity (cumsum == tril_ones·v) at the SSD and KDA chunk sizes */
    double e64 = cumba_check(64), e16 = cumba_check(16);
    int cf = (e64>1e-9)||(e16>1e-9);
    fprintf(stderr,"[cumba] tril-ones·v vs direct cumsum: CS=64 err=%.2e  CS=16 err=%.2e  %s\n",
            e64,e16, cf?"FAIL":"OK");
    fail |= cf;

    /* Chunked SSD-naive == sequential recurrence, across group counts and chunk layouts.
     * fp32/double accumulation differs only in summation order → rel err ~1e-12; tol 1e-6 is ample. */
    fail |= run_case("ssd-1chunk-G1",  (ssd_dims){.H=4,.P=8, .Nst=16,.G=1,.CS=8, .NC=1}, 1e-6);
    fail |= run_case("ssd-multi-G1",   (ssd_dims){.H=4,.P=8, .Nst=16,.G=1,.CS=8, .NC=4}, 1e-6);
    fail |= run_case("ssd-multi-G2",   (ssd_dims){.H=8,.P=16,.Nst=16,.G=2,.CS=16,.NC=3}, 1e-6);
    fail |= run_case("ssd-multi-GH",   (ssd_dims){.H=4,.P=16,.Nst=32,.G=4,.CS=16,.NC=3}, 1e-6);
    fail |= run_case("ssd-CS64",       (ssd_dims){.H=4,.P=64,.Nst=128,.G=1,.CS=64,.NC=2}, 1e-6);

    fprintf(stderr, fail ? "\nTEST_SSD_CHUNK: FAIL\n" : "\nTEST_SSD_CHUNK: PASS\n");
    return fail ? 1 : 0;
}
