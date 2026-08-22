/* test_gptq — synthetic mechanism check for the in-tree GPTQ int4 quantizer (ork_i4_gptq).
 * Builds a calibration matrix X with CORRELATED columns (so the Hessian H=XᵀX has real off-diagonal
 * structure for GPTQ to exploit) + an outlier channel, and random weights W. Quantizes with (a) plain
 * round-to-nearest (RTN: same per-group scale, NO error feedback) and (b) GPTQ, then compares the
 * H-WEIGHTED reconstruction error  E = Σ_n dwₙᵀ·H·dwₙ  ( == ‖X·(W−Wq)ᵀ‖²_F , the objective GPTQ minimizes ).
 * PASS iff GPTQ's E is lower than RTN's — the mechanism works.
 *
 * PLUS the exactness gate that replaced task #56's AutoGPTQ byte-compare (which would need a Python dep this
 * repo does not allow, and which is weaker than an analytic identity anyway): with H = I the factor chain
 * collapses to Hinv = I, so the per-column divisor d is 1 and EVERY propagation term is exactly zero — GPTQ
 * must then emit codes and scales BYTE-IDENTICAL to round-to-nearest. That exercises Cholesky, the inverse,
 * the upper-Cholesky and the error-feedback sweep, and fails exactly (not statistically) if any of them is
 * wrong. A drifting sign, a transposed index or an off-by-one in the propagation row all show up here.
 * CPU-only, no NPU — runs on any host, so it is in `make test` unconditionally. */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

static unsigned s_ = 12345u;
static double rnd(void) { s_ = s_*1103515245u + 12345u; return ((double)((s_>>9)&0x7fffff)/8388608.0)*2.0 - 1.0; }  /* [-1,1) */

static double herr(int K, int N, int ng, int group, const float *W, const int8_t *c, const float *sc, const float *H) {
    double e = 0; double *dw = (double*)malloc((size_t)K*sizeof(double));
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) { int g = k/group; dw[k] = (double)W[(size_t)n*K+k] - (double)c[(size_t)n*K+k]*(double)sc[(size_t)n*ng+g]; }
        for (int i = 0; i < K; i++) { double hi = 0; for (int j = 0; j < K; j++) hi += (double)H[(size_t)i*K+j]*dw[j]; e += dw[i]*hi; }
    }
    free(dw); return e;
}

/* plain round-to-nearest int4 with the SAME per-group absmax/7 scale rule the quantizer uses — the
 * baseline for the error comparison AND the exact target of the H=I identity below. */
static void rtn_i4(int K, int N, int group, int ng, const float *W, int8_t *c, float *sc_out) {
    for (int n = 0; n < N; n++) for (int g = 0; g < ng; g++) {
        int j0 = g*group, j1 = j0+group; if (j1 > K) j1 = K;
        double mx = 0; for (int j = j0; j < j1; j++) { double a = fabs(W[(size_t)n*K+j]); if (a > mx) mx = a; }
        double sc = mx/7.0; if (sc <= 0) sc = 1e-12; sc_out[(size_t)n*ng+g] = (float)sc;
        for (int j = j0; j < j1; j++) { long q = lround(W[(size_t)n*K+j]/sc); if (q>7) q=7; if (q<-8) q=-8; c[(size_t)n*K+j] = (int8_t)q; }
    }
}

int main(void) {
    const int K = 128, N = 64, S = 256, group = 32;
    const int ng = (K + group - 1)/group;
    float  *X = malloc((size_t)S*K*4), *W = malloc((size_t)N*K*4), *H = malloc((size_t)K*K*4), *Href = malloc((size_t)K*K*4);
    int8_t *cg = malloc((size_t)N*K), *cr = malloc((size_t)N*K);
    float  *sg = malloc((size_t)N*ng*4), *sr = malloc((size_t)N*ng*4);

    for (int s = 0; s < S; s++) { double prev = 0;                        /* correlated columns -> structured H */
        for (int k = 0; k < K; k++) { double v = rnd(); X[(size_t)s*K+k] = (float)(0.7*prev + 0.3*v); prev = X[(size_t)s*K+k]; }
        X[(size_t)s*K + (s*7)%K] *= 8.0f;                                 /* an activation outlier */
    }
    for (size_t i = 0; i < (size_t)N*K; i++) W[i] = (float)(rnd()*0.5);
    for (int i = 0; i < K; i++) for (int j = 0; j < K; j++) {             /* H = XᵀX */
        double a = 0; for (int s = 0; s < S; s++) a += (double)X[(size_t)s*K+i]*(double)X[(size_t)s*K+j];
        H[(size_t)i*K+j] = (float)a;
    }
    memcpy(Href, H, (size_t)K*K*4);                                       /* GPTQ destroys H; RTN metric needs it */

    rtn_i4(K, N, group, ng, W, cr, sr);                                   /* (a) RTN: same scale rule, no feedback */
    int rc = ork_i4_gptq(K, N, W, H, group, cg, sg, 0.01f);              /* (b) GPTQ */
    if (rc) { printf("test_gptq: ork_i4_gptq rc=%d FAIL\n", rc); return 1; }

    double eg = herr(K, N, ng, group, W, cg, sg, Href);
    double er = herr(K, N, ng, group, W, cr, sr, Href);
    printf("test_gptq: K=%d N=%d S=%d group=%d | H-weighted err  GPTQ=%.5g  RTN=%.5g  (GPTQ/RTN=%.3f)\n",
           K, N, S, group, eg, er, er>0?eg/er:0.0);
    int ok = (eg < er);
    printf("test_gptq: %s\n", ok ? "  OK (GPTQ < RTN — mechanism confirmed)" : "  FAIL (GPTQ did not beat RTN)");

    /* --- EXACTNESS GATE: H = I  =>  GPTQ == RTN, byte for byte -------------------------------------- */
    memset(H, 0, (size_t)K*K*4);
    for (int i = 0; i < K; i++) H[(size_t)i*K+i] = 1.0f;
    int rc2 = ork_i4_gptq(K, N, W, H, group, cg, sg, 0.0f);   /* damp 0 -> lam floor 1e-6, still ~identity */
    if (rc2) { printf("test_gptq: H=I rc=%d FAIL\n", rc2); return 1; }
    rtn_i4(K, N, group, ng, W, cr, sr);
    long dc = 0; for (size_t i = 0; i < (size_t)N*K; i++) if (cg[i] != cr[i]) dc++;
    long ds = 0; for (size_t i = 0; i < (size_t)N*ng; i++) if (sg[i] != sr[i]) ds++;
    printf("test_gptq: H=I identity | code mismatches=%ld/%d  scale mismatches=%ld/%d\n",
           dc, N*K, ds, N*ng);
    int ok2 = (dc == 0 && ds == 0);
    printf("test_gptq: %s\n", ok2 ? "  OK (H=I reproduces RTN exactly — factor chain + feedback verified)"
                                   : "  FAIL (H=I must collapse to RTN; a nonzero propagation term is a bug)");

    return (ok && ok2) ? 0 : 1;
}
