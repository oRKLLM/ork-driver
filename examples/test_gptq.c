/* test_gptq — synthetic mechanism check for the in-tree GPTQ int4 quantizer (ork_gptq_i4).
 * Builds a calibration matrix X with CORRELATED columns (so the Hessian H=XᵀX has real off-diagonal
 * structure for GPTQ to exploit) + an outlier channel, and random weights W. Quantizes with (a) plain
 * round-to-nearest (RTN: same per-group scale, NO error feedback) and (b) GPTQ, then compares the
 * H-WEIGHTED reconstruction error  E = Σ_n dwₙᵀ·H·dwₙ  ( == ‖X·(W−Wq)ᵀ‖²_F , the objective GPTQ minimizes ).
 * PASS iff GPTQ's E is lower than RTN's — the mechanism works. This is NOT the correctness gate; that is a
 * byte/PPL comparison vs AutoGPTQ on a fixed (W,H) golden (task #56). CPU-only, no NPU — runs on any host. */
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

    for (int n = 0; n < N; n++) for (int g = 0; g < ng; g++) {            /* (a) RTN, same per-group scale, no feedback */
        int j0 = g*group, j1 = j0+group; if (j1 > K) j1 = K;
        double mx = 0; for (int c = j0; c < j1; c++) { double a = fabs(W[(size_t)n*K+c]); if (a > mx) mx = a; }
        double sc = mx/7.0; if (sc <= 0) sc = 1e-12; sr[(size_t)n*ng+g] = (float)sc;
        for (int c = j0; c < j1; c++) { long q = lround(W[(size_t)n*K+c]/sc); if (q>7) q=7; if (q<-8) q=-8; cr[(size_t)n*K+c] = (int8_t)q; }
    }
    int rc = ork_gptq_i4(K, N, W, H, group, cg, sg, 0.01f);              /* (b) GPTQ */
    if (rc) { printf("test_gptq: ork_gptq_i4 rc=%d FAIL\n", rc); return 1; }

    double eg = herr(K, N, ng, group, W, cg, sg, Href);
    double er = herr(K, N, ng, group, W, cr, sr, Href);
    printf("test_gptq: K=%d N=%d S=%d group=%d | H-weighted err  GPTQ=%.5g  RTN=%.5g  (GPTQ/RTN=%.3f)\n",
           K, N, S, group, eg, er, er>0?eg/er:0.0);
    int ok = (eg < er);
    printf("test_gptq: %s\n", ok ? "OK (GPTQ < RTN — mechanism confirmed)" : "FAIL (GPTQ did not beat RTN)");
    return ok ? 0 : 1;
}
