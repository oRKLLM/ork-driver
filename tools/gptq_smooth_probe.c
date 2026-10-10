/* gptq_smooth_probe — does SmoothQuant compose with ork_i4_gptq, or does it break its numerics?
 *
 * WHY A UNIT PROBE. End-to-end the answer costs ~15 min per cycle (GPTQ pack build + 64-window PPL) and
 * confounds four independent things: the pack path, the persist/load round-trip, the Hessian basis, and
 * GPTQ itself. Three debugging cycles each blamed a different one of those and each was wrong. This
 * isolates the last one: no NPU, no pack, no model — just the quantizer, in seconds.
 *
 * THE IDENTITY UNDER TEST.  out = W·a  =  (s⊙W)·(a⊘s)
 * so quantizing the SMOOTHED weight and feeding it SMOOTHED-DOWN activations must reconstruct the same
 * product as quantizing W against a. Both arms are scored on the SAME quantity — error against the
 * exact fp32 product W·aᵀ — so they are directly comparable.
 *
 * WHAT WOULD INDICT GPTQ. Smoothing makes diag(H) far more non-uniform (a column whose activations are
 * divided by s has its energy cut by s²). ork_i4_gptq damps with a single global lam = damp·mean(diag H)
 * and ZEROES any column whose diagonal lands <= 0:
 *     if (Hd[i*K+i] <= 0.0) { ...; for (n) Wd[n*K+i] = 0.0; }
 * If smoothing pushes columns under that threshold, their weights are destroyed outright — which would
 * look exactly like the measured end-to-end failure (PPL 3e8: destroyed, not degraded).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "ork_npu.h"
/* Verbatim copy of src/npu/norm.c's ork_fwht_norm — inlined rather than linked, because norm.c drags in
 * the NPU symbols and this probe must stay CPU-only. Same convention: in-place, then 1/sqrt(n). */
static void probe_fwht(float *v, int n) {
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += len << 1)
            for (int j = i; j < i + len; j++) { float a = v[j], b = v[j+len]; v[j] = a + b; v[j+len] = a - b; }
    const float s = 1.0f / sqrtf((float) n);
    for (int i = 0; i < n; i++) v[i] *= s;
}

/* Rotate every b-sized block of a K-vector, exactly as ork_w4a4_rot does. */
static void rot(float *v, int K, int b) { for (int off = 0; off < K; off += b) probe_fwht(v + off, b); }

static uint64_t rs = 88172645463325252ULL;
static double rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) / 9007199254740992.0; }
static double gauss(void) { double u = rnd(), v = rnd(); if (u < 1e-12) u = 1e-12; return sqrt(-2.0*log(u))*cos(6.283185307179586*v); }

/* H = AᵀA over M rows (lower+upper, full matrix as ork_i4_gptq expects). */
static void hess(int M, int K, const float *A, float *H) {
    memset(H, 0, (size_t)K*K*sizeof(float));
    for (int m = 0; m < M; m++) {
        const float *a = A + (size_t)m*K;
        for (int i = 0; i < K; i++) {
            const float ai = a[i];
            if (ai == 0.0f) continue;
            for (int j = 0; j < K; j++) H[(size_t)i*K + j] += ai * a[j];
        }
    }
}

/* Relative error of the quantized product against the exact one, over all (m,n). */
static double prod_err(int M, int N, int K, const float *A, const float *W,
                       const int8_t *codes, const float *scales) {
    double num = 0.0, den = 0.0;
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        double exact = 0.0, got = 0.0;
        const float sc = scales[n];                        /* per-row (group<=0) */
        for (int k = 0; k < K; k++) {
            exact += (double)W[(size_t)n*K + k] * (double)A[(size_t)m*K + k];
            got   += (double)codes[(size_t)n*K + k] * (double)sc * (double)A[(size_t)m*K + k];
        }
        const double d = exact - got;
        num += d*d; den += exact*exact;
    }
    return den > 0.0 ? sqrt(num/den) : -1.0;
}

int main(int argc, char **argv) {
    const int K = argc > 1 ? atoi(argv[1]) : 512;
    const int N = argc > 2 ? atoi(argv[2]) : 64;
    const int M = argc > 3 ? atoi(argv[3]) : 512;      /* rows of calibration/eval activations */
    const float alpha = argc > 4 ? (float)atof(argv[4]) : 0.5f;

    float *W  = malloc((size_t)N*K*sizeof *W);
    float *A  = malloc((size_t)M*K*sizeof *A);
    float *Ws = malloc((size_t)N*K*sizeof *Ws);
    float *As = malloc((size_t)M*K*sizeof *As);
    float *H  = malloc((size_t)K*K*sizeof *H);
    float *s  = malloc((size_t)K*sizeof *s);
    int8_t *codes = malloc((size_t)N*K);
    float  *scal  = malloc((size_t)N*sizeof *scal);

    for (size_t i = 0; i < (size_t)N*K; i++) W[i] = (float)(gauss() * 0.02);
    /* Activations WITH outlier channels — the thing smoothing exists for. 3% of channels get 30x. */
    for (int k = 0; k < K; k++) {
        const double g = (rnd() < 0.03) ? 30.0 : 1.0;
        for (int m = 0; m < M; m++) A[(size_t)m*K + k] = (float)(gauss() * g);
    }

    /* SmoothQuant scale, same construction as ggml-ork's ork_smooth_build. */
    double logsum = 0.0;
    for (int k = 0; k < K; k++) {
        double ae = 0.0; for (int m = 0; m < M; m++) { const double v = A[(size_t)m*K+k]; ae += v*v; }
        ae = sqrt(ae / (double)M);
        double wm = 1e-12; for (int n = 0; n < N; n++) { const double v = fabs(W[(size_t)n*K+k]); if (v > wm) wm = v; }
        s[k] = (float)(pow(ae < 1e-12 ? 1e-12 : ae, alpha) / pow(wm, 1.0 - alpha));
        logsum += log(s[k]);
    }
    { const double gm = exp(logsum / K); for (int k = 0; k < K; k++) s[k] = (float)(s[k] / gm); }
    { float lo = s[0], hi = s[0];
      for (int k = 1; k < K; k++) { if (s[k] < lo) lo = s[k]; if (s[k] > hi) hi = s[k]; }
      printf("smoothing scale range %.4f .. %.4f (alpha=%.2f)\n", lo, hi, alpha); }

    for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) Ws[(size_t)n*K+k] = W[(size_t)n*K+k] * s[k];
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) As[(size_t)m*K+k] = A[(size_t)m*K+k] / s[k];

    /* ---- ARM 1: plain GPTQ --------------------------------------------------------------------- */
    hess(M, K, A, H);
    { double mn = H[0], mx = H[0];
      for (int i = 1; i < K; i++) { const double d = H[(size_t)i*K+i]; if (d < mn) mn = d; if (d > mx) mx = d; }
      printf("arm1 diag(H): min %.6g max %.6g  ratio %.3g\n", mn, mx, mx / (mn > 0 ? mn : 1e-300)); }
    int rc1 = ork_i4_gptq(K, N, W, H, 0, codes, scal, 0.01f);
    const double e1 = rc1 ? -1.0 : prod_err(M, N, K, A, W, codes, scal);

    /* ---- ARM 2: SmoothQuant + GPTQ -------------------------------------------------------------
     * Scored against the SAME exact product: the smoothed weight is fed smoothed-down activations, so
     * if the identity holds this must land near arm 1. */
    hess(M, K, As, H);
    { double mn = H[0], mx = H[0]; int nz = 0;
      for (int i = 0; i < K; i++) { const double d = H[(size_t)i*K+i];
          if (i == 0 || d < mn) mn = d; if (i == 0 || d > mx) mx = d; if (d <= 0.0) nz++; }
      printf("arm2 diag(H): min %.6g max %.6g  ratio %.3g   columns with diag<=0 (ZEROED by gptq): %d\n",
             mn, mx, mx / (mn > 0 ? mn : 1e-300), nz); }
    int rc2 = ork_i4_gptq(K, N, Ws, H, 0, codes, scal, 0.01f);
    const double e2 = rc2 ? -1.0 : prod_err(M, N, K, As, Ws, codes, scal);

    /* ---- ARM 3: SmoothQuant + ROTATION + GPTQ — the production path exactly ---------------------
     * Production derives W' = rot(s*W) and feeds it a' = rot(a/s), accumulating H from rot(a/s). The
     * rotation is orthogonal and block-diagonal, so it cancels in the product and arm 3 must land near
     * arm 2. If it does not, the interaction of smoothing with the ROTATION is the bug — which is the
     * one thing arms 1-2 do not exercise and the production path does. */
    {
        const int b = K & (-K);                     /* largest power-of-2 block dividing K */
        float *Wr = malloc((size_t)N*K*sizeof *Wr);
        float *Ar = malloc((size_t)M*K*sizeof *Ar);
        for (int n = 0; n < N; n++) { memcpy(Wr + (size_t)n*K, Ws + (size_t)n*K, (size_t)K*sizeof(float)); rot(Wr + (size_t)n*K, K, b); }
        for (int m = 0; m < M; m++) { memcpy(Ar + (size_t)m*K, As + (size_t)m*K, (size_t)K*sizeof(float)); rot(Ar + (size_t)m*K, K, b); }
        hess(M, K, Ar, H);
        int rc3 = ork_i4_gptq(K, N, Wr, H, 0, codes, scal, 0.01f);
        const double e3 = rc3 ? -1.0 : prod_err(M, N, K, Ar, Wr, codes, scal);
        printf("  arm3  Smooth + ROTATION + GPTQ  rc=%d  err %.6f   (b=%d)\n", rc3, e3, b);
        free(Wr); free(Ar);
    }

    printf("\nrelative product error vs exact W*a:\n");
    printf("  arm1  GPTQ plain          rc=%d  err %.6f\n", rc1, e1);
    printf("  arm2  SmoothQuant + GPTQ  rc=%d  err %.6f\n", rc2, e2);
    if (e1 > 0 && e2 > 0)
        printf("\n  ratio arm2/arm1 = %.3f  -> %s\n", e2/e1,
               e2 <= e1*1.5 ? "COMPOSES (smoothing does not break GPTQ's numerics)"
                            : "GPTQ NUMERICS BREAK under smoothing — the end-to-end failure is HERE");
    return 0;
}
