/* ork_gptq.c — GPTQ int4 weight quantization, IN-TREE, NO external deps (hand-rolled dense linear algebra).
 *
 * Error-compensated column-sequential int4 rounding (Frantar, Ashkboos, Hoefler, Alistarh 2022, "GPTQ:
 * Accurate Post-Training Quantization for Generative Pre-trained Transformers"); the algorithm mirrors
 * AutoGPTQ's `GPTQ.quantize` up to fp tolerance. Produces UNIFORM symmetric int4 codes in [-8,7] + a
 * per-(row,group) scale, i.e. the NATIVE-W4A4 form (dequant w = code*scale) — so it composes with the
 * Hadamard rotation (QuaRot = rotate then GPTQ) and feeds ork's native int4 doorbell.
 *
 * STATUS: UNVALIDATED until compared against AutoGPTQ on a fixed (W,H) golden (task #56). Gated at the pack
 * (ORK_GPTQ) — off by default; nothing routes through it until the AutoGPTQ cross-check passes.
 *
 * Cost: O(N*K^2/2) per weight (the error-feedback sweep) — a heavy ONE-TIME pack step, not inference.
 *
 * No NPU, no libc beyond math/stdlib — a pure CPU quantizer callable from the orkpack-build path. */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"

/* --- dense K×K double linear algebra (row-major), self-contained ------------------------------------ */

/* Lower Cholesky in place: A (sym PD) -> L lower, A = L·Lᵀ; upper zeroed. 0 ok, -1 not positive-definite. */
static int gptq_chol_lower(int K, double *A) {
    for (int i = 0; i < K; i++) {
        for (int j = 0; j <= i; j++) {
            double s = A[(size_t)i*K + j];
            for (int k = 0; k < j; k++) s -= A[(size_t)i*K + k] * A[(size_t)j*K + k];
            if (i == j) { if (s <= 0.0) return -1; A[(size_t)i*K + i] = sqrt(s); }
            else          A[(size_t)i*K + j] = s / A[(size_t)j*K + j];
        }
    }
    for (int i = 0; i < K; i++) for (int j = i + 1; j < K; j++) A[(size_t)i*K + j] = 0.0;
    return 0;
}

/* Given lower L (from gptq_chol_lower), out = (L·Lᵀ)⁻¹ = (L⁻¹)ᵀ·(L⁻¹) (symmetric). */
static int gptq_inv_from_chol(int K, const double *L, double *out) {
    double *Li = (double*)calloc((size_t)K*K, sizeof(double));   /* Li = L⁻¹ (lower) */
    if (!Li) return -2;
    for (int c = 0; c < K; c++) {                                /* forward-solve L·Li[:,c] = e_c */
        Li[(size_t)c*K + c] = 1.0 / L[(size_t)c*K + c];
        for (int i = c + 1; i < K; i++) {
            double s = 0.0;
            for (int k = c; k < i; k++) s -= L[(size_t)i*K + k] * Li[(size_t)k*K + c];
            Li[(size_t)i*K + c] = s / L[(size_t)i*K + i];
        }
    }
    for (int i = 0; i < K; i++) for (int j = 0; j < K; j++) {     /* out = Liᵀ·Li ; Li lower -> k >= max(i,j) */
        double s = 0.0;
        for (int k = (i > j ? i : j); k < K; k++) s += Li[(size_t)k*K + i] * Li[(size_t)k*K + j];
        out[(size_t)i*K + j] = s;
    }
    free(Li);
    return 0;
}

/* Upper Cholesky in place: A (sym PD) -> U upper, A = Uᵀ·U; lower zeroed. The GPTQ inverse-Cholesky factor
 * of H⁻¹: U[j][j] is the per-column error scale, U[j][c] (c>=j) the propagation row. 0 ok, -1 not PD. */
static int gptq_chol_upper(int K, double *A) {
    for (int j = 0; j < K; j++) {
        for (int i = 0; i <= j; i++) {
            double s = A[(size_t)i*K + j];
            for (int k = 0; k < i; k++) s -= A[(size_t)k*K + i] * A[(size_t)k*K + j];
            if (i == j) { if (s <= 0.0) return -1; A[(size_t)i*K + j] = sqrt(s); }
            else          A[(size_t)i*K + j] = s / A[(size_t)i*K + i];
        }
    }
    for (int j = 0; j < K; j++) for (int i = j + 1; i < K; i++) A[(size_t)i*K + j] = 0.0;
    return 0;
}

/* GPTQ int4 quant. W:[N(out)×K(in)] fp32 (W[n*K+k]); H:[K×K] calibration Hessian = XᵀX (destroyed);
 * group: int4 group size along K (<=0 => per-row); codes:[N×K] int8 [-8,7]; scales:[N×ceil(K/group)] fp32
 * (dequant w=code*scale); damp: Hessian damping fraction (AutoGPTQ default 0.01). 0 ok, <0 on error. */
int ork_gptq_i4(int K, int N, const float *W, float *H, int group,
                int8_t *codes, float *scales, float damp) {
    if (K <= 0 || N <= 0 || !W || !H || !codes || !scales) return -1;
    const int G  = (group > 0 && group < K) ? group : K;
    const int ng = (K + G - 1) / G;

    double *Hd  = (double*)malloc((size_t)K*K*sizeof(double));
    double *Wd  = (double*)malloc((size_t)N*K*sizeof(double));
    double *Hin = (double*)malloc((size_t)K*K*sizeof(double));
    if (!Hd || !Wd || !Hin) { free(Hd); free(Wd); free(Hin); return -2; }

    double dmean = 0; for (int i = 0; i < K; i++) dmean += (double)H[(size_t)i*K + i];
    dmean /= (double)K;
    double lam = (double)damp * dmean; if (lam <= 0.0) lam = 1e-6;

    for (size_t i = 0; i < (size_t)K*K; i++) Hd[i]  = (double)H[i];
    for (size_t i = 0; i < (size_t)N*K; i++) Wd[i]  = (double)W[i];
    for (int i = 0; i < K; i++) {                                 /* dead input channel -> unit diag, W col 0 */
        if (Hd[(size_t)i*K + i] <= 0.0) {
            for (int j = 0; j < K; j++) { Hd[(size_t)i*K + j] = 0; Hd[(size_t)j*K + i] = 0; }
            Hd[(size_t)i*K + i] = 1.0;
            for (int n = 0; n < N; n++) Wd[(size_t)n*K + i] = 0.0;
        } else Hd[(size_t)i*K + i] += lam;
    }

    if (gptq_chol_lower(K, Hd) != 0)      { free(Hd); free(Wd); free(Hin); return -3; }  /* Hd -> L */
    if (gptq_inv_from_chol(K, Hd, Hin)!=0){ free(Hd); free(Wd); free(Hin); return -2; }  /* Hin = H⁻¹ */
    if (gptq_chol_upper(K, Hin) != 0)     { free(Hd); free(Wd); free(Hin); return -3; }  /* Hin -> U upper */

    for (int j = 0; j < K; j++) {
        const int g = j / G;
        if (j % G == 0) {                                        /* new group: per-row symmetric absmax/7 scale */
            int j1 = j + G; if (j1 > K) j1 = K;
            for (int n = 0; n < N; n++) {
                double mx = 0; for (int c = j; c < j1; c++) { double a = fabs(Wd[(size_t)n*K + c]); if (a > mx) mx = a; }
                double sc = mx / 7.0; if (sc <= 0.0) sc = 1e-12;
                scales[(size_t)n*ng + g] = (float)sc;
            }
        }
        double d = Hin[(size_t)j*K + j]; if (d == 0.0) d = 1e-12;
        for (int n = 0; n < N; n++) {
            const double sc = (double)scales[(size_t)n*ng + g];
            const double w  = Wd[(size_t)n*K + j];
            long q = lround(w / sc); if (q > 7) q = 7; if (q < -8) q = -8;
            codes[(size_t)n*K + j] = (int8_t)q;
            const double err = (w - (double)q * sc) / d;         /* Cholesky-scaled residual */
            double *row = Wd + (size_t)n*K;
            const double *hr = Hin + (size_t)j*K;
            for (int c = j + 1; c < K; c++) row[c] -= err * hr[c]; /* propagate into not-yet-quantized columns */
        }
    }
    free(Hd); free(Wd); free(Hin);
    return 0;
}
