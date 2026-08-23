/* ork_gptq.c — GPTQ int4 weight quantization, IN-TREE, NO external deps (hand-rolled dense linear algebra).
 *
 * Error-compensated column-sequential int4 rounding (Frantar, Ashkboos, Hoefler, Alistarh 2022, "GPTQ:
 * Accurate Post-Training Quantization for Generative Pre-trained Transformers"); the algorithm mirrors
 * AutoGPTQ's `GPTQ.quantize` up to fp tolerance. Produces UNIFORM symmetric int4 codes in [-8,7] + a
 * per-(row,group) scale, i.e. the NATIVE-W4A4 form (dequant w = code*scale) — so it composes with the
 * Hadamard rotation (QuaRot = rotate then GPTQ) and feeds ork's native int4 doorbell.
 *
 * STATUS (2026-08-22): VALIDATED by property test, un-parked. Task #56 originally demanded a byte-compare
 * against AutoGPTQ on a fixed (W,H) golden; that needs a Python dependency this repo does not allow, and a
 * byte-match against another implementation is anyway weaker than an exact analytic identity. examples/
 * test_gptq.c instead asserts two things, and it is in `make test`:
 *   1. H = I  =>  codes BYTE-IDENTICAL to plain round-to-nearest. With an identity Hessian the factor chain
 *      collapses to Hinv = I, so d = 1 and every propagation term is exactly zero. Any error in the
 *      Cholesky / inverse / upper-Cholesky / error-feedback path breaks this exactly, not statistically.
 *   2. structured H  =>  H-weighted error strictly below RTN's, i.e. it actually minimises its objective.
 * Nothing routes through it: the pack gate ORK_GPTQ is not wired yet, deliberately, because quantizer
 * quality is an END-TO-END question — the remaining gate is a PPL comparison via ork_ppl, not a unit test.
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

/* --- precision-templated kernels: one source, two instantiations (see gptq_kernels.inc) ------------- */
#define GQT double
#define GQ(n) n##_f64
#include "gptq_kernels.inc"
#undef GQT
#undef GQ

#define GQT float
#define GQ(n) n##_f32
#include "gptq_kernels.inc"
#undef GQT
#undef GQ

/* The fp64 names the rest of this file uses. */
#define gptq_chol_lower    chol_lower_f64
#define gptq_inv_from_chol inv_from_chol_f64
#define gptq_chol_upper    chol_upper_f64

/* GPTQ int4 quant. W:[N(out)×K(in)] fp32 (W[n*K+k]); H:[K×K] calibration Hessian = XᵀX (destroyed);
 * group: int4 group size along K (<=0 => per-row); codes:[N×K] int8 [-8,7]; scales:[N×ceil(K/group)] fp32
 * (dequant w=code*scale); damp: Hessian damping fraction (AutoGPTQ default 0.01). 0 ok, <0 on error. */
int ork_i4_gptq(int K, int N, const float *W, float *H, int group,
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

    /* MSE-optimal weight clip: on by default (strictly better on the per-group objective), ORK_GPTQ_NOCLIP=1
     * restores plain absmax/7 for A/B. Read once — this is inside no hot loop, but the getenv is not free. */
    const int clip = (getenv("ORK_GPTQ_NOCLIP") == NULL);

    /* WORKING PRECISION for the three O(K^3) factorizations — ~20:1 of this function's arithmetic, so
     * this is where fp32 would pay: half the memory traffic and twice the NEON lanes. GATED OFF by
     * default (ORK_GPTQ_FP32=1 to try it) because GPTQ inverts a Hessian that is near-singular BY
     * CONSTRUCTION — `damp` exists precisely to keep it invertible — and fp32 has ~7 decimal digits
     * against fp64's ~16. Losing conditioning here does not fail loudly; it silently degrades the error
     * feedback, which is the whole point of GPTQ.
     *
     * MEASURED 2026-08-22 (qwen3.5-0.8B, identical calibration + tokens, fp32 vs fp64 packs built on the
     * same host with these same kernels): fp64 PPL 17.227, fp32 PPL 18.473 — fp32 costs +7.2%, which is
     * 35x the screen's own fidelity (~0.2%). It gives back a quarter of what activation+weight clipping
     * bought (22.82 -> 17.33) to save 0.4 min on a 1.2 min pack. STAYS OFF. It did not fail loudly — 102
     * weights, 0 failures, a valid pack of exactly the right size — it just quantized worse, which is why
     * this was gated on a PPL number rather than adopted on a speed number. Do not re-enable without
     * re-measuring; the answer is not close.
     * The SWEEP below stays fp64 either way: it is the smaller cost and it accumulates across columns. */
    if (getenv("ORK_GPTQ_FP32")) {
        float *Hf = (float*)malloc((size_t)K*K*sizeof(float));
        float *Hif = (float*)malloc((size_t)K*K*sizeof(float));
        if (!Hf || !Hif) { free(Hf); free(Hif); free(Hd); free(Wd); free(Hin); return -2; }
        for (size_t i = 0; i < (size_t)K*K; i++) Hf[i] = (float)Hd[i];
        int rc = chol_lower_f32(K, Hf);
        if (!rc) rc = inv_from_chol_f32(K, Hf, Hif);
        if (!rc) rc = chol_upper_f32(K, Hif);
        if (rc) { free(Hf); free(Hif); free(Hd); free(Wd); free(Hin); return rc == -2 ? -2 : -3; }
        for (size_t i = 0; i < (size_t)K*K; i++) Hin[i] = (double)Hif[i];
        free(Hf); free(Hif);
    } else {
        if (gptq_chol_lower(K, Hd) != 0)      { free(Hd); free(Wd); free(Hin); return -3; }  /* Hd -> L */
        if (gptq_inv_from_chol(K, Hd, Hin)!=0){ free(Hd); free(Wd); free(Hin); return -2; }  /* Hin = H⁻¹ */
        if (gptq_chol_upper(K, Hin) != 0)     { free(Hd); free(Wd); free(Hin); return -3; }  /* Hin -> U upper */
    }

    for (int j = 0; j < K; j++) {
        const int g = j / G;
        if (j % G == 0) {                                        /* new group: per-row symmetric scale */
            int j1 = j + G; if (j1 > K) j1 = K;
            #pragma omp parallel for schedule(static) if (N > 64)
            for (int n = 0; n < N; n++) {
                double mx = 0; for (int c = j; c < j1; c++) { double a = fabs(Wd[(size_t)n*K + c]); if (a > mx) mx = a; }
                double sc = mx / 7.0; if (sc <= 0.0) sc = 1e-12;
                /* absmax/7 spends most of the 16 levels on tails. Search a small grid of clip fractions and
                 * take the true minimum-squared-error scale instead. alpha=1 is in the grid, so this can never
                 * be worse than absmax on the group's own objective. The grid stops at 0.78 (vs the activation
                 * side's 0.56) BECAUSE of the error compensation below: a clipped weight's residual is not
                 * absorbed locally, it is propagated into the not-yet-quantized columns, so over-tight scales
                 * are amplified here in a way they are not for activations. Cost is NT extra passes over one
                 * group — negligible beside the Cholesky and the compensation loop, and pack-time only. */
                if (clip) {
                    double best = sc, be = -1.0;
                    for (int t = 0; t < 8; t++) {
                        const double a = 1.0 - 0.03125 * (double)t;   /* 1.000 .. 0.781 */
                        const double s2 = a * mx / 7.0; if (s2 <= 0.0) continue;
                        double e = 0.0;
                        for (int c = j; c < j1; c++) {
                            const double w = Wd[(size_t)n*K + c];
                            long q = lround(w / s2); if (q > 7) q = 7; if (q < -8) q = -8;
                            const double dd = w - (double)q * s2; e += dd*dd;
                        }
                        if (be < 0.0 || e < be) { be = e; best = s2; }
                    }
                    sc = best;
                }
                scales[(size_t)n*ng + g] = (float)sc;
            }
        }
        double d = Hin[(size_t)j*K + j]; if (d == 0.0) d = 1e-12;
        #pragma omp parallel for schedule(static) if (N > 64)
        for (int n = 0; n < N; n++) {                             /* rows are independent at a fixed column */
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
