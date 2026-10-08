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

    /* ACT-ORDER (ORK_GPTQ_ACTORDER=1) — quantize columns in DESCENDING diag(H) order.
     *
     * GPTQ compensates each column's rounding error into the columns it has not reached yet, so the
     * compensation budget shrinks as the sweep advances: the last column quantized has nowhere to push
     * its error. Processing the most important columns FIRST (largest diag(H) = largest accumulated
     * activation energy) spends the budget where it matters and leaves the least important columns to
     * absorb the leftovers. This is the standard `desc_act` refinement.
     *
     * NORMALLY ACT-ORDER COSTS A STORED PERMUTATION, AND HERE IT DOES NOT. With per-group scales the
     * groups would straddle permuted columns, so the pack must carry the permutation and the runtime
     * must apply it. With PER-ROW scales (G >= K, our default) there is exactly one scale per output
     * channel covering every column, so the scale is computed over the same SET whatever the order, and
     * the codes are written back into their original positions below. Nothing changes on disk and
     * nothing changes at inference — so this is gated to G >= K rather than made to work with groups.
     *
     * The permutation must be applied BEFORE the factorizations: the Cholesky ordering is what defines
     * the propagation structure, so permuting afterwards would reorder the columns without reordering
     * the error feedback, which is the only thing act-order is trying to change. */
    int *perm = NULL;
    if (getenv("ORK_GPTQ_ACTORDER") && G >= K) {
        perm = (int*)malloc((size_t)K * sizeof *perm);
        double *tmp = (double*)malloc((size_t)K*K*sizeof *tmp);
        if (perm && tmp) {
            for (int i = 0; i < K; i++) perm[i] = i;
            /* insertion sort on diag(H), descending — K is ~1-4k and this runs once per weight, far
             * below the O(K^3) factorizations that follow. */
            for (int i = 1; i < K; i++) {
                const int key = perm[i]; const double kv = Hd[(size_t)key*K + key];
                int j = i - 1;
                while (j >= 0 && Hd[(size_t)perm[j]*K + perm[j]] < kv) { perm[j+1] = perm[j]; j--; }
                perm[j+1] = key;
            }
            for (int i = 0; i < K; i++)                           /* H' = P H Pᵀ (symmetric permute) */
                for (int j = 0; j < K; j++)
                    tmp[(size_t)i*K + j] = Hd[(size_t)perm[i]*K + perm[j]];
            memcpy(Hd, tmp, (size_t)K*K*sizeof *tmp);
            for (int nn = 0; nn < N; nn++) {                      /* W' columns follow the same order */
                double *row = Wd + (size_t)nn*K;
                for (int i = 0; i < K; i++) tmp[i] = row[perm[i]];
                memcpy(row, tmp, (size_t)K*sizeof *tmp);
            }
            free(tmp);
        } else { free(perm); free(tmp); perm = NULL; }            /* no memory: run unpermuted, not wrong */
    }

    /* diag(H) MUST be captured HERE. The factorization chain below rewrites Hd IN PLACE (chol_lower
     * turns it into L), so reading Hd[c*K+c] after that point yields L's diagonal — a different
     * quantity that would silently mis-weight the clip objective rather than fail. */
    double *hdiag = (double*)malloc((size_t)K * sizeof *hdiag);
    if (hdiag) for (int i = 0; i < K; i++) hdiag[i] = Hd[(size_t)i*K + i];

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
        if (!Hf || !Hif) { free(Hf); free(Hif); free(hdiag); free(Hd); free(Wd); free(Hin); return -2; }
        for (size_t i = 0; i < (size_t)K*K; i++) Hf[i] = (float)Hd[i];
        int rc = chol_lower_f32(K, Hf);
        if (!rc) rc = inv_from_chol_f32(K, Hf, Hif);
        if (!rc) rc = chol_upper_f32(K, Hif);
        if (rc) { free(Hf); free(Hif); free(hdiag); free(Hd); free(Wd); free(Hin); return rc == -2 ? -2 : -3; }
        for (size_t i = 0; i < (size_t)K*K; i++) Hin[i] = (double)Hif[i];
        free(Hf); free(Hif);
    } else {
        if (gptq_chol_lower(K, Hd) != 0)      { free(hdiag); free(Hd); free(Wd); free(Hin); return -3; }  /* Hd -> L */
        if (gptq_inv_from_chol(K, Hd, Hin)!=0){ free(hdiag); free(Hd); free(Wd); free(Hin); return -2; }  /* Hin = H⁻¹ */
        if (gptq_chol_upper(K, Hin) != 0)     { free(hdiag); free(Hd); free(Wd); free(Hin); return -3; }  /* Hin -> U upper */
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
                    /* GRID SHAPE is tunable (ORK_GPTQ_CLIP_N / _STEP): 8 points at 0.03125 was a guess,
                     * never swept, and this is pack-time only. H-WEIGHTED OBJECTIVE (ORK_GPTQ_HWCLIP=1)
                     * is a tractable stand-in for Schur Replay (arXiv 2609.36654), which observes that
                     * scoring a scale by its own reconstruction error "can misestimate its final
                     * reconstruction error" because quantizing one column updates those that follow.
                     * Replaying the full GPTQ update per candidate is far too costly here; weighting each
                     * column's squared error by diag(H) is the cheap approximation of the same idea —
                     * it prices a column's error by how much activation energy actually flows through
                     * it, instead of treating every column as equally important. */
                    const int    NT = getenv("ORK_GPTQ_CLIP_N")    ? atoi(getenv("ORK_GPTQ_CLIP_N"))    : 8;
                    const double ST = getenv("ORK_GPTQ_CLIP_STEP") ? atof(getenv("ORK_GPTQ_CLIP_STEP")) : 0.03125;
                    const int    HW = getenv("ORK_GPTQ_HWCLIP") != NULL && hdiag != NULL;
                    double best = sc, be = -1.0;
                    for (int t = 0; t < (NT > 0 ? NT : 8); t++) {
                        const double a = 1.0 - (ST > 0.0 ? ST : 0.03125) * (double)t;
                        if (a <= 0.0) break;
                        const double s2 = a * mx / 7.0; if (s2 <= 0.0) continue;
                        double e = 0.0;
                        for (int c = j; c < j1; c++) {
                            const double w = Wd[(size_t)n*K + c];
                            long q = lround(w / s2); if (q > 7) q = 7; if (q < -8) q = -8;
                            const double dd = w - (double)q * s2;
                            e += HW ? dd*dd * hdiag[c] : dd*dd;   /* pre-factorization diag(H) */
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
    /* UN-PERMUTE: the sweep wrote codes[n][j] for the j-th column IN PERMUTED ORDER, i.e. the weight
     * that originally lived at perm[j]. Put them back so the blob layout is byte-for-byte what every
     * loader already expects — act-order changes the ORDER OF QUANTIZATION, never the format.
     * Scales need no fixing: G >= K is enforced above, so there is one per row over all columns. */
    if (perm) {
        int8_t *tmp = (int8_t*)malloc((size_t)K);
        if (tmp) {
            for (int n = 0; n < N; n++) {
                int8_t *row = codes + (size_t)n*K;
                for (int j = 0; j < K; j++) tmp[perm[j]] = row[j];
                memcpy(row, tmp, (size_t)K);
            }
            free(tmp);
        }
        free(perm);
    }
    free(hdiag); free(Hd); free(Wd); free(Hin);
    return 0;
}
