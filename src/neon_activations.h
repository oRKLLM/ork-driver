/* src/neon_activations.h — CPU/NEON vectorized transformer activation + normalization kernels.
 *
 * Phase 1B locked all non-matmul ops (RMSNorm, SwiGLU/SiLU, softmax) to the CPU. These are the
 * optimized ARMv8 NEON implementations that replace the scalar expf()-based fallback loops the
 * examples used. Each has a scalar reference path for non-NEON builds (and for validation).
 *
 * The kernels operate in place on the page-aligned buffers the NPU matmul core writes, to avoid
 * a second pass over LPDDR (read-modify-write the same cache lines the NPU just produced).
 */
#ifndef ORK_NEON_ACTIVATIONS_H
#define ORK_NEON_ACTIVATIONS_H

/* RMSNorm: o[i] = x[i] * w[i] / sqrt(mean(x^2) + eps). Safe in place (o may alias x). */
void ork_rmsnorm_f32(float *o, const float *x, const float *w, int n, float eps);

/* L2Norm: o[i] = x[i] / sqrt(sum(x^2) + eps) (no mean, no weight) — the q/k normalization used by the
 * Gated-Delta-Net path (GGML_OP_L2_NORM). Safe in place. */
void ork_l2norm_f32(float *o, const float *x, int n, float eps);

/* SwiGLU gate: g[i] = silu(g[i]) * u[i], where silu(x) = x * sigmoid(x). In place on g. */
void ork_silu_mul_f32(float *g, const float *u, int n);

/* SwiGLU gate, out-of-place: out[i] = silu(g[i]) * u[i] (g and u unchanged). */
void ork_silu_mul_to_f32(float *out, const float *g, const float *u, int n);

/* Plain SiLU in place: x[i] = silu(x[i]). */
void ork_silu_f32(float *x, int n);

/* Numerically-stable softmax over x[0..n) in place. */
void ork_softmax_f32(float *x, int n);

/* Scalar references (libm expf) — exposed for the validation harness to diff against. */
void ork_rmsnorm_f32_ref(float *o, const float *x, const float *w, int n, float eps);
void ork_l2norm_f32_ref(float *o, const float *x, int n, float eps);
void ork_silu_mul_f32_ref(float *g, const float *u, int n);

#endif /* ORK_NEON_ACTIVATIONS_H */
