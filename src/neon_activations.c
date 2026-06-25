/* src/neon_activations.c — ARMv8 NEON activation/normalization kernels. See neon_activations.h. */
#include "neon_activations.h"
#include <math.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

/* Vectorized expf — Cephes minimax (range-reduce x = n*ln2 + r, degree-5 poly for exp(r), scale by 2^n).
 * Max relative error ~1e-6 over [-88,88]; never leaves the vector registers (no libm call). */
static inline float32x4_t vexpq_f32(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-88.3762626647949f)), vdupq_n_f32(88.3762626647949f));
    /* n = round(x / ln2) */
    float32x4_t fx = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504088896341f)));
    /* r = x - n*ln2, ln2 split into hi+lo for precision (Cephes C1/C2) */
    x = vfmsq_f32(x, fx, vdupq_n_f32(0.693359375f));
    x = vfmsq_f32(x, fx, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t z = vmulq_f32(x, x);
    /* degree-5 poly */
    float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
    y = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), y, x);
    y = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), y, x);
    y = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), y, x);
    y = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), y, x);
    y = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), y, x);
    y = vfmaq_f32(x, y, z);                 /* y = y*z + x        */
    y = vaddq_f32(y, vdupq_n_f32(1.0f));    /* + 1.0              */
    /* 2^n via exponent bits */
    int32x4_t pow2n = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(127)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(pow2n));
}

/* sigmoid(x) = 1 / (1 + exp(-x)) */
static inline float32x4_t vsigmoidq_f32(float32x4_t x) {
    float32x4_t e = vexpq_f32(vnegq_f32(x));
    return vdivq_f32(vdupq_n_f32(1.0f), vaddq_f32(vdupq_n_f32(1.0f), e));
}

void ork_rmsnorm_f32(float *o, const float *x, const float *w, int n, float eps) {
    /* sum of squares — 4 independent NEON accumulators (break the dependency chain) + prefetch */
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __builtin_prefetch(x + i + 64, 0, 0);
        float32x4_t v0 = vld1q_f32(x + i),     v1 = vld1q_f32(x + i + 4);
        float32x4_t v2 = vld1q_f32(x + i + 8), v3 = vld1q_f32(x + i + 12);
        a0 = vfmaq_f32(a0, v0, v0); a1 = vfmaq_f32(a1, v1, v1);
        a2 = vfmaq_f32(a2, v2, v2); a3 = vfmaq_f32(a3, v3, v3);
    }
    float ss = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < n; i++) ss += x[i] * x[i];

    float scale = 1.0f / sqrtf(ss / (float) n + eps);
    float32x4_t vs = vdupq_n_f32(scale);
    for (i = 0; i + 4 <= n; i += 4)
        vst1q_f32(o + i, vmulq_f32(vmulq_f32(vld1q_f32(x + i), vs), vld1q_f32(w + i)));
    for (; i < n; i++) o[i] = x[i] * scale * w[i];
}

void ork_silu_mul_f32(float *g, const float *u, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __builtin_prefetch(g + i + 32, 1, 0);
        float32x4_t x = vld1q_f32(g + i);
        float32x4_t silu = vmulq_f32(x, vsigmoidq_f32(x));    /* x * sigmoid(x) */
        vst1q_f32(g + i, vmulq_f32(silu, vld1q_f32(u + i)));  /* * up */
    }
    for (; i < n; i++) { float x = g[i]; g[i] = (x / (1.0f + expf(-x))) * u[i]; }
}

void ork_silu_mul_to_f32(float *out, const float *g, const float *u, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(g + i);
        float32x4_t silu = vmulq_f32(x, vsigmoidq_f32(x));
        vst1q_f32(out + i, vmulq_f32(silu, vld1q_f32(u + i)));
    }
    for (; i < n; i++) { float x = g[i]; out[i] = (x / (1.0f + expf(-x))) * u[i]; }
}

void ork_silu_f32(float *x, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        vst1q_f32(x + i, vmulq_f32(v, vsigmoidq_f32(v)));
    }
    for (; i < n; i++) { float v = x[i]; x[i] = v / (1.0f + expf(-v)); }
}

void ork_softmax_f32(float *x, int n) {
    /* max */
    float32x4_t vm = vdupq_n_f32(-INFINITY); int i = 0;
    for (; i + 4 <= n; i += 4) vm = vmaxq_f32(vm, vld1q_f32(x + i));
    float m = vmaxvq_f32(vm);
    for (; i < n; i++) if (x[i] > m) m = x[i];
    /* exp(x - m) + running sum */
    float32x4_t vmax = vdupq_n_f32(m), vsum = vdupq_n_f32(0);
    for (i = 0; i + 4 <= n; i += 4) {
        float32x4_t e = vexpq_f32(vsubq_f32(vld1q_f32(x + i), vmax));
        vst1q_f32(x + i, e); vsum = vaddq_f32(vsum, e);
    }
    float s = vaddvq_f32(vsum);
    for (; i < n; i++) { float e = expf(x[i] - m); x[i] = e; s += e; }
    /* normalize */
    float32x4_t vinv = vdupq_n_f32(1.0f / s);
    for (i = 0; i + 4 <= n; i += 4) vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), vinv));
    for (; i < n; i++) x[i] /= s;
}

#else  /* portable scalar fallback (non-ARM builds: workstation/CI) */
void ork_rmsnorm_f32(float *o, const float *x, const float *w, int n, float eps) { ork_rmsnorm_f32_ref(o, x, w, n, eps); }
void ork_silu_mul_f32(float *g, const float *u, int n) { ork_silu_mul_f32_ref(g, u, n); }
void ork_silu_mul_to_f32(float *out, const float *g, const float *u, int n) { for (int i = 0; i < n; i++) { float x = g[i]; out[i] = (x / (1.0f + expf(-x))) * u[i]; } }
void ork_silu_f32(float *x, int n) { for (int i = 0; i < n; i++) { float v = x[i]; x[i] = v / (1.0f + expf(-v)); } }
void ork_softmax_f32(float *x, int n) {
    float m = -INFINITY; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}
#endif

/* ---- scalar references (always libm-exact) — the validation/profiling baseline ---- */
void ork_rmsnorm_f32_ref(float *o, const float *x, const float *w, int n, float eps) {
    float s = 0; for (int i = 0; i < n; i++) s += x[i] * x[i];
    s = 1.0f / sqrtf(s / (float) n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * s * w[i];
}
void ork_silu_mul_f32_ref(float *g, const float *u, int n) {
    for (int i = 0; i < n; i++) { float x = g[i]; g[i] = (x / (1.0f + expf(-x))) * u[i]; }
}
