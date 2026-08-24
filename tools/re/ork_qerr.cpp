/* ork_qerr — DIRECT weight-reconstruction error: our int4 quantizer vs ggml's Q4_0, on identical tensors.
 *
 * Why this exists. Against CPU Q4_0 our best W4A4 configuration sits 1.84x higher on excess perplexity, and
 * three hypotheses for the gap have now been tested: scale granularity (explains ~38%), activation width
 * (~12.5%), and rotation-vs-grouping antagonism (refuted -- rotation still helps at G=32). Rather than
 * guess a fourth time, measure the quantizer itself.
 *
 * It reports relative Frobenius error ||W - W_hat|| / ||W|| per tensor for:
 *   ggml Q4_0          - 32-weight blocks, one fp16 scale, absmax/-8
 *   ours per-channel   - one fp32 scale per output channel, absmax/7
 *   ours G=32          - one fp32 scale per (channel, 32-weight K-group), absmax/7
 *   ours G=32 rotated  - FWHT within the block, then G=32
 *
 * The rotated case is measured in the ROTATED basis, which is exact rather than a shortcut: the normalized
 * FWHT is orthogonal, so ||W - R^T Q(RW)|| == ||RW - Q(RW)||.
 *
 * GPTQ is deliberately absent. This isolates the QUANTIZER; GPTQ is error feedback against a calibration
 * distribution and would confound what is being compared. If our G=32 reconstructs at least as well as
 * Q4_0 here, then the perplexity deficit is NOT in the weight quantizer and the search moves to the
 * activation path, the accumulation, or the dequant. That is the whole point of the measurement. */
#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

static void fwht_norm(float * a, int n) {
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += len << 1)
            for (int j = i; j < i + len; j++) {
                const float u = a[j], v = a[j + len];
                a[j] = u + v; a[j + len] = u - v;
            }
    const float s = 1.0f / sqrtf((float) n);
    for (int i = 0; i < n; i++) a[i] *= s;
}

/* per-(channel, group) absmax/7 symmetric int4, reconstructed in place. G<=0 => whole channel. */
static double ours(const float * W, int K, int N, int G, bool rot, int b) {
    std::vector<float> col(K);
    double num = 0.0, den = 0.0;
    for (int n = 0; n < N; n++) {
        memcpy(col.data(), W + (size_t) n * K, (size_t) K * sizeof(float));
        if (rot) for (int off = 0; off < K; off += b) fwht_norm(col.data() + off, b);
        const int g = (G > 0) ? G : K;
        for (int k0 = 0; k0 < K; k0 += g) {
            const int k1 = (k0 + g < K) ? k0 + g : K;
            float mx = 1e-9f;
            for (int k = k0; k < k1; k++) { const float v = fabsf(col[k]); if (v > mx) mx = v; }
            const float s = mx / 7.0f, inv = 1.0f / s;
            for (int k = k0; k < k1; k++) {
                int q = (int) lrintf(col[k] * inv); q = q > 7 ? 7 : (q < -8 ? -8 : q);
                const double d = (double) col[k] - (double) q * s;
                num += d * d; den += (double) col[k] * (double) col[k];
            }
        }
    }
    return (den > 0.0) ? sqrt(num / den) : 0.0;
}


/* ---- MATMUL-LEVEL error -------------------------------------------------------------------------------
 * Weight reconstruction error is NOT comparable across schemes at the matmul level, because the dot product
 * averages independent per-element errors: our elementwise 0.097 becomes 0.027 over K, and Q4_0's 0.0887
 * would shrink by a similar factor. Comparing one scheme's matmul error against another's WEIGHT error is
 * the mistake this section exists to stop.
 *
 * So: same activations, same reference, both schemes end to end.
 *   Q4_0  = Q4_0 weights x Q8_0 activations (what ggml actually computes)
 *   ours  = int4/int8 weights x int4/int8 activations, per-(row,group) scales, optional rotation
 * Reference is fp64 over the ORIGINAL f32 weights and activations. Reconstructed values are used directly
 * rather than simulating the integer MAC -- with exact scaling the two are identical, and the error is what
 * is being measured.
 *
 * Activations are synthetic and SPIKED (1% of entries x10) rather than plain Gaussian: real activations are
 * heavy-tailed, and a clean Gaussian would understate outliers, which is precisely what rotation exists to
 * handle. A Gaussian-only probe would flatter the unrotated schemes. */
static uint32_t g_rs = 12345u;
static float rnd_norm(void) {
    g_rs ^= g_rs << 13; g_rs ^= g_rs >> 17; g_rs ^= g_rs << 5;
    const float u1 = ((g_rs >> 8) & 0xffffff) / (float) 0x1000000 + 1e-7f;
    g_rs ^= g_rs << 13; g_rs ^= g_rs >> 17; g_rs ^= g_rs << 5;
    const float u2 = ((g_rs >> 8) & 0xffffff) / (float) 0x1000000;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}
/* symmetric per-block quant+dequant in place; qmax 7 => int4, 127 => int8. blk<=0 => whole row. */
static void quant_row(float * v, int K, int blk, int qmax) {
    const int g = (blk > 0) ? blk : K;
    for (int k0 = 0; k0 < K; k0 += g) {
        const int k1 = (k0 + g < K) ? k0 + g : K;
        float mx = 1e-9f;
        for (int k = k0; k < k1; k++) { const float a = fabsf(v[k]); if (a > mx) mx = a; }
        const float s = mx / (float) qmax, inv = 1.0f / s;
        for (int k = k0; k < k1; k++) {
            int q = (int) lrintf(v[k] * inv);
            const int lo = -(qmax + 1) > -128 ? -(qmax + 1) : -128;
            q = q > qmax ? qmax : (q < lo ? lo : q);
            v[k] = (float) q * s;
        }
    }
}

/* mode: 0 = ggml Q4_0 weights + Q8_0 activations; 1 = ours per-channel W4A4; 2 = ours G=32 W4A4;
 *       3 = ours G=32 rotated W4A4; 4 = ours G=32 rotated W4A8 */
static double mm_err(const float * W, int K, int N, int M, int mode, int b) {
    const int NS = N < 256 ? N : 256;
    std::vector<float> A((size_t) M * K), Aq((size_t) M * K), wref(K), wq(K);
    for (size_t i = 0; i < A.size(); i++) {
        float a = rnd_norm();
        if ((i % 97) == 0) a *= 10.0f;                 /* heavy tail: ~1% outliers */
        A[i] = a;
    }
    Aq = A;
    for (int m = 0; m < M; m++) {
        float * r = Aq.data() + (size_t) m * K;
        if (mode == 3 || mode == 4) for (int off = 0; off < K; off += b) fwht_norm(r + off, b);
        if      (mode == 0) quant_row(r, K, 32, 127);  /* ggml pairs Q4_0 weights with Q8_0 activations */
        else if (mode == 4) quant_row(r, K, 32, 127);
        else if (mode == 1) quant_row(r, K,  0, 7);
        else                quant_row(r, K, 32, 7);
    }
    /* Q4_0 weights come from ggml itself, so the comparison uses ITS quantizer, not a reimplementation. */
    std::vector<float> Wq((size_t) K * N);
    if (mode == 0) {
        std::vector<char> q((size_t) ggml_row_size(GGML_TYPE_Q4_0, K) * N);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, W, q.data(), 0, N, K, nullptr);
        const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q4_0);
        for (int n = 0; n < N; n++)
            tt->to_float((const char *) q.data() + (size_t) n * ggml_row_size(GGML_TYPE_Q4_0, K),
                         Wq.data() + (size_t) n * K, K);
    } else {
        for (int n = 0; n < N; n++) {
            float * c = Wq.data() + (size_t) n * K;
            memcpy(c, W + (size_t) n * K, (size_t) K * sizeof(float));
            if (mode == 3 || mode == 4) for (int off = 0; off < K; off += b) fwht_norm(c + off, b);
            quant_row(c, K, (mode == 1) ? 0 : 32, 7);
        }
    }
    double num = 0.0, den = 0.0;
    for (int n = 0; n < NS; n++) {
        const float * wr = W + (size_t) n * K;         /* reference: ORIGINAL weights, unrotated */
        const float * wqp = Wq.data() + (size_t) n * K;
        for (int m = 0; m < M; m++) {
            const float * ar = A.data()  + (size_t) m * K;
            const float * aq = Aq.data() + (size_t) m * K;
            double ref = 0.0, got = 0.0;
            for (int k = 0; k < K; k++) { ref += (double) ar[k] * (double) wr[k];
                                          got += (double) aq[k] * (double) wqp[k]; }
            num += (got - ref) * (got - ref); den += ref * ref;
        }
    }
    return (den > 0.0) ? sqrt(num / den) : 0.0;
}

static double ggml_q4_0(const float * W, int K, int N) {
    const int64_t nel = (int64_t) K * N;
    std::vector<char>  q((size_t) ggml_row_size(GGML_TYPE_Q4_0, K) * N);
    std::vector<float> r((size_t) nel);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, W, q.data(), 0, N, K, nullptr);
    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q4_0);
    for (int n = 0; n < N; n++)
        tt->to_float((const char *) q.data() + (size_t) n * ggml_row_size(GGML_TYPE_Q4_0, K),
                     r.data() + (size_t) n * K, K);
    double num = 0.0, den = 0.0;
    for (int64_t i = 0; i < nel; i++) {
        const double d = (double) W[i] - (double) r[i];
        num += d * d; den += (double) W[i] * (double) W[i];
    }
    return (den > 0.0) ? sqrt(num / den) : 0.0;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [name-substr] [max-tensors]\n", argv[0]); return 2; }
    const char * want = (argc > 2) ? argv[2] : "weight";
    const int    maxt = (argc > 3) ? atoi(argv[3]) : 8;

    ggml_context * mc = nullptr;
    gguf_init_params ip = { /*no_alloc=*/false, /*ctx=*/&mc };
    gguf_context * gg = gguf_init_from_file(argv[1], ip);
    if (!gg || !mc) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    printf("%-34s %6s %6s | %8s %8s %8s %8s\n", "tensor", "K", "N", "Q4_0", "per-ch", "G=32", "G32+rot");
    int done = 0;
    double sQ = 0, sC = 0, sG = 0, sR = 0;
    double mQ = 0, mC = 0, mG = 0, mR = 0, mA = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(mc); t && done < maxt; t = ggml_get_next_tensor(mc, t)) {
        if (ggml_n_dims(t) != 2 || !strstr(t->name, want)) continue;
        const int K = (int) t->ne[0], N = (int) t->ne[1];
        if (K % 32 || K < 64) continue;
        std::vector<float> W((size_t) K * N);
        if (t->type == GGML_TYPE_F32) memcpy(W.data(), t->data, W.size() * sizeof(float));
        else {
            const ggml_type_traits * tt = ggml_get_type_traits(t->type);
            if (!tt->to_float) continue;
            for (int n = 0; n < N; n++)
                tt->to_float((const char *) t->data + (size_t) n * t->nb[1], W.data() + (size_t) n * K, K);
        }
        int b = 1; while ((b << 1) <= K && (K % (b << 1)) == 0) b <<= 1;   /* largest pow2 dividing K */
        const double q = ggml_q4_0(W.data(), K, N);
        const double c = ours(W.data(), K, N, 0,  false, b);
        const double g = ours(W.data(), K, N, 32, false, b);
        const double r = ours(W.data(), K, N, 32, true,  b);
        printf("%-34s %6d %6d | %8.5f %8.5f %8.5f %8.5f\n", t->name, K, N, q, c, g, r);
        mQ += mm_err(W.data(), K, N, 16, 0, b);
        mC += mm_err(W.data(), K, N, 16, 1, b);
        mG += mm_err(W.data(), K, N, 16, 2, b);
        mR += mm_err(W.data(), K, N, 16, 3, b);
        mA += mm_err(W.data(), K, N, 16, 4, b);
        sQ += q; sC += c; sG += g; sR += r; done++;
    }
    if (done) {
        printf("%-34s %6s %6s | %8.5f %8.5f %8.5f %8.5f   (mean of %d)\n",
               "MEAN", "", "", sQ/done, sC/done, sG/done, sR/done, done);
        printf("\nMATMUL-level relative error (same activations, fp64 reference, spiked inputs):\n");
        printf("  %-30s %8.5f\n", "ggml Q4_0 (W4 x A8)",     mQ/done);
        printf("  %-30s %8.5f\n", "ours per-channel (W4A4)", mC/done);
        printf("  %-30s %8.5f\n", "ours G=32 (W4A4)",        mG/done);
        printf("  %-30s %8.5f\n", "ours G=32 rotated (W4A4)",mR/done);
        printf("  %-30s %8.5f\n", "ours G=32 rotated (W4A8)",mA/done);
    }
    gguf_free(gg); ggml_free(mc);
    return done ? 0 : 1;
}
