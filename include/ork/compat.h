/* ork/compat.h — DEPRECATED old spellings, kept so the ggml-ork fork keeps building across the
 * dtype-first rename (MODULARIZE_PLAN.md / docs/NAMING_MIGRATION.md).
 *
 * These are the ONLY pre-rename names still accepted, and only the ones ggml-ork.cpp actually calls.
 * They are exact inline forwards — no behaviour of their own. New code must not use them.
 *
 * REMOVAL CONDITION: delete this header once every fork branch that references the old names has been
 * rebased onto the new ones. Check with:
 *   git -C <llama.cpp> grep -l "ork_mm_run\b" $(git for-each-ref --format=%(refname:short) refs/heads)
 * Until then check 11 exempts them via tools/naming_exempt.txt. */
#ifndef ORK_COMPAT_H
#define ORK_COMPAT_H

static inline int ork_mm_build_f16_silu_lut(ork_npu *ctx, double Gmax, short *lut, double *S_out, double *R_out, double *out_scale_out) { return ork_f16_mm_build_silu_lut(ctx, Gmax, lut, S_out, R_out, out_scale_out); }
static inline ork_w       * ork_mm_f16_scratch(ork_npu *ctx, int K, int N) { return ork_f16_mm_scratch(ctx, K, N); }
static inline ork_w       * ork_mm_import_i8(ork_npu *ctx, int K, int N, const void *blob, size_t n, size_t bf_off) { return ork_i8_mm_import(ctx, K, N, blob, n, bf_off); }
static inline int ork_mm_inflate_i8_to_f16(ork_npu *ctx, ork_w *w, const int8_t *i8, const float *bscale, int K, int N) { return ork_i8_mm_inflate_to_f16(ctx, w, i8, bscale, K, N); }
static inline ork_w       * ork_mm_load_i4(ork_npu *ctx, int K, int N, const void *blob, size_t n) { return ork_i4_mm_load(ctx, K, N, blob, n); }
static inline ork_w       * ork_mm_load_i4_import(ork_npu *ctx, int K, int N, const void *blob, size_t n) { return ork_i4_mm_load_import(ctx, K, N, blob, n); }
static inline ork_w       * ork_mm_load_i4a8(ork_npu *ctx, int K, int N, const void *blob, size_t n) { return ork_i4a8_mm_load(ctx, K, N, blob, n); }
static inline ork_w       * ork_mm_load_i4a8_import(ork_npu *ctx, int K, int N, const void *blob, size_t n) { return ork_i4a8_mm_load_import(ctx, K, N, blob, n); }
static inline ork_w       * ork_mm_load_i8(ork_npu *ctx, int K, int N, const void *blob, size_t n) { return ork_i8_mm_load(ctx, K, N, blob, n); }
static inline ork_w       * ork_mm_load_i8_import(ork_npu *ctx, int K, int N, const void *blob, size_t n) { return ork_i8_mm_load_import(ctx, K, N, blob, n); }
static inline ork_w       * ork_mm_pack(ork_npu *ctx, int K, int N, const ork_f16 *B) { return ork_f16_mm_pack(ctx, K, N, B); }
static inline ork_w       * ork_mm_pack_i4(ork_npu *ctx, int K, int N, const int8_t *B) { return ork_i4_mm_pack(ctx, K, N, B); }
static inline ork_w       * ork_mm_pack_i4_grouped(ork_npu *ctx, int K, int N, const int8_t *B, int G) { return ork_i4_mm_pack_grouped(ctx, K, N, B, G); }
static inline ork_w       * ork_mm_pack_i4a8(ork_npu *ctx, int K, int N, const float *f32, float *bscale_out) { return ork_i4a8_mm_pack(ctx, K, N, f32, bscale_out); }
static inline ork_w       * ork_mm_pack_i4a8_im(ork_npu *ctx, int K, int N, const float *f32, const float *imatrix, float *bscale_out) { return ork_i4a8_mm_pack_im(ctx, K, N, f32, imatrix, bscale_out); }
static inline ork_w       * ork_mm_pack_i8(ork_npu *ctx, int K, int N, const int8_t *B) { return ork_i8_mm_pack(ctx, K, N, B); }
static inline ork_w       * ork_mm_pack_i8_dequant(ork_npu *ctx, int K, int N, ork_dequant_row_fn dequant, void *dctx, float *bscale_out) { return ork_i8_mm_pack_dequant(ctx, K, N, dequant, dctx, bscale_out); }
static inline ork_w       * ork_mm_pack_i8_import(ork_npu *ctx, int K, int N, const int8_t *B) { return ork_i8_mm_pack_import(ctx, K, N, B); }
static inline int ork_mm_repack_f16(ork_npu *ctx, ork_w *w, int K, int N, const ork_f16 *B) { return ork_f16_mm_repack(ctx, w, K, N, B); }
static inline int ork_mm_repack_i8(ork_npu *ctx, ork_w *w, int K, int N, const int8_t *B) { return ork_i8_mm_repack(ctx, w, K, N, B); }
static inline int ork_mm_run(ork_npu *ctx, ork_w *w, int M, const ork_f16 *A, float *C) { return ork_f16_mm_run(ctx, w, M, A, C); }
static inline int ork_mm_run_chain_i4(ork_npu *ctx, int S, const ork_mm_task_i4 *tasks) { return ork_i4_mm_run_chain(ctx, S, tasks); }
static inline int ork_mm_run_chain_i8(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks) { return ork_i8_mm_run_chain(ctx, S, tasks); }
static inline int ork_mm_run_f16_silu(ork_npu *ctx, ork_w *w, int M, const ork_f16 *A, float *C, unsigned out_bias, unsigned idx_off, unsigned cfg4068, const short *lut, int nlut) { return ork_f16_mm_run_silu(ctx, w, M, A, C, out_bias, idx_off, cfg4068, lut, nlut); }
static inline int ork_mm_run_i4(ork_npu *ctx, ork_w *w, int M, const int8_t *A, int32_t *C) { return ork_i4_mm_run(ctx, w, M, A, C); }
static inline int ork_mm_run_i4_grouped(ork_npu *ctx, ork_w *w, int M, const int8_t *A, const float *aScale, const float *bScale, float *C) { return ork_i4_mm_run_grouped(ctx, w, M, A, aScale, bScale, C); }
static inline int ork_mm_run_i8(ork_npu *ctx, ork_w *w, int M, const int8_t *A, int32_t *C) { return ork_i8_mm_run(ctx, w, M, A, C); }
static inline int ork_mm_run_i8_out16(ork_npu *ctx, ork_w *w, int M, const int8_t *A, short *C, int mult, int shift) { return ork_i8_mm_run_out16(ctx, w, M, A, C, mult, shift); }
static inline int ork_mm_run_i8_out8(ork_npu *ctx, ork_w *w, int M, const int8_t *A, int8_t *C, int mult, int shift) { return ork_i8_mm_run_out8(ctx, w, M, A, C, mult, shift); }
static inline int ork_mm_run_i8_silu(ork_npu *ctx, ork_w *w, int M, const int8_t *A, int8_t *C, int r_mult, int r_shift, unsigned out_bias, unsigned idx_off, unsigned cfg4068, const short *lut, int nlut) { return ork_i8_mm_run_silu(ctx, w, M, A, C, r_mult, r_shift, out_bias, idx_off, cfg4068, lut, nlut); }
static inline int ork_mm_run_stream_f16(ork_npu *c, int S, const ork_mm_task_f16 *tasks) { return ork_f16_mm_run_stream(c, S, tasks); }
static inline int ork_mm_run_stream_f16_chain(ork_npu *c, int S, const ork_mm_task_f16 *tasks) { return ork_f16_mm_run_stream_chain(c, S, tasks); }
static inline int ork_mm_run_stream_i8(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks) { return ork_i8_mm_run_stream(ctx, S, tasks); }
static inline int ork_npu_add_f16(ork_npu *ctx, const ork_f16 *a, const ork_f16 *b, int M, int N, ork_f16 *out, double *us) { return ork_f16_npu_add(ctx, a, b, M, N, out, us); }
static inline int ork_npu_ewmul_f16(ork_npu *ctx, const ork_f16 *up, const ork_f16 *silu, int M, int N, ork_f16 *out, double *us) { return ork_f16_npu_ewmul(ctx, up, silu, M, N, out, us); }
static inline int ork_npu_ewmul_i8(ork_npu *ctx, const int8_t *up, const int8_t *silu, int M, int N, int mult, int shift, int8_t *out, double *us) { return ork_i8_npu_ewmul(ctx, up, silu, M, N, mult, shift, out, us); }
static inline int ork_npu_exp_i16(ork_npu *ctx, const short *in, int M, int N, double in_scale, double out_scale, short *out, double *us) { return ork_i16_npu_exp(ctx, in, M, N, in_scale, out_scale, out, us); }
static inline int ork_npu_gelu_i8(ork_npu *ctx, const signed char *in, int M, int N, double in_scale, double out_scale, signed char *out, double *us) { return ork_i8_npu_gelu(ctx, in, M, N, in_scale, out_scale, out, us); }
static inline int ork_npu_mul_perchan_f16(ork_npu *ctx, const ork_f16 *a, const ork_f16 *b, int M, int N, ork_f16 *out, double *us) { return ork_f16_npu_mul_perchan(ctx, a, b, M, N, out, us); }
static inline int ork_npu_rmsnorm_f16(ork_npu *ctx, int M, int n, const ork_f16 *x, const ork_f16 *w, float eps, ork_f16 *out) { return ork_f16_npu_rmsnorm(ctx, M, n, x, w, eps, out); }
static inline int ork_npu_rope_neox_f16(ork_npu *ctx, const ork_f16 *x, int hd, int nrow, const int *pos, double freq_base, ork_f16 *out) { return ork_f16_npu_rope_neox(ctx, x, hd, nrow, pos, freq_base, out); }
static inline int ork_npu_row_max_i8(ork_npu *ctx, const signed char *a, int M, int N, signed char *out, double *us) { return ork_i8_npu_row_max(ctx, a, M, N, out, us); }
static inline int ork_npu_silu_i16(ork_npu *ctx, const short *in, int M, int N, double in_scale, double out_scale, short *out, double *us) { return ork_i16_npu_silu(ctx, in, M, N, in_scale, out_scale, out, us); }
static inline int ork_npu_silu_i8(ork_npu *ctx, const signed char *in, int M, int N, double in_scale, double out_scale, signed char *out, double *us) { return ork_i8_npu_silu(ctx, in, M, N, in_scale, out_scale, out, us); }
static inline int ork_npu_softmax_f16(ork_npu *ctx, int M, int n, const ork_f16 *x, ork_f16 *out) { return ork_f16_npu_softmax(ctx, M, n, x, out); }
static inline size_t ork_pack_i4a8_cpu_blob(ork_npu *ctx, int K, int N, const float *f32, const float *imatrix, int nf4, void *out, size_t cap) { return ork_i4a8_pack_cpu_blob(ctx, K, N, f32, imatrix, nf4, out, cap); }
static inline ork_stream_entry * ork_stream_pool_add_i8(ork_stream_pool *p, int K, int N, const void *blob, size_t n) { return ork_i8_stream_pool_add(p, K, N, blob, n); }
static inline size_t ork_w_dump_bf_i8_cpu(ork_npu *ctx, int K, int N, const int8_t *B, void *out, size_t cap) { return ork_i8_w_dump_bf_cpu(ctx, K, N, B, out, cap); }
static inline size_t ork_w_dump_i4a8(const ork_w *w, void *out, size_t cap) { return ork_i4a8_w_dump(w, out, cap); }
static inline size_t ork_w_dump_i8_cpu(ork_npu *ctx, int K, int N, const int8_t *B, void *out, size_t cap) { return ork_i8_w_dump_cpu(ctx, K, N, B, out, cap); }
static inline size_t ork_w_dump_i8_cpu_st(ork_npu *ctx, int K, int N, const int8_t *B, void *out, size_t cap) { return ork_i8_w_dump_cpu_st(ctx, K, N, B, out, cap); }

#endif /* ORK_COMPAT_H */
