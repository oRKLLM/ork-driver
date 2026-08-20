/* ork/run.h — Matmul entrypoints
 *
 * The run surface: plain matmuls per precision, the fused matmul+activation output stages,
 * requant-out variants, and the sliced (K/N-decomposed) rescue path.
 *
 * Part of the public API. Do NOT include this directly — include <ork_npu.h>, which is the
 * umbrella and the only supported entry point; these parts are a readability split of it
 * (ork_npu.h was 1519 lines) and their boundaries may move. Types live in ork_npu.h above
 * the includes, so this header is not self-contained by design. */
#ifndef ORK_RUN_H
#define ORK_RUN_H
/**
 * @brief Run C[M,N] = A[M,K] x packed weights (int8/w8a8). Run dtype must match the pack dtype.
 * @param w Weight from ork_mm_pack_i8().
 * @param M Activation rows; any M>=1 (tiled + scheduled internally, no caller-visible cap).
 * @param A Row-major int8 activations, M*K elements.
 * @param C Output, M*N int32 raw sums, row-major (caller-allocated). Apply the per-tensor/-channel
 *          scales in the caller. May be an ork_dma_alloc() buffer (zero-copy).
 * @return 0 on success, negative on error (bad args / submit failure).
 */
int          ork_mm_run_i8(ork_npu *ctx, ork_w *w, int M, const int8_t  *A, int32_t *C);
/* WEDGE-PRONE REFUSAL. A run entry returns this DISTINCT code (not the generic -1) when it declines a shape
 * it has no VERIFIED path for — rather than attempt a blocking submit that could hard-wedge the NPU. It is a
 * clean, catchable signal: the caller may RESCUE the op by decomposing it onto the doorbell via the sliced
 * primitive (ork_mm_*_sliced), which only ever emits verified c_base tiles. A shape that runs normally never
 * returns this, so a rescue keyed on it costs working shapes nothing. Value chosen far from 0/-1/-2/-3. */
#define ORK_RC_WEDGE_PRONE (-501)
/* SLICE-AND-DICE (ork_slice.h): pack a matmul as a set of c_base doorbell tiles (K-slice + N-tile) so a
 * wide-K / wide-N op runs ENTIRELY on the doorbell — one chained submit, K-slice int32-accumulate + N-tile
 * scatter — bit-exact vs the reference matmul. Pack once, run many. `nc` = doorbell cores (0=all; 1 avoids
 * the non-even-N colsplit issue #36). The foundation for the doorbell owning every submit (SLICE_AND_DICE_PLAN.md).
 *
 * PRECISION-GENERAL SURFACE: the decomposer (tile geometry) is dtype-agnostic; the handle carries its dtype
 * and pack/run dispatch to the per-precision doorbell envelope. `B`/`A` point at the dtype's element type
 * (int8_t for ORK_DT_I8; ork_f16 for ORK_DT_F16); `C` is int32[M,N] for int8, fp32[M,N] for fp16. ONLY
 * ORK_DT_I8 is live today (the q8_0 compute path + the only precision the multi-core doorbell accepts as
 * tiles); ORK_DT_F16 / ORK_DT_I4 pack returns NULL until their doorbell tile path is built (see OPS_REGISTRY). */
typedef enum { ORK_DT_F16 = 0, ORK_DT_I8 = 1, ORK_DT_I4 = 2 } ork_dtype;   /* matches the internal DT_ values */
typedef struct ork_w_sliced ork_w_sliced;
ork_w_sliced *ork_mm_pack_sliced(ork_npu *ctx, int K, int N, const void *B, int dtype);
int           ork_mm_run_sliced (ork_npu *ctx, ork_w_sliced *w, int M, const void *A, void *C, int nc);
void          ork_mm_free_sliced(ork_npu *ctx, ork_w_sliced *w);
/* FUSED SwiGLU gate matmul: C = silu(requant(A·W)) as int8 [M*N], activation applied IN the matmul's
 * SDP output stage (no extra submit / round-trip). Resident full-K int8 weight (K%512==0, K<=4096).
 * R = r_mult/2^r_shift + out_bias is the SCALAR OUT_CVT (so quantize A PER-TENSOR, not per-row);
 * idx_off/cfg4068 = the SiLU index-map params; lut/nlut optional (NULL = fixed silu*S LUT). rk3588
 * only. The productionizable on-NPU-activation path (standalone SDP ops lose ~8x — RE-roadmap M4.6). */
int          ork_mm_run_i8_silu(ork_npu *ctx, ork_w *w, int M, const int8_t *A, int8_t *C,
                                int r_mult, int r_shift, unsigned out_bias, unsigned idx_off,
                                unsigned cfg4068, const short *lut, int nlut);
/* Same as ork_mm_run_i8_silu but INT32 output (silu NOT quantized to int8): out_i32 = R*V16[idx]+out_bias,
 * unclamped. With a fine-scale LUT (out_scale ~ silu_max/8000) this is ~13-14 bit silu, recovering the
 * quality the int8 silu output loses (the FFN-chain gap) while keeping silu free on-NPU. C is int32 [M*N]. */
int          ork_mm_run_i8_silu32(ork_npu *ctx, ork_w *w, int M, const int8_t *A, int *C,
                                  int r_mult, int r_shift, unsigned out_bias, unsigned idx_off,
                                  unsigned cfg4068, const short *lut, int nlut);
/* fp16 gate matmul + fused SiLU with fp16->fp32 output (NO int8 activation quant) — the "end-goal" precise
 * on-NPU gate. Recovers the PPL the int8 silu output loses, but the fp16 matmul is ~3.3x int8 (net-loss today,
 * gated OFF, built for a future int8-win pipeline). w = fp16 weight (ork_mm_pack), A = fp16 [M,K], C = fp32
 * [M,N] silu(gate). K%32, N%16, N<=nmax. 0/ok,-1,-2,-3. WIP: fp16 LUT calibration approximate. */
int          ork_mm_run_f16_silu(ork_npu *ctx, ork_w *w, int M, const ork_f16 *A, float *C,
                                 unsigned out_bias, unsigned idx_off, unsigned cfg4068,
                                 const short *lut, int nlut);
/* GENERIC fp16 fused-output PWL LUT builder (fp16 twin of the int8/int16 act_lut_i8/i16(fn,...)): bakes an
 * arbitrary fn(x) into the SDP output-stage LUT for x in [in_lo,in_hi]. Pack the reduce/gate weight as -S*W
 * (S returned); runtime C_out=R*LUT[idx(-S*x)] and fn(x)=C_out*out_scale. fn(x,ctx) carries params via ctx
 * (NULL for parameter-free fns). Fills lut[1030]. 0/ok, -2 fail. The silu/rsqrt builders below are wrappers. */
int          ork_mm_build_f16_lut(ork_npu *ctx, double (*fn)(double, void *), void *fnctx,
                                  double in_lo, double in_hi, short *lut,
                                  double *S_out, double *R_out, double *out_scale_out);
/* FUSED matmul + output-stage activation: C[M,N] = fn(A·B) in ONE submit — the activation rides the matmul's
 * DPU output stage (no separate submit, no CPU<->NPU crossing). This is the "no-crossing chain" that makes an
 * on-NPU non-matmul op (softmax exp / RMSNorm rsqrt / SwiGLU silu) a WIN instead of a submit-floor-bound loss.
 * fn(x,ctx) is the activation; the matmul output must fall in [in_lo,in_hi] (the LUT band). K%32<=2048,
 * N%16<=nmax. rk3588 PPU-fuse-gated. 0/ok, -2 shape/SoC(PPU off), -1 wedge/alloc. */
int          ork_mm_run_f16_act(ork_npu *ctx, int K, int N, const ork_f16 *B, int M, const ork_f16 *A, float *C,
                                double (*fn)(double, void *), void *fnctx, double in_lo, double in_hi);
/* PACK-ONCE / RUN-MANY split of the fused activation, for RESIDENT reuse (fused exp/rsqrt/silu in a seq): pack
 * bakes the calibrated LUT + out-scale into the weight (ork_mm_pack_f16_fused_act), run replays it with no
 * re-pack (ork_mm_run_f16_fused_act, C[M,N]=fn(A·B), one submit). Same single-signed band rule as _act; the
 * run path carries no fn pointer (crosses a seq/socket). NULL/-2 on bad shape/mixed-sign/no-baked-LUT. */
ork_w       *ork_mm_pack_f16_fused_act(ork_npu *ctx, int K, int N, const ork_f16 *B,
                                       double (*fn)(double, void *), void *fnctx, double in_lo, double in_hi);
int          ork_mm_run_f16_fused_act(ork_npu *ctx, ork_w *w, int M, const ork_f16 *A, float *C);
/* Calibrate the fp16 fused SiLU for a gate spanning [-Gmax,Gmax] (caps Gmax → the fp16 spread band, then
 * ork_mm_build_f16_lut). silu(gate)=C_out*out_scale; pack the gate weight as -S*W. See Exp-2026-07-05. */
int          ork_mm_build_f16_silu_lut(ork_npu *ctx, double Gmax, short *lut,
                                       double *S_out, double *R_out, double *out_scale_out);
/* Calibrate the fp16 fused rsqrt for on-NPU rmsnorm(n_feat=n)/l2norm(n_feat=1): a reduce-matmul packed as
 * -S*W emits scale=1/sqrt(ss/n_feat+eps)=C_out*out_scale for ss=sum(x^2) in [ss_min,ss_max]. Build once per
 * layer/range + cache (runs probe submits). PRECISION VARIANTS: this is the fp16, fused-into-the-reduce
 * rsqrt; the int8/int16 STANDALONE rsqrt of a tensor is ork_npu_rsqrt_i8/i16 (act_lut_i8/i16, same rsqrt_f).
 * Same op, different precision + fusion regime — both kept. */
int          ork_mm_build_f16_rsqrt_lut(ork_npu *ctx, int n_feat, double eps, double ss_min, double ss_max,
                                        short *lut, double *S_out, double *R_out, double *out_scale_out);
/* FUSED SwiGLU up matmul: C = clamp_i8(round( (A·W_up) * G * gain )) as int8 [M*N], the element-wise
 * multiply by G (= silu(gate) from ork_mm_run_i8_silu) applied IN the up matmul's SDP output stage.
 * gain = mult/2^shift = s_up*s_silu/s_out. G is dense int8 [M*N]. Resident full-K int8 weight
 * (K%512==0, K<=4096). Together with ork_mm_run_i8_silu this is the full fused SwiGLU FFN inner. */
int          ork_mm_run_i8_ewmul(ork_npu *ctx, ork_w *w, int M, const int8_t *A, const int8_t *G,
                                 int8_t *C, int mult, int shift);
/* Resident int8 matmul with int8-requantized output: C = clamp_i8(round((A·W)*mult/2^shift)) [M*N].
 * Keeps intermediates int8 on the NPU (the "up" projection feeding a chained FFN-inner EW-mul). */
int          ork_mm_run_i8_out8(ork_npu *ctx, ork_w *w, int M, const int8_t *A, int8_t *C,
                                int mult, int shift);
/* int8 matmul with INT16 requant output (set_i16_out): C=clamp_i16(round((A·W)*mult/2^shift)) [M*N] int16,
 * COMPACT-LINEAR — feeds an int16 SDP seq op resident with no PC-chain/layout bridge. K%512. */
int          ork_mm_run_i8_out16(ork_npu *ctx, ork_w *w, int M, const int8_t *A, short *C, int mult, int shift);
/* int4 (W4A4): A int4 ([-8,7] in int8, row-major), C int32 raw sum — apply scales:
 * C_real[m][n] = aScale[m]*bScale[n]*C[m][n]. Run dtype must match the pack dtype. 0 ok / negative err. */
int          ork_mm_run_i4(ork_npu *ctx, ork_w *w, int M, const int8_t  *A, int32_t *C);

/* Async pipelined submit (orkd client + ring only; ORK_USE_ORKD=1 ORK_ORKD_RING=1). ork_mm_submit enqueues one
 * matmul for w (any precision — dtype taken from w) WITHOUT blocking and returns a ticket; ork_mm_collect(ticket)
 * copies C when it lands. Returns <0 if unavailable (no ring, or the op is too big for a ring slot — use the
 * synchronous ork_mm_run* which auto-falls back to the socket). Keeping several ops in flight (up to the ring
 * depth) overlaps each op's transport with the NPU compute of the ones ahead — the decode-pipeline path.
 * A is int8/int4 (1B/elem) or fp16 (2B/elem); C is int32/fp32 (4B/elem), sized M*N by the caller. */
int          ork_mm_submit(ork_npu *ctx, ork_w *w, int M, const void *A);
int          ork_mm_collect(ork_npu *ctx, int ticket, void *C);
/* grouped int4 (per-group W4A4 dequant): A int4 [M*K] ([-8,7] in int8); aScale [M*(K/G)] (per row,
 * per group), bScale [(K/G)*N] (per group, per channel). C fp32 [M*N] = dequantized result. Pair
 * with ork_mm_pack_i4_grouped. (Cost: K/G submits/core — larger G = fewer submits, coarser scale.) */
int          ork_mm_run_i4_grouped(ork_npu *ctx, ork_w *w, int M, const int8_t *A,
                                   const float *aScale, const float *bScale, float *C);

#endif /* ORK_RUN_H */
