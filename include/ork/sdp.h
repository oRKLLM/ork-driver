/* ork/sdp.h — SDP activation ops, elementwise, norms — the production surface
 *
 * The on-NPU output-stage (SDP) ops the library and its consumers actually call: elementwise
 * mul/add, SiLU/GELU/exp/rsqrt, per-channel multiply, row-max, RMSNorm, RoPE, softmax, the
 * activation-LUT builders, and the generic PC-chain runner. Every declaration here has a real
 * caller in the library, the examples, or ggml-ork.
 *
 * Part of the public API. Do NOT include this directly — include <ork_npu.h>, which is the
 * umbrella and the only supported entry point. Types live in ork_npu.h above the includes
 * (and ork/sdp.h, included first), so this header is not self-contained by design. */
#ifndef ORK_SDP_H
#define ORK_SDP_H

/* Profiling: read accumulated run_multicore phase times (us) and call count since process start.
 * setup = pre-dispatch checks + mc_ensure + cres memset; submit = pool dispatch + workers + NPU;
 * copy = cres->C memcpy. Any pointer may be NULL. Pins integration overhead vs the NPU itself. */
void         ork_npu_run_timing(double *setup, double *submit, double *copy, long *n);
void         ork_npu_mc_timing(int core, double *copy, double *submit, double *acc, long *n);

/* PPU fused-output stage (step 1): run ONE full-K int8 matmul at (M,K,N) with the int8-REQUANTIZED
 * output stage instead of int32. out_i8 = clamp_i8((acc_i32 * mult) >> shift); identity = (0x4000,14).
 * Isolated bit-exact test bed for the on-NPU fused path (SiLU/EW-mul build on this int8-output stage).
 * A[M*K], B[K*N] row-major int8; C[M*N] int8 out; us = warm-submit time. 0/ok, -1 wedged, -2 bad dims.
 * See ork_ppu_fuse_enabled(); the CPU/NEON requant path stays the default fallback. */
int          ork_npu_probe_i8_out8(ork_npu *ctx, int M, int K, int N, const int8_t *A, const int8_t *B,
                                   int mult, int shift, int8_t *C, double *us);

/* ork-NATIVE fused-SiLU LUT generator: build ork's OWN silu LUT for the fused-output path (no RKNN
 * dependence). Measures ork's index(acc) for (r_mult,r_shift,cfg4068) via one calibration submit, then
 * builds lut[1030] = silu curve matched to ork's mapping for (in_scale,out_scale). Do this ONCE per
 * register config; run matmuls via ork_npu_probe_i8_silu_cfg(..,r_mult,r_shift,0,0xffffc000,cfg4068,lut,1030,..).
 * Pick R=r_mult/2^r_shift ~= 660*in_scale so the acc range spans silu's transition. Validated ~1 int8.
 * 0/ok, -1 fail. See tools/silu_native.c. */
int          ork_mm_silu_build_lut(ork_npu *ctx, double in_scale, double out_scale,
                                   int r_mult, int r_shift, uint32_t cfg4068, int16_t *lut);
/* Fused-SiLU probe with the decoded output-stage knobs exposed: r_mult/r_shift = the unified scale
 * R = r_mult/2^r_shift (reg 0x4084/0x4088; sets both the acc->LUT-index step and the LUT->output gain);
 * out_bias = reg 0x4080 (output bias / asymmetric zero-point); idx_off = reg 0x4110 (LUT-index offset C0);
 * cfg4068 = reg 0x4068 (per-scale field, no observed output effect). lut != NULL overrides the LUT contents
 * (for the staircase/const-LUT calibration harness); lut==NULL uses the fixed captured silu curve. */
int          ork_npu_probe_i8_silu_cfg(ork_npu *ctx, int M, int K, int N, const int8_t *A, const int8_t *B,
                                       int r_mult, int r_shift, uint32_t out_bias, uint32_t idx_off,
                                       uint32_t cfg4068, const int16_t *lut, int nlut, int8_t *C, double *us);

/* On-NPU element-wise MULTIPLY of two int8 [M][N] tensors (e.g. the SwiGLU inner silu(gate)⊙up):
 * out[m*N+n] = clamp_i8(round(up[m][n]*silu[m][n] * mult/2^shift)) computed on the NPU (standalone SDP op).
 * Handles the NVDLA feature-cube marshaling internally; symmetric int8 (zero-points 0). mult in 0..0x7fff
 * (OUT_CVT_SCALE is signed 16-bit); gain = mult/2^shift (= s_up·s_silu/s_out for SwiGLU). Supported shape:
 * M=8,N=64 (the captured op geometry) — other shapes return -2 pending cube-dim generalization. rk3588-gated
 * (returns -3 elsewhere). 0/ok, -1 wedged, -2 bad shape. Validated bit-exact vs CPU (examples/test_ewmul_i8). */
int          ork_npu_ewmul_i8(ork_npu *ctx, const int8_t *up, const int8_t *silu, int M, int N,
                              int mult, int shift, int8_t *out, double *us);

/* fp16 element-wise MULTIPLY of two [M][N] fp16 tensors on the NPU: out[m][n] = up[m][n]*silu[m][n] (no
 * requant — fp16 standalone SDP op). NVDLA fp16 feature-cube marshaled internally. up/silu/out are fp16
 * bit-patterns. Supported shape M=8,N=64; rk3588-gated. 0/ok, -1 wedged, -2 bad shape, -3 non-rk3588. */
int          ork_npu_ewmul_f16(ork_npu *ctx, const ork_f16 *up, const ork_f16 *silu, int M, int N,
                               ork_f16 *out, double *us);

/* int16 element-wise MULTIPLY of two [M][N] int16 tensors on the NPU: out=clamp_i16(round(up*silu*mult/2^shift)).
 * The w4a4 path's EW precision (ork's int4 matmul outputs int16). NVDLA int16 feature-cube (atom-8) marshaled
 * internally; symmetric zero-points. mult in 0..0x7fff. Shape M=8,N=64; rk3588-gated. 0/ok,-1,-2,-3. */
int          ork_npu_ewmul_i16(ork_npu *ctx, const int16_t *up, const int16_t *silu, int M, int N,
                               int mult, int shift, int16_t *out, double *us);

/* On-NPU normalization primitives (fp16 in/out), per row of an [M][n] tensor:
 *   rmsnorm: out = x / sqrt(mean(x^2)+eps) * w      (w[n] weight)
 *   l2norm:  out = x / sqrt(sum(x^2)+eps)           (no mean, no weight — the GDN q/k normalize)
 * GATED (default OFF → computed on CPU, bit-exact via the fp32 NEON refs, so callers get a stable
 * entrypoint today). A fully on-NPU norm needs the PPU transcendental (rsqrt) whose regcmd capture was
 * blocked by the interposer's 4KB buffer-dump truncation; with that fixed the PPU-native path can be
 * captured and slotted in here. ORK_NORM_NPU=1 selects the NPU path once it exists (today a no-op
 * selector — still CPU-correct). out may alias x. 0/ok, -1 alloc, -2 bad args. */
int          ork_npu_rmsnorm_f16(ork_npu *ctx, int M, int n, const ork_f16 *x, const ork_f16 *w, float eps, ork_f16 *out);

/* NEOX RoPE on the NPU (rope type 2). x[nrow][hd] fp16 row-major (nrow = heads*tokens; each row a head_dim
 * vector); pos[nrow] = per-row token position. Composed as x⊙COS + rot_half(x)⊙SIN (2 ewmul + 1 add on NPU).
 * hd even + multiple of 8. 0/ok, <0 on failure. Keeps Q/K rotation on the NPU data path (attention chaining). */
int          ork_npu_rope_neox_f16(ork_npu *ctx, const ork_f16 *x, int hd, int nrow, const int *pos, double freq_base, ork_f16 *out);
int          ork_npu_l2norm_f16 (ork_npu *ctx, int M, int n, const ork_f16 *x,                     float eps, ork_f16 *out);

/* CHAIN ASSEMBLER: one pre-built program in a heterogeneous PC-chain (see ork_npu_chain_progs).
 * rc/nwords = the program's regcmd words (caller-built, with its own buffer addresses); enable_mask/
 * regcfg_amount = its rknpu_task fields (matmul 0xd/108, SDP 0x18/varies); desc_slot = the word index of
 * this program's PC next-descriptor (where 0x0010/0x0101 next-addr is WRITTEN — matmul=216); -1 if the op
 * carries no descriptor slot (then it can only be the LAST program). */
typedef struct { const uint32_t *rc; int nwords; unsigned enable_mask; int regcfg_amount; int desc_slot; } ork_chain_prog;
/* Submit N pre-built programs as ONE PC-chain (task_number=N, single ioctl) — chains a whole NPU-only
 * run (attention block / FFN inner) into one submit. Non-last programs need a PC next-descriptor slot in
 * their regcmd (matmul has one; a program lacking it can only be last). 0/ok, -2 bad-args/no-slot, -1 wedge. */
int          ork_npu_chain_progs(ork_npu *ctx, int n, const ork_chain_prog *progs, int dom);

/* CHAIN ASSEMBLER increment-1: data-connected int8-matmul(int16-out) -> int16-silu in ONE PC-chain (the
 * matmul's int16 output buffer IS the silu's input). Validates the intermediate-buffer bridge. Computes
 * out = clamp_i16(silu(requant_i16(A[M,K]xB[K,N])*in_scale)/out_scale). gate_out (nullable) = the matmul
 * int16 output read back via EWCUBEH, to localize a mismatch. 0/ok,-1 wedge,-2 dims,-3 SoC. rk3588-gated. */
int          ork_npu_chain_gatesilu_i16(ork_npu *ctx, int M, int K, int N, const signed char *A, const signed char *B,
                                        int mult, int shift, double in_scale, double out_scale,
                                        short *gate_out, short *out, double *us);

/* Standalone on-NPU SiLU (activation-LUT SDP op): applies the PPU silu LUT to a single int8 input [M][N] via
 * the 69-reg/enable=0x18 standalone op (REGCMD_SILU_STD), reprogrammed to (M,N). Two submits (LUT-load + op).
 * SDP: idx=(in*R)>>6 + C0; out=clamp_i8(R*LUT-interp(idx) + out_bias), R=r_mult/2^r_shift. Caller supplies the
 * scale regs + LUT (lut==NULL keeps the captured curve). RE/calibration entry (measure idx(in) via a ramp LUT
 * then build the curve). in/out int8 [M*N], N%16==0; rk3588-gated. 0/ok,-1 wedged,-2 shape,-3 SoC. */
int          ork_npu_probe_silu_std(ork_npu *ctx, const signed char *in, int M, int N,
                                    int r_mult, int r_shift, unsigned out_bias, unsigned idx_off,
                                    unsigned cfg4064, unsigned cfg4068, const short *lut, int nlut,
                                    signed char *out, double *us);

/* fp16 standalone activation-LUT op — RE probe. Applies the PPU LUT to a single fp16 input [M][N] via
 * REGCMD_SILU_STD_F16. Two submits (LUT-load + op). in/out fp16 [M*N], N%8==0. 0/ok,-1,-2,-3. */
int          ork_npu_probe_silu_std_f16(ork_npu *ctx, const ork_f16 *in, int M, int N,
                                        unsigned idx_off, unsigned cfg4064, unsigned cfg4068,
                                        const short *lut, int nlut, ork_f16 *out, double *us);

/* int16 (w16a16i) standalone activation-LUT op — RE probe. Same requant-LUT math as the int8 op but int16
 * I/O (atom-8 cube) via REGCMD_SILU_STD_I16. Two submits (LUT-load + op). in/out int16 [M*N], N%8==0. */
int          ork_npu_probe_silu_std_i16(ork_npu *ctx, const short *in, int M, int N,
                                        int r_mult, int r_shift, unsigned out_bias, unsigned idx_off,
                                        unsigned cfg4064, unsigned cfg4068, const short *lut, int nlut,
                                        short *out, double *us);

/* On-NPU SiLU (int8): out[m*N+n] = clamp_i8(round( silu(in[m][n]*in_scale) / out_scale )) via the standalone
 * SDP activation-LUT op. Calibrates the op's index map once per ctx, builds the silu curve for (in_scale,
 * out_scale), loads it, runs the op. in/out int8 [M*N], N%16==0; rk3588-gated. 0/ok,-1 wedged,-2 shape,-3 SoC. */
int          ork_npu_silu_i8(ork_npu *ctx, const signed char *in, int M, int N,
                             double in_scale, double out_scale, signed char *out, double *us);
/* #38 RE: replay a captured int8 matmul regcmd (rkllm's, from regcmd_capture) on ork's submit path.
 * Adata/Bdata: captured A / tiled-weight bytes (NULL => garbage, timing only). Cout: computed int32 output
 * (for correctness check vs captured C). Patches A/B/C addrs, single-core, times `iters`. 0/ok. */
int          ork_npu_replay_i8(ork_npu *ctx, const unsigned *regcmd, int rn, int M, int K, int N,
                               const signed char *Adata, int Abytes, const signed char *Bdata, int Bbytes, int *Cout, int iters, double *us);
/* #39 weight-resident M-fold chain: P width-w mfold tiles, shared K*N weight, one task_number=P submit. */
int          ork_npu_mfold_chain(ork_npu *ctx, int P, int w, int K, int N,
                                 const signed char *Apacked, const signed char *Bpacked,
                                 int *Craw, int iters, double *us);
/* #39 UNIFIED per-tile fold chain: P tiles of per-tile width ws[t], real operands (concat nc16 A / woff B),
 * Craw out (concat c4), optional WEIGHT_REUSE on tiles t>0. For [state-setter+big-M] and weight-reuse chains. */
int          ork_npu_mfold_chain_v(ork_npu *ctx, int P, const int *ws, int K, int N,
                                    const unsigned *tiles, int rn, const signed char *Apacked,
                                    const signed char *Bpacked, int *Craw, int wreuse, int iters, double *us);
/* #39 FULL-OP fold: C[M,N]=A[M,K]xW[K,N] int8 via N-split across cores (each core single-core-folds a <=1216-wide
 * slice concurrently) — rkllm's wide-projection scheme. K=FOLD_REF_K, N<=3*1216, M<=128. A/Cout row-major. */
int          ork_npu_fold_op_i8(ork_npu *ctx, int K, int N, const signed char *Wraw, int M,
                                const signed char *Araw, int *Cout, int iters, double *us);
/* #39 mfold RESIDENT-weight variant: fold matmul from a PRE-PACKED w->Bfold (ork_mm_load_fold_i8 / orkpack v5),
 * no per-call fold_woff repack. K=FOLD_REF_K, N<=3*1216, M<=128. A/Cout row-major. 0/ok, -1/-2/-3 as fold_op. */
int          ork_npu_fold_run_w(ork_npu *ctx, ork_w *w, int M, const signed char *Araw, int *Cout, int iters, double *us);
/* #39 Path-1 CANONICAL output-stage state-setter: rewrite EVERY output-stage register in a REGCMD_I8_N regcmd to
 * its first-principles value (from the full-prefill invariant scan) — across BOTH output blocks 0x1001 (DPU/SDP)
 * and 0x801 (PDP/aux output-dims mirror) — leaving DST_BASE_ADDR (the only output-stage IOVA) for the caller.
 * Zero a proven fold tile's 0x1001+0x801 blocks, stamp, and the whole output stage is rebuilt from understood
 * values — the loader a delta-encoded big-M tile inherits. surfadd = 0x40c0 SURFACE_ADD (128*M matched small-M).
 * Returns the number of registers stamped, or <0 on bad args. */
int          ork_npu_sdp_stamp(unsigned *rc, int rn, int M, int N, unsigned surfadd);

/* On-NPU SiLU (int16 / w16a16i) — EXPERIMENTAL, not yet bit-exact. Runs the standalone int16 activation-LUT op,
 * but the int16 op's index-gain response to 0x4068 differs from the int8 op's (which is decoded), so the LUT
 * index model is approximate pending an int16-op gain sweep. in/out int16 [M*N], N%8==0. 0/ok,-1,-2,-3. */
int          ork_npu_silu_i16(ork_npu *ctx, const short *in, int M, int N,
                              double in_scale, double out_scale, short *out, double *us);
/* PHASE 0 chained-FFN probe: a heterogeneous 2-task PC-chain (int8 matmul -> int16 silu) in ONE submit,
 * to prove the hardware walks the matmul->pure-SDP transition without a per-op host round-trip. `out` gets
 * silu(in) (verifies the silu task ran); *mm_ran is set if the chained matmul task also ran. 0/ok,-1,-2,-3. */
int          ork_npu_chain_mm_silu_i16(ork_npu *ctx, const short *in, int M, int N,
                              double in_scale, double out_scale, short *out, int *mm_ran, double *us);

/* On-NPU element-wise ADD (int8): out = clamp_i8(round( (a*a_scale + b*b_scale)/out_scale )) via the 2-input SDP
 * ALU=add op. Symmetric quant. Residual add (a_scale==b_scale==out_scale) => out=clamp_i8(a+b), bit-exact.
 * in/out int8 [M*N], N%16==0; rk3588-gated. 0/ok,-1 wedged,-2 shape,-3 SoC. */
/* On-NPU per-row MAX-REDUCE (int8): out[m]=max_n a[m*N+n]. Batched pairwise-max tree on the SDP EW ALU
 * (EW_ALU_ALGO=MAX). N%16, reduces N->16 on-NPU + 16-wide CPU tail. Reusable: softmax max, max-pool, top-k. */
int          ork_npu_row_max_i8(ork_npu *ctx, const signed char *a, int M, int N, signed char *out, double *us);
/* On-NPU PER-CHANNEL scale (int8): out[m][n]=clamp(a[m][n]*b[n]*mult>>(shift-14)); b[N] broadcast across rows
 * (EW operand per-channel mode, ERDMA_DATA_MODE=0). N%16. Reusable: softmax normalize, LayerNorm affine, requant. */
int          ork_npu_mul_perchan_i8(ork_npu *ctx, const signed char *a, const signed char *b, int M, int N, int mult, int shift, signed char *out, double *us);
/* fp16 per-channel scale: out[m][n]=a[m][n]*b[n], b[N] broadcast across rows (ERDMA_DATA_MODE=0). N%8. Quant-free. */
int          ork_npu_mul_perchan_f16(ork_npu *ctx, const ork_f16 *a, const ork_f16 *b, int M, int N, ork_f16 *out, double *us);
/* int16 per-channel scale: out[m][n]=clamp_i16(a[m][n]*b[n]*mult>>shift), b[N] broadcast. N%8. Chain intermediate. */
int          ork_npu_mul_perchan_i16(ork_npu *ctx, const short *a, const short *b, int M, int N, int mult, int shift, short *out, double *us);
int          ork_npu_add_i8(ork_npu *ctx, const signed char *a, const signed char *b, int M, int N,
                            double a_scale, double b_scale, double out_scale, signed char *out, double *us);

/* On-NPU fp16 element-wise ADD (residual): out = a + b in fp16 via the 2-input SDP ALU=add op. in/out fp16
 * [M*N], N%8==0; rk3588-gated. 0/ok,-1,-2,-3. */
int          ork_npu_add_f16(ork_npu *ctx, const ork_f16 *a, const ork_f16 *b, int M, int N, ork_f16 *out, double *us);

/* On-NPU GELU (int8): out = clamp_i8(round( gelu(in*in_scale)/out_scale )) via the same standalone SDP
 * activation-LUT op as SiLU (the LUT holds the GELU curve). Bit-exact-class. N%16==0. 0/ok,-1,-2,-3. */
int          ork_npu_gelu_i8(ork_npu *ctx, const signed char *in, int M, int N,
                             double in_scale, double out_scale, signed char *out, double *us);
/* On-NPU GELU (int16 / w16a16i): same op, GELU curve. RKNN-class accuracy. N%8==0. 0/ok,-1,-2,-3. */
int          ork_npu_gelu_i16(ork_npu *ctx, const short *in, int M, int N,
                              double in_scale, double out_scale, short *out, double *us);

/* On-NPU rsqrt (int8/int16) — RMSNorm building block, via the same activation-LUT op (rsqrt curve, positive
 * domain): out = clamp(round( rsqrt(in*in_scale)/out_scale )). 0/ok,-1,-2,-3. */
int          ork_npu_rsqrt_i8(ork_npu *ctx, const signed char *in, int M, int N, double in_scale, double out_scale, signed char *out, double *us);
int          ork_npu_rsqrt_i16(ork_npu *ctx, const short *in, int M, int N, double in_scale, double out_scale, short *out, double *us);
/* On-NPU exp (int8/int16) — softmax building block, activation-LUT (exp curve): out=clamp(round(exp(in*is)/os)). */
int          ork_npu_exp_i8(ork_npu *ctx, const signed char *in, int M, int N, double in_scale, double out_scale, signed char *out, double *us);
/* exp with a scalar GLOBAL-max subtract baked into the LUT: out = clamp_i8(round( exp((in-max)*in_scale)/out_scale )).
 * Softmax numerator with stable max-subtract but NO per-row op (max = scalar global max in int8-input units). */
int          ork_npu_exp_i8_biased(ork_npu *ctx, const signed char *in, int M, int N, double in_scale, double out_scale, double max, signed char *out, double *us);
int          ork_npu_exp_i16(ork_npu *ctx, const short *in, int M, int N, double in_scale, double out_scale, short *out, double *us);
/* Composed fused softmax over each row of [M][n] (fp16): out = exp(x-max)/Σexp(x-max). Gated ORK_SOFTMAX_NPU:
 * exp (ork_npu_exp_i16) + the Σ reduction (reduce-matmul) run on the NPU, max + normalize on CPU; falls back
 * to a full CPU softmax when the gate/PPU-fuse is off or n%32!=0. 0/ok, -2 bad args. */
int          ork_npu_softmax_f16(ork_npu *ctx, int M, int n, const ork_f16 *x, ork_f16 *out);

/* On-NPU int16 element-wise ADD — EXPERIMENTAL, not bit-exact over the signed range: the SRDMA/X1 operand halves
 * negative values (int16-specific SDP X1 sign/shift behavior not yet decoded); positive is exact. out =
 * clamp_i16(round((a*a_scale + b*b_scale)/out_scale)). in/out int16 [M*N], N%8==0. 0/ok,-1,-2,-3. */
int          ork_npu_add_i16(ork_npu *ctx, const short *a, const short *b, int M, int N,
                             double a_scale, double b_scale, double out_scale, short *out, double *us);

/* Standalone int8 element-wise ADD — RE probe (settable scale regs). 2-input SDP op with ALU=add. Caller sets
 * out scale mult/shift, b-operand scale bscale, and zero-points za/zb/zo. a/b/out int8 [M*N], N%16==0. 0/ok,-1,-2,-3. */
int          ork_npu_probe_add_i8(ork_npu *ctx, const signed char *a, const signed char *b, int M, int N,
                                  int mult, int shift, unsigned bscale, int za, int zb, int zo, signed char *out, double *us);

/* Runtime gate for the PPU fused-output path. Gated on the SoC detected at startup: returns 1 only on
 * a validated PPU target (currently rk3588). On any other chip, ork-driver emits int32 output and the
 * caller's CPU/NEON requant+activation stage runs — identical numerics. The fused stage is a HW-specific
 * optimization (RE'd against the rk3588 PPU register layout), never a dependency. */
int          ork_ppu_fuse_enabled(ork_npu *ctx);

/* RE/calibration only (Tier 4b): ONE multi-M int4 submit with the M-count regs set to M (synth_i4
 * mc=M). A:(K/32,M,32), B:(N/64,K/32,64,32) native. Copies the RAW int16 output (M*N, NO de-tile) to
 * `raw` so the caller can deduce the multi-M C layout vs a CPU reference. 0/ok, -1 wedged, -2 dims.
 * See tools/i4_multim_probe.c. */
int          ork_npu_probe_i4_mm(ork_npu *ctx, int M, int K, int N,
                                 const signed char *A, const signed char *B, short *raw);
void         ork_i8_fuzz_add(unsigned int blk, unsigned int reg, unsigned int val);
void         ork_f16_fuzz_add(unsigned int blk, unsigned int reg, unsigned int val);
int          ork_npu_probe_f16_mm_f16out(ork_npu *ctx, int M, int K, int N,
                                 const unsigned short *A, const unsigned short *B, unsigned short *out);
#endif /* ORK_SDP_H */
