/* ork/ops.h — SDP activation ops, norms, folds — and the RE probes interleaved with them
 *
 * MIXED BY HISTORY, not by design. The production SDP ops (ewmul/add/silu/gelu/exp/rsqrt,
 * per-channel multiply, RMSNorm, RoPE, softmax) are interleaved here with the probe/replay/
 * fuzz surface that was written alongside them while the ops were being reverse-engineered.
 * The split that produced this file is CONTIGUOUS — it preserves the original order exactly —
 * so it could not separate the two. Doing that needs a declaration-level pass; see
 * MODULARIZE_PLAN.md round 3.
 *
 * Part of the public API. Do NOT include this directly — include <ork_npu.h>, which is the
 * umbrella and the only supported entry point; these parts are a readability split of it
 * (ork_npu.h was 1519 lines) and their boundaries may move. Types live in ork_npu.h above
 * the includes, so this header is not self-contained by design. */
#ifndef ORK_OPS_H
#define ORK_OPS_H
/* Profiling: read accumulated run_multicore phase times (us) and call count since process start.
 * setup = pre-dispatch checks + mc_ensure + cres memset; submit = pool dispatch + workers + NPU;
 * copy = cres->C memcpy. Any pointer may be NULL. Pins integration overhead vs the NPU itself. */
void         ork_npu_run_timing(double *setup, double *submit, double *copy, long *n);

/* Profiling (read via ork_npu_mc_timing; the ORK_MCPROF env gate is removed): per-core phase times (us) inside the multi-core prefill (M>1)
 * path — copy (activation host-copy + bsync), submit (regcmd + ioctl + result bsync), acc (host
 * accumulate), and the submit count. Pins why large-M multi-core barely scales. Call _reset before
 * the timed region. core in [0, cores). */
void         ork_npu_mc_reset(void);
void         ork_npu_mc_timing(int core, double *copy, double *submit, double *acc, long *n);
double       ork_npu_mc_synth(int core);   /* host synth+bsync subset of `submit` (overlappable); ioctl/NPU = submit - synth */

/* RE/calibration only: probe this SoC's single-submit K-tile ceiling. Runs ONE M=1 full-K int8
 * submit at (K,N) (N <= SoC N-cap, K%32, N%32) on its own buffers. Returns 0 if the submit
 * completed (C[N] int32 valid — validate vs CPU), -1 if it wedged (K exceeds the per-op K-tile
 * cap; recoverable), -2 on bad dims. See tools/ksubmit_probe.c. */
int          ork_npu_probe_single_i8(ork_npu *ctx, int K, int N, const int8_t *A, const int8_t *B, int32_t *C);

/* PPU fused-output stage (step 1): run ONE full-K int8 matmul at (M,K,N) with the int8-REQUANTIZED
 * output stage instead of int32. out_i8 = clamp_i8((acc_i32 * mult) >> shift); identity = (0x4000,14).
 * Isolated bit-exact test bed for the on-NPU fused path (SiLU/EW-mul build on this int8-output stage).
 * A[M*K], B[K*N] row-major int8; C[M*N] int8 out; us = warm-submit time. 0/ok, -1 wedged, -2 bad dims.
 * See ork_ppu_fuse_enabled(); the CPU/NEON requant path stays the default fallback. */
int          ork_npu_probe_i8_out8(ork_npu *ctx, int M, int K, int N, const int8_t *A, const int8_t *B,
                                   int mult, int shift, int8_t *C, double *us);

/* PHASE 1 (#35 chained FFN): int8 matmul with the INT16-requantized output stage (set_i16_out).
 * out_i16 = clamp_i16(round(acc_i32 * mult / 2^shift)); identity = (0x4000,14). C[M*N] int16 out
 * (raw device layout). The RE crux for the on-NPU matmul->int16-silu handoff. 0/ok, -1 wedged, -2 dims. */
int          ork_npu_probe_i16_out(ork_npu *ctx, int M, int K, int N, const int8_t *A, const int8_t *B,
                                   int mult, int shift, short *C, double *us);

/* ork-NATIVE fused-SiLU LUT generator: build ork's OWN silu LUT for the fused-output path (no RKNN
 * dependence). Measures ork's index(acc) for (r_mult,r_shift,cfg4068) via one calibration submit, then
 * builds lut[1030] = silu curve matched to ork's mapping for (in_scale,out_scale). Do this ONCE per
 * register config; run matmuls via ork_npu_probe_i8_silu_cfg(..,r_mult,r_shift,0,0xffffc000,cfg4068,lut,1030,..).
 * Pick R=r_mult/2^r_shift ~= 660*in_scale so the acc range spans silu's transition. Validated ~1 int8.
 * 0/ok, -1 fail. See tools/silu_native.c. */
int          ork_mm_silu_build_lut(ork_npu *ctx, double in_scale, double out_scale,
                                   int r_mult, int r_shift, uint32_t cfg4068, int16_t *lut);
/* Fused EXP LUT for the coalesced chain (softmax): HW-chains exp onto the score matmul via run_chain_i8_gsilu.
 * Scores must be <=0 (post-max domain). Same signature/calibration as the silu LUT. 0/ok, -1 fail. */
int          ork_mm_chain_build_exp_lut(ork_npu *ctx, double in_scale, double out_scale,
                                        int r_mult, int r_shift, uint32_t cfg4068, int16_t *lut);

/* PPU FUSED SiLU (step 2): full-K int8 matmul with SiLU applied on-chip via the LUT output stage. Two
 * sequential submits (LUT-load into PPU SRAM, then matmul reading it). A[M*K],B[K*N] int8; C[M*N] int8.
 * 0/ok (executed), -1 wedged, -2 bad. WIP: replays the capture scale (per-scale LUT gen is pending). */
int          ork_npu_probe_i8_silu(ork_npu *ctx, int M, int K, int N, const int8_t *A, const int8_t *B,
                                   int8_t *C, double *us);
/* Fused-SiLU probe with the decoded output-stage knobs exposed: r_mult/r_shift = the unified scale
 * R = r_mult/2^r_shift (reg 0x4084/0x4088; sets both the acc->LUT-index step and the LUT->output gain);
 * out_bias = reg 0x4080 (output bias / asymmetric zero-point); idx_off = reg 0x4110 (LUT-index offset C0);
 * cfg4068 = reg 0x4068 (per-scale field, no observed output effect). lut != NULL overrides the LUT contents
 * (for the staircase/const-LUT calibration harness); lut==NULL uses the fixed captured silu curve. */
int          ork_npu_probe_i8_silu_cfg(ork_npu *ctx, int M, int K, int N, const int8_t *A, const int8_t *B,
                                       int r_mult, int r_shift, uint32_t out_bias, uint32_t idx_off,
                                       uint32_t cfg4068, const int16_t *lut, int nlut, int8_t *C, double *us);

/* RE/validation for the fused EW-mul (SwiGLU dual-input) output stage: full-K int8 matmul A*B whose output
 * stage int8-requantizes the accumulator and multiplies it by a SECOND input G (= silu(gate)); returns
 * C[M*N] int8. Splices the 0x50xx second-DPU lane (regcfg 108->126). A[M*K] B[K*N] G[M*N] row-major int8.
 * First-run contract: validate at the captured shape (M=8,N=32) — see EWMUL_WIP.md. 0/ok, -1 wedged, -2 dims. */
int          ork_npu_probe_i8_ewmul(ork_npu *ctx, int M, int K, int N, const int8_t *A, const int8_t *B,
                                    const int8_t *G, int mult, int shift, int8_t *C, double *us);

/* Path (b): submit RKNN's captured EW-mul op VERBATIM (REGCMD_EWMUL) with ork's buffers, only repointing
 * addresses. Tests whether the templatized op executes on ork's submit path (RKNN's own geometry). Buffers
 * mirror the captured handle layout: in->input(4KiB), wt->weights(32KiB,+0x2300), gl->silu(8KiB,+0x400),
 * out<-output(4KiB). 0/ok, -1 wedged. ORK_EW_DUMP=1 prints the regcmd and skips the submit. */
int          ork_npu_probe_i8_ewmul_tmpl(ork_npu *ctx, const void *in, int Isz, const void *wt, int Wsz,
                                         const void *gl, int Gsz, void *out, int Osz, double *us);

/* Path (b) matmul replay: the captured matmul-shaped EW-mul op (K=512,N=64,M=8) with ork's tile packing.
 * out = requant(A*B) (mul) G, G=silu(gate) int8. A[8*512] B[512*64] G[8*64] C[8*64] int8. 0/ok,-1,-2. */
int          ork_npu_probe_i8_ewmul_lin(ork_npu *ctx, const int8_t *A, const int8_t *B, const int8_t *G,
                                        int8_t *C, double *us);

/* Standalone SDP element-wise MULTIPLY (NVDLA standalone SDP layer, both operands from memory): out[i] =
 * clamp_i8(round(a[i]*b[i]*gain)+bias). a,b,out int8[n] (n<=4096). The clean on-NPU element-wise path. 0/ok. */
int          ork_npu_probe_i8_mul(ork_npu *ctx, const int8_t *a, const int8_t *b, int n, int8_t *out, double *us);

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
/* PROBE: chain two int8 SDP ewmuls (ewmul0 middle @desc_slot=138 -> ewmul1) via ork_npu_chain_progs and verify
 * both vs CPU ref. *t0_ok = middle SDP op correct carrying a forward descriptor; *t1_ok = chain walked forward
 * through the SDP slot. 0/ok (see *t0_ok / *t1_ok), -3 no-PPU, -2 alloc, -1 wedge. Proves int8 SDP HW-chains. */
int          ork_npu_probe_sdp_chain_fwd(ork_npu *ctx, int *t0_ok, int *t1_ok);
/* STAGE-1 PROBE: [matmul->ewmul(SDP middle)->matmul] NONBLOCK chain on the begin_mc recipe (warmed scratch +
 * clean-before), completion via the terminal matmul sentinel. *ok = all three outputs bit-exact. 0/ok,-1/-2/-3. */
int          ork_npu_probe_seq_hetero(ork_npu *ctx, int *ok);
/* Self-test: chain 2 plain int8 matmuls (all-ones) and verify BOTH tasks execute. *t0_cnt / *t1_cnt = count
 * of M*N int32 slots == K or 2K (near M*N => that task ran). Validates chain_progs w/ a real task0. */
int          ork_npu_chain_selftest(ork_npu *ctx, int *t0_cnt, int *t1_cnt);

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

/* Faithful fp16 replay: run RKNN's fp16 LUT-load program (`loader`/`ln`) + the fp16 compute op verbatim,
 * patching only I/O + M/N. Uses the fp16 (LE-table) loader, not the int8 LO loader. in/out fp16 [M*N], N%8==0. */
int          ork_npu_replay_full_f16(ork_npu *ctx, const unsigned *loader, int ln, const ork_f16 *in, int M, int N, ork_f16 *out, double *us);
/* RE: replay the captured vendor forward-softmax 9-task PC-chained graph verbatim (capture geometry:
 * reduction=64, 256 rows). in/out = raw 32768-byte buffer images; fill `in` uniform -> softmax=1/64. */
int          ork_npu_replay_softmax_f16(ork_npu *ctx, const void *in, void *out, double *us);

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

/* Replay an ASSEMBLED int16 LUT-op (tools/re/assemble_op.c output): stream `lut` via the loader, run `regcmd`
 * verbatim (RKNN's matched index/scale params baked in), patch only addresses + M/N. Bit-exact to RKNN, no
 * index decode. in/out int16 [M*N], N%8==0. 0/ok,-1,-2,-3. */
int          ork_npu_replay_lut_i16(ork_npu *ctx, const unsigned *regcmd, int rn, const short *lut, int nlut,
                                    const short *in, int M, int N, short *out, double *us);
/* #38 RE: replay a captured int8 matmul regcmd (rkllm's, from regcmd_capture) on ork's submit path.
 * Adata/Bdata: captured A / tiled-weight bytes (NULL => garbage, timing only). Cout: computed int32 output
 * (for correctness check vs captured C). Patches A/B/C addrs, single-core, times `iters`. 0/ok. */
int          ork_npu_replay_i8(ork_npu *ctx, const unsigned *regcmd, int rn, int M, int K, int N,
                               const signed char *Adata, int Abytes, const signed char *Bdata, int Bbytes, int *Cout, int iters, double *us);
/* #39 A-layout solver: replay a fixed captured regcmd for nvar A-variants reusing ONE buffer set (stable IOVA,
 * wedge-safe). Avar = nvar A-images each `astride` bytes; Couts = nvar contiguous M*N int32 results. 0/ok. */
int          ork_npu_replay_i8_sweep(ork_npu *ctx, const unsigned *regcmd, int rn, int M, int K, int N,
                               const signed char *Avar, int nvar, int astride, const signed char *Bdata, int Bbytes, int *Couts);
/* #39 A-layout mapper: for nk0 weight one-hot positions Bpos (ork_woff byte for (n0,k0)), recover per output
 * slot the A byte offset the fold read. Fills rpos[nk0*M] (raw C int32 index), aoff[nk0*M] (A byte offset),
 * cnt[nk0] (slots found). One buffer set, wedge-safe. 0/ok. */
int          ork_npu_replay_i8_amap(ork_npu *ctx, const unsigned *regcmd, int rn, int M, int K, int N,
                               const unsigned *Bpos, int nk0, int n0, int *rpos, int *aoff, int *cnt);
/* #39 PORT (RE): dump ork's synth_i8 regcmd for (mc,K,N) to diff vs rkllm's captured regcmd. Returns word count. */
int          ork_npu_synth_i8_dump(ork_npu *ctx, int mc, int K, int N, unsigned *out, int outn);
/* #39 weight-resident M-fold chain: P width-w mfold tiles, shared K*N weight, one task_number=P submit. */
int          ork_npu_mfold_chain(ork_npu *ctx, int P, int w, int K, int N,
                                 const signed char *Apacked, const signed char *Bpacked,
                                 int *Craw, int iters, double *us);
/* #39 same, but each task is a CAPTURED bit-exact tile regcmd (tile_rc/trn) with only addresses re-based. */
int          ork_npu_mfold_chain_cap(ork_npu *ctx, int P, int w, int K, int N,
                                     const unsigned *tile_rc, int trn,
                                     const signed char *Apacked, const signed char *Bpacked,
                                     int *Craw, int iters, double *us);
/* #39 TIMING probe: replay P DIFFERENT captured tiles (tiles=P*rn words) in one chain, shared weight, zeroed
 * operands. The timing slope vs P reveals weight re-DMA (linear) vs resident reuse (sublinear). */
int          ork_npu_mfold_chain_multi(ork_npu *ctx, int P, int w, int K, int N,
                                       const unsigned *tiles, int rn, int iters, double *us);
/* #39 UNIFIED per-tile fold chain: P tiles of per-tile width ws[t], real operands (concat nc16 A / woff B),
 * Craw out (concat c4), optional WEIGHT_REUSE on tiles t>0. For [state-setter+big-M] and weight-reuse chains. */
int          ork_npu_mfold_chain_v(ork_npu *ctx, int P, const int *ws, int K, int N,
                                    const unsigned *tiles, int rn, const signed char *Apacked,
                                    const signed char *Bpacked, int *Craw, int wreuse, int iters, double *us);
/* #39 Path-1 TOKEN-TILER executor: run P fold sub-tiles of one M_total-token batch as ONE multi-task submit over a
 * SHARED batch cube (M_total x K nc16 in, M_total x N c4 out, shared woff weight). Tile t handles rows
 * [row_off[t], row_off[t]+m) at byte offset row_off[t]*16. Caller prepares each tile's regcmd (per-size skeleton +
 * the 4 M_total regs patched: 0x4024=16*M_total, 0x107c=M_total, 0x1080=M_total-m, 0x40c0=128*M_total). This is
 * rkllm's real fold — a batch amortized over few big-M tiles. Returns 0/ok, us=avg submit. */
int          ork_npu_fold_batch(ork_npu *ctx, int Mtot, int K, int N, int P, const int *row_off,
                                const unsigned *tiles, int rn, const signed char *Apacked,
                                const signed char *Bpacked, int *Craw, int ncore, int iters, double *us);
/* #39 FOLD MATMUL run-path: C[M,N] int32 = A[M,K] int8 x W[K,N] int8 via the token-fold (M<=128 tiled into
 * <=36-row sub-tiles, one shared-cube multi-task submit). Baked per-size templates => only K=FOLD_REF_K(3584),
 * N=FOLD_REF_N(1216), M<=128; returns -1 otherwise (caller keeps the standard path). A/W/Cout row-major. */
int          ork_npu_fold_run_i8(ork_npu *ctx, int K, int N, const signed char *Wraw, int M,
                                 const signed char *Araw, int *Cout, int ncore, int iters, double *us);
/* #39 FULL-OP fold: C[M,N]=A[M,K]xW[K,N] int8 via N-split across cores (each core single-core-folds a <=1216-wide
 * slice concurrently) — rkllm's wide-projection scheme. K=FOLD_REF_K, N<=3*1216, M<=128. A/Cout row-major. */
int          ork_npu_fold_op_i8(ork_npu *ctx, int K, int N, const signed char *Wraw, int M,
                                const signed char *Araw, int *Cout, int iters, double *us);
/* #39 mfold RESIDENT-weight variant: fold matmul from a PRE-PACKED w->Bfold (ork_mm_load_fold_i8 / orkpack v5),
 * no per-call fold_woff repack. K=FOLD_REF_K, N<=3*1216, M<=128. A/Cout row-major. 0/ok, -1/-2/-3 as fold_op. */
int          ork_npu_fold_run_w(ork_npu *ctx, ork_w *w, int M, const signed char *Araw, int *Cout, int iters, double *us);
/* #39 SHARED-INPUT fold batch: nw weights (K=FOLD_REF_K, resident Bfold, same domain) sharing input A[M,K] — QKV
 * or gate+up. The nc16 input marshal is done ONCE and reused across all nw folds. Couts[i] row-major. 0/-1/-2/-3. */
int          ork_npu_fold_batch_w(ork_npu *ctx, int nw, ork_w **ws, int M, const signed char *Araw, int **Couts, int iters, double *us);
/* #39 mfold orkpack companions: dump the fold_woff-layout weight blob PURE-CPU (out=NULL to size), and load it
 * back into a resident ork_w carrying only w->Bfold. K=FOLD_REF_K(3584), N<=3*FOLD_REF_N(1216), N%32. */
size_t       ork_w_dump_fold_i8_cpu(ork_npu *ctx, int K, int N, const signed char *B, void *out, size_t cap);
ork_w       *ork_mm_load_fold_i8(ork_npu *ctx, int K, int N, const void *blob, size_t n);
/* Attach a fold blob to an EXISTING ork_w (packed/loaded normally) so ork_mm_run_i8 auto-routes small M<=64
 * through the fold for the eligible q/o shapes. 0 ok (or already attached), <0 on shape/size mismatch. */
int          ork_w_attach_fold_i8(ork_npu *ctx, ork_w *w, const void *blob, size_t n);
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
/* Tier 11 doorbell pipeline profiler: N serial int8 matmuls, BLOCKING vs NONBLOCK + DRAM-doorbell busy-poll.
 * Fills per-op us for each + a correctness flag each. See Optimization-Roadmap Tier 11. */
int          ork_npu_doorbell_prof(ork_npu *ctx, int M, int K, int N, int iters, double *block_us, double *nb_us, int *ok_block, int *ok_nb);
/* Tier 11 overlap probe: CPU router (cpu_reps x 512x512 fp32 GEMV) in the shadow of an async NPU op.
 * Fills npu_solo / cpu_solo / overlap_wall (us/iter). hidden% = (npu_solo+cpu_solo-overlap_wall)/min(...). */
int          ork_npu_overlap_prof(ork_npu *ctx, int M, int K, int N, int cpu_reps, int iters, double *npu_solo, double *cpu_solo, double *overlap_wall, int *ok);
/* RE (WIP): full-chain replay of vendor gemm+reshape (task0-10); returns gemm output (contiguous) + reshape
 * output (atom-8) so caller verifies reshape_out==atom8(gemm_out). Loads gemm_mul_image.bin. See RESHAPE_WIP.md. */
int          ork_npu_replay_reshape_f16(ork_npu *ctx, unsigned short *gemm_raw, int gemm_words, unsigned short *reshape_raw, int reshape_words, double *us);
/* RE (WIP): vendor fp16 contiguous->atom-8 RESHAPE base op (task4) with a constructed permutation weight.
 * Reads the output RAW for layout inspection. N=64/M<=8 only (captured geometry). See RESHAPE_WIP.md. */
int          ork_npu_reshape_probe_f16(ork_npu *ctx, int M, int N, const unsigned short *src, unsigned short *out_raw, int out_words, double *us);
/* LOOPBACK Pass-2: standalone SDP reads INT32 accumulator from DRAM, per-channel scale + requant -> int16.
 * out[m][n]=clamp_i16(a_i32[m][n]*b[n]*mult>>shift). Routes around the broken CNA->DPU requant-WDMA. */
int          ork_npu_requant_perchan_i32(ork_npu *ctx, const int *a, const short *b, int M, int N, int mult, int shift, short *out, double *us);
/* CHAIN: int8-matmul(int16-out) -> per-channel-scale in ONE PC-chain (A·V->normalize building block for the
 * single-submit attention chain). out=clamp_i16(requant(A[M,K]xB[K,N],m1,s1)*scale[n]*m2>>s2). K%32,N%32,M<=64. */
int          ork_npu_chain_mm_perchan_i16(ork_npu *ctx, int M, int K, int N, const signed char *A, const signed char *B,
                                          const short *scale, int m1, int s1, int m2, int s2, short *out, double *us);
int          ork_npu_chain_mm_perchan_f16(ork_npu *ctx, int M, int K, int N, const unsigned short *A, const unsigned short *B,
                                          const unsigned short *scale, unsigned short *out, double *us);
int          ork_npu_f16_gap_probe(ork_npu *ctx, int M, int Kp, int N, int use_gap, long *nz0, long *nz1, double *us); /* (B') cross-slice drain-gap probe */
/* RE/calibration only: per-core-fd concurrency probe. 3-core fp16 N-column-split matmul where EACH core submits
 * on its OWN fresh DRM fd (not the shared ctx fd), to test whether per-core-fd isolation changes the concurrent-
 * fetch wedge. mode 0 = each core gets its own weight copy; mode 1 = one shared dma-heap weight imported into
 * every core's fd. A[M,K],B[K,N] fp16 row-major; Cout[M,N] fp32 (fp16-out converted). K%32,N%16,N<=nmax,M<=64,
 * N%(cores*16)==0. *us = concurrent submit wall time. Returns 0 on a completed run (incl. a wedge — Cout shows
 * it), -2 bad shape, -1 open/alloc. See tools/percore_fd_probe.c. */
int          ork_npu_f16_percore_probe(ork_npu *ctx, int M, int K, int N, const ork_f16 *A, const ork_f16 *B, float *Cout, double *us, int mode);
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

/* RE/calibration only: run ONE full-K int8 submit at (M,K,N) in ork's current M-tile mode (mode=0)
 * or the rkllm-captured M-tile mode (mode=1: 0x1010=0x20 const, 0x1044=(K/64)*M, 0x107c=4*M,
 * 0x1040=0xb1-0xf*(ceil(M/8)-1)). Returns the full C[M*N] int32 + warm-submit us. Tests whether
 * rkllm's larger-M-per-submit is bit-exact on ork and lifts effective TOPS. 0/ok, -1 wedged, -2 bad.
 * A[M*K], B[K*N] row-major int8; C[M*N] int32. See tools/mtile_probe.c. */
int          ork_npu_probe_mtile_i8(ork_npu *ctx, int M, int K, int N, int mode,
                                    const int8_t *A, const int8_t *B, int32_t *C, double *us);

/* RE/calibration only: does batching tasks per RKNPU_SUBMIT amortize the per-submit round-trip
 * floor? Runs `ntask` identical small int8 matmuls as ntask separate ioctls vs one ioctl with
 * task_number=ntask, writing the two total wall times (us). Returns 0/ok, -1 wedged, -2 bad dims.
 * See tools/batch_probe.c. */
int          ork_npu_probe_batch(ork_npu *ctx, int ntask, int K, int N, double *us_unbatched, double *us_batched);

/* RE/calibration only: probe in-place K-slicing of a full-K weight buffer. Packs B[Kfull,N] fp16
 * full-K, runs one M=1 submit over k in [0,Kp), with up to `nov` regcmd overrides (block 0x0201)
 * to hunt the per-N-tile weight stride register. C[N] should equal the Kp-partial sum if slicing
 * works. Returns 0/ok, -1 wedged, -2 bad dims. See tools/slice_probe.c. */
int          ork_npu_probe_slice_f16(ork_npu *ctx, int Kfull, int N, int Kp, int nov,
                                     const uint32_t *ovr_reg, const uint32_t *ovr_val,
                                     const ork_f16 *A, const ork_f16 *B, float *C);

/* RE/calibration only: probe W4A4 (int4 A x int4 B -> int16 C) using the captured RK3588 regcmd
 * (REGCMD_I4, M=4) and the DOCUMENTED native tile layouts (A:(K/32,M,32) B:(N/64,K/32,64,32)
 * C:(N/8,M,8)). A is [4*K], B is [K*N] row-major, int4 values stored as int8 in [-8,7]. `nibB`/`nibA`
 * (0/1) toggle the 2-int4-per-byte nibble order (the one detail the docs don't pin). C is [4*N]
 * int16 (de-tiled). `nov`/`ovr_*` patch extra CNA (0x0201) regs. Returns 0/ok, -1 wedged/abort,
 * -2 bad dims (K%32, N%64). See tools/i4_probe.c. */
int          ork_npu_probe_i4(ork_npu *ctx, int M, int K, int N, int nibB, int nibA, int nov,
                              const uint32_t *ovr_reg, const uint32_t *ovr_val,
                              const signed char *A, const signed char *B, short *C);

/* RE/calibration only (Tier 4b): ONE multi-M int4 submit with the M-count regs set to M (synth_i4
 * mc=M). A:(K/32,M,32), B:(N/64,K/32,64,32) native. Copies the RAW int16 output (M*N, NO de-tile) to
 * `raw` so the caller can deduce the multi-M C layout vs a CPU reference. 0/ok, -1 wedged, -2 dims.
 * See tools/i4_multim_probe.c. */
int          ork_npu_probe_i4_mm(ork_npu *ctx, int M, int K, int N,
                                 const signed char *A, const signed char *B, short *raw);
/* RE fuzzer hooks (tools/i4_multim_fuzz.c): queue arbitrary (block,reg,val) overrides applied at the end of
 * every synth_i4 (int4 regcmd). Inert until added; clear before each probe. Sweep the multi-M K-schedule space. */
void         ork_i4_fuzz_clear(void);
void         ork_i4_fuzz_add(unsigned int blk, unsigned int reg, unsigned int val);
/* int8 analogs (batch-mode RE): overrides applied at the end of synth_i8; raw int32 multi-M output probe
 * (2*M*N int32, room for a stride-2 batch layout). See tools/i4_multim_fuzz.c int8 modes. */
void         ork_i8_fuzz_clear(void);
void         ork_i8_fuzz_add(unsigned int blk, unsigned int reg, unsigned int val);
int          ork_npu_probe_i8_mm(ork_npu *ctx, int M, int K, int N,
                                 const signed char *A, const signed char *B, int *raw);
/* fp16 analogs (batch-mode mapping): A/B are fp16 bit patterns (uint16), output raw fp32 (2*M*N floats). */
void         ork_f16_fuzz_clear(void);
void         ork_f16_fuzz_add(unsigned int blk, unsigned int reg, unsigned int val);
int          ork_npu_probe_f16_mm(ork_npu *ctx, int M, int K, int N,
                                 const unsigned short *A, const unsigned short *B, float *raw);
int          ork_npu_probe_f16_mm_f16out(ork_npu *ctx, int M, int K, int N,
                                 const unsigned short *A, const unsigned short *B, unsigned short *out);
int          ork_npu_probe_f16_stridedA(ork_npu *ctx, int M, int K, int N, const unsigned short *A, int apitch,
                                 const unsigned short *B, unsigned short *out);
int          ork_npu_mm_perchan_f16_fused(ork_npu *ctx, int M, int K, int N, const unsigned short *A,
                                 const unsigned short *B, const unsigned short *scale, unsigned short *out);
int          ork_npu_mul_perchan_f16_contig(ork_npu *ctx, const ork_f16 *a, const ork_f16 *b, int M, int N,
                                 ork_f16 *out, double *us);
int          ork_npu_mm_perchan_f16(ork_npu *ctx, int M, int K, int N, const unsigned short *A,
                                 const unsigned short *B, const unsigned short *scale, unsigned short *out, double *us);
int          ork_npu_mm_perchan_f16_diag(ork_npu *ctx, int M, int K, int N, const unsigned short *A,
                                 const unsigned short *B, const unsigned short *scale, unsigned short *out, double *us);

/* RE/calibration only: run S chained M=1 full-K int8 matmuls using PC-chaining in a single submit */
int          ork_npu_probe_chain_i8(ork_npu *ctx, int S, int K, int N, const int8_t *A,
                                    const int8_t *B, int32_t *C);

/* RE/calibration only: benchmark S chained matmuls vs S separate submits using pre-allocated memory */
int          ork_npu_benchmark_chain(ork_npu *ctx, int S, int K, int N, int iters);

#endif /* ORK_OPS_H */
