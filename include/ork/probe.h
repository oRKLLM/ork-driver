/* ork/probe.h — Reverse-engineering probes, replays and fuzz hooks — NOT the supported API
 *
 * Everything here exists to interrogate the hardware, not to serve a workload: the probe, replay and fuzz
 * surface, the register-dump and profiling counters, and the fold/perchan
 * experiments. Split out of ork/ops.h on EVIDENCE, not naming — each of these has no caller in
 * src/, no example, and no ggml-ork use; they are reached only from tools/, and eight are
 * currently referenced from nowhere at all. Kept exported because tools/ is how the RE record
 * is reproduced, but nothing in a production path should appear here.
 *
 * Part of the public API. Do NOT include this directly — include <ork_npu.h>, which is the
 * umbrella and the only supported entry point. Types live in ork_npu.h above the includes
 * (and ork/sdp.h, included first), so this header is not self-contained by design. */
#ifndef ORK_PROBE_H
#define ORK_PROBE_H


/* Profiling (read via ork_npu_mc_timing; the ORK_MCPROF env gate is removed): per-core phase times (us) inside the multi-core prefill (M>1)
 * path — copy (activation host-copy + bsync), submit (regcmd + ioctl + result bsync), acc (host
 * accumulate), and the submit count. Pins why large-M multi-core barely scales. Call _reset before
 * the timed region. core in [0, cores). */
void         ork_npu_mc_reset(void);
/* Doorbell phase split for the M=1 / colsplit regime, which the orki_mc_* counters do not cover
 * (they are M>1 mcworker only). begin = regcmd synth + bsync + NONBLOCK ioctl; end = sentinel poll +
 * writeback + teardown. Needs ORK_PROFILE=1. Reset with ork_npu_db_reset. */
void         ork_npu_db_timing(double *begin_us, long *begin_n, double *end_us, long *end_n);
void         ork_npu_db_reset(void);
double       ork_npu_db_poll(void);   /* the WAIT half of `end`; end-poll = drain + accumulate + writeback */
double       ork_npu_mc_synth(int core);   /* host synth+bsync subset of `submit` (overlappable); ioctl/NPU = submit - synth */

/* RE/calibration only: probe this SoC's single-submit K-tile ceiling. Runs ONE M=1 full-K int8
 * submit at (K,N) (N <= SoC N-cap, K%32, N%32) on its own buffers. Returns 0 if the submit
 * completed (C[N] int32 valid — validate vs CPU), -1 if it wedged (K exceeds the per-op K-tile
 * cap; recoverable), -2 on bad dims. See tools/ksubmit_probe.c. */
int          ork_i8_npu_probe_single(ork_npu *ctx, int K, int N, const int8_t *A, const int8_t *B, int32_t *C);

/* PHASE 1 (#35 chained FFN): int8 matmul with the INT16-requantized output stage (set_i16_out).
 * out_i16 = clamp_i16(round(acc_i32 * mult / 2^shift)); identity = (0x4000,14). C[M*N] int16 out
 * (raw device layout). The RE crux for the on-NPU matmul->int16-silu handoff. 0/ok, -1 wedged, -2 dims. */
int          ork_i16_npu_probe_out(ork_npu *ctx, int M, int K, int N, const int8_t *A, const int8_t *B,
                                   int mult, int shift, short *C, double *us);

/* PPU FUSED SiLU (step 2): full-K int8 matmul with SiLU applied on-chip via the LUT output stage. Two
 * sequential submits (LUT-load into PPU SRAM, then matmul reading it). A[M*K],B[K*N] int8; C[M*N] int8.
 * 0/ok (executed), -1 wedged, -2 bad. WIP: replays the capture scale (per-scale LUT gen is pending). */
int          ork_i8_npu_probe_silu(ork_npu *ctx, int M, int K, int N, const int8_t *A, const int8_t *B,
                                   int8_t *C, double *us);

/* RE/validation for the fused EW-mul (SwiGLU dual-input) output stage: full-K int8 matmul A*B whose output
 * stage int8-requantizes the accumulator and multiplies it by a SECOND input G (= silu(gate)); returns
 * C[M*N] int8. Splices the 0x50xx second-DPU lane (regcfg 108->126). A[M*K] B[K*N] G[M*N] row-major int8.
 * First-run contract: validate at the captured shape (M=8,N=32) — see EWMUL_WIP.md. 0/ok, -1 wedged, -2 dims. */
int          ork_i8_npu_probe_ewmul(ork_npu *ctx, int M, int K, int N, const int8_t *A, const int8_t *B,
                                    const int8_t *G, int mult, int shift, int8_t *C, double *us);

/* Path (b): submit RKNN's captured EW-mul op VERBATIM (REGCMD_EWMUL) with ork's buffers, only repointing
 * addresses. Tests whether the templatized op executes on ork's submit path (RKNN's own geometry). Buffers
 * mirror the captured handle layout: in->input(4KiB), wt->weights(32KiB,+0x2300), gl->silu(8KiB,+0x400),
 * out<-output(4KiB). 0/ok, -1 wedged. ORK_EW_DUMP=1 prints the regcmd and skips the submit. */
int          ork_i8_npu_probe_ewmul_tmpl(ork_npu *ctx, const void *in, int Isz, const void *wt, int Wsz,
                                         const void *gl, int Gsz, void *out, int Osz, double *us);

/* Path (b) matmul replay: the captured matmul-shaped EW-mul op (K=512,N=64,M=8) with ork's tile packing.
 * out = requant(A*B) (mul) G, G=silu(gate) int8. A[8*512] B[512*64] G[8*64] C[8*64] int8. 0/ok,-1,-2. */
int          ork_i8_npu_probe_ewmul_lin(ork_npu *ctx, const int8_t *A, const int8_t *B, const int8_t *G,
                                        int8_t *C, double *us);

/* Standalone SDP element-wise MULTIPLY (NVDLA standalone SDP layer, both operands from memory): out[i] =
 * clamp_i8(round(a[i]*b[i]*gain)+bias). a,b,out int8[n] (n<=4096). The clean on-NPU element-wise path. 0/ok. */
int          ork_i8_npu_probe_mul(ork_npu *ctx, const int8_t *a, const int8_t *b, int n, int8_t *out, double *us);
/* PROBE: chain two int8 SDP ewmuls (ewmul0 middle @desc_slot=138 -> ewmul1) via ork_npu_chain_progs and verify
 * both vs CPU ref. *t0_ok = middle SDP op correct carrying a forward descriptor; *t1_ok = chain walked forward
 * through the SDP slot. 0/ok (see *t0_ok / *t1_ok), -3 no-PPU, -2 alloc, -1 wedge. Proves int8 SDP HW-chains. */
int          ork_npu_probe_sdp_chain_fwd(ork_npu *ctx, int *t0_ok, int *t1_ok);
/* STAGE-1 PROBE: [matmul->ewmul(SDP middle)->matmul] NONBLOCK chain on the begin_mc recipe (warmed scratch +
 * clean-before), completion via the terminal matmul sentinel. *ok = all three outputs bit-exact. 0/ok,-1/-2/-3. */
int          ork_npu_probe_seq_hetero(ork_npu *ctx, int *ok);

/* Faithful fp16 replay: run RKNN's fp16 LUT-load program (`loader`/`ln`) + the fp16 compute op verbatim,
 * patching only I/O + M/N. Uses the fp16 (LE-table) loader, not the int8 LO loader. in/out fp16 [M*N], N%8==0. */
int          ork_f16_npu_replay_full(ork_npu *ctx, const unsigned *loader, int ln, const ork_f16 *in, int M, int N, ork_f16 *out, double *us);
/* RE: replay the captured vendor forward-softmax 9-task PC-chained graph verbatim (capture geometry:
 * reduction=64, 256 rows). in/out = raw 32768-byte buffer images; fill `in` uniform -> softmax=1/64. */
int          ork_f16_npu_replay_softmax(ork_npu *ctx, const void *in, void *out, double *us);

/* Replay an ASSEMBLED int16 LUT-op (tools/re/assemble_op.c output): stream `lut` via the loader, run `regcmd`
 * verbatim (RKNN's matched index/scale params baked in), patch only addresses + M/N. Bit-exact to RKNN, no
 * index decode. in/out int16 [M*N], N%8==0. 0/ok,-1,-2,-3. */
int          ork_i16_npu_replay_lut(ork_npu *ctx, const unsigned *regcmd, int rn, const short *lut, int nlut,
                                    const short *in, int M, int N, short *out, double *us);
/* #39 A-layout solver: replay a fixed captured regcmd for nvar A-variants reusing ONE buffer set (stable IOVA,
 * wedge-safe). Avar = nvar A-images each `astride` bytes; Couts = nvar contiguous M*N int32 results. 0/ok. */
int          ork_i8_npu_replay_sweep(ork_npu *ctx, const unsigned *regcmd, int rn, int M, int K, int N,
                               const signed char *Avar, int nvar, int astride, const signed char *Bdata, int Bbytes, int *Couts);
/* #39 PORT (RE): dump ork's synth_i8 regcmd for (mc,K,N) to diff vs rkllm's captured regcmd. Returns word count. */
int          ork_i8_npu_synth_dump(ork_npu *ctx, int mc, int K, int N, unsigned *out, int outn);
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
int          ork_i8_npu_fold_run(ork_npu *ctx, int K, int N, const signed char *Wraw, int M,
                                 const signed char *Araw, int *Cout, int ncore, int iters, double *us);
/* #39 SHARED-INPUT fold batch: nw weights (K=FOLD_REF_K, resident Bfold, same domain) sharing input A[M,K] — QKV
 * or gate+up. The nc16 input marshal is done ONCE and reused across all nw folds. Couts[i] row-major. 0/-1/-2/-3. */
int          ork_npu_fold_batch_w(ork_npu *ctx, int nw, ork_w **ws, int M, const signed char *Araw, int **Couts, int iters, double *us);
/* #39 mfold orkpack companions: dump the fold_woff-layout weight blob PURE-CPU (out=NULL to size), and load it
 * back into a resident ork_w carrying only w->Bfold. K=FOLD_REF_K(3584), N<=3*FOLD_REF_N(1216), N%32. */
size_t       ork_i8_w_dump_fold_cpu(ork_npu *ctx, int K, int N, const signed char *B, void *out, size_t cap);
ork_w       *ork_i8_mm_load_fold(ork_npu *ctx, int K, int N, const void *blob, size_t n);
/* Attach a fold blob to an EXISTING ork_w (packed/loaded normally) so ork_i8_mm_run auto-routes small M<=64
 * through the fold for the eligible q/o shapes. 0 ok (or already attached), <0 on shape/size mismatch. */
int          ork_i8_w_attach_fold(ork_npu *ctx, ork_w *w, const void *blob, size_t n);
/* Tier 11 doorbell pipeline profiler: N serial int8 matmuls, BLOCKING vs NONBLOCK + DRAM-doorbell busy-poll.
 * Fills per-op us for each + a correctness flag each. See Optimization-Roadmap Tier 11. */
int          ork_npu_doorbell_prof(ork_npu *ctx, int M, int K, int N, int iters, double *block_us, double *nb_us, int *ok_block, int *ok_nb);
/* Tier 11 overlap probe: CPU router (cpu_reps x 512x512 fp32 GEMV) in the shadow of an async NPU op.
 * Fills npu_solo / cpu_solo / overlap_wall (us/iter). hidden% = (npu_solo+cpu_solo-overlap_wall)/min(...). */
int          ork_npu_overlap_prof(ork_npu *ctx, int M, int K, int N, int cpu_reps, int iters, double *npu_solo, double *cpu_solo, double *overlap_wall, int *ok);
/* RE (WIP): full-chain replay of vendor gemm+reshape (task0-10); returns gemm output (contiguous) + reshape
 * output (atom-8) so caller verifies reshape_out==atom8(gemm_out). Loads gemm_mul_image.bin. See RESHAPE_WIP.md. */
int          ork_f16_npu_replay_reshape(ork_npu *ctx, unsigned short *gemm_raw, int gemm_words, unsigned short *reshape_raw, int reshape_words, double *us);
/* LOOPBACK Pass-2: standalone SDP reads INT32 accumulator from DRAM, per-channel scale + requant -> int16.
 * out[m][n]=clamp_i16(a_i32[m][n]*b[n]*mult>>shift). Routes around the broken CNA->DPU requant-WDMA. */
int          ork_npu_requant_perchan_i32(ork_npu *ctx, const int *a, const short *b, int M, int N, int mult, int shift, short *out, double *us);
/* CHAIN: int8-matmul(int16-out) -> per-channel-scale in ONE PC-chain (A·V->normalize building block for the
 * single-submit attention chain). out=clamp_i16(requant(A[M,K]xB[K,N],m1,s1)*scale[n]*m2>>s2). K%32,N%32,M<=64. */
int          ork_i16_npu_chain_mm_perchan(ork_npu *ctx, int M, int K, int N, const signed char *A, const signed char *B,
                                          const short *scale, int m1, int s1, int m2, int s2, short *out, double *us);
int          ork_f16_npu_chain_mm_perchan(ork_npu *ctx, int M, int K, int N, const unsigned short *A, const unsigned short *B,
                                          const unsigned short *scale, unsigned short *out, double *us);
int          ork_f16_npu_gap_probe(ork_npu *ctx, int M, int Kp, int N, int use_gap, long *nz0, long *nz1, double *us); /* (B') cross-slice drain-gap probe */
/* RE/calibration only: per-core-fd concurrency probe. 3-core fp16 N-column-split matmul where EACH core submits
 * on its OWN fresh DRM fd (not the shared ctx fd), to test whether per-core-fd isolation changes the concurrent-
 * fetch wedge. mode 0 = each core gets its own weight copy; mode 1 = one shared dma-heap weight imported into
 * every core's fd. A[M,K],B[K,N] fp16 row-major; Cout[M,N] fp32 (fp16-out converted). K%32,N%16,N<=nmax,M<=64,
 * N%(cores*16)==0. *us = concurrent submit wall time. Returns 0 on a completed run (incl. a wedge — Cout shows
 * it), -2 bad shape, -1 open/alloc. See tools/percore_fd_probe.c. */
int          ork_f16_npu_percore_probe(ork_npu *ctx, int M, int K, int N, const ork_f16 *A, const ork_f16 *B, float *Cout, double *us, int mode);

/* RE/calibration only: run ONE full-K int8 submit at (M,K,N) in ork's current M-tile mode (mode=0)
 * or the rkllm-captured M-tile mode (mode=1: 0x1010=0x20 const, 0x1044=(K/64)*M, 0x107c=4*M,
 * 0x1040=0xb1-0xf*(ceil(M/8)-1)). Returns the full C[M*N] int32 + warm-submit us. Tests whether
 * rkllm's larger-M-per-submit is bit-exact on ork and lifts effective TOPS. 0/ok, -1 wedged, -2 bad.
 * A[M*K], B[K*N] row-major int8; C[M*N] int32. See tools/mtile_probe.c. */
int          ork_i8_npu_probe_mtile(ork_npu *ctx, int M, int K, int N, int mode,
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
int          ork_f16_npu_probe_slice(ork_npu *ctx, int Kfull, int N, int Kp, int nov,
                                     const uint32_t *ovr_reg, const uint32_t *ovr_val,
                                     const ork_f16 *A, const ork_f16 *B, float *C);

/* RE/calibration only: probe W4A4 (int4 A x int4 B -> int16 C) using the captured RK3588 regcmd
 * (REGCMD_I4, M=4) and the DOCUMENTED native tile layouts (A:(K/32,M,32) B:(N/64,K/32,64,32)
 * C:(N/8,M,8)). A is [4*K], B is [K*N] row-major, int4 values stored as int8 in [-8,7]. `nibB`/`nibA`
 * (0/1) toggle the 2-int4-per-byte nibble order (the one detail the docs don't pin). C is [4*N]
 * int16 (de-tiled). `nov`/`ovr_*` patch extra CNA (0x0201) regs. Returns 0/ok, -1 wedged/abort,
 * -2 bad dims (K%32, N%64). See tools/i4_probe.c. */
int          ork_i4_npu_probe(ork_npu *ctx, int M, int K, int N, int nibB, int nibA, int nov,
                              const uint32_t *ovr_reg, const uint32_t *ovr_val,
                              const signed char *A, const signed char *B, short *C);
/* RE fuzzer hooks (tools/i4_multim_fuzz.c): queue arbitrary (block,reg,val) overrides applied at the end of
 * every synth_i4 (int4 regcmd). Inert until added; clear before each probe. Sweep the multi-M K-schedule space. */
void         ork_i4_fuzz_clear(void);
void         ork_i4_fuzz_add(unsigned int blk, unsigned int reg, unsigned int val);
/* int8 analogs (batch-mode RE): overrides applied at the end of synth_i8; raw int32 multi-M output probe
 * (2*M*N int32, room for a stride-2 batch layout). See tools/i4_multim_fuzz.c int8 modes. */
void         ork_i8_fuzz_clear(void);
int          ork_i8_npu_probe_mm(ork_npu *ctx, int M, int K, int N,
                                 const signed char *A, const signed char *B, int *raw);
/* fp16 analogs (batch-mode mapping): A/B are fp16 bit patterns (uint16), output raw fp32 (2*M*N floats). */
void         ork_f16_fuzz_clear(void);
int          ork_f16_npu_probe_mm(ork_npu *ctx, int M, int K, int N,
                                 const unsigned short *A, const unsigned short *B, float *raw);
int          ork_f16_npu_probe_stridedA(ork_npu *ctx, int M, int K, int N, const unsigned short *A, int apitch,
                                 const unsigned short *B, unsigned short *out);
int          ork_f16_npu_mm_perchan_fused(ork_npu *ctx, int M, int K, int N, const unsigned short *A,
                                 const unsigned short *B, const unsigned short *scale, unsigned short *out);
int          ork_f16_npu_mul_perchan_contig(ork_npu *ctx, const ork_f16 *a, const ork_f16 *b, int M, int N,
                                 ork_f16 *out, double *us);
int          ork_f16_npu_mm_perchan(ork_npu *ctx, int M, int K, int N, const unsigned short *A,
                                 const unsigned short *B, const unsigned short *scale, unsigned short *out, double *us);
int          ork_f16_npu_mm_perchan_diag(ork_npu *ctx, int M, int K, int N, const unsigned short *A,
                                 const unsigned short *B, const unsigned short *scale, unsigned short *out, double *us);


#endif /* ORK_PROBE_H */
