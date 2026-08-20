/* ork/bmm.h — Batched dynamic GEMM (attention), floor-decomp diagnostics, mode-transition hooks, SSM
 *
 * The batched GEMM primitive attention is built on, plus the submit-floor decomposition
 * counters, the mode-transition RE hooks, and the Mamba-2 / SSD scan entrypoints.
 *
 * Part of the public API. Do NOT include this directly — include <ork_npu.h>, which is the
 * umbrella and the only supported entry point; these parts are a readability split of it
 * (ork_npu.h was 1519 lines) and their boundaries may move. Types live in ork_npu.h above
 * the includes, so this header is not self-contained by design. */
#ifndef ORK_BMM_H
#define ORK_BMM_H
/* ---- BATCHED DYNAMIC GEMM (the "attention" primitive) --------------------------------------------
 * For each b in [0,nbatch): C[b] = A[b] * B[b], with A[b] = [M,K], B[b] = [K,N], C[b] = [M,N],
 * each batch dense/contiguous row-major. UNLIKE ork_i8_mm_run/i4/run (which take a RESIDENT static
 * 2-D weight packed once), BOTH operands here are DYNAMIC activations — B[b] is packed ephemerally
 * each call. This is the primitive attention (scores = Q·Kᵀ, out = scores·V, looped per head) and the
 * Gated-Delta-Net chunked matmuls (delta-net-base.cpp) need — ggml's batched MUL_MAT (ne[2]/ne[3]>1
 * with a computed src0) maps directly onto it. The caller arranges any transpose so B is [K,N].
 * i8:  A,B int8   → C int32 (exact).           K%32==0, N%32==0.
 * i4:  A,B int4-in-int8 [-8,7] → C int32.       K%32==0, N%64==0.
 * f16: A,B fp16   → C fp32.                     K%32==0, N%16==0.
 * nbatch>0, M>0. Returns 0 on success, <0 on error. Correctness-first (per-batch submit); submit-floor
 * amortization via chaining is a follow-up. */
int          ork_i8_bmm  (ork_npu *ctx, int nbatch, int M, int K, int N, const int8_t  *A, const int8_t  *B, int32_t *C);
int          ork_i4_bmm  (ork_npu *ctx, int nbatch, int M, int K, int N, const int8_t  *A, const int8_t  *B, int32_t *C);
int          ork_bmm_fp16(ork_npu *ctx, int nbatch, int M, int K, int N, const ork_f16 *A, const ork_f16 *B, float   *C);

/* Strided batched GEMM — the SAME batched primitive, but each operand is addressed by explicit
 * ELEMENT strides instead of assuming dense row-major, so a permuted / view operand offloads WITHOUT
 * the caller first materializing a contiguous copy (ggml_cont). This is what real attention needs:
 * QKᵀ/AV read a permuted-Q ([d,H,T]→[d,T,H]) and a KV-cache view (rows strided by the cache's padded
 * head layout), which ggml_is_contiguous rejects — so those ops were declined and stayed on CPU.
 *   A[m,k] = A + b*abs + m*as_m + k*as_k
 *   B[k,n] = B + b*bbs + k*bs_k + n*bs_n
 *   C[m,n] = C + b*cbs + m*cs_m + n*cs_n
 * All strides are in ELEMENTS (not bytes). Set the natural values (as_m=K,as_k=1,bs_k=N,bs_n=1,
 * cs_m=N,cs_n=1, abs=M*K,bbs=K*N,cbs=M*N) to reproduce the dense ork_bmm_* above. The contraction dim
 * (k) is typically stride-1 in attention; strided-k is supported (gathered) but slower. Each batch's
 * strided operands are gathered into contiguous scratch, then the dense pack/run path runs unchanged —
 * so numerics are identical to ork_bmm_*. Same dtype/shape rules. Returns 0/ok, <0 on error. */
typedef struct { long as_m, as_k, bs_k, bs_n, cs_m, cs_n, abs, bbs, cbs; } ork_bmm_strides;
int          ork_i8_bmm_strided  (ork_npu *ctx, int nbatch, int M, int K, int N, const int8_t  *A, const int8_t  *B, int32_t *C, const ork_bmm_strides *s);
int          ork_i4_bmm_strided  (ork_npu *ctx, int nbatch, int M, int K, int N, const int8_t  *A, const int8_t  *B, int32_t *C, const ork_bmm_strides *s);
int          ork_bmm_fp16_strided(ork_npu *ctx, int nbatch, int M, int K, int N, const ork_f16 *A, const ork_f16 *B, float   *C, const ork_bmm_strides *s);

/* Math utilities for caller-driven quantization/transformations */
void         ork_fwht_norm(float *v, int n);

/* ---- FLOOR-DECOMP diagnostics (submit-floor RE) --------------------------------------------------
 * Decompose the per-submit floor. ork_npu_floor_timing returns, accumulated since the last reset:
 *   ioctl_us    = total wall-clock time spent INSIDE the blocking SUBMIT ioctl (kernel job setup +
 *                 register programming + NPU execution + completion-wait).
 *   hw_us       = SUM of the kernel-reported sub.hw_elapse_time across those ioctls (the hardware's own
 *                 NPU-busy view). Comparing hw_us to ioctl_us splits real-NPU-compute from driver
 *                 dispatch/wait overhead — the core of the poll-granularity hypothesis.
 *   hw_raw_last = the last raw sub.hw_elapse_time value (to infer whether the kernel reports ns or us).
 *   n           = number of SUBMIT ioctls counted. */
void         ork_npu_floor_timing(double*ioctl_us,double*hw_us,long long*hw_raw_last,long*n);
void         ork_npu_floor_reset(void);

/* ---- MODE-TRANSITION RE hooks (tools/mode_probe.c) -----------------------------------------------
 * Standalone SDP ops (ork_f16_npu_ewmul/_i16, ork_i16_npu_exp/silu_i16/…) reprogram the NPU pipeline but
 * do NOT update the driver's cached matmul mode state (last_dt/warmed), so a following same-dtype matmul
 * skips its reset/re-warm and can wedge (errno=110). These expose the two candidate mitigations:
 *   ork_npu_mode_invalidate — clear the cached mode state only (next matmul re-warms itself; no HW reset).
 *   ork_npu_mode_reset      — explicit HW ACT_RESET + invalidate (heavyweight, always safe). */
void         ork_npu_mode_invalidate(ork_npu *ctx);
void         ork_npu_mode_reset(ork_npu *ctx);

/* FUSED SSD-SCAN MATMUL BENCH (SSM-on-NPU RE): chain one Mamba-2/SSD layer's group-batched scan matmuls
 * into ONE PC-chained submit with resident all-ones operands, vs the same matmuls as N separate submits.
 * Measures the per-submit-floor amortization of the fused on-NPU scan. fused_us/persub_us = per-iter wall;
 * ok_out=1 if the fused chain is bit-correct (every output==K). Returns 0/ok, <0 error. Board only. */
/* (b) layout probe: one fp16 matmul via the raw-synth fused-chain mechanism with ROW-MAJOR operands
 * A[M,K],B[K,N]->C[M,N] fp32 — decides if a real-operand fused SSD scan can stage row-major directly.
 * K%32,N%16. 0/ok,<0. rk3588 diagnostic. */
int          ork_f16_ssd_probe_rawmm(ork_npu *c,int M,int K,int N,const ork_f16 *A,const ork_f16 *B,float *C);
/* (b) fused-mm probe: one fp16 matmul via the fused-chain mechanism with B PACKED (ork_f16_mm_pack tiling) +
 * A row-major + C dense — tests whether the real-operand fused SSD chain can reuse ork_f16_mm_pack for B. 0/ok,<0. */
int          ork_f16_ssd_probe_fusedmm(ork_npu *c,int M,int K,int N,const ork_f16 *A,const ork_f16 *B,float *C);
/* FUSED batched fp16 GEMM: drop-in for ork_bmm_fp16 (nbatch matmuls, both operands dynamic) but chains all
 * nbatch matmuls into ONE PC-chained submit — amortizes the ~48us/submit floor across the batch (the SSD
 * scan per-stage H-batch). Packed-B (ork_f16_mm_pack) + row-major-A + dense-C; numerically identical to
 * ork_bmm_fp16. SINGLE-CORE (a PC chain runs on one core). Single-slice (K<=ks, N<=nmax), nb<=64. 0/ok,<0. */
int          ork_bmm_fp16_fused(ork_npu *c,int nb,int M,int K,int N,const ork_f16 *A,const ork_f16 *B,float *C);
/* STREAMED batched fp16 GEMM: drop-in for ork_bmm_fp16 but dispatches the nbatch INDEPENDENT matmuls
 * round-robin across ALL NPU cores (fp16 twin of ork_i8_mm_run_stream) — each core pulls the next matmul
 * and runs a single-core submit on itself (no barrier; CPU-prep of op N+1 overlaps NPU of op N). For the
 * SSD scan's per-stage H independent matmuls: ~3-5x the single-core chain. Packed-B + row-major-A + dense-C,
 * numerically identical to ork_bmm_fp16. Single-slice, nb>=1. 0/ok,<0. */
/* M ENVELOPE (both stream entrypoints): one fp16 program is only correct up to a measured row
 * ceiling set by the CBUF bank split (see orki_f16_mcap in src/npu/f16/regcmd.c) — e.g. 352 @K=512,
 * 176 @K=1024, 256 @K=128, 16384/K for non-pow2 K. A larger M is M-TILED internally, so any M is
 * valid when every task carries the SAME M. Tasks with DIFFERING M above the ceiling cannot be
 * tiled as a batch and are REFUSED (-2) rather than miscomputed. */
typedef struct { ork_w *w; int M; const ork_f16 *A; float *C; } ork_mm_task_f16;
int          ork_f16_mm_run_stream(ork_npu *c, int S, const ork_mm_task_f16 *tasks);
/* CHAINED-MULTICORE fp16 stream: PC-chains each core's round-robin-assigned matmuls into ONE task_number>1
 * submit (amortizes the ~48us submit floor over many programs) while keeping 3-core parallelism. Same
 * task/operand semantics as run_stream_f16; single-slice fp16 (K%32,N%16). For the on-NPU SSM scan stages. */
int          ork_f16_mm_run_stream_chain(ork_npu *c, int S, const ork_mm_task_f16 *tasks);
int          ork_bmm_fp16_stream(ork_npu *c,int nb,int M,int K,int N,const ork_f16 *A,const ork_f16 *B,float *C);
/* (b) mixed-precision chain probe: FP16 matmul task + INT16 silu-SDP task in ONE submit, both validated —
 * confirms fp16 & int16 tasks coexist in a PC-chain (the fused SSD scan's matmul+elementwise mix). 0/ok,<0. */
int          ork_ssd_probe_mixchain(ork_npu *c,int *mm_ok,int *silu_ok,double *us);
/* SSM_SCAN (Mamba-2) on the NPU — the ggml-ork GGML_OP_SSM_SCAN kernel. Chunked mode-5 scan (matmul spine
 * on the NPU pooled 3-core stream, elementwise on CPU). Matches ggml_compute_forward_ssm_scan_f32:
 * softplus(dt), scalar decay A{nh}, NO D skip, output y (x-shaped) + s_new. Contiguous ggml layout:
 * s/s_new{nc,nr,nh,ns} x/y{nr,nh,nt,ns} dt{nh,nt,ns} A{nh} B/C{nc,ng,nt,ns}. nc%32,nr%16,nh%ng==0. 0/ok,<0. */
int          ork_ssm_scan_f32(ork_npu *c,int nc,int nr,int nh,int ng,int nt,int ns,
                              const float *s0,const float *x,const float *dt,const float *A,
                              const float *B,const float *C,float *y,float *s_new);
int          ork_ssd_fused_scan_bench(ork_npu *c,int H,int P,int Nst,int G,int CS,int NC,int iters,int dtype,int perhead,
                                      double *fused_us,double *persub_us,int *ok_out);  /* dtype:1=int8,0=fp16; perhead:1=fp16-stable per-head Y_diag */

#endif /* ORK_BMM_H */
