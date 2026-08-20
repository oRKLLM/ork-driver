/* ork/chain.h — Static chain entrypoints, round-robin dispatch, streams, and async wrappers
 *
 * Whole-subgraph submits (the FFN chain, attention chains, per-layer), the concurrent
 * round-robin dispatch across cores, the SW round-robin streams, and the async wrappers.
 *
 * Part of the public API. Do NOT include this directly — include <ork_npu.h>, which is the
 * umbrella and the only supported entry point; these parts are a readability split of it
 * (ork_npu.h was 1519 lines) and their boundaries may move. Types live in ork_npu.h above
 * the includes, so this header is not self-contained by design. */
#ifndef ORK_CHAIN_H
#define ORK_CHAIN_H
/* Heterogeneous single-core NONBLOCK chain: run ONE group of int8 ops (matmul + int8 SDP, e.g. EWMUL_I8) as one
 * PC-chain; terminal MUST be a matmul (its int32 sentinel gates completion). Returns a handle (drain with
 * ork_dyn_seq_end) or NULL if ineligible (non-int8 / M>64 / non-conforming K / terminal not a matmul / kind not
 * yet supported) — caller then runs the ops via the SW break. The scheduler slices a sequence into groups. */
ork_dyn_chain *ork_i8_dyn_begin_seq(ork_npu *ctx, int n, const ork_seq_op *ops);
/* Multi-core: groups = contiguous op slices [gstart[g],gstart[g+1]); gstart[0]=0, gstart[ngroups]=n. Each group
 * is a dependent chain (terminal op = matmul); INDEPENDENT groups are load-balanced whole onto nc cores (nc<=0
 * = all) and run in parallel. Drain with ork_dyn_seq_end (polls every core's terminal). NULL if ineligible. */
ork_dyn_chain *ork_i8_dyn_begin_seq_mc(ork_npu *ctx, int n, const ork_seq_op *ops, int ngroups, const int *gstart, int nc);
int          ork_dyn_seq_end(ork_dyn_chain *h);   /* poll every core's terminal sentinel + per-op copy-back; 0/ok,-1 timeout,-2 bad */

/* Like ork_i8_mm_run_chain but task[gate_task] gets a FUSED int8 SiLU output stage (set_i8_silu): its C
 * receives int8 silu(gate) (M*N bytes) instead of int32; the silu LUT is streamed to SDP SRAM once before
 * the chain. Chains [gate*silu -> up -> ...] in ONE submit. lut/params as ork_i8_mm_run_silu (build with
 * ork_mm_silu_build_lut). Single M-tile per task for now (M <= chain mcap). 0/ok,-1 wedge,-2 dims. */
int          ork_i8_mm_run_chain_gsilu(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks, int gate_task,
                                       int r_mult, int r_shift, unsigned out_bias, unsigned idx_off,
                                       unsigned cfg4068, const short *lut, int nlut);
/* OPTION B: task[sdp_task] is a STANDALONE int8 silu-SDP op reading task[sdp_task-1]'s (gate) output via
 * aliased buffers (the vendor matmul->SDP pattern); the gate task emits int8 (set_i8_out8, requant
 * gate_mult/gate_shift). The silu LUT for (in_scale,out_scale) is built internally (same as ork_i8_npu_silu).
 * tasks[sdp_task].C gets int8 silu (M*N bytes). Single M-tile per task. 0/ok,-1 wedge,-2 dims,-3 SoC. */
int          ork_i8_mm_run_chain_sdpsilu(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks, int sdp_task,
                                         int gate_mult, int gate_shift, double in_scale, double out_scale);

/* GENERAL heterogeneous FFN chain. Per-task op: kind 0=matmul(int32 out) 1=matmul(int8 out, requant
 * mult/shift) 2=silu-SDP 3=ewmul-SDP; SDP tasks read prior tasks' outputs by index in0/in1 (aliased). Chains
 * e.g. [gate(1) -> silu(2,in0=gate) -> up(1) -> glu(3,in0=silu,in1=up) -> down(0)] in ONE submit. Silu LUT for
 * (in_scale,out_scale) built internally. tasks[i].C gets that op's output. Single M-tile/task. 0/ok,-1,-2,-3. */
typedef struct { int kind; int in0, in1; int mult, shift; } ork_chain_op;
int          ork_i8_mm_run_chain_ffn(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks,
                                     const ork_chain_op *ops, double in_scale, double out_scale);
/* Same chain, but the kind-2 SDP task applies EXP (softmax numerator) — HW-chains [QK^T -> exp -> reduce] in
 * ONE submit, intermediates on-chip. Scores must be <=0 (post-max domain). 0/ok,-1 wedge,-2 dims,-3 SoC. */
int          ork_i8_mm_run_chain_ffn_exp(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks,
                                         const ork_chain_op *ops, double in_scale, double out_scale);
/* Same, but the fused exp bakes in a scalar max-subtract: e = exp((score-max_bias)*in_scale)/out_scale. A
 * max_bias >= every real score keeps args <=0 (int8 exp never saturates) and cancels in the softmax normalize —
 * makes the fused core correct on REAL (positive) scores without a live per-query max. 0.0 = plain exp. */
int          ork_i8_mm_run_chain_ffn_exp_biased(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks,
                                         const ork_chain_op *ops, double in_scale, double out_scale, double max_bias);
/* CONCURRENT round-robin: dispatch `nchains` homogeneous fused-exp chains across all NPU cores at once (one per
 * core, atomic work-stealing), each on its own per-core scratch. chains[i]=that chain's S[i]-task array; ops the
 * shared op graph; scales the shared exp requant. Prefill throughput (~3x). Cores must be WARM first. 0/ok,<0 err. */
int          ork_mm_run_chains_rr(ork_npu *ctx, int nchains, const ork_mm_task_i8 *const *chains, const int *S,
                                  const ork_chain_op *ops, double in_scale, double out_scale);
/* Same concurrent round-robin, but the fused exp bakes in a scalar max_bias (e=exp((score-max_bias)*in_scale)/
 * out_scale) so the chains are correct on REAL scores without a live max — fans N attention cores' fused
 * [QK^T->exp->reduce,e.V] chains across the NPU cores. Reloads the per-core device LUT on rebuild. 0/ok,<0 err. */
int          ork_mm_run_chains_rr_biased(ork_npu *ctx, int nchains, const ork_mm_task_i8 *const *chains, const int *S,
                                  const ork_chain_op *ops, double in_scale, double out_scale, double max_bias);
/* ORKD coalesced SwiGLU FFN: run the whole [gate->silu->up->glu->down] inner as ONE daemon-side HW-chained
 * submit (ORKD_FFN / orkd_ffn_i8) against the 3 ALREADY-RESIDENT weights (wg/wu/wd must be daemon-imported —
 * is_orkd — e.g. from the orkpack; no re-import). A = int8 activation [M,K]; out = int32 down output [M,Kd].
 * Per-stage fixed-point requant (gate/up/glu each a mult>>shift on the int32 accumulator -> int8 intermediate)
 * + the silu (in_scale,out_scale) LUT, exactly as orkd_ffn_i8 / ork_i8_mm_run_chain_ffn expect (see
 * orkd_ffn_probe for the scale model). Per-tensor int8 coalesced path (fast; one submit instead of per-op).
 * Returns -3 if no daemon or any weight is not resident (caller should fall back to fd-local run_chain_i8_ffn). */
int          ork_mm_ffn_orkd(ork_npu *ctx, ork_w *wg, ork_w *wu, ork_w *wd,
                             int M, int K, int Nff, int Kd,
                             int gate_mult, int gate_shift, int up_mult, int up_shift, int glu_mult, int glu_shift,
                             double in_scale, double out_scale,
                             const int8_t *A, int32_t *out);
/* ORKD fused attention core [QK^T->exp->reduce,e.V] (chainav) as ONE daemon submit against 3 resident weights:
 * wkt=K^T[Kp,Nk], wones=ones[Nk,32], wv=V[Nk,dv]. Q=[Nq,Kp] int8. Sigma=[Nq,32] + av=[Nq,dv] int32 out (attn =
 * av/Sigma, caller normalizes). r_mult/r_shift = QK^T score requant; in/out_scale = exp LUT. -3 if not resident. */
int          ork_mm_attn_orkd(ork_npu *ctx, ork_w *wkt, ork_w *wones, ork_w *wv,
                              int Nq, int Nk, int Kp, int dv, int r_mult, int r_shift,
                              double in_scale, double out_scale, double max_bias,
                              const int8_t *Q, int32_t *Sigma, int32_t *av);
/* RR variant: nchains fused-attn chains (per-chain wkt/wv, shared wones) fanned across cores in ONE round-trip
 * through orkd. Q = nchains*Nq*Kp int8 (chain-major); Sigma = nchains*Nq*32, av = nchains*Nq*dv int32. All
 * weights daemon-resident. -3 if not resident / no daemon. */
int          ork_mm_attn_rr_orkd(ork_npu *ctx, int nchains, ork_w *const *wkt, ork_w *wones, ork_w *const *wv,
                              int Nq, int Nk, int Kp, int dv, int r_mult, int r_shift,
                              double in_scale, double out_scale, double max_bias,
                              const int8_t *Q, int32_t *Sigma, int32_t *av);
/* ORKD whole-decode-layer core: rmsnorm -> qkv -> q/k-norm+rope -> attn -> o -> +res -> ffn-norm -> gate/up
 * -> silu-glu -> down -> +res, as ONE op against 7 resident int8 weights (wq/wk/wv/wo, wg/wu=[D,Nff], wd=[Nff,D]).
 * CPU glue (rmsnorm/rope/attn/silu, all fp32) interleaved with the doorbell matmuls; x/Kc/Vc/x_out are fp32.
 * Transport-transparent: routes to the daemon (orkd_layer_i8) when ctx is an orkd client, else runs locally on
 * ctx's NPU — the orkd daemon's handler calls THIS on its own direct ctx, so lib and orkd share one compute core.
 * NOTE: decode-on-NPU is a MEASURED loss vs CPU (see wiki/OPS_REGISTRY); this exists for lib<->orkd parity and
 * correctness, not as a perf path. Returns 0/ok, -3 = bad args / weights not resident under orkd. */
struct ork_layer_dims { uint32_t D, H, Hkv, dk, dv, Nff, nkv, pos; double attn_scale, rope_base; };
int          ork_i8_mm_layer(ork_npu *ctx, const struct ork_layer_dims *d,
                             ork_w *wq, ork_w *wk, ork_w *wv, ork_w *wo, ork_w *wg, ork_w *wu, ork_w *wd,
                             const float *attn_norm, const float *q_norm, const float *ffn_norm,
                             const float *x, const float *Kc, const float *Vc, float *x_out);
int          ork_i4_mm_run_chain(ork_npu *ctx, int S, const ork_mm_task_i4 *tasks);
/* EXPERIMENTAL: int4 incremental-task batch (vendor task_number=N pattern) — one resident int4 weight,
 * M rows, task[0]=full + task[1..]=12-config incremental (advance only A/C; weight loaded once), ONE
 * submit. Not Hcap-capped (unlike ork_i4_batch stride-2). int8 A, int32 C, N<=64 (single N-block). */
/* Async round-robin stream: S independent int8 matmuls dispatched dynamically across NPU cores (pull
 * model, no barrier). For batches of independent matmuls (e.g. EAGLE-3 verification). 0/ok, -1/-2 err. */
int          ork_i8_mm_run_stream(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks);
/* SMALL-K int8 round-robin stream (int8 twin of run_stream_f16): S single-slice int8 matmuls (K%32,N%16,
 * NOT the K%512 full-K path) across NPU cores; A int8 [M,K] per task, C int32 [M,N]. For the on-NPU SSM
 * scan's per-head gram/output stages (tiny K) with 3-core batching. Caller quantizes A + dequants int32. */
int          ork_i8_mm_run_stream_sk(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks);
/* int4 (W4A4) async round-robin stream: S independent matmuls across NPU cores (a task's M rows become M
 * single-row regcmds PC-chained on its core). Weights single-slice (Sn==1 && Sk==1). 0/ok, -1/-2 err. */
int          ork_i4_mm_run_stream(ork_npu *ctx, int S, const ork_mm_task_i4 *tasks);

/* Async submit (CPU‖NPU overlap foundation), PATH-AGNOSTIC. The RKNPU submit ioctl blocks the caller
 * until the NPU job finishes; the NPU is single-stream (one queue), so async here means the BLOCKING
 * submit runs on a worker thread while the CALLING thread does independent CPU work, joining at the
 * dependency. This is a DISPATCH-level wrapper around the synchronous run functions, so it works for
 * fp16 (ork_f16_mm_run), int8 (ork_i8_mm_run), int4 (ork_i4_mm_run), and the chain/stream variants — same
 * numerics as the synchronous run (reused verbatim). Each launcher returns a handle immediately (NULL
 * on bad args → fall back to the matching synchronous run); ork_async_wait joins, returns the result
 * (0/ok, <0 err) and frees the handle. CONTRACT: keep at most ONE async job in flight and issue no
 * other ork_mm_* on the same ctx between launch and wait (only independent CPU work) — the NPU is
 * single-stream. The task arrays passed to the chain/stream launchers must stay valid until wait. */
typedef struct ork_async ork_async;
ork_async   *ork_mm_run_async        (ork_npu *ctx, ork_w *w, int M, const ork_f16 *A, float   *C);
ork_async   *ork_i8_mm_run_async     (ork_npu *ctx, ork_w *w, int M, const int8_t  *A, int32_t *C);
ork_async   *ork_i4_mm_run_async     (ork_npu *ctx, ork_w *w, int M, const int8_t  *A, int32_t *C);
ork_async   *ork_i8_mm_run_chain_async (ork_npu *ctx, int S, const ork_mm_task_i8 *tasks);
ork_async   *ork_i4_mm_run_chain_async (ork_npu *ctx, int S, const ork_mm_task_i4 *tasks);
ork_async   *ork_i8_mm_run_stream_async(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks);
ork_async   *ork_i4_mm_run_stream_async(ork_npu *ctx, int S, const ork_mm_task_i4 *tasks);
int          ork_async_wait(ork_async *h);
/* CPU the most recent async worker was placed on at entry (sched_getcpu), -1 if none/non-Linux.
 * Diagnostic + regression test for the worker-pinning pattern (worker lands on a big core). */
int          ork_npu_last_async_cpu(ork_npu *ctx);

#endif /* ORK_CHAIN_H */
