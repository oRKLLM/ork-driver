/* ork/seq.h — Heterogeneous op-sequence scheduler
 *
 * The op/composite vocabulary, the validated op->op chaining LOOKUP table (never an
 * algorithm), and ork_submit_seq, the one generic submit surface over them.
 *
 * Part of the public API. Do NOT include this directly — include <ork_npu.h>, which is the
 * umbrella and the only supported entry point; these parts are a readability split of it
 * (ork_npu.h was 1519 lines) and their boundaries may move. Types live in ork_npu.h above
 * the includes, so this header is not self-contained by design. */
#ifndef ORK_SEQ_H
#define ORK_SEQ_H
/* ---- Heterogeneous op-sequence scheduler (ork_submit_seq) ----------------------------------------
 * Ingest a mixed sequence of NPU ops (any precision, matmul or SDP/activation) and run it correctly by
 * routing each op to the ONE execution model it is reliable on. Different op types are reliable on
 * different models (a hard, exhaustively-established finding): int8 AND fp16 matmul with conforming K
 * (K%512==0 && K<=4096, M<=64, Sn==1; fp16 also M*K<=32768) are bit-exact on the thread-free HW-chain
 * DOORBELL (ork_dyn_begin_mc, NONBLOCK poll) — fp16 requires host (malloc) A, the same convention int8
 * uses. int4, non-conforming int8/fp16, and SDP/activation ops are NOT doorbell-reliable but ARE reliable
 * on the SW-chain / thread-pool + blocking-completion model (run_stream_f16 / run_stream_i4 / the SDP fns).
 *
 * The scheduler batches maximal runs of consecutive HW-chainable ops into ONE doorbell submit and BREAKS
 * the chain to the SW model at every op that isn't. A doorbell run is ONE dtype + ONE domain (begin_mc's
 * requirement), so the run also breaks at a dtype change (i8<->f16) — each dtype gets its own doorbell and
 * begin_mc's internal ork_npu_enter fires the mode transition at the boundary. Each break's precision/chain
 * mode transition is handled by the driver's table-driven ork_npu_enter/XSPEC layer — so op classification
 * is DATA (a per-op-kind row of {hw, marker, XSPEC profile, ork_chain_kind, dispatch}), not branches.
 * Adding a new precision/op is one more row (an hw=1 row rides the doorbell when seq_hw_ok() accepts it and
 * falls back to its SW dispatch fn otherwise); the scheduler loop is unchanged regardless of how many kinds.
 *
 * PHASE 1 (now): CORRECTNESS + RELIABILITY of a mixed sequence across the HW<->SW transitions. No
 * CPU/NPU overlap yet (that is phase 2 — see tools/test_submit_seq.c). Ops execute in order. */
typedef enum {
    ORK_OP_MM_I8 = 0,   /* int8 matmul:  w=DT_I8 weight, A int8[M,K], C int32[M,N]  (HW doorbell if K conforms) */
    ORK_OP_MM_F16,      /* fp16 matmul:  w=DT_F16 weight, A fp16[M,K] (HOST mem), C fp32[M,N]  (HW doorbell if K conforms & M*K<=32768, else SW run_stream_f16) */
    ORK_OP_MM_I4,       /* int4 matmul:  w=DT_I4 weight, A int8[M,K] (HOST mem), C int32[M,N]  (HW doorbell if M==1 & single-slice, else SW run_stream_i4) */
    ORK_OP_SILU_F16,    /* fp16 SiLU activation (SDP): A fp16[M,N] -> C fp16[M,N]   (SW SDP; needs LUT — TODO row) */
    ORK_OP_EWMUL_F16,   /* fp16 elementwise mul (SDP): A*B fp16[M,N] -> C fp16[M,N] (SW: ork_f16_npu_ewmul) */
    ORK_OP_SILU_I8,     /* int8 SiLU (SDP): A i8[M,N] -> C i8[M,N] (in_scale,out_scale; SW: ork_i8_npu_silu) */
    ORK_OP_GELU_I8,     /* int8 GELU (SDP): A i8[M,N] -> C i8[M,N] (in_scale,out_scale; SW: ork_i8_npu_gelu) */
    ORK_OP_EWMUL_I8,    /* int8 elementwise mul (SDP): A*B i8[M,N] -> C i8[M,N] (mult,shift; SW: ork_i8_npu_ewmul) */
    ORK_OP_ADD_I8,      /* int8 elementwise add (SDP): A+B i8[M,N] -> C i8[M,N] (a_scale=in,b_scale,out_scale; SW: ork_i8_npu_add) */
    ORK_OP_ADD_F16,     /* fp16 elementwise add (SDP): A+B f16[M,N] -> C f16[M,N] (SW: ork_f16_npu_add) */
    ORK_OP_SILU_I16,    /* int16 SiLU (SDP): A i16[M,N] -> C i16[M,N] (in_scale,out_scale; SW: ork_i16_npu_silu) — the ACCURATE higher-precision SiLU (fp16 SiLU is not viable on this NPU: fused=garbage-PPL, standalone SDP=broken) */
    /* --- ops beyond the seq-scheduler subset (values >=11 append-only, ABI-stable). These are supported,
     * PROVEN ops (each backed by a working probe/example — see OPS_REGISTRY.md) that the SDK addresses by
     * enum + industry name (ork_op_name / ork_op_from_name); not all are seq-dispatchable (their SEQ_CLASS
     * row has fn=NULL). "op = the meaning"; the regcmd impl is chosen by ork_impl_mode. --- */
    ORK_OP_MM_I4_GROUPED,          /* float-grouped int4 matmul (ork_i4_mm_run_grouped)             example: i4 / int4_bench */
    ORK_OP_GELU_I16,               /* int16 GELU (SDP; ork_i16_npu_gelu)                            example: test_gelu */
    ORK_OP_RSQRT_I8,               /* int8 rsqrt (SDP; ork_i8_npu_rsqrt)                            example: test_gelu / op_profile */
    ORK_OP_EXP_I8,                 /* int8 exp (SDP; ork_i8_npu_exp)                                example: test_gelu / op_profile */
    ORK_OP_EXP_I16,                /* int16 exp (SDP; ork_i16_npu_exp)                              example: test_ssd_chunk_npu / mode_probe */
    ORK_OP_MUL_I16,                /* int16 elementwise mul (SDP; ork_i16_npu_ewmul) — EXPERIMENTAL example: test_ewmul_i16 */
    ORK_OP_ADD_I16,                /* int16 elementwise add (SDP; ork_i16_npu_add) — EXPERIMENTAL   example: test_add / add16_probe */
    ORK_OP_MUL_PERCHANNEL_I8,      /* per-channel scale int8 (SDP; ork_i8_npu_mul_perchan)          example: bs_scale_probe */
    ORK_OP_MUL_PERCHANNEL_F16,     /* per-channel scale fp16 (SDP; ork_f16_npu_mul_perchan)         example: bs_scale_probe */
    ORK_OP_MUL_PERCHANNEL_I16,     /* per-channel scale int16 (SDP; ork_i16_npu_mul_perchan)        example: bs_scale_probe */
    ORK_OP_MATMUL_PERCHANNEL_F16,  /* fp16 matmul -> per-channel scale (ork_f16_npu_mm_perchan)     example: mm_perchan_f16_probe */
    ORK_OP_REQUANTIZE_PERCHANNEL_I32,/* int32->int16 per-channel requant (ork_npu_requant_perchan_i32) — PARTIAL  example: requant_i32_probe */
    ORK_OP_SOFTMAX_F16,            /* fp16 softmax (replay; ork_f16_npu_replay_softmax)             example: softmax_probe / softmax_replay */
    ORK_OP_REDUCEMAX_I8,           /* int8 row-max reduction (ork_i8_npu_row_max)                   example: max_reduce_probe */
    ORK_OP_RESHAPE_F16,            /* fp16 reshape/permute (ork_f16_npu_replay_reshape)             example: reshape_probe */
    ORK_OP_ROPE_NEOX_F16,          /* fp16 NEOX RoPE (ork_f16_npu_rope_neox)                        example: rope_probe */
    ORK_OP_MATMUL_SILU_I8,         /* int8 matmul + fused SiLU output stage (ork_i8_mm_run_silu)    example: fused_silu_test */
    ORK_OP_MATMUL_REQUANT_I8,      /* int8 matmul + int8 requant output stage (ork_i8_mm_run_out8)  example: fused_ffn_probe */
    /* --- primitive ops the initial derivation missed (found by the 2026-07-22 completeness sweep) --- */
    ORK_OP_MATMUL_SILU_I32,        /* int8 matmul + fused SiLU, INT32 output (un-requantized; ork_i8_mm_run_silu32) example: silu32_check */
    ORK_OP_RMSNORM_F16,            /* fp16 RMSNorm — every transformer layer (ork_f16_npu_rmsnorm)  example: test_bmm */
    ORK_OP_L2NORM_F16,             /* fp16 L2 normalize (ork_f16_npu_l2norm)                        example: test_bmm */
    ORK_OP_RSQRT_I16,              /* int16 rsqrt (SDP; ork_i16_npu_rsqrt) — RMSNorm 1/√ + softmax 1/Σ  example: silu_i16 family (rsqrt/exp/gelu/silu int16 LUT) */
    ORK_OP_MM_F16_F16OUT,          /* fp16 matmul with CONTIGUOUS fp16 output (ork_f16_mm_run_f16out) — A1 bridge: feeds an fp16 SDP op with no f32→f16 narrow  example: f16out_probe */
    ORK_OP_MATMUL_I16OUT_I8,       /* int8 matmul with INT16 compact-linear output (ork_i8_mm_run_out16) — feeds an int16 SDP op resident, no PC-chain  example: i16out_seq_probe */
    ORK_OP_NKIND
} ork_seq_kind;
typedef ork_seq_kind ork_op;       /* canonical SDK op enum; ork_seq_kind is the historical name (the seq scheduler is one consumer) */

/* Execution mode selecting WHICH regcmd implementation of an op runs. The SDK submits an op/composite by
 * enum + mode; the driver resolves the regcmd (a hw-chained op and a standalone op share one ork_op value
 * but different regcmd byte templates). */
typedef enum {
    ORK_IMPL_STANDALONE = 0,   /* one op, its own submit (blocking or doorbell) */
    ORK_IMPL_HW_CHAINED,       /* op rides a HW PC-chain / doorbell-seq alongside others */
    ORK_IMPL_SW_CHAINED,       /* op runs in a software-broken sequence (ork_submit_seq SW path) */
    ORK_IMPL_NMODE
} ork_impl_mode;

/* Industry-standard op name <-> enum resolution (SDK-exported). Names follow ONNX where a primitive exists
 * (matmul/mul/add/softmax/gelu/exp/reshape/reducemax) and community LLM conventions otherwise (silu/rsqrt/
 * rope/requantize/perchannel), dtype as a C-identifier suffix (_i8/_i16/_i4/_f16). */
const char  *ork_op_name(ork_op k);              /* enum -> "silu_i16"; NULL if out of range */
ork_op       ork_op_from_name(const char *name); /* "silu_i16" -> enum; ORK_OP_NKIND if unknown */
const char  *ork_impl_mode_name(ork_impl_mode m);/* enum -> "hw_chained"; NULL if out of range */

/* Composite (multi-op) primitives — one submit combining several ops. Supported set derived from PRE-SESSION
 * examples (origin/main) only; DEAD/unvalidated chains (chain_mm_perchan_i16 = hangs, chain_gatesilu_i16 =
 * no pre-session example) are intentionally absent. */
/* NAMING CONVENTION (hybrid — this NPU is not industry-standard and has severe limitations, so we borrow
 * standards where they map and use NPU-specific qualifiers where they don't):
 *   - WEIGHTED ops (matmul / FFN): WxAy quantization notation (industry-standard) — W<weight-bits>A<act-bits>,
 *     e.g. w8a8 (int8/int8), w16a16 (fp16), w4a4 (int4). This is the speed/quality axis.
 *   - MIXED precision: append a component qualifier naming the higher-precision (sensitive) part, e.g.
 *     `_f16gate` (fp16 gate matmul while the rest is int8), `_i16silu` (int16 SiLU). This is how a
 *     speed-for-quality optimization is expressed: base WxAy + which component is kept precise.
 *   - SDP / elementwise ops (no weights): WxAy does not apply → keep the dtype suffix (_i8/_i16/_f16).
 *   - Non-op mechanisms named for what they do (graph_replay), not the internal fn (was "replay"/"bmm"). */
typedef enum {
    ORK_COMPOSITE_CHAIN_MATMUL_W8A8 = 0,      /* batch of independent int8 matmuls (ork_i8_mm_run_chain)              example: chain_gu_silu_probe */
    ORK_COMPOSITE_FFN_SWIGLU_W8A8,            /* int8 SwiGLU FFN inner (ork_i8_mm_run_chain_ffn)                     example: chain_gu_silu_probe */
    ORK_COMPOSITE_FFN_GATE_SILU_W8A8,         /* FFN, fused gate-SiLU output stage (ork_i8_mm_run_chain_gsilu)       example: chain_gu_silu_probe */
    ORK_COMPOSITE_FFN_GATE_SDPSILU_W8A8,      /* FFN, gate matmul + standalone SDP SiLU (ork_i8_mm_run_chain_sdpsilu) example: chain_gu_silu_probe */
    ORK_COMPOSITE_CHAIN_MATMUL_PERCHANNEL_W16A16,/* fp16 matmul -> per-channel scale chain (ork_f16_npu_chain_mm_perchan) example: chain_mm_perchan_f16_probe */
    ORK_COMPOSITE_SEQ,                        /* heterogeneous op sequence, any precision (ork_submit_seq)           example: test_submit_seq / sdp_chain_probe */
    /* --- mixed-precision / batched composites (added 2026-07-22) --- */
    ORK_COMPOSITE_MATMUL_SILU_W16A16_I16SILU, /* MIXED: fp16 matmul (w16a16) + int16 SiLU (_i16silu) fused in ONE
                                               * PC-chain — the quality-path building block (fp16 on the sensitive
                                               * gate matmul, int16 SiLU). (ork_ssd_probe_mixchain)  example: mixchain_probe */
    ORK_COMPOSITE_MATMUL_BATCHED_FUSED_W16A16,/* fp16 fused batched matmul (attention / SSD A.V)
                                               * (ork_bmm_fp16_fused)               example: test_bmm_fused / ssd_fusedchain_probe */
    ORK_COMPOSITE_GRAPH_REPLAY_F16,           /* fp16 op-graph replay mechanism (ork_f16_npu_replay_full) example: replay_f16_full_test */
    ORK_COMPOSITE_NKIND
} ork_composite;
const char   *ork_composite_name(ork_composite k);              /* enum -> "ffn_swiglu_w8a8"; NULL if out of range */
ork_composite ork_composite_from_name(const char *name);        /* "ffn_swiglu_w8a8" -> enum; ORK_COMPOSITE_NKIND if unknown */

/* regcmd -> op binding: which ork_op (and impl mode) a REGCMD_* byte template implements. Every REGCMD_*
 * base template in the driver MUST have a binding (enforced by check_registry.sh — 0 orphan regcmds), so no
 * regcmd exists that isn't the implementation of an exported op. Returns ORK_OP_NKIND if the name is unbound. */
ork_op ork_regcmd_op(const char *regcmd_name, ork_impl_mode *mode_out);

/* --- Deterministic op->op chaining: a validated LOOKUP, never an algorithm. ------------------------------
 * How two consecutive ops may combine is decided by a table, not by heuristics. Every (from,to) permutation
 * is exhaustively validated on-silicon (tools/mode_probe transition matrix) and its result baked into
 * ork_chain_table[from][to]. The chain assembler asks ork_chain_lookup(from,to) and obeys it — so a
 * transition that wedges the NPU is DISALLOWed structurally, not discovered at runtime (this replaces the
 * heuristic get_node_chain_type/seq_hw_ok path that let an unvalidated transition hang). */
typedef enum {
    ORK_CHAIN_DISALLOW = 0,  /* do NOT chain with the CURRENT driver config; run as independent submits with a
                              * mode reset between (always correct — the safe baseline, and the DEFAULT for any
                              * pair not yet validated). IMPORTANT: a DISALLOW from a mode_probe fix=none wedge
                              * means the transition is unsafe with the transition config we CURRENTLY apply
                              * (ork_npu_enter/XSPEC profile + regcmd) — it is NOT necessarily a hardware limit.
                              * Many DISALLOW pairs are a MISSING/WRONG transition template: e.g. matmul->int8-SDP
                              * wedges as separate submits (no int8-SDP-tuned XSPEC profile), yet the SAME
                              * transition is HW-safe in a PC-chain (FFN gate->silu). Such pairs are candidates
                              * for a transition-template fix — test mode_probe fix=RESET; if safe, add that
                              * reset/profile to ork_npu_enter and UPGRADE the cell to SW. Treat DISALLOW as
                              * "unsupported by our config (yet)", not "the NPU can't". */
    ORK_CHAIN_SW,            /* SW-chain: sequence the two ops without a HW handoff (validated safe to
                              * sequence, but not HW-chainable). Fallback for pairs that can't HW-chain. */
    ORK_CHAIN_HW,            /* HW-chain: the two ops ride ONE PC-chain / doorbell-seq submit (validated
                              * safe AND fast). */
    ORK_CHAIN_NRULE
} ork_chain_rule;
ork_chain_rule ork_chain_lookup(ork_op from, ork_op to);   /* validated transition rule; DISALLOW if unknown */

/* SINGLE SOURCE for op->op chain rules. One X-macro emits BOTH the runtime table (ork_ops.c) AND named enum
 * constants ORK_CR__<from>__<to> — because C cannot index a const array in a constant expression, compile-time
 * checks need the rules as enum constants (which ARE constant expressions). List every VALIDATED pair (HW/SW
 * and explicit DISALLOW); unlisted pairs default to DISALLOW at runtime. Each pair is validated by the
 * exhaustive op->op campaign (tools/mode_probe); the seeds here are the ones already proven in OPS_REGISTRY. */
/* HW pairs = ride one PC-chain (proven by the chain probes). SW pairs = safe to SEQUENCE as separate
 * submits (proven by the mode_probe 7-op transition campaign, 2026-07-21: all 49 ordered pairs among
 * {matmul_f16, matmul_i8, exp_i16, silu_i16, mul_i16, mul_f16, add_f16} ran fix=none with no wedge on a
 * fresh process each, board recovered). NOTE: matmul_i8->mul_i16 is SW (sequenceable) even though its *HW*
 * 2-input-SDP chain HANGS (chain_mm_perchan_probe) — DISALLOW means "can't even sequence", which the
 * campaign found for NONE of these pairs. Pairs among the other (uncampaigned) ops stay unlisted = DISALLOW. */
#define ORK_CHAIN_LIST(X) \
    /* --- HW-chainable (one PC-chain) --- */ \
    X(ORK_OP_MM_I8,    ORK_OP_MM_I8,    ORK_CHAIN_HW)  /* matmul->matmul run (adjacent fusion; chain_gu_silu_probe) */ \
    X(ORK_OP_MM_I8,    ORK_OP_SILU_I8,  ORK_CHAIN_HW)  /* FFN: gate matmul -> SiLU */ \
    X(ORK_OP_SILU_I8,  ORK_OP_MM_I8,    ORK_CHAIN_HW)  /* FFN: SiLU -> up matmul */ \
    X(ORK_OP_MM_I8,    ORK_OP_EWMUL_I8, ORK_CHAIN_HW)  /* FFN: up matmul -> GLU mul */ \
    X(ORK_OP_EWMUL_I8, ORK_OP_MM_I8,    ORK_CHAIN_HW)  /* FFN: GLU mul -> down matmul */ \
    X(ORK_OP_MM_F16,   ORK_OP_MUL_PERCHANNEL_F16, ORK_CHAIN_HW) /* attn A.V -> per-channel scale (mm_perchan_f16_probe) */ \
    /* --- SW-chainable (separate-submit safe; mode_probe campaign 2026-07-21, all fix=none SAFE) --- */ \
    X(ORK_OP_MM_F16,   ORK_OP_MM_F16,   ORK_CHAIN_SW) X(ORK_OP_MM_F16,   ORK_OP_MM_I8,    ORK_CHAIN_SW) \
    X(ORK_OP_MM_F16,   ORK_OP_EXP_I16,  ORK_CHAIN_SW) X(ORK_OP_MM_F16,   ORK_OP_SILU_I16, ORK_CHAIN_SW) \
    X(ORK_OP_MM_F16,   ORK_OP_MUL_I16,  ORK_CHAIN_SW) X(ORK_OP_MM_F16,   ORK_OP_EWMUL_F16,ORK_CHAIN_SW) \
    X(ORK_OP_MM_F16,   ORK_OP_ADD_F16,  ORK_CHAIN_SW) \
    X(ORK_OP_MM_I8,    ORK_OP_MM_F16,   ORK_CHAIN_SW) X(ORK_OP_MM_I8,    ORK_OP_EXP_I16,  ORK_CHAIN_SW) \
    X(ORK_OP_MM_I8,    ORK_OP_SILU_I16, ORK_CHAIN_SW) X(ORK_OP_MM_I8,    ORK_OP_MUL_I16,  ORK_CHAIN_SW) \
    X(ORK_OP_MM_I8,    ORK_OP_EWMUL_F16,ORK_CHAIN_SW) X(ORK_OP_MM_I8,    ORK_OP_ADD_F16,  ORK_CHAIN_SW) \
    X(ORK_OP_EXP_I16,  ORK_OP_MM_F16,   ORK_CHAIN_SW) X(ORK_OP_EXP_I16,  ORK_OP_MM_I8,    ORK_CHAIN_SW) \
    X(ORK_OP_EXP_I16,  ORK_OP_EXP_I16,  ORK_CHAIN_SW) X(ORK_OP_EXP_I16,  ORK_OP_SILU_I16, ORK_CHAIN_SW) \
    X(ORK_OP_EXP_I16,  ORK_OP_MUL_I16,  ORK_CHAIN_SW) X(ORK_OP_EXP_I16,  ORK_OP_EWMUL_F16,ORK_CHAIN_SW) \
    X(ORK_OP_EXP_I16,  ORK_OP_ADD_F16,  ORK_CHAIN_SW) \
    X(ORK_OP_SILU_I16, ORK_OP_MM_F16,   ORK_CHAIN_SW) X(ORK_OP_SILU_I16, ORK_OP_MM_I8,    ORK_CHAIN_SW) \
    X(ORK_OP_SILU_I16, ORK_OP_EXP_I16,  ORK_CHAIN_SW) X(ORK_OP_SILU_I16, ORK_OP_SILU_I16, ORK_CHAIN_SW) \
    X(ORK_OP_SILU_I16, ORK_OP_MUL_I16,  ORK_CHAIN_SW) X(ORK_OP_SILU_I16, ORK_OP_EWMUL_F16,ORK_CHAIN_SW) \
    X(ORK_OP_SILU_I16, ORK_OP_ADD_F16,  ORK_CHAIN_SW) \
    X(ORK_OP_MUL_I16,  ORK_OP_MM_F16,   ORK_CHAIN_SW) X(ORK_OP_MUL_I16,  ORK_OP_MM_I8,    ORK_CHAIN_SW) \
    X(ORK_OP_MUL_I16,  ORK_OP_EXP_I16,  ORK_CHAIN_SW) X(ORK_OP_MUL_I16,  ORK_OP_SILU_I16, ORK_CHAIN_SW) \
    X(ORK_OP_MUL_I16,  ORK_OP_MUL_I16,  ORK_CHAIN_SW) X(ORK_OP_MUL_I16,  ORK_OP_EWMUL_F16,ORK_CHAIN_SW) \
    X(ORK_OP_MUL_I16,  ORK_OP_ADD_F16,  ORK_CHAIN_SW) \
    X(ORK_OP_EWMUL_F16,ORK_OP_MM_F16,   ORK_CHAIN_SW) X(ORK_OP_EWMUL_F16,ORK_OP_MM_I8,    ORK_CHAIN_SW) \
    X(ORK_OP_EWMUL_F16,ORK_OP_EXP_I16,  ORK_CHAIN_SW) X(ORK_OP_EWMUL_F16,ORK_OP_SILU_I16, ORK_CHAIN_SW) \
    X(ORK_OP_EWMUL_F16,ORK_OP_MUL_I16,  ORK_CHAIN_SW) X(ORK_OP_EWMUL_F16,ORK_OP_EWMUL_F16,ORK_CHAIN_SW) \
    X(ORK_OP_EWMUL_F16,ORK_OP_ADD_F16,  ORK_CHAIN_SW) \
    X(ORK_OP_ADD_F16,  ORK_OP_MM_F16,   ORK_CHAIN_SW) X(ORK_OP_ADD_F16,  ORK_OP_MM_I8,    ORK_CHAIN_SW) \
    X(ORK_OP_ADD_F16,  ORK_OP_EXP_I16,  ORK_CHAIN_SW) X(ORK_OP_ADD_F16,  ORK_OP_SILU_I16, ORK_CHAIN_SW) \
    X(ORK_OP_ADD_F16,  ORK_OP_MUL_I16,  ORK_CHAIN_SW) X(ORK_OP_ADD_F16,  ORK_OP_EWMUL_F16,ORK_CHAIN_SW) \
    X(ORK_OP_ADD_F16,  ORK_OP_ADD_F16,  ORK_CHAIN_SW) \
    /* --- campaign 2 (2026-07-21, int8-SDP ops added to mode_probe): fp16-matmul -> the remaining int16 SDP
     *     ops are SW-safe; matmul -> int8-SDP HARD-WEDGES (validated: MM_F16->SILU_I8 hung, MM_F16->GELU_I8
     *     hard-wedged the NPU, MM_I8->SILU_I8 hung). matmul->int8-SDP is a mode-switch hazard for BOTH fp16
     *     and int8 matmul — whereas matmul->int16-SDP is safe. The remaining matmul->int8-SDP pairs are
     *     DISALLOW-by-default (unlisted); the ones below are listed explicitly to document the VALIDATED
     *     wedge (so ORK_ASSERT_CHAIN_STEP reports "DISALLOWed" rather than "unvalidated"). --- */ \
    X(ORK_OP_MM_F16,   ORK_OP_GELU_I16, ORK_CHAIN_SW) X(ORK_OP_MM_F16,   ORK_OP_ADD_I16,  ORK_CHAIN_SW) \
    X(ORK_OP_MM_F16,   ORK_OP_SILU_I8,  ORK_CHAIN_DISALLOW) /* fp16-matmul -> int8-SDP: hung (NO_OUTPUT); no HW chain proven */ \
    X(ORK_OP_MM_F16,   ORK_OP_GELU_I8,  ORK_CHAIN_DISALLOW) /* fp16-matmul -> int8-SDP: HARD-WEDGE (power-cycle) */
    /* NOTE: MM_I8->SILU_I8 (int8-matmul -> int8-SiLU) is listed HW above (the FFN gate->silu PC-chain,
     * chain_gu_silu_probe). mode_probe's SEPARATE-submit MM_I8->SILU_I8 HUNG, but that's the SW path — the op
     * IS HW-chainable in one submit (how the FFN uses it), so HW stands. Separate-submit int8-matmul->int8-SDP
     * is NOT SW-safe (do not downgrade the FFN's HW pair to SW). */

/* Named enum constants per validated pair — usable in _Static_assert (unlike a const-array index). */
enum {
#define X(f, t, r) ORK_CR__##f##__##t = (r),
    ORK_CHAIN_LIST(X)
#undef X
    ORK_CR__SENTINEL = 0
};
/* Compile-time rule for a statically-known transition. Undeclared for an unlisted (never-validated) pair ->
 * referencing it is a compile error, so a static chain over an unvalidated transition won't build. */
#define ORK_CHAIN_RULE(f, t) ORK_CR__##f##__##t
/* Assert a statically-declared chain STEP is chainable (HW or SW). Fails to compile on DISALLOW/unvalidated —
 * this is how the SDK's fixed composites are validated at build time (no runtime failing check needed). */
#define ORK_ASSERT_CHAIN_STEP(f, t) \
    _Static_assert(ORK_CHAIN_RULE(f, t) != ORK_CHAIN_DISALLOW, \
        "chain step " #f " -> " #t " is DISALLOWed or unvalidated")

/* Single generic submit surface (declared after ork_seq_op, below): the SDK addresses ops by enum + mode;
 * the driver resolves the regcmd impl and validates each transition via ork_chain_lookup. See ork_submit /
 * ork_submit_chain after the ork_seq_op definition. */
typedef struct {
    ork_seq_kind kind;
    ork_w      *w;                 /* matmul weight (NULL for weightless SDP ops) */
    int         M, N;              /* M rows; N is taken from w for matmuls, supplied here for weightless SDP ops */
    const void *A;                 /* primary input (int8/fp16 A, or SDP operand) */
    const void *B;                 /* second SDP operand (ewmul/add); NULL otherwise */
    void       *C;                 /* output */
    double      in_scale, out_scale;  /* SDP scales: silu/gelu in/out; add uses in_scale as a_scale + b_scale/out_scale */
    double      b_scale;              /* SDP add: b operand scale (a_scale = in_scale) */
    int         mult, shift;          /* SDP ewmul_i8 requant (out = clamp(A*B*mult>>shift)) */
    int         group;                /* dependency grouping (default 0 = ungrouped, legacy per-op scheduling).
                                       * >0: CONTIGUOUS ops sharing a group id form ONE sequential chain (kept on
                                       * one core, in order); a group-id change starts a new INDEPENDENT chain.
                                       * A contiguous run of group>0 ops rides ork_i8_dyn_begin_seq_mc — SDP ops
                                       * stay in the doorbell chain instead of forcing a blocking SW break. The
                                       * run's terminal op (each group's last) must be a matmul (sentinel). */
} ork_seq_op;
/* Run the n-op sequence in order, HW-batching + SW-breaking as above. 0/ok, -1 a submit failed/wedged,
 * -2 bad args, -3 an op-kind whose dispatch is not yet wired (documented TODO row, e.g. SILU_F16). */
int          ork_submit_seq(ork_npu *ctx, const ork_seq_op *ops, int n);
/* Generic enum-driven submit surface (see the ork_op / ork_impl_mode / ork_chain_lookup design above).
 * ork_submit runs ONE op via its dispatch (mode advisory for a single op — chaining is ork_submit_chain).
 * ork_submit_chain PARTITIONS the sequence at every DISALLOW transition (those ops run as separate submits,
 * never chained) and routes each maximal run through ork_submit_seq — it never FAILS on a transition (a
 * disallowed pair is split, not rejected; correctness is guaranteed by the table + the compile-time composite
 * asserts, since orkd + SDK ship together). Returns: 0 ok; -3 an op has no generic dispatch (use its typed
 * ork_npu_* entry); -2 bad args/op; -1 a submit failed. */
int          ork_submit(ork_npu *ctx, ork_op op, ork_impl_mode mode, const ork_seq_op *args);
int          ork_submit_chain(ork_npu *ctx, const ork_seq_op *ops, int n);
#endif /* ORK_SEQ_H */
