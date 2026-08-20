/* ork_op_manifest.h — single source of truth for the lib<->orkd op-parity contract.
 *
 * The ork-driver NPU op API exists in two forms that MUST stay in lockstep:
 *   - the in-process lib API  (ork_mm_* / ork_npu_* / ork_kv_* / ork_submit_seq), and
 *   - the orkd transport API   (orkd_*), which serializes the same op over the daemon socket.
 * The design intent is drop-in interoperability: any op you can do direct you can do through orkd, and vice
 * versa (orkd is a dev-time single-stream safety wrapper, not a separate feature set). When the two drift —
 * an op added to one side only, or a signature changed on one side — code silently loses parity.
 *
 * Each ORK_OP(base, lib, orkd) row names an op that must exist on BOTH sides. examples/test_api_parity.c walks
 * this table at COMPILE TIME (takes the address of both <lib> and <orkd>), so a missing/renamed symbol on
 * either side is a build error, and it additionally pins the high-risk coalesced/matmul ops to explicit
 * prototypes so a silent signature change trips the build too.
 *
 * The two prototypes are deliberately NOT byte-identical: the lib takes (ork_npu*, ork_w*) while orkd takes
 * (orkd_conn*, uint64_t id[, dims]) — a socket cannot carry a pointer, and the weight id does not encapsulate
 * K/N the way ork_w does. So this asserts CORRESPONDENCE (same op set; shapes pinned), not literal identity
 * (see AskUserQuestion decision, 2026-07-24). When you add an op to either API, add its twin + a row here. */
#ifndef ORK_OP_MANIFEST_H
#define ORK_OP_MANIFEST_H

#define ORK_OP_TABLE(X) \
  /* --- pack / import / free ------------------------------------------------------------------ */ \
  X(pack_i8,      ork_i8_mm_pack,       orkd_pack_i8)      \
  X(pack_f16,     ork_f16_mm_pack,          orkd_pack_f16)     \
  X(pack_i4,      ork_i4_mm_pack,       orkd_pack_i4)      \
  X(free,         ork_mm_free,          orkd_free_weight)  \
  /* --- matmul run --------------------------------------------------------------------------- */ \
  X(run_i8,       ork_i8_mm_run,        orkd_run_i8)       \
  X(run_f16,      ork_f16_mm_run,           orkd_run_f16)      \
  X(run_i4,       ork_i4_mm_run,        orkd_run_i4)       \
  X(run_chain_i8, ork_i8_mm_run_chain,  orkd_run_chain_i8) \
  X(submit_seq,   ork_submit_seq,       orkd_submit_seq)   \
  /* --- coalesced whole-op cores ------------------------------------------------------------- */ \
  X(ffn_i8,       ork_mm_ffn_orkd,      orkd_ffn_i8)       \
  X(attn_i8,      ork_mm_attn_orkd,     orkd_attn_i8)      \
  X(attn_rr_i8,   ork_mm_attn_rr_orkd,  orkd_attn_rr_i8)   \
  X(layer_i8,     ork_i8_mm_layer,      orkd_layer_i8)     \
  /* --- resident KV -------------------------------------------------------------------------- */ \
  X(kv_append,    ork_kv_append,        orkd_kv_append)    \
  /* --- SDP activation ops (int8) ------------------------------------------------------------ */ \
  X(silu_i8,      ork_i8_npu_silu,      orkd_silu_i8)      \
  X(gelu_i8,      ork_i8_npu_gelu,      orkd_gelu_i8)      \
  X(ewmul_i8,     ork_i8_npu_ewmul,     orkd_ewmul_i8)     \
  X(add_i8,       ork_i8_npu_add,       orkd_add_i8)       \
  X(rsqrt_i8,     ork_i8_npu_rsqrt,     orkd_rsqrt_i8)     \
  X(exp_i8,       ork_i8_npu_exp,       orkd_exp_i8)       \
  /* --- SDP activation ops (int16) ----------------------------------------------------------- */ \
  X(silu_i16,     ork_i16_npu_silu,     orkd_silu_i16)     \
  X(gelu_i16,     ork_i16_npu_gelu,     orkd_gelu_i16)     \
  X(ewmul_i16,    ork_i16_npu_ewmul,    orkd_ewmul_i16)    \
  X(add_i16,      ork_i16_npu_add,      orkd_add_i16)      \
  X(rsqrt_i16,    ork_i16_npu_rsqrt,    orkd_rsqrt_i16)    \
  X(exp_i16,      ork_i16_npu_exp,      orkd_exp_i16)

/* Count of ops in the table (a coarse tripwire: if you add an orkd_* op but forget the lib twin AND this
 * table, ORK_OP_COUNT stays stale — the per-op existence checks are the real guarantee, this is a reminder). */
#define ORK_OP_COUNT_X(a,b,c) +1
#define ORK_OP_COUNT (0 ORK_OP_TABLE(ORK_OP_COUNT_X))

#endif /* ORK_OP_MANIFEST_H */
