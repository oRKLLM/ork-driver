/* ork_ops.c — the SDK op-name registry: industry-standard name string <-> ork_op enum.
 *
 * Every supported (PROVEN) op is addressed by a unique enum value AND a canonical industry name string
 * (ONNX where a primitive exists, community LLM conventions otherwise). The single source is ORK_OP_LIST
 * below; the name table is generated from it, and a _Static_assert forces the list to enumerate EXACTLY
 * ORK_OP_NKIND ops. So adding an op to the `ork_op` enum without giving it a name here is a COMPILE ERROR
 * (this is the compile-time "unknown op" guard — no op may exist un-named / un-exported). Each op's working
 * probe is recorded in OPS_REGISTRY.md; check_registry.sh enforces every regcmd binds to one of these ops.
 */
#include "ork_npu.h"
#include <string.h>

/* (enum, "industry name") — the ONE list. Keep in enum order for readability; order is not load-bearing
 * (the table is enum-indexed). Naming (see the convention block in ork_npu.h): WEIGHTED ops (matmul family)
 * use WxAy quant notation (w8a8/w16a16/w4a4); weightless SDP/elementwise/norm ops keep a <op>_<dtype> suffix
 * (dtype in {i4,i8,i16,i32,f16}) since WxAy does not apply without a weight. */
#define ORK_OP_LIST(X) \
    X(ORK_OP_MM_I8,                    "matmul_w8a8")                \
    X(ORK_OP_MM_F16,                   "matmul_w16a16")              \
    X(ORK_OP_MM_I4,                    "matmul_w4a4")                \
    X(ORK_OP_SILU_F16,                 "silu_f16")                   \
    X(ORK_OP_EWMUL_F16,                "mul_f16")                    \
    X(ORK_OP_SILU_I8,                  "silu_i8")                    \
    X(ORK_OP_GELU_I8,                  "gelu_i8")                    \
    X(ORK_OP_EWMUL_I8,                 "mul_i8")                     \
    X(ORK_OP_ADD_I8,                   "add_i8")                     \
    X(ORK_OP_ADD_F16,                  "add_f16")                    \
    X(ORK_OP_SILU_I16,                 "silu_i16")                   \
    X(ORK_OP_MM_I4_GROUPED,            "matmul_w4a4_grouped")        \
    X(ORK_OP_GELU_I16,                 "gelu_i16")                   \
    X(ORK_OP_RSQRT_I8,                 "rsqrt_i8")                   \
    X(ORK_OP_EXP_I8,                   "exp_i8")                     \
    X(ORK_OP_EXP_I16,                  "exp_i16")                    \
    X(ORK_OP_MUL_I16,                  "mul_i16")                    \
    X(ORK_OP_ADD_I16,                  "add_i16")                    \
    X(ORK_OP_MUL_PERCHANNEL_I8,        "mul_perchannel_i8")          \
    X(ORK_OP_MUL_PERCHANNEL_F16,       "mul_perchannel_f16")         \
    X(ORK_OP_MUL_PERCHANNEL_I16,       "mul_perchannel_i16")         \
    X(ORK_OP_MATMUL_PERCHANNEL_F16,    "matmul_perchannel_w16a16")   \
    X(ORK_OP_REQUANTIZE_PERCHANNEL_I32,"requantize_perchannel_i32")  \
    X(ORK_OP_SOFTMAX_F16,              "softmax_f16")                \
    X(ORK_OP_REDUCEMAX_I8,             "reducemax_i8")               \
    X(ORK_OP_RESHAPE_F16,              "reshape_f16")                \
    X(ORK_OP_ROPE_NEOX_F16,            "rope_neox_f16")              \
    X(ORK_OP_MATMUL_SILU_I8,           "matmul_silu_w8a8")           \
    X(ORK_OP_MATMUL_REQUANT_I8,        "matmul_requant_w8a8")        \
    X(ORK_OP_MATMUL_SILU_I32,          "matmul_silu_w8a8_i32out")    \
    X(ORK_OP_RMSNORM_F16,              "rmsnorm_f16")                \
    X(ORK_OP_L2NORM_F16,               "l2norm_f16")

static const char *const ork_op_names[ORK_OP_NKIND] = {
#define X(e, n) [e] = n,
    ORK_OP_LIST(X)
#undef X
};

/* Compile-time completeness: the list MUST enumerate exactly ORK_OP_NKIND ops. If you add an ork_op enum
 * value (bumping ORK_OP_NKIND) without adding it to ORK_OP_LIST above, this fails to compile. */
enum { ORK_OP_LIST_COUNT = 0
#define X(e, n) + 1
    ORK_OP_LIST(X)
#undef X
};
_Static_assert(ORK_OP_LIST_COUNT == ORK_OP_NKIND,
    "ork_op enum and ORK_OP_LIST are out of sync: every op must have an industry name string here "
    "(a new ork_op with no name = unexported op, which is not allowed)");

const char *ork_op_name(ork_op k) {
    return ((int) k >= 0 && (int) k < ORK_OP_NKIND) ? ork_op_names[k] : (const char *) 0;
}

ork_op ork_op_from_name(const char *name) {
    if (name) for (int k = 0; k < ORK_OP_NKIND; k++)
        if (ork_op_names[k] && strcmp(ork_op_names[k], name) == 0) return (ork_op) k;
    return ORK_OP_NKIND;   /* unknown */
}

static const char *const ork_impl_mode_names[ORK_IMPL_NMODE] = {
    [ORK_IMPL_STANDALONE] = "standalone",
    [ORK_IMPL_HW_CHAINED] = "hw_chained",
    [ORK_IMPL_SW_CHAINED] = "sw_chained",
};
const char *ork_impl_mode_name(ork_impl_mode m) {
    return ((int) m >= 0 && (int) m < ORK_IMPL_NMODE) ? ork_impl_mode_names[m] : (const char *) 0;
}

/* ============================ deterministic op->op chaining table ============================
 * ork_chain_table[from][to] is a LOOKUP, not an algorithm — GENERATED from ORK_CHAIN_LIST (ork_npu.h), the
 * SAME single source as the compile-time ORK_CR__ constants, so the runtime table and the static asserts can
 * never drift. Unlisted pairs default to 0 = ORK_CHAIN_DISALLOW (correct-but-unchained; independent submits,
 * never wedges). Only listed pairs (validated by tools/mode_probe / OPS_REGISTRY) are HW/SW-chained. */
static const unsigned char ork_chain_table[ORK_OP_NKIND][ORK_OP_NKIND] = {
#define X(f, t, r) [f][t] = (unsigned char) (r),
    ORK_CHAIN_LIST(X)
#undef X
};
_Static_assert(sizeof(ork_chain_table) == (unsigned long) ORK_OP_NKIND * ORK_OP_NKIND,
    "ork_chain_table must span the full [ORK_OP_NKIND][ORK_OP_NKIND] permutation matrix");

ork_chain_rule ork_chain_lookup(ork_op from, ork_op to) {
    if ((int) from < 0 || (int) from >= ORK_OP_NKIND) return ORK_CHAIN_DISALLOW;
    if ((int) to   < 0 || (int) to   >= ORK_OP_NKIND) return ORK_CHAIN_DISALLOW;
    return (ork_chain_rule) ork_chain_table[from][to];
}

/* Compile-time validation of the SDK's fixed composites: their op sequences are structural (known at build
 * time), so every transition is asserted here. A composite whose sequence includes a DISALLOWed/unvalidated
 * step FAILS TO COMPILE — the enforcement lives at build time, not in a runtime failing check (orkd + SDK
 * ship together, so any chain the runtime could emit is drawn from these fixed patterns + matmul runs). */
ORK_ASSERT_CHAIN_STEP(ORK_OP_MM_I8,    ORK_OP_SILU_I8);              /* ffn_swiglu_i8 / ffn_gate_silu_i8: gate -> silu */
ORK_ASSERT_CHAIN_STEP(ORK_OP_SILU_I8,  ORK_OP_MM_I8);               /* ffn_swiglu_i8: silu -> up matmul */
ORK_ASSERT_CHAIN_STEP(ORK_OP_MM_I8,    ORK_OP_EWMUL_I8);            /* ffn_swiglu_i8: up -> GLU mul */
ORK_ASSERT_CHAIN_STEP(ORK_OP_EWMUL_I8, ORK_OP_MM_I8);               /* ffn_swiglu_i8: GLU mul -> down matmul */
ORK_ASSERT_CHAIN_STEP(ORK_OP_MM_I8,    ORK_OP_MM_I8);               /* chain_matmul_i8: adjacent matmul run */
ORK_ASSERT_CHAIN_STEP(ORK_OP_MM_F16,   ORK_OP_MUL_PERCHANNEL_F16);  /* chain_matmul_perchannel_f16: A.V -> scale */

/* ============================ composite (multi-op) name registry ============================ */
#define ORK_COMPOSITE_LIST(X) \
    X(ORK_COMPOSITE_CHAIN_MATMUL_W8A8,            "chain_matmul_w8a8")            \
    X(ORK_COMPOSITE_FFN_SWIGLU_W8A8,              "ffn_swiglu_w8a8")              \
    X(ORK_COMPOSITE_FFN_GATE_SILU_W8A8,           "ffn_gate_silu_w8a8")           \
    X(ORK_COMPOSITE_FFN_GATE_SDPSILU_W8A8,        "ffn_gate_sdpsilu_w8a8")        \
    X(ORK_COMPOSITE_CHAIN_MATMUL_PERCHANNEL_W16A16,"chain_matmul_perchannel_w16a16") \
    X(ORK_COMPOSITE_SEQ,                          "seq")                          \
    X(ORK_COMPOSITE_MATMUL_SILU_W16A16_I16SILU,   "matmul_silu_w16a16_i16silu")   \
    X(ORK_COMPOSITE_MATMUL_BATCHED_FUSED_W16A16,  "matmul_batched_fused_w16a16")  \
    X(ORK_COMPOSITE_GRAPH_REPLAY_F16,             "graph_replay_f16")

static const char *const ork_composite_names[ORK_COMPOSITE_NKIND] = {
#define X(e, n) [e] = n,
    ORK_COMPOSITE_LIST(X)
#undef X
};
enum { ORK_COMPOSITE_LIST_COUNT = 0
#define X(e, n) + 1
    ORK_COMPOSITE_LIST(X)
#undef X
};
_Static_assert(ORK_COMPOSITE_LIST_COUNT == ORK_COMPOSITE_NKIND,
    "ork_composite enum and ORK_COMPOSITE_LIST are out of sync: every composite must have a name string");

const char *ork_composite_name(ork_composite k) {
    return ((int) k >= 0 && (int) k < ORK_COMPOSITE_NKIND) ? ork_composite_names[k] : (const char *) 0;
}
ork_composite ork_composite_from_name(const char *name) {
    if (name) for (int k = 0; k < ORK_COMPOSITE_NKIND; k++)
        if (ork_composite_names[k] && strcmp(ork_composite_names[k], name) == 0) return (ork_composite) k;
    return ORK_COMPOSITE_NKIND;
}

/* ============================ regcmd -> op binding ============================
 * Which ork_op (and impl mode) each REGCMD_* byte template implements. An op may have several regcmd impls
 * across modes (e.g. mul_f16: standalone REGCMD_MUL_F16 vs chain-safe REGCMD_MUL_F16_CHAIN) — that IS the
 * (op x mode) -> regcmd model. check_registry.sh enforces every REGCMD_* base defined in npu.c appears here,
 * so no regcmd exists that isn't the implementation of an exported op (0 orphan regcmds / 0 unexported ops). */
#define ORK_REGCMD_BIND(X) \
    X("REGCMD_I8",             ORK_OP_MM_I8,        ORK_IMPL_HW_CHAINED) \
    X("REGCMD_I4",             ORK_OP_MM_I4,        ORK_IMPL_HW_CHAINED) \
    X("REGCMD_ADD",            ORK_OP_ADD_I8,       ORK_IMPL_SW_CHAINED) \
    X("REGCMD_ADD_F16",        ORK_OP_ADD_F16,      ORK_IMPL_SW_CHAINED) \
    X("REGCMD_ADD_I16",        ORK_OP_ADD_I16,      ORK_IMPL_SW_CHAINED) \
    X("REGCMD_EW_LANE",        ORK_OP_EWMUL_I8,     ORK_IMPL_HW_CHAINED) \
    X("REGCMD_EWMUL",          ORK_OP_EWMUL_I8,     ORK_IMPL_HW_CHAINED) \
    X("REGCMD_EWMUL_LIN",      ORK_OP_EWMUL_I8,     ORK_IMPL_HW_CHAINED) \
    X("REGCMD_MUL",            ORK_OP_EWMUL_I8,     ORK_IMPL_HW_CHAINED) \
    X("REGCMD_MUL_F16",        ORK_OP_EWMUL_F16,    ORK_IMPL_STANDALONE) \
    X("REGCMD_MUL_F16_CHAIN",  ORK_OP_EWMUL_F16,    ORK_IMPL_HW_CHAINED) \
    X("REGCMD_MUL_F16_NOTCH",  ORK_OP_EWMUL_F16,    ORK_IMPL_HW_CHAINED) \
    X("REGCMD_MUL_I16",        ORK_OP_MUL_I16,      ORK_IMPL_STANDALONE) \
    X("REGCMD_RESHAPE_F16",    ORK_OP_RESHAPE_F16,  ORK_IMPL_STANDALONE) \
    X("REGCMD_SILU_STD",       ORK_OP_SILU_I8,      ORK_IMPL_HW_CHAINED) \
    X("REGCMD_SILU_STD_F16",   ORK_OP_SILU_F16,     ORK_IMPL_HW_CHAINED) \
    X("REGCMD_SILU_STD_I16",   ORK_OP_SILU_I16,     ORK_IMPL_HW_CHAINED) \
    X("REGCMD_SILU_LUT",       ORK_OP_SILU_I8,      ORK_IMPL_STANDALONE) \
    X("REGCMD_SILU_F16_T2",    ORK_OP_SILU_F16,     ORK_IMPL_HW_CHAINED)

static const struct { const char *rc; ork_op op; ork_impl_mode mode; } ork_regcmd_binds[] = {
#define X(s, o, m) { s, o, m },
    ORK_REGCMD_BIND(X)
#undef X
};
ork_op ork_regcmd_op(const char *regcmd_name, ork_impl_mode *mode_out) {
    if (regcmd_name)
        for (unsigned i = 0; i < sizeof(ork_regcmd_binds) / sizeof(ork_regcmd_binds[0]); i++)
            if (strcmp(ork_regcmd_binds[i].rc, regcmd_name) == 0) {
                if (mode_out) *mode_out = ork_regcmd_binds[i].mode;
                return ork_regcmd_binds[i].op;
            }
    return ORK_OP_NKIND;
}
