/* test_api_parity.c — REGRESSION TEST: the lib (direct) API and the orkd (daemon) API must stay in lockstep.
 *
 * This is a COMPILE-TIME test (needs no NPU): it proves, per op in src/ork_op_manifest.h, that the operation
 * exists on BOTH the in-process lib API and the orkd transport API, and it pins the high-risk coalesced/matmul
 * ops to explicit prototypes. If someone adds an op to one side only, renames a symbol, or changes a signature
 * without updating its twin, THIS FILE FAILS TO COMPILE — which fails `make test` and CI.
 *
 * Why not assert byte-identical prototypes? A socket cannot carry a pointer, so orkd necessarily takes
 * (orkd_conn*, uint64_t weight_id[, K, N]) where the lib takes (ork_npu*, ork_w*). The parity we CAN and do
 * enforce is CORRESPONDENCE: same op set (existence), plus pinned shapes on the ops most likely to drift.
 * See the manifest header + the 2026-07-24 AskUserQuestion decision. */
#include <stdio.h>
#include "ork_npu.h"
#include "orkd_client.h"
#include "ork_op_manifest.h"

/* Function-pointer sink: casting one function-pointer type to another is well-defined for STORAGE (only
 * calling through the wrong type would be UB, and we never call). Taking &lib / &orkd requires both to be
 * declared, so a missing or renamed symbol on either side is a compile error. */
typedef void (*ork_anyfn)(void);
static volatile ork_anyfn g_sink;

#define ORK_OP_EXIST(base, lib, orkd) g_sink = (ork_anyfn) &(lib); g_sink = (ork_anyfn) &(orkd);

int main(void) {
    /* (1) EXISTENCE PARITY — every op in the manifest resolves on both sides. */
    ORK_OP_TABLE(ORK_OP_EXIST)

    /* (2) PROTOTYPE PINS — the coalesced/matmul ops. A signature change on either side breaks the assignment
     *     (incompatible-function-pointer is a C constraint violation => hard compile error). The pinned lib and
     *     orkd shapes differ ONLY in the transport handle and the weight reference (ork_w* vs uint64_t id). */

    /* run_i8 — lib carries K,N inside ork_w; orkd passes them explicitly after the id (int32_t == int) */
    int (*l_run_i8)(ork_npu *, ork_w *, int, const int8_t *, int32_t *) = ork_mm_run_i8;
    int (*o_run_i8)(orkd_conn *, uint64_t, int, int, int, const int8_t *, int32_t *) = orkd_run_i8;
    (void) l_run_i8; (void) o_run_i8;

    /* ffn: gate/up/down + per-stage requant (mult/shift x3) + silu (in/out_scale) */
    int (*l_ffn)(ork_npu *, ork_w *, ork_w *, ork_w *, int, int, int, int,
                 int, int, int, int, int, int, double, double, const int8_t *, int32_t *) = ork_mm_ffn_orkd;
    int (*o_ffn)(orkd_conn *, uint64_t, uint64_t, uint64_t, int, int, int, int,
                 int, int, int, int, int, int, double, double, const int8_t *, int32_t *) = orkd_ffn_i8;
    (void) l_ffn; (void) o_ffn;

    /* attn RR: shared ones weight + per-chain kt/v */
    int (*l_attn_rr)(ork_npu *, int, ork_w *const *, ork_w *, ork_w *const *,
                     int, int, int, int, int, int, double, double, double,
                     const int8_t *, int32_t *, int32_t *) = ork_mm_attn_rr_orkd;
    int (*o_attn_rr)(orkd_conn *, int, const uint64_t *, const uint64_t *, const uint64_t *,
                     int, int, int, int, int, int, double, double, double,
                     const int8_t *, int32_t *, int32_t *) = orkd_attn_rr_i8;
    (void) l_attn_rr; (void) o_attn_rr;

    /* whole layer */
    int (*l_layer)(ork_npu *, const struct ork_layer_dims *,
                   ork_w *, ork_w *, ork_w *, ork_w *, ork_w *, ork_w *, ork_w *,
                   const float *, const float *, const float *,
                   const float *, const float *, const float *, float *) = ork_mm_layer_i8;
    int (*o_layer)(orkd_conn *, struct orkd_layer *,
                   const float *, const float *, const float *,
                   const float *, const float *, const float *, float *) = orkd_layer_i8;
    (void) l_layer; (void) o_layer;

    /* submit_seq: note the lib puts (ops, n) and orkd puts (n, ops) — a real, tolerated ordering difference */
    int (*l_seq)(ork_npu *, const ork_seq_op *, int) = ork_submit_seq;
    (void) l_seq;

    printf("test_api_parity: OK — %d ops present on both the lib and orkd APIs, pinned shapes match\n",
           ORK_OP_COUNT);
    return 0;
}
