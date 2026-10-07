/* ork_slice_f16.h — fp16 op shape-adapter: the SDK "chef", fp16 half.
 *
 * Sibling of ork_slice.h, NOT an extension of it, and deliberately so. The two precisions do not
 * share a caps struct because they do not share the rules: int8 slices K on a 512 multiple, fp16 on
 * 32; and int8's M ceiling is a constant per K (mg_max*64) while fp16's depends on N as well. One
 * struct serving both would have to encode a rule that is false for one of them — which is exactly
 * the class of mistake that produced the narrow-N wedge (r103), where an int8-derived element
 * budget was reused for fp16 at 2 B/elem.
 *
 * WHAT THIS IS FOR. ork_f16_mm_pack admits any K%32 / N%16 shape, but the hardware's correctness
 * envelope is narrower than that and depends on all three dims. A caller that wants to know BEFORE
 * building a model — which expert width is fast, which M will be tiled, which shape is refused —
 * had no way to ask. This answers it off-device: pure logic, no NPU, unit-testable on a workstation.
 *
 * THE M CEILING IS A CALLBACK, NOT A NUMBER. ork_slice.h takes `int mmax` and its own comment
 * concedes the problem ("mmax is K-dependent in reality ... 64 is the safe floor"), which pushes the
 * hard part back onto the caller and invites a stale constant. Here the ceiling is a function of the
 * ACTUAL (K-slice, N-tile) a program will see, so a plan cannot contain a tile the runtime would
 * have to clamp. Pass ork_f16_mm_mmax (ork_npu.h) as the callback for the real envelope — it works
 * on an ork_npu_init_offline context, so model design needs no board.
 *
 * WHAT A PLAN CANNOT PROMISE. M is a RUNTIME value — batch size, sequence length. A plan fixes the
 * K/N decomposition and the M STEP, but the caller still chooses how many rows to hand over. The
 * runtime therefore keeps enforcing the envelope regardless of any plan or pack signature; this
 * header makes the plan agree with the runtime, it does not replace it.
 *
 * TILE SEMANTICS (half-open ranges), same as ork_slice.h:
 *   - N-tiles partition [0,N) and M-tiles partition [0,M): independent output blocks.
 *   - K-slices partition [0,K): PARTIAL sums the caller accumulates. NOTE fp16 accumulates in fp32,
 *     so unlike int8 this is NOT bit-exact against the un-sliced op — it is usually MORE accurate
 *     (shorter accumulations condition better), but it is a different number. Do not assume equality.
 *
 * Header-only C11 (usable from C and C++), no deps.
 */
#ifndef ORK_SLICE_F16_H
#define ORK_SLICE_F16_H
#include <stddef.h>
#include "ork_slice.h"   /* ork_tile — the tile type is shared even though the caps are not */

/* Largest M (rows) one fp16 program may carry for this K-slice and N-tile. Must be the SAME rule
 * the runtime applies, or a plan will be silently re-tiled (or worse, accepted and hang). */
typedef int (*ork_f16_mmax_fn)(int Kslice, int Ntile, void *ud);

typedef struct ork_slice_f16_caps {
    int kmax;              /* max K per slice (the SoC fp16 K-slice, soc->ks: 2048 on RK3588) */
    int kmul;              /* K alignment: 32 for fp16 (NOT int8's 512) */
    int nmax;              /* max N per tile (Sn==1 ceiling: soc->nmax, 8192) */
    int nmul;              /* N alignment: 16 */
    ork_f16_mmax_fn mmax;  /* REQUIRED. NULL => every shape is refused, by design: there is no safe
                            * constant to default to, and inventing one is the original bug. */
    void *ud;              /* passed to mmax (e.g. the ork_npu context) */
} ork_slice_f16_caps;

/* RK3588 shape limits. mmax is left NULL on purpose — fill it with ork_f16_mm_mmax and the context.*/
static inline ork_slice_f16_caps ork_slice_f16_caps_rk3588(void) {
    ork_slice_f16_caps c;
    c.kmax = 2048; c.kmul = 32; c.nmax = 8192; c.nmul = 16; c.mmax = 0; c.ud = 0;
    return c;
}

/* RESOLVED CAPS IDENTITY — what a pack was planned against.
 *
 * A pack stamp records this, not merely "the SDK built me". The failure that actually bites is a
 * pack planned against DIFFERENT caps than the machine loading it (another SoC, another K-slice
 * size, a changed envelope rule) — a bare "SDK-built" flag is true in that case and tells you
 * nothing. Keep it FNV-1a and 32-bit: it goes in a blob header, and it only has to detect
 * difference, not resist forgery.
 *
 * `rule` identifies the mmax RULE in force, not a value — the values are per-shape. Bump it when the
 * envelope changes meaning (a new measurement, a new bank-split law), so old packs stop matching. */
#define ORK_F16_MMAX_RULE 2u   /* 1 = plain measured mcap; 2 = + the r103 narrow-N WEIGHT_BANK>=2 cap */

static inline unsigned ork_slice_f16_capid(const ork_slice_f16_caps *c, const char *soc_id) {
    unsigned h = 2166136261u;
    #define ORK__MIX(v) do { unsigned v_ = (unsigned)(v); \
        for (int b_ = 0; b_ < 4; b_++) { h ^= (v_ >> (b_*8)) & 0xffu; h *= 16777619u; } } while (0)
    ORK__MIX(c->kmax); ORK__MIX(c->kmul); ORK__MIX(c->nmax); ORK__MIX(c->nmul);
    ORK__MIX(ORK_F16_MMAX_RULE);
    #undef ORK__MIX
    if (soc_id) for (const char *p = soc_id; *p; p++) { h ^= (unsigned char)*p; h *= 16777619u; }
    return h ? h : 1u;   /* never 0: 0 means "unstamped" in a header */
}

/* Is this shape runnable at all? 0 = yes; -2 = misaligned / non-positive; -3 = no mmax callback;
 * -4 = the envelope refuses it (mmax came back < 1 for some tile). Cheap, no allocation. */
static inline int ork_slice_f16_check(int K, int N, const ork_slice_f16_caps *c) {
    if (!c || K <= 0 || N <= 0) return -2;
    if (K % c->kmul || N % c->nmul) return -2;
    if (!c->mmax) return -3;
    const int ks = ork__slice_step(c->kmax, c->kmul);
    const int ns = ork__slice_step(c->nmax, c->nmul);
    for (int k0 = 0; k0 < K; k0 += ks) { int kp = (K - k0 < ks) ? (K - k0) : ks;
        for (int n0 = 0; n0 < N; n0 += ns) { int nt = (N - n0 < ns) ? (N - n0) : ns;
            if (c->mmax(kp, nt, c->ud) < 1) return -4; } }
    return 0;
}

/* Decompose into tiles the fp16 run path can execute directly.
 *
 * Loop order is N -> K -> M, matching the runtime (npu.c run_loop: for ns, for ks, chunk, for m0).
 * It matters: the M step is recomputed per (K-slice, N-tile) because the ceiling depends on both,
 * so M-tiles are NOT uniform across the output — a planner that hoisted the M loop would produce a
 * decomposition the runtime does not actually use.
 *
 * Returns the tile count, -1 if it would exceed `max`, or the ork_slice_f16_check code on a bad
 * shape. */
static inline int ork_slice_f16_matmul(int K, int N, int M, const ork_slice_f16_caps *c,
                                       ork_tile *out, int max) {
    int rc = ork_slice_f16_check(K, N, c);
    if (rc) return rc;
    if (M <= 0 || !out || max <= 0) return -2;
    const int ks = ork__slice_step(c->kmax, c->kmul);
    const int ns = ork__slice_step(c->nmax, c->nmul);
    int n = 0;
    for (int n0 = 0; n0 < N; n0 += ns) { int n1 = n0 + ns < N ? n0 + ns : N;
        for (int k0 = 0; k0 < K; k0 += ks) { int k1 = k0 + ks < K ? k0 + ks : K;
            int ms = c->mmax(k1 - k0, n1 - n0, c->ud);
            if (ms < 1) return -4;
            for (int m0 = 0; m0 < M; m0 += ms) { int m1 = m0 + ms < M ? m0 + ms : M;
                if (n >= max) return -1;
                out[n].k0 = k0; out[n].k1 = k1; out[n].n0 = n0; out[n].n1 = n1;
                out[n].m0 = m0; out[n].m1 = m1;
                n++; } } }
    return n;
}

/* Program count without materializing the tiles — for sizing a buffer, or for costing a shape at
 * model-design time (each tile is one NPU program, and the per-program floor is what decides
 * whether a narrow shape is worth offloading at all). Same codes on error. */
static inline int ork_slice_f16_count(int K, int N, int M, const ork_slice_f16_caps *c) {
    int rc = ork_slice_f16_check(K, N, c);
    if (rc) return rc;
    if (M <= 0) return -2;
    const int ks = ork__slice_step(c->kmax, c->kmul);
    const int ns = ork__slice_step(c->nmax, c->nmul);
    long n = 0;
    for (int n0 = 0; n0 < N; n0 += ns) { int n1 = n0 + ns < N ? n0 + ns : N;
        for (int k0 = 0; k0 < K; k0 += ks) { int k1 = k0 + ks < K ? k0 + ks : K;
            int ms = c->mmax(k1 - k0, n1 - n0, c->ud);
            if (ms < 1) return -4;
            n += (M + ms - 1) / ms; } }
    return n > 0x7fffffff ? -1 : (int)n;
}

#endif /* ORK_SLICE_F16_H */
