/* ork_slice.h — op shape-adapter ("slice-and-dice"), the SDK "chef".
 *
 * Decomposes an int8 matmul C[M,N] = A[M,K] * B[K,N] into a set of "c_base" tiles that the doorbell
 * (ork_dyn) can run directly — Sn==1 (N-slice <= nmax), K-slice <= kmax (& %kmul), M-tile <= mmax. This is
 * what lets EVERY op ride the single doorbell submission path (no blocking fallback, no submit race, no
 * unverified wide-K/wide-N colsplit) — see SLICE_AND_DICE_PLAN.md.
 *
 * Semantics of the tile set (half-open ranges):
 *   - N-tiles partition [0,N) and M-tiles partition [0,M): independent output blocks (scatter).
 *   - K-slices partition [0,K): PARTIAL sums — the caller runs each and ACCUMULATES int32 over k
 *     (int8*int8->int32 accumulation is exact & associative, so K-slicing is bit-exact vs the un-sliced op).
 * So for output block (m,n), C[m,n] = sum over the K-slices of matmul(A[m-tile, k-slice], B[k-slice, n-tile]).
 *
 * Header-only C11 (usable from C and C++), no deps. Pure logic — no NPU, unit-testable off-device.
 */
#ifndef ORK_SLICE_H
#define ORK_SLICE_H
#include <stddef.h>

typedef struct { int k0, k1, n0, n1, m0, m1; } ork_tile;   /* half-open [.0,.1) */

typedef struct {
    int kmax;   /* max K per tile (c_base: <= 4096 on RK3588) */
    int kmul;   /* K alignment (512): every K-slice size % kmul == 0 (requires K % kmul == 0) */
    int nmax;   /* max N per tile (Sn==1 ceiling: 8192) */
    int nmul;   /* N alignment (16): every N-slice size % nmul == 0 (requires N % nmul == 0) */
    int mmax;   /* max M per tile (weight-DMA-amortization cap, ~ mg_max*64; caller sets per SoC) */
} ork_slice_caps;

/* RK3588 defaults. mmax is K-dependent in reality (mg_max*64: 128@K2048 .. 64@K4096); pass a value valid for
 * the largest K-slice you'll run (<= kmax). 64 is the safe floor for K up to 4096. */
static inline ork_slice_caps ork_slice_caps_rk3588(void) {
    ork_slice_caps c; c.kmax = 4096; c.kmul = 512; c.nmax = 8192; c.nmul = 16; c.mmax = 64; return c;
}

/* Largest multiple of `mul` that is <= cap (the aligned tile step). */
static inline int ork__slice_step(int cap, int mul) { int s = (cap / mul) * mul; return s < mul ? mul : s; }

/* Decompose into c_base tiles. Writes up to `max` tiles to `out`; returns the tile count, or -1 if it would
 * exceed `max`, or -2 on a bad/ unaligned shape (K%kmul, N%nmul, or non-positive dims) — the pad case
 * (increment 3) is NOT handled here yet, it's flagged so the caller doesn't silently miscompute. */
static inline int ork_slice_matmul(int K, int N, int M, const ork_slice_caps *c, ork_tile *out, int max) {
    if (K <= 0 || N <= 0 || M <= 0 || !c) return -2;
    if (K % c->kmul || N % c->nmul) return -2;              /* alignment: increment-1 rejects; pad is increment 3 */
    const int ks = ork__slice_step(c->kmax, c->kmul);       /* K step (aligned) */
    const int ns = ork__slice_step(c->nmax, c->nmul);       /* N step (aligned) */
    const int ms = c->mmax;                                 /* M step */
    int n = 0;
    for (int m0 = 0; m0 < M; m0 += ms) { int m1 = m0 + ms < M ? m0 + ms : M;
        for (int n0 = 0; n0 < N; n0 += ns) { int n1 = n0 + ns < N ? n0 + ns : N;
            for (int k0 = 0; k0 < K; k0 += ks) { int k1 = k0 + ks < K ? k0 + ks : K;
                if (n >= max) return -1;
                out[n].k0 = k0; out[n].k1 = k1; out[n].n0 = n0; out[n].n1 = n1; out[n].m0 = m0; out[n].m1 = m1;
                n++; } } }
    return n;
}

/* Number of K-slices per output block (how many partials accumulate into one C[m,n]). */
static inline int ork_slice_kparts(int K, const ork_slice_caps *c) {
    int ks = ork__slice_step(c->kmax, c->kmul); return (K + ks - 1) / ks;
}
#endif /* ORK_SLICE_H */
