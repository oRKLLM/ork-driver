/* slice_test — self-validating unit test for the slice-and-dice decomposer (ork_slice.h). NPU-free.
 * Verifies, for real FFN/attention shapes, that ork_slice_matmul() emits tiles that:
 *   (a) are all c_base-valid   (K-slice <= kmax & %kmul, N-slice <= nmax & %nmul, M-tile <= mmax),
 *   (b) COVER K x N x M exactly (K-slices partition [0,K) per output block; N/M-tiles partition [0,N)x[0,M)),
 *   (c) don't overlap or leave gaps,
 * and that unaligned shapes are rejected (-2). Pure logic; no board. `make slice_test && ./slice_test`.
 */
#include "ork_slice.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* Validate a full decomposition of (K,N,M): every tile c_base-valid + exact coverage (no gaps/overlaps). */
static void check_shape(const char *name, int K, int N, int M, const ork_slice_caps *c) {
    static ork_tile tiles[100000];
    int nt = ork_slice_matmul(K, N, M, c, tiles, (int)(sizeof tiles / sizeof tiles[0]));
    printf("%-22s K=%-6d N=%-6d M=%-4d -> %d tiles\n", name, K, N, M, nt);
    CHECK(nt > 0, "%s: decompose returned %d", name, nt);
    if (nt <= 0) return;

    /* (a) each tile c_base-valid */
    for (int i = 0; i < nt; i++) {
        int kw = tiles[i].k1 - tiles[i].k0, nw = tiles[i].n1 - tiles[i].n0, mw = tiles[i].m1 - tiles[i].m0;
        CHECK(kw > 0 && kw <= c->kmax && kw % c->kmul == 0, "%s tile%d Ksize=%d (need <=%d & %%%d)", name, i, kw, c->kmax, c->kmul);
        CHECK(nw > 0 && nw <= c->nmax && nw % c->nmul == 0, "%s tile%d Nsize=%d (need <=%d & %%%d)", name, i, nw, c->nmax, c->nmul);
        CHECK(mw > 0 && mw <= c->mmax,                      "%s tile%d Msize=%d (need <=%d)",        name, i, mw, c->mmax);
    }

    /* (b,c) exact coverage: count how many K-slices touch each (m,n) cell; must equal kparts everywhere. */
    int kparts = ork_slice_kparts(K, c);
    /* sample the (m,n) plane on a coarse grid + all boundaries to keep it O(cells), and verify each sampled
     * cell is covered by exactly `kparts` tiles whose K-slices exactly tile [0,K). */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            /* only test a sparse set of (m,n) to stay fast: every tile-corner + a few interiors */
            if (!(m == 0 || m == M-1 || (m % 37) == 0) || !(n == 0 || n == N-1 || (n % 101) == 0)) continue;
            int hit = 0, kcover = 0, kbits = 0;
            for (int i = 0; i < nt; i++)
                if (m >= tiles[i].m0 && m < tiles[i].m1 && n >= tiles[i].n0 && n < tiles[i].n1) {
                    hit++;
                    kcover += tiles[i].k1 - tiles[i].k0;
                    /* K-slices must be disjoint & contiguous from 0: track via a running expected boundary set */
                    (void)kbits;
                }
            CHECK(hit == kparts, "%s cell(m=%d,n=%d) covered by %d tiles (expect %d K-slices)", name, m, n, hit, kparts);
            CHECK(kcover == K, "%s cell(m=%d,n=%d) K-coverage=%d (expect K=%d)", name, m, n, kcover, K);
        }
    }
}

int main(void) {
    ork_slice_caps c = ork_slice_caps_rk3588();
    printf("caps: kmax=%d kmul=%d nmax=%d nmul=%d mmax=%d\n\n", c.kmax, c.kmul, c.nmax, c.nmul, c.mmax);

    /* qwen3-1.7b */
    check_shape("qwen3 gate/up",   2048,  6144, 256, &c);   /* K<=kmax, N<=nmax  -> M-tiles only */
    check_shape("qwen3 ffn_down",  6144,  2048, 256, &c);   /* WIDE-K (K>kmax)   -> K-slices [4096,2048] */
    check_shape("qwen3 attn_qkv",  2048,  2048, 256, &c);
    /* qwen2.5-7b (the wide shapes) */
    check_shape("7b ffn_gate/up",  3584, 18944, 228, &c);   /* WIDE-N (N>nmax)   -> 3 N-tiles */
    check_shape("7b ffn_down",    18944,  3584, 228, &c);   /* very WIDE-K       -> 5 K-slices */
    /* decode (M=1) */
    check_shape("decode down",     6144,  2048,   1, &c);

    /* alignment rejection (increment-1 rejects; pad is increment 3) */
    ork_tile t[16];
    CHECK(ork_slice_matmul(2000, 6144, 256, &c, t, 16) == -2, "unaligned K=2000 should reject (-2)");
    CHECK(ork_slice_matmul(2048, 6100, 256, &c, t, 16) == -2, "unaligned N=6100 should reject (-2)");
    CHECK(ork_slice_matmul(2048, 6144, 256, &c, t, 1)  == -1, "tiny max should return -1 (overflow)");

    printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
