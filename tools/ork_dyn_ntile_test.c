/* ork_dyn_ntile_test — P1b/G1: validate N-tiling (Sn>1) on the doorbell spine (ork_dyn_begin_mc).
 *
 * An op whose N exceeds nmax (8192 on RK3588) packs into Sn>1 N-slices; the doorbell now emits one
 * strided-output sub-program per slice (each writes a disjoint column range of C at row-stride N).
 * This asserts that path is bit-exact vs a CPU int32 reference, for M=1 and M>1, single- and multi-core,
 * across Sn=2 and Sn=3 — with COLUMN-VARYING weights so a column-offset / row-stride bug cannot hide
 * (all-ones weights would make every column identical and mask a striding defect).
 *
 * If ork_dyn_begin_mc cannot chain across N-slice weight buffers (kernel CDMA "cdma address wild",
 * errno 110), the doorbell misses -> ork_dyn_end returns <S-1 and the auto-dump fires: that is the
 * empirical signal (per the no-fallback plan) to pivot G1 to per-N-slice separate submits.
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int8_t bval(int k, int n) { return (int8_t)(((n % 7) - 3) + (k % 3)); }   /* varies by BOTH k and column n */
static int8_t aval(int m, int k) { return (int8_t)((k % 5) - 2 + (m % 3)); }

/* cmode: 0 = resident ork_dma_alloc C (non-cacheable, => DIRECT/zero-copy output);
 *        1 = plain malloc C (non-resident => NON-DIRECT: NPU writes cacheable mcc scratch, end() copies back).
 * The cmode isolates the cacheable-scratch coherency path from the direct path at any M/Sn. */
static int one_case_m(ork_npu *c, int K, int N, int M, int nc, int cmode) {
    int Sn = (N + 8191) / 8192;
    int8_t *B = malloc((size_t)K * N);
    for (int k = 0; k < K; k++) for (int n = 0; n < N; n++) B[(size_t)k * N + n] = bval(k, n);
    ork_w *w = ork_mm_pack_i8(c, K, N, B);
    if (!w) { printf("  [K=%d N=%d M=%d nc=%d] pack fail\n", K, N, M, nc); free(B); return 1; }

    int8_t *A = malloc((size_t)M * K);   /* HOST memory (doorbell stages A via memcpy) */
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) A[(size_t)m * K + k] = aval(m, k);

    int32_t *C = cmode ? (int32_t*)malloc((size_t)M * N * sizeof(int32_t))
                       : (int32_t*)ork_dma_alloc(c, (size_t)M * N * sizeof(int32_t));
    if (!C) { printf("  [K=%d N=%d M=%d nc=%d] C alloc fail\n", K, N, M, nc); free(A); free(B); return 1; }
    if (cmode) memset(C, 0, (size_t)M * N * sizeof(int32_t));

    const char *cm = cmode ? "scratch" : "direct ";
    ork_mm_task_i8 t = { .w = w, .M = M, .A = A, .C = C };
    ork_dyn_chain *h = ork_dyn_begin_mc(c, 1, &t, nc);
    if (!h) { printf("  [%s K=%d N=%d M=%d nc=%d Sn=%d] begin_mc=NULL (UNEXPECTED reject)\n", cm, K, N, M, nc, Sn); if (cmode) free(C); else ork_dma_free(c, C); free(A); free(B); return 1; }
    int done = ork_dyn_end(h);

    int bad = 0;
    (void)done;   /* ork_dyn_end returns the last completed entry index (S-1 for op-partition, nc-1 for a
                   * colsplit's internal tiles) — informational; correctness is the bit-exact scan below.
                   * A real doorbell miss leaves sentinel/zeros -> caught as a mismatch. */
    long mism = 0; int fm = -1, fn = -1; int32_t fg = 0, fe = 0;
    long sc_ok[8] = {0}, sc_zero[8] = {0}, sc_wrong[8] = {0};
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        int32_t acc = 0; for (int k = 0; k < K; k++) acc += (int32_t)aval(m, k) * (int32_t)bval(k, n);
        int32_t got = C[(size_t)m * N + n]; int sl = n / 8192; if (sl > 7) sl = 7;
        if (got == acc) sc_ok[sl]++;
        else { if (!mism) { fm = m; fn = n; fg = got; fe = acc; } mism++; if (got == 0) sc_zero[sl]++; else sc_wrong[sl]++; }
    }
    if (mism) { printf("  [%s K=%d N=%d M=%d nc=%d Sn=%d] MISMATCH at [%d,%d] got=%d exp=%d (total %ld)\n", cm, K, N, M, nc, Sn, fm, fn, fg, fe, mism);
        for (int sl = 0; sl < Sn; sl++) printf("      slice%d [%d,%d): ok=%ld zero=%ld wrong=%ld\n", sl, sl*8192, sl*8192+8192, sc_ok[sl], sc_zero[sl], sc_wrong[sl]);
        bad = 1; }
    if (!bad) printf("  [%s K=%d N=%d M=%d nc=%d Sn=%d] OK (bit-exact)\n", cm, K, N, M, nc, Sn);

    if (cmode) free(C); else ork_dma_free(c, C); free(A); free(B);
    return bad;
}
int main(void) {
    ork_npu *c = ork_npu_init();
    if (!c) { printf("init fail\n"); return 1; }
    printf("[ork_dyn_ntile_test] N-tiling on the doorbell — nmax=8192, column-varying weights\n");
    int fail = 0;
    /* M=1 wide-N: resident dma C (DIRECT zero-copy output) — bit-exact. */
    printf("-- M=1 wide-N (direct zero-copy output) --\n");
    fail += one_case_m(c, 512, 16384, 1, 1, 0);   /* nc=1 single-core direct (dma zero-copy OK) */
    fail += one_case_m(c, 512, 16384, 1, 3, 1);   /* nc=3 => colsplit multi-core: dma output is ZC-OUT-unsafe, use cacheable */
    fail += one_case_m(c, 512, 24576, 1, 1, 0);   /* nc=1 single-core direct */
    /* M>1: routed through scratch + copy-back/scatter to the caller's (cacheable) C — the supported output
     * convention. (Output zero-copy to a resident dma buffer at M>1 is the separate ZC-OUT opt-in, off by
     * default and coherency-unsafe; not exercised here.) */
    printf("-- M>1 single-slice + wide-N (scratch copy-back / scatter, cacheable output) --\n");
    fail += one_case_m(c, 512, 8192, 8, 1, 1);    /* M>1 Sn=1 scratch straight copy-back */
    fail += one_case_m(c, 512, 16384, 8, 1, 1);   /* M>1 Sn=2 scatter */
    fail += one_case_m(c, 512, 16384, 8, 3, 1);   /* M>1 Sn=2 scatter, multi-core requested */
    fail += one_case_m(c, 512, 24576, 8, 1, 1);   /* M>1 Sn=3 scatter */
    /* G2 K-split: wide-K (K>4096 => Sk K-slice partials + host accumulate). Sn==1. */
    printf("-- K-split (wide-K, Sn==1) — M=1 decode + M>1 prefill --\n");
    fail += one_case_m(c, 8192,  2048, 1,  1, 1);   /* Sk=8, M=1 decode */
    fail += one_case_m(c, 8192,  2048, 1,  1, 1);   /* Sk=8, M=1 (K-split accumulate to a cacheable C; dma-out is ZC-OUT) */
    fail += one_case_m(c, 18944, 3584, 1,  1, 1);   /* Sk~19, ffn_down-like decode */
    /* M>1 = A-gather + [M,N] partials + [M,N] accumulate. N kept modest on the M=64 cases so the O(M*N*K)
     * CPU reference stays fast (the 17MB-partial N=3584/M=64 shape is the same code, just a bigger bcreate). */
    fail += one_case_m(c, 8192,  2048, 8,  1, 1);   /* Sk=8,  M=8  prefill */
    fail += one_case_m(c, 8192,  512,  64, 1, 1);   /* Sk=8,  M=64 prefill */
    fail += one_case_m(c, 18944, 3584, 8,  1, 1);   /* Sk~19, M=8  ffn_down-like prefill */
    fail += one_case_m(c, 18944, 512,  64, 1, 1);   /* Sk~19, M=64 prefill */
    /* wide-M prefill (M>64): M-tiled into mtile_cap-row programs. K=4096 => cap=64 so M>64 multi-tiles. */
    printf("-- wide-M prefill (M>64, Sn==1, K<=4096) --\n");
    fail += one_case_m(c, 4096, 512,  128, 1, 1);   /* cap=64 => 2 M-tiles */
    fail += one_case_m(c, 2048, 1024, 256, 1, 1);   /* cap=128 => 2 M-tiles */
    fail += one_case_m(c, 4096, 1024, 256, 1, 1);   /* cap=64 => 4 M-tiles */
    /* P3: sub-nmax column-split across cores (M=1 int8, Sn==1, nc>1) — matches run_multicore's N-split. */
    printf("-- P3 colsplit: M=1 decode, N-columns split across 3 cores --\n");
    fail += one_case_m(c, 512,  2048, 1, 3, 1);   /* nc=3 colsplit (cacheable; multi-core dma is ZC-OUT-unsafe) */
    fail += one_case_m(c, 512,  2048, 1, 3, 1);   /* nc=3 colsplit, cacheable output */
    fail += one_case_m(c, 2048, 4096, 1, 3, 1);   /* bigger K/N */
    fail += one_case_m(c, 3072, 8192, 1, 3, 1);   /* N=nmax, Sn==1 */
    /* M>1 prefill colsplit: column-split across cores + M-tiling within each core + strided col copy-back. */
    printf("-- P3 colsplit: M>1 prefill, N-columns split across 3 cores --\n");
    fail += one_case_m(c, 2048, 4096, 8,   3, 1);   /* M=8  (cap=128 => 1 M-tile/core) */
    fail += one_case_m(c, 2048, 4096, 64,  3, 1);   /* M=64 (cap=128 => 1 M-tile/core) */
    fail += one_case_m(c, 4096, 2048, 128, 3, 1);   /* M=128 (cap=64 => 2 M-tiles/core within the colsplit) */
    /* P3: wide-N colsplit (Sn>1, M=1) — each core's column range spans N-slices (ffn_gate/up decode shape). */
    printf("-- P3 colsplit: wide-N (Sn>1) M=1 decode, columns span slices across cores --\n");
    fail += one_case_m(c, 2048, 16384, 1, 3, 1);   /* Sn=2, nc=3 (cores span the slice boundary) */
    fail += one_case_m(c, 3584, 18944, 1, 3, 1);   /* Sn=3, ffn_gate/up-like */
    fail += one_case_m(c, 2048, 24576, 1, 3, 1);   /* Sn=3 (cacheable; multi-core dma is ZC-OUT-unsafe) */
    /* P3: wide-K colsplit (K>4096, Sn==1, M=1) — per-core K-split accumulate over the column range (ffn_down decode). */
    printf("-- P3 colsplit: wide-K (K>4096) M=1 decode, per-core K-split accumulate --\n");
    fail += one_case_m(c, 8192,  2048, 1, 3, 1);   /* Sk=8,  columns split across 3 cores */
    fail += one_case_m(c, 18944, 3584, 1, 3, 1);   /* Sk~19, ffn_down-like */
    /* STAGE-3 RULE-OUT: the EXACT Qwen3-1.7B int8 matmul shapes (n_embd=2048, ff=6144, kv=1024), M=1 decode
     * + M=64 prefill, bit-exact vs CPU. If these all pass, the matmul CORE is correct for the model => the
     * bench garbage is upstream (backend act-quant/dequant), not stage 3. */
    printf("-- STAGE-3 rule-out: exact Qwen3-1.7B matmul shapes (M=1 decode + M=64 prefill) --\n");
    fail += one_case_m(c, 2048, 2048, 1,  3, 1);   /* attn q/o  decode */
    fail += one_case_m(c, 2048, 1024, 1,  3, 1);   /* attn k/v  decode (GQA) */
    fail += one_case_m(c, 2048, 6144, 1,  3, 1);   /* ffn gate/up decode */
    fail += one_case_m(c, 6144, 2048, 1,  3, 1);   /* ffn down  decode (K>4096 => per-core K-split) */
    fail += one_case_m(c, 2048, 2048, 64, 3, 1);   /* attn q/o  prefill */
    fail += one_case_m(c, 2048, 6144, 64, 3, 1);   /* ffn gate/up prefill */
    fail += one_case_m(c, 6144, 2048, 64, 3, 1);   /* ffn down  prefill */
    ork_npu_free(c);
    if (fail) { printf("ORK_DYN_NTILE_TEST: FAIL (%d cases)\n", fail); return 1; }
    printf("ORK_DYN_NTILE_TEST: PASS\n");
    return 0;
}
