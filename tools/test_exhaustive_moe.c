/* test_exhaustive_moe — static, gated "exhaustive MoE ring": chain every expert into ONE NPU
 * submission and gate the unselected ones with a per-token mask written into a DMA scratchpad,
 * so the regcmd packet stays immutable across decode steps (no per-token CPU recompile).
 *
 *   make test_exhaustive_moe && sudo ./test_exhaustive_moe [E] [K] [N] [iters]
 *   sudo ./test_exhaustive_moe            # 4 experts, K=N=2048, M=1 (decode)
 *
 * WHAT THIS IS (and the two places it diverges from a naive reading of the design):
 *
 *  1. Submission model. STALE: "the kernel rejects task_number>1" (batch_probe) is REFUTED -- vendor int4
 *     batches multi-task with task_number=rows, and our chain assemblers do it routinely. Original note: it
 *     times out). The *proven* monolithic-chain mechanism is task_number=1 + 0x0010 NEXT-pointer
 *     patching, which is exactly what ork_i8_mm_run_chain() already synthesizes: E expert matmuls
 *     strung into a single driver flight. So we chain, we do not "fuse N tasks".
 *
 *  2. The gate/mask node. An element-wise multiply node *on the NPU* (PPU) is not a validated
 *     primitive — Phase 1B (activations on the RK3588 PPU) replayed its init clean but wedged on
 *     the compute submit. So we realize the 0/1 gate through a proven path: each expert's input
 *     activation row lives in a single contiguous scratchpad (the "Mask Buffer"); to gate expert e
 *     OFF we zero its row, so its int8 matmul emits an all-zero output and collapses out of the
 *     accumulation. The regcmd chain never changes — only the scratchpad bytes do, per token. The
 *     final accumulation (sum over experts) is done on the CPU; at M=1 that is N int32 adds.
 *
 *     The scratchpad is HOST memory through the gather path, NOT an ork_dma_alloc DMA_BUF. The
 *     zero-copy DMA-A path is read raw and so requires A pre-tiled into the hardware layout; at M=1
 *     with natural-order activations it returns wrong results, and per-expert offset slices read
 *     past the buffer (SIGBUS). Zero-copy A was only ever validated for large-M prefill, where the
 *     library does the tiling. The gather path tiles A internally and is correct at M=1 — and the
 *     immutable-chain / per-token-data-mask question this harness asks does not depend on the
 *     buffer being a DMA_BUF.
 *
 * This measures the real question: does computing all E experts in ONE submission beat hitting
 * separate sequential ioctls? We report chain(1 trip) vs sequential-all(E trips, same work) and
 * vs sequential-selected(top-k trips, less work) — the genuine submit-amortization-vs-extra-FLOPs
 * trade. Random data: timing + the zero-mask-collapse correctness check, NOT a model-level check.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e6 + t.tv_nsec*1e-3; }
static unsigned int rs = 0x9e3779b9u;
static int rnd8(void) { rs = rs*1103515245u + 12345u; return (int)((rs >> 16) & 0xff) - 128; }

int main(int argc, char **argv) {
    int E     = argc > 1 ? atoi(argv[1]) : 4;
    int K     = argc > 2 ? atoi(argv[2]) : 2048;
    int N     = argc > 3 ? atoi(argv[3]) : 2048;
    int iters = argc > 4 ? atoi(argv[4]) : 200;
    const int M = 1;                          /* decode: one token */
    if (K % 32 || N % 16) { fprintf(stderr, "need K%%32==0, N%%16==0\n"); return 1; }
    if (E < 1 || E > 32)  { fprintf(stderr, "E out of range\n"); return 1; }

    ork_npu *c = ork_npu_init();
    if (!c) { fprintf(stderr, "ork_npu_init failed\n"); return 1; }

    /* Per-expert weights (immutable, packed once). Keep each expert's plaintext B so we can build a
     * CPU reference for the zero-mask-collapse check. */
    ork_w  **w  = calloc((size_t)E, sizeof *w);
    int8_t **Bx = calloc((size_t)E, sizeof *Bx);
    for (int e = 0; e < E; e++) {
        Bx[e] = malloc((size_t)K * N);
        for (size_t j = 0; j < (size_t)K * N; j++) Bx[e][j] = (int8_t)rnd8();
        w[e] = ork_i8_mm_pack(c, K, N, Bx[e]);
        if (!w[e]) { fprintf(stderr, "pack_i8 failed at expert %d\n", e); return 1; }
    }

    /* The scratchpad Mask Buffer: one contiguous host buffer holding all E expert input rows back
     * to back. Gating expert e OFF = zeroing its K-byte slot here (the only thing that mutates
     * between tokens). Host memory + gather path (correct at M=1); see header note on why not a
     * DMA_BUF. Outputs are a second region, one N-int32 slot per expert. */
    int8_t  *Abuf = malloc((size_t)E * K);
    int32_t *Cbuf = malloc((size_t)E * N * sizeof(int32_t));

    /* Real input row reused for every selected expert (so the CPU reference is shared). */
    int8_t *row = malloc((size_t)K);
    for (int k = 0; k < K; k++) row[k] = (int8_t)rnd8();

    /* Routing decision for this token: select even experts, gate odd ones to 0.0.
     * (E=4 -> experts 0,2 ON; 1,3 OFF, exactly the requested pattern.) */
    float *mask = malloc((size_t)E * sizeof(float));
    for (int e = 0; e < E; e++) mask[e] = (e % 2 == 0) ? 1.0f : 0.0f;
    int nsel = 0; for (int e = 0; e < E; e++) if (mask[e] != 0.0f) nsel++;

    /* Write the masked inputs into the scratchpad: selected -> real row, gated -> zeros. */
    for (int e = 0; e < E; e++) {
        if (mask[e] != 0.0f) memcpy(Abuf + (size_t)e * K, row, (size_t)K);
        else                 memset(Abuf + (size_t)e * K, 0,   (size_t)K);
    }

    ork_mm_task_i8 *tasks = calloc((size_t)E, sizeof *tasks);
    for (int e = 0; e < E; e++) {
        tasks[e].w = w[e]; tasks[e].M = M;
        tasks[e].A = Abuf + (size_t)e * K;
        tasks[e].C = Cbuf + (size_t)e * N;
    }

    printf("exhaustive MoE ring: E=%d K=%d N=%d M=%d  mask=[", E, K, N, M);
    for (int e = 0; e < E; e++) printf("%s%.0f", e ? "," : "", mask[e]);
    printf("]  (%d/%d experts selected)\n", nsel, E);

    /* Warm + one chained flight. */
    int rc = ork_i8_mm_run_chain(c, E, tasks);
    if (rc) { fprintf(stderr, "run_chain_i8 warmup failed rc=%d\n", rc); return 1; }

    /* --- Correctness: did the zero-masks collapse the gated experts without corrupting the rest? --- */
    int bad = 0;
    for (int e = 0; e < E && bad < 8; e++) {
        const int32_t *Ce = Cbuf + (size_t)e * N;
        if (mask[e] == 0.0f) {                       /* gated: every output must be exactly 0 */
            for (int n = 0; n < N; n++) if (Ce[n] != 0) {
                printf("  COLLAPSE FAIL expert %d col %d: got %d, expected 0\n", e, n, Ce[n]);
                if (++bad >= 8) break;
            }
        } else {                                     /* selected: must equal CPU int32 matmul */
            for (int n = 0; n < N && bad < 8; n++) {
                int32_t ref = 0; for (int k = 0; k < K; k++) ref += (int)row[k] * (int)Bx[e][(size_t)k*N+n];
                if (Ce[n] != ref) { printf("  SELECTED MISMATCH expert %d col %d: got %d exp %d\n", e, n, Ce[n], ref); bad++; }
            }
        }
    }
    /* Final accumulator: sum the ring's outputs; gated experts contribute 0 by construction. */
    {
        int32_t *acc = calloc((size_t)N, sizeof(int32_t));
        for (int e = 0; e < E; e++) for (int n = 0; n < N; n++) acc[n] += Cbuf[(size_t)e*N + n];
        int accbad = 0;
        for (int n = 0; n < N && accbad < 4; n++) {
            int32_t ref = 0;
            for (int e = 0; e < E; e++) if (mask[e] != 0.0f) ref += Cbuf[(size_t)e*N + n];
            if (acc[n] != ref) { printf("  ACCUM MISMATCH col %d: got %d exp %d\n", n, acc[n], ref); accbad++; bad++; }
        }
        free(acc);
    }
    printf("  correctness: %s (zero-mask collapse + selected matmul + accumulation)\n", bad ? "FAIL" : "OK");

    /* --- Timing: chained single-trip vs simulated multi-trip sequential dispatch --- */
    double t0 = now_us();
    for (int it = 0; it < iters; it++) { rc = ork_i8_mm_run_chain(c, E, tasks); if (rc) { fprintf(stderr, "chain rc=%d\n", rc); return 1; } }
    double us_chain = (now_us() - t0) / iters;

    t0 = now_us();
    for (int it = 0; it < iters; it++)
        for (int e = 0; e < E; e++) { rc = ork_i8_mm_run(c, w[e], M, Abuf + (size_t)e*K, Cbuf + (size_t)e*N); if (rc) { fprintf(stderr, "seq rc=%d\n", rc); return 1; } }
    double us_seq_all = (now_us() - t0) / iters;

    t0 = now_us();
    for (int it = 0; it < iters; it++)
        for (int e = 0; e < E; e++) if (mask[e] != 0.0f) { rc = ork_i8_mm_run(c, w[e], M, Abuf + (size_t)e*K, Cbuf + (size_t)e*N); if (rc) { fprintf(stderr, "seqsel rc=%d\n", rc); return 1; } }
    double us_seq_sel = (now_us() - t0) / iters;

    /* The architecture this actually points at: a DYNAMIC chain of just the selected experts in one
     * submit — submit amortization WITHOUT computing the masked ones. (The chain is rebuilt per
     * token from the routing decision; that synthesis cost is on the CPU, off the NPU's clock.) */
    ork_mm_task_i8 *tsel = calloc((size_t)nsel, sizeof *tsel);
    for (int e = 0, s = 0; e < E; e++) if (mask[e] != 0.0f) tsel[s++] = tasks[e];
    rc = ork_i8_mm_run_chain(c, nsel, tsel);  /* warm */
    t0 = now_us();
    for (int it = 0; it < iters; it++) { rc = ork_i8_mm_run_chain(c, nsel, tsel); if (rc) { fprintf(stderr, "chainsel rc=%d\n", rc); return 1; } }
    double us_chain_sel = (now_us() - t0) / iters;
    free(tsel);

    printf("\n  timing over %d warm iters (us/decode-step):\n", iters);
    printf("    exhaustive chain (1 submit,  %d experts) : %8.1f\n", E, us_chain);
    printf("    seq-all          (%d submits, %d experts) : %8.1f   (implied per-submit floor %.1f)\n", E, E, us_seq_all, us_seq_all / E);
    printf("    seq-select       (%d submits, %d experts) : %8.1f   (top-k, separate ioctls)\n", nsel, nsel, us_seq_sel);
    printf("    dyn-chain-select (1 submit,  %d experts) : %8.1f   (top-k, one chained submit) <- best\n", nsel, us_chain_sel);
    printf("\n  delta vs the exhaustive ring:\n");
    printf("    exhaustive vs seq-all          : %+.1f us  (%.2fx)   <- submit amortization, identical work\n", us_seq_all - us_chain, us_seq_all / us_chain);
    printf("    exhaustive vs seq-select       : %+.1f us  (%.2fx)   <- exhaustive one-trip vs top-k multi-trip\n", us_seq_sel - us_chain, us_seq_sel / us_chain);
    printf("    exhaustive vs dyn-chain-select : %+.1f us  (%.2fx)   <- the real alternative: chain only selected\n", us_chain_sel - us_chain, us_chain_sel / us_chain);

    free(row); free(mask); free(tasks); free(Abuf); free(Cbuf);
    for (int e = 0; e < E; e++) free(Bx[e]);
    free(Bx); free(w);
    ork_npu_free(c);          /* frees packed weights */
    return bad ? 2 : 0;
}
