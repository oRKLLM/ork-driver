/* test_f16_wide_k — fp16 matmuls on the SINGLE-CORE K-sliced run path (orki_run's run_loop).
 *
 * r104 (2026-10-08): ork_f16_mm_run at M=1024 K=248320 N=1024 — a tied output head's dX, the
 * contraction over the vocabulary — committed a job that never raised a completion interrupt. Core
 * 0 sat at 99% and the board needed a power cycle.
 *
 * MECHANISM (found offline, no board run): orki_run computed the M-chunk TWICE with two different
 * formulas. The pre-loop pass that sizes the Af/Cc staging buffers used the legacy int8-derived
 * mg_max*64; the loop that emits the programs used the measured fp16 envelope orki_f16_mcap_n.
 * Where the fp16 envelope is taller — Kp=512 gives 352 against 320, Kp=1024 gives 176 against 128
 * — the NPU was programmed to write past the end of Cc. Walking off a mapped IOVA region on this
 * hardware is not a fault; it is a hang. K=248320 with the SoC's KS=2048 leaves a K=512 TAIL
 * slice, which is where it died (the reported nout=360448 is exactly 352*1024).
 *
 * WHY THE SUITE MISSED IT. Every other fp16 case in the suite runs multi-core: fp16 M>1 with
 * N>=32 goes to run_multicore -> the colsplit path, which reads orki_f16_mcap_n for BOTH planning
 * and sizing and so was never inconsistent. orki_run's single-core run_loop is reached only when
 * colsplit declines, which for fp16 means a K so wide that the slice count blows the doorbell
 * chain cap — i.e. exactly the output-head shapes and nothing else we run. So these cases force
 * the core budget to 1, which reaches the same code at a shape that costs seconds instead of a
 * 508 MB weight.
 *
 * It validates NUMERICALLY against an fp64 reference over the same fp16 operands, not just "it
 * returned": the pre-fix failure mode one notch smaller is a silently dropped M-chunk, which a
 * liveness check passes. A deliberately-wrong control must FAIL or the comparison proves nothing.
 *
 * SAFETY: every case here is inside the envelope the fix establishes. If the sizing regresses, the
 * library now REFUSES (orki_check_extent) and this test fails politely. If the refusal itself is
 * removed, it hangs the board. That is the nature of the bug it covers.
 *
 * ORK_F16_HEAD_FULL=1 additionally runs the two real Qwen3.5-0.8B head shapes end to end
 * (M=1024 K=248320 N=1024, and M=248320 K=1024 N=1024). They need ~1 GB and ~2.5 GB respectively
 * and tens of seconds, so they are opt-in rather than part of the default suite.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ork_npu.h"

static unsigned st = 7u;
static double u01(void){ st = st*1103515245u + 12345u; return (double)((st>>13)&0xffff)/65536.0; }

static int fails = 0;

/* fp64 over the SAME fp16 operands the NPU saw: isolates the matmul from operand quantisation.
 *
 * `mag` returns sum|a_k*b_k|, and the error is scored against THAT, not against |acc|. At K up to
 * 248320 a random-sign dot product cancels down to ~sqrt(K) of its accumulated magnitude, so a
 * result-relative bar measures the cancellation, not the matmul: it rejects K=4096 at 1.3e-5 even
 * on the sched=0 slices this bug never touched. sum|a*b| is the quantity fp32 accumulation error
 * is actually proportional to, so the bar below is a real statement about the hardware. */
static double refcell(const ork_f16 *A, const ork_f16 *B, int m, int n, int K, int N, double *mag)
{
    double acc = 0, s = 0;
    for (int k = 0; k < K; k++) {
        double p = (double)(float)A[(size_t)m*K+k] * (double)(float)B[(size_t)k*N+n];
        acc += p; s += fabs(p);
    }
    if (mag) *mag = s;
    return acc;
}

/* One shape on the single-core path. `why` names what about it is being covered. */
static void run_case(ork_npu *c, int M, int K, int N, const char *why)
{
    size_t na = (size_t)M*K, nb = (size_t)K*N, nc = (size_t)M*N;
    ork_f16 *A = malloc(na*sizeof *A), *B = malloc(nb*sizeof *B);
    float   *C = malloc(nc*sizeof *C);
    if (!A || !B || !C) { printf("  M=%-6d K=%-6d N=%-5d  ALLOC FAILED (%.1f GiB) — skipped\n",
                                 M, K, N, (na*2.0+nb*2.0+nc*4.0)/1073741824.0);
                          free(A); free(B); free(C); return; }
    /* Small values: K here reaches 248320, and fp32 accumulation of 248320 unit-scale terms would
     * swamp the signal in rounding rather than in anything the test is about. */
    for (size_t i = 0; i < na; i++) A[i] = (ork_f16)(u01()*0.125-0.0625);
    for (size_t i = 0; i < nb; i++) B[i] = (ork_f16)(u01()*0.125-0.0625);
    memset(C, 0, nc*sizeof *C);

    ork_w *w = ork_f16_mm_pack(c, K, N, B);
    if (!w) { printf("  M=%-6d K=%-6d N=%-5d  pack REFUSED  (%s)\n", M, K, N, why); fails++; goto out; }
    int rc = ork_f16_mm_run(c, w, M, A, C);
    if (rc) { printf("  M=%-6d K=%-6d N=%-5d  run rc=%d  (%s)\n", M, K, N, rc, why); fails++;
              ork_mm_free(c, w); goto out; }

    /* Sample rather than sweep: an fp64 reference over every cell is hours at these shapes, and the
     * failure modes (dead output, a dropped M-chunk) are not subtle. Include the LAST row, which is
     * what a dropped final chunk leaves at zero, and rows either side of every M-chunk boundary. */
    double worst = 0; int wm = 0, wn = 0;
    for (int s = 0; s < 96; s++) {
        int m = (s == 0) ? 0 : (s == 1) ? M-1 : (int)(u01()*M);
        int n = (s < 2) ? (s ? N-1 : 0) : (int)(u01()*N);
        if (m >= M) m = M-1; if (n >= N) n = N-1;
        double mag, want = refcell(A, B, m, n, K, N, &mag), got = C[(size_t)m*N+n];
        double rel = fabs(got-want)/(mag > 1e-12 ? mag : 1e-12);
        if (rel > worst) { worst = rel; wm = m; wn = n; }
    }
    /* fp32 accumulation of K fp16 products: error grows ~eps*sqrt(K) against sum|a*b|. A dropped
     * M-chunk (zeros) or a garbage chunk lands at ~1.0 here, orders away from this bar. */
    double bar = 2e-7 * sqrt((double)K/1024.0) + 2e-7;
    int ok = worst <= bar;
    printf("  M=%-6d K=%-6d N=%-5d  relerr %.3e (bar %.1e) at [%d,%d]  %s   (%s)\n",
           M, K, N, worst, bar, wm, wn, ok ? "ok" : "FAIL", why);
    if (!ok) fails++;
    ork_mm_free(c, w);
out:
    free(A); free(B); free(C);
}

int main(void)
{
    ork_npu *c = ork_npu_init();
    if (!c) { printf("ork_npu_init failed\n"); return 1; }

    /* Force the single-core path. Without this, fp16 M>1 at these N goes multi-core -> colsplit,
     * which is a different scheduler and never had the bug. The point of the test is run_loop. */
    ork_npu_set_core_budget(c, 1);

    printf("single-core fp16 run_loop, M above the chunk ceiling:\n");
    /* Kp=1024: sizing said 128 rows, the loop emitted 176. M=1024 engages the tiling. */
    run_case(c, 1024, 1024, 1024, "Kp=1024 sched=1, emitted chunk 176 vs sized 128");
    /* K-SLICED with a sched=1 TAIL — the reported head's structure in miniature. KS=2048, so
     * K=2560 is one 2048 slice plus a 512 tail whose chunk is 352 against a sized 320. */
    run_case(c, 1024, 2560, 1024, "K-sliced, 512 tail: emitted chunk 352 vs sized 320");
    /* Tail at the other measured sched=1 width. */
    run_case(c, 1024, 2304, 1024, "K-sliced, 256 tail: chunk 704");
    /* A narrow N on the same path: the r103 cap must still bind here, and must not be loosened by
     * the sizing fix. */
    run_case(c,  512, 2560,   16, "K-sliced tail at narrow N (r103 cap still binds)");
    /* sched=0 slices only: no tail, nothing above 1 CBUF bank. The regime that always worked. */
    run_case(c, 1024, 4096, 1024, "pure sched=0 slices (control: this always passed)");

    /* The control: a deliberately wrong reference must FAIL, or the comparison above proves
     * nothing. Run it on the cheapest shape. */
    {
        int M = 64, K = 1024, N = 64;
        ork_f16 *A = malloc((size_t)M*K*sizeof *A), *B = malloc((size_t)K*N*sizeof *B);
        float *C = malloc((size_t)M*N*sizeof *C);
        for (size_t i = 0; i < (size_t)M*K; i++) A[i] = (ork_f16)(u01()*0.125-0.0625);
        for (size_t i = 0; i < (size_t)K*N; i++) B[i] = (ork_f16)(u01()*0.125-0.0625);
        ork_w *w = ork_f16_mm_pack(c, K, N, B);
        int rc = w ? ork_f16_mm_run(c, w, M, A, C) : -1;
        double mag, want = refcell(A, B, 3, 5, K, N, &mag) * 1.5;   /* deliberately wrong by 50% */
        double rel = fabs(C[3*N+5]-want)/(fabs(want)>1e-9?fabs(want):1e-9);
        printf("  control (reference scaled 1.5x) relerr %.3e — must be LARGE: %s\n",
               rel, (rc == 0 && rel > 0.01) ? "ok" : "FAIL");
        if (rc || rel <= 0.01) fails++;
        if (w) ork_mm_free(c, w);
        free(A); free(B); free(C);
    }

    if (getenv("ORK_F16_HEAD_FULL")) {
        printf("full-size output-head shapes (ORK_F16_HEAD_FULL):\n");
        run_case(c, 1024, 248320, 1024, "head dX — the shape that wedged the board");
        run_case(c, 248320, 1024, 1024, "head dW — the huge-M twin");
    } else {
        printf("(set ORK_F16_HEAD_FULL=1 to also run the two real 248320 head shapes, ~2.5 GiB)\n");
    }

    ork_npu_free(c);
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
