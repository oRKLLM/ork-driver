/* test_f16_narrow_n — fp16 matmuls with a NARROW output width, at and past the M-chunk boundary.
 *
 * r103 (2026-10-06): ork_f16_mm_pack + ork_f16_mm_run at K=1024 N=16 was bit-exact at an M-chunk of
 * 128 and HUNG at 144 — job committed, no completion interrupt, core 0 pinned at 100%, the board
 * dead until a power cycle. N=32 hung the same way at 176. The pack check admits these shapes
 * (N%16==0), so the library accepted a shape its own run path could not execute.
 *
 * The suite had no narrow-N coverage at all: every fp16 case ran a wide N, where the M-chunk tops
 * out at the same cap but the weight streams and the starved WEIGHT_BANK is survivable. So this
 * test exists to make the regression impossible to reintroduce silently, and it deliberately runs
 * M ABOVE the single-chunk ceiling so the internal M-tiling actually engages — M=1024 at K=1024 is
 * the shape that hung in a real training step (the GDN in_proj_b forward).
 *
 * It validates numerically, not just "it returned": a guard that silently produced zeros would pass
 * a liveness check. Each case is compared against an fp64 reference over the SAME fp16 operands, so
 * the bar measures the matmul and not fp16 rounding, and a deliberately-wrong control must FAIL or
 * the comparison proves nothing.
 *
 * SAFETY: every case here is intended to be inside the guarded envelope. If the guard regresses,
 * this test does not fail politely — it hangs the board and costs a power cycle. That is the nature
 * of the bug it covers; it is still far better to find it here than in a training run.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ork_npu.h"

static unsigned st = 99u;
static double u01(void){ st = st*1103515245u + 12345u; return (double)((st>>13)&0xffff)/65536.0; }

static int fails = 0;

/* fp64 over the same fp16 operands the NPU saw: isolates the matmul from operand quantisation. */
static double refcell(const ork_f16 *A, const ork_f16 *B, int m, int n, int K, int N)
{
    double acc = 0;
    for (int k = 0; k < K; k++) acc += (double)(float)A[(size_t)m*K+k] * (double)(float)B[(size_t)k*N+n];
    return acc;
}

static void run_case(ork_npu *c, int M, int K, int N, const char *why)
{
    size_t na = (size_t)M*K, nb = (size_t)K*N, nc = (size_t)M*N;
    ork_f16 *A = malloc(na*sizeof *A), *B = malloc(nb*sizeof *B);
    float   *C = malloc(nc*sizeof *C);
    if (!A || !B || !C) { printf("  alloc failed\n"); fails++; return; }
    for (size_t i = 0; i < na; i++) A[i] = (ork_f16)(u01()*2.0-1.0);
    for (size_t i = 0; i < nb; i++) B[i] = (ork_f16)(u01()*2.0-1.0);
    memset(C, 0, nc*sizeof *C);

    ork_w *w = ork_f16_mm_pack(c, K, N, B);
    if (!w) { printf("  M=%-5d K=%-5d N=%-5d  pack REFUSED  (%s)\n", M, K, N, why); fails++; goto out; }
    int rc = ork_f16_mm_run(c, w, M, A, C);
    if (rc) { printf("  M=%-5d K=%-5d N=%-5d  run rc=%d  (%s)\n", M, K, N, rc, why); fails++;
              ork_mm_free(c, w); goto out; }

    /* sample the output surface rather than every cell: K=1024 x M=1024 in fp64 is slow and the
     * failure modes here (dead output, a dropped M-chunk) are not subtle. Include the LAST row,
     * which is the one a dropped final chunk would leave at zero. */
    double worst = 0; int zeros = 0;
    for (int s = 0; s < 24; s++) {
        int m = (s == 23) ? M-1 : (int)((long)s*M/24);
        int n = (int)((long)s*N/24) % N;
        double got = C[(size_t)m*N+n], want = refcell(A, B, m, n, K, N);
        double den = fabs(want) > 1e-3 ? fabs(want) : 1e-3;
        double rel = fabs(got-want)/den;
        if (rel > worst) worst = rel;
        if (got == 0.0) zeros++;
    }
    int bad = !(worst < 5e-3) || zeros > 2;
    printf("  M=%-5d K=%-5d N=%-5d  relerr %.2e  %s   (%s)\n",
           M, K, N, worst, bad ? "FAIL" : "ok", why);
    if (bad) fails++;
    ork_mm_free(c, w);
out:
    free(A); free(B); free(C);
}

int main(void)
{
    ork_npu *c = ork_npu_init();
    if (!c) { printf("ork_npu_init failed (no /dev/dri/cardN?)\n"); return 1; }

    printf("fp16 NARROW-N at and past the M-chunk boundary (r103 regression)\n");
    printf("  the shapes below hung the board before the N-aware M-cap; a hang here IS the failure\n\n");

    /* the exact reported boundary: 128 passed, 144 and 176 hung */
    run_case(c, 128,  1024, 16, "r103: largest chunk that always worked");
    run_case(c, 144,  1024, 16, "r103: first reported HANG");
    run_case(c, 176,  1024, 16, "r103: the mcap chunk, also hung");
    run_case(c, 176,  1024, 32, "r103: not N=16-specific");
    /* M well above one chunk, so the internal tiling runs many programs — the real training shape */
    run_case(c, 1024, 1024, 16, "r103: the GDN in_proj_b forward that started it");
    /* neighbours of the guard, to catch an off-by-one in the cap either way */
    run_case(c, 256,  1024, 48, "unmeasured width, inside the conservative guard");
    run_case(c, 352,  512,  16, "K=512 narrow N at its own mcap");
    run_case(c, 704,  256,  16, "K=256 narrow N at its own mcap");
    /* wide N must be untouched by the guard — this is the measured-good path */
    run_case(c, 176,  1024, 3584, "wide N: must keep the full envelope");

    /* positive control: a deliberately wrong reference must be rejected, or the bar above is empty */
    printf("\npositive control (must FAIL):\n");
    {
        int M=64,K=1024,N=16; size_t na=(size_t)M*K, nb=(size_t)K*N, nc=(size_t)M*N;
        ork_f16 *A=malloc(na*sizeof *A), *B=malloc(nb*sizeof *B); float *C=malloc(nc*sizeof *C);
        for (size_t i=0;i<na;i++) A[i]=(ork_f16)(u01()*2.0-1.0);
        for (size_t i=0;i<nb;i++) B[i]=(ork_f16)(u01()*2.0-1.0);
        ork_w *w=ork_f16_mm_pack(c,K,N,B);
        int ok = w && ork_f16_mm_run(c,w,M,A,C)==0;
        double worst=0;
        if (ok) for (int s=0;s<8;s++){ int m=(int)((long)s*M/8), n=(int)((long)s*N/8)%N;
            double want=refcell(A,B,m,n,K,N)*1.5;      /* WRONG on purpose */
            double den=fabs(want)>1e-3?fabs(want):1e-3;
            double rel=fabs(C[(size_t)m*N+n]-want)/den; if(rel>worst) worst=rel; }
        printf("  scaled reference  relerr %.2e  %s\n", worst, worst>5e-3?"FAIL (as required)":"PASSED — the bar is empty");
        if (!(worst > 5e-3)) fails++;
        if (w) ork_mm_free(c,w);
        free(A); free(B); free(C);
    }

    ork_npu_free(c);
    printf("\n%s\n", fails==0 ? "F16 NARROW-N OK" : "F16 NARROW-N FAILED");
    return fails != 0;
}
