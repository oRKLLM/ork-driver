/* percore_mode_probe — is the int4<->int8 precision-mode hazard PER-CORE or DEVICE-GLOBAL?
 *
 * The mode-transition layer models precision as ONE global scalar (ork_npu.last_dt) and answers every
 * int4<->int8 handoff with a device-wide RKNPU_ACT_RESET. That reset is MEASURED at 105 ms, and a 27B
 * mixed-tier pack pays 160 of them per forward = 16.8 s of a 161.6 s run (10.4%). Suppressing it on the
 * SAME core corrupts the output (PPL 10.44 -> 5.0e6), so on one core it is load-bearing.
 *
 * But RK3588 is THREE independent NVDLA-derived cores, each with its own CNA/DPU/PPU and CBUF, each
 * executing its own regcmd — and precision (0x4010 PROC_PRECISION) is programmed BY the regcmd. If the
 * hazard is per-core, then running int4 on one core and int8 on another needs NO reset at all, and
 * last_dt should be per-core rather than a scalar. That would make tier-mixing nearly free.
 *
 * Uses the CHAIN mechanisms deliberately. The plain run paths enter modes I8(1)/I4(2) and a first version
 * of this probe found phase B COHERENT there — i.e. they do not exercise the hazard at all, so the
 * question could not be asked. The 27B corruption came from I4_CHAIN(4) <-> I8_CHAIN(3), which is what
 * ork_i8_mm_run_chain / ork_i4_mm_run_chain drive.
 *
 * A=B=1 (both dtypes) => C[m][n] = sum_k 1*1 = K. Self-validating; exits nonzero if a phase that should
 * be coherent is not. Four phases, two of them controls:
 *
 *   A  same core, resets ON    -> MUST be ok      (baseline: the normal path works)
 *   B  same core, resets OFF   -> EXPECT bad      (positive control: the hazard is real and detectable)
 *   C  split cores, resets OFF -> THE QUESTION    (ok => per-core; bad => device-global)
 *   D  split cores, resets ON  -> MUST be ok      (sanity: splitting alone breaks nothing)
 *
 * If B comes back "ok" the probe is not detecting the hazard and C proves nothing — that is why B is here.
 * BOARD: sudo tools/util/npu_guard.sh -- ./percore_mode_probe [K N M ITERS]
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void ork_npu_set_test_core(int core);
extern void ork_npu_set_xspec_noreset(int on);
extern void ork_npu_set_xspec_noclear(int on);

/* one alternating pass: int8 CHAIN on core c8, int4 CHAIN on core c4, ITERS times.
 * Chains are the mechanism that ping-pongs I8_CHAIN(3) <-> I4_CHAIN(4) — the transition that costs
 * 105 ms and that corrupts output when suppressed on a single core. Returns #incoherent chain runs. */
enum { S = 3 };
static int alternate(ork_npu *c, const ork_mm_task_i8 *t8, const ork_mm_task_i4 *t4,
                     int N, int K, int N4, int K4, int c8, int c4, int iters)
{
    int bad = 0;
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < S; i++) memset(t8[i].C, 0, (size_t) t8[i].M * N * 4);
        ork_npu_set_test_core(c8);
        if (ork_i8_mm_run_chain(c, S, t8)) { fprintf(stderr, "  i8 chain rc!=0\n"); return -1; }
        for (int i = 0; i < S; i++) { int32_t *C = t8[i].C;
            for (size_t e = 0; e < (size_t) t8[i].M * N; e++) if (C[e] != K) { bad++; goto i4turn; } }
    i4turn:
        for (int i = 0; i < S; i++) memset(t4[i].C, 0, (size_t) t4[i].M * N4 * 4);
        ork_npu_set_test_core(c4);
        if (ork_i4_mm_run_chain(c, S, t4)) { fprintf(stderr, "  i4 chain rc!=0\n"); return -1; }
        for (int i = 0; i < S; i++) { int32_t *C = t4[i].C;
            for (size_t e = 0; e < (size_t) t4[i].M * N4; e++) if (C[e] != K4) { bad++; break; } }
    }
    return bad;
}

int main(int argc, char **argv)
{
    int K = argc > 1 ? atoi(argv[1]) : 512;
    int N = argc > 2 ? atoi(argv[2]) : 64;
    int M = argc > 3 ? atoi(argv[3]) : 8;
    int IT = argc > 4 ? atoi(argv[4]) : 8;

    ork_npu *c = ork_npu_init();
    if (!c) { fprintf(stderr, "ork_npu_init failed\n"); return 1; }
    int cores = ork_npu_cores(c);
    printf("percore_mode_probe: K=%d N=%d M=%d iters=%d cores=%d soc=%s\n", K, N, M, IT, cores, ork_npu_soc(c));
    if (cores < 2) { printf("  need >=2 cores — SKIP\n"); ork_npu_free(c); return 0; }

    int8_t *B = malloc((size_t) K * N); memset(B, 1, (size_t) K * N);
    /* DIFFERENT shapes per dtype on purpose: identical shapes leave the cached buffer sizes valid, so
     * skipping the size clear is harmless and the hazard cannot appear. At 27B the int4 and i4a8 weights
     * differ wildly (5120x5120 attn_output vs 17408x5120 ffn_down), which is the condition under test. */
    int K4 = K * 2, N4 = N * 2;
    int8_t *B4 = malloc((size_t) K4 * N4); memset(B4, 1, (size_t) K4 * N4);
    ork_w *w8 = ork_i8_mm_pack(c, K, N, B);
    ork_w *w4 = ork_i4_mm_pack(c, K4, N4, B4);
    if (!w8 || !w4) { printf("  pack failed (w8=%p w4=%p) — unsupported shape\n", (void*)w8, (void*)w4);
                      free(B); free(B4); ork_npu_free(c); return 1; }
    /* chain tasks: M=1 each (ork_i4_mm_run_chain requires M==1), own A/C per task */
    ork_mm_task_i8 t8[S]; ork_mm_task_i4 t4[S]; int8_t *tA[S], *tA4[S]; int32_t *tC8[S], *tC4[S];
    for (int i = 0; i < S; i++) {
        tA[i]  = malloc((size_t) K);  memset(tA[i], 1, (size_t) K);
        tA4[i] = malloc((size_t) K4); memset(tA4[i], 1, (size_t) K4);
        tC8[i] = malloc((size_t) N * 4); tC4[i] = malloc((size_t) N4 * 4);
        t8[i] = (ork_mm_task_i8){ w8, 1, tA[i], tC8[i], 0 };
        t4[i] = (ork_mm_task_i4){ w4, 1, tA4[i], tC4[i] };
    }
    (void) M;

    struct { const char *name; int c8, c4, noreset, noclear, expect_ok; } PH[] = {
        { "A same  core  reset ON  clear ON ", 0, 0, 0, 0,  1 },
        { "B same  core  reset OFF clear ON ", 0, 0, 1, 0,  0 },   /* is the RESET load-bearing? */
        { "C same  core  reset ON  clear OFF", 0, 0, 0, 1,  0 },   /* or the CLEARS? */
        { "D same  core  reset OFF clear OFF", 0, 0, 1, 1,  0 },   /* positive control: reproduce 27B */
        { "E split cores reset OFF clear OFF", 0, 1, 1, 1, -1 },   /* THE QUESTION */
        { "F split cores reset ON  clear ON ", 0, 1, 0, 0,  1 },   /* sanity */
    };
    int fail = 0, hazard_detected = 0, percore = -1;
    for (unsigned p = 0; p < sizeof PH / sizeof PH[0]; p++) {
        ork_npu_set_xspec_noreset(PH[p].noreset);
        ork_npu_set_xspec_noclear(PH[p].noclear);
        int bad = alternate(c, t8, t4, N, K, N4, K4, PH[p].c8, PH[p].c4, IT);
        const char *verdict = bad < 0 ? "RUN-ERROR" : bad ? "INCOHERENT" : "ok";
        printf("  %s  i8->core%d i4->core%d  %-10s (%d/%d bad)\n",
               PH[p].name, PH[p].c8, PH[p].c4, verdict, bad < 0 ? 0 : bad, IT * 2);
        if (PH[p].expect_ok == 1 && bad != 0) { printf("      ^ CONTROL FAILED — a path that must work does not\n"); fail = 1; }
        if (p == 3) hazard_detected = (bad > 0);   /* D = both suppressed, same core */
        if (p == 4) percore = (bad == 0);        /* E = both suppressed, split cores */
    }
    ork_npu_set_xspec_noreset(0); ork_npu_set_xspec_noclear(0);

    printf("\n  VERDICT: ");
    if (!hazard_detected)
        printf("INCONCLUSIVE — phase B stayed coherent, so this shape does not exercise the hazard.\n"
               "           Phase E proves nothing — do not read it as evidence either way.\n");
    else if (percore)
        printf("PER-CORE — the hazard vanishes when the two precisions use different cores.\n"
               "           last_dt can become per-core and the 105 ms reset dropped for split work.\n");
    else
        printf("DEVICE-GLOBAL — splitting cores does NOT avoid it; the reset is required.\n"
               "           Core partitioning cannot remove the transition cost; avoid tier mixing instead.\n");

    for (int i = 0; i < S; i++) { free(tA[i]); free(tA4[i]); free(tC8[i]); free(tC4[i]); }
    ork_w_free(w8); ork_w_free(w4); free(B); free(B4); ork_npu_free(c);
    return fail;
}
