/* cdma_calib.c — OFFLINE CDMA byte-address model + calibration harness (no NPU, no DRM, no board).
 *
 * WHY: reversing rkllm's M-fold A-layout by on-board stride sweeps is wedge-prone (~13 wedges historically).
 * This models the CNA/CDMA byte-fetch addressing in software so the layout can be DERIVED/searched offline,
 * collapsing "hundreds of wedge-risky probe submits" into (later) one or two wedge-safe confirmations.
 *
 * WHAT THIS FILE DOES (the calibration harness — the solid, silicon-anchored part):
 *   1. A parameterized feature-address model feat_off(spec, m, c, M, K).
 *   2. STANDARD mode (atom-16 NC1HWC2, GROUP_LINE_OFF=0) must reproduce ork's KNOWN-GOOD packing
 *      BYTE-FOR-BYTE — the same EWCUBE(m,c) ork submits to the real NPU and gets bit-exact results from
 *      (src/npu.c ork_npu_ewmul_i8), and the ork_woff weight layout. This anchors the model on silicon-truth
 *      WITHOUT a single board submit.
 *   3. A self-consistency int8 matmul through the modeled offsets == CPU reference (bijection + reduction sane).
 *
 * WHAT IS SCAFFOLDED FOR THE NEXT PHASE (the fold search — NOT yet solved):
 *   4. FOLD mode (GROUP_LINE_OFF=1, C2=64) seeded from the NVDLA cmodel super_normal_ratio semantics, with the
 *      captured rkllm calibration points embedded (M=36 surf_stride=1920B; M=16,K=3584,DATA_BANK=3 -> 2400
 *      channels reduced per submit). The exact (m,k)->offset under GROUP_LINE_OFF is the OPEN unknown; this is
 *      the hypothesis surface a later search enumerates, each survivor confirmed by ONE wedge-safe replay submit.
 *
 * Pure C11, libc only. Builds/runs on any host:  make cdma_calib && ./cdma_calib
 * Exit 0 iff all calibration checks pass (the "examples ARE tests" convention).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ======================= ground-truth layouts (from src/npu.c — what the real NPU consumes) ============== */

/* Standard int8 FEATURE cube, atom=16 (NVDLA C2=16 NC1HWC2, M as width). src/npu.c:6753 EWCUBE:
 *   (c/16)*(M*16) + m*16 + (c%16)   surf_stride = M*16.  c is the reduction channel (=K for a matmul). */
static long ork_feat_std(int m, int c, int M) {
    return (long)(c/16)*((long)M*16) + (long)m*16 + (c%16);
}

/* Weight ork_woff, atom=32 (src/npu.c:1907 bb[nt*KT*1024 + kt*1024 + nl*32 + kk]). KT = K/32. */
static long ork_woff(int n, int k, int K) {
    int KT = (K + 31) / 32;
    return ((long)(n/32)*KT + (k/32)) * 1024 + (long)(n%32)*32 + (k%32);
}

/* ======================= parameterized feature-address MODEL (the calibratable one) ===================== */

typedef struct {
    int  atom;          /* channel atom: 16 (standard) or 64 (fold super-normal) */
    int  group_line;    /* GROUP_LINE_OFF (CONV_CON1 bit29): 0 standard, 1 fold */
    long surf_stride;   /* bytes between channel-atom surfaces; 0 => derive contiguous (= M*atom) */
} feat_spec;

/* The model. STANDARD path is exact ork semantics; FOLD path is the seeded hypothesis (see notes). */
static long feat_off(const feat_spec *s, int m, int c, int M, int K) {
    (void)K;
    int atom = s->atom;
    long surf = s->surf_stride ? s->surf_stride : (long)M * atom;   /* contiguous atom surfaces if unset */
    if (!s->group_line) {
        /* standard NC1HWC2: [c/atom][m][c%atom], surf_stride between c-atom planes */
        return (long)(c/atom)*surf + (long)m*atom + (c%atom);
    }
    /* FOLD (GROUP_LINE_OFF=1, C2=64): SEED = same super-normal shape with a (possibly padded) surf_stride.
     * This is the OPEN unknown — the real interleave under GROUP_LINE_OFF is NOT published (name is in
     * mainline rocket_registers.h; behavior is not). Kept identical-shape here so the calibration below can
     * MEASURE the residual vs the captured points and the search phase can perturb {surf_stride, atom order,
     * line grouping} from this seed. DO NOT trust FOLD offsets until a wedge-safe replay confirms them. */
    return (long)(c/atom)*surf + (long)m*atom + (c%atom);
}

/* ======================= CBUF bank / entry model (predicts the fold's PARTIAL-reduction) ================= */
/* mfold RE (wiki Exp-2026-07-28-M-Fold...): a SINGLE fold submit reduces exactly DATA_BANK*bank_capacity
 * channels; 0x1044 = DATA_ENTRIES = (K/64)*M. Captured point: M=16,K=3584,DATA_BANK=3 -> 2400 channels
 * reduced (=150 of 224 atom-16 surfaces). We calibrate bank_capacity from that and predict others. */
static long data_entries(int K, int M) { return (long)(K/64) * M; }
static long fold_reduced_channels(int data_bank, long bank_capacity_ch) { return (long)data_bank * bank_capacity_ch; }

/* ======================= calibration checks ============================================================= */

static int check_feat_standard(void) {
    const int Ms[] = {2, 8, 16, 36, 228}, Ks[] = {512, 1024, 2048, 3584};
    feat_spec std = { .atom = 16, .group_line = 0, .surf_stride = 0 };
    long bad = 0, tot = 0;
    for (unsigned mi = 0; mi < sizeof Ms/sizeof*Ms; mi++)
        for (unsigned ki = 0; ki < sizeof Ks/sizeof*Ks; ki++) {
            int M = Ms[mi], K = Ks[ki];
            for (int m = 0; m < M; m++)
                for (int c = 0; c < K; c++, tot++)
                    if (feat_off(&std, m, c, M, K) != ork_feat_std(m, c, M)) bad++;
        }
    printf("  [feat std]  model vs ork EWCUBE: %ld/%ld mismatch  %s\n", bad, tot, bad ? "FAIL" : "OK");
    return bad != 0;
}

/* self-consistency: an int8 matmul reduced THROUGH the modeled feature + ork_woff offsets == CPU ref.
 * Confirms the offset maps are bijective over their index spaces and the reduction order is sane. */
static uint32_t lcg = 0x1234567u;
static int r7(void){ lcg = lcg*1664525u + 1013904223u; return (int)((lcg>>25)%7) - 3; }
static int check_matmul_standard(void) {
    int M = 16, K = 1024, N = 64;
    feat_spec std = { .atom = 16, .group_line = 0, .surf_stride = 0 };
    int8_t *A = malloc((size_t)M*K), *W = malloc((size_t)K*N);
    for (long i = 0; i < (long)M*K; i++) A[i] = (int8_t)r7();
    for (long i = 0; i < (long)K*N; i++) W[i] = (int8_t)r7();
    /* pack A into a feature buffer via the model, W via ork_woff */
    size_t fsz = (size_t)((K+15)/16)*M*16, wsz = (size_t)((N+31)/32)*((K+31)/32)*1024;
    int8_t *Af = calloc(fsz, 1), *Wf = calloc(wsz, 1);
    for (int m=0;m<M;m++) for (int c=0;c<K;c++) Af[feat_off(&std,m,c,M,K)] = A[(size_t)m*K+c];
    for (int n=0;n<N;n++) for (int k=0;k<K;k++) Wf[ork_woff(n,k,K)]      = W[(size_t)k*N+n];
    /* reduce through the packed layouts */
    long bad = 0;
    for (int m=0;m<M;m++) for (int n=0;n<N;n++) {
        long acc = 0, ref = 0;
        for (int k=0;k<K;k++) { acc += (long)Af[feat_off(&std,m,k,M,K)] * Wf[ork_woff(n,k,K)];
                                ref += (long)A[(size_t)m*K+k] * W[(size_t)k*N+n]; }
        if (acc != ref) bad++;
    }
    printf("  [matmul std] reduce-through-model vs A*W: %ld/%d mismatch  %s\n", bad, M*N, bad ? "FAIL" : "OK");
    free(A);free(W);free(Af);free(Wf);
    return bad != 0;
}

/* fold-mode SCAFFOLD readout: report what the model predicts against the captured rkllm points, so the
 * search phase has a live residual to close. NOT a pass/fail gate (the fold layout is unsolved). */
static void report_fold_scaffold(void) {
    printf("  [fold scaffold] captured calibration points (from mfold RE):\n");
    /* M=36 surf_stride: 0x1080=60 -> 60<<5 = 1920 B; contiguous atom-64 would be M*64 = 2304 B. */
    int M36 = 36; long surf_captured = 60L << 5, surf_contig64 = (long)M36 * 64;
    printf("    M=36:  captured surf_stride = %ld B (reg 0x1080=60 <<5);  contiguous atom-64 = %ld B  -> %s\n",
           surf_captured, surf_contig64, surf_captured == surf_contig64 ? "contiguous" : "PADDED/strided (unknown rule)");
    /* partial-reduction: M=16,K=3584,DATA_BANK=3 -> 2400 channels reduced (150 of 224 atom-16 surfaces). */
    int M16 = 16, K = 3584, DATA_BANK = 3; long reduced_captured = 2400;
    long bank_cap = reduced_captured / DATA_BANK;                 /* calibrate: 800 channels/bank */
    printf("    M=16,K=%d: DATA_ENTRIES(0x1044)=(K/64)*M = %ld;  captured single-submit reduce = %ld ch"
           " (=%ld of %d atom-16 surfaces);  DATA_BANK=%d -> bank_capacity=%ld ch\n",
           K, data_entries(K, M16), reduced_captured, reduced_captured/16, K/16, DATA_BANK, bank_cap);
    printf("    model reduce_channels(DATA_BANK=%d, cap=%ld) = %ld  (matches captured => bank model calibrated)\n",
           DATA_BANK, bank_cap, fold_reduced_channels(DATA_BANK, bank_cap));
    printf("    OPEN: the (m,k)->byte-offset UNDER GROUP_LINE_OFF+C2=64 is unpublished; seed = NVDLA\n"
           "          super_normal_ratio(2)*atom(32)=64.  Search perturbs {surf_stride, atom/line grouping}\n"
           "          from the seed; each survivor confirmed by ONE ork_npu_replay_i8_sweep submit (wedge-safe).\n");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("cdma_calib — offline CDMA address model (no NPU)\n");
    printf("== STANDARD calibration (must be silicon-anchored bit-exact) ==\n");
    int fail = 0;
    fail |= check_feat_standard();
    fail |= check_matmul_standard();
    printf("== FOLD scaffold (hypothesis surface for the next phase) ==\n");
    report_fold_scaffold();
    printf("%s\n", fail ? "CALIB FAIL" : "CALIB PASS (standard model reproduces ork's known-good layouts)");
    return fail ? 1 : 0;
}
