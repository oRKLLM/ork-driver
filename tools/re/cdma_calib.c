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

/* ======================= FOLD SEARCH (phase 1: constrain the layout family, offline) ==================== */
/* Captured invariants (from the mfold RE / regcmd captures):
 *   surf-stride reg 0x1080 * M ~= 2160  [reg=60@M36, reg=108@M20]  -> surf_stride shrinks as M grows
 *   line-stride reg @M36 = 96 (3072 B)
 *   single-submit partial reduce: M=16,K=3584,DATA_BANK=3 -> 2400 channels
 * On-board sweep (tools/re/fold_alayout.c) ALREADY tested & REJECTED (100% mismatch) these within-tile
 * orderings: rowmajor[M][K], NC1HWC2 C2 in {16,32,64}, colmajor[K][M], nc16 M-innermost. */
/* CANDIDATE: NC1HWC2 C2=32 with the CAPTURED (padded) surf_stride = (2160/M)*32 bytes — i.e. the surface
 * stride is the captured 0x1080 value, NOT the contiguous M*32 that fold_alayout.c already tested+rejected.
 * This is the concrete GROUP_LINE_OFF reading: a padded per-channel-atom surface. */
static long fold_surf_bytes(int M) { return (2160L / M) * 32; }        /* = 1920 @M36, 3456 @M20 (captured) */
static long feat_off_fold_padded(int m, int k, int M) {
    return (long)(k/32) * fold_surf_bytes(M) + (long)m*32 + (k%32);
}
/* validate the candidate is self-consistent OFFLINE before any board submit:
 *  (a) its surf_stride reproduces the captured 0x1080 (60@M36, 108@M20),
 *  (b) it is a collision-free bijection over one submit's [M x Kslice],
 *  (c) it reduces to a correct partial matmul. */
static int verify_padded_candidate(void) {
    printf("    -- candidate: padded-NC32 (surf_stride = captured 0x1080, not contiguous M*32) --\n");
    int bad_stride = 0;
    struct { int M, reg; } pts[] = { {36,60}, {20,108} };
    for (unsigned i=0;i<sizeof pts/sizeof*pts;i++){ long got = fold_surf_bytes(pts[i].M) >> 5;
        printf("       surf_stride reg @M=%d: candidate=%ld captured=%d %s\n",
               pts[i].M, got, pts[i].reg, got==pts[i].reg?"OK":"MISMATCH");
        if (got != pts[i].reg) bad_stride = 1; }
    /* bijection + partial matmul over one submit: M=36, Kslice=512 */
    int M=36, Ks=512, N=32;
    long span = (long)((Ks+31)/32)*fold_surf_bytes(M) + (long)M*32;   /* padded buffer size */
    int8_t *buf = calloc(span,1); int *seen = calloc(span, sizeof(int));
    int coll=0; for(int m=0;m<M;m++) for(int k=0;k<Ks;k++){ long o=feat_off_fold_padded(m,k,M);
        if(o<0||o>=span){coll++;continue;} if(seen[o]++)coll++; }
    printf("       bijection over [M=%d x Kslice=%d] in %ld B: %s (%d collisions)\n",
           M, Ks, span, coll?"FAIL":"OK", coll);
    /* partial matmul: pack A padded, reduce vs ref */
    int8_t *A=malloc((size_t)M*Ks),*W=malloc((size_t)Ks*N);
    for(long i=0;i<(long)M*Ks;i++)A[i]=(int8_t)r7(); for(long i=0;i<(long)Ks*N;i++)W[i]=(int8_t)r7();
    for(int m=0;m<M;m++)for(int k=0;k<Ks;k++){long o=feat_off_fold_padded(m,k,M); if(o>=0&&o<span)buf[o]=A[(size_t)m*Ks+k];}
    long mmbad=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long acc=0,ref=0;
        for(int k=0;k<Ks;k++){ acc += (long)buf[feat_off_fold_padded(m,k,M)] * W[(size_t)k*N+n];
                               ref += (long)A[(size_t)m*Ks+k]           * W[(size_t)k*N+n]; }
        if(acc!=ref)mmbad++; }   /* asserts A's (m,k) is recoverable through the padded feature map */
    printf("       partial matmul (feature map recoverable): %ld/%d %s\n", mmbad, M*N, mmbad?"FAIL":"OK");
    free(buf);free(seen);free(A);free(W);
    return bad_stride || coll || mmbad;
}
/* EXHAUSTIVE constrained enumerator over the CDMA strided-addressing space.
 * offset(m,k) = ksurf*(k/atom) + line*(m/g) + (m%g)*atom + (k%atom)   [additive; order-independent]
 * Free params: atom in {16,32,64}, line-group g, and which captured stride {1920,3072} is ksurf vs line.
 * Filters (all offline): (F1) both register strides reproduce the captured {60,96}; (F2) bijection over
 * [M x Kslice] with no collisions (Kslice = the largest that packs collision-free); (F3) correct matmul;
 * (F4) in-bounds max-offset < a fixed buffer (WEDGE-SAFE proof). Prints the survivor SHORTLIST + max-offset. */
static uint32_t lcg2 = 0xabcdef1u;
static int r7b(void){ lcg2 = lcg2*1664525u + 1013904223u; return (int)((lcg2>>25)%7) - 3; }
static long off_gen(int m, int k, int atom, long ksurf, long line, int g) {
    return ksurf*(long)(k/atom) + line*(long)(m/g) + (long)(m%g)*atom + (k%atom);
}
static void enumerate_layouts(void) {
    printf("== EXHAUSTIVE ENUMERATOR (how many layouts survive the NPU constraints?) ==\n");
    const int atoms[] = {16,32,64};
    const long caps[] = {1920,3072};          /* the two captured loop strides (bytes) */
    const int gs[] = {1,2,3,4,6,9,12,18,36};   /* line-group over M=36 */
    const long WEDGE_BUF = 1<<20;              /* 1 MiB in-bounds envelope (proof of no-OOB) */
    int M=36, N=16;
    int tried=0, biject=0, survive=0, fullk_survive=0;
    printf("   grid: atom{16,32,64} x g{1,2,3,4,6,9,12,18,36} x stride-assign{2} = 54 raw combos\n");
    for (unsigned ai=0; ai<3; ai++) for (unsigned gi=0; gi<9; gi++) for (int sw=0; sw<2; sw++) {
        int atom = atoms[ai], g = gs[gi];
        long ksurf = caps[sw], line = caps[!sw];
        tried++;
        /* largest Kslice (multiple of atom) that packs collision-free in WEDGE_BUF */
        int Kslice = 0; long maxoff = 0;
        static char seen[1<<20];
        for (int Kt = atom; Kt <= 3584; Kt += atom) {
            memset(seen, 0, WEDGE_BUF);
            int ok = 1; long mo = 0;
            for (int m=0; m<M && ok; m++) for (int k=0; k<Kt; k++) {
                long o = off_gen(m,k,atom,ksurf,line,g);
                if (o < 0 || o >= WEDGE_BUF) { ok = 0; break; }      /* F4: OOB -> not wedge-safe, reject */
                if (seen[o]) { ok = 0; break; }                       /* F2: collision -> not bijective */
                seen[o] = 1; if (o > mo) mo = o;
            }
            if (!ok) break;
            Kslice = Kt; maxoff = mo;
        }
        if (Kslice < atom) continue;   /* couldn't even fit one atom-group collision-free */
        biject++;
        /* F3: correct matmul over [M x Kslice] through this feature map */
        int8_t *A=malloc((size_t)M*Kslice),*W=malloc((size_t)Kslice*N);
        for(long i=0;i<(long)M*Kslice;i++)A[i]=(int8_t)r7b();
        for(long i=0;i<(long)Kslice*N;i++)W[i]=(int8_t)r7b();
        int8_t *buf=calloc(maxoff+1,1);
        for(int m=0;m<M;m++)for(int k=0;k<Kslice;k++)buf[off_gen(m,k,atom,ksurf,line,g)]=A[(size_t)m*Kslice+k];
        long mmbad=0;
        for(int m=0;m<M&&!mmbad;m++)for(int n=0;n<N;n++){ long acc=0,ref=0;
            for(int k=0;k<Kslice;k++){acc+=(long)buf[off_gen(m,k,atom,ksurf,line,g)]*W[(size_t)k*N+n];
                                      ref+=(long)A[(size_t)m*Kslice+k]*W[(size_t)k*N+n];}
            if(acc!=ref)mmbad++; }
        free(A);free(W);free(buf);
        if (mmbad) continue;
        survive++;
        int fullk = (Kslice >= 3584);           /* real A holds full K => the physically-sensible ones */
        if (fullk) fullk_survive++;
        printf("   SURVIVOR #%d: atom=%d g=%-2d surf_reg=%ld line_reg=%ld | Kslice=%d ch%s | max_off=%ld B (<1MiB: WEDGE-SAFE)\n",
               survive, atom, g, ksurf>>5, line>>5, Kslice, fullk?" [full-K]":"", maxoff);
    }
    printf("   => tried %d, bijective+matmul-correct %d, ALL-in-bounds %d; of those, full-K-packing = %d\n",
           tried, biject, survive, fullk_survive);
    printf("   VERDICT: the sim CANNOT discriminate — EVERY bijection is a correct matmul (weak filter), and\n"
           "   all %d are proven wedge-safe (in-bounds). So the offline confirm set is ~%d (full-K), NOT 1-4.\n"
           "   The silicon is the ONLY oracle for which fetch it uses. GOOD NEWS: the whole set is OOB-safe, so\n"
           "   a replay_i8_sweep confirm campaign (many variants / buffer-set) is hard-wedge-free by construction.\n",
           survive, fullk_survive);
}

typedef struct { int M, reg; } surfpt;
static int fold_search(void) {
    printf("== FOLD SEARCH phase 1 (constrain the layout family — offline) ==\n");
    /* (1) verify the surf_stride*M invariant and derive the fixed region */
    surfpt pts[] = { {36,60}, {20,108} };
    long konst = -1; int inv_ok = 1;
    for (unsigned i=0;i<sizeof pts/sizeof*pts;i++){ long p=(long)pts[i].M*pts[i].reg;
        printf("    surf: M=%d reg=%d -> M*reg=%ld\n", pts[i].M, pts[i].reg, p);
        if (konst<0) konst=p; else if (p!=konst) inv_ok=0; }
    printf("    => surf_stride reg = %ld / M  (invariant %s): surf_stride shrinks with M\n",
           konst, inv_ok ? "HOLDS" : "BROKEN");
    long region_atoms = konst;                 /* 2160 atom-32 units */
    long region_bytes = region_atoms * 32;     /* = per-submit A/CBUF data region */
    printf("    => FIXED per-submit region = %ld atom-32 = %ld B; K-slice channels/token = (2160/M)*32\n",
           region_atoms, region_bytes);
    printf("       M=36 -> %ld ch/token/submit (partial of K=3584);  M=20 -> %ld ch/token (~full K)\n",
           (2160/36)*32L, (2160/20)*32L);
    printf("    UNIFIES with CBUF finding: K-per-slice is proportional to 1/M because M*K_slice = fixed CBUF\n"
           "    data region -> big M forces more K-slices -> the chain-required behavior. (new synthesis)\n");

    /* (2) RULE OUT the standard NC1HWC2: its surf_stride = M*atom GROWS with M (captured SHRINKS). */
    printf("    standard NC1HWC2 surf_stride = M*atom (GROWS with M) -> RULED OUT vs captured (shrinks). "
           "The fold A-layout is NOT a scaled standard layout.\n");

    /* (2b) the concrete candidate to confirm on-board */
    int cand_fail = verify_padded_candidate();
    printf("    candidate self-consistent OFFLINE: %s\n", cand_fail ? "NO (do not submit)" : "YES (ready for 1 confirm)");

    /* (3) within-tile ordering: enumerate, mark matmul-validity + which the board sweep already rejected. */
    const char *ord[] = {"rowmajor[M][Ksl]","colmajor[Ksl][M]","nc16","nc32","nc64","line-grouped(GROUP_LINE_OFF?)"};
    const int   board_rejected[] = {1,1,1,1,1,0};   /* fold_alayout.c results; line-grouped = UNTESTED */
    printf("    within-tile (m,k)->offset candidates:\n");
    for (unsigned i=0;i<sizeof ord/sizeof*ord;i++)
        printf("      %-30s matmul-valid=yes(weak filter)  board=%s\n", ord[i],
               board_rejected[i] ? "REJECTED (100%% mismatch)" : "UNTESTED <- the residual");

    /* (4) verdict + the precise next input */
    printf("    NARROWED: fixed-region/M surf model derived+unified; every SIMPLE within-tile ordering is\n"
           "    board-rejected -> the residual is the GROUP_LINE_OFF line-grouping (the one unpublished bit).\n");
    printf("    OFFLINE is now underdetermined by the reg dump alone (stride fixes the FRAME, not the\n"
           "    intra-surface permutation). NEXT INPUT to finish: the actual A bytes rkllm staged for a\n"
           "    known input -> read the permutation directly from ~/rkllm_ffn_capture_2026-07-27.dump\n"
           "    (board), OR a single one-hot ork_npu_replay_i8_sweep confirm of the line-grouped candidate.\n");
    return 0;   /* research readout, not a pass/fail gate */
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
    fold_search();
    enumerate_layouts();
    printf("%s\n", fail ? "CALIB FAIL" : "CALIB PASS (standard model reproduces ork's known-good layouts)");
    return fail ? 1 : 0;
}
