/* f16_bankdiff — OFFLINE regcmd diff for the narrow-N wedge. No board, no NPU, no wedge risk.
 *
 * r103 reported: an fp16 matmul with narrow N hangs once the M-chunk exceeds 128 rows at K=1024
 * (128 PASS, 144 HANG, N=16 and N=32 both). Each on-board reproduction costs a power cycle, so the
 * first question -- what actually differs between the emitted programs -- is answered here instead,
 * by calling the REAL emitter (orki_f16_synth) and decoding the register that changes.
 *
 * 0x1040 is RK_CNA_CBUF_CON0, the CBUF bank split: DATA_BANK[3:0] / WEIGHT_BANK[7:4], 12 banks of
 * 32 KB. orki_f16_synth computes it as
 *      scale = K/256,  base = 177 - 15(scale-1),  slope = 15*scale,  mg = ceil(mc/64)
 *      v = base - slope*(mg-1),  clamped UP to 0x1b
 * so WEIGHT_BANK shrinks as the row-group count grows, and the clamp is what happens when the
 * formula would drive it to zero. This prints v, both bank fields, their byte capacities, and what
 * the operands actually need -- which is the comparison the hang hypothesis turns on.
 *
 * Build (host or board, no device needed):
 *   cc -O2 -Iinclude -Isrc -o /tmp/bd tools/f16_bankdiff.c src/npu/f16/regcmd.c -lm   # if it links alone
 *   or: make f16_bankdiff
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"
#include "npu/internal.h"

#ifndef REGCMD_N
/* regcmd.c owns the template length; we only need a buffer big enough to hand it. */
#define RC_WORDS 4096
#else
#define RC_WORDS REGCMD_N
#endif

static void row(int mc, int K, int N)
{
    int sched = orki_f16_sched(K);
    int cap   = orki_f16_mcap(K, sched);

    /* Recompute the split the way orki_f16_synth does, so we can decode it without reaching into
     * the emitted buffer's template offsets. Verified against the emitter below. */
    double scale = (double)K / 256.0;
    int base  = (int)(177.0 - 15.0 * (scale - 1.0));
    int slope = (int)(15.0 * scale);
    int mg    = (mc + 63) / 64; if (mg < 1) mg = 1;
    int raw   = sched ? base - slope * (mg - 1) : 0xb1;   /* sched=0 keeps the template default */
    int v     = raw;
    int clamped = 0;
    if (sched && v < 0x1b) { v = 0x1b; clamped = 1; }

    int dbank = v & 0xf, wbank = (v >> 4) & 0xf;
    long dcap = (long)dbank * 32768, wcap = (long)wbank * 32768;
    long dneed = (long)mc * K * 2;            /* activation rows, fp16 */
    long wneed = (long)K * N * 2;             /* the weight segment, fp16 */

    printf("  mc=%-5d N=%-5d mg=%d  raw=%-5d v=0x%02x%s  DATA=%2d (%7ld B, need %8ld %s)  "
           "WEIGHT=%2d (%6ld B, need %8ld %s)\n",
           mc, N, mg, raw, v, clamped ? "*" : " ",
           dbank, dcap, dneed, dneed <= dcap ? "ok " : "OVER",
           wbank, wcap, wneed, wneed <  wcap ? "ok " : (wneed == wcap ? "EXACT" : "OVER"));
}

int main(int argc, char **argv)
{
    int K = argc > 1 ? atoi(argv[1]) : 1024;
    printf("fp16 CBUF bank split (0x1040 = CNA_CBUF_CON0), K=%d  sched=%d  mcap=%d\n",
           K, orki_f16_sched(K), orki_f16_mcap(K, orki_f16_sched(K)));
    printf("  * = the formula underflowed and was clamped up to 0x1b\n");
    printf("  WEIGHT 'EXACT' means the segment fills the bank with ZERO slack\n\n");

    printf("r103's reported shapes (N=16 and N=32, the hang is in mc 128->144):\n");
    int mcs[] = { 16, 64, 128, 129, 144, 176 };
    for (unsigned i = 0; i < sizeof mcs / sizeof mcs[0]; i++) row(mcs[i], K, 16);
    printf("\n");
    for (unsigned i = 0; i < sizeof mcs / sizeof mcs[0]; i++) row(mcs[i], K, 32);

    printf("\nwide N at the same row counts (the configuration that is known to WORK):\n");
    for (unsigned i = 0; i < sizeof mcs / sizeof mcs[0]; i++) row(mcs[i], K, 3584);

    printf("\nthe N-AWARE guard (orki_f16_mcap_n) -- what the run path will actually chunk to:\n");
    int Ks[] = {256, 512, 1024}, Ns[] = {16, 32, 48, 64, 128, 3584};
    for (unsigned a = 0; a < sizeof Ks/sizeof Ks[0]; a++) {
        int kk = Ks[a], sc = orki_f16_sched(kk);
        printf("  K=%-5d mcap=%-4d ->", kk, orki_f16_mcap(kk, sc));
        for (unsigned b = 0; b < sizeof Ns/sizeof Ns[0]; b++)
            printf("  N=%-5d %4d", Ns[b], orki_f16_mcap_n(kk, sc, Ns[b]));
        printf("\n");
    }

    /* Cross-check the decode against the real emitter: emit at mc=128 and mc=144 and confirm the
     * word we predicted is the word that lands. A decode that disagrees with the emitter would
     * make everything above fiction. */
    printf("\nemitter cross-check (orki_f16_synth):\n");
    uint32_t *a = calloc(RC_WORDS, 4), *b = calloc(RC_WORDS, 4);
    if (!a || !b) { printf("  alloc failed\n"); return 1; }
    orki_f16_synth(a, 128, K, 16, 0x1000, 0x2000, 0x3000, orki_f16_sched(K), 57344);
    orki_f16_synth(b, 144, K, 16, 0x1000, 0x2000, 0x3000, orki_f16_sched(K), 57344);
    int ndiff = 0;
    for (int i = 0; i < RC_WORDS; i++) if (a[i] != b[i]) {
        if (ndiff < 24) printf("  word %4d: mc=128 0x%08x   mc=144 0x%08x\n", i, a[i], b[i]);
        ndiff++;
    }
    printf("  %d words differ between the PASSING mc=128 and the HANGING mc=144\n", ndiff);
    free(a); free(b);
    return 0;
}
