/* test_pack_gate — the persisted-weight header must actually REFUSE, not merely exist.
 *
 * A gate that never fires is worse than no gate: it reads as protection while admitting exactly the
 * packs it was added to stop. The round-trip tests (test_offline_load, test_f16_load,
 * test_i4_dump_cpu) prove the header does not BREAK anything; this proves it does something. Every
 * case below is a blob the loader must reject, and the honest round-trip must still be accepted —
 * a gate that refuses everything would pass a refusal-only test.
 *
 * The cases are the real failure modes, not arbitrary corruption:
 *   - NO HEADER: a .orkpack written by a build before the header existed. This is the one that
 *     actually ships, because packs are a derived cache and some will be stale on disk.
 *   - WRONG capid: built against another SoC's geometry, or an envelope rule that has since moved.
 *     Byte-for-byte the same SIZE as a good pack, which is precisely why ork/context.h said
 *     ork-driver "cannot self-detect" it before.
 *   - WRONG fmt: the tile layout itself changed.
 *   - WRONG K/N/dtype in the header: the container handed us the wrong weight's bytes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ork_npu.h"

static int fails = 0;

static void must_reject(const char *what, ork_npu *c, int K, int N,
                        void *blob, size_t n, int f16)
{
    ork_w *w = f16 ? ork_f16_mm_load(c, K, N, blob, n) : ork_i8_mm_load(c, K, N, blob, n);
    printf("  %-44s %s\n", what, w ? "ACCEPTED — gate is inert, FAIL" : "refused  ok");
    if (w) { fails++; ork_mm_free(c, w); }
}

int main(void)
{
    ork_npu *c = ork_npu_init();
    if (!c) { printf("ork_npu_init failed\n"); return 1; }
    const int K = 512, N = 256;
    printf("persisted-weight header: the gate must refuse (hdr=%zu bytes)\n\n", ork_pack_hdr_bytes());

    /* ---- int8 ---- */
    int8_t *B = malloc((size_t)K*N);
    for (size_t i = 0; i < (size_t)K*N; i++) B[i] = (int8_t)((i*37) & 0x7f) - 64;
    ork_w *w = ork_i8_mm_pack(c, K, N, B);
    if (!w) { printf("int8 pack failed\n"); return 1; }
    size_t n = ork_w_dump(w, NULL, 0);
    unsigned char *good = malloc(n);
    if (ork_w_dump(w, good, n) != n) { printf("int8 dump short\n"); return 1; }
    ork_mm_free(c, w);

    /* the honest blob must LOAD — otherwise every refusal below is vacuous */
    ork_w *ok = ork_i8_mm_load(c, K, N, good, n);
    printf("  %-44s %s\n", "an honest stamped pack", ok ? "loaded   ok" : "REFUSED — gate is too strict, FAIL");
    if (!ok) fails++; else ork_mm_free(c, ok);

    unsigned char *bad = malloc(n);

    memcpy(bad, good, n); ((uint32_t*)bad)[0] ^= 0xffu;          /* magic */
    must_reject("corrupt magic", c, K, N, bad, n, 0);

    memcpy(bad, good, n); ((uint32_t*)bad)[1] += 1u;             /* fmt */
    must_reject("pack from a different format version", c, K, N, bad, n, 0);

    memcpy(bad, good, n); ((uint32_t*)bad)[2] ^= 0x5a5au;        /* capid */
    must_reject("pack built against different caps/SoC", c, K, N, bad, n, 0);

    memcpy(bad, good, n); ((int32_t*)bad)[4] = K*2;              /* K */
    must_reject("header K disagrees with the request", c, K, N, bad, n, 0);

    memcpy(bad, good, n); ((uint32_t*)bad)[3] = 0xdeadu;         /* dtype */
    must_reject("header dtype disagrees", c, K, N, bad, n, 0);

    /* a pre-header pack: the body alone, exactly what an older build wrote */
    must_reject("UNSTAMPED pack (pre-header build)", c, K, N,
                good + ork_pack_hdr_bytes(), n - ork_pack_hdr_bytes(), 0);

    /* truncated to less than a header */
    must_reject("blob shorter than the header", c, K, N, good, 16, 0);

    free(bad); free(good); free(B);

    /* ---- fp16: the same gate, through a different loader ---- */
    printf("\n  fp16 loader, same gate:\n");
    ork_f16 *Bf = malloc((size_t)K*N*sizeof *Bf);
    for (size_t i = 0; i < (size_t)K*N; i++) Bf[i] = (ork_f16)(((int)(i%17)-8)*0.01f);
    ork_w *wf = ork_f16_mm_pack(c, K, N, Bf);
    if (!wf) { printf("fp16 pack failed\n"); return 1; }
    size_t nf = ork_w_dump(wf, NULL, 0);
    unsigned char *gf = malloc(nf);
    ork_w_dump(wf, gf, nf); ork_mm_free(c, wf);
    ork_w *okf = ork_f16_mm_load(c, K, N, gf, nf);
    printf("  %-44s %s\n", "an honest stamped fp16 pack", okf ? "loaded   ok" : "REFUSED — FAIL");
    if (!okf) fails++; else ork_mm_free(c, okf);
    unsigned char *bf = malloc(nf);
    memcpy(bf, gf, nf); ((uint32_t*)bf)[2] ^= 0x1234u;
    must_reject("fp16 pack with a foreign capid", c, K, N, bf, nf, 1);
    must_reject("fp16 UNSTAMPED pack", c, K, N, gf + ork_pack_hdr_bytes(), nf - ork_pack_hdr_bytes(), 1);
    free(bf); free(gf); free(Bf);

    ork_npu_free(c);
    printf("\n%s\n", fails == 0 ? "PACK GATE OK" : "PACK GATE FAILED");
    return fails != 0;
}
