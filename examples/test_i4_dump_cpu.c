/* test_i4_dump_cpu — ork_i4_w_dump_cpu must be BYTE-IDENTICAL to ork_i4_mm_pack() + ork_w_dump().
 *
 * The CPU tiler exists so a native-W4A4 .orkpack can be built without an NPU (the int8 tier has had
 * ork_i8_w_dump_cpu for a while; int4 was the last tier pinned to the board just to write a file). It
 * reproduces the CNA's weight TILE LAYOUT, and a layout can be subtly wrong in ways no accuracy test
 * would catch — a swapped nibble order or an off-by-one page stride still "works" until the NPU reads it.
 *
 * So this asserts the strongest available property: pack the SAME weight through the NPU, dump it, and
 * memcmp. Identical or fail — there is no tolerance to hide in. Shapes cover a single tile, a K-slice
 * (K > ORK_I4_KS), a wide-N slice (N > nmax forces Sn>1, exercising the Sn-major/Sk-minor walk order),
 * and a non-power-of-two K, since the page-pad only shows up when Kp*Nc/2 is not page-aligned.
 * Board only (the oracle needs the NPU); the tiler under test does not. */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint32_t g = 987654321u;
static int8_t r4(void){ g ^= g<<13; g ^= g>>17; g ^= g<<5; return (int8_t)(((int)(g & 0xf)) - 8); }  /* int4 [-8,7] */

static int one(ork_npu *c, ork_npu *off, int K, int N, const char *tag) {
    int8_t *B = malloc((size_t)K*N);
    if (!B) { printf("  [%-10s] OOM\n", tag); return 1; }
    for (size_t i = 0; i < (size_t)K*N; i++) B[i] = r4();

    size_t need = ork_i4_w_dump_cpu(c, K, N, B, NULL, 0);          /* size query */
    if (!need) { printf("  [%-10s] K=%d N=%d: size query returned 0 FAIL\n", tag, K, N); free(B); return 1; }

    ork_w *w = ork_i4_mm_pack(c, K, N, B);                          /* the ORACLE: pack on the NPU ... */
    if (!w) { printf("  [%-10s] K=%d N=%d: ork_i4_mm_pack FAIL\n", tag, K, N); free(B); return 1; }
    size_t nref = ork_w_dump(w, NULL, 0);
    uint8_t *ref = malloc(nref ? nref : 1), *cpu = malloc(need);
    if (!ref || !cpu) { printf("  [%-10s] OOM\n", tag); ork_mm_free(c, w); free(B); return 1; }
    ork_w_dump(w, ref, nref);                                       /* ... and read its bytes back */
    ork_mm_free(c, w);

    int fail = 0;
    if (nref != need) { printf("  [%-10s] K=%d N=%d: size %zu (cpu) != %zu (npu) FAIL\n", tag, K, N, need, nref); fail = 1; }
    else {
        memset(cpu, 0xAA, need);                                    /* poison so a short write is caught */
        if (ork_i4_w_dump_cpu(c, K, N, B, cpu, need) != need) { printf("  [%-10s] cpu dump short FAIL\n", tag); fail = 1; }
        else {
            size_t bad = 0, first = (size_t)-1;
            for (size_t i = 0; i < need; i++) if (cpu[i] != ref[i]) { if (first == (size_t)-1) first = i; bad++; }
            if (bad) { printf("  [%-10s] K=%d N=%d: %zu/%zu bytes differ (first @%zu: cpu=0x%02x npu=0x%02x) FAIL\n",
                              tag, K, N, bad, need, first, cpu[first], ref[first]); fail = 1; }
            else {
                /* OFFLINE arm: the same tiler driven by a context with NO device (ork_npu_init_offline).
                 * This is the property that lets a .orkpack be built off-board — and it has to be asserted
                 * against the NPU's own bytes, not merely against the on-board CPU tiler, because the whole
                 * risk of moving pack builds to another machine is that the caps used there differ. */
                uint8_t *offb = malloc(need);
                if (!offb) { printf("  [%-10s] OOM\n", tag); fail = 1; }
                else {
                    memset(offb, 0x55, need);
                    size_t no = ork_i4_w_dump_cpu(off, K, N, B, offb, need);
                    if (no != need)              { printf("  [%-10s] offline size %zu != %zu FAIL\n", tag, no, need); fail = 1; }
                    else if (memcmp(offb, ref, need)) {
                        size_t bo = 0, fo = (size_t)-1;
                        for (size_t i = 0; i < need; i++) if (offb[i] != ref[i]) { if (fo == (size_t)-1) fo = i; bo++; }
                        printf("  [%-10s] K=%d N=%d: OFFLINE differs from NPU in %zu/%zu bytes (first @%zu) FAIL\n",
                               tag, K, N, bo, need, fo); fail = 1;
                    }
                    else {
                        /* ROUND TRIP the offline LOADER. ork_i4_w_dump_cpu is asserted byte-identical above,
                         * but ork_i4_mm_load's offline un-tiler — the inverse walk that lets a pack be READ
                         * without an NPU — had no test at all. An un-tiler that is subtly wrong yields a
                         * model that still runs and still produces plausible perplexity, which is exactly
                         * the failure mode that wasted hours here: offline and board disagreed by 54% on
                         * one pack and the loader was never on the suspect list because it was untested.
                         * dump -> load -> dump must be a fixed point. */
                        ork_w *rw = ork_i4_mm_load(off, K, N, offb, need);
                        if (!rw) { printf("  [%-10s] offline LOAD returned NULL FAIL\n", tag); fail = 1; }
                        else {
                            size_t n2 = ork_w_dump(rw, NULL, 0);
                            uint8_t *b2 = malloc(n2 ? n2 : 1);
                            if (n2 != need || !b2) { printf("  [%-10s] round-trip size %zu != %zu FAIL\n", tag, n2, need); fail = 1; }
                            else {
                                ork_w_dump(rw, b2, n2);
                                size_t bad2 = 0, f2 = (size_t)-1;
                                for (size_t i = 0; i < need; i++) if (b2[i] != offb[i]) { if (f2 == (size_t)-1) f2 = i; bad2++; }
                                if (bad2) { printf("  [%-10s] K=%d N=%d: LOAD round-trip differs in %zu/%zu bytes (first @%zu: %02x vs %02x) FAIL\n",
                                                   tag, K, N, bad2, need, f2, b2[f2], offb[f2]); fail = 1; }
                                else printf("  [%-10s] K=%-5d N=%-6d %8zu bytes BYTE-IDENTICAL (npu == cpu == offline, load round-trips)\n", tag, K, N, need);
                            }
                            free(b2); ork_mm_free(off, rw);
                        }
                    }
                    free(offb);
                }
            }
        }
    }
    free(ref); free(cpu); free(B);
    return fail;
}

int main(void) {
    ork_npu *c = ork_npu_init();
    if (!c) { printf("test_i4_dump_cpu: no NPU — skipping\n"); return 0; }
    printf("test_i4_dump_cpu: ork_i4_w_dump_cpu vs ork_i4_mm_pack+ork_w_dump (SoC=%s)\n", ork_npu_soc(c));
    /* Same SoC as the live device — an offline context tiled for a DIFFERENT SoC is precisely the silent
     * mis-pack this test exists to catch, so pin it to what the board actually reports. */
    ork_npu *off = ork_npu_init_offline(ork_npu_soc(c));
    if (!off) { printf("test_i4_dump_cpu: ork_npu_init_offline(%s) FAILED\n", ork_npu_soc(c)); ork_npu_free(c); return 1; }

    int fail = 0;
    fail |= one(c, off, 1024,  1024, "single");    /* one tile */
    fail |= one(c, off, 3584,  1024, "ksplit");    /* K > ORK_I4_KS -> Sk>1 */
    fail |= one(c, off, 1024, 16384, "widen");     /* N > nmax      -> Sn>1, exercises the walk order */
    fail |= one(c, off, 2560,   576, "oddshape");  /* non-pow2 K and N%64 -> page-pad is non-trivial */
    printf("test_i4_dump_cpu: %s\n", fail ? "FAIL" : "ALL BYTE-IDENTICAL (offline packing is safe)");
    ork_npu_free(off);
    ork_npu_free(c);
    return fail;
}
