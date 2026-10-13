/* tools/re/probe_gate.h — the two gates a CONFIGURATION probe must pass before its timing means
 * anything. Header-only, no dependencies beyond libc, so including it costs a probe nothing.
 *
 * WHY THIS EXISTS. On 2026-10-12 a Wb-geometry sweep at the frontier produced, in the same table:
 *
 *   - the INCORRECT configurations were the FASTEST rows in the sweep (1714 us wrong vs 1767 us
 *     correct). Ranking by time therefore SELECTS the broken config. This is not bad luck: a config
 *     that drops work, mis-tiles M or reads a short extent finishes sooner precisely because it did
 *     less. "Fastest" is evidence of a defect as readily as of a win.
 *   - a checksum taken on the probe's PATTERNED input agreed on geometries that were wrong. Re-running
 *     the identical geometries on full-random input moved a published table from 0.0% wrong to 100.0%
 *     wrong. Structured inputs make distinct tilings coincide; random inputs do not.
 *
 * So neither gate is sufficient alone, and a probe that reports a winner on one of them is not a weaker
 * probe -- it is a probe that can print the wrong answer with full confidence. Both, on random data, or
 * the number is not admissible. See OPS_REGISTRY.md "Probe admissibility".
 *
 * USE: fill inputs with ork_probe_fill_random, take ork_probe_ck of each output, and pass every
 * candidate through ork_probe_gate against an anchor configuration measured at the SAME shape. Print
 * the ranking only from rows it returned 1 for.
 */
#ifndef ORK_PROBE_GATE_H
#define ORK_PROBE_GATE_H

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

/* Full-random bytes from a named seed. Reproducible across runs and boards (it is its own LCG, not
 * rand()), which is what makes an anchor checksum comparable at all, but NOT patterned: every byte is
 * independent. Use this for weights AND activations -- randomising only one side still lets distinct
 * tilings coincide on the other. */
static inline void ork_probe_fill_random(void *buf, size_t n, unsigned seed) {
    unsigned char *b = (unsigned char *) buf;
    uint32_t s = seed ? seed : 0x9E3779B9u;
    for (size_t i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; b[i] = (unsigned char) (s >> 24); }
}

static inline unsigned long long ork_probe_ck(const void *p, size_t n) {   /* FNV-1a */
    const unsigned char *b = (const unsigned char *) p;
    unsigned long long h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

/* Returns 1 iff this candidate's number may be reported. `anchor_us <= 0` means "no time gate", for a
 * probe whose anchor is correctness only.
 *
 * The time gate is for the OTHER direction of the same mistake: the driver self-heals a timed-out
 * submit and eventually returns the RIGHT answer, so a config can be bit-exact and still be broken --
 * it just paid ~1000x. A candidate that matches the anchor but took far longer than it is a defect
 * wearing a correct checksum, which is the reason `rc == 0` is not a gate either. */
static inline int ork_probe_gate(const char *label,
                                 unsigned long long ck, unsigned long long anchor_ck,
                                 double us, double anchor_us) {
    if (ck != anchor_ck) {
        fprintf(stderr, "[probe-gate] %s REJECTED: checksum %016llx != anchor %016llx"
                        " (%.1f us -- do NOT rank it; a wrong config is typically the FAST one)\n",
                label, ck, anchor_ck, us);
        return 0;
    }
    if (anchor_us > 0 && us > 4.0 * anchor_us) {
        fprintf(stderr, "[probe-gate] %s REJECTED: bit-exact but %.1f us vs anchor %.1f us (%.0fx)"
                        " -- that is the self-heal path returning the right answer late\n",
                label, us, anchor_us, us / anchor_us);
        return 0;
    }
    return 1;
}

#endif /* ORK_PROBE_GATE_H */
