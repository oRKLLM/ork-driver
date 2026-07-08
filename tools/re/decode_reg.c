/* decode_reg.c — decode an ork-driver regcmd dump into (domain, addr, value) triples, and optionally diff two.
 *
 * The RK NPU regcmd stream is a sequence of 64-bit register writes. Each write is two u32 words as emitted by
 * tools/re/regcmd_capture.c hexdumps ("[nnn] w0 w1 w2 w3"):
 *     word0 = addr(low16)   | (value_low16  << 16)
 *     word1 = value_high16  | (domain(e.g. 0x1001/0x2001) << 16)
 * i.e. reg[addr] = ((word1 & 0xffff) << 16) | (word0 >> 16), in register domain (word1 >> 16).
 *
 * Usage:
 *   decode_reg < dump.txt                 # decode all [nnn]-prefixed hex lines from a capture dump
 *   decode_reg a.txt b.txt                # decode both and print only the registers that DIFFER (RE diff)
 *
 * Feed it the region between a "--- regcmd (N u32 words) ---" header and the trailer. It ignores non-hex
 * lines, so piping a whole dump section is fine. RE tool only; builds with plain cc.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

struct reg { unsigned dom, addr; uint32_t val; };

/* parse a dump's "[nnn] hhhhhhhh hhhhhhhh ..." lines into a flat u32 array; returns count */
static int parse_words(FILE *f, uint32_t *w, int max) {
    char line[512]; int n = 0;
    while (fgets(line, sizeof line, f)) {
        char *p = strchr(line, ']');            /* skip the "[nnn]" index prefix if present */
        p = p ? p + 1 : line;
        char *tok = strtok(p, " \t\n");
        while (tok && n < max) {
            /* accept only 8-hex-digit tokens */
            int len = strlen(tok), ok = (len == 8);
            for (int i = 0; ok && i < 8; i++) if (!((tok[i]>='0'&&tok[i]<='9')||(tok[i]>='a'&&tok[i]<='f')||(tok[i]>='A'&&tok[i]<='F'))) ok = 0;
            if (ok) w[n++] = (uint32_t)strtoul(tok, NULL, 16);
            tok = strtok(NULL, " \t\n");
        }
    }
    return n;
}

/* decode word pairs into register writes; stops at the all-zero / trailer region */
static int decode(uint32_t *w, int nw, struct reg *r, int max) {
    int n = 0;
    for (int k = 0; k + 1 < nw && n < max; k += 2) {
        uint32_t w0 = w[k], w1 = w[k+1];
        unsigned addr = w0 & 0xffff, dom = w1 >> 16;
        if (addr == 0 && w1 == 0) break;        /* trailer / padding */
        /* keep all known register blocks: 0101 PC, 0201 CNA, 0801 DPU, 1001 PPU/core, 2001 CDMA.
         * (previously only 1001/2001 — which silently dropped the CNA/DPU blocks that carry K dims,
         *  weight IOVA, and 0x107c CBUF entries-per-slice; do NOT re-narrow this filter.) */
        if (dom != 0x0101 && dom != 0x0201 && dom != 0x0801 && dom != 0x1001 && dom != 0x2001) continue;
        uint32_t val = ((w1 & 0xffff) << 16) | (w0 >> 16);
        r[n++] = (struct reg){ dom, addr, val };
    }
    return n;
}

static uint32_t lookup(struct reg *r, int n, unsigned dom, unsigned addr, int *found) {
    for (int i = 0; i < n; i++) if (r[i].dom == dom && r[i].addr == addr) { *found = 1; return r[i].val; }
    *found = 0; return 0;
}

int main(int argc, char **argv) {
    static uint32_t wa[8192], wb[8192];
    static struct reg ra[2048], rb[2048];
    if (argc >= 3) {
        FILE *fa = fopen(argv[1], "r"), *fb = fopen(argv[2], "r");
        if (!fa || !fb) { perror("fopen"); return 1; }
        int na = decode(wa, parse_words(fa, wa, 8192), ra, 2048);
        int nb = decode(wb, parse_words(fb, wb, 8192), rb, 2048);
        printf("# DIFF %s (%d regs) vs %s (%d regs)\n", argv[1], na, argv[2], nb);
        printf("# %-6s %-6s %-10s %-10s\n", "dom", "addr", argv[1], argv[2]);
        for (int i = 0; i < na; i++) {
            int f; uint32_t vb = lookup(rb, nb, ra[i].dom, ra[i].addr, &f);
            if (!f) printf("  %04x   %04x   %08x   <absent>\n", ra[i].dom, ra[i].addr, ra[i].val);
            else if (vb != ra[i].val) printf("  %04x   %04x   %08x   %08x\n", ra[i].dom, ra[i].addr, ra[i].val, vb);
        }
        for (int i = 0; i < nb; i++) { int f; lookup(ra, na, rb[i].dom, rb[i].addr, &f); if (!f) printf("  %04x   %04x   <absent>   %08x\n", rb[i].dom, rb[i].addr, rb[i].val); }
        return 0;
    }
    int n = decode(wa, parse_words(stdin, wa, 8192), ra, 2048);
    printf("# %d register writes\n# dom  addr    value\n", n);
    for (int i = 0; i < n; i++) printf("  %04x  0x%04x  0x%08x\n", ra[i].dom, ra[i].addr, ra[i].val);
    return 0;
}
