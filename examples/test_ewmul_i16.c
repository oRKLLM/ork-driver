/* examples/test_ewmul_i16.c — validate the on-NPU int16 element-wise MULTIPLY vs a CPU reference.
 * ork_npu_ewmul_i16 computes out=clamp_i16(round(up*silu*mult/2^shift)) on the NPU (standalone SDP int16 op).
 * This is the w4a4 path's EW precision (ork's int4 matmul outputs int16). Exits nonzero on any mismatch;
 * skips gracefully (exit 0) off-board / non-PPU SoC.
 *   make test_ewmul_i16 && sudo ./test_ewmul_i16            (board only)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include "ork_npu.h"

#define EM 8
#define EN 64

static long clampi16(long v){ if(v>32767)v=32767; if(v<-32768)v=-32768; return v; }

static int run_case(ork_npu *c, const char *name, const int16_t *up, const int16_t *si, int mult, int shift, int tol){
    static int16_t out[EM*EN];
    double us = 0;
    int r = ork_npu_ewmul_i16(c, up, si, EM, EN, mult, shift, out, &us);
    if (r) { printf("  %-16s FAIL (rc=%d)\n", name, r); return 1; }
    int mism = 0; long mx = 0;
    for (int i = 0; i < EM*EN; i++) {
        long prod = (long)up[i] * (long)si[i];
        long ref = clampi16((prod * mult) >> shift);
        long d = labs((long)out[i] - ref);
        if (d > tol) { mism++; if (d > mx) mx = d; }
    }
    printf("  %-16s %s mism=%d/%d max|err|=%ld  gain=0x%x/2^%d  (%.1f us)\n",
           name, mism ? "FAIL" : "ok  ", mism, EM*EN, mx, mult, shift, us);
    return mism ? 1 : 0;
}

int main(void){
    ork_npu *c = ork_npu_init();
    if (!c) { printf("ork_npu_init failed (board only) — SKIP\n"); return 0; }
    if (!ork_ppu_fuse_enabled(c)) { printf("on-NPU EW-mul not enabled on this SoC — SKIP\n"); ork_npu_free(c); return 0; }

    printf("on-NPU int16 element-wise MUL (w4a4 path) vs CPU ref [M=%d N=%d]:\n", EM, EN);
    static int16_t up[EM*EN], si[EM*EN];
    int fail = 0;

    /* gain=1 (mult=0x4000, shift=14): out = clamp_i16(up*silu) EXACTLY — assert bit-exact (tol=0) */
    for (int m=0;m<EM;m++) for (int n=0;n<EN;n++){ up[m*EN+n]=(int16_t)(((m*13+n*7)%400)-200); si[m*EN+n]=(int16_t)(((n*11+m*5)%160)-80); }
    fail |= run_case(c, "mixed",     up, si, 0x4000, 14, 0);
    for (int i=0;i<EM*EN;i++){ up[i]=(int16_t)((i&1)?300:-300); si[i]=(int16_t)((i%5)*20-40); }
    fail |= run_case(c, "alt-sign",  up, si, 0x4000, 14, 0);
    for (int i=0;i<EM*EN;i++){ up[i]=(int16_t)((i%2)?4000:-4000); si[i]=(int16_t)((i%2)?30:-30); }  /* clamp int16 */
    fail |= run_case(c, "clamp",     up, si, 0x4000, 14, 0);
    for (int i=0;i<EM*EN;i++){ up[i]=(int16_t)((i%2)?0:123); si[i]=(int16_t)((i%3)?0:77); }
    fail |= run_case(c, "zeros",     up, si, 0x4000, 14, 0);
    /* scaled gain 1/16 -> within ±1 (rounding) */
    for (int m=0;m<EM;m++) for (int n=0;n<EN;n++){ up[m*EN+n]=(int16_t)(((m*17+n)%600)-300); si[m*EN+n]=(int16_t)(((n*3+m)%200)-100); }
    fail |= run_case(c, "scaled 1/16", up, si, 0x0400, 14, 1);

    ork_npu_free(c);
    printf("%s\n", fail ? "FAIL" : "ALL OK");
    return fail;
}
