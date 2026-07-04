/* examples/test_ewmul_i8.c — validate the on-NPU element-wise MULTIPLY (the SwiGLU inner silu(gate)⊙up)
 * against a CPU reference. ork_npu_ewmul_i8 computes out[m][n] = clamp_i8(round(up*silu * mult/2^shift)) on
 * the NPU via the standalone SDP element-wise op (NVDLA feature-cube marshaled internally). Each case
 * self-checks vs the CPU model and the program exits nonzero on any mismatch.
 *
 * Skips gracefully (exit 0) off-board (ork_npu_init NULL) or on a non-PPU SoC (ork_ppu_fuse_enabled 0) —
 * the on-NPU EW-mul is RE'd against the rk3588 PPU/SDP layout only.
 *
 *   make test_ewmul_i8 && sudo ./test_ewmul_i8            (board only)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include "ork_npu.h"

#define EM 8
#define EN 64

static int clampi8(long v){ if(v>127)v=127; if(v<-128)v=-128; return (int)v; }

/* one case: run on NPU, compare vs CPU clamp((up*silu*mult)>>shift); tol=0 for exact (gain=1) cases,
 * tol=1 for scaled gains (rounding). returns nonzero on failure. */
static int run_case(ork_npu *c, const char *name, const signed char *up, const signed char *si,
                    int mult, int shift, int tol){
    static int8_t out[EM*EN];
    double us = 0;
    int r = ork_npu_ewmul_i8(c, (const int8_t*)up, (const int8_t*)si, EM, EN, mult, shift, out, &us);
    if (r) { printf("  %-20s FAIL (rc=%d)\n", name, r); return 1; }
    int mism = 0, mx = 0;
    for (int i = 0; i < EM*EN; i++) {
        long prod = (long)up[i] * (long)si[i];
        int ref = clampi8((prod * mult) >> shift);
        int d = abs((int)out[i] - ref);
        if (d > tol) { mism++; if (d > mx) mx = d; }
    }
    printf("  %-20s %s mism=%d/%d max|err|=%d  gain=0x%x/2^%d  (%.1f us)\n",
           name, mism ? "FAIL" : "ok  ", mism, EM*EN, mx, mult, shift, us);
    return mism ? 1 : 0;
}

int main(void){
    ork_npu *c = ork_npu_init();
    if (!c) { printf("ork_npu_init failed (board only) — SKIP\n"); return 0; }
    if (!ork_ppu_fuse_enabled(c)) { printf("on-NPU EW-mul not enabled on this SoC — SKIP\n"); ork_npu_free(c); return 0; }

    printf("on-NPU element-wise MUL (SwiGLU inner) vs CPU ref [M=%d N=%d]:\n", EM, EN);
    static signed char up[EM*EN], si[EM*EN];
    int fail = 0;

    /* gain = 1 (mult=0x4000, shift=14): out = clamp(up*silu) EXACTLY — assert bit-exact (tol=0) */
    for (int m=0;m<EM;m++) for (int n=0;n<EN;n++){ up[m*EN+n]=(signed char)(((m*3+n)%15)-7); si[m*EN+n]=(signed char)(((n*2+m)%11)-5); }
    fail |= run_case(c, "ramps",        up, si, 0x4000, 14, 0);
    for (int i=0;i<EM*EN;i++){ up[i]=(signed char)((i&1)?5:-5); si[i]=(signed char)((i%3)-1); }
    fail |= run_case(c, "alt-sign",     up, si, 0x4000, 14, 0);
    for (int i=0;i<EM*EN;i++){ up[i]=(signed char)((i%2)?100:-100); si[i]=(signed char)((i%2)?7:-3); }
    fail |= run_case(c, "signs x100",   up, si, 0x4000, 14, 0);
    for (int i=0;i<EM*EN;i++){ up[i]=(signed char)((i%2)?0:11); si[i]=(signed char)((i%3)?0:9); }
    fail |= run_case(c, "zeros",        up, si, 0x4000, 14, 0);
    for (int i=0;i<EM*EN;i++){ up[i]=127; si[i]=(signed char)((i%2)?127:-127); }   /* clamp ±127 */
    fail |= run_case(c, "clamp 127",    up, si, 0x4000, 14, 0);

    /* scaled gains: out = clamp(round(up*silu*gain)) — allow ±1 (rounding mode) */
    for (int m=0;m<EM;m++) for (int n=0;n<EN;n++){ up[m*EN+n]=(signed char)(((m*5+n)%25)-12); si[m*EN+n]=(signed char)(((n+m*7)%20)-10); }
    fail |= run_case(c, "scaled 1/2",   up, si, 0x2000, 14, 1);
    fail |= run_case(c, "scaled 1/4",   up, si, 0x1000, 14, 1);

    ork_npu_free(c);
    printf("%s\n", fail ? "FAIL" : "ALL OK");
    return fail;
}
