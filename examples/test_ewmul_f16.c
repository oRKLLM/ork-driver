/* examples/test_ewmul_f16.c — validate the on-NPU fp16 element-wise MULTIPLY vs a CPU reference.
 * ork_npu_ewmul_f16 computes out[m][n] = up[m][n]*silu[m][n] in fp16 on the NPU (standalone SDP fp16 op,
 * NVDLA feature-cube marshaled internally). Each case checks vs the CPU fp16 product; exits nonzero on error.
 * Skips gracefully (exit 0) off-board / on a non-PPU SoC.
 *   make test_ewmul_f16 && sudo ./test_ewmul_f16            (board only)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ork_npu.h"

#define EM 8
#define EN 64

static int run_case(ork_npu *c, const char *name, const ork_f16 *up, const ork_f16 *si){
    static ork_f16 out[EM*EN];
    double us = 0;
    int r = ork_npu_ewmul_f16(c, up, si, EM, EN, out, &us);
    if (r) { printf("  %-14s FAIL (rc=%d)\n", name, r); return 1; }
    int bad = 0; float mx = 0;
    for (int i = 0; i < EM*EN; i++) {
        float u = (float)up[i], s = (float)si[i], got = (float)out[i];
        float exp = (float)(ork_f16)(u * s);                 /* fp16 product (round to fp16) */
        float e = fabsf(got - exp), tol = fabsf(exp) * 0.02f + 0.02f;   /* fp16 rounding slack */
        if (e > tol) { bad++; if (e > mx) mx = e; }
    }
    printf("  %-14s %s bad=%d/%d max|err|=%.4f  (%.1f us)\n", name, bad ? "FAIL" : "ok  ", bad, EM*EN, mx, us);
    return bad ? 1 : 0;
}

int main(void){
    ork_npu *c = ork_npu_init();
    if (!c) { printf("ork_npu_init failed (board only) — SKIP\n"); return 0; }
    if (!ork_ppu_fuse_enabled(c)) { printf("on-NPU EW-mul not enabled on this SoC — SKIP\n"); ork_npu_free(c); return 0; }

    printf("on-NPU fp16 element-wise MUL (SwiGLU inner) vs CPU ref [M=%d N=%d]:\n", EM, EN);
    static ork_f16 up[EM*EN], si[EM*EN];
    int fail = 0;

    for (int i=0;i<EM*EN;i++){ up[i]=(ork_f16)((((i*3)%13)-6)*0.5f); si[i]=(ork_f16)((((i*5)%7)-3)*0.25f); }
    fail |= run_case(c, "fractional", up, si);
    for (int i=0;i<EM*EN;i++){ up[i]=(ork_f16)(((i&1)?3.0f:-3.0f)); si[i]=(ork_f16)((i%5)-2); }
    fail |= run_case(c, "alt-sign",  up, si);
    for (int i=0;i<EM*EN;i++){ up[i]=(ork_f16)(((i%17)-8)*1.5f); si[i]=(ork_f16)(((i%9)-4)*2.0f); }
    fail |= run_case(c, "larger",    up, si);
    for (int i=0;i<EM*EN;i++){ up[i]=(ork_f16)((i%2)?0.0f:2.5f); si[i]=(ork_f16)((i%3)?0.0f:1.75f); }
    fail |= run_case(c, "zeros",     up, si);

    ork_npu_free(c);
    printf("%s\n", fail ? "FAIL" : "ALL OK");
    return fail;
}
