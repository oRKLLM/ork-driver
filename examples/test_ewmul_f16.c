/* examples/test_ewmul_f16.c — validate the on-NPU fp16 element-wise MULTIPLY vs a CPU reference.
 * ork_npu_ewmul_f16 computes out[m][n] = up[m][n]*silu[m][n] in fp16 on the NPU (standalone SDP fp16 op,
 * NVDLA feature-cube marshaled internally). Each case checks vs the CPU fp16 product; exits nonzero on error.
 * Runs several (M,N) shapes to exercise the generalized geometry (N must be a multiple of 8 for fp16 atom-8).
 * Skips gracefully (exit 0) off-board / on a non-PPU SoC.
 *   make test_ewmul_f16 && sudo ./test_ewmul_f16            (board only)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ork_npu.h"

#define MAXE (128*512)

static int run_case(ork_npu *c, const char *name, int M, int N, const ork_f16 *up, const ork_f16 *si){
    static ork_f16 out[MAXE];
    double us = 0;
    int r = ork_npu_ewmul_f16(c, up, si, M, N, out, &us);
    if (r) { printf("  %-12s [%dx%-4d] FAIL (rc=%d)\n", name, M, N, r); return 1; }
    int bad = 0; float mx = 0;
    for (int i = 0; i < M*N; i++) {
        float u = (float)up[i], s = (float)si[i], got = (float)out[i];
        float exp = (float)(ork_f16)(u * s);                 /* fp16 product (round to fp16) */
        float e = fabsf(got - exp), tol = fabsf(exp) * 0.02f + 0.02f;   /* fp16 rounding slack */
        if (e > tol) { bad++; if (e > mx) mx = e; }
    }
    printf("  %-12s [%dx%-4d] %s bad=%d/%d max|err|=%.4f  (%.1f us)\n", name, M, N, bad ? "FAIL" : "ok  ", bad, M*N, mx, us);
    return bad ? 1 : 0;
}

static int run_shape(ork_npu *c, int M, int N){
    static ork_f16 up[MAXE], si[MAXE];
    int fail = 0;
    for (int i=0;i<M*N;i++){ up[i]=(ork_f16)((((i*3)%13)-6)*0.5f); si[i]=(ork_f16)((((i*5)%7)-3)*0.25f); }
    fail |= run_case(c, "fractional", M, N, up, si);
    for (int i=0;i<M*N;i++){ up[i]=(ork_f16)(((i&1)?3.0f:-3.0f)); si[i]=(ork_f16)((i%5)-2); }
    fail |= run_case(c, "alt-sign",  M, N, up, si);
    for (int i=0;i<M*N;i++){ up[i]=(ork_f16)(((i%17)-8)*1.5f); si[i]=(ork_f16)(((i%9)-4)*2.0f); }
    fail |= run_case(c, "larger",    M, N, up, si);
    for (int i=0;i<M*N;i++){ up[i]=(ork_f16)((i%2)?0.0f:2.5f); si[i]=(ork_f16)((i%3)?0.0f:1.75f); }
    fail |= run_case(c, "zeros",     M, N, up, si);
    return fail;
}

int main(void){
    ork_npu *c = ork_npu_init();
    if (!c) { printf("ork_npu_init failed (board only) — SKIP\n"); return 0; }
    if (!ork_ppu_fuse_enabled(c)) { printf("on-NPU EW-mul not enabled on this SoC — SKIP\n"); ork_npu_free(c); return 0; }

    static const int shapes[][2] = { {8,64}, {16,64}, {8,128}, {32,256}, {1,8}, {4,512}, {8,4096}, {128,64} };
    int fail = 0;
    for (unsigned s=0; s<sizeof(shapes)/sizeof(shapes[0]); s++){
        int M=shapes[s][0], N=shapes[s][1];
        printf("on-NPU fp16 element-wise MUL (SwiGLU inner) vs CPU ref [M=%d N=%d]:\n", M, N);
        fail |= run_shape(c, M, N);
    }
    ork_npu_free(c);
    printf("%s\n", fail ? "FAIL" : "ALL OK");
    return fail;
}
