/* examples/test_ewmul_i16.c — validate the on-NPU int16 element-wise MULTIPLY vs a CPU reference.
 * ork_npu_ewmul_i16 computes out=clamp_i16(round(up*silu*mult/2^shift)) on the NPU (standalone SDP int16 op).
 * This is the w4a4 path's EW precision (ork's int4 matmul outputs int16). Runs several (M,N) shapes to
 * exercise the generalized geometry (N must be a multiple of 8 for int16 atom-8). Exits nonzero on any
 * mismatch; skips gracefully (exit 0) off-board / non-PPU SoC.
 *   make test_ewmul_i16 && sudo ./test_ewmul_i16            (board only)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include "ork_npu.h"

#define MAXE (128*512)

static long clampi16(long v){ if(v>32767)v=32767; if(v<-32768)v=-32768; return v; }

static int run_case(ork_npu *c, const char *name, int M, int N, const int16_t *up, const int16_t *si, int mult, int shift, int tol){
    static int16_t out[MAXE];
    double us = 0;
    int r = ork_npu_ewmul_i16(c, up, si, M, N, mult, shift, out, &us);
    if (r) { printf("  %-14s [%dx%-4d] FAIL (rc=%d)\n", name, M, N, r); return 1; }
    int mism = 0; long mx = 0;
    for (int i = 0; i < M*N; i++) {
        long prod = (long)up[i] * (long)si[i];
        long ref = clampi16((prod * mult) >> shift);
        long d = labs((long)out[i] - ref);
        if (d > tol) { mism++; if (d > mx) mx = d; }
    }
    printf("  %-14s [%dx%-4d] %s mism=%d/%d max|err|=%ld  gain=0x%x/2^%d  (%.1f us)\n",
           name, M, N, mism ? "FAIL" : "ok  ", mism, M*N, mx, mult, shift, us);
    return mism ? 1 : 0;
}

static int run_shape(ork_npu *c, int M, int N){
    static int16_t up[MAXE], si[MAXE];
    int fail = 0;
    /* gain=1 (mult=0x4000, shift=14): out = clamp_i16(up*silu) EXACTLY — assert bit-exact (tol=0) */
    for (int m=0;m<M;m++) for (int n=0;n<N;n++){ up[m*N+n]=(int16_t)(((m*13+n*7)%400)-200); si[m*N+n]=(int16_t)(((n*11+m*5)%160)-80); }
    fail |= run_case(c, "mixed",     M, N, up, si, 0x4000, 14, 0);
    for (int i=0;i<M*N;i++){ up[i]=(int16_t)((i&1)?300:-300); si[i]=(int16_t)((i%5)*20-40); }
    fail |= run_case(c, "alt-sign",  M, N, up, si, 0x4000, 14, 0);
    for (int i=0;i<M*N;i++){ up[i]=(int16_t)((i%2)?4000:-4000); si[i]=(int16_t)((i%2)?30:-30); }  /* clamp int16 */
    fail |= run_case(c, "clamp",     M, N, up, si, 0x4000, 14, 0);
    for (int i=0;i<M*N;i++){ up[i]=(int16_t)((i%2)?0:123); si[i]=(int16_t)((i%3)?0:77); }
    fail |= run_case(c, "zeros",     M, N, up, si, 0x4000, 14, 0);
    /* scaled gain 1/16 -> within ±1 (rounding) */
    for (int m=0;m<M;m++) for (int n=0;n<N;n++){ up[m*N+n]=(int16_t)(((m*17+n)%600)-300); si[m*N+n]=(int16_t)(((n*3+m)%200)-100); }
    fail |= run_case(c, "scaled 1/16", M, N, up, si, 0x0400, 14, 1);
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
        printf("on-NPU int16 element-wise MUL (w4a4 path) vs CPU ref [M=%d N=%d]:\n", M, N);
        fail |= run_shape(c, M, N);
    }
    ork_npu_free(c);
    printf("%s\n", fail ? "FAIL" : "ALL OK");
    return fail;
}
