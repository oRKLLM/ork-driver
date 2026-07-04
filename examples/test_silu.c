/* examples/test_silu.c — validate the on-NPU standalone SiLU (ork_npu_silu_i8) vs a CPU reference.
 * ork_npu_silu_i8 computes out[m][n] = clamp_i8(round( silu(in*in_scale)/out_scale )) on the NPU via the
 * standalone SDP activation-LUT op (index map calibrated once per ctx, silu curve built per scale). Each
 * case self-checks vs the CPU model over several (M,N) shapes and scales; exits nonzero on any mismatch.
 * Skips gracefully (exit 0) off-board / on a non-PPU SoC.
 *   make test_silu && sudo ./test_silu            (board only)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ork_npu.h"

#define MAXE (128*512)

static int clampi8(long v){ if(v>127)v=127; if(v<-128)v=-128; return (int)v; }
static double siluf(double x){ return x/(1.0+exp(-x)); }

/* one case: NPU silu vs CPU clamp_i8(round(silu(in*in_scale)/out_scale)). LUT interp + int8 rounding -> tol=2. */
static int run_case(ork_npu *c, int M, int N, double in_scale, double out_scale, int tol){
    static signed char in[MAXE], out[MAXE];
    for(int i=0;i<M*N;i++) in[i]=(signed char)(((i*7)%256)-128);   /* spread across the int8 range */
    double us=0;
    int r = ork_npu_silu_i8(c, in, M, N, in_scale, out_scale, out, &us);
    if(r){ printf("  [%dx%-4d] is=%.4f os=%.4f FAIL (rc=%d)\n", M, N, in_scale, out_scale, r); return 1; }
    int mism=0, mx=0;
    for(int i=0;i<M*N;i++){
        int ref = clampi8(lround(siluf(in[i]*in_scale)/out_scale));
        int d = abs((int)out[i]-ref); if(d>tol){ mism++; if(d>mx)mx=d; }
    }
    printf("  [%dx%-4d] is=%.4f os=%.4f %s mism=%d/%d max|err|=%d  (%.1f us)\n",
           M, N, in_scale, out_scale, mism?"FAIL":"ok  ", mism, M*N, mx, us);
    return mism?1:0;
}

int main(void){
    ork_npu *c = ork_npu_init();
    if(!c){ printf("ork_npu_init failed (board only) — SKIP\n"); return 0; }
    if(!ork_ppu_fuse_enabled(c)){ printf("on-NPU SiLU not enabled on this SoC — SKIP\n"); ork_npu_free(c); return 0; }

    printf("on-NPU standalone SiLU vs CPU ref:\n");
    int fail = 0;
    static const int shapes[][2] = { {8,64}, {16,64}, {8,128}, {32,256}, {1,16}, {4,512}, {8,4096} };
    for(unsigned s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++){
        int M=shapes[s][0], N=shapes[s][1];
        fail |= run_case(c, M, N, 0.0625, 0.0625, 2);
    }
    /* a couple of alternate scales at the base shape */
    fail |= run_case(c, 8, 64, 0.03125, 0.0625, 2);
    fail |= run_case(c, 8, 64, 0.05,    0.05,   2);

    ork_npu_free(c);
    printf("%s\n", fail ? "FAIL" : "ALL OK");
    return fail;
}
