/* examples/test_silu_native.c — validate the ork-NATIVE fused-SiLU LUT generator (ork_mm_silu_build_lut).
 *
 * Builds ork's own silu LUT for a given (in_scale, out_scale) via one calibration submit, then runs a
 * matmul with SiLU fused on-chip and checks the output against the correct mathematical silu. This
 * exercises the whole ork-native path (measure index(acc) -> build LUT -> fused run) end-to-end.
 *
 * Skips gracefully off-board / on non-PPU SoCs (ork_npu_init NULL, or the fused path unavailable).
 *
 *   make test_silu_native && sudo ./test_silu_native
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ork_npu.h"

#define M 1
#define K 512
#define N 64

static double silu(double x){ return x/(1.0+exp(-x)); }

static int check(ork_npu *ctx, double in_s, double out_s, int r_mult, int r_shift, uint32_t cfg4068){
    static int16_t lut[1030];
    if (ork_mm_silu_build_lut(ctx, in_s, out_s, r_mult, r_shift, cfg4068, lut)){
        printf("  build_lut FAILED (in=%.2e out=%.2e)\n", in_s, out_s); return 1;
    }
    /* run a validation matmul: acc[n] = (n-32)*step sweeping silu's active range */
    static signed char A[K], B[K*N]; static int8_t C[N];
    for (int k=0;k<K;k++) A[k]=1;
    int accmax=(int)(7.0/in_s), step=(2*accmax)/(N-1); if(step<1)step=1;
    for (int n=0;n<N;n++){ int T=-accmax+n*step; int b=T/K; for(int k=0;k<K;k++) B[k*N+n]=(signed char)(b+(k<(T-b*K)?1:0)); }
    if (ork_i8_npu_probe_silu_cfg(ctx,M,K,N,A,B,r_mult,r_shift,0u,0xffffc000u,cfg4068,lut,1030,C,0)){
        printf("  fused run FAILED\n"); return 1;
    }
    int mx=0; long s=0;
    for (int n=0;n<N;n++){ int acc=0; for(int k=0;k<K;k++) acc+=A[k]*B[k*N+n];
        double v=silu(acc*in_s)/out_s; long t=lround(v); if(t>127)t=127; if(t<-128)t=-128;
        int e=abs((int)C[n]-(int)t); s+=e; if(e>mx)mx=e; }
    double mean=(double)s/N;
    int ok = (mean < 2.0) && (mx <= 6);   /* ~int8 rounding floor */
    printf("  %s in=%.2e out=%.2e R=0x%x/2^%d: mean|err|=%.2f max=%d\n",
           ok?"ok  ":"FAIL", in_s,out_s,r_mult,r_shift,mean,mx);
    return ok?0:1;
}

int main(void){
    ork_npu *ctx = ork_npu_init();
    if (!ctx){ printf("ork_npu_init failed (board only) — SKIP\n"); return 0; }
    printf("ork-native fused SiLU (build ork's own LUT, correct silu):\n");
    int fail=0;
    /* R ~= 660*in_scale so the acc range spans silu's transition; cfg4068 fixed */
    fail |= check(ctx, 3.75e-4, 0.05, 0x4000, 0x10, 0x56391300u);
    fail |= check(ctx, 7.50e-4, 0.10, 0x4000, 0x10, 0x56391300u);
    fail |= check(ctx, 2.50e-4, 0.04, 0x4000, 0x10, 0x56391300u);
    ork_npu_free(ctx);
    printf("%s\n", fail?"FAIL":"ALL OK");
    return fail;
}
