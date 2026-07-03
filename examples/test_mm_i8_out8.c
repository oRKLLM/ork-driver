/* examples/test_mm_i8_out8.c — validate the PPU int8-REQUANTIZED output stage (fused-path step 1).
 *
 * The matmul normally writes an int32 accumulator (INT8_MM_INT8_TO_INT32); the caller then requantizes
 * on the CPU/NEON. ork_npu_probe_i8_out8 instead runs the matmul with the int8-output stage on-chip
 * (set_i8_out8): out_i8 = clamp_i8((acc_i32 * mult) >> shift). This is the foundation the fused SiLU
 * (LUT) and the SwiGLU elementwise-mul are layered on. Here we prove it bit-exact vs the CPU model,
 * for identity requant (mult=0x4000, shift=14 -> pass-through-then-clamp) and a couple of scales.
 *
 * NOTE: this exercises the on-NPU fused-output path directly (via the probe), independent of the
 * ork_ppu_fuse_enabled() runtime gate — the gate governs the *production* run path, which keeps the
 * CPU/NEON requant as its default fallback. See src/npu.c set_i8_out8 for the design rationale.
 *
 *   make test_mm_i8_out8 && sudo ./test_mm_i8_out8       (board only — needs /dev/dri + rknpu)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include "ork_npu.h"

static unsigned sd = 0x2468ace;
static int rnd(void){ sd = sd*1103515245u + 12345u; return (sd>>16)&0xff; }

/* CPU reference for the int8-output requant stage. The hardware applies out =
 * clamp_i8( round_half_to_even(acc * mult / 2^shift) ) — convergent (banker's) rounding, verified
 * bit-exact on silicon (ties to even: 30.5->30, 39.5->40, -7.5->-8, 0.5->0). */
static int8_t requant_i8(int32_t acc, int mult, int shift){
    long long v = (long long)acc * (long long)mult;
    long long q = v >> shift;                 /* arithmetic floor */
    long long rem = v - (q << shift);         /* 0 .. 2^shift-1 */
    long long half = 1LL << (shift - 1);
    if (rem > half || (rem == half && (q & 1))) q++;   /* round half to even */
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

static int check(ork_npu *ctx, int M, int K, int N, int mult, int shift){
    int8_t *A = malloc((size_t)M*K), *B = malloc((size_t)K*N), *C = malloc((size_t)M*N);
    /* small values so acc stays in a range the requant maps into int8 meaningfully */
    for (size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)((rnd()&7)-3);
    for (size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)((rnd()&7)-3);
    double us=0;
    int r = ork_npu_probe_i8_out8(ctx, M, K, N, A, B, mult, shift, C, &us);
    if (r){ printf("  probe_i8_out8 M=%d K=%d N=%d rc=%d (%s)\n", M,K,N,r, r==-1?"WEDGED":"bad dims"); free(A);free(B);free(C); return 1; }
    int bad=0, first=1;
    for (int i=0;i<M;i++) for (int n=0;n<N;n++){
        int32_t acc=0; for (int k=0;k<K;k++) acc += (int)A[(size_t)i*K+k]*(int)B[(size_t)k*N+n];
        int8_t ref = requant_i8(acc, mult, shift);
        int8_t got = C[(size_t)i*N+n];
        if (got != ref){ bad++; if(first){ printf("    first mism @[%d,%d]: acc=%d ref=%d got=%d\n", i,n,acc,ref,got); first=0; } }
    }
    printf("  %s M=%d K=%d N=%d mult=0x%x shift=%d  mism=%d  (%.1f us)\n",
           bad?"WRONG":"ok  ", M,K,N, mult, shift, bad, us);
    free(A);free(B);free(C);
    return bad?1:0;
}

int main(void){
    ork_npu *ctx = ork_npu_init();
    if (!ctx){ printf("ork_npu_init failed (board only)\n"); return 0; }  /* skip gracefully off-board */
    int fail=0;
    printf("PPU int8-output requantize stage — bit-exact vs CPU model\n");
    /* K>=512: the synth_i8 0x1040 K-reduction schedule is validated there (the prefill domain). At tiny
     * K (64/128) the schedule formula's small-K branch is out of its validated range and miscomputes. */
    /* identity requant: mult=0x4000, shift=14 -> acc, then clamp to int8 */
    fail |= check(ctx, 1,  2048, 64,  0x4000, 14);
    fail |= check(ctx, 32, 512,  64,  0x4000, 14);
    fail |= check(ctx, 32, 2048, 2048,0x4000, 14);
    /* scaled requant (banker's rounding): 0.5x, 0.125x, and an arbitrary mult/shift */
    fail |= check(ctx, 32, 512,  64,  0x2000, 14);
    fail |= check(ctx, 32, 2048, 2048,0x1000, 15);
    fail |= check(ctx, 16, 4096, 128, 0x2ab0, 15);
    ork_npu_free(ctx);
    printf("%s\n", fail?"FAIL":"ALL OK");
    return fail;
}
