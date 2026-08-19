/* test_nf4_decode.c — DRAM-realistic M=1 (decode) NF4/int4 GEMV probe + PRFM prefetch sweep.
 *
 * WHY NOT test_i4_gemm: that probe hammers ONE 0.5 MB weight, which stays L2/L3-resident — so it cannot
 * measure DRAM-latency hiding. Real MoE decode streams a DIFFERENT expert weight every call (8 experts x
 * 3 proj x 40 layers/token, ~0.5 MB each = far past the 3 MB L3), so every K-step is a cold DRAM read.
 * This probe reproduces that: NEXP distinct weights (default 64 = 32 MB) walked once per pass, M=1.
 *
 * Reports us/expert AND achieved GB/s (weight bytes / time) — the number that says whether the decode
 * inner loop is at the RK3588 memory-controller limit (~20-25 GB/s single-cluster) or leaving bandwidth
 * on the table (i.e. whether prefetch/latency-hiding has anything to win).
 *
 * Prefetch A/B (compile-time, no env):
 *   gcc -O3 -march=armv8.2-a+dotprod -Iinclude -DORK_PRF_DIST=0   -o /tmp/d0 examples/test_nf4_decode.c -lm
 *   gcc -O3 -march=armv8.2-a+dotprod -Iinclude -DORK_PRF_DIST=128 -o /tmp/d128 examples/test_nf4_decode.c -lm
 * Header-only (ork_native_cpu.h) — no libork/NPU needed. Runs in well under a second.
 *   ./test_nf4_decode [NEXP] [K] [N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "ork_native_cpu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static void fillf (float *p,size_t n,unsigned s){ for(size_t i=0;i<n;i++){ s=s*1103515245u+12345u; p[i]=((int)((s>>16)%1000)-500)/500.0f; } }
static void filli8(int8_t*p,size_t n,unsigned s){ for(size_t i=0;i<n;i++){ s=s*1103515245u+12345u; p[i]=(int8_t)((int)((s>>17)%255)-127); } }

int main(int argc,char**argv){
    const int NEXP = argc>1?atoi(argv[1]):64;
    const int K    = argc>2?atoi(argv[2]):2048;
    const int N    = argc>3?atoi(argv[3]):512;
    const size_t wbytes = (size_t)K*N/2;                 /* nibble plane per expert */
    printf("NF4/int4 M=1 DECODE probe: %d experts x K=%d N=%d = %.1f MB streamed/pass "
           "(L3 is ~3 MB, so this is DRAM-cold like real decode)\n", NEXP,K,N, (double)(NEXP*wbytes)/1048576.0);
    printf("prefetch: ORK_PRF_DIST=%d ORK_PRF_LOC=%d\n", (int)ORK_PRF_DIST, (int)ORK_PRF_LOC);

    float *W = malloc((size_t)N*K*sizeof(float)); fillf(W,(size_t)N*K,7);
    int8_t lut16[16]; float *bsc = malloc((size_t)N*sizeof(float));
    uint8_t **nib = malloc(NEXP*sizeof*nib);
    for(int e=0;e<NEXP;e++){ nib[e]=malloc(wbytes);
        ork_cpu_pack(ORK_CPU_NF4,K,N,W,nib[e],NULL,NULL,NULL,NULL,bsc,lut16);
        /* de-duplicate the pages so each expert is genuinely distinct memory (no COW/page sharing) */
        nib[e][e%wbytes] ^= 0x11; }
    int8_t *A = malloc((size_t)K); filli8(A,(size_t)K,23);
    float  *out= malloc((size_t)N*sizeof(float));

    for (int fmt=0; fmt<2; fmt++) {
        ork_cpu_w w; memset(&w,0,sizeof w);
        w.fmt = fmt? ORK_CPU_NF4 : ORK_CPU_I4; w.bscale=bsc; w.K=K; w.N=N;
        if (fmt) w.nf4_lut = vld1q_s8(lut16);
        double best=1e18;
        for(int r=0;r<5;r++){
            double t=now_us();
            for(int e=0;e<NEXP;e++){ w.nibble=nib[e]; ork_cpu_gemv_m1(&w,A,0.01f,out,0,N); }
            double u=now_us()-t; if(u<best)best=u;
        }
        const double gbs = (double)(NEXP*wbytes) / (best*1e-6) / 1e9;
        printf("  %-4s: %8.1f us/pass  %6.1f us/expert  %5.2f GB/s  (out[0]=%.3f)\n",
               fmt?"NF4":"I4", best, best/NEXP, gbs, out[0]);
    }
    printf("  -> compare GB/s across -DORK_PRF_DIST builds; if GB/s is already ~20+, the loop is at the\n");
    printf("     memory-controller limit and prefetch/latency-hiding has nothing left to win.\n");
    return 0;
}
