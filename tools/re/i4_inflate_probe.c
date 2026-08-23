/* i4_inflate_probe — how fast is int4 -> int8 weight inflation, and can prefill hide it?
 *
 * i4a8 keeps int4 STORAGE (half the disk/IOVA of int8) but computes on the int8 MAC, so every weight must
 * be expanded nibble->byte before use — and under multi-domain streaming that expansion repeats on every
 * domain swap. The question is whether that cost is affordable given int8 compute is ~1.7x faster than
 * W4A4 (~180 vs ~107 t/s prefill measured earlier), and whether it can be hidden behind NPU compute during
 * prefill, which is compute-bound.
 *
 * Estimating it from DRAM bandwidth is not good enough: this session already saw a naive scalar loop run at
 * 3.4 GB/s where ~20 was assumed, a 6x error. So measure the actual loop, scalar and NEON, at sizes that
 * span cache-resident and DRAM-resident, and report GB/s of TRAFFIC (read + write) plus the wall time a
 * full 0.8B-class model's worth of inflation would cost. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + 1e-9*t.tv_nsec; }

/* scalar: one nibble pair -> two sign-extended int8 */
static void inflate_scalar(const uint8_t *src, int8_t *dst, size_t n_bytes){
    for (size_t i = 0; i < n_bytes; i++){
        const uint8_t b = src[i];
        int lo = b & 0xf, hi = (b >> 4) & 0xf;
        dst[2*i+0] = (int8_t)(lo >= 8 ? lo - 16 : lo);
        dst[2*i+1] = (int8_t)(hi >= 8 ? hi - 16 : hi);
    }
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
/* NEON: 16 packed bytes -> 32 int8. Sign-extend a nibble by <<4 then >>4 (arithmetic), which is branchless
 * and exactly the [-8,7] mapping the scalar path computes. */
static void inflate_neon(const uint8_t *src, int8_t *dst, size_t n_bytes){
    size_t i = 0;
    const uint8x16_t mask = vdupq_n_u8(0x0f);
    for (; i + 16 <= n_bytes; i += 16){
        const uint8x16_t v  = vld1q_u8(src + i);
        const int8x16_t  lo = vreinterpretq_s8_u8(vandq_u8(v, mask));
        const int8x16_t  hi = vreinterpretq_s8_u8(vshrq_n_u8(v, 4));
        const int8x16_t  ls = vshrq_n_s8(vshlq_n_s8(lo, 4), 4);   /* sign-extend low nibble  */
        const int8x16_t  hs = vshrq_n_s8(vshlq_n_s8(hi, 4), 4);   /* sign-extend high nibble */
        int8x16x2_t z; z.val[0] = ls; z.val[1] = hs;               /* interleave to lo,hi,lo,hi... */
        vst2q_s8(dst + 2*i, z);
    }
    if (i < n_bytes) inflate_scalar(src + i, dst + 2*i, n_bytes - i);
}
#endif

int main(int argc, char **argv){
    const double model_gib = argc > 1 ? atof(argv[1]) : 0.4;   /* int4 bytes of a 0.8B model ~= 0.4 GiB */
    printf("i4_inflate_probe: int4 -> int8 expansion throughput (traffic = read + write)\n");
    const size_t sizes[] = { 64*1024, 1024*1024, 8*1024*1024, 64*1024*1024 };
    for (size_t s = 0; s < sizeof sizes/sizeof *sizes; s++){
        const size_t nb = sizes[s];
        uint8_t *src = malloc(nb); int8_t *dst = malloc(2*nb);
        if (!src || !dst) { printf("  OOM at %zu\n", nb); return 1; }
        for (size_t i = 0; i < nb; i++) src[i] = (uint8_t)(i * 37u);
        const int it = nb < (4u<<20) ? 200 : 10;
        inflate_scalar(src, dst, nb);
        double t0 = now(); for (int r = 0; r < it; r++) inflate_scalar(src, dst, nb); double ts = (now()-t0)/it;
        double gbs_s = (double)(3*nb) / ts / 1e9;
        double tn = -1, gbs_n = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        inflate_neon(src, dst, nb);
        t0 = now(); for (int r = 0; r < it; r++) inflate_neon(src, dst, nb); tn = (now()-t0)/it;
        gbs_n = (double)(3*nb) / tn / 1e9;
        /* exactness: NEON must equal scalar, or the speed is meaningless */
        int8_t *ref = malloc(2*nb); inflate_scalar(src, ref, nb);
        int bad = memcmp(ref, dst, 2*nb) != 0; free(ref);
        printf("  %6zu KiB int4  scalar %7.2f GB/s   NEON %7.2f GB/s (%4.1fx) %s\n",
               nb>>10, gbs_s, gbs_n, ts/tn, bad ? "MISMATCH!" : "exact");
#else
        printf("  %6zu KiB int4  scalar %7.2f GB/s\n", nb>>10, gbs_s);
#endif
        if (s == sizeof sizes/sizeof *sizes - 1){
            /* input-bytes/s = nb/t (the probe's GB/s figure counts read+write traffic = 3*nb). */
            const double bytes = model_gib * 1073741824.0;
            const double ms_s = bytes / (nb / ts) * 1e3;
            const double ms_n = tn > 0 ? bytes / (nb / tn) * 1e3 : 0.0;
            printf("\n  a %.2f GiB int4 model inflates in %.0f ms (scalar) / %.0f ms (NEON)\n", model_gib, ms_s, ms_n);
            /* Against a 512-token prefill: int8 ~180 t/s = 2.84 s, W4A4 ~107 t/s = 4.79 s (measured earlier,
             * different model — indicative, not exact). */
            printf("  vs a 512-tok prefill: int8 2.84 s + inflate %.0f ms = %.2f s  |  W4A4 4.79 s  -> %.2fx\n",
                   ms_n, 2.84 + ms_n/1e3, 4.79 / (2.84 + ms_n/1e3));
        }
        free(src); free(dst);
    }
    return 0;
}
