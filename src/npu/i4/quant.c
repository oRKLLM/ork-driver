/* npu/i4/quant.c — int4 / NF4 quantisation, imatrix scale selection.
 *
 * Part of the i4 datapath; shared declarations in npu/i4/i4.h. Split out of npu/i4.c for the
 * same reason i8 is a folder: one datapath, sized for reading. */
#define _GNU_SOURCE   /* CPU_SET/pthread_setaffinity_np, as npu.c does */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include <math.h>
#include "ork_regs.h"
#include "regcmd_i8.h"
#include "orkd_proto.h"
#include "npu/internal.h"
#include "npu/core.h"
#include "regcmd_i4.h"
#include "npu/i4/i4.h"
#define ORK_IM_CLIP_N ((int)(sizeof(ORK_IM_CLIP_GRID)/sizeof(ORK_IM_CLIP_GRID[0])))
static const float ORK_IM_CLIP_GRID[] = { 1.0f, 0.92f, 0.85f, 0.78f, 0.70f, 0.62f, 0.55f };

void orki_quant_chan_i4(const float *fr, int K, float scale, int sr, uint32_t *seed, uint8_t *nib, float *qf32) {
    float inv = scale > 0 ? 1.0f/scale : 0.0f;
    for (int k = 0; k < K; k++) {
        int q;
        if (sr) { float u = (float)(ork_xs32(seed) >> 8) * (1.0f/16777216.0f);  /* u in [0,1) */
                  q = (int)floorf(fr[k]*inv + u); }
        else      q = (int)lrintf(fr[k]*inv);
        if (q > 7) q = 7; else if (q < -7) q = -7;
        qf32[k] = (float)q;
        uint8_t nb = (uint8_t)(q & 0xf);              /* low nibble holds the signed 4-bit code */
        if (k & 1) nib[k>>1] |= (uint8_t)(nb << 4); else nib[k>>1] = nb;
    }
}

void orki_expand_chan_i4_f32(const uint8_t *nib, int K, float *qf32) {
    int k = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    uint8x8_t vlo = vdup_n_u8(0x0f);
    for (; k <= K - 16; k += 16) {
        uint8x8_t pk = vld1_u8(nib + (k>>1));                 /* 8 bytes = 16 nibbles */
        int8x8_t even = vreinterpret_s8_u8(vand_u8(pk, vlo)); /* low nibbles  (codes k,k+2,...) */
        int8x8_t odd  = vreinterpret_s8_u8(vshr_n_u8(pk, 4)); /* high nibbles (codes k+1,...) */
        /* sign-extend 4-bit: shift the nibble into the top of an int8, then arithmetic-shift back */
        even = vshr_n_s8(vshl_n_s8(even, 4), 4);
        odd  = vshr_n_s8(vshl_n_s8(odd,  4), 4);
        int8x8x2_t zip = vzip_s8(even, odd);                  /* interleave -> code order */
        int8x16_t codes = vcombine_s8(zip.val[0], zip.val[1]);
        int16x8_t lo16 = vmovl_s8(vget_low_s8(codes));
        int16x8_t hi16 = vmovl_s8(vget_high_s8(codes));
        vst1q_f32(qf32 + k,      vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))));
        vst1q_f32(qf32 + k + 4,  vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))));
        vst1q_f32(qf32 + k + 8,  vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))));
        vst1q_f32(qf32 + k + 12, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))));
    }
#endif
    for (; k < K; k++) {
        uint8_t nb = (k & 1) ? (nib[k>>1] >> 4) : (nib[k>>1] & 0xf);
        int8_t c = (int8_t)(nb << 4) >> 4;                    /* sign-extend 4-bit */
        qf32[k] = (float)c;
    }
}

void orki_quant_chan_nf4(const float *fr, int K, float absmax, int sr, uint32_t *seed, uint8_t *nib, uint8_t *qidx) {
    float inv = absmax > 0 ? 1.0f/absmax : 0.0f;
    for (int k = 0; k < K; k++) {
        float wn = fr[k]*inv; if (wn > 1.0f) wn = 1.0f; else if (wn < -1.0f) wn = -1.0f;
        /* find bracketing pair: hi = first level >= wn */
        int hi = 0; while (hi < 15 && ORKI_NF4_LEVELS[hi] < wn) hi++;
        int lo = hi > 0 ? hi-1 : 0;
        int idx;
        if (lo == hi) idx = hi;
        else {
            float dlo = ORKI_NF4_LEVELS[hi]-ORKI_NF4_LEVELS[lo];     /* span > 0 */
            float t = (wn - ORKI_NF4_LEVELS[lo]) / dlo;             /* fractional pos in [0,1] toward hi */
            if (sr) { float u = (float)(ork_xs32(seed) >> 8) * (1.0f/16777216.0f); /* u in [0,1) */
                      idx = (t > u) ? hi : lo; }                   /* P(hi) = t */
            else      idx = (t >= 0.5f) ? hi : lo;                 /* nearest level */
        }
        qidx[k] = (uint8_t)idx;
        uint8_t nb = (uint8_t)(idx & 0xf);
        if (k & 1) nib[k>>1] |= (uint8_t)(nb << 4); else nib[k>>1] = nb;
    }
}

void orki_inflate_chan_nf4_f32(const uint8_t *qidx, int K, const int8_t lut[16], float *qf32) {
    int k = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int8x16_t vlut = vld1q_s8(lut);
    for (; k <= K - 16; k += 16) {
        uint8x16_t vi = vld1q_u8(qidx + k);
        int8x16_t codes = vqtbl1q_s8(vlut, vi);               /* table lookup: code = lut[idx] */
        int16x8_t lo16 = vmovl_s8(vget_low_s8(codes));
        int16x8_t hi16 = vmovl_s8(vget_high_s8(codes));
        vst1q_f32(qf32 + k,      vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))));
        vst1q_f32(qf32 + k + 4,  vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))));
        vst1q_f32(qf32 + k + 8,  vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))));
        vst1q_f32(qf32 + k + 12, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))));
    }
#endif
    for (; k < K; k++) qf32[k] = (float)lut[qidx[k]];
}

void orki_expand_chan_i4_i8(const uint8_t *nib, int K, int8_t *i8) {
    int k = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    uint8x8_t vlo = vdup_n_u8(0x0f);
    for (; k <= K - 16; k += 16) {
        uint8x8_t pk = vld1_u8(nib + (k>>1));                 /* 8 bytes = 16 nibbles */
        int8x8_t even = vreinterpret_s8_u8(vand_u8(pk, vlo)); /* low nibbles  (codes k,k+2,...) */
        int8x8_t odd  = vreinterpret_s8_u8(vshr_n_u8(pk, 4)); /* high nibbles (codes k+1,...) */
        even = vshr_n_s8(vshl_n_s8(even, 4), 4);              /* sign-extend 4-bit */
        odd  = vshr_n_s8(vshl_n_s8(odd,  4), 4);
        int8x8x2_t zip = vzip_s8(even, odd);                  /* interleave -> code order */
        vst1q_s8(i8 + k, vcombine_s8(zip.val[0], zip.val[1]));
    }
#endif
    for (; k < K; k++) {
        uint8_t nb = (k & 1) ? (nib[k>>1] >> 4) : (nib[k>>1] & 0xf);
        i8[k] = (int8_t)(nb << 4) >> 4;                       /* sign-extend 4-bit */
    }
}

static float wq_err_chan(const float *fr, int K, float absmax, int nf4, const float *im, float *dq) {
    if (nf4) {
        float sc = absmax / 127.0f, inv = absmax > 0 ? 1.0f/absmax : 0.0f;
        for (int k = 0; k < K; k++) {
            float wn = fr[k]*inv; if (wn > 1.0f) wn = 1.0f; else if (wn < -1.0f) wn = -1.0f;
            int hi = 0; while (hi < 15 && ORKI_NF4_LEVELS[hi] < wn) hi++;
            int lo = hi > 0 ? hi-1 : 0, idx;
            if (lo == hi) idx = hi;
            else { float t = (wn-ORKI_NF4_LEVELS[lo])/(ORKI_NF4_LEVELS[hi]-ORKI_NF4_LEVELS[lo]); idx = (t >= 0.5f) ? hi : lo; }
            dq[k] = (float)((int8_t)lrintf(ORKI_NF4_LEVELS[idx]*127.0f)) * sc;  /* match the int8 LUT path */
        }
    } else {
        float scale = absmax / 7.0f, inv = scale > 0 ? 1.0f/scale : 0.0f;
        for (int k = 0; k < K; k++) {
            int q = (int)lrintf(fr[k]*inv); if (q > 7) q = 7; else if (q < -7) q = -7;
            dq[k] = (float)q * scale;
        }
    }
    float e = 0.0f;
    for (int k = 0; k < K; k++) { float d = fr[k]-dq[k]; e += (im ? im[k] : 1.0f) * d*d; }
    return e;
}

float orki_wq_best_absmax(const float *fr, int K, float rawabsmax, int nf4, const float *im, float *dq) {
    float best_abs = rawabsmax, best_e = wq_err_chan(fr, K, rawabsmax, nf4, im, dq);
    for (int g = 1; g < ORK_IM_CLIP_N; g++) {
        float cand = rawabsmax * ORK_IM_CLIP_GRID[g];
        float e = wq_err_chan(fr, K, cand, nf4, im, dq);
        if (e < best_e) { best_e = e; best_abs = cand; }
    }
    return best_abs;
}
