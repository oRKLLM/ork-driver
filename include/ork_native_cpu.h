/* ork_native_cpu.h — CPU-side NEON GEMV kernels for the ork-native tiered weight formats.
 *
 * Part of the Hybrid Precision Decode Architecture (see the ork-driver wiki). One storage format per
 * tensor, read by both engines: the NPU consumes int4/int8 (its native W4A4/W8A8 modes), the CPU
 * consumes int4/NF4/int5/int6 here (int8 is NPU-only — kept off the CPU's heaviest bandwidth format).
 *
 * Storage is per output channel n, LEAN 32-block layout (matching tools/int5_probe.c, validated
 * lossless + throughput on RK3588):
 *   nibble plane  [N][K/2]   : byte (16b+i) low=w[32b+i], high=w[32b+16+i]   (int4/NF4 index/code)
 *   bit4 plane    [N][K/8]   : (int5,int6) 5th bit, per 32-block: [0,1]=lo half, [2,3]=hi half
 *   bit5 plane    [N][K/8]   : (int6 only) 6th/sign bit, same block layout
 *   bscale[N]                : per-channel dequant scale (fp32)
 *   nf4_lut[16]              : (NF4 only) int8 codebook = round(level*127)
 * Activation: int8 [K] with a per-tensor ascale (absmax/127). Output C[n] fp32 = ascale*bscale[n]*dot.
 *
 * Free widths (byte-aligned, memory-bound at M=1): int4 (nibble), NF4 (nibble+vqtbl). ALU-priced
 * (sub-byte non-nibble, M=1 decode levers only): int5 (~1.37x int8 time), int6 (~1.88x). K%32==0.
 */
#ifndef ORK_NATIVE_CPU_H
#define ORK_NATIVE_CPU_H
#include <stdint.h>
#include <stddef.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

typedef enum { ORK_CPU_I4=0, ORK_CPU_NF4=1, ORK_CPU_I5=2, ORK_CPU_I6=3, ORK_CPU_I8=4 } ork_cpu_fmt;

static const uint8_t ORK_CPU_BITSEL[16]={1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};

/* int4 uniform: 32 wts/iter, low nibbles=w[0..15], high=w[16..31], sign-extend 4-bit. reads K/2 */
static inline int32_t ork_dot_i4(const int8_t*a,const uint8_t*b4,int K){
    int32x4_t ac=vdupq_n_s32(0); uint8x16_t m=vdupq_n_u8(0x0f);
    for(int k=0,kb=0;k+32<=K;k+=32,kb+=16){
        uint8x16_t pk=vld1q_u8(b4+kb);
        int8x16_t lo=vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(vandq_u8(pk,m)),4),4);
        int8x16_t hi=vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(vshrq_n_u8(pk,4)),4),4);
        ac=vdotq_s32(ac,lo,vld1q_s8(a+k)); ac=vdotq_s32(ac,hi,vld1q_s8(a+k+16));
    }
    return vaddvq_s32(ac);
}
/* NF4: nibble INDEX -> int8 code via vqtbl LUT (free). reads K/2 */
static inline int32_t ork_dot_nf4(const int8_t*a,const uint8_t*b4,int8x16_t lut,int K){
    int32x4_t ac=vdupq_n_s32(0); uint8x16_t m=vdupq_n_u8(0x0f);
    for(int k=0,kb=0;k+32<=K;k+=32,kb+=16){
        uint8x16_t pk=vld1q_u8(b4+kb);
        ac=vdotq_s32(ac,vqtbl1q_s8(lut,vandq_u8(pk,m)),vld1q_s8(a+k));
        ac=vdotq_s32(ac,vqtbl1q_s8(lut,vshrq_n_u8(pk,4)),vld1q_s8(a+k+16));
    }
    return vaddvq_s32(ac);
}
/* int5: nibble + bit4 plane, sign-extend 5-bit. reads 5K/8 (ALU-priced) */
static inline int32_t ork_dot_i5(const int8_t*a,const uint8_t*nb,const uint8_t*b4p,int K){
    int32x4_t ac=vdupq_n_s32(0); uint8x16_t m=vdupq_n_u8(0x0f);
    uint8x16_t bsel=vld1q_u8(ORK_CPU_BITSEL), c4=vdupq_n_u8(0x10);
    for(int k=0,kb=0,km=0;k+32<=K;k+=32,kb+=16,km+=4){
        uint8x16_t pk=vld1q_u8(nb+kb), lo=vandq_u8(pk,m), hi=vshrq_n_u8(pk,4);
        uint8x16_t l4=vandq_u8(vtstq_u8(vcombine_u8(vdup_n_u8(b4p[km]),  vdup_n_u8(b4p[km+1])),bsel),c4);
        uint8x16_t h4=vandq_u8(vtstq_u8(vcombine_u8(vdup_n_u8(b4p[km+2]),vdup_n_u8(b4p[km+3])),bsel),c4);
        int8x16_t lo5=vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(vorrq_u8(lo,l4)),3),3);
        int8x16_t hi5=vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(vorrq_u8(hi,h4)),3),3);
        ac=vdotq_s32(ac,lo5,vld1q_s8(a+k)); ac=vdotq_s32(ac,hi5,vld1q_s8(a+k+16));
    }
    return vaddvq_s32(ac);
}
/* int6: nibble + bit4 + bit5 planes, sign-extend 6-bit. reads 3K/4 (ALU-priced) */
static inline int32_t ork_dot_i6(const int8_t*a,const uint8_t*nb,const uint8_t*b4p,const uint8_t*b5p,int K){
    int32x4_t ac=vdupq_n_s32(0); uint8x16_t m=vdupq_n_u8(0x0f);
    uint8x16_t bsel=vld1q_u8(ORK_CPU_BITSEL), c4=vdupq_n_u8(0x10), c5=vdupq_n_u8(0x20);
    for(int k=0,kb=0,km=0;k+32<=K;k+=32,kb+=16,km+=4){
        uint8x16_t pk=vld1q_u8(nb+kb), lo=vandq_u8(pk,m), hi=vshrq_n_u8(pk,4);
        uint8x16_t l4=vandq_u8(vtstq_u8(vcombine_u8(vdup_n_u8(b4p[km]),  vdup_n_u8(b4p[km+1])),bsel),c4);
        uint8x16_t l5=vandq_u8(vtstq_u8(vcombine_u8(vdup_n_u8(b5p[km]),  vdup_n_u8(b5p[km+1])),bsel),c5);
        uint8x16_t h4=vandq_u8(vtstq_u8(vcombine_u8(vdup_n_u8(b4p[km+2]),vdup_n_u8(b4p[km+3])),bsel),c4);
        uint8x16_t h5=vandq_u8(vtstq_u8(vcombine_u8(vdup_n_u8(b5p[km+2]),vdup_n_u8(b5p[km+3])),bsel),c5);
        int8x16_t lo6=vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(vorrq_u8(vorrq_u8(lo,l4),l5)),2),2);
        int8x16_t hi6=vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(vorrq_u8(vorrq_u8(hi,h4),h5)),2),2);
        ac=vdotq_s32(ac,lo6,vld1q_s8(a+k)); ac=vdotq_s32(ac,hi6,vld1q_s8(a+k+16));
    }
    return vaddvq_s32(ac);
}
/* int8: reads K */
static inline int32_t ork_dot_i8(const int8_t*a,const int8_t*b,int K){
    int32x4_t ac=vdupq_n_s32(0); for(int k=0;k+16<=K;k+=16) ac=vdotq_s32(ac,vld1q_s8(b+k),vld1q_s8(a+k));
    return vaddvq_s32(ac);
}

/* One packed ork-native weight (per output channel n contiguous in each plane). */
typedef struct {
    ork_cpu_fmt fmt;
    const uint8_t *nibble;   /* [N][K/2]  (i4/nf4/i5/i6)   */
    const uint8_t *bit4;     /* [N][K/8]  (i5/i6)          */
    const uint8_t *bit5;     /* [N][K/8]  (i6)             */
    const int8_t  *i8;       /* [N][K]    (i8)             */
    const float   *bscale;   /* [N]                        */
    int8x16_t      nf4_lut;  /* (nf4)                      */
    int K, N;
} ork_cpu_w;

/* M=1 decode GEMV: out[n] = ascale * bscale[n] * dot(A_i8, W_n), for n in [n0,n1). */
static inline void ork_cpu_gemv_m1(const ork_cpu_w*w,const int8_t*A,float ascale,float*out,int n0,int n1){
    int K=w->K; size_t kh=(size_t)K/2, ke=(size_t)K/8;
    for(int n=n0;n<n1;n++){ int32_t d;
        switch(w->fmt){
        case ORK_CPU_I4:  d=ork_dot_i4 (A,w->nibble+(size_t)n*kh,K); break;
        case ORK_CPU_NF4: d=ork_dot_nf4(A,w->nibble+(size_t)n*kh,w->nf4_lut,K); break;
        case ORK_CPU_I5:  d=ork_dot_i5 (A,w->nibble+(size_t)n*kh,w->bit4+(size_t)n*ke,K); break;
        case ORK_CPU_I6:  d=ork_dot_i6 (A,w->nibble+(size_t)n*kh,w->bit4+(size_t)n*ke,w->bit5+(size_t)n*ke,K); break;
        default:          d=ork_dot_i8 (A,w->i8+(size_t)n*K,K); break;
        }
        out[n]=ascale*w->bscale[n]*(float)d;
    }
}
#endif /* NEON */
#endif /* ORK_NATIVE_CPU_H */
