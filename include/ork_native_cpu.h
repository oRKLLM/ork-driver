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
#include <math.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

typedef enum { ORK_CPU_I4=0, ORK_CPU_NF4=1, ORK_CPU_I5=2, ORK_CPU_I6=3, ORK_CPU_I8=4 } ork_cpu_fmt;

/* The dot kernels use vdotq_s32 (dotprod ISA). This header is included by TUs (e.g. ggml-ork.cpp) whose
 * global -march may not enable dotprod, so tag each kernel with target("+dotprod") — vdotq's always_inline
 * then succeeds regardless of the TU's baseline. No-op when the TU already has dotprod. */
#ifndef ORK_NATIVE_TARGET
#define ORK_NATIVE_TARGET __attribute__((target("+dotprod")))
#endif

static const uint8_t ORK_CPU_BITSEL[16]={1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};

/* int4 uniform, ork-driver Bi4 CONSECUTIVE layout (byte j = w[2j]low|w[2j+1]high, matches
 * expand_chan_i4_i8): 32 wts/iter; vld2 deinterleaves the activation into even/odd. sign-extend 4-bit. */
static inline ORK_NATIVE_TARGET int32_t ork_dot_i4(const int8_t*a,const uint8_t*b4,int K){
    int32x4_t ac=vdupq_n_s32(0); uint8x16_t m=vdupq_n_u8(0x0f);
    for(int k=0,kb=0;k+32<=K;k+=32,kb+=16){
        uint8x16_t pk=vld1q_u8(b4+kb);                       /* w[k..k+31] as 16 consecutive pairs */
        int8x16_t lo=vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(vandq_u8(pk,m)),4),4);  /* w[k],w[k+2],..,w[k+30] */
        int8x16_t hi=vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(vshrq_n_u8(pk,4)),4),4);/* w[k+1],..,w[k+31] */
        int8x16x2_t a2=vld2q_s8(a+k);                        /* a2.val[0]=a[even], a2.val[1]=a[odd] */
        ac=vdotq_s32(ac,lo,a2.val[0]); ac=vdotq_s32(ac,hi,a2.val[1]);
    }
    return vaddvq_s32(ac);
}
/* NF4: Bi4 CONSECUTIVE nibble INDEX -> int8 code via vqtbl LUT (free). vld2 activation. */
static inline ORK_NATIVE_TARGET int32_t ork_dot_nf4(const int8_t*a,const uint8_t*b4,int8x16_t lut,int K){
    int32x4_t ac=vdupq_n_s32(0); uint8x16_t m=vdupq_n_u8(0x0f);
    for(int k=0,kb=0;k+32<=K;k+=32,kb+=16){
        uint8x16_t pk=vld1q_u8(b4+kb);
        int8x16x2_t a2=vld2q_s8(a+k);
        ac=vdotq_s32(ac,vqtbl1q_s8(lut,vandq_u8(pk,m)),a2.val[0]);
        ac=vdotq_s32(ac,vqtbl1q_s8(lut,vshrq_n_u8(pk,4)),a2.val[1]);
    }
    return vaddvq_s32(ac);
}
/* int5: nibble + bit4 plane, sign-extend 5-bit. reads 5K/8 (ALU-priced) */
static inline ORK_NATIVE_TARGET int32_t ork_dot_i5(const int8_t*a,const uint8_t*nb,const uint8_t*b4p,int K){
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
static inline ORK_NATIVE_TARGET int32_t ork_dot_i6(const int8_t*a,const uint8_t*nb,const uint8_t*b4p,const uint8_t*b5p,int K){
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
static inline ORK_NATIVE_TARGET int32_t ork_dot_i8(const int8_t*a,const int8_t*b,int K){
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
/* ---- PACK (offline, plain C): f32[N][K] -> the lean 32-block planes the dots read ----
 * Per output channel n: absmax -> per-channel scale; quantize; lay out in the 32-block nibble/bit-plane
 * form. NF4 stores the nearest-codebook INDEX in the nibble (lut = round(level*127) applied at inflate).
 * out sizes: nibble N*K/2, bit4 N*K/8, bit5 N*K/8, i8 N*K, bscale N. K%32==0. */
static const float ORK_NF4_LVL[16]={-1.0f,-0.6961928f,-0.5250731f,-0.3949175f,-0.2844414f,-0.1847734f,
    -0.0910500f,0.0f,0.0795803f,0.1609302f,0.2461123f,0.3379152f,0.4407098f,0.5626170f,0.7229568f,1.0f};
static inline int ork_cpu_pack(ork_cpu_fmt fmt,int K,int N,const float*W,
        uint8_t*nibble,uint8_t*bit4,uint8_t*bit5,int8_t*i8,float*bscale,int8_t nf4_lut_out[16]){
    if(K%32) return -1;
    if(fmt==ORK_CPU_NF4 && nf4_lut_out) for(int i=0;i<16;i++) nf4_lut_out[i]=(int8_t)lrintf(ORK_NF4_LVL[i]*127.0f);
    for(int n=0;n<N;n++){ const float*fr=W+(size_t)n*K; float mx=1e-9f;
        for(int k=0;k<K;k++){ float v=fr[k]<0?-fr[k]:fr[k]; if(v>mx)mx=v; }
        float lev = fmt==ORK_CPU_I4?7:fmt==ORK_CPU_I5?15:fmt==ORK_CPU_I6?31:fmt==ORK_CPU_I8?127:127;
        float sc = fmt==ORK_CPU_NF4 ? mx : mx/lev;   /* NF4: normalize by absmax for index select */
        float inv = sc>0?1.0f/sc:0.0f;
        if(bscale) bscale[n] = fmt==ORK_CPU_NF4 ? sc/127.0f : sc;  /* NF4 codes = level*127 -> dequant mx/127 */
        if(fmt==ORK_CPU_I8){ for(int k=0;k<K;k++){int q=(int)lrintf(fr[k]*inv); i8[(size_t)n*K+k]=(int8_t)(q>127?127:q<-127?-127:q);} continue; }
        uint8_t*nb=nibble+(size_t)n*(K/2);
        /* int4 / NF4: ork-driver Bi4 CONSECUTIVE layout — byte j = code(w[2j]) | code(w[2j+1])<<4. */
        if(fmt==ORK_CPU_I4 || fmt==ORK_CPU_NF4){
            for(int j=0;j<K/2;j++){ int c0,c1; int kl=2*j, kh=2*j+1;
                if(fmt==ORK_CPU_NF4){ float wl=fr[kl]*inv,wh=fr[kh]*inv; if(wl>1)wl=1;if(wl<-1)wl=-1; if(wh>1)wh=1;if(wh<-1)wh=-1;
                    int il=0,ih=0; float dl=1e9f,dh=1e9f; for(int q=0;q<16;q++){ float a=ORK_NF4_LVL[q]-wl;a=a<0?-a:a; if(a<dl){dl=a;il=q;} float cc=ORK_NF4_LVL[q]-wh;cc=cc<0?-cc:cc; if(cc<dh){dh=cc;ih=q;} }
                    c0=il; c1=ih;
                } else { c0=(int)lrintf(fr[kl]*inv); if(c0>7)c0=7;if(c0<-7)c0=-7; c1=(int)lrintf(fr[kh]*inv); if(c1>7)c1=7;if(c1<-7)c1=-7; }
                nb[j]=(uint8_t)((c0&0xf)|((c1&0xf)<<4));
            }
            continue;
        }
        /* int5 / int6: LEAN 32-block nibble + bit4(/bit5) planes (CPU-only formats). */
        uint8_t*b4=bit4?bit4+(size_t)n*(K/8):0; uint8_t*b5=bit5?bit5+(size_t)n*(K/8):0;
        if(b4) for(int i=0;i<K/8;i++) b4[i]=0; if(b5) for(int i=0;i<K/8;i++) b5[i]=0;
        int lim = fmt==ORK_CPU_I5?15:31;
        for(int b=0;b<K/32;b++){ int km=b*4;
            for(int i=0;i<16;i++){ int kl=32*b+i, kh=32*b+16+i;
                int cl=(int)lrintf(fr[kl]*inv); if(cl>lim)cl=lim;if(cl<-lim)cl=-lim;
                int ch=(int)lrintf(fr[kh]*inv); if(ch>lim)ch=lim;if(ch<-lim)ch=-lim;
                nb[16*b+i]=(uint8_t)((cl&0xf)|((ch&0xf)<<4));
                if(b4){ if((cl>>4)&1) b4[km+i/8]|=(uint8_t)(1<<(i&7)); if((ch>>4)&1) b4[km+2+i/8]|=(uint8_t)(1<<(i&7)); }
                if(b5){ if((cl>>5)&1) b5[km+i/8]|=(uint8_t)(1<<(i&7)); if((ch>>5)&1) b5[km+2+i/8]|=(uint8_t)(1<<(i&7)); }
            }
        }
    }
    return 0;
}
#endif /* NEON */
#endif /* ORK_NATIVE_CPU_H */
