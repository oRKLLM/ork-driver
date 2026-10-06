/* f16_transpose_probe — how fast can the HOST transpose, and is NEON worth it?
 *
 * WHY THIS IS A CPU PROBE AT ALL. The transpose is the measured bottleneck of a training step
 * (49.5 ms for one [4096,1024], 34% of a dW, more than a forward matmul), so the first question is
 * whether the NPU can do it. It cannot, today, and that is probe-anchored rather than an opinion:
 *   - an on-device contiguous->cube relayout IS the vendor RESHAPE op (OPS_REGISTRY line 104);
 *     that campaign is BLOCKED on undecoded read-geometry (0x107c/0x1080) and sits at RE-stage
 *     with reshape_probe_f16. The fp16 matmul HANGS writing the cube; int8 writes LINEAR only.
 *   - ork_bmm_fp16_strided's "strided operand" is NOT a device transpose: bmm.h says each batch is
 *     "gathered into contiguous scratch, then the dense pack/run path runs unchanged", and
 *     "strided-k is supported (gathered) but slower". It would relocate this loop, not remove it.
 *   - a matmul cannot produce it: C[M,N] = A[M,K] B[K,N] always has A's rows and B's columns, so
 *     X^T can only come from the weight PACK (a host tile-transpose) or from RESHAPE.
 * So the host does it, and the only question left is how well.
 *
 * WHAT THE BASELINE ACTUALLY MEASURES. The 49.5 ms is a naive doubly-nested loop: one side of it is
 * always striding. At [4096,1024] that is 8 MB of output at ~170 MB/s on a core that streams ~8 GB/s,
 * so it is a cache-miss benchmark, not a memory-bandwidth one. Blocking should recover most of it
 * before any SIMD is involved, which is worth knowing separately -- if blocking alone is enough, the
 * NEON path is complexity with no return.
 *
 * ARMS
 *   naive     the current baseline, for reference
 *   block B   cache-tiled, scalar, B in {16,32,64}
 *   neon      4x4 register transpose (vtrnq/vcombine) + vcvt_f16_f32, inside a 64x64 cache tile
 *
 * Both conversions are covered: fp32->fp16 (what a training step needs -- activations are fp32) and
 * fp16->fp16 (the generic primitive). Every arm is checked elementwise against the naive result
 * before it is timed; a fast wrong transpose is the easy failure here.
 *
 * CPU only -- touches no NPU, so no npu_guard needed. Run it on the BOARD: this is an A76 question
 * and a workstation number would say nothing.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "ork_npu.h"
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static double now_us(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6 + t.tv_nsec/1e3;
}

/* ---- fp32 [R,C] -> fp16 [C,R] ---------------------------------------------------------------- */
static void t32_naive(ork_f16 *d,const float *s,int R,int C){
    for(int r=0;r<R;r++) for(int c=0;c<C;c++) d[(size_t)c*R+r]=(ork_f16)s[(size_t)r*C+c];
}
static void t32_block(ork_f16 *d,const float *s,int R,int C,int B){
    for(int r0=0;r0<R;r0+=B) for(int c0=0;c0<C;c0+=B){
        int rm=r0+B<R?r0+B:R, cm=c0+B<C?c0+B:C;
        for(int r=r0;r<rm;r++) for(int c=c0;c<cm;c++)
            d[(size_t)c*R+r]=(ork_f16)s[(size_t)r*C+c];
    }
}
#ifdef __ARM_NEON
/* 4x4 float block: four contiguous loads, a register transpose, four fp16 stores down the columns. */
static inline void blk4x4_32(ork_f16 *d,const float *s,int R,int C,int r,int c){
    float32x4_t a0=vld1q_f32(s+(size_t)(r+0)*C+c), a1=vld1q_f32(s+(size_t)(r+1)*C+c);
    float32x4_t a2=vld1q_f32(s+(size_t)(r+2)*C+c), a3=vld1q_f32(s+(size_t)(r+3)*C+c);
    float32x4x2_t t0=vtrnq_f32(a0,a1), t1=vtrnq_f32(a2,a3);
    float32x4_t b0=vcombine_f32(vget_low_f32 (t0.val[0]),vget_low_f32 (t1.val[0]));
    float32x4_t b1=vcombine_f32(vget_low_f32 (t0.val[1]),vget_low_f32 (t1.val[1]));
    float32x4_t b2=vcombine_f32(vget_high_f32(t0.val[0]),vget_high_f32(t1.val[0]));
    float32x4_t b3=vcombine_f32(vget_high_f32(t0.val[1]),vget_high_f32(t1.val[1]));
    vst1_f16((__fp16*)d+(size_t)(c+0)*R+r, vcvt_f16_f32(b0));
    vst1_f16((__fp16*)d+(size_t)(c+1)*R+r, vcvt_f16_f32(b1));
    vst1_f16((__fp16*)d+(size_t)(c+2)*R+r, vcvt_f16_f32(b2));
    vst1_f16((__fp16*)d+(size_t)(c+3)*R+r, vcvt_f16_f32(b3));
}
static void t32_neon(ork_f16 *d,const float *s,int R,int C,int B){
    int R4=R&~3, C4=C&~3;
    for(int r0=0;r0<R4;r0+=B) for(int c0=0;c0<C4;c0+=B){
        int rm=r0+B<R4?r0+B:R4, cm=c0+B<C4?c0+B:C4;
        for(int r=r0;r<rm;r+=4) for(int c=c0;c<cm;c+=4) blk4x4_32(d,s,R,C,r,c);
    }
    for(int r=R4;r<R;r++) for(int c=0;c<C;c++) d[(size_t)c*R+r]=(ork_f16)s[(size_t)r*C+c];
    for(int r=0;r<R4;r++) for(int c=C4;c<C;c++) d[(size_t)c*R+r]=(ork_f16)s[(size_t)r*C+c];
}
#endif

/* ---- fp16 [R,C] -> fp16 [C,R] ---------------------------------------------------------------- */
static void t16_naive(ork_f16 *d,const ork_f16 *s,int R,int C){
    for(int r=0;r<R;r++) for(int c=0;c<C;c++) d[(size_t)c*R+r]=s[(size_t)r*C+c];
}
static void t16_block(ork_f16 *d,const ork_f16 *s,int R,int C,int B){
    for(int r0=0;r0<R;r0+=B) for(int c0=0;c0<C;c0+=B){
        int rm=r0+B<R?r0+B:R, cm=c0+B<C?c0+B:C;
        for(int r=r0;r<rm;r++) for(int c=c0;c<cm;c++) d[(size_t)c*R+r]=s[(size_t)r*C+c];
    }
}
#ifdef __ARM_NEON
/* 8x8 fp16 block via three zip stages — the fp16 analogue of the 4x4 float transpose. */
static inline void blk8x8_16(ork_f16 *d,const ork_f16 *s,int R,int C,int r,int c){
    /* TRN, not ZIP: a transpose exchanges 2x2 blocks, widening 16 -> 32 -> 64 bits per stage. */
    uint16x8_t a[8];
    for(int i=0;i<8;i++) a[i]=vreinterpretq_u16_f16(vld1q_f16((const __fp16*)s+(size_t)(r+i)*C+c));
    uint16x8x2_t p0=vtrnq_u16(a[0],a[1]), p1=vtrnq_u16(a[2],a[3]),
                 p2=vtrnq_u16(a[4],a[5]), p3=vtrnq_u16(a[6],a[7]);
    uint32x4x2_t q0=vtrnq_u32(vreinterpretq_u32_u16(p0.val[0]),vreinterpretq_u32_u16(p1.val[0])),
                 q1=vtrnq_u32(vreinterpretq_u32_u16(p0.val[1]),vreinterpretq_u32_u16(p1.val[1])),
                 q2=vtrnq_u32(vreinterpretq_u32_u16(p2.val[0]),vreinterpretq_u32_u16(p3.val[0])),
                 q3=vtrnq_u32(vreinterpretq_u32_u16(p2.val[1]),vreinterpretq_u32_u16(p3.val[1]));
    #define J(x,y,half) vreinterpretq_f16_u64(vcombine_u64( \
            vget_##half##_u64(vreinterpretq_u64_u32(x)), vget_##half##_u64(vreinterpretq_u64_u32(y))))
    float16x8_t o[8];
    o[0]=J(q0.val[0],q2.val[0],low);  o[4]=J(q0.val[0],q2.val[0],high);
    o[1]=J(q1.val[0],q3.val[0],low);  o[5]=J(q1.val[0],q3.val[0],high);
    o[2]=J(q0.val[1],q2.val[1],low);  o[6]=J(q0.val[1],q2.val[1],high);
    o[3]=J(q1.val[1],q3.val[1],low);  o[7]=J(q1.val[1],q3.val[1],high);
    #undef J
    for(int i=0;i<8;i++) vst1q_f16((__fp16*)d+(size_t)(c+i)*R+r, o[i]);
}
static void t16_neon(ork_f16 *d,const ork_f16 *s,int R,int C,int B){
    int R8=R&~7, C8=C&~7;
    for(int r0=0;r0<R8;r0+=B) for(int c0=0;c0<C8;c0+=B){
        int rm=r0+B<R8?r0+B:R8, cm=c0+B<C8?c0+B:C8;
        for(int r=r0;r<rm;r+=8) for(int c=c0;c<cm;c+=8) blk8x8_16(d,s,R,C,r,c);
    }
    for(int r=R8;r<R;r++) for(int c=0;c<C;c++) d[(size_t)c*R+r]=s[(size_t)r*C+c];
    for(int r=0;r<R8;r++) for(int c=C8;c<C;c++) d[(size_t)c*R+r]=s[(size_t)r*C+c];
}
#endif

static int bad32(const ork_f16*got,const ork_f16*ref,size_t n){
    for(size_t i=0;i<n;i++) if(!((float)got[i]==(float)ref[i])) return 1;
    return 0;
}

int main(int argc,char**argv){
    int shapes[][2] = {{4096,1024},{1024,3584},{3584,1024},{4096,3584}};
    int ns = (int)(sizeof shapes/sizeof shapes[0]);
    int iters = argc>1?atoi(argv[1]):3;

    printf("host transpose, fp32[R,C] -> fp16[C,R]   best of %d   (A76; CPU only, no NPU)\n\n",iters);
    printf("%-12s %9s %9s %9s %9s %9s   %s\n",
           "shape","naive","blk16","blk32","blk64","neon64","speedup vs naive");
    for(int s=0;s<ns;s++){
        int R=shapes[s][0], C=shapes[s][1];
        size_t n=(size_t)R*C;
        float   *src=malloc(n*sizeof *src);
        ork_f16 *ref=malloc(n*sizeof *ref), *got=malloc(n*sizeof *got);
        if(!src||!ref||!got){ printf("alloc failed\n"); return 1; }
        unsigned st=12345;
        for(size_t i=0;i<n;i++){ st=st*1103515245u+12345u;
            src[i]=(float)((int)((st>>13)&0xffff)-32768)/32768.0f; }

        double t[5]={1e30,1e30,1e30,1e30,1e30};
        for(int it=0;it<iters;it++){
            double a=now_us(); t32_naive(ref,src,R,C);        a=now_us()-a; if(a<t[0])t[0]=a;
            int B[3]={16,32,64};
            for(int b=0;b<3;b++){
                memset(got,0,n*sizeof *got);
                double x=now_us(); t32_block(got,src,R,C,B[b]); x=now_us()-x;
                if(bad32(got,ref,n)){ printf("  blk%d WRONG at %dx%d\n",B[b],R,C); return 1; }
                if(x<t[1+b])t[1+b]=x;
            }
#ifdef __ARM_NEON
            memset(got,0,n*sizeof *got);
            double y=now_us(); t32_neon(got,src,R,C,64); y=now_us()-y;
            if(bad32(got,ref,n)){ printf("  neon WRONG at %dx%d\n",R,C); return 1; }
            if(y<t[4])t[4]=y;
#endif
        }
        double best=t[1]; for(int i=2;i<5;i++) if(t[i]<best) best=t[i];
        printf("[%4d,%4d] %8.0fus %8.0fus %8.0fus %8.0fus %8.0fus   %.1fx  (%.1f GB/s out)\n",
               R,C,t[0],t[1],t[2],t[3],t[4], t[0]/best,
               (double)n*sizeof(ork_f16)/(best*1e3));
        free(src);free(ref);free(got);
    }

    printf("\nfp16[R,C] -> fp16[C,R]   (the generic primitive)\n");
    printf("%-12s %9s %9s %9s   %s\n","shape","naive","blk64","neon64","speedup");
    for(int s=0;s<ns;s++){
        int R=shapes[s][0], C=shapes[s][1];
        size_t n=(size_t)R*C;
        ork_f16 *src=malloc(n*sizeof *src), *ref=malloc(n*sizeof *ref), *got=malloc(n*sizeof *got);
        if(!src||!ref||!got){ printf("alloc failed\n"); return 1; }
        unsigned st=999;
        for(size_t i=0;i<n;i++){ st=st*1103515245u+12345u;
            src[i]=(ork_f16)((float)((int)((st>>13)&0xffff)-32768)/32768.0f); }
        double t[3]={1e30,1e30,1e30};
        for(int it=0;it<iters;it++){
            double a=now_us(); t16_naive(ref,src,R,C); a=now_us()-a; if(a<t[0])t[0]=a;
            memset(got,0,n*sizeof *got);
            double x=now_us(); t16_block(got,src,R,C,64); x=now_us()-x;
            if(bad32(got,ref,n)){ printf("  blk64 WRONG at %dx%d\n",R,C); return 1; }
            if(x<t[1])t[1]=x;
#ifdef __ARM_NEON
            memset(got,0,n*sizeof *got);
            double y=now_us(); t16_neon(got,src,R,C,64); y=now_us()-y;
            if(bad32(got,ref,n)){ printf("  neon WRONG at %dx%d\n",R,C); return 1; }
            if(y<t[2])t[2]=y;
#endif
        }
        double best=t[1]<t[2]?t[1]:t[2];
        printf("[%4d,%4d] %8.0fus %8.0fus %8.0fus   %.1fx  (%.1f GB/s out)\n",
               R,C,t[0],t[1],t[2],t[0]/best,(double)n*sizeof(ork_f16)/(best*1e3));
        free(src);free(ref);free(got);
    }
#ifndef __ARM_NEON
    printf("\n(built without NEON — the neon columns are absent)\n");
#endif
    return 0;
}
