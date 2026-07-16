/* int5_probe — validate the int5 storage primitive (nibble plane + MSB bit-plane) + free NEON inflate.
 * int5 code q in [-15,15] stored as: nibble[K/2] (low 4 bits, 2/byte) + msb[K/8] (bit 4, 8/byte) = 5K/8 B.
 * Inflate = NEON: nibble->int8 (zip), merge the 5th bit at position 4 (vtst spread + vorr), sign-extend
 * (shl#3->shr#3). Verifies the round-trip is LOSSLESS (inflated int8 == packed int5 code) and measures the
 * GEMV throughput vs int4/int8 to confirm the inflate hides behind DRAM latency (memory-bound-free).
 *   make int5_probe && ./int5_probe [iters] [threads]        (CPU-only, no NPU — board-safe)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

static int gK,gN,gNT;
static int8_t  *gA;                 /* int8 activation [K] */
static int8_t  *gB8;                /* int8 weights [N][K] */
static uint8_t *gB4;                /* int4 nibbles [N][K/2] */
static uint8_t *gN5, *gM5;          /* int5: nibble [N][K/2] + msb [N][K/8] */
static int32_t *gC;
static volatile int32_t sink;

/* int8 dot (reads K bytes) */
static int32_t dot_i8(const int8_t*a,const int8_t*b,int K){
    int32x4_t ac=vdupq_n_s32(0); int k=0;
    for(;k+16<=K;k+=16) ac=vdotq_s32(ac,vld1q_s8(b+k),vld1q_s8(a+k));
    return vaddvq_s32(ac);
}
/* LEAN int4 dot: 32 weights/iter, low nibbles=w[0..15], high=w[16..31] (no zip). reads K/2 bytes */
static int32_t dot_i4(const int8_t*a,const uint8_t*b4,int K){
    int32x4_t ac=vdupq_n_s32(0); uint8x16_t m=vdupq_n_u8(0x0f); int k=0,kb=0;
    for(;k+32<=K;k+=32,kb+=16){
        uint8x16_t pk=vld1q_u8(b4+kb);
        int8x16_t lo=vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(vandq_u8(pk,m)),4),4);   /* w0..w15  */
        int8x16_t hi=vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(vshrq_n_u8(pk,4)),4),4); /* w16..w31 */
        ac=vdotq_s32(ac,lo,vld1q_s8(a+k));
        ac=vdotq_s32(ac,hi,vld1q_s8(a+k+16));
    }
    return vaddvq_s32(ac);
}
static const uint8_t BITSEL[16]={1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};
/* LEAN int5 dot: nibble (K/2) + even/odd MSB sub-planes (2B lo + 2B hi per 32-block, K/8 total). reads 5K/8 */
static int32_t dot_i5(const int8_t*a,const uint8_t*nb,const uint8_t*mb,int K){
    int32x4_t ac=vdupq_n_s32(0); uint8x16_t m=vdupq_n_u8(0x0f);
    uint8x16_t bsel=vld1q_u8(BITSEL), b4v=vdupq_n_u8(0x10);
    int k=0,kb=0,km=0;
    for(;k+32<=K;k+=32,kb+=16,km+=4){
        uint8x16_t pk=vld1q_u8(nb+kb);
        uint8x16_t lo=vandq_u8(pk,m), hi=vshrq_n_u8(pk,4);
        uint8x16_t mlo=vandq_u8(vtstq_u8(vcombine_u8(vdup_n_u8(mb[km]),  vdup_n_u8(mb[km+1])),bsel),b4v);
        uint8x16_t mhi=vandq_u8(vtstq_u8(vcombine_u8(vdup_n_u8(mb[km+2]),vdup_n_u8(mb[km+3])),bsel),b4v);
        int8x16_t lo5=vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(vorrq_u8(lo,mlo)),3),3); /* w0..w15  5-bit */
        int8x16_t hi5=vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(vorrq_u8(hi,mhi)),3),3); /* w16..w31 5-bit */
        ac=vdotq_s32(ac,lo5,vld1q_s8(a+k));
        ac=vdotq_s32(ac,hi5,vld1q_s8(a+k+16));
    }
    return vaddvq_s32(ac);
}
/* NF4 dot: nibble INDEX (0..15) -> int8 code via vqtbl LUT (single-instr lookup). reads K/2 bytes */
static int8x16_t gLUT;
static int32_t dot_nf4(const int8_t*a,const uint8_t*b4,int K){
    int32x4_t ac=vdupq_n_s32(0); uint8x16_t m=vdupq_n_u8(0x0f); int k=0,kb=0;
    for(;k+32<=K;k+=32,kb+=16){
        uint8x16_t pk=vld1q_u8(b4+kb);
        int8x16_t lo=vqtbl1q_s8(gLUT,vandq_u8(pk,m));    /* LUT[low idx]  -> w0..w15  */
        int8x16_t hi=vqtbl1q_s8(gLUT,vshrq_n_u8(pk,4));  /* LUT[high idx] -> w16..w31 */
        ac=vdotq_s32(ac,lo,vld1q_s8(a+k));
        ac=vdotq_s32(ac,hi,vld1q_s8(a+k+16));
    }
    return vaddvq_s32(ac);
}
typedef struct{int lo,hi,mode;}job;
static void* wk(void*p){ job*j=p; cpu_set_t s;CPU_ZERO(&s);CPU_SET(4+((j->lo/((gN+gNT-1)/gNT))%4),&s);pthread_setaffinity_np(pthread_self(),sizeof s,&s);
    if(j->mode==8) for(int n=j->lo;n<j->hi;n++) gC[n]=dot_i8(gA,gB8+(size_t)n*gK,gK);
    else if(j->mode==4) for(int n=j->lo;n<j->hi;n++) gC[n]=dot_i4(gA,gB4+(size_t)n*(gK/2),gK);
    else if(j->mode==6) for(int n=j->lo;n<j->hi;n++) gC[n]=dot_nf4(gA,gB4+(size_t)n*(gK/2),gK);
    else for(int n=j->lo;n<j->hi;n++) gC[n]=dot_i5(gA,gN5+(size_t)n*(gK/2),gM5+(size_t)n*(gK/8),gK);
    return NULL; }
static double run(int mode){ pthread_t th[8]; job jb[8]; int per=(gN+gNT-1)/gNT;
    for(int t=0;t<gNT;t++){jb[t]=(job){t*per,(t+1)*per<gN?(t+1)*per:gN,mode}; pthread_create(&th[t],0,wk,&jb[t]);}
    for(int t=0;t<gNT;t++) pthread_join(th[t],0); sink+=gC[0]; return 0; }

int main(int argc,char**argv){
    int iters=argc>1?atoi(argv[1]):20; gNT=argc>2?atoi(argv[2]):4;
    int K=3584,N=18944; gK=K; gN=N;
    printf("int5_probe: M=1 K=%d N=%d  int4=%.0fMB int5=%.0fMB int8=%.0fMB  %d threads\n",
           K,N,(double)N*K/2/1e6,(double)N*K*5/8/1e6,(double)N*K/1e6,gNT);
    gA=malloc(K); for(int k=0;k<K;k++) gA[k]=(int8_t)((k%17)-8);
    gB8=malloc((size_t)N*K); gB4=malloc((size_t)N*K/2);
    gN5=malloc((size_t)N*K/2); gM5=malloc((size_t)N*K/8);
    gC=malloc((size_t)N*4);
    /* fill weights with int5 codes q in [-15,15]; pack all forms in the LEAN 32-block layout:
     * nibble byte (16b+i): low=w[32b+i], high=w[32b+16+i]; msb 4 bytes/block: [km,km+1]=lo half, [km+2,km+3]=hi */
    #define QCODE(n,k) ( ((int)(((size_t)(n)*131+(k)*7)%31)) - 15 )   /* deterministic in [-15,15] */
    for(size_t n=0;n<(size_t)N;n++){
        int8_t *b8=gB8+n*K; uint8_t *b4=gB4+n*(K/2), *n5=gN5+n*(K/2), *m5=gM5+n*(K/8);
        memset(m5,0,K/8);
        for(int b=0;b<K/32;b++){ int km=b*4;
            for(int i=0;i<16;i++){ int kl=32*b+i, kh=32*b+16+i; int ql=QCODE(n,kl), qh=QCODE(n,kh);
                b8[kl]=(int8_t)ql; b8[kh]=(int8_t)qh;
                int q4l=ql>7?7:ql<-7?-7:ql, q4h=qh>7?7:qh<-7?-7:qh;
                b4[16*b+i]=(uint8_t)((q4l&0xf)|((q4h&0xf)<<4));
                n5[16*b+i]=(uint8_t)((ql&0xf)|((qh&0xf)<<4));
                if((ql>>4)&1) m5[km   + i/8] |= (uint8_t)(1<<(i&7));
                if((qh>>4)&1) m5[km+2 + i/8] |= (uint8_t)(1<<(i&7));
            }
        }
    }
    /* ---- CORRECTNESS: int5 inflate must reproduce q exactly (lossless round-trip) ---- */
    int bad=0; for(int n=0;n<8 && !bad;n++){ const uint8_t*nb=gN5+(size_t)n*(K/2), *mb=gM5+(size_t)n*(K/8);
        for(int b=0;b<K/32 && !bad;b++){ int km=b*4;
            for(int i=0;i<16;i++){ int nl=nb[16*b+i]&0xf, nh=nb[16*b+i]>>4;
                int ml=(mb[km+i/8]>>(i&7))&1, mh=(mb[km+2+i/8]>>(i&7))&1;
                int vl=(int8_t)((nl|(ml<<4))<<3); vl>>=3; int vh=(int8_t)((nh|(mh<<4))<<3); vh>>=3;
                int kl=32*b+i, kh=32*b+16+i;
                if(vl!=QCODE(n,kl)||vh!=QCODE(n,kh)){printf("  MISMATCH n=%d b=%d i=%d\n",n,b,i);bad=1;break;} } } }
    /* also a full GEMV correctness check vs scalar int5 dot */
    { long ref=0; const int8_t*b8=gB8; for(int k=0;k<K;k++) ref+=(long)gA[k]*b8[k];   /* row0, int8==int5 codes */
      run(5); if(gC[0]!=ref){ printf("  GEMV MISMATCH row0: i5=%d ref=%ld\n",gC[0],(int)ref); bad=1; } }
    printf("  lossless round-trip + GEMV: %s\n", bad?"FAIL":"PASS");

    { int8_t lt[16]; for(int i=0;i<16;i++) lt[i]=(int8_t)(i-8); gLUT=vld1q_s8(lt); }   /* any 16-entry LUT (speed test) */
    run(4); run(6); run(5); run(8);   /* warm */
    double t0=now_us(); for(int i=0;i<iters;i++) run(4); double t4=(now_us()-t0)/iters;
    t0=now_us(); for(int i=0;i<iters;i++) run(6); double t6=(now_us()-t0)/iters;
    t0=now_us(); for(int i=0;i<iters;i++) run(5); double t5=(now_us()-t0)/iters;
    t0=now_us(); for(int i=0;i<iters;i++) run(8); double t8=(now_us()-t0)/iters;
    printf("  int4 (uniform):  %8.1f us  %6.1f GB/s (K/2 read)\n",  t4, (double)N*K/2/t4/1e3);
    printf("  NF4  (LUT vqtbl):%8.1f us  %6.1f GB/s (K/2 read)\n",  t6, (double)N*K/2/t6/1e3);
    printf("  int5 (bit-plane):%8.1f us  %6.1f GB/s (5K/8 read)\n", t5, (double)N*K*5/8/t5/1e3);
    printf("  int8:            %8.1f us  %6.1f GB/s (K   read)\n",  t8, (double)N*K/t8/1e3);
    printf("  ★ memory-bound-free? (time vs int8's mem-bound, lower time=better): int4 %.2fx NF4 %.2fx int5 %.2fx  [1.0=int8 time]\n",
           t4/t8, t6/t8, t5/t8);
    return bad?2:0;
}
