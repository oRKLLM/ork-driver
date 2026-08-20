/* npu/f16.c — the fp16 (W16A16) datapath.
 *
 * regcmd synthesis and output-stage setup for fp16, the fused matmul+activation path (SiLU / arbitrary
 * LUT-defined f(x) in one submit) and its LUT builders, per-channel fp16 multiply/scale, the fp16
 * elementwise ops, the round-robin and chained-multicore fp16 streams, batched fp16 GEMM, and the vendor
 * RE replays (forward softmax, contiguous->atom-8 reshape, standalone SiLU).
 *
 * Lifted verbatim from npu.c by the precision split (MODULARIZE_PLAN.md round 1). */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include "ork_regs.h"
#include "regcmd_array_4x32x16.h"
#include "regcmd_i8.h"
#include "regcmd_silu.h"
#include "regcmd_ewmul.h"
#include "regcmd_softmax_f16.h"
#include "regcmd_softmax_wt.h"
#include "regcmd_reshape.h"
#include "npu/internal.h"
#include <fcntl.h>
#include "npu/core.h"

/* f16-owned types and state (moved with their functions) */
struct { uint32_t blk, reg, val; } orki_f16_fovr[16]; int orki_f16_fovr_n=0;
struct tile_i8f16_arg { f16 *bb; const int8_t *Bi; const float *bscale; int KT, k0, n0, N; };
struct f16lut_rsqrt_ctx { int n_feat; double eps; };
struct f16act_neg { double (*fn)(double,void*); void *ctx; };
struct ork_rsh_patch { ork_npu *c; struct rknpu_task marker; uint32_t idx; uint32_t delay_us;
    int rcmode; uint32_t *rcword; uint32_t rcval0,rcval1; };  /* rcmode: patch a regcmd word in c->regcmd instead of a descriptor */
struct ork_pcfd_arg { int fd, core; struct buf *tk; int rc; };
struct streamw_f16 { ork_npu *c; int core; int S; const ork_mm_task_f16 *tasks; int *ctr; int rc; };
struct streamw_f16ch { ork_npu *c; int core; int ncore; int S; const ork_mm_task_f16 *tasks; int rc; };


void ork_f16_fuzz_clear(void){ orki_f16_fovr_n=0; }

void ork_f16_fuzz_add(uint32_t blk,uint32_t reg,uint32_t val){ if(orki_f16_fovr_n<16){ orki_f16_fovr[orki_f16_fovr_n].blk=blk; orki_f16_fovr[orki_f16_fovr_n].reg=reg; orki_f16_fovr[orki_f16_fovr_n].val=val; orki_f16_fovr_n++; } }

void orki_set_f16_out(uint32_t*rc,int N,int stride){
    int s=stride>0?stride:N; (void)s;
    unsigned r04=getenv("ORK_F16OUT_4004")?strtoul(getenv("ORK_F16OUT_4004"),0,0):0x0000000e;
    /* 0x4010 DATA_FORMAT fields (rocket registers.xml, confirmed by our ewmul fp16=0x48000002):
     * OUT_PRECISION[31:29] | IN_PRECISION[28:26] | PROC_PRECISION[2:0], precision enum int8=0/int16=1/fp16=2.
     * INT8->FP16 = OUT=fp16(2)<<29 | IN=int8(0)<<26 | PROC=int8(0) = 0x40000000. */
    unsigned r10=getenv("ORK_F16OUT_4010")?strtoul(getenv("ORK_F16OUT_4010"),0,0):0x40000000;
    unsigned r50=getenv("ORK_F16OUT_4050")?strtoul(getenv("ORK_F16OUT_4050"),0,0):0x0000036e;
    unsigned rc0=getenv("ORK_F16OUT_40c0")?strtoul(getenv("ORK_F16OUT_40c0"),0,0):0x00000040; /* 2-byte elem */
    unsigned r84=getenv("ORK_F16OUT_4084")?strtoul(getenv("ORK_F16OUT_4084"),0,0):0x00000001; /* gain */
    unsigned r88=getenv("ORK_F16OUT_4088")?strtoul(getenv("ORK_F16OUT_4088"),0,0):0x00000000; /* shift */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_S_POINTER,r04);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_PRECISION,r10);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_DATA_CUBE_NOTCH,(((N/4)-1)<<16)|((N/4)-1));
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_DST_N_DIMS,((N-1)<<16)|(N-1));
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_BS_OW_CFG,r50);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_DST_N2,N-1);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_OFFSET,0);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_SCALE,r84);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_SHIFT,r88);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_SURFACE_ADD,rc0);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_R40C4,0);
}

void orki_set_f16_out_fp16in(uint32_t*rc,int M,int N){
    /* fp16-in fp16-out DPU output stage. Precision/bypass/CVT regs from a captured VENDOR fp16->fp16 MATMUL
     * (~/rknn_sdk/cap_fp16f16.dec): 0x4040=0x53 (BS FULLY bypassed — a MATMUL, not the conv's BS-active 0x20150
     * that hung it), 0x4010 OUT=fp16, 0x4084 FP32TOFP16_EN(bit16). The vendor's own capture was CONTIGUOUS
     * (0x4024=0x10, 0x40c0=0x20) — but to CHAIN into the per-channel SDP we instead emit the ATOM-8 cube the
     * SDP reads (orki_set_mul_geom layout: surface stride M*16, PC16/EWCUBEH). So the geometry here mirrors
     * orki_set_mul_geom's OUTPUT side (0x4024/0x4030/0x403c/0x4058/0x405c/0x40c0 = M*16 / M-1 / N), overriding synth's
     * contiguous fp32-atom geometry. M param needed for the M*16 surface stride. */
    /* Precision/bypass/CVT regs from the captured vendor fp16->fp16 MATMUL (cap_fp16f16.dec). */
    orki_setrn(rc,REGCMD_N,RK_DPU_S_POINTER,0x0000000e);
    orki_setrn(rc,REGCMD_N,RK_DPU_FEATURE_MODE_CFG,0x000001e4);                       /* FEATURE_MODE OUTPUT_MODE=2 */
    orki_setrn(rc,REGCMD_N,RK_DPU_OUT_PRECISION,0x48000002);                       /* DATA_FORMAT fp16/fp16/fp16 */
    orki_setrn(rc,REGCMD_N,RK_DPU_BS_CFG,0x00000053);                       /* BS FULLY BYPASSED (matmul, NOT conv's 0x20150) */
    orki_setrn(rc,REGCMD_N,RK_DPU_BS_OW_CFG,0x00000126);                       /* BS_OW_CFG */
    orki_setrn(rc,REGCMD_N,RK_DPU_BN_CFG,0x00000053);                       /* BN fully bypassed */
    orki_setrn(rc,REGCMD_N,RK_DPU_EW_CFG,0x00000383);                       /* EW fully bypassed */
    orki_setrn(rc,REGCMD_N,RK_DPU_EW_CVT_SCALE,0x00000001);
    orki_setrn(rc,REGCMD_N,RK_DPU_OUT_CVT_OFFSET,0x00000000);
    orki_setrn(rc,REGCMD_N,RK_DPU_OUT_CVT_SCALE,0x00010001);                       /* OUT_CVT_SCALE: FP32TOFP16_EN=1 | scale=1  <-- KEY */
    orki_setrn(rc,REGCMD_N,RK_DPU_OUT_CVT_SHIFT,0x00000000);
    orki_setrn(rc,REGCMD_N,RK_DPU_R40C4,0x00000000);
    if(getenv("ORK_F16_ATOM8")){
        /* ATOM-8 fp16 output — geometry taken EXACTLY from the bit-exact atom-8 SDP (REGCMD_MUL_F16 / orki_set_mul_geom):
         * surface stride M*16, 0x4050 BS_OW_CFG=0x2 (OD_BYPASS, SIZE_E=0 — NOT the contiguous 0x126 or the int16
         * 0x248 that hung), 0x4038=0. This makes the fp16 matmul emit the PC16 atom-8 layout the SDP reads, so
         * ork_npu_mul_perchan_f16 can consume it directly (no reshape, no notch). */
        uint32_t s=(uint32_t)(M*16);
        orki_setrn(rc,REGCMD_N,RK_DPU_DST_SURF_STRIDE,s);                            /* DST_SURF_STRIDE = M*16 */
        orki_setrn(rc,REGCMD_N,RK_DPU_DATA_CUBE_WIDTH,(uint32_t)(M-1));
        orki_setrn(rc,REGCMD_N,RK_DPU_DATA_CUBE_HEIGHT,0);
        orki_setrn(rc,REGCMD_N,RK_DPU_DATA_CUBE_NOTCH,0);
        orki_setrn(rc,REGCMD_N,RK_DPU_DST_N_DIMS,(uint32_t)(((N-1)<<16)|(N-1)));
        orki_setrn(rc,REGCMD_N,RK_DPU_BS_OW_CFG,0x00000002);                   /* BS_OW_CFG = 0x2 (atom-8; KEY) */
        orki_setrn(rc,REGCMD_N,RK_DPU_DST_N2,(uint32_t)(N-1));
        orki_setrn(rc,REGCMD_N,RK_DPU_WDMA_SIZE_1,(uint32_t)(M-1));
        orki_setrn(rc,REGCMD_N,RK_DPU_SURFACE_ADD,s);                            /* SURFACE_ADD = M*16 */
    } else {
        (void)M;                                                      /* CONTIGUOUS [M][N] (cap_fp16f16); PROVEN 512/512 standalone */
        orki_setrn(rc,REGCMD_N,RK_DPU_DATA_CUBE_NOTCH,(((N/8)-1)<<16)|((N/8)-1));
        orki_setrn(rc,REGCMD_N,RK_DPU_SURFACE_ADD,0x00000020);
    }
}

static void tile_i8_to_f16_range(int lo,int hi,void *a){
    struct tile_i8f16_arg *t=a; int KT=t->KT,N=t->N,k0=t->k0,n0=t->n0;
    for(int nt=lo;nt<hi;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++){
        int n=n0+nt*16+nl; float s=t->bscale?t->bscale[n]:1.0f;
        f16 *dst=t->bb+((size_t)nt*KT*16*32+(size_t)kt*16*32+(size_t)nl*32);
        const int8_t *src=t->Bi+(size_t)(k0+kt*32)*N+n;   /* src[kk*N] = element (k0+kt*32+kk, n) */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        /* NEON: gather the 32 strided (by N) int8, then widen int8->int32->f32, x s (broadcast), ->f16.
         * The scale is per-channel so constant across the 32 K-values -> a single vdup broadcast. Same ops
         * (float mul then f16 RNE cast) as the scalar path => BIT-IDENTICAL (validated by jit_inflate_check). */
        int8_t buf[32]; for(int kk=0;kk<32;kk++) buf[kk]=src[(size_t)kk*N];
        float32x4_t vs=vdupq_n_f32(s);
        for(int b=0;b<32;b+=8){
            int16x8_t i16=vmovl_s8(vld1_s8(buf+b));
            float32x4_t f0=vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16))),vs);
            float32x4_t f1=vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16))),vs);
        #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            vst1_f16((float16_t*)(dst+b),  vcvt_f16_f32(f0));
            vst1_f16((float16_t*)(dst+b+4),vcvt_f16_f32(f1));
        #else
            float tmp[8]; vst1q_f32(tmp,f0); vst1q_f32(tmp+4,f1);
            for(int j=0;j<8;j++) dst[b+j]=(f16)tmp[j];
        #endif
        }
#else
        for(int kk=0;kk<32;kk++) dst[kk]=(f16)((float)src[(size_t)kk*N]*s);
#endif
    }
}

ork_w *ork_mm_f16_scratch(ork_npu *c,int K,int N){
    if(K%32||N%16) return NULL;
    int KS=c->soc->ks, NMAX=c->soc->nmax;
    int Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    ork_w *w=calloc(1,sizeof *w); if(!w) return NULL;
    w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_F16;w->owns=1;w->domain=ork_dom(c->pack_domain);
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf)); if(!w->Bb){free(w);return NULL;}
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;(void)n0;(void)k0;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=orki_bcreate(c->fd,(size_t)Kp*Nc*2,0x403,w->domain);
        if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++)orki_bdestroy(c->fd,&w->Bb[i]); free(w->Bb);free(w);return NULL; }
        /* fresh buffers need the double init-sync (a single TO leaves the device side uninitialized). */
        orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}}
    return w;
}

int ork_mm_inflate_i8_to_f16(ork_npu *c,ork_w *w,const int8_t *i8,const float *bscale,int K,int N){
    if(!w || w->dtype!=DT_F16 || !w->Bb) return -1;
    if(w->K!=K || w->N!=N || !i8) return -2;
    int KS=c->soc->ks, NMAX=c->soc->nmax, Sk=w->Sk, Sn=w->Sn;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/16;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; if(!b->cpu) return -1;
        struct tile_i8f16_arg ta={b->cpu,i8,bscale,KT,k0,n0,N};
        ork_parallel_for(NN,tile_i8_to_f16_range,&ta);
        orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}}
    return 0;
}

int ork_mm_repack_f16(ork_npu *c,ork_w *w,int K,int N,const f16 *B){
    if(!w || w->dtype!=DT_F16 || !w->Bb) return -1;
    if(w->K!=K || w->N!=N) return -2;
    int KS=c->soc->ks, NMAX=c->soc->nmax, Sk=w->Sk, Sn=w->Sn;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/16;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; if(!b->cpu) return -1; f16*bb=b->cpu;
        for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
            bb[(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(k0+kt*32+kk)*N+(n0+nt*16+nl)];
        orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}}
    /* DERIVED-COPY COHERENCE (defect fix). The fp16 MULTI-CORE colsplit does NOT read Bb: it builds a
     * CONTIGUOUS concatenation of the Sk K-slice tiles ONCE and caches it on the weight (w->Bbc for
     * Sn==1, w->Bbc_ns[] for Sn>1) behind a *_valid latch that was never cleared. A repack that
     * refreshed only Bb was therefore INVISIBLE to ork_mm_run()/run_multicore — every multi-core submit
     * after the first kept computing against the FIRST weight ever packed into this slot, silently.
     * Single-core orki_run() and ork_mm_run_stream_f16{,_chain} read Bb directly and were always correct,
     * which is why this only surfaced in a repack-per-batch-slice caller (ggml-ork's ork_bmm_fp16:
     * batch slice 0 correct, every later slice stale — probe: scratchpad bmm_probe, Hkv=16/rk2=1 gave
     * NRMSE 1.37 from head 1 on multi-core, PASS with ORK_NPU_MC=1). Refresh the copies in place: K,N
     * are unchanged so the sizes match, and keeping them valid avoids any bcreate/bdestroy churn. */
    if(w->Bbc_valid && w->Bbc.cpu){ size_t off=0;
        for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS; size_t sz=(size_t)Kp*N*2;
            if(off+sz>w->Bbc.size || !w->Bb[ks].cpu) break;
            memcpy((char*)w->Bbc.cpu+off, w->Bb[ks].cpu, sz); off+=sz;}
        orki_bsync(c->fd,&w->Bbc,RKNPU_MEM_SYNC_TO_DEVICE);}
    if(w->Bbc_ns_valid && w->Bbc_ns){
        for(int ns=0;ns<Sn;ns++){ if(!w->Bbc_ns[ns].cpu) continue;
            int c0=ns*NMAX, sw=(N-c0<NMAX)?(N-c0):NMAX; size_t off=0;
            for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS; size_t sz=(size_t)Kp*sw*2;
                if(off+sz>w->Bbc_ns[ns].size || !w->Bb[(size_t)ns*Sk+ks].cpu) break;
                memcpy((char*)w->Bbc_ns[ns].cpu+off, w->Bb[(size_t)ns*Sk+ks].cpu, sz); off+=sz;}
            orki_bsync(c->fd,&w->Bbc_ns[ns],RKNPU_MEM_SYNC_TO_DEVICE);}}
    return 0;
}

int ork_f16_colsplit(void){ static int v=-1; if(v<0){const char*e=getenv("ORK_F16_COLSPLIT"); v=e?atoi(e):1; } return v; }   /* colsplit is the ONLY fp16 multicore path (#45); ORK_F16_COLSPLIT=0 -> single-core fp16 reference (never mcworker) */

static inline int orki_f16_mtile(int K,int M){
    double scale=(double)K/256.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale);
    int mg_max = base>=0x1b ? (base-0x1b)/slope+1 : 0; int chunk = mg_max*64;
    const char*e=getenv("ORK_F16_MTILE"); if(e){ int v=atoi(e); if(v>0)chunk=v; }
    if(chunk<1)chunk=1; if(chunk>M)chunk=M; return chunk;
}

static void set_f16_silu(uint32_t*rc,uint32_t out_bias,uint32_t idx_off,uint32_t cfg4068){
    { const char*e=getenv("ORK_F16_C4004"); uint32_t v=e?(uint32_t)strtoul(e,0,0):0x0030;
      orki_setrn(rc,REGCMD_N,RK_DPU_S_POINTER,v); orki_setrn(rc,REGCMD_N,RK_SDP_5004,v); } /* activation mode on */
    /* 0x4010 = fp16 output CVT (post-LUT). Deliberately kept at REGCMD's 0xa8000002 (fp16->fp32); overriding
     * is WEDGE-PRONE (proc-precision mismatch). ORK_F16_C4010 for the upper-bank RE probe only. */
    { const char*e=getenv("ORK_F16_C4010"); if(e) orki_setrn(rc,REGCMD_N,RK_DPU_OUT_PRECISION,(uint32_t)strtoul(e,0,0)); }
    /* index/output gain (0x4084/0x4088): REGCMD's default is ~1 -> gate barely moves the LUT index (curve
     * under-sampled). Env-override to spread gate over the LUT (fp16 analog of the int8 acc->index R). */
    { const char*g=getenv("ORK_F16_R84"); if(g){ orki_setrn(rc,REGCMD_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)strtoul(g,0,0));
        const char*s=getenv("ORK_F16_R88"); orki_setrn(rc,REGCMD_N,RK_DPU_OUT_CVT_SHIFT,s?(uint32_t)strtoul(s,0,0):0); } }
    { const char*e=getenv("ORK_F16_C4060"); orki_setrn(rc,REGCMD_N,RK_DPU_BN_CFG,e?(uint32_t)strtoul(e,0,0):0x00020040); }   /* silu LUT-stage config (shared with the int8 fused path) */
    /* 0x4064 = fp16 index-scale param. REGCMD's default gives a small gate-dependent spread; 0xffff7dc8
     * (standalone silu) COLLAPSES it. Keep REGCMD's default unless env-overridden (calibration RE). */
    { const char*e=getenv("ORK_F16_C4064"); if(e) orki_setrn(rc,REGCMD_N,RK_DPU_BN_ALU_CFG,(uint32_t)strtoul(e,0,0)); }
    /* 0x4044 = BS_ALU_OPERAND (za), a PRE-LUT bias on the accumulator. The fp16 index only spreads for
     * NEGATIVE acc; setting za shifts the gate negative so positive gates fall into the spreading region
     * (negatives then clamp ~0, ~= silu(neg)). Env-overridable for the calibration crack. */
    { const char*e=getenv("ORK_F16_ZA"); if(e) orki_setrn(rc,REGCMD_N,RK_DPU_BS_ALU_CFG,(uint32_t)strtoul(e,0,0)); }
    orki_setrn(rc,REGCMD_N,RK_DPU_BN_MUL_CFG,cfg4068);
    { const char*e=getenv("ORK_F16_C4070"); orki_setrn(rc,REGCMD_N,RK_DPU_EW_CFG,e?(uint32_t)strtoul(e,0,0):0x00000302); }
    orki_setrn(rc,REGCMD_N,RK_DPU_OUT_CVT_OFFSET,out_bias);
    { const char*e=getenv("ORK_F16_C4108"); orki_setrn(rc,REGCMD_N,RK_DPU_R4108,e?(uint32_t)strtoul(e,0,0):0x00000068); }
    { const char*e=getenv("ORK_F16_C410C"); orki_setrn(rc,REGCMD_N,RK_DPU_R410C,e?(uint32_t)strtoul(e,0,0):0x00050500); }
    orki_setrn(rc,REGCMD_N,RK_DPU_R4110,idx_off);
    { const char*e=getenv("ORK_F16_C411C"); orki_setrn(rc,REGCMD_N,RK_DPU_R411C,e?(uint32_t)strtoul(e,0,0):0x00004000); }
    { const char*e=getenv("ORK_F16_C4128"); orki_setrn(rc,REGCMD_N,RK_DPU_R4128,e?(uint32_t)strtoul(e,0,0):0x40320000); }
    { const char*e=getenv("ORK_F16_C412C"); orki_setrn(rc,REGCMD_N,RK_DPU_R412C,e?(uint32_t)strtoul(e,0,0):0x000001a0); }
    /* 0x4010/0x40c0/0x4050/0x4084/0x4088 deliberately UNTOUCHED: REGCMD's fp16 output CVT is kept. */
}

int ork_mm_run_f16_silu(ork_npu *c,ork_w *w,int M,const ork_f16 *A,float *C,
                        uint32_t out_bias,uint32_t idx_off,uint32_t cfg4068,const int16_t *lut,int nlut){
    if(!ork_ppu_fuse_enabled(c)) return -3;
    /* fp16 weights live in w->Bb tiles (Bf is int8-only). Fused silu needs the WHOLE-K weight in one buffer,
     * so require a single tile: Sk==1 (K within one fp16 K-slice, <=2048) and Sn==1 (N<=nmax). */
    if(w->dtype!=DT_F16 || !w->Bb || w->Sk!=1 || w->Sn!=1) return -2;
    int fd=c->fd,K=w->K,N=w->N,NMAX=c->soc->nmax,CBUF=c->soc->cbuf_elems;
    if(K%32 || N%16 || N>NMAX) return -2;
    if(CBUF>32768) CBUF=32768;                              /* fp16 keeps its validated 32768 tiling */
    if(w->domain!=c->dom_active || (w->domain!=0 && !c->dom_save)) orki_dom_activate(c,w->domain);
    if(DT_F16!=c->last_dt){ int kw=ork_f16warm()&&ORK_KW_DT(c->last_dt); if(!kw)c->warmed=0; if(!ork_nothrash()&&!kw)c->ccsz=0; c->last_dt=DT_F16; }   /* NOTHRASH: reuse Cc, no realloc under IOVA pressure (see orki_run()) */
    int chunk=orki_f16_mtile(K,M);   /* fp16 M-tile = the 0x1040 schedule's bit-exact ceiling mg_max*64 (was hardcoded 16, ~4-20x too small); ORK_F16_MTILE overrides */
    size_t maxaf=(size_t)chunk*K*2, maxout=(size_t)chunk*NMAX*4;   /* A fp16 (2B), C fp32 (4B) */
    if(c->Af.size<maxaf || c->Af.domain!=c->dom_active){ orki_bdestroy(fd,&c->Af); c->Af=orki_bcreate(fd,maxaf,0x403,c->dom_active); if(!c->Af.cpu)return -2; }
    if(c->ccsz<maxout || c->Cc.domain!=c->dom_active){ orki_bdestroy(fd,&c->Cc); c->Cc=orki_bcreate(fd,maxout,0x403,c->dom_active); c->ccsz=maxout; c->warmed=0; if(!c->Cc.cpu)return -2; }
    struct buf Lrc=orki_bcreate(fd,(size_t)REGCMD_SILU_LUT_N*4,0x403,c->dom_active); if(!Lrc.cpu)return -2;
    struct buf Lsc=orki_bcreate(fd,4096,0x403,c->dom_active); if(!Lsc.cpu){orki_bdestroy(fd,&Lrc);return -2;}
    memcpy(Lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
    orki_setrn((uint32_t*)Lrc.cpu,REGCMD_SILU_LUT_N,RK_DPU_DST_BASE_ADDR,(uint32_t)Lsc.dma);
    if(lut){ uint32_t*lr=(uint32_t*)Lrc.cpu; int j=0;
        for(int k=0;k+1<REGCMD_SILU_LUT_N;k+=2){ if((lr[k]&0xffff)==0x4104){ int32_t v=(j<nlut)?(int32_t)lut[j]:0; j++;
            lr[k]=0x4104|((uint32_t)(v&0xffff)<<16); lr[k+1]=(0x1001u<<16)|(((uint32_t)v>>16)&0xffff); } } }
    orki_bsync(fd,&Lrc,RKNPU_MEM_SYNC_TO_DEVICE);
    int rc_ret=0;
    { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
      t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=1097; t->regcmd_addr=Lrc.dma;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      /* ping-pong OFF (0x1 = RKNPU_JOB_PC, NOT 0x5) for the LUT-load submit: ping-pong (1<<2) signals the
       * task "config done" the instant its register config completes, racing the LUT's SRAM-commit side
       * effect — the following matmul submit then reads a half-committed LUT -> wrong silu -> garbage output
       * (non-deterministic, worsens with scale: bit-exact-looking per-op but PPL blows up over many tokens).
       * See AGENTS.md / NPU-Quirks "Ping-pong races a chained task's side effect". */
      struct rknpu_submit ls;memset(&ls,0,sizeof ls);ls.flags=0x1;ls.task_number=1;ls.task_obj_addr=c->task.obj;ls.core_mask=RKNPU_CORE0_MASK;ls.fence_fd=-1;ls.timeout=orki_ew_timeout_ms();ls.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      if(orki_rknpu_submit_ioctl(fd,&ls,c->dom_active)) rc_ret=-1; }
    /* fp16 single-N-tile (N<=NMAX); K single-slice (caller keeps K within the fp16 envelope). */
    for(int m0=0;m0<M && rc_ret==0;m0+=chunk){ int mc=(M-m0<chunk)?(M-m0):chunk; if(mc<=0)continue;
        ork_f16*ad=c->Af.cpu; for(int r=0;r<mc;r++)for(int j=0;j<K;j++) ad[(size_t)r*K+j]=A[(size_t)(m0+r)*K+j];
        orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
        uint32_t rc[REGCMD_N];
        orki_synth(rc,mc,K,N,(uint32_t)c->Af.dma,(uint32_t)w->Bb[0].dma,(uint32_t)c->Cc.dma,1,CBUF);
        set_f16_silu(rc,out_bias,idx_off,cfg4068);
        memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
        { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
          t->enable_mask=0x1d; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=REGCMD_N; t->regcmd_addr=c->regcmd.dma;
          orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); }
        if(orki_submit1(c)){ rc_ret=-1; break; }
        float*cc=c->Cc.cpu; for(int r=0;r<mc;r++)for(int n=0;n<N;n++) C[(size_t)(m0+r)*N+n]=cc[(size_t)r*N+n];
    }
    orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);
    return rc_ret;
}

int ork_mm_build_f16_lut(ork_npu *c, double (*fn)(double,void*), void *fnctx,
                         double in_lo, double in_hi, int16_t *lut, double *S_out, double *R_out, double *out_scale_out){
    if(!ork_ppu_fuse_enabled(c) || in_hi<=in_lo || !fn) return -2;
    const int Kp=512, Np=64;
    double atgt = getenv("ORK_F16_ATGT") ? atof(getenv("ORK_F16_ATGT")) : 150.0;
    double amax = fabs(in_lo)>fabs(in_hi) ? fabs(in_lo) : fabs(in_hi); if(amax<=0) return -2;
    double S = atgt/amax;                                    /* acc=-S*x lands in the fp16 negative spread band */
    ork_f16 *A=malloc((size_t)8*Kp*2), *B=malloc((size_t)Kp*Np*2); float *C=malloc((size_t)8*Np*4);
    if(!A||!B||!C){ free(A);free(B);free(C); return -2; }
    for(int i=0;i<8*Kp;i++)A[i]=(ork_f16)1.0f;
    double tru[64];
    for(int n=0;n<Np;n++){ tru[n]=in_lo+(in_hi-in_lo)*n/(double)(Np-1); double b=(-S*tru[n])/(double)Kp; for(int k=0;k<Kp;k++)B[(size_t)k*Np+n]=(ork_f16)b; }
    ork_w *w=ork_mm_pack(c,Kp,Np,B); if(!w){ free(A);free(B);free(C); return -2; }
    #define F16LRUN() ork_mm_run_f16_silu(c,w,8,A,C,0,0xffffc000u,0x56391100u,lut,1030)
    int rc=-2;
    for(int i=0;i<1030;i++)lut[i]=1000; if(F16LRUN()){goto done;} double o1=C[32];
    for(int i=0;i<1030;i++)lut[i]=3000; if(F16LRUN()){goto done;} double o2=C[32];
    double R=(o2-o1)/2000.0, bias=o1-R*1000.0; if(fabs(R)<1e-9){goto done;}
    for(int i=0;i<1030;i++)lut[i]=(int16_t)(i-512); if(F16LRUN()){goto done;}
    int idx[64]; for(int n=0;n<Np;n++) idx[n]=(int)lround((C[n]-bias)/R)+512;
    double omax=0; for(int n=0;n<Np;n++){ double f=fabs(fn(tru[n],fnctx)); if(f>omax)omax=f; }
    double out_scale = omax/8000.0; if(out_scale<=0) out_scale=1e-3;
    int set[1030]; for(int i=0;i<1030;i++){lut[i]=0;set[i]=0;}
    for(int n=0;n<Np;n++){ int i=idx[n]; if(i<0||i>1029)continue;
        double v=(fn(tru[n],fnctx)/out_scale - bias)/R; long q=lround(v); if(q>32767)q=32767; if(q<-32768)q=-32768; lut[i]=(int16_t)q; set[i]=1; }
    int lo=-1,hi=-1; for(int i=0;i<1030;i++)if(set[i]){lo=i;break;} for(int i=1029;i>=0;i--)if(set[i]){hi=i;break;}
    if(lo<0)goto done;
    for(int i=0;i<lo;i++)lut[i]=lut[lo]; for(int i=hi+1;i<1030;i++)lut[i]=lut[hi];
    for(int i=lo;i<=hi;i++){ if(set[i])continue; int a=i,b=i; while(a>lo&&!set[a])a--; while(b<hi&&!set[b])b++;
        lut[i]=(int16_t)(lut[a]+(lut[b]-lut[a])*(i-a)/(b-a)); }
    if(S_out)*S_out=S; if(R_out)*R_out=R; if(out_scale_out)*out_scale_out=out_scale; rc=0;
done:
    #undef F16LRUN
    ork_mm_free(c,w); free(A);free(B);free(C); return rc;
}

static double f16lut_silu(double x, void *ctx){ (void)ctx; return orki_silu_f(x); }

static double f16lut_rsqrt(double x, void *ctx){ struct f16lut_rsqrt_ctx *p=ctx; return 1.0/sqrt(x/(double)p->n_feat + p->eps); }

static double f16act_negtramp(double u, void *p){ struct f16act_neg *q=p; return q->fn(-u,q->ctx); }

ork_w *ork_mm_pack_f16_fused_act(ork_npu *c, int K, int N, const ork_f16 *B,
                                 double (*fn)(double,void*), void *fnctx, double in_lo, double in_hi){
    if(!ork_ppu_fuse_enabled(c)) return NULL;
    if(!c||!B||!fn||K%32||N%16||in_hi<=in_lo) return NULL;
    double packsign, blo, bhi; double (*bfn)(double,void*); void *bctx; struct f16act_neg neg;
    if(in_lo >= 0){ bfn=fn; bctx=fnctx; blo=in_lo; bhi=in_hi; packsign=-1.0; }
    else if(in_hi <= 0){ neg.fn=fn; neg.ctx=fnctx; bfn=f16act_negtramp; bctx=&neg; blo=-in_hi; bhi=-in_lo; packsign=+1.0; }
    else return NULL;   /* mixed sign — unsupported by the single-signed fp16 index */
    int16_t *lut=malloc(1030*sizeof(int16_t)); if(!lut) return NULL;
    double S=0,R=0,osc=0;
    if(ork_mm_build_f16_lut(c,bfn,bctx,blo,bhi,lut,&S,&R,&osc)){ free(lut); return NULL; }
    ork_f16 *Bs=malloc((size_t)K*N*sizeof(ork_f16)); if(!Bs){ free(lut); return NULL; }
    for(size_t i=0;i<(size_t)K*N;i++) Bs[i]=(ork_f16)(packsign*S*(double)(float)B[i]);   /* acc = packsign*S*(A·B) < 0 */
    ork_w *w=ork_mm_pack(c,K,N,Bs); free(Bs);
    if(!w){ free(lut); return NULL; }
    w->fa_lut=lut; w->fa_osc=osc;   /* baked into the resident weight; freed by ork_mm_free */
    return w;
}

int ork_mm_run_f16_fused_act(ork_npu *c, ork_w *w, int M, const ork_f16 *A, float *C){
    if(!c||!w||!A||!C||M<1) return -2;
    if(!w->fa_lut) return -2;   /* not a fused-activation weight (use ork_mm_pack_f16_fused_act) */
    int rc = ork_mm_run_f16_silu(c,w,M,A,C,0,0xffffc000u,0x56391100u,w->fa_lut,1030);   /* matmul + fused LUT, 1 submit */
    if(rc==0){ double osc=w->fa_osc; for(size_t i=0;i<(size_t)M*w->N;i++) C[i]=(float)((double)C[i]*osc); }   /* recover fn(A·B) */
    return rc;
}

int ork_mm_run_f16_act(ork_npu *c, int K, int N, const ork_f16 *B, int M, const ork_f16 *A, float *C,
                       double (*fn)(double,void*), void *fnctx, double in_lo, double in_hi){
    if(!A||!C||M<1) return -2;
    ork_w *w=ork_mm_pack_f16_fused_act(c,K,N,B,fn,fnctx,in_lo,in_hi);   /* calibrate + orki_pack (one-shot) */
    if(!w) return -2;
    int rc=ork_mm_run_f16_fused_act(c,w,M,A,C);
    ork_mm_free(c,w);
    return rc;
}

int ork_mm_build_f16_silu_lut(ork_npu *c, double Gmax, int16_t *lut, double *S_out, double *R_out, double *out_scale_out){
    if(Gmax<=0) return -2;
    double gcap = getenv("ORK_F16_GCAP") ? atof(getenv("ORK_F16_GCAP")) : 40.0;
    if(gcap>0 && Gmax>gcap) Gmax=gcap;
    return ork_mm_build_f16_lut(c, f16lut_silu, NULL, -Gmax, Gmax, lut, S_out, R_out, out_scale_out);
}

int ork_mm_build_f16_rsqrt_lut(ork_npu *c, int n_feat, double eps, double ss_min, double ss_max,
                               int16_t *lut, double *S_out, double *R_out, double *out_scale_out){
    if(n_feat<1) return -2;
    if(ss_min<0) ss_min=0;
    struct f16lut_rsqrt_ctx ctx = { n_feat, eps };
    return ork_mm_build_f16_lut(c, f16lut_rsqrt, &ctx, ss_min, ss_max, lut, S_out, R_out, out_scale_out);
}

int ork_npu_probe_f16_mm(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B,float *raw){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%16||N>c->soc->nmax||M<1||M>64) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N*2,0x403,-1); if(!W.cpu) return -2;   /* fp16 weight: 2 B/elem */
    int NN=N/16,KT=K/32; uint16_t*bb=W.cpu;      /* fp16 weight tile [Ntile=16][Ktile=32][16][32] */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*16+nl)];
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)2*M*N*4,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}   /* fp32 out, 2x */
    uint16_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_N];
    orki_synth(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF);   /* ork_f16_fuzz overrides apply inside */
    struct buf extra[2] = {W, O};
    if (orki_validate_regcmd("probe_f16_mm", c, rc, REGCMD_N, NULL, extra, 2)) { orki_bdestroy(fd,&W); orki_bdestroy(fd,&O); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t to_ms=60000; { const char*e=getenv("ORK_I4_PROBE_TO_MS"); if(e){ unsigned v=(unsigned)strtoul(e,0,0); if(v)to_ms=v; } }
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1;
    for(int rep=0;rep<2;rep++){ sub.timeout=to_ms; if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; continue; }
        orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; }
    if(ok==0) memcpy(raw,O.cpu,(size_t)2*M*N*4);
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);
    return ok;
}

int ork_npu_probe_f16_mm_f16out(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B,uint16_t *out){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64||(N&7)) return -2;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)
    struct buf W=orki_bcreate(fd,(size_t)K*N*2,0x403,-1); if(!W.cpu) return -2;
    int NN=N/16,KT=K/32; uint16_t*bb=W.cpu;
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*16+nl)];
    size_t osz=(size_t)M*N*2; if(osz<4096)osz=4096;
    struct buf O=orki_bcreate(fd,osz,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;} memset(O.cpu,0,osz);
    /* ORK_F16_ROWPITCH=S: DISCOVERY probe — store the activation rows at pitch S>K (padding between rows) and
     * read them via CNA LINE_STRIDE=S/8. Validates the CNA reads STRIDED/non-contiguous activations directly
     * (the densify lever). Default: contiguous (pitch=K). */
    int rowpitch=getenv("ORK_F16_ROWPITCH")?atoi(getenv("ORK_F16_ROWPITCH")):K;
    uint16_t*ad=c->Af.cpu;
    if(rowpitch!=K){ for(int j=0;j<M*rowpitch;j++)ad[j]=0xdead; for(int m=0;m<M;m++)for(int k=0;k<K;k++)ad[(size_t)m*rowpitch+k]=A[(size_t)m*K+k]; }
    else for(int j=0;j<M*K;j++)ad[j]=A[j];
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_NONE);                  /* prime fp16 pipeline (layer owns the reset; keep-warm-aware) */
    uint32_t rc[REGCMD_N];
    int sched=getenv("ORK_F16_SCHED")?atoi(getenv("ORK_F16_SCHED")):((K&(K-1))==0 && K>=128 && K<2048);  /* run_stream_f16 rule; small K => 0 */
    orki_synth(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,sched,CBUF);
    if(rowpitch!=K) orki_setrn(rc,REGCMD_N,RK_CNA_DMA_CON1,rowpitch/8);        /* CNA LINE_STRIDE = pitch/8 surfaces (strided activation) */
    if(!getenv("ORK_F16_FP32OUT")) orki_set_f16_out_fp16in(rc,M,N);        /* vendor fp16-out stage (atom-8); skip => synth's native fp32-out (compute sanity) */
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task*t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0xd; t->int_mask=0x300; t->int_clear=0x1ffff;
      t->regcfg_amount=108; t->regcmd_addr=(uint32_t)c->regcmd.dma; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); }
    uint32_t to_ms=3000; { const char*e=getenv("ORK_EW_TIMEOUT"); if(e){ unsigned v=(unsigned)strtoul(e,0,0); if(v)to_ms=v; } }
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    sub.timeout=to_ms;
    int ok=-1;
    for(int rep=0;rep<2;rep++){ if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; continue; } orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; }  /* fp16 cold 2-pass re-warm */
    if(ok==0){
        int ewc=getenv("ORK_F16_ATOM8")?1:0;                          /* readback matches the output layout: default CONTIGUOUS; ORK_F16_ATOM8 => atom-8 */
        for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=*(uint16_t*)((char*)O.cpu+(ewc?EWCUBEH(m,n):((size_t)(m*N+n)*2)));
        if(getenv("ORK_F16_RAWDUMP")){ uint16_t*o=(uint16_t*)O.cpu; int nz=0,ne=osz/2; int first[16],nf=0;
            for(int i=0;i<ne;i++) if(o[i]){ nz++; if(nf<16){ first[nf++]=i; } }
            fprintf(stderr,"[f16raw] osz=%zu nonzero=%d/%d  first offsets(elem):",osz,nz,ne);
            for(int i=0;i<nf;i++) fprintf(stderr," %d=%.3g",first[i],(double)*(ork_f16*)&o[first[i]]);
            fprintf(stderr,"\n"); }
        ok=0; }
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);
    #undef EWCUBEH
    return ok;
}

int ork_mm_run_f16_f16out(ork_npu *c, ork_w *w, int M, const ork_f16 *A, ork_f16 *out){
    if(!c||!w||!A||!out) return -2;
    if(w->dtype!=DT_F16||w->Sn!=1||w->Sk!=1||!w->Bb) return -2;              /* single-slice fp16 (K<=ks, N<=nmax) */
    int K=w->K, N=w->N, fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64||(N&7)) return -2;
    if(w->domain!=c->dom_active || (w->domain!=0 && !c->dom_save)) orki_dom_activate(c,w->domain);   /* activate the weight's IOMMU domain */
    size_t osz=(size_t)M*N*2; if(osz<4096)osz=4096;
    struct buf O=orki_bcreate(fd,osz,0x403,w->domain); if(!O.cpu) return -1; memset(O.cpu,0,osz);
    uint16_t *ad=c->Af.cpu; const uint16_t *as=(const uint16_t*)A; for(int j=0;j<M*K;j++) ad[j]=as[j];   /* stage A (contiguous fp16) */
    orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_NONE);                          /* prime fp16 pipeline (keep-warm-aware) */
    uint32_t rc[REGCMD_N];
    int sched=((K&(K-1))==0 && K>=128 && K<2048);                            /* run_stream_f16 rule; small K => 0 */
    orki_synth(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)w->Bb[0].dma,(uint32_t)O.dma,sched,CBUF);
    orki_set_f16_out_fp16in(rc,M,N);                                              /* PROVEN vendor fp16-out stage (default CONTIGUOUS) */
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task*t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0xd; t->int_mask=0x300; t->int_clear=0x1ffff;
      t->regcfg_amount=108; t->regcmd_addr=(uint32_t)c->regcmd.dma; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); }
    struct rknpu_submit sub; memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=1; sub.task_obj_addr=c->task.obj;
    sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1; sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; sub.timeout=orki_mm_timeout_ms();
    int ok=-1;
    for(int rep=0;rep<2;rep++){ if(orki_rknpu_submit_ioctl(fd,&sub,w->domain)){ ok=-1; continue; } orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; }   /* fp16 cold 2-pass re-warm */
    if(ok==0){ uint16_t *od=(uint16_t*)out, *os=(uint16_t*)O.cpu; for(size_t i=0;i<(size_t)M*N;i++) od[i]=os[i]; }   /* CONTIGUOUS fp16 readback */
    orki_bdestroy(fd,&O);
    return ok;
}

int ork_npu_probe_f16_stridedA(ork_npu *c,int M,int K,int N,const uint16_t *A,int apitch,const uint16_t *B,uint16_t *out){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64||(N&7)||apitch<K||(apitch&7)) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N*2,0x403,-1); if(!W.cpu) return -2;
    { int NN=N/16,KT=K/32; uint16_t*bb=W.cpu;
      for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*16+nl)]; }
    size_t asz=(size_t)M*apitch*2; if(asz<4096)asz=4096; size_t osz=(size_t)M*N*2; if(osz<4096)osz=4096;
    struct buf Adev=orki_bcreate(fd,asz,0x403,-1), O=orki_bcreate(fd,osz,0x403,-1);
    if(!Adev.cpu||!O.cpu){ orki_bdestroy(fd,&W);orki_bdestroy(fd,&Adev);orki_bdestroy(fd,&O); return -2; }
    { uint16_t*ad=Adev.cpu; for(size_t i=0;i<asz/2;i++)ad[i]=0xdead;                 /* junk padding between rows */
      for(int m=0;m<M;m++)for(int k=0;k<K;k++) ad[(size_t)m*apitch+k]=A[(size_t)m*K+k]; }  /* A row @ pitch (as the KV-view sits in the DMA buffer) */
    memset(O.cpu,0,osz);
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&Adev,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_NONE);
    uint32_t rc[REGCMD_N];
    int sched=((K&(K-1))==0 && K>=128 && K<2048);
    orki_synth(rc,M,K,N,(uint32_t)Adev.dma,(uint32_t)W.dma,(uint32_t)O.dma,sched,CBUF);     /* activation base = the DMA buffer (ZERO-COPY, no c->Af) */
    orki_setrn(rc,REGCMD_N,RK_CNA_DMA_CON1,apitch/8);                                          /* CNA LINE_STRIDE = apitch/8 surfaces (read the strided view) */
    orki_set_f16_out_fp16in(rc,M,N);
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task*t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0xd; t->int_mask=0x300; t->int_clear=0x1ffff;
      t->regcfg_amount=108; t->regcmd_addr=(uint32_t)c->regcmd.dma; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); }
    uint32_t to_ms=3000; { const char*e=getenv("ORK_EW_TIMEOUT"); if(e){ unsigned v=(unsigned)strtoul(e,0,0); if(v)to_ms=v; } }
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};sub.timeout=to_ms;
    int ok=-1;
    for(int rep=0;rep<2;rep++){ if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; continue; } orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; }
    if(ok==0) for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=((uint16_t*)O.cpu)[(size_t)m*N+n];  /* contiguous fp16 out */
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&Adev);orki_bdestroy(fd,&O);
    return ok;
}

int ork_npu_mm_perchan_f16_fused(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B,
                                 const uint16_t *scale,uint16_t *out){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64||(N&7)) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N*2,0x403,-1); if(!W.cpu) return -2;
    int NN=N/16,KT=K/32; uint16_t*bb=W.cpu;
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*16+nl)];
    size_t osz=(size_t)M*N*2; if(osz<4096)osz=4096;
    struct buf O=orki_bcreate(fd,osz,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;} memset(O.cpu,0,osz);
    struct buf SB=orki_bcreate(fd,4096,0x403,-1); if(!SB.cpu){orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);return -2;} memset(SB.cpu,0,4096);
    { ork_f16*sb=(ork_f16*)SB.cpu; for(int n=0;n<N;n++) sb[n]=*(const ork_f16*)&scale[n]; }   /* per-channel scale, CONTIGUOUS [N] fp16 */
    uint16_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j];
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&SB,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_NONE);                  /* prime fp16 pipeline (layer owns the reset) */
    uint32_t base[REGCMD_N], rc[REGCMD_I8_EW_N];
    int sched=((K&(K-1))==0 && K>=128 && K<2048);
    orki_synth(base,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,sched,CBUF);
    orki_set_f16_out_fp16in(base,M,N);                                   /* main lane: fp16 CONTIGUOUS out */
    orki_splice_ew_lane(rc,base);                                        /* add the 0x50xx EW-operand lane */
    /* EW-mul: multiply the on-chip fp16 accumulator by the per-channel operand. fp16 EW config + per-channel ERDMA. */
    uint32_t ewcfg=getenv("ORK_F16EW_CFG")?strtoul(getenv("ORK_F16EW_CFG"),0,0):0x108003c4;   /* vendor fp16 EW mul */
    uint32_t erdma=getenv("ORK_F16EW_ERDMA")?strtoul(getenv("ORK_F16EW_ERDMA"),0,0):0x00000008;/* per-channel + 2-byte */
    uint32_t r4050=getenv("ORK_F16EW_4050")?strtoul(getenv("ORK_F16EW_4050"),0,0):0x00000127; /* out row cfg + EW-enable bit0 */
    orki_setrn(rc,REGCMD_I8_EW_N,RK_DPU_EW_CFG,ewcfg);
    orki_setrn(rc,REGCMD_I8_EW_N,RK_DPU_EW_CVT_OFFSET,0x00000000);
    orki_setrn(rc,REGCMD_I8_EW_N,RK_DPU_EW_CVT_SCALE,0x00000001);
    orki_setrn(rc,REGCMD_I8_EW_N,RK_DPU_BS_OW_CFG,r4050);
    { const char*e=getenv("ORK_F16EW_4010"); if(e) orki_setrn(rc,REGCMD_I8_EW_N,RK_DPU_OUT_PRECISION,(uint32_t)strtoul(e,0,0)); } /* fp16 DATA_FORMAT + EW-enable bits sweep */
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5004,0x0000000e);
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5008,0x00000001);               /* RDMA_OPERATION_ENABLE */
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_500C,(uint32_t)(M-1));          /* WIDTH=M-1 */
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5010,0x00000000);              /* HEIGHT=1 */
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5014,(uint32_t)(N-1));          /* CHANNEL=N-1 */
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5034,erdma);
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5038,(uint32_t)SB.dma);         /* EW operand = scale[N] */
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5018,(uint32_t)SB.dma);
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_501C,0x00000002);              /* BRDMA_DATA_USE=1 */
    memcpy(c->regcmd.cpu,rc,REGCMD_I8_EW_N*4); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task*tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk);
    tk->enable_mask=0x1d; tk->int_mask=0x300; tk->int_clear=0x1ffff;               /* 0x1d: enable EW/second lane */
    tk->regcfg_amount=REGCMD_I8_EW_N/2; tk->regcmd_addr=(uint32_t)c->regcmd.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    uint32_t to_ms=3000; { const char*e=getenv("ORK_EW_TIMEOUT"); if(e){ unsigned v=(unsigned)strtoul(e,0,0); if(v)to_ms=v; } }
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};sub.timeout=to_ms;
    int ok=-1;
    for(int rep=0;rep<2;rep++){ if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; continue; } orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; }
    if(ok==0) for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=((uint16_t*)O.cpu)[(size_t)m*N+n];  /* CONTIGUOUS */
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);orki_bdestroy(fd,&SB);
    return ok;
}

int ork_npu_ewmul_f16(ork_npu *c,const ork_f16 *up,const ork_f16 *silu,int M,int N,ork_f16 *out,double *us){
    if(c && c->daemon){ if(us)*us=0; return orkd_ewmul_f16(c->daemon,up,silu,M,N,out); }   /* Path B: SDP on the daemon */
    int fd=c->fd, dom=c->dom_active;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;               /* N multiple of the fp16 atom (8) */
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)   /* BYTE offset, fp16 atom=8, surf_stride=M*16 */
    const uint16_t *u16=(const uint16_t*)up,*s16=(const uint16_t*)silu; uint16_t *o16=(uint16_t*)out;
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;                  /* fp16 cube = M*N*2 bytes */
    struct buf A=orki_bcreate(fd,sz,0x403,dom); if(!A.cpu)return -2;
    struct buf B=orki_bcreate(fd,sz,0x403,dom); if(!B.cpu){orki_bdestroy(fd,&A);return -2;}
    struct buf O=orki_bcreate(fd,sz,0x403,dom); if(!O.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);return -2;}
    memset(A.cpu,0,sz);memset(B.cpu,0,sz);memset(O.cpu,0,sz);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int p=EWCUBEH(m,n);
        *(uint16_t*)((char*)A.cpu+p)=u16[m*N+n]; *(uint16_t*)((char*)B.cpu+p)=s16[m*N+n]; }
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (XP_SDP=RC_SDPKW, keep-warm-aware; == the old inline reset) */
    uint32_t rc[REGCMD_MUL_F16_N]; memcpy(rc,REGCMD_MUL_F16,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_MUL_F16_N,M,N);
    orki_setrn(rc,REGCMD_MUL_F16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,REGCMD_MUL_F16_N,RK_SDP_5018,(uint32_t)A.dma);
    orki_setrn(rc,REGCMD_MUL_F16_N,RK_SDP_5038,(uint32_t)B.dma);
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    /* NONBLOCK DOORBELL (spine uniformity + system latency): a blocking submit stalls this thread in-kernel for
     * the whole op (uninterruptible, D-state-wedge-prone; in orkd it also blocks servicing other clients).
     * Submit NONBLOCK with ping-pong OFF, and detect completion by a FULL-SURFACE poll of the fp16 output seeded
     * with an out-of-range poison (inf = 0x7c00, impossible for a bounded ewmul). fp16 write-order isn't
     * last-element-last, so poll the WHOLE cube (the int4/matmul poll technique). No timer — the poison is a real
     * completion signal. (int8 SDP output has no free poison value; those need a trailing witness task instead.) */
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ volatile uint16_t*db=(volatile uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); *db=0x7c00; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); }
    __asm__ volatile("dsb ish":::"memory");
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=(ork_ppflags()&~0x4u)|0x2u;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms(); double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,dom)){ double cap=(double)orki_ew_timeout_ms()*1000.0;
        for(;;){ int done=1; for(int m=0;m<M&&done;m++)for(int n=0;n<N;n++){ volatile uint16_t*db=(volatile uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); __asm__ volatile("dc civac,%0"::"r"(db):"memory"); if(*db==0x7c00u){done=0;break;} } if(done)break; if(ork_now_us()-t0>cap)break; }
        orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) o16[m*N+n]=*(uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(us)*us=t1; }
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&O);
    #undef EWCUBEH
    return ok;
}

int ork_npu_rope_neox_f16(ork_npu *c, const ork_f16 *x, int hd, int nrow, const int *pos, double freq_base, ork_f16 *out){
    if(!c||!x||!pos||!out||hd<2||(hd&7)||nrow<1) return -2;
    int hd2=hd/2; size_t sz=(size_t)nrow*hd*sizeof(ork_f16);
    ork_f16 *cosT=malloc(sz),*sinT=malloc(sz),*xr=malloc(sz),*t1=malloc(sz),*t2=malloc(sz);
    if(!cosT||!sinT||!xr||!t1||!t2){ free(cosT);free(sinT);free(xr);free(t1);free(t2); return -1; }
    for(int r=0;r<nrow;r++){ double p=(double)pos[r];
        for(int i=0;i<hd2;i++){ double th=p*pow(freq_base,-2.0*(double)i/(double)hd); float cc=(float)cos(th), ss=(float)sin(th);
            cosT[(size_t)r*hd+i]=(ork_f16)cc; cosT[(size_t)r*hd+i+hd2]=(ork_f16)cc;
            sinT[(size_t)r*hd+i]=(ork_f16)(-ss); sinT[(size_t)r*hd+i+hd2]=(ork_f16)ss; }
        for(int i=0;i<hd2;i++){ xr[(size_t)r*hd+i]=x[(size_t)r*hd+i+hd2]; xr[(size_t)r*hd+i+hd2]=x[(size_t)r*hd+i]; } }
    int rc=0;
    if(ork_npu_ewmul_f16(c,x,cosT,nrow,hd,t1,NULL)) rc=-1;
    else if(ork_npu_ewmul_f16(c,xr,sinT,nrow,hd,t2,NULL)) rc=-1;
    else if(ork_npu_add_f16(c,t1,t2,nrow,hd,out,NULL)) rc=-1;
    free(cosT);free(sinT);free(xr);free(t1);free(t2);
    return rc;
}

int ork_npu_mm_perchan_f16(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B,
                           const uint16_t *scale,uint16_t *out,double *us){
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(K%32||N<1||N>c->soc->nmax||M<1||M>64) return -2;
    if(N%32==0){                                                    /* fast path: N is a valid tile width */
        uint16_t *G=malloc((size_t)M*N*2); if(!G) return -1;
        int rc=ork_npu_probe_f16_mm_f16out(c,M,K,N,A,B,G);          /* fp16 matmul -> CONTIGUOUS fp16 G (512/512) */
        if(rc==0) rc=ork_npu_mul_perchan_f16(c,(const ork_f16*)G,(const ork_f16*)scale,M,N,(ork_f16*)out,us); /* per-channel scale */
        free(G);
        return rc;
    }
    /* N%32!=0 (e.g. decode: N=queries=1): pad N up to a 32-multiple, zero-pad the B columns + scale vector,
     * compute the padded matmul+scale, then extract the real N columns. The zero-pad columns produce 0 (matmul
     * of a zero weight column) so they don't perturb the real columns; the tiny extra compute is negligible. */
    int Np=((N+31)/32)*32;
    uint16_t *Bp=malloc((size_t)K*Np*2), *scp=malloc((size_t)Np*2), *G=malloc((size_t)M*Np*2), *op=malloc((size_t)M*Np*2);
    if(!Bp||!scp||!G||!op){ free(Bp);free(scp);free(G);free(op); return -1; }
    for(int k=0;k<K;k++){ const uint16_t*br=B+(size_t)k*N; uint16_t*pr=Bp+(size_t)k*Np;
        for(int n=0;n<N;n++)pr[n]=br[n]; for(int n=N;n<Np;n++)pr[n]=0; }
    for(int n=0;n<N;n++)scp[n]=scale[n]; for(int n=N;n<Np;n++)scp[n]=0;
    int rc=ork_npu_probe_f16_mm_f16out(c,M,K,Np,A,Bp,G);
    if(rc==0) rc=ork_npu_mul_perchan_f16(c,(const ork_f16*)G,(const ork_f16*)scp,M,Np,(ork_f16*)op,us);
    if(rc==0) for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=op[(size_t)m*Np+n];
    free(Bp);free(scp);free(G);free(op);
    return rc;
}

int ork_npu_mm_perchan_f16_diag(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B,
                                const uint16_t *scale,uint16_t *out,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64||(N&31)) return -2;    /* N%32 for the diagonal matmul's K=N */
    /* DEVICE-RESIDENT: G stays on the NPU between the two matmuls — zero CPU touch of the intermediate. */
    #define TILE(dst,src,KK,NN) do{ int NT=(NN)/16,KT=(KK)/32; uint16_t*bb=(dst); \
        for(int nt=0;nt<NT;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++) \
          bb[(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk]=(src)[(size_t)(kt*32+kk)*(NN)+(nt*16+nl)]; }while(0)
    size_t gsz=(size_t)M*N*2; if(gsz<4096)gsz=4096;
    struct buf W1=orki_bcreate(fd,(size_t)K*N*2,0x403,-1), G=orki_bcreate(fd,gsz,0x403,-1),
               W2=orki_bcreate(fd,(size_t)N*N*2,0x403,-1), O=orki_bcreate(fd,gsz,0x403,-1);
    if(!W1.cpu||!G.cpu||!W2.cpu||!O.cpu){ orki_bdestroy(fd,&W1);orki_bdestroy(fd,&G);orki_bdestroy(fd,&W2);orki_bdestroy(fd,&O); return -1; }
    TILE(W1.cpu,B,K,N);                                             /* weight1 = B[K][N] */
    { uint16_t*d=W2.cpu; memset(d,0,(size_t)N*N*2); uint16_t*Draw=calloc((size_t)N*N,2);   /* weight2 = diag(scale)[N][N] */
      for(int n=0;n<N;n++) Draw[(size_t)n*N+n]=scale[n]; TILE(W2.cpu,Draw,N,N); free(Draw); }
    memset(G.cpu,0,gsz); memset(O.cpu,0,gsz);
    { uint16_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; }    /* activation1 = A (only host->dev copy of an INPUT) */
    orki_bsync(fd,&W1,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&W2,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&G,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_NONE);                  /* layer owns the reset (keep-warm-aware; no redundant manual reset) */
    uint32_t to_ms=3000; { const char*e=getenv("ORK_EW_TIMEOUT"); if(e){ unsigned v=(unsigned)strtoul(e,0,0); if(v)to_ms=v; } }
    struct rknpu_task*tk=(struct rknpu_task*)c->task.cpu;
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};sub.timeout=to_ms;
    int ok=0; double t0=ork_now_us();
    /* matmul1: A·B -> G (contiguous fp16, device); activation from c->Af. matmul2: G·diag -> O; activation = G (device). */
    struct { int K2,N2; uint32_t aA,aW,aO; } pass[2]={ {K,N,(uint32_t)c->Af.dma,(uint32_t)W1.dma,(uint32_t)G.dma},
                                                       {N,N,(uint32_t)G.dma,(uint32_t)W2.dma,(uint32_t)O.dma} };
    if(getenv("ORK_DIAG_CHAIN")){
        /* SINGLE-SUBMIT: PC-chain matmul1 -> matmul2 (both uniform enable=0xd, like run_chain_i8). G resident. */
        static uint32_t mm1[REGCMD_N], mm2[REGCMD_N];
        int s1=((K&(K-1))==0 && K>=128 && K<2048), s2=((N&(N-1))==0 && N>=128 && N<2048);
        orki_synth(mm1,M,K,N,pass[0].aA,pass[0].aW,pass[0].aO,s1,CBUF); orki_set_f16_out_fp16in(mm1,M,N);
        orki_synth(mm2,M,N,N,pass[1].aA,pass[1].aW,pass[1].aO,s2,CBUF); orki_set_f16_out_fp16in(mm2,M,N);
        ork_chain_prog progs[2]={ {mm1,REGCMD_N,0xd,108,216}, {mm2,REGCMD_N,0xd,108,-1} };
        int crc=ork_npu_chain_progs(c,2,progs,c->dom_active);
        if(!crc){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); if(us)*us=ork_now_us()-t0;
            for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=((uint16_t*)O.cpu)[(size_t)m*N+n]; }
        orki_bdestroy(fd,&W1);orki_bdestroy(fd,&G);orki_bdestroy(fd,&W2);orki_bdestroy(fd,&O);
        #undef TILE
        return crc;
    }
    for(int p=0; p<2 && ok==0; p++){
        uint32_t rc[REGCMD_N];
        int sched=((pass[p].K2&(pass[p].K2-1))==0 && pass[p].K2>=128 && pass[p].K2<2048);
        orki_synth(rc,M,pass[p].K2,pass[p].N2,pass[p].aA,pass[p].aW,pass[p].aO,sched,CBUF);
        orki_set_f16_out_fp16in(rc,M,pass[p].N2);
        memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
        memset(tk,0,sizeof *tk); tk->enable_mask=0xd; tk->int_mask=0x300; tk->int_clear=0x1ffff;
        tk->regcfg_amount=108; tk->regcmd_addr=(uint32_t)c->regcmd.dma; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        int done=-1; for(int rep=0;rep<2;rep++){ if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ done=-1; continue; }
            if(p==0) orki_bsync(fd,&G,RKNPU_MEM_SYNC_FROM_DEVICE|RKNPU_MEM_SYNC_TO_DEVICE); else orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); done=0; }
        if(done) ok=-1;
    }
    if(ok==0){ if(us)*us=ork_now_us()-t0;
        for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=((uint16_t*)O.cpu)[(size_t)m*N+n]; }  /* CONTIGUOUS */
    orki_bdestroy(fd,&W1);orki_bdestroy(fd,&G);orki_bdestroy(fd,&W2);orki_bdestroy(fd,&O);
    #undef TILE
    return ok;
}

int ork_npu_mul_perchan_f16(ork_npu *c,const ork_f16 *a,const ork_f16 *b,int M,int N,ork_f16 *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;
    #define PCH16(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)         /* fp16 cube byte offset, atom=8, surf=M*16 */
    const uint16_t *a16=(const uint16_t*)a,*b16=(const uint16_t*)b; uint16_t *o16=(uint16_t*)out;
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,-1), O=orki_bcreate(fd,sz,0x403,-1), B=orki_bcreate(fd,sz,0x403,-1);
    if(!A.cpu||!O.cpu||!B.cpu){ orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&B); return -2; }
    memset(A.cpu,0,sz); memset(O.cpu,0,sz); memset(B.cpu,0,sz);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) *(uint16_t*)((char*)A.cpu+PCH16(m,n))=a16[(size_t)m*N+n];
    for(int n=0;n<N;n++) ((uint16_t*)B.cpu)[n]=b16[n];               /* per-channel vector CONTIGUOUS [N] (fp16) */
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (XP_SDP=RC_SDPKW, keep-warm-aware; == the old inline reset) */
    uint32_t rc[REGCMD_MUL_F16_N]; memcpy(rc,REGCMD_MUL_F16,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_MUL_F16_N,M,N);
    orki_setrn(rc,REGCMD_MUL_F16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,REGCMD_MUL_F16_N,RK_SDP_5018,(uint32_t)A.dma);
    orki_setrn(rc,REGCMD_MUL_F16_N,RK_SDP_5038,(uint32_t)B.dma);
    orki_setrn(rc,REGCMD_MUL_F16_N,RK_SDP_5034,0x00000008);            /* ERDMA_DATA_MODE=0 (per-channel) + DATA_SIZE=TWO_BYTE */
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub; memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=1;
    sub.task_obj_addr=c->task.obj; sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1;
    sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; sub.timeout=orki_ew_timeout_ms();
    int ok=-1; double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; if(us)*us=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) o16[(size_t)m*N+n]=*(uint16_t*)((char*)O.cpu+PCH16(m,n)); }
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&O); orki_bdestroy(fd,&B);
    #undef PCH16
    return ok;
}

int ork_npu_mul_perchan_f16_contig(ork_npu *c,const ork_f16 *a,const ork_f16 *b,int M,int N,ork_f16 *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>64||N<8||N>8192||(N&7)) return -2;
    const uint16_t *a16=(const uint16_t*)a,*b16=(const uint16_t*)b; uint16_t *o16=(uint16_t*)out;
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,-1), O=orki_bcreate(fd,sz,0x403,-1), B=orki_bcreate(fd,4096,0x403,-1);
    if(!A.cpu||!O.cpu||!B.cpu){ orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&B); return -2; }
    memset(A.cpu,0,sz); memset(O.cpu,0,sz); memset(B.cpu,0,4096);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) ((uint16_t*)A.cpu)[(size_t)m*N+n]=a16[(size_t)m*N+n];  /* CONTIGUOUS in */
    /* PER-ELEMENT operand (vendor DATA_MODE=2): per 8-channel tile t, an [8 chan][M row] block = scale[8t+j]
     * broadcast across rows, channel-major (matches the channel-major output). */
    { int NT0=N/8; uint16_t*bb=B.cpu; int chmaj=getenv("ORK_MULC_OPROW")?0:1;
      ork_f16 one=(ork_f16)1.0f; int ones=getenv("ORK_MULC_ONES")?1:0;
      for(int t=0;t<NT0;t++)for(int j=0;j<8;j++)for(int m=0;m<M;m++)
        bb[(size_t)t*M*8 + (chmaj?(j*M+m):(m*8+j))]=ones?*(uint16_t*)&one:b16[t*8+j]; }
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (XP_SDP=RC_SDPKW, keep-warm-aware; leaves last_dt) */
    /* TILED per the vendor (task13): one SDP task per 8-channel group (CHANNEL=7). Each tile reads its 8
     * contiguous columns of A via NOTCH (LINE skip = N-8 for the row stride) and writes an 8xM atom block to
     * its slot in O. Notch (0x5048/0x504c) is verbatim for N=64; other N derives LINE_NOTCH=N-8. */
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    int NT=N/8; double t0=ork_now_us(); int ok=0;
    for(int t=0; t<NT && ok==0; t++){
        uint32_t rc[REGCMD_MUL_F16_NOTCH_N]; memcpy(rc,REGCMD_MUL_F16_NOTCH,sizeof rc);
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_SDP_5018,(uint32_t)(A.dma+(size_t)t*8*2));   /* input cols 8t..8t+7 */
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_SDP_5038,(uint32_t)(B.dma+(size_t)t*M*8*2)); /* per-element operand block for tile t */
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_DST_BASE_ADDR,(uint32_t)(O.dma+(size_t)t*M*16));  /* tile slot: 8xM atom block */
        /* COMPACT atom-8 output for the 8-channel tile (override the vendor's downstream 0x400 stride) so the
         * readback O+t*M*16 + m*16 + j*2 matches (1 surface, M rows). */
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_DST_SURF_STRIDE,(uint32_t)(M*16));                  /* DST_SURF_STRIDE */
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_DATA_CUBE_WIDTH,(uint32_t)(M-1));
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_DATA_CUBE_HEIGHT,0);
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_DST_N_DIMS,(uint32_t)((7<<16)|7));             /* 8 channels */
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_DST_N2,7);
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_WDMA_SIZE_1,(uint32_t)(M-1));
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_SURFACE_ADD,(uint32_t)(M*16));
        if(getenv("ORK_MULC_HROW")){                                                    /* rows as HEIGHT lines (LINE_NOTCH acts between heights) */
            orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_SDP_500C,0);                             /* RDMA WIDTH = 1 */
            orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_SDP_5010,(uint32_t)(M-1));               /* RDMA HEIGHT = M-1 (rows) */
            orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_DATA_CUBE_WIDTH,0);                             /* DPU WIDTH = 1 */
            orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_DATA_CUBE_HEIGHT,(uint32_t)(M-1));               /* DPU HEIGHT = M-1 */
        } else {
            orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_SDP_500C,(uint32_t)(M-1));               /* WIDTH = M-1 */
        }
        /* vendor task13 is PER-ELEMENT (DATA_MODE=2, compiler broadcast [1,1,N]->[M,N]); switch to PER-CHANNEL
         * (DATA_MODE=0) so an [N] scale operand applies per output channel. */
        uint32_t ewcfg=getenv("ORK_MULC_EW")?strtoul(getenv("ORK_MULC_EW"),0,0):0x20800384; /* vendor: EW mul, DATA_MODE=2 (per-element), fp16 */
        uint32_t erdma=getenv("ORK_MULC_ERDMA")?strtoul(getenv("ORK_MULC_ERDMA"),0,0):0x8000000a; /* vendor: per-element, 2-byte */
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_EW_CFG,ewcfg);
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_SDP_5034,erdma);
        if(N!=64){ orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_SDP_5048,(uint32_t)((N-8)<<19)); } /* SRC_DMA_CFG.LINE_NOTCH_ADDR[31:19]=N-8 (row-stride skip) */
        memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
        tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
        struct rknpu_submit sub; memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=1;
        sub.task_obj_addr=c->task.obj; sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1;
        sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; sub.timeout=orki_ew_timeout_ms();
        if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; break; }
    }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); if(us)*us=ork_now_us()-t0;
        /* read back: tile t's slot O+t*M*16 is CHANNEL-MAJOR (channel j outer, row m inner): element (j*M+m). */
        for(int t=0;t<NT;t++)for(int m=0;m<M;m++)for(int j=0;j<8;j++)
            o16[(size_t)m*N + t*8 + j]=*(uint16_t*)((char*)O.cpu + (size_t)t*M*16 + (size_t)(j*M + m)*2);
        if(getenv("ORK_MULC_RAW")){ uint16_t*o=(uint16_t*)O.cpu; int nz=0,ne=(int)((size_t)NT*M*16/2); for(int i=0;i<ne;i++) if(o[i])nz++;
            fprintf(stderr,"[mulc_raw] nonzero=%d/%d first16:",nz,ne); for(int i=0;i<16;i++) fprintf(stderr," %.3g",(double)*(ork_f16*)&o[i]); fprintf(stderr,"\n"); }
        if(getenv("ORK_MULC_DUMP")){ ork_f16*a=(ork_f16*)A.cpu,*b=(ork_f16*)B.cpu,*o=(ork_f16*)O.cpu;
            fprintf(stderr,"[dump] scale[0..7]:"); for(int j=0;j<8;j++)fprintf(stderr," %g",(double)b[j]); fprintf(stderr,"\n");
            for(int m=0;m<M;m++){ fprintf(stderr,"[dump] A[%d][0..7]:",m); for(int j=0;j<8;j++)fprintf(stderr," %g",(double)a[(size_t)m*N+j]); fprintf(stderr,"\n"); }
            fprintf(stderr,"[dump] O tile0 [0..%d]:",M*8-1); for(int i=0;i<M*8;i++)fprintf(stderr," %g",(double)o[i]); fprintf(stderr,"\n"); }
    }
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&O); orki_bdestroy(fd,&B);
    return ok;
}

static void *ork_rsh_patcher(void *p){ struct ork_rsh_patch *a=p; usleep(a->delay_us);
    if(a->rcmode){ a->rcword[0]=a->rcval0; a->rcword[1]=a->rcval1; orki_bsync(a->c->fd,&a->c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE); }
    else { struct rknpu_task *tk=(struct rknpu_task*)a->c->task.cpu; tk[a->idx]=a->marker; orki_bsync(a->c->fd,&a->c->task,RKNPU_MEM_SYNC_TO_DEVICE); }
    __asm__ __volatile__("dsb sy":::"memory"); return NULL; }

int ork_npu_replay_reshape_f16(ork_npu *c,uint16_t *gemm_raw,int gemm_words,uint16_t *reshape_raw,int reshape_words,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    const uint32_t IB=0xfffed000u; const size_t ISZ=0x13000;   /* vendor image span */
    /* task0..21: {imgoff, enable, regcfg_amount} — the FULL first forward pass from the capture */
    static const struct { uint32_t off; uint32_t en; uint32_t amt; } TK[22]={
        {0xd8c0,0x18,69},{0xdb40,0xd,108},{0xdec0,0xd,108},{0xe240,0xd,108},{0xe5c0,0xd,108},
        {0xe940,0xd,13},{0xea00,0xd,12},{0xea80,0xd,12},{0xeb00,0xd,12},{0xeb80,0xd,12},{0xec00,0xd,12},
        {0xec80,0x18,69},{0xef00,0x18,69},{0xf180,0x18,69},{0xf400,0x18,25},{0xf500,0x18,9},{0xf580,0x18,9},
        {0xf600,0x18,9},{0xf680,0x18,9},{0xf700,0x18,9},{0xf780,0x18,9},{0xf800,0x18,69}};
    int T0=0; { const char*e=getenv("ORK_RESHAPE_T0"); if(e){int v=atoi(e); if(v>=0&&v<22)T0=v;} } /* start task (skip SDP convert) */
    int NT=22-T0; { const char*e=getenv("ORK_RESHAPE_NT"); if(e){int v=atoi(e); if(v>=1&&v<=22-T0)NT=v;} }
    #define TK(j) TK[(T0)+(j)]
    const char *path=getenv("ORK_RESHAPE_IMG"); if(!path)path="gemm_mul_image.bin";
    FILE *f=fopen(path,"rb"); if(!f) return -2;
    struct buf BIG=orki_bcreate(fd,ISZ,0x403,-1); if(!BIG.cpu){fclose(f);return -2;}
    memset(BIG.cpu,0,ISZ);
    size_t rd=fread(BIG.cpu,1,ISZ,f); fclose(f); if(rd<ISZ){ orki_bdestroy(fd,&BIG); return -2; }
    /* inject DISTINCT input (@0xfffef000, imgoff 0x2000) so gemm_out is non-degenerate -> derivable perm */
    if(getenv("ORK_RESHAPE_INJECT")){ uint16_t*x=(uint16_t*)((char*)BIG.cpu+0x2000);      /* task0 input @0xfffef000 */
        for(int i=0;i<512;i++){ float v=(float)((i%37)-8)*0.25f; __fp16 h=(__fp16)v; memcpy(&x[i],&h,2); } }
    int ginj=getenv("ORK_RESHAPE_GINJ")!=0;
    if(ginj){ uint16_t*g=(uint16_t*)((char*)BIG.cpu+0x3000);          /* GEMM output @0xffff0000: DISTINCT so reshape perm is derivable (start at task2, T0=2) */
        for(int i=0;i<512;i++){ __fp16 h=(__fp16)(float)(i+1); memcpy(&g[i],&h,2); }
        memset((char*)BIG.cpu+0x3680,0,0x400); }                      /* zero ONLY reshape-out so fresh writes show */
    ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_NONE);   /* warm/reset BEFORE we write c->regcmd (enter may use it) */
    uint32_t delta=(uint32_t)BIG.dma - IB;   /* DATA addresses -> BIG */
    /* #2 fix: put regcmds in c->regcmd (the path that EXECUTES), DATA stays in BIG. Compute each task's
     * c->regcmd word offset; copy its regcmd out of the image; rebase DATA addrs -> BIG, and point the chain
     * 0x0010 -> the NEXT task's c->regcmd offset (NOT BIG). (ORK_RESHAPE_INIMG = old in-image path for A/B.) */
    int inimg=getenv("ORK_RESHAPE_INIMG")!=0;
    /* ★ TERMINAL COMPLETABILITY: a 12-reg reshape DELTA cannot be the chain's last task (only 108-reg or the
     * 13-reg first-delta task5 raise DONE as terminus — task5 has an extra 0x1040=0x201b write the 12-reg
     * deltas lack). In the vendor graph the deltas are never last. So promote the terminal 12-reg delta to
     * task5's 13-reg form, patched to the terminal group's own in/out addresses. (ORK_RESHAPE_NOTERM = off.) */
    int termfix = !getenv("ORK_RESHAPE_NOTERM") && !inimg;
    uint32_t *cr=(uint32_t*)c->regcmd.cpu; uint32_t cro[22]; uint32_t eamt[22]; int prom[22];
    { uint32_t cur=0; for(int j=0;j<NT;j++){ prom[j]= termfix && (int)TK(j).amt==12; /* every 12-reg delta -> 13-reg task5 form */
        eamt[j]=prom[j]?13u:(uint32_t)TK(j).amt; cro[j]=cur; cur+=eamt[j]*2+16; } }
    for(int j=0;j<NT;j++){
        int last=(j==NT-1);
        uint32_t srcoff = prom[j] ? TK[5].off : TK(j).off;   /* TK[5] = the completable 13-reg delta */
        int nw=(int)eamt[j]*2+16;   /* regcfg entries*2 + trailer */
        uint32_t pin=0,pout=0;   /* this delta's OWN captured in/out (patch task5-form to it) */
        if(prom[j]){ uint32_t*orc=(uint32_t*)((char*)BIG.cpu+TK(j).off); int onw=(int)TK(j).amt*2+16;
            for(int k=0;k+1<onw;k+=2){ unsigned a=orc[k]&0xffff,d=orc[k+1]>>16; uint32_t v=((orc[k+1]&0xffff)<<16)|(orc[k]>>16);
                if(a==0x1070&&d==0x0201)pin=v; if(a==0x4020&&d==0x1001)pout=v; } }
        uint32_t *rc = inimg ? (uint32_t*)((char*)BIG.cpu+TK(j).off) : (cr+cro[j]);
        if(!inimg) memcpy(rc,(char*)BIG.cpu+srcoff,(size_t)nw*4);
        for(int k=0;k+1<nw;k+=2){ unsigned a=rc[k]&0xffff,d=rc[k+1]>>16; uint32_t v=((rc[k+1]&0xffff)<<16)|(rc[k]>>16);
            if(d==0x0101 && a==0x0010){ /* chain: next-regcmd addr */
                uint32_t nx = (inimg? (uint32_t)BIG.dma+TK(j+1<NT?j+1:j).off : (uint32_t)c->regcmd.dma+cro[j+1<NT?j+1:j]*4);
                if(!last){ rc[k]=0x0010|((nx&0xffff)<<16); rc[k+1]=(0x0101u<<16)|((nx>>16)&0xffff); }
                else if(getenv("ORK_RESHAPE_SELFLOOP")){ uint32_t sp=(uint32_t)c->regcmd.dma+cro[j]*4; /* self-loop: 0x0010 -> own regcmd (free-run probe) */
                    rc[k]=0x0010|((sp&0xffff)<<16); rc[k+1]=(0x0101u<<16)|((sp>>16)&0xffff); }
                else { rc[k]=0; rc[k+1]=0; } }                             /* terminate last */
            else if(d==0x0101 && a==0x0014){ if(last&&getenv("ORK_RESHAPE_SELFLOOP")){ uint32_t na=(eamt[j]+3)/2; rc[k]=0x0014|((na&0xffff)<<16); rc[k+1]=(0x0101u<<16)|((na>>16)&0xffff); } /* self-loop: next-amount = own */
                else if(last){ rc[k]=0; rc[k+1]=0; } /* null on last */
                else { uint32_t na=(eamt[j+1]+3)/2; rc[k]=0x0014|((na&0xffff)<<16); rc[k+1]=(0x0101u<<16)|((na>>16)&0xffff); } } /* next-amount = (eff_amt[j+1]+3)/2 (handles promotion) */
            else if(prom[j] && a==0x1070 && d==0x0201){ uint32_t nv=pin+delta;  rc[k]=a|((nv&0xffff)<<16); rc[k+1]=(d<<16)|((nv>>16)&0xffff); } /* patch task5-form to THIS group */
            else if(prom[j] && a==0x4020 && d==0x1001){ uint32_t nv=pout+delta; rc[k]=a|((nv&0xffff)<<16); rc[k+1]=(d<<16)|((nv>>16)&0xffff); }
            else if(v>=IB && v<IB+ISZ){ uint32_t nv=v+delta; rc[k]=a|((nv&0xffff)<<16); rc[k+1]=(d<<16)|((nv>>16)&0xffff); } } /* DATA -> BIG */
    }
    /* ZERO gemm-out (0x3000) + reshape-out (0x3a00) so FRESH computation is distinguishable from baked image data. */
    if(getenv("ORK_RESHAPE_ZERO")){ memset((char*)BIG.cpu+0x3000,0,0x400); memset((char*)BIG.cpu+0x3680,0,0x400); }
    orki_bsync(fd,&BIG,RKNPU_MEM_SYNC_TO_DEVICE);
    if(!inimg) orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t tn=(uint32_t)NT; { const char*e=getenv("ORK_RESHAPE_TN"); if(e){unsigned v=(unsigned)strtoul(e,0,0); if(v>=1)tn=v;} } /* TREADMILL = #descriptors (task_number must match; a big RING beats the prefetcher) */
    if(tn>13000)tn=13000;                                          /* c->task=512KB / ~40B per rknpu_task */
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof(*tk)*tn);
    for(int j=0;j<NT;j++){ tk[j].enable_mask=TK(j).en; tk[j].int_mask=0x300; tk[j].int_clear=0x1ffff;
        tk[j].regcfg_amount=eamt[j]; tk[j].regcmd_addr = inimg ? (uint32_t)BIG.dma+TK(j).off : (uint32_t)c->regcmd.dma+cro[j]*4; }
    for(uint32_t j=(uint32_t)NT;j<tn;j++) tk[j]=tk[NT-1];          /* RING: replicate the (completable) last task to fill the treadmill */
    /* Wall-#2 live-patch test: whole ring = cheap dummy (does NOT write 0x3000); MARKER = the GEMM (DOES write
     * 0x3000). A bg thread hot-patches descriptor[tn/2] (far ahead) to the marker mid-submit. If 0x3000 ends up
     * nonzero, the NPU re-read the patched descriptor from DRAM when it arrived -> live far-ahead patch HONORED. */
    pthread_t pth; struct ork_rsh_patch parg; int patching=0;
    if(getenv("ORK_RESHAPE_PATCH") && tn>(uint32_t)NT){
        parg.c=c; parg.marker=tk[0];                              /* GEMM marker (writes 0x3000) */
        for(uint32_t j=0;j<tn;j++) tk[j]=tk[NT-1];                /* ring = cheap reshape-delta dummy */
        parg.idx=tn/2; { const char*e=getenv("ORK_RESHAPE_PATCH_US"); parg.delay_us=e?(uint32_t)strtoul(e,0,0):50; }
        parg.rcmode=0;
        if(getenv("ORK_RESHAPE_PATCHRC")){ /* alt: patch the shared DUMMY regcmd's 0x4020 output -> 0x3000, mid-orki_run (tests regcmd live re-read) */
            uint32_t *drc=cr+cro[NT-1]; int dnw=(int)eamt[NT-1]*2+16; parg.rcword=NULL;
            for(int k=0;k+1<dnw;k+=2){ if((drc[k]&0xffff)==0x4020 && (drc[k+1]>>16)==0x1001){ parg.rcword=drc+k;
                uint32_t nv=(uint32_t)BIG.dma+0x3000; parg.rcval0=0x4020|((nv&0xffff)<<16); parg.rcval1=(0x1001u<<16)|((nv>>16)&0xffff); break; } }
            if(parg.rcword) parg.rcmode=1; }
        memset((char*)BIG.cpu+0x3000,0,0x400); orki_bsync(fd,&BIG,RKNPU_MEM_SYNC_TO_DEVICE); patching=1;
    }
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub; memset(&sub,0,sizeof sub);
    uint32_t flg=0x5; { const char*e=getenv("ORK_RESHAPE_FLAGS"); if(e)flg=(uint32_t)strtoul(e,0,0); }  /* vendor used 0x5 */
    sub.flags=flg; sub.task_number=tn; sub.task_obj_addr=c->task.obj; sub.core_mask=RKNPU_CORE0_MASK;
    sub.fence_fd=-1; sub.timeout=orki_ew_timeout_ms();
    sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,tn}; /* vendor sets all 3 */
    if(patching) pthread_create(&pth,NULL,ork_rsh_patcher,&parg);   /* fire the mid-submit far-ahead descriptor patch */
    int ok=-1; double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bsync(fd,&BIG,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; if(us)*us=ork_now_us()-t0; }
    else orki_bsync(fd,&BIG,RKNPU_MEM_SYNC_FROM_DEVICE);
    if(patching){ pthread_join(pth,NULL); int mnz=0; uint16_t*mg=(uint16_t*)((char*)BIG.cpu+0x3000); for(int i=0;i<512;i++)if(mg[i])mnz++;
        fprintf(stderr,"[patch] ring=%u idx=%u delay=%uus -> gemm-marker(0x3000) nz=%d -> %s\n",
            tn,parg.idx,parg.delay_us,mnz, mnz?"WRITTEN (live far-ahead patch HONORED)":"zero (patch NOT honored / prefetched)"); }
    if(gemm_raw){ uint16_t*g=(uint16_t*)((char*)BIG.cpu+0x3000); for(int i=0;i<gemm_words;i++)gemm_raw[i]=g[i]; }       /* 0xffff0000 */
    uint32_t roff=0x3680; { const char*e=getenv("ORK_RESHAPE_ROUT"); if(e)roff=(uint32_t)strtoul(e,0,0); } /* reshape-out read offset (0x3680 atom-8 base; 0x3280=task2 out) */
    if(reshape_raw){ uint16_t*r=(uint16_t*)((char*)BIG.cpu+roff); for(int i=0;i<reshape_words;i++)reshape_raw[i]=r[i]; }
    orki_bdestroy(fd,&BIG);
    #undef TK
    return ok;
}

int ork_npu_reshape_probe_f16(ork_npu *c,int M,int N,const uint16_t *src,uint16_t *out_raw,int out_words,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(N!=64||M<1||M>8) return -2;   /* WIP: captured pattern/geometry is M=8,N=64 */
    static const int WPOS[64]={0,65,136,201,272,337,408,473,544,609,680,745,816,881,952,1017,1026,1091,1162,
        1227,1298,1363,1434,1499,1570,1635,1706,1771,1842,1907,1978,2043,2052,2117,2188,2253,2324,2389,2460,
        2525,2596,2661,2732,2797,2868,2933,3004,3069,3078,3143,3214,3279,3350,3415,3486,3551,3622,3687,3758,
        3823,3894,3959,4030,4095};
    size_t isz=(size_t)M*N*2; if(isz<4096)isz=4096;
    size_t osz=(size_t)M*N*2*2; if(osz<8192)osz=8192;   /* generous output room */
    struct buf In=orki_bcreate(fd,isz,0x403,-1), W=orki_bcreate(fd,8192,0x403,-1), O=orki_bcreate(fd,osz,0x403,-1);
    if(!In.cpu||!W.cpu||!O.cpu){ orki_bdestroy(fd,&In);orki_bdestroy(fd,&W);orki_bdestroy(fd,&O); return -2; }
    memset(In.cpu,0,isz); memset(W.cpu,0,8192); memset(O.cpu,0,osz);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) ((uint16_t*)In.cpu)[(size_t)m*N+n]=src[(size_t)m*N+n];
    for(int i=0;i<64;i++) ((uint16_t*)W.cpu)[WPOS[i]]=0x3c00;   /* fp16 1.0 permutation (channel reorder) */
    orki_bsync(fd,&In,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_NONE);
    uint32_t rc[REGCMD_RESHAPE_F16_N]; memcpy(rc,REGCMD_RESHAPE_F16,sizeof rc);
    orki_setrn(rc,REGCMD_RESHAPE_F16_N,RK_CNA_FEATURE_DATA_ADDR,(uint32_t)In.dma);    /* input base (CNA activation) */
    orki_setrn(rc,REGCMD_RESHAPE_F16_N,RK_CNA_WEIGHT_DATA_ADDR,(uint32_t)W.dma);     /* weight base (permutation) */
    orki_setrn(rc,REGCMD_RESHAPE_F16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);    /* output base (DPU) */
    { const char*e;   /* RE: reconcile the reshape read geometry to OUR contiguous [M][N] input pitch */
      if((e=getenv("ORK_RSH_107C"))) orki_setrn(rc,REGCMD_RESHAPE_F16_N,RK_CNA_DMA_CON1,(uint32_t)strtoul(e,0,0));
      if((e=getenv("ORK_RSH_1080"))) orki_setrn(rc,REGCMD_RESHAPE_F16_N,RK_CNA_DMA_CON2,(uint32_t)strtoul(e,0,0)); }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task*t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0xd; t->int_mask=0x300; t->int_clear=0x1ffff;
      t->regcfg_amount=108; t->regcmd_addr=(uint32_t)c->regcmd.dma; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); }
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};sub.timeout=orki_ew_timeout_ms();
    int ok=-1; double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; if(us)*us=ork_now_us()-t0; }
    else { orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); }   /* coherent buffer: read partial write on fail too */
    if(out_raw){ int w=(int)(osz/2); if(w>out_words)w=out_words; for(int i=0;i<w;i++) out_raw[i]=((uint16_t*)O.cpu)[i]; }
    orki_bdestroy(fd,&In);orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);
    return ok;
}

int ork_npu_add_f16(ork_npu *c,const ork_f16 *a,const ork_f16 *b,int M,int N,ork_f16 *out,double *us){
    if(c && c->daemon){ if(us)*us=0; return orkd_add_f16(c->daemon,a,b,M,N,out); }   /* Path B: SDP on the daemon */
    int fd=c->fd, dom=c->dom_active;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)
    const uint16_t *a16=(const uint16_t*)a,*b16=(const uint16_t*)b; uint16_t *o16=(uint16_t*)out;
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    if(orki_ppu_scratch3(c,sz)) return -2;                          /* reuse persistent scratch (no per-op churn) */
    struct buf A=c->ppu_a, B=c->ppu_b, O=c->ppu_o;
    memset(A.cpu,0,sz);memset(B.cpu,0,sz);memset(O.cpu,0,sz);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int p=EWCUBEH(m,n);
        *(uint16_t*)((char*)A.cpu+p)=a16[m*N+n]; *(uint16_t*)((char*)B.cpu+p)=b16[m*N+n]; }
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (XP_SDP=RC_SDPKW, keep-warm-aware; == the old inline reset) */
    uint32_t rc[REGCMD_ADD_F16_N]; memcpy(rc,REGCMD_ADD_F16,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_ADD_F16_N,M,N);
    orki_setrn(rc,REGCMD_ADD_F16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,REGCMD_ADD_F16_N,RK_SDP_5018,(uint32_t)A.dma);
    orki_setrn(rc,REGCMD_ADD_F16_N,RK_SDP_5038,(uint32_t)B.dma);
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    /* NONBLOCK DOORBELL (spine uniformity): ping-pong OFF + full-surface fp16 inf-poison poll — see ork_npu_ewmul_f16. */
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ volatile uint16_t*db=(volatile uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); *db=0x7c00; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); }
    __asm__ volatile("dsb ish":::"memory");
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=(ork_ppflags()&~0x4u)|0x2u;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms(); double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,dom)){ double cap=(double)orki_ew_timeout_ms()*1000.0;
        for(;;){ int done=1; for(int m=0;m<M&&done;m++)for(int n=0;n<N;n++){ volatile uint16_t*db=(volatile uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); __asm__ volatile("dc civac,%0"::"r"(db):"memory"); if(*db==0x7c00u){done=0;break;} } if(done)break; if(ork_now_us()-t0>cap)break; }
        orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) o16[m*N+n]=*(uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(us)*us=t1; }
    /* scratch persists (ppu_scratch3) — not destroyed */
    #undef EWCUBEH
    return ok;
}

int ork_npu_probe_silu_std_f16(ork_npu *c,const ork_f16 *in,int M,int N,
                               uint32_t idx_off,uint32_t cfg4064,uint32_t cfg4068,
                               const int16_t *lut,int nlut,ork_f16 *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)   /* fp16 atom-8, 2-byte, surf_stride=M*16 */
    const uint16_t *i16=(const uint16_t*)in; uint16_t *o16=(uint16_t*)out;
    /* #35 FIX: allocate this op's buffers + submit in the CURRENTLY-ACTIVE domain, NOT a hardcoded dom0.
     * A standalone SDP LUT-op runs right after a matmul that may have orki_dom_activate()'d a NON-0 domain
     * (multi-domain FFN chain); if the buffers live in dom0 and it submits iommu_domain_id=0 while
     * c->dom_active is that other domain, the submit WEDGES (errno 110) — REPRODUCED in isolation by
     * i16_shape_probe [G] (dom0 matmul->silu clean; dom1 matmul->silu wedges). Match the active domain. */
    int dom = getenv("ORK_I16_DOM0") ? 0 : c->dom_active;   /* ORK_I16_DOM0: A/B — force dom0 (disable the domain fix) */
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,dom); if(!A.cpu)return -2;
    struct buf O=orki_bcreate(fd,sz,0x403,dom); if(!O.cpu){orki_bdestroy(fd,&A);return -2;}
    struct buf Lrc=orki_bcreate(fd,(size_t)REGCMD_SILU_LUT_N*4,0x403,dom); if(!Lrc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);return -2;}
    struct buf Lsc=orki_bcreate(fd,4096,0x403,dom); if(!Lsc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);return -2;}
    memset(A.cpu,0,sz);memset(O.cpu,0,sz);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) *(uint16_t*)((char*)A.cpu+EWCUBEH(m,n))=i16[m*N+n];
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);

    /* submit 1: LUT-load */
    memcpy(Lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
    orki_setrn((uint32_t*)Lrc.cpu,REGCMD_SILU_LUT_N,RK_DPU_DST_BASE_ADDR,(uint32_t)Lsc.dma);
    if(lut){ uint32_t*lr=(uint32_t*)Lrc.cpu; int j=0;
        for(int k=0;k+1<REGCMD_SILU_LUT_N;k+=2){ if((lr[k]&0xffff)==0x4104){ int32_t v=(j<nlut)?(int32_t)lut[j]:0; j++;
            lr[k]=0x4104|((uint32_t)(v&0xffff)<<16); lr[k+1]=(0x1001u<<16)|(((uint32_t)v>>16)&0xffff); } } }
    orki_bsync(fd,&Lrc,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
      t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=1097; t->regcmd_addr=Lrc.dma;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      /* ping-pong OFF (0x1, NOT ork_ppflags's 0x5) for the LUT-load: ping-pong swaps register banks the
       * instant the task's config completes, racing the LUT's SRAM-commit side effect. Standalone there's
       * nothing to race, but IN A CHAIN (after a preceding matmul) the race soft-resets the NPU (#35 int16
       * silu in-chain wedge). Matches ork_mm_run_i8_silu's LUT-load + AGENTS.md "ping-pong OFF for LUT chains". */
      struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x1;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.timeout=orki_ew_timeout_ms();sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc); return -1; }
    }

    /* submit 2: standalone fp16 activation op */
    uint32_t rc[REGCMD_SILU_STD_F16_N]; memcpy(rc,REGCMD_SILU_STD_F16,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_SILU_STD_F16_N,M,N);
    orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_SDP_5040,0);
    orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_SDP_5038,0);
    orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_SDP_5018,(uint32_t)A.dma);
    orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_DPU_R4110,idx_off);
    orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_DPU_BN_ALU_CFG,cfg4064);
    orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_DPU_BN_MUL_CFG,cfg4068);
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk);
    tk->enable_mask=0x18; tk->int_mask=0x300; tk->int_clear=0x1ffff; tk->regcfg_amount=69; tk->regcmd_addr=c->regcmd.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms(); double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,dom)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) o16[m*N+n]=*(uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(us)*us=t1; }
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);
    #undef EWCUBEH
    return ok;
}

int ork_npu_replay_full_f16(ork_npu *c,const uint32_t *loader,int ln,const ork_f16 *in,int M,int N,ork_f16 *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)||ln<2200||ln>2300) return -2;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)
    const uint16_t *i16=(const uint16_t*)in; uint16_t *o16=(uint16_t*)out;
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,-1); if(!A.cpu)return -2;              /* orig x */
    struct buf S=orki_bcreate(fd,sz,0x403,-1); if(!S.cpu){orki_bdestroy(fd,&A);return -2;} /* stage-1 sigmoid intermediate */
    struct buf O=orki_bcreate(fd,sz,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&S);return -2;}
    struct buf Lrc=orki_bcreate(fd,(size_t)ln*4,0x403,-1); if(!Lrc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&S);orki_bdestroy(fd,&O);return -2;}
    struct buf Lsc=orki_bcreate(fd,4096,0x403,-1); if(!Lsc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&S);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);return -2;}
    memset(A.cpu,0,sz);memset(S.cpu,0,sz);memset(O.cpu,0,sz);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) *(uint16_t*)((char*)A.cpu+EWCUBEH(m,n))=i16[m*N+n];
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&S,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    #define SUBMIT1(REG,RN,RA) do{ memcpy(c->regcmd.cpu,(REG),(size_t)(RN)*4); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE); \
        struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk); \
        tk->enable_mask=0x18; tk->int_mask=0x300; tk->int_clear=0x1ffff; tk->regcfg_amount=(RA); tk->regcmd_addr=c->regcmd.dma; \
        orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); \
        struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.timeout=orki_ew_timeout_ms();sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; \
        if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bdestroy(fd,&A);orki_bdestroy(fd,&S);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc); return -1; } }while(0)
    /* submit 1: fp16 LUT-load (verbatim; patch only the scratch out addr) — uses Lrc not c->regcmd (2210 words) */
    memcpy(Lrc.cpu,loader,(size_t)ln*4);
    orki_setrn((uint32_t*)Lrc.cpu,ln,RK_DPU_DST_BASE_ADDR,(uint32_t)Lsc.dma);
    orki_bsync(fd,&Lrc,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
      t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=1097; t->regcmd_addr=Lrc.dma;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.timeout=orki_ew_timeout_ms();sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bdestroy(fd,&A);orki_bdestroy(fd,&S);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc); return -1; }
    }
    /* submit 2: stage-1 (REGCMD_SILU_STD_F16, sigmoid via LE-LUT) VERBATIM, x@A -> sigmoid@S */
    { uint32_t rc[REGCMD_SILU_STD_F16_N]; memcpy(rc,REGCMD_SILU_STD_F16,sizeof rc);
      orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)S.dma);
      orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_SDP_5018,(uint32_t)A.dma);
      SUBMIT1(rc,REGCMD_SILU_STD_F16_N,69); }
    /* submit 3: stage-2 (REGCMD_SILU_F16_T2, x*sigmoid) VERBATIM, sigmoid@S (0x5018) * x@A (0x5038) -> O */
    { uint32_t rc[REGCMD_SILU_F16_T2_N]; memcpy(rc,REGCMD_SILU_F16_T2,sizeof rc);
      orki_setrn(rc,REGCMD_SILU_F16_T2_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
      orki_setrn(rc,REGCMD_SILU_F16_T2_N,RK_SDP_5018,(uint32_t)S.dma);
      orki_setrn(rc,REGCMD_SILU_F16_T2_N,RK_SDP_5038,(uint32_t)A.dma);
      SUBMIT1(rc,REGCMD_SILU_F16_T2_N,69); }
    orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) o16[m*N+n]=*(uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(us)*us=0;
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&S);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);
    #undef SUBMIT1
    #undef EWCUBEH
    return 0;
}

int ork_npu_replay_softmax_f16(ork_npu *c, const void *in, void *out, double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    struct buf IN=orki_bcreate(fd,32768,0x403,-1), OUT=orki_bcreate(fd,32768,0x403,-1),
               SCR=orki_bcreate(fd,237568,0x403,-1), LUT=orki_bcreate(fd,65536,0x403,-1),
               WT=orki_bcreate(fd,(size_t)(SM_WT_WORDS+64)*4,0x403,-1);
    if(!IN.cpu||!OUT.cpu||!SCR.cpu||!LUT.cpu||!WT.cpu){
        orki_bdestroy(fd,&IN);orki_bdestroy(fd,&OUT);orki_bdestroy(fd,&SCR);orki_bdestroy(fd,&LUT);orki_bdestroy(fd,&WT); return -2; }
    memcpy(IN.cpu,in,32768); memset(OUT.cpu,0,32768); memset(SCR.cpu,0,237568); memset(LUT.cpu,0,65536);
    memcpy(WT.cpu,SM_WT,(size_t)SM_WT_WORDS*4);
    orki_bsync(fd,&IN,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&OUT,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&SCR,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&LUT,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&WT,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    /* address-rebase helper: patch a task's regcmd in place (addr regs -> our buffers; chain -> `nextdma`
     * or 0 to terminate). Returns nothing; operates on rc[0..nw). */
    #define SM_REBASE(rc,nw,nextdma,nextamt) do{ for(int k=0;k+1<(nw);k+=2){ \
        unsigned _a=(rc)[k]&0xffff, _d=(rc)[k+1]>>16; uint32_t _v=(((rc)[k+1]&0xffff)<<16)|((rc)[k]>>16); \
        int _ia=(_d==0x1001&&_a==0x4020)||(_d==0x2001&&(_a==0x5018||_a==0x5038))||(_d==0x0201&&(_a==0x1070||_a==0x1110)); \
        if(_ia){ uint32_t _nv=_v; \
            if      (_v>=0xfffb6000u&&_v<0xfffbe000u) _nv=(uint32_t)IN.dma +(_v-0xfffb6000u); \
            else if (_v>=0xfffae000u&&_v<0xfffb6000u) _nv=(uint32_t)OUT.dma+(_v-0xfffae000u); \
            else if (_v>=0xfffbe000u&&_v<0xffff8000u) _nv=(uint32_t)SCR.dma+(_v-0xfffbe000u); \
            else if (_v>=0xffffa280u&&_v<0xffffab00u) _nv=(uint32_t)WT.dma +(_v-0xffffa280u); \
            else if (_v>=0xffffab00u&&_v<0xfffff000u) _nv=(uint32_t)LUT.dma+(_v-0xffffab00u); \
            if(_nv!=_v){ (rc)[k]=_a|((_nv&0xffff)<<16); (rc)[k+1]=(_d<<16)|((_nv>>16)&0xffff); } } \
        /* PC-chain descriptor: 0x0010 = next task's regcmd addr, 0x0014 = next task's register-AMOUNT,
         * RECOMPUTED as (nextamt+3)/2 for the ACTUAL next task (verified: amt 69->0x24, 1097->0x226; ==
         * the kernel's (amt+EXTRA+scale-1)/scale-1 with EXTRA=4,scale=2). MUST be recomputed, not preserved:
         * the captured value encodes the CAPTURE-order next task, which is wrong for a reordered chain.
         * nextdma==0 (last/no-chain) -> zero both to terminate. cf. run_chain_i8 (0x0014=0x0037 for its 108s). */ \
        if(_d==0x0101&&_a==0x0010){ if(nextdma){ uint32_t _nx=(uint32_t)(nextdma); (rc)[k]=0x0010|((_nx&0xffff)<<16); (rc)[k+1]=(0x0101u<<16)|((_nx>>16)&0xffff); } else { (rc)[k]=0; (rc)[k+1]=0; } } \
        if(_d==0x0101&&_a==0x0014){ if(nextdma){ uint32_t _na=(uint32_t)(((nextamt)+3)/2); (rc)[k]=0x0014|((_na&0xffff)<<16); (rc)[k+1]=(0x0101u<<16)|((_na>>16)&0xffff); } else { (rc)[k]=0; (rc)[k+1]=0; } } } }while(0)
    /* ORK_SM_FULLIMG: FAITHFUL single-delta full-image hardware-chain replay. The vendor's 5 buffers are ONE
     * contiguous IOVA image (0xfffae000 out, 0xfffb6000 in, 0xfffbe000 scratch, 0xffff8000 regcmd+weights),
     * and task4's LUT-write (0x4020=0xffffab00) points INTO the regcmd region — a relationship my split
     * IN/OUT/SCR/LUT/regcmd buffers destroy. Here: ONE big buffer covering the image, every task's regcmd at
     * its captured offset, and EVERY address blanket-rebased by a SINGLE delta (BIG.dma - 0xfffae000) so all
     * internal references (chain 0x0010, data addrs, weights, task4->regcmd write) stay consistent. 0x0014
     * (amount, small) isn't in the addr range so it's preserved. Tasks laid out at the vendor's exact offsets
     * (capture order), task_number=9, one submit — the maximally-faithful hardware chain. */
    if(!getenv("ORK_SM_PERTASK")){   /* DEFAULT = single-submit hardware chain (contiguous image, ping-pong off) */
        double t0=ork_now_us();
        const uint32_t IB=0xfffae000u, IE=0xfffff000u; size_t ISZ=IE-IB;   /* out|in|scratch|regcmd+wt */
        static const uint32_t TADDR[9]={0xffffab00u,0xffffad80u,0xffffb000u,0xffffb280u,0xffffb380u,0xffffd600u,0xffffd880u,0xffffdc00u,0xffffde40u};
        struct buf BIG=orki_bcreate(fd,ISZ,0x403,-1);
        if(!BIG.cpu){ orki_bdestroy(fd,&IN);orki_bdestroy(fd,&OUT);orki_bdestroy(fd,&SCR);orki_bdestroy(fd,&LUT);orki_bdestroy(fd,&WT); return -2; }
        memset(BIG.cpu,0,ISZ);
        uint32_t delta=(uint32_t)BIG.dma - IB;
        memcpy((char*)BIG.cpu + (0xfffb6000u-IB), in, 32768);                       /* input @ h4 */
        memcpy((char*)BIG.cpu + (0xffffa280u-IB), SM_WT, (size_t)SM_WT_WORDS*4);     /* weights @ 0xffffa280 */
        int NT=getenv("ORK_SM_NTASK")?atoi(getenv("ORK_SM_NTASK")):9; if(NT<1)NT=1; if(NT>9)NT=9;   /* RE: bisect */
        struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,(size_t)NT*sizeof *tk);
        for(int j=0;j<NT;j++){ int t=j; uint32_t *rc=(uint32_t*)((char*)BIG.cpu + (TADDR[j]-IB));   /* capture order = identity */
            int slot=(j<8)?(int)((TADDR[j+1]-TADDR[j])/4):SM_TASK_WORDS[t]; int nw=SM_TASK_WORDS[t]; if(nw>slot)nw=slot;
            memcpy(rc,SM_TASKS[t],(size_t)nw*4);
            for(int k=0;k+1<nw;k+=2){ unsigned a=rc[k]&0xffff,d=rc[k+1]>>16; uint32_t v=((rc[k+1]&0xffff)<<16)|(rc[k]>>16);
                /* BLANKET single-delta rebase: shift EVERY reference into the vendor image by delta. In the
                 * contiguous-image regime this is correct even for 1001:0x4110 (exp-LUT read ptr into the h2
                 * staging area) — validated empirically: blanket reaches task5 (counter 5), whitelist only
                 * task4 (counter 4). 0x0014 (amount, small) is out of [IB,IE) so it's untouched. */
                if(v>=IB && v<IE){ uint32_t nv=v+delta; rc[k]=a|((nv&0xffff)<<16); rc[k+1]=(d<<16)|((nv>>16)&0xffff); } }
            tk[j].enable_mask=SM_TASK_ENABLE[t]; tk[j].int_mask=0x300; tk[j].int_clear=0x1ffff;
            tk[j].regcfg_amount=SM_TASK_AMT[t]; tk[j].regcmd_addr=(uint32_t)BIG.dma + (TADDR[j]-IB);
        }
        orki_bsync(fd,&BIG,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        struct rknpu_submit s; memset(&s,0,sizeof s);
        /* flags=0x1 (RKNPU_JOB_PC only) — match the vendor softmax. The default 0x5 sets RKNPU_JOB_PINGPONG
         * (0x4) too; ping-pong double-buffers the register banks across tasks, and its bank-swap racing the
         * LUT-load commit is the flaky task4->task5 stall. The vendor runs softmax with ping-pong OFF. */
        s.flags=0x1; s.task_number=(uint32_t)NT; s.task_obj_addr=c->task.obj; s.core_mask=RKNPU_CORE0_MASK; s.fence_fd=-1; s.timeout=3000;
        s.subcore_task[0]=(struct rknpu_subcore_task){0,(uint32_t)NT};
        fprintf(stderr,"[softmax-replay] FULLIMG: %d-task chain, contiguous image, single-delta rebase, PINGPONG OFF (flags=0x1)\n",NT);
        int r=orki_rknpu_submit_ioctl(fd,&s,-1)?-1:0;
        if(r==0){ orki_bsync(fd,&BIG,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(out,BIG.cpu,32768); if(us)*us=ork_now_us()-t0; }  /* output @ h5 = BIG+0 */
        else fprintf(stderr,"[softmax-replay] FULLIMG chain failed\n");
        orki_bdestroy(fd,&BIG); orki_bdestroy(fd,&IN);orki_bdestroy(fd,&OUT);orki_bdestroy(fd,&SCR);orki_bdestroy(fd,&LUT);orki_bdestroy(fd,&WT);
        return r;
    }
    /* KERNEL-SEQUENCED (the accel/rocket model): submit each task as its OWN task_number=1 program, in
     * dataflow order 0..8 — one task per PC program, no in-regcmd PC-chain (0101:0x0010) rewriting. Each
     * task is the proven single-task submit path; intermediates persist in the resident SCR/IN/OUT/LUT
     * buffers between submits, and NO reset happens between them (orki_act(RESET) ran once above), so the exp
     * LUT that task4 loads into DPU SRAM survives for task5's exp. This is the correct sequencing model:
     * the earlier hardware-chained (task_number=9 + 0101 chain) attempts stalled at the chained LUT load. */
    static const int ORDER[9]={0,1,2,3,4,5,6,7,8};   /* capture/dataflow order */
    int rc_ret=0; double t0=ork_now_us();
    /* ORK_SM_1SUBMIT: the CHEAP hardware-chained form — ONE ioctl, task_number=9, ~one submit-floor cost.
     * The vendor runs softmax exactly this way (captured task_number=27), so it IS possible on rknpu. Replicate
     * the PROVEN run_chain/run_multicore recipe EXACTLY: concat regcmds, REWRITE the 0101:0x0010 chain to the
     * next task (terminate the last), op_idx=0 (memset), and populate ALL THREE subcore_task[]={0,9} (the
     * single-subcore / chain-zeroed / op_idx=1 variants each failed differently). */
    if(getenv("ORK_SM_1SUBMIT")){
        /* ONE-submit hardware chain replaying the VENDOR'S EXACT TASK LAYOUT. The vendor places each task's
         * regcmd at a 64-byte-ALIGNED offset (captured regcmd_addrs 0xffffab00,ad80,b000,b280,b380,d600,d880,
         * dc00,de40 -> relative words below). My earlier TIGHT packing landed task4 misaligned (+80 mod 128)
         * -> hung entering task4; the vendor's aligned layout is what the hardware chain-walk needs. Capture
         * order (0,1,..,8) so 0x0014 next-amounts match; chain 0x0010 -> next task's aligned offset. */
        static const int VOFF[9]={0,160,320,480,544,2752,2912,3136,3280};   /* vendor relative offsets (words) */
        /* ORK_SM_SPLIT: TWO hardware-chained submits split at the task4->task5 wall (the LUT-load->exp
         * transition that hangs in one chain): submit A = tasks [0..4] (loads the exp LUT into DPU SRAM),
         * submit B = tasks [5..8] (exp uses the persisted LUT). The submit boundary lets the LUT SRAM commit
         * and re-arms the PC cleanly for task5. Both chains use the vendor-aligned layout + 0x0014 recompute. */
        if(getenv("ORK_SM_SPLIT")){
            static const int SEG[2][2]={{0,5},{5,9}};   /* [start,end) task index ranges */
            for(int seg=0; seg<2 && rc_ret==0; seg++){ int s0=SEG[seg][0], s1=SEG[seg][1], ntt=s1-s0;
                uint32_t *rcb=(uint32_t*)c->regcmd.cpu;
                /* SEGMENT-LOCAL offsets from 0 (each submit starts its regcmd at c->regcmd.dma+0, where task5
                 * works in the per-task path), preserving the vendor SLOT sizes (=aligned spacing). */
                int loc[6]={0}; for(int j=0;j<ntt;j++){ int gi=s0+j; int slot=(gi<8)?(VOFF[gi+1]-VOFF[gi]):SM_TASK_WORDS[ORDER[gi]]; loc[j+1]=loc[j]+slot; }
                struct rknpu_task *tks=(struct rknpu_task*)c->task.cpu; memset(tks,0,(size_t)ntt*sizeof *tks);
                for(int j=0;j<ntt;j++){ int gi=s0+j, t=ORDER[gi]; uint32_t *rc=rcb+loc[j];
                    int slot=loc[j+1]-loc[j]; int nw=SM_TASK_WORDS[t]; if(nw>slot)nw=slot;
                    memcpy(rc,SM_TASKS[t],(size_t)nw*4);
                    uint32_t nextdma=(j<ntt-1)?(uint32_t)(c->regcmd.dma+(size_t)loc[j+1]*4):0u;
                    SM_REBASE(rc,nw,nextdma,(j<ntt-1)?SM_TASK_AMT[ORDER[gi+1]]:0);
                    tks[j].enable_mask=SM_TASK_ENABLE[t]; tks[j].int_mask=0x300; tks[j].int_clear=0x1ffff;
                    tks[j].regcfg_amount=SM_TASK_AMT[t]; tks[j].regcmd_addr=c->regcmd.dma+(size_t)loc[j]*4;
                }
                orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
                struct rknpu_submit s; memset(&s,0,sizeof s);
                s.flags=0x5; s.task_number=(uint32_t)ntt; s.task_obj_addr=c->task.obj; s.core_mask=RKNPU_CORE0_MASK; s.fence_fd=-1; s.timeout=3000;
                s.subcore_task[0]=(struct rknpu_subcore_task){0,(uint32_t)ntt};
                if(orki_rknpu_submit_ioctl(fd,&s,-1)){ fprintf(stderr,"[softmax-replay] SPLIT seg %d ([%d,%d)) failed\n",seg,s0,s1); rc_ret=-1; }
            }
            fprintf(stderr,"[softmax-replay] 2SUBMIT-SPLIT: chain [0..4] + chain [5..8] (vendor-aligned)\n");
            if(rc_ret==0){ orki_bsync(fd,&OUT,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(out,OUT.cpu,32768); if(us)*us=ork_now_us()-t0; }
            orki_bdestroy(fd,&IN);orki_bdestroy(fd,&OUT);orki_bdestroy(fd,&SCR);orki_bdestroy(fd,&LUT);orki_bdestroy(fd,&WT);
            return rc_ret;
        }
        int NT=getenv("ORK_SM_NTASK")?atoi(getenv("ORK_SM_NTASK")):9; if(NT<1)NT=1; if(NT>9)NT=9;  /* RE: bisect */
        uint32_t *rcbuf=(uint32_t*)c->regcmd.cpu;
        struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,(size_t)NT*sizeof *tk);
        for(int j=0;j<NT;j++){ int t=ORDER[j]; uint32_t *rc=rcbuf+VOFF[j];
            int slot=(j<8)?(VOFF[j+1]-VOFF[j]):SM_TASK_WORDS[t]; int nw=SM_TASK_WORDS[t]; if(nw>slot)nw=slot;  /* fit the slot */
            memcpy(rc,SM_TASKS[t],(size_t)nw*4);
            uint32_t nextdma=(j<NT-1)?(uint32_t)(c->regcmd.dma+(size_t)VOFF[j+1]*4):0u;
            SM_REBASE(rc,nw,nextdma,(j<NT-1)?SM_TASK_AMT[ORDER[j+1]]:0);
            tk[j].enable_mask=SM_TASK_ENABLE[t]; tk[j].int_mask=0x300; tk[j].int_clear=0x1ffff;
            tk[j].regcfg_amount=SM_TASK_AMT[t]; tk[j].regcmd_addr=c->regcmd.dma+(size_t)VOFF[j]*4;
        }
        orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
        orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        struct rknpu_submit s; memset(&s,0,sizeof s);
        s.flags=0x5; s.task_number=(uint32_t)NT; s.task_obj_addr=c->task.obj; s.core_mask=RKNPU_CORE0_MASK; s.fence_fd=-1; s.timeout=3000;
        s.subcore_task[0]=(struct rknpu_subcore_task){0,(uint32_t)NT};
        fprintf(stderr,"[softmax-replay] 1SUBMIT: %d-task hardware chain, vendor-exact aligned layout\n",NT);
        if(orki_rknpu_submit_ioctl(fd,&s,-1)){ fprintf(stderr,"[softmax-replay] 9-task chain failed\n"); rc_ret=-1; }
        if(rc_ret==0){ orki_bsync(fd,&OUT,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(out,OUT.cpu,32768); if(us)*us=ork_now_us()-t0; }
        orki_bdestroy(fd,&IN);orki_bdestroy(fd,&OUT);orki_bdestroy(fd,&SCR);orki_bdestroy(fd,&LUT);orki_bdestroy(fd,&WT);
        return rc_ret;
    }
    for(int j=0;j<9 && rc_ret==0;j++){ int t=ORDER[j]; int nw=SM_TASK_WORDS[t];
        uint32_t *rc=(uint32_t*)c->regcmd.cpu; memcpy(rc,SM_TASKS[t],(size_t)nw*4);
        SM_REBASE(rc,nw,0,0);                        /* addr remap only; nextdma=0 zeros the chain words */
        struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk);
        tk->enable_mask=SM_TASK_ENABLE[t]; tk->int_mask=0x300; tk->int_clear=0x1ffff;
        tk->regcfg_amount=SM_TASK_AMT[t]; tk->regcmd_addr=c->regcmd.dma;
        orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
        orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        struct rknpu_submit s; memset(&s,0,sizeof s);
        s.flags=0x5; s.task_number=1; s.task_obj_addr=c->task.obj; s.core_mask=RKNPU_CORE0_MASK;
        s.fence_fd=-1; s.timeout=3000;
        s.subcore_task[0]=s.subcore_task[1]=s.subcore_task[2]=(struct rknpu_subcore_task){0,1};
        if(orki_rknpu_submit_ioctl(fd,&s,-1)){ fprintf(stderr,"[softmax-replay] task %d (enable=0x%x amt=%d) submit failed\n",t,SM_TASK_ENABLE[t],SM_TASK_AMT[t]); rc_ret=-1; }
    }
    if(rc_ret==0){ orki_bsync(fd,&OUT,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(out,OUT.cpu,32768); if(us)*us=ork_now_us()-t0; }
    #undef SM_REBASE
    orki_bdestroy(fd,&IN);orki_bdestroy(fd,&OUT);orki_bdestroy(fd,&SCR);orki_bdestroy(fd,&LUT);orki_bdestroy(fd,&WT);
    return rc_ret;
}

int ork_npu_chain_mm_perchan_f16(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B,
                                 const uint16_t *scale,uint16_t *out,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems, dom=c->dom_active;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64||(N&7)) return -2;
    ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_HW);                    /* prime the fp16 pipeline (2-pass cold re-warm) — synth's fp16 matmul writes zeros unwarmed */
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    struct buf W=orki_bcreate(fd,(size_t)K*N*2,0x403,dom), G=orki_bcreate(fd,sz,0x403,dom), O=orki_bcreate(fd,sz,0x403,dom), SB=orki_bcreate(fd,4096,0x403,dom);
    if(!W.cpu||!G.cpu||!O.cpu||!SB.cpu){ orki_bdestroy(fd,&W);orki_bdestroy(fd,&G);orki_bdestroy(fd,&O);orki_bdestroy(fd,&SB); return -1; }
    { int NN=N/16,KT=K/32; uint16_t*bb=W.cpu;                                     /* fp16 weight tile [N/16][K/32][16][32] */
      for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*16+nl)]; }
    memset(G.cpu,0,sz); memset(O.cpu,0,sz); memset(SB.cpu,0,4096);
    { uint16_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; }                   /* fp16 activation raw [M][K] */
    uint32_t r34=0x00000008u;                                                     /* ERDMA per-channel + 2-byte (fp16 SDP) */
    { ork_f16*sb=(ork_f16*)SB.cpu; for(int n=0;n<N;n++) sb[n]=*(const ork_f16*)&scale[n]; }  /* fp16 per-channel scale CONTIGUOUS [N] */
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&G,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&SB,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    static uint32_t mm[REGCMD_N], pc[REGCMD_MUL_F16_CHAIN_N];
    int sched=((K&(K-1))==0 && K>=128 && K<2048);                                /* run_stream_f16's rule; small K => sched=0 (sched=1 miscomputes small K) */
    orki_synth(mm,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)G.dma,sched,CBUF); /* prog0: FP16 matmul -> G (CONTIGUOUS [M][N]) */
    orki_set_f16_out_fp16in(mm,M,N);                                                   /* fp16-out ATOM-8 stage (matches the SDP's orki_set_mul_geom layout) */
    memcpy(pc,REGCMD_MUL_F16_CHAIN,sizeof pc);
    orki_set_mul_geom(pc,REGCMD_MUL_F16_CHAIN_N,M,N);
    orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_SDP_5018,(uint32_t)G.dma);                /* INPUT = matmul OUTPUT (bridge) */
    orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_SDP_5038,(uint32_t)SB.dma);
    orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_SDP_5034,r34);
    double t0=ork_now_us();
    ork_chain_prog progs[2]={ {mm,REGCMD_N,0xd,108,216}, {pc,REGCMD_MUL_F16_CHAIN_N,0x18,69,-1} };
    int crc=ork_npu_chain_progs(c,2,progs,dom);
    if(!crc){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); if(us)*us=ork_now_us()-t0;
        for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=*(uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); }
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&G);orki_bdestroy(fd,&O);orki_bdestroy(fd,&SB);
    #undef EWCUBEH
    return crc;
}

int ork_npu_f16_gap_probe(ork_npu *c, int M, int Kp, int N, int use_gap, long *nz0, long *nz1, double *us) {
    int fd = c->fd, CBUF = c->soc->cbuf_elems, dom = c->dom_active;
    if (!ork_ppu_fuse_enabled(c)) return -3;
    if (Kp % 32 || N % 32 || N > c->soc->nmax || M < 1 || M > 64 || (N & 7)) return -2;
    ork_npu_enter(c, DT_F16, XP_STREAM_F16, OCK_HW);
    size_t gsz = (size_t)M * N * 2; if (gsz < 4096) gsz = 4096;
    struct buf W0 = orki_bcreate(fd,(size_t)Kp*N*2,0x403,dom), W1 = orki_bcreate(fd,(size_t)Kp*N*2,0x403,dom);
    struct buf G0 = orki_bcreate(fd,gsz,0x403,dom), G1 = orki_bcreate(fd,gsz,0x403,dom);
    struct buf GI = orki_bcreate(fd,gsz,0x403,dom), GO = orki_bcreate(fd,gsz,0x403,dom), SB = orki_bcreate(fd,4096,0x403,dom);
    if (!W0.cpu||!W1.cpu||!G0.cpu||!G1.cpu||!GI.cpu||!GO.cpu||!SB.cpu) {
        orki_bdestroy(fd,&W0);orki_bdestroy(fd,&W1);orki_bdestroy(fd,&G0);orki_bdestroy(fd,&G1);orki_bdestroy(fd,&GI);orki_bdestroy(fd,&GO);orki_bdestroy(fd,&SB); return -1; }
    { int NN=N/16, KT=Kp/32; uint16_t *b0=W0.cpu, *b1=W1.cpu;   /* fp16 weight tile [N/16][Kp/32][16][32]; W0 all 1.0, W1 all 1.0 shifted (distinct) */
      for (int nt=0;nt<NN;nt++) for (int kt=0;kt<KT;kt++) for (int nl=0;nl<16;nl++) for (int kk=0;kk<32;kk++) {
          size_t o=(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk; b0[o]=0x3c00; b1[o]=0x3c00; } }   /* 0x3c00 = fp16 1.0 */
    { uint16_t *ad=c->Af.cpu; for (int j=0;j<M*Kp;j++) ad[j]=0x3c00; }   /* A = 1.0 */
    memset(G0.cpu,0,gsz); memset(G1.cpu,0,gsz); memset(GI.cpu,0,gsz); memset(GO.cpu,0,gsz); memset(SB.cpu,0,4096);
    { uint16_t *sb=SB.cpu; for (int n=0;n<N;n++) sb[n]=0x3c00; }   /* identity per-channel scale (fp16 1.0) */
    orki_bsync(fd,&W0,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&W1,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&G0,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&G1,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&GI,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&GO,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&SB,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    static uint32_t mm0[REGCMD_N], mm1[REGCMD_N], pc[REGCMD_MUL_F16_CHAIN_N];
    int sched = ((Kp&(Kp-1))==0 && Kp>=128 && Kp<2048);
    orki_synth(mm0,M,Kp,N,(uint32_t)c->Af.dma,(uint32_t)W0.dma,(uint32_t)G0.dma,sched,CBUF); orki_set_f16_out_fp16in(mm0,M,N);
    orki_synth(mm1,M,Kp,N,(uint32_t)c->Af.dma,(uint32_t)W1.dma,(uint32_t)G1.dma,sched,CBUF); orki_set_f16_out_fp16in(mm1,M,N);
    memcpy(pc,REGCMD_MUL_F16_CHAIN,sizeof pc); orki_set_mul_geom(pc,REGCMD_MUL_F16_CHAIN_N,M,N);
    orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_DPU_DST_BASE_ADDR,(uint32_t)GO.dma);
    orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_SDP_5018,(uint32_t)GI.dma);   /* gap INPUT = dummy scratch (NOT the matmul output) */
    orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_SDP_5038,(uint32_t)SB.dma);
    orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_SDP_5034,0x00000008);
    double t0 = ork_now_us(); int crc;
    if (use_gap) { ork_chain_prog p[3] = { {mm0,REGCMD_N,0xd,108,216}, {pc,REGCMD_MUL_F16_CHAIN_N,0x18,69,138}, {mm1,REGCMD_N,0xd,108,-1} };
        crc = ork_npu_chain_progs(c,3,p,dom); }
    else { ork_chain_prog p[2] = { {mm0,REGCMD_N,0xd,108,216}, {mm1,REGCMD_N,0xd,108,-1} };
        crc = ork_npu_chain_progs(c,2,p,dom); }
    if (us) *us = ork_now_us()-t0;
    long z0=0, z1=0;
    if (!crc) { orki_bsync(fd,&G0,RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd,&G1,RKNPU_MEM_SYNC_FROM_DEVICE);
        uint16_t *g0=G0.cpu, *g1=G1.cpu; for (int e=0;e<M*N;e++){ if(g0[e])z0++; if(g1[e])z1++; } }
    if (nz0) *nz0=z0; if (nz1) *nz1=z1;
    orki_bdestroy(fd,&W0);orki_bdestroy(fd,&W1);orki_bdestroy(fd,&G0);orki_bdestroy(fd,&G1);orki_bdestroy(fd,&GI);orki_bdestroy(fd,&GO);orki_bdestroy(fd,&SB);
    return crc;
}

static void *ork_pcfd_thread(void *vp){
    struct ork_pcfd_arg *a=vp; struct rknpu_submit sub; memset(&sub,0,sizeof sub);
    sub.flags=ork_ppflags(); sub.task_number=1; sub.task_obj_addr=a->tk->obj; sub.fence_fd=-1;   /* BLOCKING per-core submit */
    sub.core_mask=1u<<a->core;
    sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,1};
    sub.timeout=orki_mm_timeout_ms();
    ork_kmsg("PCFD core=%d submit START fd=%d task_obj=0x%llx", a->core, a->fd, (unsigned long long)a->tk->obj);
    a->rc = orki_rknpu_submit_ioctl(a->fd,&sub,0);   /* buffers all live in domain 0 */
    ork_kmsg("PCFD core=%d submit DONE rc=%d", a->core, a->rc);
    return NULL;
}

int ork_npu_f16_percore_probe(ork_npu*c,int M,int K,int N,const ork_f16*A,const ork_f16*B,float*Cout,double*us,int mode){
    if(!c) return -3;
    int CBUF=c->soc->cbuf_elems;
    int cores=c->soc->cores; if(cores>ORK_MAXCORE) cores=ORK_MAXCORE; if(cores<1) cores=1;
    if(M<1||M>64||K%32||N%16||N>c->soc->nmax) return -2;
    if(N%(cores*16)) return -2;
    int Ncol=N/cores;
    const char*card=getenv("ORK_NPU_CARD"); if(!card)card=c->soc->card;
    int sched=((K&(K-1))==0 && K>=128 && K<2048);
    int cfd[ORK_MAXCORE]; for(int i=0;i<ORK_MAXCORE;i++) cfd[i]=-1;
    struct buf wbuf[ORK_MAXCORE]={{0}}, wimp[ORK_MAXCORE]={{0}}, abuf[ORK_MAXCORE]={{0}},
               cob[ORK_MAXCORE]={{0}}, rcb[ORK_MAXCORE]={{0}}, tkb[ORK_MAXCORE]={{0}};
    int shared_dbuf=-1, ret=-1;
    for(int i=0;i<cores;i++){ cfd[i]=open(card,O_RDWR); if(cfd[i]<0) goto done; orki_act(cfd[i],RKNPU_POWER_ON,0); }
    if(mode==1){   /* ONE shared full-N weight imported into every fd; tile the FULL B once via the primary map */
        size_t wsz=(size_t)K*N*2;
        wimp[0]=orki_bimport(cfd[0],wsz,0); if(!wimp[0].cpu) goto done; shared_dbuf=wimp[0].heap_fd;
        for(int i=1;i<cores;i++){ wimp[i]=orki_bimport_fd(cfd[i],shared_dbuf,wsz,0); if(!wimp[i].cpu) goto done; wimp[i].heap_fd=0; }
        int NNf=N/16, KT=K/32; ork_f16*bb=wimp[0].cpu;                          /* full tile [N/16][K/32][16][32] */
        for(int nt=0;nt<NNf;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
            bb[(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*16+nl)];
        for(int i=0;i<cores;i++) orki_bsync(cfd[i],&wimp[i],RKNPU_MEM_SYNC_TO_DEVICE);   /* clean each fd's mapping of the import */
    }
    for(int i=0;i<cores;i++){
        int n0=i*Ncol; uint32_t aB;
        if(mode==1){ aB=(uint32_t)(wimp[i].dma + (size_t)n0*K*2); }   /* col-tile slice: n0%16==0 => byte off = n0*K*2 (n0/16 whole tiles, each K/32*16*32*2 = K*32*2 B) */
        else {
            wbuf[i]=orki_bcreate(cfd[i],(size_t)K*Ncol*2,0x403,0); if(!wbuf[i].cpu) goto done;
            int NN=Ncol/16, KT=K/32; ork_f16*bb=wbuf[i].cpu;                   /* per-core tile [Ncol/16][K/32][16][32] of B's cols [n0,n0+Ncol) */
            for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
                bb[(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(n0+nt*16+nl)];
            orki_bsync(cfd[i],&wbuf[i],RKNPU_MEM_SYNC_TO_DEVICE);
            aB=(uint32_t)wbuf[i].dma;
        }
        abuf[i]=orki_bcreate(cfd[i],(size_t)M*K*2,0x403,0); if(!abuf[i].cpu) goto done;   /* full [M][K] activation (every core needs all of A) */
        { ork_f16*ad=abuf[i].cpu; for(int j=0;j<M*K;j++) ad[j]=A[j]; }
        orki_bsync(cfd[i],&abuf[i],RKNPU_MEM_SYNC_TO_DEVICE);
        size_t csz=(size_t)M*Ncol*4; if(csz<4096) csz=4096;              /* fp16-out uses M*Ncol*2; over-alloc to 4B/elem, harmless */
        cob[i]=orki_bcreate(cfd[i],csz,0x403,0); if(!cob[i].cpu) goto done;
        memset(cob[i].cpu,0,csz); orki_bsync(cfd[i],&cob[i],RKNPU_MEM_SYNC_TO_DEVICE);   /* seed */
        rcb[i]=orki_bcreate(cfd[i],(size_t)REGCMD_N*4,0x403,0); if(!rcb[i].cpu) goto done;
        tkb[i]=orki_bcreate(cfd[i],4096,0x40b,0); if(!tkb[i].cpu) goto done;
        uint32_t rc[REGCMD_N];
        orki_synth(rc,M,K,Ncol,(uint32_t)abuf[i].dma,aB,(uint32_t)cob[i].dma,sched,CBUF);
        orki_set_f16_out_fp16in(rc,M,Ncol);                                   /* fp16-out, contiguous [M][Ncol] (no ORK_F16_ATOM8) */
        memcpy(rcb[i].cpu,rc,(size_t)REGCMD_N*4);
        orki_bsync(cfd[i],&rcb[i],RKNPU_MEM_SYNC_TO_DEVICE);
        struct rknpu_task*t=(struct rknpu_task*)tkb[i].cpu; memset(t,0,sizeof *t);
        t->enable_mask=0xd; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=108; t->regcmd_addr=rcb[i].dma;
        orki_bsync(cfd[i],&tkb[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    }
    { double t0=ork_now_us();                                            /* concurrent per-core-fd submit: one blocking thread/core */
      pthread_t th[ORK_MAXCORE]; struct ork_pcfd_arg ar[ORK_MAXCORE]; int made=0;
      for(int i=0;i<cores;i++){ ar[i]=(struct ork_pcfd_arg){cfd[i],i,&tkb[i],0};
          if(pthread_create(&th[i],NULL,ork_pcfd_thread,&ar[i])!=0) break; made++; }
      for(int i=0;i<made;i++) pthread_join(th[i],NULL);
      if(us) *us=ork_now_us()-t0; }
    for(int i=0;i<cores;i++){ int n0=i*Ncol;                             /* read fp16-out back, de-column into Cout[M,N] fp32 */
        orki_bsync(cfd[i],&cob[i],RKNPU_MEM_SYNC_FROM_DEVICE);
        const ork_f16*cf=(const ork_f16*)cob[i].cpu;
        for(int m=0;m<M;m++) for(int n=0;n<Ncol;n++) Cout[(size_t)m*N+n0+n]=(float)cf[(size_t)m*Ncol+n];
    }
    ret=0;
done:
    for(int i=0;i<cores;i++){
        if(cfd[i]<0) continue;
        orki_bdestroy(cfd[i],&tkb[i]); orki_bdestroy(cfd[i],&rcb[i]); orki_bdestroy(cfd[i],&cob[i]); orki_bdestroy(cfd[i],&abuf[i]);
        if(mode==1) orki_bdestroy(cfd[i],&wimp[i]); else orki_bdestroy(cfd[i],&wbuf[i]);   /* wimp[0] closes the shared dbuf once (heap_fd zeroed on i>0) */
        close(cfd[i]);
    }
    return ret;
}

int ork_npu_probe_slice_f16(ork_npu *c,int Kfull,int N,int Kp,int nov,
                            const uint32_t *ovr_reg,const uint32_t *ovr_val,
                            const f16 *A,const f16 *B,float *C){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(Kfull%32||Kp%32||N%16||N>c->soc->nmax||Kp>Kfull) return -2;
    struct buf W=orki_bcreate(fd,(size_t)Kfull*N*2,0x403,-1); if(!W.cpu) return -2;
    int NN=N/16,KTf=Kfull/32; f16*bb=W.cpu;     /* full-K fp16 layout [Ntile][KTfull][16][32] */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KTf;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KTf*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*16+nl)];
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)N*4,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}
    f16*ad=c->Af.cpu; for(int j=0;j<Kp;j++)ad[j]=A[j]; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t rc[REGCMD_N];
    orki_synth(rc,1,Kp,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF);
    orki_setrn(rc,REGCMD_N,RK_CNA_CBUF_CON0,0xb1);
    for(int i=0;i<nov && i<4;i++) orki_setr(rc,REGCMD_N,0x201,ovr_reg[i],ovr_val[i]);
    struct buf extra[2] = {W, O};
    if (orki_validate_regcmd("probe_slice_f16", c, rc, REGCMD_N, NULL, extra, 2)) { orki_bdestroy(fd,&W); orki_bdestroy(fd,&O); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1;
    for(int rep=0;rep<2;rep++){ sub.timeout=orki_mm_timeout_ms();
        if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; continue; }
        orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(C,O.cpu,(size_t)N*4); ok=0; }
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);
    return ok;
}

int ork_ssd_probe_rawmm_f16(ork_npu*c,int M,int K,int N,const f16*A,const f16*B,float*C){
    if(!c||M<1||K<1||N<1||K%32||N%16) return -2;
    int fd=c->fd,CBUF=c->soc->cbuf_elems,dom=-1,ret=0;
    struct buf Ab=orki_bcreate(fd,(size_t)M*K*2,0x403,dom),Bb=orki_bcreate(fd,(size_t)K*N*2,0x403,dom),Cb=orki_bcreate(fd,(size_t)M*N*4,0x403,dom);
    if(!Ab.cpu||!Bb.cpu||!Cb.cpu){ ret=-3; goto done; }
    memcpy(Ab.cpu,A,(size_t)M*K*2); memcpy(Bb.cpu,B,(size_t)K*N*2); memset(Cb.cpu,0,(size_t)M*N*4);
    orki_bsync(fd,&Ab,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Bb,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Cb,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0); c->warmed=0; c->last_dt=DT_F16;
    { uint32_t *rc=calloc(REGCMD_I8_N,4); if(!rc){ ret=-3; goto done; }
      orki_synth(rc,M,K,N,(uint32_t)Ab.dma,(uint32_t)Bb.dma,(uint32_t)Cb.dma,1,CBUF);
      ork_chain_prog p={rc,REGCMD_I8_N,0xd,108,216};
      ret=ork_npu_chain_progs(c,1,&p,dom); free(rc); }
    if(ret) goto done;
    orki_bsync(fd,&Cb,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(C,Cb.cpu,(size_t)M*N*4);
done:
    orki_bdestroy(fd,&Ab);orki_bdestroy(fd,&Bb);orki_bdestroy(fd,&Cb);
    return ret;
}

int ork_ssd_probe_fusedmm_f16(ork_npu*c,int M,int K,int N,const f16*A,const f16*B,float*C){
    if(!c||M<1||K<1||N<1||K%32||N%16) return -2;
    ork_w *w=ork_mm_pack(c,K,N,B); if(!w) return -3;
    if(w->Sk!=1||w->Sn!=1){ ork_mm_free(c,w); return -2; }   /* probe: single tile only */
    int fd=c->fd,CBUF=c->soc->cbuf_elems,dom=w->domain,ret=0;
    struct buf Ab=orki_bcreate(fd,(size_t)M*K*2,0x403,dom), Cb=orki_bcreate(fd,(size_t)M*N*4,0x403,dom);
    if(!Ab.cpu||!Cb.cpu){ ret=-3; goto done2; }
    memcpy(Ab.cpu,A,(size_t)M*K*2); memset(Cb.cpu,0,(size_t)M*N*4);
    orki_bsync(fd,&Ab,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Cb,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0); c->warmed=0; c->last_dt=DT_F16;
    { uint32_t *rc=calloc(REGCMD_I8_N,4); if(!rc){ ret=-3; goto done2; }
      orki_synth(rc,M,K,N,(uint32_t)Ab.dma,(uint32_t)w->Bb[0].dma,(uint32_t)Cb.dma,1,CBUF);
      ork_chain_prog p={rc,REGCMD_I8_N,0xd,108,216};
      ret=ork_npu_chain_progs(c,1,&p,dom); free(rc); }
    if(ret) goto done2;
    orki_bsync(fd,&Cb,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(C,Cb.cpu,(size_t)M*N*4);
done2:
    orki_bdestroy(fd,&Ab);orki_bdestroy(fd,&Cb); ork_mm_free(c,w);
    return ret;
}

int ork_bmm_fp16_fused(ork_npu*c,int nb,int M,int K,int N,const f16*A,const f16*B,float*C){
    if(!c||nb<1||nb>64||M<1||K<1||N<1||K%32||N%16) return -2;
    /* B3 (chain_progs retirement): the batch of nb independent fp16 matmuls now rides the NONBLOCK-doorbell
     * fp16 PC-chain (ork_mm_run_stream_f16_chain) instead of the legacy ork_npu_chain_progs. The stream primitive
     * owns staging + warm/reset (enters DT_F16/XP_STREAM_F16 — fp16 content tracked AS fp16, unifying this with
     * the other fp16 scan stages; the old path entered DT_I8_CHAIN, whose "genuine int8->fp16 switch needs a
     * reset" caveat this removes). Spreads the batch across cores rather than one core-0 chain. Bit-exact
     * (test_bmm_fused: fused == per-op ork_bmm_fp16 == CPU). chain_progs stays for the tools/ RE probes. */
    ork_w **w=calloc(nb,sizeof(ork_w*));
    ork_mm_task_f16 *tk=malloc((size_t)nb*sizeof(ork_mm_task_f16));
    if(!w||!tk){ free(w); free(tk); return -3; }
    int ret=0;
    for(int b=0;b<nb;b++){
        w[b]=ork_mm_pack(c,K,N,B+(size_t)b*K*N);
        if(!w[b]||w[b]->Sk!=1||w[b]->Sn!=1){ ret=-3; goto done3; }
        tk[b]=(ork_mm_task_f16){w[b],M,A+(size_t)b*M*K,C+(size_t)b*M*N};
    }
    ret=ork_mm_run_stream_f16_chain(c,nb,tk);
done3:
    for(int b=0;b<nb;b++) if(w[b]) ork_mm_free(c,w[b]);
    free(w); free(tk);
    return ret;
}

static void *stream_worker_f16(void *vp){
    struct streamw_f16 *a=vp; ork_npu *c=a->c; int fd=c->fd, i=a->core, CBUF=c->soc->cbuf_elems;
    if(CBUF>32768) CBUF=32768;                     /* fp16 M-scheduler validated to the 32768-tile */
    orki_pin_big_core(i);
    int k; a->rc=0;
    uint32_t rc[REGCMD_I8_N];
    while((k=__atomic_fetch_add(a->ctr,1,__ATOMIC_SEQ_CST))<a->S){
        const ork_mm_task_f16 *t=&a->tasks[k]; ork_w *w=t->w; int M=t->M, K=w->K, N=w->N;
        int sched=(K&(K-1))==0 && K>=128 && K<2048;
        memcpy(c->maf[i].cpu, t->A, (size_t)M*K*2); orki_bsync(fd,&c->maf[i],RKNPU_MEM_SYNC_TO_DEVICE);
        memset(rc,0,REGCMD_I8_N*4);
        orki_synth(rc, M, K, N, (uint32_t)c->maf[i].dma, (uint32_t)w->Bb[0].dma, (uint32_t)c->mcc[i].dma, sched, CBUF);
        memcpy(c->mrc[i].cpu, rc, REGCMD_I8_N*4); orki_bsync(fd,&c->mrc[i],RKNPU_MEM_SYNC_TO_DEVICE);
        struct rknpu_task *mt=c->mtk[i].cpu; memset(mt,0,sizeof *mt);
        mt[0].enable_mask=0xd; mt[0].int_mask=0x300; mt[0].int_clear=0x1ffff; mt[0].regcfg_amount=108; mt[0].regcmd_addr=c->mrc[i].dma;
        orki_bsync(fd,&c->mtk[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        struct rknpu_submit sub; memset(&sub,0,sizeof sub);
        sub.flags=ork_ppflags(); sub.task_number=1; sub.task_obj_addr=c->mtk[i].obj; sub.core_mask=1u<<i; sub.fence_fd=-1;
        sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,1};
        sub.timeout=orki_mm_timeout_ms();
        int reps=c->mwarm[i]?1:2;
        for(int rep=0;rep<reps;rep++){ if(orki_rknpu_submit_ioctl(fd,&sub,w->domain)){ if(rep==reps-1)a->rc=-1; continue; } orki_bsync(fd,&c->mcc[i],RKNPU_MEM_SYNC_FROM_DEVICE); }
        c->mwarm[i]=1;
        memcpy(t->C, c->mcc[i].cpu, (size_t)M*N*4);
    }
    return NULL;
}

int ork_mm_run_stream_f16(ork_npu *c, int S, const ork_mm_task_f16 *tasks){
    if(!c||S<1||!tasks) return -2;
    if(tasks[0].w && (tasks[0].w->domain!=c->dom_active || (tasks[0].w->domain!=0 && !c->dom_save))) orki_dom_activate(c,tasks[0].w->domain);
    for(int i=0;i<S;i++){ ork_w *w=tasks[i].w;
        if(!w||w->dtype!=DT_F16||tasks[i].M<=0) return -2;
        if(w->Sn!=1||w->Sk!=1||!w->Bb) return -2;              /* single-slice fp16 (K<=ks,N<=nmax) */
        if(w->K%32||w->N%16) return -2; }
    /* P3 SPINE MIGRATION: fp16 stream onto the NONBLOCK doorbell (ork_dyn_begin_mc), like run_stream_i8. The
     * doorbell fp16 path accepts single-slice small-K (K%32) shapes (uses Bb + the K-dependent sched); A stays
     * host (the fp16 doorbell stages A into maf). Build the neutral ork_mm_task_i8 view — w carries dtype=DT_F16,
     * A/C byte-reinterpreted (f16 A, fp32 C). fp16 already full-surface seeds + M<=64 always-cleans (interleave-
     * safe), and rknpu_submit_ioctl retries a transient submit-rejection. No legacy fallback (miss => -1). */
    if(S>1024) return -2;
    ork_mm_task_i8 ti[1024];
    for(int i=0;i<S;i++) ti[i]=(ork_mm_task_i8){ tasks[i].w, tasks[i].M, (const int8_t*)tasks[i].A, (int32_t*)tasks[i].C };
    int nc=orki_budget(c,2); if(nc>ORK_MAXCORE)nc=ORK_MAXCORE; if(nc>S)nc=S; if(nc<1)nc=1;
    ork_dyn_chain *h=ork_dyn_begin_mc(c,S,ti,nc);
    if(!h) return -1;
    int d=ork_dyn_end(h);
    return (d==S-1)?0:-1;
}

static void *stream_worker_f16ch(void *vp){
    struct streamw_f16ch *a=vp; ork_npu *c=a->c; int fd=c->fd, i=a->core, ncore=a->ncore, S=a->S, CBUF=c->soc->cbuf_elems;
    if(CBUF>32768) CBUF=32768;
    orki_pin_big_core(i);
    a->rc=0;
    int cnt=0; for(int k=i;k<S;k+=ncore) cnt++;
    if(cnt==0) return NULL;
    uint32_t rc[REGCMD_I8_N];
    struct rknpu_task *mt=c->mtk[i].cpu;
    int p=0;
    for(int k=i;k<S;k+=ncore,p++){
        const ork_mm_task_f16 *t=&a->tasks[k]; ork_w *w=t->w; int M=t->M, K=w->K, N=w->N;
        int sched=(K&(K-1))==0 && K>=128 && K<2048;
        memcpy((char*)c->maf[i].cpu + (size_t)p*M*K*2, t->A, (size_t)M*K*2);
        memset(rc,0,REGCMD_I8_N*4);
        orki_synth(rc, M, K, N, (uint32_t)(c->maf[i].dma + (size_t)p*M*K*2), (uint32_t)w->Bb[0].dma,
              (uint32_t)(c->mcc[i].dma + (size_t)p*M*N*4), sched, CBUF);
        if(p<cnt-1){ uint64_t next=c->mrc[i].dma + (size_t)(p+1)*REGCMD_I8_N*4;   /* PC-chain to next program */
            rc[216]=0x0010|((next&0xffff)<<16); rc[217]=(0x0101u<<16)|((uint32_t)(next>>16)&0xffff); rc[218]=0x0014|(0x0037u<<16); }
        memcpy((char*)c->mrc[i].cpu + (size_t)p*REGCMD_I8_N*4, rc, REGCMD_I8_N*4);
        memset(&mt[p],0,sizeof mt[p]); mt[p].enable_mask=0xd; mt[p].int_mask=0x300; mt[p].int_clear=0x1ffff;
        mt[p].regcfg_amount=108; mt[p].regcmd_addr=c->mrc[i].dma + (size_t)p*REGCMD_I8_N*4;
    }
    orki_bsync(fd,&c->maf[i],RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&c->mrc[i],RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&c->mtk[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub; memset(&sub,0,sizeof sub);
    sub.flags=ork_ppflags(); sub.task_number=cnt; sub.task_obj_addr=c->mtk[i].obj; sub.core_mask=1u<<i; sub.fence_fd=-1;
    sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)cnt};
    sub.timeout=orki_mm_timeout_ms();
    int reps=c->mwarm[i]?1:2;
    for(int rep=0;rep<reps;rep++){ if(orki_rknpu_submit_ioctl(fd,&sub,a->tasks[i].w->domain)){ if(rep==reps-1)a->rc=-1; continue; } orki_bsync(fd,&c->mcc[i],RKNPU_MEM_SYNC_FROM_DEVICE); }
    c->mwarm[i]=1;
    p=0; for(int k=i;k<S;k+=ncore,p++){ const ork_mm_task_f16 *t=&a->tasks[k]; int M=t->M,N=t->w->N;
        memcpy(t->C, (char*)c->mcc[i].cpu + (size_t)p*M*N*4, (size_t)M*N*4); }
    return NULL;
}

int ork_mm_run_stream_f16_chain(ork_npu *c, int S, const ork_mm_task_f16 *tasks){
    if(!c||S<1||!tasks) return -2;
    if(tasks[0].w && (tasks[0].w->domain!=c->dom_active || (tasks[0].w->domain!=0 && !c->dom_save))) orki_dom_activate(c,tasks[0].w->domain);
    for(int i=0;i<S;i++){ ork_w *w=tasks[i].w;
        if(!w||w->dtype!=DT_F16||tasks[i].M<=0) return -2;
        if(w->Sn!=1||w->Sk!=1||!w->Bb) return -2;
        if(w->K%32||w->N%16) return -2; }
    int fd=c->fd;
    ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_SW);
    int nc=orki_budget(c,2); if(nc>ORK_MAXCORE)nc=ORK_MAXCORE; if(nc>S)nc=S; if(nc<1)nc=1;
    if(orki_mc_ensure(c,nc)) return -1;
    int per=(S+nc-1)/nc;                                   /* max programs a single core owns */
    size_t needrc=(size_t)per*REGCMD_I8_N*4, needtk=(size_t)per*sizeof(struct rknpu_task);
    size_t maxMK=(size_t)per*tasks[0].M*tasks[0].w->K*2, maxMN4=(size_t)per*tasks[0].M*tasks[0].w->N*4;
    for(int i=0;i<nc;i++){
        if(c->mrc[i].size<needrc){ orki_bdestroy(fd,&c->mrc[i]); c->mrc[i]=orki_bcreate(fd,needrc,0x403,c->dom_active); if(!c->mrc[i].cpu)return -1; c->mwarm[i]=0; }
        if(c->mtk[i].size<needtk){ orki_bdestroy(fd,&c->mtk[i]); c->mtk[i]=orki_bcreate(fd,needtk,0x40b,c->dom_active); if(!c->mtk[i].cpu)return -1; }
        if(c->maf[i].size<maxMK){ orki_bdestroy(fd,&c->maf[i]); c->maf[i]=orki_bcreate(fd,maxMK,0x403,c->dom_active); if(!c->maf[i].cpu)return -1; }
        if(c->mccsz[i]<maxMN4){ orki_bdestroy(fd,&c->mcc[i]); c->mcc[i]=orki_bcreate(fd,maxMN4,0x403,c->dom_active); c->mccsz[i]=maxMN4; if(!c->mcc[i].cpu)return -1; c->mwarm[i]=0; } }
    int rc=0; orki_npu_pool_ensure(c);
    struct streamw_f16ch sw[ORK_MAXCORE];
    for(int i=0;i<nc;i++) sw[i]=(struct streamw_f16ch){c,i,nc,S,tasks,0};
    pthread_mutex_lock(&c->pmu);
    c->pjob=sw; c->pjob_nc=nc; c->pjob_fn=stream_worker_f16ch; c->pjob_stride=sizeof(struct streamw_f16ch);
    c->pdone=0; c->pgen++; pthread_cond_broadcast(&c->pgo);
    pthread_mutex_unlock(&c->pmu);
    stream_worker_f16ch(&sw[0]);
    pthread_mutex_lock(&c->pmu); while(c->pdone<nc-1) pthread_cond_wait(&c->pdn,&c->pmu); pthread_mutex_unlock(&c->pmu);
    for(int i=0;i<nc;i++) if(sw[i].rc) rc=-1;
    c->warmed=1;
    return rc;
}

int ork_bmm_fp16_stream(ork_npu*c,int nb,int M,int K,int N,const f16*A,const f16*B,float*C){
    if(!c||nb<1||M<1||K<1||N<1||K%32||N%16) return -2;
    ork_w **w=calloc(nb,sizeof(ork_w*)); ork_mm_task_f16 *tk=malloc((size_t)nb*sizeof(ork_mm_task_f16));
    if(!w||!tk){ free(w);free(tk); return -3; }
    int ret=0;
    for(int b=0;b<nb;b++){ w[b]=ork_mm_pack(c,K,N,B+(size_t)b*K*N);
        if(!w[b]||w[b]->Sk!=1||w[b]->Sn!=1){ ret=-3; goto done; }
        tk[b]=(ork_mm_task_f16){w[b],M,A+(size_t)b*M*K,C+(size_t)b*M*N}; }
    ret=ork_mm_run_stream_f16(c,nb,tk);
done:
    for(int b=0;b<nb;b++) if(w[b]) ork_mm_free(c,w[b]);
    free(w);free(tk);
    return ret;
}

int ork_bmm_fp16_strided(ork_npu *c, int nbatch, int M, int K, int N,
                         const f16 *A, const f16 *B, float *C, const ork_bmm_strides *s){
    if(!c||!A||!B||!C||!s) return -1;
    if(nbatch<1||M<1||K<1||N<1) return -2;
    if(K%32||N%16) return -2;
    f16 *Ac=malloc((size_t)M*K*sizeof(f16)), *Bc=malloc((size_t)K*N*sizeof(f16));
    int cdense=orki_bmm_c_dense(s,N); float *Cc = cdense?NULL:malloc((size_t)M*N*sizeof(float));
    if(!Ac||!Bc||(!cdense&&!Cc)){ free(Ac);free(Bc);free(Cc); return -3; }
    int rc=0;
    for(int b=0;b<nbatch;b++){
        orki_bmm_gather_f16(Bc,B+(long)b*s->bbs,K,N,s->bs_k,s->bs_n);
        orki_bmm_gather_f16(Ac,A+(long)b*s->abs,M,K,s->as_m,s->as_k);
        ork_w *w=ork_mm_pack(c,K,N,Bc); if(!w){ rc=-3; break; }
        float *Cout = cdense ? C+(long)b*s->cbs : Cc;
        int r=ork_mm_run(c,w,M,Ac,Cout);
        ork_mm_free(c,w);
        if(r){ rc=-5; break; }
        if(!cdense) orki_bmm_scatter_i32((int32_t*)(C+(long)b*s->cbs),(const int32_t*)Cc,M,N,s->cs_m,s->cs_n);
    }
    free(Ac);free(Bc);free(Cc);
    return rc;
}

int ork_bmm_fp16(ork_npu *c, int nbatch, int M, int K, int N,
                 const f16 *A, const f16 *B, float *C){
    ork_bmm_strides s=orki_bmm_natural(M,K,N); return ork_bmm_fp16_strided(c,nbatch,M,K,N,A,B,C,&s);
}

int ork_npu_rmsnorm_f16(ork_npu *c,int M,int n,const f16 *x,const f16 *w,float eps,f16 *out){
    if(!c||!x||!w||!out||M<1||n<1) return -2;
    float *ss=malloc((size_t)M*sizeof(float)), *sc=malloc((size_t)M*sizeof(float)); int have_ss=0, have_sc=0;
    if(ss && ork_norm_reduce_npu(c,M,n,x,ss)==0) have_ss=1;                 /* sum(x^2) on NPU (any n) */
    if(have_ss && sc && ork_norm_rsqrt_npu(c,M,n,(double)eps,ss,sc)==0) have_sc=1; /* rsqrt on NPU (K=512) */
    for(int m=0;m<M;m++){ const f16 *xr=x+(size_t)m*n; f16 *o=out+(size_t)m*n; float s;
        if(have_sc) s=sc[m];
        else { double sumsq; if(have_ss) sumsq=(double)ss[m]; else { sumsq=0; for(int i=0;i<n;i++){ double v=(double)xr[i]; sumsq+=v*v; } }
               s=(float)(1.0/sqrt(sumsq/(double)n+(double)eps)); }
        for(int i=0;i<n;i++) o[i]=(f16)((float)xr[i]*s*(float)w[i]); }
    free(ss); free(sc); return 0;
}

int ork_npu_l2norm_f16(ork_npu *c,int M,int n,const f16 *x,float eps,f16 *out){
    if(!c||!x||!out||M<1||n<1) return -2;
    float *ss=malloc((size_t)M*sizeof(float)), *sc=malloc((size_t)M*sizeof(float)); int have_ss=0, have_sc=0;
    if(ss && ork_norm_reduce_npu(c,M,n,x,ss)==0) have_ss=1;
    if(have_ss && sc && ork_norm_rsqrt_npu(c,M,1,(double)eps,ss,sc)==0) have_sc=1; /* nf=1: 1/sqrt(ss+eps) */
    for(int m=0;m<M;m++){ const f16 *xr=x+(size_t)m*n; f16 *o=out+(size_t)m*n; float s;
        if(have_sc) s=sc[m];
        else { double sumsq; if(have_ss) sumsq=(double)ss[m]; else { sumsq=0; for(int i=0;i<n;i++){ double v=(double)xr[i]; sumsq+=v*v; } }
               s=(float)(1.0/sqrt(sumsq+(double)eps)); }
        for(int i=0;i<n;i++) o[i]=(f16)((float)xr[i]*s); }
    free(ss); free(sc); return 0;
}

int ork_npu_softmax_f16(ork_npu *c,int M,int n,const f16 *x,f16 *out){
    if(!c||!x||!out||M<1||n<1) return -2;
    float *mx=malloc((size_t)M*sizeof(float)), *e=malloc((size_t)M*n*sizeof(float)), *s=malloc((size_t)M*sizeof(float));
    if(!mx||!e||!s){ free(mx);free(e);free(s); return -1; }
    for(int m=0;m<M;m++){ float mv=(float)x[(size_t)m*n]; for(int j=1;j<n;j++){ float v=(float)x[(size_t)m*n+j]; if(v>mv)mv=v; } mx[m]=mv; }
    int have_npu=0;
    /* Composition: max (CPU) -> exp(x-max) on the NPU (SDP act-LUT, int16) -> Sum + scale on CPU.
     * The Sum is intentionally NOT a reduce-matmul here: an activation(exp)->matmul(reduce) submit
     * reliably ETIMEDOUTs on the stateful activation->matmul mode-switch (the reverse order,
     * matmul->activation, is fine — cf. the rsqrt path), and self-healing per row-batch just discards
     * the good NPU exp. So exp rides the NPU (the transcendental win) and the cheap Sigma stays on CPU. */
    if(ork_softmax_npu_enabled() && n%32==0){
        float lo=0; for(int m=0;m<M;m++){ float mv=mx[m]; for(int j=0;j<n;j++){ float d=(float)x[(size_t)m*n+j]-mv; if(d<lo)lo=d; } }
        double in_scale=(-lo)/32000.0; if(in_scale<=0) in_scale=1e-6; double out_scale=1.0/32000.0;
        int16_t *xi=malloc((size_t)M*n*2), *ei=malloc((size_t)M*n*2);
        if(xi&&ei){
            for(int m=0;m<M;m++) for(int j=0;j<n;j++){ long q=lround(((double)((float)x[(size_t)m*n+j]-mx[m]))/in_scale); if(q<-32768)q=-32768; if(q>32767)q=32767; xi[(size_t)m*n+j]=(int16_t)q; }
            if(ork_npu_exp_i16(c,xi,M,n,in_scale,out_scale,ei,NULL)==0){                 /* exp(x-max) on NPU */
                for(int m=0;m<M;m++){ double sm=0; for(int j=0;j<n;j++){ double d=(double)ei[(size_t)m*n+j]*out_scale; e[(size_t)m*n+j]=(float)d; sm+=d; } s[m]=(float)sm; }
                have_npu=1;
            }
        }
        free(xi);free(ei);
    }
    if(!have_npu){ for(int m=0;m<M;m++){ double sm=0; for(int j=0;j<n;j++){ float d=expf((float)x[(size_t)m*n+j]-mx[m]); e[(size_t)m*n+j]=d; sm+=d; } s[m]=(float)sm; } }
    for(int m=0;m<M;m++){ float inv=1.0f/s[m]; for(int j=0;j<n;j++) out[(size_t)m*n+j]=(f16)(e[(size_t)m*n+j]*inv); }
    free(mx);free(e);free(s); return 0;
}

void orki_synth(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC,int sched,int cbuf){
    memcpy(rc,REGCMD,REGCMD_N*4);
    orki_setrn(rc,REGCMD_N,RK_CNA_DATA_SIZE1,((K-1)<<16)|K);orki_setrn(rc,REGCMD_N,RK_CNA_WEIGHT_SIZE0,K*N*2);orki_setrn(rc,REGCMD_N,RK_CNA_WEIGHT_SIZE1,K*2);
    orki_setrn(rc,REGCMD_N,RK_CNA_CBUF_CON1,K/32);orki_setrn(rc,REGCMD_N,RK_CNA_FC_DATA_SIZE1,K);orki_setrn(rc,REGCMD_N,RK_CNA_DMA_CON1,K/8);
    orki_setrn(rc,REGCMD_N,RK_CNA_DATA_SIZE0,0x10000|mc);orki_setrn(rc,REGCMD_N,RK_CNA_DATA_SIZE0_MIR,0x10000|mc);orki_setrn(rc,REGCMD_N,RK_CNA_DATA_SIZE3,mc);
    orki_setrn(rc,REGCMD_N,RK_DPU_DATA_CUBE_HEIGHT,mc-1);orki_setrn(rc,REGCMD_N,RK_DPU_WDMA_SIZE_1,(mc-1)<<16);orki_setrn(rc,REGCMD_N,RK_PDP_OUT_M,(mc-1)<<16);
    orki_setrn(rc,REGCMD_N,RK_DPU_DST_N_DIMS,((N-1)<<16)|(N-1));orki_setrn(rc,REGCMD_N,RK_DPU_DST_N2,N-1);orki_setrn(rc,REGCMD_N,RK_DPU_DATA_CUBE_NOTCH,(((N/4)-1)<<16)|((N/4)-1));
    orki_setrn(rc,REGCMD_N,RK_CNA_WEIGHT_SIZE2,0x1010000|N);orki_setrn(rc,REGCMD_N,RK_PDP_OUT_N,N-1);
    if(sched){
        int R=cbuf/K; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; } int rows=(mc+1<R)?(mc+1):R; orki_setrn(rc,REGCMD_N,RK_CNA_CONV_CON2,16*rows);
        double scale=(double)K/256.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale),mg=(mc+63)/64; if(mg<1)mg=1;  /* ceil: see orki_synth_i8 (65..127-row tile needs the next K-schedule) */
        int v=base-slope*(mg-1); if(v<0x1b)v=0x1b; orki_setrn(rc,REGCMD_N,RK_CNA_CBUF_CON0,v);
    } else { orki_setrn(rc,REGCMD_N,RK_CNA_CONV_CON2,16*(mc+1)); }
    orki_setrn(rc,REGCMD_N,RK_CNA_FEATURE_DATA_ADDR,aA);orki_setrn(rc,REGCMD_N,RK_CNA_WEIGHT_DATA_ADDR,aB);orki_setrn(rc,REGCMD_N,RK_DPU_DST_BASE_ADDR,aC);
    for(int i=0;i<orki_f16_fovr_n;i++) orki_setr(rc,REGCMD_N,orki_f16_fovr[i].blk,orki_f16_fovr[i].reg,orki_f16_fovr[i].val);  /* RE fuzzer overrides (raw/unbounded) */
}
