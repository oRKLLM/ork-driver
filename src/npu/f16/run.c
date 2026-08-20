/* npu/f16/run.c — fp16 execution: pack/run, fused matmul+activation, the LUT builders.
 *
 * Part of the f16 datapath; shared declarations in npu/f16/f16.h. Split out of npu/f16.c for the
 * same reason i8 is a folder: one datapath, sized for reading. */
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
#include "npu/f16/f16.h"

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
