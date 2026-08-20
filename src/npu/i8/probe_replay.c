/* npu/i8/probe_replay.c — int8 REPLAY probes: replay a captured regcmd verbatim, the A-variant sweep and A-layout mapper, and the fused-SiLU config probes.
 *
 * Part of the int8 (W8A8) datapath. Split out of the 1,270-line npu/i8/probe.c (MODULARIZE_PLAN.md
 * round 6) as a CONTIGUOUS line range, so the pieces concatenate back to the original byte for byte.
 * Interface types in npu/internal.h, substrate in npu/core.h. Board-only diagnostics. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include "ork_regs.h"
#include "regcmd_i8.h"
#include "regcmd_i4.h"
#include "regcmd_silu.h"
#include "regcmd_ewmul.h"
#include "regcmd_fold_refs.h"
#include "npu/internal.h"
#include "npu/core.h"
#include "npu/i8/i8.h"
#include "spine_kernels.h"
int ork_i8_npu_replay(ork_npu *c, const uint32_t *regcmd, int rn, int M, int K, int N,
                      const int8_t *Adata, int Abytes, const int8_t *Bdata, int Bbytes, int32_t *Cout, int iters, double *us){
    int fd=c->fd; if(fd<0) return -3; if(rn<8 || rn>2048 || M<1 || (K%32) || (N%16)) return -2;
    int dom=c->dom_active;
    size_t bsz=(Bbytes>(int)((size_t)K*N))?(size_t)Bbytes:(size_t)K*N;   /* B may be a STRIDED span > K*N (rkllm column-tile) */
    /* #39 SAFETY RIG: oversize A/B (like C) with a large guard margin so a malformed mfold DMA descriptor that
     * reads out-of-bounds lands in MAPPED memory (garbage -> diagnosable) instead of hanging the AXI/IOMMU bus
     * (the hard-wedge failure mode). The extra pages are mapped in the same IOVA domain and never freed early. */
    size_t aszg=(size_t)M*K*8+(1u<<20), bszg=bsz*8+(1u<<20);
    struct buf A =orki_bcreate(fd,aszg,0x403,dom);            if(!A.cpu)  return -2;
    struct buf B =orki_bcreate(fd,bszg,0x403,dom);            if(!B.cpu) {orki_bdestroy(fd,&A);return -2;}
    memset(A.cpu,0,aszg); memset(B.cpu,0,bszg);          /* zero the guard region so OOB reads see zeros, not stale */
    size_t cszg=(size_t)M*N*4*8+65536;                   /* #39: oversize C 8x as an OOB guard while RE'ing the mfold output stride (a wrong stride writes in-bounds -> diagnosable, not a wedge) */
    struct buf Cc=orki_bcreate(fd,cszg,0x403,dom);            if(!Cc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);return -2;}
    struct buf RCb=orki_bcreate(fd,(size_t)rn*4,0x403,dom);   if(!RCb.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&Cc);return -2;}
    if(Adata){ size_t acp=(Abytes>0)?(size_t)Abytes:(size_t)M*K; if(acp>aszg)acp=aszg; memcpy(A.cpu,Adata,acp); }   /* real captured A (full span, incl. any fold padding), clamped to the buffer */
    else memset(A.cpu,1,(size_t)M*K);   /* garbage (timing only) */
    if(Bdata) memcpy(B.cpu,Bdata,(Bbytes>0)?(size_t)Bbytes:(size_t)K*N); else memset(B.cpu,1,bsz);   /* real captured strided weight, or garbage */
    memset(Cc.cpu,0,cszg);
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t *rc=(uint32_t*)RCb.cpu; memcpy(rc,regcmd,(size_t)rn*4);
    /* #39 SAFETY: neutralize any PC chain descriptor (0x0101:0x0010 next-regcmd-addr / 0x0014 next-amount)
     * baked into a CAPTURED regcmd. We submit task_number=1 (single task), so the chain link is meaningless
     * here — but if left intact it points at the CAPTURING process's next-task IOVA, which is UNMAPPED in
     * ork's address space; the NPU walks the chain to it and IOMMU HARD-WEDGES the board. Zero the value
     * halves (keep offset/block) so next-addr=0 (end of chain) and next-amount=0. (validate_mfold's rfile
     * path already did this; doing it here makes every ork_i8_npu_replay caller — incl. replay_mm_i8 — safe.) */
    for(int k=0;k+1<rn;k+=2){ unsigned o=rc[k]&0xffff, b=(rc[k+1]>>16)&0xffff;
        if(b==0x101 && (o==0x0010||o==0x0014)){ rc[k]&=0xffff; rc[k+1]&=0xffff0000u; } }
    orki_setrn(rc,rn,RK_CNA_FEATURE_DATA_ADDR,(uint32_t)A.dma);            /* A (activations) */
    orki_setrn(rc,rn,RK_CNA_WEIGHT_DATA_ADDR,(uint32_t)B.dma);            /* B (weights) */
    orki_setrn(rc,rn,RK_DPU_DST_BASE_ADDR,(uint32_t)Cc.dma);          /* C output IOVA (the ONLY output address; 0x40c0/SURFACE_ADD is a config value, NOT an address — do not patch it) */
    if(getenv("ORK_REPLAY_IOVA")) fprintf(stderr,"[replay-iova] A=%#x B=%#x C=%#x RC=%#x task=%#x\n",(unsigned)A.dma,(unsigned)B.dma,(unsigned)Cc.dma,(unsigned)RCb.dma,(unsigned)c->task.dma);  /* #39: observe IOVA stability across runs (intermittency diag) */
    orki_bsync(fd,&RCb,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk);
    tk->enable_mask=0xd; tk->int_mask=0x300; tk->int_clear=0x1ffff; tk->regcfg_amount=108; tk->regcmd_addr=RCb.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    int ret=-1; struct rknpu_submit sub;
    #define _RSUB() do{ memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=1; sub.task_obj_addr=c->task.obj; \
        sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1; sub.timeout=orki_mm_timeout_ms(); sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; }while(0)
    /* #39 intermittency probe: the raw mfold replay bypasses ork_npu_enter, so it never does the
     * mode-ENTRY ACT_RESET that the normal path fires when the schedule/precision mode changes. The
     * mfold is an unusual-mode regcmd entered right after an int8-normal warmup — hypothesis: the
     * missing entry-reset leaves stale NPU state that intermittently stalls/wedges the fold. Gate a
     * reset here so we can A/B whether it makes the submit deterministic (one-time ~107ms, before the
     * timed loop, so it doesn't taint the per-submit timing). */
    if(getenv("ORK_REPLAY_RESET")){ struct rknpu_action a={.flags=RKNPU_ACT_RESET,.value=0}; ioctl(fd,DRM_IOCTL_RKNPU_ACTION,&a); }
    _RSUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ goto done; }          /* warm */
    orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_FROM_DEVICE);
    if(Cout) memcpy(Cout,Cc.cpu,(size_t)M*N*4);           /* computed output for correctness check vs captured C */
    { double t0=ork_now_us();
      for(int i=0;i<iters;i++){ _RSUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ goto done; } }
      if(us) *us=(ork_now_us()-t0)/(iters>0?iters:1); ret=0; }
    #undef _RSUB
done:
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&B); orki_bdestroy(fd,&Cc); orki_bdestroy(fd,&RCb); return ret;
}

/* #39 A-layout solver: submit a FIXED (captured) regcmd for `nvar` A-variants, reusing ONE buffer set so the
 * IOVA is stable across submits. The fresh-alloc-per-call path (ork_i8_npu_replay called N times) intermittently
 * wedges on the fold; buffer reuse is the safe pattern (cf. replay_mm_i8's iters loop, which never wedged).
 * Avar = nvar A-images, each `astride` bytes; Bdata = shared weight; Couts = nvar contiguous M*N int32 results.
 * A warm submit (variant 0) precedes the measured loop. Returns 0/ok, -2 bad shape, <0 on submit error. */
int ork_i8_npu_replay_sweep(ork_npu *c, const uint32_t *regcmd, int rn, int M, int K, int N,
        const int8_t *Avar, int nvar, int astride, const int8_t *Bdata, int Bbytes, int32_t *Couts){
    int fd=c->fd; if(fd<0) return -3; if(rn<8||rn>2048||M<1||(K%32)||(N%16)||nvar<1) return -2;
    int dom=c->dom_active;
    size_t bsz=(Bbytes>(int)((size_t)K*N))?(size_t)Bbytes:(size_t)K*N;
    size_t aszg=(size_t)M*K*8+(1u<<20), bszg=bsz*8+(1u<<20), cszg=(size_t)M*N*4*8+65536;
    struct buf A =orki_bcreate(fd,aszg,0x403,dom);          if(!A.cpu)  return -2;
    struct buf B =orki_bcreate(fd,bszg,0x403,dom);          if(!B.cpu) {orki_bdestroy(fd,&A);return -2;}
    struct buf Cc=orki_bcreate(fd,cszg,0x403,dom);          if(!Cc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);return -2;}
    struct buf RCb=orki_bcreate(fd,(size_t)rn*4,0x403,dom); if(!RCb.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&Cc);return -2;}
    memset(B.cpu,0,bszg); if(Bdata)memcpy(B.cpu,Bdata,(Bbytes>0)?(size_t)Bbytes:(size_t)K*N); else memset(B.cpu,1,bsz);
    orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t *rc=(uint32_t*)RCb.cpu; memcpy(rc,regcmd,(size_t)rn*4);
    for(int k=0;k+1<rn;k+=2){ unsigned o=rc[k]&0xffff,b=(rc[k+1]>>16)&0xffff; if(b==0x101&&(o==0x0010||o==0x0014)){rc[k]&=0xffff;rc[k+1]&=0xffff0000u;} }
    orki_setrn(rc,rn,RK_CNA_FEATURE_DATA_ADDR,(uint32_t)A.dma);
    orki_setrn(rc,rn,RK_CNA_WEIGHT_DATA_ADDR,(uint32_t)B.dma);
    orki_setrn(rc,rn,RK_DPU_DST_BASE_ADDR,(uint32_t)Cc.dma);
    orki_bsync(fd,&RCb,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk);
    tk->enable_mask=0xd; tk->int_mask=0x300; tk->int_clear=0x1ffff; tk->regcfg_amount=108; tk->regcmd_addr=RCb.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    int ret=-1; struct rknpu_submit sub;
    #define _SSUB() do{ memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=1; sub.task_obj_addr=c->task.obj; \
        sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1; sub.timeout=orki_mm_timeout_ms(); sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; }while(0)
    size_t acp=(size_t)astride; if(acp>aszg)acp=aszg;
    memset(A.cpu,0,aszg); memcpy(A.cpu,Avar,acp); orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);   /* warm w/ variant 0 */
    _SSUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)) goto sdone;
    for(int v=0; v<nvar; v++){
        memset(A.cpu,0,aszg); memcpy(A.cpu, Avar+(size_t)v*astride, acp); orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);
        memset(Cc.cpu,0,(size_t)M*N*4); orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_TO_DEVICE);
        _SSUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)) goto sdone;
        orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_FROM_DEVICE);
        memcpy(Couts+(size_t)v*M*N, Cc.cpu, (size_t)M*N*4);
    }
    ret=0;
    #undef _SSUB
sdone:
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&B); orki_bdestroy(fd,&Cc); orki_bdestroy(fd,&RCb); return ret;
}


int ork_i8_npu_probe_silu(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,int8_t *C,double *us){
    /* g2 captured register set (a known-good replay): R=0x51aa/2^0x14, bias=-97, idx_off, 0x4068 field. */
    return ork_i8_npu_probe_silu_cfg(c,M,K,N,A,B,0x51aa,0x14,0xffffff9fu,0xffffc000u,0x56391100u,NULL,0,C,us);
}

int ork_i8_npu_probe_silu_cfg(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,
                              int r_mult,int r_shift,uint32_t out_bias,uint32_t idx_off,uint32_t cfg4068,
                              const int16_t *lut,int nlut,int8_t *C,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64) return -2;
    /* MULTI-DOMAIN: the FFN chain calls this per-layer fused-SiLU LUT-calibration probe with c->dom_active set
     * to the layer's (non-0) domain; the probe's buffers + submits must MATCH it, else submitting
     * iommu_domain_id=0 while c->dom_active!=0 WEDGES (errno 110) — the same hazard already fixed for the i16
     * SiLU probes (ork_i16_npu_probe_silu_std). dom==0 for every single-domain caller => behavior-preserving.
     * ORK_SILU_DOM0=1 forces dom0 (A/B / disable the fix). */
    int dom = getenv("ORK_SILU_DOM0") ? 0 : c->dom_active;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,dom); if(!W.cpu) return -2;
    int NN=N/32,KT=K/32; int8_t*bb=W.cpu;
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)];
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)M*N,0x403,dom); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}
    struct buf Lrc=orki_bcreate(fd,(size_t)REGCMD_SILU_LUT_N*4,0x403,dom); if(!Lrc.cpu){orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);return -2;} /* LUT-load regcmd */
    struct buf Lsc=orki_bcreate(fd,4096,0x403,dom); if(!Lsc.cpu){orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);return -2;} /* LUT-load scratch (reg 0x4020) */
    int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    /* Prime against a fresh-buffer stale-read/wedge — but ONLY when the NPU isn't already int8-warm. The chain
     * builds the fused-SiLU LUT via this probe once per layer during prep; an unconditional ~107ms ACT_RESET
     * fired ~once/layer (28 cold resets on the first forward pass, hurting cold TTFT/pp128). When int8-live the
     * probe's own 3-rep warmup (below) flushes stale reads, so the reset is redundant. ORK_PROBE_RESET=1 forces
     * it (fallback). Standalone/cold callers still reset. LUT calibration is unaffected (validated bit-exact). */
    if(!ORK_I8_LIVE(c->last_dt) || getenv("ORK_PROBE_RESET")) orki_act(fd,RKNPU_ACT_RESET,0);

    /* ---- submit 1: LUT-load (enable=0x18, regcfg=1097) — streams the silu LUT into PPU SRAM ---- */
    memcpy(Lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
    orki_setrn((uint32_t*)Lrc.cpu,REGCMD_SILU_LUT_N,RK_DPU_DST_BASE_ADDR,(uint32_t)Lsc.dma); /* patch the one output addr */
    if(lut){ uint32_t*lr=(uint32_t*)Lrc.cpu; int j=0;   /* override LUT data: stream lut[] into the 0x4104 writes in order */
        for(int k=0;k+1<REGCMD_SILU_LUT_N;k+=2){ if((lr[k]&0xffff)==0x4104){ int32_t v=(j<nlut)?(int32_t)lut[j]:0; j++;
            lr[k]=0x4104|((uint32_t)(v&0xffff)<<16); lr[k+1]=(0x1001u<<16)|(((uint32_t)v>>16)&0xffff); } } }
    orki_bsync(fd,&Lrc,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
      t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=1097; t->regcmd_addr=Lrc.dma;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.timeout=orki_mm_timeout_ms();sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc); return -1; }
    }

    /* ---- submit 2: matmul compute (enable=0x1d, regcfg=108) reading the resident LUT ---- */
    uint32_t rc[REGCMD_I8_N];
    orki_i8_synth(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    orki_i8_set_silu(rc,N,0,r_mult,r_shift,out_bias,idx_off,cfg4068);
    if(getenv("ORK_FUSED_DUMP")){ for(int k=0;k+1<REGCMD_I8_N;k+=2) fprintf(stderr,"PROBE %04x=%08x l=%04x\n",rc[k]&0xffff,((rc[k]>>16)&0xffff)|((rc[k+1]&0xffff)<<16),rc[k+1]>>16); }
    struct buf extra[2]={W,O};
    if(orki_validate_regcmd("probe_i8_silu",c,rc,REGCMD_I8_N,NULL,extra,2)){ orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
      t->enable_mask=0x1d; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=108; t->regcmd_addr=c->regcmd.dma;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      int ok=-1; double t1=0;
      for(int rep=0;rep<3;rep++){ sub.timeout=orki_mm_timeout_ms(); double t0=ork_now_us();
          if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ ok=-1; break; }
          orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
      if(ok==0){ memcpy(C,O.cpu,(size_t)M*N); if(us)*us=t1; }
      orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);
      return ok;
    }
}
