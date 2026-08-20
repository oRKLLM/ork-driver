/* npu/i16/probe.c — int16 probes and LUT replays.
 *
 * Part of the i16 datapath; shared declarations in npu/i16/i16.h. Split out of npu/i16.c for the
 * same reason i8 is a folder: one datapath, sized for reading. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include "ork_regs.h"
#include "regcmd_array_4x32x16.h"
#include "regcmd_i8.h"
#include "regcmd_silu.h"
#include "regcmd_ewmul.h"
#include "npu/internal.h"
#include "npu/core.h"
#include "npu/i16/i16.h"

int ork_npu_probe_i16_out(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,
                          int mult,int shift,int16_t *C,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    int NN=N/32,KT=K/32; int8_t*bb=W.cpu;     /* int8 tile layout [Ntile][Ktile][32][32], full K */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)];
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    size_t obytes=getenv("ORK_I16_OBYTES")?strtoul(getenv("ORK_I16_OBYTES"),0,0):(size_t)M*N*2;  /* enlarge to capture a strided layout */
    if(obytes<(size_t)M*N*2) obytes=(size_t)M*N*2;
    struct buf O=orki_bcreate(fd,obytes,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}  /* int16 output */
    memset(O.cpu,0,obytes); orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I8_N];
    orki_synth_i8(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    if(getenv("ORK_MM_F16OUT")) orki_set_f16_out(rc,N,0);          /* SHIM test: int8 matmul -> fp16 OUT_CVT (2-byte) */
    else if(getenv("ORK_MM_I32OUT")) { /* CONTROL: skip set_i16_out -> synth_i8's default int32 output (works standalone) */ }
    else                        orki_set_i16_out(rc,N,0,mult,shift); /* rewrite output stage: int32 -> int16 requantize */
    /* TOGGLE SWEEP: restore individual output-stage regs to their int32 (completing) values to isolate the
     * WDMA terminal-count stall. Each ORK_MM_R<reg>=<hex> overrides one reg AFTER set_i16_out. */
    { const char*e;
      if((e=getenv("ORK_MM_R4010"))) orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_PRECISION,(uint32_t)strtoul(e,0,0));
      if((e=getenv("ORK_MM_R4038"))) orki_setrn(rc,REGCMD_I8_N,RK_DPU_DATA_CUBE_NOTCH,(uint32_t)strtoul(e,0,0));
      if((e=getenv("ORK_MM_R4050"))) orki_setrn(rc,REGCMD_I8_N,RK_DPU_BS_OW_CFG,(uint32_t)strtoul(e,0,0));
      if((e=getenv("ORK_MM_R4084"))) orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)strtoul(e,0,0));
      if((e=getenv("ORK_MM_R4088"))) orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)strtoul(e,0,0));
      if((e=getenv("ORK_MM_R40c0"))) orki_setrn(rc,REGCMD_I8_N,RK_DPU_SURFACE_ADD,(uint32_t)strtoul(e,0,0)); }
    if(getenv("ORK_MM_DUMPRC")){ const char*tag=getenv("ORK_MM_I32OUT")?"I32":"I16"; /* dump the assembled 0x40xx output stage for diffing */
        for(int k=0;k+1<REGCMD_I8_N;k+=2){ uint32_t r=rc[k]&0xffff, v=((rc[k]>>16)&0xffff)|((rc[k+1]&0xffff)<<16);
            if(r>=0x4000 && r<0x4100) fprintf(stderr,"[%s] 0x%04x = 0x%08x\n",tag,r,v); } }
    struct buf extra[2] = {W, O};
    if (orki_validate_regcmd("probe_i16_out", c, rc, REGCMD_I8_N, NULL, extra, 2)) { orki_bdestroy(fd,&W); orki_bdestroy(fd,&O); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0;
    for(int rep=0;rep<3;rep++){ sub.timeout=orki_mm_timeout_ms();
        double t0=ork_now_us();
        if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; break; }
        orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    if(ok==0){ memcpy(C,O.cpu,(size_t)M*N*2); if(us)*us=t1;
        if(getenv("ORK_MM_DUMPOUT")){ /* LAYOUT MAP: caller made values distinct; print int16-slot -> value for every nonzero slot in the (enlarged) O */
            const int16_t*oc=O.cpu; long ns=(long)(obytes/2); int shown=0; long firstrow_end=-1, secondrow_start=-1; int prev=-1;
            fprintf(stderr,"[i16map] obytes=%zu (%ld slots), M=%d N=%d — nonzero slots:\n",obytes,ns,M,N);
            for(long i=0;i<ns;i++){ int v=oc[i]; if(v!=0){ if(shown<64) fprintf(stderr,"  [%ld]=%d",i,v);
                if(prev>=0 && v<prev && secondrow_start<0){ firstrow_end=i-1; secondrow_start=i; } prev=v; shown++; if(shown%8==0&&shown<=64)fprintf(stderr,"\n"); } }
            fprintf(stderr,"\n[i16map] total nonzero=%d  (row-wrap at slot ~%ld => row byte-stride ~%ld)\n",shown,secondrow_start,secondrow_start*2); } }
    else if(getenv("ORK_MM_DUMPOUT")){ /* partial-write signature: how far did the WDMA get before the stall? */
        orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); int16_t*oc=O.cpu; long last=-1; int tot=0,rows=0;
        for(long i=0;i<(long)M*N;i++) if(oc[i]){ tot++; if(i>last)last=i; }
        for(int m=0;m<M;m++){ int rnz=0; for(int n=0;n<N;n++) if(oc[(size_t)m*N+n])rnz++; if(rnz)rows++; }
        fprintf(stderr,"[dumpout] M=%d N=%d: nonzero=%d/%d  rows-with-data=%d/%d  last-nz-elem=%ld (row %ld/%d)\n",
                M,N,tot,M*N,rows,M,last,last<0?-1:last/N,M); }
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);
    return ok;
}

int ork_npu_replay_lut_i16(ork_npu *c,const uint32_t *regcmd,int rn,const int16_t *lut,int nlut,
                           const int16_t *in,int M,int N,int16_t *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)||rn>REGCMD_SILU_STD_I16_N) return -2;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)
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
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) *(int16_t*)((char*)A.cpu+EWCUBEH(m,n))=in[m*N+n];
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    memcpy(Lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
    orki_setrn((uint32_t*)Lrc.cpu,REGCMD_SILU_LUT_N,RK_DPU_DST_BASE_ADDR,(uint32_t)Lsc.dma);
    { uint32_t*lr=(uint32_t*)Lrc.cpu; int j=0;
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
    uint32_t rc[REGCMD_SILU_STD_I16_N]; memset(rc,0,sizeof rc); memcpy(rc,regcmd,(size_t)rn*4);
    orki_set_mul_geom(rc,rn,M,N);
    orki_setrn(rc,rn,RK_SDP_5040,0); orki_setrn(rc,rn,RK_SDP_5038,0);
    orki_setrn(rc,rn,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,rn,RK_SDP_5018,(uint32_t)A.dma);
    memcpy(c->regcmd.cpu,rc,(size_t)rn*4); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk);
    tk->enable_mask=0x18; tk->int_mask=0x300; tk->int_clear=0x1ffff; tk->regcfg_amount=(uint32_t)(rn/2-4); tk->regcmd_addr=c->regcmd.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms(); double t0=ork_now_us();
    /* ORK_I16_DUMP: dump the exact submit context so the WEDGING chain submit can be diffed byte-for-byte
     * against the CLEAN probe submit of the same shape (in-model instrumentation, #35). */
    if(getenv("ORK_I16_DUMP")){ static long n=0;
        fprintf(stderr,"[i16dump #%ld] M=%d N=%d dom=%d dom_active=%d core=0x%x | A.dma=0x%llx A.dom=%d O.dma=0x%llx O.dom=%d Lrc.dma=0x%llx(d%d) Lsc.dma=0x%llx(d%d) | regcmd.dma=0x%llx regcfg=%u task.obj=0x%llx last_dt=%d warmed=%d\n",
            ++n,M,N,dom,c->dom_active,sub.core_mask,
            (unsigned long long)A.dma,A.domain,(unsigned long long)O.dma,O.domain,
            (unsigned long long)Lrc.dma,Lrc.domain,(unsigned long long)Lsc.dma,Lsc.domain,
            (unsigned long long)c->regcmd.dma,tk->regcfg_amount,(unsigned long long)c->task.obj,c->last_dt,c->warmed);
        fflush(stderr); }
    if(!orki_rknpu_submit_ioctl(fd,&sub,dom)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[m*N+n]=*(int16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(us)*us=t1; }
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);
    #undef EWCUBEH
    return ok;
}

int ork_npu_probe_silu_std_i16(ork_npu *c,const int16_t *in,int M,int N,
                               int r_mult,int r_shift,uint32_t out_bias,uint32_t idx_off,
                               uint32_t cfg4064,uint32_t cfg4068,const int16_t *lut,int nlut,
                               int16_t *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;
    if(r_mult<0||r_mult>0x7fff||r_shift<0||r_shift>31) return -2;
    orki_last_op="silu_i16_op"; orki_last_K=M; orki_last_N=N; orki_last_wdom=0; orki_last_import=0;   /* accurate wedge telemetry (no validate_regcmd here) */
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)   /* int16 atom-8, 2-byte, surf_stride=M*16 */
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
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) *(int16_t*)((char*)A.cpu+EWCUBEH(m,n))=in[m*N+n];
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);

    /* Build the LUT-load regcmd content + the activation regcmd ONCE. */
    memcpy(Lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
    orki_setrn((uint32_t*)Lrc.cpu,REGCMD_SILU_LUT_N,RK_DPU_DST_BASE_ADDR,(uint32_t)Lsc.dma);
    if(lut){ uint32_t*lr=(uint32_t*)Lrc.cpu; int j=0;
        for(int k=0;k+1<REGCMD_SILU_LUT_N;k+=2){ if((lr[k]&0xffff)==0x4104){ int32_t v=(j<nlut)?(int32_t)lut[j]:0; j++;
            lr[k]=0x4104|((uint32_t)(v&0xffff)<<16); lr[k+1]=(0x1001u<<16)|(((uint32_t)v>>16)&0xffff); } } }
    orki_bsync(fd,&Lrc,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t rc[REGCMD_SILU_STD_I16_N]; memcpy(rc,REGCMD_SILU_STD_I16,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_SILU_STD_I16_N,M,N);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_SDP_5040,0);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_SDP_5038,0);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_SDP_5018,(uint32_t)A.dma);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)r_mult);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)r_shift);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_OFFSET,out_bias);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_R4110,idx_off);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_BN_ALU_CFG,cfg4064);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_BN_MUL_CFG,cfg4068);
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);

    /* #35: the int16 silu is a STANDALONE pure-SDP op (enable_mask 0x18). IN A CHAIN, its submit wedges in
     * the chain CONTEXT (works standalone at every shape, i16_shape_probe). NO userspace mitigation fixes
     * it: ping-pong OFF, ACT_RESET, a pipeline-DRAIN (tiny single-core matmul first), and reset+retry ALL
     * fail — even the calib's 64x64 probe wedges on every attempt after soft-resets. Only the kernel
     * self-heal limps it through (correct output, ~1 reset/layer). And the FUSED int16 output is CLOSED
     * (int8-only, ⚠ note below), so the standalone op is the only int16 path. => int16 silu NOT viable
     * in-chain with current NPU understanding — needs kernel-level reset or a deeper pipeline fix. Left as
     * the RE artifact (ORK_FFN_SILU_I16). Shipped coherent path is all-CPU-silu. ping-pong OFF (LUT-op). */
    /* #35 RESOLVED: the per-call ACT_RESET was pure OVERHEAD, not a wedge guard. The in-chain "wedge" was a
     * MISREAD — dmesg "RKNPU: soft reset" counts the DELIBERATE ACT_RESET (ORK_DEBUG_RESET: 23 act calls ->
     * 27 dmesg entries), not hardware wedges (errno=110 count = 0 in a full chain run). Removing it: int16
     * chain prefill 28->68 tok/s, PPL 19.02 unchanged, 0 real wedges. Default OFF; ORK_I16_RESET re-enables. */
    if(getenv("ORK_I16_RESET")) orki_act(fd,RKNPU_ACT_RESET,0);
    /* MODE-TRANSITION FIX (mode_probe RE): this LUT op memsets the SHARED c->task to its own SDP descriptor
     * (regcfg_amount 1097 then 69, enable_mask 0x18) and previously left it that way. The single-core matmul
     * path (orki_run()/submit1) does NOT rebuild c->task — it relies on the init value (regcfg_amount=108,
     * enable_mask=0xd, regcmd_addr=c->regcmd.dma) persisting. So a later SINGLE-CORE matmul (e.g. the SSD
     * CumBA bmm, N=16) submitted a 108-word matmul regcmd under this stale 69-reg/0x18 SDP task -> the NPU
     * dispatched no task (task counter 0x0) -> errno=110 wedge that ACT_RESET can't clear (poisoned software
     * descriptor, not HW state). The plain 2-input SDP ops (ewmul/add) never wedged because they already
     * save+restore these fields. Save them here and restore on every exit, exactly like ewmul. */
    struct rknpu_task *tk0=(struct rknpu_task*)c->task.cpu;
    uint32_t sv_amt=tk0->regcfg_amount, sv_en=tk0->enable_mask; uint64_t sv_addr=tk0->regcmd_addr;
    #define SILU_RESTORE_TASK() do{ tk0->regcfg_amount=sv_amt; tk0->enable_mask=sv_en; tk0->regcmd_addr=sv_addr; \
        orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE); }while(0)
    { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
      t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=1097; t->regcmd_addr=Lrc.dma;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x1;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.timeout=orki_ew_timeout_ms();sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ SILU_RESTORE_TASK(); orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc); return -1; } }
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk);
    tk->enable_mask=0x18; tk->int_mask=0x300; tk->int_clear=0x1ffff; tk->regcfg_amount=69; tk->regcmd_addr=c->regcmd.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x1;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms(); double t0=ork_now_us();
    /* ORK_I16_DUMP: dump the exact submit context so the WEDGING chain submit can be diffed byte-for-byte
     * against the CLEAN probe submit of the same shape (in-model instrumentation, #35). */
    if(getenv("ORK_I16_DUMP")){ static long n=0;
        fprintf(stderr,"[i16dump #%ld] M=%d N=%d dom=%d dom_active=%d core=0x%x | A.dma=0x%llx A.dom=%d O.dma=0x%llx O.dom=%d Lrc.dma=0x%llx(d%d) Lsc.dma=0x%llx(d%d) | regcmd.dma=0x%llx regcfg=%u task.obj=0x%llx last_dt=%d warmed=%d\n",
            ++n,M,N,dom,c->dom_active,sub.core_mask,
            (unsigned long long)A.dma,A.domain,(unsigned long long)O.dma,O.domain,
            (unsigned long long)Lrc.dma,Lrc.domain,(unsigned long long)Lsc.dma,Lsc.domain,
            (unsigned long long)c->regcmd.dma,tk->regcfg_amount,(unsigned long long)c->task.obj,c->last_dt,c->warmed);
        fflush(stderr); }
    if(!orki_rknpu_submit_ioctl(fd,&sub,dom)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[m*N+n]=*(int16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(us)*us=t1; }
    SILU_RESTORE_TASK();     /* leave the shared c->task as the matmul path expects (see save above) */
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);
    #undef EWCUBEH
    #undef SILU_RESTORE_TASK
    return ok;
}
