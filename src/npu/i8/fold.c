/* npu/i8/fold.c — the weight-resident M-FOLD chain: per-size sub-tile templates, the fold scratch cache, batch/run entrypoints.
 *
 * Part of the int8 (W8A8) datapath, the driver's production path. Lifted verbatim from npu.c by the
 * precision split (MODULARIZE_PLAN.md round 1); interface types in npu/internal.h, substrate in npu/core.h. */
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

int ork_npu_mfold_chain_cap(ork_npu *c, int P, int w, int K, int N, const uint32_t *tile_rc, int trn,
                            const int8_t *Apacked, const int8_t *Bpacked, int32_t *Craw, int iters, double *us){
    int fd=c->fd; if(fd<0) return -3; if(P<1||P>64||w<1||w>64||(K%32)||(N%16)||!tile_rc||trn<108) return -2;
    int dom=c->dom_active;
    size_t tileA=(size_t)w*K, tileC=(size_t)w*N;
    size_t asz=(size_t)P*tileA*8+(1u<<20), bsz=(size_t)K*N*8+(1u<<20), csz=(size_t)P*tileC*4*8+65536;
    struct buf A =orki_bcreate(fd,asz,0x403,dom);            if(!A.cpu)  return -2;
    struct buf B =orki_bcreate(fd,bsz,0x403,dom);            if(!B.cpu) {orki_bdestroy(fd,&A);return -2;}
    struct buf Cc=orki_bcreate(fd,csz,0x403,dom);            if(!Cc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);return -2;}
    struct buf RC=orki_bcreate(fd,(size_t)P*REGCMD_I8_N*4,0x403,dom); if(!RC.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&Cc);return -2;}
    memset(A.cpu,0,asz); memset(B.cpu,0,bsz); memset(Cc.cpu,0,csz);
    memcpy(A.cpu,Apacked,(size_t)P*tileA); memcpy(B.cpu,Bpacked,(size_t)K*N);
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t *rcbuf=(uint32_t*)RC.cpu;
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu;
    int ncopy = trn<REGCMD_I8_N ? trn : REGCMD_I8_N;   /* copy the FULL tile incl. its descriptor region, like ork_npu_replay_i8 */
    for(int t=0;t<P;t++){
        uint32_t rc[REGCMD_I8_N]; memset(rc,0,sizeof rc);
        memcpy(rc, tile_rc, (size_t)ncopy*4);                                                  /* captured tile verbatim */
        /* neutralize the captured PC-chain descriptor IN PLACE — zero only the VALUE halves of the
         * 0x0010(next-addr)/0x0014(next-amount) block-0x101 entries, KEEPING the reg-ids. This is EXACTLY
         * what ork_npu_replay_i8 does (validate_layout proved that single-task path 0/9728 bit-exact);
         * zeroing the whole words destroys the 0x0014 terminator and the lone task hangs. */
        for(int k=0;k+1<REGCMD_I8_N;k+=2){ unsigned o=rc[k]&0xffff, b=(rc[k+1]>>16)&0xffff;
            if(b==0x101 && (o==0x0010||o==0x0014)){ rc[k]&=0xffff; rc[k+1]&=0xffff0000u; } }
        int wreuse = (t>0) && getenv("ORK_MFOLD_WREUSE");
        orki_setrn(rc,REGCMD_I8_N,RK_CNA_FEATURE_DATA_ADDR,(uint32_t)(A.dma + (uint64_t)t*tileA));   /* this tile's A rows */
        if(!wreuse) orki_setrn(rc,REGCMD_I8_N,RK_CNA_WEIGHT_DATA_ADDR,(uint32_t)B.dma);               /* shared weight; on a REUSE tile leave it (fetch is skipped; re-writing may trigger release) */
        orki_setrn(rc,REGCMD_I8_N,RK_DPU_DST_BASE_ADDR,(uint32_t)(Cc.dma + (uint64_t)t*tileC*4));     /* this tile's C */
        /* #39 NOVEL: set CNA_CBUF_CON0 WEIGHT_REUSE (0x1040 bit13, per rocket_registers.h) on non-loader tiles so
         * the HW reuses the weight tile 0 left resident in CBUF and SKIPS the re-DMA — a HW feature rkllm never
         * uses (bit13=0 in all captured tiles). ORK_MFOLD_WREUSE=1 to try; tile 0 loads normally. */
        if(wreuse){
            uint32_t cur=0; for(int k=0;k+1<REGCMD_I8_N;k+=2) if((rc[k]&0xffff)==0x1040 && ((rc[k+1]>>16)&0xffff)==0x201){ cur=rc[k]>>16; break; }
            orki_setrn(rc,REGCMD_I8_N,RK_CNA_CBUF_CON0, cur|0x2000u);
        }
        /* #39 EXPERIMENT: append "inherited" reg-writes a mid-chain big-M tile OMITS (it relies on earlier tasks
         * having set them). ORK_MFOLD_INJECT="blk:reg:val,..." appends them after the 108-reg tile, bumping
         * regcfg_amount, to test whether a big-M tile can be made SELF-CONTAINED (run standalone). */
        int nreg=108;
        { const char*inj=getenv("ORK_MFOLD_INJECT");
          if(inj){ char b[256]; strncpy(b,inj,sizeof b-1); b[sizeof b-1]=0;
            for(char*p=strtok(b,",");p && 2*nreg+8<REGCMD_I8_N;p=strtok(NULL,",")){ unsigned bl,rg,vl;
              if(sscanf(p,"%x:%x:%x",&bl,&rg,&vl)==3){ rc[2*nreg]=(vl<<16)|(rg&0xffff); rc[2*nreg+1]=((bl&0xffff)<<16); nreg++; } } } }
        int d=2*nreg; uint32_t namt=(uint32_t)((nreg+3)/2);
        if(t<P-1){ uint64_t nxt = RC.dma + (uint64_t)(t+1)*REGCMD_I8_N*4;                        /* link to next task */
            rc[d]=0x0010|((uint32_t)(nxt&0xffff)<<16); rc[d+1]=(0x0101u<<16)|((uint32_t)(nxt>>16)&0xffff);
            rc[d+2]=0x0014|(namt<<16);                 rc[d+3]=(0x0101u<<16)|0;
        } else { rc[d]=0x0010; rc[d+1]=(0x0101u<<16); rc[d+2]=0x0014; rc[d+3]=(0x0101u<<16); } /* terminal: reg-ids present, val 0 */
        memcpy(rcbuf + (size_t)t*REGCMD_I8_N, rc, sizeof rc);
        memset(&tk[t],0,sizeof tk[t]);
        tk[t].enable_mask=0xd; tk[t].int_mask=0x300; tk[t].int_clear=0x1ffff;
        tk[t].regcfg_amount=(uint32_t)nreg; tk[t].regcmd_addr=RC.dma + (uint64_t)t*REGCMD_I8_N*4;
    }
    orki_bsync(fd,&RC,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    int ret=-1; struct rknpu_submit sub;
    #define _CAPSUB() do{ memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=(uint32_t)P; \
        sub.task_obj_addr=c->task.obj; sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1; sub.timeout=orki_mm_timeout_ms(); \
        sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)P}; }while(0)
    _CAPSUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ goto capdone; }        /* warm */
    orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_FROM_DEVICE);
    if(Craw) memcpy(Craw,Cc.cpu,(size_t)P*tileC*4);
    { double t0=ork_now_us();
      for(int i=0;i<iters;i++){ _CAPSUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ goto capdone; } }
      if(us)*us=(ork_now_us()-t0)/(iters>0?iters:1); ret=0; }
    #undef _CAPSUB
capdone:
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&Cc);orki_bdestroy(fd,&RC); return ret;
}

int ork_npu_mfold_chain_multi(ork_npu *c, int P, int w, int K, int N, const uint32_t *tiles, int rn,
                              int iters, double *us){
    int fd=c->fd; if(fd<0) return -3; if(P<1||P>64||w<1||w>64||(K%32)||(N%16)||!tiles||rn<108||rn>512) return -2;
    int dom=c->dom_active;
    size_t tileA=(size_t)w*K, tileC=(size_t)w*N;
    size_t asz=(size_t)P*tileA*8+(1u<<20), bsz=(size_t)K*N*8+(1u<<20), csz=(size_t)P*tileC*4*8+65536;
    struct buf A =orki_bcreate(fd,asz,0x403,dom);            if(!A.cpu)  return -2;
    struct buf B =orki_bcreate(fd,bsz,0x403,dom);            if(!B.cpu) {orki_bdestroy(fd,&A);return -2;}
    struct buf Cc=orki_bcreate(fd,csz,0x403,dom);            if(!Cc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);return -2;}
    struct buf RC=orki_bcreate(fd,(size_t)P*REGCMD_I8_N*4,0x403,dom); if(!RC.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&Cc);return -2;}
    memset(A.cpu,0,asz); memset(B.cpu,0,bsz); memset(Cc.cpu,0,csz);
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t *rcbuf=(uint32_t*)RC.cpu;
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu;
    int ncopy = rn<REGCMD_I8_N ? rn : REGCMD_I8_N;
    for(int t=0;t<P;t++){
        uint32_t rc[REGCMD_I8_N]; memset(rc,0,sizeof rc);
        memcpy(rc, tiles+(size_t)t*rn, (size_t)ncopy*4);                                       /* this task's captured regcmd */
        for(int k=0;k+1<REGCMD_I8_N;k+=2){ unsigned o=rc[k]&0xffff, b=(rc[k+1]>>16)&0xffff;    /* neutralize captured descriptor */
            if(b==0x101 && (o==0x0010||o==0x0014)){ rc[k]&=0xffff; rc[k+1]&=0xffff0000u; } }
        orki_setrn(rc,REGCMD_I8_N,RK_CNA_FEATURE_DATA_ADDR,(uint32_t)(A.dma + (uint64_t)t*tileA));
        orki_setrn(rc,REGCMD_I8_N,RK_CNA_WEIGHT_DATA_ADDR,(uint32_t)B.dma);                          /* shared resident weight */
        orki_setrn(rc,REGCMD_I8_N,RK_DPU_DST_BASE_ADDR,(uint32_t)(Cc.dma + (uint64_t)t*tileC*4));
        if(t<P-1){ uint64_t nxt = RC.dma + (uint64_t)(t+1)*REGCMD_I8_N*4;
            rc[216]=0x0010|((uint32_t)(nxt&0xffff)<<16); rc[217]=(0x0101u<<16)|((uint32_t)(nxt>>16)&0xffff);
            rc[218]=0x0014|(0x0037u<<16);                rc[219]=(0x0101u<<16)|0;
        } else { rc[216]=0x0010; rc[217]=(0x0101u<<16); rc[218]=0x0014; rc[219]=(0x0101u<<16); }
        memcpy(rcbuf + (size_t)t*REGCMD_I8_N, rc, sizeof rc);
        memset(&tk[t],0,sizeof tk[t]);
        tk[t].enable_mask=0xd; tk[t].int_mask=0x300; tk[t].int_clear=0x1ffff;
        tk[t].regcfg_amount=108; tk[t].regcmd_addr=RC.dma + (uint64_t)t*REGCMD_I8_N*4;
    }
    orki_bsync(fd,&RC,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    int ret=-1; struct rknpu_submit sub;
    #define _MSUB() do{ memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=(uint32_t)P; \
        sub.task_obj_addr=c->task.obj; sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1; sub.timeout=orki_mm_timeout_ms(); \
        sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)P}; }while(0)
    _MSUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ goto mdone; }
    { double t0=ork_now_us();
      for(int i=0;i<iters;i++){ _MSUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ goto mdone; } }
      if(us)*us=(ork_now_us()-t0)/(iters>0?iters:1); ret=0; }
    #undef _MSUB
mdone:
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&Cc);orki_bdestroy(fd,&RC); return ret;
}

int ork_npu_mfold_chain_v(ork_npu *c, int P, const int *ws, int K, int N, const uint32_t *tiles, int rn,
                          const int8_t *Apacked, const int8_t *Bpacked, int32_t *Craw, int wreuse, int iters, double *us){
    int fd=c->fd; if(fd<0) return -3; if(P<1||P>64||(K%32)||(N%16)||!tiles||rn<108||rn>512) return -2;
    int dom=c->dom_active;
    size_t Aoff[65], Coff[65], atot=0, ctot=0;   /* per-tile byte/elem offsets in the concat packing */
    for(int t=0;t<P;t++){ Aoff[t]=atot; Coff[t]=ctot; atot+=(size_t)ws[t]*K; ctot+=(size_t)ws[t]*N; }
    size_t asz=atot*8+(1u<<20), bsz=(size_t)K*N*8+(1u<<20), csz=ctot*4*8+65536;
    struct buf A =orki_bcreate(fd,asz,0x403,dom);            if(!A.cpu)  return -2;
    struct buf B =orki_bcreate(fd,bsz,0x403,dom);            if(!B.cpu) {orki_bdestroy(fd,&A);return -2;}
    struct buf Cc=orki_bcreate(fd,csz,0x403,dom);            if(!Cc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);return -2;}
    struct buf RC=orki_bcreate(fd,(size_t)P*REGCMD_I8_N*4,0x403,dom); if(!RC.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&Cc);return -2;}
    memset(A.cpu,0,asz); memset(B.cpu,0,bsz); memset(Cc.cpu,0,csz);
    memcpy(A.cpu,Apacked,atot); memcpy(B.cpu,Bpacked,(size_t)K*N);
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t *rcbuf=(uint32_t*)RC.cpu; struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu;
    int ncopy = rn<REGCMD_I8_N ? rn : REGCMD_I8_N;
    for(int t=0;t<P;t++){
        uint32_t rc[REGCMD_I8_N]; memset(rc,0,sizeof rc);
        memcpy(rc, tiles+(size_t)t*rn, (size_t)ncopy*4);
        for(int k=0;k+1<REGCMD_I8_N;k+=2){ unsigned o=rc[k]&0xffff, b=(rc[k+1]>>16)&0xffff;
            if(b==0x101 && (o==0x0010||o==0x0014)){ rc[k]&=0xffff; rc[k+1]&=0xffff0000u; } }
        orki_setrn(rc,REGCMD_I8_N,RK_CNA_FEATURE_DATA_ADDR,(uint32_t)(A.dma + Aoff[t]));
        if(!(wreuse&&t>0)) orki_setrn(rc,REGCMD_I8_N,RK_CNA_WEIGHT_DATA_ADDR,(uint32_t)B.dma);
        orki_setrn(rc,REGCMD_I8_N,RK_DPU_DST_BASE_ADDR,(uint32_t)(Cc.dma + Coff[t]*4));
        if(wreuse&&t>0){ uint32_t cur=0; for(int k=0;k+1<REGCMD_I8_N;k+=2) if((rc[k]&0xffff)==0x1040 && ((rc[k+1]>>16)&0xffff)==0x201){ cur=rc[k]>>16; break; }
            orki_setrn(rc,REGCMD_I8_N,RK_CNA_CBUF_CON0, cur|0x2000u); }
        if(t<P-1){ uint64_t nxt = RC.dma + (uint64_t)(t+1)*REGCMD_I8_N*4;
            rc[216]=0x0010|((uint32_t)(nxt&0xffff)<<16); rc[217]=(0x0101u<<16)|((uint32_t)(nxt>>16)&0xffff);
            rc[218]=0x0014|(0x0037u<<16);                rc[219]=(0x0101u<<16)|0;
        } else { rc[216]=0x0010; rc[217]=(0x0101u<<16); rc[218]=0x0014; rc[219]=(0x0101u<<16); }
        memcpy(rcbuf + (size_t)t*REGCMD_I8_N, rc, sizeof rc);
        memset(&tk[t],0,sizeof tk[t]);
        tk[t].enable_mask=0xd; tk[t].int_mask=0x300; tk[t].int_clear=0x1ffff;
        tk[t].regcfg_amount=108; tk[t].regcmd_addr=RC.dma + (uint64_t)t*REGCMD_I8_N*4;
    }
    orki_bsync(fd,&RC,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    int ret=-1; struct rknpu_submit sub;
    #define _VSUB() do{ memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=(uint32_t)P; \
        sub.task_obj_addr=c->task.obj; sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1; sub.timeout=orki_mm_timeout_ms(); \
        sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)P}; }while(0)
    _VSUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ goto vdone; }
    orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_FROM_DEVICE);
    if(Craw) memcpy(Craw,Cc.cpu,ctot*4);
    { double t0=ork_now_us(); for(int i=0;i<iters;i++){ _VSUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ goto vdone; } }
      if(us)*us=(ork_now_us()-t0)/(iters>0?iters:1); ret=0; }
    #undef _VSUB
vdone:
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&Cc);orki_bdestroy(fd,&RC); return ret;
}

static int ork_fold_submit_all(int fd,int dom,struct buf *TK,const int *gsz,int nc){
    if(nc<=1){ struct ork_fbc_arg a={fd,dom,0,gsz[0],0,&TK[0]}; ork_fbc_thread(&a); return a.rc?-1:0; }
    pthread_t th[3]; struct ork_fbc_arg ar[3]; int made=0,e=0;
    for(int cc=0;cc<nc;cc++){ ar[cc]=(struct ork_fbc_arg){fd,dom,cc,gsz[cc],0,&TK[cc]};
        if(pthread_create(&th[cc],NULL,ork_fbc_thread,&ar[cc])!=0){ e=1; break; } made++; }
    for(int cc=0;cc<made;cc++){ pthread_join(th[cc],NULL); if(ar[cc].rc) e=1; }
    return e?-1:0;
}

int ork_npu_fold_batch(ork_npu *c, int Mtot, int K, int N, int P, const int *row_off,
                       const uint32_t *tiles, int rn, const int8_t *Apacked, const int8_t *Bpacked,
                       int32_t *Craw, int ncore, int iters, double *us){
    int fd=c->fd; if(fd<0) return -3; if(P<1||P>64||Mtot<1||Mtot>256||(K%32)||(N%16)||!tiles||rn<108||rn>512) return -2;
    int dom=c->dom_active;
    /* nc CONCURRENT per-core submits (core_mask=1u<<c each, own task buffer) — the proven ork multi-core path.
     * (The single-ioctl core_mask=0x7 + subcore-split, rkllm's mechanism, HARD-WEDGES ork's setup — see the
     * 2026-07-30 Experiment Log; ork's kernel path wants per-core submits.) Never more cores than tiles. */
    int nc = ncore>=3 ? 3 : (ncore==2 ? 2 : 1); if(nc>P) nc=P; if(nc>c->soc->cores) nc=c->soc->cores;
    /* per-core contiguous task groups: core g runs tiles [gstart[g], gstart[g]+gsz[g]) as its own PC-chain */
    int gstart[3]={0,0,0}, gsz[3]={0,0,0};
    { int base=P/nc, rem=P%nc, s=0; for(int g=0;g<nc;g++){ gsz[g]=base+(g<rem?1:0); gstart[g]=s; s+=gsz[g]; } }
    size_t asz=(size_t)Mtot*K*8+(1u<<20), bsz=(size_t)K*N*8+(1u<<20), csz=(size_t)Mtot*N*4*8+65536;
    struct buf A =orki_bcreate(fd,asz,0x403,dom);            if(!A.cpu)  return -2;
    struct buf B =orki_bcreate(fd,bsz,0x403,dom);            if(!B.cpu) {orki_bdestroy(fd,&A);return -2;}
    struct buf Cc=orki_bcreate(fd,csz,0x403,dom);            if(!Cc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);return -2;}
    struct buf RC=orki_bcreate(fd,(size_t)P*REGCMD_I8_N*4,0x403,dom); if(!RC.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&Cc);return -2;}
    /* one OWN task buffer PER CORE in `dom` (like the multicore run path's c->mtk[i]) — each holds its core's
     * tasks from index 0, so each per-core submit uses subcore_task={0,gsz}. (Using c->task, which lives in the
     * default domain, EINVALs when dom!=default.) */
    struct buf TK[3]={{0}}; for(int g=0;g<nc;g++){ TK[g]=orki_bcreate(fd,(size_t)(gsz[g]+2)*sizeof(struct rknpu_task),0x40b,dom);
        if(!TK[g].cpu){ for(int q=0;q<g;q++) orki_bdestroy(fd,&TK[q]); orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&Cc);orki_bdestroy(fd,&RC); return -2; } }
    memset(A.cpu,0,asz); memset(B.cpu,0,bsz); memset(Cc.cpu,0,csz);
    memcpy(A.cpu,Apacked,(size_t)Mtot*K); memcpy(B.cpu,Bpacked,(size_t)K*N);
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t *rcbuf=(uint32_t*)RC.cpu;
    int ncopy = rn<REGCMD_I8_N ? rn : REGCMD_I8_N;
    for(int t=0;t<P;t++){
        uint32_t rc[REGCMD_I8_N]; memset(rc,0,sizeof rc);
        memcpy(rc, tiles+(size_t)t*rn, (size_t)ncopy*4);
        for(int k=0;k+1<REGCMD_I8_N;k+=2){ unsigned o=rc[k]&0xffff, b=(rc[k+1]>>16)&0xffff;   /* neutralize captured descriptor */
            if(b==0x101 && (o==0x0010||o==0x0014)){ rc[k]&=0xffff; rc[k+1]&=0xffff0000u; } }
        uint64_t roff=(uint64_t)row_off[t]*16;                                                  /* row offset in the shared cube */
        orki_setrn(rc,REGCMD_I8_N,RK_CNA_FEATURE_DATA_ADDR,(uint32_t)(A.dma+roff));
        orki_setrn(rc,REGCMD_I8_N,RK_CNA_WEIGHT_DATA_ADDR,(uint32_t)B.dma);                           /* shared weight */
        orki_setrn(rc,REGCMD_I8_N,RK_DPU_DST_BASE_ADDR,(uint32_t)(Cc.dma+roff));
        /* which core-group owns this tile, and its local index */
        int g=0; while(g<nc-1 && t>=gstart[g+1]) g++;
        int j=t-gstart[g], glast=(t==gstart[g]+gsz[g]-1);
        /* chain WITHIN the group; the last tile of each group is terminal */
        if(!glast){ uint64_t nxt = RC.dma + (uint64_t)(t+1)*REGCMD_I8_N*4;
            rc[216]=0x0010|((uint32_t)(nxt&0xffff)<<16); rc[217]=(0x0101u<<16)|((uint32_t)(nxt>>16)&0xffff);
            rc[218]=0x0014|(0x0037u<<16);                rc[219]=(0x0101u<<16)|0;
        } else { rc[216]=0x0010; rc[217]=(0x0101u<<16); rc[218]=0x0014; rc[219]=(0x0101u<<16); }
        /* #39 task-1 WEIGHT-REUSE loader-recipe sweep (ORK_WR_MODE): on a reuse tile (j>0 in its core-group) OR
         * WEIGHT_REUSE (0x1040 bit13) so the HW skips the weight re-DMA and reads the group loader's resident weight
         * (shared B). If ANY mode is BOTH bit-exact AND faster than mode 0, cross-tile reuse IS achievable (fold
         * could stream weight once/slice). 1=WEIGHT_REUSE; 2=+DATA_REUSE(bit12); 3=+FC_SKIP_EN(0x1060) on reuse
         * tiles; 4=+FC_SKIP_EN on the loader too. */
        { static int wrm=-2; static uint32_t fcs=0; static int fcsset=-1;
          if(wrm==-2){ const char*e=getenv("ORK_WR_MODE"); wrm=e?atoi(e):0; }
          if(fcsset==-1){ const char*e=getenv("ORK_WR_FCSKIP"); fcsset=e?1:0; fcs=e?(uint32_t)strtoul(e,0,0):0; }
          if(wrm>0 && j>0){                                   /* reuse tile: skip the weight re-DMA */
            for(int k=0;k+1<REGCMD_I8_N;k+=2) if((rc[k]&0xffff)==0x1040 && (rc[k+1]>>16)==0x201){
                uint32_t v=(rc[k]>>16)|0x2000u; if(wrm>=2) v|=0x1000u; rc[k]=(v<<16)|0x1040; break; }
            /* FC_SKIP (0x1060): FC_SKIP_EN[0] | FC_SKIP_DATA[31:16] — sweep the companion count via ORK_WR_FCSKIP
             * (raw 0x1060 value; e.g. 0x0e000001 = FC_SKIP_DATA=K, EN=1). EN alone errored the submit. */
            if(fcsset) orki_setr(rc,REGCMD_I8_N,0x201,0x1060,fcs); } }
        memcpy(rcbuf + (size_t)t*REGCMD_I8_N, rc, sizeof rc);
        struct rknpu_task *tkg=(struct rknpu_task*)TK[g].cpu; memset(&tkg[j],0,sizeof tkg[j]);
        tkg[j].enable_mask=0xd; tkg[j].int_mask=0x300; tkg[j].int_clear=0x1ffff;
        tkg[j].regcfg_amount=108; tkg[j].regcmd_addr=RC.dma + (uint64_t)t*REGCMD_I8_N*4;
    }
    orki_bsync(fd,&RC,RKNPU_MEM_SYNC_TO_DEVICE);
    for(int g=0;g<nc;g++) orki_bsync(fd,&TK[g],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    int ret=-1;
    if(ork_fold_submit_all(fd,dom,TK,gsz,nc)) goto fbdone;                                       /* warm */
    orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_FROM_DEVICE);
    if(Craw) memcpy(Craw,Cc.cpu,(size_t)Mtot*N*4);
    { double t0=ork_now_us(); for(int i=0;i<iters;i++){ if(ork_fold_submit_all(fd,dom,TK,gsz,nc)) goto fbdone; }
      if(us)*us=(ork_now_us()-t0)/(iters>0?iters:1); ret=0; }
fbdone:
    for(int g=0;g<nc;g++) orki_bdestroy(fd,&TK[g]);
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&Cc);orki_bdestroy(fd,&RC); return ret;
}

static const uint32_t* fold_ref_for(int m){ for(int i=0;i<FOLD_REFS_N;i++) if(FOLD_REFS[i].m==m) return FOLD_REFS[i].rc; return 0; }

static int fold_pidx(const uint32_t*pr,int np,unsigned blk,unsigned reg){ for(int j=0;j<np;j++) if((pr[2*j+1]>>16)==blk&&(pr[2*j]&0xffff)==reg) return j; return -1; }

static void fold_setv(uint32_t*pr,int*np,unsigned blk,unsigned reg,uint32_t v){ int j=fold_pidx(pr,*np,blk,reg); if(j<0){j=*np;(*np)++;} pr[2*j]=(reg&0xffff)|((v&0xffff)<<16); pr[2*j+1]=(blk<<16)|((v>>16)&0xffff); }

static int fold_build_tile(int m,int Mtot,uint32_t*out232){
    const uint32_t*cap=fold_ref_for(m); if(!cap) return -1;
    static const unsigned OUT1001[20]={0x4098,0x409c,0x40a0,0x40a4,0x40a8,0x40ac,0x40c0,0x40c4,0x4100,0x4104,0x4108,0x410c,0x4110,0x4114,0x4118,0x411c,0x4120,0x4124,0x4128,0x412c};
    uint32_t pr[2*130]; int np=0;
    for(int k=0;k+1<216;k+=2){ uint32_t w0=cap[k],w1=cap[k+1]; unsigned reg=w0&0xffff,blk=w1>>16;
        if(w0==0&&w1==0) continue; if(blk==0x101&&(reg==0x0010||reg==0x0014)) continue;
        int j=fold_pidx(pr,np,blk,reg); if(j<0){pr[2*np]=w0;pr[2*np+1]=w1;np++;}else{pr[2*j]=w0;pr[2*j+1]=w1;} }
    for(int i=0;i<20;i++) if(fold_pidx(pr,np,0x1001,OUT1001[i])<0) fold_setv(pr,&np,0x1001,OUT1001[i],0);
    fold_setv(pr,&np,0x81,0x0008,0x000d);            /* doorbell */
    fold_setv(pr,&np,0x41,0x0000,0);
    fold_setv(pr,&np,0x201,0x100c,0x20000000);        /* GROUP_LINE (batch sub-tile) */
    fold_setv(pr,&np,0x1001,0x4024,(uint32_t)(16*Mtot));            /* 4 M_total regs */
    fold_setv(pr,&np,0x201,0x107c,(uint32_t)(Mtot<128?Mtot:128));
    fold_setv(pr,&np,0x201,0x1080,(uint32_t)(Mtot-m));
    fold_setv(pr,&np,0x1001,0x40c0,(uint32_t)(128*Mtot));
    if(np>108) return -2;
    memset(out232,0,232*4); for(int j=0;j<2*np;j++) out232[j]=pr[j];
    return 0;
}

static size_t fold_nc16(int m,int cc,int w){ return (size_t)(cc/16)*((size_t)w*16)+(size_t)m*16+(cc%16); }

static size_t fold_woff(int n,int k,int K){ int KT=(K+31)/32; return ((size_t)(n/32)*KT+(k/32))*1024+(size_t)(n%32)*32+(k%32); }

static size_t fold_c4(int m,int n,int w){ return (size_t)(n/4)*((size_t)w*4)+(size_t)m*4+(n%4); }

int ork_npu_fold_run_i8(ork_npu*c,int K,int N,const int8_t*Wraw,int M,const int8_t*Araw,int32_t*Cout,int ncore,int iters,double*us){
    if(!c||c->fd<0) return -3;
    if(K!=FOLD_REF_K||N!=FOLD_REF_N||M<1||M>128) return -1;
    static const int SZ[]={36,32,28,24,20,16,14,12,10,8,6,4,2,1};
    int mm[64],roff[64],P=0,rem=M,off=0;
    while(rem>0){ for(unsigned s=0;s<sizeof SZ/sizeof*SZ;s++) if(SZ[s]<=rem){mm[P]=SZ[s];roff[P]=off;off+=SZ[s];rem-=SZ[s];P++;break;} if(P>=64) break; }
    uint32_t *tiles=malloc((size_t)P*232*4); if(!tiles) return -2;
    for(int t=0;t<P;t++) if(fold_build_tile(mm[t],M,tiles+(size_t)t*232)){ free(tiles); return -2; }
    int8_t*Ap=calloc((size_t)M*K,1),*Wp=calloc((size_t)K*N,1); int32_t*Craw=calloc((size_t)M*N,4);
    if(!Ap||!Wp||!Craw){ free(tiles);free(Ap);free(Wp);free(Craw); return -2; }
    for(int i=0;i<M;i++)for(int k=0;k<K;k++) Ap[fold_nc16(i,k,M)]=Araw[(size_t)i*K+k];
    for(int k=0;k<K;k++)for(int n=0;n<N;n++) Wp[fold_woff(n,k,K)]=Wraw[(size_t)k*N+n];
    int r=ork_npu_fold_batch(c,M,K,N,P,roff,tiles,232,Ap,Wp,Craw,ncore,iters,us);
    if(!r&&Cout) for(int i=0;i<M;i++)for(int n=0;n<N;n++) Cout[(size_t)i*N+n]=Craw[fold_c4(i,n,M)];
    free(tiles);free(Ap);free(Wp);free(Craw); return r?-2:0;
}

int ork_npu_fold_op_i8(ork_npu*c,int K,int N,const int8_t*Wraw,int M,const int8_t*Araw,int32_t*Cout,int iters,double*us){
    int fd=c->fd; if(fd<0) return -3;
    const int NS=FOLD_REF_N; int nslice=(N+NS-1)/NS;
    if(K!=FOLD_REF_K||M<1||M>128||nslice<1||nslice>3) return -1;
    int dom=c->dom_active;
    static const int SZ[]={36,32,28,24,20,16,14,12,10,8,6,4,2,1};
    int mt[64],roff[64],P=0,rem=M,off=0;
    while(rem>0){ for(unsigned s=0;s<sizeof SZ/sizeof*SZ;s++) if(SZ[s]<=rem){mt[P]=SZ[s];roff[P]=off;off+=SZ[s];rem-=SZ[s];P++;break;} if(P>=64) break; }
    uint32_t *tmpl=malloc((size_t)P*232*4); if(!tmpl) return -2;
    for(int t=0;t<P;t++) if(fold_build_tile(mt[t],M,tmpl+(size_t)t*232)){ free(tmpl); return -2; }
    size_t asz=(size_t)M*K*8+(1u<<20);
    struct buf A=orki_bcreate(fd,asz,0x403,dom); if(!A.cpu){free(tmpl);return -2;}
    memset(A.cpu,0,asz);
    { int8_t*Ap=(int8_t*)A.cpu; for(int i=0;i<M;i++)for(int k=0;k<K;k++) Ap[fold_nc16(i,k,M)]=Araw[(size_t)i*K+k]; }
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf W[3]={{0}},Cc[3]={{0}},RC[3]={{0}},TK[3]={{0}}; int rc_ret=-1;
    size_t wsz=(size_t)K*NS*8+(1u<<20), csz=(size_t)M*NS*4*8+65536;
    for(int s=0;s<nslice;s++){
        W[s]=orki_bcreate(fd,wsz,0x403,dom); Cc[s]=orki_bcreate(fd,csz,0x403,dom);
        RC[s]=orki_bcreate(fd,(size_t)P*REGCMD_I8_N*4,0x403,dom); TK[s]=orki_bcreate(fd,(size_t)(P+2)*sizeof(struct rknpu_task),0x40b,dom);
        if(!W[s].cpu||!Cc[s].cpu||!RC[s].cpu||!TK[s].cpu) goto opdone;
        memset(W[s].cpu,0,wsz); memset(Cc[s].cpu,0,csz);
        int n0=s*NS, ns=(N-n0<NS)?(N-n0):NS; int8_t*Ws=(int8_t*)W[s].cpu;
        for(int k=0;k<K;k++)for(int n=0;n<ns;n++) Ws[fold_woff(n,k,K)]=Wraw[(size_t)k*N+(n0+n)];
        orki_bsync(fd,&W[s],RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Cc[s],RKNPU_MEM_SYNC_TO_DEVICE);
        uint32_t*rcbuf=(uint32_t*)RC[s].cpu; struct rknpu_task*tk=(struct rknpu_task*)TK[s].cpu;
        for(int t=0;t<P;t++){
            uint32_t rc[REGCMD_I8_N]; memcpy(rc, tmpl+(size_t)t*232, (size_t)REGCMD_I8_N*4);
            uint64_t rf=(uint64_t)roff[t]*16;
            orki_setrn(rc,REGCMD_I8_N,RK_CNA_FEATURE_DATA_ADDR,(uint32_t)(A.dma+rf));
            orki_setrn(rc,REGCMD_I8_N,RK_CNA_WEIGHT_DATA_ADDR,(uint32_t)W[s].dma);
            orki_setrn(rc,REGCMD_I8_N,RK_DPU_DST_BASE_ADDR,(uint32_t)(Cc[s].dma+rf));
            if(t<P-1){ uint64_t nxt=RC[s].dma+(uint64_t)(t+1)*REGCMD_I8_N*4;
                rc[216]=0x0010|((uint32_t)(nxt&0xffff)<<16); rc[217]=(0x0101u<<16)|((uint32_t)(nxt>>16)&0xffff);
                rc[218]=0x0014|(0x0037u<<16); rc[219]=(0x0101u<<16)|0;
            } else { rc[216]=0x0010; rc[217]=(0x0101u<<16); rc[218]=0x0014; rc[219]=(0x0101u<<16); }
            memcpy(rcbuf+(size_t)t*REGCMD_I8_N, rc, sizeof rc);
            memset(&tk[t],0,sizeof tk[t]); tk[t].enable_mask=0xd; tk[t].int_mask=0x300; tk[t].int_clear=0x1ffff;
            tk[t].regcfg_amount=108; tk[t].regcmd_addr=RC[s].dma+(uint64_t)t*REGCMD_I8_N*4;
        }
        orki_bsync(fd,&RC[s],RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&TK[s],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    }
    { int gszP[3]={P,P,P};
      if(ork_fold_submit_all(fd,dom,TK,gszP,nslice)) goto opdone;                 /* warm (concurrent, core s = slice s) */
      for(int s=0;s<nslice;s++) orki_bsync(fd,&Cc[s],RKNPU_MEM_SYNC_FROM_DEVICE);
      if(Cout) for(int s=0;s<nslice;s++){ int n0=s*NS,ns=(N-n0<NS)?(N-n0):NS; int32_t*cs=(int32_t*)Cc[s].cpu;
          for(int i=0;i<M;i++)for(int n=0;n<ns;n++) Cout[(size_t)i*N+(n0+n)]=cs[fold_c4(i,n,M)]; }
      double t0=ork_now_us(); for(int it=0;it<iters;it++){ if(ork_fold_submit_all(fd,dom,TK,gszP,nslice)) goto opdone; }
      if(us)*us=(ork_now_us()-t0)/(iters>0?iters:1); rc_ret=0; }
opdone:
    for(int s=0;s<3;s++){ if(W[s].cpu)orki_bdestroy(fd,&W[s]); if(Cc[s].cpu)orki_bdestroy(fd,&Cc[s]); if(RC[s].cpu)orki_bdestroy(fd,&RC[s]); if(TK[s].cpu)orki_bdestroy(fd,&TK[s]); }
    orki_bdestroy(fd,&A); free(tmpl); return rc_ret;
}

static void fold_scratch_free1(ork_npu *c, struct fold_scratch *fs){
    if(!fs) return;
    if(fs->Cc) for(int s=0;s<fs->nslice;s++) if(fs->Cc[s].cpu) orki_bdestroy(c->fd,&fs->Cc[s]);
    if(fs->RC) for(int s=0;s<fs->nslice;s++) if(fs->RC[s].cpu) orki_bdestroy(c->fd,&fs->RC[s]);
    if(fs->TK) for(int s=0;s<fs->nslice;s++) if(fs->TK[s].cpu) orki_bdestroy(c->fd,&fs->TK[s]);
    free(fs->Cc); free(fs->RC); free(fs->TK); free(fs->tmpl); free(fs);
}

void orki_fold_scratch_free(ork_npu *c){   /* free ALL entries + the shared A (teardown) */
    for(int i=0;i<c->fold_scr_n;i++){ fold_scratch_free1(c,c->fold_scr[i]); c->fold_scr[i]=NULL; }
    c->fold_scr_n=0;
    if(c->fold_A.cpu) orki_bdestroy(c->fd,&c->fold_A);
    c->fold_A.cpu=NULL; c->fold_A_M=0; c->fold_A_dom=0;
}

static int fold_A_ensure(ork_npu *c, int M, int dom){
    if(c->fold_A.cpu && c->fold_A_M==M && c->fold_A_dom==dom) return 0;
    if(c->fold_A.cpu) orki_bdestroy(c->fd,&c->fold_A);
    size_t asz=(size_t)M*FOLD_REF_K*8+(1u<<20);
    c->fold_A=orki_bcreate(c->fd,asz,0x403,dom); if(!c->fold_A.cpu){ c->fold_A_M=0; return -1; }
    memset(c->fold_A.cpu,0,asz); c->fold_A_M=M; c->fold_A_dom=dom; return 0;   /* zero the nc16 pad once */
}

static void fold_fill_A(ork_npu *c, int M, const int8_t *Araw){
    int8_t*Ap=(int8_t*)c->fold_A.cpu; int K=FOLD_REF_K;
    for(int i=0;i<M;i++)for(int k=0;k<K;k++) Ap[fold_nc16(i,k,M)]=Araw[(size_t)i*K+k];
    orki_bsync(c->fd,&c->fold_A,RKNPU_MEM_SYNC_TO_DEVICE);
}

static struct fold_scratch *fold_scr_get(ork_npu *c, int M, int N, int dom){
    for(int i=0;i<c->fold_scr_n;i++){ struct fold_scratch*e=c->fold_scr[i]; if(e&&e->M==M&&e->N==N&&e->domain==dom) return e; }
    const int NS=FOLD_REF_N; int nslice=(N+NS-1)/NS; int fd=c->fd;
    struct fold_scratch *fs=calloc(1,sizeof *fs); if(!fs) return NULL;
    fs->M=M; fs->N=N; fs->nslice=nslice; fs->domain=dom;
    fs->tmpl=malloc((size_t)64*232*4); if(!fs->tmpl){ free(fs); return NULL; }
    static const int SZ[]={36,32,28,24,20,16,14,12,10,8,6,4,2,1};
    int P=0,rem=M,off=0;
    while(rem>0){ int pick=-1; for(unsigned s=0;s<sizeof SZ/sizeof*SZ;s++) if(SZ[s]<=rem){ pick=SZ[s]; break; }
        if(pick<0) break;
        fs->roff[P]=off; off+=pick; rem-=pick;
        if(fold_build_tile(pick,M,fs->tmpl+(size_t)P*232)){ fold_scratch_free1(c,fs); return NULL; }
        if(++P>=64) break; }
    fs->P=P;
    size_t csz=(size_t)M*NS*4*8+65536;
    fs->Cc=calloc(nslice,sizeof(struct buf)); fs->RC=calloc(nslice,sizeof(struct buf)); fs->TK=calloc(nslice,sizeof(struct buf));
    if(!fs->Cc||!fs->RC||!fs->TK){ fold_scratch_free1(c,fs); return NULL; }
    for(int s=0;s<nslice;s++){
        fs->Cc[s]=orki_bcreate(fd,csz,0x403,dom);
        fs->RC[s]=orki_bcreate(fd,(size_t)P*REGCMD_I8_N*4,0x403,dom);
        fs->TK[s]=orki_bcreate(fd,(size_t)(P+2)*sizeof(struct rknpu_task),0x40b,dom);
        if(!fs->Cc[s].cpu||!fs->RC[s].cpu||!fs->TK[s].cpu){ fold_scratch_free1(c,fs); return NULL; }
        struct rknpu_task*tk=(struct rknpu_task*)fs->TK[s].cpu;
        for(int t=0;t<P;t++){ memset(&tk[t],0,sizeof tk[t]); tk[t].enable_mask=0xd; tk[t].int_mask=0x300; tk[t].int_clear=0x1ffff;
            tk[t].regcfg_amount=108; tk[t].regcmd_addr=fs->RC[s].dma+(uint64_t)t*REGCMD_I8_N*4; }
        orki_bsync(fd,&fs->TK[s],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    }
    if(c->fold_scr_n<8) c->fold_scr[c->fold_scr_n++]=fs;         /* insert; evict oldest if full (rare: >8 distinct shapes) */
    else { fold_scratch_free1(c,c->fold_scr[0]); memmove(&c->fold_scr[0],&c->fold_scr[1],7*sizeof c->fold_scr[0]); c->fold_scr[7]=fs; }
    return fs;
}

static int fold_run_one(ork_npu *c, ork_w *w, struct fold_scratch *fs, int dom, int32_t *Cout){
    int fd=c->fd, P=fs->P, N=fs->N, M=fs->M, nslice=fs->nslice; const int NS=FOLD_REF_N, NC=3;
    for(int s=0;s<nslice;s++){
        uint32_t*rcbuf=(uint32_t*)fs->RC[s].cpu;
        for(int t=0;t<P;t++){
            uint32_t rc[REGCMD_I8_N]; memcpy(rc, fs->tmpl+(size_t)t*232, (size_t)REGCMD_I8_N*4);
            uint64_t rf=(uint64_t)fs->roff[t]*16;
            orki_setrn(rc,REGCMD_I8_N,RK_CNA_FEATURE_DATA_ADDR,(uint32_t)(c->fold_A.dma+rf));
            orki_setrn(rc,REGCMD_I8_N,RK_CNA_WEIGHT_DATA_ADDR,(uint32_t)w->Bfold[s].dma);
            orki_setrn(rc,REGCMD_I8_N,RK_DPU_DST_BASE_ADDR,(uint32_t)(fs->Cc[s].dma+rf));
            if(t<P-1){ uint64_t nxt=fs->RC[s].dma+(uint64_t)(t+1)*REGCMD_I8_N*4;
                rc[216]=0x0010|((uint32_t)(nxt&0xffff)<<16); rc[217]=(0x0101u<<16)|((uint32_t)(nxt>>16)&0xffff);
                rc[218]=0x0014|(0x0037u<<16); rc[219]=(0x0101u<<16)|0;
            } else { rc[216]=0x0010; rc[217]=(0x0101u<<16); rc[218]=0x0014; rc[219]=(0x0101u<<16); }
            memcpy(rcbuf+(size_t)t*REGCMD_I8_N, rc, sizeof rc);
        }
        orki_bsync(fd,&fs->RC[s],RKNPU_MEM_SYNC_TO_DEVICE);
        orki_bsync(fd,&fs->Cc[s],RKNPU_MEM_SYNC_TO_DEVICE);   /* clean CPU C lines before the DPU writes (DMA coherency) */
    }
    { int gsz[3]={P,P,P}; for(int r=0;r*NC<nslice;r++){ int rn=nslice-r*NC; if(rn>NC)rn=NC;
        if(ork_fold_submit_all(fd,dom,&fs->TK[r*NC],gsz,rn)) return -2; } }
    for(int s=0;s<nslice;s++) orki_bsync(fd,&fs->Cc[s],RKNPU_MEM_SYNC_FROM_DEVICE);
    if(Cout) for(int s=0;s<nslice;s++){ int n0=s*NS,ns=(N-n0<NS)?(N-n0):NS; int32_t*cs=(int32_t*)fs->Cc[s].cpu;
        for(int i=0;i<M;i++)for(int n=0;n<ns;n++) Cout[(size_t)i*N+(n0+n)]=cs[fold_c4(i,n,M)]; }
    return 0;
}

int ork_npu_fold_run_w(ork_npu*c, ork_w*w, int M, const int8_t*Araw, int32_t*Cout, int iters, double*us){
    int fd=c->fd; if(fd<0||!w||!w->Bfold) return -3; (void)fd;
    int K=w->K; int nslice=w->fold_ns;
    if(K!=FOLD_REF_K||M<1||M>128||nslice<1||nslice>64) return -1;
    int dom=w->domain;
    if(fold_A_ensure(c,M,dom)) return -2;
    struct fold_scratch *fs=fold_scr_get(c,M,w->N,dom); if(!fs) return -2;
    fold_fill_A(c,M,Araw);
    if(fold_run_one(c,w,fs,dom,Cout)) return -2;
    if(iters>0){ double t0=ork_now_us(); for(int it=0;it<iters;it++){ fold_fill_A(c,M,Araw); if(fold_run_one(c,w,fs,dom,NULL)) return -2; }
        if(us)*us=(ork_now_us()-t0)/iters; }
    return 0;
}

int ork_npu_fold_batch_w(ork_npu*c, int nw, ork_w**ws, int M, const int8_t*Araw, int32_t**Couts, int iters, double*us){
    if(c->fd<0||nw<1||!ws||!ws[0]) return -3;
    int dom=ws[0]->domain;
    if(M<1||M>128) return -1;
    for(int i=0;i<nw;i++){ ork_w*w=ws[i]; if(!w||!w->Bfold||w->K!=FOLD_REF_K||w->fold_ns<1||w->fold_ns>64||w->domain!=dom) return -1; }
    if(fold_A_ensure(c,M,dom)) return -2;
    struct fold_scratch *fs[16]; if(nw>16) return -1;
    for(int i=0;i<nw;i++){ fs[i]=fold_scr_get(c,M,ws[i]->N,dom); if(!fs[i]) return -2; }
    fold_fill_A(c,M,Araw);                                        /* ONCE — shared by all nw weights */
    for(int i=0;i<nw;i++) if(fold_run_one(c,ws[i],fs[i],dom,Couts?Couts[i]:NULL)) return -2;
    if(iters>0){ double t0=ork_now_us(); for(int it=0;it<iters;it++){ fold_fill_A(c,M,Araw);
            for(int i=0;i<nw;i++) if(fold_run_one(c,ws[i],fs[i],dom,NULL)) return -2; }
        if(us)*us=(ork_now_us()-t0)/iters; }
    return 0;
}

size_t ork_w_dump_fold_i8_cpu(ork_npu *c, int K, int N, const int8_t *B, void *out, size_t cap){
    (void)c;
    if(!B || K!=FOLD_REF_K || N<1 || (N%32)) return 0;
    const int NS=FOLD_REF_N; int nslice=(N+NS-1)/NS; if(nslice>64) return 0;   /* N up to 64*1216 (covers gate/up N=18944) */
    size_t off=0;
    for(int s=0;s<nslice;s++){ int n0=s*NS, sw=(N-n0<NS)?(N-n0):NS; size_t tsz=orki_pgup((size_t)K*sw);
        if(out){ if(off+tsz>cap) return 0;
            int8_t *ws=(int8_t*)out+off; memset(ws,0,tsz);
            for(int k=0;k<K;k++)for(int n=0;n<sw;n++) ws[fold_woff(n,k,K)]=B[(size_t)k*N+(n0+n)]; }
        off+=tsz; }
    return off;
}

ork_w *ork_mm_load_fold_i8(ork_npu *c,int K,int N,const void *blob,size_t n){
    if(K!=FOLD_REF_K || N<1 || (N%32)) return NULL;
    const int NS=FOLD_REF_N; int nslice=(N+NS-1)/NS; if(nslice>64) return NULL;
    size_t need=0; for(int s=0;s<nslice;s++){int n0=s*NS,sw=(N-n0<NS)?(N-n0):NS; need+=orki_pgup((size_t)K*sw);}
    if(n!=need) return NULL;
    ork_w *w=calloc(1,sizeof *w); if(!w) return NULL;
    w->K=K;w->N=N;w->dtype=DT_I8;w->owns=1;w->domain=ork_dom(c->pack_domain);
    w->Bfold=calloc(nslice,sizeof(struct buf)); w->fold_ns=nslice;
    size_t off=0;
    for(int s=0;s<nslice;s++){int n0=s*NS,sw=(N-n0<NS)?(N-n0):NS; size_t tsz=orki_pgup((size_t)K*sw);
        struct buf*b=&w->Bfold[s]; *b=orki_bcreate(c->fd,tsz,0x403,w->domain);
        if(!b->cpu){ for(int i=0;i<s;i++) orki_bdestroy(c->fd,&w->Bfold[i]); free(w->Bfold); free(w); return NULL; }
        memcpy(b->cpu,(const char*)blob+off,tsz); off+=tsz;
        orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE); }
    return w;
}

int ork_w_attach_fold_i8(ork_npu *c, ork_w *w, const void *blob, size_t n){
    if(!c||!w||w->K!=FOLD_REF_K||w->N<1||(w->N%32)) return -1;
    if(w->Bfold) return 0;
    const int NS=FOLD_REF_N; int nslice=(w->N+NS-1)/NS; if(nslice<1||nslice>64) return -1;
    size_t need=0; for(int s=0;s<nslice;s++){int n0=s*NS,sw=(w->N-n0<NS)?(w->N-n0):NS; need+=orki_pgup((size_t)w->K*sw);}
    if(n!=need) return -1;
    struct buf *bf=calloc(nslice,sizeof(struct buf)); if(!bf) return -2;
    size_t off=0;
    for(int s=0;s<nslice;s++){int n0=s*NS,sw=(w->N-n0<NS)?(w->N-n0):NS; size_t tsz=orki_pgup((size_t)w->K*sw);
        bf[s]=orki_bcreate(c->fd,tsz,0x403,w->domain);
        if(!bf[s].cpu){ for(int i=0;i<s;i++)orki_bdestroy(c->fd,&bf[i]); free(bf); return -2; }
        memcpy(bf[s].cpu,(const char*)blob+off,tsz); off+=tsz;
        orki_bsync(c->fd,&bf[s],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(c->fd,&bf[s],RKNPU_MEM_SYNC_TO_DEVICE); }
    w->Bfold=bf; w->fold_ns=nslice; return 0;
}
