/* npu/i8/dyn_seq.c — the HETEROGENEOUS single-core NONBLOCK chain: ork_dyn_begin_seq_i8{,_mc} and ork_dyn_seq_end.
 *
 * Part of the int8 (W8A8) datapath. Split out of the 1,000-line npu/i8/dyn.c (MODULARIZE_PLAN.md
 * round 10) as a CONTIGUOUS line range — dyn.c had ZERO file-scope statics, so the cut costs no
 * linkage at all. Interface types in npu/internal.h, substrate in npu/core.h. */
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
#include <arm_neon.h>
#include "ork_regs.h"
#include "regcmd_array_4x32x16.h"
#include "regcmd_i8.h"
#include "regcmd_i4.h"
#include "regcmd_silu.h"
#include "regcmd_ewmul.h"
#include "regcmd_fold_refs.h"
#include "npu/internal.h"
#include "npu/core.h"
#include "npu/i8/i8.h"
#include "spine_kernels.h"

ork_dyn_chain *ork_dyn_begin_seq_i8_mc(ork_npu *c, int n, const ork_seq_op *ops, int ngroups, const int *gstart, int nc){
    if(!c||n<1||n>256||!ops||ngroups<1||ngroups>n||!gstart) return NULL;
    if(!ork_ppu_fuse_enabled(c)) return NULL;
    if(gstart[0]!=0||gstart[ngroups]!=n) return NULL;
    unsigned dom=0; int have_dom=0; int has_sdp=0;
    for(int i=0;i<n;i++){ if(!orki_seq_op_ok(&ops[i],&dom,&have_dom)) return NULL; if(ops[i].kind!=ORK_OP_MM_I8) has_sdp=1; }
    for(int g=0;g<ngroups;g++){ if(gstart[g+1]<=gstart[g]) return NULL; if(ops[gstart[g+1]-1].kind!=ORK_OP_MM_I8) return NULL; }  /* each group terminal = matmul (SDP terminal -> B2 witness upstream) */
    /* int16 SiLU HW-chain: one resident SDP LUT per chain, so all silu ops must share (in_scale,out_scale), and
     * force SINGLE-CORE (one SDP SRAM to load). A prologue LUT-load runs below before the chain submit. */
    int has_silu=0, silu_kind=0; double silu_is=0, silu_os=0;
    for(int i=0;i<n;i++) if(ops[i].kind==ORK_OP_SILU_I16 || ops[i].kind==ORK_OP_SILU_I8){
        if(!has_silu){ has_silu=1; silu_kind=ops[i].kind; silu_is=ops[i].in_scale; silu_os=ops[i].out_scale; }
        else if(ops[i].kind!=silu_kind || ops[i].in_scale!=silu_is || ops[i].out_scale!=silu_os) return NULL; }  /* one resident LUT/chain */
    if(nc<1||nc>c->soc->cores) nc=c->soc->cores; if(nc>ngroups) nc=ngroups;
    if(has_silu) nc=1;
    /* greedy load-balance: assign each group to the least-loaded core (cost ~ matmul weight-DMA K*N + SDP M*N) */
    long load[ORK_MAXCORE]; int core_of[256]; for(int i=0;i<nc;i++)load[i]=0;
    for(int g=0;g<ngroups;g++){ long cost=0; for(int i=gstart[g];i<gstart[g+1];i++){ const ork_seq_op*o=&ops[i];
            cost += (o->kind==ORK_OP_MM_I8)?(long)o->w->K*o->w->N:(long)o->M*o->N; }
        int best=0; for(int i=1;i<nc;i++) if(load[i]<load[best]) best=i; core_of[g]=best; load[best]+=cost; }
    if(orki_mc_ensure(c,nc)) return NULL;
    ork_npu_enter(c, 3 /*DT_I8_CHAIN*/, XP_CHAIN_NT, OCK_HW);
    if(have_dom && (dom!=c->dom_active || (dom && !c->dom_save))) orki_dom_activate(c, dom);
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    /* per-core program count + staging/output need; total output for the shared seq_out */
    int Pc[ORK_MAXCORE]; size_t afn[ORK_MAXCORE]; size_t outneed=0;
    for(int i=0;i<nc;i++){ Pc[i]=0; afn[i]=0; }
    for(int g=0;g<ngroups;g++){ int i=core_of[g];
        for(int p=gstart[g];p<gstart[g+1];p++){ const ork_seq_op*o=&ops[p]; Pc[i]++;
            if(o->kind==ORK_OP_MM_I8){ afn[i]+=(size_t)o->M*o->w->K; outneed+=(size_t)o->M*o->w->N*4; }
            else if(o->kind==ORK_OP_SILU_I16){ afn[i]+=(size_t)o->M*o->N*2; outneed+=(size_t)o->M*o->N*2; }   /* int16 unary */
            else if(o->kind==ORK_OP_SILU_I8){ afn[i]+=(size_t)o->M*o->N; outneed+=(size_t)o->M*o->N; }        /* int8 unary */
            else { afn[i]+=(size_t)2*o->M*o->N; outneed+=(size_t)o->M*o->N; } } }
    for(int i=0;i<nc;i++){ if((size_t)Pc[i]*REGCMD_I8_N*4>c->mrc[i].size || (size_t)Pc[i]*sizeof(struct rknpu_task)>c->mtk[i].size) return NULL;
        if(afn[i]>c->maf[i].size){ orki_bdestroy(fd,&c->maf[i]); c->maf[i]=orki_bcreate(fd,afn[i],0x403,c->dom_active); if(!c->maf[i].cpu) return NULL; } }
    ork_dyn_chain *h=calloc(1,sizeof *h); if(!h) return NULL;
    h->c=c; h->S=n; h->P=n; h->mc=0; h->seq=1; h->seq_nc=nc; h->dom=have_dom?dom:0;
    h->seq_out=orki_bcreate(fd, outneed<4096?4096:outneed, 0x403, c->dom_active);
    if(!h->seq_out.cpu){ free(h); return NULL; }
    memset(h->seq_out.cpu,0,h->seq_out.size);
    /* int16-SiLU HW-chain prologue: load the silu curve into SDP SRAM ONCE (ping-pong OFF — the #35 LUT
     * SRAM-commit race), resident across the chain. orki_build_act_lut16's calibration is a STANDALONE probe, run
     * here BEFORE the chain submit (cached after the first call). Lrc/Lsc stay alive until seq_end. */
    if(has_silu){
        int16_t lut[1030];   /* the LUT loader (REGCMD_SILU_LUT) is common; only the curve values + compute idx params differ by precision */
        if(silu_kind==ORK_OP_SILU_I16){ if(orki_build_act_lut16(c, orki_silu_f, silu_is, silu_os, lut)){ orki_bdestroy(fd,&h->seq_out); free(h); return NULL; } }
        else { if(orki_silu_calibrate_idx(c)){ orki_bdestroy(fd,&h->seq_out); free(h); return NULL; } orki_silu_build_curve(c, orki_silu_f, silu_is, silu_os, lut); }
        h->silu_lrc=orki_bcreate(fd,(size_t)REGCMD_SILU_LUT_N*4,0x403,c->dom_active);
        h->silu_lsc=orki_bcreate(fd,4096,0x403,c->dom_active);
        if(!h->silu_lrc.cpu||!h->silu_lsc.cpu){ if(h->silu_lrc.cpu)orki_bdestroy(fd,&h->silu_lrc); if(h->silu_lsc.cpu)orki_bdestroy(fd,&h->silu_lsc); orki_bdestroy(fd,&h->seq_out); free(h); return NULL; }
        memcpy(h->silu_lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
        orki_setrn((uint32_t*)h->silu_lrc.cpu,REGCMD_SILU_LUT_N,RK_DPU_DST_BASE_ADDR,(uint32_t)h->silu_lsc.dma);
        { uint32_t*lr=(uint32_t*)h->silu_lrc.cpu; int j=0; for(int k=0;k+1<REGCMD_SILU_LUT_N;k+=2){ if((lr[k]&0xffff)==0x4104){ int32_t v=(j<1030)?(int32_t)lut[j]:0; j++;
            lr[k]=0x4104|((uint32_t)(v&0xffff)<<16); lr[k+1]=(0x1001u<<16)|(((uint32_t)v>>16)&0xffff); } } }
        orki_bsync(fd,&h->silu_lrc,RKNPU_MEM_SYNC_TO_DEVICE);
        { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=1097; t->regcmd_addr=h->silu_lrc.dma;
          orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
          struct rknpu_submit ls;memset(&ls,0,sizeof ls);ls.flags=0x1;ls.task_number=1;ls.task_obj_addr=c->task.obj;ls.core_mask=RKNPU_CORE0_MASK;ls.fence_fd=-1;ls.timeout=orki_ew_timeout_ms();ls.subcore_task[0]=(struct rknpu_subcore_task){0,1};
          if(orki_rknpu_submit_ioctl(fd,&ls,c->dom_active)){ orki_bdestroy(fd,&h->silu_lrc); orki_bdestroy(fd,&h->silu_lsc); orki_bdestroy(fd,&h->seq_out); free(h); return NULL; } }
        h->silu_lut=1;
    }
    size_t coff=0;
    struct rknpu_submit subs[ORK_MAXCORE];
    for(int i=0;i<nc;i++){ struct buf *RC=&c->mrc[i], *AF=&c->maf[i];
        memset(RC->cpu,0,(size_t)Pc[i]*REGCMD_I8_N*4);
        struct rknpu_task *tk=(struct rknpu_task*)c->mtk[i].cpu; memset(tk,0,(size_t)Pc[i]*sizeof *tk);
        size_t astage=0; int pp=0; int last_gi=-1;
        for(int g=0;g<ngroups;g++){ if(core_of[g]!=i) continue;
            for(int p=gstart[g];p<gstart[g+1];p++){
                int is_core_last = 0; /* determined below by scanning ahead */
                (void)is_core_last;
                /* next program on THIS core: the next op p+1 within group, or the first op of the next group on this core */
                int nx_pp=-1, nx_kind=0;
                if(pp+1<Pc[i]){ nx_pp=pp+1;
                    int np = p+1; if(np<gstart[g+1]) nx_kind=ops[np].kind;
                    else { for(int g2=g+1;g2<ngroups;g2++) if(core_of[g2]==i){ nx_kind=ops[gstart[g2]].kind; break; } } }
                orki_seq_build_op(h,&ops[p],p,RC,AF,tk,&astage,&coff,pp,nx_pp,nx_kind,CBUF);
                last_gi=p; pp++;
            } }
        h->seq_term_c[i]=last_gi;                        /* the core's last program (a matmul) = sentinel */
        orki_bsync(fd,AF,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,RC,RKNPU_MEM_SYNC_TO_DEVICE);
        orki_bsync(fd,&c->mtk[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        memset(&subs[i],0,sizeof subs[i]);
        subs[i].flags = has_sdp ? (0x1u|0x2u) : (ork_ppflags()|0x2u);
        subs[i].task_number=(uint32_t)Pc[i]; subs[i].task_obj_addr=c->mtk[i].obj; subs[i].core_mask=1u<<i; subs[i].fence_fd=-1;
        subs[i].subcore_task[0]=subs[i].subcore_task[1]=subs[i].subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)Pc[i]};
    }
    orki_bsync(fd,&h->seq_out,RKNPU_MEM_SYNC_TO_DEVICE);   /* clean-before: no dirty CPU line evicts over NPU writes */
    /* seed each core's terminal matmul's per-row last-col int32 sentinel */
    for(int i=0;i<nc;i++){ int ti=h->seq_term_c[i]; if(ti<0)continue; int M=h->oM[ti], N=h->nout[ti]/(M?M:1);
        volatile int32_t*o=(volatile int32_t*)((char*)h->seq_out.cpu+h->ooff[ti]);
        for(int m=0;m<M;m++){ volatile int32_t*db=&o[(size_t)m*N+(N-1)]; *db=ORK_DYN_SENT; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); } }
    __asm__ volatile("dsb ish":::"memory");
    ork_install_term();
    for(int i=0;i<nc;i++){ subs[i].timeout=orki_mm_timeout_ms(); if(orki_rknpu_submit_ioctl(fd,&subs[i],c->dom_active)){ /* a core rejected: drain what we can */ } }
    return h;
}

/* ================= HETEROGENEOUS SINGLE-CORE NONBLOCK CHAIN (ork_dyn_begin_seq_i8) =================
 * Run ONE group of int8 ops [matmul + int8 SDP ...] as one core's PC-chain on begin_mc's recipe (mc_ensure
 * mrc/maf + a chain-owned warmed output scratch, clean-before, 64B-aligned program slots, per-op forward
 * descriptor), NONBLOCK, ping-pong OFF (an SDP task is present). The TERMINAL op MUST be a matmul — its int32
 * 0x7fffffff last-col sentinel gates completion (int8 SDP output has no free poison). ork_dyn_seq_end() polls
 * the terminal + does per-op copy-back (matmul int32 dense; SDP int8 EWCUBE de-marshalled). Returns NULL if
 * ineligible (caller then runs the ops via the SW break). Stage 2: MM_I8 + EWMUL_I8; ADD/SILU/GELU follow.
 * SINGLE group / single core here; the scheduler slices a sequence into groups and (Stage 3) spreads them. */
ork_dyn_chain *ork_dyn_begin_seq_i8(ork_npu *c, int n, const ork_seq_op *ops){
    int gs[2]={0,n}; return ork_dyn_begin_seq_i8_mc(c,n,ops,1,gs,1);
}

int ork_dyn_seq_end(ork_dyn_chain *h){
    if(!h||!h->seq) return -2;
    ork_npu *c=h->c; int fd=c->fd; int rc=0;
    int nc = h->seq_nc>0 ? h->seq_nc : 1;
    orki_in_doorbell=1; double t0=ork_now_us(); int landed=0;
    for(;;){ int done=1;                                 /* wait until EVERY core's terminal matmul sentinel is overwritten */
        for(int i=0;i<nc && done;i++){ int ti=h->seq_term_c[i]; if(ti<0)continue; int Mt=h->oM[ti], Nt=h->nout[ti]/(Mt?Mt:1);
            volatile int32_t *o=(volatile int32_t*)((char*)h->seq_out.cpu+h->ooff[ti]);
            for(int m=0;m<Mt;m++){ volatile int32_t*db=&o[(size_t)m*Nt+(Nt-1)]; __asm__ volatile("dc civac,%0"::"r"(db):"memory"); if(*db==ORK_DYN_SENT){done=0;break;} } }
        if(done){landed=1;break;} if(orki_ork_term||ork_now_us()-t0>3e6) break; }
    orki_in_doorbell=0;
    if(!landed) rc=-1;
    orki_bsync(fd,&h->seq_out,RKNPU_MEM_SYNC_FROM_DEVICE);
    for(int i=0;i<h->S;i++){ if(!h->dst[i]) continue; int M=h->oM[i], N=h->nout[i]/(M?M:1);
        if(h->oesz8[i]==4){ memcpy(h->dst[i], (char*)h->seq_out.cpu+h->ooff[i], (size_t)M*N*4); }
        else if(h->oesz8[i]==2){ const char*src=(const char*)h->seq_out.cpu+h->ooff[i]; int16_t*dst=(int16_t*)h->dst[i];   /* int16 EWCUBEH (atom-8, 2-byte) -> row-major */
            for(int m=0;m<M;m++)for(int nn=0;nn<N;nn++) dst[m*N+nn]=*(const int16_t*)(src + ((size_t)(nn/8)*(M*16) + (size_t)m*16 + (size_t)(nn%8)*2)); }
        else { const int8_t*src=(const int8_t*)((char*)h->seq_out.cpu+h->ooff[i]); int8_t*dst=(int8_t*)h->dst[i];
            for(int m=0;m<M;m++)for(int nn=0;nn<N;nn++) dst[m*N+nn]=src[ORK_SEQCUBE(m,nn,M)]; } }
    if(h->silu_lut){ orki_bdestroy(fd,&h->silu_lrc); orki_bdestroy(fd,&h->silu_lsc); }   /* release the resident LUT buffers */
    orki_bdestroy(fd,&h->seq_out); free(h);
    if(orki_ork_term){ int k=0; sigaction(SIGTERM,&orki_prev_sig[k],NULL); raise(SIGTERM); }
    return rc;
}
