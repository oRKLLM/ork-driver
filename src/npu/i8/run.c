/* npu/i8/run.c — int8 execution: run_i8 and the fused-activation variants, standalone SDP ops, activations, streams, batched GEMM.
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
#include "orkd_proto.h"
#include "npu/internal.h"
#include "npu/core.h"
#include "npu/i8/i8.h"
#include "spine_kernels.h"

int ork_mm_run_i8(ork_npu *c,ork_w *w,int M,const int8_t *A,int32_t *C){
    if(w && w->is_orkd){   /* Path B: int8 run on the daemon — ring transport if attached, else socket */
        orkd_set_op_domain(c->daemon, (uint32_t)w->domain);   /* v2: carry this weight's domain with the op so the daemon zero-copy-swaps to it */
        /* ZERO-COPY transport (ORK_ORKD_ZC): A+C ride SCM_RIGHTS dma-buf fds; the daemon reads A / writes C IN
         * PLACE (no ~M*K + M*N*4 byte-copy over the socket). Measures the transport delta on the big prefill ops
         * (the socket/ring paths copy). -2 = no dma-heap -> fall back to the copy path. */
        static int zc = -1; if(zc<0) zc = getenv("ORK_ORKD_ZC") ? 1 : 0;
        if(zc){ int r=orkd_run_i8_zc2(c->daemon, w->orkd_id, M, w->K, w->N, A, C); if(r!=-2) return r; }
        if(c && c->daemon && orkd_has_ring(c->daemon)){ int r=orkd_ring_run(c->daemon,w->orkd_id,M,w->K,w->N,ORKD_DT_I8,A,C); if(r!=-2) return r; }
        return orkd_run_i8(c->daemon, w->orkd_id, M, w->K, w->N, A, C); }
    if(w->dtype!=DT_I8) return -1;
    if(orki_check_overlap("ork_mm_run_i8", (uintptr_t)A, (uintptr_t)A + (size_t)M * w->K, (uintptr_t)C, (uintptr_t)C + (size_t)M * w->N * 4)) return -1;
    /* ORK_RUN_TRACE=N: dump A(int8 input) + C(int32 output) stats for the first N int8 matmuls of the real
     * orki_run (NOT a synthetic tool). A garbage => backend act-quant (stage 2); C garbage w/ sane A => weight
     * bytes/tiling (stage 1); both sane => the garbage is dequant/scales (stage 4). Flushed per line. */
    static int rtr=-1; if(rtr<0){ const char*e=getenv("ORK_RUN_TRACE"); rtr=e?atoi(e):0; }
    static long rn=0; long myn=rn++; int trace = rtr && myn<(long)rtr;
    if(trace){ long amn=127,amx=-128,anz=0; size_t na=(size_t)M*w->K;
        for(size_t i=0;i<na;i++){ int8_t v=A[i]; if(v<amn)amn=v; if(v>amx)amx=v; if(v)anz++; }
        fprintf(stderr,"[RUN#%ld] i8 M=%d K=%d N=%d Sk=%d Sn=%d bf=%d | A[int8] min=%ld max=%ld nz=%ld/%zu head=[%d %d %d %d]\n",
            myn,M,w->K,w->N,w->Sk,w->Sn,w->Bf?1:0,amn,amx,anz,na,A[0],A[1],A[2],A[3]); fflush(stderr); }
    /* #39 SELECTIVE mfold: a q/o-class weight (K=3584, wide N in <=3 slices) carrying a resident fold weight
     * (orkpack Bfold) at SMALL M<=64 wins ~1.1-1.44x via the token-fold's compute-under-DMA overlap. Larger M
     * and the FFN (gate/up N=18944) are DRAM-BW-bound (no win), and k/v (1 slice) has no N-parallelism — so
     * auto-engage ONLY when Bfold is present (eligible shape), fold_ns is the 2-3 slice winning band, and M is
     * in-envelope. This makes it ubatch-selective for free: small batches fold, large batches take normal. */
    if(w->Bfold && w->fold_ns>=2 && w->fold_ns<=3 && M>=1 && M<=64 && !getenv("ORK_NOFOLD")){
        double tf = orki_ork_prof ? ork_now_us() : 0;
        int rf = ork_npu_fold_run_w(c,w,M,(const int8_t*)A,C,0,NULL);
        if(rf==0){ if(orki_ork_prof){ orki_prof_i8_us+=ork_now_us()-tf; orki_prof_i8_calls++; } return 0; }
        /* fold declined (alloc/shape) — fall through to the normal path, no behavior change */
    }
    double t0 = orki_ork_prof ? ork_now_us() : 0;
    int r = orki_run(c,w,M,A,C);
    if(orki_ork_prof){ orki_prof_i8_us+=ork_now_us()-t0; orki_prof_i8_calls++; }
    if(trace){ long cmn=2147483647L,cmx=-2147483648L,cz=0; size_t nc=(size_t)M*w->N;
        for(size_t i=0;i<nc;i++){ int32_t v=C[i]; if(v<cmn)cmn=v; if(v>cmx)cmx=v; if(!v)cz++; }
        fprintf(stderr,"[RUN#%ld]   -> rc=%d C[int32] min=%ld max=%ld zero=%ld/%zu head=[%d %d %d %d]\n",
            myn,r,cmn,cmx,cz,nc,C[0],C[1],C[2],C[3]); fflush(stderr); }
    return r;
}

int orki_slice_run_i8(ork_npu *c, ork_w_sliced *w, int M, const int8_t *A, int32_t *C, int nc) {
    if (!c || !w || !A || !C || M < 1) return -1;
    int ks = w->ks, ns = w->ns;   /* balanced tile step baked at orki_pack (equal-width N-tiles >= cores) */
    int nks = w->nks, nnt = w->nnt, S = nks * nnt, K = w->K, N = w->N, Kpad = w->Kpad;
    /* SINGLE chained doorbell submit over EVERY tile (K-slices x N-tiles) — one begin/end, not nks*nnt
     * round-trips. ork_dyn_begin_mc distributes the S tasks across the nc cores and chains each core's tasks
     * into one PC-chain (weight streamed in one pass), the wedge-safe mirror of the mcworker's CHAIN-KSPLIT.
     * A per K-slice (shared across its N-tiles); each tile's [M,Nw] partial lands in its own plain-malloc slot
     * (forces direct=0 scratch copy-back — see the note above ork_mm_pack_sliced); the K-slices are then
     * summed host-side into C (the NPU has no on-device C+= mode). */
    int8_t  *Aslc = malloc((size_t) M * Kpad);                            /* [nks] gathered A[:, k0:k1] blocks (padded K, zero tail) */
    int32_t *part = malloc((size_t) nks * M * N * sizeof(int32_t));        /* one [M,Nw] slot per tile */
    ork_mm_task_i8 *tasks = malloc((size_t) S * sizeof *tasks);
    if (!Aslc || !part || !tasks) { free(Aslc); free(part); free(tasks); return -1; }
    size_t aoff = 0, poff = 0;
    for (int ki = 0; ki < nks; ki++) { int k0 = ki*ks, k1 = k0+ks < Kpad ? k0+ks : Kpad, Ks = k1-k0;   /* over Kpad */
        int8_t *aptr = Aslc + aoff;
        int real = K - k0; if (real > Ks) real = Ks; if (real < 0) real = 0;   /* real A cols this slice; PAD tail -> 0 */
        for (int m = 0; m < M; m++) { memcpy(aptr + (size_t) m*Ks, A + (size_t) m*K + k0, real);
                                      if (real < Ks) memset(aptr + (size_t) m*Ks + real, 0, Ks - real); }
        aoff += (size_t) M * Ks;
        for (int ni = 0; ni < nnt; ni++) { int n0 = ni*ns, n1 = n0+ns < N ? n0+ns : N, Nw = n1-n0;
            tasks[ki*nnt + ni] = (ork_mm_task_i8){ w->sub[ki*nnt + ni], M, aptr, part + poff };
            poff += (size_t) M * Nw; } }
    int rc = 0;
    ork_dyn_chain *h = ork_dyn_begin_mc(c, S, tasks, nc);                  /* ONE doorbell submit for all tiles */
    if (!h) rc = -1;                                                       /* a c_base tile outside the doorbell envelope -> fail; the caller refuses (ORK_RC_WEDGE_PRONE). NEVER a blocking fall-back (#45) */
    else if (ork_dyn_end(h) < 0) rc = -1;
    if (!rc) {   /* PER-CORE PARALLEL ks-outer NEON accumulate (Stage 3) — balanced disjoint C-column ranges */
        int anc = nc>0 ? nc : c->soc->cores; if(anc>ORK_MAXCORE) anc=ORK_MAXCORE; if(anc<1) anc=1;
        struct slc_acc acc[ORK_MAXCORE];
        for(int i=0;i<anc;i++){ int cc0=(int)((long)i*N/anc), cc1=(int)((long)(i+1)*N/anc);
            acc[i]=(struct slc_acc){ tasks, C, nks, nnt, ns, N, M, cc0, cc1 }; }
        if(anc==1){ orki_slice_acc_worker(&acc[0]); }
        else {
            orki_npu_pool_ensure(c);
            pthread_mutex_lock(&c->pmu);
            c->pjob=acc; c->pjob_nc=anc; c->pjob_fn=orki_slice_acc_worker; c->pjob_stride=sizeof(struct slc_acc);
            c->pdone=0; c->pgen++; pthread_cond_broadcast(&c->pgo);
            pthread_mutex_unlock(&c->pmu);
            orki_slice_acc_worker(&acc[0]);   /* core 0 runs on the calling thread */
            pthread_mutex_lock(&c->pmu); while(c->pdone < anc-1) pthread_cond_wait(&c->pdn,&c->pmu); pthread_mutex_unlock(&c->pmu);
        }
    }
    free(Aslc); free(part); free(tasks); return rc;
}

int ork_mm_run_i8_silu(ork_npu *c,ork_w *w,int M,const int8_t *A,int8_t *C,
                       int r_mult,int r_shift,uint32_t out_bias,uint32_t idx_off,uint32_t cfg4068,
                       const int16_t *lut,int nlut){
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(w->dtype!=DT_I8 || !w->Bf) return -2;
    int fd=c->fd,K=w->K,N=w->N,NMAX=c->soc->nmax,CBUF=c->soc->cbuf_elems;
    if(K%512 || K>4096 || N%32) return -2;
    if(w->domain!=c->dom_active || (w->domain!=0 && !c->dom_save)) orki_dom_activate(c,w->domain);  /* submit's buffers must live in the weight's domain (mirror orki_run()) */
    if(DT_I8!=c->last_dt){ c->warmed=0; c->ccsz=0; c->last_dt=DT_I8; }
    int chunk=orki_fused_mtile(K,M);                          /* mg_max*64 M-tile per submit (was 64: doubled prefill submits) */
    size_t maxaf=(size_t)chunk*K, maxout=(size_t)chunk*NMAX;
    /* realloc Af/Cc if too small OR in the WRONG domain: chained FFN weights (gate/up/down) can live in
     * different IOMMU domains, and orki_dom_activate() above may have switched dom_active — a submit against a
     * buffer whose IOVA was reserved in another domain faults the NPU (soft reset). Keep them in dom_active. */
    if(c->Af.size<maxaf || c->Af.domain!=c->dom_active){ orki_bdestroy(fd,&c->Af); c->Af=orki_bcreate(fd,maxaf,0x403,c->dom_active); if(!c->Af.cpu)return -2; }
    if(c->ccsz<maxout || c->Cc.domain!=c->dom_active){ orki_bdestroy(fd,&c->Cc); c->Cc=orki_bcreate(fd,maxout,0x403,c->dom_active); c->ccsz=maxout; c->warmed=0; if(!c->Cc.cpu)return -2; }
    /* build the LUT-load regcmd once (fixed silu*S PWL LUT, or the supplied lut[]) */
    struct buf Lrc=orki_bcreate(fd,(size_t)REGCMD_SILU_LUT_N*4,0x403,c->dom_active); if(!Lrc.cpu)return -2;
    struct buf Lsc=orki_bcreate(fd,4096,0x403,c->dom_active); if(!Lsc.cpu){orki_bdestroy(fd,&Lrc);return -2;}
    memcpy(Lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
    orki_setrn((uint32_t*)Lrc.cpu,REGCMD_SILU_LUT_N,RK_DPU_DST_BASE_ADDR,(uint32_t)Lsc.dma);
    if(lut){ uint32_t*lr=(uint32_t*)Lrc.cpu; int j=0;
        for(int k=0;k+1<REGCMD_SILU_LUT_N;k+=2){ if((lr[k]&0xffff)==0x4104){ int32_t v=(j<nlut)?(int32_t)lut[j]:0; j++;
            lr[k]=0x4104|((uint32_t)(v&0xffff)<<16); lr[k+1]=(0x1001u<<16)|(((uint32_t)v>>16)&0xffff); } } }
    orki_bsync(fd,&Lrc,RKNPU_MEM_SYNC_TO_DEVICE);
    /* Stream the LUT into PPU SRAM ONCE (enable 0x18) — it persists across the matmul submits below, so we
     * DON'T reload per M-tile (that was ~half the submits at prefill). Then per N-slice x M-tile: matmul +
     * fused-SiLU via orki_submit1() (warmed c->Cc + correct domain — a hand-rolled submit to a fresh buffer gives
     * bias-only output; submit1's warmed c->Cc is the validated bit-exact path vs the probe). */
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
    for(int ns=0;ns<w->Sn && rc_ret==0;ns++){ int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
        uint64_t wbase=w->Bf[ns].dma;
        orki_bsync(fd,&w->Bf[ns],RKNPU_MEM_SYNC_TO_DEVICE);   /* re-sync the resident weight to device */
        for(int m0=0;m0<M && rc_ret==0;m0+=chunk){ int mc=(M-m0<chunk)?(M-m0):chunk; if(mc<=0)continue;
            int8_t*ad=c->Af.cpu; for(int r=0;r<mc;r++)for(int j=0;j<K;j++) ad[(size_t)r*K+j]=A[(size_t)(m0+r)*K+j];
            orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
            /* matmul + fused-SiLU output stage -> int8 in c->Cc (via submit1: warmup + domain) */
            uint32_t rc[REGCMD_I8_N];
            orki_synth_i8(rc,mc,K,Nc,(uint32_t)c->Af.dma,(uint32_t)wbase,(uint32_t)c->Cc.dma,1,CBUF,0);
            orki_set_i8_silu(rc,Nc,0,r_mult,r_shift,out_bias,idx_off,cfg4068);
            if(orki_validate_regcmd("i8_silu",c,rc,REGCMD_I8_N,w,NULL,0)){ rc_ret=-1; break; }   /* stamp real op/weight so a submit-failure dump isn't a STALE probe label (mis-diagnosis) + sanity-check addrs */
            memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
            { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
              t->enable_mask=0x1d; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=108; t->regcmd_addr=c->regcmd.dma;
              orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); }
            if(orki_submit1(c)){ rc_ret=-1; break; }
            int8_t*cc=c->Cc.cpu; for(int r=0;r<mc;r++)for(int n=0;n<Nc;n++) C[(size_t)(m0+r)*N+(n0+n)]=cc[(size_t)r*Nc+n];
        }
    }
    orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);
    return rc_ret;
}

int ork_mm_run_i8_silu32(ork_npu *c,ork_w *w,int M,const int8_t *A,int32_t *C,
                         int r_mult,int r_shift,uint32_t out_bias,uint32_t idx_off,uint32_t cfg4068,
                         const int16_t *lut,int nlut){
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(w->dtype!=DT_I8 || !w->Bf) return -2;
    int fd=c->fd,K=w->K,N=w->N,NMAX=c->soc->nmax,CBUF=c->soc->cbuf_elems;
    if(K%512 || K>4096 || N%32) return -2;
    if(w->domain!=c->dom_active || (w->domain!=0 && !c->dom_save)) orki_dom_activate(c,w->domain);
    if(DT_I8!=c->last_dt){ c->warmed=0; c->ccsz=0; c->last_dt=DT_I8; }
    int chunk=orki_fused_mtile(K,M);
    size_t maxaf=(size_t)chunk*K, maxout=(size_t)chunk*NMAX*4;   /* int32 output: 4 bytes/elem */
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
    for(int ns=0;ns<w->Sn && rc_ret==0;ns++){ int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
        uint64_t wbase=w->Bf[ns].dma;
        orki_bsync(fd,&w->Bf[ns],RKNPU_MEM_SYNC_TO_DEVICE);
        for(int m0=0;m0<M && rc_ret==0;m0+=chunk){ int mc=(M-m0<chunk)?(M-m0):chunk; if(mc<=0)continue;
            int8_t*ad=c->Af.cpu; for(int r=0;r<mc;r++)for(int j=0;j<K;j++) ad[(size_t)r*K+j]=A[(size_t)(m0+r)*K+j];
            orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
            uint32_t rc[REGCMD_I8_N];
            orki_synth_i8(rc,mc,K,Nc,(uint32_t)c->Af.dma,(uint32_t)wbase,(uint32_t)c->Cc.dma,1,CBUF,0);  /* default = int32 out */
            orki_set_i8_silu32(rc,Nc,r_mult,r_shift,out_bias,idx_off,cfg4068);
            memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
            { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
              t->enable_mask=0x1d; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=108; t->regcmd_addr=c->regcmd.dma;
              orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); }
            if(orki_submit1(c)){ rc_ret=-1; break; }
            /* output element width from the sweep (int16=2,int32=4). Sign-extend to the int32 C[] the caller reads. */
            static int obytes=0; if(!obytes){ const char*e=getenv("ORK_SILU_OBYTES"); obytes=e?atoi(e):2; }
            if(obytes==2){ int16_t*cc=(int16_t*)c->Cc.cpu; for(int r=0;r<mc;r++)for(int n=0;n<Nc;n++) C[(size_t)(m0+r)*N+(n0+n)]=cc[(size_t)r*Nc+n]; }
            else         { int32_t*cc=(int32_t*)c->Cc.cpu; for(int r=0;r<mc;r++)for(int n=0;n<Nc;n++) C[(size_t)(m0+r)*N+(n0+n)]=cc[(size_t)r*Nc+n]; }
        }
    }
    orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);
    return rc_ret;
}

double orki_silu_f(double x);   /* fwd decl (defined below near the other activation refs) */
/* ork_mm_build_f16_lut — GENERIC fp16 fused-output PWL LUT builder (the fp16 twin of the int8/int16
 * orki_act_lut_i8/i16(fn,...) path). Bakes an arbitrary scalar fn(x) into the SDP output-stage LUT for inputs
 * x in [in_lo,in_hi]. The fp16 SDP index only spreads for NEGATIVE acc, so a reduce/gate matmul packed as
 * -S*W (S returned) drives acc=-S*x into the spread band; runtime C_out = R*LUT[idx(acc)] and the caller
 * recovers fn(x) = C_out*out_scale. 2-pass self-contained probe: measure R/bias, then idx(acc), then fill
 * LUT[idx] with fn at the sample points + interpolate. S = ATGT/max|in| (ORK_F16_ATGT); out_scale =
 * max_n|fn(sample_n)|/8000. fn(x,ctx) carries params via ctx (e.g. {n_feat,eps} for rsqrt; NULL for silu).
 * 0/ok, -2 fail. Consolidates the former ork_mm_build_f16_silu_lut + _rsqrt_lut (now thin wrappers). */
/* fn adapters for ork_mm_build_f16_lut (silu takes no params; rsqrt carries n_feat/eps in ctx) */

/* ---- FUSED matmul + output-stage activation: C = fn(A·B) in ONE submit (the "no-crossing chain") --------
 * The activation rides the matmul's DPU output stage — no separate activation submit, no CPU<->NPU crossing,
 * no mode-switch re-warm. This is the ONLY regime where an on-NPU non-matmul op beats CPU/NEON (a standalone
 * op is submit-floor-bound; see RE-roadmap M4.6). Mechanism: build the fn PWL LUT over the matmul-output range
 * [in_lo,in_hi] (ork_mm_build_f16_lut), pack the weight as -S*B so acc=-S*(A·B) lands in the fp16 index-spread
 * band, run the fused-LUT matmul (ork_mm_run_f16_silu is fn-agnostic — it applies whatever LUT it's given),
 * then recover fn(x)=C*out_scale. This is what realizes on-NPU softmax(exp)/RMSNorm(rsqrt)/SwiGLU(silu) as a
 * fused output stage rather than a losing standalone op. The matmul output MUST fall in [in_lo,in_hi] (the
 * LUT's calibrated band; values outside clamp to the edge). Single tile: K%32<=2048, N%16<=nmax. rk3588
 * PPU-fuse-gated. 0/ok, -2 shape/SoC(PPU off), -1 wedge/alloc. `w_scratch` unused (kept for ABI clarity). */
/* negate trampoline: build the LUT for g(u)=fn(-u) so a NEGATIVE-input range maps to positive u (the builder
 * needs positive input, and the fp16 SDP index only spreads one sign). */

/* PACK-ONCE fused-activation weight: calibrate the fn PWL LUT over [in_lo,in_hi] and pack the S-scaled weight,
 * baking BOTH into the resident ork_w (w->fa_lut / w->fa_osc). This factors the calibration OUT of the per-call
 * path so a fused-activation matmul can be RESIDENT — packed once, run many times via ork_mm_run_f16_fused_act
 * (no per-call LUT rebuild / re-pack), which is what lets fused exp/rsqrt/silu compose in a resident seq (a
 * standalone re-pack every call defeats residency). Same single-signed constraint as ork_mm_run_f16_act:
 * in_lo>=0 (positive-input, pack -S*B) or in_hi<=0 (negative-input via neg-trampoline, pack +S*B); mixed -> NULL.
 * The returned weight carries fn baked in — the run path needs NO fn pointer (so it crosses a socket/seq op). */

/* RUN a pre-packed fused-activation weight: C = fn(A·B) in ONE submit, reusing the resident LUT/scale. The
 * fused-activation twin of ork_mm_run_stream_f16 — no calibration, no re-pack. -2 if w has no baked LUT. */


/* Thin wrapper: fused SiLU LUT for a gate spanning [-Gmax,Gmax]. Caps Gmax (ORK_F16_GCAP, default 40) so S
 * stays in the fp16 index-spread band (a wide range collapses the mapping), then defers to the generic
 * builder. silu(gate) = C_out*out_scale at runtime; pack the gate weight as -S*W. See Exp-2026-07-05. */
/* Thin wrapper: fused rsqrt LUT baking 1/sqrt(ss/n_feat+eps) over ss in [ss_min,ss_max] — for on-NPU
 * rmsnorm (n_feat=n) / l2norm (n_feat=1). A reduce-matmul packed as -S*W emits the norm SCALE directly.
 * PRECISION VARIANTS: this is the fp16, fused-into-the-reduce rsqrt. The int8/int16 STANDALONE rsqrt of a
 * tensor is ork_npu_rsqrt_i8/i16 (built via orki_act_lut_i8/i16 with the same orki_rsqrt_f). Same op, different
 * precision + fusion regime — keep both. Build once per (layer,range) + cache (it runs probe submits). */

/* Resident-weight fused UP matmul + element-wise MULTIPLY by G (=silu(gate)) in the SDP output stage:
 * C = clamp_i8( round( (A·W_up) * G * gain ) ), gain = mult/2^shift = s_up*s_silu/s_out. Completes the
 * fused SwiGLU (gate via ork_mm_run_i8_silu -> G; up here). The 2nd operand G is fetched by the SDP
 * DPU_RDMA (0x5038) — set_i8_ewmul + orki_splice_ew_lane graft it onto ork's conv+int8-out program (regcfg
 * 108->126, enable 0x1d). Resident full-K int8 weight (K%512==0, K<=4096), N-tiled, single-core,
 * M-tile<=64/submit. G is dense int8 [M*N] (same layout as the output). rk3588 only. 0/ok,-1,-2,-3. */
int ork_mm_run_i8_ewmul(ork_npu *c,ork_w *w,int M,const int8_t *A,const int8_t *G,int8_t *C,int mult,int shift){
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(w->dtype!=DT_I8 || !w->Bf) return -2;
    int fd=c->fd,K=w->K,N=w->N,NMAX=c->soc->nmax,CBUF=c->soc->cbuf_elems;
    if(K%512 || K>4096 || N%32) return -2;
    if(w->domain!=c->dom_active || (w->domain!=0 && !c->dom_save)) orki_dom_activate(c,w->domain);
    if(DT_I8!=c->last_dt){ c->warmed=0; c->ccsz=0; c->last_dt=DT_I8; }
    int chunk=64; if(chunk>M)chunk=M;
    size_t maxaf=(size_t)chunk*K, maxout=(size_t)chunk*NMAX;
    if(c->Af.size<maxaf){ orki_bdestroy(fd,&c->Af); c->Af=orki_bcreate(fd,maxaf,0x403,c->dom_active); if(!c->Af.cpu)return -2; }
    if(c->ccsz<maxout){ orki_bdestroy(fd,&c->Cc); c->Cc=orki_bcreate(fd,maxout,0x403,c->dom_active); c->ccsz=maxout; c->warmed=0; if(!c->Cc.cpu)return -2; }
    /* 2nd-input (G) buffer, over-allocated >=64KiB so the captured 0x5020/0x5038 partner offsets land in-bounds */
    size_t gsz=(size_t)chunk*NMAX; if(gsz<0x10000)gsz=0x10000;
    struct buf Gb=orki_bcreate(fd,gsz,0x403,c->dom_active); if(!Gb.cpu)return -2;
    int rc_ret=0;
    for(int ns=0;ns<w->Sn && rc_ret==0;ns++){ int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
        uint64_t wbase=w->Bf[ns].dma;
        for(int m0=0;m0<M && rc_ret==0;m0+=chunk){ int mc=(M-m0<chunk)?(M-m0):chunk; if(mc<=0)continue;
            int8_t*ad=c->Af.cpu; for(int r=0;r<mc;r++)for(int j=0;j<K;j++) ad[(size_t)r*K+j]=A[(size_t)(m0+r)*K+j];
            orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
            int8_t*gd=Gb.cpu; memset(gd,0,gsz); for(int r=0;r<mc;r++)for(int n=0;n<Nc;n++) gd[(size_t)r*Nc+n]=G[(size_t)(m0+r)*N+(n0+n)];
            orki_bsync(fd,&Gb,RKNPU_MEM_SYNC_TO_DEVICE);
            orki_act(fd,RKNPU_ACT_RESET,0);
            uint32_t base[REGCMD_I8_N], rc[REGCMD_I8_EW_N];
            orki_synth_i8(base,mc,K,Nc,(uint32_t)c->Af.dma,(uint32_t)wbase,(uint32_t)c->Cc.dma,1,CBUF,0);
            orki_splice_ew_lane(rc,base);
            orki_set_i8_ewmul(rc,mc,Nc,0,mult,shift,(uint32_t)Gb.dma);
            memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
            { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
              t->enable_mask=0x1d; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=REGCMD_I8_EW_N/2; t->regcmd_addr=c->regcmd.dma;
              orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); }
            if(orki_submit1(c)){ rc_ret=-1; break; }   /* submit1: warmed c->Cc + correct domain (a hand-rolled submit gives bias-only) */
            int8_t*cc=c->Cc.cpu; for(int r=0;r<mc;r++)for(int n=0;n<Nc;n++) C[(size_t)(m0+r)*N+(n0+n)]=cc[(size_t)r*Nc+n];
        }
    }
    orki_bdestroy(fd,&Gb);
    return rc_ret;
}

int ork_mm_run_i8_out8(ork_npu *c,ork_w *w,int M,const int8_t *A,int8_t *C,int mult,int shift){
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(w->dtype!=DT_I8 || !w->Bf) return -2;
    int fd=c->fd,K=w->K,N=w->N,NMAX=c->soc->nmax,CBUF=c->soc->cbuf_elems;
    if(K%512 || K>4096 || N%32) return -2;
    if(w->domain!=c->dom_active || (w->domain!=0 && !c->dom_save)) orki_dom_activate(c,w->domain);
    if(DT_I8!=c->last_dt){ c->warmed=0; c->ccsz=0; c->last_dt=DT_I8; }
    int chunk=orki_fused_mtile(K,M);                          /* mg_max*64 M-tile per submit (was 64) */
    size_t maxaf=(size_t)chunk*K, maxout=(size_t)chunk*NMAX;
    /* realloc on size OR domain change (see ork_mm_run_i8_silu: cross-domain reuse faults the NPU) */
    if(c->Af.size<maxaf || c->Af.domain!=c->dom_active){ orki_bdestroy(fd,&c->Af); c->Af=orki_bcreate(fd,maxaf,0x403,c->dom_active); if(!c->Af.cpu)return -2; }
    if(c->ccsz<maxout || c->Cc.domain!=c->dom_active){ orki_bdestroy(fd,&c->Cc); c->Cc=orki_bcreate(fd,maxout,0x403,c->dom_active); c->ccsz=maxout; c->warmed=0; if(!c->Cc.cpu)return -2; }
    int rc_ret=0;
    for(int ns=0;ns<w->Sn && rc_ret==0;ns++){ int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
        uint64_t wbase=w->Bf[ns].dma; orki_bsync(fd,&w->Bf[ns],RKNPU_MEM_SYNC_TO_DEVICE);
        for(int m0=0;m0<M && rc_ret==0;m0+=chunk){ int mc=(M-m0<chunk)?(M-m0):chunk; if(mc<=0)continue;
            int8_t*ad=c->Af.cpu; for(int r=0;r<mc;r++)for(int j=0;j<K;j++) ad[(size_t)r*K+j]=A[(size_t)(m0+r)*K+j];
            orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
            uint32_t rc[REGCMD_I8_N];
            orki_synth_i8(rc,mc,K,Nc,(uint32_t)c->Af.dma,(uint32_t)wbase,(uint32_t)c->Cc.dma,1,CBUF,0);
            orki_set_i8_out8(rc,Nc,0,mult,shift);
            memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
            { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
              t->enable_mask=0xd; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=108; t->regcmd_addr=c->regcmd.dma;
              orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); }
            if(orki_submit1(c)){ rc_ret=-1; break; }
            int8_t*cc=c->Cc.cpu; for(int r=0;r<mc;r++)for(int n=0;n<Nc;n++) C[(size_t)(m0+r)*N+(n0+n)]=cc[(size_t)r*Nc+n];
        }
    }
    return rc_ret;
}

int ork_mm_run_i8_out16(ork_npu *c,ork_w *w,int M,const int8_t *A,short *C,int mult,int shift){
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(w->dtype!=DT_I8 || !w->Bf) return -2;
    int fd=c->fd,K=w->K,N=w->N,NMAX=c->soc->nmax,CBUF=c->soc->cbuf_elems;
    if(K%512 || K>4096 || N%32) return -2;
    if(w->domain!=c->dom_active || (w->domain!=0 && !c->dom_save)) orki_dom_activate(c,w->domain);
    if(DT_I8!=c->last_dt){ c->warmed=0; c->ccsz=0; c->last_dt=DT_I8; }
    int chunk=orki_fused_mtile(K,M);
    size_t maxaf=(size_t)chunk*K, maxout=(size_t)chunk*NMAX*2;                 /* int16 output = 2 bytes/elem */
    if(c->Af.size<maxaf || c->Af.domain!=c->dom_active){ orki_bdestroy(fd,&c->Af); c->Af=orki_bcreate(fd,maxaf,0x403,c->dom_active); if(!c->Af.cpu)return -2; }
    if(c->ccsz<maxout || c->Cc.domain!=c->dom_active){ orki_bdestroy(fd,&c->Cc); c->Cc=orki_bcreate(fd,maxout,0x403,c->dom_active); c->ccsz=maxout; c->warmed=0; if(!c->Cc.cpu)return -2; }
    int rc_ret=0;
    for(int ns=0;ns<w->Sn && rc_ret==0;ns++){ int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
        uint64_t wbase=w->Bf[ns].dma; orki_bsync(fd,&w->Bf[ns],RKNPU_MEM_SYNC_TO_DEVICE);
        for(int m0=0;m0<M && rc_ret==0;m0+=chunk){ int mc=(M-m0<chunk)?(M-m0):chunk; if(mc<=0)continue;
            int8_t*ad=c->Af.cpu; for(int r=0;r<mc;r++)for(int j=0;j<K;j++) ad[(size_t)r*K+j]=A[(size_t)(m0+r)*K+j];
            orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
            uint32_t rc[REGCMD_I8_N];
            orki_synth_i8(rc,mc,K,Nc,(uint32_t)c->Af.dma,(uint32_t)wbase,(uint32_t)c->Cc.dma,1,CBUF,0);
            orki_set_i16_out(rc,Nc,0,mult,shift);                                  /* int32 acc -> int16 LINEAR out */
            memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
            { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
              t->enable_mask=0xd; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=108; t->regcmd_addr=c->regcmd.dma;
              orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); }
            if(orki_submit1(c)){ rc_ret=-1; break; }
            short*cc=c->Cc.cpu; for(int r=0;r<mc;r++)for(int n=0;n<Nc;n++) C[(size_t)(m0+r)*N+(n0+n)]=cc[(size_t)r*Nc+n];   /* int16 LINEAR readback */
        }
    }
    return rc_ret;
}

int ork_npu_ewmul_i8(ork_npu *c,const int8_t *up,const int8_t *silu,int M,int N,int mult,int shift,int8_t *out,double *us){
    if(c && c->daemon){ if(us)*us=0; return orkd_ewmul_i8(c->daemon,up,silu,M,N,mult,shift,out); }   /* Path B: SDP on the daemon */
    int fd=c->fd, dom=c->dom_active;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<16||N>8192||(N&15)) return -2;             /* N multiple of the int8 atom (16) */
    if(mult<0||mult>0x7fff||shift<0||shift>31) return -2;
    #define EWCUBE(m,n) (((n)/16)*(M*16) + (m)*16 + ((n)%16))    /* NVDLA cube, atom=16, surf_stride=M*16 */
    size_t sz=(size_t)M*N; if(sz<4096)sz=4096;                    /* int8 cube = M*N bytes */
    struct buf A=orki_bcreate(fd,sz,0x403,dom); if(!A.cpu)return -2;
    struct buf B=orki_bcreate(fd,sz,0x403,dom); if(!B.cpu){orki_bdestroy(fd,&A);return -2;}
    struct buf O=orki_bcreate(fd,sz,0x403,dom); if(!O.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);return -2;}
    memset(A.cpu,0,sz);memset(B.cpu,0,sz);memset(O.cpu,0,sz);
    int8_t*ac=A.cpu,*bc=B.cpu;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int p=EWCUBE(m,n); ac[p]=up[m*N+n]; bc[p]=silu[m*N+n]; }
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    /* no per-call ACT_RESET: it costs tens of ms/op and is not needed for the SDP element-wise op
     * (validated bit-exact without it) — the reset only mattered for entering int8-matmul mode. */
    uint32_t rc[REGCMD_MUL_N]; memcpy(rc,REGCMD_MUL,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_MUL_N,M,N);
    orki_setrn(rc,REGCMD_MUL_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);        /* output */
    orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5018,(uint32_t)A.dma);        /* up  (SRDMA)  */
    orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5038,(uint32_t)B.dma);        /* silu (ERDMA) */
    orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)mult);         /* OUT_CVT_SCALE = gain mantissa */
    orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)shift);        /* OUT_CVT_SHIFT */
    orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_OFFSET,0);                      /* zo = 0 (OUT_CVT_OFFSET) */
    orki_setrn(rc,REGCMD_MUL_N,RK_DPU_BS_ALU_CFG,0);                      /* za = 0 (BS_ALU_OPERAND) */
    orki_setrn(rc,REGCMD_MUL_N,RK_DPU_EW_CVT_OFFSET,0);                      /* zb = 0 (EW_CVT_OFFSET) */
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms(); double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,dom)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ int8_t*oc=O.cpu; for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[m*N+n]=oc[EWCUBE(m,n)]; if(us)*us=t1; }
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&O);
    #undef EWCUBE
    return ok;
}


int ork_npu_row_max_i8(ork_npu *c, const int8_t *a, int M, int N, int8_t *out, double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<16||N>8192||(N&15)) return -2;
    #define RMCUBE(m,n) (((n)/16)*(M*16) + (m)*16 + ((n)%16))
    size_t sz=(size_t)M*N; if(sz<4096)sz=4096;
    struct buf W0=orki_bcreate(fd,sz,0x403,-1), W1=orki_bcreate(fd,sz,0x403,-1);
    if(!W0.cpu||!W1.cpu){ orki_bdestroy(fd,&W0);orki_bdestroy(fd,&W1); return -2; }
    memset(W0.cpu,0,sz); memset(W1.cpu,0,sz);
    { int8_t*wc=W0.cpu; for(int m=0;m<M;m++)for(int n=0;n<N;n++) wc[RMCUBE(m,n)]=a[(size_t)m*N+n]; }
    orki_bsync(fd,&W0,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&W1,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (XP_SDP=RC_SDPKW, keep-warm-aware; == the old inline reset) */
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18;
    struct buf *cur=&W0,*oth=&W1; int ok=0, cur_n=N; double t0=ork_now_us();
    while(cur_n>16){
        int h=cur_n/2;
        uint32_t rc[REGCMD_ADD_N]; memcpy(rc,REGCMD_ADD,sizeof rc);
        orki_set_mul_geom(rc,REGCMD_ADD_N,M,h);
        orki_setrn(rc,REGCMD_ADD_N,RK_DPU_DST_BASE_ADDR,(uint32_t)oth->dma);                                  /* out */
        orki_setrn(rc,REGCMD_ADD_N,RK_SDP_5018,(uint32_t)cur->dma);                                  /* a = ch [0,h) */
        orki_setrn(rc,REGCMD_ADD_N,RK_SDP_5038,(uint32_t)(cur->dma+(uint32_t)((size_t)(h/16)*M*16))); /* b = ch [h,2h) */
        orki_setrn(rc,REGCMD_ADD_N,RK_DPU_OUT_CVT_SCALE,0x4000); orki_setrn(rc,REGCMD_ADD_N,RK_DPU_OUT_CVT_SHIFT,28);      /* identity out-cvt */
        orki_setrn(rc,REGCMD_ADD_N,RK_DPU_EW_CVT_SCALE,0x4000);                                              /* identity b-scale */
        orki_setrn(rc,REGCMD_ADD_N,RK_DPU_BS_ALU_CFG,0); orki_setrn(rc,REGCMD_ADD_N,RK_DPU_EW_CVT_OFFSET,0); orki_setrn(rc,REGCMD_ADD_N,RK_DPU_OUT_CVT_OFFSET,0);
        orki_setrn(rc,REGCMD_ADD_N,RK_DPU_EW_CFG,0x904002c0);                                          /* EW_ALU_ALGO = MAX(0) */
        memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
        orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
        struct rknpu_submit sub; memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=1;
        sub.task_obj_addr=c->task.obj; sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1;
        sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; sub.timeout=orki_ew_timeout_ms();
        if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; break; }
        { struct buf*t=cur; cur=oth; oth=t; } cur_n=h;
    }
    if(ok==0){
        orki_bsync(fd,cur,RKNPU_MEM_SYNC_FROM_DEVICE); int8_t*r=cur->cpu;
        for(int m=0;m<M;m++){ int mx=-128; for(int n=0;n<cur_n;n++){ int v=r[RMCUBE(m,n)]; if(v>mx)mx=v; } out[m]=(int8_t)mx; }
        if(us)*us=ork_now_us()-t0;
    }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bdestroy(fd,&W0); orki_bdestroy(fd,&W1);
    #undef RMCUBE
    return ok;
}

int ork_npu_mul_perchan_i8(ork_npu *c,const int8_t *a,const int8_t *b,int M,int N,int mult,int shift,int8_t *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<16||N>8192||(N&15)) return -2;
    #define PCCUBE(m,n) (((n)/16)*(M*16) + (m)*16 + ((n)%16))
    size_t sz=(size_t)M*N; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,-1), O=orki_bcreate(fd,sz,0x403,-1), B=orki_bcreate(fd,sz,0x403,-1);
    if(!A.cpu||!O.cpu||!B.cpu){ orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&B); return -2; }
    memset(A.cpu,0,sz); memset(O.cpu,0,sz); memset(B.cpu,0,sz);
    { int8_t*ac=A.cpu; for(int m=0;m<M;m++)for(int n=0;n<N;n++) ac[PCCUBE(m,n)]=a[(size_t)m*N+n]; }
    { int8_t*bc=B.cpu; for(int n=0;n<N;n++) bc[n]=b[n]; }                        /* per-channel vector: CONTIGUOUS [N] (not surface-strided) */
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (XP_SDP=RC_SDPKW, keep-warm-aware; == the old inline reset) */
    uint32_t rc[REGCMD_MUL_N]; memcpy(rc,REGCMD_MUL,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_MUL_N,M,N);
    orki_setrn(rc,REGCMD_MUL_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);            /* output */
    orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5018,(uint32_t)A.dma);            /* a (SRDMA, per-element) */
    orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5038,(uint32_t)B.dma);            /* b (ERDMA / EW_BASE) */
    orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5034,0x00000004);                /* ERDMA_DATA_MODE=0 (per-channel) + DATA_SIZE=1 */
    orki_setrn(rc,REGCMD_MUL_N,RK_DPU_BS_ALU_CFG,0); orki_setrn(rc,REGCMD_MUL_N,RK_DPU_EW_CVT_OFFSET,0); orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_OFFSET,0); /* za/zb/zo = 0 (drop captured zero-points) */
    orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)mult); orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)shift);
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub; memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=1;
    sub.task_obj_addr=c->task.obj; sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1;
    sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; sub.timeout=orki_ew_timeout_ms();
    int ok=-1; double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; if(us)*us=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ int8_t*oc=O.cpu; for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=oc[PCCUBE(m,n)]; }
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&O); orki_bdestroy(fd,&B);
    #undef PCCUBE
    return ok;
}

int ork_npu_add_i8(ork_npu *c,const int8_t *a,const int8_t *b,int M,int N,
                   double a_scale,double b_scale,double out_scale,int8_t *out,double *us){
    if(c && c->daemon){ if(us)*us=0; return orkd_add_i8(c->daemon,a,b,M,N,a_scale,b_scale,out_scale,out); }   /* Path B: SDP on the daemon */
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<16||N>8192||(N&15)||out_scale<=0) return -2;
    double ca=a_scale/out_scale, cb=b_scale/out_scale, cmax=(ca>cb?ca:cb); if(cmax<=0) return -2;
    /* mults are Q(S) with headroom: keep <=0x4000 (validated safe range; 0x4000 == coeff 1 at S=14). */
    int S=14; while(S>0 && cmax*(double)(1u<<S) > 0x4000) S--;
    while(S<30 && cmax*(double)(1u<<(S+1)) <= 0x4000) S++;
    long ma=lround(ca*(double)(1u<<S)), mb=lround(cb*(double)(1u<<S));
    if(ma>0x4000)ma=0x4000; if(mb>0x4000)mb=0x4000;
    return ork_npu_probe_add_i8(c,a,b,M,N,(int)ma,S+14,(uint32_t)mb,0,0,0,out,us);
}

int ork_mm_silu_build_lut(ork_npu*c, double in_scale, double out_scale,
                          int r_mult, int r_shift, uint32_t cfg4068, int16_t *lut){
    return orki_chain_build_lut_fn(c, orki_silu_f, in_scale, out_scale, r_mult, r_shift, cfg4068, lut);
}

int orki_silu_calibrate_idx(ork_npu *c){
    if(c->silu_idx_ok) return 0;
    const int M=4,N=64;                       /* 256 elems = each int8 value exactly once */
    int8_t in[256],out[256]; int16_t lut[1030];
    for(int i=0;i<256;i++) in[i]=(int8_t)(i-128);
    for(int i=0;i<1030;i++){ int v=i-512; if(v>32767)v=32767; if(v<-32768)v=-32768; lut[i]=(int16_t)v; }
    if(ork_npu_probe_silu_std(c,in,M,N,0x2000,14,0,ORK_SILU_IDXOFF,ORK_SILU_C4064,ORK_SILU_C4068,lut,1030,out,0)) return -1;
    for(int v=0;v<256;v++) c->silu_idx[v]=-1;
    for(int i=0;i<M*N;i++){ int v=(uint8_t)in[i]; int o=out[i]; if(o>-127&&o<127) c->silu_idx[v]=(short)(2*o+512); }
    c->silu_idx_ok=1; return 0;
}

void orki_silu_build_curve_biased(ork_npu *c,double(*f)(double),double in_scale,double out_scale,double bias,int16_t *lut){
    int set[1030]; for(int i=0;i<1030;i++){lut[i]=0;set[i]=0;}
    for(int vv=-128;vv<128;vv++){ int idx=c->silu_idx[(uint8_t)vv]; if(idx<0||idx>1029)continue;
        double val=f((vv-bias)*in_scale)/out_scale; long q=lround(val); if(q>32767)q=32767; if(q<-32768)q=-32768;
        lut[idx]=(int16_t)q; set[idx]=1; }
    int lo=-1,hi=-1; for(int i=0;i<1030;i++)if(set[i]){lo=i;break;} for(int i=1029;i>=0;i--)if(set[i]){hi=i;break;}
    if(lo<0)return; for(int i=0;i<lo;i++)lut[i]=lut[lo]; for(int i=hi+1;i<1030;i++)lut[i]=lut[hi];
    for(int i=lo;i<=hi;i++){ if(set[i])continue; int a=i,b=i; while(a>lo&&!set[a])a--; while(b<hi&&!set[b])b++;
        lut[i]=(int16_t)(lut[a]+(lut[b]-lut[a])*(i-a)/(b-a)); }
}

void orki_silu_build_curve(ork_npu *c,double(*f)(double),double in_scale,double out_scale,int16_t *lut){
    orki_silu_build_curve_biased(c,f,in_scale,out_scale,0.0,lut);   /* plain curve = no bias */
}

static int act_lut_i8_biased(ork_npu *c,double(*f)(double),double bias,const int8_t *in,int M,int N,double in_scale,double out_scale,int8_t *out,double *us){
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<16||N>8192||(N&15)) return -2;
    if(orki_silu_calibrate_idx(c)) return -1;
    int16_t lut[1030]; orki_silu_build_curve_biased(c,f,in_scale,out_scale,bias,lut);
    return ork_npu_probe_silu_std(c,in,M,N,0x4000,14,0,ORK_SILU_IDXOFF,ORK_SILU_C4064,ORK_SILU_C4068,lut,1030,out,us);
}

int orki_act_lut_i8(ork_npu *c,double(*f)(double),const int8_t *in,int M,int N,double in_scale,double out_scale,int8_t *out,double *us){
    return act_lut_i8_biased(c,f,0.0,in,M,N,in_scale,out_scale,out,us);
}

int ork_npu_silu_i8(ork_npu *c,const int8_t *in,int M,int N,double in_scale,double out_scale,int8_t *out,double *us){
    if(c && c->daemon){ if(us)*us=0; return orkd_silu_i8(c->daemon,in,M,N,in_scale,out_scale,out); }   /* Path B: SDP on the daemon */
    return orki_act_lut_i8(c,orki_silu_f,in,M,N,in_scale,out_scale,out,us);
}

int ork_npu_gelu_i8(ork_npu *c,const int8_t *in,int M,int N,double in_scale,double out_scale,int8_t *out,double *us){
    if(c && c->daemon){ if(us)*us=0; return orkd_gelu_i8(c->daemon,in,M,N,in_scale,out_scale,out); }   /* Path B: SDP on the daemon */
    return orki_act_lut_i8(c,orki_gelu_f,in,M,N,in_scale,out_scale,out,us);
}

int ork_npu_rsqrt_i8(ork_npu *c,const int8_t *in,int M,int N,double in_scale,double out_scale,int8_t *out,double *us){
    if(c && c->daemon){ if(us)*us=0; return orkd_rsqrt_i8(c->daemon,in,M,N,in_scale,out_scale,out); }   /* Path B: SDP on the daemon */
    return orki_act_lut_i8(c,orki_rsqrt_f,in,M,N,in_scale,out_scale,out,us);
}

int ork_npu_exp_i8(ork_npu *c,const int8_t *in,int M,int N,double in_scale,double out_scale,int8_t *out,double *us){
    if(c && c->daemon){ if(us)*us=0; return orkd_exp_i8(c->daemon,in,M,N,in_scale,out_scale,out); }   /* Path B: SDP on the daemon */
    return orki_act_lut_i8(c,orki_exp_f,in,M,N,in_scale,out_scale,out,us);
}

int ork_npu_exp_i8_biased(ork_npu *c,const int8_t *in,int M,int N,double in_scale,double out_scale,double max,int8_t *out,double *us){
    return act_lut_i8_biased(c,orki_exp_f,max,in,M,N,in_scale,out_scale,out,us);
}

int orki_silu_calibrate_idx16(ork_npu *c){
    if(c->silu_idx16_ok) return 0;
    const int M=64,N=64;                      /* 4096 samples across the full int16 range (step 16) */
    static int16_t in[SILU16_NS],out[SILU16_NS]; int16_t lut[1030];
    for(int s=0;s<SILU16_NS;s++) in[s]=(int16_t)(-32768 + s*SILU16_QSTEP);
    for(int i=0;i<1030;i++){ int v=i-512; if(v>32767)v=32767; if(v<-32768)v=-32768; lut[i]=(int16_t)v; }
    /* NB: runs LAZILY on the first silu call — in the FFN chain that's right after a MULTI-CORE matmul,
     * so this pure-SDP probe hits the chain-context wedge (retry does NOT help — it wedges every attempt
     * even after soft-resets). See ork_npu_probe_silu_std_i16 (#35). Standalone it's clean. */
    if(ork_npu_probe_silu_std_i16(c,in,M,N,0x4000,14,0,ORK_SILU16_IDXOFF,ORK_SILU16_C4064,ORK_SILU16_C4068,lut,1030,out,0)) return -1;
    for(int s=0;s<SILU16_NS;s++){ int o=out[s]; c->silu_idx16[s]=(o>-490&&o<510)?(short)(o+512):(short)-32768; }
    c->silu_idx16_ok=1; return 0;
}


int ork_mm_run_stream_i8(ork_npu *c, int S, const ork_mm_task_i8 *tasks) {
    if (!c || S < 1 || !tasks) return -2;
    /* per-core scratch lives in the active domain; stream tasks share one domain (tasks[0].w) */
    if (tasks[0].w && (tasks[0].w->domain != c->dom_active || (tasks[0].w->domain!=0 && !c->dom_save))) orki_dom_activate(c, tasks[0].w->domain);
    const int mrc_cap = 65536 / (REGCMD_I8_N * 4);
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        if (!w || w->dtype != DT_I8 || tasks[i].M <= 0) return -2;
        if (w->Sn != 1 || !w->Bf) return -2;
        // The full-K Bf single-submit is only schedule-valid for K%512==0 && K<=4096 (same envelope as
        // orki_run()'s M>1 Bf path; the 0x1040 K-reduction schedule breaks outside it). Caller must fall back
        // to per-task run_i8 (which K-splits) for other K. Return -3 so it's distinguishable.
        if (w->K % 512 != 0 || w->K > 4096) return -3;
        if ((tasks[i].M + orki_chain_fullk_mcap_i8(c, w->K) - 1) / orki_chain_fullk_mcap_i8(c, w->K) > mrc_cap) return -2;
    }
    /* P3 SPINE MIGRATION (submit consolidation). The accepted shapes (Sn==1, K%512==0, K<=4096, Bf) are
     * exactly the doorbell's multi-core int8 envelope, so route the S independent tasks onto ONE NONBLOCK
     * doorbell submit (ork_dyn_begin_mc, block-distributed across cores) instead of the blocking round-robin
     * stream_worker. NO legacy fallback — a doorbell miss returns -1 (ork_dyn_end auto-dumps the stuck
     * descriptor). ork_dyn_begin_mc owns the mode-enter, per-core scratch sizing, and the per-task copy-back
     * to each tasks[i].C. (Interleave-safe: the colsplit/mc scratch now full-surface seeds+polls — see
     * colsplit's full-surface SENT seed + always-clean bsync — so a doorbell stream group no longer leaves the
     * shared per-core scratch dirty in a way that races an interleaved single run_i8.) */
    int nc = orki_budget(c, 2); if (nc > ORK_MAXCORE) nc = ORK_MAXCORE; if (nc > S) nc = S; if (nc < 1) nc = 1;
    ork_dyn_chain *h = ork_dyn_begin_mc(c, S, tasks, nc);
    if (!h) return -1;
    int d = ork_dyn_end(h);
    return (d == S - 1) ? 0 : -1;
}

int ork_mm_run_stream_i8_sk(ork_npu *c, int S, const ork_mm_task_i8 *tasks){
    if(!c||S<1||!tasks) return -2;
    if(tasks[0].w && (tasks[0].w->domain!=c->dom_active || (tasks[0].w->domain!=0 && !c->dom_save))) orki_dom_activate(c,tasks[0].w->domain);
    size_t maxMK=0, maxMN4=0;
    for(int i=0;i<S;i++){ ork_w *w=tasks[i].w;
        if(!w||w->dtype!=DT_I8||tasks[i].M<=0) return -2;
        if(w->Sn!=1||w->Sk!=1||!w->Bb) return -2;              /* single-slice int8 (K<=ks,N<=nmax) */
        if(w->K%32||w->N%16) return -2;
        size_t mk=(size_t)tasks[i].M*w->K, mn=(size_t)tasks[i].M*w->N*4;
        if(mk>maxMK)maxMK=mk; if(mn>maxMN4)maxMN4=mn; }
    int fd=c->fd;
    /* int8-live entry (last_dt=3); keep-warm across int8<->fp16 stage transitions under ORK_SSM_KEEPWARM */
    ork_npu_enter(c,3,XP_STREAM_I8,OCK_SW);  /* small-K int8 stream: same →I8_CHAIN transition as run_stream_i8 (profiles converged 2026-07-14) */
    int nc=orki_budget(c,2); if(nc>ORK_MAXCORE)nc=ORK_MAXCORE; if(nc>S)nc=S; if(nc<1)nc=1;
    if(orki_mc_ensure(c,nc)) return -1;
    for(int i=0;i<nc;i++){
        if(c->maf[i].size<maxMK){ orki_bdestroy(fd,&c->maf[i]); c->maf[i]=orki_bcreate(fd,maxMK,0x403,c->dom_active); if(!c->maf[i].cpu)return -1; }
        if(c->mccsz[i]<maxMN4){ orki_bdestroy(fd,&c->mcc[i]); c->mcc[i]=orki_bcreate(fd,maxMN4,0x403,c->dom_active); c->mccsz[i]=maxMN4; if(!c->mcc[i].cpu)return -1; c->mwarm[i]=0; } }
    int rc=0; orki_npu_pool_ensure(c);
    struct streamw_i8sk sw[ORK_MAXCORE]; int ctr=0;
    for(int i=0;i<nc;i++) sw[i]=(struct streamw_i8sk){c,i,S,tasks,&ctr,0};
    pthread_mutex_lock(&c->pmu);
    c->pjob=sw; c->pjob_nc=nc; c->pjob_fn=orki_stream_worker_i8sk; c->pjob_stride=sizeof(struct streamw_i8sk);
    c->pdone=0; c->pgen++; pthread_cond_broadcast(&c->pgo);
    pthread_mutex_unlock(&c->pmu);
    orki_stream_worker_i8sk(&sw[0]);                    /* core 0 on the calling thread */
    pthread_mutex_lock(&c->pmu); while(c->pdone<nc-1) pthread_cond_wait(&c->pdn,&c->pmu); pthread_mutex_unlock(&c->pmu);
    for(int i=0;i<nc;i++) if(sw[i].rc) rc=-1;
    c->warmed=1;
    return rc;
}

ork_async *ork_mm_run_i8_async (ork_npu *c, ork_w *w, int M, const int8_t  *A, int32_t *C){
    if (!c || !w || w->dtype != DT_I8 || M < 1) return NULL;
    return ork_async_launch((struct ork_async){ .kind=OAK_I8, .c=c, .w=w, .M=M, .A=A, .C=C }); }

ork_async *ork_mm_run_stream_i8_async(ork_npu *c, int S, const ork_mm_task_i8 *tasks){
    if (!c || S < 1 || !tasks) return NULL;
    return ork_async_launch((struct ork_async){ .kind=OAK_STREAM_I8, .c=c, .S=S, .tasks=tasks }); }

int ork_bmm_i8_strided(ork_npu *c, int nbatch, int M, int K, int N,
                       const int8_t *A, const int8_t *B, int32_t *C, const ork_bmm_strides *s){
    if(!c||!A||!B||!C||!s) return -1;
    if(nbatch<1||M<1||K<1||N<1) return -2;
    if(K%32||N%32) return -2;
    int8_t *Ac=malloc((size_t)M*K), *Bc=malloc((size_t)K*N);
    int cdense=orki_bmm_c_dense(s,N); int32_t *Cc = cdense?NULL:malloc((size_t)M*N*sizeof(int32_t));
    if(!Ac||!Bc||(!cdense&&!Cc)){ free(Ac);free(Bc);free(Cc); return -3; }
    orki_bmm_gather_i8(Bc,B+s->abs*0+s->bbs*0,K,N,s->bs_k,s->bs_n); /* batch-0 B for the pack */
    ork_w *w = ork_mm_pack_i8(c, K, N, Bc);
    int rc=0;
    if(!w) rc=-3;
    for(int b=0;!rc&&b<nbatch;b++){
        if(b>0){ orki_bmm_gather_i8(Bc,B+(long)b*s->bbs,K,N,s->bs_k,s->bs_n);
                 if(ork_mm_repack_i8(c,w,K,N,Bc)){ rc=-4; break; } }
        orki_bmm_gather_i8(Ac,A+(long)b*s->abs,M,K,s->as_m,s->as_k);
        int32_t *Cout = cdense ? C+(long)b*s->cbs : Cc;
        if(ork_mm_run_i8(c,w,M,Ac,Cout)){ rc=-5; break; }
        if(!cdense) orki_bmm_scatter_i32(C+(long)b*s->cbs,Cc,M,N,s->cs_m,s->cs_n);
    }
    if(w) ork_mm_free(c,w);
    free(Ac);free(Bc);free(Cc);
    return rc;
}

int ork_bmm_i8(ork_npu *c, int nbatch, int M, int K, int N,
               const int8_t *A, const int8_t *B, int32_t *C){
    ork_bmm_strides s=orki_bmm_natural(M,K,N); return ork_bmm_i8_strided(c,nbatch,M,K,N,A,B,C,&s);
}
