/* npu/i8/dyn_ctl.c — the doorbell CONTROL surface: spin probe, anomaly dump, step accounting, append, mid-flight halt, and multi-core resubmit recovery.
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

int ork_dyn_spin_probe(ork_npu *c, int S, const ork_mm_task_i8 *tasks, int spin_us, int *spin_alive) {
    if (!c || S < 1 || S > 500 || !tasks) return -1;
    for (int i = 0; i < S; i++) { ork_w *w = tasks[i].w;
        if (!w || w->dtype != DT_I8 || tasks[i].M != 1 || w->Sn != 1 || w->K % 512 || w->K > 4096) return -1;
        if (!orki_dma_find(c, (void*)tasks[i].C)) return -1; }
    if (tasks[0].w->domain != c->dom_active || (tasks[0].w->domain && !c->dom_save)) orki_dom_activate(c, tasks[0].w->domain);
    ork_npu_enter(c, 3 /*DT_I8_CHAIN*/, XP_CHAIN_NT, OCK_HW);
    int fd = c->fd, CBUF = c->soc->cbuf_elems, N = tasks[0].w->N; unsigned dom = tasks[0].w->domain;
    int P = S + 1;   /* program 0 = spin; 1..S = real tasks 0..S-1 */
    if ((size_t)P * REGCMD_I8_N * 4 > c->regcmd.size) return -1;
    struct buf ascr[512]; int nascr = 0;
    struct buf spinC = orki_bcreate(fd, (size_t)N * 4, 0x403, c->dom_active);
    #define SPIN_CLEAN() do { for (int j=0;j<nascr;j++) orki_bdestroy(fd,&ascr[j]); orki_bdestroy(fd,&spinC); } while (0)
    if (!spinC.cpu) { return -1; }
    uint32_t rc[REGCMD_I8_N + 4];
    /* program 0: spin = task[0]'s matmul writing spinC, self-looping */
    { ork_w *w = tasks[0].w; int K = w->K;
      struct buf s = orki_bcreate(fd,(size_t)K,0x403,c->dom_active); if(!s.cpu){SPIN_CLEAN();return -1;}
      memcpy(s.cpu,tasks[0].A,(size_t)K); orki_bsync(fd,&s,RKNPU_MEM_SYNC_TO_DEVICE); ascr[nascr++]=s;
      uint32_t bdma=w->Bf?(uint32_t)w->Bf[0].dma:(uint32_t)w->Bb[0].dma;
      memset(rc,0,sizeof rc); orki_synth_i8(rc,1,K,N,(uint32_t)s.dma,bdma,(uint32_t)spinC.dma,1,CBUF,0);
      uint64_t self=c->regcmd.dma;   /* self-loop */
      rc[216]=0x0010|((self&0xffff)<<16); rc[217]=(0x0101<<16)|((self>>16)&0xffff);
      rc[218]=0x0014|(0x0037u<<16); rc[219]=(0x0101<<16);
      memcpy((char*)c->regcmd.cpu,rc,REGCMD_I8_N*4); }
    int32_t *outptr[500];
    for (int i=0;i<S;i++){ int p=i+1; ork_w *w=tasks[i].w; int K=w->K;
      struct buf s=orki_bcreate(fd,(size_t)K,0x403,c->dom_active); if(!s.cpu){SPIN_CLEAN();return -1;}
      memcpy(s.cpu,tasks[i].A,(size_t)K); orki_bsync(fd,&s,RKNPU_MEM_SYNC_TO_DEVICE); ascr[nascr++]=s;
      struct buf *cb=orki_dma_find(c,(void*)tasks[i].C);
      uint32_t cdma=(uint32_t)(cb->dma+((const char*)tasks[i].C-(const char*)cb->cpu));
      uint32_t bdma=w->Bf?(uint32_t)w->Bf[0].dma:(uint32_t)w->Bb[0].dma;
      memset(rc,0,sizeof rc); orki_synth_i8(rc,1,K,N,(uint32_t)s.dma,bdma,cdma,1,CBUF,0);
      if(p<P-1){ uint64_t nx=c->regcmd.dma+(size_t)(p+1)*REGCMD_I8_N*4;
        rc[216]=0x0010|((nx&0xffff)<<16); rc[217]=(0x0101<<16)|((nx>>16)&0xffff);
        rc[218]=0x0014|(0x0037u<<16); rc[219]=(0x0101<<16); }
      memcpy((char*)c->regcmd.cpu+(size_t)p*REGCMD_I8_N*4,rc,REGCMD_I8_N*4); outptr[i]=(int32_t*)tasks[i].C; }
    orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *t=c->task.cpu; memset(t,0,(size_t)P*sizeof *t);
    for(int p=0;p<P;p++){ t[p].enable_mask=0xd;t[p].int_mask=0x300;t[p].int_clear=0x1ffff;t[p].regcfg_amount=108;t[p].regcmd_addr=c->regcmd.dma+(size_t)p*REGCMD_I8_N*4; }
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    { volatile int32_t*sd=(volatile int32_t*)((int32_t*)spinC.cpu+(N-1)); *sd=ORK_DYN_SENT; __asm__ volatile("dc cvac,%0"::"r"(sd):"memory"); }
    for(int i=0;i<S;i++){ volatile int32_t*db=(volatile int32_t*)(outptr[i]+(N-1)); *db=ORK_DYN_SENT; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); }
    __asm__ volatile("dsb ish":::"memory");
    struct rknpu_submit sub; memset(&sub,0,sizeof sub);
    sub.flags=ork_ppflags()|0x2u; sub.task_number=P; sub.task_obj_addr=c->task.obj; sub.core_mask=1; sub.fence_fd=-1;
    sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)P};
    sub.timeout=orki_mm_timeout_ms();
    c->warmed=1;
    if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ SPIN_CLEAN(); return -1; }
    if(spin_us>0){ struct timespec ts={spin_us/1000000,(long)(spin_us%1000000)*1000}; nanosleep(&ts,0); }
    /* liveness: spin slot written (loop ran) AND no real output touched (loop parked, didn't leak forward) */
    int spinran=0,leaked=0;
    { volatile int32_t*sd=(volatile int32_t*)((int32_t*)spinC.cpu+(N-1)); __asm__ volatile("dc civac,%0"::"r"(sd):"memory"); if(*sd!=ORK_DYN_SENT)spinran=1; }
    for(int i=0;i<S;i++){ volatile int32_t*db=(volatile int32_t*)(outptr[i]+(N-1)); __asm__ volatile("dc civac,%0"::"r"(db):"memory"); if(*db!=ORK_DYN_SENT)leaked++; }
    if(spin_alive) *spin_alive = (spinran && leaked==0);
    /* REDIRECT program 0 -> program 1 (jump into the real chain) */
    { uint32_t*p0=(uint32_t*)((char*)c->regcmd.cpu); uint64_t nx=c->regcmd.dma+(size_t)REGCMD_I8_N*4;
      p0[216]=0x0010|((nx&0xffff)<<16); p0[217]=(0x0101<<16)|((nx>>16)&0xffff);
      p0[218]=0x0014|(0x0037u<<16); p0[219]=(0x0101<<16);
      __asm__ volatile("dc cvac,%0"::"r"(&p0[216]):"memory"); __asm__ volatile("dc cvac,%0"::"r"(&p0[218]):"memory"); __asm__ volatile("dsb ish":::"memory"); }
    double t0=ork_now_us(); for(;;){ int done=0; for(int i=0;i<S;i++){volatile int32_t*db=(volatile int32_t*)(outptr[i]+(N-1)); __asm__ volatile("dc civac,%0"::"r"(db):"memory"); if(*db!=ORK_DYN_SENT)done++;} if(done>=S||ork_now_us()-t0>2e6)break; }
    for(int i=0;i<S;i++){ struct buf*cb=orki_dma_find(c,(void*)tasks[i].C); orki_bsync(fd,cb,RKNPU_MEM_SYNC_FROM_DEVICE); }
    int comp=0; for(int i=0;i<S;i++){ volatile int32_t*db=(volatile int32_t*)(outptr[i]+(N-1)); __asm__ volatile("dc civac,%0"::"r"(db):"memory"); if(*db!=ORK_DYN_SENT)comp++; }
    SPIN_CLEAN();
    return comp;
}

int ork_dyn_progress(ork_dyn_chain *h) { if (!h) return -1; int hi = -1;
    for (int i = 0; i < h->S; i++) if (ork_dyn_done_i(h, i)) hi = i;   /* per-row: task done = ALL its rows' last cols written */
    return hi; }

/* Chain-aware anomaly dump. Uses the doorbell DETECTOR (ork_dyn_progress) to name the STUCK descriptor — the
 * first op that did NOT land = progress+1 — and extracts THAT op's context: its regcmd slot DMA address, baked
 * output C address + current doorbell value, and the in-regcmd next-descriptor words 216..219 (feed to
 * tools/re/decode_reg for a register post-mortem). Plus the per-op doorbell map and the context-level
 * ork_npu_dump_state (freq/volt/hw_elapse/int_status). Fire on an anomaly BEFORE a wedge/reboot loses it.
 * Single-core chains (mc uses per-core regcmd buffers — only the map + context are shown there). */
void ork_dyn_dump(ork_dyn_chain *h, const char *label){
    if(!h) return; ork_npu *c=h->c; int N=h->N;
    ork_npu_dump_state(c,label);
    int prog=ork_dyn_progress(h), stuck=prog+1;
    fprintf(stderr,"[DYN-DUMP %s] S=%d P=%d reserve=%d spin_end=%d mc=%d | progress=%d landed => STUCK op #%d\n",
            label?label:"", h->S, h->P, h->reserve, h->spin_end, h->mc, prog, (stuck<h->P)?stuck:-1);
    fprintf(stderr,"  doorbell map [#=landed .=stuck]: ");
    for(int i=0;i<h->S;i++) fprintf(stderr,"%c", ork_dyn_done_i(h,i)?'#':'.');
    fprintf(stderr,"\n");
    if(!h->mc && stuck>=0 && stuck<h->reserve){
        /* dump the [prev, STUCK, next] descriptor window: the linkage around the stall. A correct chain has
         * prev.next-desc -> stuck's regcmd_dma, and stuck.next-desc -> next's regcmd_dma; a mismatch or a
         * bad address here IS the fault. Slots < S are real ops (have an output doorbell); >= S are spin/tail. */
        int lo=stuck-1; if(lo<0)lo=0; int hi=stuck+1; if(hi>=h->reserve)hi=h->reserve-1;
        for(int p=lo;p<=hi;p++){
            uint32_t *rc=(uint32_t*)((char*)c->regcmd.cpu+(size_t)p*REGCMD_I8_N*4);
            unsigned long long rdma=(unsigned long long)(c->regcmd.dma+(size_t)p*REGCMD_I8_N*4);
            const char *tag=(p<stuck)?"prev ":(p==stuck)?"STUCK":"next ";
            if(p<h->S && h->outptr[p]){ volatile int32_t *db=(volatile int32_t*)(h->outptr[p]+(N-1)); __asm__ volatile("dc civac,%0"::"r"(db):"memory");
                fprintf(stderr,"  [%s] op #%d: regcmd_dma=0x%llx C=%p doorbell=%d(0x%08x) next-desc=%08x %08x %08x %08x\n",
                        tag,p,rdma,(void*)h->outptr[p],(int)*db,(unsigned)*db,rc[216],rc[217],rc[218],rc[219]);
            } else {
                fprintf(stderr,"  [%s] slot #%d (spin/tail): regcmd_dma=0x%llx next-desc=%08x %08x %08x %08x\n",
                        tag,p,rdma,rc[216],rc[217],rc[218],rc[219]);
            }
        }
        uint32_t *rcs=(uint32_t*)((char*)c->regcmd.cpu+(size_t)stuck*REGCMD_I8_N*4);
        fprintf(stderr,"  regcmd[STUCK #%d] head (decode_reg):",stuck); for(int w=0;w<12;w++) fprintf(stderr," %08x",rcs[w]); fprintf(stderr," ...\n");
    }
}

int ork_dyn_max_steps(void) { return 1024; }             /* per-chain step cap: API arrays (regcmd holds ~2340, task buf ~13107) */

int ork_dyn_steps(ork_dyn_chain *h) { return h ? h->P : -1; }             /* total steps submitted in this chain */

int ork_dyn_remaining(ork_dyn_chain *h) { if (!h) return -1; int p = ork_dyn_progress(h); return h->P - (p + 1); }  /* steps not yet completed (budget left) */

int ork_dyn_append(ork_dyn_chain *h, const ork_mm_task_i8 *task) {
    if (!h || h->mc || !task) return -1;   /* mc handles use per-core buffers; append is single-buffer only */
    ork_npu *c = h->c; int fd = c->fd, CBUF = c->soc->cbuf_elems;
    ork_w *w = task->w;
    if (!w || w->dtype != DT_I8 || task->M != 1 || w->Sn != 1) return -1;
    if (w->K % 512 || w->K > 4096 || w->N != h->N) return -1;
    if (w->Sk != 1 && !w->Bf) return -1;
    if (h->P >= h->reserve) return -2;                 /* reserved budget exhausted — wrap to a fresh chain */
    int prog = ork_dyn_progress(h);
    if (prog >= h->P - 1 - ORK_DYN_HEADROOM) return 1; /* sequencer too close to the terminator — lost the race */
    int idx = h->P, K = w->K, N = w->N;
    struct buf *cb = orki_dma_find(c, (void*)task->C); if (!cb) return -1;
    uint32_t adma;                                     /* A -> scratch (zero-copy A miscomputes at M=1) */
    struct buf *ab = orki_dma_find(c, (void*)task->A);
    if (ab) { orki_bsync(fd, ab, RKNPU_MEM_SYNC_TO_DEVICE); adma = (uint32_t)(ab->dma + ((const char*)task->A - (const char*)ab->cpu)); }
    else { if (h->nascr >= 1024) return -2; struct buf s = orki_bcreate(fd, (size_t)K, 0x403, c->dom_active); if (!s.cpu) return -1;
           memcpy(s.cpu, task->A, (size_t)K); orki_bsync(fd, &s, RKNPU_MEM_SYNC_TO_DEVICE); adma = (uint32_t)s.dma; h->ascr[h->nascr++] = s; }
    uint32_t cdma = (uint32_t)(cb->dma + ((const char*)task->C - (const char*)cb->cpu));
    uint32_t bdma = w->Bf ? (uint32_t)w->Bf[0].dma : (uint32_t)w->Bb[0].dma;
    uint32_t rc[REGCMD_I8_N + 4]; memset(rc, 0, sizeof rc);
    orki_synth_i8(rc, 1, K, N, adma, bdma, cdma, 1, CBUF, 0);   /* new program: NO continue descriptor => it is the new terminator */
    memcpy((char*)c->regcmd.cpu + (size_t)idx * REGCMD_I8_N * 4, rc, REGCMD_I8_N * 4);
    h->outbuf[idx] = cb; h->outptr[idx] = (int32_t*)task->C;
    { volatile int32_t *db = (volatile int32_t*)(h->outptr[idx] + (N - 1)); *db = ORK_DYN_SENT; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); }
    /* flush the new program to DRAM BEFORE pointing the prior terminator at it (the sequencer must see a valid target) */
    { char *p = (char*)c->regcmd.cpu + (size_t)idx * REGCMD_I8_N * 4; for (size_t off = 0; off < (size_t)REGCMD_I8_N * 4; off += 64) __asm__ volatile("dc cvac,%0"::"r"(p + off):"memory"); }
    __asm__ volatile("dsb ish":::"memory");
    /* rewrite prior terminator (idx-1) -> continue into idx: the extend */
    uint32_t *pv = (uint32_t*)((char*)c->regcmd.cpu + (size_t)(idx - 1) * REGCMD_I8_N * 4);
    uint64_t nx = c->regcmd.dma + (size_t)idx * REGCMD_I8_N * 4;
    pv[216] = 0x0010 | ((nx & 0xffff) << 16); pv[217] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
    pv[218] = 0x0014 | (0x0037u << 16);        pv[219] = (0x0101 << 16);
    __asm__ volatile("dc cvac,%0"::"r"(&pv[216]):"memory"); __asm__ volatile("dc cvac,%0"::"r"(&pv[218]):"memory"); __asm__ volatile("dsb ish":::"memory");
    h->P++; h->S++;
    return 0;
}

int ork_dyn_halt(ork_dyn_chain *h, int at) { if (!h || h->mc || at < 0) return -1;   /* mc: per-core, no single-buffer halt */
    int _lim = h->spin_end ? h->spin_end : h->P; if (at >= _lim - 1) return -1;   /* halt within real programs, or within the spin tail if present */
    uint32_t *rcp = (uint32_t*)((char*)h->c->regcmd.cpu + (size_t)at * REGCMD_I8_N * 4);
    rcp[218] = 0x0014;   /* 0x0014 | amount 0 => sequencer stops after program `at` */
    __asm__ volatile("dc cvac,%0"::"r"(&rcp[218]):"memory"); __asm__ volatile("dsb ish":::"memory");
    return 0; }

int ork_dyn_end(ork_dyn_chain *h) { if (!h) return -1; int fd = h->c->fd;
    /* SPIN TEARDOWN (safety): a persistent spin tail keeps re-reading the scratch/regcmd after the real outputs
     * land, so freeing below would race an in-flight re-orki_run (IOMMU fault / wedge). Null-terminate EVERY reserved
     * spin slot first: wherever the sequencer currently is, its next slot is now terminal, so it stops within
     * ~1-2 tasks — then the drain + free are safe. Bounded, no race (all slots terminal, not a chased frontier). */
    if (h->spin_end) {
        for (int p = h->P; p < h->spin_end; p++) {
            uint32_t *rcp = (uint32_t*)((char*)h->c->regcmd.cpu + (size_t)p * REGCMD_I8_N * 4);
            rcp[218] = 0x0014; __asm__ volatile("dc cvac,%0"::"r"(&rcp[218]):"memory");
        }
        __asm__ volatile("dsb ish":::"memory");
    }
    /* Wait until EVERY task's every row is done (not just the highest index — multi-core cores finish
     * out of order, so a high task done does NOT imply the lower ones are). 500us-no-progress = stall/halt. */
    orki_in_doorbell = 1;   /* graceful SIGTERM: the poll below breaks on orki_ork_term and drains before the process ends */
    int edone[1024];
    int last = -1;
    int recov_max = (h->mc_nc > 0 && (h->mc_dt == DT_I8 || h->mc_dt == DT_I4)) ? 6 : 0;   /* mc int8/int4 rounds carry stashed context to resubmit; a few retries clear a sticky/correlated drop */
    double t0 = ork_now_us();
    if (h->prepolled) { last = h->S - 1; for (int i = 0; i < h->S; i++) edone[i] = 1; }   /* parallel colsplit: workers already drained all cores — skip the redundant poll (its miss-timeout was the ~500ms stall) */
    else for (int recov = 0; ; recov++) {
        t0 = ork_now_us(); int lastn = -1; double lastchg = t0;
        for (int i = 0; i < h->S; i++) edone[i] = 0;   /* per-entry done cache: stop re-polling (re-civac'ing) a core once it's complete */
        /* miss-detection timeout: a real mc int8 stream/decode op lands well under 300ms, so a sentinel still
         * stuck at that mark is the dropped-round miss (detect fast, resubmit cheap). Non-recoverable chains
         * (single-core, fp16, exhausted retries) keep the full 3s completion wait. */
        double miss_to = (recov < recov_max) ? (h->i4batch ? 2000000.0 : 300000.0) : 3e6;   /* #54 i4batch (M-batched BCHAIN): a legit job — esp. the first submit after an iommu-domain switch — can exceed 300ms; use the old bch 2s window so we don't false-recover a completing job. Per-row/int8 keep the fast 300ms detect. */
        for (;;) { int n = 0; for (int i = 0; i < h->S; i++) { if (!edone[i]) edone[i] = ork_dyn_done_i(h, i); n += edone[i]; }
            if (n >= h->S) break;                               /* all tasks, all rows */
            if (orki_ork_term) break;                              /* SIGTERM/SIGINT: stop waiting, drain + writeback below, then re-raise */
            /* The 500us-no-progress early break exists ONLY to detect a single-core halt/append that stopped
             * the chain short (ork_dyn_halt/append reject h->mc). For an mc chain there is no halt, so a plateau
             * just means cores haven't caught up yet — breaking there bailed with n<S. mc: wait all-done/timeout. */
            if (n != lastn) { lastn = n; lastchg = ork_now_us(); }
            else if (!h->mc && ork_now_us() - lastchg > 500.0) break;   /* single-core only: no progress => halted */
            double el = ork_now_us() - t0;
            if (el > miss_to) break;
            /* HEAVY-JOB BACKOFF: after a tight window that fully covers dispatch-bound work (decode ~<1ms), sleep
             * briefly between polls so the busy-poll's per-row civac stops contending with a large NPU writeback
             * (the M>1 prefill regression: M=256 was 2.8x slower under a pure spin). Decode stays tight (el<1ms =>
             * no sleep, no latency cost); a multi-ms prefill adds only ~poll granularity (<=50us on ~8ms). */
            if (el > 1000.0) { struct timespec ts = {0, 50000}; nanosleep(&ts, NULL); } }
        last = ork_dyn_progress(h);
        if (last >= h->S - 1 || orki_ork_term) break;            /* all done, or interrupted */
        if (recov < recov_max) {                               /* dropped mc int8 round (output never landed): recover + resubmit + re-poll */
            if (getenv("ORK_MC_DIAG")) fprintf(stderr, "[MC-RECOVER] mc int8 round output never landed (attempt %d) — reset + resubmit\n", recov);
            if (h->mc_dt == DT_I4) h->c->dom_dirty = 1;   /* #54: an int4 doorbell DROP happened -> a stuck job may linger in this domain even after recover (the reap fires only on the next SAME-DOMAIN submit, not across a switch). Mark so dom_activate reaps it (ork_dom_flush_if_dirty) BEFORE switching away -> the switch to the next domain won't time out. */
            orki_mc_recover_resubmit(h);
            continue;
        }
        break;   /* recovery exhausted / not recoverable -> fall through to trace + auto-dump */
    }
    if (getenv("ORK_DYN_TRACE")) { double _el = ork_now_us() - t0; int _nd = 0; for (int i = 0; i < h->S; i++) _nd += edone[i];
        fprintf(stderr, "[dyn_end] S=%d mc=%d done=%d/%d last=%d elapsed=%.0fus %s\n", h->S, h->mc, _nd, h->S, last, _el,
            _nd < h->S ? "INCOMPLETE(timeout/term)" : "all-done");
        for (int i = 0; i < h->S; i++) if (!edone[i]) { int Me = h->oM[i]?h->oM[i]:1, no = h->nout[i]?h->nout[i]:h->N, Nx = no/Me;
            fprintf(stderr, "  task%d NOT-done: outptr[Nx-1]=%d (SENT=%d) nout=%d oSk=%d ostride=%d dst=%p\n",
                i, h->outptr[i][Nx-1], (int)ORK_DYN_SENT, no, h->oSk[i], h->ostride[i], (void*)h->dst[i]); }
        fflush(stderr); }
    /* AUTO CORE-DUMP on a real miss: a plain chain that drained INCOMPLETE (not all S landed) is a doorbell
     * dispatch/completion failure — capture the stuck-descriptor post-mortem before the state is lost. Skip
     * intentionally-truncated chains (spin_end = bulk-terminated by design). This is the forcing function:
     * every path we move onto the spine auto-reports its misses, pointing at the doorbell fix. */
    if (last < h->S - 1 && !h->spin_end) {
        if (h->mc_nc > 0) {   /* DIAG: per-core submit rc (all 0 = the accepted-but-never-dispatched doorbell-drop, after
                               * orki_mc_recover_resubmit's retries also failed). NOTE: dma_rw/int_status read 0-always on this
                               * kernel, so they are NOT diagnostic — the output sentinel (this miss) is the only signal. */
            fprintf(stderr, "[MC-DIAG] nc=%d submit_rc=[", h->mc_nc);
            for (int i = 0; i < h->mc_nc && i < 8; i++) fprintf(stderr, "%d ", h->mc_rc[i]);
            fprintf(stderr, "] (recover exhausted; dma_rw/int_status unreliable on this kernel)\n");
        }
        ork_dyn_dump(h, "ork_dyn_end incomplete (doorbell miss)");
    }
    struct buf *done[1024]; int nd = 0;
    for (int i = 0; i < h->S; i++) { struct buf *b = h->outbuf[i]; int seen = 0;
        for (int j = 0; j < nd; j++) if (done[j] == b) seen = 1;
        if (!seen) { orki_bsync(fd, b, RKNPU_MEM_SYNC_FROM_DEVICE); if (nd < 1024) done[nd++] = b; } }
    if (h->i4batch) {   /* #54 BCHAIN de-tile: widen each core's int16 tiles -> caller's int32 C (mcc synced above). dst[i]=NULL so the generic writeback below skips these. */
        for (int i = 0; i < h->S; i++)
            orki_bch_db_cells(h->c, i, h->b_c0[i], h->b_c1[i], h->b_Wb, h->b_N, h->b_NG, h->b_M, h->b_H, h->b_Wmax, h->b_C, 2, -1);
    }
    /* mc: outputs were written to the in-domain per-core scratch (outptr) — copy each back to the caller's C.
     * int4 (esz==2): the NPU wrote an int16 accumulator; widen int16->int32 into the caller's int32 C. */
    int NMAXe = h->c->soc->nmax;
    for (int i = 0; i < h->S; i++) if (h->dst[i]) { int no = h->nout[i] ? h->nout[i] : h->N;
        int Me = h->oM[i] ? h->oM[i] : 1, Ne = no / Me;
        /* K-SPLIT ACCUMULATE: scratch holds oSk partial [M,N] blocks ([ks][m][n]); sum over ks into C[M,N].
         * esz==2 (int4, A2): the partials are int16 (widen-sum -> int32); else (int8) int32 partials (unchanged). */
        if (h->oSk[i] > 1) {
            int Sk = h->oSk[i], Nn = no / (Sk * Me); int32_t *d = h->dst[i];
            size_t ds = h->ostride[i] > 0 ? (size_t)h->ostride[i] : (size_t)Nn;   /* colsplit wide-K: land the summed [M,Ncore] in C's columns at row-stride N */
            if (h->mc_dt == DT_F16) {
                /* SINGLE full-surface civac VERIFY (this thread only, after every core's per-slice BLOCKING submits
                 * returned): the fp16 f16->f32 DPU writeback can lag the blocking-submit completion, so confirm every
                 * partial word has DRAINED (!= the SENT seed) before summing — else the accumulate reads a mid-drain
                 * partial (the ~1/40 wedge/wrong-answer). This is the documented completion protocol: ONE full-surface
                 * verify here, NOT a per-core 3-thread civac-scan (which thrashes the NPU writeback). fp16 K-split only. */
                { volatile int32_t *o = (volatile int32_t*)h->outptr[i]; double pt = ork_now_us();
                  for (;;) { int all = 1; for (int e = 0; e < no; e++) { __asm__ volatile("dc civac,%0"::"r"(&o[e]):"memory"); if (o[e] == ORK_DYN_SENT) { all = 0; break; } }
                      if (all) break; double el = ork_now_us() - pt; if (el > 3e6) break; if (el > 1000.0) { struct timespec ts = {0, 50000}; nanosleep(&ts, NULL); } } }
                const float *src = (const float*)h->outptr[i]; float *df = (float*)d;   /* colsplit wide-K fp16: sum Sk f32 partials, ks-ascending == mcworker, bit-exact. */
                const size_t kstride = (size_t)Me * Nn;
                for (int m = 0; m < Me; m++) { const float *base = src + (size_t)m * Nn; float *dr = df + (size_t)m * ds; int n = 0;
                    for (; n + 4 <= Nn; n += 4) { float32x4_t acc = vld1q_f32(base + n);
                        for (int ks = 1; ks < Sk; ks++) acc = vaddq_f32(acc, vld1q_f32(base + (size_t)ks * kstride + n));
                        vst1q_f32(dr + n, acc); }
                    for (; n < Nn; n++) { float acc = base[n]; for (int ks = 1; ks < Sk; ks++) acc += base[(size_t)ks * kstride + n]; dr[n] = acc; } } }
            else if (h->esz == 2) { const int16_t *src = (const int16_t*)h->outptr[i];
                for (int m = 0; m < Me; m++) for (int n = 0; n < Nn; n++) {
                    int64_t acc = 0; for (int ks = 0; ks < Sk; ks++) acc += src[(size_t)ks * Me * Nn + (size_t)m * Nn + n];
                    d[(size_t)m * ds + n] = (int32_t)acc; } }
            else { const int32_t *src = (const int32_t*)h->outptr[i];   /* colsplit wide-K int8: sum Sk int32 partials.
                * NEON over n (4-wide): integer add is associative + wraps mod 2^32 identically to the scalar
                * (int32_t)int64_acc, so BIT-EXACT. Row m of partial ks lives at src[ks*Me*Nn + m*Nn]. */
                const size_t kstride = (size_t)Me * Nn;
                for (int m = 0; m < Me; m++) {
                    const int32_t *base = src + (size_t)m * Nn; int32_t *dr = d + (size_t)m * ds; int n = 0;
                    for (; n + 4 <= Nn; n += 4) {
                        int32x4_t acc = vld1q_s32(base + n);
                        for (int ks = 1; ks < Sk; ks++) acc = vaddq_s32(acc, vld1q_s32(base + (size_t)ks * kstride + n));
                        vst1q_s32(dr + n, acc); }
                    for (; n < Nn; n++) { int32_t acc = base[n]; for (int ks = 1; ks < Sk; ks++) acc += base[(size_t)ks * kstride + n]; dr[n] = acc; } }
            }
        }
        else if (h->oscat[i]) {   /* BOUNDARY-SCATTER (balanced wide-N): scratch is segment-major [M,segw] blocks
            * (each program wrote contiguously => no notch). Place each block into C at its column sub-range,
            * cutting [c0,c0+Ne) at nmax slice boundaries — the exact inverse of begin_colsplit's emission. */
            const int32_t *src = (const int32_t*)h->outptr[i]; int32_t *d = h->dst[i];   /* d = C + c0 */
            int c0 = h->ocol0[i]; size_t blk = 0; int cur = c0, coff = 0;
            while (coff < Ne) {
                int ns = cur / NMAXe, sl1 = (ns + 1) * NMAXe; if (sl1 > h->N) sl1 = h->N;
                int segend = (c0 + Ne < sl1) ? (c0 + Ne) : sl1, segw = segend - cur;
                for (int m = 0; m < Me; m++) memcpy(&d[(size_t)m * h->ostride[i] + coff], &src[blk + (size_t)m * segw], (size_t)segw * 4);
                blk += (size_t)Me * segw; coff += segw; cur = segend;
            }
        }
        else
        /* SCATTER (M>1 wide-N): the scratch holds Sn contiguous [M,Nc] slice blocks; place each block into
         * C's column range [n0,n0+Nc) at row-stride Ne. (int8/int32 only — Sn>1 M>1 is not an int4/fp16 shape.) */
        if (h->esz != 2 && Me > 1 && Ne > NMAXe) {
            const int32_t *src = (const int32_t*)h->outptr[i]; int32_t *d = h->dst[i]; size_t blk = 0;
            for (int n0 = 0; n0 < Ne; n0 += NMAXe) { int Nc = (Ne - n0 < NMAXe) ? (Ne - n0) : NMAXe;
                for (int m = 0; m < Me; m++) memcpy(&d[(size_t)m * Ne + n0], &src[blk + (size_t)m * Nc], (size_t)Nc * 4);
                blk += (size_t)Me * Nc; }
        }
        else if (h->esz == 2) { const int16_t *o=(const int16_t*)h->outptr[i]; int32_t *d=h->dst[i]; for (int e=0;e<no;e++) d[e]=o[e]; }
        else if (h->ostride[i] > 0) {   /* colsplit M>1: [Me,Ne] scratch -> dst column-slice at row-stride ostride */
            const int32_t *src = (const int32_t*)h->outptr[i]; int32_t *d = h->dst[i];
            for (int m = 0; m < Me; m++) memcpy(&d[(size_t)m * h->ostride[i]], &src[(size_t)m * Ne], (size_t)Ne * 4); }
        else memcpy(h->dst[i], h->outptr[i], (size_t)no * 4); }
    __asm__ volatile("dsb ish":::"memory");   /* ensure the copy-back/scatter stores complete before the caller reads C (esp. a non-cacheable ork_dma_alloc dst) */
    for (int i = 0; i < h->nascr; i++) orki_bdestroy(fd, &h->ascr[i]);   /* free scratch A copies */
    int r = last; free(h);
    orki_in_doorbell = 0;
    if (orki_ork_term) {   /* a SIGTERM/SIGINT arrived mid-poll: we've drained + written back cleanly — now honor it */
        sigaction(SIGTERM, &orki_prev_sig[0], NULL); sigaction(SIGINT, &orki_prev_sig[1], NULL); raise(SIGTERM);
    }
    return r; }












void orki_mc_recover_resubmit(ork_dyn_chain *h);   /* shared doorbell recover (defined below); grouped drain rides it */

/* P3: sub-nmax N-COLUMN tiling across cores on the NONBLOCK doorbell (int8, Sn==1, K<=4096 Bf, M 1..64).
 * A single matmul C[M,N] is split by N-columns across nc cores exactly as run_multicore does (t0=i*NN/nc);
 * each core computes its [M,Ncore] column range into per-core scratch (M-tiled into orki_mtile_cap(K)-row chained
 * programs when M>cap), NONBLOCK, and end() copies each range back to C's columns [c0,c0+Ncore) at row-stride
 * N (strided for M>1). This gives the doorbell run_multicore's TILE-parallel multi-core for one matmul (the
 * op-partition would otherwise pin a single op to one core). Scratch+copy-back (not direct output): multi-core
 * direct output to a shared resident C is the unsafe ZC-OUT-multicore case; the copy-back after poll is
 * single-threaded => coherent. */
/* DOORBELL per-core parallel dispatch (DEFAULT; ORK_COLSPLIT_SERIAL forces the legacy inline path): each core
 * submits AND polls/accumulates its OWN core on its own pool thread
 * (barrier-synced fire), mirroring the mcworker/stream_worker. CRITICAL: the per-core poll is O(1) — just the
 * last output word — NOT ork_dyn_done_i's O(no) full-surface civac scan; 3 threads full-scanning ~190K words
 * each thrash the memory bus against the NPU writeback and serialize it (that was the earlier failure). The
 * one-time full-surface VERIFY still happens in ork_dyn_end after all cores land (recovery lives there too). */
double orki_f16_slice_us;   /* running max of a SUCCESSFULLY-landed fp16 K-slice completion (us). Drives the AUTO sentinel detect timeout = 1.5x this (adaptive per shape, vs a flat 800ms). Benign cross-core race — a heuristic, not correctness. */


void orki_mc_recover_resubmit(ork_dyn_chain *h){
    ork_npu *c = h->c; int fd = c->fd;
    if(getenv("ORK_MC_DIAG")) fprintf(stderr,"[mc-recover] doorbell MISS -> ACT_RESET + resubmit | dom=%d S=%d nc=%d esz=%d\n", h->mc_dom, h->S, h->mc_nc, h->esz);
    struct rknpu_action a; memset(&a, 0, sizeof a); a.flags = RKNPU_ACT_RESET; ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &a);
    { struct timespec ts = {0, 1000000}; nanosleep(&ts, NULL); }   /* let the reset fully settle before resubmit — a resubmit into a not-yet-quiesced NPU re-drops (sticky miss) */
    /* #54 MULTI-DOMAIN: the ACT_RESET above DROPS the NPU's IOMMU domain state. Resubmitting into a non-0
     * mc_dom then triggers a domain switch that TIMES OUT ("switch iommu domain time out, id: N") and poisons
     * all subsequent switches (dmesg-confirmed cascade). Re-establish mc_dom's page table with a fresh native
     * anchor BEFORE the resubmit so the switch lands cleanly. No-op for domain 0 (always established). */
    if(h->mc_dom > 0) ork_dom_reanchor(c, h->mc_dom);
    struct buf *cl[1024]; int ncl = 0;                                    /* re-clean output surfaces to DRAM */
    for (int x = 0; x < h->S; x++) { struct buf *b = h->outbuf[x]; int seen = 0;
        for (int j = 0; j < ncl; j++) if (cl[j] == b) seen = 1;
        if (!seen && b) { orki_bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE); if (ncl < 1024) cl[ncl++] = b; } }
    if (h->esz == 2) {   /* int4: int16 output, full-surface SENT16 (write-order not last-col-last) */
        for (int x = 0; x < h->S; x++) { int no = h->nout[x]; volatile int16_t *o = (volatile int16_t*)h->outptr[x];
            for (int e = 0; e < no; e++){ o[e] = ORK_DYN_SENT16; __asm__ volatile("dc cvac,%0"::"r"(&o[e]):"memory"); } }
    } else for (int x = 0; x < h->S; x++) { int Mx = h->oM[x]?h->oM[x]:1, Nx = h->nout[x]?h->nout[x]/Mx:h->N;   /* re-seed sentinels */
        if (h->mc_seed_all) for (int m=0;m<Mx;m++) for (int n=0;n<Nx;n++){ volatile int32_t *db=(volatile int32_t*)(h->outptr[x]+(size_t)m*Nx+n); *db=ORK_DYN_SENT; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); }
        else for (int m=0;m<Mx;m++){ volatile int32_t *db=(volatile int32_t*)(h->outptr[x]+(size_t)m*Nx+(Nx-1)); *db=ORK_DYN_SENT; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); } }
    __asm__ volatile("dsb ish":::"memory");
    for (int i = 0; i < h->mc_nc && i < ORK_MAXCORE; i++) if (h->mc_Pc[i]) {   /* resubmit each core */
        orki_bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd, &c->mrc[i], RKNPU_MEM_SYNC_TO_DEVICE);
        orki_bsync(fd, &c->mtk[i], RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
        h->mc_subs[i].timeout = (h->esz==2) ? orki_i4_submit_tmo_ms() : orki_mm_timeout_ms(); orki_rknpu_submit_ioctl(fd, &h->mc_subs[i], h->mc_dom); }   /* #54 int4 (esz==2): bounded timeout so a re-dropped recover job stays reapable (TCLEAN) */
}
