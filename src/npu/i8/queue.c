/* npu/i8/queue.c — thin wrappers OVER the dynamic chain: the chunk-pipeline submit queue (ork_dyn_queue_*) and the
 * precompiled-program cache (ork_pc_*, regime A -- fixed chain, pinned buffers)
 *
 * Part of the int8 datapath; shared declarations in npu/i8/i8.h. Split out of i8/dyn.c, which had grown
 * to hold three separate subsystems. */
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

ork_dyn_queue *ork_dyn_queue_create(ork_npu *c, int chunk_max, int ncore) {
    if (!c) return NULL; int mx = ork_dyn_max_steps(); if (chunk_max <= 0 || chunk_max > mx) chunk_max = mx;
    if (ncore <= 0) ncore = 1; if (ncore > c->soc->cores) ncore = c->soc->cores;
    ork_dyn_queue *q = calloc(1, sizeof *q); if (!q) return NULL;
    q->c = c; q->chunk_max = chunk_max; q->ncore = ncore; q->linger_us = ORK_SUBMIT_FLOOR_US; return q; }

void ork_dyn_queue_set_linger(ork_dyn_queue *q, int us) { if (q) q->linger_us = us; }   /* default = one submit floor */

int  ork_dyn_queue_linger_us(ork_dyn_queue *q) { return q ? q->linger_us : -1; }

int ork_dyn_queue_push(ork_dyn_queue *q, const ork_mm_task_i8 *task) {
    if (!q || !task) return -1;
    if (q->n == q->cap) { int nc = q->cap ? q->cap * 2 : 64; void *t = realloc(q->tasks, (size_t)nc * sizeof *q->tasks); if (!t) return -1; q->tasks = t; q->cap = nc; }
    q->tasks[q->n++] = *task; q->last_push_us = ork_now_us(); return 0; }

int ork_dyn_queue_flush(ork_dyn_queue *q) {
    if (!q) return -1; if (q->h || q->submitted >= q->n) return 0;
    int cnt = q->n - q->submitted; if (cnt > q->chunk_max) cnt = q->chunk_max;
    q->h = (q->ncore > 1) ? ork_dyn_begin_mc(q->c, cnt, q->tasks + q->submitted, q->ncore)
                          : ork_dyn_begin   (q->c, cnt, q->tasks + q->submitted);
    if (!q->h) return -1;
    q->submitted += cnt; return 0; }

int ork_dyn_queue_pending(ork_dyn_queue *q) { return q ? q->n - q->submitted : -1; }   /* not-yet-submitted count */

/* Idle-transition halt (the linger wiring): once the producer has drained the queue AND the linger window has
 * elapsed since the last push, null-terminate the flying chain just ahead of the sequencer (0x0014=0 via the
 * validated ork_dyn_halt) so a chain with unspent reserve/spin ahead of the frontier stops early and the NPU
 * goes idle instead of running out its reserved budget. linger_us is the grace window before giving up on more
 * work arriving. No-op (returns 0) if nothing is flying, work is still pending, we are within the linger window,
 * the chain is multi-core (halt is single-buffer only — mc self-terminates per-core), or the frontier is already
 * at the terminator. Returns 1 iff it halted. (Visible effect only for a reserved/persistent chain: a plain
 * self-terminating chunk already stops at its own frontier; this is a no-op for it, by design.) */
int ork_dyn_queue_idle(ork_dyn_queue *q) {
    if (!q || !q->h || q->submitted < q->n) return 0;                       /* nothing flying, or work still pending */
    if (ork_now_us() - q->last_push_us < (double)q->linger_us) return 0;    /* still inside the linger grace window */
    ork_dyn_chain *h = q->h;
    if (h->mc) return 0;                                                    /* mc: no single-buffer halt */
    int at = ork_dyn_progress(h) + 1 + ORK_DYN_HEADROOM;                    /* halt just ahead of the sequencer */
    int lim = h->spin_end ? h->spin_end : h->P;                            /* spin tail extends the haltable range past P */
    if (at >= lim - 1) return 0;                                          /* already at/near terminator — nothing to cut */
    return ork_dyn_halt(h, at) == 0 ? 1 : 0;
}

int ork_dyn_queue_drain(ork_dyn_queue *q) {
    if (!q) return -1; int done = 0; if (!q->h) ork_dyn_queue_flush(q);
    while (q->h) { int d = ork_dyn_end(q->h); q->h = NULL; if (d >= 0) done += d + 1;
        if (q->submitted < q->n) ork_dyn_queue_flush(q); }
    q->n = 0; q->submitted = 0; return done; }

void ork_dyn_queue_destroy(ork_dyn_queue *q) { if (!q) return; if (q->h) ork_dyn_end(q->h); free(q->tasks); free(q); }

ork_pc_chain *ork_pc_compile(ork_npu *c, int S, const ork_mm_task_i8 *tasks) {
    if (!c || S < 1 || S > 512 || !tasks) return NULL;
    for (int i = 0; i < S; i++) { ork_w *w = tasks[i].w;
        if (!w || w->dtype != DT_I8 || tasks[i].M != 1 || w->Sn != 1 || w->K % 512 || w->K > 4096) return NULL;
        if (w->Sk != 1 && !w->Bf) return NULL;
        if (!orki_dma_find(c, (void*)tasks[i].C)) return NULL; }
    if (tasks[0].w->domain != c->dom_active || (tasks[0].w->domain && !c->dom_save)) orki_dom_activate(c, tasks[0].w->domain);
    ork_npu_enter(c, 3 /*DT_I8_CHAIN*/, XP_CHAIN_NT, OCK_HW);
    int fd = c->fd, CBUF = c->soc->cbuf_elems, N = tasks[0].w->N;
    ork_pc_chain *pc = calloc(1, sizeof *pc); if (!pc) return NULL;
    pc->c = c; pc->S = S; pc->N = N; pc->dom = tasks[0].w->domain;
    /* Park the static regcmd pool + per-op A-scratch in on-chip NPU SRAM when present — it is otherwise-unused
     * memory, so residing there frees DRAM/IOVA for the rest of the pipeline. Detection-gated: bcreate drops
     * TRY_ALLOC_SRAM to DRAM when orki_sram_total==0 (stock kernel/DTB) and fails an over-budget/contended alloc
     * over to DRAM per-buffer (so large chains spill gracefully). Opt out with ORK_PC_NO_SRAM (benchmarks). */
    unsigned pcsf = getenv("ORK_PC_NO_SRAM") ? 0 : RKNPU_MEM_TRY_ALLOC_SRAM;
    pc->pool = orki_bcreate(fd, (size_t)S * REGCMD_I8_N * 4, 0x403 | pcsf, c->dom_active);
    if (!pc->pool.cpu) { free(pc); return NULL; }
    uint32_t rc[REGCMD_I8_N + 4];
    for (int i = 0; i < S; i++) { ork_w *w = tasks[i].w; int K = w->K;
        pc->ascr[i] = orki_bcreate(fd, (size_t)K, 0x403 | pcsf, c->dom_active);
        if (!pc->ascr[i].cpu) { for (int j=0;j<i;j++) orki_bdestroy(fd,&pc->ascr[j]); orki_bdestroy(fd,&pc->pool); free(pc); return NULL; }
        memcpy(pc->ascr[i].cpu, tasks[i].A, (size_t)K); orki_bsync(fd, &pc->ascr[i], RKNPU_MEM_SYNC_TO_DEVICE);
        pc->asrc[i] = tasks[i].A; pc->Ksz[i] = K;
        struct buf *cb = orki_dma_find(c, (void*)tasks[i].C);
        uint32_t cdma = (uint32_t)(cb->dma + ((const char*)tasks[i].C - (const char*)cb->cpu));
        uint32_t bdma = w->Bf ? (uint32_t)w->Bf[0].dma : (uint32_t)w->Bb[0].dma;
        memset(rc, 0, sizeof rc);
        orki_synth_i8(rc, 1, K, N, (uint32_t)pc->ascr[i].dma, bdma, cdma, 1, CBUF, 0);   /* A/C addresses BAKED IN */
        if (orki_validate_regcmd("ork_pc", c, rc, REGCMD_I8_N, w, pc->ascr, i+1)) { for (int j=0;j<=i;j++) orki_bdestroy(fd,&pc->ascr[j]); orki_bdestroy(fd,&pc->pool); free(pc); return NULL; }
        if (i < S - 1) { uint64_t nx = pc->pool.dma + (size_t)(i+1) * REGCMD_I8_N * 4;
            rc[216] = 0x0010 | ((nx & 0xffff) << 16); rc[217] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
            rc[218] = 0x0014 | (0x0037u << 16);        rc[219] = (0x0101 << 16); }
        memcpy((char*)pc->pool.cpu + (size_t)i * REGCMD_I8_N * 4, rc, REGCMD_I8_N * 4);
        pc->outbuf[i] = cb; pc->outptr[i] = (int32_t*)tasks[i].C;
    }
    orki_bsync(fd, &pc->pool, RKNPU_MEM_SYNC_TO_DEVICE);   /* pool is STATIC — synced once, never re-synth'd */
    return pc;
}

int ork_pc_run(ork_pc_chain *pc) {
    if (!pc) return -1; ork_npu *c = pc->c; int fd = c->fd, S = pc->S, N = pc->N;
    for (int i = 0; i < S; i++) { memcpy(pc->ascr[i].cpu, pc->asrc[i], (size_t)pc->Ksz[i]); orki_bsync(fd, &pc->ascr[i], RKNPU_MEM_SYNC_TO_DEVICE); }
    struct rknpu_task *t = c->task.cpu; memset(t, 0, (size_t)S * sizeof *t);
    for (int p = 0; p < S; p++) { t[p].enable_mask = 0xd; t[p].int_mask = 0x300; t[p].int_clear = 0x1ffff;
        t[p].regcfg_amount = 108; t[p].regcmd_addr = pc->pool.dma + (size_t)p * REGCMD_I8_N * 4; }
    orki_bsync(fd, &c->task, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub; memset(&sub, 0, sizeof sub);
    sub.flags = ork_ppflags() | 0x2u; sub.task_number = S; sub.task_obj_addr = c->task.obj; sub.core_mask = 1; sub.fence_fd = -1;
    sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)S};
    sub.timeout = orki_mm_timeout_ms();
    #define ORK_PC_SEED() do { for (int x=0;x<S;x++){ volatile int32_t*db=(volatile int32_t*)(pc->outptr[x]+(N-1)); *db=ORK_DYN_SENT; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); } __asm__ volatile("dsb ish":::"memory"); } while(0)
    ORK_PC_SEED();
    if (!pc->warmed) {   /* cold: throwaway NONBLOCK warm pass, poll, reseed (blocking multi-task submit EINVALs) */
        if (!orki_rknpu_submit_ioctl(fd, &sub, pc->dom)) { double tw=ork_now_us();
            for(;;){ int a=1; for(int x=0;x<S;x++){ volatile int32_t*db=(volatile int32_t*)(pc->outptr[x]+(N-1)); __asm__ volatile("dc civac,%0"::"r"(db):"memory"); if(*db==ORK_DYN_SENT){a=0;break;} } if(a||ork_now_us()-tw>2e6)break; } }
        pc->warmed = 1; ORK_PC_SEED();
    }
    if (orki_rknpu_submit_ioctl(fd, &sub, pc->dom)) return -1;
    double t0 = ork_now_us(); int last = -1;
    for (;;) { int hi=-1; for (int i=0;i<S;i++){ volatile int32_t*db=(volatile int32_t*)(pc->outptr[i]+(N-1)); __asm__ volatile("dc civac,%0"::"r"(db):"memory"); if(*db!=ORK_DYN_SENT)hi=i; }
        if (hi>=S-1){ last=hi; break; } if (ork_now_us()-t0>2e6){ last=hi; break; } }
    struct buf *done[512]; int nd=0;
    for (int i=0;i<S;i++){ struct buf*b=pc->outbuf[i]; int seen=0; for(int j=0;j<nd;j++) if(done[j]==b)seen=1; if(!seen){ orki_bsync(fd,b,RKNPU_MEM_SYNC_FROM_DEVICE); if(nd<512)done[nd++]=b; } }
    return last;
}

void ork_pc_free(ork_pc_chain *pc) { if (!pc) return; int fd = pc->c->fd;
    for (int i=0;i<pc->S;i++) orki_bdestroy(fd,&pc->ascr[i]); orki_bdestroy(fd,&pc->pool); free(pc); }
