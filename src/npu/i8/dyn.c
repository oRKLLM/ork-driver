/* npu/i8/dyn.c — the dynamic steered-submission API: NONBLOCK chains, doorbell progress, colsplit, the submit queue, the precompiled-program cache.
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

int ork_dyn_grouped_end(ork_dyn_chain *h);  /* B: grouped-int4 float scale-accumulate drain */
int ork_dyn_end(ork_dyn_chain *h);
/* TASK #4: multi-M int4 onto the NONBLOCK doorbell spine. Decompose the M rows of one int4 weight into a chain
 * of M=1 int4 programs distributed across cores via ork_dyn_begin_mc_i4 (the validated M=1 int4 doorbell: full-
 * surface int16 seed+poll, int16->int32 widen, and — task #4 — the same drop-recover as int8). Bit-identical to
 * per-row. Single-slice only (the doorbell's envelope); returns -4 (caller refuses — ORK_RC_WEDGE_PRONE) for
 * wide-N/K or when the M-program chain doesn't fit the per-core regcmd/task buffers. */
int orki_run_i4_mc_db(ork_npu *c, ork_w *w, int M, const int8_t *A, int32_t *C, int nc){
    ork_mm_task_i8 *tk = malloc((size_t)M * sizeof *tk); if(!tk) return -4;
    for(int m=0;m<M;m++) tk[m]=(ork_mm_task_i8){ w, 1, A + (size_t)m*w->K, C + (size_t)m*w->N };
    ork_dyn_chain *h = ork_dyn_begin_mc_i4(c, M, tk, nc);
    free(tk);
    if(!h) return -4;   /* shape/buffer limit -> caller refuses (ORK_RC_WEDGE_PRONE) */
    int d = ork_dyn_end(h);
    return (d == M-1) ? 0 : -1;
}

ork_dyn_chain *ork_dyn_begin_colsplit(ork_npu *c, const ork_mm_task_i8 *t, int ncreq);   /* fwd: fp16 colsplit routed from run_multicore */
#define ORK_RC_F16_SC (-502)   /* internal run_multicore->orki_run() signal: fp16 fallback, retry the single-core fp16 reference (never the blocking mcworker) */
void ork_install_term(void);   /* fwd: graceful-SIGTERM install (defined near the doorbell poll) */
/* SLICE-AND-DICE RESCUE (#33): a shape run_multicore has no verified single-submit path for would
 * return ORK_RC_WEDGE_PRONE. If orki_pack() pre-built doorbell tiles for it (w->sliced), RUN it on those
 * instead — one chained doorbell submit over c_base tiles, bit-exact. If there are no tiles (a shape
 * we don't pre-slice) OR the sliced run itself errors, REFUSE — never a blocking fall-back (#45). */
int orki_slice_rescue_or_refuse(ork_npu *c,ork_w *w,int M,const void *A,void *C,int nc){
    if(w->sliced){ int rs=ork_mm_run_sliced(c,w->sliced,M,A,C,nc); if(rs>=0) return rs; }
    return ORK_RC_WEDGE_PRONE;
}

int ork_dyn_done_i(ork_dyn_chain *h, int i){
    if (h->i4batch) {   /* #54 BCHAIN tile output: mode-1 last-program civac gate (cheap, per-poll), then on pass bsync + mode-3 full verify (once). Matches bch_db_worker's completion check; ork_dyn_end owns the recover. */
        ork_npu *c = h->c;
        if (!orki_bch_db_cells(c, i, h->b_c0[i], h->b_c1[i], h->b_Wb, h->b_N, h->b_NG, h->b_M, h->b_H, h->b_Wmax, NULL, 1, h->b_NT[i]-1)) return 0;
        orki_bsync(c->fd, &c->mcc[i], RKNPU_MEM_SYNC_FROM_DEVICE);
        return orki_bch_db_cells(c, i, h->b_c0[i], h->b_c1[i], h->b_Wb, h->b_N, h->b_NG, h->b_M, h->b_H, h->b_Wmax, NULL, 3, -1);
    }
    int M = h->oM[i] ? h->oM[i] : 1; int no = h->nout[i] ? h->nout[i] : h->N; int Nx = M ? no/M : no;
    if (h->esz == 2) {   /* int4: int16 output; its write-order over N is NOT last-col-last, so poll the FULL row */
        volatile int16_t *o = (volatile int16_t*)h->outptr[i];
        for (int e = 0; e < no; e++){ __asm__ volatile("dc civac,%0"::"r"(&o[e]):"memory"); if (o[e]==ORK_DYN_SENT16) return 0; }
        return 1; }
    int NMAXd = h->c->soc->nmax;
    if (h->oSk[i] > 1) {   /* K-SPLIT: oSk partial [M,N] blocks are one PC-chain (programs run sequentially), so the
        * LAST program's last word (base[no-1]) lands LAST -> an O(1) completion GATE. Only once it lands do the
        * full-surface VERIFY (covers any residual per-block lag). Was a full O(no) civac scan EVERY poll iteration,
        * pathological on the large wide-K prefill surface (~524K words -> ~10 t/s ffn_down). */
        volatile int32_t *base = (volatile int32_t*)h->outptr[i];
        __asm__ volatile("dc civac,%0"::"r"(&base[no-1]):"memory"); if (base[no-1]==ORK_DYN_SENT) return 0;   /* fast gate */
        for (int e = 0; e < no; e++){ __asm__ volatile("dc civac,%0"::"r"(&base[e]):"memory"); if (base[e]==ORK_DYN_SENT) return 0; }
        return 1; }
    if ((M > 1 && Nx > NMAXd) || h->oscat[i]) {   /* SCATTER layout: scratch is Sn contiguous [M,Nc] blocks. The block (stride=0)
        * output's write-order over N is NOT reliably last-col-last (like the int4 int16 output above), so a
        * per-row-last-col poll fires before the whole block drains -> partial scatter -> non-deterministic
        * zeros. Poll the FULL surface: done only when EVERY scratch word is non-sentinel. */
        volatile int32_t *base = (volatile int32_t*)h->outptr[i];
        for (int e = 0; e < no; e++){ __asm__ volatile("dc civac,%0"::"r"(&base[e]):"memory"); if (base[e]==ORK_DYN_SENT) return 0; }
        return 1; }
    for (int m = 0; m < M; m++){ volatile int32_t *db=(volatile int32_t*)(h->outptr[i]+(size_t)m*Nx+(Nx-1));
        __asm__ volatile("dc civac,%0"::"r"(db):"memory"); if (*db==ORK_DYN_SENT) return 0; }
    return 1; }

ork_dyn_chain *ork_dyn_begin(ork_npu *c, int S, const ork_mm_task_i8 *tasks) {
    ork_install_term();   /* graceful SIGTERM: make the async poll interruptible */
    if (!c || S < 1 || S > 1024 || !tasks) return NULL;   /* S==1 is valid: a 1-program chain (task_number=1 runs it) */
    for (int i = 0; i < S; i++) { ork_w *w = tasks[i].w;
        if (!w || w->dtype != DT_I8 || tasks[i].M != 1 || w->Sn != 1) return NULL;
        if (w->K % 512 || w->K > 4096) return NULL;
        if (w->Sk != 1 && !w->Bf) return NULL;
        if (!orki_dma_find(c, (void*)tasks[i].C)) return NULL; }   /* C must be resident (doorbell poll + writeback); A is copied to scratch */
    if (tasks[0].w->domain != c->dom_active || (tasks[0].w->domain && !c->dom_save)) orki_dom_activate(c, tasks[0].w->domain);
    ork_npu_enter(c, 3 /*DT_I8_CHAIN*/, XP_CHAIN_NT, OCK_HW);
    int fd = c->fd, CBUF = c->soc->cbuf_elems, P = S;
    ork_dyn_chain *h = calloc(1, sizeof *h); if (!h) return NULL;
    h->c = c; h->S = S; h->P = P; h->N = tasks[0].w->N; h->dom = tasks[0].w->domain;
    uint32_t rc[REGCMD_I8_N + 4];
    const void *seenA[1024]; uint32_t seenAdma[1024]; int nseenA = 0;   /* dedup scratch copies of a shared A pointer */
    for (int i = 0; i < S; i++) { ork_w *w = tasks[i].w; int K = w->K, N = w->N;
        struct buf *cb = orki_dma_find(c, (void*)tasks[i].C);
        /* A: copy into a scratch DMA buffer (zero-copy A is bit-wrong at M=1 — the known ZC-A M=1 bug),
         * deduped by pointer (a shared activation is copied once). */
        uint32_t adma;
        struct buf *ab = orki_dma_find(c, (void*)tasks[i].A);
        if (ab) { orki_bsync(fd, ab, RKNPU_MEM_SYNC_TO_DEVICE); adma = (uint32_t)(ab->dma + ((const char*)tasks[i].A - (const char*)ab->cpu)); }
        else { int hit = -1; for (int j = 0; j < nseenA; j++) if (seenA[j] == tasks[i].A) { hit = j; break; }
               if (hit >= 0) adma = seenAdma[hit];
               else { struct buf s = orki_bcreate(fd, (size_t)K, 0x403, c->dom_active); if (!s.cpu) { for (int j=0;j<h->nascr;j++) orki_bdestroy(fd,&h->ascr[j]); free(h); return NULL; }
                      memcpy(s.cpu, tasks[i].A, (size_t)K); orki_bsync(fd, &s, RKNPU_MEM_SYNC_TO_DEVICE); adma = (uint32_t)s.dma; h->ascr[h->nascr++] = s;
                      if (nseenA < 1024) { seenA[nseenA] = tasks[i].A; seenAdma[nseenA] = adma; nseenA++; } } }
        uint32_t cdma = (uint32_t)(cb->dma + ((const char*)tasks[i].C - (const char*)cb->cpu));
        uint32_t bdma = w->Bf ? (uint32_t)w->Bf[0].dma : (uint32_t)w->Bb[0].dma;
        memset(rc, 0, sizeof rc);
        orki_synth_i8(rc, 1, K, N, adma, bdma, cdma, 1, CBUF, 0);
        if (orki_validate_regcmd("ork_dyn", c, rc, REGCMD_I8_N, w, h->ascr, h->nascr)) { for (int j=0;j<h->nascr;j++) orki_bdestroy(fd,&h->ascr[j]); free(h); return NULL; }
        if (i < P - 1) { uint64_t nx = c->regcmd.dma + (size_t)(i+1) * REGCMD_I8_N * 4;
            rc[216] = 0x0010 | ((nx & 0xffff) << 16); rc[217] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
            rc[218] = 0x0014 | (0x0037u << 16);       rc[219] = (0x0101 << 16); }
        memcpy((char*)c->regcmd.cpu + (size_t)i * REGCMD_I8_N * 4, rc, REGCMD_I8_N * 4);
        h->outbuf[i] = cb; h->outptr[i] = (int32_t*)tasks[i].C;
    }
    /* RESERVE a budget: submit task_number=reserve (>= S) so the chain can be EXTENDED in-flight (ork_dyn_append)
     * up to `reserve`. task_number is IMMUTABLE post-submit — the HW walking past it aborts the job (measured:
     * -22), so the budget must be reserved up front. Reserved slots [S,reserve) are pre-filled with program S-1
     * (a valid terminator) so their task descriptors are valid; the HW never reaches them (the frontier
     * terminator halts it) until append fills them with real work. This is the pre-allocation the wrap needs. */
    int reserve = P; { const char *e = getenv("ORK_DYN_RESERVE"); if (e) { int r = atoi(e); if (r > reserve) reserve = r; if (reserve > 1024) reserve = 1024; } }
    h->reserve = reserve;
    for (int p = P; p < reserve; p++) memcpy((char*)c->regcmd.cpu + (size_t)p * REGCMD_I8_N * 4, rc, REGCMD_I8_N * 4);   /* rc still holds program S-1 (terminator) */
    orki_bsync(fd, &c->regcmd, RKNPU_MEM_SYNC_TO_DEVICE);   /* TERMINATING chain (0..S-1) for the warm pass; the spin tail (if any) is applied AFTER warm — see below */
    /* task_number=reserve PC-chain model (like run_chain_i8): kernel programs PC from task[0]; the HW walks via
     * each program's in-regcmd next-descriptor (0x0010/0x0014), bounded by task_number. ALL reserve descriptors
     * must be present (task_number=1 runs program 0 ONLY — measured; it does NOT walk). */
    struct rknpu_task *t = c->task.cpu;
    #define ORK_DYN_TASKS() do { memset(t, 0, (size_t)reserve * sizeof *t); for (int p = 0; p < reserve; p++) { \
        t[p].enable_mask = 0xd; t[p].int_mask = 0x300; t[p].int_clear = 0x1ffff; \
        t[p].regcfg_amount = 108; t[p].regcmd_addr = c->regcmd.dma + (size_t)p * REGCMD_I8_N * 4; } } while (0)
    ORK_DYN_TASKS();
    orki_bsync(fd, &c->task, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub; memset(&sub, 0, sizeof sub);
    sub.flags = ork_ppflags(); sub.task_number = reserve; sub.task_obj_addr = c->task.obj; sub.core_mask = 1; sub.fence_fd = -1;
    sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)reserve};
    sub.timeout = orki_mm_timeout_ms();
    int _dbg = getenv("ORK_DYN_DEBUG") != NULL;
    sub.flags |= 0x2u;   /* NONBLOCK: returns so the host can steer/poll */
    /* Cold mode-establishment needs a warm pass, but a BLOCKING multi-task submit (flags 0x5) EINVALs +
     * soft-resets here. So warm with a throwaway NONBLOCK pass (which works), poll it to completion, then
     * re-seed and run the real pass. Cold without any warm is flaky (miscomputes — measured 1/16). */
    #define ORK_DYN_SEED() do { for (int i = 0; i < S; i++) { volatile int32_t *db = (volatile int32_t*)(h->outptr[i] + (h->N - 1)); \
        *db = ORK_DYN_SENT; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); } __asm__ volatile("dsb ish":::"memory"); } while (0)
    if (!c->warmed) {
        ORK_DYN_SEED(); sub.timeout = orki_mm_timeout_ms();
        if (!orki_rknpu_submit_ioctl(fd, &sub, h->dom)) {                 /* warm pass */
            double tw = ork_now_us(); for (;;) { int alld = 1;
                for (int i = 0; i < S; i++) { volatile int32_t *db = (volatile int32_t*)(h->outptr[i] + (h->N - 1));
                    __asm__ volatile("dc civac,%0"::"r"(db):"memory"); if (*db == ORK_DYN_SENT) { alld = 0; break; } }
                if (alld || ork_now_us() - tw > 2e6) break; } }
        c->warmed = 1;
    }
    if (_dbg) fprintf(stderr, "[dyn] warm phase done (warmed=%d), spin=%d reserve=%d — applying spin + real submit next\n", c->warmed, (reserve > P) ? 1 : 0, reserve);
    /* PERSISTENT SPIN TAIL (on whenever reserve>P — a reserve budget implies a persistent spin) — applied HERE, AFTER warm, so the warm pass ran the
     * TERMINATING chain (0..S-1, the proven protocol — it completes cleanly, no in-flight job) and does NOT
     * collide with this real submit on the single-stream NPU (the overlap that hung/soft-reset the earlier
     * version). Forward-chain the reserved slots (S-1 -> S -> ... -> reserve-1; reserve-1 stays terminal) so the
     * real submit re-runs program S-1's matmul idempotently through the reserve budget and stays BUSY, haltable
     * early by ork_dyn_halt / ork_dyn_queue_idle (spin_end-aware). Bounded — no self-loop, no redirect. Progress
     * can't advance past S-1 in the spin (tail re-writes the same C), so a clean halt must lead the sequencer from
     * near the start; ork_dyn_end bulk-terminates the tail before freeing (safe teardown). Per-slot spin doorbell
     * for late/long-spin halting is a follow-up. */
    if (reserve > P) {
        /* DEDICATED NO-OP SPIN TAIL. Re-running program S-1 (the real matmul, SAME doorbell output C) as chained
         * tasks ABORTS the job — verified by bisection: reserve>P non-spin = 8/8, spin-by-re-run = 0/8 (whole job
         * yields nothing). So the tail is a dedicated matmul that reuses the resident weight B but reads a zeroed
         * throwaway spinA and writes a throwaway spinC (a plain bcreate buffer, NOT one of the polled doorbell
         * outputs) — so it never touches the real outputs' coherent buffers. Fill reserved slots [S..reserve-1]
         * with it, forward-chained (reserve-1 terminal), and chain the last real program S-1 -> slot S. Real
         * outputs 0..S-1 land; the tail spins harmlessly. spinA/spinC park in h->ascr (freed by ork_dyn_end). */
        ork_w *lw = tasks[S - 1].w; int lK = lw->K, lN = lw->N;
        uint32_t lB = lw->Bf ? (uint32_t)lw->Bf[0].dma : (uint32_t)lw->Bb[0].dma;
        struct buf spinA = orki_bcreate(fd, (size_t)lK, 0x403, c->dom_active);
        struct buf spinC = orki_bcreate(fd, (size_t)lN * 4, 0x403, c->dom_active);
        if (spinA.cpu && spinC.cpu && h->nascr + 2 <= 1024) {
            memset(spinA.cpu, 0, (size_t)lK); orki_bsync(fd, &spinA, RKNPU_MEM_SYNC_TO_DEVICE);
            h->ascr[h->nascr++] = spinA; h->ascr[h->nascr++] = spinC;
            uint32_t rcs[REGCMD_I8_N + 4]; memset(rcs, 0, sizeof rcs);
            orki_synth_i8(rcs, 1, lK, lN, (uint32_t)spinA.dma, lB, (uint32_t)spinC.dma, 1, CBUF, 0);
            for (int p = P; p < reserve; p++) {
                uint32_t *slot = (uint32_t*)((char*)c->regcmd.cpu + (size_t)p * REGCMD_I8_N * 4);
                memcpy(slot, rcs, REGCMD_I8_N * 4);
                if (p < reserve - 1) { uint64_t nx = c->regcmd.dma + (size_t)(p + 1) * REGCMD_I8_N * 4;
                    slot[216] = 0x0010 | ((nx & 0xffff) << 16); slot[217] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
                    slot[218] = 0x0014 | (0x0037u << 16);       slot[219] = (0x0101 << 16); }   /* else: reserve-1 terminal (rcs has a 0 descriptor) */
            }
            uint32_t *lr = (uint32_t*)((char*)c->regcmd.cpu + (size_t)(P - 1) * REGCMD_I8_N * 4);   /* real S-1 -> first spin slot */
            uint64_t nx = c->regcmd.dma + (size_t)P * REGCMD_I8_N * 4;
            lr[216] = 0x0010 | ((nx & 0xffff) << 16); lr[217] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
            lr[218] = 0x0014 | (0x0037u << 16);       lr[219] = (0x0101 << 16);
            h->spin_end = reserve;
            orki_bsync(fd, &c->regcmd, RKNPU_MEM_SYNC_TO_DEVICE);
        } else { if (spinA.cpu) orki_bdestroy(fd, &spinA); if (spinC.cpu) orki_bdestroy(fd, &spinC); }   /* alloc failed: no spin, stay terminating */
    }
    if (_dbg) fprintf(stderr, "[dyn] spin applied (spin_end=%d); seeding + real submit now\n", h->spin_end);
    ORK_DYN_SEED();
    sub.timeout = orki_mm_timeout_ms();
    int rr = orki_rknpu_submit_ioctl(fd, &sub, h->dom);
    if (_dbg) fprintf(stderr, "[dyn] NONBLOCK submit rc=%d (flags=0x%x task_number=%u)\n", rr, sub.flags, sub.task_number);
    if (rr) { free(h); return NULL; }
    if (_dbg) { usleep((unsigned)(P*80u+4000u)); for (int i = 0; i < S && i < 8; i++) { volatile int32_t *db=(volatile int32_t*)(h->outptr[i]+(h->N-1));
        __asm__ volatile("dc civac,%0"::"r"(db):"memory"); fprintf(stderr, "[dyn] post-NB   out[%d][N-1]=%d (want %d)\n", i, *db, tasks[i].w->K); } }
    return h;
}

ork_dyn_chain *ork_dyn_begin_mc(ork_npu *c, int S, const ork_mm_task_i8 *tasks, int nc) {
    if (!c || S < 1 || S > 1024 || !tasks) return NULL;
    ork_install_term();   /* graceful SIGTERM: make the async poll interruptible (covers colsplit + i4 dispatch too) */
    if (nc < 1 || nc > c->soc->cores) nc = c->soc->cores;
    /* P3 sub-nmax column-tiling: a single int8 matmul multi-cored by N-column split across cores. base:
     * Sn==1 & K<=4096 & Bf (any M). M=1 also does wide-N (Sn>1, K<=4096, Bf; core's range spans slices) and
     * wide-K (Sn==1, K>4096; per-core K-split accumulate over Bb K-slices — no Bf). M>1 wide-N/wide-K are dispatched by run_multicore's
     * colsplit (this direct ork_dyn_begin_mc entry serves the M=1 / SSM-stream callers). */
    { const ork_w *cw = tasks[0].w; int cM = tasks[0].M, ci8 = (cw->dtype == DT_I8 && (cw->N / 32) >= 2);
      int c_base = ci8 && cw->Sn == 1 && cw->K <= 4096 && cw->Bf;
      int c_wideN = ci8 && cw->Sn > 1 && cw->K <= 4096 && cw->Bf;   /* any M: colsplit base path M-tiles + N-slices */
      int c_wideK = ci8 && cw->Sn == 1 && (cw->K > 4096 || !cw->Bf);  /* any M: colsplit wide-K M-tiles the K-slice programs. K<=4096 no-Bf (ORK_NO_BF) also rides the Bf-free K-split here (matches run_multicore r_wideK) — else it falls through to the M>64 !Bf NULL refuse (was mis-read as #36 N=3584; really this gate gap) */
      (void)cM;
      if (S == 1 && nc > 1 && (c_base || c_wideN || c_wideK)) return ork_dyn_begin_colsplit(c, &tasks[0], nc); }
      /* fp16 colsplit is routed ONLY from run_multicore (which falls back to the single-core fp16 reference on NULL) — NOT here, so
       * direct ork_dyn_begin_mc callers (e.g. the SSM stream/pool fp16 path) keep their pre-Stage-1 behavior. */
    if (nc > S) nc = S;
    int dt = tasks[0].w->dtype;
    if (dt == DT_I4) return ork_dyn_begin_mc_i4(c, S, tasks, nc);   /* int4 (int16 out, M=1) has its own branch */
    if (dt != DT_I8 && dt != DT_F16) return NULL;   /* async doorbell: int8 (int32 out) or fp16 (fp32 out) — both 4-byte C */
    /* fp16 doorbell: bit-exact and enabled by default (was WIP-gated). The prior "residual ~10-20% all-16
     * cold-race" that kept it gated was NOT a chaining/coherency defect — it was the TEST feeding the A
     * activation from an ork_dma_alloc (zero-copy) buffer. This path STAGES A into maf via memcpy; a DMA-A
     * source's CPU writes are not coherently readable by that staged read (partial bytes -> partial-K sums
     * that look like a nondeterministic chaining bug). Proven by dump-and-diff vs the working
     * run_stream_f16_chain: byte-identical regcmd (only the output-C address word differs), identical
     * ACT_RESET, identical submit — the ONLY difference was the A source. With host (malloc) A — the same
     * convention the int8 doorbell already requires — the DEFAULT fp16 path (seed-all clean-before +
     * poll-to-done + civac) is bit-exact: 28/28 over 2 shapes (K512N256M2, K1024N512M4), multi-core.
     * A must be host memory (as for int8 A/D); a zero-copy DMA-A source is unsupported here. Legacy
     * opt-in env ORK_DYN_F16 is now a no-op. */
    for (int i = 0; i < S; i++) { ork_w *w = tasks[i].w;
        /* M>1 supported up to 64 rows/op: one regcmd/task holds the whole M-tile only within the 0x1040
         * mg_max*64 K-reduction cap (64 @ K<=4096, larger @ smaller K), so 64 is universally safe here;
         * a bigger M would need multi-regcmd tiling the chain can't express, so the caller uses sync. */
        if (!w || w->dtype != dt || tasks[i].M < 1) return NULL;
        /* M>64: plain wide-M PREFILL, M-tiled into mg_max*64-row programs. int8 + Sn==1 + K<=4096 with a
         * full-K Bf only (each M-tile is one full-K program). Combining M>64 with N-tiling / K-split / fp16
         * (a 2D/3D program grid) is a follow-up; those stay M<=64. */
        if (tasks[i].M > 64 && (dt != DT_I8 || w->Sn != 1 || w->K > 4096 || !w->Bf)) return NULL;
        /* G1 N-tiling: int8 accepts Sn>1 (each N-slice = one strided-output sub-op, synth_i8 stride arg).
         * fp16 stays Sn==1 — the fp16 `orki_synth()` has no output-stride arg, so a strided column-slice can't be
         * expressed there yet (fp16 N-tiling is a follow-up). */
        if (w->Sn != 1 && dt != DT_I8) return NULL;
        if (dt == DT_I8 ? (w->K % 512) : (w->K % 32)) return NULL;   /* int8: full-K Bf schedule needs K%512. fp16: single-slice small-K (the SSM scan, K%32) allowed — uses Bb + the K-dependent sched below. */
        /* G2 K-split: K>4096 (int8) rides Sk per-K-slice partials + host accumulate. Sn==1 (wide-K without
         * wide-N — the ffn_down shape). M 1..64: each K-slice's Kp<=KS(=1024) gives an mg_max*64 M-tile cap
         * >=320, so the whole M-tile fits ONE program per K-slice (no M-chunking); a defensive per-slice cap
         * check in the build rejects a pathological ORK_KTILE. fp16 K>4096 unsupported. */
        int ksplit = (dt == DT_I8 && w->K > 4096);
        if (w->K > 4096 && !ksplit) return NULL;
        if (ksplit && w->Sn != 1) return NULL;
        if (dt == DT_F16 && (size_t)tasks[i].M * w->K > 32768) return NULL;   /* fp16 M-tile validated <=32768; larger miscomputes (latent fp16 scheduler bug) */
        if (w->Sk != 1 && !w->Bf && !ksplit) return NULL;   /* non-ksplit Sk>1 needs the full-K Bf; ksplit uses the Bb K-slices */
        if (w->domain != tasks[0].w->domain) return NULL; }   /* all tasks one domain (single submit domain) */
    if (tasks[0].w->domain != c->dom_active || (tasks[0].w->domain && !c->dom_save)) orki_dom_activate(c, tasks[0].w->domain);
    if (dt == DT_F16) ork_npu_enter(c, DT_F16, XP_STREAM_F16, OCK_HW);   /* fp16 pipeline (layer owns reset, keep-warm-aware) */
    else              ork_npu_enter(c, 3 /*DT_I8_CHAIN*/, XP_CHAIN_NT, OCK_HW);
    if (orki_mc_ensure(c, nc)) return NULL;
    int fd = c->fd, CBUF = c->soc->cbuf_elems;
    if (dt == DT_F16 && CBUF > 32768) CBUF = 32768;   /* fp16 M-scheduler validated only to the 32768 tile (int8-only cbuf raise) */
    ork_dyn_chain *h = calloc(1, sizeof *h); if (!h) return NULL;
    h->c = c; h->S = S; h->P = S; h->N = tasks[0].w->N; h->dom = tasks[0].w->domain; h->reserve = S; h->mc = 1;
    unsigned dom = tasks[0].w->domain;
    int N0 = tasks[0].w->N;
    /* DIRECT vs COPY-BACK: write the caller's C in place (zero-copy) iff EVERY C is a resident DMA buffer
     * ALREADY in the submit's domain (`dom`). True for single-domain (domain 0), and for multi-domain when
     * the caller places C in the node's domain (Lever 1: per-domain output). Any mismatch/absence => in-domain
     * per-core scratch c->mcc[i] + copy-back in end. */
    int direct = 1;
    for (int i = 0; i < S; i++) { struct buf *cb = orki_dma_find(c, (void*)tasks[i].C); if (!cb || cb->domain != (int)dom) { direct = 0; break; } }
    /* M>1 NEVER uses direct (zero-copy) output: the direct path is coherency-unreliable for M>1 (validated —
     * M=8 direct output drops thousands of words to 0, non-deterministically, at Sn==1 AND Sn>1; the same
     * ZC-OUT class that keeps output zero-copy off by default). The scratch path (NPU -> cacheable mcc ->
     * bsync FROM_DEVICE -> CPU copy/scatter to the caller's C) is the reliable completion barrier and is
     * bit-exact. So force scratch for any M>1 op; end() straight-copies (Sn==1) or scatters (Sn>1 wide-N). */
    for (int i = 0; i < S; i++) if (tasks[i].M > 1 || tasks[i].w->K > 4096) { direct = 0; break; }   /* M>1 (ZC-OUT unsafe) and K-split (K>4096: partials+accumulate) both require the scratch path (K<=4096 uses full-K Bf, direct OK) */
    if (getenv("ORK_DYN_DEBUG")) fprintf(stderr, "[dyn_mc] S=%d nc=%d dt=%d dom=%u direct=%d N=%d\n", S, nc, dt, dom, direct, N0);
    if (!direct) for (int i = 0; i < nc; i++) { int lo=(int)((long)i*S/nc), hi=(int)((long)(i+1)*S/nc), P=hi-lo; if (P<1) continue;
        size_t osz = 0; for (int p = lo; p < hi; p++) osz += (size_t)(tasks[p].w->K > 4096 ? tasks[p].w->Sk : 1) * tasks[p].M * tasks[p].w->N * 4;   /* per-op M*N; K-split (K>4096) holds Sk partials */
        if (c->mccsz[i] < osz) { orki_bdestroy(fd, &c->mcc[i]); c->mcc[i] = orki_bcreate(fd, osz, 0x403, c->dom_active);
            if (!c->mcc[i].cpu) { free(h); return NULL; } c->mccsz[i] = osz; c->mwarm[i] = 0; } }   /* fresh scratch => "cold" so the clean-before-round fires (dirty-line coherency) */
    uint32_t rc[REGCMD_I8_N + 4];
    struct rknpu_submit subs[ORK_MAXCORE]; int Pc[ORK_MAXCORE]; memset(Pc, 0, sizeof Pc);
    int NMAX = c->soc->nmax;
    for (int i = 0; i < nc; i++) {
        int lo = (int)((long)i * S / nc), hi = (int)((long)(i+1) * S / nc), nop = hi - lo;
        if (nop < 1) { Pc[i] = 0; continue; }
        /* PROGRAM count (task_number) is decoupled from OP count: an op with Sn>1 N-slices emits Sn
         * chained programs (each a strided-output sub-op), so a core's program count is sum-of-Sn, not nop. */
        int Pcore = 0; for (int p = lo; p < hi; p++) { ork_w *ww = tasks[p].w; int MM = tasks[p].M;   /* programs/op: K-split=>Sk, wide-M=>ceil(M/cap), else Sn (N-tile/plain) */
            if (ww->K > 4096) Pcore += ww->Sk; else if (MM > 64) Pcore += (MM + orki_mtile_cap(ww->K) - 1) / orki_mtile_cap(ww->K); else Pcore += ww->Sn; }
        Pc[i] = Pcore;
        if ((size_t)Pcore * REGCMD_I8_N * 4 > c->mrc[i].size || (size_t)Pcore * sizeof(struct rknpu_task) > c->mtk[i].size) { free(h); return NULL; }
        /* A-staging need: K-split GATHERS every K-slice's [M,Kp] tile (sum = M*K) and all Sk tiles must be
         * resident at once (one chained submit); non-K-split stages [M,K]*esz. Grow maf (default 256KB) if a
         * wide-K M>1 op needs more (e.g. ffn_down M=64 => ~1.2MB). */
        size_t afneed = 0; for (int p = lo; p < hi; p++) { ork_w *ww = tasks[p].w; afneed += (size_t)tasks[p].M * ww->K * ((ww->K > 4096) ? 1 : ((dt == DT_F16) ? 2 : 1)); }
        if (c->maf[i].size < afneed) { orki_bdestroy(fd, &c->maf[i]); c->maf[i] = orki_bcreate(fd, afneed, 0x403, c->dom_active); if (!c->maf[i].cpu) { free(h); return NULL; } }
        struct buf *RC = &c->mrc[i], *AF = &c->maf[i], *CC = &c->mcc[i]; struct rknpu_task *tk = (struct rknpu_task*)c->mtk[i].cpu;
        size_t astage = 0, coff = 0; int pp = 0;   /* pp = program (task) index within this core's chain */
        for (int p = 0; p < nop; p++) {
            const ork_mm_task_i8 *t = &tasks[lo+p]; ork_w *w = t->w; int K = w->K, N = w->N, M = t->M, Sn = w->Sn, Sk = w->Sk;
            size_t esz = (dt == DT_F16) ? 2 : 1;         /* fp16 activation is 2 bytes/elem; int8 is 1 */
            size_t asz = (size_t)M * K * esz;
            if (K > 4096) {   /* ------- G2 K-SPLIT (int8, Sn==1, M 1..64): Sk partial programs + host accumulate --
                * Each K-slice ks computes a [M,N] partial from A[:, k0:k0+Kp] x Bb[ks]; end() SUMS the Sk
                * partials into C (the NPU has no on-device C+= mode). A_ks is GATHERED into a contiguous
                * [M,Kp] tile (synth_i8 reads A row-stride Kp; the caller's A has row-stride K) — for M==1 the
                * gather is a plain Kp-byte copy. Output is always scratch (partials + accumulate). */
                int KS = orki_int8_ks(c); int gi = lo + p;
                uint32_t cbase = (uint32_t)(CC->dma + coff);
                for (int ks = 0; ks < Sk; ks++) {
                    int k0 = ks * KS, Kp = (K - k0 < KS) ? (K - k0) : KS; int sched = (Kp == 1024 || Kp == 512);
                    /* defensive: the whole M-tile must fit one program for this Kp (mg_max*64 K-reduction cap).
                     * default KS=1024 gives cap>=320 so M<=64 always fits; a pathological ORK_KTILE could not. */
                    if (M > orki_mtile_cap(Kp)) { free(h); return NULL; }   /* would need M-tile chunking within a K-slice — not implemented */
                    if (astage + (size_t)M * Kp > AF->size) { free(h); return NULL; }
                    for (int r = 0; r < M; r++) memcpy((char*)AF->cpu + astage + (size_t)r*Kp, (const char*)t->A + (size_t)r*K + k0, Kp);   /* gather [M,Kp] */
                    uint32_t aks = (uint32_t)(AF->dma + astage); astage += (size_t)M * Kp;
                    memset(rc, 0, sizeof rc);
                    orki_synth_i8(rc, M, Kp, N, aks, (uint32_t)w->Bb[ks].dma, cbase + (uint32_t)((size_t)ks * M * N * 4), sched, CBUF, 0);   /* [M,N] partial ks */
                    if (orki_validate_regcmd("ork_dyn_mc_ks", c, rc, REGCMD_I8_N, w, NULL, 0)) { free(h); return NULL; }
                    if (pp < Pcore - 1) { uint64_t nx = RC->dma + (size_t)(pp+1) * REGCMD_I8_N * 4;
                        rc[216] = 0x0010 | ((nx & 0xffff) << 16); rc[217] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
                        rc[218] = 0x0014 | (0x0037u << 16);       rc[219] = (0x0101 << 16); }
                    memcpy((char*)RC->cpu + (size_t)pp * REGCMD_I8_N * 4, rc, REGCMD_I8_N * 4);
                    struct rknpu_task tt; memset(&tt, 0, sizeof tt); tt.enable_mask = 0xd; tt.int_mask = 0x300;
                    tt.int_clear = 0x1ffff; tt.regcfg_amount = 108; tt.regcmd_addr = RC->dma + (size_t)pp * REGCMD_I8_N * 4;
                    tk[pp] = tt; pp++;
                }
                h->outbuf[gi] = CC; h->outptr[gi] = (int32_t*)((char*)CC->cpu + coff); h->dst[gi] = (int32_t*)t->C;
                h->nout[gi] = Sk * M * N; h->oM[gi] = M; h->oSk[gi] = Sk;   /* Sk partials of [M,N]; end() sums to [M,N] */
                coff += (size_t)Sk * M * N * 4;
                continue;
            }
            if (astage + asz > AF->size) { free(h); return NULL; }
            memcpy((char*)AF->cpu + astage, t->A, asz); uint32_t adma = (uint32_t)(AF->dma + astage); astage += asz;   /* A shared across the op's N-slices */
            if (M > 64) {   /* ------- wide-M PREFILL (Sn==1, K<=4096 full-K Bf): M-tile into mtile_cap-row programs -------
                * Each M-tile computes rows [m0,m0+mc) of [M,N] (A rows are contiguous: adma+m0*K); the tiles
                * write disjoint row ranges of the [M,N] scratch, chained; end() straight-copies [M,N] to C. */
                int mcap = orki_mtile_cap(K); int gi = lo + p;
                uint32_t bdma = w->Bf ? (uint32_t)w->Bf[0].dma : (uint32_t)w->Bb[0].dma;
                uint32_t cbase = (uint32_t)(CC->dma + coff);   /* scratch (direct forced off for M>1) */
                for (int m0 = 0; m0 < M; m0 += mcap) { int mc = (M - m0 < mcap) ? (M - m0) : mcap;
                    memset(rc, 0, sizeof rc);
                    orki_synth_i8(rc, mc, K, N, adma + (uint32_t)((size_t)m0 * K), bdma, cbase + (uint32_t)((size_t)m0 * N * 4), 1, CBUF, 0);
                    if (orki_validate_regcmd("ork_dyn_mc_mt", c, rc, REGCMD_I8_N, w, NULL, 0)) { free(h); return NULL; }
                    if (pp < Pcore - 1) { uint64_t nx = RC->dma + (size_t)(pp+1) * REGCMD_I8_N * 4;
                        rc[216] = 0x0010 | ((nx & 0xffff) << 16); rc[217] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
                        rc[218] = 0x0014 | (0x0037u << 16);       rc[219] = (0x0101 << 16); }
                    memcpy((char*)RC->cpu + (size_t)pp * REGCMD_I8_N * 4, rc, REGCMD_I8_N * 4);
                    struct rknpu_task tt; memset(&tt, 0, sizeof tt); tt.enable_mask = 0xd; tt.int_mask = 0x300;
                    tt.int_clear = 0x1ffff; tt.regcfg_amount = 108; tt.regcmd_addr = RC->dma + (size_t)pp * REGCMD_I8_N * 4;
                    tk[pp] = tt; pp++;
                }
                h->outbuf[gi] = CC; h->outptr[gi] = (int32_t*)((char*)CC->cpu + coff); h->dst[gi] = (int32_t*)t->C;
                h->nout[gi] = M * N; h->oM[gi] = M; h->oSk[gi] = 0;
                coff += (size_t)M * N * 4;
                continue;
            }
            struct buf *cb = direct ? orki_dma_find(c, (void*)t->C) : NULL;
            /* Per-op output base: C in place (direct) or in-domain scratch. The Sn N-slices write DISJOINT
             * column ranges [n0,n0+Nc) of this [M,N]-laid-out output, each at row-stride N (synth_i8 stride arg)
             * -> no host scatter (the strided write lands columns directly; scratch stays plain [M,N] for copy-back). */
            uint32_t cbase = direct ? (uint32_t)(cb->dma + ((const char*)t->C - (const char*)cb->cpu))
                                    : (uint32_t)(CC->dma + coff);
            /* SCATTER (M>1 wide-N): each slice writes a CONTIGUOUS [M,Nc] scratch block (stride=0) at block
             * offset M*n0; end() scatters to C columns. NON-scatter Sn>1 (M=1) writes the column-slice in
             * place with a strided (row-stride N) output; Sn==1 is the plain single program. */
            int scat = (Sn > 1 && M > 1);
            for (int ns = 0; ns < Sn; ns++) {
                int n0 = ns * NMAX, Nc = (N - n0 < NMAX) ? (N - n0) : NMAX;
                uint32_t bdma = w->Bf ? (uint32_t)w->Bf[ns].dma : (uint32_t)w->Bb[(size_t)ns * Sk].dma;   /* slice weight: Bf[ns] full-K, or Bb[ns] when Sk==1 */
                uint32_t cdma = scat ? cbase + (uint32_t)((size_t)M * n0 * 4)                              /* contiguous [M,Nc] block */
                                     : cbase + (uint32_t)((size_t)n0 * 4);                                 /* column offset into [M,N] output */
                memset(rc, 0, sizeof rc);
                if (dt == DT_F16) { int schedf = ((K & (K-1)) == 0 && K >= 128);                            /* fp16 0x1040 sched: on for pow2 K>=128 (the doorbell's original always-on covered K512/1024/2048), off only for small/non-pow2 K (the SSM scan). NO <2048 upper bound — K=2048 (test_bmm) MUST stay sched=1 or the job hangs. */
                                    orki_synth   (rc, M, K, Nc, adma, bdma, cdma, schedf, CBUF); }               /* fp16: Sn==1 only (no stride arg) — guarded above */
                else              orki_synth_i8(rc, M, K, Nc, adma, bdma, cdma, 1, CBUF, scat ? 0 : ((Sn > 1) ? N : 0));  /* scatter=contiguous; else strided column-slice / single */
                if (orki_validate_regcmd("ork_dyn_mc", c, rc, REGCMD_I8_N, w, NULL, 0)) { free(h); return NULL; }
                if (pp < Pcore - 1) { uint64_t nx = RC->dma + (size_t)(pp+1) * REGCMD_I8_N * 4;
                    rc[216] = 0x0010 | ((nx & 0xffff) << 16); rc[217] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
                    rc[218] = 0x0014 | (0x0037u << 16);       rc[219] = (0x0101 << 16); }
                memcpy((char*)RC->cpu + (size_t)pp * REGCMD_I8_N * 4, rc, REGCMD_I8_N * 4);
                struct rknpu_task tt; memset(&tt, 0, sizeof tt); tt.enable_mask = 0xd; tt.int_mask = 0x300;
                tt.int_clear = 0x1ffff; tt.regcfg_amount = 108; tt.regcmd_addr = RC->dma + (size_t)pp * REGCMD_I8_N * 4;
                tk[pp] = tt; pp++;
            }
            /* Doorbell tracking is per-OP (one entry): the LAST N-slice covers column N-1, so it writes each
             * row's completion sentinel (C[m][N-1]) last -> the per-op last-col poll is unchanged by N-tiling. */
            int gi = lo + p;
            if (direct) { h->outbuf[gi] = cb; h->outptr[gi] = (int32_t*)t->C; h->dst[gi] = NULL; }   /* poll C in place, no copy */
            else        { h->outbuf[gi] = CC; h->outptr[gi] = (int32_t*)((char*)CC->cpu + coff); h->dst[gi] = (int32_t*)t->C; }
            h->nout[gi] = M * N; h->oM[gi] = M;   /* doorbell = last of M*N; end() copies M*N int32 back */
            coff += (size_t)M * N * 4;
        }
        memset(&subs[i], 0, sizeof subs[i]);
        subs[i].flags = ork_ppflags() | 0x2u; subs[i].task_number = Pcore; subs[i].task_obj_addr = c->mtk[i].obj;
        subs[i].core_mask = 1u << i; subs[i].fence_fd = -1;
        subs[i].subcore_task[0] = subs[i].subcore_task[1] = subs[i].subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)Pcore};
    }
    /* Seed the doorbell(s). int8: every row's last col (the M-tile writes a row's last col last, so that
     * word is the per-row completion sentinel). fp16: seed EVERY output element — fp16 direct-output needs
     * a full clean-before-write (dc cvac cleans each output cache line to DRAM) or dirty/stale CPU lines in
     * the output interior RACE the NPU's real-round writes and corrupt it (the same DMA cache-coherency
     * class as the ORK_ZC_OUT fix). int8's warm/last-col seed is sufficient for int8; only fp16 needs the
     * full-surface clean. Seeding all also lets end() poll any element, but done_i still uses last-cols.
     * N-tiled int8 (Sn>1) ALSO needs the full-surface clean: the strided column-slice writes leave the row
     * interior uncovered by the last-col seed, so dirty/stale CPU lines there race the NPU writes (the same
     * coherency class) -> non-deterministic zeros. seed_all folds Sn>1 into the fp16 full-clean branch. */
    int seed_all = (dt == DT_F16); for (int _si = 0; _si < S; _si++) if (tasks[_si].w->Sn > 1 || tasks[_si].w->K > 4096) { seed_all = 1; break; }   /* full-surface clean for N-tile (Sn>1) and K-split (K>4096) scratch */
    #define ORK_MC_SEED() do { for (int x = 0; x < S; x++) { int Mx=h->oM[x]?h->oM[x]:1, Nx=h->nout[x]?h->nout[x]/Mx:h->N; \
        if (seed_all) { for (int m=0;m<Mx;m++) for (int n=0;n<Nx;n++){ volatile int32_t *db = (volatile int32_t*)(h->outptr[x] + (size_t)m*Nx + n); \
            *db = ORK_DYN_SENT; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); } } \
        else { for (int m=0;m<Mx;m++){ volatile int32_t *db = (volatile int32_t*)(h->outptr[x] + (size_t)m*Nx + (Nx-1)); \
        *db = ORK_DYN_SENT; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); } } } __asm__ volatile("dsb ish":::"memory"); } while (0)
    #define ORK_MC_ROUND() do { h->mc_nc = nc; h->dma_rw0 = ork_npu_dma_rw(c); for (int i = 0; i < nc; i++) if (Pc[i]) { \
        orki_bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd, &c->mrc[i], RKNPU_MEM_SYNC_TO_DEVICE); \
        orki_bsync(fd, &c->mtk[i], RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE); \
        subs[i].timeout = orki_mm_timeout_ms(); { int _rc = orki_rknpu_submit_ioctl(fd, &subs[i], dom); if (i < 8) h->mc_rc[i] = _rc; } } } while (0)
    /* Clean the whole per-op output surface to DRAM (dedup by buffer) so no dirty CPU line can evict over the
     * NPU's writes; marks the cores warm. Re-runnable (used by the cold clean-before AND the dispatch-recover). */
    #define ORK_MC_CLEAN() do { struct buf *_cl[1024]; int _ncl = 0; \
        for (int x = 0; x < S; x++) { struct buf *b = h->outbuf[x]; int seen = 0; \
            for (int j = 0; j < _ncl; j++) if (_cl[j] == b) seen = 1; \
            if (!seen && b) { orki_bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE); if (_ncl < 1024) _cl[_ncl++] = b; } } \
        for (int i = 0; i < nc; i++) c->mwarm[i] = 1; } while (0)
    int cold = 0; for (int i = 0; i < nc; i++) if (Pc[i] && !c->mwarm[i]) cold = 1;
    /* INTERLEAVE-SAFE (decode/stream, all tasks M<=64): this output surface (per-core scratch or resident C) is
     * reused across doorbell ops interleaved in the decode loop (e.g. run_stream groups between single run_i8),
     * which can leave a dirty CPU line that evicts over the NPU write and resurrects a mid/last-col SENT
     * (test_stream_interleave g/u). Force the full-surface clean-before EVERY round for that regime — same as the
     * cold path. Prefill (M>64) is not interleaved and keeps cold-only (the per-op full clean costs latency at
     * large M — mirrors colsplit's M<=64 hardening gate). */
    int allsmall = 1; for (int i = 0; i < S; i++) if (tasks[i].M > 64) { allsmall = 0; break; }
    /* COLD FRESH-BUFFER COHERENCY (the real cold bug — NOT a pipeline-warm issue; blocking vs NONBLOCK is
     * irrelevant). The first cold call writes into a FRESHLY bcreate'd output scratch (c->mcc[i], or a fresh
     * caller DMA buffer): its CPU cache holds dirty/uninitialized lines. The NPU writes the result to DRAM,
     * then those dirty lines evict and OVERWRITE ~half of it with zeros (the ORK_ZC_OUT class, fixed in
     * 3fad74a: clean-before + invalidate-after). Also forced EVERY round in the interleaved decode/stream
     * regime (allsmall = all M<=64): a reused output surface can hold a dirty CPU line that evicts over the
     * NPU write and resurrects a mid/last-col SENT (test_stream_interleave). Prefill (M>64) is cold-only. */
    if (cold || allsmall) ORK_MC_CLEAN();
    /* Stash the round context so ork_dyn_end can RESUBMIT it on a not-dispatched miss (the ~1/4000 concurrent
     * NONBLOCK dispatch race; see orki_mc_recover_resubmit + the ork_dyn_end recover loop). int8 only (fp16 drains
     * in-submit below). c->maf/mrc/mtk[i] hold this round's data and aren't reused until end(), so a resubmit
     * from the stashed subs[] is valid. */
    h->mc_dom = dom; h->mc_seed_all = seed_all; h->mc_dt = dt;
    for (int i = 0; i < nc && i < ORK_MAXCORE; i++) { h->mc_subs[i] = subs[i]; h->mc_Pc[i] = Pc[i]; }
    ORK_MC_SEED();
    ORK_MC_ROUND();   /* single NONBLOCK round, cold or warm (the doorbell win) */
    if (dt == DT_F16) {
        /* fp16 drains in-submit (int8 stays async — the doorbell win is int8's). Polling the real round to
         * completion HERE (vs deferring the first poll to ork_dyn_end) removes a per-run race where end()
         * read last-cols that briefly showed the throwaway warm round's stale values (fp16's warm job retires
         * slower than int8's), reporting "done" in ~14us with wrong output. Poll-to-done in place makes end()
         * see a settled surface. (This + the clean-before seed cut fp16 flakiness ~80%->~10%; a residual
         * systemic ~10-20% cold-round mis-establish race persists — see the gate comment — so fp16 stays
         * gated. int8 needs none of this.) */
        double tp = ork_now_us();
        for(;;){ int alld=1; for(int x=0;x<S;x++) if(!ork_dyn_done_i(h,x)){alld=0;break;} if(alld||ork_now_us()-tp>3e6) break; }
        /* Full-surface invalidate-read of every output element after the doorbell fires. done_i (last-cols)
         * signals the tile's row is written, but for fp16 the interior settles a touch later; this sweep both
         * lets it settle and freshly invalidates every output line so end()/caller reads DRAM, not a stale
         * line. Empirically this trims the per-task near-misses (14-15/16); the systemic all-16 race is
         * separate (see gate comment) and NOT cured here. Cheap (M*N civac per op). int8 does not need it. */
        for (int x = 0; x < S; x++) { int Mx=h->oM[x]?h->oM[x]:1, Nx=h->nout[x]?h->nout[x]/Mx:h->N;
            for (long e=0;e<(long)Mx*Nx;e++){ volatile int32_t*db=(volatile int32_t*)(h->outptr[x]+e); __asm__ volatile("dc civac,%0"::"r"(db):"memory"); (void)*db; } }
    }
    return h;
}

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
static double g_f16_slice_us;   /* running max of a SUCCESSFULLY-landed fp16 K-slice completion (us). Drives the AUTO sentinel detect timeout = 1.5x this (adaptive per shape, vs a flat 800ms). Benign cross-core race — a heuristic, not correctness. */
struct ork_csub { ork_npu *c; int i; struct rknpu_submit *subs; ork_w *w; ork_dyn_chain *h; int hardened; int active; int ksbar; };

static void *ork_csub_worker(void *vp){ struct ork_csub *a = vp; ork_npu *c = a->c; int i = a->i, fd = c->fd;
    int cold = !c->mwarm[i];   /* capture BEFORE the bsync section sets mwarm=1 — fp16 uses it for the cold-buffer warmup */
    if (a->active) {
        if (a->h->oSk[i] <= 1) orki_bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE);   /* wide-K (oSk>1) shares the gathered A in maf[0], already flushed by the build gather — skip the redundant per-core maf orki_bsync (unused for i>0, double for i=0) */
        orki_bsync(fd, &c->mrc[i], RKNPU_MEM_SYNC_TO_DEVICE);
        orki_bsync(fd, &c->mtk[i], RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
        if (a->hardened || !c->mwarm[i]) orki_bsync(fd, &c->mcc[i], RKNPU_MEM_SYNC_TO_DEVICE);
        c->mwarm[i] = 1;
    }
    /* NO barrier + BLOCKING submit — EXACTLY the mcworker: 3 pool threads, each does a blocking submit on its
     * core and the kernel-waits (no userspace poll). ORK_COLSPLIT_NB flips to nonblock+poll for comparison. */
    if (a->active) {
      if (a->h->mc_dt == DT_F16 && a->h->oSk[i] > 1 && !a->h->f16_contig) {   /* CONTIG builds ONE chained submit -> take the single-submit else-path (like int8), NOT the per-slice SW-chain */
        /* AUTO SW-CHAIN: fp16 K-split cannot HW-chain across distinct Bb[ks] weight buffers (the next task's base
         * latches while the prior 2x-long fp16 weight-DMA is still draining -> cross-boundary prefetch -> cdma-wild
         * -> NPU soft-reset/wedge). So submit each K-slice's [kstart,np_ks) range SEPARATELY and BLOCKING — the
         * blocking return fully drains that slice's weight-DMA before the next slice's base is programmed (the
         * inter-op barrier the HW chain lacks). Ping-pong stays ON within each slice's same-buffer M-tile chain
         * (safe). The build terminated the regcmd chain at each slice boundary (kb[]); the oSk f32 accumulate below
         * then sums the Sk partials. Tiling recomputed identically to the build (deterministic). */
        ork_w *w = a->w; int M = a->h->oM[i], Sk = a->h->oSk[i], K = w->K;
        int KS = c->soc->ks, CBUFf = (c->soc->cbuf_elems > 32768) ? 32768 : c->soc->cbuf_elems, RBf = CBUFf, kstart = 0;
        const char *sge = getenv("ORK_F16_STAGGER"); int stag_us = sge ? atoi(sge) : 0;   /* variant A: per-core-index fetch stagger (µs) */
        /* fp16 SELF-HEAL — OPT-IN (ORK_F16_RECOV), default OFF. NAIVE retry-same-slice is HARMFUL and DISPROVEN:
         * resubmitting the identical 3-core concurrent fp16 slice re-triggers the SAME concurrent-fetch CDMA wild,
         * and hammering submits at a mid-soft-reset NPU ESCALATES a recoverable soft-reset into a HARD WEDGE
         * (board 10.3.0.236 hard-wedged 2026-08-04; power-cycle recovery, SPI survived). The correct self-heal must
         * de-escalate on fault (serial/single-core recompute — no concurrent wild — after the reset settles), NOT
         * resubmit the same concurrent slice. Kept gated for that future design; DO NOT default-on the naive retry. */
        int f16_recov = getenv("ORK_F16_RECOV") != NULL;
        const char *dte = getenv("ORK_F16_DETECT_MS"); int detect_ms = dte ? atoi(dte) : 500;   /* GROUNDED fast detect: a legit fp16 slice is ms-scale (~15ms max), so the blocking miss-timeout is the ANALOG of int8's ~300ms sentinel miss — tighten from the arbitrary 8s to ~500ms (30x the max legit slice, 16x faster detect). */
        /* REAP PRECONDITION (Task #50): the nonblock submit timeout MUST be < the poll-detect window, else a dropped
         * (benign-miss) job is NOT yet past its timeout when we go to reap it — rknpu_job_timeout_clean only reaps a
         * job whose age >= its timeout — so it lingers as subcore_data->job and the nc=1 backstop queues behind it
         * 60-180s -> rknpu_job_abort. Detect is ~1.5x the measured slice (g_f16_slice_us); set the submit timeout to
         * ~1x the slice (still >> a legit slice, which lands via IRQ before any timeout_clean regardless). Bootstrap
         * 60ms (< the 800ms bootstrap detect) until the first slice sets the baseline; never exceed detect_ms. */
        int recov_tmo = g_f16_slice_us > 0 ? (int)(g_f16_slice_us/1000) + 2 : 60;
        if (recov_tmo > detect_ms) recov_tmo = detect_ms; if (recov_tmo < 3) recov_tmo = 3;
        int f16_sentinel = (getenv("ORK_F16_SENTINEL") != NULL) && !c->f16_force_blocking;   /* force_blocking overrides -> blocking (heal). SENTINEL RECOVERY: submit NONBLOCK + CPU poll-drain each slice
            * (last-word gate + full-slice civac verify), tight timeout. Fast wedge-detect (~poll timeout, not the 8s blocking
            * timeout) AND the CPU never blocks in-kernel (no D-state hard-wedge); on a stuck sentinel -> mc_error -> run-level
            * recovery. Keeps the barrier's per-slice drain semantics (poll drains before advancing). int8-decode-path style. */
        const char *spte = getenv("ORK_F16_SENTINEL_TMO_US"); double f16_poll_ovr = spte ? atof(spte) : 0.0;   /* explicit detect-timeout override (us); 0 => AUTO = 1.5x the measured slice-completion time (g_f16_slice_us) */
        for (int ks = 0; ks < Sk; ks++) {
            int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS;
            int sched = ((Kp&(Kp-1))==0 && Kp>=128 && Kp<2048), R = RBf/Kp; if (R<1) R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
            double scale=(double)Kp/256.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale), mg=base>=0x1b?(base-0x1b)/slope+1:0;
            int kcap=mg*64; if(!sched)kcap=(RBf/2)/Kp; if(kcap<4*R)kcap=sched?4*R:((RBf/2)/Kp); if(kcap<1)kcap=1;
            int np_ks = (M + kcap - 1) / kcap;
            struct rknpu_submit s; memset(&s, 0, sizeof s);
            s.flags = ork_ppflags() | (f16_sentinel ? 0x2u : 0u);   /* SENTINEL: NONBLOCK (0x2) — CPU polls, never blocks in-kernel. Else BLOCKING (ping-pong safe within one buffer). */
            s.timeout = recov_tmo;
            s.task_start = (uint32_t)kstart; s.task_number = (uint32_t)np_ks;
            s.task_obj_addr = c->mtk[i].obj; s.core_mask = 1u << i; s.fence_fd = -1;
            s.subcore_task[0] = s.subcore_task[1] = s.subcore_task[2] = (struct rknpu_subcore_task){(uint32_t)kstart, (uint32_t)np_ks};
            if (i == 0) orki_trace_submit(&s);   /* ORK_TRACE: dump the doorbell fp16 per-slice regcmd sequence (core 0) */
            if (stag_us && i > 0) { struct timespec ts = {0, (long)i * stag_us * 1000}; nanosleep(&ts, NULL); }   /* variant A: offset core i so the 3 cores don't hit the CDMA with fp16 weight fetches at the same instant */
            int src = orki_rknpu_submit_ioctl(fd, &s, w->domain);
            if (f16_sentinel) {   /* SENTINEL RECOVERY: NONBLOCK submit returned immediately — now CPU poll-DRAIN this slice's
                * partial (last-word gate, then full-slice civac verify), with a TIGHT timeout. Fast wedge-detect + CPU never
                * stuck in-kernel. The full surface was SENT-seeded (hardened_w=1 for fp16). Timeout/reject -> mc_error -> run-level recovery. */
                if (src) { c->mc_error = 1;
                    fprintf(stderr, "[F16-WEDGE-DETECT] core=%d Kslice=%d submit_rc=%d (REJECTED at submit) mtk.int_status=0x%x\n",
                            i, ks, src, ((struct rknpu_task*)c->mtk[i].cpu)[0].int_status); fflush(stderr);
                    ork_kmsg("WEDGE-DETECT core=%d Kslice=%d submit_rc=%d (REJECTED)", i, ks, src); }
                else { volatile int32_t *o = (volatile int32_t*)a->h->outptr[i];
                    int Sk_ = a->h->oSk[i] ? a->h->oSk[i] : 1, per = a->h->nout[i] / Sk_;   /* M*Ncore words per K-slice partial */
                    /* AUTO detect timeout = 150% of the measured slice-completion time (g_f16_slice_us, updated on every
                     * successful land) — adaptive per shape instead of a flat 800ms: once the warm/first slice sets the
                     * baseline (~ms), detect drops to ~20ms, cutting the detect-to-reset VULNERABILITY WINDOW ~40x (the
                     * period the wedged NPU sits un-halted, able to keep wild-writing). 20ms floor so a tiny first
                     * measurement + jitter never false-positive; 800ms bootstrap until the first success; env overrides. */
                    double f16_poll_tmo = f16_poll_ovr > 0 ? f16_poll_ovr : (g_f16_slice_us > 0 ? 1.5 * g_f16_slice_us : 800000.0);
                    if (f16_poll_ovr <= 0 && f16_poll_tmo < 20000.0) f16_poll_tmo = 20000.0;
                    size_t last = (size_t)ks * per + per - 1; double pt = ork_now_us(); int wedged = 0;
                    for (;;) { __asm__ volatile("dc civac,%0"::"r"(&o[last]):"memory");   /* O(1) LAST-WORD gate: cheap wedge-detect (a full-slice scan per poll iter was ~7x slower). Coherency for the accumulate is the FROM_DEVICE bsync below; a real wedge faults early so the last word stays SENT. */
                        if (o[last] != ORK_DYN_SENT) { double el = ork_now_us() - pt; if (el > g_f16_slice_us) g_f16_slice_us = el; break; }   /* landed: update the slice-time baseline that drives the 1.5x auto timeout */
                        double el = ork_now_us() - pt; if (el > f16_poll_tmo) { wedged = 1; break; }
                        if (el > 500.0) { struct timespec ts = {0, 50000}; nanosleep(&ts, NULL); } }
                    if (wedged) { c->mc_error = 1;   /* IMMEDIATE located dump — earliest possible, before the barrier/join, so it
                        * streams out BEFORE any hard-hang loses it. Names the faulting (core, K-slice) + poll elapsed + the
                        * last-word value that stayed SENT (the doorbell that never landed). */
                        __asm__ volatile("dc civac,%0"::"r"(&o[last]):"memory");
                        fprintf(stderr, "[F16-WEDGE-DETECT] core=%d Kslice=%d POLL-MISS after %.0fus (tmo=%.0f) last-word[%zu]=0x%08x (SENT=stuck) mtk.int_status=0x%x\n",
                                i, ks, ork_now_us()-pt, f16_poll_tmo, last, (unsigned)o[last], ((struct rknpu_task*)c->mtk[i].cpu)[0].int_status); fflush(stderr);
                        ork_kmsg("WEDGE-DETECT core=%d Kslice=%d POLL-MISS %.0fus last=0x%08x", i, ks, ork_now_us()-pt, (unsigned)o[last]); } }
            } else {   /* BLOCKING (default): the kernel drains this slice; ORK_F16_RECOV (opt-in, DISPROVEN) resubmits on fault */
                for (int rt = 0; f16_recov && src && rt < 4; rt++) {
                    struct timespec rs = {0, 3000000 + (long)i * 2000000}; nanosleep(&rs, NULL);
                    src = orki_rknpu_submit_ioctl(fd, &s, w->domain);
                }
                if (src) c->mc_error = 1;
            }
            kstart += np_ks;
            /* LOCKSTEP: wait for every core to finish this K-slice before any core programs the next slice's
             * base — keeps all cores fetching the SAME Bb[ks] concurrently (never two distinct fp16 weight
             * buffers on the CDMA at once = the concurrent-cross-buffer wild). The blocking submit above has
             * already drained this core's slice, so the barrier only gates the CPU-side loop advance. */
            if (a->ksbar) pthread_barrier_wait(&c->b_ioctl);
        }
        if (i == 0) { const char *fwe = getenv("ORK_F16_FORCE_WEDGE");   /* TEST-ONLY fault injection (no real NPU fault; submits
            * above all succeeded): mark a simulated wedge for the FIRST N colsplit attempts (countdown), then succeed. N small
            * (e.g. 2) exercises the reset+resubmit HEAL path (must heal by attempt N+1, bit-exact); N huge (e.g. 99) exhausts
            * recovery and exercises the nc=1 de-escalation BACKSTOP. Post-barrier, core 0 only. */
            if (fwe) { static int fw_remain = -1; if (fw_remain < 0) fw_remain = atoi(fwe); if (fw_remain > 0) { c->mc_error = 1; fw_remain--; } } }
      } else {
        int nb = getenv("ORK_COLSPLIT_NB") != NULL;
        if (!nb) a->subs[i].flags &= ~0x2u;   /* blocking (default here) */
        a->subs[i].timeout = (a->h->mc_dt == DT_F16 && orki_mm_timeout_ms() > 8000) ? 8000 : orki_mm_timeout_ms();   /* fp16 (incl. CONTIG): cap wedge-detect at 8s so a CDMA-wild fault falls to the run-level reset+resubmit fast, not a 60s stall */
        int _csrc = orki_rknpu_submit_ioctl(fd, &a->subs[i], a->w->domain);
        if (_csrc && a->h->mc_dt == DT_F16) c->mc_error = 1;   /* CONTIG fp16 wedge: signal the run-level recovery (reset + resubmit / nc=1 backstop) */
        if (i == 0 && a->h->f16_contig) { const char *fwe = getenv("ORK_F16_FORCE_WEDGE");   /* TEST-ONLY fault injection for the
            * CONTIG in-place heal loop (no real NPU fault; the submit above succeeded): flag a simulated drop for the first N
            * begin passes (countdown), then succeed. N small (e.g. 2) must heal by pass N+1 bit-exact; N huge exhausts the heal
            * and exercises the nc=1 backstop. Core 0 only, after this pass's submit. */
            if (fwe) { static int fw_remain = -1; if (fw_remain < 0) fw_remain = atoi(fwe); if (fw_remain > 0) { c->mc_error = 1; fw_remain--; } } }
        if (nb) { volatile int32_t *o = (volatile int32_t*)a->h->outptr[i]; int no = a->h->nout[i];
            double pt = ork_now_us();
            for (;;) { __asm__ volatile("dc civac,%0"::"r"(&o[no-1]):"memory");
                if (o[no-1] != ORK_DYN_SENT) {   /* O(1) gate: last word landed -> full-surface VERIFY once (the
                    * K-split partial surface's write-order isn't last-word-last, so confirm EVERY word before
                    * declaring done — else the host-accumulate reads a premature partial = wrong output). Cheap:
                    * only scans once the gate passes (near-done), and each thread scans ONLY its own core. */
                    int all = 1; for (int e = 0; e < no; e++) { __asm__ volatile("dc civac,%0"::"r"(&o[e]):"memory"); if (o[e] == ORK_DYN_SENT) { all = 0; break; } }
                    if (all) break;
                }
                double el = ork_now_us() - pt; if (el > 3e6) break;
                if (el > 1000.0) { struct timespec ts = {0, 50000}; nanosleep(&ts, NULL); } } }
      }
        orki_bsync(fd, &c->mcc[i], RKNPU_MEM_SYNC_FROM_DEVICE);
        /* WIDE-K PARALLEL ACCUMULATE: sum this core's Sk [M,Ncore] partials into its C columns HERE, in this
         * pool thread — matching mcworker's chain-ksplit (each core accumulates its own partials in parallel)
         * instead of the SERIAL sum in ork_dyn_end (measured ~31ms serial vs ~13ms NPU submit on ffn_down =
         * the whole doorbell-vs-mcworker wide-K gap). The blocking submit guarantees every partial has landed;
         * the 3 cores write DISJOINT C column ranges (dst=C+c0), so no cross-core race. NEON int32 (bit-exact:
         * integer add is associative). dst[i]=NULL => ork_dyn_end's copy-back skips this core. */
        if (a->h->oSk[i] > 1 && a->h->dst[i]) {   /* PER-CORE PARALLEL accumulate (int8 AND fp16) — each core sums its
            * own Sk [M,Ncore] partials in this pool thread (per-core parallel accumulate). The blocking per-slice
            * submits + the FROM_DEVICE bsync above guarantee this core's partials have landed in DRAM; the lockstep
            * barrier (csub_barrier) guarantees no core crossed into a later Bb[ks] while another was still fetching
            * this one (the concurrent-cross-buffer CDMA wild that produced the wrong-answers). So fp16 no longer needs
            * the single-threaded full-surface verify+accumulate in ork_dyn_end (5-9x slower — it serialized all cores'
            * partials through one thread); that path stays as a dormant fallback (unreached: this sets dst[i]=NULL). */
            int Me = a->h->oM[i] ? a->h->oM[i] : 1, Sk = a->h->oSk[i], no = a->h->nout[i], Nn = no/(Sk*Me);
            size_t ds = a->h->ostride[i] > 0 ? (size_t)a->h->ostride[i] : (size_t)Nn, kstride = (size_t)Me*Nn;
            /* ks-OUTER (cache-friendly, like mcworker's chain-ksplit accumulate): read each K-slice partial
             * CONTIGUOUSLY and accumulate into the small C column block (stays hot in cache across the Sk
             * passes). The prior (m,n)-outer/ks-inner order scattered every element's reads across Sk
             * partials 1.2MB apart -> cache/TLB thrash (measured ~8-10ms vs ~2ms). Bit-exact (int add assoc;
             * fp16 f32-add order matches mcworker's ks-ascending sum lane-for-lane). */
            if (a->h->mc_dt == DT_F16) {   /* fp16: f32 partials + f32 accumulate (Stage 1) */
                const float *src = (const float*)a->h->outptr[i]; float *d = (float*)a->h->dst[i];
                for (int m = 0; m < Me; m++) { const float *bs = src + (size_t)m*Nn; float *dr = d + (size_t)m*ds;
                    for (int n = 0; n < Nn; n++) dr[n] = bs[n]; }   /* ks=0 init */
                for (int ks = 1; ks < Sk; ks++) { const float *sk = src + (size_t)ks*kstride;
                    for (int m = 0; m < Me; m++) { const float *bs = sk + (size_t)m*Nn; float *dr = d + (size_t)m*ds; int n = 0;
                        for (; n+4 <= Nn; n += 4) vst1q_f32(dr+n, vaddq_f32(vld1q_f32(dr+n), vld1q_f32(bs+n)));
                        for (; n < Nn; n++) dr[n] += bs[n]; } }
            } else {
                const int32_t *src = (const int32_t*)a->h->outptr[i]; int32_t *d = a->h->dst[i];
                for (int m = 0; m < Me; m++) { const int32_t *bs = src + (size_t)m*Nn; int32_t *dr = d + (size_t)m*ds;
                    for (int n = 0; n < Nn; n++) dr[n] = bs[n]; }   /* ks=0 init */
                for (int ks = 1; ks < Sk; ks++) { const int32_t *sk = src + (size_t)ks*kstride;
                    for (int m = 0; m < Me; m++) { const int32_t *bs = sk + (size_t)m*Nn; int32_t *dr = d + (size_t)m*ds; int n = 0;
                        for (; n+4 <= Nn; n += 4) vst1q_s32(dr+n, vaddq_s32(vld1q_s32(dr+n), vld1q_s32(bs+n)));
                        for (; n < Nn; n++) dr[n] += bs[n]; } }
            }
            a->h->dst[i] = NULL;   /* accumulated per-core; ork_dyn_end copy-back skips this i */
        }
    }
    return NULL;
}

ork_dyn_chain *ork_dyn_begin_colsplit(ork_npu *c, const ork_mm_task_i8 *t, int ncreq) {
    ork_w *w = t->w; int K = w->K, N = w->N, M = t->M, fd = c->fd, CBUF = c->soc->cbuf_elems;
    int dt = w->dtype;   /* DT_I8 today; fp16/int4 branches keyed on this (Stage 0: dt==DT_I8 == byte-identical) */
    int nt_sz = (dt == DT_F16) ? 16 : 32, NN = N / nt_sz, mcap = orki_mtile_cap(K), NMAX_C = c->soc->nmax;   /* col-tile width: int8 32, fp16 16 (each Kp*32 BYTES: int8 32x1, fp16 16x2); mcap rows/program (int8); NMAX_C = N-slice width */
    int nc = ncreq; if (nc > NN) nc = NN; if (nc > c->soc->cores) nc = c->soc->cores; if (nc < 1) nc = 1;
    if (dt == DT_F16 && w->Sk > 1 && nc > 2 && getenv("ORK_F16_2CORE")) nc = 2;   /* variant B: fewer concurrent fp16 fetchers (reliability/speed trade) */
    if (w->domain != c->dom_active || (w->domain && !c->dom_save)) orki_dom_activate(c, w->domain);
    /* Enter the correct PRECISION mode. fp16 must enter the fp16 mode EXACTLY as mcworker does
     * (DT_F16, XP_MC_MM) — entering the int8-chain mode (DT_I8_CHAIN) for fp16 leaves the stateful
     * regcmd datapath mismatched, which intermittently DMA-wilds / soft-resets the NPU on the fp16
     * submits. int8 keeps the HW-chain mode. This is the doorbell-vs-mcworker difference. */
    if (dt == DT_F16) ork_npu_enter(c, DT_F16, XP_MC_MM, OCK_NONE);
    else              ork_npu_enter(c, 3 /*DT_I8_CHAIN*/, XP_CHAIN_NT, OCK_HW);
    if (orki_mc_ensure(c, nc)) return NULL;
    ork_dyn_chain *h = calloc(1, sizeof *h); if (!h) return NULL;
    h->c = c; h->S = nc; h->P = nc; h->N = N; h->dom = w->domain; h->reserve = nc; h->mc = 1;
    h->mc_dt = dt;   /* set EARLY: ork_csub_worker (runs before the tail below) reads it for the accumulate dtype */
    struct rknpu_submit subs[ORK_MAXCORE]; int Pc[ORK_MAXCORE]; memset(Pc, 0, sizeof Pc);
    uint32_t rc[REGCMD_I8_N + 4];
    for (int i = 0; i < nc; i++) {
        int t0 = (int)((long)i * NN / nc), t1 = (int)((long)(i+1) * NN / nc), Ncore = (t1 - t0) * nt_sz, c0 = t0 * nt_sz;
        /* BALANCED wide-N (Sn>1): each core owns the even ~N/nc contiguous column range [c0,c1) (t0=i*NN/nc,
         * bit-exact to mcworker) — this keeps the per-core weight-DMA volume balanced (the wall-clock lever;
         * a whole-slice-per-core split imbalanced it 8192/8192/2560 and cost 1.3x). The range can CROSS an
         * nmax slice boundary; the base emission below cuts it into within-slice segments, each written to a
         * SEGMENT-MAJOR contiguous scratch block (synth stride=0 => no notch), and end() BOUNDARY-SCATTERS the
         * blocks to C[c0..c1) at row-stride N (h->oscat). Balanced AND notch-free — the two levers the
         * descriptor-array dump pinned as the mcworker-vs-doorbell gap (2026-08-03). */
        if (Ncore <= 0) { Pc[i] = 0; continue; }
        struct buf *RC = &c->mrc[i], *AF = &c->maf[i]; struct rknpu_task *tk = (struct rknpu_task*)c->mtk[i].cpu;
        size_t aesz = (dt == DT_F16) ? 2 : 1;   /* A element bytes: fp16 2, int8 1 */
        if ((size_t)M * K * aesz > AF->size) { orki_bdestroy(fd, &c->maf[i]); c->maf[i] = orki_bcreate(fd, (size_t)M*K*aesz, 0x403, c->dom_active); if (!c->maf[i].cpu) { free(h); return NULL; } AF = &c->maf[i]; }
        if (dt == DT_F16) {   /* fp16 colsplit (Stage 1): K-sliced Bb + host f32 accumulate; Sn==1 (gated). Mirrors the
            * int8 WIDE-K branch with orki_synth()/f32/fp16-chunk. base (Sk==1) => single partial (accumulate is a copy).
            * Weight offset t0*Kp*32 and the 108-reg task are IDENTICAL to int8/mcworker (only orki_synth()+Bb+dtype differ). */
            int CBUFf = (CBUF > 32768) ? 32768 : CBUF;   /* fp16 M-scheduler is validated only to the 32768-tile; a larger cbuf miscomputes mc>~cap (mcworker applies the same cap) */
            int KS = c->soc->ks, RBf = CBUFf;   /* fp16: RB = cbuf (int8 doubles it) */
            struct rknpu_task *tkf = (struct rknpu_task*)c->mtk[i].cpu;
            size_t ksz = (size_t)w->Sk * M * Ncore * 4;   /* Sk f32 partials [ks][M][Ncore] */
            if (c->mccsz[i] < ksz) { orki_bdestroy(fd, &c->mcc[i]); c->mcc[i] = orki_bcreate(fd, ksz, 0x403, c->dom_active);
                if (!c->mcc[i].cpu) { free(h); return NULL; } c->mccsz[i] = ksz; c->mwarm[i] = 0; }
            struct buf *CC = &c->mcc[i];
            /* pre-grow mrc/mtk for this core: fp16 chunks are small (<=8 @ K>=2048) => Sk*ceil(M/chunk) programs.
             * bound generously (512); tkf re-fetched after any grow. */
            size_t needrc = (size_t)512 * REGCMD_N * 4, needtk = (size_t)512 * sizeof(struct rknpu_task);
            if (RC->size < needrc) { orki_bdestroy(fd, &c->mrc[i]); c->mrc[i] = orki_bcreate(fd, needrc, 0x403, c->dom_active);
                if (!c->mrc[i].cpu) { free(h); return NULL; } RC = &c->mrc[i]; c->mwarm[i] = 0; }
            if (c->mtk[i].size < needtk) { orki_bdestroy(fd, &c->mtk[i]); c->mtk[i] = orki_bcreate(fd, needtk, 0x40b, c->dom_active);
                if (!c->mtk[i].cpu) { free(h); return NULL; } tkf = (struct rknpu_task*)c->mtk[i].cpu; }
            struct buf *AFS = &c->maf[0];   /* gather A ONCE (shared, read-only across cores): fp16 [Sk][M][Kp] */
            if (i == 0) { f16 *afg = (f16*)AFS->cpu; const f16 *Af = (const f16*)t->A; size_t goff = 0;
              for (int ks = 0; ks < w->Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS;
                  for (int m = 0; m < M; m++) memcpy(afg + goff + (size_t)m*Kp, Af + (size_t)m*K + k0, (size_t)Kp*2);   /* per-row memcpy (== int8 wide-K gather); Sk==1 => contiguous. Scalar j-loop was a big fixed cost on low-M shapes. */
                  goff += (size_t)M*Kp; }
              orki_bsync(fd, AFS, RKNPU_MEM_SYNC_TO_DEVICE); }
            uint32_t a_base = (uint32_t)AFS->dma;
            int contig = (w->Sn == 1) && !getenv("ORK_F16_NO_CONTIG");   /* (A) DEFAULT-ON for Sn==1: ONE chained submit/core over a CONTIGUOUS weight (Bbc) — no cross-buffer boundary => no HW cross-boundary prefetch => no CDMA wild => no drop (validated 500-iter 0-drop + shape-suite bit-exact, 2.6x). Sn>1 (multi-N-slice) keeps the per-slice path + the recovery net. ORK_F16_NO_CONTIG opts out for A/B. */
            if (contig && i == 0 && !w->Bbc_valid) {   /* build the contiguous weight ONCE (single-threaded build): concat all Sk K-slice tiles into one buffer */
                size_t tot = 0; for (int ks = 0; ks < w->Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS; tot += (size_t)Kp*N*2; }
                w->Bbc = orki_bcreate(fd, tot, 0x403, w->domain);
                if (w->Bbc.cpu) { size_t off = 0;
                    for (int ks = 0; ks < w->Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS; size_t sz = (size_t)Kp*N*2;
                        memcpy((char*)w->Bbc.cpu + off, w->Bb[ks].cpu, sz); off += sz; }
                    orki_bsync(fd, &w->Bbc, RKNPU_MEM_SYNC_TO_DEVICE); w->Bbc_valid = 1; }
            }
            if (contig && !w->Bbc_valid) contig = 0;   /* alloc failed -> fall back to the per-slice SW-chain */
            h->f16_contig = contig;
            int gapbase = (contig && getenv("ORK_F16_GAP")) ? atoi(getenv("ORK_F16_GAP")) : 0;   /* (B'') per-core DIFFERENTIAL spin-stagger base: core i gets gapbase*(nc-1-i) identity-perchan filler tasks BETWEEN slices, so core (nc-1) launches ks+1 first and core 0 last — OFFSETS the 3 cores' fp16 fetches in time to de-conflict the concurrent-fetch wild (not a synchronized drain — a staggered restart) */
            int gap = gapbase > 0;
            const int GAPM = 8, GAPN = 64;   /* tiny perchan geometry — pure time-filler; GAPM<=64 (perchan cap), GAPN%32==0 */
            uint32_t gap_pc[REGCMD_MUL_F16_CHAIN_N];
            if (gap) {
                if (i == 0 && !w->Bgap_valid) {   /* build the dummy gap buffers ONCE (shared, single-threaded here) */
                    w->Bgap[0]=orki_bcreate(fd,4096,0x403,w->domain); w->Bgap[1]=orki_bcreate(fd,4096,0x403,w->domain); w->Bgap[2]=orki_bcreate(fd,4096,0x403,w->domain);
                    if (w->Bgap[0].cpu && w->Bgap[1].cpu && w->Bgap[2].cpu) {
                        memset(w->Bgap[0].cpu,0,4096); memset(w->Bgap[1].cpu,0,4096);
                        { uint16_t *sb=(uint16_t*)w->Bgap[2].cpu; for (int e=0;e<GAPN;e++) sb[e]=0x3c00; }   /* identity per-channel scale (fp16 1.0) */
                        orki_bsync(fd,&w->Bgap[0],RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&w->Bgap[1],RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&w->Bgap[2],RKNPU_MEM_SYNC_TO_DEVICE);
                        w->Bgap_valid = 1;
                    }
                }
                if (!w->Bgap_valid) gap = 0;
                else { memcpy(gap_pc, REGCMD_MUL_F16_CHAIN, sizeof gap_pc); orki_set_mul_geom(gap_pc, REGCMD_MUL_F16_CHAIN_N, GAPM, GAPN);
                    orki_setrn(gap_pc, REGCMD_MUL_F16_CHAIN_N, RK_DPU_DST_BASE_ADDR, (uint32_t)w->Bgap[1].dma);
                    orki_setrn(gap_pc, REGCMD_MUL_F16_CHAIN_N, RK_SDP_5018, (uint32_t)w->Bgap[0].dma);
                    orki_setrn(gap_pc, REGCMD_MUL_F16_CHAIN_N, RK_SDP_5038, (uint32_t)w->Bgap[2].dma);
                    orki_setrn(gap_pc, REGCMD_MUL_F16_CHAIN_N, RK_SDP_5034, 0x00000008); }
            }
            int np2 = 0; size_t goff = 0, sloff = 0; char kb[512] = {0}; unsigned char pcp[600] = {0};   /* pcp[p]=1 => program p is a perchan drain-gap (not a matmul) */
            for (int ks = 0; ks < w->Sk; ks++) {
                int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS;
                int sched = ((Kp&(Kp-1))==0 && Kp>=128 && Kp<2048);   /* fp16 sched window (NO HISCHED here) */
                int R = RBf/Kp; if (R<1) R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
                double scale=(double)Kp/256.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale), mg_max = base>=0x1b ? (base-0x1b)/slope+1 : 0;
                int kcap = mg_max*64; if(!sched) kcap=(RBf/2)/Kp; if(kcap<4*R) kcap=sched?4*R:((RBf/2)/Kp); if(kcap<1) kcap=1;   /* fp16 M-tile cap — NEVER raise: >cap miscomputes (npu.c ~4770) */
                uint32_t wbase = (uint32_t)((contig ? (w->Bbc.dma + sloff) : w->Bb[ks].dma) + (uint64_t)t0 * Kp * 32);   /* CONTIG: slice base = Bbc + cumulative slice offset (one buffer); else per-slice Bb[ks]. N-tile stride Kp*32 (== mcworker) */
                for (int m0 = 0; m0 < M; m0 += kcap) { int mc = (M-m0<kcap)?(M-m0):kcap;
                    if ((size_t)(np2+1) * REGCMD_N * 4 > RC->size) { free(h); return NULL; }
                    memset(rc, 0, REGCMD_N * 4);
                    orki_synth(rc, mc, Kp, Ncore, (uint32_t)(a_base + (goff + (size_t)m0*Kp)*2), wbase,
                          (uint32_t)(CC->dma + ((size_t)ks*M + m0)*Ncore*4), sched, CBUFf);
                    if (orki_validate_regcmd("ork_dyn_colsplit_f16", c, rc, REGCMD_N, w, NULL, 0)) { free(h); return NULL; }
                    memcpy((char*)RC->cpu + (size_t)np2*REGCMD_N*4, rc, REGCMD_N*4);
                    np2++;
                }
                if (!contig && np2 > 0 && np2 <= 512) kb[np2-1] = 1;   /* end of this K-slice's sub-chain (per-slice submit); CONTIG leaves it linked */
                if (gap && ks < w->Sk-1) {   /* (B'') DIFFERENTIAL spin-stagger: core i gets gapbase*(nc-1-i) filler perchans between slices (padded into REGCMD_N slots; HW reads each one's 69 regs) — offsets this core's ks+1 launch vs the others */
                    int gapK = gapbase * (nc - 1 - i);
                    for (int g = 0; g < gapK && np2 < 500; g++) {
                        if ((size_t)(np2+1) * REGCMD_N * 4 > RC->size) break;
                        memcpy((char*)RC->cpu + (size_t)np2*REGCMD_N*4, gap_pc, REGCMD_MUL_F16_CHAIN_N*4);
                        pcp[np2] = 1; np2++;
                    }
                }
                goff += (size_t)M*Kp; sloff += (size_t)Kp*N*2;
            }
            for (int p = 0; p < np2; p++) { uint32_t *pr = (uint32_t*)((char*)RC->cpu + (size_t)p*REGCMD_N*4);
                if (p < np2-1 && (contig || !kb[p])) { uint64_t nx = RC->dma + (size_t)(p+1)*REGCMD_N*4;   /* CONTIG: link ALL (one chain across slices + gaps); SW-chain: link only WITHIN a K-slice */
                    int slot = pcp[p] ? 138 : 216;                    /* perchan links at its 138 slot (regcfg*2), matmul at 216 */
                    int nreg = pcp[p+1] ? ((69+3)/2) : ((108+3)/2);   /* next-amount = the NEXT program's register count */
                    pr[slot]   = 0x0010 | ((nx & 0xffff) << 16); pr[slot+1] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
                    pr[slot+2] = 0x0014 | ((uint32_t)nreg << 16); pr[slot+3] = (0x0101 << 16); }
                struct rknpu_task tt; memset(&tt, 0, sizeof tt); tt.int_mask = 0x300; tt.int_clear = 0x1ffff;
                tt.enable_mask = pcp[p] ? 0x18 : 0xd; tt.regcfg_amount = pcp[p] ? 69 : 108;   /* perchan: enable 0x18, 69 regs; matmul: 0xd, 108 */
                tt.regcmd_addr = RC->dma + (size_t)p*REGCMD_N*4; tkf[p] = tt; }
            h->outbuf[i] = CC; h->outptr[i] = (int32_t*)CC->cpu; h->nout[i] = w->Sk * M * Ncore; h->oM[i] = M; h->oSk[i] = w->Sk;
            h->dst[i] = (int32_t*)((char*)t->C + (size_t)c0 * 4); h->ostride[i] = t->cstride ? t->cstride : N;   /* f32 accumulate/copy-back -> C columns at row-stride N (cstride override: fp16 wide-N per-slice writes a sub-N result into the wider C at full stride) */
            Pc[i] = np2;
            memset(&subs[i], 0, sizeof subs[i]);
            subs[i].flags = (gap ? 0x1u : ork_ppflags()) | 0x2u; subs[i].task_number = np2; subs[i].task_obj_addr = c->mtk[i].obj;   /* gap chain carries an SDP (perchan) task -> ping-pong OFF (0x1); worker clears 0x2 -> blocking */
            subs[i].core_mask = 1u << i; subs[i].fence_fd = -1;
            subs[i].subcore_task[0] = subs[i].subcore_task[1] = subs[i].subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)np2};
            continue;
        }
        uint32_t adma = (uint32_t)AF->dma;
        if (K <= 4096) memcpy(AF->cpu, t->A, (size_t)M * K);   /* full A[M,K] for base/wide-N. WIDE-K (K>4096) re-gathers A into AF as [Sk][M][Kp] below, so this full copy would be pure waste there — skip it. */
        size_t osz = (size_t)M * Ncore * 4;
        if (c->mccsz[i] < osz) { orki_bdestroy(fd, &c->mcc[i]); c->mcc[i] = orki_bcreate(fd, osz, 0x403, c->dom_active);
            if (!c->mcc[i].cpu) { free(h); return NULL; } c->mccsz[i] = osz; c->mwarm[i] = 0; }
        struct buf *CC = &c->mcc[i];
        if (K > 4096 || !w->Bf) {   /* WIDE-K (K>4096) or NO-Bf (K<=4096, ORK_NO_BF) colsplit (Sn==1, ANY M): the Bf-FREE per-core K-split — Sk*(M-tile) partial
            * [mc,Ncore] programs over this core's column range; end() SUMS the Sk [M,Ncore] partials into
            * C[c0:c1) at row-stride N (ork_dyn_end oSk accumulate, [ks][m][n] layout). M>1: A's K-slice rows are
            * strided by the FULL K, but synth reads A contiguous [mc,Kp], so gather A[M,K] -> AF as a per-slice-
            * contiguous [Sk][M][Kp] layout first. Each K-slice's Kp<=KS gives an mg_max*64 cap >=320, so M<=256
            * prefill fits one program/slice; M-tiled defensively for larger M. */
            int KS = orki_int8_ks(c);
            size_t ksz = (size_t)w->Sk * M * Ncore * 4;
            if (c->mccsz[i] < ksz) { orki_bdestroy(fd, &c->mcc[i]); c->mcc[i] = orki_bcreate(fd, ksz, 0x403, c->dom_active);
                if (!c->mcc[i].cpu) { free(h); return NULL; } c->mccsz[i] = ksz; c->mwarm[i] = 0; CC = &c->mcc[i]; }
            /* A[M,K]->[Sk][M][Kp] is IDENTICAL for every core (A is not core-dependent — only the weight column
             * range differs per core). Gather it ONCE into maf[0] (shared, read-only) and point every core's
             * regcmd at it, instead of the redundant per-core gather that dominated the serial build (~1.4ms x
             * nc). maf[0] is sized M*K by the i==0 realloc above; the 3 cores only READ it (no write race). */
            struct buf *AFS = &c->maf[0];
            if (i == 0) { int8_t *afg = (int8_t*)AFS->cpu; size_t goff = 0;
              for (int ks = 0; ks < w->Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS;
                  for (int m = 0; m < M; m++) memcpy(afg + goff + (size_t)m*Kp, (const int8_t*)t->A + (size_t)m*K + k0, (size_t)Kp);
                  goff += (size_t)M*Kp; }
              orki_bsync(fd, AFS, RKNPU_MEM_SYNC_TO_DEVICE); }
            uint32_t a_base = (uint32_t)AFS->dma;   /* shared gathered A for all cores */
            int np2 = 0; size_t goff = 0;
            for (int ks = 0; ks < w->Sk; ks++) {
                int k0 = ks * KS, Kp = (K - k0 < KS) ? (K - k0) : KS; int sched = (Kp == 1024 || Kp == 512);
                int kcap = orki_mtile_cap(Kp); if (kcap < 1) kcap = 1;   /* rows/program for this K-slice */
                uint32_t wbase = (uint32_t)(w->Bb[ks].dma + (uint64_t)(c0 / nt_sz) * Kp * nt_sz);   /* K-slice ks weight, column sub-range */
                for (int m0 = 0; m0 < M; m0 += kcap) { int mc = (M - m0 < kcap) ? (M - m0) : kcap;
                    if ((size_t)(np2+1) * REGCMD_I8_N * 4 > RC->size) { free(h); return NULL; }
                    memset(rc, 0, sizeof rc);
                    orki_synth_i8(rc, mc, Kp, Ncore, (uint32_t)(a_base + goff + (size_t)m0*Kp), wbase,
                             (uint32_t)(CC->dma + ((size_t)ks * M + m0) * Ncore * 4), sched, CBUF, 0);   /* [mc,Ncore] rows [m0,+mc) of partial ks */
                    if (orki_validate_regcmd("ork_dyn_colsplit_ks", c, rc, REGCMD_I8_N, w, NULL, 0)) { free(h); return NULL; }
                    memcpy((char*)RC->cpu + (size_t)np2 * REGCMD_I8_N * 4, rc, REGCMD_I8_N * 4);
                    np2++;
                }
                goff += (size_t)M*Kp;
            }
            for (int p = 0; p < np2; p++) { uint32_t *pr = (uint32_t*)((char*)RC->cpu + (size_t)p * REGCMD_I8_N * 4);
                if (p < np2 - 1) { uint64_t nx = RC->dma + (size_t)(p+1) * REGCMD_I8_N * 4;
                    pr[216] = 0x0010 | ((nx & 0xffff) << 16); pr[217] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
                    pr[218] = 0x0014 | (0x0037u << 16);       pr[219] = (0x0101 << 16); }
                struct rknpu_task tt; memset(&tt, 0, sizeof tt); tt.enable_mask = 0xd; tt.int_mask = 0x300;
                tt.int_clear = 0x1ffff; tt.regcfg_amount = 108; tt.regcmd_addr = RC->dma + (size_t)p * REGCMD_I8_N * 4; tk[p] = tt; }
            h->outbuf[i] = CC; h->outptr[i] = (int32_t*)CC->cpu; h->nout[i] = w->Sk * M * Ncore; h->oM[i] = M; h->oSk[i] = w->Sk;
            h->dst[i] = (int32_t*)((char*)t->C + (size_t)c0 * 4); h->ostride[i] = t->cstride ? t->cstride : N;   /* accumulate -> C columns at row-stride N (cstride override: int8 no-Bf/wide-K wide-N per-N-slice writes a sub-N result into the wider C at full stride) */
            Pc[i] = np2;
            memset(&subs[i], 0, sizeof subs[i]);
            subs[i].flags = ork_ppflags() | 0x2u; subs[i].task_number = np2; subs[i].task_obj_addr = c->mtk[i].obj;
            subs[i].core_mask = 1u << i; subs[i].fence_fd = -1;
            subs[i].subcore_task[0] = subs[i].subcore_task[1] = subs[i].subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)np2};
            continue;
        }
        int np = 0, nseg = 0;   /* programs for this core: (within-slice SEGMENT x M-tile), segment-major scratch */
        /* This core owns the CONTIGUOUS column range [c0,c1) of C, which may CROSS nmax slice boundaries. Cut it
         * into within-slice segments; lay the scratch SEGMENT-MAJOR ([M,segw] blocks back-to-back) and write each
         * program contiguously (synth stride=0 => Ncol==row-stride, NO notch). Balanced (even [c0,c1)) + notch-free.
         * end() boundary-scatters each block to its C column sub-range at row-stride N (h->oscat, using h->ocol0). */
        { int c1e = c0 + Ncore; size_t segbase = 0; int cur = c0;
          while (cur < c1e) {
              int ns = cur / NMAX_C, sl1 = (ns + 1) * NMAX_C; if (sl1 > N) sl1 = N;
              int segend = (c1e < sl1) ? c1e : sl1, segw = segend - cur, is0 = cur - ns * NMAX_C;
              uint32_t wbase = (uint32_t)(w->Bf[ns].dma + (uint64_t)(is0 / nt_sz) * K * nt_sz);   /* slice ns, in-slice col offset */
              for (int m0 = 0; m0 < M; m0 += mcap) { int mc = (M - m0 < mcap) ? (M - m0) : mcap;
                  if ((size_t)(np+1) * REGCMD_I8_N * 4 > RC->size) { free(h); return NULL; }
                  memset(rc, 0, sizeof rc);
                  orki_synth_i8(rc, mc, K, segw, adma + (uint32_t)((size_t)m0 * K), wbase,
                           (uint32_t)(CC->dma + (segbase + (size_t)m0 * segw) * 4), 1, CBUF, 0);   /* [mc,segw] contiguous block */
                  if (orki_validate_regcmd("ork_dyn_colsplit", c, rc, REGCMD_I8_N, w, NULL, 0)) { free(h); return NULL; }
                  memcpy((char*)RC->cpu + (size_t)np * REGCMD_I8_N * 4, rc, REGCMD_I8_N * 4);
                  np++;
              }
              segbase += (size_t)M * segw; cur = segend; nseg++;
          }
        }
        for (int p = 0; p < np; p++) { uint32_t *pr = (uint32_t*)((char*)RC->cpu + (size_t)p * REGCMD_I8_N * 4);
            if (p < np - 1) { uint64_t nx = RC->dma + (size_t)(p+1) * REGCMD_I8_N * 4;
                pr[216] = 0x0010 | ((nx & 0xffff) << 16); pr[217] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
                pr[218] = 0x0014 | (0x0037u << 16);       pr[219] = (0x0101 << 16); }
            struct rknpu_task tt; memset(&tt, 0, sizeof tt); tt.enable_mask = 0xd; tt.int_mask = 0x300;
            tt.int_clear = 0x1ffff; tt.regcfg_amount = 108; tt.regcmd_addr = RC->dma + (size_t)p * REGCMD_I8_N * 4; tk[p] = tt; }
        h->outbuf[i] = CC; h->outptr[i] = (int32_t*)CC->cpu; h->nout[i] = M * Ncore; h->oM[i] = M; h->oSk[i] = 0;
        h->dst[i] = (int32_t*)((char*)t->C + (size_t)c0 * 4); h->ostride[i] = N; h->ocol0[i] = c0;
        h->oscat[i] = (nseg > 1);   /* boundary-scatter ONLY when the balanced range crosses a slice boundary (>=2 segments);
            * a single-segment core (all base Sn==1, incl. attention) is a plain contiguous [M,Ncore] block -> keep the
            * cheap per-row-last-col done + ostride copy-back (oscat=0). Setting oscat unconditionally forced the
            * pathological full-surface poll on the base path and tanked native attention 73->13. */
        Pc[i] = np;
        memset(&subs[i], 0, sizeof subs[i]);
        subs[i].flags = ork_ppflags() | 0x2u; subs[i].task_number = np; subs[i].task_obj_addr = c->mtk[i].obj;
        subs[i].core_mask = 1u << i; subs[i].fence_fd = -1;
        subs[i].subcore_task[0] = subs[i].subcore_task[1] = subs[i].subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)np};
    }
    /* HARDENED (decode/stream, M<=64): this per-core scratch is reused across doorbell ops interleaved in the
     * decode loop, which can leave it dirty so a stale CPU line evicts over the NPU write and resurrects a
     * mid-row SENT. Harden with a FULL-surface SENT seed + the ALWAYS-clean bsync below (flushes it to DRAM so
     * nothing evicts over the NPU write); done_i then trusts each row's last col (row-major, last-col-last —
     * same as begin_mc's plain-int8 path). Cheap at small M. PREFILL (M>64, large output) is not interleaved,
     * so keep the original last-col seed + cold-only clean — the per-op full-buffer flush tripped test_speed's
     * latency floor at M=512. Bit-exact either way; this only changes the completion barrier, not the math. */
    /* FULL-surface SENT seed + clean-before when the done-check polls the full surface (not per-row last-col):
     * (a) M<=64 interleaved decode/stream; (b) WIDE-K (K>4096) — oSk partial [Sk][M][Ncore] surface, done only
     * when every partial word lands; (c) WIDE-N (Sn>1, M>1) — Sn scatter blocks whose write-order isn't
     * last-col-last. Without this the M>64 last-col seed mismatches the full-surface poll -> completion misses ->
     * 3s cap -> recover-resubmit stall (measured ~10 t/s on ffn_down). The per-op full flush is small vs the
     * wide K-split/scatter compute. Plain base (Sn==1,K<=4096,M>64) keeps the cheap last-col seed. */
    int hardened = (M <= 64) || (w->K > 4096) || (w->Sn > 1) || (dt == DT_F16 && w->Sk > 1);   /* fp16 K-split partials: write-order not last-col-last => full-surface seed (NONBLOCK path only) */
    /* STAGE 2: the PARALLEL path submits BLOCKING (ork_csub_worker clears the nonblock bit) + sets prepolled, so
     * completion = the blocking ioctl return + bsync FROM_DEVICE — the full-surface SENT seed is NEVER polled
     * there (pure waste: O(M*Ncore) CPU writes/core, the doorbell-specific wide-N host tax native doesn't pay).
     * Skip it and use cold-only clean-before (hardened_w=0). The NONBLOCK path (no PARALLEL, or COLSPLIT_NB)
     * still seeds + polls sentinels, so it keeps the full-surface seed + hardened clean-before. */
    int parallel_blocking = (nc > 1 && !getenv("ORK_COLSPLIT_SERIAL") && !getenv("ORK_COLSPLIT_NB"));
    /* fp16 K-split: the blocking submit's completion can precede the f32 writeback drain, so ork_csub_worker
     * runs a full-surface civac VERIFY even on the parallel-blocking path. That verify needs the SENT seed
     * flushed to DRAM first -> seed the surface AND force the clean-before (hardened_w=1) for fp16 here. */
    int fp16_hard = (dt == DT_F16 && w->Sk > 1);
    int hardened_w = parallel_blocking ? (fp16_hard ? 1 : 0) : hardened;
    /* fp16 K-split lockstep barrier: the residual wedge (during-submit "cdma address wild" + occasional
     * plausible-wrong partial) is a CONCURRENT CROSS-BUFFER FETCH — cores loop their Sk slices independently,
     * so core A can be fetching Bb[ks+1] while core B is still on Bb[ks] = two distinct fp16 weight buffers on
     * the CDMA at once (fp16's 2-byte weights double the fetch bytes; int8 tolerates it, fp16 wilds). A
     * per-slice pthread barrier marches all cores in lockstep: every core finishes slice ks before ANY starts
     * ks+1, so at any instant all cores fetch the SAME Bb[ks] (same-buffer concurrency is benign). Pure CPU
     * sync, no NPU risk. Requires every dispatched core active (else the barrier count nc deadlocks) — for the
     * fp16 colsplit nc<=NN so all cores get columns, but guard anyway and fall back to independent loops. */
    int csub_barrier = 0;
    if (fp16_hard && nc > 1 && parallel_blocking && !getenv("ORK_F16_NOBAR")) { csub_barrier = 1; for (int i = 0; i < nc; i++) if (!Pc[i]) csub_barrier = 0; }
    /* fp16 DROP RECOVERY (Task #50). campaign4 (clean board, sentinel, live-traced) PROVED int8's in-place resubmit
     * model does NOT transfer to fp16: the concurrent-fetch drop is STICKY — reset+resubmit of the identical 3-core
     * fp16 slice re-drops EVERY attempt (observed Kslice=1, cores alternating, tries 1/2/3 all POLL-MISS), and by the
     * 3rd reset a re-launched worker HUNG in an uninterruptible D-state (the kernel `continue wait`, see ork_dummy_probe's
     * header), forcing a power-cycle. So we do NOT resubmit the fp16 slice. On a worker-detected drop: (1) CLEAN-REAP the
     * stuck slice job via ork_npu_reap_stuck (per-core nonblock dummy -> rknpu_job_timeout_clean; avoids the D-state +
     * the close(fd) refcount-UAF), then (2) HEALTH-GATE with ork_dummy_probe — a tiny INT8 nonblock host-bounded op on
     * the same fd (can NEVER hang: int8's drop is a transient dispatch race, not fp16's sticky wild). c->mc_error stays
     * set so the run-level de-escalates to the nc=1 bit-exact backstop (single-core fp16 = no concurrent fetch = no drop).
     * Fast + SAFE: bounded nonblock recovery + a guaranteed-correct single-core recompute, never the resubmit thrash. */
    if (csub_barrier) pthread_barrier_init(&c->b_ioctl, NULL, nc);
    if (!parallel_blocking || fp16_hard)
    for (int i = 0; i < nc; i++) if (Pc[i]) {
        if (hardened) { int no = h->nout[i]; volatile int32_t *o = h->outptr[i]; for (int e = 0; e < no; e++) o[e] = ORK_DYN_SENT; }
        else { int Mx = h->oM[i], Nx = h->nout[i]/Mx; for (int m = 0; m < Mx; m++) {
            volatile int32_t *db = h->outptr[i] + (size_t)m*Nx + (Nx-1); *db = ORK_DYN_SENT; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); } } }
    __asm__ volatile("dsb ish":::"memory");
    if (nc > 1 && !getenv("ORK_COLSPLIT_SERIAL")) {   /* DEFAULT: per-core submit+accumulate on pool threads (ork_csub_worker); ORK_COLSPLIT_SERIAL forces the legacy inline path below */
        struct ork_csub cs[ORK_MAXCORE];
        for (int i = 0; i < nc; i++) cs[i] = (struct ork_csub){ c, i, subs, w, h, hardened_w, Pc[i] != 0, csub_barrier };
        orki_npu_pool_ensure(c);
        pthread_mutex_lock(&c->pmu); c->pjob = cs; c->pjob_nc = nc; c->pjob_fn = ork_csub_worker;
        c->pjob_stride = sizeof(struct ork_csub); c->pdone = 0; c->pgen++; pthread_cond_broadcast(&c->pgo);
        pthread_mutex_unlock(&c->pmu);
        ork_csub_worker(&cs[0]);   /* core 0 on this thread; cores 1..nc-1 on pool threads */
        pthread_mutex_lock(&c->pmu); while (c->pdone < nc - 1) pthread_cond_wait(&c->pdn, &c->pmu); pthread_mutex_unlock(&c->pmu);
        if (csub_barrier) pthread_barrier_destroy(&c->b_ioctl);
        h->prepolled = 1;   /* workers already submitted + drained every core; ork_dyn_end skips its poll */
    } else
    for (int i = 0; i < nc; i++) if (Pc[i]) {
        orki_bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd, &c->mrc[i], RKNPU_MEM_SYNC_TO_DEVICE);
        orki_bsync(fd, &c->mtk[i], RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
        if (hardened || !c->mwarm[i]) orki_bsync(fd, &c->mcc[i], RKNPU_MEM_SYNC_TO_DEVICE);   /* clean-before: ALWAYS for the
            * interleaved decode/stream regime (M<=64 — a shared-scratch dirty line would evict over the NPU write and
            * resurrect a mid-row SENT); cold-only for prefill (M>64, not interleaved — avoids the per-op full flush). */
        c->mwarm[i] = 1;
        subs[i].timeout = orki_mm_timeout_ms(); orki_rknpu_submit_ioctl(fd, &subs[i], w->domain);
    }
    if (fp16_hard && c->mc_error) {   /* dropped fp16 K-slice — DON'T resubmit (sticky re-drop + D-state hang, campaign4).
        * Clean-reap the stuck job + int8 health-gate, then leave mc_error set for the run-level nc=1 bit-exact backstop. */
        /* RESET-FIRST (campaign5 kernel-source root-cause): the vendor rknpu_job_abort() path (hit whenever a submit
         * errors/times out) does rknpu_iommu_domain_put() -> msleep(100) -> rknpu_soft_reset() — a 100ms window with the
         * IOMMU torn down but the NPU CDMA NOT yet halted, so a wild in-flight fp16 fetch escapes into kernel RAM ->
         * slab corruption -> panic (netconsole: Oops in an unrelated proc's kmem_cache_alloc). rknpu_job_timeout_clean()
         * is safe ONLY because it soft-resets BEFORE teardown. So HALT the DMA ourselves FIRST — RKNPU_ACT_RESET
         * (-> rknpu_soft_reset) as the very first recovery action + a settle — so that if any later recovery submit
         * errors into abort, its domain_put window has NO live DMA to escape. Closes the kernel-ordering hole. */
        { struct rknpu_action ra; memset(&ra, 0, sizeof ra); ra.flags = RKNPU_ACT_RESET; ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &ra);
          struct timespec ts = {0, 5000000}; nanosleep(&ts, NULL); }   /* 5ms settle: let the CDMA fully quiesce before any further submit */
        ork_kmsg("F16 drop (K=%d N=%d M=%d) -> RESET-FIRST (halt DMA) + reap_stuck + int8 dummy health-gate + de-escalate to nc=1 (NO fp16 resubmit)", w->K, w->N, M);
        ork_npu_reap_stuck(c, nc);
        int ok = ork_dummy_probe(c);
        ork_kmsg("F16 drop recovery: int8 dummy-probe %s (mc_error stays set -> nc=1 recompute)", ok ? "PASS (NPU dispatching)" : "FAIL (still wedged)");
        for (int z = 0; z < ORK_MAXCORE; z++) c->mwarm[z] = 0;   /* reap/probe reset+reused the scratch -> force a cold re-warm on the nc=1 backstop */
    }
    /* Stash the round context so ork_dyn_end recovers a dropped colsplit round (the M=1 int8 decode path also
     * hits the ~1/2000 doorbell-drop: one core's N-column slice never lands, leaving its re-seeded sentinel
     * column = SENT). colsplit is int8-only; hardened (M<=64) = full-surface seed, matching orki_mc_recover_resubmit. */
    h->mc_nc = nc; h->mc_dt = dt; h->mc_dom = w->domain; h->mc_seed_all = hardened;   /* mc_dt: I8 recover; fp16 (Stage 1) => recov_max 0 (drains in-submit) */
    for (int i = 0; i < nc && i < ORK_MAXCORE; i++) { h->mc_subs[i] = subs[i]; h->mc_Pc[i] = Pc[i]; }
    return h;
}
