/* npu/i8/dyn.c — the NONBLOCK doorbell BEGIN paths: ork_dyn_begin and the multi-core ork_dyn_begin_mc, plus the drain/completion helpers they share.
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

int orki_slice_rescue_or_refuse(ork_npu *c,ork_w *w,int M,const void *A,void *C,int nc){
    if(w->sliced){ int rs=ork_mm_run_sliced(c,w->sliced,M,A,C,nc); if(rs>=0) return rs; }
    return ORK_RC_WEDGE_PRONE;
}
#include "spine_kernels.h"

int ork_dyn_grouped_end(ork_dyn_chain *h);  /* B: grouped-int4 float scale-accumulate drain */
int ork_dyn_end(ork_dyn_chain *h);
/* TASK #4: multi-M int4 onto the NONBLOCK doorbell spine. Decompose the M rows of one int4 weight into a chain
 * of M=1 int4 programs distributed across cores via ork_i4_dyn_begin_mc (the validated M=1 int4 doorbell: full-
 * surface int16 seed+poll, int16->int32 widen, and — task #4 — the same drop-recover as int8). Bit-identical to
 * per-row. Single-slice only (the doorbell's envelope); returns -4 (caller refuses — ORK_RC_WEDGE_PRONE) for
 * wide-N/K or when the M-program chain doesn't fit the per-core regcmd/task buffers. */


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

/* ================= DYNAMIC STEERED SUBMISSION API (validated by tools/steer_probe + doorbell_id_probe) =====
 * A run_chain_i8 chain, but submitted NONBLOCK so the host can (a) watch per-op progress via each op's output
 * doorbell (dc civac poll), (b) HALT it mid-flight to free the NPU early (write 0x0014=0 into a future op's
 * live regcmd descriptor — the sequencer reads it from DRAM at exec-time), (c) later, redirect the next-pointer
 * for runtime routing. v1 constraints: M=1 per task, single-slice conforming K (K%512==0, K<=4096), and A/C
 * resident in ork_dma_alloc buffers (so we hold DMA addrs + poll outputs coherently). One program per task
 * (P==S). Steering must lead the sequencer by ~1-2 ops (time it off ork_dyn_progress). */
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
        orki_i8_synth(rc, 1, K, N, adma, bdma, cdma, 1, CBUF, 0);
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
            orki_i8_synth(rcs, 1, lK, lN, (uint32_t)spinA.dma, lB, (uint32_t)spinC.dma, 1, CBUF, 0);
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

/* HW-doorbell eligibility: the exact acceptance ork_dyn_begin_mc enforces per task. int8/fp16 = single-slice,
 * conforming K%512 && K<=4096, M<=64, Sn==1 (fp16 adds M*K<=32768). int4 = M==1, single K/N-slice (its HW
 * chain is M=1-only and writes int16). An op of an hw=1 KIND that fails this downgrades to the SW break path
 * (its SEQ_CLASS fn). Kept in lockstep with begin_mc / begin_mc_i4's per-task guards — change both together. */
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
    if (dt == DT_I4) return ork_i4_dyn_begin_mc(c, S, tasks, nc);   /* int4 (int16 out, M=1) has its own branch */
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
     * opt-in env ORK_DYN_F16 is removed (it was already a no-op). */
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
         * fp16 stays Sn==1 — the fp16 `orki_f16_synth()` has no output-stride arg, so a strided column-slice can't be
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
        /* fp16 M envelope. WAS `M*K > 32768` — that constant is the int8 one (32768 BYTES at
         * 1 B/elem) and is 2x TOO LOOSE for fp16's 2 B/elem, so this path MISCOMPUTED at
         * non-pow2 K (measured: K=384 real ceiling 42, this permitted 85; K=640 -> 25 vs 51).
         * orki_f16_mcap is the measured envelope (1 CBUF bank sched=0, 11 banks sched=1). */
        if (dt == DT_F16 && tasks[i].M > orki_f16_mcap(w->K, orki_f16_sched(w->K))) return NULL;
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
                    orki_i8_synth(rc, M, Kp, N, aks, (uint32_t)w->Bb[ks].dma, cbase + (uint32_t)((size_t)ks * M * N * 4), sched, CBUF, 0);   /* [M,N] partial ks */
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
                    orki_i8_synth(rc, mc, K, N, adma + (uint32_t)((size_t)m0 * K), bdma, cbase + (uint32_t)((size_t)m0 * N * 4), 1, CBUF, 0);
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
                                    orki_f16_synth   (rc, M, K, Nc, adma, bdma, cdma, schedf, CBUF); }               /* fp16: Sn==1 only (no stride arg) — guarded above */
                else              orki_i8_synth(rc, M, K, Nc, adma, bdma, cdma, 1, CBUF, scat ? 0 : ((Sn > 1) ? N : 0));  /* scatter=contiguous; else strided column-slice / single */
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
