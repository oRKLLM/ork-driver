/* npu/i8/colsplit.c — the COLUMN-SPLIT multicore doorbell path: each core owns a contiguous N-range of one wide
 * submit, with a per-K-slice lockstep barrier and parallel f32 accumulate. Also the fp16 wide-K multicore
 * path (ORK_F16_COLSPLIT), which is why it is its own file rather than part of the chain lifecycle
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

ork_dyn_chain *ork_dyn_begin_colsplit(ork_npu *c, const ork_mm_task_i8 *t, int ncreq);   /* fwd: fp16 colsplit routed from run_multicore */
#define ORK_RC_F16_SC (-502)   /* internal run_multicore->orki_run() signal: fp16 fallback, retry the single-core fp16 reference (never the blocking mcworker) */
void ork_install_term(void);   /* fwd: graceful-SIGTERM install (defined near the doorbell poll) */
/* SLICE-AND-DICE RESCUE (#33): a shape run_multicore has no verified single-submit path for would
 * return ORK_RC_WEDGE_PRONE. If orki_pack() pre-built doorbell tiles for it (w->sliced), RUN it on those
 * instead — one chained doorbell submit over c_base tiles, bit-exact. If there are no tiles (a shape
 * we don't pre-slice) OR the sliced run itself errors, REFUSE — never a blocking fall-back (#45). */

static void *ork_csub_worker(void *vp){ struct ork_csub *a = vp; ork_npu *c = a->c; int i = a->i, fd = c->fd;
    /* (a `cold` capture used to live here for an fp16 cold-buffer warmup that was never implemented —
     * the variable was dead and the comment claimed a behaviour that did not exist. Measured 2026-08-27:
     * a cold warmup on this path changes nothing; the post-ACT_RESET drop is not a cold-buffer effect.) */
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
        int KS = c->soc->ks, kstart = 0;
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
         * 60-180s -> rknpu_job_abort. Detect is ~1.5x the measured slice (orki_f16_slice_us); set the submit timeout to
         * ~1x the slice (still >> a legit slice, which lands via IRQ before any timeout_clean regardless). Bootstrap
         * 60ms (< the 800ms bootstrap detect) until the first slice sets the baseline; never exceed detect_ms. */
        int recov_tmo = orki_f16_slice_us > 0 ? (int)(orki_f16_slice_us/1000) + 2 : 60;
        if (recov_tmo > detect_ms) recov_tmo = detect_ms; if (recov_tmo < 3) recov_tmo = 3;
        int f16_sentinel = (getenv("ORK_F16_SENTINEL") != NULL) && !c->f16_force_blocking;   /* force_blocking overrides -> blocking (heal). SENTINEL RECOVERY: submit NONBLOCK + CPU poll-drain each slice
            * (last-word gate + full-slice civac verify), tight timeout. Fast wedge-detect (~poll timeout, not the 8s blocking
            * timeout) AND the CPU never blocks in-kernel (no D-state hard-wedge); on a stuck sentinel -> mc_error -> run-level
            * recovery. Keeps the barrier's per-slice drain semantics (poll drains before advancing). int8-decode-path style. */
        const char *spte = getenv("ORK_F16_SENTINEL_TMO_US"); double f16_poll_ovr = spte ? atof(spte) : 0.0;   /* explicit detect-timeout override (us); 0 => AUTO = 1.5x the measured slice-completion time (orki_f16_slice_us) */
        for (int ks = 0; ks < Sk; ks++) {
            int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS;
            /* kcap MUST match the tiling below (it derives np_ks, the program count) — both now
             * come from the one measured envelope so they cannot disagree. */
            int kcap = orki_f16_mcap(Kp, orki_f16_sched(Kp));
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
                    /* AUTO detect timeout = 150% of the measured slice-completion time (orki_f16_slice_us, updated on every
                     * successful land) — adaptive per shape instead of a flat 800ms: once the warm/first slice sets the
                     * baseline (~ms), detect drops to ~20ms, cutting the detect-to-reset VULNERABILITY WINDOW ~40x (the
                     * period the wedged NPU sits un-halted, able to keep wild-writing). 20ms floor so a tiny first
                     * measurement + jitter never false-positive; 800ms bootstrap until the first success; env overrides. */
                    double f16_poll_tmo = f16_poll_ovr > 0 ? f16_poll_ovr : (orki_f16_slice_us > 0 ? 1.5 * orki_f16_slice_us : 800000.0);
                    if (f16_poll_ovr <= 0 && f16_poll_tmo < 20000.0) f16_poll_tmo = 20000.0;
                    size_t last = (size_t)ks * per + per - 1; double pt = ork_now_us(); int wedged = 0;
                    for (;;) { __asm__ volatile("dc civac,%0"::"r"(&o[last]):"memory");   /* O(1) LAST-WORD gate: cheap wedge-detect (a full-slice scan per poll iter was ~7x slower). Coherency for the accumulate is the FROM_DEVICE bsync below; a real wedge faults early so the last word stays SENT. */
                        if (o[last] != ORK_DYN_SENT) { double el = ork_now_us() - pt; if (el > orki_f16_slice_us) orki_f16_slice_us = el; break; }   /* landed: update the slice-time baseline that drives the 1.5x auto timeout */
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
                double el = ork_now_us() - pt;
                if (el > 3e6) {   /* SENTINEL NEVER LANDED => the op did not run. Falling through here used
                    * to drop straight into the FROM_DEVICE bsync and the host accumulate, so a dropped tile
                    * was summed as if it had completed -- silent wrong output. Flag it the same way the fp16
                    * per-slice poll above does (mc_error -> run-level recovery / nc=1 backstop). */
                    c->mc_error = 1;
                    fprintf(stderr, "[ork] ERROR: colsplit NB sentinel never landed (core=%d nout=%d, waited %.0f ms) — the op did NOT run; flagging mc_error instead of accumulating stale output\n",
                            i, no, el/1000.0);
                    break; }
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
            * int8 WIDE-K branch with orki_f16_synth()/f32/fp16-chunk. base (Sk==1) => single partial (accumulate is a copy).
            * Weight offset t0*Kp*32 and the 108-reg task are IDENTICAL to int8/mcworker (only orki_f16_synth()+Bb+dtype differ). */
            int CBUFf = (CBUF > 32768) ? 32768 : CBUF;   /* fp16 M-scheduler is validated only to the 32768-tile; a larger cbuf miscomputes mc>~cap (mcworker applies the same cap) */
            int KS = c->soc->ks;
            struct rknpu_task *tkf = (struct rknpu_task*)c->mtk[i].cpu;
            size_t ksz = (size_t)w->Sk * M * Ncore * 4;   /* Sk f32 partials [ks][M][Ncore] */
            if (c->mccsz[i] < ksz) { orki_bdestroy(fd, &c->mcc[i]); c->mcc[i] = orki_bcreate(fd, ksz, 0x403, c->dom_active);
                if (!c->mcc[i].cpu) { free(h); return NULL; } c->mccsz[i] = ksz; c->mwarm[i] = 0; }
            struct buf *CC = &c->mcc[i];
            /* pre-grow mrc/mtk for this core: fp16 chunks are small (<=8 @ K>=2048) => Sk*ceil(M/chunk) programs.
             * bound generously (512); tkf re-fetched after any grow. */
            size_t needrc = (size_t)512 * REGCMD_N * 4, needtk = (size_t)512 * sizeof(struct rknpu_task);
            /* keep the SRAM request on the regrow path too, else a grown chain silently drops to DRAM */
            if (RC->size < needrc) { orki_bdestroy(fd, &c->mrc[i]);
                c->mrc[i] = orki_bcreate(fd, needrc, 0x403 | (getenv("ORK_NO_SRAM_DB") ? 0u : (uint32_t)RKNPU_MEM_TRY_ALLOC_SRAM), c->dom_active);
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
                int sched = orki_f16_sched(Kp);
                /* fp16 M-tile cap = the MEASURED envelope (orki_f16_mcap). The old
                 * mg_max*64 / (RBf/2)/Kp / 4*R expression was wrong in BOTH directions: too small
                 * at K=512/1024 (320/128 vs a real 352/176) and far too LARGE at K=128 (4*R=1024
                 * vs a real 256), which is how ork_f16_mm_run silently miscomputed at K=128 for
                 * any M in [257,1472] — this multicore/colsplit path is the one it takes. */
                int kcap = orki_f16_mcap(Kp, sched);
                uint32_t wbase = (uint32_t)((contig ? (w->Bbc.dma + sloff) : w->Bb[ks].dma) + (uint64_t)t0 * Kp * 32);   /* CONTIG: slice base = Bbc + cumulative slice offset (one buffer); else per-slice Bb[ks]. N-tile stride Kp*32 (== mcworker) */
                for (int m0 = 0; m0 < M; m0 += kcap) { int mc = (M-m0<kcap)?(M-m0):kcap;
                    if ((size_t)(np2+1) * REGCMD_N * 4 > RC->size) { free(h); return NULL; }
                    memset(rc, 0, REGCMD_N * 4);
                    orki_f16_synth(rc, mc, Kp, Ncore, (uint32_t)(a_base + (goff + (size_t)m0*Kp)*2), wbase,
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
                    orki_i8_synth(rc, mc, Kp, Ncore, (uint32_t)(a_base + goff + (size_t)m0*Kp), wbase,
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
                  orki_i8_synth(rc, mc, K, segw, adma + (uint32_t)((size_t)m0 * K), wbase,
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
        { struct rknpu_action ra; memset(&ra, 0, sizeof ra); ra.flags = RKNPU_ACT_RESET; orki_io_ok(fd, DRM_IOCTL_RKNPU_ACTION, &ra, "ACT_RESET(f16-drop recover)", 1);
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
