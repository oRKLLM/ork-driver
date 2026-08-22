/* npu/i4/chain.c — PC-chained and doorbell int4 graphs, MoE expert coalescing.
 *
 * Part of the i4 datapath; shared declarations in npu/i4/i4.h. Split out of npu/i4.c for the
 * same reason i8 is a folder: one datapath, sized for reading. */
#define _GNU_SOURCE   /* CPU_SET/pthread_setaffinity_np, as npu.c does */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include <math.h>
#include "ork_regs.h"
#include "regcmd_i8.h"
#include "orkd_proto.h"
#include "npu/internal.h"
#include "npu/core.h"
#include "regcmd_i4.h"
#include "npu/i4/i4.h"

int orki_i4_validate=-1;   /* ORK_I4_VALIDATE: per-program regcmd validation (DEBUG, off by default) */

/* BCHAIN rows-per-weight-stream ceiling. MEASURED, not derived. Full write-up: wiki
 * "Exp-2026-08-21 Native W4A4 Prefill Hang".
 *
 *     K      256  512  768  1024  1536  2048  2560  3072  3584  4096  5120  6144  8192
 *     Hmax    64   32   22    16    11     8     7     4     4     4     3     3     2
 *     ceil    64   32   22    16    11     8     7   [ 6     5 ]   4   [ 4 ]   3     2
 *
 * THE TABLE IS THE RULE. No closed form fits these thirteen points and interpolation is unsafe in
 * BOTH directions: ceil overshoots at 3072/3584/5120 (miscompute or hang) and undershoots at 6144.
 * The fractional part does not decide it either — K=768 and K=3072 both have frac .33 and round
 * OPPOSITE ways (22 vs 4). floor, next_pow2 and CBUF bank-containment each die on a specific point
 * (see the wiki page). An unmeasured K falls back to floor(12288/K) = 0.75x the naive 16384/K, 0.75
 * being the largest margin any measured point needed (K=3072: 4 vs 5.33) — an empirical safety
 * factor, NOT a guarantee. Measure before trusting a new K.
 * Exceeding the ceiling does one of two things, and only the second shows up in rc:
 *     rc=0, DIFFERENT checksum -> SILENT MISCOMPUTE     (K=2048 H>=9; K=1024 H=5..8)
 *     rc=-1 after ~15 s        -> doorbell never lands  (K=3072 H>=5; K=1024 H<=4)
 * The latter is "native W4A4 hangs at prefill": ACCEPTED on every core (submit_rc=0) but the NPU
 * never starts it (hw_elapse=0, int_status=0), so the recover loop resets 6x then fails — determin-
 * istic per (K,H), 5/5. So rc is not a validity test and neither is timing; only a checksum compare
 * is (H merely re-tiles M, so every valid H must be BIT-IDENTICAL). Measured N=1024 M=32, and K=768
 * re-measured at M=64 gives the same 22 => M-independent. An UNMEASURED K is a known risk with a
 * silent failure mode: pin it with i4_hcap_probe first (method on the wiki page). H<2 is refused by
 * the callers (-4), routing the shape to the proven per-row doorbell. */
static int orki_i4_hcap(int K){
    static const short KT[] = {256,512,768,1024,1536,2048,2560,3072,3584,4096,5120,6144,8192};
    static const short HT[] = { 64,  32,  22,   16,   11,    8,    7,    4,    4,    4,    3,    3,    2};
    for (unsigned i = 0; i < sizeof KT / sizeof *KT; i++) if (KT[i] == K) return HT[i];
    int H = 12288 / K;        /* unmeasured: conservative fallback, see above */
    return H > 64 ? 64 : H;   /* 64 = where measurement stops (K<256 unprobed), not a HW bound */
}

int ork_i4_mm_run_chain(ork_npu *c, int S, const ork_mm_task_i4 *tasks) {
    if (!c) return -1;
    if (S < 1 || S > 1024) return -2;
    if (!tasks) return -2;

    /* Single matmul: use the optimized run_i4 path (multi-core column-split) rather than the
     * single-core chain path. Chaining only pays off when batching S>1 independent matmuls. */
    if (S == 1) return ork_i4_mm_run(c, tasks[0].w, tasks[0].M, tasks[0].A, tasks[0].C);

    /* chained weights share one submit => one domain; swap in that domain's scratch */
    if (tasks[0].w && (tasks[0].w->domain != c->dom_active || (tasks[0].w->domain!=0 && !c->dom_save))) orki_dom_activate(c, tasks[0].w->domain);
    int fd = c->fd;
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        if (!w || w->dtype != DT_I4) return -2;
        if (tasks[i].M != 1) return -2;
        if (w->Sn != 1 || w->Sk != 1) return -2;
        if (orki_check_overlap("ork_i4_mm_run_chain", (uintptr_t)tasks[i].A, (uintptr_t)tasks[i].A + (size_t)tasks[i].M * w->K, (uintptr_t)tasks[i].C, (uintptr_t)tasks[i].C + (size_t)tasks[i].M * w->N * 4)) return -1;
    }

    ork_npu_enter(c, 4 /* DT_I4_CHAIN */, XP_I4CHAIN, OCK_HW);

    int ok = 0;
    int max_K = 0, max_N = 0;
    for (int i = 0; i < S; i++) {
        if (tasks[i].w->K > max_K) max_K = tasks[i].w->K;
        if (tasks[i].w->N > max_N) max_N = tasks[i].w->N;
        struct buf *abuf = orki_dma_find(c, tasks[i].A);
        if (abuf) orki_bsync(fd, abuf, RKNPU_MEM_SYNC_FROM_DEVICE);
    }

    struct buf chain_A = orki_bcreate(fd, (size_t)S * max_K, 0x403, c->dom_active);
    struct buf chain_C = orki_bcreate(fd, (size_t)S * max_N * 2, 0x403, c->dom_active);
    if (!chain_A.cpu || !chain_C.cpu) {
        if (chain_A.cpu) orki_bdestroy(fd, &chain_A);
        if (chain_C.cpu) orki_bdestroy(fd, &chain_C);
        return -1;
    }

    uint32_t act_dma[1024];
    uint32_t out_dma[1024];

    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        uint8_t *A_dst = (uint8_t*)chain_A.cpu + (size_t)i * max_K;
        orki_i4_tile_Aslice(A_dst, tasks[i].A, 0, w->K);
        act_dma[i] = (uint32_t)(chain_A.dma + (size_t)i * max_K);
        out_dma[i] = (uint32_t)(chain_C.dma + (size_t)i * max_N * 2);
    }
    orki_bsync(fd, &chain_A, RKNPU_MEM_SYNC_TO_DEVICE);

    /* Clean-before-write the int16 output scratch. chain_C is bcreate'd fresh each call and the kernel can
     * hand back a recycled DMA region carrying dirty CPU cache lines (from a prior occupant). Those lines
     * evict to DRAM AFTER the NPU writes chain_C, clobbering the NPU output -> "correct run 0, garbage runs
     * 1+" on warm reuse. Dirty the whole surface then clean it to DRAM (TO_DEVICE) so no stale line survives
     * to evict later -- same full-surface clean-before as ork_i4_dyn_begin_mc's doorbell scratch. */
    memset(chain_C.cpu, 0, (size_t)S * max_N * 2);
    orki_bsync(fd, &chain_C, RKNPU_MEM_SYNC_TO_DEVICE);

    struct buf extra[2] = {chain_A, chain_C};
    uint32_t rc[REGCMD_I4_N];

    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        orki_i4_synth(rc, 1, w->K, w->N, act_dma[i], (uint32_t)w->Bb[0].dma, out_dma[i]);
        if (orki_validate_regcmd("run_chain_i4", c, rc, REGCMD_I4_N, w, extra, 2)) { ok = -1; goto cleanup; }

        if (i < S - 1) {
            uint64_t next_dma = c->regcmd.dma + (i + 1) * REGCMD_I4_N * 4;
            rc[216] = 0x0010 | ((next_dma & 0xffff) << 16);
            rc[217] = (0x0101 << 16) | ((next_dma >> 16) & 0xffff);
            rc[218] = 0x0014 | (0x0037 << 16);
            rc[219] = (0x0101 << 16) | (0);
        } else {
            rc[216] = 0;
            rc[217] = 0;
            rc[218] = 0x00000014;
            rc[219] = 0x01010000;
        }
        memcpy((char*)c->regcmd.cpu + i * REGCMD_I4_N * 4, rc, sizeof(rc));
    }
    orki_bsync(fd, &c->regcmd, RKNPU_MEM_SYNC_TO_DEVICE);

    /* Mirror the validated i4_mcworker multi-task path exactly: one rknpu_task per chained regcmd,
     * task_number=S, subcore={0,S}, and the same reps/submit discipline (rknpu_submit_ioctl with a
     * cold-buffer warmup rep + orki_bsync(C) between reps). The kernel programs first_task+last_task and
     * the HW PC-chain (rc[216..219]) walks the middle; it waits until the HW task counter reaches S. */
    struct rknpu_task *t = c->task.cpu;
    memset(t, 0, (size_t)S * sizeof(struct rknpu_task));
    for (int i = 0; i < S; i++) {
        t[i].enable_mask = 0xd;
        t[i].int_mask = 0x300;
        t[i].int_clear = 0x1ffff;
        t[i].regcfg_amount = 116;
        t[i].regcmd_addr = c->regcmd.dma + (size_t)i * REGCMD_I4_N * 4;
    }
    orki_bsync(fd, &c->task, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);

    static int tc = -2;
    if (tc == -2) { const char* e = getenv("ORK_NPU_TESTCORE"); tc = e ? atoi(e) : 0; if (tc < 0 || tc > 2) tc = 0; }

    struct rknpu_submit sub; memset(&sub, 0, sizeof sub);
    sub.flags = ork_ppflags(); sub.task_number = S; sub.task_obj_addr = c->task.obj; sub.fence_fd = -1;
    sub.core_mask = 1u << tc;
    sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, S};

    int reps = c->warmed ? 1 : 2;
    for (int rep = 0; rep < reps; rep++) {
        int last = (rep == reps - 1);
        sub.timeout = orki_mm_timeout_ms();
        if (orki_rknpu_submit_ioctl(fd, &sub, tasks[0].w->domain)) { if (last) { ok = -1; goto cleanup; } continue; }
        orki_bsync(fd, &chain_C, RKNPU_MEM_SYNC_FROM_DEVICE);
    }
    c->warmed = 1;

    orki_bsync(fd, &chain_C, RKNPU_MEM_SYNC_FROM_DEVICE);
    for (int i = 0; i < S; i++) {
        int16_t *o = (int16_t*)((uint8_t*)chain_C.cpu + (size_t)i * max_N * 2);
        int32_t *C = tasks[i].C;
        int N = tasks[i].w->N;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        int col = 0;
        for (; col <= N - 16; col += 16) {
            int16x8_t vo16_0 = vld1q_s16(&o[col]);
            int16x8_t vo16_1 = vld1q_s16(&o[col + 8]);
            vst1q_s32(&C[col], vmovl_s16(vget_low_s16(vo16_0)));
            vst1q_s32(&C[col + 4], vmovl_s16(vget_high_s16(vo16_0)));
            vst1q_s32(&C[col + 8], vmovl_s16(vget_low_s16(vo16_1)));
            vst1q_s32(&C[col + 12], vmovl_s16(vget_high_s16(vo16_1)));
        }
        for (; col < N; col++) {
            C[col] = o[col];
        }
#else
        for (int col = 0; col < N; col++) {
            C[col] = o[col];
        }
#endif

        struct buf *cbuf = orki_dma_find(c, tasks[i].C);
        if (cbuf) orki_bsync(fd, cbuf, RKNPU_MEM_SYNC_TO_DEVICE);
    }

cleanup:
    orki_bdestroy(fd, &chain_A);
    orki_bdestroy(fd, &chain_C);
    return ok;
}

int ork_i4_dyn_probe(ork_npu *c, int S, const ork_mm_task_i4 *tasks) {
    if (!c) return -1;
    if (S < 1 || S > 1024) return -2;
    if (!tasks) return -2;
    if (tasks[0].w && (tasks[0].w->domain != c->dom_active || (tasks[0].w->domain!=0 && !c->dom_save))) orki_dom_activate(c, tasks[0].w->domain);
    int fd = c->fd;
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        if (!w || w->dtype != DT_I4) return -2;
        if (tasks[i].M != 1) return -2;                      /* int4 HW chain is M=1 only (like run_chain_i4) */
        if (w->Sn != 1 || w->Sk != 1) return -2;
    }
    ork_npu_enter(c, 4 /* DT_I4_CHAIN */, XP_I4CHAIN, OCK_HW);
    int ok = 0, max_K = 0, max_N = 0;
    for (int i = 0; i < S; i++) {
        if (tasks[i].w->K > max_K) max_K = tasks[i].w->K;
        if (tasks[i].w->N > max_N) max_N = tasks[i].w->N;
        struct buf *abuf = orki_dma_find(c, tasks[i].A);
        if (abuf) orki_bsync(fd, abuf, RKNPU_MEM_SYNC_FROM_DEVICE);
    }
    struct buf chain_A = orki_bcreate(fd, (size_t)S * max_K, 0x403, c->dom_active);
    struct buf chain_C = orki_bcreate(fd, (size_t)S * max_N * 2, 0x403, c->dom_active);
    if (!chain_A.cpu || !chain_C.cpu) { if (chain_A.cpu) orki_bdestroy(fd,&chain_A); if (chain_C.cpu) orki_bdestroy(fd,&chain_C); return -1; }
    uint32_t act_dma[1024], out_dma[1024];
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        orki_i4_tile_Aslice((uint8_t*)chain_A.cpu + (size_t)i * max_K, tasks[i].A, 0, w->K);
        act_dma[i] = (uint32_t)(chain_A.dma + (size_t)i * max_K);
        out_dma[i] = (uint32_t)(chain_C.dma + (size_t)i * max_N * 2);
    }
    orki_bsync(fd, &chain_A, RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf extra[2] = {chain_A, chain_C};
    uint32_t rc[REGCMD_I4_N];
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        orki_i4_synth(rc, 1, w->K, w->N, act_dma[i], (uint32_t)w->Bb[0].dma, out_dma[i]);
        if (orki_validate_regcmd("ork_i4_dyn_probe", c, rc, REGCMD_I4_N, w, extra, 2)) { ok = -1; goto cleanup; }
        if (i < S - 1) { uint64_t next_dma = c->regcmd.dma + (i + 1) * REGCMD_I4_N * 4;
            rc[216] = 0x0010 | ((next_dma & 0xffff) << 16); rc[217] = (0x0101 << 16) | ((next_dma >> 16) & 0xffff);
            rc[218] = 0x0014 | (0x0037 << 16); rc[219] = (0x0101 << 16) | (0);
        } else { rc[216] = 0; rc[217] = 0; rc[218] = 0x00000014; rc[219] = 0x01010000; }
        memcpy((char*)c->regcmd.cpu + i * REGCMD_I4_N * 4, rc, sizeof(rc));
        if (i == 0 && getenv("ORK_I4PROBE_DUMP")) { fprintf(stderr,"[i4probe] op0 regcmd desc rc[216..219]=%08x %08x %08x %08x  aA=%08x aC=%08x\n",rc[216],rc[217],rc[218],rc[219],act_dma[0],out_dma[0]); }
    }
    orki_bsync(fd, &c->regcmd, RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *t = c->task.cpu;
    memset(t, 0, (size_t)S * sizeof(struct rknpu_task));
    for (int i = 0; i < S; i++) { t[i].enable_mask = 0xd; t[i].int_mask = 0x300; t[i].int_clear = 0x1ffff;
        t[i].regcfg_amount = 116; t[i].regcmd_addr = c->regcmd.dma + (size_t)i * REGCMD_I4_N * 4; }
    orki_bsync(fd, &c->task, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
    /* seed the FULL int16 output surface with the sentinel (int4's int16 write order over N is NOT
     * guaranteed last-col-last, unlike int8/fp16, so a single last-col sentinel poll races; poll ALL
     * elements written — mirrors the fp16 full-surface seed). */
    for (int i = 0; i < S; i++) { int N = tasks[i].w->N; int16_t *o = (int16_t*)((uint8_t*)chain_C.cpu + (size_t)i*max_N*2);
        for (int col = 0; col < N; col++){ volatile int16_t *db = (volatile int16_t*)&o[col];
            *db = ORK_I4_SENT16; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); } }
    __asm__ volatile("dsb ish":::"memory");
    struct rknpu_submit sub; memset(&sub, 0, sizeof sub);
    sub.flags = ork_ppflags() | 0x2u;                        /* DELTA 1: NONBLOCK (vs run_chain_i4's blocking) */
    sub.task_number = S; sub.task_obj_addr = c->task.obj; sub.fence_fd = -1; sub.core_mask = 1;
    sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)S};
    sub.timeout = orki_mm_timeout_ms();
    c->warmed = 1;
    if (orki_rknpu_submit_ioctl(fd, &sub, tasks[0].w->domain)) { ok = -1; goto cleanup; }
    /* DELTA 2: poll the FULL int16 output surface to completion (every element != sentinel) instead of a
     * blocking wait. Op i is "done" only when ALL N of its int16 columns have been overwritten. */
    double t0 = ork_now_us();
    for (;;) { int alld = 1;
        for (int i = 0; i < S && alld; i++) { int N = tasks[i].w->N; int16_t *o = (int16_t*)((uint8_t*)chain_C.cpu + (size_t)i*max_N*2);
            for (int col = 0; col < N; col++){ volatile int16_t *db = (volatile int16_t*)&o[col];
                __asm__ volatile("dc civac,%0"::"r"(db):"memory"); if (*db == ORK_I4_SENT16){ alld = 0; break; } } }
        if (alld || ork_now_us() - t0 > 3e6) break; }
    orki_bsync(fd, &chain_C, RKNPU_MEM_SYNC_FROM_DEVICE);
    for (int i = 0; i < S; i++) {                            /* int16 -> int32 de-tile into caller C (== run_chain_i4) */
        int16_t *o = (int16_t*)((uint8_t*)chain_C.cpu + (size_t)i * max_N * 2);
        int32_t *C = tasks[i].C; int N = tasks[i].w->N;
        for (int col = 0; col < N; col++) C[col] = o[col];
        struct buf *cbuf = orki_dma_find(c, tasks[i].C); if (cbuf) orki_bsync(fd, cbuf, RKNPU_MEM_SYNC_TO_DEVICE);
    }
cleanup:
    orki_bdestroy(fd, &chain_A); orki_bdestroy(fd, &chain_C);
    return ok;
}

static int bch_db_cells_off(ork_npu *c,int i,int c0,int c1,int Wb,int N,int NG,int M,int H,int Wmax,int32_t *C,int mode,int only_tk,int tk_base){
    int tk=0;   /* #54 tk_base: this weight's program slot 0 within the shared per-core chain (0 for single-weight; the expert's cumulative program offset for coalesced multi-expert) */
    for(int nc2=c0;nc2<c1;nc2++){ int n0=nc2*Wb, Wc=(N-n0<Wb)?(N-n0):Wb, NBc=Wc/64;
        for(int g=0;g<NG;g++){ int Hg=(M-g*H<H)?(M-g*H):H;
            if(only_tk>=0 && tk!=only_tk){ tk++; continue; }   /* poll fast-gate: only the given program (last program lands last — chain runs in order) */
            int16_t *og=(int16_t*)c->mcc[i].cpu + (size_t)(tk_base+tk)*(size_t)(4*H*Wmax)*64;
            int ran=0;   /* mode 4: did THIS program write anything (>=1 non-sentinel cell = it ran)? */
            for(int j=0;j<Hg;j++) for(int b=0;b<NBc;b++){ size_t base=(size_t)(4*j+4*Hg*b)*64;   /* a 64-int16 block = 2 x 64B cache lines */
                if(mode==0){ for(int cc=0;cc<64;cc++) og[base+cc]=ORK_DYN_SENT16; }
                else if(mode==1){ __asm__ volatile("dc civac,%0"::"r"(&og[base]):"memory"); __asm__ volatile("dc civac,%0"::"r"(&og[base+32]):"memory");
                    for(int cc=0;cc<64;cc++) if(((volatile int16_t*)og)[base+cc]==ORK_DYN_SENT16) return 0; }
                else if(mode==3){ for(int cc=0;cc<64;cc++) if(og[base+cc]==ORK_DYN_SENT16) return 0; }
                else if(mode==4){ for(int cc=0;cc<64;cc++) if(og[base+cc]!=ORK_DYN_SENT16){ ran=1; break; } }   /* #54 COLLISION-TOLERANT landing: SENT16 (0x7fff) IS a reachable W4A4 int16 output, so mode 1/3 (any-cell==sentinel => not-landed) FALSE-MISS on a legit 0x7fff cell -> recover -> ACT_RESET -> multi-domain corruption. A REAL miss = the program NEVER ran = EVERY cell still sentinel; a landed program has >=1 non-sentinel cell (residual 0x7fff cells are real values, de-tiled correctly). Use ONLY at the poll timeout (by then a run program is fully written). */
                else { int32_t *crow=C+(size_t)(g*H+j)*N+n0+b*64; for(int cc=0;cc<64;cc++) crow[cc]=og[base+cc]; } }
            if(mode==4 && !ran) return 0;   /* this program is ENTIRELY sentinel => it truly never ran => real drop */
            tk++; } }
    return 1;
}

int orki_bch_db_cells(ork_npu *c,int i,int c0,int c1,int Wb,int N,int NG,int M,int H,int Wmax,int32_t *C,int mode,int only_tk){
    return bch_db_cells_off(c,i,c0,c1,Wb,N,NG,M,H,Wmax,C,mode,only_tk,0);   /* single-weight: base slot 0 */
}

static void *bch_db_worker(void *vp){
    struct bchdbw *a=vp; ork_npu *c=a->c; int fd=c->fd, i=a->core, NT=a->NT;
    int K=a->K,N=a->N,NG=a->NG,M=a->M,H=a->H,Wb=a->Wb,Wmax=a->Wmax; unsigned dom=a->dom;
    a->rc=0; if(NT<1) return NULL;
    orki_pin_big_core(i);
    uint8_t *abase=c->maf[i].cpu; memset(abase,0,(size_t)NG*(size_t)(2*H)*(K/2));   /* A packed once per M-group (stride-2) */
    for(int g=0;g<NG;g++){ int Hg=(M-g*H<H)?(M-g*H):H;
        for(int j=0;j<Hg;j++) orki_i4_tile_Aslice(abase+(size_t)(g*2*H+2*j)*(K/2), a->A+(size_t)(g*H+j)*K, 0, K); }
    orki_bsync(fd,&c->maf[i],RKNPU_MEM_SYNC_TO_DEVICE);
    int tk=0;
    for(int nc2=a->c0;nc2<a->c1;nc2++){ int n0=nc2*Wb, Wc=(N-n0<Wb)?(N-n0):Wb;
        uint32_t wdma=(uint32_t)(a->w->Bb[0].dma + (uint64_t)(n0/64)*K*32);
        for(int g=0;g<NG;g++){ int Hg=(M-g*H<H)?(M-g*H):H; uint32_t rc[REGCMD_I4_N];
            uint32_t aA=(uint32_t)c->maf[i].dma+(uint32_t)(g*2*H)*(K/2);
            uint32_t aC=(uint32_t)c->mcc[i].dma+(uint32_t)tk*(4*H*Wmax)*64*2;
            memset(rc,0,sizeof rc); orki_i4_synth(rc, 2*Hg, K, Wc, aA, wdma, aC);
            /* #54 int4 HW WEIGHT-REUSE (NEVER tried on int4 — only int8 M-fold #39). The BCHAIN loop is already
             * weight-stationary (N-tile outer, M-group g inner sharing wdma), so g==0 loads the N-tile weight and
             * g>0 can REUSE the CBUF-resident weight + skip the re-DMA by setting CNA_CBUF_CON0[13]=WEIGHT_REUSE
             * (0x2000). int4 differs from int8 (2x weight density, different CBUF banks, 0x1040 is the "poison"
             * K-schedule reg) so this is test-and-see: it may stay bit-exact, may hit the int8 "data-refetch"
             * correctness gap, or may trip the 0x1040 sensitivity. ORK_I4_WREUSE=1 weight-reuse, 2 +DATA_REUSE[12]. */
            if(g>0){ static int wr=-1; if(wr<0){ const char*e=getenv("ORK_I4_WREUSE"); wr=e?atoi(e):1; }   /* DEFAULT ON: measured stable ~6% (gate/up ~9%) bit-exact on the MoE expert-triple (test_moe_smoke, M=32). ORK_I4_WREUSE=0 opts out. */
                if(wr){ unsigned bits=((wr&1)?0x2000u:0)|((wr&2)?0x1000u:0); uint32_t v1040=0;
                    for(int k=0;k+1<REGCMD_I4_N;k+=2) if((rc[k]&0xffff)==0x1040 && (rc[k+1]>>16)==0x201){ v1040=((rc[k]>>16)&0xffff)|((rc[k+1]&0xffff)<<16); break; }
                    orki_setr(rc,REGCMD_I4_N,0x201,0x1040,v1040|bits); } }
            if(orki_i4_validate && orki_validate_regcmd("bch_db_worker", c, rc, REGCMD_I4_N, a->w, NULL, 0)){ a->rc=-1; c->mc_error=1; return NULL; }   /* per-program validate is a DEBUG check (ORK_I4_VALIDATE); off by default — it scales with program count and blk never had it */
            if(tk<NT-1){ uint32_t nd=(uint32_t)(c->mrc[i].dma+(size_t)(tk+1)*REGCMD_I4_N*4);
                rc[216]=0x0010|((nd&0xffff)<<16); rc[217]=(0x0101<<16)|((nd>>16)&0xffff);
                rc[218]=0x0014|(0x0037<<16); rc[219]=(0x0101<<16)|0; }
            else { rc[216]=0; rc[217]=0; rc[218]=0x00000014; rc[219]=0x01010000; }
            memcpy((char*)c->mrc[i].cpu+(size_t)tk*REGCMD_I4_N*4, rc, REGCMD_I4_N*4); tk++; } }
    orki_bsync(fd,&c->mrc[i],RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *t=c->mtk[i].cpu; memset(t,0,(size_t)NT*sizeof*t);
    for(int q=0;q<NT;q++){ t[q].enable_mask=0xd; t[q].int_mask=0x300; t[q].int_clear=0x1ffff;
        t[q].regcfg_amount=116; t[q].regcmd_addr=c->mrc[i].dma+(uint64_t)q*REGCMD_I4_N*4; }
    orki_bsync(fd,&c->mtk[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    memset(&a->sub,0,sizeof a->sub);
    a->sub.task_number=(uint32_t)NT; a->sub.task_obj_addr=c->mtk[i].obj;
    a->sub.core_mask=1u<<i; a->sub.fence_fd=-1;
    a->sub.subcore_task[0]=a->sub.subcore_task[1]=a->sub.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)NT};
    /* #54 int4 completion mode — 3 ways to wait for the chain + drain the DPU writeback:
     *   0 BLOCKVERIFY (default): seed SENT16, then BLOCKING submit (cheap IRQ kernel-wait, parallel across cores
     *     exactly like int8 colsplit's ork_csub_worker) + prepolled=0 so ork_dyn_end runs the SENT16 drain-verify
     *     ONCE, post-completion (the job is already done -> the gate lands immediately/after a short residual
     *     writeback lag) — NOT the from-t=0 repeated scan. int8 gets away with bare blocking because its writeback
     *     is coherent by job-done; int4's DPU writeback LAGS completion (why bare blocking miscomputes), so it
     *     still needs the one verify. Best of both: int8's blocking efficiency + int4's needed drain-verify.
     *     Measured: the nonblock host-poll (mode 1) cost 22-36% over blocking on the MoE expert shapes.
     *   1 NONBLOCK (ORK_I4_NB — historical): seed + nonblock doorbell + ork_dyn_end polls from t=0 (pipelined; the expensive
     *     int4 SENT16 scan spins the whole HW-exec window -> the 22-36%). Kept for A/B.
     *   2 BLOCKING (ORK_I4_BLOCKING): bare blocking, prepolled=1, NO drain-verify -> MISCOMPUTES. A/B only. */
    /* MEASURED (test_moe_prog, 2026-08): BLOCKVERIFY(0) 768us > NONBLOCK(1) 712us > BLOCKING(2, miscomputes) 555us.
     * BLOCKVERIFY lost: the drain-verify (mandatory for int4 correctness — DPU writeback lags job-done) costs MORE
     * than the blocking submit saves, and NONBLOCK already overlaps its poll-spin with the HW-exec window. So the
     * "555 blocking" is unattainable-while-correct; NONBLOCK is the best CORRECT option. Default = NONBLOCK(1). */
    static int i4mode=-1; if(i4mode<0) i4mode = getenv("ORK_I4_BLOCKVERIFY")?0 : (getenv("ORK_I4_BLOCKING")?2 : 1);
    if(i4mode!=2){   /* BLOCKVERIFY(0) + NONBLOCK(1): seed the SENT16 landing sentinel so ork_dyn_end can drain-verify */
        orki_bch_db_cells(c,i,a->c0,a->c1,Wb,N,NG,M,H,Wmax,NULL,0,-1); orki_bsync(fd,&c->mcc[i],RKNPU_MEM_SYNC_TO_DEVICE); __asm__ volatile("dsb ish":::"memory");
        if(i4mode==1){   /* NONBLOCK doorbell: submit + return; ork_dyn_end (prepolled=0) polls from t=0 */
            a->sub.flags=ork_ppflags()|0x2u; a->sub.timeout=orki_i4_submit_tmo_ms();
            orki_rknpu_submit_ioctl(fd,&a->sub,dom); c->mwarm[i]=1; a->rc=0; return NULL; }
        /* BLOCKVERIFY: fall through to the blocking submit; ork_dyn_end (prepolled=0) does the SHORT post-completion drain-verify */
    }
    a->sub.flags=ork_ppflags();   /* BLOCKING (no |0x2): parallel IRQ kernel-wait across cores (like int8 colsplit); a dropped job aborts -> rknpu_iommu_domain_put (no leak). */
    c->mwarm[i]=1;
    for(int attempt=0; attempt<4; attempt++){
        a->sub.timeout=orki_mm_timeout_ms();
        int rc=orki_rknpu_submit_ioctl(fd,&a->sub,dom);   /* BLOCKING (0x2 cleared): kernel-waits for job_done or aborts at timeout */
        if(rc==0){ a->rc=0; return NULL; }
        if(orki_ork_term){ a->rc=0; return NULL; }
        if(getenv("ORK_MC_DIAG")) fprintf(stderr,"[bch] BLOCKING drop core=%d dom=%u attempt=%d rc=%d -> resubmit (kernel aborted+domain_put, no leak)\n",i,dom,attempt,rc);
    }
    a->rc=-2; return NULL;
}

int orki_i4_run_bchain_db(ork_npu *c, ork_w *w, int M, const int8_t *A, int32_t *C, int nc){
    if(w->dtype!=DT_I4 || w->Sk!=1 || w->Sn!=1 || (w->N%64) || M<2) return -4;
    int fd=c->fd, K=w->K, N=w->N;
    /* accurate wedge telemetry: the BCHAIN worker skips validate_regcmd by default, so orki_last_op would
     * otherwise stay stale (mislabelling BCHAIN submits as the last mc_i4 op in ORK_PRESUBMIT_TRACE). */
    orki_last_op="run_i4_bchain_db"; orki_last_K=K; orki_last_N=N; orki_last_wdom=w->domain;
    orki_last_import=(w->own_buf_valid && w->own_buf.heap_fd>0) || (w->own_bufs && w->n_own_bufs>0 && w->own_bufs[0].heap_fd>0)
                  || (w->Bb && w->Bb[0].heap_fd>0);
    /* H = rows per batched submit: the weight streams ONCE per H rows, so H IS the int4
     * prefill weight-reuse factor. Buffers are sized from H at runtime (need_af/need_o below).
     * ORK_I4_H overrides it for RE — without that a probe can only measure this one value. */
    int H=orki_i4_hcap(K);   /* MEASURED ceiling — see orki_i4_hcap; exceeding it hangs or miscomputes */
    /* ORK_I4_DBNK=n: H_max scales with the DATA_BANK count (measured H_max = DBNK*16384/K), so this
     * must move together with the 0x1040 split written in i4/run.c. */
    { const char*d=getenv("ORK_I4_DBNK"); if(d){ int n=atoi(d); if(n>=1&&n<=11){ H=n*orki_i4_hcap(K); if(H>64)H=64; } } }
    { const char*e=getenv("ORK_I4_H"); if(e){int v=atoi(e); if(v>0) H=v;} }   /* re-read: lets one probe process sweep */
    if(H<2) return -4;
    int Wb=(131072/K)&~63;
    /* N-tile width = the WEIGHT-BANK WIDTH. 131072 int4 elements = 65536 B = one CBUF weight bank.
     * This is a CONTAINMENT requirement, not a budget: a native multi-M batch is valid only WITHIN
     * one weight bank (there is no D_BANK to hold a second), so a chunk that straddles banks hits the
     * 0x1040 "poison pill" and computes row 0 only. Derived + pass/fail-matched in the wiki entry
     * Exp-2026-07-08-INT4-BChain-Batch-Chain. DO NOT RAISE IT.
     * Confirmed the hard way 2026-08-21: sweeping ORK_I4_WB above this at K=2048 N=1024 M=128 gives
     * 1-2 SECOND submits (vs 2176 us at the correct Wb=64) — the driver self-healing a straddled
     * batch, which can even return a correct checksum. An earlier reading of that sweep as "the
     * budget is 5.5x too small" was wrong on both counts: it is not a budget, and checksum-only
     * probing cannot distinguish a valid config from a self-healed timeout.
     * STILL OPEN (small): the `&~63` discards up to 63 columns of real bank capacity at K that do not
     * divide 131072 — e.g. K=768 allows 170 columns/bank but rounds to 128. Capturing that needs the
     * de-tile below generalised off its exact Wmax=Wb/64 block count; keep any probe a multiple of 64
     * until then, or a host-arithmetic break reads as a hardware failure.
     * ORK_I4_WB overrides for RE. */
    { const char*e=getenv("ORK_I4_WB"); if(e){int v=atoi(e); if(v>0) Wb=v;} }   /* re-read: lets one probe process sweep */
    if(Wb<64)Wb=64; if(Wb>N)Wb=N;
    int NC=(N+Wb-1)/Wb, NG=(M+H-1)/H, Wmax=Wb/64;
    if(nc<1)nc=1; if(nc>NC)nc=NC; if(nc>c->soc->cores)nc=c->soc->cores; if(nc>ORK_MAXCORE)nc=ORK_MAXCORE;
    if(getenv("ORK_BCH_DEBUG")){ int ntmax=0; for(int i=0;i<nc;i++){ int lc0=(int)((long)i*NC/nc),lc1=(int)((long)(i+1)*NC/nc); int nt=(lc1-lc0)*NG; if(nt>ntmax)ntmax=nt; }
        fprintf(stderr,"[bch] K=%d N=%d M=%d H=%d Wb=%d NC=%d NG=%d nc=%d NTmax=%d\n",K,N,M,H,Wb,NC,NG,nc,ntmax); fflush(stderr); }
    unsigned dom=w->domain;
    if(w->domain!=c->dom_active || (w->domain && !c->dom_save)) orki_dom_activate(c,w->domain);
    if(getenv("ORK_I4_DIAG")) fprintf(stderr,"[i4diag] bchain w=%p submit_dom=%u dom_active=%d | Bb0.dma=0x%llx Bb0.domain=%d Bb0.obj=0x%llx Bb0.heap_fd=%d | K=%d N=%d M=%d Wb=%d NC=%d\n",
        (void*)w, dom, c->dom_active, (unsigned long long)w->Bb[0].dma, w->Bb[0].domain, (unsigned long long)w->Bb[0].obj, w->Bb[0].heap_fd, K, N, M, Wb, NC);
    ork_npu_enter(c, 4 /*DT_I4_CHAIN*/, XP_I4CHAIN, OCK_HW);
    if(orki_mc_ensure(c,nc)) return -1;
    if(getenv("ORK_I4_DIAG")) fprintf(stderr,"[i4diag] scratch dom=%d | W[0x%llx,+0x%llx) mtk_all=0x%llx mrc0=0x%llx maf0=0x%llx mcc0=0x%llx | overlap-check vs W\n",
        c->dom_active, (unsigned long long)w->Bb[0].dma, (unsigned long long)((size_t)K*N/2),
        (unsigned long long)c->mtk_all.dma, (unsigned long long)c->mrc[0].dma, (unsigned long long)c->maf[0].dma, (unsigned long long)c->mcc[0].dma);
    if(orki_i4_validate<0) orki_i4_validate=getenv("ORK_I4_VALIDATE")?1:0;   /* init once on the calling thread (before dispatch) */
    /* pre-size every core's buffers SINGLE-THREADED (no concurrent bcreate in the workers) */
    struct bchdbw args[ORK_MAXCORE];
    for(int i=0;i<nc;i++){
        int lc0=(int)((long)i*NC/nc), lc1=(int)((long)(i+1)*NC/nc), NT=(lc1-lc0)*NG;
        args[i]=(struct bchdbw){c,i,lc0,lc1,NT,K,N,NG,M,H,Wb,Wmax,w,A,C,dom,{0},0};
        if(NT<1) continue;
        size_t need_rc=(size_t)NT*REGCMD_I4_N*4, need_af=(size_t)NG*(size_t)(2*H)*(K/2);
        size_t need_o=(size_t)NT*(size_t)(4*H*Wmax)*64*2, need_tk=(size_t)NT*sizeof(struct rknpu_task);
        /* #54 SRAM SECONDARY-PIPE probe (ORK_MOE_SRAM_SCRATCH): route the SMALL per-program scratch — regcmd,
         * activation (maf), tasks, output (mcc) — into on-chip SRAM (separate port from DRAM; weight Bb stays
         * DRAM). Tests whether feeding the tiny-op reads/writes off a second port speeds the per-program floor.
         * bcreate fails SRAM over to DRAM if full. */
        static int msram=-1; if(msram<0) msram=getenv("ORK_MOE_SRAM_SCRATCH")?1:0;   /* SRAM alloc rejects the 0x400 IOMMU-align flag -> drop it + add TRY_ALLOC_SRAM (matches the working ork_dma_alloc_sram path) */
        unsigned f3 = msram ? (0x003u|RKNPU_MEM_TRY_ALLOC_SRAM) : 0x403u;             /* data scratch (mrc/maf/mcc): cacheable+non-contig, SRAM when on */
        unsigned fb = msram ? (0x00bu|RKNPU_MEM_TRY_ALLOC_SRAM) : 0x40bu;             /* task buf (mtk): +KERNEL_MAPPING for the kernel read */
        if(c->mrc[i].size<need_rc){ orki_bdestroy(fd,&c->mrc[i]); c->mrc[i]=orki_bscratch(c,need_rc,(int)f3,c->dom_active); c->mwarm[i]=0; }
        if(c->maf[i].size<need_af){ orki_bdestroy(fd,&c->maf[i]); c->maf[i]=orki_bscratch(c,need_af,(int)f3,c->dom_active); }
        if(c->mtk[i].size<need_tk){ orki_bdestroy(fd,&c->mtk[i]); c->mtk[i]=orki_bscratch(c,need_tk,(int)fb,c->dom_active); }
        if(c->mccsz[i]<need_o){ orki_bdestroy(fd,&c->mcc[i]); c->mcc[i]=orki_bscratch(c,need_o,(int)f3,c->dom_active); c->mccsz[i]=need_o; c->mwarm[i]=0; }
        if(!c->mrc[i].cpu||!c->maf[i].cpu||!c->mtk[i].cpu||!c->mcc[i].cpu) return -1;
    }
    c->mc_error=0; orki_in_doorbell=1;
    /* PARALLEL: each pool worker builds + seeds + BLOCKING-submits (kernel drains) its own core; a drop aborts
     * (rknpu_job_abort -> domain_put, no refcount leak) and the worker resubmits. Mirrors int8 ork_csub_worker. */
    orki_npu_pool_ensure(c);
    pthread_mutex_lock(&c->pmu); c->pjob=args; c->pjob_nc=nc; c->pjob_fn=bch_db_worker; c->pjob_stride=sizeof(struct bchdbw);
    c->pdone=0; c->pgen++; pthread_cond_broadcast(&c->pgo); pthread_mutex_unlock(&c->pmu);
    bch_db_worker(&args[0]);                                                              /* core 0 on the calling thread */
    pthread_mutex_lock(&c->pmu); while(c->pdone<nc-1) pthread_cond_wait(&c->pdn,&c->pmu); pthread_mutex_unlock(&c->pmu);
    orki_in_doorbell=0;
    int missed=0; for(int i=0;i<nc;i++){ if(args[i].rc==-1) return -1; if(args[i].rc==-2) missed=1; }   /* -1 build err, -2 blocking exhausted */
    /* #54 The BLOCKING workers already drained every core (prepolled) — ork_dyn_end just de-tiles the int16 tiles
     * into C (i4batch hook -> bch_db_cells mode-2) and frees h. No leak (blocking abort domain_put's drops), so no
     * refcount-based recover / reap-at-boundary is needed. On an exhausted drop (rc==-2) fall back (return -1). */
    ork_dyn_chain *h=calloc(1,sizeof *h); if(!h) return -1;
    h->c=c; h->S=nc; h->P=nc; h->N=N; h->mc=1; h->esz=2; h->dom=dom; h->mc_nc=nc; h->mc_dt=DT_I4; h->mc_dom=dom;
    h->prepolled = getenv("ORK_I4_BLOCKING")?1:0;   /* #54 BLOCKVERIFY(default)+NONBLOCK: prepolled=0 => ork_dyn_end drain-verifies (BLOCKVERIFY's verify is short, post-completion). Pure ORK_I4_BLOCKING: prepolled=1, no verify (miscomputes; A/B only). */
    h->i4batch=1; h->b_H=H; h->b_Wb=Wb; h->b_Wmax=Wmax; h->b_NG=NG; h->b_M=M; h->b_N=N; h->b_C=C;
    for(int i=0;i<nc && i<ORK_MAXCORE;i++){
        h->outbuf[i]=&c->mcc[i]; h->outptr[i]=(int32_t*)c->mcc[i].cpu;
        h->nout[i]=(int)((size_t)args[i].NT*(size_t)(4*H*Wmax)*64); h->oM[i]=1;
        h->b_c0[i]=args[i].c0; h->b_c1[i]=args[i].c1; h->b_NT[i]=args[i].NT;
        h->mc_subs[i]=args[i].sub; h->mc_Pc[i]=args[i].NT;
    }
    if(missed){ free(h); return -1; }   /* a core's blocking submit never completed — don't de-tile garbage; caller falls back */
    int last=ork_dyn_end(h);   /* prepolled: skips poll, de-tiles (i4batch) into C, frees h */
    return (last==nc-1) ? 0 : -1;
}

static void *bch_mw_worker(void *vp){
    struct bchmw *a=vp; ork_npu *c=a->c; int fd=c->fd, i=a->core;
    int K=a->K,N=a->N,H=a->H,Wb=a->Wb,Wmax=a->Wmax,NC=a->NC; unsigned dom=a->dom;
    a->rc=0; if(a->e0>=a->e1) return NULL; orki_pin_big_core(i);
    int NTtot=0; size_t AFtot=0;
    for(int e=a->e0;e<a->e1;e++){ int NG=(a->ex[e].M+H-1)/H; NTtot+=NC*NG; AFtot+=(size_t)NG*(size_t)(2*H)*(K/2); }
    memset(c->maf[i].cpu,0,AFtot);
    /* stage every expert's A (per M-group, stride-2), contiguously in maf */
    size_t aoff=0;
    for(int e=a->e0;e<a->e1;e++){ const ork_mm_task_i4 *t=&a->ex[e]; int M=t->M, NG=(M+H-1)/H;
        for(int g=0;g<NG;g++){ int Hg=(M-g*H<H)?(M-g*H):H;
            for(int j=0;j<Hg;j++) orki_i4_tile_Aslice((uint8_t*)c->maf[i].cpu+aoff+(size_t)(g*2*H+2*j)*(K/2), t->A+(size_t)(g*H+j)*K, 0, K); }
        aoff+=(size_t)NG*(size_t)(2*H)*(K/2); }
    orki_bsync(fd,&c->maf[i],RKNPU_MEM_SYNC_TO_DEVICE);
    /* build the chained regcmd across all experts (tk = cumulative program index) */
    aoff=0; int tk=0;
    for(int e=a->e0;e<a->e1;e++){ const ork_mm_task_i4 *t=&a->ex[e]; ork_w *w=t->w; int M=t->M, NG=(M+H-1)/H;
        for(int nc2=0;nc2<NC;nc2++){ int n0=nc2*Wb, Wc=(N-n0<Wb)?(N-n0):Wb;
            uint32_t wdma=(uint32_t)(w->Bb[0].dma + (uint64_t)(n0/64)*K*32);
            for(int g=0;g<NG;g++){ int Hg=(M-g*H<H)?(M-g*H):H; uint32_t rc[REGCMD_I4_N];
                uint32_t aA=(uint32_t)c->maf[i].dma+(uint32_t)(aoff+(size_t)(g*2*H)*(K/2));
                uint32_t aC=(uint32_t)c->mcc[i].dma+(uint32_t)tk*(4*H*Wmax)*64*2;
                memset(rc,0,sizeof rc); orki_i4_synth(rc, 2*Hg, K, Wc, aA, wdma, aC);
                if(g>0){ static int wr=-1; if(wr<0){ const char*e=getenv("ORK_I4_WREUSE"); wr=e?atoi(e):1; }   /* #54 coalesce path weight-reuse (same as bch_db_worker; default ON, ~8-15% gate/up bit-exact) */
                    if(wr){ unsigned bits=((wr&1)?0x2000u:0)|((wr&2)?0x1000u:0); uint32_t v1040=0;
                        for(int k=0;k+1<REGCMD_I4_N;k+=2) if((rc[k]&0xffff)==0x1040 && (rc[k+1]>>16)==0x201){ v1040=((rc[k]>>16)&0xffff)|((rc[k+1]&0xffff)<<16); break; }
                        orki_setr(rc,REGCMD_I4_N,0x201,0x1040,v1040|bits); } }
                if(tk<NTtot-1){ uint32_t nd=(uint32_t)(c->mrc[i].dma+(size_t)(tk+1)*REGCMD_I4_N*4);
                    rc[216]=0x0010|((nd&0xffff)<<16); rc[217]=(0x0101<<16)|((nd>>16)&0xffff);
                    rc[218]=0x0014|(0x0037<<16); rc[219]=(0x0101<<16)|0; }
                else { rc[216]=0; rc[217]=0; rc[218]=0x00000014; rc[219]=0x01010000; }
                memcpy((char*)c->mrc[i].cpu+(size_t)tk*REGCMD_I4_N*4, rc, REGCMD_I4_N*4); tk++; } }
        aoff+=(size_t)NG*(size_t)(2*H)*(K/2); }
    orki_bsync(fd,&c->mrc[i],RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tt=c->mtk[i].cpu; memset(tt,0,(size_t)NTtot*sizeof*tt);
    for(int q=0;q<NTtot;q++){ tt[q].enable_mask=0xd; tt[q].int_mask=0x300; tt[q].int_clear=0x1ffff;
        tt[q].regcfg_amount=116; tt[q].regcmd_addr=c->mrc[i].dma+(uint64_t)q*REGCMD_I4_N*4; }
    orki_bsync(fd,&c->mtk[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    memset(&a->sub,0,sizeof a->sub);
    a->sub.flags=ork_ppflags()|0x2u; a->sub.task_number=(uint32_t)NTtot; a->sub.task_obj_addr=c->mtk[i].obj;
    a->sub.core_mask=1u<<i; a->sub.fence_fd=-1;
    a->sub.subcore_task[0]=a->sub.subcore_task[1]=a->sub.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)NTtot};
    { int tb=0; for(int e=a->e0;e<a->e1;e++){ int NG=(a->ex[e].M+H-1)/H;   /* seed SENT16 for every expert's cells */
        bch_db_cells_off(c,i,0,NC,Wb,N,NG,a->ex[e].M,H,Wmax,NULL,0,-1,tb); tb+=NC*NG; } }
    orki_bsync(fd,&c->mcc[i],RKNPU_MEM_SYNC_TO_DEVICE); __asm__ volatile("dsb ish":::"memory");
    a->sub.timeout=orki_i4_submit_tmo_ms(); orki_rknpu_submit_ioctl(fd,&a->sub,dom); c->mwarm[i]=1;   /* #54 bounded timeout: a dropped coalesced job is then reapable by the per-expert fallback's timeout_clean */
    int NGl=(a->ex[a->e1-1].M+H-1)/H, NTl=NC*NGl, tbl=NTtot-NTl;   /* last program of the whole chain lands last */
    double t0=ork_now_us();
    for(;;){
        if(bch_db_cells_off(c,i,0,NC,Wb,N,NGl,a->ex[a->e1-1].M,H,Wmax,NULL,1,NTl-1,tbl)){   /* last-program civac gate */
            orki_bsync(fd,&c->mcc[i],RKNPU_MEM_SYNC_FROM_DEVICE);
            int ok=1, vb=0;
            for(int e=a->e0;e<a->e1 && ok;e++){ int NG=(a->ex[e].M+H-1)/H;
                if(!bch_db_cells_off(c,i,0,NC,Wb,N,NG,a->ex[e].M,H,Wmax,NULL,3,-1,vb)) ok=0; vb+=NC*NG; }
            if(ok){ int db=0; for(int e=a->e0;e<a->e1;e++){ int NG=(a->ex[e].M+H-1)/H;
                    bch_db_cells_off(c,i,0,NC,Wb,N,NG,a->ex[e].M,H,Wmax,a->ex[e].C,2,-1,db); db+=NC*NG; }
                a->rc=0; return NULL; } }
        if(orki_ork_term){ a->rc=0; return NULL; }
        double el=ork_now_us()-t0; if(el>300000.0){ a->rc=-2; return NULL; }
        if(el>1000.0){ struct timespec ts={0,50000}; nanosleep(&ts,NULL); }
    }
}

int orki_i4_run_experts_bchain_db(ork_npu *c, const ork_mm_task_i4 *ex, int ntask, int nc){
    int fd=c->fd, K=ex[0].w->K, N=ex[0].w->N;
    /* H = rows per batched submit: the weight streams ONCE per H rows, so H IS the int4
     * prefill weight-reuse factor. Buffers are sized from H at runtime (need_af/need_o below).
     * ORK_I4_H overrides it for RE — without that a probe can only measure this one value. */
    int H=orki_i4_hcap(K);   /* MEASURED ceiling — see orki_i4_hcap; exceeding it hangs or miscomputes */
    /* ORK_I4_DBNK=n: H_max scales with the DATA_BANK count (measured H_max = DBNK*16384/K), so this
     * must move together with the 0x1040 split written in i4/run.c. */
    { const char*d=getenv("ORK_I4_DBNK"); if(d){ int n=atoi(d); if(n>=1&&n<=11){ H=n*orki_i4_hcap(K); if(H>64)H=64; } } }
    { const char*e=getenv("ORK_I4_H"); if(e){int v=atoi(e); if(v>0) H=v;} }   /* re-read: lets one probe process sweep */
    if(H<2) return -4;
    int Wb=(131072/K)&~63;
    /* N-tile width = the WEIGHT-BANK WIDTH. 131072 int4 elements = 65536 B = one CBUF weight bank.
     * This is a CONTAINMENT requirement, not a budget: a native multi-M batch is valid only WITHIN
     * one weight bank (there is no D_BANK to hold a second), so a chunk that straddles banks hits the
     * 0x1040 "poison pill" and computes row 0 only. Derived + pass/fail-matched in the wiki entry
     * Exp-2026-07-08-INT4-BChain-Batch-Chain. DO NOT RAISE IT.
     * Confirmed the hard way 2026-08-21: sweeping ORK_I4_WB above this at K=2048 N=1024 M=128 gives
     * 1-2 SECOND submits (vs 2176 us at the correct Wb=64) — the driver self-healing a straddled
     * batch, which can even return a correct checksum. An earlier reading of that sweep as "the
     * budget is 5.5x too small" was wrong on both counts: it is not a budget, and checksum-only
     * probing cannot distinguish a valid config from a self-healed timeout.
     * STILL OPEN (small): the `&~63` discards up to 63 columns of real bank capacity at K that do not
     * divide 131072 — e.g. K=768 allows 170 columns/bank but rounds to 128. Capturing that needs the
     * de-tile below generalised off its exact Wmax=Wb/64 block count; keep any probe a multiple of 64
     * until then, or a host-arithmetic break reads as a hardware failure.
     * ORK_I4_WB overrides for RE. */
    { const char*e=getenv("ORK_I4_WB"); if(e){int v=atoi(e); if(v>0) Wb=v;} }   /* re-read: lets one probe process sweep */
    if(Wb<64)Wb=64; if(Wb>N)Wb=N;
    int NC=(N+Wb-1)/Wb, Wmax=Wb/64;
    if(nc<1)nc=1; if(nc>ntask)nc=ntask; if(nc>c->soc->cores)nc=c->soc->cores; if(nc>ORK_MAXCORE)nc=ORK_MAXCORE;
    unsigned dom=ex[0].w->domain;
    if(dom!=(unsigned)c->dom_active || (dom && !c->dom_save)) orki_dom_activate(c,(int)dom);
    orki_last_op="run_i4_experts_bchain_db"; orki_last_K=K; orki_last_N=N; orki_last_wdom=(int)dom;
    ork_npu_enter(c, 4 /*DT_I4_CHAIN*/, XP_I4CHAIN, OCK_HW);
    if(orki_mc_ensure(c,nc)) return -1;
    struct bchmw args[ORK_MAXCORE];
    int totalNT=0; for(int e=0;e<ntask;e++){ int NG=(ex[e].M+H-1)/H; totalNT+=NC*NG; }
    int e_start=0, priorNT=0;
    for(int i=0;i<nc;i++){
        int targetNT=(int)((long)(i+1)*totalNT/nc), e_end=e_start, accNT=0;
        while(e_end<ntask){ int NG=(ex[e_end].M+H-1)/H; if(priorNT+accNT+NC*NG>targetNT && e_end>e_start) break; accNT+=NC*NG; e_end++; }
        if(i==nc-1) e_end=ntask;
        args[i]=(struct bchmw){c,i,e_start,e_end,ex,K,N,H,Wb,Wmax,NC,dom,{0},0};
        priorNT+=accNT; e_start=e_end;
        int cNT=0; size_t cAF=0; for(int e=args[i].e0;e<args[i].e1;e++){ int NG=(ex[e].M+H-1)/H; cNT+=NC*NG; cAF+=(size_t)NG*(size_t)(2*H)*(K/2); }
        if(cNT<1) continue;
        size_t need_rc=(size_t)cNT*REGCMD_I4_N*4, need_o=(size_t)cNT*(size_t)(4*H*Wmax)*64*2, need_tk=(size_t)cNT*sizeof(struct rknpu_task);
        if(c->mrc[i].size<need_rc){ orki_bdestroy(fd,&c->mrc[i]); c->mrc[i]=orki_bscratch(c,need_rc,0x403,c->dom_active); c->mwarm[i]=0; }
        if(c->maf[i].size<cAF){ orki_bdestroy(fd,&c->maf[i]); c->maf[i]=orki_bscratch(c,cAF,0x403,c->dom_active); }
        if(c->mtk[i].size<need_tk){ orki_bdestroy(fd,&c->mtk[i]); c->mtk[i]=orki_bscratch(c,need_tk,0x40b,c->dom_active); }
        if(c->mccsz[i]<need_o){ orki_bdestroy(fd,&c->mcc[i]); c->mcc[i]=orki_bscratch(c,need_o,0x403,c->dom_active); c->mccsz[i]=need_o; c->mwarm[i]=0; }
        if(!c->mrc[i].cpu||!c->maf[i].cpu||!c->mtk[i].cpu||!c->mcc[i].cpu) return -1;
    }
    c->mc_error=0; orki_in_doorbell=1;
    orki_npu_pool_ensure(c);
    pthread_mutex_lock(&c->pmu); c->pjob=args; c->pjob_nc=nc; c->pjob_fn=bch_mw_worker; c->pjob_stride=sizeof(struct bchmw);
    c->pdone=0; c->pgen++; pthread_cond_broadcast(&c->pgo); pthread_mutex_unlock(&c->pmu);
    bch_mw_worker(&args[0]);
    pthread_mutex_lock(&c->pmu); while(c->pdone<nc-1) pthread_cond_wait(&c->pdn,&c->pmu); pthread_mutex_unlock(&c->pmu);
    orki_in_doorbell=0;
    int bad=0; for(int i=0;i<nc;i++){ if(args[i].rc==-1) return -1; if(args[i].rc==-2) bad=1; }
    if(bad){   /* a core dropped -> re-run ITS experts serially via the proven per-expert BCHAIN (its first submit's
                * timeout_clean reaps the dropped coalesced job — bounded i4_submit_tmo_ms made it past-timeout) */
        for(int i=0;i<nc;i++){ if(args[i].rc!=-2) continue;
            for(int e=args[i].e0;e<args[i].e1;e++) if(orki_i4_run_bchain_db(c, ex[e].w, ex[e].M, ex[e].A, ex[e].C, nc)) return -1; }
    }
    ork_dom_flush_if_dirty(c);   /* #54: clear any stuck job (from a coalesced drop or the per-expert fallback) BEFORE the mcc bdestroy / next op switches domains. In-domain, post-join, safe. No-op unless a real miss occurred. */
    /* #54 RESIDENT MULTI-DOMAIN: the coalesced OUTPUT scratch (mcc, ~33 MB/core for a whole _exps tensor) is
     * TRANSIENT — its results are already de-tiled to the host C above. Weights stay resident per-domain, but if
     * mcc were left allocated it would be PARKED per-domain by dom_activate and accumulate one ~100 MB copy per
     * domain -> across the auto-sized ~16 domains that's ~1.6 GB of bcreate scratch, exhausting the kernel GEM/CMA
     * pool so a fresh orki_bcreate (even a tiny mtk_all) EINVALs at ~the 5th domain. Free it here so only the ACTIVE
     * domain's mcc exists; the next run re-allocs it in its own domain. (int8's per-op scratch is tiny so it never
     * hit this; the big COALESCED output is int4-MoE-specific.) mrc/maf/mtk are ~MB and reused by other paths. */
    for(int i=0;i<nc;i++){ if(c->mcc[i].cpu){ orki_bdestroy(fd,&c->mcc[i]); c->mcc[i]=(struct buf){0}; c->mccsz[i]=0; c->mwarm[i]=0; } }
    return 0;
}

int ork_i4_mm_run_experts(ork_npu *c, const ork_mm_task_i4 *ex, int ntask, int nc){
    if(!c || ntask<1 || !ex) return -1;
    for(int e=0;e<ntask;e++){ ork_w*w=ex[e].w;
        if(!w||w->dtype!=DT_I4||w->Sk!=1||w->Sn!=1||(w->N%64)||ex[e].M<1) return -2;
        if(w->domain!=ex[0].w->domain || w->K!=ex[0].w->K || w->N!=ex[0].w->N) return -2; }  /* one submit => one domain + one shape */
    return orki_i4_run_experts_bchain_db(c, ex, ntask, nc);   /* M-batched BCHAIN programs chained across experts */
}
/* ============ B: GROUPED int4 (W4A4 per-K-group scales) on the NONBLOCK doorbell ============
 * ork_i4_mm_run_grouped's doorbell path. Row-decomposed (M=1 tasks across cores); each row emits Sn*Sk programs
 * (K-slice = gsize G, Sk = K/G groups). Output = Sk int16 partial blocks of [N] per row. Unlike A2's int-sum,
 * the drain (ork_dyn_grouped_end) FLOAT scale-accumulates: C[m][n] = sum_g aS[m*Sk+g]*bS[g*N+n]*partial_g[n]
 * (matches the grouped drain). NULL if ineligible (chain/scratch too big) -> caller refuses (ORK_RC_WEDGE_PRONE; the blocking i4_mcworker_g path is removed #45). */
/* int4 (W4A4) multi-core NONBLOCK doorbell — the DT_I4 sibling of ork_dyn_begin_mc. int4 is structurally
 * different from int8/fp16 at the hardware level: the datapath writes an int16 (2-byte) accumulator (widened
 * to int32 on the host, esz=2) and its HW chain is M=1 only — so it CANNOT share the 4-byte-C body. This keeps
 * that body byte-identical and specialises the int4 divergences: tile_i4_Aslice host-A staging (0.5 B/elem),
 * synth_i4 / REGCMD_I4_N stride / regcfg_amount=116, an int16 per-core output scratch (ALWAYS copy-back — the
 * NPU never writes the caller's int32 C in place), and a FULL-SURFACE int16 sentinel seed that doubles as the
 * clean-before-write the fresh/reused scratch needs (int4's int16 write-order over N is not last-col-last, so
 * both the seed and the completion poll cover the whole row). Proven bit-exact single-core by ork_i4_dyn_probe;
 * this is the productionised multi-core form the scheduler dispatches. Host (malloc) A only, as for int8/fp16.
 * end() drains via the esz-aware ork_dyn_done_i and widens int16->int32 into the caller's C. */
ork_dyn_chain *ork_i4_dyn_begin_mc(ork_npu *c, int S, const ork_mm_task_i8 *tasks, int nc);  /* int4 M=1 doorbell (defined below) */
int orki_i4_submit_tmo_ms(void);   /* #54 bounded int4 doorbell submit timeout (TCLEAN reap precondition); defined near the int4 workers */
ork_dyn_chain *ork_i4_dyn_begin_mc_grouped(ork_npu *c, int M, ork_w *w, const int8_t *A, const float *aScale, const float *bScale, float *Cf, int nc);  /* B: grouped-int4 doorbell */
/* #54 COALESCE: run MANY int4 experts (each M>=1 rows) through ONE nonblock doorbell. Decompose every expert's
 * M rows into M=1 tasks and hand the WHOLE set to ork_i4_dyn_begin_mc — the doorbell distributes+chains them
 * across the cores in one submit-set per core (the HW chaining is the doorbell's job; we don't hand-wire it).
 * Collapses the per-expert submit storm (2059 matmuls x 3 cores) to ~nc submits per _exps tensor. All experts
 * MUST share one iommu domain (the doorbell = one submit = one domain); the caller streams a layer's experts
 * into a single domain. Returns 0 ok, -4 refuse (chain/buffer too big -> caller falls back), -1 error. */
int orki_i4_run_experts_bchain_db(ork_npu *c, const ork_mm_task_i4 *ex, int ntask, int nc);   /* multi-expert BCHAIN (defined below) */
/* Async pipelined submit (precision-agnostic) — orkd+ring mode only. Enqueue one matmul for w WITHOUT blocking
 * and get a ticket; ork_mm_collect(ticket) reads C later. Returns <0 if unavailable (no ring, or the op is too
 * big for a ring slot — use the synchronous ork_f16_mm_run* instead). Keeping several ops in flight lets each op's
 * transport (memcpy + handshake) overlap the NPU compute of the ones ahead of it — the decode pipeline. */
int ork_mm_submit(ork_npu *c, ork_w *w, int M, const void *A){
    if(!c || !c->daemon || !w || !w->is_orkd || !orkd_has_ring(c->daemon)) return -1;
    uint32_t dt = w->dtype==DT_F16 ? ORKD_DT_F16 : w->dtype==DT_I4 ? ORKD_DT_I4 : ORKD_DT_I8;
    return orkd_ring_submit(c->daemon, w->orkd_id, M, w->K, w->N, dt, A);
}

int ork_dyn_grouped_end(ork_dyn_chain *h) {
    if (!h || !h->i4g) return -2;
    ork_npu *c = h->c; int fd = c->fd, rc = 0;
    /* Drain on the SHARED doorbell recover loop (mirrors ork_dyn_end@12140): poll all rows; on a dropped round
     * (mc int4 output never landed) orki_mc_recover_resubmit (RESET + re-seed SENT16 via its esz==2 branch + resubmit
     * each core) and re-poll, up to recov_max; auto-dump only a TRUE stall (recover exhausted). The grouped begin
     * already stashed mc_nc/mc_dt=DT_I4/mc_dom/mc_subs/mc_Pc, so this self-heals exactly like the int8 mc path. */
    orki_in_doorbell = 1;
    int recov_max = (h->mc_nc > 0 && h->mc_dt == DT_I4) ? 6 : 0;
    int landed = 0, edone[1024];
    for (int recov = 0; ; recov++) {
        double t0 = ork_now_us();
        for (int i = 0; i < h->S && i < 1024; i++) edone[i] = 0;
        double miss_to = (recov < recov_max) ? 300000.0 : 3e6;   /* fast miss-detect while retries remain, else full completion wait */
        for (;;) { int n = 0; for (int x = 0; x < h->S; x++) { if (!edone[x]) edone[x] = ork_dyn_done_i(h, x); n += edone[x]; }
            if (n >= h->S) { landed = 1; break; }
            if (orki_ork_term) break;
            double el = ork_now_us() - t0; if (el > miss_to) break;
            if (el > 1000.0) { struct timespec ts = {0, 50000}; nanosleep(&ts, NULL); } }
        if (landed || orki_ork_term) break;
        if (recov < recov_max) { if (getenv("ORK_MC_DIAG")) fprintf(stderr, "[MC-RECOVER grp] int4 grouped round never landed (attempt %d) — reset+resubmit\n", recov);
            h->c->dom_dirty = 1;   /* #54: int4 drop -> reap-at-boundary (see ork_dom_flush_if_dirty) */
            orki_mc_recover_resubmit(h); continue; }
        break;
    }
    orki_in_doorbell = 0;
    if (!landed) { rc = -1; ork_dyn_dump(h, "grouped-i4 doorbell miss (recover exhausted)"); }
    struct buf *done[1024]; int nd = 0;
    for (int i = 0; i < h->S; i++) { struct buf *b = h->outbuf[i]; int seen = 0;
        for (int j = 0; j < nd; j++) if (done[j] == b) seen = 1;
        if (!seen && b) { orki_bsync(fd, b, RKNPU_MEM_SYNC_FROM_DEVICE); if (nd < 1024) done[nd++] = b; } }
    int N = h->i4g_N, Sk = h->i4g_Sk; const float *aS = h->i4g_aS, *bS = h->i4g_bS; float *Cf = h->i4g_Cf;
    for (int m = 0; m < h->S; m++) { const int16_t *blk = (const int16_t*)h->outptr[m]; float *cr = Cf + (size_t)m * N;
        for (int n = 0; n < N; n++) { float acc = 0;
            for (int g = 0; g < Sk; g++) acc += aS[(size_t)m*Sk+g] * bS[(size_t)g*N+n] * (float)blk[(size_t)g*N+n];
            cr[n] = acc; } }
    __asm__ volatile("dsb ish":::"memory");
    free(h);
    if (orki_ork_term) { sigaction(SIGTERM, &orki_prev_sig[0], NULL); raise(SIGTERM); }
    return rc;
}
ork_dyn_chain *ork_i4_dyn_begin_mc(ork_npu *c, int S, const ork_mm_task_i8 *tasks, int nc) {
    if (nc < 1 || nc > c->soc->cores) nc = c->soc->cores; if (nc > S) nc = S;
    for (int i = 0; i < S; i++) { ork_w *w = tasks[i].w;
        if (!w || w->dtype != DT_I4 || tasks[i].M != 1 || w->Sk > 16) return NULL;   /* int4 HW chain: M=1; A1: Sn>1 N-tiled; A2: Sk>1 K-split (int16 partials summed in end) — Sk bounded (per-row A-slice array) */
        if (w->domain != tasks[0].w->domain) return NULL; }   /* one submit => one domain */
    if (tasks[0].w->domain != c->dom_active || (tasks[0].w->domain && !c->dom_save)) orki_dom_activate(c, tasks[0].w->domain);
    ork_npu_enter(c, 4 /*DT_I4_CHAIN*/, XP_I4CHAIN, OCK_HW);
    if (orki_mc_ensure(c, nc)) return NULL;
    int fd = c->fd;
    ork_dyn_chain *h = calloc(1, sizeof *h); if (!h) return NULL;
    h->c = c; h->S = S; h->P = S; h->N = tasks[0].w->N; h->dom = tasks[0].w->domain; h->reserve = S; h->mc = 1; h->esz = 2;
    unsigned dom = tasks[0].w->domain;
    /* int4 ALWAYS copy-back: per-core in-domain int16 scratch (M=1 => N int16/op), widened in end() */
    for (int i = 0; i < nc; i++) { int lo=(int)((long)i*S/nc), hi=(int)((long)(i+1)*S/nc), P=hi-lo; if (P<1) continue;
        size_t osz = 0; for (int p = lo; p < hi; p++) osz += (size_t)tasks[p].w->N * tasks[p].w->Sk * 2;   /* Sk int16 partial blocks/row (A2 K-split; Sk==1 => N int16) */
        if (c->mccsz[i] < osz) { orki_bdestroy(fd, &c->mcc[i]); c->mcc[i] = orki_bcreate(fd, osz, 0x403, c->dom_active);
            if (!c->mcc[i].cpu) { free(h); return NULL; } c->mccsz[i] = osz; c->mwarm[i] = 0; } }
    uint32_t rc[REGCMD_I4_N];
    int NMAX = c->soc->nmax, KS = ORK_I4_KS;
    struct rknpu_submit subs[ORK_MAXCORE]; int Pc[ORK_MAXCORE]; memset(Pc, 0, sizeof Pc);
    for (int i = 0; i < nc; i++) {
        int lo = (int)((long)i * S / nc), hi = (int)((long)(i+1) * S / nc), P = hi - lo;
        if (P < 1) { Pc[i] = 0; continue; }
        /* PROGRAM count decoupled from row-task count: a row emits Sn*Sk programs (N-slices x K-slices). */
        int Pcore = 0; for (int p = lo; p < hi; p++) Pcore += tasks[p].w->Sn * tasks[p].w->Sk;
        Pc[i] = Pcore;
        if ((size_t)Pcore * REGCMD_I4_N * 4 > c->mrc[i].size || (size_t)Pcore * sizeof(struct rknpu_task) > c->mtk[i].size) { free(h); return NULL; }
        struct buf *RC = &c->mrc[i], *AF = &c->maf[i], *CC = &c->mcc[i]; struct rknpu_task *tk = (struct rknpu_task*)c->mtk[i].cpu;
        size_t astage = 0, coff = 0; int pp = 0;
        for (int p = lo; p < hi; p++) {
            const ork_mm_task_i8 *t = &tasks[p]; ork_w *w = t->w; int K = w->K, N = w->N, Sn = w->Sn, Sk = w->Sk;
            /* A2 K-SPLIT: stage this row's Sk activation K-slices (each Kp nibbles, 0.5 B/elem), shared by the
             * row's N-slices. Sk<=16 (guarded above); each slice reads A[:, ks*KS : ks*KS+Kp]. */
            uint32_t aslice[16];
            for (int ks = 0; ks < Sk; ks++) { int k0 = ks * KS, Kp = (K - k0 < KS) ? (K - k0) : KS; size_t asz = (size_t)Kp / 2;
                if (astage + asz > AF->size) { free(h); return NULL; }
                orki_i4_tile_Aslice((uint8_t*)AF->cpu + astage, (const int8_t*)t->A, k0, Kp);
                aslice[ks] = (uint32_t)(AF->dma + astage); astage += asz; }
            /* A1 N-tile x A2 K-split: one program per (N-slice ns, K-slice ks). The row's output is Sk blocks
             * of [N] int16 (block ks = the K-slice-ks partial; column-slices write their [Nc] within it); the
             * drain SUMS the Sk blocks per column -> int32 C (oSk). Sk==1 => one block = A1's plain widen. */
            for (int ns = 0; ns < Sn; ns++) {
                int n0 = ns * NMAX, Nc = (N - n0 < NMAX) ? (N - n0) : NMAX;
                for (int ks = 0; ks < Sk; ks++) {
                    int k0 = ks * KS, Kp = (K - k0 < KS) ? (K - k0) : KS;
                    uint32_t cdma = (uint32_t)(CC->dma + coff + (size_t)ks * N * 2 + (size_t)n0 * 2);   /* block ks, columns [n0,n0+Nc) */
                    struct buf *WT = &w->Bb[(size_t)ns * Sk + ks];
                    uint32_t bdma = (uint32_t)WT->dma;                                                   /* weight N-slice ns, K-slice ks */
                    if (getenv("ORK_I4_DIAG")) { unsigned char *bc=(unsigned char*)WT->cpu;
                        fprintf(stderr,"[i4diag] mc_i4 wdom=%d dom_active=%d imported=%d ns=%d ks=%d Kp=%d Nc=%d | bdma=0x%llx obj=0x%llx size=%zu cpu=%p bytes[0..7]=",
                            w->domain, c->dom_active, (w->own_bufs&&w->n_own_bufs>0)||w->own_buf_valid, ns, ks, Kp, Nc,
                            (unsigned long long)WT->dma, (unsigned long long)WT->obj, WT->size, WT->cpu);
                        if(bc) for(int z=0;z<8;z++) fprintf(stderr,"%02x ",bc[z]); else fprintf(stderr,"(null)");
                        fprintf(stderr,"| aslice=0x%x cdma=0x%x\n", aslice[ks], cdma); fflush(stderr); }
                    memset(rc, 0, sizeof rc);
                    orki_i4_synth(rc, 1, Kp, Nc, aslice[ks], bdma, cdma);
                    if (orki_validate_regcmd("ork_dyn_mc_i4", c, rc, REGCMD_I4_N, w, NULL, 0)) { free(h); return NULL; }
                    if (pp < Pcore - 1) { uint64_t nx = RC->dma + (size_t)(pp+1) * REGCMD_I4_N * 4;
                        rc[216] = 0x0010 | ((nx & 0xffff) << 16); rc[217] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
                        rc[218] = 0x0014 | (0x0037u << 16);       rc[219] = (0x0101 << 16); }
                    memcpy((char*)RC->cpu + (size_t)pp * REGCMD_I4_N * 4, rc, REGCMD_I4_N * 4);
                    struct rknpu_task tt; memset(&tt, 0, sizeof tt); tt.enable_mask = 0xd; tt.int_mask = 0x300;
                    tt.int_clear = 0x1ffff; tt.regcfg_amount = 116; tt.regcmd_addr = RC->dma + (size_t)pp * REGCMD_I4_N * 4;   /* int4 = 116 regs */
                    tk[pp] = tt; pp++;
                }
            }
            int gi = p;
            h->outbuf[gi] = CC; h->outptr[gi] = (int32_t*)((char*)CC->cpu + coff); h->dst[gi] = (int32_t*)t->C;
            h->nout[gi] = Sk * N; h->oM[gi] = 1; h->oSk[gi] = Sk;   /* Sk int16 partial blocks of [N]; end() sums -> int32 C */
            coff += (size_t)Sk * N * 2;
        }
        memset(&subs[i], 0, sizeof subs[i]);
        subs[i].flags = ork_ppflags() | 0x2u; subs[i].task_number = pp; subs[i].task_obj_addr = c->mtk[i].obj;
        subs[i].core_mask = 1u << i; subs[i].fence_fd = -1;
        subs[i].subcore_task[0] = subs[i].subcore_task[1] = subs[i].subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)pp};
    }
    /* seed the FULL int16 output surface (clean-before-write for fresh/reused scratch; = the probe's fix) */
    for (int x = 0; x < S; x++) { int N = h->nout[x]; volatile int16_t *o = (volatile int16_t*)h->outptr[x];
        for (int col = 0; col < N; col++){ o[col] = ORK_DYN_SENT16; __asm__ volatile("dc cvac,%0"::"r"(&o[col]):"memory"); } }
    __asm__ volatile("dsb ish":::"memory");
    for (int i = 0; i < nc; i++) if (Pc[i]) {
        orki_bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd, &c->mrc[i], RKNPU_MEM_SYNC_TO_DEVICE);
        orki_bsync(fd, &c->mtk[i], RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
        subs[i].timeout = orki_i4_submit_tmo_ms(); orki_rknpu_submit_ioctl(fd, &subs[i], dom); }   /* #54 bounded (int4 doorbell): a dropped submit must be PAST its timeout by the poll window so ork_dyn_end's recover resubmit reaps it via rknpu_job_timeout_clean. With the 8s mm_timeout_ms a dom-0 drop's stuck job stayed unreaped -> iommu_domain_refcount>0 -> the switch to dom 1 TIMED OUT at scale (the 35B wedge; the small probe never dropped). */
    for (int i = 0; i < nc; i++) c->mwarm[i] = 1;
    /* TASK #4: stash context so ork_dyn_end recovers a dropped int4 round (same ~1/2000 doorbell-drop; the
     * esz==2 branch of orki_mc_recover_resubmit re-seeds the full int16 surface). */
    h->mc_nc = nc; h->mc_dt = DT_I4; h->mc_dom = dom;
    for (int i = 0; i < nc && i < ORK_MAXCORE; i++) { h->mc_subs[i] = subs[i]; h->mc_Pc[i] = Pc[i]; }
    return h;   /* async: end() drains via the esz==2 full-surface int16 poll, then widens int16->int32 into C */
}

ork_dyn_chain *ork_i4_dyn_begin_mc_grouped(ork_npu *c, int M, ork_w *w, const int8_t *A,
                                                  const float *aScale, const float *bScale, float *Cf, int nc) {
    if (!w || w->dtype != DT_I4 || !w->gsize || M < 1 || M > 1024) return NULL;
    int G = w->gsize, K = w->K, N = w->N, Sn = w->Sn, Sk = w->Sk;   /* grouped: Sk = K/G groups */
    (void)K;
    if (Sk > 256) return NULL;                                       /* bounds the per-row aslice[]/program count */
    if (nc < 1 || nc > c->soc->cores) nc = c->soc->cores; if (nc > M) nc = M;
    if (getenv("ORK_GRP_DEBUG")) { fprintf(stderr, "[grp] M=%d K=%d N=%d G=%d Sk=%d Sn=%d nc=%d progs/core~%d\n",
        M, w->K, N, G, Sk, Sn, nc, (M+nc-1)/nc * Sn * Sk); fflush(stderr); }
    if (w->domain != c->dom_active || (w->domain && !c->dom_save)) orki_dom_activate(c, w->domain);
    ork_npu_enter(c, 4 /*DT_I4_CHAIN*/, XP_I4CHAIN, OCK_HW);
    if (orki_mc_ensure(c, nc)) return NULL;
    int fd = c->fd, NMAX = c->soc->nmax;
    ork_dyn_chain *h = calloc(1, sizeof *h); if (!h) return NULL;
    h->c = c; h->S = M; h->P = M; h->N = N; h->dom = w->domain; h->reserve = M; h->mc = 1; h->esz = 2;
    h->i4g = 1; h->i4g_aS = aScale; h->i4g_bS = bScale; h->i4g_Cf = Cf; h->i4g_N = N; h->i4g_Sk = Sk;
    unsigned dom = w->domain;
    for (int i = 0; i < nc; i++) { int lo=(int)((long)i*M/nc), hi=(int)((long)(i+1)*M/nc), P=hi-lo; if (P<1) continue;
        size_t osz = (size_t)P * Sk * N * 2;                         /* rows-on-core x Sk int16 blocks of [N] */
        if (c->mccsz[i] < osz) { orki_bdestroy(fd, &c->mcc[i]); c->mcc[i] = orki_bcreate(fd, osz, 0x403, c->dom_active);
            if (!c->mcc[i].cpu) { free(h); return NULL; } c->mccsz[i] = osz; c->mwarm[i] = 0; } }
    uint32_t rc[REGCMD_I4_N];
    struct rknpu_submit subs[ORK_MAXCORE]; int Pc[ORK_MAXCORE]; memset(Pc, 0, sizeof Pc);
    for (int i = 0; i < nc; i++) { int lo=(int)((long)i*M/nc), hi=(int)((long)(i+1)*M/nc), P=hi-lo; if (P<1) { Pc[i]=0; continue; }
        int Pcore = P * Sn * Sk; Pc[i] = Pcore;
        if ((size_t)Pcore * REGCMD_I4_N * 4 > c->mrc[i].size || (size_t)Pcore * sizeof(struct rknpu_task) > c->mtk[i].size) { free(h); return NULL; }
        struct buf *RC = &c->mrc[i], *AF = &c->maf[i], *CC = &c->mcc[i]; struct rknpu_task *tk = (struct rknpu_task*)c->mtk[i].cpu;
        size_t astage = 0, coff = 0; int pp = 0;
        for (int m = lo; m < hi; m++) { const int8_t *Arow = A + (size_t)m * w->K;
            uint32_t aslice[256];                                    /* this row's Sk group A-slices (each G nibbles) */
            for (int g = 0; g < Sk; g++) { size_t asz = (size_t)G / 2;
                if (astage + asz > AF->size) { free(h); return NULL; }
                orki_i4_tile_Aslice((uint8_t*)AF->cpu + astage, Arow, g * G, G);
                aslice[g] = (uint32_t)(AF->dma + astage); astage += asz; }
            for (int ns = 0; ns < Sn; ns++) { int n0 = ns * NMAX, Nc = (N - n0 < NMAX) ? (N - n0) : NMAX;
                for (int g = 0; g < Sk; g++) {
                    uint32_t cdma = (uint32_t)(CC->dma + coff + (size_t)g * N * 2 + (size_t)n0 * 2);   /* block g, cols [n0,n0+Nc) */
                    uint32_t bdma = (uint32_t)w->Bb[(size_t)ns * Sk + g].dma;
                    memset(rc, 0, sizeof rc);
                    orki_i4_synth(rc, 1, G, Nc, aslice[g], bdma, cdma);
                    if (orki_validate_regcmd("ork_dyn_mc_i4g", c, rc, REGCMD_I4_N, w, NULL, 0)) { free(h); return NULL; }
                    if (pp < Pcore - 1) { uint64_t nx = RC->dma + (size_t)(pp+1) * REGCMD_I4_N * 4;
                        rc[216] = 0x0010 | ((nx & 0xffff) << 16); rc[217] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
                        rc[218] = 0x0014 | (0x0037u << 16);       rc[219] = (0x0101 << 16); }
                    memcpy((char*)RC->cpu + (size_t)pp * REGCMD_I4_N * 4, rc, REGCMD_I4_N * 4);
                    struct rknpu_task tt; memset(&tt, 0, sizeof tt); tt.enable_mask = 0xd; tt.int_mask = 0x300;
                    tt.int_clear = 0x1ffff; tt.regcfg_amount = 116; tt.regcmd_addr = RC->dma + (size_t)pp * REGCMD_I4_N * 4;
                    tk[pp] = tt; pp++;
                } }
            int gi = m; h->outbuf[gi] = CC; h->outptr[gi] = (int32_t*)((char*)CC->cpu + coff); h->dst[gi] = NULL;
            h->nout[gi] = Sk * N; h->oM[gi] = 1;                     /* Sk int16 partial blocks; grouped_end float-accumulates */
            coff += (size_t)Sk * N * 2;
        }
        memset(&subs[i], 0, sizeof subs[i]);
        subs[i].flags = ork_ppflags() | 0x2u; subs[i].task_number = pp; subs[i].task_obj_addr = c->mtk[i].obj; subs[i].core_mask = 1u << i; subs[i].fence_fd = -1;
        subs[i].subcore_task[0] = subs[i].subcore_task[1] = subs[i].subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)pp};
    }
    for (int x = 0; x < M; x++) { int no = h->nout[x]; volatile int16_t *o = (volatile int16_t*)h->outptr[x];
        for (int e = 0; e < no; e++) { o[e] = ORK_DYN_SENT16; __asm__ volatile("dc cvac,%0"::"r"(&o[e]):"memory"); } }
    __asm__ volatile("dsb ish":::"memory");
    for (int i = 0; i < nc; i++) if (Pc[i]) {
        orki_bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd, &c->mrc[i], RKNPU_MEM_SYNC_TO_DEVICE);
        orki_bsync(fd, &c->mtk[i], RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
        subs[i].timeout = orki_i4_submit_tmo_ms(); orki_rknpu_submit_ioctl(fd, &subs[i], dom); }   /* #54 bounded (int4 doorbell): a dropped submit must be PAST its timeout by the poll window so ork_dyn_end's recover resubmit reaps it via rknpu_job_timeout_clean. With the 8s mm_timeout_ms a dom-0 drop's stuck job stayed unreaped -> iommu_domain_refcount>0 -> the switch to dom 1 TIMED OUT at scale (the 35B wedge; the small probe never dropped). */
    for (int i = 0; i < nc; i++) c->mwarm[i] = 1;
    ork_install_term();
    return h;
}
