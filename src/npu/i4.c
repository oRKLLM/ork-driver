/* npu/i4.c — the int4 (W4A4) execution paths: the chained/streamed/batched submit families.
 *
 * ork_mm_run_chain_i4 and ork_dyn_i4_probe (PC-chained int4), the bank-width-tiled BCHAIN doorbell path
 * (bch_db_* + run_i4_bchain_db — ORK_I4_BCHAIN, bit-exact multi-M prefill at production K that sidesteps
 * the 0x1040 schedule wall), the MoE expert-coalesced variant (bch_mw_worker + run_i4_experts_bchain_db),
 * and the async round-robin int4 stream (stream_worker_i4 + ork_mm_run_stream_i4).
 *
 * Lifted verbatim out of npu.c by the precision split (MODULARIZE_PLAN.md round 1). The int4 PACK/quant
 * side (quant_chan_i4, the NF4 codebook, i4a8 persist, tile_i4_*) is still in the scaffold: it interleaves
 * with the int8 pack code, which is layer-organized, and comes out in a later sweep. */
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

static const float ORK_IM_CLIP_GRID[] = { 1.0f, 0.92f, 0.85f, 0.78f, 0.70f, 0.62f, 0.55f };
#define ORK_IM_CLIP_N ((int)(sizeof(ORK_IM_CLIP_GRID)/sizeof(ORK_IM_CLIP_GRID[0])))
#include "regcmd_i4.h"

int ork_mm_run_chain_i4(ork_npu *c, int S, const ork_mm_task_i4 *tasks) {
    if (!c) return -1;
    if (S < 1 || S > 1024) return -2;
    if (!tasks) return -2;

    /* Single matmul: use the optimized run_i4 path (multi-core column-split) rather than the
     * single-core chain path. Chaining only pays off when batching S>1 independent matmuls. */
    if (S == 1) return ork_mm_run_i4(c, tasks[0].w, tasks[0].M, tasks[0].A, tasks[0].C);

    /* chained weights share one submit => one domain; swap in that domain's scratch */
    if (tasks[0].w && (tasks[0].w->domain != c->dom_active || (tasks[0].w->domain!=0 && !c->dom_save))) orki_dom_activate(c, tasks[0].w->domain);
    int fd = c->fd;
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        if (!w || w->dtype != DT_I4) return -2;
        if (tasks[i].M != 1) return -2;
        if (w->Sn != 1 || w->Sk != 1) return -2;
        if (orki_check_overlap("ork_mm_run_chain_i4", (uintptr_t)tasks[i].A, (uintptr_t)tasks[i].A + (size_t)tasks[i].M * w->K, (uintptr_t)tasks[i].C, (uintptr_t)tasks[i].C + (size_t)tasks[i].M * w->N * 4)) return -1;
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
        orki_tile_i4_Aslice(A_dst, tasks[i].A, 0, w->K);
        act_dma[i] = (uint32_t)(chain_A.dma + (size_t)i * max_K);
        out_dma[i] = (uint32_t)(chain_C.dma + (size_t)i * max_N * 2);
    }
    orki_bsync(fd, &chain_A, RKNPU_MEM_SYNC_TO_DEVICE);

    /* Clean-before-write the int16 output scratch. chain_C is bcreate'd fresh each call and the kernel can
     * hand back a recycled DMA region carrying dirty CPU cache lines (from a prior occupant). Those lines
     * evict to DRAM AFTER the NPU writes chain_C, clobbering the NPU output -> "correct run 0, garbage runs
     * 1+" on warm reuse. Dirty the whole surface then clean it to DRAM (TO_DEVICE) so no stale line survives
     * to evict later -- same full-surface clean-before as ork_dyn_begin_mc_i4's doorbell scratch. */
    memset(chain_C.cpu, 0, (size_t)S * max_N * 2);
    orki_bsync(fd, &chain_C, RKNPU_MEM_SYNC_TO_DEVICE);

    struct buf extra[2] = {chain_A, chain_C};
    uint32_t rc[REGCMD_I4_N];

    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        orki_synth_i4(rc, 1, w->K, w->N, act_dma[i], (uint32_t)w->Bb[0].dma, out_dma[i]);
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

/* EXPERIMENTAL int4 NONBLOCK-doorbell probe — byte-for-byte the ork_mm_run_chain_i4 build (M=1 int4 PC-chain,
 * host A staged via tile_i4_Aslice, int16 output scratch), with EXACTLY TWO deltas vs the working reference:
 *   (1) submit flags get NONBLOCK (|0x2u) — the ioctl returns immediately instead of blocking to completion;
 *   (2) completion is detected by polling an int16 output-SENTINEL (0x7fff seeded into each op's last int16
 *       column, dc cvac'd) rather than the blocking ioctl's implicit done.
 * Everything else — synth_i4, the rc[216..219] chain descriptor, regcfg_amount=116, task_number=S, the int16
 * ->int32 de-tile — is identical. This isolates the ONE question the coordinator posed: does the int4 int16-
 * output datapath survive the doorbell's non-blocking sentinel poll (or does int16 output + async race it)? */
#define ORK_I4_SENT16 ((int16_t)0x7fff)
int ork_dyn_i4_probe(ork_npu *c, int S, const ork_mm_task_i4 *tasks) {
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
        orki_tile_i4_Aslice((uint8_t*)chain_A.cpu + (size_t)i * max_K, tasks[i].A, 0, w->K);
        act_dma[i] = (uint32_t)(chain_A.dma + (size_t)i * max_K);
        out_dma[i] = (uint32_t)(chain_C.dma + (size_t)i * max_N * 2);
    }
    orki_bsync(fd, &chain_A, RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf extra[2] = {chain_A, chain_C};
    uint32_t rc[REGCMD_I4_N];
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        orki_synth_i4(rc, 1, w->K, w->N, act_dma[i], (uint32_t)w->Bb[0].dma, out_dma[i]);
        if (orki_validate_regcmd("ork_dyn_i4_probe", c, rc, REGCMD_I4_N, w, extra, 2)) { ok = -1; goto cleanup; }
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

/* #52 (B): BCHAIN batch-chain on the NONBLOCK doorbell (self-healing) — the int4 M>1 prefill path.
 * Reuses BCHAIN's proven H-row batch orki_synth (synth_i4 mc=2*Hg) + bank-width Wb=131072/K N-tiling + de-tile
 * (og[(4*j+4*Hg*b)*64+cc] -> C[g*H+j][n0+b*64+cc]) VERBATIM, but submits NONBLOCK (flags|0x2) and detects
 * completion by civac-polling the WRITTEN (strided) int16 cell-set for the SENT16 sentinel — the batch output
 * is a 2D-tiled strided subset (write-order not last-col-last, like the per-row int4 path), so we seed+poll
 * exactly the written cells and never the padding. A dropped round self-heals via RESET+re-seed+resubmit
 * (orki_mc_recover_resubmit model), so a miss never hard-wedges (unlike the removed blocking BCHAIN). Single host
 * thread submits all cores nonblock then polls them. -4 = ineligible (caller falls back); -1 = unrecovered. */
/* mode 0=seed SENT16 (values only; caller bsyncs TO_DEVICE), 1=civac gate (invalidate per 64B line then check),
 * 3=plain full-verify (after bsync FROM_DEVICE, no per-cell DC), 2=de-tile int16->int32 into C. Cache ops are
 * PER-CACHE-LINE (32 int16/line), never per-element — per-element dc over the M*N surface was a ~60ms host wall. */
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
static int g_i4_validate=-1;   /* ORK_I4_VALIDATE: per-program regcmd validation (DEBUG, off by default) */
/* #54 TCLEAN precondition (mirror fp16 recov_tmo, npu.c ~11238, task #50): the int4 doorbell NONBLOCK submit
 * timeout MUST be < the poll-detect window (ORK_I4_POLL_MS). rknpu_job_timeout_clean — which runs at the top of
 * every nonblock submit and is the ONLY thing that cleanly reaps a DROPPED job (ACT_RESET can't; source-confirmed
 * rknpu_soft_reset is HW-only) — reaps a job only once it is aged >= its own submit timeout. So a dropped job
 * submitted with an 8s timeout is NOT reapable at the 2s resubmit -> it lingers as a stuck job -> the next
 * iommu-domain switch times out -> cascade/wedge. Bounding the timeout below the poll window makes every resubmit
 * (in-worker or post-join) reap the prior drop. A REAL job lands via its completion IRQ well inside the poll
 * window and is gone before any reap check, so bounding never false-reaps a completing job. Default 3/4 of the
 * poll window; ORK_I4_SUBMIT_TMO_MS overrides. */
int orki_i4_submit_tmo_ms(void){
    static int t=-1;
    if(t<0){ const char*e=getenv("ORK_I4_SUBMIT_TMO_MS");
        if(e) t=atoi(e);
        else { const char*p=getenv("ORK_I4_POLL_MS"); double pm=p?atof(p):2000.0; t=(int)(pm*0.75); }
        if(t<10) t=10; }
    return t;
}
struct bchdbw { ork_npu *c; int core, c0, c1, NT, K, N, NG, M, H, Wb, Wmax; ork_w *w; const int8_t *A; int32_t *C; unsigned dom; struct rknpu_submit sub; int rc; };
/* One NPU core's share of the BCHAIN batch-chain: build its (N-chunk x M-group) programs into its pre-allocated
 * per-core buffers, seed, NONBLOCK submit, poll ITS core (last-program civac gate -> bsync -> full verify), then
 * de-tile ITS core. Runs on the npu_pool (parallel build+de-tile across cores; de-tile overlaps the sibling
 * cores' still-running compute). NO reset here — a global RKNPU_ACT_RESET would corrupt live sibling cores; a
 * completion miss returns rc=-2 and the caller recovers serially after join. rc: 0 ok / -1 build-err / -2 miss. */
static void *bch_db_worker(void *vp){
    struct bchdbw *a=vp; ork_npu *c=a->c; int fd=c->fd, i=a->core, NT=a->NT;
    int K=a->K,N=a->N,NG=a->NG,M=a->M,H=a->H,Wb=a->Wb,Wmax=a->Wmax; unsigned dom=a->dom;
    a->rc=0; if(NT<1) return NULL;
    orki_pin_big_core(i);
    uint8_t *abase=c->maf[i].cpu; memset(abase,0,(size_t)NG*(size_t)(2*H)*(K/2));   /* A packed once per M-group (stride-2) */
    for(int g=0;g<NG;g++){ int Hg=(M-g*H<H)?(M-g*H):H;
        for(int j=0;j<Hg;j++) orki_tile_i4_Aslice(abase+(size_t)(g*2*H+2*j)*(K/2), a->A+(size_t)(g*H+j)*K, 0, K); }
    orki_bsync(fd,&c->maf[i],RKNPU_MEM_SYNC_TO_DEVICE);
    int tk=0;
    for(int nc2=a->c0;nc2<a->c1;nc2++){ int n0=nc2*Wb, Wc=(N-n0<Wb)?(N-n0):Wb;
        uint32_t wdma=(uint32_t)(a->w->Bb[0].dma + (uint64_t)(n0/64)*K*32);
        for(int g=0;g<NG;g++){ int Hg=(M-g*H<H)?(M-g*H):H; uint32_t rc[REGCMD_I4_N];
            uint32_t aA=(uint32_t)c->maf[i].dma+(uint32_t)(g*2*H)*(K/2);
            uint32_t aC=(uint32_t)c->mcc[i].dma+(uint32_t)tk*(4*H*Wmax)*64*2;
            memset(rc,0,sizeof rc); orki_synth_i4(rc, 2*Hg, K, Wc, aA, wdma, aC);
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
            if(g_i4_validate && orki_validate_regcmd("bch_db_worker", c, rc, REGCMD_I4_N, a->w, NULL, 0)){ a->rc=-1; c->mc_error=1; return NULL; }   /* per-program validate is a DEBUG check (ORK_I4_VALIDATE); off by default — it scales with program count and blk never had it */
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
     *   1 NONBLOCK (ORK_I4_NB): seed + nonblock doorbell + ork_dyn_end polls from t=0 (pipelined; the expensive
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
int orki_run_i4_bchain_db(ork_npu *c, ork_w *w, int M, const int8_t *A, int32_t *C, int nc){
    if(w->dtype!=DT_I4 || w->Sk!=1 || w->Sn!=1 || (w->N%64) || M<2) return -4;
    int fd=c->fd, K=w->K, N=w->N;
    /* accurate wedge telemetry: the BCHAIN worker skips validate_regcmd by default, so orki_last_op would
     * otherwise stay stale (mislabelling BCHAIN submits as the last mc_i4 op in ORK_PRESUBMIT_TRACE). */
    orki_last_op="run_i4_bchain_db"; orki_last_K=K; orki_last_N=N; orki_last_wdom=w->domain;
    orki_last_import=(w->own_buf_valid && w->own_buf.heap_fd>0) || (w->own_bufs && w->n_own_bufs>0 && w->own_bufs[0].heap_fd>0)
                  || (w->Bb && w->Bb[0].heap_fd>0);
    int H=16384/K; if(H>16)H=16; if(H<2) return -4;
    int Wb=(131072/K)&~63; if(Wb<64)Wb=64; if(Wb>N)Wb=N;
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
    if(g_i4_validate<0) g_i4_validate=getenv("ORK_I4_VALIDATE")?1:0;   /* init once on the calling thread (before dispatch) */
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
/* #54 COALESCED multi-expert BCHAIN. Each core handles a contiguous range of experts (whole N per expert) and
 * chains ALL its experts' M-batched BCHAIN programs into ONE nonblock doorbell submit — same fast programs as
 * run_i4_bchain_db, but coalesced across experts (per-tensor, not per-expert; ~nc submits instead of nc x
 * n_experts). Per-expert A staging into maf, cumulative program index into the shared chain, per-expert de-tile
 * via bch_db_cells_off. M>=2 only (decode/M=1 is CPU). Drop -> serial per-expert BCHAIN fallback (correctness). */
struct bchmw { ork_npu *c; int core, e0, e1; const ork_mm_task_i4 *ex; int K,N,H,Wb,Wmax,NC; unsigned dom; struct rknpu_submit sub; int rc; };
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
            for(int j=0;j<Hg;j++) orki_tile_i4_Aslice((uint8_t*)c->maf[i].cpu+aoff+(size_t)(g*2*H+2*j)*(K/2), t->A+(size_t)(g*H+j)*K, 0, K); }
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
                memset(rc,0,sizeof rc); orki_synth_i4(rc, 2*Hg, K, Wc, aA, wdma, aC);
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
int orki_run_i4_experts_bchain_db(ork_npu *c, const ork_mm_task_i4 *ex, int ntask, int nc){
    int fd=c->fd, K=ex[0].w->K, N=ex[0].w->N;
    int H=16384/K; if(H>16)H=16; if(H<2) return -4;
    int Wb=(131072/K)&~63; if(Wb<64)Wb=64; if(Wb>N)Wb=N;
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
            for(int e=args[i].e0;e<args[i].e1;e++) if(orki_run_i4_bchain_db(c, ex[e].w, ex[e].M, ex[e].A, ex[e].C, nc)) return -1; }
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
struct streamw4 { ork_npu *c; int core; int S; const ork_mm_task_i4 *tasks; int *ctr; int rc; };
static void *stream_worker_i4(void *vp) {
    struct streamw4 *a = vp; ork_npu *c = a->c; int fd = c->fd, i = a->core;
    orki_pin_big_core(i);
    int k; a->rc = 0;
    uint32_t rc[REGCMD_I4_N];
    while ((k = __atomic_fetch_add(a->ctr, 1, __ATOMIC_SEQ_CST)) < a->S) {
        const ork_mm_task_i4 *t = &a->tasks[k];
        ork_w *w = t->w; int M = t->M, K = w->K, N = w->N;
        uint32_t bdma = (uint32_t)w->Bb[0].dma;
        uint8_t *abase = c->maf[i].cpu;                       /* per-row stride K bytes (nibble-pack uses K/2) */
        /* ROUND-ROBIN + BATCH: if the native batch scheduler is on and this task's weight fits resident
         * (N*K<=131072), batch its M rows in H-row submits (mc=2H, stride-2 A at slot 2j, de-tile 4j+4H*b) —
         * one core batches a whole small matmul, round-robin, no barrier. Else fall through to per-row. */
        int Hcap = 16384 / K; if (Hcap > 16) Hcap = 16; if (Hcap < 1) Hcap = 1;
        if (ork_i4_batch() && Hcap >= 2 && M >= 2 && (size_t)N * K <= 131072) {
            int NBc = N / 64;
            for (int m0 = 0; m0 < M; m0 += Hcap) {
                int H = (M - m0 < Hcap) ? (M - m0) : Hcap;
                for (int j = 0; j < H; j++)                   /* real row j at A-slot 2j (stride-2 input) */
                    orki_tile_i4_Aslice(abase + (size_t)(2 * j) * (K / 2), t->A + (size_t)(m0 + j) * K, 0, K);
                orki_bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE);
                memset(rc, 0, sizeof rc);
                orki_synth_i4(rc, 2 * H, K, N, (uint32_t)c->maf[i].dma, bdma, (uint32_t)c->mcc[i].dma);
                rc[216] = 0; rc[217] = 0; rc[218] = 0x00000014; rc[219] = 0x01010000;   /* single task */
                memcpy(c->mrc[i].cpu, rc, REGCMD_I4_N * 4);
                orki_bsync(fd, &c->mrc[i], RKNPU_MEM_SYNC_TO_DEVICE);
                struct rknpu_task *mt = c->mtk[i].cpu; memset(mt, 0, sizeof *mt);
                mt[0].enable_mask = 0xd; mt[0].int_mask = 0x300; mt[0].int_clear = 0x1ffff;
                mt[0].regcfg_amount = 116; mt[0].regcmd_addr = c->mrc[i].dma;
                orki_bsync(fd, &c->mtk[i], RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
                struct rknpu_submit sub; memset(&sub, 0, sizeof sub);
                sub.flags = ork_ppflags(); sub.task_number = 1; sub.task_obj_addr = c->mtk[i].obj; sub.core_mask = 1u << i; sub.fence_fd = -1;
                sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, 1};
                int reps = c->mwarm[i] ? 1 : 2;
                for (int rep = 0; rep < reps; rep++) { int last = (rep == reps - 1); sub.timeout = orki_mm_timeout_ms();
                    if (orki_rknpu_submit_ioctl(fd, &sub, w->domain)) { if (last) a->rc = -1; continue; }
                    orki_bsync(fd, &c->mcc[i], RKNPU_MEM_SYNC_FROM_DEVICE); }
                c->mwarm[i] = 1;
                int16_t *o = c->mcc[i].cpu; int32_t *C = t->C;   /* de-tile: o[(4j+4H*b)*64+cc] -> C[m0+j][b*64+cc] */
                for (int j = 0; j < H; j++) for (int b = 0; b < NBc; b++) {
                    size_t base = (size_t)(4 * j + 4 * H * b) * 64;
                    int32_t *crow = C + (size_t)(m0 + j) * N + b * 64;
                    for (int cc = 0; cc < 64; cc++) crow[cc] = o[base + cc];
                }
            }
            continue;   /* task done via batch; skip the per-row path below */
        }
        for (int m = 0; m < M; m++) orki_tile_i4_Aslice(abase + (size_t)m * K, t->A + (size_t)m * K, 0, K);
        orki_bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE);
        for (int m = 0; m < M; m++) {                         /* one single-row regcmd per row, PC-chained */
            memset(rc, 0, sizeof rc);
            orki_synth_i4(rc, 1, K, N, (uint32_t)(c->maf[i].dma + (size_t)m * K), bdma,
                     (uint32_t)(c->mcc[i].dma + (size_t)m * N * 2));
            if (m < M - 1) {
                uint64_t nd = c->mrc[i].dma + (size_t)(m + 1) * REGCMD_I4_N * 4;
                rc[216] = 0x0010 | ((nd & 0xffff) << 16); rc[217] = (0x0101 << 16) | ((nd >> 16) & 0xffff);
                rc[218] = 0x0014 | (0x0037 << 16); rc[219] = (0x0101 << 16) | (0);
            } else { rc[216] = 0; rc[217] = 0; rc[218] = 0x00000014; rc[219] = 0x01010000; }
            memcpy((char *)c->mrc[i].cpu + (size_t)m * REGCMD_I4_N * 4, rc, REGCMD_I4_N * 4);
        }
        orki_bsync(fd, &c->mrc[i], RKNPU_MEM_SYNC_TO_DEVICE);
        struct rknpu_task *mt = c->mtk[i].cpu; memset(mt, 0, (size_t)M * sizeof *mt);
        for (int q = 0; q < M; q++) {
            mt[q].enable_mask = 0xd; mt[q].int_mask = 0x300; mt[q].int_clear = 0x1ffff;
            mt[q].regcfg_amount = 116; mt[q].regcmd_addr = c->mrc[i].dma + (size_t)q * REGCMD_I4_N * 4;
        }
        orki_bsync(fd, &c->mtk[i], RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
        struct rknpu_submit sub; memset(&sub, 0, sizeof sub);
        sub.flags = ork_ppflags(); sub.task_number = M; sub.task_obj_addr = c->mtk[i].obj; sub.core_mask = 1u << i; sub.fence_fd = -1;
        sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)M};
        /* Prime THIS core's buffers on its first use (mwarm[i]): a freshly-allocated NPU output buffer
         * returns stale on its first write, so the first task to land on a core does a throwaway warmup
         * rep then the real rep. Per-core (not an outer double-pass) because tasks are pulled round-robin
         * — a core that idles in pass 0 would otherwise stay unprimed and zero-fill the next task it grabs.
         * Same idiom as the run_multicore / chain workers. */
        int reps = c->mwarm[i] ? 1 : 2;
        for (int rep = 0; rep < reps; rep++) {
            int last = (rep == reps - 1);
            sub.timeout = orki_mm_timeout_ms();
            if (orki_rknpu_submit_ioctl(fd, &sub, w->domain)) { if (last) a->rc = -1; continue; }
            orki_bsync(fd, &c->mcc[i], RKNPU_MEM_SYNC_FROM_DEVICE);
        }
        c->mwarm[i] = 1;
        int16_t *o = c->mcc[i].cpu; int32_t *C = t->C;        /* widen int16 NPU output -> int32 caller C */
        for (int row = 0; row < M; row++) {
            int16_t *orow = o + (size_t)row * N; int32_t *crow = C + (size_t)row * N;
            for (int col = 0; col < N; col++) crow[col] = orow[col];
        }
    }
    return NULL;
}
int ork_mm_run_stream_i4(ork_npu *c, int S, const ork_mm_task_i4 *tasks) {
    if (!c || S < 1 || !tasks) return -2;
    /* per-core scratch lives in the active domain; stream tasks share one domain (tasks[0].w) */
    if (tasks[0].w && (tasks[0].w->domain != c->dom_active || (tasks[0].w->domain!=0 && !c->dom_save))) orki_dom_activate(c, tasks[0].w->domain);
    const int mrc_cap = 65536 / (REGCMD_I4_N * 4);
    size_t maxMK = 0, maxMN2 = 0;
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        if (!w || w->dtype != DT_I4 || tasks[i].M <= 0) return -2;
        if (w->Sn != 1 || w->Sk != 1) return -2;              /* single-slice weight only (no K/N split) */
        if (tasks[i].M > mrc_cap) return -2;                  /* M single-row regcmds must fit one mrc buffer */
        size_t mk = (size_t)tasks[i].M * w->K, mn = (size_t)tasks[i].M * w->N * 2;
        /* batch (msched) output spans up to 4*HCAP(=16)*N int16 = 128*N bytes; only for tasks that FIT the
         * weight orki_budget (N*K<=131072, i.e. small N), so the bump is small. */
        if(ork_i4_batch() && (size_t)w->N*w->K<=131072){ size_t bn=(size_t)128*w->N; if(bn>mn)mn=bn; }
        if (mk > maxMK) maxMK = mk; if (mn > maxMN2) maxMN2 = mn;
    }
    int fd = c->fd;
    int cold = 0;   /* warmup pass needed on a fresh stream-i4 mode OR freshly-allocated per-core buffer */
    if (ork_npu_enter(c, 5 /* DT_I4_STREAM */, XP_I4_STREAM, OCK_SW)) cold = 1;
    int nc = orki_budget(c, 2); if (nc > ORK_MAXCORE) nc = ORK_MAXCORE; if (nc > S) nc = S; if (nc < 1) nc = 1;
    if (orki_mc_ensure(c, nc)) return -1;
    for (int i = 0; i < nc; i++) {
        if (c->maf[i].size < maxMK) { orki_bdestroy(fd, &c->maf[i]); c->maf[i] = orki_bcreate(fd, maxMK, 0x403, c->dom_active); if (!c->maf[i].cpu) return -1; cold = 1; }
        if (c->mccsz[i] < maxMN2) { orki_bdestroy(fd, &c->mcc[i]); c->mcc[i] = orki_bcreate(fd, maxMN2, 0x403, c->dom_active); c->mccsz[i] = maxMN2; if (!c->mcc[i].cpu) return -1; cold = 1; }
    }
    int rc = 0;
    if (cold) for (int i = 0; i < nc; i++) c->mwarm[i] = 0;   /* fresh mode/buffer => each core re-primes (per-core warmup in the worker) */
    orki_npu_pool_ensure(c);
    struct streamw4 sw[ORK_MAXCORE];
    /* Single dispatch: priming is per-core inside the worker (mwarm[i]), so the result is correct
     * regardless of how the round-robin atomic counter assigns tasks to cores — no outer double-pass
     * (which left a core that idled in the first pass unprimed, zero-filling whatever task it then grabbed). */
    int ctr = 0;
    for (int i = 0; i < nc; i++) sw[i] = (struct streamw4){c, i, S, tasks, &ctr, 0};
    pthread_mutex_lock(&c->pmu);
    c->pjob = sw; c->pjob_nc = nc; c->pjob_fn = stream_worker_i4; c->pjob_stride = sizeof(struct streamw4);
    c->pdone = 0; c->pgen++; pthread_cond_broadcast(&c->pgo);
    pthread_mutex_unlock(&c->pmu);
    stream_worker_i4(&sw[0]);                             /* core 0 on the calling thread */
    pthread_mutex_lock(&c->pmu); while (c->pdone < nc - 1) pthread_cond_wait(&c->pdn, &c->pmu); pthread_mutex_unlock(&c->pmu);
    for (int i = 0; i < nc; i++) if (sw[i].rc) rc = -1;
    c->warmed = 1;
    return rc;
}

/* (continued) int4 pack/quant — lifted in the second i4 pass, once the int8 pack code it interleaved
 * with had moved to src/npu/i8/pack.c. */

int ork_i4_batch(void){ static int v=-1; if(v<0){const char*e=getenv("ORK_I4_MSCHED"); v=e?(atoi(e)?1:0):1;} return v; }

static int ork_i4_nsub(void){ static int v=-1; if(v<0){const char*e=getenv("ORK_I4_NSUB"); v=(e&&atoi(e))?1:0;} return v; }  /* default OFF: under pinned-DDR benchmarking N-subslice is ~neutral (submit overhead cancels the weight-DMA saving); the "1.1-1.2x" seen earlier was an unpinned-governor artifact. Kept opt-in. */

void ork_i4_fuzz_clear(void){ orki_i4_fovr_n=0; }

void ork_i4_fuzz_add(uint32_t blk,uint32_t reg,uint32_t val){ if(orki_i4_fovr_n<16){ orki_i4_fovr[orki_i4_fovr_n].blk=blk; orki_i4_fovr[orki_i4_fovr_n].reg=reg; orki_i4_fovr[orki_i4_fovr_n].val=val; orki_i4_fovr_n++; } }

void orki_synth_i4(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC){
    memcpy(rc,REGCMD_I4,REGCMD_I4_N*4);
    orki_setrn(rc,REGCMD_I4_N,RK_CNA_DATA_SIZE1,((K-1)<<16)|K);       /* K range (element count) */
    orki_setrn(rc,REGCMD_I4_N,RK_CNA_WEIGHT_SIZE0,(K*N)/2);             /* weight bytes: int4 = 0.5 B/elem */
    orki_setrn(rc,REGCMD_I4_N,RK_CNA_WEIGHT_SIZE1,K/2);                 /* weight row bytes */
    orki_setrn(rc,REGCMD_I4_N,RK_CNA_CBUF_CON1,(K+127)/128);        /* K-passes: ceil(K/128) (captured scaling) */
    orki_setrn(rc,REGCMD_I4_N,RK_CNA_FC_DATA_SIZE1,K);
    orki_setrn(rc,REGCMD_I4_N,RK_CNA_WEIGHT_SIZE2,0x1010000|N);orki_setrn(rc,REGCMD_I4_N,RK_PDP_OUT_N,N-1);
    /* N-output-stride regs, parameterized for wide-N single-submit (verified vs N=64 & N=128
     * captures: 0x403c=(N-1)dup, 0x4058=N-1, 0x3018=N-1 above). 0x40c0/0x4050 are CONSTANT across N
     * (0x80/0x7fe — left at REGCMD_I4). */
    orki_setrn(rc,REGCMD_I4_N,RK_DPU_DST_N_DIMS,((N-1)<<16)|(N-1));
    orki_setrn(rc,REGCMD_I4_N,RK_DPU_DST_N2,N-1);
    /* Multi-M scheduler (mc>1) — native batch mode (NVDLA D_BATCH_NUMBER analog): one submit computes H
     * rows with the weight streamed ONCE, output at stride-2 (logical row m -> physical row 2m; int16 result
     * in an int32-stepped DMA). Reached only via ORK_I4_MSCHED (gated OFF; production callers pass mc=1 ->
     * the proven per-row PC-chain in i4_mcworker). Exp-2026-07-07 (tools/i4_multim_probe.c) established:
     *   - 0x1040 (K-reduction schedule) is DECISIVE, not the earlier-claimed inert set: the int8 FORMULA
     *     for it corrupts int4 (188@K64/72@K2048); leaving it at the captured base 177 works but only for
     *     the capture's K. -> ORK_I4_1040/ORK_I4_1010 overrides + ORK_I4_MREGS bitmask expose it for RE.
     *   - With mregs=0x1f (all int8-ported regs EXCEPT 0x1040) the batch mode is BIT-EXACT at K=64 (the
     *     capture K): all H rows land at stride-2, single 64-wide N-block. Verified M=2,4 @ K=64.
     *   - REMAINING WALL: row count is capture-K-tuned (K=32->1 row, K=64->4, K=128->2, K=256->1, K>=512->1)
     *     and multi-N-block output is a 2D surface (offsets (b*H+m)*2N at N=128), so at production K (2048)
     *     only row 0 computes. Cracking it needs a systematic 0x1040 x K x mc x CNA/CBUF sweep (the batch
     *     activation-cube budget reg) — see the wiki Exp-2026-07-07 log. Until then W4A4 stays 1-submit/row. */
    if(mc>1){
        int mc_phys = 2 * mc;
        /* ORK_I4_MREGS bitmask — which regs unlock native multi-M. Default 0x1f = the full int8-ported set
         * MINUS 0x1040. Fuzzing (Exp-2026-07-07, tools/i4_multim_probe.c) proved 0x1040 (the int8 K-reduction
         * schedule) is the POISON PILL: every config that sets it corrupts row 0 and rows>0; every config
         * without it computes bit-exact stride-2 (logical row m -> physical row 2m). int4's K-reduction uses
         * the captured base 0x1040 (K-independent, unlike int8) — which the M=1 path already never overrides.
         *   0x01 M-count 0x1020/0x1084/0x102c   0x02 0x4034(PPU rows)   0x04 0x3014(DPU)
         *   0x08 0x4038(out width/4)            0x10 0x1010(CNA hint)   0x20 0x1040(K-schedule=POISON) */
        static int mregs=-1; if(mregs<0){const char*e=getenv("ORK_I4_MREGS"); mregs=e?(int)strtoul(e,0,0):0x5f;}
        orki_setrn(rc,REGCMD_I4_N,RK_DPU_WDMA_SIZE_1,0);                                   /* the trigger (always) */
        /* 0x107c = K/16 : the batch activation-cube-size reg (Exp-2026-07-07 fuzz). Captured as 4 (tuned to
         * the M=4/K=64 capture); setting it to K/16 restores the native 4-ROW batch at ANY K — bit-exact at
         * K=512/1024/2048 (0x20/0x40/0x80). This lifts multi-M from 1 row to 4 rows/submit at production K
         * (4x fewer weight streams). Narrow: only K/16 gives 4 (neighbors give 2); 4 is the cap for this reg. */
        if(mregs&0x40) orki_setrn(rc,REGCMD_I4_N,RK_CNA_DMA_CON1,(uint32_t)(K/16));
        if(mregs&0x01){ orki_setrn(rc,REGCMD_I4_N,RK_CNA_DATA_SIZE0,0x10000|mc_phys);orki_setrn(rc,REGCMD_I4_N,RK_CNA_DATA_SIZE0_MIR,0x10000|mc_phys);orki_setrn(rc,REGCMD_I4_N,RK_CNA_DATA_SIZE3,mc_phys); }
        if(mregs&0x02) orki_setrn(rc,REGCMD_I4_N,RK_DPU_DATA_CUBE_HEIGHT,mc_phys-1);
        if(mregs&0x04) orki_setrn(rc,REGCMD_I4_N,RK_PDP_OUT_M,(mc_phys-1)<<16);
        if(mregs&0x08) orki_setrn(rc,REGCMD_I4_N,RK_DPU_DATA_CUBE_NOTCH,(((N/4)-1)<<16)|((N/4)-1));
        if(mregs&0x10) orki_setrn(rc,REGCMD_I4_N,RK_CNA_CONV_CON2,16*(mc_phys+1));
        if(mregs&0x20){ double scale=(double)K/256.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale),mg=mc_phys/64; if(mg<1)mg=1;
            int v=base-slope*(mg-1); if(v<0x1b)v=0x1b; orki_setrn(rc,REGCMD_I4_N,RK_CNA_CBUF_CON0,v); }
        /* ORK_I4_1040: direct override of the K-reduction schedule reg (RE: find the int4 multi-row value —
         * the int8 formula corrupts, omitting it leaves only row0 for K>64). Applied last, wins over mregs&0x20. */
        { const char*e=getenv("ORK_I4_1040"); if(e) orki_setrn(rc,REGCMD_I4_N,RK_CNA_CBUF_CON0,(uint32_t)strtoul(e,0,0)); }
        /* ORK_I4_1010: override CNA row/activation-cube reg (RE: multi-M computes rows_computed*K=256 elems —
         * a fixed activation-cube budget; find the reg that enlarges it so K=2048 gets >1 row). */
        { const char*e=getenv("ORK_I4_1010"); if(e) orki_setrn(rc,REGCMD_I4_N,RK_CNA_CONV_CON2,(uint32_t)strtoul(e,0,0)); }
    }
    orki_setrn(rc,REGCMD_I4_N,RK_CNA_FEATURE_DATA_ADDR,aA);orki_setrn(rc,REGCMD_I4_N,RK_CNA_WEIGHT_DATA_ADDR,aB);orki_setrn(rc,REGCMD_I4_N,RK_DPU_DST_BASE_ADDR,aC);
    for(int i=0;i<orki_i4_fovr_n;i++) orki_setr(rc,REGCMD_I4_N,orki_i4_fovr[i].blk,orki_i4_fovr[i].reg,orki_i4_fovr[i].val);  /* RE fuzzer overrides (win over all) */
}

static void quant_chan_i4(const float *fr, int K, float scale, int sr, uint32_t *seed, uint8_t *nib, float *qf32) {
    float inv = scale > 0 ? 1.0f/scale : 0.0f;
    for (int k = 0; k < K; k++) {
        int q;
        if (sr) { float u = (float)(ork_xs32(seed) >> 8) * (1.0f/16777216.0f);  /* u in [0,1) */
                  q = (int)floorf(fr[k]*inv + u); }
        else      q = (int)lrintf(fr[k]*inv);
        if (q > 7) q = 7; else if (q < -7) q = -7;
        qf32[k] = (float)q;
        uint8_t nb = (uint8_t)(q & 0xf);              /* low nibble holds the signed 4-bit code */
        if (k & 1) nib[k>>1] |= (uint8_t)(nb << 4); else nib[k>>1] = nb;
    }
}

static void expand_chan_i4_f32(const uint8_t *nib, int K, float *qf32) {
    int k = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    uint8x8_t vlo = vdup_n_u8(0x0f);
    for (; k <= K - 16; k += 16) {
        uint8x8_t pk = vld1_u8(nib + (k>>1));                 /* 8 bytes = 16 nibbles */
        int8x8_t even = vreinterpret_s8_u8(vand_u8(pk, vlo)); /* low nibbles  (codes k,k+2,...) */
        int8x8_t odd  = vreinterpret_s8_u8(vshr_n_u8(pk, 4)); /* high nibbles (codes k+1,...) */
        /* sign-extend 4-bit: shift the nibble into the top of an int8, then arithmetic-shift back */
        even = vshr_n_s8(vshl_n_s8(even, 4), 4);
        odd  = vshr_n_s8(vshl_n_s8(odd,  4), 4);
        int8x8x2_t zip = vzip_s8(even, odd);                  /* interleave -> code order */
        int8x16_t codes = vcombine_s8(zip.val[0], zip.val[1]);
        int16x8_t lo16 = vmovl_s8(vget_low_s8(codes));
        int16x8_t hi16 = vmovl_s8(vget_high_s8(codes));
        vst1q_f32(qf32 + k,      vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))));
        vst1q_f32(qf32 + k + 4,  vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))));
        vst1q_f32(qf32 + k + 8,  vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))));
        vst1q_f32(qf32 + k + 12, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))));
    }
#endif
    for (; k < K; k++) {
        uint8_t nb = (k & 1) ? (nib[k>>1] >> 4) : (nib[k>>1] & 0xf);
        int8_t c = (int8_t)(nb << 4) >> 4;                    /* sign-extend 4-bit */
        qf32[k] = (float)c;
    }
}

static void quant_chan_nf4(const float *fr, int K, float absmax, int sr, uint32_t *seed, uint8_t *nib, uint8_t *qidx) {
    float inv = absmax > 0 ? 1.0f/absmax : 0.0f;
    for (int k = 0; k < K; k++) {
        float wn = fr[k]*inv; if (wn > 1.0f) wn = 1.0f; else if (wn < -1.0f) wn = -1.0f;
        /* find bracketing pair: hi = first level >= wn */
        int hi = 0; while (hi < 15 && ORKI_NF4_LEVELS[hi] < wn) hi++;
        int lo = hi > 0 ? hi-1 : 0;
        int idx;
        if (lo == hi) idx = hi;
        else {
            float dlo = ORKI_NF4_LEVELS[hi]-ORKI_NF4_LEVELS[lo];     /* span > 0 */
            float t = (wn - ORKI_NF4_LEVELS[lo]) / dlo;             /* fractional pos in [0,1] toward hi */
            if (sr) { float u = (float)(ork_xs32(seed) >> 8) * (1.0f/16777216.0f); /* u in [0,1) */
                      idx = (t > u) ? hi : lo; }                   /* P(hi) = t */
            else      idx = (t >= 0.5f) ? hi : lo;                 /* nearest level */
        }
        qidx[k] = (uint8_t)idx;
        uint8_t nb = (uint8_t)(idx & 0xf);
        if (k & 1) nib[k>>1] |= (uint8_t)(nb << 4); else nib[k>>1] = nb;
    }
}

static void inflate_chan_nf4_f32(const uint8_t *qidx, int K, const int8_t lut[16], float *qf32) {
    int k = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int8x16_t vlut = vld1q_s8(lut);
    for (; k <= K - 16; k += 16) {
        uint8x16_t vi = vld1q_u8(qidx + k);
        int8x16_t codes = vqtbl1q_s8(vlut, vi);               /* table lookup: code = lut[idx] */
        int16x8_t lo16 = vmovl_s8(vget_low_s8(codes));
        int16x8_t hi16 = vmovl_s8(vget_high_s8(codes));
        vst1q_f32(qf32 + k,      vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))));
        vst1q_f32(qf32 + k + 4,  vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))));
        vst1q_f32(qf32 + k + 8,  vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))));
        vst1q_f32(qf32 + k + 12, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))));
    }
#endif
    for (; k < K; k++) qf32[k] = (float)lut[qidx[k]];
}

void orki_expand_chan_i4_i8(const uint8_t *nib, int K, int8_t *i8) {
    int k = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    uint8x8_t vlo = vdup_n_u8(0x0f);
    for (; k <= K - 16; k += 16) {
        uint8x8_t pk = vld1_u8(nib + (k>>1));                 /* 8 bytes = 16 nibbles */
        int8x8_t even = vreinterpret_s8_u8(vand_u8(pk, vlo)); /* low nibbles  (codes k,k+2,...) */
        int8x8_t odd  = vreinterpret_s8_u8(vshr_n_u8(pk, 4)); /* high nibbles (codes k+1,...) */
        even = vshr_n_s8(vshl_n_s8(even, 4), 4);              /* sign-extend 4-bit */
        odd  = vshr_n_s8(vshl_n_s8(odd,  4), 4);
        int8x8x2_t zip = vzip_s8(even, odd);                  /* interleave -> code order */
        vst1q_s8(i8 + k, vcombine_s8(zip.val[0], zip.val[1]));
    }
#endif
    for (; k < K; k++) {
        uint8_t nb = (k & 1) ? (nib[k>>1] >> 4) : (nib[k>>1] & 0xf);
        i8[k] = (int8_t)(nb << 4) >> 4;                       /* sign-extend 4-bit */
    }
}

static void tile_direct_i4_i8(ork_npu *c, ork_w *w, int K, int N, int kind, int8_t *i8scratch) {
    if (kind == ORK_QK_CODEBOOK_NF4) {
        int8_t lut[16]; for (int i = 0; i < 16; i++) lut[i] = (int8_t)lrintf(ORKI_NF4_LEVELS[i]*127.0f);
        for (int n = 0; n < N; n++) orki_inflate_chan_nf4_i8(w->Bi4 + (size_t)n*(K/2), K, lut, i8scratch + (size_t)n*K);
    } else {
        for (int n = 0; n < N; n++) orki_expand_chan_i4_i8(w->Bi4 + (size_t)n*(K/2), K, i8scratch + (size_t)n*K);
    }
    orki_tile_i8_to_tiles(c, w, K, N, i8scratch);
}

static float wq_err_chan(const float *fr, int K, float absmax, int nf4, const float *im, float *dq) {
    if (nf4) {
        float sc = absmax / 127.0f, inv = absmax > 0 ? 1.0f/absmax : 0.0f;
        for (int k = 0; k < K; k++) {
            float wn = fr[k]*inv; if (wn > 1.0f) wn = 1.0f; else if (wn < -1.0f) wn = -1.0f;
            int hi = 0; while (hi < 15 && ORKI_NF4_LEVELS[hi] < wn) hi++;
            int lo = hi > 0 ? hi-1 : 0, idx;
            if (lo == hi) idx = hi;
            else { float t = (wn-ORKI_NF4_LEVELS[lo])/(ORKI_NF4_LEVELS[hi]-ORKI_NF4_LEVELS[lo]); idx = (t >= 0.5f) ? hi : lo; }
            dq[k] = (float)((int8_t)lrintf(ORKI_NF4_LEVELS[idx]*127.0f)) * sc;  /* match the int8 LUT path */
        }
    } else {
        float scale = absmax / 7.0f, inv = scale > 0 ? 1.0f/scale : 0.0f;
        for (int k = 0; k < K; k++) {
            int q = (int)lrintf(fr[k]*inv); if (q > 7) q = 7; else if (q < -7) q = -7;
            dq[k] = (float)q * scale;
        }
    }
    float e = 0.0f;
    for (int k = 0; k < K; k++) { float d = fr[k]-dq[k]; e += (im ? im[k] : 1.0f) * d*d; }
    return e;
}

static float wq_best_absmax(const float *fr, int K, float rawabsmax, int nf4, const float *im, float *dq) {
    float best_abs = rawabsmax, best_e = wq_err_chan(fr, K, rawabsmax, nf4, im, dq);
    for (int g = 1; g < ORK_IM_CLIP_N; g++) {
        float cand = rawabsmax * ORK_IM_CLIP_GRID[g];
        float e = wq_err_chan(fr, K, cand, nf4, im, dq);
        if (e < best_e) { best_e = e; best_abs = cand; }
    }
    return best_abs;
}

ork_w *ork_mm_pack_i4a8(ork_npu *c, int K, int N, const float *f32, float *bscale_out) {
    return ork_mm_pack_i4a8_im(c, K, N, f32, NULL, bscale_out);
}

ork_w *ork_mm_pack_i4a8_im(ork_npu *c, int K, int N, const float *f32, const float *imatrix, float *bscale_out) {
    if (K % 32 || N % 32) return NULL;
    int sr = getenv("ORK_SR") != NULL; uint32_t seed = 0x2545F491u;   /* SR PRNG: fixed seed => deterministic/testable */
    int nf4 = getenv("ORK_NF4") != NULL;   /* ORK_NF4: non-uniform NF4 codebook instead of the uniform int4 grid */
    int KS = 1024, NMAX = c->soc->nmax, Sk = (K+KS-1)/KS, Sn = (N+NMAX-1)/NMAX;
    ork_w *w = calloc(1, sizeof *w); if (!w) return NULL;
    w->K = K; w->N = N; w->Sk = Sk; w->Sn = Sn; w->dtype = DT_I8; w->owns = 1; w->domain=ork_dom(c->pack_domain); w->quant_kind = nf4 ? ORK_QK_CODEBOOK_NF4 : ORK_QK_UNIFORM;
    w->Bb = calloc((size_t)Sk*Sn, sizeof(struct buf));
    if (!w->Bb) { free(w); return NULL; }
    for (int ns = 0; ns < Sn; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX;
      for (int ks = 0; ks < Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS;
        struct buf *b = &w->Bb[(size_t)ns*Sk+ks]; *b = orki_bcreate(c->fd, (size_t)Kp*Nc, 0x403, w->domain);
        if (!b->cpu) { for (int i = 0; i < ns*Sk+ks; i++) orki_bdestroy(c->fd, &w->Bb[i]); free(w->Bb); free(w); return NULL; } } }
    if (K <= 10752 && !getenv("ORK_NO_BF")) { w->Bf = calloc(Sn, sizeof(struct buf)); int ok = 1;
        for (int ns = 0; ns < Sn && ok; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX;
            struct buf *b = &w->Bf[ns]; *b = orki_bcreate(c->fd, (size_t)K*Nc, 0x403, w->domain); if (!b->cpu) ok = 0; }
        if (!ok) { for (int ns = 0; ns < Sn; ns++) orki_bdestroy(c->fd, &w->Bf[ns]); free(w->Bf); w->Bf = NULL; } }
    /* compact int4 nibble store (n-major, K contiguous): the memory-win form, kept on the ork_w */
    w->Bi4_bytes = (size_t)N * (K/2);
    w->Bi4 = malloc(w->Bi4_bytes);
    /* retain per-channel dequant scale on the ork_w so the compact int4 form can be dumped self-contained */
    w->bscale = malloc((size_t)N * sizeof(float));
    /* int8 expansion scratch (f32 codes) + per-channel inv for the int8 tiler (codes are exact, inv=1) */
    float *qf32 = malloc((size_t)N * K * sizeof(float));
    float *inv  = malloc((size_t)N * sizeof(float));
    /* NF4: a per-tensor int8 LUT = round(level*127), and an index scratch (0..15) to inflate through it */
    int8_t nf4_lut[16]; uint8_t *qidx = NULL;
    if (nf4) { for (int i = 0; i < 16; i++) nf4_lut[i] = (int8_t)lrintf(ORKI_NF4_LEVELS[i]*127.0f);
               qidx = malloc((size_t)N * K); }
    /* imatrix path: reused per-channel dequant scratch[K] for the clip-grid search (NULL imatrix => unused) */
    float *imdq = imatrix ? malloc((size_t)K * sizeof(float)) : NULL;
    if (!w->Bi4 || !w->bscale || !qf32 || !inv || (nf4 && !qidx) || (imatrix && !imdq)) {
        free(qf32); free(inv); free(qidx); free(imdq); ork_w_free(w); return NULL; }
    for (int n = 0; n < N; n++) {
        const float *fr = f32 + (size_t)n*K; float mx = 1e-9f; int k = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        float32x4_t vmx = vdupq_n_f32(1e-9f);
        for (; k <= K-4; k += 4) vmx = vmaxq_f32(vmx, vabsq_f32(vld1q_f32(fr + k)));
        float m[4]; vst1q_f32(m, vmx); float a=m[0]>m[1]?m[0]:m[1], bb=m[2]>m[3]?m[2]:m[3]; mx=a>bb?a:bb;
#endif
        for (; k < K; k++) { float v = fabsf(fr[k]); if (v > mx) mx = v; }
        if (imatrix) mx = wq_best_absmax(fr, K, mx, nf4, imatrix, imdq);  /* clip-grid scale selection */
        uint8_t *nib = w->Bi4 + (size_t)n*(K/2);
        if (nf4) { w->bscale[n] = mx / 127.0f;         /* int8 LUT range +-127 */
                   quant_chan_nf4(fr, K, mx, sr, &seed, nib, qidx + (size_t)n*K); }
        else     { float scale = mx / 7.0f;            /* int4 range +-7 (NOT 127) */
                   w->bscale[n] = scale;
                   quant_chan_i4(fr, K, scale, sr, &seed, nib, qf32 + (size_t)n*K); }
        if (bscale_out) bscale_out[n] = w->bscale[n];  /* back-compat: caller's out array (optional; w->bscale is canonical) */
        inv[n] = 1.0f;                                 /* qf32 holds exact codes; no rescale */
    }
    /* inflate the compact nibble store -> int8 f32 codes (validates the pack/inflate round-trip is what we tile) */
    if (nf4) for (int n = 0; n < N; n++) inflate_chan_nf4_f32(qidx + (size_t)n*K, K, nf4_lut, qf32 + (size_t)n*K);
    else     for (int n = 0; n < N; n++) expand_chan_i4_f32(w->Bi4 + (size_t)n*(K/2), K, qf32 + (size_t)n*K);
    orki_tile_f32_i8(c, w, K, N, qf32, inv);                /* REUSE the int8 DMA/tiling path (no dup) */
    free(qf32); free(inv); free(qidx); free(imdq);
    return w;
}

size_t ork_w_dump_i4a8(const ork_w *w, void *out, size_t cap){
    if(!w || !w->Bi4 || !w->bscale) return 0;
    size_t hdr=sizeof(struct ork_i4a8_hdr), sc=(size_t)w->N*sizeof(float), nib=(size_t)w->K*w->N/2;
    size_t need=hdr+sc+nib;
    if(!out) return need;
    if(cap<need) return 0;
    struct ork_i4a8_hdr h={ORK_I4A8_MAGIC, ORK_I4A8_VER, w->K, w->N, w->quant_kind};
    char *p=out;
    memcpy(p,&h,hdr); p+=hdr;
    memcpy(p,w->bscale,sc); p+=sc;
    memcpy(p,w->Bi4,nib);
    return need;
}

ork_w *ork_mm_load_i4a8(ork_npu *c, int K, int N, const void *blob, size_t n){
    if(K%32 || N%32) return NULL;
    size_t hdr=sizeof(struct ork_i4a8_hdr), sc=(size_t)N*sizeof(float), nib=(size_t)K*N/2;
    if(n != hdr+sc+nib) return NULL;
    const char *p=blob;
    struct ork_i4a8_hdr h; memcpy(&h,p,hdr); p+=hdr;
    if(h.magic!=ORK_I4A8_MAGIC || h.version!=ORK_I4A8_VER || h.K!=K || h.N!=N) return NULL;
    if(h.quant_kind!=ORK_QK_UNIFORM && h.quant_kind!=ORK_QK_CODEBOOK_NF4) return NULL;
    int KS=1024, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    ork_w *w=calloc(1,sizeof *w); if(!w) return NULL;
    w->K=K; w->N=N; w->Sk=Sk; w->Sn=Sn; w->dtype=DT_I8; w->owns=1; w->domain=ork_dom(c->pack_domain); w->quant_kind=(uint8_t)h.quant_kind;
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    if(!w->Bb){ free(w); return NULL; }
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;
        struct buf *b=&w->Bb[(size_t)ns*Sk+ks]; *b=orki_bcreate(c->fd,(size_t)Kp*Nc,0x403,w->domain);
        if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) orki_bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; } } }
    if(K<=10752 && !getenv("ORK_NO_BF")){ w->Bf=calloc(Sn,sizeof(struct buf)); int ok=1;
        for(int ns=0;ns<Sn && ok;ns++){ int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
            struct buf *b=&w->Bf[ns]; *b=orki_bcreate(c->fd,(size_t)K*Nc,0x403,w->domain); if(!b->cpu) ok=0; }
        if(!ok){ for(int ns=0;ns<Sn;ns++) orki_bdestroy(c->fd,&w->Bf[ns]); free(w->Bf); w->Bf=NULL; } }
    /* retain the compact store + scales so the loaded weight re-dumps byte-identically */
    w->Bi4_bytes=nib; w->Bi4=malloc(nib); w->bscale=malloc(sc);
    if(!w->Bi4 || !w->bscale){ ork_mm_free(c,w); return NULL; }
    memcpy(w->bscale,p,sc); p+=sc;
    memcpy(w->Bi4,p,nib);
    /* ORK_DIRECT_I4: inflate nibbles STRAIGHT to int8-tiled (1 byte/elem scratch, no f32 round-trip,
     * no re-quant) — bit-identical to the f32 path. Default off; preserves the f32 path for review. */
    if(getenv("ORK_DIRECT_I4")){
        int8_t *i8=malloc((size_t)N*K);
        if(!i8){ ork_mm_free(c,w); return NULL; }
        tile_direct_i4_i8(c, w, K, N, w->quant_kind, i8);
        free(i8);
        return w;
    }
    float *qf32=malloc((size_t)N*K*sizeof(float)), *inv=malloc((size_t)N*sizeof(float));
    if(!qf32 || !inv){ free(qf32); free(inv); ork_mm_free(c,w); return NULL; }
    int8_t nf4_lut[16];
    if(w->quant_kind==ORK_QK_CODEBOOK_NF4) for(int i=0;i<16;i++) nf4_lut[i]=(int8_t)lrintf(ORKI_NF4_LEVELS[i]*127.0f);
    for(int nn=0;nn<N;nn++){
        if(w->quant_kind==ORK_QK_CODEBOOK_NF4){
            /* NF4 store keeps the 0..15 index in the nibble; inflate via the int8 LUT */
            const uint8_t *nibp=w->Bi4+(size_t)nn*(K/2); float *qf=qf32+(size_t)nn*K;
            for(int k=0;k<K;k++){ uint8_t idx=(k&1)?(nibp[k>>1]>>4):(nibp[k>>1]&0xf); qf[k]=(float)nf4_lut[idx]; }
        } else expand_chan_i4_f32(w->Bi4+(size_t)nn*(K/2), K, qf32+(size_t)nn*K);
        inv[nn]=1.0f;                                  /* qf32 holds exact codes; no rescale */
    }
    orki_tile_f32_i8(c, w, K, N, qf32, inv);                /* REUSE the int8 DMA/tiling path (no dup) */
    free(qf32); free(inv);
    return w;
}

ork_w *ork_mm_load_i4a8_import(ork_npu *c, int K, int N, const void *blob, size_t n){
    if(K%32 || N%32 || orki_dmaheap_open()<0) return NULL;
    size_t hdr=sizeof(struct ork_i4a8_hdr), sc=(size_t)N*sizeof(float), nib=(size_t)K*N/2;
    if(n != hdr+sc+nib) return NULL;
    const char *p=blob;
    struct ork_i4a8_hdr h; memcpy(&h,p,hdr); p+=hdr;
    if(h.magic!=ORK_I4A8_MAGIC || h.version!=ORK_I4A8_VER || h.K!=K || h.N!=N) return NULL;
    if(h.quant_kind!=ORK_QK_UNIFORM && h.quant_kind!=ORK_QK_CODEBOOK_NF4) return NULL;
    int KS=1024, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    ork_w *w=calloc(1,sizeof *w); if(!w) return NULL;
    w->K=K; w->N=N; w->Sk=Sk; w->Sn=Sn; w->dtype=DT_I8; w->owns=1; w->domain=ork_dom(c->pack_domain); w->quant_kind=(uint8_t)h.quant_kind;
    ork_dom_prime(c, w->domain);   /* establish a non-0 domain with a native anchor BEFORE importing (same quirk as i8) */
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf)); if(!w->Bb){ free(w); return NULL; }
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;(void)n0;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=orki_bimport(c->fd,(size_t)Kp*Nc,w->domain);
        if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) orki_bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; } } }
    if(K%512==0 && K<=4096 && !getenv("ORK_NO_BF")){ w->Bf=calloc(Sn,sizeof(struct buf)); int ok=1;
        for(int ns=0;ns<Sn && ok;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
            struct buf*b=&w->Bf[ns]; *b=orki_bimport(c->fd,(size_t)K*Nc,w->domain); if(!b->cpu) ok=0; }
        if(!ok){ for(int ns=0;ns<Sn;ns++) orki_bdestroy(c->fd,&w->Bf[ns]); free(w->Bf); w->Bf=NULL; } }
    w->Bi4_bytes=nib; w->Bi4=malloc(nib); w->bscale=malloc(sc);
    if(!w->Bi4 || !w->bscale){ ork_mm_free(c,w); return NULL; }
    memcpy(w->bscale,p,sc); p+=sc; memcpy(w->Bi4,p,nib);
    int8_t *i8=malloc((size_t)N*K); if(!i8){ ork_mm_free(c,w); return NULL; }
    if(w->quant_kind==ORK_QK_CODEBOOK_NF4){ int8_t lut[16]; for(int i=0;i<16;i++) lut[i]=(int8_t)lrintf(ORKI_NF4_LEVELS[i]*127.0f);
        for(int nn=0;nn<N;nn++) orki_inflate_chan_nf4_i8(w->Bi4+(size_t)nn*(K/2),K,lut,i8+(size_t)nn*K);
    } else for(int nn=0;nn<N;nn++) orki_expand_chan_i4_i8(w->Bi4+(size_t)nn*(K/2),K,i8+(size_t)nn*K);
    orki_tile_i8_to_import_tiles(c,w,K,N,i8);
    free(i8);
    return w;
}

void ork_slice_inflate_i4a8_kind(const ork_w *w, float *qf32, int kind) {
    if (!w || !w->Bi4) return;
    int K = w->K, N = w->N;
    if (kind == ORK_QK_CODEBOOK_NF4) {
        int8_t lut[16]; for (int i = 0; i < 16; i++) lut[i] = (int8_t)lrintf(ORKI_NF4_LEVELS[i]*127.0f);
        for (int nn = 0; nn < N; nn++) {
            const uint8_t *nibp = w->Bi4 + (size_t)nn*(K/2); float *qf = qf32 + (size_t)nn*K;
            for (int k = 0; k < K; k++) { uint8_t idx = (k&1) ? (nibp[k>>1]>>4) : (nibp[k>>1]&0xf); qf[k] = (float)lut[idx]; }
        }
    } else {
        for (int nn = 0; nn < N; nn++) expand_chan_i4_f32(w->Bi4 + (size_t)nn*(K/2), K, qf32 + (size_t)nn*K);
    }
}

void ork_slice_inflate_i4a8(const ork_w *w, float *qf32) { ork_slice_inflate_i4a8_kind(w, qf32, w ? w->quant_kind : 0); }

void ork_slice_direct_i4a8_kind(ork_npu *c, ork_w *w, int8_t *i8scratch, int kind) {
    if (!w || !w->Bi4) return;
    tile_direct_i4_i8(c, w, w->K, w->N, kind, i8scratch);
}

struct ork_stream_entry *ork_stream_pool_add_i4a8(struct ork_stream_pool *p, int K, int N, const void *blob, size_t n){
    if(!p) return NULL;
    /* validate + materialize an int4 source ork_w from the blob (host-side Bi4+bscale), inflate into the
     * entry's staging dma-buf, then drop the temporary source (we only needed its nibble store to fill). */
    ork_w *src=ork_mm_load_i4a8(p->c,K,N,blob,n);  /* allocates resident DMA too — temporary; freed below */
    if(!src) return NULL;
    struct ork_stream_entry *e=orki_pool_new_entry(p,K,N);
    if(!e){ ork_mm_free(p->c,src); return NULL; }
    ork_stage_fill(p->c,e->stg,src);               /* the ONE-TIME inflate into RAM-resident staging */
    ork_mm_free(p->c,src);
    return e;
}

static void tile_i4_Bslice(uint8_t*dst,const int8_t*B,int K,int N,int k0,int Kp,int n0,int Nc){
    int KT=Kp/32, NB=Nc/64; memset(dst,0,(size_t)Kp*Nc/2);
    for(int nb=0;nb<NB;nb++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<64;nl++)for(int kk=0;kk<32;kk++){
        size_t idx=(((size_t)nb*KT+kt)*64+nl)*32+kk;
        dst[idx/2]|= (uint8_t)(B[(size_t)(k0+kt*32+kk)*N+(n0+nb*64+nl)]&0xf) << ((idx&1)?4:0);
    }
}

void orki_tile_i4_Aslice(uint8_t*dst,const int8_t*Arow,int k0,int Kp){
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int col = 0;
    int8x16_t vmask = vdupq_n_s8(0x0f);
    for (; col <= Kp - 16; col += 16) {
        int8x16_t v = vld1q_s8(&Arow[k0 + col]);
        int8x16_t vmasked = vandq_s8(v, vmask);
        int8x16x2_t tuzp = vuzpq_s8(vmasked, vmasked);
        uint8x8_t veven_low = vreinterpret_u8_s8(vget_low_s8(tuzp.val[0]));
        uint8x8_t vodd_low  = vreinterpret_u8_s8(vget_low_s8(tuzp.val[1]));
        uint8x8_t vodd_shifted = vshl_n_u8(vodd_low, 4);
        uint8x8_t vcombined = vorr_u8(veven_low, vodd_shifted);
        vst1_u8(&dst[col / 2], vcombined);
    }
#else
    int KT=Kp/32; memset(dst,0,(size_t)Kp/2);
    for(int kt=0;kt<KT;kt++)for(int kk=0;kk<32;kk++){
        size_t idx=(size_t)kt*32+kk;
        dst[idx/2]|= (uint8_t)(Arow[k0+kt*32+kk]&0xf) << ((idx&1)?4:0);
    }
#endif
}

static void tile_i4_Aslice_mm(uint8_t*dst,const int8_t*A,int M,int K,int k0,int Kp){
    int KT=Kp/32; memset(dst,0,(size_t)M*Kp/2);
    for(int kt=0;kt<KT;kt++)for(int m=0;m<M;m++)for(int kk=0;kk<32;kk++){
        size_t idx=((size_t)kt*M+m)*32+kk; uint8_t v=(uint8_t)(A[(size_t)m*K+k0+kt*32+kk]&0xf);
        dst[idx/2]|= (idx&1)?(v<<4):v;
    }
}

ork_w *ork_mm_pack_i4(ork_npu *c,int K,int N,const int8_t *B){
    if(c && c->daemon){ if(K%32||N%64) return NULL; uint64_t id=orkd_pack_i4(c->daemon,K,N,B); if(!id) return NULL; ork_w *w=calloc(1,sizeof *w); if(!w) return NULL; w->is_orkd=1; w->orkd_id=id; w->K=K; w->N=N; w->dtype=DT_I4; return w; }   /* Path B: int4 pack in the daemon */
    if(K%32||N%64) return NULL;
    int KS=ORK_I4_KS, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;  /* wide N-slices ≤ nmax */
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I4; w->owns=1; w->domain=ork_dom(c->pack_domain);
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    for(int ns=0;ns<Sn;ns++)for(int ks=0;ks<Sk;ks++){
        int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=orki_bcreate(c->fd,(size_t)Kp*Nc/2,0x403,w->domain);
        if(!b->cpu){
            fprintf(stderr,"[ork] ERROR: bcreate failed to allocate weight buffer Bb[%zu] in pack_i4 (size=%zu)\n",(size_t)ns*Sk+ks,(size_t)Kp*Nc/2);
            for(int i=0;i<ns*Sk+ks;i++) orki_bdestroy(c->fd,&w->Bb[i]);
            ork_w_free(w); return NULL;
        }
        tile_i4_Bslice(b->cpu,B,K,N,k0,Kp,n0,Nc);
        orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);
    }
    /* SLICE-AND-DICE RESCUE (#33): pre-build doorbell tiles for a REFUSE-PRONE int4 shape (Sn>1 => N>nmax,
     * or K>8192 => BCHAIN H<2) so ork_mm_run_i4's refuse (ORK_RC_WEDGE_PRONE) instead RUNS the shape by
     * decomposing it into BCHAIN-legal sub-tiles (raw nibble B only in scope here — pack_i4 keeps none). The
     * reachable trigger is fused/wide-N int4 prefill (Sn>1, per-core program count > cap). Gated so well-behaved
     * int4 (Sn==1, K<=8192) builds nothing. !orki_in_slice_pack: the sub-tiles below don't recurse. */
    if(!orki_in_slice_pack && !getenv("ORK_NO_SLICE_RESCUE") && ((Sn>1 || K>8192) || getenv("ORK_SLICE_ALL")))
        w->sliced = ork_mm_pack_sliced(c, K, N, B, DT_I4);
    return w;
}

ork_w *ork_mm_load_i4(ork_npu *c,int K,int N,const void *blob,size_t n){
    if(K%32||N%64) return NULL;
    int KS=ORK_I4_KS, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t need=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS; need+=orki_pgup((size_t)Kp*Nc/2);}}
    if(n!=need) return NULL;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I4; w->owns=1; w->domain=ork_dom(c->pack_domain);
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    size_t off=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=orki_bcreate(c->fd,(size_t)Kp*Nc/2,0x403,w->domain);
        if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) orki_bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; }
        memcpy(b->cpu,(const char*)blob+off,b->size); off+=b->size;
        orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}}
    return w;
}

ork_w *ork_mm_load_i4_arena(ork_npu *c,int K,int N,const void *blob,size_t n){
    if(K%32||N%64) return NULL;
    if(orki_dmaheap_open()<0) return NULL;
    int KS=ORK_I4_KS, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t need=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;(void)n0;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS; need+=orki_pgup((size_t)Kp*Nc/2);}}
    if(n!=need) return NULL;
    int dom=ork_dom(c->pack_domain); if(dom<0||dom>=64) return NULL;   /* arena tracks [64] domains */
    ork_w *w=calloc(1,sizeof *w); if(!w) return NULL;
    w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I4; w->owns=0; w->domain=dom;   /* shared-chunk views: owns nothing */
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf)); if(!w->Bb){ free(w); return NULL; }
    ork_dom_prime(c, dom);   /* native anchor establishes a non-0 domain BEFORE importing into it (mirror int8) */
    size_t chunk_mb=16; const char*cm=getenv("ORK_IMPORT_CHUNK_MB"); if(cm){ long v=atol(cm); if(v>0) chunk_mb=(size_t)v; }
    size_t chunk_cap=chunk_mb<<20;
    struct buf *cur=&c->i4arena_cur[dom];
    if(!cur->cpu || c->i4arena_off[dom]+need > cur->size){   /* switch chunk: keep this whole weight in ONE chunk */
        size_t csz = need>chunk_cap ? need : chunk_cap;      /* a single expert weight never exceeds 16MB in practice */
        struct buf nb=orki_bimport(c->fd,csz,dom);
        if(!nb.cpu){ free(w->Bb); free(w); return NULL; }
        if(c->i4arena_n>=c->i4arena_cap){ int nc2=c->i4arena_cap?c->i4arena_cap*2:64;
            struct buf*na=realloc(c->i4arena,(size_t)nc2*sizeof*na);
            if(!na){ orki_bdestroy(c->fd,&nb); free(w->Bb); free(w); return NULL; }
            c->i4arena=na; c->i4arena_cap=nc2; }
        /* fds stay OPEN until teardown (bdestroy closes them). 16MB chunks => ~320 fds for the 35B, under ulimit
         * (the EMFILE that forced per-weight fd-sealing was ~9k PER-EXPERT imports; consolidation removes it).
         * Sealing a chunk mid-load — while later chunks in the same domain are still being written — corrupted
         * the next chunk's reads (weight-32 miscompute); int8 only closes fds after a weight's whole load. */
        c->i4arena_curi[dom]=c->i4arena_n; c->i4arena[c->i4arena_n++]=nb; *cur=nb; c->i4arena_off[dom]=0;
    }
    orki_dmabuf_sync(cur->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);   /* bracket the CPU fill (mirror int8) */
    size_t off=c->i4arena_off[dom], boff=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;(void)n0;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS; size_t raw=(size_t)Kp*Nc/2, ts=orki_pgup(raw);
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks];
        b->handle=cur->handle; b->obj=cur->obj; b->dma=cur->dma+off; b->cpu=(char*)cur->cpu+off; b->size=ts; b->heap_fd=0; b->domain=dom;
        memcpy(b->cpu,(const char*)blob+boff,raw); off+=ts; boff+=ts; }}
    orki_dmabuf_sync(cur->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);
    c->i4arena_off[dom]=off;
    return w;
}

ork_w *ork_mm_load_i4_import(ork_npu *c,int K,int N,const void *blob,size_t n){
    if(K%32||N%64) return NULL;
    if(orki_dmaheap_open()<0) return NULL;
    int KS=ORK_I4_KS, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t need=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;(void)n0;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS; need+=orki_pgup((size_t)Kp*Nc/2);}}
    if(n!=need) return NULL;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I4; w->owns=1; w->domain=ork_dom(c->pack_domain);
    c->scratch_import=1;   /* weights are now bimported into their domains -> the run scratch must import too (bcreate EINVALs alongside imports); see bscratch */
    ork_dom_prime(c, w->domain);   /* #54: establish the non-0 domain with ONE native anchor (Quirk 1, NPU-Quirks.md) — MATCH ork_mm_load_i8_import exactly. (Was ork_dom_reanchor/bdestroy+recreate-per-import: destroying the buffer that established the domain is suspected of breaking its lazy-init state at scale; prime-once is the documented proven guard.) */
    /* #54: pre-allocate THIS domain's run scratch NOW, while it is still LIGHT (only the anchor). mc_ensure's
     * mtk_all + per-core mrc/mtk/maf are kernel-mapped and MUST be bcreate — allocated later (at the first run,
     * after this domain fills with imports) a fresh bcreate EINVALs amid the imports (the mc_ensure mtk_all
     * failure). dom_activate makes w->domain the active set (parking the prior domain's); mc_ensure + the mcc
     * ensure below alloc once per domain (idempotent: skip if already sized). Generous mcc (>= the expert BCHAIN
     * need_o ~688 KiB) so BCHAIN never re-grows (=re-bcreates) it in the now-heavy domain. */
    if(w->domain>0){
        orki_dom_activate(c, w->domain);
        orki_mc_ensure(c, c->soc->cores);
        size_t mcc_need = (size_t)2*1024*1024;   /* covers expert BCHAIN need_o with margin */
        for(int i=0;i<c->soc->cores;i++)
            if(c->mccsz[i] < mcc_need){ orki_bdestroy(c->fd,&c->mcc[i]); c->mcc[i]=orki_bscratch(c,mcc_need,0x403,c->dom_active); if(c->mcc[i].cpu){ c->mccsz[i]=mcc_need; c->mwarm[i]=0; } }
    }
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    int consolidate = !getenv("ORK_NO_CONSOLIDATE_IMPORT");
    if(consolidate){
        size_t chunk_mb = 16; const char*cm=getenv("ORK_IMPORT_CHUNK_MB"); if(cm){ long v=atol(cm); if(v>0) chunk_mb=(size_t)v; }
        size_t chunk_cap = chunk_mb<<20;
        int ntiles=Sk*Sn, cap_chunks=ntiles+1;
        w->own_bufs=calloc(cap_chunks,sizeof(struct buf)); w->n_own_bufs=0;
        struct buf cur; cur.cpu=NULL; size_t coff=0, csz=0;
        int ns,ks; size_t boff=0;
        for(ns=0;ns<Sn && consolidate;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;(void)n0;
          for(ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS; size_t ts=orki_pgup((size_t)Kp*Nc/2);
            if(!cur.cpu || coff+ts>csz){
                if(cur.cpu) orki_dmabuf_sync(cur.heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);
                size_t rem = need - boff;                       /* cap the chunk to THIS weight's remaining need: a small per-expert weight (~0.5 MiB) must NOT grab a full chunk_cap (16 MiB) chunk — that burned ~16 MiB IOVA PER expert (~15k experts) and blew the domains. */
                csz = rem < chunk_cap ? rem : chunk_cap; if(csz < ts) csz = ts;
                cur = orki_bimport(c->fd,csz,w->domain);
                if(!cur.cpu){ consolidate=0; break; }
                orki_dmabuf_sync(cur.heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
                w->own_bufs[w->n_own_bufs++]=cur; coff=0;
            }
            struct buf*b=&w->Bb[(size_t)ns*Sk+ks];
            b->handle=cur.handle; b->obj=cur.obj; b->dma=cur.dma+coff; b->cpu=(char*)cur.cpu+coff; b->size=ts;
            memcpy(b->cpu,(const char*)blob+boff,(size_t)Kp*Nc/2); coff+=ts; boff+=ts;}}
        if(consolidate){ if(cur.cpu) orki_dmabuf_sync(cur.heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE); w->owns=0; }
        else { for(int i=0;i<w->n_own_bufs;i++) orki_bdestroy(c->fd,&w->own_bufs[i]);
            free(w->own_bufs); w->own_bufs=NULL; w->n_own_bufs=0;
            memset(w->Bb,0,(size_t)Sk*Sn*sizeof(struct buf)); w->owns=1; }
    }
    if(!consolidate){
        size_t off=0;
        for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;(void)n0;
          for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;
            struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=orki_bimport(c->fd,(size_t)Kp*Nc/2,w->domain);
            if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) orki_bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; }
            orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
            memcpy(b->cpu,(const char*)blob+off,(size_t)Kp*Nc/2); off+=b->size;
            orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
    }
    /* SEAL the dma-buf fds now that the CPU fill + coherency sync are done. The GEM handle (from MEM_CREATE) and
     * the mmap keep the buffer alive for NPU reads (via IOMMU b->dma); a resident int4 weight is READ-ONLY after
     * load, so no later dmabuf_sync is needed (verified: dmabuf_sync is called only in the load/import path, never
     * per-run). This drops the held-fd count from ~1 per weight (~16k for the 35B MoE -> blew the fd ulimit) to ~0:
     * load_i4_import runs per-weight, so each weight's fds close before the next weight imports. heap_fd=0 tells
     * bdestroy the fd is already closed (it still MEM_DESTROYs via the handle). Consolidated tiles are VIEWS
     * (heap_fd=0 already, never bdestroy'd individually); only the chunks (own_bufs) and per-tile bufs hold fds. */
    if(!getenv("ORK_NO_SEAL")){   /* TEST: does closing the dma-buf fd (int4-only; int8 keeps them open) alias the next import? */
    for(int i=0;i<w->n_own_bufs;i++){ if(w->own_bufs[i].heap_fd>0){ close(w->own_bufs[i].heap_fd); w->own_bufs[i].heap_fd=0; } }
    for(int i=0;i<Sk*Sn;i++){ if(w->Bb[i].heap_fd>0){ close(w->Bb[i].heap_fd); w->Bb[i].heap_fd=0; } }
    }
    return w;
}

ork_w *ork_mm_pack_i4_grouped(ork_npu *c,int K,int N,const int8_t *B,int G){
    if(K%32||N%64||G%32||K%G||G>ORK_I4_KS) return NULL;
    int NMAX=c->soc->nmax, Sk=K/G, Sn=(N+NMAX-1)/NMAX;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I4;w->gsize=G; w->domain=ork_dom(c->pack_domain);
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    /* Reserve the whole weight as ONE dedicated DMA buffer (own_buf); each group-tile is a 4KB-aligned
     * VIEW into it (shared obj, dma=own_buf.dma+off). Collapses the per-group bcreate storm to a single
     * allocation => fast warmup, no IOVA-handle OOM, and — crucially — RECLAIMABLE: ork_mm_free destroys
     * own_buf (returning its IOVA to the 4 GiB window), so drop/reload of a grouped weight does NOT leak
     * (streaming / MoE-swap). The whole region is flushed to device in a single bsync. Falls back to
     * per-tile owning orki_bcreate (also reclaimable) if the dedicated alloc fails. */
    size_t wtotal=0;
    for(int ns=0;ns<Sn;ns++)for(int g=0;g<Sk;g++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
        wtotal += (((size_t)G*Nc/2)+4095u)&~(size_t)4095u; }
    struct buf own=orki_bcreate(c->fd,wtotal,0x403,w->domain);
    if(own.cpu){
        w->own_buf=own; w->own_buf_valid=1;
        size_t off=0;
        for(int ns=0;ns<Sn;ns++)for(int g=0;g<Sk;g++){
            int k0=g*G,n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX; size_t ts=(size_t)G*Nc/2;
            struct buf*b=&w->Bb[(size_t)ns*Sk+g];
            b->handle=own.handle; b->obj=own.obj; b->dma=own.dma+off; b->cpu=(char*)own.cpu+off; b->size=ts;
            tile_i4_Bslice(b->cpu,B,K,N,k0,G,n0,Nc);
            off += (ts+4095u)&~(size_t)4095u;
        }
        orki_bsync_off(c->fd,own.obj,0,wtotal,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        orki_bsync_off(c->fd,own.obj,0,wtotal,RKNPU_MEM_SYNC_TO_DEVICE);
    } else {
        w->owns=1;   /* per-tile owning bcreate: reclaimable by ork_mm_free */
        for(int ns=0;ns<Sn;ns++)for(int g=0;g<Sk;g++){
            int k0=g*G,n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
            struct buf*b=&w->Bb[(size_t)ns*Sk+g]; *b=orki_bcreate(c->fd,(size_t)G*Nc/2,0x403,w->domain);
            if(!b->cpu){ fprintf(stderr,"[ork] ERROR: weight alloc failed (G=%d Nc=%d) in pack_i4_grouped\n",G,Nc);
                for(int i=0;i<ns*Sk+g;i++) orki_bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; }
            tile_i4_Bslice(b->cpu,B,K,N,k0,G,n0,Nc);
            orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);
        }
    }
    return w;
}

ork_w *ork_mm_pack_i4_to_i8(ork_npu *c, int K, int N, const int8_t *B) {
    /* The core fallback: it takes int4-range values ([-8,7]) unpacked in int8_t containers,
     * but physically packs them into the highly optimized int8 resident weight layout.
     * This simply defers to ork_mm_pack_i8, yielding maximum int8 silicon execution speed
     * while the caller (e.g. ggml-ork) maintains the 50% footprint reduction on disk. */
    return ork_mm_pack_i8(c, K, N, B);
}

int ork_mm_run_i4_experts(ork_npu *c, const ork_mm_task_i4 *ex, int ntask, int nc){
    if(!c || ntask<1 || !ex) return -1;
    for(int e=0;e<ntask;e++){ ork_w*w=ex[e].w;
        if(!w||w->dtype!=DT_I4||w->Sk!=1||w->Sn!=1||(w->N%64)||ex[e].M<1) return -2;
        if(w->domain!=ex[0].w->domain || w->K!=ex[0].w->K || w->N!=ex[0].w->N) return -2; }  /* one submit => one domain + one shape */
    return orki_run_i4_experts_bchain_db(c, ex, ntask, nc);   /* M-batched BCHAIN programs chained across experts */
}

int ork_mm_run_i4(ork_npu *c,ork_w *w,int M,const int8_t *A,int32_t *C){
    if(w && w->is_orkd){   /* Path B: int4 run on the daemon — ring transport if attached, else socket */
        orkd_set_op_domain(c->daemon, (uint32_t)w->domain);   /* v2: carry this weight's domain with the op */
        if(c && c->daemon && orkd_has_ring(c->daemon)){ int r=orkd_ring_run(c->daemon,w->orkd_id,M,w->K,w->N,ORKD_DT_I4,A,C); if(r!=-2) return r; }
        return orkd_run_i4(c->daemon, w->orkd_id, M, w->K, w->N, A, C); }
    if(!w||w->dtype!=DT_I4) return -1;
    /* Multi-domain: the submit's regcmd/task/scratch AND the weight must live in the SAME iommu domain. Activate
     * this weight's domain before the int4 submit — mirror the int8 run paths. Without it a resident int4 weight
     * in domain N submits against the stale dom_active (e.g. 0) -> RKNPU_SUBMIT EINVAL(22) -> self-heal reset ->
     * retry (correctness held via the reset, but every cross-domain expert submit thrashed -> very slow). */
    if(w->domain!=c->dom_active || (w->domain!=0 && !c->dom_save)) orki_dom_activate(c,w->domain);
    if(orki_check_overlap("ork_mm_run_i4", (uintptr_t)A, (uintptr_t)A + (size_t)M * w->K, (uintptr_t)C, (uintptr_t)C + (size_t)M * w->N * 4)) return -1;
    int NB=w->N/64;                            /* total 64-wide N-blocks (column-split granularity) */
    int nc=orki_budget(c, M); if(nc>NB)nc=NB; if(nc<1)nc=1;   /* ≥1 N-block/core; nc==1 = serial */
    /* DEFAULT int4 M>1 prefill: BCHAIN batch-chain on the NONBLOCK doorbell (run_i4_bchain_db) — H-row native
     * batches (synth_i4 mc>1) + bank-width Wb=131072/K N-tiling + weight-loaded-once chaining, self-healing on
     * the doorbell spine. Bit-exact, ~18-25x over the per-row doorbell, and serves the large-M shapes the
     * per-row path refuses (#52). Falls through (-4) to the per-row doorbell for decode (M=1)/non-qualifying. */
    /* #33 TEST/A-B hook: force a tile-bearing int4 shape onto the slice rescue (bit-exact validation). */
    if(w->sliced && getenv("ORK_FORCE_SLICE_RESCUE")){ int rs=ork_mm_run_sliced(c,w->sliced,M,A,C,nc); if(rs>=0) return rs; }
    if(getenv("ORK_I4_DIAG")) fprintf(stderr,"[i4diag] run_i4 K=%d N=%d M=%d Sk=%d Sn=%d dom=%d imported=%d -> %s\n",
        w->K,w->N,M,w->Sk,w->Sn,w->domain,(w->own_bufs&&w->n_own_bufs>0)||w->own_buf_valid,
        (M>=2 && w->Sk==1 && w->Sn==1 && (w->N%64)==0)?"BCHAIN":(w->sliced&&M>=2)?"SLICE":"mc_i4(per-row)");
    /* #54: DEFAULT int4 M>1 prefill = BCHAIN (M-batched, the perf path) — now PORTED to ride the SHARED ork_dyn_end
     * drain (poll + orki_mc_recover_resubmit + reap-at-boundary), so it is multi-domain-safe like int8 colsplit AND
     * keeps its M-batching. bch_db_worker builds+submits only; ork_dyn_end owns the drain (i4batch hooks). Falls
     * through (-4) to the per-row doorbell (orki_run_i4_mc_db) for decode (M=1) / non-qualifying shapes. */
    if(M>=2 && w->Sk==1 && w->Sn==1 && (w->N%64)==0){ int r=orki_run_i4_bchain_db(c,w,M,A,C,nc); if(r!=-4) return r; }
    /* Wide refuse-prone int4 PREFILL (Sn>1 or K>8192 — the shapes pack built w->sliced for): the per-row
     * orki_run_i4_mc_db below CAN run these but only per-row (~6x slower); route M>=2 straight to the BCHAIN-tiled
     * rescue (measured 663ms -> 107ms at M=128 N=16384). Decode (M==1) stays on the per-row path (cheap). */
    if(M>=2 && w->sliced){ int rs=ork_mm_run_sliced(c,w->sliced,M,A,C,nc); if(rs>=0) return rs; }
    /* decode (M=1) + non-batch shapes ride the per-row doorbell chain (ork_dyn_begin_mc_i4): Sk>1/Sn>1 via
     * chained column/K-slice programs. -4 (over-large chain / unsupported int4 shape) => refuse (rescue-eligible).
     * All blocking int4 paths (i4_mcworker / INCR / CBATCH / blocking BCHAIN) are removed (#45/#52). */
    if(!orki_ork_prof){ int r=orki_run_i4_mc_db(c,w,M,A,C,nc); if(r!=-4) return r;
        if(w->sliced){ int rs=ork_mm_run_sliced(c,w->sliced,M,A,C,nc); if(rs>=0) return rs; }   /* #33: rescue the refused shape via BCHAIN sub-tiles, else refuse */
        return ORK_RC_WEDGE_PRONE; }
    double t0=ork_now_us(); int r=orki_run_i4_mc_db(c,w,M,A,C,nc); orki_prof_i4_us+=ork_now_us()-t0; orki_prof_i4_calls++;
    if(r!=-4) return r;
    if(w->sliced){ int rs=ork_mm_run_sliced(c,w->sliced,M,A,C,nc); if(rs>=0) return rs; }   /* #33: rescue */
    return ORK_RC_WEDGE_PRONE;
}

int ork_mm_run_i4_grouped(ork_npu *c,ork_w *w,int M,const int8_t *A,const float *aScale,const float *bScale,float *C){
    if(!w||w->dtype!=DT_I4||!w->gsize) return -1;
    if(orki_check_overlap("ork_mm_run_i4_grouped", (uintptr_t)A, (uintptr_t)A + (size_t)M * w->K, (uintptr_t)C, (uintptr_t)C + (size_t)M * w->N * 4)) return -1;
    /* Grouped int4 runs on the NONBLOCK doorbell (row-decomposed Sn*Sk chain + float scale-accumulate
     * drain). NULL (chain/scratch too big / ineligible) => refuse (rescue-eligible); the blocking
     * i4_mcworker_g path is removed (#45). */
    ork_dyn_chain *hg=ork_dyn_begin_mc_i4_grouped(c,M,w,A,aScale,bScale,C,0);
    if(hg) return ork_dyn_grouped_end(hg)?-1:0;
    /* #33 GROUPED RESCUE (M-chunk): the doorbell refused because the per-core program count
     * (rows/core)*Sn*Sk exceeds the regcmd-buffer cap (~70). Rows are INDEPENDENT, so M-CHUNK the rows —
     * each chunk is a full grouped matmul (all groups + all N, so the per-group float drain stays intact)
     * writing its own CONTIGUOUS C rows: no weight repack, no scale slicing, no scatter, just a recursive
     * call with fewer rows (which no longer refuses once rows/core*Sn*Sk <= cap). Only possible when a SINGLE
     * row fits (Sn*Sk <= cap); wide-N (Sn>1) or fine-group large-K (Sn*Sk > cap even at 1 row) still refuses —
     * needs N-tile / K-group-slice (a follow-on). The M>1 gate bounds the recursion (bottoms out at M==1). */
    int G=w->gsize, Sk=w->K/G, per_row=w->Sn*Sk;
    if(M>1 && per_row>0 && per_row<=64){                                 /* 64 = margin under the ~70-program/core cap */
        int nc=orki_budget(c,M); if(nc<1)nc=1; int rpc=64/per_row; if(rpc<1)rpc=1;
        int Msub=rpc*nc; if(Msub>=M) Msub=M-1; if(Msub<1) Msub=1;
        int ok=1;
        for(int m0=0;m0<M && ok;m0+=Msub){ int mm=(M-m0<Msub)?(M-m0):Msub;
            if(ork_mm_run_i4_grouped(c,w,mm, A+(size_t)m0*w->K, aScale+(size_t)m0*Sk, bScale, C+(size_t)m0*w->N)) ok=0; }
        if(ok) return 0;
    }
    return ORK_RC_WEDGE_PRONE;
}

ork_w_sliced *orki_slice_pack_i4(ork_npu *c, int K, int N, const int8_t *B) {
    if (N % 64) return NULL;                                             /* pack_i4 requires N%64 (a real int4 weight satisfies it); N is not padded */
    int Kpad = ((K + 31) / 32) * 32;                                    /* pad K to 32 (pack_i4 K%32); zero rows contribute 0 -> bit-exact */
    int ks = 8192, ns = 8192;                                           /* K-slice <=8192 (BCHAIN H>=2); N-tile <=8192 (Sn==1). both %64 & %32 */
    int nks = (Kpad + ks - 1) / ks, nnt = (N + ns - 1) / ns;
    struct ork_w_sliced *w = calloc(1, sizeof *w); if (!w) return NULL;
    w->K = K; w->N = N; w->Kpad = Kpad; w->dtype = DT_I4; w->cap = ork_slice_caps_rk3588(); w->nks = nks; w->nnt = nnt; w->ks = ks; w->ns = ns;
    w->sub = calloc((size_t) nks * nnt, sizeof(ork_w *));
    int8_t *blk = malloc((size_t) ks * ns);
    if (!w->sub || !blk) { free(blk); ork_mm_free_sliced(c, w); return NULL; }
    orki_in_slice_pack = 1;
    for (int ki = 0; ki < nks; ki++) { int k0 = ki*ks, k1 = k0+ks < Kpad ? k0+ks : Kpad, Ks = k1-k0;
        for (int ni = 0; ni < nnt; ni++) { int n0 = ni*ns, n1 = n0+ns < N ? n0+ns : N, Nw = n1-n0;
            for (int k = 0; k < Ks; k++) { if (k0+k < K) memcpy(blk + (size_t) k*Nw, B + (size_t)(k0+k)*N + n0, Nw);   /* real nibble row */
                                           else          memset(blk + (size_t) k*Nw, 0, Nw); }                        /* PAD row -> zero */
            ork_w *sw = ork_mm_pack_i4(c, Ks, Nw, blk);                 /* Sk==1, Sn==1, N%64 tile -> BCHAIN-eligible */
            if (!sw) { orki_in_slice_pack = 0; free(blk); ork_mm_free_sliced(c, w); return NULL; }
            w->sub[ki*nnt + ni] = sw; } }
    orki_in_slice_pack = 0; free(blk); return w;
}

int orki_slice_run_i4(ork_npu *c, ork_w_sliced *w, int M, const int8_t *A, int32_t *C, int nc) {
    if (!c || !w || !A || !C || M < 1) return -1;
    int ks = w->ks, ns = w->ns, nks = w->nks, nnt = w->nnt, S = nks * nnt, K = w->K, N = w->N, Kpad = w->Kpad;
    int8_t  *Aslc = malloc((size_t) M * Kpad);
    int32_t *part = malloc((size_t) nks * M * N * sizeof(int32_t));
    ork_mm_task_i8 *tasks = malloc((size_t) S * sizeof *tasks);
    if (!Aslc || !part || !tasks) { free(Aslc); free(part); free(tasks); return -1; }
    size_t aoff = 0, poff = 0; int rc = 0;
    for (int ki = 0; ki < nks && !rc; ki++) { int k0 = ki*ks, k1 = k0+ks < Kpad ? k0+ks : Kpad, Ks = k1-k0;
        int8_t *aptr = Aslc + aoff;
        int real = K - k0; if (real > Ks) real = Ks; if (real < 0) real = 0;   /* real A cols this slice; PAD tail -> 0 */
        for (int m = 0; m < M; m++) { memcpy(aptr + (size_t) m*Ks, A + (size_t) m*K + k0, real);
                                      if (real < Ks) memset(aptr + (size_t) m*Ks + real, 0, Ks - real); }
        aoff += (size_t) M * Ks;
        for (int ni = 0; ni < nnt && !rc; ni++) { int n0 = ni*ns, n1 = n0+ns < N ? n0+ns : N, Nw = n1-n0;
            int32_t *ptile = part + poff; poff += (size_t) M * Nw;
            tasks[ki*nnt + ni] = (ork_mm_task_i8){ w->sub[ki*nnt + ni], M, aptr, ptile };
            int nct = nc>0 ? nc : c->soc->cores; int nb = Nw/64; if (nct > nb) nct = nb; if (nct < 1) nct = 1;
            int r = (M >= 2) ? orki_run_i4_bchain_db(c, w->sub[ki*nnt + ni], M, aptr, ptile, nct)   /* BCHAIN batch on the doorbell */
                             : orki_run_i4_mc_db    (c, w->sub[ki*nnt + ni], M, aptr, ptile, nct);  /* M==1 per-row doorbell */
            if (r < 0) rc = -1; } }
    if (!rc) {   /* per-core PARALLEL ks-outer int32 accumulate + N scatter (same worker as int8) */
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
            orki_slice_acc_worker(&acc[0]);
            pthread_mutex_lock(&c->pmu); while(c->pdone < anc-1) pthread_cond_wait(&c->pdn,&c->pmu); pthread_mutex_unlock(&c->pmu);
        }
    }
    free(Aslc); free(part); free(tasks); return rc;
}

static void tile_i4_A(uint8_t*dst,const int8_t*A,int M,int K,int nib){
    int KT=K/32; memset(dst,0,(size_t)M*K/2);
    for(int kt=0;kt<KT;kt++)for(int m=0;m<M;m++)for(int kk=0;kk<32;kk++){
        size_t idx=((size_t)kt*M+m)*32+kk; uint8_t v=(uint8_t)(A[(size_t)m*K+kt*32+kk]&0xf);
        dst[idx/2]|= ((idx&1)^nib)?(v<<4):v;
    }
}

static void tile_i4_B(uint8_t*dst,const int8_t*B,int K,int N,int nib){
    int KT=K/32,NT=N/64; memset(dst,0,(size_t)K*N/2);
    for(int nt=0;nt<NT;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<64;nl++)for(int kk=0;kk<32;kk++){
        size_t idx=(((size_t)nt*KT+kt)*64+nl)*32+kk;
        uint8_t v=(uint8_t)(B[(size_t)(kt*32+kk)*N + (nt*64+nl)]&0xf);
        dst[idx/2]|= ((idx&1)^nib)?(v<<4):v;
    }
}

int ork_npu_probe_i4(ork_npu *c,int M,int K,int N,int nibB,int nibA,int nov,
                     const uint32_t *ovr_reg,const uint32_t *ovr_val,
                     const int8_t *A,const int8_t *B,int16_t *C){
    int fd=c->fd;
    if(K%32||N%64||N>c->soc->nmax) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N/2,0x403,-1); if(!W.cpu) return -2;        /* B int4: half bytes */
    tile_i4_B(W.cpu,B,K,N,nibB);
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)M*N*2,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}  /* int16 C, M rows */
    /* M-tiling: the captured W4A4 program runs M=1 per task; we replicate it per row. Each row's A is
     * its own native (K/32,1,32) block (contiguous K/2 bytes); each row's C is (N/8,1,8) = N int16. */
    uint8_t*ad=c->Af.cpu;
    for(int m=0;m<M;m++) tile_i4_A(ad+(size_t)m*(K/2), A+(size_t)m*K, 1, K, nibA);
    orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=0;
    for(int m=0;m<M && ok==0;m++){
        orki_act(fd,RKNPU_ACT_RESET,0);
        uint32_t rc[REGCMD_I4_N];
        orki_synth_i4(rc,1,K,N,(uint32_t)(c->Af.dma+(size_t)m*(K/2)),(uint32_t)W.dma,(uint32_t)(O.dma+(size_t)m*N*2));
        for(int i=0;i<nov && i<4;i++) orki_setr(rc,REGCMD_I4_N,0x201,ovr_reg[i],ovr_val[i]);
        struct buf extra[2] = {W, O};
        if (orki_validate_regcmd("probe_i4", c, rc, REGCMD_I4_N, NULL, extra, 2)) { orki_bdestroy(fd,&W); orki_bdestroy(fd,&O); return -1; }
        memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
        sub.timeout=orki_mm_timeout_ms(); ok=-1;
        for(int rep=0;rep<2;rep++){ if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; continue; }
            orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; }
        if(ok==0){ int16_t*cr=(int16_t*)((char*)O.cpu+(size_t)m*N*2);   /* row m: native (N/8,1,8) */
            for(int nt=0;nt<N/8;nt++)for(int nl=0;nl<8;nl++) C[(size_t)m*N + nt*8+nl] = cr[nt*8+nl]; }
    }
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);
    return ok;
}

int ork_npu_probe_i4_mm(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,int16_t *raw){
    int fd=c->fd;
    if(K%32||N%64||N>c->soc->nmax||M<1) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N/2,0x403,-1); if(!W.cpu) return -2;
    tile_i4_B(W.cpu,B,K,N,0);
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)2*M*N*2,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}  /* 2x: stride-2 multi-M writes physical rows 0..2(M-1) */
    /* A layout selector via ORK_I4_ALAY: 0=(K/32,M,32) interleaved, 1=per-row contiguous (K/32,1,32)
     * x M (what the captured M=1 program reads). Lets the probe tell whether the program is single-row. */
    { int alay=getenv("ORK_I4_ALAY")?atoi(getenv("ORK_I4_ALAY")):0;
      if(alay) for(int m=0;m<M;m++) tile_i4_A((uint8_t*)c->Af.cpu+(size_t)m*(K/2),A+(size_t)m*K,1,K,0);
      else tile_i4_A(c->Af.cpu,A,M,K,0); }
    orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I4_N];
    orki_synth_i4(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma);
    struct buf extra[2] = {W, O};
    if (orki_validate_regcmd("probe_i4_mm", c, rc, REGCMD_I4_N, NULL, extra, 2)) { orki_bdestroy(fd,&W); orki_bdestroy(fd,&O); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    /* Fill c->task with the INT4 regcfg count (116 = REGCMD_I4_N/2). init/reset stamp the shared c->task
     * with the int8 count (108); a 108-reg task over a 116-reg int4 regcmd -> kernel EINVAL. Normal int4
     * runs fill their own MC task bufs; the probe uses the legacy c->task, so it must set the int4 count. */
    { struct rknpu_task t; memset(&t,0,sizeof t); t.enable_mask=0xd; t.int_mask=0x300; t.int_clear=0x1ffff; t.regcfg_amount=REGCMD_I4_N/2; t.regcmd_addr=c->regcmd.dma;
      memcpy(c->task.cpu,&t,sizeof t); orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); }
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    /* ORK_I4_PROBE_TO_MS: submit timeout (default 60s). The fuzzer sets this low (e.g. 1500) so a wedging
     * candidate has the KERNEL time out the job fast and return an error in-process — the fuzzer blacklists
     * it and continues, with no external SIGINT-during-submit (the documented wedge/corruption risk). */
    uint32_t to_ms=60000; { const char*e=getenv("ORK_I4_PROBE_TO_MS"); if(e){ unsigned v=(unsigned)strtoul(e,0,0); if(v) to_ms=v; } }
    int ok=-1;
    for(int rep=0;rep<2;rep++){ sub.timeout=to_ms; if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; continue; }
        orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; }
    if(ok==0) memcpy(raw,O.cpu,(size_t)2*M*N*2);   /* caller supplies a 2*M*N int16 buffer (stride-2) */
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);
    return ok;
}

int ork_bmm_i4_strided(ork_npu *c, int nbatch, int M, int K, int N,
                       const int8_t *A, const int8_t *B, int32_t *C, const ork_bmm_strides *s){
    if(!c||!A||!B||!C||!s) return -1;
    if(nbatch<1||M<1||K<1||N<1) return -2;
    if(K%32||N%64) return -2;
    int8_t *Ac=malloc((size_t)M*K), *Bc=malloc((size_t)K*N);
    int cdense=orki_bmm_c_dense(s,N); int32_t *Cc = cdense?NULL:malloc((size_t)M*N*sizeof(int32_t));
    if(!Ac||!Bc||(!cdense&&!Cc)){ free(Ac);free(Bc);free(Cc); return -3; }
    int rc=0;
    for(int b=0;b<nbatch;b++){
        orki_bmm_gather_i8(Bc,B+(long)b*s->bbs,K,N,s->bs_k,s->bs_n);
        orki_bmm_gather_i8(Ac,A+(long)b*s->abs,M,K,s->as_m,s->as_k);
        ork_w *w=ork_mm_pack_i4(c,K,N,Bc); if(!w){ rc=-3; break; }
        int32_t *Cout = cdense ? C+(long)b*s->cbs : Cc;
        int r=ork_mm_run_i4(c,w,M,Ac,Cout);
        ork_mm_free(c,w);
        if(r){ rc=-5; break; }
        if(!cdense) orki_bmm_scatter_i32(C+(long)b*s->cbs,Cc,M,N,s->cs_m,s->cs_n);
    }
    free(Ac);free(Bc);free(Cc);
    return rc;
}

int ork_bmm_i4(ork_npu *c, int nbatch, int M, int K, int N,
               const int8_t *A, const int8_t *B, int32_t *C){
    ork_bmm_strides s=orki_bmm_natural(M,K,N); return ork_bmm_i4_strided(c,nbatch,M,K,N,A,B,C,&s);
}

size_t ork_pack_i4a8_cpu_blob(ork_npu *c, int K, int N, const float *f32, const float *imatrix, int nf4, void *out, size_t cap){
    (void)c;
    if(K%32 || N%32 || !f32) return 0;
    size_t hdr=sizeof(struct ork_i4a8_hdr), sc=(size_t)N*sizeof(float), nibsz=(size_t)K*N/2, need=hdr+sc+nibsz;
    if(!out) return need;
    if(cap<need) return 0;
    nf4 = nf4 ? 1 : 0;   /* codebook routed by the caller (source-based), not an env flag */
    int sr  = getenv("ORK_SR")!=NULL; uint32_t seed=0x2545F491u;   /* per-call seed matches ork_mm_pack_i4a8_im */
    struct ork_i4a8_hdr h={ORK_I4A8_MAGIC, ORK_I4A8_VER, K, N, (uint32_t)(nf4?ORK_QK_CODEBOOK_NF4:ORK_QK_UNIFORM)};
    char    *p=(char*)out;
    float   *bscale=(float*)(p+hdr);
    uint8_t *Bi4   =(uint8_t*)(p+hdr+sc);
    float   *qf32=malloc((size_t)K*sizeof(float));       /* quant_chan_i4 code byproduct; reused per channel */
    uint8_t *qidx=nf4?malloc((size_t)K):NULL;
    float   *imdq=imatrix?malloc((size_t)K*sizeof(float)):NULL;
    if(!qf32 || (nf4&&!qidx) || (imatrix&&!imdq)){ free(qf32); free(qidx); free(imdq); return 0; }
    for(int n=0;n<N;n++){
        const float *fr=f32+(size_t)n*K; float mx=1e-9f; int k=0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        float32x4_t vmx=vdupq_n_f32(1e-9f);
        for(;k<=K-4;k+=4) vmx=vmaxq_f32(vmx,vabsq_f32(vld1q_f32(fr+k)));
        float m[4]; vst1q_f32(m,vmx); float a=m[0]>m[1]?m[0]:m[1], bb=m[2]>m[3]?m[2]:m[3]; mx=a>bb?a:bb;
#endif
        for(;k<K;k++){ float v=fabsf(fr[k]); if(v>mx) mx=v; }
        if(imatrix) mx=wq_best_absmax(fr,K,mx,nf4,imatrix,imdq);
        uint8_t *nib=Bi4+(size_t)n*(K/2);
        if(nf4){ bscale[n]=mx/127.0f; quant_chan_nf4(fr,K,mx,sr,&seed,nib,qidx); }
        else   { float scale=mx/7.0f; bscale[n]=scale; quant_chan_i4(fr,K,scale,sr,&seed,nib,qf32); }
    }
    memcpy(p,&h,hdr);   /* header last: bscale/Bi4 already in place */
    free(qf32); free(qidx); free(imdq);
    return need;
}
