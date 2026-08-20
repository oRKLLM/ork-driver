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

int g_i4_validate=-1;   /* ORK_I4_VALIDATE: per-program regcmd validation (DEBUG, off by default) */

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

int ork_mm_run_i4_experts(ork_npu *c, const ork_mm_task_i4 *ex, int ntask, int nc){
    if(!c || ntask<1 || !ex) return -1;
    for(int e=0;e<ntask;e++){ ork_w*w=ex[e].w;
        if(!w||w->dtype!=DT_I4||w->Sk!=1||w->Sn!=1||(w->N%64)||ex[e].M<1) return -2;
        if(w->domain!=ex[0].w->domain || w->K!=ex[0].w->K || w->N!=ex[0].w->N) return -2; }  /* one submit => one domain + one shape */
    return orki_run_i4_experts_bchain_db(c, ex, ntask, nc);   /* M-batched BCHAIN programs chained across experts */
}
