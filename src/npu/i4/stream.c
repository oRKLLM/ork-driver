/* npu/i4/stream.c — the async int4 stream and int4 probes.
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
                    orki_i4_tile_Aslice(abase + (size_t)(2 * j) * (K / 2), t->A + (size_t)(m0 + j) * K, 0, K);
                orki_bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE);
                memset(rc, 0, sizeof rc);
                orki_i4_synth(rc, 2 * H, K, N, (uint32_t)c->maf[i].dma, bdma, (uint32_t)c->mcc[i].dma);
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
        for (int m = 0; m < M; m++) orki_i4_tile_Aslice(abase + (size_t)m * K, t->A + (size_t)m * K, 0, K);
        orki_bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE);
        for (int m = 0; m < M; m++) {                         /* one single-row regcmd per row, PC-chained */
            memset(rc, 0, sizeof rc);
            orki_i4_synth(rc, 1, K, N, (uint32_t)(c->maf[i].dma + (size_t)m * K), bdma,
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

int ork_i4_mm_run_stream(ork_npu *c, int S, const ork_mm_task_i4 *tasks) {
    if(c && c->fd<0) return orki_cpu_chain_i4(S,tasks);   /* OFFLINE: no device to chain over */
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

void ork_i4_fuzz_clear(void){ orki_i4_fovr_n=0; }

void ork_i4_fuzz_add(uint32_t blk,uint32_t reg,uint32_t val){ if(orki_i4_fovr_n<16){ orki_i4_fovr[orki_i4_fovr_n].blk=blk; orki_i4_fovr[orki_i4_fovr_n].reg=reg; orki_i4_fovr[orki_i4_fovr_n].val=val; orki_i4_fovr_n++; } }

int ork_i4_npu_probe(ork_npu *c,int M,int K,int N,int nibB,int nibA,int nov,
                     const uint32_t *ovr_reg,const uint32_t *ovr_val,
                     const int8_t *A,const int8_t *B,int16_t *C){
    int fd=c->fd;
    if(K%32||N%64||N>c->soc->nmax) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N/2,0x403,-1); if(!W.cpu) return -2;        /* B int4: half bytes */
    orki_i4_tile_B(W.cpu,B,K,N,nibB);
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)M*N*2,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}  /* int16 C, M rows */
    /* M-tiling: the captured W4A4 program runs M=1 per task; we replicate it per row. Each row's A is
     * its own native (K/32,1,32) block (contiguous K/2 bytes); each row's C is (N/8,1,8) = N int16. */
    uint8_t*ad=c->Af.cpu;
    for(int m=0;m<M;m++) orki_i4_tile_A(ad+(size_t)m*(K/2), A+(size_t)m*K, 1, K, nibA);
    orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=0;
    for(int m=0;m<M && ok==0;m++){
        orki_act(fd,RKNPU_ACT_RESET,0);
        uint32_t rc[REGCMD_I4_N];
        orki_i4_synth(rc,1,K,N,(uint32_t)(c->Af.dma+(size_t)m*(K/2)),(uint32_t)W.dma,(uint32_t)(O.dma+(size_t)m*N*2));
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

int ork_i4_npu_probe_mm(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,int16_t *raw){
    int fd=c->fd;
    if(K%32||N%64||N>c->soc->nmax||M<1) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N/2,0x403,-1); if(!W.cpu) return -2;
    orki_i4_tile_B(W.cpu,B,K,N,0);
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)2*M*N*2,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}  /* 2x: stride-2 multi-M writes physical rows 0..2(M-1) */
    /* A layout selector via ORK_I4_ALAY: 0=(K/32,M,32) interleaved, 1=per-row contiguous (K/32,1,32)
     * x M (what the captured M=1 program reads). Lets the probe tell whether the program is single-row. */
    { int alay=getenv("ORK_I4_ALAY")?atoi(getenv("ORK_I4_ALAY")):0;
      if(alay) for(int m=0;m<M;m++) orki_i4_tile_A((uint8_t*)c->Af.cpu+(size_t)m*(K/2),A+(size_t)m*K,1,K,0);
      else orki_i4_tile_A(c->Af.cpu,A,M,K,0); }
    orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I4_N];
    orki_i4_synth(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma);
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
