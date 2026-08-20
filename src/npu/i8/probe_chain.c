/* npu/i8/probe_chain.c — int8 PC-CHAIN probes: the chain benchmark, the generic ork_npu_chain_progs runner, and the heterogeneous / SDP-forward / self-test chain probes.
 *
 * Part of the int8 (W8A8) datapath. Split out of the 1,270-line npu/i8/probe.c (MODULARIZE_PLAN.md
 * round 6) as a CONTIGUOUS line range, so the pieces concatenate back to the original byte for byte.
 * Interface types in npu/internal.h, substrate in npu/core.h. Board-only diagnostics. */
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
#include "npu/internal.h"
#include "npu/core.h"
#include "npu/i8/i8.h"
#include "spine_kernels.h"
int ork_npu_benchmark_chain(ork_npu *c, int S, int K, int N, int iters) {
    int fd = c->fd, CBUF = c->soc->cbuf_elems;
    if (K % 32 || N % 32 || N > c->soc->nmax || S < 1 || S > 64) return -2;
    
    struct buf W = orki_bcreate(fd, (size_t)K * N, 0x403,-1);
    struct buf A = orki_bcreate(fd, (size_t)S * K, 0x403,-1);
    struct buf O = orki_bcreate(fd, (size_t)S * 4096, 0x403,-1);
    
    struct buf regs_chain = orki_bcreate(fd, (size_t)S * REGCMD_I8_N * 4, 0x403,-1);
    struct buf regs_sep = orki_bcreate(fd, (size_t)S * REGCMD_I8_N * 4, 0x403,-1);
    
    struct buf task_chain = orki_bcreate(fd, (size_t)S * sizeof(struct rknpu_task), 0x40b,-1);
    struct buf task_sep = orki_bcreate(fd, (size_t)S * sizeof(struct rknpu_task), 0x40b,-1);
    
    if (!W.cpu || !A.cpu || !O.cpu || !regs_chain.cpu || !regs_sep.cpu || !task_chain.cpu || !task_sep.cpu) {
        fprintf(stderr, "[ork] ERROR: failed to allocate benchmark_chain buffers (IOMMU full?)\n");
        orki_bdestroy(fd, &W); orki_bdestroy(fd, &A); orki_bdestroy(fd, &O);
        orki_bdestroy(fd, &regs_chain); orki_bdestroy(fd, &regs_sep);
        orki_bdestroy(fd, &task_chain); orki_bdestroy(fd, &task_sep);
        return -2;
    }
    
    memset(W.cpu, 1, (size_t)K * N);
    memset(A.cpu, 1, (size_t)S * K);
    orki_bsync(fd, &W, RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd, &A, RKNPU_MEM_SYNC_TO_DEVICE);
    
    uint32_t rc[REGCMD_I8_N];
    for (int i = 0; i < S; i++) {
        uint32_t act_dma = (uint32_t)(A.dma + i * K);
        uint32_t out_dma = (uint32_t)(O.dma + i * 4096);
        orki_synth_i8(rc, 1, K, N, act_dma, (uint32_t)W.dma, out_dma, 1, CBUF, 0);
        orki_setrn(rc, REGCMD_I8_N,RK_CNA_CBUF_CON0, 0xb1);
        
        if (i < S - 1) {
            uint64_t next_dma = regs_chain.dma + (i + 1) * REGCMD_I8_N * 4;
            rc[216] = 0x0010 | ((next_dma & 0xffff) << 16);
            rc[217] = (0x0101 << 16) | ((next_dma >> 16) & 0xffff);
            rc[218] = 0x0014 | (0x0037 << 16);
            rc[219] = (0x0101 << 16) | (0);
        } else {
            rc[216] = 0; rc[217] = 0; rc[218] = 0x00000014; rc[219] = 0x01010000;
        }
        memcpy((char*)regs_chain.cpu + i * REGCMD_I8_N * 4, rc, sizeof(rc));
    }
    orki_bsync(fd, &regs_chain, RKNPU_MEM_SYNC_TO_DEVICE);
    
    for (int i = 0; i < S; i++) {
        uint32_t act_dma = (uint32_t)(A.dma + i * K);
        uint32_t out_dma = (uint32_t)(O.dma + i * 4096);
        orki_synth_i8(rc, 1, K, N, act_dma, (uint32_t)W.dma, out_dma, 1, CBUF, 0);
        orki_setrn(rc, REGCMD_I8_N,RK_CNA_CBUF_CON0, 0xb1);
        rc[216] = 0; rc[217] = 0; rc[218] = 0x00000014; rc[219] = 0x01010000;
        memcpy((char*)regs_sep.cpu + i * REGCMD_I8_N * 4, rc, sizeof(rc));
    }
    orki_bsync(fd, &regs_sep, RKNPU_MEM_SYNC_TO_DEVICE);
    
    struct rknpu_task *tk_chain = task_chain.cpu;
    memset(tk_chain, 0, S * sizeof(struct rknpu_task));
    for (int i = 0; i < S; i++) {
        tk_chain[i].enable_mask = 0xd;
        tk_chain[i].int_mask = 0x300;
        tk_chain[i].int_clear = 0x1ffff;
        tk_chain[i].regcfg_amount = 108;
        tk_chain[i].regcmd_addr = regs_chain.dma + i * REGCMD_I8_N * 4;
    }
    orki_bsync(fd, &task_chain, RKNPU_MEM_SYNC_TO_DEVICE);
    
    struct rknpu_task *tk_sep = task_sep.cpu;
    memset(tk_sep, 0, S * sizeof(struct rknpu_task));
    for (int i = 0; i < S; i++) {
        tk_sep[i].enable_mask = 0xd;
        tk_sep[i].int_mask = 0x300;
        tk_sep[i].int_clear = 0x1ffff;
        tk_sep[i].regcfg_amount = 108;
        tk_sep[i].regcmd_addr = regs_sep.dma + i * REGCMD_I8_N * 4;
    }
    orki_bsync(fd, &task_sep, RKNPU_MEM_SYNC_TO_DEVICE);
    
    orki_act(fd, RKNPU_ACT_RESET, 0);
    struct rknpu_submit sub; memset(&sub, 0, sizeof(sub));
    sub.flags = ork_ppflags();
    sub.task_number = S;
    sub.task_obj_addr = task_chain.obj;
    sub.core_mask = RKNPU_CORE0_MASK;
    sub.fence_fd = -1;
    sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)S};
    sub.timeout = orki_mm_timeout_ms();
    if (orki_rknpu_submit_ioctl(fd, &sub, -1)) {
        perror("Warmup failed");
    }
    orki_bsync(fd, &O, RKNPU_MEM_SYNC_FROM_DEVICE);
    
    double t_sep_start = ork_now_us();
    for (int it = 0; it < iters; it++) {
        for (int s = 0; s < S; s++) {
            struct rknpu_task *tk_dest = c->task.cpu;
            memcpy(tk_dest, &tk_sep[s], sizeof(struct rknpu_task));
            orki_bsync(fd, &c->task, RKNPU_MEM_SYNC_TO_DEVICE);
            
            struct rknpu_submit sub_s; memset(&sub_s, 0, sizeof(sub_s));
            sub_s.flags = 0x5;
            sub_s.task_number = 1;
            sub_s.task_obj_addr = c->task.obj;
            sub_s.core_mask = RKNPU_CORE0_MASK;
            sub_s.fence_fd = -1;
            sub_s.subcore_task[0] = sub_s.subcore_task[1] = sub_s.subcore_task[2] = (struct rknpu_subcore_task){0, 1};
            sub_s.timeout = 60000;
            if (orki_rknpu_submit_ioctl(fd, &sub_s, -1)) {
                perror("Separate submit failed");
                break;
            }
            orki_bsync(fd, &O, RKNPU_MEM_SYNC_FROM_DEVICE);
        }
    }
    double t_sep = ork_now_us() - t_sep_start;
    
    double t_chain_start = ork_now_us();
    for (int it = 0; it < iters; it++) {
        struct rknpu_submit sub_c; memset(&sub_c, 0, sizeof(sub_c));
        sub_c.flags = 0x5;
        sub_c.task_number = S;
        sub_c.task_obj_addr = task_chain.obj;
        sub_c.core_mask = RKNPU_CORE0_MASK;
        sub_c.fence_fd = -1;
        sub_c.subcore_task[0] = sub_c.subcore_task[1] = sub_c.subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)S};
        sub_c.timeout = 60000;
        if (orki_rknpu_submit_ioctl(fd, &sub_c, -1)) {
            perror("Chained submit failed");
            break;
        }
        orki_bsync(fd, &O, RKNPU_MEM_SYNC_FROM_DEVICE);
    }
    double t_chain = ork_now_us() - t_chain_start;
    
    double avg_sep = t_sep / iters;
    double avg_chain = t_chain / iters;
    
    printf("  %d separate submits: total time = %.0f us (avg per submit loop = %.1f us, per matmul = %.1f us)\n",
           S, t_sep, avg_sep, avg_sep / S);
    printf("  1 chained submit (x%d): total time = %.0f us (avg per submit loop = %.1f us, per matmul = %.1f us)\n",
           S, t_chain, avg_chain, avg_chain / S);
    printf("  Speedup: %.2fx\n", avg_sep / avg_chain);
    
    orki_bdestroy(fd, &W); orki_bdestroy(fd, &A); orki_bdestroy(fd, &O);
    orki_bdestroy(fd, &regs_chain); orki_bdestroy(fd, &regs_sep);
    orki_bdestroy(fd, &task_chain); orki_bdestroy(fd, &task_sep);
    return 0;
}

/* CHAIN ASSEMBLER self-test: chain TWO plain int8 matmuls via ork_npu_chain_progs and verify BOTH tasks
 * EXECUTE + produce output -- the exact thing Phase-0 could not (an early task0 that actually runs). Uses
 * WORKING primitives only (synth_i8 native int32 output; no set_i16_out). Layout-agnostic: A=all-1, W0=all-1,
 * W1=all-2 -> every C0 element == K, every C1 element == 2K, regardless of output tiling. Fills *t0_cnt/
 * *t1_cnt = count of the M*N int32 slots matching K / 2K (near M*N => that task ran; 0 => it didn't).
 * 0/ok, -1 wedge, -2 dims/alloc. rk3588. */
int ork_npu_chain_progs(ork_npu *c, int n, const ork_chain_prog *progs, int dom){
    if(!c||!progs||n<1||n>1024) return -2;
    int fd=c->fd;
    /* warm-state management (mirror run_chain_i8): entering a chain from a non-int8-live mode resets +
     * unwarms so the WARM-UP reps below fire; DT_I8<->DT_I8_CHAIN is not a real mode change (keepwarm). */
    ork_npu_enter(c, 3 /* DT_I8_CHAIN */, XP_CHAIN_NT, OCK_HW);
    size_t off[1024], total=0;
    /* Each task's regcmd MUST start on a 64-byte (16-word) boundary — the HW chain-walk hangs "entering" a
     * misaligned successor task (vendor RE: tight packing landed a task at +80 mod 128 and hung; the vendor
     * lays every task 64B-aligned). Matmul-only chains never tripped this (REGCMD_I8_N=224w=896B is 64B-aligned,
     * so contiguous packing stays aligned), but a 146-word SDP task (584B) knocks every following task off the
     * boundary. Round each task's start up to 16 words. */
    for(int i=0;i<n;i++){ if(!progs[i].rc||progs[i].nwords<2) return -2; total=(total+15)&~(size_t)15; off[i]=total; total+=(size_t)progs[i].nwords; }
    if(total*4 > c->regcmd.size || (size_t)n*sizeof(struct rknpu_task) > c->task.size) return -2;
    uint32_t *base=(uint32_t*)c->regcmd.cpu;
    for(int i=0;i<n;i++){
        memcpy(base+off[i], progs[i].rc, (size_t)progs[i].nwords*4);
        if(i<n-1){
            /* WRITE this program's PC next-descriptor at its designated slot (like run_chain_i8's word 216).
             * The slot is not a pre-existing pattern in the template -- the chaining code creates it. */
            int slot=progs[i].desc_slot;
            if(slot<0 || slot+3>=progs[i].nwords) return -2;   /* this op can't be a MIDDLE program */
            uint32_t *rc=base+off[i];
            uint64_t nx=(uint64_t)c->regcmd.dma + off[i+1]*4; int nreg=(progs[i+1].regcfg_amount+3)/2;
            rc[slot]  =0x0010 | ((uint32_t)(nx&0xffff)<<16);
            rc[slot+1]=(0x0101u<<16) | (uint32_t)((nx>>16)&0xffff);
            rc[slot+2]=0x0014 | ((uint32_t)nreg<<16);
            rc[slot+3]=(0x0101u<<16);
        }
    }
    orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *t=c->task.cpu; memset(t,0,(size_t)n*sizeof(struct rknpu_task));
    for(int i=0;i<n;i++){ t[i].enable_mask=progs[i].enable_mask; t[i].int_mask=0x300; t[i].int_clear=0x1ffff;
        t[i].regcfg_amount=progs[i].regcfg_amount; t[i].regcmd_addr=(uint32_t)((uint64_t)c->regcmd.dma + off[i]*4); }
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    /* ping-pong (ork_ppflags, typically 0x5) is safe ONLY for register-config-only chains (all int8 matmul
     * tasks, like run_chain_i8); ANY SDP/LUT task (enable != 0xd) needs ping-pong OFF (0x1) so a bank swap
     * doesn't race a LUT SRAM commit (AGENTS.md "ping-pong OFF for LUT chains"). */
    int has_sdp=0; for(int i=0;i<n;i++) if(progs[i].enable_mask!=0xd) has_sdp=1;
    struct rknpu_submit s; memset(&s,0,sizeof s);
    s.flags = has_sdp ? 0x1 : ork_ppflags(); s.task_number=(uint32_t)n; s.task_obj_addr=c->task.obj;
    s.core_mask=RKNPU_CORE0_MASK; s.fence_fd=-1; s.timeout=orki_ew_timeout_ms();
    s.subcore_task[0]=s.subcore_task[1]=s.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)n};
    /* WARM-UP: a COLD int8-matmul chain submit yields EMPTY output (run_chain_i8 reps=2 cold). Submit reps
     * times; the last produces the result. c->warmed is managed by the DT_I8_CHAIN block at entry. */
    int reps = c->warmed ? 1 : 2, rr=0;
    if(getenv("ORK_CHAIN_DBG")) fprintf(stderr,"[chain_progs] n=%d dom=%d flags=0x%x warmed(pre)=%d reps=%d regcmd.dma=0x%llx task.obj=0x%llx | "
        "t0{en=0x%x rcfg=%d addr=0x%x} t%d{en=0x%x rcfg=%d addr=0x%x}\n", n,dom,s.flags,reps,reps,(unsigned long long)c->regcmd.dma,(unsigned long long)c->task.obj,
        t[0].enable_mask,t[0].regcfg_amount,t[0].regcmd_addr, n-1,t[n-1].enable_mask,t[n-1].regcfg_amount,t[n-1].regcmd_addr);
    for(int rep=0; rep<reps; rep++){ int e=orki_rknpu_submit_ioctl(fd,&s,dom); rr = e?-1:0;
        if(getenv("ORK_CHAIN_DBG")) fprintf(stderr,"[chain_progs] submit rep %d -> %d (errno=%d)\n",rep,e,errno); }
    c->warmed = 1;
    return rr;
}

int ork_npu_probe_seq_hetero(ork_npu *c, int *ok){
    if(ok)*ok=0;
    if(!c||!ork_ppu_fuse_enabled(c)) return -3;
    int fd=c->fd, M=8, K=512, N=64, mult=0x4000, shift=14, CBUF=c->soc->cbuf_elems;
    #define SEWC(m,n) (((n)/16)*(M*16) + (m)*16 + ((n)%16))
    if(orki_mc_ensure(c,1)) return -1;
    ork_npu_enter(c, 3 /*DT_I8_CHAIN*/, XP_CHAIN_NT, OCK_HW);
    /* pack an all-ones int8 weight [K,N] -> C = K everywhere */
    int8_t *wb=malloc((size_t)K*N); if(!wb) return -2; for(int i=0;i<K*N;i++) wb[i]=1;
    ork_w *w=ork_mm_pack_i8(c,K,N,wb); free(wb); if(!w) return -2;
    uint32_t wdma = w->Bf ? (uint32_t)w->Bf[0].dma : (uint32_t)w->Bb[0].dma;
    /* ewmul inputs + CPU ref (int8) */
    int8_t r1[512],s1[512],ref1[512]; uint32_t g=555;
    for(int i=0;i<M*N;i++){ r1[i]=(int8_t)(((g=g*1103515245u+12345u)>>20&0x7))-3; s1[i]=(int8_t)(((g=g*1103515245u+12345u)>>20&0x7))-3; }
    for(int i=0;i<M*N;i++){ long v=lround((long)r1[i]*s1[i]*mult/(double)(1<<shift)); ref1[i]=(int8_t)(v>127?127:v<-128?-128:v); }
    /* stage into maf[0]: matmul A [M,K] all-ones @0 (shared by both matmuls); ewmul A/B cube-laid after it */
    struct buf *AF=&c->maf[0], *RC=&c->mrc[0];
    size_t offA=0, offEwA=(size_t)M*K, offEwB=offEwA+(size_t)M*N;
    if(offEwB+(size_t)M*N > AF->size) { return -2; }
    memset(AF->cpu,0,offEwB+(size_t)M*N);
    { int8_t*a=(int8_t*)AF->cpu; for(int i=0;i<M*K;i++) a[offA+i]=1;
      for(int m=0;m<M;m++)for(int n=0;n<N;n++){ a[offEwA+SEWC(m,n)]=r1[m*N+n]; a[offEwB+SEWC(m,n)]=s1[m*N+n]; } }
    orki_bsync(fd,AF,RKNPU_MEM_SYNC_TO_DEVICE);
    /* output scratch: matmul0 [M,N]i32 @0, ewmul [M,N]i8 @2048, matmul2 [M,N]i32 @2560 */
    size_t oMM0=0, oEW=(size_t)M*N*4, oMM2=oEW+(size_t)M*N;
    struct buf OUT=orki_bcreate(fd,8192,0x403,c->dom_active); if(!OUT.cpu) return -2;
    memset(OUT.cpu,0,8192);
    uint32_t o0=(uint32_t)(OUT.dma+oMM0), oe=(uint32_t)(OUT.dma+oEW), o2=(uint32_t)(OUT.dma+oMM2);
    /* build 3 programs at 224-word (64B-aligned) slots in mrc[0] */
    uint32_t *base=(uint32_t*)RC->cpu; memset(base,0,3*(size_t)REGCMD_I8_N*4);
    uint32_t am=(uint32_t)(AF->dma+offA);
    { uint32_t rc[REGCMD_I8_N]; memset(rc,0,sizeof rc);
      orki_synth_i8(rc,M,K,N,am,wdma,o0,1,CBUF,0);                                  /* prog0 matmul -> o0 */
      uint64_t nx=RC->dma + (size_t)1*REGCMD_I8_N*4; int amt=(69+3)/2;         /* -> prog1 (SDP regcfg 69) */
      rc[216]=0x0010|((uint32_t)(nx&0xffff)<<16); rc[217]=(0x0101u<<16)|(uint32_t)((nx>>16)&0xffff);
      rc[218]=0x0014|((uint32_t)amt<<16);         rc[219]=(0x0101u<<16);
      memcpy(base+0*REGCMD_I8_N, rc, REGCMD_I8_N*4); }
    { uint32_t rc[REGCMD_MUL_N]; memcpy(rc,REGCMD_MUL,sizeof rc); orki_set_mul_geom(rc,REGCMD_MUL_N,M,N);
      orki_setrn(rc,REGCMD_MUL_N,RK_DPU_DST_BASE_ADDR,oe); orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5018,(uint32_t)(AF->dma+offEwA)); orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5038,(uint32_t)(AF->dma+offEwB));
      orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)mult); orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)shift);
      orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_OFFSET,0); orki_setrn(rc,REGCMD_MUL_N,RK_DPU_BS_ALU_CFG,0); orki_setrn(rc,REGCMD_MUL_N,RK_DPU_EW_CVT_OFFSET,0);
      uint64_t nx=RC->dma + (size_t)2*REGCMD_I8_N*4; int amt=(108+3)/2;        /* -> prog2 (matmul regcfg 108) */
      rc[138]=0x0010|((uint32_t)(nx&0xffff)<<16); rc[139]=(0x0101u<<16)|(uint32_t)((nx>>16)&0xffff);
      rc[140]=0x0014|((uint32_t)amt<<16);         rc[141]=(0x0101u<<16);
      memcpy(base+1*REGCMD_I8_N, rc, REGCMD_MUL_N*4); }
    { uint32_t rc[REGCMD_I8_N]; memset(rc,0,sizeof rc);
      orki_synth_i8(rc,M,K,N,am,wdma,o2,1,CBUF,0);                                  /* prog2 matmul -> o2 (TERMINAL) */
      memcpy(base+2*REGCMD_I8_N, rc, REGCMD_I8_N*4); }
    orki_bsync(fd,RC,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->mtk[0].cpu; memset(tk,0,3*sizeof *tk);
    tk[0].enable_mask=0xd;  tk[0].int_mask=0x300; tk[0].int_clear=0x1ffff; tk[0].regcfg_amount=108; tk[0].regcmd_addr=(uint32_t)(RC->dma+0*REGCMD_I8_N*4);
    tk[1].enable_mask=0x18; tk[1].int_mask=0x300; tk[1].int_clear=0x1ffff; tk[1].regcfg_amount=69;  tk[1].regcmd_addr=(uint32_t)(RC->dma+1*REGCMD_I8_N*4);
    tk[2].enable_mask=0xd;  tk[2].int_mask=0x300; tk[2].int_clear=0x1ffff; tk[2].regcfg_amount=108; tk[2].regcmd_addr=(uint32_t)(RC->dma+2*REGCMD_I8_N*4);
    orki_bsync(fd,&c->mtk[0],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    /* clean-before: whole OUT to DRAM (no dirty CPU line evicts over the NPU writes — begin_mc's cold recipe) */
    orki_bsync(fd,&OUT,RKNPU_MEM_SYNC_TO_DEVICE);
    /* seed the TERMINAL matmul (prog2 @ o2) last-col-per-row int32 sentinel */
    volatile int32_t *t2=(volatile int32_t*)((char*)OUT.cpu+oMM2);
    for(int m=0;m<M;m++){ volatile int32_t*db=&t2[(size_t)m*N+(N-1)]; *db=ORK_DYN_SENT; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); }
    __asm__ volatile("dsb ish":::"memory");
    struct rknpu_submit s; memset(&s,0,sizeof s);
    s.flags=0x1u|0x2u;   /* PC | NONBLOCK; ping-pong OFF (SDP present) */
    s.task_number=3; s.task_obj_addr=c->mtk[0].obj; s.core_mask=RKNPU_CORE0_MASK; s.fence_fd=-1; s.timeout=orki_mm_timeout_ms();
    s.subcore_task[0]=(struct rknpu_subcore_task){0,3};
    int e=orki_rknpu_submit_ioctl(fd,&s,c->dom_active);
    int okall=0;
    if(e==0){ double t0=ork_now_us(); int landed=0;
        for(;;){ int done=1; for(int m=0;m<M;m++){ volatile int32_t*db=&t2[(size_t)m*N+(N-1)]; __asm__ volatile("dc civac,%0"::"r"(db):"memory"); if(*db==ORK_DYN_SENT){done=0;break;} } if(done){landed=1;break;} if(ork_now_us()-t0>3e6)break; }
        if(landed){ orki_bsync(fd,&OUT,RKNPU_MEM_SYNC_FROM_DEVICE);
            int32_t*c0=(int32_t*)((char*)OUT.cpu+oMM0), *c2=(int32_t*)((char*)OUT.cpu+oMM2); int8_t*ew=(int8_t*)((char*)OUT.cpu+oEW);
            int n0=0,n2=0,ne=0;
            for(int i=0;i<M*N;i++){ if(c0[i]==K)n0++; if(c2[i]==K)n2++; }
            for(int m=0;m<M;m++)for(int n=0;n<N;n++) if(ew[SEWC(m,n)]==ref1[m*N+n]) ne++;
            if(getenv("ORK_SEQ_DBG")) fprintf(stderr,"[seq-hetero] matmul0 #==K=%d/%d  ewmul #match=%d/%d  matmul2 #==K=%d/%d\n",n0,M*N,ne,M*N,n2,M*N);
            okall = (n0==M*N) && (ne==M*N) && (n2==M*N); } }
    if(ok)*ok=okall;
    orki_bdestroy(fd,&OUT);
    #undef SEWC
    return e?-1:0;
}

/* PROBE (int8 SDP on the HW-chain): can a standalone SDP op (ewmul, enable=0x18, regcfg=69) be a MIDDLE program
 * in a PC-chain, walking FORWARD through its next-descriptor? Decode of the vendor's working SDP chain
 * (regcmd_softmax_f16.h SM_TASK0) proved REGCMD_MUL is ALREADY chain-native: its tail (words 138..145) is
 * byte-identical in STRUCTURE to SM_TASK0 — a terminal descriptor at word 138 (=2*regcfg; next-addr 0 / amt 0)
 * followed by the 0x0041/0x0018/0x0081 op-enable trailer. So the port is NOT a template change; it is feeding
 * SDP progs (desc_slot=138) through the PROVEN ork_npu_chain_progs (which already handles SDP: per-prog
 * desc_slot + has_sdp ping-pong-off + reps=2 cold warm-up). Chains [ewmul0(desc_slot=138) -> ewmul1(last)] and
 * verifies BOTH outputs vs the CPU ref: *t0_ok = the middle SDP op computed correctly carrying a forward
 * descriptor; *t1_ok = the chain WALKED forward through the SDP op's slot. Both ok => int8 SDP HW-chains. */
int ork_npu_probe_sdp_chain_fwd(ork_npu *c, int *t0_ok, int *t1_ok){
    if(t0_ok)*t0_ok=0; if(t1_ok)*t1_ok=0;
    if(!c||!ork_ppu_fuse_enabled(c)) return -3;
    int fd=c->fd, M=8, N=64, mult=0x4000, shift=14;
    #define EWC(m,n) (((n)/16)*(M*16) + (m)*16 + ((n)%16))
    size_t sz=(size_t)M*N; if(sz<4096)sz=4096;
    struct buf A0=orki_bcreate(fd,sz,0x403,-1),B0=orki_bcreate(fd,sz,0x403,-1),O0=orki_bcreate(fd,sz,0x403,-1);
    struct buf A1=orki_bcreate(fd,sz,0x403,-1),B1=orki_bcreate(fd,sz,0x403,-1),O1=orki_bcreate(fd,sz,0x403,-1);
    if(!A0.cpu||!B0.cpu||!O0.cpu||!A1.cpu||!B1.cpu||!O1.cpu){ orki_bdestroy(fd,&A0);orki_bdestroy(fd,&B0);orki_bdestroy(fd,&O0);orki_bdestroy(fd,&A1);orki_bdestroy(fd,&B1);orki_bdestroy(fd,&O1); return -2; }
    int8_t r0[512],s0[512],r1[512],s1[512],ref0[512],ref1[512]; uint32_t g=12345;
    for(int i=0;i<M*N;i++){ r0[i]=(int8_t)(((g=g*1103515245u+12345u)>>20&0x7))-3; s0[i]=(int8_t)(((g=g*1103515245u+12345u)>>20&0x7))-3;
                            r1[i]=(int8_t)(((g=g*1103515245u+12345u)>>20&0x7))-3; s1[i]=(int8_t)(((g=g*1103515245u+12345u)>>20&0x7))-3; }
    for(int i=0;i<M*N;i++){ long v0=lround((long)r0[i]*s0[i]*mult/(double)(1<<shift)),v1=lround((long)r1[i]*s1[i]*mult/(double)(1<<shift));
                            ref0[i]=(int8_t)(v0>127?127:v0<-128?-128:v0); ref1[i]=(int8_t)(v1>127?127:v1<-128?-128:v1); }
    memset(A0.cpu,0,sz);memset(B0.cpu,0,sz);memset(O0.cpu,0,sz);memset(A1.cpu,0,sz);memset(B1.cpu,0,sz);memset(O1.cpu,0,sz);
    int8_t*a0=A0.cpu,*b0=B0.cpu,*a1=A1.cpu,*b1=B1.cpu;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ a0[EWC(m,n)]=r0[m*N+n]; b0[EWC(m,n)]=s0[m*N+n]; a1[EWC(m,n)]=r1[m*N+n]; b1[EWC(m,n)]=s1[m*N+n]; }
    orki_bsync(fd,&A0,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&B0,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O0,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&A1,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&B1,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O1,RKNPU_MEM_SYNC_TO_DEVICE);
    /* build the two ewmul regcmds exactly like the standalone ork_npu_ewmul_i8 (geom + addrs + scale) */
    uint32_t rc0[REGCMD_MUL_N],rc1[REGCMD_MUL_N];
    memcpy(rc0,REGCMD_MUL,sizeof rc0); orki_set_mul_geom(rc0,REGCMD_MUL_N,M,N);
    orki_setrn(rc0,REGCMD_MUL_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O0.dma); orki_setrn(rc0,REGCMD_MUL_N,RK_SDP_5018,(uint32_t)A0.dma); orki_setrn(rc0,REGCMD_MUL_N,RK_SDP_5038,(uint32_t)B0.dma);
    orki_setrn(rc0,REGCMD_MUL_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)mult); orki_setrn(rc0,REGCMD_MUL_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)shift);
    orki_setrn(rc0,REGCMD_MUL_N,RK_DPU_OUT_CVT_OFFSET,0); orki_setrn(rc0,REGCMD_MUL_N,RK_DPU_BS_ALU_CFG,0); orki_setrn(rc0,REGCMD_MUL_N,RK_DPU_EW_CVT_OFFSET,0);
    memcpy(rc1,REGCMD_MUL,sizeof rc1); orki_set_mul_geom(rc1,REGCMD_MUL_N,M,N);
    orki_setrn(rc1,REGCMD_MUL_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O1.dma); orki_setrn(rc1,REGCMD_MUL_N,RK_SDP_5018,(uint32_t)A1.dma); orki_setrn(rc1,REGCMD_MUL_N,RK_SDP_5038,(uint32_t)B1.dma);
    orki_setrn(rc1,REGCMD_MUL_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)mult); orki_setrn(rc1,REGCMD_MUL_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)shift);
    orki_setrn(rc1,REGCMD_MUL_N,RK_DPU_OUT_CVT_OFFSET,0); orki_setrn(rc1,REGCMD_MUL_N,RK_DPU_BS_ALU_CFG,0); orki_setrn(rc1,REGCMD_MUL_N,RK_DPU_EW_CVT_OFFSET,0);
    /* THE PORT: SDP progs through the proven chainer. ewmul0 is a MIDDLE program (desc_slot=138 = the chain-native
     * descriptor slot decoded from SM_TASK0); ewmul1 is the terminal (desc_slot=-1). chain_progs writes ewmul0's
     * forward descriptor at 138, detects enable!=0xd -> ping-pong OFF, and does the reps=2 cold warm-up. */
    /* HYPOTHESIS: a HW chain must BEGIN with a matmul (enable=0xd) task — the kernel programs the PC from task0,
     * and every working chain (FFN, chain_mm_silu) starts with a matmul; an SDP task0 (this probe's earlier form,
     * and the vendor's hardware-chained softmax) HANGS. So prepend an all-ones matmul task0; ewmul0 becomes a true
     * MIDDLE SDP task (carries the fwd descriptor at 138 like the FFN chain's silu). ORK_SDP_NOMM=1 = old SDP-first
     * form (control). Matmul: M=8,K=64,N=64, A=c->Af all-ones, W all-ones -> C=K=64 (sanity, not read by ewmul). */
    int nomm=!!getenv("ORK_SDP_NOMM"); int CBUF=c->soc->cbuf_elems, MK=8*64, KN=64*64;
    struct buf W=orki_bcreate(fd,(size_t)KN,0x403,-1), C=orki_bcreate(fd,(size_t)8*64*4,0x403,-1);
    static uint32_t rmm[REGCMD_I8_N];
    if(!nomm){
        if(!W.cpu||!C.cpu){ orki_bdestroy(fd,&W);orki_bdestroy(fd,&C);orki_bdestroy(fd,&A0);orki_bdestroy(fd,&B0);orki_bdestroy(fd,&O0);orki_bdestroy(fd,&A1);orki_bdestroy(fd,&B1);orki_bdestroy(fd,&O1); return -2; }
        { int8_t*wb=W.cpu; for(int i=0;i<KN;i++)wb[i]=1; int8_t*ad=c->Af.cpu; for(int i=0;i<MK;i++)ad[i]=1; }
        memset(C.cpu,0,(size_t)8*64*4);
        orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&C,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
        orki_act(fd,RKNPU_ACT_RESET,0);
        orki_synth_i8(rmm,8,64,64,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)C.dma,1,CBUF,0);
    }
    ork_chain_prog progs[3]={ {rmm,REGCMD_I8_N,0xd,108,216}, {rc0,REGCMD_MUL_N,0x18,69,138}, {rc1,REGCMD_MUL_N,0x18,69,-1} };
    /* dom=-1 (default domain) to MATCH the orki_bcreate(...,-1) buffers + c->regcmd/c->task (a domain-0 submit
     * mismatches them, see ork_npu_chain_selftest). nomm control: SDP-first 2-task chain (expected to hang). */
    int rc = nomm ? ork_npu_chain_progs(c,2,progs+1,-1) : ork_npu_chain_progs(c,3,progs,-1);
    if(rc==0){ orki_bsync(fd,&O0,RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd,&O1,RKNPU_MEM_SYNC_FROM_DEVICE);
        int ok0=1,ok1=1;
        for(int m=0;m<M;m++)for(int n=0;n<N;n++){ if(*(int8_t*)((char*)O0.cpu+EWC(m,n))!=ref0[m*N+n])ok0=0; if(*(int8_t*)((char*)O1.cpu+EWC(m,n))!=ref1[m*N+n])ok1=0; }
        if(t0_ok)*t0_ok=ok0; if(t1_ok)*t1_ok=ok1; }
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&C);
    orki_bdestroy(fd,&A0);orki_bdestroy(fd,&B0);orki_bdestroy(fd,&O0);orki_bdestroy(fd,&A1);orki_bdestroy(fd,&B1);orki_bdestroy(fd,&O1);
    #undef EWC
    return rc;
}

int ork_npu_chain_selftest(ork_npu *c, int *t0_cnt, int *t1_cnt){
    /* dom=-1 (default domain) to match the WORKING raw-submit paths (probe_i8_mm, softmax_replay); the init
     * buffers c->Af/c->regcmd/c->task live in the default domain, so a domain-0 submit mismatches them. */
    int fd=c->fd, CBUF=c->soc->cbuf_elems, dom=-1;
    const int M=8, K=64, N=64;
    struct buf W0=orki_bcreate(fd,(size_t)K*N,0x403,dom), W1=orki_bcreate(fd,(size_t)K*N,0x403,dom);
    struct buf C0=orki_bcreate(fd,(size_t)M*N*4,0x403,dom), C1=orki_bcreate(fd,(size_t)M*N*4,0x403,dom);
    if(!W0.cpu||!W1.cpu||!C0.cpu||!C1.cpu){ orki_bdestroy(fd,&W0);orki_bdestroy(fd,&W1);orki_bdestroy(fd,&C0);orki_bdestroy(fd,&C1); return -2; }
    { int8_t*b0=W0.cpu,*b1=W1.cpu; for(size_t i=0;i<(size_t)K*N;i++){ b0[i]=1; b1[i]=2; } }   /* uniform -> tile layout irrelevant */
    memset(C0.cpu,0,(size_t)M*N*4); memset(C1.cpu,0,(size_t)M*N*4);
    { int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=1; }
    orki_bsync(fd,&W0,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&W1,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&C0,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&C1,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    static uint32_t r0[REGCMD_I8_N], r1[REGCMD_I8_N];
    orki_synth_i8(r0,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W0.dma,(uint32_t)C0.dma,1,CBUF,0);
    orki_synth_i8(r1,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W1.dma,(uint32_t)C1.dma,1,CBUF,0);
    ork_chain_prog progs[2]={ {r0,REGCMD_I8_N,0xd,108,216}, {r1,REGCMD_I8_N,0xd,108,-1} };
    int nprog = getenv("ORK_GS_N1") ? 1 : 2;   /* ORK_GS_N1: single-task chain_progs (isolate chaining from the matmul itself) */
    int crc=ork_npu_chain_progs(c,nprog,progs,dom);
    if(crc){ orki_bdestroy(fd,&W0);orki_bdestroy(fd,&W1);orki_bdestroy(fd,&C0);orki_bdestroy(fd,&C1); return crc==-1?-1:-2; }
    orki_bsync(fd,&C0,RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd,&C1,RKNPU_MEM_SYNC_FROM_DEVICE);
    int n0=0,n1=0; int32_t*c0=C0.cpu,*c1=C1.cpu;
    int32_t mx0=0,mx1=0; for(int i=0;i<M*N;i++){ if(c0[i]==K)n0++; if(c1[i]==2*K)n1++;
        if(c0[i]>mx0)mx0=c0[i]; if(c1[i]>mx1)mx1=c1[i]; }
    fprintf(stderr,"[selftest] K=%d 2K=%d | C0 max=%d first=[%d %d %d %d] | C1 max=%d first=[%d %d %d %d]\n",
            K,2*K,mx0,c0[0],c0[1],c0[2],c0[3],mx1,c1[0],c1[1],c1[2],c1[3]);
    if(t0_cnt)*t0_cnt=n0; if(t1_cnt)*t1_cnt=n1;
    orki_bdestroy(fd,&W0);orki_bdestroy(fd,&W1);orki_bdestroy(fd,&C0);orki_bdestroy(fd,&C1);
    return 0;
}
