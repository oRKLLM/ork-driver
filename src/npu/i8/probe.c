/* npu/i8/probe.c — int8 RE probes, replays and benchmarks (board-only diagnostics).
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
#include "regcmd_i8.h"
#include "regcmd_i4.h"
#include "regcmd_silu.h"
#include "regcmd_ewmul.h"
#include "regcmd_fold_refs.h"
#include "npu/internal.h"
#include "npu/core.h"
#include "npu/i8/i8.h"
#include "spine_kernels.h"

struct ork_regovr orki_i8_fovr[16]; int orki_i8_fovr_n=0;

void ork_i8_fuzz_clear(void){ orki_i8_fovr_n=0; }

void ork_i8_fuzz_add(uint32_t blk,uint32_t reg,uint32_t val){ if(orki_i8_fovr_n<16){ orki_i8_fovr[orki_i8_fovr_n].blk=blk; orki_i8_fovr[orki_i8_fovr_n].reg=reg; orki_i8_fovr[orki_i8_fovr_n].val=val; orki_i8_fovr_n++; } }

int ork_npu_probe_mtile_i8(ork_npu *c,int M,int K,int N,int mode,
                           const int8_t *A,const int8_t *B,int32_t *C,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    int NN=N/32,KT=K/32; int8_t*bb=W.cpu;     /* int8 tile layout [Ntile][Ktile][32][32], full K */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)];
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)M*N*4,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}
    int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I8_N];
    orki_synth_i8(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    if(mode==1){   /* override with the rkllm-captured M-tile program */
        orki_setrn(rc,REGCMD_I8_N,RK_CNA_CONV_CON2,0x20);
        orki_setrn(rc,REGCMD_I8_N,RK_CNA_CBUF_CON1,(K/64)*M);
        orki_setrn(rc,REGCMD_I8_N,RK_CNA_DMA_CON1,4*M);
        int mg=(M+7)/8; int v=0xb1-0x0f*(mg-1); if(v<0x1b)v=0x1b;
        orki_setrn(rc,REGCMD_I8_N,RK_CNA_CBUF_CON0,v);
        orki_setrn(rc,REGCMD_I8_N,RK_CNA_DATA_SIZE0,(M<<16)|1);
        orki_setrn(rc,REGCMD_I8_N,RK_CNA_DATA_SIZE0_MIR,(M<<16)|1);
    }
    struct buf extra[2] = {W, O};
    if (orki_validate_regcmd("probe_mtile_i8", c, rc, REGCMD_I8_N, NULL, extra, 2)) { orki_bdestroy(fd,&W); orki_bdestroy(fd,&O); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0;
    for(int rep=0;rep<3;rep++){ sub.timeout=orki_mm_timeout_ms();   /* rep0/1 warmup, rep2 timed */
        double t0=ork_now_us();
        if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; break; }
        orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    if(ok==0){ memcpy(C,O.cpu,(size_t)M*N*4); if(us)*us=t1; }
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);
    return ok;
}

int ork_npu_probe_single_i8(ork_npu *c,int K,int N,const int8_t *A,const int8_t *B,int32_t *C){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    int NN=N/32,KT=K/32; int8_t*bb=W.cpu;     /* int8 tile layout [Ntile][Ktile][32][32], full K */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)];
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)N*4,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}
    int8_t*ad=c->Af.cpu; for(int j=0;j<K;j++)ad[j]=A[j]; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);                 /* prime for int8 / clear any prior wedge */
    uint32_t rc[REGCMD_I8_N];
    orki_synth_i8(rc,1,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_CBUF_CON0,0xb1);
    struct buf extra[2] = {W, O};
    if (orki_validate_regcmd("probe_single_i8", c, rc, REGCMD_I8_N, NULL, extra, 2)) { orki_bdestroy(fd,&W); orki_bdestroy(fd,&O); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1;
    for(int rep=0;rep<2;rep++){ sub.timeout=orki_mm_timeout_ms();   /* rep0 warmup (cold buffer stale), rep1 real */
        if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; continue; }
        orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(C,O.cpu,(size_t)N*4); ok=0; }
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);
    return ok;
}

int ork_npu_probe_i8_out8(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,
                          int mult,int shift,int8_t *C,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    int NN=N/32,KT=K/32; int8_t*bb=W.cpu;     /* int8 tile layout [Ntile][Ktile][32][32], full K */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)];
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)M*N,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}  /* int8 output: M*N bytes */
    int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I8_N];
    orki_synth_i8(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    orki_set_i8_out8(rc,N,0,mult,shift);           /* rewrite output stage: int32 -> int8 requantize */
    struct buf extra[2] = {W, O};
    if (orki_validate_regcmd("probe_i8_out8", c, rc, REGCMD_I8_N, NULL, extra, 2)) { orki_bdestroy(fd,&W); orki_bdestroy(fd,&O); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0;
    for(int rep=0;rep<3;rep++){ sub.timeout=orki_mm_timeout_ms();   /* rep0/1 warmup, rep2 timed */
        double t0=ork_now_us();
        if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; break; }
        orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    if(ok==0){ memcpy(C,O.cpu,(size_t)M*N); if(us)*us=t1; }
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);
    return ok;
}

int ork_npu_probe_i8_mm(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,int32_t *raw){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    int NN=N/32,KT=K/32; int8_t*bb=W.cpu;     /* int8 weight tile [Ntile][Ktile][32][32], full K */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)];
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)2*M*N*4,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}  /* 2x, int32 */
    int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I8_N];
    int i8sched=1; { const char*e=getenv("ORK_I8_PROBE_SCHED"); if(e) i8sched=atoi(e); }  /* 0 = no 0x1040 streaming schedule (batch test) */
    orki_synth_i8(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,i8sched,CBUF,0);   /* ork_i8_fuzz overrides apply inside */
    struct buf extra[2] = {W, O};
    if (orki_validate_regcmd("probe_i8_mm", c, rc, REGCMD_I8_N, NULL, extra, 2)) { orki_bdestroy(fd,&W); orki_bdestroy(fd,&O); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t to_ms=60000; { const char*e=getenv("ORK_I4_PROBE_TO_MS"); if(e){ unsigned v=(unsigned)strtoul(e,0,0); if(v)to_ms=v; } }
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1;
    for(int rep=0;rep<2;rep++){ sub.timeout=to_ms; if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; continue; }
        orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; }
    if(ok==0) memcpy(raw,O.cpu,(size_t)2*M*N*4);
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);
    return ok;
}

int ork_npu_probe_i8_ewmul(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,const int8_t *G,
                           int mult,int shift,int8_t *C,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(!ork_ppu_fuse_enabled(c)) return -3;   /* PPU EW-mul RE'd against the rk3588 PPU layout only */
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    int NN=N/32,KT=K/32; int8_t*bb=W.cpu;     /* int8 tile layout [Ntile][Ktile][32][32], full K */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)];
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)M*N,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}         /* int8 output */
    /* over-allocate the 2nd-input buffer: the captured 0x5020/0x5038 partner offsets (+0x4080/+0x400 from
     * base) must land IN-BOUNDS or the IOMMU faults (submit timeout). 64 KiB covers them for these shapes. */
    size_t gsz=(size_t)M*N; if(gsz<0x10000)gsz=0x10000;
    struct buf Gb=orki_bcreate(fd,gsz,0x403,-1); if(!Gb.cpu){orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);return -2;} /* 2nd input */
    memset(Gb.cpu,0,gsz); memcpy(Gb.cpu,G,(size_t)M*N); orki_bsync(fd,&Gb,RKNPU_MEM_SYNC_TO_DEVICE);
    int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t base[REGCMD_I8_N], rc[REGCMD_I8_EW_N];
    orki_synth_i8(base,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    orki_splice_ew_lane(rc,base);                                    /* insert the 0x50xx second-input lane */
    orki_set_i8_ewmul(rc,M,N,0,mult,shift,(uint32_t)Gb.dma);           /* int8 requant + EW-mul + 2nd-input addr */
    if(getenv("ORK_EW_DUMP")){                                  /* inspect the assembled regcmd, no submit */
        printf("# assembled EW-mul regcmd (%d entries) aG=0x%x aC=0x%x\n",REGCMD_I8_EW_N/2,(uint32_t)Gb.dma,(uint32_t)O.dma);
        for(int k=0;k+1<REGCMD_I8_EW_N;k+=2){uint32_t w0=rc[k],w1=rc[k+1];
            printf("  [%3d] reg=%04x lane=%04x val=%08x\n",k/2,w0&0xffff,w1>>16,((w0>>16)&0xffff)|((w1&0xffff)<<16));}
        orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Gb); return 0;
    }
    struct buf extra[3] = {W, O, Gb};
    if (orki_validate_regcmd("probe_i8_ewmul", c, rc, REGCMD_I8_EW_N, NULL, extra, 3)) { orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Gb); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    /* bump the task's register-config count 108 -> 126 AND enable_mask 0xd -> 0x1d (the 0x10 bit enables the
     * PPU / second DPU lane; the captured EW-mul op runs at enable=0x1d, same as the SiLU compute op) for
     * this submit, then restore (c->task is shared). */
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu;
    uint32_t saved=tk->regcfg_amount, saved_en=tk->enable_mask;
    tk->regcfg_amount=REGCMD_I8_EW_N/2; tk->enable_mask=0x1d; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0;
    for(int rep=0;rep<3;rep++){ sub.timeout=orki_ew_timeout_ms();
        double t0=ork_now_us();
        if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; break; }
        orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=saved; tk->enable_mask=saved_en; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);  /* restore shared task */
    if(ok==0){ memcpy(C,O.cpu,(size_t)M*N); if(us)*us=t1; }
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Gb);
    return ok;
}

int ork_npu_probe_i8_ewmul_tmpl(ork_npu *c,const void*in,int Isz,const void*wt,int Wsz,
                                const void*gl,int Gsz,void*out,int Osz,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;   /* rk3588 PPU only */
    struct buf I=orki_bcreate(fd,4096,0x403,-1); if(!I.cpu) return -2;
    struct buf Wt=orki_bcreate(fd,32768,0x403,-1); if(!Wt.cpu){orki_bdestroy(fd,&I);return -2;}
    struct buf Gl=orki_bcreate(fd,8192,0x403,-1); if(!Gl.cpu){orki_bdestroy(fd,&I);orki_bdestroy(fd,&Wt);return -2;}
    struct buf O=orki_bcreate(fd,4096,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&I);orki_bdestroy(fd,&Wt);orki_bdestroy(fd,&Gl);return -2;}
    memset(I.cpu,0,4096); memset(Wt.cpu,0,32768); memset(Gl.cpu,0,8192); memset(O.cpu,0,4096);
    if(in) memcpy(I.cpu, in, Isz<4096?Isz:4096);
    if(wt) memcpy((char*)Wt.cpu+0x2300, wt, Wsz<(32768-0x2300)?Wsz:(32768-0x2300)); /* weights at captured +0x2300 */
    if(gl) memcpy((char*)Gl.cpu+0x400, gl, Gsz<(8192-0x400)?Gsz:(8192-0x400));      /* silu(gate) at captured +0x400 */
    orki_bsync(fd,&I,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&Wt,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&Gl,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_EWMUL_N]; memcpy(rc,REGCMD_EWMUL,sizeof rc);
    /* ORK_EW_MULT/ORK_EW_SHIFT: override the captured requant (0x4084/0x4088, captured >>29 kills small acc)
     * so the output is interpretable when driving the verbatim op with our own uniform data. */
    { const char*em=getenv("ORK_EW_MULT"),*es=getenv("ORK_EW_SHIFT");
      if(em) orki_setrn(rc,REGCMD_EWMUL_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)strtoul(em,0,0));
      if(es) orki_setrn(rc,REGCMD_EWMUL_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)strtoul(es,0,0)); }
    orki_setrn(rc,REGCMD_EWMUL_N,RK_CNA_FEATURE_DATA_ADDR,(uint32_t)I.dma);          /* input x */
    orki_setrn(rc,REGCMD_EWMUL_N,RK_CNA_WEIGHT_DATA_ADDR,(uint32_t)Wt.dma+0x2300);  /* weights */
    orki_setrn(rc,REGCMD_EWMUL_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);          /* output */
    orki_setrn(rc,REGCMD_EWMUL_N,RK_SDP_5018,(uint32_t)Gl.dma+0x400);   /* 2nd-input silu(gate) */
    orki_setrn(rc,REGCMD_EWMUL_N,RK_SDP_5038,(uint32_t)Gl.dma+0x800);   /* 2nd-input partner */
    orki_setrn(rc,REGCMD_EWMUL_N,RK_SDP_5020,(uint32_t)Wt.dma+0x2480);  /* 2nd-input param (in weight buf) */
    if(getenv("ORK_EW_DUMP")){ printf("# verbatim EW-mul regcmd, in=0x%x wt=0x%x gl=0x%x out=0x%x\n",
        (uint32_t)I.dma,(uint32_t)Wt.dma,(uint32_t)Gl.dma,(uint32_t)O.dma);
        for(int k=0;k+1<REGCMD_EWMUL_N;k+=2) printf("  [%3d] reg=%04x lane=%04x val=%08x\n",k/2,rc[k]&0xffff,rc[k+1]>>16,((rc[k]>>16)&0xffff)|((rc[k+1]&0xffff)<<16));
        orki_bdestroy(fd,&I);orki_bdestroy(fd,&Wt);orki_bdestroy(fd,&Gl);orki_bdestroy(fd,&O); return 0; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t sa=tk->regcfg_amount,se=tk->enable_mask;
    tk->regcfg_amount=126; tk->enable_mask=0x1d; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms();
    double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=sa; tk->enable_mask=se; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ if(out)memcpy(out,O.cpu,Osz<4096?Osz:4096); if(us)*us=t1; }
    orki_bdestroy(fd,&I);orki_bdestroy(fd,&Wt);orki_bdestroy(fd,&Gl);orki_bdestroy(fd,&O);
    return ok;
}

int ork_npu_probe_i8_ewmul_lin(ork_npu *c,const int8_t *A,const int8_t *B,const int8_t *G,int8_t *C,double *us){
    const int M=8,K=512,N=64; int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;   /* rk3588 PPU only */
    struct buf Wt=orki_bcreate(fd,0x18000,0x403,-1); if(!Wt.cpu) return -2;   /* weights + 0x5020 param region */
    int NN=N/32,KT=K/32; int8_t*bb=Wt.cpu;
    /* LAYOUT-INVARIANT probe trick: fill the WHOLE weight buffer with B[0] so the matmul reads the same
     * value no matter how RKNN's op tiles it (acc = K*A*B[0] for every output, independent of layout).
     * Then overlay ork's tile packing (a no-op when B is uniform). Lets P1 run before RKNN's A/B layout
     * is reversed. For non-uniform B this region still holds B[0] outside the tile — only valid for the
     * uniform-input probes. */
    memset(bb,(unsigned char)B[0],0x18000);
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)];
    { const char*e=getenv("ORK_EW_S20"); if(e){ uint32_t v=(uint32_t)strtoul(e,0,0); for(int i=0;i<N;i++)((uint32_t*)((char*)Wt.cpu+0x8080))[i]=v; } }
    orki_bsync(fd,&Wt,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&Wt,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,4096,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&Wt);return -2;}
    struct buf Gb=orki_bcreate(fd,4096,0x403,-1); if(!Gb.cpu){orki_bdestroy(fd,&Wt);orki_bdestroy(fd,&O);return -2;}
    memset(O.cpu,0,4096); memset(Gb.cpu,0,4096); memcpy(Gb.cpu,G,(size_t)M*N); orki_bsync(fd,&Gb,RKNPU_MEM_SYNC_TO_DEVICE);
    int8_t*ad=c->Af.cpu; memset(ad,(unsigned char)A[0],0x8000); for(int j=0;j<M*K;j++)ad[j]=A[j];  /* fill for layout-invariance */
    orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_EWMUL_LIN_N]; memcpy(rc,REGCMD_EWMUL_LIN,sizeof rc);
    /* EMITTER: inject ork's matmul geometry into RKNN's EW template (unless ORK_EW_VERBATIM) so the conv
     * engine reads ork's [Nt][Kt][32][32] A/B layout while keeping RKNN's register order + EW output stage. */
    if(!getenv("ORK_EW_VERBATIM")){
        orki_apply_ork_geom(rc,REGCMD_EWMUL_LIN_N,M,K,N,c->soc->cbuf_elems);
        /* Also inject ork's DENSE int8-out output-stage byte config (set_i8_out8) so the output writes dense
         * [M][N] where ork reads it — the template's RKNN output-byte config produced bias-only garbage.
         * Keep the SDP element-wise MULTIPLY enable (0x4070=0x904002c4) + 0x4050 bit0 (2nd-input enable). */
        orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_DPU_OUT_PRECISION,0x00000000);    /* int8-out (clear int32 bit) */
        orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_DPU_DATA_CUBE_NOTCH,(((N/16)-1)<<16)|((N/16)-1)); /* dense output group stride */
        orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_DPU_BS_OW_CFG,0x00000125);    /* int8 row config + 2nd-input enable (bit0) */
        orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_DPU_SURFACE_ADD,0x00000020);    /* output element size = 1 byte (dense) */
        orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_DPU_OUT_CVT_OFFSET,0x00000000);    /* zero bias for the probe */
        orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_SDP_500C,M-1);           /* 2nd-lane geometry -> ork's (M,N) */
        orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_SDP_5010,M-1);
        orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_SDP_5014,N-1);
        orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_SDP_5040,N);
        orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_SDP_504C,N);
        orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_SDP_506C,N);
    }
    /* ORK_EW_MULT/SHIFT override the captured requant so small test accumulators requant to a readable value. */
    { const char*em=getenv("ORK_EW_MULT"),*es=getenv("ORK_EW_SHIFT");
      if(em) orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)strtoul(em,0,0));
      if(es) orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)strtoul(es,0,0)); }
    uint32_t gstride = getenv("ORK_EW_VERBATIM") ? 0x80 : (uint32_t)N;  /* 2nd-input row stride */
    orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_CNA_FEATURE_DATA_ADDR,(uint32_t)c->Af.dma);        /* input A */
    orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_CNA_WEIGHT_DATA_ADDR,(uint32_t)Wt.dma);           /* up-weights */
    orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);            /* output */
    orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_SDP_5018,(uint32_t)Gb.dma);           /* silu(gate) 2nd input */
    orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_SDP_5038,(uint32_t)Gb.dma+gstride);   /* 2nd-input partner = base+stride */
    orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_SDP_5020,(uint32_t)Wt.dma+0x8080);    /* 2nd-input param (in weight buf) */
    /* ORK_EW_NOMUL: turn OFF the EW multiply (0x4070 -> plain 0x0302) to read R=requant(acc) alone —
     * isolates "does the matmul+requant work" from "does the multiply work". */
    if(getenv("ORK_EW_NOMUL")) orki_setrn(rc,REGCMD_EWMUL_LIN_N,RK_DPU_EW_CFG,0x00000302);
    if(getenv("ORK_EW_DUMP")){ for(int k=0;k+1<REGCMD_EWMUL_LIN_N;k+=2) printf("  [%3d] reg=%04x lane=%04x val=%08x\n",k/2,rc[k]&0xffff,rc[k+1]>>16,((rc[k]>>16)&0xffff)|((rc[k+1]&0xffff)<<16));
        orki_bdestroy(fd,&Wt);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Gb); return 0; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t sa=tk->regcfg_amount,se=tk->enable_mask;
    tk->regcfg_amount=126; tk->enable_mask=0x1d; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms(); double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=sa; tk->enable_mask=se; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ memcpy(C,O.cpu,(size_t)M*N); if(us)*us=t1; }
    if(ok==0 && getenv("ORK_EW_SCAN")){ int nz=0,first=-1; signed char*o=O.cpu;
        for(int i=0;i<4096;i++){ if(o[i]){ nz++; if(first<0)first=i; } }
        printf("  [scan] output-buffer nonzero bytes=%d first@0x%x  bytes@first: ",nz,first);
        for(int i=(first<0?0:first);i<(first<0?16:first+16);i++)printf("%d ",o[i]); printf("\n"); }
    orki_bdestroy(fd,&Wt);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Gb);
    return ok;
}

int ork_npu_probe_i8_mul(ork_npu *c,const int8_t *a,const int8_t *b,int n,int8_t *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(n<1||n>4096) return -2;
    struct buf A=orki_bcreate(fd,4096,0x403,-1); if(!A.cpu)return -2;
    struct buf B=orki_bcreate(fd,4096,0x403,-1); if(!B.cpu){orki_bdestroy(fd,&A);return -2;}
    struct buf O=orki_bcreate(fd,4096,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);return -2;}
    memset(A.cpu,0,4096);memset(B.cpu,0,4096);memset(O.cpu,0,4096);
    memcpy(A.cpu,a,n);memcpy(B.cpu,b,n);
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);   /* clear device-side output (op writes only part -> avoid stale) */
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_MUL_N]; memcpy(rc,REGCMD_MUL,sizeof rc);
    orki_setrn(rc,REGCMD_MUL_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);        /* output */
    orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5018,(uint32_t)A.dma);        /* operand a (SRDMA) */
    orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5038,(uint32_t)B.dma);        /* operand b (ERDMA element-wise) */
    { const char*em=getenv("ORK_EW_MULT"),*es=getenv("ORK_EW_SHIFT"),*eb=getenv("ORK_EW_BIAS");
      const char*co=getenv("ORK_EW_COFF"),*cs=getenv("ORK_EW_CSCL");
      if(em) orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)strtoul(em,0,0));
      if(es) orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)strtoul(es,0,0));
      if(eb) orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_OFFSET,(uint32_t)strtoul(eb,0,0));
      if(co) orki_setrn(rc,REGCMD_MUL_N,RK_DPU_EW_CVT_OFFSET,(uint32_t)strtoul(co,0,0));   /* EW_CVT_OFFSET = zb (operand b zero-pt) */
      if(cs) orki_setrn(rc,REGCMD_MUL_N,RK_DPU_EW_CVT_SCALE,(uint32_t)strtoul(cs,0,0));   /* EW_CVT_SCALE (operand b) */
      const char*ao=getenv("ORK_EW_AOFF");
      if(ao) orki_setrn(rc,REGCMD_MUL_N,RK_DPU_BS_ALU_CFG,(uint32_t)strtoul(ao,0,0));   /* BS_ALU_OPERAND = za (operand a zero-pt) */
      /* SDP DPU ALU-mode registers (rocket rocket_registers.h: 0x4040=BS_CFG, 0x4048=BS_MUL_CFG,
       * 0x4070=EW_CFG w/ EW_ALU_ALGO bits[19:16]: MAX=0,MIN=1,SUM=2,EQL=3). Override to retarget the
       * ALU function (e.g. ADD-routing + algo=0 => on-NPU max(a,b)). RE hooks for building max-reduce. */
      const char*r40=getenv("ORK_EW_R40"),*r48=getenv("ORK_EW_R48"),*r70=getenv("ORK_EW_R70");
      if(r40) orki_setrn(rc,REGCMD_MUL_N,RK_DPU_BS_CFG,(uint32_t)strtoul(r40,0,0));
      if(r48) orki_setrn(rc,REGCMD_MUL_N,RK_DPU_BS_MUL_CFG,(uint32_t)strtoul(r48,0,0));
      if(r70) orki_setrn(rc,REGCMD_MUL_N,RK_DPU_EW_CFG,(uint32_t)strtoul(r70,0,0));
      const char*r34=getenv("ORK_EW_R34");   /* ERDMA_CFG 0x5034: ERDMA_DATA_MODE bits[31:30] — per-channel operand-b broadcast RE */
      if(r34) orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5034,(uint32_t)strtoul(r34,0,0)); }
    if(getenv("ORK_EW_DUMP")){ for(int k=0;k+1<REGCMD_MUL_N;k+=2) printf("  [%3d] reg=%04x lane=%04x val=%08x\n",k/2,rc[k]&0xffff,rc[k+1]>>16,((rc[k]>>16)&0xffff)|((rc[k+1]&0xffff)<<16));
        orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&O); return 0; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t sa=tk->regcfg_amount,se=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms(); double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=sa; tk->enable_mask=se; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ memcpy(out,O.cpu,n); if(us)*us=t1; }
    if(ok==0 && getenv("ORK_EW_SCAN")){ signed char*o=O.cpu; int nz=0,first=-1,last=-1;
        for(int i=0;i<4096;i++){ if(o[i]){ nz++; if(first<0)first=i; last=i; } }
        printf("  [scan4k] nonzero=%d span[0x%x..0x%x]  vals@first: ",nz,first,last);
        for(int i=(first<0?0:first);i<(first<0?16:first+24)&&i<4096;i++)printf("%d ",o[i]); printf("\n"); }
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&O);
    return ok;
}

int ork_npu_probe_add_i8(ork_npu *c,const int8_t *a,const int8_t *b,int M,int N,
                         int mult,int shift,uint32_t bscale,int za,int zb,int zo,int8_t *out,double *us){
    int fd=c->fd, dom=c->dom_active;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<16||N>8192||(N&15)) return -2;
    #define EWCUBE(m,n) (((n)/16)*(M*16) + (m)*16 + ((n)%16))
    size_t sz=(size_t)M*N; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,dom); if(!A.cpu)return -2;
    struct buf B=orki_bcreate(fd,sz,0x403,dom); if(!B.cpu){orki_bdestroy(fd,&A);return -2;}
    struct buf O=orki_bcreate(fd,sz,0x403,dom); if(!O.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);return -2;}
    memset(A.cpu,0,sz);memset(B.cpu,0,sz);memset(O.cpu,0,sz);
    int8_t*ac=A.cpu,*bc=B.cpu;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int p=EWCUBE(m,n); ac[p]=a[m*N+n]; bc[p]=b[m*N+n]; }
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (XP_SDP=RC_SDPKW, keep-warm-aware; == the old inline reset) */
    uint32_t rc[REGCMD_ADD_N]; memcpy(rc,REGCMD_ADD,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_ADD_N,M,N);
    orki_setrn(rc,REGCMD_ADD_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,REGCMD_ADD_N,RK_SDP_5018,(uint32_t)A.dma);
    orki_setrn(rc,REGCMD_ADD_N,RK_SDP_5038,(uint32_t)B.dma);
    orki_setrn(rc,REGCMD_ADD_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)mult);
    orki_setrn(rc,REGCMD_ADD_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)shift);
    orki_setrn(rc,REGCMD_ADD_N,RK_DPU_EW_CVT_SCALE,bscale);
    orki_setrn(rc,REGCMD_ADD_N,RK_DPU_BS_ALU_CFG,(uint32_t)za);
    orki_setrn(rc,REGCMD_ADD_N,RK_DPU_EW_CVT_OFFSET,(uint32_t)zb);
    orki_setrn(rc,REGCMD_ADD_N,RK_DPU_OUT_CVT_OFFSET,(uint32_t)zo);
    { const char*r70=getenv("ORK_EW_R70"); if(r70) orki_setrn(rc,REGCMD_ADD_N,RK_DPU_EW_CFG,(uint32_t)strtoul(r70,0,0)); }  /* EW_ALU_ALGO override: SUM(2)->MAX(0)/MIN(1) */
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms(); double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,dom)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[m*N+n]=*(int8_t*)((char*)O.cpu+EWCUBE(m,n)); if(us)*us=t1; }
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&O);
    #undef EWCUBE
    return ok;
}

int ork_npu_probe_silu_std(ork_npu *c,const int8_t *in,int M,int N,
                           int r_mult,int r_shift,uint32_t out_bias,uint32_t idx_off,
                           uint32_t cfg4064,uint32_t cfg4068,const int16_t *lut,int nlut,
                           int8_t *out,double *us){
    int fd=c->fd, dom=c->dom_active;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<16||N>8192||(N&15)) return -2;
    if(r_mult<0||r_mult>0x7fff||r_shift<0||r_shift>31) return -2;
    #define EWCUBE(m,n) (((n)/16)*(M*16) + (m)*16 + ((n)%16))    /* int8 atom-16 cube, surf_stride=M*16 */
    size_t sz=(size_t)M*N; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,dom); if(!A.cpu)return -2;
    struct buf O=orki_bcreate(fd,sz,0x403,dom); if(!O.cpu){orki_bdestroy(fd,&A);return -2;}
    struct buf Lrc=orki_bcreate(fd,(size_t)REGCMD_SILU_LUT_N*4,0x403,dom); if(!Lrc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);return -2;}
    struct buf Lsc=orki_bcreate(fd,4096,0x403,dom); if(!Lsc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);return -2;}
    memset(A.cpu,0,sz);memset(O.cpu,0,sz);
    int8_t*ac=A.cpu; for(int m=0;m<M;m++)for(int n=0;n<N;n++) ac[EWCUBE(m,n)]=in[m*N+n];
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);

    /* ---- submit 1: LUT-load (enable=0x18, regcfg=1097) ---- */
    memcpy(Lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
    orki_setrn((uint32_t*)Lrc.cpu,REGCMD_SILU_LUT_N,RK_DPU_DST_BASE_ADDR,(uint32_t)Lsc.dma);
    if(lut){ uint32_t*lr=(uint32_t*)Lrc.cpu; int j=0;
        for(int k=0;k+1<REGCMD_SILU_LUT_N;k+=2){ if((lr[k]&0xffff)==0x4104){ int32_t v=(j<nlut)?(int32_t)lut[j]:0; j++;
            lr[k]=0x4104|((uint32_t)(v&0xffff)<<16); lr[k+1]=(0x1001u<<16)|(((uint32_t)v>>16)&0xffff); } } }
    orki_bsync(fd,&Lrc,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
      t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=1097; t->regcmd_addr=Lrc.dma;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.timeout=orki_ew_timeout_ms();sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ if(getenv("ORK_SILU_DBG"))fprintf(stderr,"[silu_std] orki_submit1 (LUT-load) WEDGED\n"); orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc); return -1; }
      if(getenv("ORK_SILU_DBG"))fprintf(stderr,"[silu_std] orki_submit1 (LUT-load) ok\n");
    }

    /* ---- submit 2: standalone SiLU op (enable=0x18, regcfg=69) reading the resident LUT ---- */
    uint32_t rc[REGCMD_SILU_STD_N]; memcpy(rc,REGCMD_SILU_STD,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_SILU_STD_N,M,N);
    orki_setrn(rc,REGCMD_SILU_STD_N,RK_SDP_5040,0);                 /* single-input: no ERDMA 2nd operand */
    orki_setrn(rc,REGCMD_SILU_STD_N,RK_SDP_5038,0);                 /* (orki_set_mul_geom is for the 2-input EW-mul) */
    orki_setrn(rc,REGCMD_SILU_STD_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);   /* output */
    orki_setrn(rc,REGCMD_SILU_STD_N,RK_SDP_5018,(uint32_t)A.dma);   /* input (SRDMA) */
    orki_setrn(rc,REGCMD_SILU_STD_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)r_mult);  /* R mantissa */
    orki_setrn(rc,REGCMD_SILU_STD_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)r_shift); /* R shift */
    orki_setrn(rc,REGCMD_SILU_STD_N,RK_DPU_OUT_CVT_OFFSET,out_bias);          /* out_bias */
    orki_setrn(rc,REGCMD_SILU_STD_N,RK_DPU_R4110,idx_off);           /* C0 index offset */
    orki_setrn(rc,REGCMD_SILU_STD_N,RK_DPU_BN_ALU_CFG,cfg4064);           /* index param */
    orki_setrn(rc,REGCMD_SILU_STD_N,RK_DPU_BN_MUL_CFG,cfg4068);           /* index param */
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    /* full task setup — submit-1 repointed regcmd_addr at the LUT-load buffer, so re-point it here */
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk);
    tk->enable_mask=0x18; tk->int_mask=0x300; tk->int_clear=0x1ffff; tk->regcfg_amount=69; tk->regcmd_addr=c->regcmd.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms(); double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,dom)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[m*N+n]=*(int8_t*)((char*)O.cpu+EWCUBE(m,n)); if(us)*us=t1; }
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);
    #undef EWCUBE
    return ok;
}


int ork_npu_replay_i8(ork_npu *c, const uint32_t *regcmd, int rn, int M, int K, int N,
                      const int8_t *Adata, int Abytes, const int8_t *Bdata, int Bbytes, int32_t *Cout, int iters, double *us){
    int fd=c->fd; if(fd<0) return -3; if(rn<8 || rn>2048 || M<1 || (K%32) || (N%16)) return -2;
    int dom=c->dom_active;
    size_t bsz=(Bbytes>(int)((size_t)K*N))?(size_t)Bbytes:(size_t)K*N;   /* B may be a STRIDED span > K*N (rkllm column-tile) */
    /* #39 SAFETY RIG: oversize A/B (like C) with a large guard margin so a malformed mfold DMA descriptor that
     * reads out-of-bounds lands in MAPPED memory (garbage -> diagnosable) instead of hanging the AXI/IOMMU bus
     * (the hard-wedge failure mode). The extra pages are mapped in the same IOVA domain and never freed early. */
    size_t aszg=(size_t)M*K*8+(1u<<20), bszg=bsz*8+(1u<<20);
    struct buf A =orki_bcreate(fd,aszg,0x403,dom);            if(!A.cpu)  return -2;
    struct buf B =orki_bcreate(fd,bszg,0x403,dom);            if(!B.cpu) {orki_bdestroy(fd,&A);return -2;}
    memset(A.cpu,0,aszg); memset(B.cpu,0,bszg);          /* zero the guard region so OOB reads see zeros, not stale */
    size_t cszg=(size_t)M*N*4*8+65536;                   /* #39: oversize C 8x as an OOB guard while RE'ing the mfold output stride (a wrong stride writes in-bounds -> diagnosable, not a wedge) */
    struct buf Cc=orki_bcreate(fd,cszg,0x403,dom);            if(!Cc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);return -2;}
    struct buf RCb=orki_bcreate(fd,(size_t)rn*4,0x403,dom);   if(!RCb.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&Cc);return -2;}
    if(Adata){ size_t acp=(Abytes>0)?(size_t)Abytes:(size_t)M*K; if(acp>aszg)acp=aszg; memcpy(A.cpu,Adata,acp); }   /* real captured A (full span, incl. any fold padding), clamped to the buffer */
    else memset(A.cpu,1,(size_t)M*K);   /* garbage (timing only) */
    if(Bdata) memcpy(B.cpu,Bdata,(Bbytes>0)?(size_t)Bbytes:(size_t)K*N); else memset(B.cpu,1,bsz);   /* real captured strided weight, or garbage */
    memset(Cc.cpu,0,cszg);
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t *rc=(uint32_t*)RCb.cpu; memcpy(rc,regcmd,(size_t)rn*4);
    /* #39 SAFETY: neutralize any PC chain descriptor (0x0101:0x0010 next-regcmd-addr / 0x0014 next-amount)
     * baked into a CAPTURED regcmd. We submit task_number=1 (single task), so the chain link is meaningless
     * here — but if left intact it points at the CAPTURING process's next-task IOVA, which is UNMAPPED in
     * ork's address space; the NPU walks the chain to it and IOMMU HARD-WEDGES the board. Zero the value
     * halves (keep offset/block) so next-addr=0 (end of chain) and next-amount=0. (validate_mfold's rfile
     * path already did this; doing it here makes every ork_npu_replay_i8 caller — incl. replay_mm_i8 — safe.) */
    for(int k=0;k+1<rn;k+=2){ unsigned o=rc[k]&0xffff, b=(rc[k+1]>>16)&0xffff;
        if(b==0x101 && (o==0x0010||o==0x0014)){ rc[k]&=0xffff; rc[k+1]&=0xffff0000u; } }
    orki_setrn(rc,rn,RK_CNA_FEATURE_DATA_ADDR,(uint32_t)A.dma);            /* A (activations) */
    orki_setrn(rc,rn,RK_CNA_WEIGHT_DATA_ADDR,(uint32_t)B.dma);            /* B (weights) */
    orki_setrn(rc,rn,RK_DPU_DST_BASE_ADDR,(uint32_t)Cc.dma);          /* C output IOVA (the ONLY output address; 0x40c0/SURFACE_ADD is a config value, NOT an address — do not patch it) */
    if(getenv("ORK_REPLAY_IOVA")) fprintf(stderr,"[replay-iova] A=%#x B=%#x C=%#x RC=%#x task=%#x\n",(unsigned)A.dma,(unsigned)B.dma,(unsigned)Cc.dma,(unsigned)RCb.dma,(unsigned)c->task.dma);  /* #39: observe IOVA stability across runs (intermittency diag) */
    orki_bsync(fd,&RCb,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk);
    tk->enable_mask=0xd; tk->int_mask=0x300; tk->int_clear=0x1ffff; tk->regcfg_amount=108; tk->regcmd_addr=RCb.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    int ret=-1; struct rknpu_submit sub;
    #define _RSUB() do{ memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=1; sub.task_obj_addr=c->task.obj; \
        sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1; sub.timeout=orki_mm_timeout_ms(); sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; }while(0)
    /* #39 intermittency probe: the raw mfold replay bypasses ork_npu_enter, so it never does the
     * mode-ENTRY ACT_RESET that the normal path fires when the schedule/precision mode changes. The
     * mfold is an unusual-mode regcmd entered right after an int8-normal warmup — hypothesis: the
     * missing entry-reset leaves stale NPU state that intermittently stalls/wedges the fold. Gate a
     * reset here so we can A/B whether it makes the submit deterministic (one-time ~107ms, before the
     * timed loop, so it doesn't taint the per-submit timing). */
    if(getenv("ORK_REPLAY_RESET")){ struct rknpu_action a={.flags=RKNPU_ACT_RESET,.value=0}; ioctl(fd,DRM_IOCTL_RKNPU_ACTION,&a); }
    _RSUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ goto done; }          /* warm */
    orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_FROM_DEVICE);
    if(Cout) memcpy(Cout,Cc.cpu,(size_t)M*N*4);           /* computed output for correctness check vs captured C */
    { double t0=ork_now_us();
      for(int i=0;i<iters;i++){ _RSUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ goto done; } }
      if(us) *us=(ork_now_us()-t0)/(iters>0?iters:1); ret=0; }
    #undef _RSUB
done:
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&B); orki_bdestroy(fd,&Cc); orki_bdestroy(fd,&RCb); return ret;
}

int ork_npu_replay_i8_sweep(ork_npu *c, const uint32_t *regcmd, int rn, int M, int K, int N,
        const int8_t *Avar, int nvar, int astride, const int8_t *Bdata, int Bbytes, int32_t *Couts){
    int fd=c->fd; if(fd<0) return -3; if(rn<8||rn>2048||M<1||(K%32)||(N%16)||nvar<1) return -2;
    int dom=c->dom_active;
    size_t bsz=(Bbytes>(int)((size_t)K*N))?(size_t)Bbytes:(size_t)K*N;
    size_t aszg=(size_t)M*K*8+(1u<<20), bszg=bsz*8+(1u<<20), cszg=(size_t)M*N*4*8+65536;
    struct buf A =orki_bcreate(fd,aszg,0x403,dom);          if(!A.cpu)  return -2;
    struct buf B =orki_bcreate(fd,bszg,0x403,dom);          if(!B.cpu) {orki_bdestroy(fd,&A);return -2;}
    struct buf Cc=orki_bcreate(fd,cszg,0x403,dom);          if(!Cc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);return -2;}
    struct buf RCb=orki_bcreate(fd,(size_t)rn*4,0x403,dom); if(!RCb.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&Cc);return -2;}
    memset(B.cpu,0,bszg); if(Bdata)memcpy(B.cpu,Bdata,(Bbytes>0)?(size_t)Bbytes:(size_t)K*N); else memset(B.cpu,1,bsz);
    orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t *rc=(uint32_t*)RCb.cpu; memcpy(rc,regcmd,(size_t)rn*4);
    for(int k=0;k+1<rn;k+=2){ unsigned o=rc[k]&0xffff,b=(rc[k+1]>>16)&0xffff; if(b==0x101&&(o==0x0010||o==0x0014)){rc[k]&=0xffff;rc[k+1]&=0xffff0000u;} }
    orki_setrn(rc,rn,RK_CNA_FEATURE_DATA_ADDR,(uint32_t)A.dma);
    orki_setrn(rc,rn,RK_CNA_WEIGHT_DATA_ADDR,(uint32_t)B.dma);
    orki_setrn(rc,rn,RK_DPU_DST_BASE_ADDR,(uint32_t)Cc.dma);
    orki_bsync(fd,&RCb,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk);
    tk->enable_mask=0xd; tk->int_mask=0x300; tk->int_clear=0x1ffff; tk->regcfg_amount=108; tk->regcmd_addr=RCb.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    int ret=-1; struct rknpu_submit sub;
    #define _SSUB() do{ memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=1; sub.task_obj_addr=c->task.obj; \
        sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1; sub.timeout=orki_mm_timeout_ms(); sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; }while(0)
    size_t acp=(size_t)astride; if(acp>aszg)acp=aszg;
    memset(A.cpu,0,aszg); memcpy(A.cpu,Avar,acp); orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);   /* warm w/ variant 0 */
    _SSUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)) goto sdone;
    for(int v=0; v<nvar; v++){
        memset(A.cpu,0,aszg); memcpy(A.cpu, Avar+(size_t)v*astride, acp); orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);
        memset(Cc.cpu,0,(size_t)M*N*4); orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_TO_DEVICE);
        _SSUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)) goto sdone;
        orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_FROM_DEVICE);
        memcpy(Couts+(size_t)v*M*N, Cc.cpu, (size_t)M*N*4);
    }
    ret=0;
    #undef _SSUB
sdone:
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&B); orki_bdestroy(fd,&Cc); orki_bdestroy(fd,&RCb); return ret;
}

int ork_npu_replay_i8_amap(ork_npu *c, const uint32_t *regcmd, int rn, int M, int K, int N,
        const uint32_t *Bpos, int nk0, int n0, int32_t *rpos, int32_t *aoff, int32_t *cnt){
    int fd=c->fd; if(fd<0)return -3; if(rn<8||rn>2048||M<1||(K%32)||(N%16)||nk0<1)return -2;
    int dom=c->dom_active;
    size_t aszg=(size_t)M*K*8+(1u<<20), bszg=(size_t)K*N*8+(1u<<20), cszg=(size_t)M*N*4*8+65536;
    struct buf A =orki_bcreate(fd,aszg,0x403,dom);          if(!A.cpu)  return -2;
    struct buf B =orki_bcreate(fd,bszg,0x403,dom);          if(!B.cpu) {orki_bdestroy(fd,&A);return -2;}
    struct buf Cc=orki_bcreate(fd,cszg,0x403,dom);          if(!Cc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);return -2;}
    struct buf RCb=orki_bcreate(fd,(size_t)rn*4,0x403,dom); if(!RCb.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&Cc);return -2;}
    uint32_t *rc=(uint32_t*)RCb.cpu; memcpy(rc,regcmd,(size_t)rn*4);
    for(int k=0;k+1<rn;k+=2){ unsigned o=rc[k]&0xffff,b=(rc[k+1]>>16)&0xffff; if(b==0x101&&(o==0x0010||o==0x0014)){rc[k]&=0xffff;rc[k+1]&=0xffff0000u;} }
    orki_setrn(rc,rn,RK_CNA_FEATURE_DATA_ADDR,(uint32_t)A.dma);
    orki_setrn(rc,rn,RK_CNA_WEIGHT_DATA_ADDR,(uint32_t)B.dma);
    orki_setrn(rc,rn,RK_DPU_DST_BASE_ADDR,(uint32_t)Cc.dma);
    orki_bsync(fd,&RCb,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk);
    tk->enable_mask=0xd; tk->int_mask=0x300; tk->int_clear=0x1ffff; tk->regcfg_amount=108; tk->regcmd_addr=RCb.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    int ret=-1; struct rknpu_submit sub; int8_t *ab=(int8_t*)A.cpu; int32_t *cb=(int32_t*)Cc.cpu;
    #define _ASUB() do{ memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=1; sub.task_obj_addr=c->task.obj; \
        sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1; sub.timeout=orki_mm_timeout_ms(); sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; }while(0)
    #define _FILLA(pl) do{ for(size_t j=0;j<aszg;j++) ab[j]=(int8_t)((pl<0)?1:((j>>(7*(pl)))&0x7f)); orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); }while(0)
    /* warm */
    memset(B.cpu,0,bszg); ((int8_t*)B.cpu)[Bpos[0]]=1; orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);
    _FILLA(-1); _ASUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)) goto adone;
    for(int i=0;i<nk0;i++){
        memset(B.cpu,0,bszg); ((int8_t*)B.cpu)[Bpos[i]]=1; orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);
        /* presence: A=1 -> C==1 at this k0's output slots */
        _FILLA(-1); memset(cb,0,(size_t)M*N*4); orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_TO_DEVICE);
        _ASUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)) goto adone; orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_FROM_DEVICE);
        int t=0; for(int idx=0; idx<M*N && t<M; idx++) if(cb[idx]!=0){ rpos[(size_t)i*M+t]=idx; t++; }
        cnt[i]=t;
        for(int p=0;p<3;p++){
            _FILLA(p); memset(cb,0,(size_t)M*N*4); orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_TO_DEVICE);
            _ASUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)) goto adone; orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_FROM_DEVICE);
            for(int tt=0;tt<t;tt++){ int32_t d=cb[rpos[(size_t)i*M+tt]]&0x7f;
                if(p==0) aoff[(size_t)i*M+tt]=d; else aoff[(size_t)i*M+tt]|=(int32_t)d<<(7*p); }
        }
    }
    ret=0;
    #undef _ASUB
    #undef _FILLA
adone:
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&B); orki_bdestroy(fd,&Cc); orki_bdestroy(fd,&RCb); return ret;
}

int ork_npu_probe_i8_silu(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,int8_t *C,double *us){
    /* g2 captured register set (a known-good replay): R=0x51aa/2^0x14, bias=-97, idx_off, 0x4068 field. */
    return ork_npu_probe_i8_silu_cfg(c,M,K,N,A,B,0x51aa,0x14,0xffffff9fu,0xffffc000u,0x56391100u,NULL,0,C,us);
}

int ork_npu_probe_i8_silu_cfg(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,
                              int r_mult,int r_shift,uint32_t out_bias,uint32_t idx_off,uint32_t cfg4068,
                              const int16_t *lut,int nlut,int8_t *C,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64) return -2;
    /* MULTI-DOMAIN: the FFN chain calls this per-layer fused-SiLU LUT-calibration probe with c->dom_active set
     * to the layer's (non-0) domain; the probe's buffers + submits must MATCH it, else submitting
     * iommu_domain_id=0 while c->dom_active!=0 WEDGES (errno 110) — the same hazard already fixed for the i16
     * SiLU probes (ork_npu_probe_silu_std_i16). dom==0 for every single-domain caller => behavior-preserving.
     * ORK_SILU_DOM0=1 forces dom0 (A/B / disable the fix). */
    int dom = getenv("ORK_SILU_DOM0") ? 0 : c->dom_active;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,dom); if(!W.cpu) return -2;
    int NN=N/32,KT=K/32; int8_t*bb=W.cpu;
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)];
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)M*N,0x403,dom); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}
    struct buf Lrc=orki_bcreate(fd,(size_t)REGCMD_SILU_LUT_N*4,0x403,dom); if(!Lrc.cpu){orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);return -2;} /* LUT-load regcmd */
    struct buf Lsc=orki_bcreate(fd,4096,0x403,dom); if(!Lsc.cpu){orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);return -2;} /* LUT-load scratch (reg 0x4020) */
    int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    /* Prime against a fresh-buffer stale-read/wedge — but ONLY when the NPU isn't already int8-warm. The chain
     * builds the fused-SiLU LUT via this probe once per layer during prep; an unconditional ~107ms ACT_RESET
     * fired ~once/layer (28 cold resets on the first forward pass, hurting cold TTFT/pp128). When int8-live the
     * probe's own 3-rep warmup (below) flushes stale reads, so the reset is redundant. ORK_PROBE_RESET=1 forces
     * it (fallback). Standalone/cold callers still reset. LUT calibration is unaffected (validated bit-exact). */
    if(!ORK_I8_LIVE(c->last_dt) || getenv("ORK_PROBE_RESET")) orki_act(fd,RKNPU_ACT_RESET,0);

    /* ---- submit 1: LUT-load (enable=0x18, regcfg=1097) — streams the silu LUT into PPU SRAM ---- */
    memcpy(Lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
    orki_setrn((uint32_t*)Lrc.cpu,REGCMD_SILU_LUT_N,RK_DPU_DST_BASE_ADDR,(uint32_t)Lsc.dma); /* patch the one output addr */
    if(lut){ uint32_t*lr=(uint32_t*)Lrc.cpu; int j=0;   /* override LUT data: stream lut[] into the 0x4104 writes in order */
        for(int k=0;k+1<REGCMD_SILU_LUT_N;k+=2){ if((lr[k]&0xffff)==0x4104){ int32_t v=(j<nlut)?(int32_t)lut[j]:0; j++;
            lr[k]=0x4104|((uint32_t)(v&0xffff)<<16); lr[k+1]=(0x1001u<<16)|(((uint32_t)v>>16)&0xffff); } } }
    orki_bsync(fd,&Lrc,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
      t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=1097; t->regcmd_addr=Lrc.dma;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.timeout=orki_mm_timeout_ms();sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc); return -1; }
    }

    /* ---- submit 2: matmul compute (enable=0x1d, regcfg=108) reading the resident LUT ---- */
    uint32_t rc[REGCMD_I8_N];
    orki_synth_i8(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    orki_set_i8_silu(rc,N,0,r_mult,r_shift,out_bias,idx_off,cfg4068);
    if(getenv("ORK_FUSED_DUMP")){ for(int k=0;k+1<REGCMD_I8_N;k+=2) fprintf(stderr,"PROBE %04x=%08x l=%04x\n",rc[k]&0xffff,((rc[k]>>16)&0xffff)|((rc[k+1]&0xffff)<<16),rc[k+1]>>16); }
    struct buf extra[2]={W,O};
    if(orki_validate_regcmd("probe_i8_silu",c,rc,REGCMD_I8_N,NULL,extra,2)){ orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
      t->enable_mask=0x1d; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=108; t->regcmd_addr=c->regcmd.dma;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      int ok=-1; double t1=0;
      for(int rep=0;rep<3;rep++){ sub.timeout=orki_mm_timeout_ms(); double t0=ork_now_us();
          if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ ok=-1; break; }
          orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
      if(ok==0){ memcpy(C,O.cpu,(size_t)M*N); if(us)*us=t1; }
      orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);
      return ok;
    }
}
int ork_npu_doorbell_prof(ork_npu *c,int M,int K,int N,int iters,double *block_us,double *nb_us,int *ok_block,int *ok_nb){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    memset(W.cpu,1,(size_t)K*N);                                  /* int8 weight all-1 (layout-agnostic) */
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)M*N*4,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}   /* int32 out */
    int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=1; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);  /* act all-1 -> out=K */
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I8_N]; orki_synth_i8(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task*t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0xd; t->int_mask=0x300; t->int_clear=0x1ffff;
    t->regcfg_amount=108; t->regcmd_addr=(uint32_t)c->regcmd.dma; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub; memset(&sub,0,sizeof sub); sub.task_number=1; sub.task_obj_addr=c->task.obj;
    sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1; sub.timeout=4000; sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    volatile int32_t *dbell=(volatile int32_t*)((char*)O.cpu + (size_t)(M*N-1)*4);  /* last output int32 = doorbell */
    const int32_t SENT=0x7ffffff;                                  /* matmul (all-1, K) can't produce this */
    volatile int32_t *o0=(volatile int32_t*)O.cpu, *ol=(volatile int32_t*)dbell;  /* check endpoints */
    /* ---- BLOCKING: flags=0x5 via the proper submit path (domain/bookkeeping) ---- */
    *o0=SENT; *ol=SENT; __asm__ volatile("dc cvac,%0"::"r"(o0):"memory"); __asm__ volatile("dc cvac,%0"::"r"(ol):"memory"); __asm__ volatile("dsb ish":::"memory"); /* seed endpoints (like the doorbell) so CIVAC can read fresh */
    sub.flags=0x5; orki_rknpu_submit_ioctl(fd,&sub,-1); orki_rknpu_submit_ioctl(fd,&sub,-1);  /* warm (mode + first-cold) */
    double t0=ork_now_us();
    for(int i=0;i<iters;i++){ sub.flags=0x5; if(orki_rknpu_submit_ioctl(fd,&sub,-1)){*ok_block=0;} }
    if(block_us)*block_us=(ork_now_us()-t0)/iters;
    { for(long s=0;s<2000000L && (*o0==SENT||*ol==SENT);s++){ __asm__ volatile("dc civac,%0"::"r"(o0):"memory"); __asm__ volatile("dc civac,%0"::"r"(ol):"memory"); __asm__ volatile("dsb ish":::"memory"); }
      *ok_block=(*o0==K && *ol==K);
      if(getenv("ORK_DBELL_DBG"))fprintf(stderr,"[dbg-block] o[0]=%d o[last]=%d K=%d\n",*o0,*ol,K); }
    /* ---- NONBLOCK + doorbell busy-poll: flags=0x7 (adds NONBLOCK 0x2), submit returns ~5µs, spin on the sentinel ---- */
    t0=ork_now_us(); int polled_ok=1;
    for(int i=0;i<iters;i++){
        *dbell=SENT; __asm__ volatile("dc cvac, %0"::"r"(dbell):"memory"); __asm__ volatile("dsb ish":::"memory"); /* seed sentinel in DRAM */
        sub.flags=0x7;                                             /* PC|PINGPONG|NONBLOCK */
        if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ polled_ok=0; break; }
        long s=0; for(;s<20000000L;s++){ __asm__ volatile("dc civac, %0"::"r"(dbell):"memory"); __asm__ volatile("dsb ish":::"memory");
            if(*dbell!=SENT) break; }                              /* busy-poll: NPU overwrote the doorbell = done */
        if(s>=20000000L){ polled_ok=0; break; }                   /* poll timed out */
    }
    if(nb_us)*nb_us=(ork_now_us()-t0)/iters;
    orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE);
    { int32_t*o=O.cpu; *ok_nb=(polled_ok && o[0]==K && o[M*N-1]==K);
      if(getenv("ORK_DBELL_DBG"))fprintf(stderr,"[dbg-nb] o[0]=%d o[last]=%d K=%d polled_ok=%d\n",o[0],o[M*N-1],K,polled_ok); }
    orki_bdestroy(fd,&W); orki_bdestroy(fd,&O);
    return 0;
}

int ork_npu_overlap_prof(ork_npu *c,int M,int K,int N,int cpu_reps,int iters,
                         double *npu_solo,double *cpu_solo,double *overlap_wall,int *ok){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    memset(W.cpu,1,(size_t)K*N); orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)M*N*4,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}
    int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=1; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I8_N]; orki_synth_i8(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task*t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0xd; t->int_mask=0x300; t->int_clear=0x1ffff;
    t->regcfg_amount=108; t->regcmd_addr=(uint32_t)c->regcmd.dma; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub; memset(&sub,0,sizeof sub); sub.task_number=1; sub.task_obj_addr=c->task.obj;
    sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1; sub.timeout=4000; sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    volatile int32_t *dbell=(volatile int32_t*)((char*)O.cpu + (size_t)(M*N-1)*4);
    const int32_t SENT=0x7ffffff;
    const int RN=512; float*Wc=malloc((size_t)RN*RN*4),*xc=malloc(RN*4),*yc=malloc(RN*4);  /* CPU "router" state */
    if(!Wc||!xc||!yc){free(Wc);free(xc);free(yc);orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);return -2;}
    for(int i=0;i<RN*RN;i++)Wc[i]=(float)(((unsigned)i*2654435761u)>>28)*0.01f; for(int i=0;i<RN;i++)xc[i]=1.0f;
#define CPU_ROUTER() do{ for(int r=0;r<cpu_reps;r++){ for(int a=0;a<RN;a++){ float acc=0; const float*wr=Wc+(size_t)a*RN; \
        for(int b=0;b<RN;b++)acc+=wr[b]*xc[b]; yc[a]=acc; } xc[0]=yc[RN-1]*1e-9f; } }while(0)  /* cross-rep dep -> no DCE */
    sub.flags=0x5; orki_rknpu_submit_ioctl(fd,&sub,-1); orki_rknpu_submit_ioctl(fd,&sub,-1);  /* warm */
    int okk=1;
    /* (1) NPU solo: nonblock submit + doorbell poll, NO cpu work */
    double t0=ork_now_us();
    for(int i=0;i<iters;i++){ *dbell=SENT; __asm__ volatile("dc cvac,%0"::"r"(dbell):"memory"); __asm__ volatile("dsb ish":::"memory");
        sub.flags=0x7; if(orki_rknpu_submit_ioctl(fd,&sub,-1)){okk=0;break;}
        long s=0; for(;s<20000000L && *dbell==SENT;s++){ __asm__ volatile("dc civac,%0"::"r"(dbell):"memory"); __asm__ volatile("dsb ish":::"memory"); }
        if(s>=20000000L){okk=0;break;} }
    if(npu_solo)*npu_solo=(ork_now_us()-t0)/iters;
    /* (2) CPU solo: the router GEMVs alone, no NPU */
    t0=ork_now_us(); for(int i=0;i<iters;i++){ CPU_ROUTER(); } if(cpu_solo)*cpu_solo=(ork_now_us()-t0)/iters;
    volatile float sink=xc[0]; (void)sink;
    /* (3) OVERLAP: nonblock submit, run the router in the shadow, THEN poll for NPU completion */
    t0=ork_now_us();
    for(int i=0;i<iters;i++){ *dbell=SENT; __asm__ volatile("dc cvac,%0"::"r"(dbell):"memory"); __asm__ volatile("dsb ish":::"memory");
        sub.flags=0x7; if(orki_rknpu_submit_ioctl(fd,&sub,-1)){okk=0;break;}
        CPU_ROUTER();                                              /* CPU works while the NPU crunches */
        long s=0; for(;s<20000000L && *dbell==SENT;s++){ __asm__ volatile("dc civac,%0"::"r"(dbell):"memory"); __asm__ volatile("dsb ish":::"memory"); }
        if(s>=20000000L){okk=0;break;} }
    if(overlap_wall)*overlap_wall=(ork_now_us()-t0)/iters;
    *ok=okk;
    free(Wc);free(xc);free(yc); orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);
#undef CPU_ROUTER
    return 0;
}

int ork_npu_probe_bs_scale(ork_npu *c,const int8_t *a,const int8_t *scale,int M,int N,int8_t *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<16||N>8192||(N&15)) return -2;
    #define BSCUBE(m,n) (((n)/16)*(M*16) + (m)*16 + ((n)%16))
    size_t sz=(size_t)M*N; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,-1), O=orki_bcreate(fd,sz,0x403,-1), S=orki_bcreate(fd,4096,0x403,-1);
    if(!A.cpu||!O.cpu||!S.cpu){ orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&S); return -2; }
    memset(A.cpu,0,sz); memset(O.cpu,0,sz); memset(S.cpu,0,4096);
    { int8_t*ac=A.cpu; for(int m=0;m<M;m++)for(int n=0;n<N;n++) ac[BSCUBE(m,n)]=a[(size_t)m*N+n]; }
    /* per-channel scale vector b[N]: try the EW-operand cube layout for a single width row (width=1). */
    { int8_t*sc=S.cpu; for(int n=0;n<N;n++) sc[(n/16)*16 + (n%16)]=scale[n]; }
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&S,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (XP_SDP=RC_SDPKW, keep-warm-aware; == the old inline reset) */
    /* EW MUL out=a*b, b (ERDMA 0x5038) as a PER-CHANNEL vector via ERDMA_DATA_MODE (0x5034 bits[31:30]). */
    uint32_t rc[REGCMD_MUL_N]; memcpy(rc,REGCMD_MUL,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_MUL_N,M,N);
    orki_setrn(rc,REGCMD_MUL_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);            /* output */
    orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5018,(uint32_t)A.dma);            /* input a (SRDMA, per-element) */
    orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5038,(uint32_t)S.dma);            /* scale b (ERDMA / EW_BASE) */
    #define ENV32(nm,def) (getenv(nm)?(uint32_t)strtoul(getenv(nm),0,0):(uint32_t)(def))
    orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5034,ENV32("ORK_ERDMA",0x40000000)); /* ERDMA_CFG: ERDMA_DATA_MODE bits[31:30] (sweep per-channel) */
    orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_SCALE,ENV32("ORK_BS_GAIN",0x00004000)); orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_SHIFT,28); /* out gain */
    #undef ENV32
    if(getenv("ORK_BS_DUMP")){ for(int k=0;k+1<REGCMD_MUL_N;k+=2){ unsigned rg=rc[k]&0xffff,ln=rc[k+1]>>16; uint32_t v=((rc[k]>>16)&0xffff)|((rc[k+1]&0xffff)<<16); if(rg==0x4020||rg==0x4070||rg==0x5018||rg==0x5034||rg==0x5038) printf("  reg=%04x lane=%04x val=%08x\n",rg,ln,v);} }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub; memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=1;
    sub.task_obj_addr=c->task.obj; sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1;
    sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; sub.timeout=orki_ew_timeout_ms();
    int ok=-1; double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; if(us)*us=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ int8_t*oc=O.cpu; for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=oc[BSCUBE(m,n)]; }
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&O); orki_bdestroy(fd,&S);
    #undef BSCUBE
    return ok;
}

int ork_npu_probe_batch(ork_npu*c,int ntask,int K,int N,double*us_unbatched,double*us_batched){
    int fd=c->fd,CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||ntask<1||ntask>32) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    memset(W.cpu,1,(size_t)K*N); orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)N*4,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}
    int8_t*ad=c->Af.cpu; memset(ad,1,K); orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I8_N]; orki_synth_i8(rc,1,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_CBUF_CON0,0xb1);
    struct buf extra[2] = {W, O};
    if (orki_validate_regcmd("probe_batch", c, rc, REGCMD_I8_N, NULL, extra, 2)) { orki_bdestroy(fd,&W); orki_bdestroy(fd,&O); return -1; }
    for (int i = 0; i < ntask; i++) {
        memcpy((char*)c->regcmd.cpu + i * sizeof(rc), rc, sizeof(rc));
    }
    orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task*t=c->task.cpu;                 /* task[] array: ntask tasks, separate regcmd spaces */
    for(int i=0;i<ntask;i++){memset(&t[i],0,sizeof t[i]);t[i].flags=0;t[i].op_idx=i;t[i].enable_mask=0xd;t[i].int_mask=0x300;t[i].int_clear=0x1ffff;t[i].regcfg_amount=108;t[i].regcmd_addr=c->regcmd.dma + i * sizeof(rc);}
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    /* single-core: set all subcore_task entries to avoid kernel UAPI timeout/Oops */
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.timeout=orki_mm_timeout_ms();
    sub.task_number=1; sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,1};
    if(orki_rknpu_submit_ioctl(fd,&sub,-1)){orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);return -1;} orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); /* warm */
    double t0=ork_now_us();                          /* (a) ntask separate ioctls */
    for(int i=0;i<ntask;i++){ sub.task_number=1; sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,1};
        if(orki_rknpu_submit_ioctl(fd,&sub,-1)){orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);return -1;} orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); }
    *us_unbatched=ork_now_us()-t0;
    sub.task_number=ntask; sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)ntask};
    t0=ork_now_us();                                 /* (b) one ioctl, ntask tasks */
    if(orki_rknpu_submit_ioctl(fd,&sub,-1)){perror("batched SUBMIT");orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);return -1;} orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE);
    *us_batched=ork_now_us()-t0;
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O); return 0;
}

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
