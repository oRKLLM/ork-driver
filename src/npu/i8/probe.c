/* npu/i8/probe.c — int8 RE probes: the fuzz-override hooks and the MATMUL probes (mtile, single-submit, int8-out, mm).
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

/* RE (int8 batch-mode A/B): run one int8 matmul via synth_i8 and return the RAW int32 output (no
 * requantize, no de-tile). Mirrors probe_i8_out8 (weight [NT][KT][32][32], A raw-copied to Af). Any
 * ork_i8_fuzz_add overrides apply inside synth_i8, so a caller can flip int8 stream->batch (0x405c=0 etc.)
 * and observe the resulting output layout. Output buffer is 2*M*N int32 (room for a stride-2 batch layout).
 * 0/ok, -1 wedged, -2 dims. Submit timeout honors ORK_I4_PROBE_TO_MS (fast-fail for wedgy fuzz values). */
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
