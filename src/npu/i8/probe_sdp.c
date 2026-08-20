/* npu/i8/probe_sdp.c — int8 SDP / element-wise RE probes: ewmul (three template variants), mul, add, standalone SiLU.
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
int ork_i8_npu_probe_ewmul(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,const int8_t *G,
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
    orki_i8_synth(base,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    orki_splice_ew_lane(rc,base);                                    /* insert the 0x50xx second-input lane */
    orki_i8_set_ewmul(rc,M,N,0,mult,shift,(uint32_t)Gb.dma);           /* int8 requant + EW-mul + 2nd-input addr */
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

int ork_i8_npu_probe_ewmul_tmpl(ork_npu *c,const void*in,int Isz,const void*wt,int Wsz,
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

int ork_i8_npu_probe_ewmul_lin(ork_npu *c,const int8_t *A,const int8_t *B,const int8_t *G,int8_t *C,double *us){
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

int ork_i8_npu_probe_mul(ork_npu *c,const int8_t *a,const int8_t *b,int n,int8_t *out,double *us){
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

/* Standalone int8 element-wise ADD — RE probe (settable scale regs to decode the add structure). out[m][n] from
 * the 2-input SDP op with ALU=add (REGCMD_ADD), reprogrammed to (M,N) by orki_set_mul_geom. Caller sets za(0x4044),
 * zb(0x4074), zo(0x4080), out scale mult(0x4084)/shift(0x4088), and the b-operand scale bscale(0x4078); the
 * ALU-mode regs (0x4040/0x4048/0x4070) stay from the template. a/b/out int8 [M*N], N%16==0. 0/ok,-1,-2,-3. */
int ork_i8_npu_probe_add(ork_npu *c,const int8_t *a,const int8_t *b,int M,int N,
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

/* ── Standalone on-NPU SiLU (activation-LUT SDP op) — RE probe ─────────────────────────────────────
 * Applies the PPU activation LUT to a SINGLE int8 memory input [M][N] via the standalone 69-reg/enable=0x18
 * SDP op (REGCMD_SILU_STD), reprogrammed to (M,N) by orki_set_mul_geom. Two submits on the single-stream NPU:
 * (1) LUT-load (REGCMD_SILU_LUT, streams the int16 curve into PPU SRAM); (2) the standalone op reads it.
 * SDP math: idx=(in*R)>>6 + C0(idx_off); out=clamp_i8(R*LUT-interp(idx) + out_bias); R=r_mult/2^r_shift.
 * The caller supplies the scale regs (r_mult,r_shift,out_bias,idx_off,cfg4064,cfg4068) and the LUT; lut==NULL
 * keeps the captured curve. This is the RE/calibration entry (measure idx(in) with a ramp LUT, then build the
 * silu curve at those indices — same 2-pass scheme as ork_mm_silu_build_lut but through THIS op, not a matmul).
 * in/out int8 [M*N] row-major; N%16==0. 0/ok, -1 wedged, -2 bad shape, -3 non-rk3588. */
int ork_i8_npu_probe_silu_std(ork_npu *c,const int8_t *in,int M,int N,
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
