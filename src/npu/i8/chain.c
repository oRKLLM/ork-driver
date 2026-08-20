/* npu/i8/chain.c — PC-chained int8 graphs: run_chain_i8 and its FFN/SDP/gsilu variants, round-robin dispatch, the orkd layer/attn/ffn wrappers.
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
#include "orkd_proto.h"
#include "npu/internal.h"
#include "npu/core.h"
#include "npu/i8/i8.h"
#include "spine_kernels.h"

int ork_npu_chain_mm_silu_i16(ork_npu *c,const int16_t *in,int M,int N,double in_scale,double out_scale,
                              int16_t *out,int *mm_ran,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems, dom=c->dom_active;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;
    if(orki_silu_calibrate_idx16(c)) return -1;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)
    /* build the int16 silu LUT curve for (in_scale,out_scale) — same as orki_act_lut_i16 */
    static double qsum[1030]; static int qn[1030];
    for(int k=0;k<1030;k++){ qsum[k]=0; qn[k]=0; }
    for(int s=0;s<SILU16_NS;s++){ int k=c->silu_idx16[s]; if(k<0||k>1029)continue; qsum[k]+=-32768.0+s*SILU16_QSTEP; qn[k]++; }
    int16_t lut[1030]; int lo=-1,hi=-1;
    for(int k=0;k<1030;k++){ if(qn[k]){ if(lo<0)lo=k; hi=k; double q_in=qsum[k]/qn[k]; double val=orki_silu_f(q_in*in_scale)/out_scale;
        long q=lround(val); if(q>32767)q=32767; if(q<-32768)q=-32768; lut[k]=(int16_t)q; } else lut[k]=0; }
    if(lo<0) return -1;
    for(int k=0;k<lo;k++)lut[k]=lut[lo]; for(int k=hi+1;k<1030;k++)lut[k]=lut[hi];
    for(int k=lo;k<=hi;k++){ if(qn[k])continue; int a=k,b=k; while(a>lo&&!qn[a])a--; while(b<hi&&!qn[b])b++;
        lut[k]=(int16_t)(lut[a]+(lut[b]-lut[a])*(k-a)/(b-a)); }

    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,dom), O=orki_bcreate(fd,sz,0x403,dom);
    struct buf Lrc=orki_bcreate(fd,(size_t)REGCMD_SILU_LUT_N*4,0x403,dom), Lsc=orki_bcreate(fd,4096,0x403,dom);
    struct buf Wd=orki_bcreate(fd,32*32,0x403,dom), Ad=orki_bcreate(fd,32,0x403,dom), Cd=orki_bcreate(fd,32*4,0x403,dom);
    if(!A.cpu||!O.cpu||!Lrc.cpu||!Lsc.cpu||!Wd.cpu||!Ad.cpu||!Cd.cpu){ goto fail; }
    memset(A.cpu,0,sz); memset(O.cpu,0,sz); memset(Cd.cpu,0,32*4);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) *(int16_t*)((char*)A.cpu+EWCUBEH(m,n))=in[m*N+n];
    { int8_t*wd=Wd.cpu,*ad=Ad.cpu; for(int i=0;i<32*32;i++)wd[i]=1; for(int i=0;i<32;i++)ad[i]=1; }   /* all-1s -> C[n]=32 iff task0 matmul ran */
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&Wd,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&Ad,RKNPU_MEM_SYNC_TO_DEVICE);

    /* submit 1: LUT-load (separate; loads the silu LUT into SDP SRAM, ping-pong OFF) */
    memcpy(Lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
    orki_setrn((uint32_t*)Lrc.cpu,REGCMD_SILU_LUT_N,RK_DPU_DST_BASE_ADDR,(uint32_t)Lsc.dma);
    { uint32_t*lr=(uint32_t*)Lrc.cpu; int j=0; for(int k=0;k+1<REGCMD_SILU_LUT_N;k+=2){ if((lr[k]&0xffff)==0x4104){ int32_t v=(j<1030)?(int32_t)lut[j]:0; j++;
        lr[k]=0x4104|((uint32_t)(v&0xffff)<<16); lr[k+1]=(0x1001u<<16)|(((uint32_t)v>>16)&0xffff); } } }
    orki_bsync(fd,&Lrc,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task*t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=1097; t->regcmd_addr=Lrc.dma;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit s;memset(&s,0,sizeof s);s.flags=0x1;s.task_number=1;s.task_obj_addr=c->task.obj;s.core_mask=RKNPU_CORE0_MASK;s.fence_fd=-1;s.timeout=orki_ew_timeout_ms();s.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      if(orki_rknpu_submit_ioctl(fd,&s,dom)) goto fail; }

    /* build the 2-task chain in c->regcmd: [0]=matmul (32x32) with descriptor -> [1]=silu */
    { uint32_t *mm=(uint32_t*)c->regcmd.cpu;                  /* task0 regcmd at word 0 */
      uint32_t *si=(uint32_t*)((char*)c->regcmd.cpu + (size_t)REGCMD_I8_N*4);   /* task1 regcmd */
      memset(mm,0,REGCMD_I8_N*4);
      orki_synth_i8(mm,1,32,32,(uint32_t)Ad.dma,(uint32_t)Wd.dma,(uint32_t)Cd.dma,1,CBUF,0);
      uint64_t nx=c->regcmd.dma+(size_t)REGCMD_I8_N*4;         /* next = silu regcmd addr */
      mm[216]=0x0010|((nx&0xffff)<<16); mm[217]=(0x0101u<<16)|((nx>>16)&0xffff);
      mm[218]=0x0014|(((69+3)/2)<<16);  mm[219]=(0x0101u<<16)|0;   /* next register-amount = (silu regcfg+3)/2 */
      memcpy(si,REGCMD_SILU_STD_I16,(size_t)REGCMD_SILU_STD_I16_N*4);
      orki_set_mul_geom(si,REGCMD_SILU_STD_I16_N,M,N);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_SDP_5040,0);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_SDP_5038,0);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_SDP_5018,(uint32_t)A.dma);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_SCALE,0x4000u);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_SHIFT,14u);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_OFFSET,0u);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_R4110,ORK_SILU16_IDXOFF);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_BN_ALU_CFG,ORK_SILU16_C4064);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_BN_MUL_CFG,ORK_SILU16_C4068);
      orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
      struct rknpu_task*tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,2*sizeof *tk);
      tk[0].enable_mask=0xd;  tk[0].int_mask=0x300; tk[0].int_clear=0x1ffff; tk[0].regcfg_amount=108; tk[0].regcmd_addr=c->regcmd.dma;
      tk[1].enable_mask=0x18; tk[1].int_mask=0x300; tk[1].int_clear=0x1ffff; tk[1].regcfg_amount=69;  tk[1].regcmd_addr=nx;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit s;memset(&s,0,sizeof s);s.flags=0x1;s.task_number=2;s.task_obj_addr=c->task.obj;s.core_mask=RKNPU_CORE0_MASK;s.fence_fd=-1;s.timeout=orki_ew_timeout_ms();
      s.subcore_task[0]=(struct rknpu_subcore_task){0,2};
      double t0=ork_now_us();
      if(orki_rknpu_submit_ioctl(fd,&s,dom)) goto fail;
      orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd,&Cd,RKNPU_MEM_SYNC_FROM_DEVICE);
      if(us)*us=ork_now_us()-t0;
    }
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[m*N+n]=*(int16_t*)((char*)O.cpu+EWCUBEH(m,n));
    if(mm_ran){ int32_t*cd=Cd.cpu; int nz=0; for(int i=0;i<32;i++) if(cd[i]!=0) nz=1; *mm_ran=nz; }   /* did task0 (matmul) run? */
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);orki_bdestroy(fd,&Wd);orki_bdestroy(fd,&Ad);orki_bdestroy(fd,&Cd);
    #undef EWCUBEH
    return 0;
fail:
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);orki_bdestroy(fd,&Wd);orki_bdestroy(fd,&Ad);orki_bdestroy(fd,&Cd);
    #undef EWCUBEH
    return -1;
}

int ork_npu_chain_gatesilu_i16(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,
                               int mult,int shift,double in_scale,double out_scale,
                               int16_t *gate_out,int16_t *out,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems, dom=c->dom_active;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64||(N&7)) return -2;
    if(orki_silu_calibrate_idx16(c)) return -1;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)
    /* silu LUT for (in_scale,out_scale) -- same construction as orki_act_lut_i16 / Phase-0 */
    static double qsum[1030]; static int qn[1030];
    for(int k=0;k<1030;k++){ qsum[k]=0; qn[k]=0; }
    for(int s=0;s<SILU16_NS;s++){ int k=c->silu_idx16[s]; if(k<0||k>1029)continue; qsum[k]+=-32768.0+s*SILU16_QSTEP; qn[k]++; }
    int16_t lut[1030]; int lo=-1,hi=-1;
    for(int k=0;k<1030;k++){ if(qn[k]){ if(lo<0)lo=k; hi=k; double q_in=qsum[k]/qn[k]; double val=orki_silu_f(q_in*in_scale)/out_scale;
        long q=lround(val); if(q>32767)q=32767; if(q<-32768)q=-32768; lut[k]=(int16_t)q; } else lut[k]=0; }
    if(lo<0) return -1;
    for(int k=0;k<lo;k++)lut[k]=lut[lo]; for(int k=hi+1;k<1030;k++)lut[k]=lut[hi];
    for(int k=lo;k<=hi;k++){ if(qn[k])continue; int a=k,b=k; while(a>lo&&!qn[a])a--; while(b<hi&&!qn[b])b++;
        lut[k]=(int16_t)(lut[a]+(lut[b]-lut[a])*(k-a)/(b-a)); }

    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,dom);
    struct buf G=orki_bcreate(fd,sz,0x403,dom), O=orki_bcreate(fd,sz,0x403,dom);       /* G = matmul int16 out = silu in */
    struct buf Lrc=orki_bcreate(fd,(size_t)REGCMD_SILU_LUT_N*4,0x403,dom), Lsc=orki_bcreate(fd,4096,0x403,dom);
    if(!W.cpu||!G.cpu||!O.cpu||!Lrc.cpu||!Lsc.cpu){ fprintf(stderr,"[gatesilu] buffer alloc failed\n"); goto gfail; }
    { int NN=N/32,KT=K/32; int8_t*bb=W.cpu;                                  /* int8 weight tile [Nt][Kt][32][32] */
      for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)]; }
    memset(G.cpu,0,sz); memset(O.cpu,0,sz);
    { int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; }
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&G,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);

    /* submit 1: silu LUT-load (separate, ping-pong off). ORK_GS_NOLUT skips it -> matmul becomes the FIRST
     * NPU op after orki_act(RESET) (diagnostic: does a preceding SDP LUT-load poison the following int8 matmul?). */
    if(!getenv("ORK_GS_NOLUT")){
    memcpy(Lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
    orki_setrn((uint32_t*)Lrc.cpu,REGCMD_SILU_LUT_N,RK_DPU_DST_BASE_ADDR,(uint32_t)Lsc.dma);
    { uint32_t*lr=(uint32_t*)Lrc.cpu; int j=0; for(int k=0;k+1<REGCMD_SILU_LUT_N;k+=2){ if((lr[k]&0xffff)==0x4104){ int32_t v=(j<1030)?(int32_t)lut[j]:0; j++;
        lr[k]=0x4104|((uint32_t)(v&0xffff)<<16); lr[k+1]=(0x1001u<<16)|(((uint32_t)v>>16)&0xffff); } } }
    orki_bsync(fd,&Lrc,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task*t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=1097; t->regcmd_addr=Lrc.dma;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit s;memset(&s,0,sizeof s);s.flags=0x1;s.task_number=1;s.task_obj_addr=c->task.obj;s.core_mask=RKNPU_CORE0_MASK;s.fence_fd=-1;s.timeout=orki_ew_timeout_ms();s.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      if(orki_rknpu_submit_ioctl(fd,&s,dom)){ fprintf(stderr,"[gatesilu] LUT-load submit failed (errno=%d)\n",errno); goto gfail; } }
    }

    /* build the 2 chained programs: [0] matmul int16-out -> G ; [1] silu G -> O */
    static uint32_t mm[REGCMD_I8_N], si[REGCMD_SILU_STD_I16_N];
    orki_synth_i8(mm,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)G.dma,1,CBUF,0);
    orki_set_i16_out(mm,N,0,mult,shift);
    memcpy(si,REGCMD_SILU_STD_I16,(size_t)REGCMD_SILU_STD_I16_N*4);
    orki_set_mul_geom(si,REGCMD_SILU_STD_I16_N,M,N);
    orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_SDP_5040,0);
    orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_SDP_5038,0);
    orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_SDP_5018,(uint32_t)G.dma);            /* silu INPUT = matmul OUTPUT */
    orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_SCALE,0x4000u);
    orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_SHIFT,14u);
    orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_OFFSET,0u);
    orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_R4110,ORK_SILU16_IDXOFF);
    orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_BN_ALU_CFG,ORK_SILU16_C4064);
    orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_BN_MUL_CFG,ORK_SILU16_C4068);

    double t0=ork_now_us();
    if(getenv("ORK_GS_SEQ")){
        /* KERNEL-SEQUENCED bridge (the accel/rocket model): matmul(->G) and silu(<-G) as TWO separate
         * task_number=1 submits, no reset between (act RESET ran above). Isolates the int16 DATA bridge
         * (does the matmul int16 output layout match the silu EWCUBEH input?) from the HW chain-walk. */
        fprintf(stderr,"[gatesilu-seq] submitting matmul (int16-out) task_number=1...\n");
        memcpy(c->regcmd.cpu,mm,REGCMD_I8_N*4); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
        { struct rknpu_task*t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0xd; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=108; t->regcmd_addr=(uint32_t)c->regcmd.dma;
          orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
          struct rknpu_submit s;memset(&s,0,sizeof s);s.flags=0x1;s.task_number=1;s.task_obj_addr=c->task.obj;s.core_mask=RKNPU_CORE0_MASK;s.fence_fd=-1;s.timeout=orki_ew_timeout_ms();s.subcore_task[0]=(struct rknpu_subcore_task){0,1};
          if(orki_rknpu_submit_ioctl(fd,&s,dom)){ fprintf(stderr,"[gatesilu-seq] matmul submit failed errno=%d\n",errno); goto gfail; } }
        fprintf(stderr,"[gatesilu-seq] matmul OK; submitting silu task_number=1...\n");
        orki_bsync(fd,&G,RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd,&G,RKNPU_MEM_SYNC_TO_DEVICE);   /* keep G resident for the silu */
        memcpy(c->regcmd.cpu,si,REGCMD_SILU_STD_I16_N*4); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
        { struct rknpu_task*t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=69; t->regcmd_addr=(uint32_t)c->regcmd.dma;
          orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
          struct rknpu_submit s;memset(&s,0,sizeof s);s.flags=0x1;s.task_number=1;s.task_obj_addr=c->task.obj;s.core_mask=RKNPU_CORE0_MASK;s.fence_fd=-1;s.timeout=orki_ew_timeout_ms();s.subcore_task[0]=(struct rknpu_subcore_task){0,1};
          if(orki_rknpu_submit_ioctl(fd,&s,dom)){ fprintf(stderr,"[gatesilu-seq] silu submit failed errno=%d\n",errno); goto gfail; } }
    } else {
        ork_chain_prog progs[2] = { { mm, REGCMD_I8_N, 0xd, 108, 216 }, { si, REGCMD_SILU_STD_I16_N, 0x18, 69, -1 } };
        int crc=ork_npu_chain_progs(c,2,progs,dom);
        if(crc){ fprintf(stderr,"[gatesilu] chain_progs rc=%d (errno=%d) — -2 no-descriptor-slot/bad-args, -1 submit wedge\n",crc,errno); goto gfail; }
    }
    orki_bsync(fd,&G,RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE);
    if(us)*us=ork_now_us()-t0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){
        if(gate_out) gate_out[m*N+n]=*(int16_t*)((char*)G.cpu+EWCUBEH(m,n));
        out[m*N+n]=*(int16_t*)((char*)O.cpu+EWCUBEH(m,n));
    }
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&G);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);
    #undef EWCUBEH
    return 0;
gfail:
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&G);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);
    #undef EWCUBEH
    return -1;
}

int ork_npu_probe_chain_i8(ork_npu *c, int S, int K, int N, const int8_t *A, const int8_t *B, int32_t *C) {
    int fd = c->fd, CBUF = c->soc->cbuf_elems;
    if (K % 32 || N % 32 || N > c->soc->nmax || S < 1 || S > 32) return -2;
    struct buf W = orki_bcreate(fd, (size_t)K * N, 0x403,-1); if (!W.cpu) return -2;
    int NN = N / 32, KT = K / 32; int8_t *bb = W.cpu;
    for (int nt = 0; nt < NN; nt++) for (int kt = 0; kt < KT; kt++) for (int nl = 0; nl < 32; nl++) for (int kk = 0; kk < 32; kk++)
        bb[(size_t)nt * KT * 32 * 32 + (size_t)kt * 32 * 32 + nl * 32 + kk] = B[(size_t)(kt * 32 + kk) * N + (nt * 32 + nl)];
    orki_bsync(fd, &W, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd, &W, RKNPU_MEM_SYNC_TO_DEVICE);
    
    struct buf O = orki_bcreate(fd, (size_t)S * 4096, 0x403,-1); if (!O.cpu) { orki_bdestroy(fd, &W); return -2; }
    
    int8_t *ad = c->Af.cpu;
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < K; j++) ad[i * K + j] = A[i * K + j];
    }
    orki_bsync(fd, &c->Af, RKNPU_MEM_SYNC_TO_DEVICE);
    
    orki_act(fd, RKNPU_ACT_RESET, 0);
    
    uint32_t rc[REGCMD_I8_N];
    for (int i = 0; i < S; i++) {
        uint32_t act_dma = (uint32_t)(c->Af.dma + i * K);
        uint32_t out_dma = (uint32_t)(O.dma + i * 4096);
        orki_synth_i8(rc, 1, K, N, act_dma, (uint32_t)W.dma, out_dma, 1, CBUF, 0);
        orki_setrn(rc, REGCMD_I8_N,RK_CNA_CBUF_CON0, 0xb1);
        struct buf extra[2] = {W, O};
        if (orki_validate_regcmd("probe_chain_i8", c, rc, REGCMD_I8_N, NULL, extra, 2)) { orki_bdestroy(fd,&W); orki_bdestroy(fd,&O); return -1; }
        
        if (i < S - 1) {
            uint64_t next_dma = c->regcmd.dma + (i + 1) * REGCMD_I8_N * 4;
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
        memcpy((char*)c->regcmd.cpu + i * sizeof(rc), rc, sizeof(rc));
    }
    orki_bsync(fd, &c->regcmd, RKNPU_MEM_SYNC_TO_DEVICE);
    
    struct rknpu_task *t = c->task.cpu;
    memset(t, 0, S * sizeof(struct rknpu_task));
    for (int i = 0; i < S; i++) {
        t[i].enable_mask = 0xd;
        t[i].int_mask = 0x300;
        t[i].int_clear = 0x1ffff;
        t[i].regcfg_amount = 108;
        t[i].regcmd_addr = c->regcmd.dma + i * REGCMD_I8_N * 4;
    }
    orki_bsync(fd, &c->task, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
    
    struct rknpu_submit sub; memset(&sub, 0, sizeof(sub));
    sub.task_start = 0;
    sub.task_number = 1; // HIDE the chained tasks from the kernel! The kernel rejects task_number > 1.
    sub.task_counter = 0;
    sub.priority = 0;
    sub.task_obj_addr = c->task.obj;
    sub.core_mask = RKNPU_CORE_AUTO_MASK;
    sub.subcore_task[0].task_start = 0;
    sub.subcore_task[0].task_number = 1; // Hide from subcore logic too!
    
    int ok = -1;
    for (int rep = 0; rep < 2; rep++) {
        sub.timeout = orki_mm_timeout_ms();
        if (orki_rknpu_submit_ioctl(fd, &sub, -1)) { ok = -1; continue; }
        orki_bsync(fd, &O, RKNPU_MEM_SYNC_FROM_DEVICE);
        for (int i = 0; i < S; i++) {
            memcpy(C + i * N, (char*)O.cpu + i * 4096, (size_t)N * 4);
        }
        ok = 0;
    }
    
    orki_bdestroy(fd, &W); orki_bdestroy(fd, &O);
    return ok;
}

int orki_chain_fullk_mcap_i8(ork_npu *c, int K) {
    int RB = 2 * c->soc->cbuf_elems, R = RB / K; if (R < 1) R = 1;
    { int rp2 = 1; while (rp2 * 2 <= R) rp2 *= 2; R = rp2; }
    double scale = (double)K / 512.0; int base = (int)(177.0 - 15.0 * (scale - 1.0)), slope = (int)(15.0 * scale);
    int mg_max = base >= 0x1b ? (base - 0x1b) / slope + 1 : 0;
    int chunk = mg_max * 64; if (chunk < 1) chunk = 1;   /* schedule-valid max rows (mg_max*64), not R-1 — see "weight-DMA amortization" in AGENTS.md */
    return chunk;
}

int ork_mm_run_chain_i8(ork_npu *c, int S, const ork_mm_task_i8 *tasks) {
    if (c && c->daemon){   /* Path B: fused chain on the daemon (all task weights are daemon-resident is_orkd) */
        if (S < 1) return -2;
        orkd_chain_task_c *ct = malloc((size_t)S * sizeof *ct);
        if (!ct) return -1;
        int ok = 1;
        for (int i = 0; i < S; i++){ ork_w *w = tasks[i].w;
            if (!w || !w->is_orkd){ ok = 0; break; }
            ct[i].weight_id = w->orkd_id; ct[i].M = tasks[i].M; ct[i].K = w->K; ct[i].N = w->N; ct[i].A = tasks[i].A; ct[i].C = tasks[i].C; }
        if (ok) orkd_set_op_domain(c->daemon, (uint32_t)tasks[0].w->domain);   /* v2: a chain is single-domain — carry it (co-resident tasks share it) */
        int rc = ok ? orkd_run_chain_i8(c->daemon, S, ct) : -2;
        free(ct);
        return rc;
    }
    return orki_run_chain_i8_impl(c, S, tasks, NULL, -1); }

int ork_mm_layer_i8(ork_npu *c, const struct ork_layer_dims *d,
                    ork_w *wq, ork_w *wk, ork_w *wv, ork_w *wo, ork_w *wg, ork_w *wu, ork_w *wd,
                    const float *attn_norm, const float *q_norm, const float *ffn_norm,
                    const float *x, const float *Kc, const float *Vc, float *x_out){
    if (!c || !d || !wq||!wk||!wv||!wo||!wg||!wu||!wd || !attn_norm||!q_norm||!ffn_norm||!x||!Kc||!Vc||!x_out) return -3;
    if (c->daemon){   /* orkd transport: weights must be daemon-resident; forward the whole layer in one round-trip */
        if (!wq->is_orkd||!wk->is_orkd||!wv->is_orkd||!wo->is_orkd||!wg->is_orkd||!wu->is_orkd||!wd->is_orkd) return -3;
        struct orkd_layer h; memset(&h,0,sizeof h);
        h.wq=wq->orkd_id; h.wk=wk->orkd_id; h.wv=wv->orkd_id; h.wo=wo->orkd_id;
        h.wg=wg->orkd_id; h.wu=wu->orkd_id; h.wd=wd->orkd_id;
        h.D=d->D; h.H=d->H; h.Hkv=d->Hkv; h.dk=d->dk; h.dv=d->dv; h.Nff=d->Nff; h.nkv=d->nkv;
        h.pos=d->pos; h.attn_scale=d->attn_scale; h.rope_base=d->rope_base; h.domain=wq->domain;
        return orkd_layer_i8(c->daemon, &h, attn_norm, q_norm, ffn_norm, x, Kc, Vc, x_out);
    }
    /* direct: run the layer locally on ctx's NPU (spine CPU glue + doorbell matmuls, same-thread coherent) */
    int D=(int)d->D,H=(int)d->H,Hkv=(int)d->Hkv,dk=(int)d->dk,dv=(int)d->dv,Nff=(int)d->Nff,nkv=(int)d->nkv;
    int Nq=H*dk,Nkv=Hkv*dk,rk2=(Hkv>0)?H/Hkv:1;
    int rc=-1;
    int32_t *Cq=ork_dma_alloc(c,(size_t)Nq*4),*Ck=ork_dma_alloc(c,(size_t)Nkv*4),*Cv=ork_dma_alloc(c,(size_t)Nkv*4),
            *Co=ork_dma_alloc(c,(size_t)D*4),*Cg=ork_dma_alloc(c,(size_t)Nff*4),*Cu=ork_dma_alloc(c,(size_t)Nff*4),*Cd=ork_dma_alloc(c,(size_t)D*4);
    int8_t *xn8=malloc(D),*ao8=malloc(Nq),*xf8=malloc(D),*ac8=malloc(Nff);
    float *xo=malloc((size_t)D*4),*qf=malloc((size_t)Nq*4),*ao=malloc((size_t)Nq*4),*x1=malloc((size_t)D*4),
          *xf=malloc((size_t)D*4),*gf=malloc((size_t)Nff*4),*uf=malloc((size_t)Nff*4),*act=malloc((size_t)Nff*4);
    if (Cq&&Ck&&Cv&&Co&&Cg&&Cu&&Cd&&xn8&&ao8&&xf8&&ac8&&xo&&qf&&ao&&x1&&xf&&gf&&uf&&act){
        uint64_t wset[7]={(uint64_t)(uintptr_t)wq,(uint64_t)(uintptr_t)wk,(uint64_t)(uintptr_t)wv,(uint64_t)(uintptr_t)wo,
                          (uint64_t)(uintptr_t)wg,(uint64_t)(uintptr_t)wu,(uint64_t)(uintptr_t)wd};
        int warm_hit = c->layer_warmed && !memcmp(c->layer_warm, wset, sizeof wset);
        if (!warm_hit){   /* warm the doorbell ONCE per weight-set (a cold doorbell returns garbage); steady-state per-token hits */
            int8_t *wa=calloc(Nff<D?D:Nff,1); if(wa) memset(wa,1,(size_t)(Nff<D?D:Nff));
            if (wa){ orki_layer_mm(c,wq,wa,D,Nq,Cq);orki_layer_mm(c,wk,wa,D,Nkv,Ck);orki_layer_mm(c,wv,wa,D,Nkv,Cv);orki_layer_mm(c,wo,wa,Nq,D,Co);
                     orki_layer_mm(c,wg,wa,D,Nff,Cg);orki_layer_mm(c,wu,wa,D,Nff,Cu);orki_layer_mm(c,wd,wa,Nff,D,Cd); free(wa); }
            memcpy(c->layer_warm, wset, sizeof wset); c->layer_warmed=1;
        }
        float sx=(spine_rmsnorm(x,attn_norm,D,1e-6f,xo), spine_quant(xo,D,xn8));
        int m = orki_layer_mm(c,wq,xn8,D,Nq,Cq)||orki_layer_mm(c,wk,xn8,D,Nkv,Ck)||orki_layer_mm(c,wv,xn8,D,Nkv,Cv);
        if (!m){
            for (int i=0;i<Nq;i++) qf[i]=Cq[i]/sx;
            for (int h=0;h<H;h++){ float*qh=qf+(size_t)h*dk; spine_rmsnorm(qh,q_norm,dk,1e-6f,qh); spine_rope_neox(qh,dk,(int)d->pos,(float)d->rope_base);
                spine_attn(qh, Kc+(size_t)(h/rk2)*nkv*dk, Vc+(size_t)(h/rk2)*nkv*dv, nkv,dk,dv,(float)d->attn_scale, ao+(size_t)h*dv); }
            float sa=spine_quant(ao,Nq,ao8);
            if (!orki_layer_mm(c,wo,ao8,Nq,D,Co)){
                for (int i=0;i<D;i++) x1[i]=x[i]+Co[i]/sa;
                float sf=(spine_rmsnorm(x1,ffn_norm,D,1e-6f,xf), spine_quant(xf,D,xf8));
                if (!(orki_layer_mm(c,wg,xf8,D,Nff,Cg)||orki_layer_mm(c,wu,xf8,D,Nff,Cu))){
                    for (int i=0;i<Nff;i++){ gf[i]=Cg[i]/sf; uf[i]=Cu[i]/sf; } spine_silu_glu(gf,uf,Nff,act);
                    float sac=spine_quant(act,Nff,ac8);
                    if (!orki_layer_mm(c,wd,ac8,Nff,D,Cd)){ for (int i=0;i<D;i++) x_out[i]=x1[i]+Cd[i]/sac; rc=0; }
                }
            }
        }
    }
    if(Cq)ork_dma_free(c,Cq);if(Ck)ork_dma_free(c,Ck);if(Cv)ork_dma_free(c,Cv);if(Co)ork_dma_free(c,Co);
    if(Cg)ork_dma_free(c,Cg);if(Cu)ork_dma_free(c,Cu);if(Cd)ork_dma_free(c,Cd);
    free(xn8);free(ao8);free(xf8);free(ac8);free(xo);free(qf);free(ao);free(x1);free(xf);free(gf);free(uf);free(act);
    return rc;
}

int ork_mm_run_chain_i8_gsilu(ork_npu *c, int S, const ork_mm_task_i8 *tasks, int gate_task,
                              int r_mult, int r_shift, uint32_t out_bias, uint32_t idx_off, uint32_t cfg4068,
                              const int16_t *lut, int nlut) {
    if (gate_task < 0 || gate_task >= S || !lut) return -2;
    struct chain_silu_spec ss = { NULL, gate_task, -1, r_mult, r_shift, 0, 0, out_bias, idx_off, 0, cfg4068, lut, nlut };
    return orki_run_chain_i8_impl(c, S, tasks, &ss, -1);
}

int ork_mm_run_chain_i8_sdpsilu(ork_npu *c, int S, const ork_mm_task_i8 *tasks, int sdp_task,
                                int gate_mult, int gate_shift, double in_scale, double out_scale) {
    if (sdp_task < 1 || sdp_task >= S) return -2;
    if (!ork_ppu_fuse_enabled(c)) return -3;
    if (orki_silu_calibrate_idx(c)) return -1;
    static int16_t lut[1030]; orki_silu_build_curve(c, orki_silu_f, in_scale, out_scale, lut);   /* same curve as orki_act_lut_i8 */
    struct chain_silu_spec ss = { NULL, -1, sdp_task, 0x4000, 14, gate_mult, gate_shift, 0,
                                  ORK_SILU_IDXOFF, ORK_SILU_C4064, ORK_SILU_C4068, lut, 1030 };
    return orki_run_chain_i8_impl(c, S, tasks, &ss, -1);
}

int ork_mm_run_chain_i8_ffn(ork_npu *c, int S, const ork_mm_task_i8 *tasks,
                            const ork_chain_op *ops, double in_scale, double out_scale) {
    if (S < 1 || !ops) return -2;
    if (!ork_ppu_fuse_enabled(c)) return -3;
    if (orki_silu_calibrate_idx(c)) return -1;
    static int16_t lut[1030]; orki_silu_build_curve(c, orki_silu_f, in_scale, out_scale, lut);
    struct chain_silu_spec ss = { ops, -1, -1, 0x4000, 14, 0, 0, 0,
                                  ORK_SILU_IDXOFF, ORK_SILU_C4064, ORK_SILU_C4068, lut, 1030 };
    return orki_run_chain_i8_impl(c, S, tasks, &ss, -1);
}

int ork_mm_run_chain_i8_ffn_exp_biased(ork_npu *c, int S, const ork_mm_task_i8 *tasks,
                                       const ork_chain_op *ops, double in_scale, double out_scale, double max_bias) {
    if (S < 1 || !ops) return -2;
    if (!ork_ppu_fuse_enabled(c)) return -3;
    if (orki_silu_calibrate_idx(c)) return -1;
    /* build-once: the exp curve depends only on (in_scale,out_scale,max_bias); skip the host rebuild when
     * unchanged so the static lut CONTENTS are stable across calls (which is what makes orki_run_chain_i8_impl's
     * pointer-keyed LUT cache correct). Per-layer biases that DIFFER within a process force a rebuild each call
     * (correct, but defeats the pointer cache for that call) — prefer a single process-wide bias (e.g. the int8
     * score-ceiling) so this stays a one-time build. */
    static int16_t lut[1030]; static double c_is=-1, c_os=-1, c_bias=-1e300;
    if (in_scale != c_is || out_scale != c_os || max_bias != c_bias) {
        orki_silu_build_curve_biased(c, orki_exp_f, in_scale, out_scale, max_bias, lut); c_is=in_scale; c_os=out_scale; c_bias=max_bias;
        /* contents of `lut` changed IN PLACE (same address). orki_run_chain_i8_impl keys its device-LUT (re)load on the
         * ss->lut POINTER (chain_lut_p[cc]), so an in-place rebuild would leave the STALE curve resident on the NPU
         * SRAM. Invalidate the pointer cache on all cores so the next submit rebuilds the core Lrc + reuploads the
         * new curve. (Without this, a per-call-varying in_scale/bias serves the FIRST call's LUT — measured: same
         * in_scale gave 0.098 fresh vs 0.83 as a 2nd call.) */
        for (int i=0;i<ORK_MAXCORE;i++) c->chain_lut_p[i] = NULL; }
    struct chain_silu_spec ss = { ops, -1, -1, 0x4000, 14, 0, 0, 0,
                                  ORK_SILU_IDXOFF, ORK_SILU_C4064, ORK_SILU_C4068, lut, 1030 };
    return orki_run_chain_i8_impl(c, S, tasks, &ss, -1);
}

int ork_mm_run_chain_i8_ffn_exp(ork_npu *c, int S, const ork_mm_task_i8 *tasks,
                                const ork_chain_op *ops, double in_scale, double out_scale) {
    return ork_mm_run_chain_i8_ffn_exp_biased(c, S, tasks, ops, in_scale, out_scale, 0.0);
}

int orki_run_chain_i8_impl(ork_npu *c, int S, const ork_mm_task_i8 *tasks, const struct chain_silu_spec *ss, int force_core) {
    if (!c) return -1;
    if (S < 1 || S > 1024) return -2;
    if (!tasks) return -2;
    if (tasks[0].w) {  /* chained weights share one submit => one domain; swap in that domain's scratch */
        if (tasks[0].w->domain != c->dom_active || (tasks[0].w->domain!=0 && !c->dom_save)) orki_dom_activate(c, tasks[0].w->domain);
    }

    /* step-1 core-parameterize: the whole chain (LUT-load + program submit) runs on this one core, so
     * round-robin can place independent chains on different cores. Default 0 = core 0 (unchanged). */
    /* force_core>=0 (concurrent rr dispatch: the pool worker passes its own core) overrides the ctx's
     * chain_core; force_core<0 uses the (clamped) c->chain_core as before. */
    int chain_cc = (force_core>=0 && force_core<c->soc->cores) ? force_core
                 : ((c->chain_core>=0 && c->chain_core<c->soc->cores) ? c->chain_core : 0);
    /* A single matmul has nothing to chain — dispatch to the optimized run_i8 path (multi-core
     * N-split / full-K single-submit decode via the auto-tuner). The chain path is single-core and
     * allocs per-call scratch, so it must only be used to batch S>1 independent matmuls. */
    if (S == 1) return ork_mm_run_i8(c, tasks[0].w, tasks[0].M, tasks[0].A, tasks[0].C);

    int fd = c->fd, CBUF = c->soc->cbuf_elems;
    int KS_CHAIN = orki_int8_ks(c);   // K-slice size for the FFN chain down projection (default 1024)

    /* increment 2 (concurrent round-robin): this chain runs entirely on core `chain_cc` using its OWN scratch
     * set (chain_rc/tk/lrc/lsc[chain_cc]), so chains dispatched to different cores never share DRAM. Lazily
     * allocated per core in the current domain (single-domain assumption, as the prior shared LUT buffers had).
     * regcmd 1MB (~1160 programs) / task 256KB bound the chain size; P is bounds-checked at the build site. */
    struct buf *RC=&c->chain_rc[chain_cc], *TK=&c->chain_tk[chain_cc], *LRC=&c->chain_lrc[chain_cc], *LSC=&c->chain_lsc[chain_cc];
    if(!RC->cpu){ *RC=orki_bcreate(fd,1048576,0x403,c->dom_active); *TK=orki_bcreate(fd,262144,0x40b,c->dom_active);
        if(!RC->cpu||!TK->cpu){ return -2; } }

    // 1. Validate all tasks
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        if (!w) return -2;
        if (w->dtype != DT_I8) return -2;
        if (tasks[i].M <= 0) return -2;
        if (w->K % 32 || w->N % 32) return -2;
        // Each chain link is ONE full-K submit. Need a single-slice weight: either Sk==1 (Bb[0] holds
        // the whole K) or a Bf full-K buffer (K<=10752; built by pack/repack for the MoE experts).
        // K=2048 experts pack Sk=2 but carry Bf — use it so they can chain. N must be a single slice.
        if (w->Sn != 1) return -2;
        // FFN chain down (K-split): a matmul task reading a prior output (in0>=0) with K=Nff>4096 is chained
        // as w->Sk K-slice programs (Bb[ks], glu K-slice activation) writing partials, host-summed after.
        int ffn_ksplit = ss && ss->ops && ss->ops[i].in0 >= 0 &&
                         (ss->ops[i].kind == OP_MM32 || ss->ops[i].kind == OP_MM8) && w->K > 4096;
        if (ffn_ksplit) { if (w->K % KS_CHAIN != 0 || !w->Bb) return -2; continue; }
        if (w->Sk != 1 && !w->Bf) return -2;
        // The full-K Bf submit uses orki_synth_i8(sched=1), whose 0x1040 K-reduction schedule is only valid for
        // K%512==0 && K<=4096 (same envelope as orki_run()'s M>1 Bf path; 512/1024 are covered, 1536-4096 too).
        // For other K (e.g. 768 down_proj, or K>4096) a full-K single submit is WRONG — reject so the caller
        // falls back to per-task run_i8 (which K-splits correctly). -3 distinguishes this from bad-arg -2.
        if (w->K % 512 != 0 || w->K > 4096) return -3;
        // M>mcap is fine — the synth loop M-tiles it into multiple chained programs (Step B below).
        if (orki_check_overlap("ork_mm_run_chain_i8", (uintptr_t)tasks[i].A, (uintptr_t)tasks[i].A + (size_t)tasks[i].M * w->K, (uintptr_t)tasks[i].C, (uintptr_t)tasks[i].C + (size_t)tasks[i].M * w->N * 4)) return -1;
    }

    /* P1a — SPINE MIGRATION (submit consolidation). Route the plain (ss==NULL) M=1 single-slice resident-C
     * matmul chain onto the ONE NONBLOCK doorbell submit (ork_dyn_begin_mc) instead of this path's hand-rolled
     * blocking submit. NO legacy fallback — git is the recovery, not a runtime hedge: a doorbell miss returns
     * an error AFTER ork_dyn_end auto-dumps the stuck-descriptor core-dump, so the failure surfaces with
     * diagnostics (the forcing function toward the doorbell fix) instead of being masked. Non-routable shapes
     * (M>1, K-split, SDP/silu via ss) still take the chain path below until their gaps (G1..G5) are migrated. */
    if (!ss) {
        /* P3 (#3): route the FULL non-ss chain — ANY M, not just M==1 — onto the doorbell spine. The M>1 tasks
         * use the doorbell's scratch+copy-back (M-tiling for M>64); M==1 with a non-resident C also falls back
         * to scratch there, so no dma_find gate is needed. The pre-checks above already met ork_dyn_begin_mc's
         * envelope (int8, Sn==1, K%512==0, K<=4096), EXCEPT it needs a full-K Bf for the wide-M (M>64) path — so
         * the only shape it can't take is a M>64 task on a Sk==1 Bb-only weight (no Bf); leave that on the chain
         * below. NO legacy fallback for the routed shapes (a miss returns -1 after ork_dyn_end auto-dumps). */
        int routable = 1;
        for (int i = 0; i < S; i++) if (tasks[i].M > 64 && !tasks[i].w->Bf) { routable = 0; break; }
        if (routable) {
            ork_dyn_chain *h = ork_dyn_begin_mc(c, S, tasks, 1);   /* single-core doorbell spine (chain stays single-core) */
            if (!h) return -1;                                     /* rejected/alloc-fail — surface it, no fallback */
            int d = ork_dyn_end(h);                                /* auto-dumps on an incomplete drain */
            return (d == S - 1) ? 0 : -1;                          /* all landed = ok; miss = error (dumped) */
        }
    }

    // 2. State transition for int8 mode. DT_I8 <-> DT_I8_CHAIN is NOT a hardware mode change (see line 42),
    // so under ORK_MIXED_NOTHRASH we KEEP the warm state across the transition — mirroring the mcworker
    // keepwarm (line 3501). Without this, entering the chain from a plain int8 op re-warmed every call
    // (reps=2), which is the per-layer thrash that made HW-chaining the FFN a net loss. Only a NON-int8
    // predecessor (fp16/int4) needs the reset+re-warm.
    /* ork_npu_enter mutates SHARED mode state (last_dt, possible ACT_RESET) — under concurrent rr dispatch
     * (force_core>=0) N workers would race it and a reset mid-flight on a sibling core wedges. The rr wrapper
     * (ork_mm_run_chains_rr) enters the mode ONCE single-threaded before dispatch; workers skip it here. */
    if (force_core < 0) ork_npu_enter(c, 3 /* DT_I8_CHAIN */, XP_CHAIN_NT, ss ? OCK_FUSED : OCK_HW);  /* fused static graph (in-chain SDP/LUT) when ss!=NULL, else pure-matmul hw chain */

    // 3. Resolve buffers and cache coherency
    struct buf tmp_A[1024];
    struct buf tmp_C[1024];
    struct buf Lrc = {0}, Lsc = {0};   /* fused-SiLU LUT buffers (only used when ss != NULL); zero so cleanup is safe on early goto */
    int do_lut = ss && !getenv("ORK_GSILU_NOLUT");   /* bisection: ORK_GSILU_NOLUT drops the LUT-load submit */
    memset(tmp_A, 0, sizeof(tmp_A));
    memset(tmp_C, 0, sizeof(tmp_C));

    uint32_t act_dma[1024];
    uint32_t out_dma[1024];
    struct buf *cbufs[1024];
    memset(cbufs, 0, sizeof(cbufs));

    int ok = 0;
    /* STATIC-GRAPH ioctl reduction: dedup the TO_DEVICE bsync of DMA buffers SHARED across chained tasks.
     * A segment's independent matmuls (Q/K/V, gate/up) read the SAME input activation; if it lives in one
     * resident DMA buffer (dma_find hits), sync it ONCE for the whole segment instead of once per task.
     * (No-op until the caller quantizes the shared input into a DMA buffer — the ggml-ork segment path.) */
    struct buf *dsynced[1024]; int ndsynced = 0;
    #define ALREADY_SYNCED(B) ({ int _hit=0; for(int _j=0;_j<ndsynced;_j++) if(dsynced[_j]==(B)){_hit=1;break;} \
                                 if(!_hit && ndsynced<1024) dsynced[ndsynced++]=(B); _hit; })
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        int M = tasks[i].M;
        int K = w->K;
        int N = w->N;

        // Resolve input activations buffer
        struct buf *abuf = orki_dma_find(c, tasks[i].A);
        if (abuf) {
            if (!ALREADY_SYNCED(abuf)) orki_bsync(fd, abuf, RKNPU_MEM_SYNC_TO_DEVICE);   // shared input: sync once
            act_dma[i] = (uint32_t)(abuf->dma + ((const char*)tasks[i].A - (const char*)abuf->cpu));
        } else {
            tmp_A[i] = orki_bcreate(fd, (size_t)M * K, 0x403, c->dom_active);
            if (!tmp_A[i].cpu) { ok = -1; goto cleanup; }
            memcpy(tmp_A[i].cpu, tasks[i].A, (size_t)M * K);
            orki_bsync(fd, &tmp_A[i], RKNPU_MEM_SYNC_TO_DEVICE);
            act_dma[i] = (uint32_t)tmp_A[i].dma;
        }

        // Resolve output buffer
        struct buf *cbuf = orki_dma_find(c, tasks[i].C);
        if (cbuf) {
            if (!ALREADY_SYNCED(cbuf)) orki_bsync(fd, cbuf, RKNPU_MEM_SYNC_TO_DEVICE);   // shared output region: sync once
            out_dma[i] = (uint32_t)(cbuf->dma + ((const char*)tasks[i].C - (const char*)cbuf->cpu));
            cbufs[i] = cbuf;
        } else {
            // FFN chain down-K-split task: needs Sk partial buffers [Sk][M][N] int32 (host-summed after).
            int ffn_ksplit = ss && ss->ops && ss->ops[i].in0 >= 0 &&
                             (ss->ops[i].kind == OP_MM32 || ss->ops[i].kind == OP_MM8) && K > 4096;
            size_t osz = ffn_ksplit ? (size_t)(K / KS_CHAIN) * M * N * 4 : (size_t)M * N * 4;
            tmp_C[i] = orki_bcreate(fd, osz, 0x403, c->dom_active);
            if (!tmp_C[i].cpu) { ok = -1; goto cleanup; }
            orki_bsync(fd, &tmp_C[i], RKNPU_MEM_SYNC_TO_DEVICE);
            out_dma[i] = (uint32_t)tmp_C[i].dma;
        }
    }

    // 4. Synthesize REGCMD blocks and link the chain
    struct buf extra[2048];
    int extra_n = 0;
    for (int j = 0; j < S; j++) {
        if (tmp_A[j].cpu) extra[extra_n++] = tmp_A[j];
        if (tmp_C[j].cpu) extra[extra_n++] = tmp_C[j];
    }
    // Each task is one full-K matmul of M rows; a single submit handles <= mcap rows, so a task with
    // M>mcap expands into ceil(M/mcap) M-tile programs (offsetting into its A/C buffers). ALL programs
    // across ALL tasks are PC-chained into one submit. Count total programs P first (must fit buffers).
    int prog_off[1025];   // prog_off[i] = first program index of task i (S<=1024)
    int P = 0;
    for (int i = 0; i < S; i++) {
        prog_off[i] = P;
        int ffn_ksplit = ss && ss->ops && ss->ops[i].in0 >= 0 &&
                         (ss->ops[i].kind == OP_MM32 || ss->ops[i].kind == OP_MM8) && tasks[i].w->K > 4096;
        if (ffn_ksplit) { P += tasks[i].w->K / KS_CHAIN; continue; }   // down: one program per K-slice
        int mcap = orki_chain_fullk_mcap_i8(c, tasks[i].w->K);
        P += (tasks[i].M + mcap - 1) / mcap;
    }
    if (P > 1024) { ok = -2; goto cleanup; }   // too many M-tiles for the chain regcmd/task buffers

    // run_chain_i8 is SINGLE-CORE: it PC-chains all P programs into ONE submit (low latency, one ioctl
    // for S matmuls — e.g. decode QKV/gate-up sharing an input). Cross-core throughput is now served by
    // ork_mm_run_stream_i8 (async round-robin, ~3x); the old barrier fan-out here was superseded (~1.3x).
    // task op kind: ops[] path, else legacy (silu-SDP at sdp_task; everything else matmul). SDP-containing
    // chains assume single M-tile per task (P==S, program p == task p) so this program<->task map holds.
    #define CHAIN_KIND(ii) (ss ? (ss->ops ? ss->ops[ii].kind : ((ii)==ss->sdp_task ? OP_SILU : OP_MM32)) : OP_MM32)
    uint32_t rc[REGCMD_I8_N + 4];
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        int M = tasks[i].M, K = w->K, N = w->N, mcap = orki_chain_fullk_mcap_i8(c, K);
        // full-K single submit: Bf[0] (the K<=10752 full-K layout, e.g. Sk=2 experts) if present,
        // else Bb[0] (which holds the whole K only when Sk==1). Both are the synth_i8 tile layout.
        uint32_t bdma = w->Bf ? (uint32_t)w->Bf[0].dma : (uint32_t)w->Bb[0].dma;
        int p = prog_off[i];
        // FFN chain down-K-split: emit w->Sk K-slice matmul programs (each Bb[ks], reading the glu K-slice
        // glu[:,ks*KS:(ks+1)*KS] -- a contiguous EWCUBE sub-range at ks*KS*M bytes -- into partial ks). All
        // chained; the Sk int32 partials are host-summed after the submit into tasks[i].C.
        if (ss && ss->ops && ss->ops[i].in0 >= 0 && (ss->ops[i].kind == OP_MM32 || ss->ops[i].kind == OP_MM8) && K > 4096) {
            int Sk = K / KS_CHAIN;
            uint32_t glu_dma = out_dma[ss->ops[i].in0];
            for (int ks = 0; ks < Sk; ks++, p++) {
                memset(rc, 0, sizeof(rc));
                orki_bsync(fd, &w->Bb[ks], RKNPU_MEM_SYNC_TO_DEVICE);
                orki_synth_i8(rc, M, KS_CHAIN, N, glu_dma + (uint32_t)((size_t)ks * KS_CHAIN * M),
                         (uint32_t)w->Bb[ks].dma, out_dma[i] + (uint32_t)((size_t)ks * M * N * 4), 1, CBUF, 0);
                if (orki_validate_regcmd("run_chain_i8", c, rc, REGCMD_I8_N, w, extra, extra_n)) { ok = -1; goto cleanup; }
                if (p < P - 1) {   // chain to the next K-slice (or terminate on the last)
                    uint64_t next_dma = RC->dma + (size_t)(p + 1) * REGCMD_I8_N * 4;
                    rc[216] = 0x0010 | ((next_dma & 0xffff) << 16);
                    rc[217] = (0x0101 << 16) | ((next_dma >> 16) & 0xffff);
                    rc[218] = 0x0014 | (0x0037u << 16);   // next = matmul K-slice (108 regs)
                    rc[219] = (0x0101 << 16);
                }
                memcpy((char*)RC->cpu + (size_t)p * REGCMD_I8_N * 4, rc, REGCMD_I8_N * 4);
            }
            continue;
        }
        for (int m0 = 0; m0 < M; m0 += mcap, p++) {
            int mc = (M - m0 < mcap) ? (M - m0) : mcap;
            memset(rc, 0, sizeof(rc));
            // Let orki_synth_i8(sched=1) set the 0x1040 K-reduction schedule from mc (= ceil(mc/64) group).
            // Do NOT hardcode it (the old 0xb1 was an M=1 value; for mc>16 it computes rows past the
            // first 64-group against the wrong K-partition — same class as the full-K prefill bug).
            int kind = CHAIN_KIND(i);
            if (kind == OP_SILU || kind == OP_EWMUL) {
                // STANDALONE SDP op (REGCMD_SILU_STD / REGCMD_MUL, enable 0x18/regcfg 69) reading prior tasks'
                // output buffers via ALIASING -- the vendor matmul->SDP pattern (data via shared addresses).
                int in0 = ss->ops ? ss->ops[i].in0 : (i - 1);          // legacy silu reads task i-1
                const uint32_t *tmpl = (kind == OP_SILU) ? REGCMD_SILU_STD : REGCMD_MUL;
                int tn = (kind == OP_SILU) ? REGCMD_SILU_STD_N : REGCMD_MUL_N;
                memcpy(rc, tmpl, (size_t)tn * 4);
                orki_set_mul_geom(rc, tn, mc, N);
                orki_setrn(rc, tn,RK_DPU_DST_BASE_ADDR, out_dma[i]);              // output
                orki_setrn(rc, tn,RK_SDP_5018, out_dma[in0]);            // operand a = prior output, ALIASED
                if (kind == OP_SILU) {
                    orki_setrn(rc, tn,RK_SDP_5040, 0); orki_setrn(rc, tn,RK_SDP_5038, 0);   // single-input
                    orki_setrn(rc, tn,RK_DPU_OUT_CVT_SCALE, (uint32_t)ss->r_mult); orki_setrn(rc, tn,RK_DPU_OUT_CVT_SHIFT, (uint32_t)ss->r_shift);
                    orki_setrn(rc, tn,RK_DPU_OUT_CVT_OFFSET, ss->out_bias);
                    orki_setrn(rc, tn,RK_DPU_R4110, ss->idx_off); orki_setrn(rc, tn,RK_DPU_BN_ALU_CFG, ss->cfg4064); orki_setrn(rc, tn,RK_DPU_BN_MUL_CFG, ss->cfg4068);
                } else {   // OP_EWMUL: out = a (*) b ; b = second prior output (silu*up for glu)
                    int in1 = ss->ops[i].in1;
                    orki_setrn(rc, tn,RK_SDP_5038, out_dma[in1]);        // operand b, ALIASED
                    orki_setrn(rc, tn,RK_DPU_OUT_CVT_SCALE, (uint32_t)ss->ops[i].mult); orki_setrn(rc, tn,RK_DPU_OUT_CVT_SHIFT, (uint32_t)ss->ops[i].shift);
                    orki_setrn(rc, tn,RK_DPU_OUT_CVT_OFFSET, 0); orki_setrn(rc, tn,RK_DPU_BS_ALU_CFG, 0); orki_setrn(rc, tn,RK_DPU_EW_CVT_OFFSET, 0);
                }
            } else {
                // activation source: ops[i].in0 >= 0 -> a PRIOR task's output (aliased, e.g. down reads glu);
                // else tasks[i].A (the normal external activation). SDP-output->matmul-input is the vendor pattern.
                uint32_t a_dma = (ss && ss->ops && ss->ops[i].in0 >= 0)
                               ? out_dma[ss->ops[i].in0]
                               : act_dma[i] + (uint32_t)((size_t)m0 * K);
                orki_synth_i8(rc, mc, K, N, a_dma,
                         bdma, out_dma[i] + (uint32_t)((size_t)m0 * N * 4), 1, CBUF, 0);
                if (ss && !ss->ops && i == ss->task && !getenv("ORK_GSILU_NOSILU")) orki_set_i8_silu(rc, N, 0, ss->r_mult, ss->r_shift, ss->out_bias, ss->idx_off, ss->cfg4068);
                else if (kind == OP_MM8) orki_set_i8_out8(rc, N, 0, ss->ops ? ss->ops[i].mult : ss->gate_mult, ss->ops ? ss->ops[i].shift : ss->gate_shift);  // int8 out (feeds an SDP task)
                else if (ss && !ss->ops && ss->sdp_task >= 1 && i == ss->sdp_task - 1) orki_set_i8_out8(rc, N, 0, ss->gate_mult, ss->gate_shift);   // legacy gate->silu
                if (orki_validate_regcmd("run_chain_i8", c, rc, REGCMD_I8_N, w, extra, extra_n)) { ok = -1; goto cleanup; }
            }
            if (p < P - 1) {   // PC-chain: this program jumps to the next; the last keeps the template's raise-interrupt tail
                uint64_t next_dma = RC->dma + (size_t)(p + 1) * REGCMD_I8_N * 4;
                int nk = CHAIN_KIND(i + 1);
                int next_is_sdp = (nk == OP_SILU || nk == OP_EWMUL);   // next program's regcfg: SDP=69 -> amt 36; matmul=108 -> amt 55
                uint32_t namt = next_is_sdp ? (uint32_t)((69 + 3) / 2) : 0x0037;
                // THIS program's PC next-descriptor lives at word 2*regcfg (matmul 108 -> 216; SDP 69 -> 138).
                // The PC reads regcfg reg-pairs then the descriptor; a middle SDP task needs it at 138, not 216.
                int cur_sdp = (kind == OP_SILU || kind == OP_EWMUL);
                int dw = cur_sdp ? (69 * 2) : (108 * 2);
                rc[dw]   = 0x0010 | ((next_dma & 0xffff) << 16);
                rc[dw+1] = (0x0101 << 16) | ((next_dma >> 16) & 0xffff);
                rc[dw+2] = 0x0014 | (namt << 16);
                rc[dw+3] = (0x0101 << 16) | (0);
            }
            memcpy((char*)RC->cpu + (size_t)p * REGCMD_I8_N * 4, rc, REGCMD_I8_N * 4);
        }
    }
    orki_bsync(fd, RC, RKNPU_MEM_SYNC_TO_DEVICE);

    // FUSED-SiLU LUT-load: stream the gate task's silu LUT into SDP SRAM ONCE before the chain (enable 0x18,
    // ping-pong OFF). It persists into the chain submit; the gate task's set_i8_silu output stage reads it.
    if (do_lut) {
        /* LUT-build-once cache (PER-CORE): chain_lrc/lsc[cc] hold this core's patched LUT DRAM; re-patch only when
         * the LUT identity changes for this core. With ORK_CHAIN_LUT_STICKY also skip the load submit when core cc's
         * physically-per-core SDP SRAM still holds this LUT — saves the ~148us/call load on tight same-LUT loops
         * (decode / round-robin). Per-core buffers make this concurrency-safe: sibling cores never share this DRAM. */
        static int sticky=-1; if(sticky<0) sticky=getenv("ORK_CHAIN_LUT_STICKY")?1:0;
        int cc = chain_cc;   /* this chain's target core (hoisted above) */
        if (!LRC->cpu) {   /* one-time per-core alloc */
            *LRC = orki_bcreate(fd, (size_t)REGCMD_SILU_LUT_N * 4, 0x403, c->dom_active);
            *LSC = orki_bcreate(fd, 4096, 0x403, c->dom_active);
            if (!LRC->cpu || !LSC->cpu) { ok = -2; goto cleanup; }
        }
        if (c->chain_lut_p[cc] != ss->lut) {   /* (re)build THIS core's Lrc only when its LUT identity changes */
            memcpy(LRC->cpu, REGCMD_SILU_LUT, REGCMD_SILU_LUT_N * 4);
            orki_setrn((uint32_t*)LRC->cpu, REGCMD_SILU_LUT_N,RK_DPU_DST_BASE_ADDR, (uint32_t)LSC->dma);
            { uint32_t *lr=(uint32_t*)LRC->cpu; int j=0;
              for (int k=0; k+1<REGCMD_SILU_LUT_N; k+=2) if ((lr[k]&0xffff)==0x4104) {
                  int32_t v = (j<ss->nlut) ? (int32_t)ss->lut[j] : 0; j++;
                  lr[k]=0x4104|((uint32_t)(v&0xffff)<<16); lr[k+1]=(0x1001u<<16)|(((uint32_t)v>>16)&0xffff); } }
            orki_bsync(fd, LRC, RKNPU_MEM_SYNC_TO_DEVICE);
            c->chain_lut_p[cc] = ss->lut; c->chain_lut_devloaded[cc] = 0;   /* new LUT patched -> this core's SRAM copy is stale */
        }
        /* load the LUT into core cc's (physically per-core) SDP SRAM; skip only if sticky AND already resident on cc */
        if (!(sticky && c->chain_lut_devloaded[cc])) {
            struct rknpu_task *lt=TK->cpu; memset(lt,0,sizeof *lt);
            lt->enable_mask=0x18; lt->int_mask=0x300; lt->int_clear=0x1ffff; lt->regcfg_amount=1097; lt->regcmd_addr=LRC->dma;
            orki_bsync(fd,TK,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
            c->chain_task_built[cc]=0;   /* the LUT-load just memset this core's task DRAM -> chain array no longer resides there */
            struct rknpu_submit ls; memset(&ls,0,sizeof ls); ls.flags=0x1; ls.task_number=1; ls.task_obj_addr=TK->obj;
            ls.core_mask=1u<<cc; ls.fence_fd=-1; ls.timeout=orki_ew_timeout_ms();
            /* ALL THREE subcore_task[] must be populated even for a single core (npu.c:3365): on core_mask=1<<cc
             * the kernel commits subcore_task[cc]; leaving it zero NULL-derefs rknpu_job_subcore_commit and wedges.
             * This LUT-load previously set only [0] -> fine on core 0, but the FIRST submit on core 1/2 wedged. */
            ls.subcore_task[0]=ls.subcore_task[1]=ls.subcore_task[2]=(struct rknpu_subcore_task){0,1};
            if (orki_rknpu_submit_ioctl(fd,&ls,tasks[0].w->domain)) { ok=-1; goto cleanup; }
            c->chain_lut_devloaded[cc] = 1;
        }
    }

    // One rknpu_task per program (P total, PC-chained), one single-core submit.
    int submit_ok = 0;
    struct rknpu_task *t = TK->cpu;
    /* task-config cache (per-core): the P-program array (enable/regcfg/regcmd_addr — all shape-stable, regcmd_addr
     * fixed to chain_rc[cc].dma) is identical every call for a fixed chain shape on this core. Skip the rebuild+bsync
     * when sticky + the LUT is resident on cc AND the shape (chain_task_P[cc]) matches AND chain_tk[cc] still holds
     * this core's array (chain_task_built[cc]; a LUT-load on cc memsets chain_tk[cc] and clears it). Per-core buffers
     * mean NO cross-core clobber — concurrent rr chains each own their chain_tk[core]. */
    { static int sticky=-1; if(sticky<0) sticky=getenv("ORK_CHAIN_LUT_STICKY")?1:0;
      int cached = sticky && c->chain_lut_devloaded[chain_cc] && c->chain_task_built[chain_cc] && c->chain_task_P[chain_cc]==P;
      if(!cached){
        memset(t, 0, (size_t)P * sizeof(struct rknpu_task));
        for (int p = 0; p < P; p++) {   // default: matmul task
            t[p].enable_mask = 0xd; t[p].int_mask = 0x300; t[p].int_clear = 0x1ffff;
            t[p].regcfg_amount = 108;
            t[p].regcmd_addr = RC->dma + (size_t)p * REGCMD_I8_N * 4;
        }
        for (int i = 0; i < S; i++) {   // SDP tasks (silu/ewmul): enable 0x18, regcfg 69 (single program at prog_off[i])
            int k = CHAIN_KIND(i);
            if (k == OP_SILU || k == OP_EWMUL) { t[prog_off[i]].enable_mask = 0x18; t[prog_off[i]].regcfg_amount = 69; }
        }
        orki_bsync(fd, TK, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
        c->chain_task_P[chain_cc] = P; c->chain_task_built[chain_cc] = 1;   /* this core's chain_tk now holds the P-array */
      } }
    struct rknpu_submit sub; memset(&sub, 0, sizeof(sub));
    // ping-pong OFF (0x1) for any silu chain (SDP/LUT task present) so a bank swap doesn't race the LUT SRAM
    // commit (AGENTS.md); plain matmul chains keep ork_ppflags() (register-config-only, ping-pong safe).
    /* P3 #5: the fused (ss) chain rides the NONBLOCK doorbell spine instead of a blocking submit. The chain is
     * PC-sequential, so the FINAL op landing => the whole chain has drained; we seed that op's output with the
     * doorbell sentinel before each submit and spin-poll it after (last-word gate, then a full-surface verify
     * for any residual write-order lag). Gated to an int32 final output (the FFN's down matmul): ORK_DYN_SENT
     * is a safe sentinel there and a matmul's last-col-last writeback makes the gate sound. An int8-output
     * final op has no safe sentinel, so it keeps the blocking submit; SDP middle ops (silu/ewmul) ride the
     * NONBLOCK chain regardless. */
    int fi = S - 1, kf = CHAIN_KIND(fi);
    /* mirror the completion's ffn_ksplit exactly: K-split partials live in scratch (tmp_C), never a resident cbuf */
    int ksf = ss && ss->ops && ss->ops[fi].in0 >= 0 && (kf == OP_MM32 || kf == OP_MM8) && tasks[fi].w->K > 4096 && !cbufs[fi];
    int fdb_on = ss && ((kf == OP_MM32) || ksf);            /* NONBLOCK only when the final output is int32 (safe sentinel) */
    volatile int32_t *fdb = NULL; size_t fno = 0; struct buf *fbuf = NULL;
    if (fdb_on) {
        fdb = (volatile int32_t *)(cbufs[fi] ? (void *)tasks[fi].C : tmp_C[fi].cpu);
        fno = ksf ? (size_t)(tasks[fi].w->K / KS_CHAIN) * tasks[fi].M * tasks[fi].w->N
                  : (size_t)tasks[fi].M * tasks[fi].w->N;
        fbuf = cbufs[fi] ? cbufs[fi] : &tmp_C[fi];
    }

    sub.flags = ss ? 0x1u : ork_ppflags(); sub.task_number = P; sub.task_obj_addr = TK->obj;
    sub.core_mask = 1u << chain_cc; sub.fence_fd = -1;   /* step-1: chain runs on its parameterized core */
    sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)P};
    /* RE PROBE (ORK_STEER_HALT_AT=<p>): does the PC sequencer read each program's chain descriptor from DRAM
     * at EXECUTION time (steerable mid-flight) or pre-cache the whole chain at submit? Submit NONBLOCK, then
     * overwrite program p's next-amount word (0x0014, amount 0 = documented clean halt) in the LIVE regcmd
     * DRAM. If the chain halts at p (outputs beyond p stay unwritten) -> read-from-DRAM -> dynamic steering /
     * our-own-submit-API is reachable; if it runs to the end -> pre-cached. Matmul chains only (dw=216). */
    int steer_at; { const char*e=getenv("ORK_STEER_HALT_AT"); steer_at = e?atoi(e):-1; }   /* per-call: probe warms unset, then sets it */
    int do_steer = (steer_at >= 0 && steer_at < P-1 && !ss);
    if (do_steer || fdb_on) sub.flags |= 0x2u;              /* NONBLOCK: do_steer (RE probe) or the fused-chain doorbell */
    int reps = do_steer ? 1 : (c->warmed ? 1 : 2);
    for (int rep = 0; rep < reps; rep++) {
        int last = (rep == reps - 1);
        sub.timeout = orki_mm_timeout_ms();
        if (fdb_on) {   /* seed the final-output sentinel and clean it to DRAM before the NONBLOCK submit */
            for (size_t e = 0; e < fno; e++) fdb[e] = 0x7fffffff;   /* == ORK_DYN_SENT (defined below) */
            orki_bsync(fd, fbuf, RKNPU_MEM_SYNC_TO_DEVICE);
        }
        if (orki_rknpu_submit_ioctl(fd, &sub, tasks[0].w->domain)) { if (last) { perror("SUBMIT chained"); submit_ok = -1; } continue; }
        submit_ok = 0;
        if (fdb_on) {   /* doorbell drain: last-word gate, then a full-surface verify (bounded) */
            double pt = ork_now_us(), cap = (double)orki_mm_timeout_ms() * 1000.0;
            for (;;) { __asm__ volatile("dc civac,%0"::"r"(&fdb[fno-1]):"memory"); if (fdb[fno-1] != 0x7fffffff) break; if (ork_now_us()-pt > cap) break; }
            for (;;) { int dn = 1; for (size_t e = 0; e < fno; e++) { __asm__ volatile("dc civac,%0"::"r"(&fdb[e]):"memory"); if (fdb[e] == 0x7fffffff) { dn = 0; break; } } if (dn || ork_now_us()-pt > cap) break; }
        }
        if (do_steer) {   /* mid-flight: halt program steer_at by zeroing its next-amount (0x0014) in DRAM */
            uint32_t *rcp = (uint32_t*)((char*)RC->cpu + (size_t)steer_at * REGCMD_I8_N * 4);
            rcp[218] = 0x0014;   /* 0x0014 | (amount 0) => sequencer stops after this program */
            __asm__ volatile("dc cvac,%0"::"r"(&rcp[218]):"memory"); __asm__ volatile("dsb ish":::"memory");
            usleep((unsigned)(P*60u + 8000u));   /* drain the (halted-or-full) chain before the FROM bsync */
            if (getenv("ORK_VERBOSE")) fprintf(stderr, "[STEER] NONBLOCK submit + halt-inject at prog %d/%d\n", steer_at, P);
        }
    }
    c->warmed = 1;

    // 7. Sync memory back and copy results
    for (int i = 0; i < S; i++) {
        if (cbufs[i]) {
            orki_bsync(fd, cbufs[i], RKNPU_MEM_SYNC_FROM_DEVICE);
        } else {
            orki_bsync(fd, &tmp_C[i], RKNPU_MEM_SYNC_FROM_DEVICE);
            if (submit_ok == 0) {
                int ffn_ksplit = ss && ss->ops && ss->ops[i].in0 >= 0 &&
                                 (ss->ops[i].kind == OP_MM32 || ss->ops[i].kind == OP_MM8) && tasks[i].w->K > 4096;
                size_t mn = (size_t)tasks[i].M * tasks[i].w->N;
                if (ffn_ksplit) {   // host-sum the Sk int32 K-slice partials -> tasks[i].C
                    int Sk = tasks[i].w->K / KS_CHAIN;
                    const int32_t *pp = (const int32_t *)tmp_C[i].cpu; int32_t *dst = tasks[i].C;
                    for (size_t e = 0; e < mn; e++) { int32_t acc = 0; for (int ks = 0; ks < Sk; ks++) acc += pp[(size_t)ks*mn + e]; dst[e] = acc; }
                } else {
                    memcpy(tasks[i].C, tmp_C[i].cpu, mn * 4);
                }
            }
        }
    }
    ok = submit_ok;

cleanup:
    for (int i = 0; i < S; i++) {
        orki_bdestroy(fd, &tmp_A[i]);
        orki_bdestroy(fd, &tmp_C[i]);
    }
    orki_bdestroy(fd, &Lrc); orki_bdestroy(fd, &Lsc);   /* fused-SiLU LUT buffers (no-op when ss==NULL: {0}) */
    return ok;
}

ork_async *ork_mm_run_chain_i8_async (ork_npu *c, int S, const ork_mm_task_i8 *tasks){
    if (!c || S < 1 || !tasks) return NULL;
    return ork_async_launch((struct ork_async){ .kind=OAK_CHAIN_I8, .c=c, .S=S, .tasks=tasks }); }
