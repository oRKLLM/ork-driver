/* npu/i8/probe_prof.c — int8 submit-floor and overlap PROFILING probes, plus the batch / b-scale acceptance probes.
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
int ork_npu_doorbell_prof(ork_npu *c,int M,int K,int N,int iters,double *block_us,double *nb_us,int *ok_block,int *ok_nb){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    memset(W.cpu,1,(size_t)K*N);                                  /* int8 weight all-1 (layout-agnostic) */
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)M*N*4,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}   /* int32 out */
    int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=1; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);  /* act all-1 -> out=K */
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I8_N]; orki_i8_synth(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
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
    uint32_t rc[REGCMD_I8_N]; orki_i8_synth(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
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
    if(M<1||M>ORK_SDP_MAXM||N<16||N>8192||(N&15)) return -2;
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
    uint32_t rc[REGCMD_I8_N]; orki_i8_synth(rc,1,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
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
