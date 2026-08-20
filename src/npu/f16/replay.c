/* npu/f16/replay.c — vendor RE replays (forward softmax, atom-8 reshape, standalone SiLU).
 *
 * Part of the f16 datapath; shared declarations in npu/f16/f16.h. Split out of npu/f16.c for the
 * same reason i8 is a folder: one datapath, sized for reading. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include "ork_regs.h"
#include "regcmd_array_4x32x16.h"
#include "regcmd_i8.h"
#include "regcmd_silu.h"
#include "regcmd_ewmul.h"
#include "regcmd_softmax_f16.h"
#include "regcmd_softmax_wt.h"
#include "regcmd_reshape.h"
#include "npu/internal.h"
#include <fcntl.h>
#include "npu/core.h"
#include "npu/f16/f16.h"

static void *ork_rsh_patcher(void *p){ struct ork_rsh_patch *a=p; usleep(a->delay_us);
    if(a->rcmode){ a->rcword[0]=a->rcval0; a->rcword[1]=a->rcval1; orki_bsync(a->c->fd,&a->c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE); }
    else { struct rknpu_task *tk=(struct rknpu_task*)a->c->task.cpu; tk[a->idx]=a->marker; orki_bsync(a->c->fd,&a->c->task,RKNPU_MEM_SYNC_TO_DEVICE); }
    __asm__ __volatile__("dsb sy":::"memory"); return NULL; }

int ork_npu_replay_reshape_f16(ork_npu *c,uint16_t *gemm_raw,int gemm_words,uint16_t *reshape_raw,int reshape_words,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    const uint32_t IB=0xfffed000u; const size_t ISZ=0x13000;   /* vendor image span */
    /* task0..21: {imgoff, enable, regcfg_amount} — the FULL first forward pass from the capture */
    static const struct { uint32_t off; uint32_t en; uint32_t amt; } TK[22]={
        {0xd8c0,0x18,69},{0xdb40,0xd,108},{0xdec0,0xd,108},{0xe240,0xd,108},{0xe5c0,0xd,108},
        {0xe940,0xd,13},{0xea00,0xd,12},{0xea80,0xd,12},{0xeb00,0xd,12},{0xeb80,0xd,12},{0xec00,0xd,12},
        {0xec80,0x18,69},{0xef00,0x18,69},{0xf180,0x18,69},{0xf400,0x18,25},{0xf500,0x18,9},{0xf580,0x18,9},
        {0xf600,0x18,9},{0xf680,0x18,9},{0xf700,0x18,9},{0xf780,0x18,9},{0xf800,0x18,69}};
    int T0=0; { const char*e=getenv("ORK_RESHAPE_T0"); if(e){int v=atoi(e); if(v>=0&&v<22)T0=v;} } /* start task (skip SDP convert) */
    int NT=22-T0; { const char*e=getenv("ORK_RESHAPE_NT"); if(e){int v=atoi(e); if(v>=1&&v<=22-T0)NT=v;} }
    #define TK(j) TK[(T0)+(j)]
    const char *path=getenv("ORK_RESHAPE_IMG"); if(!path)path="gemm_mul_image.bin";
    FILE *f=fopen(path,"rb"); if(!f) return -2;
    struct buf BIG=orki_bcreate(fd,ISZ,0x403,-1); if(!BIG.cpu){fclose(f);return -2;}
    memset(BIG.cpu,0,ISZ);
    size_t rd=fread(BIG.cpu,1,ISZ,f); fclose(f); if(rd<ISZ){ orki_bdestroy(fd,&BIG); return -2; }
    /* inject DISTINCT input (@0xfffef000, imgoff 0x2000) so gemm_out is non-degenerate -> derivable perm */
    if(getenv("ORK_RESHAPE_INJECT")){ uint16_t*x=(uint16_t*)((char*)BIG.cpu+0x2000);      /* task0 input @0xfffef000 */
        for(int i=0;i<512;i++){ float v=(float)((i%37)-8)*0.25f; __fp16 h=(__fp16)v; memcpy(&x[i],&h,2); } }
    int ginj=getenv("ORK_RESHAPE_GINJ")!=0;
    if(ginj){ uint16_t*g=(uint16_t*)((char*)BIG.cpu+0x3000);          /* GEMM output @0xffff0000: DISTINCT so reshape perm is derivable (start at task2, T0=2) */
        for(int i=0;i<512;i++){ __fp16 h=(__fp16)(float)(i+1); memcpy(&g[i],&h,2); }
        memset((char*)BIG.cpu+0x3680,0,0x400); }                      /* zero ONLY reshape-out so fresh writes show */
    ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_NONE);   /* warm/reset BEFORE we write c->regcmd (enter may use it) */
    uint32_t delta=(uint32_t)BIG.dma - IB;   /* DATA addresses -> BIG */
    /* #2 fix: put regcmds in c->regcmd (the path that EXECUTES), DATA stays in BIG. Compute each task's
     * c->regcmd word offset; copy its regcmd out of the image; rebase DATA addrs -> BIG, and point the chain
     * 0x0010 -> the NEXT task's c->regcmd offset (NOT BIG). (ORK_RESHAPE_INIMG = old in-image path for A/B.) */
    int inimg=getenv("ORK_RESHAPE_INIMG")!=0;
    /* ★ TERMINAL COMPLETABILITY: a 12-reg reshape DELTA cannot be the chain's last task (only 108-reg or the
     * 13-reg first-delta task5 raise DONE as terminus — task5 has an extra 0x1040=0x201b write the 12-reg
     * deltas lack). In the vendor graph the deltas are never last. So promote the terminal 12-reg delta to
     * task5's 13-reg form, patched to the terminal group's own in/out addresses. (ORK_RESHAPE_NOTERM = off.) */
    int termfix = !getenv("ORK_RESHAPE_NOTERM") && !inimg;
    uint32_t *cr=(uint32_t*)c->regcmd.cpu; uint32_t cro[22]; uint32_t eamt[22]; int prom[22];
    { uint32_t cur=0; for(int j=0;j<NT;j++){ prom[j]= termfix && (int)TK(j).amt==12; /* every 12-reg delta -> 13-reg task5 form */
        eamt[j]=prom[j]?13u:(uint32_t)TK(j).amt; cro[j]=cur; cur+=eamt[j]*2+16; } }
    for(int j=0;j<NT;j++){
        int last=(j==NT-1);
        uint32_t srcoff = prom[j] ? TK[5].off : TK(j).off;   /* TK[5] = the completable 13-reg delta */
        int nw=(int)eamt[j]*2+16;   /* regcfg entries*2 + trailer */
        uint32_t pin=0,pout=0;   /* this delta's OWN captured in/out (patch task5-form to it) */
        if(prom[j]){ uint32_t*orc=(uint32_t*)((char*)BIG.cpu+TK(j).off); int onw=(int)TK(j).amt*2+16;
            for(int k=0;k+1<onw;k+=2){ unsigned a=orc[k]&0xffff,d=orc[k+1]>>16; uint32_t v=((orc[k+1]&0xffff)<<16)|(orc[k]>>16);
                if(a==0x1070&&d==0x0201)pin=v; if(a==0x4020&&d==0x1001)pout=v; } }
        uint32_t *rc = inimg ? (uint32_t*)((char*)BIG.cpu+TK(j).off) : (cr+cro[j]);
        if(!inimg) memcpy(rc,(char*)BIG.cpu+srcoff,(size_t)nw*4);
        for(int k=0;k+1<nw;k+=2){ unsigned a=rc[k]&0xffff,d=rc[k+1]>>16; uint32_t v=((rc[k+1]&0xffff)<<16)|(rc[k]>>16);
            if(d==0x0101 && a==0x0010){ /* chain: next-regcmd addr */
                uint32_t nx = (inimg? (uint32_t)BIG.dma+TK(j+1<NT?j+1:j).off : (uint32_t)c->regcmd.dma+cro[j+1<NT?j+1:j]*4);
                if(!last){ rc[k]=0x0010|((nx&0xffff)<<16); rc[k+1]=(0x0101u<<16)|((nx>>16)&0xffff); }
                else if(getenv("ORK_RESHAPE_SELFLOOP")){ uint32_t sp=(uint32_t)c->regcmd.dma+cro[j]*4; /* self-loop: 0x0010 -> own regcmd (free-run probe) */
                    rc[k]=0x0010|((sp&0xffff)<<16); rc[k+1]=(0x0101u<<16)|((sp>>16)&0xffff); }
                else { rc[k]=0; rc[k+1]=0; } }                             /* terminate last */
            else if(d==0x0101 && a==0x0014){ if(last&&getenv("ORK_RESHAPE_SELFLOOP")){ uint32_t na=(eamt[j]+3)/2; rc[k]=0x0014|((na&0xffff)<<16); rc[k+1]=(0x0101u<<16)|((na>>16)&0xffff); } /* self-loop: next-amount = own */
                else if(last){ rc[k]=0; rc[k+1]=0; } /* null on last */
                else { uint32_t na=(eamt[j+1]+3)/2; rc[k]=0x0014|((na&0xffff)<<16); rc[k+1]=(0x0101u<<16)|((na>>16)&0xffff); } } /* next-amount = (eff_amt[j+1]+3)/2 (handles promotion) */
            else if(prom[j] && a==0x1070 && d==0x0201){ uint32_t nv=pin+delta;  rc[k]=a|((nv&0xffff)<<16); rc[k+1]=(d<<16)|((nv>>16)&0xffff); } /* patch task5-form to THIS group */
            else if(prom[j] && a==0x4020 && d==0x1001){ uint32_t nv=pout+delta; rc[k]=a|((nv&0xffff)<<16); rc[k+1]=(d<<16)|((nv>>16)&0xffff); }
            else if(v>=IB && v<IB+ISZ){ uint32_t nv=v+delta; rc[k]=a|((nv&0xffff)<<16); rc[k+1]=(d<<16)|((nv>>16)&0xffff); } } /* DATA -> BIG */
    }
    /* ZERO gemm-out (0x3000) + reshape-out (0x3a00) so FRESH computation is distinguishable from baked image data. */
    if(getenv("ORK_RESHAPE_ZERO")){ memset((char*)BIG.cpu+0x3000,0,0x400); memset((char*)BIG.cpu+0x3680,0,0x400); }
    orki_bsync(fd,&BIG,RKNPU_MEM_SYNC_TO_DEVICE);
    if(!inimg) orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t tn=(uint32_t)NT; { const char*e=getenv("ORK_RESHAPE_TN"); if(e){unsigned v=(unsigned)strtoul(e,0,0); if(v>=1)tn=v;} } /* TREADMILL = #descriptors (task_number must match; a big RING beats the prefetcher) */
    if(tn>13000)tn=13000;                                          /* c->task=512KB / ~40B per rknpu_task */
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof(*tk)*tn);
    for(int j=0;j<NT;j++){ tk[j].enable_mask=TK(j).en; tk[j].int_mask=0x300; tk[j].int_clear=0x1ffff;
        tk[j].regcfg_amount=eamt[j]; tk[j].regcmd_addr = inimg ? (uint32_t)BIG.dma+TK(j).off : (uint32_t)c->regcmd.dma+cro[j]*4; }
    for(uint32_t j=(uint32_t)NT;j<tn;j++) tk[j]=tk[NT-1];          /* RING: replicate the (completable) last task to fill the treadmill */
    /* Wall-#2 live-patch test: whole ring = cheap dummy (does NOT write 0x3000); MARKER = the GEMM (DOES write
     * 0x3000). A bg thread hot-patches descriptor[tn/2] (far ahead) to the marker mid-submit. If 0x3000 ends up
     * nonzero, the NPU re-read the patched descriptor from DRAM when it arrived -> live far-ahead patch HONORED. */
    pthread_t pth; struct ork_rsh_patch parg; int patching=0;
    if(getenv("ORK_RESHAPE_PATCH") && tn>(uint32_t)NT){
        parg.c=c; parg.marker=tk[0];                              /* GEMM marker (writes 0x3000) */
        for(uint32_t j=0;j<tn;j++) tk[j]=tk[NT-1];                /* ring = cheap reshape-delta dummy */
        parg.idx=tn/2; { const char*e=getenv("ORK_RESHAPE_PATCH_US"); parg.delay_us=e?(uint32_t)strtoul(e,0,0):50; }
        parg.rcmode=0;
        if(getenv("ORK_RESHAPE_PATCHRC")){ /* alt: patch the shared DUMMY regcmd's 0x4020 output -> 0x3000, mid-orki_run (tests regcmd live re-read) */
            uint32_t *drc=cr+cro[NT-1]; int dnw=(int)eamt[NT-1]*2+16; parg.rcword=NULL;
            for(int k=0;k+1<dnw;k+=2){ if((drc[k]&0xffff)==0x4020 && (drc[k+1]>>16)==0x1001){ parg.rcword=drc+k;
                uint32_t nv=(uint32_t)BIG.dma+0x3000; parg.rcval0=0x4020|((nv&0xffff)<<16); parg.rcval1=(0x1001u<<16)|((nv>>16)&0xffff); break; } }
            if(parg.rcword) parg.rcmode=1; }
        memset((char*)BIG.cpu+0x3000,0,0x400); orki_bsync(fd,&BIG,RKNPU_MEM_SYNC_TO_DEVICE); patching=1;
    }
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub; memset(&sub,0,sizeof sub);
    uint32_t flg=0x5; { const char*e=getenv("ORK_RESHAPE_FLAGS"); if(e)flg=(uint32_t)strtoul(e,0,0); }  /* vendor used 0x5 */
    sub.flags=flg; sub.task_number=tn; sub.task_obj_addr=c->task.obj; sub.core_mask=RKNPU_CORE0_MASK;
    sub.fence_fd=-1; sub.timeout=orki_ew_timeout_ms();
    sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,tn}; /* vendor sets all 3 */
    if(patching) pthread_create(&pth,NULL,ork_rsh_patcher,&parg);   /* fire the mid-submit far-ahead descriptor patch */
    int ok=-1; double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bsync(fd,&BIG,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; if(us)*us=ork_now_us()-t0; }
    else orki_bsync(fd,&BIG,RKNPU_MEM_SYNC_FROM_DEVICE);
    if(patching){ pthread_join(pth,NULL); int mnz=0; uint16_t*mg=(uint16_t*)((char*)BIG.cpu+0x3000); for(int i=0;i<512;i++)if(mg[i])mnz++;
        fprintf(stderr,"[patch] ring=%u idx=%u delay=%uus -> gemm-marker(0x3000) nz=%d -> %s\n",
            tn,parg.idx,parg.delay_us,mnz, mnz?"WRITTEN (live far-ahead patch HONORED)":"zero (patch NOT honored / prefetched)"); }
    if(gemm_raw){ uint16_t*g=(uint16_t*)((char*)BIG.cpu+0x3000); for(int i=0;i<gemm_words;i++)gemm_raw[i]=g[i]; }       /* 0xffff0000 */
    uint32_t roff=0x3680; { const char*e=getenv("ORK_RESHAPE_ROUT"); if(e)roff=(uint32_t)strtoul(e,0,0); } /* reshape-out read offset (0x3680 atom-8 base; 0x3280=task2 out) */
    if(reshape_raw){ uint16_t*r=(uint16_t*)((char*)BIG.cpu+roff); for(int i=0;i<reshape_words;i++)reshape_raw[i]=r[i]; }
    orki_bdestroy(fd,&BIG);
    #undef TK
    return ok;
}

int ork_npu_reshape_probe_f16(ork_npu *c,int M,int N,const uint16_t *src,uint16_t *out_raw,int out_words,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(N!=64||M<1||M>8) return -2;   /* WIP: captured pattern/geometry is M=8,N=64 */
    static const int WPOS[64]={0,65,136,201,272,337,408,473,544,609,680,745,816,881,952,1017,1026,1091,1162,
        1227,1298,1363,1434,1499,1570,1635,1706,1771,1842,1907,1978,2043,2052,2117,2188,2253,2324,2389,2460,
        2525,2596,2661,2732,2797,2868,2933,3004,3069,3078,3143,3214,3279,3350,3415,3486,3551,3622,3687,3758,
        3823,3894,3959,4030,4095};
    size_t isz=(size_t)M*N*2; if(isz<4096)isz=4096;
    size_t osz=(size_t)M*N*2*2; if(osz<8192)osz=8192;   /* generous output room */
    struct buf In=orki_bcreate(fd,isz,0x403,-1), W=orki_bcreate(fd,8192,0x403,-1), O=orki_bcreate(fd,osz,0x403,-1);
    if(!In.cpu||!W.cpu||!O.cpu){ orki_bdestroy(fd,&In);orki_bdestroy(fd,&W);orki_bdestroy(fd,&O); return -2; }
    memset(In.cpu,0,isz); memset(W.cpu,0,8192); memset(O.cpu,0,osz);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) ((uint16_t*)In.cpu)[(size_t)m*N+n]=src[(size_t)m*N+n];
    for(int i=0;i<64;i++) ((uint16_t*)W.cpu)[WPOS[i]]=0x3c00;   /* fp16 1.0 permutation (channel reorder) */
    orki_bsync(fd,&In,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_NONE);
    uint32_t rc[REGCMD_RESHAPE_F16_N]; memcpy(rc,REGCMD_RESHAPE_F16,sizeof rc);
    orki_setrn(rc,REGCMD_RESHAPE_F16_N,RK_CNA_FEATURE_DATA_ADDR,(uint32_t)In.dma);    /* input base (CNA activation) */
    orki_setrn(rc,REGCMD_RESHAPE_F16_N,RK_CNA_WEIGHT_DATA_ADDR,(uint32_t)W.dma);     /* weight base (permutation) */
    orki_setrn(rc,REGCMD_RESHAPE_F16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);    /* output base (DPU) */
    { const char*e;   /* RE: reconcile the reshape read geometry to OUR contiguous [M][N] input pitch */
      if((e=getenv("ORK_RSH_107C"))) orki_setrn(rc,REGCMD_RESHAPE_F16_N,RK_CNA_DMA_CON1,(uint32_t)strtoul(e,0,0));
      if((e=getenv("ORK_RSH_1080"))) orki_setrn(rc,REGCMD_RESHAPE_F16_N,RK_CNA_DMA_CON2,(uint32_t)strtoul(e,0,0)); }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task*t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0xd; t->int_mask=0x300; t->int_clear=0x1ffff;
      t->regcfg_amount=108; t->regcmd_addr=(uint32_t)c->regcmd.dma; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); }
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};sub.timeout=orki_ew_timeout_ms();
    int ok=-1; double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; if(us)*us=ork_now_us()-t0; }
    else { orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); }   /* coherent buffer: read partial write on fail too */
    if(out_raw){ int w=(int)(osz/2); if(w>out_words)w=out_words; for(int i=0;i<w;i++) out_raw[i]=((uint16_t*)O.cpu)[i]; }
    orki_bdestroy(fd,&In);orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);
    return ok;
}

int ork_npu_probe_silu_std_f16(ork_npu *c,const ork_f16 *in,int M,int N,
                               uint32_t idx_off,uint32_t cfg4064,uint32_t cfg4068,
                               const int16_t *lut,int nlut,ork_f16 *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)   /* fp16 atom-8, 2-byte, surf_stride=M*16 */
    const uint16_t *i16=(const uint16_t*)in; uint16_t *o16=(uint16_t*)out;
    /* #35 FIX: allocate this op's buffers + submit in the CURRENTLY-ACTIVE domain, NOT a hardcoded dom0.
     * A standalone SDP LUT-op runs right after a matmul that may have orki_dom_activate()'d a NON-0 domain
     * (multi-domain FFN chain); if the buffers live in dom0 and it submits iommu_domain_id=0 while
     * c->dom_active is that other domain, the submit WEDGES (errno 110) — REPRODUCED in isolation by
     * i16_shape_probe [G] (dom0 matmul->silu clean; dom1 matmul->silu wedges). Match the active domain. */
    int dom = getenv("ORK_I16_DOM0") ? 0 : c->dom_active;   /* ORK_I16_DOM0: A/B — force dom0 (disable the domain fix) */
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,dom); if(!A.cpu)return -2;
    struct buf O=orki_bcreate(fd,sz,0x403,dom); if(!O.cpu){orki_bdestroy(fd,&A);return -2;}
    struct buf Lrc=orki_bcreate(fd,(size_t)REGCMD_SILU_LUT_N*4,0x403,dom); if(!Lrc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);return -2;}
    struct buf Lsc=orki_bcreate(fd,4096,0x403,dom); if(!Lsc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);return -2;}
    memset(A.cpu,0,sz);memset(O.cpu,0,sz);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) *(uint16_t*)((char*)A.cpu+EWCUBEH(m,n))=i16[m*N+n];
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);

    /* submit 1: LUT-load */
    memcpy(Lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
    orki_setrn((uint32_t*)Lrc.cpu,REGCMD_SILU_LUT_N,RK_DPU_DST_BASE_ADDR,(uint32_t)Lsc.dma);
    if(lut){ uint32_t*lr=(uint32_t*)Lrc.cpu; int j=0;
        for(int k=0;k+1<REGCMD_SILU_LUT_N;k+=2){ if((lr[k]&0xffff)==0x4104){ int32_t v=(j<nlut)?(int32_t)lut[j]:0; j++;
            lr[k]=0x4104|((uint32_t)(v&0xffff)<<16); lr[k+1]=(0x1001u<<16)|(((uint32_t)v>>16)&0xffff); } } }
    orki_bsync(fd,&Lrc,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
      t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=1097; t->regcmd_addr=Lrc.dma;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      /* ping-pong OFF (0x1, NOT ork_ppflags's 0x5) for the LUT-load: ping-pong swaps register banks the
       * instant the task's config completes, racing the LUT's SRAM-commit side effect. Standalone there's
       * nothing to race, but IN A CHAIN (after a preceding matmul) the race soft-resets the NPU (#35 int16
       * silu in-chain wedge). Matches ork_mm_run_i8_silu's LUT-load + AGENTS.md "ping-pong OFF for LUT chains". */
      struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x1;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.timeout=orki_ew_timeout_ms();sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc); return -1; }
    }

    /* submit 2: standalone fp16 activation op */
    uint32_t rc[REGCMD_SILU_STD_F16_N]; memcpy(rc,REGCMD_SILU_STD_F16,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_SILU_STD_F16_N,M,N);
    orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_SDP_5040,0);
    orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_SDP_5038,0);
    orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_SDP_5018,(uint32_t)A.dma);
    orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_DPU_R4110,idx_off);
    orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_DPU_BN_ALU_CFG,cfg4064);
    orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_DPU_BN_MUL_CFG,cfg4068);
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk);
    tk->enable_mask=0x18; tk->int_mask=0x300; tk->int_clear=0x1ffff; tk->regcfg_amount=69; tk->regcmd_addr=c->regcmd.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms(); double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,dom)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) o16[m*N+n]=*(uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(us)*us=t1; }
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);
    #undef EWCUBEH
    return ok;
}

/* Faithful fp16 replay: run RKNN's fp16 LUT-LOAD program (loader/ln, the LE-table exponential-mode loader with
 * RKNN's curve baked in) verbatim + the fp16 compute op (REGCMD_SILU_STD_F16) verbatim — patching only the I/O
 * addresses + M/N. Unlike ork_npu_probe_silu_std_f16, this uses the fp16 loader (NOT the int8 LO-table loader)
 * and keeps the compute's baked index params. in/out fp16 [M*N], N%8==0. 0/ok,-1,-2,-3. */
int ork_npu_replay_full_f16(ork_npu *c,const uint32_t *loader,int ln,const ork_f16 *in,int M,int N,ork_f16 *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)||ln<2200||ln>2300) return -2;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)
    const uint16_t *i16=(const uint16_t*)in; uint16_t *o16=(uint16_t*)out;
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,-1); if(!A.cpu)return -2;              /* orig x */
    struct buf S=orki_bcreate(fd,sz,0x403,-1); if(!S.cpu){orki_bdestroy(fd,&A);return -2;} /* stage-1 sigmoid intermediate */
    struct buf O=orki_bcreate(fd,sz,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&S);return -2;}
    struct buf Lrc=orki_bcreate(fd,(size_t)ln*4,0x403,-1); if(!Lrc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&S);orki_bdestroy(fd,&O);return -2;}
    struct buf Lsc=orki_bcreate(fd,4096,0x403,-1); if(!Lsc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&S);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);return -2;}
    memset(A.cpu,0,sz);memset(S.cpu,0,sz);memset(O.cpu,0,sz);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) *(uint16_t*)((char*)A.cpu+EWCUBEH(m,n))=i16[m*N+n];
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&S,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    #define SUBMIT1(REG,RN,RA) do{ memcpy(c->regcmd.cpu,(REG),(size_t)(RN)*4); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE); \
        struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk); \
        tk->enable_mask=0x18; tk->int_mask=0x300; tk->int_clear=0x1ffff; tk->regcfg_amount=(RA); tk->regcmd_addr=c->regcmd.dma; \
        orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); \
        struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.timeout=orki_ew_timeout_ms();sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; \
        if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bdestroy(fd,&A);orki_bdestroy(fd,&S);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc); return -1; } }while(0)
    /* submit 1: fp16 LUT-load (verbatim; patch only the scratch out addr) — uses Lrc not c->regcmd (2210 words) */
    memcpy(Lrc.cpu,loader,(size_t)ln*4);
    orki_setrn((uint32_t*)Lrc.cpu,ln,RK_DPU_DST_BASE_ADDR,(uint32_t)Lsc.dma);
    orki_bsync(fd,&Lrc,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
      t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=1097; t->regcmd_addr=Lrc.dma;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.timeout=orki_ew_timeout_ms();sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bdestroy(fd,&A);orki_bdestroy(fd,&S);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc); return -1; }
    }
    /* submit 2: stage-1 (REGCMD_SILU_STD_F16, sigmoid via LE-LUT) VERBATIM, x@A -> sigmoid@S */
    { uint32_t rc[REGCMD_SILU_STD_F16_N]; memcpy(rc,REGCMD_SILU_STD_F16,sizeof rc);
      orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)S.dma);
      orki_setrn(rc,REGCMD_SILU_STD_F16_N,RK_SDP_5018,(uint32_t)A.dma);
      SUBMIT1(rc,REGCMD_SILU_STD_F16_N,69); }
    /* submit 3: stage-2 (REGCMD_SILU_F16_T2, x*sigmoid) VERBATIM, sigmoid@S (0x5018) * x@A (0x5038) -> O */
    { uint32_t rc[REGCMD_SILU_F16_T2_N]; memcpy(rc,REGCMD_SILU_F16_T2,sizeof rc);
      orki_setrn(rc,REGCMD_SILU_F16_T2_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
      orki_setrn(rc,REGCMD_SILU_F16_T2_N,RK_SDP_5018,(uint32_t)S.dma);
      orki_setrn(rc,REGCMD_SILU_F16_T2_N,RK_SDP_5038,(uint32_t)A.dma);
      SUBMIT1(rc,REGCMD_SILU_F16_T2_N,69); }
    orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) o16[m*N+n]=*(uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(us)*us=0;
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&S);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);
    #undef SUBMIT1
    #undef EWCUBEH
    return 0;
}

int ork_npu_replay_softmax_f16(ork_npu *c, const void *in, void *out, double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    struct buf IN=orki_bcreate(fd,32768,0x403,-1), OUT=orki_bcreate(fd,32768,0x403,-1),
               SCR=orki_bcreate(fd,237568,0x403,-1), LUT=orki_bcreate(fd,65536,0x403,-1),
               WT=orki_bcreate(fd,(size_t)(SM_WT_WORDS+64)*4,0x403,-1);
    if(!IN.cpu||!OUT.cpu||!SCR.cpu||!LUT.cpu||!WT.cpu){
        orki_bdestroy(fd,&IN);orki_bdestroy(fd,&OUT);orki_bdestroy(fd,&SCR);orki_bdestroy(fd,&LUT);orki_bdestroy(fd,&WT); return -2; }
    memcpy(IN.cpu,in,32768); memset(OUT.cpu,0,32768); memset(SCR.cpu,0,237568); memset(LUT.cpu,0,65536);
    memcpy(WT.cpu,SM_WT,(size_t)SM_WT_WORDS*4);
    orki_bsync(fd,&IN,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&OUT,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&SCR,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&LUT,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&WT,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    /* address-rebase helper: patch a task's regcmd in place (addr regs -> our buffers; chain -> `nextdma`
     * or 0 to terminate). Returns nothing; operates on rc[0..nw). */
    #define SM_REBASE(rc,nw,nextdma,nextamt) do{ for(int k=0;k+1<(nw);k+=2){ \
        unsigned _a=(rc)[k]&0xffff, _d=(rc)[k+1]>>16; uint32_t _v=(((rc)[k+1]&0xffff)<<16)|((rc)[k]>>16); \
        int _ia=(_d==0x1001&&_a==0x4020)||(_d==0x2001&&(_a==0x5018||_a==0x5038))||(_d==0x0201&&(_a==0x1070||_a==0x1110)); \
        if(_ia){ uint32_t _nv=_v; \
            if      (_v>=0xfffb6000u&&_v<0xfffbe000u) _nv=(uint32_t)IN.dma +(_v-0xfffb6000u); \
            else if (_v>=0xfffae000u&&_v<0xfffb6000u) _nv=(uint32_t)OUT.dma+(_v-0xfffae000u); \
            else if (_v>=0xfffbe000u&&_v<0xffff8000u) _nv=(uint32_t)SCR.dma+(_v-0xfffbe000u); \
            else if (_v>=0xffffa280u&&_v<0xffffab00u) _nv=(uint32_t)WT.dma +(_v-0xffffa280u); \
            else if (_v>=0xffffab00u&&_v<0xfffff000u) _nv=(uint32_t)LUT.dma+(_v-0xffffab00u); \
            if(_nv!=_v){ (rc)[k]=_a|((_nv&0xffff)<<16); (rc)[k+1]=(_d<<16)|((_nv>>16)&0xffff); } } \
        /* PC-chain descriptor: 0x0010 = next task's regcmd addr, 0x0014 = next task's register-AMOUNT,
         * RECOMPUTED as (nextamt+3)/2 for the ACTUAL next task (verified: amt 69->0x24, 1097->0x226; ==
         * the kernel's (amt+EXTRA+scale-1)/scale-1 with EXTRA=4,scale=2). MUST be recomputed, not preserved:
         * the captured value encodes the CAPTURE-order next task, which is wrong for a reordered chain.
         * nextdma==0 (last/no-chain) -> zero both to terminate. cf. run_chain_i8 (0x0014=0x0037 for its 108s). */ \
        if(_d==0x0101&&_a==0x0010){ if(nextdma){ uint32_t _nx=(uint32_t)(nextdma); (rc)[k]=0x0010|((_nx&0xffff)<<16); (rc)[k+1]=(0x0101u<<16)|((_nx>>16)&0xffff); } else { (rc)[k]=0; (rc)[k+1]=0; } } \
        if(_d==0x0101&&_a==0x0014){ if(nextdma){ uint32_t _na=(uint32_t)(((nextamt)+3)/2); (rc)[k]=0x0014|((_na&0xffff)<<16); (rc)[k+1]=(0x0101u<<16)|((_na>>16)&0xffff); } else { (rc)[k]=0; (rc)[k+1]=0; } } } }while(0)
    /* ORK_SM_FULLIMG (gate removed; always on): FAITHFUL single-delta full-image hardware-chain replay. The vendor's 5 buffers are ONE
     * contiguous IOVA image (0xfffae000 out, 0xfffb6000 in, 0xfffbe000 scratch, 0xffff8000 regcmd+weights),
     * and task4's LUT-write (0x4020=0xffffab00) points INTO the regcmd region — a relationship my split
     * IN/OUT/SCR/LUT/regcmd buffers destroy. Here: ONE big buffer covering the image, every task's regcmd at
     * its captured offset, and EVERY address blanket-rebased by a SINGLE delta (BIG.dma - 0xfffae000) so all
     * internal references (chain 0x0010, data addrs, weights, task4->regcmd write) stay consistent. 0x0014
     * (amount, small) isn't in the addr range so it's preserved. Tasks laid out at the vendor's exact offsets
     * (capture order), task_number=9, one submit — the maximally-faithful hardware chain. */
    if(!getenv("ORK_SM_PERTASK")){   /* DEFAULT = single-submit hardware chain (contiguous image, ping-pong off) */
        double t0=ork_now_us();
        const uint32_t IB=0xfffae000u, IE=0xfffff000u; size_t ISZ=IE-IB;   /* out|in|scratch|regcmd+wt */
        static const uint32_t TADDR[9]={0xffffab00u,0xffffad80u,0xffffb000u,0xffffb280u,0xffffb380u,0xffffd600u,0xffffd880u,0xffffdc00u,0xffffde40u};
        struct buf BIG=orki_bcreate(fd,ISZ,0x403,-1);
        if(!BIG.cpu){ orki_bdestroy(fd,&IN);orki_bdestroy(fd,&OUT);orki_bdestroy(fd,&SCR);orki_bdestroy(fd,&LUT);orki_bdestroy(fd,&WT); return -2; }
        memset(BIG.cpu,0,ISZ);
        uint32_t delta=(uint32_t)BIG.dma - IB;
        memcpy((char*)BIG.cpu + (0xfffb6000u-IB), in, 32768);                       /* input @ h4 */
        memcpy((char*)BIG.cpu + (0xffffa280u-IB), SM_WT, (size_t)SM_WT_WORDS*4);     /* weights @ 0xffffa280 */
        int NT=getenv("ORK_SM_NTASK")?atoi(getenv("ORK_SM_NTASK")):9; if(NT<1)NT=1; if(NT>9)NT=9;   /* RE: bisect */
        struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,(size_t)NT*sizeof *tk);
        for(int j=0;j<NT;j++){ int t=j; uint32_t *rc=(uint32_t*)((char*)BIG.cpu + (TADDR[j]-IB));   /* capture order = identity */
            int slot=(j<8)?(int)((TADDR[j+1]-TADDR[j])/4):SM_TASK_WORDS[t]; int nw=SM_TASK_WORDS[t]; if(nw>slot)nw=slot;
            memcpy(rc,SM_TASKS[t],(size_t)nw*4);
            for(int k=0;k+1<nw;k+=2){ unsigned a=rc[k]&0xffff,d=rc[k+1]>>16; uint32_t v=((rc[k+1]&0xffff)<<16)|(rc[k]>>16);
                /* BLANKET single-delta rebase: shift EVERY reference into the vendor image by delta. In the
                 * contiguous-image regime this is correct even for 1001:0x4110 (exp-LUT read ptr into the h2
                 * staging area) — validated empirically: blanket reaches task5 (counter 5), whitelist only
                 * task4 (counter 4). 0x0014 (amount, small) is out of [IB,IE) so it's untouched. */
                if(v>=IB && v<IE){ uint32_t nv=v+delta; rc[k]=a|((nv&0xffff)<<16); rc[k+1]=(d<<16)|((nv>>16)&0xffff); } }
            tk[j].enable_mask=SM_TASK_ENABLE[t]; tk[j].int_mask=0x300; tk[j].int_clear=0x1ffff;
            tk[j].regcfg_amount=SM_TASK_AMT[t]; tk[j].regcmd_addr=(uint32_t)BIG.dma + (TADDR[j]-IB);
        }
        orki_bsync(fd,&BIG,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        struct rknpu_submit s; memset(&s,0,sizeof s);
        /* flags=0x1 (RKNPU_JOB_PC only) — match the vendor softmax. The default 0x5 sets RKNPU_JOB_PINGPONG
         * (0x4) too; ping-pong double-buffers the register banks across tasks, and its bank-swap racing the
         * LUT-load commit is the flaky task4->task5 stall. The vendor runs softmax with ping-pong OFF. */
        s.flags=0x1; s.task_number=(uint32_t)NT; s.task_obj_addr=c->task.obj; s.core_mask=RKNPU_CORE0_MASK; s.fence_fd=-1; s.timeout=3000;
        s.subcore_task[0]=(struct rknpu_subcore_task){0,(uint32_t)NT};
        fprintf(stderr,"[softmax-replay] FULLIMG: %d-task chain, contiguous image, single-delta rebase, PINGPONG OFF (flags=0x1)\n",NT);
        int r=orki_rknpu_submit_ioctl(fd,&s,-1)?-1:0;
        if(r==0){ orki_bsync(fd,&BIG,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(out,BIG.cpu,32768); if(us)*us=ork_now_us()-t0; }  /* output @ h5 = BIG+0 */
        else fprintf(stderr,"[softmax-replay] FULLIMG chain failed\n");
        orki_bdestroy(fd,&BIG); orki_bdestroy(fd,&IN);orki_bdestroy(fd,&OUT);orki_bdestroy(fd,&SCR);orki_bdestroy(fd,&LUT);orki_bdestroy(fd,&WT);
        return r;
    }
    /* KERNEL-SEQUENCED (the accel/rocket model): submit each task as its OWN task_number=1 program, in
     * dataflow order 0..8 — one task per PC program, no in-regcmd PC-chain (0101:0x0010) rewriting. Each
     * task is the proven single-task submit path; intermediates persist in the resident SCR/IN/OUT/LUT
     * buffers between submits, and NO reset happens between them (orki_act(RESET) ran once above), so the exp
     * LUT that task4 loads into DPU SRAM survives for task5's exp. This is the correct sequencing model:
     * the earlier hardware-chained (task_number=9 + 0101 chain) attempts stalled at the chained LUT load. */
    static const int ORDER[9]={0,1,2,3,4,5,6,7,8};   /* capture/dataflow order */
    int rc_ret=0; double t0=ork_now_us();
    /* ORK_SM_1SUBMIT: the CHEAP hardware-chained form — ONE ioctl, task_number=9, ~one submit-floor cost.
     * The vendor runs softmax exactly this way (captured task_number=27), so it IS possible on rknpu. Replicate
     * the PROVEN run_chain/run_multicore recipe EXACTLY: concat regcmds, REWRITE the 0101:0x0010 chain to the
     * next task (terminate the last), op_idx=0 (memset), and populate ALL THREE subcore_task[]={0,9} (the
     * single-subcore / chain-zeroed / op_idx=1 variants each failed differently). */
    if(getenv("ORK_SM_1SUBMIT")){
        /* ONE-submit hardware chain replaying the VENDOR'S EXACT TASK LAYOUT. The vendor places each task's
         * regcmd at a 64-byte-ALIGNED offset (captured regcmd_addrs 0xffffab00,ad80,b000,b280,b380,d600,d880,
         * dc00,de40 -> relative words below). My earlier TIGHT packing landed task4 misaligned (+80 mod 128)
         * -> hung entering task4; the vendor's aligned layout is what the hardware chain-walk needs. Capture
         * order (0,1,..,8) so 0x0014 next-amounts match; chain 0x0010 -> next task's aligned offset. */
        static const int VOFF[9]={0,160,320,480,544,2752,2912,3136,3280};   /* vendor relative offsets (words) */
        /* ORK_SM_SPLIT: TWO hardware-chained submits split at the task4->task5 wall (the LUT-load->exp
         * transition that hangs in one chain): submit A = tasks [0..4] (loads the exp LUT into DPU SRAM),
         * submit B = tasks [5..8] (exp uses the persisted LUT). The submit boundary lets the LUT SRAM commit
         * and re-arms the PC cleanly for task5. Both chains use the vendor-aligned layout + 0x0014 recompute. */
        if(getenv("ORK_SM_SPLIT")){
            static const int SEG[2][2]={{0,5},{5,9}};   /* [start,end) task index ranges */
            for(int seg=0; seg<2 && rc_ret==0; seg++){ int s0=SEG[seg][0], s1=SEG[seg][1], ntt=s1-s0;
                uint32_t *rcb=(uint32_t*)c->regcmd.cpu;
                /* SEGMENT-LOCAL offsets from 0 (each submit starts its regcmd at c->regcmd.dma+0, where task5
                 * works in the per-task path), preserving the vendor SLOT sizes (=aligned spacing). */
                int loc[6]={0}; for(int j=0;j<ntt;j++){ int gi=s0+j; int slot=(gi<8)?(VOFF[gi+1]-VOFF[gi]):SM_TASK_WORDS[ORDER[gi]]; loc[j+1]=loc[j]+slot; }
                struct rknpu_task *tks=(struct rknpu_task*)c->task.cpu; memset(tks,0,(size_t)ntt*sizeof *tks);
                for(int j=0;j<ntt;j++){ int gi=s0+j, t=ORDER[gi]; uint32_t *rc=rcb+loc[j];
                    int slot=loc[j+1]-loc[j]; int nw=SM_TASK_WORDS[t]; if(nw>slot)nw=slot;
                    memcpy(rc,SM_TASKS[t],(size_t)nw*4);
                    uint32_t nextdma=(j<ntt-1)?(uint32_t)(c->regcmd.dma+(size_t)loc[j+1]*4):0u;
                    SM_REBASE(rc,nw,nextdma,(j<ntt-1)?SM_TASK_AMT[ORDER[gi+1]]:0);
                    tks[j].enable_mask=SM_TASK_ENABLE[t]; tks[j].int_mask=0x300; tks[j].int_clear=0x1ffff;
                    tks[j].regcfg_amount=SM_TASK_AMT[t]; tks[j].regcmd_addr=c->regcmd.dma+(size_t)loc[j]*4;
                }
                orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
                struct rknpu_submit s; memset(&s,0,sizeof s);
                s.flags=0x5; s.task_number=(uint32_t)ntt; s.task_obj_addr=c->task.obj; s.core_mask=RKNPU_CORE0_MASK; s.fence_fd=-1; s.timeout=3000;
                s.subcore_task[0]=(struct rknpu_subcore_task){0,(uint32_t)ntt};
                if(orki_rknpu_submit_ioctl(fd,&s,-1)){ fprintf(stderr,"[softmax-replay] SPLIT seg %d ([%d,%d)) failed\n",seg,s0,s1); rc_ret=-1; }
            }
            fprintf(stderr,"[softmax-replay] 2SUBMIT-SPLIT: chain [0..4] + chain [5..8] (vendor-aligned)\n");
            if(rc_ret==0){ orki_bsync(fd,&OUT,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(out,OUT.cpu,32768); if(us)*us=ork_now_us()-t0; }
            orki_bdestroy(fd,&IN);orki_bdestroy(fd,&OUT);orki_bdestroy(fd,&SCR);orki_bdestroy(fd,&LUT);orki_bdestroy(fd,&WT);
            return rc_ret;
        }
        int NT=getenv("ORK_SM_NTASK")?atoi(getenv("ORK_SM_NTASK")):9; if(NT<1)NT=1; if(NT>9)NT=9;  /* RE: bisect */
        uint32_t *rcbuf=(uint32_t*)c->regcmd.cpu;
        struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,(size_t)NT*sizeof *tk);
        for(int j=0;j<NT;j++){ int t=ORDER[j]; uint32_t *rc=rcbuf+VOFF[j];
            int slot=(j<8)?(VOFF[j+1]-VOFF[j]):SM_TASK_WORDS[t]; int nw=SM_TASK_WORDS[t]; if(nw>slot)nw=slot;  /* fit the slot */
            memcpy(rc,SM_TASKS[t],(size_t)nw*4);
            uint32_t nextdma=(j<NT-1)?(uint32_t)(c->regcmd.dma+(size_t)VOFF[j+1]*4):0u;
            SM_REBASE(rc,nw,nextdma,(j<NT-1)?SM_TASK_AMT[ORDER[j+1]]:0);
            tk[j].enable_mask=SM_TASK_ENABLE[t]; tk[j].int_mask=0x300; tk[j].int_clear=0x1ffff;
            tk[j].regcfg_amount=SM_TASK_AMT[t]; tk[j].regcmd_addr=c->regcmd.dma+(size_t)VOFF[j]*4;
        }
        orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
        orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        struct rknpu_submit s; memset(&s,0,sizeof s);
        s.flags=0x5; s.task_number=(uint32_t)NT; s.task_obj_addr=c->task.obj; s.core_mask=RKNPU_CORE0_MASK; s.fence_fd=-1; s.timeout=3000;
        s.subcore_task[0]=(struct rknpu_subcore_task){0,(uint32_t)NT};
        fprintf(stderr,"[softmax-replay] 1SUBMIT: %d-task hardware chain, vendor-exact aligned layout\n",NT);
        if(orki_rknpu_submit_ioctl(fd,&s,-1)){ fprintf(stderr,"[softmax-replay] 9-task chain failed\n"); rc_ret=-1; }
        if(rc_ret==0){ orki_bsync(fd,&OUT,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(out,OUT.cpu,32768); if(us)*us=ork_now_us()-t0; }
        orki_bdestroy(fd,&IN);orki_bdestroy(fd,&OUT);orki_bdestroy(fd,&SCR);orki_bdestroy(fd,&LUT);orki_bdestroy(fd,&WT);
        return rc_ret;
    }
    for(int j=0;j<9 && rc_ret==0;j++){ int t=ORDER[j]; int nw=SM_TASK_WORDS[t];
        uint32_t *rc=(uint32_t*)c->regcmd.cpu; memcpy(rc,SM_TASKS[t],(size_t)nw*4);
        SM_REBASE(rc,nw,0,0);                        /* addr remap only; nextdma=0 zeros the chain words */
        struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk);
        tk->enable_mask=SM_TASK_ENABLE[t]; tk->int_mask=0x300; tk->int_clear=0x1ffff;
        tk->regcfg_amount=SM_TASK_AMT[t]; tk->regcmd_addr=c->regcmd.dma;
        orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
        orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        struct rknpu_submit s; memset(&s,0,sizeof s);
        s.flags=0x5; s.task_number=1; s.task_obj_addr=c->task.obj; s.core_mask=RKNPU_CORE0_MASK;
        s.fence_fd=-1; s.timeout=3000;
        s.subcore_task[0]=s.subcore_task[1]=s.subcore_task[2]=(struct rknpu_subcore_task){0,1};
        if(orki_rknpu_submit_ioctl(fd,&s,-1)){ fprintf(stderr,"[softmax-replay] task %d (enable=0x%x amt=%d) submit failed\n",t,SM_TASK_ENABLE[t],SM_TASK_AMT[t]); rc_ret=-1; }
    }
    if(rc_ret==0){ orki_bsync(fd,&OUT,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(out,OUT.cpu,32768); if(us)*us=ork_now_us()-t0; }
    #undef SM_REBASE
    orki_bdestroy(fd,&IN);orki_bdestroy(fd,&OUT);orki_bdestroy(fd,&SCR);orki_bdestroy(fd,&LUT);orki_bdestroy(fd,&WT);
    return rc_ret;
}
