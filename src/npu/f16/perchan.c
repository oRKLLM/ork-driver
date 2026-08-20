/* npu/f16/perchan.c — per-channel multiply/scale, elementwise, norm helpers.
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

int ork_npu_mm_perchan_f16_fused(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B,
                                 const uint16_t *scale,uint16_t *out){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64||(N&7)) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N*2,0x403,-1); if(!W.cpu) return -2;
    int NN=N/16,KT=K/32; uint16_t*bb=W.cpu;
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*16+nl)];
    size_t osz=(size_t)M*N*2; if(osz<4096)osz=4096;
    struct buf O=orki_bcreate(fd,osz,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;} memset(O.cpu,0,osz);
    struct buf SB=orki_bcreate(fd,4096,0x403,-1); if(!SB.cpu){orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);return -2;} memset(SB.cpu,0,4096);
    { ork_f16*sb=(ork_f16*)SB.cpu; for(int n=0;n<N;n++) sb[n]=*(const ork_f16*)&scale[n]; }   /* per-channel scale, CONTIGUOUS [N] fp16 */
    uint16_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j];
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&SB,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_NONE);                  /* prime fp16 pipeline (layer owns the reset) */
    uint32_t base[REGCMD_N], rc[REGCMD_I8_EW_N];
    int sched=((K&(K-1))==0 && K>=128 && K<2048);
    orki_synth(base,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,sched,CBUF);
    orki_set_f16_out_fp16in(base,M,N);                                   /* main lane: fp16 CONTIGUOUS out */
    orki_splice_ew_lane(rc,base);                                        /* add the 0x50xx EW-operand lane */
    /* EW-mul: multiply the on-chip fp16 accumulator by the per-channel operand. fp16 EW config + per-channel ERDMA. */
    uint32_t ewcfg=getenv("ORK_F16EW_CFG")?strtoul(getenv("ORK_F16EW_CFG"),0,0):0x108003c4;   /* vendor fp16 EW mul */
    uint32_t erdma=getenv("ORK_F16EW_ERDMA")?strtoul(getenv("ORK_F16EW_ERDMA"),0,0):0x00000008;/* per-channel + 2-byte */
    uint32_t r4050=getenv("ORK_F16EW_4050")?strtoul(getenv("ORK_F16EW_4050"),0,0):0x00000127; /* out row cfg + EW-enable bit0 */
    orki_setrn(rc,REGCMD_I8_EW_N,RK_DPU_EW_CFG,ewcfg);
    orki_setrn(rc,REGCMD_I8_EW_N,RK_DPU_EW_CVT_OFFSET,0x00000000);
    orki_setrn(rc,REGCMD_I8_EW_N,RK_DPU_EW_CVT_SCALE,0x00000001);
    orki_setrn(rc,REGCMD_I8_EW_N,RK_DPU_BS_OW_CFG,r4050);
    { const char*e=getenv("ORK_F16EW_4010"); if(e) orki_setrn(rc,REGCMD_I8_EW_N,RK_DPU_OUT_PRECISION,(uint32_t)strtoul(e,0,0)); } /* fp16 DATA_FORMAT + EW-enable bits sweep */
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5004,0x0000000e);
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5008,0x00000001);               /* RDMA_OPERATION_ENABLE */
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_500C,(uint32_t)(M-1));          /* WIDTH=M-1 */
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5010,0x00000000);              /* HEIGHT=1 */
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5014,(uint32_t)(N-1));          /* CHANNEL=N-1 */
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5034,erdma);
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5038,(uint32_t)SB.dma);         /* EW operand = scale[N] */
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5018,(uint32_t)SB.dma);
    orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_501C,0x00000002);              /* BRDMA_DATA_USE=1 */
    memcpy(c->regcmd.cpu,rc,REGCMD_I8_EW_N*4); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task*tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk);
    tk->enable_mask=0x1d; tk->int_mask=0x300; tk->int_clear=0x1ffff;               /* 0x1d: enable EW/second lane */
    tk->regcfg_amount=REGCMD_I8_EW_N/2; tk->regcmd_addr=(uint32_t)c->regcmd.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    uint32_t to_ms=3000; { const char*e=getenv("ORK_EW_TIMEOUT"); if(e){ unsigned v=(unsigned)strtoul(e,0,0); if(v)to_ms=v; } }
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};sub.timeout=to_ms;
    int ok=-1;
    for(int rep=0;rep<2;rep++){ if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; continue; } orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; }
    if(ok==0) for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=((uint16_t*)O.cpu)[(size_t)m*N+n];  /* CONTIGUOUS */
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);orki_bdestroy(fd,&SB);
    return ok;
}

int ork_npu_ewmul_f16(ork_npu *c,const ork_f16 *up,const ork_f16 *silu,int M,int N,ork_f16 *out,double *us){
    if(c && c->daemon){ if(us)*us=0; return orkd_ewmul_f16(c->daemon,up,silu,M,N,out); }   /* Path B: SDP on the daemon */
    int fd=c->fd, dom=c->dom_active;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;               /* N multiple of the fp16 atom (8) */
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)   /* BYTE offset, fp16 atom=8, surf_stride=M*16 */
    const uint16_t *u16=(const uint16_t*)up,*s16=(const uint16_t*)silu; uint16_t *o16=(uint16_t*)out;
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;                  /* fp16 cube = M*N*2 bytes */
    struct buf A=orki_bcreate(fd,sz,0x403,dom); if(!A.cpu)return -2;
    struct buf B=orki_bcreate(fd,sz,0x403,dom); if(!B.cpu){orki_bdestroy(fd,&A);return -2;}
    struct buf O=orki_bcreate(fd,sz,0x403,dom); if(!O.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);return -2;}
    memset(A.cpu,0,sz);memset(B.cpu,0,sz);memset(O.cpu,0,sz);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int p=EWCUBEH(m,n);
        *(uint16_t*)((char*)A.cpu+p)=u16[m*N+n]; *(uint16_t*)((char*)B.cpu+p)=s16[m*N+n]; }
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (XP_SDP=RC_SDPKW, keep-warm-aware; == the old inline reset) */
    uint32_t rc[REGCMD_MUL_F16_N]; memcpy(rc,REGCMD_MUL_F16,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_MUL_F16_N,M,N);
    orki_setrn(rc,REGCMD_MUL_F16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,REGCMD_MUL_F16_N,RK_SDP_5018,(uint32_t)A.dma);
    orki_setrn(rc,REGCMD_MUL_F16_N,RK_SDP_5038,(uint32_t)B.dma);
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    /* NONBLOCK DOORBELL (spine uniformity + system latency): a blocking submit stalls this thread in-kernel for
     * the whole op (uninterruptible, D-state-wedge-prone; in orkd it also blocks servicing other clients).
     * Submit NONBLOCK with ping-pong OFF, and detect completion by a FULL-SURFACE poll of the fp16 output seeded
     * with an out-of-range poison (inf = 0x7c00, impossible for a bounded ewmul). fp16 write-order isn't
     * last-element-last, so poll the WHOLE cube (the int4/matmul poll technique). No timer — the poison is a real
     * completion signal. (int8 SDP output has no free poison value; those need a trailing witness task instead.) */
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ volatile uint16_t*db=(volatile uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); *db=0x7c00; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); }
    __asm__ volatile("dsb ish":::"memory");
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=(ork_ppflags()&~0x4u)|0x2u;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms(); double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,dom)){ double cap=(double)orki_ew_timeout_ms()*1000.0;
        for(;;){ int done=1; for(int m=0;m<M&&done;m++)for(int n=0;n<N;n++){ volatile uint16_t*db=(volatile uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); __asm__ volatile("dc civac,%0"::"r"(db):"memory"); if(*db==0x7c00u){done=0;break;} } if(done)break; if(ork_now_us()-t0>cap)break; }
        orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) o16[m*N+n]=*(uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(us)*us=t1; }
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&O);
    #undef EWCUBEH
    return ok;
}

int ork_npu_mm_perchan_f16(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B,
                           const uint16_t *scale,uint16_t *out,double *us){
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(K%32||N<1||N>c->soc->nmax||M<1||M>64) return -2;
    if(N%32==0){                                                    /* fast path: N is a valid tile width */
        uint16_t *G=malloc((size_t)M*N*2); if(!G) return -1;
        int rc=ork_npu_probe_f16_mm_f16out(c,M,K,N,A,B,G);          /* fp16 matmul -> CONTIGUOUS fp16 G (512/512) */
        if(rc==0) rc=ork_npu_mul_perchan_f16(c,(const ork_f16*)G,(const ork_f16*)scale,M,N,(ork_f16*)out,us); /* per-channel scale */
        free(G);
        return rc;
    }
    /* N%32!=0 (e.g. decode: N=queries=1): pad N up to a 32-multiple, zero-pad the B columns + scale vector,
     * compute the padded matmul+scale, then extract the real N columns. The zero-pad columns produce 0 (matmul
     * of a zero weight column) so they don't perturb the real columns; the tiny extra compute is negligible. */
    int Np=((N+31)/32)*32;
    uint16_t *Bp=malloc((size_t)K*Np*2), *scp=malloc((size_t)Np*2), *G=malloc((size_t)M*Np*2), *op=malloc((size_t)M*Np*2);
    if(!Bp||!scp||!G||!op){ free(Bp);free(scp);free(G);free(op); return -1; }
    for(int k=0;k<K;k++){ const uint16_t*br=B+(size_t)k*N; uint16_t*pr=Bp+(size_t)k*Np;
        for(int n=0;n<N;n++)pr[n]=br[n]; for(int n=N;n<Np;n++)pr[n]=0; }
    for(int n=0;n<N;n++)scp[n]=scale[n]; for(int n=N;n<Np;n++)scp[n]=0;
    int rc=ork_npu_probe_f16_mm_f16out(c,M,K,Np,A,Bp,G);
    if(rc==0) rc=ork_npu_mul_perchan_f16(c,(const ork_f16*)G,(const ork_f16*)scp,M,Np,(ork_f16*)op,us);
    if(rc==0) for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=op[(size_t)m*Np+n];
    free(Bp);free(scp);free(G);free(op);
    return rc;
}

int ork_npu_mm_perchan_f16_diag(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B,
                                const uint16_t *scale,uint16_t *out,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64||(N&31)) return -2;    /* N%32 for the diagonal matmul's K=N */
    /* DEVICE-RESIDENT: G stays on the NPU between the two matmuls — zero CPU touch of the intermediate. */
    #define TILE(dst,src,KK,NN) do{ int NT=(NN)/16,KT=(KK)/32; uint16_t*bb=(dst); \
        for(int nt=0;nt<NT;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++) \
          bb[(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk]=(src)[(size_t)(kt*32+kk)*(NN)+(nt*16+nl)]; }while(0)
    size_t gsz=(size_t)M*N*2; if(gsz<4096)gsz=4096;
    struct buf W1=orki_bcreate(fd,(size_t)K*N*2,0x403,-1), G=orki_bcreate(fd,gsz,0x403,-1),
               W2=orki_bcreate(fd,(size_t)N*N*2,0x403,-1), O=orki_bcreate(fd,gsz,0x403,-1);
    if(!W1.cpu||!G.cpu||!W2.cpu||!O.cpu){ orki_bdestroy(fd,&W1);orki_bdestroy(fd,&G);orki_bdestroy(fd,&W2);orki_bdestroy(fd,&O); return -1; }
    TILE(W1.cpu,B,K,N);                                             /* weight1 = B[K][N] */
    { uint16_t*d=W2.cpu; memset(d,0,(size_t)N*N*2); uint16_t*Draw=calloc((size_t)N*N,2);   /* weight2 = diag(scale)[N][N] */
      for(int n=0;n<N;n++) Draw[(size_t)n*N+n]=scale[n]; TILE(W2.cpu,Draw,N,N); free(Draw); }
    memset(G.cpu,0,gsz); memset(O.cpu,0,gsz);
    { uint16_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; }    /* activation1 = A (only host->dev copy of an INPUT) */
    orki_bsync(fd,&W1,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&W2,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&G,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_NONE);                  /* layer owns the reset (keep-warm-aware; no redundant manual reset) */
    uint32_t to_ms=3000; { const char*e=getenv("ORK_EW_TIMEOUT"); if(e){ unsigned v=(unsigned)strtoul(e,0,0); if(v)to_ms=v; } }
    struct rknpu_task*tk=(struct rknpu_task*)c->task.cpu;
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};sub.timeout=to_ms;
    int ok=0; double t0=ork_now_us();
    /* matmul1: A·B -> G (contiguous fp16, device); activation from c->Af. matmul2: G·diag -> O; activation = G (device). */
    struct { int K2,N2; uint32_t aA,aW,aO; } pass[2]={ {K,N,(uint32_t)c->Af.dma,(uint32_t)W1.dma,(uint32_t)G.dma},
                                                       {N,N,(uint32_t)G.dma,(uint32_t)W2.dma,(uint32_t)O.dma} };
    if(getenv("ORK_DIAG_CHAIN")){
        /* SINGLE-SUBMIT: PC-chain matmul1 -> matmul2 (both uniform enable=0xd, like run_chain_i8). G resident. */
        static uint32_t mm1[REGCMD_N], mm2[REGCMD_N];
        int s1=((K&(K-1))==0 && K>=128 && K<2048), s2=((N&(N-1))==0 && N>=128 && N<2048);
        orki_synth(mm1,M,K,N,pass[0].aA,pass[0].aW,pass[0].aO,s1,CBUF); orki_set_f16_out_fp16in(mm1,M,N);
        orki_synth(mm2,M,N,N,pass[1].aA,pass[1].aW,pass[1].aO,s2,CBUF); orki_set_f16_out_fp16in(mm2,M,N);
        ork_chain_prog progs[2]={ {mm1,REGCMD_N,0xd,108,216}, {mm2,REGCMD_N,0xd,108,-1} };
        int crc=ork_npu_chain_progs(c,2,progs,c->dom_active);
        if(!crc){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); if(us)*us=ork_now_us()-t0;
            for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=((uint16_t*)O.cpu)[(size_t)m*N+n]; }
        orki_bdestroy(fd,&W1);orki_bdestroy(fd,&G);orki_bdestroy(fd,&W2);orki_bdestroy(fd,&O);
        #undef TILE
        return crc;
    }
    for(int p=0; p<2 && ok==0; p++){
        uint32_t rc[REGCMD_N];
        int sched=((pass[p].K2&(pass[p].K2-1))==0 && pass[p].K2>=128 && pass[p].K2<2048);
        orki_synth(rc,M,pass[p].K2,pass[p].N2,pass[p].aA,pass[p].aW,pass[p].aO,sched,CBUF);
        orki_set_f16_out_fp16in(rc,M,pass[p].N2);
        memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
        memset(tk,0,sizeof *tk); tk->enable_mask=0xd; tk->int_mask=0x300; tk->int_clear=0x1ffff;
        tk->regcfg_amount=108; tk->regcmd_addr=(uint32_t)c->regcmd.dma; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        int done=-1; for(int rep=0;rep<2;rep++){ if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ done=-1; continue; }
            if(p==0) orki_bsync(fd,&G,RKNPU_MEM_SYNC_FROM_DEVICE|RKNPU_MEM_SYNC_TO_DEVICE); else orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); done=0; }
        if(done) ok=-1;
    }
    if(ok==0){ if(us)*us=ork_now_us()-t0;
        for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=((uint16_t*)O.cpu)[(size_t)m*N+n]; }  /* CONTIGUOUS */
    orki_bdestroy(fd,&W1);orki_bdestroy(fd,&G);orki_bdestroy(fd,&W2);orki_bdestroy(fd,&O);
    #undef TILE
    return ok;
}

int ork_npu_mul_perchan_f16(ork_npu *c,const ork_f16 *a,const ork_f16 *b,int M,int N,ork_f16 *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;
    #define PCH16(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)         /* fp16 cube byte offset, atom=8, surf=M*16 */
    const uint16_t *a16=(const uint16_t*)a,*b16=(const uint16_t*)b; uint16_t *o16=(uint16_t*)out;
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,-1), O=orki_bcreate(fd,sz,0x403,-1), B=orki_bcreate(fd,sz,0x403,-1);
    if(!A.cpu||!O.cpu||!B.cpu){ orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&B); return -2; }
    memset(A.cpu,0,sz); memset(O.cpu,0,sz); memset(B.cpu,0,sz);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) *(uint16_t*)((char*)A.cpu+PCH16(m,n))=a16[(size_t)m*N+n];
    for(int n=0;n<N;n++) ((uint16_t*)B.cpu)[n]=b16[n];               /* per-channel vector CONTIGUOUS [N] (fp16) */
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (XP_SDP=RC_SDPKW, keep-warm-aware; == the old inline reset) */
    uint32_t rc[REGCMD_MUL_F16_N]; memcpy(rc,REGCMD_MUL_F16,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_MUL_F16_N,M,N);
    orki_setrn(rc,REGCMD_MUL_F16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,REGCMD_MUL_F16_N,RK_SDP_5018,(uint32_t)A.dma);
    orki_setrn(rc,REGCMD_MUL_F16_N,RK_SDP_5038,(uint32_t)B.dma);
    orki_setrn(rc,REGCMD_MUL_F16_N,RK_SDP_5034,0x00000008);            /* ERDMA_DATA_MODE=0 (per-channel) + DATA_SIZE=TWO_BYTE */
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub; memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=1;
    sub.task_obj_addr=c->task.obj; sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1;
    sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; sub.timeout=orki_ew_timeout_ms();
    int ok=-1; double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; if(us)*us=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) o16[(size_t)m*N+n]=*(uint16_t*)((char*)O.cpu+PCH16(m,n)); }
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&O); orki_bdestroy(fd,&B);
    #undef PCH16
    return ok;
}

/* ork_npu_mul_perchan_f16_contig — per-channel fp16 MUL that reads a CONTIGUOUS [M][N] fp16 input (the native
 * fp16 matmul output layout), via the vendor task13 config (FLYING_MODE + NOTCH addressing, EW_CFG=0x20800384,
 * ERDMA=0x8000000a). This is the SDP that matches the fp16 matmul's contiguous output — closing the chain / a
 * pure-NPU 2-submit without the CPU atom-8 repack. a=[M][N] contiguous fp16, b=[N] scale, out=[M][N]. Captured
 * at M=8,N=64; notch is verbatim for N=64. ORK_MULC_* env for on-board geometry RE. 0/ok,-1,-2,-3. */
int ork_npu_mul_perchan_f16_contig(ork_npu *c,const ork_f16 *a,const ork_f16 *b,int M,int N,ork_f16 *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>64||N<8||N>8192||(N&7)) return -2;
    const uint16_t *a16=(const uint16_t*)a,*b16=(const uint16_t*)b; uint16_t *o16=(uint16_t*)out;
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,-1), O=orki_bcreate(fd,sz,0x403,-1), B=orki_bcreate(fd,4096,0x403,-1);
    if(!A.cpu||!O.cpu||!B.cpu){ orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&B); return -2; }
    memset(A.cpu,0,sz); memset(O.cpu,0,sz); memset(B.cpu,0,4096);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) ((uint16_t*)A.cpu)[(size_t)m*N+n]=a16[(size_t)m*N+n];  /* CONTIGUOUS in */
    /* PER-ELEMENT operand (vendor DATA_MODE=2): per 8-channel tile t, an [8 chan][M row] block = scale[8t+j]
     * broadcast across rows, channel-major (matches the channel-major output). */
    { int NT0=N/8; uint16_t*bb=B.cpu; int chmaj=getenv("ORK_MULC_OPROW")?0:1;
      ork_f16 one=(ork_f16)1.0f; int ones=getenv("ORK_MULC_ONES")?1:0;
      for(int t=0;t<NT0;t++)for(int j=0;j<8;j++)for(int m=0;m<M;m++)
        bb[(size_t)t*M*8 + (chmaj?(j*M+m):(m*8+j))]=ones?*(uint16_t*)&one:b16[t*8+j]; }
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (XP_SDP=RC_SDPKW, keep-warm-aware; leaves last_dt) */
    /* TILED per the vendor (task13): one SDP task per 8-channel group (CHANNEL=7). Each tile reads its 8
     * contiguous columns of A via NOTCH (LINE skip = N-8 for the row stride) and writes an 8xM atom block to
     * its slot in O. Notch (0x5048/0x504c) is verbatim for N=64; other N derives LINE_NOTCH=N-8. */
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    int NT=N/8; double t0=ork_now_us(); int ok=0;
    for(int t=0; t<NT && ok==0; t++){
        uint32_t rc[REGCMD_MUL_F16_NOTCH_N]; memcpy(rc,REGCMD_MUL_F16_NOTCH,sizeof rc);
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_SDP_5018,(uint32_t)(A.dma+(size_t)t*8*2));   /* input cols 8t..8t+7 */
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_SDP_5038,(uint32_t)(B.dma+(size_t)t*M*8*2)); /* per-element operand block for tile t */
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_DST_BASE_ADDR,(uint32_t)(O.dma+(size_t)t*M*16));  /* tile slot: 8xM atom block */
        /* COMPACT atom-8 output for the 8-channel tile (override the vendor's downstream 0x400 stride) so the
         * readback O+t*M*16 + m*16 + j*2 matches (1 surface, M rows). */
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_DST_SURF_STRIDE,(uint32_t)(M*16));                  /* DST_SURF_STRIDE */
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_DATA_CUBE_WIDTH,(uint32_t)(M-1));
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_DATA_CUBE_HEIGHT,0);
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_DST_N_DIMS,(uint32_t)((7<<16)|7));             /* 8 channels */
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_DST_N2,7);
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_WDMA_SIZE_1,(uint32_t)(M-1));
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_SURFACE_ADD,(uint32_t)(M*16));
        if(getenv("ORK_MULC_HROW")){                                                    /* rows as HEIGHT lines (LINE_NOTCH acts between heights) */
            orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_SDP_500C,0);                             /* RDMA WIDTH = 1 */
            orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_SDP_5010,(uint32_t)(M-1));               /* RDMA HEIGHT = M-1 (rows) */
            orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_DATA_CUBE_WIDTH,0);                             /* DPU WIDTH = 1 */
            orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_DATA_CUBE_HEIGHT,(uint32_t)(M-1));               /* DPU HEIGHT = M-1 */
        } else {
            orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_SDP_500C,(uint32_t)(M-1));               /* WIDTH = M-1 */
        }
        /* vendor task13 is PER-ELEMENT (DATA_MODE=2, compiler broadcast [1,1,N]->[M,N]); switch to PER-CHANNEL
         * (DATA_MODE=0) so an [N] scale operand applies per output channel. */
        uint32_t ewcfg=getenv("ORK_MULC_EW")?strtoul(getenv("ORK_MULC_EW"),0,0):0x20800384; /* vendor: EW mul, DATA_MODE=2 (per-element), fp16 */
        uint32_t erdma=getenv("ORK_MULC_ERDMA")?strtoul(getenv("ORK_MULC_ERDMA"),0,0):0x8000000a; /* vendor: per-element, 2-byte */
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_DPU_EW_CFG,ewcfg);
        orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_SDP_5034,erdma);
        if(N!=64){ orki_setrn(rc,REGCMD_MUL_F16_NOTCH_N,RK_SDP_5048,(uint32_t)((N-8)<<19)); } /* SRC_DMA_CFG.LINE_NOTCH_ADDR[31:19]=N-8 (row-stride skip) */
        memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
        tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
        struct rknpu_submit sub; memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=1;
        sub.task_obj_addr=c->task.obj; sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1;
        sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; sub.timeout=orki_ew_timeout_ms();
        if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; break; }
    }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); if(us)*us=ork_now_us()-t0;
        /* read back: tile t's slot O+t*M*16 is CHANNEL-MAJOR (channel j outer, row m inner): element (j*M+m). */
        for(int t=0;t<NT;t++)for(int m=0;m<M;m++)for(int j=0;j<8;j++)
            o16[(size_t)m*N + t*8 + j]=*(uint16_t*)((char*)O.cpu + (size_t)t*M*16 + (size_t)(j*M + m)*2);
        if(getenv("ORK_MULC_RAW")){ uint16_t*o=(uint16_t*)O.cpu; int nz=0,ne=(int)((size_t)NT*M*16/2); for(int i=0;i<ne;i++) if(o[i])nz++;
            fprintf(stderr,"[mulc_raw] nonzero=%d/%d first16:",nz,ne); for(int i=0;i<16;i++) fprintf(stderr," %.3g",(double)*(ork_f16*)&o[i]); fprintf(stderr,"\n"); }
        if(getenv("ORK_MULC_DUMP")){ ork_f16*a=(ork_f16*)A.cpu,*b=(ork_f16*)B.cpu,*o=(ork_f16*)O.cpu;
            fprintf(stderr,"[dump] scale[0..7]:"); for(int j=0;j<8;j++)fprintf(stderr," %g",(double)b[j]); fprintf(stderr,"\n");
            for(int m=0;m<M;m++){ fprintf(stderr,"[dump] A[%d][0..7]:",m); for(int j=0;j<8;j++)fprintf(stderr," %g",(double)a[(size_t)m*N+j]); fprintf(stderr,"\n"); }
            fprintf(stderr,"[dump] O tile0 [0..%d]:",M*8-1); for(int i=0;i<M*8;i++)fprintf(stderr," %g",(double)o[i]); fprintf(stderr,"\n"); }
    }
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&O); orki_bdestroy(fd,&B);
    return ok;
}

int ork_npu_add_f16(ork_npu *c,const ork_f16 *a,const ork_f16 *b,int M,int N,ork_f16 *out,double *us){
    if(c && c->daemon){ if(us)*us=0; return orkd_add_f16(c->daemon,a,b,M,N,out); }   /* Path B: SDP on the daemon */
    int fd=c->fd, dom=c->dom_active;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)
    const uint16_t *a16=(const uint16_t*)a,*b16=(const uint16_t*)b; uint16_t *o16=(uint16_t*)out;
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    if(orki_ppu_scratch3(c,sz)) return -2;                          /* reuse persistent scratch (no per-op churn) */
    struct buf A=c->ppu_a, B=c->ppu_b, O=c->ppu_o;
    memset(A.cpu,0,sz);memset(B.cpu,0,sz);memset(O.cpu,0,sz);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int p=EWCUBEH(m,n);
        *(uint16_t*)((char*)A.cpu+p)=a16[m*N+n]; *(uint16_t*)((char*)B.cpu+p)=b16[m*N+n]; }
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (XP_SDP=RC_SDPKW, keep-warm-aware; == the old inline reset) */
    uint32_t rc[REGCMD_ADD_F16_N]; memcpy(rc,REGCMD_ADD_F16,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_ADD_F16_N,M,N);
    orki_setrn(rc,REGCMD_ADD_F16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,REGCMD_ADD_F16_N,RK_SDP_5018,(uint32_t)A.dma);
    orki_setrn(rc,REGCMD_ADD_F16_N,RK_SDP_5038,(uint32_t)B.dma);
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    /* NONBLOCK DOORBELL (spine uniformity): ping-pong OFF + full-surface fp16 inf-poison poll — see ork_npu_ewmul_f16. */
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ volatile uint16_t*db=(volatile uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); *db=0x7c00; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); }
    __asm__ volatile("dsb ish":::"memory");
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=(ork_ppflags()&~0x4u)|0x2u;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms(); double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,dom)){ double cap=(double)orki_ew_timeout_ms()*1000.0;
        for(;;){ int done=1; for(int m=0;m<M&&done;m++)for(int n=0;n<N;n++){ volatile uint16_t*db=(volatile uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); __asm__ volatile("dc civac,%0"::"r"(db):"memory"); if(*db==0x7c00u){done=0;break;} } if(done)break; if(ork_now_us()-t0>cap)break; }
        orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) o16[m*N+n]=*(uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(us)*us=t1; }
    /* scratch persists (ppu_scratch3) — not destroyed */
    #undef EWCUBEH
    return ok;
}

int ork_npu_chain_mm_perchan_f16(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B,
                                 const uint16_t *scale,uint16_t *out,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems, dom=c->dom_active;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64||(N&7)) return -2;
    ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_HW);                    /* prime the fp16 pipeline (2-pass cold re-warm) — synth's fp16 matmul writes zeros unwarmed */
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    struct buf W=orki_bcreate(fd,(size_t)K*N*2,0x403,dom), G=orki_bcreate(fd,sz,0x403,dom), O=orki_bcreate(fd,sz,0x403,dom), SB=orki_bcreate(fd,4096,0x403,dom);
    if(!W.cpu||!G.cpu||!O.cpu||!SB.cpu){ orki_bdestroy(fd,&W);orki_bdestroy(fd,&G);orki_bdestroy(fd,&O);orki_bdestroy(fd,&SB); return -1; }
    { int NN=N/16,KT=K/32; uint16_t*bb=W.cpu;                                     /* fp16 weight tile [N/16][K/32][16][32] */
      for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*16+nl)]; }
    memset(G.cpu,0,sz); memset(O.cpu,0,sz); memset(SB.cpu,0,4096);
    { uint16_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; }                   /* fp16 activation raw [M][K] */
    uint32_t r34=0x00000008u;                                                     /* ERDMA per-channel + 2-byte (fp16 SDP) */
    { ork_f16*sb=(ork_f16*)SB.cpu; for(int n=0;n<N;n++) sb[n]=*(const ork_f16*)&scale[n]; }  /* fp16 per-channel scale CONTIGUOUS [N] */
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&G,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&SB,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    static uint32_t mm[REGCMD_N], pc[REGCMD_MUL_F16_CHAIN_N];
    int sched=((K&(K-1))==0 && K>=128 && K<2048);                                /* run_stream_f16's rule; small K => sched=0 (sched=1 miscomputes small K) */
    orki_synth(mm,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)G.dma,sched,CBUF); /* prog0: FP16 matmul -> G (CONTIGUOUS [M][N]) */
    orki_set_f16_out_fp16in(mm,M,N);                                                   /* fp16-out ATOM-8 stage (matches the SDP's orki_set_mul_geom layout) */
    memcpy(pc,REGCMD_MUL_F16_CHAIN,sizeof pc);
    orki_set_mul_geom(pc,REGCMD_MUL_F16_CHAIN_N,M,N);
    orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_SDP_5018,(uint32_t)G.dma);                /* INPUT = matmul OUTPUT (bridge) */
    orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_SDP_5038,(uint32_t)SB.dma);
    orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_SDP_5034,r34);
    double t0=ork_now_us();
    ork_chain_prog progs[2]={ {mm,REGCMD_N,0xd,108,216}, {pc,REGCMD_MUL_F16_CHAIN_N,0x18,69,-1} };
    int crc=ork_npu_chain_progs(c,2,progs,dom);
    if(!crc){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); if(us)*us=ork_now_us()-t0;
        for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=*(uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); }
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&G);orki_bdestroy(fd,&O);orki_bdestroy(fd,&SB);
    #undef EWCUBEH
    return crc;
}


