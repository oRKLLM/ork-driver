/* npu/f16/probe.c — fp16 probes and per-core diagnostics.
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

int ork_f16_npu_probe_mm(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B,float *raw){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%16||N>c->soc->nmax||M<1||M>64) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N*2,0x403,-1); if(!W.cpu) return -2;   /* fp16 weight: 2 B/elem */
    int NN=N/16,KT=K/32; uint16_t*bb=W.cpu;      /* fp16 weight tile [Ntile=16][Ktile=32][16][32] */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*16+nl)];
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)2*M*N*4,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}   /* fp32 out, 2x */
    uint16_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_N];
    orki_f16_synth(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF);   /* ork_f16_fuzz overrides apply inside */
    struct buf extra[2] = {W, O};
    if (orki_validate_regcmd("probe_f16_mm", c, rc, REGCMD_N, NULL, extra, 2)) { orki_bdestroy(fd,&W); orki_bdestroy(fd,&O); return -1; }
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

int ork_f16_npu_probe_mm_f16out(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B,uint16_t *out){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64||(N&7)) return -2;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)
    struct buf W=orki_bcreate(fd,(size_t)K*N*2,0x403,-1); if(!W.cpu) return -2;
    int NN=N/16,KT=K/32; uint16_t*bb=W.cpu;
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*16+nl)];
    size_t osz=(size_t)M*N*2; if(osz<4096)osz=4096;
    struct buf O=orki_bcreate(fd,osz,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;} memset(O.cpu,0,osz);
    /* ORK_F16_ROWPITCH=S: DISCOVERY probe — store the activation rows at pitch S>K (padding between rows) and
     * read them via CNA LINE_STRIDE=S/8. Validates the CNA reads STRIDED/non-contiguous activations directly
     * (the densify lever). Default: contiguous (pitch=K). */
    int rowpitch=getenv("ORK_F16_ROWPITCH")?atoi(getenv("ORK_F16_ROWPITCH")):K;
    uint16_t*ad=c->Af.cpu;
    if(rowpitch!=K){ for(int j=0;j<M*rowpitch;j++)ad[j]=0xdead; for(int m=0;m<M;m++)for(int k=0;k<K;k++)ad[(size_t)m*rowpitch+k]=A[(size_t)m*K+k]; }
    else for(int j=0;j<M*K;j++)ad[j]=A[j];
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_NONE);                  /* prime fp16 pipeline (layer owns the reset; keep-warm-aware) */
    uint32_t rc[REGCMD_N];
    int sched=getenv("ORK_F16_SCHED")?atoi(getenv("ORK_F16_SCHED")):((K&(K-1))==0 && K>=128 && K<2048);  /* run_stream_f16 rule; small K => 0 */
    orki_f16_synth(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,sched,CBUF);
    if(rowpitch!=K) orki_setrn(rc,REGCMD_N,RK_CNA_DMA_CON1,rowpitch/8);        /* CNA LINE_STRIDE = pitch/8 surfaces (strided activation) */
    if(!getenv("ORK_F16_FP32OUT")) orki_f16_set_out_fp16in(rc,M,N);        /* vendor fp16-out stage (atom-8); skip => synth's native fp32-out (compute sanity) */
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task*t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0xd; t->int_mask=0x300; t->int_clear=0x1ffff;
      t->regcfg_amount=108; t->regcmd_addr=(uint32_t)c->regcmd.dma; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); }
    uint32_t to_ms=3000; { const char*e=getenv("ORK_EW_TIMEOUT"); if(e){ unsigned v=(unsigned)strtoul(e,0,0); if(v)to_ms=v; } }
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    sub.timeout=to_ms;
    int ok=-1;
    for(int rep=0;rep<2;rep++){ if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; continue; } orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; }  /* fp16 cold 2-pass re-warm */
    if(ok==0){
        int ewc=getenv("ORK_F16_ATOM8")?1:0;                          /* readback matches the output layout: default CONTIGUOUS; ORK_F16_ATOM8 => atom-8 */
        for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=*(uint16_t*)((char*)O.cpu+(ewc?EWCUBEH(m,n):((size_t)(m*N+n)*2)));
        if(getenv("ORK_F16_RAWDUMP")){ uint16_t*o=(uint16_t*)O.cpu; int nz=0,ne=osz/2; int first[16],nf=0;
            for(int i=0;i<ne;i++) if(o[i]){ nz++; if(nf<16){ first[nf++]=i; } }
            fprintf(stderr,"[f16raw] osz=%zu nonzero=%d/%d  first offsets(elem):",osz,nz,ne);
            for(int i=0;i<nf;i++) fprintf(stderr," %d=%.3g",first[i],(double)*(ork_f16*)&o[first[i]]);
            fprintf(stderr,"\n"); }
        ok=0; }
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);
    #undef EWCUBEH
    return ok;
}

int ork_f16_npu_probe_stridedA(ork_npu *c,int M,int K,int N,const uint16_t *A,int apitch,const uint16_t *B,uint16_t *out){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64||(N&7)||apitch<K||(apitch&7)) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N*2,0x403,-1); if(!W.cpu) return -2;
    { int NN=N/16,KT=K/32; uint16_t*bb=W.cpu;
      for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*16+nl)]; }
    size_t asz=(size_t)M*apitch*2; if(asz<4096)asz=4096; size_t osz=(size_t)M*N*2; if(osz<4096)osz=4096;
    struct buf Adev=orki_bcreate(fd,asz,0x403,-1), O=orki_bcreate(fd,osz,0x403,-1);
    if(!Adev.cpu||!O.cpu){ orki_bdestroy(fd,&W);orki_bdestroy(fd,&Adev);orki_bdestroy(fd,&O); return -2; }
    { uint16_t*ad=Adev.cpu; for(size_t i=0;i<asz/2;i++)ad[i]=0xdead;                 /* junk padding between rows */
      for(int m=0;m<M;m++)for(int k=0;k<K;k++) ad[(size_t)m*apitch+k]=A[(size_t)m*K+k]; }  /* A row @ pitch (as the KV-view sits in the DMA buffer) */
    memset(O.cpu,0,osz);
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&Adev,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_NONE);
    uint32_t rc[REGCMD_N];
    int sched=((K&(K-1))==0 && K>=128 && K<2048);
    orki_f16_synth(rc,M,K,N,(uint32_t)Adev.dma,(uint32_t)W.dma,(uint32_t)O.dma,sched,CBUF);     /* activation base = the DMA buffer (ZERO-COPY, no c->Af) */
    orki_setrn(rc,REGCMD_N,RK_CNA_DMA_CON1,apitch/8);                                          /* CNA LINE_STRIDE = apitch/8 surfaces (read the strided view) */
    orki_f16_set_out_fp16in(rc,M,N);
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task*t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0xd; t->int_mask=0x300; t->int_clear=0x1ffff;
      t->regcfg_amount=108; t->regcmd_addr=(uint32_t)c->regcmd.dma; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); }
    uint32_t to_ms=3000; { const char*e=getenv("ORK_EW_TIMEOUT"); if(e){ unsigned v=(unsigned)strtoul(e,0,0); if(v)to_ms=v; } }
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};sub.timeout=to_ms;
    int ok=-1;
    for(int rep=0;rep<2;rep++){ if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; continue; } orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; }
    if(ok==0) for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=((uint16_t*)O.cpu)[(size_t)m*N+n];  /* contiguous fp16 out */
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&Adev);orki_bdestroy(fd,&O);
    return ok;
}

int ork_f16_npu_gap_probe(ork_npu *c, int M, int Kp, int N, int use_gap, long *nz0, long *nz1, double *us) {
    int fd = c->fd, CBUF = c->soc->cbuf_elems, dom = c->dom_active;
    if (!ork_ppu_fuse_enabled(c)) return -3;
    if (Kp % 32 || N % 32 || N > c->soc->nmax || M < 1 || M > 64 || (N & 7)) return -2;
    ork_npu_enter(c, DT_F16, XP_STREAM_F16, OCK_HW);
    size_t gsz = (size_t)M * N * 2; if (gsz < 4096) gsz = 4096;
    struct buf W0 = orki_bcreate(fd,(size_t)Kp*N*2,0x403,dom), W1 = orki_bcreate(fd,(size_t)Kp*N*2,0x403,dom);
    struct buf G0 = orki_bcreate(fd,gsz,0x403,dom), G1 = orki_bcreate(fd,gsz,0x403,dom);
    struct buf GI = orki_bcreate(fd,gsz,0x403,dom), GO = orki_bcreate(fd,gsz,0x403,dom), SB = orki_bcreate(fd,4096,0x403,dom);
    if (!W0.cpu||!W1.cpu||!G0.cpu||!G1.cpu||!GI.cpu||!GO.cpu||!SB.cpu) {
        orki_bdestroy(fd,&W0);orki_bdestroy(fd,&W1);orki_bdestroy(fd,&G0);orki_bdestroy(fd,&G1);orki_bdestroy(fd,&GI);orki_bdestroy(fd,&GO);orki_bdestroy(fd,&SB); return -1; }
    { int NN=N/16, KT=Kp/32; uint16_t *b0=W0.cpu, *b1=W1.cpu;   /* fp16 weight tile [N/16][Kp/32][16][32]; W0 all 1.0, W1 all 1.0 shifted (distinct) */
      for (int nt=0;nt<NN;nt++) for (int kt=0;kt<KT;kt++) for (int nl=0;nl<16;nl++) for (int kk=0;kk<32;kk++) {
          size_t o=(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk; b0[o]=0x3c00; b1[o]=0x3c00; } }   /* 0x3c00 = fp16 1.0 */
    { uint16_t *ad=c->Af.cpu; for (int j=0;j<M*Kp;j++) ad[j]=0x3c00; }   /* A = 1.0 */
    memset(G0.cpu,0,gsz); memset(G1.cpu,0,gsz); memset(GI.cpu,0,gsz); memset(GO.cpu,0,gsz); memset(SB.cpu,0,4096);
    { uint16_t *sb=SB.cpu; for (int n=0;n<N;n++) sb[n]=0x3c00; }   /* identity per-channel scale (fp16 1.0) */
    orki_bsync(fd,&W0,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&W1,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&G0,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&G1,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&GI,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&GO,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&SB,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    static uint32_t mm0[REGCMD_N], mm1[REGCMD_N], pc[REGCMD_MUL_F16_CHAIN_N];
    int sched = ((Kp&(Kp-1))==0 && Kp>=128 && Kp<2048);
    orki_f16_synth(mm0,M,Kp,N,(uint32_t)c->Af.dma,(uint32_t)W0.dma,(uint32_t)G0.dma,sched,CBUF); orki_f16_set_out_fp16in(mm0,M,N);
    orki_f16_synth(mm1,M,Kp,N,(uint32_t)c->Af.dma,(uint32_t)W1.dma,(uint32_t)G1.dma,sched,CBUF); orki_f16_set_out_fp16in(mm1,M,N);
    memcpy(pc,REGCMD_MUL_F16_CHAIN,sizeof pc); orki_set_mul_geom(pc,REGCMD_MUL_F16_CHAIN_N,M,N);
    orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_DPU_DST_BASE_ADDR,(uint32_t)GO.dma);
    orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_SDP_5018,(uint32_t)GI.dma);   /* gap INPUT = dummy scratch (NOT the matmul output) */
    orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_SDP_5038,(uint32_t)SB.dma);
    orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_SDP_5034,0x00000008);
    double t0 = ork_now_us(); int crc;
    if (use_gap) { ork_chain_prog p[3] = { {mm0,REGCMD_N,0xd,108,216}, {pc,REGCMD_MUL_F16_CHAIN_N,0x18,69,138}, {mm1,REGCMD_N,0xd,108,-1} };
        crc = ork_npu_chain_progs(c,3,p,dom); }
    else { ork_chain_prog p[2] = { {mm0,REGCMD_N,0xd,108,216}, {mm1,REGCMD_N,0xd,108,-1} };
        crc = ork_npu_chain_progs(c,2,p,dom); }
    if (us) *us = ork_now_us()-t0;
    long z0=0, z1=0;
    if (!crc) { orki_bsync(fd,&G0,RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd,&G1,RKNPU_MEM_SYNC_FROM_DEVICE);
        uint16_t *g0=G0.cpu, *g1=G1.cpu; for (int e=0;e<M*N;e++){ if(g0[e])z0++; if(g1[e])z1++; } }
    if (nz0) *nz0=z0; if (nz1) *nz1=z1;
    orki_bdestroy(fd,&W0);orki_bdestroy(fd,&W1);orki_bdestroy(fd,&G0);orki_bdestroy(fd,&G1);orki_bdestroy(fd,&GI);orki_bdestroy(fd,&GO);orki_bdestroy(fd,&SB);
    return crc;
}

int ork_f16_npu_percore_probe(ork_npu*c,int M,int K,int N,const ork_f16*A,const ork_f16*B,float*Cout,double*us,int mode){
    if(!c) return -3;
    int CBUF=c->soc->cbuf_elems;
    int cores=c->soc->cores; if(cores>ORK_MAXCORE) cores=ORK_MAXCORE; if(cores<1) cores=1;
    if(M<1||M>64||K%32||N%16||N>c->soc->nmax) return -2;
    if(N%(cores*16)) return -2;
    int Ncol=N/cores;
    const char*card=getenv("ORK_NPU_CARD"); if(!card)card=c->soc->card;
    int sched=((K&(K-1))==0 && K>=128 && K<2048);
    int cfd[ORK_MAXCORE]; for(int i=0;i<ORK_MAXCORE;i++) cfd[i]=-1;
    struct buf wbuf[ORK_MAXCORE]={{0}}, wimp[ORK_MAXCORE]={{0}}, abuf[ORK_MAXCORE]={{0}},
               cob[ORK_MAXCORE]={{0}}, rcb[ORK_MAXCORE]={{0}}, tkb[ORK_MAXCORE]={{0}};
    int shared_dbuf=-1, ret=-1;
    for(int i=0;i<cores;i++){ cfd[i]=open(card,O_RDWR); if(cfd[i]<0) goto done; orki_act_opt(cfd[i],RKNPU_POWER_ON,0); }
    if(mode==1){   /* ONE shared full-N weight imported into every fd; tile the FULL B once via the primary map */
        size_t wsz=(size_t)K*N*2;
        wimp[0]=orki_bimport(cfd[0],wsz,0); if(!wimp[0].cpu) goto done; shared_dbuf=wimp[0].heap_fd;
        for(int i=1;i<cores;i++){ wimp[i]=orki_bimport_fd(cfd[i],shared_dbuf,wsz,0); if(!wimp[i].cpu) goto done; wimp[i].heap_fd=0; }
        int NNf=N/16, KT=K/32; ork_f16*bb=wimp[0].cpu;                          /* full tile [N/16][K/32][16][32] */
        for(int nt=0;nt<NNf;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
            bb[(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*16+nl)];
        for(int i=0;i<cores;i++) orki_bsync(cfd[i],&wimp[i],RKNPU_MEM_SYNC_TO_DEVICE);   /* clean each fd's mapping of the import */
    }
    for(int i=0;i<cores;i++){
        int n0=i*Ncol; uint32_t aB;
        if(mode==1){ aB=(uint32_t)(wimp[i].dma + (size_t)n0*K*2); }   /* col-tile slice: n0%16==0 => byte off = n0*K*2 (n0/16 whole tiles, each K/32*16*32*2 = K*32*2 B) */
        else {
            wbuf[i]=orki_bcreate(cfd[i],(size_t)K*Ncol*2,0x403,0); if(!wbuf[i].cpu) goto done;
            int NN=Ncol/16, KT=K/32; ork_f16*bb=wbuf[i].cpu;                   /* per-core tile [Ncol/16][K/32][16][32] of B's cols [n0,n0+Ncol) */
            for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
                bb[(size_t)nt*KT*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(n0+nt*16+nl)];
            orki_bsync(cfd[i],&wbuf[i],RKNPU_MEM_SYNC_TO_DEVICE);
            aB=(uint32_t)wbuf[i].dma;
        }
        abuf[i]=orki_bcreate(cfd[i],(size_t)M*K*2,0x403,0); if(!abuf[i].cpu) goto done;   /* full [M][K] activation (every core needs all of A) */
        { ork_f16*ad=abuf[i].cpu; for(int j=0;j<M*K;j++) ad[j]=A[j]; }
        orki_bsync(cfd[i],&abuf[i],RKNPU_MEM_SYNC_TO_DEVICE);
        size_t csz=(size_t)M*Ncol*4; if(csz<4096) csz=4096;              /* fp16-out uses M*Ncol*2; over-alloc to 4B/elem, harmless */
        cob[i]=orki_bcreate(cfd[i],csz,0x403,0); if(!cob[i].cpu) goto done;
        memset(cob[i].cpu,0,csz); orki_bsync(cfd[i],&cob[i],RKNPU_MEM_SYNC_TO_DEVICE);   /* seed */
        rcb[i]=orki_bcreate(cfd[i],(size_t)REGCMD_N*4,0x403,0); if(!rcb[i].cpu) goto done;
        tkb[i]=orki_bcreate(cfd[i],4096,0x40b,0); if(!tkb[i].cpu) goto done;
        uint32_t rc[REGCMD_N];
        orki_f16_synth(rc,M,K,Ncol,(uint32_t)abuf[i].dma,aB,(uint32_t)cob[i].dma,sched,CBUF);
        orki_f16_set_out_fp16in(rc,M,Ncol);                                   /* fp16-out, contiguous [M][Ncol] (no ORK_F16_ATOM8) */
        memcpy(rcb[i].cpu,rc,(size_t)REGCMD_N*4);
        orki_bsync(cfd[i],&rcb[i],RKNPU_MEM_SYNC_TO_DEVICE);
        struct rknpu_task*t=(struct rknpu_task*)tkb[i].cpu; memset(t,0,sizeof *t);
        t->enable_mask=0xd; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=108; t->regcmd_addr=rcb[i].dma;
        orki_bsync(cfd[i],&tkb[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    }
    { double t0=ork_now_us();                                            /* concurrent per-core-fd submit: one blocking thread/core */
      pthread_t th[ORK_MAXCORE]; struct ork_pcfd_arg ar[ORK_MAXCORE]; int made=0;
      for(int i=0;i<cores;i++){ ar[i]=(struct ork_pcfd_arg){cfd[i],i,&tkb[i],0};
          if(pthread_create(&th[i],NULL,ork_pcfd_thread,&ar[i])!=0) break; made++; }
      for(int i=0;i<made;i++) pthread_join(th[i],NULL);
      if(us) *us=ork_now_us()-t0; }
    for(int i=0;i<cores;i++){ int n0=i*Ncol;                             /* read fp16-out back, de-column into Cout[M,N] fp32 */
        orki_bsync(cfd[i],&cob[i],RKNPU_MEM_SYNC_FROM_DEVICE);
        const ork_f16*cf=(const ork_f16*)cob[i].cpu;
        for(int m=0;m<M;m++) for(int n=0;n<Ncol;n++) Cout[(size_t)m*N+n0+n]=(float)cf[(size_t)m*Ncol+n];
    }
    ret=0;
done:
    for(int i=0;i<cores;i++){
        if(cfd[i]<0) continue;
        orki_bdestroy(cfd[i],&tkb[i]); orki_bdestroy(cfd[i],&rcb[i]); orki_bdestroy(cfd[i],&cob[i]); orki_bdestroy(cfd[i],&abuf[i]);
        if(mode==1) orki_bdestroy(cfd[i],&wimp[i]); else orki_bdestroy(cfd[i],&wbuf[i]);   /* wimp[0] closes the shared dbuf once (heap_fd zeroed on i>0) */
        close(cfd[i]);
    }
    return ret;
}

int ork_f16_npu_probe_slice(ork_npu *c,int Kfull,int N,int Kp,int nov,
                            const uint32_t *ovr_reg,const uint32_t *ovr_val,
                            const f16 *A,const f16 *B,float *C){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(Kfull%32||Kp%32||N%16||N>c->soc->nmax||Kp>Kfull) return -2;
    struct buf W=orki_bcreate(fd,(size_t)Kfull*N*2,0x403,-1); if(!W.cpu) return -2;
    int NN=N/16,KTf=Kfull/32; f16*bb=W.cpu;     /* full-K fp16 layout [Ntile][KTfull][16][32] */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KTf;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KTf*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*16+nl)];
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)N*4,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}
    f16*ad=c->Af.cpu; for(int j=0;j<Kp;j++)ad[j]=A[j]; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t rc[REGCMD_N];
    orki_f16_synth(rc,1,Kp,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF);
    orki_setrn(rc,REGCMD_N,RK_CNA_CBUF_CON0,0xb1);
    for(int i=0;i<nov && i<4;i++) orki_setr(rc,REGCMD_N,0x201,ovr_reg[i],ovr_val[i]);
    struct buf extra[2] = {W, O};
    if (orki_validate_regcmd("probe_slice_f16", c, rc, REGCMD_N, NULL, extra, 2)) { orki_bdestroy(fd,&W); orki_bdestroy(fd,&O); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1;
    for(int rep=0;rep<2;rep++){ sub.timeout=orki_mm_timeout_ms();
        if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; continue; }
        orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(C,O.cpu,(size_t)N*4); ok=0; }
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);
    return ok;
}

int ork_f16_ssd_probe_rawmm(ork_npu*c,int M,int K,int N,const f16*A,const f16*B,float*C){
    if(!c||M<1||K<1||N<1||K%32||N%16) return -2;
    int fd=c->fd,CBUF=c->soc->cbuf_elems,dom=-1,ret=0;
    struct buf Ab=orki_bcreate(fd,(size_t)M*K*2,0x403,dom),Bb=orki_bcreate(fd,(size_t)K*N*2,0x403,dom),Cb=orki_bcreate(fd,(size_t)M*N*4,0x403,dom);
    if(!Ab.cpu||!Bb.cpu||!Cb.cpu){ ret=-3; goto done; }
    memcpy(Ab.cpu,A,(size_t)M*K*2); memcpy(Bb.cpu,B,(size_t)K*N*2); memset(Cb.cpu,0,(size_t)M*N*4);
    orki_bsync(fd,&Ab,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Bb,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Cb,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0); c->warmed=0; c->last_dt=DT_F16;
    { uint32_t *rc=calloc(REGCMD_I8_N,4); if(!rc){ ret=-3; goto done; }
      orki_f16_synth(rc,M,K,N,(uint32_t)Ab.dma,(uint32_t)Bb.dma,(uint32_t)Cb.dma,1,CBUF);
      ork_chain_prog p={rc,REGCMD_I8_N,0xd,108,216};
      ret=ork_npu_chain_progs(c,1,&p,dom); free(rc); }
    if(ret) goto done;
    orki_bsync(fd,&Cb,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(C,Cb.cpu,(size_t)M*N*4);
done:
    orki_bdestroy(fd,&Ab);orki_bdestroy(fd,&Bb);orki_bdestroy(fd,&Cb);
    return ret;
}

int ork_f16_ssd_probe_fusedmm(ork_npu*c,int M,int K,int N,const f16*A,const f16*B,float*C){
    if(!c||M<1||K<1||N<1||K%32||N%16) return -2;
    ork_w *w=ork_f16_mm_pack(c,K,N,B); if(!w) return -3;
    if(w->Sk!=1||w->Sn!=1){ ork_mm_free(c,w); return -2; }   /* probe: single tile only */
    int fd=c->fd,CBUF=c->soc->cbuf_elems,dom=w->domain,ret=0;
    struct buf Ab=orki_bcreate(fd,(size_t)M*K*2,0x403,dom), Cb=orki_bcreate(fd,(size_t)M*N*4,0x403,dom);
    if(!Ab.cpu||!Cb.cpu){ ret=-3; goto done2; }
    memcpy(Ab.cpu,A,(size_t)M*K*2); memset(Cb.cpu,0,(size_t)M*N*4);
    orki_bsync(fd,&Ab,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Cb,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0); c->warmed=0; c->last_dt=DT_F16;
    { uint32_t *rc=calloc(REGCMD_I8_N,4); if(!rc){ ret=-3; goto done2; }
      orki_f16_synth(rc,M,K,N,(uint32_t)Ab.dma,(uint32_t)w->Bb[0].dma,(uint32_t)Cb.dma,1,CBUF);
      ork_chain_prog p={rc,REGCMD_I8_N,0xd,108,216};
      ret=ork_npu_chain_progs(c,1,&p,dom); free(rc); }
    if(ret) goto done2;
    orki_bsync(fd,&Cb,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(C,Cb.cpu,(size_t)M*N*4);
done2:
    orki_bdestroy(fd,&Ab);orki_bdestroy(fd,&Cb); ork_mm_free(c,w);
    return ret;
}
