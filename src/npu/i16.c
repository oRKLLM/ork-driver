/* npu/i16.c — the int16 datapath.
 *
 * int16 regcmd output stage, the int16 SDP activations (silu/gelu/rsqrt/exp via the LUT op), int16
 * elementwise and per-channel multiply, the int16 HW-chained matmul+activation graphs, and the int16
 * probes/replays. int16 is the accuracy tier between int8 and fp16 for the SDP ops -- notably the
 * on-NPU SiLU that works in-chain where fp16 does not.
 *
 * Lifted verbatim from npu.c and the i8 modules by the precision split (MODULARIZE_PLAN.md round 1). */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include "ork_regs.h"
#include "regcmd_array_4x32x16.h"
#include "regcmd_i8.h"
#include "regcmd_silu.h"
#include "regcmd_ewmul.h"
#include "npu/internal.h"
#include "npu/core.h"

static void orki_synth_i16(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC,int sched,int cbuf){
    orki_synth(rc,mc,K,N,aA,aB,aC,sched,cbuf);                 /* fp16 2-byte geometry base */
    uint32_t con1=0x20000090u; const char*e=getenv("ORK_I16_CON1"); if(e) con1=(uint32_t)strtoul(e,NULL,0);
    orki_setrn(rc,REGCMD_N,RK_CNA_CONV_CON1,con1);             /* FP16(2)->INT16(1) precision */
}

void orki_set_i16_out(uint32_t*rc,int N,int stride,int mult,int shift){
    int s=stride>0?stride:N;
    /* SOLVED (wedge + layout) 2026-07-21 via i16out_fix_probe layout sweep: the int16 output stage now writes
     * COMPACT CONTIGUOUS int16 (row-major m*N+n, M*N*2 bytes, 128/128 correct, no stall, no buffer overflow).
     *  - 0x4010=0x20000000: OUT_PRECISION=int16 (bits[31:29]=1). The old 0x0000 (=int8 precision) while emitting
     *    2-byte elements was the WDMA terminal-count/precision MISMATCH that STALLED the op (the standalone wedge).
     *  - 0x4038=(s/8-1)|(N/8-1): the ORIGINAL comment's INTENDED "N/8 (2x denser than int32)" — the old code had
     *    a typo N/16 in the low half (-> wrong width -> stall). (N/4 = the fp16 value = a wasteful 2N-wide surface
     *    that overflows M*N*2 buffers; N/8 is the compact one.)
     *  - 0x4050=0x36e, 0x40c0=0x40: the 2-byte surface (0x248 was wrong/int8-ish -> partial/stall).
     * NOTE: output is LINEAR (m*N+n), NOT the EWCUBEH cube the standalone int16 SiLU stages its host input into;
     * feeding it to the SiLU in a chain needs the SiLU input geom set to read linear (or the int8-bridge template). */
    unsigned r10=getenv("ORK_I16OUT_4010")?strtoul(getenv("ORK_I16OUT_4010"),0,0):0x20000000;
    unsigned r50=getenv("ORK_I16OUT_4050")?strtoul(getenv("ORK_I16OUT_4050"),0,0):0x0000036e;
    unsigned rc0=getenv("ORK_I16OUT_40c0")?strtoul(getenv("ORK_I16OUT_40c0"),0,0):0x0040;
    unsigned r38=getenv("ORK_I16OUT_4038")?strtoul(getenv("ORK_I16OUT_4038"),0,0):(unsigned)((((s/8)-1)<<16)|((N/8)-1));
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_PRECISION,r10);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_DATA_CUBE_NOTCH,r38);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_BS_OW_CFG,r50);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_SURFACE_ADD,rc0);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_SCALE,mult);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_SHIFT,shift);
}

int ork_npu_probe_i16_out(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,
                          int mult,int shift,int16_t *C,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    int NN=N/32,KT=K/32; int8_t*bb=W.cpu;     /* int8 tile layout [Ntile][Ktile][32][32], full K */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)];
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    size_t obytes=getenv("ORK_I16_OBYTES")?strtoul(getenv("ORK_I16_OBYTES"),0,0):(size_t)M*N*2;  /* enlarge to capture a strided layout */
    if(obytes<(size_t)M*N*2) obytes=(size_t)M*N*2;
    struct buf O=orki_bcreate(fd,obytes,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}  /* int16 output */
    memset(O.cpu,0,obytes); orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I8_N];
    orki_synth_i8(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    if(getenv("ORK_MM_F16OUT")) orki_set_f16_out(rc,N,0);          /* SHIM test: int8 matmul -> fp16 OUT_CVT (2-byte) */
    else if(getenv("ORK_MM_I32OUT")) { /* CONTROL: skip set_i16_out -> synth_i8's default int32 output (works standalone) */ }
    else                        orki_set_i16_out(rc,N,0,mult,shift); /* rewrite output stage: int32 -> int16 requantize */
    /* TOGGLE SWEEP: restore individual output-stage regs to their int32 (completing) values to isolate the
     * WDMA terminal-count stall. Each ORK_MM_R<reg>=<hex> overrides one reg AFTER set_i16_out. */
    { const char*e;
      if((e=getenv("ORK_MM_R4010"))) orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_PRECISION,(uint32_t)strtoul(e,0,0));
      if((e=getenv("ORK_MM_R4038"))) orki_setrn(rc,REGCMD_I8_N,RK_DPU_DATA_CUBE_NOTCH,(uint32_t)strtoul(e,0,0));
      if((e=getenv("ORK_MM_R4050"))) orki_setrn(rc,REGCMD_I8_N,RK_DPU_BS_OW_CFG,(uint32_t)strtoul(e,0,0));
      if((e=getenv("ORK_MM_R4084"))) orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)strtoul(e,0,0));
      if((e=getenv("ORK_MM_R4088"))) orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)strtoul(e,0,0));
      if((e=getenv("ORK_MM_R40c0"))) orki_setrn(rc,REGCMD_I8_N,RK_DPU_SURFACE_ADD,(uint32_t)strtoul(e,0,0)); }
    if(getenv("ORK_MM_DUMPRC")){ const char*tag=getenv("ORK_MM_I32OUT")?"I32":"I16"; /* dump the assembled 0x40xx output stage for diffing */
        for(int k=0;k+1<REGCMD_I8_N;k+=2){ uint32_t r=rc[k]&0xffff, v=((rc[k]>>16)&0xffff)|((rc[k+1]&0xffff)<<16);
            if(r>=0x4000 && r<0x4100) fprintf(stderr,"[%s] 0x%04x = 0x%08x\n",tag,r,v); } }
    struct buf extra[2] = {W, O};
    if (orki_validate_regcmd("probe_i16_out", c, rc, REGCMD_I8_N, NULL, extra, 2)) { orki_bdestroy(fd,&W); orki_bdestroy(fd,&O); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0;
    for(int rep=0;rep<3;rep++){ sub.timeout=orki_mm_timeout_ms();
        double t0=ork_now_us();
        if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; break; }
        orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    if(ok==0){ memcpy(C,O.cpu,(size_t)M*N*2); if(us)*us=t1;
        if(getenv("ORK_MM_DUMPOUT")){ /* LAYOUT MAP: caller made values distinct; print int16-slot -> value for every nonzero slot in the (enlarged) O */
            const int16_t*oc=O.cpu; long ns=(long)(obytes/2); int shown=0; long firstrow_end=-1, secondrow_start=-1; int prev=-1;
            fprintf(stderr,"[i16map] obytes=%zu (%ld slots), M=%d N=%d — nonzero slots:\n",obytes,ns,M,N);
            for(long i=0;i<ns;i++){ int v=oc[i]; if(v!=0){ if(shown<64) fprintf(stderr,"  [%ld]=%d",i,v);
                if(prev>=0 && v<prev && secondrow_start<0){ firstrow_end=i-1; secondrow_start=i; } prev=v; shown++; if(shown%8==0&&shown<=64)fprintf(stderr,"\n"); } }
            fprintf(stderr,"\n[i16map] total nonzero=%d  (row-wrap at slot ~%ld => row byte-stride ~%ld)\n",shown,secondrow_start,secondrow_start*2); } }
    else if(getenv("ORK_MM_DUMPOUT")){ /* partial-write signature: how far did the WDMA get before the stall? */
        orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); int16_t*oc=O.cpu; long last=-1; int tot=0,rows=0;
        for(long i=0;i<(long)M*N;i++) if(oc[i]){ tot++; if(i>last)last=i; }
        for(int m=0;m<M;m++){ int rnz=0; for(int n=0;n<N;n++) if(oc[(size_t)m*N+n])rnz++; if(rnz)rows++; }
        fprintf(stderr,"[dumpout] M=%d N=%d: nonzero=%d/%d  rows-with-data=%d/%d  last-nz-elem=%ld (row %ld/%d)\n",
                M,N,tot,M*N,rows,M,last,last<0?-1:last/N,M); }
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);
    return ok;
}

int ork_npu_mul_perchan_i16(ork_npu *c,const int16_t *a,const int16_t *b,int M,int N,int mult,int shift,int16_t *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;
    if(mult<0||mult>0x7fff||shift<0||shift>31) return -2;
    #define PC16(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)         /* 2-byte atom=8 cube */
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,-1), O=orki_bcreate(fd,sz,0x403,-1), B=orki_bcreate(fd,sz,0x403,-1);
    if(!A.cpu||!O.cpu||!B.cpu){ orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&B); return -2; }
    memset(A.cpu,0,sz); memset(O.cpu,0,sz); memset(B.cpu,0,sz);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) *(int16_t*)((char*)A.cpu+PC16(m,n))=a[(size_t)m*N+n];
    for(int n=0;n<N;n++) ((int16_t*)B.cpu)[n]=b[n];                  /* per-channel vector CONTIGUOUS [N] int16 */
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (XP_SDP=RC_SDPKW, keep-warm-aware; == the old inline reset) */
    uint32_t rc[REGCMD_MUL_I16_N]; memcpy(rc,REGCMD_MUL_I16,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_MUL_I16_N,M,N);
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_5018,(uint32_t)A.dma);
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_5038,(uint32_t)B.dma);
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_5034,0x00000008);            /* ERDMA_DATA_MODE=0 (per-channel) + DATA_SIZE=TWO_BYTE */
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)mult); orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)shift);
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_OUT_CVT_OFFSET,0); orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_BS_ALU_CFG,0); orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_EW_CVT_OFFSET,0);
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub; memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=1;
    sub.task_obj_addr=c->task.obj; sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1;
    sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; sub.timeout=orki_ew_timeout_ms();
    int ok=-1; double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; if(us)*us=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=*(int16_t*)((char*)O.cpu+PC16(m,n)); }
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&O); orki_bdestroy(fd,&B);
    #undef PC16
    return ok;
}

int ork_npu_add_i16(ork_npu *c,const int16_t *a,const int16_t *b,int M,int N,
                    double a_scale,double b_scale,double out_scale,int16_t *out,double *us){
    if(c && c->daemon){ if(us)*us=0; return orkd_add_i16(c->daemon,a,b,M,N,a_scale,b_scale,out_scale,out); }   /* Path B: SDP on the daemon */
    int fd=c->fd, dom=c->dom_active;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)||out_scale<=0) return -2;
    double ca=a_scale/out_scale, cb=b_scale/out_scale, cmax=(ca>cb?ca:cb); if(cmax<=0) return -2;
    int S=14; while(S>0 && cmax*(double)(1u<<S) > 0x4000) S--;
    while(S<30 && cmax*(double)(1u<<(S+1)) <= 0x4000) S++;
    long ma=lround(ca*(double)(1u<<S)), mb=lround(cb*(double)(1u<<S));
    if(ma>0x4000)ma=0x4000; if(mb>0x4000)mb=0x4000;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,dom); if(!A.cpu)return -2;
    struct buf B=orki_bcreate(fd,sz,0x403,dom); if(!B.cpu){orki_bdestroy(fd,&A);return -2;}
    struct buf O=orki_bcreate(fd,sz,0x403,dom); if(!O.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);return -2;}
    memset(A.cpu,0,sz);memset(B.cpu,0,sz);memset(O.cpu,0,sz);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int p=EWCUBEH(m,n);
        *(int16_t*)((char*)A.cpu+p)=a[m*N+n]; *(int16_t*)((char*)B.cpu+p)=b[m*N+n]; }
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (XP_SDP=RC_SDPKW, keep-warm-aware; == the old inline reset) */
    uint32_t rc[REGCMD_ADD_I16_N]; memcpy(rc,REGCMD_ADD_I16,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_ADD_I16_N,M,N);
    orki_setrn(rc,REGCMD_ADD_I16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,REGCMD_ADD_I16_N,RK_SDP_5018,(uint32_t)A.dma);
    orki_setrn(rc,REGCMD_ADD_I16_N,RK_SDP_5038,(uint32_t)B.dma);
    /* EXPERIMENTAL: int16 add is NOT bit-exact over the signed range. Isolated: the ERDMA/X2 operand (0x5038, via
     * 0x4078) is exact for BOTH signs; the SRDMA/X1 operand (0x5018, via 0x4084) is exact for POSITIVE but HALVES
     * NEGATIVES (int8's X1 didn't — a int16-specific X1 sign/shift behavior). 0x4048 and the shift couple into it,
     * and configs that fix negatives double-count or scale x2. A clean signed fix needs a full X1/X2 sub-module
     * scale decode (multi-reg, sign-dependent) — not yet cracked. Env overrides (ORK_ADD16_R48/84/88/78) for RE. */
    uint32_t r48=0x40000000,r84=(uint32_t)ma,r88=(uint32_t)(S+14),r78=(uint32_t)mb; const char*e;
    if((e=getenv("ORK_ADD16_R48")))r48=(uint32_t)strtoul(e,0,16);
    if((e=getenv("ORK_ADD16_R84")))r84=(uint32_t)strtoul(e,0,16);
    if((e=getenv("ORK_ADD16_R88")))r88=(uint32_t)strtoul(e,0,16);
    if((e=getenv("ORK_ADD16_R78")))r78=(uint32_t)strtoul(e,0,16);
    orki_setrn(rc,REGCMD_ADD_I16_N,RK_DPU_BS_MUL_CFG,r48);
    orki_setrn(rc,REGCMD_ADD_I16_N,RK_DPU_OUT_CVT_SCALE,r84);
    orki_setrn(rc,REGCMD_ADD_I16_N,RK_DPU_OUT_CVT_SHIFT,r88);
    orki_setrn(rc,REGCMD_ADD_I16_N,RK_DPU_EW_CVT_SCALE,r78);
    orki_setrn(rc,REGCMD_ADD_I16_N,RK_DPU_BS_ALU_CFG,0); orki_setrn(rc,REGCMD_ADD_I16_N,RK_DPU_EW_CVT_OFFSET,0); orki_setrn(rc,REGCMD_ADD_I16_N,RK_DPU_OUT_CVT_OFFSET,0);
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms(); double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,dom)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[m*N+n]=*(int16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(us)*us=t1; }
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&O);
    #undef EWCUBEH
    return ok;
}

int ork_npu_replay_lut_i16(ork_npu *c,const uint32_t *regcmd,int rn,const int16_t *lut,int nlut,
                           const int16_t *in,int M,int N,int16_t *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)||rn>REGCMD_SILU_STD_I16_N) return -2;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)
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
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) *(int16_t*)((char*)A.cpu+EWCUBEH(m,n))=in[m*N+n];
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    memcpy(Lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
    orki_setrn((uint32_t*)Lrc.cpu,REGCMD_SILU_LUT_N,RK_DPU_DST_BASE_ADDR,(uint32_t)Lsc.dma);
    { uint32_t*lr=(uint32_t*)Lrc.cpu; int j=0;
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
    uint32_t rc[REGCMD_SILU_STD_I16_N]; memset(rc,0,sizeof rc); memcpy(rc,regcmd,(size_t)rn*4);
    orki_set_mul_geom(rc,rn,M,N);
    orki_setrn(rc,rn,RK_SDP_5040,0); orki_setrn(rc,rn,RK_SDP_5038,0);
    orki_setrn(rc,rn,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,rn,RK_SDP_5018,(uint32_t)A.dma);
    memcpy(c->regcmd.cpu,rc,(size_t)rn*4); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk);
    tk->enable_mask=0x18; tk->int_mask=0x300; tk->int_clear=0x1ffff; tk->regcfg_amount=(uint32_t)(rn/2-4); tk->regcmd_addr=c->regcmd.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms(); double t0=ork_now_us();
    /* ORK_I16_DUMP: dump the exact submit context so the WEDGING chain submit can be diffed byte-for-byte
     * against the CLEAN probe submit of the same shape (in-model instrumentation, #35). */
    if(getenv("ORK_I16_DUMP")){ static long n=0;
        fprintf(stderr,"[i16dump #%ld] M=%d N=%d dom=%d dom_active=%d core=0x%x | A.dma=0x%llx A.dom=%d O.dma=0x%llx O.dom=%d Lrc.dma=0x%llx(d%d) Lsc.dma=0x%llx(d%d) | regcmd.dma=0x%llx regcfg=%u task.obj=0x%llx last_dt=%d warmed=%d\n",
            ++n,M,N,dom,c->dom_active,sub.core_mask,
            (unsigned long long)A.dma,A.domain,(unsigned long long)O.dma,O.domain,
            (unsigned long long)Lrc.dma,Lrc.domain,(unsigned long long)Lsc.dma,Lsc.domain,
            (unsigned long long)c->regcmd.dma,tk->regcfg_amount,(unsigned long long)c->task.obj,c->last_dt,c->warmed);
        fflush(stderr); }
    if(!orki_rknpu_submit_ioctl(fd,&sub,dom)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[m*N+n]=*(int16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(us)*us=t1; }
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);
    #undef EWCUBEH
    return ok;
}

int ork_npu_chain_mm_perchan_i16(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,
                                 const int16_t *scale,int m1,int s1,int m2,int s2,int16_t *out,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems, dom=c->dom_active;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64||(N&7)) return -2;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,dom), G=orki_bcreate(fd,sz,0x403,dom), O=orki_bcreate(fd,sz,0x403,dom), SB=orki_bcreate(fd,4096,0x403,dom);
    if(!W.cpu||!G.cpu||!O.cpu||!SB.cpu){ orki_bdestroy(fd,&W);orki_bdestroy(fd,&G);orki_bdestroy(fd,&O);orki_bdestroy(fd,&SB); return -1; }
    { int NN=N/32,KT=K/32; int8_t*bb=W.cpu;
      for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)]; }
    memset(G.cpu,0,sz); memset(O.cpu,0,sz); memset(SB.cpu,0,4096);
    { int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; }
    /* ALL-INT16 dtype-matched chain (the fp16-out matmul writeout is a HW dead end — see orki_set_f16_out_fp16in NOTE).
     * int8 matmul -> INT16 G (set_i16_out, PROVEN by the gate->silu chain) -> INT16 2-input per-channel SDP
     * (REGCMD_MUL_I16, PROC_PRECISION=1, DATA_FORMAT 0x24000001). Both int16 => dtype-path matched (the same fix
     * that unhung the fp16 pair), and the matmul CAN emit int16 (unlike fp16). PC16/EWCUBEH atom-8 layout is shared
     * between set_i16_out output and the int16 SDP input (the gatesilu bridge).
     * RESULT (2026-07-14): HANGS (errno=110) — the int16 2-input SDP (REGCMD_MUL_I16) is NOT chain-safe: it keeps
     * the BS-ALU ACTIVE (0x4040=0x20050) + relies on reset-provided state, unlike the vendor fp16 chain SDP
     * (REGCMD_MUL_F16_CHAIN, BS fully bypassed 0x53). Dtype-match is necessary but NOT sufficient; the SDP must
     * also be the chain-safe (BS/BN bypassed) variant, which we only have for fp16. So neither end lines up:
     * fp16 has the chain-safe SDP but no matmul writeout; int16 has the matmul writeout but no chain-safe SDP.
     * Next lever = capture a vendor GEMM with fp16-out (how the vendor emits fp16 from a matmul-class op). */
    uint32_t r34=0x00000008u;                                                 /* ERDMA per-channel + 2-byte (int16 SDP) */
    { int16_t*sb=(int16_t*)SB.cpu; for(int n=0;n<N;n++) sb[n]=scale[n]; }      /* int16 per-channel scale CONTIGUOUS [N] */
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&G,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&SB,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    if(getenv("ORK_I16_ENTER")) ork_npu_enter(c,DT_I8,XP_SC_MM,OCK_NONE);   /* layer entry */
    else orki_act(fd,RKNPU_ACT_RESET,0);                                         /* gatesilu-style bare reset (proven for this chained int16-out matmul) */
    static uint32_t mm[REGCMD_I8_N], pc[REGCMD_MUL_I16_N];
    orki_synth_i8(mm,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)G.dma,1,CBUF,0);   /* prog0: matmul INT16-out -> G */
    orki_set_i16_out(mm,N,0,m1,s1);                                                        /* int16 G (m1/s1 requant) — matches the int16 SDP */
    { const char*e=getenv("ORK_I16_MM4010"); if(e) orki_setrn(mm,REGCMD_I8_N,RK_DPU_OUT_PRECISION,(uint32_t)strtoul(e,0,0)); } /* dtype-path: match matmul G-write precision to the SDP's read precision */
    /* prog1: INT16 2-input per-channel SDP (REGCMD_MUL_I16), patched exactly as the bit-exact standalone
     * ork_npu_mul_perchan_i16: per-channel ERDMA (0x5034=0x08, b=[N] contiguous), m2/s2 requant, clear the
     * standalone-only captured zero-points (0x4080/0x4044/0x4074). */
    if(getenv("ORK_I16_MULTMPL")){   /* OLD: standalone-captured REGCMD_MUL_I16 (hangs chained — no chained-ERDMA arming) */
        memcpy(pc,REGCMD_MUL_I16,sizeof pc);
        orki_set_mul_geom(pc,REGCMD_MUL_I16_N,M,N);
        orki_setrn(pc,REGCMD_MUL_I16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
        orki_setrn(pc,REGCMD_MUL_I16_N,RK_SDP_5018,(uint32_t)G.dma);
        orki_setrn(pc,REGCMD_MUL_I16_N,RK_SDP_5038,(uint32_t)SB.dma);
        orki_setrn(pc,REGCMD_MUL_I16_N,RK_SDP_5034,r34);
        orki_setrn(pc,REGCMD_MUL_I16_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)m2); orki_setrn(pc,REGCMD_MUL_I16_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)s2);
        orki_setrn(pc,REGCMD_MUL_I16_N,RK_DPU_OUT_CVT_OFFSET,0); orki_setrn(pc,REGCMD_MUL_I16_N,RK_DPU_BS_ALU_CFG,0); orki_setrn(pc,REGCMD_MUL_I16_N,RK_DPU_EW_CVT_OFFSET,0);
        orki_setrn(pc,REGCMD_MUL_I16_N,RK_DPU_BS_CFG,0x00000053); orki_setrn(pc,REGCMD_MUL_I16_N,RK_DPU_BN_CFG,0x00000053);
    } else {   /* CHAIN-SAFE int16 SDP = the PROVEN chained fp16 template (REGCMD_MUL_F16_CHAIN, vendor conv->mul
                * chained-ERDMA arming), patched precision fp16->int16: 0x4010 int16 DATA_FORMAT, int16 requant
                * (m2/s2), ERDMA per-channel 2-byte. The chain-safe arming (0x5004/0x5008/0x5044/BS-bypass) is
                * inherited verbatim — that's what REGCMD_MUL_I16 lacked. */
        memcpy(pc,REGCMD_MUL_F16_CHAIN,REGCMD_MUL_F16_CHAIN_N*4);
        orki_set_mul_geom(pc,REGCMD_MUL_F16_CHAIN_N,M,N);
        orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
        orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_SDP_5018,(uint32_t)G.dma);
        orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_SDP_5038,(uint32_t)SB.dma);
        orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_SDP_5034,0x00000008);                     /* ERDMA per-channel + 2-byte */
        uint32_t r10=getenv("ORK_I16_R4010")?strtoul(getenv("ORK_I16_R4010"),0,0):0x24000001; /* int16 DATA_FORMAT (was fp16 0x48000002) */
        orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_DPU_OUT_PRECISION,r10);
        orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)m2); orki_setrn(pc,REGCMD_MUL_F16_CHAIN_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)s2); /* int16 requant (was FP32TOFP16_EN) */
    }
    double t0=ork_now_us();
    int crc;
    if(getenv("ORK_I16_SEQ")){
        /* O(M·N) PURE-NPU per-channel scale via SEQUENCED submits (the 2-input SDP isn't chain-safe, but as a
         * SEPARATE submit it reads the matmul's atom-8 G in place — SAME layout, NO repack, NO reshape). The SDP
         * is element-wise per-channel = O(M·N), vs the fp16 diagonal-matmul's O(M·N²). int16 matmul writes atom-8
         * natively (set_i16_out) so — unlike fp16 — no contiguous↔atom-8 bridge is needed. G device-resident. */
        /* mirror the WORKING standalone (mul_perchan_i16): the task already points at c->regcmd from init — only
         * flip regcfg_amount + enable_mask (NO memset / regcmd_addr, which would clobber init's task config). */
        struct rknpu_task*tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask; crc=0;
        memcpy(c->regcmd.cpu,mm,REGCMD_I8_N*4); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
        { tk->regcfg_amount=108; tk->enable_mask=0xd; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
          struct rknpu_submit s;memset(&s,0,sizeof s);s.flags=ork_ppflags();s.task_number=1;s.task_obj_addr=c->task.obj;s.core_mask=RKNPU_CORE0_MASK;s.fence_fd=-1;s.timeout=orki_ew_timeout_ms();s.subcore_task[0]=(struct rknpu_subcore_task){0,1};
          if(orki_rknpu_submit_ioctl(fd,&s,dom)){crc=-1;fprintf(stderr,"[i16seq] MATMUL submit failed errno=%d dom=%d\n",errno,dom);} else orki_bsync(fd,&G,RKNPU_MEM_SYNC_FROM_DEVICE|RKNPU_MEM_SYNC_TO_DEVICE); }   /* matmul -> atom-8 int16 G (device-resident) */
        if(!crc){ ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer */
          memcpy(c->regcmd.cpu,pc,REGCMD_MUL_I16_N*4); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
          tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
          struct rknpu_submit s;memset(&s,0,sizeof s);s.flags=ork_ppflags();s.task_number=1;s.task_obj_addr=c->task.obj;s.core_mask=RKNPU_CORE0_MASK;s.fence_fd=-1;s.timeout=orki_ew_timeout_ms();s.subcore_task[0]=(struct rknpu_subcore_task){0,1};
          if(orki_rknpu_submit_ioctl(fd,&s,dom)){crc=-1;fprintf(stderr,"[i16seq] SDP submit failed errno=%d dom=%d\n",errno,dom);} else orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); }              /* SDP per-channel scale (reads G atom-8 in place) -> O */
        tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);   /* restore shared task */
    } else {
        ork_chain_prog progs[2]={ {mm,REGCMD_I8_N,0xd,108,216}, {pc,REGCMD_MUL_I16_N,0x18,69,-1} };
        crc=ork_npu_chain_progs(c,2,progs,dom);
        if(crc){ orki_bsync(fd,&G,RKNPU_MEM_SYNC_FROM_DEVICE); int gnz=0; int16_t*g=(int16_t*)G.cpu; for(int i=0;i<M*N;i++) if(g[i])gnz++;
            fprintf(stderr,"[i16chain] chain_progs crc=%d errno=%d — G nonzero=%d/%d (task0 matmul %s)\n",crc,errno,gnz,M*N,gnz?"COMPLETED":"did NOT complete"); }
    }
    if(!crc){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); if(us)*us=ork_now_us()-t0;
        for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=*(int16_t*)((char*)O.cpu+EWCUBEH(m,n)); }  /* int16 O */
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&G);orki_bdestroy(fd,&O);orki_bdestroy(fd,&SB);
    #undef EWCUBEH
    return crc;
}

int orki_act_lut_i16(ork_npu *c,double(*f)(double),const int16_t *in,int M,int N,double in_scale,double out_scale,int16_t *out,double *us){
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;
    int16_t lut[1030];
    if(orki_build_act_lut16(c,f,in_scale,out_scale,lut)) return -1;
    return ork_npu_probe_silu_std_i16(c,in,M,N,0x4000,14,0,ORK_SILU16_IDXOFF,ORK_SILU16_C4064,ORK_SILU16_C4068,lut,1030,out,us);
}

int ork_npu_gelu_i16(ork_npu *c,const int16_t *in,int M,int N,double in_scale,double out_scale,int16_t *out,double *us){
    if(c && c->daemon){ if(us)*us=0; return orkd_gelu_i16(c->daemon,in,M,N,in_scale,out_scale,out); }   /* Path B: SDP on the daemon */
    return orki_act_lut_i16(c,orki_gelu_f,in,M,N,in_scale,out_scale,out,us);
}

int ork_npu_rsqrt_i16(ork_npu *c,const int16_t *in,int M,int N,double in_scale,double out_scale,int16_t *out,double *us){
    if(c && c->daemon){ if(us)*us=0; return orkd_rsqrt_i16(c->daemon,in,M,N,in_scale,out_scale,out); }   /* Path B: SDP on the daemon */
    return orki_act_lut_i16(c,orki_rsqrt_f,in,M,N,in_scale,out_scale,out,us);
}

int ork_npu_exp_i16(ork_npu *c,const int16_t *in,int M,int N,double in_scale,double out_scale,int16_t *out,double *us){
    if(c && c->daemon){ if(us)*us=0; return orkd_exp_i16(c->daemon,in,M,N,in_scale,out_scale,out); }   /* Path B: SDP on the daemon */
    return orki_act_lut_i16(c,orki_exp_f,in,M,N,in_scale,out_scale,out,us);
}
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
int ork_npu_probe_silu_std_i16(ork_npu *c,const int16_t *in,int M,int N,
                               int r_mult,int r_shift,uint32_t out_bias,uint32_t idx_off,
                               uint32_t cfg4064,uint32_t cfg4068,const int16_t *lut,int nlut,
                               int16_t *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;
    if(r_mult<0||r_mult>0x7fff||r_shift<0||r_shift>31) return -2;
    orki_last_op="silu_i16_op"; orki_last_K=M; orki_last_N=N; orki_last_wdom=0; orki_last_import=0;   /* accurate wedge telemetry (no validate_regcmd here) */
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)   /* int16 atom-8, 2-byte, surf_stride=M*16 */
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
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) *(int16_t*)((char*)A.cpu+EWCUBEH(m,n))=in[m*N+n];
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);

    /* Build the LUT-load regcmd content + the activation regcmd ONCE. */
    memcpy(Lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
    orki_setrn((uint32_t*)Lrc.cpu,REGCMD_SILU_LUT_N,RK_DPU_DST_BASE_ADDR,(uint32_t)Lsc.dma);
    if(lut){ uint32_t*lr=(uint32_t*)Lrc.cpu; int j=0;
        for(int k=0;k+1<REGCMD_SILU_LUT_N;k+=2){ if((lr[k]&0xffff)==0x4104){ int32_t v=(j<nlut)?(int32_t)lut[j]:0; j++;
            lr[k]=0x4104|((uint32_t)(v&0xffff)<<16); lr[k+1]=(0x1001u<<16)|(((uint32_t)v>>16)&0xffff); } } }
    orki_bsync(fd,&Lrc,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t rc[REGCMD_SILU_STD_I16_N]; memcpy(rc,REGCMD_SILU_STD_I16,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_SILU_STD_I16_N,M,N);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_SDP_5040,0);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_SDP_5038,0);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_SDP_5018,(uint32_t)A.dma);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)r_mult);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)r_shift);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_OFFSET,out_bias);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_R4110,idx_off);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_BN_ALU_CFG,cfg4064);
    orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_BN_MUL_CFG,cfg4068);
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);

    /* #35: the int16 silu is a STANDALONE pure-SDP op (enable_mask 0x18). IN A CHAIN, its submit wedges in
     * the chain CONTEXT (works standalone at every shape, i16_shape_probe). NO userspace mitigation fixes
     * it: ping-pong OFF, ACT_RESET, a pipeline-DRAIN (tiny single-core matmul first), and reset+retry ALL
     * fail — even the calib's 64x64 probe wedges on every attempt after soft-resets. Only the kernel
     * self-heal limps it through (correct output, ~1 reset/layer). And the FUSED int16 output is CLOSED
     * (int8-only, ⚠ note below), so the standalone op is the only int16 path. => int16 silu NOT viable
     * in-chain with current NPU understanding — needs kernel-level reset or a deeper pipeline fix. Left as
     * the RE artifact (ORK_FFN_SILU_I16). Shipped coherent path is all-CPU-silu. ping-pong OFF (LUT-op). */
    /* #35 RESOLVED: the per-call ACT_RESET was pure OVERHEAD, not a wedge guard. The in-chain "wedge" was a
     * MISREAD — dmesg "RKNPU: soft reset" counts the DELIBERATE ACT_RESET (ORK_DEBUG_RESET: 23 act calls ->
     * 27 dmesg entries), not hardware wedges (errno=110 count = 0 in a full chain run). Removing it: int16
     * chain prefill 28->68 tok/s, PPL 19.02 unchanged, 0 real wedges. Default OFF; ORK_I16_RESET re-enables. */
    if(getenv("ORK_I16_RESET")) orki_act(fd,RKNPU_ACT_RESET,0);
    /* MODE-TRANSITION FIX (mode_probe RE): this LUT op memsets the SHARED c->task to its own SDP descriptor
     * (regcfg_amount 1097 then 69, enable_mask 0x18) and previously left it that way. The single-core matmul
     * path (orki_run()/submit1) does NOT rebuild c->task — it relies on the init value (regcfg_amount=108,
     * enable_mask=0xd, regcmd_addr=c->regcmd.dma) persisting. So a later SINGLE-CORE matmul (e.g. the SSD
     * CumBA bmm, N=16) submitted a 108-word matmul regcmd under this stale 69-reg/0x18 SDP task -> the NPU
     * dispatched no task (task counter 0x0) -> errno=110 wedge that ACT_RESET can't clear (poisoned software
     * descriptor, not HW state). The plain 2-input SDP ops (ewmul/add) never wedged because they already
     * save+restore these fields. Save them here and restore on every exit, exactly like ewmul. */
    struct rknpu_task *tk0=(struct rknpu_task*)c->task.cpu;
    uint32_t sv_amt=tk0->regcfg_amount, sv_en=tk0->enable_mask; uint64_t sv_addr=tk0->regcmd_addr;
    #define SILU_RESTORE_TASK() do{ tk0->regcfg_amount=sv_amt; tk0->enable_mask=sv_en; tk0->regcmd_addr=sv_addr; \
        orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE); }while(0)
    { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
      t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=1097; t->regcmd_addr=Lrc.dma;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x1;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.timeout=orki_ew_timeout_ms();sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ SILU_RESTORE_TASK(); orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc); return -1; } }
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk);
    tk->enable_mask=0x18; tk->int_mask=0x300; tk->int_clear=0x1ffff; tk->regcfg_amount=69; tk->regcmd_addr=c->regcmd.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x1;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms(); double t0=ork_now_us();
    /* ORK_I16_DUMP: dump the exact submit context so the WEDGING chain submit can be diffed byte-for-byte
     * against the CLEAN probe submit of the same shape (in-model instrumentation, #35). */
    if(getenv("ORK_I16_DUMP")){ static long n=0;
        fprintf(stderr,"[i16dump #%ld] M=%d N=%d dom=%d dom_active=%d core=0x%x | A.dma=0x%llx A.dom=%d O.dma=0x%llx O.dom=%d Lrc.dma=0x%llx(d%d) Lsc.dma=0x%llx(d%d) | regcmd.dma=0x%llx regcfg=%u task.obj=0x%llx last_dt=%d warmed=%d\n",
            ++n,M,N,dom,c->dom_active,sub.core_mask,
            (unsigned long long)A.dma,A.domain,(unsigned long long)O.dma,O.domain,
            (unsigned long long)Lrc.dma,Lrc.domain,(unsigned long long)Lsc.dma,Lsc.domain,
            (unsigned long long)c->regcmd.dma,tk->regcfg_amount,(unsigned long long)c->task.obj,c->last_dt,c->warmed);
        fflush(stderr); }
    if(!orki_rknpu_submit_ioctl(fd,&sub,dom)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[m*N+n]=*(int16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(us)*us=t1; }
    SILU_RESTORE_TASK();     /* leave the shared c->task as the matmul path expects (see save above) */
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);
    #undef EWCUBEH
    #undef SILU_RESTORE_TASK
    return ok;
}
int ork_npu_ewmul_i16(ork_npu *c,const int16_t *up,const int16_t *silu,int M,int N,int mult,int shift,int16_t *out,double *us){
    if(c && c->daemon){ if(us)*us=0; return orkd_ewmul_i16(c->daemon,up,silu,M,N,mult,shift,out); }   /* Path B: SDP on the daemon */
    int fd=c->fd, dom=c->dom_active;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;               /* N multiple of the int16 atom (8) */
    if(mult<0||mult>0x7fff||shift<0||shift>31) return -2;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)   /* 2-byte atom=8 cube (fp16/int16), surf_stride=M*16 */
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;                  /* int16 cube = M*N*2 bytes */
    struct buf A=orki_bcreate(fd,sz,0x403,dom); if(!A.cpu)return -2;
    struct buf B=orki_bcreate(fd,sz,0x403,dom); if(!B.cpu){orki_bdestroy(fd,&A);return -2;}
    struct buf O=orki_bcreate(fd,sz,0x403,dom); if(!O.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);return -2;}
    memset(A.cpu,0,sz);memset(B.cpu,0,sz);memset(O.cpu,0,sz);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int p=EWCUBEH(m,n);
        *(int16_t*)((char*)A.cpu+p)=up[m*N+n]; *(int16_t*)((char*)B.cpu+p)=silu[m*N+n]; }
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (XP_SDP=RC_SDPKW, keep-warm-aware; == the old inline reset) */
    uint32_t rc[REGCMD_MUL_I16_N]; memcpy(rc,REGCMD_MUL_I16,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_MUL_I16_N,M,N);
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_5018,(uint32_t)A.dma);
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_5038,(uint32_t)B.dma);
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)mult);     /* OUT_CVT_SCALE (gain mantissa) */
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)shift);    /* OUT_CVT_SHIFT */
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_OUT_CVT_OFFSET,0);                  /* zo=0 */
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_BS_ALU_CFG,0);                  /* za=0 */
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_EW_CVT_OFFSET,0);                  /* zb=0 */
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=orki_ew_timeout_ms(); double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,dom)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[m*N+n]=*(int16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(us)*us=t1; }
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&O);
    #undef EWCUBEH
    return ok;
}

int ork_npu_silu_i16(ork_npu *c,const int16_t *in,int M,int N,double in_scale,double out_scale,int16_t *out,double *us){
    if(c && c->daemon){ if(us)*us=0; return orkd_silu_i16(c->daemon,in,M,N,in_scale,out_scale,out); }   /* Path B: SDP on the daemon */
    return orki_act_lut_i16(c,orki_silu_f,in,M,N,in_scale,out_scale,out,us);
}
