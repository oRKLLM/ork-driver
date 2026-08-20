/* npu/i16/act.c — int16 SDP activations and elementwise.
 *
 * Part of the i16 datapath; shared declarations in npu/i16/i16.h. Split out of npu/i16.c for the
 * same reason i8 is a folder: one datapath, sized for reading. */
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
#include "npu/i16/i16.h"

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
int ork_npu_requant_perchan_i32(ork_npu *c,const int32_t *a,const int16_t *b,int M,int N,int mult,int shift,int16_t *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;
    if(mult<0||mult>0x7fff||shift<0||shift>31) return -2;
    #define PC32(m,n) (((n)/8)*(M*32) + (m)*32 + ((n)%8)*4)          /* 4-byte atom=8 cube (int32 in) */
    #define PC16(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)          /* 2-byte atom=8 cube (int16 out) */
    size_t sza=(size_t)M*N*4; if(sza<4096)sza=4096;
    size_t szo=(size_t)M*N*2; if(szo<4096)szo=4096;
    struct buf A=orki_bcreate(fd,sza,0x403,-1), O=orki_bcreate(fd,szo,0x403,-1), B=orki_bcreate(fd,4096,0x403,-1);
    if(!A.cpu||!O.cpu||!B.cpu){ orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&B); return -2; }
    memset(A.cpu,0,sza); memset(O.cpu,0,szo); memset(B.cpu,0,4096);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) *(int32_t*)((char*)A.cpu+PC32(m,n))=a[(size_t)m*N+n];
    for(int n=0;n<N;n++) ((int16_t*)B.cpu)[n]=b[n];                  /* per-channel vector CONTIGUOUS [N] int16 */
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (keep-warm-aware) */
    uint32_t rc[REGCMD_MUL_I16_N]; memcpy(rc,REGCMD_MUL_I16,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_MUL_I16_N,M,N);
    #define RQENV(nm,def) (getenv(nm)?(uint32_t)strtoul(getenv(nm),0,0):(uint32_t)(def))
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_OUT_PRECISION,RQENV("ORK_RQ_4010",0x30000001)); /* OUT int16 | IN int32 | PROC int16 */
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_5040,RQENV("ORK_RQ_MSTRIDE",(uint32_t)(M*32))); /* main int32 surf stride */
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_5018,(uint32_t)A.dma);        /* main input = int32 G */
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_5038,(uint32_t)B.dma);        /* per-channel scale vector */
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_5034,RQENV("ORK_RQ_5034",0x08)); /* operand per-channel, DATA_SIZE=2 (int16 b) */
    { const char*e=getenv("ORK_RQ_5044"); if(e) orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_5044,(uint32_t)strtoul(e,0,0)); } /* main-RDMA FEATURE_MODE: IN_PRECISION[17:15] */
    /* OVER-FETCH hack: RDMA input dims (0x500c width / 0x5014 channel) are DECOUPLED from the DPU output dims
     * (0x4058/0x405c). If the RDMA is stuck 2-byte fetching 2E but the DPU consumes 4E (int32) -> 50% starve,
     * INFLATE the RDMA element count so it fetches 4E bytes -> both terminal counts hit together -> clean. */
    { const char*e;
      if((e=getenv("ORK_RQ_5014"))) orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_5014,(uint32_t)strtoul(e,0,0)); /* RDMA cube CHANNEL */
      if((e=getenv("ORK_RQ_500C"))) orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_500C,(uint32_t)strtoul(e,0,0)); /* RDMA cube WIDTH  */ }
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)mult); orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)shift);
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_OUT_CVT_OFFSET,0); orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_BS_ALU_CFG,0); orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_EW_CVT_OFFSET,0);
    #undef RQENV
    if(getenv("ORK_RQ_DUMP")){ for(int k=0;k+1<REGCMD_MUL_I16_N;k+=2){ unsigned rg=rc[k]&0xffff; uint32_t v=((rc[k]>>16)&0xffff)|((rc[k+1]&0xffff)<<16);
        if(rg==0x4010||rg==0x4020||rg==0x4024||rg==0x40c0||rg==0x5018||rg==0x5034||rg==0x5038||rg==0x5040||rg==0x5044||rg==0x4084||rg==0x4088) fprintf(stderr,"  [rq] reg=%04x val=%08x\n",rg,v);} }
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
    else if(getenv("ORK_RQ_DUMP")){ int nz=0; int16_t*oc=O.cpu; for(size_t i=0;i<(size_t)M*N;i++) if(oc[i])nz++; fprintf(stderr,"  [rq] submit FAILED (errno path); O nonzero=%d/%d\n",nz,M*N); }
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&O); orki_bdestroy(fd,&B);
    #undef PC32
    #undef PC16
    return ok;
}
