/* npu/i16/chain.c — HW-chained int16 matmul+activation graphs.
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
