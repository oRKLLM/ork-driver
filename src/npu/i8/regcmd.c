/* npu/i8/regcmd.c — int8 regcmd synthesis and output-stage setup (synth_i8, the requant/SiLU/ewmul output stages).
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
#include "npu/internal.h"
#include "npu/core.h"
#include "npu/i8/i8.h"
#include "spine_kernels.h"

void orki_synth_i8(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC,int sched,int cbuf,int stride){
    memcpy(rc,REGCMD_I8,REGCMD_I8_N*4);
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_DATA_SIZE1,((K-1)<<16)|K);orki_setrn(rc,REGCMD_I8_N,RK_CNA_WEIGHT_SIZE0,K*N);orki_setrn(rc,REGCMD_I8_N,RK_CNA_WEIGHT_SIZE1,K);
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_CBUF_CON1,(K+63)/64);orki_setrn(rc,REGCMD_I8_N,RK_CNA_FC_DATA_SIZE1,K);orki_setrn(rc,REGCMD_I8_N,RK_CNA_DMA_CON1,K/16);
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_DATA_SIZE0,0x10000|mc);orki_setrn(rc,REGCMD_I8_N,RK_CNA_DATA_SIZE0_MIR,0x10000|mc);orki_setrn(rc,REGCMD_I8_N,RK_CNA_DATA_SIZE3,mc);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_DATA_CUBE_HEIGHT,mc-1);orki_setrn(rc,REGCMD_I8_N,RK_DPU_WDMA_SIZE_1,(mc-1)<<16);orki_setrn(rc,REGCMD_I8_N,RK_PDP_OUT_M,(mc-1)<<16);
    int s=stride>0?stride:N;
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_DST_N_DIMS,((s-1)<<16)|(N-1));orki_setrn(rc,REGCMD_I8_N,RK_DPU_DST_N2,N-1);orki_setrn(rc,REGCMD_I8_N,RK_DPU_DATA_CUBE_NOTCH,(((s/4)-1)<<16)|((N/4)-1));
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_WEIGHT_SIZE2,0x1010000|N);orki_setrn(rc,REGCMD_I8_N,RK_PDP_OUT_N,N-1);
    if(sched){
        int R=(2*cbuf)/K; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
        /* DEBUG knob (ORK_RCAP, default off): override the 0x1010 row-count hint. MEASURED NEUTRAL —
         * 0x1010 is only a hint; it does NOT change correctness or speed (the real single-core lever is
         * the M-tile size mg_max*64, set by the caller — see AGENTS.md "weight-DMA amortization"). Kept
         * only for RE/diagnostics. */
        static int rcap=-2; if(rcap==-2){const char*e=getenv("ORK_RCAP"); rcap=e?atoi(e):-1;}
        if(rcap>0) R=rcap;
        int rows=(mc+1<R)?(mc+1):R; orki_setrn(rc,REGCMD_I8_N,RK_CNA_CONV_CON2,16*rows);
        /* 0x1040 = K-reduction schedule, selected per 64-row group. MUST be ceil(mc/64): a tile of
         * 65..127 rows spills past the first 64-row group, so it needs the NEXT schedule (the one a
         * full 128-row tile uses), not the <=64 schedule. floor(mc/64) gave 65..127-row tiles the
         * <=64 schedule -> rows 64..mc-1 computed against the wrong K-partition (the prefill bug). */
        double scale=(double)K/512.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale),mg=(mc+63)/64; if(mg<1)mg=1;
        int v=base-slope*(mg-1); if(v<0x1b)v=0x1b;
        static int r1040=-2; if(r1040==-2){const char*e=getenv("ORK_R1040"); r1040=e?atoi(e):-1;}
        if(r1040>0) v=r1040;   /* EXPERIMENTAL: override the CBUF_CON0 bank alloc (rknn's captured 0x75 for the 30-row tile) */
        orki_setrn(rc,REGCMD_I8_N,RK_CNA_CBUF_CON0,v);
    } else { orki_setrn(rc,REGCMD_I8_N,RK_CNA_CONV_CON2,16*(mc+1)); }
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_FEATURE_DATA_ADDR,aA);orki_setrn(rc,REGCMD_I8_N,RK_CNA_WEIGHT_DATA_ADDR,aB);orki_setrn(rc,REGCMD_I8_N,RK_DPU_DST_BASE_ADDR,aC);
    for(int i=0;i<orki_i8_fovr_n;i++) orki_setr(rc,REGCMD_I8_N,orki_i8_fovr[i].blk,orki_i8_fovr[i].reg,orki_i8_fovr[i].val);  /* RE fuzzer overrides (win over all) */
}

void orki_synth_i8_mfold(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC,int cbuf){
    orki_synth_i8(rc,mc,K,N,aA,aB,aC,0,cbuf,0);                       /* ork baseline; non-delta regs already match */
    /* --- register-level clone of rkllm's captured M=36 mfold (named via ork_regs.h) --- */
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_CONV_CON1,      OKV_CONV1_GROUP_LINE);            /* GROUP_LINE_OFF feature-read (rkllm SETS this for the big-M fold tasks) */
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_CONV_CON2,      0x20);                     /* FEATURE_GRAINS=2 */
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_DATA_SIZE0,     OKC_DATA_SIZE0(mc,1));     /* fold M into CNA WIDTH (W=M,H=1) */
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_DATA_SIZE0_MIR, OKC_DATA_SIZE0(mc,1));
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_DATA_SIZE_BATCH,mc);                       /* 0x1028 = M */
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_DATA_SIZE3,     mc);                       /* 0x102c = M */
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_CBUF_CON1,(uint32_t)(K/64)*mc);           /* DATA_ENTRIES = (K/64)*M */
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_DMA_CON2,  (uint32_t)(2160/mc));           /* REAL surface stride 2160/M (60@36,108@20); NOT the wedge-prone sentinel */
    /* CBUF/DMA schedule — rkllm M=36 LITERALS (per-M formula still open; validated at M=36 only) */
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_DMA_CON1,       0x60);                     /* 0x107c */
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_CBUF_CON0,      0x84);                     /* 0x1040 bank split + K-reduction schedule */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_DST_SURF_STRIDE,0x600);                   /* 0x4024 output surface stride */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_SURFACE_ADD,    0x3000);                  /* 0x40c0 surface/elem-size CONFIG */
    /* extra CNA regs rkllm sets (constant across the captured M values) */
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_CONV_CON3, 0x9);
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_CVT_CON0, 0xb);
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_CVT_CON1, 0x10000); orki_setrn(rc,REGCMD_I8_N,RK_CNA_CVT_CON2, 0x10000);
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_CVT_CON3, 0x10000); orki_setrn(rc,REGCMD_I8_N,RK_CNA_CVT_CON4, 0x10000);
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_DMA_CON0, 0xf000f);
    orki_setrn(rc,REGCMD_I8_N,RK_PDP_R3010, 0x1);                               /* 0x0801:0x3010 */
    /* DPU output stage */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_PRECISION,  OKV_OUT_PREC_INT32);      /* 0x4010 int32 accumulate-out */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_DATA_CUBE_WIDTH, mc-1);                   /* 0x4030 out WIDTH=M-1 */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_DATA_CUBE_HEIGHT,0);                      /* 0x4034 */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_DATA_CUBE_NOTCH, 0);                      /* 0x4038 no notch */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_WDMA_SIZE_1,     mc-1);                   /* 0x405c WIDTH_WDMA=M-1 */
    orki_setrn(rc,REGCMD_I8_N,RK_PDP_OUT_M,           mc-1);
    for(int i=0;i<orki_i8_fovr_n;i++) orki_setr(rc,REGCMD_I8_N,orki_i8_fovr[i].blk,orki_i8_fovr[i].reg,orki_i8_fovr[i].val);
}

void orki_set_i8_out8(uint32_t*rc,int N,int stride,int mult,int shift){
    int s=stride>0?stride:N;
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_PRECISION,0);                                   /* clear the 0x8000 int32-output bit -> int8 out */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_DATA_CUBE_NOTCH,(((s/16)-1)<<16)|((N/16)-1));         /* output group stride: int8 packs 4x denser than int32 (N/16 vs N/4) */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_BS_OW_CFG,0x0124);                             /* int8 output row byte-stride config (const; int32=0x07fc) */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_SURFACE_ADD,0x0020);                             /* output element size = 1 byte (int32=0x0080) */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_SCALE,mult);                              /* requant multiplier */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_SHIFT,shift);                             /* requant shift (>>) */
}

void orki_set_i8_silu(uint32_t*rc,int N,int stride,int r_mult,int r_shift,
                        uint32_t out_bias,uint32_t idx_off,uint32_t cfg4068){
    orki_set_i8_out8(rc,N,stride,r_mult,r_shift);       /* int8-output byte layout + the unified scale R (0x4084/0x4088) */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_S_POINTER,0x0030); orki_setrn(rc,REGCMD_I8_N,RK_SDP_5004,0x0030); /* activation mode on */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_PRECISION,0x44e0);     /* LUT/activation enable (output-stage high byte 0x44) */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_BN_CFG,0x00020040); /* activation mode bit 0x0002 (fixed) */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_BN_MUL_CFG,cfg4068);    /* per-scale field, no observed output effect (replay) */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_EW_CFG,0x00000302); /* fixed */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_OFFSET,out_bias);   /* output bias / asymmetric zero-point (silu(0) -> out_bias) */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_R4108,0x00000068); /* LUT config (fixed) */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_R410C,0x00050500); /* LUT config (fixed) */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_R4110,idx_off);    /* index offset -> C0 (silu-zero index ~512) */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_R411C,0x00004000); /* fixed config (scale-independent) */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_R4128,0x40320000); /* fixed config */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_R412C,0x000001a0); /* fixed config */
}

void orki_set_i8_ewmul(uint32_t*rc,int M,int N,int stride,int mult,int shift,uint32_t aG){
    orki_set_i8_out8(rc,N,stride,mult,shift);                         /* ork conv + int8-out; OUT_CVT gain=0x4084/88 */
    int s=stride>0?stride:N;
    if(getenv("ORK_EW_REGOP")){
        /* DECISIVE ISOLATION: EW multiply with a REGISTER-constant operand (EW_OP_SRC=0, bit6=0), NO RDMA.
         * out = up_acc * EW_OP_VALUE_0. If this works, the EW stage is fine on ork geometry & the RDMA is the
         * wedge; if it wedges, the EW stage itself is incompatible with ork's dense conv geometry. */
        orki_setrn(rc,REGCMD_I8_N,RK_DPU_EW_CFG,0x90400284);          /* EW_CFG mul, OP_SRC=0 (register operand) */
        orki_setrn(rc,REGCMD_I8_N,RK_DPU_EW_OP_VALUE_0,(uint32_t)strtoul(getenv("ORK_EW_REGOP"),0,0)); /* EW_OP_VALUE_0 */
        orki_setrn(rc,REGCMD_I8_N,RK_DPU_BS_OW_CFG,0x00000125);
        orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_PRECISION,0x000000e0);
    } else if(!getenv("ORK_EW_NOMUL")){
        /* EW_CFG: multiply(bit2), operand from RDMA(bit6), operand-CVT active(bit8=0), LUT+ReLU bypass(7,9).
         * ORK_EW_CFG / ORK_EW_ERDMA override EW_CFG / ERDMA_CFG for int8-vs-int16 data-size tuning. */
        uint32_t ewcfg=0x904002c4; { const char*e=getenv("ORK_EW_CFG"); if(e) ewcfg=(uint32_t)strtoul(e,0,0); }
        orki_setrn(rc,REGCMD_I8_N,RK_DPU_EW_CFG,ewcfg);
        orki_setrn(rc,REGCMD_I8_N,RK_DPU_EW_CVT_OFFSET,0x00000000);          /* EW_CVT_OFFSET_VALUE = 0 (silu zero-point 0) */
        orki_setrn(rc,REGCMD_I8_N,RK_DPU_EW_CVT_SCALE,0x00000001);          /* EW_CVT_SCALE=1, SHIFT=0 (unity operand cvt) */
        /* output-stage EW-active bits (required so the DPU expects the element-wise/RDMA stage; mode bits,
         * geometry-independent). ORK_EW_NO50=skip individual ones during bisection. */
        orki_setrn(rc,REGCMD_I8_N,RK_DPU_BS_OW_CFG,0x00000125);          /* out row cfg + EW-enable bit0 (out8=0x124) */
        orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_PRECISION,0x000000e0);          /* DATA_FORMAT EW bits (out8=0) */
        { const char*eo=getenv("ORK_EW_COFF"); if(eo) orki_setrn(rc,REGCMD_I8_N,RK_DPU_EW_CVT_OFFSET,(uint32_t)strtoul(eo,0,0)); }
        { const char*es=getenv("ORK_EW_CSCL"); if(es) orki_setrn(rc,REGCMD_I8_N,RK_DPU_EW_CVT_SCALE,(uint32_t)strtoul(es,0,0)); }
        /* DPU_RDMA (0x50xx): fetch silu(gate) as the element-wise operand at EW_BASE (0x5038).
         * ORK_EW_SPTR overrides RDMA_S_POINTER (0x5004): 0xe = PP mode (producer/consumer ping-pong, needs a
         * partner to advance the pointer — deadlocks in a standalone shot); try 0/1 for single-shot no-PP. */
        { uint32_t sp=0x0000000e; const char*e=getenv("ORK_EW_SPTR"); if(e) sp=(uint32_t)strtoul(e,0,0);
          orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5004,sp); }
        orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5008,0x00000001);       /* RDMA_OPERATION_ENABLE (only in EW-memory path) */
        orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_500C,M-1);              /* RDMA_DATA_CUBE_WIDTH  = M-1 */
        orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5010,0x00000000);       /* RDMA_DATA_CUBE_HEIGHT = 0 (H=1) */
        orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5014,N-1);              /* RDMA_DATA_CUBE_CHANNEL= N-1 */
        uint32_t erdma=0x40000004; { const char*e=getenv("ORK_EW_ERDMA"); if(e) erdma=(uint32_t)strtoul(e,0,0); }
        orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5034,erdma);            /* RDMA_ERDMA_CFG: enable(bit0=0)+data_size/mode */
        orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5038,aG);               /* RDMA_EW_BASE_ADDR = silu(gate) */
        orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5018,aG);               /* RDMA_SRC_BASE_ADDR (valid) */
        /* rocket rkt_regcmd.c element-wise: the operand is read via the BRDMA channel from SRC_BASE (0x5018),
         * with BRDMA_DATA_USE=1 (0x501c bits1-4 => value 0x2). (I earlier wrongly DISABLED BRDMA -> the RDMA
         * never delivered the operand -> DPU hung.) NRDMA off; BS_BASE valid (not the stale RKNN addr). */
        orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_501C,0x00000002);       /* RDMA_BRDMA_CFG: BRDMA_DATA_USE=1 (on) */
        orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5028,0x00000000);       /* RDMA_NRDMA_CFG: NRDMA_DATA_USE=0 (off) */
        orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5020,aG);               /* RDMA_BS_BASE_ADDR: valid (not stale RKNN) */
        orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5040,s);                /* RDMA_EW_SURF_STRIDE (dense = N) */
        orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_504C,s);
        orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_506C,s);
        /* 0x5044 (FEATURE_MODE_CFG int8) + 0x5068 (RDMA_WEIGHT=0x01010101) come from REGCMD_EW_LANE as-is. */
        /* ORK_EW_STRIDE overrides EW_SURF_STRIDE (cube atom-16 layout probe: M*16) */
        { const char*e=getenv("ORK_EW_STRIDE"); if(e){ uint32_t v=(uint32_t)strtoul(e,0,0);
            orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_5040,v); orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_506C,v); orki_setrn(rc,REGCMD_I8_EW_N,RK_SDP_504C,v); } }
        /* PC_OPERATION_ENABLE (regcmd trailer, reg 0x8 lane 0x81): ork's REGCMD_I8 trailer has 0x0d
         * (CNA|CORE|DPU) — the DPU_RDMA block (bit 0x10) is NOT enabled, so the RDMA never runs and the DPU
         * waits forever. RKNN's EW op has 0x1d here. Enable the RDMA block in the regcmd's own op-enable. */
        orki_setrn(rc,REGCMD_I8_EW_N,RK_PC_OPERATION_ENABLE,0x0000001d);
    }
    /* ORK_EW_BIAS overrides 0x4080 (output offset/bias) */
    { const char*e=getenv("ORK_EW_BIAS"); if(e) orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_OFFSET,(uint32_t)strtoul(e,0,0)); }
}

void orki_set_i8_silu32(uint32_t*rc,int N,int r_mult,int r_shift,uint32_t out_bias,uint32_t idx_off,uint32_t cfg4068){
    /* SWEEP knobs: output-format registers env-configurable to find the matmul+LUT non-int8 output encoding.
     * PREC = bits[1:0] of 0x4010 (int8=0, int16=1, fp16=2; bit31=int32-bypass-CVT). int8+silu = 0x44e0. */
    static uint32_t r4010=0,r40c0=0,r4050=0,r84=0,r88=0; static int div38=0,ovg=0,init=0;
    if(!init){ init=1; const char*e;
        e=getenv("ORK_SILU_4010"); r4010=e?(uint32_t)strtoul(e,0,0):0x000044e1u;   /* default: PREC=1 (int16), CVT kept */
        e=getenv("ORK_SILU_40C0"); r40c0=e?(uint32_t)strtoul(e,0,0):0x40u;          /* 2-byte element (int8=0x20,int32=0x80) */
        e=getenv("ORK_SILU_4050"); r4050=e?(uint32_t)strtoul(e,0,0):0x0124u;        /* row byte-stride (int8=0x124,int32=0x7fc) */
        e=getenv("ORK_SILU_38DIV"); div38=e?atoi(e):8;                              /* group stride divisor (int8=16,int32=4,int16=8) */
        const char*g=getenv("ORK_SILU_4084"); if(g){ ovg=1; r84=(uint32_t)strtoul(g,0,0);
            const char*s=getenv("ORK_SILU_4088"); r88=s?(uint32_t)strtoul(s,0,0):0; } }  /* CVT gain override (fp16=0x00010001) */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_SCALE,ovg?r84:(uint32_t)r_mult);   /* CVT gain (int R mantissa, or fp16 0x00010001) */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_SHIFT,ovg?r88:(uint32_t)r_shift);  /* CVT shift */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_S_POINTER,0x0030); orki_setrn(rc,REGCMD_I8_N,RK_SDP_5004,0x0030); /* activation mode on */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_PRECISION,r4010);              /* output precision (PREC field) + LUT/activation enable */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_SURFACE_ADD,r40c0);              /* output element size */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_BS_OW_CFG,r4050);              /* output row byte-stride config */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_DATA_CUBE_NOTCH,(((N/div38)-1)<<16)|((N/div38)-1)); /* output group stride */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_BN_CFG,0x00020040);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_BN_MUL_CFG,cfg4068);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_EW_CFG,0x00000302);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_OFFSET,out_bias);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_R4108,0x00000068);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_R410C,0x00050500);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_R4110,idx_off);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_R411C,0x00004000);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_R4128,0x40320000);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_R412C,0x000001a0);
}
void orki_set_mul_geom(uint32_t *rc,int n,int M,int N){
    uint32_t sstride=(uint32_t)(M*16);
    orki_setrn(rc,n,RK_SDP_500C,(uint32_t)(M-1));          /* RDMA_DATA_CUBE_WIDTH  = M-1 */
    orki_setrn(rc,n,RK_SDP_5010,0);                        /* RDMA_DATA_CUBE_HEIGHT = 0 (H=1) */
    orki_setrn(rc,n,RK_SDP_5014,(uint32_t)(N-1));          /* RDMA_DATA_CUBE_CHANNEL= N-1 */
    orki_setrn(rc,n,RK_SDP_5040,sstride);                  /* RDMA_EW_SURF_STRIDE = M*16 */
    orki_setrn(rc,n,RK_DPU_DST_SURF_STRIDE,sstride);                  /* output surface stride */
    orki_setrn(rc,n,RK_DPU_DATA_CUBE_WIDTH,(uint32_t)(M-1));
    orki_setrn(rc,n,RK_DPU_DST_N_DIMS,(uint32_t)(((N-1)<<16)|(N-1)));
    orki_setrn(rc,n,RK_DPU_DST_N2,(uint32_t)(N-1));
    orki_setrn(rc,n,RK_DPU_WDMA_SIZE_1,(uint32_t)(M-1));
    orki_setrn(rc,n,RK_DPU_SURFACE_ADD,sstride);                  /* SURFACE_ADD = M*16 */
}

void orki_apply_ork_geom(uint32_t*rc,int n,int mc,int K,int N,int cbuf){
    orki_setrn(rc,n,RK_CNA_DATA_SIZE1,((K-1)<<16)|K);orki_setrn(rc,n,RK_CNA_WEIGHT_SIZE0,K*N);orki_setrn(rc,n,RK_CNA_WEIGHT_SIZE1,K);
    orki_setrn(rc,n,RK_CNA_CBUF_CON1,(K+63)/64);orki_setrn(rc,n,RK_CNA_FC_DATA_SIZE1,K);orki_setrn(rc,n,RK_CNA_DMA_CON1,K/16);
    orki_setrn(rc,n,RK_CNA_DATA_SIZE0,0x10000|mc);orki_setrn(rc,n,RK_CNA_DATA_SIZE0_MIR,0x10000|mc);orki_setrn(rc,n,RK_CNA_DATA_SIZE3,mc);
    orki_setrn(rc,n,RK_DPU_DATA_CUBE_HEIGHT,mc-1);orki_setrn(rc,n,RK_DPU_WDMA_SIZE_1,(mc-1)<<16);orki_setrn(rc,n,RK_PDP_OUT_M,(mc-1)<<16);
    orki_setrn(rc,n,RK_DPU_DST_N_DIMS,((N-1)<<16)|(N-1));orki_setrn(rc,n,RK_DPU_DST_N2,N-1);orki_setrn(rc,n,RK_DPU_DATA_CUBE_NOTCH,(((N/4)-1)<<16)|((N/4)-1));
    orki_setrn(rc,n,RK_CNA_WEIGHT_SIZE2,0x1010000|N);orki_setrn(rc,n,RK_PDP_OUT_N,N-1);
    int R=(2*cbuf)/K; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
    int rows=(mc+1<R)?(mc+1):R; orki_setrn(rc,n,RK_CNA_CONV_CON2,16*rows);
    double scale=(double)K/512.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale),mg=(mc+63)/64; if(mg<1)mg=1;
    int v=base-slope*(mg-1); if(v<0x1b)v=0x1b; orki_setrn(rc,n,RK_CNA_CBUF_CON0,v);
}

void orki_splice_ew_lane(uint32_t*rc,const uint32_t*base){
    memcpy(rc,               base,             216*4);                 /* 108 register entries (0x10xx/0x30xx/0x40xx) */
    memcpy(rc+216,           REGCMD_EW_LANE,   REGCMD_EW_LANE_N*4);    /* 18 second-lane entries (0x50xx) */
    memcpy(rc+216+REGCMD_EW_LANE_N, base+216,  8*4);                   /* the original end-of-regcmd trailer, now last */
}
