/* npu/f16/regcmd.c — fp16 regcmd synthesis, output stages and the fuzz hooks.
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

struct ork_regovr orki_f16_fovr[16]; int orki_f16_fovr_n=0;

void ork_f16_fuzz_clear(void){ orki_f16_fovr_n=0; }

void ork_f16_fuzz_add(uint32_t blk,uint32_t reg,uint32_t val){ if(orki_f16_fovr_n<16){ orki_f16_fovr[orki_f16_fovr_n].blk=blk; orki_f16_fovr[orki_f16_fovr_n].reg=reg; orki_f16_fovr[orki_f16_fovr_n].val=val; orki_f16_fovr_n++; } }

void orki_set_f16_out(uint32_t*rc,int N,int stride){
    int s=stride>0?stride:N; (void)s;
    unsigned r04=getenv("ORK_F16OUT_4004")?strtoul(getenv("ORK_F16OUT_4004"),0,0):0x0000000e;
    /* 0x4010 DATA_FORMAT fields (rocket registers.xml, confirmed by our ewmul fp16=0x48000002):
     * OUT_PRECISION[31:29] | IN_PRECISION[28:26] | PROC_PRECISION[2:0], precision enum int8=0/int16=1/fp16=2.
     * INT8->FP16 = OUT=fp16(2)<<29 | IN=int8(0)<<26 | PROC=int8(0) = 0x40000000. */
    unsigned r10=getenv("ORK_F16OUT_4010")?strtoul(getenv("ORK_F16OUT_4010"),0,0):0x40000000;
    unsigned r50=getenv("ORK_F16OUT_4050")?strtoul(getenv("ORK_F16OUT_4050"),0,0):0x0000036e;
    unsigned rc0=getenv("ORK_F16OUT_40c0")?strtoul(getenv("ORK_F16OUT_40c0"),0,0):0x00000040; /* 2-byte elem */
    unsigned r84=getenv("ORK_F16OUT_4084")?strtoul(getenv("ORK_F16OUT_4084"),0,0):0x00000001; /* gain */
    unsigned r88=getenv("ORK_F16OUT_4088")?strtoul(getenv("ORK_F16OUT_4088"),0,0):0x00000000; /* shift */
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_S_POINTER,r04);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_PRECISION,r10);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_DATA_CUBE_NOTCH,(((N/4)-1)<<16)|((N/4)-1));
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_DST_N_DIMS,((N-1)<<16)|(N-1));
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_BS_OW_CFG,r50);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_DST_N2,N-1);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_OFFSET,0);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_SCALE,r84);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_OUT_CVT_SHIFT,r88);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_SURFACE_ADD,rc0);
    orki_setrn(rc,REGCMD_I8_N,RK_DPU_R40C4,0);
}

void orki_set_f16_out_fp16in(uint32_t*rc,int M,int N){
    /* fp16-in fp16-out DPU output stage. Precision/bypass/CVT regs from a captured VENDOR fp16->fp16 MATMUL
     * (~/rknn_sdk/cap_fp16f16.dec): 0x4040=0x53 (BS FULLY bypassed — a MATMUL, not the conv's BS-active 0x20150
     * that hung it), 0x4010 OUT=fp16, 0x4084 FP32TOFP16_EN(bit16). The vendor's own capture was CONTIGUOUS
     * (0x4024=0x10, 0x40c0=0x20) — but to CHAIN into the per-channel SDP we instead emit the ATOM-8 cube the
     * SDP reads (orki_set_mul_geom layout: surface stride M*16, PC16/EWCUBEH). So the geometry here mirrors
     * orki_set_mul_geom's OUTPUT side (0x4024/0x4030/0x403c/0x4058/0x405c/0x40c0 = M*16 / M-1 / N), overriding synth's
     * contiguous fp32-atom geometry. M param needed for the M*16 surface stride. */
    /* Precision/bypass/CVT regs from the captured vendor fp16->fp16 MATMUL (cap_fp16f16.dec). */
    orki_setrn(rc,REGCMD_N,RK_DPU_S_POINTER,0x0000000e);
    orki_setrn(rc,REGCMD_N,RK_DPU_FEATURE_MODE_CFG,0x000001e4);                       /* FEATURE_MODE OUTPUT_MODE=2 */
    orki_setrn(rc,REGCMD_N,RK_DPU_OUT_PRECISION,0x48000002);                       /* DATA_FORMAT fp16/fp16/fp16 */
    orki_setrn(rc,REGCMD_N,RK_DPU_BS_CFG,0x00000053);                       /* BS FULLY BYPASSED (matmul, NOT conv's 0x20150) */
    orki_setrn(rc,REGCMD_N,RK_DPU_BS_OW_CFG,0x00000126);                       /* BS_OW_CFG */
    orki_setrn(rc,REGCMD_N,RK_DPU_BN_CFG,0x00000053);                       /* BN fully bypassed */
    orki_setrn(rc,REGCMD_N,RK_DPU_EW_CFG,0x00000383);                       /* EW fully bypassed */
    orki_setrn(rc,REGCMD_N,RK_DPU_EW_CVT_SCALE,0x00000001);
    orki_setrn(rc,REGCMD_N,RK_DPU_OUT_CVT_OFFSET,0x00000000);
    orki_setrn(rc,REGCMD_N,RK_DPU_OUT_CVT_SCALE,0x00010001);                       /* OUT_CVT_SCALE: FP32TOFP16_EN=1 | scale=1  <-- KEY */
    orki_setrn(rc,REGCMD_N,RK_DPU_OUT_CVT_SHIFT,0x00000000);
    orki_setrn(rc,REGCMD_N,RK_DPU_R40C4,0x00000000);
    if(getenv("ORK_F16_ATOM8")){
        /* ATOM-8 fp16 output — geometry taken EXACTLY from the bit-exact atom-8 SDP (REGCMD_MUL_F16 / orki_set_mul_geom):
         * surface stride M*16, 0x4050 BS_OW_CFG=0x2 (OD_BYPASS, SIZE_E=0 — NOT the contiguous 0x126 or the int16
         * 0x248 that hung), 0x4038=0. This makes the fp16 matmul emit the PC16 atom-8 layout the SDP reads, so
         * ork_npu_mul_perchan_f16 can consume it directly (no reshape, no notch). */
        uint32_t s=(uint32_t)(M*16);
        orki_setrn(rc,REGCMD_N,RK_DPU_DST_SURF_STRIDE,s);                            /* DST_SURF_STRIDE = M*16 */
        orki_setrn(rc,REGCMD_N,RK_DPU_DATA_CUBE_WIDTH,(uint32_t)(M-1));
        orki_setrn(rc,REGCMD_N,RK_DPU_DATA_CUBE_HEIGHT,0);
        orki_setrn(rc,REGCMD_N,RK_DPU_DATA_CUBE_NOTCH,0);
        orki_setrn(rc,REGCMD_N,RK_DPU_DST_N_DIMS,(uint32_t)(((N-1)<<16)|(N-1)));
        orki_setrn(rc,REGCMD_N,RK_DPU_BS_OW_CFG,0x00000002);                   /* BS_OW_CFG = 0x2 (atom-8; KEY) */
        orki_setrn(rc,REGCMD_N,RK_DPU_DST_N2,(uint32_t)(N-1));
        orki_setrn(rc,REGCMD_N,RK_DPU_WDMA_SIZE_1,(uint32_t)(M-1));
        orki_setrn(rc,REGCMD_N,RK_DPU_SURFACE_ADD,s);                            /* SURFACE_ADD = M*16 */
    } else {
        (void)M;                                                      /* CONTIGUOUS [M][N] (cap_fp16f16); PROVEN 512/512 standalone */
        orki_setrn(rc,REGCMD_N,RK_DPU_DATA_CUBE_NOTCH,(((N/8)-1)<<16)|((N/8)-1));
        orki_setrn(rc,REGCMD_N,RK_DPU_SURFACE_ADD,0x00000020);
    }
}

inline int orki_f16_mtile(int K,int M){
    double scale=(double)K/256.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale);
    int mg_max = base>=0x1b ? (base-0x1b)/slope+1 : 0; int chunk = mg_max*64;
    const char*e=getenv("ORK_F16_MTILE"); if(e){ int v=atoi(e); if(v>0)chunk=v; }
    if(chunk<1)chunk=1; if(chunk>M)chunk=M; return chunk;
}

/* RE fuzzer hook for fp16 (batch-mode mapping): overrides applied at the END of orki_synth(). Inert by default. */
/* fp16 twin of fused_mtile: the fp16 0x1040 K-reduction schedule (orki_synth() uses scale=K/256, vs int8's K/512
 * since int8 packs 2 rows per CBUF slot) gives the SAME bit-exact M-tile ceiling mg_max*64. The old
 * ork_mm_run_f16_silu chunk=16 was a stale over-conservative cap far below this (64 @K2048, 320 @K512) —
 * bit-exact validated (tools/silu_f16_check: M-tile 16==32==64 @K2048, 16==320 @K512, 384>ceil DIFFERS).
 * ORK_F16_MTILE overrides (validation / probing above the ceiling). */
/* RE (fp16 batch-mode mapping): raw fp32 output of one fp16 matmul via orki_synth(). Weight tile [NT][KT][16][32]
 * (N-tile=16), A raw-copied [M][K] fp16, output fp32 (2*M*N floats, room for a batch layout). A/B are fp16
 * bit patterns (uint16). ork_f16_fuzz overrides apply inside orki_synth(). 0/ok -1 wedged -2 dims. */
void orki_synth(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC,int sched,int cbuf){
    memcpy(rc,REGCMD,REGCMD_N*4);
    orki_setrn(rc,REGCMD_N,RK_CNA_DATA_SIZE1,((K-1)<<16)|K);orki_setrn(rc,REGCMD_N,RK_CNA_WEIGHT_SIZE0,K*N*2);orki_setrn(rc,REGCMD_N,RK_CNA_WEIGHT_SIZE1,K*2);
    orki_setrn(rc,REGCMD_N,RK_CNA_CBUF_CON1,K/32);orki_setrn(rc,REGCMD_N,RK_CNA_FC_DATA_SIZE1,K);orki_setrn(rc,REGCMD_N,RK_CNA_DMA_CON1,K/8);
    orki_setrn(rc,REGCMD_N,RK_CNA_DATA_SIZE0,0x10000|mc);orki_setrn(rc,REGCMD_N,RK_CNA_DATA_SIZE0_MIR,0x10000|mc);orki_setrn(rc,REGCMD_N,RK_CNA_DATA_SIZE3,mc);
    orki_setrn(rc,REGCMD_N,RK_DPU_DATA_CUBE_HEIGHT,mc-1);orki_setrn(rc,REGCMD_N,RK_DPU_WDMA_SIZE_1,(mc-1)<<16);orki_setrn(rc,REGCMD_N,RK_PDP_OUT_M,(mc-1)<<16);
    orki_setrn(rc,REGCMD_N,RK_DPU_DST_N_DIMS,((N-1)<<16)|(N-1));orki_setrn(rc,REGCMD_N,RK_DPU_DST_N2,N-1);orki_setrn(rc,REGCMD_N,RK_DPU_DATA_CUBE_NOTCH,(((N/4)-1)<<16)|((N/4)-1));
    orki_setrn(rc,REGCMD_N,RK_CNA_WEIGHT_SIZE2,0x1010000|N);orki_setrn(rc,REGCMD_N,RK_PDP_OUT_N,N-1);
    if(sched){
        int R=cbuf/K; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; } int rows=(mc+1<R)?(mc+1):R; orki_setrn(rc,REGCMD_N,RK_CNA_CONV_CON2,16*rows);
        double scale=(double)K/256.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale),mg=(mc+63)/64; if(mg<1)mg=1;  /* ceil: see orki_synth_i8 (65..127-row tile needs the next K-schedule) */
        int v=base-slope*(mg-1); if(v<0x1b)v=0x1b; orki_setrn(rc,REGCMD_N,RK_CNA_CBUF_CON0,v);
    } else { orki_setrn(rc,REGCMD_N,RK_CNA_CONV_CON2,16*(mc+1)); }
    orki_setrn(rc,REGCMD_N,RK_CNA_FEATURE_DATA_ADDR,aA);orki_setrn(rc,REGCMD_N,RK_CNA_WEIGHT_DATA_ADDR,aB);orki_setrn(rc,REGCMD_N,RK_DPU_DST_BASE_ADDR,aC);
    for(int i=0;i<orki_f16_fovr_n;i++) orki_setr(rc,REGCMD_N,orki_f16_fovr[i].blk,orki_f16_fovr[i].reg,orki_f16_fovr[i].val);  /* RE fuzzer overrides (raw/unbounded) */
}
