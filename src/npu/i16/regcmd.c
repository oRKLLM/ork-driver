/* npu/i16/regcmd.c — int16 regcmd synthesis and output stage.
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

static void orki_synth_i16(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC,int sched,int cbuf){
    orki_f16_synth(rc,mc,K,N,aA,aB,aC,sched,cbuf);                 /* fp16 2-byte geometry base */
    uint32_t con1=0x20000090u; const char*e=getenv("ORK_I16_CON1"); if(e) con1=(uint32_t)strtoul(e,NULL,0);
    orki_setrn(rc,REGCMD_N,RK_CNA_CONV_CON1,con1);             /* FP16(2)->INT16(1) precision */
}

void orki_i16_set_out(uint32_t*rc,int N,int stride,int mult,int shift){
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
