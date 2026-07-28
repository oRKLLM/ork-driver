/* ork_regs.h — ork's named NPU regcmd register layer, CROSS-REFERENCED to (not copied from) the mainline accel/rocket driver
 * (rocket_registers.h) + RK3588 TRM ch.36 / NVDLA. The regcmd synth code sets registers BY NAME via
 * SETRN(), which validates the value against the register's defined bit-width (a value with bits outside
 * the register's fields is a bug — caught here, not silently truncated on silicon). Raw/unbounded writes
 * for reverse-engineering go through SETRAW() (explicit block+offset+value, no bounds).
 *
 * This is the naming layer for the register-remap refactor (task #40). Each row: {block, offset, used-bit
 * mask, name}. The mask is the union of the register's documented fields — the "bounded range". Registers
 * not yet cross-referenced carry OKR_ANY (0xffffffff) and a TODO name; tighten as rocket names are confirmed.
 *
 * Blocks (regcmd (block,offset) pairs): 0x0201 CNA (conv/matmul input), 0x1001 DPU (output stage),
 * 0x0801 PDP/aux output dims, 0x0101 PC (program-chain). See wiki regcmd-ISA-Reference.
 */
#ifndef ORK_REGS_H
#define ORK_REGS_H
#include <stdint.h>

#define OKR_ANY 0xffffffffu

typedef struct { uint16_t blk; uint16_t off; uint32_t mask; const char *name; } ork_reg_desc;

/* Symbolic register IDs. Order is arbitrary; the descriptor table is indexed by these. */
enum ork_reg_id {
    /* ── CNA block 0x0201 (conv/matmul input) ───────────────────────────────────────── */
    RK_CNA_CONV_CON1,       /* 0x100c CONV_MODE/IN_PRECISION/PROC_PRECISION/NONALIGN_DMA */
    RK_CNA_CONV_CON2,       /* 0x1010 FEATURE_GRAINS[13:4]/KERNEL_GROUP[23:16] */
    RK_CNA_DATA_SIZE0,      /* 0x1020 DATAIN_WIDTH[26:16]/DATAIN_HEIGHT[10:0] */
    RK_CNA_DATA_SIZE1,      /* 0x1024 DATAIN_CHANNEL[15:0]/DATAIN_CHANNEL_REAL[29:16] (=K) */
    RK_CNA_DATA_SIZE_BATCH, /* 0x1028 batch/atomics (=M in the fold) */
    RK_CNA_DATA_SIZE3,      /* 0x102c DATAOUT_ATOMICS[21:0]/SURF_MODE[23:22] */
    RK_CNA_WEIGHT_SIZE0,    /* 0x1030 weight byte count */
    RK_CNA_WEIGHT_SIZE1,    /* 0x1034 bytes per kernel (=K int8) */
    RK_CNA_WEIGHT_SIZE2,    /* 0x1038 dims (0x1010000|N) */
    RK_CNA_CBUF_CON0,       /* 0x1040 DATA_BANK[3:0]/WEIGHT_BANK[7:4]/FC_DATA_BANK[10:8]/REUSE bits */
    RK_CNA_CBUF_CON1,       /* 0x1044 DATA_ENTRIES[13:0] */
    RK_CNA_FEATURE_DATA_ADDR,/* 0x1070 input feature IOVA */
    RK_CNA_DMA_CON1,        /* 0x107c burst length / line stride */
    RK_CNA_DMA_CON2,        /* 0x1080 SURF_STRIDE[27:0] (8-byte units) */
    RK_CNA_DATA_SIZE0_MIR,  /* 0x1084 CNA_DATA_SIZE0 mirror */
    RK_CNA_FC_DATA_SIZE1,   /* 0x1088 DMA_CHANNEL[15:0] (=K, FC/matmul) */
    RK_CNA_WEIGHT_DATA_ADDR,/* 0x1110 weight IOVA */
    /* ── DPU block 0x1001 (output stage) ─────────────────────────────────────────────── */
    RK_DPU_OUT_PRECISION,   /* 0x4010 precision/mode (int8->i32 0x80000000, etc.) */
    RK_DPU_DST_BASE_ADDR,   /* 0x4020 output C IOVA */
    RK_DPU_DST_SURF_STRIDE, /* 0x4024 DST_SURF_STRIDE[31:4] */
    RK_DPU_DATA_CUBE_WIDTH, /* 0x4030 WIDTH[12:0] */
    RK_DPU_DATA_CUBE_HEIGHT,/* 0x4034 HEIGHT[12:0]/MINMAX_CTL[24:22] */
    RK_DPU_DATA_CUBE_NOTCH, /* 0x4038 NOTCH_ADDR_0[12:0]/NOTCH_ADDR_1[28:16] */
    RK_DPU_DST_N_DIMS,      /* 0x403c output N dims ((s-1)<<16)|(N-1) */
    RK_DPU_DST_N2,          /* 0x4058 N-1 */
    RK_DPU_WDMA_SIZE_1,     /* 0x405c WIDTH_WDMA[12:0]/HEIGHT_WDMA[28:16] */
    RK_DPU_SURFACE_ADD,     /* 0x40c0 SURF_ADD[31:4] — ABSOLUTE output IOVA */
    /* ── PDP/aux block 0x0801 (output dims mirror) ───────────────────────────────────── */
    RK_PDP_OUT_M,           /* 0x3014 output M dim ((M-1)<<16) */
    RK_PDP_OUT_N,           /* 0x3018 output N dim (N-1) */
    /* ── PC block 0x0101 (program chaining) ──────────────────────────────────────────── */
    RK_PC_NEXT_ADDR,        /* 0x0010 next regcmd IOVA (0=end) */
    RK_PC_NEXT_AMOUNT,      /* 0x0014 next register-amount */
    /* ── DPU output-stage (block 0x1001) — bias(BS)/batchnorm(BN)/elementwise(EW)/requant(OUT_CVT) ─── */
    RK_DPU_S_POINTER,            /* 4004 DPU_S_POINTER */
    RK_DPU_FEATURE_MODE_CFG,     /* 400c DPU_FEATURE_MODE_CFG */
    RK_DPU_BS_CFG,               /* 4040 DPU_BS_CFG */
    RK_DPU_BS_ALU_CFG,           /* 4044 DPU_BS_ALU_CFG */
    RK_DPU_BS_MUL_CFG,           /* 4048 DPU_BS_MUL_CFG */
    RK_DPU_BS_OW_CFG,            /* 4050 DPU_BS_OW_CFG */
    RK_DPU_BN_CFG,               /* 4060 DPU_BN_CFG */
    RK_DPU_BN_ALU_CFG,           /* 4064 DPU_BN_ALU_CFG */
    RK_DPU_BN_MUL_CFG,           /* 4068 DPU_BN_MUL_CFG */
    RK_DPU_EW_CFG,               /* 4070 DPU_EW_CFG */
    RK_DPU_EW_CVT_OFFSET,        /* 4074 DPU_EW_CVT_OFFSET_VALUE */
    RK_DPU_EW_CVT_SCALE,         /* 4078 DPU_EW_CVT_SCALE_VALUE */
    RK_DPU_OUT_CVT_OFFSET,       /* 4080 DPU_OUT_CVT_OFFSET */
    RK_DPU_OUT_CVT_SCALE,        /* 4084 DPU_OUT_CVT_SCALE */
    RK_DPU_OUT_CVT_SHIFT,        /* 4088 DPU_OUT_CVT_SHIFT */
    RK_DPU_EW_OP_VALUE_0,        /* 4090 DPU_EW_OP_VALUE_0 */
    RK_DPU_R40C4,                /* 40c4 inferred (not in rocket hdr) */
    RK_DPU_R4108,                /* 4108 inferred */
    RK_DPU_R410C,                /* 410c inferred */
    RK_DPU_R4110,                /* 4110 inferred */
    RK_DPU_R411C,                /* 411c inferred */
    RK_DPU_R4128,                /* 4128 inferred */
    RK_DPU_R412C,                /* 412c inferred */
    RK_PC_OPERATION_ENABLE,      /* 0081:0008 PC_OPERATION_ENABLE */
    /* ── SDP/activation block 0x2001 (0x50xx) — silu/gelu/ewmul/requant op regs (inferred by offset) ─── */
    RK_SDP_5004, RK_SDP_5008, RK_SDP_500C, RK_SDP_5010, RK_SDP_5014, RK_SDP_5018, RK_SDP_501C, RK_SDP_5020,
    RK_SDP_5028, RK_SDP_5034, RK_SDP_5038, RK_SDP_5040, RK_SDP_5044, RK_SDP_5048, RK_SDP_504C, RK_SDP_506C,
    RK_REG__COUNT
};

/* Descriptor table. mask = union of the register's documented field bits (the bounded range). Address and
 * fully-packed registers use OKR_ANY (any 32-bit is legal). Keep in enum order. */
static const ork_reg_desc ORK_REGS[RK_REG__COUNT] = {
    [RK_CNA_CONV_CON1]        = {0x201, 0x100c, OKR_ANY,     "CNA_CONV_CON1"},
    [RK_CNA_CONV_CON2]        = {0x201, 0x1010, 0x00ff3fff,  "CNA_CONV_CON2"},     /* FEATURE_GRAINS|KERNEL_GROUP */
    [RK_CNA_DATA_SIZE0]       = {0x201, 0x1020, 0x07ff07ff,  "CNA_DATA_SIZE0"},    /* WIDTH[26:16]|HEIGHT[10:0] */
    [RK_CNA_DATA_SIZE1]       = {0x201, 0x1024, 0x3fffffff,  "CNA_DATA_SIZE1"},    /* CHANNEL|CHANNEL_REAL */
    [RK_CNA_DATA_SIZE_BATCH]  = {0x201, 0x1028, OKR_ANY,     "CNA_DATA_SIZE_BATCH"},
    [RK_CNA_DATA_SIZE3]       = {0x201, 0x102c, 0x00ffffff,  "CNA_DATA_SIZE3"},    /* DATAOUT_ATOMICS[21:0]|SURF_MODE */
    [RK_CNA_WEIGHT_SIZE0]     = {0x201, 0x1030, OKR_ANY,     "CNA_WEIGHT_SIZE0"},
    [RK_CNA_WEIGHT_SIZE1]     = {0x201, 0x1034, OKR_ANY,     "CNA_WEIGHT_SIZE1"},
    [RK_CNA_WEIGHT_SIZE2]     = {0x201, 0x1038, OKR_ANY,     "CNA_WEIGHT_SIZE2"},
    [RK_CNA_CBUF_CON0]        = {0x201, 0x1040, 0x00003fff,  "CNA_CBUF_CON0"},     /* banks + reuse bits */
    [RK_CNA_CBUF_CON1]        = {0x201, 0x1044, 0x00003fff,  "CNA_CBUF_CON1"},     /* DATA_ENTRIES[13:0] */
    [RK_CNA_FEATURE_DATA_ADDR]= {0x201, 0x1070, OKR_ANY,     "CNA_FEATURE_DATA_ADDR"},
    [RK_CNA_DMA_CON1]         = {0x201, 0x107c, OKR_ANY,     "CNA_DMA_CON1"},
    [RK_CNA_DMA_CON2]         = {0x201, 0x1080, 0x0fffffff,  "CNA_DMA_CON2"},      /* SURF_STRIDE[27:0] */
    [RK_CNA_DATA_SIZE0_MIR]   = {0x201, 0x1084, 0x07ff07ff,  "CNA_DATA_SIZE0_MIR"},
    [RK_CNA_FC_DATA_SIZE1]    = {0x201, 0x1088, 0x0000ffff,  "CNA_FC_DATA_SIZE1"}, /* DMA_CHANNEL[15:0] */
    [RK_CNA_WEIGHT_DATA_ADDR] = {0x201, 0x1110, OKR_ANY,     "CNA_WEIGHT_DATA_ADDR"},
    [RK_DPU_OUT_PRECISION]    = {0x1001, 0x4010, OKR_ANY,    "DPU_OUT_PRECISION"},
    [RK_DPU_DST_BASE_ADDR]    = {0x1001, 0x4020, OKR_ANY,    "DPU_DST_BASE_ADDR"},
    [RK_DPU_DST_SURF_STRIDE]  = {0x1001, 0x4024, OKR_ANY,    "DPU_DST_SURF_STRIDE"},
    [RK_DPU_DATA_CUBE_WIDTH]  = {0x1001, 0x4030, 0x00001fff, "DPU_DATA_CUBE_WIDTH"},
    [RK_DPU_DATA_CUBE_HEIGHT] = {0x1001, 0x4034, 0x01ffffff, "DPU_DATA_CUBE_HEIGHT"},
    [RK_DPU_DATA_CUBE_NOTCH]  = {0x1001, 0x4038, 0x1fff1fff, "DPU_DATA_CUBE_NOTCH"},
    [RK_DPU_DST_N_DIMS]       = {0x1001, 0x403c, OKR_ANY,    "DPU_DST_N_DIMS"},
    [RK_DPU_DST_N2]           = {0x1001, 0x4058, OKR_ANY,    "DPU_DST_N2"},
    [RK_DPU_WDMA_SIZE_1]      = {0x1001, 0x405c, 0x1fff1fff, "DPU_WDMA_SIZE_1"},
    [RK_DPU_SURFACE_ADD]      = {0x1001, 0x40c0, OKR_ANY,    "DPU_SURFACE_ADD"},
    [RK_PDP_OUT_M]            = {0x801, 0x3014, OKR_ANY,     "PDP_OUT_M"},
    [RK_PDP_OUT_N]            = {0x801, 0x3018, OKR_ANY,     "PDP_OUT_N"},
    [RK_PC_NEXT_ADDR]         = {0x101, 0x0010, OKR_ANY,     "PC_NEXT_ADDR"},
    [RK_PC_NEXT_AMOUNT]       = {0x101, 0x0014, OKR_ANY,     "PC_NEXT_AMOUNT"},
    [RK_DPU_S_POINTER]        = {0x1001, 0x4004, OKR_ANY,    "DPU_S_POINTER"},
    [RK_DPU_FEATURE_MODE_CFG] = {0x1001, 0x400c, OKR_ANY,    "DPU_FEATURE_MODE_CFG"},
    [RK_DPU_BS_CFG]           = {0x1001, 0x4040, OKR_ANY,    "DPU_BS_CFG"},
    [RK_DPU_BS_ALU_CFG]       = {0x1001, 0x4044, OKR_ANY,    "DPU_BS_ALU_CFG"},
    [RK_DPU_BS_MUL_CFG]       = {0x1001, 0x4048, OKR_ANY,    "DPU_BS_MUL_CFG"},
    [RK_DPU_BS_OW_CFG]        = {0x1001, 0x4050, OKR_ANY,    "DPU_BS_OW_CFG"},
    [RK_DPU_BN_CFG]           = {0x1001, 0x4060, OKR_ANY,    "DPU_BN_CFG"},
    [RK_DPU_BN_ALU_CFG]       = {0x1001, 0x4064, OKR_ANY,    "DPU_BN_ALU_CFG"},
    [RK_DPU_BN_MUL_CFG]       = {0x1001, 0x4068, OKR_ANY,    "DPU_BN_MUL_CFG"},
    [RK_DPU_EW_CFG]           = {0x1001, 0x4070, OKR_ANY,    "DPU_EW_CFG"},
    [RK_DPU_EW_CVT_OFFSET]    = {0x1001, 0x4074, OKR_ANY,    "DPU_EW_CVT_OFFSET"},
    [RK_DPU_EW_CVT_SCALE]     = {0x1001, 0x4078, OKR_ANY,    "DPU_EW_CVT_SCALE"},
    [RK_DPU_OUT_CVT_OFFSET]   = {0x1001, 0x4080, OKR_ANY,    "DPU_OUT_CVT_OFFSET"},
    [RK_DPU_OUT_CVT_SCALE]    = {0x1001, 0x4084, OKR_ANY,    "DPU_OUT_CVT_SCALE"},
    [RK_DPU_OUT_CVT_SHIFT]    = {0x1001, 0x4088, OKR_ANY,    "DPU_OUT_CVT_SHIFT"},
    [RK_DPU_EW_OP_VALUE_0]    = {0x1001, 0x4090, OKR_ANY,    "DPU_EW_OP_VALUE_0"},
    [RK_DPU_R40C4]            = {0x1001, 0x40c4, OKR_ANY,    "DPU_R40C4"},
    [RK_DPU_R4108]            = {0x1001, 0x4108, OKR_ANY,    "DPU_R4108"},
    [RK_DPU_R410C]            = {0x1001, 0x410c, OKR_ANY,    "DPU_R410C"},
    [RK_DPU_R4110]            = {0x1001, 0x4110, OKR_ANY,    "DPU_R4110"},
    [RK_DPU_R411C]            = {0x1001, 0x411c, OKR_ANY,    "DPU_R411C"},
    [RK_DPU_R4128]            = {0x1001, 0x4128, OKR_ANY,    "DPU_R4128"},
    [RK_DPU_R412C]            = {0x1001, 0x412c, OKR_ANY,    "DPU_R412C"},
    [RK_PC_OPERATION_ENABLE]  = {0x81,   0x0008, OKR_ANY,    "PC_OPERATION_ENABLE"},
    [RK_SDP_5004]={0x2001,0x5004,OKR_ANY,"SDP_5004"},[RK_SDP_5008]={0x2001,0x5008,OKR_ANY,"SDP_5008"},
    [RK_SDP_500C]={0x2001,0x500c,OKR_ANY,"SDP_500C"},[RK_SDP_5010]={0x2001,0x5010,OKR_ANY,"SDP_5010"},
    [RK_SDP_5014]={0x2001,0x5014,OKR_ANY,"SDP_5014"},[RK_SDP_5018]={0x2001,0x5018,OKR_ANY,"SDP_5018"},
    [RK_SDP_501C]={0x2001,0x501c,OKR_ANY,"SDP_501C"},[RK_SDP_5020]={0x2001,0x5020,OKR_ANY,"SDP_5020"},
    [RK_SDP_5028]={0x2001,0x5028,OKR_ANY,"SDP_5028"},[RK_SDP_5034]={0x2001,0x5034,OKR_ANY,"SDP_5034"},
    [RK_SDP_5038]={0x2001,0x5038,OKR_ANY,"SDP_5038"},[RK_SDP_5040]={0x2001,0x5040,OKR_ANY,"SDP_5040"},
    [RK_SDP_5044]={0x2001,0x5044,OKR_ANY,"SDP_5044"},[RK_SDP_5048]={0x2001,0x5048,OKR_ANY,"SDP_5048"},
    [RK_SDP_504C]={0x2001,0x504c,OKR_ANY,"SDP_504C"},[RK_SDP_506C]={0x2001,0x506c,OKR_ANY,"SDP_506C"},
};

/* ══ Named register VALUES + field composers (the "recipe" layer) ═══════════════════════════════════
 * setrn() names the REGISTER; these name the VALUE. Set a mode-like register by named intent
 * (setrn(rc,n,REG,OKV_*)) and compose packed fields with the OKC_* macros — so a config is a named
 * choice, not a magic hex constant. Directly motivated by the M-fold RE, where every stall/wedge traced
 * to a misread magic value: 0x40c0 as an IOVA (it's an elem-size config), 0x100c's int32-mode bit left on
 * (stalls the fold), 0x1080 as a stride (it's a "contiguous" sentinel). (C enums can't hold 32-bit values
 * like 0x80000000 cleanly, so these are typed uint32_t constants grouped by the register they belong to.) */

/* RK_DPU_OUT_PRECISION (0x4010) — output/accumulate datapath precision */
#define OKV_OUT_PREC_INT8      0x00000000u
#define OKV_OUT_PREC_INT32     0x80000000u

/* RK_CNA_CONV_CON1 (0x100c) — CNA datapath mode. The 0x20000000 bit selects the standard int32-output
 * datapath; the M-fold schedule REQUIRES it cleared (with it set, the folded op STALLS). */
#define OKV_CONV1_STD          0x20000000u   /* standard (non-folded) matmul */
#define OKV_CONV1_MFOLD        0x00000000u   /* M-fold schedule */

/* RK_DPU_SURFACE_ADD (0x40c0) — element/surface-size CONFIG (NOT an address; do not patch as an IOVA). */
#define OKV_ELEMSZ_INT8        0x00000020u
#define OKV_ELEMSZ_INT32       0x00000080u
#define OKV_SURFADD_MFOLD      0x00000400u   /* M-fold NC1HWC2 surface config (from rkllm's task3) */

/* RK_CNA_DMA_CON2 (0x1080) — input surface stride, OR this "read contiguous / no re-stride" sentinel,
 * which the M-fold needs to reduce FULL K (a computed stride gives a partial reduction). */
#define OKV_DMA2_CONTIGUOUS    0x0fffffe8u

/* Packed-field composers. */
#define OKC_CBUF_CON0(data_bank, weight_bank) ((uint32_t)((((weight_bank)&0xf)<<4) | ((data_bank)&0xf)))  /* RK_CNA_CBUF_CON0 0x1040 */
#define OKC_DATA_SIZE0(width, height)         ((uint32_t)((((width)&0x7ff)<<16) | ((height)&0x7ff)))       /* RK_CNA_DATA_SIZE0 0x1020/0x1084 (DATAIN_WIDTH|HEIGHT) */

#endif /* ORK_REGS_H */
