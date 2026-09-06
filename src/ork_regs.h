/**
 * @file ork_regs.h
 * @brief ork's named NPU regcmd register layer — the naming layer for the register-remap refactor (task #40).
 *
 * CROSS-REFERENCED to (not copied from) the mainline accel/rocket driver (rocket_registers.h) + RK3588 TRM
 * ch.36 / NVDLA. The regcmd synth code sets registers BY NAME via @c setrn(), which validates the value
 * against the register's defined bit-width (a value with bits outside the register's fields is a bug —
 * caught here, not silently truncated on silicon). Raw/unbounded writes for reverse-engineering go through
 * @c setr() (explicit block+offset+value, no bounds).
 *
 * Each row of #ORK_REGS is an #ork_reg_desc: {block, offset, used-bit mask, name, description}. The mask is
 * the union of the register's documented fields — the "bounded range". Registers not yet cross-referenced
 * carry #OKR_ANY (0xffffffff) and an inferred name; tighten as rocket names are confirmed.
 *
 * The value/recipe layer below (@c OKV_* named values, @c OKC_* field composers) names the VALUE a register
 * is set to, so a config is a named choice rather than a magic constant. @c tools/re/regcmd_decode consumes
 * both layers to annotate a captured regcmd by register name, value name, and this description.
 *
 * @par Blocks (regcmd (block,offset) pairs)
 *   0x0201 CNA (conv/matmul input), 0x1001 DPU (output stage), 0x0801 PDP/aux output dims,
 *   0x0101 PC (program-chain), 0x2001 SDP (activation). See wiki regcmd-ISA-Reference.
 */
#ifndef ORK_REGS_H
#define ORK_REGS_H
#include <stdint.h>

#define OKR_ANY 0xffffffffu   /**< @brief field mask sentinel: any 32-bit value is legal (address / fully-packed / unconfirmed register). */

/**
 * @struct ork_reg_desc
 * @brief One row of the register-name table: how to locate a regcmd register and what it means.
 * @var ork_reg_desc::blk   regcmd block id (0x0201 CNA, 0x1001 DPU, 0x0801 PDP/aux, 0x0101 PC, 0x2001 SDP)
 * @var ork_reg_desc::off   register offset within the block
 * @var ork_reg_desc::mask  union of the register's documented field bits (the bounded range; OKR_ANY = any 32-bit)
 * @var ork_reg_desc::name  short symbolic name (rocket-cross-referenced where confirmed)
 * @var ork_reg_desc::desc  one-line human description; overlaid next to each decode by tools/re/regcmd_decode.
 */
typedef struct { uint16_t blk; uint16_t off; uint32_t mask; const char *name; const char *desc; } ork_reg_desc;

/**
 * @enum ork_reg_id
 * @brief Symbolic register IDs; the ORK_REGS[] descriptor table is indexed by these. Order is arbitrary.
 *        The @c \@brief on each member is mirrored into that register's ORK_REGS[].desc string.
 */
enum ork_reg_id {
    /* ── CNA block 0x0201 (conv/matmul input) ───────────────────────────────────────── */
    RK_CNA_CONV_CON1,       /**< 0x100c @brief CNA_CONV_CON1: NONALIGN_DMA[30]/GROUP_LINE_OFF[29]/DECONV[16]/PROC_PRECISION[9:7]/IN_PRECISION[6:4]/CONV_MODE[3:0]. 0x20000000=GROUP_LINE_OFF (M-fold read). */
    RK_CNA_CONV_CON2,       /**< 0x1010 @brief FEATURE_GRAINS[13:4]/KERNEL_GROUP[23:16]: atoms fetched per pass (perf hint). */
    RK_CNA_DATA_SIZE0,      /**< 0x1020 @brief Input feature cube DATAIN_WIDTH[26:16]/DATAIN_HEIGHT[10:0]; the M-fold puts M in WIDTH, HEIGHT=1. */
    RK_CNA_DATA_SIZE1,      /**< 0x1024 @brief Reduction channel count DATAIN_CHANNEL[15:0]/CHANNEL_REAL[29:16] (= K). */
    RK_CNA_DATA_SIZE_BATCH, /**< 0x1028 @brief DATAIN batch / atomics count (= M rows in the fold). */
    RK_CNA_DATA_SIZE3,      /**< 0x102c @brief Output DATAOUT_ATOMICS[21:0]/SURF_MODE[23:22] (= M). */
    RK_CNA_WEIGHT_SIZE0,    /**< 0x1030 @brief Total weight byte count (K*N int8). */
    RK_CNA_WEIGHT_SIZE1,    /**< 0x1034 @brief Weight bytes per kernel / K stride (= K int8). */
    RK_CNA_WEIGHT_SIZE2,    /**< 0x1038 @brief Weight cube dims (kernel count N + flags, 0x1010000|N). */
    RK_CNA_CBUF_CON0,       /**< 0x1040 @brief CBUF bank split DATA_BANK[3:0]/WEIGHT_BANK[7:4]/FC_DATA_BANK[10:8]+reuse; also carries the K-reduction schedule. */
    RK_CNA_CBUF_CON1,       /**< 0x1044 @brief CBUF DATA_ENTRIES[13:0] = (K/64)*M feature rows held resident. */
    RK_CNA_FEATURE_DATA_ADDR,/**< 0x1070 @brief Input feature (A / activations) IOVA. */
    RK_CNA_DMA_CON1,        /**< 0x107c @brief Feature DMA burst length / line stride. */
    RK_CNA_DMA_CON2,        /**< 0x1080 @brief Feature surface stride, SIGNED 28-bit two's complement. w1-low is the sign extension (0xfff for negatives, 0x000 for positives — verified 0/4114 violations, NOT a mode field). Planner-chosen per tile (-3*M common at small M). The old "0x0fffffe8 sentinel" was just -24 read as unsigned. */
    RK_CNA_DATA_SIZE0_MIR,  /**< 0x1084 @brief Mirror of CNA_DATA_SIZE0 (WIDTH/HEIGHT); must equal 0x1020. */
    RK_CNA_FC_DATA_SIZE1,   /**< 0x1088 @brief FC/matmul DMA_CHANNEL[15:0] (= K). */
    RK_CNA_WEIGHT_DATA_ADDR,/**< 0x1110 @brief Weight (B) IOVA. */
    /* ── DPU block 0x1001 (output stage) ─────────────────────────────────────────────── */
    RK_DPU_OUT_PRECISION,   /**< 0x4010 @brief Output-stage datapath precision: 0=int8, 0x80000000=int32 accumulate-out. */
    RK_DPU_DST_BASE_ADDR,   /**< 0x4020 @brief Output (C) base IOVA. */
    RK_DPU_DST_SURF_STRIDE, /**< 0x4024 @brief Output DST_SURF_STRIDE[31:4] between NC1HWC2 surfaces. */
    RK_DPU_DATA_CUBE_WIDTH, /**< 0x4030 @brief Output cube WIDTH-1[12:0] (= M-1 in the fold). */
    RK_DPU_DATA_CUBE_HEIGHT,/**< 0x4034 @brief Output cube HEIGHT-1[12:0] / MINMAX_CTL[24:22]. */
    RK_DPU_DATA_CUBE_NOTCH, /**< 0x4038 @brief Output NC1HWC2 notch offsets NOTCH_ADDR_0[12:0]/1[28:16]; 0 = no notch. */
    RK_DPU_DST_N_DIMS,      /**< 0x403c @brief Output N dims: dst stride and N-1 (((s-1)<<16)|(N-1)). */
    RK_DPU_DST_N2,          /**< 0x4058 @brief Output N-1 (secondary). */
    RK_DPU_WDMA_SIZE_1,     /**< 0x405c @brief Write-DMA WIDTH_WDMA[12:0]/HEIGHT_WDMA[28:16] (-1 each). */
    RK_DPU_SURFACE_ADD,     /**< 0x40c0 @brief Output surface/element-size CONFIG (NOT an IOVA): 0x20 int8, 0x80 int32, 0x400 M-fold. */
    /* ── PDP/aux block 0x0801 (output dims mirror) ───────────────────────────────────── */
    RK_PDP_OUT_M,           /**< 0x3014 @brief PDP/aux output M dim ((M-1)<<16). */
    RK_PDP_OUT_N,           /**< 0x3018 @brief PDP/aux output N dim (N-1). */
    /* ── PC block 0x0101 (program chaining) ──────────────────────────────────────────── */
    RK_PC_NEXT_ADDR,        /**< 0x0010 @brief PC next-regcmd IOVA for hardware task chaining (0 = end of chain). */
    RK_PC_NEXT_AMOUNT,      /**< 0x0014 @brief PC next-task register-write count = (n+3)/2. */
    /* ── DPU output-stage (block 0x1001) — bias(BS)/batchnorm(BN)/elementwise(EW)/requant(OUT_CVT) ─── */
    RK_DPU_S_POINTER,            /**< 0x4004 @brief DPU surface/bank ping-pong pointer. */
    RK_DPU_FEATURE_MODE_CFG,     /**< 0x400c @brief DPU feature / flying-mode config. */
    RK_DPU_BS_CFG,               /**< 0x4040 @brief Bias (BS) stage config / bypass. */
    RK_DPU_BS_ALU_CFG,           /**< 0x4044 @brief Bias ALU op config. */
    RK_DPU_BS_MUL_CFG,           /**< 0x4048 @brief Bias multiply config. */
    RK_DPU_BS_OW_CFG,            /**< 0x4050 @brief Bias output-write config. */
    RK_DPU_BN_CFG,               /**< 0x4060 @brief Batch-norm (BN) stage config / bypass. */
    RK_DPU_BN_ALU_CFG,           /**< 0x4064 @brief BN ALU op config. */
    RK_DPU_BN_MUL_CFG,           /**< 0x4068 @brief BN multiply config. */
    RK_DPU_EW_CFG,               /**< 0x4070 @brief Elementwise (EW) stage config: EW_OP_TYPE/EW_OP_SRC (mul/add). */
    RK_DPU_EW_CVT_OFFSET,        /**< 0x4074 @brief EW input convert offset. */
    RK_DPU_EW_CVT_SCALE,         /**< 0x4078 @brief EW input convert scale. */
    RK_DPU_OUT_CVT_OFFSET,       /**< 0x4080 @brief Output requant convert offset. */
    RK_DPU_OUT_CVT_SCALE,        /**< 0x4084 @brief Output requant convert scale (multiplier). */
    RK_DPU_OUT_CVT_SHIFT,        /**< 0x4088 @brief Output requant convert shift. */
    RK_DPU_EW_OP_VALUE_0,        /**< 0x4090 @brief EW immediate operand 0. */
    RK_DPU_R40C4,                /**< 0x40c4 @brief Inferred DPU output-stage register (name unconfirmed). */
    RK_DPU_R4108,                /**< 0x4108 @brief Inferred DPU output-stage register (name unconfirmed). */
    RK_DPU_R410C,                /**< 0x410c @brief Inferred DPU output-stage register (name unconfirmed). */
    RK_DPU_R4110,                /**< 0x4110 @brief Inferred DPU output-stage register (name unconfirmed). */
    RK_DPU_R411C,                /**< 0x411c @brief Inferred DPU output-stage register (name unconfirmed). */
    RK_DPU_R4128,                /**< 0x4128 @brief Inferred DPU output-stage register (name unconfirmed). */
    RK_DPU_R412C,                /**< 0x412c @brief Inferred DPU output-stage register (name unconfirmed). */
    RK_PC_OPERATION_ENABLE,      /**< 0081:0008 @brief PC_OPERATION_ENABLE doorbell (kick the programmed op). */
    /* ── SDP/activation block 0x2001 (0x50xx) — silu/gelu/ewmul/requant op regs (inferred by offset) ─── */
    RK_SDP_5004, RK_SDP_5008, RK_SDP_500C, RK_SDP_5010, RK_SDP_5014, RK_SDP_5018, RK_SDP_501C, RK_SDP_5020,
    RK_SDP_5028, RK_SDP_5034, RK_SDP_5038, RK_SDP_5040, RK_SDP_5044, RK_SDP_5048, RK_SDP_504C, RK_SDP_506C,
    /* ── task #40: registers observed in rkllm's mfold capture, inferred (name unconfirmed) ────────── */
    RK_BLK0_R0000, RK_BLK41_R0000, RK_CNA_CONV_CON3, RK_CNA_CVT_CON0,
    RK_CNA_CVT_CON1, RK_CNA_CVT_CON2, RK_CNA_CVT_CON3, RK_CNA_CVT_CON4,
    RK_CNA_FC_CON0, RK_CNA_FC_CON1, RK_CNA_R1068, RK_CNA_FC_CON2,
    RK_CNA_DMA_CON0, RK_CNA_R1100, RK_CNA_R1104, RK_CNA_R1140,
    RK_CNA_R1144, RK_CNA_R1148, RK_CNA_R114C, RK_CNA_R1150,
    RK_CNA_R1154, RK_CNA_R1158, RK_CNA_R115C, RK_CNA_R1160,
    RK_CNA_R1164, RK_CNA_R1168, RK_CNA_R116C, RK_CNA_R1170,
    RK_CNA_R1174, RK_CNA_R1178, RK_CNA_R117C, RK_CNA_R1180,
    RK_CNA_R1184, RK_PDP_R3010, RK_PDP_R301C, RK_PDP_R3030,
    RK_DPU_R4014, RK_DPU_R404C, RK_DPU_R4054, RK_DPU_R406C,
    RK_DPU_R407C, RK_DPU_R4094, RK_DPU_R4098, RK_DPU_R409C,
    RK_DPU_R40A0, RK_DPU_R40A4, RK_DPU_R40A8, RK_DPU_R40AC,
    RK_DPU_R4100, RK_DPU_R4104, RK_DPU_R4114, RK_DPU_R4118,
    RK_DPU_R4120, RK_DPU_R4124,
    RK_REG__COUNT
};

/* Descriptor table. mask = union of the register's documented field bits (the bounded range). Address and
 * fully-packed registers use OKR_ANY (any 32-bit is legal). Keep in enum order. */
static const ork_reg_desc ORK_REGS[RK_REG__COUNT] = {
    [RK_CNA_CONV_CON1]        = {0x201, 0x100c, OKR_ANY,     "CNA_CONV_CON1",        "CNA_CONV_CON1: GROUP_LINE_OFF[29] (fold read) + precision[9:4] (int8=0) + CONV_MODE[3:0]"},
    [RK_CNA_CONV_CON2]        = {0x201, 0x1010, 0x00ff3fff,  "CNA_CONV_CON2",        "FEATURE_GRAINS[13:4]|KERNEL_GROUP[23:16]: atoms per pass (perf hint)"},
    [RK_CNA_DATA_SIZE0]       = {0x201, 0x1020, 0x07ff07ff,  "CNA_DATA_SIZE0",       "input cube WIDTH[26:16]|HEIGHT[10:0]; fold puts M in WIDTH, HEIGHT=1"},
    [RK_CNA_DATA_SIZE1]       = {0x201, 0x1024, 0x3fffffff,  "CNA_DATA_SIZE1",       "reduction CHANNEL[15:0]|CHANNEL_REAL[29:16] (= K)"},
    [RK_CNA_DATA_SIZE_BATCH]  = {0x201, 0x1028, OKR_ANY,     "CNA_DATA_SIZE_BATCH",  "DATAIN batch / atomics count (= M rows in the fold)"},
    [RK_CNA_DATA_SIZE3]       = {0x201, 0x102c, 0x00ffffff,  "CNA_DATA_SIZE3",       "output DATAOUT_ATOMICS[21:0]|SURF_MODE[23:22] (= M)"},
    [RK_CNA_WEIGHT_SIZE0]     = {0x201, 0x1030, OKR_ANY,     "CNA_WEIGHT_SIZE0",     "total weight byte count (K*N int8)"},
    [RK_CNA_WEIGHT_SIZE1]     = {0x201, 0x1034, OKR_ANY,     "CNA_WEIGHT_SIZE1",     "weight bytes per kernel / K stride (= K int8)"},
    [RK_CNA_WEIGHT_SIZE2]     = {0x201, 0x1038, OKR_ANY,     "CNA_WEIGHT_SIZE2",     "weight cube dims (kernel count N + flags, 0x1010000|N)"},
    [RK_CNA_CBUF_CON0]        = {0x201, 0x1040, 0x00003fff,  "CNA_CBUF_CON0",        "DATA_BANK[3:0] WEIGHT_BANK[7:4] FC_DATA_BANK[10:8] DATA_REUSE[12] WEIGHT_REUSE[13] (rocket_registers.h). rkllm leaves REUSE=0. 2026-09-06 SOLVED (tools/wr_diag.c): reuse is correct IFF K*segw <= WEIGHT_BANK capacity (banks x 32KB); correct_fraction = min(1, capacity/(K*segw)), verified at 1.0x (0.0%% wrong), 2x (49.8%%), 10.9x (89.3%%). The loader streams the whole weight but the banks retain only the LAST capacity bytes, so the reuse tile reads stale bytes beyond that -- deterministic, confined to reuse tiles, correct columns a contiguous run at the end. This is the vendor conv rule and it transfers; #39 observed an out-of-envelope case, not a hardware limit. Recoverable via N-tiling to fit the banks + widening WEIGHT_BANK here (4->11 banks); the widening is what makes it pay"},
    [RK_CNA_CBUF_CON1]        = {0x201, 0x1044, 0x00003fff,  "CNA_CBUF_CON1",        "CBUF DATA_ENTRIES[13:0] = (K/64)*M resident feature rows"},
    [RK_CNA_FEATURE_DATA_ADDR]= {0x201, 0x1070, OKR_ANY,     "CNA_FEATURE_DATA_ADDR","input feature (A / activations) IOVA"},
    [RK_CNA_DMA_CON1]         = {0x201, 0x107c, OKR_ANY,     "CNA_DMA_CON1",         "feature DMA burst length / line stride"},
    [RK_CNA_DMA_CON2]         = {0x201, 0x1080, 0x0fffffff,  "CNA_DMA_CON2",         "feature surface stride, SIGNED 28-bit two's complement (w1-low = sign extension: 0xfff for negatives, NOT a mode field — 0/4114 violations in capture); planner-chosen per tile (-3*M common at small M). The '0x0fffffe8 sentinel' was just -24 (=-3*8) read as unsigned"},
    [RK_CNA_DATA_SIZE0_MIR]   = {0x201, 0x1084, 0x07ff07ff,  "CNA_DATA_SIZE0_MIR",   "mirror of CNA_DATA_SIZE0 (WIDTH/HEIGHT); must equal 0x1020"},
    [RK_CNA_FC_DATA_SIZE1]    = {0x201, 0x1088, 0x0000ffff,  "CNA_FC_DATA_SIZE1",    "FC/matmul DMA_CHANNEL[15:0] (= K)"},
    [RK_CNA_WEIGHT_DATA_ADDR] = {0x201, 0x1110, OKR_ANY,     "CNA_WEIGHT_DATA_ADDR", "weight (B) IOVA"},
    [RK_DPU_OUT_PRECISION]    = {0x1001, 0x4010, OKR_ANY,    "DPU_OUT_PRECISION",    "output-stage precision: 0=int8, 0x80000000=int32 accumulate-out"},
    [RK_DPU_DST_BASE_ADDR]    = {0x1001, 0x4020, OKR_ANY,    "DPU_DST_BASE_ADDR",    "output (C) base IOVA"},
    [RK_DPU_DST_SURF_STRIDE]  = {0x1001, 0x4024, OKR_ANY,    "DPU_DST_SURF_STRIDE",  "output DST_SURF_STRIDE[31:4] between NC1HWC2 surfaces"},
    [RK_DPU_DATA_CUBE_WIDTH]  = {0x1001, 0x4030, 0x00001fff, "DPU_DATA_CUBE_WIDTH",  "output cube WIDTH-1[12:0] (= M-1 in the fold)"},
    [RK_DPU_DATA_CUBE_HEIGHT] = {0x1001, 0x4034, 0x01ffffff, "DPU_DATA_CUBE_HEIGHT", "output cube HEIGHT-1[12:0] / MINMAX_CTL[24:22]"},
    [RK_DPU_DATA_CUBE_NOTCH]  = {0x1001, 0x4038, 0x1fff1fff, "DPU_DATA_CUBE_NOTCH",  "output NC1HWC2 notch NOTCH_ADDR_0[12:0]/1[28:16]; 0 = no notch"},
    [RK_DPU_DST_N_DIMS]       = {0x1001, 0x403c, OKR_ANY,    "DPU_DST_N_DIMS",       "output N dims: dst stride and N-1 (((s-1)<<16)|(N-1))"},
    [RK_DPU_DST_N2]           = {0x1001, 0x4058, OKR_ANY,    "DPU_DST_N2",           "output N-1 (secondary)"},
    [RK_DPU_WDMA_SIZE_1]      = {0x1001, 0x405c, 0x1fff1fff, "DPU_WDMA_SIZE_1",      "write-DMA WIDTH_WDMA[12:0]/HEIGHT_WDMA[28:16] (-1 each)"},
    [RK_DPU_SURFACE_ADD]      = {0x1001, 0x40c0, OKR_ANY,    "DPU_SURFACE_ADD",      "output surface/element-size CONFIG (NOT an IOVA): 0x20 int8, 0x80 int32, 0x400 M-fold"},
    [RK_PDP_OUT_M]            = {0x801, 0x3014, OKR_ANY,     "PDP_OUT_M",            "PDP/aux output M dim = M-1 (16-bit; verified M-1 not (M-1)<<16 from the M=8 capture)"},
    [RK_PDP_OUT_N]            = {0x801, 0x3018, OKR_ANY,     "PDP_OUT_N",            "PDP/aux output N dim (N-1)"},
    [RK_PC_NEXT_ADDR]         = {0x101, 0x0010, OKR_ANY,     "PC_NEXT_ADDR",         "PC next-regcmd IOVA for HW task chaining (0 = end of chain)"},
    [RK_PC_NEXT_AMOUNT]       = {0x101, 0x0014, OKR_ANY,     "PC_NEXT_AMOUNT",       "PC next-task register-write count = (n+3)/2"},
    [RK_DPU_S_POINTER]        = {0x1001, 0x4004, OKR_ANY,    "DPU_S_POINTER",        "DPU surface/bank ping-pong pointer"},
    [RK_DPU_FEATURE_MODE_CFG] = {0x1001, 0x400c, OKR_ANY,    "DPU_FEATURE_MODE_CFG", "DPU feature / flying-mode config"},
    [RK_DPU_BS_CFG]           = {0x1001, 0x4040, OKR_ANY,    "DPU_BS_CFG",           "bias (BS) stage config / bypass"},
    [RK_DPU_BS_ALU_CFG]       = {0x1001, 0x4044, OKR_ANY,    "DPU_BS_ALU_CFG",       "bias ALU op config"},
    [RK_DPU_BS_MUL_CFG]       = {0x1001, 0x4048, OKR_ANY,    "DPU_BS_MUL_CFG",       "bias multiply config"},
    [RK_DPU_BS_OW_CFG]        = {0x1001, 0x4050, OKR_ANY,    "DPU_BS_OW_CFG",        "bias output-write config"},
    [RK_DPU_BN_CFG]           = {0x1001, 0x4060, OKR_ANY,    "DPU_BN_CFG",           "batch-norm (BN) stage config / bypass"},
    [RK_DPU_BN_ALU_CFG]       = {0x1001, 0x4064, OKR_ANY,    "DPU_BN_ALU_CFG",       "BN ALU op config"},
    [RK_DPU_BN_MUL_CFG]       = {0x1001, 0x4068, OKR_ANY,    "DPU_BN_MUL_CFG",       "BN multiply config"},
    [RK_DPU_EW_CFG]           = {0x1001, 0x4070, OKR_ANY,    "DPU_EW_CFG",           "elementwise (EW) stage config: EW_OP_TYPE/EW_OP_SRC (mul/add)"},
    [RK_DPU_EW_CVT_OFFSET]    = {0x1001, 0x4074, OKR_ANY,    "DPU_EW_CVT_OFFSET",    "EW input convert offset"},
    [RK_DPU_EW_CVT_SCALE]     = {0x1001, 0x4078, OKR_ANY,    "DPU_EW_CVT_SCALE",     "EW input convert scale"},
    [RK_DPU_OUT_CVT_OFFSET]   = {0x1001, 0x4080, OKR_ANY,    "DPU_OUT_CVT_OFFSET",   "output requant convert offset"},
    [RK_DPU_OUT_CVT_SCALE]    = {0x1001, 0x4084, OKR_ANY,    "DPU_OUT_CVT_SCALE",    "output requant convert scale (multiplier)"},
    [RK_DPU_OUT_CVT_SHIFT]    = {0x1001, 0x4088, OKR_ANY,    "DPU_OUT_CVT_SHIFT",    "output requant convert shift"},
    [RK_DPU_EW_OP_VALUE_0]    = {0x1001, 0x4090, OKR_ANY,    "DPU_EW_OP_VALUE_0",    "EW immediate operand 0"},
    [RK_DPU_R40C4]            = {0x1001, 0x40c4, OKR_ANY,    "DPU_R40C4",            "inferred DPU output-stage register (name unconfirmed)"},
    [RK_DPU_R4108]            = {0x1001, 0x4108, OKR_ANY,    "DPU_R4108",            "inferred DPU output-stage register (name unconfirmed)"},
    [RK_DPU_R410C]            = {0x1001, 0x410c, OKR_ANY,    "DPU_R410C",            "inferred DPU output-stage register (name unconfirmed)"},
    [RK_DPU_R4110]            = {0x1001, 0x4110, OKR_ANY,    "DPU_R4110",            "inferred DPU output-stage register (name unconfirmed)"},
    [RK_DPU_R411C]            = {0x1001, 0x411c, OKR_ANY,    "DPU_R411C",            "inferred DPU output-stage register (name unconfirmed)"},
    [RK_DPU_R4128]            = {0x1001, 0x4128, OKR_ANY,    "DPU_R4128",            "inferred DPU output-stage register (name unconfirmed)"},
    [RK_DPU_R412C]            = {0x1001, 0x412c, OKR_ANY,    "DPU_R412C",            "inferred DPU output-stage register (name unconfirmed)"},
    [RK_PC_OPERATION_ENABLE]  = {0x81,   0x0008, OKR_ANY,    "PC_OPERATION_ENABLE",  "PC_OPERATION_ENABLE doorbell (kick the programmed op)"},
    [RK_SDP_5004]={0x2001,0x5004,OKR_ANY,"SDP_5004","inferred SDP/activation reg (silu/gelu/ewmul), name unconfirmed"},[RK_SDP_5008]={0x2001,0x5008,OKR_ANY,"SDP_5008","inferred SDP/activation reg, name unconfirmed"},
    [RK_SDP_500C]={0x2001,0x500c,OKR_ANY,"SDP_500C","inferred SDP/activation reg, name unconfirmed"},[RK_SDP_5010]={0x2001,0x5010,OKR_ANY,"SDP_5010","inferred SDP/activation reg, name unconfirmed"},
    [RK_SDP_5014]={0x2001,0x5014,OKR_ANY,"SDP_5014","inferred SDP/activation reg, name unconfirmed"},[RK_SDP_5018]={0x2001,0x5018,OKR_ANY,"SDP_5018","inferred SDP/activation reg, name unconfirmed"},
    [RK_SDP_501C]={0x2001,0x501c,OKR_ANY,"SDP_501C","inferred SDP/activation reg, name unconfirmed"},[RK_SDP_5020]={0x2001,0x5020,OKR_ANY,"SDP_5020","inferred SDP/activation reg, name unconfirmed"},
    [RK_SDP_5028]={0x2001,0x5028,OKR_ANY,"SDP_5028","inferred SDP/activation reg, name unconfirmed"},[RK_SDP_5034]={0x2001,0x5034,OKR_ANY,"SDP_5034","inferred SDP/activation reg, name unconfirmed"},
    [RK_SDP_5038]={0x2001,0x5038,OKR_ANY,"SDP_5038","inferred SDP/activation reg, name unconfirmed"},[RK_SDP_5040]={0x2001,0x5040,OKR_ANY,"SDP_5040","inferred SDP/activation reg, name unconfirmed"},
    [RK_SDP_5044]={0x2001,0x5044,OKR_ANY,"SDP_5044","inferred SDP/activation reg, name unconfirmed"},[RK_SDP_5048]={0x2001,0x5048,OKR_ANY,"SDP_5048","inferred SDP/activation reg, name unconfirmed"},
    [RK_SDP_504C]={0x2001,0x504c,OKR_ANY,"SDP_504C","inferred SDP/activation reg, name unconfirmed"},[RK_SDP_506C]={0x2001,0x506c,OKR_ANY,"SDP_506C","inferred SDP/activation reg, name unconfirmed"},
    /* task #40: registers observed in rkllm's mfold capture (K=3584 N=1216), inferred — name/mask unconfirmed */
    [RK_BLK0_R0000]={0x0,0x0000,OKR_ANY,"BLK0_R0000","inferred BLK0 reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_BLK41_R0000]={0x41,0x0000,OKR_ANY,"BLK41_R0000","inferred BLK41 reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_CONV_CON3]={0x201,0x1014,OKR_ANY,"CNA_CONV_CON3","conv mode 3: NN_MODE[30:28]/atrous dilation/deconv stride (rkllm=0x9)"},
    [RK_CNA_CVT_CON0]={0x201,0x104c,OKR_ANY,"CNA_CVT_CON0","input-feature convert CVT_CON0 (requant enable/round mode; rkllm=0xb)"},
    [RK_CNA_CVT_CON1]={0x201,0x1050,OKR_ANY,"CNA_CVT_CON1","input convert CVT_SCALE0[31:16]/CVT_OFFSET0[15:0] (rkllm=0x10000 => scale=1,off=0 identity)"},
    [RK_CNA_CVT_CON2]={0x201,0x1054,OKR_ANY,"CNA_CVT_CON2","input convert CVT_SCALE1/CVT_OFFSET1 (rkllm identity)"},
    [RK_CNA_CVT_CON3]={0x201,0x1058,OKR_ANY,"CNA_CVT_CON3","input convert CVT_SCALE2/CVT_OFFSET2 (rkllm identity)"},
    [RK_CNA_CVT_CON4]={0x201,0x105c,OKR_ANY,"CNA_CVT_CON4","input convert CVT_SCALE3/CVT_OFFSET3 (rkllm identity)"},
    [RK_CNA_FC_CON0]={0x201,0x1060,0xffff0001,"CNA_FC_CON0","FC(matmul)-mode: FC_SKIP_DATA[31:16]/FC_SKIP_EN[0]"},
    [RK_CNA_FC_CON1]={0x201,0x1064,0x0001ffff,"CNA_FC_CON1","FC data offset[16:0]"},
    [RK_CNA_R1068]={0x201,0x1068,OKR_ANY,"CNA_R1068","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_FC_CON2]={0x201,0x1074,0x0001ffff,"CNA_FC_CON2","FC weight offset[16:0]"},
    [RK_CNA_DMA_CON0]={0x201,0x1078,0x800f000f,"CNA_DMA_CON0","feature/weight DMA burst: OV4K_BYPASS[31]/WEIGHT_BURST_LEN[19:16]/DATA_BURST_LEN[3:0] (rkllm=0xf000f)"},
    [RK_CNA_R1100]={0x201,0x1100,OKR_ANY,"CNA_R1100","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R1104]={0x201,0x1104,OKR_ANY,"CNA_R1104","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R1140]={0x201,0x1140,OKR_ANY,"CNA_R1140","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R1144]={0x201,0x1144,OKR_ANY,"CNA_R1144","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R1148]={0x201,0x1148,OKR_ANY,"CNA_R1148","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R114C]={0x201,0x114c,OKR_ANY,"CNA_R114C","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R1150]={0x201,0x1150,OKR_ANY,"CNA_R1150","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R1154]={0x201,0x1154,OKR_ANY,"CNA_R1154","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R1158]={0x201,0x1158,OKR_ANY,"CNA_R1158","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R115C]={0x201,0x115c,OKR_ANY,"CNA_R115C","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R1160]={0x201,0x1160,OKR_ANY,"CNA_R1160","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R1164]={0x201,0x1164,OKR_ANY,"CNA_R1164","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R1168]={0x201,0x1168,OKR_ANY,"CNA_R1168","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R116C]={0x201,0x116c,OKR_ANY,"CNA_R116C","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R1170]={0x201,0x1170,OKR_ANY,"CNA_R1170","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R1174]={0x201,0x1174,OKR_ANY,"CNA_R1174","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R1178]={0x201,0x1178,OKR_ANY,"CNA_R1178","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R117C]={0x201,0x117c,OKR_ANY,"CNA_R117C","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R1180]={0x201,0x1180,OKR_ANY,"CNA_R1180","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_CNA_R1184]={0x201,0x1184,OKR_ANY,"CNA_R1184","inferred CNA reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_PDP_R3010]={0x801,0x3010,OKR_ANY,"PDP_R3010","inferred PDP reg (rkllm=0x1); name unconfirmed"},
    [RK_PDP_R301C]={0x801,0x301c,OKR_ANY,"PDP_R301C","inferred PDP reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_PDP_R3030]={0x801,0x3030,OKR_ANY,"PDP_R3030","inferred PDP reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R4014]={0x1001,0x4014,OKR_ANY,"DPU_R4014","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R404C]={0x1001,0x404c,OKR_ANY,"DPU_R404C","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R4054]={0x1001,0x4054,OKR_ANY,"DPU_R4054","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R406C]={0x1001,0x406c,OKR_ANY,"DPU_R406C","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R407C]={0x1001,0x407c,OKR_ANY,"DPU_R407C","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R4094]={0x1001,0x4094,OKR_ANY,"DPU_R4094","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R4098]={0x1001,0x4098,OKR_ANY,"DPU_R4098","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R409C]={0x1001,0x409c,OKR_ANY,"DPU_R409C","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R40A0]={0x1001,0x40a0,OKR_ANY,"DPU_R40A0","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R40A4]={0x1001,0x40a4,OKR_ANY,"DPU_R40A4","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R40A8]={0x1001,0x40a8,OKR_ANY,"DPU_R40A8","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R40AC]={0x1001,0x40ac,OKR_ANY,"DPU_R40AC","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R4100]={0x1001,0x4100,OKR_ANY,"DPU_R4100","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R4104]={0x1001,0x4104,OKR_ANY,"DPU_R4104","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R4114]={0x1001,0x4114,OKR_ANY,"DPU_R4114","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R4118]={0x1001,0x4118,OKR_ANY,"DPU_R4118","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R4120]={0x1001,0x4120,OKR_ANY,"DPU_R4120","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
    [RK_DPU_R4124]={0x1001,0x4124,OKR_ANY,"DPU_R4124","inferred DPU reg; rkllm writes 0 (scratch/clear), name unconfirmed"},
};

/**
 * @name Named register values + field composers (the "recipe" layer)
 * @brief @c setrn() names the REGISTER; these name the VALUE. Set a mode-like register by named intent
 *        (@c setrn(rc,n,REG,OKV_*)) and compose packed fields with the @c OKC_* macros — so a config is a
 *        named choice, not a magic hex constant. Directly motivated by the M-fold RE, where every stall/wedge
 *        traced to a misread magic value: 0x40c0 as an IOVA (it's an elem-size config), 0x100c's int32-mode
 *        bit (see @ref OKV_CONV1_GROUP_LINE), 0x1080 as a stride (see @ref OKV_DMA2_CONTIGUOUS). C enums can't hold
 *        32-bit values like 0x80000000 cleanly, so these are typed uint32_t constants grouped by register.
 * @{
 */

/** @brief #RK_DPU_OUT_PRECISION (0x4010) = int8 output/accumulate datapath. */
#define OKV_OUT_PREC_INT8      0x00000000u
/** @brief #RK_DPU_OUT_PRECISION (0x4010) = int32 accumulate-out datapath. */
#define OKV_OUT_PREC_INT32     0x80000000u

/** @brief #RK_CNA_CONV_CON1 (0x100c) GROUP_LINE_OFF bit (bit 29, per mainline rocket_registers.h). It lives in
 *  the w1/extra half of the reg-write, so 0x100c must be decoded WIDE. NOTE: not an "int32-output datapath" bit
 *  (an earlier mislabel) — precision is CONV_CON1[9:4] (int8 = 0). RE finding (2026-07-30, full-register diff of
 *  in-chain tiles): GROUP_LINE_OFF is the ONLY functional mode bit that differs between the big-M fold tiles that
 *  DEPEND on resident weight (M=24/36, WEIGHT_BANK=8, cannot run standalone) and the small-M tiles that self-load
 *  the full weight (WEIGHT_BANK=11) — the weight-DMA config regs (0x1110/0x1030/0x1034/0x1038/0x1088/0x1024) are
 *  IDENTICAL in both. So GROUP_LINE_OFF is the prime suspect for the "reuse resident weight" (fold) mode; whether
 *  it alone suppresses the weight re-DMA is UNPROVEN (needs a full-chain replay + timing to confirm). */
#define OKV_CONV1_GROUP_LINE   0x20000000u
/** @brief #RK_CNA_CONV_CON1 (0x100c) GROUP_LINE_OFF clear — the PLAIN feature-read: self-loads the full weight,
 *  standalone-capable (all bit-exact standalone tiles, incl. the M=8 chain, use this — and re-DMA the weight). */
#define OKV_CONV1_PLAIN        0x00000000u

/** @brief #RK_DPU_SURFACE_ADD (0x40c0) = int8 element/surface size (CONFIG, not an IOVA). */
#define OKV_ELEMSZ_INT8        0x00000020u
/** @brief #RK_DPU_SURFACE_ADD (0x40c0) = int32 element/surface size (CONFIG, not an IOVA). */
#define OKV_ELEMSZ_INT32       0x00000080u
/** @brief #RK_DPU_SURFACE_ADD (0x40c0) = M-fold NC1HWC2 surface config (rkllm's M=8 task3; big-M tasks use 0x3000). */
#define OKV_SURFADD_MFOLD      0x00000400u

/** @brief DEPRECATED / DO NOT USE as a value. #RK_CNA_DMA_CON2 (0x1080) is a SIGNED 28-bit surface stride;
 *  `0x0fffffe8` is simply `-24` (= -3*M at M=8) in two's complement, read as UNSIGNED. `w1&0xfff` is the sign
 *  extension of a negative stride (0/4114 violations in the capture), NOT a mode/sentinel. This misread — a
 *  small negative stride mistaken for a ~268 MB stride — misled the mfold RE for weeks. Kept only so old @refs
 *  resolve; the decoder treats 0x1080 as signed (see regcmd_decode is_signed()). */
#define OKV_DMA2_CONTIGUOUS    0x0fffffe8u

/** @brief Compose #RK_CNA_CBUF_CON0 (0x1040): DATA_BANK[3:0] | WEIGHT_BANK[7:4]. */
#define OKC_CBUF_CON0(data_bank, weight_bank) ((uint32_t)((((weight_bank)&0xf)<<4) | ((data_bank)&0xf)))
/** @brief Compose #RK_CNA_DATA_SIZE0 (0x1020) / #RK_CNA_DATA_SIZE0_MIR (0x1084): DATAIN_WIDTH[26:16] | HEIGHT[10:0]. */
#define OKC_DATA_SIZE0(width, height)         ((uint32_t)((((width)&0x7ff)<<16) | ((height)&0x7ff)))
/** @} */

#endif /* ORK_REGS_H */
