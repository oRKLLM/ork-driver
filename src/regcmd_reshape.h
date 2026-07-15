/* src/regcmd_reshape.h — VENDOR fp16 contiguous->atom-8 RESHAPE op (RE template, WIP).
 *
 * Extracted from the Gemm*[1,1,N] capture (tools/re/captures/gemm_mul_f16.dump), task[4] — the first of the
 * reshape stage (op_idx=5, enable=0xd, regcfg_amount=108). The vendor per-channel-scaled fp16 matmul is a
 * 13-task graph: task1 = the fp16 GEMM (contiguous out); task[4..10] = the contiguous->atom-8 RESHAPE (task4
 * = full 108-reg base, task5-10 = 12-13 reg chained-delta repointing input/output per 8-channel group);
 * task11-13 = the per-channel EW-mul SDPs. This reshape is the ONLY on-NPU way to feed the fp16 SDP an atom-8
 * layout WITHOUT a CPU repack (the fp16 MATMUL cannot write atom-8 directly — it HANGS; wiki disc #2), and it
 * cannot be built from synth() (synth needs N%16/K%32; an atom-8 group is N_out=8, sub-granularity).
 *
 * Word format (ork setr): rc[k]=(val_lo<<16)|reg ; rc[k+1]=(lane<<16)|val_hi ; value=(val_hi<<16)|val_lo.
 *
 * DECODE (task4) — what's understood + what BLOCKS a from-scratch build:
 *   0x100c=0x120  CNA_CONV_CON1 = fp16 (SAME as synth fp16 -> weight is likely UNCOMPRESSED fp16, not DCOMP).
 *   0x1070=<in>   input activation base (PATCH).                 0x1110=<wt> weight base (PATCH) — DATA MISSING.
 *   0x107c=0x20   CNA LINE_STRIDE = 32 surfaces  <-- does NOT match [M=8][N=64] (row = 64 elem = 8 surf). The
 *                 reshape read geometry is NOT a naive [M][N] row-strided gather — needs full decode.
 *   0x1080=0x0fffffe8  CNA SURF_STRIDE = near-wrap negative  <-- the reformat trick (reverse/interleaved read?).
 *   0x4010=0x48000002  fp16/fp16/fp16 out (SAME as our proven fp16-out).   0x4020=<out> output base (PATCH).
 *   0x4024=0x10, 0x40c0=0x20, 0x4050=0x126  = CONTIGUOUS output geometry (NOT atom-8) -> each group op writes
 *                 CONTIGUOUS, and the atom-8 layout emerges from per-group OUTPUT-BASE placement (task5 delta).
 *   trailer[216] 0xb9400010/0x0101 = chain desc 0x0010 -> next-regcmd (task5); 0x0014 = next-reg-amount 8.
 *
 * BLOCKERS to a standalone build (see RESHAPE_WIP.md):
 *   (1) WEIGHT DATA not captured — the .dump has MEM_CREATE + regcmd only, not buffer contents. Need a
 *       re-capture WITH weight-buffer dumps (Colima VM shim) OR to construct the exact identity/permutation.
 *   (2) The read geometry (0x107c=32, 0x1080=0x0fffffe8) is not yet decoded to a general (M,N) formula.
 *   (3) task5-10 are chained-DELTAs on this base; a standalone per-group op = this base + the delta merged.
 * Values below are the captured op VERBATIM (M=8,N=64); addresses (0x1070/0x1110/0x4020) are placeholders. */
#define REGCMD_RESHAPE_F16_N 232
static const uint32_t REGCMD_RESHAPE_F16[REGCMD_RESHAPE_F16_N]={
0x201b1040,0x02010000,0x00001104,0x02010000,0x00001100,0x02010000,0x0120100c,0x02010000,
0x000e4004,0x10010000,0x0120100c,0x02010000,0x00201010,0x02010000,0x00091014,0x02010000,
0x00011020,0x02010008,0x00081024,0x02010007,0x00011028,0x02010000,0x0001102c,0x02010000,
0x20001030,0x02010000,0x00801034,0x02010000,0x00401038,0x02010801,0x201b1040,0x02010000,
0x00021044,0x02010000,0x000b104c,0x02010000,0x00001050,0x02010001,0x00001054,0x02010001,
0x00001058,0x02010001,0x0000105c,0x02010001,0x00001060,0x02010000,0x00001064,0x02010000,
0x00001068,0x02010000,0x03001070,0x0201ffff,0x00001074,0x02010000,0x000f1078,0x0201000f,
0x0020107c,0x02010000,0xffe81080,0x02010fff,0x00011084,0x02010008,0x00081088,0x02010000,
0x00001100,0x02010000,0x00001104,0x02010000,0x20c01110,0x0201ffff,0x00001140,0x02010000,
0x00001144,0x02010000,0x00001148,0x02010000,0x0000114c,0x02010000,0x00001150,0x02010000,
0x00001154,0x02010000,0x00001158,0x02010000,0x0000115c,0x02010000,0x00001160,0x02010000,
0x00001164,0x02010000,0x00001168,0x02010000,0x0000116c,0x02010000,0x00001170,0x02010000,
0x00001174,0x02010000,0x00001178,0x02010000,0x0000117c,0x02010000,0x00001180,0x02010000,
0x00001184,0x02010000,0x02003010,0x08010000,0x00003014,0x08010000,0x003f3018,0x08010000,
0x0000301c,0x08010000,0x00003030,0x08010000,0x01e4400c,0x10010000,0x00024010,0x10014800,
0x00004014,0x10010000,0x07004020,0x1001ffff,0x00104024,0x10010000,0x00004030,0x10010000,
0x00004034,0x10010000,0x003f4038,0x1001003f,0x003f403c,0x1001003f,0x00534040,0x10010000,
0x00004044,0x10010000,0x00004048,0x10010000,0x0000404c,0x10010000,0x01264050,0x10010000,
0x00004054,0x10010000,0x003f4058,0x10010000,0x0000405c,0x10010000,0x00534060,0x10010000,
0x00004064,0x10010000,0x00004068,0x10010000,0x0000406c,0x10010000,0x03834070,0x10010000,
0x00004074,0x10010000,0x00014078,0x10010000,0x0000407c,0x10010000,0x00004080,0x10010000,
0x00014084,0x10010001,0x00004088,0x10010000,0x00004090,0x10010000,0x00004094,0x10010000,
0x00004098,0x10010000,0x0000409c,0x10010000,0x000040a0,0x10010000,0x000040a4,0x10010000,
0x000040a8,0x10010000,0x000040ac,0x10010000,0x002040c0,0x10010000,0x000040c4,0x10010000,
0x00004100,0x10010000,0x00004104,0x10010000,0x00004108,0x10010000,0x0000410c,0x10010000,
0x00004110,0x10010000,0x00004114,0x10010000,0x00004118,0x10010000,0x0000411c,0x10010000,
0x00004120,0x10010000,0x00004124,0x10010000,0x00004128,0x10010000,0x0000412c,0x10010000,
0xb9400010,0x0101ffff,0x00080014,0x01010000,0x00000000,0x00410000,0x000d0008,0x00810000,
0x201b1040,0x02010000,0x00001104,0x02010000,0x00001100,0x02010000,0x0120100c,0x02010000,
};
