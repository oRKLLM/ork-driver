# MFOLD recapture WIP (task #39 — offline route)

## Goal
Resolve the mfold schedule by CAPTURING rkllm's real M-fold chain OFFLINE (LD_PRELOAD shim), not by
on-board synth sweeps (which are intractable — see below). One clean rkllm run gives the per-task
schedule for the true chain, which resolves the two inconsistent theories AND recovers the one lost
constant (`OKV_CONV1_MFOLD` = whatever rkllm writes to reg `0x100c`).

## Current hypothesis
rkllm runs the K=3584/N=1216 int8 fold as a weight-resident HW-chain of width-≤8 tiles
(task_number=3/6, 3-core round-robin). The correct per-M schedule is derivable from a single capture of
that chain. Once derived + verified self-consistent across the chain's tasks, encode it into a COMMITTED
`synth_i8_mfold` (with the macro properly defined) so it's never orphaned again.

## Proven / ruled out
- LAYOUT is CONFIRMED (bit-exact, unique): input C2-16 `(k/16)*(M*16)+m*16+(k%16)`; weight ork_woff
  `((n/32)*KT+(k/32))*1024+(n%32)*32+(k%32)`, KT=(K+31)/32; output C2-4 `(n/4)*(M*4)+m*4+(n%4)`.
- ON-BOARD SYNTH SWEEPS ARE INTRACTABLE: board hard-wedges (cold-cycle) on nearly every mfold submit;
  offline sim (cdma_calib) can't discriminate candidates. Do NOT hand-tune constants + resubmit.
- TWO INCONSISTENT SCHEDULE THEORIES exist, each with a single fragile bit-exact point:
  - M=36-LITERALS (board + local repo HEAD 8c892e5, uncommitted working tree): CONV_CON1=GROUP_LINE
    (0x20000000), CVT_CON block, schedule regs FIXED at M=36 (0x1080=2160/mc, 0x107c=0x60, 0x1040=0x84,
    0x4024=0x600, 0x40c0=0x3000). Wedges at M=8 (oversized strides). "validated at M=36 ONLY".
  - PER-M FORMULAS (orphaned in scratch npu_board.c ONLY): CONV_CON1=OKV_CONV1_MFOLD (macro DEFINED
    NOWHERE), no CVT_CON, 0x1080=-3*mc, 0x107c=min(4*mc,128), 0x1040=OKC_CBUF_CON0(ceil(M/8),12-db),
    0x4024=16*mc, 0x40c0=128*mc, +0x104c=0x0b. Bit-exact M=8 ONCE then errno-110'd.
  - They DISAGREE at M=36 (16*36=576 vs 1536; 128*36=4608 vs 12288; 4*36=144 vs 96) → not the same
    synth with different constants; genuinely different theories. The capture settles which (if either).

## Working-tree state (as of session start)
- Board ~/ork-driver: branch main, HEAD f446052, UNCOMMITTED synth_i8_mfold = M=36-literal theory
  (src/npu.c ~892-923). validate_mfold binary built OK from it.
- Local ork-driver: branch feat/attn-biased-rr, HEAD 8c892e5 (committed fold_chain_test.c, UNVALIDATED),
  same M=36-literal synth in working tree. NO formula version committed anywhere.
- Scratch: /private/tmp/.../scratchpad/npu_board.c = the orphaned FORMULA synth + ork_npu_mfold_chain.
- fold_chain_test.c committed (8c892e5); ork_npu_mfold_chain lives in scratch npu_board.c (not on board).

## Capture mechanism (tools/re/regcmd_capture.c → rknpu_dump.so)
LD_PRELOAD shim intercepts ioctl/mmap on the librkllmrt process; on DRM_IOCTL_RKNPU_SUBMIT it parses the
task descriptors + hex-dumps regcmd. `RKDUMP_MM=1` (+ RKDUMP_MM_K/RKDUMP_MM_N, default 3584/1216) keeps
the LARGEST-M matching program and WALKS THE CHAIN (0101:0x0010 next-addr / 0x0014 amount), writing:
  /tmp/mm_regcmd.txt, /tmp/mm_meta.txt (M/K/N/wbytes/wstride/spans/offsets),
  /tmp/mm_chain_meta.txt (per-task: Kfield,M,Aadr,Cadr,DATA_ENTRIES,CBUF_CON0,next,namt),
  /tmp/mm_chain_<i>.txt (each chain task's raw regcmd, ≤224 words),
  /tmp/mm_{A,weight,C}.bin (operand images for bit-exact offline replay).
Build: `gcc -shared -fPIC -O2 -Itools/re -o /tmp/rknpu_dump.so tools/re/regcmd_capture.c -ldl`
(needs tools/re/rknpu_ioctl.h; confirm include path). Run: `sudo env LD_PRELOAD=/tmp/rknpu_dump.so
LD_LIBRARY_PATH=<librkllmrt dir> RKDUMP_MM=1 RKDUMP_MM_K=3584 RKDUMP_MM_N=1216 <rkllm_demo> <7B.rkllm> ...`
Parse: tools/re/parse_mfold.py reads a dump with `--- regcmd (N u32 words)` blocks (word0=(val<<16)|reg,
word1=(block<<16)|extra) → per-M schedule table for K=3584 N=1216.

## RESULT (2026-07-29): recapture DONE. The premise "synth the schedule from M" is REFUTED.
Analyzed the existing full capture ~/rkllm_ffn_capture_2026-07-27.dump (29MB, 4647 regcmd blocks, 3563
tasks at K=3584 N=1216, 13 distinct output widths) with tools/re/analyze_schedule.py. Findings:

1. WORD ENCODING (definitive, baked into analyze_schedule.py header): each reg write is a u32 pair
   w0=(val16<<16)|reg16, w1=(block16<<16)|extra16. VALUE = w0>>16 (16-bit); NOT (w0>>16)|((w1&0xffff)<<16)
   — the 32-bit form corrupts 16-bit regs (only a few address regs 0x1070/0x1110/0x4020 are truly 32-bit).
   This was the trap that made the first analysis pass match 0 tasks.

2. OKV_CONV1_MFOLD = 0x0  (reg 0x100c is CONSTANT 0 across ALL fold tasks). = OKV_CONV1_PLAIN, which the
   board ALREADY defines. The board's M=36 synth used GROUP_LINE (0x20000000) — that was WRONG.

3. CONSTANT shape regs: 0x100c=0, 0x104c=0xb. (0x1020 DATAIN-W=1, 0x1084 H=1, 0x1010 grains=0x20 also const.)

4. THE CRUX — the disputed schedule regs (0x1040 CBUF-bank, 0x107c DMA_CON1, 0x1080 DMA_CON2, 0x4024
   DST_SURF_STRIDE, 0x40c0 SURFACE_ADD) are NOT a function of M. They VARY across tasks at the same M,
   the same output-width, and the same (rows, K-slice) factoring (persistent #k multi-value cells). They
   are OUTPUTS of rkllm's internal per-sub-block TILING PLANNER (K-slice x row-tile x core/column split x
   output position) — deterministic given the full tiling context, but multi-variable and NOT closed-form
   in the matmul shape. 0x4024/0x40c0 are additionally address/stride (ork computes those itself).
   -> BOTH prior theories were single-sub-block snapshots mislabeled as f(M): "M=36 literals" = one tile's
      values; "per-M formulas" (-3*M,16*M,...) = another tile's, fit through the origin. Neither matches the
      capture at M=36 once grouped correctly (THEORY CHECK: NEITHER). This is why on-board M-sweeps were
      intractable and the scratch formula was flaky.

## Strategic reframe for #39 (DECISION NEEDED)
"Encode rkllm's fast fold schedule as a formula of shape" is ILL-POSED — there is no such formula; the
regs are rkllm's tiling-planner output. Remaining real paths, both different from the whole arc so far:
  (A) CAPTURE-AND-REPLAY rkllm's exact regcmd chains for the specific shapes the target model uses (the
      RKDUMP_MM replay path already replays a captured chain bit-exactly). Brittle/shape-specific but real.
  (B) REVERSE-ENGINEER rkllm's tiling planner fully (large, open-ended; it's a heuristic, not a formula).
Payoff is bounded by the residual prefill gap (ork already ~90-97% of rkllm), so weigh effort vs gain.
Banked regardless: analyze_schedule.py (RE tool), OKV_CONV1_MFOLD=0, the word-encoding, and this reframe.


## Board gotchas
- Single-stream NPU; never run ork + rkllm concurrently (IOMMU wedge — memory npu-single-stream).
- rkllm capture is LOW risk; the wedge risk is ONLY ork's mfold synth (avoid until schedule is derived).
- Hard wedge recovery: HA "Rock 5B Plug" off (~20s drain) / on; POST via ~/ork-driver/ork_dyn_spin_diag.
- Bash classifier was intermittently unavailable this session; reads/writes to local files unaffected.
