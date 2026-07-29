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

## (A) CHAIN STRUCTURE — captured + characterized 2026-07-29 (rk_bench_short + RKDUMP_MM, tools/re/submit_extract.py)
- rkllm's librkllmrt runs fine via the native C++ benches in ~/rkbench/ (rk_bench_short <model> <maxctx>
  <maxnew> <ncore>). The node harness (rkllm_bench.js -> orkllm_napi.node) CRASHES (std::out_of_range,
  0 submits) — do NOT use it; use ~/rkbench/rk_bench_short.
- Capture cmd (WORKS, low-risk): cd ~/rkbench && sudo env LD_PRELOAD=/tmp/rknpu_dump.so LD_LIBRARY_PATH=.
  ORK_SUBMIT_ONLY=1 RKDUMP_MM=1 RKDUMP_MM_K=3584 RKDUMP_MM_N=1216 timeout 120 ./rk_bench_short
  ~/Qwen2.5-7B...w8a8.rkllm 512 4 3   -> /tmp/mm_{regcmd,meta,chain_meta}.txt + mm_{A,weight,C}.bin.
  NOTE: ORK_SUBMIT_ONLY suppresses per-task regcmd hexdumps (keep OFF if you want all 21 task regcmds in the
  log). The 0x0010 chain-walk only follows ~4 tasks; the real chain is the task_number=21 DESCRIPTOR ARRAY.
- ONE MATMUL = a task_number=21 chain, per core (core=0x1/0x2/0x4 round-robin), domain=1, flags=0x5.
- EVERY TASK IS FULL-K: Ks=64*DATA_ENTRIES/M = 3584 = K for all. NO K-slicing. Each task is a complete
  A[M,3584]*W[3584,1216] matmul for an M-row tile. DATA_ENTRIES(0x1044)=56*M exactly (56=K/64).
- rkllm tiles the prefill's TOTAL M into VARIABLE row-tiles: widths seen = 1,2,4,6,8,10,12,14,20,24,36
  (NOT uniform, NOT width-8). Weight loaded once, tiles chained (weight-resident).
- SHAPE-CLEAN regs (safe to synth from M): 0x100c=0 (=OKV_CONV1_PLAIN=OKV_CONV1_MFOLD), 0x104c=0xb,
  0x1044=56*M, output cube WIDTH(0x4030)=M-1, N(0x1038)=1216, K(0x1024)=3584.
- PLANNER-STATE regs (NOT f(M,N) — vary even at fixed (M,N), e.g. M=6 -> CBUF 0xa2/0xb1/0xa2): 0x1040 CBUF
  bank split, 0x107c DMA_CON1, 0x1080 DMA_CON2, 0x4024 DST_SURF_STRIDE, 0x40c0 SURFACE_ADD (0x4024/0x40c0
  are also output address/stride ork computes itself).

## (A) NEXT: single-tile bit-exact REPLAY MVP
- Each tile is a COMPLETE full-K/full-N matmul -> independently CPU-comparable. MVP: replay rkllm's EXACT
  captured regcmd for ONE tile (start M=36, its operands are dumped) via ork_npu_replay_i8 (rebase the 3
  addrs FEATURE/WEIGHT/DST to ork buffers), feed ork-packed A/W in the confirmed layout (C2-16 in / C2-4
  out / ork_woff weight), compare to CPU ref. This is validate_mfold but with rkllm's regcmd, not ork synth.
  If bit-exact + no wedge -> capture-replay is viable; scale to a per-M tile-program library + a row-tile
  chain executor (weight-resident, 3-core RR). If it wedges -> even rkllm's exact regcmd won't run rebased.
- To get a CLEAN per-tile regcmd: extract from the verbose original dump (submit_extract.py) OR re-capture
  WITHOUT ORK_SUBMIT_ONLY so every task's regcmd is logged, then pick a full-N (N=0x4c0) tile of the wanted M.

## (A) REPLAY MVP progress (tools/re/replay_tile.c)
- replay_tile feeds rkllm's captured regcmd (/tmp/mm_regcmd.txt, 232 words, the M=36 full-K/full-N tile) +
  rkllm's operand bytes through ork_npu_replay_i8 (patches FEATURE/WEIGHT/DST addrs, task_number=1), then
  compares ork's raw output buffer to rkllm's dumped output (/tmp/mm_C.bin) — zero layout assumptions.
- **KEY POSITIVE: rkllm's exact regcmd EXECUTES ON ork with NO WEDGE** (repeatable, ~640 us/submit). This
  de-risks the whole capture-replay idea: what wedged all along was ork's WRONG SYNTH, not the fold path.
- OUTPUT not yet bit-exact: raw compare 28692/29184 mismatch, ork=37 vs rkllm=8941 at idx0. ork's value is
  far too small for a full K=3584 dot-product -> ork isn't reading the weights/A rkllm's program expects.
  Layout is irrelevant to the raw compare (same regcmd + same bytes must match), so it's an INPUT-COVERAGE
  or CHAIN-CONTEXT issue. Hypotheses being tested:
  (i) weight span: fed only first K*N; rkllm dumped Wspan=4*K*N -> the fold may read STRIDED across the full
      span. [testing: feed full mm_weight.bin]
  (ii) chain context: this M=36 tile is 1 of a task_number=21 chain; if weights are CBUF-resident from a
      prior task (loaded once), a standalone single-task replay has no resident weights -> near-zero output.
      If (i) doesn't fix it, the tile is NOT standalone and (A) needs a FULL-CHAIN replay (all 21 tasks in
      one submit), not single-task. That's the chain-replay executor (extend ork_npu_replay_i8 to the
      task_number=21 descriptor array with per-task regcmd + rebased addrs).

## (A) LAYOUT is ALSO unconfirmed vs rkllm (offline check, 2026-07-29)
- Pulled rkllm's EXACT operands for one tile locally (scp /tmp/mm_{A,weight,C}.bin, mm_regcmd.txt) and ran the
  DECISIVE offline check (scratch offline_layout_check.py / layout_solve.py): does rkllm's OWN A*W (de-tiled
  via our "confirmed" fold layouts) == rkllm's OWN captured C?  ANSWER: NO. 0/14 sampled outputs match.
- The A*W magnitudes are the RIGHT SCALE (hundreds..thousands, like rkllm's C) -> K is fully summed with real
  data, but the INDEX MAPPING is wrong. Swept width-pad {16,24,32,48} x A{C2-16,rowmaj,transposed} x
  W{ork_woff,[k][n],[n][k]} x C{C2-4,rowmaj} -> NOTHING matches (best 0/4).
- CONCLUSION: the "confirmed unique" fold layout (C2-16 in / ork_woff / C2-4 out) was only ever validated
  against ORK'S OWN single-tile synth (validate_mfold M=8), NEVER against rkllm's real data. Both the schedule
  AND the layout need solving from captured data. (The one bit-exact M=8 was ork-self-consistent, not rkllm.)
- This is now a BOUNDED OFFLINE puzzle (no board/wedge risk): I have rkllm's exact A, W, C, regcmd locally.
  Open sub-questions for the solve: (a) is this M=24 tile a STANDALONE full matmul or a chain sub-block whose C
  accumulates / is post-scaled (w8a8 per-channel)?  (b) exact A C2 layout + width padding; (c) exact weight
  tiling; (d) exact C2-4 output mapping. NEXT: systematic offline layout solve from the captured triple.

## (A) NET STATE
- WORKS: capture pipeline; rkllm's regcmd EXECUTES on ork with NO WEDGE (~640us) -> capture-replay mechanically
  viable; the historic wedges were ork's WRONG SYNTH, not the fold path.
- OPEN: exact fold LAYOUT (refuted vs rkllm data) and exact per-tile SCHEDULE (planner output) both still need
  offline solving from captured data before a bit-exact replay is possible.

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
