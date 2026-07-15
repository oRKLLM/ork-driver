# RESHAPE_WIP — on-NPU fp16 contiguous→atom-8 reshape (the pure-NPU O(M·N) close)

**Goal.** Build an on-NPU op that reshapes a CONTIGUOUS `[M][N]` fp16 matmul output into the ATOM-8 cube layout
`PCH16(m,n)=(n/8)*(M*16)+m*16+(n%8)*2` that the fp16 per-channel SDP (`ork_npu_mul_perchan_f16`) reads — so the
O(M·N) fp16 attention-normalize path (fp16 matmul → reshape → fp16 SDP) is PURE-NPU with no CPU repack.

Board RK3588 `10.3.0.236`; ork-driver `main` (this WIP uncommitted). Working close today (no reshape needed):
`ork_npu_mm_perchan_f16_diag` (diagonal 2nd matmul, pure-NPU, O(M·N²)) + `ORK_DIAG_CHAIN` (one submit).

## Why a dedicated op is REQUIRED (all proven this session / prior)
- The fp16 MATMUL cannot write atom-8 directly — it HANGS (re-confirmed 2026-07-15: `ORK_F16_ATOM8=1
  ./f16_mm_f16out_probe` = rc=-1, 0 bytes; contiguous = 512/512). So the atom-8 layout must be MADE by a copy.
- The SDP/DPU-RDMA cannot read raw contiguous `[M][N]` (disc #3, `mul_perchan_f16_contig` 149/512) — it needs
  the atom-8 layout as its surface, so the reshape is unavoidable for the SDP path.
- `synth()` cannot build the reshape: an atom-8 group is `N_out=8` (one fp16 8-channel atom), sub-granularity
  vs synth's `N%16`/`K%32`. The vendor uses a stripped CNA-passthrough (below), not a synth matmul.

## Vendor mechanism (decoded from tools/re/captures/gemm_mul_f16.dump; template in src/regcmd_reshape.h)
The vendor Gemm*[1,1,N] (M=8,K=32,N=64) is a 13-task graph:
- task1 (op_idx=3, 108 reg, enable=0xd) = the fp16 GEMM, CONTIGUOUS out.
- task4..10 (op_idx=5, enable=0xd) = the RESHAPE. task4 = full 108-reg base (REGCMD_RESHAPE_F16); task5-10 =
  12-13 reg CHAINED-DELTAS repointing input(0x1070)/output(0x4020) per 8-channel group. Each writes CONTIGUOUS
  (0x4024=0x10, 0x40c0=0x20, 0x4050=0x126) — the atom-8 emerges from per-group OUTPUT-BASE placement (g·M·16).
- task11-13 (enable=0x18, 69 reg) = the per-channel EW-mul SDPs (already templatized: REGCMD_MUL_F16*).

task4 key regs (value=(val_hi<<16)|val_lo): 0x100c=0x120 (fp16 CNA — SAME as synth fp16 ⇒ weight likely
UNCOMPRESSED fp16, not DCOMP as earlier speculated); 0x4010=0x48000002 (fp16 out, proven); 0x1070/0x1110/0x4020
= in/weight/out bases (patched); **0x107c=0x20 (LINE_STRIDE=32 surf) and 0x1080=0x0fffffe8 (SURF_STRIDE, ≈ -24
signed)** = the reformat read geometry — does NOT match a naive `[M=8][N=64]` row gather (row=8 surf), so it is
NOT yet decoded to a general (M,N) formula.

## BLOCKERS (what stops a standalone build right now)
1. **Weight DATA missing.** The dump has `MEM_CREATE`+regcmd only, not buffer CONTENTS. 0x1110 points at a
   weight we don't have. For a passthrough reshape it's an identity/permutation; 0x100c=fp16 suggests the
   synth uncompressed fp16 tile format may work, but unconfirmed.
2. **Read geometry undecoded.** 0x107c=32 / 0x1080=0x0fffffe8 need a multi-(M,N) capture to solve the stride
   formula (one capture underdetermines it).
3. **Chained-delta.** task5-10 are deltas on task4; a standalone per-group op = task4 base + the delta merged.

## NEXT CONCRETE STEPS (the disciplined unblock — a re-capture campaign)
1. On the Colima VM (.239, `~/rknnenv`): `build_gemm_mul.py` already exists — build gemm_mul at 2-3 shapes
   (e.g. M=8/N=64, M=8/N=128, M=16/N=64) to solve the geometry stride formulas.
2. Capture on board WITH weight-buffer content dumps (the `~/rknn_sdk` shim + a buffer-dump flag) to get the
   0x1110 weight DATA for task4.
3. Decode task4-10 geometry across shapes → derive LINE_STRIDE/SURF_STRIDE(M,N) + the per-group output-base
   delta → generalize REGCMD_RESHAPE_F16 patching.
4. Build `ork_npu_reshape_c2a8_f16(c, src[M*N], M, N, dst_atom8)`: N/8 chained group-copies (task4 base +
   task5-style deltas), one submit. Validate: reshape a known [M][N] → feed `ork_npu_mul_perchan_f16`'s SDP
   directly (skip its CPU repack) → bit-exact per-channel scale, self-completes.
5. Wire into `ork_npu_mm_perchan_f16` behind a flag (replace the CPU repack with the on-NPU reshape).

## SAFETY
Board-safety non-negotiable: timeout every NPU cmd (`ORK_EW_TIMEOUT`/`ORK_MM_TIMEOUT`), verify `test_matmul`
between wedge-risk runs, SIGTERM not kill-9, `ssh board 'sudo reboot'` on NPU wedge. Do NOT blind-submit the
misunderstood 232-word regcmd with a guessed weight (wedge risk) — decode first (steps 1-3).

## CAMPAIGN RESULTS (2026-07-15 — re-capture kicked off + first decode)
Built gemm_mul at 3 shapes on the Colima VM (build_gemm_mul_shapes.py, ~/rknnenv) + captured regcmd on the
board (`tools/re/captures/cap_{M8_N64,M8_N128,M16_N64}.dump`, RKDUMP_WORDS=400). Decoded task4 (reshape base)
geometry across shapes (scratchpad decode_task.py):

- ★ **GEMM (task1) output is CONTIGUOUS `[M][N]`** (0x4024=0x10, 0x40c0=0x20, 0x4050=0x126, 0x4038=(N/8-1)) —
  IDENTICAL to our fp16 matmul output. So our `ork_npu_probe_f16_mm_f16out` output is EXACTLY the reshape's
  input — de-risked. (Earlier worry that the reshape reads a tiled intermediate is WRONG for the input.)
- ★ **Reshape is M-TILED, not N/8-grouped.** Reshape-op COUNT scales with M, not N: M8_N64 & M8_N128 both = 1×
  108-reg base + 6 small deltas; M16_N64 = 2× 108-reg bases + 5 deltas. So the vendor tiles the reshape over
  M-rows (≈8 rows per 108-reg base), NOT one copy per 8-channel group as first assumed.
- Decoded task4 register formulas (val=(val_hi<<16)|val_lo):
  - `0x107c` LINE_STRIDE = **32 CONSTANT** (all M,N) — the reshape READ stride is fixed 32 surfaces (512B), NOT
    the source row width. Needs reconciling with the contiguous [M][N] input (the read walks it in an M-tiled
    pattern, not row-major). KEY open item.
  - `0x1080` SURF_STRIDE = **M-32 (signed)** (M8→0x0fffffe8=-24, M16→0x0ffffff0=-16).
  - `0x1020` / `0x1084` low16 = **M/8** (M8→1, M16→2); `0x1010` = 0x20(M8)/0x30(M16); `0x405c` high = M-tile idx.
  - `0x4038` = **((N-1)<<16)|63** (M8_N64 0x3f003f, M8_N128 0x7f003f); `0x4058`=63, `0x40c0`=0x20 CONSTANT.
  - INVARIANT: 0x1014=9, 0x1024=0x00070008, 0x1030=0x2000, 0x1034=0x80, 0x1038=0x08010040, 0x4024=0x10, 0x4050=0x126.

## WEIGHT SOLVED + OP RUNS STANDALONE (2026-07-15, cont.)
- ★ **WEIGHT is capturable + constructible.** The shim (regcmd_capture.c:147-153) ALREADY dumps every tracked
  buffer's content — the weight was just past `RKDUMP_WORDS=400`. Re-captured M8_N64 with RKDUMP_WORDS=14336
  (`tools/re/captures/cap_M8_N64_full.dump`): weight sits at offset 0x10c0 in handle=2. It is a **64-entry
  PERMUTATION, ALL values == fp16 1.0** (`out[m][n]=in[m][perm(n)]`, a channel reorder). Positions (fp16-elem
  idx in the 8192B weight): 0,65,136,201,272,... (diffs alternate 65,71 with a 9 every 16 — the atom-8 perm in
  the CNA tile layout). Encoded in `ork_npu_reshape_probe_f16` (WPOS[64]).
- ★ **The reshape op RUNS + SELF-COMPLETES standalone** (rc=0, ~120µs): `ork_npu_reshape_probe_f16` submits
  REGCMD_RESHAPE_F16 (task4) with the constructed permutation weight + our contiguous [8][64] input +
  task_number=1, enable=0xd. No hang. `tools/reshape_probe.c` (native __fp16).
- ✗ **BUT output = 64× inf (wrong).** Unchanged by 0x107c/0x1080 read-pitch overrides. task4 is a MID-CHAIN
  task (chains from task1-3 via the 0x0010 descriptor); standalone it lacks the pipeline state those tasks set
  (CBUF residency / pipeline regs not in task4's own regcmd), so the accumulate saturates. The single-op path
  is exhausted — the reshape needs the FULL CHAIN context.

## FULL-CHAIN REPLAY RUNS (2026-07-15, cont.) — reshape executes in context
Extracted the full vendor IOVA image (`tools/re/captures/cap_M8_N64_full.dump` -> scratchpad extract_image.py
-> gemm_mul_image.bin, 77824B, IB=0xfffed000..IE=0x100000000) + the 22-task list. Dataflow decoded:
  input x @0xfffef000 -> task0(convert) -> 0xffff0480 -> task1(GEMM) -> **0xffff0000 (contiguous [M][N])** ->
  task2-10(reshape) -> **0xffff0a00 (atom-8)** -> task11-13(SDP per-channel scale) -> ... -> 0xfffed000 (out).
Built `ork_npu_replay_reshape_f16` (src/npu.c): loads the image, blanket single-delta rebases task0-10, submits
them as ONE 11-task chain, hands back gemm_out(@0x3000) + reshape_out(@0x3a00) for in-place verification. Tool
`tools/reshape_probe.c`. RESULT (board):
- ★ **The chain RUNS: GEMM output (405/512 nonzero) AND reshape output (2142 nonzero) both produced.** The
  reshape mechanism executes in-context (task4 is no longer starved — the FULLIMG replay gives it task1-3's
  pipeline state). **103/512 already match the PCH16 atom-8 formula.** Huge step past the standalone-inf wall.
- ✗ **errno=110 (deterministic, self-heals, same partial data each run).** NOT the chain walk-off (nulling
  task10's 0x0010/0x0014 descriptor changed nothing) -> one of task0-10 doesn't raise completion (candidates:
  task0 input-convert SDP, or a reshape task). Data is still readable (coherent buffer).
- ✗ **Layout derivation confounded:** the vendor's baked input yields degenerate gemm values (many 0x0001), so
  "find value in reshape_out" is ambiguous. Need CONTROLLED unique data in gemm_out to derive the exact perm.

## HARD WALL: multi-task chain replay does NOT cleanly execute (2026-07-15, cont.)
Made the replay bisectable (ORK_RESHAPE_NT 1-22), flags-configurable (ORK_RESHAPE_FLAGS, vendor=0x5),
input-injectable (ORK_RESHAPE_INJECT), output-zeroable (ORK_RESHAPE_ZERO, PRECISE 0x400 regions — the earlier
0x1000 zero clobbered task0-out @0x3480). Findings that overturn the "chain runs" read:
- **The earlier "405/2142 nonzero + 103/512 PCH16" was BAKED capture data** (the image file carries the vendor's
  captured buffer contents). With outputs zeroed pre-submit:
- ★ **gemm_out (0xffff0000) STAYS ZERO** (task1 GEMM never lands its output) while reshape-region tasks DO write
  (798-2078 nonzero) — contradictory for a sequential chain → the multi-task chain is NOT walking/executing as
  modeled (GEMM output not landing; downstream tasks reading baked buffers).
- **errno=110 persists across EVERY NT (2..22) and flags (0x1/0x5)** — even NT=2 (convert+GEMM) never completes.
  So it is NOT a specific reshape task and NOT the walk-off; it's the multi-task submit / chain-walk / PC-program
  SETUP for THIS graph (softmax's FULLIMG worked for ITS graph; this gemm+reshape graph doesn't replay the same).
- **Dataflow more tangled than gemm(0xffff0000)->reshape(0xffff0a00):** the reshape tasks read 0xffff0600 (not
  0xffff0000), and with 0xffff0000 zeroed the reshape still produces output → the 2-buffer model is wrong.

## SHARPENED BLOCKER (2026-07-15, cont.): the REPLAY PATH doesn't execute rebased vendor tasks
Tried all-3-subcore (sub.subcore_task[0]=[1]=[2]={0,NT}, matching the vendor capture + run_chain_i8), flags
0x1/0x5, NT bisect 1..22. **NT=1 (task0 alone, via the replay) ALSO errno=110** — yet the standalone
`ork_npu_reshape_probe_f16` (task4 copied to c->regcmd, patched addrs) DOES execute. So #2 is NOT primarily the
multi-task chain-walk; it's the **regcmd-in-BIG-image + blanket-delta-rebase replay mechanism** failing to run a
single rebased vendor task. Precise-zero test confirms task1's GEMM output never lands (all "nonzero" was baked
image data outside the zeroed regions). Candidates: (a) SDP task0 (en=0x18) needs setup the blanket rebase
doesn't preserve; (b) regcmd living in a 0x403 data buffer vs c->regcmd behaves differently; (c) the rebase
corrupts a non-address value, or misses a needed reference. Distinct from run_chain_i8 (which builds regcmds
fresh in c->regcmd at a fixed stride, not a rebased vendor image).

## ★ #2 CRACKED at the mechanism level (2026-07-15, cont.) — reshape executes + dataflow mapped
- **Root of the "nothing executes": the SDP input-convert task0 (en=0x18) doesn't run standalone in the replay**
  (NT=1 errno=110 in BOTH regcmd-in-image and c->regcmd paths), and the chain starts with it. **Skipping it
  (ORK_RESHAPE_T0=1, start at the GEMM) UNLOCKS the matmul chain: gemm_out lands FRESH (469/512 nonzero, was 0
  when zeroed).** So the blocker was task0, not the replay mechanism or multi-task chaining.
- Added the c->regcmd replay path (regcmds copied to c->regcmd, chain 0x0010 -> c->regcmd, DATA stays in BIG;
  ORK_RESHAPE_INIMG = old path), ORK_RESHAPE_T0 (start task), ORK_RESHAPE_GINJ (pre-fill gemm-out distinct).
- **Full dataflow DAG decoded** (per-task 0x1070/0x4020): task1 GEMM 0xffff0480->0xffff0000; task2 TRANSPOSE
  0xffff0000->0xffff0280; task3-10 = 8 atom-8 GROUP writes at **0xffff0680 + g*0x80** (each = M*16=128B, one
  8-channel group). So the reshape output BASE is **0xffff0680 (offset 0x3680)** — NOT 0xffff0a00 (that's just
  task10's last group). Fixed the probe read offset -> 0x3680.
- **The reshape RUNS and produces structured atom-8 output**: g[5][0]->r[0], g[5][1]->r[8], g[5][2]->r[16]
  (clear n*8 stride), **76-120/512 PCH16 match** (partial — chain still errno=110 so not all groups/M-tiles
  land, and non-unique gemm values confound exact derivation).

## ★ CLEAN COMPLETION ACHIEVED (2026-07-15, cont.) — full GEMM+reshape chain rc=0
Bisected T0=1 NT=1..10: **NT≤5 complete, NT=6 hangs**. Root cause: **12-reg reshape deltas (task6-10) cannot
sustain a PC-chain** — only 108-reg tasks and the 13-reg first-delta (task5) do. task5 has an extra
`0x1040=0x201b` write the 12-reg deltas lack; that write is required for chain continuation (mid-chain AND
terminus), not just completion. FIX (in `ork_npu_replay_reshape_f16`): **promote EVERY 12-reg delta to task5's
13-reg form**, patched to that delta's own captured 0x1070/0x4020 (in/out), + **recompute each task's `0x0014`
next-amount = (eff_amt[j+1]+3)/2** (the promotion changes amounts, so captured next-amounts go stale → hang).
RESULT: **the full GEMM+reshape chain task1-10 completes rc=0, ~75µs** (NT=7/8/10 all clean). The reshape output
is structured (`g[5][n]->r[n*8]` clean stride). (ORK_RESHAPE_NOTERM disables the promotion for A/B.)

## PERMUTATION DERIVATION (2026-07-15, cont.) — reshape is COUPLED to the GEMM output tiling
Distinct-input derive (T0=2 start at transpose + GINJ fp16(i+1) at the reshape input 0xffff0000, read
reshape-out @0xffff0680, recover source idx = round(v)-1): **only 139/512 values map, positions scattered**
(input(0,0)→r[210], NOT r[0]=PCH16 nor r[0]=transpose). So the multi-stage reshape is NOT a clean standalone
value-permutation of a plain-contiguous input — **task2 (the "transpose") is coupled to the GEMM's specific
fp16 output TILING**, which plain row-major GINJ doesn't match. (Earlier PCH16 "matches" on real data were
duplicate-value coincidences.)
Also unresolved: **the reshape writes 8 groups at 0xffff0680+g*0x80, but the SDP (task13) reads 0xffff0a00**
(= task10's LAST group), not 0x680 — so either the atom-8 base the SDP consumes is 0xa00 (groups laid
differently than I model), or there's another consolidation step. The 0x680-vs-0xa00 dataflow must be resolved
before the permutation formula is trustworthy.

## REMAINING (deeper than expected)
1. Resolve the reshape-output base (0x680 groups vs 0xa00 SDP-read) + the GEMM output tiling task2 consumes.
2. Cleanest validation is NOT value-tracing but END-TO-END: feed a REAL ork fp16-matmul output (matching the
   GEMM's tiling) through the reshape → the SDP → compare per-channel-scaled result to CPU (bit-exact). That
   is effectively the "build the op" step (matmul→reshape→SDP integration), not a standalone derive.
3. Generalize (M,N) + the 64-entry weight permutation, build `ork_npu_reshape_c2a8_f16`, wire into mul_perchan_f16.

## WHAT'S SOLID (this campaign)
Weight solved (64-entry all-1.0 perm); execution unlocked (skip SDP task0); dataflow largely mapped
(GEMM→task2→8 groups); **CLEAN COMPLETION of the full task1-10 chain (rc=0, ~75µs)** via all-deltas-promoted-
to-13-reg + next-amount recompute. The exact atom-8 permutation + end-to-end bit-exact is the remaining
(genuinely deeper) work; diagonal close covers attention meanwhile.

## ASSESSMENT
Genuine multi-session research now. The vendor reshape runs as a single op (needs pipeline context) and the
weight is SOLVED (64-entry all-1.0 permutation), but faithfully REPLAYING the chain to validate/extract it hits
a multi-task-execution wall needing deep protocol RE. Payoff stays negligible for attention — the diagonal close
`ork_npu_mm_perchan_f16_diag` already delivers pure-NPU O(M·N²), fine for small head_dim. Recommend banking here
unless the general large-N O(M·N) reshape is independently wanted.

## HARNESS (all on disk, uncommitted — resumable)
src/regcmd_reshape.h (task4 base), src/npu.c `ork_npu_reshape_probe_f16` (+ORK_RSH_107C/1080), tools/reshape_probe.c,
tools/re/captures/cap_{M8_N64,M8_N128,M16_N64,M8_N64_full}.dump, scratchpad/{decode_task,decode_all,wpat,extract_buf}.py
+ build_gemm_mul_shapes.py + vm_build_shapes.sh.

## STATUS
Campaign LAUNCHED + first decode done: gemm-output-contiguous CONFIRMED (our matmul feeds it directly),
reshape M-tiling structure mapped, task4 geometry formulas ~80% decoded. Remaining = deltas + weight + op
build (scoped above). Artifacts: src/regcmd_reshape.h, tools/re/captures/cap_M*.dump, scratchpad decode_task.py
+ build_gemm_mul_shapes.py. The diagonal close covers the attention use case meanwhile.
