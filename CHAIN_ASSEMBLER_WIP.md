# Chain assembler — WIP recovery doc

Goal (user: "build the chain assembler next"): chain a whole NPU-only run (FFN inner; attention block)
into ONE `task_number=N` submit, so per-op round-trips collapse. This is THE submit-reduction lever
(per-node on-NPU regressed: 9807 submits, 76 vs 97 tok/s). Board 10.3.0.236 (non-prod, wedge-OK).
Branch: ork-driver `feat/static-graph`. All gated; default path untouched (~97 tok/s).

## Landed
- **Core `ork_npu_chain_progs(c, n, progs[], dom)`** (commit 5755089): submits N pre-built heterogeneous
  programs as one PC-chain. Packs contiguously into c->regcmd (content-driven stride); for each non-last
  program, SCANS for its PC next-descriptor slot (word where reg==0x0010 paired with domain 0x0101) and
  links next-addr + next-regcfg-amount. One rknpu_task/program (own enable_mask/regcfg), ping-pong OFF.
  Compiles clean on board. Mechanism already validated by run_chain_i8 (homogeneous) + Phase-0 (matmul->silu).

## Key facts / mechanics
- Next-descriptor: `rc[slot]=0x0010|(next_lo<<16); rc[slot+1]=0x0101<<16|next_hi; rc[slot+2]=0x0014|(nreg<<16);
  rc[slot+3]=0x0101<<16` where `nreg=(next.regcfg_amount+3)/2`. Last program keeps its raise-interrupt tail.
- enable_mask: matmul=0xd (regcfg 108), SDP silu/ewmul=0x18 (silu regcfg 69).
- Matmul int16 output: `set_i16_out(rc,N,stride,mult,shift)` rewrites the DPU output stage (0x4010 precision,
  0x4038 group stride N/8-denser, 0x4050 row byte-stride 0x0248, 0x40c0 elem=2B, 0x4084/0x4088 requant
  mult/shift). Validated standalone via ork_npu_probe_i16_out.
- **Phase-0 (ork_npu_chain_mm_silu_i16) only proved the chain WALKS** — its matmul output (Cd) and silu
  input (A) are SEPARATE buffers, NOT data-connected. The real FFN chain needs matmul-out == silu-in.

## THE CRUX (increment 1, in progress)
Data-connected matmul-int16-out -> silu-int16-in: does the int8-matmul int16 OUTPUT layout
(set_i16_out: 0x4050=0x0248 row byte-stride, 0x4038 group stride, cube tiling) match the int16 silu's
EXPECTED INPUT layout (0x5018 input addr; Phase-0's EWCUBEH cube addressing)? If yes, point silu-in at
the matmul's int16 output buffer and the bridge works. If not -> RE the layout xform (may need a repack
or a matching stride on the silu input regs).

## FFN chain plan (SwiGLU), each transition a bridge to validate
1. gate = A[M,K] x Wg[K,N]  (int8 matmul, set_i16_out -> int16 G)      [0xd]
2. silu(G) -> S (int16)                                                 [0x18]   <- increment 1 = 1->2 bridge
3. up   = A[M,K] x Wu[K,N]  (int8 matmul, set_i16_out -> int16 U)      [0xd]
4. glu  = S (*) U -> H (ewmul int16)                                    [0x18]
5. down = H[M,N] x Wd[N,K]  (int8 matmul; N=6144 contraction -> K-split -> multiple programs) [0xd...]
- down K-split = multiple chained matmul programs w/ accumulate; the one many-program sub-chain.
- Intermediate buffers G/S/U/H resident in one domain; addresses threaded through each op's regcmd.

## BLOCKER (increment-1 result, 2026-07-12)
Ran the data-connected matmul(i16-out)->silu chain: HANGS (dmesg `task counter: 0x0`, require mask 0x300
never raised, 60s timeout, board soft-reset + self-healed). Diagnosis:
- The `-2` first seen was my chain_progs SCANNING for a pre-existing 0x0010/0x0101 descriptor; the slot is
  WRITTEN at a fixed offset (matmul=word 216), not pre-existing. Fixed -> explicit desc_slot per program.
- Offline template dump: words 216-217 = 0 (next-addr), 218-219 = 0x0014/0x0101 (end-descriptor). set_i16_out
  regs live at words 118-188 -> NO clobber either way. Clobber hypothesis dead.
- **THE REAL FINDING**: re-ran Phase-0 chain_probe -> rc=0, silu(task1) CORRECT, but **mm_ran(task0)=0**.
  Phase-0 NEVER actually ran the matmul task0 (its matmul was a dummy nothing consumed; only the silu was
  validated). So the heterogeneous 2-task chain where **task0 (matmul) actually EXECUTES and feeds task1**
  has never worked. My increment-1 hangs because its silu DEPENDS on task0's (absent) output.
- OPEN RE: why does task0 (the first, matmul) not execute in the PC-walk? The kernel programs the PC from
  the first task and the HW walks descriptors (AGENTS.md). Suspect the multi-task PC programming / subcore
  setup / task ordering. Cross-ref: vendor 27-task softmax chain decode + rocket_job.c multi-task IRQ re-arm.

## DECODED: how the vendor chain feeds task N -> task N+1 (2026-07-12)
Decoded the captured 9-task softmax chain (regcmd_softmax_f16.h) + CONFIRMED the replay PASSES on-board
(single submit, task_number=9, ping-pong OFF, rc=0, softmax=1/64 exact). Per-task in/out addresses:
  t0 SDP  in=IN(fffb6000)      -> out fffc6040
  t1 SDP  in=IN+0x800          -> out fffc6140
  t2 CNA  feat=fffc6040(=t0)   wt=ffffa680 -> out fffc7040        <- 0xd matmul reading t0's output
  t3 SDP  in=IN, in2=fffc7040(=t2) -> out fffcf040                <- SDP reading t2's (matmul) output via 0x5038
  t4 SDP  in=out=ffffab00 (exp LUT-load into SRAM)
  t5 SDP  in=fffcf040(=t3) LUTrd=ffffc000 -> out fffd7040 (exp)
  t6 CNA  feat=fffd7040(=t5)   -> out fffc6000
  t7 CNA  feat=fffc6000(=t6)   -> out fffc8000
  t8 SDP  in=fffd7040(=t5), in2=fffc8000(=t7) -> OUT(fffae000)    <- final divide, TWO producers
**MECHANISM: pure ADDRESS ALIASING through resident scratch buffers.** Task writes output to a scratch
IOVA (0x4020 for SDP/DPU out; CNA output for 0xd); a downstream task names that SAME IOVA as its input
(0x5018/0x5038 SDP, 0x1070 CNA feature). NO forwarding datapath, NO chain-output register -- the PC
descriptor (0x0010/0x0014) only sequences WHICH regcmd runs next; data flows through shared addresses in a
persistent image with NO act(RESET) between tasks. It's a DAG (t5 feeds both t6 and t8), not a linear pipe.
=> matmul(0xd)->silu(0x18) is PROVEN (t2->t3). My aliasing (matmul->G->silu via 0x5018) is structurally right.

WORKING RECIPE (from the passing replay): ONE contiguous IOVA image (tasks + data + weights), single-delta
rebase, tasks at 64-byte-aligned slots (VOFF/TADDR), 0x0014 next-amount recomputed = (next_amt+3)/2,
flags=0x1 (ping-pong OFF), task_number=N, subcore_task[0]={0,N}, one submit.

WHY increment-1 HANGS (task counter 0x0): NOT the aliasing/transition (proven). Phase-0's plain matmul task0
also never ran (mm_ran=0) -- it completed only because its silu was self-sufficient. My silu DEPENDS on
task0's output, so when task0 (int16 matmul via set_i16_out) stalls, the whole chain stalls. Prime suspect:
set_i16_out's FIXED 0x4050=0x0248 row-stride is not N-derived -> at N=64 the int16 write likely runs OOB /
stalls the DPU in-chain. NEXT: (a) verify/fix set_i16_out output stride is N-correct (bound the G write), OR
(b) rebuild the chain the vendor's way (contiguous image + aligned slots) rather than hand-rolled separate
buffers. Cheapest decisive test: kernel-sequenced 2-submit bridge (matmul->G, then silu<-G, separate submits,
no reset) to confirm the int16 bridge data is coherent BEFORE fighting the HW-chain walk.

## ROOT CAUSE of increment-1 (2026-07-12): the int16-OUTPUT matmul (set_i16_out) WEDGES
Kernel-sequenced bridge test (ORK_GS_SEQ) pinned it: the **matmul submit itself wedges (errno=110)** as a
plain task_number=1 op -- NOT a chain-walk issue. Isolation:
- ORK_GS_NOLUT (matmul is the FIRST op after act RESET, no silu LUT-load before it): STILL wedges -> not a
  mode/ordering poison from the LUT-load.
- Ran the PROVEN i16out_probe (ork_npu_probe_i16_out) directly at BOTH 8x32x64 and 8x512x64: BOTH hang
  (~60s submit timeout). So set_i16_out int16-output matmul wedges regardless of my code / K.
- NPU is HEALTHY: softmax_replay PASSES (1/64 exact) + plain int8 test_matmul runs clean (mism=0) in the
  same session. So it's specifically the int8-matmul-with-int16-output stage that wedges, not the board.
=> The summary's "ork_npu_probe_i16_out validated int16 output" does NOT hold on silicon now (its Test1 is
   all-ones/layout-independent and may never have exercised a clean submit). The int16 DATA BRIDGE premise
   (matmul writes int16 directly for the silu to read) is blocked by a broken int16-output matmul.

## RE(A) RESULT: set_i16_out is broken in MULTIPLE output-stage regs, no int16 reference exists
Diffed the DPU output-stage 0x40xx regs: int32 base (REGCMD_I8, works) vs vendor CNA producers t2/t6
(0xd -> feed SDP, work, but FP16) vs set_i16_out. Deviations in set_i16_out:
- **0x4010 (DATA_FORMAT)**: set to 0x0000 = OUT_PRECISION[31:29]=int8(!), NOT int16(0x20000000). Real bug
  (int32 base=0x80000000=int32; vendor fp16 t6=0x48000002=fp16). Precision said int8 while strides said 2-byte.
- 0x40c0: set_i16_out=0x40; vendor 2-byte producer=0x2000; int32 base=0x80. set value matches neither.
- 0x4050: set_i16_out=0x248; vendor 2-byte=0x126; int32=0x7fc. Geometry-dependent, unmatched.
- 0x4038/0x4058/0x405c/0x4034 also differ from the vendor producer.
TESTED (env overrides, no recompile): 0x4010=0x20000000 -> STILL wedges; +0x40c0=0x2000 -> STILL wedges.
=> Multi-register-broken. And there is NO captured int8->int16-output reference (the only working 2-byte
   producer is fp16 + different geometry). Making int8-matmul->int16-output work is a REFERENCE-LESS full
   output-stage RE -- high cost, uncertain payoff. set_i16_out was an incomplete experiment, not a primitive.
CONCLUSION: do NOT resurrect set_i16_out for the bridge. Pivot to a bridge with a WORKING reference.

## ggml-ork INTEGRATION + ork_bench (2026-07-12): CHAIN5 engages on the real model; bench blocked by IOVA footprint
Landed the chain assembler to ork-driver main (pushed) + bumped both fork submodules (local feat/ork-static-graph
+ board ~/llama-q4-v0659). Wired ggml-ork ORK_FFN_CHAIN5: prep packs per-tensor fc.wu/fc.wd (+ rough scales
s_up/s_glu); the handler calls ork_mm_run_chain_i8_ffn for [gate->silu->up->glu->down(K-split)] one submit.
Also fixed a PRE-EXISTING ORK_FFN_CHAIN decode bug (GLU supports_op claimed at M>=1 but the handler declines
M<32 -> claimed-but-uncomputed -> graph_compute -1; gated the GLU claim on M>=32).
RESULTS (Qwen3-1.7B-Q8_K_XL, orkpack, P=256):
- BASELINE all-int8 per-op (ORK_FFN_CHAIN+FUSED_SILU): prefill **76.86 tok/s** (decode fails separately -
  a distinct pre-existing decode issue; prefill is the valid number).
- CHAIN5: **ENGAGES + runs the real full-width FFN inner (Nff=6144 Kd=2048) in ONE submit on silicon**
  ("[ORK FFN-CHAIN5] one-submit FFN inner ok" per layer, confirmed). BUT no stable clean prefill number:
  the extra per-tensor fc.wu/fc.wd packs push the IOVA footprint over -> intermittent then consistent
  warmup-decode graph_compute -1 / soft-reset. This is the documented FFN-chain footprint / multi-domain
  non-determinism ([[ffn-chain-footprint-one-domain]], [[multi-domain-runtime]]), NOT the chain assembler.
CONCLUSION: integration DONE (CHAIN5 wired + demonstrably running the real FFN inner one-submit). A clean
CHAIN5 speed number needs a footprint workstream (compact orkpack + PRE-PACKED chain weights so fc.wu/wd
aren't live-packed on top of the resident model + domain sizing). AND the expected speed is ~NEUTRAL anyway:
prefill is CPU-bound (act-quant + attention compute; NPU ~17-83% idle - established) and run_chain_i8 is
SINGLE-CORE, so collapsing the FFN's NPU submits doesn't move the CPU-bound prefill wall-clock (exactly what
the prior static-graph work found: matmul chaining is coherent-but-neutral on CPU-bound prefill).
NEXT (for a real speed win): the lever is CPU-side (act-quant/attention on NPU or overlap), not FFN submit
count. The chain assembler is the validated primitive for when the NPU slice becomes the bottleneck.

## ★★★ FULL FFN INNER CHAIN WORKS (2026-07-12): [gate->silu->up->glu->down] ONE submit, bit-exact
FFN5 (chain_gu_silu_probe ORK_GSILU_FFN5, M=8 K=512 Nff=512, all-ones): rc=0, gate=32, silu=61, up=32,
glu=61, **down=31232 = glu(61)*512 EXACT** (all 4096/4096, no wedge). All FIVE ops chained in ONE submit:
matmul -> SDP -> matmul -> SDP -> matmul.
- Added: a MATMUL task can read a PRIOR task's output as its activation (ops[i].in0>=0 -> out_dma[in0];
  -1 -> tasks[i].A). down reads glu (task3) -> validates the SDP-output -> matmul-activation-input bridge
  (down=31232 exact). BOTH directions of matmul<->SDP aliasing now proven.
- The general assembler ork_mm_run_chain_i8_ffn(S, tasks, ops[], in_scale, out_scale) handles the full mixed
  chain from a per-task op list. This is the FFN assembler.
REMAINING for production:
  1. K-SPLIT down for real Nff=6144 (down contraction 6144 > 4096; test used Nff=512, single program). run_chain
     needs to K-split a single matmul task into accumulate-programs WITHIN the chain (methodology "internal
     K-split != boundary"). Currently a task with K>4096 returns -3.
  2. Wire into ggml-ork: replace the per-node FFN inner with one ork_mm_run_chain_i8_ffn call; per-row/per-channel
     scales (not the test's per-tensor); measure prefill vs baseline + ork_ppl coherence.
  3. Fused RESIDUAL on down's output stage (user): add the residual to down's accumulator before requant
     (set_i8_out8-stage add / SDP EW-add) -> keeps the residual in the chain, precision-preserving.
  4. Attention block chain [QK^T -> softmax-SDP -> A.V] as the second assembled unit (same primitives).

## ★★ 4-TASK FFN CHAIN WORKS (2026-07-12): [gate -> silu -> up -> glu] one submit, bit-coherent
General per-task-op assembler `ork_mm_run_chain_i8_ffn(S, tasks, ops[], in_scale, out_scale)`: ops[i].kind =
OP_MM32/OP_MM8/OP_SILU/OP_EWMUL; SDP tasks read prior outputs by index (in0/in1), aliased. Chains
`[gate(MM8) -> silu(SDP,in0=gate) -> up(MM8) -> glu(EWMUL,in0=silu,in1=up)]` in ONE submit.
RESULT (chain_gu_silu_probe ORK_GSILU_FFN4, K=512 all-ones): rc=0, gate=32, silu=61, up=32, glu=61 — all
512/512, no wedge. FOUR heterogeneous ops (matmul->SDP->matmul->SDP) chained coherently.
KEY FIX: the PC next-descriptor of a MIDDLE task lives at word **2*regcfg** (matmul 108 -> 216; SDP 69 -> 138),
NOT always 216. The 2-task chain worked only because the silu was last (no descriptor); a middle SDP task's
forward-descriptor must be written at 138. (This is the general rule for chaining any regcfg-N task.)
=> The FFN inner minus down is a working one-submit chain. REMAINING: add **down** (task4, MM32, reads glu;
   K-split since down K = Nff = 6144 > 4096 -> multiple chained programs + accumulate). Then wire into ggml-ork
   + measure prefill vs baseline + coherence (ork_ppl). The fused residual (user) = a set_i8_out8-stage add on
   down's output.

## ★ OPTION B WORKS (2026-07-12): one-submit [gate matmul(int8-out) -> silu-SDP] coherent
ork_mm_run_chain_i8_sdpsilu: task[sdp_task] is a STANDALONE int8 silu-SDP op (REGCMD_SILU_STD, 0x18/69)
reading task[sdp_task-1]'s output via ALIASED buffers; the gate matmul gets set_i8_out8 (plain int8 output);
silu LUT built internally (silu_build_curve, same as ork_npu_silu_i8); ping-pong OFF for the chain.
RESULT (chain_gu_silu_probe ORK_GSILU_SDP, K=512 all-ones):
  rc=0  task0 gate int8 = 32 (512*R=1/16) 512/512  |  task1 silu int8 = 61 (~silu(3.0)/os=60) 512/512
  => ONE SUBMIT, no wedge, BIT-COHERENT. The vendor matmul->SDP pattern works for our int8 FFN.
PROVES: (1) a matmul with int8 output (set_i8_out8) DOES chain as a non-last task -> the earlier wedge was
specifically the FUSED-ACTIVATION output stage (set_i8_silu), not int8-output; (2) a standalone silu-SDP task
chains + reads the gate's output via address aliasing (the decoded vendor mechanism); (3) the int8 matmul
output cube MATCHES the silu-SDP EWCUBE input -> the bridge is coherent.
=> The assembler foundation is VALIDATED. Build out the full FFN inner as separate chained tasks:
   [gate matmul(int8) -> silu-SDP -> up matmul(int8) -> glu-ewmul-SDP -> down matmul]. Each is a plain matmul
   or a standalone SDP op (both proven to chain), data via aliased buffers. NEXT: add up + glu-ewmul (the
   ewmul SDP op reads silu-out AND up-out as its two operands -> 0x5018/0x5038 aliased), then down (K-split).

## BISECTION RESULT (2026-07-12): the fused-silu OUTPUT STAGE won't survive run_chain_i8's chain submit
3-way bisection of run_chain_i8_gsilu [gate*silu -> up] via chain_gu_silu_probe env gates:
- (1) PLAIN ss=NULL (ork_mm_run_chain_i8): rc=0, gate 512/512, up 512/512 -> **PLAIN PATH OK** (refactor clean;
  run_chain_i8 executes BOTH chained tasks -> chain_progs's empty-output bug was its own, not run_chain_i8).
- (2) NOSILU (LUT-load ON, set_i8_silu OFF): rc=0, up 512/512, gate ran as plain int32 -> **LUT-load-in-chain OK**.
- (3) NOLUT (set_i8_silu ON, LUT-load OFF): **WEDGED**.  + FULL (both): WEDGED.  + SILU-LAST ([up -> gate*silu],
  silu as the last task): **WEDGED** too.
=> The wedge is the **set_i8_silu fused-activation output stage on ANY task in a run_chain_i8 chain** (non-last
   OR last), independent of the LUT-load. Yet it works STANDALONE via ork_mm_run_i8_silu. The difference:
   ork_mm_run_i8_silu submits via **submit1() with the WARMED c->Cc buffer** ("a hand-rolled submit to a fresh
   buffer gives bias-only output" ~npu.c 4313); run_chain_i8 writes each task to a FRESH tmp_C via the chain
   submit path. The fused-silu output stage requires submit1/warmed-c->Cc; the chain submit path can't drive it.

## NEXT (fix options for the fused gate*silu in a chain)
- A. Route the gate task's output through the warmed c->Cc inside run_chain_i8_impl when ss (mirror
     ork_mm_run_i8_silu's submit1 buffer handling) -- may require submit1-style warming, not the chain submit.
- B. VENDOR PATTERN (cleaner): don't fuse silu into the matmul output at all -- make silu its OWN SDP task in
     the chain (like the softmax t4/t5). I.e. chain [gate matmul(int32/int8) -> silu-SDP -> up ...] with silu
     as a standalone 0x18 task. But a standalone int8 silu SDP reading the gate's chained output = the
     matmul->SDP data bridge (address aliasing, decoded above) -- which is what the assembler needs anyway.
     This sidesteps the "fused output stage won't chain" wall entirely and matches how the vendor does it.
- Recommend B: it's the vendor-proven structure (matmul and activation are SEPARATE chained tasks, data via
  aliased buffers) and avoids the fused-output-stage-vs-chain-submit incompatibility.

## run_chain_i8_gsilu built (extends the proven base) -- FIRST RUN WEDGED (2026-07-12)
Refactored run_chain_i8 -> static run_chain_i8_impl(...,ss) + public ork_mm_run_chain_i8 (ss=NULL, IDENTICAL
behavior, plain path preserved) + new ork_mm_run_chain_i8_gsilu(ss): the gate task's regcmd gets set_i8_silu
(int8 silu-out), + the silu LUT streamed to SDP SRAM once before the chain. Probe chain_gu_silu_probe:
[gate*silu(int8) -> up(int32)] one submit, all-ones (gate=up=K), K=512 (run_chain_i8 envelope K%512, <=4096).
FIRST RUN: WEDGED (no output, gsilu hung; board self-healed, load 0.63). NOT the plain path (ss=NULL untouched).
Suspects for the fused-silu-in-chain wedge (NEXT):
 - the "hand-rolled submit to a FRESH buffer gives bias-only output" warning (ork_mm_run_i8_silu ~4313): the
   fused-silu matmul may REQUIRE the warmed c->Cc buffer, but run_chain_i8 writes fresh tmp_C. Try routing the
   gate task's output through c->Cc, or warming tmp_C first.
 - LUT-load-in-chain ordering vs the warm block's act(RESET): my LUT-load runs AFTER the build loop but the
   warm block's act(RESET) ran at entry -- confirm the LUT survives into the chain submit (it should; single
   submit gap). vs ork_mm_run_i8_silu which does LUT-load then per-tile matmul via submit1.
 - mixed int8(gate)+int32(up) output in one chain -- untested combination.
 - reps=2 warm-up re-running the chain (incl the silu task) -- fine for output, but interacts with LUT state?
De-risk: first confirm ss=NULL (plain run_chain_i8) still works post-refactor (it's byte-identical, but
verify via ORK_GU_CHAIN or a plain 2-matmul run). Then bisect the silu wedge (drop the LUT-load; drop set_i8_silu).

## chain_progs is a DEAD END -- extend the PROVEN run_chain_i8 instead (2026-07-12)
Built a self-test (ork_npu_chain_selftest): chain 2 plain int8 matmuls (all-ones -> every out == K) via
ork_npu_chain_progs. RESULT: EMPTY output (C0/C1 all 0) even at **n=1** (single task, no chaining) -- the
submit returns rc=0 (no WARNING, no wedge; errno=22 is stale) but the matmul computes nothing. Ruled out:
warm-up (reps=2 mirror of run_chain_i8), all-3 subcore_task, ppflags (0x5), dom (-1 and 0 both empty). The
bug is subtle and unfound -- my from-scratch raw-submit deviates from run_chain_i8 in some way I can't see.
=> STOP debugging chain_progs. **run_chain_i8 ALREADY chains N int8 matmuls in one submit and WORKS**
   (ggml-ork ORK_GU_CHAIN chains gate+up via ork_mm_run_chain_i8, line ~4013). It correctly warms, resolves
   buffers (tmp_A/dma_find + w->Bf/Bb), submits, and produces output. The assembler must be built by
   EXTENDING run_chain_i8 to heterogeneous tasks (apply set_i8_silu to the gate task + load the silu LUT),
   NOT on the parallel chain_progs. chain_progs kept (gated/probe-only) but superseded.

CONCRETE increment (corrected): a run_chain_i8 variant / option where task[i] carries an optional silu-LUT
flag -> that task's regcmd gets set_i8_silu (fused gate*silu, int8 out) instead of the plain int32 output.
Chain [gate*silu -> up] in one submit reusing run_chain_i8's proven machinery. Then glu-ewmul + down.

## ANSWER: how ORK_FFN_CHAIN bridges matmul->silu (read ggml-ork.cpp 3975-4112)
Three paths; only ONE is pure-NPU:
1. **FUSED int8 (pure-NPU, round-trip-free): `ork_mm_run_i8_silu`** (npu.c 4286). Loads the SiLU LUT into SDP
   SRAM once (enable 0x18, ping-pong OFF), then the gate matmul applies silu+requant IN ITS OUTPUT STAGE via
   **`set_i8_silu(rc,N,0,r_mult,r_shift,out_bias,idx_off,cfg4068)`** -> int8 out. THIS is the working
   "fuse into the SDP output stage" mechanism. Coarse (PPL 55) from int8-out + per-tensor scale, NOT a broken
   mechanism. Variant **`ork_mm_run_i8_silu32`** (4395) emits INT32 (fine precision). A fused SwiGLU already
   exists (4671): gate-silu -> G, then up's SDP ewmul fetches G as its 2nd operand -> glu on-NPU.
2. int16 (higher accuracy, NOT pure-NPU): gate matmul -> NATIVE int32 -> **CPU** dequant(per-row x per-chan)
   + requant->int16 -> ork_npu_silu_i16 (NPU) -> **CPU** dequant. Two CPU steps between the NPU ops. This is
   why set_i16_out was never load-bearing (confirmed: the production int16 path requants on CPU).
3. CPU silu (shipped default): int32 matmul -> CPU fp32 silu.

**=> set_i16_out is a DEAD END. The working matmul->silu primitive is the output-stage fused LUT (set_i8_silu),
which is exactly the SDP-output-stage fusion the user flagged for the residual.** The assembler should chain
the EXISTING fused ops (ork_mm_run_i8_silu / _silu32 + the fused-SwiGLU ewmul), and the residual fuses into
the SAME output stage (set_i8_silu / down-proj output: accumulator + residual before the requant/LUT).

## STRATEGIC PIVOT (recommended): reuse ggml-ork's EXISTING working matmul->silu bridge
ggml-ork's on-NPU int16-silu FFN chain WORKS today (coherent PPL 19.02, [[int16-silu-pipeline-transition-wedge]]).
It must bridge matmul->silu somehow (likely int32-matmul-out + a CPU or on-NPU int32->int16 requant, since
set_i16_out doesn't work). NEXT: read how ORK_FFN_CHAIN in ggml-ork.cpp actually does the gate-matmul ->
silu handoff -- that IS the proven bridge to build the assembler on, instead of the broken int16-out matmul.
If it's a CPU requant, the "one-submit FFN chain" needs the requant folded into an on-NPU task (option C) or
the whole FFN kept fp16 (vendor pattern, ~3.3x matmul cost). Decide after reading the working path.

## BRIDGE FORK (needs a call) -- how the FFN matmul feeds the silu without the int16-out matmul
- A. RE-fix set_i16_out: why does int8-matmul + int16-output wedge? Suspects: 0x4010 precision value,
     fixed 0x4050=0x0248 row-stride (not N-derived), 0x4038 group stride. Deep output-stage RE.
- B. int32-native bridge: matmul writes its NATIVE int32 output (proven to work); the silu/SDP consumes int32
     and requants internally (does the SDP int16-silu accept an int32 input surface + scale? -> RE 0x5xxx).
- C. requant task in-chain: matmul(int32) -> tiny requant op (int32->int16) -> silu. One extra task, all-NPU.
- D. vendor pattern is FP16 intermediates (softmax t2->t3 is fp16, works) BUT int8-MAC->fp16-out is NOT a
     datapath ([[int8-fp16-fused-not-a-datapath]]) -> not available for w8a8. Only a full fp16 matmul path.
Note: ggml-ork's working on-NPU int16 silu ([[int16-silu-pipeline-transition-wedge]]) likely requants
int32->int16 on CPU between matmul and silu (NOT a pure-NPU bridge) -- consistent with set_i16_out not working.

## RESIDUAL via SDP OUTPUT-STAGE FUSION (user direction, 2026-07-12) -- supersedes CPU-seam residual
Precision-preserving on-NPU residual = FUSE the add INTO the matmul's SDP output stage (accumulator +
residual BEFORE the int16/fp16 cast), NOT a separate ewmul/add op. The int32 accumulator + residual add
happens in high precision, then one cast -> no fp16 accumulation blow-up over layers, AND no CPU seam (the
residual stays in the chain). This is the NVDLA SDP element-wise / bias add sub-unit (a second operand read
from DRAM, added at the X1/X2 stage before the Y activation/cast). Applies to the FFN down-proj and the
attention O-proj (each ends the block + owns the residual). RE target: the SDP EW-add operand-addr + enable
regs in the 0x40xx block (cross-ref rocket_registers.h SDP/DPU + NVDLA SDP docs). This removes the residual
seam that fragments the chain -> a whole attention block or FFN can end with a fused-residual output.

## Next steps
1. [in progress] Build + validate increment-1 data-connected matmul-i16-out -> silu chain (new fn +
   chain_gatesilu_probe). Coherence = silu(matmul(A,W)) within tol; rc=0 no wedge.
2. Add up + glu (ewmul) -> 4-op chain; validate.
3. Add down (K-split sub-chain) -> full FFN inner one submit; validate vs separate-op FFN.
4. Wire into ggml-ork FFN handler behind a flag; measure prefill vs 97 baseline + coherence (ork_ppl).
5. Attention-block chain (QK^T + softmax + A.V) as the second assembled unit.

## Rules
- Coherence-gate every increment (output vs separate-op reference). errno=110 = real wedge (NOT dmesg
  "soft reset"). `timeout` every board NPU cmd; SIGTERM not kill -9. Board self-heals; hard wedge -> plug.
- Commit each validated increment. Never clobber board WIP (board tree == feat/static-graph, synced by file).
