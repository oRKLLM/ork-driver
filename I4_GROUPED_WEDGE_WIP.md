# I4 grouped-W4A4 prefill wedge — WIP recovery doc

## CORRECTION (2026-08-07): int4 SHOULD be a dt-branch in ork_dyn_begin_colsplit (the reference)
Earlier "BCHAIN can't fold — no h" was WRONG. The int8 REFERENCE is `ork_dyn_begin_colsplit`@11118: a
dt-parametrized doorbell batch-chain that builds per-core chains into an ork_dyn_chain h, runs PARALLEL
per-core workers (ork_csub_worker / npu_pool, prepolled), and rides ork_dyn_end for recover+dump. It already
carries dt==DT_I8 AND dt==DT_F16 branches (fp16 was Stage 1 / #47). => int4 belongs as a **dt==DT_I4 branch
there**, mirroring fp16 — keeps parallel per-core perf AND unifies onto the shared drain. Feasible, correct.

PLAN (mirror the fp16 Stage-1 branch @11150; this is a #46-#50-scale staged port, bit-exact + perf-gated):
  1. Route int4 M>=2 (currently -> run_i4_bchain_db@4413) into ork_dyn_begin_colsplit with dt==DT_I4.
  2. In colsplit, add the dt==DT_I4 per-core branch: A-slice gather (tile_i4_Aslice), synth_i4 per (ns,g) into
     the core's chain (as run_i4_bchain_db@13850 does), int16 partial scratch in mcc[i], set h->esz=2, h->mc_dt=DT_I4.
  3. FINALIZE (the int4 tail, analogous to fp16's f32 accumulate / int8 oscat): the bch de-tile int16->int32 C
     (bch_db_cells mode 2 logic) as the per-core accumulate in the worker, so ork_dyn_end's prepolled path drains it.
  4. Ride ork_dyn_end: set h->prepolled after the workers drain (like colsplit int8/fp16) so recover+dump come free;
     delete run_i4_bchain_db's bespoke poll/reset once parity holds (keep it until then).
  5. VALIDATE: make test byte-identical all precisions; i4/test_chain_i4/perplexity_i4 bit-exact; A/B perf every
     int4 shape decode M=1 + prefill M>=2 vs run_i4_bchain_db (no regression — colsplit's balanced split + parallel
     accumulate should match/beat it). Then grouped + this int4-BCHAIN + int8/fp16 ALL ride colsplit+ork_dyn_end.
SCALE: fp16 colsplit was 5 staged tasks (#46-#50). This is comparable careful work (bit-exact int4 port into the
shared int8/fp16 colsplit fn = high blast radius) + full validation — a focused effort, not a single fumes-pass.

## SINGLE-DRAIN-CORE UNIFICATION (2026-08-07) — scope reality (evidence-backed)
DONE (ca5aff6): all int4 paths FUNCTIONALLY on the doorbell recover+dump — grouped shares
mc_recover_resubmit + ork_dyn_dump with int8's ork_dyn_end; M=1 rides ork_dyn_end; BCHAIN has its own
equivalent recover. make test all-precisions PASS; native-W4A4 storm-free.

ACHIEVABLE (safe, cleanliness only — NOT done here): fold int8/fp16 (ork_dyn_end) + grouped
(ork_dyn_grouped_end) onto ONE `ork_dyn_drain_poll(h)` helper (dedup the now-duplicated poll+recover+dump
loop, ork_dyn_end@12153-12210 == grouped@10890+). CAUTION: ork_dyn_end is the shared int8+fp16 drain
(highest blast radius) and its g_in_doorbell lifecycle spans into finalize — extract carefully, `make test`
byte-identical all precisions. Zero functional gain (recover+dump already there) -> low priority; fresh context.

INFEASIBLE without perf regression: **fold BCHAIN in.** run_i4_bchain_db@13777 uses NO ork_dyn_chain h at
all — struct bchdbw args[] across PARALLEL npu_pool workers (bch_db_worker), poll/de-tile via bch_db_cells
(per-line int16->int32), own reset loop@13836. Rewriting it onto the serial h-based SENT16 drain would lose
its parallel per-core de-tile = a direct perf regression (violates the no-regression bar) + correctness risk.
BCHAIN stays its own drain BY NECESSITY; it already has recover (could add ork_dyn_dump on exhausted recover@13836
for dump-parity — a small safe follow-on). A true single core spanning both would need a higher-level abstraction
(per-core worker fn + recover cb + dump) — large, not clearly better; only if a future need justifies it.


## REIMPLEMENTATION SPEC (2026-08-07) — "int4 fully rides the doorbell, no perf regression"
DEFECT (#33 spec miss): int4 paths drain via BESPOKE code, not the shared doorbell drain ork_dyn_end
(auto-dump@12189 + recover loop@12163). Inventory:
  - int8/fp16: ork_dyn_end (THE doorbell drain — poll + mc_recover_resubmit + auto ork_dyn_dump). GOOD.
  - grouped W4A4: ork_dyn_begin_mc_i4_grouped (submit, on doorbell) + ork_dyn_grouped_end@10885 — BESPOKE
    poll, NO recover, NO dump (float scale-accumulate tail). <- worst offender.
  - BCHAIN int4: run_i4_bchain_db@13777 — its OWN 6-reset recover loop@13813 + de-tile (parallel; not ork_dyn_end).
  - M=1 int4: run_i4_mc_db -> ork_dyn_begin_mc_i4 -> ork_dyn_end? (verify; if yes it already rides it).

PLAN (staged, each `make test` byte-identical + A/B perf-gated):
  1. EXTRACT a drain-core from ork_dyn_end: `ork_dyn_drain(h, finalize_cb)` = the poll-all-S + recover-missed
     (mc_recover_resubmit) + auto ork_dyn_dump, with the op-specific tail as a callback:
       - int8/fp16 finalize = existing int32-widen/copy/oscat (behavior byte-identical -> regression-proof).
       - grouped finalize   = the float scale-accumulate (move from ork_dyn_grouped_end).
       - BCHAIN finalize    = the per-line de-tile (move from run_i4_bchain_db); drop its private reset loop.
     Keep h->i4g / h->esz / h->oM etc. as the per-op descriptors the finalize_cb reads.
  2. Point ork_dyn_grouped_end + run_i4_bchain_db (+ verify M=1) at ork_dyn_drain. Delete the bespoke polls/resets.
  3. PERF GATE (AGENTS mandatory): A/B every int4 shape BOTH regimes vs current HEAD — decode M=1 (per-row) and
     prefill M>=2 (grouped + BCHAIN + the M-chunk rescue). No regression allowed; update thresholds only if faster.
     NOTE the known perf pathology: native-W4A4 M-chunk rescue = ~1806 micro-submits/forward (SLOW at prefill by
     design). Riding the doorbell fixes ROBUSTNESS (self-heal + dump), NOT that slowness — do not conflate; the
     "no regression" bar is vs current int4, and native-W4A4-at-prefill stays opt-in (route B/W8A8 is the default).
  4. Validate: make test all knobs + mode_probe + the ORK_MIXED_W4A4 full-model prefill (self-heals now, dumps a
     true stall) + ORK_QUANT=4 route-B unaffected.
GATING for this whole effort: fresh session, full context (core drain shared by int8/fp16 = highest blast radius).


## DATA-DRIVEN REFRAME (2026-08-07, from the doorbell dump — supersedes "hard wedge")
Wired ork_dyn_dump into the grouped drain (ork_dyn_grouped_end@~10891) + repro'd native-W4A4 prefill
(ORK_QUANT=4 ORK_MIXED_W4A4=1, F16, P=32) with ORK_GRP_DEBUG. FINDINGS:
  - DYN-DUMP fired **0 times** — NO userspace-detected stall; every grouped submit completed.
  - **1806** grouped ops in ONE forward: the M-chunk rescue explodes each native-W4A4 matmul into a swarm
    of tiny M=3/M=2 submits (M_padded=32 -> ~11 chunks, x28 layers x all projections).
  - Run was CUT BY TIMEOUT mid-warmup (last line = sched_reserve), NOT stalled.
=> The "wedge" is dominantly a PERFORMANCE PATHOLOGY (≈1806 micro-submits/forward -> pathologically slow ->
   timeout cuts it) + INTERMITTENT kernel-recovered doorbell misses (the earlier "reset storms" — recovered
   transparently, which is why no dump fired), with only a rare genuine hard-hang. Matches AGENTS: native
   W4A4 "loses to int8 at prefill" — it is not meant to run prefill. Default int4 = route B (W8A8), fast+robust.
ARCHITECTURAL FIX (the real one, per user): make the grouped drain RIDE ork_dyn_end (auto-dump@12189 +
   recover loop) instead of bespoke ork_dyn_grouped_end (no recover, silent timeout). Keep only the float
   scale-accumulate as the custom tail. Then intermittent misses self-heal + rare true stalls auto-dump.
   (Bolted a stopgap ork_dyn_dump into ork_dyn_grouped_end this session — uncommitted; the ride-ork_dyn_end
   refactor is the correct replacement.) The perf pathology (M-chunk swarm) is inherent to native-W4A4-at-prefill.


## RESOLVED (2026-08-07) via route B — int4 build+serve now run on the NPU
Re-routed `ORK_QUANT=4` off the fragile native-W4A4 grouped path onto the **W4A8-inflate (i4a8)**
path: COMPUTE = W8A8 on the NPU (int4 weights inflated int4->int8, `mul_mat_i8` — robust, no wedge),
STORAGE = compact i4a8. Two changes in ggml-ork.cpp:
  1. dispatch (~6155): `qbits==4` -> `mul_mat_i8` unless `ORK_HADAMARD`/`ORK_MIXED_W4A4` (native W4A4 now opt-in only).
  2. `ork_orkpack_tier` (~937): `ORK_QUANT=4` forces tier 4 (compact i4a8 storage) regardless of source.
  (+ earlier: int4 persist via pure-CPU `ork_pack_i4a8_cpu_blob`, no `!npu_busy` skip.)
RESULT (Qwen3-1.7B Q8 gguf, ORK_QUANT=4): build packs 196 weights -> 674 MiB compact .q4.orkpack, no wedge;
timed run reads it clean (loaded 193/packed 0), prefill ~122 t/s / decode ~2.53 t/s @ P=64 on the NPU.
NOTE: i4a8 storage re-quantizes to crude symmetric int4 (~+36% PPL vs int8 — inherent int4 tradeoff).
The native-W4A4 grouped prefill wedge below remains a KNOWN issue but is now opt-in-only (not on any default path).



## Goal
Make the on-NPU **int4 orkpack build** work. Root blocker: native-W4A4 (`ORK_QUANT=4`) prefill
**wedges the NPU** (unkillable hang + `RKNPU: soft reset` storm), so the convert forward never
finishes packing. User chose: FIX the native-W4A4 prefill wedge (not the i4a8/W8A8 workaround).

## What is proven
- Auto-persist framework DONE + validated for **int8** (build-if-absent, footer `quant_sig` v5,
  regenerate+delete-old, warnings, SUCCESS msg). Files: `ggml/src/ggml-ork/ggml-ork.cpp`,
  `ggml/include/ggml-ork.h`, `ork-driver/tools/re/ork_bench.cpp`. int8 orkpack builds clean (2.3 GiB).
- int4 build (`ORK_QUANT=4`) wedges. `ork_bench` dispatch → `mul_mat_i4` (native W4A4 grouped,
  `ggml-ork.cpp:1904`). Prefill pads `M_padded=((M+31)/32)*32` → M=4 becomes **32**.
- `ork_mm_run_i4_grouped` (`npu.c:4992`): direct doorbell refuses when per-core programs
  `(M/nc)*Sn*Sk` exceed ~70 → #33 M-chunk rescue recurses down to M=1 chunks. Each M=1 chunk emits
  **Sk programs** (grouped doorbell `ork_dyn_begin_mc_i4_grouped` `npu.c:10818`).
- Down-proj: K=6144, G=128 → **Sk=48**. Passing `one_i4g` test is K=2048 → **Sk=16**.
  HYPOTHESIS (UNCONFIRMED): a grouped M=1 submit at high Sk (≈48) wedges; Sk=16 works.

## Next step (contained repro — do NOT full-model wedge)
- Added `ORK_GRP_DEBUG` geometry log in `ork_dyn_begin_mc_i4_grouped` + `ORK_TEST_GRP` K-sweep in
  `examples/test_slice_rescue.c` (M=1, N=2048, G=128, K=2048/4096/6144/8192 → Sk=16/32/48/64).
- Build `test_slice_rescue`, run `sudo env ORK_TEST_GRP=1 ORK_GRP_DEBUG=1 timeout 90 ./test_slice_rescue`.
  Last geometry line with no following "OK" = the wedging Sk. A wedge hangs unkillably → graceful reboot after.

## RESULT of the contained repro (2026-08-07) — hypothesis DISPROVEN
`ORK_TEST_GRP` sweep (M=1 Sk=16/32/48/64, AND M=32 Sk=48 = the exact convert shape, progs/core~528
via the rescue) ALL PASS: `rc=0, maxerr=0.0000`, no wedge, board stayed healthy. So the grouped
W4A4 kernel is CORRECT + non-wedging in isolation at every shape the model uses. Combined with the
earlier BCHAIN standalone pass (K=6144 M=128 clean), **both int4 kernels are fine standalone.**

=> The int4 wedge is a FULL-MODEL-CONTEXT / integration effect (same as the P=128 wedge): cumulative
IOVA/domain/warm state or op-sequence across the 28-layer forward, NOT a single-shape kernel bug.
Kernel-level hypotheses (chain length, Sk) are both ruled out.

## Next step to localize (board-WEDGING — needs a reboot after)
Sync `src/npu.c` (has ORK_GRP_DEBUG + ORK_BCH_DEBUG) into the llama.cpp submodule, rebuild
`libggml-ork.so`, run the full int4 build with `ORK_GRP_DEBUG=1 ORK_BCH_DEBUG=1`, capture stderr:
the LAST `[grp]`/`[bch]` line before the reset storm = the wedging op IN CONTEXT (gives shape, but
since all shapes pass standalone the cause is the context — expect to then need IOVA/domain/warm
state inspection at that call). This localizes but likely won't immediately fix; it's an integration RE hunt.

## ACTUAL ROOT CAUSE of "packed 0" (2026-08-07, deterministic — localized)
Persist is wired PER COMPUTE-MODE in ggml-ork.cpp:
  - int8 `mul_mat_i8` -> `ork_persist_write` (call @1490)
  - hadamard W4A4 `mul_mat_i4_hadamard` -> `ork_persist_write_i4native` (call @2089)
  - GROUPED native-W4A4 `mul_mat_i4` (@1904) -> **NO persist call** => packed 0.
`ORK_QUANT=4` w/o `ORK_HADAMARD` dispatches to `mul_mat_i4` (grouped, dispatch @6157). It packs the
grouped weight into the in-memory `ctx->wcache` (ork_mm_pack_i4_grouped @1958) for COMPUTE but never
writes the orkpack. So the int4 build for the plain-W4A4 mode can't produce a pack — not a wedge, a gap.
(The instrumented full-model run @2026-08-07 COMPLETED the forward, no wedge, packed 0 — the wedge is
intermittent + separate.)

Applied so far: routed `ork_persist_write`'s int4 branch to the pure-CPU `ork_pack_i4a8_cpu_blob`
(removes the !ork_npu_busy skip + NPU dep) — CORRECT for the i4a8/mixed path, but that path isn't
called by `mul_mat_i4`, so it does NOT fix ORK_QUANT=4 grouped. Keep it (improvement, byte-identical).

## Fix plan for grouped-W4A4 persist (the remaining work)
1. Add a persist call in `mul_mat_i4` (write mode) mirroring @2089: dump the grouped `ow.w` (DT_I4 via
   ork_w_dump) + per-group `ow.bscale` (NG*N) as DT_I4_NATIVE. CHECK the read path populates `ctx->wcache`
   (keyed by the raw src0 ptr) from the orkpack — `mul_mat_i4` reads wcache, NOT the generic ow.w loader,
   so a naive persist would be written-but-unused. May need a wcache-preload from the pack at read.
2. Resolve read-back bscale-format match (grouped per-group NG*N vs i4native per-channel N).
3. THEN the intermittent grouped-prefill wedge (separate; kernels pass standalone — integration/context).
This is a scoped feature-completion + board validation, best as a focused follow-on.

## setdt=1 RESULT (2026-08-07): SAFE but INSUFFICIENT — wedge still open
Applied XSPEC[XP_SDP].setdt 0->1 (npu.c:4673). `make test` PASSES all precisions incl.
TEST_MODE_TRANSITION + CHAIN_XITION (byte-identical, no regression) — so the change is safe + a valid
correctness fix for the documented mm-after-SDP-skips-reset case. BUT the native-W4A4 full-model prefill
(`ORK_QUANT=4 ORK_MIXED_W4A4=1`, F16 gguf, P=64) STILL reset-storms (soft-reset count 15->64; no clean
prefill line; process exits, board recovers, no hard hang). => the SDP-transition was NOT the (whole)
cause. make test has no full-28-layer native-W4A4 prefill, so it can't catch this. setdt=1 is currently
UNCOMMITTED (on top of ork-driver 4b8cac7) — safe to keep or revert; does not achieve the wedge fix.

NEXT (deeper cause, fresh context): instrument the ACTUAL wedging submit. The reset storm = the kernel
timing out a hung grouped submit. With setdt=1 the post-SDP reset now fires, yet it still hangs — so the
hang is INSIDE the grouped submit under full-model conditions (IOVA/domain/wcache/DRAM-BW), not the entry
reset. Steps: ORK_GRP_DEBUG full-model prefill -> last [grp] before the storm; then compare that op's
buffers/domain/warm state vs the standalone pass (which is clean). Candidate: port the BCHAIN 6-reset
recover loop (run_i4_bchain_db@13813) into the grouped drain (ork_dyn_grouped_end@10885 has NO recover
loop) so a full-model doorbell-miss self-heals instead of storming.

## THE WEDGE ITSELF (still open) — nature + leading hypothesis + exact next steps
NATURE: the grouped submit HANGS in HW in full-model context → kernel `rknpu` soft-resets on the submit
timeout (`RKNPU: soft reset, num: 6`, repeated). INTERMITTENT (one instrumented full-model run completed).
Both kernels (BCHAIN + grouped) pass STANDALONE at every model shape → it's a CONTEXT-STATE trigger, not a
shape/kernel bug. Route B made this OPT-IN ONLY (ORK_MIXED_W4A4/ORK_HADAMARD); default int4 = W8A8, no wedge.

ROOT CAUSE (code-confirmed, one-row fix — HIGH CONFIDENCE):
  XP_I4CHAIN is RC_ALWAYS (XSPEC npu.c:4671) → entering the int4 chain ALWAYS resets, so a missing ENTRY
  reset is NOT it. The culprit is **XP_SDP has setdt=0** (XSPEC:4673): the SDP op (attn/ewmul/softmax) does
  NOT write c->last_dt. So the real-graph sequence grouped-i4 -> SDP -> grouped-i4 keeps last_dt==I4_CHAIN
  ACROSS the SDP op → the 2nd grouped op sees "no mode change" → SKIPS its reset → runs on the datapath SDP
  left → HW hang. Standalone never interleaves SDP, so it never triggers (explains standalone-passes / model-wedges).
  AGENTS.md explicitly names this fix: "flipping XP_SDP.setdt=1 to fix the fp16-mm->SDP->fp16-mm wedge" (Phase-2).
  int4-grouped is the same pattern.
FIX: XSPEC[XP_SDP].setdt 0 -> 1 (npu.c:4673, last field). Then post-SDP matmul re-transitions -> RC_ALWAYS resets.
VALIDATION REQUIRED before trusting (AGENTS mode-transition protocol — affects EVERY mm-after-SDP path, all precisions):
  - `make test` byte-identical at default + `ORK_SSM_KEEPWARM=0` + `ORK_MIXED_NOTHRASH=1`.
  - `make mode_probe && sudo ./mode_probe` — confirm no transition pair newly resets/wedges.
  - `ORK_DEBUG_RESET=1` reset-count diff (the 28->1 keep-warm behavior must stay intact; setdt=1 may add resets).
  - Repro gone: `ORK_MIXED_W4A4=1` full-model prefill no longer wedges.
  NOTE: unvalidated, NOT yet applied — could regress int8/fp16 mm<->SDP. Land only after the above passes.

EXACT NEXT STEPS (fresh session, per AGENTS mode-transition test protocol):
  1. Reproduce: `ORK_MIXED_W4A4=1 ORK_GRP_DEBUG=1` full-model prefill (ork_bench P>=32 on the F16 gguf), stderr
     to a board file. The last `[grp]` line before the reset storm = the wedging op IN CONTEXT (it will be a
     shape that PASSES standalone — confirming it's the context, not the shape).
  2. Confirm the mode-transition angle with `ORK_DEBUG_RESET=1` (log every ACT_RESET + caller) + `mode_probe`:
     check the SDP/softmax -> I4_CHAIN entry's reset behavior vs a pure grouped->grouped entry.
  3. If confirmed: flip the XSPEC row for XP_I4CHAIN's reset condition (RC_*) so the cross-mode entry resets.
     One-row edit in XSPEC[]. Re-validate: `make test` byte-identical (all knobs) + `mode_probe` + reset-count diff.
  4. If NOT a mode transition: instrument DRAM-BW/poll-timeout at the hang (grouped poll is 3e6us in
     ork_dyn_grouped_end@10890) — a full-model DRAM-BW-contention miss would need the BCHAIN-style 6-reset
     recover loop ported into the grouped drain (grouped currently has NO recover loop, unlike run_i4_bchain_db@13813).
  Each attempt can wedge -> `sudo reboot` (graceful, SSH alive). Board reboots getting slow (~>180s).

## Board ops
- Board 10.3.0.236, single-stream. A wedge = unkillable proc holding NPU + reset storm → `sudo reboot`
  (SSH stays alive; graceful reboot is safe — did NOT corrupt SPI). Reboots getting slow (~>180 s).
  Governors: `performance` on dmc + cpu4-7. Do NOT `kill -9` an in-flight submit.

## Tree state
- Local + board `~/llama.cpp` on `diverged-history`, ork-driver submodule 3c8e652 + UNCOMMITTED:
  auto-persist (validated int8) + int4 threshold-all-qbits + this WIP's debug hooks. Nothing committed yet.
