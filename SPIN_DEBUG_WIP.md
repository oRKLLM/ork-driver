# Spin-tail debug WIP (persistent chain + linger)

## Goal
Persistent spin tail for ork_dyn chains: reserved slots forward-chained so the NPU stays busy through the
reserve budget (bounded, no self-loop/redirect), haltable early by linger (ork_dyn_queue_idle / ork_dyn_halt),
safe teardown. Env-gated ORK_DYN_SPIN + ORK_DYN_RESERVE. All in src/npu.c, branch streamline-arch.

## State of tree (uncommitted on streamline-arch, on top of committed ba77747 + a878169)
- src/npu.c: spin tail + spin_end field + spin_end-aware ork_dyn_halt/ork_dyn_queue_idle + safe teardown in
  ork_dyn_end (bulk-terminate spin before free).
- Makefile: ork_dyn_spin_test target. tools/ork_dyn_spin_test.c: validator (S=8, small reserve).
- Committed (safe, make-test-clean): ba77747 SRAM static table, a878169 linger idle-exit halt.

## Bugs found + fixes (chronological)
1. WARM/REAL OVERLAP (fixed): ork_dyn_begin warm-submits then real-submits the same chain; with the spin
   applied BEFORE warm, the warm job kept spinning into the real submit (single-stream collision). FIX: apply
   the spin forward-chain AFTER the warm pass (warm = terminating chain = proven protocol), then real submit.
   -> This was a PHANTOM lead: the "hang" it was chasing was actually bug #2, which crashes BEFORE begin.
2. TEST-HARNESS SIGBUS (fixed): the test did `memset(O,0,...)` on the ork_dma_alloc OUTPUT buffer. Those
   buffers are device/non-cacheable; a bulk memset's unaligned SIMD stores SIGBUS on ARM (exit 135). This
   crashed the test in SETUP, before the spin code ran at all — which masqueraded as a "hang" through the
   pipe+timeout. FIX: removed the memsets (NPU writes outputs; begin seeds the doorbell with aligned words).
   KEY LESSON: never CPU-memset an ork_dma_alloc buffer. Also: earlier hang-diagnosis was chasing this crash.
3. SPIN FORWARD-CHAIN ABORTS THE JOB (OPEN, but LOCALIZED): reserve=32 WITH spin -> 0/8, highest=-1.
   BISECTED: reserve=32 NO spin -> 8/8 OK; baseline -> 8/8 OK. So reserve>P is fine; the bug is SPECIFICALLY
   the spin forward-chain. Signature (WHOLE job yields nothing, not just the tail) => re-running the last
   matmul in-chain (slots 8..31 = copies of program S-1, SAME A/B/C addresses, forward-chained) ABORTS the
   job so no outputs commit. ROOT CAUSE (hypothesis, high-confidence): re-executing an identical matmul
   (same output address) as chained tasks is not valid / aborts.
   FIX DIRECTION: dedicated no-op spin program (reuse resident weight B, zeroed throwaway spinA, throwaway
   spinC distinct from doorbell outputs), forward-chained. IMPLEMENTED in ork_dyn_begin spin block.

4. DEDICATED NO-OP SPIN — ABORT FIXED, RESIDUAL ESTABLISHMENT RACE (OPEN, known-hard class): with the no-op
   spin, the job NO LONGER ABORTS — real programs execute (case 1 [halt, cold->warm]: end highest=7). BUT
   outputs are flaky: case 1 (with warm pass) = 5/8; case 2 (warm SKIPPED, c->warmed already 1) = 0/8,
   highest=-1; and MORE spin => MORE corruption. Board healthy, no wedge.
   DIAGNOSIS: this is the SAME async warm/coherency-establishment race the wiki documents for the fp16 doorbell
   (Exp-2026-07-16 §7b: ~85%, "never bit-exact despite ~10 establishment/priming/structure hypotheses").
   The persistent-spin async real submit has a cold/warm establishment + output-coherency race, worsened by
   the concurrent spin. NOT a simple bug — it's the hard async-doorbell establishment class.
   NEXT (future session, substantial): fp16-doorbell-grade establishment protocol + per-op spin doorbell +
   rigorous output invalidate/settle sequencing. OR accept the async persistent-spin shares fp16's fundamental
   limitation and rely on the PROVEN bounded chunk+resubmit wrap (self-terminating chains, which ARE reliable —
   ork_pc/ork_dyn_begin non-spin = bit-exact). The reliable win is chunk+resubmit; persistent-spin-feed is the
   hard async frontier.

## SESSION OUTCOME
Built persistent spin tail; fixed 3 bugs (warm/real overlap; test SIGBUS-on-dma_alloc-memset; spin-abort from
re-running the real matmul). Reached "spin runs, real programs execute" but a residual async-establishment race
(5/8 warm, 0/8 no-warm) remains — the known-hard fp16-doorbell class. Board hard-wedged 3x this session (all
recovered via plug cycle, SPI intact); STOPPED board work. Spin code is env-gated OFF (ORK_DYN_SPIN),
board-safe, UNCOMMITTED. Committed: ba77747 (SRAM), a878169 (linger).

## Diagnosis so far / verified
- NPU + ork_dyn_begin baseline are HEALTHY: ork_pc_bench (uses ork_dyn_begin, reserve=P) = 1.95x, outputs ok.
- My npu.c edits do NOT break the baseline (spin gated by ORK_DYN_SPIN + reserve>P; ork_pc_bench unaffected).
- The 0/8 is specific to the spin/reserve>P real submit. Hypotheses to test NEXT (fresh session, healthy board):
  a. WARM job with task_number=reserve(32) but terminating at program 7 may NOT complete (kernel expects 32
     tasks, only 8 ran) -> NPU left busy -> real submit blocked/no-op. Test: does warm(reserve=32, terminating)
     complete? Compare highest after a plain reserve>P NON-spin begin+end (no ORK_DYN_SPIN) — if that is also
     0/8, the bug is the reserve>P warm/completion path, NOT the spin forward-chain.
  b. end() teardown (bulk-terminate 8..31) may race/abort the real submit before 0..7 compute.
  c. The forward-chained real submit's descriptor may be malformed (re-running program 7 in-chain).
- Instrument: per-op doorbell logging (ORK_DYN_DEBUG shows submit rc; add per-op landing times), dmesg errno
  110 / job abort correlation, try reserve=P+1 (minimal) NON-spin vs spin to bisect a vs c.

## State of tree
- Spin code is env-gated OFF (ORK_DYN_SPIN unset) -> default path unchanged, make-test-clean. UNCOMMITTED.
- Committed & solid: ba77747 (SRAM static table), a878169 (linger idle-exit halt). Do NOT lose the spin WIP
  (AGENTS.md: never discard experimental code without consulting).

## Board-ops
- 10.3.0.236, NPU SRAM 956KiB. Rebooted several times this session (SPI survived). Currently healthy.
- Hard wedge -> HA plug "Rock 5B Plug" off/on (~20s boot). Soft reset in dmesg != wedge. NPU wedge -> sudo reboot.
- timeout every NPU cmd; SIGTERM not -9 (a -9/SIGTERM mid-submit leaves in-flight job -> degrades NPU across runs).
- ~18 board runs this session; 2 hard wedges recovered. STOP if pushing further degrades it.

## Board-ops
- 10.3.0.236, NPU SRAM 956KiB. Hard wedge -> HA plug "Rock 5B Plug" off/on (~20s boot, SPI survived x2 today).
- Soft reset in dmesg != wedge (often deliberate ACT_RESET). Wedge = no ping/ssh.
- timeout every NPU cmd; SIGTERM not -9; force-recompile after rsync (touch src/npu.c).
