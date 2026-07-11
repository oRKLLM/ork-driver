# int8 batch-mode RE → stream/batch selector → int4 streaming — WIP (2026-07-08)

**Goal:** get int8 into BATCH mode (it currently STREAMS), diff stream-int8 vs batch-int8 regcmd to isolate
the **stream/batch mode-selector register(s)**, then apply the inverse to int4 to unlock int4 STREAMING
(→ escape the 16384 CBUF cap, ~16× at K=2048). Roadmap item ②→①.

## Static diff (synth_i8 streaming vs synth_i4 batch) — selector candidates
| reg | int8 STREAM (synth_i8) | int4 BATCH (synth_i4) | note |
|---|---|---|---|
| 0x405c (1001) | `(mc-1)<<16` (contiguous M-stride) | **0** (HW-default → stride-2) | ★ prime selector |
| 0x1040 (201) | mg K-schedule formula | base 177 (unset) | ★ schedule; poisons int4 when set w/ batch output |
| 0x1020/0x102c/0x1084 (201) | `mc` | `2*mc` (stride-2 encode) | output-row encoding |
| 0x4034 (1001), 0x3014 (801) | `mc-1` | `2*mc-1` | rows |
| 0x1010 (201) | `16*rows` (R=2cbuf/K) | `16*(2mc+1)` | CNA hint (neutral) |
| 0x107c (201) | `K/16` | `K/16` | SAME (entry-per-slice) |
| 0x1044 (201) | ceil(K/64) | ceil(K/128) | dtype density (not mode) |
| 0x1030/0x1034 | K*N / K | (K*N)/2 / K/2 | dtype density (not mode) |

**Hypothesis:** `0x405c` selects the OUTPUT mode (contiguous stream vs stride-2 batch), and `0x1040` selects
the K-reduction SCHEDULE. Stream needs BOTH the streaming schedule (`0x1040`=mg) AND contiguous output
(`0x405c=(mc-1)<<16`). int4 "poisoned" on `0x1040` earlier because we set the streaming schedule while
leaving the batch output (`0x405c=0`) — a mode mismatch, not a hard incompatibility.

## Plan
1. [in progress] int8 fuzz hooks (`ork_i8_fuzz_add/clear`) + `ork_npu_probe_i8_mm` (mirror the int4 probe;
   int32 output). Confirm: does `0x405c=0` flip int8 from contiguous (stream) to stride-2 (batch)? And does
   int8-batch then become CBUF-residency-limited (rows×K≤budget) like int4? → proves 0x405c is the selector.
2. Diff the working stream-int8 regcmd vs the confirmed batch-int8 regcmd word-by-word → full selector set.
3. Apply the inverse (stream config) to int4: `0x405c=(mc-1)<<16` + `0x1040`=mg + `mc` (not 2mc) +
   contiguous readback, at K=2048 → hunt rows×K > 16384 (streaming signature, contiguous output).

## Board-ops (10.3.0.236)
NPU single-stream; `sudo timeout -s INT`; DDR governor must be pinned `performance`@2112MHz (resets to
`dmc_ondemand` on some reboots — re-pin before benching). Board hard-wedged 4× today (HA "Rock 5B Plug"
power-cycle recovers). orkllm up on little cores. Build natively; scp each source to its explicit path.

## RESULT 1 (2026-07-08): int4 + partial int8-stream config → ≤1 row (preliminary: batch-only)
Applied a hand-assembled int8-stream config to int4 via ork_i4_fuzz_add (contiguous 0x405c=(M-1)<<16,
M-count=M not 2M, 0x4038 stream out-width, 0x1040 swept incl the int8 value 0x84) at K=2048 M=16 N=64.
Result: best = **1 contiguous row** (batch baseline 8, cap 8) — collapses to the pre-0x107c "only row 0"
regime. So int4 REQUIRES the batch trigger (0x405c=0 + stride-2) for multi-row; the streaming config gives
a single row. Preliminary → **int4 appears batch-only; the 16384/K CBUF cap (8 @ K=2048) is the real ceiling.**
CAVEAT: partial config — 0x1044 (K-passes), 0x1010 (hint), and the resident WEIGHT LAYOUT were left at
int4-batch values; a true stream pass may need a different weight layout. NOT fully conclusive.

## NEXT (rigorous): the int8-batch controlled A/B (original plan)
Get int8 (which DOES stream) into BATCH mode via the batch recipe (0x405c=0 etc. — ork_i8_fuzz_add hooks
already added), validate int8-batch bit-exact, then `diff` stream-int8 vs batch-int8 regcmd word-by-word to
get the COMPLETE, verified selector set (the regs + weight-layout implications my partial config guessed).
Apply the exact inverse to int4. Needs the int8-mm-raw probe (int8 tiling is the tiled ork_w Bb structure —
more work than the int4 probe). This removes the guesswork; my shortcut gave a strong-but-partial negative.

## RESULT 2 + CONCLUSION (2026-07-08): modes are DTYPE-LOCKED — int4 streaming not viable via config
int8-batch A/B (ork_npu_probe_i8_mm, K=512 M=8 N=64):
- STREAM baseline: rows 0..7 at physrows 0..7 (contiguous, correct) ✓ — int8 streams all 8 rows.
- BATCH (int4 recipe: 0x405c=0 + mc_phys=2M), WITH streaming schedule: all rows X (garbage).
- BATCH with schedule OFF (sched=0, no 0x1040): STILL all rows X (garbage).
=> int8 does NOT enter the int4-style batch mode via 0x405c=0/mc_phys, schedule or not.

Combined with RESULT 1 (int4 + int8-stream config -> 1 row): the batch/stream mechanisms are **dtype-locked**.
int4 = batch-only (CBUF-resident, 0x405c=0/stride-2); int8 = stream-only (0x1040 schedule, mg_max*64). Neither
crosses via these registers. So the "get int8 into batch, diff, transfer selector to int4" plan is REFUTED —
there is no reachable int8 batch mode to diff. **int4 streaming = not viable via register config** on this
silicon; int4's 8-row batch cap at K=2048 is the real ceiling.
CAVEATS (why "not-yet-viable" not "impossible"): int8-batch garbage may be a WEIGHT-LAYOUT mismatch (int8 batch
might need a different resident weight tiling than the stream tile_i8_range layout — untested); one shape only.
A breakthrough would need a dtype-specific batch/stream config (likely w/ matching weight layout) we haven't found.
