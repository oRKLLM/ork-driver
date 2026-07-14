# Mode-transition layer (`ork_npu_enter`) — landing record + Phase-2 catalogue

**Landed:** 2026-07-14 · **Branch:** `feat/mode-transition-layer` (off `origin/main` @ 93a8595).
Durable docs: AGENTS.md §"Mode-transition layer" and wiki *Exp-2026-07-14 Mode-Transition Layer*.

## What landed (Phase 1 — behavior-preserving)

The NPU precision-mode transition policy (ACT_RESET / re-warm `warmed`,`mwarm[]` / buffer-realloc
`ccsz`,`mccsz[]`) was copy-pasted, drifted, inline into ~16 run/stream/chain/int4 entrypoints. It is
now owned by ONE function `ork_npu_enter(c, to_marker, profile)` + the `XSPEC[]` policy table in
`src/npu.c`. Each profile is a **byte-for-byte transcription** of the site it replaced (validated
`make test` byte-identical, both keep-warm knob settings), so Phase-1 behavior is UNCHANGED. The drift
is now visible AS DATA and a policy change is a one-row edit.

**16 clean transition guards → 12 profiles:**

| profile | sites replaced | target marker |
|---|---|---|
| `XP_MC_MM` | `run_multicore` | F16/I8 (dynamic) |
| `XP_SC_MM` | `run` single-core | F16/I8 (dynamic) |
| `XP_CHAIN_NT` | `run_chain_i8_impl`, `ork_npu_chain_progs` | I8_CHAIN(3) |
| `XP_STREAM_I8` | `run_stream_i8` | I8_CHAIN(3) |
| `XP_STREAM_I8B` | stream int8 variant | I8_CHAIN(3) |
| `XP_STREAM_F16` | stream f16 (×2) | F16 |
| `XP_I4_MC` | `run_i4_mc`, `run_i4_grouped` | I4 |
| `XP_I4_MWARM` | `run_i4_incr_mc`, `run_i4_bchain`, `run_i4_cbatch` | I4 |
| `XP_I4_INCR` | `ork_mm_run_i4_incr` (caller-local `warm`) | I4 |
| `XP_I4CHAIN` | `ork_mm_run_chain_i4` | I4_CHAIN(4) |
| `XP_I4_STREAM` | stream i4 (caller-local `cold`) | I4_STREAM(5) |
| `XP_SDP` | *(defined, NOT wired — see #1 below)* | transient |

`ork_npu_enter` returns 1 iff a real transition fired, so the two sites gating a caller-local warmup
flag (`XP_I4_INCR`'s `warm`, `XP_I4_STREAM`'s `cold`) stay byte-identical.

## Guiding principle — discerning the CORRECT version (Phase 2)

The tests all passed *even with the drifted glue*, so correctness cannot tell us which version of a
transition is right. Tiebreakers, in order:
1. **If the candidates are otherwise byte-identical (same correctness), prefer the best-performing
   version** — bench it and let perf decide.
2. **Use the experiment logs + board-quirks data** (wiki Experiment Log / NPU-Quirks) to discern or
   hypothesize the correct version.
3. **When logs contradict, favor the most recent entry** — later findings are more likely accurate.
4. **When a contradiction shows a wiki entry is wrong, FIX or flag it** — edit the entry to be correct
   once the error is understood (or add a dated "this is wrong because…" note if not yet certain), so
   the wiki stays self-correcting and doesn't mislead the next agent.

## Phase-2 catalogue — sites/behaviors LEFT for 1-by-1 evaluation

Premise of the refactor: a transition that doesn't map cleanly is *probably* a bug — but some drift is
intentional, so each is decided individually in Phase 2 (a one-row `XSPEC` edit, re-validated by
`make test` + bench), using the guiding principle above. **Do not flip these blind.**

1. **SDP / activation ops not wired (the main event).** `XP_SDP` is defined + documented but no op calls
   it. The ops still do their own entry `ACT_RESET` and **leave `last_dt` untouched** → the
   fp16-mm→SDP→fp16-mm wedge (wiki Exp-2026-07-12). They are also mutually inconsistent:
   `ork_npu_ewmul_i8` does **no** reset (comment: not needed for the SDP element-wise op),
   `ork_npu_ewmul_f16` **does**. Phase-2 fix: route SDP-op entry resets through `ork_npu_enter(c,…,
   XP_SDP)`, flip `XP_SDP.setdt=1`, add `SDP→matmul` reset rows. Keep the `c->task` LUT-descriptor axis
   SEPARATE — `ACT_RESET` does not fix it (Exp-2026-07-12).
2. **Chain profiles ignore `ORK_SSM_KEEPWARM`.** `XP_CHAIN_NT` uses `KWP_NTL` (nothrash-only); a chain
   after an fp16 op resets where `XP_SC_MM`/stream (`KWP_F16`) keep warm. Decide whether chains should
   honor the (default-on) SSM keep-warm.
3. **`XP_STREAM_I8` vs `XP_STREAM_I8B` divergence.** Both target I8_CHAIN(3): `run_stream_i8` gates
   reset on `!I8_LIVE && !kw`, the other on `!kw`. Identical at default knobs, differ at
   `ORK_SSM_KEEPWARM=0`. Likely should be one profile.
4. **Diagnostic forced-fp16 resets skipped** (`ork_ssd_fused`, `ork_ssd_probe_rawmm/fusedmm_f16`):
   unconditional `act(); warmed=0; last_dt=DT_F16` with NO `if(dt!=last_dt)` guard, so a `setdt`
   profile's `from==to` early-return would drop the intentional forced reset → NOT cleanly replaceable.
   Board-only diagnostics; low priority (add a `force` profile if wired).
5. **Conditional prime resets skipped**: the `ORK_I16_RESET`-gated int16 reset and the fused-SiLU LUT
   `!ORK_I8_LIVE || ORK_PROBE_RESET` prime — env/warm-conditional and leave `last_dt`; model as a
   conditional SDP variant.
6. **int4 clear-gating asymmetries** (encoded faithfully, worth reconciling): `XP_I4_MC` clears `mccsz`
   on `!nothrash`; `XP_I4_MWARM` clears no size; `XP_I4_INCR` clears nothing (caller-local `warm`).

## How to work on this
See AGENTS.md §"Mode-transition layer" for the modify/add/test recipe. Scratch doc — fold into the
wiki and delete once Phase 2 is complete.
