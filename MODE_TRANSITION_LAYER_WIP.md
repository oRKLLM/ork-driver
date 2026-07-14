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
transition is right.

**The drift itself is the bug.** When two paths do the *same logical transition* differently, that
divergence is a defect to fix even if all versions are correct — converge to a single best-performing
version (single source of truth), whether or not the diverging path is currently exercised. Correctness
parity doesn't excuse drift; it just means perf + clarity decide the survivor. (First confirm they truly
are the same transition — item 1 was an apparent "drift" that turned out to be correct, distinct
behavior, not a divergent copy.)

Tiebreakers for picking the surviving version, in order:
1. **If the candidates are otherwise byte-identical (same correctness), prefer the best-performing
   version** — bench it and let perf decide.
2. **Use the experiment logs + board-quirks data** (wiki Experiment Log / NPU-Quirks) to discern or
   hypothesize the correct version.
3. **When logs contradict, favor the most recent entry** — later findings are more likely accurate.
4. **When a contradiction shows a wiki entry is wrong, FIX or flag it** — edit the entry to be correct
   once the error is understood (or add a dated "this is wrong because…" note if not yet certain), so
   the wiki stays self-correcting and doesn't mislead the next agent.

## Phase-2 progress (2026-07-14)

**Chaining mechanism is now explicit transition state.** `ork_npu_enter` takes a 4th arg
`chain` (`enum ork_chain_kind`: `OCK_NONE`/`OCK_SW`/`OCK_HW`/`OCK_FUSED`), recorded as `c->last_chain`,
so the policy can branch on the mechanism where it genuinely matters. Threaded from all 16 call sites
(streams→`OCK_SW`, `run_chain_i8`/`chain_progs`→`OCK_HW`, `run_chain_i8_impl` with a silu spec→`OCK_FUSED`,
per-matmul/int4-batch→`OCK_NONE`). **Behavior-preserving** (records state only; no `switch(chain)`
branch yet) — validated `make test` byte-identical. The hook is documented in `ork_npu_enter`.

**Real-pipeline mechanism bench (ork_bench, RK3588, my refactored ork-driver, governors perf):**
- qwen3-1.7b-q8_0 default: prefill **59.99** / decode **5.89** tok/s, coherent. XPROF: `MC_MM` only.
- qwen3-1.7b `ORK_FUSE=1`: 63.01 / 5.87, coherent, `MC_MM` only (fusion neutral, confirmed).
- qwen3-1.7b `ORK_FFN_CHAIN=1`: **broke** (`PRIME_FD` domain-alloc, decode 0), died before `CHAIN_NT`.
- `ORK_VERBOSE`: `ork_dispatch_i8` **never fired** → sw/hw chain grouping does not trigger.

**Finding: the sw/hw/fused chain profiles are DORMANT in a real dense pipeline.** The ggml graph
interleaves rope/norm between mul_mats so the chain-walk never reaches ≥2 consecutive independent int8
nodes; MoE experts run on CPU by default (`ORK_MOE_NPU` off); `ORK_FFN_CHAIN` fails on this model's
footprint. So `XP_CHAIN_NT`/`XP_STREAM_*` are exercised only by `mode_probe`/`test_ssd_chunk_npu` and
narrow configs (MoE-on-NPU, cross-domain, FFN-chain-compact). **The user's hunch is confirmed for the
fused static graph** — it chains differently (in-chain SDP/LUT, heavy footprint, fragile) and must not
be blindly converged with the pure-matmul hw chain; hence `OCK_FUSED` is distinguished. Items 2/3 below
therefore validate on **correctness** (mode_probe/test_ssd), not a real-pipeline speed bench that can't
trigger them; convergence of the sw/hw keep-warm predicates is the remaining open decision.

## Phase-2 catalogue — sites/behaviors LEFT for 1-by-1 evaluation

Premise of the refactor: a transition that doesn't map cleanly is *probably* a bug — but some drift is
intentional, so each is decided individually in Phase 2 (a one-row `XSPEC` edit, re-validated by
`make test` + bench), using the guiding principle above. **Do not flip these blind.**

1. **SDP / activation ops — RESOLVED 2026-07-14: NOT a bug, do NOT wire `XP_SDP`.** *(This item's
   Phase-1 framing was WRONG; corrected here per guiding-principle #4 — keep the record self-correcting.)*
   The Phase-1 catalogue mis-attributed the "SDP→matmul wedge" to SDP leaving `last_dt` untouched.
   Wrong: the wedge was the **`c->task` LUT-descriptor poisoning** — a *separate* axis (nuance #1),
   already fixed in **98c00b1** (LUT ops save/restore `c->task`). `Exp-2026-07-12` states outright "the
   wedge has nothing to do with `last_dt`"; `test_mode_transition` regresses it and passes.
   Board **`mode_probe` (2026-07-14, full matrix below)** confirms **every** SDP→matmul pair is **SAFE
   with `fix=none`** — including the once-failing `ewmul_f16→MM_F16` — and that `fix=none` is *faster*:
   SDP→int8-matmul is ~0.3ms with `none` vs **~105ms** when `invalidate`/`reset` forces a re-warm. So
   flipping `XP_SDP.setdt=1` would re-introduce the exact ~105ms/transition churn `ORK_SSM_KEEPWARM`
   removes, for **zero** correctness gain. **Decision (principle #1: correctness ties → best perf):**
   leave SDP ops as-is (correctly leave `last_dt` untouched); keep `XP_SDP` reserved/unused. The
   `ewmul_i8`(no reset) vs `ewmul_f16`(reset) difference is op-local self-correctness, not a precision
   transition — deferred, low priority.

   **mode_probe SDP→matmul matrix (RK3588, 2026-07-14; `B(rc,ms)` = victim matmul):**
   ```
   SDP op        -> MM_F16                         -> MM_I8
                    none     inval    reset           none      inval     reset
   EXP_I16          0.2 ok   0.3 ok   0.3 ok          0.3 ok    105 ok    107 ok
   SILU_I16         0.2 ok   0.3 ok   0.3 ok          0.4 ok    105 ok    107 ok
   EWMUL_I16        0.2 ok   0.3 ok   0.3 ok          0.5 ok    107 ok    107 ok
   EWMUL_F16        0.2 ok   0.3 ok   0.3 ok          0.4 ok    107 ok    107 ok
   ADD_F16          0.2 ok   0.3 ok   0.3 ok          0.5 ok    107 ok    107 ok
   ```
   All SAFE; no wedges. `none` (keep-warm) is optimal for every pair.
2. **Chain profiles honor `ORK_SSM_KEEPWARM` — FIXED 2026-07-14 (★ ~105ms/transition win).**
   `XP_CHAIN_NT` used `KWP_NTL` + `RC_NOTLIVE`, so a chain entered from an fp16 op ate a **full ACT_RESET
   soft-reset** where the stream profiles kept warm. `chain_xition_probe` (manufactured fp16→int8-chain
   across a wide K/N/M sweep, since the path is dormant in a dense pipeline) measured the cost: **chain(hw)
   after-fp16 reset-cost ≈ 53,538 µs avg (individual ~104–107 ms), stream(sw) ≈ 0 — and BOTH coherent**,
   proving the reset was pure drift, not correctness. Converged `XP_CHAIN_NT` → `KWP_MC` + `RC_NOTLIVE_NOTKW`
   (at `to=3`, `KWP_MC` = keep-warm if `f16warm&KW(from)` or `nothrash&INT(from)` — the unified predicate).
   Re-ran the probe: chain(hw) reset-cost **53,538 → −1.1 µs**, all coherent. `make test` still passes.
3. **`XP_STREAM_I8`/`XP_STREAM_I8B` unified — FIXED 2026-07-14.** Both targeted I8_CHAIN(3) with only
   non-default-knob reset/warm-cond differences (gratuitous drift). Converged both to `KWP_MC` +
   `RC_NOTLIVE_NOTKW` + `WC_NOTLIVE_NOTKW`, which made them identical, then **collapsed to one profile**
   (`XP_STREAM_I8`); `run_stream_i8_sk` repointed to it and `XP_STREAM_I8B` removed. Single source of truth.
4. **Diagnostic forced-fp16 resets — ASSESSED 2026-07-14: intentional, out of scope (leave).**
   (`ork_ssd_fused`, `ork_ssd_probe_rawmm/fusedmm_f16`) do an unconditional `act(); warmed=0;
   last_dt=DT_F16` with NO guard — a deliberate *forced reinit* in board-only diagnostic/bench code, not
   a precision-mode *transition*. A `setdt` profile's `from==to` early-return would (correctly) drop the
   forced reset, so they are not the same construct. Not production drift; left as-is.
5. **Conditional prime resets — ASSESSED 2026-07-14: correct as-is (leave).** The `ORK_I16_RESET`-gated
   reset and the fused-SiLU LUT `!ORK_I8_LIVE || ORK_PROBE_RESET` prime are env/warm-gated and leave
   `last_dt` untouched — which is exactly the correct SDP-axis behavior confirmed by the Phase-2 mode_probe
   finding (SDP ops leaving `last_dt` alone is optimal, not a bug). Not drift.
6. **int4 clear-gating — ASSESSED 2026-07-14: plausibly mechanism-legit, deferred with rationale.**
   `XP_I4_MC` (run_i4_mc/grouped, standard multicore output), `XP_I4_MWARM` (incr_mc/bchain/cbatch,
   batch-chain paths that size their own grow-only per-call buffers), and `XP_I4_INCR` (single-core, static
   local buffers) use *different buffer-management strategies*, so the size-clear difference is not obviously
   the *same* transition (guiding-principle caveat: confirm same-transition before converging). Converging
   would need an int4-specific manufactured probe to prove equivalence; low value (int4 is the experimental
   tier, W8A8 is production) and non-zero risk → deferred, not a Phase-2 blocker.

## How to work on this
See AGENTS.md §"Mode-transition layer" for the modify/add/test recipe. Scratch doc — fold into the
wiki and delete once Phase 2 is complete.
