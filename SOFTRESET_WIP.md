# WIP: intermittent fused-op NPU soft-reset (task #7)

## Goal
Eliminate the intermittent NPU soft-reset on the fused fast path WITHOUT re-adding a per-op
`RKNPU_ACT_RESET` (which costs ~107ms/op and drops pp128 below baseline).

## Code map (src/npu.c, verified 2026-07-05)
- `act(fd,RKNPU_ACT_RESET,0)` = the ~107ms reset. Design intent (line ~39): reset only when ENTERING
  int8 from fp16/int4/cold, NOT per matmul.
- `run()` int8 entry (3009) + single-core (3515): reset only if `dt==DT_I8 && !ORK_I8_LIVE(last_dt)`
  (i.e. transitioning into int8 from a non-int8 dtype). Within an all-int8 chain: NO reset fires.
- `ork_mm_run_i8_silu` (3670): fused gate. Entry (3678) clears warmed/ccsz, NO reset. Sequence per call:
  LUT-load submit (enable **0x18**, regcfg 1097) -> per N-slice/M-tile matmul+silu (enable **0x1d**) via submit1.
- `ork_mm_run_i8_out8` (3782): int8-out matmul, no reset.
- `ork_mm_run_i8_ewmul` (3966): fused matmul+ewmul, STILL has per-submit `act(RESET)` at 4002 (the slow/safe form).
- `submit1` (2349): 1 rep if warmed else 2; each submit `timeout=60000` (60s). Soft-reset = job timeout.
- HYBRID chain (ggml-ork) uses: `ork_mm_run_i8_silu` (gate) + `ork_mm_run_i8` (up/down) + CPU ewmul.
  Enable-mask toggles per layer: 0x18 -> 0x1d (silu) -> 0xd (up/down) -> back to 0x18 next layer.

## Hypothesis (initial)
Toggling the activation/LUT stage enable (0x1d fused-silu <-> 0xd plain matmul <-> 0x18 LUT-load) without a
state clear occasionally leaves the PPU activation module in a bad state -> next fused submit's int_status
never reaches 0x300 -> 60s job timeout -> kernel soft reset. Need a CHEAP targeted clear, not the 107ms reset.

## Status
- [x] REPRODUCE: 40-chunk hybrid chain → **29+ soft resets, 0 job-timeouts**. CONFIRMED reproduces.
- [x] Key facts:
  - Resets are **periodic ~1/0.95s**, run continuously (25+ in a row), NOT bursty-per-submit.
  - **ZERO job-timeouts** → NOT the kernel recovering a 60s hang. No IRQ/IOMMU error line precedes them;
    bare `RKNPU: soft reset, num: 6`. So a reset is being triggered directly, fast (~107ms, output stays
    correct — PPL is sane ~20).
  - **NOT orkllm**: node(712) has no /dev/dri fd → orkllm doesn't use the NPU. My chain is the only NPU user.
  - Rate ≈ **1 reset per LAYER** (28 layers/chunk, ~20s/chunk under this run → ~0.7s/layer ≈ the ~0.95s spacing).
- [ ] BISECT (next, needs NPU free): `ORK_FFN_SILU_CPU_GMAX=0` forces ALL gate silu to CPU (no fused 0x1d/0x18
  ops, only plain 0xd matmuls). If resets vanish → the fused-silu op (LUT-load 0x18 or fused 0x1d) is the trigger.
- [ ] Then: find where the reset originates (our code path has NO explicit act(RESET) for all-int8; kernel must
  reset on some condition tied to the 0x18/0x1d op) + cheap fix.

## BISECTION RESULT (2026-07-05, 4 cold chunks each)
| config | resets | PPL |
| :--- | :--- | :--- |
| chain_fused (ORK_FFN_CHAIN=1) | 29 | 16.5 |
| chain_cpusilu (SILU_CPU_GMAX=0, NO fused op) | 29 | **8.91** |
| nonchain baseline | 1 | 8.93 |
- **Fused silu is NOT the trigger** (cpusilu resets identically 29×).
- It's the **chain path itself**, ~once per layer FIRST-TOUCH (29≈28 layers across 4 chunks, NOT 112 → cold only).
- Baseline resets once (initial). All-CPU-silu chain = **baseline PPL 8.91** (confirms silu-int8 is the whole gap).
- dmesg = bare `soft reset, num: 6`, NO iommu/fault/timeout/irq line preceding.
- Chain vs baseline diff: chain uses SINGLE-CORE `ork_mm_run_i8`/`submit1` (core_mask=1<<tc) for FFN; baseline
  uses multi-core `run()`. So the chain introduces a per-layer single↔multi CORE-MASK toggle. Prime suspect.

## NEXT: instrument act() (ORK_DEBUG_RESET) — is OUR code calling RKNPU_ACT_RESET, or the kernel?
If our count == dmesg delta → our code (bisect the ~5 act(RESET) call sites). If << → kernel autonomous
(likely core-mask/idle-core reset on the single↔multi toggle).

## Revised hypothesis
The reset is ~1/layer and tied to the fused-silu op (LUT-load enable 0x18 and/or fused enable 0x1d), NOT a
per-submit hang. Either the kernel soft-resets around the enable-0x18 LUT-stream op, or our code triggers it
once per silu call. Bisection with SILU_CPU_GMAX=0 will confirm. NB: 0 job-timeouts means the ~107ms reset is
NOT from submit1's 60s timeout — it's fast, so it may be cheap enough that eliminating it is a modest win, OR
it's the ~107ms that accumulates (28/chunk × ~0.1s = ~2.8s/chunk overhead). Quantify vs a no-reset baseline.

## Board ops
Board 10.3.0.236. NPU single-stream. `sudo timeout` on all NPU cmds. Never kill -9. Wedge -> `ssh reboot`.
Build: `~/llama-ppu` (vendored ork-driver submodule at ggml/src/ggml-ork/ork-driver).

## Tree state
Local ork-driver feature/ffn-fusion @ 6c9fa35 (clean). Board submodule = same + uncommitted (my synced edits).
