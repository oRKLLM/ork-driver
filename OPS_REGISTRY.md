# ork-driver — NPU Op Capability Registry

**READ THIS BEFORE reverse-engineering, re-fixing, or building on ANY NPU op.**
The point of this file is to stop re-derivation: every op/chain/handler that has been
tried is listed here with its **status** and the **probe that proves it**. If you think an
op is broken or missing, check here first — it may already be solved, or already known-dead.

## Chain-transition matrix (op→op) — validated by mode_probe campaign 2026-07-21

`ork_chain_table` (ork_ops.c, single-sourced from `ORK_CHAIN_LIST` in ork_npu.h) records how each ordered
op→op transition may combine: `HW` (one PC-chain), `SW` (safe as separate back-to-back submits), `DISALLOW`
(default for unvalidated pairs — run independently). Census: **HW=6, SW=50, DISALLOW=785** of 841.
- **SW** from the `mode_probe A B 0` (fix=none) campaigns: all 49 pairs among {matmul_f16, matmul_i8,
  exp_i16, silu_i16, mul_i16, mul_f16, add_f16} SAFE; plus `matmul_f16→{gelu_i16, add_i16}` SAFE.
- **HW=6** from the chain probes (chain_gu_silu_probe FFN inner + mm_perchan_f16_probe).
- **`matmul → int8-SDP` = DISALLOW (validated HARD-WEDGE, campaign 2 2026-07-21)**: `MM_F16→SILU_I8` hung,
  `MM_F16→GELU_I8` **hard-wedged the NPU** (needed a power-cycle), `MM_I8→SILU_I8` hung (D-state). It's a
  fp16/int8-matmul → int8-SDP mode-switch hazard — whereas `matmul → int16-SDP` is all-SAFE (so the int16
  SDP path is the safe one; prefer it over int8-SDP in chains). **Cost 3 power-cycles to establish; do NOT
  re-run the int8-SDP transition campaign.** The int8-SDP ops are broadly wedge-prone as post-matmul
  separate-submit victims — left DISALLOW-by-default.
- **SW ≠ HW**: separate-submit safety ≠ PC-chain safety. `matmul_i8→mul_i16` is SW (only its *HW* 2-input SDP
  chain hangs). Conversely `matmul_i8→silu_i8` is **HW** (the FFN gate→silu PC-chain, chain_gu_silu_probe)
  even though its *separate-submit* (SW) path hangs in mode_probe — the HW capability is what the table records.
- Pending/uncampaigned: A=int16/fp16/int8-SDP as contaminator (rows 2–14) were not reached; perchan/softmax/
  reshape/rope have no mode_probe harness. Those transitions stay DISALLOW.

**DISALLOW is a driver-config verdict, not a hardware one.** A `fix=none` wedge means the transition is
unsafe with the transition template we CURRENTLY apply (`ork_npu_enter`/`XSPEC` profile + regcmd) — the NPU
may well support it with the right config. Strong evidence for `matmul→int8-SDP`: (a) it is **HW-safe in a
PC-chain** (the FFN gate→silu, `chain_gu_silu_probe`), so the NPU can do it; (b) the int8-SDP ops never got
the transition tuning the int16-SDP ops did — `XP_SDP` was validated "SDP→matmul safe, no reset" *for the
int16/fp16 SDP ops only*. So the separate-submit `matmul→int8-SDP` wedge is a **missing/wrong int8-SDP
transition template**, not a HW limit. **Fix path** (do NOT brute-force fix=none again — it hard-wedges):
run `mode_probe A B 2` (fix=RESET) on a wedged pair; if RESET makes it safe, add that reset/profile to
`ork_npu_enter` for `matmul→int8-SDP` and UPGRADE the cell `DISALLOW→SW`. Treat every DISALLOW as a candidate
driver fix, not a dead end.

## How to read / maintain this registry (probe-verdict anchored)

- **Status is anchored to a probe — enforced at build time.** Every `PROVEN` / `PARTIAL` /
  `DEAD` row must name a probe/test (or explicitly say `(no ... probe)`), and every probe/op it
  cites must exist. `make check-registry` (run by `make all`; `tools/check_registry.sh`, no NPU)
  **fails the build** otherwise — so "a status with no probe" is a compile error, not a red flag
  you have to spot while reading.
- **Status only changes when a probe is re-run.** Don't upgrade/downgrade a row from a
  code reading or a hunch — run the probe, then edit the row.
- **On touching an op:** if you change an op's behavior, re-run its probe and update the
  row in the SAME commit (see `AGENTS.md` §"Op capability registry").
- The `*_WIP.md` docs are **backstory/detail**. This file is the **index of truth**. When
  a WIP doc and this registry disagree, a probe re-run settles it — see CONTRADICTIONS.

### Status enum

| status | meaning |
|---|---|
| **PROVEN** | bit-exact / coherence validated on-silicon by a named probe. Safe to build on. |
| **PARTIAL** | runs without wedging but is NOT fully correct (ratio given), or works only in a narrow regime. |
| **DEAD** | wedges / hangs / abandoned. **Do not use.** The "use instead" column points to the live path. |
| **WIP** | in progress or walk-proof only; not yet a dependable building block. |

---

## matmul output stages (`set_*` regcmd rewriters, `src/npu.c`)

| name | purpose | status | probe → verdict | gotchas / layout | use / notes |
|---|---|---|---|---|---|
| `set_i8_out8` (847) | int32 acc → int8 requant (banker's round) | **PROVEN** | `i8out_map_probe` → writes **int8-LINEAR** (m·N+n) | identity = mult `0x4000`/shift `14`. **Only valid K≥512** (small-K schedule miscomputes requant). | basis for `set_i8_silu`, `run_i8_out8` |
| `set_i16_out` (861) | int32 acc → **int16** compact output | **PROVEN (standalone), 2026-07-21** | `i16out_fix_probe` → 128/128; `chain_i16silu_probe` → "BRIDGE WORKS — feeds the int16 SiLU coherently" | Output is **LINEAR (m·N+n), NOT EWCUBEH**. Regs: `0x4010=0x20000000` (int16 precision), `0x4038=(s/8-1)|(N/8-1)`, `0x4050=0x36e`, `0x40c0=0x40`. Old `0x4010=0` + N/16 typo = the WDMA-terminal/precision mismatch that stalled it. Env `ORK_I16OUT_*`. | supersedes the stale "set_i16_out wedges" note (Contradiction #1) |
| `set_f16_out` (891) | shim: fp16 OUT_CVT on int8 matmul | **DEAD / wedge-prone** | (no passing probe) — "writes `0x4084=1` WITHOUT FP32TOFP16_EN, fp16 CVT never enabled → hangs fp16 matmul" | — | **use `set_f16_out_fp16in`** |
| `set_f16_out_fp16in` (920) | fp16-in fp16-out stage from vendor capture | **PROVEN 512/512** | `mm_perchan_f16_probe` (composes it via `ork_npu_probe_f16_mm_f16out`) → "CONTIGUOUS fp16 G (512/512)" | default CONTIGUOUS; `ORK_F16_ATOM8` → EWCUBEH atom-8 cube for chaining. `0x4084=0x00010001`, `0x4040=0x53`. | used by `ork_npu_chain_mm_perchan_f16` |
| `set_i8_silu` (996) | SiLU fused into matmul output (PWL LUT, enable 0x1d) | **PROVEN (curve)** | `fused_silu_test` → "VALIDATED bit-exact" | `R` sets BOTH acc→index step AND LUT→out gain. **But as an FFN quality path it's inaccurate (PPL 55 at all gmax)** — see ggml `ORK_FFN_SILU_I16` instead. | curve proven; quality-dead |
| `set_i8_silu32` (5219) | as above, int32 (un-requantized) output | WIP | (no dedicated pass probe) | int32 out | — |
| `ork_ppu_fuse_enabled` (969) | runtime gate for all fused-output paths | **PROVEN gate** | returns 1 only on rk3588, CPU/NEON fallback otherwise (no probe — trivial runtime gate) | no env var | — |

## chain assemblers (PC-chain / doorbell)

| name | purpose | status | probe → verdict | gotchas | use / notes |
|---|---|---|---|---|---|
| `ork_dyn_begin_seq_i8` / `_mc` (10430/10332) | **the working chain assembler** — heterogeneous [matmul+SDP…] as ONE self-healing NONBLOCK doorbell | **PROVEN** | `sdp_chain_probe` → "int8 SDP rides the doorbell (Stage 4). PASS" | TERMINAL op MUST be a matmul (int32 sentinel gates completion); int8-SDP-terminal needs a **B2 witness matmul**; ping-pong OFF; **has a `SILU_I16` kind** (single-core + LUT prologue). | **preferred production chain path** |
| `ork_submit_seq` (12899) | sequence scheduler: batches HW-eligible ops into doorbell submits, SW-breaks rest | **PROVEN** | `test_submit_seq` → heterogeneous seq bit-exact (i8↔f16↔i4) | `seq_hw_ok`: int8/fp16 K%512 & K≤4096, M≤64, Sn==1; int4 M==1, Sk≤16. `SILU_F16` row = NULL (dead, see #4). Routes to `ORKD_SEQ` if daemon up. | — |
| attention seq subset: `REDUCEMAX_I8` / `MUL_PERCHANNEL_F16`+`_I8` / `RSQRT_I16` / `ROPE_NEOX_F16` / `RMSNORM_F16` seq adapters (task #20) | wire the softmax/RMSNorm/rope primitives into `ork_submit_seq` (SW-break SDP dispatch) | **PROVEN** | `softmax_seq_probe` → each adapter dispatches vs CPU (reducemax bit-exact, mul_perchan 2.4e-4, rsqrt_i16 137 LSB RKNN-class, rope 8.8e-4) + end-to-end on-NPU softmax `row_max→exp→Σ-reduce` COHERENT (max\|err\| 1.8e-4) | `RSQRT_I16` appended to enum (i8 pre-existed). `ROPE_NEOX_F16`/`RMSNORM_F16` seq ops = SW-break only (chain transitions DISALLOW, no mode_probe); rope pos via `o->B`+freq_base via `o->in_scale`; rmsnorm gain via `o->B`+eps via `o->in_scale`. Also route through **orkd `ORKD_SEQ` Path B** (npu.c sizing switch extended; daemon reconstructs generically → its own Path A) — `orkd_seq_probe` batches all 6 SDP ops in ONE round-trip, bit-identical to direct | — |
| full attention layer as an `ork_submit_seq` pipeline (task #20) | assemble RMSNorm→QKV→rope→QK^T→softmax→A·V→O→residual as a 13-op seq | **PROVEN (coherent, direct mode)** | `attn_chain_probe` → full single-head layer COHERENT vs CPU (max\|err\| 6.2e-4, mae 9.6e-5, 13 ops rc=0); `attn_chain_probe` core-only (QK^T→softmax→A·V) also validated | Direct-mode coherence only. Intermediates flow through HOST buffers; each dtype bridge (int8-quant, x−max, int16→f16, 1/Σ) + runtime K^T/V densify+pack is a CPU step between ops that blocks batching. orkd `ORKD_SEQ` Path B now carries the attention SDP ops (validated, `orkd_seq_probe`). **Full chain ALSO validated through orkd** (`attn_chain_probe` under `ORK_USE_ORKD=1`): all 7 fp16 matmuls become daemon-resident automatically via `ork_mm_pack`→`orkd_pack_f16` (static Wq/Wk/Wv/Wo/ones AND runtime-packed K^T/V), the 6 SDP ops route Path B → 13-op chain COHERENT, bit-identical to direct (max\|err\| 6.2e-4). Stable across 30+ runs; ONE early transient (~82% off + one op rc≠0) did not reproduce in 28 retries. Matmuls here are SW-stream (K non-conforming, not %512) so they do NOT ride the doorbell's cold-output clean — the matching documented fragility is the int16-LUT lazy calibration wedging as first-op-after-multicore-matmul. HARDENED: orkd startup `orkd_warmup` runs a tiny `exp_i16` (clean int16-LUT calibration, before any matmul) + fp16 matmul, so serving is never cold (logs `warmup rc=0`; disable `ORKD_NO_WARMUP`). NOTE: each op is still its own round-trip (CPU dtype bridges block batching); one-round-trip resident chain needs on-NPU bridges (f32→f16, x−max, requant) | — |
| A2 on-device intermediate residency (ORKD_SEQ v3, task #20) | back-reference: a batched op's A/B aliasing a prior op's C ⇒ consume that resident output on-device (skip re-upload, `a_src`/`b_src`) + keep it resident, not shipped (`c_keep`) | **PROVEN** | `orkd_resident_probe` → batched `rmsnorm→Q/K/V` with `xn` aliased across the 3 matmuls: COHERENT via orkd (max\|err\| 6.2e-4, xn uploaded 0×/never returned). Non-aliasing seqs byte-identical (regression: `orkd_seq_probe`, `attn_chain_probe` still pass) | Zero public-API change — detected by pointer-aliasing in `ork_submit_seq` Path B (same aliasing the direct path already needs); proto bumped v2→v3 (`a_src`/`b_src`/`c_keep` in `struct orkd_seq_op`, default 0 = pre-v3). Reach limited to already-adjacent (bridge-free) op pairs until A1 closes the dtype bridges | — |
| A1+A2 payoff: resident `rmsnorm→QKV→rope` front (task #20) | 6-op contiguous chain — RMSNorm → Q/K/V via `MM_F16_F16OUT` (f16 out) → rope(Q),rope(K); `xn`/`Q`/`K` aliased resident | **PROVEN** | `qkvrope_probe` → COHERENT (max\|err\| 1.977e-3) direct AND via orkd, bit-identical. All fp16, every transition bridge-free (rmsnorm f16→matmul; matmul **f16-out**→rope f16-in — the A1 bridge). Under orkd only x/gain/pos in + V/Qr/Kr out = **ONE round-trip vs 6** per-op | fp16 needs an abs error floor (~4e-3) over a multi-op chain, not pure relative. The attention layer front is now a single resident submit | — |
| A1+A2: resident attention layer BACK `O proj → residual` (task #20) | `O = attn_out · Wo` (`MM_F16_F16OUT`, f16 out) → `y = x + O` (`add_f16`), `O` aliased resident | **PROVEN** | `oresidual_probe` → COHERENT (max\|err\| 8.4e-4) direct AND orkd, bit-identical. Same bridge-free f16-out-matmul→add recipe as the front; ONE round-trip vs 2. | Attention layer FRONT (rmsnorm→QKV→rope) and BACK (O→residual) are now BOTH single resident submits; only the MID softmax island (mixed-dtype, multi-session) remains for a fully-resident layer | — |
| A1 fp16-output matmul: `ORK_OP_MM_F16_F16OUT` / `ork_mm_run_f16_f16out` (6005) | matmul emits CONTIGUOUS fp16 (not f32) via the proven `set_f16_out_fp16in` vendor stage, consuming a packed resident `ork_w` (`w->Bb`) | **PROVEN** | `f16out_probe` → matmul-alone bit-exact vs f16-narrowed CPU; `MM_F16_F16OUT→mul_perchan` with intermediate aliased = bit-exact direct AND via orkd (A1+A2 together, no f32→f16 bridge) | Single-tile (M≤64, single-slice), single-core, own submit ⇒ SEQ_CLASS hw=0 (SW dispatch, not the f32-out doorbell). Kills the pervasive matmul→SDP f32→f16 bridge; e.g. `MM_F16_F16OUT→rope`/`→mul_perchan` become adjacent+resident. | — |
| A1 exp→Σ bridge: symmetric int8 reduce (path B) | `exp_i8`(int8 out)→`MM_I8`(e·ones_i8)→int32 Σ instead of `exp_i16`→[int16→f16 cast]→`MM_F16` | **PROVEN** | `int8reduce_probe` → Σ within 0.9% of CPU exp-sum (max rel-err 9e-3, 0/64 rows>3%), bit-identical direct AND orkd (e_i8 A2-resident). out_scale cancels in the normalize so int8 is fine for Σ | **KEY HW FACT: RK3588 is symmetric-precision (README:105 "all mixed types reject at create") — a single int16-in/fp16-out op is a DEAD END (verified: my hand-built `0x4010=0x44000002`+FP32TOFP16 cast got ETIMEDOUT/rejected, self-healed). Bridges must stay same-precision; int8 reduce is the symmetric answer. **x−max broadcast-sub (last A1 bridge) — INVESTIGATED, not landed:** needs a per-channel ADD (bias=−max) in the transposed layout, but the ERDMA/EW_ALU path is MUL-only — only the `EW_MUL` stage broadcasts operand-b per-channel (proves `mul_perchan`); the `EW_ALU` add stage does not (swept 0x5034 modes: 0x08 best but 69% wrong, 0x40000008 99% wrong, 0x80/0xc0000008 reject). The MUL/ADD f16 templates differ ONLY at `0x4070` (EW_MUL vs EW_ALU=SUM). **CLOSED via matmul-broadcast + add** (proven ops, zero register RE): `-max_bc = MM_F16_F16OUT(max_padded[M,32] · -ones[32,N])` (K=32 outer product, col0=max ⇒ every element = -max[m], fp16 out) then `add_f16(scores, -max_bc)` = scores-max. `xmax_probe`: COHERENT direct (max\|err\| 1.95e-3) AND via orkd as one aliased 2-op seq (A2-resident: -max_bc fed forward on-device). NOTE: the earlier "add_f16 garbages" was a PROBE bug (`(ork_f16)f2h(x)` bit-cast into native `_Float16` → huge/inf operands overflowed + stalled add's inf-poison poll) — `add_f16` and the transition are fine. The BS_ALU/ERDMA per-channel-add path was abandoned (unproven + wedge-prone); the matmul-broadcast is strictly better | — |
| softmax exp — HW-CHAINED via fused output-stage (M4.8, `ork_mm_run_f16_act` fn=exp) | `C = exp(Q·Kᵀ)` — exp rides the score matmul's DPU output stage in ONE submit (no separate exp op, no matmul→SDP crossing) | **PROVEN** | `fused_exp_probe` → COHERENT (max\|err\| 1.45e-4) one submit, scores≤0 (post-max, single-signed fp16 index). **Chosen over the transposed int8/int16 softmax: HW-chained (1 submit vs many) + exp free + comparable accuracy.** | **M4.8 is a general technique** — HW-chains any `matmul→pointwise-LUT` (exp/silu/gelu/rsqrt): softmax exp on QK^T, FFN silu on gate-proj, RMSNorm rsqrt on the reduce. Does NOT apply to non-pointwise transitions (rope, residual-add, per-channel scale — those stay A1 f16-out + A2-resident, or the output-stage per-channel path). Full resident softmax = fused-exp(QK^T) + Σ-reduce-matmul + normalize; still needs the max-subtract (global-max scalar bias, or the on-NPU x-max matmul-broadcast). Follow-up: packed-`ork_w` fused-exp for seq-composition | — |
| (a) full fused-exp softmax island + (b) fused rsqrt/silu (task #20) | softmax = fused-exp(QK^T) + on-NPU Σ-reduce(e·ones) + normalize; shared M4.8 fused-act also does RMSNorm rsqrt + FFN silu | **PROVEN** | `fused_softmax_probe` → softmax COHERENT vs CPU (max\|err\| 8e-3, mae 1.5e-3, exp HW-chained one submit + on-NPU reduce). `fused_act_probe` → rsqrt COHERENT (1.8e-4), silu within fp16 bound. | The softmax problem is solved via the fp16 fused-exp path (no int8, no f16→int8 quantize — that was the wrong path). Full one-round-trip residency needs the packed-`ork_w` fused-exp as a seq op + on-NPU max-subtract (the x-max matmul-broadcast) — all pieces proven, remaining work is composition | — |
| pack-once / run-many resident fused activation (`ork_mm_pack_f16_fused_act` + `ork_mm_run_f16_fused_act`, npu.c 5492+) | split of `ork_mm_run_f16_act`: pack bakes the calibrated PWL LUT + out-scale INTO the weight (`w->fa_lut`/`w->fa_osc`); run replays `C=fn(A·B)` in one submit with NO re-pack. `run_f16_act` is now the thin one-shot wrapper (pack+run+free) over the same path. | **PROVEN** | `fused_resident_probe` → exp packed ONCE, run TWICE on different Q against the SAME resident weight, both COHERENT (max\|err\| 1.6e-4, 0/2048). | The enabler for composing fused exp/rsqrt/silu inside a RESIDENT seq — the per-call LUT-rebuild+re-pack in `run_f16_act` would defeat residency. Run path carries no `fn` pointer (baked into the weight) so it crosses a seq/socket. Direct-mode proven. **The fused-exp→reduce handoff to make it a resident softmax seq is the open piece — see next row.** | — |
| f16-OUT on the fused-LUT path (fused exp → fp16 out for a bridge-free reduce handoff) | override the DPU output CVT (0x4010=0x48000002 fp16→fp16 + 0x4084 FP32TOFP16_EN) while keeping the LUT stage active, so `C=fn(A·B)` writes CONTIGUOUS fp16 the Σ-reduce consumes resident (out-scale cancels in P=e/Σ) | **DEAD (HW-rejected)** | tried `fused_f16out_probe` (reverted): RKNPU_SUBMIT ETIMEDOUT (errno 110) + self-healing reset; bailed at S1 before the reduce, board recovered clean | **STRUCTURAL, not a wrong constant — do NOT re-sweep.** The proven fp16-out stage (`set_f16_out_fp16in`) needs BS(0x4040=0x53)/BN(0x4060=0x53)/EW(0x4070=0x383) FULLY BYPASSED, but the LUT rides those exact stages (`set_f16_silu` sets 0x4060/0x4070 active) → fp16-out CVT + active-LUT-stage = proc-precision mismatch (same dead class as the int16→f16 cast, README:105 symmetric-precision). The fused-LUT output stage is fp32-out ONLY. **Resident fused-exp softmax needs a different handoff** (options: on-device f32→f16 narrow op between exp and reduce; or use the ALREADY-PROVEN int8 exp→Σ symmetric reduce — `int8reduce_probe` — as the resident softmax island instead of the fused-fp16 exp). | — |
| `MATMUL_I16OUT_I8` seq op (`ork_mm_run_i8_out16`, int8 matmul→int16 out via `set_i16_out`) | int32-acc → int16 COMPACT-LINEAR output, fed to an int16 SDP op resident (A2) instead of hardware PC-chaining | **PROVEN (direct)** | `i16out_seq_probe` → S1 int16 matmul bit-exact (0 LSB); S2 `MM_I8_OUT16→exp_i16` resident == S0 standalone `exp_i16` (63 LSB, identical) ⇒ **A2 resident-forwarding sidesteps the `set_i16_out` PC-chain issue** (LINEAR int16 out IS the contiguous host layout the int16 SDP reads — no cube bridge, no chain fragility). | K≥512 (int8 requant; K=128 rejects → NOT usable for softmax head_dim). orkd Path B returns -2 (daemon-resident weight lacks `w->Bf` that `run_i8_out16` needs — orkd-weight follow-up). Generalizes the specialized `gatesilu` hardware chain to any int8-mm→int16-SDP via the seq | — |
| K-padding sidesteps the small-K int8-requant wall (task #20 high-effort path) | zero-pad the contraction dim (head_dim 128 → 512): the 384 padded rows are `0·0=0` terms that don't change the sum, so a K-padded int8 QK^T is EXACT and rides the PROVEN K=512 requant path — no schedule RE, no wedge risk | **PROVEN** | `kpad_qkt_probe` → int8 QK^T at d=128 (padded to 512) vs CPU over d=128: max\|err\| 1 LSB (requant rounding), 0/4096 off | Reframes crack (c) from "risky schedule RE" to a zero-RE sidestep. Cost: 4× contraction MACs (cheap for int8/short context). **Unlocks the all-int8 resident softmax FRONT: int8 QK^T → int8 scores at real head_dim.** Remaining for a fully-resident int8 softmax: bake a GLOBAL-max scalar bias into `exp_i8` (per-row max needs the dead per-channel-add; global max is a scalar → bakeable, and correct+stable since the constant cancels in P=e/Σ) → `exp_i8`→`MM_I8` reduce (PROVEN, `int8reduce_probe`) → deferred normalize (fold 1/Σ into A·V). Open: scalar-max-bias into exp_i8; int8 score precision for softmax quality | — |
| scalar global-max-biased `exp_i8` (`ork_npu_exp_i8_biased`, `act_lut_i8_biased`/`silu_build_curve_biased`) | bake a scalar max-subtract into the exp LUT: `out = clamp_i8(exp((x-max)*in_scale)/out_scale)`, max = the scalar GLOBAL max | **PROVEN** | `exp_biased_probe` → softmax `P=e/Σ` COHERENT vs CPU (max\|err\| 1.9e-3, 0/2048); plain `exp_i8` saturates 989/2048 (48%) on the same scores (why the subtract is needed); biased exp 3.2 LSB (int8-LUT class) | The numerically-stable softmax numerator WITHOUT a per-row op (per-row max needs the dead per-channel-add; a scalar global max ≥ every row max so every arg ≤0 → exp∈(0,1], no int8 overflow — and the constant cancels in `P=e/Σ`). For a single-submit resident chain the bias must be a STATIC calibrated bound (runtime gmax = a 2nd pass), tight-calibrated to avoid int8 underflow. Direct only (Path B TODO). Pairs with [[K-padded QK^T]] + `exp_i8`→`MM_I8` reduce (`int8reduce_probe`) → the resident int8 softmax pieces are ALL proven; remaining = assembly into one seq + on-NPU A·V connection | — |
| resident int8 softmax seq assembly `[MATMUL_REQUANT_I8 → EXP_I8 → MM_I8]` (task #20) | compose the three proven pieces (K-pad int8 QK^T → scalar-max-biased exp → int8 Σ-reduce) into ONE `ork_submit_seq`, scores/e A2-resident | **PARTIAL — wedge solved via orkd; a reduce correctness bug remains** | `softmax_i8_seq_probe`. DIRECT: scores bit-exact (1 LSB) but wedges (int8-LUT-after-matmul trap → hang → `sudo reboot`). ORKD (N=64): **rc=0, NO wedge** — orkd's hardened path runs exp-after-matmul without hanging — but reduced Σ is garbage (direct Σ≈1, orkd Σ≈2e8 ⇒ the terminal MM_I8 reduce isn't writing its output under the resident chain). | **(B) orkd routing: the wedge is SOLVED** — `MATMUL_REQUANT_I8` routes through orkd (`reqi8_probe` ORKD K=512 → 1 LSB; the old "-3 TODO" was STALE, daemon-resident weight has `Bf` now), and the orkd-served seq does NOT hit the direct-mode exp-after-matmul wedge. **ROOT-CAUSE ISOLATED (daemon instrumentation, `ORKD_SEQ_DEBUG` in handle_seq + `softmax_i8_split_probe`):** the softmax logic is SOUND — the bug is in the **orkd SW-`run()` path for `MM_I8`**. Daemon seqdbg proved: Stage A forwards scores correctly (op1 Asum=-73 = op0 Csum=-73) and produces CORRECT `e` (Csum(i8)=252479, avg~62); the reduce RECEIVES the correct `e` (Asum(i8)=252479) but outputs GARBAGE Σ (uninitialized-looking int32, ss[0]=-67108106). Cross-referencing configs against `seq_hw_ok` (M≤64 && K%512==0 && Sn==1 → doorbell, else SW `run()`): **doorbell path works through orkd** (`int8reduce_probe`, M=64/K=512), **SW-`run()` path is BROKEN through orkd** — Stage B (M=64/K=64, K%512≠0→SW) garbage, N=512 chain (M=512>64→SW) garbage. **Direct SW-`run()` is CORRECT at all K** (`minimal_i8mm_probe`, plain MM_I8 bit-exact K=32..1024) — so it's specifically the SW path ON THE DAEMON. Same weight (`handle_pack`→`ork_mm_pack_i8`, non-orkd local weight) and same `run()` code as direct, so the defect is in the daemon's environment/dispatch for the SW branch (cores=3, warmup state, or transport of the SW-path output), not the math. **FIX DIRECTIONS:** (a) force int8 seq matmuls onto the proven doorbell path — M≤64 tiling + K-pad to %512 (the user's auto-pad idea + M-tiling); ONLY covers M≤64 (prefill M>64 still SW-path). (b) fix the daemon SW-`run()` path (the general fix). **(b) FURTHER NARROWED (fresh-board, deterministic seqdbg to /tmp/orkd-seqdbg.log via `ORKD_SEQ_DEBUG`):** the reduce `run()` returns rc=0 but `c->Cc` retains STALE data — its seqdbg `C[i32:0..3]` is byte-identical to the PRIOR op's `scores` buffer (malloc-reuse leftover), so `run()`'s `memcpy(C,c->cres,need)` copied a `c->cres` that was never overwritten by the reduce's matmul submit. So the reduce's NPU submit didn't compute/write `c->Cc` in the daemon's resident-chain context. **NOT multi-core** — `ORK_NPU_MC=1` (single-core) still garbage. Prime suspect: matmul-after-SDP-LUT mode transition (the reduce follows `exp_i8` in the chain), but the standalone-reduce control (`minimal_i8mm_probe` through orkd, no preceding exp) could not be captured — it wedges the daemon into **D-state (uninterruptible, unkillable, holds the NPU)**, forcing a reboot. **META-BLOCKER = orkd observability/lifecycle over ssh:** empty output capture (daemonized orkd holds ssh stdout), D-state daemons that need a reboot, and `sudo reboot` stuck (degraded systemd → use a kernel `sysrq` reboot: `echo 1>/proc/sys/kernel/sysrq; echo b>/proc/sysrq-trigger`). **Next session MUST first make orkd debug iteration reliable** (probe output→board file; deterministic seqdbg log — DONE; a way to run/kill the daemon cleanly), THEN submit-level trace the matmul-after-SDP reduce. LESSON: never leave orkd lingering between runs (stale-daemon confound). | — |
| `MATMUL_REQUANT_I8` seq adapter (int8 matmul→int8 out) | wire `ork_mm_run_i8_out8` into `ork_submit_seq` (mult/shift via `o->mult`/`o->shift`) | **PROVEN (direct + ORKD, K≥512)** | `reqi8_probe` → max\|err\| 1 LSB at K=512 direct AND ORKD (K=64 FAILS: requant needs K≥512, registry `set_i8_out8`). ORKD Path B routes (daemon-resident weight has `Bf`; the earlier "-3 TODO" was stale). | For int8 K≥512 matmul→int8 chains (FFN-style); NOT usable for the small-K (K=32) x-max broadcast (that stays f16). Confirms an all-int8 softmax island is NOT clean — small-K broadcasts want f16, exp wants int8/int16, int8-requant wants K≥512 → the full resident island is inherently mixed-dtype, a multi-session integration | — |
| `ork_mm_run_chain_i8_ffn` (9009) | GENERAL FFN chain: per-task OP_MM32/MM8/SILU/EWMUL, SDP reads prior by index | **PROVEN** | `chain_gu_silu_probe` → "FULL FFN INNER CHAIN WORKS (gate,silu,up,glu,down, ONE submit)" | ops[].in0/in1 aliased; MM8 when feeding SDP; single M-tile/task. Reads matmul output **LINEAR** in-chain. | the coalesced FFN inner |
| `ork_mm_run_chain_i8_gsilu` (8980) | chain with FUSED int8 SiLU on gate task | **PROVEN** | `chain_gu_silu_probe` → "REAL-WIDTH FFN INNER … ONE submit, EXACT" | built on run_chain_i8 + set_i8_silu | quality note: int8 fused silu inaccurate |
| `ork_mm_run_chain_i8` / `run_chain_i8_impl` (8961/9020) | batch S>1 independent int8 matmuls, one submit | **PROVEN** | `moe_expert_probe` → chain correctness PASS | per-task M ≤ chain cap; routes to orkd | — |
| `ork_npu_chain_gatesilu_i16` (8217) | data-connected int8-mm(int16-out) → int16-silu bridge | **PROVEN (small-shape)** | `chain_i16silu_probe` → "BRIDGE WORKS" | K%32,N%32, N≤nmax, M≤64, N&7; G(matmul int16-out) is silu's EWCUBEH input. | uses `set_i16_out` |
| `ork_npu_chain_mm_perchan_f16` (8424) | fp16 mm → fp16-out → vendor per-channel SDP (attn A·V) | **PROVEN** | `mm_perchan_f16_probe` → "ALL OK — bit-exact, on-NPU" | needs 2-pass cold re-warm (`OCK_HW`) else zeros; `set_f16_out_fp16in` + `REGCMD_MUL_F16_CHAIN` | the working fp16 chain |
| `ork_dyn_begin_mc` / `_colsplit` (9972/9835) | int8 multi-core doorbell (M-tile/K-split/N-colsplit auto) | **PROVEN** | `ork_dyn_ntile_test` → "OK (bit-exact)" M=1 & M>1 | bit-exact ceiling `mg_max*64`; ~1/4000 dispatch-drop auto-recovered | — |
| `ork_dyn_begin_mc_i4` / `_grouped` (9648/9738) | int4 W4A4 M=1 doorbell (int16 out) / float-grouped int4 | **PROVEN** | `i4_doorbell_probe` → "DOORBELL PASS — bit-exact"; grouped gtest maxerr 0.0000 | int16-sentinel clean-before; Sn>1 & Sk≤16 (grouped Sk≤256) | — |
| `ork_npu_chain_mm_perchan_i16` (8325) | int8-mm(int16-out) → int16 **2-input per-channel** SDP | **DEAD — HANGS (errno 110)** | `chain_mm_perchan_probe` → 176/512, hangs; (2026-07-14) "int16 2-input SDP `REGCMD_MUL_I16` is NOT chain-safe — keeps BS-ALU active `0x4040=0x20050`" | Dtype-match necessary but NOT sufficient; only fp16 has a chain-safe 2-input SDP. | **use fp16 (`chain_mm_perchan_f16`)**. NOT the int16 SiLU path (that one works — don't conflate). |
| `ork_npu_chain_mm_silu_i16` (8051) | Phase-0 walk-proof (data NOT connected) | WIP (walk-proof) | proves the chain WALKS; buffers separate | — | superseded by `gatesilu_i16` |
| `ork_npu_chain_progs` (h:573) | general PC-chain core (address-aliasing DAG) | **DEAD / superseded** | `CHAIN_ASSEMBLER_WIP.md`: "abandoned … never worked end-to-end. Do not build on it" (no passing probe) | kept only for 8 RE probes | **use `ork_dyn_begin_seq_i8` / `ork_submit_seq`** |

## SDP ops (activation & elementwise, enable 0x18)

| name | purpose | status | probe → verdict | gotchas |
|---|---|---|---|---|
| `ork_npu_silu_i16` (+gelu/rsqrt/exp_i16) (8493) | int16 activation via LUT | **PROVEN (accurate)** | HW-chained (2c2f9b9), "max\|err\|=75 RKNN-class"; ggml note "~325× more accurate than int8 fused (0.28 vs 92 @gmax132)" | atom-8 EWCUBEH (N%8), dslot 138; lazy idx calibration wedges if FIRST op after multi-core matmul → `silu_calibrate_idx16` |
| `ork_npu_silu_i8`/`gelu_i8`/`rsqrt_i8`/`exp_i8` (7995+) | int8 activation via LUT | **PROVEN** | `test_activations` PASS, "max\|err\|=2" | atom-16 ORK_SEQCUBE (N%16) |
| `ork_mm_run_i8_silu` (5140) | gate matmul + fused SiLU | **PROVEN bit-exact** | `fused_silu_test` → "VALIDATED"; `fused_mtile_check` OK | per-TENSOR quant; rk3588-gated |
| `ork_npu_ewmul_i8`/`_f16` (6331/6375) | 2-input elementwise multiply | **PROVEN** | `ewmul_probe` PASS | `_i16` variant EXPERIMENTAL (int16 not bit-exact over signed range) |
| `ork_npu_add_i8`/`_f16` (7174/7190) | 2-input add | **PROVEN** | domain-propagation fix 2026-07-20 (no dedicated add probe; via `make test`) | `_i16` (7257) EXPERIMENTAL — "int16 add NOT bit-exact over signed range" |
| `ork_npu_mul_perchan_f16`/`_i8` (6684/7085) | per-channel scale | **PROVEN** | `bs_scale_probe` → "per-channel scale (i8+fp16) WORKS at general geometry" | fp16 contig vs atom-8 cube variants |
| `ork_npu_mm_perchan_f16`/`_fused`/`_diag` (6586/6033/6619) | fp16 mm → per-channel scale | **PROVEN 512/512** | `mm_perchan_f16_probe`/`_diag` → "ALL OK — bit-exact" | composes f16_mm_f16out + mul_perchan_f16 |
| `ork_npu_replay_softmax_f16` (7490) | replay vendor 9-task softmax verbatim | **PROVEN (replay)** | "replays bit-exact, task_number=9, rc=0"; `softmax_reduce_probe` → "exp→reduce WORKS" | fixed capture geometry (N=64, 256 rows) |
| `ork_npu_row_max_i8` (6535) | row-max reduction | **PROVEN** | `max_reduce_probe` OK | — |
| `ork_mm_run_i8_ewmul` (5529) | up-matmul × G fused in SDP output | **DEAD — WEDGES** | `fused_silu_test` → "fused EW-mul WEDGES (DPU_RDMA graft wall)"; opt-in `ORK_TEST_EWMUL` | **use standalone `ork_npu_ewmul_i8`** |
| `ork_mm_run_f16_silu` (5360) | fp16 gate matmul + fused fp16 SiLU | **DEAD — not viable** | "garbage PPL 9072 vs 9.15 int8; gated OFF" | **use `ork_npu_silu_i16` or CPU silu** |
| `ork_npu_requant_perchan_i32` (7026) | standalone SDP int32→int16 per-channel requant | **PARTIAL — loopback not viable** | (2026-07-15) "SDP clamps to 2-byte read → errno 110"; over-fetch fix rc=0 "BUT processes as TWO int16 lanes, not true int32" | int16-range accumulators usable (even-channel low half); full int32 needs lane recombine (open) |
| `ork_npu_replay_reshape_f16` / `rope_neox_f16` (6856/6424) | reshape/permute, NEOX RoPE fp16 | WIP | `reshape_probe_f16`, `rope_probe` | RE-stage |

## ggml FFN handlers (env-selectable, `ggml-ork.cpp`)

| flag | purpose | status | probe / evidence | notes |
|---|---|---|---|---|
| (default) | int8 gate matmul + **all-CPU fp32 SiLU** | **PROVEN / shipped** | "all-CPU-silu is the shipped default"; PPL-validated | `fc.silu_cpu = !(GATE_F16\|\|F16)` |
| `ORK_FFN_SILU_I16` (328) | int8 gate matmul + on-NPU int16 SiLU | **PROVEN** | "~325× more accurate than int8 fused LUT (0.28 vs 92 @gmax132)"; PPL A/B | the accurate integer silu path; gate `ORK_FFN_SILU_I16_GMAX` |
| `ORK_FFN_F16` (316) | ALL-fp16 FFN inner (no int8 activation quant) | **PROVEN** | coherent (~18 PPL); gate `ORK_FFN_F16_GMAX` | 2× IOVA → ~5 layers/4GiB; N-chunked (24MB single tile fails under IOVA frag) |
| `ORK_FFN_F16_JIT` (333) | host int8+bscale inflated into shared fp16 scratch (W8A16) | **PROVEN** | `jit_inflate_check` → "PASS (== direct fp16 pack, bit-exact)" | decouples fp16 layer count from IOVA cap |
| `ORK_FFN_F16_CPUSILU` (340) | plain fp16 gate matmul + EXACT CPU silu | **PROVEN** | "plain matmul + fp32 CPU silu is exact"; PPL-validated | needs `ORK_FFN_F16` or `ORK_FFN_GATE_F16` |
| `ORK_FFN_SILU_CPU_GMAX` (324) | high-gmax layers: per-channel gate + CPU silu | **PROVEN** | "baseline quality, ~2.3× fused-silu gate cost" | per-layer policy |
| `ORK_GU_CHAIN` (281) | HW-chain up+gate into ONE run_chain_i8 submit/M-tile | **PROVEN** | "falls back to per-op on any failure, output always valid" (no unit probe; runtime fallback-guarded) | K%512 & ≤4096, Nff single-slice |
| `ORK_FFN_CHAIN` (292) | per-tensor SwiGLU chain, int8 intermediates on-NPU | **PARTIAL / WIP** | "**disabled under orkd (fd-local ops)**. Open work (#20) = route through orkd `ORKD_SEQ`" | needs SmoothQuant per-layer; M<32 skip |
| `ORK_FFN_GATE_F16` (310) | fp16 gate + fused fp16 SiLU | **PARTIAL** | fused fp16 silu not viable unless `_CPUSILU` set (no probe) | — |
| `ORK_GATE_ABLATE` (4250) | diagnostic: isolate gate's 3 error sources | diagnostic | — | unset = normal |

## orkd daemon RPC surface

| name | purpose | status | evidence | notes |
|---|---|---|---|---|
| `ORKD_SEQ` → `handle_seq` | route heterogeneous op sequence through daemon | **PROVEN** | "seq-grp-silu16/silu8 DIRECT + ROUTED … make test ALL PASS, 0 wedges" | one round-trip batches S ops |
| `orkd_run_chain_i8` | fused int8 chain on daemon | **PROVEN** | `orkd_ffn_probe` → routed chain bit-exact; "fused chain through orkd (f97db3f)" | single-domain (tasks[0].w→domain) |
| `ORKD_DOM_REQ` / per-client IOMMU domains | per-client domain on packs/work | **PROVEN (corrected 2026-07-20)** | `domain_correct` PASSES 2-domain int8; `orkd_dom_api` PASS | earlier "kernel broke multi-domain" was a self-induced runtime wedge, NOT real |
| shm ring `orkd_run_i8_ring` | remove per-op socket round-trip | **PROVEN** | `orkd_ring_probe` → "bit-exact + ~1.9× at M=1"; SYNC ring is the real ~1.6× win | async gives ~0 single-client gain (daemon serial) |
| multi-consumer (N clients) | concurrent clients on one daemon | **PROVEN to N=6** | "≥3-client deadlock FIXED (53e46f7), board-proven N=6"; `orkd_2proc` PASS 2&3 | supersedes stale `test_orkd_2conn_seq` "N≥3 hangs" note (#2) |

---

## CONTRADICTIONS / STALE NOTES (resolved by probe verdicts)

1. **`set_i16_out` "abandoned/wedges" (CHAIN_ASSEMBLER_WIP.md) vs "SOLVED, works" (code).**
   RESOLVED: the MD's blanket note is **STALE** for `set_i16_out` itself — it was fixed
   later the same day (`6fb8b9f`; old wedge = `0x4010=0`/N-16 typo precision mismatch).
   `chain_i16silu_probe` → "BRIDGE WORKS." The MD's *other* point — the `chain_progs`
   PC-chain route is superseded by the NONBLOCK doorbell seq — still stands.

2. **N≥3 clients "HANGS" (`test_orkd_2conn_seq` comment, 2026-07-19) vs "FIXED to N=6" (ORKD_WIP).**
   RESOLVED: the probe comment **predates** the `53e46f7` poll/revents deadlock fix. STALE.
   Re-run `test_orkd_2conn_seq 4+` to update the comment.

3. **int16 chain-safety: matmul-int16-out "PROVEN" vs `chain_mm_perchan_i16` "HANGS".**
   NOT a contradiction — two different ops. The matmul **int16 output** is proven; the
   **int16 2-input per-channel SDP** (`REGCMD_MUL_I16`) is the one that hangs (BS-ALU stays
   active). The int16 **single-input SiLU** (`ork_npu_silu_i16`) is PROVEN and chain-safe.
   Don't conflate the three — this trap cost a debugging detour.

4. **fp16 SiLU: partially-live env flags vs "not viable" verdict.**
   The `ork_mm_run_f16_silu` / `SILU_F16` seq target / `ORK_FFN_GATE_F16` fused path exist
   in the API but fp16 fused SiLU is NOT VIABLE (garbage PPL). `SEQ_CLASS[]`'s
   `ORK_OP_SILU_F16` row is a NULL/TODO dead entry. **Do not wire fp16 fused SiLU** — use
   int16 SiLU or CPU silu.

5. **gmax gating scaffolding is inert.** The persisted gmax sidecar/profile has "No active
   consumer yet (shipped policy is all-CPU-silu)." It's foundation for a future selective
   int16-silu path (blocked, #35). Don't assume gmax policy affects the shipped path.

---
*Extracted 2026-07-21 from `src/npu.c`, `tools/*.c`, `ggml-ork.cpp`, and root `*_WIP.md`.
Line numbers drift — grep the symbol. To change a status, re-run the named probe first.*
