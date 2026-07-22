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
