# Tiered int4-park / JIT-int8 streaming residence — WIP program tracker

**Branch:** `feat/tiered-i4-residence` (off `feat/npu-doorbell`).
**Goal:** run a >IOVA model (the 35B-A3B) on the NPU by parking weights int4 (compact/fit) and materializing
int8 JUST-IN-TIME at the domain swap via the free CPU unpack — CPU-int4 / NPU-int8 per-engine optimum.

## Design (validated in pieces this session)
- **NPU→int8, CPU→int4** (measured): NPU int4-unpack is a HW-feed bottleneck (~10 GB/s < int8 28.5); CPU int4
  register-unpack is FREE (cpu_i4_vs_i8: 1.94x, both DRAM-bound ~24 GB/s).
- **int4 = universal store** (half footprint → fits) + CPU-compute form; **int8 = NPU-compute form**, produced
  JIT at the domain boundary by the free CPU unpack.
- **JIT-at-domain-swap** (NOT once-at-load): keeps the model int4-parked (compact); only the ACTIVE domain
  (≤4GiB working set) holds int8, materialized per swap, reclaimed after. That's what makes the 35B fit.

## Slices (ordered)
1. **[THIS] JIT int4→int8 materialization primitive** — validate pack_i4_to_i8 (int4→int8 ork_w) produces
   correct int8 for the NPU + the inflate is cheap (free CPU unpack). Probe `jit_i4_i8_probe`.
2. **In-place JIT inflate into a reused domain buffer** — the streaming form (no per-swap ork_w re-alloc);
   inflate int4-parked → int8 into a pre-allocated active-domain DMA buffer.
3. **Tier-map** — hot/sensitive → int8-resident (where it fits); cold/big tail → int4-parked + JIT-swap
   (ORK_ORKPACK_TIERMAP + the multi-domain ORK_DOMAINS streaming).
4. **q4-GGUF → int4 orkpack** for the 35B (dequant q4 → int4 store; conversion-box job) — the model's own
   Q4/Q6/Q8 mixed precision maps to the tier.
5. **ggml-ork wiring + submodule bump** — the decode/prefill path uses int4-park + JIT-int8-swap + CPU-int4.
6. **35B end-to-end** on the board (residence + streaming + monitor).

## Board/ops
- Streaming/multi-domain is the wedge-prone area (IOVA imported-weight faults, multi-domain-runtime non-det).
  Guard: single-domain first, errno=110 detect, timeout-drain, SIGTERM not kill-9.
- 3 board recoveries this session already — validate each slice in isolation before the 21GB 35B streaming.

## State
- [x] Branch + WIP.
- [x] Slice 1 (jit_i4_i8_probe): JIT int4->int8 CORRECT (run == int8, bit-exact) but naive pack_i4_to_i8 inflate = 274ms (the TILE layout, not the unpack). => Slice 2 = in-place inflate into a pre-tiled reused domain buffer (park int4 tiled; per-swap nibble-expand -> int8 in place; target ~free-unpack ~1-2ms/68MB).
- [ ] Slice 2: in-place JIT inflate (pre-tiled int4 -> int8 into reused domain buffer).

## ★ REFRAME (2026-07-15) — CPU-int4-primary collapses the streaming program
Two facts converged: (a) `pack_i4_to_i8` is just `pack_i8` (no cheap inflate — the 274ms is a full int8 pack),
(b) CPU decodes int4 at 2x int8 and BEATS the NPU on the big weight (1.4 vs 2.4ms) → CPU-int4 is the PRIMARY
decode engine, NPU-int8 is a HELPER on a small share. Therefore only a PORTION (the NPU's IOVA-fitting slice)
needs int8 — upconverted ONCE, resident, NOT streamed. So:
- DROP Slice 2 (in-place streaming inflate) + Slice 3 (streaming tier-map) — no streaming; the bulk stays
  int4-in-RAM for the CPU, the NPU slice is upconverted once.
- The 35B is ALREADY Q4_K_XL → no q4→int4 conversion needed for the CPU bulk (it's llama.cpp's Q4 CPU path,
  already 3.2 t/s). DROP Slice 4.
- REMAINING = the aggregate co-work (already built/measured): CPU-Q4 bulk ‖ NPU-int8 fixed resident slice
  (pack_i8 of a portion that fits IOVA). This is the expert-split / MoE-offload mechanism, ~1.1x measured
  (CPU-Q4 already fast → NPU is a modest parallel helper, not a multiplier).
- So "run the 35B on the NPU" = keep it Q4-on-CPU (primary) + a fixed NPU-int8 slice in parallel. No streaming,
  no conversion, no JIT-per-swap. The lever is bounded (~1.1x); the big streaming build was unnecessary.

## ★★ BENCH FINDING (2026-07-15) — PATH-B on 35B: NPU 7×/expert at PREFILL, IOVA is the wall
Ran ork_bench 35B-A3B-Q4_K_XL, ORK_MOE_NPU=1 PATHB=1 PATHB_PARK=1 FRAC=0.15 (spec config: CPU=int4/Q4_K
native ‖ NPU=int8 minimal IOVA slice, concurrent). Prefill PATH-B profile (57 calls):
  NPU-experts=954 CPU-experts=5619 | npu=119ms cpu=5064ms | => NPU 0.125 ms/expert vs CPU 0.90 ms/expert
  => at PREFILL (batched M) the NPU is ~7× FASTER PER EXPERT. NPU wildly UNDERLOADED at FRAC=0.15
     (119ms of a 5064ms window). Balancing npu_t≈cpu_t => ~88% of experts to NPU => ~7× prefill MoE win.
  This is NOT 1.1× — the concurrent split + user's "NPU-heavy on prefill" is a genuine multi-× prefill lever.
- REGIME (confirmed): PATH-B only engages at `batched` (ORK_MOE_BATCH_MINM, M_e>=8) => decode (M=1) already
  falls to CPU. So "prefill NPU-heavy / decode CPU-heavy" is partly free; need a tunable per-regime FRAC.
- ★ BINDING CONSTRAINT = IOVA (as the user called out). One domain = 3900 MiB fits only ~954 experts (~14.5%),
  and even that STARVES the dense (attn/shared) weights: with default HOT_GIB=2.5, experts(2.5G)+dense(~1.4G)
  hit the 3900 cap => "IOVA guard refusing MEM_CREATE" => dense pack fails => "warmup prefill FAILED" (abort,
  NOT a wedge — board stayed healthy). So two IOVA claimants: MoE-expert pool AND dense weights.
- FIX for a clean completing run: cap ORK_MOE_HOT_GIB=1.8 (experts 1.8G + dense 1.4G = 3.2G, ~700M headroom).
- LEVER to raise the NPU prefill share past 14.5% (toward the 88% balance): EXPAND IOVA via ORK_DOMAINS>1
  (spread experts across N IOMMU domains; ~6 domains reaches the balance point). Multi-domain is wedge-prone
  (multi-domain-runtime: non-deterministic layout, swap-bound) => validate carefully, single-domain clean first.

## ★★ IOVA-CAPACITY CONSTRAINT (2026-07-15) — 35B backbone alone ~fills one domain
Four single-domain runs all abort at "warmup prefill FAILED" (IOVA guard, graceful — NOT wedge, board healthy):
  FRAC=0.15 default HOT_GIB=2.5 -> 954 experts, domain 0 at 3899 MiB (over 3900) -> dense pack Bb[0] fatal
  HOT_GIB=1.8/1.0 -> STILL 954 experts (get_hot budget counts K*N ~1MiB/expert, all 954 fit under 1GiB) -> 3899
  HOT_GIB=0.5 -> 512 experts (~512MiB) BUT domain STILL at 3889 -> +16MiB dense pack fatal
=> The 35B-A3B BACKBONE (attn/shared/embed resident wcache) alone ≈ 3.4-3.9 GiB — it nearly FILLS one 3900 MiB
   domain by itself. So single-domain CANNOT hold backbone + ANY MoE-expert offload. HOT_GIB barely helps
   (experts are only ~1 GiB; the backbone is the fill). The baseline (MOE_NPU off) completes because experts
   go to CPU (no IOVA). => Multi-domain (ORK_DOMAINS>=2) is REQUIRED to fit backbone + expert offload, not
   just to scale the NPU prefill share. This is the designed fix ("spread domains" per the guard message).
- overlap-eff: 0.58x (cold, 954 experts, 3508ms first-touch packs in-window) -> 0.80x (512 experts, fewer
  cold packs). Warm/amortized should approach max(npu_t,cpu_t). The eff drag is cold first-touch packing.
- NPU still ~7×/expert at prefill (npu 64ms/512e=125us vs cpu per-expert ~730us). Lever intact; IOVA is the gate.
