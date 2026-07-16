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
