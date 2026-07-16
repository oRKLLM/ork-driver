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

## ★★★ SPEC CONFIG COMPLETES (2026-07-15) — ORK_DOMAINS=2, PATH-B, PARK, FRAC=0.15
35B-A3B-Q4_K_XL, ork_bench P128/G64: **prefill 20.50 tok/s | decode 3.71 tok/s**. NO WEDGE, board healthy.
Multi-domain fit: 4.29 GiB resident across 2 domains (dom0 1.94 / dom1 2.36), per-token churn=0.
- Concurrent CPU-int4(Q4_K) ‖ NPU-int8 MoE split WORKS end-to-end. But NPU contribution MARGINAL:
  PATH-B npu=367ms ‖ cpu=13743ms; hot-expert hit-rate 15.9% (IOVA-limited pool) -> 84% experts CPU cold GEMV.
- ★ BOTTLENECK is NOT the MoE split: wallsum 78s = quant(act) 25s (32%) + resolve(weight-stream) 24s (31%)
  [dequant src->f32 14.5s + pack i8->IOVA 9.1s] + run 28s (36%). The 22GB model streamed through the 7800MiB
  2-domain window => constant weight re-resolution DOMINATES. NPU MoE (367ms) is noise vs this.
- Implication for the plan: raising the NPU prefill share (bigger FRAC / more domains) helps only the 367ms
  slice; the win is capped by the streaming+quant overhead until THOSE are attacked (resident .orkpack so no
  re-resolve; act-quant reuse). The concurrent split is correct + safe but not the dominant lever at 35B-stream.
- NEXT: controlled baseline (MOE_NPU off, same 2 domains) to isolate PATH-B delta.

## ★★★ CONTROLLED RESULT (2026-07-15) — PATH-B LOSES on 35B (streaming-bound, not compute-bound)
                    prefill    decode
  BASELINE (MoE->CPU, 2 dom)  22.77      6.14   tok/s
  SPEC (PATH-B MoE->NPU)      20.50      3.71   tok/s
  delta                       0.90x      0.60x
=> MoE-NPU offload LOSES both regimes on the 35B. Re-derived on the current stack => moe-offload-closed HOLDS.
Causes (both match the user's framing):
- PREFILL 0.90x: NPU MoE slice=367ms but enabling it ADDS 26s of NPU expert run-time (baseline run 2.2s ->
  spec 28s) + gather/quant/park + IOVA pressure on the STREAMING path (24s weight re-resolve) that actually
  dominates. Compute wasn't the bottleneck => splitting compute didn't help; it hurt (overhead).
- DECODE 0.60x: MOE_NPU=1 routes decode (M=1) experts to the NPU hot-pool = the warned ~3x M=1 loss. The
  user's regime rule (decode->CPU) is NOT enforced yet (task #8). Enforcing it recovers decode to ~6.14.
- PREREQUISITE LEVER: the 24s weight re-resolve (dequant Q4_K->f32->int8->tile every pass) is the real wall.
  A FULL resident .orkpack (pre-tiled int8 bytes, load-not-repack; board has only a 126MB STUB) collapses it.
  Only AFTER that does the concurrent compute split become the deciding lever. Plan sound; split is premature
  at 35B-stream until streaming is fixed. Caveat: multi-domain layout is non-deterministic (some run variance).

## ★★★★ ACCURACY DE-RISK (2026-07-15) — no int4 is BOTH accurate AND free-to-upconvert
CPU-only probe (llama.cpp tools/cpu_i4_vs_q4k.cpp, real ggml Q4_K kernel), K3584 N18944, rel-RMSE vs f32:
  int4 per-channel (FREE inflate):        0.1479   (unusable — 10x worse than Q4_K)
  int4 per-64-block (no free inflate):    0.0627   (per-block recovers 2.4x but loses free-inflate + 4x>Q4_K)
  Q4_K per-32-block+super (ggml):         0.0145   (best)
  speed: ork per-channel int4 only 1.06-1.12x faster than Q4_K (both ~24 GB/s, memory-bound)
=> The FREE int4->int8 nibble-expand ONLY works for per-channel UNIFORM int4, which is unusably lossy (14.8%).
   Accurate 4-bit needs per-block scales (Q4_K), which CANNOT free-inflate (block scale must be applied = real
   dequant = the resolve tax). So "CPU-int4 + free-upconvert-to-NPU-int8" is OUT on accuracy. Q4_K is ALREADY
   the optimal CPU 4-bit (accurate + bandwidth-competitive) => keep the CPU on Q4_K; don't convert to ork-int4.

## ★★★★★ REVISED DECODE DESIGN (2026-07-15, user) — no-upconvert + pre-tiled-int8 + hide-swap-behind-CPU
Sidesteps the accuracy wall by removing the runtime upconvert:
- NPU tier = PRE-TILED int8 in the orkpack (resolve paid ONCE at build, never at runtime). Decode "domain swap"
  = pure DMA of pre-quantized int8 tiles into IOVA (no dequant, no upconvert).
- PER-TOKEN dynamic residence (not a fixed sensitive share -> NPU would idle): swap in the token's ACTIVE
  experts; NPU picks up its share once the swap lands.
- HIDE the swap: CPU starts the token on its Q4_K share immediately; swap latency masked by the CPU window.
- CPU stays Q4_K (accuracy data says don't touch it). No upconvert anywhere.
- Near-term constraint: "no upconvert" + int4-default => needs NPU W4A4 (deferred). So near-term NPU tier=int8.
- FEASIBILITY GATE = swap-hiding budget: per-token active int8 tiles must DMA into IOVA within the CPU window
  (~1.1GB/tok @ ~11GB/s ≈ 100ms vs ~160ms CPU decode). Plausible but TIGHT — the next thing to measure.

## ★★★★★★ int5 + LUT DE-RISK (2026-07-15) — int5→int8→W8A8 unblocks the decode design WITHOUT W4A4
Weight-recon rel-RMSE (Gaussian weights, tools/cpu_i4_vs_q4k.cpp):
  int4 per-chan uniform (free inflate) 0.1559 | NF4 per-chan LUT (free) 0.1093 | int5 per-chan 0.0728
  int4 per-64-block 0.1076 | Q4_K per-32-block 0.0713
- ★ int5 per-channel = 0.0728 ≈ Q4_K 0.0713 — padding int4->int5 (16->32 levels) HALVES the step, matches Q4_K,
  and KEEPS per-channel scale (inflate-friendly). User's int5 instinct validated.
- ★ NF4 codebook LUT (the "lut in the orkpack") = 0.1093, 1.4x better than uniform int4, nibble-packs + free LUT.
- ★★ KEY: int5 value [-15,15] fits an int8 byte EXACTLY => int5->int8 expand is LOSSLESS. So
  int5-storage -> int8-container -> W8A8 = Q4_K-grade WEIGHT accuracy + INT8 ACTIVATIONS (coherent). This
  SIDESTEPS the W4A4 PPL-104 problem (which was int4 ACTIVATIONS, not weights). => the decode design is
  buildable NEAR-TERM without the deferred W4A4: NPU least-sensitive share = int5->int8->W8A8 (accurate,
  coherent, repack-free-ish = cheap lossless widen, hidden behind the swap). True zero-copy W4A4 = future.
- Caveats: Gaussian synthetic (real weights have outliers where Q4_K per-block pulls ahead — validate int5 on
  a real tensor); int5 doesn't nibble-align (needs 8-vals/5-bytes bit-packing, more complex unpack than int4).

## ★★★★★★★ FORMAT DECISION (2026-07-15) — NF4 (free vqtbl inflate), NOT int5 (ALU-bound at M=1)
tools/int5_probe.c (M=1 GEMV, 4 threads, board), lossless round-trip PASS for all:
  int4 uniform 23.4 GB/s (0.52x int8 time, acc 0.156) | NF4 vqtbl 22.7 GB/s (0.54x, acc 0.109) | int5
  bit-plane 11.8 GB/s (1.29x int8 time — SLOWER than int8!, acc 0.073) | int8 24.4 GB/s
- ★ int5's 5th-bit merge (bit-plane spread) is ALU-BOUND at M=1 — doesn't hide, ends up slower than int8.
  No cheaper int5 encoding (bit-packed = worse cross-byte ALU; vqtbl2 needs the same 5-bit index assembly).
  int5 DROPPED for decode.
- ★ NF4 (int4 index -> int8 via single vqtbl1q LUT) is MEMORY-BOUND-FREE (22.7 = int4 speed) with better
  accuracy (0.109 vs int4 0.156). The vqtbl lookup hides fully. This is the free-inflate format to use.
- ★ REVISED format decision:
  - CPU bulk STAYS Q4_K (memory-bound, acc 0.071, zero refactor — already optimal).
  - NPU offload tier = NF4 (free vqtbl inflate -> int8 -> W8A8, repack-free, compact, least-sensitive OK).
  - int8-storage for any full-accuracy NPU tier (zero-copy W8A8, 2x mem).
  This kills the resolve tax for the NPU feed WITHOUT int5's ALU cost. Lean int4/NF4 inflate confirmed free;
  int8 write to the DMA tile remains the only non-free NPU-feed cost (fundamental for an int8-reading NPU).

## ★★★★★★★★ BUILD PHASE (2026-07-15) — autonomous end-to-end build
DONE + VALIDATED (ork-driver, board CPU-only):
- CPU kernel ladder (int4/NF4/int5/int6/int8) NEON dots — lossless round-trip PASS; int4/NF4 free
  (0.56-0.57x int8 time), int5/6 ALU-priced levers (1.37/1.88x). tools/int5_probe.c.
- ork_cpu_pack (f32->tiered) + ork_native_cpu.h reusable header — pack->gemv vs f32 coherent:
  int4 .157 / NF4 .107 / int5 .068 / int6 .035 / int8 .009. tools/ork_native_roundtrip.c.
- NF4 fast vqtbl inflate already in ork-driver (inflate_chan_nf4_i8).
KEY: the CORE tiered orkpack needs NO new code — int4+NF4 (ORK_NF4, quant_kind in blob) + int8
  (source-type tier) + CPU-only build (ORK_ORKPACK_CPU) ALL EXIST. So the prerequisite full orkpack
  (NF4 bulk + int8 sensitive) is buildable with current ggml-ork. int5/int6 = optional refinements.
REMAINING (the genuinely-new integration):
- #12 CPU reads ork-native int4/NF4 (not ggml Q4_K) — THE crux; CPU-only run validates (board-safe).
- #14 swap-hidden decode pipeline (CPU int4 ‖ NPU int8 non-blocking) — needs board (wedge-risky, END).
- #15 full orkpack build on 35B + e2e via ork_bench+bench_monitored.sh — board (wedge-risky, END).
- int5/int6 orkpack dtype (DT_I5/I6) + dump/load — refinement (CPU-only).
SCHEDULING: safe code + compile now; wedge-risky board (orkpack build, pipeline, e2e) batched at END.

## BUILD PROGRESS (2026-07-16)
IN FLIGHT (board, monitored): full NF4 orkpack build — ork_bench ORK_MOE_NPU=1 ORK_NF4=1 ORK_PERSIST=
  full-nf4.orkpack. CONVERT mode persists ALL experts (ork_persist_write_experts) + computes MoE on CPU
  (board-safe, no NPU MoE submit). Target ~20GB. (First attempt w/o MOE_NPU only dumped 1.4GB dense — the
  sparse-touch gap; MOE_NPU=1 makes the expert handler fire + dump all experts.)
DONE + VALIDATED this phase:
- CPU kernel ladder + ork_cpu_pack aligned to ork-driver Bi4 CONSECUTIVE layout (int4/NF4) so CPU reads the
  EXACT orkpack bytes (true single format); int5/6 lean. Roundtrip coherent (int4 .157/NF4 .107/int5 .068/
  int6 .035/int8 .009). include/ork_native_cpu.h.
- hybrid_decode_probe.c (decode pipeline aggregate validator) — built, runs END-phase (NPU-free).
- nf4_fit.c (model-fitted NF4 codebook, Lloyd-Max) — built, runs when base model lands.
END-PHASE PLAN (after orkpack done + NPU free + base model down):
1. Validate orkpack ~20GB + loads coherent (ork_bench read-mode).
2. nf4_sample (extract per-channel-normalized weights from base model) -> nf4_fit -> model NF4 LUT.
3. hybrid_decode_probe -> does CPU-int4-bulk ‖ NPU-int8 overlap win at M=1 (gates #14).
4. #12 wire ggml-ork cold-expert CPU path to read orkpack NF4 (ork_native_cpu.h) — CPU-only validate (PPL).
5. #14 swap-hidden pipeline (if probe wins) + #15 e2e bench via ork_bench + bench_monitored.sh.
