# DFlash IOVA-domain fix — WIP recovery doc

_Last updated: 2026-07-09 (session start)_

## Goal
Give the DFlash **draft** model its own IOMMU domain(s) so it doesn't collide with / evict the
**target**'s resident weights on the RK3588 NPU. Confirmed by the user as THE DFlash-specific speed
lever. Must be handled automatically by ork-driver/ggml-ork (NOT via an `ORK_DOMAINS` env).

## Current hypothesis (to prove/refute with the board diagnostic)
DFlash-on-NPU is ~10× SLOWER than plain 9B decode:
- plain 9B on NPU  = 1.73 tok/s decode (SSM floor: Gated-DeltaNet layers on CPU, 628 splits)
- DFlash on NPU    = 0.18 tok/s, **7.35 s per target forward** (20 forwards / 27 tok / 147 s; tau 2.70, accept 1.70/16)

A batched M=16 verify forward should cost ~the same as M=1 (weights stream once). 7.35 s vs ~0.58 s
(M=1) ⇒ **~12× penalty**. Prime suspect: draft+target don't stay co-resident, so switching
draft→target→draft **evicts + re-packs** (quant+tile+import) weights every forward (~7 s = the re-pack).

Fix = both resident in SEPARATE domains, no eviction → target forward drops to ~0.58 s.

## Two candidate root causes (diagnostic disambiguates)
- A. wcache budget too small → LRU evicts (`ork_mm_free` at ggml-ork.cpp:470) → re-pack each forward.
- B. Draft weights land in the SAME domains as the target → per-domain IOVA overflow → eviction/re-pack.

## Key code facts (read, not yet changed)
- `g_ork_ctx` (ggml-ork.cpp:4103) is a **process-global singleton** — target & draft share one ork
  ctx + domain cursor/bytes. Draft loads AFTER target in the same process.
- `n_gpu_layers=0` ⇒ weights pack **lazily** on first matmul (ork buffer type IS a real weight buft,
  4256-4260). So `ork_weight_domain` (424-447) assigns domains as matmuls execute; cursor advances
  only at layer boundaries.
- Auto n_domains from .orkpack footprint (4127-4166); `ORK_DOMAINS` is override-only. Live-pack (no
  persist idx) ⇒ n_domains=8 (4170).
- Residency dump earlier showed target in domains 0-3 (~11.4 GiB), domains 4-5 EMPTY ⇒ **draft did
  NOT get its own NPU domain** (either ran CPU, or collided). This is the collision to root-cause.

## Test harness (board = `board` alias, 10.3.0.236 khazad-dum, user michael)
- Build: `~/llama-q4-v0659/build/bin/ork_bench` (ork-driver submodule under it).
- Target: `/var/lib/orkllm/models/deepreinforce-ai/Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q8_0.gguf`
- Draft:  `/var/lib/orkllm/models/onion515/ornith-9b-dflash/ornith-9b-dflash-q5_k_m.gguf`
- Run: `sudo timeout -s INT 300 env ORK_VERBOSE=1 ORK_DFLASH_DRAFT=<draft> ./ork_bench <target> /tmp/dfp.txt 32 8`
- Board safety: single-stream NPU; `timeout -s INT` (never kill -9); NPU wedge+SSH ok → `sudo reboot`;
  hard wedge → Rock 5B Plug power-cycle (Home Assistant MCP).

## State of working tree
- Fixes (a)/(b) done in oRKLLM (device selection + pool.js GGUF NPU) — UNCOMMITTED, separate from this.
- ork-driver / ggml-ork: NO changes yet for the domain fix.

## ★★★ PIVOTAL FINDING (2026-07-09) — it's NOT domains, it's the min_m gate ★★★
ORK_VERBOSE DFlash diag (block=16, target=9B Q8_0, draft=q5_k_m):
- `[ork RESIDENT] final: 0.00 GiB resident across 8 domains; churn = 0` — **ZERO NPU residency**.
- `graph splits = 1` every forward (vs plain-9B's 628) — **entire target graph runs on CPU**.
- 0 "packed weight" lines; M seen by supports_op: M=1(3063) M=16(2514, target verify) M=17(3450,
  draft) M=512(355, attn-score/dynamic src0) + M=2/4/9/10/11/13 (accepted-len variants).

Root cause = `ggml_backend_ork_device_supports_op` MUL_MAT gate (ggml-ork.cpp ~4403-4421):
- Q8_0 target ⇒ not grouped/expert ⇒ `threshold = min(min_m,32) = 32`.
- L4411 drops threshold→1 ONLY when `n_domains<=1` (single-domain). DFlash 9B is **n_domains=8**
  (live multi-domain), so threshold STAYS 32.
- Target verify M = block_size = **16 < 32** ⇒ `pass_m_threshold=false` ⇒ **declined → CPU**.
So the DFlash batched verify — the entire point of DFlash (kill the M=1 regime) — never reaches the
NPU because block(16) < min_m(32). Domain collision is a SECOND-order issue that only bites once
matmuls actually route to the NPU.

Answers the user's question: plain-9B PREFILL (bs=32, M=32≥threshold) hits NPU (628 splits); plain-9B
DECODE (M=1<32) is CPU (the SSM/decode floor). DFlash verify M=16 falls in the dead zone: >1 (we WANT
NPU) but <32 (declined). "Validated on NPU earlier" must have used block≥32 or single-domain.

## Fix design (two parts)
1. **Route the batched verify to the NPU.** The M<32 gate is tuned for M=1 decode loss; a batched
   M=16 verify of a 9B amortizes the submit floor far better. Options: (a) block_size≥32 (clears the
   existing gate, no code change, but may lower acceptance); (b) DFlash-aware / lower min_m for the
   verify. TEST which M actually wins on NPU for the 9B before hardcoding.
2. **Domain co-residence (the original ask).** Once matmuls route to NPU, ensure draft+target both
   stay resident (separate domains, no eviction/re-import thrash). The earlier 147s/7.35s-per-forward
   run was likely M-on-NPU WITH domain thrash — so this still matters, second.

## ★★★★ SECOND PIVOT (2026-07-09) — domains are FINE; the issue is M < NPU crossover ★★★★
`ORK_MINM=8` run (block=16, M=16 verify routed to NPU) — FULL resident dump:
- **Target: 11.38 GiB fully resident** (dom0-3: 3.20/3.20/3.20/1.77), **churn = 0**.
- **Draft: 1.98 GiB fully resident** (dom0-3: 0.52/0.58/0.58/0.30), **churn = 0**.
- 4 ork ctx inits (target+draft each make buft+backend); both models co-resident, NO eviction, NO
  thrash. The 243 "packed weight" lines = ONE-TIME initial packing, not churn (churn counter = 0).
- Draft shares physical IOMMU domains 0-3 with the target but with headroom (dom0 3.20+0.52=3.72 <
  4 GiB) — so a dedicated draft domain is NOT needed. **The domain hypothesis is refuted by data.**
- No fault / no soft-reset ⇒ batched M>1 on non-0 domains is SAFE (goes through mcworker_pref, not
  the M==1 mcworker_dec errno-22 path).

RESULT: NPU **26.57s (0.30 tok/s)** vs CPU **16.03s (0.50 tok/s)** — **NPU LOSES at M=16.** Both fully
resident, zero churn ⇒ the loss is NOT residency/domains. It's the M=16 batched verify being BELOW
the NPU crossover: the ~365µs×~197-submit floor + per-forward multi-domain scratch switches are ~FLAT
in M, while CPU cost is ~LINEAR in M. So small-M loses on NPU; NPU wins only at larger M (prefill
regime). This is EXACTLY what min_m=32 protects against — the gate is correct.

## Real fix (reframed): make the verify batch clear the crossover, not split domains
- DFlash's premise ("batched verify kills the M=1 regime") only pays off if block ≥ the NPU crossover
  (~M=32 for the 9B multi-domain). block=16 sits in the dead zone (>1 but <crossover).
- block ≥ 32 clears the EXISTING min_m=32 gate with NO code change. Trade-off: bigger block usually
  lowers acceptance (tokens/cycle) → net tok/s = f(forward_time, acceptance) must be measured.
- If NPU crosses CPU at block=32/64 AND acceptance holds → the fix is "set DFlash block to the NPU
  crossover" (a DFlash config default, ork-side gate unchanged). If NPU still loses even at the
  crossover, DFlash-on-NPU for a >4GiB SSM-hybrid target is not viable and the target verify should
  stay on CPU (like plain decode) — DFlash's value would then be CPU-only spec-decode.

## Also note (not chased): acceptance
tau=2.0 (accept 1/16) in these runs vs the commit's validated tau=6.06 — almost certainly my G=8/short
sample (4 cycles, high variance), NOT a regression. Using G=48 now to get real acceptance.

## ★★★★★ THIRD FINDING — acceptance is the dominant blocker, not NPU/CPU ★★★★★
block=32 NPU (default gate, G=48): cycles=18, **mean_accept=1.78/32**, tau=2.78, 50 tok in 179.27s =
**0.28 tok/s** (5.0s/forward). Bigger block did NOT help net throughput.
- mean_accept 1.78/32 (tau 2.78) vs the validated commit's **tau=6.06**. At this acceptance DFlash is a
  NET LOSS vs plain decode (~1.73 tok/s) regardless of NPU vs CPU. The NPU-crossover question is moot
  until acceptance is fixed.
- ork_bench's run_dflash_bench loop is **byte-for-byte equivalent** to the reference
  `examples/dflash/dflash.cpp` skip-ahead loop (same grab_features / encode_append / argmax / state
  checkpoint / accept-longest-prefix + bonus). So NO obvious port bug — ork_bench is faithful.
- Candidate causes of low accept: (a) PROMPT variance — "why is the sky blue" is short/creative =
  high branching = low acceptance; the validated 6.06 may have used a more predictable prompt; (b) the
  draft GGUF (ornith-9b-dflash-q5_k_m, Jul 9 02:33) quality/conversion. Draft PREDATES the validated
  commit (Jul 9 10:48) so likely same draft ⇒ leans toward PROMPT variance, not a converter regression.

RUNNING: reference `llama-dflash` (GGML_DISABLE_VULKAN=1, block=16, same models/prompt) → establishes
the acceptance CEILING. If ref≈2.78 ⇒ ork_bench faithful + low-accept is prompt/model (validated 6.06
was a favorable config); if ref≫ork_bench ⇒ a real port/config gap to hunt.

## CONSOLIDATED CONCLUSION (for the user)
1. Fixes (a)/(b): DONE in source (device-selection + pool.js NPU), uncommitted.
2. Assigned "IOVA domain fix": **not needed** — target (11.38G) + draft (1.98G) already co-resident,
   churn=0, no eviction. Domain hypothesis refuted by the residency dump.
3. The min_m=32 gate silently routed the M=16 verify to CPU ⇒ "validated on NPU" was CPU all along
   (residency never checked before). Correct to surface.
4. M=16/32 batched verify LOSES to CPU on the NPU (submit-floor-bound below the ~M32 crossover) — the
   gate is right. Bigger block clears the gate but doesn't win net.
5. Acceptance (tau ~2.78 vs validated 6.06) is the real DFlash blocker; likely prompt/config, pending
   the reference run.

## ★★★★★★ RESOLVED: acceptance WORKS — it's prompt-dependent, not a bug ★★★★★★
Reference `llama-dflash`, SAME models, two prompts (pure CPU, ORK_OFF=1):
- creative "why is the sky blue" (block16): tau=2.72, 0.65 tok/s.
- predictable "Count from one to thirty: one, two, three, four, five, six, seven, eight,": **tau=7.000,
  accepted 6.000/16, 2.51 tok/s.**
⇒ DFlash mechanism is CORRECT (draft/context/encoder wiring fine). Acceptance scales with prompt
predictability (7 on structured, 2.7 on creative). The validated commit's tau=6.06 was just a
favorable prompt — NO regression, NO port bug (ork_bench == reference within variance).
⇒ At tau=7, **DFlash on CPU (2.51 tok/s) BEATS plain 9B decode (~1.73 tok/s)** — DFlash IS viable on
CPU for predictable content. NPU-verify stays the losing path (below crossover + soft-resets).

## User directive (2026-07-09): monitor-wrapper for ork_bench
Always wrap ork_bench NPU runs in `tools/bench_monitored.sh` to VALIDATE NPU usage via live telemetry
(NPU load from /sys/kernel/debug/rknpu/load, needs root; RAM-BW from devfreq/dmc/load; CPU/GPU/swap;
avg+peak). Corroborates the residency dump with actual utilization%.
Path: `~/llama-q4-v0659/ggml/src/ggml-ork/ork-driver/tools/bench_monitored.sh --label L -- <cmd>`.
RUNNING: monitored ork_bench DFlash (default gate, predictable prompt) → expect NPU%≈0 (CPU verify).

## Decision (user): CHASE ACCEPTANCE FIRST → shown WORKING (prompt-dependent). DFlash viable on CPU.

## ★★★★★★★ NEW DIRECTION (user 2026-07-09): add the ATTENTION (batched dynamic GEMM) kernel ★★★★★★★
WHY: Ornith-9B is CPU-bound because its ops are declined by ggml-ork's static-weight-only MUL_MAT gate.
Confirmed: qwen35 GDN uses the CHUNKED (matmul) form — `src/models/delta-net-base.cpp
build_delta_net_chunking` has real `ggml_mul_mat` (kb/kq), but they're BATCHED (ne[2]=n_chunks,
ne[3]=heads·seqs>1) + DYNAMIC (computed src0) → `src0_static_2d` gate declines → CPU. Same for the
attention core (QK^T/AV: operands are activations). So the fix is a batched/dynamic-operand GEMM.

DONE (ork-driver): `ork_bmm_i8/i4/fp16(ctx,nbatch,M,K,N,A,B,C)` — C[b]=A[b][M,K]·B[b][K,N], both
operands dynamic, B[b] packed per-batch (i8 reuses one buf via repack; i4/fp16 pack+free per batch;
chaining = perf follow-up). include/ork_npu.h + src/npu.c (before ork_fwht_norm). Self-validating test
tools/re/test_bmm.c. **VALIDATED on board (fresh dir ~/ork-bmm): i8 EXACT, i4 EXACT, fp16 maxrel
0.0004 — all 3 dtypes PASS, no wedge.** Attention-shaped (nbatch=heads, M=qlen, K=head_dim, N=kvlen).

NEXT (task #23, ggml-ork): supports_op ACCEPT batched (ne[2]/ne[3]>1) / dynamic-src0 MUL_MAT under a
flag (ORK_ATTN); graph_compute per-batch quantize src0/src1 (i8/i4) or fp16 → ork_bmm_* → dequant.
Then coherence (ork_ppl) + NPU% (bench_monitored) on a pure-attention model, then qwen35 GDN. CAVEAT:
submit floor (~365µs × nbatch × 2 × layers) is the real risk at decode/small-chunk — chaining +
prefill-large-M are where it can win; measure before claiming.

## #23 ggml-ork batched-MUL_MAT → ork_bmm_fp16 (IMPLEMENTED, validating)
ggml-ork.cpp: supports_op ACCEPTS batched (ne[2]/ne[3]>1) contiguous f16/f32 MUL_MAT under `ORK_ATTN`
(K%32,N%16); graph_compute routes them to new `ggml_backend_ork_bmm_fp16` (per-batch slice → fp16 →
ork_bmm_fp16 → f32 dst, GQA/broadcast-aware). Synced ork_bmm/norm into board submodule; `cmake --build
--target ggml-ork` OK, libggml-ork.so relinked. Validating on Ornith-9B (GDN batched chunked matmuls):
coherence + NPU% (bench_monitored) + routing (pack count) vs ORK_ATTN-off baseline (prefill 27 tok/s,
NPU 1.5%/18%). fp16 path first (no int8 scale bookkeeping); i8/i4 batched = follow-up.
RESULT (Ornith-9B, ORK_ATTN=1): batched matmuls DO route (splits 628→644, coherent rc=0, no wedge) but
impact ~ZERO — prefill 27.63 vs 27.30, NPU 1.6%/19% vs 1.5%/18%, CPU 55% still pegged. CONFIRMS: the
GDN CPU wall is the SEQUENTIAL recurrence (gated_delta_net/solve_tri/cumsum/exp)+attn-core+norms, NOT
the batched matmuls (too small a slice + submit-floor). GDN-on-NPU via matmul offload = neutral.
Complement (pure-attention Qwen3-1.7B, ORK_ATTN on vs off): **NOTHING routed** — splits 480 both, NPU
2.4%/7% both, prefill 73.5→75.1 (noise). The real attention matmuls (QKᵀ/AV) read NON-CONTIGUOUS views
(KV-cache + permuted Q) → the `ggml_is_contiguous` gate declines them. #23 CONCLUSION: batched-matmul
offload is NEUTRAL on both classes — GDN matmuls too small (sequential ops dominate); attention matmuls
non-contiguous (+ would be submit-floor-bound per-head). ork_bmm+ORK_ATTN = built, coherent, GATED OFF;
not a win for GDN/attention on RK3588. Matches the whole session's pattern: small-M/small-op loses to
the ~365µs submit floor; NPU's real win = the large static-weight projections the existing path handles.
#23 DONE.

## test_bmm now a make-test EXAMPLE (examples/test_bmm.c) — PASSED via `make test-only T=test_bmm`.

## Uncommitted (consult before discarding): oRKLLM fixes a/b; ork-driver ork_bmm_* + norms + test_bmm
## (examples/) + ork_bench ubatch fix; llama.cpp ggml-ork ORK_ATTN batched dispatch. Board: ~/ork-bmm
## (standalone test), ~/llama-q4-v0659 (ggml-ork build, libggml-ork rebuilt with ork_bmm).
