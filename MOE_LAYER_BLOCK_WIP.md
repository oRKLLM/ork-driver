# MoE per-layer expert-block residency (WIP) — the "ORK_MOE_REMAP" task, redesigned

## Origin
User asked to wire a zero-copy "page remap" so a MoE domain swap holds only the active
experts, prefetched behind prefill, and make it the **default** MoE routing (no permanent env gate).

## RECKONING (2026-08): most of this is ALREADY BUILT — reconcile, don't reinvent
Consulting the wiki (should have FIRST) changed the picture:
- **Multi-domain residence (Exp-2026-06-28)** is the real churn fix: 4 GB IOVA cap is PER-DOMAIN; rknpu
  exposes 8 domains -> ~32 GB resident, bit-exact, ZERO per-token churn, via iommu_domain_id (already
  wired by the auto-sizer, task #30). librkllmrt does exactly this: 8 MEM_CREATE TOTAL, no per-token churn.
- **Zero-copy dma-buf EXISTS** (ork_dma_import/ork_dma_import_fd/bimport: dma-buf mmap'd for CPU fill +
  IOVA-mapped for NPU, same pages, no copy). My "must memcpy mmap->IOVA" claim was WRONG.
- **MoE offload already built + measured (Exp-2026-06-23)**: handler, routing (ORK_NO_EXPERT_REPACK),
  repack-pool, group-by-expert prefill — measured 0.28 t/s vs 22.59 CPU (~80x). BUT that page blamed the
  SUBMIT FLOOR; S0 shows the submit floor is now 2% (chain fixed it) and the CHURN is 98% -> the wall
  MOVED, and multi-domain residence is what removes churn. So a residence-based re-attempt is warranted.
- **Layer-streaming + .orkpack + budgeted-LRU + (N_cores+1)-slice prefetch**: built/designed; DDR headroom
  ~90% => prefetch hides for free. Only needed for models > physical RAM.

**RE-SCOPE of #54:** make MoE experts RESIDENT across the 8 domains (librkllmrt recipe): partition
layers->domains, one resident buffer per domain (bcreate ~8x total, TO_DEV once, never freed), chained
submits + dom_activate per layer-group, ZERO per-token churn. S1's ork_mm_load_i8_block = the per-domain
consolidator (one bcreate, experts as views). S0's crash = the MoE path NOT using multi-domain residence
(crammed domain 0, botched advance to domain 1). Caveat: NPU runs int8 (int4 inflates resident), so 35B
experts ~30 GB int8 = at the 32 GB edge -> needs native-int4-resident or a small streamed remainder
(prefetch-hidden) for THIS model; a model that fits ~32 GB int8 is the clean case.

## 35B PATH = int4-NATIVE-RESIDENT (W4A4), not int8-inflate. PROC-PRECISION OVERLOAD added (compiles).
WHY (decisive): int8-inflate = 30GB > RAM -> STREAM -> ~8.7s expert loads/prefill -> loses to CPU (3s), and
prefetch CAN'T hide it (load 8.7s >> compute 0.1s; break-even needs M_e~970, i.e. ~31k-token batch). The ONLY
NPU escape is int4-RESIDENT: ~15GB FITS RAM+IOVA -> RESIDENT -> load ONCE at model-load, no per-prefill load.
Native-W4A4's per-row weight re-read penalty (why it loses for DENSE large-M) is TINY at MoE's M_e~2 -> the
tradeoff flips: W4A4-resident wins for sparse-MoE experts.
NUANCE: native int4 MAC can't apply the NF4 CODEBOOK (codes are indices, not values) -> W4A4 = UNIFORM int4
(code=value*scale), not NF4. Accuracy recovered via ORK_HADAMARD (rotation ~ NF4-competitive, runs native).
So "NF4-at-W4A4" isn't native-runnable; uniform-int4 + Hadamard is the on-NPU path.
DONE (compiles EXIT=0, dense-safe): ork_resolve_weight_i8 gained `int proc_prec=0`. proc_prec=1 -> int4-native
branch calls ork_mm_load_i4 (DT_I4 resident, run via ork_mm_run_i4). Default 0 = int8-inflate (unchanged).
NATIVE-DT_I4 EXPERT PACK — SCOPED (2026-08): the FORMAT + LOAD already exist (dense W4A4):
  ork_persist_write_i4native (ggml-ork.cpp:1123) dumps a DT_I4 ork_w (ork_w_dump) as ORKPACK_DT_I4_NATIVE
  (dtype 5) + bscale; ork_persist_load_i4native (1141) reloads via ork_mm_load_i4. The proc_prec=1 resolve
  branch (1435) already routes to ork_mm_load_i4. So dtype/format/load = DONE.
  GAP = the EXPERT pack. ork_persist_write_experts (1210) is a #pragma omp parallel CPU-ONLY loop (no NPU,
  by design, to saturate cores over 30720 experts). Two blockers:
   - the dense native pack builds ow.w via the NPU ork_mm_pack_i4 (FWHT-rotate->int4->tile->bcreate); calling
     that inside the omp-parallel loop WEDGES the single-stream NPU.
   - there is NO CPU-only native-int4 pack (only ork_pack_i4a8_cpu_blob for O4N1; no ork_pack_i4_cpu_blob).
  Two options to close it:
   (a) NEW ork-driver CPU pack ork_pack_i4_cpu_blob: dequant->FWHT-rotate->per-channel int4->DT_I4 tile->dump,
       BYTE-IDENTICAL to ork_mm_pack_i4+ork_w_dump. Keeps convert fast+parallel. Correctness-critical
       (a tile/rotation mismatch silently corrupts every expert) — validate vs the NPU pack byte-for-byte.
   (b) SERIAL NPU pack for the native-expert tier: pull it OUT of the omp loop, ork_mm_pack_i4(+Hadamard) per
       expert -> ork_persist_write_i4native. Correct, reuses existing code, SLOW one-time convert.
  Recommend (b) first (correctness, reuse) -> validate the whole W4A4 path -> then (a) for convert speed.

REMAINING to make it FUNCTIONAL (substantial; touch orkpack format + run dispatch -> focused fresh build):
 1. ork_persist_write_experts must emit NATIVE DT_I4 dumps (ork_mm_pack_i4 + ork_w_dump), NOT the O4N1
    compact form (ork_mm_load_i4 returns NULL on O4N1). New expert pack path / orkpack tier.
 2. MoE grouped run must use the int4 doorbell (ork_mm_run_i4 / colsplit/BCHAIN #49-53) for DT_I4 experts —
    the current bigA/run_stream_i8 path is int8-only (tasks[x].w DT_I8). Parallel int4 grouped run.
 3. ORK_HADAMARD accuracy (rotate act+weight) wired for the expert path.
 4. Wire the MoE split to pass proc_prec=1 (only after 1+2) — under ORK_MIXED_W4A4/residence-can't-fit-int8.
APIs confirmed present: ork_mm_pack_i4 (289), ork_mm_load_i4 (345), ork_mm_run_i4, run_i4_bchain (#49-53).

## STREAM PER-LAYER EVICT DONE (2026-08, compiles clean EXIT=0) — 35B runs bounded; PREFETCH is the perf piece
Added: ork_weight.is_expert flag; ork_wcache_evict_experts(ctx,budget) (LRU evict of is_expert entries ONLY,
dense untouched, respects wcache_pin); split now resolves experts on NPU in BOTH modes (removed the
!residence_stream gate), tags is_expert=true; STREAM mode pre-evicts experts to ORK_MOE_STREAM_MB (default
2 GiB) before each _exps resolve. Eviction is BETWEEN synchronous MUL_MAT_ID calls -> no mid-chain UAF.
=> the 35B (STREAM) now RUNS on the NPU bounded (no OOM, no domain-0 cram, no crash).

CRUX INSIGHT (why evict != fast): a SINGLE prefill touches each expert ONCE, so each expert is loaded
(inflate nibble->int8 + IOVA-map) once regardless -> ~2560 loads / ~8.7s stays ON THE CRITICAL PATH in
STREAM. The evict only BOUNDS residence; it does NOT reduce the per-prefill load. So STREAM prefill on the
35B runs but likely LOSES to CPU (21.3) until the load is HIDDEN.
 - RESIDENT (fitting model): load ONCE at model-load, churn-free for all later prefills/decodes -> the win.
 - STREAM (35B): re-loads experts each prefill -> the PREFETCH OVERLAP (dual/triple-slot: inflate layer L+1
   while the NPU computes L; DDR ~90% idle hides it) is what takes the load off the critical path = the perf
   piece. Build AFTER validating the evict runs bounded. (N_cores+1)-slice design in Exp-2026-06-25 addendum.
VALIDATE NEXT: reboot + ORK_QUANT=4 ORK_MOE_NPU=1 ORK_PROFILE=1 on the 35B mixed (experts-in-orkpack) ->
confirm it RUNS bounded (no crash, resident stays ~ORK_MOE_STREAM_MB), read moe_pack (expect churn still high
= load-on-critical-path, motivating prefetch), A/B vs CPU 21.3/8.69.

## RUN-FLOW WIRING DONE (2026-08, compiles clean EXIT=0, no warnings) — RESIDENT path live; STREAM=CPU (superseded: STREAM now runs on NPU bounded)
Batched-prefill split (ggml-ork.cpp ~3690/3705): hot_s type ork_hot_slot* -> ork_weight* (same .w/.bscale;
run-flow uses at 3797/3823/3867 unchanged). Resolve: RESIDENT (!ctx->residence_stream) -> ork_resolve_weight_i8
(expert=e, allow_evict=false -> co-resident for the chain, no mid-submit UAF) -> &wcache entry; STREAM or
miss -> cpu_expert (CPU NF4). get_hot no longer on this path (PATH-B, off-by-default, still uses it). eff_cap
removed. Dense untouched -> make test bit-exact.
STATE: RESIDENT layer-streamer (fitting MoE) = experts NPU-resident via shared import+domains, S0 churn gone.
STREAM (35B int8-inflate ~30GB > ~22GB RAM budget) = experts CPU (correct fallback, no crash) — proper
per-layer streaming-EVICT (co-resident within layer + evict prior layer, targeting EXPERT wcache entries not
dense) is the remaining follow-on for oversized-on-NPU.
VALIDATION NOTE: the 35B is STREAM->CPU, so it does NOT exercise the RESIDENT NPU path. To validate RESIDENT:
a MoE whose (dense+experts int8) footprint fits ~22GB (small MoE), or force via ORK_RESIDENCE_RAM_MB (only if
it truly fits RAM). reboot + ORK_QUANT=4 ORK_MOE_NPU=1 ORK_PROFILE=1 -> expect moe_pack churn ~0 (was 95%),
A/B vs CPU 21.3/8.69, bit-exact-ish (W4A4 incoherent-by-design; compare vs int4 CPU ref).

## (superseded) loader overload + RESIDENCE MODE built — run-flow wiring was the last step (now done)
RESIDENCE MODE (ggml-ork.cpp, compiles EXIT=0): ctx->residence_stream set ONCE by the auto-sizer from the
inflated footprint vs a RAM budget (~72% physical, ORK_RESIDENCE_RAM_MB override) — RESIDENT (fits: load
once, never evict) vs STREAM-by-layer (oversized). ctx->residence_footprint / residence_ram_budget stored.
This is the eviction POLICY that replaces per-call allow_evict (dense migration = task #55). 35B int8-inflate
~30GB > ~22GB budget => STREAM; a fitting MoE => RESIDENT.
LOADER OVERLOAD (done, compiles): ork_resolve_weight_i8(..., expert>=0) resolves an _exps slice via the same
streaming (import+domain+evict+wcache); dense (expert<0) byte-identical.
REMAINING — RUN-FLOW WIRING (batched-prefill, off-by-default ORK_MOE_NPU path), consume the mode:
 - RESIDENT: resolve all S experts via ork_resolve_weight_i8(allow_evict=false, expert=e); store &it->second;
   hot_s: ork_hot_slot* -> ork_weight* (same .w/.bscale fields); feed it->second.w into tasks[x].w. No evict.
 - STREAM: at each MUL_MAT_ID call START, evict prior layers' experts down to a ~2-layer budget (targeting
   EXPERT entries, NOT dense — shared wcache caveat), then resolve this tensor's S experts allow_evict=false
   (co-resident for the chain), run. Eviction only BETWEEN calls -> no mid-chain UAF.
 - Slot-type thread-through touches ~3800-3850 (tasks[x].w=hot_s[x]->w @3814, ork_w_domain(hot_s[0]->w) @3788,
   + the bscale scatter/combine). Map ALL hot_s uses before changing the type.
Then reboot + ORK_MOE_NPU=1 ORK_PROFILE=1 A/B vs CPU 21.3/8.69; churn should collapse from 95%.

## OLD PROGRESS NOTE (superseded): loader OVERLOADED for experts
The layer-streamer = OVERLOAD the dense loader (user's call), NOT a parallel cache, NOT touching dense.
DONE (ggml-ork.cpp, compiles clean EXIT=0, dense byte-identical by construction):
  ork_resolve_weight_i8(..., int expert = -1) — expert>=0 resolves ONE _exps slice via the SAME streaming
  path as dense (zero-copy import + ork_weight_domain + advance + evict + shared ctx->wcache). 4 edits, all
  no-ops for dense (expert<0): (a) signature +expert param; (b) key x = src0->data + expert*nb2;
  (c) persist lookup = ork_expert_key(name,expert); (d) persist-miss with expert>=0 returns end() (experts
  are orkpack-only here; miss -> CPU NF4). Dense path unchanged -> make test stays bit-exact.
WIRING CRUX (found while pushing the run-flow wiring): the batched run needs ALL S experts of a layer
CO-RESIDENT at submit (tasks[x].w = slot->w for every x, then one chained run_stream_i8). So the resolve
MUST pass allow_evict=false (else resolving expert i+1 evicts expert i that the chain still refs -> UAF;
this is the exact allow_evict guard at ork_resolve_weight_i8 ~1343). Store &it->second (unordered_map keeps
element pointers valid across inserts). BUT allow_evict=false => no eviction => experts accumulate; the 35B's
int8-inflated experts ~30GB > 31GB RAM, so it MUST evict OLD LAYERS. => the real streamer = CO-RESIDENT
WITHIN a layer + EVICT ACROSS layers (budget ~2 layers; free the previous layer's experts at each layer
boundary). That reconciliation IS the layer-streamer (not a wiring detail); wrong = UAF or IOVA/RAM wedge.
 - fits-in-RAM models (small MoE, or 35B kept int4-native-resident ~15GB not int8-inflate): wire cleanly
   with allow_evict=false, NO cross-layer evict needed (full residence). Validate here FIRST.
 - oversized int8 (35B int8-inflate 30GB): needs the cross-layer eviction (a per-layer FIFO free). Extra piece.

REMAINING (the WIRING — contained to the off-by-default ORK_MOE_NPU batched-prefill path):
  In ggml_backend_ork_mul_mat_id_i8's batched split, replace get_hot(e,...) at ~3470 (PATH-B) and ~3727
  (main split) with ork_resolve_weight_i8(ctx, src0, K, N, src0->nb[1], type, to_float, /*allow_evict*/true,
  /*expert*/e). It returns a ctx->wcache iterator; use it->second.w + it->second.bscale.data() where the
  code used the ork_hot_slot's .w/.bscale (SAME field names/types -> change the local slot vectors from
  ork_hot_slot* to ork_weight*). Feed it->second.w into the existing tasks[x].w grouped run (run_stream_i8 /
  run_chain). Streaming budget ~1-2 layers so eviction is FIFO-by-layer (prefill touches each expert once).
  get_hot stays for decode/single-domain fallback (untouched). Decode already CPU (npu_budget=0 @ M=1).
  Then: reboot + ORK_QUANT=4 ORK_MOE_NPU=1 ORK_PROFILE=1 -> confirm moe_pack churn collapses from 95%,
  no domain-0 cram, A/B vs CPU 21.3/8.69. Prefetch double-buffer = measured-gated follow-on.

## LOCKED DESIGN + EXACT IMPLEMENTATION RECIPE (2026-08, user-aligned) — LAYER-STREAMER, not get_hot
DESIGN (confirmed with user): storage = NF4 orkpack mmap (~15GB); PREFILL = NPU layer-streamer (per layer:
inflate that layer's NF4 experts->int8 into a small resident IOVA slot, grouped int8 doorbell run;
double-buffer 2 slots ~1.5GB, prefetch L+1 behind L; DDR ~90% idle hides it); DECODE = CPU NEON NF4 from
mmap (ork_cpu_gemv_m1, no IOVA/domains/get_hot). get_hot = the OLD decode-on-NPU hot cache = DEAD (decode
is CPU); do NOT put IOVA logic in it, do NOT use it for prefill.

VALIDATED DIAGNOSIS (2026-08, rebooted, clean, big-core, ORK_MOE_NPU=1 on the 16.9GB experts-in-orkpack):
existing batched-prefill (which calls get_hot) => pack/repack 95% (2560 first-touch packs @3.4ms),
chain-submit 3%, "domain 0 full -> advancing to domain 1" (get_hot crams ONE domain, doesn't spread).
prefill 5.98 / decode 0.11 (decode number is an ork_bench artifact; real CPU decode = 8.69 via llama-bench).
CONFIRMS: get_hot is the wrong mechanism; the churn is per-expert MEM_CREATE in one domain.

EXACT RECIPE (non-duplicative — reuse the DENSE streaming, don't build a parallel cache):
 - The dense wcache streaming loader is INLINED at ggml-ork.cpp ~1349-1431: ork_wcache.find(x) -> on miss,
   ork_weight_domain(K*N, layer)+set_pack_domain, ZERO-COPY import (ork_mm_load_i8_import/load_i4a8_import,
   ORK_NO_IMPORT->copy), advance-on-full loop, emplace in ctx->wcache, evict via ork_wcache_evict/ork_spool.
 - Adapt it for EXPERTS: key x = src0->data + (size_t)e*src0->nb[2]; persist entry = persist_idx[
   ork_expert_key(src0->name, e)]; bscale trailer per-expert. Factor 1349-1431 into a callable helper
   ork_expert_stream_get(ctx, src0, e, K, N) -> {ork_w*, bscale} that both the dense path and the MoE
   batched path call (behavior-preserving extraction for dense = must stay make-test bit-exact).
 - In the MoE batched-prefill split, REPLACE get_hot(e,...) at ggml-ork.cpp ~3470 (PATH-B) and ~3727 (main
   split) with ork_expert_stream_get(...). Feed its ork_w into the existing tasks[x].w grouped run
   (run_stream_i8 / run_chain). Set the streaming budget to ~1-2 layers (double-buffer) so eviction is FIFO-
   by-layer (prefill touches each expert once). Leave get_hot in place for decode/single-domain fallback.
 - Decode: unchanged (npu_budget=0 at M=1 -> all cpu_expert -> CPU NF4). No IOVA.
 - Prefetch double-buffer (overlap L+1 inflate with L compute): the existing (N_cores+1)-slice design
   (Exp-2026-06-25 addendum) — add AFTER the streamer works, measured-gated.
CAUTION: the extraction touches the SHARED dense loader (every model, make-test). Do it behavior-preserving
+ bit-exact make test BEFORE wiring experts. Wedge-prone: reboot for clean NPU, SIGTERM not SIGKILL, attended.
BAR: prefill approach/beat CPU 21.3 t/s; decode = CPU 8.69 (already). NF4 quality needs F16-source experts.

## IMPLEMENTED (2026-08, compiles clean on board) — get_hot wired to existing multi-domain residence  [REVERTED — wrong mechanism per user; see LOCKED DESIGN above]
ggml-ork.cpp get_hot (~3358) — 3 edits, all EXISTING infra, no new primitives, no duplication:
 1. LOAD: ork_weight_domain(K*N, ork_layer_of) + set_pack_domain + ZERO-COPY import
    (ork_mm_load_i8_import / ork_mm_load_i4a8_import, ORK_NO_IMPORT->copy) + advance-on-IOVA-full loop —
    the dense wcache pattern (1381-1417). Was: single-domain COPY load (load_i8/load_i4a8), the S0 churn.
 2. LRU OFF when n_domains>1: the budget-return + evict-while is now gated `if (n_domains<=1)` (residence,
    not hot-K; single-domain LRU kept as fallback). "get_hot LRU not needed."
 3. domain_bytes[_edom] += K*N at commit so ork_weight_domain's byte-balancer advances at layer boundaries.
Decode already CPU-only by DEFAULT (split_frac=0 -> npu_budget=0 at M=1 -> all cpu_expert) — "iova prefill
only" already holds, no gate added. Contained to the off-by-default ORK_MOE_NPU path (no default regression).
Compiles: cmake --build build --target ork_bench -> Built target ggml-ork, EXIT=0.
NEXT (S4 validation, wedge-prone board run): ORK_QUANT=4 ORK_MOE_NPU=1 ORK_PROFILE=1 on the mixed gguf
(experts-in-orkpack) -> confirm moe_pack churn collapses (was 98%), no domain-advance crash, A/B vs CPU
21.3 prefill/8.69 decode. Needs board reboot (clean NPU) + attended. NF4 quality needs F16-source experts.

## CORRECTION (2026-08): the board HARD-WEDGE was likely a SELF-INFLICTED parallel-run, NOT native-W4A4.
Two llama-perplexity runs (full 3-arm ppl_arms.log + short ppl_short.log) were launched without confirming
the first's processes EXITED (only that the log said "done") — a timed-out/stuck native-W4A4 arm holding the
NPU + a second run starting = TWO concurrent NPU users = single-stream IOMMU violation -> hard wedge (SSH
down, needs plug power-cycle). Do NOT record "native W4A4 hard-wedges the board" — CONFOUNDED. What IS real:
native W4A4 is ~23x slower dense-prefill (the PPL arms TIMED OUT). PROCESS FIX: strictly serialize board NPU
runs — one at a time, gate each launch on `pgrep -f llama-* / ork_bench` being CLEAN (not just log "done"),
SIGTERM + verify death before the next. PPL for W4A4 weight quality should be measured in W4A8 (fast, safe),
never a long native-W4A4 run. Uniform+Hadamard PPL still UNMEASURED (both attempts timed out / wedged).

## DELIVERABLE (user, 2026-08): ~15GB full-model NATIVE-int4 orkpack for the 35B MoE, GPTQ-applied,
## ALL weights (attention + every expert) on the int4/W4A4 path — experts NOT leaking to the GGUF/CPU path.

MEASURED (wiki_tiny, 35B mixed-gguf, HYBRID = ork-int4 attention + llama.cpp-Kquant experts on CPU):
  NF4  (dense W4A8, experts Q4_K/Q6_K CPU) = 7.87    uniform+Had (dense W4A4, experts CPU) = 8.83
  ^ CONFLATED: only ATTENTION (f16 in gguf) took the ork int4 path; experts (ffn_*_exps = Q4_K/Q6_K)
    stayed on the llama.cpp GGUF/CPU path. NOT an all-weights-W4A4 number. Gguf tensor types confirmed
    via llama-gguf: attn_qkv=f16, ffn_gate/up_exps=q4_K, ffn_down_exps=q6_K, output=q6_K, token_embd=q4_K.

THE EXACT GAP (ggml-ork.cpp:1441 comment): "falls to CPU. Enable end-to-end once ork_persist_write_experts
emits native DT_I4." Scaffolding that ALREADY EXISTS: mul_mat_group_i4 (W4A4 expert compute, 2388), multi-
domain residence (ork_weight_domain/domain_advance, 2419-2425), ork_persist_write_experts (1211, iterates
ALL experts regardless of routing) — but it emits the O4N1/int8-inflate (W4A8) tier, NOT native DT_I4.

SCOPE CORRECTION (verified 2026-08): the dispatcher (ggml-ork.cpp:6275-6276) routes GGML_OP_MUL_MAT_ID to
ONLY ggml_backend_ork_mul_mat_id_i8 — there is NO W4A4 routed-expert compute path. mul_mat_group_i4 (2388)
is grouped-DENSE (qkv), NOT the _exps experts. So all-weights-W4A4 requires BUILDING the W4A4 routed-expert
path, not just a persist branch:
  (1) NEW mul_mat_id_i4 compute handler: route tokens->experts, per-active-expert int4+Hadamard pack
      (resident, multi-domain), ork_mm_run_i4, per-channel dequant. (the hard part: MoE routing + int4 +
      residence + WEDGE-SAFETY on ~15GB across 8 domains.)
  (2) native-DT_I4 expert persist (below).
  (3) dispatcher: route MUL_MAT_ID -> i4 when qbits==4 && hadamard.
  (4) 15GB 8-domain residence validation (wedge-prone; single-stream, SIGTERM, monitored).
This is the bulk of task #54 and is a MULTI-DAY build, THEN Stage-2 GPTQ on top. Do not understate it.

## USER DECISION (2026-08): GO STRAIGHT AT THE FULL W4A4 EXPERT PATH. Experts MUST be W4A4 (int4 activations),
## not int4-weights-only. Committed to the multi-day build. Execution stages (each board-validated, single-stream):

STAGE A — mul_mat_id_i4 (NEW W4A4 routed-expert compute handler). Mirror ggml_backend_ork_mul_mat_id_i8
  (ggml-ork.cpp:3162) for the ROUTING/structure (src0 experts[K,N,n_expert], src1 input[K,1,n_tokens],
  ids[n_used,n_tokens]; persist_mode==2 CPU-convert branch 3181-3210 stays as-is; token bucketing by expert
  3231-3236; hot-partition/all-active/hot_budget residency 3246-3267; cold-expert CPU vec_dot fallback
  3269+). REPLACE the int8 pack/run with the int4+Hadamard pack/run from mul_mat_group_i4 (2388-2447):
  per active expert e -> FWHT-rotate each of N weight cols (block b=K&-K), per-channel mx/7 int4 clamp[-8,7]
  into bi[k*N+n], ork_weight_domain()/ork_npu_set_pack_domain()/ork_mm_pack_i4() with domain_advance spill,
  resident in wcache keyed by expert host-ptr (is_expert=true); rotate+int4-quant the token activations ONCE
  (2436-2445); ork_mm_run_i4 per expert (M=routed rows, padded 32); per-channel fp32 dequant scatter to dst.
  Wedge-safety: reuse the mul_mat_group_i4 domain spill + wcache_evict_experts; NEVER a blocking submit.
STAGE B — native-DT_I4 expert persist: in ork_persist_write_experts (1211), add a branch for qbits==4&&hadamard
  that emits ORKPACK_DT_I4_NATIVE per expert (serial ork_mm_pack_i4 + ork_w_dump per slice, mirror the dense
  ork_persist_write_i4native 1124 + dense native quantize 2155-2176), NOT the tier==4 O4N1 path (1256). Keep
  the CPU-convert compute (3186) so convert stays off the flaky MoE submit.
STAGE C — dispatcher: at 6275 route GGML_OP_MUL_MAT_ID -> mul_mat_id_i4 when (qbits==4 && hadamard), else _i8.
STAGE D — build the ~15GB orkpack (ORK_QUANT=4 ORK_HADAMARD=1 ORK_PERSIST=<path> on 35B, slow forward,
  single-stream, SIGTERM timeout, monitored). Verify 8-domain residence holds (no wedge) + correctness vs CPU
  reference on a few tokens -> RTN all-weights-W4A4 PPL (wiki_tiny) vs the 7.87/8.83 hybrids.
STAGE E — GPTQ (Stage 2): offline on 239 (M5 Max 128GB, 35B safetensors). Per-expert Hessians (route calib
  tokens to experts) + attention Hessians -> ork_gptq_i4 -> int4 codes -> bake into Stage-B emit (replace RTN
  rounding) -> GPTQ all-weights-W4A4 PPL. quant_sig ORK_GPTQ bit already added.
RECOMMENDATION: implement STAGE A in a FRESH focused session (300-line intricate NPU handler; do NOT write it
at the tail of a long context). This WIP is the execution recipe. Board = single-stream, SIGTERM, pgrep -f.

## IMPLEMENTATION STATUS (2026-08, done directly per user "no agents"):
DESIGN REVISION (user steer): do NOT clone mul_mat_id_i8 into a separate mul_mat_id_i4. The routing/bucketing/
scatter is precision-agnostic; ONLY the inner quant/pack/run differs. So EXTEND the int8 handler with an early
native-W4A4 branch that returns before the int8 machinery (int8 path byte-unchanged). No dispatcher change needed.
- STAGE A DONE: native-W4A4 branch inside ggml_backend_ork_mul_mat_id_i8 (right after `const bool bcast=...`),
  gated qbits==4&&hadamard. Buckets tokens by expert; per expert: load native-DT_I4 from orkpack (ork_mm_load_i4,
  carries GPTQ codes) OR cold-pack (FWHT + per-channel int4 RTN + ork_mm_pack_i4); multi-domain resident
  (ork_weight_domain/domain_advance), is_expert=1; FWHT+int4-quant activations; ork_mm_run_i4; per-(row,ch)
  dequant. CPU f32 GEMV fallback (dequant from src0) if all domains full. Returns true.
- STAGE B DONE: ork_persist_write_experts has a serial qbits==4&&hadamard branch emitting ORKPACK_DT_I4_NATIVE
  per expert (FWHT + int4 + ork_mm_pack_i4 + ork_w_dump), deduped per ork_expert_key. (before the O4N1 loop.)
- STAGE C: NOT NEEDED (branch is inside the existing handler; dispatcher unchanged).
- COMPILES CLEAN on board (Built target llama-perplexity, exit 0, 2026-08).
- RUNNING NOW: first true all-weights-W4A4 correctness/PPL (ORK_QUANT=4 ORK_HADAMARD=1, wiki_tiny, cold-pack, no
  persist) -> ~/ork-outbox/w4a4_full_ppl.log. Expect a number DIFFERENT from the 8.83 hybrid (experts now int4+A4
  on NPU, not Q4_K/Q6_K-A16 on CPU). Watch for wedge (15GB residence; CPU-fallback guards overflow).
NEXT: (D) build the ~15GB orkpack with ORK_PERSIST once compute is correct; (E) GPTQ offline on 239 -> bake codes.

## VALIDATION RESULTS (2026-08, wiki_tiny, 35B):
- Run 1 (ORK_QUANT=4 ORK_HADAMARD=1, NO ORK_MOE_NPU): PPL 8.8347 — BIT-IDENTICAL to the dense-only run.
  ROOT CAUSE: supports_op (ggml-ork.cpp:7123) returns false for MUL_MAT_ID unless ORK_MOE_NPU set -> experts
  never reached my branch, ran on CPU (Q4_K/Q6_K). No code bug; need ORK_MOE_NPU=1 to route experts to ork.
- Run 2 (+ORK_MOE_NPU=1): experts NOW enter my W4A4 branch. Board did NOT wedge (up fine, clean SIGTERM) —
  the CPU-fallback + single-stream discipline held. BUT flooded:
    "[ork] domain {1..7} full — advancing residence to domain N+1"  then
    "bcreate failed to allocate weight buffer Bb[0] in pack_i4 (size=524288)" / "CREATE: Invalid argument".
  => ALL 8 domains exhaust, then pack_i4 EINVALs -> CPU fallback for the rest -> PPL again ~8.83 (experts CPU).
  Footprint accounting is CORRECT: ork_mm_pack_i4 (src/npu.c:4274) allocs Kp*Nc/2 (nibbles), ~15GB total.

CORRECTION (user caught it): footprint is ~15 GiB, NOT 30 — llama-gguf r does TWO read passes so grep -c
double-counted layers (real n_layers=40, not 80). 40 x 3 exps x 128 MiB = ~15 GiB int4 experts. FITS the
8x3GiB=24GiB window, so capacity is NOT the wall. Cold run exhaustion cause: auto-sizer else-branch (6722)
n_domains=8 + default domain_fill_cap=3GiB (=24GiB), BUT native int4 also builds a full-K Bf (formula 6685:
tile+tile), ~1.5-2x resident -> 15GiB balloons to ~22-30GiB -> overflows 24. FIX: ORK_NO_BF=1 sheds the Bf
(sizer already accounts for it) AND/OR build orkpack first then run read-mode (sizer computes real 15GiB
footprint from persist_idx -> byte-balances domains). RESIDENT 15GiB IS FEASIBLE; ran it wrong (cold + Bf).

RESOLVED (2026-08): built the 16 GiB W4A4 orkpack (30930 weights, 0 pack failures) then ran READ-mode. The
adaptive-Bf change (auto-sizer, no env gate) fired correctly: "Bf auto-shed: with-Bf 31.25 GiB > budget 22.35
-> keep 15.62 GiB Bb-only RESIDENT"; auto n_domains=7 @2.29 GiB/dom, RESIDENT (not STREAM, no EINVAL). So the
cold-run exhaustion WAS the missing footprint (cold path had no persist_idx -> couldn't size/shed). Workflow:
BUILD orkpack (ORK_MOE_NPU=1 ORK_QUANT=4 ORK_HADAMARD=1 ORK_PERSIST=<p> --chunks 1) THEN READ (same env). The
true all-weights-W4A4 PPL is computing now (w4a4_read_ppl.log). n_layers=40 (llama-gguf r double-counts -> 80).

FINAL BLOCKER (precise, 2026-08): read-mode load fails universally — "0.00 GiB resident, 182 CREATE: Invalid
argument" (NOT dom>0; domain 0 fails too). CAUSE = RAM double-alloc: read mmaps the 16 GiB orkpack with
MAP_POPULATE (prefaulted) THEN ork_mm_load_i4 (npu.c:4298) does bcreate+memcpy = ANOTHER ~16 GiB DMA -> 32 GiB
> 32 GB RAM -> MEM_CREATE EINVAL for every tile. Build worked (write mode fwrites, no competing mmap). The
i8/i4a8 read paths avoid this via ZERO-COPY dma-buf import (ork_mm_load_i8_import / ork_mm_load_i4a8_import: map
the mmap'd orkpack pages straight into IOVA, no copy) — but NATIVE int4 has ONLY ork_mm_load_i4 (bcreate), NO
import variant. FIX (bounded driver add): ork_mm_load_i4_import mirroring ork_mm_load_i8_import (import the
native-int4 Bb tile bytes zero-copy into IOVA across domains), then use it in BOTH ork_persist_load_i4native
(dense) and the Stage-A expert branch instead of ork_mm_load_i4. That removes the 2nd 16 GiB -> 15.62 GiB fits.

DONE (2026-08): ork_mm_load_i4_import added (npu.c, after ork_mm_load_i4) — bimport (dma-heap+PRIME_FD) twin of
ork_mm_load_i4, consolidated-chunk multi-domain-safe (mirrors ork_mm_load_i8_import), int4 tile Kp*Nc/2, KS=
ORK_I4_KS, no Bf. Decl in ork_npu.h. Wired import-first (bcreate fallback) at BOTH ork_persist_load_i4native
(dense) and the Stage-A expert mk() lambda. Compiles+links clean (npu.c + ggml-ork.cpp both recompile).
RESULT: CREATE: Invalid fails dropped 182 -> ~14; the import LOADS weights resident (was 0.00 GiB before).

NEXT BLOCKER (new, 2026-08): read-mode now FAILS decode ret=-3 after loading only ~2.97 GiB (domain 0=2.95,
dom1=0.02). Root line: "[ork] ERROR: mc_ensure failed to allocate mtk_all task buffer (IOMMU full?)" ->
graph_compute -1 -> decode -3. I.e. the RUN's scratch/task buffer (mtk_all, multi-core) can't allocate because
domain 0 is packed to 2.95 GiB (past the 2.29 cap, at the ~2.9 GiB hard edge) with WEIGHTS, leaving no IOVA
headroom for run scratch. This is weight-IOVA vs run-scratch CONTENTION in the multi-domain layout. Note read-
mode-with-experts-on-NPU has NEVER completed (before import it flooded CREATE + was SIGTERM'd; the 8.83 was the
NO-ORK_MOE_NPU experts-on-CPU run). LEADS: (1) reserve run-scratch headroom per domain (lower domain_fill_cap
so each domain leaves room for mtk_all), or (2) allocate mtk_all/output in a dedicated spare domain (ork_domain
already has a "last (spare) domain" notion at npu.c:588 for non-layer allocs — ensure the run scratch uses it),
or (3) mc_ensure should advance/spill to a domain with headroom instead of failing. Board healthy throughout
(rc=0, no wedge). SESSION VERY LONG -> continue this with fresh context; WIP has full state.

TRUE ROOT CAUSE (traced 2026-08): decode -3 = mc_ensure (npu.c:4541) allocates the run task buffer
  c->mtk_all = bcreate(sizeof(rknpu_task)*ORK_MAXCORE, 0x40b, c->dom_active)  — TINY (~KB), but via bcreate
  (MEM_CREATE), which returns EINVAL in a domain already holding bimported weights / any non-0 domain. This is
  the SAME allocator asymmetry the weight fix exploited: bimport (dma-heap+PRIME_FD) works across domains,
  bcreate (MEM_CREATE) does NOT. The weight IMPORT fix (ork_mm_load_i4_import) made weights load resident, but
  the RUN SCRATCH still uses bcreate -> fails the moment a run's weight lives in a filled/non-0 domain.
  Evidence: headroom fix (10 dom @1.62) still failed at domain 0=2.20 GiB (tiny alloc, not "full") — it's
  bcreate-in-a-bimport-filled-domain, not capacity.
NEXT FIX (real, not a tweak): the per-run scratch must not use bcreate in a weight domain. Options:
  (A) bimport the run scratch (mtk_all + task/regcmd/Cc/Af/mrc/mtk/maf/mcc as needed) into dom_active like the
      weights — the consistent multi-domain-safe allocator; OR
  (B) pre-allocate the (tiny, persistent, reused) mtk_all + friends ONCE in domain 0 at ork_npu_init BEFORE any
      weights load (domain 0 empty -> bcreate succeeds), IF the submit ABI allows the task buffer to live in a
      different domain than the weight (needs checking — task buffer is submit metadata, may be domain-agnostic);
  (C) reserve a dedicated EMPTY scratch domain (the auto-sizer already has a "spare last domain" notion) and
      allocate all run scratch there via bimport.
  Recommend (A) or (C) — safest/general. This is a driver run-path change touching the shared scratch alloc;
  do NOT blind-hack at long-context tail (wedge risk). Verify vs int8 multi-domain (does int8 hit this too, or
  does its scratch stay in dom 0?). SESSION VERY LONG -> continue fresh; state fully captured here.

COMPLETE MODEL + OVERLOAD PLAN (user: "overload for int4, don't touch shared int8"; 2026-08):
Scratch is PER-DOMAIN: dom_activate (npu.c:672) parks/restores each domain's scratch set (dom_save[]) and, on a
domain's FIRST touch, bcreate's regcmd(2MB)/task(512KB)/Af(256KB) IN THAT DOMAIN (line 700). mc_ensure (4538)
lazily bcreate's mtk_all(~KB)/mrc(64KB)/mtk(64KB)/maf(256KB) in c->dom_active. Scratch is SMALL (~3.5 MB/domain)
and there IS headroom (domain ~1.6 GiB weights, ~2.9 hard) — but bcreate (MEM_CREATE GEM) returns EINVAL in a
domain already holding bimport (PRIME_FD) weights: MEM_CREATE and PRIME imports don't coexist in one domain. So
for int4 RESIDENT (weights bimported), EVERY domain's scratch bcreate fails -> mc_ensure -3.
OVERLOAD (int4-only, int8 untouched): at BOTH sites (dom_activate:700 first-touch regcmd/task/Af AND mc_ensure
mtk_all/mrc/mtk/maf), when c->last_dt==DT_I4 (or a resident-int4 flag), allocate scratch via bimport instead of
bcreate so it coexists with the bimported weights in-domain. SUBTLETY TO VERIFY (do NOT blind-swap): the task
buffers use flag 0x40b (vs 0x403 data); bimport(fd,size,dom) takes NO flag arg — confirm bimport's buffer
semantics are valid for a task/descriptor buffer (coherency), else the submit reads a bad task -> wedge. Suggest
a tiny board probe (bimport a task buffer, run one int4 matmul in a non-0 domain filled with a bimported weight)
BEFORE wiring both sites. int8 path (bcreate, domain-0 or its own scenario) stays byte-unchanged.
STATE: everything up to here fixed + committed (import weights, adaptive Bf, headroom). This last piece is a
careful 2-site driver overload with a flag-semantics probe — the responsible next increment, NOT a blind edit.

## PIVOT 2026-08-08: bimport scratch is FULLY DEAD -> switch the whole int4 expert arena to BCREATE (all-bcreate).
Second scratch test (bcreate task buffers + bimport DATA buffers) ALSO WEDGED (test_i4_domains D<l) => bimport
scratch of ANY kind (task OR NPU-read data/regcmd) wedges, not just task buffers. Scratch MUST be bcreate.
bcreate can't coexist with ~GiB of bimports in a domain (EINVAL). RESOLUTION: make the EXPERT ARENA bcreate too
(ork_mm_load_i4_arena now bcreate, not bimport) -> domains are ALL-BCREATE (weights + scratch) = the int8-proven
multi-domain model, ZERO coexistence problem. Justified: probe PROVED bcreate int4 non-0 domain is bit-exact
(run_pack dom=4), and native/bcreate weights are IMMUNE to the imported-SG first-touch corruption (dom_anchor
comment) — so the summary's "int4 must be imported" was stale/wrong. bscratch reverted to always-bcreate;
scratch_import unused; init mc_ensure pre-alloc removed (not needed with all-bcreate); bimport_f/KERNEL_MAPPING
refactor left in (harmless, weights/import wrapper still use it). IOVA check: ~5GiB / ~1900MiB-cap spread =>
~3-4 domains x ~2GiB reserved < 3900MiB/domain ceiling; OK. NEXT: probe run_arena (bcreate) must stay bit-exact
+ NO wedge, then llama rebuild + 35B PPL. Board WEDGED from the 2nd scratch test -> power-cycle needed.

## SCRATCH FIX 2026-08-08: bimport-scratch is DEAD even with KERNEL_MAPPING; task buffers MUST be bcreate-early.
Tried: bimport_f(memflags) carrying RKNPU_MEM_KERNEL_MAPPING (0x8) for the kernel-read task buffers (the old
bimport-scratch dropped flags to 0 -> malformed task). RESULT: STILL WEDGES (test_i4_domains D<l). => the
rknpu import path does NOT kernel-vmap a foreign dma-buf, so an imported task buffer is unreadable by the
kernel regardless of the flag. bimport-scratch for kernel-read (0x40b) buffers is PERMANENTLY DEAD.
CORRECT SPLIT (bscratch now): kernel-read task/descriptor buffers (flags & KERNEL_MAPPING: mtk_all, mtk[], task)
=> bcreate ALWAYS. NPU-DMA'd data buffers (0x403: regcmd/mrc, maf, mcc, Af, output) => bimport when
scratch_import (coexist with weight imports). BUT bcreate of a 0x40b buffer STILL fails in an import-heavy
domain -> the remaining piece: PRE-ALLOCATE the kernel-read task buffers per-domain WHILE THE DOMAIN IS LIGHT
(at ork_dom_prime first-touch, before/with the anchor bcreate), at a GENEROUS fixed size so mc_ensure never
needs to grow (=re-bcreate) them in a now-heavy domain. c->task already alloc'd at dom_prime; ALSO pre-alloc
mtk_all + per-core mtk[] there. (Data buffers mrc/maf/mcc grow freely — they bimport, coexist.)
ALTERNATIVE if pre-alloc is fiddly: allocate task buffers in DOMAIN 0 always (kernel reads via vaddr, may not
need the submit's iommu domain) — cleaner but unverified that task_obj can differ from submit iommu_domain_id.
Board WEDGED (D<l) from this test -> power-cycle needed.

## 35B ARENA RUN 2026-08-08: arena FIXED the wedge; exposed the SCRATCH-COEXISTENCE wall (summary's blocker).
Arena worked: trace shows op=run_i4_bchain_db (accurate telemetry) imp=20 (was 9345!) — consolidation
confirmed at real scale. NEW failure (clean -1/-3, NO wedge, board fine): mc_ensure bcreate of mtk_all
fails in domain 0 after ~1378 MiB of bimported arena chunks piled there. NOT the IOVA ceiling (3900 MiB,
domain 0 only 1378) => a genuine kernel bcreate EINVAL: a fresh MEM_CREATE (bcreate) can't be placed in a
domain already holding ~GiB of imported SG mappings (IOVA fragmentation). bscratch is bcreate-only (bimport
task buffers WEDGE — malformed task). Probe run_arena PASSED because 60 tiny experts = 15 MB in the domain
(under the coexistence threshold); the 35B hits 1378 MiB. TWO contributing issues:
 1. SCRATCH COEXISTENCE (the wall): bcreate run-scratch fails once a domain has ~GiB of bimports.
    FIX CANDIDATES: (a) pre-allocate the run scratch (mtk_all + mrc/mtk/maf/mcc) per-domain at dom_prime
    time — BEFORE any import fills the domain (like the anchor bcreate) — so bcreate gets a clean IOVA slot;
    mc_ensure then reuses it. (b) revisit bimport-scratch (understand the malformed-task wedge). (a) preferred.
 2. EXPERTS PILE INTO DOMAIN 0 (not spreading): dom[0]=1378, dom[1..15]~0. byte-balanced advance triggers at
    ~1900 MiB cap; arena's 256MB chunk-rounding also wastes IOVA. Spreading experts across all 16 domains
    (advance sooner / by import-count — the user's request/assign guidance) keeps each domain light => scratch
    fits AND fewer chunks/domain. Likely need BOTH (1) and (2).
REPRO for the scratch wall (cheap): probe — bimport ~6x256MB into one domain, then bcreate a small buf; expect
EINVAL. Validate fix (a) there before the 35B.

## FIX IMPLEMENTED + PROBE-VALIDATED 2026-08-08: per-domain bimport ARENA (consolidation, user's chosen fix).
ork_mm_load_i4_arena (npu.c, new; int4-only, int8/fp16 untouched): weight tiles are base+offset VIEWS into a
PERSISTENT per-domain bimport arena (few large ~256MB dma-buf chunks shared across MANY experts) instead of
one dma-buf/expert. Same anchor+bimport+bimport-scratch alloc pattern (lowest risk), just coarser. Weight
owns nothing (owns=0, own_bufs=NULL -> ork_mm_free skips); chunks freed once in ork_npu_free. Struct fields:
i4arena/i4arena_n/i4arena_cap + i4arena_cur[64]/i4arena_off[64]. ggml-ork.cpp mk() routes pe-loads to it
(ORK_I4_NO_ARENA falls back to per-weight import). ORK_I4_ARENA_MB tunes chunk size (default 256).
PROBE VALIDATION (test_i4_domains, run_arena): 60 distinct K=512 N=2048 experts arena-loaded into ONE domain
=> ALL bit-exact (worst_maxerr=0), sharing ~1 chunk (60x256KB=15MB < 256MB) instead of 60 imports. Mechanism
confirmed. Also re-confirmed: pack + per-weight-import M=96 K=512 N=2048 in non-0 domain = BCHAIN bit-exact.
NEXT: rebuild board llama.cpp with updated ork-driver + ggml-ork, re-run 35B PPL. Expect ~20 arena chunks
total (5GiB/256MB) vs 9345 imports => ~5 mappings/domain (was ~2340) => no wedge. Then read the PPL number.

## ROOT CAUSE (STRONG, 2026-08-08): per-domain IOMMU MAPPING-COUNT saturation from lazy per-expert imports.
Trace (scratchpad/ppl35b/presubmit.trace) shows imp (bimport count) climbing 1->903->...->9345 monotonically
while ONLY 4 domains are ever used (submit_dom in {0,1,2,3}); dom[0..3] ~1.2-1.4 GiB each at the wedge
(iova_total 5081 MiB), domains 4-15 EMPTY. Experts lazy-import one dma-buf EACH into the wcache during the
forward (ggml-ork.cpp:3314-3324, mk()->ork_mm_load_i4_import). WHY only 4 domains: auto-sizer leaves
domain_layers=0 => BYTE-BALANCED ork_weight_domain (ggml-ork.cpp:598-607) advances a domain ONLY when it
hits the BYTE cap (~1900 MiB, line 6748) at a layer boundary. The NPU WEDGES first, at ~2340 imports /
~1.3 GiB per domain — WELL under the byte cap — so the cursor never advances past 3 and 4-15 stay empty.
=> the limit is NUMBER OF SEPARATE dma-buf MAPPINGS per domain (~2340), not bytes. Byte-balanced advance is
blind to mapping count. Cheap probe validated the primitive is fine (K=512 N=2048 M=96 pack+import non-0 =>
BCHAIN bit-exact), so the wedge is emergent from mapping saturation, not shape/correctness.
FIX FORK (needs decision):
 (A) SPREAD by import-count: make ork_weight_domain also advance when a domain's import COUNT hits a safe
     cap (e.g. ~512) => 9345 experts over ~16 domains (~584/dom) under the wedge threshold. Simple; aligns
     with the user's "request/assign a domain each time one is needed" guidance. Uses all 16 domains.
 (B) CONSOLIDATE imports: pack many experts into a few large dma-bufs per domain (extend own_bufs across
     experts) => far fewer mappings/domain regardless of count. Robust; bigger change.
 (A)+(B) both reduce mappings/domain; (A) is the smaller first step. NEED a cheap probe to (1) confirm the
 ~2340/domain mapping wedge and (2) find the safe per-domain import cap, before the 35B re-run.

## CORRECTION 2026-08-08: the "experts on mc_i4" conclusion below was a STALE-LABEL ARTIFACT.
g_last_op (the ORK_PRESUBMIT_TRACE op= field) is set ONLY inside validate_regcmd (npu.c:310). BCHAIN's
worker SKIPS validate_regcmd by default (ORK_I4_VALIDATE off) => it never updates g_last_op => BCHAIN
submits inherit the last mc_i4 label. So the 26k "ork_dyn_mc_i4" submits are almost certainly BCHAIN
(the correct path), mislabelled. Confirmed by geometry: K=512 N=2048 has Sn=1,Sk=1 (both pack AND import
set Sn=(N+nmax-1)/nmax=1, Sk=1), M=Mp>=2 => run_i4 dispatches to BCHAIN at 4501, not mc_i4. tasks=96 fits
BCHAIN's per-core chained program count too. => experts ARE on the proven path; the wedge is a BCHAIN
RELIABILITY issue partway through prefill (~26k submits ~= mid-network), NOT a wrong-path routing bug.
STAGED FIXES (Mac-side, need board): (1) g_last_op now set at top of run_i4_bchain_db (accurate wedge
telemetry). (2) ORK_I4_DIAG dispatch-path print in ork_mm_run_i4 (BCHAIN/SLICE/mc_i4). (3) probe extended
with the EXACT ffn_down shape K=512 N=2048 M=96 (pack+import, non-0 domain) to reproduce the wedge cheaply.
NEXT (post power-cycle): run the cheap probe FIRST (seconds) — if K=512 N=2048 M=96 import wedges there,
fix on the probe; if it PASSES, the wedge is scale/duration-dependent => re-run 35B with the fixed telemetry
to get the ACCURATE wedging op+shape.

## 35B PPL RUN 2026-08-08 (instrumented, ORK_PRESUBMIT_TRACE) — WEDGE ROOT-CAUSE PINNED (SUPERSEDED, see CORRECTION above):
Ran 26,000+ int4 submits successfully (the multi-domain int4 forward SCALES far), then WEDGED (D<l,
in-kernel) on submit #26078: op=ork_dyn_mc_i4 K=512 N=2048 wdom=3 imported=0 core=0x2 tasks=96. The LAST
200 submits are ALL this shape (ffn_down), advancing (not looping). live=9442 bufs, imp=9345.
=> EVERY expert FFN is running through the PER-ROW mc_i4 path (ork_dyn_begin_mc_i4), NOT the proven-safe
BCHAIN (run_i4_bchain_db). mc_i4 at prefill explodes into ~26k tiny submits and eventually wedges the NPU.
WHY mc_i4 not BCHAIN: expert branch (ggml-ork.cpp:3364) calls ork_mm_run_i4(Mp) which routes M>=2 to
BCHAIN at npu.c:4501 — GATED on w->Sk==1 && w->Sn==1 && N%64==0. nmax=8192, KS=10752 => a FRESH pack of
K=512 N=2048 has Sn=1,Sk=1 (bchain-eligible). But the trace shows mc_i4 => the IMPORTED/persisted expert
weight must carry Sn>1 or Sk>1 (tasks=96 with Sn=32 => S=3 rows; consistent with N=2048/64=32 N-slices).
HYPOTHESIS: ork_persist_load_i4native / ork_mm_load_i4_import re-tiles (or the persist format stores) the
weight with Sn=N/64 (64-wide slices) instead of Sn=1 => bchain refused => wedge-prone per-row path.
NEXT (cheap, post-reboot): flip test_i4_domains to K=512 N=2048; print w->Sn,w->Sk for pack vs import;
confirm pack->bchain, import->mc_i4; then FIX so imported wide-N experts are bchain-eligible (Sn=1) OR wire
the int4 slice-and-dice rescue (#33) at the bchain-refuse site for M>=2 wide-N int4 (proven-safe tiles).
ALSO: separate recoverable mc_ensure mtk_all bcreate failure during the -fit probe pass (0 resident) -> -1.
Trace + log saved: scratchpad/ppl35b/{presubmit.trace,ppl35b.log}. Board WEDGED (D<l) -> needs power-cycle.

## PROBE RESULT 2026-08-08 (part 2 — root-cause of the M=1 import bug narrowed, PPL path CLEARED):
KEY WIN: the PREFILL/PPL primitive is PROVEN CORRECT. import M=128 BCHAIN in non-0 domains (1,2,3) is
bit-exact (maxerr=0). Perplexity is prefill-only (M = ctx chunk, M>1) -> rides BCHAIN -> the int4
multi-domain machinery works for the PPL run. The 35B PPL hang is therefore NOT a primitive-correctness
bug; it is a SCALE issue (mc_ensure run-scratch bcreate in a heavily-import-filled domain, and/or domain
count) — instrument the actual hang site on the 35B, do not re-derive the primitive.
M=1 IMPORT BUG (decode only, does NOT block PPL): import + M=1 (run_i4_mc_db/ork_dyn_begin_mc_i4) + NON-0
domain miscomputes WHOLESALE (all ~512 cols wrong, domain-variant). Ruled out: weight address, content,
domain, scratch domain (all identical to the passing pack-M=1 case per ORK_I4_DIAG), and a missing weight
bsync TO_DEVICE (ORK_I4_WBSYNC did NOT fix). import M=1 in domain 0 PASSES; pack M=1 non-0 PASSES; import
M=128 non-0 PASSES. => an IMPORTED dma-buf mapped into a NON-0 iommu domain is read wrong by the mc_i4 M=1
schedule specifically (HW read-pattern/mapping hazard). FIX (proven-correct, deferred until decode matters):
route M=1 imported weights through BCHAIN by padding M=1->M=2 and taking row 0 (BCHAIN is bit-exact here).
Diagnostics left in ork_dyn_begin_mc_i4: ORK_I4_DIAG (per-program wdom/dom_active/imported/bdma/bytes).

## PROBE RESULT 2026-08-08 (test_i4_domains, cheap + deterministic — REPLACES the deadlock theory):
FIX-B THEORY (swap races in-flight nonblock job) IS DISPROVEN: rapid A->B->A->B alternation across two
established domains PASSES bit-exact (both M=128 and M=1). The doorbell output-poll IS sufficient; the
domain swap does NOT race the in-flight job.
REAL BUG (reproduced, deterministic, seconds): IMPORTED int4 weight + M=1 (run_i4_mc_db per-row doorbell)
+ NON-0 domain => MISCOMPUTE (maxerr ~3300-3990, NOT a hang). Contrast that localizes it:
  - domain 0, any path: OK
  - non-0, PACKED (per-tile bcreate), M=1: OK      <- run_i4_mc_db is fine for packed weights
  - non-0, IMPORTED (own_bufs consolidated views), M=128 BCHAIN: OK   <- import layout is fine for BCHAIN
  - non-0, IMPORTED, M=1: FAIL                     <- ONLY this corner
=> bug is in how run_i4_mc_db (per-row) addresses IMPORTED (own_bufs base+offset view) weight tiles in a
   NON-0 domain. run_i4_bchain_db (M>1) reads the same imported tiles correctly. Diff the two on weight
   tile base/IOVA + domain handling. This is the DECODE path for resident experts (M=1) — must fix.
NOTE: the 35B perplexity hang (prefill, M>1 BCHAIN) is likely a SEPARATE issue (mc_ensure run-scratch
bcreate in a heavily-import-filled domain), since BCHAIN import in non-0 domains passes here on small weights.

## TWO ARCHITECTURE FIXES (user-directed, 2026-08) — my impl got both wrong:
FIX A — DOMAIN REQUEST/ASSIGN (not static numbering): domains are a LIMITED kernel-coordinated pool; each new
domain must be REQUESTED and ASSIGNED, not picked as dom_next++. Only the ORKD path does this correctly:
orkd_domain_alloc -> ORKD_DOM_REQ ("client requests a domain id from the daemon's coordinated pool", orkd_proto.h:66,
orkd_client.c:116). The DIRECT path ork_npu_domain_alloc just returns dom_next++ (npu.c:165 "hands out 1,2,3,…") —
it does NOT request/verify against the kernel pool. My auto-sizer picked 16 STATIC ids (cap=1000) -> likely
over-ran the real pool -> submit to an unassigned domain = silent switch-deadlock. "May already be handled outside
your code" = orkd owns request/assign; running DIRECT with a static n_domains bypasses it. FIX: either run under
orkd (proper request/assign) OR make direct-mode request+be-assigned each domain from the actual kernel pool
(and cap n_domains at the real pool size, not an arbitrary 16). Earlier runs with fewer domains (7-10) got
further precisely because they stayed within the pool.
FIX B (below) — DOORBELL-COORDINATED SWAP + DRAIN.

## ROOT CAUSE (user-directed, confirmed 2026-08): DOORBELL must COORDINATE the domain swap; it doesn't. [FIX B]
CONFIRMED: the int4 BCHAIN doorbell submit is NONBLOCK — npu.c:13869 `a->sub.flags=ork_ppflags()|0x2u` (0x2 =
RKNPU_JOB_NONBLOCK). The submit ioctl RETURNS before the HW/kernel job retires; the worker then polls the OUTPUT
cells for landing. But a domain swap is done OUTSIDE the doorbell's completion coordination — dom_activate at
run_i4_bchain_db:13897 (and my redundant one in ork_mm_run_i4) — so switching the IOMMU domain for the next op
races the PRIOR op's still-in-flight nonblock kernel job -> silent deadlock (blocked in-kernel, load 0.00, no
error). int8 multi-domain works because its submit path effectively drains before switching.
FIX (user's architecture): the DOORBELL owns the swap — it must DRAIN its own nonblock submit (wait the kernel
job to RETIRE, not just output-landed) before any domain switch. Concrete options: (a) set sub.fence_fd (currently
-1) on each nonblock doorbell submit and WAIT the fence before the next dom_activate; (b) a blocking drain submit
after the round; (c) fold dom_activate INTO the doorbell's post-landing/retire coordination so no swap happens
with a job in flight. Remove the OUTSIDE dom_activate (ork_mm_run_i4) — swap belongs to the doorbell only.
VALIDATE CHEAP: extend examples/test_i4_domains.c to rapidly ALTERNATE nonblock int4 submits across 2+ established
domains in a loop (reproduce the swap-races-in-flight deadlock in seconds), then confirm the drain-before-swap fix.
(no gdb on board; needs the probe + maybe fence-wait instrumentation to implement correctly — a focused effort.)

## CURRENT BLOCKER (2026-08): SILENT DEADLOCK in the int4 multi-domain FORWARD (loads clean, hangs chunk 1).
Full fix chain landed + committed (all real): over-alloc chunk-cap, dom_activate in ork_mm_run_i4, import-only
experts, close-fd SEAL (load_i4_import) -> no EMFILE, --no-warmup -> no M=1 warmup stall, domain fill cap 1000MiB
(~16 domains) -> run-scratch bcreate coexists (no mc_ensure EINVAL). RESULT: the 35B W4A4 now LOADS the full
15.62 GiB across 16 domains with EVERY failure counter 0 (EMFILE/SUBMIT/CREATE/mc_ensure all 0), reaches
"calculating perplexity over 8 chunks", then HANGS in chunk 1's forward at load=0.00 — blocked, not spinning, NO
error, no recover-loop-exhaustion. Hard silent deadlock. (No gdb on board; /proc/stack empty.)
LEADING HYPOTHESIS (the earlier-dismissed one, now most likely): the FORWARD rapidly dom_activate-SWITCHES domains
(16 domains, ~40 layers x many ops -> hundreds of switches/chunk), and the int4 doorbell is NONBLOCK — switching
to domain B while domain A's nonblock submit hasn't retired deadlocks the kernel IOMMU switch (or a lock). int8
multi-domain works likely because it quiesces (blocking) before switching. FIX DIRECTION: QUIESCE (wait NPU idle /
drain the in-flight nonblock doorbell) BEFORE dom_activate on the int4 path. Validate CHEAPLY: extend
examples/test_i4_domains.c to ALTERNATE submits across 2+ ESTABLISHED domains rapidly in a loop (reproduce the
switch-while-in-flight deadlock in seconds), then confirm quiesce-before-switch fixes it — NOT more 30-min 35B runs.
OPS: the hung run needs a reboot to clear (SIGTERM ineffective). Board boots NVMe (SD=bootloader-only). Probe runs
that stall the M=1 path dirty the NPU -> reboot between probe runs.

## PROBE RESULT (2026-08, examples/test_i4_domains.c) — "kernel can't swap domains" was WRONG (probe artifact).
Clean run on a FRESH boot, domains ESTABLISHED via ork_npu_activate_domain (= the ork_dom_prime native anchor
the working int8/import path uses):
  DOMAIN 0: pack/import x M=128/M=1 all OK.
  DOMAIN 1 (established): [pack] M=128 OK, [pack] M=1 OK, [import] M=128 OK  <-- int4 BCHAIN (M>1 PREFILL)
    doorbell WORKS in a NON-0 domain, pack AND import, bit-exact. So multi-domain int4 PREFILL is fine.
  DOMAIN 1 [import] M=1: STUCK "ork_dyn_end doorbell miss, hw_elapse=0, op#0" -> the M=1 PER-ROW doorbell
    misses in a non-0 domain. This is the ONE real bug, and it's DECODE-only (M=1); prefill is M>1 bchain.
PROBE BUGS I had (NOT driver bugs): (a) first probe never established the domain (no ork_npu_activate_domain)
  -> switch-to-unestablished-domain timeout (my false "kernel bug"); (b) used ork_w_free (doesn't reclaim IOVA)
  -> domains 2,3 bcreate-failed from the leak (now ork_mm_free).
OPS: the M=1 non-0-domain stall DIRTIES the NPU (switch-timeout cascade) -> REBOOT between probe runs, else the
  next run inherits a wedged IOMMU (domain-0 CREATE fails + segfault). Board boots NVMe (~1min, fsck sometimes 5).
IMPLICATION for the 35B: op#0 stall was "run_i4_bchain_db" (M>1) — but the probe shows bchain works in dom 1.
  So re-examine: (1) the llama-perplexity WARMUP is an M=1 empty run -> hits the M=1 non-0-domain doorbell-miss
  FIRST (wedges before prefill)?; (2) or a domain >1 / not-established-in-the-35B-flow. NEXT: fix the M=1 per-row
  int4 doorbell in non-0 domains (ork_dyn_end / run_i4_mc_db) to mirror how bchain establishes+submits per-domain;
  OR route M=1 decode to CPU during the warmup so it never M=1-submits int4 in a non-0 domain.

## (superseded framing) CURRENT STATE (2026-08) — bug chain fixed; CORE REMAINING = int4 DOORBELL in non-0 domains.
Fixed this session (all in tree): (1) ork_mm_load_i4_import (bimport, multi-domain weights); (2) chunk cap to
weight need (was 16MiB/expert -> IOVA blowup); (3) bscratch reverted to bcreate (int8-style, correct); (4) dense
ork_persist_load_i4native + Stage-A expert use ork_weight_domain/domain_advance + accounting; (5) dom_activate at
top of ork_mm_run_i4; (6) import-only experts (no bcreate fallback -> killed the infinite self-heal loop on
bcreate'd experts); (7) close-fd-after-load SEAL in load_i4_import (heap_fd closed post-sync -> ~0 held fds, no
ulimit dependency; per-weight so peak stays low; verified dmabuf_sync is load-only + bdestroy MEM_DESTROYs via
handle). fd ulimit workaround (131072) also in the run script (now redundant with #7).
RESULT PROGRESSION: 0-resident/flood -> 12.6GiB resident (self-heal loop on bcreate'd experts) -> (import-only) ->
decode -3 "run_i4_bchain_db incomplete (recover exhausted) hw_elapse=0 STUCK op#0" = the int4 BCHAIN DOORBELL
submit does NOT land in a NON-0 domain. Surfaced on op#0 (dense) once dense started distributing (#4). Experts
(always non-0) hit the same. dom_activate (#5) is necessary but NOT sufficient — the doorbell submit + the
self-heal RESET path must also target the weight's domain.
CORE REMAINING WORK (focused int4-doorbell effort, not more blind full-model runs): make run_i4_bchain_db +
run_i4_mc_db + ork_dyn_* doorbell submit + the self-heal reset ALL honor the weight's iommu domain (submit
iommu_domain_id = weight domain; re-activate after RKNPU_ACT_RESET which clobbers dom state). Validate with a
SMALL isolated int4 multi-domain probe (2 domains, 1 weight each) rather than the 35B (30-min cycles + un-killable
runs forcing reboots). Consider pinning dense to domain 0 (fits 1.65GiB, bchain proven there) so only the expert
path needs the non-0-domain doorbell fix.
OPS NOTE: these runs resist SIGTERM (sudo+timeout wrapper eats it; binary stuck in doorbell recover) -> reboots.
Board boots NVMe (SD=bootloader-only now); a reboot takes ~1min (+fsck sometimes ~5min).

*** RETRACTED: "FUNDAMENTAL WALL" WAS WRONG (2026-08). *** User: "we use domains for >4gb, look at the code."
Correct: int8 already does >4GiB across domains with bcreate scratch — no wall. The int4 failures were a BUG I
introduced: ork_mm_load_i4_import used the CONSOLIDATED-CHUNK import (16 MiB chunk_cap) but is called PER small
per-expert weight (~0.5 MiB, single tile) -> each expert grabbed a whole 16 MiB chunk -> ~16 MiB IOVA burned per
expert x ~15k experts -> domains overfilled instantly (domain 0 to 2.2 GiB on a few weights) -> the downstream
bcreate scratch "failures" and the whole cascade. NOT a bcreate-vs-import coexistence problem.
FIXES APPLIED (2026-08, Mac-side, awaiting board power-cycle to test):
 (1) ork_mm_load_i4_import (npu.c): cap the chunk to the weight's remaining need (csz=min(chunk_cap, need-boff),
     >=ts) so a small expert allocates ~its size, not 16 MiB. THE key fix.
 (2) bscratch (npu.c): REVERTED to always bcreate (the scratch-bimport was the wrong fix + WEDGED on bimport'd
     0x40b task buffers). With (1), domains have room and bcreate scratch works exactly like int8's multi-domain.
 (3) ork_persist_load_i4native (ggml-ork.cpp): dense int4 now goes through ork_weight_domain + domain_advance +
     domain_bytes accounting (it bypassed them before -> experts over-admitted to domain 0). Mirrors int8 resolve.
NEXT: user power-cycles board -> compile on board -> rerun read (ORK_MOE_NPU=1 + w4a4 orkpack) -> expect resident
15.62 GiB across domains + a real all-weights-W4A4 PPL. scratch_import field left defined-but-unused (harmless).

(obsolete) FUNDAMENTAL WALL HIT (2026-08, board WEDGED): pressed on with the scratch overload. Fixed the gate timing via
c->scratch_import (set in ork_mm_load_i4_import; bscratch bimports when set — timing-independent). This made the
run scratch bimport... and the board HARD-WEDGED (ping fail) during the first int4 submit. CAUSE: bimport'd TASK
DESCRIPTOR buffers (mtk_all/mtk, flag 0x40b) — bimport uses flags=0, producing a malformed task the NPU executes
-> wedge. This was the flagged risk, now confirmed empirically.
=> THE REAL CONFLICT (RK3588 IOMMU): weights >4GiB need PRIME-import (bimport); the run's TASK/descriptor scratch
needs MEM_CREATE (bcreate, flag 0x40b) semantics; and a submit needs BOTH in the SAME iommu domain. But bcreate
EINVALs in a bimport-filled domain (graceful) AND bimport'd task buffers wedge (fatal). So you cannot have
>4GiB of imported weights + a valid task buffer co-resident in one domain for an NPU submit. THIS IS THE SAME
WALL as memory moe-ork-dense-only-experts-cpu-floored (experts CPU-floored by the 4GiB IOVA). All-experts-W4A4-
RESIDENT-on-NPU is blocked by this HW/driver constraint, not a tuning issue.
POSSIBLE REAL FIXES (all deeper): (1) a MEM_CREATE variant that coexists with PRIME imports in a domain (kernel/
driver); (2) keep task/regcmd scratch in a small RESERVED bcreate-only region per domain that imports never touch
(needs the IOVA allocator to partition create-vs-import space); (3) cross-domain submit (scratch in dom0, weights
in domN) if the HW/kernel allows — likely not. Until then: experts run W4A4-ON-CPU (correct quantization, slow)
or W4A8-inflate on NPU — the resident-NPU-W4A4-experts path is HW-walled.
CODE STATE: the bscratch/scratch_import changes (npu.c) WILL RE-WEDGE if an int4-resident run is repeated. Do NOT
re-run ORK_MOE_NPU=1 + the w4a4 orkpack until the task buffers are moved OFF bimport. Consider reverting bscratch
(keep as experimental branch) so the clean CPU-fallback (completes, ~8.83 hybrid) returns. ork_mm_load_i4_import
+ adaptive-Bf + headroom are GOOD keepers; the scratch-bimport is the wedging part.
BOARD: hard-wedged post-run; HA MCP unreachable from the agent env -> USER must power-cycle Rock 5B Plug.

(superseded) THE REAL BLOCKER (core of #54): 8 domains x ~4GiB = 32GiB nominal, but ~15GB of int4 experts EXHAUSTS them
(all 8 go "full") and then bcreate returns EINVAL (not ENOMEM). EINVAL smells like a driver/IOMMU limit, not
plain capacity (cf. memory [[mfold-0x1080-wedge]], #31 Bb[4] bcreate failure). NEXT LEADS: (1) what is the
effective per-domain IOVA cap the auto-sizer uses vs real 4GiB; (2) why 15GB fills 32GB nominal — is each
expert packed once (wcache key = src0->data+e*nb2, should be unique) or is there duplication/no-evict growth;
(3) is there a TOTAL-IOVA limit across domains (rknpu) < 8*4GiB; (4) instrument domain_bytes[] at exhaustion.
COMPUTE PATH IS DONE + WEDGE-SAFE; the residence/IOVA is the remaining hard problem. GPTQ (E) is blocked on it.

## (superseded framing below — kept for reference)
STAGE 1 (RTN, on-device milestone): make ork_persist_write_experts (1211) emit NATIVE DT_I4 — mirror the
dense ork_persist_write_i4native (1124) + the dense native quantize (2155-2176: FWHT-rotate col, per-channel
mx/7 scale, int4 clamp, ork_mm_pack_i4, dump). Wire experts proc_prec=1 in ork_resolve_weight_i8 (1374).
Build the ~15GB orkpack (ORK_QUANT=4 ORK_HADAMARD=1 + expert-int4 route + ORK_PERSIST=write, slow forward).
Verify 8-domain residence holds (no wedge). -> FIRST true all-weights-W4A4 PPL (RTN) vs 7.87 hybrid-NF4.

STAGE 2 (GPTQ): offline on 239 (M5 Max 128GB, has 35B safetensors). Per-tensor Hessians (attention, sees all
tokens) + per-expert Hessians (MoE routing). ork_gptq_i4 -> int4 codes. Bake into the orkpack quantize step
(replace RTN at 2166 dense / 2414 group / the expert emit). -> GPTQ all-weights-W4A4 PPL.

## GPTQ → W4A4 INTEGRATION (concrete, 2026-08). GPTQ IS the W4A4 quantizer, not a detour.
Goal: GPTQ uniform-int4 + Hadamard -> native DT_I4 (W4A4) pack -> W4A4 PPL vs NF4's 9.07.
GPTQ's value is W4A4-specific: its Hessian error-feedback fixes the int4 ROUNDING that W4A4 is
sensitive to; W4A8 is already robust so GPTQ barely moves it -> W4A8 PPL is deliberately SKIPPED.

INJECTION POINT (clean, no driver/tiler change): ggml-ork.cpp:2167-2173, the native-W4A4 quantize.
  Today: per-channel absmax RTN -> `bi[]` int4 codes + `ow.bscale[n]=mx/7`, then ork_mm_pack_i4(K,N,bi).
  `ork_mm_pack_i4(ctx,K,N,const int8_t*B)` ALREADY takes pre-quantized int4 codes ([-8,7]) — exactly
  ork_gptq_i4's output. So the swap is LOCAL:
    - GPTQ group=K -> ONE scale per output channel == the native format's per-channel bscale[n]. (match)
    - replace the RTN loop with ork_gptq_i4(K,N,W_rowmajor,H,/*group=*/K,codes,scales,damp);
      lay codes into `bi` (k*N+n order) + set ow.bscale[n]=scales[n]; then ork_mm_pack_i4 unchanged.
  Gate the whole swap on getenv("ORK_GPTQ") (already in ork_build_sig bit<<10, 2026-08).

THE ONE BOARD/MODEL-GATED DEPENDENCY: the per-weight calibration Hessian H[K*K] = X^T X.
  Needs a CALIBRATION FORWARD (real activations) -> a TWO-PASS build (the current pack is lazy
  first-touch; GPTQ needs full H BEFORE packing a weight). Design:
    pass 1 (persist_mode==2 + ORK_GPTQ): at each MUL_MAT accumulate H[src0] += src1^T src1 over N
           calibration tokens; DEFER packing. Memory: K*K*4 per distinct weight while accumulating.
    pass 2: with H complete per weight, run the injection above -> native DT_I4 pack (NPU tiler).
  This restructure + capture is the remaining work; it can only be RUN/validated on the board+model.

STATUS: GPTQ core done+byte-exact (Mac numpy ref + board self-test 0.60x RTN). quant_sig hook DONE.
Injection point located + proven code-compatible (pack_i4 takes codes). Hessian-capture two-pass:
NOT yet written (board-gated to validate; do not blind-implement the restructure).

## DESIGN DECISION (user, 2026-08): W4A4 ⟹ ALWAYS HADAMARD, no env gate.
Native W4A4 must ALWAYS apply the block-Hadamard rotation — it makes int4 ACTIVATIONS coherent (the
DOMINANT W4A4 error; the NF4 weight-codebook is the SECONDARY term). So "native W4A4" = "Hadamard-rotated
int4" by definition, not ORK_HADAMARD opt-in. Wire: dispatch native_w4a4 -> ALWAYS mul_mat_i4_hadamard
(retire un-rotated mul_mat_i4 for shipping = diagnostic only); native-expert pack -> always rotated. Drop
ORK_HADAMARD as a knob (route by model=W4A4 / residence-can't-fit-int8), per the no-env-knobs philosophy.
HW-CHECK (rocket_registers.h, verified): NO weight value-codebook/LUT in the CNA (only sparse DCOMP + the
SDP post-MAC activation LUT) -> NF4-native-W4A4 IMPOSSIBLE, confirmed. Bonus lever: CNA_CONV_CON1 has SEPARATE
IN_PRECISION[6:4] + PROC_PRECISION[10:8] -> a HW int4-in/int8-proc upcast (uniform int4 15GB RESIDENT + int8
compute, no 30GB materialization) — a separate footprint rescue worth investigating.
PPL test (1.7B wikitext-2) IN FLIGHT: NF4(W4A8) vs uniform-W4A4 vs uniform-W4A4+Hadamard -> validates
"always Hadamard" (expect uniform-no-had WORST; uniform+had ~ NF4, since W4A4 error is activation-dominated).

## TARGET CONFIG (user, decisive): W4A4 via NF4 — prefill NPU, decode CPU NEON.
Experts stored + RESIDENT as NF4/int4 (Bi4, DT_I4), NOT inflated to int8. Consequences:
 - Footprint ~15 GB (int4), not 30 GB (int8) -> fits ~4 of 8 domains COMFORTABLY. The "30 GB int8 at
   the 32 GB edge" caveat is GONE.
 - Prefill compute = NATIVE W4A4 on the NPU via the int4 COLSPLIT/BCHAIN doorbell (#49-53, already
   default + working) — NOT the int8-inflate (route B) the MoE path uses today, and NOT the old
   wedge-prone run_i4_grouped/mfold.
 - Decode = CPU NEON via the existing NF4-dequant vec_dot (inflate_chan_nf4_i8 / ork_nf4_lut in get_hot).
   M>1 gate on MUL_MAT_ID supports_op: prefill(M>1)->NPU, decode(M=1)->CPU.
get_hot's LRU is NOT required (user):
 - DECODE = CPU: no IOVA, no domains, no get_hot at all — CPU NEON NF4 reads the mmap'd orkpack per token.
 - PREFILL = NPU: experts fit resident (~15 GB int4 < ~4 domains) so there is NOTHING TO EVICT. get_hot's
   whole job (admit hottest-K + evict under a 2.5 GB budget) IS the churn. Replace it with STATIC FULL
   RESIDENCE: load ALL experts int4-resident across domains ONCE (via ork_domain_for), keep a resident
   handle per (tensor,expert); prefill just looks up the handle + runs the int4 doorbell. librkllmrt model
   (pack once, zero churn). Drop the LRU, don't wire it to domains.
So the mul_mat_id handler: (a) first-touch -> load-all-experts-resident-across-domains (int4/DT_I4,
retain Bi4, NOT load_i4a8 int8 inflate); (b) M>1 -> grouped run on the int4 doorbell against resident
handles; (c) M=1 -> CPU NEON NF4 (existing), never touches NPU/IOVA. get_hot LRU deleted for this path.

## REVISED PLAN — wire the EXISTING infra (no new driver primitives). S1 code REVERTED.
S1's ork_mm_load_i8_block + ork_moe_block + test_moe_block were REVERTED (git checkout Makefile
include/ork_npu.h src/npu.c; rm example) — they duplicated existing consolidation + multi-domain infra.

Grounded gap (ggml-ork.cpp): the multi-domain residence machinery EXISTS and the DENSE path uses it —
  * ork_domain_for(ctx, layer, bytes) (~554) picks a domain by layer (layer/domain_layers), per-domain
    fill cap ~3 GiB; ork_domain_advance(ctx) (~601) = ork_npu_set_pack_domain to the next domain on full.
  * dense/persist paths call set_pack_domain + advance-on-full (1382-1416, 1507, 2220-2261, 2366).
But get_hot (MoE expert residence, ~3321-3381) does NOT — no ork_domain_for/set_pack_domain there. So
experts pack into the ambient domain + thrash the 2.5 GiB hot_budget LRU instead of spreading across the
8 domains. THAT is S0's "crammed domain 0 -> advance to domain 1 -> CREATE: Invalid argument" crash.

WIRING (existing infra only):
 1. In get_hot, before packing/loading an expert, call ork_domain_for(ctx, layer_of(tensor), bytes) +
    ork_npu_set_pack_domain + advance-on-IOVA-full — exactly the dense path's pattern (1382-1416).
    (layer index derivable from src0->name via ork_layer_of.)
 2. Size n_domains (auto-sizer) to cover the EXPERT footprint too (30 GB int8 -> 8 domains), not just
    dense. Today n_domains defaults 1 / sized for dense only.
 3. Residence not eviction: with experts spread across 8 domains they FIT (~30 GB < 32 GB) -> stop the
    hot_budget LRU thrash (raise/disable the per-tensor evict when multi-domain-resident). librkllmrt =
    zero per-token MEM_DESTROY; that's the target.
 4. Decode M>1 gate on the MUL_MAT_ID supports_op so decode stays CPU (unchanged from before).
Validation: same A/B vs the 21.3/8.69 CPU bar; bit-exact; watch for the per-submit-reads-ONE-domain rule
(a layer's experts must be co-located in one domain OR dom_activate per expert — ork_domain_for's
by-layer partition keeps a layer's experts together, which satisfies it).
Only if load-time MEM_CREATE count (30720 experts) proves a problem do we consider a per-domain
consolidator — and then EXTEND the existing ORK_CONSOLIDATE_I8 / own_bufs path, not a parallel new fn.

## Feasibility finding (why literal per-expert remap is the wrong mechanism)
The rknpu ioctl set is only `MEM_CREATE` / `MEM_MAP` / `MEM_DESTROY` (src/npu.c bcreate@486,
bimport@551, bimport_fd@587). **There is NO in-place PTE remap** (swap physical pages under a
fixed IOVA). So:
- `bcreate` = MEM_CREATE(fresh DRAM)+MEM_MAP+mmap → then memcpy tiled bytes. (today's per-expert `ork_mm_load_i8`)
- `bimport`/`bimport_fd` = map an existing dma-buf's pages into IOVA, NO memcpy — but STILL pays MEM_CREATE.
- Per-expert import removes only the ~tens-of-µs memcpy (experts are ~1.75 MiB), NOT the per-expert
  MEM_CREATE churn (the documented real cost, ggml-ork.cpp:3395), and risks the foreign-mapping
  chain-walk fault (src/npu.c:255 "one bimport per tile faults the chain-walk").

## The design that DOES remove the churn: per-LAYER expert-block residency
Key numbers (Qwen3.6-35B-A3B): expert gate/up [2048×512], down [512×2048] → ~1.75 MiB NF4/expert.
- One tensor's 256 experts ≈ 128 MiB; a layer's 3 proj tensors ≈ **384–448 MiB — fits 4 GB IOVA whole.**
- Orkpack stores a tensor's 256 experts CONTIGUOUSLY (confirmed: ork_persist_write appends sequentially).
- Driver already supports base+offset expert VIEWS into one buffer (own_buf, src/npu.c:255).

Mechanism:
1. Map **one layer's expert block(s)** with ~3 MEM_CREATE (one per proj tensor), experts = offset views.
   → ~3 maps/layer vs ~600 (200 active experts × 3 proj) bcreate+memcpy today. Zero hot-path memcpy.
2. **Routing-independent** → the next map is always "layer L+1's block", known before token 0
   (NOT dependent on layer L's router output). So the whole prefetch schedule is deterministic.
3. **Double-buffer**: background thread MEM_CREATEs layer L+1's block while NPU computes layer L's
   grouped GEMMs. Keep 2–3 layer-blocks resident (≤1.3 GiB < 4 GiB) so prefetch always runs ahead.
4. Within a mapped layer, existing group-by-expert (ggml-ork.cpp:3177) + all-active grouped GEMM
   (3200-3412) runs unchanged — it just addresses offset views instead of per-expert bcreate'd bufs.

The C(256,8) permutation count NEVER enters: we map per-LAYER (120 blocks total, 2–3 resident),
routing only selects offset views inside an already-mapped block.

## Regime
- **Prefill (M>1)**: the target. Grouped GEMMs amortize submit over M_e; block residency kills churn.
- **Decode (M=1)**: stays CPU (NPU submit-floor-bound at M=1; DRAM-BW wall). Not moved.

## Open risks / must-measure
- **Small-GEMM occupancy**: at B=64, M_e≈2 → tiny GEMMs (K=2048,N=512,M=2); the NPU 17× edge shrinks
  toward 1×. Needs LARGE prefill batch (256+) to grow M_e. This, not residency, may still cap the win.
- **Are orkpack blobs directly NPU-consumable, or does load_i4a8 re-tile?** If re-tiling is needed, do
  it ONCE at model load into a persistent dma-buf per layer-block (not on the hot path).
- Remap/MEM_CREATE cost of a 128 MiB block vs 256 small ones (should be far fewer TLB/pagetable ops).

## Staged plan (each stage board-gated: make test bit-exact, SIGTERM not SIGKILL, sudo reboot on wedge)
- **S0 (measure first)**: ONE run — `ORK_QUANT=4 ORK_MOE_NPU=1 ORK_PROFILE=1 ork_bench mixed.gguf
  <prompt> 64 8` — builds the experts-in-orkpack (build pass persists experts since ORK_MOE_NPU makes
  supports_op accept MUL_MAT_ID; experts from Q4_K source → uniform-int4, fine for a timing test),
  then runs the profiled prefill. Instrumentation ALREADY EXISTS (ggml-ork.cpp:2454-2479), read:
    * `[ork MoE-VERIFY] first-touch live-packs=N (…ms/pack)` = per-expert admission/churn cost.
    * `[ork MoE-chain] … pack/repack X% … chain-submit Y%` = churn vs compute.
  VERDICT RULE: if pack% >> chain-submit%, per-expert MEM_CREATE churn is the cost → build S1.
  SERIALIZE after the Q4_K_M CPU baseline (single board; NPU S0 is wedge-prone → run ATTENDED).
  Board note: mixed.q4.orkpack currently has NO experts (dense-only) — DELETE it first so S0 rebuilds with experts.
- **S1 (DONE 2026-08, board-validated)**: driver primitive `ork_mm_load_i8_block()` (src/npu.c, after
  ork_mm_load_i8) — ONE native bcreate holds all experts' Bb (+ optional Bf); each expert is an ork_w
  with owns=0 base+offset VIEWS (Bb/Bf heap_fd=-1 so ork_mm_free skips them). Opaque `ork_moe_block`
  owns the backing buf(s); `ork_moe_block_free` reclaims once. API + accessors in include/ork_npu.h.
  int8 only (int4 inflates+re-tiles — later). Test: examples/test_moe_block.c (in Makefile EXAMPLES) —
  8 experts block-loaded, each RUN byte-identical to individual ork_mm_load_i8 AND to a CPU int32 ref:
  `test_moe_block: NE=8 K=2048 N=512 M=16 -> OK` (EXIT=0, RK3588, post-reboot clean NPU). Additive
  change (no existing path touched); full `make test` + sbc_attest refresh deferred to S5.
- **S2**: frontend — in get_hot's batched (all_active) branch, load a whole layer-block once (offset
  views) instead of per-expert ork_mm_load. Bit-exact vs current.
- **S3**: double-buffer prefetch via the DOMAIN API (the swap unit). A submit runs against ONE
  iommu_domain_id (~4 GB window each); dom_activate switches the active one and only parks/restores
  SCRATCH — weights DON'T move. So: one domain per resident layer-block (ork_npu_domain_alloc ->
  set_pack_domain -> ork_mm_load_i8_block), dom_activate(domain_L) per layer, rotate a 2–3 domain pool
  (evict L-1's block, reuse its domain for the prefetched next layer). Routing-independent => the next
  fill target is always known (perfect prefetch). This is what sidesteps the S0 crash: S0 crammed ALL
  layers' experts into ONE domain -> 4 GB ceiling -> botched advance to domain 1 (CREATE: Invalid
  argument). Per-block-per-domain never fills past one layer and rotates instead of accumulating.
  NB: bcreate is NOT zero-copy (fresh DRAM alloc in the domain + copy-in); the cheap per-layer switch
  is dom_activate (scratch swap), not a weight re-map.

  COST MODEL (bcreate = DRAM alloc + IOMMU SG-map, BOTH; S0 3.4ms/call is mostly per-CALL fixed
  overhead — ioctl + IOMMU/TLB + 2 bsync — not per-byte). So the IOVA MAP should be created ONCE and
  REUSED, never rebuilt per swap. Three regimes:
   (1) working set fits resident (small MoE / enough domains): bcreate once per layer at load, then
       per-token is ONLY dom_activate. Zero per-token IOVA cost. IDEAL.
   (2) streaming (35B: ~30GB int8 experts don't fit 31GB RAM): allocate a POOL of 2-3 block buffers
       ONCE (bcreate 2-3× total), then per layer memcpy the layer's expert bytes from the mmap'd
       orkpack into a REUSED pool buffer + bsync — NO MEM_CREATE, NO IOMMU remap (same IOVA, new
       contents; valid — "write-once" is convention, not HW). Per-layer cost = memcpy+bsync only.
   (3) re-bcreate per layer (~40/prefill): still ~44× better than 1776 but strictly worse than (2).
  => S1's ork_mm_load_i8_block (fresh bcreate/call) is regime (3)/(1)-load. S3 must add a REFILL variant
     (memcpy new layer bytes into an existing block buffer + bsync, no bcreate) to reach regime (2).
- **S4**: A/B (block-residency vs current alloc/free) at prefill B∈{64,256,512}. If it WINS →
  make it the default MoE-on-NPU path, un-gated (drop the temporary A/B toggle).
- **S5**: validate (make test), OPS_REGISTRY row, refresh sbc_attest.txt, WIP→done.

## Gate
Un-gated default ONLY if A/B beats both current alloc/free AND the CPU expert path end-to-end
(prefill). Else keep experts on CPU (still-correct path), document the negative. Decode stays CPU.

## RESULTS

### CPU baseline (the bar) — Q4_K_M standard, pure CPU, big-cores (taskset 4-7, -t4), llama-bench
- **pp64 (prefill) = 21.30 t/s, tg32 (decode) = 8.69 t/s** (qwen35moe 35B-A3B, 19.7 GiB, fits RAM).
- CRITICAL: `ork_bench` hardcodes `-t 4` with NO cpu affinity → runs on the little A55 cores (cpu0-3).
  ALL ork MoE measurements MUST use `taskset -c 4-7` or they're little-core-crippled (this is what
  produced the bogus 0.11 t/s earlier). Fix ork_bench affinity, or always taskset.

### S0 (DONE 2026-08) — churn confirmed, S1 GREENLIT
Built experts-in-orkpack (30930 weights, 16.1 GiB — 30720 experts uniform-int4 + 210 dense NF4).
Profiled (ORK_MOE_NPU=1 ORK_MOE_HOT_GIB=1.8 ORK_PROFILE=1, big-core):
- `[ork MoE-chain] pack/repack 6053ms (98%) | chain-submit 108ms (2%)`
- `first-touch live-packs=1776 (3.4 ms/pack) | avg S(tasks/call)=118` (grouped batching WORKS)
- pack split: dequant 0ms, quant 0ms → the 3.4ms/pack is PURE MEM_CREATE+IOMMU-map+DMA per expert.
**Verdict: per-expert MEM_CREATE churn = 98% of MoE time; NPU compute = 2%.** Block residency
(~120 layer-block maps, resident+prefetched, vs 1776 per-expert maps) should collapse MoE overhead
from ~6100ms toward the ~108ms compute floor. S1 justified.

Two issues S0 surfaced to fix in the build:
1. all_active mode does NOT honor ORK_MOE_HOT_GIB — it fills IOVA to the 3900 MiB ceiling then
   overflows (domain-advance "CREATE: Invalid argument"). Per-layer-block eviction (keep 2-3 layers)
   is the fix. Until then, cap harder / it wedges the process in an rknpu ioctl (needs reboot).
2. warmup DECODE (M=1) routes experts to NPU (supports_op MUL_MAT_ID has no M-gate) and fails
   ret=-3 on the overflow. Decode must stay CPU → add an M>1 gate to the MUL_MAT_ID accept (S2).

## Bench artifacts on board (10.3.0.236, /home/michael/)
- qwen3.6-35b-a3b-f16.gguf (69G), -mixed.gguf (22.8G, exps Q4_K/dense F16), -q4km.gguf (building),
  -mixed.q4.orkpack (642 MiB, dense NF4 only — experts NOT packed; S0 needs an ORK_MOE_NPU=1 build).
