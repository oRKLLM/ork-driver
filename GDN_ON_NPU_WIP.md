# GDN-on-NPU — WIP / recovery record (feat/gdn only)

**Scope:** everything about the on-NPU **Gated-DeltaNet** (delta-rule linear attention: Qwen3-Next /
Qwen3.5 / Ornith-1.0-9B) scan. This branch (`feat/gdn`) is where the GDN work lives; it is deliberately
kept OFF `main` because **GDN-on-NPU is coherent + fully wired but NOT a performance win** (see verdict).
This doc is the handoff so a fresh agent can resume, revive, or reuse it without re-deriving. Delete/fold
into the wiki [[Exp 2026-07-13 Linear-Attention Scans on NPU]] if this ever lands or is abandoned.

**Last updated:** 2026-07-13.

---

## TL;DR verdict

Built end-to-end, wired into `ggml-ork`, and **validated byte-coherent in a real Qwen3.5 model** — but the
on-NPU GDN scan **loses to ggml's CPU scan** and the gap widens with scale. Not-yet-viable. Parked here,
`ORK_GDN_NPU`-gated (off), **no regression to any other path**.

Per-stage isolation (`ORK_GDN_PROF`), NPU-scan vs CPU-scan (projections int8-on-NPU both sides, warm):
| model | quant | NPU-GDN scan | CPU-GDN scan | ratio |
|---|---|---|---|---|
| Qwen3.5-0.8B | Q8 | ~262 ms | ~135 ms | ~1.9× slower |
| Qwen3.5-9B | Q4 | ~748 ms | ~156 ms | ~4.8× slower |

NPU-scan breakdown: **NPU matmul-submit 44% · CPU UT-solve 38% · prep(fp16 cast) 14% · post 4%.**

**Why it loses (the real result):** (1) the chunked form we must use to make the scan into NPU matmuls needs
a **CPU forward-substitution UT-transform** (`(I+A)⁻¹`, 38% of the scan) — but ggml's CPU GDN op is a
*solve-free per-token recurrence* that never pays it; (2) NPU submit overhead on many small 128×128
matmuls×heads×layers; (3) **no scale crossover** — GDN's state is a fixed 128×128/head that fits in cache,
so the CPU recurrence stays fast at any size (unlike Mamba-2, whose memory-bound recurrence craters at scale
and gave SSM its 2× win). An earlier "0.8B 1.14× win" was single-run noise (CPU jumped 112→150 between runs).

---

## What's proven / ruled out

- ✅ **Math correct.** `examples/test_gdn_chunk.c` (pure CPU) proves the chunked delta-rule (with the
  UT-transform by forward substitution) is bit-exact to the definitional recurrence (~1e-11) at the real
  d=128/CS=64. In `make test`.
- ✅ **Kernel coherent on hardware.** `examples/test_gdn_chunk_npu.c` → `ork_gdn_scan_f32`: rel-L2 ~4e-4 vs
  the sequential recurrence at d=128/CS=64, multi-chunk, multi-seq. RC=0.
- ✅ **End-to-end coherent in a real model.** Qwen3.5-0.8B-Q8 + Qwen3.5-9B-Q4 (unsloth GGUFs on the board):
  `ORK_GDN_NPU=1` routes the op to the NPU and produces **byte-identical output** vs CPU. No wedges.
- ✅ **Keep-warm is essential** (same as SSM): without `ORK_SSM_KEEPWARM=1` the fp16-scan/int8-projection
  interleave churns `ACT_RESET` per layer → 25 tok/s; with it → ~130. (Keep-warm is now default-on on `main`.)
- ❌ **Not a perf win** (see verdict). ❌ **No crossover expected** with model size (cache-resident CPU baseline).
- ⚠️ **GQA `rep>1` path (Hk<Hv) NOT exercised on a real model** — Qwen3.5-0.8B is Hk=Hv=16. A model with
  Hk=16/Hv=32 (Qwen3-Next / larger) would exercise the marshaller's key-head replication. Code is there,
  logic matches the CPU op (`iq1 = iv1 % neq1`), but unverified end-to-end.

## Two bugs fixed during bring-up (instructive)
1. **ST6 (Sdelta) M=d=128 hit the documented fp16 large-M-tile bug** (AGENTS §weight-DMA; SSM only ever used
   M=CS=64) → the kernel tiles ST6's d key-rows into fp16-safe M≤64 slices.
2. **d=128 output blew up — it was the TEST DATA, not the kernel.** Unnormalized random `k` makes `k·k`~√d,
   so `(I+A)⁻¹` goes ill-conditioned and the solve diverges. Real GDN **L2-normalizes q,k per head** exactly
   to prevent this; the harness now normalizes (kernel correctly assumes normalized q,k, like ggml).

---

## Code map (this branch)

**ork-driver (`feat/gdn`, this repo):**
- `src/npu.c`: `ork_gdn_scan_f32` (the kernel — 6 fp16 matmul stages on `ork_mm_run_stream_f16_chain` + CPU
  forward-subst UT-transform), `struct gdn_pool` + `gdn_pool_ensure/free` (persistent pool), `ork_gdn_cs`
  (`ORK_GDN_CS`, default 64), `ORK_GDN_PROF` per-stage timer (`g_gdn_*` + `ork_gdn_prof_dump`), `ORK_GDN_DBG`
  per-stage norm dump. Teardown in `ork_npu_free`.
- `include/ork_npu.h`: `ork_gdn_scan_f32` decl (layout: q/k/v `[ns,nt,nh,d]`, g/beta `[ns,nt,nh]`, s0/s_new
  `[ns,nh,d,d]` key-major, o `[ns,nt,nh,d]`).
- `tools/gdn_opcount.c`: floor-model op-count (94.7% matmul — but the model is optimistic; it compares
  NPU-matmul vs a hypothetical CPU-matmul, NOT vs ggml's actual solve-free recurrence).
- `examples/test_gdn_chunk.c` (CPU ref) + `examples/test_gdn_chunk_npu.c` (NPU) — both in `make test`.
- Also carries the SSM perf defaults (shared `npu.c`) — those are the real `main` deliverable (@ `6ecb168`).

**llama.cpp-rockchip fork (NOT yet split/committed — see "fork state"):**
- `ggml/src/ggml-ork/ggml-ork.cpp`: `gdn_densify` + `ggml_backend_ork_gated_delta_net` marshaller (densify
  non-contiguous q/k/v/g/β + state key/val transpose `ggml s[val*d+key]` ↔ kernel `[key*d+val]` + GQA
  `hv%Hk` replication; o writes straight to `dst->data`), `GATED_DELTA_NET` cases in supports_op (gated
  `ORK_GDN_NPU`, K==1/GDA g_ne0==1/GQA-divides/d%16/nt≥64) + graph_compute.
- `ggml/src/ggml-backend.cpp`: **weightless-op offload** path in `ggml_backend_sched_backend_id_from_cur`,
  scoped to `GGML_OP_GATED_DELTA_NET` (GDN has no `USAGE_WEIGHTS` src so the sched won't route it otherwise;
  requires supports_op && offload_op, gated by `op_offload`). This is a core-ggml change — reusable for any
  weightless activation-only op (attention).
- ggml CPU ground truth to shadow: `ggml/src/ggml-cpu/ops.cpp:10614` (`gated_delta_net_one_chunk`); op ctor
  `ggml/src/ggml.c:6261`; the model's build path `src/models/delta-net-base.cpp`.

## Build & run (board `10.3.0.236`)
- CPU ref + NPU test: `make test_gdn_chunk test_gdn_chunk_npu && ./test_gdn_chunk && sudo ./test_gdn_chunk_npu`.
- End-to-end A/B (fork's ork_bench + Qwen3.5 GGUFs already on the board under
  `/var/lib/orkllm/models/unsloth/Qwen3.5-{0.8B,4B,9B}-GGUF/`):
  `sudo env ORK_GDN_NPU=1 ORK_SSM_KEEPWARM=1 ORK_GDN_PROF=1 ./build/bin/ork_bench <gguf> ~/ssm_prompt.txt 128 4`
  vs the same without `ORK_GDN_NPU` (CPU-GDN). Compare the `[ork GDN PROF]` line + the prefill tok/s.
- Board gotchas: single-stream NPU (serialize runs); `timeout`+`sudo` every board cmd; SIGTERM not `kill -9`
  on an in-flight submit; `[gdn dbg]` lines appearing = the op routed to the NPU (kernel ran).

## Fork state (IMPORTANT — not yet cleanly committed)
- **Workstation fork** `/Users/michael/Dev/llama.cpp` (branch `feat/ssm-npu`, 393 ahead of `master`): the SSM
  marshaller + SSM size-gate AND the GDN marshaller + cases + weightless-offload are ALL **uncommitted
  together** in the working tree. The SSM half belongs on the fork's SSM/master path; the GDN half belongs on
  a fork `feat/gdn`. **This split has not been done yet** — next agent should mirror the ork-driver split
  (checkpoint GDN to fork `feat/gdn`, keep SSM marshaller+size-gate on the main path). The fork's
  master-landing follows its deletion-aware-squash protocol (see the ggml-ork fork merge memory).
- **Board fork** `~/llama.cpp` (branch `integ/chain-streaming`, diverged): the GDN edits were applied via a
  python patch (`/tmp/gdn_patch*.py`); backups at `*.pre_gdn.bak`. Its vendored `ggml/src/ggml-ork/ork-driver`
  was overwritten (rsync) with this branch's `npu.c`.

---

## Revival path (what a breakthrough would need)
1. **Move the UT-solve onto the NPU** — kill the 38% CPU forward-substitution (blocked triangular inverse via
   small GEMMs / Neumann iteration). Even then, submit overhead (44%) must drop a lot.
2. **Cheaper submit** — the scan is many small 128×128 matmuls; better chaining / fewer submits.
3. **A crater-prone CPU baseline** — a linear-attention variant whose CPU form is memory-bound (like Mamba-2)
   rather than cache-resident would give the NPU a crossover. GDN's fixed 128×128/head state does not.

## Reusable machinery (already valuable — the real payoff of this work)
The **weightless-op offload** (ggml-backend), **densify marshaller**, **fused-multicore fp16 stream**, and
**keep-warm** are exactly what **attention-on-NPU** (softmax + attention matmuls) needs — the #1
re-evaluation target now that GDN proved they work end-to-end. See the memory
`reevaluate-decisions-post-gdn` for the full re-eval list.
