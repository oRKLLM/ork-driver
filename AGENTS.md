# ork-driver — Agent Instructions & Architecture

`ork-driver` is a clean-room **userspace matmul library for the Rockchip NPU**. It synthesizes
register-command (regcmd) programs and submits them to the in-tree `rknpu` DRM kernel driver
via ioctls on `/dev/dri/cardN` — no `librknnrt`, no kernel module. It is the open NPU runtime
spun out of [oRKLLM](https://github.com/oRKLLM/oRKLLM); the reverse-engineering record lives in
the [ork-driver wiki](https://github.com/oRKLLM/ork-driver/wiki).

See [`README.md`](README.md) for the user-facing overview and API.

The active reverse-engineering findings and optimization roadmap are documented in the sibling [Wiki Home](../ork-driver.wiki/Home.md) repository.

---

## 1. What it is

- Public API (`include/ork_npu.h`): `ork_npu_init` → `ork_mm_pack(K,N,B)` → `ork_mm_run(M,A,C)`.
- `C[M,N]` (fp32) = `A[M,K]` (fp16) × `B[K,N]` (fp16); arbitrary M/K/N (`K%32==0`, `N%16==0`)
  via K-split + N-tiling + a single-submit M-scheduler, with resident weights.
- One binary supports every Rockchip NPU: the SoC is detected at runtime from the device tree.

---

## 2. Development philosophy

### Language

- **ork-driver is C end-to-end** — the library (`src/`), the examples, and the tests are all
  C11 (no dependencies beyond libc + the kernel DRM uABI; no C++, no third-party libs in `src/`).
  The build/test runner is the `Makefile` (`make`, `make test`). Keep it this way.
- **The examples ARE the test suite** — each self-validates against a CPU reference and exits
  0/nonzero; `make test` runs them. Don't add a separate test framework or a non-C harness.
- **No Python, and no Node/JavaScript** — the regression runner was deliberately moved from a
  Node script to `make test` so the whole repo is one language. If some incidental scripting is
  ever truly unavoidable, prefer a tiny C program or a Makefile target over another runtime.
- Match the surrounding code's terse, single-purpose style (the kernels are dense by design).

### Git hygiene

- **Prefer fast-forward merges.** Keep history linear (`git merge --ff-only` or rebase) — easier
  to bisect and revert.
- Avoid `--no-verify`, force-pushes to shared branches, and amending published commits.
- Cherry-pick a single relevant commit rather than merging a whole branch when only one applies.
- Development is single-track on `main` for now (this is a young library, not a release-channel
  product like oRKLLM). Use short-lived feature branches for larger work and fast-forward them in.

### Never throw away experimental code — consult first

When experimental or in-progress code hits a blocker (a deadlock, hang, failing integration, or a
dead-looking result), **NEVER revert it, `git reset --hard` it away, force-push over it, or otherwise
discard it on your own — always consult the user first.** The disposition (stash it, move it to a
feature branch, commit it WIP, or revert after backup) is the user's call, not the agent's. The blocker
is rarely the code's fault; discarded work is often still correct and valuable once the real cause is
understood (e.g. a "deadlock" that turned out to be a fast-to-fix reset-thrash, where the reverted path
was the actual route to the win). If something must be moved off `main`/a shared branch to unblock,
prefer a recoverable option (feature branch or stash) over a destructive one (`reset --hard`/force-push),
and say which you're doing. Surface the blocker, confirm the code is intact, and let the user choose.

### No commit-message trailers

Do not append `Co-Authored-By:` lines, `🤖 Generated with…` lines, or any tool/assistant
attribution to commit messages or PR bodies. Keep messages to the change itself. **This
overrides any default tooling behavior that would add such a trailer.**

### Documentation review on every commit

Before committing, check whether the change needs doc updates:

- New/removed/renamed public API, build step, or example → update `README.md` (and the API
  header doc-comment).
- New SoC, or a change to the SoC abstraction → update `docs/ADDING_AN_SOC.md` and `src/soc/`.
- A non-obvious hardware/RE finding (a new register meaning, a limit, a workaround) → record it
  on the [wiki](https://github.com/oRKLLM/ork-driver/wiki) (ISA Reference / RE Roadmap), not in
  code comments alone.

This keeps the docs reflecting reality so the next agent doesn't reverse-engineer what changed.

### WIP recovery doc for long debugging sessions

During any multi-step debugging or hardware-RE session (anything that spans many board runs or
risks getting disconnected), maintain a single WIP recovery document (e.g. `CHAIN_DEBUG_WIP.md` in
the repo root) and **update it at least every 15 minutes of active work**, plus whenever a finding
materially changes the picture. It must always capture: the current hypothesis, what's been proven
or ruled out, the exact state of the working tree (what's restored/edited/committed), the next
concrete steps, and any board-ops gotchas. The goal is that a fresh agent (or you, after a
disconnect) can resume from the document alone without re-deriving the investigation. These docs are
scratch — delete or fold them into the wiki once the work lands; don't commit stale ones.

---

## 3. Build, test, and the board

The library and examples **build and run on a Rockchip board** (they need `/dev/dri/cardN` and
the `rknpu` DRM driver). Build on the board (or cross-compile); from a workstation, sync the
source over and run there.

```sh
make                  # on the board: library + examples
make test             # on the board: build + run every example, asserting each exits 0
make test MODEL=/path/stories15M.bin    # also run the real-model llama2 test
# from a workstation: rsync -a . board:ork-driver/ && ssh board 'cd ork-driver && make test'
```

- The validated board is RK3588 (SBC IP: `10.3.0.236`).
- The NPU is **single-stream**: `make test` runs the examples serially; a wedged submit can stall the next — keep that in mind when adding tests.
  - **SBC Hard Wedge**: If the board becomes completely/hard wedged (unresponsive over SSH), use the Home Assistant MCP tools to power cycle the **"Rock 5B Plug"** smart plug. Specifically, call the `mcp_context-forge_home-assistant-hassturnoff(name="Rock 5B Plug")` tool to turn off the smart plug, followed by `mcp_context-forge_home-assistant-hassturnon(name="Rock 5B Plug")` to turn it back on.
    - **If a plug power-cycle does NOT bring it back** (plug reports `on`, but no ping/SSH after several minutes): the SPI bootloader has likely been corrupted (a hard power-cut mid-NPU-submit, or the abrupt loss, can corrupt SPI — the board then won't boot even with power restored). Recovery requires PHYSICAL access: **reflash the SPI image, then reboot from the Belkin power supply.** This was sufficient on `10.3.0.236` — **the SSD did NOT need reseating** (rootfs was intact; only the SPI bootloader was gone). A remote plug cycle cannot fix this — hand off to a physically-present operator.
    - **Avoid the cause:** never `kill -9` an in-flight NPU submit (it can wedge the NPU/IOMMU and force this hard-reset path). Use `SIGTERM` and let the submit drain; use `timeout` on board NPU commands so they self-terminate cleanly.
  - **NPU Wedge**: If only the NPU is wedged (submission hang/timeout, but the SBC is still responsive via SSH), execute a graceful reboot command over SSH to restart the board: `ssh board 'sudo reboot'`.
- A `.bin` test model (e.g. `stories15M`) is needed only for the `llama2` example; it is
  git-ignored and `make test` skips that test gracefully when `MODEL` is absent.
- **The examples ARE the tests** — each self-validates against a CPU reference (NPU output must
  match within fp16 tolerance; NaN/inf or dead output must fail). When you fix a hardware-behavior
  bug, add a shape/config case to the relevant example so `make test` would have caught it.
- **No macOS Binary Transfers:** Never push or copy local macOS binaries (from arm64/x86_64 Mac workstation builds) to the target SBC. Only source files should be synchronized (e.g., via git or targeted rsync) and then built natively on the board.

---

## 4. Architecture & repo layout

```
include/ork_npu.h      public API (init / pack / run / soc introspection)
src/npu.c              core: raw DRM submit, regcmd synthesis, K-split + N-tiling + M-scheduler
src/soc.{h,c}          runtime device-tree SoC detection + caps registry
src/soc/<chip>.c       one file per SoC: core count, CBUF budget, output-width cap, K-slice
src/rknpu_ioctl.h      open DRM uABI of the upstream rknpu kernel driver
src/regcmd_*.h         captured regcmd templates (our RE; no proprietary content)
examples/              test_matmul · layer · decode · model · llama2 — the test suite (each
                       self-validates vs CPU; MHA/GQA/arbitrary-head_dim covered). `make test` runs them.
tools/re/               NPU reverse-engineering toolkit (capture→decode→templatize new ops/NPUs; README there)
tools/re/regcmd_capture.c LD_PRELOAD calibration-capture shim (for adding SoCs / new ops)
docs/ADDING_AN_SOC.md   how to add/validate a SoC (RE narrative + scratch live on the wiki)
```

### Weight-DMA amortization (the single-core M-tile lever)

The single-core int8 matmul is **weight-DMA-bound, not compute- or row-bound**: every M-tile submit
re-streams the entire `K×N` weight from DRAM (~11 GB/s), and that cost is *independent of how many
rows the tile carries* — measured `µs/submit` is flat from mc=4→31 and scales ~linearly with N. So the
per-row throughput lever is **rows per weight-stream**: a bigger M-tile amortizes the one weight load
over more rows and cuts total weight re-reads (`M / mc` of them).

The M-tile size `mc` is therefore capped at the **largest value the hardware computes correctly**, which
is the `0x1040` K-reduction schedule limit `mg_max*64` — **not** `R-1` where `R = pow2_floor(2*cbuf/K)`.
The old `R-1` / "CBUF-resident rows" cap was a **disproven RE finding**: activations *stream* (they need
not be CBUF-resident), reg `0x1010` is only a perf hint (correctness is identical regardless of it), and
`mg_max*64` is the exact bit-exact ceiling — validated `mc=mg_max*64` is correct and `mc+1` miscomputes
at every K (704@K512, 320@K1024, 128@K2048, 64@K3584/4096). Raising the cap from `R-1` (~31) to
`mg_max*64` gave **~2.1× single-core / ~1.6× three-core at K=2048, ~1.5× at K=3584, bit-exact** (2026-06-30).

Consequences: `cbuf_elems` no longer sets the int8 M-tile size (it only feeds the neutral `0x1010` hint);
it still governs the fp16 path. This fix is int8 full-K only — the wide-K (`K>4096`) K-slice path and fp16
keep their own caps (fp16 has a separate latent large-tile M-scheduler bug). When touching any M-tile cap,
the bound is `mg_max*64`; never reintroduce an `R-1`/`pow2_floor` ceiling.

### Multi-SoC: data, not branches

The regcmd ISA and DRM path are **shared** across the RK35xx family; only *parameters* differ
(core count, CBUF/SRAM budget, output-width cap). Therefore: **one `main` branch, runtime
detection, one caps file per chip in `src/soc/` — never a per-flavor branch or fork.** Adding or
validating a SoC is one file + a regression run — see [`docs/ADDING_AN_SOC.md`](docs/ADDING_AN_SOC.md).
RK3588 is hardware-validated; RK3576 shares the code path with inherited (untested) params and
`ork_npu_init` warns until `validated=1`.

### Open-source reference: the mainline `accel/rocket` driver (consult when stuck on HW semantics)

The RK3588 NPU is **NVDLA-derived**, and there is now a clean, mainline, **open-source** kernel driver for
this exact hardware: **`drivers/accel/rocket`** (Tomeu Vizoso) + its userspace in Mesa3D, documented at
[docs.kernel.org/accel/rocket](https://docs.kernel.org/accel/rocket/). **When stuck on a register meaning, a
submit/interrupt/DMA behavior, or "why does the hardware do X" — consult it FIRST**, before guessing on-board
(each wrong guess is a wedge-risk board run). It is the authoritative open cross-reference:

- **`rocket_registers.h`** — names our numeric registers (built from the RK3588 TRM ch.36 + NVDLA). Our DPU
  output-stage `0x40xx`, the PC block `0x00xx`, CNA `0x10xx`, CDMA `0x50xx` all map to named macros there.
- **`rocket_job.c`** — the real job-submission ABI: the NPU runs **one task per PC program** (`TASK_CON` /
  `OPERATION_ENABLE`), and the **kernel IRQ handler re-arms the next task** — sequencing is kernel-driven via
  completion interrupts (`PC_INTERRUPT_*_DPU_0/DPU_1`), NOT hardware-chained through a concatenated regcmd
  buffer. DMA faults surface as `PC_INTERRUPT_RAW_STATUS_DMA_READ_ERROR`. (This is how the multi-task
  softmax-replay hang was diagnosed — see `FWD_SOFTMAX_RE_WIP.md`.)
- **NVDLA docs** (nvdla.org) — the fixed pipeline (Conv/MAC → SDP → PDP → CDP), fused vs independent mode,
  and the fixed-function unit limits (e.g. CDP LRN reduction window n≤9).

Cross-referencing these three turned numeric RE into named, understood behavior repeatedly; make it the
default move when a hardware question blocks progress. Findings from it belong on the wiki
[regcmd ISA Reference](https://github.com/oRKLLM/ork-driver/wiki) / `NPU-Quirks`.

**Multi-task hardware chaining (one submit, `task_number>1`).** The kernel programs the PC from the FIRST task
only; the hardware then walks the chain per task via the in-regcmd descriptor `0101:0x0010` (next regcmd addr) +
`0101:0x0014` (next register-amount `=(n+3)/2`). Rules for chaining a heterogeneous op-graph (softmax, fused
attention, norms) — full write-up: wiki *Exp-2026-07-10 Forward Softmax and RKNPU Chaining*:
- recompute `0x0014` for the actual next task (zeroing it hangs at transition 1);
- task stride is content-driven (slot ≥ each task's regcmd), NOT fixed alignment;
- replay a contiguous single-buffer image with a single-delta rebase (some ops write INTO the regcmd region);
- **ping-pong OFF (`submit flags = RKNPU_JOB_PC = 0x1`, not `0x5`) for any chain with a LUT-load** — ping-pong
  (`RKNPU_JOB_PINGPONG`, `1<<2`) swaps register banks the instant a task's *register config* is done, racing a
  LUT-load's SRAM-commit side effect (non-deterministic stall). `run_chain_i8` keeps `0x5` only because uniform
  int8-matmul tasks are register-config-only. See `NPU-Quirks` "Ping-pong races a chained task's side effect".

### Mode-transition layer (`ork_npu_enter`)

The NPU's regcmd datapath is **stateful**: a "mode" (precision/schedule) is tracked in software via
`c->last_dt` plus warm flags (`warmed`, `mwarm[]`) and buffer-size caches (`ccsz`, `mccsz[]`). Moving
between modes may need an `RKNPU_ACT_RESET`, a cold 2-pass re-warm, and/or a buffer realloc. That
policy used to be **copy-pasted inline into ~16 run/stream/chain/int4 entrypoints**, each drifted.
It is now owned by **one function**, `ork_npu_enter(c, to_marker, profile)` in `src/npu.c`, driven by
the **`XSPEC[]` policy table** (one row per historical transition site). Every run path calls it first;
the drift is now visible **as data** (e.g. the nothrash-keyed chain profile ignores `ORK_SSM_KEEPWARM`
where the matmul/stream profiles honor it). See the big comment above `XSPEC[]` for the exhaustive
`from→to` matrix and the modes `{ COLD(-1), F16(0), I8(1), I4(2), I8_CHAIN(3), I4_CHAIN(4),
I4_STREAM(5) }` plus the transient `SDP` (activation/ewmul).

**To CHANGE a transition's policy:** edit that profile's **one row** in `XSPEC[]`. A row is
`{ kwp, rst, wtg, wc, stg, sc, setdt }` — a keep-warm predicate selector (`KWP_*`), a reset condition
(`RC_*`), and warm/size *clear-target* (`TG_*`: scalar `warmed`/`ccsz`, per-core `mwarm[]`/`mccsz[]`,
or both) + *clear-condition* (`WC_*`) selectors, and whether it writes `last_dt` (`setdt`). No `if`
chains to hunt down.

**To ADD a new op/dtype transition:** (1) pick/define its `last_dt` marker; (2) add a `XP_*` profile to
the enum and a matching `XSPEC[]` row; (3) call `ork_npu_enter(c, marker, XP_*)` at the site (it
returns 1 iff a real transition fired — use that to drive any caller-local warmup flag, as
`XP_I4_INCR`/`XP_I4_STREAM` do). If none of the existing `KWP_*`/`RC_*`/`WC_*` selectors express the
behavior, add a selector value and its `case` in `ork_npu_enter` (each `case` is a literal predicate).

**To TEST a transition change — MANDATORY, on the board (RK3588):**
- `make test` — the primary gate. It exercises every mode: `test_matmul`/`layer`/`decode`/`model`
  (fp16+int8), `i4`/`test_chain_i4`/`perplexity_i4` (int4), `test_stream_interleave` (stream),
  `test_ssd_chunk_npu` (int8↔fp16 SSM), the SDP ops (`test_ewmul_{i8,f16,i16}`/`test_silu`/`test_gelu`/
  `test_add`), and **`test_mode_transition`**. A *behavior-preserving* refactor must pass **byte-identical**.
- Run it once **per keep-warm knob** too: default, then `ORK_SSM_KEEPWARM=0` and `ORK_MIXED_NOTHRASH=1`
  (the profiles differ only at non-default knobs).
- `make mode_probe && sudo env ORK_MM_TIMEOUT=2500 timeout 300 ./mode_probe` — the op→op transition
  matrix; confirm which pairs reset/wedge is unchanged.
- `ORK_DEBUG_RESET=1` — logs every `ACT_RESET` with a counter + caller; diff the count/sites before vs
  after (the "28→1" keep-warm behavior must be intact).

**Phase status.** The consolidation landed **behavior-preserving** (Phase 1). Sites that do **not** map
cleanly (drifted or specially-motivated) are **left as-is and catalogued in `MODE_TRANSITION_LAYER_WIP.md`** for a
Phase-2 1-by-1 evaluation — that is where behavior changes (e.g. flipping `XP_SDP.setdt=1` to fix the
fp16-mm→SDP→fp16-mm wedge, or making the chain profiles honor `ORK_SSM_KEEPWARM`) happen, each a
one-row edit re-validated by `make test` + bench. **Do not conflate** the precision-mode reset (which
the layer owns) with the `c->task` LUT-descriptor axis (`Exp-2026-07-12`): `ACT_RESET` does not fix the
latter, and it stays an op responsibility.

---

## 5. Constraints & scope

- **No Rockchip proprietary code or binaries** in this repo. `rknpu_ioctl.h` is the open kernel
  DRM uABI. The proprietary `librknnrt.so` is needed only to *run* `tools/regcmd_capture.c` for
  SoC calibration — never to build or use the library.
- This is **not a kernel driver** — it's a userspace library over the existing `rknpu` DRM
  driver. Be precise about that in docs and messages.
- Independent, community project — **not affiliated with or endorsed by Rockchip**. "Rockchip",
  "RK3576", "RK3588", "RKNN" are trademarks of Rockchip.
- Both **fp16** (`ork_mm_pack`/`run`, fp16 A·B → fp32 C) and **int8/w8a8** (`ork_mm_pack_i8`/
  `run_i8`, int8 A·B → int32 C) are supported. Keep the public API stable when extending.
  Note: mixing both dtypes on one context works, but switching modes triggers a one-time NPU
  re-warm (the regcmd mode is stateful) — see `run()` in `src/npu.c`.

---

## 6. Benchmarking & performance (MANDATORY procedure)

The open stack is `ork-driver → ggml-ork → llama.cpp` (the [`oRKLLM/llama.cpp-rockchip`](https://github.com/oRKLLM/llama.cpp-rockchip) fork vendors this repo as a submodule and adds the `ggml-ork` backend). Full history + every experiment: the **[Optimization Roadmap](https://github.com/oRKLLM/ork-driver/wiki/Optimization-Roadmap)**, **[Experiment Log](https://github.com/oRKLLM/ork-driver/wiki/Experiment-Log)**, and **[Benchmark Standards](https://github.com/oRKLLM/ork-driver/wiki/Benchmark-Standards-and-Methodology)** wiki pages.

### Before stating ANY performance number — run the standardized benchmark
1. **Same model on both runtimes.** ggml-ork: a GGUF (Q8_0). Closed baseline: the matching `.rkllm` for librkllmrt (full-model) or the RKNN matmul API (per-matmul). Never compare across model sizes or paths.
2. **Use `llama-bench`, NOT short `llama-cli` prompts** (warmup + repetition; short prompts + the lazy weight-pack produce false prefill numbers). For matmul-level work use `tools/rknn_vs_ork` / `tools/mc_prof`.
3. **Verify governors at max BEFORE timing:** DDR `/sys/class/devfreq/dmc/governor`=performance@2112MHz, CPU A76 `cpu4..7` performance@2.4GHz. A parked DDR governor ~halves decode.
4. **Run with `-t <big-core-count>` (4 on RK3588), not `-t 8`** — the little A55 cores drag the threadpool barrier and contend with the NPU-driver threads. This alone is the difference between losing to and beating librkllmrt.
5. Report BOTH prefill and decode, ≥2 reps, and state model / path / warm-vs-cold.
6. **Validate any matmul-path change against the CPU reference** (`make quant`, RMS unchanged) — `mc_prof`/`rknn_vs_ork` use dummy data and do NOT check correctness.

Reference (Qwen3-1.7B-w8a8, RK3588 board `10.3.0.236`, `-t 4`, warm): librkllmrt **184 prefill / 11.71 decode** tok/s; open stack **178 (97%) / 12.8 (109%, beats it)**.

### Managing Baseline Thresholds
When committing new performance optimizations to `ork-driver` or `ggml-ork`, you MUST run the integration benchmark script (`tools/bench_two_turn.sh` via `make bench-llama`) on the Rockchip SBC to ensure there is no significant degradation (e.g., M-scheduling bugs or KV-cache penalties). 
If your optimizations successfully increase the performance baselines above the current documented numbers, you must explicitly update the threshold constants in the test scripts (e.g., `BASELINE_DECODE` in `tools/bench_two_turn.sh`) to the new, higher values before merging. Do not allow baselines to stagnate or decay.
### Env knobs (set on the `llama-bench`/`mc_prof`/`quant` command)
| var | effect |
|---|---|
| `ORK_FUSE=1` | QKV/gate-up fusion (off — measured neutral) |
| `ORK_NO_AFFINITY=1` | don't pin NPU-driver threads to big cores (default: pin) |
| `ORK_ZC_OUT=1` | output zero-copy (off by default — **correct** at the matmul level since the DMA cache-coherency fix `3fad74a` (+ Sn>1 `033a45b`), but ~0 end-to-end gain, so opt-in; **not safe under concurrent multi-core** — the coherency bsyncs aren't serialized across cores) |
| `ORK_PROFILE=1` | per-section timing (quant / NPU run / dequant; decode vs prefill; run_multicore phases) — printed by ggml-ork on free |
| `ORK_QUANT=4` | int4 W4A4 instead of int8 (experimental, incoherent) |
| `ORK_DECODE_MC=1` | let M=1 (decode) matmuls split N-tiles across all 3 cores (default: single-core at M=1). +1.62× decode on the big FFN projections (7B-Q8_0, 0.92→1.49 t/s); off by default because small-N decode matmuls lose the multi-core barrier to the single-core dispatch floor. Residual gap to rkllm is the synchronous-execution wall, not core count |

### Diagnostic tools (board only; not in `all`/`test`)
- `make rknn_vs_ork RKNN_DIR=/tmp/rknn && sudo env LD_LIBRARY_PATH=/tmp/rknn ./rknn_vs_ork [iters] [a]` — per-matmul ork vs the closed RKNN matmul API (same int8 (M,K,N)); `a` arg = the AC-layout probe.
- `make mc_prof && sudo ./mc_prof [M] [K] [N] [iters]` — per-core copy/submit/accumulate breakdown; `ORK_TEST_DMA=1` puts A in a zero-copy DMA buffer.
- `make batch_probe && sudo ./batch_probe [ntask]` — multi-task-per-submit probe (it times out; the kernel rejects `task_number>1`).
- `sudo tools/bench_monitored.sh --label L -- <cmd>` — wraps any workload (e.g. `llama-bench`/`llama-completion`) and samples RK3588 resource use (RAM, RAM-bandwidth via DMC `devfreq/dmc/load`, NPU, GPU, CPU, swap — same sources as oRKLLM's `monitor.js`), printing per-resource **avg + peak**. Sampler pins to the little cores (`--sampler-cpus`, default `0-3`) so it doesn't perturb the workload. Makes a tok/s number attributable: it revealed decode is **latency/serialization-bound, NOT bandwidth-bound** (decode leaves NPU ~90% idle and RAM-BW at ~26% avg / 75% peak — nothing saturates). Isolate decode with `llama-bench -p 0 -n N` or a short-prompt/long-generation completion.

### Zero-copy DMA (`ork_dma_alloc`/`ork_dma_free`)
NPU-coherent CPU-mapped buffers; a matmul whose A/C live in one has the regcmd read/write it in place (no host memcpy). **Input zero-copy (A): validated correct, default on, −17% on the full-K prefill matmul.** **Output zero-copy (C): correct but off by default** — the original corruption was a DMA cache-coherency bug (CPU dirty/stale lines racing the NPU's writes), fixed in `3fad74a` (bsync clean-before + invalidate-after; Sn>1 strided in `033a45b`) and validated at the matmul level. It stays opt-in (`ORK_ZC_OUT`) because it measured **~0 end-to-end gain** (writeout saved is negligible vs prefill), not because it's wrong. Caveat: **unsafe under concurrent multi-core** (the per-core bsyncs don't serialize with the NPU writes — `run_chain_i8` multi-core falls back to single-core for DMA-buffer tasks). Realizing input zero-copy end-to-end still needs a ggml-ork DMA buffer type (the open Stage-2 item).
