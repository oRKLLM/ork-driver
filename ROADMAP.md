# ork-driver Roadmap / TODO

Status of the open userspace NPU matmul stack. The reverse-engineering is done and packaged;
what's left is engineering. Empirical detail and the reasoning behind each item live in the
[Performance wiki](https://github.com/oRKLLM/ork-driver/wiki/Performance) and
[regcmd ISA Reference](https://github.com/oRKLLM/ork-driver/wiki/regcmd-ISA-Reference).

## Done

- **fp16 + int8/w8a8 matmul** — `ork_mm_pack[_i8]` / `ork_mm_run[_i8]`, K-split × N-tiling, validated vs CPU.
- **Real model end-to-end** — `stories15M` (NPU matmul + CPU ops), logits match CPU fp16.
- **GQA / arbitrary head_dim** — examples generalized (Qwen3 16/8 GQA).
- **tok/s benchmark vs `librkllmrt`** — same-state, Qwen3-1.7B w8a8 (`examples/bench.c`, `tools/rkllm_bench.c`).
- **Multi-core NPU** (`ORK_NPU_MC=n`) — N-split across cores, clamped to `soc->cores`; **decode 1.69×**.
- **Per-core full-K single-submit decode** (`ORK_FULLK_DEC`) — decode **~11 tok/s ≈ 96% of librkllmrt**.
- **Prefill CPU path** — flash-style attention + persistent thread pool pinned to the perf cores; **~1.6×**.
- **Auto-tuner** — multi-core + full-K int8 decode are now the **default per-matmul choice** (no env
  needed): cores chosen per matmul (≤ budget, ≥2 N-tiles/core), full-K decode auto-built when
  int8 & K≤10752 & multi-core & **it fits the IOMMU** (`bcreate`-success guard — abandons it and
  falls back to K-split otherwise, never crashes). Policy via `ork_npu_set_core_budget(ctx,n)`;
  `ORK_NPU_MC`/`ORK_FULLK_DEC` are now debug overrides. Decode 11 tok/s & prefill ~94 with zero env.
- **Per-channel int8 quant** — the correct fp32→int8→matmul→dequant pattern (per-output-channel
  weight + per-row activation scales) validated end-to-end vs fp32 (`examples/quant.c`, ~0.5% RMS
  error). The pattern a real engine follows; the bench keeps a dummy scale for pure throughput.
- **Persistent NPU worker pool** — the multi-core path no longer does per-matmul
  `pthread_create`/`join`; workers are spawned once and signalled per matmul. Decode unchanged
  (~11 tok/s — it's **bandwidth-bound near librkllmrt, not spawn-bound**), prefill slightly up;
  the value is lower overhead + cleaner scaling under many small matmuls.
- **Calibration tools** — `ksubmit_probe` (K ceiling), `slice_probe`, `attn_cost`, `telemetry_sample.sh`.

## Remaining (highest leverage first)

1. **int4 / `w4a16` path** — the big one. Hardware-native (RK3588 CNA has int4 weight modes;
   Rockchip exposes `w4a16`/`w4a8` + QINT8/INT4 mixing). Not yet RE'd (we have fp16 + int8). Worth
   it for two reasons: (a) 4-bit is the format real models ship in; (b) decode is weight-read-bound,
   so half the bytes ≈ **~2× decode**. *regcmd is independent of librknnrt — we use it only to RE
   the int4 weight packing + the CNA bit-width reg + how per-group/channel fp16 scales are applied,
   then ork_mm_* emits the int4 regcmd itself.* Mixed int4/int8 is then per-op dtype selection (not
   within a matmul); q3 is *not* a native mode (int4/int8/fp16 only); "q4.5" ≈ `w4a16` (int4 +
   per-group fp16 scales).
   **Status (in progress):** capture toolchain set up — librknnrt 2.3.2 + `rknn_matmul_api.h` on the
   board, `tools/int4_capture.c` (rknn_matmul int4 probe), `regcmd_capture` shim. **Pipeline
   validated end-to-end on int8** (captured the int8 `rknn_matmul` regcmd). **Blocker:** librknnrt
   2.3.2's `rknn_matmul` rejects *all* int4 types on RK3588 ("unsupported … in this platform"; only
   int8 works). So the int4 reference must come from **librkllmrt running a `w4a16` `.rkllm`** (the
   LLM runtime does int4) captured under the same shim. A `Qwen3-1.7B …rk3576-w4a16` model was tried
   but librkllmrt **rejects it on the rk3588 board** (`target_platform does not match` → init fails).
   So the runnable reference must be **(a)** an **rk3588 w4a16** `.rkllm` on this board, or **(b)** a
   capture on an **actual rk3576 board** (NanoPi M5) — the int4 weight packing/bit-width reg transfer
   across the RK35xx family, only the scheduler params (which we have for rk3588) differ. Once
   captured: diff vs `REGCMD_I8` → `synth_i4` + `ork_mm_pack_i4`/`ork_mm_run_i4`, validate like
   `examples/quant.c`. Capture pipeline proven on int8 (`tools/int4_capture.c` + `regcmd_capture`).
2. **llama.cpp-rockchip integration** — wire `libork_npu.a` in as the matmul backend so it runs real
   models via `llama-cli`/`llama-bench` with a tokenizer, not just the standalone examples. Makes the
   stack usable and properly benchmarkable.
3. **SoC validation** — RK3576 params are inherited from RK3588 (`validated=0`): needs an on-board
   run + tune, then `validated=1`. RK3562/RK3568 not added (see `docs/ADDING_AN_SOC.md`).
4. **Chunked prefill** — the only remaining long-prefill lever (O(M²) attention). Engine-level
   scheduling, not an ork-driver change. See the [Heterogeneous Serving wiki](https://github.com/oRKLLM/ork-driver/wiki/Heterogeneous-Serving-and-Scheduling).

Low value (measured): **CPU/NPU op overlap** (decode is 97% NPU; prefill ops already threaded);
**decode's last ~4%** to librkllmrt (run-to-run noise / thread+DMA overhead).

## Closed dead-ends (don't re-walk)

- **Attention-on-NPU** — de-risk floor looked great (~2.5× @M=256) but integration *regressed*
  prefill: 896 tiny per-head matmuls' per-submit/multi-core-spawn overhead + fp16↔int8 dtype-RESET
  thrash + the CPU gather/transpose/softmax/scatter (~as much as it saves). CPU flash attention wins.
- **Single-layout via a weight stride register** — no per-N-tile stride register exists; the NPU
  consumes weights as a contiguous stream, so a full-K buffer can't be K-sliced in place.
- **M>1 full-K single-submit for prefill** — the M>1 large-K `0x1040` schedule is unsolved, hunting
  it crashed the board, and prefill is CPU-bound so it wouldn't help anyway.
- **Naive multi-core / removing the per-call RESET** — both Oops the kernel (see the ISA reference
  state-gotchas: populate all `subcore_task[]`; the per-submit `RKNPU_ACT_RESET` is load-bearing
  wedge-recovery).
