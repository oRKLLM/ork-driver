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
- **Calibration tools** — `ksubmit_probe` (K ceiling), `slice_probe`, `attn_cost`, `telemetry_sample.sh`.

## Remaining (highest leverage first)

1. **int4 / `w4a16` path** — the big one. Hardware-native (RK3588 CNA has int4 weight modes;
   Rockchip exposes `w4a16`/`w4a8` + QINT8/INT4 mixing). Not yet RE'd (we have fp16 + int8). Worth
   it for two reasons: (a) 4-bit is the format real models ship in; (b) decode is weight-read-bound,
   so half the bytes ≈ **~2× decode**. **Approach:** capture librknnrt doing a `w4a16` matmul with
   `tools/regcmd_capture`, diff vs the int8 regcmd, decode the 4-bit weight packing + the CNA
   bit-width regs (same method that cracked int8). **Prerequisite:** a `w4a16` `.rkllm` (or a
   librknnrt int4 probe) on the board — none present today (only w8a8). Mixed int4/int8 is then
   per-op dtype selection (not within a matmul). q3 is *not* a native mode (int4/int8/fp16 only);
   "q4.5" ≈ `w4a16` (int4 + per-group fp16 scales).
2. **llama.cpp-rockchip integration** — wire `libork_npu.a` in as the matmul backend so it runs real
   models via `llama-cli`/`llama-bench` with a tokenizer, not just the standalone examples. Makes the
   stack usable and properly benchmarkable.
3. **SoC validation** — RK3576 params are inherited from RK3588 (`validated=0`): needs an on-board
   run + tune, then `validated=1`. RK3562/RK3568 not added (see `docs/ADDING_AN_SOC.md`).
4. **NPU-side persistent thread pool** — the multi-core path still does per-matmul
   `pthread_create`/`join`; a persistent pool (like the CPU one) would cut that overhead and push
   multi-core past 1.69×.
5. **Chunked prefill** — the only remaining long-prefill lever (O(M²) attention). Engine-level
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
