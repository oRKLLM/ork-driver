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

0. **int4 W4A4 matmul — DONE ✅ (2026-06-12), public API.** ork-driver does a real int4×int4→int32
   matmul on the NPU via its own regcmd (maxerr=0 vs CPU), independent of librknnrt — the first open
   *regcmd* int4 on RK3588. `ork_mm_pack_i4`/`ork_mm_run_i4` are callable like the fp16/int8 paths
   (`examples/i4.c`, in `make test`): N-tile at 64, K-split at the 10752 ceiling (== int8) with int32
   accumulate, M-tile by row — validated across M-tiling, N-tiling (N=256), and K-split (K=12288).
   The quant path (fp32→int4→NPU→dequant `C_real=aScale[m]*bScale[n]*C`) gives ~10% RMS (W4A4 is
   4-bit on both operands — coarse; this is why Hadamard helps, and why w4a16 below is more accurate).
   How: captured librknnrt type-10 W4A4 (`tools/int4_capture.c`, `B_layout=NATIVE`) → `src/regcmd_i4.h`
   → `synth_i4`; native tile layouts documented in `rknn_matmul_api.h` (A `(K/32,M,32)`,
   B `(N/64,K/32,64,32)`, C `(N/8,M,8)`); the runtime's "12 tasks" = 4-task M-tiling × 3 subcores,
   each an M=1 GEMM. **Remaining:** int4 multi-core (reuse the int8 multi-core path), N>64
   single-submit (parameterize the int4 N-output regs), real per-group scales, llama.cpp wiring.

   **w4a16 (fused fp16×int4) is a HARDWARE dead-end on RK3588.** The NPU MAC only multiplies MATCHING
   precisions: the only supported rknn_matmul types are 1 (f16×f16), 2/3/9 (int8×int8), 10 (int4×int4)
   — every mixed type (5 f16×i8, 7=w4a16 f16×i4, 8, 11 i8×i4, 12, 15) returns "unsupported matmul dtype
   in this platform", a silicon limit not a packaging choice. Proven directly: a fully-derived fp16×int4
   regcmd (correct fp32-output group, `0x100c=0x20000160`, fp16 activation regs) runs but reads the
   fp16 activations as integers → garbage, because there is no fp16×int4 MAC mode to enable. So w4a16
   models run EITHER by dequantizing int4 weights → fp16 once and running f16×f16 (saves disk/load but
   no decode-bandwidth win — weights stream as fp16), OR via the W4A4 path above (int4×int4, 2× decode,
   needs activation quant + a Hadamard rotation for accuracy). W4A4 is thus the correct/only native
   low-bit matmul on RK3588. (The non-functional fp16×int4 synth probe was reverted; RE record stands here.)

1. **int4 / `w4a16` path** — the big one. Hardware-native (RK3588 CNA has int4 weight modes;
   Rockchip exposes `w4a16`/`w4a8` + QINT8/INT4 mixing). Not yet RE'd (we have fp16 + int8). Worth
   it for two reasons: (a) 4-bit is the format real models ship in; (b) decode is weight-read-bound,
   so half the bytes ≈ **~2× decode**. *regcmd is independent of librknnrt — we use it only to RE
   the int4 weight packing + the CNA bit-width reg + how per-group/channel fp16 scales are applied,
   then ork_mm_* emits the int4 regcmd itself.* Mixed int4/int8 is then per-op dtype selection (not
   within a matmul); q3 is *not* a native mode (int4/int8/fp16 only); "q4.5" ≈ `w4a16` (int4 +
   per-group fp16 scales).
   **Status (2026-06-12) — capture abandoned as unnecessary; deriving from the fp16↔int8 delta.**
   *No runnable int4 reference is reachable on this board, and it turns out we don't need one.*
   librknnrt 2.3.2's `rknn_matmul` rejects *all* int4 types on RK3588 (int8-only); the runtime does
   not officially expose int4 here. The w4a16 `.rkllm` reference is a catch-22: a `…rk3576-w4a16`
   model is platform-rejected, and the `…rk3588-w4a16-grq` model **segfaults on load** under
   v1.1.0–v1.1.4 (loader older than the model's toolkit format) yet is **platform-rejected** by
   v1.2.0–v1.2.3 (stricter target_platform check). No runtime version both accepts *and* loads it.
   **But the NPU silicon supports int4** — so ork-driver emits the int4 regcmd directly, bypassing the
   runtime's gate (the entire point of the project). We don't need a capture; we **derive** the format
   from the fp16↔int8 register delta (both already RE'd) and **validate against a CPU reference**
   (like `examples/quant.c`), since we control both sides.
   **Derived plan (from the `REGCMD`/`REGCMD_I8` diff):** the precision axis (fp↔int) is encoded in
   `0x100c` (CNA), `0x3010` (DPU), `0x4010`/`0x40c0` (PPU), and a per-port float-bit (`…ffff` vs
   `…fffe` on `0x1070`/`0x1110`/`0x4020`); fp16 sets "float" bits that int8 clears. **w4a16 is mixed —
   int4 weights × fp16 activations** — so its regcmd is a *hybrid*: activation/feature regs take the
   **fp16** template values (16-bit activations), weight regs take new int4 values, byte-linear-halved
   from int8: `0x1030=K*N/2`, `0x1034=K/2`, `0x1044=ceil(K/128)`, `0x107c=K/32`, rows budget
   `4*cbuf/K`. Three unknowns the 2-point delta can't give: **(a)** the int8↔int4 weight-bit-width
   subfield (likely a low field of `0x100c`, which reads `0x0000` for int8), **(b)** the int4 nibble
   packing order within the `[16][32]` weight tile, **(c)** per-group fp16 scale application. Resolve
   by **bounded sweep against a CPU reference**: factor out (c) by baking a single global scale into
   the reference, then sweep (a)+(b) (a few values each) until NPU output matches — no librkllmrt
   needed. Then `synth_i4` + `ork_mm_pack_i4`/`ork_mm_run_i4`, validate end-to-end like
   `examples/quant.c`; per-group scales added after the plain-int4 matmul validates.
   **Sweep result (2026-06-12, `tools/i4_probe.c` + `synth_i4`/`ork_npu_probe_i4`):** built the
   harness and swept **both bases (fp16, int8) x 18 `0x100c` candidates x 4 nibble layouts** vs a CPU
   reference. **No match, no near-miss (<50 err).** fp16-base runs but computes wrong (the fp16 weight
   datapath doesn't decode packed nibbles); int8-base mostly runs once the output/M-scheduler regs are
   correctly N-parameterized (built on `synth_i8`, not a bare template — that cut wedges 72->20) but is
   still wrong. **Conclusion: the int4 enable is a *coordinated multi-block* config (CNA bit-width +
   DPU/PPU precision + the right nibble tile geometry + likely the per-group-scale path), not a single
   `0x100c` field — too large a space to brute-force blind.** The 2-point fp16<->int8 delta pins the
   weight-size scaling (solid) and the structure, but cannot converge the int4 enable without a real
   reference. **Realistic unblock: capture on the NanoPi M5 (rk3576, 10.6.0.14)** where librkllmrt
   *does* run int4 (an rk3576 w4a16 `.rkllm` under the proven `regcmd_capture` shim); the int4 weight
   packing + CNA bit-width regs transfer across RK35xx (only scheduler params differ, and we have those
   for rk3588). `i4_probe` (with the `base`/`i4mode`/`layout`/override knobs) is then the validator.
   Safety note: the sweep ran 72->20 consecutive wedges serially with no crash — per-call
   `RKNPU_ACT_RESET` + serial submits is safe; only *concurrent* wedging storms the queue.
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
