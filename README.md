# ork-driver

**A clean-room userspace matmul library for the Rockchip NPU.**

`ork-driver` drives the RK35xx NPU **directly**, by synthesizing register-command (regcmd)
programs and submitting them to the in-tree `rknpu` DRM kernel driver via ioctls on
`/dev/dri/cardN`. It does **not** use Rockchip's proprietary `librknnrt`, and it is **not** a
kernel module — it is a small userspace library that talks to the kernel driver that already
ships with your board. The regcmd ISA was reverse-engineered from scratch; the result is an
open, dependency-free fp16 + int8/w8a8 matmul primitive fast enough to run a real LLM.

> Status: **RK3588 validated on hardware** (Radxa ROCK 5B). On Qwen3-1.7B w8a8 it does **decode
> ~11 tok/s (~96% of the closed `librkllmrt`)** and prefill ~94 tok/s — multi-core and the full-K
> int8 decode layout are chosen automatically (no tuning flags); per-channel int8 quant is
> validated (~0.5% error). **int4 (W4A4) matmul is validated too** (`ork_mm_pack_i4`/`ork_mm_run_i4`,
> the first open *regcmd*-based int4 on RK3588 — maxerr=0 vs CPU, with K-split/N-tiling/M-tiling);
> `w4a16` (fp16 activations) is next. RK3576 shares the driver/ISA — the path works but its tuning
> params are inherited and need on-device validation (the library warns until then). See
> [ROADMAP.md](ROADMAP.md) and [SoC support](#soc-support).

## What it does

```c
#include "ork_npu.h"

ork_npu *ctx = ork_npu_init();              // detects the SoC, opens the NPU
ork_w   *w   = ork_mm_pack(ctx, K, N, B);   // pack B[K,N] fp16 once, resident on the NPU
ork_mm_run(ctx, w, M, A, C);                // C[M,N] fp32 = A[M,K] fp16 x B[K,N]  (many times)
ork_npu_free(ctx);
```

- **`C[M,N]` (fp32) = `A[M,K]` (fp16) × `B[K,N]` (fp16)**, arbitrary `M`/`K`/`N`
  (`K%32==0`, `N%16==0`). K-split + N-tiling + M-tiling are handled internally.
- **Resident weights**: pack once, stream activations — the transformer access pattern.
- **One binary, every chip**: SoC detected at runtime from the device tree.

The matmul engine is the foundation; `examples/` builds on it up to a full LLM forward pass.

## Quantization: int8, int4, NF4, mixed-precision

The same `pack` → `run` shape is available quantized. The NPU MAC is int8, so int4 is a
**storage** win (half the bytes on disk / in host RAM); weights inflate to int8 just before the
tiled DMA.

```c
// w8a8: int8 weights, int8 activations -> int32 accumulate (A int8[M,K], C int32[M,N])
ork_w *w = ork_mm_pack_i8_f32(ctx, K, N, Bf32, bscale);  ork_mm_run_i8(ctx, w, M, A, C);

// w4a8: 4-bit weight STORAGE, int8 compute. Uniform grid by default;
//   ORK_NF4=1 -> NF4 (normal-float-4) codebook (better for Gaussian-ish weights);
//   ORK_SR=1  -> stochastic rounding.
ork_w *w4 = ork_mm_pack_i4a8(ctx, K, N, Bf32, bscale);

// w4a8 + importance matrix: per-input-channel weights pick a clip-optimal per-channel scale.
ork_w *wi = ork_mm_pack_i4a8_im(ctx, K, N, Bf32, imatrix /*len K, NULL=uniform*/, bscale);
```

- **Compact int4 persist** — `ork_w_dump_i4a8` serializes the nibble store + per-channel scales
  (`'O4N1'` blob, ~½ the tiled-int8 dump); `ork_mm_load_i4a8` reloads it straight into NPU DMA
  (inflate → tile). `ork_w_bscale` / `ork_w_quant_kind` expose the stored scales / codebook.
- **Mixed-precision allocation** — `tools/gguf_tier_map.c` reads any GGUF's per-tensor quant
  types and maps them onto `{int8, int4}` tiers by an effective-bits threshold (the
  accuracy↔memory dial); `--emit-map` writes a `name<TAB>tier` file. The llama.cpp-rockchip
  `ggml-ork` backend uses this to build a mixed int8/int4-NF4 `.orkpack` (int4 for the bulk,
  int8 for importance-bumped tensors), quantizing **values from an fp16 source** with the GGUF
  used only as the allocation oracle.

## Build & run (on a Rockchip board)

Requires the `rknpu` DRM driver (stock on Rockchip Linux), a C compiler, and access to
`/dev/dri/cardN` (run as a user in the `render`/`video` group, or via `sudo`).

```sh
make                 # builds the library + examples
sudo ./test_matmul   # fp16 matmul validation vs CPU (incl. non-power-of-2 K, N-tiling, decode)
sudo ./model 12      # 12-layer transformer body, NPU vs CPU reference
```

## Integrating into another project

ork-driver is **built from source** — it compiles for the board's ARM64 + kernel DRM uABI, so
you build it on the target (or cross-compile for `aarch64`), not download a generic binary. It
has **no external dependencies** (just libc + the kernel headers already in the repo). Build a
library and link the C ABI in `include/ork_npu.h`:

```sh
make lib          # produces libork_npu.a (static) and libork_npu.so (shared)
make install      # → $(PREFIX)/lib/{libork_npu.a,libork_npu.so} + $(PREFIX)/include/ork_npu.h
```

- **C / C++ (e.g. the llama.cpp-rockchip backend):** statically link `libork_npu.a` (no runtime
  `.so` dependency) — `cc your.c -lork_npu` after `make install`, or drop the `src/*.c` straight
  into your build. The header is the entire contract: `ork_npu_init` / `ork_mm_pack[_i8]` /
  `ork_mm_run[_i8]`.
- **Other languages (Python / Node / Rust):** link `libork_npu.so` and FFI against the same C
  ABI (`dlopen` / `ctypes` / `node-ffi` / `bindgen`).

Cross-compile example: `make lib CC=aarch64-linux-gnu-gcc`.

**Tuning is automatic.** The library picks the parallelization per matmul — multi-core N-split
across the NPU cores, and a full-K single-submit int8 decode layout when it fits the IOMMU — so a
caller just packs and runs. To bound it (e.g. reserve cores for another workload) call
`ork_npu_set_core_budget(ctx, n)`.

## Examples (each self-validates against a CPU reference)

| Example | What it demonstrates |
| :--- | :--- |
| `test_matmul` | the matmul API across shapes (K-split, N-tiling, non-pow2 K, decode, big LM-head N) |
| `layer` | one Llama/Qwen decoder layer — NPU projections + CPU RMSNorm/RoPE/softmax/SwiGLU |
| `decode` | incremental token decode with a KV cache (M=1 projections) |
| `model` | N stacked decoder layers (a transformer body) |
| `llama2` | a **real trained model** end-to-end — Karpathy's `stories15M` — greedy generation; NPU logits match CPU-fp16 to ~0.01 |
| `test_chain_i4` | isolated hardware validation of chained int4 (W4A4) tasks |

`llama2` needs `stories15M.bin` (Karpathy's [tinyllamas](https://huggingface.co/karpathy/tinyllamas)):
```sh
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
sudo ./llama2 stories15M.bin 32
```

## SoC support

The regcmd ISA and DRM submit path are **shared** across the RK35xx family — what differs is
*data*, not code: NPU core count, the on-chip CBUF/SRAM budget (which sets the M-tiling rows),
and the matmul output-width cap. So there are **no per-chip branches**: one `main`, runtime
detection, and a per-SoC caps file under [`src/soc/`](src/soc/).

| SoC | Cores | Status |
| :--- | :--- | :--- |
| RK3588 | 3 | ✅ validated on hardware |
| RK3576 | 2 | ⚠️ inherited params — needs on-device validation |
| RK3562 / RK3568 | 1 | not yet added |

Adding or validating a SoC is one file + a regression run — see
[docs/ADDING_AN_SOC.md](docs/ADDING_AN_SOC.md).

## Tests

The examples **are** the test suite — each self-validates its NPU result against a CPU
reference (and the matmul/layer/decode/model examples sweep MHA, GQA, non-power-of-2 K,
N-tiling, and decode). `make test` builds and runs them on the board, asserting each exits 0;
a wall timeout catches NPU hangs. Requires NPU hardware, no proprietary deps.

```sh
make test                  # build + run all examples; "ALL TESTS PASSED" on success
make test MODEL=/path/to/stories15M.bin   # also run the real-model llama2 test (skipped if absent)
make bench-llama           # run the integration benchmark script tools/bench_two_turn.sh
```

From a workstation, sync the source to the board and run it there, e.g.
`rsync -a . board:ork-driver/ && ssh board 'cd ork-driver && make test'`.

## How it works

A matmul is lowered to a sequence of NPU register writes (the regcmd "ISA": `(reg, block,
value)` triples across the CNA/DPU/PPU/PC blocks), placed in a DMA buffer, and run via
`RKNPU_SUBMIT` on the card node. The library reverse-engineers the weight tile layout
(`[Ntile][Ktile][16][32]`), the feature/output addressing, and the NPU's internal M-tiling
scheduler, then handles the practical limits (contraction-dim and output-width caps, cold-start
state) so callers just see `C = A·B`. The full reverse-engineering record lives in the
[ork-driver wiki](https://github.com/oRKLLM/ork-driver/wiki) (start with the [regcmd ISA Reference](https://github.com/oRKLLM/ork-driver/wiki/regcmd-ISA-Reference)).

## Status & roadmap

What's done (fp16/int8 matmul, multi-core, decode ≈ closed runtime, prefill flash attention) and
what's left (int4/`w4a16`, llama.cpp integration, auto-tuner, more SoCs) — with the closed
dead-ends — is tracked in **[ROADMAP.md](ROADMAP.md)**.

## Credits & scope

Independent, community project — **not affiliated with or endorsed by Rockchip**. "Rockchip",
"RK3576", "RK3588", "RKNN" are trademarks of Rockchip. `rknpu_ioctl.h` is the open DRM uABI of
the upstream kernel driver. ork-driver contains **no Rockchip proprietary code or binaries**.

ISC licensed — see [LICENSE](LICENSE).
