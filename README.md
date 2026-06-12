# ork-driver

**A clean-room userspace matmul library for the Rockchip NPU.**

`ork-driver` drives the RK35xx NPU **directly**, by synthesizing register-command (regcmd)
programs and submitting them to the in-tree `rknpu` DRM kernel driver via ioctls on
`/dev/dri/cardN`. It does **not** use Rockchip's proprietary `librknnrt`, and it is **not** a
kernel module — it is a small userspace library that talks to the kernel driver that already
ships with your board. The regcmd ISA was reverse-engineered from scratch; the result is an
open, dependency-free fp16 matmul primitive fast enough to run a real LLM.

> Status: **RK3588 validated on hardware** (Radxa ROCK 5B). RK3576 shares the same driver and
> ISA — the code path works, but its tuning parameters are inherited and need on-device
> validation (the library prints a warning until then). See [SoC support](#soc-support).

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

## Build & run (on a Rockchip board)

Requires the `rknpu` DRM driver (stock on Rockchip Linux), a C compiler, and access to
`/dev/dri/cardN` (run as a user in the `render`/`video` group, or via `sudo`).

```sh
make                 # builds the library + examples
sudo ./test_matmul   # fp16 matmul validation vs CPU (incl. non-power-of-2 K, N-tiling, decode)
sudo ./model 12      # 12-layer transformer body, NPU vs CPU reference
```

Embed in another project:

```sh
make libork_npu.a    # static lib + include/ork_npu.h
```

## Examples (each self-validates against a CPU reference)

| Example | What it demonstrates |
| :--- | :--- |
| `test_matmul` | the matmul API across shapes (K-split, N-tiling, non-pow2 K, decode, big LM-head N) |
| `layer` | one Llama/Qwen decoder layer — NPU projections + CPU RMSNorm/RoPE/softmax/SwiGLU |
| `decode` | incremental token decode with a KV cache (M=1 projections) |
| `model` | N stacked decoder layers (a transformer body) |
| `llama2` | a **real trained model** end-to-end — Karpathy's `stories15M` — greedy generation; NPU logits match CPU-fp16 to ~0.01 |

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

`test/regression.mjs` builds the library + examples on a board and runs them (each asserts a
CPU-validated result; a wall timeout catches NPU hangs). Requires NPU hardware, no proprietary
deps.

```sh
BOARD=user@host node test/regression.mjs          # full suite
BOARD=user@host node test/regression.mjs llama2   # filter
```

## How it works

A matmul is lowered to a sequence of NPU register writes (the regcmd "ISA": `(reg, block,
value)` triples across the CNA/DPU/PPU/PC blocks), placed in a DMA buffer, and run via
`RKNPU_SUBMIT` on the card node. The library reverse-engineers the weight tile layout
(`[Ntile][Ktile][16][32]`), the feature/output addressing, and the NPU's internal M-tiling
scheduler, then handles the practical limits (contraction-dim and output-width caps, cold-start
state) so callers just see `C = A·B`. The full reverse-engineering log lives in the
[oRKLLM wiki](https://github.com/oRKLLM/oRKLLM/wiki).

## Credits & scope

Independent, community project — **not affiliated with or endorsed by Rockchip**. "Rockchip",
"RK3576", "RK3588", "RKNN" are trademarks of Rockchip. `rknpu_ioctl.h` is the open DRM uABI of
the upstream kernel driver. ork-driver contains **no Rockchip proprietary code or binaries**.

ISC licensed — see [LICENSE](LICENSE).
