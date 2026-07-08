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

## Zero-copy import & streaming models bigger than the IOVA window

The NPU's IOMMU is 32-bit, so only ~4 GiB of weights are DMA-mappable at once. Two surfaces help:

**Zero-copy import** — allocate a dma-buf the NPU reads *in place* (no second allocation, no
host→device copy):

```c
void *p = ork_dma_import(ctx, bytes);      // dma-heap buffer, mmap'd + IOMMU-mapped
memcpy(p, tiled_bytes, bytes);             // fill once (pre-tiled weights, or an activation A)
ork_dma_import_sync(ctx, p, bytes);        // flush CPU writes -> device (dma-buf cache clean)
// ... pass p as A/C to ork_mm_run, exactly like an ork_dma_alloc buffer ...
ork_dma_import_free(ctx, p);
```

`ork_mm_load_i8_import` / `ork_mm_load_i4a8_import` are the import-backed loaders: same blob and
byte-identical result as `ork_mm_load_i8` / `ork_mm_load_i4a8`, but each resident tile is an
imported dma-buf (saves the kernel page alloc). All four return `NULL` if the dma-heap is absent so
the caller falls back. Import eliminates the *copy*, not the 4 GiB *cap*.

**Streaming pool** (`ork_stream_pool_*`) — for models too big to keep resident, hold a set of
**already-inflated int8 weights resident in CPU RAM** (budget by RAM, far larger than the 4 GiB IOVA
window) and map/unmap them to IOVA cheaply on demand. A cache *hit* pays only the cheap `MEM_CREATE`
import (~170 µs @ 4 MiB), skipping the expensive int4→int8 inflate (paid **once** on `add`) and the
expensive `MEM_DESTROY` (paid only on **evict**):

```c
ork_stream_pool  *pool = ork_stream_pool_create(ctx);
ork_stream_entry *e = ork_stream_pool_add_i4a8(pool, K, N, blob, n); // fill = inflate, ONCE
// per use (cache hit): cheap map -> run -> unmap; entry stays filled in RAM after unmap
ork_stream_pool_map(pool, e); ork_stream_pool_run(pool, e, M, A, C); ork_stream_pool_unmap(pool, e);
ork_stream_pool_remove(pool, e);   // the caller's eviction frees the RAM buffer
```

The pool provides the **lifecycle only** (hold-in-RAM, cheap map/unmap, free); the LRU/eviction
**policy and RAM budget live in the caller**. Both stores are covered: `add_i8` (fill = copy the
stored tile bytes) and `add_i4a8` (fill = inflate the nibbles). A transient prefetch double-buffer is
just a small pool the caller fills ahead on a background thread. `ork_stream_pool_create` returns
`NULL` if the dma-heap is unavailable (fall back to `ork_mm_load_i8` + `ork_mm_run_i8`).

> Note: a *transient* ring that maps **and unmaps every swap** does not reach resident speed — the
> per-swap `MEM_DESTROY` (~0.5–2 ms) is irreducible overhead on top of the submit, and the inflate
> only hides behind the submit at large M (crossover ≈ M 256 on a 2048×2048 layer). The RAM-resident
> cache wins by keeping that cost off the per-submit hot path.

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

The single-core matmul is **weight-DMA-bound** — each M-tile submit re-streams the whole `K×N`
weight from DRAM — so the kernel picks the **largest M-tile the `0x1040` schedule allows**
(`mg_max*64`) to amortize that weight stream over as many activation rows as possible (~2× single-core
vs the earlier conservative tile). See AGENTS.md *"Weight-DMA amortization"* for the full account.

## Status & roadmap

What's done (fp16/int8 matmul, multi-core, decode ≈ closed runtime, prefill flash attention) and
what's left (int4/`w4a16`, llama.cpp integration, auto-tuner, more SoCs) — with the closed
dead-ends — is tracked in **[ROADMAP.md](ROADMAP.md)**.

## Enabling the NPU on-chip SRAM (optional, advanced)

The RK35xx NPU can use a slice of the SoC's on-chip system SRAM as a fast, DMA-able buffer — a
"second memory interface" alongside DRAM. On RK3588 that region (`sram@ff001000`) is ~956 KB.
**Stock vendor kernels ship with it disabled for the NPU**: the driver's SRAM support is compiled
out (`# CONFIG_ROCKCHIP_RKNPU_SRAM is not set`) *and* the device tree hands the system SRAM to the
video decoder (`rkvdec`) instead of the NPU. So out of the box the NPU reports **0 bytes** of SRAM.
Turning it on takes two changes — a kernel config flag and a device-tree edit — which means
**rebuilding your board's kernel**. This is optional; the library works without it.

> ⚠️ A bad kernel or DTB can leave the board unable to boot. Install the new kernel/DTB
> **additively** (never overwrite the stock ones), keep the stock kernel as a bootable fallback,
> and have a rescue boot (e.g. an SD card) ready before you reboot.

1. **Get the kernel source** matching your running kernel (`uname -r`) — the vendor branch your
   distro built from. For the RK35xx *vendor* kernel that both **DietPi and Armbian** ship, that is
   the Armbian Rockchip vendor tree — [`armbian/linux-rockchip`](https://github.com/armbian/linux-rockchip),
   branch **`rk-6.1-rkr5.1`** (Rockchip's `rk-6.1` BSP, = 6.1.115). DietPi does not maintain its own
   kernel; its build system ([`MichaIng/build`](https://github.com/MichaIng/build)) pins exactly this —
   `KERNELSOURCE=https://github.com/armbian/linux-rockchip`, `KERNELBRANCH=rk-6.1-rkr5.1`,
   `KERNELPATCHDIR=rk35xx-vendor-6.1`. Match the branch to your `uname -r` (a different point release
   uses a different `rkrX.Y` branch).
2. **Seed the config from the running kernel** and flip one flag, using a distinct local version so
   the result installs *alongside* the stock kernel:
   ```sh
   zcat /proc/config.gz > .config          # or: cp /boot/config-$(uname -r) .config
   ./scripts/config --enable ROCKCHIP_RKNPU_SRAM
   ./scripts/config --set-str LOCALVERSION -sram
   make olddefconfig
   ```
   (The driver also needs `CONFIG_NO_GKI=y` — vendor kernels set it — and the NPU running in IOMMU
   mode, which is the default.)
3. **Wire the SRAM to the NPU in the device tree.** In the SoC `.dtsi`, give an SRAM region under
   the `mmio-sram` controller to the `rknpu` node via a `rockchip,sram` phandle, and remove any
   conflicting claim on that region. The simplest approach — if you don't need hardware video
   decode — is to repurpose the region the decoder uses: assign the whole syssram region to one
   child node, reference it from `rknpu`, and drop the `rockchip,sram` refs on the `rkvdec` nodes.
   On RK3588 the region is `sram@ff001000` (size `0xef000` = 956 KB):
   ```dts
   &rknpu {
       rockchip,sram = <&rknpu_sram>;   /* a region node under sram@ff001000 */
   };
   ```
4. **Build** the kernel, modules, and device trees: `make Image modules dtbs`.
5. **Install additively and flip the boot selection.** Install the kernel under a versioned name,
   `make modules_install` (its `LOCALVERSION` dir won't collide with the stock modules), and place a
   copy of the edited DTB. Point the bootloader (extlinux entry, `boot.scr`, or the `/boot` symlinks
   your distro uses) at the new kernel + DTB, leaving the stock kernel + DTB entries intact as the
   fallback.
6. **Reboot and verify** with the SRAM probe, which calls the driver's `RKNPU_GET_TOTAL_SRAM_SIZE`
   ioctl:
   ```sh
   cc -O2 -Isrc -o sram_probe tools/sram_probe.c && sudo ./sram_probe
   # before:  NPU SRAM total=0 bytes
   # after:   NPU SRAM total=978944 bytes (956.0 KB), free=978944 bytes
   ```

Once enabled, allocate an NPU buffer with the `RKNPU_MEM_TRY_ALLOC_SRAM` flag and the allocator
places it in SRAM when it fits (falling back to normal DMA memory otherwise) — a small, fast,
DMA-addressable on-chip region for the NPU. The mechanism (config flag + DT phandle) is the same
across the RK35xx family; only the region name and size differ by SoC. Note: staging int8 matmul
*weights* here showed no speedup on RK3588 — the weight-load bottleneck is the NPU's on-chip CDMA
path, not DRAM bandwidth, and weights large enough to be DRAM-bound don't fit in 956 KB — so treat
SRAM as a capability to build on (scratch, small hot buffers), not an automatic matmul win.

## Troubleshooting

### Board won't boot after a hard NPU wedge (solid-blue LED, no network)

A repeatedly bad NPU submit can hard-wedge the device beyond what a `sudo reboot` (or even a
power-cycle) clears — the board comes up to a **solid-blue LED and never reaches the network**.
On RK3588 boards this is made worse by **using a non-official power supply** (the board is
power-supply sensitive; a marginal PSU won't reliably cold-boot it). A bad state can end up in
the **SPI bootloader**, so swapping the SSD alone doesn't fix it.

Milder cases recover with a physical **cold boot** — press the power button (and, on some
boards, the recovery/reset button next to it) rather than relying on a smart-plug power-cycle.

If it still hangs at the blue LED, the recovery that worked (DietPi on a Radxa ROCK 5B, SSD +
SPI boot) was to **re-flash the SPI bootloader from a known-good microSD**:

1. Confirm it isn't the SSD — remove the SSD; if the blue-LED hang persists, the boot fault is in
   the SPI/bootloader, not the disk.
2. **Erase the SPI flash** and boot from a **known-good microSD** (it should boot).
3. From the booted system, **re-flash the SPI** (`dietpi-config` → *Advanced Options* → flash
   bootloader to SPI).
4. Reinstall the SSD and remove the microSD. (It may still refuse to boot at this point.)
5. Re-insert the microSD — the board then boots **from the SSD** (the microSD presence completes
   the boot).
6. Shut down, remove the microSD, and boot once more — it now boots cleanly from the SSD.

**Prevention:** use the official power supply, and **stop NPU runs with `SIGINT`, never
`SIGKILL`** — `kill -9` skips the driver's cleanup path and leaks IOVA (`failed to allocate
IOVA: -12`), which forces reboots and is the kind of repeated-bad-submit churn that hard-wedges
the device in the first place.

## Credits & scope

Independent, community project — **not affiliated with or endorsed by Rockchip**. "Rockchip",
"RK3576", "RK3588", "RKNN" are trademarks of Rockchip. `rknpu_ioctl.h` is the open DRM uABI of
the upstream kernel driver. ork-driver contains **no Rockchip proprietary code or binaries**.

ISC licensed — see [LICENSE](LICENSE).
