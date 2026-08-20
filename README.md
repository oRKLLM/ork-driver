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

ork_npu *ctx = ork_npu_init();              // detects the SoC, opens the NPU DIRECTLY (in-process, default)
ork_w   *w   = ork_mm_pack(ctx, K, N, B);   // pack B[K,N] fp16 once, resident on the NPU
ork_mm_run(ctx, w, M, A, C);                // C[M,N] fp32 = A[M,K] fp16 x B[K,N]  (many times)
ork_npu_free(ctx);
```

Two transports, selected by **which init you call** (same API afterward):
`ork_npu_init()` opens the NPU **directly** in-process (the default — one process owns the single-stream
NPU); `ork_npu_init_orkd()` instead routes every op through the **orkd** daemon, which owns the NPU and
serializes submits so multiple processes can share it without wedging the IOMMU. `ork_npu_uses_orkd(ctx)`
reports which is in effect. For back-compat `ork_npu_init()` still honors the legacy `ORK_USE_ORKD=1` env
(delegating to the orkd path), but new callers should pick the entry point explicitly.

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

## Heterogeneous CPU∥NPU dispatcher (`ork_spine`)

`include/ork_spine.h` (header-only, C + C++) drives an **op DAG across execution units** — an NPU unit (the
doorbell on its own worker thread, so the NPU stays single-stream) plus one or more CPU worker units pinned to
spare big cores — from a single poll loop. Each op declares its dependencies (a bitmask of earlier ops) and a
placement (`ORK_PL_NPU` / `ORK_PL_CPU` / `ORK_PL_EITHER`); independent NPU and CPU work runs concurrently and a
dependent op waits for its producer.

```c
#include "ork_spine.h"
ork_spine_unit U[2];
ork_spine_unit_start(&U[0], ORK_UNIT_NPU, 4);   // one NPU unit (single-stream), big core 4
ork_spine_unit_start(&U[1], ORK_UNIT_CPU, 6);   // CPU worker, big core 6
ork_spine_op ops[] = { {0, ORK_PL_NPU, npu_fn, &a0, 0}, {0, ORK_PL_CPU, cpu_fn, &a1, 0}, ... };
ork_spine_run(U, 2, ops, nops);                 // runs the DAG to completion, overlapped
ork_spine_unit_stop(&U[0]); ork_spine_unit_stop(&U[1]);
```

**Coherency (hard rule):** an NPU-produced dma buffer read by a CPU unit on another thread needs a `dc civac`
on BOTH sides — the NPU op flushes with `ork_spine_civac_range()` after `ork_dyn_end`, and the consumer
invalidates with `ork_spine_civac_range()` before reading. At most **one** NPU unit (the doorbell can't be driven
concurrently; `ork_spine_run` rejects more). See `examples/test_spine.c` and `tools/spine_sched_probe.c`. General
infrastructure (prefill CPU-quant∥NPU-matmul, mixed CPU/NPU op-graphs, bridging IOMMU domain-swap latency) — not
tied to any one model path.

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
  `.so` dependency) — `cc your.c -lork_npu` after `make install`, or drop the library sources straight
  into your build (`src/*.c src/npu/*.c src/npu/*/*.c src/soc/*.c`, with `-Iinclude -Isrc`; the
  Makefile's CORE variable is the authoritative list).
  The header is the entire contract: `ork_npu_init` / `ork_mm_pack[_i8]` /
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

Because the test inputs are fixed-seed deterministic, the NPU output is a constant, so the
reference-bound tests (`test_matmul`, `quant`, `test_sn3`) compare an **fnv64 checksum of the NPU
output against an embedded static golden** (FNV-1a 64-bit — a dependency-free non-cryptographic hash,
~1 xor+multiply per byte, sub-ms even on a multi-MB output; change-detection needs speed, not
collision resistance, so md5/sha would be needless overhead) rather than recomputing the O(M·N·K) CPU reference every
run (a wide-K shape's reference is billions of MACs — minutes on the CPU, ~0.5 s on the NPU). This
cut those tests by ~40–200× (e.g. `test_matmul` 195 s → 5 s, `quant` 39.8 s → 1 s).

**Invalidation is deliberate, human-in-the-loop.** The golden is trusted until a checksum *mismatch*
occurs — which happens only when the NPU output actually changes. A mismatch **fails the test**
(never silently absorbed): you then decide whether the change was an intended output-altering edit —
regenerate the golden — or a regression — fix the code. The CPU reference is kept precisely for these
two moments (regenerate / diagnose); it does not run on the fast path. (This applies only to tests
bounded by a *recomputed reference*; a test bounded by the NPU op itself can't be short-circuited —
you must run the op to produce the output you'd hash.)

```sh
sudo env ORK_REGEN=1 ./test_matmul     # print fresh golden checksums to paste into the test
sudo env ORK_FULL_REF=1 ./test_matmul  # force the full CPU reference (diagnose a mismatch)
```

From a workstation, sync the source to the board and run it there, e.g.
`rsync -a . board:ork-driver/ && ssh board 'cd ork-driver && make test'`.

Builds are incremental: the core sources compile once to shared `-fPIC` objects reused across
every example, and the compiler is wrapped in `ccache` when available (`make NO_CCACHE=1` to
disable), so a re-build after an edit is seconds, not minutes. On a full-pass `make test` the
runner writes `tests/sbc_attest.txt` — a sha256 of the NPU-output-determining sources. CI can't
run the tests (no NPU on the runners), but it runs `make check-attest` (pure sha256 + grep) to
**gate every push and version bump on that attestation**: if a hashed source or a golden changed
since the last passing board run, the check fails, catching a commit that was not re-validated on
the SBC. It cannot prove the tests pass on the NPU, only that the board run happened.

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


## Capability × precision matrix

Which datapath implements what. Regenerate with `make matrix` (`tools/precision_matrix.sh` derives it
from the source tree, so it cannot drift from the code). A dagger means the capability is provided by a
shared implementation rather than that precision's own module — supported, just not its own code; those
are asserted in `tools/precision_overrides.tsv`, and each must cite the symbol doing the work, which
`make check-registry` verifies still exists.

| capability | i8 | f16 | i4 | i16 |
|---|:--:|:--:|:--:|:--:|
| regcmd synth | ✅ | ✅ | ✅ | ✅ |
| output stage (requant) | ✅ | ✅ | — | ✅ |
| fused-act output stage | ✅ | — | — | — |
| pack weights | ✅ | ✅ | ✅ | — |
| load / .orkpack persist | ✅ | — | ✅ | — |
| zero-copy import / adopt | ✅ | — | ✅ | — |
| quantise from f32 | ✅ | — | ✅ | — |
| run — single core | ✅ | ✅† | ✅ | — |
| run — multicore | ✅ | ✅† | ✅ | — |
| run — HW chain | ✅ | — | ✅ | ✅† |
| run — async stream | ✅ | ✅ | ✅ | — |
| run — NONBLOCK doorbell | ✅ | ✅† | ✅ | — |
| batched GEMM (bmm) | ✅ | ✅ | ✅ | — |
| fused matmul+activation | ✅ | ✅ | — | — |
| SDP activations | ✅ | — | — | ✅ |
| elementwise mul | ✅ | ✅ | — | ✅ |
| elementwise add | ✅ | ✅ | — | ✅ |
| per-channel multiply | ✅ | ✅ | — | ✅ |
| slice-and-dice tiles | ✅ | — | ✅ | — |
| M-fold chain | ✅ | — | — | — |
| MoE expert coalescing | — | — | ✅ | — |
| resident KV | ✅ | — | — | — |
| probes / RE replay | ✅ | ✅ | ✅ | ✅ |
| regcmd fuzz hooks | ✅ | ✅ | ✅ | — |

† **f16 / run — single core** — ork_mm_run / orki_run in npu.c dispatch fp16 — the dispatcher is dtype-agnostic by design, so there is no token to match (`orki_run`)
† **f16 / run — multicore** — i8/colsplit.c is the ONLY fp16 multicore path (#45) (`ork_dyn_begin_colsplit`)
† **f16 / run — NONBLOCK doorbell** — same colsplit path — fp16 wide-K rides the doorbell (`ork_dyn_begin_colsplit`)
† **i16 / run — HW chain** — the i16 chain rides the general PC-chain core in i8/probe.c rather than a dedicated i16 one (`ork_npu_chain_progs`)

**Most blanks are by design, not a TODO.** int4 has no SDP/activation row because the RK3588 datapath is
W8A8 *or* W4A4 symmetric — int4 activations are int4, and the SDP LUT op consumes int8/int16, so there is
no int4 activation path to implement. int16 is the *activation* precision (the accuracy tier between int8
and fp16 for SDP ops), not a weight-storage tier, which is why it has no pack/load/import/quantise row and
only 18 functions to int8's 167. fp16 weights are not persisted to `.orkpack` — the pack format is
int8/int4 — so fp16 has no persist or zero-copy-import row. MoE expert coalescing is int4-only because the
auto-profile packs experts as NF4.

The genuinely open gaps, and whether they are worth closing, are tracked on the wiki
([Precision Capability Matrix](https://github.com/oRKLLM/ork-driver/wiki)).

## Environment variables (gates & knobs)

The default path uses **no environment variables** — `ork_npu_init` → `pack` → `run` just works, and
the ggml-ork backend's product surface is the 2-option load-time config
(`ggml_backend_ork_set_load_config(dflash, silu_int8_fused)`), not env vars. Everything below is for
development, A/B measurement, and reverse-engineering. Gates live in **two** places: the ork-driver
library (`src/`) and the ggml-ork backend (in the `llama.cpp-rockchip` fork); both are listed.

Truthy = `1`/`true`/`yes`/`on` unless a value is noted. Unset = off / default.

### On-NPU op placement (which ops run on the NPU)
| var | effect |
|---|---|
| `ORK_ATTN` | attention QKᵀ·V matmuls on the NPU via the fp16 batched path (prefill M>1 only; decode stays CPU) |
| `ORK_SOFTMAX_NPU` | softmax `exp()` on the NPU (int16 SDP act-LUT); max/sum/÷ stay on CPU |
| `ORK_NORM_NPU` | RMSNorm on the NPU (rsqrt act-LUT) |
| `ORK_FFN_CHAIN` | run the SwiGLU FFN inner (gate·SiLU → up → glu → down) as one round-trip-free on-NPU chain |
| `ORK_FFN_SILU_I16` | FFN SiLU as the coherent standalone int16 SDP op (PPL ~19) |
| `ORK_FFN_FUSED_SILU` | FFN SiLU fused into the gate matmul (all-int8; lower coherence) |
| `ORK_FFN_SILU_CPU_GMAX=<thr>` | per-layer: CPU fp32 SiLU only where gate-|max| > `thr`, else fused |
| `ORK_FFN_F16`, `ORK_FFN_GATE_F16`, `ORK_FFN_F16_CPUSILU`, `ORK_FFN_F16_JIT` | fp16 FFN / fp16 gate variants |
| `ORK_GU_CHAIN` | HW-chain gate+up into one `run_chain_i8` submit (DMA-resident shared input) |
| `ORK_FUSE`, `ORK_NO_FUSE`, `ORK_FUSE_MINM=<M>` | group-fuse independent same-input matmuls (q/k/v, gate/up); min-M to fuse |
| `ORK_PPU_SILU`/`_GELU`/`_GLU`/`_ADD`/`_OPS`/`_MINM` | route these activation/elementwise ops to the PPU |
| `ORK_OFF` | force **everything** to CPU (`supports_op`→false) — the same-binary CPU baseline |

### Quantization / precision
| var | effect |
|---|---|
| `ORK_QUANT=4` | int4 W4A4 instead of int8 W8A8 |
| `ORK_HADAMARD` | int4 per-channel + block-Hadamard (FWHT) rotation |
| `ORK_HYBRID` | hybrid: FFN 4-bit, attention 8-bit |
| `ORK_MIXED_DISPATCH`, `ORK_MIXED_W4A4` | per-tensor tier dispatch from the GGUF's own precision; opt the 4-bit tier into native W4A4 |
| `ORK_NF4`, `ORK_SR` | NF4 codebook; stochastic rounding |

### Performance / runtime
| var | effect |
|---|---|
| `ORK_MIXED_NOTHRASH` | keep NPU warm across int8↔int4↔chain transitions (no per-op re-warm) |
| `ORK_NO_AFFINITY` | don't pin NPU-driver threads to the big cores |
| `ORK_ZC_OUT` | output zero-copy (matmul writes C in place; off by default — see AGENTS.md) |
| `ORK_PRECOMP_RC` | cache the M=1 decode regcmd |
| `ORK_DECODE_MC` | split M=1 decode N-tiles across all cores |
| `ORK_I4_MSCHED`, `ORK_I4_INCR`, `ORK_I4_BCHAIN` | int4 batch strategies (in-task batch / incremental+multicore / bank-chain) |
| `ORK_NPU_MC=<n>`, `ORK_MCAP=<M>`, `ORK_KTILE`, `ORK_RCAP` | core count / M-tile / K-tile / residency caps (override the SoC defaults) |
| `ORK_NO_BF`, `ORK_KEEP_BF` | drop / keep the full-K `Bf` weight buffer (compact footprint vs fused-silu need) |

### orkpack / streaming / multi-domain (ggml-ork)
| var | effect |
|---|---|
| *(automatic)* | the `.orkpack` path is DERIVED from the loaded model (`<model-dir>/<model-basename>.orkpack`); it is loaded if present, else built on first run and cached. No env var needed. |
| `ORK_ORKPACK_PATH=<path>` | DEVELOPMENT override of the derived `.orkpack` path (`ORK_PERSIST` is removed and aborts with guidance) |
| `ORK_ORKPACK_TIERMAP`, `ORK_ORKPACK_I4_FFN`, `ORK_ORKPACK_I4_ABOVE_MB`, `ORK_ORKPACK_TIER_FROM_SRC`, `ORK_ORKPACK_CPU` | per-tensor tier selection when converting an orkpack |
| `ORK_DOMAINS=<n>` | **DEPRECATED** — the domain count is auto-sized from the resident orkpack footprint. Honored **only as a clamp-UP** (force *more* domains than auto, never fewer): a value below the auto count under-provisions IOVA (last-domain overflow → Bf PRIME-fail → warmup wedge) and is ignored with a warning, while a larger value is a valid escape hatch when the auto footprint under-counts. Use `ORK_DOMAIN_LAYERS` for manual layer→domain control. |
| `ORK_DOMAIN_LAYERS` | manual layers-per-domain (advanced; the auto-sizer byte-balances by default) |
| `ORK_WCACHE_BUDGET_MB`, `ORK_STREAM_RAM_BUDGET_MB` | resident-weight / RAM streaming LRU budgets |
| `ORK_EVICT_SRC`, `ORK_NO_IMPORT`, `ORK_IMPORT_CHUNK_MB`, `ORK_NO_CONSOLIDATE_IMPORT` | zero-copy import controls |

### MoE (ggml-ork)
| var | effect |
|---|---|
| `ORK_MOE_NPU` | offload MoE experts (MUL_MAT_ID) to the NPU (experimental; off by default) |
| `ORK_MOE_HOT`, `ORK_MOE_HOT_GIB`, `ORK_MOE_BATCH_MINM`, `ORK_MOE_PHASE_EVICT`, `ORK_MOE_STREAM`, `ORK_MOE_PATHB*`, `ORK_MOE_ALL_ACTIVE`, `ORK_MOE_*_THREADS` | hot-expert residency, batched regime, phase-evict, PATH-b sub-graph, CPU threading |

### Profiling / debug / tracing
| var | effect |
|---|---|
| `ORK_PROFILE` | per-section timing (quant / NPU run / dequant; run_multicore phases) |
| `ORK_SEG_TIME` | per-op-category NPU-handler wall time (QKV / QKᵀ / softmax / A·V / O / FFN) |
| `ORK_VERBOSE` | per-node dispatch log (which compute fn each op hits) |
| `ORK_TRACE`, `ORK_PRESUBMIT_TRACE`, `ORK_DUMP_GRAPH`, `ORK_ATTN_TRACE` | submit / pre-submit / graph-node / attention-bmm traces |
| `ORK_DEBUG_RESET`, `ORK_DUMP_FAIL`, `ORK_LOAD_PROF`, `ORK_BUFPROBE` | reset accounting, dump the failing regcmd, load-time profile, buffer probe |

### Board / device
| var | effect |
|---|---|
| `ORK_NPU_CARD=<n>` | pick `/dev/dri/cardN` |
| `ORK_DMA_HEAP`, `ORK_IOMMU_DOMAIN`, `ORK_IOVA_CEIL_MB` | dma-heap name, forced domain id, IOVA ceiling |
| `ORK_NO_SIGCLEAN` | disable the graceful SIGTERM IOMMU teardown |
| `ORK_NO_GOV_WARN` | silence the CPU/DDR governor warning |

### Internal RE / calibration register probes (advanced — not for normal use)
These override individual regcmd registers or op params to sweep hardware behaviour on-board during
reverse-engineering; each is read by a specific `tools/re/*` probe. Not needed to *use* the library.
Families: `ORK_SILU_40xx` / `ORK_SILU_OBYTES` / `ORK_SILU_38DIV` (fused-SiLU output stage),
`ORK_I16OUT_*` / `ORK_F16OUT_*` (int16/fp16 matmul output-stage encoding), `ORK_F16_C*` / `ORK_F16_R*` /
`ORK_F16_ZA` / `ORK_F16_MTILE` / `ORK_F16_GCAP` (fp16 fused-SiLU calibration), `ORK_ADD16_R*` (int16 add),
`ORK_EW_*` (element-wise-mul RE), `ORK_SM_*` (softmax-replay layout), `ORK_I4_1010`/`_1040`/`_CB_*`/`_ALAY`/
`_NSUB`/`_MREGS` (int4 scheduler/bank RE), `ORK_R1040`, `ORK_SMALLTILE*`, `ORK_*_PROBE_*`,
`ORK_CONSOLIDATE_I8`, `ORK_DIRECT_I4`, `ORK_I16_DOM0`/`_RESET`/`_CON1`/`_DUMP`,
`ORK_PROBE_RESET`, `ORK_NPU_TESTCORE`, `ORK_FUSED_DUMP`/`_MTILE`, `ORK_GATE_ABLATE`, `ORK_IMATRIX`.
Grep `ORK_` in `src/` and the ggml-ork backend for the exhaustive list.

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
