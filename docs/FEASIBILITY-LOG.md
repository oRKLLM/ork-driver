# ggml-backend-rknpu — a from-scratch ggml backend for the RK3588 NPU

> Historical feasibility & engineering log (migrated from the oRKLLM wiki). Explains why `ork-driver` drives the NPU via raw regcmd instead of the closed `librknnrt`. The current state lives in [REVERSE-ENGINEERING.md](REVERSE-ENGINEERING.md).

> ⚠️ **CORRECTION (Milestone 15, 2026-06-12):** the decode numbers in Milestones 7, 10, 13 and 14 were taken with the board's **DRAM throttled to 528 MHz** (¼ of the RK3588's 2112 MHz max) — its DMC DVFS governor was parked at the lowest step. Decode is bandwidth-bound, so this crushed the NPU's numbers specifically. **At full DDR the NPU is competitive on decode and clearly wins prefill** — the "CPU wins, replace the runtime" verdict in M13/M14 is RETRACTED. See **Milestone 15** below.


Goal: write a fresh `ggml_backend_rknpu` that drives the Rockchip NPU directly through its DRM kernel driver (`/dev/dri/renderD*`), so a current llama.cpp/ggml (with native EAGLE-3, multi-layer hidden states, speculative decoding) can run on the NPU — instead of being stuck on Rockchip's frozen, closed `librkllmrt.so` fork. This is the "sustainable architecture" identified in [RKLLM Runtime Internals](RKLLM-Runtime-Internals).

Code lives in the oRKLLM repo under `experimental/ggml-rknpu/`. This page is the running engineering log.

## TL;DR feasibility

- **Device + memory + job-submission plumbing: buildable, and the memory layer is now validated on real hardware** (see Milestone 1). The DRM uapi is fully known.
- **The matmul compute is the wall.** The RK3588 NPU is a fixed-function NVDLA-derived **INT8/FP16 convolution** accelerator, not a programmable matmul unit. Two routes to matmul, both hard:
  1. **`rknn_matmul_api`** (closed `librknnrt.so`) lowers matmul→1×1 conv for you (supports FP16, INT8, INT4, FP16×INT4) — but needs offline weight relayout to native `NC1HWC2`, and a prior ggml RKNPU2 backend ([marty1885](https://github.com/marty1885/llama.cpp/tree/rknpu2)) found it impractical for LLM *decode* (batch-1 GEMV is far below the conv array's GEMM peak; INT8-only mixing; relayout fights ggml).
  2. **Raw DRM regcmd** — bypass the runtime and emit register-command buffers ourselves. The opcode table / `RKNPU_OFFSET_PC_*` register values are **undocumented**; only [mtx512/rk3588-npu](https://github.com/mtx512/rk3588-npu) has partially RE'd them.
- **Honest verdict:** buildable as a research backend; the practicality wall (LLM decode is GEMV-bound and the NPU wants big INT8 GEMM/conv) is well-documented and independently reproduced. We proceed eyes-open — even a partial backend (e.g. INT8 GEMM offload for prefill/prompt) is useful and is genuinely novel open work.

## Target interface (verified on the board)

- Board: RK3588, kernel `6.1.115-vendor-rk35xx`, RKNPU driver (debugfs reports hw v0.9.8; ioctl drv_version 0.3.140), 3 NPU cores @ 1 GHz, IOMMU enabled.
- NPU = DRM render node **`/dev/dri/renderD129`** (`DRIVER=RKNPU`, `of_node=npu`). Render node ⇒ no DRM auth/master needed; user needs the `render` group (or root).
- UAPI: `drivers/rknpu/include/rknpu_ioctl.h` (Rockchip BSP `develop-6.1`), transcribed verbatim into `experimental/ggml-rknpu/rknpu_ioctl.h`.
- Ioctls (`DRM_IOWR('d', 0x40+n, struct)`): `RKNPU_ACTION`(0), `RKNPU_SUBMIT`(1), `RKNPU_MEM_CREATE`(2), `RKNPU_MEM_MAP`(3), `RKNPU_MEM_DESTROY`(4), `RKNPU_MEM_SYNC`(5).
- Submission model: `RKNPU_SUBMIT` carries a `rknpu_task[]` array (in a GEM buffer at `task_obj_addr`); each `rknpu_task.regcmd_addr` points at a register-command buffer of `regcfg_amount` 64-bit `(op,reg,value)` words that program the conv/MAC engine. `core_mask` (bit per core) + `subcore_task[5]` partition work across the 3 cores.

## Milestone 1 (2026-06-12): device + memory layer validated ✅

`experimental/ggml-rknpu/rknpu_probe.c` (self-contained C, no libdrm) opens `renderD129`, queries the driver, and does a full GEM buffer round-trip. On the board:

```
opened /dev/dri/renderD129 (fd=3)
  hw_version    = 0x46495245      ("FIRE" magic)
  drv_version   = 0.3.140
  iommu_enabled = 1
  freq          = 1000000000 Hz
  total/free_sram = 0
MEM_CREATE ok: handle=1 dma_addr=0xfffff000 obj_addr=0xffff00014404dc00 size=4096
mmap+write+sync ok: first bytes a5 a5 a5 ...
MEM_DESTROY ok
```

So from our own code we can: talk to the driver, allocate IOMMU-mapped DMA buffers, `mmap` them to the CPU, write, and sync to device. That is exactly `ggml_backend_buffer` / `ggml_backend_buffer_type` in ggml terms — the backend's allocator is effectively done.

Notes: `MEM_CREATE` flags used = `NON_CONTIGUOUS|CACHEABLE|IOMMU|ZEROING|KERNEL_MAPPING`. `dma_addr` is the device IOVA to feed into regcmd/task buffers. `total_sram=0` ⇒ this board exposes no on-chip SRAM scratch (everything via DDR+IOMMU).

## Milestone 2 (2026-06-12): NPU matmul validated via `rknn_matmul_api` ✅

`experimental/ggml-rknpu/rknpu_matmul_test.c` runs a real FP16 matmul on the NPU through Rockchip's `rknn_matmul_api` (closed `librknnrt.so`, fetched from rknn-toolkit2). A=`[4×32]`=1.0, B=`[32×16]`=1.0 → C should be all `K=32`:

```
created. io sizes A=256 B=1024 C=256
C[0..4] = 32.0 32.0 32.0 32.0   (expect 32)
NPU matmul result: CORRECT
```

So from our own code the NPU now computes a numerically-correct matmul. Notes:
- `type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32`; `B_layout=0` (normal `[K,N]`, lib relayouts internally); RK3588 alignment: `K%32`, `N%16` for fp16.
- `io.{A,B,C}.size` report the per-tensor byte sizes to `rknn_create_mem`; `rknn_tensor_mem.virt_addr` is the CPU mapping.
- Relevant LLM dtype modes exist in the API: `RKNN_FLOAT16_MM_INT4_TO_FLOAT32` (7) and `…_TO_BFLOAT16` (12) — the weight-only-quant case, the next thing to validate.

**Where this leaves the backend:** both foundations are proven on hardware — the **memory layer** (Milestone 1, raw DRM) and the **compute layer** (Milestone 2, matmul). A `ggml_backend_rknpu` can now wrap `rknn_matmul_api` for its `GGML_OP_MUL_MAT` (route 1: fast to working, closed-lib dep). The raw-DRM regcmd route (route 2) reuses the Milestone-1 memory layer + the `RKNPU_SUBMIT` path but still needs the undocumented regcmd opcodes.

> **Toolchain note:** building these probes required `gcc` + `libc6-dev` + `linux-libc-dev` on the board, and the matmul test needs `librknnrt.so` + `rknn_api.h`/`rknn_matmul_api.h` from [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) (`rknpu2/runtime/Linux/librknn_api/`). The repo commits only the test sources, not the closed lib.

## Milestone 3 (2026-06-12): the backend is written and compiles into ggml ✅

`experimental/ggml-rknpu/ggml-rknpu.cpp` — the actual `ggml_backend_rknpu`, a BLAS-style CPU-assist backend: host (CPU) buffers, claims only `GGML_OP_MUL_MAT`, offloads it to the NPU via `rknn_matmul_api`. Implements the full ggml backend interface (backend/device/reg vtables, `supports_op`, `graph_compute`) + header + `CMakeLists.txt`, wired into ggml's build (`GGML_RKNPU` option, `ggml_add_backend(RKNPU)`, registration in `ggml-backend-reg.cpp`). `INTEGRATION.md` has the drop-in steps for any llama.cpp checkout.

Built against current llama.cpp on the board:
```
-- Including RKNPU backend
[ 6%] Building CXX object ggml/src/ggml-rknpu/.../ggml-rknpu.cpp.o
[ 8%] Linking CXX shared library bin/libggml-rknpu.so
[ 8%] Built target ggml-rknpu
```
Clean compile + link, no errors. The backend is a real, registered ggml device.

**MUL_MAT mapping** (v0): ggml `dst = src1 · src0ᵀ` → rknn `C[M,N] = A[M,K] · B[K,N]` with `A=src1` (M=ne11, K=ne10), `B=src0ᵀ` (K,N), `C→dst`. `supports_op` claims `MUL_MAT` only for FP16 weights × F32 activations, 2D, contiguous, `K%32==0 && N%16==0` (RK3588 fp16 alignment); the scheduler keeps everything else on CPU. Build deps: `gcc/g++/make/cmake` + `librknnrt.so` on the board.

## Milestone 4 (2026-06-12): MUL_MAT runs on the NPU inside ggml, correctly ✅

`experimental/ggml-rknpu/test_rknpu_mulmat.cpp` builds a ggml graph with one `MUL_MAT` (fp16 weight `[K=32,N=16]` × f32 activation `[K=32,M=4]`), allocates host tensors, and runs it via `ggml_backend_graph_compute(rknpu_backend, gf)`:
```
rknpu supports this MUL_MAT? 1
graph_compute status = 0
c[0..4] = 32.0 32.0 32.0 32.0   (expect 32)
NPU MUL_MAT in ggml: CORRECT
```
So the backend is validated **end-to-end through ggml's own framework**: `supports_op` claims it → the graph executes on the RKNPU backend → the op is offloaded to the NPU via `rknn_matmul_api` → result is numerically correct. **`ggml_backend_rknpu` is a working v0.**

Note on `test-backend-ops -b RKNPU -o MUL_MAT`: it reports `Backend RKNPU: OK` but every one of its generated cases printed `not supported [RKNPU]` and ran on CPU — they're all batched (`bs=[8,3]`) GEMV / f32×f32, none matching the v0 `supports_op` (fp16 weight × f32, 2D, aligned). That's correct gating (no wrong results, no crashes), not NPU execution — hence the targeted test above. Broadening `supports_op` (batching, GEMV, padding, quant) is the next work.

**Build speed:** `GGML_CCACHE=ON` (default) + `ccache` installed → incremental rebuilds of the backend are cached/fast.

## Milestone 5 (2026-06-12): generalized — padding, batching, F16/F32 ✅

Lifted three of the four v0 constraints (`experimental/ggml-rknpu/ggml-rknpu.cpp`):
- **Arbitrary K/N** — zero-pad to `Kp=align(K,32)`, `Np=align(N,16)` (padding K with zeros leaves the dot-product unchanged; extra N-columns are ignored on copy-out). M is unconstrained.
- **Batched / broadcast** — loop over `(i2,i3)` planes with ggml's broadcast (`r2=ne12/ne02`, `r3=ne13/ne03`), one NPU matmul per plane, reusing the rknn ctx.
- **F16 *or* F32** weights and activations — converted to fp16 for the NPU.

Validated with `test-backend-ops -b RKNPU -o MUL_MAT`:
```
total MUL_MAT cases:                      1409
ran on NPU (compared vs CPU): 181  -> all OK
gated to CPU (not supported): 1228
```
**181 cases now execute on the NPU and match the CPU reference** (was 0 in v0). The 1228 still on CPU are: quantized weights (q4_0/q8_0/…), permuted/non-contiguous (`per=[0,2,1,3]`), and some batched cases — all correctly gated. Incremental rebuild after editing the backend: **~5 s** (ccache + make).

## Status of the four "go beyond" steps (2026-06-12)

| Step | Status |
|---|---|
| 1. Arbitrary K/N (padding) | ✅ done (Milestone 5) |
| 1. Batched/broadcast planes | ✅ done (Milestone 5) |
| 1. F16/F32 in/out | ✅ done (Milestone 5) |
| 3a. **Quantized weights** (q4_0/q8_0/… via dequant→fp16) | ✅ done (Milestone 6) — 631 NPU cases, all match CPU |
| 2. **Permuted/non-contiguous activations** (src1) | ✅ stride-aware A-fill; weights (src0) stay contiguous |
| 4. **GEMV-vs-GEMM benchmark** | ✅ done (Milestone 7) — prefill 548 / decode 8 GFLOP/s → hybrid |

### Remaining (next session)
- **Safe per-weight cache** (Step 3 proper): hang the dequant'd/relayout'd `B` off the weight tensor (`extra`/buffer hook with a free path), not a raw pointer map — turns the per-call `O(N·K)` weight prep into a one-time cost. Required before the backend is usable in a real run (decode would otherwise re-prep weights every token). The data-ptr cache was tried and reverted (stale-buffer hazard).
- **Native INT4/INT8** (`RKNN_FLOAT16_MM_INT4`): repack ggml quant blocks → RKNN native layout + group scales, to keep the 4-bit memory-bandwidth win instead of expanding to fp16.
- **Hybrid `offload_op`**: route only large-M (prefill) MUL_MAT to the NPU per the benchmark verdict.
- src0-permuted weights (rare) and remaining gated types.

## Milestone 6 (2026-06-12): quantized weights run on the NPU ✅

`supports_op` accepts any block-quantized `src0` with a `to_float` (q4_0/q8_0/q4_K/…); `mul_mat` dequantizes each weight row → fp32 → fp16 and feeds the NPU path. `test-backend-ops -b RKNPU -o MUL_MAT`: NPU-executed cases **181 → 631** (now incl. q4_0 & friends), all match CPU, zero failures. Caveat: this is **dequant→fp16** — broad coverage, but the 4-bit weight expands to fp16 before the NPU, so no memory-bandwidth win (native INT4/INT8 explored in Milestone 8).

## Milestone 7 (2026-06-12): GEMV-vs-GEMM benchmark — the practicality verdict ✅

Standalone steady-state NPU matmul (fp16, weight pre-loaded), `K=2048`:

| workload | shape | NPU |
|---|---|---|
| decode GEMV | M=1, N=2048 | 8.1 GFLOP/s (1.04 ms) |
| decode GEMV (ffn) | M=1, N=5504 | 9.0 GFLOP/s |
| decode GEMV (lm_head) | M=1, N=32000 | 10.9 GFLOP/s |
| prefill GEMM | M=256 | 442 GFLOP/s |
| prefill GEMM | M=512 | 548 GFLOP/s |

**~60× gap** between prefill (GEMM) and decode (GEMV) — the conv/MAC array is starved at batch-1 but near-peak for large-M GEMM. First statement of the **prefill→NPU, decode→CPU** verdict (later refined by the 3-way Milestone 11 and the sustained-vs-burst Milestone 12, which shows the Mali GPU competitive for sustained decode).

## Milestone 8 (2026-06-12): quant on the NPU — what's actually possible ✅

"Can the backend do quants?" — yes, but with a sharp limitation, measured (`experimental/ggml-rknpu/rknpu_bench_q.c`, timing only):

| dtype | decode GEMV (M=1) | prefill GEMM (M=512) |
|---|---|---|
| fp16 × fp16 | 7.7 GFLOP/s | 534 GFLOP/s |
| **int8 × int8** | **12.6** | **1391** (2.6× fp16) |
| fp16 × int8 (weight-only) | **create FAILED** | **create FAILED** |
| fp16 × int4 (weight-only) | **create FAILED** | **create FAILED** |

- **Quantized weights run today via dequant→fp16** (Milestone 6, 631 cases) — correct, but the 4-bit weight expands to fp16 before the NPU, so **no memory-bandwidth win** (the main point of quant, esp. for decode).
- **The ideal LLM mode — fp16 activations × int4/int8 weights — fails `rknn_matmul_create`** on this RK3588/librknnrt (likely needs explicit `B_quant_type`/`group_size`/scale setup, or is genuinely unsupported — the "no fp×int mixing" limitation marty1885 reported). So there's no easy direct weight-only-quant NPU path here.
- **int8 × int8 works and is 2.6× fp16 for prefill** (1391 GFLOP/s ≈ 1.4 TFLOP/s), but requires quantizing *activations* to int8 too (dynamic quant + accuracy handling) and still leaves decode GEMV at ~12 GFLOP/s (< CPU).

**Net:** on RK3588, NPU quant is either dequant→fp16 (have it, no bandwidth win) or full int8×int8 (prefill-only win, needs activation quant). Either way it reinforces the hybrid policy — prefill on NPU (int8 makes it ~1.4 TFLOP/s), decode on CPU. The bandwidth-optimal 4-bit-direct path isn't reachable through this runtime; that's a hard wall, not just effort.

### Confirmed: weight-only int modes are unsupported on this platform
Re-tested `fp16×int4` and `fp16×int8` with every `B_quant_type` (per-layer/channel/group) and `group_size` — all return `-5` with an explicit runtime error:
```
E RKNN: rknn_matmul_create_dynamic_shape, unsupported matmul dtype :
        RKNN_FLOAT16_MM_INT4_TO_FLOAT32 in this platform   (and ..._INT8...)
```
So the bandwidth-optimal **4-bit-weight-direct path is a hard platform limitation** of this RK3588/librknnrt, not a config issue. NPU quant options reduce to: **dequant→fp16** (have it; no bandwidth win) or **int8×int8** (prefill win, needs activation int8 quant). Decode stays CPU-bound either way.

## Milestone 9 (2026-06-12): correct weight cache + the per-call-overhead finding ✅

The weight cache is now implemented correctly — prepared fp16 `B` stored on `src0->extra` (per tensor object), gated to leaf weights (`op==NONE`, single-plane). This fixes the earlier stale-hit bug: ggml zeroes a recycled tensor struct (`extra==NULL`) → cache miss → recompute, so no wrong results. Validated: **631 NPU cases, 0 failures**.

Backend-level benchmark (`experimental/ggml-rknpu/cache_bench.cpp`), `q4_0 [2048×2048]`, per-call MUL_MAT:

| | call 1 (miss) | steady (cache hit) | CPU |
|---|---|---|---|
| decode GEMV (M=1) | 160 ms | **38.8 ms** | 0.44 ms |
| prefill GEMM (M=256) | 147 ms | **48.6 ms** | 56.5 ms |

- **The cache works**: it removed the ~120 ms `q4_0` dequant from every call (160→38.8 ms).
- **But it exposed the next bottleneck**: steady-state ~38 ms vs ~1 ms for the standalone pre-loaded matmul (Milestone 7). The gap is **per-call `rknn_matmul_create` + the 8 MB weight upload into NPU memory**, which the backend still redoes every call. The host-side `B` is cached; the *device-side* ctx + B-mem are not.
- **Two hard truths from the CPU column:** (1) decode GEMV — even at the standalone ~1 ms ceiling, the NPU **loses to the CPU's 0.44 ms**; quant/cache can't fix that. (2) prefill GEMM — NPU ceiling ~8 ms vs CPU 56 ms is a **~7× win**, but only if the per-call ctx+upload overhead is eliminated.

**Conclusion:** the correct weight cache is necessary but not sufficient. The remaining work to realize the prefill win is a **persistent rknn ctx + device weight-mem cache** keyed by `(weight, M, Kp, Np)` — upload the weight to NPU memory once, reuse the ctx across calls. With that, prefill should approach the ~8 ms ceiling and beat the CPU; decode stays on the CPU regardless. This is the concrete next step.

## Milestone 10 (2026-06-12): persistent ctx+weight cache — prefill beats CPU 6× ✅

Implemented the device-side persistence: the `rknn_matmul_ctx` + A/B/C device mems + the **uploaded weight `B`** are cached on the weight's `tensor->extra`, per `(M,Kp,Np)`. `rknn_matmul_create`, the dequant, and the 8 MB weight upload now happen **once per weight**; each call only refills `A` (activations) → `run` → copies `C`.

**The NPU can't hold a ctx+weight-mem for every weight** — it exhausts after ~150 persistent ctxs (`rknn_matmul_run failed`). So a global **LRU cap (32)** evicts + destroys the oldest. With that, `test-backend-ops` passes (**3/3 backends: CPU, RKNPU, Vulkan**).

Steady-state per-call, `q4_0 [2048×2048]`:

| | per-call (M9) | **persistent (M10)** | CPU |
|---|---|---|---|
| prefill GEMM (M=256) | 48.6 ms | **9.3 ms** | 57.3 ms |
| decode GEMV (M=1) | 38.8 ms | **0.87 ms** | 0.50 ms |

- **Prefill on the NPU now beats the CPU ~6×** (9.3 vs 57 ms) — the prefill win is realized, hitting the standalone ceiling.
- **Decode** dropped to the ~1 ms ceiling but **still loses** to the CPU's 0.5 ms — decode stays on the CPU, as predicted.
- First call per weight is still ~160 ms (one-time dequant + upload), amortized over reuse.

**Scaling caveat (important):** a real model has *more* matmul weights than the cap (and the NPU can't hold them all as persistent ctxs). So **full-model decode would thrash** the cache (evict+rebuild every token) — persistent reuse only pays off when a small set of weights is hot (prefill of a layer, repeated shapes, benchmarks). The scalable design would be a **single shared ctx with weight-swap** (or the NPU full-model offload that `librkllmrt` already does). This is the structural reason the practical role is **prefill offload**, not full-model NPU execution.

## Milestone 11 (2026-06-12): CPU vs NPU vs Mali-GPU — the full split picture ✅

All three backends registered in one ggml build (CPU + RKNPU + Vulkan/Mali). `three_way_bench.cpp`, `q4_0 [2048×2048]`, steady-state per-call (tensors resident in each backend's own buffer):

| | decode GEMV (M=1) | prefill GEMM (M=256) |
|---|---|---|
| **CPU** | **0.48 ms** | 57.0 ms |
| **NPU** | 0.85 ms | **8.8 ms** (6.5×) |
| **GPU (Mali/Vulkan)** | 0.94 ms | 42.6 ms |

- **Decode → CPU.** The CPU wins; the **Mali GPU is the *worst*** (0.94 ms). Decode is memory-bandwidth-bound and all three share the same LPDDR (~25 GB/s), so no unit gets more bandwidth — the GPU just adds the most overhead. No quant/cache changes this.
- **Prefill → NPU.** The NPU wins decisively (8.8 ms, 6.5× CPU). Mali only modestly beats the CPU (42.6 vs 57).
- **The Mali GPU has no winning role** for LLM matmul on RK3588 — slowest at decode, a distant second at prefill. Its useful niches are elsewhere (the Eagle-3 draft head, or offloading non-LLM work to free the CPU).

### Can we split tensors across CPU/GPU/NPU? — answered
Yes, at op granularity. ggml's `ggml_backend_sched` routes each op to a registered backend via `supports_op`, inserting cross-backend copies. With all three registered (as here), the optimal policy is an **`offload_op` that sends large-M (prefill) MUL_MAT to the NPU and leaves decode/GEMV on the CPU** — Mali unused for matmul. (No intra-op tensor parallelism: one op runs on one backend.)

## Milestone 12 (2026-06-12): sustained vs burst decode — the ranking flips ✅

A 30-iteration burst of a sub-ms op measures **cold/idle clocks + per-op overhead**, not steady state. Sustained (18 s per backend, per-second windows) tells a different story. Decode GEMV `q4_0 [2048×2048]`:

| | burst (M11) | **sustained 18 s** | trend |
|---|---|---|---|
| GPU (Mali) | 0.94 ms (worst) | **0.331 ms / 3024 it/s** (best) | ramps cold→warm |
| CPU | 0.48 ms (best) | 0.474 ms / 2108 it/s | rock steady |
| NPU | 0.85 ms | 1.215 ms / 823 it/s (worst) | **degrades ~14%** after ~13 s |

- **The burst conclusion was wrong for the GPU.** Cold, Mali is slowest; warm (sustained), it's **fastest** — a 3× swing from DVFS ramp-up. The burst never let it clock up.
- **Temps stayed 40–47 °C over 54 s** → this is DVFS warm-up, **not thermal throttling**. A multi-minute run would be needed to expose real thermal limits.
- **The NPU mildly degrades** under sustained tiny-op load (~14% after 13 s) — slowest of the three for sustained decode.

**Caveat (don't over-read this):** it's *one* 2 MB matmul (q4_0). At 0.33 ms that's ~6 GB/s — **dispatch/throughput-bound, not aggregate-bandwidth-bound**. Real full-model decode reads the *whole model* (GBs) per token across many matmuls, which IS bandwidth-bound — there the three units converge toward the shared-DRAM ceiling, so the per-matmul GPU win may not fully translate. But the lesson holds: **measure sustained, not burst**, and the GPU is far more competitive for decode than the burst implied.

### Revised split picture
- **Prefill → NPU** (8.8 ms, 6.5× CPU) — unchanged, decisive.
- **Decode →** sustained, **GPU(Mali) ≈ 0.33 ms beats CPU 0.47 ms**; NPU worst. So a 3-way `offload_op` could even use Mali for decode and NPU for prefill — though full-model bandwidth effects must be measured with a real model (llama-bench) before committing.

## Milestone 13 (2026-06-12): REAL-MODEL llama-bench — the decisive verdict ✅

The microbenchmarks (single matmul, warm/cached/reused) all flattered the accelerators. A real model settles it. `llama-bench`, Qwen2.5-0.5B q4_0, RK3588:

| config | prefill pp128 (t/s) | decode tg32 (t/s) |
|---|---|---|
| **CPU only** | **351** | **56.8** |
| CPU + RKNPU prefill (MIN_M=16) | 339 (*slower*) | 56.8 |
| RKNPU all ops (MIN_M=0) | 330 | 57.0 |
| Vulkan / Mali | 62 | 30.7 |

**Plain CPU wins decisively — both prefill and decode. Neither accelerator helps:**
- **RKNPU prefill offload is *slower* than CPU** (339 vs 351). The synthetic prefill win (Milestone 7/10: 8.8 ms, 6.5× CPU) was an artifact of **reusing one already-uploaded weight**. A real prefill pass touches every weight *once* → cache miss → dequant+upload dominates → no win. Weights can't stay NPU-resident (ctx limit ~150, LRU cap), so there's no amortization.
- **Decode `tg` is identical with RKNPU on or off (56.8).** ggml's scheduler only offloads *large* ops to ACCEL backends, so M≈1 decode GEMVs stay on the CPU regardless of `supports_op` — which is also the right call (NPU/GPU lose decode).
- **Vulkan/Mali is much slower** for the real model (pp 62, tg 30.7) — the per-op dispatch/sync overhead across a real graph swamps the single-warm-matmul advantage seen in the microbench.

### The honest conclusion of the whole effort
A from-scratch `ggml_backend_rknpu` is **real, correct, and works** (Milestones 1–12) — but on real LLM inference on RK3588 it **does not beat the CPU**, and neither does the Mali GPU via Vulkan. Root causes are structural, not bugs:
1. Per-op matmul offload can't keep weights NPU-resident → re-upload dominates (the NPU's advantage needs weights staged once, which doesn't fit).
2. Decode is bandwidth-bound on shared LPDDR; the CPU has the least overhead.
3. The single-matmul/sustained microbenchmarks were misleading vs end-to-end.

**For NPU-accelerated LLMs on RK3588, Rockchip's full-model `librkllmrt` remains the only path that delivers** — it stages the whole model on the NPU once and pipelines it, exactly avoiding the per-op upload problem. A per-op ggml backend is a validated dead end for *speed* on this chip; its only remaining value would be running *open* llama.cpp features the closed runtime lacks, accepting CPU-class speed.

**Lesson (for the wiki): benchmark the real workload.** Burst → sustained → single-matmul → full-model each flipped the conclusion; only the real-model `llama-bench` is ground truth.

## Milestone 14 (2026-06-12): can we *replace* the closed runtime? — head-to-head ✅

The per-op backend is a dead end, and the closed `librkllmrt` does full-model NPU offload. So the real question: is the closed runtime even worth keeping? Apples-to-apples, **same model (Qwen3-1.7B), same board**:

| | `librkllmrt` (w8a8, NPU) | **open llama.cpp (q4_0, CPU)** |
|---|---|---|
| **decode** | 5.5 tok/s | **17.1 tok/s — 3.1× faster** |
| **prefill** | 97 tok/s | 91.5 tok/s — comparable |

**Open llama.cpp on the CPU is ~3× faster at decode and ties prefill** — i.e. the closed NPU runtime offers *no advantage* here, and is actually slower. Why:
1. **Decode is bandwidth-bound** (read the whole model per token from shared LPDDR). The NPU shares that DRAM → no benefit; whoever moves fewer bytes with least overhead wins.
2. **Quant: librkllmrt is locked to `w8a8` (8-bit); llama.cpp uses `q4_0` (4-bit)** → half the bytes/token → ~2× decode. llama.cpp can also go q3/q2 for more; librkllmrt can't go below w4a16.
3. The NPU's only real strength (large-M GEMM) helps *prefill*, which is a tie at this scale and a minor part of interactive use.

### Verdict: replacing the closed runtime is viable — and likely a *win*
Building oRKLLM's serving on **open llama.cpp (CPU, optionally +Vulkan for some ops)** instead of `librkllmrt`:
- **Faster decode** (3× here via q4), comparable prefill.
- **Quant flexibility** (q4/q3/q2/q8 vs fixed w8a8/w4a16).
- **The whole open ecosystem**: EAGLE-3 speculative decoding, every model architecture, grammars/JSON, LoRA, active upstream development — none of which the frozen closed fork offers.
- No closed `librkllmrt` dependency, no per-version reverse-engineering.

Caveats to confirm before committing: (a) test more sizes (a 4B `w4a16` rkllm would narrow the quant gap — match quant for a fair fight); (b) librkllmrt's NPU prefill may pull ahead on long prompts (compute-bound) — measure TTFT on real prompts; (c) the Mali GPU path stays unused (slowest, per Milestones 11–13).

**Bottom line:** the entire NPU thread converges here — for LLM *serving* on RK3588, the NPU (closed runtime or a custom backend) is not the win it appears; **open llama.cpp on the CPU is the stronger, more flexible, fully-open path**, and on this evidence it's faster too. The NPU's niche is narrow (prefill of long prompts), and even there it's contestable.

## Milestone 15 (2026-06-12): the DRAM-throttle confound — fair NPU-vs-CPU at full clocks ✅

Two reviewer challenges (thank you) overturned Milestones 13–14: (1) I had compared `librkllmrt` **w8a8 (8-bit)** to llama.cpp **q4 (4-bit)** — a 2× bytes/token handicap; (2) maybe the NPU wasn't running optimally. A web-search **control metric** settled it: published rkllm on Qwen3-1.7B/RK3588 is **~13.6 tok/s** decode, but oRKLLM measured **5.5** — a 2.5× shortfall. Root cause: not oRKLLM (its `generate_time_ms` is librkllmrt's own perf struct), not NPU cores/clock (3 cores, 1.0 GHz), but the **DMC governor parked DDR at 528 MHz** vs the 2112 MHz max — a 4× bandwidth deficit, which a bandwidth-bound decode feels directly.

Fixed (`dmc` + cpufreq governors → `performance`) and re-measured, **Qwen3-1.7B, same board:**

| | NPU `librkllmrt` w8a8 | CPU llama.cpp Q8_0 | CPU llama.cpp Q4_0 |
|---|---|---|---|
| **decode (tok/s)** | **11.2** | 11.4 | 16.6 |
| **prefill (tok/s)** | **134.7** | 73.0 | 91.0 |

- **NPU decode 5.5 → 11.2** (~2×) purely from unthrottling DDR → bandwidth-bound confirmed, and now in line with the published ~13.6.
- **CPU barely moved** (Q8 11.8→11.4) → the A76 cores are **compute-bound**, not bandwidth-bound, at this size. That asymmetry is why DDR freq helps the NPU but not the CPU.
- **Matched 8-bit: decode is a tie (11.2 ≈ 11.4); NPU wins prefill 1.85× (135 vs 73).**
- CPU's only decode edge is via lower-bit quant (q4 16.6) — a quant lever, not a hardware win; a `w4a16` rkllm would likely retake it.

### Retraction
**M13/M14's "CPU wins decisively, replace the closed runtime" is withdrawn.** At matched quant and full clocks the closed NPU runtime is *competitive on decode and clearly better on prefill* — it is doing its job. The honest verdict on replacing `librkllmrt`:
- **For speed alone: no.** The NPU ties/wins at matched precision and the per-op ggml backend (M1–12) still can't compete (per-op upload). Replacing with CPU llama.cpp would trade away prefill (1.85×) for nothing on decode.
- **The remaining case for an open path is *features/flexibility*, not speed**: quant choice (q2–q8), EAGLE-3, every architecture, grammars, active upstream. Worth it only if those outweigh losing NPU prefill.

### Methodology lesson (the big one)
**Always verify the board isn't power-throttled before benchmarking, and always match quant.** Every decode number across this entire log before M15 was taken at ¼ DRAM clock; the qualitative "decode is bandwidth-bound" held, but the CPU-vs-NPU *ranking* inverted once the confound was removed. A web-search control metric is what exposed it — when your number is 2.5× off the community's for the same model, the bench rig is suspect, not the conclusion.

## Milestone 16 (2026-06-12): the prefill gap — why it's structural, and the KV-cache reality ✅

After M15 (full DDR, matched quant) the *only* remaining gap is prefill: librkllmrt **134 tok/s** vs CPU llama.cpp **~90–95**. Can open llama.cpp close it? Measured, Qwen3-1.7B Q4_0, pp256, full DDR:

| config | prefill (tok/s) |
|---|---|
| CPU only (4×A76 ceiling) | 89.8 |
| CPU, 4 big cores pinned (`taskset -c 4-7 -t 4`) | 94.6 |
| CPU, 6 threads (spills onto A55) | 67.8 ⚠️ |
| CPU + ggml-rknpu prefill offload (MIN_M=16) | 94.8 |
| **librkllmrt (full-model NPU)** | **134** |

**Neither open path reaches it:**
- **CPU is at its silicon ceiling (~94).** The A76 has `asimddp` (dotprod) but **no `i8mm`**; ~94 tok/s is the 4×A76 dotprod-GEMM limit (≈0.3 INT8 TOPS). Thread/quant tricks don't get past it — 6 threads is *slower* (the 4×A55 little cores drag the barrier). The NPU's 134 comes from dedicated 6-TOPS matmul silicon.
- **The per-op NPU backend stays upload-bound (+5% only).** Even at M=256 each weight is dequantized (q4→fp16), transposed, padded and re-uploaded every forward pass; a 1.7B model has ~196 weight matrices vs the 32-slot weight cache → thrash. The matmul is fast; feeding it isn't.
- **librkllmrt wins because weights are NPU-resident in native int8**, staged once at load. Its prefill lead is a direct consequence of full-model offload — exactly what a per-op ggml backend cannot replicate.

### KV cache is NOT compatible across runtimes (corrects the M15 "hybrid" aside)
A tempting hybrid — prefill on librkllmrt, decode on llama.cpp — is **impossible**: the KV caches can't cross.
- Separate runtimes, separate memory: librkllmrt's KV lives in the closed NPU runtime (`.rkllmcache` format); llama.cpp's is ggml tensors in host/GPU memory. No API bridges them mid-sequence.
- Different numerics/layout: w8a8 + librkllmrt RoPE/attention vs q4_0 + llama.cpp — K/V differ in quant, per-layer/head layout, and values. Byte-incompatible *and* numerically divergent.
- oRKLLM's prefix cache (`src/cache.js`) is `.rkllmcache`-specific; llama.cpp has its own `llama_state_seq_save_file`. The two prefix caches are mutually unreadable.

So the only real cross-runtime hybrid is **per-request engine selection** (route a whole request to one engine — no KV sharing). Intra-llama.cpp NPU-prefill+CPU-decode shares the ggml KV but only buys +5%.

### Verdict — how to address the prefill gap
1. **Eliminate it, don't optimize it (best ROI, already built).** For chat, only the new turn needs prefill; the prefix is unchanged. oRKLLM's prefix KV cache + `prefillAndCache` already skip re-prefill (63–100% reduction). Invest in hit-rate/window, not raw GEMM speed.
2. **Keep librkllmrt for prefill-heavy / long-cold-prompt work.** Its 1.4–1.85× prefill win is real and structural; pair it with CPU llama.cpp for decode/features via per-request routing.
3. **Match it openly only by re-implementing full-model NPU offload** (resident native-int8 weights). Multi-month; rebuilds librkllmrt's core. Narrow niche.

**The prefill gap is the one thing the NPU genuinely does better — and it's structural, not a tuning miss.**

## Next steps (worklist)

- [x] **Memory layer** validated on hardware (Milestone 1, raw DRM `MEM_*`).
- [x] **Matmul compute** validated on hardware (Milestone 2, FP16 via `rknn_matmul_api`).
- [ ] **ggml backend skeleton** — implement `ggml_backend_i` + `ggml_backend_buffer_type_i` (buffers via the Milestone-1 ioctls *or* `rknn_create_mem`) and `GGML_OP_MUL_MAT` via `rknn_matmul_api` (Milestone 2). Register a `ggml_backend_rknpu` device. Needs a ggml/llama.cpp checkout to compile against.
- [ ] **FP16×INT4 weight-quant matmul** (`RKNN_FLOAT16_MM_INT4_TO_FLOAT32`) + the native weight relayout (`rknn_B_normal_layout_to_native_layout`, inner pack C2=16 INT8 / 8 FP16, N→64B align for INT4) — the LLM-relevant path.
- [ ] **Benchmark GEMV (batch-1 decode) vs GEMM (prefill)** on-NPU vs CPU — this is the practicality question marty1885 hit; decides whether the backend helps decode, prefill-only, or neither.
- [ ] **(route 2, no closed lib)** `RKNPU_SUBMIT` smoke test → emit matmul regcmd ourselves (needs the undocumented opcode table + `RKNPU_OFFSET_PC_*`; reuses the Milestone-1 memory layer).

## Sources

- rknpu DRM driver + uapi: https://github.com/rockchip-linux/kernel/tree/develop-6.1/drivers/rknpu
- Tomeu Vizoso RK3588 NPU RE / mainline `rocket` driver + Mesa Teflon: https://blog.tomeuvizoso.net , https://docs.kernel.org/accel/rocket/index.html
- Raw-DRM RE PoC: https://github.com/mtx512/rk3588-npu
- Prior ggml RKNPU2 backend (via rknn_matmul_api) + verdict: https://github.com/marty1885/llama.cpp/tree/rknpu2 , https://clehaxze.tw
- `rknn_matmul_api.h`: https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Linux/librknn_api/include/rknn_matmul_api.h
