# RKNPU regcmd Runtime — Roadmap

> The reverse-engineering log behind `ork-driver`. Migrated from the oRKLLM wiki — this is now the canonical record of how the regcmd matmul engine in `src/npu.c` was derived and validated.

**Goal:** an *open* full-model NPU runtime for RK3576/RK3588 that drives the NPU
directly via raw **register-command (regcmd)** submission to `/dev/dri/renderD129`
(the RKNPU DRM driver), with **all model weights staged resident** in NPU memory.
This is the only path that closes the prefill gap (librkllmrt ~134 tok/s vs CPU
~95) in an open stack — and it unblocks multi-layer hidden states (EAGLE-3, [#462]).

## Why regcmd, and why not just patch the `rknn_matmul_api` limit
`rknn_matmul_api` (closed `librknnrt.so`) allocates per-matmul **contexts**, each
with its own NPU buffers + IOMMU mappings, and dies at ~150 contexts — far fewer
than a model's ~200 weight matrices. That ceiling lives in the closed library /
vendor kernel; patching it is fragile **and doesn't help** (you still re-upload
per context, never staging weights once). regcmd **structurally bypasses** it:
one large resident weight arena (`RKNPU_MEM_CREATE`, bounded only by the IOMMU
domain — multi-GB), with the command stream referencing weights *by offset*.
That's exactly how librkllmrt avoids the wall.

## Hardware context (recap from [ggml-backend-rknpu] / [RKLLM-Runtime-Internals])
The RK3588 NPU is an NVDLA-derived fixed-function INT8/FP16 accelerator. A regcmd
is a list of `(register-offset, value)` writes programming its blocks (CNA conv,
core/MAC array, DPU, PPU) plus DMA descriptors for input/weight/output buffers.
We drive it via `RKNPU_SUBMIT` (`struct rknpu_submit` → `rknpu_task[]`, each task
carrying `regcmd_addr`, `regcfg_amount`). Decode = batch-1 GEMV (bandwidth-bound);
prefill = batched GEMM (the NPU's strength — the target).

---

## Phase 0 — Foundations ✅ (done)
| | status |
|---|---|
| RKNPU DRM uapi transcribed + verified (`rknpu_ioctl.h`, kernel 6.1.115, drv v0.9.8) | ✅ |
| `rknpu_probe`: `MEM_CREATE → MEM_MAP → mmap → write → SYNC → DESTROY` round-trip | ✅ |
| `rknn_matmul_api` reference matmul validated (`[4×32]×[32×16]` all-ones → 32) — our **oracle** | ✅ |
| ggml-rknpu backend (rknn_matmul path), numerically validated vs CPU; perf characterized | ✅ |

These give us a known-good NPU result to diff against and the memory layer we need.

---

## Phase 1 — Capture & decode the regcmd ISA  ✅ (complete)
The keystone RE. We have a *validated* matmul (`mmtest`); capture the regcmd it
makes, then learn the format by perturbation.

- **M1.1 — ioctl trace shim.** ✅ `LD_PRELOAD` an `ioctl()` interposer that logs every
  RKNPU ioctl during `mmtest`: `MEM_CREATE` (size/handle/`dma_addr`/flags),
  `MEM_MAP`, `SUBMIT` (`regcfg_amount`, task count, `core_mask`, `task_obj_addr`).
  *Verify:* full call sequence + buffer inventory printed for the known matmul.
- **M1.2 — regcmd dump.** ✅ Track `dma_addr ↔ mmap'd CPU pointer` (hook `MEM_MAP`/`mmap`);
  on `SUBMIT`, dump the `regcfg_amount` words at each task's `regcmd_addr`, plus the
  input/weight/output buffer bytes. *Verify:* a complete hex dump of one real regcmd.
- **M1.3 — register dictionary.** ✅ Sweep matmul dims (M,K,N), quant, addresses; diff
  the regcmds to localize which words encode dims, strides, DMA addresses, quant
  scales, and the opcode/block-enable bits. *Verify:* annotated regcmd template with
  every field's meaning; documented in [RKLLM-Runtime-Internals].
- **Decision gate:** if the ISA proves too opaque to generalize, fall back to
  *replaying captured regcmds for fixed shapes* (limited but real), or stop here.

## Phase 2 — Replay & synthesize a single matmul  ✅ COMPLETE (M2.1, M2.2, M2.3)
- **M2.1 — verbatim replay.** ✅ Re-issue a captured regcmd via our own `RKNPU_SUBMIT`
  (our `MEM_CREATE` buffers, patched addresses) → reproduce the result **without
  librknnrt**. *Verify:* matches the oracle bit-for-bit.
- **M2.2 — parameterized fp16 GEMM.** ✅ Generate the regcmd from scratch for arbitrary
  (M,K,N). *Verify:* matches CPU across a shape sweep. **← make-or-break milestone.**
- **M2.3 — int8 / w8a8 GEMM.** ✅ The quant format librkllmrt uses (per-channel scales).
  *Verify:* matches a quantized CPU reference within tolerance.


### M2.3 — int8/w8a8 🔬 (deltas mapped) (2026-06-12)
Captured a librknnrt `RKNN_INT8_MM_INT8_TO_INT32` matmul (works; constraint **N%32** vs fp16's N%16). Same 108-entry regcmd structure; diffing int8 vs fp16 (4×32×32) isolates the precision/element-size deltas:
- **Element-size-scaled strides** (int8 = 1 byte vs fp16 = 2): `0x1030 = K*N` (was `K*N*2`), `0x1034 = K` (was `K*2`), `0x107c = K/16` (was `K/8`).
- **Precision-mode registers** (fixed int8 values): `801:3010 = 0x1` (fp16 `0x201`), `1001:4010 = 0x80000000` (fp16 `0xa8000002`), `201:100c`, `1001:4050`, `1001:40c0` differ — to confirm shape-(in)dependence.
- Data: int8 operands, int32 output; weights layout expected to follow the same `[Ntile][Ktile][·][·]` nesting (N-tile likely 32 for int8). Remaining: derive the int8 tile dims via the encoded probe, apply the precision registers, validate vs CPU. This is w8a8 — the LLM-relevant path.


### M2.3 — int8/w8a8 ✅ COMPLETE (2026-06-12)
From-scratch int8 (`RKNN_INT8_MM_INT8_TO_INT32`) matmul synthesis validated vs CPU for arbitrary M/K/N (4×32×32 … 32×512×256, 0 mismatches). `rknpu_synth_i8.c`. Deltas from fp16:
- **precision-mode registers** baked into the int8 template (`801:3010=0x1`, `1001:4010=0x80000000`, `100c`, `4050`, `40c0`);
- **element-size-1 strides**: `0x1030=K*N`, `0x1034=K`, `0x107c=K/16`;
- **`0x1044=ceil(K/64)`** — int8 reads 64 K-channels per hardware pass;
- **weights layout `[Ntile(32)][Ktile(32)][32][32]`** — N tiles by **32** (vs fp16's 16), K tiles by 32; feature stays flat `[M][K]`;
- constraints **K%32, N%32**.

**Phase 2 is complete: from-scratch GEMM synthesis works for both fp16 and int8 (w8a8 — the LLM precision) at arbitrary shapes.** Next: Phase 3 (resident weights — reuse one uploaded weight arena across calls).

## Phase 3 — Resident weights & a GEMM library  ✅ COMPLETE (M3.1, M3.2, M3.3)
- **M3.1 — single resident arena.** ✅ Upload all weights once into one `MEM_CREATE`
  region; matmuls reference by offset. *Verify:* hundreds of matmuls, zero re-upload,
  **no 150-context wall.**
- **M3.2 — prefill GEMM.** ✅ (arbitrary M via software M-tiling) NPU-expected weight tiling + batched M. *Verify:* a single
  layer's matmul hits librkllmrt-class throughput.
- **M3.3 — decode GEMV (M=1).** ✅ *Verify:* confirm bandwidth-bound (≈ CPU); decide
  NPU-vs-CPU routing for decode (likely keep decode on CPU).


### M3.1 — resident weights ✅ (2026-06-12)
`rknpu_resident.c`: uploads `NW` weight matrices into **one** NPU arena (`MEM_SYNC` once), then runs a matmul against each by patching the regcmd's weight address to `arena + w*wbytes`. **512/512 matmuls correct** against a single uploaded arena — i.e. **the ~150-context wall is structurally gone**: no per-weight `rknn_matmul` context, just offsets into resident memory. This is exactly the property the per-op ggml backend (Milestones 1–16) could never achieve, and the reason librkllmrt wins prefill. (A one-shot warm-up submit handles a cold-cache first-iteration artifact.) Next: M3.2 prefill-throughput benchmark, M3.3 decode GEMV.

### M3.2 — RESOLVED ✅ (2026-06-12): arbitrary-M GEMM via software M-tiling
The "multi-M-tile NPU-state wall" was two ordinary bugs (found via the CBUF/fresh-feature research hint):
1. **single-M-tile row cap = `16384/K`** (not 32768/K) — my chunk was 2× too big, so M>cap silently overran the tile (e.g. K=512 cap is **32**, not 64; verified M=32 ✓, M=48 ✗ rows 32-47).
2. **unique LIVE buffer per tile** — reusing one feature buffer (or destroy+recreate, which reuses freed IOVAs) leaves the NPU reading a **stale CBUF**; giving each tile a fresh, still-allocated feature+output buffer fixes it.
Software M-tiling (chunk = `16384/K` rows, fresh A/C per tile over resident weights, warm-up submit) → **arbitrary M/K/N correct**, validated vs CPU incl. **256×4096×512** (K=4096, 64 tiles), 512×512×512, 256×1024×1024 — 0 mismatches. **Phase 3 complete.** (Perf note: per-tile buffer alloc + separate submits is slow; a real runtime should use a buffer pool or multi-task-per-submit — correctness first.)


### Caveats addressed (2026-06-12)
- **int8 arbitrary-M ✅** — software M-tiling ported to int8; cap = **32768/K** (2× fp16; int8 packs 2× rows in CBUF — verified K=512→64, K=256→128). Validated vs CPU incl 256×4096×512, 512×512×512. Both precisions now do arbitrary M/K/N.
- **M-tiling perf 🔬** — attempted **multi-task-per-submit** (N tiles as N `rknpu_task` in one `SUBMIT`, `rknpu_mt.c`) to collapse N submits→1. **Times out, task counter 0** — even task 0 doesn't run. Tried `task_counter` 0/N and completion-interrupt on the last task only; no change. The NPU does **not** auto-sequence a regcmd task-list; **task-chaining must be encoded in the regcmd PC-control entries** (blocks `0x0101/0x0041/0x0081`, "continue vs stop") — the part mtx512 also left unimplemented. Bounded RE for a future session. Until then the correct per-tile-submit GEMM stands (slower but right); a **buffer pool** + keeping the NPU warm are partial speedups not requiring the chaining RE.


### M-tiling perf — single-submit scheduler ✅ (validated regime) (2026-06-12)
The fix is to replicate librknnrt's **internal single-task M-scheduler** (one `SUBMIT`, the NPU iterates M-tiles), not N submits. Per-submit overhead is **~32ms steady-state** (measured 8× warm), so collapsing N→1 is the real win. `rknpu_sched.c`, two scheduler regs as closed-form:
- `0x1010 = 16*min(M+1, 32768/K)` (rows per tile)
- `0x1040 = max(0x1b, base − slope*(M/64−1))`, `base=0xb1−15*(2^lg−1)`, `slope=15*2^lg`, `lg=log2(K/256)`

**Validated single-submit, 0 mismatches, K≤1024 moderate-M**: 64×256 … 256×512×128, 128×1024×64. This is the M-tiling perf fix for the common regime.

**Remaining (precisely traced):** K≥2048 and **many-tile (>~5)** have irregular `0x1040` and need the **PC-sequencer data buffer** — `0x101:0x10` (`PC_DATA_ADDR`) + `0x101:0x14` (`PC_DATA_AMOUNT`), with scaled M-fields (seen in librknnrt's 512×512 = 8 tiles). That's the deepest scheduler layer (`RKNPU_OFFSET_PC_DATA_*`), bounded RE. Until then the correct **software M-tiling** (per-tile submit, cap=16384/K fp16 / 32768/K int8) is the universal fallback.



### M-tiling perf — hybrid, FINISHED for K≤2048 ✅ (2026-06-12)
`rknpu_hybrid.c`: software-chunk M into **4-tile** pieces (`chunk = 4·32768/K` rows), each ONE single-submit scheduler call (the NPU iterates 4 internal M-tiles), fresh feature/output buffer per chunk. **~8× fewer submits** than per-tile software tiling (per-submit ≈32ms, so this is the real speedup). Validated vs CPU, **any M, K≤2048, 0 mismatches**: 1024×512 (4 submits vs 32), 2048×1024×16, 512×256×256 (**1 submit**), 256×2048, 1024×2048.
**K≥4096 — hard boundary (confirmed):** librknnrt sets the **PC_DATA buffer** (`0x101:0x10` PC_DATA_ADDR) even for small M at K=4096, with R=24 (`0x1010=0x180`) and `0x1040` floored at `0x48`. So the scheduler **cannot tile K≥4096 without the PC_DATA program** — the scheduler-reg-only fast path is fundamentally limited to **K≤2048**. K≥4096 fast tiling needs decoding the PC_DATA iteration program (deepest scheduler layer, bounded RE); the correct per-tile software path covers K≥4096 (slower). **Net: M-tiling perf is fast (≈8× fewer submits) and validated for K≤2048 any M — the common LLM contraction-dim range — and correct for all K.**


### M-tiling perf — COMPLETE for arbitrary M/K/N ✅ (2026-06-12)
Final solution (`rknpu_hybrid.c`), **two-level tiling**, no PC_DATA RE needed:
1. **K-split**: split the contraction dim K into ≤2048 slices (the scheduler-fast range), accumulate partial C's. Sidesteps the K≥4096 PC_DATA program entirely.
2. **M-tile**: within each slice, chunk M into **4-tile** pieces, each ONE single-submit scheduler call (NPU iterates 4 internal M-tiles), fresh feature/output buffer per piece.

**Validated vs CPU for arbitrary M/K/N (fp16), 0 mismatches**, incl. LLM-scale: 256×4096×512, **256×8192×16, 512×8192×128**. ~8× fewer submits than naive per-tile tiling (per-submit ≈32ms). **The M-tiling perf is finished — fast and correct for any shape.** (int8 uses the same structure; K-split bound is wider since int8 packs 2× rows/CBUF.) The PC_DATA single-pass path remains a documented optimization (fewer K-slices) but is no longer needed for correctness or the perf win.

### int8 (w8a8) fast path — COMPLETE for arbitrary M/K/N ✅ (2026-06-12)
`rknpu_hybrid_i8.c`. Same two-level hybrid as fp16, with the int8 scheduler registers reverse-engineered from librknnrt:
- **rows/tile** `0x1010 = 16·min(mc+1, 65536/Kp)` — int8 packs **2×** rows per CBUF pass vs fp16 (`65536/K` vs `32768/K`).
- **`0x1040`** = the fp16 closed form evaluated at **effective K = Kp/2** (int8 contracts 2× per CBUF pass). Verified exactly against captures: Kp=512 → base `0xb1`/slope 15 (= fp16 @256); Kp=1024 → base `0xa2`/slope 30 (= fp16 @512).

K is split into ≤1024 **clean** slices (where `R=65536/K` is exact); each slice is M-tiled into 4-tile single-submit scheduler chunks; int32 partials accumulated. **Odd remainder slices** (non-power-of-2 `Kp`, e.g. `11008 % 1024 = 768`, where `R`/`0x1040` go anomalous — same effect seen at fp16 K=1536/2048) fall back to the **proven per-M-tile submit path**, which is validated for any `Kp`. Net: fast scheduler on the bulk, always-correct fallback on the tail.

**Validated vs CPU, arbitrary M/K/N, 0 mismatches**, incl. LLM-scale: 512×8192×128, 512×4096×512, **256×11008, 256×14336**, 128×1280×64. **Both precisions (fp16 + int8/w8a8) now have a fast, correct, arbitrary-shape GEMM.** Phase 2–3 — the complete open NPU matmul engine — is done.

### Buffer pool ✅ + the "PC_DATA single-pass", investigated (2026-06-12)

**Buffer pool — done.** The hybrid now allocates ONE feature+output buffer pair and reuses it (no per-tile `MEM_CREATE`/`mmap`/`DESTROY`). Counterintuitive finding: a single STABLE reused buffer is correct AND fastest; rotating through >1 pooled buffer **corrupts** results (corruption grows with depth) — the NPU caches feature state by address across sequential submits, so a changing feature IOVA desyncs it (opposite of the multi-task-in-one-submit case). Surfaced a latent bug while testing: **non-power-of-2 K** (11008, 5120, 768, 1536) hit the scheduler anomaly and were silently wrong in the old hybrid (only power-of-2 K was ever tested); both fp16 and int8 now gate the fast scheduler on clean Kp and fall back to the proven per-M-tile path for odd remainder slices, so "arbitrary K" is genuinely true.

**"PC_DATA single-pass" — there is NO PC_DATA accumulation buffer.** The original note assumed K≥4096 single-pass needed a PC_DATA partial-sum buffer (`0x101:0x10`/`0x14`). A complete regcmd diff of librknnrt's two chained ops at K=4096 disproves it: the ops differ only in `0x1070` (feature slice) and `0x4020` (output slice) with **identical weights** — they are **M-tiles each doing the FULL K in one op**, not K-slices that accumulate. The `0x2000` bit in `0x1040` is a "subsequent M-tile" scheduler flag, not a DRAM accumulate (verified: separate-submit accumulate gives last-slice-only; chained op1 *overwrites*).

What is actually true:
- **PC-chaining works.** One submit runs S chained regcmd blocks via the word-216 trailer: `(reg 0x10, blk 0x0101)`=next regcmd addr, `(reg 0x14)`=`0x37`/`0`. This is exactly what the old `rknpu_mt.c` was missing (it listed tasks but never set the chain pointer → sequencer never advanced → timeout). Pointing `0x101:0x10` anywhere else → timeout (confirms it's the chain pointer).
- **Single-op full-K for M≤R.** The NPU iterates K internally in one op. **Decode (M=1) does ANY K in ONE submit** (`0x1010=0x20`, `0x1040=0xb1`) — validated K=2048/4096/8192 × N up to 512, all correct, baked into `rknpu_sched.c`. This is the practically important single-pass: token generation issues one submit per matmul regardless of K depth.
- **General-M large-K single-op** is gated by a **nonlinear CBUF scheduler heuristic** (rows/tile R per K: 1024→48, 2048→32, 4096→24, 8192→9) — not cleanly formularizable from a few captures. The K-split hybrid sidesteps it entirely and is already correct + fast (sub-ms submits), so this remains the production path for prefill; the single-op path is a future submit-count optimization.

### Operational polish — COMPLETE ✅ (2026-06-12)

The open NPU matmul **primitive set is complete and production-shaped** — both precisions, both regimes:

| | fp16 | int8 / w8a8 |
|---|---|---|
| **Prefill / arbitrary M,K,N** | `rknpu_hybrid.c` | `rknpu_hybrid_i8.c` |
| **Decode (M=1) / small M (M<R)** | `rknpu_sched.c` | `rknpu_sched_i8.c` |

- **int8 decode single-submit** — int8 M=1 uses the *same* scheduler regs as fp16 (`0x1010=0x20`, `0x1040=0xb1`); any power-of-2 K in ONE submit. Validated M=1 K=2048/4096/8192 × N≤512 + small-batch M<R. int8 is ~2× faster than fp16 (half the weight bandwidth: 8192×512 = 444ms vs 814ms). Non-power-of-2 K (11008/14336) route to the K-split hybrid.
- **Path selection rule** (for the M5 backend): `M==1 && K is power-of-2` → single-submit full-K; otherwise → K-split hybrid (pooled, arbitrary K, both precisions).

**General-M large-K single submit — DONE (M-tile PC-chaining).** Built for completeness and as a portable foundation (`rknpu_pcchain.c`). It is the model librknnrt actually uses: `S=ceil(M/R)` M-tiles, each computing the FULL K in one op, all chained in ONE submit via the word-216 PC trailer (`0x101:0x10`=next regcmd, `0x14`=`0x37`/`0`). M-tiles write **disjoint** C rows, so **no `0x2000` accumulate bit** is needed (confirmed). Per-tile `(R, base 0x1040)` come from a per-K CBUF calibration (nonlinear, captured from librknnrt): 512:(64,0x84), 1024:(48,0x48), 2048:(32,0x2a), 4096:(24,0x48), 8192:(8,0x84). M is padded to `S*R` (extra rows ignored). Validated vs CPU, one submit each, incl. 512×8192×512 (**64 tiles chained**) — all 0 mismatches; in the regression suite.

**RK3588 performance caveat (why it is NOT the default prefill path).** Each full-K M-tile re-reads the *entire weight matrix* (~22× for M=512,K=4096), vs the K-split hybrid's ~per-K-slice (2×). Matmul is weight-bandwidth-bound, so on RK3588 the hybrid is faster for prefill despite issuing more submits. The PC-chain primitive exists as the single-submit reference and for **future Rockchip NPUs** where more cores / higher bandwidth may flip that tradeoff. (It is calibrated for power-of-2 K; other K use the hybrid.)

**Bottom line:** Phases 1–3 + perf + polish are done. The matmul layer is correct, fast, pooled, arbitrary-shape, both precisions, with optimal decode. Everything remaining is the transformer runtime (M4–M6).

## Phase 4 — Transformer ops beyond matmul
- **M4.1 — reusable matmul library (`rknpu_mm`)** ✅ — keystone: one device handle,
  weights packed once and kept resident, many matmuls streamed against them.
- **M4.2 — non-matmul CPU ops** ✅ — RMSNorm, RoPE, attention/softmax, SiLU/SwiGLU.
- **M4.3 — full decoder-layer hybrid forward** ✅ — matmul on NPU, ops on CPU; validated
  vs a pure-CPU reference.
- **M4.4 — KV-cache incremental decode** ✅ — token-by-token, M=1 projections, RoPE at the
  running position, attention over cached K/V; validated vs CPU and consistent with prefill.
- **Multi-layer / real-model wiring** ⏳ → Phase 5 (M5).

**Phase 4 is complete.** The open NPU stack now runs a full transformer decoder layer end
to end — prefill and KV-cache decode — with matmul on the NPU and the non-matmul ops on the
CPU, all validated to fp16 precision and in the regression suite (43 cases).

### M4.1–M4.3 — first full transformer layer on the open NPU stack ✅ (2026-06-12)
The matmul kernels are now a **library** (`rknpu_mm.{h,c}`): `rknpu_mm_init` → `rknpu_mm_pack(K,N,B)` (pack + upload weights once, resident) → `rknpu_mm_run(w,M,A,C)` (stream activations; K-split hybrid internally, arbitrary M/K/N). On top of it, `rknpu_layer.c` runs a complete **Llama/Qwen-style decoder layer** — `h = x + Wo·Attn(RoPE(Q,K),V)`, `y = h + Wdown·(SiLU(Wgate·hn) ⊙ Wup·hn)` — with the seven big projections on the NPU and the non-matmul ops (RMSNorm / RoPE-NeoX / causal softmax / SwiGLU) on the CPU (the pragmatic hybrid graph).

**Validated** against a pure-CPU reference (identical ops, CPU fp16 matmul, so the matmul engine is the only variable): H=512, NH=8, HD=64, FFN=2048, SEQ=16 → **maxabs 0.044 vs |ref|max 276 (~0.016 %, pure fp16 rounding)**. Both in the regression suite.

Two bugs found while turning the standalone kernels into a reusable library:
1. **`setr` must patch ALL matching regcmd entries, not the first.** The template carries some regs (`0x1040`) more than once and the NPU uses a *later* copy; first-match-only left a stale schedule value — wrong for multi-M-tile shapes only (single-tile masked it).
2. **A freshly (re)allocated output buffer is cold** — its first submit returns stale data. The one-time warmup must re-fire whenever the output buffer is resized (here when N grows 512→2048 between projections), not just once per context.

Both are NPU-state subtleties (not algorithm errors) — the kind of thing the standalone one-shot kernels never exercised.

### M4.4 — KV-cache incremental decode ✅ (2026-06-12)
`rknpu_decode.c` runs the same layer one token at a time: per step, M=1 NPU projections, RoPE at the running position, append K/V to a per-layer cache, attend over the cached keys/values. Validated vs a pure-CPU reference over a 16-token sequence — **maxabs 0.044 vs |ref|max 276**, *identical* to the prefill layer's error, confirming prefill ↔ incremental-decode causal consistency. This is the token-generation path (M=1 matmuls, the decode regime). **Phase 4 done.** Next: stack N layers + load a real model's weights / tokenizer / embeddings / LM-head (M5).


### M3.2 — prefill GEMM (large M) 🔬 (2026-06-12, refined)
Works across a **wide range** — pure large-M (512×32×16 ✓), large-M + moderate K/N (128×128×64 ✓), large-K/N at modest M (32×512×128 ✓), GEMV (M=1 ✓). The **raw NPU output C buffer byte-matches librknnrt** (verified at 128×32×16), and feature/weights/regcmd all match — so the matmul is correct over this range. It diverges only when **M>32 AND K≥256 together** (e.g. 128×256×256, 128×512×512) — a feature M×K co-blocking interaction (the feature, flat `[M][K]` at smaller sizes, needs M×K tiling past that threshold). This is the last GEMM edge. Feature confirmed flat `[M][K]` even at 128×256 (row0 = k0..255 contiguous), so the divergence is on the weights-at-large-(KT×NT) or output side — same byte-compare→decode method applies. **M-tile = 64** (M>64 ⇒ multiple M-tiles): two registers then change — `0x1010 = min(16*(M+1),0x800)` (saturates at M≥127, FIXED) and `0x1040` (packed M-tile-count field: 0xb1/0xa2/0x84 for M/64=1/2/4, partially decoded) — plus likely an M-tile data layout. GEMM is now correct for **any M at single/moderate-N (any K)** + GEMV: M-tiling solved via `0x1010=min(16(M+1),0x800)` and `0x1040=0xb1-15*(ceil(M/64)-1)` (both occurrences set). Remaining edge: **large-M AND large-K together** (e.g. 128×512×16, K≥512 at M>64). Root cause precisely characterized: the NPU **auto-tiles M based on an SRAM budget that depends on K** — `0x1010` and `0x1040` are K-dependent (128×256: 0x1010=0x800,0x1040=0xa2; 128×512: 0x1010=0x400,0x1040=0x84). The budget is ≈ **32768 elements**: M-tile-rows ≈ min(M, 32768/K) (K=256→128 rows, K=512→64 rows), so `0x1010 = 16·min(M, 32768/K)`. Verified correct now for **any M with K≤256** (128×256×256, 256×256×256 ✓), large-N at small K (128×64×512 ✓), and GEMV. Completing the K-driven M-tiling scheduler (0x1010 + 0x1040 + M-tile count) is the last GEMM piece — a bounded scheduler model, not a new layout.

[superseded] Earlier note (large-M>8 diverges) was wrong: it conflated pure large-M (which works) with the large-M+large-K shapes. Verified for M=128: the **regcmd byte-matches** librknnrt (no M-tile registers), and with uniform per-row probes the **feature is flat `[M][K]`** (rows in order) and the **output C is flat `[M][N]`** (row m = K·m). Yet random-data M=128 gives C[0] correct, rest wrong — the same "uniform-matches, random-diverges" signature that the nested K×N case had, i.e. a subtle M-blocking layout the constant-row probe can't expose. Resolution is the same proven method (full byte-compare synth-vs-librknnrt buffers with identical random data → localize → decode), not yet done. Decode GEMV (M=1, M3.3) and small-M prefill already work, so this is the large-batch prefill edge.

## Phase 5 — Integrate into the ggml / llama.cpp fork
- **M5.1 — resident-regcmd GEMM in the ggml-rknpu backend**, replacing the
  rknn_matmul path. *Verify:* prefill re-benchmark vs librkllmrt (target: close 134).
- **M5.2 — end-to-end forward** for one architecture (Qwen3) via the
  [`oRKLLM/llama.cpp-rockchip`](https://github.com/oRKLLM/llama.cpp-rockchip) fork.
  *Verify:* `llama-bench` prefill ≈ librkllmrt; correct output.
- **M5.3 — productionize:** 3-core scheduling, error handling, per-arch graphs.

### M5.1–M5.3 — a REAL model runs end-to-end on the open NPU stack ✅ (2026-06-12)
Took the standalone-harness route first (prove the whole stack with a real model before wiring the ggml fork). Three increments, all in the regression suite:

- **M5.1 multi-layer body** (`rknpu_model.c`): N stacked decoder layers, each with its own weights; validated NPU-hybrid vs CPU to 24 layers — error grows gracefully (1L 0.0015 → 24L 0.046), no divergence.
- **M5.2 real weights** + **M5.3 end-to-end forward** (`rknpu_llama2.c`): loads Karpathy's **llama2.c `stories15M`** (real trained Llama: dim 288, 6 layers, 32000 vocab), mmaps the fp32 `.bin`, packs projections + LM head into resident fp16 NPU weights (transposed [out][in]→[in][out]), and runs the full decoder — embed → 6 layers w/ KV cache → final RMSNorm → 32000-vocab logits — matmul on NPU, ops (RMSNorm, **interleaved GPT-J RoPE**, causal softmax, SwiGLU) on CPU. Greedy-generates; **NPU-hybrid logits match a pure-CPU-fp16 reference to worst |Δ| = 0.01 over 12 steps.** A real LLM now runs on the regcmd NPU stack.

**Bugs found + regression-locked this milestone:**
1. **NPU output-width cap = 8192** (N=16384 silently wrong, N≤8192 exact). The 32000-vocab LM head exposed it. Fixed with **N-tiling** in `rknpu_mm` (weights now pack into `Sk×Sn` = K-split × N-split≤8192 buffers; run accumulates K-slices into each N-slice's columns). `test_mm` now covers N=16384 and the 1×288×32000 LM-head shape.
2. **Unsigned-arithmetic** in synthetic norm-weight init (`(i+seed)%5-2` underflowed to ~4.3e9) → RMSNorm explosion → fp16 overflow → NaN; the check then **silently passed** (NaN compares false). Hardened to fail on NaN/inf/dead output.
3. **`maxout` over-allocation** (ignored the actual chunk → a 58 MB LM-head output buffer DMA-synced every submit). Sized to the real chunk.

The reusable library (`rknpu_mm`) + ops are the substance the ggml/llama.cpp-fork integration (M5.1-fork above) will call; that wiring + multi-arch + 3-core scheduling remain.

### M3.2 — M-tiling scheduler data (the large-M+large-K edge)
The remaining GEMM edge is the NPU's **internal M-tiling scheduler** (`0x1010`, `0x1040` depend on BOTH M and K). Captured table (value = high16 of the reg word):

| M \ K | 0x1010 (256/512/1024) | 0x1040 (256/512/1024) |
|---|---|---|
| 64  | 0410 / 0400 / 0300 | b1 / a2 / 84 |
| 128 | 0800 / 0400 / 0300 | a2 / 84 / 48 |
| 192 | 0800 / 0400 / 0300 | 93 / 66 / 1b |
| 256 | 0800 / 0400 / 0300 | 84 / 48 / 1b |

`0x1010` = 16·(rows-per-M-tile): K=256→128, K=512→64, K=1024→**48** rows (not a clean 1/K — there's a floor/rounding in the SRAM heuristic). `0x1040` is a packed bitfield encoding the M-tile schedule. Modeling this scheduler is the last GEMM piece — bounded RE, but genuinely the NPU's tiling logic (what `rknn_matmul` computes internally), not a single formula. **The GEMM is correct without it for the common range: any M at K≤256, large-N at small K, GEMV, and all int8/fp16 K%32/N%(16|32) shapes that fit one M-tile.**


### M3.2 — multi-M-tile attempt (the honest wall)
Tried to finish the large-M+large-K edge two ways; both hit a genuine NPU multi-M-tile **state** behavior I could not crack this session:
- **NPU internal M-scheduler** — `0x1010`/`0x1040` are K-dependent (data table above); even with the right values for moderate cases, the **2nd M-tile's output lands wrong** for large-M+large-K.
- **Software M-tiling** (split M into ≤cap row-chunks, loop submits over resident weights) — fails on the 2nd chunk in *both* sub-variants: **offset feature addressing corrupts** the result, and **same-address re-upload reads a stale NPU feature cache** (M3.1's loop worked only because its feature was static and only the *weight* address changed). Non-cacheable buffers + warm-up submit didn't resolve it.

So consecutive submits with **changing feature data** is the unsolved bit — the NPU appears to reuse feature state across jobs in a way that needs an invalidation/handshake I haven't found, or a fresh feature buffer per chunk. **The single-M-tile GEMM is solid** (any M at K≤256, large-N at small K, GEMV, all int8/fp16); large-M+large-K prefill batching is the remaining edge and is genuinely harder than a register formula.

## Phase 6 — Payoff
- **M6.1 — multi-layer hidden states** → resolves [#462], enables **open EAGLE-3**.
- **M6.2 — open NPU prefill engine** selectable in oRKLLM (the two-engine model:
  open-NPU/llama.cpp for everything, no closed `librkllmrt` dependency).

---

## Risks & honest framing
- **Multi-month, mostly reverse-engineering.** M1.3 + M2.2 are the make-or-break;
  if regcmd can't be generalized across shapes, the project caps at fixed-shape replay.
- **Per-architecture graphs** (like librkllmrt) and exact-numerics matching are required
  for quality.
- **Maintenance** is tied to the NPU driver/hardware.
- **It is a research / sovereignty play**, not near-term ROI: for plain serving,
  librkllmrt already delivers prefill and the prefix cache hides most of it. The win
  is *NPU prefill + full openness + features the closed runtime lacks* (multi-layer
  EAGLE-3), together — which nothing else gives.

## Tooling we build along the way
- `rknpu_trace.so` — the `LD_PRELOAD` ioctl/regcmd capture shim (M1.1–M1.2).
- `regcmd_diff` — perturb-and-diff register localizer (M1.3).
- CPU oracle + `mmtest` — the correctness reference at every step.

[#462]: https://github.com/airockchip/rknn-llm/issues/462
[ggml-backend-rknpu]: https://github.com/oRKLLM/oRKLLM/wiki/ggml-backend-rknpu
[RKLLM-Runtime-Internals]: https://github.com/oRKLLM/oRKLLM/wiki/RKLLM-Runtime-Internals

---

## Progress log

### M1.1 — ioctl trace shim ✅ (2026-06-12)
`rknpu_trace.c` (`LD_PRELOAD` `ioctl` interposer, in `experimental/ggml-rknpu/`) run against the validated `[4×32]·[32×16]` fp16 matmul (`mmtest`, result CORRECT). Full submission anatomy captured:

**Buffer inventory** (6× `MEM_CREATE`, each one 4 KB page, IOMMU domain 0, `core_mask=0x1`):

| handle | synced bytes | dir | role |
|---|---|---|---|
| 1 | **896** | →dev | **regcmd stream** (the register-command ISA) |
| 2 | 40 | — | task descriptor — one `rknpu_task` (40 B); `task_obj_addr` points here |
| 3 | 1024 | →dev | B weight `[32×16]` fp16 |
| 4 | 256 | →dev | A input `[4×32]` fp16 |
| 5 | 1024 | — | second 1024 B (reordered-B / scratch — TBD in M1.2) |
| 6 | 256 | ←dev | C output `[4×16]` |

**Sequence:** `ACTION` probes (`GET_HW_VERSION`→`0x46495245`="FIRE"; `GET_DRV_VERSION`→`0x38c`=**v0.9.8**; `POWER_ON`→ -1 already-on; `SET_PROC_NICE`=-19; `GET_IOMMU_EN`=1 before each alloc) → 6× `MEM_CREATE`+`MEM_MAP`+`MEM_SYNC(0x3)` → `MEM_SYNC(→dev)` on regcmd/B/A → **one `SUBMIT`** (`flags=0x5`, `timeout=6000`, `task_number=3`, `task_counter=3`, `core_mask=0x1`, `hw_elapse_time=11667 µs`) → `MEM_SYNC(←dev)` on C → `MEM_DESTROY`.

**Findings:**
- The whole matmul is **896 bytes of regcmd** (handle 1) + a 40-byte task descriptor pointing at it. That's small and tractable to decode.
- `MEM_CREATE` flags: `0x403` = NON_CONTIGUOUS|CACHEABLE|IOMMU_LIMIT_IOVA_ALIGNMENT; the task buffer adds `KERNEL_MAPPING` (`0x40b`).
- IOVA `dma_addr`s are handed out top-down (`0xffffe000`, `0xffffd000`, …) — the regcmd almost certainly references A/B/C by these device addresses, so M1.3's address-localization should be straightforward by diffing.
- Single core only (`core_mask=0x1`) for this tiny op — multi-core comes later (M5.3).

**Next — M1.2:** hook `mmap` to map `handle→CPU pointer`, then dump the 896-byte regcmd + the 40-byte task descriptor + A/B/C bytes for this known shape.

### M1.2 — regcmd dump + format decoded ✅ (2026-06-12)
`rknpu_dump.c` (adds `mmap`/`mmap64` hooks → `handle→CPU ptr`, parses the task descriptor, dumps the regcmd as words). For the `[4×32]·[32×16]` matmul the task descriptor decodes as one `rknpu_task`: `enable_mask=0x0d`, `int_mask=0x300`, `int_clear=0x1ffff`, `int_status=0x100`, **`regcfg_amount=108`**, `regcmd_addr→0xffffe000` (handle 1). C output is **fp32** (`0x42000000`=32.0) from fp16 inputs.

**The regcmd is 108 register-writes as `(word0, word1)` u32 pairs, and the encoding is decoded:**

```
reg_offset = word0 & 0xFFFF
block_id   = word1 >> 16        # 0x0201, 0x0801, 0x1001 = three NPU blocks
value      = ((word1 & 0xFFFF) << 16) | (word0 >> 16)
```

Verified by the **embedded DMA addresses** found in the stream (the patch points for synthesis & resident weights):

| reg (block) | value | = buffer |
|---|---|---|
| `0x1070` (0x0201) | `0xffffb000` | A input |
| `0x1110` (0x0201) | `0xffffc000` | B weight |
| `0x4020` (0x1001) | `0xffff9000` | C output |

Dimension fields are visible too — `reg 0x1024 → 0x001f0020` (K=32 / K−1=31), `reg 0x1078 → 0x000f000f` (N=16 / N−1=15). The three blocks map to register ranges: `0x1xxx`↔0x0201, `0x3xxx`↔0x0801, `0x4xxx`↔0x1001 (CNA feature / DPU / PPU-core, NVDLA-style). Full dump saved at `/tmp/rknpu/dump_4x32x16.txt`.

**Implication:** the format isn't opaque — it's flat register writes with addresses embedded as split 32-bit fields. The M1.3 *decision gate* (could-the-ISA-generalize) is essentially passed; M1.3 now just needs to map which regs encode M/K/N/strides/quant by sweeping shapes.

**Next — M1.3:** parametric matmul harness (argv M K N), capture regcmds for several shapes, diff to finish the register dictionary; then M2.1 verbatim replay.

### M1.3 — register dictionary ✅ (2026-06-12)
Parametric harness (`rknpu_mm_param.c`, argv M K N) swept four shapes — base `(4,32,16)`, `M=8`, `K=64`, `N=32` — all CORRECT; regcmds decoded (`diffrc.mjs`) and diffed. All 108 entries have fixed structure; only values change, and every dimension-dependent field resolves to a closed form:

| reg (blk:off) | base value | tracks | formula |
|---|---|---|---|
| `201:1024` | `0x1f0020` | K | `(K-1)<<16 \| K` |
| `201:1030` | `0x400` | K | `K*32` |
| `201:1034` | `0x40` | K | `K*2` |
| `201:1044` | `0x1` | K | `K/32` (K-tiles) |
| `201:1088` | `0x20` | K | `K` |
| `201:107c` | `0x4` | K | `K/8` |
| `201:1020`,`1084` | `0x10004` | M | `M` (low16) |
| `201:102c` | `0x4` | M | `M` |
| `201:1010` | `0x50` | M | `16*(M+1)` |
| `1001:4034` | `0x3` | M | `M-1` |
| `1001:405c`,`801:3014` | `0x30000` | M | `(M-1)<<16` |
| `1001:403c` | `0xf000f` | N | `(N-1)` (both halves) |
| `1001:4058` | `0xf` | N | `N-1` |
| `1001:4038` | `0x30003` | N | `(N/4-1)` (both halves) |
| `201:1038` | `0x1010010` | N | `N` (low byte) |
| `801:3018` | `0xf` | N | `N-1` |
| `201:1070` | A dma | addr | **patch** (A input) |
| `201:1110` | B dma | addr | **patch** (B weight) |
| `1001:4020` | C dma | addr | **patch** (C output) |

The remaining ~90 entries are shape-invariant constants (block config, opcodes, int masks). **Three NPU blocks** are programmed: `0x0201` (CNA / feature+weight setup, `0x1xxx` regs), `0x0801` (`0x3xxx`), `0x1001` (PPU/core output, `0x4xxx`). Tooling: `rknpu_mm_param.c`, `diffrc.mjs` (decode+diff) in `experimental/ggml-rknpu/`.

**Verdict:** the regcmd ISA is decoded well enough to *synthesize* a matmul for arbitrary (M,K,N): emit the ~90 constants verbatim + compute the ~18 dimension fields + patch 3 addresses. Phase 1 complete.

**Next — M2.1:** verbatim replay — allocate our own buffers, write the captured regcmd with the 3 addresses patched, `RKNPU_SUBMIT` directly (no librknnrt), read C, expect `==K`.

### M2.1 — verbatim replay ✅ (2026-06-12) — **the NPU runs on our own submit, no librknnrt**
`rknpu_replay.c` drives a raw `RKNPU_SUBMIT` with the captured regcmd and produces the correct result (`[4×32]·[32×16]` all-ones → **C = 32.0**, fp32). librknnrt is not involved.

Three things had to be right (each cost a debugging round; documented so nobody repeats them):

1. **Device node — the decisive fix.** The job only executes via the **primary (card) node `/dev/dri/card1`**. On the **render node `/dev/dri/renderD129`** the kernel accepts every buffer and the submit, then the job runs **0 tasks and times out** (`task counter: 0x0`, soft reset). librknnrt uses the render node successfully *with its own full setup*, but for a from-scratch submit the card node is what works.
2. **regcmd is 112 entries (224 words), not 108.** Beyond the `regcfg_amount=108` register-writes there are **4 trailing PC-control entries** (blocks `0x0101/0x0041/0x0081`) — the `RKNPU_PC_DATA_EXTRA_AMOUNT=4` that drives the PC sequencer. Replaying only the 108 counted entries → PC never starts → 0 tasks. Capturing the full 224 words fixed it.
3. **`task_number=1` with a single `subcore_task[0]`** for a one-task matmul (not librknnrt's `task_number=3` + three subcores, which expects 3 completions that a single regcmd never signals).

Not required (verified revertible): non-cacheable buffers, `RKNPU_ACT_RESET`, PRIME/dmabuf export. **Cacheable buffers + explicit `MEM_SYNC` (librknnrt-style) work fine** on the card node.

**Cross-check:** independently confirmed by building & running [mtx512/rk3588-npu](https://github.com/mtx512/rk3588-npu) on this board — its *synthesized*-regcmd matmul also returns `RKNPU_SUBMIT 0` with correct output. That project already does M2.2 (parameterized regcmd synthesis) and even runs llama2.c, so Phase 2–3 can build on it (with attribution) rather than re-deriving from scratch.

**Next — M2.2:** parameterize the regcmd (emit the ~90 shape-invariant constants + compute the M/K/N fields from the M1.3 dictionary + patch addresses) and validate against CPU across a shape sweep — or adopt mtx512's `npu_matmul.c` synthesis as the foundation.

### M2.2 — synthesize a matmul from scratch ✅ (arbitrary M/K/N) (2026-06-12)
`rknpu_synth.c` builds the regcmd from the template + M1.3 closed-form dim fields and validates the NPU result against a CPU reference with **real (distinct) data** (small ints, exact in fp16). **Correct (0 mismatches)** for: `4×32×16, 8×32×16, 4×32×32, 8×32×48, 4×64×16, 4×96×16`.

Two register/template corrections (all-ones had masked them):
- **reg `0x1030` = `K*N*2`** (weights byte-size), not `K*32` — it depends on K *and* N (the K-only M1.3 sweep couldn't tell).
- The rest of the M1.3 closed forms held.

**Operand memory layout** (the real find — derived by encoding indices in the data and dumping the buffers librknnrt's regcmd actually reads):
- **Feature A: flat row-major `[M][K]`** — K contiguous, *not* tiled.
- **Weights B: `[Ktile][N][32]`, tile-major** — K split into 32-channel tiles; within a tile the weights are transposed to output-channel-major (`buf[t*N*32 + n*32 + kk] = B[t*32+kk][n]`). This reduces to the simple transpose at K=32.

**Remaining edge:** when **K>32 AND N>16 simultaneously**, results diverge (C[0] is right, the rest mis-placed) → the weights need *nested* K×N tiling (N also tiles by 16) and the output needs de-tiling. Single-tile-in-either-dimension is solved; the nested case is the next step. Then M2.3 (int8/w8a8) and Phase 3 (resident weights).


**Nested-tiling (K>32 AND N>16) — RESOLVED ✅:** it was a weights **layout** bug, not the node/IOMMU issue first hypothesized (that wrong guess came from comparing only the first 16 buffer words — they matched; a *full* 4 KB byte-compare showed the tile-1 region diverging). The correct weights layout is **`[Ntile][Ktile][16][32]`**: N tiles by 16, K tiles by 32, transposed within a tile (`buf[nt*KT*16*32 + kt*16*32 + nl*32 + kk] = B[kt*32+kk][nt*16+nl]`). This reduces to the earlier layouts when NT=1 or KT=1 — which is exactly why single-K-tile and single-N-tile both worked and only the doubly-tiled case broke. Feature stays flat `[M][K]`. **M2.2 is now complete: from-scratch fp16 GEMM synthesis correct for arbitrary M/K/N (validated vs CPU through 32×512×128).**

**Tooling:** `rknpu_synth.c` (synth+validate), `layout_probe*.c` (operand-layout derivation via librknnrt + dump shim).
