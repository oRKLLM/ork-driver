# Training a 0.75B hybrid SLM on RK3588 NPUs via ork-driver — scoping report

**Date:** 2026-10-05 · **Board:** RK3588 `.236`, kernel `6.1.115-vendor-rk35xx-fence` · **Library:** ork-driver @ `81428a7`
**Benchmarks:** `tools/train_scope_bench.c`, `tools/train_dw_bench.c` (this branch; self-validating against fp64)

Every figure is tagged **[M]** measured on this board today, **[D]** derived from [M] by arithmetic, **[S]** source-read, or **[E]** estimated.

---

## 0. Verdict first

**No-go for NPU-side training of this model on this hardware, by ~20–45×.** The stop condition in the brief was reached, so deliverables 6 (multi-board) and parts of 2 are scoped rather than measured — detail in §7.

The cause is not the one the brief anticipated. It is not re-pack cost, and it is not fp16 numerics. **It is that ork-driver's fp16 fast path requires the matmul's contraction dimension to be a power of two below 2048** — and in a transformer backward pass, most contraction dims are neither.

| | |
|---|---|
| FFN-only projection, best config | **~82 tok/s/board [D]** |
| With GDN backward, attention, optimizer | **~30–50 tok/s/board [E]** |
| Reference (M5 Max, MLX, same model) | 1,400 tok/s |
| Boards needed to match one M5 Max | **~30–45, before fabric overhead [D]** |

---

## 1. The governing constraint

`src/npu/f16/regcmd.c` **[S]**:

```c
int orki_f16_sched(int K){ return (K&(K-1))==0 && K>=128 && K<2048; }

int orki_f16_mcap(int K,int sched){
    if(!sched)                        cap = 16384/K;    /* 1 CBUF bank  */
    else if(K==256||K==512||K==1024)  cap = 180224/K;   /* 11 banks — measured exact */
    else                              cap = 16384/K;
}
```

The 11-bank schedule — 11× the rows per program — is reachable **only when the contraction dim is a power of two in [128, 2048)**. Everything else gets one bank. Map that onto a transformer FFN with `d=1024, inter=3584`:

| matmul | contraction | sched | mcap | measured |
|---|---|---|---|---|
| fwd gate/up `[M,1024]×[1024,3584]` | 1024 ✅ | 1 | 176 | **907 GFLOP/s [M]** |
| fwd down `[M,3584]×[3584,1024]` | 3584 ❌ | 0 | **4** | **75.7 GFLOP/s [M]** |
| dX for down | 1024 ✅ | 1 | 176 | ~599–907 **[M]** |
| dX for gate/up | 3584 ❌ | 0 | **4** | ~75.7 **[M]** |
| **dW (any layer)** | **= token count** | depends on batch | | 104–122 **[M]** |

**12× spread between the fast and slow path, and the slow path covers the majority of backward work.**

### 1.1 The cap is real, not conservatism — tested

The comment `"unmeasured sched=1 K: stay on the 1-bank bound"` suggested headroom. There is none. Scanning `ORK_F16_MTILE` upward at K=3584 (upward, never bisecting — the predicate is non-monotonic) **[M]**:

| mcap | GFLOP/s | max rel err | |
|---|---|---|---|
| 4 (default) | 75.7 | 0.00e+00 | correct |
| 8 | 68.7 | 0.00e+00 | correct, no faster |
| 16 | 296.2 | **2.18e+01** | **miscompute** |
| 32 | 437.3 | **2.11e+01** | **miscompute** |

The arithmetic agrees: one CBUF bank is 32768 B, a K=3584 fp16 row is 7168 B, so **4.57 rows fit**. `mcap=4` *is* the hardware capacity. The 5.8× available at mcap=32 is unreachable because it computes the wrong answer.

> Do not "optimise" this cap. It is load-bearing, and the failure is silent — a 2100% error with no fault, no hang, and no log line.

---

## 2. Deliverable 1 — operation inventory

| op | forward | backward | status |
|---|---|---|---|
| Linear `X·W` | `ork_f16_mm_pack`/`run` **[S]** | — | **NPU** |
| `dX = dY·Wᵀ` | — | needs Wᵀ packed | **NPU + CPU transpose** (no transposed-B primitive) |
| `dW = Xᵀ·dY` | — | needs Xᵀ materialised + dY packed | **NPU + CPU transpose + per-step pack** |
| Attention fwd | `ork_bmm_fp16_strided` **[S]** | — | **NPU** (strided operands avoid `ggml_cont`) |
| Attention bwd | — | missing | **write** |
| GDN recurrence fwd | — | — | **CPU** — on-NPU GDN scan measured to *lose* 1.9× @0.8B / 4.8× @9B, gap widening |
| GDN recurrence bwd | — | missing | **write, CPU** — expected dominant cost |
| RMSNorm | `ork_f16_npu_rmsnorm` **[S]** | missing | fwd NPU / bwd **write** |
| SwiGLU | `ork_f16_mm_run_silu`, `ork_f16_npu_ewmul` **[S]** | missing | fwd NPU / bwd **write** |
| Softmax | `ork_f16_npu_softmax` **[S]** | missing | fwd NPU / bwd **write** |
| Cross-entropy | missing | missing | **write, CPU** |
| KL term | missing | missing | **write, CPU** |
| Adam8bit | missing | — | **write, CPU** |
| MoE router + gather/scatter | partial (inference) | missing | **write** |

`grep -rinE "backward|gradient|_grad|adam|optimiz" include/` returns **nothing** **[S]**. There is no autograd, no backward op, and no optimizer. Every backward entry above is new code.

**Two primitives that do more than expected**, both already present:

- **`ork_f16_mm_repack(ctx, w, K, N, B)`** — re-tiles into an *existing* `ork_w`, no bcreate/bfree. The allocate-once/refresh-per-step pattern the brief asks about **already exists** as API.
- **`ork_bmm_fp16_strided`** — explicit element strides, so `dW = Xᵀ·dY` is expressible as `as_m=1, as_k=K` without a caller-side transpose. **But** the doc states strided operands are "gathered into contiguous scratch, then the dense pack/run path runs unchanged" **[S]** — so the transpose and the pack both still happen, just inside the library.

---

## 3. Deliverable 3 — the weight-update problem (measured)

Pack cost is independent of M (it is K×N work), which made this the cheapest decisive test.

| weight | bytes | pack µs **[M]** | repack µs **[M]** | pack GB/s **[D]** | repack GB/s **[D]** |
|---|---|---|---|---|---|
| `[1024,3584]` | 7.34 MB | 11,928 | **5,491** | 0.62 | **1.34** |
| `[3584,1024]` | 7.34 MB | 13,776 | **5,932** | 0.53 | **1.24** |
| `[1024,1024]` | 2.10 MB | 3,688 | **1,089** | 0.57 | **1.93** |

**`repack` is 2.3–3.4× cheaper than `pack`** **[D]** — the MEM_CREATE/MEM_DESTROY round trip is roughly two-thirds of a pack, and avoiding it is free. **Use `ork_f16_mm_scratch` once per weight at init, then `ork_f16_mm_repack` every step.** Never pack-and-free in a training loop.

**But re-packing is not the bottleneck.** `repack/run` is 0.076–0.17 **[M]** — under 17% of the matmul it enables. The brief's hypothesis (and the sibling session's) that re-pack cost decides feasibility is **not supported**: it is real, bounded, and the smallest of the three overheads.

IOVA: the model's resident fp16 weights are ~1.5 GB **[D]** for 0.75B params, comfortably inside the measured ~3.9 GiB ceiling, so **IOVA is not a constraint at this model size** — one domain suffices, which also avoids the multi-domain defect class entirely.

### 3.1 What *is* expensive: the CPU transpose

| | µs **[M]** | share of dW |
|---|---|---|
| CPU transpose of X `[4096,1024]` | **49,539** | 34% |
| pack dY | 12,671 | 9% |
| NPU dW matmul | 82,095 | 57% |
| **total per dW** | **144,305** | |

A single strided fp32→fp16 transpose costs **49.5 ms** — more than a forward matmul. There is no NPU transpose primitive, and both backward matmuls need one. At ~100 weight matrices this is ~2–5 s/step of pure CPU work **[E]**.

---

## 4. Deliverable 4 — microbenchmarks

`M = 4096` (batch 4 × seq 1024), 3 reps, fp64-checked **[M]**:

| shape | pack µs | repack µs | run µs | **GFLOP/s** | repack/run |
|---|---|---|---|---|---|
| `[M,1024]×[1024,3584]` | 11,928 | 5,491 | 33,150 | **907** | 0.17 |
| `[M,3584]×[3584,1024]` | 13,776 | 5,932 | 396,903 | **75.7** | 0.015 |
| `[M,1024]×[1024,1024]` | 3,688 | 1,089 | 14,342 | **599** | 0.076 |

Peak observed **907 GFLOP/s/board** on the favourable shape; **per NPU core ≈ 302 GFLOP/s [D]** (3 cores). Against ~6 TOPS int8 headline, the fp16 path reaches roughly 15% of nominal on its best shape and **1.3% on the FFN down-projection**.

### 4.1 dW error — the honest decomposition

"fp16 error" conflates operand quantisation with accumulation. Separated against fp64, gradients log-uniform over 1e-6…1e-2 **[M]**:

| micro-batch | accumulation err | **total err** | flushed to zero |
|---|---|---|---|
| M=4096 | 3.37e-07 | **1.02e-03** | 0.00% |
| M=1024 | 8.99e-07 | **8.40e-03** | 0.00% |
| M=512 | 1.55e-06 | **2.52e-02** | 0.00% |

**Accumulation is excellent** (≤1.6e-06 — the fp32 accumulator does its job). **Total error is operand quantisation**, and at M=4096 the 1.0e-03 is unremarkable for mixed precision with fp32 master weights. **Numerics are not the blocker.**

Two caveats, stated rather than smoothed: error *grows* as the micro-batch shrinks (2.5e-02 at M=512 would be concerning), and these are a sampled max over 16 outputs, so they are cancellation-sensitive. If this project proceeds, that wants a full-surface check, not a sample.

> **A methodology note worth keeping.** The first version of this harness reported `0.00e+00` error at every shape. That was the *generator*, not the hardware: operands drawn as multiples of 2⁻⁹ accumulate exactly in fp32. A benign distribution makes a numerics test report success unconditionally.

### 4.2 Micro-batch does not rescue dW

dW's contraction dim *is* the token count, so a power-of-two micro-batch < 2048 puts it on the fast schedule. Tile count fell 256 → 6; throughput moved 1.17× **[M]**.

| micro-batch | tiles | matmul µs | GFLOP/s | total µs | **per 4096 tok [D]** |
|---|---|---|---|---|---|
| 4096 | 256 | 82,095 | 104.6 | 144,305 | 144 ms |
| **1024** | **6** | 17,530 | **122.5** | 26,345 | **105 ms** ✅ |
| 512 | 3 | 14,615 | 73.5 | 19,202 | 154 ms |

**M=1024 is optimal, worth 1.37×** — and the gain comes mostly from the shorter CPU transpose, not the NPU. dW is not dispatch-bound.

---

## 5. Deliverable 5 — throughput projection

Per 1024-token micro-batch, per FFN layer, from measured per-shape rates **[D]**:

| | ms |
|---|---|
| fwd gate + up (fast) | 16.8 |
| fwd down (slow, contraction 3584) | 99 |
| dX down (fast) | 8.4 |
| dX gate + up (slow) | 198 |
| dW × 3 | ~180 |
| **per layer** | **~502** |
| **× 24 layers** | **~12.4 s** |
| **tokens/s, FFN only** | **~82 [D]** |

Excluded and additive: GDN backward on A76 (18 layers — the brief's own hypothesis, and consistent with the measured on-NPU GDN scan losing 1.9–4.8×), attention backward (6 layers), Adam8bit over 0.75B params, norms, CE, KL. Those put the realistic figure at **~30–50 tok/s/board [E]**.

**The slow-path matmuls are 59% of FFN step time [D]** — contraction 3584 alone. That is the single biggest lever and it is an architecture choice, not a tuning one (§7.1).

---

## 6. Deliverable 2 — integration paths

| path | exists | to write | CPU fallback | est. eng-weeks **[E]** |
|---|---|---|---|---|
| **(a) PyTorch-aarch64 custom autograd** | autograd, optimizer, CE, KL | ork-driver C ext; transpose+pack per op; GDN bwd | GDN, norms, CE, optimizer | **10–18** |
| **(b) ggml-opt + ggml-ork** | ggml-opt fwd/bwd graph; ggml-ork MUL_MAT routing | MUL_MAT **backward** routing; GDN op + bwd; MoE bwd | most non-matmul | **12–22** |
| **(c) bespoke C loop on ork_spine** | spine DAG dispatcher; all fwd primitives | entire backward, optimizer, data pipeline | — | **20–35** |

**Recommendation if this proceeds: (b).** ggml-opt already owns the autograd graph and optimizer, ggml-ork already routes MUL_MAT, and the gap is backward routing rather than a new framework. (a) duplicates ggml-ork's work inside a heavier runtime; (c) is the most control and by far the most code.

**But the recommendation is moot at the measured throughput** — all three paths inherit the §1 constraint, because it lives below all of them in ork-driver's fp16 scheduler.

---

## 7. Go/no-go

The brief asked for thresholds and the cheapest decisive measurement. Both resolved:

| threshold | result |
|---|---|
| ≥ 400 tok/s/board (≈3.5 boards ≈ one M5 Max) | ❌ **~30–50 [E]** |
| dW error ≤ 1e-2 | ✅ **1.0e-03 at M=4096 [M]** |
| re-pack < 50% of matmul | ✅ **7.6–17% [M]** |
| IOVA fits one domain | ✅ **~1.5 GB of 3.9 GiB [D]** |

**Three of four pass. Throughput fails, by 8–13× against a deliberately generous threshold.** Deliverable 6 (multi-board) was not measured: at 30–50 tok/s/board you need ~30–45 boards to match one M5 Max, so fabric bandwidth is not the limiting term and measuring it would not change the decision.

### 7.1 What would change the answer

1. **FFN intermediate ≤ 1024 and a power of two.** Removes the 12× penalty on down-projection and its dX — 59% of FFN step time **[D]**. Costs model capacity; `inter=2048` does *not* qualify (the predicate is strict `< 2048`).
2. **Extend `orki_f16_sched` to non-power-of-two K.** The 11-bank schedule is a *bank-split encoding* question, not a silicon limit — the CBUF has 12 banks regardless of K. If a valid split exists for K=3584, the cap goes 4 → 50 and the slow path gets ~12×. This is the highest-value single investigation and it is ork-driver work, not training work. The non-monotonic predicate and the `0x1b` schedule-byte floor (forcing below it hangs the submit) are the known hazards.
3. **An NPU-side or NEON-optimised transpose.** 49.5 ms per operand **[M]** is 34% of dW.
4. **Train in int8.** The int8 envelope (`mg_max*64`) is far better characterised — mcap 128 at K=2048 vs fp16's 4 at K=3584 — but int8 gradients are a much harder numerical problem than anything measured here.

### Recommended next step

**Do not start the training stack.** Spend ~1 engineer-week on item 2 instead: determine whether a valid CBUF bank split exists for non-power-of-two K in the fp16 schedule. `tools/re/f16_mcap_probe.c` is the existing probe; `ORK_F16_MTILE` overrides the cap; scan upward, never bisect.

That one question gates a 12× swing on the majority of backward work. If it resolves favourably the projection moves to roughly 300–500 tok/s/board **[E]** and this becomes worth re-scoping. If it does not, the answer is settled and the model should train on the M5 Max.

---

## Appendix — reproduction

```sh
make train_scope_bench train_dw_bench
sudo tools/util/npu_guard.sh -- timeout 300 ./train_scope_bench 256 5 0      # pack costs, no submits
sudo tools/util/npu_guard.sh -- timeout 900 ./train_scope_bench 4096 3 1     # matmul + error
sudo tools/util/npu_guard.sh -- timeout 600 ./train_dw_bench 1024 1024 1024 -6 -2
ORK_F16_MTILE=<n> ...                                                        # M-cap scan; scan UPWARD
```

Board safety: always via `npu_guard.sh --`, always with `timeout`, SIGTERM only — a hard kill mid-submit wedges the IOMMU, and a power cut mid-submit can corrupt SPI and require physical reflash.
