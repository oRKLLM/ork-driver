# Training a 0.75B hybrid SLM on RK3588 NPUs via ork-driver — scoping report

> # ✅ IT WORKS — correctness-first result (2026-10-05)
>
> **The report below answers "is it fast enough?" (no, by ~10–25×). That was the wrong question.** Asked
> instead whether it can work *correctly*, even at 25× slower than ideal, the answer is **yes, and it is
> demonstrated on hardware** — a real training loop, gradients verified against numerical derivatives,
> loss converging.
>
> | | one layer | two layers (chained) |
> |---|---|---|
> | dW vs fp64, same fp16 operands | 3.853e-07 **PASS** | 8.443e-08 **PASS** |
> | dW vs central finite differences | 1.210e-03 **PASS** | 4.862e-02 **PASS** |
> | loss fell monotonically, 12 steps | **PASS** 937383→569701 | **PASS** 639813→36221 |
>
> **Attention backward — the last unvalidated NPU path — also passes**, against an fp64 oracle over
> identical fp16 operands (`tools/train_attn_bwd_proof.c`, `tools/attn_bwd_fp64_ref.c`):
>
> | | dV | dQ | dK |
> |---|---|---|---|
> | NPU vs fp64 oracle | **3.789e-04** | **2.826e-04** | **2.989e-04** |
>
> So the **complete NPU-side training surface is validated**: linear forward, `dX`, `dW`, cross-layer
> chaining, and attention `dQ`/`dK`/`dV`.
>
> Heterogeneous split, measured: **NPU ~7.0 ms** for the six attention matmuls (contractions over `D=128`
> and `T=256`, both on the fast `sched=1` path) against **CPU ~1.8 ms** for softmax forward/backward and
> four transposes. The memory-bound reduction stays off the NPU, consistent with this board's rule that
> engines are additive on different resources and zero-sum on the same.
>
> `tools/train_step_proof.c`, `tools/train_2layer_proof.c` — board `.236`, governors pinned.
>
> **Method note that cost four detours and is the most reusable thing here.** The attention probe first
> reported `dV` passing while `dQ`/`dK` failed at 4.7e-01 — *and the loss still fell every step*. Two
> plausible explanations were both wrong (fp16 underflow of `dS`: loss scaling moved the error 4.683e-01 →
> 4.690e-01, nothing; near-uniform softmax: `dK` got **worse** with more spread). What settled it was a
> **CPU-only fp64 oracle**, which isolated three candidates in order: formula (correct, 7e-08), NPU path
> (correct, 3e-04), and the finite-difference check — which was the fault. It differentiates an
> fp16-*rounded* forward and divides by `max(|fd|,1e-4)`, so where the true derivative is small the
> forward's rounding noise is inflated without bound; `dK` read as 38× while being correct to four digits.
> **Build the oracle first. A cleverer check is not a substitute for a known-good reference**, and this is
> the fourth time in this session that a harness rather than the hardware was at fault.
>
> **Why the finite-difference check is the load-bearing one.** Checking dW against an fp64 matmul of the
> same operands only asks "is this a correct matmul?" — a transposed, mis-strided or stale gradient passes
> it, and the loss can still fall. Only the numerical derivative of the *actual loss* catches that. In the
> two-layer case W1 reaches the loss **solely through `dH = dY·W2ᵀ`**, so check 2 there is specifically the
> proof that gradients flow correctly ACROSS a layer boundary.
>
> **No new library primitives were required.** `dX` and `dW` are ordinary matmuls with transposed operands,
> so both proofs run against the shipped API with `src/` untouched. All NPU operands are resident weights
> refreshed via `ork_f16_mm_repack` — no IOMMU alloc/free in the inner loop.
>
> **Two things that silently break it**, both load-bearing and neither obvious:
> - **fp32 master weights are mandatory.** An SGD increment at a sane lr is below fp16 resolution, so an
>   in-place fp16 update rounds every step to zero — the loss sits flat while appearing to run.
> - **The backward pass needs two per-step transposes** a single-layer test never exercises: `Hᵀ` (an
>   activation) and `W2ᵀ` (a weight). Only `Xᵀ` is loop-invariant.
>
> **What this changes about the plan.** The remaining work is integration, not hardware and not numerics:
> autograd wiring, GDN backward, CE/KL, Adam8bit — all of which run on **CPU in fp32**, where correctness
> is not in question and slowness does not matter. Path **(b) ggml-opt + ggml-ork** (§6) is unaffected and
> still the recommendation. The §7 no-go stands *as a throughput statement only* and should not be read as
> a feasibility one.
>
> **If this is productionised**, new primitives acquire the full mechanical tax: `make check-registry`
> fails the build on an `OPS_REGISTRY.md` row without a named probe or citing a dead symbol; the examples
> ARE the suite, so each op needs a self-validating one in `make test`; `tests/sbc_attest.txt` must be
> refreshed from a board run; and naming is enforced dtype-first (`ork_f16_mm_grad_w`, not
> `ork_backward_*`). The gradient matmuls need none of that — they are existing calls. **An
> `ork_f16_transpose` would**, and it is the one worth adding: the CPU transpose is 49.5 ms per operand,
> the single largest cost in the backward pass.

**Date:** 2026-10-05 · **Board:** RK3588 `.236`, kernel `6.1.115-vendor-rk35xx-fence` · **Library:** ork-driver @ `81428a7`
**Benchmarks:** `tools/train_scope_bench.c`, `tools/train_dw_bench.c` (this branch; self-validating against fp64)

Every figure is tagged **[M]** measured on this board today, **[D]** derived from [M] by arithmetic, **[S]** source-read, or **[E]** estimated.

---

> # ⚠ RETRACTION — read this first (2026-10-05)
>
> **The `75.7 GFLOP/s` figure for `fwd down` (K=3584), used throughout §1, §4, §5 and §9, was measured
> with the DDR governor unpinned after a reboot. Re-measured at DDR 2112 MHz it is 453.4 GFLOP/s.**
>
> Consequences, all of them mine to own:
>
> | claim | status |
> |---|---|
> | "12× spread between the fast and slow fp16 path" | **WRONG** — the real spread is 902 vs 453 ≈ **2×** |
> | "6.04× from `soc->ks` 2048→1024" | **WITHDRAWN** — `ks=2048` 453.4 vs `ks=1024` 451.3, i.e. **noise** |
> | "fp16 `ks` is unapplied tech debt costing an order of magnitude" | **WITHDRAWN** — it costs ~0% |
> | "Sk=4 bug in the fp16 K-slice path" | **WRONG** — `f16_sk_sweep` passes Sk=1…7 standalone |
> | §5 projection ~82 tok/s (FFN only) | **too pessimistic** — recomputed ~168 tok/s **[D]** |
> | §0 verdict: no-go by ~20–45× | **STANDS**, margin narrows to roughly **10–25×** |
> | §4.1 dW error decomposition | **STANDS** — single-session, internally controlled |
> | §9.2 `0x1b` refutation + the weight-tile law | **STANDS** — internally controlled A/B |
> | §3 pack/repack ratios | **STANDS** — single-session |
>
> **Root cause of the error: I compared a measurement taken before two board reboots against one taken
> after, and attributed the difference to a configuration change.** Governors reset on boot. Every
> cross-session comparison in the original text is therefore unsafe; only within-session A/Bs survive,
> which is why the surviving rows above are exactly the internally-controlled ones.
>
> **Standing rule added: re-baseline after any reboot, and never compare across one.** This is the same
> class of error as the arm-position bias documented in the Experiment Log — a difference in measurement
> conditions read as a difference in the thing being measured.
>
> **AMENDED 2026-10-05 — read §9 before §1.1 and §7.** Peer review corrected the *attribution* of the
> central finding (mcap=4 is an unprogrammed register, not a hardware capacity), narrowed its impact
> (int8 is unaffected, so production prefill is not), and a follow-up measurement refuted the obvious
> fix. The verdict below is unchanged; the reasoning behind it is materially different.

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
| Attention bwd | — | **PROVEN [M]** — `dV=Pᵀ·dO`, `dP=dO·Vᵀ`, `dQ=dS·K`, `dK=dSᵀ·Q` all ~3e-04 vs fp64 | **NPU matmuls + CPU softmax-bwd** — no new primitive |
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

---

## 9. Amendments after peer review and follow-up measurement

### 9.1 mcap=4 is an unprogrammed register, not a hardware capacity — §1.1 was wrong on attribution

`orki_f16_synth` writes the bank split **only** in the sched=1 branch **[S]**:

```c
if(sched){ ... orki_setrn(rc,REGCMD_N,RK_CNA_CBUF_CON0,v); }
else     { orki_setrn(rc,REGCMD_N,RK_CNA_CONV_CON2,16*(mc+1)); }   /* no CBUF_CON0 at all */
```

So at non-power-of-two K the split is never programmed and inherits the template default in
`regcmd_array_4x32x16.h` (lines 4 and 7, word `0x00b11040`): **`0xb1` = DATA_BANK 1, WEIGHT_BANK 11**
**[S]**. One data bank ÷ a 7168 B row = 4.57 → the observed mcap=4. **That is the template's allocation,
not the silicon's** — the CBUF has 12 banks at K=3584 like anywhere else; 11 are assigned to weights and
never reassigned. It also explains the miscompute better than §1.1 did: at mc=16 the activations want
114,688 B against 32,768 — the tile was raised without moving the banks that hold it, so it went fast and
silently wrong instead of faulting.

**The `K < 2048` gate is a guard rail around a broken formula, not a hardware statement.** At K=3584,
`scale=14`, so `base = (int)(177.0 - 15.0*13) = -18` **[S]** — the affine split goes negative past
`scale ≈ 12.8` and the `if(v<0x1b)v=0x1b` clamp does all the work.

### 9.2 The obvious fix is refuted — and it wedges the board **[M]**

`0x1b` is `0xb1` nibble-swapped (DATA 11 / WEIGHT 1), and 11×32768/7168 = 50.28 → exactly the 180224/K
figure, so "force `0x1040=0x1b`, mc ≤ 50" looked like a one-register fix. Tested via `ork_f16_fuzz_add`
(three arms, with a positive control). **Arm C did not miscompute — it hung**: doorbell sentinel never
landed at 20 s, twice; soft reset num 6; colsplit de-escalated to single core; the library then failed
rather than returning stale output. It left an orphaned job holding core 0.

Cause: the two registers are written **as a pair**. A real sched=1 emission at K=3584 would set
`CONV_CON2 = 16 × min(mc+1, pow2_floor(cbuf/K)) = 16 × 16 = 256`; the test left the sched=0 value
`16×51 = 816` while declaring 11 data banks. **So the hypothesis is untested, not refuted** — what failed
was the test. The pair to try is `{CBUF_CON0=0x1b, CONV_CON2=256, mc ≤ 50}`. Not attempted: the board was
down, and a second attempt at a mechanism that already wedged it needs an explicit operator decision.

**UPDATE — tested with the registers paired, and it HUNG AGAIN [M].** `{CBUF_CON0=0x1b,
CONV_CON2=256, mc=50}` at K=3584: sentinel never landed, soft reset num 6, orphaned job on core 0,
second reboot required. **The hypothesis is refuted by measurement, not merely untested.**

**Why — and this explains the power-of-two gate completely.** The pack layout is `[NT][KT][16][32]`, so
an N-tile is 16 columns and one tile of weight is `K × 16 × 2` bytes. That must fit the weight
allocation:

```
K × 16 × 2  ≤  WEIGHT_BANK × 32768      ⟺      K ≤ WEIGHT_BANK × 1024
```

`0x1b` sets **WEIGHT_BANK = 1**, so it is valid only for **K ≤ 1024**. At K=3584 one weight tile is
114,688 B = 3.5 banks, so the hardware is told it has 32 KB for a tile that cannot fit, and it stalls
waiting for weight data that never arrives — a hang, not a wrong answer, exactly as observed twice.

**And this is what the `K < 2048` gate actually encodes.** sched=1 admits K ∈ {128, 256, 512, 1024} —
*precisely the K values where a single weight bank holds one tile*. K=2048 would need two, and the gate
excludes it. The power-of-two condition is incidental; **the real constraint is `K ≤ 1024`, forced by the
`if(v<0x1b)v=0x1b` clamp pinning WEIGHT_BANK to 1.** The formula cannot express a split with more weight
banks, so it cannot serve large K no matter how the affine terms behave.

**The derived maximum valid split at K=3584 [D]** — untested, and a third board attempt was not taken
without an explicit decision after two wedges:

```
WEIGHT_BANK = ceil(3584/1024) = 4 ;  DATA_BANK = 12 − 4 = 8   →  0x48
mcap = 8 × 32768 / 7168 = 36.6 → 36        (not the 50 that 0x1b implied)
CONV_CON2 = 256 as before
```

That predicts **36 rows/tile, a 9× improvement over mcap=4, not 12.5×** — and it is consistent with the
earlier scan, where mc=8 was correct and mc=16 was not.

**Source arithmetic for the original (now-refuted) idea [S].** Evaluate the sched=1 branch at the
*known-good, measured-exact* K=1024 / mcap=176 configuration:

```
R = 57344/1024 = 56 → pow2_floor = 32 ;  rows = min(177,32) = 32  →  CONV_CON2 = 512
scale = 4 ; base = (int)(177−15·3) = 132 ; slope = 60 ; mg = (176+63)/64 = 3
v = 132 − 60·2 = 12  →  clamped  →  0x1b
```

**The configuration that is already validated bit-exact emits exactly `0x1b` — DATA_BANK 11,
WEIGHT_BANK 1.** So `0x1b` is a *proven-good* large-tile bank split, not a speculative value; K=3584
simply never receives it. It also disposes of the "mc=50 is multi-pass" worry: the working K=1024 case is
equally multi-pass (mc=176 against rows=32), so the hardware already carries that routinely.

### 9.3 int8 is unaffected — the impact is fp16-only, and not the production prefill path

`orki_i8_synth` has the identical structure, but **every int8 caller passes `sched` as a literal `1`**
**[S]** (`core/colsplit.c`, `core/dyn_ctl.c`, `ssm.c`) — there is no `orki_i8_sched()`. int8 always
programs the split at any K, which matches previously-measured geom dumps (K=3072 → `0x66`, K=3584 →
`0x57`, both non-power-of-two, both satisfying `mc·K = DATA_BANK·32KB`).

**So the earlier claim of "12× on production prefill" is withdrawn** — production prefill is int8 W8A8.
The fp16 win is real but scoped to: the orkd F16 dtype, `ork_f16_mm_run_silu`, the attention bmm
(`ork_bmm_fp16`), the SSM/GDN scan path, and fp16-native models.

**Shared root cause, which is the more useful framing.** Both dtypes carry the same broken affine formula
(`base` negative past `scale ≈ 12.8`) and dodge it differently — **int8 K-splits above 4096**, so
`scale ≤ 8` keeps the formula valid and the banks programmed; **fp16 disables the schedule above 2048**,
which skips the register write and inherits one data bank. One defect, two mitigations, only one of which
costs an order of magnitude. The durable fix removes the formula for both: `data_banks =
ceil(mc·K·2/32768)`, `weight_banks = 12 − data_banks`, both ≥ 1, no power-of-two dependency.

### 9.4 NEON supplementation — the CPU beats the NPU on every slow-path shape **[E]**

The shapes where the NPU is crippled are unusually CPU-friendly. `[4096,3584]×[3584,1024]` is 30.1 GFLOP
over ~53.5 MB → **AI ≈ 563 FLOP/byte [D]**: compute-bound, so a NEON arm will not contend for the bus
(this project's additivity rule: additive on different resources, zero-sum on the same).

| | GFLOP/s |
|---|---|
| NPU, contraction 3584 | **75.7 [M]** |
| 4× A76 @2.4 GHz NEON fp16 FMLA, peak | 307 **[D]** |
| realistic GEMM at 40–60% of peak | **120–185 [E]** |

**And a transposed GEMM is free on CPU** — an index order, not a materialised buffer. So dX and dW also
shed the 49.5 ms transpose **[M]** and the 5.5 ms repack **[M]** that the NPU path cannot avoid:

| per layer, 1024-tok micro-batch | NPU | NEON **[E]** |
|---|---|---|
| dX gate/up: matmul + 2 transposes + 2 repacks | 198 + 38 + 11 = **247 ms** | **~100 ms** |
| dW (measured end-to-end 82 GFLOP/s incl. plumbing) | **26.3 ms [M]** | **~14 ms** |

Following it through, the split becomes **NPU: fwd gate/up only, 15.0 GFLOP, ~16.8 ms** vs **CPU:
everything else, 52.7 GFLOP, ~351 ms [E]** — i.e. **the NPU does ~5% of a training step.** Throughput
improves to ~122 tok/s **[E]** from 82, but the verdict hardens rather than softens: this is *CPU training
with a 5% NPU assist*, and 4× A76 alone is ~20× short of an M5 Max.

**Two qualifications, and the first is the important one:**

1. **This recommendation is CONDITIONAL ON §9.2 STAYING UNRESOLVED, and the two findings are coupled.**
   The mcap 16/32 scan measured 296–437 GFLOP/s — fast but wrong, because the banks had not moved. If
   `{0x1b, 256}` lands bit-exact, the NPU returns to roughly 300–437 GFLOP/s at K=3584, comfortably above
   NEON's 120–185, and **this recommendation inverts.** Read as a pair: the answer is probably "fix the
   encoding", and "use the CPU" is the fallback if it cannot be fixed.
2. **CPU capacity is not free.** Prefill on this board already measures ~88% CPU-bound (act-quant ~45%,
   resolve ~43%, NPU ~11%) **[M, prior]**. A NEON GEMM arm contends with act-quant, the 49.5 ms
   transposes and Adam8bit, so 120–185 GFLOP/s is a *standalone* upper bound an end-to-end step will not
   realise. "Move it to the CPU" sounds free and is not.

What survives both qualifications is the dispatch rule this board keeps teaching: **dispatch on resource
profile, not engine identity.** At 563 FLOP/byte the shape is compute-bound and genuinely will not fight
the ~30 GB/s bus, which is precisely the condition under which a second engine is additive here.

### 9.4a It IS tech debt — the hardware does this today, in int8 **[M]/[S]**

Asked directly: is the fp16 cap a hardware limit or unfinished software? **Unfinished software**, and the
proof is that the same silicon does it correctly in another dtype.

`orki_i8_synth` and `orki_f16_synth` carry the **identical formula**, differing only in a divisor that is
itself correct (fp16 rows are 2 bytes, so `K/256 ≡ 2K/512`) **[S]**:

```
int8:  scale = K/512 ;  base = (int)(177.0 − 15.0·(scale−1))
fp16:  scale = K/256 ;  base = (int)(177.0 − 15.0·(scale−1))
```

`base` goes negative past `scale = 12.8`. The two dtypes then diverge:

| | valid to | actual | result |
|---|---|---|---|
| int8 | K ≤ 6554 | **K-splits at 4096** → scale ≤ 8 | ✅ in range; programs its banks at any K |
| fp16 | K ≤ 3277 | **no K-split** → K=3584 → scale 14 | ❌ out of range; gate added; banks discarded |

**The root cause is that fp16 lacks int8's K-split.** int8 stays inside the formula's valid range by
construction; fp16 runs off the end, and a `K < 2048` gate was added to route around the breakage rather
than fix it — silently throwing away the bank allocation as a side effect. Unfinished generalisation.

Confirmed at the hardware level: int8 at K=3584 evaluates to `base = (int)(177−90) = 87 = 0x57`
(DATA 7 / WEIGHT 5), satisfying `mc·K = 64×3584 = 7×32768` **[D]** — a non-power-of-two K with a
correctly programmed split, in production, today.

### 9.4b Bank split is necessary but NOT sufficient — the residual is a third register **[M]**

Four configurations tested at K=3584, M=1024 (arm A reference = mcap 4 / `0xb1`; arm B = tile raised,
banks unmoved, as the positive control):

| banks | mc | GFLOP/s | differing / 1,048,576 | |
|---|---|---|---|---|
| `0xb1` (1/11) | 4 | 196.0 | — | default, correct |
| `0xb1` (1/11) | 36 | 474.8 | 811,008 | control fires ✅ |
| `0x1b` (11/1) | 50 | — | — | **HANG** (weight tile > WEIGHT_BANK) |
| **`0x48`** (8/4) | 36 | **472.3** | **7,840** (0.75%) | 2.4× faster |
| **`0x57`** (7/5) | 32 | **453.3** | **9,040** (0.86%) | int8's own value at this K |

Moving the banks removes the hang and eliminates **~99% of the error** (811,008 → ~8,000), and buys
**~2.4×**. But a ~0.8% residual persists **at every split, including the one int8 itself uses here** —
so the residual is **bank-split-independent** and is therefore a third piece of state, not the allocation.

**Prime suspect: `CBUF_CON1` / `DATA_ENTRIES`.** fp16 sets it to `K/32` with **no `mc` dependence**
**[S]**, while the int8 fold path computes `(K/64)·mc`. Raising the tile without updating it leaves the
data buffer under-described — the right shape for a small residual. **Untested.**

**The parameter search stops here.** Four attempts, two board wedges, and this repo's method record
documents an eight-hypothesis register hunt that was ultimately fitted to an artifact. The remaining work
is not more guessing: it is **porting int8's approach** (K-split to stay in range, or a byte-derived
split `data_banks = ceil(mc·K·2/32768)`, `weight_banks = 12 − data_banks`) using a working reference
implementation in the same codebase — a bounded engineering task, not an RE expedition.

**Revised value of §7.1 item 2: ~2.4× demonstrated, bit-exactness one register away, with a reference
implementation to copy.** That is a stronger case than the original estimate, and it is now ork-driver
work with a known shape rather than an open question.

### 9.4c K-SPLIT: the shippable fix — 1.2–1.6×, bit-safe, zero RE **[M]**

Since int8 stays in the formula's valid range *by K-splitting*, the obvious move is to give fp16 the same
treatment — and it needs no guessed registers at all, because it makes the **already-validated** sched=1
path applicable instead of trying to repair sched=0.

`3584 = 1024 + 1024 + 1024 + 512` — every chunk is in the measured-exact set `{128,256,512,1024}`, mcap
176/352 instead of 4. (2×1792 would not work: 1792 is not a power of two, so the gate still refuses it.)

| | µs **[M]** | |
|---|---|---|
| monolithic (shipped, mcap 4) | 38,145 | 197.0 GFLOP/s |
| K-split total | 43,110 | 0.88× |
| └ **NPU runs** | **20,446** | **1.87× faster** |
| └ packs (4) | 11,277 | |
| └ A K-slice copies | 2,465 | |
| └ host fp32 accumulate | 5,400 | 2.2 GB/s — plain scalar loop |

**The raw total is misleading** and the first reading of it was unfair: the monolithic figure excludes its
pack while K-split's includes four. Corrected **[D]**:

| compared fairly | monolithic | K-split | |
|---|---|---|---|
| **training** (weights change; both must pack) | 13,776 + 38,145 = 51,921 | 43,110 | **1.20×** |
| **inference** (weights resident; neither packs) | 38,145 | 28,311 | **1.35×** |
| inference + NEON accumulate **[E]** | 38,145 | ~24,000 | **~1.59×** |

Note the four chunk packs (11,277 µs) are **cheaper than the single monolithic pack** (13,776 µs) — same
total elements — so chunking adds no packing cost. The accumulate is unoptimised scalar at 2.2 GB/s and
is the obvious next improvement.

**It is also MORE accurate than the shipped path**: max relerr vs fp64 **1.778e-06 (split)** vs
**2.874e-06 (monolithic)** **[M]**. Splitting a long accumulation into fp32 partials improves
conditioning. The 797,719 differing elements are the summation-order change, not error — this is
correctly *not* bit-identical.

**The two routes, compared:**

| approach | speedup | correct? | RE needed | board risk |
|---|---|---|---|---|
| bank poke (`0x48`/`0x57`) | 2.4× | **No** — 0.75–0.86% wrong, cause unknown | yes, unresolved | **wedged the board twice** |
| **K-split** | **1.2–1.6×** | **Yes — more accurate than baseline** | **none** | none observed |

**Recommendation: K-split.** It is smaller than the register fix promised but it is real, it inherits
correctness from configurations already validated bit-exact, and it is ordinary engineering rather than
reverse engineering. The register path remains open as a later upside if someone identifies the third
piece of state (§9.4b) — the two are complementary, not alternatives.

### 9.5 Standing rule adopted: no numerics check without a positive control

This harness produced a false pass (operands as multiples of 2⁻⁹ accumulate exactly in fp32 → `0.00e+00`
at every shape), and this repo produced another last month (a 2-bit LCG whose period equalled `mcap·K`
made consecutive M-tiles bit-identical; eight hypotheses were fitted to a pattern that did not exist in
silicon). Same class, different mechanisms: **benign test data makes the check vacuous, and a vacuous
pass is worse than no check because it ends the investigation.**

`tools/f16_bankswap_probe.c` implements the rule — a configuration that MUST miscompute, and the probe
returns `VOID` and refuses to report the hypothesis arm if the control comes back clean.

### 9.6 Board state

The arm-C wedge left core 0 held by an orphaned job (zero render-node holders, no dmesg progress). The
documented recovery is a graceful `sudo reboot` over SSH; that was **denied by this session's permission
classifier** and is left for the operator. **The board takes no work until it is cleared.**
