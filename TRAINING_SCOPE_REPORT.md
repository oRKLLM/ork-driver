# Training on the RK3588 NPU via ork-driver — feasibility, proven

**Date:** 2026-10-05 · **Board:** RK3588 `.236`, kernel `6.1.115-vendor-rk35xx-fence`, governors pinned
(DDR 2112 MHz / A76 2.4 GHz) · **Library:** ork-driver @ `81428a7`, **`src/` unmodified**

Tags: **[M]** measured on this board · **[D]** derived from [M] · **[S]** read from source · **[E]** estimated.

---

## 1. Verdict

**Training works on this hardware, correctly, today — with no changes to ork-driver.**

Every gradient a transformer training step needs has been computed on the NPU and validated against an
fp64 reference. The goal was a first *correct* training loop, speed secondary; that goal is met.

| path | validation | result |
|---|---|---|
| linear forward `Y = X·W` | — | ✅ |
| `dW = Xᵀ·dY` | vs fp64, same fp16 operands | **3.853e-07** |
| `dW` | vs central finite differences | **1.210e-03** |
| `dX` chaining across two layers | vs finite differences (W1 reaches loss only via `dH`) | **4.862e-02** |
| attention `dV = Pᵀ·dO` | vs fp64 oracle | **3.789e-04** |
| attention `dQ = dS·K` | vs fp64 oracle | **2.826e-04** |
| attention `dK = dSᵀ·Q` | vs fp64 oracle | **2.989e-04** |
| one-layer SGD, 12 steps | loss monotone | 937383 → **569701** |
| two-layer SGD, 12 steps | loss monotone | 639813 → **36221** |

**No new library primitives were needed.** `dX` and `dW` are ordinary matmuls with transposed operands —
existing `ork_f16_mm_pack` / `ork_f16_mm_run` calls. That is why all of this was validated without
touching `src/`.

Probes: `tools/train_step_proof.c`, `tools/train_2layer_proof.c`, `tools/train_attn_bwd_proof.c`,
`tools/attn_bwd_fp64_ref.c` (CPU-only oracle).

---

## 2. The minimal correct training loop

What the proofs actually run, and the shape a real implementation should take.

```
NPU (fp16 in, fp32 out)              CPU (fp32)
────────────────────────             ──────────────────────────────
Y   = X · W                          loss, dY
dW  = Xᵀ · dY                        transposes (Xᵀ, Hᵀ, Wᵀ, dSᵀ)
dX  = dY · Wᵀ                        softmax fwd/bwd, RMSNorm bwd, SwiGLU bwd
attention: S=Q·Kᵀ, O=P·V,            cross-entropy, KL
           dV, dP, dQ, dK            optimizer + fp32 MASTER weights
```

### 2.1 Three things that silently break it

**fp32 master weights are mandatory.** An SGD increment at a sane learning rate is far below fp16
resolution. Updating fp16 weights in place rounds every step to zero — **the loss sits flat while the
run looks healthy**. Keep masters in fp32, cast to fp16 per step, `ork_f16_mm_repack` into the resident
weight.

**Convergence is not evidence of a correct gradient.** The attention probe's first run had `dQ`/`dK`
wrong by 47% **and the loss still fell every step**. A wrong-but-correlated gradient is still a descent
direction. Validate against a reference, not against the loss curve.

**The backward pass needs per-step transposes a single-layer test never exercises** — `Hᵀ` (an
activation) and `W2ᵀ` (a weight) change every step; only `Xᵀ` is loop-invariant. There is no NPU
transpose, so these are CPU work. See §5.

---

## 3. Inventory

| operation | forward | backward | where |
|---|---|---|---|
| Linear | `ork_f16_mm_pack`/`run` **[S]** | `dX`, `dW` — same calls, transposed operands **[M]** | **NPU, proven** |
| Attention | `ork_bmm_fp16_strided` **[S]** | `dV`,`dP`,`dQ`,`dK` — four matmuls **[M]** | **NPU, proven** |
| Softmax | `ork_f16_npu_softmax` **[S]** | `dS = P⊙(dP − rowsum)` | **CPU** — AI < 1, memory-bound |
| RMSNorm | `ork_f16_npu_rmsnorm` **[S]** | write | CPU |
| SwiGLU | `ork_f16_mm_run_silu`, `ork_f16_npu_ewmul` **[S]** | write | CPU |
| Cross-entropy, KL | — | write | CPU |
| GDN recurrence | — | write | **CPU** — the on-NPU scan is measured to *lose* 1.9×@0.8B / 4.8×@9B |
| Adam8bit | — | — | CPU |
| MoE router + gather/scatter | partial (inference) | write | CPU |
| **Transpose** | **missing** | **missing** | CPU today — §5 |

`grep -rinE "backward|gradient|_grad|adam|optimiz" include/` returns nothing **[S]**: there is no
autograd, no backward op, no optimizer. Those are the framework, and they are the bulk of the work —
but they are ordinary CPU fp32 code where correctness is not in question.

---

## 4. Shape rules — these constrain model design

**Hard refusals [S]** (`ork_f16_mm_pack` returns NULL): `K % 32 == 0`, `N % 16 == 0`. Fine for
d ∈ {1024, 1536, 2048} and head dims 64/128. **It bites on vocabulary**: 49152 ✅, 50257 ❌ — pad it in
the model config, not at runtime.

**Fast-path rule [S].** `orki_f16_sched(K)` enables the 11-bank schedule only when the matmul's
*contraction* dim is a **power of two in [128, 2048)**. Outside that it falls to a one-bank schedule with
a much smaller M-tile. This is a performance property, not a correctness one — **M is safely tiled to the
cap either way** (`chunk = orki_f16_mcap(...); if(chunk>M) chunk=M`), so an out-of-envelope shape is slow,
not wrong.

Practical consequence for model design: **choose dimensions so contraction dims are powers of two below
2048.** Attention is naturally well-shaped (contractions over `d`=128 and `T`), as the proofs confirmed.
FFN down-projections (contraction = intermediate size) are the ones that miss it.

**IOVA [D]:** ~1.5 GB of resident fp16 weights for a 0.75B model against a measured ~3.9 GiB ceiling.
Not a constraint at this size; one domain suffices, which also avoids the multi-domain defect class.

---

## 5. Costs, for planning

Not a go/no-go — the goal is correctness. These are for sizing the work.

**Pack vs repack [M]** — allocate once, refresh per step:

| weight | pack µs | repack µs | repack/run |
|---|---|---|---|
| `[1024,3584]` | 11,928 | **5,491** | 0.17 |
| `[3584,1024]` | 13,776 | **5,932** | 0.015 |
| `[1024,1024]` | 3,688 | **1,089** | 0.076 |

`ork_f16_mm_repack` is **2.3–3.4× cheaper than pack** — the MEM_CREATE/MEM_DESTROY round trip is most of
a pack. Never pack-and-free in a training loop. Re-packing is **not** a bottleneck: under 17% of the
matmul it enables.

**The CPU transpose is the real cost [M].** One `[4096,1024]` fp32→fp16 strided transpose is **49.5 ms**
— more than a forward matmul, and 34% of a `dW`:

| dW at M=4096 | µs |
|---|---|
| CPU transpose of X | **49,539** |
| pack dY | 12,671 |
| NPU matmul | 82,095 |
| **total** | **144,305** |

**`ork_f16_transpose` is the one primitive worth adding**, and the only item that would take the full
registry/attest/naming tax (§7).

**Heterogeneous split, measured on attention backward [M]:** NPU **~7.0 ms** for six matmuls, CPU
**~1.8 ms** for softmax forward/backward and four transposes. The memory-bound reduction belongs on the
CPU — consistent with this board's rule that engines are additive on *different* resources and zero-sum
on the same.

**Reference matmul rates at pinned governors, M=4096 [M]:** K=1024/N=3584 ≈ 907 GFLOP/s; K=1024/N=1024
≈ 590; K=3584/N=1024 ≈ 453.

---

## 6. Numerics

**dW error, decomposed against fp64 [M].** "fp16 error" conflates two effects that differ by orders of
magnitude:

| micro-batch | accumulation error | total error (incl. operand quantisation) |
|---|---|---|
| M=4096 | 3.37e-07 | **1.02e-03** |
| M=1024 | 8.99e-07 | 8.40e-03 |
| M=512 | 1.55e-06 | 2.52e-02 |

Accumulation is excellent — the fp32 accumulator does its job. The total is operand quantisation to
fp16, and at M=4096 it is unremarkable for mixed precision with fp32 masters. **Numerics are not a
blocker.** Note the error grows as the micro-batch shrinks (cancellation-sensitive); these are sampled
maxima, so a production harness wants a full-surface check.

**Loss scaling was *not* required** in the attention backward — tested, and it moved the error 4.683e-01
→ 4.690e-01, i.e. not at all. It remains correct practice for fp16 training generally; it was not the
fix here.

---

## 7. Remaining work

**In ork-driver — one item.** `ork_f16_transpose`. As a library op it takes the full mechanical tax:
`make check-registry` **fails the build** if its `OPS_REGISTRY.md` row names no probe or cites a dead
symbol; the examples ARE the suite, so it needs a self-validating one in `make test`; `tests/sbc_attest.txt`
must be refreshed from a board run; naming is enforced dtype-first. The gradient matmuls need none of
this — they are existing calls.

**On top of ork-driver — the bulk, all CPU fp32.** Autograd graph, GDN recurrence backward (the largest
single piece of real maths), RMSNorm/SwiGLU/softmax backward, cross-entropy, KL, Adam8bit, data pipeline,
checkpointing, MoE router backward if MoE.

**Recommended path: ggml-opt + ggml-ork [E].** ggml-opt already owns the autograd graph and optimizer,
ggml-ork already routes `MUL_MAT` to ork-driver, and the gradient matmuls are `MUL_MAT`s. The gap is
backward routing plus the CPU ops above — **~12–22 engineer-weeks [E]**. A PyTorch-aarch64 custom op
duplicates ggml-ork inside a heavier runtime; a bespoke C loop on `ork_spine` is the most control and by
far the most code.

---

## 8. Risks specific to training

| risk | why training differs from inference | mitigation |
|---|---|---|
| **Intermittent fp16 colsplit fault** (documented: concurrent-fetch CDMA wild, prevented within a buffer by CONTIG) | inference tolerates a rare bad result; **training integrates it into the weights and poisons the run** | per-step gradient sanity check — norm bounds, NaN/inf. Cheap, not optional |
| Vocabulary not `% 16` | a hard pack refusal | pad in the model config |
| NPU is single-stream | concurrent submits wedge the IOMMU | one submit owner |
| Shape sweep is wider than inference's | more chance of hitting an unvalidated envelope | envelope is conservative by default; M is safely tiled |

---

## 9. Method notes

Four times in this work a **measurement harness, not the hardware, was at fault**. They are recorded
because each was expensive and each is avoidable.

1. **A numerics check with no positive control is vacuous.** A generator emitting multiples of 2⁻⁹
   accumulates exactly in fp32, so every shape reported `0.00e+00` error. Any numerics check needs a
   configuration that MUST fail; if it passes, the check is broken.
2. **Never compare across a reboot.** Governors reset on boot. A baseline taken before two reboots vs a
   measurement taken after produced a phantom 6.04× that was entirely a DDR clock difference. Re-baseline.
3. **Finite differences against a *rounded* forward are not a reference.** The attention check
   differentiated an fp16 forward and divided by `max(|fd|,1e-4)`; where the true derivative is small,
   rounding noise is inflated without bound. It reported `dK` at **38×** for a gradient correct to
   **3.0e-04**.
4. **Build the oracle first.** A CPU-only fp64 reference (`attn_bwd_fp64_ref.c`) took minutes, needed no
   board, and isolated in one run what four board experiments could not: formula correct (7e-08), NPU
   path correct (3e-04), **check wrong**. A cleverer check is not a substitute for a known-good reference.

Two hypotheses are recorded as wrong rather than quietly dropped: fp16 underflow of `dS` (loss scaling
changed nothing) and near-uniform softmax (`dK` got *worse* with more logit spread).

---

## Appendix — reproduction

```sh
make train_step_proof train_2layer_proof train_attn_bwd_proof
sudo tools/util/npu_guard.sh -- timeout 600 ./train_step_proof   256 1024 1024 12 2e-4
sudo tools/util/npu_guard.sh -- timeout 600 ./train_2layer_proof 128 512 512 512 12 1e-4
sudo tools/util/npu_guard.sh -- timeout 900 ./train_attn_bwd_proof 256 128 2.0
cc -O2 -o /tmp/ref tools/attn_bwd_fp64_ref.c -lm && /tmp/ref 128 64 2.0 1e-5   # CPU only, no board
```

Board safety: always via `npu_guard.sh --`, always with `timeout`, SIGTERM only — a hard kill mid-submit
wedges the IOMMU, and a power cut mid-submit can corrupt SPI and need a physical reflash.
