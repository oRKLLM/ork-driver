# Training on the RK3588 NPU via ork-driver — feasibility, proven

**Date:** 2026-10-05 · **Board:** RK3588 `.236`, kernel `6.1.115-vendor-rk35xx-fence`, governors pinned
(DDR 2112 MHz / A76 2.4 GHz) · **Library:** ork-driver @ `81428a7`, **`src/` unmodified**

Tags: **[M]** measured on this board · **[D]** derived from [M] · **[S]** read from source · **[E]** estimated.

---

## 1. Verdict

**Every gradient path is validated; an end-to-end step of the actual model has not run yet — and no
change to ork-driver was needed to get here.**

**Every matmul gradient has been computed on the NPU and validated against fp64; the non-matmul backward
ops are implemented and validated on the CPU; GDN backward is implemented in C in both the recurrent and
the chunkwise-parallel form, along with the four layer ops around it.** The goal was a first *correct*
training loop, speed secondary.

| path | validated against | result |
|---|---|---|
| linear forward `Y = X·W` | — | ✅ |
| `dW = Xᵀ·dY` | fp64, same fp16 operands | **3.853e-07** |
| `dW1` **chained across two layers** | **fp64 oracle** | **2.581e-04** |
| `dW2` | fp64 oracle | 1.982e-04 |
| attention `dV = Pᵀ·dO` | fp64 oracle | 3.789e-04 |
| attention `dQ = dS·K` | fp64 oracle | 2.826e-04 |
| attention `dK = dSᵀ·Q` | fp64 oracle | 2.989e-04 |
| softmax `dS` | implied by dQ/dK + fp64 formula check | 7.4e-08 / 1.3e-06 |
| RMSNorm `dx`,`dg` · SwiGLU `da`,`db` · CE · KL | fp64 central differences, **with positive controls** | 4.8e-07 … within fd floor |
| **GDN** `dq,dk,dv,dα,dβ` (recurrent form) | fp64 central differences, **with positive control** | **1.1e-08 … 3.2e-07** |
| **GDN chunkwise-parallel** (the matmul form) | the recurrent backward, same cotangents | **1.2e-14** at `d=128` — machine precision |
| GDN chunkwise, `T=1024 d=128` | **MLX fp32 fixtures**, both its paths | 1.0e-05 (fp32 build), 9.7e-06 (fp64) |
| GDN layer: conv1d, q/k RMSNorm, decay, output gate | fp64 central differences, 4 positive controls | 1.4e-10 … 5.5e-08 |

**Every row above is an oracle or an fp64 formula check.** Earlier drafts also listed finite differences
against an *fp16-rounded* forward (dW 1.21e-03, chaining 4.86e-02); those are removed — §9.3 explains why
that method is not a reference, and both claims are now carried by fp64 oracles instead.

*Sanity, not validation:* one- and two-layer SGD both reduced the loss monotonically over 12 steps
(937383→569701, 639813→36221). §2.1 is explicit that convergence is **not** evidence of a correct
gradient — a wrong-but-correlated gradient still descends — so these are recorded here and nowhere else.

**No new library primitives were needed.** `dX` and `dW` are ordinary matmuls with transposed operands —
existing `ork_f16_mm_pack` / `ork_f16_mm_run` calls. That is why all of this was validated without
touching `src/`.

Probes: `tools/train_step_proof.c`, `tools/train_2layer_proof.c`, `tools/train_attn_bwd_proof.c`,
`tools/attn_bwd_fp64_ref.c`, `tools/bwd_ops_fp64_check.c`, `tools/gdn_bwd_fp64_ref.c`,
`tools/gdn_chunk_bwd.c`, `tools/gdn_layer_bwd_check.c` (the last five CPU-only oracles).

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

| operation | forward | backward | where | validated |
|---|---|---|---|---|
| Linear | `ork_f16_mm_pack`/`run` **[S]** | `dX`, `dW` — same calls, transposed operands | **NPU** | ✅ 3.9e-07 vs fp64 |
| Linear, cross-layer | — | `dH` chaining | **NPU** | ✅ 2.6e-04 vs fp64 oracle |
| Attention | `ork_bmm_fp16_strided` **[S]** | `dV`,`dP`,`dQ`,`dK` — four matmuls | **NPU** | ✅ ~3e-04 vs fp64 oracle |
| Softmax | `ork_f16_npu_softmax` **[S]** | `dS = P⊙(dP − rowsum)·scale` | **CPU** — AI < 1 | ✅ *implied by dQ/dK* |
| RMSNorm | `ork_f16_npu_rmsnorm` **[S]** | `dx` incl. the `r`-coupling term; `dg` | **CPU** | ✅ 4.8e-07 / 4.9e-08 |
| SwiGLU | `ork_f16_mm_run_silu`, `ork_f16_npu_ewmul` **[S]** | `da`, `db`; `silu'(x)=σ(1+x(1−σ))` | **CPU** | ✅ 1.5e-07 / 2.4e-07 |
| Cross-entropy | write (trivial) | `(p − y)/B` | **CPU** | ✅ within fd noise floor |
| KL (distillation) | write (trivial) | `(p − q)/B` | **CPU** | ✅ within fd noise floor |
| **GDN recurrence** | chunkwise, in C | chunkwise backward, hand-derived, in C | **mixed** — the chunkwise form is matmuls (NPU-shaped); the sequential one is not | ✅ 1.2e-14 vs the recurrence; 1e-05 vs MLX at `T=1024 d=128` |
| **GDN layer** (conv1d, q/k norm, decay, out gate) | — | all four, in C | **CPU** — elementwise | ✅ 1.4e-10 … 5.5e-08 |
| Adam8bit | — | — | CPU | ❌ ordinary |
| MoE router + gather/scatter | partial (inference) | write | CPU | ❌ |
| **Transpose** | **missing** | **missing** | CPU today — §5 | the one primitive worth adding |

**Softmax backward is proven, not pending** — `dQ` and `dK` are computed *from* `dS`, so they cannot
agree with the fp64 oracle at 3e-04 unless `dS` is right; and `attn_bwd_fp64_ref.c` validates the same
formula against fp64 central differences independently (7.4e-08 / 1.3e-06).

**RMSNorm, SwiGLU, CE and KL backward are implemented and validated** in `tools/bwd_ops_fp64_check.c` —
CPU-only, no board, both sides fp64 so the check tests the *formulae* rather than any precision
question. The two easy things to get wrong are covered: RMSNorm's `dx` coupling through `r`, and
`silu'`.

**GDN backward is done, in both forms.** `tools/gdn_bwd_fp64_ref.c` derives and validates the recurrent
form (`dq` 1.1e-08, `dk` 3.2e-07, `dv` 3.1e-07, `dα` 1.7e-08, `dβ` 4.7e-08 vs fp64 central differences).
Its convention matches the production implementation exactly — that one computes
`u_t = β_t(v_t − S_{t−1} g_t k_t)`, `S_t = g_t S_{t−1} + u_t k_tᵀ`, which is this file's
`Ŝ=αS`, `e=v−Ŝk`, `S=Ŝ+βekᵀ` with `α≡g` and `βe≡u`.

**The chunkwise-parallel form — the one that is matmuls — is ported and hand-derived**
(`tools/gdn_chunk_bwd.c`). This matters because the sequential recurrence advances a `[Dv,Dk]` state once
per token: serial depth `T`, all outer products and mat-vecs, which is precisely the work shape that
leaves a matrix unit idle. The chunkwise form cuts serial depth to `T/C` and makes the intra-chunk work
dense matmuls — `K Kᵀ`, `K Sᵀ`, `Q Kᵀ`, `P U`, `(U·tail)ᵀ K`, each an `ork_f16_mm_*` call.

**Nothing could be transcribed.** The MLX reference implements the chunkwise *forward* by hand but gets
its backward from `mx.vjp` through that same forward — deliberately, so the adjoint cannot drift from the
function it differentiates. C has no autograd, so the adjoint had to be derived, and the failure mode
MLX was avoiding is the one this risks. Hence two oracles rather than one:

| check (at `T=128, d=128, C=64`) | result |
|---|---|
| chunkwise forward vs the sequential recurrence | 1.1e-14 (`Y`), 7.1e-15 (`S_final`) |
| chunkwise backward vs the sequential backward, same cotangents | **1.7e-15 … 1.2e-14** on all six gradients |
| chunkwise backward vs central differences of its *own* forward | 5.6e-10 … 2.4e-07 |
| chunk length 1, 2, 3, 5, 7, 8, 16, 32, T (incl. ragged tails) | identical at every length |
| two positive controls (gate coupling via `M`; the Gram-matrix `dK` term) | both FAIL, as required |

The backward-vs-backward agreement is at machine precision because the two are algebraically the same
function — an identity check, not a tolerance one. Only one step is not a transcription of a forward
line: the solve. `U = (I+M)⁻¹B` gives `dB = (I+M)⁻ᵀdU` (a transposed triangular solve) and `dM = −Z Uᵀ`;
the inverse is formed in neither direction.

**And it is checked against MLX at the real shape and precision** (`tools/gdn_fixture_check.c`). fp64
agreement with the sequential recurrence settles the algebra and says nothing about numerics at
`d=128, T=1024` in fp32, where the live quantities are `γ_t/γ_s` ratios spanning orders of magnitude and
a triangular solve. `qwen35-lora-moe/tools/export_gdn_fixtures.py` writes a flat fp32 fixture — inputs,
cotangents, and **two** sets of gradients, one from MLX autograd through the sequential reference and
one from the production chunkwise custom VJP. Disagreeing with both is a porting error; disagreeing
with one is a numerical one. At `T=1024, d=128, C=64`, with `g` drawn through the real
`exp(−exp(A_log)·softplus(·))` parameterisation so the small decays are exercised:

| build | vs MLX sequential | vs MLX chunkwise |
|---|---|---|
| fp64 | 8.0e-07 … 3.6e-05 | 2.0e-06 … 2.9e-05 |
| **fp32** | 7.1e-06 … 4.7e-05 | 7.0e-06 … 2.2e-05 |

MLX's own two paths agree to **4.3e-05** on this host, so that is the ceiling on what any of this can
claim — the C implementation is inside it. The fixture header carries that self-consistency figure and
whether the exporting host's fp32 matmul is degraded (measured 8.3e-04 on an M5 Max against 2.7e-07 on
an M3 Max, same code, because the matrix unit multiplies in bf16), and the C side takes its bar from
those rather than inventing a tolerance.

The C checker **includes** `gdn_chunk_bwd.c` and compiles it at both precisions (`-DGDN_REAL=float`)
rather than reimplementing it: one implementation, two builds. A second copy would drift from the
first, which is the failure mode this whole exercise is built to avoid.

**And the recurrence is not the layer.** Qwen3.5's GDN block wraps it in four more pieces, all now
derived and checked in `tools/gdn_layer_bwd_check.c`, each with a positive control: the depthwise causal
**conv1d + silu** (`d/dx`, `d/dstate`, `d/dweight`, `d/dbias`), the **scaled RMSNorm on q and k** — note
`mx.fast.rms_norm`, *not* L2, which differ by √D and would be a silent `1/√D` error — the **decay
parameterisation** `β=σ(b)`, `g=exp(−exp(A_log)·softplus(a+dt_bias))` including the per-head `A_log` and
`dt_bias` parameters, and the **gated RMSNorm** output `silu(z)·rms_norm(h,w)`.

What remains for GDN is **not mathematics**: numerics at real width (`d=128`, `T=1024`, fp16 rather than
fp64), which wants fp32 fixtures from the MLX implementation, and the wiring.

`grep -rinE "backward|gradient|_grad|adam|optimiz" include/` returns nothing **[S]**: ork-driver has no
autograd, no backward op and no optimizer — nor should it (§7.1). Everything still outstanding is
framework plumbing or a port, not unvalidated mathematics.

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

**Applied to MoE expert widths [D].** The expert down-projection contracts over the expert FFN width, so
that width decides the path. Pick from the left column:

| expert FFN width | power of two in [128, 2048)? | down-projection |
|---|---|---|
| 128, 256, 512, 1024 | ✅ | **fast path** |
| 56 | — | **hard refusal**: not a multiple of 32 (K) or 16 (N) |
| 64, 96, 224, 768, 1152, 1536 | ❌ | slow path |
| 2048 and above | ❌ (the bound is strict) | slow path |

**Expert count is free only when experts are designed with their own width.** If experts are made by
SPLITTING the dense FFN — which is the plan here — the count *sets* the width, `3584 / E`, and the two
constraints collide:

| experts (0.8B, FFN 3584) | width | result |
|---|---|---|
| 7 / 14 / 28 | 512 / 256 / 128 | **fast path** |
| 16 | 224 | slow path (not a power of two) |
| 64 | 56 | **hard refusal** — neither `K%32` nor `N%16` |

The same arithmetic on the 4B (FFN 9216): 96 and 144 experts give widths 96 and 64 — both legal, both
slow. **Choose the expert count from the divisor, not the other way round**: on a split-FFN MoE the
viable counts are the ones whose quotient is a power of two in [128, 2048).

The gate/up projections contract over `d_model`, so that wants to be a power of two under 2048 too.

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

**Whether loss scaling is needed is UNTESTED.** An earlier draft reported it as unnecessary on the
grounds that applying it moved an attention-backward error 4.683e-01 → 4.690e-01. That number was the
finite-difference artifact of §9.3, not a gradient error: the gradient was already correct, so loss
scaling had nothing to move and the experiment could only ever have come out that way. It says nothing
either way. The real question — fp16 underflow in small gradients at long sequence length, where the
magnitudes are far below anything these probes exercise — has not been asked. Assume loss scaling is
needed, as in any fp16 training, until a run at realistic length says otherwise.

---

## 7. Remaining work

### 7.1 Where the code lives — ork-driver stays a matmul library

**Exactly one thing belongs inside ork-driver: `ork_f16_transpose`.** It is a device primitive and the
largest backward-pass cost. Everything else belongs *on top*, for reasons that are the project's own:
AGENTS.md scopes ork-driver as "a userspace **matmul** library", with a 900-line-per-file budget and a
probe-anchored op registry — an autograd graph, optimizer state, a data pipeline and checkpointing are
none of those. And there is a direct precedent: **`ggml-ork` lives in the llama.cpp-rockchip fork, not
in ork-driver** — the backend adapter sits in the consumer repo. Training should take the same shape,
which also makes the ggml-opt path below the natural one.

The probes in `tools/` stay where they are: they are evidence that the boundary holds, not the framework.

### 7.2 The work

**In ork-driver — one item.** `ork_f16_transpose`. As a library op it takes the full mechanical tax:
`make check-registry` **fails the build** if its `OPS_REGISTRY.md` row names no probe or cites a dead
symbol; the examples ARE the suite, so it needs a self-validating one in `make test`; `tests/sbc_attest.txt`
must be refreshed from a board run; naming is enforced dtype-first. The gradient matmuls need none of
this — they are existing calls.

**On top of ork-driver — all CPU fp32 except the GDN chunk matmuls.** What is left after §3:

| | status |
|---|---|
| RMSNorm, SwiGLU, softmax, CE, KL backward | **done and validated** — `tools/bwd_ops_fp64_check.c` |
| GDN backward, recurrent and chunkwise | **done** — `gdn_bwd_fp64_ref.c`, `gdn_chunk_bwd.c` |
| GDN layer: conv1d, q/k norm, decay, output gate | **done** — `gdn_layer_bwd_check.c` |
| GDN numerics at real width | **done** — `gdn_fixture_check.c` against MLX fp32 fixtures at `T=1024, d=128`, fp32 and fp64 builds |
| autograd graph | adopt ggml-opt rather than write |
| **fp16 K-split of wide-K matmuls** | **optional, conditional — see below** |
| Adam8bit, data pipeline, checkpointing, MoE router bwd | ordinary |

### 7.3 K-split: a conditional, partly-retracted option

Decomposing a wide-K fp16 matmul into power-of-two slices measured, in one controlled session at
**M=1024**, 1.87× on the NPU portion and **1.2–1.6× overall** once packing is accounted on both sides —
and it was **more accurate** (1.778e-06 vs 2.874e-06 vs fp64), because shorter accumulations condition
better. That accuracy result is internally controlled and stands.

**But the speed result does not generalise.** A clean A/B of the library's own K-slice size at
**M=4096** (`soc->ks` 2048 vs 1024, same session, governors pinned) showed **no difference**: 453.4 vs
451.3 GFLOP/s. The two are consistent only if the benefit is M-dependent — the monolithic path measures
197 GFLOP/s at M=1024 and 453 at M=4096, so there is far more to recover at low M — but **that was not
isolated**, and an earlier 6.04× claim in this area turned out to be a clock artifact (§9.2). Treat
K-split as a low-M option worth re-measuring, not as a settled win.

### 7.4 The one thing nothing above establishes

**No end-to-end step of the actual model has run.** Every gradient path is validated in isolation and
the two-layer proof chains `dX` across layers, but a 24-layer hybrid with real tokens, a real optimizer
and a real data pipeline is an integration, and integrations fail for reasons unit checks do not see:
shape plumbing, state carried across chunk and sequence boundaries, memory at `T=1024`, and fp16
underflow at magnitudes these probes never produce (§6). That run is the next milestone, and it is the
one that converts "every part is correct" into "training works".

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
cc -O2 -o /tmp/bwd tools/bwd_ops_fp64_check.c -lm && /tmp/bwd                  # CPU only, no board
make train_step_proof train_2layer_proof train_attn_bwd_proof
sudo tools/util/npu_guard.sh -- timeout 600 ./train_step_proof   256 1024 1024 12 2e-4
sudo tools/util/npu_guard.sh -- timeout 600 ./train_2layer_proof 128 512 512 512 12 1e-4
sudo tools/util/npu_guard.sh -- timeout 900 ./train_attn_bwd_proof 256 128 2.0
cc -O2 -o /tmp/ref tools/attn_bwd_fp64_ref.c -lm && /tmp/ref 128 64 2.0 1e-5   # CPU only, no board
cc -O2 -o /tmp/gdn tools/gdn_bwd_fp64_ref.c -lm   && /tmp/gdn 24 8 8 1e-6      # CPU only, no board
cc -O2 -o /tmp/gcb tools/gdn_chunk_bwd.c -lm     && /tmp/gcb 128 128 128 64   # CPU only, no board
cc -O2 -o /tmp/glc tools/gdn_layer_bwd_check.c -lm && /tmp/glc                # CPU only, no board

# GDN at real width, against MLX. In qwen35-lora-moe (that is where MLX lives; ork-driver has no Python):
#   .venv/bin/python tools/export_gdn_fixtures.py --T 1024 --dk 128 --dv 128 --out /tmp/gdn1024.fix
cc -O2 -Itools              -o /tmp/gfc64 tools/gdn_fixture_check.c -lm && /tmp/gfc64 /tmp/gdn1024.fix
cc -O2 -Itools -DGDN_REAL=float -o /tmp/gfc32 tools/gdn_fixture_check.c -lm && /tmp/gfc32 /tmp/gdn1024.fix
```

Board safety: always via `npu_guard.sh --`, always with `timeout`, SIGTERM only — a hard kill mid-submit
wedges the IOMMU, and a power cut mid-submit can corrupt SPI and need a physical reflash.
