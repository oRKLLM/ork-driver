# PPU standalone SDP ops — RE decode (2026-07-10)

## ★★★★★★ OPEN-SOURCE REGISTER NAMES (Mesa "Rocket" / accel-rocket driver + NVDLA) ★★★★★★
The RK3588 NPU is NVDLA-derived; the mainline **`drivers/accel/rocket/rocket_registers.h`** (Tomeu Vizoso,
built from the RK3588 TRM ch.36 + NVDLA docs/source + ONNC) NAMES the registers we decoded numerically.
Blocks (absolute offsets): **PC 0x0000, CNA 0x1000 (conv), CORE 0x3000, DPU 0x4000 (output stage)**. Our
"block 0x1001 / offset 0x40xx" IS the **DPU output-processing stage** (NVDLA-SDP-equivalent). Named map:
| our offset | Rocket macro | meaning |
|---|---|---|
| 0x4020 | REG_DPU_DST_BASE_ADDR | output base addr (NOT the coeff buffer — my earlier label was wrong) |
| 0x4024 | REG_DPU_DST_SURF_STRIDE | output surface stride |
| 0x4030 / 0x4034 / 0x403c | DPU_DATA_CUBE_WIDTH / HEIGHT(+minmax) / CHANNEL | **GEOMETRY** (the dims to generalize) |
| 0x4040 | DPU_BS_CFG (+BS_ALU 0x4044, BS_MUL 0x4048, BS_RELUX_CMP 0x404c) | bias/scale + simple activation (RELU-x) |
| 0x4060 | DPU_BN_CFG (+BN_ALU 0x4064, BN_MUL 0x4068, BN_RELUX_CMP 0x406c) | batch-norm affine (scale+bias) |
| 0x4070 | DPU_EW_CFG | element-wise + **LUT** (EW_LUT_BYPASS bit; EW_RELUX_EN 0x400, EW_RELU_BYPASS 0x200) |
| 0x40c0 | DPU_SURFACE_ADD | surface add |

CONSEQUENCES (big):
- **Activations are now well-specified**: relu/relu6 = DPU_BS/BN RELU-x (RELUX_EN + RELUX_CMP_VALUE clamp);
  silu/gelu/sigmoid = DPU_EW LUT (DPU_EW_CFG, EW_LUT_BYPASS=0) — matches the earlier 0x4080-clamp / 0x4104-LUT
  fused-activation RE. Emit these deliberately, no fuzzing.
- **Geometry generalization = DPU_DATA_CUBE_WIDTH/HEIGHT/CHANNEL + DST_BASE/STRIDE** — the set_sdp_geom() helper.
- **β UN-BLOCKED (reframes #24 pt.2)**: β (the ^-0.5) is baked into the **EW LUT** (gated by DPU_EW_CFG 0x4070 =
  0x108003c4 → LUT active for lrn), i.e. the SAME programmable-LUT path already RE'd for SiLU (0x4104 stream).
  So β=0.5 IS settable by loading an x^-0.5 LUT — not a hard blocker. (My earlier "0x4020=coeff buffer" was
  wrong: 0x4020=DST output; re-analyze the LRN param buffers with correct roles.)
- NOT in rocket's header (conv-focused): the 0x50xx block (our 2nd-read/reduction path) + the explicit LUT
  DATA PORT. NEXT: NVDLA **CDP** (cross-channel data processor = the LRN/reduction unit) + **SDP LUT** spec for
  the reduction (Σx²) + LUT-load format; or the untruncated rocket header / Mesa registers.xml.
Sources: docs.kernel.org/accel/rocket, drivers/accel/rocket/rocket_registers.h, nvdla.org (hw/sw), blog.tomeuvizoso.net.

## ★★★★★★★ DECISIVE (NVDLA CDP spec): NPU-native rmsnorm/l2norm is HARDWARE-BLOCKED ★★★★★★★
NVDLA CDP (Cross-channel Data Processor = the LRN unit): `out = src × f(Σsrc²)`, `f(x)=1/((k+α/n·x)^β)` via a
LUT ("RESMO"). **The reduction window n is HW-limited to {3,5,7,9}.** RMSNorm/L2Norm need a FULL-channel
Σx² (n = d_model = 2048–4096) → the CDP physically can't (window ≤ 9). This EXPLAINS the campaign: size=5/7
ran on NPU, size=63 fell to CPU (n>9 unsupported); and it's why librkllmrt/rocket run norms on CPU. The
pow (`x^-0.5`) IS doable via the SDP LUT (loads "any non-linear op"), but the **full-channel reduction has NO
fixed-function path** on this NPU (only CDP@n≤9, or a matmul x·xᵀ = the submit-floor-bound tiny op proven to
lose in #23). ⇒ the CDP fixed-function LRN OP can't do it (n≤9). **CORRECTION (don't overstate "hardware-blocked"):
rmsnorm DOES decompose into NPU-fittable pieces — square (DPU_EW) → full-channel sum as a MATMUL/conv
contraction `(x⊙x)·ones[d,1]` (CACC contracts K=d unboundedly; the CDP window limit is irrelevant to the
matmul path) → rsqrt (SDP LUT) → scale (DPU_EW). So it's POSSIBLE, not hardware-impossible.** The real
blocker is PERFORMANCE: that's 3–4 separate submit-floored NPU ops per norm (×~2/layer), the reduce-matmul
is degenerate (N=1, violates N%16/32 tiling → pad-waste; tiny GEMV at decode) — ~1.5ms of submit overhead
vs a ~µs FUSED NEON pass. Guaranteed loss (the #23 submit-floor wall), which is why librkllmrt/rocket/us all
keep norms on CPU. **rmsnorm/l2norm stay CPU (gated primitives keep the API ready) — not because silicon
can't, but because the decomposition fragments into small ops that lose to the fused pass. CLOSED on perf.**
CONSTRUCTIVE takeaway: ACTIVATIONS remain viable — relu/relu6 = DPU_BS/BN RELU-x, silu/gelu/sigmoid = DPU_EW
LUT — all now named (rocket header) + geometry via DPU_DATA_CUBE_*. If on-NPU pointwise activations are ever
wanted, that path is clear and buildable (no fuzzing). But per #23 (small-op submit-floor) they'd likely be
neutral+gated too; the durable value is the NAMED register map for future work.



Decoded librknnrt's standalone `.rknn` ops captured under the interposer at full `RKDUMP_WORDS=16384`
(the 4KB truncation that wedged the old replay is gone). **All run clean on the NPU (`ret=0`, 1 submit)** —
standalone SDP ops, including *reductions*, are viable; the Phase-1B wedge was truncation (H1), not
architecture. Captures: board `/tmp/{relu,op_relu6,op_lrn,op_softmax,conv}_cap.log`; decoded maps in the
session scratchpad `*_decoded.txt`.

## The SDP op family (enable=0x18)
`relu`, `relu6`, `lrn`, (and `softmax` mostly) are ONE template: `task.enable=0x18`, `regcfg_amount≈69`,
3 identical per-core tasks, register blocks **PPU `0x1001`** + **CDMA `0x2001`**. Buffers: handle1=regcmd/task,
handle2=input (fp16), handle3=output, handle4/5=coefficient/LUT buffers.

### Geometry / config registers (PPU 0x1001) — shared
`0x4030=0x0f 0x4034=0x0f` (dims−1), `0x403c=0x3f003f` / `0x4058=0x3f` / `0x405c=0xf000f` (surface), `0x40c0=0x1000`
(size), `0x4020`=coefficient-buffer dma, `0x4024=0x1000` (its size). CDMA `0x2001`: `0x500c/0x5010=0x0f` (dims),
`0x5014=0x3f`, `0x5018`=primary input dma, `0x5040`=stride, `0x5044=0x17849` (mode). Generalizing these
(dims/strides) for arbitrary `[M,n]` is the remaining geometry work (mirror `set_mul_geom` in npu.c).

### The op selector: `0x4040`
`0x12` = pointwise/clamp (relu/relu6); **`0x53` = reduction class (lrn, softmax)**. This is the register that
turns the SDP into a reducing op.

## clamp(relu) → norm(lrn) delta — only 7 registers
| reg | relu | lrn | role |
|---|---|---|---|
| `1001:4040` | `0x12` | `0x53` | op-mode: pointwise → reduction |
| `1001:4070` | `0x383` | `0x108003c4` | reduction/exponent param (the `^β`) |
| `1001:4020` | h4 dma | h5 dma | coefficient/LUT buffer |
| `2001:5038` | `0` | input dma (`0xffff6000`) | **2nd read of the input = the Σx² accumulate** |
| `2001:5034` | `0x1` | `0x40000008` | that read's mode |
| `2001:5040` | `0` | `0x1000` | that read's stride |
| `2001:5044` | (n/a) | `0x17849` | reduction descriptor |

So a reduction = **feed the input a second time through CDMA `0x5038` + set op-mode `0x4040=0x53` + the
power via `0x4070` + a coefficient buffer**. LRN computes `x/(k+α·Σx²)^β`; β=0.5 ≡ RMSNorm/L2Norm rsqrt.

## Remaining per-op work to reach bit-exact ork_npu_* (unblocked, sized — NOT done yet)
1. **Decode the coefficient buffer (handle 5, the `02fb01fa…` data)** — how α/k/β (and any LUT) are encoded.
   This is the gate to setting RMSNorm params (α=1/n, k=eps, β=0.5) vs LRN's window params.
2. **Generalize geometry** (`0x4030/34`, `0x403c/58/5c`, `0x40c0`, CDMA dims/stride) from the captured
   fixed shape to arbitrary `[M,n]` — mirror `set_mul_geom`.
3. **Build `REGCMD_SDP_NORM` template + emitter** in `src/npu.c`; fill the already-scaffolded gated
   `ork_npu_rmsnorm_f16`/`ork_npu_l2norm_f16` (they're CPU-correct today, `ORK_NORM_NPU` selector reserved).
4. **Validate bit-exact vs the CPU refs** (`tools/re/test_bmm.c` harness) + measure vs NEON (keep gated
   until it beats CPU — the submit floor still applies to small rows).

Order of value: rmsnorm/l2norm (via LRN) first; softmax next (multi-stage, `0x4040=0x53` + a `0xd` stage);
relu/relu6/silu are pointwise (relu = clamp, `0x4040=0x12`; silu = the `0x18`+LUT path already partly RE'd
in [[ppu-activation-on-npu]] / PPU_FUSED_ACTIVATION_WIP.md).

## #24 controlled-RE campaign — UNBLOCKED + partial param map, but β not locatable (2026-07-10 pt.2)
UNBLOCKED the build: op_lrn's recipe is `tools/re/models/build_ops.py` — **torch `nn.LocalResponseNorm`
via `torch.onnx.export`, do_quantization=False** (a hand-built ONNX `LRN` node goes to CPU; the torch
export goes to NPU). Replicated it in the .239 Colima venv (torch 2.2.0) with an α/β/k/size sweep — all 5
variants submit on NPU (69 regs each). PARAM MAP (diff base vs each):
- **α (1e-4→1e-3): coefficient buffer handle 5** (+handle 2). **k/bias (1→2): handle 5** (all of it).
- **β (0.75→1.0): BYTE-IDENTICAL model** (regs + all 5 handles + task buffer unchanged). **size (5→7):
  BYTE-IDENTICAL too.** ⇒ β/size are NOT in the captured regcmd/buffers.
CONSEQUENCE: rmsnorm/l2norm need **β=0.5** (the √) — but β isn't in the captured regcmd (RKNN's fp16
NPU-LRN appears to use a fixed-form pow, ignoring/approximating β, or an uncaptured LUT). So the LRN
template can express α/k but NOT β=0.5 → **can't cleanly reprogram LRN→rmsnorm via this route.** To settle
whether NPU-LRN even honors β: compare base-vs-β OUTPUT (needs output dump) — deferred (per #23 an NPU
norm is submit-floor-neutral+gated anyway, so low ROI). Env/recipe now documented ([[colima-vm-239]]) so
resuming is cheap. Fresh LRN .rknn + captures on board /tmp/lrn_{base,alpha,beta,k2,size7}*.

## #24 EARLIER (superseded): BLOCKED on the rknn build recipe
Set up rknn-toolkit2 2.3.2 in the .239 Colima VM ([[colima-vm-239]]) and built an LRN α/β/bias/size
param-sweep, transferred + captured on the board. **BLOCKER:** every freshly-built LRN — fp16 AND int8,
size 5/7/31/63, incl. the canonical AlexNet (α=1e-4,β=0.75,bias=1,size=5) — is routed to **CPU by
rknn's compiler (0 NPU submits)**, while the pre-existing `~/rknn_sdk/op_lrn.rknn` (SAME toolkit 2.3.2,
same [1,64,16,16] shape) submits 1 NPU op (compiled to an `RKNN_OP_NNBG` node). So NPU-LRN placement
depends on a build recipe (rknn.config/flags) NOT reproducible from the op_lrn artifact — can't generate
varied-param NPU regcmds → can't diff α/β/bias→registers → can't parameterize the emitter via this route.
NOT worth grinding rknn.config permutations: [[ppu-activation-on-npu]] #23 showed small-op NPU offload is
submit-floor-NEUTRAL, so an NPU norm lands neutral+gated regardless. Options to resume: (i) find the
op_lrn build script/recipe; (ii) verbatim-replay the one op_lrn regcmd (drivability proof, params
unknown); (iii) accept CPU norms (where librkllmrt runs them too). Primitives stay scaffolded+gated.

## ★★★★★★★★ rsqrt-LUT DONE — fused reduce+rsqrt in ONE NPU submit, validated 0.12% (2026-07-10 pt.3)
Built `ork_mm_build_f16_rsqrt_lut` (npu.c; header) — the norm twin of the SiLU LUT builder: bakes
`1/sqrt(ss/n+eps)` into the fp16 fused-output PWL LUT, reduce weight packed as **-S** so `acc=-S·Σx²` lands
in the fp16 negative index-spread band, calibrated per-layer to `[ss_min,ss_max]`. Runtime reuses
`ork_mm_run_f16_silu` (the LUT is generic): `sq·(-S)` → `C=R·LUT[idx(acc)]` → `scale=C·out_scale`.
VALIDATED (tools/re/test_rsqrt_lut.c, n=2048, M=8): NPU-emitted scale vs CPU 1/sqrt = **maxrel 0.0012**.
⇒ the FULL on-NPU norm is: CPU square → **1 fused NPU submit (reduce+rsqrt)** → CPU broadcast-scale. The
3-4-submit-floor concern collapses to ONE submit — the chaining the user asked for.
CONSTRAINTS/remaining: fused reduce+rsqrt needs K=n_feat≤2048 (fp16 single-tile Sk=1); for n=4096 DECOUPLE
= plain K-split reduce (any n, already built) + a TINY (K-small) rsqrt-LUT op on the scalar ss. Production
wiring needs per-layer LUT CACHE + ss-range calibration (build-once, not per-call — it runs probe submits).
Even fused, standalone still loses to the ~µs CPU NEON pass (#23 floor) — the win is fusing into an
adjacent matmul's output stage (now reachable: DPU output-stage registers are named). Infra + validated.

## FINAL LANDING (2026-07-10) — FULLY on-NPU norm for ANY n, gated + tested
- **rmsnorm/l2norm (ork_npu_rmsnorm_f16/l2norm_f16, gated ORK_NORM_NPU):** BOTH the reduction sum(x²)
  (K-split matmul, any n) AND the rsqrt run on the NPU; only the final broadcast-scale is CPU. Validated
  ORK_NORM_NPU=1: n=128 **0.0013**, n=4096 **0.0015**. Default (gate off) = CPU, unchanged.
- **rsqrt on NPU — two paths, both validated:** FUSED reduce+rsqrt in one submit (K=n≤2048, ork_mm_build_
  f16_rsqrt_lut) = **0.0006**; DECOUPLED K=512 rsqrt-on-ss (any n, ork_norm_rsqrt_npu) = **0.0005**.
- **★ THE K=32 BUG (resolved):** the decoupled rsqrt-on-ss first used K=32 → the fp16 fused-silu tiling is
  DEGENERATE at K=32 (acc~0 → constant output → ~10% err). FIX = K=**512** (the LUT builder's known-good
  geometry) with the scalar ss fed DENSE + NORMALIZED (A[m,k]=ss/G ∀k, weight=-S·G/512 → acc=-S·ss, A~O(1),
  small weight = the probe regime). Now 0.0005. `ork_norm_rsqrt_npu` is wired into the norms (not unused).
- **ALL primitives in `make test`** (examples/test_bmm.c): ork_bmm_i8/i4/fp16 (exact/0.0004), rmsnorm/l2norm
  (0.0005 CPU-path; 0.0013 NPU-path under ORK_NORM_NPU), ork_l2norm_f32 (exact), rsqrt-LUT fused 0.0006 +
  decoupled 0.0005 (both SKIP cleanly if PPU-fuse off). `make test_rsqrt_lut` validates both standalone too.
- Ceiling unchanged: standalone still loses to CPU NEON; the win needs fusing into an adjacent matmul (DPU
  output-stage regs named). All gated off by default. Everything uncommitted.

## Per-op remaining-RE ledger (come back to these)
| op | decoded so far | remaining RE to reach bit-exact ork_npu_* |
|---|---|---|
| **rmsnorm / l2norm** (via `lrn`) | enable=0x18, op-mode **`0x4040=0x53`** (reduction class — shared by lrn+softmax; relu=`0x12` pointwise), reduction=2nd CDMA read `0x5038`, coeff buf=handle5 `0x4020`. **Norm power/scale (α/k/β) = reg `0x4070`: lrn=`0x108003c4` (LRN-specific; softmax & relu both use `0x383`).** Coeff buffer (h5) = BULK data (6300/8192 u32 differ vs relu's h4; dominant fill lrn `0x7c01` vs relu `0xfe00` + a fixed marshaling ramp `02fb01fa…`) — NOT a scalar param block. | **#24 read-only RE DONE to here (no wedge risk).** Remaining = PARAMETERIZE via the **controlled-RE campaign** (SAFE, the build_pair.py method that decoded silu/relu): on .239 rknn-toolkit2, build LRN models with KNOWN varied α/k/β (+ identical everything else), capture, diff → map α/k/β → (`0x4070` bits + coeff-buffer slots). THEN geometry-generalize + emitter + on-board validate (the emitter/submit step is the wedge-aware one — do NOT guess registers). Alt: verbatim replay of the captured regcmd (low-risk, proves drivability, fixed geometry). NOTE: given #23 showed small-op NPU offload is submit-floor-neutral on these models, an NPU norm will likely land neutral+gated too — the primitive is scaffolded/gated so the codebase is ready regardless. |
| **softmax** (`op_softmax`) | 4 tasks: `0x18/0x18/0xd/0x18`; op-mode `0x4040=0x53`; a distinct `enable=0xd` middle stage | decode the `0xd` stage (max-subtract + exp/normalize); how the 4 stages chain; geometry; emitter; validate vs `ork_softmax_f32` |
| **relu / relu6** (clamp) | enable=0x18, op-mode `0x4040=0x12`, 69 regs; clamp bound in the output-stage regs (`0x4080`-ish, cf. fused ReLU=0xff80 in [[ppu-activation-on-npu]]) | isolate the clamp-min/max reg for relu6's upper bound; geometry generalize; emitter; validate |
| **silu / gelu / sigmoid** (LUT) | `0x18`+LUT: `0x4100`=index port, `0x4104`=data port (256 int8 values @ output scale); fused-output map already RE'd (PPU_FUSED_ACTIVATION_WIP.md) | STANDALONE (non-fused) geometry; regenerate the 256-entry LUT per activation+scale; emitter; validate vs `ork_silu_f32` |
| geometry (shared) | `0x4030/0x4034` dims−1, `0x403c/0x4058/0x405c` surface, `0x40c0` size; CDMA `0x500c/0x5010/0x5014` dims, `0x5040` stride, `0x5044` mode | derive the dims/stride formulas for arbitrary `[M,n]` (mirror `set_mul_geom`); one `set_sdp_geom()` helper serves all ops |

## Wedge, definitively
Not a hardware bug and not architectural — the old standalone replay fed a truncated (4KB of 20KB) sigmoid
LUT program. Full capture (`RKDUMP_WORDS≥6144`) + replaying the complete regcmd is the fix. librknnrt runs
every one of these clean.
