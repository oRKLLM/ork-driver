# Attention-on-NPU re-derivation after the mode-transition-layer refactor (v0.7.0)

**Why:** the `ork_npu_enter`/`XSPEC` refactor fixed two drifted resets (`e54288d` ~107ms SDP entry,
`7d0d459` ~105ms fp16→chain). My recent "attention-on-NPU loses" verdicts were all measured on the
drifted glue. Re-deriving on hardware (board 10.3.0.236, v0.7.0 @ 22b7480).

## Re-derived on hardware (all FLIPPED from the pre-refactor verdicts)

| metric | pre-refactor | v0.7.0 | note |
|---|---|---|---|
| SDP ewmul_f16 / add per-op | ~107 ms | **~22–30 µs** | drift gone; `test_ewmul_f16`/`test_add` |
| softmax-on-NPU coherence | coherent | coherent (max\|err\| ~2e-5) | `softmax_probe` |
| softmax-on-NPU cost (M=256,n=256..1024) | "loses 2.7×" | **wins ~1.8×** (687µs vs 1235µs CPU) | `softmax_cost` (new tool) |
| attn matmul floor M=128 (1.7B) | submit-bound loss | **104 ms, wins 2.5×** vs CPU 264ms | `attn_cost 128` |
| attn matmul floor M=256 (1.7B) | — | **183 ms, wins 6.8×** vs CPU 1252ms | `attn_cost 256` |
| attn floor 7B M=256 | — | 371 ms (236.8µs/mm), repack +20ms | `attn_cost7b` (need CPU bucket) |

**Conclusion so far:** every "attention-on-NPU loses" verdict from this session was a drifted-glue
artifact. Floor + softmax both WIN post-refactor. The fused-attention chain is worth completing.

## New/uncommitted here
- `tools/softmax_cost.c` + Makefile target (diagnostic; on board too). Not committed.

## Fork end-to-end (feat/attn-npu @ 0.7.0) — DONE + BLOCKED on model
- Submodule bumped 0.6.89 → 0.7.0, `ORK_ATTN_CPU` flags restored, committed `6226b8993`, pushed;
  board fork synced + rebuilt (OpenMP ON, ccache).
- **Two-consecutive-`stream_f16_chain` (nkvp≠DV) is COHERENT on v0.7.0** — `two_stream_f16_probe`,
  bit-exact all iters (QK^T N=256 → HD; A·V N=128 → L2). The garbage bug was drifted transition/size
  state, now converged into `ork_npu_enter`/`XSPEC`. **4th conclusion flipped.**
- **End-to-end blocked by a broken model, NOT by attention/ork-driver:** every qwen3-1.7b GGUF on the
  board (q8_0, f16, regen-Q4) emits garbage EVEN WITH `ORK_OFF=1` (NPU fully bypassed, supports_op→false).
  `--verbose-prompt` shows `control-looking token: 128247 '</s>' ... probably a bug in the model` — a
  bad vocab/tokenizer from the conversion. Base build is PROVEN GOOD: mamba2-2.7b under ORK_OFF is
  coherent ("...a city of contrasts. It is a city of history").
- Footprint: `ORK_NO_BF=1` fits the 1.7B int8 in one domain (baseline ran clean, EXIT=0). The earlier
  `CREATE: Bad address` OOM was the dual Bb+Bf layout hitting the single-domain 4GiB HW IOVA cap
  ("up to 1 domains"). Note: `ork_iova_reserve` guard (3900MB) UNDER-COUNTS real IOVA (EFAULT fired
  before the guard) — worth fixing the accounting.

## END-TO-END VALIDATED (2026-07-14) — coherent, profiled
- Re-converted **Qwen2.5-1.5B-Instruct → q8_0** on the .239 Colima VM (arch qwen2, vocab 151936
  CORRECT, template present; the board's old qwen3-1.7b GGUFs had a bad vocab). sha e6898e7a…, on board.
- **Attention-on-NPU is COHERENT end-to-end**: baseline (attn CPU) and `ORK_ATTN=1` (QK^T + A·V on NPU)
  give identical correct output ("Paris"; "Red, Blue, Green"). The refactor's re-derivation holds in a real model.
- **Perf (ORK_NO_BF=1, -t4):** pp256 108 vs 115 (0.94×), pp512 113 vs 142 (0.79×), pp1024 103 vs 125
  (0.83×). Slower end-to-end; decode unchanged (M=1, attention path not engaged).
- **Bottleneck profile (ORK_ATTN_PROF, pp512, per-stage cumulative):**
  softmax(CPU) 62% ‖ QK^T-submit 14% ‖ densify+repack QK 11% ‖ A·V-submit 7% ‖ densify V 5% ‖ scatter 1%.
  **NPU matmuls = 21%, CPU marshalling = 79%, and the naive single-threaded CPU softmax alone = 62%.**
  The matmul-floor win (attn_cost) is real but swamped by the handler's scalar softmax (ggml's baseline
  does the same softmax vectorized+threaded on 4 cores).
- **THE fix = fused chain with softmax ON the NPU** (softmax_cost proved on-NPU softmax wins 1.8× post-refactor);
  secondary: parallelize/vectorize the handler softmax + densify (OpenMP). Instrumentation lives in
  `ggml_backend_ork_flash_attn_ext` (ORK_ATTN_PROF), uncommitted on the board's ggml-ork.cpp.

## FUSED SOFTMAX-ON-NPU CHAIN — BUILT + PROFILED (2026-07-14)
Built the fused chain (QK^T fp16 → softmax with exp ON NPU via ork_npu_exp_i16, one batched SDP submit
in-chain → A·V fp16), gated default-on (`ORK_ATTN_SM_CPU=1` selects CPU softmax). **Bit-coherent**
("Red, Blue, Green"). Per-step profile (ORK_ATTN_PROF, pp512, 56 ops, ms):
`dqk=214 qk=288 | SM=3568 [prep=468 quant=508 exp-NPU=2094 norm=497] | dv=99 av=137 sc=103  total=4409`.
**exp-on-NPU submit = 2094 ms = 47% of the op**, and SM total 3568 ms is 3.5× the 4-core CPU softmax (1025 ms).
Three configs @pp512: baseline 142 · attn-NPU+CPU-softmax 115 (0.81×) · attn-NPU+fused-softmax-NPU 89 (0.63×).
**Confirmed empirically:** on-NPU exp (~30 ns/el) can't beat vectorized CPU expf (~5 ns/el @4-core); no
fusion fixes it (amortizes submit, not per-el compute) + quant/dequant round-trips add 1 s. Softmax stays CPU.
Genuine NPU win (QK^T/A·V matmul floor 2.5-6.8× isolated) is only ~10% of the op → too small to carry it at
1.5B/pp512. Instrumentation + fused-chain + OpenMP CPU-softmax path all uncommitted on board ggml-ork.cpp.
Flow-chart artifact published.

## LONG-CONTEXT + 7B DATA (2026-07-14) — per-op offload plateaus, then REGRESSES
- 7B pp512: baseline 47.8 · attn-NPU+cpuSM 43.2 (0.90×) · fused-SM 42.0 (0.88×, exp-NPU REJECTED at
  M=H*N=14336 → silent CPU fallback, wasted quant). 7B pp1024+ OOM (IOVA guard, domain-0 full).
- 1.5B long ctx: pp1024 0.85× · pp2048 0.88× (best) · **pp4096 0.66× (regresses)**. The CPU marshalling
  (densify/softmax-prep/scatter) is O(N²) → grows WITH the attention it accelerates. Per-op offload
  STRUCTURALLY can't win; only a fused NPU-native chain (no marshalling) can. This is the decisive finding.

## FUSED-CHAIN BUILD (started 2026-07-14) — the only path to end-to-end-NPU win
Enabler CONFIRMED (softmax_reduce_probe): exp(NPU SDP)→Σ(NPU reduce-matmul) works post-refactor, no
ETIMEDOUT, coherent (rel-err 1%). The mode-switch that forced CPU-reduce is now carried by ork_npu_enter.
So full on-NPU softmax (max→exp→sum→norm) is BUILDABLE. Caveat: as separate submits exp+reduce ≈ single-
thread CPU (~4× slower than 4-core CPU); the WIN depends on SINGLE-SUBMIT overlap (SDP concurrent with
matmul units) — UNMEASURED, the decisive experiment.
Plan: (1) ✅ enabler. (2) general on-NPU softmax at arbitrary n (max-reduce SDP + exp-LUT + sum-reduce
SDP + normalize) — generalize the vendor fixed-n=64 replay (SM_WT / 9-task decode). (3) assemble
QK^T(fp16 MM)→softmax(SDP)→A·V(fp16 MM) as ONE PC-chain (ping-pong off, keep-warm; cf ork_mm_run_chain_i8_ffn).
(4) NPU-native KV to kill densify/scatter (bigger: whole-layer-on-NPU). (5) A/B vs baseline — does overlap
hide the softmax? Target choice (throughput vs CPU-offload/saturation) changes only the success bar, not the build.

## BUILD PROGRESS (2026-07-14) — end-to-end-NPU softmax, primitive by primitive
- ✅ Stage 0 enabler: exp(SDP)→Σ(reduce-matmul) works post-refactor (softmax_reduce_probe), no ETIMEDOUT.
- ✅ Stage 1: exp + Σ BOTH on NPU inside real attention (handler sm_npu path + reduce-matmul e·ones[nkvp,16]).
  COHERENT ("Red, Blue, Green"). Profile pp512: SM=4004 [prep=464 quant=580 exp-NPU=2124 red-NPU=540 norm=296].
  Reusable: on-NPU Σ-reduce now drives attention (ORK_ATTN_SM_SUMCPU=1 to A/B). pp512 85 t/s (slower, expected/accepted).
- Remaining CPU in softmax: **max** (frontier primitive), quantize (LUT pre-step), per-row divide (→ recip-LUT+ewmul).

### PRIMITIVE COLLECTION (the reusable payoff — for other ops + other RK NPUs)
- on-NPU Σ-reduce (e·ones matmul) ✅ · on-NPU exp (SDP LUT) ✅
- **on-NPU max-reduce** ❌ — mechanism LOCATED: SDP ALU-function field = reg 0x4040 (low byte 0x50=MUL/0x40=ADD;
  ADD also flips 0x4048=0x40000000, 0x4070=0x9042_02c0). NVDLA SDP_X ALU has MAX/MIN/SUM algos → a MAX code in
  0x4040 gives pairwise max(a,b); max-reduce = log2(n) EW-max passes (or PDP max-pool, unsupported). PLAN: extend
  ork_npu_probe_i8_mul with ORK_EW_ALU (0x4040) + 0x4070 env hooks, fuzz for MAX (low-risk single ewmul submit,
  n≤4096), then build ork_npu_max_reduce + wire as softmax 2a. Then normalize-on-NPU (recip LUT + ewmul).
- on-NPU per-row normalize ❌ — recip(1/Σ) via LUT + broadcast ewmul (or fold 1/Σ into A·V output).

### ★ SDP ALU MAX/MIN PRIMITIVE — CONFIRMED ON SILICON (2026-07-14, from rocket driver, NOT fuzzed)
Reg **0x4070 = REG_DPU_EW_CFG**, field **EW_ALU_ALGO = bits[19:16] (mask 0x000f0000)**. Enum (NVDLA
firmware map_alu_op[] order, our ADD template pins SUM=2): **MAX=0, MIN=1, SUM=2, EQL=3.** ALU-active
config (ADD routing): 0x4040=0x00020040, 0x4048=0x40000000, 0x4070=0x904002c0 | (algo<<16).
VALIDATED (sdp_max_fuzz, a=0 ramp b): algo0→ReLU(b)=max, algo1→min, algo2→sum. rc=0, no wedge, ONE submit.
Env hooks added to ork_npu_probe_i8_mul: ORK_EW_R40/R48/R70. → belongs on wiki (ISA Reference).
NEXT: ork_npu_max_reduce (pairwise-max TREE, log2(n) EW-max passes, or strided) → softmax 2a on-NPU.

### ★ on-NPU MAX-REDUCE — VALIDATED bit-exact (2026-07-14, max_reduce_probe)
Pairwise-max TREE composing EW-max: reduce row of n via log2(n) passes maxing the two contiguous halves.
BUILD on the ADD path (ork_npu_add_i8 with a_scale=b_scale=out_scale=1 => identity int ALU) + ORK_EW_R70
override to EW_ALU_ALGO=MAX(0). BIT-EXACT vs CPU (MUL-template path had a +1/+2 requant bias — use ADD path).
Constraint: add_i8 needs N>=16 => tree reduces n->16 on-NPU, trivial 16-wide tail (fold onto NPU later or CPU).
Env hook added to ork_npu_probe_add_i8: ORK_EW_R70. Reusable: softmax max, max-pool, top-k, clamp.
Primitive collection now: Σ-reduce ✓ · exp ✓ · EW max/min ✓ · max-reduce ✓ · normalize (recip+bcast) = NEXT
(recip via rsqrt(Σ²)=1/Σ [reuse ork_norm_rsqrt_npu] + per-row broadcast scale). Then wire max-reduce as softmax 2a + fuse.

### exact-vs-fast design principle (2026-07-14, measured)
Max-reduce: EXACT(ADD path)=4 passes 1881us/reduce bit-exact; BIASED(MUL path)=8 passes 854541us/reduce
(per-call ACT_RESET) biased-high. EXACT is ~450× FASTER + exact => ship EXACT ONLY, no biased variant.
PRINCIPLE: design primitives to ALLOW an exact/fast flag, DEFAULT exact, add the fast path ONLY when a
measurement shows it's genuinely cheaper AND a caller tolerates the approximation (e.g. softmax max-invariance
could opt into a coarse max/exp). Don't add approximate variants speculatively — measure the gap first.

## FUSION ASSEMBLY — grinding (2026-07-14)
- ✅ M1: BATCHED on-NPU row-max `ork_npu_row_max_i8` (npu.c) — all M rows in log2(N/16) EW-max submits via
  base-offset (b=a+(h/16)*M*16), N->16 on-NPU + 16 CPU tail. BIT-EXACT (max_reduce_probe: 64x256 64/64,
  256x512 256/256; ~1ms/256rows). Header-declared, public. The last reduction primitive.
- M2 (next): wire row_max into the attention softmax step-2a (max on NPU). Scores are fp16→need int8/int16
  quantize-for-max (coarse OK, softmax max-invariant) OR an int16 row_max. Validate end-to-end coherence.
- M3: keep scores resident in NPU buffers (QK^T today writes to host → round-trip); feed max/exp/Σ in place.
- M4: single heterogeneous PC-chain (fp16 MM → SDP max/exp/Σ → MM), normalize (1/Σ) folded into A·V out-cvt.

- ✅ M2: row_max wired into attention softmax 2a (ORK_ATTN_MAX_CPU=1 forces CPU). COHERENT ("Red,Blue,Green").
  Coarse int8 quantize-for-max (softmax max-invariant => exact output). Guards M<=8192 (CPU fallback for
  bigger batches/pp1024+ — chunk later). Profile pp512: SM=4907 [prep=1503 quant=454 exp-NPU=2120 red-NPU=543
  norm=288]; pp512 78.7 t/s. **Softmax max+exp+Σ ALL on NPU now.** Only CPU softmax compute left = normalize
  divide (288ms) + quant glue. sm_q8/sm_maxq scratch added.
- M2.5 (next): normalize on NPU — 1/Σ via reciprocal (rsqrt(Σ²)) + per-row broadcast scale (fold into Pf or
  A·V out-cvt). The awkward one (per-row scalar broadcast — SDP BS per-channel scale or tiling).
- M3: scores resident in NPU buffers (kill round-trip + the CPU quantize/cube-marshal glue).
- M4: single heterogeneous PC-chain.

### ★ PER-CHANNEL SCALE — CONFIRMED ON SILICON (2026-07-14) — the transposed-normalize mechanism
Native per-channel broadcast IS supported on RK3588 — on the EW operand (ERDMA), NOT the BS stage (whose
rocket BRDMA_CFG is simplified to DATA_USE only, no PER_KERNEL). Mechanism: EW-mul out[m][n]=a[m][n]*b[n],
b a length-N vector at EW_BASE (0x5038), with **ERDMA_CFG (0x5034) ERDMA_DATA_MODE bits[31:30] = 0** (keep
DATA_SIZE bits[3:2], i.e. reg=0x00000004). VALIDATED (bs_scale_probe, working MUL path 8x64): DATA_MODE=0 →
per-channel broadcast (all widths scaled by b[n], row-invariant, rc=0); DATA_MODE=1 (template default) =
per-element; 2,3 = hang. Registers from rocket_registers.h; mode pinned empirically (earlier hangs were my
override zeroing DATA_SIZE, not a HW wall). Env hook ORK_EW_R34 on ork_npu_probe_i8_mul. REUSABLE: normalize,
LayerNorm/RMSNorm affine, per-channel requant. => transposed normalize = EW-mul by 1/Σ per-channel vector, native.

### ★ ork_npu_mul_perchan_i8 — GENERAL-GEOMETRY per-channel scale, BIT-EXACT (2026-07-14)
out[m][n]=clamp(a[m][n]*b[n]*mult>>shift), b[N] broadcast across rows. Config: REGCMD_MUL + set_mul_geom(M,N)
+ ERDMA_CFG(0x5034)=0x04 (DATA_MODE=0 per-channel) + zero za/zb/zo (0x4044/0x4074/0x4080) + out gain
(0x4084=mult,0x4088=DIRECT shift, gain=mult/2^shift). **b vector CONTIGUOUS [N]** (NOT surface-strided — the
key fix; channels>=16 read 0 otherwise). BIT-EXACT 8x64/64x256/128x512 (bs_scale_probe). ~80us. Public in header.
Reusable: softmax normalize (b=1/Σ), LayerNorm/RMSNorm affine, per-channel requant. Gotchas learned: MUL-path
0x4088 is DIRECT shift (ADD path is +14); MUL template carries nonzero captured zero-points (must clear).

### ork_npu_mul_perchan_f16 — fp16 per-channel scale, BIT-EXACT (2026-07-14)
Port of the i8 recipe to REGCMD_MUL_F16: b CONTIGUOUS [N] fp16, 0x5034=0x08 (DATA_MODE=0 per-channel +
DATA_SIZE=TWO_BYTE), quant-free (no gain/zero-points). BIT-EXACT (max|err|=0) 8x64/64x256/128x512. Public.
=> the attention normalize engine: out=e*(1/Σ)[perchannel]. Both i8+f16 per-channel scale validated on silicon.

### ★ TRANSPOSED-NORMALIZE MATH — VALIDATED ON NPU (2026-07-14, tnorm_probe) — coherent
Ô[d][m]=Σ_j V[j][d]e[m][j] via ork_mm(A=V^T[DV][K], W=e^T[K][N]) then out[m][d]=Ô[d][m]*(1/Σ)[m] via
ork_npu_mul_perchan_f16 (1/Σ per-query=per-channel). max|err|=0.0002 vs CPU, 0 bad. LESS-INVASIVE than full
transposition: keeps QK^T+max(row_max)+exp+Σ NON-transposed (M2 state), only A·V is transposed (swap operands:
weight=e^T repacked, activation=V^T densified) + per-channel 1/Σ on the [DV][N] output + transposed scatter.
NO width-max, NO transposed QK^T needed. Core math proven; handler wiring is the remaining integration.

### ★★ TRANSPOSED SOFTMAX LANDED — full softmax on NPU, COHERENT end-to-end (2026-07-14)
Wired transposed A·V + per-channel normalize into the FLASH_ATTN handler (fork feat/attn-npu, gated
ORK_ATTN_TNORM_OFF=1 = old path). COHERENT ("Red,Blue,Green") in qwen2.5-1.5b. Profile confirms transposed
path active (norm 298→962 = H mul_perchan submits, dv 100→730 = e^T transpose + V^T densify, sc 105→221).
ENTIRE attention on NPU: QK^T + max(row_max) + exp + Σ(reduce-mm) + NORMALIZE(per-channel 1/Σ) + A·V(transposed).
ZERO CPU compute in the softmax loop. pp512 71.8 t/s (vs 77.8 old-path CPU-normalize, 142 baseline — slower as
expected: normalize=H per-channel-scale submits + transpose marshaling; goal was on-NPU, achieved).
Pool: added wet[h] (e^T weight), Oh16, invS. Handler: 2d captures invS=1/Σ; tnorm_ok gate; transposed A·V
(weight=e^T repacked, act=V^T densified) → Ô[DV][N] → per-head ork_npu_mul_perchan_f16(Ô,1/Σ) → transposed scatter.

## Next: M4 — collapse to a single PC-chain (QK^T→max→exp→Σ→A·V→normalize one submit); reduce H mul_perchan to 1
- Only known-good attention model on board = qwen2.5-7b-instruct-q8_0 (canonical baseline, 3-file).
  7B needs multi-domain footprint (`ORK_DOMAINS` + `ORK_NO_BF`) → then full end-to-end coherence +
  per-op profile + flow chart (the user's follow-on ask). Multi-domain layout is non-deterministic /
  swap-bound (see [[multi-domain-runtime]]) — a separate footprint task.
- OR: re-convert a small qwen3/qwen2.5 attention model with a correct vocab (.239 conversion box).

## Board-ops
- timeout every NPU cmd; SIGTERM never kill -9; wedge→`ssh board 'sudo reboot'`; hard-wedge→Rock 5B Plug.
- never copy macOS binaries to board — rsync SOURCE, build natively.
