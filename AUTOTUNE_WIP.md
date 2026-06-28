# WIP: NPU matmul tiling AUTOTUNER (Qwen2.5-7B prefill)

## Goal
Systematically find the per-shape tiling config that maximizes NPU prefill utilization
(lever toward librkllmrt 73.6 tok/s pp), instead of hand-iterating. Two phases:
- **Phase A**: search the EXISTING knobs ork already parameterizes (M-tile via SMALLTILE,
  N-subtile, core count, chain-ksplit budget) at the `ork_mm_run_i8` level, bit-exact-gate
  vs CPU ref (once/shape), pick min-time, cache per (SoC, shape).
- **Phase B** (harder, WEDGE ZONE): add a K-tile knob (`ORK_KTILE`) — slice K into K_tile
  chunks so R(K_tile)=pow2_floor(2*cbuf/K_tile) is BIGGER → larger M-tile allowed → more
  M-amortization, esp. ffn_down K=18944. Gate it, validate_regcmd + fallback, bit-exact MANDATORY.

## Board / checkout
- Board 10.3.0.236 (`board`). ISOLATED checkout `~/ork-st` (rsync copy, NOT git; do NOT touch
  ~/ork-driver f26a3fd). Source synced from local branch `re/rkllm-mtile-decode`.
- Governors performance (dmc + cpu4). orkllm UP (little cores, no contention) — leave it.
- Build native on board: `cd ~/ork-st && make <target>`. Run: `sudo timeout -s INT <s> ./<bin>`.
- Safety: pgrep+WAIT, SIGINT only, wedge→`sudo reboot`+governors+test_matmul.

## KEY PRIOR FINDINGS (from KSLICE_RE_WIP.md, settle the design)
- rkllm has NO secret K-slice; full K every submit. ork ALREADY chains M-tiles into ONE submit
  (chain-prefill, npu.c:1956) → per-submit floor amortized once/chunk. Route B (multi-task
  M-offset) does NOT exist.
- The single-task M ceiling is R = pow2_floor(2*cbuf/K), which SHRINKS as K grows. K=3584→R=16
  (M-tile cap 15). The lever to raise R = K-tiling the CBUF residency (Phase B).
- ork ALREADY K-tiles in the weight layout: int8 KS=1024 (npu.c:2454), partials accumulated
  host-side fp32. So K-tiling EXISTS structurally; Phase B = make K_tile a search knob and see if
  smaller K_tile (bigger R, more M-amortization) wins net of more slices/accumulation.
- mtile_probe mode0 (ork program, single core, bit-exact): K=2048(R=32) M=36→205 GOPS;
  K=3584(R=16) M=16→91, WRONG at M>=20. Single-task ceiling ≈ R.

## ork knob map (what synth_i8/run_i8 expose)
- cbuf_elems=32768, nmax=8192, soc.ks=2048 (fp16); int8 KS hardcoded 1024 (npu.c:2454).
- R_int8 = pow2_floor(2*32768/K). K=3584→16, K=1024→64, K=512→128, K=18944→3(→R=2 after pow2).
- Full-K chain-prefill path: K<=4096 & Bf present. M-tile chunk capped R-1.
- Wide-K K-split path: K>4096, KS=1024 slices, chained (use_chain_ksplit).
- Env knobs: ORK_SMALLTILE/_M/_N, ORK_CHAIN_PREFILL, ORK_CHAIN_KSPLIT/_MB, core budget
  (ork_npu_set_core_budget?), ORK_KTILE (NEW, phase B).

## 7B prefill shapes (M=256)
| name | K | N | path |
|---|---|---|---|
| Q/O proj | 3584 | 3584 | full-K |
| K/V proj | 3584 | 512 | full-K |
| gate/up | 3584 | 18944 | full-K (N-tiled, nmax=8192) |
| down | 18944 | 3584 | wide-K K-split |

## STATE
- [done] explored npu.c, synth_i8, chain-prefill, chain-ksplit, probe_mtile_i8. Built ork-st,
  test_matmul PASSES (incl ChainPrefill M=256 K=18944 N=3584).
- [next] write tools/autotune.c (Phase A), build, run on shapes.

## PHASE A RESULTS (autotune, M=256, bit-exact-gated OK, median GOPS)
| shape | K | N | 1c | 2c | 3c | best |
|---|---|---|---|---|---|---|
| q_proj  | 3584 | 3584 | 256 | 549 | 700 | 3c 700 |
| kv_proj | 3584 | 512  | 159 | 420 | 479 | 3c 479 |
| o_proj  | 3584 | 3584 | 255 | 548 | 698 | 3c 698 |
| gate_up | 3584 | 18944 | 264 | 563 | 731 | 3c 731 |
| down    | 18944| 3584 | 498 | (pending) | (pending) | — |
- SMALLTILE_M sweep (q_proj,kv_proj @3c): STM 8→433, 12→593, 15→700, 16→697, 32→702.
  => SMALLTILE gives NO win: chunk is clamped to R-1=15 for K=3584 (the doc'd CBUF cap), so
  STM>=15 all converge to the default. STM<15 is WORSE. The DEFAULT is already at the R ceiling.
- => Phase A finding: best config = 3 cores, default M-tile. The existing knobs are saturated;
  the only lever left to raise per-core util is K-tiling to lift R (Phase B).
- down (K=18944) already 498 GOPS at 1 core: K-split (KS=1024 → R=64) already gives big M-tiles.

## PHASE B IMPLEMENTED (ORK_KTILE, gated, committed-local)
- int8_ks(c) helper reads ORK_KTILE (mult of 32, else ignored→1024). Wired into pack KS, mcworker
  KS, run_multicore alloc KS, single-core KS. Bf (full-K) SUPPRESSED when KTILE<K so K=3584 takes
  the K-split path → per-slice R=pow2_floor(2*32768/kt) bigger than full-K's R=16.
  kt=512→R=128, kt=1024→R=64, kt=1792→R=32 (vs full-K R=16 cap).
- Reuses the PROVEN per-tile K-split accumulate (bit-exact for arbitrary Kp); no new regcmd;
  validate_regcmd on every program. WEDGE-SAFE. Bit-exact MANDATORY — autotune gates it.
## PHASE B RESULTS (KTILE sweep, 3 cores, M=256, bit-exact OK)
| shape | default GOPS | KT=512 | KT=1024 | KT=1792 | best |
|---|---|---|---|---|---|
| q_proj  | 700 | 725 | **1118** | 629 | KT=1024 +60% |
| o_proj  | 698 | 729 | **1104** | -   | KT=1024 +58% |
| gate_up | 731 | -   | **994**  | -   | KT=1024 +36% |
| kv_proj | 479 | 380 | 476      | -   | default (N-bound, no win) |
| down    | 1413 (already K-split KS=1024) | — | — | — | default |
- WIN: KTILE=1024 (R 16→64) lifts the wide K=3584 matmuls 36–60%, BIT-EXACT. KT=512 (R=128) and
  KT=1792 (R=32, non-sched) are worse — KT=1024 is the sweet spot (matches the proven sched Kp).
- COMMITTED LOCAL (workstation branch re/rkllm-mtile-decode): 60ccdf6.

## END-TO-END BENCH (pp256, % of librkllmrt 73.6)
- .orkpack CAVEAT: ORK_KTILE is a PACK-TIME layout knob (int8_ks() drives pack KS). ork_mm_load_i8
  (the .orkpack path) hardcodes KS=1024 and the blob is dumped at KS=1024, so a loaded weight won't
  honor KTILE. The end-to-end "after" must FRESH-PACK (ork_mm_pack_i8) for KTILE to apply, OR the
  .orkpack format must carry KS (open follow-up). Bench: before=default build; after=ggml-ork built
  against KTILE driver + ORK_KTILE=1024 + fresh pack.
### RESULT (warm pp256, GGML_NO_REPACK=1, -t4, r3, governors max, orkllm up, fresh-pack)
- BEFORE (default):    pp256 = 11.96 ± 0.01 t/s = 16.2% of librkllmrt 73.6
- AFTER (ORK_KTILE=1024): pp256 = 11.46 ± 0.02 t/s = 15.6% of 73.6
- => HONEST: the matmul-level KTILE win (36–60% GOPS) does NOT survive end-to-end; pp256 is
  flat/slightly worse (−4%). The model runs CORRECTLY (valid t/s, no NaN/crash) → KTILE is
  bit-exact end-to-end. The extra K-slicing adds host overhead (per-slice A re-tile + extra
  accumulation passes + more regcmd programs) that cancels the per-submit GOPS gain. PREFILL IS
  NOT BOUND BY THE MATMUL COMPUTE KTILE ACCELERATES — the gap to 73.6 is host/ggml-ork-side
  (quant/dequant, submit-floor count, threadpool), consistent with the decode-bottleneck finding.
- KTILE stays GATED-OFF by default (zero risk; the win is real at the matmul level for any future
  path that IS matmul-GOPS-bound, e.g. M>1 batched verify).
- Vendored ork-driver in ~/llama.cpp restored after bench (backup /tmp/npu.c.bak). KTILE committed
  in workstation repo 60ccdf6 (NOT pushed).
