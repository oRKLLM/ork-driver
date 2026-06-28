# WIP: rkllm K-handling / prefill-utilization RE (decode of 0x107c / 0x1044 / 0x1040 / 0x1010)

Goal (as briefed): decode rkllm's "fine K-slice" that supposedly gives ~6× MAC utilization vs ork,
and produce a replication recipe for `synth_i8`.

## TL;DR — the premise is WRONG; there is NO fine K-slice. The real divergence is the M axis.

The fresh rkllm capture (`~/fused2.log`, Qwen2.5-7B-Instruct-w8a8, K=3584 N=1216) shows rkllm
runs **the full K=3584 in every submit/task** (no K-slice, no on-NPU partial-sum reduction).
`0x1024`=K, `0x1034`=K (full weight row), `0x1088`=K, `0x1030`=K*N (full weight bytes) — identical
to ork's full-K path. **K is not partitioned at all.**

What actually differs: **rkllm runs large M per task (M up to 36) with `0x1010` held CONSTANT at
0x20**, whereas ork clamps the M-tile to `R-1 ≈ 15` (because ork ties `0x1010 = 16*min(M+1,R)` to a
CBUF row budget). So the lever is the M-tile size, not K-slicing — and ork already knows the exact
register program (its M=1 decode path is bit-identical to rkllm's per-task program); it just doesn't
USE it for M>1.

## Decoded field semantics (rkllm, K=3584)

Encoding: value = `(word0>>16) | ((word1&0xffff)<<16)` (same as ork `setr`).
`mc` = `0x102c` = the per-task M (token count).

| field | rkllm value | formula | ork synth_i8 value | divergence |
|---|---|---|---|---|
| 0x1024 | K | `((K-1)<<16)\|K` | same | none — full K |
| 0x1034 | 3584 | `K` (full weight row) | same | none |
| 0x1088 | 3584 | `K` | same | none |
| 0x1030 | K*N | `K*N` | same | none |
| **0x1044** | **56*M** | **`(K/64)*M` = ceil(K/64)*M** | `ceil(K/64)` (const, no *M) | **rkllm multiplies by M** |
| **0x107c** | **~4*M** | scales with M (4,8,16,24,32,40,48,56,64,96,128…) | `K/16` (=224 const) | **rkllm is M-scaled, NOT K/16** |
| **0x1010** | **0x20 (const)** | constant 32 for ALL M | `16*min(M+1,R)` (grows to 0x100) | **rkllm never inflates it** |
| 0x1040 | 0xb1/0xa2/0x93/0x84 | `0xb1 - 0x0f*(ceil(M/8)-1)` | `177-15*(scale-1)…` keyed to K & M/64 | different keying (see below) |
| 0x102c/0x1020/0x1084 | M | M, (M<<16)\|1 | M, 0x10000\|M | encoding nuance only |

### The 0x1040 schedule — EXACT closed form (this is new, and clean)
`0x1040 = 0xb1 - 0x0f*(ceil(M/8) - 1)`, i.e. M-groups of 8, step −15 per group:
- M 1..8  → 0xb1 (177)
- M 9..16 → 0xa2 (162)
- M 17..24→ 0x93 (147)
- M 25..32→ 0x84 (132)
- (M=36 observed 0x84, i.e. clamped/reused — only outlier; predicted 0x75)
Matches every captured M=1..32 exactly. Note this is keyed to **M**, not K — confirming the schedule
selects the internal M-tile partition, not a K reduction.

### 0x107c: scales with M, exact rule still fuzzy
Observed (M→0x107c): 1→4, 2→8, 4→16, 6→24, 8→32, 10→40, 12→48, 14→56, 16→64, then noisier for the
multi-task submits (20→128 or 56, 24→96, 28→64, 32→128, 36→96). For M≤16 it is cleanly `4*M`. The
larger-M values appear to depend on the M-subtile *position within the submit* (the 36-task submits
reuse a few 0x107c values across tasks). Pinned: `0x107c = 4*M` for a single full-M task (M≤16);
the multi-task variation is a scheduler artifact, not load-bearing for the single-task replication.

## ork's K-handling (contrast)
`synth_i8` (src/npu.c:404): full K (≤4096 via Bf, else K-split at KS=1024). 0x1044=ceil(K/64),
0x107c=K/16, 0x1088=K, 0x1010=16*min(M+1,R) with R=pow2_floor(2*cbuf/K). At K=3584, cbuf=32768:
R=pow2_floor(65536/3584)=pow2_floor(18)=16 → ork clamps the M-tile to **R-1 = 15 rows**
(see the HARD CONSTRAINT comment at src/npu.c:1972). The M=1 decode path (npu.c:1926/1938/3085)
OVERRIDES 0x1040=0xb1 and leaves 0x1010=0x20 — which is **exactly rkllm's M=1 program.**

## THE MECHANISM (revised)
NOT "fine K-slice → tiny working set → big R". K is full in both. The difference:
- rkllm packs **M up to 36 tokens into one task** (0x102c=M), full K, `0x1010` fixed at 0x20.
  One weight stream (K*N int8) is loaded once and reused across all M rows → weight-load amortized
  over M=36.
- ork packs **only M≤15 per tile** (R-1), so for a 48-token prefill chunk it issues ~4× as many
  tiles, re-streaming the K*N weight each time → weight DMA dominates → low MAC utilization.

So if ork's reported ~6× gap is real, the most likely cause is **weight re-streaming from the small
M-tile**, not K-slicing. Bigger M per submit at fixed 0x1010 is the lever to test.

⚠️ CAVEAT: ork's `0x1010 = 16*min(M+1,R)` and the R-1 clamp were derived from earlier captures and
encode a real bit-exactness constraint (npu.c:1972-1977: "an M-tile of more than R-1 rows spills past
the CBUF-resident window and the rows beyond R-1 compute against the wrong K-partition"). rkllm
holding 0x1010=0x20 at M=36 means EITHER (a) 0x1010 is not the CBUF-row cap ork thinks it is, or
(b) the constant-0x20 + M-scaled 0x1044/0x107c is a DIFFERENT, correct scheduler mode that ork has
never used. This must be settled by a bit-exact experiment before any production change.

## REPLICATION RECIPE for synth_i8 (to test)
Add a "rkllm-mode" M-tile that, for a full-K int8 submit, sets:
- `0x1010 = 0x20` (constant, NOT 16*min(M+1,R))
- `0x102c = M`, `0x1020 = (M<<16)|1`, `0x1084 = (M<<16)|1`  (rkllm uses low-half=1, not ork's |M)
- `0x1044 = (K/64) * M`   (NOT ceil(K/64))
- `0x107c = 4 * M`        (NOT K/16)
- `0x1040 = 0xb1 - 0x0f*(ceil(M/8)-1)`, floored at 0x1b
- 0x1024/0x1030/0x1034/0x1088 unchanged (full K, full weight)
- output dims 0x4034/0x405c/0x3014 = M-1 as usual
Then raise the prefill M-tile ceiling from R-1 to e.g. 32/36 and measure bit-exactness + TOPS.
Guard with validate_regcmd + fall back to the current path.

## EXPERIMENT RESULTS (tools/mtile_probe.c, RK3588 board, governors=perf, NPU idle)

Ran ork-current M-tile mode (mode0) vs the rkllm-field-override mode (mode1) at full K, sweeping M,
each validated bit-exact vs a CPU int8 reference, timing the warm submit -> effective GOPS.

K=2048 N=128 (ork R=pow2_floor(2*32768/2048)=32, so ork allows M up to 31):
```
   M | mode0 (ork)            | mode1 (rkllm-fields)
   1 | OK  80us    7 GOPS     | OK   80us  7 GOPS
   8 | OK  79us   53 GOPS     | WRONG
  16 | OK  80us  104 GOPS     | WRONG
  32 | OK  88us  191 GOPS     | WRONG
  36 | OK  93us  202 GOPS     | WRONG
```
K=3584 N=128 (ork R=pow2_floor(65536/3584)=16, ork clamps M-tile to R-1=15):
```
   M | mode0 (ork)            | mode1 (rkllm-fields)
   8 | OK  98us   75 GOPS     | WRONG
  16 | OK 105us  140 GOPS     | WRONG
  20 | WRONG (ork breaks)     | WRONG
  36 | WRONG                  | WRONG
```

### What the experiment PROVES
1. **The per-submit cost is a fixed floor (~80-100us) independent of M** within the resident-row
   budget. Bigger M = more MACs in the same wall-time = linearly higher GOPS. At K=2048, GOPS goes
   7 (M=1) -> 104 (M=16) -> 202 (M=36): a **~29× utilization lift purely from amortizing the submit
   floor over more M rows.** THIS is the real lever — not K-slicing.
2. **ork's wall is its M-tile cap at R-1.** mode0 is bit-exact while M<=R-1 and goes WRONG beyond
   (K=3584: breaks at M=20, since R=16). At K=2048 (R=32) it's correct to M=36. The production
   prefill path (npu.c:2081 Tier 1c-ii) sets `chunk = min(mg_max*64, R-1, M)` -> at K=3584 it
   M-tiles at <=15, re-paying the ~100us floor every 15 tokens; rkllm pays it every 36.
3. **rkllm's mode is NOT reproducible by those 4 register overrides alone.** mode1 is WRONG from
   M=2 up. rkllm's large-M submit is a MULTI-TASK submit where all tasks share one A/B/C base
   (fused2.log: 6 tasks, same 0x1070/0x1110/0x4020, mc=2,4,6,10,12,14) — the per-task M-row OFFSET
   is encoded in a register not yet mapped (likely the 0x1020/0x1084 low half, or a CNA feature
   base-offset). So replicating rkllm needs the multi-task M-subtile offset scheme, not just the
   single-task field formulas.

### The gap, quantified
ork is M-capped at R-1 = pow2_floor(2*cbuf/K)-1, which SHRINKS as K grows: K=2048->M<=31,
K=3584->M<=15, K=4096->M<=15. The Qwen2.5-7B layers are K=3584, so ork is stuck at M<=15 (~140
GOPS) while rkllm runs M=36 (extrapolating the same floor: ~370 GOPS). That ~2.6× at the matmul
level is consistent with the prefill-throughput gap. The "~6×" in the brief likely also folds in
rkllm's 3-core fan-out (each core a different N-slice/M-subtile) + warm-weight pipelining.

## REPLICATION RECIPE (revised, the real one)
The lever is "raise the M-per-submit ceiling at large K." Two routes:
- **(A) Lift ork's R-1 clamp where the hardware still computes correctly.** The experiment shows
  ork's current single-task program is only correct to M=R-1. To exceed it, the CBUF-resident row
  window must be re-tiled OR the rkllm multi-task offset scheme adopted. Needs the offset register
  (next RE step: re-capture rkllm with M varied and diff 0x1020/0x1084/CNA-base across the subtasks
  that share C-base).
- **(B) Multi-task M-subtile submit (rkllm's actual mechanism):** one RKNPU_SUBMIT, N tasks, each a
  different M-subtile of the SAME A/B/C base, with a per-task row offset. This is the documented
  "task_number>1 wedges" path (tools/batch_probe.c) — but rkllm DOES use task_number=6/36
  successfully, so the wedge was a config bug (subcore_task layout), not a hard limit. Re-investigate
  the multi-task submit using rkllm's exact subcore_task split (fused2.log: 36-task submit =
  subcore[0..2] task_start=24 task_number=2 each -> the kernel DOES accept multi-task when the
  subcore split matches). THIS is the highest-value next step.

## STATUS / NEXT
- [DONE] Decoded rkllm's scheme from fused2.log. Premise (fine K-slice) REFUTED; lever = M-per-submit.
- [DONE] Exact field semantics: 0x1044=(K/64)*M, 0x107c=4*M, 0x1010=0x20 const,
  0x1040=0xb1-0xf*(ceil(M/8)-1). Divergence from ork pinned.
- [DONE] Bit-exact experiment: proved the fixed-floor amortization mechanism (~29× GOPS lift M1->M36
  at K=2048); proved ork's M<=R-1 wall (K=3584 -> M<=15); proved rkllm-field-override alone is WRONG
  (rkllm uses a multi-task M-subtile submit, offset reg unmapped).
- [PARTIAL] Diffed the 6 tasks of SUBMIT#0 (share A/B/C base, mc=2,4,6,10,12,14). Varying regs:
  0x1020/0x1084=(mc<<16)|1, 0x1028/0x102c=mc, 0x1040=ceil(M/8) sched, 0x1044=(K/64)*mc, 0x107c=4*mc,
  0x1080=-3*mc (signed), 0x4024=16*mc, 0x40c0=128*(mc/2), 0x3014/0x405c/0x4030=mc-1. NOTABLY there is
  NO plain byte-offset register that varies like a row base (every varying reg scales with mc as a
  COUNT/extent, not a start). So either (i) the 6 tasks are alternative whole-matmul programs the
  scheduler picks among (unlikely — task_number=6 means all execute), or (ii) the M-row base offset
  lives in the feature/output IOVA itself per task (but 0x1070/0x4020 are identical across tasks) or
  in a CNA reg not in the 108-reg dump. RESOLUTION NEEDED: re-capture rkllm at two different total-M
  prompts and diff — the register that tracks cumulative-M-start is the offset. (mtile_probe mode1
  failing from M=2 is consistent: the offset/extent split isn't captured.)
- [NEXT] Finish mapping the M-subtile offset, then implement route (B) multi-task M-subtile submit;
  re-test task_number>1 with rkllm's exact subcore_task split (subcore[0..2] task_start=24 number=2).
- Board: isolated checkout ~/ork-st (do NOT touch ~/ork-driver). Logs: ~/fused2.log ~/fused_trace.log.
  Experiment tool: ~/ork-st/mtile_probe (built). Decode scripts: scratchpad/decode*.py, compare.py.
- Commit LOCAL only; do not push. (local: npu.c+header+Makefile+tools/mtile_probe.c added.)
