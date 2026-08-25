# WIP — int4 NONBLOCK doorbell: ROOT-CAUSED as a DISPATCH FAILURE (2026-08-25)

## ★ ROOT CAUSE (hardware-confirmed): the job is accepted and NEVER STARTS

A custom kernel watchdog (patch 05, below) samples the PC completed-task counter per core at 1 kHz while a
job is outstanding. On every stall it reports:

```
core 0 NO TASK PROGRESS for 64 samples (64000 us), task counter stuck at 0x0, job age 114955 us
core 1 ... task counter stuck at 0x0, job age 115024 us
core 2 ... task counter stuck at 0x0, job age 115004 us
```

**The PC task counter is 0** — the hardware never executed a single task. The submit returns rc=0 and the
job simply never dispatches. This is the registry's long-standing "accepted but never dispatched
(submit_rc=0, hw_elapse=0)" note, now confirmed from a hardware counter instead of inferred.

**CONTROL (essential — a register that always read 0 would look identical):** with the threshold lowered to
8 ms, a HEALTHY ~30 ms job would report if the counter never advanced. Measured: **3 reports across ~900
core-jobs** (matching the 2 observed stalls), not hundreds. So the counter does advance on healthy work and
the offset is right.

**This CORRECTS the earlier "partial mid-write" reading in this document.** Cores are assigned CONTIGUOUS
op ranges (`lo = i*S/nc, hi = (i+1)*S/nc`), so a core that never runs leaves ITS contiguous slice of the
output at the sentinel — which presents exactly as "one contiguous run, ending at the last element,
starting on a 64-element boundary". Nothing was ever written and then stopped; one or more CORES never ran.

**What this means for the fixes measured below:** the drain barrier and the (accidental) aggressive reaping
both help because they clear stale/unretired job state before the next submit — i.e. they improve DISPATCH
success, not write completion. That is consistent with every measurement in this document.

**All THREE cores report simultaneously with counter 0** — the whole submit set fails to dispatch, not one
core losing its job.

### Kernel watchdog: power safety took three attempts (record so nobody repeats them)

- **v1** read NPU MMIO directly from the hrtimer callback. A register read on a powered-down block is an
  external abort => instant HARD hang (no ping, power-cycle required). It hung the board on its first NPU
  workload.
- **v2** guarded with `pm_runtime_get_if_in_use(rknpu_dev->dev)`. STILL WRONG on RK3588: the driver keeps
  its OWN refcount (`power_refcount`/`power_lock`) and `rknpu_power_off()` puts the three PER-CORE genpds
  (`genpd_dev_npu0/1/2`) directly, so a reference on the parent does not hold them up. It survived only
  because the deferred power-off keeps the NPU up during a busy run — luck, not correctness.
- **v3 (correct)**: the hrtimer only sets cadence and `queue_work()`s; ALL MMIO happens in PROCESS context
  holding `power_lock`, which `rknpu_power_off()` must also take, so the genpds cannot drop mid-read.
  Deliberately does NOT call `rknpu_power_get()` — that powers the NPU ON just to read a counter and would
  pin it up forever; check `power_refcount > 0` and skip instead. `mutex_trylock` so the watchdog can never
  delay a submit. Measured: `pm_skip = 332 / ~35000` samples (0.9%), so coverage is essentially unaffected.
  Entirely driver-side; nothing in userspace has to guard anything.

**Immediate lever:** the watchdog detects this with certainty in 8-64 ms (counter==0 AND job outstanding),
versus the current 614 ms guessed window followed by a device-wide ACT_RESET. Wiring that back to userspace
turns ~3.6 s per miss into ~10 ms + a targeted resubmit, without needing to know WHY dispatch fails.

## (superseded framing below) permanent mid-write round stall

**Status 2026-08-25.** Active RE track, promoted to PRIMARY by the user: prefill/tiered-precision work is
blocked behind it. Scratch doc — fold into the wiki and delete when the stall is understood.

## The defect

On the 27B W4A4 multi-domain prefill, 1–4 doorbell rounds per run never complete. Each costs ~3.6 s via
`orki_mc_recover_resubmit` (ACT_RESET + re-anchor + resubmit). Regression over 12 runs:
`time = 23.01 s + 3.61 s x misses` (R² 0.967). The miss-free baseline, 23.0 s, reproduces to 0.1 s — so
**eliminating the stall is worth ~23%** of prefill; faster detection is worth only ~4%.

Numerics are never wrong: PPL = 10.6780 on every run at every setting. The recovery always produces the
correct answer, so this is a throughput defect, not a correctness one.

## What the failing round looks like (measured, `ORK_MC_DIAG=1`)

The NPU writes a **contiguous prefix** of the output surface and then stops forever:

```
op #15/64 dt=i4 esz=2 i4batch=0 | sentinel 9920/10240 (96.88%) first=320 last=10239 runs=1
op #7/64  ...                   | sentinel 1856/10240 (18.12%) first=8384 last=10239 runs=1
op #20/64 ...                   | sentinel  128/10240  (1.25%) first=10112 last=10239 runs=1
op #0/3   i4batch=1             | sentinel 98304/709632 (13.85%) runs=1536 runlen=64 period=256
```

- `runs=1`, always ending at the LAST element, always starting on a multiple of 64.
- The BCHAIN (`i4batch=1`) STRIDED pattern is probably the SAME failure seen through a tiled layout:
  contiguous in PROGRAM order becomes period-256/runlen-64 in ELEMENT order. Treat as one mechanism until
  shown otherwise.
- Stop point is **non-deterministic** (1, 5, 24, 118, 131 … rows) and the op index is random across the
  chain (#0, #1, #5, #6, #7, #8, #12, #13, #15, #16, #18, #20, #63) — including the chain's FIRST op.

## Ruled out — each by measurement, do not re-litigate

| hypothesis | how it died |
|---|---|
| miss window too tight | miss rate flat across 300 ms / 614 ms / 2 s / **30 s** windows |
| the round would finish if we waited | at a 30 s window it is STILL absent — 300x normal op duration |
| int16 sentinel value collision (0x7fff IS a reachable W4A4 accumulator) | pattern is one contiguous run, not scattered singletons; and see next row |
| stale CPU read / cache visibility | kernel `orki_bsync(FROM_DEVICE)` re-count recovers NOTHING (3904 vs 3904, etc.) |
| missing `dsb` between `dc civac` and the load | real ARM bug, FIXED anyway (kept), but miss rate unchanged 2/0/3/2 |
| kernel killed the job | dmesg during a miss shows NO fault, reset, timeout — only domain switches |
| domain switch racing an in-flight job | `ORK_DOM_SETTLE_US` 1 ms → 20 ms → 60 ms: 5/8/6 misses, no trend |
| seed ordering (cleans not complete before submit) | all three int4 seed sites already end in `dsb ish` |
| a forbidden shape / bad M,K,N | non-deterministic stop point + random op + correct PPL on resubmit |

**Caveat on two of these.** (a) A settle is a *sleep*, so it only rules out "the job needed more time", not a
switch racing work that is already wedged. (b) Recovery does ACT_RESET + resubmit, so the op re-runs cold and
standalone — identical PPL proves the shape is fine IN ISOLATION, NOT that the in-chain op boundary is sound.

## ROOT CAUSE NARROWED (2026-08-25): the misses are TWO SHAPES, not a random rate

12 misses over 4 reps, and EVERY one is one of exactly two shapes — zero misses on anything else, across
~2000 submits / ~1.1M programs per run and 7 different domains (0,2,3,4,5,7,9):

| shape | count | path |
|---|---|---|
| `K=17408 M=1 N=5120 Sk=2` | 8 | per-row doorbell (`ork_i4_dyn_begin_mc`), the 27B `ffn_down` |
| BCHAIN, chain `N=8192` (`i4batch=1`) | 4 | `run_i4_bchain_db` |

So the registry's "~1/4000 dispatch-drop" is an AVERAGE OVER A BIMODAL POPULATION: nearly every op never
fails, and two wide shapes fail repeatedly. Per SUBMIT our rate is ~1/500 — 8x worse than 1/4000 — while
per PROGRAM it is ~1/300000. Quoting a single rate for this defect is misleading; quote the shape.

Not domain-specific (7 domains), not position-specific (op index random, submit counter at miss ranges
48..1973), and not correlated with anything else measured.

**Both failing shapes are the widest tensors, with UNEVEN slicing.** `ORK_I4_KS = 10752`, so K=17408 splits
`10752 + 6656` — the first slice sits EXACTLY at the validated single-submit K ceiling and leaves a 6656
remainder. With `nmax` N-tiling on top, the op's programs are the cross product of {10752, 6656} x {nmax,
N-remainder}. This is the same "marginal geometry exposed as a SLICE REMAINDER" class the registry already
documents for the BCHAIN H table (K=2560 standalone-correct but non-deterministically wrong as a remainder
of K=18944; fixed by backing H off; MECHANISM UNKNOWN).

**Prediction to test:** make the slicing even (or back `ORK_I4_KS` off from the ceiling) and the stalls
should stop. COST: `ORK_I4_KS` is used at PACK time (`i4/pack.c` slices the weight into `Bb` tiles), so
changing it invalidates the 11 GB `.orkpack` and needs a rebuild ON THE SBC.

## STANDALONE REPRODUCER (2026-08-25) — `tools/re/i4_widek_stall_probe.c`

`make i4_widek_stall_probe` — no model, no 11 GB pack, ~1-2 min per data point instead of a 30 min
multi-domain cycle. Drives the REAL path (`ork_i4_mm_pack` + `ork_i4_mm_run`), so the Sk=2 slicing matches
the model. Detection is BY WALL TIME (a stalled rep pays the miss window + reset, ~618 ms or ~2149 ms, vs
~30 ms healthy) because the recover loop makes the OUTPUT correct — correctness cannot see this defect.

```sh
sudo ORK_NPU_LOCK_WAIT=1200 tools/util/npu_guard.sh -- env ORK_MM_TIMEOUT=3000 \
  ./i4_widek_stall_probe <reps> <K> <N> <slow_ms> <ndom> <M>
# the failing config:  ./i4_widek_stall_probe 4000 17408 5120 100 2 64
```

**Quantified (n=400/cell, K=17408 N=5120):**

| M (chain depth) | domains | stalls/400 | rate |
|---|---|---|---|
| 1 | 1 | 0 | 0.00% |
| 64 | 1 | 3 | 0.75% |
| 1 | 2 | 2 | 0.50% |
| 64 | 2 | 10 | 2.50% |

Both chain depth and domain count raise the rate and they compound. Per OP (not per rep) the M=1/ndom=2
cell is the worst (1/200 vs 1/8533 for M=64/ndom=1) => **domain switching is the dominant per-op factor,
chain depth secondary.**

**Stalls come in CONSECUTIVE PAIRS**: ~618 ms (the learned window) immediately followed by ~2149 ms (the
2 s catch-all). So the recovery's ACT_RESET leaves the NEXT round primed to stall too, and one event really
costs ~2.8 s. Worth attacking on its own — it doubles the price of every miss.

### Sample-size discipline (learned the hard way here)

At a ~2.5% rate, **n=400 gives only ~10 events (Poisson +-32%)** — enough for direction, NOT for magnitude.
Two errors this caused, both corrected: (1) a single 60-rep run reproducing 2/60 while another 60-rep run of
the IDENTICAL config gave 0/60, which briefly looked like "all three ingredients are required" — it was
noise; (2) an even-vs-uneven K test (17408 vs 21504 vs 10752) that returned 0/60 everywhere INCLUDING the
positive control, so it measured nothing. **Use >=4000 reps (~100 events) for any rate comparison; reserve
n=400 for direction. Always run the positive control in the same pass.**

## CONFIRMED CONTRIBUTOR: the domain-switch retirement race (n=4000, 2026-08-25)

| `ORK_DOM_SETTLE_US` | stalls / 4000 | rate | median rep |
|---|---|---|---|
| 0 | 210 | **5.25%** | 30.3 ms |
| 10000 (10 ms) | 80 | **2.00%** | 40.1 ms |

210 vs 80 is ~7 sigma (Poisson sigma 14.5 / 8.9). The `#54 RETIREMENT BARRIER` comment in `orki_dom_activate`
was RIGHT about the mechanism: a doorbell op is "done" when its output cell lands, which is BEFORE the kernel
retires the job, so switching the IOMMU domain races the still-in-flight job. **This reverses the earlier
"settle makes no difference" entry** — that test was 3 model runs at ~2 events each, i.e. no power at all.

**But the blind `nanosleep` is the wrong instrument.** 10 ms buys a 2.6x rate cut and costs +33% on the
median rep (30 -> 40 ms) on EVERY switch, and still leaves 2%. A fixed delay can only trade throughput for
probability; it can never be correct, because it does not actually know when the job retired.

**Proposed fix — a real drain barrier.** Replace the sleep with something that WAITS for retirement:
a tiny BLOCKING submit on each core that had work returns only once the kernel has retired the prior job on
that core (jobs are queued per core). Cost ~167 us (the measured submit floor) versus 10 ms — ~60x cheaper
than the sleep AND an actual barrier rather than a guess. Validate on the probe at n>=4000 against both
`ORK_DOM_SETTLE_US=0` (210/4000) and `=10000` (80/4000); target is ~0 at ~30 ms median.

**The rate PLATEAUS — there is a second cause (n=4000/cell, slow=400ms):**

| settle | stalls/4000 | rate |
|---|---|---|
| 0 | 210 | 5.25% |
| 10 ms | 80 | **2.00%** |
| 30 ms | 110 | 2.75% |
| 100 ms | 118 | 2.95% |

It drops 0 -> 10 ms then FLATTENS in a 2-3% band (80 vs 110 is only ~2.2 sigma; 10/30/100 ms are barely
distinguishable, and longer is if anything slightly worse). So the switch race accounts for ~3.25 of the
5.25 points (~60%) and **a residual ~2% is a DIFFERENT defect**. A drain barrier should reach ~2%, NOT 0 —
do not expect it to close the issue, and re-measure the residual once it lands.

**THRESHOLD TRAP (cost one wasted 25-min sweep).** The stall detector is a wall-time threshold, and the
settle INFLATES every rep: at settle=100 ms the median rep is 130 ms, so a 100 ms threshold flagged
4000/4000 as "stalls". Any settle sweep MUST use a threshold above the inflated baseline and below the
~618 ms stall signature — 400 ms works across 0..100 ms of settle. The 0/10 ms cells were unaffected
(medians 30/40 ms) so those counts stand.

**Loose end:** `mismatched/rc-err runs: 1` appeared at settle=30 ms in both passes, where every other cell
reports 0. A CORRECTNESS event, not just a stall. n=1, unexplained — watch for it.

## Drain barrier: BUILT, 5.8x on the probe, but NO end-to-end gain (2026-08-25)

`orki_dom_drain` (src/npu/core/domain.c) — tiny BLOCKING submit per core before a domain switch, reusing
this domain's existing scratch (c->Af for A/B/C, c->regcmd/c->task, which the mc doorbell path does not
use). `ORK_DOM_DRAIN=0` restores the old blind sleep. DEFAULT ON. `make test` PASSES; attest refreshed.

Order-controlled B-A-B on the probe, n=4000/cell: **drain 159/8000 = 1.99%** vs **1 ms sleep 1376/12000 =
11.47%** — 5.8x fewer stalls, median UNCHANGED (30.4 vs 31.3 ms), mean 3x better. Lands exactly on the ~2%
floor the settle sweep plateaued at, i.e. it removes the race component entirely.

**BUT on the real 27B it is a WASH** (interleaved, 3 reps each): old 2/4/2 = 8 misses, drain 3/1/3 = 7;
30.8/34.7/30.8 s vs 31.9/27.9/31.4 s. Why: the probe at ndom=2 switches domains EVERY rep, so its 11% is
race-dominated; the model switches far less often relative to work, so its 2-4 misses/run are dominated by
the RESIDUAL cause. **The reproducer is faithful to "a stall" but over-weights the race vs production** —
weight any future probe result against a model run before believing it transfers.

Kept ON anyway: strictly better than the sleep it replaces, free in the common path, single-domain never
reaches it (dom_activate returns early when dom == dom_active).

**Post-switch dummy: NO effect (A-B-A, n=4000).** drain=1 -> 106, drain=2 (pre+post) -> 104, drain=1 -> 78.
A dummy in the NEW domain after the switch changes nothing, so the residual is NOT a
first-submit-into-a-freshly-switched-domain hazard. Hypothesis dead.

## Kernel: the fence path exists but is COMPILED OUT

`CONFIG_ROCKCHIP_RKNPU_FENCE is not set` on the board kernel, yet `drivers/rknpu/rknpu_job.c` fully
implements it: `RKNPU_JOB_FENCE_OUT` (1<<4) -> `rknpu_fence_alloc` -> `rknpu_fence_get_fd` -> a sync_file fd
in `args->fence_fd`, signalled by the completion IRQ. `FENCE_IN` (1<<3) waits an incoming fence first.
`dma_fence` is a GENERIC Linux primitive (drivers/dma-buf), not an rknpu or NVDLA concept.

Why it matters beyond the domain barrier: `ork_dyn_end`'s job is whole-chain completion, which is exactly
what a fence reports — so the fence can replace the SENTINEL-TIMEOUT DRAIN entirely (no learned window, no
2 s catch-all, no O(no) `dc civac` poll on the hot path, CPU sleeps on a pollable fd). Sentinels stay only
for MID-CHAIN steering and for the error path (scan to find WHICH op stalled, only once a fence times out).
Fences are per-JOB (per submit); per-OP progress inside a HW chain still needs sentinels or a new ioctl.

**A stall does NOT signal the fence.** `dma_fence_signal(job->fence)` occurs at exactly one site, in the
`RKNPU_JOB_DONE` completion path; `rknpu_job_abort` / `rknpu_job_timeout_clean` tear the job down without
signalling, so a waiter would block for its own full timeout. Hence **patch 04** (below).

### Board kernel rebuild in progress (~/kbuild/linux-rockchip on the SBC)

armbian/linux-rockchip @ 1cd878b3ac49, patches 01+02 applied by `git apply`, **03 applied MANUALLY** (its
hunk header is a bare `@@` with no line numbers, so neither `git apply` nor `patch` can place it — see
`~/kbuild/p03.py`), plus:
- `CONFIG_ROCKCHIP_RKNPU_FENCE=y`
- `CONFIG_LOCALVERSION="-vendor-rk35xx-fence"` — a SEPARATE version so it gets its own /lib/modules tree
  and the running kernel is untouched (the running one was built with gcc 13.3.0, the board has 14.2, and
  `CONFIG_MODVERSIONS=y`, so a same-version rebuild could break module loading).
- **patch 04 (NEW, ours):** in `rknpu_job_free`, signal an unsignalled fence with `-ETIMEDOUT` before
  `dma_fence_put`, so a torn-down (stalled) job WAKES its waiter instead of making it burn a timeout.

Boot safety: `CONFIG_NVME_CORE/BLK_DEV_NVME/EXT4_FS=y` and `STMMAC_ETH/DWMAC_ROCKCHIP=y` are all BUILT-IN,
so the board boots and keeps SSH even if no module loads. Install = new `vmlinuz-*-fence` + flip the
`/boot/Image` symlink (DietPi boots via boot.scr + Image/dtb/initrd.img symlinks); revert = flip back.

## Better than a fence: the PC TASK COUNTER register (2026-08-25, from rknpu_job.c)

The rknpu interrupt is **per submit-BATCH, not per task**: `rknpu_job_done()` splits a job into
`ceil(task_number / max_submit_number)` batches, takes an IRQ after each and re-commits the next, and only
the LAST sets `RKNPU_JOB_DONE` + signals the fence. So per-op completion is never surfaced, and true
per-task IRQs would need `max_submit_number = 1` (serializes commits, one IRQ per op — likely a net loss).

**But the hardware already exposes exact per-op progress in a register.** The driver's own timeout logging
reads `REG_READ(config->pc_task_status_offset) & config->pc_task_number_mask` = the PC's completed-task
counter. That is precisely what the sentinel poll reconstructs from output DATA. Exposing it read-only to
userspace (ioctl, or a read-only mmap of the register page) would replace the whole sentinel mechanism with
ONE register read: no `dc civac`, no full-surface scans, no value-collision/stale-read failure modes, no
guessed timeout. Smaller than a fence redesign and strictly more informative. **Top candidate for custom
driver work.**

Caveat: this removes SENTINEL polling, not spinning. An IRQ/fence wakeup is ~5-20 us; decode-regime ops are
tens of us, so sleeping can lose to a tight spin — the original reason the doorbell busy-polls. Right shape
is hybrid: sleep until near the EWMA-predicted completion, then spin the tail on the task counter.

## The timeout UNITS BUG is LOAD-BEARING — do not fix it alone (A-B-A, n=4000, 2026-08-25)

`rknpu_job_timeout_clean` compares `ktime_us_delta(now, job->timestamp) >= args->timeout` — MICROseconds
against a value every other driver site treats as milliseconds (`msecs_to_jiffies`, `args->timeout * 1000`).
Our int4 submits pass 1500 (ms), so the real reap threshold is **1.5 ms**, not 1.5 s.

Tested from userspace with `ORK_I4_KTMO_MUL` (scales ONLY the kernel-facing value — the first attempt scaled
`orki_i4_submit_tmo_ms()` itself, which is ALSO a host sleep in `ork_dom_flush_if_dirty`, turning every dirty
boundary into a 100 s stall; that run had to be killed):

| MUL | kernel reap threshold | stalls/4000 |
|---|---|---|
| 1 | 1500 us (buggy) | 48 (1.20%) |
| 1000 | 1500000 us (intended) | **120 (3.00%)** |
| 1 | 1500 us | 46 (1.15%) |

**"Fixing" it makes stalls 2.6x WORSE** (~6 sigma). So the hypothesis that timeout_clean kills legitimate
in-flight jobs is REFUTED. What actually happens: the doorbell drains before the next submit, so at submit
time the prior job has landed its sentinels but is often NOT RETIRED — and the 1.5 ms threshold reaps it.
**The units bug is accidentally doing the drain barrier's job, for free, on every submit.** Restore the
intended value and that cleanup stops, unretired jobs linger, stalls rise.

Corroborates the retirement mechanism from a second, independent direction: the drain barrier and aggressive
reaping both help, and both work by retiring the prior job before the next submit.

**DO NOT correct the units bug on its own — only together with a no-forward-progress watchdog.**

Related design constraint (user, 2026-08-25): the architecture deliberately MINIMISES submits (keep work on
the doorbell; ideally the only resubmit is when the chain hits max task length and wraps). That breaks the
driver's cleanup model, which is SUBMIT-TRIGGERED — `timeout_clean` runs only when another submit arrives on
that core. Fewer submits => a wedged job sits unnoticed longer => the D-state teardown cascade. A timer-based
watchdog is the only cleanup that works for a one-long-chain design.

## Open hypotheses

1. **Task-boundary / HW-chaining race** (user). Being tested via `ORK_DYN_NOCHAIN=1` (see below).
2. **Marginal-H / stale-CBUF residue.** `OPS_REGISTRY.md` (`run_i4_bchain_db` row) documents a
   non-deterministic corruption fixed 2026-08-22 by LOWERING H, concluded "stale-CBUF-residue exposed by a
   marginal H", **MECHANISM UNKNOWN**. Same non-determinism, same dtype, same spine. Strong candidate.
3. **The known ~1/4000 dispatch drop.** `ork_dyn_begin_mc` is marked PROVEN with the caveat
   "~1/4000 dispatch-drop auto-recovered" — never root-caused, just papered over by the recover loop.

Prior art says those cases were "accepted but NEVER STARTED" (`submit_rc=0`, `hw_elapse=0`). Ours
**partially executes**, which is new information and may be a different (or better-observed) mode.

## Next steps

0. **Test the marginal-geometry prediction** — the top lead now. Either rebuild a pack at a lower
   `ORK_I4_KS` (even split), or build a faithful standalone reproducer first (below) and sweep K there.
1. **Read `hw_elapse` at miss time** (`ork_npu_dump_state`) — separates "never started" from "started and
   wedged" at the hardware, instead of inferring from the output. Cheapest decisive next measurement.
   NOTE `dma_rw`/`int_status` read 0-always on this kernel and are NOT diagnostic.
2. `ORK_DYN_NOCHAIN=1` A/B is INCONCLUSIVE and DANGEROUS — rep1 gave 0 misses/23.6 s but 0 occurs
   naturally in chained runs too (n=1 proves nothing), and rep2 WEDGED the NPU into an unrecoverable
   domain-switch-timeout cascade needing a reboot. Do not rerun in this form.
2b. **Build a faithful standalone reproducer.** `tools/i4_doorbell_probe 200 17408 5120` does NOT work:
   at that shape BOTH the doorbell `[B]` and the blocking reference `[A]` return "rc errs" because the
   probe's reference `run_chain_i4` is bounded at K<=4096. A reproducer must drive `ork_i4_mm_pack` +
   `ork_i4_mm_run` (M=1) so it takes the same Sk=2 sliced path the model does.
3. If chaining is exonerated, test hypothesis 2: sweep H down on the failing shapes and see if the rate
   falls smoothly the way the K=2560 bug did (7:~50% 6:5/8 5:4/8 4:0/8).

## Tree state

- `src/npu/i8/dyn.c` — **KEEP**: batched invalidate + `dsb ish` before the load in all four poll variants
  (genuine ARM correctness fix; not the cause). The `ORK_DYN_NOCHAIN` macro added here was REVERTED —
  int4 never reaches it (`dyn.c:268` returns early), which is Tier 19 on the roadmap.
- `src/npu/i8/dyn_ctl.c` — **DIAGNOSTIC, remove later**: learned miss window (2 s catch-all trimmed to
  `ORK_MISS_K` x EWMA once known; measured a wash vs fixed 300 ms) + the `[MC-RECOVER]` pattern/bsync dump.
- `src/npu/i4/chain.c` — **DIAGNOSTIC, remove later**: `orki_i4_submit_maybe_nochain` (`ORK_DYN_NOCHAIN=1`).
- `tools/size_budget.txt` — chain.c budget **temporarily** 935 → 955 for that helper. **Restore to 935**
  when the probe is deleted.
- Nothing committed. Wiki has Tier 19 committed locally (not pushed).

## Board ops

RK3588 `10.3.0.236`, kernel `6.1.115-vendor-rk35xx-sram` (carries kernel patch 03). Always
`sudo tools/util/npu_guard.sh --`. Re-pin DDR/CPU governors after ANY reboot (a parked DMC governor
roughly halves throughput and silently invalidates timings). Repro:

```sh
sudo ORK_NPU_LOCK_WAIT=600 tools/util/npu_guard.sh -- taskset -c 4-7 \
  env ORK_MC_DIAG=1 ORK_QUANT=4 ORK_MIXED_W4A4=1 timeout 900 \
  ./bin/ork_ppl ~/q27_rtn_v6.orkpack ~/llama.cpp/README.md 64 64 1
```
