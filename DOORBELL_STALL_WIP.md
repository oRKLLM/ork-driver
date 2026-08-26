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

## Correlated dmesg (ORK_KMSG_SUBMIT=1) + what it ruled out

`ORK_KMSG_SUBMIT=1` stamps every submit into /dev/kmsg (cached fd, one write) so userspace submits and the
kernel watchdog interleave in ONE stream:

```
ork: sub#2500 dom=0 cm=0x1 tasks=1   fl=0x1   <- drain-barrier dummies (blocking) in the OUTGOING domain
ork: sub#2503 dom=1 cm=0x1 tasks=832 fl=0x7   <- first real submits into the NEW domain
ork: sub#2504 dom=1 cm=0x2 tasks=864 fl=0x7
ork: sub#2505 dom=1 cm=0x4 tasks=864 fl=0x7
RKNPU: core 0 NO TASK PROGRESS ... tasks=0/832 done (counter 0x0), cm=0x1 dom=1, job age 115004 us
```

The watchdog log now prints **completed/submitted** so it is self-interpreting: `0/832` = never started.

**⚠ THE STAMP PERTURBS THE RATE** — 3.25% -> 6.00% with it on (one extra syscall per submit, ~3800/run).
Use it for LOCALISATION ONLY; never quote a rate from a stamped run.

**Ordering: the domain switch is BEFORE the failing submit** (drain dummies in dom=0, real submits in dom=1,
watchdog reports dom=1). **But that is NOT discriminating at ndom=2** — the probe alternates domains every
rep, so EVERY round is "first submit after a switch", failing or not. The discriminating evidence is the
factorial: `M=64, ndom=1` still fails 3/400, i.e. **failures occur with no domain switch at all**. The
switch raises the probability; it is not necessary.

### Also ruled out

- **Ping-pong** (A-B-A, n=4000): ON 146 / OFF 111 / ON 116 per 4000. The two ON arms bracket 146 and 116
  (~1.9 sigma apart on their own), OFF sits inside that spread — ~1.3 sigma vs the bracket mean. No effect.
  Without the bracket, 146->111 would have looked like a 24% win.
- **Kernel batch re-commit**: RK3588 `max_submit_number = 4095` and our submits carry 832-864 tasks, so the
  job is committed **once**, not in batches. `rknpu_job_done`'s re-commit path is not involved.
- **Counter wrap**: `pc_task_number_mask = 0xfff` (wraps at 4096) and 832 < 4095, so `0x0` cannot be a wrap
  artifact; healthy jobs advance the same register.

### LANDMINE (latent, not our bug — do not create one)

`rknpu_job_subcore_commit` reads `args->subcore_task[core_index + 2]` when `use_core_num == 3`, i.e. indices
**2,3,4** — while our code fills only [0],[1],[2]. An all-core submit (`core_mask = 0x7`) would therefore
commit `task_number = 0` on cores 1 and 2 and the PC would silently never run — exactly the symptom we spent
this session chasing. Every submit in the tree currently uses a SINGLE-core mask (`1u<<i` /
`RKNPU_CORE0_MASK`), so `use_core_num == 1` and the safe `case 1:` path is taken. **If you ever add a
core_mask=0x7 submit, fill `subcore_task[2..4]`.**

## SHAPE SWEEP (2026-08-25, n=4000/cell) — geometry does NOT matter, magnitude does

**Per-row path (i4batch=0, M=1, ndom=2) — the model's DOMINANT failure mode:**

| K | K-slices | stalls/4000 | rate | median |
|---|---|---|---|---|
| 10752 | **1** (no split) | **6** | 0.15% | 3.76 ms |
| 17408 | 2 (uneven 10752+6656) | 15 | 0.38% | 5.73 ms |
| 17408 (repeat) | 2 | 21 | 0.53% | 5.71 ms |
| 21504 | 2 (**even** 2x10752) | 25 | 0.62% | 7.11 ms |

**BCHAIN path (i4batch=1, M=64):** 17408 -> 85 (2.12%), 21504 -> 96 (2.40%), 10752 -> 76 (1.90%).

**CLOSED: the "marginal geometry / uneven slice remainder" hypothesis.** It was the top lead (the registry's
K=2560 H-table precedent supported it) and it is WRONG on BOTH paths: the EVEN split (21504) fails at least
as often as the uneven one (17408). Do not revisit without new evidence.

**Split COUNT matters** on the per-row path: Sk=1 gives 6/4000 vs 15-25 for Sk=2 (~3 sigma).

**But it is NOT a per-program lottery.** "Each program independently risks failure" fits the per-row points
(1 prog 0.15%, 2 prog ~0.3-0.6%) and then breaks completely: BCHAIN submits ~850 programs, which that model
predicts should fail ~72% of the time versus the 2-3% measured. Duration correlates better (3.76/5.73/7.11
ms -> 0.15/0.45/0.62%) but superlinearly, and three points do not justify fitting a curve.

**So: the failing shapes are not special — they are just the BIGGEST.** The model's ffn_down is the largest,
longest int4 op in the graph, which explains 12/12 misses landing there with no special geometric property.

## ⚠ THE KERNEL PROGRESS WATCHDOG (patch 05) DESTABILISES THE BOARD — keep it OFF by default

Three incidents attributable to it: v1 hard-hung (unguarded MMIO, fixed), and v3 COLD-RESET the board twice
(instant SoC reset, no panic, no console, ramoops cleared — DRAM gone, so it is a cold reset not a Linux
reboot). Discriminator, same cell (K=17408 M=1 ndom=2 n=4000):

| NPU watchdog | outcome |
|---|---|
| ON (`wd_period_us=1000`) | board cold-reset mid-run |
| **OFF (`wd_period_us=0`)** | completed, 21/4000, same boot_id, board healthy |

An earlier PMIC/thermal hypothesis for these resets is REFUTED by that discriminator (idle temps 28-30 C,
trips at 75/80/85 C; and the reset follows the watchdog, not the workload). **Set `wd_period_us=0` for any
routine run;** enable it only for deliberate diagnosis, accepting a possible reset.

## Log capture — what actually persists (learned after losing 3 wedges' evidence)

- `/var/log` is **DietPi RAMlog** (`AUTO_SETUP_LOGGING_INDEX=-1`, hourly clear) — nothing survives a reboot.
- journald says `Storage=persistent` but its files are TRUNCATED and `journalctl -k` returns "No entries".
- ramoops/pstore IS configured (dmesg-0/1, console, pmsg regions) and would hold a **panic**, but our
  failure is a COLD reset that clears DRAM, so pstore comes back empty every time.
- **netconsole WORKS and is now persistent** (`/etc/modules-load.d/netconsole.conf` +
  `/etc/modprobe.d/netconsole.conf`, target `6666@10.3.0.238` via the GATEWAY mac). Receiver: a UDP listener
  on the Mac (`/tmp/netcon_listen.py` -> `/tmp/netcon.log`).
- **TRAP that cost an hour:** default `console_loglevel = 4`, so only messages BELOW level 4 reach the
  console and hence netconsole. `<4>`/KERN_WARNING test messages were silently dropped and I wrongly
  concluded netconsole was broken; a `<3>`/KERN_ERR message arrived instantly. `kernel.printk = 8 4 1 7` is
  now persisted in `/etc/sysctl.d/`. **Absence of low-priority lines in an old capture is NOT evidence the
  event did not happen** — e.g. `RKNPU: soft reset` is LOG_INFO (level 6) and was invisible at loglevel 4.
- `RKNPU: soft reset, num: 6` — `num` is `num_srsts`, the COUNT OF DT RESET LINES (constant 6 on RK3588),
  not a reset counter. Count log lines, not that field.

## EXPERIMENT 1 (2026-08-26): the job IS committed; the hardware never starts

Kernel counters (patch 07) over one 4000-rep run, K=17408 M=1 ndom=2:

```
commit=16075   irq=16065   done=16065   nojob=0   unpow=0     stall-recover events: 10
```

**commit - done = 10 = exactly the stall count.** Each stall is one job COMMITTED to the PC that never
completed.

- **irq == done** -> every interrupt that reached the handler produced a completion. **NO interrupts lost.**
  This REFUTES the "lost completion IRQ leaves subcore_data->job set, next submit sits on todo_list" theory
  — the job was not queued-and-forgotten, it was committed.
- **nojob = 0** -> no spurious interrupts.
- **unpow = 0** -> the Change-6 power guard never fired here, so patch 06 is NOT converting crashes into
  lost completions (a caveat raised when it landed; now cleared for this workload).

**The window is now: `rknpu_job_subcore_commit()` writes the PC registers -> hardware never begins -> no
interrupt.** Consistent with the PC completed-task counter reading 0x0.

**NEXT: read the PC registers BACK immediately after the commit write and compare healthy vs stalled.**
- readback wrong (OPERATION_ENABLE / task number reads 0) => the write did not land: visibility/ordering,
  missing barrier, or clocks gated at commit time. Driver-fixable.
- readback correct but nothing runs => genuine hardware condition; chase the PC state machine (the mainline
  `accel/rocket` driver names these registers — see AGENTS.md).

## EXPERIMENT 2 (2026-08-26): the COMMIT PATH IS EXONERATED — registers are byte-identical

Patch 08 snapshots what the driver WROTE to the PC registers at commit and what reads BACK, reported by
the watchdog on a stall, with a MATCHED healthy control (same `task_ctrl=0x7002` signature).

```
HEALTHY  data_addr=0xffd2f000 amount=0x3b int_mask=0x300 task_ctrl=0x1002 status(pre)=0x5000 status(post)=0x5000
STALLED  data_addr=0xffd2f000 amount=0x3b int_mask=0x300 task_ctrl=0x1002 status(pre)=0x5000 status(post)=0x5000
```

**Every field identical**, including the same `data_addr` values appearing in both groups. The driver
computes the right values, they land, and the PC state after the `PC_OP_EN` pulse is indistinguishable
between a job that runs and one that does not.

**Two false leads killed here, both from missing/наmismatched controls — the recurring failure mode:**
1. `task_ctrl` wrote `0x7002` / read `0x1002` looked wrong. It is not: the register drops the `0x6` and
   keeps bit 0 (pp_en) + task count. Healthy jobs behave identically.
2. `status(pre)` `0x7000` (healthy) vs `0x5000` (stalled) looked like a real pre-commit state difference.
   It was the UNMATCHED control: 1-task/pp-off jobs read `0x7000`, 2-task/pp-on jobs read `0x5000`. With
   matched jobs both read `0x5000`.
   *(And the first matched attempt logged ZERO healthy samples — the workload repeats a 4-job cycle, one
   real `0x7002` op plus three `0x6001` drain dummies, and sampling every 500th completion aliased because
   500 %% 4 == 0. Never sample a periodic workload on a modulo; log the first N matches instead.)*

## Where that leaves it — the software chain is fully exonerated

userspace OK -> ioctl OK -> queue OK (exp 1: commit-done deficit == stall count, irq == done) ->
commit OK -> registers OK (exp 2) -> **DIVERGENCE** -> no execution, no interrupt, task counter 0x0.

**Leading hypothesis: the PC's DESCRIPTOR FETCH FROM DRAM, through the IOMMU.** Registers only carry
POINTERS (`data_addr` -> regcmd, `task_base_addr` -> the task array); identical pointers can still fetch
stale or unmapped data. This fits every measurement:
- registers identical (the pointer is right, the DATA or its MAPPING is not)
- **domain switching triples the rate** — a switch changes page tables; a stale IOMMU TLB entry would make
  the fetch fail silently
- all three cores fail together — they share the iommu domain
- retirement/aggressive reaping help — more time for writes/mappings to settle
- scales with op magnitude, not geometry — more descriptor/regcmd bytes to become visible

**NEXT:** correlate `rk_iommu` faults with stalls (the IOMMU driver already logs page faults), and try an
explicit IOMMU TLB/domain flush after `dom_activate` to see whether the rate moves.

## EXPERIMENTS 3-5 (2026-08-26): IOMMU, task array, and cacheability

**3. IOMMU faults do NOT correlate — there are none.** 14 stalls in one run; `Page fault` 0, `BUS_ERROR` 0,
`not attached to domain` 0, `switch iommu domain time out` 0. So the descriptor fetch is NOT hitting an
unmapped/stale IOVA — a bad mapping would fault. **Refutes the stale-TLB form of the hypothesis.**

**4. The task array is NOT stale.** Userspace now stamps a monotonic sequence into `rknpu_task.op_idx`
(a field the uABI declares and the driver NEVER reads); the kernel logs what it sees at commit. Stalled
commits report `seq = 737, 1045, 1325, 2731, 6397` — all distinct, increasing, tracking run progress over
~8000 descriptors. A stale kernel-mapping read would show repeated or old stamps. **The kernel reads
exactly what userspace just wrote.** (Caveat: the HEALTHY log line did not get the seq field — the format
patch applied to only one of the two sites — but the stalled values are conclusive on their own.)

**5. "Just make the buffers uncached" is NOT a viable test.** `ORK_RC_UNCACHED=1` (drops
`RKNPU_MEM_CACHEABLE`) makes the probe die with **SIGBUS**: the kernel then maps the buffer as
Device memory, and on ARM64 UNALIGNED ACCESS TO DEVICE MEMORY FAULTS — our `memcpy`/NEON writes into the
regcmd are legal on Normal cacheable memory and illegal on Device. Knob kept for reference; do not expect
it to run.

### Where the elimination stands

| stage | verdict |
|---|---|
| userspace build/submit | OK |
| ioctl + queue | OK (exp 1: `commit-done` == stall count, `irq == done`) |
| commit register writes | OK (exp 2: byte-identical healthy vs stalled) |
| IOMMU mapping | OK (exp 3: zero faults) |
| task descriptor freshness | OK (exp 4: fresh monotonic stamps) |
| **regcmd DATA as the NPU DMA sees it** | **UNTESTED — the remaining candidate** |

**NEXT:** the regcmd buffer (`mrc`, flags `0x403`) is CACHEABLE DRAM and has NO kernel mapping, so nothing
has ever verified the bytes the NPU actually fetches. Add `RKNPU_MEM_KERNEL_MAPPING` (0x8) to that
allocation so the kernel can read it at commit, and log a checksum of the first N words for healthy vs
stalled jobs. If a stalled commit's regcmd reads as stale/zero, that is the cause; if it is correct, the
whole software path is exonerated and the fault is in the PC's execution start itself.

## EXPERIMENTS 6-7 (2026-08-26): regcmd data is fine; timeout_clean is the JANITOR, not the killer

**6. The regcmd bytes DO reach DRAM.** `ORK_RC_VERIFY=1` checksums the buffer from cache, invalidates with
a FROM_DEVICE sync, and re-checksums from DRAM. **0 mismatches over 4000 reps / 5 stalls.**
POSITIVE CONTROL (`ORK_RC_VERIFY=2`, corrupts a word after the flush) DOES report
`REGCMD MISMATCH`, so the detector is proven capable of firing. The last untested software link is clean.

**7. `rknpu_job_timeout_clean` does NOT cause the stall.** Source reading suggested a strong mechanism: it
soft-resets the NPU and drops the running job whenever `ktime_us_delta(now, job->timestamp) >=
args->timeout` — MICROseconds vs a millisecond value, so an effective 1.5 ms threshold against jobs that
run a median 5.7 ms. That predicts exactly the `commit - done` deficit we measure.

Counter (patch 13) says: `stalls=14, commit-done=14, treap=14` — an exact match that LOOKS like
confirmation. **But `last_kill_age_us = 157689`.** The reaped job was **157 ms** old, not 1.5-5.7 ms. That
is the recovery timeline (stall -> watchdog reports ~115 ms -> userspace ACT_RESET + resubmit -> that
resubmit's timeout_clean reaps the long-dead job). One reap per stall, hence the exact match.
**timeout_clean is the JANITOR, not the killer** — and this would have been believed as a root cause if the
KILL AGE had not been logged alongside the count. Corroborated independently: the watchdog reports the task
counter as `0x0`, i.e. the job never started, so it was never killed mid-execution.

### Elimination table (updated)

| stage | verdict |
|---|---|
| userspace build/submit | OK |
| ioctl + queue | OK (exp 1) |
| commit register writes | OK (exp 2, matched control) |
| IOMMU mapping | OK (exp 3, zero faults) |
| task descriptor freshness | OK (exp 4, fresh monotonic stamps) |
| regcmd bytes in DRAM | OK (exp 6, with positive control) |
| timeout_clean killing live jobs | REFUTED (exp 7, kill age 157 ms) |
| **PC start after a correct commit** | **the remaining gap** |

Every software stage is now exonerated with a control. The job is committed with byte-identical registers,
valid mappings, fresh descriptors and correct regcmd bytes in DRAM — and the PC does not begin. Still open
whether that is a driver-visible condition or silicon.

**Method note for whoever continues:** three separate false root causes this session came from a count
matching without a corroborating measurement (lost IRQs, marginal K geometry, timeout_clean). Always log a
second, independent quantity (an age, a sequence, a checksum) next to any count that "matches".

## EXPERIMENTS 8-9 (2026-08-26): the PC block is identical, and a re-pulse does NOT rescue

**8. Full PC/INT register block, matched healthy vs stalled — BYTE-IDENTICAL.**

```
HEALTHY  00000000 .. ffd2f000 0000003b 0000000f 00000000 00000300 .. 00001002 .. 00011000 | en=00000000
STALLED  00000000 .. ffd2f000 0000003b 0000000f 00000000 00000300 .. 00001002 .. 00011000 | en=00000000
```
(map: 0x00 VERSION, 0x08 PC_OP_EN, 0x10 DATA_ADDR, 0x14 AMOUNT, 0x20 INT_MASK, 0x28 INT_STATUS,
0x2c INT_RAW, 0x30 TASK_CTRL, 0x3c TASK_STATUS, 0xf008 ENABLE_MASK.)

`ENABLE_MASK = 0` and `INT_STATUS = 0` are NORMAL — they read the same on jobs that run. Do not chase
either. **The fault is not visible anywhere in the documented PC/INT register set.**

**9. START-CONFIRM-AND-RETRY (`wd_kick=1`) does NOT rescue the job.** On a detected stall the watchdog
re-pulses `PC_OP_EN` for that core (registers untouched and already verified correct). Result: **4 kicks,
4 stalls — none recovered**; `commit - done` deficit unchanged. **So it is NOT a missed start trigger.**
That was the most hopeful cheap fix and it is refuted. Only `ACT_RESET` + resubmit recovers.

### Conclusion of the software investigation

Everything the driver does is correct and verified with controls: queue, commit, registers, IOMMU mapping,
descriptor freshness, regcmd bytes in DRAM, and the whole PC register block. The job does not start, raises
no interrupt (INT_RAW = 0, so the hardware is not even signalling internally), and cannot be restarted by
re-triggering — only a reset clears it. Remaining possibilities:
1. a hung SUB-BLOCK (CNA / DPU / PPU have their own register windows, never dumped), or
2. genuine silicon behaviour with no software-visible cause.

### Therefore the practical target is CHEAP RECOVERY, not prevention

Only a reset recovers, so make the reset cheap. `rknpu_soft_reset` asserts ALL SIX DT reset lines
device-wide (`num: 6` = `num_srsts`, the count of `resets` phandles — NOT a counter), destroying warm state
on all three cores; that plus the 614 ms detection window plus the resubmit is the ~3.6 s per miss.
**If those 6 lines are per-core, asserting only the stalled core's reset would collapse the recovery cost
without needing the root cause.** Check the DT `reset-names` for the rknpu node.

Measured levers already in hand (all condition-based, cause-agnostic):
- retire before switching domains (drain barrier): 11.47% -> 1.99%
- bound programs per submit: Sk=1 0.15%, Sk=2 0.38-0.62%, BCHAIN ~850 progs 2-3%
- minimise domain switches: ndom=2 roughly triples the rate
- geometry (even/uneven K split) is irrelevant — do not constrain alignment

## EXPERIMENT 10 (2026-08-26): per-core reset + re-commit ALSO fails to rescue

Implemented `rknpu_soft_reset_core()` (asserts only `srst_a{N}` + `srst_h{N}`, no `msleep(100)`, no global
`soft_reseting` flag) and `rknpu_job_kick_stalled()` (per-core reset then re-commit the same job), wired to
the watchdog as `wd_kick=2`.

| arm | stalls/4000 | mean | max | commit-done |
|---|---|---|---|---|
| kick off | 13 | 26.03 ms | 52633 ms | 13 (= 1/stall) |
| kick=2 | 7 | 9.91 ms | 2125 ms | **14 (= 2/stall)** |

Mechanically fine — 7 kicks, all returning 0. **But it does NOT rescue the job:** arm B's deficit is TWO
uncompleted commits per stall (the original AND the re-commit). A successful rescue would give one.

Do not be fooled by the mean/max: arm A contains a single 52.6 s outlier (recovery exhausted) which alone
is half its total time; without it A ~51 s vs B ~40 s. And 13 vs 7 stalls is ~1.4 sigma. The kick's apparent
win is an outlier artifact.

**Finding: the stuck condition SURVIVES a per-core AXI+AHB reset.** So it is not confined to that core's bus
interfaces. Only the full device-wide `rknpu_soft_reset` recovers — and that also re-attaches the IOMMU
domain afterwards, which the per-core path deliberately skips. **Next cheap test: per-core reset PLUS the
IOMMU domain re-attach**, to find out which half of the device-wide sequence is actually doing the work.

Kept in the tree: `rknpu_soft_reset_core()` is correct and useful regardless (no msleep, no global stall of
`rknpu_job_next`); it just is not sufficient on its own.

## ★ EXPERIMENT 11 (2026-08-26): the IOMMU RE-ATTACH is the essential recovery ingredient

Recovery-ingredient factorial, n=4000/cell, `deficit = commit - done` over the cell:

| mode | ingredients | stalls | deficit/stall |
|---|---|---|---|
| 0 | baseline, no kick | 9 | 1.0 |
| 2 | per-core reset + re-commit | 4 | **2.0** |
| 4 | pre-pulse + per-core reset + re-commit | 6 | **2.0** |
| 3 | per-core reset + **IOMMU re-attach** + re-commit | 9 | **1.0** |
| 5 | pre-pulse + per-core reset + **IOMMU** + re-commit | 10 | **1.0** |

**Read deficit/stall, not the stall count** (counts are noise at this n; the ratio is per-stall). Baseline
1.0 = the original commit is lost. **Without the IOMMU re-attach: 2.0 — the re-commit fails too. With it:
1.0 — the re-commit SUCCEEDS.**

**Why:** `iommu_detach_device()` immediately followed by `iommu_attach_device()` of the SAME domain is not
unbind/rebind, it is a RE-PROGRAMMING idiom — it makes the IOMMU driver rewrite the device translation-table
registers. The NPU's `rk_iommu` shares the NPU's reset/power domain, so **resetting the core wipes its MMU
programming**; a re-commit without restoring it has a valid-looking IOVA in `PC_DATA_ADDR` and no
translation behind it. That is why mode 2 failed, and it was predictable before running it.

**The pre-pulse is irrelevant** (2 vs 4, 3 vs 5 identical), confirming mode 1: re-triggering `PC_OP_EN`
does nothing.

### Consequences

- Recovery does NOT need the device-wide six-line reset or its `msleep(100)`. It needs
  **per-core reset + IOMMU re-attach + re-commit**.
- **But the IOMMU re-attach CANNOT be async.** Bare from the watchdog it raced
  `rknpu_iommu_dma_map_sg` and PANICKED the board (`__iommu_attach_group` -> NULL deref at 0x40 -> fatal
  oops). It is safe only with `reset_lock` held and `soft_reseting = true` (the global dispatch quiesce),
  which is what `rknpu_soft_reset` does. So recovery is cheaper, not free.
- **STILL OPEN:** userspace continued to record these as >400 ms events even in modes 3/5, so the
  end-to-end latency has NOT collapsed yet despite the re-commit completing. Measure where that time goes
  (watchdog detect latency ~115 ms + quiesce + re-attach + re-commit) before claiming the win.

## ⚠ RETRACTION + BUG: experiment 11's "IOMMU re-attach is essential" is UNSUPPORTED

**Retraction.** The kick was returning **-EBUSY on every stall**: `mutex_trylock(&reset_lock)` failed
(a device-wide `rknpu_soft_reset` holds it across its `msleep(100)` and one is almost always in flight),
and my code **returned before the re-commit**. So mode 3's "deficit 1.0/stall" did NOT mean the re-commit
succeeded — it meant **no re-commit was attempted**. Mode 2's 2.0 meant it was attempted and failed. The
metric conflated "never tried" with "tried and worked", and the good number was read as success.
**The claim that the IOMMU re-attach is the essential ingredient is unsupported.** (Fourth time this
session a matching number misled: log the RETURN VALUE next to the count.)

**A real bug I introduced, still unfixed: use-after-free in `rknpu_job_kick_stalled`.** It reads
`subcore_data->job` under `irq_lock`, RELEASES the lock, then resets the core and re-commits that pointer.
In between, the job can complete and be freed by `cleanup_work`. Symptom: repeated panics in code with
nothing to do with the NPU —
```
pc : cfs_rq_clock_pelt+0x14/0x54     (CFS scheduler)
pc : aa_put_label+0x18/0x6c          (AppArmor)
```
classic slab corruption. Mode 3 is worst because the IOMMU re-attach widens the window enormously.
**Fix before any further kick testing:** hold `irq_lock` across the sequence, or take a reference on the
job. `wd_kick` defaults to 0, so the board is safe as it stands.

**A pre-existing driver wart my patch exposed (FIXED):** `rknpu_soft_reset` treated a failed `trylock` as
SUCCESS (`return 0`) without resetting anything. Harmless while it was the only resetter; once the per-core
path also took `reset_lock`, userspace's `RKNPU_ACT_RESET` silently became a no-op and the workload ran
~10x slower (2.7 vs 34 commits/s). Now it waits for the in-flight reset instead. **That ~10x was my bug,
not the inherent cost of the quiesce** — an earlier claim to the contrary was wrong.

## ★ RECOVERY LATENCY MEASURED — detection is ~100% of it

```
detect=115016us  reset=21us  iommu=0us  recommit=7us  total=28us
detect=114015us  reset=21us  iommu=0us  recommit=7us  total=28us
```

**The recovery ACTION costs 28 microseconds.** Per-core reset 21 us, re-commit 7 us. Detection costs
**115 ms** — over 99.9% of the latency, and it is a pure threshold (`wd_samples=64` at a nominal 1 kHz;
the effective sample period is ~1.8 ms because `queue_work` throttles it). Against a 5.7 ms op, 8-16
samples would detect in ~15-30 ms with ample margin.

So the achievable ceiling is roughly **3.6 s -> ~20 ms per miss**, and none of it requires knowing the root
cause — PROVIDED a working recovery action is found (mode 2 is confirmed not to be one).

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
## Session 2026-08-26 — doorbell stall: IOMMU theories refuted, board-killer found

### Board-stability bugs found and fixed (these were blocking every experiment)

1. **`ork_npu_reap_stuck` use-after-free (OUR code, `src/npu/core/device.c`).** It fired
   NONBLOCK dummy jobs, slept 3 ms, then `orki_bdestroy()`d the A/B/C buffers those jobs still
   referenced. Not a rare race: the function is only called when a core is already stuck, i.e.
   exactly when the dummy cannot complete in 3 ms. Killed the board from three different sites
   (`rb_erase`/`drm_vma_node_revoke`, `sg_free_table`/`__free_pages`, and a later `fork` via slab
   corruption) — always seconds AFTER it returned, which is why it read as a kernel bug.
   FIX: one BLOCKING submit per core after the dummies; it cannot return until that core has
   retired everything queued ahead of it. Positive drain, not a timing guess.

2. **Kernel: NULL iommu domain dereferenced (patch 33/34, `rknpu_iommu.c`).**
   `rknpu_iommu_dma_map_sg` / `_unmap_sg` / `_alloc_iova` all do `domain->iova_cookie` with no
   NULL check. The switch failure path does `detach(src)` then, if both attaches fail, returns
   with the device attached to NOTHING — so the next GEM op panics at offset 0x40.
   Reachable in practice: a stalled job cannot be drained -> `switch iommu domain time out`.
   FIX: `rknpu_iommu_cookie()` helper returns NULL instead of faulting; plus the switch's
   failure path now attaches any still-valid domain rather than leaving the device detached.
   STRONG CANDIDATE for the long-standing "NPU wedge needs a power cycle".

3. **Kernel: watchdog cadence (no rebuild needed).** `wd_period_us=1000` costs ~100x throughput
   (2.8 commits/s vs 283-490/s warm). Detection is `wd_samples * wd_period_us`, so
   `period=10000, samples=12` keeps the same ~120 ms detection at 1/10th the sampling.

### Stall root cause — what is now RULED OUT (all with counters/controls, not readings)

| hypothesis | test | result |
|---|---|---|
| deferred attach leaves stale page table | `rk_dbg_attach_deferred` (patch 29) | REFUTED: 0 deferred vs 1017 programmed during the run |
| domain switch races an in-flight job | `rknpu_dbg_switch_inflight` (patch 30) | REFUTED: 0 in-flight vs 1012 idle |
| the MMU itself is wedged | read RK_MMU_STATUS at stall (patch 31) | REFUTED: all 4 banks `0x19` = paging, **idle**, replay-empty |
| recovery needs post-reset settle time | kick_mode 6, settle 2/10/40 ms (patch 32) | REFUTED: 0 recovered out of 20 stalls |

An **idle** MMU at stall time is the strongest single clue: the NPU core never issued a memory
request at all. The failure is upstream of the IOMMU entirely.

### Recovery: what works

- `KICKED JOB COMPLETED` ~4.47 ms after the kick, extremely reproducible (4466/4471/4473/4478 us).
- Only the IOMMU detach/attach variant recovers: **3/3 with it, 0/11 without**.
- And it is NOT the elapsed time (settle refuted above) — so something in
  `rk_iommu_enable()` (force-reset of all MMU banks / ZAP_CACHE / paging cycle / clk_bulk) is
  the active ingredient. Which one is the open question.
- Latency: detect ~115 ms (tunable) + reset 21 us + iommu 4.6-42 ms + recommit 7 us.

### Open / next

- **UNRESOLVED CONFOUND:** kick_mode 3 drew 0 stalls in 2500 reps twice, while mode 6 drew 8-10
  in the same script. A kick cannot prevent a stall, so this is probably an ordering artifact —
  an interleaved A-B-A-B (`/tmp/p35.sh`) was launched to settle it and has not yet completed.
- Narrow which part of `rk_iommu_enable()` is curative (force-reset vs ZAP vs paging cycle vs
  clock) — that is the cheap-recovery lever.
- Residual domain-0 asymmetry: extra domains get `iommu_get_dma_cookie()` + a hand-patched
  `domain->type |= __IOMMU_DOMAIN_DMA_API`, skipping `iova_reserve_iommu_regions()` that the
  real `iommu_dma_init_domain()` path runs. Roadmap item, not chased.
- Nothing pushed. Kernel patches 07-34 not yet exported to wiki attachments (only 04/05/06 are).
- `src/npu/core/device.c` is in ATTEST_SRCS -> the reap-stuck fix needs a board `make test` and a
  refreshed `tests/sbc_attest.txt` in the same commit.

### LATE-SESSION CORRECTION — the 3/3 recovery result is RETRACTED

Patch 35 made `rknpu_job_abort()` clear `subcore_data->job` unconditionally (it previously left a
freed pointer published whenever `job->irq_entry[i]` was set). With that fixed, the kick now finds
NULL and declines, and mode 3 recovers **0/5** instead of 3/3.

So the earlier "IOMMU re-attach recovers 3/3" almost certainly measured a **use-after-free**: the
kick was re-committing a job `abort` had already freed. It completed (`KICKED JOB COMPLETED` ~4.47 ms,
suspiciously identical every time) because the memory was still mapped. Do not build on that result.
The IOMMU-vs-no-IOMMU 3/3-vs-0/11 split is therefore also unsafe to cite.

### The corruption chain, fully traced

    stall -> job stuck -> rknpu_iommu_domain_get_and_switch() waits 6 s -> "switch iommu domain time out"
      -> rknpu_gem_object_destroy() cannot unmap the IOVA -> "RKNPU: failed to destroy memory"
      -> it releases the GEM object ANYWAY
      -> shmem inode evicted with pages still mapped:
             BUG: Bad page cache ... "still mapped when deleted"
             (filemap_unaccount_folio / shmem_undo_range / shmem_evict_inode)
      -> allocator corrupt from here on; the box panics later somewhere unrelated
         (cfs_rq_clock_pelt, __memcg_kmem_charge_page via fork, sg_free_table/__free_pages)

That last hop is the remaining unfixed bug and the highest-value next fix: **when the IOMMU unmap
fails, `rknpu_gem_object_destroy` must NOT free the object.** Deliberately leaking the buffer is
strictly better than corrupting the page allocator.

Also still unfixed: my kick's mode-3 detach/attach can leave the device detached (patch 36 checks
the return and falls back, but 83 switch timeouts still appeared afterwards, so it is not fully
contained). Given the retraction above, the kick has no demonstrated benefit and should be left
DISARMED (`wd_kick=0`, `wd_period_us=0` -- both are the defaults) until the GEM-destroy fix lands.

### Board / kernel state at session end
- Running kernel #12 (patches 29-36), watchdog disarmed, no probe running.
- Kernel source of truth: `board:~/kbuild/linux-rockchip` (NOT /tmp -- tmpfs, wiped on reboot).
- VM builds: `tools/util/vm_kbuild.sh --keep` = ~30 s sync+build+install (SBC incremental was 2-5 min);
  `--stop` shuts it down. It now syncs the WHOLE driver plus `drivers/iommu/rockchip-iommu.c`.

### CORRECTION to "the corruption chain" above — the last hop is WRONG

Read the code instead of inferring it:

- `rknpu_gem_object_destroy()` on 3 failed switches **returns early** — it does NOT free. It skips
  `rknpu_gem_free_buf()`, `rknpu_gem_release()` and `rknpu_iommu_domain_put()`. So it already
  leaks-rather-than-corrupts, and the proposed "don't free when the unmap fails" fix is a NO-OP.
- Skipping `rknpu_iommu_domain_put()` there is also CORRECT, not a leak:
  `rknpu_iommu_domain_get_and_switch()` `atomic_inc`s the refcount only on its two success paths
  (`domain_id == current` and `refcount==0 -> switch ok`); the timeout and switch-failure paths
  return without taking a reference. The parallel I drew to kernel change 02 does not hold.
- `rknpu_gem_free_object()` is DRM's `->free` callback and just calls `..._object_destroy()`, so an
  early return there leaks the `drm_gem_object` — it does not release it.

What the corruption report actually says: `mapcount:1` + "still mapped when deleted" from
`shmem_evict_inode` means a **CPU page-table mapping** (a userspace VMA) still referenced the page
when the shmem inode was truncated. That is an mmap/VMA accounting problem, NOT the IOMMU unmap.
The precise defect is UNIDENTIFIED. Next instrumentation should target the GEM mmap/vm_close path
(`rknpu_gem_mmap` / `drm_gem_vm_close` / the probe's own munmap-vs-destroy ordering), not the IOMMU.

Still true and unaffected: the reap-stuck UAF fix, the NULL-domain guards, the abort dangling-pointer
fix, the watchdog cadence finding, the four refuted stall hypotheses, and the retraction of the
3/3 recovery result.

### PAUSED 2026-08-26 — board handed to another agent

Board access stopped mid-investigation at the user's request. Resume point below.

**Board state left behind:** kernel #12 `6.1.115-vendor-rk35xx-fence` (changes 4-9), watchdog DISARMED
(`wd_kick=0`, `wd_period_us=0`), no probe, no orkd. Changes 7/8 are ACTIVE and behaviour-changing, so any
measurement taken by another agent on this board is not comparable with pre-#12 numbers — record `uname -v`.

**Where the "still mapped when deleted" hunt got to.** Chasing why a shmem page has `mapcount:1` when the
inode is evicted:

- `orki_bdestroy()` (`src/npu.c:268`) is CORRECT: `munmap()` then `MEM_DESTROY`.
- **`ork_sig_teardown()` (`src/npu.c:261`) is NOT**: the SIGTERM handler issues `MEM_DESTROY` on every live
  buffer **without munmapping first**. This session SIGTERMed the probe repeatedly, so it ran often.
  Whether that is actually illegal is unresolved — the driver uses stock `drm_gem_mmap` with
  `drm_gem_vm_open`/`drm_gem_vm_close` as `vm_ops`, which DO refcount, so the object should survive until
  the VMA closes. So this is SUSPICIOUS BUT NOT YET A PROVEN BUG.
- **Candidate, UNVERIFIED:** `rknpu_gem_mmap_obj()` has an `err_close_vm:` path that calls
  `drm_gem_vm_close(vma)` explicitly when `rknpu_gem_mmap_buffer()` fails, then returns the error. If the
  VFS mmap-failure path also invokes `vm_ops->close`, that is a double reference drop -> premature free ->
  exactly "still mapped when deleted". NOT confirmed: it needs (a) checking whether `mmap_region()` calls
  `->close` on the `call_mmap()` failure path for this kernel, and (b) confirming `rknpu_gem_mmap_buffer()`
  ever actually fails in our workload — it may well be an unreachable path here, in which case this is a
  dead end and the real cause is elsewhere.

**Next steps when the board is free** (cheapest first):
1. Desk work, no board: read `mm/mmap.c` `mmap_region()` in this tree and settle whether `->close` runs on
   the `call_mmap()` error path. Kills or promotes the candidate above for free.
2. Add a counter to `rknpu_gem_mmap_buffer()`'s failure path. If it never fires under the probe, the
   candidate is dead.
3. Only then instrument VMA open/close vs GEM free ordering.

### DESK WORK 2026-08-26 (no board) — candidate killed, stronger one found

**1. The `err_close_vm` double-put candidate is DEAD.** Read `mmap_region()` in this tree
(`mm/mmap.c`, 6.1.115):

```c
error = call_mmap(file, vma);
if (error)
        goto unmap_and_free_vma;     /* jumps PAST the close label */
close_and_free_vma:
        if (vma->vm_ops && vma->vm_ops->close)
                vma->vm_ops->close(vma);
unmap_and_free_vma:                  /* <- lands here */
        fput(vma->vm_file);
```

`vm_ops->close` is NOT called on the `call_mmap()` failure path, so rknpu's explicit
`drm_gem_vm_close(vma)` in `err_close_vm` is CORRECT and NECESSARY (it drops the reference
`drm_gem_mmap()` took). Standard DRM idiom. Not a double drop.

**2. Mechanism of "still mapped when deleted", established.** rknpu inserts the GEM's shmem pages
into the VMA itself with `vm_insert_page()` (`rknpu_gem.c:983,1122`) and CLEARS `VM_PFNMAP`
(`:1148`), so they are ordinary mapped pages and raise `page->_mapcount`. But `vma->vm_file` stays
the DRM node — there is no `get_file(obj->filp)` anywhere. So the shmem inode's `i_mmap` tree has no
record of these VMAs and truncate/evict CANNOT unmap them; it can only report
"still mapped when deleted". The ONLY thing preventing that is the GEM refcount
(`drm_gem_mmap_obj` does `drm_gem_object_get`; `vm_open`/`vm_close` are plain get/put).
=> the corruption requires a GEM reference imbalance / premature free, nothing else.

**3. ★ LEADING CANDIDATE (code-supported, NOT yet proven): unreferenced pointer held across a
blocking wait.** `rknpu_gem_object_find()` (`include/rknpu_gem.h`) is the lookup-then-put
anti-pattern:

```c
obj = drm_gem_object_lookup(filp, handle);   /* takes a ref */
if (!obj) return NULL;
rknpu_gem_object_put(obj);                   /* drops it immediately */
return to_rknpu_obj(obj);                    /* caller gets an UNREFERENCED pointer */
```

(Its doc comment claims "gem object reference count would be increased" — that comment is FALSE.
Do not trust it; it is how this was missed.)

`rknpu_gem_destroy_ioctl()` then holds that unreferenced pointer across
`rknpu_iommu_domain_get_and_switch()`, which blocks up to **6 s per attempt, 3 attempts = ~18 s**,
and dereferences `rknpu_obj->iommu_domain_id` inside the loop. If the process dies in that window
(SIGTERM -> DRM file close -> all handles released -> objects freed) the pointer dangles.

That is EXACTLY the combination this session generated over and over: stalls forcing the 6 s
switch timeout, plus me SIGTERMing the probe. It also fits `ork_sig_teardown()` firing MEM_DESTROY
on every live buffer at signal time.

**Proposed fix (untested, needs the board):** hold a real reference for the duration —
`drm_gem_object_lookup()` at the top of `rknpu_gem_destroy_ioctl` and `drm_gem_object_put()` on
every exit path — or stop `find()` dropping the ref and make all three callers put. The second is
cleaner but touches `rknpu_gem_create_ioctl` and the map-offset path too.

**How to prove it when the board frees up:** the window is huge (seconds), so it should be easy —
add a counter/log in `rknpu_gem_destroy_ioctl` around the wait recording `handle` + whether the
handle still resolves after the wait, then run the stall probe and SIGTERM it mid-stall.

### ★ REFRAME 2026-08-26 — the stall is NOT doorbell/NONBLOCK-specific

This document, and the wiki pages that grew out of it, frame the problem as "the int4 NONBLOCK
doorbell miss". That framing named where the symptom was FIRST SEEN, not its cause, and it is now
contradicted by evidence.

The 1.5B-Q8 `ork_bench` repro stalls on **synchronous (blocking) submits**:

- it fails in `ork_dyn_colsplit`, which issues blocking submits by default ("NO barrier + BLOCKING
  submit — EXACTLY the mcworker", `src/npu/core/colsplit.c:48`); the `ork_dyn_` prefix is misleading
- the ~60 s wait is `rknpu_job_wait()`, which ONLY the blocking branch of `rknpu_submit()` enters
- proof by absence: the watchdog logged `wd_ticks=0` across 8 stalls precisely because its arm site
  was NONBLOCK-only (fixed in #patch39). Had those submits been NONBLOCK it would have armed.
- the fast-abort acts only on `!(job->flags & RKNPU_JOB_ASYNC)` and it fired, cutting submit
  failures 14 -> 2. It could not have touched an async job.

And the signature is IDENTICAL to the async 27B W4A4 case: job committed, `tasks=0/N done`,
PC completed-task counter stuck at `0x0`, no interrupt, no IOMMU fault, MMU idle.

**So the same dispatch failure occurs on both submit paths, and the doorbell is not implicated.**
That is consistent with everything else ruled out today (commit registers byte-identical, regcmd
bytes verified in DRAM, MMU idle, zero faults) and moves the remaining explanation further toward a
dispatch-level hardware issue that is independent of how the job was submitted.

Consequence for anyone reading the older material: treat "doorbell" in these documents as a
historical label for the first-observed instance, NOT as a claim about the mechanism. Do not
restrict future experiments to the NONBLOCK path.

### Measurement note — how coverage is counted WITHOUT circularity

"The watchdog only sees N of M stalls" must not use the watchdog for both numbers. Two
watchdog-independent denominators exist:

- **userspace**: submit failures returning `-ETIMEDOUT` (errno 110), counted by ork-driver/ork_bench
  regardless of whether the sampler noticed;
- **kernel, but not the watchdog**: `rknpu_job_wait()` itself logs
  `"failed to wait job, task counter: %d, ... elapsed time: %lldus"` on EVERY timed-out wait.
  `grep -c "failed to wait job"` is therefore a kernel-side count that does not come from the
  watchdog.

Coverage = `wd_stalls / failed-to-wait-lines`. Residual blind spot, stated honestly: a stall seen by
NEITHER instrument (never sampled and never timing out). Userspace times out on everything
eventually, so that class should be empty, but these two instruments cannot prove it is.

## DESIGN — reshaping submission so short inter-commit gaps cannot occur

Desk work following the change-16 correlation (every stalled job committed 0.24-0.9 ms after the
previous commit on that core) and the change-49 causal test (enforcing 500 us spacing cut stalls
86 -> 12). The `min_commit_gap_us` knob is a DIAGNOSTIC, not a shipping fix: it busy-waits with
`udelay()` because `rknpu_job_next()` can be reached from the completion IRQ, it burns CPU, and it
is blind — it waits a fixed time whether or not the hardware needs it.

### What actually happens today

Commits reach the hardware from exactly three places, all funnelling through `rknpu_job_next()`:

| caller | context | when |
|---|---|---|
| `rknpu_job_done()` (`rknpu_job.c:676`) | **completion IRQ** | previous job finished -> commit the next |
| `rknpu_job_schedule()` (`:883`) | process | a new submit arrives and the core is idle |
| `rknpu_job_timeout_clean()` (`:728`) | process | after reaping |

`rknpu_job_next()` promotes one job off `todo_list`, stamps `hw_commit_time`, and calls
`rknpu_job_commit()` **inline and immediately**. The programming sequence is:

```
PC_DATA_ADDR <- regcmd_addr        PC_DATA_AMOUNT <- regcfg_amount+extra
INT_MASK / INT_CLEAR
PC_TASK_CONTROL <- (0x6|pp) << task_number_bits | task_number     <- the trigger; 0x6 self-clears
PC_DMA_BASE_ADDR <- task_base_addr
```

**Nothing in that sequence asks the hardware whether it is ready to accept a new program.** So the
inter-commit gap is simply "how long the previous job took, plus IRQ latency". A burst of short jobs
(colsplit slices) therefore produces sub-millisecond gaps as a matter of course — the driver has no
pacing mechanism at all, by construction.

### Why the completion path is the generator

The dominant path is completion-IRQ -> immediate commit. That is the one place where the gap is
bounded below only by how fast the previous job ran. `ork_dyn_colsplit` issues many small slices per
core, so consecutive commits land within hundreds of microseconds. This is also why the effect was
invisible in the 27B int4 work: bigger jobs, longer gaps.

### Four candidate reshapes

**A. Coalesce at the source (userspace, ork-driver) — biggest win, most work.**
Submit N slices as ONE job with `task_number > 1` instead of N single-task jobs. The hardware then
walks the task chain itself via the in-regcmd descriptor (`0101:0x0010` next-addr, `0101:0x0014`
next-amount — already documented in AGENTS.md and already implemented for other paths as
`run_chain_i8` / `ork_i4_bchain`). There is no second commit, so there is no gap to get wrong.
Removes the failure mode rather than pacing around it, and it also removes per-submit overhead
(~167 us floor). Cost: a colsplit chain-assembler; userspace, not kernel; must respect the
`max_submit_number = 4095` / `pc_task_number_mask = 0xfff` limits.

**B. Kernel-side pacing via hrtimer — pragmatic bridge, no busy-wait.**
Keep a per-core `next_allowed_commit`. In `rknpu_job_next()`, if the deadline has not passed, arm an
hrtimer for the remainder and commit from its callback instead of inline. Legal from IRQ context
(arming an hrtimer is), and unlike `udelay()` it does not hold a CPU. Cost: deliberately idles the
core for the gap; needs care that a job cannot be committed twice or reordered, and that the timer is
cancelled on reset/abort teardown. Strictly better than the current knob, still blind.

**C. Hardware handshake — the principled fix.**
Poll a PC readiness/idle status before writing `PC_TASK_CONTROL`, instead of timing. Correct by
construction, costs nothing when the block is already ready, and self-scales across SoCs and job
sizes. Requires identifying the right bit — and there is already a lead: the commit snapshot records
`status(pre)` as **0x5000 in some commits and 0x7000 in others**, and `RK_MMU`-style status semantics
are named in mainline `drivers/accel/rocket/rocket_registers.h`, which AGENTS.md already designates as
the cross-reference to consult before guessing on hardware. Cost: RE to confirm the bit's meaning; the
poll must be bounded (it can run in IRQ context) with a fallback to B's behaviour on timeout.

**D. Move the completion->commit hop into a workqueue.**
Cheapest structurally: `rknpu_job_done()` queues work instead of committing inline, which both gets
out of IRQ context (so sleeping waits become legal) and incidentally adds tens of microseconds. But
the added latency is incidental, not guaranteed to exceed whatever the hardware needs, so on its own
it is a mitigation, not a fix. Useful mainly as the enabler that makes B or C implementable with
sleeping primitives.

### Recommendation and sequencing

1. **Prove the mechanism before building anything.** Extend the profiler to record `status(pre)` for
   every commit and correlate it against stalled/healthy. If stalls line up with one status value, C
   is correct and cheap. This reuses existing machinery and needs no refactor.
2. **Ship A for the workload that hurts.** Coalescing colsplit slices removes the gap entirely and
   pays back the per-submit floor as a bonus. It is the only option that improves throughput rather
   than trading it away.
3. **Use C as the general-purpose safety net**, with B as the fallback if the status bit cannot be
   identified.
4. Retire `min_commit_gap_us` once one of the above lands; keep it until then as the reproducer.

### What would falsify this direction

If the `min_gap=1500` cell does NOT reduce stalls further than `min_gap=500`, or if the repeat
`min_gap=0` control does not return to ~86 stalls, then spacing is not the mechanism and this design
is premature — the 86 -> 12 drop would be drift or a side effect of perturbing timing generally.
That control is the gate; do not build A/B/C before it reads clean.

### ★ CAUSAL TEST RESULT — the spacing hypothesis FAILED its pre-registered criterion

| `min_commit_gap_us` | rc | wall | stalls | enforced |
|---|---|---|---|---|
| 0 (control) | 0 | 81 s | 86 | 0 |
| 500 | 124 | 323 s | **12** | 9 |
| 1500 | 124 | 351 s | **46** | 72 |
| 0 (repeat control) | 139 (SIGSEGV) | 2 s | — | — |

Two independent failures:

1. **No dose-response; it inverts.** 1500 us enforced spacing 8x more often (72 vs 9) yet produced 4x
   MORE stalls than 500 us. The pre-registered criterion was "if min_gap=1500 does not reduce stalls
   further than 500, spacing is not the mechanism". It did not.
2. **The repeat control crashed** (userspace SIGSEGV; kernel clean, 0 oops lines), so there is no second
   baseline and drift cannot be excluded on the 86 -> 12 either.

Both enforced cells also timed out (323 s / 351 s vs the control's 81 s), so enforcement costs a great
deal and does not reliably buy anything.

**Therefore the change-16 correlation stands as correlation only.** Do NOT build hrtimer pacing
(design option B) or any timer/threshold in the submit path on this basis — the premise it would rest
on was tested and declined to confirm. Options A/B/C in the design above are all PARKED pending a
mechanism.

**Better explanation to test next: the gap is a SYMPTOM of which op is running, not a cause.** Bursts
of small fast jobs are a workload phase, and that phase may be stall-prone for reasons unrelated to
spacing. This fits the independently-observed clustering on particular shapes (`task_number=4` on
cores 0x2/0x4, `task_number=2` on 0x1, `task_number=9` on all three) and explains why perturbing
timing moves the count around without a clean dose-response.

**Next step (cheap, same machinery, no refactor):** re-aim the profiler from the TIMING axis to the
SHAPE axis — record `task_number`, `regcfg_amount`, `core_mask`, `iommu_domain_id` per job and split
stalled vs healthy on those. If stalls belong to particular op shapes that is far more actionable than
pacing; if shape shows nothing while timing does, the timer proposal revives on firmer ground.

**Method note worth keeping:** pre-registering the falsification condition BEFORE running the sweep is
what made this readable. The 86 -> 12 result on its own looked like a win and would very likely have
been written up as one.

### ★ SHAPE-AXIS RESULT — the dominant factor is the IOMMU DOMAIN, not the op shape

Userspace already logs the shape of every FAILED submit, so the stalled-side distribution was known.
What it cannot give is the BASE RATE — how often each shape runs successfully — and without that
"stalls happen on task_number=4" is uninformative if that shape is most of the traffic. Kernel-side
per-commit shape logging (`prof_mask` bit 2) supplies the denominator.

Stall rate by shape (`task_number/core_mask/domain`), one 1.5B-Q8 run:

| shape | commits | stalls | rate |
|---|---|---|---|
| 4/0x4/**2**, 4/0x2/**2**, 2/0x1/**2** | 7 each | 6 each | **85.7%** |
| 4/0x4/**1**, 4/0x2/**1** | 9 | 4 | 44.4% |
| 9/*/**0** | 11 | 2 | 18.2% |
| 2/0x1/**1** | 53 | 5 | 9.4% |
| 2/0x4/**1**, 2/0x2/**1** | 44 | 1 | 2.3% |

**Aggregated by domain: dom 2 = 18/21 (85.7%), dom 0 = 6/33 (18%), dom 1 = 20/200 (10%).**
`task_number` and `core_mask` are second-order inside that. Domain 2 is ~8.5x worse than domain 1.

**This converges with a finding parked earlier the same day.** `rknpu_iommu_switch_domain()` creates
extra domains with `iommu_get_dma_cookie()` plus a hand-patched
`dst_domain->type |= __IOMMU_DOMAIN_DMA_API`, which SKIPS `iova_reserve_iommu_regions()` that the real
`iommu_dma_init_domain()` path performs. So domain 0 has the device's reserved/MSI regions carved out
of its IOVA space and domains >0 do not. Two independent lines now point at the same asymmetry.

It also retro-explains an early result: on the int4 probe `ndom=2` TRIPLED the stall rate vs `ndom=1`,
which was read at the time as "domain switching is implicated". The better reading is
**"domain >0 is implicated"**.

**Caveat:** `prof_mask=3` costs ~500 printks and the run timed out (479 commits logged vs 1069 in an
unlogged run), so ABSOLUTE rates here are perturbed by the instrumentation. The RELATIVE comparison
between shapes within the same run is not, and that is the claim being made.

**Next test (cheap, decisive):** force everything into domain 0 and see whether stalls collapse. If
they do, the extra-domain construction is the mechanism and the fix is to build domains >0 through the
proper `iommu_dma_init_domain()` path rather than `iommu_get_dma_cookie()` + type patching. That is a
far more tractable target than either pacing or a hardware handshake.

## ★★ ROOT CAUSE LOCALIZED — stalls are the first ~3 commits after an IOMMU domain switch

One run, 447 commits, 35 stalls, `prof_mask=3`:

| position after switch | commits | stalls | rate |
|---|---|---|---|
| 0 (first after switch) | 26 | 12 | **46.2%** |
| 1-2 | 50 | 23 | **46.0%** |
| 3-9 | 116 | **0** | **0.0%** |
| 10+ | 255 | **0** | **0.0%** |

Controlling for position (`sinceswitch >= 3` only): dom 0 = 0/180, dom 1 = 0/171, dom 2 = 0/20.
**The domain-index effect disappears completely.**

**Every stall in the run happened within 3 commits of a domain switch. Nothing stalled after that, in
371 commits.**

### This invalidates two earlier conclusions from this same investigation

- **"Domain >0 is implicated" (the shape-axis result) is WITHDRAWN.** dom 2 looked catastrophic only
  because it is visited briefly, so a large fraction of its commits sit right after a switch. The
  `iova_reserve_iommu_regions` asymmetry is a red herring for this bug.
- **The inter-commit-gap correlation (change 16) is also a confound.** The first jobs into a
  freshly-switched domain arrive in a burst, so short gaps CO-OCCUR with post-switch position. That is
  why enforcing spacing produced no dose-response and inverted at 1500 us — it was pacing a symptom.

### It explains previously unexplained results

- `ndom=2` tripled the stall rate vs `ndom=1` on the int4 probe: more switches, not worse domains.
- The 27B int4 case stalls less: fewer switches per unit of work.
- "MMU idle at stall time" fits: the core never issued a memory request because its own post-switch
  state is wrong, not because translation failed.

### We already suspected this and never measured it

`src/npu/core/domain.c:223` carries a comment about the *"first-submit-into-a-freshly-switched-domain
hazard"*, and `ork_dom_prime` exists to prime a domain. The hazard was written down in our own
userspace and never quantified.

### Caveats

Single run; 35 stalls. The separation (0/371 beyond position 2) is enormous, but it should be
replicated before anything is built on it. Note also `prof_mask=3` perturbs absolute rates (run timed
out) — the position comparison is within-run and unaffected.

### Next

1. **Replicate** on a second run, and on the int4 probe with `ndom=2`.
2. Then test the mechanism: does a quiesce/settle/prime after `iommu_attach_device` — before the first
   commit — eliminate it? `ork_dom_prime` already exists userspace-side; the kernel equivalent would be
   refusing to commit until the NPU is confirmed idle post-switch.

### ★ COOLDOWN CAUSAL TEST — FAILED. Time-since-switch is not causal either.

A-B-A-B, `ork_bench` 1.5B-Q8, kernel #29:

| cooldown | rc | wall | **stalls** | cooldowns fired |
|---|---|---|---|---|
| 0 | 0 | 83 s | **86** | 0 |
| 1000 us | 0 | 82 s | **86** | 105 |
| 0 | 0 | 82 s | **86** | 0 |
| 1000 us | 0 | 81 s | **86** | 105 |

**The cooldown fired 105 times per treatment cell and the stall count did not move: 86 every run.**

That is the THIRD correlate of "just after a domain switch" to survive replication and then fail its
causal test — after inter-commit spacing and after domain identity. All three are real correlations;
none is the mechanism. The common thread is that **delaying does not help**, which argues the problem
is not a settling time at all.

Silver lining: the workload is now an exceptionally good A/B substrate — 86 stalls and 7.19 tok/s on
every single run, so any real effect is immediately visible.

### Sub-block register dump (CNA 0x1000 / DPU 0x4000 / CDMA 0x5000) — INCONCLUSIVE

Built to test whether the NPU is stuck mid-read (a DMA outstanding upstream of translation) or never
issued one at all. Result: the healthy and stalled snapshots are **byte-identical**, all three blocks,
both cores. CDMA reads all-zeros in both.

**Do not read that as "no hung read".** The control is likely invalid: the "healthy" snapshot fires
when a job is present with `wd_flat == 0`, which only means RECENTLY COMMITTED — plausibly the same
register state as a job that is about to be reported stalled. A genuinely-executing snapshot needs the
counter-advanced branch, which a 10 ms sampler almost never catches (the first attempt captured zero).

To make it decisive: (a) widen the window past the first 8 words per block to find any word that moves
at all, and (b) get a real running control — sample far faster, or trigger the baseline dump from a
known-executing point rather than from the watchdog.

### Working state

- Kernel #31. `switch_escalate=3` (default ON) reaps stuck jobs after 3 consecutive switch timeouts —
  it fired twice in one run and the board survived, where the same workload had previously hard-wedged
  into a plug cycle twice.
- `dom_cooldown_us` default 0 and should stay there: measured useless.
- Log capture moved to `/var/lib/ork-logs/kmsg.log` on the NVMe (`/var/log` is tmpfs and was eating
  datasets on every reboot).
