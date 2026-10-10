# WIP recovery — `.235` state, and the IOMMU domain-reference leak (2026-10-09)

**UPDATE: `.235` was reset by hand and is back up on the same kernel. The sections below describing it
as down are kept as the record of what happened.**

**`.235` was hard down and needed a physical RST.** Do not power-cycle it from Home Assistant: the only
available outlet there is `Rock 5B Plug`, which is **`.236`** (another agent owns `.236` — hands off).

## THE KERNEL ON `.235` BOOTS. This is not a boot failure.

Say this plainly because the recovery differs: **the kernel currently installed on `.235` booted
successfully, twice, and gave a working shell both times.** After the final install I rebooted,
logged in over ssh, and ran a full verification (`uname -r`, DDR, A76, device-tree, SRAM, module
params) — all good. It then wedged ~100 s into the **first NPU workload** on that kernel, not during
boot. So `/boot/RESTORE_STOCK_KERNEL.sh` will be reachable after a reset; no SD-card or serial-console
recovery should be needed.

### The exact last thing done to it

1. Installed the new kernel payload (`/var/tmp/k235-sram.tgz`) with `/var/tmp/install235b.sh`.
2. Rebooted. Booted fine. Observed A76 at 2304/2352 (overlay had been removed), so restored
   `user_overlays=rk3588-a76-2400` from `/boot/armbianEnv.txt.bak-pre-sram` and rebooted again.
3. Booted fine. `bash /var/tmp/verify235.sh` passed everything (below).
4. Set `rknpu.dom_reclaim=0`, ran `make -j8`, then
   `sudo tools/util/npu_guard.sh -- timeout -s TERM 420 ./i4`.
5. Netconsole: `allocate iova` → `soft reset, num: 6` → three `HEALTHY commit snapshot` /
   `HEALTHY PCBLK` dumps → `soft reset` → `soft reset` → **silence at 99.9 s uptime**. No panic, no
   oops. Ping and ssh dead ever since.

Nothing was `kill -9`'d. The board was not mid-`make test` — it was running the single `./i4`
example under the guard.

## What is installed on `.235` right now

| | |
|---|---|
| `/boot/Image` → | `vmlinuz-6.1.115-vendor-rk35xx-npu` (the new board-tree + SRAM build) |
| `/boot/dtb` → | `dtb-6.1.115-vendor-rk35xx-npu` |
| `/boot/uInitrd` → | `uInitrd-6.1.115-vendor-rk35xx-npu` |
| `/boot/armbianEnv.txt` | `user_overlays=rk3588-a76-2400` RESTORED (backup at `/boot/armbianEnv.txt.bak-pre-sram`) |
| stock fallback | `/boot/RESTORE_STOCK_KERNEL.sh` → `6.1.115-vendor-rk35xx`, untouched and intact |
| `rknpu.dom_reclaim` | was set to **0** at runtime; module params are not persistent, so a reboot restores the default 1 |
| ork-driver | `e6ac93d` (1.0.165), clean tree, built |

Kernel source of that build: `board-snapshot-2026-10-09` (`8fac2134ab53`, `.236`'s working tree)
+ `.235`'s stock armbian `.config` + `CONFIG_ROCKCHIP_RKNPU_SRAM=y` + the leak fix below.
Container worktree `/root/k235`, branch `b235-sram` (fix applied, uncommitted there).

Verification that passed on it before the wedge:

```
uname -r                6.1.115-vendor-rk35xx-npu
DDR                     performance @ 2400000000
cpu4 / cpu6             performance 2400000 / 2400000   (WITH the overlay; see below)
/proc/device-tree/npu@fdab0000/rockchip,sram      present
RKNPU: sram region: [0x00000000ff001000, 0x00000000ff0f0000), sram size: 0xef000   (956 KB, was 0)
```

**The A76 ungating is NOT in the board snapshot.** Without the overlay the board comes up at
cpu4=2304 / cpu6=2352; the built `rk3588-rock-5b-plus.dtb` still has `opp-2304000000 = <0xf9 0x24>`
and `opp-2400000000 = <0xf9 0x80>`. Keep `user_overlays=rk3588-a76-2400`.

## Payloads kept (so nothing has to be rebuilt)

| what | where |
|---|---|
| board-tree + SRAM + fix (what is installed) | `.235:/var/tmp/k235-sram.tgz`, `.239:/tmp/k235-sram.tgz` (sha256 `eb5d8403…`) |
| PR #390 clean 13 commits | `.239:~/k235-payloads/k235-pr390-clean13.tgz` |
| PR #390 + counters + `dom_fix` | `.239:~/k235-payloads/k235-pr390-instrumented.tgz`, `.235:/var/tmp/k235-pr390-instrumented.tgz` |
| safe installer (NEVER `tar -C /`) | `.235:/var/tmp/install235b.sh` |
| verification script | `.235:/var/tmp/verify235.sh` |
| suite runner | `.235:/var/tmp/val235.sh` |
| suite logs | `.235:/var/tmp/val{A,B0,B1}.log` (tmpfs-safe: `/var/tmp`, survives reboot) |

Container (`10.3.0.239`, `docker exec kbuild`) branches, all in the `/root/linux-rockchip` repo:

- `/root/k235` on `b235-sram` — the installed build (leak fix applied, uncommitted)
- branch `instr` (`3dbf92d51ab3`) — armbian + 13 PR #390 commits + counters + `dom_fix`
- branch `k235-rknpu-rework` — armbian + the 10-commit series (superseded)
- `/root/k390fix` — worktree of the VENDOR repo on branch `rknpu-domain-ref-leak`, commit `79b6729cb`,
  **pushed** to `oRKLLM/rk3588-kernel`

## THE LEAK — root-caused, fixed, measured (this part is COMPLETE)

`rknpu_job_schedule()` takes one `iommu_domain_refcount` reference per job and records it in
`job->dom_held`. PR #390 releases it on completion (`rknpu_job_done`), on abort (`rknpu_job_abort`)
and for the RUNNING job in `rknpu_job_timeout_clean()`. A fourth path releases nothing: the same reap
drains the rest of that core's `todo_list` into `schedule_work(&job->cleanup_work)`, and
`rknpu_job_free()` never looked at `dom_held`. One leak per queued job reaped.

Trigger: ork-driver's own `ork_npu_reap_stuck()` (`src/npu/core/device.c`), whose NONBLOCK fp16 dummy
QUEUES rather than runs when the stuck core's job is not yet past its timeout, and which the BLOCKING
drain behind it does not retire (that job is aborted, and abort unlinks only itself).

Fix: `rknpu_job_free()`, three lines, same place and shape as the fence rescue already there.

Measured on `.235` with per-callsite counters (`rknpu.dom_dbg`) and the fix behind `rknpu.dom_fix`,
one kernel, two full `make test` runs:

| arm | `[4]` freed holding a ref | `[5]` queued reaped | domain-switch timeouts | reclaim |
|---|---|---|---|---|
| `dom_fix=0 dom_reclaim=1` | **3** | 3 | 1 | `refcount=3 -- forcing to 0` |
| `dom_fix=1 dom_reclaim=0` | 3 | 3 | **0** | never fires |

Written up and pushed: wiki `Exp-2026-10-09-The-Domain-Reference-Leak`, Experiment Log entry
2026-10-09 (IV), cross-links on the PR #390 page and Board Provisioning.

## Still open

- `.235` needs a physical RST. After it comes back: `make test` on the board-tree + SRAM kernel has
  **never been run** — that is the one unfinished verification of the direction change.
- The `./i4` hard wedge on the board-tree kernel is new behaviour: the same shapes fail *gracefully*
  on the PR #390 kernel and on stock armbian. Worth bisecting (the board tree carries ~49 diagnostic
  hunks plus a watchdog) before trusting that kernel for NPU work on `.235`.
- `test_slice_rescue` / the four `i4` shapes: characterised (board-specific H envelope, see
  `OPS_REGISTRY.md` and PR oRKLLM/ork-driver#26) but not fixed.

## After the RST (2026-10-09, later)

Board back on the same kernel: `uname -r` `6.1.115-vendor-rk35xx-npu`, DDR 2400, A76 2400/2400,
`rockchip,sram` present, `dom_reclaim` back to its default 1.

Ran `make test`'s full list **minus `i4` and `test_slice_rescue`** (`/var/tmp/val235_safe.sh`, same
order, under the guard): **fail=0, everything passes** — `ORKD_DOM_API`, `orkd_seq_probe`,
`chain_xition_probe`, `test_bmm_fused`, `chainrr_conc_probe`, `test_ssd_chunk_npu`,
`test_mode_transition`, and notably `TEST_F16COLSPLIT` (which FAILED on the PR #390 build).
`test_speed` 3-core 3405.0 us (scaling 1.78x) vs 3637.7 us / 1.65x on the PR #390 build, so it is now
inside the 3600 us limit. Kernel log for the whole run: 0 `reclaim`, 0 domain-switch timeouts,
0 `failed to switch` — the leak fix holds on this tree too. Log: `.235:/var/tmp/valS.log`.

**Still unrun on this kernel: `i4` and `test_slice_rescue`** — the known wedger, and `.235` has no
remote power control.

## int4 root-caused (2026-10-09, final)

`i4` + `test_slice_rescue` re-run on the board-tree + SRAM kernel. SRAM changes nothing
(`freeSRAM=956KiB` now shows in the dumps, same four failures).

**Cause 1 — the bank-width N-tile.** `Wb=(131072/K)&~63` is too wide on `.235`.
`ORK_I4_WB=64` makes M=8 K=512 N=256 and M=32 K=1024 N=128 pass `maxerr=0`, and takes
**`TEST_SLICE_RESCUE` FAIL -> PASS (all 11 cases, every golden matching the `.236` value)**.
Not universal: M=64/128/256 at K=512 N=256 pass at the default `Wb=256`.

**Cause 2 — H=8 at K=2048 overshoots.** H=2/3/4 agree; H>=6 fails. The two remaining `i4` shapes.

**A blanket `ORK_I4_H` override is UNSAFE.** `ORK_I4_WB=64 ORK_I4_H=4` makes all 16 `i4` shapes
report `maxerr=0`, but `test_slice_rescue`'s `i4-wideK` (K=10240) and `i4-rem2560` (K=18944) then
return `rc=0` with a WRONG golden. The H fix must be a per-K table edit.

**The panic.** Stranded NONBLOCK jobs (`NOT dispatching ... run_cnt 0`) on all three cores ->
power-off delay fires -> `npu1` won't idle -> stock vendor `panic("panic_on_set_idle")` in
`drivers/soc/rockchip/pm_domains.c` (verified UNCHANGED from the armbian base). Only the board tree
reaches it: `failed to set idle` appears exactly once in the whole netconsole history. Not the
watchdog (`wd_period_us=0`) and not the per-core reset (`RKNPU_JOB_STALLED` only).

Logs on the board: `/var/tmp/{i4_sram,i4_wb64,i4_wb64h4,sr_sram,sr_wb64,sr_wb64h4,valS}.log`.
