# MODULARIZE WIP — recovery doc

**Branch:** `refactor/modularize-precision` (off `main` @ `1e765a4`, v1.0.14)
**Plan:** [`MODULARIZE_PLAN.md`](MODULARIZE_PLAN.md) — read it first; this file is the live state.
**Goal of round 1:** split `src/npu.c` (15,313 lines) by precision into `src/npu/` — scaffold `npu.c`
stays, siblings `npu/{sdp,f16,i16,i4,ssm}.c` + folder `npu/i8/{regcmd,pack,fold,run,chain,dyn,probe}.c`.

---

## Current state — ROUND 1 COMPLETE

`src/npu.c`: **15,313 → 5,316** (−9,997, 65%). Largest file in the repo is now the scaffold at 5,316;
largest module file is `i8/dyn.c` at 1,046.

```
src/
  npu.c              5,316   scaffold: orki_run, run_multicore, seq scheduler, bmm dispatch,
                             norm/softmax, async, CPU pack helpers, ork_npu_init
  npu/
    internal.h         449   types, dtype predicates, env knobs, hot static inlines
    core.h             180   the substrate interface
    core/            1,326   device buf submit sched domain mode prof            (7)
    i8/              4,572   regcmd pack fold run chain dyn probe      + i8.h    (7)
    f16/             1,948   regcmd run perchan stream probe replay    + f16.h   (6)
    i4/              1,745   quant pack run chain stream               + i4.h    (5)
    i16/               774   regcmd act chain probe                    + i16.h   (4)
    sdp.c              151   shared activation curves + LUT machinery
    ssm.c              337   Mamba-2 / SSD scan
```

Every precision is a folder with its own subtree header; `core/` is the dtype-agnostic substrate;
`sdp.c`/`ssm.c` are single files because neither is a precision.

| commit | what |
|---|---|
| `189553a` | plan + WIP doc |
| `b83269e` `1824466` | build gates unhardcoded, `clean`→`$(COBJ)`, `sudo -E`, AGENTS fix |
| `c4541c2` | 43 internals → `orki_*` (3,104 sites) |
| `b685614` | `npu/internal.h` — the private ABI |
| `ae1fc70` | `npu/ssm.c` — first TU off the monolith |
| `468a11f` | check-registry: fixed a 24% blind spot |
| `38e16e9` | `npu/i4.c` — int4 chain/doorbell/stream |
| `b46b036` | `npu/core.h` — substrate interface declared up front |
| `88c7a6f` `0ed026a` | `npu/core/*.c` — all 7 substrate modules |
| `d196d95` | `npu/f16.c` — the fp16 datapath |
| `a544942` | `npu/i8/` — 7 modules |
| `824e8da` | `npu/i16.c` |
| `3f50ac5` | `npu/sdp.c` + the i4 pack/quant remnant |
| `cd1bd72` | f16/i4/i16 → folders, one layout for every precision |

**Invariants held at every commit:** `make test` ALL PASS, 0 watchdog, ACT_RESET 50 / 18 sites,
check-registry clean, every exported symbol `ork_`/`orki_`/`orkd_` prefixed. End-to-end flat
(`ork_bench` 220.0 vs 220.8 prefill, `7a8152f`).

### What round 1 taught, for round 2

1. **Boundary size is the cost, not contiguity.** The mover works by name, so a scattered module costs
   the same as a contiguous one. The "order lifts by contiguity" heuristic was measuring the wrong thing.
2. **Declare the shared interface FIRST.** Once `core.h` existed, lifts stopped rediscovering the same
   substrate. ~90% of every precision's inbound boundary was that substrate.
3. **Functions move by name; STATE does not.** Every lift's real work was the structs, tables and
   globals that live outside any function. Losing them is how the f16/i4/i16 folder pass broke.
4. **De-static exports.** Prefix at the point of de-static; a sweep afterwards needs two rounds because
   exporting one symbol reveals the next.
5. **Two C traps, each hit twice:** de-staticing a `static inline` leaves a bare `inline` (no external
   definition, links fail); an anonymous-struct global cannot be extern-declared (name the type, or move
   its only reader beside it).
6. **Never suppress the mover's brace-balance report.** It caught every structural corruption; the one
   time it was swallowed by `subprocess`, a truncated struct reached the build.

---

## PRE-SPLIT BASELINES — the numbers every later commit is diffed against

Captured on RK3588 `10.3.0.236` at `main` @ `1e765a4` + commit A, governors performance
(dmc 2112 MHz, cpu4 2304, cpu6 2352).

**1. Exported symbols over `$(COBJ)`** — 381 (`nm -g --defined-only`, `T`/`D`/`B`/`R`).
Stored: `scratchpad/baseline/symbols.txt`. Regenerate and diff after every lift; the set must be
IDENTICAL except for the deliberate `orki_*` renames from commit B.

```sh
nm -g --defined-only src/npu.o src/soc.o src/soc/rk3588.o src/soc/rk3576.o \
   src/neon_activations.o src/ork_ops.o src/orkd_client.o src/ork_gptq.o \
 | awk '$2 ~ /^[A-Z]$/ {print $2, $3}' | sort -u
```

**2. `mc_prof 256 2048 2048 20`** — the cross-TU-inlining canary (per-submit cost is what regresses
if a hot `static` stops being inlined):

```
1-core: 2379.4 us/matmul   (42 submits: copy 187.7/sub, submit 727.8/sub, acc 188.0/sub)
3-core: 1429.3 us/matmul   (scaling 1.66x)
```

**3. `ACT_RESET` across a full `make test`** — **50 calls over 18 distinct call sites**.
Offsets move when code moves, so the invariant is the **count and the site count**, not the offsets.
Top sites: `0x3e1dc`×12, `0x3de3c`×10, `0x44054`×6, `0x2ef98`×6, then 14 singletons/pairs.

**4. `make test`** — ALL TESTS PASSED, `tests/sbc_attest.txt` `CORE_SHA=f377101c…` (unchanged by
commit A: it touches no `ATTEST_SRCS` file).

⬜ **Not yet captured:** `mode_probe` op→op matrix. Capture before the first lift (commit D).

---

## ⚠️ Gotcha found while capturing the baselines (fixed in commit A)

**`sudo` strips `ORK_*`, and `make test` ran `sudo ./$t`** — so a knob set on a `make test` command
line **never reached the test binaries**. `ORK_DEBUG_RESET=1 make test` reported **0** resets; the true
number is **50**. Anything ever validated as "`make test` under `ORK_SSM_KEEPWARM=0` / `ORK_MIXED_NOTHRASH=1`"
via that route exercised the DEFAULT config, not the knob — including the procedure written into
AGENTS.md §4 and into this plan's own verification ladder.

Fixed by `SUDO ?= sudo -E` in the Makefile (3 call sites). Verified: `ORK_TRACE=1 make test-only T=test_bmm`
now emits 92 `[ork-trace]` lines where it emitted 0 before. **Use `make test` for knobbed runs only on a
tree that has this fix**; otherwise invoke the binary directly as `sudo env KNOB=1 ./test_x`.

---

## Verification ladder (run at every lift; §7 of the plan)

```sh
board -c 'make -j4'                                   # a: clean, zero new -Wall warnings
board -c 'nm -g --defined-only … | sort -u'           # b: symbol set == baseline 1
board -c 'make test'                                  # c: ALL PASS (static goldens = bit-exact)
board -c 'ORK_SSM_KEEPWARM=0 make test'               # d: and ORK_MIXED_NOTHRASH=1 (works now, see above)
board -c 'sudo env ORK_MM_TIMEOUT=2500 timeout 300 ./mode_probe'   # e: matrix unchanged
board -c 'ORK_DEBUG_RESET=1 make test'                # f: 50 resets / 18 sites
board -c 'sudo ./mc_prof 256 2048 2048 20'            # g: within baseline 2 (final only)
```

---

## Board ops

- Sync: `tools/util/sync_daemon.sh`, or `rsync -a src/ board:~/llama.cpp/ggml/src/ggml-ork/ork-driver/src/`.
  **Sync all of `src/`, never a single file** — a stale `npu.c` beside a new `npu/i8.c` builds a Frankenstein.
  ⚠️ `rsync -a a.c b.c dest/` FLATTENS both into `dest/` — pass one source per invocation, or use `-R`.
- Stale artifacts bite: a leftover `src/ork_gptq.o` from Aug 8 silently linked an old implementation.
  `make clean` before any lift's first build (commit A made `clean` purge all of `$(COBJ)`).
- Board `make test` ≈ 33 s–4 min; full build ≈ 20 s. `timeout` SIGTERMs `sudo`, which does NOT forward —
  use `sudo timeout N …`, not `timeout N sudo …`.
- Wedge: SSH alive → `sudo reboot`. SSH dead → HA plug cycle "Rock 5B Plug". Plug cycle fails → SPI
  reflash, physical. Governors reset every boot — re-pin before benchmarking.

## Parked, do not lose

- `origin/wip/moe-saga-2026-08-18` — the real GPTQ quantizer (`main` carries a `-ENOSYS` stub),
  `examples/test_gptq.c`, 9 MoE/SRAM/domain probes, `COLSPLIT_MULTIPREC_WIP.md` (stale, #45 landed),
  `MOE_LAYER_BLOCK_WIP.md` (#54, active — will touch the same `npu.c` regions; sequence, don't race).
- Fork submodule `stash@{0}`, board `stash@{0}` — redundant copies of the same.
