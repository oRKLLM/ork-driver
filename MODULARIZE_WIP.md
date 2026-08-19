# MODULARIZE WIP — recovery doc

**Branch:** `refactor/modularize-precision` (off `main` @ `1e765a4`, v1.0.14)
**Plan:** [`MODULARIZE_PLAN.md`](MODULARIZE_PLAN.md) — read it first; this file is the live state.
**Goal of round 1:** split `src/npu.c` (15,313 lines) by precision into `src/npu/` — scaffold `npu.c`
stays, siblings `npu/{sdp,f16,i16,i4,ssm}.c` + folder `npu/i8/{regcmd,pack,fold,run,chain,dyn,probe}.c`.

---

## Current state

| commit | what | status |
|---|---|---|
| 0 | this doc + `MODULARIZE_PLAN.md` | ✅ `189553a` |
| A | `check_registry.sh` globs all lib sources; `clean:` → `$(COBJ)`; `$(SUDO)` = `sudo -E` | ✅ `b83269e` (+ AGENTS fix `1824466`) |
| B | rename 43 generic internals to `orki_*` (one TU, no moves) | ✅ `c4541c2` |
| C | `src/npu/internal.h` (types, macros, `ork_now_us` inline, externs) | ✅ `b685614` |
| D | `src/npu/ssm.c` — first real TU off the monolith (317 lines) | ✅ `ae1fc70` |
| — | `check-registry` check 5: fixed a 24% blind spot (wrapped prototypes) | ✅ `468a11f` |
| E | `src/npu/i4.c` — int4 chain/doorbell/stream (728 lines) | ✅ `38e16e9` |
| F | `src/npu/f16.c` | ⬜ next |
| G–I | `i8/*` → `i16.c` → `sdp.c` (+ the i4 pack/quant sweep) | ⬜ |
| J | docs (AGENTS §4 tree, README, OPS_REGISTRY, tools/re/README) | ⬜ |
| K | attest refresh (if CORE moved) + fork CMake file list | ⬜ |

**Working tree:** on `refactor/modularize-precision`. **Board** synced to the branch, governors pinned.
**`src/npu.c`: 15,313 → 14,088 lines (−1,225).** internal.h 274, i4.c 728, ssm.c 335.

### Order changed from the plan: D is `ssm.c`, not `sdp.c`

The plan ordered `sdp.c` first as "self-contained math". It is not — the SDP substrate is **scattered
across 8 sites from line 1542 to 9443** (`sdp_canon`/`ork_npu_sdp_stamp` up in the fold neighbourhood,
the curve builders down among the i8 activations), and `silu_calibrate_idx` reaches into
`ork_npu_probe_silu_std`. `ssm` by contrast is **one contiguous block** with a 3-in/3-out boundary, so it
is the better first proof of the recipe. SDP moves later, when the i8 activations it interleaves with
have been lifted and the split is obvious. **Lesson for the remaining lifts: pick the block by measured
contiguity, not by the plan's guess at cohesion.**

### Remaining lifts, ordered by MEASURED contiguity (not the plan's guess)

`src/npu.c` after commit D, functions bucketed by name/dtype, "runs" = clusters separated by >400 lines:

| module | fns | span | runs | largest contiguous runs |
|---|---:|---:|---:|---|
| i16 | 21 | 861–14252 | 7 | **8290–9474 (1184)**, 7071–7427, 14222–14252 |
| f16 | 63 | 74–14700 | 11 | 6505–8115 (1610), 14218–14700, 5669–6071 |
| i4 | 70 | 92–14829 | 9 | 13378–14829 (1451), 3397–4571, 5175–5593 |
| i8 | 157 | 869–14762 | 10 | 2584–4529 (1945), 10746–12653 (1907), 5385–7160 (1775) |
| core (scaffold) | 250 | 62–14730 | 8 | 62–3392 (3330), 3884–5947, 8695–9991 |

**The plan's order was backwards.** It put i16 first for having the fewest functions (21). But function
count is not the effort — the number of disjoint **splice sites** is, and i16 is the *most* fragmented
bucket in the file (1.6 fns/site). Lines moved per splice:

| module | lines | sites | lines per splice |
|---|---:|---:|---:|
| i8 | ~5,500 | 37 | **148** |
| i4 | ~2,100 | 17 | 123 |
| f16 | ~1,900 | 21 | 90 |
| i16 | ~950 | 14 | **67** |

And fragmentation is not fixed — a module's sites **coalesce as its neighbours vacate**. Simulated for
i16: **14 sites now → 11 after i8 → 5 if it goes last.** Same 950 lines for a third of the splices.

**Revised order: i4 → f16 → i8 → i16.** i4 next (biggest contiguous run, 1,451 lines, moderate risk);
i8 third rather than second so the private ABI is proven across three modules before the dyn-API
entanglement; i16 last, when it has collapsed to ~5 sites. Within each lift: dominant run first,
stragglers second.

## What each lift actually costs (measured, updated per lift)

| lift | lines moved | boundary in | boundary out | internal.h after |
|---|---:|---:|---:|---:|
| D `ssm.c` | 317 | 3 | 3 | 165 |
| E `i4.c` | 728 | 21 + 4 types/enums | 4 | 274 |

The boundary grows because the monolith is **layer**-organised, so a precision module lifted from the
middle of a layer reaches back into everything around it. Expect internal.h to keep growing; that is the
private ABI becoming explicit, not a defect. Every de-static must carry an `ork_`/`orki_` prefix — after
commit E all 413 exported symbols do, which is cleaner than the pre-split baseline.

**Validation cadence (revised after the wedge):** build + `nm` symbol diff + `check-registry` per lift
(~40 s, no NPU); full `make test` once per module. **Never cap `make test` from outside** — the board's
own per-test `timeout 360` is the bound. An outer `timeout 540 make test` killed make mid-test, orphaned
`test_silu_native` on an in-flight submit, and wedged the NPU into a reboot.

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
