# ork-driver — Agent Instructions & Architecture

`ork-driver` is a clean-room **userspace matmul library for the Rockchip NPU**. It synthesizes
register-command (regcmd) programs and submits them to the in-tree `rknpu` DRM kernel driver
via ioctls on `/dev/dri/cardN` — no `librknnrt`, no kernel module. It is the open NPU runtime
spun out of [oRKLLM](https://github.com/oRKLLM/oRKLLM); the reverse-engineering record lives in
the [ork-driver wiki](https://github.com/oRKLLM/ork-driver/wiki).

See [`README.md`](README.md) for the user-facing overview and API.

The active reverse-engineering findings and optimization roadmap are documented in the sibling [Wiki Home](../ork-driver.wiki/Home.md) repository.

---

## 1. What it is

- Public API (`include/ork_npu.h`): `ork_npu_init` → `ork_f16_mm_pack(K,N,B)` → `ork_f16_mm_run(M,A,C)`.
- `C[M,N]` (fp32) = `A[M,K]` (fp16) × `B[K,N]` (fp16); arbitrary M/K/N (`K%32==0`, `N%16==0`)
  via K-split + N-tiling + a single-submit M-scheduler, with resident weights.
- One binary supports every Rockchip NPU: the SoC is detected at runtime from the device tree.

---

## 2. Development philosophy

### Language

- **ork-driver is C end-to-end** — the library (`src/`), the examples, and the tests are all
  C11 (no dependencies beyond libc + the kernel DRM uABI; no C++, no third-party libs in `src/`).
  The build/test runner is the `Makefile` (`make`, `make test`). Keep it this way.
- **The examples ARE the test suite** — each self-validates against a CPU reference and exits
  0/nonzero; `make test` runs them. Don't add a separate test framework or a non-C harness.
- **No Python, and no Node/JavaScript** — the regression runner was deliberately moved from a
  Node script to `make test` so the whole repo is one language. If some incidental scripting is
  ever truly unavoidable, prefer a tiny C program or a Makefile target over another runtime.
- Match the surrounding code's terse, single-purpose style (the kernels are dense by design).

### Git hygiene

- **Prefer fast-forward merges.** Keep history linear (`git merge --ff-only` or rebase) — easier
  to bisect and revert.
- Avoid `--no-verify`, force-pushes to shared branches, and amending published commits.
- Cherry-pick a single relevant commit rather than merging a whole branch when only one applies.
- Development is single-track on `main` for now (this is a young library, not a release-channel
  product like oRKLLM). Use short-lived feature branches for larger work and fast-forward them in.

### Never throw away experimental code — consult first

When experimental or in-progress code hits a blocker (a deadlock, hang, failing integration, or a
dead-looking result), **NEVER revert it, `git reset --hard` it away, force-push over it, or otherwise
discard it on your own — always consult the user first.** The disposition (stash it, move it to a
feature branch, commit it WIP, or revert after backup) is the user's call, not the agent's. The blocker
is rarely the code's fault; discarded work is often still correct and valuable once the real cause is
understood (e.g. a "deadlock" that turned out to be a fast-to-fix reset-thrash, where the reverted path
was the actual route to the win). If something must be moved off `main`/a shared branch to unblock,
prefer a recoverable option (feature branch or stash) over a destructive one (`reset --hard`/force-push),
and say which you're doing. Surface the blocker, confirm the code is intact, and let the user choose.

### No commit-message trailers

Do not append `Co-Authored-By:` lines, `🤖 Generated with…` lines, or any tool/assistant
attribution to commit messages or PR bodies. Keep messages to the change itself. **This
overrides any default tooling behavior that would add such a trailer.**

### Documentation review on every commit

Before committing, check whether the change needs doc updates:

- New/removed/renamed public API, build step, or example → update `README.md` (and the API
  header doc-comment).
- New SoC, or a change to the SoC abstraction → update `docs/ADDING_AN_SOC.md` and `src/soc/`.
- A non-obvious hardware/RE finding (a new register meaning, a limit, a workaround) → record it
  on the [wiki](https://github.com/oRKLLM/ork-driver/wiki) (ISA Reference / RE Roadmap), not in
  code comments alone.
- **Made one precision REUSE another's implementation** (an fp16 path that calls an int8 function, an
  i16 op that rides the int8 chain) → add a row to `tools/precision_overrides.tsv`. This is the ONE
  doc-drift direction no gate can catch: the capability matrix classifies by the dtype token in a
  symbol name, so a borrowed implementation is invisible to it and the table silently UNDER-reports
  the borrowing dtype. Nothing detects it, because there is no ground truth to diff against — the
  only defence is this line. (The reverse direction IS gated: `make check-registry` check 8 fails if
  a row cites a symbol that no longer exists.)

This keeps the docs reflecting reality so the next agent doesn't reverse-engineer what changed.

### Op capability registry — consult BEFORE touching any op

[`OPS_REGISTRY.md`](OPS_REGISTRY.md) is the **index of truth** for which NPU ops/chains/handlers
exist and what state each is in (`PROVEN` / `PARTIAL` / `DEAD` / `WIP`). **Read it before you
reverse-engineer, re-fix, or build on any op** — most "novel" NPU work has already been tried, and
the registry says whether it worked, which probe proves it, and what to use instead. It exists
specifically to stop the recurring pattern of re-deriving solved (or known-dead) paths.

Rules that keep it from rotting into another stale doc:
- **Status is probe-anchored, and it's enforced at build time.** Every `PROVEN`/`PARTIAL`/`DEAD`
  row must name a probe (or explicitly say `(no ... probe)`), and every probe/op it cites must
  exist. `make check-registry` (run automatically by `make all`, no NPU needed —
  `tools/check_registry.sh`) **fails the build** otherwise. A status with no probe is not a
  red flag you have to notice while reading — it's a compile error.
- **Status changes only after a probe re-run**, never from a code reading or hunch.
- **Touching an op = updating its row in the SAME commit.** If you change an op's behavior, re-run
  its probe and edit the registry row alongside the code (this is part of §"Documentation review").
- The `*_WIP.md` docs are backstory/detail; the registry is the index. When they disagree, a probe
  re-run settles it (see the registry's CONTRADICTIONS section).

### WIP recovery doc for long debugging sessions

During any multi-step debugging or hardware-RE session (anything that spans many board runs or
risks getting disconnected), maintain a single WIP recovery document (e.g. `CHAIN_DEBUG_WIP.md` in
the repo root) and **update it at least every 15 minutes of active work**, plus whenever a finding
materially changes the picture. It must always capture: the current hypothesis, what's been proven
or ruled out, the exact state of the working tree (what's restored/edited/committed), the next
concrete steps, and any board-ops gotchas. The goal is that a fresh agent (or you, after a
disconnect) can resume from the document alone without re-deriving the investigation. These docs are
scratch — delete or fold them into the wiki once the work lands; don't commit stale ones.

---

## 3. Build, test, and the board

The library and examples **build and run on a Rockchip board** (they need `/dev/dri/cardN` and
the `rknpu` DRM driver). Build on the board (or cross-compile); from a workstation, sync the
source over and run there.

```sh
make                  # on the board: library + examples
make test             # on the board: build + run every example, asserting each exits 0
make test MODEL=/path/stories15M.bin    # also run the real-model llama2 test
# from a workstation: rsync -a . board:ork-driver/ && ssh board 'cd ork-driver && make test'
```

- The validated board is RK3588 (SBC IP: `10.3.0.236`).
- **THE BOARD DOES NOT RUN A STOCK KERNEL.** It runs `6.1.115-vendor-rk35xx-sram`, carrying two of our
  changes: the NPU SRAM enabler (dts + `CONFIG_ROCKCHIP_RKNPU_SRAM=y`) and a `rknpu_job.c` IOMMU
  domain-refcount `put` on job timeout. Both were undocumented for weeks and were therefore a hidden
  variable in every board measurement — a multi-day IOVA-leak investigation concluded "kernel-side, not
  ours" while running a patched driver. So: **record `uname -r` with any result that matters, and treat
  results from different kernels as incomparable.** The patches, rationale, the three kernels available in
  `/boot` for bisecting, and the revert procedure (`/boot/RESTORE_STOCK_KERNEL.sh`, two symlink flips) are
  on the wiki: *Kernel-Modifications*. **The shipped vendor driver's OWN contracts and defects (IOMMU
  domains, dma-buf imports, reset paths, the domain refcount, the ms/us timeout bug) are catalogued on the
  wiki page *Vendor-Kernel-Behaviour* — read it before touching anything IOMMU-, submit- or reset-related.
  Note especially: "job committed, PC task counter 0x0, no interrupt, no error" has NEVER once been silicon
  in this project — but as of 2026-09-03 it is no longer always a page-table mismatch either. The
  counterexample is a NONBLOCK submit whose completion interrupt was DROPPED because the NPU was powered
  down under the still-running job (kernel #patch73), with the domains matching. Check job accounting
  FIRST — `cat /sys/kernel/debug/rknpu/counters` (`unpow`/`blocked_slow`) and `dmesg | grep "NOT
  dispatching"` — then the domain, then the hardware.** Kernel diffs live THERE, not here — this repo is a userspace library
  over the stock DRM uABI and should stay one. **The kernel work itself lives in
  [`oRKLLM/rk3588-kernel`](https://github.com/oRKLLM/rk3588-kernel)** (a fork of the VENDOR tree
  `rockchip-linux/kernel`, not the armbian copy): branch `rknpu-iommu-rework` is proposed upstream as
  **[PR #390](https://github.com/rockchip-linux/kernel/pull/390)**, which closes
  [issue #387](https://github.com/rockchip-linux/kernel/issues/387). PR #388 (the five refcount fixes
  alone) was CLOSED — measured insufficient for multi-domain; see the Experiment Log 2026-08-28.
- **RUN `sudo tools/util/npu_guard.sh -- <cmd>` FOR ANYTHING THAT TOUCHES THE NPU.** More than one agent
  may share this board. The guard checks who holds the render node (`/proc/*/fd` → `renderD12[89]`), that
  NPU utilisation is idle, and that no fault storm is in progress; with `--` it holds an flock for the
  lifetime of the command so check-and-launch is not a race. `ORK_NPU_LOCK_WAIT=<seconds>` waits instead
  of failing.
  - **A name-based `pgrep` is NOT sufficient**, and this is not hypothetical: a session polling
    `pgrep -x make` missed a concurrent suite because its recipe runs as `/bin/sh -c fail=0; for t in ...`,
    read the board as free, and overlapped — three `RKNPU: switch iommu domain time out` faults, plus an
    orphaned `orkd` holding an IOMMU domain that blocked the other run until it was SIGTERMed. Check the
    device, not the process name.
  - The guard is read-only (it reads `/proc`, sysfs and `dmesg`), so it is always safe to run, including
    while someone else's job is in flight.
- The NPU is **single-stream**: `make test` runs the examples serially; a wedged submit can stall the next — keep that in mind when adding tests.
  - **SBC Hard Wedge**: If the board becomes completely/hard wedged (unresponsive over SSH), use the Home Assistant MCP tools to power cycle the **"Rock 5B Plug"** smart plug. Specifically, call the `mcp_context-forge_home-assistant-hassturnoff(name="Rock 5B Plug")` tool to turn off the smart plug, followed by `mcp_context-forge_home-assistant-hassturnon(name="Rock 5B Plug")` to turn it back on.
    - **If a plug power-cycle does NOT bring it back** (plug reports `on`, but no ping/SSH after several minutes): the SPI bootloader has likely been corrupted (a hard power-cut mid-NPU-submit, or the abrupt loss, can corrupt SPI — the board then won't boot even with power restored). Recovery requires PHYSICAL access: **reflash the SPI image, then reboot from the Belkin power supply.** This was sufficient on `10.3.0.236` — **the SSD did NOT need reseating** (rootfs was intact; only the SPI bootloader was gone). A remote plug cycle cannot fix this — hand off to a physically-present operator.
    - **Avoid the cause:** never `kill -9` an in-flight NPU submit (it can wedge the NPU/IOMMU and force this hard-reset path). Use `SIGTERM` and let the submit drain; use `timeout` on board NPU commands so they self-terminate cleanly.
  - **NPU Wedge**: If only the NPU is wedged (submission hang/timeout, but the SBC is still responsive via SSH), execute a graceful reboot command over SSH to restart the board: `ssh board 'sudo reboot'`.
- A `.bin` test model (e.g. `stories15M`) is needed only for the `llama2` example; it is
  git-ignored and `make test` skips that test gracefully when `MODEL` is absent.
- **`orkd` has a behavioural gate now.** `make test` ends with `orkd_probe mm`, which auto-spawns the
  daemon, packs and runs a matmul THROUGH it, and compares against an exact integer CPU reference. Before
  this the daemon had NO test at all, so any `orkd.c` change was verified only by "it compiles" — which is
  why refactoring it was deferred (see MODULARIZE_PLAN.md round 7). It runs LAST, because the daemon takes
  ownership of the NPU, and the suite SIGTERMs any surviving `orkd` afterwards (never `-9` — an abrupt kill
  mid-submit wedges the IOMMU and costs a power-cycle). That reap is NOT optional: `orkd` self-reaps ~6 s
  after its last client disconnects, but the suite reaches the reap step sooner, so every run would
  otherwise finish with a daemon still holding the NPU — which matters when the board is shared.
- **The examples ARE the tests** — each self-validates against a CPU reference (NPU output must
  match within fp16 tolerance; NaN/inf or dead output must fail). When you fix a hardware-behavior
  bug, add a shape/config case to the relevant example so `make test` would have caught it.
- **Static golden verification (don't recompute the CPU reference every run).** Test inputs are
  fixed-seed deterministic (e.g. `test_matmul`'s `rnd()` is a fixed-seed LCG), so the NPU output is a
  constant — a wide-K reference like `{256,18944,3584}` is ~17 B MACs the NPU does in ~0.5 s but a naive
  O(M·N·K) CPU loop takes minutes. So the pass-path asserts an **O(M·N) checksum of the NPU output
  against an embedded golden** (`fnv64` + a `GOLD[]` table); the CPU reference function is **kept** and
  runs ONLY to **regenerate a golden** (`sudo env ORK_REGEN=1 ./test_matmul` → paste the printed
  `REGEN GOLD[...]` values) or to **diagnose a mismatch** (`ORK_FULL_REF=1`). Leave the golden static
  until a *deliberate* output-changing edit, then regen. In use on `test_matmul`, `quant`, `test_sn3`,
  `model` (the reference-bound tests); took full `make test` from ~10-11 min to **~33 s**. NOTE: pass env
  through `sudo env VAR=…` — a bare `VAR=… sudo …` prefix is stripped by sudo. **Same trap one level up:**
  the `make test` targets run each test under `$(SUDO)`, and plain `sudo` drops `ORK_*`, so a knob set on a
  make line reached nothing. `SUDO ?= sudo -E` (commit `b83269e`) fixes it — before that, `ORK_DEBUG_RESET=1
  make test` printed **0** resets where the real count is **50**. On an older tree, invoke the binary
  directly (`sudo env KNOB=1 ./test_x`) instead of trusting a knobbed `make test`.
- **CI board-validation gate (`make check-attest`).** The tests need the NPU, which CI runners lack — so
  CI can't run them. Instead, `make test` on the SBC (on ALL PASS) writes `tests/sbc_attest.txt` =
  sha256 of the NPU-output-determining `.c` sources + the golden-bearing tests; CI `make check-attest`
  (no NPU — sha256 + grep) fails if the tree hash differs, catching a commit that changed NPU-relevant
  code **without** a board `make test`. So: after any such change, run `make test` on the SBC and
  **commit the refreshed `tests/sbc_attest.txt`** or CI (and the version bump — `version-bump.yml` runs
  `check-attest` before bumping) will fail. Workflows: `.github/workflows/sbc-attest.yml` + `version-bump.yml`.
- **Fast rebuilds.** CORE (`npu.c`) is compiled **once** into shared `-fPIC` objects (`$(COBJ)`) that the
  examples/tests link — an `npu.c` edit rebuilds it once, not ~30×. `CC` uses **ccache** when present
  (`NO_CCACHE=1` to disable). Full build ~20 s, incremental npu.c-edit rebuild ~9 s. `make -j` parallelizes.
- **No macOS Binary Transfers:** Never push or copy local macOS binaries (from arm64/x86_64 Mac workstation builds) to the target SBC. Only source files should be synchronized (e.g., via git or targeted rsync) and then built natively on the board.
- **Board workflow helpers live in [`tools/util/`](tools/util/README.md)** (Mac-side; not needed to build/use the library):
  - **`tools/util/board`** — an SSH wrapper that runs commands on the board (host alias `board`) without nested-quote hell: `board -c 'make test'`, `board script.sh` (run a local file on the board), or pipe a script via stdin (`board <<'EOF' … EOF`). Each mode `cd`s into `~/ork-driver` first. Symlink it onto PATH (`ln -s "$PWD/tools/util/board" ~/bin/board`) to call it as `board`. Authoring board-side scripts as files and piping them avoids the escaping pitfalls of raw `ssh board '...'`.
  - **`tools/util/sync_daemon.sh`** — a 2 s-cadence bidirectional polling `rsync` over a persistent SSH master: pushes Mac source-of-truth (`src/ include/ tools/ examples/ docs/ Makefile`, no `--delete`) → `board:ork-driver/` and pulls `board:~/ork-outbox/` → a local outbox. Launch it once in the background so per-edit hand-syncing (and its cwd/path pitfalls) stops; then just build/run on the board. Put board-side artifacts you want back (dumps, logs, captured regcmds) in `~/ork-outbox/`. Source-of-truth stays the Mac; the board is build+run only (consistent with "no binary transfers").

---

## 4. Architecture & repo layout

```
include/ork_npu.h      public API — a 49-line UMBRELLA (guard, base typedefs, version) that includes:
include/ork/           context dma weights run sdp probe dynamic seq chain bmm
                         sdp.h = the supported SDP surface; probe.h = RE probes with no production
                         caller (53% of the old header). Consumers include <ork_npu.h>, never a part.
src/npu.c              SCAFFOLD: the dtype-dispatching layer only — run()/run_multicore, the
                       heterogeneous op-sequence scheduler, bmm dispatch, async wrappers,
                       slice/stage/stream-pool glue, ork_npu_init
src/npu/internal.h     private ABI: ork_npu/ork_w/buf types, dtype predicates, env knobs, hot inlines
src/npu/core.h         the SUBSTRATE interface (see the header-placement rule below)
src/npu/core/          dtype-agnostic substrate: device buf submit sched domain mode prof (+ core.h)
                         dyn dyn_ctl colsplit — the NONBLOCK doorbell spine + column-split scheduler.
                         Moved here from i8/ (roadmap Tier 19): their contracts carry no dtype, so they are
                         substrate. ork_dyn_begin_mc DISPATCHES on dtype — int4 leaves for i4/chain.c at the
                         top and never reaches the body (a diagnostic added below that point runs on nothing).
src/npu/i8/            int8 — regcmd pack fold run chain queue         (+ i8.h)
                         dyn_seq                    — int8 heterogeneous op-sequence chain (spine is in core/)
                         probe probe_sdp probe_replay probe_prof probe_chain — RE probes by family
src/npu/f16/           fp16 — regcmd run perchan stream probe replay                (+ f16.h)
src/npu/i4/            int4 — quant pack run chain stream                           (+ i4.h)
src/npu/i16/           int16 — regcmd act chain probe                               (+ i16.h)
src/npu/sdp.c          shared activation curves + LUT machinery (i8/i16/f16 all use it)
src/npu/norm.c         RMSNorm / L2 / softmax / RoPE / FWHT — an op family, not a precision
src/npu/ssm.c          Mamba-2 / SSD scan
src/orkd.c             orkd DAEMON: dispatch loop, socket accept, ring service, main
src/orkd_handlers.c    its 19 per-opcode request handlers  (interface: src/orkd_internal.h)
src/orkd_client.c      client transport: connect/spawn, A-ring, domain calls
src/orkd_client_ops.c  client RPC op wrappers              (shared helpers: src/orkd_client_internal.h)
src/soc.{h,c}          runtime device-tree SoC detection + caps registry
src/soc/<chip>.c       one file per SoC: core count, CBUF budget, output-width cap, K-slice
src/ork_regs.h         named regcmd registers (setrn); src/rknpu_ioctl.h open DRM uABI
src/regcmd_*.h         captured regcmd templates (our RE; no proprietary content)
examples/              test_matmul · layer · decode · model · llama2 — the test suite (each
                       self-validates vs CPU; MHA/GQA/arbitrary-head_dim covered). `make test` runs them.
tools/re/               NPU reverse-engineering toolkit (capture→decode→templatize new ops/NPUs; README there)
tools/re/regcmd_capture.c LD_PRELOAD calibration-capture shim (for adding SoCs / new ops)
docs/ADDING_AN_SOC.md   how to add/validate a SoC (RE narrative + scratch live on the wiki)
Doxyfile / `make docs`  scoped API docs -> docs/api/html (gitignored); excludes the regcmd data headers
```

**npu.c was 15,313 lines; it is now the scaffold.** The split is by PRECISION (2026-08-19,
`refactor/modularize-precision`). Two rules keep it from re-merging:

- **Where does my code go?** If a symbol's name, its regcmd template, or its dtype argument is
  precision-tagged, it belongs in that precision's folder. If its contract has no dtype in it, it is
  substrate — `npu/core/`. If it is an op family used by several precisions (activations, norms, the
  SSM scan), it is a peer module at `npu/<family>.c`. Only dtype DISPATCH stays in npu.c.
### Naming convention — dtype FIRST (enforced by check 11)

```
ork_<dtype>_<family>_<verb>[_<mechanism>][_<modifier>…]
     i8/i4/     mm/npu/   run/pack/  chain/stream/  silu/out8/
     i4a8/nf4/  bmm/dyn/  load       fold/slice     grouped/import
     f16/i16    w
```

`ork_i8_*` is the int8 datapath's namespace and mirrors `src/npu/i8/`. C has no namespaces — the prefix
IS the namespace, as in `sqlite3_`/`png_`. Execution **mechanism** sits before the dtype only when it is
part of the verb (`run_chain`, `run_stream`); everything else that is not universal — `silu`, `out8`,
`grouped`, `import` — is its own trailing component.

Dtype-**agnostic** surfaces carry no tag and must not gain one: `ork_dma_*`, `ork_npu_init`, `ork_pc_*`,
`ork_dyn_begin`, `ork_w_dump`, `ork_stage_fill`.

**Uniform names only where the operation is uniform.** The precisions are NOT parallel: int4 has no SDP
ops because RK3588 is W8A8 *or* W4A4 symmetric, and int16 is the ACTIVATION tier, not a weight tier. Do
not invent `ork_i4_npu_silu` or `ork_i16_mm_pack` to fill the grid — the hardware forbids them. See the
capability matrix in README.md.

Exemptions are catalogued in `tools/naming_exempt.txt` as CONVERSION (names two dtypes; prefix is the
source, destination explicit as `_to_<dst>`), AGNOSTIC, or MULTI (one implementation serving several
precisions — the matrix's dagger cases).

Full old→new table and rationale: [`docs/NAMING_MIGRATION.md`](docs/NAMING_MIGRATION.md).

- **SIZE BUDGET (enforced).** `tools/size_budget.txt` caps every `src/**` file at 900 lines by
  default, checked by `make check-registry` (check 9). `src/npu.c` has a ratcheted exception set
  just above its current size so the scaffold can only shrink. Exceeding a budget means editing
  that file and stating a reason — deliberate and reviewable, unlike drift. Prefer splitting:
  measure the seam first, because the cost is a property of the call graph, not of the line count
  (`i8/dyn.c` split three ways for ZERO crossing symbols; `orkd.c`, nearly the same size, cost 28).
- **HEADER PLACEMENT.** `npu/<name>.h` is tree-wide (internal.h, core.h). `npu/<mod>/<mod>.h` is
  PRIVATE to that folder — only `npu/<mod>/` sources may include it, and if the scaffold needs one of
  a module's symbols the declaration goes in internal.h instead. Enforced: `make check-registry`
  check 6 fails the build on any cross-folder include of a private header.

Every internal symbol that crosses a translation unit carries an `ork_`/`orki_` prefix (`orki_` for
what would otherwise be a generic name in `libork_npu.so` — `run`, `pack`, `act`, `g_last_op`). Adding
an unprefixed one is visible in `nm -g` over `$(COBJ)`.

### Weight-DMA amortization (the single-core M-tile lever)

The single-core int8 matmul is **weight-DMA-bound, not compute- or row-bound**: every M-tile submit
re-streams the entire `K×N` weight from DRAM (~11 GB/s), and that cost is *independent of how many
rows the tile carries* — measured `µs/submit` is flat from mc=4→31 and scales ~linearly with N. So the
per-row throughput lever is **rows per weight-stream**: a bigger M-tile amortizes the one weight load
over more rows and cuts total weight re-reads (`M / mc` of them).

The M-tile size `mc` is therefore capped at the **largest value the hardware computes correctly**, which
is the `0x1040` K-reduction schedule limit `mg_max*64` — **not** `R-1` where `R = pow2_floor(2*cbuf/K)`.
The old `R-1` / "CBUF-resident rows" cap was a **disproven RE finding**: activations *stream* (they need
not be CBUF-resident), reg `0x1010` is only a perf hint (correctness is identical regardless of it), and
`mg_max*64` is the exact bit-exact ceiling — validated `mc=mg_max*64` is correct and `mc+1` miscomputes
at every K (704@K512, 320@K1024, 128@K2048, 64@K3584/4096). Raising the cap from `R-1` (~31) to
`mg_max*64` gave **~2.1× single-core / ~1.6× three-core at K=2048, ~1.5× at K=3584, bit-exact** (2026-06-30).

Consequences: `cbuf_elems` no longer sets the int8 M-tile size (it only feeds the neutral `0x1010` hint).
This fix is int8 full-K only — the wide-K (`K>4096`) K-slice path keeps its own cap. **For int8, when
touching any M-tile cap the bound is `mg_max*64`; never reintroduce an `R-1`/`pow2_floor` ceiling.**

**fp16 does NOT use `mg_max*64` — it has its own measured envelope, `orki_f16_mcap(K, sched)`
(`src/npu/f16/regcmd.c`), and the "latent large-tile M-scheduler bug" is now characterised (2026-08-20,
`tools/re/f16_mcap_probe.c`).** `0x1040` is a CBUF **bank split** (`DATA_BANK[3:0]`/`WEIGHT_BANK[7:4]`,
12 banks × 32 KB), walked one bank data-ward per 64-row group; the ceiling is where `WEIGHT_BANK` would
hit 0. So the fp16 envelope is banks × 32 KB / row-bytes: **`sched=0` → 1 bank → `M ≤ 16384/K`;
`sched=1` → 11 banks → `M ≤ 180224/K`** (measured 704@K256, 352@K512, 176@K1024, and 8@K2048 … 256@K64
for sched=0 — `ceiling+1` miscomputes at every one). **K=128 is capped at a measured 256**: it needs
half-bank steps (`slope = (int)7.5 = 7`), which the encoding cannot express, so its envelope is
*non-monotonic* (256 ok, 320 bad, 384 ok, …) and only the contiguous prefix is usable.

Two traps this cost us, both now fixed: the old fp16 caps used **32768 ELEMENTS** (the int8 byte budget),
which is **2× too loose** at 2 B/elem and silently miscomputed at non-pow2 K; and `4*R` overshot at K=128
(1024 vs 256), so `ork_f16_mm_run` miscomputed for any `M ∈ [257,1472]`. **Never derive an fp16 M-tile
from an int8 constant, and never bisect for an fp16 ceiling — the predicate is not monotonic. Scan
upward.** int8 and fp16 must NOT be unified on one rule: at identical `base`/`slope` fp16 reaches the
11-bank capacity and int8 stops at the truncated `mg_max*64` (unexplained; see the wiki entry).

### Multi-SoC: data, not branches

The regcmd ISA and DRM path are **shared** across the RK35xx family; only *parameters* differ
(core count, CBUF/SRAM budget, output-width cap). Therefore: **one `main` branch, runtime
detection, one caps file per chip in `src/soc/` — never a per-flavor branch or fork.** Adding or
validating a SoC is one file + a regression run — see [`docs/ADDING_AN_SOC.md`](docs/ADDING_AN_SOC.md).
RK3588 is hardware-validated; RK3576 shares the code path with inherited (untested) params and
`ork_npu_init` warns until `validated=1`.

### Open-source reference: the mainline `accel/rocket` driver (consult when stuck on HW semantics)

The RK3588 NPU is **NVDLA-derived**, and there is now a clean, mainline, **open-source** kernel driver for
this exact hardware: **`drivers/accel/rocket`** (Tomeu Vizoso) + its userspace in Mesa3D, documented at
[docs.kernel.org/accel/rocket](https://docs.kernel.org/accel/rocket/). **When stuck on a register meaning, a
submit/interrupt/DMA behavior, or "why does the hardware do X" — consult it FIRST**, before guessing on-board
(each wrong guess is a wedge-risk board run). It is the authoritative open cross-reference:

- **`rocket_registers.h`** — names our numeric registers (built from the RK3588 TRM ch.36 + NVDLA). Our DPU
  output-stage `0x40xx`, the PC block `0x00xx`, CNA `0x10xx`, CDMA `0x50xx` all map to named macros there.
- **`rocket_job.c`** — the real job-submission ABI: the NPU runs **one task per PC program** (`TASK_CON` /
  `OPERATION_ENABLE`), and the **kernel IRQ handler re-arms the next task** — sequencing is kernel-driven via
  completion interrupts (`PC_INTERRUPT_*_DPU_0/DPU_1`), NOT hardware-chained through a concatenated regcmd
  buffer. DMA faults surface as `PC_INTERRUPT_RAW_STATUS_DMA_READ_ERROR`. (This is how the multi-task
  softmax-replay hang was diagnosed — see `FWD_SOFTMAX_RE_WIP.md`.)
- **NVDLA docs** (nvdla.org) — the fixed pipeline (Conv/MAC → SDP → PDP → CDP), fused vs independent mode,
  and the fixed-function unit limits (e.g. CDP LRN reduction window n≤9).

Cross-referencing these three turned numeric RE into named, understood behavior repeatedly; make it the
default move when a hardware question blocks progress. Findings from it belong on the wiki
[regcmd ISA Reference](https://github.com/oRKLLM/ork-driver/wiki) / `NPU-Quirks`.

**Multi-task hardware chaining (one submit, `task_number>1`).** The kernel programs the PC from the FIRST task
only; the hardware then walks the chain per task via the in-regcmd descriptor `0101:0x0010` (next regcmd addr) +
`0101:0x0014` (next register-amount `=(n+3)/2`). Rules for chaining a heterogeneous op-graph (softmax, fused
attention, norms) — full write-up: wiki *Exp-2026-07-10 Forward Softmax and RKNPU Chaining*:
- recompute `0x0014` for the actual next task (zeroing it hangs at transition 1);
- task stride is content-driven (slot ≥ each task's regcmd), NOT fixed alignment;
- replay a contiguous single-buffer image with a single-delta rebase (some ops write INTO the regcmd region);
- **ping-pong OFF (`submit flags = RKNPU_JOB_PC = 0x1`, not `0x5`) for any chain with a LUT-load** — ping-pong
  (`RKNPU_JOB_PINGPONG`, `1<<2`) swaps register banks the instant a task's *register config* is done, racing a
  LUT-load's SRAM-commit side effect (non-deterministic stall). `run_chain_i8` keeps `0x5` only because uniform
  int8-matmul tasks are register-config-only. See `NPU-Quirks` "Ping-pong races a chained task's side effect".

### Mode-transition layer (`ork_npu_enter`)

The NPU's regcmd datapath is **stateful**: a "mode" (precision/schedule) is tracked in software via
`c->last_dt` plus warm flags (`warmed`, `mwarm[]`) and buffer-size caches (`ccsz`, `mccsz[]`). Moving
between modes may need an `RKNPU_ACT_RESET`, a cold 2-pass re-warm, and/or a buffer realloc. That
policy used to be **copy-pasted inline into ~16 run/stream/chain/int4 entrypoints**, each drifted.
It is now owned by **one function**, `ork_npu_enter(c, to_marker, profile, chain)` in `src/npu/core/mode.c`, driven by
the **`XSPEC[]` policy table** (one row per historical transition site). Every run path calls it first;
the drift is now visible **as data** (e.g. the nothrash-keyed chain profile ignores `ORK_SSM_KEEPWARM`
where the matmul/stream profiles honor it). See the big comment above `XSPEC[]` for the exhaustive
`from→to` matrix and the modes `{ COLD(-1), F16(0), I8(1), I4(2), I8_CHAIN(3), I4_CHAIN(4),
I4_STREAM(5) }` plus the transient `SDP` (activation/ewmul).

**To CHANGE a transition's policy:** edit that profile's **one row** in `XSPEC[]`. A row is
`{ kwp, rst, wtg, wc, stg, sc, setdt }` — a keep-warm predicate selector (`KWP_*`), a reset condition
(`RC_*`), and warm/size *clear-target* (`TG_*`: scalar `warmed`/`ccsz`, per-core `mwarm[]`/`mccsz[]`,
or both) + *clear-condition* (`WC_*`) selectors, and whether it writes `last_dt` (`setdt`). No `if`
chains to hunt down.

**To ADD a new op/dtype transition:** (1) pick/define its `last_dt` marker; (2) add a `XP_*` profile to
the enum and a matching `XSPEC[]` row; (3) call `ork_npu_enter(c, marker, XP_*, chain)` at the site (it
returns 1 iff a real transition fired — use that to drive any caller-local warmup flag, as
`XP_I4_INCR`/`XP_I4_STREAM` do). If none of the existing `KWP_*`/`RC_*`/`WC_*` selectors express the
behavior, add a selector value and its `case` in `ork_npu_enter` (each `case` is a literal predicate).

**The `chain` arg (`ork_chain_kind`)** names the chaining mechanism in effect — `OCK_NONE` (plain
`run`/`run_multicore`/int4 batch), `OCK_SW` (`run_stream_*` round-robin), `OCK_HW` (`run_chain_i8` PC-chain),
`OCK_FUSED` (`run_chain_i8_ffn` static regcmd graph, which carries in-chain SDP/LUT). It is recorded as
transition state (`c->last_chain`) so the policy can branch on the mechanism for the *few* handoffs where
it genuinely matters. **For the common case it has NO bearing** on the precision-mode reset (the `XSPEC`
row is mechanism-agnostic), and that is fine — verified by `mode_probe` + `test_ssd_chunk_npu` that no
entry-transition currently needs it (`OCK_FUSED`'s specialness — ping-pong-off, LUT-commit — lives in
`run_chain_i8_impl`'s submit flags, not the entry reset). When you find a transition that does need
mechanism-specific handling, branch on `chain` at the documented hook in `ork_npu_enter`. Pass the
mechanism that actually applies at each call site; **don't** default everything to `OCK_NONE`.

**To TEST a transition change — MANDATORY, on the board (RK3588):**
- `make test` — the primary gate. It exercises every mode: `test_matmul`/`layer`/`decode`/`model`
  (fp16+int8), `i4`/`test_chain_i4`/`perplexity_i4` (int4), `test_stream_interleave` (stream),
  `test_ssd_chunk_npu` (int8↔fp16 SSM), the SDP ops (`test_ewmul_{i8,f16,i16}`/`test_silu`/`test_gelu`/
  `test_add`), and **`test_mode_transition`**. A *behavior-preserving* refactor must pass **byte-identical**.
- Run it once **per keep-warm knob** too: default, then `ORK_SSM_KEEPWARM=0` and `ORK_MIXED_NOTHRASH=1`
  (the profiles differ only at non-default knobs). This needs `SUDO = sudo -E` (see §3) — with plain `sudo`
  the knob never reaches the test and the run silently re-validates the DEFAULT config. Sanity-check that
  propagation works before trusting a knobbed run: `ORK_TRACE=1 make test-only T=test_bmm` must print
  `[ork-trace]` lines.
- `make mode_probe && sudo env ORK_MM_TIMEOUT=2500 timeout 300 ./mode_probe` — the op→op transition
  matrix; confirm which pairs reset/wedge is unchanged. Full sweep = 56 pairs in ~48 s, 0 WEDGE.
  (History: this was a board-killer until 2026-08-22 — the int8 LUT path left the shared `c->task`
  descriptor poisoned, so the next single-core matmul stalled 2500 ms and reset-stormed into a kernel
  Oops. Fixed by the save/restore in `ork_i8_npu_probe_silu_std`; wiki *Exp-2026-08-22 mode_probe
  Kernel Oops*. If a sweep ever starts hanging again, suspect a new op that fails to restore `c->task`.)
- `ORK_DEBUG_RESET=1` — logs every `ACT_RESET` with a counter + caller; diff the count/sites before vs
  after (the "28→1" keep-warm behavior must be intact).

**Phase status.** The consolidation landed **behavior-preserving** (Phase 1). Sites that do **not** map
cleanly (drifted or specially-motivated) are **left as-is and catalogued in the wiki "Exp-2026-07-14 Mode-Transition Layer"** for a
Phase-2 1-by-1 evaluation — that is where behavior changes (e.g. flipping `XP_SDP.setdt=1` to fix the
fp16-mm→SDP→fp16-mm wedge, or making the chain profiles honor `ORK_SSM_KEEPWARM`) happen, each a
one-row edit re-validated by `make test` + bench. **Do not conflate** the precision-mode reset (which
the layer owns) with the `c->task` LUT-descriptor axis (`Exp-2026-07-12`): `ACT_RESET` does not fix the
latter, and it stays an op responsibility.

---

## 5. Constraints & scope

- **No Rockchip proprietary code or binaries** in this repo. `rknpu_ioctl.h` is the open kernel
  DRM uABI. The proprietary `librknnrt.so` is needed only to *run* `tools/regcmd_capture.c` for
  SoC calibration — never to build or use the library.
- This is **not a kernel driver** — it's a userspace library over the existing `rknpu` DRM
  driver. Be precise about that in docs and messages.
- Independent, community project — **not affiliated with or endorsed by Rockchip**. "Rockchip",
  "RK3576", "RK3588", "RKNN" are trademarks of Rockchip.
- Both **fp16** (`ork_f16_mm_pack`/`run`, fp16 A·B → fp32 C) and **int8/w8a8** (`ork_i8_mm_pack`/
  `run_i8`, int8 A·B → int32 C) are supported. Keep the public API stable when extending.
  Note: mixing both dtypes on one context works, but switching modes triggers a one-time NPU
  re-warm (the regcmd mode is stateful) — see `run()` in `src/npu.c`.

---

## 6. Benchmarking & performance (MANDATORY procedure)

The open stack is `ork-driver → ggml-ork → llama.cpp` (the [`oRKLLM/llama.cpp-rockchip`](https://github.com/oRKLLM/llama.cpp-rockchip) fork vendors this repo as a submodule and adds the `ggml-ork` backend). Full history + every experiment: the **[Optimization Roadmap](https://github.com/oRKLLM/ork-driver/wiki/Optimization-Roadmap)**, **[Experiment Log](https://github.com/oRKLLM/ork-driver/wiki/Experiment-Log)**, and **[Benchmark Standards](https://github.com/oRKLLM/ork-driver/wiki/Benchmark-Standards-and-Methodology)** wiki pages.

### Before stating ANY performance number — run the standardized benchmark
1. **Same model on both runtimes.** ggml-ork: a GGUF (Q8_0). Closed baseline: the matching `.rkllm` for librkllmrt (full-model) or the RKNN matmul API (per-matmul). Never compare across model sizes or paths.
2. **Use `llama-bench`, NOT short `llama-cli` prompts** (warmup + repetition; short prompts + the lazy weight-pack produce false prefill numbers). For matmul-level work use `tools/rknn_vs_ork` / `tools/mc_prof`.
3. **Verify governors at max BEFORE timing:** DDR `/sys/class/devfreq/dmc/governor`=performance@2112MHz, CPU A76 `cpu4..7` performance@2.4GHz. A parked DDR governor ~halves decode.
4. **Run with `-t <big-core-count>` (4 on RK3588), not `-t 8`** — the little A55 cores drag the threadpool barrier and contend with the NPU-driver threads. This alone is the difference between losing to and beating librkllmrt.
5. Report BOTH prefill and decode, ≥2 reps, and state model / path / warm-vs-cold.
6. **Validate any matmul-path change against the CPU reference** (`make quant`, RMS unchanged) — `mc_prof`/`rknn_vs_ork` use dummy data and do NOT check correctness.

Reference (Qwen3-1.7B-w8a8, RK3588 board `10.3.0.236`, `-t 4`, warm): librkllmrt **184 prefill / 11.71 decode** tok/s; open stack **178 (97%) / 12.8 (109%, beats it)**.

### Managing Baseline Thresholds
When committing new performance optimizations to `ork-driver` or `ggml-ork`, you MUST run the integration benchmark script (`tools/bench_two_turn.sh` via `make bench-llama`) on the Rockchip SBC to ensure there is no significant degradation (e.g., M-scheduling bugs or KV-cache penalties). 
If your optimizations successfully increase the performance baselines above the current documented numbers, you must explicitly update the threshold constants in the test scripts (e.g., `BASELINE_DECODE` in `tools/bench_two_turn.sh`) to the new, higher values before merging. Do not allow baselines to stagnate or decay.
### MoE models: the profile is AUTOMATIC (no knobs) — the model type decides

A routed-MoE model (any `GGML_OP_MUL_MAT_ID`, detected at load-time graph planning) auto-selects the
measured-optimal scheme on RK3588; **nothing needs to be set**:

- orkpack tier: expert/ffn tensors → int4 with the **NF4 codebook** (fixed by model type, so one model =
  one pack scheme whatever the GGUF's own precision); attn/dense stay int8.
- **prefill (M>1)**: experts on the CPU via the batched 4×4 NF4 NEON GEMM; attn/dense int8 on the NPU.
- **decode (M==1)**: **all-CPU** (the NPU per-submit floor loses at M=1) — the fused M=1 gemv fast-path
  (one activation quant per token, direct-to-dst, persistent OpenMP pool, one-cache-line PRFM prefetch).
- The `.orkpack` path is **derived from the model** (`<model-dir>/<model-basename>.orkpack`) and BUILT once
  if absent. `ORK_ORKPACK_PATH` is a **development override only**; `ORK_PERSIST` is retired and aborts.

Measured (qwen3.6-35B-A3B, `-t4`, governors=performance, 1024-tok PPL window, board `.236`):

| | prefill t/s | decode t/s | PPL |
|---|---|---|---|
| native ggml (Q4_K experts) | 23.3 | 7.5 | 10.21 |
| **ork auto-profile** | **36.9 (1.59×)** | 6.3–6.6 (0.85×) | 10.77 (+5.5%) |

So it is a **trade**: +5.5% perplexity and ~15% slower decode for 1.59× prefill. Prefill-heavy workloads
(long prompts / RAG / batch) win; decode-dominated chat may prefer `ORK_MOE_AUTO=0` (native decode).

**MODEL RECIPE — the single biggest decode lever is a FILE property, not a knob.** Attention precision in
the GGUF trades prefill for decode, and below 5 bits the NPU declines the source entirely (`sbits<5.0`),
forfeiting the prefill path:

    llama-quantize --tensor-type attn=q8_0 --tensor-type shexp=q8_0 <f16.gguf> <out.gguf> Q4_K_M

| attn precision in the GGUF | prefill | decode |
|---|---|---|
| f16 (16b) | 38.4 | 5.26 |
| **q8_0 (8b) — balanced optimum** | **36.9** | **6.59** |
| Q4_K (4.5b — NPU declines it) | 26.8 | 7.42 |

Expert precision in the GGUF is irrelevant (experts come from the pack; re-quantizing NF4 from an already
Q4_K gguf costs only +0.6% PPL, so the zero-config derived pack needs no f16 source).

### Env knobs (set on the `llama-bench`/`mc_prof`/`quant` command)
| var | effect |
|---|---|
| `ORK_FUSE=1` | QKV/gate-up fusion (off — measured neutral) |
| `ORK_NO_AFFINITY=1` | don't pin NPU-driver threads to big cores (default: pin) |
| `ORK_ZC_OUT=1` | output zero-copy (off by default — **correct** at the matmul level since the DMA cache-coherency fix `3fad74a` (+ Sn>1 `033a45b`), but ~0 end-to-end gain, so opt-in; **not safe under concurrent multi-core** — the coherency bsyncs aren't serialized across cores) |
| `ORK_PROFILE=1` | per-section timing (quant / NPU run / dequant; decode vs prefill; run_multicore phases) — printed by ggml-ork on free |
| `ORK_QUANT=4` | int4 W4A4 instead of int8 (experimental, incoherent) |
| `ORK_DECODE_MC=1` | let M=1 (decode) matmuls split N-tiles across all 3 cores (default: single-core at M=1). +1.62× decode on the big FFN projections (7B-Q8_0, 0.92→1.49 t/s); off by default because small-N decode matmuls lose the multi-core barrier to the single-core dispatch floor. Residual gap to rkllm is the synchronous-execution wall, not core count |

### CPU/MoE probes (sub-second, no model load — the iteration tools)
- `make test_i4_gemm && sudo ./test_i4_gemm` — batched int4/NF4 GEMM vs the M=1 gemv: bit-exactness + the
  M>1 speedup (I4 2.3×, NF4 1.85×). In `make test`.
- `make test_nf4_decode && taskset -c 6 ./test_nf4_decode [NEXP] [K] [N]` — **DRAM-realistic M=1 decode**
  probe: NEXP distinct weights (default 64 = 32 MB ≫ 3 MB L3) walked once, so every K-step is a cold DRAM
  read like real MoE decode; reports µs/expert AND achieved GB/s. Use this (NOT `test_i4_gemm`, whose single
  0.5 MB weight stays L3-resident) for any memory/prefetch work. A/B prefetch at compile time:
  `gcc -O3 -march=armv8.2-a+dotprod -Iinclude -DORK_PRF_DIST=0 -o /tmp/d0 examples/test_nf4_decode.c -lm`.
  Measured: the M=1 loop is **latency-bound, not bandwidth-bound** (one A76 gets ~8 GB/s of the ~21.9 GB/s
  it can stream), which is why `PRFM` one cache line ahead (`ORK_PRF_DIST=64`) wins ~9-11%.
- `make test_moe_dispatch && sudo ./test_moe_dispatch` — does the ~18µs/program NPU dispatch tax amortize?
  (separate vs `run_chain_i4` vs `run_i4_experts` vs one big program.) It does: 3.4× decode / 3.67× prefill.

### Perplexity (quality) — use `ork_ppl`, NOT `llama-perplexity`
`sudo ./build/bin/ork_ppl <model.gguf> <text> [window=512] [ubatch=512]` (a cmake target in the fork build, alongside `ork_bench`; source is `tools/re/ork_ppl.cpp` here) — teacher-forced PPL through the
same backend/env path as `ork_bench`, one clock. **Pin `ubatch` (e.g. 512)**: `llama-perplexity` defaults to
`n_batch=2048, n_seq=4` → M=2048 → wide colsplit → `RKNPU_SUBMIT` timeouts + self-heal thrash that can wedge
the kernel NPU driver (recover with `sudo reboot`; re-pin governors afterwards, they reset).

### Diagnostic tools (board only; not in `all`/`test`)
- `make rknn_vs_ork RKNN_DIR=/tmp/rknn && sudo env LD_LIBRARY_PATH=/tmp/rknn ./rknn_vs_ork [iters] [a]` — per-matmul ork vs the closed RKNN matmul API (same int8 (M,K,N)); `a` arg = the AC-layout probe.
- `make mc_prof && sudo ./mc_prof [M] [K] [N] [iters]` — per-core copy/submit/accumulate breakdown; `ORK_TEST_DMA=1` puts A in a zero-copy DMA buffer.
- `make batch_probe && sudo ./batch_probe [ntask]` — multi-task-per-submit probe (it times out; the kernel rejects `task_number>1`).
- `sudo tools/bench_monitored.sh --label L -- <cmd>` — wraps any workload (e.g. `llama-bench`/`llama-completion`) and samples RK3588 resource use (RAM, RAM-bandwidth via DMC `devfreq/dmc/load`, NPU, GPU, CPU, swap — same sources as oRKLLM's `monitor.js`), printing per-resource **avg + peak**. Sampler pins to the little cores (`--sampler-cpus`, default `0-3`) so it doesn't perturb the workload. Makes a tok/s number attributable: it revealed decode is **latency/serialization-bound, NOT bandwidth-bound** (decode leaves NPU ~90% idle and RAM-BW at ~26% avg / 75% peak — nothing saturates). Isolate decode with `llama-bench -p 0 -n N` or a short-prompt/long-generation completion.

### Zero-copy DMA (`ork_dma_alloc`/`ork_dma_free`)
NPU-coherent CPU-mapped buffers; a matmul whose A/C live in one has the regcmd read/write it in place (no host memcpy). **Input zero-copy (A): validated correct, default on, −17% on the full-K prefill matmul.** **Output zero-copy (C): correct but off by default** — the original corruption was a DMA cache-coherency bug (CPU dirty/stale lines racing the NPU's writes), fixed in `3fad74a` (bsync clean-before + invalidate-after; Sn>1 strided in `033a45b`) and validated at the matmul level. It stays opt-in (`ORK_ZC_OUT`) because it measured **~0 end-to-end gain** (writeout saved is negligible vs prefill), not because it's wrong. Caveat: **unsafe under concurrent multi-core** (the per-core bsyncs don't serialize with the NPU writes — `run_chain_i8` multi-core falls back to single-core for DMA-buffer tasks). Realizing input zero-copy end-to-end still needs a ggml-ork DMA buffer type (the open Stage-2 item).
