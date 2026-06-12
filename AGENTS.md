# ork-driver — Agent Instructions & Architecture

`ork-driver` is a clean-room **userspace matmul library for the Rockchip NPU**. It synthesizes
register-command (regcmd) programs and submits them to the in-tree `rknpu` DRM kernel driver
via ioctls on `/dev/dri/cardN` — no `librknnrt`, no kernel module. It is the open NPU runtime
spun out of [oRKLLM](https://github.com/oRKLLM/oRKLLM); the reverse-engineering record lives in
the [ork-driver wiki](https://github.com/oRKLLM/ork-driver/wiki).

See [`README.md`](README.md) for the user-facing overview and API.

---

## 1. What it is

- Public API (`include/ork_npu.h`): `ork_npu_init` → `ork_mm_pack(K,N,B)` → `ork_mm_run(M,A,C)`.
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

This keeps the docs reflecting reality so the next agent doesn't reverse-engineer what changed.

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

- The validated board is RK3588.
- The NPU is **single-stream**: `make test` runs the examples serially; a wedged submit can
  stall the next — keep that in mind when adding tests.
- A `.bin` test model (e.g. `stories15M`) is needed only for the `llama2` example; it is
  git-ignored and `make test` skips that test gracefully when `MODEL` is absent.
- **The examples ARE the tests** — each self-validates against a CPU reference (NPU output must
  match within fp16 tolerance; NaN/inf or dead output must fail). When you fix a hardware-behavior
  bug, add a shape/config case to the relevant example so `make test` would have caught it.

---

## 4. Architecture & repo layout

```
include/ork_npu.h      public API (init / pack / run / soc introspection)
src/npu.c              core: raw DRM submit, regcmd synthesis, K-split + N-tiling + M-scheduler
src/soc.{h,c}          runtime device-tree SoC detection + caps registry
src/soc/<chip>.c       one file per SoC: core count, CBUF budget, output-width cap, K-slice
src/rknpu_ioctl.h      open DRM uABI of the upstream rknpu kernel driver
src/regcmd_*.h         captured regcmd templates (our RE; no proprietary content)
examples/              test_matmul · layer · decode · model · llama2 — the test suite (each
                       self-validates vs CPU; MHA/GQA/arbitrary-head_dim covered). `make test` runs them.
tools/regcmd_capture.c LD_PRELOAD calibration-capture shim (for adding SoCs)
docs/ADDING_AN_SOC.md   how to add/validate a SoC (RE narrative + scratch live on the wiki)
```

### Multi-SoC: data, not branches

The regcmd ISA and DRM path are **shared** across the RK35xx family; only *parameters* differ
(core count, CBUF/SRAM budget, output-width cap). Therefore: **one `main` branch, runtime
detection, one caps file per chip in `src/soc/` — never a per-flavor branch or fork.** Adding or
validating a SoC is one file + a regression run — see [`docs/ADDING_AN_SOC.md`](docs/ADDING_AN_SOC.md).
RK3588 is hardware-validated; RK3576 shares the code path with inherited (untested) params and
`ork_npu_init` warns until `validated=1`.

---

## 5. Constraints & scope

- **No Rockchip proprietary code or binaries** in this repo. `rknpu_ioctl.h` is the open kernel
  DRM uABI. The proprietary `librknnrt.so` is needed only to *run* `tools/regcmd_capture.c` for
  SoC calibration — never to build or use the library.
- This is **not a kernel driver** — it's a userspace library over the existing `rknpu` DRM
  driver. Be precise about that in docs and messages.
- Independent, community project — **not affiliated with or endorsed by Rockchip**. "Rockchip",
  "RK3576", "RK3588", "RKNN" are trademarks of Rockchip.
- Both **fp16** (`ork_mm_pack`/`run`, fp16 A·B → fp32 C) and **int8/w8a8** (`ork_mm_pack_i8`/
  `run_i8`, int8 A·B → int32 C) are supported. Keep the public API stable when extending.
  Note: mixing both dtypes on one context works, but switching modes triggers a one-time NPU
  re-warm (the regcmd mode is stateful) — see `run()` in `src/npu.c`.
