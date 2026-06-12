# Adding (or validating) a Rockchip SoC

The RK35xx NPUs share one `rknpu` DRM driver and one regcmd ISA, so ork-driver supports them
with a single code path and a **per-SoC parameter file** — never a branch or fork. Adding a
chip is: write one `src/soc/<chip>.c`, register it, and validate with the regression suite.

## 1. Add the caps file

Create `src/soc/<chip>.c`:

```c
#include "../soc.h"
const struct ork_soc ork_soc_rk35xx = {
    .id="rk35xx",            /* substring of /proc/device-tree/compatible */
    .card="/dev/dri/card1",  /* DRM node (ORK_NPU_CARD env overrides at runtime) */
    .cores=2,                /* NPU core count */
    .cbuf_elems=32768,       /* fp16 feature CBUF budget: rows-per-M-tile = cbuf_elems/K */
    .nmax=8192,              /* max matmul output width (N) per submit */
    .ks=2048,                /* K-slice size (scheduler-fast contraction range) */
    .validated=0,            /* 0 until confirmed on real hardware (init prints a warning) */
};
```

Register it in `src/soc.c`: add `extern const struct ork_soc ork_soc_rk35xx;` and an entry in
`TABLE[]`. Add the source to `CORE` in the `Makefile`.

Start by **copying RK3588's values** — they're a correct *code path* for the whole family; only
the tuning numbers may differ.

## 2. What the parameters mean (and how to find them)

| Field | Meaning | How to determine |
| :--- | :--- | :--- |
| `cores` | NPU cores | datasheet (3588:3, 3576:2, 3562/3568:1) |
| `cbuf_elems` | on-chip feature cache budget in fp16 elements; the NPU does `cbuf_elems/K` rows per internal M-tile | capture from the proprietary runtime, or bisect with `test_matmul` until large-M matmuls validate |
| `nmax` | max output columns per submit (wider `N` is tiled) | bisect: `test_matmul` with growing `N` — the first `N` that fails minus headroom |
| `ks` | contraction slice where the closed-form M-scheduler holds | usually 2048; lower only if large-K validation fails |

These come from the chip's SRAM size and DPU width. If you have the proprietary `librknnrt`,
the fastest route is to capture its regcmd for a few matmul shapes and read the `0x1010`
(rows/tile) and `0x1040` (schedule) values directly; otherwise bisect empirically with
`test_matmul`.

## 3. Validate on hardware

```sh
BOARD=user@<chip-board> node test/regression.mjs
```

`test_matmul`, `layer`, `decode`, and `model` each validate NPU output against a CPU reference.
Tune `cbuf_elems` / `nmax` until all pass, then set `validated=1`. Run `llama2` (with a real
model) as the end-to-end check.

## 4. If the ISA itself diverges

Unlikely within the RK35xx family, but if a future chip needs different *register synthesis*
(not just numbers), add function pointers to `struct ork_soc` for the diverging step and select
per-SoC implementations — still one branch, one binary.
