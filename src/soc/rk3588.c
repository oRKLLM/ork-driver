/* RK3588 — validated on hardware (Radxa ROCK 5B). Calibration: CBUF feature budget (int8 rows/M-tile
 * = 2*cbuf_elems/K, pow2-floored; fp16 = cbuf_elems/K), output-width cap 8192, K-slice 2048, 3 cores.
 *
 * cbuf_elems = 57344 (raised from the original 32768, 2026-06-30). A no-spill bit-exact sweep showed
 * the CBUF holds >=64 rows at K=3584, so 32768 was conservative; 57344 gives int8 R=32 at K=3584
 * (2x the M-tile -> halves the single-core submit count for wide-K int8, e.g. ffn_down/up). This is
 * INT8-ONLY: the run paths cap fp16's effective cbuf back to 32768 (see npu.c "int8-only cbuf raise"),
 * because the fp16 multi-row M-scheduler is validated only up to the 32768-tile and silently
 * miscomputes larger fp16 M-tiles (a latent bug: fp16 K=2048 is bit-exact at mc<=8 but wrong at
 * mc>=9 — exposed when the raise grew the chunk; fixing the fp16 M-scheduler for big tiles is a
 * separate open item). int4 does not use cbuf (synth_i4 is single-row, captured regcmd) -> unaffected.
 * Validated bit-exact end-to-end via `make test` (int8 test_sn3/test_speed, fp16 layer/model, int4
 * i4/test_chain_i4 all pass). NOTE: bigger tiles are THROUGHPUT-NEUTRAL on this per-row-bound kernel
 * (they cut submit count, not per-matmul us); the single-core per-row gap vs rknn is the real open
 * issue (systolic activation-feed), which the M-tile-row count does not control. */
#include "../soc.h"
const struct ork_soc ork_soc_rk3588 = {
    .id="rk3588", .card="/dev/dri/card1", .cores=3,
    .cbuf_elems=57344, .nmax=8192, .ks=2048, .validated=1,
};
