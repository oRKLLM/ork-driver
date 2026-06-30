/* RK3588 — validated on hardware (Radxa ROCK 5B). Calibration: CBUF feature budget
 * 32768 fp16 elems (int8 rows/M-tile = 2*cbuf_elems/K, pow2-floored), output-width cap 8192,
 * K-slice 2048, 3 NPU cores. The whole regcmd runtime was RE'd and validated against these.
 *
 * DO NOT raise cbuf_elems to get larger M-tiles (tried 57344 -> R=32 at K=3584, 2026-06-30): it is
 * (a) THROUGHPUT-NEUTRAL — this kernel is per-row-bound, so a bigger tile cuts submit count but not
 * per-matmul time (proven: 19->9 submits, identical us); and (b) it BREAKS mixed-K sequences — every
 * individual matmul stayed bit-exact multi-core, but the only change it made (K=2048 multi-core tile
 * 16->28 rows) perturbed the layer/model decoder sequences (Q/K/V/O@K=512 then down@K=2048 on one
 * context) into a silent miscompute (make test layer/model MISMATCH). NOT a tile-size/spill bug
 * (the K=512 mg=8 spill works at both values) — a cross-matmul state interaction, and not worth
 * chasing for zero throughput gain. The single-core per-row gap vs rknn is the real open issue
 * (systolic activation-feed), which the M-tile-row count does not control. */
#include "../soc.h"
const struct ork_soc ork_soc_rk3588 = {
    .id="rk3588", .card="/dev/dri/card1", .cores=3,
    .cbuf_elems=32768, .nmax=8192, .ks=2048, .validated=1,
};
