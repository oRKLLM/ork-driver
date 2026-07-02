/* RK3588 — validated on hardware (Radxa ROCK 5B). Calibration: output-width cap 8192, K-slice 2048,
 * 3 cores, CBUF feature budget cbuf_elems.
 *
 * WEIGHT-DMA AMORTIZATION (2026-06-30): single-core int8 matmul is WEIGHT-DMA-BOUND — each M-tile submit
 * re-streams the whole K*N weight from DRAM (us/submit is ~flat vs rows, ~linear in N). So bigger M-tiles
 * (more rows per weight-stream) are the per-row lever, NOT throughput-neutral: raising the M-tile cap from
 * the old (false) R-1 to the 0x1040 schedule max mg_max*64 gives ~2.1x single-core / ~1.6x 3-core at
 * K=2048, ~1.5x at K=3584, bit-exact. The cap is mg_max*64 (see npu.c) and does NOT depend on cbuf_elems.
 *
 * cbuf_elems = 57344: with the mg_max*64 M-tile in place, cbuf_elems no longer sets the int8 M-tile size
 * (it only feeds the 0x1010 row-count hint, which is MEASURED NEUTRAL) — so for int8 this value is now
 * largely vestigial. It still governs the fp16 path (M-tile = cbuf/K), where the run paths cap the
 * effective cbuf back to 32768: the fp16 multi-row M-scheduler is validated only to the 32768-tile and
 * silently miscomputes larger fp16 tiles (latent bug: fp16 K=2048 bit-exact at mc<=8, wrong at mc>=9 —
 * fixing it is a separate open item; the int8 weight-DMA fix above does not touch fp16). int4 does not
 * use cbuf (synth_i4 single-row captured regcmd) -> unaffected.
 * Validated bit-exact end-to-end via `make test` (int8 test_matmul/quant/test_sn3/test_speed, fp16
 * layer/model, int4 i4/test_chain_i4 all pass). */
#include "../soc.h"
const struct ork_soc ork_soc_rk3588 = {
    .id="rk3588", .card="/dev/dri/card1", .cores=3,
    .cbuf_elems=57344, .nmax=8192, .ks=2048, .validated=1,
};
