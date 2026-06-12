/* RK3588 — validated on hardware (Radxa ROCK 5B). Calibration: CBUF feature budget
 * 32768 fp16 elems (rows/M-tile = 32768/K), output-width cap 8192, K-slice 2048,
 * 3 NPU cores. These are the values the whole regcmd runtime was reverse-engineered
 * and validated against. */
#include "../soc.h"
const struct ork_soc ork_soc_rk3588 = {
    .id="rk3588", .card="/dev/dri/card1", .cores=3,
    .cbuf_elems=32768, .nmax=8192, .ks=2048, .validated=1,
};
