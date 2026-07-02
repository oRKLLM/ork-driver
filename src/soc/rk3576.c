/* RK3576 — NOT yet hardware-validated. Same rknpu DRM driver + regcmd ISA as RK3588, so the
 * code path is shared; these params are INHERITED from RK3588 and must be re-captured on a
 * real RK3576 (smaller SRAM may change cbuf_elems/nmax). validated=0 makes ork_npu_init warn.
 * To confirm: run test/regression.mjs on an RK3576 board and tune until green, then set
 * validated=1. See docs/ADDING_AN_SOC.md. */
#include "../soc.h"
const struct ork_soc ork_soc_rk3576 = {
    .id="rk3576", .card="/dev/dri/card1", .cores=2,
    .cbuf_elems=32768, .nmax=8192, .ks=2048, .validated=0,
};
