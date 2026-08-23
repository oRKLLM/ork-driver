/* soc.h — per-SoC capability table for the Rockchip NPU family.
 *
 * The regcmd ISA and DRM submit path are SHARED across RK35xx. What differs between chips
 * is data: NPU core count, the on-chip CBUF/SRAM budget (which sets the M-tiling rows), and
 * the matmul output-width cap. To support a new SoC you add one file here (soc/<chip>.c)
 * with its detected/captured values and register it in soc.c — no branches, no forks.
 * See docs/ADDING_AN_SOC.md.
 */
#ifndef ORK_SOC_H
#define ORK_SOC_H

struct ork_soc {
    const char *id;        /* device-tree match, e.g. "rk3588" */
    const char *card;      /* default DRM node; ORK_NPU_CARD env overrides */
    int cores;             /* NPU core count (3588:3, 3576:2, 3562/3568:1) */
    int cbuf_elems;        /* fp16 feature CBUF budget: rows-per-M-tile = cbuf_elems/K (int8: 2x) */
    int nmax;              /* max matmul output width (N) per submit; wider N is tiled */
    int ks;                /* K-slice size: the scheduler-fast contraction range */
    int validated;         /* 1 = these params confirmed on real hardware; 0 = inherited/untested */
};

/* Parse /proc/device-tree/compatible and return the matching caps, or NULL if unknown. */
const struct ork_soc *ork_soc_detect(void);
const struct ork_soc *ork_soc_by_id(const char *id);   /* by name — for offline (no-device) contexts */

#endif
