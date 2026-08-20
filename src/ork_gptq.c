/* ork_gptq.c — GPTQ int4 weight quantization: DECLARED, NOT YET IMPLEMENTED.
 *
 * include/ork_npu.h declares ork_i4_gptq() as part of the int4 pack-time API. The real implementation
 * — error-compensated column-sequential int4 rounding (Frantar, Ashkboos, Hoefler, Alistarh 2022,
 * "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"), a hand-rolled
 * dense Cholesky + error-feedback sweep with no external deps — exists but is PARKED, unvalidated: it
 * has never been compared against AutoGPTQ on a fixed (W,H) golden (task #56), so shipping it would put
 * an unverified quantizer behind a public symbol.
 *
 * This stub keeps the declaration honest: the symbol links, and any caller gets a clean -ENOSYS instead
 * of a link error or silently wrong codes. Replace it wholesale when task #56's cross-check passes; the
 * parked implementation is recoverable from the stash noted in the commit that added this file.
 *
 * Contract (unchanged, see ork_npu.h): 0 ok, <0 on error. */
#include <errno.h>
#include <stdint.h>
#include "ork_npu.h"

int ork_i4_gptq(int K, int N, const float *W, float *H, int group,
                int8_t *codes, float *scales, float damp) {
    (void)K; (void)N; (void)W; (void)H; (void)group; (void)codes; (void)scales; (void)damp;
    return -ENOSYS;   /* not implemented — see the file header (task #56) */
}
