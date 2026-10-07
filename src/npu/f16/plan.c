/* f16/plan.c — the public fp16 shape-planning surface.
 *
 * ork_f16_mm_pack accepts any K%32 / N%16 shape, but the hardware's correctness envelope is
 * narrower and depends on all three dims, so the library was able to accept a shape its own run
 * path could not execute (r103: K=1024 N=16 wedged the board above a 128-row M-chunk). The fix
 * capped the runtime; this exposes the same rule so a caller can ask FIRST.
 *
 * It is a thin wrapper on purpose. The rule lives in ONE place (orki_f16_mcap_n) and both the
 * runtime and this surface read it — a planner with its own copy of the envelope is a planner that
 * will disagree with the runtime after the next measurement, which is the failure mode the whole
 * narrow-N episode was.
 *
 * Offline-safe: it touches no device state, only soc caps, so it works on an ork_npu_init_offline()
 * context. Model design (FFN width, expert count, head dim) is then a workstation question.
 */
#include <stdlib.h>
#include <string.h>
#include "../internal.h"
#include "ork_slice_f16.h"

int ork_f16_mm_mmax(const ork_npu *ctx, int K, int N)
{
    if (!ctx || !ctx->soc) return -2;
    if (K <= 0 || N <= 0 || (K % 32) || (N % 16)) return -2;
    if (N > ctx->soc->nmax) return -2;
    return orki_f16_mcap_n(K, orki_f16_sched(K), N);
}

/* The callback shape ork_slice_f16_caps wants. ud is the context. */
static int plan_mmax(int Kslice, int Ntile, void *ud)
{
    return ork_f16_mm_mmax((const ork_npu *)ud, Kslice, Ntile);
}

/* Build the caps the planner and the pack stamp both use, so they cannot drift apart. */
void orki_f16_plan_caps(const ork_npu *ctx, ork_slice_f16_caps *out)
{
    *out = ork_slice_f16_caps_rk3588();
    if (ctx && ctx->soc) {
        if (ctx->soc->ks   > 0) out->kmax = ctx->soc->ks;      /* SoC K-slice, not a literal */
        if (ctx->soc->nmax > 0) out->nmax = ctx->soc->nmax;
    }
    out->mmax = plan_mmax;
    out->ud   = (void *)ctx;
}

unsigned orki_f16_plan_capid(const ork_npu *ctx)
{
    ork_slice_f16_caps c; orki_f16_plan_caps(ctx, &c);
    return ork_slice_f16_capid(&c, ctx && ctx->soc ? ctx->soc->id : 0);
}

int ork_f16_mm_plan(const ork_npu *ctx, int M, int K, int N, int *programs)
{
    if (programs) *programs = 0;
    if (!ctx || !ctx->soc || M <= 0) return -2;
    ork_slice_f16_caps c; orki_f16_plan_caps(ctx, &c);
    int n = ork_slice_f16_count(K, N, M, &c);
    if (n < 0) return n;
    if (programs) *programs = n;
    return 0;
}
