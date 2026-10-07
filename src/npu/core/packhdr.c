/* core/packhdr.c — the persisted-weight header: stamp on dump, refuse on load.
 *
 * WHY. ork_w_dump produced a HEADERLESS raw-tile stream, and ork/context.h says plainly why that was
 * a problem: "same-(K,N) blobs from an incompatible major have the SAME size, so ork-driver cannot
 * self-detect there" — the contract pushed version recording onto the caller's container and hoped.
 * So a pack built against one SoC's geometry, or against a superseded envelope rule, loaded silently
 * and ran wrong (or wedged). The int4 compact store already carried a header and already rejected a
 * mismatch; this gives every other persisted weight the same guarantee, in ork-driver rather than in
 * whichever consumer happened to remember.
 *
 * WHAT IS STAMPED, and why it is the resolved CAPS and not a bare "SDK-built" flag: the failure that
 * actually bites is a pack planned against DIFFERENT caps than the machine loading it. A boolean is
 * true in that case and tells you nothing. The capid mixes the SoC identity and the geometry that
 * decides tile layout and the M envelope, so a pack from another chip — or from a build whose
 * envelope rule has since moved — stops matching.
 *
 * WHY ONE PAGE. The blob is a sequence of PAGE-ALIGNED tiles (`need += orki_pgup(Kp*Nc)`), and
 * ork_i8_mm_load_import / ork_f16_mm_load_import make each tile "a page-aligned base+offset VIEW"
 * into an imported dma-buf chunk. A header of any other size would shift every tile off its page and
 * break zero-copy import. One page costs ~4 KB per weight — on a 23 GB pack with a few hundred
 * weights that is about a megabyte, and it keeps the import contract exactly as it was.
 *
 * NOT ON pack/repack. This gate is on the PERSIST path only. Training packs and re-packs weights
 * every step and never writes them to disk (measured: the training probes call ork_f16_mm_pack /
 * _repack and touch no dump or load at all), so putting validation on the pack path would charge the
 * hot loop for bytes that never reach a file.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../internal.h"
#include "ork_slice_f16.h"

/* Bump when the GEOMETRY or envelope rule changes meaning, so older packs stop matching even on the
 * same chip. This is separate from ORK_PACK_FORMAT_VERSION, which gates the byte LAYOUT. */
#define ORK_PACK_CAPS_RULE 1u

/* The SoC the capid is computed against. ork_w_dump has no ctx argument and a LIVE-packed weight
 * carries none (off_ctx is set only for offline packs), so fall back to the detected chip — a live
 * pack is by definition happening on the device. An OFFLINE pack does carry its context, which is
 * what lets an rk3588 pack be built on a workstation and still stamp rk3588. */
static const struct ork_soc *pack_soc(const ork_npu *ctx)
{
    if (ctx && ctx->soc) return ctx->soc;
    return ork_soc_detect();
}

unsigned orki_pack_capid(const ork_npu *ctx)
{
    const struct ork_soc *soc = pack_soc(ctx);
    unsigned h = 2166136261u;
    #define MIX(v) do { unsigned v_ = (unsigned)(v); \
        for (int b_ = 0; b_ < 4; b_++) { h ^= (v_ >> (b_*8)) & 0xffu; h *= 16777619u; } } while (0)
    MIX(ORK_PACK_CAPS_RULE);
    MIX(ORK_F16_MMAX_RULE);                    /* the fp16 M envelope rule (r103 narrow-N cap) */
    MIX(ork_pack_format_version());
    if (soc) {
        MIX(soc->ks); MIX(soc->nmax); MIX(soc->cbuf_elems); MIX(soc->cores);
        for (const char *p = soc->id; p && *p; p++) { h ^= (unsigned char)*p; h *= 16777619u; }
    }
    #undef MIX
    return h ? h : 1u;                         /* 0 is reserved for "unstamped" */
}

size_t ork_pack_hdr_bytes(void) { return ORK_PACK_HDR_BYTES; }

void orki_pack_stamp(const ork_npu *ctx, void *out, int dtype, int K, int N)
{
    if (!out) return;
    memset(out, 0, ORK_PACK_HDR_BYTES);        /* zero the pad: the blob must be reproducible */
    ork_pack_hdr *h = (ork_pack_hdr *)out;
    h->magic     = ORK_PACK_MAGIC;
    h->fmt       = ork_pack_format_version();
    h->capid     = orki_pack_capid(ctx);
    h->dtype     = (uint32_t)dtype;
    h->K         = K;
    h->N         = N;
    h->hdr_bytes = ORK_PACK_HDR_BYTES;
    { const struct ork_soc *soc = pack_soc(ctx);
      if (soc && soc->id) { strncpy(h->soc, soc->id, sizeof h->soc - 1); h->soc[sizeof h->soc - 1] = 0; } }
}

/* 0 = good. Negative codes are distinct on purpose: "this pack is from another chip" and "this pack
 * predates the header" need different advice, and a loader that collapses them to NULL sends the
 * caller hunting. The message is printed once per distinct cause, not per weight — a model load
 * touches hundreds of weights and a per-weight spew buries the one line that matters. */
int orki_pack_check(const ork_npu *ctx, const void *blob, size_t n, int dtype, int K, int N)
{
    static int said_unstamped = 0, said_caps = 0, said_fmt = 0;
    if (!blob || n < ORK_PACK_HDR_BYTES) return ORK_PACKERR_SHORT;
    const ork_pack_hdr *h = (const ork_pack_hdr *)blob;

    if (h->magic != ORK_PACK_MAGIC) {
        if (!said_unstamped++) fprintf(stderr,
            "[ork] refusing an UNSTAMPED persisted weight: this .orkpack predates the pack header "
            "(ork-driver %s, pack format %u). It was not built through the shape planner, so its tile "
            "geometry cannot be checked against this build. Delete the .orkpack and let it rebuild.\n",
            ORK_NPU_VERSION, ork_pack_format_version());
        return ORK_PACKERR_UNSTAMPED;
    }
    if (h->fmt != ork_pack_format_version()) {
        if (!said_fmt++) fprintf(stderr,
            "[ork] refusing a persisted weight from pack format %u (this build is %u): the on-disk "
            "tile layout changed. Delete the .orkpack and let it rebuild.\n",
            h->fmt, ork_pack_format_version());
        return ORK_PACKERR_FORMAT;
    }
    if (h->capid != orki_pack_capid(ctx)) {
        if (!said_caps++) fprintf(stderr,
            "[ork] refusing a persisted weight built against DIFFERENT caps (pack soc='%s' capid=%08x, "
            "this build soc='%s' capid=%08x): geometry or the M envelope differs, so the tiles would be "
            "read with the wrong rules. Rebuild the .orkpack on this machine.\n",
            h->soc, h->capid, pack_soc(ctx) && pack_soc(ctx)->id ? pack_soc(ctx)->id : "?", orki_pack_capid(ctx));
        return ORK_PACKERR_CAPS;
    }
    if (h->hdr_bytes != ORK_PACK_HDR_BYTES)            return ORK_PACKERR_FORMAT;
    if ((int)h->dtype != dtype || h->K != K || h->N != N) return ORK_PACKERR_SHAPE;
    return 0;
}
