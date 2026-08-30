/* npu/f16/pack.c — fp16 weight LOAD/IMPORT: the read-back half of ork_f16_mm_pack.
 *
 * fp16 could be packed onto the NPU and dumped (ork_w_dump is dtype-agnostic) but never restored, so
 * an fp16 weight could not be persisted in a .orkpack and reloaded. int8 and int4 both have the
 * load/load_import pair; this adds the fp16 one, same signatures and same semantics.
 *
 * The blob is exactly what ork_w_dump emits: the Sk*Sn resident Bb tiles concatenated in the order
 * ork_w_dump walks them, Bb[ns*Sk+ks], each padded to its page-rounded allocation size. The geometry
 * below therefore has to match orki_pack's fp16 arm exactly -- KS = soc->ks (NOT the int8 K-slice),
 * N%16 (not 32), and 2 bytes per element -- or the round-trip silently mis-slices.
 *
 * Two fp16 specifics worth knowing before touching this:
 *   - fp16 carries no per-channel bscale, so there is nothing to restore; w->bscale stays NULL. A
 *     weight produced by ork_i8_mm_inflate_to_f16 has its scale already folded into the tile bytes,
 *     so it round-trips through here identically and also wants bscale NULL.
 *   - no Bf is rebuilt. int8's load reconstructs the full-K Bf layout, but fp16's equivalent is the
 *     CONTIG copy (w->Bbc / w->Bbc_ns), which is built lazily on the first colsplit and gated by its
 *     own *_valid latch. Leaving those zeroed is what makes a loaded weight rebuild it exactly as a
 *     freshly packed one does.
 *
 * Part of the f16 datapath; shared declarations in npu/f16/f16.h.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../internal.h"
#include "../core.h"

/* Geometry of an fp16 .orkpack blob, mirroring orki_pack's fp16 arm. Returns 0 and fills the slice
 * counts + the exact byte count the blob must have, or <0 if the shape is not fp16-packable. */
static int f16_blob_geom(ork_npu *c, int K, int N, int *KS_o, int *NMAX_o,
			 int *Sk_o, int *Sn_o, size_t *need_o)
{
	int KS, NMAX, Sk, Sn;
	size_t need = 0;

	if (!c || !c->soc || K <= 0 || N <= 0) return -1;
	if (K % 32 || N % 16) return -1;              /* fp16 tile is [16][32]: N%16, not N%32 */

	KS = c->soc->ks; NMAX = c->soc->nmax;
	if (KS <= 0 || NMAX <= 0) return -1;
	Sk = (K + KS - 1) / KS; Sn = (N + NMAX - 1) / NMAX;

	for (int ns = 0; ns < Sn; ns++) {
		int n0 = ns * NMAX, Nc = (N - n0 < NMAX) ? (N - n0) : NMAX;
		for (int ks = 0; ks < Sk; ks++) {
			int k0 = ks * KS, Kp = (K - k0 < KS) ? (K - k0) : KS;
			need += orki_pgup((size_t)Kp * Nc * 2);
		}
	}
	*KS_o = KS; *NMAX_o = NMAX; *Sk_o = Sk; *Sn_o = Sn; *need_o = need;
	return 0;
}

static ork_w *f16_load_common(ork_npu *c, int K, int N, const void *blob, size_t n, int use_import)
{
	int KS, NMAX, Sk, Sn;
	size_t need, off = 0;
	ork_w *w;

	if (!blob) return NULL;
	if (c->fd < 0) return NULL;   /* no device: fp16 has no CPU-side untile twin (int8/int4 do) */
	if (f16_blob_geom(c, K, N, &KS, &NMAX, &Sk, &Sn, &need)) return NULL;
	if (n != need) return NULL;   /* exact match: a short blob would leave tiles filled with garbage */
	if (use_import && orki_dmaheap_open() < 0) return NULL;

	w = calloc(1, sizeof *w);
	if (!w) return NULL;
	w->K = K; w->N = N; w->Sk = Sk; w->Sn = Sn; w->dtype = DT_F16; w->owns = 1;
	w->domain = ork_dom(c->pack_domain);
	w->Bb = calloc((size_t)Sk * Sn, sizeof(struct buf));
	if (!w->Bb) { free(w); return NULL; }

	/* establish a non-0 domain before importing into it, exactly as the int8 import path does */
	if (use_import) ork_dom_prime(c, w->domain);

	for (int ns = 0; ns < Sn; ns++) {
		int n0 = ns * NMAX, Nc = (N - n0 < NMAX) ? (N - n0) : NMAX;
		for (int ks = 0; ks < Sk; ks++) {
			int k0 = ks * KS, Kp = (K - k0 < KS) ? (K - k0) : KS;
			size_t idx = (size_t)ns * Sk + ks;
			struct buf *b = &w->Bb[idx];

			*b = use_import ? orki_bimport(c->fd, (size_t)Kp * Nc * 2, w->domain)
					: orki_bcreate(c->fd, (size_t)Kp * Nc * 2, 0x403, w->domain);
			if (!b->cpu) {
				fprintf(stderr, "[ork] ERROR: %s failed for fp16 Bb[%zu] (size=%zu) in %s\n",
					use_import ? "bimport" : "bcreate", idx, (size_t)Kp * Nc * 2,
					use_import ? "ork_f16_mm_load_import" : "ork_f16_mm_load");
				for (size_t i = 0; i < idx; i++) orki_bdestroy(c->fd, &w->Bb[i]);
				free(w->Bb); free(w); return NULL;
			}
			/* imports: bracket the CPU fill -- rknpu MEM_SYNC does not cover foreign imports */
			if (use_import) orki_dmabuf_sync(b->heap_fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE);
			memcpy(b->cpu, (const char *)blob + off, b->size);
			off += b->size;
			if (use_import) orki_dmabuf_sync(b->heap_fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
			orki_bsync(c->fd, b, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
			orki_bsync(c->fd, b, RKNPU_MEM_SYNC_TO_DEVICE);
		}
	}
	/* w->bscale stays NULL, and Bbc/Bbc_ns stay unbuilt with their *_valid latches clear, so the
	 * first colsplit builds CONTIG from these tiles exactly as it would after a live pack. */
	return w;
}

/* Load pre-tiled fp16 weight bytes (ork_w_dump / .orkpack) into NPU-resident tiles. Same blob format
 * and round-trip as ork_f16_mm_pack + ork_w_dump. NULL on shape/size mismatch. */
ork_w *ork_f16_mm_load(ork_npu *c, int K, int N, const void *blob, size_t n)
{
	return f16_load_common(c, K, N, blob, n, 0);
}

/* As ork_f16_mm_load, but each resident tile is a dma-buf the NPU reads in place (dma-heap +
 * PRIME import) rather than a MEM_CREATE allocation. Saves the kernel page allocation, not the host
 * fill. NULL if import is unavailable -- the caller falls back to ork_f16_mm_load. */
ork_w *ork_f16_mm_load_import(ork_npu *c, int K, int N, const void *blob, size_t n)
{
	return f16_load_common(c, K, N, blob, n, 1);
}
