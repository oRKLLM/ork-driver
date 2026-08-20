/* npu/core/core.h — PRIVATE to src/npu/core/.
 *
 * The substrate's own internals: helpers the core modules share with each other but that nothing
 * outside core/ calls. The substrate's OUTWARD interface is npu/core.h, one level up.
 *
 * That split is the tree's header-placement rule, stated in npu/core.h and enforced by
 * check-registry check 6: a header INSIDE a folder is private to it; a header BESIDE a folder is that
 * subsystem's interface. Every module folder carries its own <mod>/<mod>.h for exactly this reason —
 * i8/i8.h, f16/f16.h, i4/i4.h, i16/i16.h, and this one.
 *
 * If something here starts being called from outside core/, move its declaration up to npu/core.h
 * rather than including this file — the gate will fail the build if you do the latter. */
#ifndef ORK_NPU_CORE_PRIVATE_H
#define ORK_NPU_CORE_PRIVATE_H
#include "npu/internal.h"
#include "npu/core.h"

struct buf orki_bimport_f(int fd,size_t size,int domain,uint32_t memflags);
struct buf *orki_warena_reserve(ork_npu *c,size_t need,size_t *base);
void ork_dma_bsync_to_device(ork_npu *c, void *ptr, size_t size);
void ork_dma_clean_to_device(ork_npu *c, void *ptr, size_t size);
size_t ork_iova_ceiling(void);
int ork_iova_reserve(int dom,size_t need);
int orki_dom_reserve(ork_npu *c, int need);
void orki_dump_submit(struct rknpu_submit *sub);
void orki_warn_if_governor_parked(void);
int orki_reimport_inplace(int fd, struct buf *b);
int ork_all_cores_mask(cpu_set_t *s);
void *ork_pool_worker(void *a);
void ork_pool_init(void);
void *orki_npu_pool_worker(void *vp);
extern struct ork_npu *orki_npu_ctx;

int orki_is_valid_dma_addr(ork_npu *c, uint32_t addr, const ork_w *w, const struct buf *extra, int extra_n);

#endif /* ORK_NPU_CORE_PRIVATE_H */
