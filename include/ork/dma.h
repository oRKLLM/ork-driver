/* ork/dma.h — Zero-copy DMA buffers and dma-buf import
 *
 * NPU-coherent CPU-mapped buffers, and importing a dma-buf someone else allocated (the
 * cross-process primitive behind the orkd daemon data plane).
 *
 * Part of the public API. Do NOT include this directly — include <ork_npu.h>, which is the
 * umbrella and the only supported entry point; these parts are a readability split of it
 * (ork_npu.h was 1519 lines) and their boundaries may move. Types live in ork_npu.h above
 * the includes, so this header is not self-contained by design. */
#ifndef ORK_DMA_H
#define ORK_DMA_H
/**
 * @brief Allocate an NPU-coherent, CPU-mapped buffer for zero-copy activations/outputs.
 *
 * Put the activation A and/or output C here and the matmul reads/writes them in place — no host
 * gather/writeout memcpy (the ~33% prefill residual vs the closed runtime). ork_f16_mm_run* detects
 * residency automatically: pass the returned pointer as A/C exactly like a malloc'd one.
 * @param size Bytes to allocate.
 * @return CPU-visible pointer, or NULL on failure / zero-copy table full (fall back to malloc).
 */
void        *ork_dma_alloc(ork_npu *ctx, size_t size);
/** @brief Like ork_dma_alloc but requests on-chip NPU SRAM residence (fails over to DRAM if none/full). */
void        *ork_dma_alloc_sram(ork_npu *ctx, size_t size);
/** @brief Like ork_dma_alloc but with explicit RKNPU_MEM_* flags (CACHEABLE=0x2, WRITE_COMBINE=0x4, TRY_ALLOC_SRAM=0x100). */
void        *ork_dma_alloc_flags(ork_npu *ctx, size_t size, unsigned flags);
/** @brief Free a buffer returned by ork_dma_alloc(). */
void         ork_dma_free (ork_npu *ctx, void *ptr);

/* Zero-copy IMPORT: allocate a dma-buf (from /dev/dma_heap/system), mmap it, and IOMMU-map the
 * EXISTING pages into the NPU — no second allocation, no copy. Caller fills the returned pointer with
 * the (pre-tiled) bytes, then calls ork_dma_import_sync once to flush them to the device; the NPU then
 * reads them in place across all submits (write-once-read-many weights). Returns the CPU pointer (pass
 * it as A/C to ork_f16_mm_run exactly like an ork_dma_alloc one — it is registered in the same zero-copy
 * table), or NULL on failure (dma-heap absent / IOVA full) so the caller can fall back to ork_dma_alloc.
 * Still 32-bit-IOVA-capped (does not escape the ~4 GiB window); it eliminates the COPY, not the cap. */
void        *ork_dma_import(ork_npu *ctx, size_t size);
void         ork_dma_import_sync(ork_npu *ctx, void *ptr, size_t size);  /* clean CPU writes -> device (size 0 = whole buffer) */
void         ork_dma_import_free(ork_npu *ctx, void *ptr);
/* Import an EXTERNAL dma-buf fd (e.g. received over SCM_RIGHTS from another process) into the NPU's IOMMU
 * domain and register it for zero-copy — the returned CPU pointer maps the shared buffer, and passing a ptr
 * into it as A/C to ork_f16_mm_run* makes the NPU read/write it IN PLACE (no copy). Takes ownership of the fd
 * (closed by ork_dma_free/ork_dma_import_free). NULL on failure. Enables the orkd daemon to run a matmul
 * directly against a client's shared buffer. Same 32-bit IOVA cap as ork_dma_import. */
void        *ork_dma_import_fd(ork_npu *ctx, int dmabuf_fd, size_t size);

#endif /* ORK_DMA_H */
