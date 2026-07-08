/* ork_npu.h — userspace matmul library for the Rockchip NPU (regcmd / raw DRM).
 *
 * This is NOT a kernel driver: it submits register-command programs to the in-tree
 * `rknpu` DRM driver via ioctls on /dev/dri/cardN — no librknnrt, no kernel module.
 * It reverse-engineers the regcmd ISA (shared across the RK35xx NPU family) and drives
 * fp16 matmul directly. SoC-specific parameters (CBUF budget, output-width cap, NPU core
 * count) are detected at runtime from the device tree — one binary supports every chip.
 *
 *   C[M,N] (fp32) = A[M,K] (fp16, row-major) x B[K,N] (fp16, row-major)
 *
 * Typical use (transformer): pack each weight matrix once (resident on the NPU), then
 * stream activations through ork_mm_run. Arbitrary M/K/N (K-split + N-tiling internally).
 */
#ifndef ORK_NPU_H
#define ORK_NPU_H
#include <stdint.h>   /* int8_t, int32_t, ... */
#include <stddef.h>   /* size_t — used by the DMA / dump / load / stream-pool APIs below */

typedef _Float16 ork_f16;
typedef struct ork_npu ork_npu;     /* device context (one per process) */
typedef struct ork_w   ork_w;       /* resident packed weights for one B[K,N] */

/**
 * @brief Library version, semver (e.g. "0.6.20"). Bump MINOR on backward-compatible API adds.
 *
 * Compile-time string; ork_npu_version() returns the same value at runtime, optionally suffixed with
 * a short git hash ("MAJOR.MINOR.PATCH+g<hash>") when built with -DORK_GIT_HASH (the Makefile injects
 * it where git is available).
 */
#define ORK_NPU_VERSION "0.6.50"
/** @brief Runtime library version. @return "MAJOR.MINOR.PATCH" or "MAJOR.MINOR.PATCH+g<hash>". */
const char  *ork_npu_version(void);

/**
 * @brief On-disk pack-format (.orkpack) compatibility token — the library's MAJOR version.
 *
 * A persisted weight (ork_w_dump / ork_w_dump_i4a8) is only binary-compatible with builds that share
 * this value. It is deliberately the MAJOR component of ORK_NPU_VERSION, because a change that alters
 * the persisted bytes' meaning is by definition NOT backward-compatible and therefore REQUIRES a major
 * bump — while MINOR/PATCH releases are backward-compatible and keep existing .orkpacks valid. What
 * mandates a major bump: a resident tile LAYOUT / geometry change (K-slice size, the SoC output-width
 * cap / N-tiling, the 32x32 block or Bb dump order) or a weight QUANT change (int8/int4 scale rule,
 * int4 nibble packing, NF4 codebook).
 *
 * ork-driver stamps this into the int4 blob header and rejects a mismatch on load (ork_mm_load_i4a8
 * returns NULL). The int8 dump (ork_w_dump) is a headerless raw-tile stream — same-(K,N) blobs from an
 * incompatible major have the SAME size, so ork-driver cannot self-detect there; the caller's on-disk
 * container (e.g. oRKLLM's .orkpack file) MUST record ork_pack_format_version() next to the bytes and
 * regenerate the cache at startup when it differs from this build's value.
 */
uint32_t     ork_pack_format_version(void);

/**
 * Un-pin the calling thread so it may run on ALL online CPU cores, overriding any inherited
 * affinity (e.g. a host that pinned its process/threads to the big cluster). Intended for the
 * CPU-bound weight dequant/quant/pack worker threads a caller spawns during a one-time pack:
 * there is no live inference to protect, so the pack should saturate every core. ork-driver's
 * own pack pool (ork_parallel_for) already does this; call this from external worker threads
 * (e.g. ggml-ork's std::thread dequant/quant loop) that ork-driver doesn't create.
 */
void         ork_unpin_current_thread(void);

/**
 * @brief NPU-availability gate: 1 if the NPU appears in use (any core loaded), 0 if idle.
 *
 * For a hybrid pack scheduler that routes each weight's tiling to the NPU (its pack path) only when
 * the device is idle, and to the CPU (tile from pagecache into DRAM, then zero-copy import) otherwise —
 * so a background .orkpack conversion never steals cycles from live inference in another process.
 * Reads the kernel's rolling per-core load counter; best-effort (returns 0 if unreadable). Cheap to
 * poll per weight.
 */
int          ork_npu_busy(ork_npu *ctx);

/**
 * @brief Open the NPU, detect the SoC from the device tree, and power it on.
 * @return Device context (one per process), or NULL on failure — no NPU present, or no permission to
 *         open /dev/dri/cardN (the process needs access to the DRM render node).
 */
ork_npu     *ork_npu_init(void);
/** @brief Power off the NPU and free a context obtained from ork_npu_init(). */
void         ork_npu_free(ork_npu *ctx);

/**
 * @brief Dump (and reset) the ORK_LOAD_PROF phase breakdown of the .orkpack import load path
 *        (dma-heap alloc / mmap / PRIME_FD / MEM_CREATE / Bb memcpy / Bf re-tile), to stderr.
 *
 * No-op unless ORK_LOAD_PROF is set. Called automatically at ork_npu_free(); also callable directly
 * to profile a load in isolation (e.g. a standalone convert+reload driver). Diagnostic, like ORK_PROFILE.
 */
void         ork_load_prof_dump(void);

/* SoC introspection */
/** @brief SoC name detected from the device tree, e.g. "rk3588", "rk3576". */
const char  *ork_npu_soc(const ork_npu *ctx);
/** @brief Number of NPU cores on this SoC (RK3588 = 3). */
int          ork_npu_cores(const ork_npu *ctx);
/** @brief 1 if this SoC's parameters are hardware-validated; 0 if inherited/untested (init warns). */
int          ork_npu_validated(const ork_npu *ctx);

/**
 * @brief Cap how many NPU cores the auto-tuner may use per matmul (policy hint).
 * @param n Max cores; n<=0 means all SoC cores (the default). Multi-core and the full-K int8 decode
 *          layout are still chosen automatically per matmul — this only bounds them (e.g. to reserve
 *          a core for another workload).
 */
void         ork_npu_set_core_budget(ork_npu *ctx, int n);

/* Per-weight NPU IOMMU domain placement. The rk_iommu 32-bit IOVA window (~4 GiB) is per
 * iommu_domain_id, so a model larger than 4 GiB can stay FULLY resident (no streaming, no per-token
 * map/unmap) by placing its weights across several domains. Call this before ork_mm_pack_i8 /
 * ork_mm_load_i8 (and the fp16/int4 variants): each weight packed/loaded afterward lands its resident
 * tiles in `domain` and records it; ork_mm_run* then submits that weight's matmuls against the same
 * domain automatically. domain<0 reverts to the process default (env ORK_IOMMU_DOMAIN, else 0). */
void         ork_npu_set_pack_domain(ork_npu *ctx, int domain);
int          ork_w_domain(const ork_w *w);   /* the IOMMU domain a packed weight resides in */

/**
 * @brief Allocate an NPU-coherent, CPU-mapped buffer for zero-copy activations/outputs.
 *
 * Put the activation A and/or output C here and the matmul reads/writes them in place — no host
 * gather/writeout memcpy (the ~33% prefill residual vs the closed runtime). ork_mm_run* detects
 * residency automatically: pass the returned pointer as A/C exactly like a malloc'd one.
 * @param size Bytes to allocate.
 * @return CPU-visible pointer, or NULL on failure / zero-copy table full (fall back to malloc).
 */
void        *ork_dma_alloc(ork_npu *ctx, size_t size);
/** @brief Free a buffer returned by ork_dma_alloc(). */
void         ork_dma_free (ork_npu *ctx, void *ptr);

/* Zero-copy IMPORT: allocate a dma-buf (from /dev/dma_heap/system), mmap it, and IOMMU-map the
 * EXISTING pages into the NPU — no second allocation, no copy. Caller fills the returned pointer with
 * the (pre-tiled) bytes, then calls ork_dma_import_sync once to flush them to the device; the NPU then
 * reads them in place across all submits (write-once-read-many weights). Returns the CPU pointer (pass
 * it as A/C to ork_mm_run exactly like an ork_dma_alloc one — it is registered in the same zero-copy
 * table), or NULL on failure (dma-heap absent / IOVA full) so the caller can fall back to ork_dma_alloc.
 * Still 32-bit-IOVA-capped (does not escape the ~4 GiB window); it eliminates the COPY, not the cap. */
void        *ork_dma_import(ork_npu *ctx, size_t size);
void         ork_dma_import_sync(ork_npu *ctx, void *ptr, size_t size);  /* clean CPU writes -> device (size 0 = whole buffer) */
void         ork_dma_import_free(ork_npu *ctx, void *ptr);

/* Load pre-tiled int8 weight bytes (ork_w_dump / .orkpack) into NPU-resident storage WITHOUT the
 * alloc+memcpy of ork_mm_load_i8: each tile is imported zero-copy (dma-buf the NPU reads in place).
 * Same blob format and round-trip as ork_mm_load_i8; returns NULL on shape/size mismatch or if import
 * is unavailable (caller falls back to ork_mm_load_i8). Weights are write-once: filled+synced here,
 * read-only across every submit. ork_mm_free / ork_w_free release the imports (MEM_DESTROY + close fd). */
ork_w       *ork_mm_load_i8_import(ork_npu *ctx, int K, int N, const void *blob, size_t n);

/**
 * @brief Pack + upload B[K,N] (row-major) into the NPU-resident tile layout; reuse across runs (fp16).
 * @param K Inner/contraction dim. Must be K%32==0.
 * @param N Output columns. Must be N%16==0.
 * @param B Row-major fp16 weights, K*N elements.
 * @return Resident weight handle for ork_mm_run(), or NULL on bad dims (K%32!=0 or N%16!=0).
 */
ork_w       *ork_mm_pack   (ork_npu *ctx, int K, int N, const ork_f16  *B);
/**
 * @brief Pack + upload B[K,N] (row-major) into the NPU-resident tile layout; reuse across runs (int8/w8a8).
 * @param K Inner/contraction dim. Must be K%32==0.
 * @param N Output columns. Must be N%32==0.
 * @param B Row-major int8 weights, K*N elements.
 * @return Resident weight handle for ork_mm_run_i8(), or NULL on bad dims (K%32!=0 or N%32!=0).
 */
ork_w       *ork_mm_pack_i8(ork_npu *ctx, int K, int N, const int8_t   *B);
/* re-tile int8 B into an existing same-shape ork_w (reuses its DMA; no alloc/free) — for pooling
 * reused weights (MoE experts) without churning/fragmenting the NPU IOMMU. 0 ok / -1 / -2 mismatch. */
int          ork_mm_repack_i8(ork_npu *ctx, ork_w *w, int K, int N, const int8_t *B);
/* NEON-fused pack/repack DIRECTLY from f32[N][K] (n-major): per-channel symmetric int8 quant + tile in
 * one cache-friendly pass (no transpose scratch). Writes per-channel bscale[N]. For dequantizing weight
 * sources (e.g. Q4_K MoE experts via to_float) without the slow strided f32->int8 transpose. */
ork_w       *ork_mm_pack_i8_f32(ork_npu *ctx, int K, int N, const float *f32, float *bscale_out);
int          ork_mm_repack_i8_f32(ork_npu *ctx, ork_w *w, int K, int N, const float *f32, float *bscale_out);
/* Fused dequant->int8 pack/repack: ork-driver calls `dequant(dctx, n, dst, K)` once per output channel
 * to materialize that channel's K f32 weights into a small REUSED scratch (cache-resident — avoids the
 * full f32[N][K] buffer and its DRAM round-trip), then NEON quant+tiles it. For packing a compressed
 * weight source (e.g. Q4_K MoE experts via ggml to_float) without the cache-thrashing full-f32 pass.
 * Same int8 result as pack_i8_f32 fed the equivalent f32. Writes per-channel bscale[N]. */
typedef void (*ork_dequant_row_fn)(void *dctx, int n, float *dst, int K);
ork_w       *ork_mm_pack_i8_dequant  (ork_npu *ctx, int K, int N, ork_dequant_row_fn dequant, void *dctx, float *bscale_out);
int          ork_mm_repack_i8_dequant(ork_npu *ctx, ork_w *w, int K, int N, ork_dequant_row_fn dequant, void *dctx, float *bscale_out);
/* "Effective w4a8": int4-PRECISION weights, int8 compute, int4 STORAGE. RK3588's NPU MACs are int8-only
 * (no native w4a8), so this quantizes f32[N][K] (n-major, as ggml's to_float produces) to int4 per output
 * channel (symmetric, scale = max|w_n|/7, range [-7,7]), keeps the compact nibble-packed form on the ork_w
 * (K*N/2 bytes — the memory win), NEON-expands int4->int8 [-7,7] in software, and tiles that into the int8
 * resident layout. Runs unchanged via ork_mm_run_i8 (returns a DT_I8 ork_w). Writes per-channel bscale[N]
 * (C_real[m][n] = aScale[m]*bscale[n]*Ci[m][n]). Round-to-nearest by default; set env ORK_SR for
 * stochastic rounding (debiases quantization — dot-product error grows ~sqrt(K) not O(K)). K%32, N%32.
 * Set env ORK_NF4 to use a fixed NF4 codebook (non-uniform levels, scale=max|w_n|/127, sets
 * quant_kind=ORK_QK_CODEBOOK_NF4) instead of the uniform int4 grid — better for Gaussian-ish weights. */
/* Int4 weight-store codebook kind (ork_w.quant_kind, set by pack_i4a8 / the int4 .orkpack form). 0 = UNIFORM:
 * the 4-bit value is a uniform int4 grid level; int4->int8 inflation is a sign-extend. 1 = CODEBOOK_NF4: the
 * 4-bit value indexes a 16-entry per-tensor LUT of non-uniform levels (NF4-style) — inflate via a NEON table
 * lookup (vqtbl); better accuracy for Gaussian-ish weights. Set via env ORK_NF4 on ork_mm_pack_i4a8. */
enum { ORK_QK_UNIFORM = 0, ORK_QK_CODEBOOK_NF4 = 1 };
ork_w       *ork_mm_pack_i4a8(ork_npu *ctx, int K, int N, const float *f32, float *bscale_out);
/* As ork_mm_pack_i4a8, but with optional importance-matrix (imatrix) weighted per-channel scale
 * selection. imatrix = optional per-INPUT-channel importance, length K (NULL = uniform / current
 * absmax behavior, byte-for-byte identical to ork_mm_pack_i4a8). When non-NULL, each output channel's
 * quant scale is chosen by searching a small clip-ratio grid r*absmax to minimize the importance-
 * weighted reconstruction error Sum_k imatrix[k]*(w[n][k] - dequant)^2 — clipping trades outlier error
 * for bulk resolution; imatrix decides which input columns' error matters. Applies to both the uniform
 * and NF4 (ORK_NF4) paths. O(grid*K) per channel (pack is one-time). K%32, N%32. */
ork_w       *ork_mm_pack_i4a8_im(ork_npu *ctx, int K, int N, const float *f32, const float *imatrix, float *bscale_out);
ork_w       *ork_mm_pack_i4(ork_npu *ctx, int K, int N, const int8_t   *B);  /* int4 weights, [-8,7] in int8; K%32, N%64 */
/* int4 weights with per-group scales: K split into groups of G (G%32, K%G, G<=10752). Pair with
 * ork_mm_run_i4_grouped, which dequantizes per group into fp32. */
ork_w       *ork_mm_pack_i4_grouped(ork_npu *ctx, int K, int N, const int8_t *B, int G);

/* int4-Stored / int8-Computed Fallback (Tier 4 Memory Optimization): takes unpacked int4 weights 
 * ([-8,7] in int8 containers) and packs them into an int8 resident weight buffer. This runs on 
 * the highly optimized int8 physical hardware path, yielding maximum silicon speed.
 * Returns an int8 dtype ork_w (run with ork_mm_run_i8). */
ork_w       *ork_mm_pack_i4_to_i8(ork_npu *ctx, int K, int N, const int8_t *B);

/** @brief Free a packed weight's bookkeeping. Does NOT reclaim the NPU IOVA window — see ork_mm_free(). */
void         ork_w_free(ork_w *w);
/**
 * @brief Free a packed weight AND reclaim its NPU DMA/IOVA window (needs ctx for the device fd).
 *
 * Use this instead of ork_w_free() when you need the ~4 GiB IOVA window back (e.g. layer-streaming
 * eviction). Reclaims only per-tile-owned weights (pack/pack_i8/pack_i4); arena-view weights are left
 * to teardown.
 */
void         ork_mm_free(ork_npu *ctx, ork_w *w);
size_t       ork_w_bytes(const ork_w *w);   /* resident NPU bytes (Bb+Bf) — for a streaming cache's IOVA budget */
int          ork_w_quant_kind(const ork_w *w);   /* ORK_QK_* of the int4 weight store (UNIFORM / CODEBOOK_NF4) */
/* Per-output-channel dequant scale (length N) retained on an int4-packed weight (ork_mm_pack_i4a8 /
 * ork_mm_load_i4a8); C_real[m][n] = aScale[m]*bscale[n]*Ci[m][n]. NULL for non-int4 weights. */
const float *ork_w_bscale(const ork_w *w);
/* PERSIST: dump a packed weight's tile bytes (out=NULL → size), and reload pre-tiled int8 bytes straight
 * into DMA (no dequant/quant/tile) — the .orkpack fast path that makes streaming re-packs a plain copy. */
size_t       ork_w_dump(const ork_w *w, void *out, size_t cap);
/* CPU-ONLY int8 dump: tile B[K,N] (int8, row-major) straight into `out` (plain DRAM) in the exact
 * .orkpack byte layout of ork_mm_pack_i8()+ork_w_dump(), WITHOUT allocating an NPU/IOVA buffer or any
 * DMA — the NPU is needed only at load time. Building a .orkpack is a pure-CPU, all-cores job; this
 * avoids the serial bcreate/bsync round-trip. out=NULL → return the byte size. K%32, N%32. */
size_t       ork_w_dump_i8_cpu(ork_npu *ctx, int K, int N, const int8_t *B, void *out, size_t cap);
ork_w       *ork_mm_load_i8(ork_npu *ctx, int K, int N, const void *blob, size_t n);
/* COMPACT int4 PERSIST (the streaming consumer for a mixed .orkpack): dump the COMPACT int4 nibble store
 * + per-channel scales (~half of the int8 ork_w_dump), and reload it straight into NPU DMA, inflating the
 * nibbles -> int8 (UNIFORM sign-extend / NF4 LUT, per the stored quant_kind) and re-tiling on load. Only
 * valid for an int4-packed weight (ork_mm_pack_i4a8); the LUT is derived from quant_kind, not stored.
 * Blob layout: { u32 magic 'O4N1', u32 version=1, i32 K, i32 N, u32 quant_kind } + bscale[N] (N f32) +
 * Bi4 (K*N/2 bytes). out=NULL -> required size; returns 0 if `w` has no int4 store. Loaded weight runs
 * via ork_mm_run_i8 and re-dumps byte-identically. NULL on a malformed blob / shape mismatch. */
size_t       ork_w_dump_i4a8(const ork_w *w, void *out, size_t cap);
ork_w       *ork_mm_load_i4a8(ork_npu *ctx, int K, int N, const void *blob, size_t n);
/* Zero-copy IMPORT variant of ork_mm_load_i4a8: resident tiles are dma-bufs the NPU reads in place (PRIME
 * import); the int4 nibbles inflate -> int8 directly into them. Bit-identical to ork_mm_load_i4a8 (same
 * blob, same tiled bytes, same re-dump). Returns NULL if import is unavailable (caller falls back to
 * ork_mm_load_i4a8) or on a malformed blob / shape mismatch. */
ork_w       *ork_mm_load_i4a8_import(ork_npu *ctx, int K, int N, const void *blob, size_t n);

/* CPU-side pack/dump helpers linked by the ggml-ork backend (defined at the end of npu.c; no internal
 * callers). ork_w_dump_i8_cpu_st: single-threaded int8 CPU tile — for callers that parallelize at a
 * coarser grain (one whole tensor per core) so the internal pool wouldn't nest/oversubscribe; byte-
 * identical to ork_w_dump_i8_cpu. ork_pack_i4a8_cpu_blob: CPU int4 pack straight to the compact .orkpack
 * blob (bit-identical to ork_mm_pack_i4a8_im + ork_w_dump_i4a8) with NO bcreate/IOMMU/tiling.
 * ork_mm_pack_i8_import: tile int8 B[K,N] directly into IMPORTED dma-buf chunks (uniform ~16MB chunks,
 * no native-alloc outlier — for co-resident fused per-tensor weights). K%32,N%32; dump/pack out=NULL ->
 * required byte size. */
size_t       ork_w_dump_i8_cpu_st(ork_npu *ctx, int K, int N, const int8_t *B, void *out, size_t cap);
size_t       ork_pack_i4a8_cpu_blob(ork_npu *ctx, int K, int N, const float *f32, const float *imatrix, void *out, size_t cap);
ork_w       *ork_mm_pack_i8_import(ork_npu *ctx, int K, int N, const int8_t *B);

/* ---- Streaming weight pool: a RAM-resident inflated-int8 cache with CHEAP map/unmap ----
 * For models too big to keep resident in the ~4 GiB NPU IOVA window. The caller (e.g. a layer/expert LRU)
 * keeps a set of ALREADY-INFLATED int8 weights resident in CPU RAM (budget by RAM — much larger than the
 * IOVA window) and maps/unmaps them to IOVA cheaply on demand: a cache HIT pays only the cheap MEM_CREATE
 * import (~170us@4MB), skipping the expensive int4->int8 inflate (paid ONCE on add) and the expensive
 * MEM_DESTROY (paid only on eviction). The pool provides the lifecycle ONLY — the eviction/LRU POLICY and
 * the RAM budget live in the caller. Both stores: int8 (fill=copy ork_w_dump bytes) and int4 (fill=inflate
 * the ork_w_dump_i4a8 nibbles). A transient prefetch double-buffer is just a small pool the caller fills
 * ahead on a thread. All ops bit-exact vs the equivalent ork_mm_load_*. NULL/create-fail if the dma-heap
 * is absent (caller falls back to ork_mm_load_i8 / ork_mm_load_i4a8 + ork_mm_run_i8). */
typedef struct ork_stream_pool  ork_stream_pool;
typedef struct ork_stream_entry ork_stream_entry;
ork_stream_pool  *ork_stream_pool_create(ork_npu *ctx);                 /* NULL if import unavailable */
ork_stream_entry *ork_stream_pool_add_i4a8(ork_stream_pool *p, int K, int N, const void *blob, size_t n); /* int4 store; fill=inflate (once) */
ork_stream_entry *ork_stream_pool_add_i8  (ork_stream_pool *p, int K, int N, const void *blob, size_t n); /* int8 store; fill=copy (once) */
/* int8 store from a freshly-quantized UNTILED int8 B[K][N] (row-major): tile straight into RAM staging,
 * NO transient IOVA pack (no bcreate). For a caller that just quantized a weight and wants it in the pool
 * without packing to IOVA first (which competes with the pool's mapped hot set). NULL on bad dims. */
ork_stream_entry *ork_stream_pool_add_i8_raw(ork_stream_pool *p, int K, int N, const int8_t *B);
int               ork_stream_pool_map  (ork_stream_pool *p, ork_stream_entry *e);  /* CHEAP: MEM_CREATE import -> IOVA (per submit) */
int               ork_stream_pool_run  (ork_stream_pool *p, ork_stream_entry *e, int M, const int8_t *A, int32_t *C);
void              ork_stream_pool_unmap(ork_stream_pool *p, ork_stream_entry *e);  /* MEM_DESTROY; entry STAYS in RAM */
void              ork_stream_pool_remove(ork_stream_pool *p, ork_stream_entry *e); /* free the RAM buffer (caller's evict) */
void              ork_stream_pool_free (ork_stream_pool *p);
size_t            ork_stream_entry_bytes (const ork_stream_entry *e);   /* RAM bytes held (for the caller's budget) */
int               ork_stream_entry_mapped(const ork_stream_entry *e);   /* 1 if currently IOVA-mapped */

/**
 * @brief Run C[M,N] = A[M,K] x packed weights (fp16). Run dtype must match the pack dtype.
 * @param w Weight from ork_mm_pack().
 * @param M Activation rows; any M>=1 (tiled + scheduled internally, no caller-visible cap).
 * @param A Row-major fp16 activations, M*K elements.
 * @param C Output, M*N fp32, row-major (caller-allocated). May be an ork_dma_alloc() buffer (zero-copy).
 * @return 0 on success, negative on error (bad args / submit failure).
 */
int          ork_mm_run   (ork_npu *ctx, ork_w *w, int M, const ork_f16 *A, float   *C);
/**
 * @brief Run C[M,N] = A[M,K] x packed weights (int8/w8a8). Run dtype must match the pack dtype.
 * @param w Weight from ork_mm_pack_i8().
 * @param M Activation rows; any M>=1 (tiled + scheduled internally, no caller-visible cap).
 * @param A Row-major int8 activations, M*K elements.
 * @param C Output, M*N int32 raw sums, row-major (caller-allocated). Apply the per-tensor/-channel
 *          scales in the caller. May be an ork_dma_alloc() buffer (zero-copy).
 * @return 0 on success, negative on error (bad args / submit failure).
 */
int          ork_mm_run_i8(ork_npu *ctx, ork_w *w, int M, const int8_t  *A, int32_t *C);
/* FUSED SwiGLU gate matmul: C = silu(requant(A·W)) as int8 [M*N], activation applied IN the matmul's
 * SDP output stage (no extra submit / round-trip). Resident full-K int8 weight (K%512==0, K<=4096).
 * R = r_mult/2^r_shift + out_bias is the SCALAR OUT_CVT (so quantize A PER-TENSOR, not per-row);
 * idx_off/cfg4068 = the SiLU index-map params; lut/nlut optional (NULL = fixed silu*S LUT). rk3588
 * only. The productionizable on-NPU-activation path (standalone SDP ops lose ~8x — RE-roadmap M4.6). */
int          ork_mm_run_i8_silu(ork_npu *ctx, ork_w *w, int M, const int8_t *A, int8_t *C,
                                int r_mult, int r_shift, unsigned out_bias, unsigned idx_off,
                                unsigned cfg4068, const short *lut, int nlut);
/* Same as ork_mm_run_i8_silu but INT32 output (silu NOT quantized to int8): out_i32 = R*V16[idx]+out_bias,
 * unclamped. With a fine-scale LUT (out_scale ~ silu_max/8000) this is ~13-14 bit silu, recovering the
 * quality the int8 silu output loses (the FFN-chain gap) while keeping silu free on-NPU. C is int32 [M*N]. */
int          ork_mm_run_i8_silu32(ork_npu *ctx, ork_w *w, int M, const int8_t *A, int *C,
                                  int r_mult, int r_shift, unsigned out_bias, unsigned idx_off,
                                  unsigned cfg4068, const short *lut, int nlut);
/* fp16 gate matmul + fused SiLU with fp16->fp32 output (NO int8 activation quant) — the "end-goal" precise
 * on-NPU gate. Recovers the PPL the int8 silu output loses, but the fp16 matmul is ~3.3x int8 (net-loss today,
 * gated OFF, built for a future int8-win pipeline). w = fp16 weight (ork_mm_pack), A = fp16 [M,K], C = fp32
 * [M,N] silu(gate). K%32, N%16, N<=nmax. 0/ok,-1,-2,-3. WIP: fp16 LUT calibration approximate. */
int          ork_mm_run_f16_silu(ork_npu *ctx, ork_w *w, int M, const ork_f16 *A, float *C,
                                 unsigned out_bias, unsigned idx_off, unsigned cfg4068,
                                 const short *lut, int nlut);
/* Calibrate the fp16 fused SiLU for a layer whose real gate spans [-Gmax,Gmax]: fills lut[1030] and returns
 * S (pack the gate weight as -S*W), R, and out_scale (silu(gate) = C_out * out_scale). See Exp-2026-07-05. */
int          ork_mm_build_f16_silu_lut(ork_npu *ctx, double Gmax, short *lut,
                                       double *S_out, double *R_out, double *out_scale_out);
/* FUSED SwiGLU up matmul: C = clamp_i8(round( (A·W_up) * G * gain )) as int8 [M*N], the element-wise
 * multiply by G (= silu(gate) from ork_mm_run_i8_silu) applied IN the up matmul's SDP output stage.
 * gain = mult/2^shift = s_up*s_silu/s_out. G is dense int8 [M*N]. Resident full-K int8 weight
 * (K%512==0, K<=4096). Together with ork_mm_run_i8_silu this is the full fused SwiGLU FFN inner. */
int          ork_mm_run_i8_ewmul(ork_npu *ctx, ork_w *w, int M, const int8_t *A, const int8_t *G,
                                 int8_t *C, int mult, int shift);
/* Resident int8 matmul with int8-requantized output: C = clamp_i8(round((A·W)*mult/2^shift)) [M*N].
 * Keeps intermediates int8 on the NPU (the "up" projection feeding a chained FFN-inner EW-mul). */
int          ork_mm_run_i8_out8(ork_npu *ctx, ork_w *w, int M, const int8_t *A, int8_t *C,
                                int mult, int shift);
/* int4 (W4A4): A int4 ([-8,7] in int8, row-major), C int32 raw sum — apply scales:
 * C_real[m][n] = aScale[m]*bScale[n]*C[m][n]. Run dtype must match the pack dtype. 0 ok / negative err. */
int          ork_mm_run_i4(ork_npu *ctx, ork_w *w, int M, const int8_t  *A, int32_t *C);
/* grouped int4 (per-group W4A4 dequant): A int4 [M*K] ([-8,7] in int8); aScale [M*(K/G)] (per row,
 * per group), bScale [(K/G)*N] (per group, per channel). C fp32 [M*N] = dequantized result. Pair
 * with ork_mm_pack_i4_grouped. (Cost: K/G submits/core — larger G = fewer submits, coarser scale.) */
int          ork_mm_run_i4_grouped(ork_npu *ctx, ork_w *w, int M, const int8_t *A,
                                   const float *aScale, const float *bScale, float *C);

/* Profiling: read accumulated run_multicore phase times (us) and call count since process start.
 * setup = pre-dispatch checks + mc_ensure + cres memset; submit = pool dispatch + workers + NPU;
 * copy = cres->C memcpy. Any pointer may be NULL. Pins integration overhead vs the NPU itself. */
void         ork_npu_run_timing(double *setup, double *submit, double *copy, long *n);

/* Profiling (ORK_MCPROF diagnostic): per-core phase times (us) inside the multi-core prefill (M>1)
 * path — copy (activation host-copy + bsync), submit (regcmd + ioctl + result bsync), acc (host
 * accumulate), and the submit count. Pins why large-M multi-core barely scales. Call _reset before
 * the timed region. core in [0, cores). */
void         ork_npu_mc_reset(void);
void         ork_npu_mc_timing(int core, double *copy, double *submit, double *acc, long *n);
double       ork_npu_mc_synth(int core);   /* host synth+bsync subset of `submit` (overlappable); ioctl/NPU = submit - synth */

/* RE/calibration only: probe this SoC's single-submit K-tile ceiling. Runs ONE M=1 full-K int8
 * submit at (K,N) (N <= SoC N-cap, K%32, N%32) on its own buffers. Returns 0 if the submit
 * completed (C[N] int32 valid — validate vs CPU), -1 if it wedged (K exceeds the per-op K-tile
 * cap; recoverable), -2 on bad dims. See tools/ksubmit_probe.c. */
int          ork_npu_probe_single_i8(ork_npu *ctx, int K, int N, const int8_t *A, const int8_t *B, int32_t *C);

/* PPU fused-output stage (step 1): run ONE full-K int8 matmul at (M,K,N) with the int8-REQUANTIZED
 * output stage instead of int32. out_i8 = clamp_i8((acc_i32 * mult) >> shift); identity = (0x4000,14).
 * Isolated bit-exact test bed for the on-NPU fused path (SiLU/EW-mul build on this int8-output stage).
 * A[M*K], B[K*N] row-major int8; C[M*N] int8 out; us = warm-submit time. 0/ok, -1 wedged, -2 bad dims.
 * See ork_ppu_fuse_enabled(); the CPU/NEON requant path stays the default fallback. */
int          ork_npu_probe_i8_out8(ork_npu *ctx, int M, int K, int N, const int8_t *A, const int8_t *B,
                                   int mult, int shift, int8_t *C, double *us);

/* ork-NATIVE fused-SiLU LUT generator: build ork's OWN silu LUT for the fused-output path (no RKNN
 * dependence). Measures ork's index(acc) for (r_mult,r_shift,cfg4068) via one calibration submit, then
 * builds lut[1030] = silu curve matched to ork's mapping for (in_scale,out_scale). Do this ONCE per
 * register config; run matmuls via ork_npu_probe_i8_silu_cfg(..,r_mult,r_shift,0,0xffffc000,cfg4068,lut,1030,..).
 * Pick R=r_mult/2^r_shift ~= 660*in_scale so the acc range spans silu's transition. Validated ~1 int8.
 * 0/ok, -1 fail. See tools/silu_native.c. */
int          ork_mm_silu_build_lut(ork_npu *ctx, double in_scale, double out_scale,
                                   int r_mult, int r_shift, uint32_t cfg4068, int16_t *lut);

/* PPU FUSED SiLU (step 2): full-K int8 matmul with SiLU applied on-chip via the LUT output stage. Two
 * sequential submits (LUT-load into PPU SRAM, then matmul reading it). A[M*K],B[K*N] int8; C[M*N] int8.
 * 0/ok (executed), -1 wedged, -2 bad. WIP: replays the capture scale (per-scale LUT gen is pending). */
int          ork_npu_probe_i8_silu(ork_npu *ctx, int M, int K, int N, const int8_t *A, const int8_t *B,
                                   int8_t *C, double *us);
/* Fused-SiLU probe with the decoded output-stage knobs exposed: r_mult/r_shift = the unified scale
 * R = r_mult/2^r_shift (reg 0x4084/0x4088; sets both the acc->LUT-index step and the LUT->output gain);
 * out_bias = reg 0x4080 (output bias / asymmetric zero-point); idx_off = reg 0x4110 (LUT-index offset C0);
 * cfg4068 = reg 0x4068 (per-scale field, no observed output effect). lut != NULL overrides the LUT contents
 * (for the staircase/const-LUT calibration harness); lut==NULL uses the fixed captured silu curve. */
int          ork_npu_probe_i8_silu_cfg(ork_npu *ctx, int M, int K, int N, const int8_t *A, const int8_t *B,
                                       int r_mult, int r_shift, uint32_t out_bias, uint32_t idx_off,
                                       uint32_t cfg4068, const int16_t *lut, int nlut, int8_t *C, double *us);

/* RE/validation for the fused EW-mul (SwiGLU dual-input) output stage: full-K int8 matmul A*B whose output
 * stage int8-requantizes the accumulator and multiplies it by a SECOND input G (= silu(gate)); returns
 * C[M*N] int8. Splices the 0x50xx second-DPU lane (regcfg 108->126). A[M*K] B[K*N] G[M*N] row-major int8.
 * First-run contract: validate at the captured shape (M=8,N=32) — see EWMUL_WIP.md. 0/ok, -1 wedged, -2 dims. */
int          ork_npu_probe_i8_ewmul(ork_npu *ctx, int M, int K, int N, const int8_t *A, const int8_t *B,
                                    const int8_t *G, int mult, int shift, int8_t *C, double *us);

/* Path (b): submit RKNN's captured EW-mul op VERBATIM (REGCMD_EWMUL) with ork's buffers, only repointing
 * addresses. Tests whether the templatized op executes on ork's submit path (RKNN's own geometry). Buffers
 * mirror the captured handle layout: in->input(4KiB), wt->weights(32KiB,+0x2300), gl->silu(8KiB,+0x400),
 * out<-output(4KiB). 0/ok, -1 wedged. ORK_EW_DUMP=1 prints the regcmd and skips the submit. */
int          ork_npu_probe_i8_ewmul_tmpl(ork_npu *ctx, const void *in, int Isz, const void *wt, int Wsz,
                                         const void *gl, int Gsz, void *out, int Osz, double *us);

/* Path (b) matmul replay: the captured matmul-shaped EW-mul op (K=512,N=64,M=8) with ork's tile packing.
 * out = requant(A*B) (mul) G, G=silu(gate) int8. A[8*512] B[512*64] G[8*64] C[8*64] int8. 0/ok,-1,-2. */
int          ork_npu_probe_i8_ewmul_lin(ork_npu *ctx, const int8_t *A, const int8_t *B, const int8_t *G,
                                        int8_t *C, double *us);

/* Standalone SDP element-wise MULTIPLY (NVDLA standalone SDP layer, both operands from memory): out[i] =
 * clamp_i8(round(a[i]*b[i]*gain)+bias). a,b,out int8[n] (n<=4096). The clean on-NPU element-wise path. 0/ok. */
int          ork_npu_probe_i8_mul(ork_npu *ctx, const int8_t *a, const int8_t *b, int n, int8_t *out, double *us);

/* On-NPU element-wise MULTIPLY of two int8 [M][N] tensors (e.g. the SwiGLU inner silu(gate)⊙up):
 * out[m*N+n] = clamp_i8(round(up[m][n]*silu[m][n] * mult/2^shift)) computed on the NPU (standalone SDP op).
 * Handles the NVDLA feature-cube marshaling internally; symmetric int8 (zero-points 0). mult in 0..0x7fff
 * (OUT_CVT_SCALE is signed 16-bit); gain = mult/2^shift (= s_up·s_silu/s_out for SwiGLU). Supported shape:
 * M=8,N=64 (the captured op geometry) — other shapes return -2 pending cube-dim generalization. rk3588-gated
 * (returns -3 elsewhere). 0/ok, -1 wedged, -2 bad shape. Validated bit-exact vs CPU (examples/test_ewmul_i8). */
int          ork_npu_ewmul_i8(ork_npu *ctx, const int8_t *up, const int8_t *silu, int M, int N,
                              int mult, int shift, int8_t *out, double *us);

/* fp16 element-wise MULTIPLY of two [M][N] fp16 tensors on the NPU: out[m][n] = up[m][n]*silu[m][n] (no
 * requant — fp16 standalone SDP op). NVDLA fp16 feature-cube marshaled internally. up/silu/out are fp16
 * bit-patterns. Supported shape M=8,N=64; rk3588-gated. 0/ok, -1 wedged, -2 bad shape, -3 non-rk3588. */
int          ork_npu_ewmul_f16(ork_npu *ctx, const ork_f16 *up, const ork_f16 *silu, int M, int N,
                               ork_f16 *out, double *us);

/* int16 element-wise MULTIPLY of two [M][N] int16 tensors on the NPU: out=clamp_i16(round(up*silu*mult/2^shift)).
 * The w4a4 path's EW precision (ork's int4 matmul outputs int16). NVDLA int16 feature-cube (atom-8) marshaled
 * internally; symmetric zero-points. mult in 0..0x7fff. Shape M=8,N=64; rk3588-gated. 0/ok,-1,-2,-3. */
int          ork_npu_ewmul_i16(ork_npu *ctx, const int16_t *up, const int16_t *silu, int M, int N,
                               int mult, int shift, int16_t *out, double *us);

/* Standalone on-NPU SiLU (activation-LUT SDP op): applies the PPU silu LUT to a single int8 input [M][N] via
 * the 69-reg/enable=0x18 standalone op (REGCMD_SILU_STD), reprogrammed to (M,N). Two submits (LUT-load + op).
 * SDP: idx=(in*R)>>6 + C0; out=clamp_i8(R*LUT-interp(idx) + out_bias), R=r_mult/2^r_shift. Caller supplies the
 * scale regs + LUT (lut==NULL keeps the captured curve). RE/calibration entry (measure idx(in) via a ramp LUT
 * then build the curve). in/out int8 [M*N], N%16==0; rk3588-gated. 0/ok,-1 wedged,-2 shape,-3 SoC. */
int          ork_npu_probe_silu_std(ork_npu *ctx, const signed char *in, int M, int N,
                                    int r_mult, int r_shift, unsigned out_bias, unsigned idx_off,
                                    unsigned cfg4064, unsigned cfg4068, const short *lut, int nlut,
                                    signed char *out, double *us);

/* Faithful fp16 replay: run RKNN's fp16 LUT-load program (`loader`/`ln`) + the fp16 compute op verbatim,
 * patching only I/O + M/N. Uses the fp16 (LE-table) loader, not the int8 LO loader. in/out fp16 [M*N], N%8==0. */
int          ork_npu_replay_full_f16(ork_npu *ctx, const unsigned *loader, int ln, const ork_f16 *in, int M, int N, ork_f16 *out, double *us);

/* fp16 standalone activation-LUT op — RE probe. Applies the PPU LUT to a single fp16 input [M][N] via
 * REGCMD_SILU_STD_F16. Two submits (LUT-load + op). in/out fp16 [M*N], N%8==0. 0/ok,-1,-2,-3. */
int          ork_npu_probe_silu_std_f16(ork_npu *ctx, const ork_f16 *in, int M, int N,
                                        unsigned idx_off, unsigned cfg4064, unsigned cfg4068,
                                        const short *lut, int nlut, ork_f16 *out, double *us);

/* int16 (w16a16i) standalone activation-LUT op — RE probe. Same requant-LUT math as the int8 op but int16
 * I/O (atom-8 cube) via REGCMD_SILU_STD_I16. Two submits (LUT-load + op). in/out int16 [M*N], N%8==0. */
int          ork_npu_probe_silu_std_i16(ork_npu *ctx, const short *in, int M, int N,
                                        int r_mult, int r_shift, unsigned out_bias, unsigned idx_off,
                                        unsigned cfg4064, unsigned cfg4068, const short *lut, int nlut,
                                        short *out, double *us);

/* On-NPU SiLU (int8): out[m*N+n] = clamp_i8(round( silu(in[m][n]*in_scale) / out_scale )) via the standalone
 * SDP activation-LUT op. Calibrates the op's index map once per ctx, builds the silu curve for (in_scale,
 * out_scale), loads it, runs the op. in/out int8 [M*N], N%16==0; rk3588-gated. 0/ok,-1 wedged,-2 shape,-3 SoC. */
int          ork_npu_silu_i8(ork_npu *ctx, const signed char *in, int M, int N,
                             double in_scale, double out_scale, signed char *out, double *us);

/* Replay an ASSEMBLED int16 LUT-op (tools/re/assemble_op.c output): stream `lut` via the loader, run `regcmd`
 * verbatim (RKNN's matched index/scale params baked in), patch only addresses + M/N. Bit-exact to RKNN, no
 * index decode. in/out int16 [M*N], N%8==0. 0/ok,-1,-2,-3. */
int          ork_npu_replay_lut_i16(ork_npu *ctx, const unsigned *regcmd, int rn, const short *lut, int nlut,
                                    const short *in, int M, int N, short *out, double *us);

/* On-NPU SiLU (int16 / w16a16i) — EXPERIMENTAL, not yet bit-exact. Runs the standalone int16 activation-LUT op,
 * but the int16 op's index-gain response to 0x4068 differs from the int8 op's (which is decoded), so the LUT
 * index model is approximate pending an int16-op gain sweep. in/out int16 [M*N], N%8==0. 0/ok,-1,-2,-3. */
int          ork_npu_silu_i16(ork_npu *ctx, const short *in, int M, int N,
                              double in_scale, double out_scale, short *out, double *us);

/* On-NPU element-wise ADD (int8): out = clamp_i8(round( (a*a_scale + b*b_scale)/out_scale )) via the 2-input SDP
 * ALU=add op. Symmetric quant. Residual add (a_scale==b_scale==out_scale) => out=clamp_i8(a+b), bit-exact.
 * in/out int8 [M*N], N%16==0; rk3588-gated. 0/ok,-1 wedged,-2 shape,-3 SoC. */
int          ork_npu_add_i8(ork_npu *ctx, const signed char *a, const signed char *b, int M, int N,
                            double a_scale, double b_scale, double out_scale, signed char *out, double *us);

/* On-NPU fp16 element-wise ADD (residual): out = a + b in fp16 via the 2-input SDP ALU=add op. in/out fp16
 * [M*N], N%8==0; rk3588-gated. 0/ok,-1,-2,-3. */
int          ork_npu_add_f16(ork_npu *ctx, const ork_f16 *a, const ork_f16 *b, int M, int N, ork_f16 *out, double *us);

/* On-NPU GELU (int8): out = clamp_i8(round( gelu(in*in_scale)/out_scale )) via the same standalone SDP
 * activation-LUT op as SiLU (the LUT holds the GELU curve). Bit-exact-class. N%16==0. 0/ok,-1,-2,-3. */
int          ork_npu_gelu_i8(ork_npu *ctx, const signed char *in, int M, int N,
                             double in_scale, double out_scale, signed char *out, double *us);
/* On-NPU GELU (int16 / w16a16i): same op, GELU curve. RKNN-class accuracy. N%8==0. 0/ok,-1,-2,-3. */
int          ork_npu_gelu_i16(ork_npu *ctx, const short *in, int M, int N,
                              double in_scale, double out_scale, short *out, double *us);

/* On-NPU rsqrt (int8/int16) — RMSNorm building block, via the same activation-LUT op (rsqrt curve, positive
 * domain): out = clamp(round( rsqrt(in*in_scale)/out_scale )). 0/ok,-1,-2,-3. */
int          ork_npu_rsqrt_i8(ork_npu *ctx, const signed char *in, int M, int N, double in_scale, double out_scale, signed char *out, double *us);
int          ork_npu_rsqrt_i16(ork_npu *ctx, const short *in, int M, int N, double in_scale, double out_scale, short *out, double *us);
/* On-NPU exp (int8/int16) — softmax building block, activation-LUT (exp curve): out=clamp(round(exp(in*is)/os)). */
int          ork_npu_exp_i8(ork_npu *ctx, const signed char *in, int M, int N, double in_scale, double out_scale, signed char *out, double *us);
int          ork_npu_exp_i16(ork_npu *ctx, const short *in, int M, int N, double in_scale, double out_scale, short *out, double *us);

/* On-NPU int16 element-wise ADD — EXPERIMENTAL, not bit-exact over the signed range: the SRDMA/X1 operand halves
 * negative values (int16-specific SDP X1 sign/shift behavior not yet decoded); positive is exact. out =
 * clamp_i16(round((a*a_scale + b*b_scale)/out_scale)). in/out int16 [M*N], N%8==0. 0/ok,-1,-2,-3. */
int          ork_npu_add_i16(ork_npu *ctx, const short *a, const short *b, int M, int N,
                             double a_scale, double b_scale, double out_scale, short *out, double *us);

/* Standalone int8 element-wise ADD — RE probe (settable scale regs). 2-input SDP op with ALU=add. Caller sets
 * out scale mult/shift, b-operand scale bscale, and zero-points za/zb/zo. a/b/out int8 [M*N], N%16==0. 0/ok,-1,-2,-3. */
int          ork_npu_probe_add_i8(ork_npu *ctx, const signed char *a, const signed char *b, int M, int N,
                                  int mult, int shift, unsigned bscale, int za, int zb, int zo, signed char *out, double *us);

/* Runtime gate for the PPU fused-output path. Gated on the SoC detected at startup: returns 1 only on
 * a validated PPU target (currently rk3588). On any other chip, ork-driver emits int32 output and the
 * caller's CPU/NEON requant+activation stage runs — identical numerics. The fused stage is a HW-specific
 * optimization (RE'd against the rk3588 PPU register layout), never a dependency. */
int          ork_ppu_fuse_enabled(ork_npu *ctx);

/* RE/calibration only: run ONE full-K int8 submit at (M,K,N) in ork's current M-tile mode (mode=0)
 * or the rkllm-captured M-tile mode (mode=1: 0x1010=0x20 const, 0x1044=(K/64)*M, 0x107c=4*M,
 * 0x1040=0xb1-0xf*(ceil(M/8)-1)). Returns the full C[M*N] int32 + warm-submit us. Tests whether
 * rkllm's larger-M-per-submit is bit-exact on ork and lifts effective TOPS. 0/ok, -1 wedged, -2 bad.
 * A[M*K], B[K*N] row-major int8; C[M*N] int32. See tools/mtile_probe.c. */
int          ork_npu_probe_mtile_i8(ork_npu *ctx, int M, int K, int N, int mode,
                                    const int8_t *A, const int8_t *B, int32_t *C, double *us);

/* RE/calibration only: does batching tasks per RKNPU_SUBMIT amortize the per-submit round-trip
 * floor? Runs `ntask` identical small int8 matmuls as ntask separate ioctls vs one ioctl with
 * task_number=ntask, writing the two total wall times (us). Returns 0/ok, -1 wedged, -2 bad dims.
 * See tools/batch_probe.c. */
int          ork_npu_probe_batch(ork_npu *ctx, int ntask, int K, int N, double *us_unbatched, double *us_batched);

/* RE/calibration only: probe in-place K-slicing of a full-K weight buffer. Packs B[Kfull,N] fp16
 * full-K, runs one M=1 submit over k in [0,Kp), with up to `nov` regcmd overrides (block 0x0201)
 * to hunt the per-N-tile weight stride register. C[N] should equal the Kp-partial sum if slicing
 * works. Returns 0/ok, -1 wedged, -2 bad dims. See tools/slice_probe.c. */
int          ork_npu_probe_slice_f16(ork_npu *ctx, int Kfull, int N, int Kp, int nov,
                                     const uint32_t *ovr_reg, const uint32_t *ovr_val,
                                     const ork_f16 *A, const ork_f16 *B, float *C);

/* RE/calibration only: probe W4A4 (int4 A x int4 B -> int16 C) using the captured RK3588 regcmd
 * (REGCMD_I4, M=4) and the DOCUMENTED native tile layouts (A:(K/32,M,32) B:(N/64,K/32,64,32)
 * C:(N/8,M,8)). A is [4*K], B is [K*N] row-major, int4 values stored as int8 in [-8,7]. `nibB`/`nibA`
 * (0/1) toggle the 2-int4-per-byte nibble order (the one detail the docs don't pin). C is [4*N]
 * int16 (de-tiled). `nov`/`ovr_*` patch extra CNA (0x0201) regs. Returns 0/ok, -1 wedged/abort,
 * -2 bad dims (K%32, N%64). See tools/i4_probe.c. */
int          ork_npu_probe_i4(ork_npu *ctx, int M, int K, int N, int nibB, int nibA, int nov,
                              const uint32_t *ovr_reg, const uint32_t *ovr_val,
                              const signed char *A, const signed char *B, short *C);

/* RE/calibration only (Tier 4b): ONE multi-M int4 submit with the M-count regs set to M (synth_i4
 * mc=M). A:(K/32,M,32), B:(N/64,K/32,64,32) native. Copies the RAW int16 output (M*N, NO de-tile) to
 * `raw` so the caller can deduce the multi-M C layout vs a CPU reference. 0/ok, -1 wedged, -2 dims.
 * See tools/i4_multim_probe.c. */
int          ork_npu_probe_i4_mm(ork_npu *ctx, int M, int K, int N,
                                 const signed char *A, const signed char *B, short *raw);

/* RE/calibration only: run S chained M=1 full-K int8 matmuls using PC-chaining in a single submit */
int          ork_npu_probe_chain_i8(ork_npu *ctx, int S, int K, int N, const int8_t *A,
                                    const int8_t *B, int32_t *C);

/* RE/calibration only: benchmark S chained matmuls vs S separate submits using pre-allocated memory */
int          ork_npu_benchmark_chain(ork_npu *ctx, int S, int K, int N, int iters);

/* Mixture of Experts (MoE) / Chained matmuls API */
typedef struct {
    ork_w *w;
    int M;
    const int8_t *A;
    int32_t *C;
} ork_mm_task_i8;

typedef struct {
    ork_w *w;
    int M;
    const int8_t *A;
    int32_t *C;
} ork_mm_task_i4;

int          ork_mm_run_chain_i8(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks);
int          ork_mm_run_chain_i4(ork_npu *ctx, int S, const ork_mm_task_i4 *tasks);
/* Async round-robin stream: S independent int8 matmuls dispatched dynamically across NPU cores (pull
 * model, no barrier). For batches of independent matmuls (e.g. EAGLE-3 verification). 0/ok, -1/-2 err. */
int          ork_mm_run_stream_i8(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks);
/* int4 (W4A4) async round-robin stream: S independent matmuls across NPU cores (a task's M rows become M
 * single-row regcmds PC-chained on its core). Weights single-slice (Sn==1 && Sk==1). 0/ok, -1/-2 err. */
int          ork_mm_run_stream_i4(ork_npu *ctx, int S, const ork_mm_task_i4 *tasks);

/* Async submit (CPU‖NPU overlap foundation), PATH-AGNOSTIC. The RKNPU submit ioctl blocks the caller
 * until the NPU job finishes; the NPU is single-stream (one queue), so async here means the BLOCKING
 * submit runs on a worker thread while the CALLING thread does independent CPU work, joining at the
 * dependency. This is a DISPATCH-level wrapper around the synchronous run functions, so it works for
 * fp16 (ork_mm_run), int8 (ork_mm_run_i8), int4 (ork_mm_run_i4), and the chain/stream variants — same
 * numerics as the synchronous run (reused verbatim). Each launcher returns a handle immediately (NULL
 * on bad args → fall back to the matching synchronous run); ork_async_wait joins, returns the result
 * (0/ok, <0 err) and frees the handle. CONTRACT: keep at most ONE async job in flight and issue no
 * other ork_mm_* on the same ctx between launch and wait (only independent CPU work) — the NPU is
 * single-stream. The task arrays passed to the chain/stream launchers must stay valid until wait. */
typedef struct ork_async ork_async;
ork_async   *ork_mm_run_async        (ork_npu *ctx, ork_w *w, int M, const ork_f16 *A, float   *C);
ork_async   *ork_mm_run_i8_async     (ork_npu *ctx, ork_w *w, int M, const int8_t  *A, int32_t *C);
ork_async   *ork_mm_run_i4_async     (ork_npu *ctx, ork_w *w, int M, const int8_t  *A, int32_t *C);
ork_async   *ork_mm_run_chain_i8_async (ork_npu *ctx, int S, const ork_mm_task_i8 *tasks);
ork_async   *ork_mm_run_chain_i4_async (ork_npu *ctx, int S, const ork_mm_task_i4 *tasks);
ork_async   *ork_mm_run_stream_i8_async(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks);
ork_async   *ork_mm_run_stream_i4_async(ork_npu *ctx, int S, const ork_mm_task_i4 *tasks);
int          ork_async_wait(ork_async *h);
/* CPU the most recent async worker was placed on at entry (sched_getcpu), -1 if none/non-Linux.
 * Diagnostic + regression test for the worker-pinning pattern (worker lands on a big core). */
int          ork_npu_last_async_cpu(ork_npu *ctx);

/* Math utilities for caller-driven quantization/transformations */
void         ork_fwht_norm(float *v, int n);

#endif
