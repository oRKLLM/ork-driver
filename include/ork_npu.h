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
#define ORK_NPU_VERSION "1.0.0"
/* On-disk .orkpack format version — DECOUPLED from the library MAJOR. Bump this ONLY when the persisted bytes'
 * meaning changes (tile layout/geometry or quant rule); it stays at the MAJOR of the last format-changing
 * release. The 1.0.0 release did NOT change the format, so it stays 0 (existing .orkpacks remain valid). */
#define ORK_PACK_FORMAT_VERSION 0u
/** @brief Runtime library version. @return "MAJOR.MINOR.PATCH" or "MAJOR.MINOR.PATCH+g<hash>". */
const char  *ork_npu_version(void);

/**
 * @brief On-disk pack-format (.orkpack) compatibility token — the library's MAJOR version.
 *
 * A persisted weight (ork_w_dump / ork_w_dump_i4a8) is only binary-compatible with builds that share
 * this value (ORK_PACK_FORMAT_VERSION). It tracks the MAJOR of the last format-changing release but is
 * DECOUPLED from ORK_NPU_VERSION's MAJOR (a library major bump that does NOT touch the on-disk bytes — e.g.
 * the 1.0.0 stability release — keeps this at its prior value so existing .orkpacks stay valid). Bump it
 * only on a real persisted-bytes change: a resident tile LAYOUT / geometry change (K-slice size, the SoC output-width
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
/* Pin the fused chain (run_chain_i8_*) to one NPU core (LUT-load + submit run there). Default 0. For
 * round-robin, place independent chains on different cores. Out-of-range -> 0. */
void         ork_npu_set_chain_core(ork_npu *ctx, int core);
void         ork_npu_set_chain_core_unsafe(ork_npu *ctx, int core);   /* TEST-ONLY: pin chain core, no clamp; caller must have brought the core online first (see npu.c) */

/* orkd scheduler priority for this context's subsequent submits (higher = dispatched sooner among queued work;
 * ties broken by IOMMU-domain affinity then FIFO). Only meaningful when routed through orkd (ORK_USE_ORKD);
 * a no-op on a direct-NPU context. Default 0. */
void         ork_npu_set_priority(ork_npu *ctx, unsigned prio);

/* Per-weight NPU IOMMU domain placement. The rk_iommu 32-bit IOVA window (~4 GiB) is per
 * iommu_domain_id, so a model larger than 4 GiB can stay FULLY resident (no streaming, no per-token
 * map/unmap) by placing its weights across several domains. Call this before ork_mm_pack_i8 /
 * ork_mm_load_i8 (and the fp16/int4 variants): each weight packed/loaded afterward lands its resident
 * tiles in `domain` and records it; ork_mm_run* then submits that weight's matmuls against the same
 * domain automatically. domain<0 reverts to the process default (env ORK_IOMMU_DOMAIN, else 0). */
void         ork_npu_set_pack_domain(ork_npu *ctx, int domain);
int          ork_npu_pack_domain(const ork_npu *ctx);   /* current pack domain (for save/restore) */
void         ork_npu_activate_domain(ork_npu *ctx, int domain);   /* make domain active (establish); alloc in-domain buffers after this */
int          ork_w_domain(const ork_w *w);   /* the IOMMU domain a packed weight resides in */
int          ork_npu_uses_orkd(const ork_npu *ctx);   /* 1 = context routes through orkd (serialized); 0 = direct single-stream NPU */
/* Client-managed IOMMU domains. ork_npu_domain_alloc reserves a domain (Path B: from orkd's coordinated pool
 * so multi-process clients don't collide; direct: a local id). Pack into it with ork_npu_set_pack_domain, then
 * ork_npu_domain_free when done (also auto-reclaimed when the context/connection is freed). id>0 ok, <0 error. */
int          ork_npu_domain_alloc(ork_npu *ctx);
int          ork_npu_domain_free(ork_npu *ctx, int domain);

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
/** @brief Like ork_dma_alloc but requests on-chip NPU SRAM residence (fails over to DRAM if none/full). */
void        *ork_dma_alloc_sram(ork_npu *ctx, size_t size);
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
/* Import an EXTERNAL dma-buf fd (e.g. received over SCM_RIGHTS from another process) into the NPU's IOMMU
 * domain and register it for zero-copy — the returned CPU pointer maps the shared buffer, and passing a ptr
 * into it as A/C to ork_mm_run* makes the NPU read/write it IN PLACE (no copy). Takes ownership of the fd
 * (closed by ork_dma_free/ork_dma_import_free). NULL on failure. Enables the orkd daemon to run a matmul
 * directly against a client's shared buffer. Same 32-bit IOVA cap as ork_dma_import. */
void        *ork_dma_import_fd(ork_npu *ctx, int dmabuf_fd, size_t size);

/* Load pre-tiled int8 weight bytes (ork_w_dump / .orkpack) into NPU-resident storage WITHOUT the
 * alloc+memcpy of ork_mm_load_i8: each tile is imported zero-copy (dma-buf the NPU reads in place).
 * Same blob format and round-trip as ork_mm_load_i8; returns NULL on shape/size mismatch or if import
 * is unavailable (caller falls back to ork_mm_load_i8). Weights are write-once: filled+synced here,
 * read-only across every submit. ork_mm_free / ork_w_free release the imports (MEM_DESTROY + close fd). */
ork_w       *ork_mm_load_i8_import(ork_npu *ctx, int K, int N, const void *blob, size_t n);

/* ---- ORKD_IMPORT: client-owned resident weight (client allocs the dma-buf, daemon only maps it) ----
 * CLIENT (no NPU fd): ork_dmabuf_alloc reserves a dma-heap buffer + mmaps it R/W (returns the fd, sets *ptr);
 * fill *ptr with the PRE-TILED .orkpack bytes, then ork_dmabuf_seal(fd) flushes it. Pass the fd to the daemon
 * (SCM_RIGHTS). DAEMON: ork_mm_adopt_imported_i8 PRIME-imports the fd into the current pack_domain and lays
 * Bb (+ optional Bf at bf_off) as base+offset VIEWS — no tiling, no weight alloc, bytes stay in the client's
 * buffer. total = buffer bytes; bf_off = byte offset of the full-K Bf region (0 = none). NULL on mismatch. */
int          ork_dmabuf_alloc(size_t size, void **ptr);   /* -> dma-buf fd (>=0), *ptr = mapping; -1 on failure */
void         ork_dmabuf_seal(int dmabuf_fd);              /* flush CPU writes so the imported NPU view sees them */
ork_w       *ork_mm_adopt_imported_i8(ork_npu *ctx, int K, int N, int bb_fd, int bf_fd, size_t bb_bytes, size_t bf_bytes);
/* CLIENT orchestrator (orkd): alloc dma-buf, copy the pre-tiled blob (Bb [+ Bf at bf_off]), seal, and send the
 * fd to the daemon (ORKD_IMPORT). Returns an is_orkd weight handle. n = blob bytes; bf_off = Bf offset (0=none). */
ork_w       *ork_mm_import_i8(ork_npu *ctx, int K, int N, const void *blob, size_t n, size_t bf_off);

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
/* Tier 12f — resident K/V with per-key append for decode attention. Alloc two zeroed resident int8 weights:
 * wkt = K^T[512, Lmax] (Q·K^T), wv = V[Lmax, HD]. ork_kv_append writes ONE key's tile bytes each step (no repack),
 * so decode attention keeps the KV packed and only the matmuls run per token. Run with M=1, K/N=Lmax (keys beyond
 * the current length are zero -> contribute nothing); the caller does the [1,len] softmax on the host. HD%32,
 * Lmax%32, Lmax<=nmax. Local NPU only. See src/npu.c for the tiling + scale contract (per-key ks / single vs). */
#ifndef ORK_KV_RESIDENT_T
#define ORK_KV_RESIDENT_T
typedef struct { ork_w *wkt, *wv; int HD, Lmax, Kp; uint64_t orkd_kv; } ork_kv_resident;
#endif
ork_kv_resident *ork_kv_resident_alloc(ork_npu *ctx, int HD, int Lmax);
int              ork_kv_append(ork_npu *ctx, ork_kv_resident *kv, int key, const int8_t *kcol, const int8_t *vrow);
void             ork_kv_resident_free(ork_npu *ctx, ork_kv_resident *kv);
/* re-tile int8 B into an existing same-shape ork_w (reuses its DMA; no alloc/free) — for pooling
 * reused weights (MoE experts) without churning/fragmenting the NPU IOMMU. 0 ok / -1 / -2 mismatch. */
int          ork_mm_repack_i8(ork_npu *ctx, ork_w *w, int K, int N, const int8_t *B);
/* int8 JIT-inflate to fp16 (emulated W8A16 for IOVA headroom). A gmax-selected "fp16" layer wants
 * UNQUANTIZED fp16 activations (no act-quant error) but not fp16 WEIGHTS; residing fp16 weights doubles
 * IOVA. Keep the weight host-side as compact int8 + per-channel bscale, and inflate it into ONE REUSED
 * fp16 scratch per matmul: resident IOVA = a single scratch (reused across layers), so the fp16-path
 * layer count is decoupled from the 4GiB IOVA cap and gmax becomes a pure coherence<->speed dial. The
 * fp16 MAC then runs int8-precision weights against fp16 activations (RK3588 has no native W8A16). */
/* Allocate a REUSABLE fp16 scratch weight (fp16 tile layout, sized K,N, no data). Run via ork_mm_run /
 * ork_mm_run_f16_silu after filling; reclaim with ork_mm_free. K%32, N%16. NULL on bad dims / alloc. */
ork_w       *ork_mm_f16_scratch(ork_npu *ctx, int K, int N);
/* Fill an fp16 scratch (same K,N) with wf16[k,n]=(f16)((float)i8[k*N+n]*bscale[n]). i8 row-major [K,N];
 * bscale per-output-channel [N] (NULL => scale 1). In place, no alloc. Tiled bytes are bit-identical to
 * ork_mm_pack of the row-major dequantized weight. 0/ok, <0 on bad args. */
int          ork_mm_inflate_i8_to_f16(ork_npu *ctx, ork_w *w, const int8_t *i8, const float *bscale, int K, int N);
/* Re-tile fp16 B[K,N] into an EXISTING fp16 weight (ork_mm_f16_scratch/ork_mm_pack, same K,N) — no
 * bcreate/free. fp16 twin of ork_mm_repack_i8: refresh a persistent weight POOL's data per use (kills
 * per-matmul IOMMU alloc/free churn in a dynamic-operand loop like the SSD scan). 0/ok,<0. */
int          ork_mm_repack_f16(ork_npu *ctx, ork_w *w, int K, int N, const ork_f16 *B);
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
/* CPU Bf (full-K re-tiled) blob for the .orkpack — byte-identical to the load-time Bf rebuild. 0 unless the
 * Bf run envelope (K%512==0 && K<=4096). Pass out=NULL to size. Lets a .orkpack carry Bf (no runtime rebuild). */
size_t       ork_w_dump_bf_i8_cpu(ork_npu *ctx, int K, int N, const int8_t *B, void *out, size_t cap);
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
/* Reload a pre-tiled NATIVE-W4A4 weight (ork_w_dump of a DT_I4 ork_mm_pack_i4 weight) straight into NPU DMA —
 * NO dequant/FWHT-rotate/int4-quant/tile (the cold-pack fix for the mul_mat_i4/_hadamard/group_i4 path). blob =
 * this weight's Bb dump (Kp*Nc/2 int4 bytes/tile, pgup'd, pack order). The per-channel bscale is persisted by
 * the caller and re-attached separately. Returns a DT_I4 weight (run with ork_mm_run_i4) or NULL on mismatch. */
ork_w       *ork_mm_load_i4(ork_npu *ctx, int K, int N, const void *blob, size_t n);

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
/* GENERIC fp16 fused-output PWL LUT builder (fp16 twin of the int8/int16 act_lut_i8/i16(fn,...)): bakes an
 * arbitrary fn(x) into the SDP output-stage LUT for x in [in_lo,in_hi]. Pack the reduce/gate weight as -S*W
 * (S returned); runtime C_out=R*LUT[idx(-S*x)] and fn(x)=C_out*out_scale. fn(x,ctx) carries params via ctx
 * (NULL for parameter-free fns). Fills lut[1030]. 0/ok, -2 fail. The silu/rsqrt builders below are wrappers. */
int          ork_mm_build_f16_lut(ork_npu *ctx, double (*fn)(double, void *), void *fnctx,
                                  double in_lo, double in_hi, short *lut,
                                  double *S_out, double *R_out, double *out_scale_out);
/* FUSED matmul + output-stage activation: C[M,N] = fn(A·B) in ONE submit — the activation rides the matmul's
 * DPU output stage (no separate submit, no CPU<->NPU crossing). This is the "no-crossing chain" that makes an
 * on-NPU non-matmul op (softmax exp / RMSNorm rsqrt / SwiGLU silu) a WIN instead of a submit-floor-bound loss.
 * fn(x,ctx) is the activation; the matmul output must fall in [in_lo,in_hi] (the LUT band). K%32<=2048,
 * N%16<=nmax. rk3588 PPU-fuse-gated. 0/ok, -2 shape/SoC(PPU off), -1 wedge/alloc. */
int          ork_mm_run_f16_act(ork_npu *ctx, int K, int N, const ork_f16 *B, int M, const ork_f16 *A, float *C,
                                double (*fn)(double, void *), void *fnctx, double in_lo, double in_hi);
/* PACK-ONCE / RUN-MANY split of the fused activation, for RESIDENT reuse (fused exp/rsqrt/silu in a seq): pack
 * bakes the calibrated LUT + out-scale into the weight (ork_mm_pack_f16_fused_act), run replays it with no
 * re-pack (ork_mm_run_f16_fused_act, C[M,N]=fn(A·B), one submit). Same single-signed band rule as _act; the
 * run path carries no fn pointer (crosses a seq/socket). NULL/-2 on bad shape/mixed-sign/no-baked-LUT. */
ork_w       *ork_mm_pack_f16_fused_act(ork_npu *ctx, int K, int N, const ork_f16 *B,
                                       double (*fn)(double, void *), void *fnctx, double in_lo, double in_hi);
int          ork_mm_run_f16_fused_act(ork_npu *ctx, ork_w *w, int M, const ork_f16 *A, float *C);
/* Calibrate the fp16 fused SiLU for a gate spanning [-Gmax,Gmax] (caps Gmax → the fp16 spread band, then
 * ork_mm_build_f16_lut). silu(gate)=C_out*out_scale; pack the gate weight as -S*W. See Exp-2026-07-05. */
int          ork_mm_build_f16_silu_lut(ork_npu *ctx, double Gmax, short *lut,
                                       double *S_out, double *R_out, double *out_scale_out);
/* Calibrate the fp16 fused rsqrt for on-NPU rmsnorm(n_feat=n)/l2norm(n_feat=1): a reduce-matmul packed as
 * -S*W emits scale=1/sqrt(ss/n_feat+eps)=C_out*out_scale for ss=sum(x^2) in [ss_min,ss_max]. Build once per
 * layer/range + cache (runs probe submits). PRECISION VARIANTS: this is the fp16, fused-into-the-reduce
 * rsqrt; the int8/int16 STANDALONE rsqrt of a tensor is ork_npu_rsqrt_i8/i16 (act_lut_i8/i16, same rsqrt_f).
 * Same op, different precision + fusion regime — both kept. */
int          ork_mm_build_f16_rsqrt_lut(ork_npu *ctx, int n_feat, double eps, double ss_min, double ss_max,
                                        short *lut, double *S_out, double *R_out, double *out_scale_out);
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
/* int8 matmul with INT16 requant output (set_i16_out): C=clamp_i16(round((A·W)*mult/2^shift)) [M*N] int16,
 * COMPACT-LINEAR — feeds an int16 SDP seq op resident with no PC-chain/layout bridge. K%512. */
int          ork_mm_run_i8_out16(ork_npu *ctx, ork_w *w, int M, const int8_t *A, short *C, int mult, int shift);
/* int4 (W4A4): A int4 ([-8,7] in int8, row-major), C int32 raw sum — apply scales:
 * C_real[m][n] = aScale[m]*bScale[n]*C[m][n]. Run dtype must match the pack dtype. 0 ok / negative err. */
int          ork_mm_run_i4(ork_npu *ctx, ork_w *w, int M, const int8_t  *A, int32_t *C);

/* Async pipelined submit (orkd client + ring only; ORK_USE_ORKD=1 ORK_ORKD_RING=1). ork_mm_submit enqueues one
 * matmul for w (any precision — dtype taken from w) WITHOUT blocking and returns a ticket; ork_mm_collect(ticket)
 * copies C when it lands. Returns <0 if unavailable (no ring, or the op is too big for a ring slot — use the
 * synchronous ork_mm_run* which auto-falls back to the socket). Keeping several ops in flight (up to the ring
 * depth) overlaps each op's transport with the NPU compute of the ones ahead — the decode-pipeline path.
 * A is int8/int4 (1B/elem) or fp16 (2B/elem); C is int32/fp32 (4B/elem), sized M*N by the caller. */
int          ork_mm_submit(ork_npu *ctx, ork_w *w, int M, const void *A);
int          ork_mm_collect(ork_npu *ctx, int ticket, void *C);
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

/* PHASE 1 (#35 chained FFN): int8 matmul with the INT16-requantized output stage (set_i16_out).
 * out_i16 = clamp_i16(round(acc_i32 * mult / 2^shift)); identity = (0x4000,14). C[M*N] int16 out
 * (raw device layout). The RE crux for the on-NPU matmul->int16-silu handoff. 0/ok, -1 wedged, -2 dims. */
int          ork_npu_probe_i16_out(ork_npu *ctx, int M, int K, int N, const int8_t *A, const int8_t *B,
                                   int mult, int shift, short *C, double *us);

/* ork-NATIVE fused-SiLU LUT generator: build ork's OWN silu LUT for the fused-output path (no RKNN
 * dependence). Measures ork's index(acc) for (r_mult,r_shift,cfg4068) via one calibration submit, then
 * builds lut[1030] = silu curve matched to ork's mapping for (in_scale,out_scale). Do this ONCE per
 * register config; run matmuls via ork_npu_probe_i8_silu_cfg(..,r_mult,r_shift,0,0xffffc000,cfg4068,lut,1030,..).
 * Pick R=r_mult/2^r_shift ~= 660*in_scale so the acc range spans silu's transition. Validated ~1 int8.
 * 0/ok, -1 fail. See tools/silu_native.c. */
int          ork_mm_silu_build_lut(ork_npu *ctx, double in_scale, double out_scale,
                                   int r_mult, int r_shift, uint32_t cfg4068, int16_t *lut);
/* Fused EXP LUT for the coalesced chain (softmax): HW-chains exp onto the score matmul via run_chain_i8_gsilu.
 * Scores must be <=0 (post-max domain). Same signature/calibration as the silu LUT. 0/ok, -1 fail. */
int          ork_mm_chain_build_exp_lut(ork_npu *ctx, double in_scale, double out_scale,
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

/* On-NPU normalization primitives (fp16 in/out), per row of an [M][n] tensor:
 *   rmsnorm: out = x / sqrt(mean(x^2)+eps) * w      (w[n] weight)
 *   l2norm:  out = x / sqrt(sum(x^2)+eps)           (no mean, no weight — the GDN q/k normalize)
 * GATED (default OFF → computed on CPU, bit-exact via the fp32 NEON refs, so callers get a stable
 * entrypoint today). A fully on-NPU norm needs the PPU transcendental (rsqrt) whose regcmd capture was
 * blocked by the interposer's 4KB buffer-dump truncation; with that fixed the PPU-native path can be
 * captured and slotted in here. ORK_NORM_NPU=1 selects the NPU path once it exists (today a no-op
 * selector — still CPU-correct). out may alias x. 0/ok, -1 alloc, -2 bad args. */
int          ork_npu_rmsnorm_f16(ork_npu *ctx, int M, int n, const ork_f16 *x, const ork_f16 *w, float eps, ork_f16 *out);

/* NEOX RoPE on the NPU (rope type 2). x[nrow][hd] fp16 row-major (nrow = heads*tokens; each row a head_dim
 * vector); pos[nrow] = per-row token position. Composed as x⊙COS + rot_half(x)⊙SIN (2 ewmul + 1 add on NPU).
 * hd even + multiple of 8. 0/ok, <0 on failure. Keeps Q/K rotation on the NPU data path (attention chaining). */
int          ork_npu_rope_neox_f16(ork_npu *ctx, const ork_f16 *x, int hd, int nrow, const int *pos, double freq_base, ork_f16 *out);
int          ork_npu_l2norm_f16 (ork_npu *ctx, int M, int n, const ork_f16 *x,                     float eps, ork_f16 *out);

/* CHAIN ASSEMBLER: one pre-built program in a heterogeneous PC-chain (see ork_npu_chain_progs).
 * rc/nwords = the program's regcmd words (caller-built, with its own buffer addresses); enable_mask/
 * regcfg_amount = its rknpu_task fields (matmul 0xd/108, SDP 0x18/varies); desc_slot = the word index of
 * this program's PC next-descriptor (where 0x0010/0x0101 next-addr is WRITTEN — matmul=216); -1 if the op
 * carries no descriptor slot (then it can only be the LAST program). */
typedef struct { const uint32_t *rc; int nwords; unsigned enable_mask; int regcfg_amount; int desc_slot; } ork_chain_prog;
/* Submit N pre-built programs as ONE PC-chain (task_number=N, single ioctl) — chains a whole NPU-only
 * run (attention block / FFN inner) into one submit. Non-last programs need a PC next-descriptor slot in
 * their regcmd (matmul has one; a program lacking it can only be last). 0/ok, -2 bad-args/no-slot, -1 wedge. */
int          ork_npu_chain_progs(ork_npu *ctx, int n, const ork_chain_prog *progs, int dom);
/* PROBE: chain two int8 SDP ewmuls (ewmul0 middle @desc_slot=138 -> ewmul1) via ork_npu_chain_progs and verify
 * both vs CPU ref. *t0_ok = middle SDP op correct carrying a forward descriptor; *t1_ok = chain walked forward
 * through the SDP slot. 0/ok (see *t0_ok/*t1_ok), -3 no-PPU, -2 alloc, -1 wedge. Proves int8 SDP HW-chains. */
int          ork_npu_probe_sdp_chain_fwd(ork_npu *ctx, int *t0_ok, int *t1_ok);
/* STAGE-1 PROBE: [matmul->ewmul(SDP middle)->matmul] NONBLOCK chain on the begin_mc recipe (warmed scratch +
 * clean-before), completion via the terminal matmul sentinel. *ok = all three outputs bit-exact. 0/ok,-1/-2/-3. */
int          ork_npu_probe_seq_hetero(ork_npu *ctx, int *ok);
/* Self-test: chain 2 plain int8 matmuls (all-ones) and verify BOTH tasks execute. *t0_cnt/*t1_cnt = count
 * of M*N int32 slots == K or 2K (near M*N => that task ran). Validates chain_progs w/ a real task0. */
int          ork_npu_chain_selftest(ork_npu *ctx, int *t0_cnt, int *t1_cnt);

/* CHAIN ASSEMBLER increment-1: data-connected int8-matmul(int16-out) -> int16-silu in ONE PC-chain (the
 * matmul's int16 output buffer IS the silu's input). Validates the intermediate-buffer bridge. Computes
 * out = clamp_i16(silu(requant_i16(A[M,K]xB[K,N])*in_scale)/out_scale). gate_out (nullable) = the matmul
 * int16 output read back via EWCUBEH, to localize a mismatch. 0/ok,-1 wedge,-2 dims,-3 SoC. rk3588-gated. */
int          ork_npu_chain_gatesilu_i16(ork_npu *ctx, int M, int K, int N, const signed char *A, const signed char *B,
                                        int mult, int shift, double in_scale, double out_scale,
                                        short *gate_out, short *out, double *us);

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
/* RE: replay the captured vendor forward-softmax 9-task PC-chained graph verbatim (capture geometry:
 * reduction=64, 256 rows). in/out = raw 32768-byte buffer images; fill `in` uniform -> softmax=1/64. */
int          ork_npu_replay_softmax_f16(ork_npu *ctx, const void *in, void *out, double *us);

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
/* PHASE 0 chained-FFN probe: a heterogeneous 2-task PC-chain (int8 matmul -> int16 silu) in ONE submit,
 * to prove the hardware walks the matmul->pure-SDP transition without a per-op host round-trip. `out` gets
 * silu(in) (verifies the silu task ran); *mm_ran is set if the chained matmul task also ran. 0/ok,-1,-2,-3. */
int          ork_npu_chain_mm_silu_i16(ork_npu *ctx, const short *in, int M, int N,
                              double in_scale, double out_scale, short *out, int *mm_ran, double *us);

/* On-NPU element-wise ADD (int8): out = clamp_i8(round( (a*a_scale + b*b_scale)/out_scale )) via the 2-input SDP
 * ALU=add op. Symmetric quant. Residual add (a_scale==b_scale==out_scale) => out=clamp_i8(a+b), bit-exact.
 * in/out int8 [M*N], N%16==0; rk3588-gated. 0/ok,-1 wedged,-2 shape,-3 SoC. */
/* On-NPU per-row MAX-REDUCE (int8): out[m]=max_n a[m*N+n]. Batched pairwise-max tree on the SDP EW ALU
 * (EW_ALU_ALGO=MAX). N%16, reduces N->16 on-NPU + 16-wide CPU tail. Reusable: softmax max, max-pool, top-k. */
int          ork_npu_row_max_i8(ork_npu *ctx, const signed char *a, int M, int N, signed char *out, double *us);
/* On-NPU PER-CHANNEL scale (int8): out[m][n]=clamp(a[m][n]*b[n]*mult>>(shift-14)); b[N] broadcast across rows
 * (EW operand per-channel mode, ERDMA_DATA_MODE=0). N%16. Reusable: softmax normalize, LayerNorm affine, requant. */
int          ork_npu_mul_perchan_i8(ork_npu *ctx, const signed char *a, const signed char *b, int M, int N, int mult, int shift, signed char *out, double *us);
/* fp16 per-channel scale: out[m][n]=a[m][n]*b[n], b[N] broadcast across rows (ERDMA_DATA_MODE=0). N%8. Quant-free. */
int          ork_npu_mul_perchan_f16(ork_npu *ctx, const ork_f16 *a, const ork_f16 *b, int M, int N, ork_f16 *out, double *us);
/* int16 per-channel scale: out[m][n]=clamp_i16(a[m][n]*b[n]*mult>>shift), b[N] broadcast. N%8. Chain intermediate. */
int          ork_npu_mul_perchan_i16(ork_npu *ctx, const short *a, const short *b, int M, int N, int mult, int shift, short *out, double *us);
/* Tier 11 doorbell pipeline profiler: N serial int8 matmuls, BLOCKING vs NONBLOCK + DRAM-doorbell busy-poll.
 * Fills per-op us for each + a correctness flag each. See Optimization-Roadmap Tier 11. */
int          ork_npu_doorbell_prof(ork_npu *ctx, int M, int K, int N, int iters, double *block_us, double *nb_us, int *ok_block, int *ok_nb);
/* Tier 11 overlap probe: CPU router (cpu_reps x 512x512 fp32 GEMV) in the shadow of an async NPU op.
 * Fills npu_solo / cpu_solo / overlap_wall (us/iter). hidden% = (npu_solo+cpu_solo-overlap_wall)/min(...). */
int          ork_npu_overlap_prof(ork_npu *ctx, int M, int K, int N, int cpu_reps, int iters, double *npu_solo, double *cpu_solo, double *overlap_wall, int *ok);
/* RE (WIP): full-chain replay of vendor gemm+reshape (task0-10); returns gemm output (contiguous) + reshape
 * output (atom-8) so caller verifies reshape_out==atom8(gemm_out). Loads gemm_mul_image.bin. See RESHAPE_WIP.md. */
int          ork_npu_replay_reshape_f16(ork_npu *ctx, unsigned short *gemm_raw, int gemm_words, unsigned short *reshape_raw, int reshape_words, double *us);
/* RE (WIP): vendor fp16 contiguous->atom-8 RESHAPE base op (task4) with a constructed permutation weight.
 * Reads the output RAW for layout inspection. N=64/M<=8 only (captured geometry). See RESHAPE_WIP.md. */
int          ork_npu_reshape_probe_f16(ork_npu *ctx, int M, int N, const unsigned short *src, unsigned short *out_raw, int out_words, double *us);
/* LOOPBACK Pass-2: standalone SDP reads INT32 accumulator from DRAM, per-channel scale + requant -> int16.
 * out[m][n]=clamp_i16(a_i32[m][n]*b[n]*mult>>shift). Routes around the broken CNA->DPU requant-WDMA. */
int          ork_npu_requant_perchan_i32(ork_npu *ctx, const int *a, const short *b, int M, int N, int mult, int shift, short *out, double *us);
/* CHAIN: int8-matmul(int16-out) -> per-channel-scale in ONE PC-chain (A·V->normalize building block for the
 * single-submit attention chain). out=clamp_i16(requant(A[M,K]xB[K,N],m1,s1)*scale[n]*m2>>s2). K%32,N%32,M<=64. */
int          ork_npu_chain_mm_perchan_i16(ork_npu *ctx, int M, int K, int N, const signed char *A, const signed char *B,
                                          const short *scale, int m1, int s1, int m2, int s2, short *out, double *us);
int          ork_npu_chain_mm_perchan_f16(ork_npu *ctx, int M, int K, int N, const unsigned short *A, const unsigned short *B,
                                          const unsigned short *scale, unsigned short *out, double *us);
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
/* exp with a scalar GLOBAL-max subtract baked into the LUT: out = clamp_i8(round( exp((in-max)*in_scale)/out_scale )).
 * Softmax numerator with stable max-subtract but NO per-row op (max = scalar global max in int8-input units). */
int          ork_npu_exp_i8_biased(ork_npu *ctx, const signed char *in, int M, int N, double in_scale, double out_scale, double max, signed char *out, double *us);
int          ork_npu_exp_i16(ork_npu *ctx, const short *in, int M, int N, double in_scale, double out_scale, short *out, double *us);
/* Composed fused softmax over each row of [M][n] (fp16): out = exp(x-max)/Σexp(x-max). Gated ORK_SOFTMAX_NPU:
 * exp (ork_npu_exp_i16) + the Σ reduction (reduce-matmul) run on the NPU, max + normalize on CPU; falls back
 * to a full CPU softmax when the gate/PPU-fuse is off or n%32!=0. 0/ok, -2 bad args. */
int          ork_npu_softmax_f16(ork_npu *ctx, int M, int n, const ork_f16 *x, ork_f16 *out);

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
/* RE fuzzer hooks (tools/i4_multim_fuzz.c): queue arbitrary (block,reg,val) overrides applied at the end of
 * every synth_i4 (int4 regcmd). Inert until added; clear before each probe. Sweep the multi-M K-schedule space. */
void         ork_i4_fuzz_clear(void);
void         ork_i4_fuzz_add(unsigned int blk, unsigned int reg, unsigned int val);
/* int8 analogs (batch-mode RE): overrides applied at the end of synth_i8; raw int32 multi-M output probe
 * (2*M*N int32, room for a stride-2 batch layout). See tools/i4_multim_fuzz.c int8 modes. */
void         ork_i8_fuzz_clear(void);
void         ork_i8_fuzz_add(unsigned int blk, unsigned int reg, unsigned int val);
int          ork_npu_probe_i8_mm(ork_npu *ctx, int M, int K, int N,
                                 const signed char *A, const signed char *B, int *raw);
/* fp16 analogs (batch-mode mapping): A/B are fp16 bit patterns (uint16), output raw fp32 (2*M*N floats). */
void         ork_f16_fuzz_clear(void);
void         ork_f16_fuzz_add(unsigned int blk, unsigned int reg, unsigned int val);
int          ork_npu_probe_f16_mm(ork_npu *ctx, int M, int K, int N,
                                 const unsigned short *A, const unsigned short *B, float *raw);
int          ork_npu_probe_f16_mm_f16out(ork_npu *ctx, int M, int K, int N,
                                 const unsigned short *A, const unsigned short *B, unsigned short *out);
int          ork_npu_probe_f16_stridedA(ork_npu *ctx, int M, int K, int N, const unsigned short *A, int apitch,
                                 const unsigned short *B, unsigned short *out);
int          ork_npu_mm_perchan_f16_fused(ork_npu *ctx, int M, int K, int N, const unsigned short *A,
                                 const unsigned short *B, const unsigned short *scale, unsigned short *out);
int          ork_npu_mul_perchan_f16_contig(ork_npu *ctx, const ork_f16 *a, const ork_f16 *b, int M, int N,
                                 ork_f16 *out, double *us);
int          ork_npu_mm_perchan_f16(ork_npu *ctx, int M, int K, int N, const unsigned short *A,
                                 const unsigned short *B, const unsigned short *scale, unsigned short *out, double *us);
int          ork_npu_mm_perchan_f16_diag(ork_npu *ctx, int M, int K, int N, const unsigned short *A,
                                 const unsigned short *B, const unsigned short *scale, unsigned short *out, double *us);

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

/* ---- Dynamic steered submission (NONBLOCK chain + per-op doorbell progress + mid-flight halt) ----
 * Submit an S-task int8 OR fp16 chain NONBLOCK, then watch/steer it from the host. v1: M=1/task (mc: M<=64),
 * single-slice conforming K (K%512==0, K<=4096). C is a resident ork_dma_alloc buffer (zero-copy direct
 * output) or host memory (copy-back); A is HOST (malloc) memory — begin_mc STAGES A via memcpy and a
 * zero-copy DMA-A source's CPU writes are not coherently readable by that staged read (partial-K sums).
 * Enables early-exit-to-free-the-NPU and runtime observability without a kernel round-trip per chain.
 * (See tools/ork_dyn_test.c: D=int8, E=fp16.) */
typedef struct ork_dyn_chain ork_dyn_chain;
size_t        ork_npu_sram_total(ork_npu *ctx);   /* NPU on-chip SRAM bytes (0 = none: stock kernel/DTB) */
size_t        ork_npu_sram_free (ork_npu *ctx);   /* free NPU SRAM bytes now (confirms ORK_WEIGHT_SRAM placement) */
uint64_t      ork_npu_dma_rw    (ork_npu *ctx);   /* cumulative NPU DMA rw bytes; delta across a submit = HW did work (0 => never dispatched) */
void          ork_npu_dump_state(ork_npu *ctx, const char *label);   /* snapshot NPU state (freq/volt/DMA counters) to stderr on anomaly, before a wedge destroys it */
int           ork_npu_soft_reset(ork_npu *ctx);   /* RKNPU_ACT_RESET + force re-warm; recovery step after a dump so a stuck job doesn't accumulate into a hard wedge */
int           ork_npu_recover   (ork_npu *ctx, const char *label);   /* self-heal: dump + soft-reset + dummy-op probe; 1 = recovered (continue), 0 = still broken (throw fault) */
int           ork_npu_force_fault(ork_npu *ctx);   /* DIAGNOSTIC: deliberately force a reliable NPU fault (bogus weight addr -> DMA fault); 0 = faulted as intended */
ork_dyn_chain *ork_dyn_begin(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks);   /* NONBLOCK-submit (single-core chain); NULL on bad args */
ork_dyn_chain *ork_dyn_begin_mc(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks, int nc); /* NONBLOCK across nc cores (nc<=0=all); halt/append N/A */
int          ork_dyn_progress(ork_dyn_chain *h);                                  /* highest completed op idx, -1 none */
void         ork_dyn_dump(ork_dyn_chain *h, const char *label);                   /* chain-aware anomaly dump: names the STUCK descriptor (progress+1) + its regcmd/addr/doorbell + hw_elapse */
int          ork_dyn_halt(ork_dyn_chain *h, int at);                              /* halt after op `at` (free NPU early) */
int          ork_dyn_end(ork_dyn_chain *h);                                       /* drain + writeback + free; ret highest done */
int          ork_dyn_max_steps(void);                                             /* per-chain step cap (split longer work across chains) */
int          ork_dyn_steps(ork_dyn_chain *h);                                     /* total steps submitted in this chain */
int          ork_dyn_remaining(ork_dyn_chain *h);                                 /* steps not yet completed (budget left before the chain ends) */
int          ork_dyn_append(ork_dyn_chain *h, const ork_mm_task_i8 *task);        /* extend a running chain in-flight (wrap); 1=too late, 0=ok, <0=err */
int          ork_dyn_spin_probe(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks, int spin_us, int *spin_alive); /* circular-spin keep-alive + redirect probe */
/* EXPERIMENTAL int4 NONBLOCK-doorbell probe: mirrors ork_mm_run_chain_i4 (M=1 int4 PC-chain, host A) but
 * flips the submit to NONBLOCK (0x2) + polls an int16 output-sentinel to completion, then de-tiles int16->
 * int32 into each task->C. Answers the load-bearing question of whether the int4 (int16-output) datapath
 * survives the doorbell's non-blocking sentinel poll. Returns 0/ok, <0 err. (Not a production path.) */
int          ork_dyn_i4_probe(ork_npu *ctx, int S, const ork_mm_task_i4 *tasks);
/* Submit QUEUE: chunk-pipeline over the dynamic API — accumulate tasks, run a chunk NONBLOCK while the
 * caller does other work (CPU‖NPU decode split), auto-split work > chunk_max into successive clean chunks. */
typedef struct ork_dyn_queue ork_dyn_queue;
ork_dyn_queue *ork_dyn_queue_create(ork_npu *ctx, int chunk_max, int ncore);      /* chunk_max<=0=>max_steps; ncore<=1 single-core, >1 multi-core NONBLOCK */
void         ork_dyn_queue_set_linger(ork_dyn_queue *q, int us);                  /* coalesce window; default = one submit floor */
int          ork_dyn_queue_linger_us(ork_dyn_queue *q);
int          ork_dyn_queue_push(ork_dyn_queue *q, const ork_mm_task_i8 *task);    /* enqueue a matmul */
int          ork_dyn_queue_flush(ork_dyn_queue *q);                               /* submit next chunk NONBLOCK (NPU starts) */
int          ork_dyn_queue_pending(ork_dyn_queue *q);                             /* tasks not yet submitted */
int          ork_dyn_queue_idle(ork_dyn_queue *q);                                /* on idle+linger-elapsed, halt a flying reserved/spin chain early (null-terminate); 1 if halted */
int          ork_dyn_queue_drain(ork_dyn_queue *q);                               /* finish all chunks + writeback; ret total ops */
void         ork_dyn_queue_destroy(ork_dyn_queue *q);
/* Precompiled-program cache (regime A: fixed chain, pinned buffers). Compile the chain ONCE, then re-run it
 * every token with only the activation contents refreshed — no per-token synth/validate. */
typedef struct ork_pc_chain ork_pc_chain;
ork_pc_chain *ork_pc_compile(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks);   /* build the program pool once; NULL on bad args */
int          ork_pc_run(ork_pc_chain *pc);                                        /* refresh A + NONBLOCK submit + drain; ret highest op */
void         ork_pc_free(ork_pc_chain *pc);

/* ---- Heterogeneous op-sequence scheduler (ork_submit_seq) ----------------------------------------
 * Ingest a mixed sequence of NPU ops (any precision, matmul or SDP/activation) and run it correctly by
 * routing each op to the ONE execution model it is reliable on. Different op types are reliable on
 * different models (a hard, exhaustively-established finding): int8 AND fp16 matmul with conforming K
 * (K%512==0 && K<=4096, M<=64, Sn==1; fp16 also M*K<=32768) are bit-exact on the thread-free HW-chain
 * DOORBELL (ork_dyn_begin_mc, NONBLOCK poll) — fp16 requires host (malloc) A, the same convention int8
 * uses. int4, non-conforming int8/fp16, and SDP/activation ops are NOT doorbell-reliable but ARE reliable
 * on the SW-chain / thread-pool + blocking-completion model (run_stream_f16 / run_stream_i4 / the SDP fns).
 *
 * The scheduler batches maximal runs of consecutive HW-chainable ops into ONE doorbell submit and BREAKS
 * the chain to the SW model at every op that isn't. A doorbell run is ONE dtype + ONE domain (begin_mc's
 * requirement), so the run also breaks at a dtype change (i8<->f16) — each dtype gets its own doorbell and
 * begin_mc's internal ork_npu_enter fires the mode transition at the boundary. Each break's precision/chain
 * mode transition is handled by the driver's table-driven ork_npu_enter/XSPEC layer — so op classification
 * is DATA (a per-op-kind row of {hw, marker, XSPEC profile, ork_chain_kind, dispatch}), not branches.
 * Adding a new precision/op is one more row (an hw=1 row rides the doorbell when seq_hw_ok() accepts it and
 * falls back to its SW dispatch fn otherwise); the scheduler loop is unchanged regardless of how many kinds.
 *
 * PHASE 1 (now): CORRECTNESS + RELIABILITY of a mixed sequence across the HW<->SW transitions. No
 * CPU/NPU overlap yet (that is phase 2 — see tools/test_submit_seq.c). Ops execute in order. */
typedef enum {
    ORK_OP_MM_I8 = 0,   /* int8 matmul:  w=DT_I8 weight, A int8[M,K], C int32[M,N]  (HW doorbell if K conforms) */
    ORK_OP_MM_F16,      /* fp16 matmul:  w=DT_F16 weight, A fp16[M,K] (HOST mem), C fp32[M,N]  (HW doorbell if K conforms & M*K<=32768, else SW run_stream_f16) */
    ORK_OP_MM_I4,       /* int4 matmul:  w=DT_I4 weight, A int8[M,K] (HOST mem), C int32[M,N]  (HW doorbell if M==1 & single-slice, else SW run_stream_i4) */
    ORK_OP_SILU_F16,    /* fp16 SiLU activation (SDP): A fp16[M,N] -> C fp16[M,N]   (SW SDP; needs LUT — TODO row) */
    ORK_OP_EWMUL_F16,   /* fp16 elementwise mul (SDP): A*B fp16[M,N] -> C fp16[M,N] (SW: ork_npu_ewmul_f16) */
    ORK_OP_SILU_I8,     /* int8 SiLU (SDP): A i8[M,N] -> C i8[M,N] (in_scale,out_scale; SW: ork_npu_silu_i8) */
    ORK_OP_GELU_I8,     /* int8 GELU (SDP): A i8[M,N] -> C i8[M,N] (in_scale,out_scale; SW: ork_npu_gelu_i8) */
    ORK_OP_EWMUL_I8,    /* int8 elementwise mul (SDP): A*B i8[M,N] -> C i8[M,N] (mult,shift; SW: ork_npu_ewmul_i8) */
    ORK_OP_ADD_I8,      /* int8 elementwise add (SDP): A+B i8[M,N] -> C i8[M,N] (a_scale=in,b_scale,out_scale; SW: ork_npu_add_i8) */
    ORK_OP_ADD_F16,     /* fp16 elementwise add (SDP): A+B f16[M,N] -> C f16[M,N] (SW: ork_npu_add_f16) */
    ORK_OP_SILU_I16,    /* int16 SiLU (SDP): A i16[M,N] -> C i16[M,N] (in_scale,out_scale; SW: ork_npu_silu_i16) — the ACCURATE higher-precision SiLU (fp16 SiLU is not viable on this NPU: fused=garbage-PPL, standalone SDP=broken) */
    /* --- ops beyond the seq-scheduler subset (values >=11 append-only, ABI-stable). These are supported,
     * PROVEN ops (each backed by a working probe/example — see OPS_REGISTRY.md) that the SDK addresses by
     * enum + industry name (ork_op_name / ork_op_from_name); not all are seq-dispatchable (their SEQ_CLASS
     * row has fn=NULL). "op = the meaning"; the regcmd impl is chosen by ork_impl_mode. --- */
    ORK_OP_MM_I4_GROUPED,          /* float-grouped int4 matmul (ork_mm_run_i4_grouped)             example: i4 / int4_bench */
    ORK_OP_GELU_I16,               /* int16 GELU (SDP; ork_npu_gelu_i16)                            example: test_gelu */
    ORK_OP_RSQRT_I8,               /* int8 rsqrt (SDP; ork_npu_rsqrt_i8)                            example: test_gelu / op_profile */
    ORK_OP_EXP_I8,                 /* int8 exp (SDP; ork_npu_exp_i8)                                example: test_gelu / op_profile */
    ORK_OP_EXP_I16,                /* int16 exp (SDP; ork_npu_exp_i16)                              example: test_ssd_chunk_npu / mode_probe */
    ORK_OP_MUL_I16,                /* int16 elementwise mul (SDP; ork_npu_ewmul_i16) — EXPERIMENTAL example: test_ewmul_i16 */
    ORK_OP_ADD_I16,                /* int16 elementwise add (SDP; ork_npu_add_i16) — EXPERIMENTAL   example: test_add / add16_probe */
    ORK_OP_MUL_PERCHANNEL_I8,      /* per-channel scale int8 (SDP; ork_npu_mul_perchan_i8)          example: bs_scale_probe */
    ORK_OP_MUL_PERCHANNEL_F16,     /* per-channel scale fp16 (SDP; ork_npu_mul_perchan_f16)         example: bs_scale_probe */
    ORK_OP_MUL_PERCHANNEL_I16,     /* per-channel scale int16 (SDP; ork_npu_mul_perchan_i16)        example: bs_scale_probe */
    ORK_OP_MATMUL_PERCHANNEL_F16,  /* fp16 matmul -> per-channel scale (ork_npu_mm_perchan_f16)     example: mm_perchan_f16_probe */
    ORK_OP_REQUANTIZE_PERCHANNEL_I32,/* int32->int16 per-channel requant (ork_npu_requant_perchan_i32) — PARTIAL  example: requant_i32_probe */
    ORK_OP_SOFTMAX_F16,            /* fp16 softmax (replay; ork_npu_replay_softmax_f16)             example: softmax_probe / softmax_replay */
    ORK_OP_REDUCEMAX_I8,           /* int8 row-max reduction (ork_npu_row_max_i8)                   example: max_reduce_probe */
    ORK_OP_RESHAPE_F16,            /* fp16 reshape/permute (ork_npu_replay_reshape_f16)             example: reshape_probe */
    ORK_OP_ROPE_NEOX_F16,          /* fp16 NEOX RoPE (ork_npu_rope_neox_f16)                        example: rope_probe */
    ORK_OP_MATMUL_SILU_I8,         /* int8 matmul + fused SiLU output stage (ork_mm_run_i8_silu)    example: fused_silu_test */
    ORK_OP_MATMUL_REQUANT_I8,      /* int8 matmul + int8 requant output stage (ork_mm_run_i8_out8)  example: fused_ffn_probe */
    /* --- primitive ops the initial derivation missed (found by the 2026-07-22 completeness sweep) --- */
    ORK_OP_MATMUL_SILU_I32,        /* int8 matmul + fused SiLU, INT32 output (un-requantized; ork_mm_run_i8_silu32) example: silu32_check */
    ORK_OP_RMSNORM_F16,            /* fp16 RMSNorm — every transformer layer (ork_npu_rmsnorm_f16)  example: test_bmm */
    ORK_OP_L2NORM_F16,             /* fp16 L2 normalize (ork_npu_l2norm_f16)                        example: test_bmm */
    ORK_OP_RSQRT_I16,              /* int16 rsqrt (SDP; ork_npu_rsqrt_i16) — RMSNorm 1/√ + softmax 1/Σ  example: silu_i16 family (rsqrt/exp/gelu/silu int16 LUT) */
    ORK_OP_MM_F16_F16OUT,          /* fp16 matmul with CONTIGUOUS fp16 output (ork_mm_run_f16_f16out) — A1 bridge: feeds an fp16 SDP op with no f32→f16 narrow  example: f16out_probe */
    ORK_OP_MATMUL_I16OUT_I8,       /* int8 matmul with INT16 compact-linear output (ork_mm_run_i8_out16) — feeds an int16 SDP op resident, no PC-chain  example: i16out_seq_probe */
    ORK_OP_NKIND
} ork_seq_kind;
typedef ork_seq_kind ork_op;       /* canonical SDK op enum; ork_seq_kind is the historical name (the seq scheduler is one consumer) */

/* Execution mode selecting WHICH regcmd implementation of an op runs. The SDK submits an op/composite by
 * enum + mode; the driver resolves the regcmd (a hw-chained op and a standalone op share one ork_op value
 * but different regcmd byte templates). */
typedef enum {
    ORK_IMPL_STANDALONE = 0,   /* one op, its own submit (blocking or doorbell) */
    ORK_IMPL_HW_CHAINED,       /* op rides a HW PC-chain / doorbell-seq alongside others */
    ORK_IMPL_SW_CHAINED,       /* op runs in a software-broken sequence (ork_submit_seq SW path) */
    ORK_IMPL_NMODE
} ork_impl_mode;

/* Industry-standard op name <-> enum resolution (SDK-exported). Names follow ONNX where a primitive exists
 * (matmul/mul/add/softmax/gelu/exp/reshape/reducemax) and community LLM conventions otherwise (silu/rsqrt/
 * rope/requantize/perchannel), dtype as a C-identifier suffix (_i8/_i16/_i4/_f16). */
const char  *ork_op_name(ork_op k);              /* enum -> "silu_i16"; NULL if out of range */
ork_op       ork_op_from_name(const char *name); /* "silu_i16" -> enum; ORK_OP_NKIND if unknown */
const char  *ork_impl_mode_name(ork_impl_mode m);/* enum -> "hw_chained"; NULL if out of range */

/* Composite (multi-op) primitives — one submit combining several ops. Supported set derived from PRE-SESSION
 * examples (origin/main) only; DEAD/unvalidated chains (chain_mm_perchan_i16 = hangs, chain_gatesilu_i16 =
 * no pre-session example) are intentionally absent. */
/* NAMING CONVENTION (hybrid — this NPU is not industry-standard and has severe limitations, so we borrow
 * standards where they map and use NPU-specific qualifiers where they don't):
 *   - WEIGHTED ops (matmul / FFN): WxAy quantization notation (industry-standard) — W<weight-bits>A<act-bits>,
 *     e.g. w8a8 (int8/int8), w16a16 (fp16), w4a4 (int4). This is the speed/quality axis.
 *   - MIXED precision: append a component qualifier naming the higher-precision (sensitive) part, e.g.
 *     `_f16gate` (fp16 gate matmul while the rest is int8), `_i16silu` (int16 SiLU). This is how a
 *     speed-for-quality optimization is expressed: base WxAy + which component is kept precise.
 *   - SDP / elementwise ops (no weights): WxAy does not apply → keep the dtype suffix (_i8/_i16/_f16).
 *   - Non-op mechanisms named for what they do (graph_replay), not the internal fn (was "replay"/"bmm"). */
typedef enum {
    ORK_COMPOSITE_CHAIN_MATMUL_W8A8 = 0,      /* batch of independent int8 matmuls (ork_mm_run_chain_i8)              example: chain_gu_silu_probe */
    ORK_COMPOSITE_FFN_SWIGLU_W8A8,            /* int8 SwiGLU FFN inner (ork_mm_run_chain_i8_ffn)                     example: chain_gu_silu_probe */
    ORK_COMPOSITE_FFN_GATE_SILU_W8A8,         /* FFN, fused gate-SiLU output stage (ork_mm_run_chain_i8_gsilu)       example: chain_gu_silu_probe */
    ORK_COMPOSITE_FFN_GATE_SDPSILU_W8A8,      /* FFN, gate matmul + standalone SDP SiLU (ork_mm_run_chain_i8_sdpsilu) example: chain_gu_silu_probe */
    ORK_COMPOSITE_CHAIN_MATMUL_PERCHANNEL_W16A16,/* fp16 matmul -> per-channel scale chain (ork_npu_chain_mm_perchan_f16) example: chain_mm_perchan_f16_probe */
    ORK_COMPOSITE_SEQ,                        /* heterogeneous op sequence, any precision (ork_submit_seq)           example: test_submit_seq / sdp_chain_probe */
    /* --- mixed-precision / batched composites (added 2026-07-22) --- */
    ORK_COMPOSITE_MATMUL_SILU_W16A16_I16SILU, /* MIXED: fp16 matmul (w16a16) + int16 SiLU (_i16silu) fused in ONE
                                               * PC-chain — the quality-path building block (fp16 on the sensitive
                                               * gate matmul, int16 SiLU). (ork_ssd_probe_mixchain)  example: mixchain_probe */
    ORK_COMPOSITE_MATMUL_BATCHED_FUSED_W16A16,/* fp16 fused batched matmul (attention / SSD A.V)
                                               * (ork_bmm_fp16_fused)               example: test_bmm_fused / ssd_fusedchain_probe */
    ORK_COMPOSITE_GRAPH_REPLAY_F16,           /* fp16 op-graph replay mechanism (ork_npu_replay_full_f16) example: replay_f16_full_test */
    ORK_COMPOSITE_NKIND
} ork_composite;
const char   *ork_composite_name(ork_composite k);              /* enum -> "ffn_swiglu_w8a8"; NULL if out of range */
ork_composite ork_composite_from_name(const char *name);        /* "ffn_swiglu_w8a8" -> enum; ORK_COMPOSITE_NKIND if unknown */

/* regcmd -> op binding: which ork_op (and impl mode) a REGCMD_* byte template implements. Every REGCMD_*
 * base template in the driver MUST have a binding (enforced by check_registry.sh — 0 orphan regcmds), so no
 * regcmd exists that isn't the implementation of an exported op. Returns ORK_OP_NKIND if the name is unbound. */
ork_op ork_regcmd_op(const char *regcmd_name, ork_impl_mode *mode_out);

/* --- Deterministic op->op chaining: a validated LOOKUP, never an algorithm. ------------------------------
 * How two consecutive ops may combine is decided by a table, not by heuristics. Every (from,to) permutation
 * is exhaustively validated on-silicon (tools/mode_probe transition matrix) and its result baked into
 * ork_chain_table[from][to]. The chain assembler asks ork_chain_lookup(from,to) and obeys it — so a
 * transition that wedges the NPU is DISALLOWed structurally, not discovered at runtime (this replaces the
 * heuristic get_node_chain_type/seq_hw_ok path that let an unvalidated transition hang). */
typedef enum {
    ORK_CHAIN_DISALLOW = 0,  /* do NOT chain with the CURRENT driver config; run as independent submits with a
                              * mode reset between (always correct — the safe baseline, and the DEFAULT for any
                              * pair not yet validated). IMPORTANT: a DISALLOW from a mode_probe fix=none wedge
                              * means the transition is unsafe with the transition config we CURRENTLY apply
                              * (ork_npu_enter/XSPEC profile + regcmd) — it is NOT necessarily a hardware limit.
                              * Many DISALLOW pairs are a MISSING/WRONG transition template: e.g. matmul->int8-SDP
                              * wedges as separate submits (no int8-SDP-tuned XSPEC profile), yet the SAME
                              * transition is HW-safe in a PC-chain (FFN gate->silu). Such pairs are candidates
                              * for a transition-template fix — test mode_probe fix=RESET; if safe, add that
                              * reset/profile to ork_npu_enter and UPGRADE the cell to SW. Treat DISALLOW as
                              * "unsupported by our config (yet)", not "the NPU can't". */
    ORK_CHAIN_SW,            /* SW-chain: sequence the two ops without a HW handoff (validated safe to
                              * sequence, but not HW-chainable). Fallback for pairs that can't HW-chain. */
    ORK_CHAIN_HW,            /* HW-chain: the two ops ride ONE PC-chain / doorbell-seq submit (validated
                              * safe AND fast). */
    ORK_CHAIN_NRULE
} ork_chain_rule;
ork_chain_rule ork_chain_lookup(ork_op from, ork_op to);   /* validated transition rule; DISALLOW if unknown */

/* SINGLE SOURCE for op->op chain rules. One X-macro emits BOTH the runtime table (ork_ops.c) AND named enum
 * constants ORK_CR__<from>__<to> — because C cannot index a const array in a constant expression, compile-time
 * checks need the rules as enum constants (which ARE constant expressions). List every VALIDATED pair (HW/SW
 * and explicit DISALLOW); unlisted pairs default to DISALLOW at runtime. Each pair is validated by the
 * exhaustive op->op campaign (tools/mode_probe); the seeds here are the ones already proven in OPS_REGISTRY. */
/* HW pairs = ride one PC-chain (proven by the chain probes). SW pairs = safe to SEQUENCE as separate
 * submits (proven by the mode_probe 7-op transition campaign, 2026-07-21: all 49 ordered pairs among
 * {matmul_f16, matmul_i8, exp_i16, silu_i16, mul_i16, mul_f16, add_f16} ran fix=none with no wedge on a
 * fresh process each, board recovered). NOTE: matmul_i8->mul_i16 is SW (sequenceable) even though its *HW*
 * 2-input-SDP chain HANGS (chain_mm_perchan_probe) — DISALLOW means "can't even sequence", which the
 * campaign found for NONE of these pairs. Pairs among the other (uncampaigned) ops stay unlisted = DISALLOW. */
#define ORK_CHAIN_LIST(X) \
    /* --- HW-chainable (one PC-chain) --- */ \
    X(ORK_OP_MM_I8,    ORK_OP_MM_I8,    ORK_CHAIN_HW)  /* matmul->matmul run (adjacent fusion; chain_gu_silu_probe) */ \
    X(ORK_OP_MM_I8,    ORK_OP_SILU_I8,  ORK_CHAIN_HW)  /* FFN: gate matmul -> SiLU */ \
    X(ORK_OP_SILU_I8,  ORK_OP_MM_I8,    ORK_CHAIN_HW)  /* FFN: SiLU -> up matmul */ \
    X(ORK_OP_MM_I8,    ORK_OP_EWMUL_I8, ORK_CHAIN_HW)  /* FFN: up matmul -> GLU mul */ \
    X(ORK_OP_EWMUL_I8, ORK_OP_MM_I8,    ORK_CHAIN_HW)  /* FFN: GLU mul -> down matmul */ \
    X(ORK_OP_MM_F16,   ORK_OP_MUL_PERCHANNEL_F16, ORK_CHAIN_HW) /* attn A.V -> per-channel scale (mm_perchan_f16_probe) */ \
    /* --- SW-chainable (separate-submit safe; mode_probe campaign 2026-07-21, all fix=none SAFE) --- */ \
    X(ORK_OP_MM_F16,   ORK_OP_MM_F16,   ORK_CHAIN_SW) X(ORK_OP_MM_F16,   ORK_OP_MM_I8,    ORK_CHAIN_SW) \
    X(ORK_OP_MM_F16,   ORK_OP_EXP_I16,  ORK_CHAIN_SW) X(ORK_OP_MM_F16,   ORK_OP_SILU_I16, ORK_CHAIN_SW) \
    X(ORK_OP_MM_F16,   ORK_OP_MUL_I16,  ORK_CHAIN_SW) X(ORK_OP_MM_F16,   ORK_OP_EWMUL_F16,ORK_CHAIN_SW) \
    X(ORK_OP_MM_F16,   ORK_OP_ADD_F16,  ORK_CHAIN_SW) \
    X(ORK_OP_MM_I8,    ORK_OP_MM_F16,   ORK_CHAIN_SW) X(ORK_OP_MM_I8,    ORK_OP_EXP_I16,  ORK_CHAIN_SW) \
    X(ORK_OP_MM_I8,    ORK_OP_SILU_I16, ORK_CHAIN_SW) X(ORK_OP_MM_I8,    ORK_OP_MUL_I16,  ORK_CHAIN_SW) \
    X(ORK_OP_MM_I8,    ORK_OP_EWMUL_F16,ORK_CHAIN_SW) X(ORK_OP_MM_I8,    ORK_OP_ADD_F16,  ORK_CHAIN_SW) \
    X(ORK_OP_EXP_I16,  ORK_OP_MM_F16,   ORK_CHAIN_SW) X(ORK_OP_EXP_I16,  ORK_OP_MM_I8,    ORK_CHAIN_SW) \
    X(ORK_OP_EXP_I16,  ORK_OP_EXP_I16,  ORK_CHAIN_SW) X(ORK_OP_EXP_I16,  ORK_OP_SILU_I16, ORK_CHAIN_SW) \
    X(ORK_OP_EXP_I16,  ORK_OP_MUL_I16,  ORK_CHAIN_SW) X(ORK_OP_EXP_I16,  ORK_OP_EWMUL_F16,ORK_CHAIN_SW) \
    X(ORK_OP_EXP_I16,  ORK_OP_ADD_F16,  ORK_CHAIN_SW) \
    X(ORK_OP_SILU_I16, ORK_OP_MM_F16,   ORK_CHAIN_SW) X(ORK_OP_SILU_I16, ORK_OP_MM_I8,    ORK_CHAIN_SW) \
    X(ORK_OP_SILU_I16, ORK_OP_EXP_I16,  ORK_CHAIN_SW) X(ORK_OP_SILU_I16, ORK_OP_SILU_I16, ORK_CHAIN_SW) \
    X(ORK_OP_SILU_I16, ORK_OP_MUL_I16,  ORK_CHAIN_SW) X(ORK_OP_SILU_I16, ORK_OP_EWMUL_F16,ORK_CHAIN_SW) \
    X(ORK_OP_SILU_I16, ORK_OP_ADD_F16,  ORK_CHAIN_SW) \
    X(ORK_OP_MUL_I16,  ORK_OP_MM_F16,   ORK_CHAIN_SW) X(ORK_OP_MUL_I16,  ORK_OP_MM_I8,    ORK_CHAIN_SW) \
    X(ORK_OP_MUL_I16,  ORK_OP_EXP_I16,  ORK_CHAIN_SW) X(ORK_OP_MUL_I16,  ORK_OP_SILU_I16, ORK_CHAIN_SW) \
    X(ORK_OP_MUL_I16,  ORK_OP_MUL_I16,  ORK_CHAIN_SW) X(ORK_OP_MUL_I16,  ORK_OP_EWMUL_F16,ORK_CHAIN_SW) \
    X(ORK_OP_MUL_I16,  ORK_OP_ADD_F16,  ORK_CHAIN_SW) \
    X(ORK_OP_EWMUL_F16,ORK_OP_MM_F16,   ORK_CHAIN_SW) X(ORK_OP_EWMUL_F16,ORK_OP_MM_I8,    ORK_CHAIN_SW) \
    X(ORK_OP_EWMUL_F16,ORK_OP_EXP_I16,  ORK_CHAIN_SW) X(ORK_OP_EWMUL_F16,ORK_OP_SILU_I16, ORK_CHAIN_SW) \
    X(ORK_OP_EWMUL_F16,ORK_OP_MUL_I16,  ORK_CHAIN_SW) X(ORK_OP_EWMUL_F16,ORK_OP_EWMUL_F16,ORK_CHAIN_SW) \
    X(ORK_OP_EWMUL_F16,ORK_OP_ADD_F16,  ORK_CHAIN_SW) \
    X(ORK_OP_ADD_F16,  ORK_OP_MM_F16,   ORK_CHAIN_SW) X(ORK_OP_ADD_F16,  ORK_OP_MM_I8,    ORK_CHAIN_SW) \
    X(ORK_OP_ADD_F16,  ORK_OP_EXP_I16,  ORK_CHAIN_SW) X(ORK_OP_ADD_F16,  ORK_OP_SILU_I16, ORK_CHAIN_SW) \
    X(ORK_OP_ADD_F16,  ORK_OP_MUL_I16,  ORK_CHAIN_SW) X(ORK_OP_ADD_F16,  ORK_OP_EWMUL_F16,ORK_CHAIN_SW) \
    X(ORK_OP_ADD_F16,  ORK_OP_ADD_F16,  ORK_CHAIN_SW) \
    /* --- campaign 2 (2026-07-21, int8-SDP ops added to mode_probe): fp16-matmul -> the remaining int16 SDP
     *     ops are SW-safe; matmul -> int8-SDP HARD-WEDGES (validated: MM_F16->SILU_I8 hung, MM_F16->GELU_I8
     *     hard-wedged the NPU, MM_I8->SILU_I8 hung). matmul->int8-SDP is a mode-switch hazard for BOTH fp16
     *     and int8 matmul — whereas matmul->int16-SDP is safe. The remaining matmul->int8-SDP pairs are
     *     DISALLOW-by-default (unlisted); the ones below are listed explicitly to document the VALIDATED
     *     wedge (so ORK_ASSERT_CHAIN_STEP reports "DISALLOWed" rather than "unvalidated"). --- */ \
    X(ORK_OP_MM_F16,   ORK_OP_GELU_I16, ORK_CHAIN_SW) X(ORK_OP_MM_F16,   ORK_OP_ADD_I16,  ORK_CHAIN_SW) \
    X(ORK_OP_MM_F16,   ORK_OP_SILU_I8,  ORK_CHAIN_DISALLOW) /* fp16-matmul -> int8-SDP: hung (NO_OUTPUT); no HW chain proven */ \
    X(ORK_OP_MM_F16,   ORK_OP_GELU_I8,  ORK_CHAIN_DISALLOW) /* fp16-matmul -> int8-SDP: HARD-WEDGE (power-cycle) */
    /* NOTE: MM_I8->SILU_I8 (int8-matmul -> int8-SiLU) is listed HW above (the FFN gate->silu PC-chain,
     * chain_gu_silu_probe). mode_probe's SEPARATE-submit MM_I8->SILU_I8 HUNG, but that's the SW path — the op
     * IS HW-chainable in one submit (how the FFN uses it), so HW stands. Separate-submit int8-matmul->int8-SDP
     * is NOT SW-safe (do not downgrade the FFN's HW pair to SW). */

/* Named enum constants per validated pair — usable in _Static_assert (unlike a const-array index). */
enum {
#define X(f, t, r) ORK_CR__##f##__##t = (r),
    ORK_CHAIN_LIST(X)
#undef X
    ORK_CR__SENTINEL = 0
};
/* Compile-time rule for a statically-known transition. Undeclared for an unlisted (never-validated) pair ->
 * referencing it is a compile error, so a static chain over an unvalidated transition won't build. */
#define ORK_CHAIN_RULE(f, t) ORK_CR__##f##__##t
/* Assert a statically-declared chain STEP is chainable (HW or SW). Fails to compile on DISALLOW/unvalidated —
 * this is how the SDK's fixed composites are validated at build time (no runtime failing check needed). */
#define ORK_ASSERT_CHAIN_STEP(f, t) \
    _Static_assert(ORK_CHAIN_RULE(f, t) != ORK_CHAIN_DISALLOW, \
        "chain step " #f " -> " #t " is DISALLOWed or unvalidated")

/* Single generic submit surface (declared after ork_seq_op, below): the SDK addresses ops by enum + mode;
 * the driver resolves the regcmd impl and validates each transition via ork_chain_lookup. See ork_submit /
 * ork_submit_chain after the ork_seq_op definition. */
typedef struct {
    ork_seq_kind kind;
    ork_w      *w;                 /* matmul weight (NULL for weightless SDP ops) */
    int         M, N;              /* M rows; N is taken from w for matmuls, supplied here for weightless SDP ops */
    const void *A;                 /* primary input (int8/fp16 A, or SDP operand) */
    const void *B;                 /* second SDP operand (ewmul/add); NULL otherwise */
    void       *C;                 /* output */
    double      in_scale, out_scale;  /* SDP scales: silu/gelu in/out; add uses in_scale as a_scale + b_scale/out_scale */
    double      b_scale;              /* SDP add: b operand scale (a_scale = in_scale) */
    int         mult, shift;          /* SDP ewmul_i8 requant (out = clamp(A*B*mult>>shift)) */
    int         group;                /* dependency grouping (default 0 = ungrouped, legacy per-op scheduling).
                                       * >0: CONTIGUOUS ops sharing a group id form ONE sequential chain (kept on
                                       * one core, in order); a group-id change starts a new INDEPENDENT chain.
                                       * A contiguous run of group>0 ops rides ork_dyn_begin_seq_i8_mc — SDP ops
                                       * stay in the doorbell chain instead of forcing a blocking SW break. The
                                       * run's terminal op (each group's last) must be a matmul (sentinel). */
} ork_seq_op;
/* Run the n-op sequence in order, HW-batching + SW-breaking as above. 0/ok, -1 a submit failed/wedged,
 * -2 bad args, -3 an op-kind whose dispatch is not yet wired (documented TODO row, e.g. SILU_F16). */
int          ork_submit_seq(ork_npu *ctx, const ork_seq_op *ops, int n);
/* Generic enum-driven submit surface (see the ork_op / ork_impl_mode / ork_chain_lookup design above).
 * ork_submit runs ONE op via its dispatch (mode advisory for a single op — chaining is ork_submit_chain).
 * ork_submit_chain PARTITIONS the sequence at every DISALLOW transition (those ops run as separate submits,
 * never chained) and routes each maximal run through ork_submit_seq — it never FAILS on a transition (a
 * disallowed pair is split, not rejected; correctness is guaranteed by the table + the compile-time composite
 * asserts, since orkd + SDK ship together). Returns: 0 ok; -3 an op has no generic dispatch (use its typed
 * ork_npu_* entry); -2 bad args/op; -1 a submit failed. */
int          ork_submit(ork_npu *ctx, ork_op op, ork_impl_mode mode, const ork_seq_op *args);
int          ork_submit_chain(ork_npu *ctx, const ork_seq_op *ops, int n);
/* Heterogeneous single-core NONBLOCK chain: run ONE group of int8 ops (matmul + int8 SDP, e.g. EWMUL_I8) as one
 * PC-chain; terminal MUST be a matmul (its int32 sentinel gates completion). Returns a handle (drain with
 * ork_dyn_seq_end) or NULL if ineligible (non-int8 / M>64 / non-conforming K / terminal not a matmul / kind not
 * yet supported) — caller then runs the ops via the SW break. The scheduler slices a sequence into groups. */
ork_dyn_chain *ork_dyn_begin_seq_i8(ork_npu *ctx, int n, const ork_seq_op *ops);
/* Multi-core: groups = contiguous op slices [gstart[g],gstart[g+1]); gstart[0]=0, gstart[ngroups]=n. Each group
 * is a dependent chain (terminal op = matmul); INDEPENDENT groups are load-balanced whole onto nc cores (nc<=0
 * = all) and run in parallel. Drain with ork_dyn_seq_end (polls every core's terminal). NULL if ineligible. */
ork_dyn_chain *ork_dyn_begin_seq_i8_mc(ork_npu *ctx, int n, const ork_seq_op *ops, int ngroups, const int *gstart, int nc);
int          ork_dyn_seq_end(ork_dyn_chain *h);   /* poll every core's terminal sentinel + per-op copy-back; 0/ok,-1 timeout,-2 bad */

/* Like ork_mm_run_chain_i8 but task[gate_task] gets a FUSED int8 SiLU output stage (set_i8_silu): its C
 * receives int8 silu(gate) (M*N bytes) instead of int32; the silu LUT is streamed to SDP SRAM once before
 * the chain. Chains [gate*silu -> up -> ...] in ONE submit. lut/params as ork_mm_run_i8_silu (build with
 * ork_mm_silu_build_lut). Single M-tile per task for now (M <= chain mcap). 0/ok,-1 wedge,-2 dims. */
int          ork_mm_run_chain_i8_gsilu(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks, int gate_task,
                                       int r_mult, int r_shift, unsigned out_bias, unsigned idx_off,
                                       unsigned cfg4068, const short *lut, int nlut);
/* OPTION B: task[sdp_task] is a STANDALONE int8 silu-SDP op reading task[sdp_task-1]'s (gate) output via
 * aliased buffers (the vendor matmul->SDP pattern); the gate task emits int8 (set_i8_out8, requant
 * gate_mult/gate_shift). The silu LUT for (in_scale,out_scale) is built internally (same as ork_npu_silu_i8).
 * tasks[sdp_task].C gets int8 silu (M*N bytes). Single M-tile per task. 0/ok,-1 wedge,-2 dims,-3 SoC. */
int          ork_mm_run_chain_i8_sdpsilu(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks, int sdp_task,
                                         int gate_mult, int gate_shift, double in_scale, double out_scale);

/* GENERAL heterogeneous FFN chain. Per-task op: kind 0=matmul(int32 out) 1=matmul(int8 out, requant
 * mult/shift) 2=silu-SDP 3=ewmul-SDP; SDP tasks read prior tasks' outputs by index in0/in1 (aliased). Chains
 * e.g. [gate(1) -> silu(2,in0=gate) -> up(1) -> glu(3,in0=silu,in1=up) -> down(0)] in ONE submit. Silu LUT for
 * (in_scale,out_scale) built internally. tasks[i].C gets that op's output. Single M-tile/task. 0/ok,-1,-2,-3. */
typedef struct { int kind; int in0, in1; int mult, shift; } ork_chain_op;
int          ork_mm_run_chain_i8_ffn(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks,
                                     const ork_chain_op *ops, double in_scale, double out_scale);
/* Same chain, but the kind-2 SDP task applies EXP (softmax numerator) — HW-chains [QK^T -> exp -> reduce] in
 * ONE submit, intermediates on-chip. Scores must be <=0 (post-max domain). 0/ok,-1 wedge,-2 dims,-3 SoC. */
int          ork_mm_run_chain_i8_ffn_exp(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks,
                                         const ork_chain_op *ops, double in_scale, double out_scale);
/* CONCURRENT round-robin: dispatch `nchains` homogeneous fused-exp chains across all NPU cores at once (one per
 * core, atomic work-stealing), each on its own per-core scratch. chains[i]=that chain's S[i]-task array; ops the
 * shared op graph; scales the shared exp requant. Prefill throughput (~3x). Cores must be WARM first. 0/ok,<0 err. */
int          ork_mm_run_chains_rr(ork_npu *ctx, int nchains, const ork_mm_task_i8 *const *chains, const int *S,
                                  const ork_chain_op *ops, double in_scale, double out_scale);
/* ORKD coalesced SwiGLU FFN: run the whole [gate->silu->up->glu->down] inner as ONE daemon-side HW-chained
 * submit (ORKD_FFN / orkd_ffn_i8) against the 3 ALREADY-RESIDENT weights (wg/wu/wd must be daemon-imported —
 * is_orkd — e.g. from the orkpack; no re-import). A = int8 activation [M,K]; out = int32 down output [M,Kd].
 * Requant is identity (0x4000/14) + in_scale/out_scale, matching ork_mm_run_chain_i8_ffn. This is the
 * per-tensor int8 coalesced path (fast; one submit instead of per-op). Returns -3 if no daemon or any weight
 * is not resident (caller should fall back to the fd-local ork_mm_run_chain_i8_ffn). */
int          ork_mm_ffn_orkd(ork_npu *ctx, ork_w *wg, ork_w *wu, ork_w *wd,
                             int M, int K, int Nff, int Kd, double in_scale, double out_scale,
                             const int8_t *A, int32_t *out);
int          ork_mm_run_chain_i4(ork_npu *ctx, int S, const ork_mm_task_i4 *tasks);
/* EXPERIMENTAL: int4 incremental-task batch (vendor task_number=N pattern) — one resident int4 weight,
 * M rows, task[0]=full + task[1..]=12-config incremental (advance only A/C; weight loaded once), ONE
 * submit. Not Hcap-capped (unlike ork_i4_batch stride-2). int8 A, int32 C, N<=64 (single N-block). */
int          ork_mm_run_i4_incr(ork_npu *ctx, ork_w *w, int M, const int8_t *A, int32_t *C);
/* Async round-robin stream: S independent int8 matmuls dispatched dynamically across NPU cores (pull
 * model, no barrier). For batches of independent matmuls (e.g. EAGLE-3 verification). 0/ok, -1/-2 err. */
int          ork_mm_run_stream_i8(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks);
/* SMALL-K int8 round-robin stream (int8 twin of run_stream_f16): S single-slice int8 matmuls (K%32,N%16,
 * NOT the K%512 full-K path) across NPU cores; A int8 [M,K] per task, C int32 [M,N]. For the on-NPU SSM
 * scan's per-head gram/output stages (tiny K) with 3-core batching. Caller quantizes A + dequants int32. */
int          ork_mm_run_stream_i8_sk(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks);
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

/* ---- BATCHED DYNAMIC GEMM (the "attention" primitive) --------------------------------------------
 * For each b in [0,nbatch): C[b] = A[b] * B[b], with A[b] = [M,K], B[b] = [K,N], C[b] = [M,N],
 * each batch dense/contiguous row-major. UNLIKE ork_mm_run_i8/i4/run (which take a RESIDENT static
 * 2-D weight packed once), BOTH operands here are DYNAMIC activations — B[b] is packed ephemerally
 * each call. This is the primitive attention (scores = Q·Kᵀ, out = scores·V, looped per head) and the
 * Gated-Delta-Net chunked matmuls (delta-net-base.cpp) need — ggml's batched MUL_MAT (ne[2]/ne[3]>1
 * with a computed src0) maps directly onto it. The caller arranges any transpose so B is [K,N].
 * i8:  A,B int8   → C int32 (exact).           K%32==0, N%32==0.
 * i4:  A,B int4-in-int8 [-8,7] → C int32.       K%32==0, N%64==0.
 * f16: A,B fp16   → C fp32.                     K%32==0, N%16==0.
 * nbatch>0, M>0. Returns 0 on success, <0 on error. Correctness-first (per-batch submit); submit-floor
 * amortization via chaining is a follow-up. */
int          ork_bmm_i8  (ork_npu *ctx, int nbatch, int M, int K, int N, const int8_t  *A, const int8_t  *B, int32_t *C);
int          ork_bmm_i4  (ork_npu *ctx, int nbatch, int M, int K, int N, const int8_t  *A, const int8_t  *B, int32_t *C);
int          ork_bmm_fp16(ork_npu *ctx, int nbatch, int M, int K, int N, const ork_f16 *A, const ork_f16 *B, float   *C);

/* Strided batched GEMM — the SAME batched primitive, but each operand is addressed by explicit
 * ELEMENT strides instead of assuming dense row-major, so a permuted / view operand offloads WITHOUT
 * the caller first materializing a contiguous copy (ggml_cont). This is what real attention needs:
 * QKᵀ/AV read a permuted-Q ([d,H,T]→[d,T,H]) and a KV-cache view (rows strided by the cache's padded
 * head layout), which ggml_is_contiguous rejects — so those ops were declined and stayed on CPU.
 *   A[m,k] = A + b*abs + m*as_m + k*as_k
 *   B[k,n] = B + b*bbs + k*bs_k + n*bs_n
 *   C[m,n] = C + b*cbs + m*cs_m + n*cs_n
 * All strides are in ELEMENTS (not bytes). Set the natural values (as_m=K,as_k=1,bs_k=N,bs_n=1,
 * cs_m=N,cs_n=1, abs=M*K,bbs=K*N,cbs=M*N) to reproduce the dense ork_bmm_* above. The contraction dim
 * (k) is typically stride-1 in attention; strided-k is supported (gathered) but slower. Each batch's
 * strided operands are gathered into contiguous scratch, then the dense pack/run path runs unchanged —
 * so numerics are identical to ork_bmm_*. Same dtype/shape rules. Returns 0/ok, <0 on error. */
typedef struct { long as_m, as_k, bs_k, bs_n, cs_m, cs_n, abs, bbs, cbs; } ork_bmm_strides;
int          ork_bmm_i8_strided  (ork_npu *ctx, int nbatch, int M, int K, int N, const int8_t  *A, const int8_t  *B, int32_t *C, const ork_bmm_strides *s);
int          ork_bmm_i4_strided  (ork_npu *ctx, int nbatch, int M, int K, int N, const int8_t  *A, const int8_t  *B, int32_t *C, const ork_bmm_strides *s);
int          ork_bmm_fp16_strided(ork_npu *ctx, int nbatch, int M, int K, int N, const ork_f16 *A, const ork_f16 *B, float   *C, const ork_bmm_strides *s);

/* Math utilities for caller-driven quantization/transformations */
void         ork_fwht_norm(float *v, int n);

/* ---- FLOOR-DECOMP diagnostics (submit-floor RE) --------------------------------------------------
 * Decompose the per-submit floor. ork_npu_floor_timing returns, accumulated since the last reset:
 *   ioctl_us    = total wall-clock time spent INSIDE the blocking SUBMIT ioctl (kernel job setup +
 *                 register programming + NPU execution + completion-wait).
 *   hw_us       = SUM of the kernel-reported sub.hw_elapse_time across those ioctls (the hardware's own
 *                 NPU-busy view). Comparing hw_us to ioctl_us splits real-NPU-compute from driver
 *                 dispatch/wait overhead — the core of the poll-granularity hypothesis.
 *   hw_raw_last = the last raw sub.hw_elapse_time value (to infer whether the kernel reports ns or us).
 *   n           = number of SUBMIT ioctls counted. */
void         ork_npu_floor_timing(double*ioctl_us,double*hw_us,long long*hw_raw_last,long*n);
void         ork_npu_floor_reset(void);

/* ---- MODE-TRANSITION RE hooks (tools/mode_probe.c) -----------------------------------------------
 * Standalone SDP ops (ork_npu_ewmul_f16/_i16, ork_npu_exp_i16/silu_i16/…) reprogram the NPU pipeline but
 * do NOT update the driver's cached matmul mode state (last_dt/warmed), so a following same-dtype matmul
 * skips its reset/re-warm and can wedge (errno=110). These expose the two candidate mitigations:
 *   ork_npu_mode_invalidate — clear the cached mode state only (next matmul re-warms itself; no HW reset).
 *   ork_npu_mode_reset      — explicit HW ACT_RESET + invalidate (heavyweight, always safe). */
void         ork_npu_mode_invalidate(ork_npu *ctx);
void         ork_npu_mode_reset(ork_npu *ctx);

/* FUSED SSD-SCAN MATMUL BENCH (SSM-on-NPU RE): chain one Mamba-2/SSD layer's group-batched scan matmuls
 * into ONE PC-chained submit with resident all-ones operands, vs the same matmuls as N separate submits.
 * Measures the per-submit-floor amortization of the fused on-NPU scan. fused_us/persub_us = per-iter wall;
 * ok_out=1 if the fused chain is bit-correct (every output==K). Returns 0/ok, <0 error. Board only. */
/* (b) layout probe: one fp16 matmul via the raw-synth fused-chain mechanism with ROW-MAJOR operands
 * A[M,K],B[K,N]->C[M,N] fp32 — decides if a real-operand fused SSD scan can stage row-major directly.
 * K%32,N%16. 0/ok,<0. rk3588 diagnostic. */
int          ork_ssd_probe_rawmm_f16(ork_npu *c,int M,int K,int N,const ork_f16 *A,const ork_f16 *B,float *C);
/* (b) fused-mm probe: one fp16 matmul via the fused-chain mechanism with B PACKED (ork_mm_pack tiling) +
 * A row-major + C dense — tests whether the real-operand fused SSD chain can reuse ork_mm_pack for B. 0/ok,<0. */
int          ork_ssd_probe_fusedmm_f16(ork_npu *c,int M,int K,int N,const ork_f16 *A,const ork_f16 *B,float *C);
/* FUSED batched fp16 GEMM: drop-in for ork_bmm_fp16 (nbatch matmuls, both operands dynamic) but chains all
 * nbatch matmuls into ONE PC-chained submit — amortizes the ~48us/submit floor across the batch (the SSD
 * scan per-stage H-batch). Packed-B (ork_mm_pack) + row-major-A + dense-C; numerically identical to
 * ork_bmm_fp16. SINGLE-CORE (a PC chain runs on one core). Single-slice (K<=ks, N<=nmax), nb<=64. 0/ok,<0. */
int          ork_bmm_fp16_fused(ork_npu *c,int nb,int M,int K,int N,const ork_f16 *A,const ork_f16 *B,float *C);
/* STREAMED batched fp16 GEMM: drop-in for ork_bmm_fp16 but dispatches the nbatch INDEPENDENT matmuls
 * round-robin across ALL NPU cores (fp16 twin of ork_mm_run_stream_i8) — each core pulls the next matmul
 * and runs a single-core submit on itself (no barrier; CPU-prep of op N+1 overlaps NPU of op N). For the
 * SSD scan's per-stage H independent matmuls: ~3-5x the single-core chain. Packed-B + row-major-A + dense-C,
 * numerically identical to ork_bmm_fp16. Single-slice, nb>=1. 0/ok,<0. */
typedef struct { ork_w *w; int M; const ork_f16 *A; float *C; } ork_mm_task_f16;
int          ork_mm_run_stream_f16(ork_npu *c, int S, const ork_mm_task_f16 *tasks);
/* CHAINED-MULTICORE fp16 stream: PC-chains each core's round-robin-assigned matmuls into ONE task_number>1
 * submit (amortizes the ~48us submit floor over many programs) while keeping 3-core parallelism. Same
 * task/operand semantics as run_stream_f16; single-slice fp16 (K%32,N%16). For the on-NPU SSM scan stages. */
int          ork_mm_run_stream_f16_chain(ork_npu *c, int S, const ork_mm_task_f16 *tasks);
int          ork_bmm_fp16_stream(ork_npu *c,int nb,int M,int K,int N,const ork_f16 *A,const ork_f16 *B,float *C);
/* (b) mixed-precision chain probe: FP16 matmul task + INT16 silu-SDP task in ONE submit, both validated —
 * confirms fp16 & int16 tasks coexist in a PC-chain (the fused SSD scan's matmul+elementwise mix). 0/ok,<0. */
int          ork_ssd_probe_mixchain(ork_npu *c,int *mm_ok,int *silu_ok,double *us);
/* SSM_SCAN (Mamba-2) on the NPU — the ggml-ork GGML_OP_SSM_SCAN kernel. Chunked mode-5 scan (matmul spine
 * on the NPU pooled 3-core stream, elementwise on CPU). Matches ggml_compute_forward_ssm_scan_f32:
 * softplus(dt), scalar decay A{nh}, NO D skip, output y (x-shaped) + s_new. Contiguous ggml layout:
 * s/s_new{nc,nr,nh,ns} x/y{nr,nh,nt,ns} dt{nh,nt,ns} A{nh} B/C{nc,ng,nt,ns}. nc%32,nr%16,nh%ng==0. 0/ok,<0. */
int          ork_ssm_scan_f32(ork_npu *c,int nc,int nr,int nh,int ng,int nt,int ns,
                              const float *s0,const float *x,const float *dt,const float *A,
                              const float *B,const float *C,float *y,float *s_new);
int          ork_ssd_fused_scan_bench(ork_npu *c,int H,int P,int Nst,int G,int CS,int NC,int iters,int dtype,int perhead,
                                      double *fused_us,double *persub_us,int *ok_out);  /* dtype:1=int8,0=fp16; perhead:1=fp16-stable per-head Y_diag */

#endif
