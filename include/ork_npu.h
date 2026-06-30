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
#include <stdint.h>

typedef _Float16 ork_f16;
typedef struct ork_npu ork_npu;     /* device context (one per process) */
typedef struct ork_w   ork_w;       /* resident packed weights for one B[K,N] */

/* Library version (semver). The build may also inject a short git hash via -DORK_GIT_HASH (the
 * Makefile does this when built where git is available); ork_npu_version() then returns
 * "MAJOR.MINOR.PATCH+g<hash>", else just the semver. Bump MINOR on backward-compatible API adds. */
#define ORK_NPU_VERSION "0.6.15"
const char  *ork_npu_version(void);  /* e.g. "0.3.0" or "0.3.0+g1a2b3c4" */

/* Open the NPU, detect the SoC, power on. Returns NULL on failure (no NPU / no perms). */
ork_npu     *ork_npu_init(void);
void         ork_npu_free(ork_npu *ctx);

/* SoC introspection */
const char  *ork_npu_soc(const ork_npu *ctx);    /* "rk3588", "rk3576", ... */
int          ork_npu_cores(const ork_npu *ctx);  /* NPU core count */
int          ork_npu_validated(const ork_npu *ctx); /* 1 if this SoC's params are HW-validated */

/* Policy: cap how many NPU cores the auto-tuner may use per matmul (n<=0 → all SoC cores, the
 * default). Multi-core + the full-K int8 decode layout are chosen automatically per matmul; this
 * just bounds them (e.g. reserve cores for another workload). */
void         ork_npu_set_core_budget(ork_npu *ctx, int n);

/* Per-weight NPU IOMMU domain placement. The rk_iommu 32-bit IOVA window (~4 GiB) is per
 * iommu_domain_id, so a model larger than 4 GiB can stay FULLY resident (no streaming, no per-token
 * map/unmap) by placing its weights across several domains. Call this before ork_mm_pack_i8 /
 * ork_mm_load_i8 (and the fp16/int4 variants): each weight packed/loaded afterward lands its resident
 * tiles in `domain` and records it; ork_mm_run* then submits that weight's matmuls against the same
 * domain automatically. domain<0 reverts to the process default (env ORK_IOMMU_DOMAIN, else 0). */
void         ork_npu_set_pack_domain(ork_npu *ctx, int domain);
int          ork_w_domain(const ork_w *w);   /* the IOMMU domain a packed weight resides in */

/* Zero-copy DMA buffers (NPU-coherent, CPU-mapped). Allocate the activation A and/or output C here
 * and the matmul reads/writes them in place — no host gather/writeout memcpy (the ~33% prefill
 * residual vs the closed runtime). ork_mm_run detects residency automatically; pass the returned
 * pointer as A/C exactly as a malloc'd one. NULL on failure or table-full (fall back to malloc). */
void        *ork_dma_alloc(ork_npu *ctx, size_t size);
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

/* Pack + upload B[K,N] (row-major) into NPU-resident tile layout; reuse across runs.
 * fp16: K%32==0, N%16==0.  int8: K%32==0, N%32==0.  Returns NULL on bad dims. */
ork_w       *ork_mm_pack   (ork_npu *ctx, int K, int N, const ork_f16  *B);  /* fp16 weights */
ork_w       *ork_mm_pack_i8(ork_npu *ctx, int K, int N, const int8_t   *B);  /* int8/w8a8 weights */
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

void         ork_w_free(ork_w *w);
/* Free a packed weight AND reclaim its NPU DMA/IOVA (for layer-streaming eviction; needs the ctx for
 * the device fd). Reclaims only per-tile-owned weights (pack/pack_i4/pack_i8); arena-view weights are
 * left to teardown. Use this instead of ork_w_free when you need the 4 GiB IOVA window back. */
void         ork_mm_free(ork_npu *ctx, ork_w *w);
size_t       ork_w_bytes(const ork_w *w);   /* resident NPU bytes (Bb+Bf) — for a streaming cache's IOVA budget */
int          ork_w_quant_kind(const ork_w *w);   /* ORK_QK_* of the int4 weight store (UNIFORM / CODEBOOK_NF4) */
/* Per-output-channel dequant scale (length N) retained on an int4-packed weight (ork_mm_pack_i4a8 /
 * ork_mm_load_i4a8); C_real[m][n] = aScale[m]*bscale[n]*Ci[m][n]. NULL for non-int4 weights. */
const float *ork_w_bscale(const ork_w *w);
/* PERSIST: dump a packed weight's tile bytes (out=NULL → size), and reload pre-tiled int8 bytes straight
 * into DMA (no dequant/quant/tile) — the .orkpack fast path that makes streaming re-packs a plain copy. */
size_t       ork_w_dump(const ork_w *w, void *out, size_t cap);
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
int               ork_stream_pool_map  (ork_stream_pool *p, ork_stream_entry *e);  /* CHEAP: MEM_CREATE import -> IOVA (per submit) */
int               ork_stream_pool_run  (ork_stream_pool *p, ork_stream_entry *e, int M, const int8_t *A, int32_t *C);
void              ork_stream_pool_unmap(ork_stream_pool *p, ork_stream_entry *e);  /* MEM_DESTROY; entry STAYS in RAM */
void              ork_stream_pool_remove(ork_stream_pool *p, ork_stream_entry *e); /* free the RAM buffer (caller's evict) */
void              ork_stream_pool_free (ork_stream_pool *p);
size_t            ork_stream_entry_bytes (const ork_stream_entry *e);   /* RAM bytes held (for the caller's budget) */
int               ork_stream_entry_mapped(const ork_stream_entry *e);   /* 1 if currently IOVA-mapped */

/* C[M,N] = A[M,K] x packed weights. Run dtype must match the pack dtype. Returns 0 on ok.
 *   fp16: A fp16 (row-major), C fp32.   int8: A int8 (row-major), C int32.
 *   int4 (W4A4): A int4 ([-8,7] in int8, row-major), C int32 (raw sum; apply scales:
 *                C_real[m][n] = aScale[m]*bScale[n]*C[m][n]). */
int          ork_mm_run   (ork_npu *ctx, ork_w *w, int M, const ork_f16 *A, float   *C);
int          ork_mm_run_i8(ork_npu *ctx, ork_w *w, int M, const int8_t  *A, int32_t *C);
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
