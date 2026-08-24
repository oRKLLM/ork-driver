/* ork/weights.h — Resident weight objects: pack, load, dump, free, and the streaming pool
 *
 * Everything that turns host bytes into an NPU-resident ork_w and back: the per-precision
 * packers, the .orkpack persist/load path, resident KV, and the RAM-backed streaming pool.
 *
 * Part of the public API. Do NOT include this directly — include <ork_npu.h>, which is the
 * umbrella and the only supported entry point; these parts are a readability split of it
 * (ork_npu.h was 1519 lines) and their boundaries may move. Types live in ork_npu.h above
 * the includes, so this header is not self-contained by design. */
#ifndef ORK_WEIGHTS_H
#define ORK_WEIGHTS_H
/* Load pre-tiled int8 weight bytes (ork_w_dump / .orkpack) into NPU-resident storage WITHOUT the
 * alloc+memcpy of ork_i8_mm_load: each tile is imported zero-copy (dma-buf the NPU reads in place).
 * Same blob format and round-trip as ork_i8_mm_load; returns NULL on shape/size mismatch or if import
 * is unavailable (caller falls back to ork_i8_mm_load). Weights are write-once: filled+synced here,
 * read-only across every submit. ork_mm_free / ork_w_free release the imports (MEM_DESTROY + close fd). */
ork_w       *ork_i8_mm_load_import(ork_npu *ctx, int K, int N, const void *blob, size_t n);

/* ---- ORKD_IMPORT: client-owned resident weight (client allocs the dma-buf, daemon only maps it) ----
 * CLIENT (no NPU fd): ork_dmabuf_alloc reserves a dma-heap buffer + mmaps it R/W (returns the fd, sets *ptr);
 * fill *ptr with the PRE-TILED .orkpack bytes, then ork_dmabuf_seal(fd) flushes it. Pass the fd to the daemon
 * (SCM_RIGHTS). DAEMON: ork_i8_mm_adopt_imported PRIME-imports the fd into the current pack_domain and lays
 * Bb (+ optional Bf at bf_off) as base+offset VIEWS — no tiling, no weight alloc, bytes stay in the client's
 * buffer. total = buffer bytes; bf_off = byte offset of the full-K Bf region (0 = none). NULL on mismatch. */
int          ork_dmabuf_alloc(size_t size, void **ptr);   /* -> dma-buf fd (>=0), *ptr = mapping; -1 on failure */
void         ork_dmabuf_seal(int dmabuf_fd);              /* flush CPU writes so the imported NPU view sees them */
ork_w       *ork_i8_mm_adopt_imported(ork_npu *ctx, int K, int N, int bb_fd, int bf_fd, size_t bb_bytes, size_t bf_bytes);
/* CLIENT orchestrator (orkd): alloc dma-buf, copy the pre-tiled blob (Bb [+ Bf at bf_off]), seal, and send the
 * fd to the daemon (ORKD_IMPORT). Returns an is_orkd weight handle. n = blob bytes; bf_off = Bf offset (0=none). */
ork_w       *ork_i8_mm_import(ork_npu *ctx, int K, int N, const void *blob, size_t n, size_t bf_off);

/**
 * @brief Pack + upload B[K,N] (row-major) into the NPU-resident tile layout; reuse across runs (fp16).
 * @param K Inner/contraction dim. Must be K%32==0.
 * @param N Output columns. Must be N%16==0.
 * @param B Row-major fp16 weights, K*N elements.
 * @return Resident weight handle for ork_f16_mm_run(), or NULL on bad dims (K%32!=0 or N%16!=0).
 */
ork_w       *ork_f16_mm_pack   (ork_npu *ctx, int K, int N, const ork_f16  *B);
/**
 * @brief Pack + upload B[K,N] (row-major) into the NPU-resident tile layout; reuse across runs (int8/w8a8).
 * @param K Inner/contraction dim. Must be K%32==0.
 * @param N Output columns. Must be N%32==0.
 * @param B Row-major int8 weights, K*N elements.
 * @return Resident weight handle for ork_i8_mm_run(), or NULL on bad dims (K%32!=0 or N%32!=0).
 */
ork_w       *ork_i8_mm_pack(ork_npu *ctx, int K, int N, const int8_t   *B);
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
int          ork_i8_mm_repack(ork_npu *ctx, ork_w *w, int K, int N, const int8_t *B);
/* int8 JIT-inflate to fp16 (emulated W8A16 for IOVA headroom). A gmax-selected "fp16" layer wants
 * UNQUANTIZED fp16 activations (no act-quant error) but not fp16 WEIGHTS; residing fp16 weights doubles
 * IOVA. Keep the weight host-side as compact int8 + per-channel bscale, and inflate it into ONE REUSED
 * fp16 scratch per matmul: resident IOVA = a single scratch (reused across layers), so the fp16-path
 * layer count is decoupled from the 4GiB IOVA cap and gmax becomes a pure coherence<->speed dial. The
 * fp16 MAC then runs int8-precision weights against fp16 activations (RK3588 has no native W8A16). */
/* Allocate a REUSABLE fp16 scratch weight (fp16 tile layout, sized K,N, no data). Run via ork_f16_mm_run /
 * ork_f16_mm_run_silu after filling; reclaim with ork_mm_free. K%32, N%16. NULL on bad dims / alloc. */
ork_w       *ork_f16_mm_scratch(ork_npu *ctx, int K, int N);
/* Fill an fp16 scratch (same K,N) with wf16[k,n]=(f16)((float)i8[k*N+n]*bscale[n]). i8 row-major [K,N];
 * bscale per-output-channel [N] (NULL => scale 1). In place, no alloc. Tiled bytes are bit-identical to
 * ork_f16_mm_pack of the row-major dequantized weight. 0/ok, <0 on bad args. */
int          ork_i8_mm_inflate_to_f16(ork_npu *ctx, ork_w *w, const int8_t *i8, const float *bscale, int K, int N);
/* Re-tile fp16 B[K,N] into an EXISTING fp16 weight (ork_f16_mm_scratch/ork_f16_mm_pack, same K,N) — no
 * bcreate/free. fp16 twin of ork_i8_mm_repack: refresh a persistent weight POOL's data per use (kills
 * per-matmul IOMMU alloc/free churn in a dynamic-operand loop like the SSD scan). 0/ok,<0. */
int          ork_f16_mm_repack(ork_npu *ctx, ork_w *w, int K, int N, const ork_f16 *B);
/* NEON-fused pack/repack DIRECTLY from f32[N][K] (n-major): per-channel symmetric int8 quant + tile in
 * one cache-friendly pass (no transpose scratch). Writes per-channel bscale[N]. For dequantizing weight
 * sources (e.g. Q4_K MoE experts via to_float) without the slow strided f32->int8 transpose. */
ork_w       *ork_i8_mm_pack_f32(ork_npu *ctx, int K, int N, const float *f32, float *bscale_out);
int          ork_i8_mm_repack_f32(ork_npu *ctx, ork_w *w, int K, int N, const float *f32, float *bscale_out);
/* Fused dequant->int8 pack/repack: ork-driver calls `dequant(dctx, n, dst, K)` once per output channel
 * to materialize that channel's K f32 weights into a small REUSED scratch (cache-resident — avoids the
 * full f32[N][K] buffer and its DRAM round-trip), then NEON quant+tiles it. For packing a compressed
 * weight source (e.g. Q4_K MoE experts via ggml to_float) without the cache-thrashing full-f32 pass.
 * Same int8 result as pack_i8_f32 fed the equivalent f32. Writes per-channel bscale[N]. */
typedef void (*ork_dequant_row_fn)(void *dctx, int n, float *dst, int K);
ork_w       *ork_i8_mm_pack_dequant  (ork_npu *ctx, int K, int N, ork_dequant_row_fn dequant, void *dctx, float *bscale_out);
int          ork_i8_mm_repack_dequant(ork_npu *ctx, ork_w *w, int K, int N, ork_dequant_row_fn dequant, void *dctx, float *bscale_out);
/* "Effective w4a8": int4-PRECISION weights, int8 compute, int4 STORAGE. RK3588's NPU MACs are int8-only
 * (no native w4a8), so this quantizes f32[N][K] (n-major, as ggml's to_float produces) to int4 per output
 * channel (symmetric, scale = max|w_n|/7, range [-7,7]), keeps the compact nibble-packed form on the ork_w
 * (K*N/2 bytes — the memory win), NEON-expands int4->int8 [-7,7] in software, and tiles that into the int8
 * resident layout. Runs unchanged via ork_i8_mm_run (returns a DT_I8 ork_w). Writes per-channel bscale[N]
 * (C_real[m][n] = aScale[m]*bscale[n]*Ci[m][n]). Round-to-nearest by default; set env ORK_SR for
 * stochastic rounding (debiases quantization — dot-product error grows ~sqrt(K) not O(K)). K%32, N%32.
 * Set env ORK_NF4 to use a fixed NF4 codebook (non-uniform levels, scale=max|w_n|/127, sets
 * quant_kind=ORK_QK_CODEBOOK_NF4) instead of the uniform int4 grid — better for Gaussian-ish weights. */
/* Int4 weight-store codebook kind (ork_w.quant_kind, set by pack_i4a8 / the int4 .orkpack form). 0 = UNIFORM:
 * the 4-bit value is a uniform int4 grid level; int4->int8 inflation is a sign-extend. 1 = CODEBOOK_NF4: the
 * 4-bit value indexes a 16-entry per-tensor LUT of non-uniform levels (NF4-style) — inflate via a NEON table
 * lookup (vqtbl); better accuracy for Gaussian-ish weights. Set via env ORK_NF4 on ork_i4a8_mm_pack. */
enum { ORK_QK_UNIFORM = 0, ORK_QK_CODEBOOK_NF4 = 1 };
ork_w       *ork_i4a8_mm_pack(ork_npu *ctx, int K, int N, const float *f32, float *bscale_out);
/* As ork_i4a8_mm_pack, but with optional importance-matrix (imatrix) weighted per-channel scale
 * selection. imatrix = optional per-INPUT-channel importance, length K (NULL = uniform / current
 * absmax behavior, byte-for-byte identical to ork_i4a8_mm_pack). When non-NULL, each output channel's
 * quant scale is chosen by searching a small clip-ratio grid r*absmax to minimize the importance-
 * weighted reconstruction error Sum_k imatrix[k]*(w[n][k] - dequant)^2 — clipping trades outlier error
 * for bulk resolution; imatrix decides which input columns' error matters. Applies to both the uniform
 * and NF4 (ORK_NF4) paths. O(grid*K) per channel (pack is one-time). K%32, N%32. */
ork_w       *ork_i4a8_mm_pack_im(ork_npu *ctx, int K, int N, const float *f32, const float *imatrix, float *bscale_out);
ork_w       *ork_i4_mm_pack(ork_npu *ctx, int K, int N, const int8_t   *B);  /* int4 weights, [-8,7] in int8; K%32, N%64 */
/* int4 weights with per-group scales: K split into groups of G (G%32, K%G, G<=10752). Pair with
 * ork_i4_mm_run_grouped, which dequantizes per group into fp32. */
ork_w       *ork_i4_mm_pack_grouped(ork_npu *ctx, int K, int N, const int8_t *B, int G);
/* Tag a LOADED weight as per-group. The .orkpack entry does not carry the group size — a consumer
 * recovers it from the scale COUNT (per-channel stores N, per-group stores (K/G)*N) — but
 * ork_i4_mm_run_grouped needs it on the weight. Returns 0 on success, -1 if w is not int4. */
int          ork_w_set_group(ork_w *w, int G);

/* int4-Stored / int8-Computed Fallback (Tier 4 Memory Optimization): takes unpacked int4 weights 
 * ([-8,7] in int8 containers) and packs them into an int8 resident weight buffer. This runs on 
 * the highly optimized int8 physical hardware path, yielding maximum silicon speed.
 * Returns an int8 dtype ork_w (run with ork_i8_mm_run). */
ork_w       *ork_i4_mm_pack_to_i8(ork_npu *ctx, int K, int N, const int8_t *B);

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
/* Per-output-channel dequant scale (length N) retained on an int4-packed weight (ork_i4a8_mm_pack /
 * ork_i4a8_mm_load); C_real[m][n] = aScale[m]*bscale[n]*Ci[m][n]. NULL for non-int4 weights. */
const float *ork_w_bscale(const ork_w *w);
/* PERSIST: dump a packed weight's tile bytes (out=NULL → size), and reload pre-tiled int8 bytes straight
 * into DMA (no dequant/quant/tile) — the .orkpack fast path that makes streaming re-packs a plain copy. */
size_t       ork_w_dump(const ork_w *w, void *out, size_t cap);
/* CPU-ONLY int8 dump: tile B[K,N] (int8, row-major) straight into `out` (plain DRAM) in the exact
 * .orkpack byte layout of ork_i8_mm_pack()+ork_w_dump(), WITHOUT allocating an NPU/IOVA buffer or any
 * DMA — the NPU is needed only at load time. Building a .orkpack is a pure-CPU, all-cores job; this
 * avoids the serial bcreate/bsync round-trip. out=NULL → return the byte size. K%32, N%32. */
size_t       ork_i8_w_dump_cpu(ork_npu *ctx, int K, int N, const int8_t *B, void *out, size_t cap);
/* CPU Bf (full-K re-tiled) blob for the .orkpack — byte-identical to the load-time Bf rebuild. 0 unless the
 * Bf run envelope (K%512==0 && K<=4096). Pass out=NULL to size. Lets a .orkpack carry Bf (no runtime rebuild). */
size_t       ork_i8_w_dump_bf_cpu(ork_npu *ctx, int K, int N, const int8_t *B, void *out, size_t cap);
/* CPU-ONLY native-W4A4 (DT_I4) dump: same bytes as ork_i4_mm_pack()+ork_w_dump(), tiled straight into caller
 * DRAM with no NPU/IOVA buffer and no DMA — the int4 twin of ork_i8_w_dump_cpu, and the piece that lets a
 * native-W4A4 .orkpack be BUILT off-board (the NPU is then needed only at load time). Reuses the NPU pack's
 * own tiler so the layouts cannot drift; byte-identity is asserted by test_i4_dump_cpu. out=NULL -> size.
 * K%32, N%64. */
size_t       ork_i4_w_dump_cpu(ork_npu *ctx, int K, int N, const int8_t *B, void *out, size_t cap);
ork_w       *ork_i8_mm_load(ork_npu *ctx, int K, int N, const void *blob, size_t n);
/* COMPACT int4 PERSIST (the streaming consumer for a mixed .orkpack): dump the COMPACT int4 nibble store
 * + per-channel scales (~half of the int8 ork_w_dump), and reload it straight into NPU DMA, inflating the
 * nibbles -> int8 (UNIFORM sign-extend / NF4 LUT, per the stored quant_kind) and re-tiling on load. Only
 * valid for an int4-packed weight (ork_i4a8_mm_pack); the LUT is derived from quant_kind, not stored.
 * Blob layout: { u32 magic 'O4N1', u32 version=1, i32 K, i32 N, u32 quant_kind } + bscale[N] (N f32) +
 * Bi4 (K*N/2 bytes). out=NULL -> required size; returns 0 if `w` has no int4 store. Loaded weight runs
 * via ork_i8_mm_run and re-dumps byte-identically. NULL on a malformed blob / shape mismatch. */
size_t       ork_i4a8_w_dump(const ork_w *w, void *out, size_t cap);
/* DT_I4_ROT_A8: load rotated int4 weights stored in the i4-NATIVE TILED blob format (what ork_w_dump
 * writes) and inflate them into int8 containers for the int8 MAC. ork_i4a8_mm_load handles the COMPACT
 * container instead (hdr + bscale + nibbles) and rejects this layout on its exact-size gate, which is why
 * the rotated-i4a8 tier never loaded. Scales come from the caller (this format carries none). */
ork_w *      ork_i4a8_mm_load_tiled(ork_npu *c, int K, int N, const void *blob, size_t n, int G);

/* OFFLINE codes of a CPU-backed weight (NULL when device-resident). For tests that must prove an offline
 * load recovered exactly the packed values; the un-tilers are index arithmetic and a wrong index still
 * produces plausible weights. */
const int8_t *ork_w_codes(const ork_w *w);

/* Resident IOVA bytes a weight of (K,N) will occupy at resident width `wbits` (4 = int4 nibbles, 8 = int8
 * containers, which is what an int4-on-disk i4a8/rot_a8 weight inflates to). Includes the full-K Bf
 * companion exactly when the loaders build one, plus per-tile page padding, and honours ORK_NO_BF itself.
 * want_bf: -1 follows ORK_NO_BF, 0/1 forces (a sizer needs both to test Bf against a RAM budget).
 * A multi-domain consumer MUST size from this rather than re-deriving it: the two models drift (they have),
 * and an under-count overflows a domain, whose failed IOVA allocation leaks a mapping until reboot. */
size_t       ork_w_resident_bytes(ork_npu *c, int K, int N, int wbits, int want_bf);

/* GROUPED int8 run: per-(channel, K-group) scales, int8 weights, int8 activations -- the W4A8 tier.
 * Offline is exact (shared kernel with the int4 twin); the on-device path is not implemented yet and
 * refuses loudly. aScale[m*Sk+g], bScale[g*N+n]; C is fp32. */
int          ork_i8_mm_run_grouped(ork_npu *c,ork_w *w,int M,const int8_t *A,const float *aScale,const float *bScale,float *C);

ork_w       *ork_i4a8_mm_load(ork_npu *ctx, int K, int N, const void *blob, size_t n);
/* Zero-copy IMPORT variant of ork_i4a8_mm_load: resident tiles are dma-bufs the NPU reads in place (PRIME
 * import); the int4 nibbles inflate -> int8 directly into them. Bit-identical to ork_i4a8_mm_load (same
 * blob, same tiled bytes, same re-dump). Returns NULL if import is unavailable (caller falls back to
 * ork_i4a8_mm_load) or on a malformed blob / shape mismatch. */
ork_w       *ork_i4a8_mm_load_import(ork_npu *ctx, int K, int N, const void *blob, size_t n);
/* Reload a pre-tiled NATIVE-W4A4 weight (ork_w_dump of a DT_I4 ork_i4_mm_pack weight) straight into NPU DMA —
 * NO dequant/FWHT-rotate/int4-quant/tile (the cold-pack fix for the mul_mat_i4/_hadamard/group_i4 path). blob =
 * this weight's Bb dump (Kp*Nc/2 int4 bytes/tile, pgup'd, pack order). The per-channel bscale is persisted by
 * the caller and re-attached separately. Returns a DT_I4 weight (run with ork_i4_mm_run) or NULL on mismatch. */
ork_w       *ork_i4_mm_load(ork_npu *ctx, int K, int N, const void *blob, size_t n);
/* As ork_i4_mm_load but ZERO-alloc via bimport (dma-heap + PRIME_FD) instead of bcreate (MEM_CREATE): the
 * multi-domain-safe path for a big resident int4 set (MEM_CREATE faults across domains / at scale). Use this
 * for a >4GiB resident int4 weight set (e.g. a resident MoE); falls back to ork_i4_mm_load on failure. */
ork_w       *ork_i4_mm_load_import(ork_npu *ctx, int K, int N, const void *blob, size_t n);
/* #54: consolidated int4 load — tiles are views into a shared per-domain bimport arena (few large chunks
 * across many experts) instead of one dma-buf per weight; avoids per-domain IOMMU mapping saturation on
 * big MoE. Weight owns nothing; arena is freed at ork_npu_free. int4 resident (no per-weight eviction). */
ork_w       *ork_i4_mm_load_arena(ork_npu *ctx, int K, int N, const void *blob, size_t n);

/* CPU-side pack/dump helpers linked by the ggml-ork backend (defined at the end of npu.c; no internal
 * callers). ork_i8_w_dump_cpu_st: single-threaded int8 CPU tile — for callers that parallelize at a
 * coarser grain (one whole tensor per core) so the internal pool wouldn't nest/oversubscribe; byte-
 * identical to ork_i8_w_dump_cpu. ork_i4a8_pack_cpu_blob: CPU int4 pack straight to the compact .orkpack
 * blob (bit-identical to ork_i4a8_mm_pack_im + ork_i4a8_w_dump) with NO bcreate/IOMMU/tiling.
 * ork_i8_mm_pack_import: tile int8 B[K,N] directly into IMPORTED dma-buf chunks (uniform ~16MB chunks,
 * no native-alloc outlier — for co-resident fused per-tensor weights). K%32,N%32; dump/pack out=NULL ->
 * required byte size. */
size_t       ork_i8_w_dump_cpu_st(ork_npu *ctx, int K, int N, const int8_t *B, void *out, size_t cap);
/* Device-tiled int4 blob -> the CPU GEMV's per-channel nibble layout (see ork_native_cpu.h). Lets M=1
 * decode run on the CPU from the PACK's int4 instead of the source gguf's q8. out=NULL -> required bytes
 * (N*K/2). Allocates a K*N int8 scratch internally, so call it per tensor. */
size_t       ork_i4_cpu_blob_from_tiled(ork_npu *ctx, int K, int N, const void *blob, size_t n, unsigned char *out, size_t cap);
size_t       ork_i4a8_pack_cpu_blob(ork_npu *ctx, int K, int N, const float *f32, const float *imatrix, int nf4, void *out, size_t cap);  /* nf4: 1=NF4 codebook, 0=uniform (caller routes by source: full-precision->NF4) */
/* GPTQ int4 weight quant (ork_gptq.c; in-tree, NO external deps). Error-compensated column-sequential rounding
 * with the calibration Hessian H=X^T X -> uniform symmetric int4 [-8,7] + per-(row,group) scale (native-W4A4
 * form: dequant w=code*scale; composes with Hadamard = QuaRot). W:[N(out)*K(in)] fp32; H:[K*K] (DESTROYED);
 * group<=0 => per-row; codes:[N*K] int8; scales:[N*ceil(K/group)] fp32; damp: Hessian damp (0.01 typical).
 * 0 ok, <0 on error. IMPLEMENTED 2026-08-22 (was a -ENOSYS stub). Validated by `test_gptq` in `make test`,
 * which supersedes task #56's AutoGPTQ byte-compare (that needed a Python dep the repo disallows, and an
 * analytic identity is the stronger check): H=I makes the factor chain collapse to Hinv=I, so GPTQ must emit
 * codes+scales BYTE-IDENTICAL to round-to-nearest — it does, 0/8192 and 0/256 — and on a structured H it cuts
 * the H-weighted error to 0.60x RTN. Cost is O(N*K^2) plus three O(K^3) factorisations: an offline pack step,
 * not inference. STILL NOT ROUTED: quantizer quality is an end-to-end question, so the pack keeps it behind
 * the pack gate ORK_GPTQ, which is not wired yet, until a PPL comparison (ork_ppl) shows it beats the
 * current int4 quantizer on a real model. */
int          ork_i4_gptq(int K, int N, const float *W, float *H, int group, int8_t *codes, float *scales, float damp);
ork_w       *ork_i8_mm_pack_import(ork_npu *ctx, int K, int N, const int8_t *B);

/* ---- Streaming weight pool: a RAM-resident inflated-int8 cache with CHEAP map/unmap ----
 * For models too big to keep resident in the ~4 GiB NPU IOVA window. The caller (e.g. a layer/expert LRU)
 * keeps a set of ALREADY-INFLATED int8 weights resident in CPU RAM (budget by RAM — much larger than the
 * IOVA window) and maps/unmaps them to IOVA cheaply on demand: a cache HIT pays only the cheap MEM_CREATE
 * import (~170us@4MB), skipping the expensive int4->int8 inflate (paid ONCE on add) and the expensive
 * MEM_DESTROY (paid only on eviction). The pool provides the lifecycle ONLY — the eviction/LRU POLICY and
 * the RAM budget live in the caller. Both stores: int8 (fill=copy ork_w_dump bytes) and int4 (fill=inflate
 * the ork_i4a8_w_dump nibbles). A transient prefetch double-buffer is just a small pool the caller fills
 * ahead on a thread. All ops bit-exact vs the equivalent ork_mm_load_*. NULL/create-fail if the dma-heap
 * is absent (caller falls back to ork_i8_mm_load / ork_i4a8_mm_load + ork_i8_mm_run). */
typedef struct ork_stream_pool  ork_stream_pool;
typedef struct ork_stream_entry ork_stream_entry;
ork_stream_pool  *ork_stream_pool_create(ork_npu *ctx);                 /* NULL if import unavailable */
ork_stream_entry *ork_i4a8_stream_pool_add(ork_stream_pool *p, int K, int N, const void *blob, size_t n); /* int4 store; fill=inflate (once) */
ork_stream_entry *ork_i8_stream_pool_add  (ork_stream_pool *p, int K, int N, const void *blob, size_t n); /* int8 store; fill=copy (once) */
/* int8 store from a freshly-quantized UNTILED int8 B[K][N] (row-major): tile straight into RAM staging,
 * NO transient IOVA pack (no bcreate). For a caller that just quantized a weight and wants it in the pool
 * without packing to IOVA first (which competes with the pool's mapped hot set). NULL on bad dims. */
ork_stream_entry *ork_i8_stream_pool_add_raw(ork_stream_pool *p, int K, int N, const int8_t *B);
int               ork_stream_pool_map  (ork_stream_pool *p, ork_stream_entry *e);  /* CHEAP: MEM_CREATE import -> IOVA (per submit) */
int               ork_stream_pool_run  (ork_stream_pool *p, ork_stream_entry *e, int M, const int8_t *A, int32_t *C);
void              ork_stream_pool_unmap(ork_stream_pool *p, ork_stream_entry *e);  /* MEM_DESTROY; entry STAYS in RAM */
void              ork_stream_pool_remove(ork_stream_pool *p, ork_stream_entry *e); /* free the RAM buffer (caller's evict) */
void              ork_stream_pool_free (ork_stream_pool *p);
size_t            ork_stream_entry_bytes (const ork_stream_entry *e);   /* RAM bytes held (for the caller's budget) */
int               ork_stream_entry_mapped(const ork_stream_entry *e);   /* 1 if currently IOVA-mapped */

/**
 * @brief Run C[M,N] = A[M,K] x packed weights (fp16). Run dtype must match the pack dtype.
 * @param w Weight from ork_f16_mm_pack().
 * @param M Activation rows; any M>=1 (tiled + scheduled internally, no caller-visible cap).
 * @param A Row-major fp16 activations, M*K elements.
 * @param C Output, M*N fp32, row-major (caller-allocated). May be an ork_dma_alloc() buffer (zero-copy).
 * @return 0 on success, negative on error (bad args / submit failure).
 */
int          ork_f16_mm_run   (ork_npu *ctx, ork_w *w, int M, const ork_f16 *A, float   *C);
#endif /* ORK_WEIGHTS_H */
