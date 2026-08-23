/* ork/context.h — Device context, SoC introspection, core budget, IOMMU domains
 *
 * Lifecycle (init/free), what chip we are on and what it can do, the per-core budget and
 * priority knobs, and the IOMMU-domain surface that the >4 GiB weight sets are built on.
 *
 * Part of the public API. Do NOT include this directly — include <ork_npu.h>, which is the
 * umbrella and the only supported entry point; these parts are a readability split of it
 * (ork_npu.h was 1519 lines) and their boundaries may move. Types live in ork_npu.h above
 * the includes, so this header is not self-contained by design. */
#ifndef ORK_CONTEXT_H
#define ORK_CONTEXT_H
/** @brief Runtime library version. @return "MAJOR.MINOR.PATCH" or "MAJOR.MINOR.PATCH+g<hash>". */
const char  *ork_npu_version(void);

/**
 * @brief On-disk pack-format (.orkpack) compatibility token — the library's MAJOR version.
 *
 * A persisted weight (ork_w_dump / ork_i4a8_w_dump) is only binary-compatible with builds that share
 * this value (ORK_PACK_FORMAT_VERSION). It tracks the MAJOR of the last format-changing release but is
 * DECOUPLED from ORK_NPU_VERSION's MAJOR (a library major bump that does NOT touch the on-disk bytes — e.g.
 * the 1.0.0 stability release — keeps this at its prior value so existing .orkpacks stay valid). Bump it
 * only on a real persisted-bytes change: a resident tile LAYOUT / geometry change (K-slice size, the SoC output-width
 * cap / N-tiling, the 32x32 block or Bb dump order) or a weight QUANT change (int8/int4 scale rule,
 * int4 nibble packing, NF4 codebook).
 *
 * ork-driver stamps this into the int4 blob header and rejects a mismatch on load (ork_i4a8_mm_load
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
 * @brief Open the NPU DIRECTLY (in-process), detect the SoC from the device tree, and power it on. This is
 *        the DEFAULT transport: the process owns the single-stream NPU (do not run concurrent direct-NPU
 *        processes — they wedge the IOMMU). For back-compat, the legacy ORK_USE_ORKD=1 env redirects this to
 *        the orkd client; new callers should instead choose the transport by calling the desired entry point.
 * @return Device context (one per process), or NULL on failure — no NPU present, or no permission to
 *         open /dev/dri/cardN (the process needs access to the DRM render node).
 */
ork_npu     *ork_npu_init(void);
/* Offline context: SoC caps only, NO device (fd = -1). Valid ONLY for the CPU-side surfaces — the
 * *_w_dump_cpu weight tilers — so an .orkpack can be built on any machine instead of on the board.
 * soc_id must be given explicitly ("rk3588"); there is no device tree to detect from. Any op that
 * needs hardware fails rather than running. Free with ork_npu_free as usual. */
ork_npu     *ork_npu_init_offline(const char *soc_id);
/**
 * @brief Open an orkd CLIENT context: connect (auto-spawn) the orkd daemon and route every ork_mm_* through
 *        it. The daemon owns the single-stream NPU and serializes all submits — the safe way to share it
 *        across concurrent processes. This is the counterpart entry point to ork_npu_init(): callers select
 *        the transport by CHOOSING the function (env is no longer required).
 * @return orkd-client context, or NULL if the daemon can't be reached/spawned (NO silent fallback to direct).
 */
ork_npu     *ork_npu_init_orkd(void);
/** @brief Power off the NPU and free a context obtained from ork_npu_init() / ork_npu_init_orkd(). */
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
 * map/unmap) by placing its weights across several domains. Call this before ork_i8_mm_pack /
 * ork_i8_mm_load (and the fp16/int4 variants): each weight packed/loaded afterward lands its resident
 * tiles in `domain` and records it; ork_f16_mm_run* then submits that weight's matmuls against the same
 * domain automatically. domain<0 reverts to the process default (env ORK_IOMMU_DOMAIN, else 0). */
void         ork_npu_set_pack_domain(ork_npu *ctx, int domain);
int          ork_npu_pack_domain(const ork_npu *ctx);   /* current pack domain (for save/restore) */
/* The currently ACTIVE IOMMU domain. Pure getter. Use it to place a TRANSIENT/scratch weight (e.g. an
 * attention QK^T / A.V or GDN chunk bmm's dynamic operand) in the domain that is already active, so
 * running it costs NO domain switch — a switch is where a stuck job stalls the next submit for 60 s. */
int          ork_npu_active_domain(const ork_npu *ctx);
void         ork_npu_activate_domain(ork_npu *ctx, int domain);   /* make domain active (establish); alloc in-domain buffers after this */
int          ork_w_domain(const ork_w *w);   /* the IOMMU domain a packed weight resides in */
int          ork_npu_uses_orkd(const ork_npu *ctx);   /* 1 = context routes through orkd (serialized); 0 = direct single-stream NPU */
/* Client-managed IOMMU domains. ork_npu_domain_alloc reserves a domain (Path B: from orkd's coordinated pool
 * so multi-process clients don't collide; direct: a local id). Pack into it with ork_npu_set_pack_domain, then
 * ork_npu_domain_free when done (also auto-reclaimed when the context/connection is freed). id>0 ok, <0 error. */
int          ork_npu_domain_alloc(ork_npu *ctx);
int          ork_npu_domain_free(ork_npu *ctx, int domain);
/* Pre-size the ctx for `n` IOMMU domains (the per-domain anchor + parked-scratch arrays grow to fit). The
 * backend calls this once with the auto-sizer's domain count so no per-weight array grow happens mid-load.
 * There is no fixed domain cap — `n` is whatever the auto-sizer computes. No-op for n<=1 (single-domain). */
void         ork_npu_set_ndomains(ork_npu *ctx, int n);

#endif /* ORK_CONTEXT_H */
