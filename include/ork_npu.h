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

/* Library version — matches the git tag on the ork-driver repo. */
#define ORK_NPU_VERSION "0.2.0"
const char  *ork_npu_version(void);  /* returns ORK_NPU_VERSION */

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

/* Zero-copy DMA buffers (NPU-coherent, CPU-mapped). Allocate the activation A and/or output C here
 * and the matmul reads/writes them in place — no host gather/writeout memcpy (the ~33% prefill
 * residual vs the closed runtime). ork_mm_run detects residency automatically; pass the returned
 * pointer as A/C exactly as a malloc'd one. NULL on failure or table-full (fall back to malloc). */
void        *ork_dma_alloc(ork_npu *ctx, size_t size);
void         ork_dma_free (ork_npu *ctx, void *ptr);

/* Pack + upload B[K,N] (row-major) into NPU-resident tile layout; reuse across runs.
 * fp16: K%32==0, N%16==0.  int8: K%32==0, N%32==0.  Returns NULL on bad dims. */
ork_w       *ork_mm_pack   (ork_npu *ctx, int K, int N, const ork_f16  *B);  /* fp16 weights */
ork_w       *ork_mm_pack_i8(ork_npu *ctx, int K, int N, const int8_t   *B);  /* int8/w8a8 weights */
/* re-tile int8 B into an existing same-shape ork_w (reuses its DMA; no alloc/free) — for pooling
 * reused weights (MoE experts) without churning/fragmenting the NPU IOMMU. 0 ok / -1 / -2 mismatch. */
int          ork_mm_repack_i8(ork_npu *ctx, ork_w *w, int K, int N, const int8_t *B);
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

/* RE/calibration only: probe this SoC's single-submit K-tile ceiling. Runs ONE M=1 full-K int8
 * submit at (K,N) (N <= SoC N-cap, K%32, N%32) on its own buffers. Returns 0 if the submit
 * completed (C[N] int32 valid — validate vs CPU), -1 if it wedged (K exceeds the per-op K-tile
 * cap; recoverable), -2 on bad dims. See tools/ksubmit_probe.c. */
int          ork_npu_probe_single_i8(ork_npu *ctx, int K, int N, const int8_t *A, const int8_t *B, int32_t *C);

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

/* Math utilities for caller-driven quantization/transformations */
void         ork_fwht_norm(float *v, int n);

#endif
