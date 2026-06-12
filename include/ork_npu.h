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

/* Open the NPU, detect the SoC, power on. Returns NULL on failure (no NPU / no perms). */
ork_npu     *ork_npu_init(void);
void         ork_npu_free(ork_npu *ctx);

/* SoC introspection */
const char  *ork_npu_soc(const ork_npu *ctx);    /* "rk3588", "rk3576", ... */
int          ork_npu_cores(const ork_npu *ctx);  /* NPU core count */
int          ork_npu_validated(const ork_npu *ctx); /* 1 if this SoC's params are HW-validated */

/* Pack + upload B[K,N] (row-major fp16) into NPU-resident tile layout; reuse across runs.
 * Requires K%32==0 and N%16==0. Returns NULL on bad dims. */
ork_w       *ork_mm_pack(ork_npu *ctx, int K, int N, const ork_f16 *B);
void         ork_w_free(ork_w *w);

/* C[M,N] (fp32, row-major) = A[M,K] (fp16, row-major) x packed weights. Returns 0 on ok. */
int          ork_mm_run(ork_npu *ctx, ork_w *w, int M, const ork_f16 *A, float *C);

#endif
