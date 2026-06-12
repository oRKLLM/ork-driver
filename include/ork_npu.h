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

/* Pack + upload B[K,N] (row-major) into NPU-resident tile layout; reuse across runs.
 * fp16: K%32==0, N%16==0.  int8: K%32==0, N%32==0.  Returns NULL on bad dims. */
ork_w       *ork_mm_pack   (ork_npu *ctx, int K, int N, const ork_f16  *B);  /* fp16 weights */
ork_w       *ork_mm_pack_i8(ork_npu *ctx, int K, int N, const int8_t   *B);  /* int8/w8a8 weights */
void         ork_w_free(ork_w *w);

/* C[M,N] = A[M,K] x packed weights. Run dtype must match the pack dtype. Returns 0 on ok.
 *   fp16: A fp16 (row-major), C fp32.   int8: A int8 (row-major), C int32. */
int          ork_mm_run   (ork_npu *ctx, ork_w *w, int M, const ork_f16 *A, float   *C);
int          ork_mm_run_i8(ork_npu *ctx, ork_w *w, int M, const int8_t  *A, int32_t *C);

/* RE/calibration only: probe this SoC's single-submit K-tile ceiling. Runs ONE M=1 full-K int8
 * submit at (K,N) (N <= SoC N-cap, K%32, N%32) on its own buffers. Returns 0 if the submit
 * completed (C[N] int32 valid — validate vs CPU), -1 if it wedged (K exceeds the per-op K-tile
 * cap; recoverable), -2 on bad dims. See tools/ksubmit_probe.c. */
int          ork_npu_probe_single_i8(ork_npu *ctx, int K, int N, const int8_t *A, const int8_t *B, int32_t *C);

/* RE/calibration only: probe in-place K-slicing of a full-K weight buffer. Packs B[Kfull,N] fp16
 * full-K, runs one M=1 submit over k in [0,Kp), with up to `nov` regcmd overrides (block 0x0201)
 * to hunt the per-N-tile weight stride register. C[N] should equal the Kp-partial sum if slicing
 * works. Returns 0/ok, -1 wedged, -2 bad dims. See tools/slice_probe.c. */
int          ork_npu_probe_slice_f16(ork_npu *ctx, int Kfull, int N, int Kp, int nov,
                                     const uint32_t *ovr_reg, const uint32_t *ovr_val,
                                     const ork_f16 *A, const ork_f16 *B, float *C);

#endif
