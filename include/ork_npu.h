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
 * stream activations through ork_f16_mm_run. Arbitrary M/K/N (K-split + N-tiling internally).
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
#define ORK_NPU_VERSION "1.0.53"
/* On-disk .orkpack format version — DECOUPLED from the library MAJOR. Bump this ONLY when the persisted bytes'
 * meaning changes (tile layout/geometry or quant rule); it stays at the MAJOR of the last format-changing
 * release. The 1.0.0 release did NOT change the format, so it stays 0 (existing .orkpacks remain valid). */
#define ORK_PACK_FORMAT_VERSION 0u
/* The API is split across the ork/ headers purely for readability; this umbrella is the entry point
 * and every consumer keeps including <ork_npu.h> unchanged. Order matters: the base typedefs
 * and version macros above must precede the parts, and the parts are included in their
 * original file order so no declaration moves ahead of a type it needs. */
#include "ork/context.h"
#include "ork/dma.h"
#include "ork/weights.h"
#include "ork/run.h"
#include "ork/sdp.h"      /* production SDP/activation ops — has real callers */
#include "ork/probe.h"    /* RE probes/replays/fuzz — tools-only, not the supported API */
#include "ork/dynamic.h"
#include "ork/seq.h"
#include "ork/chain.h"
#include "ork/bmm.h"

#endif
