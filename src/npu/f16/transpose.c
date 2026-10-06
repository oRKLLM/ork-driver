/* f16/transpose.c — host operand transpose, blocked + NEON.
 *
 * WHY THIS IS A HOST OP. A transpose is the largest single cost in a training step — a naive
 * [4096,1024] fp32→fp16 transpose measures 51.5 ms on an A76, more than a forward matmul and 34% of
 * a dW — so the first question was whether the NPU could do it. It cannot, today:
 *   - an on-device contiguous→cube relayout IS the vendor RESHAPE op (OPS_REGISTRY, the attention
 *     FRONT row). That campaign is BLOCKED on undecoded read geometry (0x107c / 0x1080) and sits at
 *     RE-stage behind reshape_probe_f16. The fp16 matmul HANGS writing the cube; int8 writes LINEAR.
 *   - ork_bmm_fp16_strided does not transpose on the device either: it gathers each strided operand
 *     "into contiguous scratch, then the dense pack/run path runs unchanged" (ork/bmm.h). Routing a
 *     transpose through it relocates this loop into the library; it does not remove it.
 *   - and a matmul cannot express it: C[M,N] = A[M,K]·B[K,N] always carries A's rows and B's
 *     columns, so Xᵀ can only come from the weight PACK (already a host tile-transpose) or RESHAPE.
 * So it is the host's job, and the only question is how well the host does it. If the RESHAPE decode
 * ever lands, revisit — the call sites here are the ones that would move.
 *
 * WHY IT IS FAST. The naive loop strides on one side and runs at ~1 GB/s of output on a core that
 * streams ~8 GB/s: it is a cache-miss benchmark, not a bandwidth one. Cache-blocking recovers most
 * of it (3.6×), and a register transpose inside the block recovers the rest. Measured on the board
 * (`tools/f16_transpose_probe.c`, A76 @ 2.4 GHz, governors pinned):
 *
 *   fp32[R,C] → fp16[C,R]      naive     blk32     neon        vs naive
 *     [4096,1024]             51542us   14754us    8087us        6.4×
 *     [1024,3584]             77402us    8258us    6074us       12.7×
 *     [3584,1024]             24254us    7074us    4414us        5.5×
 *     [4096,3584]            403237us   53990us   27120us       14.9×
 *   fp16 → fp16
 *     [4096,1024]             53595us   14719us    6335us        8.5×
 *     [4096,3584]            357488us   50079us   24393us       14.7×
 *
 * THE BIGGER LEVER IS NOT TRANSPOSING. dW[K,N] = Xᵀ·dY needs Xᵀ ([M,K] transposed), but the same
 * gradient in the other orientation, dWᵀ[N,K] = dYᵀ·X, needs dYᵀ ([M,N] transposed) and takes X as
 * the weight in its NATURAL layout. So a caller is free to transpose whichever of the two is
 * smaller and store that weight's master transposed — the optimizer is elementwise, so the layout
 * costs nothing. On a 0.75B FFN that is M×1024 either way instead of M×3584 on gate/up: 3.5× less
 * transpose work for no work at all. This is the same operand-swap that removed the Kᵀ pack from
 * the attention bridge (OPS_REGISTRY "method (b)"). Use it before reaching for a faster memcpy.
 *
 * THREADING: deliberately none. The library's NONBLOCK doorbell spine already lets a caller overlap
 * host work with an in-flight submit (ork_dyn_* / colsplit), so the transpose for step N+1 belongs
 * in the window where the NPU is busy with step N — not in a thread pool this file would own.
 */
#include <stdlib.h>   /* internal.h's inline env-knob readers call getenv/atoi */
#include <string.h>
#include "../internal.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define ORK_T_NEON 1
#endif

/* 64 keeps a tile's source rows in L1 on the A76 (64 KB D$) at both element sizes; measured equal
 * to or better than 16/32 on the NEON path at every probed shape. */
#define TBLK 64

/* ---- fp16 [R,C] -> fp16 [C,R] ---------------------------------------------------------------- */
#ifdef ORK_T_NEON
/* 8x8 block, three TRN stages widening 16 -> 32 -> 64 bits. TRN and not ZIP: a transpose exchanges
 * 2x2 blocks, which is what TRN does; ZIP interleaves whole vectors and silently produces a
 * permutation that is not the transpose (it passed timing and failed the elementwise check). The
 * data is moved as u16 throughout — no fp16 arithmetic — so this needs no +fp16 ISA level. */
static inline void t16_blk8(ork_f16 *d, const ork_f16 *s, int R, int C, int r, int c)
{
    uint16x8_t a[8];
    for (int i = 0; i < 8; i++) a[i] = vld1q_u16((const uint16_t *)s + (size_t)(r + i) * C + c);
    uint16x8x2_t p0 = vtrnq_u16(a[0], a[1]), p1 = vtrnq_u16(a[2], a[3]),
                 p2 = vtrnq_u16(a[4], a[5]), p3 = vtrnq_u16(a[6], a[7]);
    uint32x4x2_t q0 = vtrnq_u32(vreinterpretq_u32_u16(p0.val[0]), vreinterpretq_u32_u16(p1.val[0])),
                 q1 = vtrnq_u32(vreinterpretq_u32_u16(p0.val[1]), vreinterpretq_u32_u16(p1.val[1])),
                 q2 = vtrnq_u32(vreinterpretq_u32_u16(p2.val[0]), vreinterpretq_u32_u16(p3.val[0])),
                 q3 = vtrnq_u32(vreinterpretq_u32_u16(p2.val[1]), vreinterpretq_u32_u16(p3.val[1]));
#define ORK_J(x, y, half) vreinterpretq_u16_u64(vcombine_u64(                     \
            vget_##half##_u64(vreinterpretq_u64_u32(x)),                          \
            vget_##half##_u64(vreinterpretq_u64_u32(y))))
    uint16x8_t o[8];
    o[0] = ORK_J(q0.val[0], q2.val[0], low);  o[4] = ORK_J(q0.val[0], q2.val[0], high);
    o[1] = ORK_J(q1.val[0], q3.val[0], low);  o[5] = ORK_J(q1.val[0], q3.val[0], high);
    o[2] = ORK_J(q0.val[1], q2.val[1], low);  o[6] = ORK_J(q0.val[1], q2.val[1], high);
    o[3] = ORK_J(q1.val[1], q3.val[1], low);  o[7] = ORK_J(q1.val[1], q3.val[1], high);
#undef ORK_J
    for (int i = 0; i < 8; i++) vst1q_u16((uint16_t *)d + (size_t)(c + i) * R + r, o[i]);
}
#endif

int ork_f16_transpose(ork_f16 *dst, const ork_f16 *src, int rows, int cols)
{
    if (!dst || !src || rows <= 0 || cols <= 0) return -1;
    if (dst == (ork_f16 *)src) return -1;                 /* no in-place: the walk would self-clobber */
    int R = rows, C = cols, rv = 0, cv = 0;
#ifdef ORK_T_NEON
    rv = R & ~7; cv = C & ~7;
    for (int r0 = 0; r0 < rv; r0 += TBLK)
        for (int c0 = 0; c0 < cv; c0 += TBLK) {
            int rm = r0 + TBLK < rv ? r0 + TBLK : rv, cm = c0 + TBLK < cv ? c0 + TBLK : cv;
            for (int r = r0; r < rm; r += 8)
                for (int c = c0; c < cm; c += 8) t16_blk8(dst, src, R, C, r, c);
        }
#else
    for (int r0 = 0; r0 < R; r0 += TBLK)
        for (int c0 = 0; c0 < C; c0 += TBLK) {
            int rm = r0 + TBLK < R ? r0 + TBLK : R, cm = c0 + TBLK < C ? c0 + TBLK : C;
            for (int r = r0; r < rm; r++)
                for (int c = c0; c < cm; c++) dst[(size_t)c * R + r] = src[(size_t)r * C + c];
        }
    return 0;
#endif
    for (int r = rv; r < R; r++)                          /* ragged edges, scalar */
        for (int c = 0; c < C; c++) dst[(size_t)c * R + r] = src[(size_t)r * C + c];
    for (int r = 0; r < rv; r++)
        for (int c = cv; c < C; c++) dst[(size_t)c * R + r] = src[(size_t)r * C + c];
    return 0;
}

/* ---- fp32 [R,C] -> fp16 [C,R] (fused transpose + narrow) -------------------------------------- */
#ifdef ORK_T_NEON
/* 4x4 floats: four contiguous loads, a register transpose, four fp16 stores down the columns.
 * vcvt_f16_f32 is ARMv8-A baseline, and the result is stored through u16 so no fp16 ISA level is
 * required here either. Rounding is the hardware's round-to-nearest-even — identical to the scalar
 * (ork_f16) cast, which the example asserts elementwise rather than within a tolerance. */
static inline void t32_blk4(ork_f16 *d, const float *s, int R, int C, int r, int c)
{
    float32x4_t a0 = vld1q_f32(s + (size_t)(r + 0) * C + c), a1 = vld1q_f32(s + (size_t)(r + 1) * C + c);
    float32x4_t a2 = vld1q_f32(s + (size_t)(r + 2) * C + c), a3 = vld1q_f32(s + (size_t)(r + 3) * C + c);
    float32x4x2_t t0 = vtrnq_f32(a0, a1), t1 = vtrnq_f32(a2, a3);
    float32x4_t b0 = vcombine_f32(vget_low_f32 (t0.val[0]), vget_low_f32 (t1.val[0]));
    float32x4_t b1 = vcombine_f32(vget_low_f32 (t0.val[1]), vget_low_f32 (t1.val[1]));
    float32x4_t b2 = vcombine_f32(vget_high_f32(t0.val[0]), vget_high_f32(t1.val[0]));
    float32x4_t b3 = vcombine_f32(vget_high_f32(t0.val[1]), vget_high_f32(t1.val[1]));
    vst1_u16((uint16_t *)d + (size_t)(c + 0) * R + r, vreinterpret_u16_f16(vcvt_f16_f32(b0)));
    vst1_u16((uint16_t *)d + (size_t)(c + 1) * R + r, vreinterpret_u16_f16(vcvt_f16_f32(b1)));
    vst1_u16((uint16_t *)d + (size_t)(c + 2) * R + r, vreinterpret_u16_f16(vcvt_f16_f32(b2)));
    vst1_u16((uint16_t *)d + (size_t)(c + 3) * R + r, vreinterpret_u16_f16(vcvt_f16_f32(b3)));
}
#endif

int ork_f32_transpose_to_f16(ork_f16 *dst, const float *src, int rows, int cols)
{
    if (!dst || !src || rows <= 0 || cols <= 0) return -1;
    int R = rows, C = cols, rv = 0, cv = 0;
#ifdef ORK_T_NEON
    rv = R & ~3; cv = C & ~3;
    for (int r0 = 0; r0 < rv; r0 += TBLK)
        for (int c0 = 0; c0 < cv; c0 += TBLK) {
            int rm = r0 + TBLK < rv ? r0 + TBLK : rv, cm = c0 + TBLK < cv ? c0 + TBLK : cv;
            for (int r = r0; r < rm; r += 4)
                for (int c = c0; c < cm; c += 4) t32_blk4(dst, src, R, C, r, c);
        }
#else
    for (int r0 = 0; r0 < R; r0 += TBLK)
        for (int c0 = 0; c0 < C; c0 += TBLK) {
            int rm = r0 + TBLK < R ? r0 + TBLK : R, cm = c0 + TBLK < C ? c0 + TBLK : C;
            for (int r = r0; r < rm; r++)
                for (int c = c0; c < cm; c++) dst[(size_t)c * R + r] = (ork_f16)src[(size_t)r * C + c];
        }
    return 0;
#endif
    for (int r = rv; r < R; r++)
        for (int c = 0; c < C; c++) dst[(size_t)c * R + r] = (ork_f16)src[(size_t)r * C + c];
    for (int r = 0; r < rv; r++)
        for (int c = cv; c < C; c++) dst[(size_t)c * R + r] = (ork_f16)src[(size_t)r * C + c];
    return 0;
}
