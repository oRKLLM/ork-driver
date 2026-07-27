/* slice_dbrun_probe — increment-1 execution proof for slice-and-dice (SLICE_AND_DICE_PLAN.md).
 * Runs a WIDE-K int8 matmul (e.g. ffn_down K=6144) entirely on the DOORBELL by K-slicing it into c_base
 * tiles (each K<=4096, has Bf), one nonblocking ork_dyn submit per slice, int32-accumulating the partials.
 * Compares BIT-EXACT vs the blocking full-K reference (ork_mm_run_i8). This is the shape that WEDGED when
 * submitted whole (c_wideK, M>1, unverified); sliced into c_base it must run wedge-free + bit-exact.
 *   make slice_dbrun_probe && sudo ./slice_dbrun_probe [K=6144] [N=2048] [M=256]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include "ork_slice.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static uint32_t g = 2463534242u;
static int8_t r8(void) { g ^= g << 13; g ^= g >> 17; g ^= g << 5; return (int8_t)(((int)(g & 0x3f)) - 31); }  /* [-31,32] */
static inline void civac(volatile void *p){ __asm__ volatile("dc civac,%0"::"r"(p):"memory"); }

int main(int argc, char **argv) {
    int K = argc > 1 ? atoi(argv[1]) : 6144, N = argc > 2 ? atoi(argv[2]) : 2048, M = argc > 3 ? atoi(argv[3]) : 256;
    setvbuf(stdout, 0, _IONBF, 0);
    ork_npu *c = ork_npu_init(); if (!c) { printf("init fail\n"); return 1; }
    ork_slice_caps cap = ork_slice_caps_rk3588(); cap.mmax = M;   /* full-M per tile: the c_base doorbell M-tiles internally */
    printf("slice_dbrun: K=%d N=%d M=%d (kmax=%d) -> %d K-slices\n", K, N, M, cap.kmax, ork_slice_kparts(K, &cap));

    int8_t *A = (int8_t *) malloc((size_t) M*K), *B = (int8_t *) malloc((size_t) K*N);
    for (size_t i = 0; i < (size_t) M*K; i++) A[i] = r8();
    for (size_t i = 0; i < (size_t) K*N; i++) B[i] = r8();

    /* --- reference: blocking full-K matmul (the path slice-and-dice replaces) --- */
    ork_w *wf = ork_mm_pack_i8(c, K, N, B); if (!wf) { printf("ref pack fail\n"); return 1; }
    int32_t *Cref = (int32_t *) malloc((size_t) M*N*4);
    if (ork_mm_run_i8(c, wf, M, A, Cref)) { printf("ref run FAIL\n"); return 1; }
    ork_mm_free(c, wf);

    /* --- slice-and-dice on the doorbell: K-slice, one c_base ork_dyn submit each, int32-accumulate --- */
    const int ks = (cap.kmax / cap.kmul) * cap.kmul;
    int32_t *Cacc  = (int32_t *) calloc((size_t) M*N, 4);
    int8_t  *Aslc  = (int8_t  *) malloc((size_t) M*ks);                  /* gathered A[:, k0:k1] */
    int32_t *Cpart = (int32_t *) ork_dma_alloc(c, (size_t) M*N*4);       /* doorbell C (resident) */
    if (!Cpart) { printf("dma C fail\n"); return 1; }
    int nslice = 0, fellback = 0;
    for (int k0 = 0; k0 < K; k0 += ks) {
        int k1 = k0 + ks < K ? k0 + ks : K, Ks = k1 - k0;
        for (int m = 0; m < M; m++) memcpy(Aslc + (size_t) m*Ks, A + (size_t) m*K + k0, Ks);   /* gather A cols */
        ork_w *ws = ork_mm_pack_i8(c, Ks, N, B + (size_t) k0*N);          /* B rows k0:k1 are contiguous */
        if (!ws) { printf("slice pack FAIL k0=%d Ks=%d\n", k0, Ks); return 1; }
        ork_mm_task_i8 t = { ws, M, Aslc, Cpart };
        ork_dyn_chain *h = ork_dyn_begin_mc(c, 1, &t, 0);                 /* S==1, c_base -> colsplit (verified any-M) */
        if (!h) { fellback = 1; if (ork_mm_run_i8(c, ws, M, Aslc, Cpart)) { printf("slice run FAIL\n"); return 1; } }
        else if (ork_dyn_end(h) < 0) { printf("slice dyn_end FAIL k0=%d\n", k0); return 1; }
        for (size_t i = 0; i < (size_t) M*N; i++) { civac(Cpart + i); Cacc[i] += Cpart[i]; }   /* read-after-drain + accumulate */
        ork_mm_free(c, ws); nslice++;
    }

    long bad = 0, first = -1;
    for (size_t i = 0; i < (size_t) M*N; i++) if (Cacc[i] != Cref[i]) { if (first < 0) first = (long) i; bad++; }
    printf("%d K-slices via doorbell%s | mismatches = %ld / %d", nslice, fellback ? " (some fell back to blocking!)" : "", bad, M*N);
    if (bad) printf("  first@%ld: got %d want %d", first, Cacc[first], Cref[first]);
    printf(" -> %s\n", bad ? "FAIL" : "BIT-EXACT PASS (wide-K runs wedge-free on the doorbell)");
    return bad ? 1 : 0;
}
