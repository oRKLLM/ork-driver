/* slice_dbrun_probe — execution proof for slice-and-dice (SLICE_AND_DICE_PLAN.md), GENERAL K/N/M.
 * Runs an arbitrary int8 matmul entirely on the DOORBELL by decomposing it (ork_slice.h) into c_base tiles
 * and executing the FULL plan: for each (M-tile, N-tile) output block, accumulate int32 over its K-slices,
 * each K-slice a single nonblocking ork_dyn submit. Wide-K -> K-slice+accumulate; wide-N -> N-tile+scatter;
 * big-M -> M-tile. Compares BIT-EXACT vs the blocking full reference (ork_mm_run_i8).
 *   make slice_dbrun_probe && sudo ./slice_dbrun_probe [K=6144] [N=2048] [M=256] [nc=0(all cores)]
 * (nc=1 sidesteps the N-colsplit driver bug #36 to validate the slicer's own correctness on wide-N.)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include "ork_slice.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static uint32_t g = 2463534242u;
static int8_t r8(void) { g ^= g << 13; g ^= g >> 17; g ^= g << 5; return (int8_t)(((int)(g & 0x3f)) - 31); }
static inline void civac(volatile void *p){ __asm__ volatile("dc civac,%0"::"r"(p):"memory"); }

int main(int argc, char **argv) {
    int K = argc > 1 ? atoi(argv[1]) : 6144, N = argc > 2 ? atoi(argv[2]) : 2048, M = argc > 3 ? atoi(argv[3]) : 256;
    int nc = argc > 4 ? atoi(argv[4]) : 0;
    setvbuf(stdout, 0, _IONBF, 0);
    ork_npu *c = ork_npu_init(); if (!c) { printf("init fail\n"); return 1; }
    ork_slice_caps cap = ork_slice_caps_rk3588(); cap.mmax = M;   /* M handled by the c_base doorbell internally */

    static ork_tile tiles[8192];
    int nt = ork_slice_matmul(K, N, M, &cap, tiles, (int)(sizeof tiles / sizeof tiles[0]));
    if (nt <= 0) { printf("decompose FAIL rc=%d\n", nt); return 1; }
    printf("slice_dbrun: K=%d N=%d M=%d nc=%d -> %d tiles (%d K-slices)\n", K, N, M, nc, nt, ork_slice_kparts(K, &cap));

    int8_t *A = (int8_t *) malloc((size_t) M*K), *B = (int8_t *) malloc((size_t) K*N);
    for (size_t i = 0; i < (size_t) M*K; i++) A[i] = r8();
    for (size_t i = 0; i < (size_t) K*N; i++) B[i] = r8();

    /* reference: blocking full matmul */
    ork_w *wf = ork_mm_pack_i8(c, K, N, B); if (!wf) { printf("ref pack fail\n"); return 1; }
    int32_t *Cref = (int32_t *) malloc((size_t) M*N*4);
    if (ork_mm_run_i8(c, wf, M, A, Cref)) { printf("ref run FAIL\n"); return 1; }
    ork_mm_free(c, wf);

    /* execute the full tile plan on the doorbell. scratch sized for the largest tile. */
    int32_t *Cacc  = (int32_t *) calloc((size_t) M*N, 4);
    int8_t  *Aslc  = (int8_t  *) malloc((size_t) M * cap.kmax);            /* A[:, k0:k1] gathered */
    int8_t  *Bslc  = (int8_t  *) malloc((size_t) cap.kmax * cap.nmax);     /* B[k0:k1, n0:n1] gathered */
    int32_t *Cpart = (int32_t *) ork_dma_alloc(c, (size_t) M * cap.nmax * 4);
    if (!Cpart) { printf("dma C fail\n"); return 1; }
    int fellback = 0;
    for (int i = 0; i < nt; i++) {
        int k0 = tiles[i].k0, k1 = tiles[i].k1, Ks = k1 - k0;
        int n0 = tiles[i].n0, n1 = tiles[i].n1, Nw = n1 - n0;
        int m0 = tiles[i].m0, m1 = tiles[i].m1, Mw = m1 - m0;
        for (int m = 0; m < Mw; m++) memcpy(Aslc + (size_t) m*Ks, A + (size_t)(m0+m)*K + k0, Ks);   /* gather A cols */
        for (int k = 0; k < Ks; k++) memcpy(Bslc + (size_t) k*Nw, B + (size_t)(k0+k)*N + n0, Nw);   /* gather B block */
        ork_w *ws = ork_mm_pack_i8(c, Ks, Nw, Bslc);
        if (!ws) { printf("tile%d pack FAIL Ks=%d Nw=%d\n", i, Ks, Nw); return 1; }
        ork_mm_task_i8 t = { ws, Mw, Aslc, Cpart };
        ork_dyn_chain *h = ork_dyn_begin_mc(c, 1, &t, nc);
        if (!h) { fellback = 1; if (ork_mm_run_i8(c, ws, Mw, Aslc, Cpart)) { printf("tile%d run FAIL\n", i); return 1; } }
        else if (ork_dyn_end(h) < 0) { printf("tile%d dyn_end FAIL\n", i); return 1; }
        for (int m = 0; m < Mw; m++)                                                                /* scatter+accumulate into C[m-tile, n-tile] */
            for (int n = 0; n < Nw; n++) { civac(Cpart + (size_t) m*Nw + n); Cacc[(size_t)(m0+m)*N + (n0+n)] += Cpart[(size_t) m*Nw + n]; }
        ork_mm_free(c, ws);
    }

    long bad = 0, first = -1;
    for (size_t i = 0; i < (size_t) M*N; i++) if (Cacc[i] != Cref[i]) { if (first < 0) first = (long) i; bad++; }
    printf("%d tiles via doorbell%s | mismatches = %ld / %d", nt, fellback ? " (some fell back to blocking!)" : "", bad, M*N);
    if (bad) printf("  first@%ld: got %d want %d", first, Cacc[first], Cref[first]);
    printf(" -> %s\n", bad ? "FAIL" : "BIT-EXACT PASS");
    return bad ? 1 : 0;
}
