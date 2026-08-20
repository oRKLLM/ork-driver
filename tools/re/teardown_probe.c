/* teardown_probe.c — two questions about imported-weight teardown:
 *  (a) is the kernel __iommu_dma_unmap WARN (rknpu_gem_free_object -> iommu_dma_unmap_sg) IMPORT-specific?
 *      run mode=import vs mode=native and diff `dmesg | grep -c __iommu_dma_unmap` across each.
 *  (b) does import+free LEAK IOVA (kernel unmap incomplete -> space not reclaimed)? loop alloc→free a
 *      ~68MB weight N times in one domain; if a later alloc returns NULL (PRIME/MEM_CREATE ENOMEM) even
 *      though every prior weight was freed, the kernel isn't reclaiming = leak. native must run all N.
 *  usage: ./tp <import|native> <cycles> <domain>   (default import 40 0)
 * Pure alloc/free — no NPU submits, so no wedge risk. */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char *mode = argc > 1 ? argv[1] : "import";
    int cyc = argc > 2 ? atoi(argv[2]) : 40;
    int dom = argc > 3 ? atoi(argv[3]) : 0;
    int K = 3584, N = 18944;                 /* gate/up shape: K<=4096 so Bf is built (~68MB Bb + Bf) */
    int native = strncmp(mode, "native", 6) == 0;
    int hold = strstr(mode, "hold") != NULL;   /* hold = alloc N, DON'T free, exit -> kernel auto-cleanup */
    ork_npu *c = ork_npu_init(); if (!c) { printf("no board\n"); return 0; }
    int8_t *B = malloc((size_t) K * N); for (size_t i = 0; i < (size_t) K * N; i++) B[i] = (int8_t)(i & 0x7f);
    /* build a pre-tiled blob once (for the import path) */
    ork_npu_set_pack_domain(c, 0);
    ork_w *wt = ork_i8_mm_pack(c, K, N, B);
    size_t need = ork_w_dump(wt, NULL, 0); void *blob = malloc(need); ork_w_dump(wt, blob, need); ork_mm_free(c, wt);
    printf("mode=%s cycles=%d dom=%d K=%d N=%d (~%zuMB blob)\n", mode, cyc, dom, K, N, need >> 20);
    ork_npu_set_pack_domain(c, dom);
    int ok = 0;
    for (int i = 0; i < cyc; i++) {
        ork_w *w = native ? ork_i8_mm_pack(c, K, N, B)
                          : ork_i8_mm_load_import(c, K, N, blob, need);
        if (!w) { printf("*** cycle %d: ALLOC FAILED (mode=%s) after %d successful free-cycles => IOVA NOT reclaimed (LEAK)\n", i, mode, ok); break; }
        if (!hold) ork_mm_free(c, w);   /* reclaim; hold mode LEAVES it for the process-exit kernel cleanup */
        ok++;
        if (i % 10 == 0) printf("  cycle %d ok\n", i);
    }
    printf("=== %s: %d/%d alloc+free cycles succeeded%s ===\n", mode, ok, cyc,
           ok == cyc ? " (no leak — IOVA fully reclaimed)" : " (LEAK — reclaim incomplete)");
    return ok == cyc ? 0 : 1;
}
