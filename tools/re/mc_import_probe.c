/* mc_import_probe.c — isolate the 7B multi-core imported chain-walk timeout from the single-core case.
 * chain_import_probe proved the SINGLE-core chain (run_chain_i8) at the down-proj shape works with the
 * size-bounded chunked import. But the 7B prefill uses the MULTI-core wide-K path (mcworker_pref_ksplit,
 * K>4096 → Bb K-slice PC-chain per core), and dmesg shows THAT times out (60s, mask 0x2/0x4, domain 1).
 * This probe drives ork_mm_run_i8 at M>1 (=> multi-core) on a CHUNKED IMPORT in a non-0 domain after a
 * domain switch — the exact 7B case — to answer: is the fault mapping-count (=> smaller chunk fixes it)
 * or multi-core-specific (=> need a single-core fallback / real mc fix)?
 *   ./mip [M=128] [K=18944] [N=3584] [chunkMB via ORK_IMPORT_CHUNK_MB]
 * A rc!=0 or wrong C = the multi-core imported fault. Run under `timeout` — a fault hits the 60s NPU
 * job timeout then soft-resets; the probe should return an error rather than wedge. */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(ork_npu *c, const char *tag, ork_w *w, int M, int K, int N,
                 const int8_t *A, int32_t *C) {
    if (!w) { printf("%-30s WEIGHT NULL\n", tag); return 1; }
    memset(C, 0, (size_t)M*N*4);
    int r = ork_mm_run_i8(c, w, M, A, C);
    int ok = (r==0 && C[0]==K && C[(size_t)(M-1)*N+(N-1)]==K);
    printf("%-30s rc=%d C[0]=%d C[last]=%d (expect %d) -> %s\n",
           tag, r, C[0], C[(size_t)(M-1)*N+(N-1)], K, ok ? "OK" : "*** FAULT ***");
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    int M = argc>1 ? atoi(argv[1]) : 128;
    int K = argc>2 ? atoi(argv[2]) : 18944;
    int N = argc>3 ? atoi(argv[3]) : 3584;
    setvbuf(stdout, NULL, _IOLBF, 0);   /* line-buffered: each check's verdict flushes immediately */
    ork_npu *c = ork_npu_init();
    if (!c) { printf("no board\n"); return 0; }
    printf("M=%d K=%d N=%d (multi-core wide-K path; chunkMB=%s)\n",
           M, K, N, getenv("ORK_IMPORT_CHUNK_MB")?getenv("ORK_IMPORT_CHUNK_MB"):"16(default)");

    int8_t *B = calloc((size_t)K*N,1); for (size_t i=0;i<(size_t)K*N;i++) B[i]=1;
    int8_t *A = calloc((size_t)M*K,1); for (size_t i=0;i<(size_t)M*K;i++) A[i]=1;
    int32_t *C = calloc((size_t)M*N,4);

    /* native pack in dom0, dump to a pre-tiled blob (what a .orkpack holds) */
    ork_npu_set_pack_domain(c, 0);
    ork_w *w0 = ork_mm_pack_i8(c, K, N, B);
    int fails = 0;
    fails += check(c, "NATIVE dom0 multi-core", w0, M, K, N, A, C);
    size_t need = ork_w_dump(w0, NULL, 0);
    void *blob = malloc(need); ork_w_dump(w0, blob, need);

    /* KEY ISOLATION: NATIVE weight in a NON-0 domain, multi-core. If THIS faults, the bug is
     * "any non-0-domain multi-core" (global IOMMU attach race) => gate on domain!=0. If it's OK
     * and only the import faults, the bug is import-specific => gate on w_is_imported. */
    if(getenv("ORK_PROBE_IMPORT_FIRST")){
        printf("(ORK_PROBE_IMPORT_FIRST: no prime — first dom1 op is the IMPORT)\n");
    } else {
        ork_npu_set_pack_domain(c, 1);
        ork_w *wn1 = ork_mm_pack_i8(c, K, N, B);   /* native bcreate in dom1 == establishes the domain */
        if(getenv("ORK_PROBE_PRIME_ALLOC")){
            printf("(prime: native ALLOC only in dom1 — NO submit)\n");   /* does creating a native buf suffice? */
        } else {
            fails += check(c, "NATIVE dom1 MC (prime+submit)", wn1, M, K, N, A, C);   /* alloc + submit */
        }
    }

    /* import (chunked) into domain 1, run multi-core IN-domain */
    ork_npu_set_pack_domain(c, 1);
    ork_w *wi1 = ork_mm_load_i8_import(c, K, N, blob, need);
    fails += check(c, "IMPORT dom1 in-domain MC", wi1, M, K, N, A, C);

    /* switch away to dom0, then back to the import — the 7B case (job iommu domain id:1 after switch) */
    fails += check(c, "NATIVE dom0 (switch away)", w0, M, K, N, A, C);
    fails += check(c, "IMPORT dom1 AFTER switch MC", wi1, M, K, N, A, C);

    printf(fails ? "=== %d FAULT(s) ===\n" : "=== ALL OK ===\n", fails);
    return fails ? 1 : 0;
}
