/* examples/test_bimport_dom.c — #54 BIMPORT-ONLY multi-domain probe.
 *
 * On the current board boot, native bcreate (MEM_CREATE GEM alloc) into a NON-0 iommu domain EINVALs, so the
 * whole resident-multi-domain path wedges downstream (see test_i4_domains: dom 0 all bit-exact, dom 1 PACK
 * FAILED at the first bcreate). Hypothesis: dma-heap bimport into a non-0 domain uses a different MEM_CREATE
 * flavor that still works. With ORK_BIMPORT_DOM=1 the driver routes non-0 anchor + run scratch through bimport
 * (KERNEL_MAPPING carried for task buffers). This probe exercises ONLY the resident-expert IMPORT path (bimport
 * weight + bimport scratch) in dom 0 (baseline, must pass) then non-0 domains, verified bit-exact vs CPU ref.
 * It deliberately avoids the wedge-prone pack(bcreate-weight)/alternate/arena paths.
 *
 * PASS = dom 0 AND every non-0 domain, both M>1 (BCHAIN) and M=1, bit-exact. Run bounded + line-buffered:
 *   make test_bimport_dom && sudo env ORK_BIMPORT_DOM=1 stdbuf -oL timeout -s TERM --kill-after=10 120 ./test_bimport_dom
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include "ork_npu.h"

/* fsync'd step marker so NOTHING is lost if the board wedges mid-step (the whole point of this probe). */
#define STEP(...) do{ printf("    . "); printf(__VA_ARGS__); printf("\n"); fflush(stdout); fsync(fileno(stdout)); }while(0)

static void fill_i4(int8_t *p, size_t n, unsigned seed){
    unsigned sd = seed;
    for (size_t i = 0; i < n; i++){ sd = sd*1103515245u + 12345u; p[i] = (int8_t)((int)((sd>>17)%15) - 7); }
}
static long verify(const int8_t *A, const int8_t *B, const int32_t *C, int M, int K, int N){
    long maxe = 0;
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++){
        long s = 0; for (int k = 0; k < K; k++) s += (long)A[(size_t)m*K+k]*B[(size_t)k*N+n];
        long e = C[(size_t)m*N+n] - s; if (e < 0) e = -e; if (e > maxe) maxe = e;
    }
    return maxe;
}
/* IMPORT path (resident-expert): pack in dom0 -> dump -> import into `dom` -> run -> verify. */
static int run_import(ork_npu *c, int dom, int M, int K, int N){
    int8_t *A = malloc((size_t)M*K), *B = malloc((size_t)K*N); int32_t *C = malloc((size_t)M*N*4);
    fill_i4(A, (size_t)M*K, 41+dom); fill_i4(B, (size_t)K*N, 53+dom);
    ork_npu_set_pack_domain(c, 0);
    STEP("dom=%d M=%d: pack_i4 in dom0 ...", dom, M);
    ork_w *w0 = ork_mm_pack_i4(c, K, N, B);
    if (!w0){ printf("  [import] dom=%d M=%-3d: PACK(dump) FAILED\n", dom, M); free(A);free(B);free(C); return 1; }
    size_t tb = ork_w_dump(w0, NULL, 0);
    char *blob = malloc(tb); ork_w_dump(w0, blob, tb); ork_mm_free(c,w0);
    ork_npu_set_pack_domain(c, dom);
    STEP("dom=%d M=%d: load_i4_import (bimport weight+scratch) ...", dom, M);
    ork_w *w = ork_mm_load_i4_import(c, K, N, blob, tb);
    if (!w){ printf("  [import] dom=%d M=%-3d: IMPORT FAILED (n=%zu)\n", dom, M, tb); free(blob);free(A);free(B);free(C); return 1; }
    STEP("dom=%d M=%d: import OK -> run_i4 (NPU submit in dom) ...", dom, M);
    int rc = ork_mm_run_i4(c, w, M, A, C);
    STEP("dom=%d M=%d: run_i4 returned rc=%d", dom, M, rc);
    if (rc){ printf("  [import] dom=%d M=%-3d: RUN rc=%d (stuck/refused in domain)\n", dom, M, rc); ork_mm_free(c,w); free(blob);free(A);free(B);free(C); return 1; }
    long maxe = verify(A, B, C, M, K, N);
    printf("  [import] dom=%d M=%-3d K=%d N=%d: maxerr=%-4ld %s\n", dom, M, K, N, maxe, maxe==0?"OK":"FAIL");
    ork_mm_free(c,w); free(blob);free(A);free(B);free(C); return maxe != 0;
}

int main(void){
    ork_npu *c = ork_npu_init(); if (!c){ printf("init failed (NPU?)\n"); return 1; }
    const int K = 2048, N = 512;      /* 35B expert gate/up shape; small CPU ref */
    int fail = 0;
    ork_npu_set_ndomains(c, 4);   /* #54 match the dense/ggml-ork setup (auto-sizer calls this) — pre-size domain arrays before any switch */

    /* #54 CLEAN-NPU NON-0 FIRST: switch dom0->dom1 with a TOTALLY clean NPU (no prior dom-0 op/drop). If THIS
     * lands, the later dom-1 switch-timeout is the dom-0 TCLEAN-drop reap-gap (async cleanup_work not retired
     * before the switch); if it STILL times out, switching into a bimport-established domain is inherently broken. */
    printf("== CLEAN non-0 domain FIRST (no prior dom-0 op) ==\n");
    { int d = ork_npu_domain_alloc(c);
      if (d > 0){ ork_npu_activate_domain(c, d); printf("  -- clean-first: established domain %d --\n", d);
        fail |= run_import(c, d, 128, K, N); fail |= run_import(c, d, 1, K, N); }
      else printf("  domain_alloc returned %d\n", d); }

    printf("== IMPORT path in DOMAIN 0 (baseline — must pass) ==\n");
    fail |= run_import(c, 0, 128, K, N);   /* BCHAIN (M>1 prefill) */
    fail |= run_import(c, 0, 1,   K, N);   /* per-row (decode) */

    printf("== IMPORT path in NON-0 domains (bimport-only: ORK_BIMPORT_DOM) ==\n");
    for (int t = 0; t < 3; t++){
        int d = ork_npu_domain_alloc(c);
        if (d <= 0){ printf("  domain_alloc(#%d) returned %d\n", t, d); fail = 1; break; }
        ork_npu_activate_domain(c, d);   /* establish the domain (bimport anchor under ORK_BIMPORT_DOM) */
        printf("  -- allocated + established domain %d --\n", d);
        fail |= run_import(c, d, 128, K, N);
        fail |= run_import(c, d, 1,   K, N);
    }
    printf("== RESULT: %s ==\n", fail ? "FAIL (bimport-only multi-domain not working)" : "PASS (bimport-only multi-domain works!)");
    ork_npu_free(c);
    return fail;
}
