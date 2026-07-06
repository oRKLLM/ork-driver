/* mc_i4a8_probe.c — validate the DOMAIN-ESTABLISHMENT anchor fix for the INT4/W4A8 import path
 * (ork_mm_load_i4a8_import), not just int8. Same quirk class: bimport into a fresh non-0 IOMMU domain.
 * Method: pack_i4a8 native in dom0 -> run multi-core = reference C_ref (int32); dump the i4a8 blob;
 * load_i4a8_import into dom1 as the FIRST dom1 op (no native prime — exercises the auto-anchor inside
 * the loader) -> run multi-core = C_test. Import is bit-identical to native (same nibbles), so C_test
 * MUST equal C_ref everywhere; a mismatch (or run-to-run variance) = the domain-establishment corruption.
 * Wide-K down-proj shape (K=18944) forces the multi-core mcworker_pref_ksplit path.
 *   ./mi4  [M=128] [K=18944] [N=3584]   (run a few times; every run must print MATCH) */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv){
    setvbuf(stdout, NULL, _IOLBF, 0);
    int M = argc>1?atoi(argv[1]):128, K = argc>2?atoi(argv[2]):18944, N = argc>3?atoi(argv[3]):3584;
    ork_npu *c = ork_npu_init();
    if(!c){ printf("no board\n"); return 0; }
    printf("i4a8 M=%d K=%d N=%d (multi-core wide-K import path)\n", M, K, N);

    float *Bf = malloc((size_t)K*N*sizeof(float)); for(size_t i=0;i<(size_t)K*N;i++) Bf[i]=((i%7)-3)*0.5f; /* varied weights */
    int8_t *A = malloc((size_t)M*K); for(size_t i=0;i<(size_t)M*K;i++) A[i]=1;
    int32_t *Cref = calloc((size_t)M*N,4), *Ctest = calloc((size_t)M*N,4);
    float *bscale = malloc((size_t)N*sizeof(float));

    /* reference: native pack_i4a8 in dom0, multi-core run */
    ork_npu_set_pack_domain(c, 0);
    ork_w *w0 = ork_mm_pack_i4a8(c, K, N, Bf, bscale);
    if(!w0){ printf("pack_i4a8 failed\n"); return 1; }
    if(ork_mm_run_i8(c, w0, M, A, Cref)){ printf("native run failed\n"); return 1; }

    size_t need = ork_w_dump_i4a8(w0, NULL, 0);
    void *blob = malloc(need); ork_w_dump_i4a8(w0, blob, need);

    /* test: import into dom1 as the FIRST dom1 op (auto-anchor should fire), multi-core run */
    ork_npu_set_pack_domain(c, 1);
    ork_w *wi = ork_mm_load_i4a8_import(c, K, N, blob, need);
    if(!wi){ printf("load_i4a8_import failed (NULL)\n"); return 1; }
    /* switch away and back — the 7B sliding-window case */
    ork_mm_run_i8(c, w0, M, A, Cref);                 /* dom0 */
    if(ork_mm_run_i8(c, wi, M, A, Ctest)){ printf("import run rc!=0\n"); return 1; }

    size_t mism=0, first=(size_t)-1;
    for(size_t i=0;i<(size_t)M*N;i++) if(Cref[i]!=Ctest[i]){ if(first==(size_t)-1)first=i; mism++; }
    printf("C[0] ref=%d test=%d | C[last] ref=%d test=%d | mismatches=%zu%s\n",
           Cref[0], Ctest[0], Cref[(size_t)M*N-1], Ctest[(size_t)M*N-1], mism,
           mism?"":" -> MATCH (i4a8 import bit-identical to native)");
    printf(mism ? "*** FAULT: i4a8 import corrupts (first mismatch @ %zu) ***\n" : "=== i4a8 IMPORT OK ===\n", first);
    return mism ? 1 : 0;
}
