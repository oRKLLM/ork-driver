/* dom_race_stress.c — reproduce the RETIREMENT RACE that actually wedges (the combination the isolated
 * submit-only and bcreate-only loops both miss). Each iter:
 *   1. RUN a resident matmul in domain d1  (a compute SUBMIT — leaves d1's task retiring after the sentinel lands)
 *   2. cross-domain PACK in domain d2       (a MEM_CREATE that switches the IOMMU while d1's task is retiring)
 * That step-2 bcreate is the wedge-prone op: it crosses domains right on the heels of a compute submit, so the
 * kernel's switch-idle-wait races the un-retired tail -> "switch iommu domain time out" + gem_object_create.
 * A cross-domain pack/alloc-after-a-run is exactly what real multi-domain / multi-client workloads do.
 *
 * WITHOUT the settle (ORK_DOM_SETTLE_US=0) this should wedge; WITH it, stay clean. The step-2 pack IS the
 * detector: a failed bcreate = the wedge, caught the moment it happens (tight attribution).
 * argv: iters [M K N] (default 100000 8 512 64). Board tool; 0/ok, 1/wedge. */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint32_t g_s;
static int8_t rnd8(void){ g_s = g_s*1103515245u + 12345u; return (int8_t)((int)((g_s>>16)&0xff) - 128); }

int main(int argc,char**argv){
    int iters = argc>1 ? atoi(argv[1]) : 100000;
    int M = argc>2 ? atoi(argv[2]) : 8;
    int K = argc>3 ? atoi(argv[3]) : 512;
    int N = argc>4 ? atoi(argv[4]) : 64;
    if(M<1||K%32||N%32){ fprintf(stderr,"bad dims\n"); return 2; }
    ork_npu *c = ork_npu_init();
    if(!c){ fprintf(stderr,"init failed\n"); return 2; }
    int d1 = ork_npu_domain_alloc(c), d2 = ork_npu_domain_alloc(c);
    if(d1<=0||d2<=0||d1==d2){ fprintf(stderr,"domain_alloc failed d1=%d d2=%d\n",d1,d2); return 2; }

    int8_t *B=malloc((size_t)K*N), *A=malloc((size_t)M*K); int32_t *C=malloc((size_t)M*N*4);
    g_s=0x1234u; for(int i=0;i<K*N;i++) B[i]=rnd8();
    g_s=0x5678u; for(int i=0;i<M*K;i++) A[i]=rnd8();

    ork_npu_set_pack_domain(c, d1);
    ork_w *w1 = ork_mm_pack_i8(c, K, N, B);   /* resident weight in d1 to RUN against */
    if(!w1){ fprintf(stderr,"pack w1 failed — already wedged?\n"); return 1; }

    printf("dom_race_stress: d1=%d d2=%d, %d iters of [run d1 -> xdom pack d2] (M=%d K=%d N=%d)\n", d1,d2,iters,M,K,N);
    for(int i=0;i<iters;i++){
        if(ork_mm_run_i8(c, w1, M, A, C) != 0){ printf("*** WEDGE at RUN(d1) iter %d ***\n", i); fflush(stdout); return 1; }
        ork_npu_set_pack_domain(c, d2);
        ork_w *w2 = ork_mm_pack_i8(c, K, N, B);   /* cross-domain MEM_CREATE right after the run's submit — the race */
        if(!w2){ printf("*** WEDGE at XDOM-PACK(d2) iter %d — retirement race (run d1 -> bcreate d2) ***\n", i); fflush(stdout); return 1; }
        ork_mm_free(c, w2);
        ork_npu_set_pack_domain(c, d1);           /* (run below re-selects w1 in d1 anyway) */
        if(i && i%5000==0){ printf("  %d iters ok\n", i); fflush(stdout); }
    }
    printf("dom_race_stress: ALL OK — %d iters, NO wedge\n", iters);
    ork_npu_domain_free(c,d1); ork_npu_domain_free(c,d2); ork_npu_free(c);
    free(B);free(A);free(C);
    return 0;
}
