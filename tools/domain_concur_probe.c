/* tools/domain_concur_probe.c — do 3 NPU cores run 3 different IOMMU domains CONCURRENTLY?
 *
 * Separate from domain_probe.c (which tests >4GiB residence CAPACITY). This tests COMPUTE
 * PARALLELISM. rkllm round-robins single-core submits across cores 0/1/2 (no barrier) to keep the
 * NPU fed (91% util at decode). Two questions:
 *   (1) does ork's async round-robin (ork_mm_run_stream_i8) actually overlap 3 cores? (vs 1-core serial)
 *   (2) does putting each core's weight in a DIFFERENT domain (0/1/2) still overlap, or serialize?
 *       (= can the rk_iommu run 3 translation contexts at once → cross-domain parallelism for >4GiB)
 *
 * Method: 3 independent matmuls timed (a) 1-core serial, (b) run_stream all in domain 0 (SAME),
 * (c) run_stream in domains 0/1/2 (DIFF). speedup(b)~3x => overlap works; DIFF/SAME ~1.0 => domains
 * concurrent; >>1 => the IOMMU serializes domain switches (residence-only).
 *   make domain_concur_probe && sudo ./domain_concur_probe [M] [iters]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):256, K=3584, N=3584, iters=argc>2?atoi(argv[2]):30;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int cores=ork_npu_cores(c);
    printf("INIT OK: soc=%s cores=%d (CREATE errors above this line = init scratch; below = pack)\n", ork_npu_soc(c), cores);
    fflush(stdout);
    if(cores<3){printf("need 3 cores, have %d\n",cores);return 1;}
    int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    int8_t*A=malloc((size_t)M*K); memset(A,1,(size_t)M*K);

    int32_t *C[3]; for(int i=0;i<3;i++) C[i]=malloc((size_t)M*N*4);

    /* ---- PART 1: cross-core overlap in ONE domain (the rkllm round-robin question) ---- */
    ork_w *wsame[3];
    for(int i=0;i<3;i++){ ork_npu_set_pack_domain(c,0); wsame[i]=ork_mm_pack_i8(c,K,N,B); if(!wsame[i]){printf("pack same %d failed\n",i);return 1;} }
    ork_mm_task_i8 tsame[3]; for(int i=0;i<3;i++) tsame[i]=(ork_mm_task_i8){wsame[i],M,A,C[i]};
    ork_npu_set_core_budget(c,cores); ork_mm_run_stream_i8(c,3,tsame);                 /* warm */
    ork_npu_set_core_budget(c,1); for(int i=0;i<3;i++) ork_mm_run_i8(c,wsame[i],M,A,C[i]);
    ork_npu_set_core_budget(c,1);
    double t0=now_us(); for(int it=0;it<iters;it++){ for(int i=0;i<3;i++) ork_mm_run_i8(c,wsame[i],M,A,C[i]); } double tser=(now_us()-t0)/iters;
    ork_npu_set_core_budget(c,cores);
    t0=now_us(); for(int it=0;it<iters;it++) ork_mm_run_stream_i8(c,3,tsame); double tsm=(now_us()-t0)/iters;
    printf("\n--- 3 independent %dx%dx%d int8 matmuls (us, mean of %d) ---\n",M,K,N,iters);
    printf("  (a) 1-core serial:                  %.0f\n", tser);
    printf("  (b) run_stream SAME domain (0,0,0): %.0f   speedup = %.2fx  -> cross-core overlap %s\n",
           tsm, tser/tsm, (tser/tsm>1.8)?"WORKS (rkllm round-robin reproduced)":"WEAK");

    /* PART 2 (cross-domain) REMOVED: run_stream_i8 is single-domain (uses tasks[0].w's domain scratch);
     * cross-domain tasks submit against the wrong domain -> hang. Cross-domain concurrency would need a
     * per-task-domain run_stream variant. This probe now only measures same-domain cross-core round-robin. */
    /* (d) run_chain: PC-chain the 3 matmuls into ONE submit (rkllm's actual mechanism: task_number=3). */
    ork_npu_set_core_budget(c,1);   /* chain is single-core */
    ork_mm_run_chain_i8(c,3,tsame); /* warm */
    t0=now_us(); for(int it=0;it<iters;it++) ork_mm_run_chain_i8(c,3,tsame); double tch=(now_us()-t0)/iters;
    printf("  (d) run_chain (3 matmuls -> 1 submit, single-core): %.0f   speedup vs serial = %.2fx\n", tch, tser/tch);
    printf("  -> chaining amortizes dispatch %s\n", (tser/tch>1.3)?"YES (rkllm mechanism reproduced)":"NO");
    ork_npu_free(c); free(A); free(B); return 0;
}
