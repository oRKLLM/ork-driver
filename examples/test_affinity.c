/* examples/test_affinity.c — regression test for the worker-thread pinning pattern.
 *
 * Verifies the contract the affinity code must keep:
 *   (1) the async submit path is BIT-EXACT vs the synchronous run, and
 *   (2) the async worker is placed on a BIG core — not the caller's pinned core (collision -> no
 *       overlap) and not an A55 little core (slow + the ORK_NO_AFFINITY drift hazard).
 * The big-core-set affinity in ork_async_launch must keep the worker on the high-numbered (big)
 * cluster; ork_npu_last_async_cpu() reports where it actually landed. Placement check is skipped
 * when ORK_NO_AFFINITY is set or the topology isn't big.LITTLE-shaped (ncpu < 4).
 *
 * (The run_multicore POOL pinning is covered separately by test_speed's scaling threshold: if the
 * pool workers collided onto one core or fell to little cores, multi-core scaling would collapse.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "ork_npu.h"

int main(void){
    int M=8, K=2048, N=512;
    ork_npu *c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    long ncpu=sysconf(_SC_NPROCESSORS_ONLN);

    int8_t *B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    int8_t *A=malloc((size_t)M*K); for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)((i%7)-3);
    int32_t *Cs=malloc((size_t)M*N*4), *Ca=malloc((size_t)M*N*4);
    ork_w *w=ork_i8_mm_pack(c,K,N,B); if(!w){printf("pack failed\n");return 1;}

    ork_i8_mm_run(c,w,M,A,Cs);                       /* synchronous reference */
    ork_async *h=ork_i8_mm_run_async(c,w,M,A,Ca);    /* same submit on the async worker */
    if(!h){ printf("FAIL: async launch returned NULL\n"); return 1; }
    int rc=ork_async_wait(h);
    if(rc){ printf("FAIL: async run rc=%d\n", rc); return 1; }

    int fail=0;
    int mism=0; for(size_t i=0;i<(size_t)M*N;i++) if(Cs[i]!=Ca[i]) mism++;
    if(mism){ printf("FAIL: async vs sync mismatch (%d / %d)\n", mism, M*N); fail=1; }
    else printf("async bit-exact vs sync: OK (M=%d K=%d N=%d)\n", M,K,N);

    int acpu=ork_npu_last_async_cpu(c);
    int no_aff = getenv("ORK_NO_AFFINITY")!=NULL;
    printf("async worker CPU=%d  (ncpu=%ld, big cluster = cpu%ld..%ld)\n", acpu, ncpu, ncpu/2, ncpu-1);
    if(no_aff)            printf("affinity check SKIPPED (ORK_NO_AFFINITY set)\n");
    else if(ncpu<4)       printf("affinity check SKIPPED (ncpu=%ld not big.LITTLE-shaped)\n", ncpu);
    else if(acpu<0)       printf("affinity check SKIPPED (no cpu recorded / non-Linux)\n");
    else if(acpu<ncpu/2){ printf("FAIL: async worker on cpu%d = LITTLE core (expected big cpu%ld..%ld) — pinning pattern broken\n",
                                 acpu, ncpu/2, ncpu-1); fail=1; }
    else                  printf("async worker on a BIG core: OK\n");

    ork_w_free(w); free(A); free(B); free(Cs); free(Ca); ork_npu_free(c);
    if(!fail) printf("AFFINITY OK\n");
    return fail;
}
