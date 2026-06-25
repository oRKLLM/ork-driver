/* tools/fence_probe.c — anatomy of the post-NPU-fence latency gap (measurement, not optimization).
 *
 * Question: between the NPU matmul fence clearing and the CPU NEON activation touching the result,
 * where does the time go? We measure, over 50 single-token (M=1) matmuls of a 4096-dim row:
 *   - full ork_mm_run() time (blocking submit + bsync FROM_DEVICE invalidate + cres->C copy),
 *   - the internal phase split via ork_npu_run_timing(): setup / submit(wait+bsync) / copy(cres->C),
 *   - the DECISIVE test: time the NEON RMSNorm on the JUST-PRODUCED C vs on a guaranteed-WARM copy.
 *     If those are equal, the activation reads warm data (run already pulled it into cache via the
 *     cres->C copy) => there is NO cold-miss gap at the NEON consume to optimize.
 *
 *   make fence_probe && sudo ./fence_probe        (ORK_RT=1 is set internally via the accessor)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"
#include "neon_activations.h"

#define K 4096
#define N 4096
#define ITERS 50

static double us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW,&t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    ork_npu *c = ork_npu_init(); if(!c){ fprintf(stderr,"init failed\n"); return 1; }

    ork_f16 *B = malloc((size_t)K*N*sizeof(ork_f16));
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=(ork_f16)(((int)(i%17)-8)*0.03f);
    ork_w *w = ork_mm_pack(c, K, N, B);
    if(!w){ fprintf(stderr,"pack failed\n"); return 1; }

    ork_f16 *A = malloc((size_t)K*sizeof(ork_f16));
    for(int i=0;i<K;i++) A[i]=(ork_f16)(((i%13)-6)*0.05f);
    float *C  = malloc((size_t)N*sizeof(float));
    float *Cw = malloc((size_t)N*sizeof(float));
    float *o  = malloc((size_t)N*sizeof(float));
    float *rw = malloc((size_t)N*sizeof(float));
    for(int i=0;i<N;i++) rw[i]=0.7f;

    for(int it=0; it<10; it++) ork_mm_run(c,w,1,A,C);   /* warm: clocks, weight residency, buffers */

    double run_us=0, act_fresh=0, act_warm=0;
    double s0,sub0,cp0; long n0; ork_npu_run_timing(&s0,&sub0,&cp0,&n0);

    for(int it=0; it<ITERS; it++){
        double t0=us(); ork_mm_run(c,w,1,A,C); double t1=us();      /* matmul: submit+bsync+copy->C */
        double t2=us(); ork_rmsnorm_f32(o,C,rw,N,1e-5f); double t3=us();  /* NEON activation on fresh C */
        memcpy(Cw,C,(size_t)N*sizeof(float));                      /* guaranteed-warm copy */
        volatile float s=0; for(int i=0;i<N;i++) s+=Cw[i]; (void)s; /* ensure Cw resident */
        double t4=us(); ork_rmsnorm_f32(o,Cw,rw,N,1e-5f); double t5=us(); /* NEON activation on warm copy */
        run_us+=t1-t0; act_fresh+=t3-t2; act_warm+=t5-t4;
    }
    double s1,sub1,cp1; long n1; ork_npu_run_timing(&s1,&sub1,&cp1,&n1);
    long dn=n1-n0;

    printf("\n=== post-fence anatomy (M=1, %dx%d row, %d tokens, CLOCK_MONOTONIC_RAW) ===\n", K, N, ITERS);
    printf("ork_mm_run total            : %8.3f us/call\n", run_us/ITERS);
    if(dn>0){
        printf("  internal setup            : %8.3f us/call\n", (s1-s0)/dn);
        printf("  internal submit(wait+bsync): %8.3f us/call   <- T0..T1 (NPU wait + FROM_DEVICE invalidate)\n", (sub1-sub0)/dn);
        printf("  internal copy(cres->C)     : %8.3f us/call   <- the post-fence read of NPU output into C\n", (cp1-cp0)/dn);
        printf("  (multicore phases over %ld run_multicore calls)\n", dn);
    } else {
        printf("  (single-core path — no ORK_RT phase split; run total above is the whole submit+bsync+copy)\n");
    }
    printf("NEON RMSNorm on FRESH C     : %8.3f us/call   <- T2-T1 as you framed it\n", act_fresh/ITERS);
    printf("NEON RMSNorm on WARM copy   : %8.3f us/call   <- same kernel, data already cached\n", act_warm/ITERS);
    printf("cold-miss delta (fresh-warm): %8.3f us/call\n", (act_fresh-act_warm)/ITERS);
    printf("\nReading: if the cold-miss delta ~ 0, the activation already sees WARM data (run's cres->C copy\n");
    printf("pulled it into cache). The real post-fence cost is the internal copy phase, not a NEON cold miss.\n");
    return 0;
}
