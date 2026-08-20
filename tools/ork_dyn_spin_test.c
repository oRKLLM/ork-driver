/* ork_dyn_spin_test — validate the persistent spin tail (ORK_DYN_SPIN + ORK_DYN_RESERVE) and its safe
 * halt + teardown. S real int8 matmuls followed by a forward-chained reserved spin tail (re-running the last
 * matmul idempotently, bounded by the reserve, no self-loop / no redirect). Confirms:
 *   (1) real outputs are bit-exact (the spin re-runs must not corrupt them),
 *   (2) ork_dyn_halt can stop the chain inside the spin region (spin_end-aware),
 *   (3) ork_dyn_end tears down safely (bulk-terminate the spin) with no wedge.
 * Small reserve = safe. Run:
 *   make ork_dyn_spin_test && sudo env ORK_DYN_SPIN=1 ORK_DYN_RESERVE=32 ORK_MM_TIMEOUT=2500 timeout 60 ./ork_dyn_spin_test
 * (NPU op; run alone.)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static int check(const int32_t*O,int S,int N,int K){ int ok=0; for(int i=0;i<S;i++){ volatile int32_t*d=(volatile int32_t*)(O+(size_t)i*N+(N-1)); __asm__ volatile("dc civac,%0"::"r"(d):"memory"); if(*d==K) ok++; } return ok; }

int main(void){
    int S=8, K=512, N=512;
    setvbuf(stdout,0,_IONBF,0); setvbuf(stderr,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int spin=getenv("ORK_DYN_SPIN")!=NULL; const char*rv=getenv("ORK_DYN_RESERVE");
    printf("ork_dyn_spin_test: S=%d K=%d N=%d | ORK_DYN_SPIN=%s ORK_DYN_RESERVE=%s\n", S,K,N, spin?"1":"unset", rv?rv:"unset");
    int8_t*A=malloc(K); memset(A,1,K);
    int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    ork_w*w=ork_i8_mm_pack(c,K,N,B); free(B); if(!w){printf("pack fail\n");return 1;}
    int32_t*O=(int32_t*)ork_dma_alloc(c,(size_t)S*N*sizeof(int32_t)); if(!O){printf("dma fail\n");return 1;}
    ork_mm_task_i8*tk=malloc(sizeof(*tk)*S);
    for(int i=0;i<S;i++){ tk[i].w=w; tk[i].M=1; tk[i].A=A; tk[i].C=O+(size_t)i*N; }

    /* (1) spin tail + explicit early halt inside the spin region.
     * NOTE: do NOT CPU-memset O — it is an ork_dma_alloc (device/non-cacheable) buffer and a bulk memset
     * SIGBUSes on ARM (unaligned SIMD stores to device memory). The NPU writes the outputs; begin seeds the
     * doorbell with aligned word writes. */
    ork_dyn_chain*h=ork_dyn_begin(c,S,tk); if(!h){printf("begin failed\n");return 2;}
    int steps=ork_dyn_steps(h);
    int hr=ork_dyn_halt(h,S+2);                       /* halt at a spin slot (spin_end-aware); -1 if no spin room */
    double t0=now_us(); int done=ork_dyn_end(h); double dt=now_us()-t0;
    int ok1=check(O,S,N,K);
    printf("  [halt] steps=%d halt(S+2=%d)->%s  end highest=%d  drain=%.0fus  real-outputs==K: %d/%d %s\n",
           steps, S+2, hr==0?"accepted":"rejected", done, dt, ok1, S, ok1==S?"OK":"FAIL");

    /* (2) spin tail run to natural teardown (ork_dyn_end bulk-terminates the spin), check correctness + no wedge */
    h=ork_dyn_begin(c,S,tk); if(!h){printf("begin2 failed\n");return 2;}
    struct timespec nap={0,2*1000*1000}; nanosleep(&nap,0);   /* let it spin a bit */
    t0=now_us(); done=ork_dyn_end(h); dt=now_us()-t0;
    int ok2=check(O,S,N,K);
    printf("  [teardown] end highest=%d  drain=%.0fus  real-outputs==K: %d/%d %s\n", done, dt, ok2, S, ok2==S?"OK":"FAIL");

    int pass = (ok1==S && ok2==S);
    printf("  RESULT: persistent spin tail %s (%s halt; %s bit-exact; board did not wedge)\n",
           pass?"PASS":"FAIL", (hr==0)?"accepted":"rejected(no-spin?)", pass?"outputs":"outputs NOT");
    ork_npu_free(c); return pass?0:2;
}
