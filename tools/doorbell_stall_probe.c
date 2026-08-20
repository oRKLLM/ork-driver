/* doorbell_stall_probe — isolate WHY the doorbell FFN scheduler runs slower than the async path
 * (110-130 vs 149 t/s), when "architecturally-correct-but-no-headroom" should be ON PAR, not slower.
 * Hypothesis (user): a timing/starvation bug — the doorbell needs a thread continuously spinning to advance
 * its chain, but the scheduler's thread goes off to do silu, so the NPU chain STARVES while the CPU is busy.
 *
 * Same int8 matmul as the FFN up-proj (M=256, K=2048, N=6144), tiled 4x64 for the doorbell. Timed:
 *   A) async         : ork_i8_mm_run_async(M=256) + ork_async_wait          (the apipe path)
 *   B) doorbell NOW  : ork_dyn_begin_mc(4 tasks) + ork_dyn_end immediately   (CPU polls throughout)
 *   C) doorbell DELAY: ork_dyn_begin_mc + busy-spin(D us) + ork_dyn_end      (CPU off "doing silu")
 *
 * Decisive: if C's (total - D) >> B, the NPU did NOT advance while the CPU was busy => the doorbell starves
 * without a dedicated spinner (confirms the timing bug; fix = dedicated doorbell spinner thread). If
 * C-total ~= max(D, B), the NPU ran autonomously and the slowdown is elsewhere.
 *   make doorbell_stall_probe && sudo ./doorbell_stall_probe [M=256] [K=2048] [N=6144] [TS=64]
 * (NPU op; run alone; reboot if a later NPU op misbehaves.)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static inline void civac(volatile void*p){ __asm__ volatile("dc civac,%0"::"r"(p):"memory"); }
static void busy(double us){ double t0=now_us(); volatile double x=0; while(now_us()-t0<us){ x+=1.0; } (void)x; }

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):256, K=argc>2?atoi(argv[2]):2048, N=argc>3?atoi(argv[3]):6144, TS=argc>4?atoi(argv[4]):64;
    int nt=(M+TS-1)/TS;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    printf("doorbell_stall_probe: M=%d K=%d N=%d  doorbell tiles=%dx%d\n", M,K,N,nt,TS);
    int8_t*A=(int8_t*)malloc((size_t)M*K); memset(A,1,(size_t)M*K);   /* host (malloc): doorbell stages A, DMA-A source miscomputes */
    int8_t*B=(int8_t*)malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    ork_w*w=ork_i8_mm_pack(c,K,N,B); if(!w){printf("pack fail\n");return 1;}
    long want=(long)K;   /* A=B=1 => each dot == K */
    double a_ms=0;   /* async path skipped (wedged on fresh-pack); focus on the doorbell autonomy question */

    /* ---- B) doorbell drain-NOW (CPU polls throughout ork_dyn_end) ---- */
    int32_t*Cb=(int32_t*)ork_dma_alloc(c,(size_t)M*N*sizeof(int32_t)); if(!Cb){printf("dma_alloc B fail\n");return 1;}
    ork_mm_task_i8*tk=(ork_mm_task_i8*)malloc(sizeof(*tk)*nt);
    for(int t=0;t<nt;t++){ int m0=t*TS, mc=(M-m0<TS)?(M-m0):TS; tk[t].w=w; tk[t].M=mc; tk[t].A=A+(size_t)m0*K; tk[t].C=Cb+(size_t)m0*N; }
    double b_ms=0;
    for(int rep=0;rep<3;rep++){ memset(Cb,0,(size_t)M*N*sizeof(int32_t));
        double t0=now_us(); ork_dyn_chain*h=ork_dyn_begin_mc(c,nt,tk,0);
        if(!h){printf("  B doorbell: begin_mc NULL (ineligible)\n"); break; }
        ork_dyn_end(h); double ms=(now_us()-t0)/1000; b_ms=ms;
        civac(Cb+(size_t)(M-1)*N+(N-1)); int okc = Cb[(size_t)(M-1)*N+(N-1)]==want;
        printf("  B doorbell now     rep%d: %.2f ms  (%s)\n", rep, ms, okc?"ok":"BAD"); }

    /* ---- C) AUTONOMY: arm the chain, busy-spin with ZERO ork calls, then read progress ONCE. If progress
     *        reaches nt-1 the NPU walked the whole chain autonomously (no CPU driver needed); if it's stuck
     *        low, the chain did NOT advance while the CPU was busy => it needs a driver/spinner thread. ---- */
    for(int di=0; di<3; di++){ double D_ms=(di+1)*5.0;   /* 5, 10, 15 ms — brackets the ~13ms silu */
        memset(Cb,0,(size_t)M*N*sizeof(int32_t));
        double t0=now_us(); ork_dyn_chain*h=ork_dyn_begin_mc(c,nt,tk,0);
        if(!h){printf("  C doorbell: begin_mc NULL\n"); break; }
        busy(D_ms*1000.0);                                  /* CPU off doing "silu" — NOT calling any ork fn */
        int prog=ork_dyn_progress(h);                       /* first read after the busy delay (civacs -> true state) */
        ork_dyn_end(h); double ms=(now_us()-t0)/1000;
        civac(Cb+(size_t)(M-1)*N+(N-1)); int okc = Cb[(size_t)(M-1)*N+(N-1)]==want;
        printf("  C arm+busy(%2.0fms)+progress: progress=%d/%d after busy  total=%6.2f ms -> %s  (%s)\n",
               D_ms, prog, nt-1, ms,
               (prog>=nt-1) ? "AUTONOMOUS (finished during busy; no driver needed)"
                            : "STALLED (did NOT advance while CPU busy => needs a driver/spinner)", okc?"ok":"BAD"); }

    printf("\nSummary: async=%.2fms  doorbell-now=%.2fms.  If C total-D ~= B, the NPU runs autonomously and a\n", a_ms, b_ms);
    printf("dedicated spinner is NOT needed; if C total-D >> B, the doorbell starves without continuous polling.\n");
    return 0;
}
