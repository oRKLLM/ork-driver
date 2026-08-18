/* examples/test_sram_pipe.c — is SRAM a viable CPU->NPU pipe? Two questions:
 *
 *  #1 TRUE CPU->SRAM rate: the cacheable mapping writes at cache speed but the data sits in L1/L2 — a real pipe
 *     must FLUSH (dc cvac) it to SRAM before the NPU reads. Measure memcpy+flush together => effective GB/s that
 *     actually lands in SRAM. Compare to WC (write + dsb, no flush) and noncached (baseline).
 *
 *  #2 SEPARATE PORT: does CPU<->SRAM traffic share the DDR bus? Measure the cacheable-SRAM write+flush rate SOLO,
 *     then again while 3 big cores SATURATE DRAM (streaming a >L2 buffer). Cores are pinned so the ONLY shared
 *     resource is the memory subsystem/interconnect. If SRAM rate is ~unchanged under DRAM saturation => SRAM is a
 *     separate port (the pipe adds aggregate BW / relieves contention). If it collapses => shared bus, no win.
 *
 * PASS = runs + prints (probe). Run: make test_sram_pipe && sudo ./test_sram_pipe
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static void pin(int cpu){ cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu,&s); sched_setaffinity(0,sizeof s,&s); }

static void flush_range(void*p,size_t n){                 /* clean cache -> backing memory (SRAM) */
    for(size_t o=0;o<n;o+=64) __asm__ volatile("dc cvac,%0"::"r"((char*)p+o):"memory");
    __asm__ volatile("dsb ish":::"memory");
}
/* effective write bandwidth INCLUDING the flush to backing memory (what a real pipe pays). */
static double wrflush_bw(void*dst,const void*src,size_t n,int iters,int do_flush){
    memcpy(dst,src,n); if(do_flush) flush_range(dst,n);
    double best=0;
    for(int r=0;r<3;r++){ double t0=now_us();
        for(int i=0;i<iters;i++){ memcpy(dst,src,n); if(do_flush) flush_range(dst,n); }
        double bw=(double)n*iters/1e3/(now_us()-t0); if(bw>best) best=bw; }
    return best;
}

/* ---- DRAM saturation load: big-core threads streaming a >L2 buffer ---- */
static volatile int g_stop=0;
static void* dram_hammer(void*arg){
    int cpu=(int)(intptr_t)arg; pin(cpu);
    size_t bn=64ul*1024*1024; unsigned long long *b=malloc(bn); if(!b) return NULL;
    memset(b,1,bn); size_t w=bn/8; volatile unsigned long long sink=0;
    while(!__atomic_load_n(&g_stop,__ATOMIC_RELAXED)){
        unsigned long long a=0; for(size_t j=0;j<w;j++) a+=b[j]; sink+=a;   /* streaming read pass (DRAM-bound) */
    }
    free(b); (void)sink; return NULL;
}

int main(void){
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    printf("SRAM total=%zu KiB free=%zu KiB\n", ork_npu_sram_total(c)>>10, ork_npu_sram_free(c)>>10);
    enum { SRAM=0x100, NC=0x1, CACHE=0x2, WC=0x4 };
    size_t n=256*1024; int iters=4000;
    unsigned char *src=malloc(n); memset(src,0xa5,n);
    pin(4);                                                /* measuring thread on one big core */

    void *s_nc = ork_dma_alloc_flags(c,n,SRAM|NC);         /* noncached SRAM (baseline) */
    void *s_wc = ork_dma_alloc_flags(c,n,SRAM|NC|WC);      /* write-combine SRAM */
    void *s_ca = ork_dma_alloc_flags(c,n,SRAM|NC|CACHE);   /* cacheable SRAM */
    if(!s_ca){ printf("cacheable SRAM alloc failed\n"); return 1; }

    printf("\n#1 TRUE CPU->SRAM effective rate (%zu KiB, %d iters) — AFFINITY SWEEP:\n", n>>10, iters);
    int cores[]={0,4,5,6,7}; const char*kind[]={"A55","A76","A76","A76","A76"};  /* RK3588: 0-3 little A55, 4-7 big A76 */
    printf("  %-9s %-10s %-10s %-12s %-12s\n","core","noncached","w-combine","cache(abs)","cache+flush");
    for(int ci=0; ci<5; ci++){ pin(cores[ci]);
        double nc=wrflush_bw(s_nc,src,n,iters,0), wc=wrflush_bw(s_wc,src,n,iters,0);
        double caw=wrflush_bw(s_ca,src,n,iters,0), caf=wrflush_bw(s_ca,src,n,iters,1);
        printf("  %d (%-3s)   %7.2f    %7.2f    %8.2f     %8.2f GB/s\n", cores[ci],kind[ci], nc,wc,caw,caf); }

    printf("\n#2 SEPARATE-PORT test — cacheable-SRAM write+flush on A76 core 4, solo vs under DRAM saturation:\n");
    pin(4);                                                /* measure on core 4; hammers on 5,6,7 (distinct) */
    double solo = wrflush_bw(s_ca,src,n,iters,1);
    printf("  solo                      %7.2f GB/s\n", solo);
    g_stop=0; pthread_t th[3];
    for(int i=0;i<3;i++) pthread_create(&th[i],NULL,dram_hammer,(void*)(intptr_t)(5+i)); /* cores 5,6,7 */
    struct timespec ts={0,200*1000*1000}; nanosleep(&ts,NULL);                            /* let them ramp */
    double contended = wrflush_bw(s_ca,src,n,iters,1);
    __atomic_store_n(&g_stop,1,__ATOMIC_RELAXED); for(int i=0;i<3;i++) pthread_join(th[i],NULL);
    printf("  under 3-core DRAM sat     %7.2f GB/s  (%.0f%% of solo)\n", contended, 100.0*contended/solo);
    printf("  => %s\n", contended>=0.85*solo ? "SRAM ~unaffected: SEPARATE PORT (pipe adds aggregate BW)"
                                              : "SRAM degraded: SHARES the bus (no aggregate win)");
    free(src); ork_npu_free(c);
    return 0;
}
