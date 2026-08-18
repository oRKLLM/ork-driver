/* examples/test_sram_bw.c — CPU<->SRAM vs CPU<->DRAM transfer bandwidth.
 *
 * The question: can the on-chip NPU SRAM serve as a CPU<->NPU transfer bus that BYPASSES the shared ~28 GB/s
 * DRAM bus (freeing DRAM for weight streaming and adding aggregate bandwidth on the NEON<->NPU critical path)?
 * That only pays off if the CPU can WRITE the SRAM buffer at a rate >= its DRAM write rate — SRAM is fast for
 * the NPU, but the CPU reaches it over the interconnect and it may be mapped uncached/write-combining.
 *
 * Fair comparison: ork_dma_alloc (DRAM) vs ork_dma_alloc_sram (SRAM) are the SAME DMA-buffer mapping type —
 * only the backing memory differs. So the delta is purely DRAM-vs-SRAM, not cacheable-vs-uncacheable. A plain
 * malloc() buffer (normal cacheable DRAM) is printed as a reference. PASS = it runs + prints; this is a probe,
 * not a correctness gate. Run: make test_sram_bw && sudo ./test_sram_bw
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

/* CPU write BW: memcpy src->dst, ITERS times. */
static double wr_bw(void *dst, const void *src, size_t n, int iters){
    memcpy(dst, src, n);                                  /* warm */
    double t0=now_us(); for(int i=0;i<iters;i++) memcpy(dst, src, n);
    double us=now_us()-t0; return (double)n*iters/1e3/us;  /* GB/s */
}
/* CPU read BW: sum dst (volatile accumulate so it isn't optimized away), ITERS times. */
static double rd_bw(const void *buf, size_t n, int iters){
    volatile unsigned long long acc=0; const unsigned long long *p=(const unsigned long long*)buf; size_t w=n/8;
    double t0=now_us();
    for(int i=0;i<iters;i++){ unsigned long long a=0; for(size_t j=0;j<w;j++) a+=p[j]; acc+=a; }
    double us=now_us()-t0; (void)acc; return (double)n*iters/1e3/us; /* GB/s */
}

static void bench(const char *tag, void *buf, const void *src, size_t n, int iters){
    if(!buf){ printf("  %-10s alloc FAILED\n", tag); return; }
    double w=wr_bw(buf, src, n, iters), r=rd_bw(buf, n, iters);
    printf("  %-10s  write %7.2f GB/s   read %7.2f GB/s\n", tag, w, r);
}

int main(void){
    ork_npu *c=ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    size_t total=ork_npu_sram_total(c), free0=ork_npu_sram_free(c);
    printf("SRAM: total=%zu KiB  free=%zu KiB\n", total>>10, free0>>10);
    size_t n = 256*1024;                                  /* 256 KiB — fits the ~956 KiB SRAM budget with margin */
    int iters = 4000;                                     /* ~1 GB moved per test => stable timing */
    unsigned char *src = malloc(n); memset(src, 0xa5, n);

    /* RKNPU_MEM_* flags: NON_CONTIGUOUS=0x1, CACHEABLE=0x2, WRITE_COMBINE=0x4, TRY_ALLOC_SRAM=0x100.
     * ork_dma_alloc_sram uses 0x401 (no cacheable/WC bit) => pgprot_noncached => the slow 1.2 GB/s.
     * Here we allocate SRAM (and DRAM) with CACHEABLE and WRITE_COMBINE to see if the mapping — not the
     * memory — was the wall. (CPU-BW only; a real pipe adds a dc cvac flush (cacheable) or dsb (WC) before
     * the NPU reads. Not needed to time raw CPU write/read here.) */
    enum { SRAM=0x100, NC=0x1, CACHE=0x2, WC=0x4 };
    unsigned char *mal = malloc(n);
    printf("CPU transfer bandwidth (%zu KiB buffer, %d iters):\n", n>>10, iters);
    bench("malloc",         mal,                                 src, n, iters);  /* cacheable DRAM reference */
    bench("DRAM noncache",  ork_dma_alloc(c, n),                 src, n, iters);  /* ork_dma_alloc default */
    bench("SRAM noncache",  ork_dma_alloc_sram(c, n),            src, n, iters);  /* ork_dma_alloc_sram default (0x401) */
    bench("SRAM wcombine",  ork_dma_alloc_flags(c,n,SRAM|NC|WC), src, n, iters);  /* <-- the pipe candidate (WC) */
    bench("SRAM cacheable", ork_dma_alloc_flags(c,n,SRAM|NC|CACHE), src, n, iters); /* <-- the pipe candidate (cacheable+flush) */
    bench("DRAM wcombine",  ork_dma_alloc_flags(c,n,NC|WC),      src, n, iters);  /* WC on DRAM, for comparison */
    size_t free1 = ork_npu_sram_free(c);
    printf("SRAM free after allocs=%zu KiB (dropped %ld KiB total)\n", free1>>10, (long)((free0-free1)>>10));

    free(src); free(mal); ork_npu_free(c);
    return 0;
}
