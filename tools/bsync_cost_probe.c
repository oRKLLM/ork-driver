/* tools/bsync_cost_probe.c — quantify the DMA cache-maintenance (bsync) cost that the zero-copy path
 * pays per submit (API lead #3: would NON_CACHEABLE / DISABLE_FLUSH memory help by skipping it?).
 *
 * ork's zero-copy A/C buffers are cacheable (0x403); coherency needs a cache flush/invalidate (bsync)
 * around NPU access. The exposed ork_dma_import_sync is that op. This times it across representative
 * output sizes (decode C ~14 KB ... prefill C ~3.5 MB) so we can bound the lever: if bsync << the
 * matmul submit, non-cacheable (which makes CPU fill/readout ~10x slower, per the 0x401 WC measurement)
 * is a net LOSS and the lead is closed; DISABLE_FLUSH would also reintroduce the coherency bug the
 * bsync fix (3fad74a) solved. Decode C=M*N*4 (M=1,N=3584 -> 14 KB); prefill C (M=256,N=3584 -> 3.5 MB).
 *   make bsync_cost_probe && sudo ./bsync_cost_probe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    printf("INIT OK: soc=%s\n", ork_npu_soc(c));
    size_t sizes[]={16384, 65536, 262144, 1048576, 3670016};
    const char*lbl[]={"~decode C (14KB)","64KB","256KB","1MB","~prefill C (3.5MB)"};
    printf("\n%-20s %12s %10s %12s\n","buffer","bsync(us)","GB/s","iters");
    for(unsigned i=0;i<sizeof(sizes)/sizeof(size_t);i++){
        size_t S=sizes[i];
        void*p=ork_dma_import(c,S);
        if(!p){ printf("%-20s  ork_dma_import failed (dma-heap absent?) — skip\n", lbl[i]); continue; }
        memset(p,1,S);
        int iters = S<=262144 ? 500 : 100;
        ork_dma_import_sync(c,p,S);                              /* warm */
        double t0=now_us(); for(int it=0;it<iters;it++) ork_dma_import_sync(c,p,S);
        double us=(now_us()-t0)/iters;
        printf("%-20s %12.2f %10.1f %12d\n", lbl[i], us, (double)S/us/1e3, iters);
        ork_dma_import_free(c,p);
    }
    printf("\nbsync = cache clean/invalidate cost per submit on the zero-copy path. Compare to the matmul\n");
    printf("submit (decode ~hundreds us-ms; prefill matmul ~ms). Non-cacheable skips bsync but ~10x slower\n");
    printf("CPU fill/readout (0x401 WC = ~1GB/s vs 0x403 cacheable ~10GB/s); DISABLE_FLUSH breaks coherency.\n");
    ork_npu_free(c); return 0;
}
