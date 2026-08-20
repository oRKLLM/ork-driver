/* FIX 3 leak check: pack -> free -> pack a grouped-int4 weight in a loop and confirm IOVA/RAM
 * does NOT grow. Before the fix, ork_mm_free skipped reclaim for arena-view (owns=0) grouped
 * weights, so the resident IOVA grew every iteration until the 4 GiB window was exhausted and
 * bcreate started returning NULL (pack fails). After the fix, ork_mm_free reclaims own_buf, so
 * the same weight can be packed/freed indefinitely with flat RAM/IOVA.
 *
 * A weight of K=2048,N=2048,G=128 is ~2 MiB resident; we cycle it ITERS times (default 4096 =>
 * ~8 GiB cumulative if leaking, far past the 4 GiB IOVA cap). PASS = every pack succeeds and RSS
 * stays flat; FAIL = a pack returns NULL (IOVA leak) or RSS climbs. Run: sudo ./grouped_i4_leak [iters] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"

static long rss_kb(void){
    FILE *f=fopen("/proc/self/status","r"); if(!f) return -1; char l[256]; long v=-1;
    while(fgets(l,sizeof l,f)) if(sscanf(l,"VmRSS: %ld kB",&v)==1) break;
    fclose(f); return v;
}

int main(int argc,char**argv){
    int iters = argc>1 ? atoi(argv[1]) : 4096;
    int K=2048,N=2048,G=128;
    ork_npu *c=ork_npu_init(); if(!c){fprintf(stderr,"init failed\n");return 2;}
    int8_t *B=malloc((size_t)K*N); if(!B){return 2;} for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)((i*7)&0xf)-8;

    long rss0=0;
    for(int it=0; it<iters; it++){
        ork_w *w = ork_i4_mm_pack_grouped(c,K,N,B,G);
        if(!w){ fprintf(stderr,"FAIL: pack_i4_grouped returned NULL at iter %d (IOVA leak / exhaustion)\n", it); free(B); ork_npu_free(c); return 1; }
        size_t wb = ork_w_bytes(w);   /* note: grouped views report 0 (own_buf holds the bytes); fine for the leak check */
        (void)wb;
        ork_mm_free(c,w);
        if(it==16) rss0=rss_kb();   /* baseline after warmup */
        if(it>16 && (it%512)==0){
            long rss=rss_kb();
            printf("iter %5d  RSS %ld kB  (baseline %ld kB, delta %+ld kB)\n", it, rss, rss0, rss-rss0);
        }
    }
    long rss1=rss_kb();
    long delta = rss1-rss0;
    printf("DONE %d iters: RSS baseline %ld kB -> final %ld kB (delta %+ld kB)\n", iters, rss0, rss1, delta);
    free(B); ork_npu_free(c);
    /* allow a small slack for allocator noise; a real leak at this many iters would be hundreds of MiB */
    if(delta > 65536){ printf("FAIL: RSS grew by %ld kB (>64 MiB) — likely leak\n", delta); return 1; }
    printf("PASS: flat RSS across %d pack/free cycles (no IOVA/RAM leak)\n", iters);
    return 0;
}
