/* dom_pack_stress.c — reproduce/validate the CROSS-DOMAIN bcreate wedge (the real culprit, per the dmesg
 * gem_object_create pairing + the 1M submit-only-clean result). Loops pack(alternating domain)+free, so EACH
 * iteration issues cross-domain MEM_CREATEs (the vulnerable op that switches the IOMMU while a prior task may be
 * mid-retirement). WITHOUT the settle (ORK_DOM_SETTLE_US=0) this should hit "switch iommu domain time out"
 * within ~1/50k crossings; WITH a settle it should stay clean. Stops on the first pack failure = the wedge.
 * argv[1]=iters (default 200000). Board tool; 0/ok, 1/wedge. */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint32_t g_s;
static int8_t rnd8(void){ g_s = g_s*1103515245u + 12345u; return (int8_t)((int)((g_s>>16)&0xff) - 128); }

int main(int argc,char**argv){
    int iters = argc>1 ? atoi(argv[1]) : 200000;
    const int K=512, N=64;
    ork_npu *c = ork_npu_init();
    if(!c){ fprintf(stderr,"init failed\n"); return 2; }
    if(!ork_npu_uses_orkd(c)){ fprintf(stderr,"%s: REFUSING direct NPU access — the single-stream NPU wedges under concurrent direct use. Route through orkd: sudo env ORK_USE_ORKD=1 ORKD_BIN=$PWD/orkd %s <args>\n", argv[0], argv[0]); ork_npu_free(c); return 3; }
    int d1 = ork_npu_domain_alloc(c), d2 = ork_npu_domain_alloc(c);
    if(d1<=0||d2<=0||d1==d2){ fprintf(stderr,"domain_alloc failed d1=%d d2=%d\n",d1,d2); return 2; }
    int8_t *B = malloc((size_t)K*N);
    g_s=0x1234u; for(int i=0;i<K*N;i++) B[i]=rnd8();

    printf("dom_pack_stress: d1=%d d2=%d, %d alternating cross-domain packs\n", d1, d2, iters);
    for(int i=0;i<iters;i++){
        int d = (i&1) ? d2 : d1;
        ork_npu_set_pack_domain(c, d);
        ork_w *w = ork_mm_pack_i8(c, K, N, B);   /* cross-domain MEM_CREATE (switches from the previous pack's domain) */
        if(!w){ printf("*** WEDGE at pack %d (dom %d) — bcreate failed ***\n", i, d); fflush(stdout); return 1; }
        ork_mm_free(c, w);
        if(i && i%10000==0){ printf("  %d packs ok\n", i); fflush(stdout); }
    }
    printf("dom_pack_stress: ALL OK — %d cross-domain packs, NO wedge\n", iters);
    ork_npu_domain_free(c,d1); ork_npu_domain_free(c,d2); ork_npu_free(c);
    free(B);
    return 0;
}
