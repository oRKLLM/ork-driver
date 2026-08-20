/* dom_switch_stress.c — reproduce the rare (~1/2400) rknpu IOMMU domain-switch-idle-wait race.
 *
 * A weight lives in one IOMMU domain; running against weights in different domains back-to-back forces the
 * kernel to switch the NPU's active domain between submits. The kernel's switch waits for NPU-idle (with a
 * timeout) before reprogramming the page table; our run path drains via the output sentinel, but there is a
 * micro-gap between "output landed" and "task retired" (the kernel's idle signal, unreadable here). A switch
 * inside that gap occasionally times out -> "switch iommu domain time out", stuck until reboot.
 *
 * This does a TIGHT loop of alternating-domain int8 matmuls (one domain switch per iter) and checks each result
 * bit-exact. Stops on the first failure (run rc<0 = the wedge, or a mismatch). Prints switches survived so a
 * ~1/2400 event is caught within a few thousand iters. argv[1]=iters (default 12000). Run with
 * ORK_PRESUBMIT_TRACE set so the trace's last line pins the submit at the wedge. Board tool; 0/ok, 1/wedge.
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint32_t g_s;
static int8_t rnd8(void){ g_s = g_s*1103515245u + 12345u; return (int8_t)((int)((g_s>>16)&0xff) - 128); }
static void ref_i8(int M,int K,int N,const int8_t*A,const int8_t*B,int32_t*C){
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int32_t s=0; for(int k=0;k<K;k++) s+=(int32_t)A[m*K+k]*(int32_t)B[k*N+n]; C[m*N+n]=s; }
}

int main(int argc,char**argv){
    int iters = argc>1 ? atoi(argv[1]) : 12000;
    int M = argc>2 ? atoi(argv[2]) : 8;      /* op size is sweepable: a bigger op has a longer retirement tail, */
    int K = argc>3 ? atoi(argv[3]) : 512;    /* which should widen the switch-idle-wait race window (hunt lever) */
    int N = argc>4 ? atoi(argv[4]) : 64;
    if(M<1||K%32||N%32){ fprintf(stderr,"bad dims M=%d K=%d N=%d (K%%32==0,N%%32==0)\n",M,K,N); return 2; }
    ork_npu *c = ork_npu_init();
    if(!c){ fprintf(stderr,"init failed\n"); return 2; }
    if(!ork_npu_uses_orkd(c)){ fprintf(stderr,"%s: REFUSING direct NPU access — the single-stream NPU wedges under concurrent direct use. Route through orkd: sudo env ORK_USE_ORKD=1 ORKD_BIN=$PWD/orkd %s <args>\n", argv[0], argv[0]); ork_npu_free(c); return 3; }
    int d1 = ork_npu_domain_alloc(c), d2 = ork_npu_domain_alloc(c);
    if(d1<=0||d2<=0||d1==d2){ fprintf(stderr,"domain_alloc failed d1=%d d2=%d\n",d1,d2); return 2; }

    int noref = getenv("ORK_NO_REF") && atoi(getenv("ORK_NO_REF"));   /* skip the O(M·N·K) CPU ref (slow for big ops) — wedge/miss still caught by run_i8 rc; used for the size-sweep hunt */
    int8_t *B1=malloc((size_t)K*N), *B2=malloc((size_t)K*N), *A=malloc((size_t)M*K);
    int32_t *C=malloc((size_t)M*N*4), *R1=noref?NULL:malloc((size_t)M*N*4), *R2=noref?NULL:malloc((size_t)M*N*4);
    g_s=0x1111u; for(int i=0;i<K*N;i++) B1[i]=rnd8();
    g_s=0x2222u; for(int i=0;i<K*N;i++) B2[i]=rnd8();
    g_s=0x3333u; for(int i=0;i<M*K;i++) A[i]=rnd8();
    if(!noref){ ref_i8(M,K,N,A,B1,R1); ref_i8(M,K,N,A,B2,R2); }

    ork_npu_set_pack_domain(c,d1); ork_w *w1=ork_i8_mm_pack(c,K,N,B1);
    ork_npu_set_pack_domain(c,d2); ork_w *w2=ork_i8_mm_pack(c,K,N,B2);
    if(!w1||!w2){ fprintf(stderr,"pack failed (w1=%p w2=%p) — already wedged?\n",(void*)w1,(void*)w2); return 1; }

    printf("dom_switch_stress: d1=%d d2=%d, %d iters (1 domain switch/iter)\n", d1, d2, iters);
    int switches=0;
    for(int i=0;i<iters;i++){
        ork_w *w = (i&1) ? w2 : w1;              /* alternate -> a domain switch every iter */
        int32_t *ref = (i&1) ? R2 : R1;
        int rc = ork_i8_mm_run(c, w, M, A, C);
        if(rc!=0){ printf("*** WEDGE at iter %d (after %d clean switches): run_i8 rc=%d (dom %d) ***\n", i, switches, rc, (i&1)?d2:d1); fflush(stdout); return 1; }
        if(!noref && memcmp(C,ref,(size_t)M*N*4)!=0){ printf("*** MISMATCH at iter %d (dom %d) after %d clean switches ***\n", i, (i&1)?d2:d1, switches); fflush(stdout); return 1; }
        switches++;
        if(i && i%1000==0){ printf("  %d iters ok (%d switches)\n", i, switches); fflush(stdout); }
    }
    printf("dom_switch_stress: ALL OK — %d iters, %d clean domain switches, NO wedge\n", iters, switches);
    ork_npu_domain_free(c,d1); ork_npu_domain_free(c,d2); ork_npu_free(c);
    return 0;
}
