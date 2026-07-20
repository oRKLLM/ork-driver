/* dom_chain_stress.c — drive the ACTUAL doorbell-chain API (ork_submit_seq with a grouped run, which rides
 * ork_dyn_begin_seq_i8_mc — the NONBLOCK multi-task HW PC-chain the application is built on) across two IOMMU
 * domains, for a large op count. This is NOT a proxy: it submits real doorbell chains, alternating the domain
 * every submit, so a cross-domain switch follows every chain. If the doorbell-chain mechanism is what wedges
 * multi-domain, this reproduces "switch iommu domain time out". settle via ORK_DOM_SETTLE_US (0 to hunt).
 * argv: iters [M K N] (default 1000000 8 512 64). Board tool; 0/ok, 1/wedge-or-fail. */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint32_t g_s;
static int8_t rnd8(void){ g_s = g_s*1103515245u + 12345u; return (int8_t)((int)((g_s>>16)&0xff) - 128); }

int main(int argc,char**argv){
    int iters = argc>1 ? atoi(argv[1]) : 1000000;
    int M = argc>2 ? atoi(argv[2]) : 8;
    int K = argc>3 ? atoi(argv[3]) : 512;
    int N = argc>4 ? atoi(argv[4]) : 64;
    if(M<1||M>64||K%512||K>4096||N%32){ fprintf(stderr,"bad dims (M<=64,K%%512==0,K<=4096,N%%32==0 for the doorbell chain)\n"); return 2; }
    ork_npu *c = ork_npu_init();
    if(!c){ fprintf(stderr,"init failed\n"); return 2; }
    if(!ork_npu_uses_orkd(c)){ fprintf(stderr,"%s: REFUSING direct NPU access — the single-stream NPU wedges under concurrent direct use. Route through orkd: sudo env ORK_USE_ORKD=1 ORKD_BIN=$PWD/orkd %s <args>\n", argv[0], argv[0]); ork_npu_free(c); return 3; }
    int d1 = ork_npu_domain_alloc(c), d2 = ork_npu_domain_alloc(c);
    if(d1<=0||d2<=0||d1==d2){ fprintf(stderr,"domain_alloc failed d1=%d d2=%d\n",d1,d2); return 2; }

    int8_t *B=malloc((size_t)K*N), *A=malloc((size_t)M*K);
    int32_t *C1=malloc((size_t)M*N*4), *C2=malloc((size_t)M*N*4);
    g_s=0x1234u; for(int i=0;i<K*N;i++) B[i]=rnd8();
    g_s=0x5678u; for(int i=0;i<M*K;i++) A[i]=rnd8();
    ork_npu_set_pack_domain(c,d1); ork_w *w1=ork_mm_pack_i8(c,K,N,B);
    ork_npu_set_pack_domain(c,d2); ork_w *w2=ork_mm_pack_i8(c,K,N,B);
    if(!w1||!w2){ fprintf(stderr,"pack failed (w1=%p w2=%p) — already wedged?\n",(void*)w1,(void*)w2); return 1; }

    printf("dom_chain_stress: d1=%d d2=%d, %d grouped doorbell chains [mm->mm] alternating domains (M=%d K=%d N=%d)\n",
           d1,d2,iters,M,K,N);
    for(int i=0;i<iters;i++){
        ork_w *w = (i&1) ? w2 : w1;
        int32_t *C = (i&1) ? C2 : C1;
        ork_seq_op ops[2]; memset(ops,0,sizeof ops);   /* a 2-task grouped chain (group>0 => ork_dyn_begin_seq_i8_mc doorbell) */
        ops[0].kind=ORK_OP_MM_I8; ops[0].w=w; ops[0].M=M; ops[0].A=A; ops[0].C=C; ops[0].group=1;
        ops[1].kind=ORK_OP_MM_I8; ops[1].w=w; ops[1].M=M; ops[1].A=A; ops[1].C=C; ops[1].group=1;
        int rc = ork_submit_seq(c, ops, 2);
        if(rc!=0){ printf("*** WEDGE/FAIL at chain %d (dom %d) rc=%d ***\n", i, (i&1)?d2:d1, rc); fflush(stdout); return 1; }
        if(i && i%10000==0){ printf("  %d chains ok\n", i); fflush(stdout); }
    }
    printf("dom_chain_stress: ALL OK — %d doorbell chains across 2 domains, NO wedge\n", iters);
    ork_npu_domain_free(c,d1); ork_npu_domain_free(c,d2); ork_npu_free(c);
    free(B);free(A);free(C1);free(C2);
    return 0;
}
