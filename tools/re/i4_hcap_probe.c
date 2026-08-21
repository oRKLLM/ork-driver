/* tools/re/i4_hcap_probe.c — is int4's rows-per-batch H leaving throughput on the table?
 *
 * WHY. Unlike fp16/int8/int16, int4 has no unbounded-M miscompute risk: orki_i4_run_bchain_db already
 * TILES M into H-row batches and refuses H<2. So this is a THROUGHPUT question, not a correctness one:
 *
 *     H = 16384/K, capped at 16          (src/npu/i4/chain.c)
 *
 * Neither number was measured. 16384 is the int4 CBUF activation budget in ELEMENTS from the
 * Exp-2026-07-07 fuzz, and the 16 is inherited alongside it. But 16384 int4 elements is only 8192
 * BYTES — a QUARTER of the 32 KB CBUF bank that the 2026-08-20 bank-split work measured for the other
 * dtypes. If int4 actually gets a full bank (65536 int4 elements), H could be several times larger at
 * exactly the K where production FFN weights live:
 *
 *     K=2048: H 8 -> 16 (2x)      K=4096: H 4 -> 16 (4x)      K=8192: H 2 -> 8 (4x)
 *
 * H is rows per WEIGHT STREAM, so a larger H is directly fewer weight re-reads on the int4 prefill
 * path — the same lever the int8 M-fold work exploited.
 *
 * METHOD. Correctness here is a CHECKSUM COMPARISON ACROSS PROCESSES, which sidesteps having to model
 * int4 packing/scales: H only changes how M is tiled, so the result must be IDENTICAL for every valid
 * H. Run once at a known-good H (2, the minimum the path accepts) to get the reference checksum, then
 * once per candidate H and diff. ORK_I4_H's getenv is cached in a static, so it is one H per process —
 * drive the sweep from a shell loop, ascending, and stop at the first mismatch.
 *
 *   make i4_hcap_probe
 *   for H in 2 4 8 16 24 32; do sudo env ORK_I4_H=$H ORK_MM_TIMEOUT=2000 ./i4_hcap_probe 2048 256 64; done
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long fnv64(const void *p,size_t n){
    const unsigned char *b=p; unsigned long long h=1469598103934665603ULL;
    for(size_t i=0;i<n;i++){ h^=b[i]; h*=1099511628211ULL; } return h;
}

int main(int argc,char**argv){
    int K = argc>1?atoi(argv[1]):2048;
    int N = argc>2?atoi(argv[2]):256;
    int M = argc>3?atoi(argv[3]):64;
    if(K%32||N%64){ printf("need K%%32==0 and N%%64==0 (BCHAIN gate)\n"); return 2; }

    ork_npu *c=ork_npu_init(); if(!c){ printf("no board\n"); return 0; }

    int8_t *B=malloc((size_t)K*N), *A=malloc((size_t)M*K);
    int32_t *C=malloc((size_t)M*N*4);
    if(!A||!B||!C){ printf("OOM\n"); return 2; }
    /* int4 weights are [-8,7] carried in int8; deterministic and row-varying so a tiling error shows */
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)((int)((i*7)%15)-7);
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)((int)((i*13)%31)-15);

    ork_w *w=ork_i4_mm_pack(c,K,N,B);
    if(!w){ printf("pack fail (K=%d N=%d)\n",K,N); return 2; }
    memset(C,0,(size_t)M*N*4);
    int rc=ork_i4_mm_run(c,w,M,A,C);

    const char *h=getenv("ORK_I4_H");
    printf("K=%-6d N=%-5d M=%-5d ORK_I4_H=%-4s rc=%-3d cksum=%016llx\n",
           K,N,M,h?h:"(default)",rc, rc?0ULL:fnv64(C,(size_t)M*N*4));

    ork_mm_free(c,w); free(A);free(B);free(C); ork_npu_free(c);
    return rc?1:0;
}
