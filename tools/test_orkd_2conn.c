/* test_orkd_2conn — MULTI-CONSUMER orkd proof: ONE process opens TWO connections to one daemon.
 *
 * Two ork_npu_init() calls under ORK_USE_ORKD => two independent orkd client connections to the SAME daemon
 * (the first auto-spawns orkd, the second just connects — sequential, no fork, no spawn race). Each connection
 * is a distinct consumer with its own resident weight; the test INTERLEAVES their matmuls (A,B,A,B,...) so the
 * daemon must serialize two live consumers' submits onto the single NPU. Distinct per-consumer data catches any
 * cross-consumer corruption. Both consumers must verify bit-exact vs the CPU reference.
 *
 *   make test_orkd_2conn && sudo env ORK_USE_ORKD=1 ORKD_BIN=$PWD/orkd ./test_orkd_2conn [iters]
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M 8
#define K 512
#define N 64

/* run one matmul on connection `c` with resident weight `w` and activation A[M,K]; verify C == A·B (int32). */
static int run_verify(ork_npu *c, ork_w *w, const int8_t *A, const int8_t *B, char tag, int it){
    int32_t C[M*N], R[M*N];
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long a=0; for(int k=0;k<K;k++) a+=(long)A[m*K+k]*B[k*N+n]; R[m*N+n]=(int)a; }
    memset(C,0,sizeof C);
    if(ork_mm_run_i8(c, w, M, A, C)){ fprintf(stderr,"[%c] it=%d run FAIL\n", tag, it); return 1; }
    for(int i=0;i<M*N;i++) if(C[i]!=R[i]){ fprintf(stderr,"[%c] it=%d MISMATCH [%d] %d!=%d\n", tag, it, i, C[i], R[i]); return 1; }
    return 0;
}

int main(int argc, char **argv){
    int iters = argc>1?atoi(argv[1]):6; if(iters<1)iters=1;
    setenv("ORK_USE_ORKD","1",1);
    fprintf(stderr,"[2conn] opening TWO connections to one orkd (interleaved, %d iters each)\n", iters);
    ork_npu *cA = ork_npu_init(); if(!cA){ fprintf(stderr,"conn A init failed\n"); return 2; }
    ork_npu *cB = ork_npu_init(); if(!cB){ fprintf(stderr,"conn B init failed\n"); ork_npu_free(cA); return 2; }
    /* distinct resident weight per consumer */
    static int8_t BA[K*N], BB[K*N], AA[M*K], AB[M*K];
    unsigned g=0x12345;
    #define R8() ((int8_t)(((g=g*1103515245u+12345u)>>18&0x1f)-16))
    for(int i=0;i<K*N;i++) BA[i]=R8();
    for(int i=0;i<K*N;i++) BB[i]=R8();
    ork_w *wA = ork_mm_pack_i8(cA, K, N, BA);
    ork_w *wB = ork_mm_pack_i8(cB, K, N, BB);
    int bad=0;
    for(int it=0; it<iters && !bad; it++){
        for(int i=0;i<M*K;i++) AA[i]=R8();
        for(int i=0;i<M*K;i++) AB[i]=R8();
        if(run_verify(cA, wA, AA, BA, 'A', it)) bad|=1;   /* consumer A ... */
        if(run_verify(cB, wB, AB, BB, 'B', it)) bad|=2;   /* ... then B — interleaved on the single daemon */
    }
    if(wA) ork_mm_free(cA, wA);
    if(wB) ork_mm_free(cB, wB);
    ork_npu_free(cA);
    ork_npu_free(cB);
    #undef R8
    printf("MULTI_CONSUMER_2CONN: %s — two orkd connections, interleaved matmuls, %d iters%s\n",
           bad?"FAIL":"PASS", iters, bad?"":", both consumers bit-exact");
    return bad?1:0;
}
