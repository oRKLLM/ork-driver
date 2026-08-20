/* i4_sn_probe — A1 validation: int4 with Sn>1 (wide-N, N-tiled) on the NONBLOCK doorbell.
 *
 * nmax=8192, so a weight with N>8192 packs Sn>1 N-slices. With ORK_I4_NODB unset (default), M>=2 int4 rides
 * ork_i4_dyn_begin_mc, which A1 extended to emit Sn chained column-slice programs per row. Verifies the
 * doorbell output bit-exact vs the CPU int4 reference AND vs the blocking run_i4_mc (ORK_I4_NODB=1). Board only.
 *   make i4_sn_probe && sudo env ORKD... ./i4_sn_probe
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int run_case(ork_npu *c, int K, int N, int M, int mag){
    int8_t *B=malloc((size_t)K*N), *A=malloc((size_t)M*K);
    int32_t *C=malloc((size_t)M*N*4), *R=malloc((size_t)M*N*4);
    if(!B||!A||!C||!R){ free(B);free(A);free(C);free(R); return 2; }
    unsigned g=0x51ce4;
    /* mag=8 => full [-8,7]; mag=1 => {-1,0,1} (used for Sk>1: keeps a 10752-wide K-slice int16 partial in range
     * so the CPU int reference is valid — the HW int16 accumulator overflows large data at Kp=10752, same as
     * the blocking run_i4_mc, so full-range wide-K would diverge from the int64 CPU ref, not a doorbell bug). */
    #define I4() (mag>=8 ? (int8_t)(((g=g*1103515245u+12345u)>>20&0xf)-8) \
                         : (int8_t)((int)(((g=g*1103515245u+12345u)>>20) % (2*mag+1)) - mag))
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=I4();
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=I4();
    ork_w *w=ork_i4_mm_pack(c,K,N,B);
    if(!w){ printf("  K=%d N=%d M=%d: pack_i4 failed\n",K,N,M); free(B);free(A);free(C);free(R); return 1; }
    /* CPU int4 reference: raw int32 accumulator */
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long a=0; for(int k=0;k<K;k++) a+=(long)A[m*K+k]*B[k*N+n]; R[m*N+n]=(int)a; }
    memset(C,0,(size_t)M*N*4);
    int rc=ork_i4_mm_run(c,w,M,A,C);
    int bad = rc?1:0, nbad=0;
    if(!rc) for(size_t i=0;i<(size_t)M*N;i++) if(C[i]!=R[i]){ if(nbad<3) fprintf(stderr,"    [%zu] %d!=%d\n",i,C[i],R[i]); nbad++; }
    if(nbad) bad=1;
    int Sn = (N + 8191) / 8192;   /* nmax=8192; ork_w is opaque so infer the N-slice count */
    printf("  K=%-5d N=%-6d M=%-2d  Sn~%d  rc=%d  mism=%d/%d  -> %s\n",
           K,N,M, Sn, rc, nbad, M*N, bad?"FAIL":"PASS");
    ork_mm_free(c,w);
    free(B);free(A);free(C);free(R);
    #undef I4
    return bad?1:0;
}

int main(void){
    ork_npu *c=ork_npu_init(); if(!c){ fprintf(stderr,"init failed\n"); return 2; }
    printf("int4 multi-slice on the doorbell (%s):\n", getenv("ORK_I4_NODB")?"ORK_I4_NODB=1 blocking ref":"default doorbell");
    int bad=0;
    /* A1 — N-tile (Sn>1), full-range values (Kp<=1024 => int16 partial in range) */
    bad |= run_case(c, 512,  8192,  4, 8);   /* Sn=1 (control) */
    bad |= run_case(c, 512,  16384, 4, 8);   /* Sn=2 */
    bad |= run_case(c, 512,  18944, 2, 8);   /* Sn=3, partial last slice */
    bad |= run_case(c, 1024, 24576, 3, 8);   /* Sn=3, larger K */
    /* A2 — K-split (Sk>1, K>10752), small values so the 10752-wide int16 partial stays in range vs CPU */
    bad |= run_case(c, 21504, 8192,  2, 1);  /* Sk=2, Sn=1 — pure K-split */
    bad |= run_case(c, 21504, 16384, 2, 1);  /* Sk=2, Sn=2 — K-split x N-tile combined */
    bad |= run_case(c, 32256, 8192,  2, 1);  /* Sk=3 */
    ork_npu_free(c);
    printf("I4_SN_PROBE: %s\n", bad?"FAIL":"PASS");
    return bad?1:0;
}
