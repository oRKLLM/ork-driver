/* orkd_dom_api.c — validate CLIENT-MANAGED IOMMU DOMAINS (ork_npu_domain_alloc/free + ork_npu_set_pack_domain).
 *
 * A client explicitly requests two domains, packs a distinct int8 weight into EACH (ork_npu_set_pack_domain
 * before each pack), runs a matmul against each, and checks both bit-exact vs a CPU reference. This exercises:
 *   - ork_npu_domain_alloc  -> (Path B) ORKD_DOM_REQ, the daemon's coordinated pool
 *   - ork_npu_set_pack_domain -> the pack carries the client-chosen domain id (orkd_pack.domain)
 *   - ork_mm_pack_i8/run_i8 landing + computing in the requested domain
 *   - ork_npu_domain_free   -> ORKD_DOM_REL back to the pool
 * Works both direct (local domain ids) and routed (sudo env ORK_USE_ORKD=1 ... ./orkd_dom_api). 0/ok, nonzero/fail.
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint32_t g_s;
static int8_t rnd8(void){ g_s = g_s*1103515245u + 12345u; return (int8_t)((int)((g_s>>16)&0xff) - 128); }

/* CPU int8 matmul reference: C[m,n] = sum_k A[m,k]*B[k,n] */
static void ref_i8(int M,int K,int N,const int8_t*A,const int8_t*B,int32_t*C){
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int32_t s=0; for(int k=0;k<K;k++) s+=(int32_t)A[m*K+k]*(int32_t)B[k*N+n]; C[m*N+n]=s; }
}

static int one_domain(ork_npu*c,int dom,int M,int K,int N,uint32_t seed){
    g_s = seed;
    int8_t *B = malloc((size_t)K*N), *A = malloc((size_t)M*K);
    int32_t *C = malloc((size_t)M*N*4), *R = malloc((size_t)M*N*4);
    for(int i=0;i<K*N;i++) B[i]=rnd8();
    for(int i=0;i<M*K;i++) A[i]=rnd8();
    ork_npu_set_pack_domain(c, dom);                 /* land the weight in this client-chosen domain */
    ork_w *w = ork_mm_pack_i8(c, K, N, B);
    int rc = -1;
    if(w){
        if(ork_mm_run_i8(c, w, M, A, C)==0){
            ref_i8(M,K,N,A,B,R);
            rc = memcmp(C,R,(size_t)M*N*4)==0 ? 0 : -2;
        }
        ork_mm_free(c, w);
    }
    printf("  domain=%d  M=%d K=%d N=%d : %s\n", dom, M, K, N, rc==0?"bit-exact":(rc==-2?"MISMATCH":"run/pack FAIL"));
    free(B);free(A);free(C);free(R);
    return rc;
}

int main(void){
    ork_npu *c = ork_npu_init();
    if(!c){ fprintf(stderr,"ork_npu_init failed\n"); return 1; }
    int d1 = ork_npu_domain_alloc(c);
    int d2 = ork_npu_domain_alloc(c);
    printf("ORKD_DOM_API: requested domains d1=%d d2=%d\n", d1, d2);
    if(d1<=0 || d2<=0 || d1==d2){ fprintf(stderr,"domain_alloc failed (d1=%d d2=%d)\n",d1,d2); ork_npu_free(c); return 1; }
    int bad = 0;
    bad |= (one_domain(c, d1, 8, 512, 64, 0x1234u) != 0);
    bad |= (one_domain(c, d2, 8, 512, 64, 0x9abcu) != 0);
    bad |= (one_domain(c, d1, 16, 1024, 128, 0x55aau) != 0);   /* reuse d1 after d2 -> exercises a domain switch */
    ork_npu_domain_free(c, d1);
    ork_npu_domain_free(c, d2);
    ork_npu_free(c);
    printf("ORKD_DOM_API: %s\n", bad?"FAIL":"PASS");
    return bad ? 1 : 0;
}
