/* FIX 1 multi-domain correctness: pack int8 weights into DISTINCT IOMMU domains via
 * ork_npu_set_pack_domain, run each (interleaved, several M), and verify every result is
 * BIT-EXACT vs the int32 CPU reference. Exercises the per-weight domain threading: each weight's
 * resident tiles live in its own domain, the submit stamps iommu_domain_id from w->domain (no
 * global), and dom_activate swaps per-domain scratch. A domain mismatch would wedge or corrupt;
 * bit-exact int32 == correct placement + submit.
 *
 * Run: sudo ./domain_correct [ndom]   (default 2) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"

static unsigned s=12345; static int rnd(void){ s=s*1103515245u+12345u; return (int)((s>>16)&7)-3; }

int main(int argc,char**argv){
    int ndom = argc>1?atoi(argv[1]):2; if(ndom<1)ndom=1; if(ndom>4)ndom=4;
    int K=2048,N=512;
    ork_npu *c=ork_npu_init(); if(!c){fprintf(stderr,"init failed\n");return 2;}

    int8_t *B[4]; ork_w *w[4]; int dom_of[4];
    for(int d=0; d<ndom; d++){
        B[d]=malloc((size_t)K*N);
        for(size_t i=0;i<(size_t)K*N;i++) B[d][i]=(int8_t)(rnd() + d);  /* distinct weights per domain */
        ork_npu_set_pack_domain(c, d);          /* place this weight's tiles in domain d */
        w[d]=ork_i8_mm_pack(c,K,N,B[d]);
        if(!w[d]){ fprintf(stderr,"pack_i8 dom %d failed\n", d); return 2; }
        dom_of[d]=ork_w_domain(w[d]);
        printf("packed weight %d in domain %d (ork_w_domain reports %d)\n", d, d, dom_of[d]);
        if(dom_of[d]!=d){ fprintf(stderr,"FAIL: weight %d stamped domain %d, expected %d\n", d, dom_of[d], d); return 1; }
    }
    ork_npu_set_pack_domain(c, -1);             /* back to default for scratch/activations */

    int Ms[]={1,4,1,8,4,1};                     /* interleave decode/prefill across domains */
    int bad=0, total=0;
    int8_t *A=malloc((size_t)8*K); int32_t *C=malloc((size_t)8*N*4);
    for(int t=0;t<6;t++){
        int M=Ms[t];
        int d=t%ndom;                           /* round-robin domains -> forces dom_activate swaps */
        for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)rnd();
        if(ork_i8_mm_run(c,w[d],M,A,C)){ fprintf(stderr,"run_i8 dom %d failed\n", d); return 1; }
        for(int m=0;m<M;m++)for(int n=0;n<N;n++){
            int32_t ref=0; for(int k=0;k<K;k++) ref += (int)A[(size_t)m*K+k]*(int)B[d][(size_t)k*N+n];
            total++; if(C[(size_t)m*N+n]!=ref) bad++;
        }
        printf("  M=%-2d domain=%d : %s\n", M, d, bad? "MISMATCH":"ok");
        if(bad) break;
    }
    for(int d=0;d<ndom;d++){ ork_mm_free(c,w[d]); free(B[d]); }
    free(A); free(C); ork_npu_free(c);
    if(bad){ printf("FAIL: %d/%d elements mismatched (multi-domain corruption)\n", bad, total); return 1; }
    printf("PASS: %d-domain int8 run bit-exact vs CPU ref (%d elements checked)\n", ndom, total);
    return 0;
}
