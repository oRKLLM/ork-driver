/* test_f16_load — an fp16 weight must survive dump -> load -> run BIT-EXACTLY.
 *
 * fp16 could be packed and dumped but never restored, so fp16 weights could not live in a .orkpack
 * (issue #1). ork_f16_mm_load / ork_f16_mm_load_import add the read-back half. This asserts the only
 * property that actually matters for a persist format: a loaded weight computes the same bytes as the
 * packed one it came from. Exact equality, not a tolerance -- both sides run the same kernel on the
 * same tiles, so any difference is a mis-sliced blob, not fp16 rounding.
 *
 * A wrong un-tiler is easy to write and hard to notice: a swapped Sn/Sk walk order or an off-by-one
 * page stride still produces plausible-looking output. So the shapes below deliberately cover the
 * cases where those bugs differ from correct behaviour:
 *
 *   K=512  N=64     Sk=1 Sn=1   single tile, no slicing at all
 *   K=4096 N=64     Sk=2 Sn=1   K-slice (K > soc->ks), so tile order within an N-slice matters
 *   K=512  N=16384  Sk=1 Sn=2   wide-N (N > soc->nmax), exercising the Sn-major walk
 *   K=1568 N=80     Sk=1 Sn=1   non-power-of-two K: the page pad is not a whole tile
 *
 * Also checks that a blob of the wrong size is REFUSED rather than loaded into garbage.
 * Needs the NPU. Exits 0 on success, non-zero on any mismatch.
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint32_t g_s;
static float rnd(void){ g_s = g_s*1103515245u + 12345u; return (float)((int)((g_s>>16)&0x7ff) - 1024) / 1024.0f; }
static ork_f16 f2h(float f){ return (ork_f16)f; }

static int one_shape(ork_npu *c, int M, int K, int N)
{
    size_t na=(size_t)M*K, nb=(size_t)K*N, nc=(size_t)M*N;
    ork_f16 *A=malloc(na*sizeof *A), *B=malloc(nb*sizeof *B);
    float *C1=malloc(nc*sizeof *C1), *C2=malloc(nc*sizeof *C2), *C3=malloc(nc*sizeof *C3);
    int rc=1;
    if(!A||!B||!C1||!C2||!C3){ fprintf(stderr,"  alloc failed\n"); goto out; }

    g_s=0x51ed2716u ^ (uint32_t)(K*1000003 + N);
    for(size_t i=0;i<nb;i++) B[i]=f2h(rnd());
    for(size_t i=0;i<na;i++) A[i]=f2h(rnd());

    /* reference: pack + run */
    ork_w *w1=ork_f16_mm_pack(c,K,N,B);
    if(!w1){ fprintf(stderr,"  ork_f16_mm_pack failed\n"); goto out; }
    memset(C1,0,nc*sizeof *C1);
    if(ork_f16_mm_run(c,w1,M,A,C1)){ fprintf(stderr,"  run(packed) failed\n"); ork_mm_free(c,w1); goto out; }

    /* dump the resident tiles */
    size_t n=ork_w_dump(w1,NULL,0);
    if(!n){ fprintf(stderr,"  ork_w_dump sized 0\n"); ork_mm_free(c,w1); goto out; }
    void *blob=malloc(n);
    if(!blob || ork_w_dump(w1,blob,n)!=n){ fprintf(stderr,"  ork_w_dump short write\n"); free(blob); ork_mm_free(c,w1); goto out; }
    ork_mm_free(c,w1);

    /* a wrong-size blob must be refused, not mis-sliced */
    if(ork_f16_mm_load(c,K,N,blob,n-4096)){ fprintf(stderr,"  load accepted a SHORT blob\n"); free(blob); goto out; }

    /* load + run must reproduce the packed result exactly */
    ork_w *w2=ork_f16_mm_load(c,K,N,blob,n);
    if(!w2){ fprintf(stderr,"  ork_f16_mm_load failed (n=%zu)\n",n); free(blob); goto out; }
    memset(C2,0,nc*sizeof *C2);
    if(ork_f16_mm_run(c,w2,M,A,C2)){ fprintf(stderr,"  run(loaded) failed\n"); ork_mm_free(c,w2); free(blob); goto out; }
    ork_mm_free(c,w2);
    if(memcmp(C1,C2,nc*sizeof *C1)){
        size_t bad=0, first=(size_t)-1;
        for(size_t i=0;i<nc;i++) if(C1[i]!=C2[i]){ if(first==(size_t)-1) first=i; bad++; }
        fprintf(stderr,"  MISMATCH load: %zu/%zu differ, first idx %zu: packed %g loaded %g\n",
                bad,nc,first,(double)C1[first],(double)C2[first]);
        free(blob); goto out;
    }

    /* the dma-buf import variant, when the heap is available, must match too */
    ork_w *w3=ork_f16_mm_load_import(c,K,N,blob,n);
    if(w3){
        memset(C3,0,nc*sizeof *C3);
        int r=ork_f16_mm_run(c,w3,M,A,C3);
        ork_mm_free(c,w3);
        if(r){ fprintf(stderr,"  run(imported) failed\n"); free(blob); goto out; }
        if(memcmp(C1,C3,nc*sizeof *C1)){ fprintf(stderr,"  MISMATCH load_import\n"); free(blob); goto out; }
        printf("  M=%d K=%-5d N=%-6d blob=%6zu KB : load bit-exact, load_import bit-exact\n",M,K,N,n>>10);
    } else {
        printf("  M=%d K=%-5d N=%-6d blob=%6zu KB : load bit-exact, load_import unavailable (skipped)\n",M,K,N,n>>10);
    }
    free(blob); rc=0;
out:
    free(A); free(B); free(C1); free(C2); free(C3);
    return rc;
}

int main(void)
{
    ork_npu *c=ork_npu_init();
    if(!c){ fprintf(stderr,"ork_npu_init failed\n"); return 1; }
    printf("test_f16_load: fp16 pack -> dump -> load -> run must be bit-exact\n");
    int bad=0;
    bad |= one_shape(c, 8,  512, 64);      /* single tile                          */
    bad |= one_shape(c, 8, 4096, 64);      /* K-slice (Sk=2)                       */
    bad |= one_shape(c, 8,  512, 16384);   /* wide-N (Sn=2), Sn-major walk         */
    bad |= one_shape(c, 8, 1568, 80);      /* non-pow2 K, page pad < a whole tile  */
    ork_npu_free(c);
    printf("TEST_F16_LOAD: %s\n", bad?"FAIL":"PASS");
    return bad?1:0;
}
