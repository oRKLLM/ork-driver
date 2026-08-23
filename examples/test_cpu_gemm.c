/* test_cpu_gemm — the NEON offline GEMM must be BIT-IDENTICAL to the scalar one.
 *
 * orki_cpu_gemm_i32 is what stands in for the NPU's int4/int8 MAC when there is no device
 * (ork_npu_init_offline), and the entire argument for trusting an offline-built .orkpack is that this
 * substitution is EXACT rather than approximate: int32 accumulation is associative, so vectorising and
 * reordering it cannot change the result. That argument is only worth as much as a test of it — a NEON
 * kernel that is subtly wrong (a bad widen, a mishandled tail, a sign error on the int4 range) would still
 * produce plausible perplexity and quietly corrupt every pack built off-board.
 *
 * So: same inputs through a deliberately naive reference and through the shipped kernel, memcmp. Shapes
 * cover the vector width (N%16 == 0 and not), the k-unroll (K%4 == 0 and not), M=1 (the decode path) and
 * M>1 (the omp path), plus an all-zero activation row (the `if(!a) continue` skip) and the int8 extremes
 * where the int16 product bound (127*127) is tightest. No NPU needed. */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

void orki_cpu_gemm_i32(int M,int K,int N,const int8_t *A,const int8_t *B,int32_t *C);   /* internal */

static uint32_t g = 7777u;
static int rnd(int lo, int hi){ g ^= g<<13; g ^= g>>17; g ^= g<<5; return lo + (int)(g % (uint32_t)(hi-lo+1)); }

static void ref_gemm(int M,int K,int N,const int8_t*A,const int8_t*B,int32_t*C){
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){
        int32_t s=0;
        for(int k=0;k<K;k++) s += (int32_t)A[(size_t)m*K+k] * (int32_t)B[(size_t)k*N+n];
        C[(size_t)m*N+n]=s;
    }
}

static int one(int M,int K,int N,int lo,int hi,int zero_row,const char*tag){
    int8_t *A=malloc((size_t)M*K), *B=malloc((size_t)K*N);
    int32_t *C1=malloc((size_t)M*N*4), *C2=malloc((size_t)M*N*4);
    if(!A||!B||!C1||!C2){ printf("  [%-12s] OOM\n",tag); return 1; }
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)rnd(lo,hi);
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)rnd(lo,hi);
    if(zero_row) memset(A,0,(size_t)K);                       /* exercise the a==0 skip */
    ref_gemm(M,K,N,A,B,C1);
    memset(C2,0xAB,(size_t)M*N*4);
    orki_cpu_gemm_i32(M,K,N,A,B,C2);
    size_t bad=0, first=(size_t)-1;
    for(size_t i=0;i<(size_t)M*N;i++) if(C1[i]!=C2[i]){ if(first==(size_t)-1) first=i; bad++; }
    if(bad) printf("  [%-12s] M=%d K=%d N=%d: %zu/%zu differ (first @%zu: ref=%d got=%d) FAIL\n",
                   tag,M,K,N,bad,(size_t)M*N,first,C1[first],C2[first]);
    else    printf("  [%-12s] M=%-4d K=%-5d N=%-5d BIT-IDENTICAL\n",tag,M,K,N);
    free(A);free(B);free(C1);free(C2);
    return bad?1:0;
}

int main(void){
    printf("test_cpu_gemm: NEON offline GEMM vs scalar reference (no NPU needed)\n");
    int fail=0;
    fail |= one(  1, 1024, 1024, -8,  7, 0, "i4 M=1");        /* decode, int4 range */
    fail |= one( 32, 1024, 1024, -8,  7, 0, "i4 M=32");       /* omp path */
    fail |= one(  4,  512,  256, -128,127,0, "i8 full");      /* int8 extremes: 127*127 int16 bound */
    fail |= one(  3,  130,  144, -8,  7, 0, "K%%4 N%%16");     /* both tails at once */
    fail |= one(  2,  256,   80, -8,  7, 1, "zero-row");      /* the a==0 skip */
    fail |= one(  1,   32,   16, -8,  7, 0, "tiny");
    printf("test_cpu_gemm: %s\n", fail?"FAIL":"ALL BIT-IDENTICAL");
    return fail;
}
