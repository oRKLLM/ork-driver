/* Path B: the NORMAL ork_npu API (ork_npu_init / ork_mm_pack_i8 / ork_mm_run_i8) must give identical results
 * whether it runs the NPU directly or transparently routes through orkd (ORK_USE_ORKD=1). The example code is
 * unchanged between the two modes — that's the whole point of Path B. int8-only for now (the routed ops added
 * incrementally). Board tool, not in `make test` (orkd would contend with the direct-NPU examples).
 *   make orkd test_orkd_transparent
 *   sudo ./test_orkd_transparent                                   # direct
 *   sudo env ORK_USE_ORKD=1 ORKD_BIN=$PWD/orkd ./test_orkd_transparent   # routed through orkd
 */
#include "ork_npu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
static uint32_t g = 7;
static int8_t r8(void){ g = g*1103515245u + 12345u; return (int8_t)(((g>>16)&0x7f) - 40); }
static int one(ork_npu *c, int M, int K, int N){
    int8_t *A = malloc((size_t)M*K), *B = malloc((size_t)K*N);
    int32_t *C = malloc((size_t)M*N*4), *ref = malloc((size_t)M*N*4);
    g = 7; for (int i=0;i<M*K;i++) A[i]=r8(); for (int i=0;i<K*N;i++) B[i]=r8();
    for (int m=0;m<M;m++) for (int n=0;n<N;n++){ long a=0; for (int k=0;k<K;k++) a+=(long)A[m*K+k]*B[k*N+n]; ref[m*N+n]=(int)a; }
    ork_w *w = ork_mm_pack_i8(c, K, N, B);
    int bad = 0;
    if (!w){ printf("  pack FAIL M=%d K=%d N=%d\n", M, K, N); bad = 1; }
    else { int rc = ork_mm_run_i8(c, w, M, A, C); ork_mm_free(c, w);
        if (rc){ printf("  run FAIL M=%d K=%d N=%d rc=%d\n", M, K, N, rc); bad = 1; }
        else for (int i=0;i<M*N;i++) if (C[i]!=ref[i]){ if (bad<3) printf("  MISMATCH M=%d K=%d N=%d [%d] %d!=%d\n", M,K,N,i,C[i],ref[i]); bad++; } }
    if (!bad) printf("  ok M=%d K=%d N=%d\n", M, K, N);
    free(A); free(B); free(C); free(ref);
    return bad ? 1 : 0;
}
int main(void){
    ork_npu *c = ork_npu_init();
    if (!c){ printf("init FAIL\n"); return 1; }
    printf("mode: %s (cores=%d)\n", getenv("ORK_USE_ORKD") ? "orkd-client (routed)" : "direct", ork_npu_cores(c));
    int bad = 0;
    bad |= one(c, 8, 128, 64);
    bad |= one(c, 1, 64, 32);
    bad |= one(c, 64, 512, 64);
    bad |= one(c, 256, 512, 64);
    ork_npu_free(c);
    printf("%s\n", bad ? "FAILED" : "ALL OK");
    return bad ? 1 : 0;
}
