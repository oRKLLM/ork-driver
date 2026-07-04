/* Validate the resident-weight fused SwiGLU primitives on silicon by EQUIVALENCE to the already-
 * silicon-validated probes (same A,B,scales -> same output). No absolute-scale calibration needed:
 * the probes are the bit-exact reference.
 *   test 1: ork_mm_run_i8_silu(pack(B), A)   ==  ork_npu_probe_i8_silu(A,B)     [gate + fused SiLU]
 *   test 2: ork_mm_run_i8_ewmul(pack(B), A, G) == ork_npu_probe_i8_ewmul(A,B,G) [up + fused EW-mul]
 * Build: gcc -std=c11 -D_GNU_SOURCE -Iinclude -Isrc tools/fused_silu_test.c src/npu.c src/soc.c -o /tmp/fst
 * Run (board, sudo): sudo /tmp/fst   -> prints mism counts; 0 = the resident-weight fusion matches. */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void){
    ork_npu *c = ork_npu_init();
    if(!c){ fprintf(stderr,"init failed\n"); return 2; }
    const int M=8, K=512, N=64;
    int8_t *A=malloc(M*K), *B=malloc((size_t)K*N), *G=malloc(M*N);
    for(int i=0;i<M*K;i++) A[i]=(int8_t)((i*7)%13-6);
    for(int i=0;i<K*N;i++) B[i]=(int8_t)((i*5)%11-5);
    for(int i=0;i<M*N;i++) G[i]=(int8_t)((i*3)%9-4);

    /* pack B as a resident int8 weight (same tiling synth_i8 expects) */
    ork_w *w = ork_mm_pack_i8(c, K, N, B);
    if(!w){ fprintf(stderr,"pack failed\n"); return 2; }

    int rc=0;
    /* ---- test 0: plain resident matmul (bisect: is the weight/setup good?) vs CPU ref ---- */
    { int32_t *c_mm=malloc((size_t)M*N*4);
      int r0=ork_mm_run_i8(c,w,M,A,c_mm);
      long mism=0,maxe=0;
      for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long acc=0; for(int k=0;k<K;k++) acc+=(long)A[(size_t)m*K+k]*B[(size_t)k*N+n];
          long e=labs(acc-c_mm[(size_t)m*N+n]); if(e){mism++; if(e>maxe)maxe=e;} }
      printf("[plain matmul] resident-vs-CPU: rc=%d mism=%ld/%d max|e|=%ld %s\n", r0, mism, M*N, maxe, mism?"FAIL(weight/setup bad)":"OK(weight good)");
      free(c_mm); }
    /* ---- test 1: fused SiLU (probe's known-good g2 constants) ---- */
    int8_t *c_ref=malloc(M*N), *c_run=malloc(M*N);
    int r1 = ork_npu_probe_i8_silu(c, M, K, N, A, B, c_ref, NULL);
    int r2 = ork_mm_run_i8_silu(c, w, M, A, c_run,
                                0x51aa, 0x14, 0xffffff9fu, 0xffffc000u, 0x56391100u, NULL, 0);
    if(r1||r2){ fprintf(stderr,"silu probe=%d run=%d\n",r1,r2); rc=1; }
    else { int mism=0,maxe=0; for(int i=0;i<M*N;i++){int e=abs(c_ref[i]-c_run[i]); if(e){mism++; if(e>maxe)maxe=e;}}
        printf("[fused SiLU] resident-vs-probe: mism=%d/%d max|e|=%d %s\n", mism, M*N, maxe, mism?"FAIL":"OK");
        if(mism) rc=1; }

    /* ---- test 2: fused EW-mul (gain mult/shift = probe defaults) ---- */
    int8_t *e_ref=malloc(M*N), *e_run=malloc(M*N);
    int r3 = ork_npu_probe_i8_ewmul(c, M, K, N, A, B, G, 0x4000, 14, e_ref, NULL);
    int r4 = ork_mm_run_i8_ewmul(c, w, M, A, G, e_run, 0x4000, 14);
    if(r3||r4){ fprintf(stderr,"ewmul probe=%d run=%d\n",r3,r4); rc=1; }
    else { int mism=0,maxe=0; for(int i=0;i<M*N;i++){int e=abs(e_ref[i]-e_run[i]); if(e){mism++; if(e>maxe)maxe=e;}}
        printf("[fused EW-mul] resident-vs-probe: mism=%d/%d max|e|=%d %s\n", mism, M*N, maxe, mism?"FAIL":"OK");
        if(mism) rc=1; }

    ork_mm_free(c,w); ork_npu_free(c);
    printf("%s\n", rc?"FUSION VALIDATION FAILED":"FUSION PRIMITIVES VALIDATED (resident == probe)");
    return rc;
}
