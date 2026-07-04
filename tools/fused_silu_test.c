/* Validate the resident-weight fused SwiGLU-gate primitive on silicon by EQUIVALENCE to the already-
 * silicon-validated probe: ork_mm_run_i8_silu(pack(B),A) == ork_npu_probe_i8_silu(A,B), same scales.
 * The probe is the bit-exact reference; no absolute-scale calibration needed.
 *   ork_mm_run_i8_silu  = gate matmul with SiLU fused in the SDP output stage  -> VALIDATED bit-exact.
 *   ork_mm_run_i8_ewmul = up matmul with the EW-mul (xsilu(gate)) fused in the output stage -> the
 *     set_i8_ewmul DPU_RDMA graft still WEDGES the NPU (documented wall); gated behind ORK_TEST_EWMUL.
 * Build: gcc -std=c11 -D_GNU_SOURCE -Iinclude -Isrc tools/fused_silu_test.c src/npu.c src/soc.c \
 *        src/soc/rk3588.c src/soc/rk3576.c src/neon_activations.c -o /tmp/fst -lpthread -lm
 * Run (board): sudo /tmp/fst   -> "FUSED SILU VALIDATED" on success. */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void){
    ork_npu *c = ork_npu_init();
    if(!c){ fprintf(stderr,"init failed\n"); return 2; }
    const int M=8, K=512, N=64;
    int8_t *A=malloc(M*K), *B=malloc((size_t)K*N);
    for(int i=0;i<M*K;i++) A[i]=(int8_t)((i*7)%13-6);
    for(int i=0;i<K*N;i++) B[i]=(int8_t)((i*5)%11-5);
    ork_w *w = ork_mm_pack_i8(c, K, N, B);
    if(!w){ fprintf(stderr,"pack failed\n"); return 2; }

    int rc=0;
    /* sanity: the resident weight computes a correct plain matmul */
    { int32_t *cm=malloc((size_t)M*N*4); int r0=ork_mm_run_i8(c,w,M,A,cm); long mism=0;
      for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long acc=0; for(int k=0;k<K;k++) acc+=(long)A[(size_t)m*K+k]*B[(size_t)k*N+n];
          if(acc!=cm[(size_t)m*N+n]) mism++; }
      printf("[plain matmul] rc=%d mism=%ld/%d %s\n", r0, mism, M*N, mism?"FAIL":"OK"); if(r0||mism)rc=1; free(cm); }

    /* THE WIN: fused gate matmul + SiLU output stage == the silicon-validated probe */
    { int8_t *ref=malloc(M*N), *run=malloc(M*N);
      int r1=ork_npu_probe_i8_silu(c,M,K,N,A,B,ref,NULL);
      int r2=ork_mm_run_i8_silu(c,w,M,A,run,0x51aa,0x14,0xffffff9fu,0xffffc000u,0x56391100u,NULL,0);
      if(r1||r2){ fprintf(stderr,"silu probe=%d run=%d\n",r1,r2); rc=1; }
      else { int mism=0,maxe=0; for(int i=0;i<M*N;i++){int e=abs(ref[i]-run[i]); if(e){mism++;if(e>maxe)maxe=e;}}
          printf("[fused SiLU] resident-vs-probe: mism=%d/%d max|e|=%d %s\n",mism,M*N,maxe,mism?"FAIL":"OK");
          if(mism)rc=1; }
      free(ref); free(run); }

    /* fused EW-mul (up matmul + xsilu(gate)) — WEDGES (set_i8_ewmul DPU_RDMA graft wall); opt-in only */
    if(getenv("ORK_TEST_EWMUL")){
        int8_t *G=malloc(M*N), *er=malloc(M*N), *rr=malloc(M*N);
        for(int i=0;i<M*N;i++) G[i]=(int8_t)((i*3)%9-4);
        int r3=ork_npu_probe_i8_ewmul(c,M,K,N,A,B,G,0x4000,14,er,NULL);
        int r4=ork_mm_run_i8_ewmul(c,w,M,A,G,rr,0x4000,14);
        printf("[fused EW-mul] probe=%d run=%d (both -1 = the DPU_RDMA graft wedge)\n",r3,r4);
        free(G); free(er); free(rr);
    }

    ork_mm_free(c,w); ork_npu_free(c);
    printf("%s\n", rc?"VALIDATION FAILED":"FUSED SILU VALIDATED (resident == probe, bit-exact)");
    return rc;
}
