/* test_bmm_fused — regression for ork_bmm_fp16_fused (chained batched fp16 GEMM) AND the small-K 0x1040
 * shape-boundary fix (wiki Exp-2026-07-12-Matmul-Shape-Boundaries).
 *
 * ork_bmm_fp16_fused chains nb dynamic-operand matmuls into ONE PC-chained submit (the on-NPU SSD scan
 * per-stage H-batch). Bare synth() with sched=1 formerly ZEROED the output for fp16 K<96 (K=32/64) — the
 * 0x201:0x1040 K-reduction schedule overshooting the template default. The fix gates sched to pow2
 * K in [128,2048). This test sweeps the K boundary and asserts (a) fused == per-op ork_bmm_fp16 and
 * (b) both match a CPU reference — so a sched/geometry regression at K=32/64 fails make test.
 *
 * Skips (exit 0) with no NPU. Part of make test.
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double rl2(const float *a,const float *b,size_t n){ double num=0,den=0;
    for(size_t i=0;i<n;i++){ double e=a[i]-b[i]; num+=e*e; den+=b[i]*b[i]; } return den>0?sqrt(num/den):sqrt(num); }

static int one(ork_npu *c,int nb,int M,int K,int N){
    size_t na=(size_t)nb*M*K, nbk=(size_t)nb*K*N, nc=(size_t)nb*M*N;
    ork_f16 *A=malloc(na*2),*B=malloc(nbk*2); float *Cf=malloc(nc*4),*Cp=malloc(nc*4),*ref=malloc(nc*4);
    for(size_t i=0;i<na;i++) A[i]=(ork_f16)(((double)rand()/RAND_MAX)*2-1);
    for(size_t i=0;i<nbk;i++) B[i]=(ork_f16)(((double)rand()/RAND_MAX)*2-1);
    for(int b=0;b<nb;b++) for(int m=0;m<M;m++) for(int n=0;n<N;n++){ double a=0;
        for(int k=0;k<K;k++) a+=(double)A[((size_t)b*M+m)*K+k]*(double)B[((size_t)b*K+k)*N+n];
        ref[((size_t)b*M+m)*N+n]=(float)a; }
    int rf=ork_bmm_fp16_fused(c,nb,M,K,N,A,B,Cf);   /* chained (one submit) */
    int rp=ork_bmm_fp16      (c,nb,M,K,N,A,B,Cp);   /* per-op (reference path) */
    int fail=0;
    if(rf||rp){ fprintf(stderr,"  [nb=%d %d,%d,%d] rc fused=%d per-op=%d\n",nb,M,K,N,rf,rp); fail=1; }
    else { double ef=rl2(Cf,ref,nc), ep=rl2(Cp,ref,nc), efp=rl2(Cf,Cp,nc);
        fail = !(ef<=1e-2 && efp<=1e-3);
        fprintf(stderr,"  [nb=%d M=%d K=%d N=%d] fused-vs-cpu=%.2e per-op-vs-cpu=%.2e fused-vs-per-op=%.2e  %s\n",
                nb,M,K,N,ef,ep,efp, fail?"FAIL":"OK"); }
    free(A);free(B);free(Cf);free(Cp);free(ref);
    return fail;
}

int main(void){
    ork_npu *c=ork_npu_init();
    if(!c){ fprintf(stderr,"[test_bmm_fused] no NPU — skipping\n"); return 0; }
    fprintf(stderr,"[test_bmm_fused] SoC=%s — ork_bmm_fp16_fused across the small-K 0x1040 boundary\n",ork_npu_soc(c));
    srand(20260712);
    int fail=0;
    /* the K boundary the sched fix targets: K=32/64 (were ZERO), K=96 (first valid sched=1), K=128/256. */
    fail|=one(c,8,64,32, 64);
    fail|=one(c,8,64,64, 64);
    fail|=one(c,8,64,64, 128);
    fail|=one(c,8,64,96, 64);
    fail|=one(c,8,64,128,64);
    fail|=one(c,8,64,128,128);
    fail|=one(c,8,64,256,64);
    ork_npu_free(c);
    fprintf(stderr, fail? "\nTEST_BMM_FUSED: FAIL\n":"\nTEST_BMM_FUSED: PASS\n");
    return fail?1:0;
}
