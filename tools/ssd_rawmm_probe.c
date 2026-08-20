/* ssd_rawmm_probe — (b) slice B1: does the raw-synth fp16 fused-chain path read ROW-MAJOR operands?
 * Runs one fp16 matmul C=A·B through ork_f16_ssd_probe_rawmm (the exact synth + chain_progs mechanism the
 * fused SSD scan uses) with real random row-major A[M,K],B[K,N], and compares to a CPU reference.
 * PASS => a real-operand fused scan can stage row-major operands directly (no manual 32x32 tiling).
 * FAIL => the fused path needs ork_f16_mm_pack-style tiling (the harder route). Board only.
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):64, K=argc>2?atoi(argv[2]):128, N=argc>3?atoi(argv[3]):64;
    ork_npu *c=ork_npu_init();
    if(!c){ fprintf(stderr,"no NPU — skip\n"); return 0; }
    ork_f16 *A=malloc((size_t)M*K*sizeof(ork_f16)), *B=malloc((size_t)K*N*sizeof(ork_f16));
    float *C=malloc((size_t)M*N*sizeof(float)), *ref=malloc((size_t)M*N*sizeof(float));
    srand(20260712);
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(ork_f16)(((double)rand()/RAND_MAX)*2-1);
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=(ork_f16)(((double)rand()/RAND_MAX)*2-1);
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ double a=0;
        for(int k=0;k<K;k++) a+=(double)A[(size_t)m*K+k]*(double)B[(size_t)k*N+n];
        ref[(size_t)m*N+n]=(float)a; }
    int fused = argc>4 && argv[4][0]=='f';   /* "f" -> fused (packed-B + row-major-A); else raw row-major */
    int rc = fused ? ork_f16_ssd_probe_fusedmm(c,M,K,N,A,B,C) : ork_f16_ssd_probe_rawmm(c,M,K,N,A,B,C);
    int fail;
    if(rc){ fprintf(stderr,"probe rc=%d\n",rc); fail=1; }
    else { double num=0,den=0; for(size_t i=0;i<(size_t)M*N;i++){ double e=C[i]-ref[i]; num+=e*e; den+=ref[i]*ref[i]; }
        double rl2=den>0?sqrt(num/den):sqrt(num);
        const char *tag = fused?"fusedmm":"rawmm";
        const char *ok  = fused?"OK(packed-B + row-major-A works in fused chain)":"OK(row-major works)";
        const char *bad = fused?"FAIL(packed-B + row-major-A wrong -> A needs tiling)":"FAIL(row-major NOT read -> needs tiling)";
        fprintf(stderr,"[%s] M=%d K=%d N=%d  C[0]=%.4f ref[0]=%.4f  rel-L2=%.3e  %s\n",
                tag,M,K,N,C[0],ref[0],rl2, rl2>3e-2?bad:ok);
        fail=(rl2>3e-2); }
    free(A);free(B);free(C);free(ref); ork_npu_free(c);
    fprintf(stderr, fail?"\nSSD_RAWMM_PROBE: FAIL\n":"\nSSD_RAWMM_PROBE: PASS\n");
    return fail?1:0;
}
