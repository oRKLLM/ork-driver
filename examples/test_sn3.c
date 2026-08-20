/* Focused gate for LEVER #1: Sn>1 chain-prefill (ffn_gate/up shape K=3584 N=18944 -> Sn=3).
 * This is the path lever1 restructures (one submit per N-slice). Validates bit-exact vs an
 * OpenMP CPU int32 reference and surfaces any errno-110 / cdma-wild from run_i8. */
#include <stddef.h>
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
static unsigned sd=12345; static int rnd(void){sd=sd*1103515245u+12345u;return (int)((sd>>16)&3);}
/* FNV-1a 64-bit. rnd() is a fixed-seed LCG so inputs — and the exact int32 NPU output C — are
 * deterministic; the pass-path asserts an O(M*N) checksum of C (ALL rows) against a static golden,
 * cheaper AND wider coverage than the O(rows*N*K) OpenMP reference. The reference is kept and runs
 * ONLY to regenerate a golden (ORK_REGEN=1) or diagnose a mismatch (ORK_FULL_REF=1). */
static uint64_t fnv64(const void*p,size_t n){ const uint8_t*b=(const uint8_t*)p; uint64_t h=1469598103934665603ULL;
    for(size_t i=0;i<n;i++){ h^=b[i]; h*=1099511628211ULL; } return h; }
static int one(ork_npu*ctx,int M,int K,int N,uint64_t gold){
    printf("Sn3-gate: M=%d K=%d N=%d (Sn=%d)\n",M,K,N,(N+8191)/8192); fflush(stdout);
    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t*C=malloc((size_t)M*N*4);
    if(!A||!B||!C){printf("  OOM\n");return 1;}
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(int8_t)(rnd()-1);
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=(int8_t)(rnd()-1);
    ork_w*w=ork_i8_mm_pack(ctx,K,N,B);
    if(!w){printf("  pack_i8 FAILED (Sn=%d alloc?)\n",(N+8191)/8192);free(A);free(B);free(C);return 1;}
    errno=0;
    int rc=ork_i8_mm_run(ctx,w,M,A,C);
    if(rc){printf("  run_i8 FAILED rc=%d errno=%d (%s)\n",rc,errno,strerror(errno));ork_w_free(w);free(A);free(B);free(C);return 1;}
    uint64_t got=fnv64(C,(size_t)M*N*4);
    int regen=getenv("ORK_REGEN")!=NULL, ret;
    if(gold && got==gold && !regen && !getenv("ORK_FULL_REF")){
        printf("  ok (golden 0x%016llx; all %d rows)\n",(unsigned long long)got,M); fflush(stdout); ret=0;
    } else {   /* preserved OpenMP int32 reference: regen a golden or diagnose a mismatch */
        int rows = M<8?M:8; long bad=0;   /* verify a subset of rows fully across ALL N (per-N-slice boundaries) */
        #pragma omp parallel for reduction(+:bad)
        for(int i=0;i<rows;i++){
            for(int n=0;n<N;n++){int32_t ref=0;for(int k=0;k<K;k++)ref+=(int)A[(size_t)i*K+k]*(int)B[(size_t)k*N+n];
                if(C[(size_t)i*N+n]!=ref){ if(bad<3)printf("    mism @ (%d,%d): got %d ref %d\n",i,n,C[(size_t)i*N+n],ref); bad++; }}
        }
        {int i=M-1; for(int n=N-64;n<N;n++){int32_t ref=0;for(int k=0;k<K;k++)ref+=(int)A[(size_t)i*K+k]*(int)B[(size_t)k*N+n];   /* tail N-slice / Sn-3 edge */
            if(C[(size_t)i*N+n]!=ref){ if(bad<6)printf("    tail mism @ (%d,%d): got %d ref %d\n",i,n,C[(size_t)i*N+n],ref); bad++; }}}
        if(regen||!gold) printf("  REGEN sn3 GOLD {%d,%d,%d} = 0x%016llxULL  (ref-bad=%ld)\n",M,K,N,(unsigned long long)got,bad);
        else if(got!=gold) printf("  GOLDEN MISMATCH {%d,%d,%d} (ref-bad=%ld) — regen if intended\n",M,K,N,bad);
        printf("  %s (checked %d full rows + tail; mism=%ld)\n",bad?"WRONG":"ok",rows,bad); fflush(stdout);
        ret = (bad || (gold && got!=gold && !regen)) ? 1 : 0;
    }
    ork_w_free(w);free(A);free(B);free(C);
    return ret;
}
int main(void){
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed\n");return 1;}
    int fail=0;
    /* static goldens (fnv64 of NPU int32 C); regen with `sudo env ORK_REGEN=1 ./test_sn3`. 0 => regen. */
    fail|=one(ctx,256,3584,18944, 0x1604cd8848b2f7f9ULL);   /* ffn gate/up prefill, Sn=3 */
    fail|=one(ctx,512,3584,18944, 0x87f44d189ad7b062ULL);   /* larger M, more M-tiles chained per N-slice */
    ork_npu_free(ctx);
    printf("%s\n",fail?"SN3 GATE: FAIL":"SN3 GATE: PASS");
    return fail;
}
