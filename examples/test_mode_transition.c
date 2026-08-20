/* test_mode_transition — regression for the NPU shared-task-descriptor fix (wiki NPU-Quirks
 * "LUT-activation op poisons the shared task descriptor", branch feat/ssm-npu, 2026-07-12).
 *
 * THE BUG: a standalone LUT activation (ork_i16_npu_exp / silu_i16 / gelu / rsqrt) memset the shared
 * ctx task descriptor (c->task) to its own SDP program (regcfg_amount=69, enable_mask=0x18) and did not
 * restore it. A following SINGLE-CORE matmul reuses the init c->task (regcfg=108, 0xd) without rebuilding
 * it, so it submitted a matmul regcmd under the stale SDP descriptor -> task counter 0x0 -> errno=110
 * wedge. (Multi-core matmul rebuilds the descriptor, so it survived — which is why test_ssd_chunk_npu
 * mode 2 wedged only on the single-core CumBA N=16 while mode 1 did not.)
 *
 * THE FIX: the LUT op now saves/restores c->task on entry/exit (as ewmul/add already did).
 *
 * This test forces the exact failing order — LUT activation, then a core-budget=1 (single-core) matmul —
 * and asserts the matmul (a) does not wedge (pre-fix: rc<0 / errno=110) and (b) is numerically correct.
 * The reverse order (matmul then activation) is the always-safe control. Skips (exit 0) with no NPU.
 * Part of `make test` — the examples ARE the tests (AGENTS.md).
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* one small single-core fp16 matmul C=A*B via ork_bmm_fp16, checked vs CPU. Returns rel-L2, rc via *rcout. */
static double mm_check(ork_npu *c, int M, int K, int N, int *rcout){
    ork_f16 *A=malloc((size_t)M*K*sizeof(ork_f16)), *B=malloc((size_t)K*N*sizeof(ork_f16));
    float *C=malloc((size_t)M*N*sizeof(float)), *ref=malloc((size_t)M*N*sizeof(float));
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(ork_f16)(((double)rand()/RAND_MAX)*2-1);
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=(ork_f16)(((double)rand()/RAND_MAX)*2-1);
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ double a=0; for(int k=0;k<K;k++) a+=(double)A[(size_t)m*K+k]*(double)B[(size_t)k*N+n]; ref[(size_t)m*N+n]=(float)a; }
    int rc=ork_bmm_fp16(c,1,M,K,N,A,B,C); *rcout=rc;
    double num=0,den=0; if(!rc) for(size_t i=0;i<(size_t)M*N;i++){ double e=C[i]-ref[i]; num+=e*e; den+=ref[i]*ref[i]; }
    free(A);free(B);free(C);free(ref);
    return (!rc && den>0)? sqrt(num/den) : (rc?-1.0:0.0);
}

/* run a standalone LUT activation (exp int16) so it touches the shared descriptor */
static int lut_exp(ork_npu *c, int M, int N){
    short *in=malloc((size_t)M*N*2), *out=malloc((size_t)M*N*2); double us=0;
    for(size_t i=0;i<(size_t)M*N;i++) in[i]=(short)(-((int)(i%20000)));   /* negative -> exp in (0,1] */
    int rc=ork_i16_npu_exp(c,in,M,N,30.0/30000.0,1.0/30000.0,out,&us);
    free(in);free(out); return rc;
}

int main(void){
    ork_npu *c=ork_npu_init();
    if(!c){ fprintf(stderr,"[test_mode_transition] no NPU — skipping\n"); return 0; }
    srand(20260712);
    ork_npu_set_core_budget(c,1);   /* FORCE single-core matmul — the poisoned-descriptor path */
    int fail=0, rc;

    /* CONTROL: single-core matmul alone (no prior LUT) must work. */
    double e0=mm_check(c,64,64,16,&rc);
    if(rc){ fprintf(stderr,"[baseline] single-core matmul rc=%d (unexpected)\n",rc); fail=1; }
    else fprintf(stderr,"[baseline] single-core mm N=16: rel-L2=%.2e OK\n",e0);

    /* REGRESSION: LUT exp (int16 SDP) THEN single-core matmul — pre-fix this wedged (errno=110). */
    int re=lut_exp(c,8,64);
    double e1=mm_check(c,64,64,16,&rc);
    if(re){ fprintf(stderr,"[exp->mm] exp rc=%d\n",re); fail=1; }
    else if(rc){ fprintf(stderr,"[exp->mm] single-core matmul-after-LUT rc=%d (DESCRIPTOR BUG REGRESSED)\n",rc); fail=1; }
    else if(e1>3e-2){ fprintf(stderr,"[exp->mm] rel-L2=%.2e too high\n",e1); fail=1; }
    else fprintf(stderr,"[exp->mm] LUT-exp then single-core mm: rel-L2=%.2e OK (descriptor restored)\n",e1);

    /* Repeat to catch a stale descriptor that only bites the 2nd time. */
    lut_exp(c,8,64); double e2=mm_check(c,64,64,16,&rc);
    if(rc||e2>3e-2){ fprintf(stderr,"[exp->mm x2] rc=%d rel-L2=%.2e FAIL\n",rc,e2); fail=1; }
    else fprintf(stderr,"[exp->mm x2] rel-L2=%.2e OK\n",e2);

    ork_npu_free(c);
    fprintf(stderr, fail? "\nTEST_MODE_TRANSITION: FAIL\n":"\nTEST_MODE_TRANSITION: PASS\n");
    return fail?1:0;
}
