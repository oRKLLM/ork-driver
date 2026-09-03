/* fp16 doorbell colsplit (multi-core) regression + A/B.
 *  - No args  : SELF-VALIDATING TEST. Forces ORK_F16_COLSPLIT=1, runs a shape suite at nc=3 (colsplit) and
 *               asserts each is BIT-EXACT to the nc=1 reference (the golden single-core path). Exit 0/nonzero.
 *  - With args: PROBE. `M K N [iters]` reports us/run + checksum + vs-nc1 for the current ORK_F16_COLSPLIT.
 * Board only (needs the NPU). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ork_npu.h"
typedef ork_f16 f16;
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static unsigned long long fnv(const void*p,size_t n){ const unsigned char*b=p; unsigned long long h=1469598103934665603ULL;
    for(size_t i=0;i<n;i++){ h^=b[i]; h*=1099511628211ULL; } return h; }

/* run one shape: fill deterministic, pack, run nc=1 (ref) + nc=3 (test); return diffwords, set *us/*maxerr. */
static double g_us1;   /* last one(): nc=1 best us/run */
static long g_camp_total, g_camp_wrong;   /* ORK_CAMPAIGN: per-iter bit-exact vs nc=1 ref (catches transient wrong answers the end-only check misses) */
static int one(ork_npu*c,int M,int K,int N,int iters,double*us,double*maxerr){
    f16*A=malloc((size_t)M*K*2),*B=malloc((size_t)K*N*2); float*C=malloc((size_t)M*N*4),*R=malloc((size_t)M*N*4);
    unsigned s=12345;
    for(size_t i=0;i<(size_t)M*K;i++){ s=s*1103515245u+12345u; A[i]=(f16)(((int)(s>>16&0xff)-128)/256.0f); }
    for(size_t i=0;i<(size_t)K*N;i++){ s=s*1103515245u+12345u; B[i]=(f16)(((int)(s>>16&0xff)-128)/256.0f); }
    ork_w*w=ork_f16_mm_pack(c,K,N,B); if(!w){ free(A);free(B);free(C);free(R); *us=-1; return -1; }
    /* CHECK EVERY RUN'S RETURN. These were all discarded, and it hid a real defect for as long as this
     * test existed: a doorbell op that fails returns -1 and leaves C untouched, and the driver says so
     * ("the op did NOT run; failing instead of returning stale output"). Because only bit-exactness was
     * checked, a failed op that a LATER successful call overwrote was invisible -- the test printed PASS
     * after stalling 60s on the sentinel timeout, which is what made the suite look hung. A discarded
     * status is the same bug class this driver keeps being bitten by; do not reintroduce it. */
    #define RUN_OK(expr) do { if ((expr) != 0) { fprintf(stderr, \
        "[test_f16colsplit] %s FAILED at M=%d K=%d N=%d (the op did not run — see the driver error above)\\n", \
        #expr, M, K, N); free(A);free(B);free(C);free(R); *us=-1; return -1; } } while (0)
    ork_npu_set_core_budget(c,1); RUN_OK(ork_f16_mm_run(c,w,M,A,R));         /* nc=1 reference */
    double b1=1e18; int nc1iters = getenv("ORK_CAMPAIGN") ? 1 : iters;   /* CAMPAIGN: skip the 1000 nc=1 TIMING runs (waste + where campaign3 silently wedged); 1 ref run is enough for R */
    for(int it=0;it<nc1iters;it++){ double t0=now_us(); RUN_OK(ork_f16_mm_run(c,w,M,A,R)); double d=now_us()-t0; if(d<b1)b1=d; } g_us1=b1;
    ork_npu_set_core_budget(c,3); RUN_OK(ork_f16_mm_run(c,w,M,A,C));         /* nc=3 warm */
    if(getenv("ORK_F16_REAP_TEST")){   /* task #47: close+reopen the DRM fd (drm_release reap) mid-flight, re-import the
        * dma-buf weight in place, then re-run nc=3 — the final C-vs-R diff below validates the POST-REAP output is
        * still bit-exact (proves reopened-fd + re-import + run-path re-warm works). Needs ORK_F16_IMPORT_W. */
        int rr=ork_ctx_fd_reap(c);
        fprintf(stderr,"[REAP-TEST] ork_ctx_fd_reap rc=%d — re-running nc=3 post-reap\n",rr);
        for(size_t i=0;i<(size_t)M*N;i++) C[i]=0;                        /* clear so a non-run shows as diff, not stale-correct */
        ork_f16_mm_run(c,w,M,A,C);
    }
    double best=1e18; for(int it=0;it<iters;it++){ double t0=now_us(); RUN_OK(ork_f16_mm_run(c,w,M,A,C)); double d=now_us()-t0; if(d<best)best=d;
        if(getenv("ORK_CAMPAIGN")){ g_camp_total++; int bad=0; for(size_t i=0;i<(size_t)M*N;i++) if(C[i]!=R[i]){ bad=1; break; } if(bad) g_camp_wrong++;
            if((it%50)==0){ fprintf(stderr,"[CAMPAIGN] iter %d/%d total=%ld wrong=%ld\n",it,iters,g_camp_total,g_camp_wrong); fflush(stderr); } } }
    double me=0; int nbit=0; for(size_t i=0;i<(size_t)M*N;i++){ double e=fabs((double)C[i]-(double)R[i]); if(e>me)me=e; if(C[i]!=R[i])nbit++; }
    *us=best; *maxerr=me; ork_w_free(w); free(A); free(B); free(C); free(R); return nbit;
}
int main(int argc,char**argv){
    if(argc>1){   /* PROBE mode */
        int M=atoi(argv[1]),K=argc>2?atoi(argv[2]):4096,N=argc>3?atoi(argv[3]):512,iters=argc>4?atoi(argv[4]):20;
        ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;} double us,me; int nb=one(c,M,K,N,iters,&us,&me);
        const char*cs=getenv("ORK_F16_COLSPLIT"); printf("F16CS M=%d K=%d N=%d colsplit=%d : nc3=%8.1f us  nc1=%8.1f us  nc3/nc1=%.2fx  vs-nc1: %s maxerr=%.3e diff=%d\n",
            M,K,N,cs?atoi(cs):0,us,g_us1,us/g_us1,nb==0?"BIT-EXACT":"differ",me,nb);
        if(getenv("ORK_CAMPAIGN")) printf("CAMPAIGN: %ld runs, %ld WRONG (%.3f%%) — %s\n", g_camp_total, g_camp_wrong, g_camp_total?100.0*g_camp_wrong/g_camp_total:0.0, g_camp_wrong?"FAIL (heal produced wrong answers)":"PASS (all healed bit-exact or no drop)");
        ork_npu_free(c); return g_camp_wrong?2:0;
    }
    setenv("ORK_F16_COLSPLIT","1",1);   /* SELF-VALIDATING TEST: force colsplit, assert bit-exact vs nc=1 */
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed (NPU?)\n");return 1;}
    int shapes[][3]={{8,2048,256},{128,1024,256},{256,2048,256},{256,4096,512},{256,3584,512},{256,3584,3584},
        {128,2048,16384},{128,3584,18944}};   /* last two: WIDE-N fp16 (N>nmax=8192 => Sn=2, Sn=3) — exercise the per-N-slice CONTIG colsplit (task #45 fp16 wide-N); bit-exact vs nc=1 (mcworker) reference */
    int ns=(int)(sizeof(shapes)/sizeof(shapes[0])), fail=0;
    printf("== test_f16colsplit: fp16 nc=3 doorbell colsplit == nc=1 (bit-exact) ==\n");
    for(int i=0;i<ns;i++){ int M=shapes[i][0],K=shapes[i][1],N=shapes[i][2]; double us,me; int nb=one(c,M,K,N,8,&us,&me);
        printf("  M=%-4d K=%-5d N=%-5d  %s  maxerr=%.2e  %8.1f us/run%s\n", M,K,N, nb==0?"BIT-EXACT":"DIFFER", me, us, nb?"  <-- FAIL":"");
        if(nb) fail=1; }
    ork_npu_free(c);
    printf("%s\n", fail?"TEST_F16COLSPLIT: FAIL":"TEST_F16COLSPLIT: PASS");
    return fail;
}
