/* attn_headchain_probe — is the attention bmm's remaining cost the per-submit floor?
 *
 * attn_marshal_probe shows that once the marshalling is fixed, the fp16 run itself is what is left,
 * and that it reaches only ~76 GMAC/s at the QK^T shape against ~289 GMAC/s for fp16 dense. One
 * candidate for that gap is the submit floor: a head's matmul is small, so the ~167 us/submit floor
 * is a large fraction of it, and attention issues one run per head.
 *
 * The heads are INDEPENDENT matmuls with distinct weights, which is exactly what the stream
 * primitives are for. Three arms over the same H head-matmuls:
 *   PER-HEAD  : H x ork_f16_mm_run                (what ggml-ork does today; each run is multi-core)
 *   STREAM    : ork_f16_mm_run_stream             (round-robin across cores, one submit per program)
 *   CHAIN     : ork_f16_mm_run_stream_chain       (round-robin AND PC-chained: one submit per core)
 * Outputs are compared against the per-head arm. Board only. Diagnostic; not in `all`/`test`. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec*1e-3; }

static double maxdiff(const float*a,const float*b,size_t n){ double m=0; for(size_t i=0;i<n;i++){ double d=fabs((double)a[i]-b[i]); if(d>m)m=d; } return m; }

int main(int argc,char**argv){
    int H  = argc>1?atoi(argv[1]):16;     /* heads */
    int M  = argc>2?atoi(argv[2]):512;    /* ubatch tokens */
    int K  = argc>3?atoi(argv[3]):128;    /* QK^T: head_dim */
    int N  = argc>4?atoi(argv[4]):2048;   /* QK^T: n_kv */
    int it = argc>5?atoi(argv[5]):6;

    setvbuf(stdout,NULL,_IOLBF,0);
    ork_npu *c=ork_npu_init(); if(!c){ fprintf(stderr,"init failed\n"); return 1; }
    printf("attn_headchain_probe  H=%d  [%d,%d]x[%d,%d]  iters=%d  (ork %s)\n",H,M,K,K,N,it,ork_npu_version());

    ork_f16 *A=malloc((size_t)H*M*K*2), *B=malloc((size_t)H*K*N*2);
    float *C0=malloc((size_t)H*M*N*4), *C1=malloc((size_t)H*M*N*4), *C2=malloc((size_t)H*M*N*4);
    if(!A||!B||!C0||!C1||!C2){ fprintf(stderr,"OOM\n"); return 1; }
    for(size_t i=0;i<(size_t)H*M*K;i++) A[i]=(ork_f16)(((i*2654435761u)%1000)/1000.0f-0.5f);
    for(size_t i=0;i<(size_t)H*K*N;i++) B[i]=(ork_f16)(((i*40503u)%997)/997.0f-0.5f);

    ork_w **w=calloc(H,sizeof *w);
    for(int h=0;h<H;h++){ w[h]=ork_f16_mm_pack(c,K,N,B+(size_t)h*K*N);
        if(!w[h]){ fprintf(stderr,"pack %d failed (K=%d N=%d)\n",h,K,N); return 1; } }
    ork_mm_task_f16 *tk=malloc((size_t)H*sizeof *tk);

    double mac=(double)H*M*K*N, t;
    /* PER-HEAD */
    for(int h=0;h<H;h++) ork_f16_mm_run(c,w[h],M,A+(size_t)h*M*K,C0+(size_t)h*M*N);
    t=now_us(); for(int i=0;i<it;i++) for(int h=0;h<H;h++) ork_f16_mm_run(c,w[h],M,A+(size_t)h*M*K,C0+(size_t)h*M*N);
    double per=(now_us()-t)/it;
    printf("  PER-HEAD  %9.1f us  %6.1f GMAC/s  (%.1f us/head)\n",per,mac/per/1e3,per/H);

    /* STREAM */
    for(int h=0;h<H;h++) tk[h]=(ork_mm_task_f16){w[h],M,A+(size_t)h*M*K,C1+(size_t)h*M*N};
    int rs=ork_f16_mm_run_stream(c,H,tk);
    if(rs) printf("  STREAM    rc=%d (refused)\n",rs);
    else { t=now_us(); for(int i=0;i<it;i++) ork_f16_mm_run_stream(c,H,tk); double st=(now_us()-t)/it;
        printf("  STREAM    %9.1f us  %6.1f GMAC/s  (%.2fx)   max|diff| vs per-head %.3g\n",
               st,mac/st/1e3,per/st,maxdiff(C0,C1,(size_t)H*M*N)); }

    /* CHAIN */
    for(int h=0;h<H;h++) tk[h]=(ork_mm_task_f16){w[h],M,A+(size_t)h*M*K,C2+(size_t)h*M*N};
    int rc=ork_f16_mm_run_stream_chain(c,H,tk);
    if(rc) printf("  CHAIN     rc=%d (refused)\n",rc);
    else { t=now_us(); for(int i=0;i<it;i++) ork_f16_mm_run_stream_chain(c,H,tk); double ct=(now_us()-t)/it;
        printf("  CHAIN     %9.1f us  %6.1f GMAC/s  (%.2fx)   max|diff| vs per-head %.3g\n",
               ct,mac/ct/1e3,per/ct,maxdiff(C0,C2,(size_t)H*M*N)); }

    /* WHICH ARM IS WRONG? Head 0 against a CPU fp32 reference, so a disagreement is attributed and
     * not merely observed. ORK_CHAIN_REF=0 skips it (it is O(M*K*N) on one core). */
    if(!getenv("ORK_CHAIN_REF_OFF")){
        float *R=malloc((size_t)M*N*4);
        for(int m=0;m<M;m++) for(int n=0;n<N;n++){ float acc=0;
            for(int k=0;k<K;k++) acc+=(float)A[(size_t)m*K+k]*(float)B[(size_t)k*N+n];
            R[(size_t)m*N+n]=acc; }
        double band=0; for(size_t i=0;i<(size_t)M*N;i++){ double v=fabs(R[i]); if(v>band) band=v; }
        printf("  CPU ref (head 0, |C|max %.2f):  per-head max|diff| %.3g   chain max|diff| %.3g   stream max|diff| %.3g\n",
               band, maxdiff(C0,R,(size_t)M*N), maxdiff(C2,R,(size_t)M*N), maxdiff(C1,R,(size_t)M*N));
        free(R); }

    for(int h=0;h<H;h++) ork_mm_free(c,w[h]);
    free(w);free(tk);free(A);free(B);free(C0);free(C1);free(C2);
    ork_npu_free(c);
    return 0;
}
