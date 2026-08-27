/* dom_scale_probe.c — how many alternating two-domain cycles until the NPU stalls?
 *
 * The multi-domain bench fails while orkd_dom_api passes, and every STRUCTURAL explanation has been
 * refuted with measurements: domain count (both use two), domain 0's involvement (ORK_DOM_BASE=1
 * still fails), and a switch/submit race (the kernel's job-vs-device domain agrees on 115 of 121
 * timeouts). What is left is SCALE — the bench runs thousands of switch+submit cycles, the probe
 * runs three. This turns that into a number.
 *
 * Shape matches the bench, not orkd_dom_api: the weights are packed ONCE per domain and stay
 * resident, then each iteration is exactly one domain switch plus one matmul, verified bit-exact
 * against an exact integer CPU reference. Reports the FIRST failing iteration, so one run answers
 * "does it break, and if so at what count" instead of a sweep.
 *
 *   sudo tools/util/npu_guard.sh -- ./dom_scale_probe [iters] [M] [K] [N] [weights-per-domain] [imported]
 *
 * The last argument scales RESIDENT BYTES, which is the axis cycle-count alone does not cover: the
 * failing bench holds 957 MiB in one domain and 80 MiB in the other, while two weights hold ~1 MiB.
 *
 * exit 0 = ran all iterations clean (scale is NOT the trigger at this count).
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static uint32_t g_s;
static int8_t rnd8(void){ g_s = g_s*1103515245u + 12345u; return (int8_t)((int)((g_s>>16)&0xff) - 128); }
static void ref_i8(int M,int K,int N,const int8_t*A,const int8_t*B,int32_t*C){
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int32_t s=0; for(int k=0;k<K;k++) s+=(int32_t)A[m*K+k]*(int32_t)B[k*N+n]; C[m*N+n]=s; }
}
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3 + t.tv_nsec/1e6; }

int main(int argc, char **argv){
    int iters = argc>1 ? atoi(argv[1]) : 2000;
    int M     = argc>2 ? atoi(argv[2]) : 8;
    int K     = argc>3 ? atoi(argv[3]) : 512;
    int N     = argc>4 ? atoi(argv[4]) : 64;
    int NW    = argc>5 ? atoi(argv[5]) : 1;    /* resident weights PER DOMAIN */
    int IMP   = argc>6 ? atoi(argv[6]) : 0;    /* 1 = source weights via the dma-buf IMPORT path */

    ork_npu *c = ork_npu_init();
    if(!c){ fprintf(stderr,"ork_npu_init failed\n"); return 1; }

    int d[2] = { ork_npu_domain_alloc(c), ork_npu_domain_alloc(c) };
    if(d[0]<=0 || d[1]<=0 || d[0]==d[1]){ fprintf(stderr,"domain_alloc failed (%d,%d)\n",d[0],d[1]); ork_npu_free(c); return 1; }
    printf("dom_scale_probe: domains %d,%d  iters=%d  M=%d K=%d N=%d  nw=%d  imported=%d\n", d[0], d[1], iters, M, K, N, NW, IMP);

    /* one RESIDENT weight per domain, like the bench (pack once, run many) */
    int nres = 2*NW;
    int8_t *A = malloc((size_t)M*K), **B = calloc(nres,sizeof *B);
    int32_t *C = malloc((size_t)M*N*4), **R = calloc(nres,sizeof *R);
    ork_w **w = calloc(nres,sizeof *w);
    double mib = (double)nres*K*N/1048576.0;
    void **blob = calloc(nres, sizeof *blob);   /* kept alive for the run: the import may reference it */
    printf("  packing %d resident weights (%d per domain, %.0f MiB total)%s\n", nres, NW, mib,
           IMP ? "  [IMPORTED via dma-buf]" : "  [native pack]");
    for(int i=0;i<nres;i++){
        B[i] = malloc((size_t)K*N); R[i] = malloc((size_t)M*N*4);
        g_s = 0x1234u + (uint32_t)i*0x1111u;
        for(int j=0;j<K*N;j++) B[i][j] = rnd8();
        ork_npu_set_pack_domain(c, d[i & 1]);
        w[i] = ork_i8_mm_pack(c, K, N, B[i]);
        if(!w[i]){ fprintf(stderr,"pack failed at weight %d (domain %d) — %.0f MiB in\n", i, d[i&1], (double)i*K*N/1048576.0); ork_npu_free(c); return 1; }
        if(IMP){
            /* Round-trip through the .orkpack form and re-load via the IMPORT path, so the weight is
             * a dma-buf the NPU reads in place — the one structural difference left between this
             * probe (which passes) and the multi-domain bench (which fails: "imported=1"). Content is
             * identical, so the same CPU reference still applies. */
            size_t need = ork_w_dump(w[i], NULL, 0);
            if(!need){ fprintf(stderr,"ork_w_dump sized 0 at weight %d\n", i); ork_npu_free(c); return 1; }
            blob[i] = malloc(need);
            if(ork_w_dump(w[i], blob[i], need) != need){ fprintf(stderr,"ork_w_dump short at weight %d\n", i); ork_npu_free(c); return 1; }
            ork_mm_free(c, w[i]);
            ork_npu_set_pack_domain(c, d[i & 1]);
            w[i] = ork_i8_mm_load_import(c, K, N, blob[i], need);
            if(!w[i]){ fprintf(stderr,"ork_i8_mm_load_import failed at weight %d (domain %d)\n", i, d[i&1]); ork_npu_free(c); return 1; }
        }
    }
    g_s = 0xabcdu; for(int j=0;j<M*K;j++) A[j] = rnd8();
    for(int i=0;i<nres;i++) ref_i8(M,K,N,A,B[i],R[i]);   /* A is fixed, so each weight's answer is constant */

    int first_bad = -1, bad_kind = 0; double t0 = now_ms();
    for(int it=0; it<iters; it++){
        int i = it % nres;                               /* walks all weights => alternates domains each step */
        memset(C, 0, (size_t)M*N*4);
        if(ork_i8_mm_run(c, w[i], M, A, C) != 0){ first_bad = it; bad_kind = 1; break; }
        if(memcmp(C, R[i], (size_t)M*N*4) != 0){ first_bad = it; bad_kind = 2; break; }
        if(it && (it % 200) == 0){
            printf("  ... %d cycles clean (%.1f ms/cycle)\n", it, (now_ms()-t0)/it); fflush(stdout);
        }
    }
    double el = now_ms() - t0;

    int rc;
    if(first_bad < 0){
        printf("dom_scale_probe: PASS — %d switch+submit cycles across 2 domains, all bit-exact (%.0f ms, %.2f ms/cycle)\n",
               iters, el, el/(iters?iters:1));
        rc = 0;
    } else {
        printf("dom_scale_probe: FAIL at cycle %d of %d (%s) after %.0f ms\n", first_bad, iters,
               bad_kind==1 ? "run returned an error" : "output MISMATCH", el);
        rc = 1;
    }
    for(int i=0;i<nres;i++){ ork_mm_free(c, w[i]); free(B[i]); free(R[i]); }
    for(int i=0;i<nres;i++) free(blob[i]);
    free(blob); free(B); free(R); free(w);
    ork_npu_domain_free(c, d[0]); ork_npu_domain_free(c, d[1]);
    free(A); free(C); ork_npu_free(c);
    return rc;
}
