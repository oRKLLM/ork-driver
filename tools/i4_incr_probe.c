/* i4_incr_probe — validate + time the experimental int4 incremental-task batch (ork_mm_run_i4_incr,
 * the vendor's task_number=N pattern: one resident weight, M rows, 12-config incremental tasks in ONE
 * submit) against (1) a CPU int4 reference (correctness) and (2) the two existing paths:
 *   - library batched run_i4(M)  = STRATEGY A (stride-2 in-task, Hcap-capped), and
 *   - per-row run_i4(1) x M       = M separate submits (the decode/MoE baseline).
 * The question: does keeping the weight resident across M cheap incremental tasks beat both?
 *
 *   make i4_incr_probe && sudo ./i4_incr_probe [M] [K] [N] [iters]
 * Board-only; dummy data but CHECKS correctness of the incr path. N<=64 (single N-block).
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec*1e-3; }
static unsigned rs=0x1234567u; static int r4(void){ rs=rs*1103515245u+12345u; return (int)((rs>>17)%15)-7; }

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):16, K=argc>2?atoi(argv[2]):512, N=argc>3?atoi(argv[3]):64, iters=argc>4?atoi(argv[4]):50;
    ork_npu *c=ork_npu_init(); if(!c){ fprintf(stderr,"init failed\n"); return 1; }

    int8_t *B=malloc((size_t)K*N); for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)r4();
    int8_t *A=malloc((size_t)M*K); for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)r4();
    ork_w *w=ork_mm_pack_i4(c,K,N,B); if(!w){ fprintf(stderr,"pack_i4 failed\n"); return 1; }

    int32_t *Ci=calloc((size_t)M*N,4), *Cb=calloc((size_t)M*N,4), *Cs=calloc((size_t)M*N,4);

    /* incremental path */
    int rc=ork_mm_run_i4_incr(c,w,M,A,Ci);
    if(rc){ fprintf(stderr,"run_i4_incr FAILED rc=%d\n",rc); return 1; }

    /* correctness: CPU int4 reference for the incr output */
    int bad=0;
    for(int m=0;m<M && bad<5;m++) for(int n=0;n<N && bad<5;n++){
        long s=0; for(int k=0;k<K;k++) s+=(long)A[(size_t)m*K+k]*B[(size_t)k*N+n];
        if(Ci[(size_t)m*N+n]!=(int32_t)s){ printf("  MISMATCH [%d,%d] got %d exp %ld\n",m,n,Ci[(size_t)m*N+n],s); bad++; }
    }
    printf(bad?"  incr CORRECTNESS: FAIL\n":"  incr CORRECTNESS: OK (vs CPU int4)\n");

    /* timings */
    ork_mm_run_i4_incr(c,w,M,A,Ci);                                    /* warm */
    double t0=now_us(); for(int it=0;it<iters;it++) ork_mm_run_i4_incr(c,w,M,A,Ci); double t_incr=(now_us()-t0)/iters;

    ork_mm_run_i4(c,w,M,A,Cb);                                          /* warm STRATEGY A */
    t0=now_us(); for(int it=0;it<iters;it++) ork_mm_run_i4(c,w,M,A,Cb); double t_batch=(now_us()-t0)/iters;

    for(int m=0;m<M;m++) ork_mm_run_i4(c,w,1,A+(size_t)m*K,Cs+(size_t)m*N);   /* warm per-row */
    t0=now_us(); for(int it=0;it<iters;it++) for(int m=0;m<M;m++) ork_mm_run_i4(c,w,1,A+(size_t)m*K,Cs+(size_t)m*N); double t_sep=(now_us()-t0)/iters;

    printf("M=%d K=%d N=%d iters=%d\n",M,K,N,iters);
    printf("  incr (1 submit, %d incremental tasks) : %8.1f us  (%.1f us/row)\n", M, t_incr, t_incr/M);
    printf("  batch (STRATEGY A run_i4 M)           : %8.1f us  (%.1f us/row)\n", t_batch, t_batch/M);
    printf("  separate (per-row run_i4, %d submits)  : %8.1f us  (%.1f us/row)\n", M, t_sep, t_sep/M);
    printf("  incr vs separate: %.2fx   incr vs batch: %.2fx\n", t_sep/t_incr, t_batch/t_incr);

    ork_w_free(w); ork_npu_free(c);
    return bad?1:0;
}
