/* tools/rr_experiment.c — ROUND-ROBIN single-core vs BARRIER multi-core, using the single-core int8
 * matmul that works at R=32 (cbuf=57344). Hypothesis: rknn keeps one core ~96% compute-bound; if ork
 * round-robins INDEPENDENT single-core submits across the 3 cores (no barrier) instead of the barrier
 * N-split run_multicore, it may reach closer to 3x a single core (vs run_multicore's ~2.5x w/ barrier).
 *
 * Method: one prefill matmul K x N x M, computed three ways, all bit-exact CPU-checked:
 *   (a) 1-core            : ork_i8_mm_run, budget=1               (the working single-core R=32 matmul)
 *   (b) barrier 3-core    : ork_i8_mm_run, budget=3 (run_multicore N-split + barrier)  -- current prod
 *   (c) round-robin 3-core: split N into 3 slices, each a separate single-core matmul, dispatched
 *                           round-robin via ork_i8_mm_run_stream (no barrier, 1 task/core)
 *   make rr_experiment && sudo ./rr_experiment [M] [K] [N] [iters]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):256, K=argc>2?atoi(argv[2]):3584, N=argc>3?atoi(argv[3]):3072, iters=argc>4?atoi(argv[4]):30;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int cores=ork_npu_cores(c);
    printf("INIT soc=%s cores=%d  M=%d K=%d N=%d (R=%d/K at cbuf)\n", ork_npu_soc(c),cores,M,K,N,2*32768/K);
    if(N%3 || (N/3)%32){ printf("need N%%3==0 and (N/3)%%32==0\n"); return 1; }
    int Ns=N/3;

    int8_t*B=malloc((size_t)K*N); for(size_t i=0;i<(size_t)K*N;i++)B[i]=(int8_t)((i%5)-2);
    int8_t*A=malloc((size_t)M*K); for(size_t i=0;i<(size_t)M*K;i++)A[i]=(int8_t)((i%7)-3);
    int do_check = ((long long)M*N*K < 2000000000LL) && !getenv("ORK_RR_NOCHECK");
    int32_t*Cref=malloc((size_t)M*N*4);
    if(do_check) for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long acc=0; for(int k=0;k<K;k++)acc+=(long)A[(size_t)m*K+k]*B[(size_t)k*N+n]; Cref[(size_t)m*N+n]=(int32_t)acc; }

    /* full weight (a,b) + 3 N-slices (c). N-slice s = B columns [s*Ns,(s+1)*Ns) packed as its own ork_w. */
    ork_npu_set_pack_domain(c,0);
    ork_w*wf=ork_i8_mm_pack(c,K,N,B); if(!wf){printf("pack full failed\n");return 1;}
    int8_t*Bs=malloc((size_t)K*Ns); ork_w*ws[3]; int32_t*Cs[3];
    for(int s=0;s<3;s++){
        for(int k=0;k<K;k++)for(int j=0;j<Ns;j++) Bs[(size_t)k*Ns+j]=B[(size_t)k*N + s*Ns + j];
        ork_npu_set_pack_domain(c,0); ws[s]=ork_i8_mm_pack(c,K,Ns,Bs); if(!ws[s]){printf("pack slice %d failed\n",s);return 1;}
        Cs[s]=malloc((size_t)M*Ns*4);
    }
    int32_t*C=malloc((size_t)M*N*4);
    #define CHK(buf,stride,nn,tag) do{ int mism=0; for(int m=0;m<M;m++)for(int n=0;n<nn;n++) if((buf)[(size_t)m*(stride)+n]!=Cref[(size_t)m*N + (n)]) mism++; if(mism) printf("  [%s] MISM=%d\n",tag,mism); }while(0)

    /* (a) single-core */
    ork_npu_set_core_budget(c,1); ork_i8_mm_run(c,wf,M,A,C);
    if(do_check){ int mism=0; for(size_t i=0;i<(size_t)M*N;i++) if(C[i]!=Cref[i])mism++; printf("(a) 1-core    mism=%d\n",mism); }
    double t0=now_us(); for(int it=0;it<iters;it++) ork_i8_mm_run(c,wf,M,A,C); double t1=(now_us()-t0)/iters;

    /* (b) barrier 3-core */
    ork_npu_set_core_budget(c,cores); ork_i8_mm_run(c,wf,M,A,C);
    if(do_check){ int mism=0; for(size_t i=0;i<(size_t)M*N;i++) if(C[i]!=Cref[i])mism++; printf("(b) 3c-barrier mism=%d\n",mism); }
    t0=now_us(); for(int it=0;it<iters;it++) ork_i8_mm_run(c,wf,M,A,C); double tb=(now_us()-t0)/iters;
    { double su,sb,cp; long n; ork_npu_run_timing(&su,&sb,&cp,&n);
      if(n) printf("  barrier phase/call: setup=%.1f submit=%.1f copy(cres->C)=%.1f us  (copy=%.0f%% of barrier)\n",
                   su/n, sb/n, cp/n, 100.0*(cp/n)/tb); }

    /* (c) round-robin single-core: 3 N-slice tasks, 1 per core, via run_stream */
    ork_npu_set_core_budget(c,cores);
    ork_mm_task_i8 tk[3]; for(int s=0;s<3;s++) tk[s]=(ork_mm_task_i8){ws[s],M,A,Cs[s]};
    ork_i8_mm_run_stream(c,3,tk);
    if(do_check){ int mism=0; for(int m=0;m<M;m++)for(int s=0;s<3;s++)for(int j=0;j<Ns;j++) if(Cs[s][(size_t)m*Ns+j]!=Cref[(size_t)m*N + s*Ns + j])mism++; printf("(c) 3c-roundrobin mism=%d\n",mism); }
    t0=now_us(); for(int it=0;it<iters;it++) ork_i8_mm_run_stream(c,3,tk); double tc=(now_us()-t0)/iters;

    printf("\n  (a) 1-core         : %8.1f us\n", t1);
    printf("  (b) 3c barrier     : %8.1f us   (%.2fx vs 1-core)\n", tb, t1/tb);
    printf("  (c) 3c round-robin : %8.1f us   (%.2fx vs 1-core)  rr/barrier=%.2f\n", tc, t1/tc, tc/tb);
    ork_npu_free(c); return 0;
}
