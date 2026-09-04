/* decode_bw_probe — how does M=1 weight-DMA bandwidth scale with the number of CONCURRENT independent
 * matmuls in one doorbell batch? This is the question that decides whether dense decode can ever beat the
 * CPU on this NPU: ORK_FFN_PROF measured ~3.5 GB/s for one task and 7.2 GB/s for two, i.e. ~3.5 GB/s per
 * concurrent task, while the CPU achieves ~9.4 GB/s effective on the same weights. If that scaling
 * continues, >=3 streams matches the CPU; if it plateaus at 2, dense decode is structurally lost.
 *   sudo env ORK_MM_TIMEOUT=3000 ./decode_bw_probe [K] [N] [reps]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }
#define SMAX 6
int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):1024, N=argc>2?atoi(argv[2]):3072, REP=argc>3?atoi(argv[3]):40;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("decode_bw_probe: M=1, K=%d N=%d, weight=%.2f MB each, %d reps\n",K,N,(double)K*N/1048576.0,REP);
    int8_t *B=malloc((size_t)K*N); uint32_t g=0x9e;
    for(size_t i=0;i<(size_t)K*N;i++){ g=g*1664525u+1013904223u; B[i]=(int8_t)((g>>25)%7)-3; }
    int8_t *A=malloc((size_t)K); for(int k=0;k<K;k++)A[k]=(int8_t)(k&15);
    ork_w *w[SMAX]; int32_t *C[SMAX];
    for(int i=0;i<SMAX;i++){ w[i]=ork_i8_mm_pack(c,K,N,B); C[i]=calloc((size_t)N,4);
        if(!w[i]){ printf("pack %d failed\n",i); return 2; } }
    printf("  S  tasks  ms/batch   GB/s   GB/s per task\n");
    for(int S=1;S<=SMAX;S++){
        ork_mm_task_i8 t[SMAX];
        for(int i=0;i<S;i++) t[i]=(ork_mm_task_i8){ w[i], 1, A, C[i] };
        if(ork_i8_mm_run_chain(c,S,t)){ printf("  %d  chain rc!=0 (declined)\n",S); continue; }   /* warm */
        double t0=now_us();
        for(int r=0;r<REP;r++) if(ork_i8_mm_run_chain(c,S,t)){ printf("  %d  FAILED mid-loop\n",S); break; }
        double us=(now_us()-t0)/REP;
        double bytes=(double)S*K*N;
        printf("  %d  %5d  %8.3f  %5.2f  %5.2f\n", S, S, us/1000.0, bytes/us/1000.0, bytes/us/1000.0/S);
    }
    printf("PASS — bandwidth scaling measured\n");
    for(int i=0;i<SMAX;i++){ ork_mm_free(c,w[i]); free(C[i]); }
    ork_npu_free(c); free(A); free(B); return 0;
}
