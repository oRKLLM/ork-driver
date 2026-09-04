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
#include <pthread.h>
/* CPU MEMORY-BANDWIDTH CONTENDERS. The probe measures the NPU with an IDLE CPU; a real decode token runs
 * ggml's norms/attention/sampling on 4 big cores at the same time, competing for the SAME DRAM. If the
 * NPU's achieved bandwidth collapses under CPU load, then moving work to the NPU does not ADD a parallel
 * resource -- it splits one memory bus -- which would explain why the FFN handler gets 3.1-6.3 GB/s where
 * this probe gets 8.8-16 at identical shapes through the identical call. ORK_BW_LOAD=<threads>. */
static volatile int g_load_stop = 0;
static void *load_thread(void *arg){
    size_t n = 64u<<20; char *a = malloc(n), *b = malloc(n);
    if(!a||!b) return arg;
    memset(a,1,n);
    while(!g_load_stop) memcpy(b,a,n);
    free(a); free(b); return arg;
}
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }
#define SMAX 6
#define NWMAX 96
int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):1024, N=argc>2?atoi(argv[2]):3072, REP=argc>3?atoi(argv[3]):40;
    /* nw = how many DISTINCT weights the run cycles through. The point: this probe originally reused <=6
     * weights in a tight loop and measured 8.6 GB/s at S=1, while the real FFN decode handler -- same
     * shapes, same ork_i8_mm_run -- gets 3.5. A decode token touches 28 layers x 3 weights = 84 DISTINCT
     * 3MB regions, so if bandwidth degrades with the number of distinct weights, the handler's gap is a
     * WORKING-SET effect (IOMMU TLB / page-table-walk locality), not batching or pipelining. */
    int NW=argc>4?atoi(argv[4]):SMAX; if(NW<1)NW=1; if(NW>NWMAX)NW=NWMAX;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("decode_bw_probe: M=1, K=%d N=%d, weight=%.2f MB each, %d reps\n",K,N,(double)K*N/1048576.0,REP);
    int8_t *B=malloc((size_t)K*N); uint32_t g=0x9e;
    for(size_t i=0;i<(size_t)K*N;i++){ g=g*1664525u+1013904223u; B[i]=(int8_t)((g>>25)%7)-3; }
    int8_t *A=malloc((size_t)K); for(int k=0;k<K;k++)A[k]=(int8_t)(k&15);
    ork_w *w[NWMAX]; int32_t *C[NWMAX];
    for(int i=0;i<NW;i++){ w[i]=ork_i8_mm_pack(c,K,N,B); C[i]=calloc((size_t)N,4);
        if(!w[i]){ printf("pack %d failed (nw=%d, %.0f MB) — lower nw\n",i,NW,(double)NW*K*N/1048576.0); return 2; } }
    printf("  distinct weights nw=%d (%.0f MB resident)\n", NW, (double)NW*K*N/1048576.0);
    int NL = getenv("ORK_BW_LOAD") ? atoi(getenv("ORK_BW_LOAD")) : 0;
    pthread_t lt[8]; if(NL>8)NL=8;
    for(int i=0;i<NL;i++) pthread_create(&lt[i],0,load_thread,0);
    if(NL) { printf("  CPU contenders: %d threads streaming memcpy\n", NL); struct timespec ts={0,200*1000*1000}; nanosleep(&ts,0); }
    printf("  S  tasks  ms/batch   GB/s   GB/s per task\n");
    for(int S=1;S<=SMAX && S<=NW;S++){
        ork_mm_task_i8 t[SMAX];
        for(int i=0;i<S;i++) t[i]=(ork_mm_task_i8){ w[i], 1, A, C[i] };
        if(ork_i8_mm_run_chain(c,S,t)){ printf("  %d  chain rc!=0 (declined)\n",S); continue; }   /* warm */
        double t0=now_us();
        /* ROTATE through the pool so each rep touches a different set -- the real decode access pattern. */
        /* ORK_BW_FRESH=1: malloc/free the A and C buffers EVERY rep, as the FFN decode handler does (6
         * fresh buffers per layer, ~168 per token) instead of reusing them. If anything downstream caches
         * per-buffer state -- a DMA-mapping lookup, a warm descriptor -- fresh pointers miss every call,
         * and that is the last structural difference between this probe and the handler at identical
         * shapes through the identical entrypoint. */
        static int fresh=-1; if(fresh<0) fresh=getenv("ORK_BW_FRESH")?1:0;
        for(int r=0;r<REP;r++){
            int off=(r*S)%NW;
            int8_t *Af=A; int32_t *Cf[SMAX];
            if(fresh){ Af=malloc((size_t)K); memcpy(Af,A,(size_t)K);
                       for(int i=0;i<S;i++) Cf[i]=calloc((size_t)N,4); }
            for(int i=0;i<S;i++){ int j=(off+i)%NW; t[i]=(ork_mm_task_i8){ w[j], 1, Af, fresh?Cf[i]:C[j] }; }
            int rc=ork_i8_mm_run_chain(c,S,t);
            if(fresh){ free(Af); for(int i=0;i<S;i++) free(Cf[i]); }
            if(rc){ printf("  %d  FAILED mid-loop\n",S); break; } }
        double us=(now_us()-t0)/REP;
        double bytes=(double)S*K*N;
        printf("  %d  %5d  %8.3f  %5.2f  %5.2f\n", S, S, us/1000.0, bytes/us/1000.0, bytes/us/1000.0/S);
    }
    g_load_stop = 1; for(int i=0;i<NL;i++) pthread_join(lt[i],0);
    printf("PASS — bandwidth scaling measured\n");
    for(int i=0;i<NW;i++){ ork_mm_free(c,w[i]); free(C[i]); }
    ork_npu_free(c); free(A); free(B); return 0;
}
