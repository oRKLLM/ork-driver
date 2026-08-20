/* sram_bw_probe — is NPU-on-SRAM a SEPARATE memory path from CPU-on-DRAM?
 *
 * The CPU/NPU decode PARTITION only wins if the NPU reading weights from on-chip SRAM does NOT contend with
 * the CPU reading activations/weights from DRAM. This probe measures that contention directly:
 *   (1) CPU DRAM read-bandwidth ALONE (memcpy loop on the big cores).
 *   (2) NPU int8 matmul throughput ALONE, weight in SRAM (ORK_WEIGHT_SRAM=1) vs DRAM.
 *   (3) BOTH concurrently -> if the CPU BW and the NPU rate each hold ~their solo value, SRAM is a separate
 *       path (partition wins). If either craters, they share the memory controller (partition can't help).
 *
 *   make sram_bw_probe && sudo env ORK_WEIGHT_SRAM=1 ./sram_bw_probe
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

/* CPU DRAM bandwidth hog: streams a big buffer on the A76 big cores, reports GB/s; runs until *stop. */
static volatile int g_stop = 0;
static volatile double g_cpu_gbps = 0;
static void *cpu_dram_hog(void *arg){
    size_t BYTES = 256u<<20;   /* 256 MiB: far bigger than any cache -> real DRAM traffic */
    cpu_set_t s; CPU_ZERO(&s); for(int c=4;c<8;c++) CPU_SET(c,&s); sched_setaffinity(0,sizeof s,&s);
    char *a = malloc(BYTES), *b = malloc(BYTES); memset(a,1,BYTES); memset(b,2,BYTES);
    double t0=now_us(); size_t moved=0;
    while(!g_stop){ memcpy(b,a,BYTES); moved+=BYTES; if((moved% (BYTES*2))==0){ double dt=(now_us()-t0)/1e6; if(dt>0) g_cpu_gbps=moved/1e9/dt; } }
    double dt=(now_us()-t0)/1e6; if(dt>0) g_cpu_gbps=moved/1e9/dt;
    free(a); free(b); (void)arg; return NULL;
}

/* NPU int8 matmul rate (submits/sec) over ITER iterations of run_i8 on a resident weight. */
static double npu_rate(ork_npu*c, ork_w*w, int M, int8_t*A, int32_t*C, int iter){
    double t0=now_us(); for(int i=0;i<iter;i++) ork_i8_mm_run(c,w,M,A,C); return iter/((now_us()-t0)/1e6);
}

int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):512, N=argc>2?atoi(argv[2]):256, M=argc>3?atoi(argv[3]):1, ITER=argc>4?atoi(argv[4]):400;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    size_t total=ork_npu_sram_total(c), free0=ork_npu_sram_free(c);
    printf("sram_bw_probe: K=%d N=%d M=%d iter=%d | SRAM total=%zuKiB free=%zuKiB (weight tile=%zuKiB, ORK_WEIGHT_SRAM=%s)\n",
           K,N,M,ITER, total>>10, free0>>10, ((size_t)K*N)>>10, getenv("ORK_WEIGHT_SRAM")?"1":"0");
    int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    ork_w*w=ork_i8_mm_pack(c,K,N,B); if(!w){printf("pack fail\n");return 1;}
    size_t free1=ork_npu_sram_free(c);
    printf("  after pack: SRAM free=%zuKiB (consumed %zdKiB -> weight %s SRAM)\n",
           free1>>10, ((ssize_t)free0-(ssize_t)free1)>>10, (free0-free1)>=(size_t)K*N ? "IN" : "NOT in (DRAM)");
    int8_t*A=(int8_t*)ork_dma_alloc(c,(size_t)M*K); memset(A,1,(size_t)M*K);
    int32_t*C=(int32_t*)ork_dma_alloc(c,(size_t)M*N*4);

    ork_i8_mm_run(c,w,M,A,C);   /* warm */
    double npu_solo = npu_rate(c,w,M,A,C,ITER);
    printf("  [1] NPU solo:            %.0f submits/s\n", npu_solo);

    /* [2] CPU DRAM bandwidth solo */
    g_stop=0; pthread_t th; pthread_create(&th,NULL,cpu_dram_hog,NULL);
    double ts=now_us(); while(now_us()-ts<1.2e6) ; double cpu_solo=g_cpu_gbps;
    g_stop=1; pthread_join(th,NULL);
    printf("  [2] CPU DRAM solo:       %.1f GB/s\n", cpu_solo);

    /* [3] BOTH concurrent: CPU DRAM hog running WHILE the NPU submits */
    g_stop=0; g_cpu_gbps=0; pthread_create(&th,NULL,cpu_dram_hog,NULL);
    ts=now_us(); while(now_us()-ts<4e5) ;   /* let the hog ramp */
    double npu_both = npu_rate(c,w,M,A,C,ITER);
    double cpu_both = g_cpu_gbps;
    g_stop=1; pthread_join(th,NULL);
    printf("  [3] concurrent:          NPU %.0f submits/s (%.0f%% of solo) | CPU %.1f GB/s (%.0f%% of solo)\n",
           npu_both, 100.0*npu_both/npu_solo, cpu_both, 100.0*cpu_both/cpu_solo);
    int sep = (npu_both>0.9*npu_solo && cpu_both>0.9*cpu_solo);
    printf("  ==> %s\n", sep ? "SEPARATE memory paths (partition can win: NPU-SRAM ‖ CPU-DRAM)"
                              : "SHARED memory controller (partition limited — both degrade under contention)");
    ork_npu_free(c); return 0;
}
