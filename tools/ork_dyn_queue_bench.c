/* ork_dyn_queue_bench — demonstrate the CPU‖NPU decode-split overlap the submit queue enables.
 * The NPU runs a chunk of int8 matmuls NONBLOCK (ork_dyn_queue_flush) while the CPU does a bulk workload
 * in parallel; drain rendezvouses. We compare:
 *   T_npu   : NPU chunk alone (flush+drain, no CPU work)
 *   T_cpu   : CPU bulk alone
 *   T_seq   : NPU then CPU, serialized (what per-node ggml sync gives)  ~= T_npu + T_cpu
 *   T_over  : NPU ‖ CPU via the queue (flush, CPU bulk, drain)          ~= max(T_npu, T_cpu)
 * overlap efficiency = T_seq / T_over  (1.0 = none; ->2.0 = fully hidden when T_npu~=T_cpu).
 *   make ork_dyn_queue_bench && sudo ./ork_dyn_queue_bench [N=16] [cpu_us=0=auto]
 * (NPU op; run alone.)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

/* memory-bound CPU bulk (mimics reading int4 weights): sum over a big buffer `reps` times */
static volatile uint64_t g_sink;
static void cpu_bulk(uint8_t *buf, size_t sz, int reps){ uint64_t s=0; for(int r=0;r<reps;r++) for(size_t i=0;i<sz;i+=64) s+=buf[i]; g_sink=s; }

int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):16, K=512, Nn=512;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int8_t*A=(int8_t*)malloc(K); memset(A,1,K);
    int8_t*B=malloc((size_t)K*Nn); memset(B,1,(size_t)K*Nn);
    ork_w*w=ork_mm_pack_i8(c,K,Nn,B); if(!w){printf("pack fail\n");return 1;}
    int32_t*O=(int32_t*)ork_dma_alloc(c,(size_t)N*Nn*sizeof(int32_t)); if(!O){printf("dma fail\n");return 1;}
    ork_mm_task_i8*tk=malloc(sizeof(*tk)*N);
    for(int i=0;i<N;i++){ tk[i].w=w; tk[i].M=1; tk[i].A=A; tk[i].C=O+(size_t)i*Nn; }
    /* CPU bulk buffer ~ a decode weight slab */
    size_t bulk_sz=64u<<20; uint8_t*bulk=malloc(bulk_sz); memset(bulk,1,bulk_sz);

    printf("ork_dyn_queue_bench: N=%d int8 matmuls (M=1,K=%d,N=%d), chunk_max=%d\n", N, K, Nn, ork_dyn_max_steps());

    /* warm the mode + calibrate CPU reps to ~match one NPU chunk */
    { ork_dyn_queue*q=ork_dyn_queue_create(c,0); for(int i=0;i<N;i++) ork_dyn_queue_push(q,&tk[i]); ork_dyn_queue_flush(q); ork_dyn_queue_drain(q); ork_dyn_queue_destroy(q); }

    double t0=now_us();
    { ork_dyn_queue*q=ork_dyn_queue_create(c,0); for(int i=0;i<N;i++) ork_dyn_queue_push(q,&tk[i]);
      ork_dyn_queue_flush(q); ork_dyn_queue_drain(q); ork_dyn_queue_destroy(q); }
    double T_npu=now_us()-t0;

    /* calibrate CPU reps so T_cpu ~= T_npu (best-case overlap demo); or use arg */
    int reps=1; double t1=now_us(); cpu_bulk(bulk,bulk_sz,1); double one=now_us()-t1;
    if(argc>2 && atoi(argv[2])>0){ reps=(int)(atoi(argv[2])/ (one) ); if(reps<1)reps=1; }
    else { reps=(int)(T_npu/one); if(reps<1)reps=1; }
    t1=now_us(); cpu_bulk(bulk,bulk_sz,reps); double T_cpu=now_us()-t1;

    /* sequential: NPU chunk, THEN cpu bulk */
    double t2=now_us();
    { ork_dyn_queue*q=ork_dyn_queue_create(c,0); for(int i=0;i<N;i++) ork_dyn_queue_push(q,&tk[i]);
      ork_dyn_queue_flush(q); ork_dyn_queue_drain(q); ork_dyn_queue_destroy(q); }
    cpu_bulk(bulk,bulk_sz,reps);
    double T_seq=now_us()-t2;

    /* overlapped: flush (NPU starts), cpu bulk in parallel, drain */
    double t3=now_us();
    ork_dyn_queue*q=ork_dyn_queue_create(c,0); for(int i=0;i<N;i++) ork_dyn_queue_push(q,&tk[i]);
    ork_dyn_queue_flush(q);            /* NPU runs NONBLOCK */
    cpu_bulk(bulk,bulk_sz,reps);       /* CPU works in parallel */
    int nd=ork_dyn_queue_drain(q);     /* rendezvous */
    ork_dyn_queue_destroy(q);
    double T_over=now_us()-t3;

    /* correctness: all outputs == K */
    int ok=0; for(int i=0;i<N;i++) if(O[(size_t)i*Nn+(Nn-1)]==K) ok++;
    printf("  T_npu=%.0fus  T_cpu=%.0fus  T_seq=%.0fus  T_over=%.0fus\n", T_npu, T_cpu, T_seq, T_over);
    printf("  drained=%d, outputs==K %d/%d %s\n", nd, ok, N, ok==N?"(correct)":"(WRONG)");
    printf("  ★ overlap efficiency = T_seq/T_over = %.2fx (1.0=none, ->2.0 fully hidden when T_npu~=T_cpu)\n", T_seq/T_over);
    ork_npu_free(c); return ok==N?0:2;
}
