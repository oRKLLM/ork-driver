/* ork_dyn_queue_bench — head-to-head for the CPU‖NPU decode split, and single vs multi-core NONBLOCK.
 * N independent int8 matmuls (M=1), the NPU's decode share. Compares NPU-only wall time for:
 *   T_sc     : single-core queue   (ork_dyn_queue ncore=1  -> ork_dyn_begin, chained, 1 core)
 *   T_mc     : multi-core  queue   (ork_dyn_queue ncore=3  -> ork_dyn_begin_mc, NONBLOCK, 3 cores)
 *   T_stream : blocking 3-core stream (ork_i8_mm_run_stream — the current #14 decode dispatch)
 * then the overlap that matters: T_mc_over = mc-queue flush ‖ CPU bulk, vs T_seq = mc then CPU.
 *   make ork_dyn_queue_bench && sudo ./ork_dyn_queue_bench [N=32]
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
static volatile uint64_t g_sink;
static void cpu_bulk(uint8_t *buf, size_t sz, int reps){ uint64_t s=0; for(int r=0;r<reps;r++) for(size_t i=0;i<sz;i+=64) s+=buf[i]; g_sink=s; }
static int check(const int32_t*O,int N,int Nn,int K){ int ok=0; for(int i=0;i<N;i++) if(O[(size_t)i*Nn+(Nn-1)]==K) ok++; return ok; }
static void reseed(int32_t*O,int N,int Nn){ for(int i=0;i<N;i++){ volatile int32_t*d=(volatile int32_t*)(O+(size_t)i*Nn+(Nn-1)); *d=0; __asm__ volatile("dc civac,%0"::"r"(d):"memory"); } __asm__ volatile("dsb ish":::"memory"); }

int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):32, K=512, Nn=512;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int8_t*A=(int8_t*)malloc(K); memset(A,1,K);
    int8_t*B=malloc((size_t)K*Nn); memset(B,1,(size_t)K*Nn);
    ork_w*w=ork_i8_mm_pack(c,K,Nn,B); if(!w){printf("pack fail\n");return 1;}
    int32_t*O=(int32_t*)ork_dma_alloc(c,(size_t)N*Nn*sizeof(int32_t)); if(!O){printf("dma fail\n");return 1;}
    ork_mm_task_i8*tk=malloc(sizeof(*tk)*N);
    for(int i=0;i<N;i++){ tk[i].w=w; tk[i].M=1; tk[i].A=A; tk[i].C=O+(size_t)i*Nn; }
    size_t bulk_sz=64u<<20; uint8_t*bulk=malloc(bulk_sz); memset(bulk,1,bulk_sz);
    printf("ork_dyn_queue_bench: N=%d int8 matmuls (M=1,K=%d,N=%d)\n", N,K,Nn);

    /* ---- single-core queue ---- */
    { ork_dyn_queue*q=ork_dyn_queue_create(c,0,1); for(int i=0;i<N;i++)ork_dyn_queue_push(q,&tk[i]); ork_dyn_queue_flush(q); ork_dyn_queue_drain(q); ork_dyn_queue_destroy(q); } /*warm*/
    reseed(O,N,Nn); double t=now_us();
    { ork_dyn_queue*q=ork_dyn_queue_create(c,0,1); for(int i=0;i<N;i++)ork_dyn_queue_push(q,&tk[i]); ork_dyn_queue_flush(q); ork_dyn_queue_drain(q); ork_dyn_queue_destroy(q); }
    double T_sc=now_us()-t; int ok_sc=check(O,N,Nn,K);

    /* ---- multi-core queue (NONBLOCK, 3 cores) ---- */
    { ork_dyn_queue*q=ork_dyn_queue_create(c,0,3); for(int i=0;i<N;i++)ork_dyn_queue_push(q,&tk[i]); ork_dyn_queue_flush(q); ork_dyn_queue_drain(q); ork_dyn_queue_destroy(q); } /*warm*/
    reseed(O,N,Nn); t=now_us();
    { ork_dyn_queue*q=ork_dyn_queue_create(c,0,3); for(int i=0;i<N;i++)ork_dyn_queue_push(q,&tk[i]); ork_dyn_queue_flush(q); ork_dyn_queue_drain(q); ork_dyn_queue_destroy(q); }
    double T_mc=now_us()-t; int ok_mc=check(O,N,Nn,K);

    /* ---- blocking 3-core stream (current #14 decode dispatch) ---- */
    ork_i8_mm_run_stream(c,N,tk); /*warm*/
    reseed(O,N,Nn); t=now_us(); int src=ork_i8_mm_run_stream(c,N,tk); double T_stream=now_us()-t; int ok_st=check(O,N,Nn,K);

    /* ---- overlap: mc-queue flush ‖ CPU bulk, vs serialized ---- */
    double one; { double t1=now_us(); cpu_bulk(bulk,bulk_sz,1); one=now_us()-t1; }
    int reps=(int)(T_mc/one); if(reps<1)reps=1;
    double T_cpu; { double t1=now_us(); cpu_bulk(bulk,bulk_sz,reps); T_cpu=now_us()-t1; }
    reseed(O,N,Nn); t=now_us();
    { ork_dyn_queue*q=ork_dyn_queue_create(c,0,3); for(int i=0;i<N;i++)ork_dyn_queue_push(q,&tk[i]); ork_dyn_queue_flush(q); ork_dyn_queue_drain(q); ork_dyn_queue_destroy(q); }
    cpu_bulk(bulk,bulk_sz,reps);
    double T_seq=now_us()-t;
    reseed(O,N,Nn); t=now_us();
    ork_dyn_queue*q=ork_dyn_queue_create(c,0,3); for(int i=0;i<N;i++)ork_dyn_queue_push(q,&tk[i]);
    ork_dyn_queue_flush(q); cpu_bulk(bulk,bulk_sz,reps); int nd=ork_dyn_queue_drain(q); ork_dyn_queue_destroy(q);
    double T_over=now_us()-t; int ok_ov=check(O,N,Nn,K);

    printf("  NPU-only:  single-core=%.0fus (%s)  multi-core=%.0fus (%s)  stream3c=%.0fus rc=%d (%s)\n",
           T_sc, ok_sc==N?"ok":"WRONG", T_mc, ok_mc==N?"ok":"WRONG", T_stream, src, ok_st==N?"ok":"WRONG");
    printf("  ★ multi-core vs single-core = %.2fx ;  multi-core vs blocking-stream = %.2fx\n", T_sc/T_mc, T_stream/T_mc);
    printf("  overlap (mc ‖ cpu): T_cpu=%.0f T_seq=%.0f T_over=%.0f -> efficiency %.2fx (drained=%d, %s)\n",
           T_cpu, T_seq, T_over, T_seq/T_over, nd, ok_ov==N?"correct":"WRONG");
    ork_npu_free(c); return (ok_sc==N&&ok_mc==N&&ok_st==N&&ok_ov==N)?0:2;
}
