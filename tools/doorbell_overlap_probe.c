/* doorbell_overlap_probe — increment 2 crux: is the DOORBELL (ork_dyn_*) NPU submit truly async, so CPU glue
 * overlaps the NPU's streaming window "for free"? (Increment 1 showed ork_mm_run_i8 is host-bound and does NOT
 * overlap; the spine's NPU unit must be the doorbell.) Also confirms the doorbell path is coherent.
 *
 *   begin(S tasks)     -> NONBLOCK submit; the NPU runs the chain (kernel-driven), begin returns in ~µs
 *   <CPU glue on main> -> runs WHILE the NPU HW works (no thread needed — the async is in the HW/kernel)
 *   end()              -> drain + writeback (fast if the NPU already finished under the CPU work)
 *
 * S=8 chained M=1,K=512,N=512 matmuls (~8×256KB weight streams) give a real window. N=512 = safe single-slice.
 *   make doorbell_overlap_probe && sudo env ORK_MM_TIMEOUT=3000 ./doorbell_overlap_probe
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
static uint32_t rng=0x51edu; static int s3(void){ rng=rng*1664525u+1013904223u; return (int)((rng>>28)%3)-1; }
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static long cpu_glue(const int8_t *x, size_t n){ long acc=0; for(size_t i=0;i<n;i++){ long q=(long)x[i]>>1; if(q>63)q=63; if(q<-63)q=-63; acc+=q; } return acc; }

int main(void){
    int M=1, K=512, N=512, S=8;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    int cap=ork_dyn_max_steps(); if(S>cap)S=cap;
    printf("doorbell_overlap_probe: async doorbell chain (S=%d x M=%d K=%d N=%d) ∥ CPU glue (max_steps=%d)\n",S,M,K,N,cap);

    int8_t  *A=malloc((size_t)M*K);                 /* A is HOST malloc — the doorbell stages it via memcpy; a dma A miscomputes at M=1 */
    int32_t *C=ork_dma_alloc(c,(size_t)S*N*4);      /* C is the resident dma output (zero-copy); C+i*N per task */
    if(!A||!C){ printf("alloc FAILED\n"); return 2; }
    int8_t *Wb=malloc((size_t)K*N);
    for(int i=0;i<M*K;i++) A[i]=(int8_t)s3();
    for(size_t i=0;i<(size_t)K*N;i++) Wb[i]=(int8_t)s3();
    ork_w *W=ork_mm_pack_i8(c,K,N,Wb); if(!W){ printf("pack failed\n"); return 2; }
    ork_mm_task_i8 *tk=malloc((size_t)S*sizeof *tk);
    for(int s=0;s<S;s++) tk[s]=(ork_mm_task_i8){W,M,A,C+(size_t)s*N};

    int32_t *Cr=malloc((size_t)N*4);
    for(int n=0;n<N;n++){ long a=0; for(int k=0;k<K;k++) a+=A[k]*Wb[(size_t)k*N+n]; Cr[n]=(int32_t)a; }

    /* warm */
    { ork_dyn_chain*h=ork_dyn_begin(c,S,tk); if(h) ork_dyn_end(h); }

    /* (1) COHERENCY + async check. NOTE: the tasks fully overwrite C, so no memset — and a CPU memset/write of
     * the dma buffer C AFTER ork_dyn_end SIGBUSes (end leaves C device-owned); touch it only via the doorbell
     * or after a proper drain/re-sync. This is a hard rule for the spine's CPU worker sharing resident buffers. */
    ork_dyn_chain*h=ork_dyn_begin(c,S,tk);
    if(!h){ printf("ork_dyn_begin FAILED\n"); return 1; }
    ork_dyn_end(h);
    long bad=0; int32_t mx=0;
    for(int s=0;s<S;s++) for(int n=0;n<N;n++){ int32_t d=C[(size_t)s*N+n]-Cr[n]; if(d){bad++; if(d<0)d=-d; if(d>mx)mx=d;} }
    printf("  doorbell chain: %s (bad=%ld/%d maxdiff=%d) — async is inferred from the overlap timing below\n",
           bad?"MISMATCH":"bit-exact", bad, S*N, mx);

    /* (2) OVERLAP: serial vs (begin ∥ CPU ∥ end) */
    size_t SN=(size_t)K*N; int8_t*scr=malloc(SN); for(size_t i=0;i<SN;i++) scr[i]=(int8_t)(i*7);
    volatile long sink=0;
    double na=now_us(); for(int r=0;r<10;r++){ ork_dyn_chain*hh=ork_dyn_begin(c,S,tk); ork_dyn_end(hh); } double t_npu=(now_us()-na)/10;
    double ca=now_us(); for(int r=0;r<10;r++) sink+=cpu_glue(scr,SN);                                   double t_cpu=(now_us()-ca)/10;
    double oa=now_us();
    for(int r=0;r<10;r++){ ork_dyn_chain*hh=ork_dyn_begin(c,S,tk);   /* NPU chain starts */
        sink+=cpu_glue(scr,SN);                                       /* CPU glue while NPU streams */
        ork_dyn_end(hh); }                                           /* drain */
    double t_over=(now_us()-oa)/10;
    double eff = t_over>0 ? (t_npu+t_cpu)/t_over : 0;
    double smaller = t_cpu<t_npu?t_cpu:t_npu;
    double hidden = (t_npu+t_cpu)>t_over ? 100.0*(t_npu+t_cpu-t_over)/smaller : 0.0;
    printf("  NPU-alone %.0fus | CPU-alone %.0fus | overlapped %.0fus -> %.2fx (%.0f%% of the smaller hidden)\n",
           t_npu, t_cpu, t_over, eff, hidden>100?100:hidden);
    printf("%s\n", (bad==0 && eff>1.3) ? "PASS — doorbell async + coherent; CPU overlaps the NPU streaming window"
                 : (bad==0 ? "COHERENT but overlap weak (window too short / submit host-bound)" : "FAIL — doorbell incoherent"));
    (void)sink;
    free(A); ork_dma_free(c,C); ork_mm_free(c,W); ork_npu_free(c);
    return bad?1:0;
}
