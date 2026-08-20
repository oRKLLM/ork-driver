/* load_timing_probe.c — measure ork_i8_mm_load_import load cost per weight WITHOUT the full-model memory
 * pressure that wedges the board. Packs a representative gate/up-shape weight (K=3584 N=18944, the Bf-heavy
 * case: K<=4096 so it rebuilds full-K Bf), dumps the pre-tiled blob, then times ork_i8_mm_load_import of it
 * REPS times into a non-0 domain (anchor + chunked import + Bf re-tile — the real load path). Prints ms/load
 * and MB/s so we can see whether the byte-loop->memcpy Bf fix + the import mechanism are the load bottleneck.
 *   ./ltp [K=3584] [N=18944] [reps=4] [domain=1] */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IOLBF,0);
    int K=argc>1?atoi(argv[1]):3584, N=argc>2?atoi(argv[2]):18944, reps=argc>3?atoi(argv[3]):4, dom=argc>4?atoi(argv[4]):1;
    ork_npu*c=ork_npu_init(); if(!c){printf("no board\n");return 0;}
    size_t bytes=(size_t)K*N;
    printf("load-timing: K=%d N=%d (%.1f MB tile) reps=%d dom=%d — Bf built (K<=4096)\n", K,N, bytes/1e6, reps, dom);
    int8_t*B=malloc(bytes); for(size_t i=0;i<bytes;i++) B[i]=(int8_t)(i&0x7f);
    ork_npu_set_pack_domain(c,0);
    ork_w*w0=ork_i8_mm_pack(c,K,N,B); if(!w0){printf("pack failed\n");return 1;}
    size_t need=ork_w_dump(w0,NULL,0); void*blob=malloc(need); ork_w_dump(w0,blob,need);
    printf("blob=%.1f MB\n", need/1e6);

    ork_npu_set_pack_domain(c,dom);
    double tot=0;
    for(int r=0;r<reps;r++){
        double t0=now_ms();
        ork_w*wi=ork_i8_mm_load_import(c,K,N,blob,need);
        double dt=now_ms()-t0;
        if(!wi){printf("load %d FAILED\n",r);break;}
        printf("  load %d: %.1f ms  (%.2f GB/s over %.1f MB blob)\n", r, dt, need/1e6/dt/1e3*1e3/1e3, need/1e6);
        tot+=dt; ork_mm_free(c,wi);
    }
    printf("avg load: %.1f ms/weight  (=> ~%.0f s for a 196-weight 7B, load path only)\n", tot/reps, tot/reps*196/1000.0);
    return 0;
}
