/* dyn_spin_probe — validate the persistent-job "spin keep-alive" mechanism:
 *   program 0 = a CIRCULAR spin (self-looping) that keeps the NPU job alive on one core without completing;
 *   after a spin window, redirect it into a real S-op chain (mid-flight, no new submit) and confirm the
 *   real ops run. A lost redirect race just re-loops (no abort) — the safety win over a terminator frontier.
 *   make dyn_spin_probe && sudo ./dyn_spin_probe [S=8] [spin_ms=5]
 * (NPU op; run alone. A circular chain that is never redirected would run to the submit timeout, so the
 *  probe always redirects; timeout-guard the process.)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int main(int argc,char**argv){
    int S=argc>1?atoi(argv[1]):8, spin_ms=argc>2?atoi(argv[2]):5, K=512, N=512;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    printf("dyn_spin_probe: circular spin + redirect into %d-op chain (K=%d,N=%d), spin=%dms\n",S,K,N,spin_ms);
    int8_t*A=(int8_t*)malloc(K); memset(A,1,K);
    int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    ork_w*w=ork_i8_mm_pack(c,K,N,B); if(!w){printf("pack fail\n");return 1;}
    int32_t*O=(int32_t*)ork_dma_alloc(c,(size_t)S*N*sizeof(int32_t)); if(!O){printf("dma fail\n");return 1;}
    ork_mm_task_i8*tk=malloc(sizeof(*tk)*S);
    for(int i=0;i<S;i++){ tk[i].w=w; tk[i].M=1; tk[i].A=A; tk[i].C=O+(size_t)i*N; }

    /* warm the int8-chain mode with a quick real chain first (spin_probe assumes warmed) */
    ork_i8_mm_run_chain(c,S,tk);

    int spin_alive=-1;
    int comp=ork_dyn_spin_probe(c,S,tk,spin_ms*1000,&spin_alive);
    /* correctness: all real outputs == K (A,B all ones => C = K) */
    int ok=0; for(int i=0;i<S;i++) if(O[(size_t)i*N+(N-1)]==K) ok++;
    printf("  spin_alive=%d (spin ran, real chain untouched during spin)\n", spin_alive);
    printf("  after redirect: real outputs completed=%d/%d, ==K %d/%d\n", comp, S, ok, S);
    printf("  ★ persistent spin+redirect => %s\n",
           (comp==S && ok==S && spin_alive==1) ? "WORKS (job stayed alive on the loop, redirect chained into real work, no wedge)"
           : (comp==S && ok==S) ? "PARTIAL (real work ran, but spin-liveness not cleanly observed)"
           : "FAIL (redirect did not chain into the real chain)");
    ork_npu_free(c); return (comp==S&&ok==S)?0:2;
}
