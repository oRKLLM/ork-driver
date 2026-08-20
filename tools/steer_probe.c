/* steer_probe — DECISIVE test: does the NPU PC sequencer read each program's chain descriptor from DRAM at
 * EXECUTION time (so we can steer/halt the chain mid-flight by editing DRAM) or PRE-CACHE the whole chain at
 * submit? If the former, dynamic submit / our-own-chaining-API (early-exit to free the resource, runtime
 * redirect) is reachable on the stock driver. If the latter, it needs MMIO.
 *
 * Method: build an S-op chain, each op -> its own output slot (ork_dma_alloc, coherent). WARM it (blocking,
 * env unset) -> all S outputs == K. Re-seed all slots = SENT. Then run again with ORK_STEER_HALT_AT=H set:
 * ork-driver submits NONBLOCK and, mid-flight, zeroes program H's next-amount (0x0014) in the live DRAM
 * regcmd. Count how many slots got written. written ~= H+1 (only ops 0..H) => STEERABLE (read-from-DRAM);
 * written == S (edit ignored) => PRE-CACHED.
 *   make steer_probe && sudo ./steer_probe [S=16] [halt=8]
 * (NPU op; the halt leaves a partial kernel job — run alone, reboot if the next NPU op misbehaves.)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static inline void civac(volatile void*p){ __asm__ volatile("dc civac,%0"::"r"(p):"memory"); }
static inline void cvac (volatile void*p){ __asm__ volatile("dc cvac,%0" ::"r"(p):"memory"); }
#define SENT 0x7fffffff

int main(int argc,char**argv){
    int S=argc>1?atoi(argv[1]):16, H=argc>2?atoi(argv[2]):8, K=512, N=512;
    setvbuf(stdout,0,_IONBF,0);
    if(H<0||H>=S-1){ printf("halt must be in [0,S-2]\n"); return 1; }
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    printf("steer_probe: S=%d chained ops (K=%d,N=%d), halt-inject at prog H=%d\n",S,K,N,H);
    int8_t*A=malloc(K); memset(A,1,K);
    int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    ork_w*w=ork_i8_mm_pack(c,K,N,B); if(!w){printf("pack fail\n");return 1;}
    int32_t*Obuf=(int32_t*)ork_dma_alloc(c,(size_t)S*N*sizeof(int32_t)); if(!Obuf){printf("dma_alloc fail\n");return 1;}
    ork_mm_task_i8*tk=malloc(sizeof(ork_mm_task_i8)*S);
    for(int i=0;i<S;i++){ tk[i].w=w; tk[i].M=1; tk[i].A=A; tk[i].C=Obuf+(size_t)i*N; }

    /* WARM (env unset => no steer): full chain, all S outputs must == K */
    unsetenv("ORK_STEER_HALT_AT");
    if(ork_i8_mm_run_chain(c,S,tk)){printf("warm chain rc!=0\n");return 1;}
    int warmok=1; for(int i=0;i<S;i++) if(Obuf[(size_t)i*N+(N-1)]!=K){warmok=0;break;}
    printf("  warm full chain (all %d outputs==%d): %s\n",S,K,warmok?"PASS":"FAIL");
    if(!warmok){ ork_npu_free(c); return 2; }

    /* re-seed every slot's last elem = SENT, flush */
    for(int i=0;i<S;i++){ volatile int32_t*db=(volatile int32_t*)(Obuf+(size_t)i*N+(N-1)); *db=SENT; cvac((void*)db); }
    __asm__ volatile("dsb ish":::"memory");

    /* STEER run: ork-driver submits NONBLOCK + zeroes prog H's 0x0014 mid-flight */
    char hs[16]; snprintf(hs,sizeof hs,"%d",H); setenv("ORK_STEER_HALT_AT",hs,1);
    int rc=ork_i8_mm_run_chain(c,S,tk);
    unsetenv("ORK_STEER_HALT_AT");

    /* count written slots (invalidate to see the NPU's writes) */
    int written=0, first_unwritten=-1;
    for(int i=0;i<S;i++){ volatile int32_t*db=(volatile int32_t*)(Obuf+(size_t)i*N+(N-1)); civac((void*)db);
        if(*db!=SENT) written++; else if(first_unwritten<0) first_unwritten=i; }
    printf("  steer chain rc=%d | outputs written=%d/%d | first unwritten slot=%d\n",rc,written,S,first_unwritten);
    printf("  ★ verdict: ");
    if(written==S) printf("PRE-CACHED (edit ignored — sequencer fetched the chain at submit; mid-flight steering NOT reachable, needs MMIO)\n");
    else if(written<=H+2 && written>=H) printf("STEERABLE (halted at/near prog H=%d => sequencer reads the descriptor from DRAM at exec-time; dynamic submit reachable, stock driver)\n",H);
    else printf("PARTIAL/AMBIGUOUS (written=%d, expected S=%d or ~H+1=%d) — inspect (race timing or a different mechanism)\n",written,S,H+1);
    ork_npu_free(c); return 0;
}
