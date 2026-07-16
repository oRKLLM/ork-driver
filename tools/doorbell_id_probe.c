/* doorbell_id_probe — can the host observe PER-OP chain progress by polling (the per-op "ID doorbell")?
 *
 * Idea (user): make the doorbell a progress counter — the host busy-polls and learns WHICH op the NPU just
 * finished (fine-grained CPU-consume overlap + wedge pinpointing), vs today's single done/not-done flag.
 * This probe tests the enabling fact: when a chain (run_chain_i8, S concatenated tasks) executes, does each
 * task's OUTPUT hit DRAM as that task completes (visible via dc civac), so polling the S output slots reveals
 * completions IN ORDER mid-chain — or do they all appear at once only at chain end?
 *
 * Method: S int8 matmuls (all-ones -> each output == K), each C -> its own slot in one ork_dma_alloc buffer.
 * Seed each slot's last int32 = SENT (unproducible). Run the chain on a thread; the main thread busy-polls
 * every slot (dc civac invalidate) and timestamps when each leaves SENT. Monotonic staggered timestamps
 * (slot 0 < 1 < ... , with gaps) => per-op progress is observable. All-at-once => only whole-chain granularity.
 *   make doorbell_id_probe && sudo ./doorbell_id_probe [S=8] [K=512] [N=512]
 * (NPU op — do NOT run concurrently with another NPU workload.)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static inline void civac(volatile void*p){ __asm__ volatile("dc civac,%0"::"r"(p):"memory"); }
static inline void cvac (volatile void*p){ __asm__ volatile("dc cvac,%0" ::"r"(p):"memory"); }
static inline void dsb  (void){ __asm__ volatile("dsb ish":::"memory"); }

#define SENT 0x7fffffff

static ork_npu *gC; static int gS; static ork_mm_task_i8 *gTK; static volatile int gRc, gDone;
static void* chain_thr(void*p){ (void)p; gRc = ork_mm_run_chain_i8(gC, gS, gTK); gDone = 1; return NULL; }

int main(int argc,char**argv){
    int S=argc>1?atoi(argv[1]):8, K=argc>2?atoi(argv[2]):512, N=argc>3?atoi(argv[3]):512;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;} gC=c;
    printf("doorbell_id_probe: S=%d chained matmuls (M=1,K=%d,N=%d), per-op output-slot polling\n",S,K,N);

    int8_t*A=malloc(K); memset(A,1,K);
    int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    ork_w*w=ork_mm_pack_i8(c,K,N,B); if(!w){printf("pack fail\n");return 1;}
    /* one coherent DMA output buffer, S slots of N int32 each */
    int32_t*Obuf=(int32_t*)ork_dma_alloc(c,(size_t)S*N*sizeof(int32_t));
    if(!Obuf){printf("dma_alloc fail (need zero-copy out)\n");return 1;}
    ork_mm_task_i8*tk=malloc(sizeof(ork_mm_task_i8)*S);
    for(int i=0;i<S;i++){ tk[i].w=w; tk[i].M=1; tk[i].A=A; tk[i].C=Obuf+(size_t)i*N; }
    gTK=tk; gS=S;

    /* warm (blocking) + correctness */
    if(ork_mm_run_chain_i8(c,S,tk)){printf("chain warm rc!=0\n");return 1;}
    int ok=1; for(int i=0;i<S;i++) if(Obuf[(size_t)i*N + (N-1)]!=K){ok=0;break;}
    printf("  chain correctness (all outputs==%d): %s\n",K,ok?"PASS":"FAIL");

    /* seed each slot's last int32 = SENT, flush to DRAM */
    for(int i=0;i<S;i++){ volatile int32_t*db=(volatile int32_t*)(Obuf+(size_t)i*N+(N-1)); *db=SENT; cvac((void*)db); }
    dsb();
    /* run the chain on a thread; poll all slots on main, timestamp each completion */
    double t0[64]; for(int i=0;i<S;i++) t0[i]=-1;
    gDone=0; gRc=0;
    pthread_t th; double start=now_us();
    pthread_create(&th,0,chain_thr,0);
    int got=0;
    double deadline=start+3.0e6;   /* 3s */
    while(got<S && now_us()<deadline){
        for(int i=0;i<S;i++){ if(t0[i]>=0) continue;
            volatile int32_t*db=(volatile int32_t*)(Obuf+(size_t)i*N+(N-1)); civac((void*)db);
            if(*db!=SENT){ t0[i]=now_us()-start; got++; } }
    }
    pthread_join(th,0);
    printf("  chain rc=%d, slots observed=%d/%d\n",gRc,got,S);
    printf("  per-slot completion time (us from submit-thread start):\n");
    int monotonic=1; double prev=-1;
    for(int i=0;i<S;i++){ printf("    slot %d: %s%.1f\n", i, t0[i]<0?"(never) ":"", t0[i]);
        if(t0[i]>=0){ if(t0[i]<prev-5) monotonic=0; prev=t0[i]; } }
    /* verdict: staggered (gaps between slots) + in-order => per-op progress observable */
    double span = (t0[S-1]>=0 && t0[0]>=0) ? t0[S-1]-t0[0] : -1;
    printf("  ★ observed span slot0->slot%d = %.1f us | in-order=%s\n", S-1, span, monotonic?"YES":"NO");
    printf("  => per-op progress %s (staggered+in-order => the ID-doorbell is feasible; all-at-once => whole-chain only)\n",
           (got==S && span>20 && monotonic)?"OBSERVABLE":"NOT observable at op granularity");
    ork_npu_free(c); return ok?0:2;
}
