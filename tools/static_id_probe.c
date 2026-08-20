/* static_id_probe — can a single doorbell address show the CURRENT op's ID (the "static ID" the host reads
 * from one location), rather than polling N per-op slots? Each op computes a chosen constant ID into the
 * SAME output slot (C[0] = A(all-1) . B where B[0][0]=id → C[0]=id). The host polls that one address and
 * should observe the IDs advance 1,2,...,S as the chain runs. Confirms a single-address progress doorbell.
 *   make static_id_probe && sudo ./static_id_probe [S=8]
 * (NPU op — not concurrently with another NPU workload.)
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
#define SENT 0x7fffffff

static ork_npu*gC; static int gS; static ork_mm_task_i8*gTK; static volatile int gRc,gDone;
static void* chain_thr(void*p){ (void)p; gRc=ork_i8_mm_run_chain(gC,gS,gTK); gDone=1; return NULL; }

int main(int argc,char**argv){
    int S=argc>1?atoi(argv[1]):8, K=512, N=32;   /* K%512==0 chain-conforming; N%16==0 */
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;} gC=c;
    printf("static_id_probe: S=%d ops, each writes its ID to ONE shared doorbell slot (C[0])\n",S);
    int8_t*A=malloc(K); memset(A,1,K);                          /* all-ones activation */
    /* S weights: B_i[0*N+0]=i+1, rest 0 => C[0] = sum_k A[k]*B[k][0] = 1*(i+1) = i+1 (the ID) */
    ork_w**w=malloc(sizeof(ork_w*)*S);
    for(int i=0;i<S;i++){ int8_t*B=calloc((size_t)K*N,1); B[0*N+0]=(int8_t)(i+1); w[i]=ork_i8_mm_pack(c,K,N,B); free(B);
        if(!w[i]){printf("pack %d fail\n",i);return 1;} }
    int32_t*slot=(int32_t*)ork_dma_alloc(c,(size_t)N*sizeof(int32_t)); if(!slot){printf("dma_alloc fail\n");return 1;}
    ork_mm_task_i8*tk=malloc(sizeof(ork_mm_task_i8)*S);
    for(int i=0;i<S;i++){ tk[i].w=w[i]; tk[i].M=1; tk[i].A=A; tk[i].C=slot; }  /* ALL C -> same slot */
    gTK=tk; gS=S;

    /* warm + correctness (last op wins => slot[0]==S) */
    if(ork_i8_mm_run_chain(c,S,tk)){printf("warm rc!=0\n");return 1;}
    printf("  after chain, slot[0]=%d (expect %d = last ID): %s\n", slot[0], S, slot[0]==S?"PASS":"FAIL");

    /* seed sentinel, run on a thread, poll the ONE slot and log every distinct value seen */
    slot[0]=SENT; cvac(slot); __asm__ volatile("dsb ish":::"memory");
    int seq[256], nseq=0; int last=SENT; double t_first=-1;
    gDone=0; pthread_t th; double start=now_us();
    pthread_create(&th,0,chain_thr,0);
    double deadline=start+3e6;
    while(!gDone && now_us()<deadline){ civac(slot); int v=slot[0];
        if(v!=last){ if(nseq<256) seq[nseq++]=v; if(t_first<0&&v!=SENT)t_first=now_us()-start; last=v; } }
    civac(slot); if(slot[0]!=last && nseq<256) seq[nseq++]=slot[0];
    pthread_join(th,0);
    printf("  chain rc=%d | distinct values observed at the single slot (in order):\n   ", gRc);
    int inorder=1, prev=0, ids=0;
    for(int i=0;i<nseq;i++){ printf(" %d", seq[i]); if(seq[i]!=SENT){ if(seq[i]<prev)inorder=0; prev=seq[i]; ids++; } }
    printf("\n  ★ distinct IDs seen=%d/%d, in-order=%s (first at %.1fus)\n", ids, S, inorder?"YES":"NO", t_first);
    printf("  => static-ID doorbell %s (host reads current op ID from ONE address)\n",
           (ids>=S-1 && inorder)?"WORKS":"only shows endpoint (per-op writes not individually observable at one slot)");
    ork_npu_free(c); return 0;
}
