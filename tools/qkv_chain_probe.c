/* qkv_chain_probe — extend the static regcmd chain to the QKV GROUP: Q/K/V are 3 INDEPENDENT matmuls sharing
 * the same input A (unlike FFN's dependent gate->silu->... chain). Run them as ONE submit via run_chain_i8
 * (3 matmul tasks, all reading A) and compare to 3 separate ork_i8_mm_run submits. GQA dims (Q wide, K/V narrow).
 * All-ones -> every output == K (bit-exact gate). Win = 1 submit vs 3 (fewer round-trips).
 *   make qkv_chain_probe && sudo env ORK_MM_TIMEOUT=4000 ./qkv_chain_probe [iters] [M]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

int main(int argc,char**argv){
    int ITER=argc>1?atoi(argv[1]):50, M=argc>2?atoi(argv[2]):1;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    const int K=3584, Nq=3584, Nkv=512;                 /* GQA: 28 q-heads, 4 kv-heads, head_dim 128 */
    int8_t*A=malloc((size_t)M*K); memset(A,1,(size_t)M*K);
    int8_t*Wq=malloc((size_t)K*Nq), *Wk=malloc((size_t)K*Nkv), *Wv=malloc((size_t)K*Nkv);
    memset(Wq,1,(size_t)K*Nq); memset(Wk,1,(size_t)K*Nkv); memset(Wv,1,(size_t)K*Nkv);
    ork_npu_set_pack_domain(c,0);
    ork_w *wq=ork_i8_mm_pack(c,K,Nq,Wq), *wk=ork_i8_mm_pack(c,K,Nkv,Wk), *wv=ork_i8_mm_pack(c,K,Nkv,Wv);
    if(!wq||!wk||!wv){printf("pack failed\n");return 1;}
    int32_t*Cq=malloc((size_t)M*Nq*4), *Ck=malloc((size_t)M*Nkv*4), *Cv=malloc((size_t)M*Nkv*4);

    ork_mm_task_i8 t[3] = { {wq,M,A,Cq}, {wk,M,A,Ck}, {wv,M,A,Cv} };
    printf("qkv_chain_probe: Q[%d,%d] K/V[%d,%d]  M=%d  (K=%d)\n",K,Nq,K,Nkv,M,K);

    /* ---- CHAIN: Q,K,V in ONE submit ---- */
    int r=ork_i8_mm_run_chain(c,3,t);
    if(r){ printf("  chain rc=%d (%s) — QKV chain did NOT run\n", r, r==-1?"WEDGE":"dims/err"); ork_npu_free(c); return 1; }
    int okc=1; for(int i=0;i<M*Nq;i++)if(Cq[i]!=K){okc=0;break;}
    for(int i=0;i<M*Nkv&&okc;i++)if(Ck[i]!=K||Cv[i]!=K){okc=0;break;}
    double t0=now_us(); for(int i=0;i<ITER;i++) ork_i8_mm_run_chain(c,3,t); double ch=(now_us()-t0)/ITER;
    printf("  CHAIN (Q,K,V ONE submit): %8.1f us  bit-exact=%s\n", ch, okc?"YES":"NO");

    /* ---- SEPARATE: 3 submits ---- */
    ork_i8_mm_run(c,wq,M,A,Cq); ork_i8_mm_run(c,wk,M,A,Ck); ork_i8_mm_run(c,wv,M,A,Cv);
    int oks=1; for(int i=0;i<M*Nq;i++)if(Cq[i]!=K){oks=0;break;}
    t0=now_us(); for(int i=0;i<ITER;i++){ ork_i8_mm_run(c,wq,M,A,Cq); ork_i8_mm_run(c,wk,M,A,Ck); ork_i8_mm_run(c,wv,M,A,Cv); } double sep=(now_us()-t0)/ITER;
    printf("  SEPARATE (3 submits, each multi-core): %8.1f us  bit-exact=%s\n", sep, oks?"YES":"NO");

    /* ---- STREAM: round-robin Q/K/V across the 3 cores (concurrent, one dispatch) ---- */
    int rs=ork_i8_mm_run_stream(c,3,t);
    if(!rs){
        int okr=1; for(int i=0;i<M*Nq&&okr;i++)if(Cq[i]!=K)okr=0; for(int i=0;i<M*Nkv&&okr;i++)if(Ck[i]!=K||Cv[i]!=K)okr=0;
        t0=now_us(); for(int i=0;i<ITER;i++) ork_i8_mm_run_stream(c,3,t); double st=(now_us()-t0)/ITER;
        printf("  STREAM (round-robin 3 cores):          %8.1f us  bit-exact=%s\n", st, okr?"YES":"NO");
        printf("  ★ vs separate: chain %.2fx | stream %.2fx  (>1 = faster than 3 separate multi-core submits)\n",
               ch>0?sep/ch:0, st>0?sep/st:0);
    } else printf("  STREAM rc=%d (%s)\n", rs, rs==-1?"WEDGE":"dims/err");

    /* ---- GROUP: concatenate [Wq|Wk|Wv] -> ONE wide multi-core matmul (1 submit, full multi-core) ---- */
    int Ncat=Nq+2*Nkv;                                  /* 3584+512+512 = 4608 */
    int8_t*Wcat=malloc((size_t)K*Ncat);
    for(int k=0;k<K;k++){ int8_t*row=Wcat+(size_t)k*Ncat;
        memcpy(row,          Wq+(size_t)k*Nq,  Nq);
        memcpy(row+Nq,       Wk+(size_t)k*Nkv, Nkv);
        memcpy(row+Nq+Nkv,   Wv+(size_t)k*Nkv, Nkv); }
    ork_npu_set_pack_domain(c,0); ork_w*wcat=ork_i8_mm_pack(c,K,Ncat,Wcat);
    if(wcat){ int32_t*Cc=malloc((size_t)M*Ncat*4);
        ork_i8_mm_run(c,wcat,M,A,Cc);
        int okg=1; for(int i=0;i<M*Ncat&&okg;i++)if(Cc[i]!=K)okg=0;
        t0=now_us(); for(int i=0;i<ITER;i++) ork_i8_mm_run(c,wcat,M,A,Cc); double gr=(now_us()-t0)/ITER;
        printf("  GROUP (concat [Q|K|V], 1 multi-core submit): %8.1f us  bit-exact=%s\n", gr, okg?"YES":"NO");
        printf("  ★ vs separate: group %.2fx  (>1 = faster than 3 separate)\n", gr>0?sep/gr:0);
    } else printf("  GROUP pack failed\n");
    ork_npu_free(c); return okc?0:2;
}
