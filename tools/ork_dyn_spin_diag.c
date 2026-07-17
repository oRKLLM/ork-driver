/* ork_dyn_spin_diag — systematic instrumentation of the persistent no-op spin tail's residual race.
 * Each subtest uses a FRESH context (cold -> begin runs its warm pass) so warm-establishment is controlled,
 * and polls the real outputs DIRECTLY (dc civac) so we separate compute/coherency from the end()/teardown path.
 * A=all-1, B=all-1, K=512 => every real output element must equal K=512.
 *
 *  ST1 (Theory C: readback/teardown race): begin(spin), poll each real O[i][N-1] to LANDING (timestamp+value)
 *      with NO halt/teardown; report land count/time/value. If all land ==512 here, compute+coherency are fine
 *      and the 5/8 was the teardown/readback path.
 *  ST2 (compute correctness / short-count): aligned-seed ALL elements to SENT, begin(spin), poll to landing,
 *      classify EVERY element: unwritten(SENT) / correct(512) / wrong (+sample values, incl 512-n*32 pattern).
 *  ST3 (spin-length correlation): sweep reserve; correct-count vs spin length.
 * NOTE: never memset() an ork_dma_alloc buffer (device mem -> SIGBUS); seed with aligned int32 stores + cvac.
 *   make ork_dyn_spin_diag && sudo env ORK_MM_TIMEOUT=2500 timeout 120 ./ork_dyn_spin_diag
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#define SENT 0x7fffffff
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static inline void civac(volatile void*p){ __asm__ volatile("dc civac,%0"::"r"(p):"memory"); }
static inline void cvac (volatile void*p){ __asm__ volatile("dc cvac,%0" ::"r"(p):"memory"); }
static inline void dsb(void){ __asm__ volatile("dsb ish":::"memory"); }

/* aligned word-store seed of O[base..base+n) to SENT (NOT memset — device memory) */
static void seed(int32_t*O,size_t base,size_t n){ for(size_t j=0;j<n;j++){ O[base+j]=SENT; cvac(&O[base+j]); } dsb(); }

/* one fresh-context spin run. mode: 1=land-poll, 2=classify, 3=quiet(count). reserve via env. returns correct count. */
static int run(int mode,int reserve,int S,int K,int N){
    char rs[16]; snprintf(rs,sizeof rs,"%d",reserve); setenv("ORK_DYN_RESERVE",rs,1); setenv("ORK_DYN_SPIN","1",1);
    ork_npu*c=ork_npu_init(); if(!c){printf("  init fail\n");return -1;}
    int8_t*A=malloc(K); memset(A,1,K);
    int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    ork_w*w=ork_mm_pack_i8(c,K,N,B); free(B); if(!w){printf("  pack fail\n");ork_npu_free(c);return -1;}
    int32_t*O=(int32_t*)ork_dma_alloc(c,(size_t)S*N*4); if(!O){printf("  dma fail\n");ork_npu_free(c);return -1;}
    ork_mm_task_i8*tk=malloc(sizeof(*tk)*S);
    for(int i=0;i<S;i++){ tk[i].w=w; tk[i].M=1; tk[i].A=A; tk[i].C=O+(size_t)i*N; }
    if(mode==2) for(int i=0;i<S;i++) seed(O,(size_t)i*N,N); else for(int i=0;i<S;i++) seed(O,(size_t)i*N+(N-1),1);

    uint64_t rw0=ork_npu_dma_rw(c);
    ork_dyn_chain*h=ork_dyn_begin(c,S,tk);   /* cold: begin runs its warm pass, then spin real submit */
    if(!h){ printf("  begin fail\n"); free(A); free(tk); ork_npu_free(c); return -1; }

    /* poll each row's last col to landing (NO teardown yet) */
    double t0=now_us(); double lt[64]={0}; int lv[64]; int landed=0;
    for(int i=0;i<S;i++) lv[i]=SENT;
    while(landed<S && now_us()-t0<15000.0){
        for(int i=0;i<S;i++){ volatile int32_t*d=(volatile int32_t*)(O+(size_t)i*N+(N-1)); civac(d);
            if(*d!=SENT && lt[i]==0){ lt[i]=now_us()-t0; lv[i]=*d; landed++; } } }
    uint64_t rw1=ork_npu_dma_rw(c);
    int correct=0; for(int i=0;i<S;i++) if(lv[i]==K) correct++;
    (void)rw0;(void)rw1;
    if(landed<S || mode==1){   /* chain-aware dump: on any anomaly (names the stuck descriptor), and always in ST1 to show the format */
        ork_dyn_dump(h, landed<S?"anomaly":"ST1-healthy-demo");
    }

    if(mode==1){
        printf("  ST1 land-poll (no teardown): landed=%d/%d correct=%d/%d\n",landed,S,correct,S);
        for(int i=0;i<S;i++) printf("    row%d: %s @%.0fus val=%d %s\n", i,
            lt[i]?"landed":"NEVER", lt[i], lv[i], lv[i]==K?"OK":(lv[i]==SENT?"(unwritten)":"WRONG"));
    } else if(mode==2){
        int unwr=0,ok=0,wrong=0; int samp[8]; int ns=0;
        for(int i=0;i<S;i++) for(int j=0;j<N;j++){ volatile int32_t*d=(volatile int32_t*)(O+(size_t)i*N+j); civac(d);
            int v=*d; if(v==SENT)unwr++; else if(v==K)ok++; else { wrong++; if(ns<8){samp[ns++]=v;} } }
        printf("  ST2 classify all %d elems: correct=%d unwritten=%d wrong=%d | landed-rows=%d/%d\n",S*N,ok,unwr,wrong,landed,S);
        if(ns){ printf("    wrong samples:"); for(int i=0;i<ns;i++){ int d=K-samp[i]; printf(" %d(512-%d=%s%d*32)",samp[i],d, d%32?"~":"", d/32); } printf("\n"); }
    } else if(mode==3){
        printf("  ST3 reserve=%d: landed=%d/%d correct=%d/%d\n",reserve,landed,S,correct,S);
    }   /* mode 4: quiet, just return correct (for the stability loop) */
    ork_dyn_end(h); free(A); free(tk); ork_npu_free(c);
    return correct;
}

static int poll_landed(int32_t*O,int S,int N,double budget_us){
    double t0=now_us(); int done[64]={0}, landed=0;
    while(landed<S && now_us()-t0<budget_us){ for(int i=0;i<S;i++){ volatile int32_t*d=(volatile int32_t*)(O+(size_t)i*N+(N-1)); civac(d); if(*d!=SENT && !done[i]){done[i]=1;landed++;} } }
    return landed;
}
/* One COLD round (fresh context) + recover-retry on a dispatch miss. Returns: 0=cold OK first try,
 * 1=missed then RESCUED by recover(reset+POST)+warm-retry, 2=missed and NOT rescued, -1=setup error. */
static int run_cold(int S,int K,int N){
    setenv("ORK_DYN_SPIN","1",1); setenv("ORK_DYN_RESERVE","32",1);
    ork_npu*c=ork_npu_init(); if(!c) return -1;
    int8_t*A=malloc(K); memset(A,1,K); int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    ork_w*w=ork_mm_pack_i8(c,K,N,B); free(B);
    int32_t*O=(int32_t*)ork_dma_alloc(c,(size_t)S*N*4);
    ork_mm_task_i8*tk=malloc(sizeof(*tk)*S); for(int i=0;i<S;i++){ tk[i].w=w; tk[i].M=1; tk[i].A=A; tk[i].C=O+(size_t)i*N; }
    for(int i=0;i<S;i++) seed(O,(size_t)i*N+(N-1),1);
    uint64_t rw0=ork_npu_dma_rw(c);
    ork_dyn_chain*h=ork_dyn_begin(c,S,tk);
    int landed = h ? poll_landed(O,S,N,8000.0) : 0;
    uint64_t rw1=ork_npu_dma_rw(c);
    if(h) ork_dyn_end(h);
    int res;
    if(landed==S){ res=0; }
    else {
        fprintf(stderr,"  COLD MISS: landed=%d/%d rw_delta=%llu -> recover+retry\n",landed,S,(unsigned long long)(rw1-rw0));
        if(!ork_npu_recover(c,"cold-miss")){ res=2; }
        else { for(int i=0;i<S;i++) seed(O,(size_t)i*N+(N-1),1);
               ork_dyn_chain*h2=ork_dyn_begin(c,S,tk); int l2=h2?poll_landed(O,S,N,8000.0):0; if(h2) ork_dyn_end(h2);
               res=(l2==S)?1:2; fprintf(stderr,"  retry-after-recover: landed=%d/%d -> %s\n",l2,S, l2==S?"RESCUED":"STILL MISSING"); }
    }
    free(A); free(tk); ork_npu_free(c); return res;
}

int main(void){
    int S=8,K=512,N=512;
    setvbuf(stdout,0,_IONBF,0); setvbuf(stderr,0,_IONBF,0);
    printf("ork_dyn_spin_diag: S=%d K=%d N=%d (A=B=1 => each real output elem must = %d)\n",S,K,N,K);
    printf("== ST0: POST sanity — ork_npu_recover (dump+reset+POST) must PASS on a healthy NPU ==\n");
    { ork_npu*c=ork_npu_init(); if(c){ int r=ork_npu_recover(c,"sanity"); printf("  ST0: recover->%s (%s)\n", r?"PASS":"FAIL", r?"POST trustworthy":"POST itself broken — ST5 fault path unreliable"); ork_npu_free(c); } }
    printf("== ST1: does each real output LAND correctly before any teardown? (Theory C) ==\n");
    run(1,32,S,K,N);
    printf("== ST2: classify every output element (compute correctness / short-count) ==\n");
    run(2,32,S,K,N);
    printf("== ST3: correctness vs spin length (reserve sweep) ==\n");
    int rr[]={9,12,16,24,32,64}; for(int i=0;i<6;i++) run(3,rr[i],S,K,N);

    printf("== ST5: persistent-context stability + self-heal (detect->dump->reset->dummy->continue/fault) ==\n");
    setenv("ORK_DYN_SPIN","1",1); setenv("ORK_DYN_RESERVE","32",1);
    { ork_npu*c=ork_npu_init();
      if(!c){ printf("  init fail\n"); return 0; }
      int8_t*A=malloc(K); memset(A,1,K); int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
      ork_w*w=ork_mm_pack_i8(c,K,N,B); free(B);
      int32_t*O=(int32_t*)ork_dma_alloc(c,(size_t)S*N*4);
      ork_mm_task_i8*tk=malloc(sizeof(*tk)*S); for(int i=0;i<S;i++){ tk[i].w=w; tk[i].M=1; tk[i].A=A; tk[i].C=O+(size_t)i*N; }
      int R=20, clean=0, recov=0, faulted=0;
      for(int k=0;k<R && !faulted;k++){
          for(int i=0;i<S;i++) seed(O,(size_t)i*N+(N-1),1);
          uint64_t rw0=ork_npu_dma_rw(c);
          ork_dyn_chain*h=ork_dyn_begin(c,S,tk);
          if(!h){ fprintf(stderr,"  round %d: begin NULL\n",k); if(!ork_npu_recover(c,"begin-null")) faulted=1; continue; }
          double t0=now_us(); int done[64]={0}, landed=0;
          while(landed<S && now_us()-t0<8000.0){ for(int i=0;i<S;i++){ volatile int32_t*d=(volatile int32_t*)(O+(size_t)i*N+(N-1)); civac(d); if(*d!=SENT && !done[i]){done[i]=1;landed++;} } }
          uint64_t rw1=ork_npu_dma_rw(c);
          int correct=0; for(int i=0;i<S;i++){ volatile int32_t*d=(volatile int32_t*)(O+(size_t)i*N+(N-1)); civac(d); if(*d==K)correct++; }
          ork_dyn_end(h);
          if(correct==S){ clean++; }
          else { fprintf(stderr,"  round %d ANOMALY: landed=%d correct=%d rw_delta=%llu -> recover\n",k,landed,correct,(unsigned long long)(rw1-rw0));
                 if(ork_npu_recover(c,"st5-anomaly")) recov++; else { faulted=1; fprintf(stderr,"  >>> FAULT at round %d (NPU unrecoverable)\n",k); } }
          struct timespec nap={0,30*1000*1000}; nanosleep(&nap,0);
      }
      printf("  ST5 over %d persistent rounds: clean=%d recovered=%d faulted=%d\n", R, clean, recov, faulted);
      free(A); free(tk); ork_npu_free(c);
    }

    printf("== ST6: COLD-dispatch-miss rate + is it recoverable? (fresh context each round, recover+retry on miss) ==\n");
    { int R=30; { const char*e=getenv("ORK_ST6_ROUNDS"); if(e) R=atoi(e); } int coldok=0, rescued=0, notresc=0, err=0;
      for(int k=0;k<R;k++){ int r=run_cold(S,K,N);
          if(r==0)coldok++; else if(r==1)rescued++; else if(r==2)notresc++; else err++;
          struct timespec nap={0,20*1000*1000}; nanosleep(&nap,0); }
      printf("  ST6 over %d COLD rounds: first-try-OK=%d  missed-then-RESCUED=%d  missed-NOT-rescued=%d  err=%d\n",
             R, coldok, rescued, notresc, err);
      printf("  => cold-miss rate=%d/%d; recover(reset+POST)+retry rescued %d/%d misses\n",
             rescued+notresc, R, rescued, rescued+notresc);
    }
    printf("done.\n");
    return 0;
}
