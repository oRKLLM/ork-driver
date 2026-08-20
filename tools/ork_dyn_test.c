/* ork_dyn_test — exercise the dynamic-submit API (ork_dyn_begin/progress/halt/end).
 *  A) full run: begin, end (no halt) -> all S outputs computed (==K).
 *  B) early-exit: begin, poll progress, halt(H), end -> outputs 0..~H computed, rest untouched (NPU freed early).
 *   make ork_dyn_test && sudo ./ork_dyn_test [S=16] [H=8]
 * (NPU op; the halt leaves a partial kernel job — run alone; reboot if a later NPU op misbehaves.)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static inline void civac(volatile void*p){ __asm__ volatile("dc civac,%0"::"r"(p):"memory"); }
#define SENT 0x7fffffff

int main(int argc,char**argv){
    int S=argc>1?atoi(argv[1]):16, H=argc>2?atoi(argv[2]):8, K=512, N=512;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    printf("ork_dyn_test: S=%d ops (K=%d,N=%d), halt at H=%d\n",S,K,N,H);
    int8_t*A=(int8_t*)malloc(K); memset(A,1,K);   /* malloc (NOT ork_dma_alloc): zero-copy DMA-A miscomputes at M=1 */
    int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    ork_w*w=ork_i8_mm_pack(c,K,N,B); if(!w){printf("pack fail\n");return 1;}
    int32_t*O=(int32_t*)ork_dma_alloc(c,(size_t)S*N*sizeof(int32_t)); if(!O){printf("dma_alloc fail\n");return 1;}
    ork_mm_task_i8*tk=malloc(sizeof(ork_mm_task_i8)*S);
    for(int i=0;i<S;i++){ tk[i].w=w; tk[i].M=1; tk[i].A=A; tk[i].C=O+(size_t)i*N; }
    int nwritten(void){ int n=0; for(int i=0;i<S;i++){ volatile int32_t*d=(volatile int32_t*)(O+(size_t)i*N+(N-1)); civac((void*)d); if(*d==K)n++; } return n; }

    printf("  budget: ork_dyn_max_steps=%d (this chain S=%d)\n", ork_dyn_max_steps(), S);

    /* A) full run — no halt */
    ork_dyn_chain*h=ork_dyn_begin(c,S,tk); if(!h){printf("A: begin failed\n");return 1;}
    int done=ork_dyn_end(h);
    int na=nwritten();
    printf("  A full: end highest=%d, outputs==K: %d/%d -> %s\n", done, na, S, na==S?"PASS":"FAIL");

    /* D) M>1 begin_mc: partition S tasks (M rows each) across cores via the NON-BLOCKING doorbell; verify
     *    EVERY row's last col == K (A=B=1 -> each dot = K). The doorbell only polls the last element (row
     *    M-1), so checking all M rows validates the M-tile write-order assumption (last element written last). */
    { int MM=4;
      int8_t*Am=(int8_t*)malloc((size_t)MM*K); memset(Am,1,(size_t)MM*K);
      int32_t*Om=(int32_t*)ork_dma_alloc(c,(size_t)S*MM*N*sizeof(int32_t));
      ork_mm_task_i8*tm=malloc(sizeof(ork_mm_task_i8)*S);
      for(int i=0;i<S;i++){ tm[i].w=w; tm[i].M=MM; tm[i].A=Am; tm[i].C=Om+(size_t)i*MM*N; }
      ork_dyn_chain*hm=Om?ork_dyn_begin_mc(c,S,tm,0):NULL;
      if(hm){ ork_dyn_end(hm); int okm=0;
        for(int i=0;i<S;i++){ int allk=1; for(int m=0;m<MM;m++){ volatile int32_t*d=(volatile int32_t*)(Om+(size_t)i*MM*N+(size_t)m*N+(N-1)); civac((void*)d); if(*d!=K){allk=0;break;} } if(allk)okm++; }
        printf("  D M>1 begin_mc: M=%d, %d/%d tasks all-rows==K -> %s\n", MM, okm, S, okm==S?"PASS":"FAIL"); }
      else printf("  D M>1 begin_mc: begin_mc NULL (ineligible / per-core scratch too small) -> CHECK\n");
      free(Am); if(Om) ork_dma_free(c,Om); free(tm); }

    /* E) M>1 fp16 begin_mc: fp16 A·B -> fp32 C, verify each output element == (float)Kf (A=B=1.0 -> dot=Kf).
     *    Validates the fp16 doorbell generalization (synth, 2-byte activation, fp32 4-byte output).
     *    A is HOST (malloc) memory — NOT ork_dma_alloc: like D (int8), the doorbell stages A via memcpy and
     *    a zero-copy DMA-A source's CPU writes are not coherently readable by that staged read (partial-K
     *    sums). Feeding ork_dma_alloc A here was the sole cause of the old fp16-doorbell "flakiness". */
    { int MM=getenv("ORK_E_M")?atoi(getenv("ORK_E_M")):2, Kf=getenv("ORK_E_K")?atoi(getenv("ORK_E_K")):512, Nf=getenv("ORK_E_N")?atoi(getenv("ORK_E_N")):256; ork_f16 one=(ork_f16)1.0f;
      ork_f16*Bf=(ork_f16*)malloc((size_t)Kf*Nf*sizeof(ork_f16)); for(size_t i=0;i<(size_t)Kf*Nf;i++) Bf[i]=one;
      ork_w*wf=ork_f16_mm_pack(c,Kf,Nf,Bf);
      ork_f16*Af=wf?(ork_f16*)malloc((size_t)MM*Kf*sizeof(ork_f16)):NULL; if(Af) for(int i=0;i<MM*Kf;i++) Af[i]=one;   /* host A */
      float*Of=Af?(float*)ork_dma_alloc(c,(size_t)S*MM*Nf*sizeof(float)):NULL;   /* C = resident DMA (zero-copy direct output) */
      ork_mm_task_i8*tf=malloc(sizeof(ork_mm_task_i8)*S);
      for(int i=0;i<S;i++){ tf[i].w=wf; tf[i].M=MM; tf[i].A=(const int8_t*)Af; tf[i].C=(int32_t*)(Of+(size_t)i*MM*Nf); }
      ork_dyn_chain*hf=Of?ork_dyn_begin_mc(c,S,tf,0):NULL;
      if(hf){ ork_dyn_end(hf); int okf=0;
        for(int i=0;i<S;i++){ int allk=1; for(int m=0;m<MM;m++){ volatile float*d=(volatile float*)(Of+(size_t)i*MM*Nf+(size_t)m*Nf+(Nf-1)); civac((void*)d); if(*d<(float)Kf-2||*d>(float)Kf+2){allk=0;break;} } if(allk)okf++; }
        printf("  E M>1 fp16 begin_mc: M=%d, %d/%d tasks all-rows==%d.0 -> %s\n", MM, okf, S, Kf, okf==S?"PASS":"FAIL"); }
      else printf("  E M>1 fp16 begin_mc: begin_mc NULL (ineligible / pack fail) -> CHECK\n");
      free(Bf); if(Af)free(Af); if(Of)ork_dma_free(c,Of); free(tf); }

    /* B/C exercise ork_dyn_halt/append which leave a partial NPU job (can wedge the board on a lost race) —
     * default run stops here (A+D+E are clean begin/end). Set ORK_DYN_TEST_HALT=1 to run them. */
    if(!getenv("ORK_DYN_TEST_HALT")){ printf("  (B/C halt/append skipped; ORK_DYN_TEST_HALT=1 to run)\n"); ork_npu_free(c); return 0; }

    /* B) early-exit — halt at H once progress passes a couple ops */
    h=ork_dyn_begin(c,S,tk); if(!h){printf("B: begin failed\n");return 1;}
    double t0=now_us(); int p=-1;
    while((p=ork_dyn_progress(h))<2 && now_us()-t0<1e6) ;   /* wait until a couple ops in */
    int hr=ork_dyn_halt(h,H);
    int hd=ork_dyn_end(h);
    int nw=nwritten();
    printf("  B early-exit: halt(%d) rc=%d, progress-at-halt=%d, end highest=%d, outputs==K=%d/%d\n", H, hr, p, hd, nw, S);
    printf("  ★ early-exit %s (outputs ~%d..%d done, rest freed): nw=%d in [%d,%d] => %s\n",
           (nw<S && nw>=H)?"WORKED":"CHECK", 0, H, nw, H, H+3,
           (nw<S && nw>=H-1 && nw<=H+3)?"PASS (NPU freed early)":(nw==S?"ran-to-end (halt too late / lead too short)":"AMBIGUOUS"));

    /* C) UNINTERRUPTED WRAP (EXPERIMENTAL, opt-in): in-flight append races the kernel job model and can ABORT
     *    the NPU (soft-reset) on a lost race — see ork_dyn_append. Gated so default runs stay board-safe. */
    if(!getenv("ORK_DYN_TEST_APPEND")){ printf("  C wrap: skipped (set ORK_DYN_TEST_APPEND=1; experimental, can wedge)\n"); ork_npu_free(c); return 0; }
    for(int i=0;i<S;i++){ volatile int32_t*d=(volatile int32_t*)(O+(size_t)i*N+(N-1)); *d=SENT; civac((void*)d);} __asm__ volatile("dsb ish":::"memory");
    int START=S<8?2:4;
    { char rs[16]; snprintf(rs,sizeof rs,"%d",S); setenv("ORK_DYN_RESERVE",rs,1); }   /* reserve budget = full S so append can extend in-flight */
    h=ork_dyn_begin(c,START,tk); if(!h){printf("C: begin failed\n");return 1;}
    unsetenv("ORK_DYN_RESERVE");
    int appended=0, toolate=0;
    for(int i=START;i<S;i++){ int r,tries=0; while((r=ork_dyn_append(h,&tk[i]))==1 && tries++<200000) ;
        if(r==0) appended++; else { toolate=1; break; } }
    int cd=ork_dyn_end(h);
    int ncw=nwritten();
    printf("  C wrap: began %d, appended %d in-flight (total %d), end highest=%d, outputs==K %d/%d%s\n",
           START, appended, START+appended, cd, ncw, S, toolate?" [race lost -> planned break]":"");
    printf("  ★ uninterrupted wrap => %s\n", ncw==S?"PASS (no break: chain extended in-flight)":(toolate?"PARTIAL (lost race; needs more headroom / faster fill)":"FAIL"));
    ork_npu_free(c); return 0;
}
