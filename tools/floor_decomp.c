/* tools/floor_decomp.c — decompose the RK3588 NPU per-submit "167µs floor" into its components, and
 * test the headroom hypotheses (B poll-granularity, D 3-core concurrency, chain amortization).
 *
 * The new lever: the kernel fills rknpu_submit.hw_elapse_time (its OWN NPU-busy view, in NANOSECONDS)
 * but ork never read it. ork_npu_floor_timing() exposes it + the wall time inside the blocking SUBMIT
 * ioctl. Per call we split:
 *     run_i8 wall  =  host (regcmd synth + bsync + copy)  +  ioctl wall
 *     ioctl wall   =  hw_elapse (real NPU-busy)  +  dispatch/poll-exit  ( = ioctl - hw )
 *
 *   make floor_decomp && sudo ./floor_decomp [iters]      (board only)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec*1e-3; }

int main(int argc, char **argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int iters = (argc>1)? atoi(argv[1]) : 200; if(iters<1) iters=200;
    ork_npu *c=ork_npu_init();
    if(!c){ printf("no board / no NPU\n"); return 0; }

    /* ---------------- (1) single-submit decomposition sweep ---------------- */
    printf("=== (1) FLOOR DECOMPOSITION (int8, single-core, %d iters/shape) ===\n", iters);
    printf("all us/call. host=wall-ioctl (synth+bsync+copy). hw=NPU-busy (hw_elapse, ns->us).\n");
    printf("disp=ioctl-hw (kernel dispatch + poll-exit latency).  nsub=submits/call.\n\n");
    printf("%-14s %8s %8s %8s %8s %7s %5s %6s\n","M,K,N","wall","ioctl","host","hw","disp","nsub","hw%%");
    struct { int M,K,N; } sh[] = {
        {64,512,64},{64,512,256},{64,2048,64},{64,2048,256},
        {512,512,64},{512,2048,256},{8,512,64},{1,512,64},
    };
    for(unsigned s=0;s<sizeof(sh)/sizeof(sh[0]);s++){
        int M=sh[s].M,K=sh[s].K,N=sh[s].N;
        signed char *A=malloc((size_t)M*K),*W=malloc((size_t)K*N); int32_t *C=malloc((size_t)M*N*4);
        for(int i=0;i<M*K;i++)A[i]=1; for(int i=0;i<K*N;i++)W[i]=1;
        ork_w *w=ork_mm_pack_i8(c,K,N,W); if(!w){printf("%-14s pack fail\n","x");free(A);free(W);free(C);continue;}
        ork_mm_run_i8(c,w,M,A,C);
        ork_npu_floor_reset();
        double b0=now_us(); for(int it=0;it<iters;it++) ork_mm_run_i8(c,w,M,A,C); double wall=(now_us()-b0)/iters;
        double io,hw; long long hwl; long n; ork_npu_floor_timing(&io,&hw,&hwl,&n);
        double iou=io/n, hwu=(hw/1000.0)/n, host=wall-iou*(n/iters), disp=iou-hwu; /* iou,hwu per-submit */
        double per_call_io=io/iters, per_call_hw=(hw/1000.0)/iters;
        char lbl[24]; snprintf(lbl,sizeof lbl,"%d,%d,%d",M,K,N);
        printf("%-14s %8.1f %8.1f %8.1f %8.1f %7.1f %5ld %5.0f%%\n",
               lbl, wall, per_call_io, wall-per_call_io, per_call_hw, per_call_io-per_call_hw, n/iters,
               100.0*per_call_hw/wall);
        free(A);free(W);free(C);
    }

    /* ---------------- (2) chain amortization: is chain hw ≈ N × single hw? ---------------- */
    printf("\n=== (2) CHAIN DECOMPOSITION (M=64,K=512,N=64; 1 ioctl of N PC-chained tasks) ===\n");
    printf("if chain hw ≈ N×(single hw), the NPU re-executes each task (no HW amortization).\n");
    printf("host is amortized once across the chain (only ~1 synth+bsync set).\n\n");
    printf("%-4s %9s %9s %9s %9s %10s %10s\n","N","wall","ioctl","host","hw","hw/task","host/task");
    { const int M=64,K=512,N=64;
      signed char *A=malloc((size_t)M*K),*W=malloc((size_t)K*N);
      for(int i=0;i<M*K;i++)A[i]=1; for(int i=0;i<K*N;i++)W[i]=1;
      ork_w *w=ork_mm_pack_i8(c,K,N,W);
      static int32_t CC[8][64*64];
      int Ns[4]={1,2,4,8};
      for(int q=0;q<4;q++){ int NT=Ns[q];
        ork_mm_task_i8 mt[8]; for(int j=0;j<NT;j++){mt[j].w=w;mt[j].M=M;mt[j].A=A;mt[j].C=CC[j];}
        int rc = (NT==1)? ork_mm_run_i8(c,w,M,A,CC[0]) : ork_mm_run_chain_i8(c,NT,mt);
        if(rc){printf("N=%d rc=%d\n",NT,rc);continue;}
        ork_npu_floor_reset();
        double b0=now_us();
        for(int it=0;it<iters;it++){ if(NT==1) ork_mm_run_i8(c,w,M,A,CC[0]); else ork_mm_run_chain_i8(c,NT,mt); }
        double wall=(now_us()-b0)/iters;
        double io,hw; long long hwl; long n; ork_npu_floor_timing(&io,&hw,&hwl,&n);
        double per_call_io=io/iters, per_call_hw=(hw/1000.0)/iters;
        printf("%-4d %9.1f %9.1f %9.1f %9.1f %10.1f %10.1f\n",
               NT, wall, per_call_io, wall-per_call_io, per_call_hw, per_call_hw/NT, (wall-per_call_io)/NT);
      }
      free(A);free(W);
    }

    /* ---------------- (3) 3-core concurrency: run_stream S independent tasks ---------------- */
    printf("\n=== (3) 3-CORE CONCURRENCY (S independent M=64,K=512,N=64 matmuls) ===\n");
    printf("serial = S× run_i8 (1 core). stream = ork_mm_run_stream_i8 (round-robin 3 cores).\n");
    printf("if stream ≈ serial/3, independent tasks truly overlap across cores.\n\n");
    printf("%-4s %12s %12s %9s %12s\n","S","serial_us","stream_us","speedup","stream/S");
    { const int M=64,K=512,N=64;
      signed char *A=malloc((size_t)M*K),*W=malloc((size_t)K*N);
      for(int i=0;i<M*K;i++)A[i]=1; for(int i=0;i<K*N;i++)W[i]=1;
      /* distinct resident weights per task (independent) */
      ork_w *ws[12]; for(int j=0;j<12;j++) ws[j]=ork_mm_pack_i8(c,K,N,W);
      static int32_t CC[12][64*64];
      int Ss[4]={3,6,9,12};
      for(int q=0;q<4;q++){ int S=Ss[q];
        ork_mm_task_i8 mt[12]; for(int j=0;j<S;j++){mt[j].w=ws[j];mt[j].M=M;mt[j].A=A;mt[j].C=CC[j];}
        /* serial */
        for(int j=0;j<S;j++) ork_mm_run_i8(c,ws[j],M,A,CC[j]);   /* warm */
        double s0=now_us(); for(int it=0;it<iters;it++) for(int j=0;j<S;j++) ork_mm_run_i8(c,ws[j],M,A,CC[j]);
        double serial=(now_us()-s0)/iters;
        /* stream (cross-core) */
        int rc=ork_mm_run_stream_i8(c,S,mt);   /* warm */
        if(rc){printf("S=%d stream rc=%d\n",S,rc);continue;}
        double t0=now_us(); for(int it=0;it<iters;it++) ork_mm_run_stream_i8(c,S,mt);
        double stream=(now_us()-t0)/iters;
        /* correctness: all-ones int8 A·W over K=512 => every C element == 512 */
        int ok=1; for(int j=0;j<S&&ok;j++) for(int e=0;e<M*N;e++) if(CC[j][e]!=K){ok=0;break;}
        printf("%-4d %12.1f %12.1f %8.2fx %12.1f   %s\n", S, serial, stream, serial/stream, stream/S, ok?"OK":"BAD!");
      }
      free(A);free(W);
    }

    /* ---------------- (4) chain with RESIDENT DMA A/C: does host drop toward NPU floor? ---------------- */
    printf("\n=== (4) DMA-RESIDENT CHAIN (M=64,K=512,N=64; A,C in ork_dma_alloc buffers) ===\n");
    printf("chain's per-task bcreate/bdestroy vanishes when dma_find hits -> host should approach ioctl.\n\n");
    printf("%-4s %9s %9s %9s %9s %10s %6s\n","N","wall","ioctl","host","hw","hw/task","ok");
    { const int M=64,K=512,N=64;
      signed char *W=malloc((size_t)K*N); for(int i=0;i<K*N;i++)W[i]=1;
      ork_w *w=ork_mm_pack_i8(c,K,N,W);
      signed char *A = ork_dma_alloc(c, (size_t)M*K);
      int32_t *Cd[32]; for(int j=0;j<32;j++) Cd[j]=ork_dma_alloc(c,(size_t)M*N*4);
      if(!A||!Cd[0]){ printf("dma_alloc failed (no dma-heap?) — skipping\n"); }
      else {
        for(int i=0;i<M*K;i++) A[i]=1;
        int Ns[6]={1,2,4,8,16,32};
        for(int q=0;q<6;q++){ int NT=Ns[q];
          ork_mm_task_i8 mt[32]; for(int j=0;j<NT;j++){mt[j].w=w;mt[j].M=M;mt[j].A=A;mt[j].C=Cd[j];}
          int rc=(NT==1)?ork_mm_run_i8(c,w,M,A,Cd[0]):ork_mm_run_chain_i8(c,NT,mt);
          if(rc){printf("N=%d rc=%d\n",NT,rc);continue;}
          ork_npu_floor_reset();
          double b0=now_us();
          for(int it=0;it<iters;it++){ if(NT==1) ork_mm_run_i8(c,w,M,A,Cd[0]); else ork_mm_run_chain_i8(c,NT,mt); }
          double wall=(now_us()-b0)/iters;
          double io,hw; long long hwl; long n; ork_npu_floor_timing(&io,&hw,&hwl,&n);
          double pio=io/iters, phw=(hw/1000.0)/iters;
          int ok=1,nz=0,nk=0; for(int j=0;j<NT;j++) for(int e=0;e<M*N;e++){ if(Cd[j][e]!=K)ok=0; if(Cd[j][e]==0)nz++; if(Cd[j][e]==K)nk++; }
          printf("%-4d %9.1f %9.1f %9.1f %9.1f %10.1f %5s (Cd[0][0]=%d exp=%d; zeros=%d/%d correct=%d)\n",
                 NT, wall, pio, wall-pio, phw, phw/NT, ok?"OK":"BAD", Cd[0][0], K, nz, NT*M*N, nk);
        }
        ork_dma_free(c,A); for(int j=0;j<32;j++) ork_dma_free(c,Cd[j]);
      }
      free(W);
    }

    printf("\nDONE.\n");
    ork_npu_free(c);
    return 0;
}
