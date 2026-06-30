/* Apples-to-apples per-matmul submit cost: the SAME int8 matmul (M,K,N) via
 * ork-driver (open regcmd) vs librknnrt (closed RKNN matmul API). int8->int32,
 * B packed/uploaded once, timed over N warm iterations. RKNN matmul API is
 * single-core (rejects core mask 7), so the clean comparison pins ork to 1 core;
 * ork 3-core is shown too (what oRKLLM actually uses). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"
#include "rknn_matmul_api.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }

static double bench_ork(int M,int K,int N,int iters,int cores){
  ork_npu* c=ork_npu_init(); if(!c)return -1;
  ork_npu_set_core_budget(c,cores);
  int8_t* B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
  int8_t* A=malloc((size_t)M*K); memset(A,1,(size_t)M*K);
  int32_t* C=malloc((size_t)M*N*4);
  ork_w* w=ork_mm_pack_i8(c,K,N,B); if(!w)return -1;
  if(ork_mm_run_i8(c,w,M,A,C))return -1;
  ork_npu_mc_reset();
  double t0=now_us(); for(int i=0;i<iters;i++) ork_mm_run_i8(c,w,M,A,C);
  double us=(now_us()-t0)/iters;
  if(getenv("ORK_MCPROF") && cores>1 && M>1){            /* per-core balance check (prefill path) */
    for(int i=0;i<cores;i++){ double cp,su,ac; long n; ork_npu_mc_timing(i,&cp,&su,&ac,&n);
      if(n) fprintf(stderr,"      [%dx%dx%d c%d] submits=%ld copy=%.1f submit=%.1f acc=%.1f us/submit\n",
                    M,K,N,i,n,cp/n,su/n,ac/n); }
  }
  ork_w_free(w); ork_npu_free(c); free(A);free(B);free(C); return us;
}

static double bench_rknn_ac(int M,int K,int N,int iters,int ac_layout){
  rknn_matmul_ctx ctx; rknn_matmul_info info; rknn_matmul_io_attr io;
  memset(&info,0,sizeof info); memset(&io,0,sizeof io);
  info.M=M; info.K=K; info.N=N; info.type=RKNN_INT8_MM_INT8_TO_INT32; info.AC_layout=ac_layout;
  int ret=rknn_matmul_create(&ctx,&info,&io); if(ret<0)return -1;
  rknn_tensor_mem* Am=rknn_create_mem(ctx, io.A.size);
  rknn_tensor_mem* Bm=rknn_create_mem(ctx, io.B.size);
  rknn_tensor_mem* Cm=rknn_create_mem(ctx, io.C.size);
  memset(Am->virt_addr,1,io.A.size); memset(Bm->virt_addr,1,io.B.size);
  rknn_matmul_set_io_mem(ctx,Am,&io.A);
  rknn_matmul_set_io_mem(ctx,Bm,&io.B);
  rknn_matmul_set_io_mem(ctx,Cm,&io.C);
  if(rknn_matmul_run(ctx))return -1;
  double t0=now_us(); for(int i=0;i<iters;i++) rknn_matmul_run(ctx);
  double us=(now_us()-t0)/iters;
  rknn_destroy_mem(ctx,Am); rknn_destroy_mem(ctx,Bm); rknn_destroy_mem(ctx,Cm);
  rknn_matmul_destroy(ctx); return us;
}
static double bench_rknn(int M,int K,int N,int iters){ return bench_rknn_ac(M,K,N,iters,0); }

int main(int argc,char**argv){
  int iters=argc>1?atoi(argv[1]):200;
  /* 1c-i probe: does RKNN's native A layout (AC_layout=1) actually beat normal A (0) on this HW? */
  if(argc>2 && argv[2][0]=='a'){
    int M=512,K=2048,N=2048;
    printf("RKNN A-layout probe (%dx%dx%d int8, %d iters):\n",M,K,N,iters);
    double n0=bench_rknn_ac(M,K,N,iters,0), n1=bench_rknn_ac(M,K,N,iters,1);
    printf("  normal A (AC_layout=0): %.1f us\n  native A (AC_layout=1): %.1f us   -> native/normal %.2f\n",n0,n1,n1>0?n1/n0:0);
    return 0;
  }
  /* M-sweep: ork vs the closed RKNN matmul API across the batched-verify regime (M=1..32), to check
   * whether ork's kernel stays competitive where spec-decode verify operates (lead #1/#4 follow-up). */
  if(argc>2 && argv[2][0]=='m'){
    int KN[][2]={{3584,3584},{3584,18944}};  /* 7B attn (Q/O) + FFN (gate/up) */
    int Ms[]={1,32,64,128,256,512};
    printf("ork vs RKNN matmul API, M-sweep (int8, %d warm iters). us/matmul:\n",iters);
    for(int kn=0;kn<2;kn++){
      int K=KN[kn][0],N=KN[kn][1];
      printf("  K=%d N=%d:\n",K,N);
      printf("    %-4s %10s %10s %11s %14s\n","M","ork-1core","ork-3core","rknn-1core","ork3c/rknn");
      for(unsigned mi=0;mi<sizeof(Ms)/sizeof(int);mi++){
        int M=Ms[mi];
        double o1=bench_ork(M,K,N,iters,1), o3=bench_ork(M,K,N,iters,3), r=bench_rknn(M,K,N,iters);
        printf("    %-4d %10.1f %10.1f %11.1f %14.2f\n",M,o1,o3,r,(r>0?o3/r:0));
      }
    }
    return 0;
  }
  int shapes[][3]={{1,2048,2048},{1,2048,6144},{1,6144,2048},{1,2048,1536},{512,2048,2048}};
  printf("int8 matmul, %d warm iters. us/matmul (lower=faster):\n",iters);
  printf("  %-16s %10s %10s %10s %12s\n","M x K x N","ork-1core","ork-3core","rknn-1core","ork1/rknn");
  for(int s=0;s<5;s++){
    int M=shapes[s][0],K=shapes[s][1],N=shapes[s][2];
    double o1=bench_ork(M,K,N,iters,1), o3=bench_ork(M,K,N,iters,3), r=bench_rknn(M,K,N,iters);
    char tag[32]; snprintf(tag,sizeof tag,"%dx%dx%d",M,K,N);
    printf("  %-16s %10.1f %10.1f %10.1f %12.2f\n",tag,o1,o3,r,(r>0?o1/r:0));
  }
  return 0;
}
