/* C-Step-1 de-risk gate: does async-pipelining the INDEPENDENT matmul groups of a real
 * Qwen3-1.7B layer (QKV share one activation; gate/up share one) actually overlap the CPU
 * dequant with the NPU submit? Measures group wall SYNC vs PIPELINED + validates bit-exact.
 * PIPE pattern (async contract: <=1 in flight): submit[0]; loop{ wait[i]; submit[i+1]; dequant[i] }
 *   -> dequant[i] (CPU) overlaps submit[i+1] (NPU). n-1 of n dequants hidden.
 *   make layer_pipeline_probe && sudo ./layer_pipeline_probe [M] [iters] */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

static void quant_rows(const float*x,int8_t*xi,float*as,int M,int K){
  for(int m=0;m<M;m++){ const float*r=x+(size_t)m*K; int8_t*o=xi+(size_t)m*K; float mx=1e-9f;
    for(int k=0;k<K;k++){ float v=fabsf(r[k]); if(v>mx)mx=v; }
    as[m]=mx/127.0f; float inv=127.0f/mx;
    for(int k=0;k<K;k++){ float q=r[k]*inv; int qi=(int)(q+copysignf(0.5f,q)); o[k]=(int8_t)(qi>127?127:qi<-127?-127:qi); } } }
static void dequant_rows(const int32_t*C,float*out,const float*as,int M,int N){
  for(int m=0;m<M;m++){ float s=as[m]; const int32_t*r=C+(size_t)m*N; float*o=out+(size_t)m*N;
    for(int n=0;n<N;n++) o[n]=r[n]*s; } }

/* run a group of nw independent matmuls sharing input xi (already quantized), SYNC. */
static double group_sync(ork_npu*c,ork_w**W,int nw,int M,int K,int*Ns,const int8_t*xi,int32_t**Cout,float*as,float**Fout){
  double t0=now_us();
  for(int i=0;i<nw;i++){ ork_i8_mm_run(c,W[i],M,xi,Cout[i]); dequant_rows(Cout[i],Fout[i],as,M,Ns[i]); }
  return now_us()-t0; }
/* same group, PIPELINED: dequant[i] overlaps submit[i+1]. */
static double group_pipe(ork_npu*c,ork_w**W,int nw,int M,int K,int*Ns,const int8_t*xi,int32_t**Cout,float*as,float**Fout){
  double t0=now_us();
  ork_async*h=ork_i8_mm_run_async(c,W[0],M,xi,Cout[0]);
  for(int i=0;i<nw;i++){ ork_async_wait(h);
    if(i+1<nw) h=ork_i8_mm_run_async(c,W[i+1],M,xi,Cout[i+1]);
    dequant_rows(Cout[i],Fout[i],as,M,Ns[i]); }   /* overlaps submit[i+1] */
  return now_us()-t0; }

int main(int argc,char**argv){
  int M=argc>1?atoi(argv[1]):512; int iters=argc>2?atoi(argv[2]):28;
  ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
  int K=2048;
  /* QKV group: 3 weights K=2048 N=2048 (share input). gate/up: 2 weights K=2048 N=6144. */
  int qkvN[3]={2048,2048,2048}, guN[2]={6144,6144};
  ork_w *Wqkv[3],*Wgu[2];
  int8_t*Bbuf=malloc((size_t)K*6144); for(size_t i=0;i<(size_t)K*6144;i++)Bbuf[i]=(int8_t)((i*131+7)%255-127);
  for(int i=0;i<3;i++){ Wqkv[i]=ork_i8_mm_pack(c,K,qkvN[i],Bbuf); if(!Wqkv[i]){printf("pack fail\n");return 1;} }
  for(int i=0;i<2;i++){ Wgu[i]=ork_i8_mm_pack(c,K,guN[i],Bbuf); if(!Wgu[i]){printf("pack fail\n");return 1;} }
  float*x=malloc((size_t)M*K*4); for(size_t i=0;i<(size_t)M*K;i++)x[i]=((i*1103515245u+12345)%2000)/1000.0f-1.0f;
  int8_t*xi=malloc((size_t)M*K); float*as=malloc(M*4); quant_rows(x,xi,as,M,K);
  int32_t*Cq[3]; float*Fq[3],*Fq2[3]; for(int i=0;i<3;i++){Cq[i]=malloc((size_t)M*qkvN[i]*4);Fq[i]=malloc((size_t)M*qkvN[i]*4);Fq2[i]=malloc((size_t)M*qkvN[i]*4);}
  int32_t*Cg[2]; float*Fg[2],*Fg2[2]; for(int i=0;i<2;i++){Cg[i]=malloc((size_t)M*guN[i]*4);Fg[i]=malloc((size_t)M*guN[i]*4);Fg2[i]=malloc((size_t)M*guN[i]*4);}
  printf("layer_pipeline_probe M=%d iters=%d, %d cores\n",M,iters,ork_npu_cores(c));
  /* warm */ group_sync(c,Wqkv,3,M,K,qkvN,xi,Cq,as,Fq); group_pipe(c,Wgu,2,M,K,guN,xi,Cg,as,Fg2);
  double sq=0,pq=0,sg=0,pg=0;
  for(int r=0;r<iters;r++){ sq+=group_sync(c,Wqkv,3,M,K,qkvN,xi,Cq,as,Fq); pq+=group_pipe(c,Wqkv,3,M,K,qkvN,xi,Cq,as,Fq2); }
  for(int r=0;r<iters;r++){ sg+=group_sync(c,Wgu,2,M,K,guN,xi,Cg,as,Fg); pg+=group_pipe(c,Wgu,2,M,K,guN,xi,Cg,as,Fg2); }
  /* validate pipe==sync (bit-exact) */
  group_sync(c,Wqkv,3,M,K,qkvN,xi,Cq,as,Fq); group_pipe(c,Wqkv,3,M,K,qkvN,xi,Cq,as,Fq2);
  long mm=0; for(int i=0;i<3;i++)for(size_t j=0;j<(size_t)M*qkvN[i];j++) if(Fq[i][j]!=Fq2[i][j])mm++;
  printf("  QKV group (3x K2048xN2048): sync=%.0f us  pipe=%.0f us  speedup=%.3fx  mism=%ld\n",sq/iters,pq/iters,sq/pq,mm);
  printf("  gate/up  (2x K2048xN6144): sync=%.0f us  pipe=%.0f us  speedup=%.3fx\n",sg/iters,pg/iters,sg/pg);
  return 0;
}
