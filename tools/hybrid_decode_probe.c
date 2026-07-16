/* hybrid_decode_probe — validate the decode-pipeline aggregate claim: CPU int4 bulk ‖ NPU int8 share,
 * overlapped, per token. Models one decode token's active-expert matmuls (M=1, expert dims). Splits W
 * matmuls into an NPU int8 share (dedicated thread; chained to amortize the submit floor) and a CPU int4
 * bulk (ork_native_cpu.h GEMV on the remaining big cores). Question: does the NPU share finish INSIDE the
 * CPU bulk window (aggregate win), or does the M=1 submit floor push it past (loss)?
 *   make hybrid_decode_probe && sudo ./hybrid_decode_probe [W=96] [n_npu=12] [K=2048] [N=512]
 * (NPU op — do NOT run concurrently with another NPU workload.)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include "ork_native_cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

static int gK,gN,gW,gNnpu,gCPUthreads,gNcpu;
static ork_cpu_w *gCW; static const int8_t *gA; static float gAsc; static float *gCout;
/* NPU thread state */
static ork_npu *gNPU; static ork_mm_task_i8 *gTK; static int gTKn; static double gNpuDt; static int gNpuRc;

typedef struct{int lo,hi,core;}cjob;
static void* cpu_worker(void*p){ cjob*j=p; cpu_set_t s;CPU_ZERO(&s);CPU_SET(j->core,&s);pthread_setaffinity_np(pthread_self(),sizeof s,&s);
    for(int w=j->lo;w<j->hi;w++) ork_cpu_gemv_m1(&gCW[w],gA,gAsc,gCout+(size_t)w*gN,0,gN);
    return NULL; }
static void cpu_run(int w0,int w1){ int W=w1-w0, nt=gCPUthreads; pthread_t th[8]; cjob jb[8]; int per=(W+nt-1)/nt;
    for(int t=0;t<nt;t++){ jb[t]=(cjob){w0+t*per,(w0+(t+1)*per<w1?w0+(t+1)*per:w1),4+t}; pthread_create(&th[t],0,cpu_worker,&jb[t]); }
    for(int t=0;t<nt;t++) pthread_join(th[t],0); }
static void* npu_tramp(void*p){ (void)p; cpu_set_t s;CPU_ZERO(&s);CPU_SET(7,&s);pthread_setaffinity_np(pthread_self(),sizeof s,&s);
    double t0=now_us(); gNpuRc=ork_mm_run_chain_i8(gNPU,gTKn,gTK); gNpuDt=now_us()-t0; return NULL; }

int main(int argc,char**argv){
    int W=argc>1?atoi(argv[1]):96, Nnpu=argc>2?atoi(argv[2]):12, K=argc>3?atoi(argv[3]):2048, N=argc>4?atoi(argv[4]):512;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    gK=K;gN=N;gW=W;gNnpu=Nnpu;gCPUthreads=3; gNcpu=W-Nnpu; gNPU=c;
    printf("hybrid_decode_probe: W=%d (M=1,K=%d,N=%d) | NPU int8 share=%d, CPU int4 bulk=%d (%d threads)\n",W,K,N,Nnpu,gNcpu,gCPUthreads);

    float*Wf=malloc((size_t)N*K*4); uint64_t s=0x2545F49ULL;
    for(size_t i=0;i<(size_t)N*K;i++){ s=s*6364136223846793005ULL+1442695040888963407ULL; Wf[i]=0.06f*((int)((s>>40)&1023)-512)/512.0f; }
    uint8_t*nib=malloc((size_t)N*K/2); float*bs=malloc(N*4);
    ork_cpu_pack(ORK_CPU_I4,K,N,Wf,nib,0,0,0,bs,0);
    static ork_cpu_w cw; cw.fmt=ORK_CPU_I4; cw.nibble=nib; cw.bscale=bs; cw.K=K; cw.N=N;
    gCW=malloc(sizeof(ork_cpu_w)*W); for(int w=0;w<W;w++) gCW[w]=cw;
    int8_t*Ai8=malloc(K); for(int k=0;k<K;k++) Ai8[k]=(int8_t)((k%17)-8); gA=Ai8; gAsc=1.0f;
    gCout=malloc((size_t)W*N*4);
    int32_t*NC=malloc((size_t)Nnpu*N*4);
    int8_t*Bi8=malloc((size_t)K*N); for(size_t i=0;i<(size_t)K*N;i++) Bi8[i]=(int8_t)((i%15)-7);
    ork_mm_task_i8 *tk=malloc(sizeof(ork_mm_task_i8)*Nnpu);
    for(int i=0;i<Nnpu;i++){ ork_npu_set_pack_domain(c,0); ork_w*w8=ork_mm_pack_i8(c,K,N,Bi8); if(!w8){printf("pack_i8 fail\n");return 1;}
        tk[i].w=w8; tk[i].M=1; tk[i].A=Ai8; tk[i].C=NC+(size_t)i*N; }
    gTK=tk; gTKn=Nnpu;

    cpu_run(0,W); ork_mm_run_chain_i8(c,Nnpu,tk);   /* warm */
    double t0=now_us(); for(int r=0;r<20;r++) cpu_run(0,W); double t_cpu_all=(now_us()-t0)/20;
    t0=now_us(); for(int r=0;r<20;r++) ork_mm_run_chain_i8(c,Nnpu,tk); double t_npu=(now_us()-t0)/20;
    t0=now_us(); for(int r=0;r<20;r++) cpu_run(0,gNcpu); double t_cpu_bulk=(now_us()-t0)/20;
    double t_hy=0; for(int r=0;r<20;r++){ double h0=now_us(); pthread_t nth;
        pthread_create(&nth,0,npu_tramp,0); cpu_run(0,gNcpu); pthread_join(nth,0); t_hy+=now_us()-h0; }
    t_hy/=20;

    printf("  CPU-all (%d int4):                %8.1f us\n",W,t_cpu_all);
    printf("  NPU share solo (%d chained int8): %8.1f us\n",Nnpu,t_npu);
    printf("  CPU bulk solo (%d int4):          %8.1f us\n",gNcpu,t_cpu_bulk);
    printf("  HYBRID (bulk || NPU share):       %8.1f us  (npu_dt %.0f rc %d)\n",t_hy,gNpuDt,gNpuRc);
    printf("  ★ hybrid vs CPU-all: %.2fx  (>1 => aggregate WIN: NPU share hidden in the CPU window)\n", t_cpu_all/t_hy);
    printf("  (NPU hidden if npu_solo %.0f <= cpu_bulk %.0f)\n", t_npu, t_cpu_bulk);
    ork_npu_free(c); return 0;
}
