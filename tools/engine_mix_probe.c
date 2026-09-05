/* engine_mix_probe — how much aggregate memory throughput do CPU and NPU actually reach together, and is
 * there headroom left for a THIRD engine (Mali)?
 *
 * WHY THIS EXISTS. A previous probe reported "37.4 GB/s aggregate, ~1.5x over the best single engine" by
 * ADDING two differently-defined numbers: the NPU's *useful weight bytes* and the CPU's memcpy read+write
 * traffic. That composite exceeded the board's theoretical LPDDR peak (2112 MHz x 2 x 8 B = ~33.8 GB/s),
 * which is proof it over-counts. It is not a bus-utilisation measurement and must not be quoted as one.
 *
 * This probe fixes the metric two ways:
 *   1. ONE currency. Every engine's work is a pure weight-READ stream (decode is weight streaming; it
 *      writes almost nothing), so "bytes consumed" is unambiguous for CPU and NPU alike -- no read+write
 *      doubling, no guessing about write-allocate.
 *   2. GROUND TRUTH. It samples /sys/class/devfreq/dmc/load, the DDR controller's own utilisation counter
 *      (the same source bench_monitored.sh uses), during every measurement. So the summed GB/s can be
 *      CHECKED against what the memory controller says it actually did, instead of trusted.
 *
 * WHAT IT DECIDES. Decode has been measured latency/serialisation-bound, not bandwidth-bound (one engine
 * leaves the bus far from saturated). If that holds, adding engines wins until the bus saturates -- so the
 * question "does a 3rd engine beat 2?" reduces to: how much DMC headroom is left at 2? If two engines
 * already sit near 100%, a third can only redistribute, and no amount of Mali code changes that. Measure
 * the cheap thing first.
 *
 * Pinning is fixed throughout so configurations are comparable: NPU host thread on core 4, CPU workers on
 * cores 5,6,7.
 *
 *   make engine_mix_probe && sudo tools/util/npu_guard.sh -- ./engine_mix_probe [K] [N] [T]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

/* ---- DDR controller utilisation, sampled in the background ---- */
static volatile int g_dmc_stop=0; static volatile double g_dmc_avg=0, g_dmc_peak=0;
static void *dmc_sampler(void*a){
    cpu_set_t s; CPU_ZERO(&s); for(int i=0;i<4;i++) CPU_SET(i,&s); sched_setaffinity(0,sizeof s,&s); /* little cores */
    double sum=0; int n=0; g_dmc_peak=0;
    while(!g_dmc_stop){
        FILE*f=fopen("/sys/class/devfreq/dmc/load","r");
        if(f){ int pct=0; if(fscanf(f,"%d",&pct)==1){ sum+=pct; n++; if(pct>g_dmc_peak) g_dmc_peak=pct; } fclose(f); }
        struct timespec d={0,20*1000*1000}; nanosleep(&d,0);
    }
    g_dmc_avg = n? sum/n : 0; (void)a; return NULL;
}

/* ---- CPU engine: pure weight-READ stream (decode-shaped), one 64 MiB buffer per thread ---- */
#define MAXW 3
static volatile int g_cpu_stop=0; static volatile double g_cpu_gb[MAXW];
typedef struct { int core, gemv; } warg;
static void *cpu_stream(void*a){
    warg*w=(warg*)a; cpu_set_t s; CPU_ZERO(&s); CPU_SET(w->core,&s); sched_setaffinity(0,sizeof s,&s);
    size_t B=64u<<20; uint64_t *buf=malloc(B); if(!buf) return NULL;
    memset(buf,1,B); size_t nw=B/8;
    double t0=now_us(); size_t bytes=0; uint64_t acc=0;
    const int8_t *q=(const int8_t*)buf; size_t nq=B;
    while(!g_cpu_stop){
        if(!w->gemv){
            /* STREAM: the best case -- pure sequential reads, maximum outstanding requests. */
            for(size_t i=0;i<nw;i+=8) acc+=buf[i]+buf[i+1]+buf[i+2]+buf[i+3]+buf[i+4]+buf[i+5]+buf[i+6]+buf[i+7];
        } else {
            /* GEMV: decode-shaped -- a dequant-and-accumulate per weight byte. The ALU work per byte is
             * what limits how many loads stay in flight, which is why a real M=1 decode loop reaches only
             * ~8 GB/s on one A76 out of the ~21.9 GB/s it can stream (see test_nf4_decode). Bandwidth is
             * not the binding constraint there; memory-level parallelism is. */
            int32_t a0=0,a1=0,a2=0,a3=0;
            for(size_t i=0;i<nq;i+=4){ a0+=q[i]*3; a1+=q[i+1]*5; a2+=q[i+2]*7; a3+=q[i+3]*11; }
            acc += (uint64_t)(a0+a1+a2+a3);
        }
        bytes+=B; double dt=(now_us()-t0)/1e6; if(dt>0) g_cpu_gb[w->core-5]=bytes/1e9/dt;
    }
    __asm__ __volatile__("" : : "r"(acc) : "memory");
    free(buf); return NULL;
}
static void cpu_start(pthread_t*th,warg*wa,int n,int gemv){ g_cpu_stop=0;
    for(int i=0;i<n;i++){ g_cpu_gb[i]=0; wa[i].core=5+i; wa[i].gemv=gemv; pthread_create(&th[i],0,cpu_stream,&wa[i]); } }
static double cpu_stop(pthread_t*th,int n){ double s=0; for(int i=0;i<n;i++) s+=g_cpu_gb[i];
    g_cpu_stop=1; for(int i=0;i<n;i++) pthread_join(th[i],0); return s; }

/* ---- NPU engine: chained int8 matmul, weight bytes consumed ---- */
typedef struct { ork_npu*c; ork_mm_task_i8*tk; int T,K,N,M; volatile int stop; volatile double gb, gmac; } npu_arg;
static void *npu_stream(void*a){
    npu_arg*n=(npu_arg*)a; cpu_set_t s; CPU_ZERO(&s); CPU_SET(4,&s); sched_setaffinity(0,sizeof s,&s);
    double t0=now_us(); size_t bytes=0; size_t macs=0;
    while(!n->stop){ if(ork_i8_mm_run_chain(n->c,n->T,n->tk)<0) break;
        bytes += (size_t)n->T*n->K*n->N;
        macs  += (size_t)n->T*n->M*n->K*n->N;
        double dt=(now_us()-t0)/1e6; if(dt>0){ n->gb=bytes/1e9/dt; n->gmac=macs/1e9/dt; } }
    return NULL;
}

int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):1024, N=argc>2?atoi(argv[2]):640, T=argc>3?atoi(argv[3]):16;
    /* ORK_MIX_NPU_M>1 makes the NPU arm COMPUTE-bound: one weight stream amortised over M rows, so
     * arithmetic intensity rises and the NPU stops being a heavy DRAM client. M=1 is the decode shape. */
    int NPUM = getenv("ORK_MIX_NPU_M") ? atoi(getenv("ORK_MIX_NPU_M")) : 1;
    setvbuf(stdout,0,_IONBF,0);
    { cpu_set_t s4; CPU_ZERO(&s4); CPU_SET(4,&s4); sched_setaffinity(0,sizeof s4,&s4); }
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    int8_t*B=malloc((size_t)K*N); for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)(i&0x3f);
    ork_w*w=ork_i8_mm_pack(c,K,N,B); if(!w){ printf("pack fail\n"); return 2; }
    int8_t*A=(int8_t*)ork_dma_alloc(c,(size_t)K*(NPUM>1?NPUM:1)); memset(A,1,(size_t)K*(NPUM>1?NPUM:1));
    int32_t*Cb=(int32_t*)ork_dma_alloc(c,(size_t)T*N*4*(NPUM>1?NPUM:1));
    ork_mm_task_i8*tk=calloc(T,sizeof *tk);
    for(int j=0;j<T;j++){ tk[j].w=w; tk[j].M=NPUM; tk[j].A=A; tk[j].C=Cb+(size_t)j*N*NPUM; tk[j].cstride=0; }

    double PEAK = 2112.0*2*8/1000.0;   /* MHz * DDR * 64-bit / 1000 -> GB/s theoretical */
    printf("engine_mix_probe: K=%d N=%d T=%d | LPDDR theoretical peak %.1f GB/s (DMC 2112 MHz x2 x8B)\n",K,N,T,PEAK);
    printf("  NPU host on core 4; CPU workers on cores 5-7. GB/s = BYTES CONSUMED. NPU M=%d (%s).\n\n",
           NPUM, NPUM>1?"COMPUTE-bound: weight amortised over M rows":"memory-bound decode shape");
    printf("  %-14s %8s %8s %8s   %8s %8s\n","config","CPU GB/s","NPU GB/s","sum","DMC avg","DMC peak");

    struct { int ncpu, npu; } cfg[] = {{1,0},{2,0},{3,0},{0,1},{1,1},{2,1},{3,1}};
    double best_single=0, best_sum=0; double dmc_at_best=0;
    for(int gemv=0; gemv<2; gemv++){
    printf("\n  --- CPU arm: %s ---\n", gemv?"GEMV (decode-shaped: dequant+MAC per byte, latency-bound)"
                                              :"STREAM (best case: pure sequential reads)");
    for(unsigned ci=0; ci<sizeof cfg/sizeof cfg[0]; ci++){
        int ncpu=cfg[ci].ncpu, usenpu=cfg[ci].npu;
        pthread_t cth[MAXW]; warg wa[MAXW]; pthread_t nth; npu_arg na={c,tk,T,K,N,0,0};
        pthread_t dth; g_dmc_stop=0;
        if(usenpu) ork_i8_mm_run_chain(c,T,tk);                       /* warm */
        if(ncpu) cpu_start(cth,wa,ncpu,gemv);
        if(usenpu){ na.stop=0; pthread_create(&nth,0,npu_stream,&na); }
        struct timespec settle={0,400*1000*1000}; nanosleep(&settle,0);
        pthread_create(&dth,0,dmc_sampler,0);
        struct timespec run={2,0}; nanosleep(&run,0);
        g_dmc_stop=1; pthread_join(dth,0);
        double cg = ncpu? cpu_stop(cth,ncpu) : 0;
        double ng = 0, nmac = 0; if(usenpu){ na.stop=1; pthread_join(nth,0); ng=na.gb; nmac=ng*NPUM; }
        /* GMAC/s = weight-GB/s * M exactly (macs = T*M*K*N, bytes = T*K*N).
         * CAVEAT: at M>1 the 'bytes consumed' column counts WEIGHT ONLY, but the NPU also reads A
         * (M*K per task) and writes C (M*N*4). At M=256 that is 256 KiB + 160 KiB against 160 KiB of
         * weight -- so the GB/s column undercounts NPU DRAM traffic ~3.6x here. Trust DMC, not the sum. */
        char lbl[32]; snprintf(lbl,sizeof lbl,"%s%s%s", ncpu?"CPU":"", ncpu?(char[]){'0'+ncpu,0}:"", usenpu?(ncpu?"+NPU":"NPU"):"");
        double sum=cg+ng;
        if(NPUM>1) printf("  %-14s %8.2f %8.2f %8.2f   %7.0f%% %7.0f%%   NPU %7.1f GMAC/s\n",lbl,cg,ng,sum,g_dmc_avg,g_dmc_peak,nmac);
        else       printf("  %-14s %8.2f %8.2f %8.2f   %7.0f%% %7.0f%%\n",lbl,cg,ng,sum,g_dmc_avg,g_dmc_peak);
        if((ncpu==0||usenpu==0) && sum>best_single) best_single=sum;
        if(sum>best_sum){ best_sum=sum; dmc_at_best=g_dmc_avg; }
    }
    }
    printf("\n  best single engine %.2f GB/s | best mix %.2f GB/s = %.2fx | DMC at best mix %.0f%%\n",
           best_single,best_sum,best_single>0?best_sum/best_single:0,dmc_at_best);
    printf("  summed GB/s as a share of theoretical peak at the best mix: %.0f%%\n", best_sum/PEAK*100);
    printf("\n  HEADROOM FOR A THIRD ENGINE: %s\n",
        dmc_at_best>=90 ? "NONE — the DDR controller is already saturated; a Mali arm can only redistribute."
      : dmc_at_best>=70 ? "LIMITED — some headroom, but a 3rd engine competes for the last ~30%."
                        : "REAL — the bus is far from saturated, so decode is latency-bound and a 3rd engine should add.");
    ork_mm_free(c,w); ork_npu_free(c); return 0;
}
