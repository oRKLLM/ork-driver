/* decode_bw_probe — how does M=1 weight-DMA bandwidth scale with the number of CONCURRENT independent
 * matmuls in one doorbell batch? This is the question that decides whether dense decode can ever beat the
 * CPU on this NPU: ORK_FFN_PROF measured ~3.5 GB/s for one task and 7.2 GB/s for two, i.e. ~3.5 GB/s per
 * concurrent task, while the CPU achieves ~9.4 GB/s effective on the same weights. If that scaling
 * continues, >=3 streams matches the CPU; if it plateaus at 2, dense decode is structurally lost.
 *   sudo env ORK_MM_TIMEOUT=3000 ./decode_bw_probe [K] [N] [reps]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <math.h>
/* CPU MEMORY-BANDWIDTH CONTENDERS. The probe measures the NPU with an IDLE CPU; a real decode token runs
 * ggml's norms/attention/sampling on 4 big cores at the same time, competing for the SAME DRAM. If the
 * NPU's achieved bandwidth collapses under CPU load, then moving work to the NPU does not ADD a parallel
 * resource -- it splits one memory bus -- which would explain why the FFN handler gets 3.1-6.3 GB/s where
 * this probe gets 8.8-16 at identical shapes through the identical call. ORK_BW_LOAD=<threads>. */
static volatile int g_load_stop = 0;
/* PURE-SPIN contenders (ORK_BW_SPIN=<threads>). Different from the memcpy load above and that matters:
 * memcpy threads block on DRAM and yield the core, whereas ggml's threadpool SPIN-WAITS at barriers,
 * burning CPU. The doorbell's completion detection is itself a spin loop, so runnable spinners can
 * deschedule it and inflate the measured poll even though the NPU finished on time. */
static void *spin_thread(void *arg){
    volatile unsigned long x = 0;
    while(!g_load_stop) { for(int i=0;i<10000;i++) x += i; }
    (void)x; return arg;
}
static void *load_thread(void *arg){
    size_t n = 64u<<20; char *a = malloc(n), *b = malloc(n);
    if(!a||!b) return arg;
    memset(a,1,n);
    while(!g_load_stop) memcpy(b,a,n);
    free(a); free(b); return arg;
}
/* Kernel-side per-job HARDWARE time (#patch74). Userspace only sees hardware time PLUS however long it
 * takes to notice the sentinel, so this is the one split userspace cannot make itself. */
static void hw_zero(void){
    FILE*f; if((f=fopen("/sys/module/rknpu/parameters/hw_ns_sum","w"))){fputs("0",f);fclose(f);}
    if((f=fopen("/sys/module/rknpu/parameters/hw_n","w"))){fputs("0",f);fclose(f);} }
static void hw_read(unsigned long long*sum,unsigned long*n){
    FILE*f; *sum=0; *n=0;
    if((f=fopen("/sys/module/rknpu/parameters/hw_ns_sum","r"))){ if(fscanf(f,"%llu",sum)!=1)*sum=0; fclose(f);}
    if((f=fopen("/sys/module/rknpu/parameters/hw_n","r"))){ if(fscanf(f,"%lu",n)!=1)*n=0; fclose(f);} }
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }
#define SMAX 6
#define NWMAX 96
int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):1024, N=argc>2?atoi(argv[2]):3072, REP=argc>3?atoi(argv[3]):40;
    /* nw = how many DISTINCT weights the run cycles through. The point: this probe originally reused <=6
     * weights in a tight loop and measured 8.6 GB/s at S=1, while the real FFN decode handler -- same
     * shapes, same ork_i8_mm_run -- gets 3.5. A decode token touches 28 layers x 3 weights = 84 DISTINCT
     * 3MB regions, so if bandwidth degrades with the number of distinct weights, the handler's gap is a
     * WORKING-SET effect (IOMMU TLB / page-table-walk locality), not batching or pipelining. */
    int NW=argc>4?atoi(argv[4]):SMAX; if(NW<1)NW=1; if(NW>NWMAX)NW=NWMAX;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("decode_bw_probe: M=1, K=%d N=%d, weight=%.2f MB each, %d reps\n",K,N,(double)K*N/1048576.0,REP);
    int8_t *B=malloc((size_t)K*N); uint32_t g=0x9e;
    for(size_t i=0;i<(size_t)K*N;i++){ g=g*1664525u+1013904223u; B[i]=(int8_t)((g>>25)%7)-3; }
    int8_t *A=malloc((size_t)K); for(int k=0;k<K;k++)A[k]=(int8_t)(k&15);
    ork_w *w[NWMAX]; int32_t *C[NWMAX];
    for(int i=0;i<NW;i++){ w[i]=ork_i8_mm_pack(c,K,N,B); C[i]=calloc((size_t)N,4);
        if(!w[i]){ printf("pack %d failed (nw=%d, %.0f MB) — lower nw\n",i,NW,(double)NW*K*N/1048576.0); return 2; } }
    printf("  distinct weights nw=%d (%.0f MB resident)\n", NW, (double)NW*K*N/1048576.0);
    int NL = getenv("ORK_BW_LOAD") ? atoi(getenv("ORK_BW_LOAD")) : 0;
    int NS = getenv("ORK_BW_SPIN") ? atoi(getenv("ORK_BW_SPIN")) : 0;
    pthread_t lt[8], st[8]; if(NL>8)NL=8; if(NS>8)NS=8;
    for(int i=0;i<NL;i++) pthread_create(&lt[i],0,load_thread,0);
    for(int i=0;i<NS;i++) pthread_create(&st[i],0,spin_thread,0);
    if(NL||NS) { printf("  CPU contenders: %d memcpy + %d PURE-SPIN threads\n", NL, NS); struct timespec ts={0,200*1000*1000}; nanosleep(&ts,0); }
    printf("  S  tasks  ms/batch   GB/s   GB/s per task\n");
    for(int S=1;S<=SMAX && S<=NW;S++){
        ork_mm_task_i8 t[SMAX];
        for(int i=0;i<S;i++) t[i]=(ork_mm_task_i8){ w[i], 1, A, C[i] };
        if(ork_i8_mm_run_chain(c,S,t)){ printf("  %d  chain rc!=0 (declined)\n",S); continue; }   /* warm */
        if(getenv("ORK_PROFILE")){ ork_npu_db_reset(); hw_zero(); }   /* reset BEFORE the timed loop, else the average
                                                        * folds in init/pack warmups and reads absurdly high */
        double t0=now_us();
        /* ROTATE through the pool so each rep touches a different set -- the real decode access pattern. */
        /* ORK_BW_FRESH=1: malloc/free the A and C buffers EVERY rep, as the FFN decode handler does (6
         * fresh buffers per layer, ~168 per token) instead of reusing them. If anything downstream caches
         * per-buffer state -- a DMA-mapping lookup, a warm descriptor -- fresh pointers miss every call,
         * and that is the last structural difference between this probe and the handler at identical
         * shapes through the identical entrypoint. */
        static int fresh=-1; if(fresh<0) fresh=getenv("ORK_BW_FRESH")?1:0;
        for(int r=0;r<REP;r++){
            int off=(r*S)%NW;
            int8_t *Af=A; int32_t *Cf[SMAX];
            if(fresh){ Af=malloc((size_t)K); memcpy(Af,A,(size_t)K);
                       for(int i=0;i<S;i++) Cf[i]=calloc((size_t)N,4); }
            for(int i=0;i<S;i++){ int j=(off+i)%NW; t[i]=(ork_mm_task_i8){ w[j], 1, Af, fresh?Cf[i]:C[j] }; }
            int rc=ork_i8_mm_run_chain(c,S,t);
            if(fresh){ free(Af); for(int i=0;i<S;i++) free(Cf[i]); }
            if(rc){ printf("  %d  FAILED mid-loop\n",S); break; } }
        double us=(now_us()-t0)/REP;
        double bytes=(double)S*K*N;
        printf("  %d  %5d  %8.3f  %5.2f  %5.2f\n", S, S, us/1000.0, bytes/us/1000.0, bytes/us/1000.0/S);
        /* PER-PHASE DECOMPOSITION (ORK_PROFILE=1). copy = activation host-copy + bsync; submit = regcmd
         * synth + ioctl + result bsync; synth = the host subset of submit, so ioctl+HW = submit - synth;
         * acc = host K-slice accumulate. This is the comparison that was never made: the same numbers
         * out of the FFN decode handler tell us WHICH phase costs the handler its 2.4x. */
        if(getenv("ORK_PROFILE")){
            double bu=0,eu=0; long bn=0,en=0;
            ork_npu_db_timing(&bu,&bn,&eu,&en);
            unsigned long long hs=0; unsigned long hn=0; hw_read(&hs,&hn);
            double pl=ork_npu_db_poll();
            if(bn) printf("       doorbell: begin=%7.1f  end=%7.1f us/call | KERNEL hw=%.1f us/job (%.1f jobs/call) | end split: WAIT=%.1f  post(drain+accum+writeback)=%.1f\n",
                          bu/bn, en?eu/en:0.0, hn?(double)hs/hn/1000.0:0.0, bn?(double)hn/bn:0.0,
                          pl/bn, (eu-pl)/bn);
            ork_npu_db_reset();
        }
    }
    g_load_stop = 1; for(int i=0;i<NL;i++) pthread_join(lt[i],0); for(int i=0;i<NS;i++) pthread_join(st[i],0);
    /* FFN-SEQUENCE MODE (ORK_BW_FFNSEQ=1). Reproduce the decode FFN handler's PATTERN in the probe's
     * controlled environment: per rep, a 2-task gate+up chain (K x Nff), then a host gap (the silu), then
     * a 1-task down run (Nff x K) -- and report DOWN's doorbell poll separately. Everything else about
     * the two contexts is already proven identical (shape, Sk=3, bf=1, nc=3, NPU clock, weight form,
     * buffers, working set, CPU load), so if the poll inflates here too, the cause is the SEQUENCE and
     * not the consumer. Shapes are taken from qwen3-0.6b: K=1024, Nff=3072. */
    if(getenv("ORK_BW_FFNSEQ")){
        int Kf=1024, Nf=3072;
        int8_t *Bg=malloc((size_t)Kf*Nf), *Bd=malloc((size_t)Nf*Kf);
        for(size_t i=0;i<(size_t)Kf*Nf;i++){ g=g*1664525u+1013904223u; Bg[i]=(int8_t)((g>>25)%7)-3; Bd[i]=(int8_t)((g>>19)%7)-3; }
        ork_w *wg=ork_i8_mm_pack(c,Kf,Nf,Bg), *wu=ork_i8_mm_pack(c,Kf,Nf,Bg), *wd=ork_i8_mm_pack(c,Nf,Kf,Bd);
        int8_t *xi=malloc((size_t)Kf), *gl=malloc((size_t)Nf);
        int32_t *gi=calloc((size_t)Nf,4), *ui=calloc((size_t)Nf,4), *di=calloc((size_t)Kf,4);
        float *glf=malloc((size_t)Nf*4);
        if(wg&&wu&&wd&&xi&&gl&&gi&&ui&&di&&glf){
            for(int k=0;k<Kf;k++) xi[k]=(int8_t)(k&15);
            double gu_us=0, dn_us=0, hb=0,he=0; long hbn=0,hen=0;
            ork_mm_task_i8 t2[2]={{wg,1,xi,gi},{wu,1,xi,ui}};
            ork_i8_mm_run_chain(c,2,t2); ork_i8_mm_run(c,wd,1,gl,di);       /* warm */
            for(int r=0;r<REP;r++){
                double a=now_us(); ork_i8_mm_run_chain(c,2,t2); gu_us+=now_us()-a;
                for(int n2=0;n2<Nf;n2++){ float v=(float)gi[n2]*1e-4f, u=(float)ui[n2]*1e-4f;   /* the host silu gap */
                    glf[n2]=(v/(1.0f+expf(-v)))*u; }
                for(int n2=0;n2<Nf;n2++){ int q=(int)lrintf(glf[n2]*100.0f); gl[n2]=(int8_t)(q>127?127:q<-127?-127:q); }
                ork_npu_db_reset();
                double b=now_us(); ork_i8_mm_run(c,wd,1,gl,di); dn_us+=now_us()-b;
                { double bu=0,eu=0; long bn=0,en=0; ork_npu_db_timing(&bu,&bn,&eu,&en); hb+=bu; hbn+=bn; he+=eu; hen+=en; }
            }
            printf("  FFN-SEQ: gate+up %.0f us   down %.0f us  |  DOWN doorbell begin=%.1f end(poll)=%.1f us/call\n",
                   gu_us/REP, dn_us/REP, hbn?hb/hbn:0.0, hen?he/hen:0.0);
            printf("           (handler at these shapes: gate+up ~1005, down ~814; down begin 381.8 end 342.3)\n");
        } else printf("  FFN-SEQ: alloc/pack failed\n");
        if(wg)ork_mm_free(c,wg); if(wu)ork_mm_free(c,wu); if(wd)ork_mm_free(c,wd);
        free(Bg);free(Bd);free(xi);free(gl);free(gi);free(ui);free(di);free(glf);
    }
    printf("PASS — bandwidth scaling measured\n");
    for(int i=0;i<NW;i++){ ork_mm_free(c,w[i]); free(C[i]); }
    ork_npu_free(c); free(A); free(B); return 0;
}
