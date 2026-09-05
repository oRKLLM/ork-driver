/* npu_fixed_cost_probe — WHAT is the fixed per-task cost, and why does CPU DRAM load inflate it 2.19x?
 *
 * sram_port_probe decomposed the chained int8 matmul into time = a + b*N per task (a ~= 8.55 us fixed,
 * b*N the weight stream) and showed that under a 23.5 GB/s CPU DRAM load the two parts behave completely
 * differently: the weight stream slows only 1.22x, while the FIXED part slows 2.19x. The fixed part is
 * therefore the dominant contention cost on this path -- bigger than anything weight placement offered --
 * and it is entirely uncharacterised.
 *
 * Before trying to fix it, establish WHERE it is. The candidates split cleanly in two:
 *   HOST-side   -- building T task descriptors / regcmd per submit, ioctl, poll. Touches DRAM from the CPU,
 *                  so a CPU DRAM load inflates its latency.
 *   HARDWARE    -- the NPU's own per-task work: regcmd fetch by the PC, descriptor read, A read, C write.
 *
 * The kernel tells us which, for free: #patch74 exposes cumulative hardware time per job as
 * /sys/module/rknpu/parameters/{hw_ns_sum,hw_n} (ns summed over jobs, and the job count). Fitting BOTH
 * wall-clock and kernel-reported hardware time against N, under SPIN and under LOAD, splits the 2.19x:
 *
 *   a_hw inflates like a_wall  => the cost is ON THE NPU (regcmd/descriptor/A/C fetch latency)
 *                                 -> lever: move those small buffers to SRAM (they are NPU-read, 0x403)
 *   a_hw flat, a_wall inflates => the cost is ON THE HOST (descriptor build / submit path)
 *                                 -> lever: precompiled regcmd (ork_pc_*), fewer host writes per task
 *
 * SPIN (same cores busy, zero memory traffic) is the baseline rather than idle, so CPU occupancy is held
 * constant and the only difference between arms is DRAM traffic.
 *
 *   make npu_fixed_cost_probe && sudo tools/util/npu_guard.sh -- ./npu_fixed_cost_probe [K] [T] [reps]
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

static unsigned long long rd_u64(const char*p){
    FILE*f=fopen(p,"r"); if(!f) return 0ULL; unsigned long long v=0; if(fscanf(f,"%llu",&v)!=1) v=0; fclose(f); return v;
}
#define P_HWNS "/sys/module/rknpu/parameters/hw_ns_sum"
#define P_HWN  "/sys/module/rknpu/parameters/hw_n"

/* ---- contenders on cores 5,6,7 (host thread owns core 4) ---- */
#define NCONT 3
static volatile int g_stop=0;
static volatile double g_gbps[NCONT];
typedef struct { int core, load; } cont_arg;
static void *contender(void*arg){
    cont_arg*ca=(cont_arg*)arg;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(ca->core,&s); sched_setaffinity(0,sizeof s,&s);
    if(!ca->load){ volatile double x=1.0; while(!g_stop){ for(int i=0;i<10000;i++) x=x*1.000001+1e-9; } return NULL; }
    size_t B=64u<<20; char*a=malloc(B),*b=malloc(B);
    if(!a||!b){ free(a); free(b); return NULL; }
    memset(a,1,B); memset(b,2,B);
    double t0=now_us(); size_t mv=0;
    while(!g_stop){ memcpy(b,a,B);
        __asm__ __volatile__("" : : "r"(a),"r"(b) : "memory");   /* -O3 deletes this copy otherwise */
        mv+=B; double dt=(now_us()-t0)/1e6; if(dt>0) g_gbps[ca->core-5]=2.0*mv/1e9/dt; }
    free(a); free(b); return NULL;
}
static void cstart(pthread_t*th,cont_arg*ca,int load){ g_stop=0;
    for(int i=0;i<NCONT;i++){ g_gbps[i]=0; ca[i].core=5+i; ca[i].load=load; pthread_create(&th[i],0,contender,&ca[i]); } }
static double cstop(pthread_t*th){ double s=0; for(int i=0;i<NCONT;i++) s+=g_gbps[i];
    g_stop=1; for(int i=0;i<NCONT;i++) pthread_join(th[i],0); return s; }

/* least squares y = a + b*x over n points */
static void fit(const double*x,const double*y,int n,double*a,double*b){
    double sx=0,sy=0,sxy=0,sxx=0;
    for(int i=0;i<n;i++){ sx+=x[i]; sy+=y[i]; sxy+=x[i]*y[i]; sxx+=x[i]*x[i]; }
    *b=(n*sxy-sx*sy)/(n*sxx-sx*sx); *a=(sy-*b*sx)/n;
}

int main(int argc,char**argv){
    int K   = argc>1?atoi(argv[1]):1024;
    int T   = argc>2?atoi(argv[2]):16;
    int REP = argc>3?atoi(argv[3]):3;
    setvbuf(stdout,0,_IONBF,0);
    { cpu_set_t s4; CPU_ZERO(&s4); CPU_SET(4,&s4); sched_setaffinity(0,sizeof s4,&s4); }

    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    if(rd_u64(P_HWN)==0 && rd_u64(P_HWNS)==0)
        printf("NOTE: kernel hw counters read 0 — is #patch74 live? (hardware column will be meaningless)\n");

    int NS[3]={160,320,640}; int NMAXW=640;
    int8_t*B=malloc((size_t)K*NMAXW); for(size_t i=0;i<(size_t)K*NMAXW;i++) B[i]=(int8_t)(i&0x3f);
    int8_t*A=(int8_t*)ork_dma_alloc(c,(size_t)K); memset(A,1,(size_t)K);
    int32_t*Cb=(int32_t*)ork_dma_alloc(c,(size_t)T*NMAXW*4);
    ork_mm_task_i8*tk=calloc(T,sizeof *tk);
    ork_w *W[3]; for(int i=0;i<3;i++){ W[i]=ork_i8_mm_pack(c,K,NS[i],B); if(!W[i]){ printf("pack fail\n"); return 2; } }

    printf("npu_fixed_cost_probe: K=%d T=%d reps=%d  (per-task us; hardware = kernel hw_ns_sum/hw_n / T)\n",K,T,REP);
    double aw[2],bw[2],ah[2],bh[2]; double cpu_load=0;
    const char*CN[2]={"SPIN","LOAD"};

    for(int cond=0;cond<2;cond++){
        double xs[3],yw[3],yh[3];
        for(int i=0;i<3;i++){
            int N=NS[i];
            for(int j=0;j<T;j++){ tk[j].w=W[i]; tk[j].M=1; tk[j].A=A; tk[j].C=Cb+(size_t)j*N; tk[j].cstride=0; }
            double wsum=0,hsum=0;
            for(int r=0;r<REP;r++){
                for(int k=0;k<3;k++) ork_i8_mm_run_chain(c,T,tk);          /* warm */
                pthread_t th[NCONT]; cont_arg ca[NCONT];
                cstart(th,ca,cond); struct timespec ss={0,300*1000*1000}; nanosleep(&ss,0);
                unsigned long long ns0=rd_u64(P_HWNS), n0=rd_u64(P_HWN);
                double t0=now_us(); int IT=40;
                for(int k=0;k<IT;k++) ork_i8_mm_run_chain(c,T,tk);
                double wall=(now_us()-t0);
                unsigned long long ns1=rd_u64(P_HWNS), n1=rd_u64(P_HWN);
                double g=cstop(th); if(cond==1) cpu_load=g;
                wsum += wall/(IT*T);
                hsum += (n1>n0) ? ((double)(ns1-ns0)/1e3)/((double)(n1-n0)*T) : 0.0;
            }
            xs[i]=N; yw[i]=wsum/REP; yh[i]=hsum/REP;
            printf("  %-4s N=%-4d  wall %7.3f us/task   hardware %7.3f us/task   (hw share %4.0f%%)\n",
                   CN[cond],N,yw[i],yh[i], yw[i]>0?yh[i]/yw[i]*100:0);
        }
        fit(xs,yw,3,&aw[cond],&bw[cond]);
        fit(xs,yh,3,&ah[cond],&bh[cond]);
        printf("  %-4s FIT: wall = %6.3f + %.5f*N us   |   hardware = %6.3f + %.5f*N us\n",
               CN[cond],aw[cond],bw[cond],ah[cond],bh[cond]);
    }
    printf("\n  LOAD contenders sustained %.2f GB/s of DRAM traffic\n",cpu_load);
    printf("\n  INFLATION UNDER DRAM LOAD (LOAD / SPIN):\n");
    printf("    fixed  a:  wall %5.2fx   hardware %5.2fx\n", aw[0]>0?aw[1]/aw[0]:0, ah[0]>0?ah[1]/ah[0]:0);
    printf("    stream b:  wall %5.2fx   hardware %5.2fx\n", bw[0]>0?bw[1]/bw[0]:0, bh[0]>0?bh[1]/bh[0]:0);
    double hw_frac_of_a = ah[0]>0 && aw[0]>0 ? ah[0]/aw[0] : 0;
    printf("\n  Of the %.2f us fixed cost, %.2f us (%.0f%%) is kernel-reported HARDWARE time;\n",
           aw[0], ah[0], hw_frac_of_a*100);
    printf("  the remaining %.2f us is host/submit path.\n", aw[0]-ah[0]);
    double dw = aw[1]-aw[0], dh = ah[1]-ah[0];
    printf("\n  The fixed cost grows %+.2f us under load, of which %+.2f us (%.0f%%) is hardware and\n"
           "  %+.2f us (%.0f%%) is host.  => %s\n", dw, dh, dw!=0?dh/dw*100:0, dw-dh, dw!=0?(dw-dh)/dw*100:0,
           (dw!=0 && dh/dw > 0.6) ? "ON THE NPU — target the small NPU-read buffers (regcmd/descriptor/A/C -> SRAM)"
         : (dw!=0 && dh/dw < 0.4) ? "ON THE HOST — target the submit path (precompiled regcmd, fewer per-task host writes)"
                                  : "SPLIT between host and hardware — both levers apply");
    for(int i=0;i<3;i++) ork_mm_free(c,W[i]);
    ork_npu_free(c); return 0;
}
