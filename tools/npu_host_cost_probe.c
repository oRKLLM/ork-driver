/* npu_host_cost_probe — attack the fixed per-task cost, now that we know it is HOST work.
 *
 * npu_fixed_cost_probe split the chained int8 matmul's per-task cost time = a + b*N and found:
 *
 *     fixed a = 8.66 us, of which only 1.27 us (15%) is kernel-reported HARDWARE time
 *     under a 23.5 GB/s CPU DRAM load a inflates 2.23x -- and 113% of that growth is HOST-side
 *     (kernel hardware time for the fixed part is FLAT; only the weight stream b moves, 1.26x)
 *
 * So the contention cost that dominates this path is the host synthesising ~1-2 KB of regcmd per task and
 * flushing it, not anything the NPU does. That is exactly what the precompiled regcmd cache (ork_pc_*)
 * removes: it builds the program pool ONCE with A/C addresses baked in, and per call only refreshes A and
 * submits. Its measured value so far (1.47x on the FFN layer) was taken on a QUIET system -- where the host
 * term is 7.39 us. Under load that term is 19.4 us, so the cache should be worth substantially more exactly
 * when something else is using the memory system, i.e. in the CPU||NPU partition it was never credited for.
 *
 * FAIR COMPARISON. ork_i8_mm_run_chain routes M=1/K%512/Sn==1 chains onto the doorbell spine with nc=0 =
 * ALL THREE CORES, while ork_pc_run submits with core_mask=1 = one core. Comparing them as-is would confound
 * precompilation with core count, so the chain arm is pinned with ORK_CHAIN_NC=1.
 *
 *   make npu_host_cost_probe && sudo tools/util/npu_guard.sh -- ./npu_host_cost_probe [K] [T] [reps]
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
    FILE*f=fopen(p,"r"); if(!f) return 0ULL; unsigned long long v=0; if(fscanf(f,"%llu",&v)!=1) v=0; fclose(f); return v; }
#define P_HWNS "/sys/module/rknpu/parameters/hw_ns_sum"
#define P_HWN  "/sys/module/rknpu/parameters/hw_n"

#define NCONT 3
static volatile int g_stop=0; static volatile double g_gbps[NCONT];
typedef struct { int core, load; } cont_arg;
static void *contender(void*arg){
    cont_arg*ca=(cont_arg*)arg; cpu_set_t s; CPU_ZERO(&s); CPU_SET(ca->core,&s); sched_setaffinity(0,sizeof s,&s);
    if(!ca->load){ volatile double x=1.0; while(!g_stop){ for(int i=0;i<10000;i++) x=x*1.000001+1e-9; } return NULL; }
    size_t B=64u<<20; char*a=malloc(B),*b=malloc(B);
    if(!a||!b){ free(a); free(b); return NULL; }
    memset(a,1,B); memset(b,2,B); double t0=now_us(); size_t mv=0;
    while(!g_stop){ memcpy(b,a,B); __asm__ __volatile__("" : : "r"(a),"r"(b) : "memory");
        mv+=B; double dt=(now_us()-t0)/1e6; if(dt>0) g_gbps[ca->core-5]=2.0*mv/1e9/dt; }
    free(a); free(b); return NULL; }
static void cstart(pthread_t*th,cont_arg*ca,int load){ g_stop=0;
    for(int i=0;i<NCONT;i++){ g_gbps[i]=0; ca[i].core=5+i; ca[i].load=load; pthread_create(&th[i],0,contender,&ca[i]); } }
static double cstop(pthread_t*th){ double s=0; for(int i=0;i<NCONT;i++) s+=g_gbps[i];
    g_stop=1; for(int i=0;i<NCONT;i++) pthread_join(th[i],0); return s; }
static void fit(const double*x,const double*y,int n,double*a,double*b){
    double sx=0,sy=0,sxy=0,sxx=0; for(int i=0;i<n;i++){ sx+=x[i]; sy+=y[i]; sxy+=x[i]*y[i]; sxx+=x[i]*x[i]; }
    *b=(n*sxy-sx*sy)/(n*sxx-sx*sx); *a=(sy-*b*sx)/n; }

int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):1024, T=argc>2?atoi(argv[2]):16, REP=argc>3?atoi(argv[3]):3;
    setvbuf(stdout,0,_IONBF,0);
    { cpu_set_t s4; CPU_ZERO(&s4); CPU_SET(4,&s4); sched_setaffinity(0,sizeof s4,&s4); }
    setenv("ORK_CHAIN_NC","1",1);                 /* pin the chain arm to one core, matching ork_pc */

    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    int NS[3]={160,320,640};
    int8_t*B=malloc((size_t)K*640); for(size_t i=0;i<(size_t)K*640;i++) B[i]=(int8_t)(i&0x3f);
    int8_t*A=(int8_t*)ork_dma_alloc(c,(size_t)K); memset(A,1,(size_t)K);
    int32_t*Cb=(int32_t*)ork_dma_alloc(c,(size_t)T*640*4);
    ork_mm_task_i8*tk=calloc(T,sizeof *tk);
    ork_w*W[3]; for(int i=0;i<3;i++){ W[i]=ork_i8_mm_pack(c,K,NS[i],B); if(!W[i]){printf("pack fail\n");return 2;} }

    printf("npu_host_cost_probe: K=%d T=%d reps=%d single-core both arms (ORK_CHAIN_NC=1)\n",K,T,REP);
    double aw[2][2],bw[2][2],ah[2][2],hostc[2][2];  /* [mode][cond] : 0=chain 1=pc ; 0=SPIN 1=LOAD */
    const char*MN[2]={"chain","pc   "}, *CN[2]={"SPIN","LOAD"};
    double cpu_load=0; int pc_ok=1;

    for(int mode=0;mode<2;mode++){
        for(int cond=0;cond<2;cond++){
            double xs[3],yw[3],yh[3],hs=0;
            for(int i=0;i<3;i++){
                int N=NS[i];
                for(int j=0;j<T;j++){ tk[j].w=W[i]; tk[j].M=1; tk[j].A=A; tk[j].C=Cb+(size_t)j*N; tk[j].cstride=0; }
                ork_pc_chain *pc=NULL;
                if(mode==1){ pc=ork_pc_compile(c,T,tk);
                    if(!pc){ printf("  pc   N=%-4d  ork_pc_compile REFUSED this shape — arm unavailable\n",N); pc_ok=0; break; } }
                double wsum=0,hsum=0;
                for(int r=0;r<REP;r++){
                    for(int k=0;k<3;k++){ if(mode) ork_pc_run(pc); else ork_i8_mm_run_chain(c,T,tk); }
                    pthread_t th[NCONT]; cont_arg ca[NCONT]; cstart(th,ca,cond);
                    struct timespec ss={0,300*1000*1000}; nanosleep(&ss,0);
                    unsigned long long ns0=rd_u64(P_HWNS),n0=rd_u64(P_HWN);
                    double t0=now_us(); int IT=40;
                    for(int k=0;k<IT;k++){ if(mode) ork_pc_run(pc); else ork_i8_mm_run_chain(c,T,tk); }
                    double wall=now_us()-t0;
                    unsigned long long ns1=rd_u64(P_HWNS),n1=rd_u64(P_HWN);
                    double g=cstop(th); if(cond==1) cpu_load=g;
                    wsum+=wall/(IT*T);
                    hsum+=(n1>n0)?((double)(ns1-ns0)/1e3)/((double)(n1-n0)*T):0.0;
                }
                if(pc) ork_pc_free(pc);
                xs[i]=N; yw[i]=wsum/REP; yh[i]=hsum/REP; hs+=yw[i]-yh[i];
                printf("  %s %-4s N=%-4d  wall %7.3f us/task   hardware %7.3f   host %7.3f\n",
                       MN[mode],CN[cond],N,yw[i],yh[i],yw[i]-yh[i]);
            }
            if(!pc_ok) break;
            hostc[mode][cond]=hs/3.0;   /* host work is per-task and N-independent -> average it */
            double bh_tmp;
            fit(xs,yw,3,&aw[mode][cond],&bw[mode][cond]);
            fit(xs,yh,3,&ah[mode][cond],&bh_tmp);
            printf("  %s %-4s FIT: wall = %6.3f + %.5f*N   (hardware intercept %6.3f)\n",
                   MN[mode],CN[cond],aw[mode][cond],bw[mode][cond],ah[mode][cond]);
        }
        if(!pc_ok) break;
    }
    if(!pc_ok){ printf("\n  pc arm unavailable — cannot compare.\n"); return 3; }

    printf("\n  LOAD contenders sustained %.2f GB/s of DRAM traffic\n",cpu_load);
    printf("\n  HOST per-task cost (wall - kernel hardware time), us — the quantity precompilation targets.\n");
    printf("  It is N-independent by construction, so it is averaged over the three weight sizes.\n\n");
    printf("            %-10s %-10s %-10s\n","quiet(SPIN)","under LOAD","inflation");
    printf("    chain   %-10.2f %-10.2f %.2fx\n",hostc[0][0],hostc[0][1],hostc[0][0]>0?hostc[0][1]/hostc[0][0]:0);
    printf("    pc      %-10.2f %-10.2f %.2fx\n",hostc[1][0],hostc[1][1],hostc[1][0]>0?hostc[1][1]/hostc[1][0]:0);
    double sq=hostc[0][0]-hostc[1][0], sl=hostc[0][1]-hostc[1][1];
    printf("\n    pc saves %.2f us/task quiet (%.0f%%) and %.2f us/task under load (%.0f%%)\n",
           sq, hostc[0][0]>0?sq/hostc[0][0]*100:0, sl, hostc[0][1]>0?sl/hostc[0][1]*100:0);
    printf("    => the ABSOLUTE saving is %.2fx larger under contention.\n", sq>0?sl/sq:0);
    printf("\n    Note the two readings differ: pc's RELATIVE sensitivity is worse (%.2fx vs %.2fx) because what\n"
           "    remains is almost purely latency-exposed (an A refresh + the submit ioctl) with no bulk work to\n"
           "    dilute it. The actionable number is the absolute saving, not the ratio.\n",
           hostc[1][0]>0?hostc[1][1]/hostc[1][0]:0, hostc[0][0]>0?hostc[0][1]/hostc[0][0]:0);
    printf("\n  VERDICT: %s\n",
      (sl > sq*1.3) ? "the precompiled cache is worth MORE under CPU DRAM load than on a quiet system — it should be\n           credited accordingly in any CPU||NPU partition, where every quiet-system benchmark understates it"
                    : "the cache's value does not grow under contention");
    for(int i=0;i<3;i++) ork_mm_free(c,W[i]);
    ork_npu_free(c); return 0;
}
