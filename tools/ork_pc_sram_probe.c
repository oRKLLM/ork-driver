/* ork_pc_sram_probe — validate + TIME the precompiled/doorbell submit (ork_pc_*, the "static regcmd table")
 * with buffers in DRAM vs on-chip NPU SRAM, sweeping chain length S, OPTIONALLY under a CPU DRAM-bandwidth
 * antagonist. The antagonist (nant streaming-memcpy threads pinned to the non-measurement cores) mimics the
 * tiered decode's CPU int4/NF4 bulk saturating LPDDR while the NPU runs its share — the scenario where
 * SRAM-resident control structures would be ISOLATED from DRAM contention (decode-cpu-npu-partition thesis).
 * Without the antagonist SRAM is latency-neutral (no contention to isolate from); the question is whether it
 * stays flat while DRAM's tail blows up UNDER load. Measurement thread is pinned to its own core so the
 * effect measured is memory contention, not CPU-core contention. Reports min/p50/p99/mean us/tok per S for
 * DRAM and SRAM, the SRAM-vs-DRAM mean/p99 delta, and the aggregate antagonist bandwidth (GB/s).
 *   make ork_pc_sram_probe && sudo env ORK_MM_TIMEOUT=4000 timeout 300 ./ork_pc_sram_probe [iters=200] [nant=6]
 * (NPU op; run alone otherwise; set CPU+DDR governors to performance first.)
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
static int8_t a8(unsigned s){ return (int8_t)(((int)((s>>16)%7))-3); }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return (x>y)-(x<y); }

/* ---- CPU DRAM-bandwidth antagonist ---- */
#define MEAS_CPU 4                 /* measurement thread core (big A76); antagonists avoid it */
static volatile int g_stop=0;
static size_t g_bytes[16]; static int g_cpu[16];
static void pin(int cpu){ cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu,&s); pthread_setaffinity_np(pthread_self(),sizeof s,&s); }
static void* ant_fn(void*arg){ long id=(long)arg; pin(g_cpu[id]);
    size_t SZ=(size_t)32<<20; char*a=malloc(SZ),*b=malloc(SZ); if(!a||!b) return NULL;
    memset(a,1,SZ); memset(b,2,SZ); size_t n=0;
    while(!g_stop){ memcpy(b,a,SZ); memcpy(a,b,SZ); n+=(size_t)2*SZ; }
    g_bytes[id]=n; free(a); free(b); return NULL; }

/* time ork_pc_run: warm, then min/p50/p99/mean over iters into stat[4]. Returns 0 ok, -1 incomplete. */
static int time_pc(ork_pc_chain*pc,int S,int warm,int iters,double stat[4]){
    for(int i=0;i<warm;i++) if(ork_pc_run(pc)<S-1) return -1;
    double*ts=malloc((size_t)iters*sizeof(double)), tot=0;
    for(int i=0;i<iters;i++){ double t0=now_us(); int r=ork_pc_run(pc); double dt=now_us()-t0;
        if(r<S-1){ free(ts); return -1; } ts[i]=dt; tot+=dt; }
    qsort(ts,iters,sizeof(double),cmp_d);
    stat[0]=ts[0]; stat[1]=ts[iters/2]; stat[2]=ts[(int)(iters*0.99)]; stat[3]=tot/iters;
    free(ts); return 0;
}
static int verify(const int32_t*O,const int32_t*ref,int S,int N){
    for(int i=0;i<S;i++) if(memcmp(O+(size_t)i*N,ref,(size_t)N*sizeof(int32_t))) return i;
    return -1;
}

int main(int argc,char**argv){
    int iters=argc>1?atoi(argv[1]):200, nant=argc>2?atoi(argv[2]):0, K=512, N=512, warm=16;   /* antagonist OFF by default: the 6-thread all-core version hard-wedged the board */
    int Sset[]={8,16,32,48,64,96}, nS=(int)(sizeof Sset/sizeof Sset[0]), Smax=96;
    setvbuf(stdout,0,_IONBF,0);
    pin(MEAS_CPU);   /* measurement thread on its own big core */
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    size_t total=ork_npu_sram_total(c);
    printf("ork_pc_sram_probe SWEEP: K=%d N=%d iters=%d warm=%d nant=%d | SRAM total=%zu KiB | meas_cpu=%d\n",
           K,N,iters,warm,nant,total>>10,MEAS_CPU);
    if(!total){ printf("  0 SRAM — abort.\n"); ork_npu_free(c); return 1; }

    /* antagonist: nant streaming-memcpy threads pinned across cores 0..7 skipping MEAS_CPU */
    pthread_t ath[16]; int na=0;
    if(nant>0){ if(nant>4)nant=4;   /* A55 little cores (0-3) ONLY — the earlier 6-thread all-core antagonist starved the big-core NPU host/driver threads and hard-wedged the board */
        for(int i=0;i<nant;i++){ g_cpu[i]=i%4;
            if(!pthread_create(&ath[i],0,ant_fn,(void*)(long)i)) na++; }
        struct timespec r={0,300*1000*1000}; nanosleep(&r,0);   /* let bandwidth ramp before measuring */
        printf("  antagonist: %d streaming-memcpy threads on cores {", na);
        for(int i=0;i<na;i++) printf("%d%s",g_cpu[i],i<na-1?",":""); printf("}\n");
    }
    double ant_t0=now_us();

    int8_t*A=malloc(K); for(int k=0;k<K;k++) A[k]=a8((unsigned)(0x9e37u+k*2654435761u));
    int8_t*B=malloc((size_t)K*N); for(size_t j=0;j<(size_t)K*N;j++) B[j]=a8((unsigned)(0x85ebu+j*40503u));
    ork_w*w=ork_mm_pack_i8(c,K,N,B); if(!w){printf("pack fail\n");return 1;}
    int32_t*ref=malloc((size_t)N*4); for(int n=0;n<N;n++){ long acc=0; for(int k=0;k<K;k++) acc+=(long)A[k]*(long)B[(size_t)k*N+n]; ref[n]=(int32_t)acc; }
    free(B);

    int32_t*Od=(int32_t*)ork_dma_alloc(c,(size_t)Smax*N*4);
    int32_t*Os=(int32_t*)ork_dma_alloc_sram(c,(size_t)Smax*N*4);
    if(!Od||!Os){printf("out alloc fail\n");return 1;}
    ork_mm_task_i8*td=malloc(sizeof(*td)*Smax),*ts=malloc(sizeof(*ts)*Smax);
    for(int i=0;i<Smax;i++){ td[i].w=ts[i].w=w; td[i].M=ts[i].M=1; td[i].A=ts[i].A=A; td[i].C=Od+(size_t)i*N; ts[i].C=Os+(size_t)i*N; }

    printf("\n  S  | DRAM  min / p50 / p99 / mean (us/tok) | SRAM  min / p50 / p99 / mean | mean d | p99 d\n");
    printf("  ---+---------------------------------------+------------------------------+--------+-------\n");
    for(int si=0; si<nS; si++){ int S=Sset[si];
        setenv("ORK_PC_NO_SRAM","1",1);   /* DRAM arm */
        ork_pc_chain*pcd=ork_pc_compile(c,S,td); if(!pcd){printf("  %3d: DRAM compile fail\n",S); continue;}
        double sd[4]; int rd=time_pc(pcd,S,warm,iters,sd); int bd=verify(Od,ref,S,N); ork_pc_free(pcd);
        unsetenv("ORK_PC_NO_SRAM");   /* SRAM arm (the new default) */
        size_t fb=ork_npu_sram_free(c);
        ork_pc_chain*pcs=ork_pc_compile(c,S,ts);
        size_t fa=ork_npu_sram_free(c); long used=(long)fb-(long)fa;
        if(!pcs){ printf("  %3d: SRAM compile fail\n",S); continue; }
        if(used<=0){ printf("  %3d: SRAM over budget (used %ldKiB) skip\n",S,used>>10); ork_pc_free(pcs); continue; }
        double ss[4]; int rs=time_pc(pcs,S,warm,iters,ss); int bs=verify(Os,ref,S,N); ork_pc_free(pcs);
        if(rd||rs||bd>=0||bs>=0){ printf("  %3d: FAIL (dram op=%d rc=%d, sram op=%d rc=%d)\n",S,bd,rd,bs,rs); continue; }
        printf("  %3d | %6.0f/%6.0f/%6.0f/%6.0f          | %6.0f/%6.0f/%6.0f/%6.0f     | %+5.1f%% | %+5.1f%%\n",
               S, sd[0],sd[1],sd[2],sd[3], ss[0],ss[1],ss[2],ss[3],
               100.0*(ss[3]-sd[3])/sd[3], 100.0*(ss[2]-sd[2])/sd[2]);
    }
    double ant_dt=now_us()-ant_t0;

    g_stop=1; size_t antb=0; for(int i=0;i<na;i++){ pthread_join(ath[i],0); antb+=g_bytes[i]; }
    if(na) printf("\n  antagonist aggregate: %.1f GB/s sustained over %.1fs (%d threads)\n",
                  antb/1e9/(ant_dt/1e6), ant_dt/1e6, na);
    printf("  (mean/p99 d: SRAM vs DRAM, negative = SRAM faster. Under load, watch if DRAM tail (mean>>min)\n");
    printf("   grows while SRAM stays flat = the partition win; flat both = SRAM still neutral under contention.)\n");
    ork_npu_free(c); return 0;
}
