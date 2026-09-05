/* sram_port_probe — is on-chip SRAM a SEPARATE memory port from DRAM, measured with the NPU IN THE LOOP?
 *
 * THE QUESTION.  If the NPU streaming weights out of SRAM does not spend DRAM's bandwidth budget, then a
 * CPU/NPU decode partition can run both engines at once for a real AGGREGATE gain. Two earlier attempts
 * each answered half of it and neither is airtight:
 *
 *   test_sram_pipe   (2026-08-11)  CPU->SRAM held 104% while 3 big cores saturated DRAM => separate port
 *                                  CONFIRMED -- but the SRAM consumer was the CPU, not the NPU.
 *   sram_bw_probe    (2026-09-05)  NPU weight in SRAM under a CPU DRAM hog => flat -- but the NPU side was
 *                                  NOT weight-bandwidth-bound, so weight PLACEMENT could not have mattered.
 *
 * Why that second probe was void, in numbers: it defaulted to K=512 N=256, a 128 KiB weight, one submit per
 * iteration. 128 KiB at ~11 GB/s is ~12 us of streaming against a ~167 us per-submit floor -- ~93% of every
 * iteration was host + fixed submit cost. Moving 12 us of traffic between two memories cannot show up.
 *
 * THE FIX, and the thing that makes this one airtight: drive the NPU into a regime that is DEMONSTRABLY
 * weight-bandwidth-bound, and PROVE it in-run before believing any contention number.
 *
 *   (a) Make the weight as large as SRAM can hold (~640 KiB of the ~700 KiB usable) so each stream is ~58 us.
 *   (b) HW-chain T tasks in ONE submit, all re-reading that same weight (640 KiB > the 384 KB CBUF, so it
 *       cannot stay resident -- every task re-streams). The ~48 us fixed submit cost amortizes over T, and
 *       per-task cost becomes the stream itself.
 *   (c) PHASE 1 IS A HARD GATE. Sweep the weight size and fit time = a + b*N. Bandwidth-bound means time is
 *       ~linear in N with a small intercept; overhead-bound means it is ~flat. The run reports the streaming
 *       fraction b*N/(a+b*N) and DECLARES ITSELF VOID below ORK_MIN_FRAC (default 0.70) instead of printing
 *       a contention verdict it has not earned. This gate is precisely what both earlier probes lacked.
 *
 * Then PHASE 2 measures the 2x2 -- weight in {DRAM, SRAM} x {solo, under a 4-core DRAM hog} -- interleaved
 * A-B-A over several trials, because this board carries ~10% run-to-run spread and two separate SRAM
 * investigations have already produced a "win" that vanished on replication.
 *
 * READING IT.  Separate port => the SRAM row keeps ~100% of its solo rate under the hog while the DRAM row
 * drops, and aggregate (NPU + CPU) is higher for SRAM. Shared port => both rows degrade alike.
 *
 *   make sram_port_probe && sudo tools/util/npu_guard.sh -- ./sram_port_probe [K] [Nmax] [T] [trials]
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

/* ---- Contenders on cores 5,6,7 (main/NPU thread owns core 4) ----
 *
 * TWO modes, and the SPIN mode is what makes the memory claim separable. The NPU submit path is itself
 * host-CPU-bound (~95us of host work per submit), so a contender that merely occupies big cores slows the
 * NPU down with no memory involvement whatsoever. Measuring only against a memcpy hog cannot tell that
 * apart from real bus contention -- the first version of this probe saw NPU 20.4 -> 17.4 GB/s under load
 * and could not say whether one byte of it was memory.
 *
 * So: LOAD = memcpy (scheduling + DRAM traffic), SPIN = pure ALU (scheduling only, zero traffic), the two
 * pinned identically and run for the same duration. The MEMORY-ATTRIBUTABLE slowdown is then
 * rate_under_LOAD / rate_under_SPIN, with the scheduling term divided out. That ratio, compared between a
 * DRAM-resident and an SRAM-resident weight, is the airtight separate-port test.
 */
#define NCONT 3
static volatile int    g_stop = 0;
static volatile double g_cpu_gbps[NCONT];
static double g_cpu_solo = 0;   /* the saturator's calibrated solo rate (LOAD mode, all contenders) */

typedef struct { int core, load; } cont_arg;

static void *contender(void *arg){
    cont_arg *ca=(cont_arg*)arg;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(ca->core,&s); sched_setaffinity(0,sizeof s,&s);
    if(!ca->load){                                   /* SPIN: occupy the core, touch no memory */
        volatile double x=1.0; double t0=now_us();
        while(!g_stop){ for(int i=0;i<10000;i++) x = x*1.000001 + 1e-9; }
        (void)t0; g_cpu_gbps[ca->core-5]=0; return NULL;
    }
    size_t BYTES = 64u<<20;                          /* 64 MiB per thread >> 3 MiB L3 */
    char *a=malloc(BYTES), *b=malloc(BYTES);
    if(!a||!b){ free(a); free(b); return NULL; }
    memset(a,1,BYTES); memset(b,2,BYTES);
    double t0=now_us(); size_t moved=0;
    while(!g_stop){
        memcpy(b,a,BYTES);
        /* The destination is never read again, so at -O3 the compiler may delete this memcpy outright --
         * and DID, in the first run of this probe: the hog "reached" 7e6 GB/s (spinning on clock_gettime
         * alone), generated no traffic, and every cell dutifully reported 99.9% retention of nothing. */
        __asm__ __volatile__("" : : "r"(a), "r"(b) : "memory");
        moved+=BYTES;
        double dt=(now_us()-t0)/1e6; if(dt>0) g_cpu_gbps[ca->core-5] = 2.0*moved/1e9/dt;
    }
    free(a); free(b); return NULL;
}

/* Start NCONT contenders; returns via *out the summed GB/s after they are stopped. */
static void conts_start(pthread_t*th, cont_arg*ca, int load){
    g_stop=0; for(int i=0;i<NCONT;i++){ g_cpu_gbps[i]=0; ca[i].core=5+i; ca[i].load=load;
        pthread_create(&th[i],0,contender,&ca[i]); }
}
static double conts_stop(pthread_t*th){
    double sum=0; for(int i=0;i<NCONT;i++) sum+=g_cpu_gbps[i];
    g_stop=1; for(int i=0;i<NCONT;i++) pthread_join(th[i],0); return sum;
}

/* ---- NPU: T chained tasks per submit, every task re-streaming the same K*N weight ---- */
typedef struct { ork_npu*c; ork_mm_task_i8*tk; int T; } npu_ctx;
static double npu_gbps(npu_ctx*n, int K, int N, int iters, int *err){
    double t0=now_us();
    for(int i=0;i<iters;i++){ int r=ork_i8_mm_run_chain(n->c,n->T,n->tk); if(r<0){ *err=r; return 0; } }
    double dt=(now_us()-t0)/1e6;
    return (double)iters*n->T*(double)K*N/1e9/dt;      /* useful weight bytes streamed / s */
}

/* Pack a weight, optionally requesting on-chip SRAM. Returns 1 iff SRAM was actually granted. */
static int pack_w(ork_npu*c, int K, int N, const int8_t*B, int want_sram, ork_w**out, size_t*consumed){
    size_t f0 = ork_npu_sram_free(c);
    if(want_sram) setenv("ORK_WEIGHT_SRAM","1",1); else unsetenv("ORK_WEIGHT_SRAM");
    *out = ork_i8_mm_pack(c,K,N,B);
    unsetenv("ORK_WEIGHT_SRAM");
    if(!*out) return -1;
    size_t f1 = ork_npu_sram_free(c);
    *consumed = (f0>f1) ? (f0-f1) : 0;
    return (*consumed >= (size_t)K*N);                /* the WHOLE tile landed; 3/4 let a partial
                                                      * placement pass as success (124 KiB of a
                                                      * 160 KiB request) and faked an SRAM arm. */
}

int main(int argc,char**argv){
    int K      = argc>1?atoi(argv[1]):1024;
    int NMAXW  = argc>2?atoi(argv[2]):640;            /* biggest weight (KiB == N when K=1024) */
    int T      = argc>3?atoi(argv[3]):16;             /* chained tasks per submit */
    int TRIALS = argc>4?atoi(argv[4]):3;
    double MINFRAC = getenv("ORK_MIN_FRAC")?atof(getenv("ORK_MIN_FRAC")):0.70;
    setvbuf(stdout,0,_IONBF,0);
    { cpu_set_t s4; CPU_ZERO(&s4); CPU_SET(4,&s4); sched_setaffinity(0,sizeof s4,&s4); }  /* NPU host thread owns core 4; contenders take 5,6,7 */

    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    size_t tot=ork_npu_sram_total(c), fr=ork_npu_sram_free(c);
    printf("sram_port_probe: K=%d Nmax=%d T=%d trials=%d | SRAM total=%zuKiB free=%zuKiB\n",
           K,NMAXW,T,TRIALS,tot>>10,fr>>10);
    if(tot==0){ printf("FATAL: no NPU SRAM on this kernel/DTB — nothing to test.\n"); return 2; }
    if((size_t)K*NMAXW > fr){ printf("FATAL: weight %zuKiB > free SRAM %zuKiB\n",((size_t)K*NMAXW)>>10,fr>>10); return 2; }

    int8_t*B=malloc((size_t)K*NMAXW); for(size_t i=0;i<(size_t)K*NMAXW;i++) B[i]=(int8_t)(i&0x3f);
    int8_t*A=(int8_t*)ork_dma_alloc(c,(size_t)K); memset(A,1,(size_t)K);
    int32_t*Cb=(int32_t*)ork_dma_alloc(c,(size_t)T*NMAXW*4);
    ork_mm_task_i8 *tk=calloc(T,sizeof *tk);

    /* ================= PHASE 1 — VALIDITY GATE: is the NPU weight-bandwidth-bound here? ================= */
    printf("\n[1] VALIDITY GATE — weight-size scaling (DRAM-resident, solo). Bandwidth-bound => time ~linear in N.\n");
    int   NS[3] = { NMAXW/4, NMAXW/2, NMAXW };
    double us_per_task[3], gb[3];
    for(int i=0;i<3;i++){
        int N=NS[i]; if(N%32){ printf("  N=%d not %%32 — adjust Nmax\n",N); return 2; }
        ork_w*w=NULL; size_t cons=0;
        if(pack_w(c,K,N,B,0,&w,&cons)<0){ printf("  pack fail N=%d\n",N); return 2; }
        for(int t=0;t<T;t++){ tk[t].w=w; tk[t].M=1; tk[t].A=A; tk[t].C=Cb+(size_t)t*N; tk[t].cstride=0; }
        npu_ctx n={c,tk,T}; int err=0;
        npu_gbps(&n,K,N,3,&err);                                    /* warm */
        if(err){ printf("  chain refused (%d) at N=%d\n",err,N); return 2; }
        double g=npu_gbps(&n,K,N,60,&err);
        gb[i]=g; us_per_task[i] = (double)K*N/1e9/g*1e6;
        printf("  N=%-5d weight=%4zuKiB   %7.2f us/task   %6.2f GB/s\n", N,((size_t)K*N)>>10,us_per_task[i],g);
        ork_mm_free(c,w);
    }
    /* time = a + b*N  from the endpoints; the midpoint is the linearity check. */
    double b=(us_per_task[2]-us_per_task[0])/(double)(NS[2]-NS[0]);
    double a=us_per_task[0]-b*NS[0];
    double frac = (b*NS[2])/(a+b*NS[2]);
    double mid_pred=a+b*NS[1], mid_err=(us_per_task[1]-mid_pred)/mid_pred;
    printf("  fit: time = %.2f us + %.4f us/N  ->  streaming fraction at N=%d: %.1f%%  (linearity err at midpoint %+.1f%%)\n",
           a,b,NS[2],frac*100,mid_err*100);
    if(frac < MINFRAC){
        printf("\n  VOID — only %.1f%% of the time is weight streaming (need >=%.0f%%). The NPU is still\n"
               "  overhead-bound, so weight PLACEMENT cannot move the result and any contention number below\n"
               "  would be meaningless. This is the same defect that voided sram_bw_probe. Raise T or Nmax.\n",
               frac*100, MINFRAC*100);
        return 3;
    }
    printf("  GATE PASSED — the NPU side is weight-bandwidth-bound, so placement is now observable.\n");

    /* ================= PHASE 2 — the 2x2 contention matrix ================= */
    int N=NMAXW;
    ork_w *wd=NULL,*ws=NULL; size_t cd=0,cs=0;
    if(pack_w(c,K,N,B,0,&wd,&cd)<0){ printf("pack DRAM fail\n"); return 2; }
    int got=pack_w(c,K,N,B,1,&ws,&cs);
    if(got<0){ printf("pack SRAM fail\n"); return 2; }
    printf("\n[2] PLACEMENT: DRAM weight consumed %zuKiB of SRAM (expect 0); SRAM weight consumed %zuKiB of %zuKiB requested -> %s\n",
           cd>>10, cs>>10, ((size_t)K*N)>>10, got?"IN SRAM":"*** NOT IN SRAM (failed over to DRAM) ***");
    if(!got){ printf("  FATAL: the SRAM request did not land; the A/B would compare DRAM against DRAM.\n"); return 2; }

    /* Calibrate the LOAD contenders alone. If they do not reach a plausible DRAM rate there is no
     * contention to retain against, and the matrix below would be measuring nothing (see the -O3 note). */
    { pthread_t th[NCONT]; cont_arg ca[NCONT]; conts_start(th,ca,1);
      struct timespec s1={1,0}; nanosleep(&s1,0);
      double solo=0; for(int i=0;i<NCONT;i++) solo+=g_cpu_gbps[i];
      conts_stop(th); g_cpu_solo=solo;
      printf("\n[3] SATURATOR CHECK — %d LOAD contenders on cores 5-7: %.2f GB/s of DRAM traffic\n",NCONT,solo);
      if(solo < 4.0 || solo > 80.0){
          printf("  FATAL: implausible — the contenders are not generating real DRAM traffic, so there is no\n"
                 "  contention to measure. Refusing to print a verdict.\n"); return 4; } }

    printf("\n[4] MATRIX — %d trials, A-B-A interleaved. SPIN divides out the scheduling term.\n",TRIALS);
    printf("    (NPU GB/s = weight bytes streamed; memory-attributable = LOAD/SPIN)\n");
    double mem_d=0, mem_s=0, ret_d=0, ret_s=0, agg_d=0, agg_s=0;
    for(int t=0;t<TRIALS;t++){
        for(int p=0;p<2;p++){
            int sram = (t%2==0) ? p : !p;                 /* alternate order each trial */
            ork_w *w = sram?ws:wd;
            for(int i=0;i<T;i++){ tk[i].w=w; tk[i].M=1; tk[i].A=A; tk[i].C=Cb+(size_t)i*N; tk[i].cstride=0; }
            npu_ctx n={c,tk,T}; int err=0;
            npu_gbps(&n,K,N,3,&err);
            double solo=npu_gbps(&n,K,N,50,&err);

            pthread_t th[NCONT]; cont_arg ca[NCONT];
            struct timespec ss={0,300*1000*1000};
            conts_start(th,ca,0); nanosleep(&ss,0);
            double v_spin=npu_gbps(&n,K,N,50,&err); conts_stop(th);
            conts_start(th,ca,1); nanosleep(&ss,0);
            double v_load=npu_gbps(&n,K,N,50,&err); double cpu=conts_stop(th);

            double memr = v_spin>0 ? v_load/v_spin*100 : 0;   /* scheduling divided out */
            printf("  t%d %-4s | solo %6.2f | SPIN %6.2f (%5.1f%%) | LOAD %6.2f (%5.1f%%) | MEMORY-ONLY %6.1f%% | CPU %5.2f  agg %6.2f\n",
                   t, sram?"SRAM":"DRAM", solo, v_spin, solo>0?v_spin/solo*100:0, v_load, solo>0?v_load/solo*100:0,
                   memr, cpu, v_load+cpu);
            if(sram){ mem_s+=memr; ret_s+=(solo>0?v_load/solo*100:0); agg_s+=v_load+cpu; }
            else    { mem_d+=memr; ret_d+=(solo>0?v_load/solo*100:0); agg_d+=v_load+cpu; }
        }
    }
    mem_d/=TRIALS; mem_s/=TRIALS; ret_d/=TRIALS; ret_s/=TRIALS; agg_d/=TRIALS; agg_s/=TRIALS;
    printf("\n  MEAN  DRAM weight: raw retention %.1f%%  MEMORY-ONLY %.1f%%  aggregate %.2f GB/s\n",ret_d,mem_d,agg_d);
    printf("  MEAN  SRAM weight: raw retention %.1f%%  MEMORY-ONLY %.1f%%  aggregate %.2f GB/s\n",ret_s,mem_s,agg_s);
    printf("  DELTA (SRAM - DRAM), memory-attributable: %+.1f pp\n", mem_s-mem_d);
    printf("\n  VERDICT: %s\n",
        (mem_s-mem_d > 5.0) ? "SEPARATE PORT for NPU weight reads — SRAM placement measurably relieves DRAM contention."
      : (mem_d > 95.0 && mem_s > 95.0) ? "NEITHER placement contends — the NPU weight stream does not compete with CPU DRAM traffic at this rate, so there is no contention for SRAM to relieve. The earlier drop was SCHEDULING, not memory."
      : "NO separate-port benefit for NPU weight reads — both placements lose the same memory-attributable share.");

    /* ===== PHASE 5 — WHERE does the contention land: the weight stream, or the DRAM-resident remainder?
     *
     * Phase 4's null is the practical answer (SRAM placement buys nothing), but it does not by itself prove
     * the NPU's weight read contends. Each task is time = a + b*N: a fixed ~8.5us part (A reads, C writes,
     * regcmd/descriptor fetches -- all in DRAM in BOTH arms) plus the b*N weight stream. A 33% loss is
     * equally consistent with "the stream contends" and with "the stream never contended and the whole loss
     * is the fixed DRAM remainder". Sweeping N separates them:
     *
     *   stream immune (g=1): retention RISES with N   -- predicted ~48% at N=160 vs ~67% at N=640
     *   stream contends (f=g): retention is FLAT in N
     */
    ork_mm_free(c,wd); ork_mm_free(c,ws); wd=ws=NULL;   /* Phase 4 held 640 KiB of the pool; Phase 5 re-packs per N */
    printf("\n[5] LOCALISING THE CONTENTION — retention vs weight size (memory-attributable, LOAD/SPIN)\n");
    for(int i=0;i<3;i++){
        int Nv=NS[i];
        for(int p=0;p<2;p++){
            ork_w*w=NULL; size_t cons=0;
            int g2=pack_w(c,K,Nv,B,p,&w,&cons);
            if(g2<0){ printf("  pack fail\n"); break; }
            if(p && !g2){ printf("  N=%-5d SRAM: request did not land — skipped\n",Nv); ork_mm_free(c,w); continue; }
            for(int j=0;j<T;j++){ tk[j].w=w; tk[j].M=1; tk[j].A=A; tk[j].C=Cb+(size_t)j*Nv; tk[j].cstride=0; }
            npu_ctx n={c,tk,T}; int err=0; npu_gbps(&n,K,Nv,3,&err);
            pthread_t th[NCONT]; cont_arg ca[NCONT]; struct timespec ss={0,300*1000*1000};
            conts_start(th,ca,0); nanosleep(&ss,0); double vs=npu_gbps(&n,K,Nv,40,&err); conts_stop(th);
            conts_start(th,ca,1); nanosleep(&ss,0); double vl=npu_gbps(&n,K,Nv,40,&err); conts_stop(th);
            double strf = (0.0365*Nv)/(8.55+0.0365*Nv);
            printf("  N=%-5d %-4s  stream-frac %4.0f%%   SPIN %6.2f  LOAD %6.2f  -> memory-only retention %5.1f%%\n",
                   Nv, p?"SRAM":"DRAM", strf*100, vs, vl, vs>0?vl/vs*100:0);
            ork_mm_free(c,w);
        }
    }
    printf("  Reading it: retention RISING with N => the weight stream is NOT what contends (the fixed\n"
           "  DRAM-resident remainder is). Retention FLAT in N => the weight stream itself contends.\n");

    if(wd) ork_mm_free(c,wd); if(ws) ork_mm_free(c,ws); ork_npu_free(c);
    return 0;
}
