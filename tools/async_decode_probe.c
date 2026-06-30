/* tools/async_decode_probe.c — validate the decode async-overlap hypothesis on real matmuls.
 *
 * The decode wall (Exp-2026-06-30): with ORK_DECODE_MC=1 the NPU runs ~33% busy and cpu7 ~62% —
 * the synchronous backend pays wall ≈ t_cpu + t_npu (they take turns). The hypothesis is that
 * overlapping the CPU prep behind the NPU compute (async dispatch) collapses that toward
 * wall ≈ max(t_cpu, t_npu). This probe TESTS that directly, end-to-end, on the real decode matmuls.
 *
 * Method: pack the 7 per-layer Qwen2.5-7B-Q8_0 projections (Q,K,V,O,gate,up,down) resident, M=1,
 * 3-core (the ORK_DECODE_MC config). A "token" = run all 7. Between matmuls run a REAL representative
 * CPU-prep workload (activation requant of the K-row + an RMSNorm pass — genuine float work, not a
 * sleep), sized by a sweepable multiplier. Compare:
 *   SYNC  : for each m: cpu_prep(m); ork_mm_run_i8(m)            -> wall ≈ Σ(t_cpu + t_npu)
 *   ASYNC : pipeline — launch run(m) async; cpu_prep(m+1) overlaps; wait(m) -> wall ≈ Σ max(t_cpu,t_npu)
 * Sweep the CPU/NPU ratio (multiplier) and print sync/async/speedup vs the sum→max ceiling, so the
 * curve shows what async buys at decode's measured ratio (~1.9). Bit-exact gate: async C == sync C.
 *   make async_decode_probe && sudo ./async_decode_probe [iters]
 *
 * NOTE on dependencies: a real decode layer is partly a dependency chain (matmul m+1's input needs
 * m's output + a norm), so the pipelined number here is the ACHIEVABLE CEILING for the overlap
 * mechanism — the independent groups {Q,K,V} and {gate,up} can overlap; the chain edges can't. The
 * point is to validate the mechanism and quantify the ceiling, not to claim a full decode is this fast.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

/* representative between-matmul CPU prep: requant a K-row f32->int8 (RMSNorm + scale), `reps` passes.
 * volatile sink prevents the optimizer from deleting it. Returns nothing; cost ~ reps*K. */
static volatile int32_t g_sink;
static void cpu_prep(const float *x, int8_t *out, int K, int reps){
    for(int r=0;r<reps;r++){
        float ss=0.f; for(int i=0;i<K;i++) ss+=x[i]*x[i];
        float inv=1.f/sqrtf(ss/(float)K + 1e-6f);
        float amax=1e-9f; for(int i=0;i<K;i++){ float v=x[i]*inv; float a=fabsf(v); if(a>amax)amax=a; }
        float s=127.f/amax; int32_t acc=0;
        for(int i=0;i<K;i++){ int q=(int)lrintf(x[i]*inv*s); if(q>127)q=127; if(q<-128)q=-128; out[i]=(int8_t)q; acc+=q; }
        g_sink=acc;
    }
}

int main(int argc,char**argv){
    int iters=argc>1?atoi(argv[1]):60;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int cores=ork_npu_cores(c);
    int budget=cores; { const char*e=getenv("ORK_PROBE_CORES"); if(e) budget=atoi(e); if(budget<1)budget=1; if(budget>cores)budget=cores; }
    printf("INIT OK: soc=%s cores=%d  submit-budget=%d (3=ORK_DECODE_MC uses all cores; 1=leaves %d free to overlap)\n",
           ork_npu_soc(c), cores, budget, cores-budget);
    ork_npu_set_core_budget(c, budget);

    /* Qwen2.5-7B decode projections, M=1: {K,N} for Q,K,V,O,gate,up,down */
    int KN[7][2]={ {3584,3584},{3584,512},{3584,512},{3584,3584},{3584,18944},{3584,18944},{18944,3584} };
    const char*nm[7]={"Q","K","V","O","gate","up","down"};
    int M=1;

    ork_w *w[7]; int8_t *A[7]; int32_t *Cs[7], *Ca[7]; float *X[7]; int8_t *prep[7];
    for(int m=0;m<7;m++){
        int K=KN[m][0],N=KN[m][1];
        int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
        ork_npu_set_pack_domain(c,0);
        w[m]=ork_mm_pack_i8(c,K,N,B); free(B);
        if(!w[m]){printf("pack %s (K%d N%d) failed\n",nm[m],K,N);return 1;}
        A[m]=malloc((size_t)M*K); memset(A[m],1,(size_t)M*K);
        Cs[m]=malloc((size_t)M*N*4); Ca[m]=malloc((size_t)M*N*4);
        X[m]=malloc((size_t)K*sizeof(float)); for(int i=0;i<K;i++)X[m][i]=0.01f*((i%17)-8);
        prep[m]=malloc((size_t)K);
    }

    /* warm + per-matmul NPU time (sync) */
    for(int m=0;m<7;m++) ork_mm_run_i8(c,w[m],M,A[m],Cs[m]);
    double t_npu_tok=0;
    for(int m=0;m<7;m++){
        double t0=now_us(); for(int it=0;it<iters;it++) ork_mm_run_i8(c,w[m],M,A[m],Cs[m]);
        double tm=(now_us()-t0)/iters; t_npu_tok+=tm;
        printf("  NPU %-4s K%-5d N%-5d : %7.1f us\n", nm[m],KN[m][0],KN[m][1],tm);
    }
    printf("  -> NPU per-token (7 matmuls): %.1f us\n\n", t_npu_tok);

    /* bit-exact gate: async result == sync result (Q) */
    ork_async*h=ork_mm_run_i8_async(c,w[0],M,A[0],Ca[0]); cpu_prep(X[0],prep[0],KN[0][0],4); ork_async_wait(h);
    int mism=0; for(int j=0;j<KN[0][1];j++) if(Cs[0][j]!=Ca[0][j]) mism++;
    printf("async-vs-sync bit-exact: mism=%d %s\n\n", mism, mism?"FAIL":"OK");

    printf("%-8s %8s %9s %9s %8s %9s %8s\n","cpu/npu","t_cpu","sync(us)","async(us)","speedup","ceil(us)","of-ceil");
    int reps_set[]={2,8,18,40,70,110,170}; /* sweep CPU-prep size -> CPU/NPU ratio (reach decode ~1.9) */
    for(unsigned ri=0; ri<sizeof(reps_set)/sizeof(int); ri++){
        int reps=reps_set[ri];
        /* measure t_cpu per token at this reps */
        double tc0=now_us(); for(int m=0;m<7;m++) cpu_prep(X[m],prep[m],KN[m][0],reps); double t_cpu_tok=now_us()-tc0;

        /* SYNC: cpu_prep then blocking run, summed */
        double s0=now_us();
        for(int it=0;it<iters;it++) for(int m=0;m<7;m++){ cpu_prep(X[m],prep[m],KN[m][0],reps); ork_mm_run_i8(c,w[m],M,A[m],Cs[m]); }
        double t_sync=(now_us()-s0)/iters;

        /* ASYNC: pipeline cpu_prep(m+1) behind NPU compute(m) */
        double a0=now_us();
        for(int it=0;it<iters;it++){
            cpu_prep(X[0],prep[0],KN[0][0],reps);
            for(int m=0;m<7;m++){
                ork_async*hh=ork_mm_run_i8_async(c,w[m],M,A[m],Ca[m]);
                if(m<6) cpu_prep(X[m+1],prep[m+1],KN[m+1][0],reps);
                ork_async_wait(hh);
            }
        }
        double t_async=(now_us()-a0)/iters;

        double ceil_us = (t_cpu_tok>t_npu_tok?t_cpu_tok:t_npu_tok); /* Σ max ~ max(Σ) here since per-m dominated similarly */
        printf("%-8.2f %8.1f %9.1f %9.1f %7.2fx %9.1f %7.0f%%\n",
               t_cpu_tok/t_npu_tok, t_cpu_tok, t_sync, t_async, t_sync/t_async, ceil_us, 100.0*ceil_us/t_async);
    }
    printf("\n(decode operating point ~ cpu/npu 1.9 from measured NPU 33%%/cpu7 62%%)\n");
    ork_npu_free(c); return 0;
}
