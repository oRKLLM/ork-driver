/* tools/ssd_fusion_bench.c — Phase-1 GATE G1: the decisive fusion micro-bench (SSM_ON_NPU_PLAN.md §3.1e).
 *
 * The thesis rests on FUSING the per-chunk SSD op-graph into ONE chained submit instead of running each op
 * as its own submit (each paying the NPU submit floor). Before building the large generalized chunk
 * assembler (§1c), prove the economics cheaply.
 *
 * The DECISIVE measurement is submit-floor amortization, isolated with matmul-only ops (no activation LUT,
 * so no per-call LUT-build confound): time N matmuls as ONE chained submit vs N separate submits, at a
 * floor-dominated shape (small K, tiny arithmetic — the regime SSD chunks live in). If chaining amortizes
 * the floor here, it amortizes it for the mixed chunk chain too (SDP tasks add one shared LUT-load submit,
 * not one floor per op — the FFN chain already proves mixed matmul+SDP runs in a single submit).
 *
 * Secondary, informative: the same 4 ops as a MIXED fused chain (matmul->silu->matmul->ewmul) and as 4
 * standalone submits — these include LUT build/calibration (a per-call cost of the standalone API, which
 * fusion amortizes/avoids), so they are reported but NOT the basis of the verdict.
 *
 * G1 verdict is based on the clean matmul-chain economics: PASS if the chained submit beats N separate
 * submits by a wide margin and runs near one floor. If not, HALT (SSM_ON_NPU_PLAN gate G1).
 *
 *   make ssd_fusion_bench && sudo ./ssd_fusion_bench [iters]     (board only)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static double siluf(double x){ return x/(1.0+exp(-x)); }
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec*1e-3; }

int main(int argc, char **argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int iters = (argc>1)? atoi(argv[1]) : 100;
    if(iters<1) iters=100;
    ork_npu *c=ork_npu_init();
    if(!c){ printf("no board / no NPU\n"); return 0; }

    /* Floor-dominated, chunk-representative shape: K=512 (min the current chain envelope accepts), N=64,
     * M=64 (~one chunk of rows; single M-tile). All-ones -> results trivially checkable. */
    const int M=64, K=512, N=64;
    const int S=4;   /* ops per chain (== chunk-chain depth we care about) */
    static signed char A[64*512], W[512*64];
    for(int i=0;i<M*K;i++) A[i]=1;
    for(int i=0;i<K*N;i++) W[i]=1;
    ork_w *w = ork_i8_mm_pack(c,K,N,W);
    if(!w){ printf("pack failed\n"); ork_npu_free(c); return 1; }

    /* ================= DECISIVE: matmul-only, submit-floor amortization (no LUT) ================= */
    static int CC[8][64*64];
    /* raw single-submit floor */
    ork_i8_mm_run(c,w,M,A,CC[0]);                              /* warm */
    double st0,su0,cp0,st1,su1,cp1; long n0,n1;
    ork_npu_run_timing(&st0,&su0,&cp0,&n0);
    double b0=now_us(); for(int it=0;it<iters;it++) ork_i8_mm_run(c,w,M,A,CC[0]); double t_floor=(now_us()-b0)/iters;
    ork_npu_run_timing(&st1,&su1,&cp1,&n1);
    long dn = n1-n0; double d_setup=(st1-st0), d_submit=(su1-su0), d_copy=(cp1-cp0);   /* us totals over dn calls */

    /* Sweep chain depth: if the fused (1-ioctl, N-task PC-chain) time grows ~linearly with N, chaining is
     * NOT amortizing the per-task floor (each PC-chained task re-pays it, kernel-driven completion IRQ). */
    int Ss[3] = {2,4,8}; double t_fused[3], t_unf[3]; int chain_ok=1;
    for(int si=0; si<3; si++){
        int s=Ss[si];
        ork_mm_task_i8 mt[8];
        for(int j=0;j<s;j++){ mt[j].w=w; mt[j].M=M; mt[j].A=A; mt[j].C=CC[j]; }
        /* UNFUSED: s separate submits */
        double u0=now_us(); for(int it=0;it<iters;it++) for(int j=0;j<s;j++) ork_i8_mm_run(c,w,M,A,CC[j]); t_unf[si]=(now_us()-u0)/iters;
        /* FUSED: s tasks, ONE chained ioctl */
        int rc=ork_i8_mm_run_chain(c,s,mt);                    /* warm */
        if(rc){ printf("chain S=%d rc=%d (%s)\n", s, rc, rc==-1?"WEDGED":"err"); ork_npu_free(c); return 1; }
        double f0=now_us(); for(int it=0;it<iters;it++) ork_i8_mm_run_chain(c,s,mt); t_fused[si]=(now_us()-f0)/iters;
        for(int i=0;i<M*N;i++){ if(CC[0][i]!=K||CC[s-1][i]!=K){ chain_ok=0; break; } }
    }
    double t_fused_mm=t_fused[1], t_unfused_mm=t_unf[1];       /* S=4 headline */
    int mm_ok=chain_ok;

    /* ================= SECONDARY (informative): mixed chain + standalone SDP (include LUT build) ===== */
    const double is=3.0/32.0, os=siluf(3.0)/60.0;
    static int Cg[64*64],Cs[64*64],Cu[64*64],Ch[64*64];
    ork_mm_task_i8 t[4]={ {w,M,A,Cg},{w,M,A,Cs},{w,M,A,Cu},{w,M,A,Ch} };
    ork_chain_op ops[4]={ {1,-1,0,0x4000,18}, {2,0,0,0,0}, {1,-1,0,0x4000,18}, {3,1,2,0x4000,19} };
    int rc_mixed = ork_i8_mm_run_chain_ffn(c,4,t,ops,is,os);   /* warm (builds LUT internally each call) */
    double m0=now_us(); for(int it=0;it<iters;it++) ork_i8_mm_run_chain_ffn(c,4,t,ops,is,os); double t_mixed=(now_us()-m0)/iters;
    signed char *g8=(signed char*)Cg; int mixed_ok=(rc_mixed==0 && g8[0]==32);

    printf("\n=== GATE G1 — SSD fusion micro-bench (M=%d K=%d N=%d, %d iters, single-core) ===\n", M,K,N,iters);
    printf("DECISIVE (matmul-only, no LUT — isolates the submit-floor amortization):\n");
    printf("  raw submit floor (1 matmul submit)  : %8.1f us\n", t_floor);
    if(dn>0) printf("    floor breakdown/call (run_timing): setup=%.1f submit(ioctl+NPU)=%.1f copy=%.1f us  (n=%ld)\n",
                    d_setup/dn, d_submit/dn, d_copy/dn, dn);
    else     printf("    (run_timing not populated for this path — likely single-core non-multicore dispatch)\n");
    printf("  chain-depth sweep (fused=1 ioctl of N PC-chained tasks vs N separate submits):\n");
    printf("    %-4s %12s %12s %10s %12s\n","N","unfused_us","fused_us","speedup","fused/floor");
    for(int si=0; si<3; si++)
        printf("    %-4d %12.1f %12.1f %9.2fx %11.2fx\n", Ss[si], t_unf[si], t_fused[si], t_unf[si]/t_fused[si], t_fused[si]/t_floor);
    printf("  (if fused_us grows ~linearly with N and fused/floor ~= N, PC-chaining does NOT amortize the floor)\n");
    printf("  matmul chain correct (all==K=%d)     : %s\n", K, mm_ok?"yes":"NO");
    printf("SECONDARY (include one-time LUT build; fusion amortizes this — not the verdict basis):\n");
    printf("  MIXED fused chain [mm->silu->mm->ewmul] 1 submit : %8.1f us  (rc=%d, ran=%s)\n", t_mixed, rc_mixed, mixed_ok?"yes":"no");

    /* Verdict on the clean matmul economics: chaining must beat separate submits by a wide margin and run
     * near one floor (require >=2x speedup for S=4, and fused < 2x the single-submit floor). */
    int pass = mm_ok && (t_unfused_mm/t_fused_mm >= 2.0) && (t_fused_mm < 2.0*t_floor);
    printf("\n  G1 VERDICT: %s\n", pass ?
        "PASS — chaining amortizes the submit floor (1 submit for N ops); proceed to build the SSD op-set (§1a-1c)."
      : "FAIL — chaining does NOT clear the floor; HALT and rethink the perf thesis (SSM_ON_NPU_PLAN gate G1).");
    ork_npu_free(c);
    return pass?0:1;
}
