/* i4_widek_stall_probe.c — standalone reproducer for the int4 NONBLOCK-doorbell WIDE-K round stall.
 *
 * WHY THIS EXISTS. On the 27B W4A4 prefill, 1-4 doorbell rounds per run never complete: the NPU writes a
 * contiguous PREFIX of the output and then wedges that round permanently (proved by a 30 s wait; the data
 * is genuinely absent, not merely unseen). Each costs ~3.6 s via ACT_RESET + resubmit — ~23% of prefill.
 * Instrumenting the real model showed the misses are NOT a uniform "~1/4000 dispatch drop": 12 of 12 were
 * one of TWO shapes, and the dominant one is `K=17408 N=5120 M=1 Sk=2` — the 27B ffn_down. That is the
 * WIDEST-K op, and `ORK_I4_KS = 10752` splits it UNEVENLY as 10752 + 6656, with the first slice sitting
 * exactly at the validated single-submit K ceiling. Same "marginal geometry exposed as a slice remainder"
 * class as the BCHAIN H-table bug (OPS_REGISTRY, K=2560).
 *
 * `tools/i4_doorbell_probe` CANNOT test this shape: its blocking reference is `run_chain_i4`, bounded at
 * K<=4096, so both arms just return "rc errs". This probe drives the REAL path (`ork_i4_mm_pack` +
 * `ork_i4_mm_run`, M=1 -> ork_i4_dyn_begin_mc) so the Sk=2 slicing is the same one the model takes.
 *
 * DETECTION IS BY WALL TIME, not by output: the recover loop makes a stalled round produce the CORRECT
 * answer, so correctness cannot see it. A healthy iteration is a few ms; a stalled one pays the miss
 * window plus the reset (~600 ms+). Anything over --slow-ms is a stall.
 *
 *   ./i4_widek_stall_probe [reps=60] [K=17408] [N=5120] [slow_ms=100] [ndom=1] [M=64]
 *
 * M IS THE CHAIN DEPTH, and it matters more than it looks. int4 has no multi-row kernel on this path, so
 * ork_i4_mm_run decomposes M rows into M SEPARATE M=1 row-tasks and hands the whole set to
 * ork_i4_dyn_begin_mc as ONE chain -- which is why the model logs the stalled op as `M=1` inside an `S=64`
 * chain when prefilling a 64-token ubatch. A probe at M=1 therefore submits a 1-op chain and does NOT
 * reproduce (80/80 clean, ndom=2 also 80/80 clean). Default M=64 to match the model.
 *
 * ndom>1 packs the SAME shape into ndom IOMMU domains and round-robins them, forcing an orki_dom_activate
 * between every run. The shape ALONE does not stall (80/80 clean at ndom=1, median 5.46 ms), so the defect
 * needs context; domain switching is the first co-factor to test, since the model runs this op across 10+
 * domains and it is the largest weight in the model.
 *
 * Sweep K to find the boundary, e.g. an EVEN split (K=21504 = 2x10752) vs the uneven 17408.
 * Board op: sudo ORK_NPU_LOCK_WAIT=600 tools/util/npu_guard.sh -- env ORK_MC_DIAG=1 ./i4_widek_stall_probe
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3 + t.tv_nsec/1e6; }
static int cmpd(const void *a, const void *b){ double x=*(const double*)a, y=*(const double*)b; return x<y?-1:x>y; }

int main(int argc, char **argv){
    setvbuf(stdout, NULL, _IOLBF, 0);
    int reps = argc>1?atoi(argv[1]):60, K = argc>2?atoi(argv[2]):17408, N = argc>3?atoi(argv[3]):5120;
    double slow = argc>4?atof(argv[4]):100.0;
    int ndom = argc>5?atoi(argv[5]):1; if(ndom<1) ndom=1; if(ndom>8) ndom=8;
    int M = argc>6?atoi(argv[6]):64; if(M<1) M=1;
    if (K%32 || N%64){ printf("K%%32 and N%%64 required\n"); return 2; }

    ork_npu *c = ork_npu_init();
    if(!c){ printf("no board\n"); return 0; }
    int KS = 10752, Sk = (K+KS-1)/KS;                  /* mirrors ORK_I4_KS (internal); for reporting only */
    printf("i4 wide-K stall probe: M=%d (=> %d-op chain) K=%d N=%d reps=%d slow>%.0fms ndom=%d | K-slices ~%d (%d + %d)\n",
           M, M, K, N, reps, slow, ndom, Sk, K>KS?KS:K, K>KS?K-KS:0);

    int8_t *B = malloc((size_t)K*N);
    if(!B){ printf("OOM B (%zu MB)\n", (size_t)K*N/(1u<<20)); return 1; }
    for(size_t i=0;i<(size_t)K*N;i++) B[i] = (int8_t)((i*2654435761u >> 13) % 15) - 7;   /* [-7,7], varied */
    int8_t  *A = malloc((size_t)M*K);
    for(size_t i=0;i<(size_t)M*K;i++) A[i] = (int8_t)((i%15)-7);
    int32_t *C = calloc((size_t)M*N,4), *C0 = calloc((size_t)M*N,4);
    if(!A||!C||!C0){ printf("OOM\n"); return 1; }

    ork_w *w[8];
    for(int d=0; d<ndom; d++){
        ork_npu_set_pack_domain(c, d);
        w[d] = ork_i4_mm_pack(c, K, N, B);
        if(!w[d]){ printf("pack failed (K=%d N=%d dom=%d)\n", K, N, d); return 1; }
    }
    if(ork_i4_mm_run(c, w[0], M, A, C0)){ printf("first run rc!=0\n"); return 1; }   /* warm + reference */
    for(int d=1; d<ndom; d++) ork_i4_mm_run(c, w[d], M, A, C0);                      /* warm every domain */

    double *t = malloc(sizeof(double)*reps); int nslow=0, nbad=0; double tot=0, mx=0;
    for(int r=0;r<reps;r++){
        memset(C, 0, (size_t)M*N*4);
        double t0 = now_ms();
        int rc = ork_i4_mm_run(c, w[r % ndom], M, A, C);
        t[r] = now_ms() - t0; tot += t[r]; if(t[r]>mx) mx=t[r];
        if(rc){ nbad++; continue; }
        if(memcmp(C, C0, (size_t)M*N*4)) nbad++;       /* recovery should keep every run identical */
        if(t[r] > slow){ nslow++; printf("  [stall] rep %d: %.1f ms\n", r, t[r]); }
    }
    qsort(t, reps, sizeof(double), cmpd);
    printf("---\n  median %.2f ms | mean %.2f ms | max %.1f ms\n", t[reps/2], tot/reps, mx);
    printf("  STALLS (>%.0fms): %d / %d  (%.2f%%)   mismatched/rc-err runs: %d\n",
           slow, nslow, reps, 100.0*nslow/reps, nbad);
    printf(nslow ? "STALL REPRODUCED at K=%d N=%d\n" : "no stall at K=%d N=%d\n", K, N);
    ork_npu_free(c);
    return nslow ? 1 : 0;
}
