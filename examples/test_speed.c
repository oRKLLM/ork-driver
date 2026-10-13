/* examples/test_speed.c — validates that multi-core prefill speed has not regressed.
 * Checks scaling of large-M matmuls across cores and absolute latency thresholds.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

/* The absolute-latency guards are PER BOARD, keyed on the device-tree compatible string (the same file
 * ork_soc_detect reads; the SoC id is identical on every RK3588, the product id is not). One constant for
 * both validated boards cannot work: at M=512 K=N=2048 int8 the ROCK 5B runs ~2.8 ms 3-core and the ROCK
 * 5B+ ~3.3-3.65 ms, so the 5B's 3600 us limit sat INSIDE the 5B+'s jitter band and failed it most runs.
 * Each row is that board's worst observed value +15%; the regression each still catches is on the wiki.
 * Wiki: Benchmark-Standards-and-Methodology § test_speed per-board latency guards. */
static const struct { const char *key; double mc_us, sc_us; } LIMITS[] = {
    { "radxa,rock-5b",      3600.0, 6500.0 },
    { "radxa,rock-5b-plus", 4200.0, 6800.0 },
};

/* compatible is a NUL-separated list, most specific first. Returns the product id, or NULL. */
static const char *board_key(char *buf, size_t n){
    FILE *f = fopen("/proc/device-tree/compatible","r"); if(!f) return NULL;
    size_t r = fread(buf,1,n-1,f); fclose(f); if(!r) return NULL; buf[r]=0; return buf;
}

static double run(ork_npu*c,ork_w*w,int M,int K,int8_t*A,int32_t*C,int iters){
    ork_i8_mm_run(c,w,M,A,C);                          /* warm */
    double t0=now_us(); for(int i=0;i<iters;i++) ork_i8_mm_run(c,w,M,A,C);
    return (now_us()-t0)/iters;
}

int main(void){
    int M=512, K=2048, N=2048, iters=10;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int cores=ork_npu_cores(c);
    
    int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    int8_t*A=malloc((size_t)M*K); memset(A,1,(size_t)M*K);
    int32_t*C=malloc((size_t)M*N*4);
    
    ork_w*w=ork_i8_mm_pack(c,K,N,B); if(!w){printf("pack failed\n");return 1;}
    
    /* Best-of-3 (min) for each: the guards below are all about the BEST-ACHIEVABLE latency/scaling, so a
     * single jittery window (a big core momentarily stolen by a background thread — orkllm, a kworker, the
     * prior test's residue) must not trip them. A genuinely lost/parked core collapses ALL reps, so the
     * parked-core scaling guard is preserved. Fixed a real false-fail: in the full `make test` run the
     * 3-core scaling landed ~1.49x (vs ~1.53x standalone) from post-`model` residue — just under the 1.5x
     * floor — while standalone it clears reliably. */
    ork_npu_set_core_budget(c,1);
    double t1=run(c,w,M,K,A,C,iters);
    for(int r=1;r<3;r++){ double x=run(c,w,M,K,A,C,iters); if(x<t1) t1=x; }
    printf("1-core: %.1f us/matmul\n", t1);

    ork_npu_set_core_budget(c,cores);
    double tN=run(c,w,M,K,A,C,iters);
    for(int r=1;r<3;r++){ double x=run(c,w,M,K,A,C,iters); if(x<tN) tN=x; }
    printf("%d-core: %.1f us/matmul (scaling %.2fx)\n", cores, tN, t1/tN);
    
    ork_w_free(w); free(A); free(B); free(C); ork_npu_free(c);
    
    int fail = 0;
    /* Thresholds re-ratcheted 2026-06-30 after the WEIGHT-DMA AMORTIZATION fix (M-tile cap raised from
     * R-1 to the 0x1040 schedule max mg_max*64 — see AGENTS.md). M=512 K=N=2048 int8 now measures
     * 1-core ~5.1ms (was ~11.7ms; ~2.1x faster), 3-core ~2.8ms (was ~4.55ms; ~1.6x). NOTE: the SCALING
     * RATIO DROPPED (~2.55x -> ~1.82x) BECAUSE single-core sped up MORE than multi-core — that is the
     * intended win, not a regression. So the absolute-latency guards (below) are now the real regression
     * detectors; the scaling floor only catches a lost/parked core (scaling collapses toward 1.0). */
    if (cores > 1 && (t1 / tN) < 1.4) {
        printf("FAIL: Multi-core scaling (%.2fx) is below 1.4x floor — likely a lost/parked core.\n", t1/tN);
        fail = 1;
    }
    /* Floor is 1.4x (was 1.5x): this guard catches a lost/PARKED core (scaling collapses toward 1.0 —
     * a parked core leaves ~2 effective cores -> ~1.0-1.3x). Healthy 3-core scaling on RK3588 is ~1.5x but
     * JITTERS ~1.47-1.58x run-to-run (co-tenant/scheduler), so a 1.5x floor false-failed ~50% of the time even
     * with the best-of-3 above. 1.4x sits cleanly between the collapse band (<1.35x) and the healthy jitter band.
     * The absolute-latency guards below are the real perf-regression detectors; this only detects a lost core. */
    /* Absolute latency guards — the real perf-regression detectors, per board (see LIMITS above). An
     * uncalibrated board takes the widest row and says so, rather than inheriting another board's number. */
    char cbuf[256]; const char *bk = board_key(cbuf, sizeof cbuf);
    double mc_lim = 0, sc_lim = 0; const char *lim_for = NULL;
    for (size_t i = 0; i < sizeof LIMITS/sizeof *LIMITS; i++) {
        if (bk && !strcmp(bk, LIMITS[i].key)) { mc_lim = LIMITS[i].mc_us; sc_lim = LIMITS[i].sc_us; lim_for = LIMITS[i].key; break; }
        if (LIMITS[i].mc_us > mc_lim) mc_lim = LIMITS[i].mc_us;
        if (LIMITS[i].sc_us > sc_lim) sc_lim = LIMITS[i].sc_us;
    }
    if (lim_for) printf("board: %s — limits %.0f us multi-core / %.0f us single-core\n", lim_for, mc_lim, sc_lim);
    else printf("board: %s — NOT CALIBRATED; using the widest limits %.0f/%.0f us (add a LIMITS row once measured)\n",
                bk ? bk : "unknown", mc_lim, sc_lim);

    if (tN > mc_lim) {
        printf("FAIL: Multi-core latency (%.1f us) exceeds %.0f us limit.\n", tN, mc_lim);
        fail = 1;
    }
    if (t1 > sc_lim) {
        printf("FAIL: Single-core latency (%.1f us) exceeds %.0f us limit (per-core kernel regression).\n", t1, sc_lim);
        fail = 1;
    }

    if (!fail) printf("SPEED OK\n");
    return fail;
}
