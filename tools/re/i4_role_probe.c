/* tools/re/i4_role_probe.c — does the SMALL operand belong in the int4 weight (tiled) role?
 *
 * THE CLAIM UNDER TEST. The int4 BCHAIN loop is N-tile-outer / M-group-inner with HW weight reuse,
 * so per matmul the hardware moves
 *      weight bytes      = N*K/2                   (each N-tile streamed once)
 *      activation bytes  = NC * M*K/2              (re-streamed once per N-tile, NC = ceil(N/Wb))
 * and the staleness law Wb*K/2 <= WEIGHT_BANK*32768 caps a single N-tile at ~360 KiB. A WIDE weight
 * must therefore be tiled (NC>1) and the activation pays for it. Transposing (C^T = B^T * A^T) puts
 * the SMALL matrix in the weight role, where it fits untiled (NC==1) and both operands are read once.
 *
 * We do NOT need a transposed datapath to test that: running the SHAPE with the operand sizes
 * swapped exercises exactly the same hardware geometry. (K,N,M) = (K,M0,N0) vs (K,N0,M0).
 *
 * METHOD. Full-RANDOM int4-range data (a patterned generator can make consecutive tiles identical
 * and hide the exact corruption this sweep is looking for). Reference = the SHIPPED default config
 * for that shape. Every point is repeated and TIMED: the driver self-heals a timed-out submit and
 * returns a CORRECT checksum ~1-2 s later, so correct-but-slow is a FAILED config, not a valid one.
 * Run with ORK_I4_NOLADDER=1 so a refused geometry surfaces as rc!=0 instead of being silently
 * re-tiled by the de-escalation ladder.
 *
 *   make i4_role_probe
 *   sudo tools/util/npu_guard.sh -- env ORK_I4_NOLADDER=1 ORK_MM_TIMEOUT=2000 \
 *        ./i4_role_probe <K> <N> <M> "<cfg>;<cfg>;..."
 *   cfg = n:wb:nc   (0 = leave at the shipped default)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SLOW_X 8.0

static unsigned long long fnv64(const void *p, size_t n) {
    const unsigned char *b = p; unsigned long long h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; } return h;
}
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }

/* xorshift64* — full-period, no short-period tile aliasing */
void orki_i4_tile_Aslice(unsigned char *dst, const signed char *Arow, int k0, int Kp);  /* internal.h; host-side nibble pack of ONE activation row */
static unsigned long long rs = 88172645463325252ULL;
static unsigned long long rnd64(void) { rs ^= rs>>12; rs ^= rs<<25; rs ^= rs>>27; return rs*2685821657736338717ULL; }

int main(int argc, char **argv) {
    int K = argc>1?atoi(argv[1]):2048, N = argc>2?atoi(argv[2]):2048, M = argc>3?atoi(argv[3]):128;
    const char *cfgs = argc>4?argv[4]:"0:0:0";
    int reps = getenv("ORK_REPS")?atoi(getenv("ORK_REPS")):3;
    if (reps < 1) reps = 1; if (reps > 16) reps = 16;
    if (K%32 || N%64) { printf("need K%%32, N%%64\n"); return 2; }

    ork_npu *c = ork_npu_init(); if (!c) { printf("no board\n"); return 0; }
    int8_t *B = malloc((size_t)K*N), *A = malloc((size_t)M*K);
    int32_t *Cb = malloc((size_t)M*N*4);
    if (!A||!B||!Cb) { printf("OOM\n"); return 2; }
    /* FULL-RANDOM in the int4 code range [-7,7]; no pattern, no period. */
    for (size_t i = 0; i < (size_t)K*N; i++) B[i] = (int8_t)((int)(rnd64()%15) - 7);
    for (size_t i = 0; i < (size_t)M*K; i++) A[i] = (int8_t)((int)(rnd64()%15) - 7);
    ork_w *w = ork_i4_mm_pack(c, K, N, B); if (!w) { printf("pack fail\n"); return 2; }

    unsetenv("ORK_I4_DBNK"); unsetenv("ORK_I4_WB"); unsetenv("ORK_NPU_MC"); unsetenv("ORK_I4_H");
    char buf[32]; double base = 0; unsigned long long ref = 0;
    printf("K=%d N=%d M=%d  reps=%d  wreuse=%s  (Wb_default=%d, Hcap~%d)\n",
           K, N, M, reps, getenv("ORK_I4_WREUSE")?getenv("ORK_I4_WREUSE"):"1(default)",
           ((131072/K)&~63) > N ? N : ((131072/K)&~63), 16384/K);
    printf("%-14s %-5s %-5s %-5s %-7s %-10s %-10s %s\n",
           "cfg(n:wb:nc)", "H", "NG", "NC", "progs", "best us", "ratio", "verdict");

    char *list = strdup(cfgs), *save = NULL;
    int first = 1;
    for (char *tok = strtok_r(list, ";", &save); tok; tok = strtok_r(NULL, ";", &save)) {
        int n = 0, wb = 0, nc = 0; sscanf(tok, "%d:%d:%d", &n, &wb, &nc);
        if (n > 0) { snprintf(buf,sizeof buf,"%d",n); setenv("ORK_I4_DBNK",buf,1); } else unsetenv("ORK_I4_DBNK");
        if (wb> 0) { snprintf(buf,sizeof buf,"%d",wb); setenv("ORK_I4_WB",buf,1); } else unsetenv("ORK_I4_WB");
        if (nc> 0) { snprintf(buf,sizeof buf,"%d",nc); setenv("ORK_NPU_MC",buf,1); } else unsetenv("ORK_NPU_MC");

        int hb = 16384/K; if (hb < 1) hb = 1;
        int H = n>0 ? (n*hb > 64 ? 64 : n*hb) : hb;
        int eWb = wb>0 ? wb : ((131072/K)&~63); if (eWb < 64) eWb = 64; if (eWb > N) eWb = N;
        int NC = (N+eWb-1)/eWb, NG = (M+H-1)/H;

        double best = 0, last = 0; int rc = 0, bad = 0; unsigned long long ck = 0;
        /* one untimed warm pass, then reps timed single runs */
        memset(Cb, 0, (size_t)M*N*4);
        rc = ork_i4_mm_run(c, w, M, A, Cb);
        if (!rc) for (int r = 0; r < reps; r++) {
            memset(Cb, 0, (size_t)M*N*4);
            double t0 = now_us();
            rc = ork_i4_mm_run(c, w, M, A, Cb);
            double us = now_us() - t0; last = us;
            if (rc) break;
            ck = fnv64(Cb, (size_t)M*N*4);
            if (first) { if (!ref) ref = ck; }
            if (ck != ref) { bad = 1; break; }
            if (!first && us > SLOW_X*base) { bad = 2; break; }
            if (!best || us < best) best = us;
        }
        if (first) { base = best; first = 0; }
        const char *v = rc ? "REFUSED(rc!=0)" : bad==1 ? "WRONG" : bad==2 ? "SELFHEAL" : "ok";
        printf("%-14s %-5d %-5d %-5d %-7d %-10.1f %-10.2f %-14s last=%.1f us\n",
               tok, H, NG, NC, NG*NC, best, best?base/best:0.0, v, last);
        fflush(stdout);
    }
    free(list);
    /* HOST-SIDE activation nibble-pack cost: bch_db_worker runs exactly this over all M rows,
     * once per call, on the critical path. It scales with M (the ACTIVATION role), not with N. */
    { unsigned char *ab = malloc((size_t)M*(size_t)(K/2)); double bp = 0;
      if (ab) { for (int r = 0; r < 3; r++) { double t0 = now_us();
                    for (int j = 0; j < M; j++) orki_i4_tile_Aslice(ab + (size_t)j*(K/2), A + (size_t)j*K, 0, K);
                    double u = now_us()-t0; if (!bp || u < bp) bp = u; }
                printf("host A-pack (M=%d rows x K/2=%d B): %.1f us\n", M, K/2, bp); free(ab); } }
    unsetenv("ORK_I4_DBNK"); unsetenv("ORK_I4_WB"); unsetenv("ORK_NPU_MC");
    ork_mm_free(c, w); ork_npu_free(c);
    return 0;
}
