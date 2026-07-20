/* orkd_ring_probe — A-ring validation + latency bench. Packs an int8 weight in orkd, then runs the SAME M=K=N
 * matmul through BOTH the socket RPC (orkd_run_i8) and the shared-memory ring (orkd_run_i8_ring), asserts they
 * are bit-exact, and times per-op latency of each. The ring removes the per-op socket round-trip, so its win is
 * the transport delta (most visible at tiny M — the decode case). Board tool, not in `make test`.
 *
 *   make orkd orkd_ring_probe && sudo env ORK_USE_ORKD=1 ORKD_BIN=$PWD/orkd ./orkd_ring_probe [M] [K] [N] [iters]
 */
#include "orkd_client.h"
#include "orkd_proto.h"
#include "orkd_ring.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static long now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000000000L + t.tv_nsec; }

int main(int argc, char **argv){
    int M = argc > 1 ? atoi(argv[1]) : 1;
    int K = argc > 2 ? atoi(argv[2]) : 2048;
    int N = argc > 3 ? atoi(argv[3]) : 512;
    int iters = argc > 4 ? atoi(argv[4]) : 2000;
    setenv("ORK_USE_ORKD", "1", 1);

    orkd_conn *c = orkd_connect();
    if (!c){ fprintf(stderr, "orkd_connect failed\n"); return 2; }

    unsigned g = 0x1234567u;
    #define R8() ((int8_t)(((g = g*1103515245u + 12345u) >> 18 & 0x1f) - 16))
    int8_t *B = malloc((size_t)K * N), *A = malloc((size_t)M * K);
    int32_t *Cs = malloc((size_t)M * N * 4), *Cr = malloc((size_t)M * N * 4);
    if (!B || !A || !Cs || !Cr){ fprintf(stderr, "alloc\n"); return 2; }
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = R8();
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = R8();

    uint64_t wid = orkd_pack_i8(c, K, N, B);
    if (!wid){ fprintf(stderr, "pack failed\n"); return 2; }

    if (orkd_run_i8(c, wid, M, K, N, A, Cs)){ fprintf(stderr, "socket run failed\n"); return 2; }
    if (orkd_ring_setup(c)){ fprintf(stderr, "ring setup failed\n"); return 2; }
    int rr = orkd_run_i8_ring(c, wid, M, K, N, A, Cr);
    if (rr){ fprintf(stderr, "ring run failed rc=%d (M*N*4=%zu, slot=64K?)\n", rr, (size_t)M*N*4); return 2; }

    int bad = 0; for (int i = 0; i < M * N; i++) if (Cs[i] != Cr[i]){ if (!bad) fprintf(stderr, "mism [%d] socket=%d ring=%d\n", i, Cs[i], Cr[i]); bad++; }
    if (bad){ fprintf(stderr, "RING MISMATCH: %d/%d\n", bad, M * N); return 1; }

    long t0 = now_ns(); for (int i = 0; i < iters; i++) if (orkd_run_i8(c, wid, M, K, N, A, Cs)) return 3; long ts = now_ns() - t0;
    long t1 = now_ns(); for (int i = 0; i < iters; i++) if (orkd_run_i8_ring(c, wid, M, K, N, A, Cr)) return 3; long tr = now_ns() - t1;

    /* PIPELINED: keep a window of W ops in flight (submit-ahead, then collect) so each op's transport overlaps
     * the NPU compute of the ones ahead. Batches of W = full ring depth. */
    const int W = ORKD_RING_SLOTS;
    int tk[ORKD_RING_SLOTS];
    long t2 = now_ns();
    for (int done = 0; done < iters; ){
        int b = (iters - done < W) ? (iters - done) : W;
        for (int j = 0; j < b; j++){ tk[j] = orkd_ring_submit(c, wid, M, K, N, ORKD_DT_I8, A); if (tk[j] < 0) return 3; }
        for (int j = 0; j < b; j++){ if (orkd_ring_collect(c, tk[j], Cr)) return 3; }
        done += b;
    }
    long tp = now_ns() - t2;
    int badp = 0; for (int i = 0; i < M * N; i++) if (Cs[i] != Cr[i]) badp++;   /* pipelined result still matches */
    if (badp){ fprintf(stderr, "PIPELINED MISMATCH: %d/%d\n", badp, M * N); return 1; }

    double us_s = ts / 1e3 / iters, us_r = tr / 1e3 / iters, us_p = tp / 1e3 / iters;
    printf("orkd_ring_probe M=%d K=%d N=%d iters=%d  BIT-EXACT ✓ (sync + pipelined)\n", M, K, N, iters);
    printf("  socket RPC        : %7.2f us/op\n", us_s);
    printf("  shm ring (sync)   : %7.2f us/op   (%.2fx vs socket, %.2f us saved)\n", us_r, us_s / us_r, us_s - us_r);
    printf("  shm ring (pipe W=%d): %6.2f us/op   (%.2fx vs socket, %.2fx vs sync-ring)\n", W, us_p, us_s / us_p, us_r / us_p);
    orkd_disconnect(c);
    return 0;
    #undef R8
}
