/* test_orkd — the orkd daemon's first client: run int8 matmuls THROUGH orkd and self-validate vs a CPU
 * reference (bit-exact). Proves the daemon serves a real, test-shaped workload end-to-end.
 *
 * Standalone (NOT in the default `make test`): the other examples open /dev/dri/cardN DIRECTLY, so a
 * lingering orkd holding the NPU would contend with them (single-stream). The completion of the first-client
 * milestone is converting the WHOLE suite to route through orkd (every example a client, one daemon spun up
 * and idle-reaped around the run, no direct NPU access) — this example is the proof-of-concept for that.
 *
 *   make orkd test_orkd && sudo env ORKD_BIN=$PWD/orkd ./test_orkd
 * (sudo so the auto-spawned daemon can open the NPU). Exits 0 on ALL OK, nonzero on any mismatch.
 */
#include "orkd_client.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* deterministic int8 fill (fixed-seed LCG), range ~[-40,87] */
static uint32_t g_s = 12345u;
static int8_t r8(void){ g_s = g_s * 1103515245u + 12345u; return (int8_t)(((g_s >> 16) & 0x7f) - 40); }

static int one(orkd_conn *c, int M, int K, int N){
    int8_t *A = malloc((size_t)M*K), *B = malloc((size_t)K*N);
    int32_t *C = malloc((size_t)M*N*4), *ref = malloc((size_t)M*N*4);
    if (!A || !B || !C || !ref){ printf("  alloc FAIL\n"); free(A);free(B);free(C);free(ref); return 1; }
    for (int i = 0; i < M*K; i++) A[i] = r8();
    for (int i = 0; i < K*N; i++) B[i] = r8();
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++){
        long a = 0; for (int k = 0; k < K; k++) a += (long)A[m*K+k] * B[k*N+n]; ref[m*N+n] = (int32_t)a;
    }
    int32_t *Cz = malloc((size_t)M*N*4), *Cz2 = malloc((size_t)M*N*4);
    uint64_t w = orkd_pack_i8(c, K, N, B);
    int bad = 0;
    if (!w || !Cz || !Cz2){ printf("  pack/alloc FAIL M=%d K=%d N=%d\n", M, K, N); bad = 1; }
    else {
        int rc   = orkd_run_i8(c, w, M, K, N, A, C);        /* socket transfer */
        int rcz  = orkd_run_i8_zc(c, w, M, K, N, A, Cz);    /* A zero-copy; C over socket */
        int rcz2 = orkd_run_i8_zc2(c, w, M, K, N, A, Cz2);  /* A + C zero-copy (both by reference) */
        orkd_free_weight(c, w);
        if (rc){ printf("  run FAIL M=%d K=%d N=%d rc=%d\n", M, K, N, rc); bad = 1; }
        else for (int i = 0; i < M*N; i++) if (C[i] != ref[i]){ if (bad < 3) printf("  MISMATCH(socket) M=%d K=%d N=%d [%d] got=%d want=%d\n", M, K, N, i, C[i], ref[i]); bad++; }
        if (rcz == -2){ printf("  (zero-copy skipped: no dma-heap)\n"); }
        else if (rcz){ printf("  run-zc(A) FAIL M=%d K=%d N=%d rc=%d\n", M, K, N, rcz); bad = 1; }
        else for (int i = 0; i < M*N; i++) if (Cz[i] != ref[i]){ if (bad < 6) printf("  MISMATCH(zc-A) M=%d K=%d N=%d [%d] got=%d want=%d\n", M, K, N, i, Cz[i], ref[i]); bad++; }
        if (rcz2 == -2){ /* no dma-heap: already noted */ }
        else if (rcz2){ printf("  run-zc2(A+C) FAIL M=%d K=%d N=%d rc=%d\n", M, K, N, rcz2); bad = 1; }
        else for (int i = 0; i < M*N; i++) if (Cz2[i] != ref[i]){ if (bad < 9) printf("  MISMATCH(zc-A+C) M=%d K=%d N=%d [%d] got=%d want=%d\n", M, K, N, i, Cz2[i], ref[i]); bad++; }
    }
    if (!bad) printf("  ok M=%d K=%d N=%d (%d/%d) [socket + zc-A + zc-A+C]\n", M, K, N, M*N, M*N);
    free(A); free(B); free(C); free(ref); free(Cz); free(Cz2);
    return bad ? 1 : 0;
}

int main(void){
    orkd_conn *c = orkd_connect();
    if (!c){ fprintf(stderr, "test_orkd: no orkd (connect/spawn failed)\n"); return 1; }
    printf("test_orkd: connected client_id=%u npu_cores=%u\n", orkd_client_id(c), orkd_soc_cores(c));
    int bad = 0;
    bad |= one(c, 1,  64,  32);    /* M=1 decode */
    bad |= one(c, 8,  128, 64);    /* small prefill */
    bad |= one(c, 4,  256, 32);
    bad |= one(c, 16, 512, 64);    /* multi-tile */
    orkd_disconnect(c);
    printf("test_orkd: %s\n", bad ? "FAILED" : "ALL OK");
    return bad ? 1 : 0;
}
