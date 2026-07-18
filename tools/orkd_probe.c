/* orkd_probe — validate the orkd daemon end-to-end (board tool, NOT in `make test`).
 *
 * Auto-spawns orkd if none is running (the daemon opens the NPU), connects, prints the daemon-reported core
 * count. Modes:
 *   orkd_probe            — PING/PONG liveness, then disconnect.
 *   orkd_probe <seconds>  — hold the connection <seconds> (run two to watch them SHARE one daemon + idle-reap).
 *   orkd_probe mm         — pack a small int8 weight + run A x weight THROUGH the daemon, verify vs a CPU ref.
 *
 * Run under sudo so the auto-spawned daemon can open /dev/dri/cardN:
 *   make orkd orkd_probe && sudo env ORKD_BIN=$PWD/orkd ./orkd_probe mm
 */
#include "orkd_client.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int do_matmul(orkd_conn *c){
    enum { M = 4, K = 64, N = 32 };                 /* K%32==0, N%32==0 */
    static int8_t A[M*K], B[K*N]; static int32_t C[M*N];
    for (int i = 0; i < M*K; i++) A[i] = 1;         /* A = ones -> C[m,n] = sum_k B[k,n] */
    for (int k = 0; k < K; k++) for (int n = 0; n < N; n++) B[k*N+n] = (int8_t)((k*7 + n*3) % 5 - 2);
    uint64_t wid = orkd_pack_i8(c, K, N, B);
    if (!wid){ printf("pack FAILED\n"); return 1; }
    printf("packed weight_id=%llu\n", (unsigned long long)wid);
    if (orkd_run_i8(c, wid, M, K, N, A, C)){ printf("run FAILED\n"); orkd_free_weight(c, wid); return 2; }
    int bad = 0;
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++){
        long ref = 0; for (int k = 0; k < K; k++) ref += (long)A[m*K+k] * B[k*N+n];
        if (C[m*N+n] != ref){ if (bad < 4) printf("MISMATCH [%d,%d] got=%d want=%ld\n", m, n, C[m*N+n], ref); bad++; }
    }
    printf("matmul-through-daemon %s (%d/%d correct)\n", bad ? "WRONG" : "OK", M*N - bad, M*N);
    orkd_free_weight(c, wid);
    return bad ? 3 : 0;
}

int main(int argc, char **argv){
    const char *mode = argc > 1 ? argv[1] : "";
    int hold = (mode[0] >= '0' && mode[0] <= '9') ? atoi(mode) : 0;
    orkd_conn *c = orkd_connect();
    if (!c){ fprintf(stderr, "orkd_probe: connect/spawn FAILED\n"); return 1; }
    printf("[pid %d] connected: client_id=%u npu_cores=%u\n", (int)getpid(), orkd_client_id(c), orkd_soc_cores(c));
    int rc = 0;
    if (!strcmp(mode, "mm")) rc = do_matmul(c);
    else { if (orkd_ping(c)){ printf("ping FAILED\n"); rc = 2; } else printf("ping OK\n"); if (hold > 0) sleep(hold); }
    orkd_disconnect(c);
    printf("[pid %d] disconnected\n", (int)getpid());
    return rc;
}
