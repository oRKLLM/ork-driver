#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ork_npu.h"

int main(void) {
    ork_npu *ctx = ork_npu_init();
    if (!ctx) return 1;

    int M1 = 1, M2 = 1;
    int K = 64, N = 64;

    int8_t *A1 = malloc(M1 * K);
    int8_t *A2 = malloc(M2 * K);
    int8_t *B = malloc(K * N);
    
    unsigned sd = 12345;
    for (int i=0; i<M1*K; i++) { sd=sd*1103515245+12345; A1[i] = (int8_t)((int)((sd>>17)%15)-7); }
    for (int i=0; i<M2*K; i++) { sd=sd*1103515245+12345; A2[i] = (int8_t)((int)((sd>>17)%15)-7); }
    for (int i=0; i<K*N; i++) { sd=sd*1103515245+12345; B[i] = (int8_t)((int)((sd>>17)%15)-7); }

    ork_w *w = ork_mm_pack_i4(ctx, K, N, B);
    if (!w) { printf("pack failed\n"); return 1; }

    int32_t *C1 = calloc(M1 * N, 4);
    int32_t *C2 = calloc(M2 * N, 4);

    ork_mm_task_i4 tasks[2];
    tasks[0].w = w; tasks[0].M = M1; tasks[0].A = A1; tasks[0].C = C1;
    tasks[1].w = w; tasks[1].M = M2; tasks[1].A = A2; tasks[1].C = C2;

    int rc = ork_mm_run_chain_i4(ctx, 2, tasks);
    if (rc) { printf("run failed %d\n", rc); return 1; }

    int bad = 0;
    // verify task 1
    for (int m=0; m<M1; m++) {
        for (int n=0; n<N; n++) {
            long s=0; 
            for (int k=0; k<K; k++) s+=(long)A1[m*K+k]*B[k*N+n];
            if (C1[m*N+n] != s) {
                if (bad < 5) printf("C1[%d] = %d, expected %ld\n", m*N+n, C1[m*N+n], s);
                bad++;
            }
        }
    }
    // verify task 2
    for (int m=0; m<M2; m++) {
        for (int n=0; n<N; n++) {
            long s=0; 
            for (int k=0; k<K; k++) s+=(long)A2[m*K+k]*B[k*N+n];
            if (C2[m*N+n] != s) {
                if (bad < 10) printf("C2[%d] = %d, expected %ld\n", m*N+n, C2[m*N+n], s);
                bad++;
            }
        }
    }

    if (bad) {
        printf("FAIL: %d mismatches\n", bad);
    } else {
        printf("OK: chain_i4 output verified.\n");
    }

    free(A1); free(A2); free(B); free(C1); free(C2);
    ork_w_free(w);
    ork_npu_free(ctx);
    return bad ? 1 : 0;
}
