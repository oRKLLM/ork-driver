#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
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

    ork_w *w = ork_i4_mm_pack(ctx, K, N, B);
    if (!w) { printf("pack failed\n"); return 1; }

    int32_t *C1 = calloc(M1 * N, 4);
    int32_t *C2 = calloc(M2 * N, 4);

    ork_mm_task_i4 tasks[2];
    tasks[0].w = w; tasks[0].M = M1; tasks[0].A = A1; tasks[0].C = C1;
    tasks[1].w = w; tasks[1].M = M2; tasks[1].A = A2; tasks[1].C = C2;

    /* Call the chain REPEATEDLY on the same context: run_chain_i4 bcreate/bdestroys a fresh int16 output
     * scratch each call, and a recycled DMA region can carry dirty CPU cache lines that evict over the NPU's
     * writes -> "correct run 0, garbage runs 1+". A single call never exercised that; loop to guard it. */
    int bad = 0;
    const int REPS = 4;
    for (int rep = 0; rep < REPS; rep++) {
        /* poison the output each rep so a not-actually-written result stays wrong */
        memset(C1, 0x5a, (size_t)M1 * N * 4);
        memset(C2, 0x5a, (size_t)M2 * N * 4);

        int rc = ork_i4_mm_run_chain(ctx, 2, tasks);
        if (rc) { printf("run failed %d (rep %d)\n", rc, rep); return 1; }

        // verify task 1
        for (int m=0; m<M1; m++) {
            for (int n=0; n<N; n++) {
                long s=0;
                for (int k=0; k<K; k++) s+=(long)A1[m*K+k]*B[k*N+n];
                if (C1[m*N+n] != s) {
                    if (bad < 5) printf("rep %d C1[%d] = %d, expected %ld\n", rep, m*N+n, C1[m*N+n], s);
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
                    if (bad < 10) printf("rep %d C2[%d] = %d, expected %ld\n", rep, m*N+n, C2[m*N+n], s);
                    bad++;
                }
            }
        }
    }

    if (bad) {
        printf("FAIL: %d mismatches over %d reps\n", bad, REPS);
    } else {
        printf("OK: chain_i4 output verified (%d reps).\n", REPS);
    }

    /* ---- stream_i4: S independent W4A4 matmuls (varying M) dispatched round-robin across cores ---- */
    int SK = 128, SN = 128, NT = 4, Ms[4] = {1, 4, 8, 2};
    int8_t *SB[4]; int8_t *SA[4]; int32_t *SC[4]; ork_w *sw[4]; ork_mm_task_i4 st[4];
    for (int t = 0; t < NT; t++) {
        SB[t] = malloc((size_t)SK * SN); SA[t] = malloc((size_t)Ms[t] * SK); SC[t] = calloc((size_t)Ms[t] * SN, 4);
        for (int i = 0; i < SK * SN; i++) { sd = sd*1103515245+12345; SB[t][i] = (int8_t)((int)((sd>>17)%15)-7); }
        for (int i = 0; i < Ms[t] * SK; i++) { sd = sd*1103515245+12345; SA[t][i] = (int8_t)((int)((sd>>17)%15)-7); }
        sw[t] = ork_i4_mm_pack(ctx, SK, SN, SB[t]);
        if (!sw[t]) { printf("stream pack failed\n"); return 1; }
        st[t] = (ork_mm_task_i4){ sw[t], Ms[t], SA[t], SC[t] };
    }
    int src = ork_i4_mm_run_stream(ctx, NT, st);
    int sbad = 0;
    if (src) { printf("stream run failed %d\n", src); sbad = 1; }
    else for (int t = 0; t < NT; t++)
        for (int m = 0; m < Ms[t]; m++)
            for (int n = 0; n < SN; n++) {
                long s = 0; for (int k = 0; k < SK; k++) s += (long)SA[t][m*SK+k]*SB[t][k*SN+n];
                if (SC[t][m*SN+n] != s) { if (sbad < 5) printf("stream C[%d][%d,%d]=%d exp %ld\n", t, m, n, SC[t][m*SN+n], s); sbad++; }
            }
    printf("%s: stream_i4 S=%d (varying M=1/4/8/2) %s\n", sbad?"FAIL":"OK", NT, sbad?"":"output verified");
    for (int t = 0; t < NT; t++) { free(SB[t]); free(SA[t]); free(SC[t]); ork_w_free(sw[t]); }

    free(A1); free(A2); free(B); free(C1); free(C2);
    ork_w_free(w);
    ork_npu_free(ctx);
    return (bad || sbad) ? 1 : 0;
}
