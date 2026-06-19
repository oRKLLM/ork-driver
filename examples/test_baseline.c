/* examples/test_baseline.c — Diagnostic to dump raw NPU output vs CPU reference
 * under mc=1 and mc=4 baseline settings.
 *
 *   make test_baseline && sudo ./test_baseline
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"

static unsigned sd = 99;
static int8_t r4(void) {
    sd = sd * 1103515245 + 12345;
    return (int8_t)((int)((sd >> 10) % 15) - 7); /* [-7,7] */
}

int main(void) {
    ork_npu *ctx = ork_npu_init();
    if (!ctx) {
        printf("init failed\n");
        return 1;
    }
    
    int M = 4, K = 64, N = 64;
    int8_t *A = malloc((size_t)M * K);
    int8_t *B = malloc((size_t)K * N);
    int32_t *ref = malloc((size_t)M * N * 4);
    
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = r4();
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = r4();
    
    // Compute exact CPU reference
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int s = 0;
            for (int k = 0; k < K; k++) {
                s += A[(size_t)m * K + k] * B[(size_t)k * N + n];
            }
            ref[(size_t)m * N + n] = s;
        }
    }
    
    // Test both mc=1 and mc=4
    for (int mc = 1; mc <= 4; mc += 3) {
        printf("\n--- Testing baseline with mc=%d ---\n", mc);
        
        int16_t *raw = calloc((size_t)M * N, 2);
        for (size_t i = 0; i < (size_t)M * N; i++) raw[i] = 0x7aaa;
        
        // We call ork_npu_probe_i4_mm but we need to see if it sets mc to the second arg
        // Wait, ork_npu_probe_i4_mm calls synth_i4 with mc=M.
        // So we can call ork_npu_probe_i4_mm(ctx, mc, K, N, A, B, raw)
        int rc = ork_npu_probe_i4_mm(ctx, mc, K, N, A, B, raw);
        printf("Submit rc=%d\n", rc);
        
        if (rc == 0) {
            for (int m = 0; m < M; m++) {
                printf("  Row%d NPU: ", m);
                for (int i = 0; i < 8; i++) printf("%6d ", raw[m * N + i]);
                printf(" | CPU Ref: ");
                for (int i = 0; i < 8; i++) printf("%6d ", ref[m * N + i]);
                printf("\n");
            }
        }
        free(raw);
    }
    
    free(A);
    free(B);
    free(ref);
    ork_npu_free(ctx);
    return 0;
}
