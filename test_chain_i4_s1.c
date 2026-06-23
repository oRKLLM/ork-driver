#include <stddef.h>
#include <stdint.h>
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    ork_npu *ctx = ork_npu_init();
    int M = 1, K = 64, N = 64;
    int8_t *A = calloc(M * K, 1);
    int8_t *B = calloc(K * N, 1);
    A[0] = 1; B[0] = 1;
    ork_w *w = ork_mm_pack_i4(ctx, K, N, B);
    
    ork_mm_task_i4 tasks[1];
    tasks[0].M = M;
    tasks[0].A = A;
    tasks[0].w = w;
    tasks[0].C = calloc(M * N, 4);
    
    int rc = ork_mm_run_chain_i4(ctx, 1, tasks);
    printf("run_chain_i4 rc = %d\n", rc);
    printf("C[0] = %d\n", tasks[0].C[0]);
    return 0;
}
