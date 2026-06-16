/* tools/chain_probe.c — verify NPU register command PC-chaining and benchmark performance.
 *
 * This test benchmarks running S separate submits vs a single PC-chained submit of length S,
 * using the pre-allocated benchmark function.
 *   gcc -O2 -Wall -Iinclude -Isrc -pthread -o chain_probe tools/chain_probe.c src/npu.c src/soc.c src/soc/rk3588.c src/soc/rk3576.c -lm
 *   sudo ./chain_probe [S] [iters]
 */
#include <stdio.h>
#include <stdlib.h>
#include "ork_npu.h"

int main(int argc, char** argv) {
    ork_npu* c = ork_npu_init();
    if (!c) { printf("init failed (NPU?)\n"); return 1; }
    
    int S = argc > 1 ? atoi(argv[1]) : 8;
    int K = 256;
    int N = 64;
    int iters = argc > 2 ? atoi(argv[2]) : 200;
    
    printf("SoC %s — PC-chaining benchmark (S=%d, int8 %dx%d, iters=%d):\n", ork_npu_soc(c), S, K, N, iters);
    
    int rc = ork_npu_benchmark_chain(c, S, K, N, iters);
    if (rc != 0) {
        printf("Benchmark failed: rc=%d\n", rc);
    }
    
    ork_npu_free(c);
    return rc;
}
