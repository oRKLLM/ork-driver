/* tools/re/percapture.c — RE harness: capture the regcmd RKNN emits for PER-LAYER vs PER-CHANNEL int8
 * matmul requant, so we can find the SDP per-channel scale-array registers (to graft per-channel weight
 * scale into ork's fused-SiLU output stage). Run each mode under the LD_PRELOAD capture shim + diff.
 *
 * Uses the proprietary librknnrt (calibration only; never built into the library — see AGENTS.md).
 *   gcc -O2 -I$RKNN/rknn_sdk -o percapture percapture.c -L$RKNN/rknn_sdk -lrknnrt
 *   sudo env LD_PRELOAD=.../rknpu_dump.so LD_LIBRARY_PATH=$RKNN/rknn_sdk ./percapture <0=perlayer|1=perchannel>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_matmul_api.h"

int main(int argc, char** argv) {
    int per_channel = argc > 1 ? atoi(argv[1]) : 0;
    int mmtype = argc > 2 ? atoi(argv[2]) : 3;       /* 3=INT8_TO_INT8, 2=TO_INT32, 9=TO_FLOAT32 */
    const int M = 32, K = 512, N = 64;               /* small: keep the regcmd simple to diff */

    rknn_matmul_ctx ctx; rknn_matmul_info info; rknn_matmul_io_attr io;
    memset(&info, 0, sizeof info); memset(&io, 0, sizeof io);
    info.M = M; info.K = K; info.N = N;
    info.type = (rknn_matmul_type) mmtype;           /* output precision: reveals the 0x4010/0x40c0/0x4050 format regs */
    info.AC_layout = 0; info.B_layout = 0;
    info.B_quant_type = per_channel ? 1 : 0;         /* 0=per-layer, 1=per-channel — the diff we want */

    int ret = rknn_matmul_create(&ctx, &info, &io);
    if (ret < 0) { printf("create FAILED ret=%d\n", ret); return 1; }
    printf("== created INT8_MM_INT8_TO_INT8 M=%d K=%d N=%d B_quant_type=%d (A=%u B=%u C=%u bytes) ==\n",
           M, K, N, info.B_quant_type, (unsigned)io.A.size, (unsigned)io.B.size, (unsigned)io.C.size);

    rknn_tensor_mem* Am = rknn_create_mem(ctx, io.A.size);
    rknn_tensor_mem* Bm = rknn_create_mem(ctx, io.B.size);
    rknn_tensor_mem* Cm = rknn_create_mem(ctx, io.C.size);
    memset(Am->virt_addr, 1, io.A.size); memset(Bm->virt_addr, 1, io.B.size);
    rknn_matmul_set_io_mem(ctx, Am, &io.A);
    rknn_matmul_set_io_mem(ctx, Bm, &io.B);
    rknn_matmul_set_io_mem(ctx, Cm, &io.C);

    /* quant params: A + C per-layer (scalar); B per-layer(1) or per-channel(N) scale array */
    float sA = 0.01f, sC = 0.02f;
    float* sB = malloc(sizeof(float) * N);
    for (int n = 0; n < N; n++) sB[n] = 0.003f + 0.0001f * n;   /* DISTINCT per-channel scales -> visible in the array */
    rknn_quant_params qa; memset(&qa, 0, sizeof qa); strcpy(qa.name, "A"); qa.scale = &sA; qa.scale_len = 1;
    rknn_quant_params qc; memset(&qc, 0, sizeof qc); strcpy(qc.name, "C"); qc.scale = &sC; qc.scale_len = 1;
    rknn_quant_params qb; memset(&qb, 0, sizeof qb); strcpy(qb.name, "B");
    qb.scale = sB; qb.scale_len = per_channel ? N : 1;
    rknn_matmul_set_quant_params(ctx, &qa);
    rknn_matmul_set_quant_params(ctx, &qb);
    rknn_matmul_set_quant_params(ctx, &qc);

    printf("== run (capture shim dumps the regcmd here) ==\n");
    if (rknn_matmul_run(ctx)) { printf("run FAILED\n"); }

    rknn_destroy_mem(ctx, Am); rknn_destroy_mem(ctx, Bm); rknn_destroy_mem(ctx, Cm);
    rknn_matmul_destroy(ctx); free(sB);
    return 0;
}
