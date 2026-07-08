/* mm_cap.c — vendor RKNN matmul-API regcmd capture harness (per-type quant/layout aware).
 *
 * Enumerates any rknn_matmul_type at any (M,K,N) and runs one matmul so the LD_PRELOAD capture
 * shim (regcmd_capture.c / rknpu_dump.so) can dump the emitted regcmd. Sets A/B/C scale+zp arrays
 * sized to the quant mode so int4/mixed types do not fail with "Unsupport type bits 0".
 *
 * Used (2026-07) to map the domain-1001 0x40xx matmul engine: 0x4010=precision/mode reg,
 * 0x4034/0x405c=row count (int8/fp16 in-task batch = our M-scheduler), and to establish that
 * RK3588 supports only SYMMETRIC matmul (fp16xfp16, int8xint8, int4xint4) — every mixed type
 * (w4a16, w4a8, fp16xint8) is rejected at create with "unsupported matmul dtype in this platform".
 * See the ork-driver wiki: regcmd-ISA-Reference "Domain-1001 matmul engine".
 *
 * RE/calibration only — depends on the proprietary librknnrt (never built into the library; AGENTS.md).
 *   cc -O2 -I$RKNN_SDK -o mm_cap mm_cap.c -L$RKNN_SDK -lrknnrt
 *   sudo env LD_PRELOAD=$PWD/rknpu_dump.so LD_LIBRARY_PATH=$RKNN_SDK RKDUMP_WORDS=0 \
 *       ./mm_cap <type> <M> <K> <N> [B_layout] [B_quant] 2> cap.dump
 *   # type: rknn_matmul_type enum (2=int8->i32, 1=fp16->f32, 10=int4->i16 needs B_layout=1, ...)
 *   # B_quant: 0=per-layer, 2=per-channel, 4=per-group
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_matmul_api.h"

int main(int c, char **v) {
    int type = c > 1 ? atoi(v[1]) : 2, M = c > 2 ? atoi(v[2]) : 32,
        K = c > 3 ? atoi(v[3]) : 512, N = c > 4 ? atoi(v[4]) : 64;
    int Bl = c > 5 ? atoi(v[5]) : 0, Bq = c > 6 ? atoi(v[6]) : 0;

    rknn_matmul_ctx ctx; rknn_matmul_info info; rknn_matmul_io_attr io;
    memset(&info, 0, sizeof info); memset(&io, 0, sizeof io);
    info.M = M; info.K = K; info.N = N; info.type = (rknn_matmul_type)type;
    info.B_layout = Bl; info.B_quant_type = Bq; info.AC_layout = 0;

    int r = rknn_matmul_create(&ctx, &info, &io);
    if (r < 0) { printf("RESULT type=%d M=%d K=%d N=%d Bl=%d Bq=%d create=FAIL(%d)\n", type, M, K, N, Bl, Bq, r); return 2; }

    rknn_tensor_mem *Am = rknn_create_mem(ctx, io.A.size),
                    *Bm = rknn_create_mem(ctx, io.B.size),
                    *Cm = rknn_create_mem(ctx, io.C.size);
    memset(Am->virt_addr, 1, io.A.size); memset(Bm->virt_addr, 1, io.B.size);
    rknn_matmul_set_io_mem(ctx, Am, &io.A);
    rknn_matmul_set_io_mem(ctx, Bm, &io.B);
    rknn_matmul_set_io_mem(ctx, Cm, &io.C);

    /* B scale/zp length matches the quant mode; A and C are always per-layer (scalar). */
    int nB = Bq == 2 ? N : (Bq == 4 ? (N * (K / 32)) : 1);
    float *sB = malloc(sizeof(float) * nB); for (int i = 0; i < nB; i++) sB[i] = 0.003f + 0.0001f * i;
    int *zB = calloc(nB, sizeof(int));
    float sA = 0.01f, sC = 0.02f; int zA = 0, zC = 0;
    rknn_quant_params qa; memset(&qa, 0, sizeof qa); strcpy(qa.name, "A"); qa.scale = &sA; qa.scale_len = 1; qa.zp = &zA; qa.zp_len = 1;
    rknn_quant_params qb; memset(&qb, 0, sizeof qb); strcpy(qb.name, "B"); qb.scale = sB; qb.scale_len = nB; qb.zp = zB; qb.zp_len = nB;
    rknn_quant_params qc; memset(&qc, 0, sizeof qc); strcpy(qc.name, "C"); qc.scale = &sC; qc.scale_len = 1; qc.zp = &zC; qc.zp_len = 1;
    rknn_matmul_set_quant_params(ctx, &qa);
    rknn_matmul_set_quant_params(ctx, &qb);
    rknn_matmul_set_quant_params(ctx, &qc);

    int rr = rknn_matmul_run(ctx);
    printf("RESULT type=%d M=%d K=%d N=%d Bl=%d Bq=%d run=%s (A=%u B=%u C=%u)\n",
           type, M, K, N, Bl, Bq, rr ? "FAIL" : "OK",
           (unsigned)io.A.size, (unsigned)io.B.size, (unsigned)io.C.size);
    rknn_matmul_destroy(ctx);
    free(sB); free(zB);
    return rr ? 3 : 0;
}
