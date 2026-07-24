/* test_spine.c — regression test for the ork_spine heterogeneous CPU<->NPU dispatcher (include/ork_spine.h),
 * the productionized form of spine_sched_probe. Drives a mini "layer-shape" DAG across one NPU unit (doorbell)
 * and one CPU unit and asserts: (a) an NPU op and an independent CPU op OVERLAP (wall < serial); (b) a CPU op
 * depending on an NPU op reads the NPU output COHERENTLY (cross-unit civac handoff); (c) the whole DAG completes.
 *
 * DAG:  op0 NPU matmul -> C0            (no deps)   ─┐ overlap
 *       op1 CPU glue on scratch         (no deps)   ─┘
 *       op2 CPU bridge: civac+read C0   (dep op0)    after op0  (coherency)
 *       op3 NPU matmul -> C3            (dep op1)    after op1
 *   make test_spine && sudo env ORK_MM_TIMEOUT=3000 ./test_spine
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include "ork_spine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint32_t rng = 0x51edu;
static int s3(void){ rng = rng * 1664525u + 1013904223u; return (int)((rng >> 28) % 3) - 1; }
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }

static ork_npu *g_c;
/* The NPU-unit op: run ntk int8 matmuls (BLOCKING) on the unit's own thread, then flush the output for a
 * cross-thread consumer. Uses ork_mm_run_i8 (M=1 -> robust multi-core) rather than the ork_dyn doorbell so this
 * GATING test is deterministic — the nonblock doorbell miss-fires intermittently on this board (tasks #13/#21),
 * which spine_sched_probe exercises separately. The dispatcher is agnostic to what the op does; the overlap is
 * thread-level (NPU-unit thread || CPU-unit thread), NOT dependent on the doorbell's nonblock nature. */
struct npu_arg { ork_mm_task_i8 *tk; int ntk; const void *out; size_t outbytes; };
static long npu_fn(void *p){ struct npu_arg *a = p;
    for (int s = 0; s < a->ntk; s++) if (ork_mm_run_i8(g_c, a->tk[s].w, a->tk[s].M, a->tk[s].A, a->tk[s].C)) return -1;
    ork_spine_civac_range(a->out, a->outbytes);   /* producer flush: push the NPU output to DRAM for a cross-thread consumer */
    return 0; }
struct glue_arg { const int8_t *x; size_t n; long out; };
static long glue_fn(void *p){ struct glue_arg *a = p; long acc = 0;
    for (size_t i = 0; i < a->n; i++){ long q = (long)a->x[i] >> 1; if (q > 63) q = 63; if (q < -63) q = -63; acc += q; }
    a->out = acc; return acc; }
struct bridge_arg { const int32_t *C; int n; long sum; };
static long bridge_fn(void *p){ struct bridge_arg *a = p; ork_spine_civac_range(a->C, (size_t)a->n * 4);   /* consumer: invalidate before reading NPU output */
    long s = 0; for (int i = 0; i < a->n; i++) s += a->C[i]; a->sum = s; return s; }

int main(void){
    int M = 1, K = 512, N = 512, S = 8;
    setvbuf(stdout, 0, _IONBF, 0);
    ork_npu *c = ork_npu_init(); if (!c){ printf("init failed\n"); return 2; } g_c = c;
    printf("test_spine: DAG over {NPU-unit, CPU-unit} via ork_spine (S=%d K=%d N=%d)\n", S, K, N);

    int8_t *A = malloc((size_t)M * K), *Wb = malloc((size_t)K * N);
    for (int i = 0; i < M * K; i++) A[i] = (int8_t)s3();
    for (size_t i = 0; i < (size_t)K * N; i++) Wb[i] = (int8_t)s3();
    ork_w *W = ork_mm_pack_i8(c, K, N, Wb); if (!W){ printf("pack fail\n"); return 2; }
    int32_t *C0 = ork_dma_alloc(c, (size_t)S * N * 4), *C3 = ork_dma_alloc(c, (size_t)S * N * 4);
    if (!C0 || !C3){ printf("dma fail\n"); return 2; }
    ork_mm_task_i8 *tk0 = malloc((size_t)S * sizeof *tk0), *tk3 = malloc((size_t)S * sizeof *tk3);
    for (int s = 0; s < S; s++){ tk0[s] = (ork_mm_task_i8){W, M, A, C0 + (size_t)s * N}; tk3[s] = (ork_mm_task_i8){W, M, A, C3 + (size_t)s * N}; }
    int32_t *Cr = malloc((size_t)N * 4);
    for (int n = 0; n < N; n++){ long a = 0; for (int k = 0; k < K; k++) a += A[k] * Wb[(size_t)k * N + n]; Cr[n] = (int32_t)a; }
    long refsum = 0; for (int n = 0; n < N; n++) refsum += Cr[n];   /* op2 bridge should see S*refsum (all S rows == ref) */

    size_t SN = (size_t)K * N; int8_t *scr = malloc(SN); for (size_t i = 0; i < SN; i++) scr[i] = (int8_t)(i * 7);
    struct npu_arg na0 = {tk0, S, C0, (size_t)S * N * 4}, na3 = {tk3, S, C3, (size_t)S * N * 4};
    struct glue_arg ga = {scr, SN, 0}; struct bridge_arg ba = {C0, S * N, 0};
    ork_spine_op ops[4] = {
        {0,     ORK_PL_NPU, npu_fn,    &na0, 0, -1, 0, 0},   /* op0 NPU matmul -> C0 */
        {0,     ORK_PL_CPU, glue_fn,   &ga,  0, -1, 0, 0},   /* op1 CPU glue (independent -> overlaps op0) */
        {1<<0,  ORK_PL_CPU, bridge_fn, &ba,  0, -1, 0, 0},   /* op2 CPU bridge: read C0 after op0 */
        {1<<1,  ORK_PL_NPU, npu_fn,    &na3, 0, -1, 0, 0},   /* op3 NPU matmul -> C3 after op1 */
    };
    int NOPS = 4;

    ork_spine_unit U[2];
    ork_spine_unit_start(&U[0], ORK_UNIT_NPU, 4);   /* NPU unit -> big core 4 */
    ork_spine_unit_start(&U[1], ORK_UNIT_CPU, 6);   /* CPU unit -> big core 6 */
    /* warm both units' paths (cold doorbell returns garbage; warm the NPU op once) */
    { int g = ork_spine_unit_dispatch(&U[0], npu_fn, &na0); ork_spine_unit_wait(&U[0], g); }
    { int g = ork_spine_unit_dispatch(&U[1], glue_fn, &ga); ork_spine_unit_wait(&U[1], g); }

    double t0 = now_us();
    int rc = ork_spine_run(U, 2, ops, NOPS);
    double wall = now_us() - t0;

    ork_spine_civac_range(C0, (size_t)S * N * 4);   /* main thread: invalidate before reading NPU output */
    long bad = 0; int32_t mx = 0;
    for (int s = 0; s < S; s++) for (int n = 0; n < N; n++){ int32_t d = C0[(size_t)s * N + n] - Cr[n]; if (d){ bad++; if (d < 0) d = -d; if (d > mx) mx = d; } }
    int bridge_ok = (ba.sum == (long)S * refsum);

    /* serial baseline: each op alone THROUGH ITS UNIT (no overlap), same execution path */
    double serial = 0;
    for (int i = 0; i < NOPS; i++){ ork_spine_unit *u = (ops[i].placement == ORK_PL_NPU) ? &U[0] : &U[1];
        double a = now_us(); int g = ork_spine_unit_dispatch(u, ops[i].fn, ops[i].arg); ork_spine_unit_wait(u, g); serial += now_us() - a; }

    printf("  ork_spine_run rc=%d\n", rc);
    printf("  op0 NPU coherent: %s (bad=%ld max|d|=%d)\n", bad ? "MISMATCH" : "bit-exact", bad, mx);
    printf("  op2 bridge read C0: %s (sum=%ld want=%ld)\n", bridge_ok ? "OK" : "WRONG", ba.sum, (long)S * refsum);
    printf("  scheduler wall %.0fus vs serial %.0fus -> %.2fx (op0 || op1 overlap; op2>op0; op3>op1)\n", wall, serial, serial / wall);
    int pass = (rc == 0 && bad == 0 && bridge_ok && wall < serial);
    printf("%s\n", pass ? "PASS — ork_spine drives the DAG across NPU+CPU units, coherent + overlapped"
                        : "FAIL");
    ork_spine_unit_stop(&U[0]); ork_spine_unit_stop(&U[1]);
    free(A); free(Wb); free(tk0); free(tk3); free(Cr); free(scr);
    ork_dma_free(c, C0); ork_dma_free(c, C3); ork_mm_free(c, W); ork_npu_free(c);
    return pass ? 0 : 1;
}
