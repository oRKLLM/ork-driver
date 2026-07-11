/* tools/i16_shape_probe.c — reproduce + localize the on-NPU int16 activation op (ork_npu_silu_i16 ->
 * act_lut_i16 -> ork_npu_probe_silu_std_i16) IN-CHAIN wedge, in isolation (no full model load).
 *
 * The op is bit-accurate STANDALONE at every shape, but SOFT-RESETS the NPU when run inside the FFN chain
 * (after a matmul, with resident weights). This tool (1) sweeps shapes standalone to confirm they're clean,
 * then (2) reproduces the chain CONTEXT — a preceding matmul (single- vs multi-core) + a RESIDENT weight —
 * THEN the silu, to separate the hypotheses: is it the multi-core matmul PIPELINE TRANSITION, or the
 * RESIDENT weights / CBUF-MAC state that needs cleaning between the CNA/DPU matmul and the pure-SDP op?
 *   make i16_shape_probe && sudo ./i16_shape_probe        (board only; each wedge soft-resets + recovers)
 * rc: 0=ok, -1=wedge/timeout, -2=bad shape/pack, -3=non-rk3588.
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

static int try_shape(ork_npu *c, int M, int N){
    int16_t *in = calloc((size_t)M*N, 2), *out = calloc((size_t)M*N, 2);
    if(!in||!out){ free(in); free(out); return -99; }
    for(size_t i=0;i<(size_t)M*N;i++) in[i]=(int16_t)((i%201)-100);   /* small dummy gate values */
    double us=0; int rc = ork_npu_silu_i16(c, in, M, N, 0.01, 0.01, out, &us);
    printf("    silu M=%d N=%d -> rc=%d %s (%.0f us)\n",
           M, N, rc, rc==0?"OK":(rc==-1?"WEDGE":"shape/soc"), us);
    fflush(stdout);
    free(in); free(out);
    return rc;
}

/* Reproduce the chain context: pack a gate-like int8 weight, run a matmul on `nc` cores, then the int16
 * silu at the chain shape. keep_w!=NULL returns the packed weight resident (caller frees) to test whether
 * a RESIDENT weight (not just the matmul) is what wedges the following silu. */
static int matmul_then_silu(ork_npu *c, int nc, ork_w **keep_w){
    const int Km=2048, Nm=6144, Mm=128;   /* FFN gate-like */
    int8_t *B = calloc((size_t)Km*Nm, 1); if(!B) return -2;
    for(size_t i=0;i<(size_t)Km*Nm;i++) B[i]=(int8_t)((i%7)-3);
    ork_w *w = ork_mm_pack_i8(c, Km, Nm, B); free(B);
    if(!w){ printf("    pack fail\n"); return -2; }
    int8_t *A = calloc((size_t)Mm*Km, 1); int32_t *C = malloc((size_t)Mm*Nm*4);
    for(size_t i=0;i<(size_t)Mm*Km;i++) A[i]=(int8_t)((i%5)-2);
    ork_npu_set_core_budget(c, nc);
    int mrc = ork_mm_run_i8(c, w, Mm, A, C);
    printf("    [matmul K=%d N=%d M=%d nc=%d -> rc=%d]\n", Km, Nm, Mm, nc, mrc);
    free(A); free(C);
    int rc = try_shape(c, 128, 6144);
    if(keep_w) *keep_w = w; else ork_mm_free(c, w);
    return rc;
}

/* [D] like matmul_then_silu but the weight is IMPORTED (dma-buf), matching the orkpack chain path (the
 * recent int16 chain wedges all showed imported=1). pack -> dump -> reload-as-import -> matmul -> silu. */
static int matmul_then_silu_imported(ork_npu *c, int nc){
    const int Km=2048, Nm=6144, Mm=128;
    int8_t *B = calloc((size_t)Km*Nm, 1); if(!B) return -2;
    for(size_t i=0;i<(size_t)Km*Nm;i++) B[i]=(int8_t)((i%7)-3);
    ork_w *wp = ork_mm_pack_i8(c, Km, Nm, B); free(B);
    if(!wp){ printf("    pack fail\n"); return -2; }
    size_t nb = ork_w_dump(wp, NULL, 0); void *blob = malloc(nb); ork_w_dump(wp, blob, nb);
    ork_mm_free(c, wp);
    ork_w *w = ork_mm_load_i8_import(c, Km, Nm, blob, nb); free(blob);
    if(!w){ printf("    import fail (no dma-heap?)\n"); return -2; }
    int8_t *A = calloc((size_t)Mm*Km, 1); int32_t *C = malloc((size_t)Mm*Nm*4);
    for(size_t i=0;i<(size_t)Mm*Km;i++) A[i]=(int8_t)((i%5)-2);
    ork_npu_set_core_budget(c, nc);
    int mrc = ork_mm_run_i8(c, w, Mm, A, C);
    printf("    [IMPORTED matmul K=%d N=%d M=%d nc=%d -> rc=%d]\n", Km, Nm, Mm, nc, mrc);
    free(A); free(C);
    int rc = try_shape(c, 128, 6144);
    ork_mm_free(c, w);
    return rc;
}

/* [E]/[F] threading repro: ggml runs the FFN handler (matmul + silu) on a WORKER thread, while
 * ork_npu_init ran on the main thread. Mimic that: do the matmul+silu (or just the silu) on a spawned
 * pthread. If THIS wedges (but the main-thread version doesn't), the wedge is a thread/fd/context issue,
 * not the hardware path. */
struct targ { ork_npu *c; int mode; int rc; };   /* mode 0 = matmul+silu, 1 = silu-only */
static void *thread_fn(void *p){
    struct targ *a = p;
    if(a->mode==0) a->rc = matmul_then_silu(a->c, ork_npu_cores(a->c), NULL);
    else           a->rc = try_shape(a->c, 128, 6144);
    return NULL;
}
static int on_thread(ork_npu *c, int mode){
    struct targ a={c,mode,0}; pthread_t th;
    if(pthread_create(&th,NULL,thread_fn,&a)){ printf("    pthread_create fail\n"); return -99; }
    pthread_join(th,NULL); return a.rc;
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    ork_npu *c = ork_npu_init(); if(!c){ printf("no board\n"); return 0; }
    int cores = ork_npu_cores(c);
    printf("i16_shape_probe: reproduce the int16-silu in-chain wedge in isolation (cores=%d)\n", cores);

    printf("-- (1) STANDALONE sweep (expect all OK) --\n");
    int Ns[] = {64, 2048, 6144, 8192};
    for(int i=0;i<4;i++) try_shape(c, 8, Ns[i]);
    try_shape(c, 128, 6144);   /* the chain shape, standalone */

    printf("-- (2) CHAIN CONTEXT repro --\n");
    printf("  [A] SINGLE-core matmul then silu (isolates: does ANY matmul wedge it?):\n");
    matmul_then_silu(c, 1, NULL);
    printf("  [B] MULTI-core matmul (nc=%d) then silu (isolates: is it the multi->single transition?):\n", cores);
    matmul_then_silu(c, cores, NULL);
    printf("  [C] MULTI-core matmul + weight kept RESIDENT, then silu x2 (isolates: resident-weight state?):\n");
    { ork_w *w=NULL; matmul_then_silu(c, cores, &w);
      try_shape(c, 128, 6144);            /* 2nd silu with the weight still resident, no fresh matmul */
      if(w) ork_mm_free(c, w); }
    printf("  [D] IMPORTED (dma-buf) weight matmul then silu (matches the orkpack chain path):\n");
    matmul_then_silu_imported(c, cores);

    printf("-- (3) THREADING repro (init on main; work on a spawned worker, like ggml) --\n");
    printf("  [E] silu-ONLY on a spawned thread:\n");
    on_thread(c, 1);
    printf("  [F] matmul+silu on a spawned thread (closest to the ggml FFN handler):\n");
    on_thread(c, 0);

    ork_npu_free(c);
    return 0;
}
