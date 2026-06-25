/* examples/test_activations.c — validate + profile the NEON activation/normalization kernels.
 *
 * (1) Correctness: diff the NEON kernels against the scalar libm-expf references within tolerance.
 * (2) Profiling: time the NEON kernels vs the scalar baseline over a 4096-dim row sweep, logging
 *     the microsecond/call latency reduction.
 *
 *   make test_activations && ./test_activations          (runs on the board AND the workstation)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "neon_activations.h"

#define N 4096
#define ITERS 20000

static unsigned sd = 1234567u;
static float frand(void){ sd = sd*1103515245u + 12345u; return ((int)((sd>>9)&0xffff) - 32768) / 8192.0f; } /* ~[-4,4] */

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }

int main(void) {
    static float g[N], u[N], gr[N], x[N], w[N], o_neon[N], o_ref[N];
    for (int i = 0; i < N; i++) { g[i] = frand(); u[i] = frand(); x[i] = frand(); w[i] = 0.5f + frand()*0.1f; }

    int fail = 0;

    /* ---- (1a) SiLU-gate (SwiGLU) correctness ---- */
    float gn[N]; for (int i=0;i<N;i++){ gn[i]=g[i]; gr[i]=g[i]; }
    ork_silu_mul_f32(gn, u, N);
    ork_silu_mul_f32_ref(gr, u, N);
    double max_abs=0, max_rel=0;
    for (int i=0;i<N;i++){ double a=fabs(gn[i]-gr[i]); double r=a/(fabs(gr[i])+1e-6); if(a>max_abs)max_abs=a; if(r>max_rel)max_rel=r; }
    int ok_silu = (max_abs < 1e-3) && (max_rel < 1e-3);
    printf("[silu_mul] max_abs=%.2e max_rel=%.2e  %s\n", max_abs, max_rel, ok_silu?"OK":"FAIL"); fail |= !ok_silu;

    /* ---- (1b) RMSNorm correctness ---- */
    ork_rmsnorm_f32(o_neon, x, w, N, 1e-5f);
    ork_rmsnorm_f32_ref(o_ref, x, w, N, 1e-5f);
    max_abs=0; max_rel=0;
    for (int i=0;i<N;i++){ double a=fabs(o_neon[i]-o_ref[i]); double r=a/(fabs(o_ref[i])+1e-6); if(a>max_abs)max_abs=a; if(r>max_rel)max_rel=r; }
    int ok_rms = (max_abs < 1e-4) && (max_rel < 1e-3);
    printf("[rmsnorm ] max_abs=%.2e max_rel=%.2e  %s\n", max_abs, max_rel, ok_rms?"OK":"FAIL"); fail |= !ok_rms;

    /* ---- (1c) in-place RMSNorm safety (o aliases x) ---- */
    float xi[N]; for(int i=0;i<N;i++) xi[i]=x[i];
    ork_rmsnorm_f32(xi, xi, w, N, 1e-5f);
    max_abs=0; for(int i=0;i<N;i++){ double a=fabs(xi[i]-o_ref[i]); if(a>max_abs)max_abs=a; }
    int ok_ip = max_abs < 1e-4;
    printf("[rmsnorm ] in-place max_abs vs ref=%.2e  %s\n", max_abs, ok_ip?"OK":"FAIL"); fail |= !ok_ip;

    /* ---- (2) profiling: NEON vs scalar baseline, 4096-dim row ---- */
    volatile float sink = 0;
    /* SiLU-gate */
    for (int it=0; it<200; it++){ for(int i=0;i<N;i++)gn[i]=g[i]; ork_silu_mul_f32(gn,u,N); }        /* warm */
    double t0=now_us(); for(int it=0;it<ITERS;it++){ for(int i=0;i<N;i++)gn[i]=g[i]; ork_silu_mul_f32(gn,u,N); } double t_neon=(now_us()-t0)/ITERS; sink+=gn[0];
    t0=now_us(); for(int it=0;it<ITERS;it++){ for(int i=0;i<N;i++)gr[i]=g[i]; ork_silu_mul_f32_ref(gr,u,N); } double t_ref=(now_us()-t0)/ITERS; sink+=gr[0];
    printf("[silu_mul] N=%d  scalar(expf)=%.3f us  NEON=%.3f us  speedup=%.2fx\n", N, t_ref, t_neon, t_ref/t_neon);

    /* RMSNorm */
    for (int it=0; it<200; it++) ork_rmsnorm_f32(o_neon,x,w,N,1e-5f);
    t0=now_us(); for(int it=0;it<ITERS;it++) ork_rmsnorm_f32(o_neon,x,w,N,1e-5f); t_neon=(now_us()-t0)/ITERS; sink+=o_neon[0];
    t0=now_us(); for(int it=0;it<ITERS;it++) ork_rmsnorm_f32_ref(o_ref,x,w,N,1e-5f); t_ref=(now_us()-t0)/ITERS; sink+=o_ref[0];
    printf("[rmsnorm ] N=%d  scalar=%.3f us  NEON=%.3f us  speedup=%.2fx\n", N, t_ref, t_neon, t_ref/t_neon);
    (void)sink;

    printf("%s\n", fail ? "test_activations FAIL" : "test_activations OK");
    return fail;
}
