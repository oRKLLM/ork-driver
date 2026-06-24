/* tools/hadamard_i4_npu.c — Tier 4a Hadamard rotation for W4A4, validated ON THE NPU (not just CPU math).
 *
 * hadamard_int4.c / hadamard_real.c prove the rotation cuts int4 quant RMS in a pure-CPU model. This
 * tool proves it END-TO-END on real silicon through the public grouped-W4A4 path (ork_mm_pack_i4_grouped
 * / ork_mm_run_i4_grouped): a per-group (block) Hadamard of size G is exactly rotation-invariant within
 * each group, so it composes with per-group int4 quant WITHOUT changing the matmul result — the NPU still
 * computes A.B, just with the outliers spread so the int4 quantization is more accurate.
 *
 * Recipe (caller-driven, using the exported ork_fwht_norm):
 *   weights  B_rot[g] = H . B[g]   (rotate each length-G slice of each column, then int4-quantize)
 *   activs   A_rot[g] = H . A[g]   (rotate each length-G slice of each row,    then int4-quantize)
 *   C = sum_g (A_rot[g] . B_rot[g]) = sum_g (A[g] . B[g]) = A.B     (H orthonormal, HH^T = I)
 *
 *   make hadamard_i4_npu && sudo ./hadamard_i4_npu [M] [K] [N] [G] [outlier_scale]
 *
 * NOTE: matmul RMS is a DIAGNOSTIC. int4 error is unbiased and averages across layers + residuals, so
 * the final verdict still needs perplexity from the wired-in pipeline (stories15M / a real GGUF). This
 * tool's job is to confirm (a) the rotated path is mechanically correct on hardware and (b) it lowers
 * the on-NPU quant RMS in the presence of activation outliers (the thing Hadamard is for). */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "ork_npu.h"

/* one W4A4 grouped run on the NPU; rotate=1 applies a per-group Hadamard to A and B before quantizing.
 * Returns RMS rel err vs the fp32 reference; *mecherr = max|NPU - per-group int4 dequant| (must be ~0). */
static double run_case(ork_npu *c, const float *Af, const float *Bf, const float *Cref,
                       int M, int K, int N, int G, int rotate, double *mecherr) {
    int Sk = K / G;
    signed char *Ai = malloc((size_t)M*K), *Bi = malloc((size_t)K*N);
    float *aS = malloc((size_t)M*Sk*4), *bS = malloc((size_t)Sk*N*4), *C = malloc((size_t)M*N*4);
    float *tmp = malloc((size_t)G*4);
    /* A: per-(row,group) optional rotate then symmetric int4 quant (lim 7) */
    for (int m = 0; m < M; m++) for (int g = 0; g < Sk; g++) {
        for (int j = 0; j < G; j++) tmp[j] = Af[(size_t)m*K + g*G + j];
        if (rotate) ork_fwht_norm(tmp, G);
        float mx = 1e-9f; for (int j = 0; j < G; j++) { float a = fabsf(tmp[j]); if (a > mx) mx = a; }
        float s = mx/7, inv = 7/mx; aS[m*Sk+g] = s;
        for (int j = 0; j < G; j++) { int q = (int)lrintf(tmp[j]*inv); if (q>7) q=7; if (q<-8) q=-8; Ai[(size_t)m*K+g*G+j] = (signed char)q; }
    }
    /* B: per-(col,group) gather (strided) then optional rotate + int4 quant */
    for (int n = 0; n < N; n++) for (int g = 0; g < Sk; g++) {
        for (int j = 0; j < G; j++) tmp[j] = Bf[(size_t)(g*G+j)*N + n];
        if (rotate) ork_fwht_norm(tmp, G);
        float mx = 1e-9f; for (int j = 0; j < G; j++) { float b = fabsf(tmp[j]); if (b > mx) mx = b; }
        float s = mx/7, inv = 7/mx; bS[g*N+n] = s;
        for (int j = 0; j < G; j++) { int q = (int)lrintf(tmp[j]*inv); if (q>7) q=7; if (q<-8) q=-8; Bi[(size_t)(g*G+j)*N+n] = (signed char)q; }
    }
    ork_w *w = ork_mm_pack_i4_grouped(c, K, N, Bi, G);
    if (!w) { printf("  pack_i4_grouped failed\n"); free(Ai);free(Bi);free(aS);free(bS);free(C);free(tmp); return -1; }
    int rc = ork_mm_run_i4_grouped(c, w, M, Ai, aS, bS, C); ork_w_free(w);
    if (rc) { printf("  run_i4_grouped rc=%d\n", rc); free(Ai);free(Bi);free(aS);free(bS);free(C);free(tmp); return -1; }
    double se = 0, sr = 0, me = 0;
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        double exact = 0;                                  /* per-group int4 dequant (mechanism ref) */
        for (int g = 0; g < Sk; g++) { long p = 0; for (int j = 0; j < G; j++) p += (long)Ai[(size_t)m*K+g*G+j]*Bi[(size_t)(g*G+j)*N+n];
            exact += (double)aS[m*Sk+g]*bS[g*N+n]*p; }
        double d = C[(size_t)m*N+n] - exact; if (d < 0) d = -d; if (d > me) me = d;
        double q = C[(size_t)m*N+n] - Cref[(size_t)m*N+n]; se += q*q; sr += (double)Cref[(size_t)m*N+n]*Cref[(size_t)m*N+n];
    }
    *mecherr = me;
    free(Ai);free(Bi);free(aS);free(bS);free(C);free(tmp);
    return sr > 0 ? sqrt(se/sr) : 0;
}

int main(int argc, char **argv) {
    int M = argc>1?atoi(argv[1]):8, K = argc>2?atoi(argv[2]):2048, N = argc>3?atoi(argv[3]):256, G = argc>4?atoi(argv[4]):128;
    float oscale = argc>5?atof(argv[5]):20.0f;
    if (K % G || (G & (G-1))) { printf("need K%%G==0 and G a power of 2\n"); return 1; }
    ork_npu *c = ork_npu_init(); if (!c) { printf("init failed (NPU?)\n"); return 1; }
    printf("on-NPU W4A4 Hadamard (M=%d K=%d N=%d G=%d, activation outliers x%.0f):\n", M, K, N, G, oscale);

    float *Af = malloc((size_t)M*K*4), *Bf = malloc((size_t)K*N*4), *Cref = malloc((size_t)M*N*4);
    unsigned sd = 1234;
    /* Gaussian A + injected outliers (real LLM activations: ~Gaussian with heavy-tailed outliers) */
    for (size_t i = 0; i < (size_t)M*K; i++) { sd=sd*1103515245u+12345u; float u1=((sd>>9)+1)/(float)((1u<<23)+2);
        sd=sd*1103515245u+12345u; float u2=((sd>>9)+1)/(float)((1u<<23)+2); Af[i]=sqrtf(-2*logf(u1))*cosf(6.2831853f*u2); }
    for (int k = 0; k < K; k += 37) for (int m = 0; m < M; m++) Af[(size_t)m*K+k] *= oscale;
    for (size_t i = 0; i < (size_t)K*N; i++) { sd=sd*1103515245u+12345u; float u1=((sd>>9)+1)/(float)((1u<<23)+2);
        sd=sd*1103515245u+12345u; float u2=((sd>>9)+1)/(float)((1u<<23)+2); Bf[i]=sqrtf(-2*logf(u1))*cosf(6.2831853f*u2); }
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) { double s=0; for (int k=0;k<K;k++) s+=(double)Af[(size_t)m*K+k]*Bf[(size_t)k*N+n]; Cref[(size_t)m*N+n]=s; }

    double me_p, me_h;
    double rms_plain = run_case(c, Af, Bf, Cref, M, K, N, G, 0, &me_p);
    double rms_had   = run_case(c, Af, Bf, Cref, M, K, N, G, 1, &me_h);
    if (rms_plain < 0 || rms_had < 0) { printf("FAIL: NPU run error\n"); return 1; }
    printf("  W4A4 plain     : quant RMS vs fp32 %6.2f%%   (NPU mechanism err %.4f)\n", 100*rms_plain, me_p);
    printf("  W4A4 +Hadamard : quant RMS vs fp32 %6.2f%%   (NPU mechanism err %.4f)\n", 100*rms_had, me_h);
    int mech_ok = me_p < 0.05 && me_h < 0.05;             /* NPU must match per-group int4 dequant exactly */
    int win = rms_had < rms_plain;
    printf("  %s mechanism exact on NPU; Hadamard %s plain (%.2f%% -> %.2f%%, %.0f%% of plain)\n",
           mech_ok?"OK ":"BAD", win?"BEATS":"does NOT beat", 100*rms_plain, 100*rms_had, 100*rms_had/rms_plain);
    free(Af);free(Bf);free(Cref); ork_npu_free(c);
    return (mech_ok && win) ? 0 : 1;
}
