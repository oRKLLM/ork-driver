/* Isolate ork_mm_pack_i8_f32: pack the SAME logical weights via pack_i8 (reference, int8 input) and
 * pack_i8_f32 (f32 input + NEON quant), run_i8 both, compare. Progress prints pinpoint pack-vs-run and
 * which K wedges.  make pack_f32_probe && sudo ./pack_f32_probe [K] [N] [M] */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "ork_npu.h"

static unsigned sd = 99; static int r8(void){ sd = sd*1103515245+12345; return (int)((sd>>16)&0xff)-128; }

int main(int argc, char **argv) {
    int K = argc>1?atoi(argv[1]):512, N = argc>2?atoi(argv[2]):128, M = argc>3?atoi(argv[3]):4;
    ork_npu *c = ork_npu_init(); if (!c) { printf("init failed\n"); return 1; }
    printf("shape K=%d N=%d M=%d\n", K, N, M); fflush(stdout);

    float  *Bf = malloc((size_t)N*K*sizeof(float));   /* [N][K] n-major */
    int8_t *Bi = malloc((size_t)K*N);                 /* [K][N] for pack_i8 reference */
    float  *bsc = malloc((size_t)N*sizeof(float));
    /* per-channel quantize f32 -> int8 the SAME way pack_i8_f32 does, into the [K][N] ref layout */
    for (int n = 0; n < N; n++) {
        float mx=1e-9f; for (int k=0;k<K;k++){ float v=((r8())/64.0f); Bf[(size_t)n*K+k]=v; if(fabsf(v)>mx)mx=fabsf(v); }
        float iv=127.0f/mx;
        for (int k=0;k<K;k++){ int q=(int)lrintf(Bf[(size_t)n*K+k]*iv); Bi[(size_t)k*N+n]=(int8_t)(q>127?127:q<-127?-127:q); }
    }
    int8_t *Ai = malloc((size_t)M*K); for (size_t j=0;j<(size_t)M*K;j++) Ai[j]=(int8_t)r8();
    int32_t *C1 = malloc((size_t)M*N*4), *C2 = malloc((size_t)M*N*4);

    printf("[1] pack_i8 (reference)...\n"); fflush(stdout);
    ork_w *w1 = ork_mm_pack_i8(c, K, N, Bi);
    printf("[1] packed=%p; run_i8...\n", (void*)w1); fflush(stdout);
    int rc1 = ork_mm_run_i8(c, w1, M, Ai, C1);
    printf("[1] pack_i8 rc=%d C[0..2]=%d,%d,%d\n", rc1, C1[0], C1[1], C1[2]); fflush(stdout);

    printf("[2] pack_i8_f32 (NEON)...\n"); fflush(stdout);
    ork_w *w2 = ork_mm_pack_i8_f32(c, K, N, Bf, bsc);
    printf("[2] packed=%p; run_i8...\n", (void*)w2); fflush(stdout);
    int rc2 = ork_mm_run_i8(c, w2, M, Ai, C2);
    printf("[2] pack_i8_f32 rc=%d C[0..2]=%d,%d,%d\n", rc2, C2[0], C2[1], C2[2]); fflush(stdout);

    int mism = 0; if (!rc1 && !rc2) for (int i=0;i<M*N;i++) if (C1[i]!=C2[i]) mism++;
    printf("RESULT: rc1=%d rc2=%d mism=%d %s\n", rc1, rc2, mism, (!rc1&&!rc2&&!mism)?"MATCH":"DIFFER");
    return 0;
}
