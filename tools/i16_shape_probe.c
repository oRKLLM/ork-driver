/* tools/i16_shape_probe.c — localize the shape at which the on-NPU int16 activation op (ork_npu_silu_i16,
 * i.e. act_lut_i16 -> ork_npu_probe_silu_std_i16) wedges. The standalone op is bit-accurate at M=8,N=64 but
 * SOFT-RESETS the NPU inside the FFN chain at M=128,N=6144. This sweeps (M,N) to find whether the limit is
 * on M, on N, or on M*N — so the fix (internal tiling in act_lut_i16) can tile the right dimension.
 *   make i16_shape_probe && sudo ./i16_shape_probe        (board only; each wedge soft-resets + recovers)
 * rc: 0=ok, -1=wedge/timeout, -2=bad shape, -3=non-rk3588.
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int try_shape(ork_npu *c, int M, int N){
    int16_t *in = calloc((size_t)M*N, 2), *out = calloc((size_t)M*N, 2);
    if(!in||!out){ free(in); free(out); return -99; }
    for(size_t i=0;i<(size_t)M*N;i++) in[i]=(int16_t)((i%201)-100);   /* small dummy gate values */
    double us=0; int rc = ork_npu_silu_i16(c, in, M, N, 0.01, 0.01, out, &us);
    printf("  M=%5d N=%5d (M*N=%8zu)  -> rc=%d %s   (%.0f us)\n",
           M, N, (size_t)M*N, rc, rc==0?"OK":(rc==-1?"WEDGE":"shape/soc"), us);
    fflush(stdout);
    free(in); free(out);
    return rc;
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    ork_npu *c = ork_npu_init(); if(!c){ printf("no board\n"); return 0; }
    printf("i16_shape_probe: find the wedge boundary of the on-NPU int16 activation op\n");
    /* vary N at small M (isolate N) */
    printf("-- sweep N (M=8) --\n");
    int Ns[] = {64, 512, 1024, 2048, 4096, 6144, 8192};
    for(int i=0;i<7;i++) try_shape(c, 8, Ns[i]);
    /* vary M at small N (isolate M) */
    printf("-- sweep M (N=64) --\n");
    int Ms[] = {8, 32, 64, 128, 256, 512};
    for(int i=0;i<6;i++) try_shape(c, Ms[i], 64);
    /* the chain's actual shape + tiled candidates */
    printf("-- chain shape + candidates --\n");
    try_shape(c, 128, 6144);
    try_shape(c, 64, 6144);
    try_shape(c, 32, 6144);
    try_shape(c, 8, 6144);
    ork_npu_free(c);
    return 0;
}
