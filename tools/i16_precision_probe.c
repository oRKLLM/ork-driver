/* tools/i16_precision_probe.c — RE the CNA precision field on the FP16 path (2-byte geometry).
 *
 * CONFIRMED: 0x100c precision bits — int8 regcmd=0 (INT8), fp16 regcmd=0x120 => proc=in=2 (FP16). So by the
 * NVDLA encoding INT16=1 (0x20000090). The earlier int8-path fuzz HUNG because int16 needs 2-BYTE geometry
 * and the int8 regcmd is 1-byte. The FP16 regcmd IS 2-byte (0x1030=K*N*2) — the geometry int16 wants. So:
 * take the WORKING fp16 matmul and override 0x100c fp16(2)->int16(1); if it RUNS (rc=0, no hang), proc=1 is a
 * valid 2-byte integer datapath => int16 is buildable as fp16-geometry + int-precision + int output stage.
 * A=B=fp16 1.0 (0x3c00); fp16 ref C=K(=64.0). int16 reinterprets the bits (expected garbage) — we need RUN.
 *   make i16_precision_probe && sudo ./i16_precision_probe 0x20000090     (0x20000120 = fp16 baseline)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    unsigned int v = argc>1 ? (unsigned int)strtoul(argv[1],NULL,0) : 0x20000120u;
    ork_npu*c=ork_npu_init(); if(!c){printf("no board\n");return 0;}
    const int M=8,K=64,N=32;
    unsigned short*A=malloc((size_t)M*K*2),*B=malloc((size_t)K*N*2); float*raw=malloc((size_t)M*N*4);
    for(int i=0;i<M*K;i++)A[i]=0x3c00; for(int i=0;i<K*N;i++)B[i]=0x3c00;   /* fp16 1.0; fp16 ref C=64.0 */
    for(int i=0;i<M*N;i++)raw[i]=-1e30f;
    ork_f16_fuzz_clear();
    if(v!=0x20000120u) ork_f16_fuzz_add(0x0201,0x100c,v);
    printf("FP16-path 0x100c=0x%08x (proc=%u in=%u): submitting...\n", v, (v>>7)&7, (v>>4)&7);
    int rc=ork_f16_npu_probe_mm(c,M,K,N,A,B,raw);
    printf("FP16-path 0x100c=0x%08x -> rc=%d raw[0..3]= %.3f %.3f %.3f %.3f   (fp16 ref=64.0)\n",
           v, rc, raw[0],raw[1],raw[2],raw[3]);
    ork_npu_free(c); free(A);free(B);free(raw);
    return 0;
}
