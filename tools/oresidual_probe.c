/* oresidual_probe — task #20 (A1+A2): attention layer BACK — O projection + residual add, resident.
 *
 *   O = attn_out . Wo   (MM_F16_F16OUT -> f16 out)   op0
 *   y = x + O           (add_f16)                    op1  (B = O = op0.C, aliased resident)
 * Bridge-free (f16-out matmul -> add f16-in, the A1 bridge) and A2-resident: under orkd only attn_out/x
 * cross in + y comes back = ONE round-trip vs 2. Same recipe as the front (rmsnorm->QKV->rope). Validates y.
 *   direct: sudo env ORK_MM_TIMEOUT=3000 ./oresidual_probe [N] [d]
 *   orkd:   sudo env ORK_USE_ORKD=1 ORKD_BIN=./orkd ORK_MM_TIMEOUT=3000 ./oresidual_probe [N] [d]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static uint32_t g_rng=0x0dd1e7u;
static float frand(void){ g_rng=g_rng*1664525u+1013904223u; return (float)(g_rng>>8)/(float)(1u<<24); }

int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):32, d=argc>2?atoi(argv[2]):128;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    int viaorkd=getenv("ORK_USE_ORKD")?1:0;
    printf("oresidual_probe: N=%d d=%d  path=%s\n", N,d, viaorkd?"ORKD (one round-trip)":"direct");

    ork_f16 *attn=malloc((size_t)N*d*sizeof(ork_f16)), *x=malloc((size_t)N*d*sizeof(ork_f16)), *Wo=malloc((size_t)d*d*sizeof(ork_f16));
    ork_f16 *O=malloc((size_t)N*d*sizeof(ork_f16)), *y=malloc((size_t)N*d*sizeof(ork_f16));
    for(size_t i=0;i<(size_t)N*d;i++){ attn[i]=(ork_f16)(frand()*2.f-1.f); x[i]=(ork_f16)(frand()*2.f-1.f); }
    for(size_t i=0;i<(size_t)d*d;i++) Wo[i]=(ork_f16)((frand()*2.f-1.f)*0.1f);
    for(size_t i=0;i<(size_t)N*d;i++){ O[i]=(ork_f16)-9e3f; y[i]=(ork_f16)-9e3f; }

    /* CPU ref: O = attn.Wo ; y = x + O */
    float *ry=malloc((size_t)N*d*4);
    for(int i=0;i<N;i++) for(int o=0;o<d;o++){ double acc=0; for(int k=0;k<d;k++) acc+=(double)(float)attn[(size_t)i*d+k]*(float)Wo[(size_t)k*d+o];
        ry[(size_t)i*d+o]=(float)x[(size_t)i*d+o]+(float)acc; }

    ork_w *wo=ork_mm_pack(c,d,d,Wo); if(!wo){ printf("pack fail\n"); return 2; }
    ork_seq_op ops[2] = {
        { .kind=ORK_OP_MM_F16_F16OUT, .w=wo, .M=N, .A=attn, .C=O },
        { .kind=ORK_OP_ADD_F16, .M=N, .N=d, .A=x, .B=O, .C=y },   /* B==op0.C (O) -> resident */
    };
    int rc=ork_submit_seq(c,ops,2);
    printf("  ork_submit_seq([O=attn.Wo (f16out), y=x+O], O aliased) rc=%d\n", rc);
    if(rc){ printf("FAIL rc=%d\n", rc); ork_npu_free(c); return 1; }

    int bad=0; double me=0;
    for(int i=0;i<N*d;i++){ double e=fabs((double)y[i]-ry[i]); if(e>me)me=e; if(e>2e-2*fabs(ry[i])+4e-3) bad++; }
    printf("  y=x+attn.Wo vs CPU: max|err|=%.3e  %s (%d/%d)\n", me, bad?"MISMATCH":"COHERENT", bad, N*d);
    int fail=bad!=0;
    printf("%s\n", fail? "FAIL — O+residual back miscomputed"
                       : "PASS — layer back (O proj + residual) resident: bridge-free f16-out matmul -> add, A2-resident, one seq");
    ork_mm_free(c,wo); ork_npu_free(c);
    return fail;
}
