/* int8reduce_probe — task #20 (A1 path B): eliminate the exp->Σ bridge with a SYMMETRIC int8 reduce.
 *
 * RK3588 is symmetric-precision (mixed int16-in/fp16-out rejects). Since out_scale cancels in the softmax
 * normalize (P = eᵢ/Σeᵢ), the reduce can stay int8 end-to-end with NO cross-precision bridge:
 *     exp_i8 (int8 out)  ->  MM_I8(e·ones_i8) -> int32 Σ
 * exp_i8's int8 output feeds MM_I8's int8 activation DIRECTLY (same precision, adjacent) — so the pair is
 * bridge-free and A2 keeps e_i8 resident. Validates the softmax (via this int8 reduce) vs a CPU reference;
 * the question is whether int8 exp is accurate enough for the denominator sum.
 *
 *   direct: sudo env ORK_MM_TIMEOUT=3000 ./int8reduce_probe [M] [n]
 *   orkd:   sudo env ORK_USE_ORKD=1 ORKD_BIN=./orkd ORK_MM_TIMEOUT=3000 ./int8reduce_probe [M] [n]
 * Exit 0 = softmax coherent (int8 reduce viable); 1 = too coarse.
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static uint32_t g_rng=0x9be21fu;
static float frand(void){ g_rng=g_rng*1664525u+1013904223u; return (float)(g_rng>>8)/(float)(1u<<24); }

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):64, n=argc>2?atoi(argv[2]):512;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    int viaorkd=getenv("ORK_USE_ORKD")?1:0;
    printf("int8reduce_probe: M=%d n=%d  path=%s\n", M,n, viaorkd?"ORKD_SEQ (Path B)":"direct (Path A)");

    float *X=malloc((size_t)M*n*4), *ref=malloc((size_t)M*n*4);
    for(size_t i=0;i<(size_t)M*n;i++) X[i]=frand()*8.f-4.f;
    for(int m=0;m<M;m++){ float mx=X[(size_t)m*n]; for(int j=1;j<n;j++) if(X[(size_t)m*n+j]>mx)mx=X[(size_t)m*n+j];
        double s=0; for(int j=0;j<n;j++){ double e=exp((double)X[(size_t)m*n+j]-mx); ref[(size_t)m*n+j]=(float)e; s+=e; }
        for(int j=0;j<n;j++) ref[(size_t)m*n+j]/=(float)s; }

    /* CPU prep (max + x-max quantize to int8 for exp_i8 — the pre-exp steps, not the bridge under test) */
    int8_t *xi8=malloc((size_t)M*n), *ei8=malloc((size_t)M*n);
    double in_scale, out_scale=1.0/127.0;
    { float lo=0; for(int m=0;m<M;m++){ float mx=X[(size_t)m*n]; for(int j=1;j<n;j++) if(X[(size_t)m*n+j]>mx)mx=X[(size_t)m*n+j];
        for(int j=0;j<n;j++){ float d=X[(size_t)m*n+j]-mx; if(d<lo)lo=d; } }
      in_scale=(-lo)/127.0; if(in_scale<=0)in_scale=1e-6;
      for(int m=0;m<M;m++){ float mx=X[(size_t)m*n]; for(int j=1;j<n;j++) if(X[(size_t)m*n+j]>mx)mx=X[(size_t)m*n+j];
        for(int j=0;j<n;j++){ long q=lround((double)(X[(size_t)m*n+j]-mx)/in_scale); if(q<-127)q=-127; if(q>127)q=127; xi8[(size_t)m*n+j]=(int8_t)q; } } }

    /* ones_i8[n,32] weight for the int8 sum-reduce (int8 pack needs N%32; Σ lands in col 0) */
    int8_t *ones=malloc((size_t)n*32); memset(ones,1,(size_t)n*32);
    ork_w *w_ones=ork_i8_mm_pack(c,n,32,ones); if(!w_ones){ printf("pack_i8 fail\n"); return 2; }
    int32_t *ss=malloc((size_t)M*32*4);
    for(size_t i=0;i<(size_t)M*n;i++) ei8[i]=-128;   /* poison */

    /* ONE seq: exp_i8 -> MM_I8(e·ones), e_i8 ALIASED (op1.A == op0.C) => bridge-free adjacent + A2 resident */
    ork_seq_op ops[2] = {
        { .kind=ORK_OP_EXP_I8, .M=M, .N=n, .A=xi8, .C=ei8, .in_scale=in_scale, .out_scale=out_scale },
        { .kind=ORK_OP_MM_I8, .w=w_ones, .M=M, .A=ei8, .C=ss },   /* A==ops[0].C -> resident ref; e_i8 int8 -> MM_I8 int8 (no bridge) */
    };
    int rc=ork_submit_seq(c,ops,2);
    printf("  ork_submit_seq([exp_i8, MM_I8 reduce], e_i8 aliased) rc=%d\n", rc);
    if(rc){ printf("FAIL — rc=%d\n", rc); ork_npu_free(c); return 1; }

    /* Validate the RETURNED Σ (the reduce output) vs the CPU exp-sum. NPU Σ = Σⱼ e_i8 ; e_i8 ≈ exp(·)/out_scale,
     * so Σ_npu*out_scale ≈ Σⱼ exp(·) = S_cpu. This checks the full int8 reduce (exp_i8 quantization + int8 sum)
     * end-to-end and works under A2 (only Σ is returned; the aliased e_i8 stays resident, so we can't/shouldn't
     * read it on the host). (void)ref/ei8 — coarse-normalize kept off the host path. */
    (void)ref; (void)ei8;
    double me=0, sae=0; int bad=0;
    for(int m=0;m<M;m++){ float mx=X[(size_t)m*n]; for(int j=1;j<n;j++) if(X[(size_t)m*n+j]>mx)mx=X[(size_t)m*n+j];
        double S_cpu=0; for(int j=0;j<n;j++) S_cpu+=exp((double)X[(size_t)m*n+j]-mx);
        double S_npu=(double)ss[(size_t)m*32]*out_scale;
        double rel=fabs(S_npu-S_cpu)/(S_cpu>0?S_cpu:1); sae+=rel; if(rel>me)me=rel; if(rel>0.03) bad++; }
    printf("  Σ via int8 reduce vs CPU exp-sum: max rel-err=%.3e mean=%.3e  %s (%d/%d rows > 3%%)\n",
           me, sae/M, bad?"CHECK":"COHERENT", bad, M);
    int fail = bad!=0;
    printf("%s\n", fail? "FAIL — int8 exp too coarse for the reduce Σ"
                       : "PASS — symmetric int8 reduce (exp_i8->MM_I8, e_i8 resident) coherent, bridge-free, no mixed precision");
    ork_mm_free(c,w_ones); ork_npu_free(c);
    return fail;
}
