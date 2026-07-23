/* xmax_probe — task #20 (A1): softmax x-max broadcast-sub via proven ops (matmul-broadcast + add).
 *   -max_bc = MM_F16_F16OUT(max_padded[M,32] . -ones[32,N])   (K=32 outer product, fp16 out)
 *   shifted = add_f16(scores, -max_bc) = scores - max
 * MODE=split (default): two SEPARATE ork_submit_seq calls. ORK_XMAX_SEQ=1: one 2-op seq, -max_bc aliased.
 * ork_f16 is native _Float16 — convert directly (no bit hacks).
 *   sudo env ORK_MM_TIMEOUT=3000 ORK_EW_TIMEOUT=3000 ./xmax_probe [M] [N]     (+ORK_USE_ORKD=1 ORKD_BIN=./orkd)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static uint32_t g_rng=0x6d2b1fu;
static float frand(void){ g_rng=g_rng*1664525u+1013904223u; return (float)(g_rng>>8)/(float)(1u<<24); }

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):48, N=argc>2?atoi(argv[2]):64;
    int seqmode=getenv("ORK_XMAX_SEQ")?1:0;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    int viaorkd=getenv("ORK_USE_ORKD")?1:0;
    printf("xmax_probe: M=%d N=%d mode=%s path=%s\n", M,N, seqmode?"one-seq(aliased)":"split", viaorkd?"ORKD":"direct");

    ork_f16 *scores=malloc((size_t)M*N*sizeof(ork_f16)), *maxp=malloc((size_t)M*32*sizeof(ork_f16)), *negones=malloc((size_t)32*N*sizeof(ork_f16));
    ork_f16 *nmb=malloc((size_t)M*N*sizeof(ork_f16)), *out=malloc((size_t)M*N*sizeof(ork_f16));
    for(size_t i=0;i<(size_t)M*N;i++) scores[i]=(ork_f16)(frand()*8.f-4.f);
    for(size_t i=0;i<(size_t)M*32;i++) maxp[i]=(ork_f16)0.f;
    float *mx=malloc((size_t)M*4);
    for(int m=0;m<M;m++){ float v=(float)scores[(size_t)m*N]; for(int j=1;j<N;j++){ float s=(float)scores[(size_t)m*N+j]; if(s>v)v=s; }
        mx[m]=v; maxp[(size_t)m*32]=(ork_f16)v; }                       /* col0 = max[m], rest 0 */
    for(size_t i=0;i<(size_t)32*N;i++) negones[i]=(ork_f16)(-1.f);
    for(size_t i=0;i<(size_t)M*N;i++){ nmb[i]=(ork_f16)-12345.f; out[i]=(ork_f16)-12345.f; }

    ork_w *w=ork_mm_pack(c,32,N,negones); if(!w){ printf("pack fail\n"); return 2; }
    int rc1,rc2;
    if(seqmode){
        ork_seq_op ops[2]={ { .kind=ORK_OP_MM_F16_F16OUT, .w=w, .M=M, .A=maxp, .C=nmb },
                            { .kind=ORK_OP_ADD_F16, .M=M, .N=N, .A=scores, .B=nmb, .C=out } };
        rc1=rc2=ork_submit_seq(c,ops,2);
    } else {
        ork_seq_op o0={ .kind=ORK_OP_MM_F16_F16OUT, .w=w, .M=M, .A=maxp, .C=nmb };  rc1=ork_submit_seq(c,&o0,1);
        ork_seq_op o1={ .kind=ORK_OP_ADD_F16, .M=M, .N=N, .A=scores, .B=nmb, .C=out }; rc2=ork_submit_seq(c,&o1,1);
    }
    printf("  rc(matmul=%d add=%d)  nmb[0]=%.3f (want -max[0]=%.3f)  out[0]=%.3f\n", rc1, rc2, (float)nmb[0], -mx[0], (float)out[0]);
    if(rc1||rc2){ printf("FAIL rc\n"); ork_npu_free(c); return 1; }

    int nmb_bad=0; for(int m=0;m<M;m++) if(fabs((float)nmb[(size_t)m*N]-(-mx[m]))>2e-2*(fabs(mx[m])+1e-2)) nmb_bad++;
    int bad=0; double me=0;
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ float want=(float)scores[(size_t)m*N+n]-mx[m];
        float got=(float)out[(size_t)m*N+n]; double e=fabs(got-want); if(e>me)me=e; if(e>2e-2*(fabs(want)+1e-2)) bad++; }
    printf("  nmb(-max) bad rows=%d/%d | scores-max: max|err|=%.3e  %s (%d/%d)\n", nmb_bad, M, me, bad?"MISMATCH":"COHERENT", bad, M*N);
    int fail=bad!=0;
    printf("%s\n", fail? "FAIL — matmul-broadcast x-max miscomputed" : "PASS — x-max broadcast-sub on-NPU via proven matmul-broadcast + add");
    ork_mm_free(c,w); ork_npu_free(c);
    return fail;
}
