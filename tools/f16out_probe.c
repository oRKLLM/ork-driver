/* f16out_probe — task #20 (A1): fp16 matmul with contiguous fp16 output (ORK_OP_MM_F16_F16OUT) feeds an
 * fp16 SDP op with NO f32->f16 host bridge, so the pair is adjacent and A2 keeps the intermediate resident.
 *
 * Stage 1: MM_F16_F16OUT alone (t = A.W, fp16 out) vs CPU  — isolates the new op.
 * Stage 2: MM_F16_F16OUT -> mul_perchan_f16, intermediate t ALIASED (op1.A == op0.C) in ONE ork_submit_seq
 *          — A1 (no bridge) + A2 (t resident, not shipped). Validates out = (A.W) (x) scale vs CPU.
 *
 *   direct: sudo env ORK_MM_TIMEOUT=3000 ./f16out_probe
 *   orkd:   sudo env ORK_USE_ORKD=1 ORKD_BIN=./orkd ORK_MM_TIMEOUT=3000 ./f16out_probe
 * Exit 0 = coherent.
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* half<->float (IEEE fp16) for building operands + CPU reference */
static float h2f(uint16_t h){ uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,f; if(e==0){ if(!m)f=s<<31; else{ e=127-15+1; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; f=(s<<31)|(e<<23)|(m<<13);} } else if(e==31) f=(s<<31)|0x7f800000|(m<<13); else f=(s<<31)|((e-15+127)<<23)|(m<<13); float o; memcpy(&o,&f,4); return o; }
static uint16_t f2h(float x){ uint32_t f; memcpy(&f,&x,4); uint32_t s=(f>>16)&0x8000,e=(f>>23)&0xff,m=f&0x7fffff; if(e>=143){ if(e==255&&m) return s|0x7e00; return s|0x7c00; } if(e<=112){ if(e<103) return s; m|=0x800000; int sh=113-e; uint32_t r=m>>sh; if((m>>(sh-1))&1)r++; return s|r; } uint16_t r=s|((e-112)<<10)|(m>>13); if((m>>12)&1)r++; return r; }

static uint32_t g_rng=0x33d1e7u;
static float frand(void){ g_rng=g_rng*1664525u+1013904223u; return (float)(g_rng>>8)/(float)(1u<<24); }

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):48, K=argc>2?atoi(argv[2]):128, N=argc>3?atoi(argv[3]):128;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    int viaorkd=getenv("ORK_USE_ORKD")?1:0;
    printf("f16out_probe: M=%d K=%d N=%d  path=%s\n", M,K,N, viaorkd?"ORKD_SEQ (Path B)":"direct (Path A)");
    int fail=0;

    ork_f16 *A=malloc((size_t)M*K*2), *Wf=malloc((size_t)K*N*2), *scale=malloc((size_t)N*2);
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(ork_f16)f2h(frand()*2.f-1.f);
    for(size_t i=0;i<(size_t)K*N;i++) Wf[i]=(ork_f16)f2h((frand()*2.f-1.f)*0.1f);
    for(int j=0;j<N;j++) scale[j]=(ork_f16)f2h(frand()*2.f-1.f);

    /* CPU reference: t = A.W (fp32 acc), then narrow to fp16; out = t (x) scale (per-channel) */
    float *tref=malloc((size_t)M*N*4), *oref=malloc((size_t)M*N*4);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double s=0; for(int k=0;k<K;k++) s+=(double)h2f(((uint16_t*)A)[(size_t)m*K+k])*h2f(((uint16_t*)Wf)[(size_t)k*N+n]);
        float tf=h2f(f2h((float)s)); tref[(size_t)m*N+n]=tf; oref[(size_t)m*N+n]=tf*h2f(((uint16_t*)scale)[n]); }

    ork_w *w=ork_mm_pack(c,K,N,Wf); if(!w){ printf("pack fail\n"); return 2; }

    /* Stage 1: f16-out matmul alone */
    ork_f16 *t=malloc((size_t)M*N*2); for(size_t i=0;i<(size_t)M*N;i++) t[i]=(ork_f16)f2h(-1e3f);
    { ork_seq_op o={ .kind=ORK_OP_MM_F16_F16OUT, .w=w, .M=M, .A=A, .C=t }; int rc=ork_submit_seq(c,&o,1);
      int bad=0; double me=0; for(size_t i=0;i<(size_t)M*N;i++){ double e=fabs((double)h2f(((uint16_t*)t)[i])-tref[i]); if(e>me)me=e; if(e>2e-2*(fabs(tref[i])+1e-2)) bad++; }
      printf("  [1] MM_F16_F16OUT alone       rc=%d  max|err|=%.3e  %s (%d/%d)\n", rc, me, (!rc&&!bad)?"OK":"MISMATCH", bad, M*N); if(rc||bad) fail=1; }

    /* Stage 2: MM_F16_F16OUT -> mul_perchan, intermediate aliased (A2 resident under orkd) */
    ork_f16 *t2=malloc((size_t)M*N*2), *out=malloc((size_t)M*N*2);
    for(size_t i=0;i<(size_t)M*N;i++){ t2[i]=(ork_f16)f2h(-1e3f); out[i]=(ork_f16)f2h(-1e3f); }
    { ork_seq_op ops[2]={
        { .kind=ORK_OP_MM_F16_F16OUT, .w=w, .M=M, .A=A, .C=t2 },
        { .kind=ORK_OP_MUL_PERCHANNEL_F16, .M=M, .N=N, .A=t2, .B=scale, .C=out },   /* A==ops[0].C -> resident ref */
      };
      int rc=ork_submit_seq(c,ops,2);
      int bad=0; double me=0; for(size_t i=0;i<(size_t)M*N;i++){ double e=fabs((double)h2f(((uint16_t*)out)[i])-oref[i]); if(e>me)me=e; if(e>3e-2*(fabs(oref[i])+1e-2)) bad++; }
      printf("  [2] MM_F16_F16OUT->mul_perchan rc=%d  max|err|=%.3e  %s (%d/%d) [t resident, no f32 bridge]\n", rc, me, (!rc&&!bad)?"OK":"MISMATCH", bad, M*N); if(rc||bad) fail=1; }

    printf("%s\n", fail? "FAIL — f16-out matmul / A1 bridge miscomputed"
                       : "PASS — A1: fp16-out matmul feeds fp16 SDP bridge-free (adjacent + A2-resident)");
    ork_mm_free(c,w); ork_npu_free(c);
    return fail;
}
