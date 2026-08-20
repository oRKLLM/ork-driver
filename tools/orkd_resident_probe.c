/* orkd_resident_probe — task #20 (A2): validate on-device intermediate residency via back-reference.
 *
 * Batches a dependent, bridge-free sub-chain into ONE ork_submit_seq call:
 *     xn = RMSNorm(x, gain)      [f16]        <- op0
 *     Q  = xn . Wq               [f32]        <- op1  (A = xn == op0.C : aliased)
 *     K  = xn . Wk               [f32]        <- op2  (A = xn : aliased)
 *     V  = xn . Wv               [f32]        <- op3  (A = xn : aliased)
 * The three matmuls read rmsnorm's f16 output directly (no dtype bridge). By aliasing each matmul's A to the
 * SAME xn buffer (op0's C), the client's Path B turns them into back-references: under orkd, xn is uploaded
 * ZERO times (resident) and never shipped back (c_keep) — only x/gain go up and Q/K/V come down, in one
 * round-trip. Correct Q/K/V under orkd proves the daemon fed the resident intermediate forward.
 *
 * Run BOTH ways:
 *   direct: sudo env ORK_MM_TIMEOUT=3000 ./orkd_resident_probe
 *   orkd:   sudo env ORK_USE_ORKD=1 ORKD_BIN=./orkd ORK_MM_TIMEOUT=3000 ./orkd_resident_probe
 * Exit 0 = coherent.
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static uint32_t g_rng=0x77abc1u;
static float frand(void){ g_rng=g_rng*1664525u+1013904223u; return (float)(g_rng>>8)/(float)(1u<<24); }
static float rn(void){ return frand()*2.f-1.f; }

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):64, d=argc>2?atoi(argv[2]):128;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    int viaorkd=getenv("ORK_USE_ORKD")?1:0;
    printf("orkd_resident_probe: M=%d d=%d  path=%s\n", M,d, viaorkd?"ORKD_SEQ (Path B)":"direct (Path A)");
    float eps=1e-5f;

    ork_f16 *x=malloc((size_t)M*d*2), *gain=malloc((size_t)d*2);
    ork_f16 *Wq=malloc((size_t)d*d*2), *Wk=malloc((size_t)d*d*2), *Wv=malloc((size_t)d*d*2);
    for(size_t i=0;i<(size_t)M*d;i++) x[i]=(ork_f16)rn();
    for(int i=0;i<d;i++) gain[i]=(ork_f16)(0.5f+frand());
    for(size_t i=0;i<(size_t)d*d;i++){ Wq[i]=(ork_f16)(rn()*0.1f); Wk[i]=(ork_f16)(rn()*0.1f); Wv[i]=(ork_f16)(rn()*0.1f); }

    /* CPU reference */
    float *rQ=malloc((size_t)M*d*4), *rK=malloc((size_t)M*d*4), *rV=malloc((size_t)M*d*4);
    { float *xn=malloc((size_t)M*d*4);
      for(int i=0;i<M;i++){ double ms=0; for(int k=0;k<d;k++){ float v=(float)x[(size_t)i*d+k]; ms+=(double)v*v; } float r=1.0f/sqrtf((float)(ms/d)+eps);
          for(int k=0;k<d;k++) xn[(size_t)i*d+k]=(float)x[(size_t)i*d+k]*r*(float)gain[k]; }
      for(int i=0;i<M;i++) for(int o=0;o<d;o++){ double q=0,kk=0,vv=0; for(int k=0;k<d;k++){ float a=xn[(size_t)i*d+k];
          q+=(double)a*(float)Wq[(size_t)k*d+o]; kk+=(double)a*(float)Wk[(size_t)k*d+o]; vv+=(double)a*(float)Wv[(size_t)k*d+o]; }
          rQ[(size_t)i*d+o]=(float)q; rK[(size_t)i*d+o]=(float)kk; rV[(size_t)i*d+o]=(float)vv; }
      free(xn); }

    ork_w *wq=ork_f16_mm_pack(c,d,d,Wq), *wk=ork_f16_mm_pack(c,d,d,Wk), *wv=ork_f16_mm_pack(c,d,d,Wv);
    if(!wq||!wk||!wv){ printf("pack fail\n"); return 2; }

    /* xn is op0's OUTPUT and op1/2/3's INPUT — the same buffer (aliased) => A2 back-reference. */
    ork_f16 *xn=malloc((size_t)M*d*2);
    float *Q=malloc((size_t)M*d*4), *K=malloc((size_t)M*d*4), *V=malloc((size_t)M*d*4);
    for(size_t i=0;i<(size_t)M*d;i++){ Q[i]=K[i]=V[i]=-1e30f; }
    ork_seq_op ops[4] = {
        { .kind=ORK_OP_RMSNORM_F16, .M=M, .N=d, .A=x,  .B=gain, .C=xn, .in_scale=eps },
        { .kind=ORK_OP_MM_F16, .w=wq, .M=M, .A=xn, .C=Q },   /* A==ops[0].C -> resident ref */
        { .kind=ORK_OP_MM_F16, .w=wk, .M=M, .A=xn, .C=K },
        { .kind=ORK_OP_MM_F16, .w=wv, .M=M, .A=xn, .C=V },
    };
    int rc=ork_submit_seq(c,ops,4);
    printf("  ork_submit_seq(4 ops, xn aliased across op1-3) rc=%d\n", rc);
    if(rc){ printf("FAIL — rc=%d\n", rc); ork_npu_free(c); return 1; }

    int bad=0; double me=0;
    for(size_t i=0;i<(size_t)M*d;i++){ double eq=fabs((double)Q[i]-rQ[i]), ek=fabs((double)K[i]-rK[i]), ev=fabs((double)V[i]-rV[i]);
        if(eq>me)me=eq; if(ek>me)me=ek; if(ev>me)me=ev;
        if(eq>2e-2||ek>2e-2||ev>2e-2) bad++; }
    printf("  Q/K/V vs CPU: max|err|=%.3e  %s (%d/%d off)\n", me, bad?"MISMATCH":"COHERENT", bad, 3*M*d);

    /* transport note: xn (M*d*2 bytes) is uploaded 0x (resident) instead of 3x and never shipped back. */
    long xn_b=(long)M*d*2;
    printf("  transport: xn resident -> saved ~%ld KB (3x no re-upload + 1x no download) vs non-resident\n", (4*xn_b)/1024);

    int fail = bad!=0;
    printf("%s\n", fail? "FAIL — A2 resident chain miscomputed" : "PASS — A2 intermediate residency: xn fed forward on-device, Q/K/V coherent");
    ork_mm_free(c,wq); ork_mm_free(c,wk); ork_mm_free(c,wv);
    ork_npu_free(c);
    return fail;
}
