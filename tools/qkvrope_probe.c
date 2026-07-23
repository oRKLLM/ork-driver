/* qkvrope_probe — task #20 (A1+A2 payoff): rmsnorm -> Q/K/V -> rope(Q),rope(K) as ONE resident seq.
 *
 * The bridge-free contiguous front of an attention layer:
 *   xn = RMSNorm(x, gain)                       [f16]   op0
 *   Q  = xn . Wq   (MM_F16_F16OUT -> f16 out)            op1  (A = xn = op0.C, resident)
 *   K  = xn . Wk   (MM_F16_F16OUT)                       op2  (A = xn, resident)
 *   V  = xn . Wv   (MM_F16_F16OUT)                       op3  (A = xn, resident)   [returned]
 *   Qr = rope(Q)   (ROPE_NEOX_F16, reads Q's f16 DIRECTLY — A1 no f32->f16 bridge)  op4 (A = op1.C, resident)
 *   Kr = rope(K)                                         op5 (A = op2.C, resident)
 * Every op fp16, every transition bridge-free (rmsnorm f16 -> matmul f16-in; matmul f16-OUT -> rope f16-in).
 * xn/Q/K alias resident (A2): under orkd only x/gain/pos cross the socket in and V/Qr/Kr come back — ONE
 * round-trip for the whole front, vs 6 per-op round-trips. Validates V/Qr/Kr vs CPU.
 *
 *   direct: sudo env ORK_MM_TIMEOUT=3000 ./qkvrope_probe [N] [d]
 *   orkd:   sudo env ORK_USE_ORKD=1 ORKD_BIN=./orkd ORK_MM_TIMEOUT=3000 ./qkvrope_probe [N] [d]
 * Exit 0 = coherent.
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static uint32_t g_rng=0x51c0de1u;
static float frand(void){ g_rng=g_rng*1664525u+1013904223u; return (float)(g_rng>>8)/(float)(1u<<24); }

int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):32, d=argc>2?atoi(argv[2]):128;   /* N tokens, d head dim (%32 matmul, %8 rope) */
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    int viaorkd=getenv("ORK_USE_ORKD")?1:0;
    printf("qkvrope_probe: N=%d d=%d  path=%s\n", N,d, viaorkd?"ORKD (one round-trip)":"direct");
    float eps=1e-5f, freq_base=10000.f;

    ork_f16 *x=malloc((size_t)N*d*sizeof(ork_f16)), *gain=malloc((size_t)d*sizeof(ork_f16));
    ork_f16 *Wq=malloc((size_t)d*d*sizeof(ork_f16)), *Wk=malloc((size_t)d*d*sizeof(ork_f16)), *Wv=malloc((size_t)d*d*sizeof(ork_f16));
    int *pos=malloc((size_t)N*sizeof(int));
    for(size_t i=0;i<(size_t)N*d;i++) x[i]=(ork_f16)(frand()*2.f-1.f);
    for(int i=0;i<d;i++) gain[i]=(ork_f16)(0.5f+frand());
    for(size_t i=0;i<(size_t)d*d;i++){ Wq[i]=(ork_f16)((frand()*2.f-1.f)*0.1f); Wk[i]=(ork_f16)((frand()*2.f-1.f)*0.1f); Wv[i]=(ork_f16)((frand()*2.f-1.f)*0.1f); }
    for(int i=0;i<N;i++) pos[i]=i+1;

    /* CPU reference */
    float *rV=malloc((size_t)N*d*4), *rQr=malloc((size_t)N*d*4), *rKr=malloc((size_t)N*d*4);
    { float *xn=malloc((size_t)N*d*4), *Q=malloc((size_t)N*d*4), *K=malloc((size_t)N*d*4);
      for(int i=0;i<N;i++){ double ms=0; for(int k=0;k<d;k++){ float v=(float)x[(size_t)i*d+k]; ms+=(double)v*v; } float r=1.0f/sqrtf((float)(ms/d)+eps);
          for(int k=0;k<d;k++) xn[(size_t)i*d+k]=(float)x[(size_t)i*d+k]*r*(float)gain[k]; }
      for(int i=0;i<N;i++) for(int o=0;o<d;o++){ double q=0,kk=0,vv=0; for(int k=0;k<d;k++){ float a=xn[(size_t)i*d+k];
          q+=(double)a*(float)Wq[(size_t)k*d+o]; kk+=(double)a*(float)Wk[(size_t)k*d+o]; vv+=(double)a*(float)Wv[(size_t)k*d+o]; }
          Q[(size_t)i*d+o]=(float)q; K[(size_t)i*d+o]=(float)kk; rV[(size_t)i*d+o]=(float)vv; }
      int hd2=d/2; for(int r=0;r<N;r++){ double p=pos[r]; for(int i=0;i<hd2;i++){ double th=p*pow((double)freq_base,-2.0*i/(double)d);
          float cc=cos(th),ss=sin(th); float qa=Q[(size_t)r*d+i],qb=Q[(size_t)r*d+i+hd2],ka=K[(size_t)r*d+i],kb=K[(size_t)r*d+i+hd2];
          rQr[(size_t)r*d+i]=qa*cc-qb*ss; rQr[(size_t)r*d+i+hd2]=qb*cc+qa*ss;
          rKr[(size_t)r*d+i]=ka*cc-kb*ss; rKr[(size_t)r*d+i+hd2]=kb*cc+ka*ss; } }
      free(xn);free(Q);free(K); }

    ork_w *wq=ork_mm_pack(c,d,d,Wq), *wk=ork_mm_pack(c,d,d,Wk), *wv=ork_mm_pack(c,d,d,Wv);
    if(!wq||!wk||!wv){ printf("pack fail\n"); return 2; }
    ork_f16 *xn=malloc((size_t)N*d*sizeof(ork_f16)), *Q=malloc((size_t)N*d*sizeof(ork_f16)), *K=malloc((size_t)N*d*sizeof(ork_f16));
    ork_f16 *V=malloc((size_t)N*d*sizeof(ork_f16)), *Qr=malloc((size_t)N*d*sizeof(ork_f16)), *Kr=malloc((size_t)N*d*sizeof(ork_f16));
    for(size_t i=0;i<(size_t)N*d;i++){ V[i]=(ork_f16)-9e3f; Qr[i]=(ork_f16)-9e3f; Kr[i]=(ork_f16)-9e3f; }

    ork_seq_op ops[6] = {
        { .kind=ORK_OP_RMSNORM_F16,    .M=N, .N=d, .A=x,  .B=gain, .C=xn, .in_scale=eps },
        { .kind=ORK_OP_MM_F16_F16OUT,  .w=wq, .M=N, .A=xn, .C=Q },     /* A==op0.C (xn) resident */
        { .kind=ORK_OP_MM_F16_F16OUT,  .w=wk, .M=N, .A=xn, .C=K },
        { .kind=ORK_OP_MM_F16_F16OUT,  .w=wv, .M=N, .A=xn, .C=V },
        { .kind=ORK_OP_ROPE_NEOX_F16,  .M=N, .N=d, .A=Q, .B=pos, .C=Qr, .in_scale=freq_base },  /* A==op1.C (Q) resident */
        { .kind=ORK_OP_ROPE_NEOX_F16,  .M=N, .N=d, .A=K, .B=pos, .C=Kr, .in_scale=freq_base },  /* A==op2.C (K) resident */
    };
    int rc=ork_submit_seq(c,ops,6);
    printf("  ork_submit_seq(6 ops: rmsnorm->QKV->rope,rope; xn/Q/K aliased) rc=%d\n", rc);
    if(rc){ printf("FAIL rc=%d\n", rc); ork_npu_free(c); return 1; }

    int bad=0; double me=0;
    #define CHK(BUF,REF) for(int i=0;i<N*d;i++){ double e=fabs((double)(BUF)[i]-(REF)[i]); if(e>me)me=e; if(e>2e-2*fabs((REF)[i])+4e-3) bad++; }  /* 4e-3 abs floor = fp16 rounding over a 4-op chain */
    CHK(V,rV); CHK(Qr,rQr); CHK(Kr,rKr);
    #undef CHK
    printf("  V/Qr/Kr vs CPU: max|err|=%.3e  %s (%d/%d)\n", me, bad?"MISMATCH":"COHERENT", bad, 3*N*d);
    int fail=bad!=0;
    printf("%s\n", fail? "FAIL — resident QKV+rope front miscomputed"
                       : "PASS — rmsnorm->QKV->rope resident chain: bridge-free (A1 f16-out matmul->rope), A2-resident, one seq");
    ork_mm_free(c,wq);ork_mm_free(c,wk);ork_mm_free(c,wv); ork_npu_free(c);
    return fail;
}
