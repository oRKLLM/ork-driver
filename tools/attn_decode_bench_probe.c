/* attn_decode_bench_probe — DECODE attention, HOST-SPLIT NPU path (Tier 12e) vs CPU, with ARBITRARY scores.
 * One decode step: NH heads, single query row (Nq=1) attending to L cached keys. Per head the two heavy matmuls
 * (QK^T, then weighted e.V) run on the NPU; the [1,L] softmax runs on the HOST in fp with a REAL per-head
 * max-subtraction — so it is correct for arbitrary (signed) scores, NOT just the post-max domain. This mirrors
 * examples/decode.c:attn_decode_npu. Checks coherence vs a pure-fp CPU reference and times NPU-split vs CPU.
 * (Per-step K^T/V packing is INSIDE the timed loop — real-decode-honest; the fused single-submit chain is the
 * separate prefill track.)  sudo env ORK_MM_TIMEOUT=3000 ./attn_decode_bench_probe [NH] [L] [HD] [reps]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3 + t.tv_nsec/1e6; }

/* one head's NPU host-split attention: QK^T (NPU) -> host softmax -> e.V (NPU) -> att[HD] (fp) */
static int head_npu(ork_npu*c,const int8_t*Q,const float*Kf,const float*Vf,int L,int HD,float scale,float*att,
                    int8_t*KTp,int8_t*Vp,int8_t*w8,int32_t*scores,int32_t*attv,double*sc,float qs){
    int Kp=512;
    float kmax=1e-6f,vmax=1e-6f;
    for(int i=0;i<L*HD;i++){ float ka=fabsf(Kf[i]),va=fabsf(Vf[i]); if(ka>kmax)kmax=ka; if(va>vmax)vmax=va; }
    float ks=127.0f/kmax, vs=127.0f/vmax;
    for(int e=0;e<HD;e++)for(int j=0;j<L;j++) KTp[(size_t)e*L+j]=(int8_t)lrintf(Kf[(size_t)j*HD+e]*ks);
    int8_t Qp[512]; memset(Qp,0,Kp); for(int e=0;e<HD;e++) Qp[e]=Q[e];
    ork_w *wkt=ork_mm_pack_i8(c,Kp,L,KTp); if(!wkt) return -2;
    ork_mm_task_i8 t1={ wkt,1,Qp,scores }; int rc=ork_mm_run_chain_i8(c,1,&t1); ork_w_free(wkt); if(rc)return rc;
    double mx=-1e300; for(int j=0;j<L;j++){ sc[j]=(double)scores[j]/((double)qs*ks)*scale; if(sc[j]>mx)mx=sc[j]; }
    double Z=0; for(int j=0;j<L;j++){ sc[j]=exp(sc[j]-mx); Z+=sc[j]; } if(Z<=0)Z=1;
    double wmax=0; for(int j=0;j<L;j++){ sc[j]/=Z; if(sc[j]>wmax)wmax=sc[j]; }   /* ws=127/max: softmax weights ~1/L underflow int8 at ws=127 */
    double ws=127.0/(wmax>1e-9?wmax:1.0);
    for(int j=0;j<L;j++){ int wi=(int)lrint(sc[j]*ws); w8[j]=(int8_t)(wi>127?127:(wi<0?0:wi)); }
    for(int j=0;j<L;j++)for(int e=0;e<HD;e++) Vp[(size_t)j*HD+e]=(int8_t)lrintf(Vf[(size_t)j*HD+e]*vs);
    ork_w *wv=ork_mm_pack_i8(c,L,HD,Vp); if(!wv) return -2;
    ork_mm_task_i8 t2={ wv,1,w8,attv }; rc=ork_mm_run_chain_i8(c,1,&t2); ork_w_free(wv); if(rc)return rc;
    for(int e=0;e<HD;e++) att[e]=(float)((double)attv[e]/(ws*vs));
    return 0;
}
int main(int argc,char**argv){
    int NH=argc>1?atoi(argv[1]):32, L=argc>2?atoi(argv[2]):512, HD=argc>3?atoi(argv[3]):128, REP=argc>4?atoi(argv[4]):20;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("attn_decode_bench_probe: NH=%d L=%d HD=%d — HOST-SPLIT NPU decode attn (ARBITRARY scores) vs CPU\n",NH,L,HD);
    float scale=1.0f/sqrtf((float)HD);
    /* per-head ARBITRARY (signed) Q/K/V in fp */
    float **Qf=calloc(NH,sizeof*Qf),**Kf=calloc(NH,sizeof*Kf),**Vf=calloc(NH,sizeof*Vf);
    int8_t **Q8=calloc(NH,sizeof*Q8); float *qsv=calloc(NH,sizeof(float));
    uint32_t g=0x77;
    for(int h=0;h<NH;h++){ Qf[h]=malloc((size_t)HD*4); Kf[h]=malloc((size_t)L*HD*4); Vf[h]=malloc((size_t)L*HD*4);
        for(int e=0;e<HD;e++){ g=g*1664525u+1013904223u; Qf[h][e]=((int)((g>>26)%17)-8)*0.1f; }
        for(size_t i=0;i<(size_t)L*HD;i++){ g=g*1664525u+1013904223u; Kf[h][i]=((int)((g>>26)%17)-8)*0.1f; }
        for(size_t i=0;i<(size_t)L*HD;i++){ g=g*1664525u+1013904223u; Vf[h][i]=((int)((g>>26)%17)-8)*0.1f; }
        float qmax=1e-6f; for(int e=0;e<HD;e++){ float a=fabsf(Qf[h][e]); if(a>qmax)qmax=a; } qsv[h]=127.0f/qmax;
        Q8[h]=malloc((size_t)HD); for(int e=0;e<HD;e++) Q8[h][e]=(int8_t)lrintf(Qf[h][e]*qsv[h]);
    }
    int8_t *KTp=malloc((size_t)512*L),*Vp=malloc((size_t)L*HD),*w8=malloc((size_t)L);
    int32_t *scores=malloc((size_t)L*4),*attv=malloc((size_t)HD*4); double *sc=malloc((size_t)L*sizeof(double));
    float *att=malloc((size_t)HD*4), *catt=malloc((size_t)HD*4);
    /* warm cores + coherence check (NPU host-split vs pure-fp CPU), per head */
    double worst=0; int rc=0;
    for(int h=0;h<NH && !rc;h++){
        rc=head_npu(c,Q8[h],Kf[h],Vf[h],L,HD,scale,att,KTp,Vp,w8,scores,attv,sc,qsv[h]);
        if(rc){ printf("  head_npu rc=%d FAIL\n",rc); return 1; }
        double mx=-1e300; for(int j=0;j<L;j++){ double s=0; for(int e=0;e<HD;e++)s+=Qf[h][e]*Kf[h][(size_t)j*HD+e]; sc[j]=s*scale; if(sc[j]>mx)mx=sc[j]; }
        double Z=0; for(int j=0;j<L;j++){ sc[j]=exp(sc[j]-mx); Z+=sc[j]; }
        for(int e=0;e<HD;e++){ double a=0; for(int j=0;j<L;j++)a+=sc[j]*Vf[h][(size_t)j*HD+e]; catt[e]=(float)(a/Z); }
        double refmax=0,er=0; for(int e=0;e<HD;e++){ if(fabsf(catt[e])>refmax)refmax=fabsf(catt[e]); double d=fabs(att[e]-catt[e]); if(d>er)er=d; }
        double rel=er/(refmax>1e-6?refmax:1); if(rel>worst)worst=rel;
    }
    printf("  coherence NPU(host-split) vs CPU: worst rel-err=%.4f %s\n", worst, worst<0.08?"COHERENT":"CHECK");
    /* timed: NPU host-split (all heads, incl per-step pack) */
    double t0=now_ms();
    for(int r=0;r<REP;r++) for(int h=0;h<NH;h++) head_npu(c,Q8[h],Kf[h],Vf[h],L,HD,scale,att,KTp,Vp,w8,scores,attv,sc,qsv[h]);
    double npu_ms=(now_ms()-t0)/REP;
    /* timed: CPU fp softmax attention */
    t0=now_ms();
    for(int r=0;r<REP;r++) for(int h=0;h<NH;h++){ double mx=-1e300;
        for(int j=0;j<L;j++){ double s=0; for(int e=0;e<HD;e++)s+=Qf[h][e]*Kf[h][(size_t)j*HD+e]; sc[j]=s*scale; if(sc[j]>mx)mx=sc[j]; }
        double Z=0; for(int j=0;j<L;j++){ sc[j]=exp(sc[j]-mx); Z+=sc[j]; }
        for(int e=0;e<HD;e++){ double a=0; for(int j=0;j<L;j++)a+=sc[j]*Vf[h][(size_t)j*HD+e]; catt[e]=(float)(a/Z); } }
    double cpu_ms=(now_ms()-t0)/REP;
    printf("  CPU fp attention   : %8.3f ms/token  (%d heads)\n", cpu_ms, NH);
    printf("  NPU host-split     : %8.3f ms/token  (incl per-step K^T/V pack)\n", npu_ms);
    printf("  speedup: %.2fx  %s\n", cpu_ms/npu_ms, npu_ms<cpu_ms?"(NPU wins)":"(CPU wins)");
    printf("%s\n", worst<0.08?"PASS — host-split decode attention coherent for arbitrary scores":"FAIL — coherence");
    ork_npu_free(c);
    return worst<0.08?0:1;
}
