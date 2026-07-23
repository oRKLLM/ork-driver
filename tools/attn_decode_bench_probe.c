/* attn_decode_bench_probe — DECODE attention, HOST-SPLIT NPU (Tier 12e) vs CPU, ARBITRARY scores, and the
 * Tier 12f RESIDENT-K/V perf model. One decode step: NH heads, single query row (Nq=1) attending to L cached
 * keys. Per head the two heavy matmuls (QK^T, e.V) run on the NPU; the [1,L] softmax runs on the HOST in fp
 * with a REAL per-head max-subtraction — correct for arbitrary (signed) scores (mirrors decode.c:attn_decode_npu).
 *
 * Two NPU timings, to isolate the 12f blocker:
 *   - "per-step pack": packs K^T/V EVERY token (what attn_decode_npu does today) — packing-bound, perf-negative.
 *   - "resident K/V":  packs K^T/V ONCE, per-step does only the matmuls (models Tier 12f: pack-once + append the
 *                      one new key/value each step, an O(HD) tile-write that's ~free vs the matmuls).
 * The gap between them IS the packing overhead 12f removes.
 *   sudo env ORK_MM_TIMEOUT=3000 ./attn_decode_bench_probe [NH] [L] [HD] [reps]
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

int main(int argc,char**argv){
    int NH=argc>1?atoi(argv[1]):32, L=argc>2?atoi(argv[2]):512, HD=argc>3?atoi(argv[3]):128, REP=argc>4?atoi(argv[4]):20;
    int Kp=512;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("attn_decode_bench_probe: NH=%d L=%d HD=%d — host-split NPU decode attn (arbitrary scores)\n",NH,L,HD);
    float scale=1.0f/sqrtf((float)HD);
    /* per-head ARBITRARY (signed) Q/K/V fp + their int8 quant */
    float **Qf=calloc(NH,sizeof*Qf),**Kf=calloc(NH,sizeof*Kf),**Vf=calloc(NH,sizeof*Vf);
    int8_t **Q8=calloc(NH,sizeof*Q8); float *qsv=calloc(NH,sizeof(float)),*ksv=calloc(NH,sizeof(float)),*vsv=calloc(NH,sizeof(float));
    ork_w **wkt=calloc(NH,sizeof*wkt),**wv=calloc(NH,sizeof*wv);
    uint32_t g=0x77;
    for(int h=0;h<NH;h++){ Qf[h]=malloc((size_t)HD*4); Kf[h]=malloc((size_t)L*HD*4); Vf[h]=malloc((size_t)L*HD*4);
        for(int e=0;e<HD;e++){ g=g*1664525u+1013904223u; Qf[h][e]=((int)((g>>26)%17)-8)*0.1f; }
        for(size_t i=0;i<(size_t)L*HD;i++){ g=g*1664525u+1013904223u; Kf[h][i]=((int)((g>>26)%17)-8)*0.1f; }
        for(size_t i=0;i<(size_t)L*HD;i++){ g=g*1664525u+1013904223u; Vf[h][i]=((int)((g>>26)%17)-8)*0.1f; }
        float qmax=1e-6f,kmax=1e-6f,vmax=1e-6f;
        for(int e=0;e<HD;e++){ float a=fabsf(Qf[h][e]); if(a>qmax)qmax=a; }
        for(size_t i=0;i<(size_t)L*HD;i++){ float ka=fabsf(Kf[h][i]),va=fabsf(Vf[h][i]); if(ka>kmax)kmax=ka; if(va>vmax)vmax=va; }
        qsv[h]=127.0f/qmax; ksv[h]=127.0f/kmax; vsv[h]=127.0f/vmax;
        Q8[h]=malloc((size_t)Kp); memset(Q8[h],0,Kp); for(int e=0;e<HD;e++) Q8[h][e]=(int8_t)lrintf(Qf[h][e]*qsv[h]);
        /* RESIDENT pack (once): K^T[Kp,L] and V[L,HD] int8 */
        int8_t *KTp=calloc((size_t)Kp*L,1),*Vp=malloc((size_t)L*HD);
        for(int e=0;e<HD;e++)for(int j=0;j<L;j++) KTp[(size_t)e*L+j]=(int8_t)lrintf(Kf[h][(size_t)j*HD+e]*ksv[h]);
        for(size_t i=0;i<(size_t)L*HD;i++) Vp[i]=(int8_t)lrintf(Vf[h][i]*vsv[h]);
        wkt[h]=ork_mm_pack_i8(c,Kp,L,KTp); wv[h]=ork_mm_pack_i8(c,L,HD,Vp); free(KTp); free(Vp);
        if(!wkt[h]||!wv[h]){ printf("pack fail h=%d\n",h); return 2; }
    }
    int8_t *w8=malloc((size_t)L); int32_t *scores=malloc((size_t)L*4),*attv=malloc((size_t)HD*4);
    double *sc=malloc((size_t)L*sizeof(double)); float *att=malloc((size_t)HD*4),*catt=malloc((size_t)HD*4);

    /* one head's attention using the RESIDENT packed weights (softmax on host) */
    #define RUN_HEAD(h) do{ \
        ork_mm_task_i8 t1={ wkt[h],1,Q8[h],scores }; if(ork_mm_run_chain_i8(c,1,&t1)) return 1; \
        double mx=-1e300; for(int j=0;j<L;j++){ sc[j]=(double)scores[j]/((double)qsv[h]*ksv[h])*scale; if(sc[j]>mx)mx=sc[j]; } \
        double Z=0; for(int j=0;j<L;j++){ sc[j]=exp(sc[j]-mx); Z+=sc[j]; } if(Z<=0)Z=1; \
        double wmax=0; for(int j=0;j<L;j++){ sc[j]/=Z; if(sc[j]>wmax)wmax=sc[j]; } double ws=127.0/(wmax>1e-9?wmax:1.0); \
        for(int j=0;j<L;j++){ int wi=(int)lrint(sc[j]*ws); w8[j]=(int8_t)(wi>127?127:(wi<0?0:wi)); } \
        ork_mm_task_i8 t2={ wv[h],1,w8,attv }; if(ork_mm_run_chain_i8(c,1,&t2)) return 1; \
        for(int e=0;e<HD;e++) att[e]=(float)((double)attv[e]/(ws*vsv[h])); }while(0)

    /* coherence (resident path) vs pure-fp CPU */
    double worst=0;
    for(int h=0;h<NH;h++){ RUN_HEAD(h);
        double mx=-1e300; for(int j=0;j<L;j++){ double s=0; for(int e=0;e<HD;e++)s+=Qf[h][e]*Kf[h][(size_t)j*HD+e]; sc[j]=s*scale; if(sc[j]>mx)mx=sc[j]; }
        double Z=0; for(int j=0;j<L;j++){ sc[j]=exp(sc[j]-mx); Z+=sc[j]; }
        for(int e=0;e<HD;e++){ double a=0; for(int j=0;j<L;j++)a+=sc[j]*Vf[h][(size_t)j*HD+e]; catt[e]=(float)(a/Z); }
        double rm=0,er=0; for(int e=0;e<HD;e++){ if(fabsf(catt[e])>rm)rm=fabsf(catt[e]); double d=fabs(att[e]-catt[e]); if(d>er)er=d; }
        double rel=er/(rm>1e-6?rm:1); if(rel>worst)worst=rel; }
    printf("  coherence NPU(host-split) vs CPU: worst rel-err=%.4f %s\n", worst, worst<0.08?"COHERENT":"CHECK");

    /* timed: RESIDENT K/V (pack-once; per-step = matmuls only — the Tier 12f target) */
    double t0=now_ms();
    for(int r=0;r<REP;r++) for(int h=0;h<NH;h++) RUN_HEAD(h);
    double res_ms=(now_ms()-t0)/REP;
    /* timed: CPU fp softmax attention */
    t0=now_ms();
    for(int r=0;r<REP;r++) for(int h=0;h<NH;h++){ double mx=-1e300;
        for(int j=0;j<L;j++){ double s=0; for(int e=0;e<HD;e++)s+=Qf[h][e]*Kf[h][(size_t)j*HD+e]; sc[j]=s*scale; if(sc[j]>mx)mx=sc[j]; }
        double Z=0; for(int j=0;j<L;j++){ sc[j]=exp(sc[j]-mx); Z+=sc[j]; }
        for(int e=0;e<HD;e++){ double a=0; for(int j=0;j<L;j++)a+=sc[j]*Vf[h][(size_t)j*HD+e]; catt[e]=(float)(a/Z); } }
    double cpu_ms=(now_ms()-t0)/REP;

    printf("  CPU fp attention        : %8.3f ms/token  (%d heads)\n", cpu_ms, NH);
    printf("  NPU resident-K/V (12f)  : %8.3f ms/token  (pack-once; matmuls only)\n", res_ms);
    printf("  speedup: %.2fx  %s\n", cpu_ms/res_ms, res_ms<cpu_ms?"(NPU wins)":"(CPU wins)");
    printf("%s\n", worst<0.08?"PASS — host-split decode attn coherent (arbitrary scores); resident-K/V perf model":"FAIL");
    for(int h=0;h<NH;h++){ ork_w_free(wkt[h]); ork_w_free(wv[h]); }
    ork_npu_free(c);
    return worst<0.08?0:1;
}
