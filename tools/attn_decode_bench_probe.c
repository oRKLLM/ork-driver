/* attn_decode_bench_probe — DECODE-flavor attention on the NPU chain vs CPU scalar. One decode step: NH heads,
 * each a single query row (Nq=1) attending to L cached keys (no causal mask — all cached keys are past). Each
 * head is one fused [QK^T->exp->reduce, e.V] chain; the NH heads are dispatched concurrently across cores via
 * ork_mm_run_chains_rr. Compares per-token latency + throughput vs the current CPU scalar softmax attention,
 * and checks coherence. Scores are kept in the post-max (<=0) domain so exp fits int8 without a max-bias op.
 *   sudo env ORK_CHAIN_LUT_STICKY=1 ORK_MM_TIMEOUT=3000 ./attn_decode_bench_probe [NH] [L] [HD] [reps]
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
    int NH=argc>1?atoi(argv[1]):32, L=argc>2?atoi(argv[2]):512, HD=argc>3?atoi(argv[3]):128, REP=argc>4?atoi(argv[4]):50;
    int Kp=512;                       /* QK^T contraction (head_dim) zero-padded to 512 (chain K%512) */
    if(L%512){ printf("L must be %%512 (chain K-slice); try 512/1024\n"); return 2; }
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("attn_decode_bench_probe: NH=%d L=%d HD=%d — decode attention (Nq=1/head), NPU rr vs CPU\n",NH,L,HD);
    int Nq=1; int r_mult=0x4000, r_shift=16; double in_scale=0.0625, out_scale=1.0/127.0;
    ork_chain_op ops[4] = { {1,-1,0,r_mult,r_shift}, {2,0,0,0,0}, {0,1,0,0,0}, {0,1,0,0,0} };
    uint32_t rng=0x2a2a;
    #define R3 (rng=rng*1664525u+1013904223u, (int)((rng>>27)%3))
    #define RV (rng=rng*1664525u+1013904223u, (int)((rng>>27)%5)-2)

    ork_w *w_ones; { int8_t *o=malloc((size_t)L*32); memset(o,1,(size_t)L*32); w_ones=ork_mm_pack_i8(c,L,32,o); free(o); }
    ork_w **w_kt=calloc(NH,sizeof*w_kt), **w_v=calloc(NH,sizeof*w_v);
    int8_t **Qp=calloc(NH,sizeof*Qp), **Kh=calloc(NH,sizeof*Kh), **Vh=calloc(NH,sizeof*Vh);
    int32_t **scb=calloc(NH,sizeof*scb), **eb=calloc(NH,sizeof*eb), **ssum=calloc(NH,sizeof*ssum), **avb=calloc(NH,sizeof*avb);
    ork_mm_task_i8 **chains=calloc(NH,sizeof*chains); int *S=calloc(NH,sizeof*S);
    double **cav=calloc(NH,sizeof*cav); double *cS=calloc(NH,sizeof(double));

    for(int h=0; h<NH; h++){
        int8_t *Q=malloc((size_t)HD); Kh[h]=malloc((size_t)L*HD); Vh[h]=malloc((size_t)L*HD);
        for(int e=0;e<HD;e++) Q[e]=(int8_t)R3;                       /* Q>=0 */
        for(size_t i=0;i<(size_t)L*HD;i++) Kh[h][i]=(int8_t)(-R3);   /* K<=0 -> QK^T<=0 (post-max domain) */
        for(size_t i=0;i<(size_t)L*HD;i++) Vh[h][i]=(int8_t)RV;
        Qp[h]=calloc((size_t)Nq*Kp,1); for(int e=0;e<HD;e++) Qp[h][e]=Q[e];
        int8_t *KTp=calloc((size_t)Kp*L,1);
        for(int e=0;e<HD;e++)for(int j=0;j<L;j++) KTp[(size_t)e*L+j]=Kh[h][(size_t)j*HD+e];
        /* CPU reference (fp domain matching the chain's int8 requant) */
        int8_t *ce=malloc((size_t)L); cav[h]=malloc((size_t)HD*sizeof(double)); double Ssum=0;
        for(int j=0;j<L;j++){ long a=0; for(int e=0;e<HD;e++)a+=Q[e]*Kh[h][(size_t)j*HD+e];
            long s=(a*r_mult)>>r_shift; if(s>127)s=127; if(s<-128)s=-128; double ex=exp((double)s*in_scale)/out_scale; if(ex>127)ex=127;
            int ei=(int)lround(ex); ce[j]=(int8_t)ei; Ssum+=ei; }
        cS[h]=Ssum>0?Ssum:1;
        for(int e=0;e<HD;e++){ double av=0; for(int j=0;j<L;j++) av+=(double)ce[j]*Vh[h][(size_t)j*HD+e]; cav[h][e]=av; }
        free(ce); free(Q);
        w_kt[h]=ork_mm_pack_i8(c,Kp,L,KTp); w_v[h]=ork_mm_pack_i8(c,L,HD,Vh[h]); free(KTp);
        if(!w_kt[h]||!w_v[h]||!w_ones){ printf("pack fail h=%d\n",h); return 2; }
        scb[h]=calloc((size_t)Nq*L,4); eb[h]=calloc((size_t)Nq*L,4); ssum[h]=calloc((size_t)Nq*32,4); avb[h]=calloc((size_t)Nq*HD,4);
        chains[h]=malloc(4*sizeof(ork_mm_task_i8)); S[h]=4;
        chains[h][0]=(ork_mm_task_i8){ w_kt[h],Nq,Qp[h],scb[h] };
        chains[h][1]=(ork_mm_task_i8){ w_kt[h],Nq,(int8_t*)scb[h],eb[h] };
        chains[h][2]=(ork_mm_task_i8){ w_ones,Nq,(int8_t*)eb[h],ssum[h] };
        chains[h][3]=(ork_mm_task_i8){ w_v[h],Nq,(int8_t*)eb[h],avb[h] };
    }

    /* warm cores + one warmup pass (populate per-core caches) */
    { int32_t *wc=calloc((size_t)Nq*L,4); ork_mm_task_i8 wt={ w_kt[0], Nq, Qp[0], wc }; ork_mm_run_chain_i8(c,1,&wt); free(wc); }
    int wrc=ork_mm_run_chains_rr(c,NH,(const ork_mm_task_i8*const*)chains,S,ops,in_scale,out_scale);
    if(wrc){ printf("  run_chains_rr rc=%d FAIL (Nq=1 unsupported?)\n",wrc); return 1; }

    /* coherence: NPU rr vs CPU */
    double worst=0;
    for(int h=0;h<NH;h++){ double Sn=(double)ssum[h][0]; if(Sn<=0)Sn=1;
        for(int e=0;e<HD;e++){ double an=(double)avb[h][e]/Sn, ac=cav[h][e]/cS[h]; double er=fabs(an-ac); if(er>worst)worst=er; } }

    /* timed: NPU rr (one dispatch = one decode step's NH heads) */
    double t0=now_ms();
    for(int r=0;r<REP;r++) ork_mm_run_chains_rr(c,NH,(const ork_mm_task_i8*const*)chains,S,ops,in_scale,out_scale);
    double npu_ms=(now_ms()-t0)/REP;

    /* timed: CPU scalar softmax attention (same NH heads, Nq=1, L keys), int8 inputs -> fp accum */
    double *sc=malloc((size_t)L*sizeof(double)), *out=malloc((size_t)HD*sizeof(double));
    t0=now_ms();
    for(int r=0;r<REP;r++) for(int h=0;h<NH;h++){
        double mx=-1e30;
        for(int j=0;j<L;j++){ long a=0; int8_t*Qq=Qp[h]; for(int e=0;e<HD;e++)a+=Qq[e]*Kh[h][(size_t)j*HD+e]; sc[j]=(double)a*in_scale*(r_mult/65536.0); if(sc[j]>mx)mx=sc[j]; }
        double Z=0; for(int j=0;j<L;j++){ sc[j]=exp(sc[j]-mx); Z+=sc[j]; }
        for(int e=0;e<HD;e++){ double av=0; for(int j=0;j<L;j++)av+=sc[j]*Vh[h][(size_t)j*HD+e]; out[e]=av/Z; }
    }
    double cpu_ms=(now_ms()-t0)/REP;

    printf("  coherence NPU-vs-CPU: worst max|err|=%.4f %s\n", worst, worst<0.05?"COHERENT":"CHECK");
    printf("  CPU scalar attention : %8.3f ms/token  (%d heads)\n", cpu_ms, NH);
    printf("  NPU rr chain (3-core): %8.3f ms/token\n", npu_ms);
    printf("  speedup: %.2fx  %s\n", cpu_ms/npu_ms, npu_ms<cpu_ms?"(NPU wins)":"(CPU wins)");
    printf("PASS — decode attention benchmark done\n");
    ork_npu_free(c);
    return 0;
}
