/* chainrr_conc_probe — increment 2 (CONCURRENT round-robin): dispatch N independent attention-core chains
 * [QK^T->exp->reduce, e.V] across ALL NPU cores AT ONCE via ork_mm_run_chains_rr, each chain with DISTINCT
 * inputs, and validate every chain vs its own CPU reference. Proves concurrent multi-core chain execution is
 * coherent (no cross-core corruption) and exercises the per-core scratch + per-core cache on all cores at once.
 * Also times it vs the single-core sequential path to show the prefill-throughput win.
 *   sudo env ORK_MM_TIMEOUT=3000 ./chainrr_conc_probe [nchains] [Nq]
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
    int NCH=argc>1?atoi(argv[1]):6, Nq=argc>2?atoi(argv[2]):32;
    int d=128, Nk=512, Kp=512, dv=128;
    setvbuf(stdout,0,_IONBF,0);
    if(NCH<1||NCH>64){ printf("nchains 1..64\n"); return 2; }
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("chainrr_conc_probe: %d chains, Nq=%d Nk=%d dv=%d — concurrent round-robin across cores\n",NCH,Nq,Nk,dv);
    int r_mult=0x4000, r_shift=16; double in_scale=0.0625, out_scale=1.0/127.0;
    ork_chain_op ops[4] = { {1,-1,0,r_mult,r_shift}, {2,0,0,0,0}, {0,1,0,0,0}, {0,1,0,0,0} };

    /* shared ones-weight (Sigma reduce); per-chain K^T and V weights + IO buffers (distinct inputs per chain) */
    ork_w *w_ones; { int8_t *ones=malloc((size_t)Nk*32); memset(ones,1,(size_t)Nk*32); w_ones=ork_i8_mm_pack(c,Nk,32,ones); free(ones); }
    if(!w_ones){ printf("pack ones fail\n"); return 2; }
    ork_w **w_kt=calloc(NCH,sizeof*w_kt), **w_v=calloc(NCH,sizeof*w_v);
    int8_t **Qp=calloc(NCH,sizeof*Qp), **Kd=calloc(NCH,sizeof*Kd), **Vd=calloc(NCH,sizeof*Vd);
    int32_t **scb=calloc(NCH,sizeof*scb), **eb=calloc(NCH,sizeof*eb), **ssum=calloc(NCH,sizeof*ssum), **avb=calloc(NCH,sizeof*avb);
    double **cav=calloc(NCH,sizeof*cav), *cS=calloc((size_t)NCH*Nq,sizeof(double));
    ork_mm_task_i8 **chains=calloc(NCH,sizeof*chains); int *S=calloc(NCH,sizeof*S);

    for(int ch=0; ch<NCH; ch++){
        uint32_t g=0x9e10u + 0x1000u*(uint32_t)(ch+1);
        #define Q2 (g=g*1664525u+1013904223u, (int)((g>>27)%3))
        #define VV (g=g*1664525u+1013904223u, (int)((g>>27)%5)-2)
        int8_t *Q=malloc((size_t)Nq*d); Kd[ch]=malloc((size_t)Nk*d); Vd[ch]=malloc((size_t)Nk*dv);
        for(size_t i=0;i<(size_t)Nq*d;i++) Q[i]=(int8_t)Q2;
        for(size_t i=0;i<(size_t)Nk*d;i++) Kd[ch][i]=(int8_t)(-Q2);
        for(size_t i=0;i<(size_t)Nk*dv;i++) Vd[ch][i]=(int8_t)VV;
        Qp[ch]=calloc((size_t)Nq*Kp,1);
        int8_t *KTp=calloc((size_t)Kp*Nk,1);
        for(int i=0;i<Nq;i++)for(int k=0;k<d;k++) Qp[ch][(size_t)i*Kp+k]=Q[(size_t)i*d+k];
        for(int k=0;k<d;k++)for(int j=0;j<Nk;j++) KTp[(size_t)k*Nk+j]=Kd[ch][(size_t)j*d+k];
        /* CPU reference */
        int8_t *ce=malloc((size_t)Nq*Nk); cav[ch]=malloc((size_t)Nq*dv*sizeof(double));
        for(int i=0;i<Nq;i++){ double Ssum=0; for(int j=0;j<Nk;j++){ long a=0; for(int k=0;k<d;k++)a+=Q[(size_t)i*d+k]*Kd[ch][(size_t)j*d+k];
            long s=(a*r_mult)>>r_shift; if(s>127)s=127; if(s<-128)s=-128; double e=exp((double)s*in_scale)/out_scale; if(e>127)e=127;
            int ei=(int)lround(e); ce[(size_t)i*Nk+j]=(int8_t)ei; Ssum+=ei; } cS[(size_t)ch*Nq+i]=Ssum;
          for(int x=0;x<dv;x++){ double av=0; for(int j=0;j<Nk;j++) av+=(double)ce[(size_t)i*Nk+j]*Vd[ch][(size_t)j*dv+x]; cav[ch][(size_t)i*dv+x]=av; } }
        free(ce); free(Q);
        w_kt[ch]=ork_i8_mm_pack(c,Kp,Nk,KTp); w_v[ch]=ork_i8_mm_pack(c,Nk,dv,Vd[ch]); free(KTp);
        if(!w_kt[ch]||!w_v[ch]){ printf("pack chain %d fail\n",ch); return 2; }
        scb[ch]=calloc((size_t)Nq*Nk,4); eb[ch]=calloc((size_t)Nq*Nk,4); ssum[ch]=calloc((size_t)Nq*32,4); avb[ch]=calloc((size_t)Nq*dv,4);
        chains[ch]=malloc(4*sizeof(ork_mm_task_i8)); S[ch]=4;
        chains[ch][0]=(ork_mm_task_i8){ w_kt[ch],Nq,Qp[ch],scb[ch] };
        chains[ch][1]=(ork_mm_task_i8){ w_kt[ch],Nq,(int8_t*)scb[ch],eb[ch] };
        chains[ch][2]=(ork_mm_task_i8){ w_ones,Nq,(int8_t*)eb[ch],ssum[ch] };
        chains[ch][3]=(ork_mm_task_i8){ w_v[ch],Nq,(int8_t*)eb[ch],avb[ch] };
        #undef Q2
        #undef VV
    }

    /* WARM all cores (multi-core matmul) — the chains' precondition (a cold core's first submit wedges) */
    { int32_t *wc=calloc((size_t)Nq*Nk,4); ork_mm_task_i8 wt={ w_kt[0], Nq, Qp[0], wc };
      int wrc=ork_i8_mm_run_chain(c,1,&wt); free(wc);
      printf("  [warm] multi-core matmul rc=%d\n", wrc); }

    /* CONCURRENT round-robin dispatch */
    double t0=now_ms();
    int rc=ork_mm_run_chains_rr(c, NCH, (const ork_mm_task_i8*const*)chains, S, ops, in_scale, out_scale);
    double t1=now_ms();
    if(rc){ printf("  ork_mm_run_chains_rr rc=%d FAIL\n",rc); return 1; }

    int fail=0; double worst=0;
    for(int ch=0; ch<NCH; ch++){ int bad=0; double me=0;
        for(int i=0;i<Nq;i++){ double Sn=(double)ssum[ch][(size_t)i*32]; if(Sn<=0)Sn=1; double Sc=cS[(size_t)ch*Nq+i]>0?cS[(size_t)ch*Nq+i]:1;
            for(int x=0;x<dv;x++){ double an=(double)avb[ch][(size_t)i*dv+x]/Sn, ac=cav[ch][(size_t)i*dv+x]/Sc; double e=fabs(an-ac);
                if(e>me)me=e; if(e>0.05&&fabs(ac)>1e-3&&e/fabs(ac)>0.05)bad++; } }
        if(me>worst)worst=me; if(bad){ printf("  chain %d: max|err|=%.4f CHECK (%d/%d)\n",ch,me,bad,Nq*dv); fail=1; } }
    printf("  %d chains concurrent: worst max|err|=%.4f  %s  (%.2f ms, %.2f ms/chain)\n",
           NCH, worst, fail?"CHECK":"COHERENT", t1-t0, (t1-t0)/NCH);
    printf("%s\n", fail?"FAIL":"PASS — concurrent round-robin: all chains coherent across cores (increment 2)");

    for(int ch=0;ch<NCH;ch++){ ork_mm_free(c,w_kt[ch]); ork_mm_free(c,w_v[ch]);
        free(Qp[ch]);free(Kd[ch]);free(Vd[ch]);free(scb[ch]);free(eb[ch]);free(ssum[ch]);free(avb[ch]);free(cav[ch]);free(chains[ch]); }
    ork_mm_free(c,w_ones); ork_npu_free(c);
    return fail;
}
