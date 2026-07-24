/* chainrr_biased_probe — validate ork_mm_run_chains_rr_biased: N independent fused attention chains
 * [QK^T->exp((s-bias))->reduce,e.V] fanned CONCURRENTLY across the NPU cores from ONE dispatch, on REAL
 * (mixed-sign, wide) scores via a shared scalar-max-biased exp LUT. Each chain has its OWN Q/K/V data +
 * its OWN scratch buffers, so a stale-per-core-LUT or a wrong-core-weight leak would break THAT chain's
 * coherence (not just core 0's). The N chains share ONE (in_scale,out_scale,max_bias) — as a real layer's
 * kv-heads do — so max_bias is the GLOBAL score max across all chains (>= every score => args<=0).
 * Confirms the "LUT-cache-update op injected per core, not just once" path for the biased RR dispatch.
 *   make chainrr_biased_probe && sudo env ORK_MM_TIMEOUT=3000 ./chainrr_biased_probe [Nchains]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static uint32_t g=0x9e10u;
static int rq(void){ g=g*1664525u+1013904223u; return (int)((g>>26)%9)-4; }  /* [-4,4] */
static int vv(void){ g=g*1664525u+1013904223u; return (int)((g>>27)%5)-2; }   /* [-2,2] */

int main(int argc,char**argv){
    int NC=argc>1?atoi(argv[1]):3, Nq=32, d=128, Nk=512, Kp=512, dv=128;
    if(NC<1)NC=1; if(NC>8)NC=8;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("chainrr_biased_probe: %d biased attn chains RR across cores (Nq=%d Nk=%d dv=%d, real scores)\n",NC,Nq,Nk,dv);

    /* per-chain data + packed weights + scratch (scratch MUST be per-chain: chains run concurrently) */
    int8_t *Qp[8], *KTp[8], *V[8]; ork_w *w_kt[8], *w_v[8], *w_ones=NULL;
    int32_t *scb[8], *eb[8], *ss[8], *avb[8];
    long *raw[8]; long maxabs=1;
    for(int n=0;n<NC;n++){
        int8_t *Q=malloc((size_t)Nq*d), *K=malloc((size_t)Nk*d);
        V[n]=malloc((size_t)Nk*dv);
        for(size_t i=0;i<(size_t)Nq*d;i++) Q[i]=(int8_t)rq();
        for(size_t i=0;i<(size_t)Nk*d;i++) K[i]=(int8_t)rq();
        for(size_t i=0;i<(size_t)Nk*dv;i++) V[n][i]=(int8_t)vv();
        Qp[n]=calloc((size_t)Nq*Kp,1); KTp[n]=calloc((size_t)Kp*Nk,1);
        for(int i=0;i<Nq;i++)for(int k=0;k<d;k++) Qp[n][(size_t)i*Kp+k]=Q[(size_t)i*d+k];
        for(int k=0;k<d;k++)for(int j=0;j<Nk;j++) KTp[n][(size_t)k*Nk+j]=K[(size_t)j*d+k];
        raw[n]=malloc((size_t)Nq*Nk*sizeof(long));
        for(int i=0;i<Nq;i++)for(int j=0;j<Nk;j++){ long a=0; for(int k=0;k<d;k++) a+=Q[(size_t)i*d+k]*K[(size_t)j*d+k];
            raw[n][(size_t)i*Nk+j]=a; if(labs(a)>maxabs)maxabs=labs(a); }
        free(Q); free(K);
    }
    /* shared calib: r_mult targets int8 max ~110 over the GLOBAL raw range; max_bias = GLOBAL score max */
    int r_shift=16; int r_mult=(int)(((long)110<<r_shift)/maxabs); if(r_mult<1)r_mult=1;
    double in_scale=0.03125, out_scale=1.0/127.0;
    long smax=-128;
    for(int n=0;n<NC;n++) for(size_t i=0;i<(size_t)Nq*Nk;i++){ long s=(raw[n][i]*r_mult)>>r_shift; if(s>127)s=127; if(s<-128)s=-128; if(s>smax)smax=s; }
    double max_bias=(double)smax;
    printf("  shared calib: maxabs_raw=%ld r_mult=%d -> global score_max=%ld (=max_bias), in_scale=%.5f\n", maxabs, r_mult, smax, in_scale);

    /* per-chain CPU ref (biased) + pack weights */
    { int8_t *ones=malloc((size_t)Nk*32); memset(ones,1,(size_t)Nk*32); w_ones=ork_mm_pack_i8(c,Nk,32,ones); free(ones); }
    double *cS[8], *cav[8];
    for(int n=0;n<NC;n++){
        int8_t *ce=malloc((size_t)Nq*Nk); cS[n]=malloc((size_t)Nq*8); cav[n]=malloc((size_t)Nq*dv*sizeof(double));
        for(int i=0;i<Nq;i++){ double S=0; for(int j=0;j<Nk;j++){ long a=raw[n][(size_t)i*Nk+j];
            long s=(a*r_mult)>>r_shift; if(s>127)s=127; if(s<-128)s=-128;
            double e=exp(((double)s-max_bias)*in_scale)/out_scale; if(e>127)e=127; int ei=(int)lround(e); ce[(size_t)i*Nk+j]=(int8_t)ei; S+=ei; }
          cS[n][i]=S; for(int x=0;x<dv;x++){ double av=0; for(int j=0;j<Nk;j++) av+=(double)ce[(size_t)i*Nk+j]*V[n][(size_t)j*dv+x]; cav[n][(size_t)i*dv+x]=av; } }
        free(ce);
        w_kt[n]=ork_mm_pack_i8(c,Kp,Nk,KTp[n]); w_v[n]=ork_mm_pack_i8(c,Nk,dv,V[n]);
        if(!w_kt[n]||!w_v[n]||!w_ones){ printf("pack fail chain %d\n",n); return 2; }
        scb[n]=calloc((size_t)Nq*Nk,4); eb[n]=calloc((size_t)Nq*Nk,4); ss[n]=calloc((size_t)Nq*32,4); avb[n]=calloc((size_t)Nq*dv,4);
    }
    /* BRING UP ALL CORES (RR precondition): a multi-core matmul warms cores 0..nc-1 */
    { int32_t *wc=calloc((size_t)Nq*Nk,4); ork_mm_task_i8 wt={ w_kt[0], Nq, Qp[0], wc };
      int wrc=ork_mm_run_chain_i8(c,1,&wt); free(wc); printf("  [warm] multi-core matmul rc=%d\n", wrc); }

    /* build the N chains + shared op graph, then dispatch RR (concurrent across cores) */
    ork_mm_task_i8 tk[8][4]; const ork_mm_task_i8 *chains[8]; int S[8];
    for(int n=0;n<NC;n++){
        tk[n][0]=(ork_mm_task_i8){ w_kt[n], Nq, Qp[n],          scb[n] };
        tk[n][1]=(ork_mm_task_i8){ w_kt[n], Nq, (int8_t*)scb[n], eb[n] };
        tk[n][2]=(ork_mm_task_i8){ w_ones,  Nq, (int8_t*)eb[n],  ss[n] };
        tk[n][3]=(ork_mm_task_i8){ w_v[n],  Nq, (int8_t*)eb[n],  avb[n] };
        chains[n]=tk[n]; S[n]=4;
    }
    ork_chain_op ops[4] = { {1,-1,0,r_mult,r_shift}, {2,0,0,0,0}, {0,1,0,0,0}, {0,1,0,0,0} };
    int rc=ork_mm_run_chains_rr_biased(c, NC, chains, S, ops, in_scale, out_scale, max_bias);
    printf("  ork_mm_run_chains_rr_biased(%d chains) rc=%d\n", NC, rc);
    if(rc){ printf("FAIL rc=%d\n",rc); return 1; }

    int fail=0;
    for(int n=0;n<NC;n++){
        int bad=0; double me=0, sae=0;
        for(int i=0;i<Nq;i++){ double Sn=(double)ss[n][(size_t)i*32]; if(Sn<=0)Sn=1; double Sc=cS[n][i]>0?cS[n][i]:1;
            for(int x=0;x<dv;x++){ double an=(double)avb[n][(size_t)i*dv+x]/Sn, ac=cav[n][(size_t)i*dv+x]/Sc;
                double e=fabs(an-ac); sae+=e; if(e>me)me=e; if(e>0.05&&fabs(ac)>1e-3&&e/fabs(ac)>0.05)bad++; } }
        printf("  chain %d: attn max|err|=%.4f mae=%.4f %s (%d/%d)\n", n, me, sae/(Nq*dv), bad?"CHECK":"COHERENT", bad, Nq*dv);
        if(bad)fail=1;
    }
    printf("%s\n", fail?"FAIL — a chain diverged (per-core LUT/weight leak?)":"PASS — N biased attn chains COHERENT fanned across cores in one RR dispatch");
    for(int n=0;n<NC;n++){ ork_mm_free(c,w_kt[n]); ork_mm_free(c,w_v[n]); }
    ork_mm_free(c,w_ones); ork_npu_free(c);
    return fail;
}
