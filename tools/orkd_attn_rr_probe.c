/* orkd_attn_rr_probe — validate the ORKD_ATTN_RR route: N fused biased-attention chains fanned round-robin
 * across the daemon's NPU cores in ONE round-trip (orkd_attn_rr_i8 -> handle_attn_rr -> ork_mm_run_chains_rr_biased).
 * Each chain has DISTINCT Q/K/V on REAL (mixed-sign, wide) scores; all share the requant + scalar-max-biased exp
 * LUT (global score max). Checks each chain's attn=av/Sigma vs its own int8 CPU ref -> COHERENT proves no
 * cross-core LUT/weight leak AND the daemon dispatch is correct. Pure client (mirrors orkd_attn_probe).
 *   make orkd orkd_attn_rr_probe && sudo env ORKD_BIN=$PWD/orkd ./orkd_attn_rr_probe [Nchains]
 */
#include "orkd_client.h"
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
    orkd_conn *c=orkd_connect(); if(!c){ fprintf(stderr,"connect/spawn FAILED\n"); return 1; }
    printf("connected: client_id=%u npu_cores=%u\n", orkd_client_id(c), orkd_soc_cores(c));
    printf("orkd_attn_rr_probe: %d biased attn chains RR across cores via orkd (Nq=%d Nk=%d dv=%d, real scores)\n",NC,Nq,Nk,dv);

    int8_t *V[8]; long *raw[8]; long maxabs=1;
    int8_t *Qall=malloc((size_t)NC*Nq*Kp);   /* chain-major padded Q */
    memset(Qall,0,(size_t)NC*Nq*Kp);
    int8_t *KTp[8];
    for(int n=0;n<NC;n++){
        int8_t *Q=malloc((size_t)Nq*d), *K=malloc((size_t)Nk*d); V[n]=malloc((size_t)Nk*dv);
        for(size_t i=0;i<(size_t)Nq*d;i++) Q[i]=(int8_t)rq();
        for(size_t i=0;i<(size_t)Nk*d;i++) K[i]=(int8_t)rq();
        for(size_t i=0;i<(size_t)Nk*dv;i++) V[n][i]=(int8_t)vv();
        for(int i=0;i<Nq;i++)for(int k=0;k<d;k++) Qall[((size_t)n*Nq+i)*Kp+k]=Q[(size_t)i*d+k];
        KTp[n]=calloc((size_t)Kp*Nk,1);
        for(int k=0;k<d;k++)for(int j=0;j<Nk;j++) KTp[n][(size_t)k*Nk+j]=K[(size_t)j*d+k];
        raw[n]=malloc((size_t)Nq*Nk*sizeof(long));
        for(int i=0;i<Nq;i++)for(int j=0;j<Nk;j++){ long a=0; for(int k=0;k<d;k++) a+=Q[(size_t)i*d+k]*K[(size_t)j*d+k];
            raw[n][(size_t)i*Nk+j]=a; if(labs(a)>maxabs)maxabs=labs(a); }
        free(Q); free(K);
    }
    int r_shift=16; int r_mult=(int)(((long)110<<r_shift)/maxabs); if(r_mult<1)r_mult=1;
    double in_scale=0.03125, out_scale=1.0/127.0;
    long smax=-128;
    for(int n=0;n<NC;n++) for(size_t i=0;i<(size_t)Nq*Nk;i++){ long s=(raw[n][i]*r_mult)>>r_shift; if(s>127)s=127; if(s<-128)s=-128; if(s>smax)smax=s; }
    double max_bias=(double)smax;
    printf("  shared calib: maxabs_raw=%ld r_mult=%d -> global score_max=%ld (=max_bias), in_scale=%.5f\n", maxabs, r_mult, smax, in_scale);

    /* per-chain CPU ref (biased) */
    double *cS[8], *cav[8];
    for(int n=0;n<NC;n++){
        int8_t *ce=malloc((size_t)Nq*Nk); cS[n]=malloc((size_t)Nq*8); cav[n]=malloc((size_t)Nq*dv*sizeof(double));
        for(int i=0;i<Nq;i++){ double S=0; for(int j=0;j<Nk;j++){ long a=raw[n][(size_t)i*Nk+j];
            long s=(a*r_mult)>>r_shift; if(s>127)s=127; if(s<-128)s=-128;
            double e=exp(((double)s-max_bias)*in_scale)/out_scale; if(e>127)e=127; int ei=(int)lround(e); ce[(size_t)i*Nk+j]=(int8_t)ei; S+=ei; }
          cS[n][i]=S; for(int x=0;x<dv;x++){ double av=0; for(int j=0;j<Nk;j++) av+=(double)ce[(size_t)i*Nk+j]*V[n][(size_t)j*dv+x]; cav[n][(size_t)i*dv+x]=av; } }
        free(ce);
    }
    /* pack per-chain wkt/wv DAEMON-RESIDENT + one shared wones */
    int8_t *ones=malloc((size_t)Nk*32); memset(ones,1,(size_t)Nk*32);
    uint64_t wones=orkd_pack_i8(c,Nk,32,ones); free(ones);
    uint64_t wkt_ids[8], wones_ids[8], wv_ids[8];
    for(int n=0;n<NC;n++){ wkt_ids[n]=orkd_pack_i8(c,Kp,Nk,KTp[n]); wv_ids[n]=orkd_pack_i8(c,Nk,dv,V[n]); wones_ids[n]=wones;
        if(!wkt_ids[n]||!wv_ids[n]||!wones){ printf("pack FAILED chain %d\n",n); orkd_disconnect(c); return 1; } }
    printf("packed resident: %d chains (shared wones=%llu)\n", NC, (unsigned long long)wones);

    int32_t *ssall=calloc((size_t)NC*Nq*32,4), *avall=calloc((size_t)NC*Nq*dv,4);
    int rc=orkd_attn_rr_i8(c, NC, wkt_ids, wones_ids, wv_ids, Nq, Nk, Kp, dv, r_mult, r_shift,
                           in_scale, out_scale, max_bias, Qall, ssall, avall);
    printf("  orkd_attn_rr_i8(%d chains) rc=%d\n", NC, rc);
    if(rc){ printf("FAIL rc=%d\n",rc); orkd_disconnect(c); return 1; }

    int fail=0;
    for(int n=0;n<NC;n++){
        int32_t *ss=ssall+(size_t)n*Nq*32, *avb=avall+(size_t)n*Nq*dv;
        int bad=0; double me=0, sae=0;
        for(int i=0;i<Nq;i++){ double Sn=(double)ss[(size_t)i*32]; if(Sn<=0)Sn=1; double Sc=cS[n][i]>0?cS[n][i]:1;
            for(int x=0;x<dv;x++){ double an=(double)avb[(size_t)i*dv+x]/Sn, ac=cav[n][(size_t)i*dv+x]/Sc;
                double e=fabs(an-ac); sae+=e; if(e>me)me=e; if(e>0.05&&fabs(ac)>1e-3&&e/fabs(ac)>0.05)bad++; } }
        printf("  chain %d: attn max|err|=%.4f mae=%.4f %s (%d/%d)\n", n, me, sae/(Nq*dv), bad?"CHECK":"COHERENT", bad, Nq*dv);
        if(bad)fail=1;
    }
    printf("%s\n", fail?"FAIL — a chain diverged":"PASS — N biased attn chains COHERENT fanned across cores via ORKD_ATTN_RR (one round-trip)");
    for(int n=0;n<NC;n++) orkd_free_weight(c,wkt_ids[n]), orkd_free_weight(c,wv_ids[n]);
    orkd_free_weight(c,wones); orkd_disconnect(c);
    return fail;
}
