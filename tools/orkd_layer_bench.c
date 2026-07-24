/* orkd_layer_bench — the DECISIVE decode-perf measurement for the heterogeneous "spine" (brick 2).
 * Times ORKD_LAYER (daemon runs a WHOLE qwen3 decode layer in ONE round-trip: CPU glue + NPU doorbell
 * matmuls, resident weights) at REAL qwen3-1.7B decode dims, steady-state (doorbell warmed once). The
 * derived per-token time = N_LAYERS * t_layer answers: does one-round-trip-per-layer beat the 7.38 t/s
 * CPU-attention decode baseline? A negative result here means the ggml matcher can't win and is moot.
 *   make orkd orkd_layer_bench && sudo env ORKD_BIN=$PWD/orkd ORK_MM_TIMEOUT=3000 ./orkd_layer_bench [nkv] [iters]
 */
#include "orkd_client.h"
#include "orkd_proto.h"
#include "spine_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
static uint32_t rng=0xB0BA1u; static float frnd(void){ rng=rng*1664525u+1013904223u; return (int)(rng>>9)/4194304.0f-1.0f; }
static int i8rnd(void){ rng=rng*1664525u+1013904223u; return (int)((rng>>27)%9)-4; }
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }

int main(int argc, char**argv){
    /* qwen3-1.7B: D=2048 H=16 Hkv=8 dk=dv=128 Nff=6144 n_layers=28 */
    const int D=2048,H=16,Hkv=8,dk=128,dv=128,N_LAYERS=28,rk2=H/Hkv,Nq=H*dk,Nkv=Hkv*dk;
    int nkv = argc>1?atoi(argv[1]):512;
    int iters = argc>2?atoi(argv[2]):60;
    int Nff = argc>3?atoi(argv[3]):6144;   /* real qwen3-1.7B=6144 (down K=6144 -> run_i8); 4096 fits the doorbell for all 7 matmuls */
    setvbuf(stdout,0,_IONBF,0);
    orkd_conn *c=orkd_connect(); if(!c){ fprintf(stderr,"connect FAILED\n"); return 1; }
    printf("orkd_layer_bench: qwen3-1.7B decode dims D=%d H=%d Hkv=%d Nff=%d nkv=%d n_layers=%d iters=%d\n",
           D,H,Hkv,Nff,nkv,N_LAYERS,iters);

    int8_t *Wq=malloc((size_t)D*Nq),*Wk=malloc((size_t)D*Nkv),*Wv=malloc((size_t)D*Nkv),*Wo=malloc((size_t)Nq*D);
    int8_t *Wg=malloc((size_t)D*Nff),*Wu=malloc((size_t)D*Nff),*Wd=malloc((size_t)Nff*D);
    for(size_t i=0;i<(size_t)D*Nq;i++)Wq[i]=(int8_t)i8rnd(); for(size_t i=0;i<(size_t)D*Nkv;i++){Wk[i]=(int8_t)i8rnd();Wv[i]=(int8_t)i8rnd();}
    for(size_t i=0;i<(size_t)Nq*D;i++)Wo[i]=(int8_t)i8rnd(); for(size_t i=0;i<(size_t)D*Nff;i++){Wg[i]=(int8_t)i8rnd();Wu[i]=(int8_t)i8rnd();} for(size_t i=0;i<(size_t)Nff*D;i++)Wd[i]=(int8_t)i8rnd();
    float *attn_norm=malloc(D*4),*q_norm=malloc(dk*4),*ffn_norm=malloc(D*4),*x=malloc(D*4);
    for(int i=0;i<D;i++){attn_norm[i]=1+0.1f*frnd();ffn_norm[i]=1+0.1f*frnd();x[i]=frnd()*3;} for(int i=0;i<dk;i++)q_norm[i]=1+0.1f*frnd();
    float *Kc=malloc((size_t)Hkv*nkv*dk*4),*Vc=malloc((size_t)Hkv*nkv*dv*4);
    for(size_t i=0;i<(size_t)Hkv*nkv*dk;i++)Kc[i]=frnd(); for(size_t i=0;i<(size_t)Hkv*nkv*dv;i++)Vc[i]=frnd();

    struct orkd_layer h; memset(&h,0,sizeof h);
    h.wq=orkd_pack_i8(c,D,Nq,Wq); h.wk=orkd_pack_i8(c,D,Nkv,Wk); h.wv=orkd_pack_i8(c,D,Nkv,Wv); h.wo=orkd_pack_i8(c,Nq,D,Wo);
    h.wg=orkd_pack_i8(c,D,Nff,Wg); h.wu=orkd_pack_i8(c,D,Nff,Wu); h.wd=orkd_pack_i8(c,Nff,D,Wd);
    if(!h.wq||!h.wk||!h.wv||!h.wo||!h.wg||!h.wu||!h.wd){ printf("pack FAILED\n"); orkd_disconnect(c); return 1; }
    h.D=D;h.H=H;h.Hkv=Hkv;h.dk=dk;h.dv=dv;h.Nff=Nff;h.nkv=nkv;h.pos=nkv; h.attn_scale=1.0/sqrt((double)dk); h.rope_base=1e6;
    float *xo=malloc(D*4);

    /* warm-up (first call warms the doorbell for all 7 shapes; discard). Time each to expose a pathology. */
    int rc=0; for(int w=0;w<3;w++){ double a=now_ms(); rc=orkd_layer_i8(c,&h,attn_norm,q_norm,ffn_norm,x,Kc,Vc,xo); double b=now_ms();
        printf("  warm[%d] %.1f ms rc=%d\n",w,b-a,rc); if(rc){orkd_disconnect(c); return 1;} }

    double t0=now_ms();
    for(int it=0;it<iters;it++){ double a=now_ms(); rc=orkd_layer_i8(c,&h,attn_norm,q_norm,ffn_norm,x,Kc,Vc,xo); double b=now_ms();
        if(it<3) printf("  iter[%d] %.1f ms\n",it,b-a); if(rc){printf("iter %d rc=%d\n",it,rc); orkd_disconnect(c); return 1;} }
    double t1=now_ms();
    double per_layer_ms=(t1-t0)/iters;
    double per_tok_ms=per_layer_ms*N_LAYERS;
    double toks=1000.0/per_tok_ms;
    printf("  steady-state: %.3f ms/layer (one ORKD_LAYER round-trip)\n", per_layer_ms);
    printf("  => per-token (x%d layers): %.1f ms  =>  %.2f tok/s  (spine decode estimate)\n", N_LAYERS, per_tok_ms, toks);
    printf("  CPU-attention decode baseline = 7.38 tok/s  ->  spine is %s (%.2fx)\n",
           toks>7.38?"FASTER":"SLOWER", toks/7.38);
    orkd_disconnect(c);
    return 0;
}
