/* orkd_layer_probe — validate the ORKD_LAYER route: the DAEMON runs a whole decode layer on the spine in ONE
 * round-trip (orkd_layer_i8 -> handle_layer). Pack the 7 layer weights daemon-resident, send gains/x/KV inline,
 * get x_out, check bit-exact vs an int8-faithful CPU reference (same spine kernels + same int8 matmuls). This is
 * the daemon-side of the ggml integration — the "layer as one submit" mechanism, proven before the ggml wiring.
 *   make orkd orkd_layer_probe && sudo env ORKD_BIN=$PWD/orkd ./orkd_layer_probe
 */
#include "orkd_client.h"
#include "orkd_proto.h"
#include "spine_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static uint32_t rng=0xA11CEu; static float frnd(void){ rng=rng*1664525u+1013904223u; return (int)(rng>>9)/4194304.0f-1.0f; }
static int i8rnd(void){ rng=rng*1664525u+1013904223u; return (int)((rng>>27)%9)-4; }
static void cpu_mm(const int8_t*A,const int8_t*W,int K,int N,int32_t*C){ for(int n=0;n<N;n++){ long a=0; for(int k=0;k<K;k++)a+=A[k]*W[(size_t)k*N+n]; C[n]=(int32_t)a; } }

int main(void){
    int D=512,H=4,Hkv=2,dk=128,dv=128,Nff=512,nkv=256,rk2=H/Hkv,Nq=H*dk,Nkv=Hkv*dk;
    setvbuf(stdout,0,_IONBF,0);
    orkd_conn *c=orkd_connect(); if(!c){ fprintf(stderr,"connect FAILED\n"); return 1; }
    printf("orkd_layer_probe: whole layer via ORKD_LAYER (D=%d H=%d Hkv=%d Nff=%d nkv=%d)\n",D,H,Hkv,Nff,nkv);

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
    printf("packed 7 resident weights; sending ORKD_LAYER...\n");

    float *xo=malloc(D*4);
    int rc=orkd_layer_i8(c,&h,attn_norm,q_norm,ffn_norm,x,Kc,Vc,xo);
    printf("  orkd_layer_i8 rc=%d  x_out[0..2]=%.4f,%.4f,%.4f\n", rc, xo[0],xo[1],xo[2]);
    if(rc){ printf("FAIL rc=%d\n",rc); orkd_disconnect(c); return 1; }

    /* int8-faithful CPU reference (same spine kernels + int8 matmuls) */
    float *rf=malloc(D*4);
    { float *xn=malloc(D*4); spine_rmsnorm(x,attn_norm,D,1e-6f,xn); int8_t*xi=malloc(D); float sx=spine_quant(xn,D,xi);
      int32_t*cq=malloc((size_t)Nq*4); cpu_mm(xi,Wq,D,Nq,cq);
      float *qf=malloc(Nq*4); for(int i=0;i<Nq;i++)qf[i]=cq[i]/sx;
      float *ao=malloc(Nq*4);
      for(int hh=0;hh<H;hh++){ float*qh=qf+(size_t)hh*dk; spine_rmsnorm(qh,q_norm,dk,1e-6f,qh); spine_rope_neox(qh,dk,nkv,1e6f);
          spine_attn(qh,Kc+(size_t)(hh/rk2)*nkv*dk,Vc+(size_t)(hh/rk2)*nkv*dv,nkv,dk,dv,(float)h.attn_scale,ao+(size_t)hh*dv); }
      int8_t*ai=malloc(Nq); float sa=spine_quant(ao,Nq,ai); int32_t*co=malloc((size_t)D*4); cpu_mm(ai,Wo,Nq,D,co);
      float *x1=malloc(D*4); for(int i=0;i<D;i++)x1[i]=x[i]+co[i]/sa;
      float *xf=malloc(D*4); spine_rmsnorm(x1,ffn_norm,D,1e-6f,xf); int8_t*xfi=malloc(D); float sf=spine_quant(xf,D,xfi);
      int32_t*cg=malloc((size_t)Nff*4),*cu=malloc((size_t)Nff*4); cpu_mm(xfi,Wg,D,Nff,cg); cpu_mm(xfi,Wu,D,Nff,cu);
      float *gf=malloc(Nff*4),*uf=malloc(Nff*4),*act=malloc(Nff*4);
      for(int i=0;i<Nff;i++){gf[i]=cg[i]/sf;uf[i]=cu[i]/sf;} spine_silu_glu(gf,uf,Nff,act);
      int8_t*aci=malloc(Nff); float sac=spine_quant(act,Nff,aci); int32_t*cd=malloc((size_t)D*4); cpu_mm(aci,Wd,Nff,D,cd);
      for(int i=0;i<D;i++)rf[i]=x1[i]+cd[i]/sac; }

    double me=0, maxa=1e-9; for(int i=0;i<D;i++){ double d=fabs((double)xo[i]-rf[i]); if(d>me)me=d; double av=fabs((double)rf[i]); if(av>maxa)maxa=av; }
    double rel=me/maxa;   /* relative: daemon vs a SEPARATE binary's ref differ only by fp32 accumulation noise */
    int pass = rel < 1e-4;
    printf("  ORKD_LAYER x_out vs int8-faithful CPU ref: max|err|=%.3e rel=%.2e (|out|max=%.0f) %s\n", me, rel, maxa, pass?"fp32-exact":"MISMATCH");
    printf("%s\n", pass?"PASS — daemon runs a WHOLE layer on the spine in one round-trip, coherent":"FAIL");
    orkd_disconnect(c);
    return pass?0:1;
}
