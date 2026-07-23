/* examples/decode.c — incremental decode with a KV cache (the token-generation path), with
 * grouped-query attention + arbitrary head_dim. Each step does M=1 NPU projections, RoPE at
 * the running position, appends K/V (only n_kv_heads worth) to a per-layer cache, and attends
 * over the cache with query heads sharing kv heads. Validates NPU-hybrid decode vs a pure-CPU
 * reference across MHA / GQA / arbitrary-HD configs.
 *   make decode && sudo ./decode
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ork_npu.h"
#include "neon_activations.h"
typedef ork_f16 f16;
#define EPS 1e-5f
#define SEQ 16

typedef struct { const char*name; int H,NH,NKV,HD,FFN; } Cfg;

static void rmsnorm(float*o,const float*x,const float*w,int n){ ork_rmsnorm_f32(o,x,w,n,EPS); }
static void rope_pos(float*x,int nh,int hd,int pos){       /* per head, at absolute position */
    for(int h=0;h<nh;h++){float*v=x+(size_t)h*hd;
        for(int i=0;i<hd/2;i++){float fr=powf(10000.0f,-2.0f*i/hd),ang=pos*fr,c=cosf(ang),s=sinf(ang);
            float a=v[i],b=v[i+hd/2]; v[i]=a*c-b*s; v[i+hd/2]=a*s+b*c;}}
}
static void softmax(float*x,int n){ ork_softmax_f32(x,n); }

typedef struct { float*Kc,*Vc; int len; } kv_t;   /* cache holds KVd = NKV*HD per position */

/* ---- NPU decode attention: HOST-SPLIT (NPU matmuls + full-precision host softmax) — Tier 12e ----------------
 * One decode step's attention for ALL query heads. The two HEAVY matmuls run on the NPU per head (QK^T, then the
 * weighted e.V); the CHEAP [1,L] softmax (per-head max-subtract + exp + normalize) runs on the HOST in fp. This
 * is what ENABLES the NPU branch for real scores: because the host owns the softmax it does a genuine per-head
 * max-subtraction, so the path is CORRECT for ARBITRARY scores — no scores<=0 precondition, no int8-exp
 * approximation, no idx_off RE. (The single-submit FUSED chain / ork_mm_run_chains_rr stays reserved for PREFILL,
 * where large Nq makes on-chip exp worthwhile and its idx_off max-bias is the open RE track — roadmap Tier 12e.)
 * Cores are warmed by the surrounding projection matmuls. Returns 0 ok, <0 on error (gate falls back to CPU).
 *
 * NOTE(overhead): dispatches 2 matmuls/head SEQUENTIALLY and repacks K^T/V each step (the KV cache grows per
 * token). For long context the matmuls dominate and the NPU wins (attn_decode_bench_probe); the per-step pack +
 * per-head submit overhead is exactly what the Tier 12d doorbell dispatcher will amortize (resident K/V + heads
 * dispatched concurrently). The int8 matmul 0x1040 schedule zeroes output for K<256, so the e.V matmul (K=L) needs
 * L>=256 — guarded below; shorter contexts return -3 so the gate stays on CPU (which is faster there anyway). */
static int attn_decode_npu(ork_npu*ctx,const Cfg*c,const float*q,const kv_t*kv,float*att,int pos){
    int NH=c->NH,NKV=c->NKV,HD=c->HD,KVd=NKV*HD,grp=NH/NKV, L=pos+1, Kp=512, rc=0;
    if(L<256) return -3;   /* e.V contraction K=L below the int8 0x1040 sched floor -> let the gate use CPU */
    float scale=1.0f/sqrtf((float)HD);
    int8_t *Qp=calloc((size_t)Kp,1), *KTp=malloc((size_t)Kp*L), *Vp=malloc((size_t)L*HD), *w8=malloc((size_t)L);
    int32_t *scores=malloc((size_t)L*4), *attv=malloc((size_t)HD*4); double *sc=malloc((size_t)L*sizeof(double));
    if(!Qp||!KTp||!Vp||!w8||!scores||!attv||!sc){ rc=-2; goto done; }
    for(int hh=0; hh<NH && !rc; hh++){ int kvh=hh/grp;
        /* per-head symmetric int8 quant scales (absmax over this head's Q and this kv-head's K/V slice) */
        float qmax=1e-6f,kmax=1e-6f,vmax=1e-6f;
        for(int e=0;e<HD;e++){ float a=fabsf(q[(size_t)hh*HD+e]); if(a>qmax)qmax=a; }
        for(int j=0;j<L;j++)for(int e=0;e<HD;e++){ float ka=fabsf(kv->Kc[(size_t)j*KVd+kvh*HD+e]),va=fabsf(kv->Vc[(size_t)j*KVd+kvh*HD+e]); if(ka>kmax)kmax=ka; if(va>vmax)vmax=va; }
        float qs=127.0f/qmax, ks=127.0f/kmax, vs=127.0f/vmax;
        /* NPU pass 1 — QK^T: W=K^T[Kp,L] int8 (head_dim zero-padded to Kp), A=Q[1,Kp] int8 -> scores[1,L] int32 */
        for(int e=0;e<HD;e++)for(int j=0;j<L;j++) KTp[(size_t)e*L+j]=(int8_t)lrintf(kv->Kc[(size_t)j*KVd+kvh*HD+e]*ks);
        memset(Qp,0,Kp); for(int e=0;e<HD;e++) Qp[e]=(int8_t)lrintf(q[(size_t)hh*HD+e]*qs);
        ork_w *wkt=ork_mm_pack_i8(ctx,Kp,L,KTp); if(!wkt){ rc=-2; break; }
        ork_mm_task_i8 t1={ wkt,1,Qp,scores }; rc=ork_mm_run_chain_i8(ctx,1,&t1); ork_w_free(wkt); if(rc) break;
        /* HOST softmax (fp, REAL per-head max-subtraction) — the piece the fused chain can't do */
        double mx=-1e300; for(int j=0;j<L;j++){ sc[j]=(double)scores[j]/((double)qs*ks)*scale; if(sc[j]>mx)mx=sc[j]; }
        double Z=0; for(int j=0;j<L;j++){ sc[j]=exp(sc[j]-mx); Z+=sc[j]; } if(Z<=0)Z=1;
        /* normalize, then quantize weights with ws=127/max(weight): softmax weights are ~1/L (tiny), so a fixed
         * ws=127 would round them all to 0 (int8 underflow). Scale by the max so the peak weight hits 127. */
        double wmax=0; for(int j=0;j<L;j++){ sc[j]/=Z; if(sc[j]>wmax)wmax=sc[j]; }
        double ws=127.0/(wmax>1e-9?wmax:1.0);
        for(int j=0;j<L;j++){ int wi=(int)lrint(sc[j]*ws); w8[j]=(int8_t)(wi>127?127:(wi<0?0:wi)); }
        /* NPU pass 2 — e.V: W=V[L,HD] int8, A=weights[1,L] int8 -> att[1,HD] int32 */
        for(int j=0;j<L;j++)for(int e=0;e<HD;e++) Vp[(size_t)j*HD+e]=(int8_t)lrintf(kv->Vc[(size_t)j*KVd+kvh*HD+e]*vs);
        ork_w *wv=ork_mm_pack_i8(ctx,L,HD,Vp); if(!wv){ rc=-2; break; }
        ork_mm_task_i8 t2={ wv,1,w8,attv }; rc=ork_mm_run_chain_i8(ctx,1,&t2); ork_w_free(wv); if(rc) break;
        for(int e=0;e<HD;e++) att[(size_t)hh*HD+e]=(float)((double)attv[e]/(ws*vs));  /* de-quant: weight-scale ws * V-scale vs */
    }
done:
    free(Qp);free(KTp);free(Vp);free(w8);free(scores);free(attv);free(sc);
    return rc;
}

typedef struct { int K,N; f16 *Brow; ork_w *w; } weight;
static weight mkw(ork_npu*ctx,int K,int N,unsigned seed){weight wt={K,N,malloc((size_t)K*N*2),NULL};
    unsigned s=seed; for(size_t i=0;i<(size_t)K*N;i++){s=s*1103515245+12345;wt.Brow[i]=(f16)((((int)((s>>16)%9))-4)*(0.5f/sqrtf((float)K)));}
    wt.w=ork_mm_pack(ctx,K,N,wt.Brow); return wt;}
static void mm1(ork_npu*ctx,weight*wt,const float*Af32,float*C,int useNPU){   /* M=1 */
    int K=wt->K,N=wt->N; f16*A=malloc((size_t)K*2);
    for(int i=0;i<K;i++)A[i]=(f16)Af32[i];
    if(useNPU) ork_mm_run(ctx,wt->w,1,A,C);
    else for(int n=0;n<N;n++){float acc=0;for(int k=0;k<K;k++)acc+=(float)A[k]*(float)wt->Brow[(size_t)k*N+n];C[n]=acc;}
    free(A);
}

static void step(ork_npu*ctx,const Cfg*c,const float*x1,float*y1,kv_t*kv,const float*wn1,const float*wn2,
        weight*Wq,weight*Wk,weight*Wv,weight*Wo,weight*Wg,weight*Wu,weight*Wd,int useNPU){
    int H=c->H,NH=c->NH,NKV=c->NKV,HD=c->HD,FFN=c->FFN, Qd=NH*HD,KVd=NKV*HD,grp=NH/NKV, pos=kv->len;
    float*xn=malloc(H*4),*q=malloc(Qd*4),*k=malloc(KVd*4),*v=malloc(KVd*4),*att=malloc(Qd*4),*o=malloc(H*4),*h=malloc(H*4),*hn=malloc(H*4),*g=malloc(FFN*4),*u=malloc(FFN*4),*a=malloc(FFN*4),*d=malloc(H*4);
    rmsnorm(xn,x1,wn1,H);
    mm1(ctx,Wq,xn,q,useNPU); mm1(ctx,Wk,xn,k,useNPU); mm1(ctx,Wv,xn,v,useNPU);
    rope_pos(q,NH,HD,pos); rope_pos(k,NKV,HD,pos);
    memcpy(kv->Kc+(size_t)pos*KVd,k,KVd*4); memcpy(kv->Vc+(size_t)pos*KVd,v,KVd*4); kv->len++;
    float scale=1.0f/sqrtf((float)HD);
    /* ATTENTION — CPU/NPU routing GATED ON CONTEXT LENGTH (pos+1 = keys this token attends to).
     * WHY the gate: benchmarked (tools/attn_decode_bench_probe, rk3588) the NPU round-robin chain vs this CPU
     * scalar softmax. Decode attention is Nq=1 per head -> very low arithmetic intensity, so at SHORT context the
     * NPU is submit/weight-read-bound and CPU wins (a tie at L~512); as context GROWS the per-head QK^T and e.V
     * matmul work grows and amortizes the NPU's fixed overhead while the CPU scales linearly, so the NPU pulls
     * ahead (measured 2.6x @L=1024, 4.2x @L=2048). => route to the NPU only once context clears the crossover.
     * Threshold is env-tunable (ORK_ATTN_NPU_MIN_CTX) for experiments; the default keeps short contexts — and this
     * file's tiny SEQ=16 validation — on the CPU path so decode's existing NPU-vs-CPU check is unaffected.
     *
     * The NPU branch (attn_decode_npu) is now the HOST-SPLIT path — NPU matmuls + full-precision host softmax —
     * so it is CORRECT for arbitrary scores (real per-head max-subtraction), not just the post-max domain. It is
     * enabled by the gate; the threshold is purely the perf crossover, not a correctness guard (attn_decode_npu
     * additionally self-guards L>=256 for the int8 sched floor and returns -3 -> CPU below that).
     * TODO(doorbell-dispatcher): replace this static, per-call, either/or CPU-vs-NPU branch with the DOORBELL
     * HETEROGENEOUS DISPATCHER — one scheduler owning BOTH the CPU worker pool AND the NPU cores, routing each
     * unit of work (per head, or per layer) to CPU or NPU by context length AND live occupancy/queue depth, so
     * long-context heads stream to the NPU while short ones run on CPU, OVERLAPPED rather than exclusive. The
     * context-length test here becomes one input to that router's cost model. (Roadmap Tier 12d.) */
    /* Default DISABLED (0). The NPU branch is numerically READY (host-split, coherent for arbitrary scores), but
     * it repacks K^T/V every step — that per-step packing is ~15x the matmul cost and makes it perf-NEGATIVE vs
     * CPU until K/V are kept RESIDENT (pack-once + append), which is the Tier 12d dispatcher work. So it stays off
     * by default; set ORK_ATTN_NPU_MIN_CTX=<ctx> to force-enable it above a context length for experiments. */
    static int npu_min_ctx = -2;
    if (npu_min_ctx == -2) { const char*e=getenv("ORK_ATTN_NPU_MIN_CTX"); npu_min_ctx = e ? atoi(e) : 0; }
    int used_npu = 0;
    if (useNPU && npu_min_ctx > 0 && (pos+1) >= npu_min_ctx) used_npu = (attn_decode_npu(ctx,c,q,kv,att,pos) == 0);
    if (!used_npu) {   /* CPU scalar softmax attention: default path, and the fallback when the gate is closed/errs */
      for(int hh=0;hh<NH;hh++){int kvh=hh/grp; float sc[SEQ];
        for(int j=0;j<=pos;j++){float dt=0;for(int e=0;e<HD;e++)dt+=q[hh*HD+e]*kv->Kc[(size_t)j*KVd+kvh*HD+e];sc[j]=dt*scale;}
        softmax(sc,pos+1);
        for(int e=0;e<HD;e++){float ac=0;for(int j=0;j<=pos;j++)ac+=sc[j]*kv->Vc[(size_t)j*KVd+kvh*HD+e];att[hh*HD+e]=ac;}
      }
    }
    mm1(ctx,Wo,att,o,useNPU);
    for(int i=0;i<H;i++)h[i]=x1[i]+o[i];
    rmsnorm(hn,h,wn2,H);
    mm1(ctx,Wg,hn,g,useNPU); mm1(ctx,Wu,hn,u,useNPU);
    ork_silu_mul_to_f32(a,g,u,FFN);   /* SwiGLU: a = silu(g)*u (NEON) */
    mm1(ctx,Wd,a,d,useNPU);
    for(int i=0;i<H;i++)y1[i]=h[i]+d[i];
    free(xn);free(q);free(k);free(v);free(att);free(o);free(h);free(hn);free(g);free(u);free(a);free(d);
}
static int run_cfg(ork_npu*ctx,const Cfg*c){
    int H=c->H,NH=c->NH,NKV=c->NKV,HD=c->HD,FFN=c->FFN, Qd=NH*HD,KVd=NKV*HD;
    float*x=malloc(SEQ*H*4),*yn=malloc(SEQ*H*4),*yc=malloc(SEQ*H*4),*wn1=malloc(H*4),*wn2=malloc(H*4);
    unsigned s=42; for(int i=0;i<SEQ*H;i++){s=s*1103515245+12345;x[i]=(((int)((s>>16)%17))-8)*0.1f;}
    for(int i=0;i<H;i++){wn1[i]=1.0f+((int)((i+1u)%5)-2)*0.02f;wn2[i]=1.0f+((int)((i+1u)%7)-3)*0.02f;}
    weight Wq=mkw(ctx,H,Qd,11),Wk=mkw(ctx,H,KVd,22),Wv=mkw(ctx,H,KVd,33),Wo=mkw(ctx,Qd,H,44);
    weight Wg=mkw(ctx,H,FFN,55),Wu=mkw(ctx,H,FFN,66),Wd=mkw(ctx,FFN,H,77);
    kv_t kvn={malloc((size_t)SEQ*KVd*4),malloc((size_t)SEQ*KVd*4),0};
    kv_t kvc={malloc((size_t)SEQ*KVd*4),malloc((size_t)SEQ*KVd*4),0};
    for(int t=0;t<SEQ;t++) step(ctx,c,x+t*H,yn+t*H,&kvn,wn1,wn2,&Wq,&Wk,&Wv,&Wo,&Wg,&Wu,&Wd,1);
    for(int t=0;t<SEQ;t++) step(ctx,c,x+t*H,yc+t*H,&kvc,wn1,wn2,&Wq,&Wk,&Wv,&Wo,&Wg,&Wu,&Wd,0);
    float maxabs=0,refmax=0; int nans=0;
    for(int i=0;i<SEQ*H;i++){if(!isfinite(yn[i])||!isfinite(yc[i])){nans++;continue;}
        float e=fabsf(yn[i]-yc[i]);if(e>maxabs)maxabs=e;if(fabsf(yc[i])>refmax)refmax=fabsf(yc[i]);}
    int ok=nans==0 && refmax>1e-3f && maxabs<0.05f*refmax+1e-2f;
    printf("  %-10s H=%d NH=%d NKV=%d HD=%d (kv-cache %d/pos) : maxabs=%.4g |ref|=%.3g : %s\n",
        c->name,H,NH,NKV,HD,KVd,maxabs,refmax,ok?"OK":"MISMATCH");
    ork_w_free(Wq.w);ork_w_free(Wk.w);ork_w_free(Wv.w);ork_w_free(Wo.w);ork_w_free(Wg.w);ork_w_free(Wu.w);ork_w_free(Wd.w);
    free(Wq.Brow);free(Wk.Brow);free(Wv.Brow);free(Wo.Brow);free(Wg.Brow);free(Wu.Brow);free(Wd.Brow);
    free(kvn.Kc);free(kvn.Vc);free(kvc.Kc);free(kvc.Vc); free(x);free(yn);free(yc);free(wn1);free(wn2);
    return ok?0:1;
}
int main(void){
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    Cfg cfgs[]={{"MHA",512,8,8,64,2048},{"GQA-4x",512,8,2,64,2048},{"GQA-arbHD",512,4,2,96,2048}};
    int fail=0; printf("KV-cache decode (NPU vs CPU ref):\n");
    for(unsigned i=0;i<sizeof cfgs/sizeof*cfgs;i++) fail|=run_cfg(ctx,&cfgs[i]);
    ork_npu_free(ctx);
    printf("%s\n",fail?"FAIL":"ALL OK"); return fail?2:0;
}
