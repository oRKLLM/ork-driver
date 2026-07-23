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

/* ---- NPU decode attention (round-robin across heads) --------------------------------------------------------
 * One decode step's attention for ALL query heads, dispatched CONCURRENTLY across the NPU cores: each head is one
 * fused [QK^T -> exp -> Sigma-reduce, e.V] int8 chain, and the NH chains are round-robined over the cores via
 * ork_mm_run_chains_rr. The softmax numerator (e) and the denominator (Sigma) come back separately; we normalize
 * (att = e.V / Sigma) on the host. Cores are assumed WARM (the surrounding Q/K/V/O projection matmuls warm them
 * every step via the multi-core matmul path — see the "chain-only probe warming vs typical path" note).
 *
 * !!! NUMERIC PRECONDITION — POST-MAX DOMAIN !!!  The fused chain computes QK^T and feeds exp ON-CHIP, with no host
 * step between them, so it cannot subtract each head's max score before exp. exp(int8 score) only fits [0,1] when
 * scores are <= 0. Real decode scores are arbitrary, so this path is correct ONLY once per-head max-subtraction is
 * added. TODO(max-bias): fold the per-head max into the QK^T output-stage requant bias (needs a first-pass max, or
 * a chain exp op that reads an external host-max-subtracted int8 score buffer). Until then this branch is gated OFF
 * by default (see ORK_ATTN_NPU_MIN_CTX below) and is exercised only by tools/attn_decode_bench_probe, which
 * constructs post-max-safe inputs. Returns 0 on success, <0 on dispatch error. */
static int attn_decode_npu(ork_npu*ctx,const Cfg*c,const float*q,const kv_t*kv,float*att,int pos){
    int NH=c->NH,NKV=c->NKV,HD=c->HD,KVd=NKV*HD,grp=NH/NKV, L=pos+1, Lp=(L+511)&~511, Kp=512, Nq=1;
    /* chain graph: [0]=QK^T(int8 requant), [1]=exp(reads 0), [2]=Sigma via ones-weight(reads 1), [3]=e.V(reads 1) */
    ork_chain_op ops[4]={ {1,-1,0,0x4000,16}, {2,0,0,0,0}, {0,1,0,0,0}, {0,1,0,0,0} };
    double in_scale=0.0625, out_scale=1.0/127.0;
    /* crude symmetric int8 quant of this step's Q and the cached K/V (per-tensor absmax). TODO(calib): per-head /
     * per-channel scales + the sqrt(HD) attention scale folded into r_mult, instead of this bench-grade absmax. */
    float qmax=1e-6f; for(int i=0;i<NH*HD;i++){ float a=fabsf(q[i]); if(a>qmax)qmax=a; }
    float kmax=1e-6f,vmax=1e-6f; for(int j=0;j<L;j++)for(int e=0;e<KVd;e++){ float ka=fabsf(kv->Kc[(size_t)j*KVd+e]),va=fabsf(kv->Vc[(size_t)j*KVd+e]); if(ka>kmax)kmax=ka; if(va>vmax)vmax=va; }
    float qs=127.0f/qmax, ks=127.0f/kmax, vs=127.0f/vmax;
    ork_w *w_ones=NULL, **w_kt=calloc(NH,sizeof*w_kt), **w_v=calloc(NH,sizeof*w_v);
    int8_t **Qp=calloc(NH,sizeof*Qp); int32_t **scb=calloc(NH,sizeof*scb),**eb=calloc(NH,sizeof*eb),**ss=calloc(NH,sizeof*ss),**avb=calloc(NH,sizeof*avb);
    ork_mm_task_i8 **chains=calloc(NH,sizeof*chains); int *S=calloc(NH,sizeof*S); int rc=0;
    { int8_t *o=malloc((size_t)Lp*32); memset(o,0,(size_t)Lp*32); for(int j=0;j<L;j++)for(int n=0;n<32;n++)o[(size_t)j*32+n]=1; w_ones=ork_mm_pack_i8(ctx,Lp,32,o); free(o); }
    for(int hh=0;hh<NH && !rc;hh++){ int kvh=hh/grp;
        int8_t *KTp=calloc((size_t)Kp*Lp,1), *Vp=calloc((size_t)Lp*HD,1);
        for(int e=0;e<HD;e++)for(int j=0;j<L;j++) KTp[(size_t)e*Lp+j]=(int8_t)lrintf(kv->Kc[(size_t)j*KVd+kvh*HD+e]*ks);
        for(int j=0;j<L;j++)for(int e=0;e<HD;e++) Vp[(size_t)j*HD+e]=(int8_t)lrintf(kv->Vc[(size_t)j*KVd+kvh*HD+e]*vs);
        Qp[hh]=calloc((size_t)Nq*Kp,1); for(int e=0;e<HD;e++) Qp[hh][e]=(int8_t)lrintf(q[hh*HD+e]*qs);
        w_kt[hh]=ork_mm_pack_i8(ctx,Kp,Lp,KTp); w_v[hh]=ork_mm_pack_i8(ctx,Lp,HD,Vp); free(KTp); free(Vp);
        if(!w_kt[hh]||!w_v[hh]||!w_ones){ rc=-2; break; }
        scb[hh]=calloc((size_t)Nq*Lp,4); eb[hh]=calloc((size_t)Nq*Lp,4); ss[hh]=calloc((size_t)Nq*32,4); avb[hh]=calloc((size_t)Nq*HD,4);
        chains[hh]=malloc(4*sizeof(ork_mm_task_i8)); S[hh]=4;
        chains[hh][0]=(ork_mm_task_i8){ w_kt[hh],Nq,Qp[hh],scb[hh] };
        chains[hh][1]=(ork_mm_task_i8){ w_kt[hh],Nq,(int8_t*)scb[hh],eb[hh] };
        chains[hh][2]=(ork_mm_task_i8){ w_ones,Nq,(int8_t*)eb[hh],ss[hh] };
        chains[hh][3]=(ork_mm_task_i8){ w_v[hh],Nq,(int8_t*)eb[hh],avb[hh] };
    }
    if(!rc) rc=ork_mm_run_chains_rr(ctx,NH,(const ork_mm_task_i8*const*)chains,S,ops,in_scale,out_scale);
    if(!rc){ for(int hh=0;hh<NH;hh++){ double Sn=(double)ss[hh][0]; if(Sn<=0)Sn=1;
        for(int e=0;e<HD;e++) att[hh*HD+e]=(float)((double)avb[hh][e]/Sn/vs); } }   /* de-quant V scale */
    for(int hh=0;hh<NH;hh++){ if(w_kt[hh])ork_w_free(w_kt[hh]); if(w_v[hh])ork_w_free(w_v[hh]);
        free(Qp[hh]);free(scb[hh]);free(eb[hh]);free(ss[hh]);free(avb[hh]);free(chains[hh]); }
    if(w_ones){ ork_w_free(w_ones); }
    free(w_kt);free(w_v);free(Qp);free(scb);free(eb);free(ss);free(avb);free(chains);free(S);
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
     * TODO(doorbell-dispatcher): replace this static, per-call, either/or CPU-vs-NPU branch with the DOORBELL
     * HETEROGENEOUS DISPATCHER — one scheduler owning BOTH the CPU worker pool AND the NPU cores, routing each
     * unit of work (per head, or per layer) to CPU or NPU by context length AND live occupancy/queue depth, so
     * long-context heads stream to the NPU while short ones run on CPU, OVERLAPPED rather than exclusive. The
     * context-length test here becomes one input to that router's cost model.
     * TODO(max-bias): the NPU branch (attn_decode_npu) is numerically valid ONLY in the post-max (scores<=0)
     * domain — the fused chain has no per-head max-subtraction (see that function). Enable it for arbitrary
     * scores only after max-subtraction + the sqrt(HD)/calibrated scales land; until then the default threshold
     * keeps it dormant and it is exercised only by the post-max-safe benchmark probe. */
    static int npu_min_ctx = -1;
    if (npu_min_ctx < 0) { const char*e=getenv("ORK_ATTN_NPU_MIN_CTX"); npu_min_ctx = e ? atoi(e) : 512; }
    int used_npu = 0;
    if (useNPU && (pos+1) >= npu_min_ctx) used_npu = (attn_decode_npu(ctx,c,q,kv,att,pos) == 0);
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
