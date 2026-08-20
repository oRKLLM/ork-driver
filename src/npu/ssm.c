/* npu/ssm.c — Mamba-2 / SSD state-space scan on the NPU (ork_ssm_scan_f32) and its supporting machinery:
 * the persistent per-shape scratch POOL (the 4*nh fp16 scratch weights + CPU staging buffers, reused across
 * calls because the per-call MEM_CREATE/MEM_DESTROY churn dominated the in-model scan), the little-core
 * (A55) marshalling helper that builds the G-independent operands while the pS matmul runs on the big
 * cores, the per-stage int8 path, and the ORK_SSM_* knobs.
 *
 * Lifted verbatim out of npu.c by the precision split (MODULARIZE_PLAN.md round 1, commit D). This is the
 * first real translation unit off the monolith: a single contiguous block with a 3-in / 3-out boundary.
 * Context: the scan LOSES at 130M but WINS ~2x byte-coherent at 2.7B-Q8_0 — the crossover is real (the CPU
 * recurrence is cache-bound and degrades 33x while the NPU path degrades 11x). See the wiki SSM pages. */
#define _GNU_SOURCE   /* CPU_SET/pthread_setaffinity_np, as npu.c does */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include "ork_regs.h"
#include "regcmd_silu.h"
#include "regcmd_i8.h"
#include "npu/internal.h"
#include "npu/core.h"

static inline double ork_softplus(double v){ return v>20.0 ? v : log1p(exp(v)); }
/* SSM_SCAN (Mamba-2) via the chunked mode-5 scan: matmul spine on the NPU (persistent weight pool +
 * 3-core round-robin stream), elementwise on the CPU. Matches ggml_compute_forward_ssm_scan_f32:
 * dt_soft_plus=softplus(dt), scalar decay A{1,nh}, NO D skip (applied elsewhere in the graph), output
 * y (x-shaped) + s_new (updated state). Contiguous ggml layout:
 *   s/s_new {nc=d_state, nr=dim, nh, ns}   x/y {nr, nh, nt, ns}   dt {nh, nt, ns}   A {nh}   B/C {nc, ng, nt, ns}
 * nt is internally chunk-padded (pad tokens: dt=-inf -> softplus~0 -> zero contribution). Requires
 * nc%32, nr%16, nh%ng==0. 0/ok, <0. rk3588. */
/* ORK_SSM_I8_MASK: per-stage int8 selection bitmask (bit0=pS gram C·B, bit1=pD, bit2=pC state, bit3=pO).
 * Default 0 = all-fp16 (current). int8 stages are ~2-3x faster on the NPU but add per-op quant/dequant and
 * risk coherence; keep the recurrent-state stages (pC) fp16. Needs ORK_SSM_KEEPWARM=1 or the intra-scan
 * int8<->fp16 stage transitions re-introduce the mode-switch churn. */
static int ork_ssm_i8_mask(void){ static int v=-2; if(v==-2){const char*e=getenv("ORK_SSM_I8_MASK"); v=e?atoi(e):0;} return v; }
/* ORK_SSM_CHAIN: route the fp16 scan stages through the chained-multicore stream (one task_number>1 submit
 * per core, PC-chaining the core's heads) instead of run_stream_f16 (one submit per head). Escapes the
 * per-matmul submit floor while keeping 3-core parallelism. Off by default. */
static int ork_ssm_chain(void){ static int v=-2; if(v==-2){const char*e=getenv("ORK_SSM_CHAIN"); v=e?atoi(e):1;} return v; }  /* DEFAULT ON: fused-multicore fp16 stream, part of the validated SSM-scan win. ORK_SSM_CHAIN=0 to disable. */
/* ORK_SSM_CS: SSD chunk size (default 64). Bigger CS = fewer chunks (NC=nt/CS) but O(CS^2) intra-chunk
 * work (the G / decay-mask are CS x CS). Clamped to a %16 value in [16,256]. */
static int ork_ssm_cs(void){ static int v=-1; if(v<0){const char*e=getenv("ORK_SSM_CS"); v=e?atoi(e):128; if(v<16||v>256||v%16)v=128;} return v; }  /* DEFAULT 128: +5% short-prefill (CS≈nt capped 128); ORK_SSM_CS overrides (64 for long prefill). */
/* ORK_SSM_BATCH: batch the chunk-independent matmul stages ACROSS all NC chunks — each stage becomes ONE
 * dispatch of nh*NC matmuls (instead of nh, NC times). Only the CPU inter-chunk carry stays sequential.
 * Cuts NPU dispatches from 4*NC to 4 (fewer pool wake-ups, better floor amortization). fp16-only. */
static int ork_ssm_batch(void){ static int v=-2; if(v==-2){const char*e=getenv("ORK_SSM_BATCH"); v=e?atoi(e):0;} return v; }
/* ORK_SSM_PIPELINE: double-buffer the scan — the G-independent operands (aC/bD/bO/bC, for pD/pC/pO) are
 * marshalled on a helper thread (free 4th big core) WHILE the pS matmul runs on cores 0-2, so the ~47% CPU
 * marshalling hides behind the NPU submit instead of adding to it. Only aD (needs pS's G) stays serial. */
static int ork_ssm_pipeline(void){ static int v=-2; if(v==-2){const char*e=getenv("ORK_SSM_PIPELINE"); v=e?atoi(e):0;} return v; }
struct ssm_marshal_arg { ork_f16 *aC,*bD,*bO,*bC; const double *xbar,*Acs; const float *state,*B;
                         int nh,nr,nc,CS,ng,hpg,base,nt,seq; };
static void *ssm_marshal_gi(void *vp){   /* G-independent operand build (aC,bD,bO,bC); runs during the pS submit */
    struct ssm_marshal_arg *m=vp; int nh=m->nh,nr=m->nr,nc=m->nc,CS=m->CS,hpg=m->hpg,base=m->base,nt=m->nt,seq=m->seq,ng=m->ng;
    for(int h=0;h<nh;h++){ int g=h/hpg; const double *Ah=m->Acs+(size_t)h*CS;
        for(int i1=0;i1<nr;i1++)for(int sp=0;sp<CS;sp++) m->aC[((size_t)h*nr+i1)*CS+sp]=(ork_f16)m->xbar[((size_t)h*CS+sp)*nr+i1];
        for(int l=0;l<CS;l++)for(int i1=0;i1<nr;i1++) m->bD[((size_t)h*CS+l)*nr+i1]=(ork_f16)m->xbar[((size_t)h*CS+l)*nr+i1];
        for(int n=0;n<nc;n++)for(int i1=0;i1<nr;i1++) m->bO[((size_t)h*nc+n)*nr+i1]=(ork_f16)m->state[((size_t)h*nr+i1)*nc+n];
        for(int sp=0;sp<CS;sp++){ int t=base+sp; double ds=exp(Ah[CS-1]-Ah[sp]); const float *Bc=(t<nt)?m->B+((size_t)(seq*nt+t)*ng+g)*nc:NULL;
            for(int n=0;n<nc;n++) m->bC[((size_t)h*CS+sp)*nc+n]=(ork_f16)(Bc?ds*Bc[n]:0.0); } }
    return NULL;
}
/* persistent little-core marshalling helper: spawned once, condvar-signalled per chunk. */
static void *ssm_helper_worker(void *vp){
    ork_npu *c=vp; orki_pin_little_core(0);                 /* live on an idle A55 for the whole run */
    pthread_mutex_lock(&c->ssm_hmu);
    for(;;){ while(!c->ssm_hgen && !c->ssm_hstop) pthread_cond_wait(&c->ssm_hgo,&c->ssm_hmu);  /* hgen = pending flag */
        if(c->ssm_hstop){ pthread_mutex_unlock(&c->ssm_hmu); return NULL; }
        c->ssm_hgen=0; void *job=c->ssm_hjob; pthread_mutex_unlock(&c->ssm_hmu);
        ssm_marshal_gi(job);
        pthread_mutex_lock(&c->ssm_hmu); c->ssm_hdone=1; pthread_cond_signal(&c->ssm_hdn); }
}
static int ssm_helper_ensure(ork_npu *c){
    if(c->ssm_hspawn) return 1;
    pthread_mutex_init(&c->ssm_hmu,NULL); pthread_cond_init(&c->ssm_hgo,NULL); pthread_cond_init(&c->ssm_hdn,NULL);
    c->ssm_hgen=0; c->ssm_hstop=0; c->ssm_hdone=1;
    c->ssm_hspawn=(pthread_create(&c->ssm_hth,NULL,ssm_helper_worker,c)==0);
    return c->ssm_hspawn;
}
static void ssm_helper_fire(ork_npu *c, void *job){    /* non-blocking dispatch */
    pthread_mutex_lock(&c->ssm_hmu); c->ssm_hjob=job; c->ssm_hdone=0; c->ssm_hgen=1; pthread_cond_signal(&c->ssm_hgo); pthread_mutex_unlock(&c->ssm_hmu); }
static void ssm_helper_join(ork_npu *c){               /* wait for the fired job */
    pthread_mutex_lock(&c->ssm_hmu); while(!c->ssm_hdone) pthread_cond_wait(&c->ssm_hdn,&c->ssm_hmu); pthread_mutex_unlock(&c->ssm_hmu); }
void ork_ssm_helper_stop(ork_npu *c){                  /* teardown: called from ork_npu_free */
    if(!c->ssm_hspawn) return;
    pthread_mutex_lock(&c->ssm_hmu); c->ssm_hstop=1; pthread_cond_signal(&c->ssm_hgo); pthread_mutex_unlock(&c->ssm_hmu);
    pthread_join(c->ssm_hth,NULL); c->ssm_hspawn=0; }
/* ORK_SSM_PROF: per-section accounting for ork_ssm_scan_f32 (accumulated across calls, dumped at teardown). */
static int g_ssm_prof=-1;
static double g_ssm_prep,g_ssm_stage,g_ssm_repack,g_ssm_npu,g_ssm_post,g_ssm_stg[4]; static long g_ssm_calls;
void ork_ssm_prof_dump(void){ if(g_ssm_prof<=0||!g_ssm_calls) return;
    double t=g_ssm_prep+g_ssm_stage+g_ssm_repack+g_ssm_npu+g_ssm_post; if(t<=0)t=1;
    fprintf(stderr,"[ork SSM_PROF] %ld calls, %.1f ms | prep(softplus+cumsum+xbar) %.1f (%.0f%%) | stage(fp16 operand cast) %.1f (%.0f%%) | repack(NPU tile) %.1f (%.0f%%) | NPU(matmul submit) %.1f (%.0f%%) | post(y-assembly) %.1f (%.0f%%)\n",
        g_ssm_calls,t/1e3, g_ssm_prep/1e3,100*g_ssm_prep/t, g_ssm_stage/1e3,100*g_ssm_stage/t, g_ssm_repack/1e3,100*g_ssm_repack/t, g_ssm_npu/1e3,100*g_ssm_npu/t, g_ssm_post/1e3,100*g_ssm_post/t);
    fprintf(stderr,"[ork SSM_PROF] per-stage matmul (repack+run): pS(C.B) %.1f | pD(intra-Y) %.1f | pC(state) %.1f | pO(inter-Y) %.1f ms\n",
        g_ssm_stg[0]/1e3,g_ssm_stg[1]/1e3,g_ssm_stg[2]/1e3,g_ssm_stg[3]/1e3); }
/* int8 variant of one scan STG stage: per-col-quant the weight (Bop) via repack_i8_f32, per-row-quant the
 * activation (Aop), batched int8 stream, then dequant int32->Cop by ascale[row]*bscale[col]. Same operand
 * layout/offsets as the fp16 STG macro. Uses the persistent int8 scratch + quant temps on the ctx. */
static int ssm_stg_i8(ork_npu *c, ork_w **pool, int M,int K,int N,
        const ork_f16 *Aop, const ork_f16 *Bop, float *Cop, int nh){
    int8_t *ai8=c->ssm_ai8; int32_t *ci32=c->ssm_ci32; float *ascale=c->ssm_ascale,*bscale=c->ssm_bscale;
    int8_t *bi8=malloc((size_t)K*N); if(!bi8) return -1;   /* per-stg weight-quant temp (reused across heads) */
    int ret=0;
    for(int h=0;h<nh;h++){
        /* weight B[K,N] fp16 -> per-col int8 (bscale[n]=absmax(col)/127); repack_i8 matches run_i8's Bb tiling
         * (NOT repack_i8_f32, which tiles for the Bf/stream path -> the earlier rel-L2=1.4 layout garbage). */
        const ork_f16 *B=Bop+(size_t)h*K*N; float *bs=bscale+(size_t)h*N;
        for(int n=0;n<N;n++){ float mx=1e-9f; for(int k=0;k<K;k++){ float v=fabsf((float)B[(size_t)k*N+n]); if(v>mx)mx=v; } bs[n]=mx/127.0f; }
        for(int k=0;k<K;k++)for(int n=0;n<N;n++){ int q=(int)lrintf((float)B[(size_t)k*N+n]/bs[n]); if(q>127)q=127; if(q<-127)q=-127; bi8[(size_t)k*N+n]=(int8_t)q; }
        if(ork_i8_mm_repack(c,pool[h],K,N,bi8)){ if(getenv("ORK_SSM_DBG"))fprintf(stderr,"[ssm i8] repack_i8 h=%d rc\n",h); ret=-1; goto done; }
        /* activation A[M,K] fp16 -> per-row int8 (ascale[m]=absmax(row)/127) */
        const ork_f16 *Aa=Aop+(size_t)h*M*K; int8_t *Ai=ai8+(size_t)h*M*K; float *as=ascale+(size_t)h*M;
        for(int m=0;m<M;m++){ const ork_f16 *r=Aa+(size_t)m*K; float mx=1e-9f;
            for(int k=0;k<K;k++){ float v=fabsf((float)r[k]); if(v>mx)mx=v; }
            as[m]=mx/127.0f; float inv=127.0f/mx; int8_t *o=Ai+(size_t)m*K;
            for(int k=0;k<K;k++){ int q=(int)lrintf((float)r[k]*inv); if(q>127)q=127; if(q<-127)q=-127; o[k]=(int8_t)q; } }
        c->ssm_tki8[h]=(ork_mm_task_i8){pool[h],M,Ai,ci32+(size_t)h*M*N};
    }
    /* small-K int8 batched 3-core stream (ORK_SSM_I8_PERHEAD=1 forces the per-head single-core path for A/B) */
    if(getenv("ORK_SSM_I8_PERHEAD")){
        for(int h=0;h<nh;h++){ int rr=ork_i8_mm_run(c,pool[h],M,ai8+(size_t)h*M*K,ci32+(size_t)h*M*N);
            if(rr){ if(getenv("ORK_SSM_DBG"))fprintf(stderr,"[ssm i8] run_i8 h=%d rc=%d\n",h,rr); ret=-1; goto done; } }
    } else {
        int rr=ork_i8_mm_run_stream_sk(c,nh,c->ssm_tki8);
        if(rr){ if(getenv("ORK_SSM_DBG"))fprintf(stderr,"[ssm i8] run_stream_i8_sk nh=%d M=%d K=%d N=%d rc=%d\n",nh,M,K,N,rr); ret=-1; goto done; }
    }
    for(int h=0;h<nh;h++){ const int32_t *ci=ci32+(size_t)h*M*N; const float *as=ascale+(size_t)h*M,*bs=bscale+(size_t)h*N; float *co=Cop+(size_t)h*M*N;
        for(int m=0;m<M;m++)for(int n=0;n<N;n++) co[(size_t)m*N+n]=(float)ci[(size_t)m*N+n]*as[m]*bs[n]; }
done:
    free(bi8); return ret;
}
/* Free the persistent SSM-scan pool (scratch weights + CPU buffers) and reset its shape key. */
void orki_ssm_pool_free(ork_npu *c){
    int NS=c->ssm_nh*(c->ssm_nb>0?c->ssm_nb:1);   /* nh*nb scratch slots (nb=NC chunk-slots when batched) */
    if(c->ssm_pS) for(int h=0;h<NS;h++){ if(c->ssm_pS[h])ork_mm_free(c,c->ssm_pS[h]); if(c->ssm_pD&&c->ssm_pD[h])ork_mm_free(c,c->ssm_pD[h]); if(c->ssm_pC&&c->ssm_pC[h])ork_mm_free(c,c->ssm_pC[h]); if(c->ssm_pO&&c->ssm_pO[h])ork_mm_free(c,c->ssm_pO[h]); }
    free(c->ssm_pS);free(c->ssm_pD);free(c->ssm_pC);free(c->ssm_pO);
    free(c->ssm_aS);free(c->ssm_bS);free(c->ssm_G);free(c->ssm_aD);free(c->ssm_bD);free(c->ssm_Yd);
    free(c->ssm_aC);free(c->ssm_bC);free(c->ssm_cs);free(c->ssm_aO);free(c->ssm_bO);free(c->ssm_tmp);
    free(c->ssm_Acs);free(c->ssm_xbar);free(c->ssm_state);free(c->ssm_stp);free(c->ssm_Acl);free(c->ssm_Aclp);free(c->ssm_tk);
    c->ssm_pS=c->ssm_pD=c->ssm_pC=c->ssm_pO=NULL;
    c->ssm_aS=c->ssm_bS=c->ssm_aD=c->ssm_bD=c->ssm_aC=c->ssm_bC=c->ssm_aO=c->ssm_bO=NULL;
    c->ssm_G=c->ssm_Yd=c->ssm_cs=c->ssm_tmp=c->ssm_state=c->ssm_stp=NULL;
    c->ssm_Acs=c->ssm_xbar=c->ssm_Acl=c->ssm_Aclp=NULL; c->ssm_tk=NULL;
    if(c->ssm_pSi8) for(int h=0;h<c->ssm_nh;h++) if(c->ssm_pSi8[h])ork_mm_free(c,c->ssm_pSi8[h]);
    free(c->ssm_pSi8);free(c->ssm_ai8);free(c->ssm_ci32);free(c->ssm_ascale);free(c->ssm_bscale);free(c->ssm_f32t);free(c->ssm_tki8);
    c->ssm_pSi8=NULL; c->ssm_ai8=NULL; c->ssm_ci32=NULL; c->ssm_ascale=c->ssm_bscale=c->ssm_f32t=NULL; c->ssm_tki8=NULL; c->ssm_i8=0;
    c->ssm_nc=c->ssm_nr=c->ssm_nh=c->ssm_nb=0;
}
/* Ensure the pool is allocated for (nc,nr,nh); reuse if the shape matches, else realloc. CS is fixed (64). */
static int ssm_pool_ensure(ork_npu *c,int nc,int nr,int nh,int CS,int nb){
    if(c->ssm_pS && c->ssm_nc==nc && c->ssm_nr==nr && c->ssm_nh==nh && c->ssm_csz==CS && c->ssm_nb==nb) return 0;   /* warm reuse */
    orki_ssm_pool_free(c);
    int NS=nh*nb;   /* nh heads x nb chunk-slots (nb=NC when batched, else 1) */
    c->ssm_pS=calloc(NS,sizeof(ork_w*));c->ssm_pD=calloc(NS,sizeof(ork_w*));c->ssm_pC=calloc(NS,sizeof(ork_w*));c->ssm_pO=calloc(NS,sizeof(ork_w*));
    if(!c->ssm_pS||!c->ssm_pD||!c->ssm_pC||!c->ssm_pO){ orki_ssm_pool_free(c); return -1; }
    for(int h=0;h<NS;h++){ c->ssm_pS[h]=ork_f16_mm_scratch(c,nc,CS); c->ssm_pD[h]=ork_f16_mm_scratch(c,CS,nr); c->ssm_pC[h]=ork_f16_mm_scratch(c,CS,nc); c->ssm_pO[h]=ork_f16_mm_scratch(c,nc,nr);
        if(!c->ssm_pS[h]||!c->ssm_pD[h]||!c->ssm_pC[h]||!c->ssm_pO[h]){ c->ssm_nh=nh; c->ssm_nb=nb; orki_ssm_pool_free(c); return -1; } }
    c->ssm_aS=malloc((size_t)NS*CS*nc*2);c->ssm_bS=malloc((size_t)NS*nc*CS*2);c->ssm_G=malloc((size_t)NS*CS*CS*4);
    c->ssm_aD=malloc((size_t)NS*CS*CS*2);c->ssm_bD=malloc((size_t)NS*CS*nr*2);c->ssm_Yd=malloc((size_t)NS*CS*nr*4);
    c->ssm_aC=malloc((size_t)NS*nr*CS*2);c->ssm_bC=malloc((size_t)NS*CS*nc*2);c->ssm_cs=malloc((size_t)NS*nr*nc*4);
    c->ssm_aO=malloc((size_t)NS*CS*nc*2);c->ssm_bO=malloc((size_t)NS*nc*nr*2);c->ssm_tmp=malloc((size_t)NS*CS*nr*4);
    c->ssm_Acs=malloc((size_t)NS*CS*8);c->ssm_xbar=malloc((size_t)NS*CS*nr*8);
    c->ssm_state=malloc((size_t)nh*nr*nc*4);c->ssm_stp=malloc((size_t)NS*nr*nc*4);
    c->ssm_Acl=malloc((size_t)NS*8);c->ssm_Aclp=malloc(nh*8);c->ssm_tk=malloc((size_t)NS*sizeof(ork_mm_task_f16));
    /* int8 stage scratch + quant temps (only when a stage is int8-selected). pS weight dims K=nc,N=CS;
     * temps sized nh*CS*nc (generous — covers any single stage's M*K / M*N). */
    c->ssm_i8=ork_ssm_i8_mask();
    if(c->ssm_i8&1){
        c->ssm_pSi8=calloc(nh,sizeof(ork_w*)); if(!c->ssm_pSi8){ c->ssm_nh=nh; orki_ssm_pool_free(c); return -1; }
        int8_t *zero=calloc((size_t)nc*CS,1);
        for(int h=0;h<nh;h++){ c->ssm_pSi8[h]=ork_i8_mm_pack(c,nc,CS,zero); if(!c->ssm_pSi8[h]){ free(zero); c->ssm_nh=nh; orki_ssm_pool_free(c); return -1; } }
        free(zero);
        c->ssm_ai8=malloc((size_t)nh*CS*nc); c->ssm_ci32=malloc((size_t)nh*CS*nc*4);
        c->ssm_ascale=malloc((size_t)nh*CS*4); c->ssm_bscale=malloc((size_t)nh*nc*4);
        c->ssm_f32t=malloc((size_t)nc*CS*4); c->ssm_tki8=malloc((size_t)nh*sizeof(ork_mm_task_i8));
    }
    c->ssm_nc=nc;c->ssm_nr=nr;c->ssm_nh=nh;c->ssm_csz=CS;c->ssm_nb=nb;
    return 0;
}
int ork_ssm_scan_f32(ork_npu *c,int nc,int nr,int nh,int ng,int nt,int ns,
                     const float *s0,const float *x,const float *dt,const float *A,
                     const float *B,const float *C,float *y,float *s_new){
    if(!c||nc<1||nr<1||nh<1||ng<1||nt<1||ns<1) return -2;
    if(nc%32||nr%16||nc%16||nh%ng) return -2;
    const int CS=ork_ssm_cs(), NC=(nt+CS-1)/CS, hpg=nh/ng;
    int nb=(ork_ssm_batch() && NC>1 && !(ork_ssm_i8_mask()&1)) ? NC : 1;   /* batch across chunks (fp16-only) */
    if(ssm_pool_ensure(c,nc,nr,nh,CS,nb)) return -1;   /* persistent pool: alloc once per shape, reuse across calls */
    ork_w **pS=c->ssm_pS,**pD=c->ssm_pD,**pC=c->ssm_pC,**pO=c->ssm_pO;
    ork_f16 *aS=c->ssm_aS,*bS=c->ssm_bS,*aD=c->ssm_aD,*bD=c->ssm_bD,*aC=c->ssm_aC,*bC=c->ssm_bC,*aO=c->ssm_aO,*bO=c->ssm_bO;
    float *G=c->ssm_G,*Yd=c->ssm_Yd,*cs=c->ssm_cs,*tmp=c->ssm_tmp,*state=c->ssm_state,*stp=c->ssm_stp;
    double *Acs=c->ssm_Acs,*xbar=c->ssm_xbar,*Acl=c->ssm_Acl,*Aclp=c->ssm_Aclp;
    ork_mm_task_f16 *tk=c->ssm_tk;
    int ret=0;
    #define _NOW (g_ssm_prof?ork_now_us():0.0)
    #define STG(SID,pool,M,K,N,Aop,Bop,Cop) do{ double _r=_NOW; for(int h=0;h<nh;h++){ if(ork_f16_mm_repack(c,pool[h],K,N,(Bop)+(size_t)h*(size_t)(K)*(N))){ret=-1;goto done2;} \
        tk[h]=(ork_mm_task_f16){pool[h],M,(Aop)+(size_t)h*(size_t)(M)*(K),(Cop)+(size_t)h*(size_t)(M)*(N)}; } double _q=_NOW; g_ssm_repack+=_q-_r; \
        if((ork_ssm_chain()?ork_f16_mm_run_stream_chain:ork_f16_mm_run_stream)(c,nh,tk)){ret=-1;goto done2;} double _e=_NOW; g_ssm_npu+=_e-_q; g_ssm_stg[SID]+=_e-_r; }while(0)
    if(g_ssm_prof<0)g_ssm_prof=getenv("ORK_SSM_PROF")?1:0; g_ssm_calls++;
    /* BATCHED dispatch (nb=NC): one stage = one run over nh*NC matmuls; slot idx = cc*nh+h. */
    #define STGB(SID,pool,M,K,N,Aop,Bop,Cop) do{ double _r=_NOW; int _S=nh*NC; for(int s=0;s<_S;s++){ if(ork_f16_mm_repack(c,pool[s],K,N,(Bop)+(size_t)s*(size_t)(K)*(N))){ret=-1;goto done2;} \
        tk[s]=(ork_mm_task_f16){pool[s],M,(Aop)+(size_t)s*(size_t)(M)*(K),(Cop)+(size_t)s*(size_t)(M)*(N)}; } double _q=_NOW; g_ssm_repack+=_q-_r; \
        if((ork_ssm_chain()?ork_f16_mm_run_stream_chain:ork_f16_mm_run_stream)(c,_S,tk)){ret=-1;goto done2;} double _e=_NOW; g_ssm_npu+=_e-_q; g_ssm_stg[SID]+=_e-_r; }while(0)
    if(nb>1){
      for(int seq=0; seq<ns && !ret; seq++){
        double _t=_NOW;
        for(int cc=0;cc<NC;cc++){ int base=cc*CS;                               /* Phase 1: prep + operands (all chunks) */
            for(int h=0;h<nh;h++){ int SL=cc*nh+h; double run=0,ah=A[h];
                for(int l=0;l<CS;l++){ int t=base+l; double dtv=(t<nt)?ork_softplus(dt[(size_t)(seq*nt+t)*nh+h]):0.0;
                    run+=dtv*ah; Acs[(size_t)SL*CS+l]=run;
                    for(int i1=0;i1<nr;i1++) xbar[((size_t)SL*CS+l)*nr+i1]=(t<nt)?dtv*x[((size_t)(seq*nt+t)*nh+h)*nr+i1]:0.0; }
                Acl[SL]=Acs[(size_t)SL*CS+CS-1]; }
            for(int h=0;h<nh;h++){ int SL=cc*nh+h, g=h/hpg;
                for(int l=0;l<CS;l++){ int t=base+l; const float *Cc=(t<nt)?C+((size_t)(seq*nt+t)*ng+g)*nc:NULL;
                    for(int n=0;n<nc;n++){ ork_f16 v=Cc?(ork_f16)Cc[n]:(ork_f16)0.0f; aS[((size_t)SL*CS+l)*nc+n]=v; aO[((size_t)SL*CS+l)*nc+n]=v; } }
                for(int n=0;n<nc;n++)for(int sp=0;sp<CS;sp++){ int t=base+sp; bS[((size_t)SL*nc+n)*CS+sp]=(t<nt)?(ork_f16)B[((size_t)(seq*nt+t)*ng+g)*nc+n]:(ork_f16)0.0f; }
                for(int i1=0;i1<nr;i1++)for(int sp=0;sp<CS;sp++) aC[((size_t)SL*nr+i1)*CS+sp]=(ork_f16)xbar[((size_t)SL*CS+sp)*nr+i1];
                for(int l=0;l<CS;l++)for(int i1=0;i1<nr;i1++) bD[((size_t)SL*CS+l)*nr+i1]=(ork_f16)xbar[((size_t)SL*CS+l)*nr+i1]; } }
        g_ssm_stage+=_NOW-_t;
        STGB(0,pS,CS,nc,CS,aS,bS,G);
        _t=_NOW;
        for(int cc=0;cc<NC;cc++){ int base=cc*CS;                               /* Phase 2: decay-mask aD + bC (needs G) */
            for(int h=0;h<nh;h++){ int SL=cc*nh+h, g=h/hpg; const double *Ah=Acs+(size_t)SL*CS;
                for(int l=0;l<CS;l++)for(int sp=0;sp<CS;sp++){ double m=(sp<=l)?(double)G[((size_t)SL*CS+l)*CS+sp]*exp(Ah[l]-Ah[sp]):0.0; aD[((size_t)SL*CS+l)*CS+sp]=(ork_f16)m; }
                for(int sp=0;sp<CS;sp++){ int t=base+sp; double ds=exp(Ah[CS-1]-Ah[sp]); const float *Bc=(t<nt)?B+((size_t)(seq*nt+t)*ng+g)*nc:NULL;
                    for(int n=0;n<nc;n++) bC[((size_t)SL*CS+sp)*nc+n]=(ork_f16)(Bc?ds*Bc[n]:0.0); } } }
        g_ssm_stage+=_NOW-_t;
        STGB(1,pD,CS,CS,nr,aD,bD,Yd);
        STGB(2,pC,nr,CS,nc,aC,bC,cs);
        _t=_NOW;
        for(int h=0;h<nh;h++)for(size_t i=0;i<(size_t)nr*nc;i++) stp[(size_t)h*nr*nc+i]=s0[(size_t)seq*nh*nr*nc+(size_t)h*nr*nc+i];  /* Phase 3: carry (seq) -> es[cc] in stp */
        for(int cc=1;cc<NC;cc++) for(int h=0;h<nh;h++){ int SL=cc*nh+h, SLp=(cc-1)*nh+h; double dp=exp(Acl[SLp]);
            for(size_t i=0;i<(size_t)nr*nc;i++) stp[(size_t)SL*nr*nc+i]=(float)(dp*stp[(size_t)SLp*nr*nc+i]+cs[(size_t)SLp*nr*nc+i]); }
        for(int cc=0;cc<NC;cc++) for(int h=0;h<nh;h++){ int SL=cc*nh+h;          /* Phase 4: bO = es (transposed); aO=C already built */
            for(int n=0;n<nc;n++)for(int i1=0;i1<nr;i1++) bO[((size_t)SL*nc+n)*nr+i1]=(ork_f16)stp[((size_t)SL*nr+i1)*nc+n]; }
        g_ssm_stage+=_NOW-_t;
        STGB(3,pO,CS,nc,nr,aO,bO,tmp);
        _t=_NOW;
        for(int cc=0;cc<NC;cc++){ int base=cc*CS;                               /* Phase 5: post y (all chunks) */
            for(int h=0;h<nh;h++){ int SL=cc*nh+h; const double *Ah=Acs+(size_t)SL*CS;
                for(int l=0;l<CS;l++){ int t=base+l; if(t>=nt)continue; double sdo=exp(Ah[l]);
                    for(int i1=0;i1<nr;i1++) y[((size_t)(seq*nt+t)*nh+h)*nr+i1]=Yd[((size_t)SL*CS+l)*nr+i1]+tmp[((size_t)SL*CS+l)*nr+i1]*sdo; } } }
        for(int h=0;h<nh;h++){ int SLl=(NC-1)*nh+h; double dp=exp(Acl[SLl]);   /* s_new = last carry */
            for(size_t i=0;i<(size_t)nr*nc;i++) s_new[(size_t)seq*nh*nr*nc+(size_t)h*nr*nc+i]=(float)(dp*stp[(size_t)SLl*nr*nc+i]+cs[(size_t)SLl*nr*nc+i]); }
        g_ssm_post+=_NOW-_t;
      }
      #undef STGB
    } else if(ork_ssm_pipeline()){
      /* DOUBLE-BUFFERED per-chunk: helper thread marshals aC/bD/bO/bC during the pS submit. */
      for(int seq=0; seq<ns && !ret; seq++){
        for(size_t i=0;i<(size_t)nh*nr*nc;i++) state[i]=s0[(size_t)seq*nh*nr*nc+i];
        memset(stp,0,(size_t)nh*nr*nc*4); memset(Aclp,0,nh*8);
        for(int cc=0;cc<NC;cc++){ int base=cc*CS; double _t=_NOW;
            for(int h=0;h<nh;h++){ double run=0,ah=A[h];
                for(int l=0;l<CS;l++){ int t=base+l; double dtv=(t<nt)?ork_softplus(dt[(size_t)(seq*nt+t)*nh+h]):0.0;
                    run+=dtv*ah; Acs[(size_t)h*CS+l]=run;
                    for(int i1=0;i1<nr;i1++) xbar[((size_t)h*CS+l)*nr+i1]=(t<nt)?dtv*x[((size_t)(seq*nt+t)*nh+h)*nr+i1]:0.0; }
                Acl[h]=Acs[(size_t)h*CS+CS-1]; }
            if(cc>0) for(int h=0;h<nh;h++){ double dp=exp(Aclp[h]); for(size_t i=0;i<(size_t)nr*nc;i++){ size_t j=(size_t)h*nr*nc+i; state[j]=(float)(dp*state[j]+stp[j]); } }
            g_ssm_prep+=_NOW-_t; _t=_NOW;
            for(int h=0;h<nh;h++){ int g=h/hpg;                                 /* pS operands only (serial) */
                for(int l=0;l<CS;l++){ int t=base+l; const float *Cc=(t<nt)?C+((size_t)(seq*nt+t)*ng+g)*nc:NULL;
                    for(int n=0;n<nc;n++){ ork_f16 v=Cc?(ork_f16)Cc[n]:(ork_f16)0.0f; aS[((size_t)h*CS+l)*nc+n]=v; aO[((size_t)h*CS+l)*nc+n]=v; } }
                for(int n=0;n<nc;n++)for(int sp=0;sp<CS;sp++){ int t=base+sp; bS[((size_t)h*nc+n)*CS+sp]=(t<nt)?(ork_f16)B[((size_t)(seq*nt+t)*ng+g)*nc+n]:(ork_f16)0.0f; } }
            g_ssm_stage+=_NOW-_t;
            struct ssm_marshal_arg ha={aC,bD,bO,bC,xbar,Acs,state,B,nh,nr,nc,CS,ng,hpg,base,nt,seq};   /* helper: G-independent operands */
            int hstarted=ssm_helper_ensure(c);                                  /* persistent little-core helper */
            if(hstarted) ssm_helper_fire(c,&ha); else ssm_marshal_gi(&ha);
            double _r=_NOW;                                                     /* pS dispatch (inline; join helper before any error) */
            for(int h=0;h<nh;h++){ if(ork_f16_mm_repack(c,pS[h],nc,CS,bS+(size_t)h*nc*CS)){ if(hstarted)ssm_helper_join(c); ret=-1; goto done2; }
                tk[h]=(ork_mm_task_f16){pS[h],CS,aS+(size_t)h*CS*nc,G+(size_t)h*CS*CS}; }
            double _q=_NOW; g_ssm_repack+=_q-_r;
            int prc=(ork_ssm_chain()?ork_f16_mm_run_stream_chain:ork_f16_mm_run_stream)(c,nh,tk);
            double _e=_NOW; g_ssm_npu+=_e-_q; g_ssm_stg[0]+=_e-_r;
            if(hstarted)ssm_helper_join(c);
            if(prc){ret=-1;goto done2;}
            _t=_NOW;
            for(int h=0;h<nh;h++){ const double *Ah=Acs+(size_t)h*CS;           /* aD (needs G) serial */
                for(int l=0;l<CS;l++)for(int sp=0;sp<CS;sp++){ double m=(sp<=l)?(double)G[((size_t)h*CS+l)*CS+sp]*exp(Ah[l]-Ah[sp]):0.0; aD[((size_t)h*CS+l)*CS+sp]=(ork_f16)m; } }
            g_ssm_stage+=_NOW-_t;
            STG(1,pD,CS,CS,nr,aD,bD,Yd);
            STG(2,pC,nr,CS,nc,aC,bC,cs);
            STG(3,pO,CS,nc,nr,aO,bO,tmp);
            _t=_NOW;
            for(int h=0;h<nh;h++){ const double *Ah=Acs+(size_t)h*CS;
                for(int l=0;l<CS;l++){ int t=base+l; if(t>=nt)continue; double sdo=exp(Ah[l]);
                    for(int i1=0;i1<nr;i1++) y[((size_t)(seq*nt+t)*nh+h)*nr+i1]=Yd[((size_t)h*CS+l)*nr+i1]+tmp[((size_t)h*CS+l)*nr+i1]*sdo; } }
            g_ssm_post+=_NOW-_t;
            memcpy(stp,cs,(size_t)nh*nr*nc*4); memcpy(Aclp,Acl,nh*8);
        }
        for(int h=0;h<nh;h++){ double dp=exp(Aclp[h]); for(size_t i=0;i<(size_t)nr*nc;i++){ size_t j=(size_t)h*nr*nc+i; s_new[(size_t)seq*nh*nr*nc+j]=(float)(dp*state[j]+stp[j]); } }
      }
    } else {
    for(int seq=0; seq<ns && !ret; seq++){
        for(size_t i=0;i<(size_t)nh*nr*nc;i++) state[i]=s0[(size_t)seq*nh*nr*nc+i];
        memset(stp,0,(size_t)nh*nr*nc*4); memset(Aclp,0,nh*8);
        for(int cc=0;cc<NC;cc++){ int base=cc*CS; double _t=_NOW;
            for(int h=0;h<nh;h++){ double run=0,ah=A[h];
                for(int l=0;l<CS;l++){ int t=base+l; double dtv=(t<nt)?ork_softplus(dt[(size_t)(seq*nt+t)*nh+h]):0.0;
                    run+=dtv*ah; Acs[(size_t)h*CS+l]=run;
                    for(int i1=0;i1<nr;i1++) xbar[((size_t)h*CS+l)*nr+i1]=(t<nt)?dtv*x[((size_t)(seq*nt+t)*nh+h)*nr+i1]:0.0; }
                Acl[h]=Acs[(size_t)h*CS+CS-1]; }
            if(cc>0) for(int h=0;h<nh;h++){ double dp=exp(Aclp[h]); for(size_t i=0;i<(size_t)nr*nc;i++){ size_t j=(size_t)h*nr*nc+i; state[j]=(float)(dp*state[j]+stp[j]); } }
            g_ssm_prep+=_NOW-_t; _t=_NOW;
            for(int h=0;h<nh;h++){ int g=h/hpg;
                for(int l=0;l<CS;l++){ int t=base+l; const float *Cc=(t<nt)?C+((size_t)(seq*nt+t)*ng+g)*nc:NULL;
                    for(int n=0;n<nc;n++){ ork_f16 v=Cc?(ork_f16)Cc[n]:(ork_f16)0.0f; aS[((size_t)h*CS+l)*nc+n]=v; aO[((size_t)h*CS+l)*nc+n]=v; } }
                for(int n=0;n<nc;n++)for(int sp=0;sp<CS;sp++){ int t=base+sp; bS[((size_t)h*nc+n)*CS+sp]=(t<nt)?(ork_f16)B[((size_t)(seq*nt+t)*ng+g)*nc+n]:(ork_f16)0.0f; }
                for(int i1=0;i1<nr;i1++)for(int sp=0;sp<CS;sp++) aC[((size_t)h*nr+i1)*CS+sp]=(ork_f16)xbar[((size_t)h*CS+sp)*nr+i1];
                for(int l=0;l<CS;l++)for(int i1=0;i1<nr;i1++) bD[((size_t)h*CS+l)*nr+i1]=(ork_f16)xbar[((size_t)h*CS+l)*nr+i1];
                for(int n=0;n<nc;n++)for(int i1=0;i1<nr;i1++) bO[((size_t)h*nc+n)*nr+i1]=(ork_f16)state[((size_t)h*nr+i1)*nc+n]; }
            g_ssm_stage+=_NOW-_t;
            if((c->ssm_i8&1) && c->ssm_pSi8){ double _p=_NOW; if(ssm_stg_i8(c,c->ssm_pSi8,CS,nc,CS,aS,bS,G,nh)){ret=-1;goto done2;} double _pe=_NOW; g_ssm_npu+=_pe-_p; g_ssm_stg[0]+=_pe-_p; }
            else STG(0,pS,CS,nc,CS,aS,bS,G);
            _t=_NOW;
            for(int h=0;h<nh;h++){ int g=h/hpg; const double *Ah=Acs+(size_t)h*CS;
                for(int l=0;l<CS;l++)for(int sp=0;sp<CS;sp++){ double m=(sp<=l)?(double)G[((size_t)h*CS+l)*CS+sp]*exp(Ah[l]-Ah[sp]):0.0; aD[((size_t)h*CS+l)*CS+sp]=(ork_f16)m; }
                for(int sp=0;sp<CS;sp++){ int t=base+sp; double ds=exp(Ah[CS-1]-Ah[sp]); const float *Bc=(t<nt)?B+((size_t)(seq*nt+t)*ng+g)*nc:NULL;
                    for(int n=0;n<nc;n++) bC[((size_t)h*CS+sp)*nc+n]=(ork_f16)(Bc?ds*Bc[n]:0.0); } }
            g_ssm_stage+=_NOW-_t;
            STG(1,pD,CS,CS,nr,aD,bD,Yd);
            STG(2,pC,nr,CS,nc,aC,bC,cs);
            STG(3,pO,CS,nc,nr,aO,bO,tmp);
            _t=_NOW;
            for(int h=0;h<nh;h++){ const double *Ah=Acs+(size_t)h*CS;
                for(int l=0;l<CS;l++){ int t=base+l; if(t>=nt)continue; double sdo=exp(Ah[l]);
                    for(int i1=0;i1<nr;i1++) y[((size_t)(seq*nt+t)*nh+h)*nr+i1]=Yd[((size_t)h*CS+l)*nr+i1]+tmp[((size_t)h*CS+l)*nr+i1]*sdo; } }
            g_ssm_post+=_NOW-_t;
            memcpy(stp,cs,(size_t)nh*nr*nc*4); memcpy(Aclp,Acl,nh*8);
        }
        for(int h=0;h<nh;h++){ double dp=exp(Aclp[h]); for(size_t i=0;i<(size_t)nr*nc;i++){ size_t j=(size_t)h*nr*nc+i; s_new[(size_t)seq*nh*nr*nc+j]=(float)(dp*state[j]+stp[j]); } }
    }
    }
    #undef STG
    #undef _NOW
done2:
    if(ret) orki_ssm_pool_free(c);   /* on error drop the cache so the next call re-allocs clean; on success KEEP it warm */
    return ret;
}
int ork_ssd_probe_mixchain(ork_npu *c,int *mm_ok,int *silu_ok,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems, dom=c->dom_active;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    const int M=8,N=64; const double in_scale=8.0/32000.0, out_scale=1.0/32000.0;
    if(orki_silu_calibrate_idx16(c)) return -1;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)
    static double qsum[1030]; static int qn[1030];
    for(int k=0;k<1030;k++){ qsum[k]=0; qn[k]=0; }
    for(int s=0;s<SILU16_NS;s++){ int k=c->silu_idx16[s]; if(k<0||k>1029)continue; qsum[k]+=-32768.0+s*SILU16_QSTEP; qn[k]++; }
    int16_t lut[1030]; int lo=-1,hi=-1;
    for(int k=0;k<1030;k++){ if(qn[k]){ if(lo<0)lo=k; hi=k; double q_in=qsum[k]/qn[k]; double val=orki_silu_f(q_in*in_scale)/out_scale;
        long q=lround(val); if(q>32767)q=32767; if(q<-32768)q=-32768; lut[k]=(int16_t)q; } else lut[k]=0; }
    if(lo<0) return -1;
    for(int k=0;k<lo;k++)lut[k]=lut[lo]; for(int k=hi+1;k<1030;k++)lut[k]=lut[hi];
    for(int k=lo;k<=hi;k++){ if(qn[k])continue; int a=k,b=k; while(a>lo&&!qn[a])a--; while(b<hi&&!qn[b])b++;
        lut[k]=(int16_t)(lut[a]+(lut[b]-lut[a])*(k-a)/(b-a)); }
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,dom), O=orki_bcreate(fd,sz,0x403,dom);
    struct buf Lrc=orki_bcreate(fd,(size_t)REGCMD_SILU_LUT_N*4,0x403,dom), Lsc=orki_bcreate(fd,4096,0x403,dom);
    struct buf Wd=orki_bcreate(fd,32*32*2,0x403,dom), Ad=orki_bcreate(fd,32*2,0x403,dom), Cd=orki_bcreate(fd,32*4,0x403,dom); /* fp16 A/W, fp32 C */
    int ret=-1; int16_t *inb=malloc((size_t)M*N*2);
    if(!A.cpu||!O.cpu||!Lrc.cpu||!Lsc.cpu||!Wd.cpu||!Ad.cpu||!Cd.cpu||!inb){ goto mfail; }
    memset(A.cpu,0,sz); memset(O.cpu,0,sz); memset(Cd.cpu,0,32*4);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int16_t v=(int16_t)((m*N+n)%20000-8000); inb[m*N+n]=v; *(int16_t*)((char*)A.cpu+EWCUBEH(m,n))=v; }
    { uint16_t*wd=Wd.cpu,*ad=Ad.cpu; for(int i=0;i<32*32;i++)wd[i]=0x3c00; for(int i=0;i<32;i++)ad[i]=0x3c00; } /* fp16 1.0 -> C=32 */
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&Wd,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&Ad,RKNPU_MEM_SYNC_TO_DEVICE);
    memcpy(Lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
    orki_setrn((uint32_t*)Lrc.cpu,REGCMD_SILU_LUT_N,RK_DPU_DST_BASE_ADDR,(uint32_t)Lsc.dma);
    { uint32_t*lr=(uint32_t*)Lrc.cpu; int j=0; for(int k=0;k+1<REGCMD_SILU_LUT_N;k+=2){ if((lr[k]&0xffff)==0x4104){ int32_t v=(j<1030)?(int32_t)lut[j]:0; j++;
        lr[k]=0x4104|((uint32_t)(v&0xffff)<<16); lr[k+1]=(0x1001u<<16)|(((uint32_t)v>>16)&0xffff); } } }
    orki_bsync(fd,&Lrc,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task*t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=1097; t->regcmd_addr=Lrc.dma;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit s;memset(&s,0,sizeof s);s.flags=0x1;s.task_number=1;s.task_obj_addr=c->task.obj;s.core_mask=RKNPU_CORE0_MASK;s.fence_fd=-1;s.timeout=orki_ew_timeout_ms();s.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      if(orki_rknpu_submit_ioctl(fd,&s,dom)) goto mfail; }
    /* chain: [0]=FP16 matmul (synth) -> [1]=int16 silu */
    { uint32_t *mm=(uint32_t*)c->regcmd.cpu, *si=(uint32_t*)((char*)c->regcmd.cpu+(size_t)REGCMD_I8_N*4);
      memset(mm,0,REGCMD_I8_N*4);
      orki_f16_synth(mm,1,32,32,(uint32_t)Ad.dma,(uint32_t)Wd.dma,(uint32_t)Cd.dma,0,CBUF);   /* FP16 matmul task0 (sched=0: K=32<96 small-K 0x1040 fix) */
      uint64_t nx=c->regcmd.dma+(size_t)REGCMD_I8_N*4;
      mm[216]=0x0010|((nx&0xffff)<<16); mm[217]=(0x0101u<<16)|((nx>>16)&0xffff);
      mm[218]=0x0014|(((69+3)/2)<<16);  mm[219]=(0x0101u<<16)|0;
      memcpy(si,REGCMD_SILU_STD_I16,(size_t)REGCMD_SILU_STD_I16_N*4);
      orki_set_mul_geom(si,REGCMD_SILU_STD_I16_N,M,N);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_SDP_5040,0); orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_SDP_5038,0);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma); orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_SDP_5018,(uint32_t)A.dma);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_SCALE,0x4000u); orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_SHIFT,14u); orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_OFFSET,0u);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_R4110,ORK_SILU16_IDXOFF); orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_BN_ALU_CFG,ORK_SILU16_C4064); orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_BN_MUL_CFG,ORK_SILU16_C4068);
      orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
      struct rknpu_task*tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,2*sizeof *tk);
      tk[0].enable_mask=0xd;  tk[0].int_mask=0x300; tk[0].int_clear=0x1ffff; tk[0].regcfg_amount=108; tk[0].regcmd_addr=c->regcmd.dma;
      tk[1].enable_mask=0x18; tk[1].int_mask=0x300; tk[1].int_clear=0x1ffff; tk[1].regcfg_amount=69;  tk[1].regcmd_addr=nx;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit s;memset(&s,0,sizeof s);s.flags=0x1;s.task_number=2;s.task_obj_addr=c->task.obj;s.core_mask=RKNPU_CORE0_MASK;s.fence_fd=-1;s.timeout=orki_ew_timeout_ms();s.subcore_task[0]=(struct rknpu_subcore_task){0,2};
      double t0=ork_now_us();
      for(int rep=0;rep<2;rep++){ if(orki_rknpu_submit_ioctl(fd,&s,dom)) goto mfail;   /* rep0 primes fresh buffers */
          orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd,&Cd,RKNPU_MEM_SYNC_FROM_DEVICE); }
      if(us)*us=ork_now_us()-t0; }
    { float *cd=Cd.cpu; int ok=1; for(int i=0;i<32;i++) if(fabs(cd[i]-32.0)>0.5) ok=0; if(mm_ok)*mm_ok=ok; }   /* fp16 matmul: C=32 */
    { int ok=1,bad=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double ref=orki_silu_f(inb[m*N+n]*in_scale)/out_scale;
        double got=(double)*(int16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(fabs(got-ref)>0.03*fabs(ref)+3) bad++; }
      ok=(bad<=(M*N)/20); if(silu_ok)*silu_ok=ok; }
    ret=0;
mfail:
    free(inb); orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);orki_bdestroy(fd,&Wd);orki_bdestroy(fd,&Ad);orki_bdestroy(fd,&Cd);
    #undef EWCUBEH
    return ret;
}

int ork_ssd_fused_scan_bench(ork_npu *c,int H,int P,int Nst,int G,int CS,int NC,int iters,int dtype,int perhead,
                             double *fused_us,double *persub_us,int *ok_out){
    if(!c) return -1; if(iters<1) iters=1;
    int HG=H/G; if(HG<1)HG=1;
    int fd=c->fd, CBUF=c->soc->cbuf_elems, dom=-1;
    int f16 = (dtype==DT_F16);
    int esz = f16 ? 2 : 1;                 /* A/B element size (fp16=2B, int8=1B); C is 4B either way */
    int gb=G*NC;
    /* per-stage (nb, M, K, N). scores/cstate/Y_off group-batched (fp16-stable). Y_diag: perhead=1 uses
     * the fp16-STABLE bounded per-head form (nb=H*NC, [CS,CS]x[CS,P]); perhead=0 uses the fast group-
     * batched L-factored form (nb=G*NC, [CS,CS]x[CS,HG*P]) — faster but OVERFLOWS fp16 at large chunk decay. */
    int snb[4]={gb, gb, gb, perhead?H*NC:gb};
    int sM[4] ={CS, HG*P, HG*P, CS};
    int sK[4] ={Nst, CS,  Nst,  CS};
    int sN[4] ={CS, Nst,  CS,   perhead?P:HG*P};
    /* TILE cap: a single synth program mis-writes some dims at width==1024 (power-of-2 ISA quirk in the raw
     * synth path — run_i8 avoids it via its own tiling). Cap per-program M and N at TILE, M/N-tiled programs. */
    int TILE=512;
    int np=0; for(int s=0;s<4;s++){ int nm=(sM[s]+TILE-1)/TILE, nn=(sN[s]+TILE-1)/TILE; np+=nm*nn*snb[s]; }
    if(np<1||np>1024) return -2;
    int *tM=malloc(np*sizeof(int)),*tK=malloc(np*sizeof(int)),*tN=malloc(np*sizeof(int));
    size_t *cOff=malloc(np*sizeof(size_t));
    if(!tM||!tK||!tN||!cOff){ free(tM);free(tK);free(tN);free(cOff); return -3; }
    size_t maxA=0,maxB=0,totC=0; int t=0;
    for(int s=0;s<4;s++) for(int b=0;b<snb[s];b++){
        int fullM=sM[s], fullN=sN[s];
        for(int m0=0;m0<fullM;m0+=TILE) for(int n0=0;n0<fullN;n0+=TILE){
            int Mc=(fullM-m0<TILE)?(fullM-m0):TILE, Nc=(fullN-n0<TILE)?(fullN-n0):TILE;
            tM[t]=Mc; tK[t]=sK[s]; tN[t]=Nc;
            cOff[t]=totC; totC+=(size_t)Mc*Nc;                 /* each tile -> its own dense [Mc,Nc] region */
            size_t a=(size_t)Mc*sK[s], bb=(size_t)sK[s]*Nc;
            if(a>maxA)maxA=a; if(bb>maxB)maxB=bb; t++;
        }
    }
    struct buf Ab=orki_bcreate(fd,maxA*esz,0x403,dom), Bb=orki_bcreate(fd,maxB*esz,0x403,dom), Cb=orki_bcreate(fd,totC*4,0x403,dom);
    uint32_t *rcs=malloc((size_t)np*REGCMD_I8_N*4);
    ork_chain_prog *progs=malloc(np*sizeof(ork_chain_prog));
    int ret=0;
    if(!Ab.cpu||!Bb.cpu||!Cb.cpu||!rcs||!progs){ ret=-3; goto done; }
    if(f16){ uint16_t*pa=Ab.cpu,*pb=Bb.cpu; for(size_t i=0;i<maxA;i++)pa[i]=0x3c00; for(size_t i=0;i<maxB;i++)pb[i]=0x3c00; }  /* fp16 1.0 */
    else   { memset(Ab.cpu,1,maxA); memset(Bb.cpu,1,maxB); }
    memset(Cb.cpu,0,totC*4);
    orki_bsync(fd,&Ab,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Bb,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Cb,RKNPU_MEM_SYNC_TO_DEVICE);
    for(int i=0;i<np;i++){
        uint32_t *rc=rcs+(size_t)i*REGCMD_I8_N;
        uint32_t aC=(uint32_t)(Cb.dma+cOff[i]*4);
        if(f16) orki_f16_synth   (rc,tM[i],tK[i],tN[i],(uint32_t)Ab.dma,(uint32_t)Bb.dma,aC,1,CBUF);      /* fp16, dense [M,Nc] out */
        else    orki_i8_synth(rc,tM[i],tK[i],tN[i],(uint32_t)Ab.dma,(uint32_t)Bb.dma,aC,1,CBUF,0);    /* int8, dense [M,Nc] out */
        progs[i]=(ork_chain_prog){rc,REGCMD_I8_N,0xd,108,216};
    }
    /* fp16 needs the NPU in fp16 mode: force a reset on entry (the int8-oriented chain assembler keeps
     * warm across ORK_I8_LIVE markers, so an int8->fp16 switch would otherwise skip the reset). */
    if(f16){ orki_act(fd,RKNPU_ACT_RESET,0); c->warmed=0; c->last_dt=DT_F16; }
    /* FUSED: one chained submit of all np matmul programs */
    int rc1=ork_npu_chain_progs(c,np,progs,dom);   /* warm + wedge-check */
    if(rc1){ ret=rc1; goto done; }
    { double f0=ork_now_us(); for(int it=0;it<iters;it++) ork_npu_chain_progs(c,np,progs,dom); *fused_us=(ork_now_us()-f0)/iters; }
    orki_bsync(fd,&Cb,RKNPU_MEM_SYNC_FROM_DEVICE);
    { int okc=1; int32_t*Ci=(int32_t*)Cb.cpu; float*Cf=(float*)Cb.cpu;
      for(int i=0;i<np&&okc;i++){ size_t mn=(size_t)tM[i]*tN[i];
        for(size_t e=0;e<mn;e++){ double got = f16 ? (double)Cf[cOff[i]+e] : (double)Ci[cOff[i]+e];
          if(got!=(double)tK[i]){ if(getenv("ORK_SSD_DBG")) fprintf(stderr,"[ssd_fused] mismatch prog %d/%d M=%d K=%d N=%d elem %zu: got %g exp %d\n",i,np,tM[i],tK[i],tN[i],e,got,tK[i]); okc=0;break; } } }
      if(ok_out)*ok_out=okc; }
    /* PER-SUBMIT: the SAME programs as np separate single-task submits (each pays the floor) */
    { double p0=ork_now_us(); for(int it=0;it<iters;it++) for(int i=0;i<np;i++) ork_npu_chain_progs(c,1,&progs[i],dom); *persub_us=(ork_now_us()-p0)/iters; }
done:
    orki_bdestroy(fd,&Ab);orki_bdestroy(fd,&Bb);orki_bdestroy(fd,&Cb);
    free(rcs);free(progs);free(tM);free(tK);free(tN);free(cOff);
    return ret;
}
