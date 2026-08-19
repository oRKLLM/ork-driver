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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include "npu/internal.h"

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
        if(ork_mm_repack_i8(c,pool[h],K,N,bi8)){ if(getenv("ORK_SSM_DBG"))fprintf(stderr,"[ssm i8] repack_i8 h=%d rc\n",h); ret=-1; goto done; }
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
        for(int h=0;h<nh;h++){ int rr=ork_mm_run_i8(c,pool[h],M,ai8+(size_t)h*M*K,ci32+(size_t)h*M*N);
            if(rr){ if(getenv("ORK_SSM_DBG"))fprintf(stderr,"[ssm i8] run_i8 h=%d rc=%d\n",h,rr); ret=-1; goto done; } }
    } else {
        int rr=ork_mm_run_stream_i8_sk(c,nh,c->ssm_tki8);
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
    for(int h=0;h<NS;h++){ c->ssm_pS[h]=ork_mm_f16_scratch(c,nc,CS); c->ssm_pD[h]=ork_mm_f16_scratch(c,CS,nr); c->ssm_pC[h]=ork_mm_f16_scratch(c,CS,nc); c->ssm_pO[h]=ork_mm_f16_scratch(c,nc,nr);
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
        for(int h=0;h<nh;h++){ c->ssm_pSi8[h]=ork_mm_pack_i8(c,nc,CS,zero); if(!c->ssm_pSi8[h]){ free(zero); c->ssm_nh=nh; orki_ssm_pool_free(c); return -1; } }
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
    #define STG(SID,pool,M,K,N,Aop,Bop,Cop) do{ double _r=_NOW; for(int h=0;h<nh;h++){ if(ork_mm_repack_f16(c,pool[h],K,N,(Bop)+(size_t)h*(size_t)(K)*(N))){ret=-1;goto done2;} \
        tk[h]=(ork_mm_task_f16){pool[h],M,(Aop)+(size_t)h*(size_t)(M)*(K),(Cop)+(size_t)h*(size_t)(M)*(N)}; } double _q=_NOW; g_ssm_repack+=_q-_r; \
        if((ork_ssm_chain()?ork_mm_run_stream_f16_chain:ork_mm_run_stream_f16)(c,nh,tk)){ret=-1;goto done2;} double _e=_NOW; g_ssm_npu+=_e-_q; g_ssm_stg[SID]+=_e-_r; }while(0)
    if(g_ssm_prof<0)g_ssm_prof=getenv("ORK_SSM_PROF")?1:0; g_ssm_calls++;
    /* BATCHED dispatch (nb=NC): one stage = one run over nh*NC matmuls; slot idx = cc*nh+h. */
    #define STGB(SID,pool,M,K,N,Aop,Bop,Cop) do{ double _r=_NOW; int _S=nh*NC; for(int s=0;s<_S;s++){ if(ork_mm_repack_f16(c,pool[s],K,N,(Bop)+(size_t)s*(size_t)(K)*(N))){ret=-1;goto done2;} \
        tk[s]=(ork_mm_task_f16){pool[s],M,(Aop)+(size_t)s*(size_t)(M)*(K),(Cop)+(size_t)s*(size_t)(M)*(N)}; } double _q=_NOW; g_ssm_repack+=_q-_r; \
        if((ork_ssm_chain()?ork_mm_run_stream_f16_chain:ork_mm_run_stream_f16)(c,_S,tk)){ret=-1;goto done2;} double _e=_NOW; g_ssm_npu+=_e-_q; g_ssm_stg[SID]+=_e-_r; }while(0)
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
            for(int h=0;h<nh;h++){ if(ork_mm_repack_f16(c,pS[h],nc,CS,bS+(size_t)h*nc*CS)){ if(hstarted)ssm_helper_join(c); ret=-1; goto done2; }
                tk[h]=(ork_mm_task_f16){pS[h],CS,aS+(size_t)h*CS*nc,G+(size_t)h*CS*CS}; }
            double _q=_NOW; g_ssm_repack+=_q-_r;
            int prc=(ork_ssm_chain()?ork_mm_run_stream_f16_chain:ork_mm_run_stream_f16)(c,nh,tk);
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
