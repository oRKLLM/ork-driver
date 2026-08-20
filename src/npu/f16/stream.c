/* npu/f16/stream.c — round-robin and chained-multicore fp16 streams, colsplit, batched GEMM.
 *
 * Part of the f16 datapath; shared declarations in npu/f16/f16.h. Split out of npu/f16.c for the
 * same reason i8 is a folder: one datapath, sized for reading. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include "ork_regs.h"
#include "regcmd_array_4x32x16.h"
#include "regcmd_i8.h"
#include "regcmd_silu.h"
#include "regcmd_ewmul.h"
#include "regcmd_softmax_f16.h"
#include "regcmd_softmax_wt.h"
#include "regcmd_reshape.h"
#include "npu/internal.h"
#include <fcntl.h>
#include "npu/core.h"
#include "npu/f16/f16.h"

int ork_f16_colsplit(void){ static int v=-1; if(v<0){const char*e=getenv("ORK_F16_COLSPLIT"); v=e?atoi(e):1; } return v; }   /* colsplit is the ONLY fp16 multicore path (#45); ORK_F16_COLSPLIT=0 -> single-core fp16 reference (never mcworker) */

void *ork_pcfd_thread(void *vp){
    struct ork_pcfd_arg *a=vp; struct rknpu_submit sub; memset(&sub,0,sizeof sub);
    sub.flags=ork_ppflags(); sub.task_number=1; sub.task_obj_addr=a->tk->obj; sub.fence_fd=-1;   /* BLOCKING per-core submit */
    sub.core_mask=1u<<a->core;
    sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,1};
    sub.timeout=orki_mm_timeout_ms();
    ork_kmsg("PCFD core=%d submit START fd=%d task_obj=0x%llx", a->core, a->fd, (unsigned long long)a->tk->obj);
    a->rc = orki_rknpu_submit_ioctl(a->fd,&sub,0);   /* buffers all live in domain 0 */
    ork_kmsg("PCFD core=%d submit DONE rc=%d", a->core, a->rc);
    return NULL;
}

int ork_bmm_fp16_fused(ork_npu*c,int nb,int M,int K,int N,const f16*A,const f16*B,float*C){
    if(!c||nb<1||nb>64||M<1||K<1||N<1||K%32||N%16) return -2;
    /* B3 (chain_progs retirement): the batch of nb independent fp16 matmuls now rides the NONBLOCK-doorbell
     * fp16 PC-chain (ork_mm_run_stream_f16_chain) instead of the legacy ork_npu_chain_progs. The stream primitive
     * owns staging + warm/reset (enters DT_F16/XP_STREAM_F16 — fp16 content tracked AS fp16, unifying this with
     * the other fp16 scan stages; the old path entered DT_I8_CHAIN, whose "genuine int8->fp16 switch needs a
     * reset" caveat this removes). Spreads the batch across cores rather than one core-0 chain. Bit-exact
     * (test_bmm_fused: fused == per-op ork_bmm_fp16 == CPU). chain_progs stays for the tools/ RE probes. */
    ork_w **w=calloc(nb,sizeof(ork_w*));
    ork_mm_task_f16 *tk=malloc((size_t)nb*sizeof(ork_mm_task_f16));
    if(!w||!tk){ free(w); free(tk); return -3; }
    int ret=0;
    for(int b=0;b<nb;b++){
        w[b]=ork_mm_pack(c,K,N,B+(size_t)b*K*N);
        if(!w[b]||w[b]->Sk!=1||w[b]->Sn!=1){ ret=-3; goto done3; }
        tk[b]=(ork_mm_task_f16){w[b],M,A+(size_t)b*M*K,C+(size_t)b*M*N};
    }
    ret=ork_mm_run_stream_f16_chain(c,nb,tk);
done3:
    for(int b=0;b<nb;b++) if(w[b]) ork_mm_free(c,w[b]);
    free(w); free(tk);
    return ret;
}

static void *stream_worker_f16(void *vp){
    struct streamw_f16 *a=vp; ork_npu *c=a->c; int fd=c->fd, i=a->core, CBUF=c->soc->cbuf_elems;
    if(CBUF>32768) CBUF=32768;                     /* fp16 M-scheduler validated to the 32768-tile */
    orki_pin_big_core(i);
    int k; a->rc=0;
    uint32_t rc[REGCMD_I8_N];
    while((k=__atomic_fetch_add(a->ctr,1,__ATOMIC_SEQ_CST))<a->S){
        const ork_mm_task_f16 *t=&a->tasks[k]; ork_w *w=t->w; int M=t->M, K=w->K, N=w->N;
        int sched=(K&(K-1))==0 && K>=128 && K<2048;
        memcpy(c->maf[i].cpu, t->A, (size_t)M*K*2); orki_bsync(fd,&c->maf[i],RKNPU_MEM_SYNC_TO_DEVICE);
        memset(rc,0,REGCMD_I8_N*4);
        orki_synth(rc, M, K, N, (uint32_t)c->maf[i].dma, (uint32_t)w->Bb[0].dma, (uint32_t)c->mcc[i].dma, sched, CBUF);
        memcpy(c->mrc[i].cpu, rc, REGCMD_I8_N*4); orki_bsync(fd,&c->mrc[i],RKNPU_MEM_SYNC_TO_DEVICE);
        struct rknpu_task *mt=c->mtk[i].cpu; memset(mt,0,sizeof *mt);
        mt[0].enable_mask=0xd; mt[0].int_mask=0x300; mt[0].int_clear=0x1ffff; mt[0].regcfg_amount=108; mt[0].regcmd_addr=c->mrc[i].dma;
        orki_bsync(fd,&c->mtk[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        struct rknpu_submit sub; memset(&sub,0,sizeof sub);
        sub.flags=ork_ppflags(); sub.task_number=1; sub.task_obj_addr=c->mtk[i].obj; sub.core_mask=1u<<i; sub.fence_fd=-1;
        sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,1};
        sub.timeout=orki_mm_timeout_ms();
        int reps=c->mwarm[i]?1:2;
        for(int rep=0;rep<reps;rep++){ if(orki_rknpu_submit_ioctl(fd,&sub,w->domain)){ if(rep==reps-1)a->rc=-1; continue; } orki_bsync(fd,&c->mcc[i],RKNPU_MEM_SYNC_FROM_DEVICE); }
        c->mwarm[i]=1;
        memcpy(t->C, c->mcc[i].cpu, (size_t)M*N*4);
    }
    return NULL;
}

/* ---- fp16 ROUND-ROBIN STREAM (ork_mm_run_stream_f16) — fp16 twin of the int8 stream above ----
 * Dynamic·dynamic (both operands activations): weight is pre-packed per task (ork_w, fp16 Bb tiled), A/C
 * copied via per-core staging. Each worker pulls the next task and runs a SINGLE-CORE submit on its own
 * core (core_mask=1<<i) — so nbatch independent matmuls spread across all cores. Single M-tile (the SSD
 * scan is M<=64 <= one tile); K<96 uses sched=0 (the small-K 0x1040 fix). */
int ork_mm_run_stream_f16(ork_npu *c, int S, const ork_mm_task_f16 *tasks){
    if(!c||S<1||!tasks) return -2;
    if(tasks[0].w && (tasks[0].w->domain!=c->dom_active || (tasks[0].w->domain!=0 && !c->dom_save))) orki_dom_activate(c,tasks[0].w->domain);
    for(int i=0;i<S;i++){ ork_w *w=tasks[i].w;
        if(!w||w->dtype!=DT_F16||tasks[i].M<=0) return -2;
        if(w->Sn!=1||w->Sk!=1||!w->Bb) return -2;              /* single-slice fp16 (K<=ks,N<=nmax) */
        if(w->K%32||w->N%16) return -2; }
    /* P3 SPINE MIGRATION: fp16 stream onto the NONBLOCK doorbell (ork_dyn_begin_mc), like run_stream_i8. The
     * doorbell fp16 path accepts single-slice small-K (K%32) shapes (uses Bb + the K-dependent sched); A stays
     * host (the fp16 doorbell stages A into maf). Build the neutral ork_mm_task_i8 view — w carries dtype=DT_F16,
     * A/C byte-reinterpreted (f16 A, fp32 C). fp16 already full-surface seeds + M<=64 always-cleans (interleave-
     * safe), and rknpu_submit_ioctl retries a transient submit-rejection. No legacy fallback (miss => -1). */
    if(S>1024) return -2;
    ork_mm_task_i8 ti[1024];
    for(int i=0;i<S;i++) ti[i]=(ork_mm_task_i8){ tasks[i].w, tasks[i].M, (const int8_t*)tasks[i].A, (int32_t*)tasks[i].C };
    int nc=orki_budget(c,2); if(nc>ORK_MAXCORE)nc=ORK_MAXCORE; if(nc>S)nc=S; if(nc<1)nc=1;
    ork_dyn_chain *h=ork_dyn_begin_mc(c,S,ti,nc);
    if(!h) return -1;
    int d=ork_dyn_end(h);
    return (d==S-1)?0:-1;
}

static void *stream_worker_f16ch(void *vp){
    struct streamw_f16ch *a=vp; ork_npu *c=a->c; int fd=c->fd, i=a->core, ncore=a->ncore, S=a->S, CBUF=c->soc->cbuf_elems;
    if(CBUF>32768) CBUF=32768;
    orki_pin_big_core(i);
    a->rc=0;
    int cnt=0; for(int k=i;k<S;k+=ncore) cnt++;
    if(cnt==0) return NULL;
    uint32_t rc[REGCMD_I8_N];
    struct rknpu_task *mt=c->mtk[i].cpu;
    int p=0;
    for(int k=i;k<S;k+=ncore,p++){
        const ork_mm_task_f16 *t=&a->tasks[k]; ork_w *w=t->w; int M=t->M, K=w->K, N=w->N;
        int sched=(K&(K-1))==0 && K>=128 && K<2048;
        memcpy((char*)c->maf[i].cpu + (size_t)p*M*K*2, t->A, (size_t)M*K*2);
        memset(rc,0,REGCMD_I8_N*4);
        orki_synth(rc, M, K, N, (uint32_t)(c->maf[i].dma + (size_t)p*M*K*2), (uint32_t)w->Bb[0].dma,
              (uint32_t)(c->mcc[i].dma + (size_t)p*M*N*4), sched, CBUF);
        if(p<cnt-1){ uint64_t next=c->mrc[i].dma + (size_t)(p+1)*REGCMD_I8_N*4;   /* PC-chain to next program */
            rc[216]=0x0010|((next&0xffff)<<16); rc[217]=(0x0101u<<16)|((uint32_t)(next>>16)&0xffff); rc[218]=0x0014|(0x0037u<<16); }
        memcpy((char*)c->mrc[i].cpu + (size_t)p*REGCMD_I8_N*4, rc, REGCMD_I8_N*4);
        memset(&mt[p],0,sizeof mt[p]); mt[p].enable_mask=0xd; mt[p].int_mask=0x300; mt[p].int_clear=0x1ffff;
        mt[p].regcfg_amount=108; mt[p].regcmd_addr=c->mrc[i].dma + (size_t)p*REGCMD_I8_N*4;
    }
    orki_bsync(fd,&c->maf[i],RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&c->mrc[i],RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&c->mtk[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub; memset(&sub,0,sizeof sub);
    sub.flags=ork_ppflags(); sub.task_number=cnt; sub.task_obj_addr=c->mtk[i].obj; sub.core_mask=1u<<i; sub.fence_fd=-1;
    sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)cnt};
    sub.timeout=orki_mm_timeout_ms();
    int reps=c->mwarm[i]?1:2;
    for(int rep=0;rep<reps;rep++){ if(orki_rknpu_submit_ioctl(fd,&sub,a->tasks[i].w->domain)){ if(rep==reps-1)a->rc=-1; continue; } orki_bsync(fd,&c->mcc[i],RKNPU_MEM_SYNC_FROM_DEVICE); }
    c->mwarm[i]=1;
    p=0; for(int k=i;k<S;k+=ncore,p++){ const ork_mm_task_f16 *t=&a->tasks[k]; int M=t->M,N=t->w->N;
        memcpy(t->C, (char*)c->mcc[i].cpu + (size_t)p*M*N*4, (size_t)M*N*4); }
    return NULL;
}

/* ---- CHAINED-MULTICORE fp16 stream (ork_mm_run_stream_f16_chain) ----
 * Combines the two half-wins: PC-chaining (task_number>1, one submit amortizes the ~48us submit floor over
 * many programs — like run_chain_i8) AND 3-core parallelism (like run_stream_f16). Static strided partition:
 * core i owns tasks {i, i+nc, ...}; it synths all of them into ITS mrc[i] (each program's PC next-descriptor
 * at word 216 -> 0x0010/0x0014 links to the next), builds a cnt-entry task-descriptor array in mtk[i], and
 * issues ONE task_number=cnt submit on core i. This is the fused graph the scan wanted: N submits -> nc.
 * Matmul-only chain (register-config, no LUT) -> ping-pong safe. Escapes the per-matmul submit floor. */
int ork_mm_run_stream_f16_chain(ork_npu *c, int S, const ork_mm_task_f16 *tasks){
    if(!c||S<1||!tasks) return -2;
    if(tasks[0].w && (tasks[0].w->domain!=c->dom_active || (tasks[0].w->domain!=0 && !c->dom_save))) orki_dom_activate(c,tasks[0].w->domain);
    for(int i=0;i<S;i++){ ork_w *w=tasks[i].w;
        if(!w||w->dtype!=DT_F16||tasks[i].M<=0) return -2;
        if(w->Sn!=1||w->Sk!=1||!w->Bb) return -2;
        if(w->K%32||w->N%16) return -2; }
    int fd=c->fd;
    ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_SW);
    int nc=orki_budget(c,2); if(nc>ORK_MAXCORE)nc=ORK_MAXCORE; if(nc>S)nc=S; if(nc<1)nc=1;
    if(orki_mc_ensure(c,nc)) return -1;
    int per=(S+nc-1)/nc;                                   /* max programs a single core owns */
    size_t needrc=(size_t)per*REGCMD_I8_N*4, needtk=(size_t)per*sizeof(struct rknpu_task);
    size_t maxMK=(size_t)per*tasks[0].M*tasks[0].w->K*2, maxMN4=(size_t)per*tasks[0].M*tasks[0].w->N*4;
    for(int i=0;i<nc;i++){
        if(c->mrc[i].size<needrc){ orki_bdestroy(fd,&c->mrc[i]); c->mrc[i]=orki_bcreate(fd,needrc,0x403,c->dom_active); if(!c->mrc[i].cpu)return -1; c->mwarm[i]=0; }
        if(c->mtk[i].size<needtk){ orki_bdestroy(fd,&c->mtk[i]); c->mtk[i]=orki_bcreate(fd,needtk,0x40b,c->dom_active); if(!c->mtk[i].cpu)return -1; }
        if(c->maf[i].size<maxMK){ orki_bdestroy(fd,&c->maf[i]); c->maf[i]=orki_bcreate(fd,maxMK,0x403,c->dom_active); if(!c->maf[i].cpu)return -1; }
        if(c->mccsz[i]<maxMN4){ orki_bdestroy(fd,&c->mcc[i]); c->mcc[i]=orki_bcreate(fd,maxMN4,0x403,c->dom_active); c->mccsz[i]=maxMN4; if(!c->mcc[i].cpu)return -1; c->mwarm[i]=0; } }
    int rc=0; orki_npu_pool_ensure(c);
    struct streamw_f16ch sw[ORK_MAXCORE];
    for(int i=0;i<nc;i++) sw[i]=(struct streamw_f16ch){c,i,nc,S,tasks,0};
    pthread_mutex_lock(&c->pmu);
    c->pjob=sw; c->pjob_nc=nc; c->pjob_fn=stream_worker_f16ch; c->pjob_stride=sizeof(struct streamw_f16ch);
    c->pdone=0; c->pgen++; pthread_cond_broadcast(&c->pgo);
    pthread_mutex_unlock(&c->pmu);
    stream_worker_f16ch(&sw[0]);
    pthread_mutex_lock(&c->pmu); while(c->pdone<nc-1) pthread_cond_wait(&c->pdn,&c->pmu); pthread_mutex_unlock(&c->pmu);
    for(int i=0;i<nc;i++) if(sw[i].rc) rc=-1;
    c->warmed=1;
    return rc;
}

int ork_bmm_fp16_stream(ork_npu*c,int nb,int M,int K,int N,const f16*A,const f16*B,float*C){
    if(!c||nb<1||M<1||K<1||N<1||K%32||N%16) return -2;
    ork_w **w=calloc(nb,sizeof(ork_w*)); ork_mm_task_f16 *tk=malloc((size_t)nb*sizeof(ork_mm_task_f16));
    if(!w||!tk){ free(w);free(tk); return -3; }
    int ret=0;
    for(int b=0;b<nb;b++){ w[b]=ork_mm_pack(c,K,N,B+(size_t)b*K*N);
        if(!w[b]||w[b]->Sk!=1||w[b]->Sn!=1){ ret=-3; goto done; }
        tk[b]=(ork_mm_task_f16){w[b],M,A+(size_t)b*M*K,C+(size_t)b*M*N}; }
    ret=ork_mm_run_stream_f16(c,nb,tk);
done:
    for(int b=0;b<nb;b++) if(w[b]) ork_mm_free(c,w[b]);
    free(w);free(tk);
    return ret;
}

int ork_bmm_fp16_strided(ork_npu *c, int nbatch, int M, int K, int N,
                         const f16 *A, const f16 *B, float *C, const ork_bmm_strides *s){
    if(!c||!A||!B||!C||!s) return -1;
    if(nbatch<1||M<1||K<1||N<1) return -2;
    if(K%32||N%16) return -2;
    f16 *Ac=malloc((size_t)M*K*sizeof(f16)), *Bc=malloc((size_t)K*N*sizeof(f16));
    int cdense=orki_bmm_c_dense(s,N); float *Cc = cdense?NULL:malloc((size_t)M*N*sizeof(float));
    if(!Ac||!Bc||(!cdense&&!Cc)){ free(Ac);free(Bc);free(Cc); return -3; }
    int rc=0;
    for(int b=0;b<nbatch;b++){
        orki_bmm_gather_f16(Bc,B+(long)b*s->bbs,K,N,s->bs_k,s->bs_n);
        orki_bmm_gather_f16(Ac,A+(long)b*s->abs,M,K,s->as_m,s->as_k);
        ork_w *w=ork_mm_pack(c,K,N,Bc); if(!w){ rc=-3; break; }
        float *Cout = cdense ? C+(long)b*s->cbs : Cc;
        int r=ork_mm_run(c,w,M,Ac,Cout);
        ork_mm_free(c,w);
        if(r){ rc=-5; break; }
        if(!cdense) orki_bmm_scatter_i32((int32_t*)(C+(long)b*s->cbs),(const int32_t*)Cc,M,N,s->cs_m,s->cs_n);
    }
    free(Ac);free(Bc);free(Cc);
    return rc;
}

int ork_bmm_fp16(ork_npu *c, int nbatch, int M, int K, int N,
                 const f16 *A, const f16 *B, float *C){
    ork_bmm_strides s=orki_bmm_natural(M,K,N); return ork_bmm_fp16_strided(c,nbatch,M,K,N,A,B,C,&s);
}
