/* npu/core/device.c — context lifecycle: init/free, reset & recovery, SoC introspection, domain setters.
 * Part of the dtype-agnostic substrate; interface in npu/core.h. Lifted verbatim from npu.c by the
 * precision split (MODULARIZE_PLAN.md round 1). */
#define _GNU_SOURCE   /* CPU_SET/pthread_setaffinity_np, as npu.c does */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include "npu/internal.h"
#include "npu/core.h"
#include "npu/core/core.h"
void ork_ssm_prof_dump(void);
void ork_ssm_helper_stop(ork_npu *c);

void ork_npu_set_ndomains(ork_npu *c, int n){ if(c && n>1) orki_dom_reserve(c, n); }

/* #54 REAP-AT-DOMAIN-BOUNDARY. A genuine int4 doorbell DROP leaves a stuck job in c->dom_active whose completion
 * IRQ never fired, so the kernel's iommu_domain_refcount for that domain stays >0. The reap that clears it —
 * rknpu_job_timeout_clean — only fires at the TOP of the NEXT submit ON THAT CORE, and (crucially) that next
 * submit is normally in the NEXT domain, AFTER the switch: so get_and_switch(D+1) waits on D's refcount>0 and
 * TIMES OUT ("switch iommu domain time out, id: N") — the reap can never happen across the boundary. FIX: before
 * switching away from a dirty domain, issue a SAME-DOMAIN per-core dummy (ork_npu_reap_stuck) whose submit runs
 * timeout_clean(core i) -> clean gem_object_put of the stuck job -> refcount returns to 0, so the switch lands.
 * (ACT_RESET does NOT do this — source-confirmed rknpu_soft_reset is HW-only, leaves the job list; only
 * timeout_clean reaps. That's why the earlier ACT_RESET version failed.) No-op unless a real drop set dom_dirty
 * (rare); clean runs + single-domain pay nothing. Between-ops / pre-teardown only (no live pool workers). */
void ork_npu_reap_stuck(ork_npu *c, int nc);   /* fwd: per-core timeout_clean reap (defined below) */
int orki_i4_submit_tmo_ms(void);                    /* fwd: bounded int4 doorbell submit timeout (defined near the int4 workers) */
/* sync a sub-range of a buffer object (for arena views, which share one obj at varying offsets) */
/* Reserve `need` contiguous bytes of resident weight storage from the arena pool; returns the backing chunk
 * (and sets *base = byte offset within it) or NULL if a fresh chunk can't be allocated (caller then falls
 * back to per-tile bcreate). Chunk size = ORK_WARENA_CHUNK_MB (default 1 GiB), kept under the single-alloc
 * cap; a weight larger than the default chunk gets its own exact-size chunk. */

/* MULTI-DOMAIN SCRATCH SWAP. A submit runs in ONE iommu_domain_id, so the regcmd/task/activation/output
 * scratch a submit references must live in the same domain as the weight. dom_activate parks the current
 * active scratch into dom_save[old] and restores domain `dom`'s parked scratch (zero-initialized on first
 * use, so the run path's lazy bcreate allocates it in `dom` via c->dom_active). No-op when
 * single-domain (dom==dom_active and only domain 0 ever used). */

struct ork_npu *orki_npu_ctx = NULL;


/* replace ALL matching regcmd entries — the template repeats some regs (e.g. 0x1040) and
 * the NPU uses a later copy, so a first-match-only patch leaves stale values. */
#include <stdarg.h>
#include <sys/prctl.h>
#include "regcmd_array_4x32x16.h"
#include "ork_regs.h"
#include "regcmd_i8.h"
/* SETRN — named, bounds-checked register write (register-naming layer, task #40). Looks up the (block,
 * offset) from the rocket-cross-referenced table and validates the value fits the register's defined field
 * bits (a value with bits outside them is a bug — flagged, then written so behavior is preserved). This is
 * the normal path. RE / unbounded writes (specify block+offset+value with NO bounds) use orki_setr() directly —
 * that is the "raw override" the fuzzer hooks (ork_i8_fuzz_add/ork_f16_fuzz_add) already ride. */

void orki_warn_if_governor_parked(void){
    if(getenv("ORK_NO_GOV_WARN")) return;
    FILE*f=fopen("/sys/class/devfreq/dmc/governor","r"); if(!f) return;
    char g[64]={0};
    if(fgets(g,sizeof g,f)){ g[strcspn(g,"\n")]=0;
        if(strcmp(g,"performance")!=0)
            fprintf(stderr,"[ork] WARNING: DDR (dmc) governor is '%s', not 'performance' — decode may be ~half speed. "
                           "Pin it: echo performance | sudo tee /sys/class/devfreq/dmc/governor  (ORK_NO_GOV_WARN=1 to silence)\n",g);
    }
    fclose(f);
}

size_t ork_npu_sram_total(ork_npu *c){ (void)c; return (size_t)orki_sram_total; }

void ork_kmsg(const char *fmt, ...){
    char msg[504]; va_list ap; va_start(ap, fmt); vsnprintf(msg, sizeof msg, fmt, ap); va_end(ap);
    int kfd = open("/dev/kmsg", O_WRONLY|O_CLOEXEC);
    if(kfd >= 0){ char buf[512]; int n = snprintf(buf, sizeof buf, "<4>ork: %s", msg);
        if(n>0){ ssize_t w=write(kfd, buf, (size_t)n<sizeof buf?(size_t)n:sizeof buf-1); (void)w; } close(kfd); }
    /* ORK_F16_TRACE=<path>: ALSO mirror every step to a local, line-flushed file so a `tail -f` shows live progress
     * even when netconsole is lossy or the run never emits campaign output (e.g. a wedge-spin before the loop). */
    const char *tp = getenv("ORK_F16_TRACE");
    if(tp){ static FILE *tf; static int opened; if(!opened){ opened=1; tf=fopen(tp,"a"); }
        if(tf){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); fprintf(tf, "[%ld.%03ld] %s\n", (long)ts.tv_sec, ts.tv_nsec/1000000L, msg); fflush(tf); } }
}

void ork_npu_dump_state(ork_npu *c, const char *label){
    if(!c) return; int fd=c->fd; struct rknpu_action a;
    unsigned long long freq=0,volt=0,iommu=0,sram=0,hwv=0;
    #define ORK_Q(F,V) do{ memset(&a,0,sizeof a); a.flags=(F); if(!ioctl(fd,DRM_IOCTL_RKNPU_ACTION,&a)) V=(unsigned long long)a.value; }while(0)
    ORK_Q(RKNPU_GET_HW_VERSION,hwv); ORK_Q(RKNPU_GET_FREQ,freq); ORK_Q(RKNPU_GET_VOLT,volt);
    ORK_Q(RKNPU_GET_IOMMU_EN,iommu); ORK_Q(RKNPU_GET_FREE_SRAM_SIZE,sram);
    #undef ORK_Q
    /* The LIVE signals (established by floor_decomp / slice_replay — NOT the RKNPU_GET_*_AMOUNT DMA counters,
     * which read 0 on this kernel): (1) orki_fd_hw_raw_last = the kernel's per-submit NPU-busy time (hw_elapse),
     * captured after EVERY submit; 0 => the last job did NO work (faulted/never ran). (2) per-task int_status
     * from the shared task buffer (read after a FROM_DEVICE sync). */
    struct rknpu_task *t=(struct rknpu_task*)c->task.cpu;   /* NULL after an fd-reap (task buffer died with the fd, re-created lazily on the next run) — guard the deref */
    if(t) orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_FROM_DEVICE);
    fprintf(stderr,"[NPU-DUMP %s] hw=0x%llx freq=%llu volt=%llu iommu=%llu freeSRAM=%lluKiB | last-submit HW-busy(hw_elapse)=%lld (0=>no work) | task int_status[0..3]=%s0x%x 0x%x 0x%x 0x%x\n",
            label?label:"", hwv, freq, volt, iommu, sram>>10, (long long)orki_fd_hw_raw_last, t?"":"(no task buf) ",
            t?t[0].int_status:0u, t?t[1].int_status:0u, t?t[2].int_status:0u, t?t[3].int_status:0u);
}

int orki_reimport_inplace(int fd, struct buf *b){   /* re-map a persisted dma-buf into the reopened fd; keep b->cpu/size/heap_fd */
    if(b->heap_fd<=0 || !b->cpu) return -1;
    if(!ork_iova_reserve(b->domain, b->size)) return -1;
    struct drm_prime_handle ph; memset(&ph,0,sizeof ph); ph.fd=b->heap_fd; ph.flags=0;
    if(ioctl(fd,DRM_IOCTL_PRIME_FD_TO_HANDLE,&ph)){ ork_iova_release(b->domain,b->size); return -1; }
    struct rknpu_mem_create mc; memset(&mc,0,sizeof mc); mc.handle=ph.handle; mc.flags=0; mc.size=0; mc.core_mask=RKNPU_CORE0_MASK; mc.iommu_domain_id=b->domain;
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_CREATE,&mc)){ ork_iova_release(b->domain,b->size); return -1; }
    ork_kmsg("  reimport heap_fd=%d oldh=%u -> primeh=%u memh=%u dma=0x%llx obj=0x%llx", b->heap_fd, b->handle, ph.handle, mc.handle, (unsigned long long)mc.dma_addr, (unsigned long long)mc.obj_addr);
    b->handle=mc.handle; b->dma=mc.dma_addr; b->obj=mc.obj_addr;   /* new IOVA/handle; cpu/size/heap_fd/domain unchanged */
    orki_live_add(fd,b->handle,b->obj); return 0;
}

int ork_ctx_fd_reap(ork_npu *c){
    if(!c || c->fd<0) return -1;
    const char *card=getenv("ORK_NPU_CARD"); if(!card)card=c->soc->card;
    ork_kmsg("FD-REAP #%d: close(fd=%d) orki_imp=%d orki_live=%d — drm_release device teardown", ++orki_reap_n, c->fd, orki_imp_n, orki_live_n);
    /* ABORT any stuck/dropped in-flight job in-kernel BEFORE close. A REAL drop leaves an accepted-but-stuck job still
     * referencing a GEM object; drm_release then DOUBLE-PUTS that object (job-cleanup put + handle-release put) ->
     * refcount underflow/use-after-free -> slab corruption (vendor rknpu bug). RKNPU_ACT_RESET removes the job from the
     * scheduler first, so the subsequent close/drm_release puts each handle exactly once. (Proven necessary only for
     * REAL drops: 10 FORCE_WEDGE reaps of a CLEAN fd never underflowed; the underflow needs a stuck job at close.) */
    { struct rknpu_action ra; memset(&ra,0,sizeof ra); ra.flags=RKNPU_ACT_RESET; ioctl(c->fd,DRM_IOCTL_RKNPU_ACTION,&ra);
      struct timespec qs={0,2000000}; nanosleep(&qs,NULL); }   /* 2ms: let the reset settle + the aborted job drain before close */
    close(c->fd);
    int nf=open(card,O_RDWR); if(nf<0){ perror("FD-REAP reopen"); c->fd=-1; return -1; }
    prctl(PR_SET_TIMERSLACK,(unsigned long)1000,0UL,0UL,0UL);
    orki_act(nf,RKNPU_POWER_ON,0); orki_act(nf,RKNPU_SET_PROC_NICE,(uint32_t)-19);
    c->fd=nf;
    /* all prior IOMMU mappings + live handles are gone with the old fd */
    pthread_mutex_lock(&orki_live_mu); orki_live_n=0; orki_live_fd=nf; pthread_mutex_unlock(&orki_live_mu);
    for(int d=0; d<ORK_IOVA_NDOM; d++) orki_iova_bytes[d]=0;
    /* re-import every registered dma-buf weight IN PLACE (pages + cpu mmap persisted across the close) */
    int reimp=0, fail=0; pthread_mutex_lock(&orki_live_mu); int nimp=orki_imp_n;
    struct buf **imps=malloc((size_t)(nimp>0?nimp:1)*sizeof*imps); for(int i=0;i<nimp;i++) imps[i]=orki_imp[i];
    pthread_mutex_unlock(&orki_live_mu);
    for(int i=0;i<nimp;i++){ if(orki_reimport_inplace(nf,imps[i])==0) reimp++; else fail++; }
    free(imps);
    ork_kmsg("FD-REAP: reopened fd=%d, re-imported %d dma-buf weights (%d failed)", nf, reimp, fail);
    /* INVALIDATE bcreate'd scratch (lost with the fd) so it lazily re-creates on the fresh fd */
    #define ZB(b) memset(&(b),0,sizeof(b))
    ZB(c->regcmd); ZB(c->task); ZB(c->Af); ZB(c->Cc); ZB(c->mtk_all);
    ZB(c->ppu_a); ZB(c->ppu_b); ZB(c->ppu_o);
    for(int i=0;i<ORK_MAXCORE;i++){ ZB(c->mrc[i]); ZB(c->mtk[i]); ZB(c->maf[i]); ZB(c->mcc[i]);
        ZB(c->chain_rc[i]); ZB(c->chain_tk[i]); ZB(c->chain_lrc[i]); ZB(c->chain_lsc[i]);
        c->mccsz[i]=0; c->mwarm[i]=0; }
    #undef ZB
    c->ccsz=0; c->warmed=0; c->mc_alloc=0; c->last_dt=-1; c->last_chain=0;
    if(c->dom_save){ free(c->dom_save); c->dom_save=NULL; }   /* parked per-domain scratch died with the fd */
    if(c->dom_anchor){ free(c->dom_anchor); c->dom_anchor=NULL; }
    c->dom_cap=0; c->dom_active=0;
    return fail ? -1 : 0;
}

int ork_dummy_probe(ork_npu *c){
    int fd=c->fd, K=512, N=16, CBUF=c->soc->cbuf_elems, dom=c->dom_active;
    struct buf A=orki_bcreate(fd,(size_t)K,0x403,dom), B=orki_bcreate(fd,(size_t)K*N,0x403,dom), Cc=orki_bcreate(fd,(size_t)N*4,0x403,dom);
    if(!A.cpu||!B.cpu||!Cc.cpu){ if(A.cpu)orki_bdestroy(fd,&A); if(B.cpu)orki_bdestroy(fd,&B); if(Cc.cpu)orki_bdestroy(fd,&Cc); return 0; }
    memset(A.cpu,1,K); memset(B.cpu,1,(size_t)K*N);
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);
    volatile int32_t *db=(volatile int32_t*)((int32_t*)Cc.cpu+(N-1)); *db=0x7fffffff;
    __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); __asm__ volatile("dsb ish":::"memory");
    uint32_t rc[REGCMD_I8_N+4]; memset(rc,0,sizeof rc);
    orki_i8_synth(rc,1,K,N,(uint32_t)A.dma,(uint32_t)B.dma,(uint32_t)Cc.dma,1,CBUF,0);
    memcpy(c->regcmd.cpu,rc,REGCMD_I8_N*4); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
    t[0].enable_mask=0xd; t[0].int_mask=0x300; t[0].int_clear=0x1ffff; t[0].regcfg_amount=108; t[0].regcmd_addr=(uint32_t)c->regcmd.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit s; memset(&s,0,sizeof s);
    s.flags=0x1|0x2u;   /* PC | NONBLOCK: ioctl returns immediately, cannot enter the kernel continue-wait */
    s.task_number=1; s.task_obj_addr=c->task.obj; s.core_mask=RKNPU_CORE0_MASK; s.fence_fd=-1; s.timeout=300;
    s.subcore_task[0]=s.subcore_task[1]=s.subcore_task[2]=(struct rknpu_subcore_task){0,1};
    int rr=orki_rknpu_submit_ioctl(fd,&s,dom), ok=0;
    if(rr==0){ double t0=ork_now_us();   /* host-side bounded poll — no kernel wait, cannot hang */
        for(;;){ __asm__ volatile("dc civac,%0"::"r"(db):"memory");
            if(*db!=0x7fffffff){ ok=(*db==K); break; }
            if(ork_now_us()-t0>300000.0) break;   /* 300ms host cap => doorbell never landed => still wedged */ } }
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&B); orki_bdestroy(fd,&Cc);
    return ok;
}

int ork_npu_recover(ork_npu *c, const char *label){
    if(!c) return 0;
    ork_npu_dump_state(c,label);
    ork_npu_soft_reset(c);
    int ok=ork_dummy_probe(c);
    fprintf(stderr,"[NPU-RECOVER %s] dump + soft-reset + dummy-op -> %s\n", label?label:"", ok?"PASS (recovered, continue)":"FAIL (still broken, FAULT)");
    return ok;
}

int ork_npu_force_fault(ork_npu *c){
    if(!c) return -1; int fd=c->fd, K=512, N=16, CBUF=c->soc->cbuf_elems, dom=c->dom_active;
    struct buf A=orki_bcreate(fd,(size_t)K,0x403,dom), Cc=orki_bcreate(fd,(size_t)N*4,0x403,dom);
    if(!A.cpu||!Cc.cpu){ if(A.cpu)orki_bdestroy(fd,&A); if(Cc.cpu)orki_bdestroy(fd,&Cc); return -1; }
    memset(A.cpu,1,K); orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);
    volatile int32_t *db=(volatile int32_t*)((int32_t*)Cc.cpu+(N-1)); *db=0x7fffffff;
    __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); __asm__ volatile("dsb ish":::"memory");
    uint32_t rc[REGCMD_I8_N+4]; memset(rc,0,sizeof rc);
    orki_i8_synth(rc,1,K,N,(uint32_t)A.dma, 0x1000u /*BOGUS weight addr -> DMA fault*/, (uint32_t)Cc.dma,1,CBUF,0);
    memcpy(c->regcmd.cpu,rc,REGCMD_I8_N*4); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
    t[0].enable_mask=0xd; t[0].int_mask=0x300; t[0].int_clear=0x1ffff; t[0].regcfg_amount=108; t[0].regcmd_addr=(uint32_t)c->regcmd.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit s; memset(&s,0,sizeof s);
    s.flags=0x1|0x2u; s.task_number=1; s.task_obj_addr=c->task.obj; s.core_mask=RKNPU_CORE0_MASK; s.fence_fd=-1; s.timeout=500;
    s.subcore_task[0]=s.subcore_task[1]=s.subcore_task[2]=(struct rknpu_subcore_task){0,1};
    int rr=orki_rknpu_submit_ioctl(fd,&s,dom), landed=0;
    if(rr==0){ double t0=ork_now_us(); for(;;){ __asm__ volatile("dc civac,%0"::"r"(db):"memory"); if(*db!=0x7fffffff){landed=1;break;} if(ork_now_us()-t0>1000000.0) break; } }
    fprintf(stderr,"[FORCE-FAULT] bogus-weight matmul submit rc=%d, doorbell %s (rw counters unreliable; watch dmesg for DMA_READ_ERROR/soft reset)\n", rr, landed?"LANDED (no fault?!)":"did NOT land (faulted as intended)");
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&Cc);
    return landed;
}

/* orkd CLIENT context — the EXPLICIT orkd entry point. Connect (auto-spawn) the orkd daemon and route
 * ork_mm_* through it: the daemon owns the single-stream NPU and serializes every submit, the safe way to
 * share it across concurrent processes. NO local NPU open (the daemon owns it); the ops check c->daemon.
 * Returns NULL if the daemon can't be reached (NO silent fallback to direct — the caller decides). The daemon
 * process itself must not call this (it sets ORKD_IS_DAEMON). Callers pick transport by CHOOSING the entry
 * point: ork_npu_init() = direct (default), ork_npu_init_orkd() = orkd client. */
ork_npu *ork_npu_init_orkd(void){
    const struct ork_soc *soc=ork_soc_detect();
    if(!soc){fprintf(stderr,"[ork] ERROR: unknown SoC (no device-tree match) — cannot select NPU params\n");return NULL;}
    { const char *isd=getenv("ORKD_IS_DAEMON");
      if(isd && atoi(isd)){ fprintf(stderr,"[ork] ERROR: ork_npu_init_orkd() called inside the daemon — use ork_npu_init() (direct)\n"); return NULL; } }
    orkd_conn *dc=orkd_connect();
    if(!dc){ fprintf(stderr,"[ork] ERROR: ork_npu_init_orkd() — orkd_connect failed (daemon not reachable/spawnable)\n"); return NULL; }
    ork_npu *c=calloc(1,sizeof *c); c->fd=-1; c->soc=soc; c->daemon=dc; c->last_dt=-1; c->core_budget=soc->cores; c->pack_domain=-1; orki_npu_ctx=c;
    if(getenv("ORK_ORKD_RING")) orkd_ring_setup(dc);   /* low-latency transport: ork_f16_mm_run* + ork_mm_submit ride the ring (daemon busy-polls while attached, so opt-in) */
    if(getenv("ORK_TRACE")) fprintf(stderr,"[ork] client mode: routing through orkd (cores=%u, ring=%d)\n",orkd_soc_cores(dc),orkd_has_ring(dc));
    return c;
}

void ork_npu_free(ork_npu *c){ if(!c)return; if(c->daemon){ orkd_disconnect(c->daemon); free(c->cres); free(c); return; }   /* Path B: client mode — disconnect from orkd, no local NPU teardown */
    if(c->fd < 0){ pthread_mutex_destroy(&c->pmu); pthread_cond_destroy(&c->pgo); pthread_cond_destroy(&c->pdn); free(c); return; }   /* offline context (ork_npu_init_offline): no device, no buffers, nothing to tear down */
    int fd=c->fd; ork_dom_flush_if_dirty(c);   /* #54: clear any stuck job before teardown's per-domain MEM_DESTROYs switch domains */
    ork_load_prof_dump(); ork_ssm_prof_dump(); ork_npu_xprof_dump();
    if(getenv("ORK_DOM_PROFILE") && c->dom_sw_n){   /* domain-swap window telemetry */
        uint64_t steady_n = c->dom_sw_n - c->dom_sw_first_n; double steady_us = c->dom_sw_us - c->dom_sw_first_us;
        fprintf(stderr, "[ork DOM-SWAP] %llu real switches (%.1f us total, %.2f us max) | first-touch %llu (%.0f us, one-time bcreate) | steady swaps %llu = %.3f us total, %.4f us/swap avg\n",
            (unsigned long long)c->dom_sw_n, c->dom_sw_us, c->dom_sw_max_us,
            (unsigned long long)c->dom_sw_first_n, c->dom_sw_first_us,
            (unsigned long long)steady_n, steady_us, steady_n ? steady_us/(double)steady_n : 0.0);
    }
    if(c->seq_witness){ ork_mm_free(c, c->seq_witness); c->seq_witness=NULL; }   /* B2 terminal-SDP witness weight */
    ork_ssm_helper_stop(c);   /* join the persistent little-core helper (no-op if never spawned) */
    orki_ssm_pool_free(c);   /* release the persistent SSM-scan scratch pool (no-op if never used) */
    if(orki_ork_prof){
        if(orki_prof_i8_calls) fprintf(stderr,"[ork PROFILE] run_i8: %ld calls, %.1f ms total, %.0f us/call\n",
                                    orki_prof_i8_calls, orki_prof_i8_us/1e3, orki_prof_i8_us/orki_prof_i8_calls);
        if(orki_prof_i4_calls) fprintf(stderr,"[ork PROFILE] run_i4: %ld calls, %.1f ms total, %.0f us/call\n",
                                    orki_prof_i4_calls, orki_prof_i4_us/1e3, orki_prof_i4_us/orki_prof_i4_calls);
        if(orki_prof_submits) fprintf(stderr,"[ork PROFILE] submits: %ld ioctls, %ld programs (%.2f prog/ioctl), %ld chained(>1prog). per-i8-call: %.2f ioctls\n",
                                   orki_prof_submits, orki_prof_submit_progs, (double)orki_prof_submit_progs/orki_prof_submits, orki_prof_submit_chained,
                                   orki_prof_i8_calls?(double)orki_prof_submits/orki_prof_i8_calls:0.0);
        /* HW-vs-poll split (task #19): orki_run() total vs the SUBMIT ioctl wall vs the kernel-reported HW hw_elapse.
         * poll/idle = orki_run() - submit-ioctl-wall = the completion-wait after the NONBLOCK doorbell submit. */
        if(orki_fd_n) fprintf(stderr,"[ork FD-SPLIT] orki_run() %.1fms | %ld submit-ioctls: wall %.1fms (%.0f%%), HW hw_elapse %.1fms (%.0f%%) => poll/idle-wait %.1fms (%.0f%%) | per-ioctl: wall %.0fus HW %.0fus\n",
                                   orki_prof_i8_us/1e3, orki_fd_n, orki_fd_ioctl_us/1e3, 100.0*orki_fd_ioctl_us/(orki_prof_i8_us+1e-9),
                                   orki_fd_hw_us/1e3, 100.0*orki_fd_hw_us/(orki_prof_i8_us+1e-9),
                                   (orki_prof_i8_us-orki_fd_ioctl_us)/1e3, 100.0*(orki_prof_i8_us-orki_fd_ioctl_us)/(orki_prof_i8_us+1e-9),
                                   orki_fd_ioctl_us/orki_fd_n, orki_fd_hw_us/orki_fd_n);
    }
    if (orki_npu_ctx == c) orki_npu_ctx = NULL;
    if(c->pool_n){ pthread_mutex_lock(&c->pmu); c->pstop=1; pthread_cond_broadcast(&c->pgo); pthread_mutex_unlock(&c->pmu);
        for(int i=1;i<c->pool_n;i++) pthread_join(c->pth[i],NULL); }
    orki_fold_scratch_free(c);   /* #39 resident mfold scratch */
    orki_bdestroy(fd,&c->regcmd);orki_bdestroy(fd,&c->task);orki_bdestroy(fd,&c->Af);orki_bdestroy(fd,&c->Cc);orki_bdestroy(fd,&c->mtk_all);
    orki_bdestroy(fd,&c->ppu_a);orki_bdestroy(fd,&c->ppu_b);orki_bdestroy(fd,&c->ppu_o);   /* persistent SDP-op scratch */
    for(int i=0;i<ORK_MAXCORE;i++){orki_bdestroy(fd,&c->mrc[i]);orki_bdestroy(fd,&c->mtk[i]);orki_bdestroy(fd,&c->maf[i]);orki_bdestroy(fd,&c->mcc[i]);
        orki_bdestroy(fd,&c->chain_rc[i]);orki_bdestroy(fd,&c->chain_tk[i]);orki_bdestroy(fd,&c->chain_lrc[i]);orki_bdestroy(fd,&c->chain_lsc[i]);}   /* multi-core matmul + per-core chain scratch, one pass */
    /* free PARKED per-domain scratch (the active set above is whichever domain was last run) */
    if(c->dom_save){ for(int d=0;d<c->dom_cap;d++){ if(d==c->dom_active||!c->dom_save[d].used) continue;
        struct ork_dom_scratch *s=&c->dom_save[d];
        orki_bdestroy(fd,&s->regcmd);orki_bdestroy(fd,&s->task);orki_bdestroy(fd,&s->Af);orki_bdestroy(fd,&s->Cc);orki_bdestroy(fd,&s->mtk_all);
        for(int i=0;i<ORK_MAXCORE;i++){orki_bdestroy(fd,&s->mrc[i]);orki_bdestroy(fd,&s->mtk[i]);orki_bdestroy(fd,&s->maf[i]);orki_bdestroy(fd,&s->mcc[i]);} }
        free(c->dom_save); }
    for(int i=0;i<c->dma_n;i++) orki_bdestroy(fd,&c->dma_tab[i]);
    if(c->dom_anchor){ for(int d=0;d<c->dom_cap;d++) if(c->dom_anchor[d].cpu) orki_bdestroy(fd,&c->dom_anchor[d]); free(c->dom_anchor); }   /* per-domain native anchors */
    if(c->i4arena){ for(int i=0;i<c->i4arena_n;i++) if(c->i4arena[i].cpu) orki_bdestroy(fd,&c->i4arena[i]); free(c->i4arena); }   /* #54 int4 expert import arena (bdestroy closes each chunk's heap_fd) */
    for(int i=0;i<c->wchunk_n;i++) if(c->wchunk[i].cpu) orki_bdestroy(fd,&c->wchunk[i]);   /* weight-arena chunks: nothing freed these, so the kernel reaped them at fd close */
    orki_live_reap(fd);   /* safety net: nothing may survive into drm_gem_release (wrong-domain unmap -> leaked IOVA) */
    free(c->cres); if(fd>=0)close(fd); free(c); }

const char *ork_npu_soc(const ork_npu *c){return c->soc->id;}

int ork_npu_cores(const ork_npu *c){return c->soc->cores;}

int ork_npu_validated(const ork_npu *c){return c->soc->validated;}

void ork_npu_set_core_budget(ork_npu *c,int n){ if(!c)return; c->core_budget=(n>0&&n<=c->soc->cores)?n:c->soc->cores; }

void ork_npu_set_priority(ork_npu *c,unsigned prio){ if(c && c->daemon) orkd_set_priority(c->daemon, prio); }

/* Allocate an IOMMU domain to pack weights into (isolation + a full ~4 GiB IOVA window each). Path B: request
 * one from orkd's coordinated pool (returns id>0, or <0 if exhausted). Direct: hand out a local id (1,2,…).
 * Make it the pack target with ork_npu_set_pack_domain; return it with ork_npu_domain_free. */
int ork_npu_domain_alloc(ork_npu *c){
    if(!c) return -1;
    if(c->daemon) return orkd_domain_alloc(c->daemon);
    if(c->dom_next < 1) c->dom_next = 1;
    if(orki_dom_reserve(c, c->dom_next+1)) return -1;   /* grow arrays to cover the new domain (no fixed cap); OOM -> fail */
    return c->dom_next++;
}

int ork_npu_domain_free(ork_npu *c,int domain){
    if(!c || domain<=0) return -1;
    if(c->daemon) return orkd_domain_free(c->daemon, domain);
    return 0;   /* direct mode: domains aren't pooled — a weight's buffers free with the weight */
}

int  ork_npu_pack_domain(const ork_npu *c){ return c ? c->pack_domain : -1; }   /* current pack domain (save/restore around a domain-targeted alloc) */

/* Currently ACTIVE iommu domain (the one dom_activate last swapped in), i.e. the domain the NEXT submit
 * would run in if its weight already lives there. Pure getter, no state change. The point: a caller that
 * allocates a TRANSIENT/scratch weight (an attention or GDN bmm's dynamic operand) can place it in the
 * domain that is already active, so running it needs NO dom_activate switch — the switch is what a stuck
 * (unreaped, IRQ-never-fired) job turns into a 60 s "switch iommu domain" stall on the NEXT submit. See
 * ork_dom_flush_if_dirty / dom_dirty. Co-domain scratch sidesteps the whole boundary. */
int  ork_npu_active_domain(const ork_npu *c){ return c ? c->dom_active : 0; }

/* Make `domain` the ACTIVE iommu domain (parks/restores per-domain scratch, establishes it if fresh). A
 * DMA buffer created for a non-0 domain must be allocated while that domain is active, else it maps in the
 * currently-active domain and a submit against `domain` can't see it. Call before ork_dma_alloc-in-domain. */
void ork_npu_activate_domain(ork_npu *c, int domain){ if(c) orki_dom_activate(c, domain<0?0:domain); }

int  ork_npu_uses_orkd(const ork_npu *c){ return (c && c->daemon) ? 1 : 0; }   /* 1 = this ctx routes through orkd (serialized); 0 = DIRECT NPU (single-stream — don't run concurrent direct processes) */

int ork_npu_busy(ork_npu *ctx){
    (void)ctx;
    FILE *f=fopen("/sys/kernel/debug/rknpu/load","r");
    if(!f) return 0;
    char buf[256]; size_t n=fread(buf,1,sizeof buf-1,f); fclose(f);
    if(!n) return 0; buf[n]=0;
    /* format: "NPU load:  Core0:  0%, Core1:  0%, Core2:  0%," — busy if any core % exceeds threshold */
    for(const char *p=buf; (p=strchr(p,'%')); p++){
        const char *q=p; while(q>buf && (q[-1]==' ' || (q[-1]>='0'&&q[-1]<='9'))) q--;
        if(atoi(q)>5) return 1;
    }
    return 0;
}

void ork_npu_mode_invalidate(ork_npu *c){ if(!c) return; c->last_dt=-1; c->warmed=0; for(int i=0;i<ORK_MAXCORE;i++) c->mwarm[i]=0; }

void ork_npu_mode_reset(ork_npu *c){ if(!c) return; orki_act(c->fd,RKNPU_ACT_RESET,0); ork_npu_mode_invalidate(c); }

const char *ork_npu_version(void){
#ifdef ORK_GIT_HASH
    static char v[64]; snprintf(v, sizeof v, "%s+g%s", ORK_NPU_VERSION, ORK_GIT_HASH); return v;
#else
    return ORK_NPU_VERSION;   /* no git at build time (e.g. native board build) → semver only */
#endif
}

uint32_t ork_pack_format_version(void){ return ORK_PACK_FORMAT_VERSION; }   /* decoupled from the library MAJOR — bump only on a real on-disk format change (see ork_npu.h) */
void ork_npu_reap_stuck(ork_npu *c, int nc){
    int fd=c->fd, K=512, N=16, CBUF=c->soc->cbuf_elems;  unsigned dom=c->dom_active;
    if(nc<1) nc=1; if(nc>c->soc->cores) nc=c->soc->cores;
    struct buf A=orki_bcreate(fd,(size_t)K*2,0x403,dom), B=orki_bcreate(fd,(size_t)K*N*2,0x403,dom), Cc=orki_bcreate(fd,(size_t)N*2,0x403,dom);
    if(!A.cpu||!B.cpu||!Cc.cpu){ if(A.cpu)orki_bdestroy(fd,&A); if(B.cpu)orki_bdestroy(fd,&B); if(Cc.cpu)orki_bdestroy(fd,&Cc); return; }
    memset(A.cpu,0,(size_t)K*2); memset(B.cpu,0,(size_t)K*N*2);
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t rc[REGCMD_N]; int sched=((K&(K-1))==0 && K>=128 && K<2048);
    orki_f16_synth(rc,1,K,N,(uint32_t)A.dma,(uint32_t)B.dma,(uint32_t)Cc.dma,sched,CBUF); orki_f16_set_out_fp16in(rc,1,N);
    memcpy(c->regcmd.cpu,rc,(size_t)REGCMD_N*4); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
    t[0].enable_mask=0xd; t[0].int_mask=0x300; t[0].int_clear=0x1ffff; t[0].regcfg_amount=108; t[0].regcmd_addr=(uint32_t)c->regcmd.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    for(int i=0;i<nc;i++){
        struct rknpu_submit s; memset(&s,0,sizeof s);
        s.flags=0x1|0x2u; s.task_number=1; s.task_obj_addr=c->task.obj; s.core_mask=1u<<i; s.fence_fd=-1; s.timeout=300;
        s.subcore_task[0]=s.subcore_task[1]=s.subcore_task[2]=(struct rknpu_subcore_task){0,1};
        orki_rknpu_submit_ioctl(fd,&s,dom);   /* triggers rknpu_job_timeout_clean(core i) -> clean reap of a timed-out stuck job */
        ork_kmsg("reap-stuck: fp16 nonblock dummy core=%d (trigger timeout_clean)", i);
        struct timespec ds={0,3000000}; nanosleep(&ds,NULL);   /* let this core's dummy land + the scheduled cleanup_work run */
    }
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&B); orki_bdestroy(fd,&Cc);
}
