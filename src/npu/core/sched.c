/* npu/core/sched.c — worker pool, big/little core pinning, parallel_for, per-core multi-core scratch.
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
#include <stdarg.h>
#include <sys/prctl.h>
#include "ork_regs.h"
#include "regcmd_i8.h"
#include "npu/internal.h"
#include "npu/core.h"
#include "npu/core/core.h"

#define ORK_POOL_MAX 64
struct {
    int inited, n, quit;
    pthread_t th[ORK_POOL_MAX];
    pthread_mutex_t mu; pthread_cond_t go, done;
    void (*fn)(int,int,void*); void *ctx;
    int lo[ORK_POOL_MAX], hi[ORK_POOL_MAX];
    int gen, running;
} orki_pool = { .mu = PTHREAD_MUTEX_INITIALIZER, .go = PTHREAD_COND_INITIALIZER, .done = PTHREAD_COND_INITIALIZER };


/* DIRECT (in-process) NPU context — the DEFAULT entry point: opens the DRM card and owns the single-stream
 * NPU directly (do not run concurrent direct-NPU processes; they wedge the IOMMU). For back-compat, the legacy
 * ORK_USE_ORKD=1 env still redirects this to the orkd client (ork_npu_init_orkd) — but new callers should
 * select the transport by calling the desired entry point rather than relying on the env. */
int orki_mc_ensure(ork_npu *c,int nc);   /* fwd: #54 pre-alloc domain-0 run scratch at init (while empty) */
ork_npu *ork_npu_init(void){
    const struct ork_soc *soc=ork_soc_detect();
    if(!soc){fprintf(stderr,"[ork] ERROR: unknown SoC (no device-tree match) — cannot select NPU params\n");return NULL;}
    /* Legacy env override: ORK_USE_ORKD set (and not the daemon itself, which sets ORKD_IS_DAEMON) routes
     * through orkd. Delegates to the explicit entry point; on connect failure, falls back to the direct NPU. */
    { const char *ud=getenv("ORK_USE_ORKD"), *isd=getenv("ORKD_IS_DAEMON");
      if(ud && atoi(ud) && !(isd && atoi(isd))){
        ork_npu *c=ork_npu_init_orkd();
        if(c) return c;
        fprintf(stderr,"[ork] WARNING: ORK_USE_ORKD set but orkd_connect failed — using the local NPU\n"); } }
    if(!soc->validated) fprintf(stderr,"[ork] WARNING: %s params are inherited/untested — validate with the regression suite\n",soc->id);
    orki_warn_if_governor_parked();
    orki_ork_prof = getenv("ORK_PROFILE") ? 1 : 0;
    orki_load_prof = getenv("ORK_LOAD_PROF") ? 1 : 0;
    const char*card=getenv("ORK_NPU_CARD"); if(!card)card=soc->card;
    int fd=open(card,O_RDWR); if(fd<0){perror("open NPU card");return NULL;}
    prctl(PR_SET_TIMERSLACK, (unsigned long)1000, 0UL, 0UL, 0UL);   /* 1µs timer slack (default 50µs): precise short nanosleeps for the doorbell backoffs */
    orki_act(fd,RKNPU_GET_DRV_VERSION,0);orki_act(fd,RKNPU_POWER_ON,0);orki_act(fd,RKNPU_SET_PROC_NICE,(uint32_t)-19);
    /* Query NPU on-chip SRAM once: gates the TRY_ALLOC_SRAM->DRAM failover in orki_bcreate (see orki_sram_total). */
    { struct rknpu_action a; memset(&a,0,sizeof a); a.flags=RKNPU_GET_TOTAL_SRAM_SIZE;
      if(!ioctl(fd,DRM_IOCTL_RKNPU_ACTION,&a)) orki_sram_total=a.value;
      if(getenv("ORK_TRACE")||getenv("ORK_LOAD_PROF"))
          fprintf(stderr,"[ork] NPU SRAM: %u KiB %s\n",(unsigned)(orki_sram_total>>10),
                  orki_sram_total?"(SRAM-backed alloc available)":"(none — DRAM-only, TRY_ALLOC_SRAM fails over)"); }
    ork_npu *c=calloc(1,sizeof *c); c->fd=fd; c->soc=soc; c->last_dt=-1; c->core_budget=soc->cores; c->pack_domain=-1; c->last_async_cpu=-1;
    pthread_mutex_init(&c->pmu,NULL); pthread_cond_init(&c->pgo,NULL); pthread_cond_init(&c->pdn,NULL);
    c->regcmd=orki_bcreate(fd,2097152,0x403,-1); c->task=orki_bcreate(fd,524288,0x40b,-1); c->Af=orki_bcreate(fd,(size_t)4*32768*2,0x403,-1);
    struct rknpu_task t; memset(&t,0,sizeof t); t.enable_mask=0xd;t.int_mask=0x300;t.int_clear=0x1ffff;t.regcfg_amount=108;t.regcmd_addr=c->regcmd.dma;
    memcpy(c->task.cpu,&t,sizeof t); orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    if(!c->regcmd.cpu||!c->task.cpu||!c->Af.cpu){ork_npu_free(c);return NULL;}
    /* graceful teardown: MEM_DESTROY all live mappings on SIGTERM/SIGINT so a killed orki_run (e.g. `timeout`)
     * releases its IOMMU domains instead of stranding them until reboot (see the live-buffer registry). */
    { const char*e=getenv("ORK_NO_SIGCLEAN"); if(!(e&&atoi(e))){
        struct sigaction sa; memset(&sa,0,sizeof sa); sa.sa_handler=ork_sig_teardown; sigemptyset(&sa.sa_mask);
        sigaction(SIGTERM,&sa,NULL); sigaction(SIGINT,&sa,NULL); } }
    /* #54: pre-allocate domain-0 run scratch (mtk_all + per-core mrc/mtk/maf) NOW, while domain 0 is empty. It
     * would otherwise be bcreate'd lazily at the first forward op, AFTER the dense weights import ~GiB into
     * domain 0 — where a fresh bcreate EINVALs amid the imports (the mc_ensure mtk_all failure). Best-effort. */
    orki_mc_ensure(c, c->soc->cores);
    for(int i=0;i<c->soc->cores;i++){ size_t mcc_need=(size_t)2*1024*1024;   /* also pre-size domain-0 mcc (expert BCHAIN need_o) so it never re-bcreates in the import-heavy domain 0 */
        if(c->mccsz[i]<mcc_need){ orki_bdestroy(fd,&c->mcc[i]); c->mcc[i]=orki_bscratch(c,mcc_need,0x403,c->dom_active); if(c->mcc[i].cpu){ c->mccsz[i]=mcc_need; c->mwarm[i]=0; } } }
    orki_npu_ctx = c;
    return c;
}

/* ork_npu_init_offline — a context with SoC caps and NO device.
 *
 * WHY. Building a native-W4A4 .orkpack is pure CPU work: dequant, rotate, quantize, and TILE. The tiling
 * was the only step that used to need hardware, and ork_i4_w_dump_cpu removed that (it reads exactly one
 * thing from the context — c->soc->nmax — and is asserted byte-identical to the NPU's own pack+dump by
 * test_i4_dump_cpu). What remained was ork_npu_init itself: it opens /dev/dri/cardN and fails on any
 * machine that is not the board, so a pack build was pinned to the board for no computational reason.
 * That matters because packing is the SLOW half of every quantization experiment, it is CPU-bound, and
 * the board is both the weakest machine available and a single shared, wedge-prone resource.
 *
 * WHAT YOU GET. Only the CPU-side surfaces: the *_w_dump_cpu tilers and anything else that reads caps
 * rather than the device. fd is -1, so every ioctl path fails cleanly rather than corrupting state —
 * there is no partially-live device to get wrong. Nothing is warmed, no buffers exist, no signal handler
 * is installed (there are no IOMMU mappings to strand).
 *
 * The SoC must be named explicitly: there is no device tree to detect from, and silently defaulting would
 * produce a pack tiled for the wrong nmax — which is exactly the class of error test_i4_dump_cpu exists
 * to catch, and it would slip through unnoticed on a machine that cannot run that test. */
ork_npu *ork_npu_init_offline(const char *soc_id){
    const struct ork_soc *soc=ork_soc_by_id(soc_id);
    if(!soc){ fprintf(stderr,"[ork] ERROR: ork_npu_init_offline: unknown SoC id \"%s\"\n", soc_id?soc_id:"(null)"); return NULL; }
    ork_npu *c=calloc(1,sizeof *c);
    if(!c) return NULL;
    c->fd=-1; c->soc=soc; c->last_dt=-1; c->core_budget=soc->cores; c->pack_domain=-1; c->last_async_cpu=-1;
    pthread_mutex_init(&c->pmu,NULL); pthread_cond_init(&c->pgo,NULL); pthread_cond_init(&c->pdn,NULL);
    return c;   /* deliberately NOT orki_npu_ctx: an offline context must never become the implicit device */
}

int ork_all_cores_mask(cpu_set_t *s){
    long n=sysconf(_SC_NPROCESSORS_ONLN); if(n<1) return 0;
    CPU_ZERO(s); for(long i=0;i<n && i<CPU_SETSIZE;i++) CPU_SET((int)i,s);
    return (int)n;
}

void ork_unpin_current_thread(void){
    cpu_set_t all; if(ork_all_cores_mask(&all)) pthread_setaffinity_np(pthread_self(), sizeof all, &all);
}

void *ork_pool_worker(void *a){
    int id = (int)(intptr_t)a;
    ork_unpin_current_thread();
    int seen = 0;
    pthread_mutex_lock(&orki_pool.mu);
    for(;;){
        while(orki_pool.gen == seen && !orki_pool.quit) pthread_cond_wait(&orki_pool.go, &orki_pool.mu);
        if(orki_pool.quit){ pthread_mutex_unlock(&orki_pool.mu); return NULL; }
        seen = orki_pool.gen;
        void (*fn)(int,int,void*) = orki_pool.fn; void *ctx = orki_pool.ctx;
        int lo = orki_pool.lo[id], hi = orki_pool.hi[id];
        pthread_mutex_unlock(&orki_pool.mu);
        if(fn && hi > lo) fn(lo, hi, ctx);
        pthread_mutex_lock(&orki_pool.mu);
        if(--orki_pool.running == 0) pthread_cond_signal(&orki_pool.done);
    }
}

void ork_pool_init(void){
    if(orki_pool.inited) return;
    int cores=(int)sysconf(_SC_NPROCESSORS_ONLN); if(cores<1)cores=1;
    /* ORK_POOL_MULT oversubscribes the cores (e.g. =2 → 2 threads/core) to hide memory-latency stalls
     * in the tiling/dequant (bandwidth is not saturated, so extra threads can fill the stall bubbles). */
    int mult = getenv("ORK_POOL_MULT") ? atoi(getenv("ORK_POOL_MULT")) : 1; if(mult<1)mult=1; if(mult>8)mult=8;
    int n = cores*mult; if(n>ORK_POOL_MAX)n=ORK_POOL_MAX;
    orki_pool.n = n; orki_pool.inited = 1;
    for(int i=1;i<n;i++)   /* worker 0 == the caller; helpers 1..n-1 */
        if(pthread_create(&orki_pool.th[i], NULL, ork_pool_worker, (void*)(intptr_t)i)!=0){ orki_pool.n = i; break; }
}

/* PERSISTENT worker pool for ork_parallel_for. Spawn the workers ONCE and reuse them across every
 * call, amortizing the pthread_create/join that dominated fine-grained per-weight tiling — a fresh
 * pool per weight left the cores mostly idle in spawn/join overhead (measured: per-weight CPU tiling
 * capped ~20%). Workers are un-pinned (all cores) and sleep on a condvar between jobs; lazy-init on
 * first use, live for the process. One job at a time (the callers dispatch serially). */
void ork_parallel_for(int n, void (*fn)(int,int,void*), void *ctx){
    if(n<=0) return;
    if(n==1){ fn(0,1,ctx); return; }
    static pthread_mutex_t dispatch = PTHREAD_MUTEX_INITIALIZER;   /* pool is a shared singleton — one job at a time */
    pthread_mutex_lock(&dispatch);
    ork_pool_init();
    int nthr = orki_pool.n; if(nthr>n) nthr=n; if(nthr<1) nthr=1;   /* use the whole pool (incl. oversubscription) */
    if(nthr<=1){ pthread_mutex_unlock(&dispatch); fn(0,n,ctx); return; }
    int chunk = (n + nthr - 1) / nthr;
    pthread_mutex_lock(&orki_pool.mu);
    orki_pool.fn = fn; orki_pool.ctx = ctx;
    for(int t=0;t<orki_pool.n;t++){ int a=t*chunk; if(a>n)a=n; int b=a+chunk; if(b>n)b=n; orki_pool.lo[t]=a; orki_pool.hi[t]=b; }
    orki_pool.running = orki_pool.n - 1;   /* every helper wakes + decrements (idle ones just have empty ranges) */
    orki_pool.gen++;
    pthread_cond_broadcast(&orki_pool.go);
    pthread_mutex_unlock(&orki_pool.mu);
    if(orki_pool.hi[0] > orki_pool.lo[0]) fn(orki_pool.lo[0], orki_pool.hi[0], ctx);   /* caller runs chunk 0 */
    pthread_mutex_lock(&orki_pool.mu);
    while(orki_pool.running > 0) pthread_cond_wait(&orki_pool.done, &orki_pool.mu);
    pthread_mutex_unlock(&orki_pool.mu);
    pthread_mutex_unlock(&dispatch);
}

void orki_pin_big_core(int id){
    static int off=-1; if(off<0) off=getenv("ORK_NO_AFFINITY")?1:0;   /* cached: hot for i4 per-call */
    if(off) return;
#if defined(__linux__)
    long ncpu=sysconf(_SC_NPROCESSORS_ONLN); if(ncpu<2) return;
    int cpu=(int)ncpu-1-id; if(cpu<0) cpu=0;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu,&s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
#endif
    /* NOTE: this pins only the dedicated NPU-driver threads to their own big core. We deliberately do
     * NOT restrict the whole process / CPU threadpool to the big cluster: doing so at init was measured
     * to oversubscribe and CRATER decode at -t 8 (9.3 -> 2.3 tok/s) while not helping -t 4. The big-core
     * win for the CPU side comes from running with -t = big-core-count (e.g. -t 4 on RK3588), which is a
     * user/serving choice, not something to force here. See the Thread-Count wiki experiment. */
}

void orki_pin_little_core(int id){
    static int off=-1; if(off<0) off=getenv("ORK_NO_AFFINITY")?1:0;
    if(off) return;
#if defined(__linux__)
    long ncpu=sysconf(_SC_NPROCESSORS_ONLN); if(ncpu<2) return;
    int cpu=id; if(cpu>=ncpu) cpu=0;           /* low index = little cluster */
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu,&s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
#endif
}

void *orki_npu_pool_worker(void *vp){
    struct ork_pw *pw=vp; ork_npu *c=pw->c; int id=pw->id, mygen=0;
    orki_pin_big_core(id);                          /* keep this worker off the little cores */
    for(;;){
        pthread_mutex_lock(&c->pmu);
        while(c->pgen==mygen && !c->pstop) pthread_cond_wait(&c->pgo,&c->pmu);
        if(c->pstop){ pthread_mutex_unlock(&c->pmu); return NULL; }
        mygen=c->pgen; int nc=c->pjob_nc; void *args=c->pjob; void *(*fn)(void*)=c->pjob_fn; size_t st=c->pjob_stride; pthread_mutex_unlock(&c->pmu);
        if(id<nc){ fn((char*)args + (size_t)id*st);   /* chain_core_worker / colsplit per-core worker */
            pthread_mutex_lock(&c->pmu); if(++c->pdone==nc-1) pthread_cond_signal(&c->pdn); pthread_mutex_unlock(&c->pmu); }
    }
}

void orki_npu_pool_ensure(ork_npu *c){
    if(c->pool_n) return;
    orki_pin_big_core(0);                           /* calling thread drives NPU core 0 — keep it big too */
    c->pool_n=c->soc->cores>ORK_MAXCORE?ORK_MAXCORE:c->soc->cores;
    for(int i=1;i<c->pool_n;i++){ c->pwa[i]=(struct ork_pw){c,i}; pthread_create(&c->pth[i],NULL,orki_npu_pool_worker,&c->pwa[i]); }
}

int ork_big_core_set(cpu_set_t *s){
#if defined(__linux__)
    static int off=-1; if(off<0) off=getenv("ORK_NO_AFFINITY")?1:0;
    if(off) return 0;
    long ncpu=sysconf(_SC_NPROCESSORS_ONLN); if(ncpu<2) return 0;
    CPU_ZERO(s); for(int k=(int)(ncpu/2);k<ncpu;k++) CPU_SET(k,s);  /* top half = big cluster on RK35xx */
    return 1;
#else
    (void)s; return 0;
#endif
}
int orki_mc_ensure(ork_npu *c,int nc){
    int fd=c->fd;
    if(!c->mtk_all.cpu) {
        c->mtk_all=orki_bscratch(c, sizeof(struct rknpu_task) * ORK_MAXCORE, 0x40b, c->dom_active);
        if(!c->mtk_all.cpu) {
            fprintf(stderr, "[ork] ERROR: mc_ensure failed to allocate mtk_all task buffer (IOMMU full?)\n");
            return -1;
        }
    }
    for(int i=0;i<nc;i++){
        if(c->mrc[i].cpu) continue;        /* alloc once, per core, up to the max ever requested */
        c->mrc[i]=orki_bscratch(c,65536,0x403,c->dom_active); c->mtk[i]=orki_bscratch(c,65536,0x40b,c->dom_active); c->maf[i]=orki_bscratch(c,(size_t)4*32768*2,0x403,c->dom_active);
        if(!c->mrc[i].cpu||!c->mtk[i].cpu||!c->maf[i].cpu) {
            fprintf(stderr, "[ork] ERROR: mc_ensure failed to allocate multi-core buffers for core %d (IOMMU full?)\n", i);
            return -1;
        }
        struct rknpu_task t;memset(&t,0,sizeof t);t.enable_mask=0xd;t.int_mask=0x300;t.int_clear=0x1ffff;t.regcfg_amount=108;t.regcmd_addr=c->mrc[i].dma;
        memcpy(c->mtk[i].cpu,&t,sizeof t); orki_bsync(fd,&c->mtk[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        struct rknpu_task *tall = (struct rknpu_task*)c->mtk_all.cpu;
        tall[i] = t;
    }
    int reg_amt = (c->last_dt == DT_I4) ? 116 : 108;
    struct rknpu_task *tall = (struct rknpu_task*)c->mtk_all.cpu;
    for(int i=0;i<nc;i++){
        struct rknpu_task *t = (struct rknpu_task*)c->mtk[i].cpu;
        if (t->regcfg_amount != reg_amt) {
            t->regcfg_amount = reg_amt;
            orki_bsync(fd,&c->mtk[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        }
        if (tall[i].regcfg_amount != reg_amt) {
            tall[i].regcfg_amount = reg_amt;
        }
    }
    orki_bsync(fd,&c->mtk_all,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    return 0;
}
