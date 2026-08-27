/* npu/core/submit.c — the raw DRM submit path: action ioctls, regcmd tracing, the submit ioctl wrapper.
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
#include <dlfcn.h>
#include "npu/internal.h"
#include "npu/core.h"
#include "npu/core/core.h"

void orki_setrn(uint32_t*rc,int n,enum ork_reg_id id,uint32_t v){
    const ork_reg_desc *d=&ORK_REGS[id];
    if(v & ~d->mask) fprintf(stderr,"[ork] WARN reg %s (%04x:%04x): value %#x has bits outside field mask %#x\n",d->name,d->blk,d->off,v,d->mask);
    orki_setr(rc,n,d->blk,d->off,v);
}


void orki_act(int fd,uint32_t f,uint32_t v){
    if(f==RKNPU_ACT_RESET){ static long n=0; if(getenv("ORK_DEBUG_RESET")){ void*ra=__builtin_return_address(0); Dl_info di;
        if(dladdr(ra,&di)) fprintf(stderr,"[ork] ACT_RESET #%ld off=0x%lx obj=%s\n",++n,(unsigned long)((char*)ra-(char*)di.dli_fbase),di.dli_fname);
        else fprintf(stderr,"[ork] ACT_RESET #%ld ra=%p\n",++n,ra); } }
    struct rknpu_action a={.flags=f,.value=v};ioctl(fd,DRM_IOCTL_RKNPU_ACTION,&a);}

void orki_dump_submit(struct rknpu_submit *sub) {
    fprintf(stderr, "[ork-trace] === SUBMIT flags=0x%x timeout=%u task_number=%u core=0x%x domain=%u ===\n",
            sub->flags, sub->timeout, sub->task_number, sub->core_mask, sub->iommu_domain_id);

    if (!orki_npu_ctx) return;
    
    void *task_cpu = NULL;
    if (orki_npu_ctx->task.obj == sub->task_obj_addr) {
        task_cpu = orki_npu_ctx->task.cpu;
    } else if (orki_npu_ctx->mtk_all.obj == sub->task_obj_addr) {
        task_cpu = orki_npu_ctx->mtk_all.cpu;
    } else {
        for (int i = 0; i < ORK_MAXCORE; i++) {
            if (orki_npu_ctx->mtk[i].obj == sub->task_obj_addr) {
                task_cpu = orki_npu_ctx->mtk[i].cpu;
                break;
            }
        }
    }
    
    if (!task_cpu) {
        fprintf(stderr, "  (task buffer not found/mapped)\n");
        return;
    }
    
    struct rknpu_task *tasks = (struct rknpu_task *)task_cpu;
    for (uint32_t i = 0; i < sub->task_number; i++) {
        struct rknpu_task *t = &tasks[i];
        fprintf(stderr, "  task[%u]: flags=0x%x op_idx=%u enable=0x%x int_mask=0x%x regcfg_amount=%u regcmd_addr=0x%llx\n",
                i, t->flags, t->op_idx, t->enable_mask, t->int_mask, t->regcfg_amount, (unsigned long long)t->regcmd_addr);
        
        void *regcmd_cpu = NULL;
        uint64_t dma_base = 0;
        if (t->regcmd_addr >= orki_npu_ctx->regcmd.dma && t->regcmd_addr < orki_npu_ctx->regcmd.dma + orki_npu_ctx->regcmd.size) {
            regcmd_cpu = orki_npu_ctx->regcmd.cpu;
            dma_base = orki_npu_ctx->regcmd.dma;
        } else {
            for (int j = 0; j < ORK_MAXCORE; j++) {
                if (t->regcmd_addr >= orki_npu_ctx->mrc[j].dma && t->regcmd_addr < orki_npu_ctx->mrc[j].dma + orki_npu_ctx->mrc[j].size) {
                    regcmd_cpu = orki_npu_ctx->mrc[j].cpu;
                    dma_base = orki_npu_ctx->mrc[j].dma;
                    break;
                }
            }
        }
        
        if (regcmd_cpu) {
            uint64_t offset = t->regcmd_addr - dma_base;
            uint32_t *rc = (uint32_t *)((char *)regcmd_cpu + offset);
            int n_words = (int)t->regcfg_amount * 2 + 16;
            fprintf(stderr, "  --- regcmd (%d u32 words) ---\n", n_words);
            for (int k = 0; k < n_words; k += 4) {
                fprintf(stderr, "  [%03d] %08x %08x %08x %08x\n", k,
                        rc[k], (k+1 < n_words)?rc[k+1]:0, (k+2 < n_words)?rc[k+2]:0, (k+3 < n_words)?rc[k+3]:0);
            }
        } else {
            fprintf(stderr, "  (regcmd buffer not found for dma=0x%llx)\n", (unsigned long long)t->regcmd_addr);
        }
    }
}

void orki_trace_submit(struct rknpu_submit *sub) { if (getenv("ORK_TRACE")) orki_dump_submit(sub); }

/* Doorbell DENOMINATOR (diagnostic). The registry records the mc doorbell's dropped round as a
 * "~1/4000 dispatch-drop"; nothing counted the submits, so that rate was never checkable against a real
 * workload. Count both submits and PROGRAMS (task_number) -- a chained submit carries many programs, and
 * the per-program rate is the one comparable to 1/4000. */
unsigned long orki_submit_n = 0, orki_submit_prog = 0;
/* #patch (experiment): monotonic stamp written into rknpu_task.op_idx — a field the uABI declares
 * but the driver NEVER reads. The kernel logs the value it sees at commit; if a STALLED commit reports
 * an OLDER stamp than userspace just wrote, the kernel read a STALE task array (its kernel mapping did
 * not see our write) and programmed the PC from a stale descriptor. That is decisive, and needs no timing. */
unsigned long orki_task_seq = 0;

/* ORK_KMSG_SUBMIT=1 — stamp every submit into /dev/kmsg so USERSPACE submits and the KERNEL's watchdog
 * reports land in the SAME dmesg stream, in order. Without this the two halves of a dispatch failure cannot
 * be lined up: userspace knows the op/shape/domain, the kernel knows the core and the PC task counter, and
 * nothing links them but wall-clock eyeballing.
 *
 * Correlate like this:
 *   ork: sub#1871 dom=6 cm=0x2 tasks=21 fl=0x3 rc=0        <- this submit
 *   RKNPU: core 1 NO TASK PROGRESS ... task counter stuck at 0x0, job age 115001 us
 * `cm` is a MASK (bit i = core i), so it names which core the kernel then reports on.
 *
 * Uses a CACHED fd rather than ork_kmsg(), which open()s and close()s per call: three syscalls on the
 * submit hot path is ~5% of the 167 us dispatch floor, and this is a dispatch-TIMING defect — a heavy probe
 * would perturb the thing being measured. Off by default for the same reason. */
static void orki_submit_stamp(const struct rknpu_submit *sub, int domain, int rc){
    static int kfd = -2;
    if (kfd == -2) { kfd = getenv("ORK_KMSG_SUBMIT") ? open("/dev/kmsg", O_WRONLY|O_CLOEXEC) : -1; }
    if (kfd < 0 || !sub) return;
    char b[160];
    int n = snprintf(b, sizeof b, "<4>ork: sub#%lu dom=%d cm=0x%x tasks=%u fl=0x%x rc=%d\n",
                     orki_submit_n, domain, sub->core_mask, sub->task_number, sub->flags, rc);
    if (n > 0) { ssize_t w = write(kfd, b, (size_t)n < sizeof b ? (size_t)n : sizeof b - 1); (void)w; }
}

/* ORK_RC_VERIFY=1 (DIAGNOSTIC): did our regcmd writes actually reach DRAM, i.e. are the bytes the NPU's
 * DMA will fetch the bytes we wrote? Everything else in the dispatch chain is now exonerated (registers
 * byte-identical healthy vs stalled, zero IOMMU faults, task descriptors freshly stamped), and this is the
 * one link never verified: the regcmd is CACHEABLE DRAM with no kernel mapping, so nothing has ever read
 * back what the device sees.
 *
 * Method needs no shadow copy and no kernel change: checksum the buffer as the CPU sees it (possibly from
 * cache), then invalidate with a FROM_DEVICE sync so the next reads must come from DRAM, and checksum
 * again. A mismatch means the TO_DEVICE flush did not land and the NPU would fetch stale bytes.
 *
 * NOTE this cannot be done by mapping the buffer uncached: dropping RKNPU_MEM_CACHEABLE gets it mapped as
 * Device memory, and ARM64 faults on unaligned access to Device memory, so the regcmd memcpy SIGBUSes. */
static uint64_t orki_rc_sum(const void *p, size_t n){
    const uint32_t *w = (const uint32_t*)p; uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n/4; i++) { h ^= w[i]; h *= 1099511628211ULL; }
    return h;
}
unsigned long orki_rc_verify_n, orki_rc_verify_bad;
int orki_rc_verify(int fd, struct buf *b, size_t nbytes){
    static int on = -1; if (on < 0) { const char *e = getenv("ORK_RC_VERIFY"); on = e ? atoi(e) : 0; }
    if (!on || !b || !b->cpu || nbytes == 0) return 0;
    if (nbytes > b->size) nbytes = b->size;
    uint64_t c1 = orki_rc_sum(b->cpu, nbytes);
    orki_bsync(fd, b, RKNPU_MEM_SYNC_FROM_DEVICE);      /* invalidate -> next reads come from DRAM */
    /* POSITIVE CONTROL (ORK_RC_VERIFY=2): corrupt one word AFTER the flush so the comparison MUST fail.
     * "No mismatches" is worthless unless the detector is shown to be capable of firing — the same missing
     * -control mistake that produced two false readings earlier in this investigation. */
    if (on == 2 && orki_rc_verify_n == 0) ((volatile uint32_t*)b->cpu)[0] ^= 0xa5a5a5a5u;
    uint64_t c2 = orki_rc_sum(b->cpu, nbytes);
    if (orki_rc_verify_n == 0)
        fprintf(stderr, "[rc-verify] ACTIVE: first check, %zu bytes, cache=%016llx dram=%016llx%s\n",
                nbytes, (unsigned long long)c1, (unsigned long long)c2,
                on == 2 ? "  (positive control: MUST differ)" : "");
    orki_rc_verify_n++;
    if (c1 != c2) { orki_rc_verify_bad++;
        fprintf(stderr, "[rc-verify] REGCMD MISMATCH after flush: cache=%016llx dram=%016llx bytes=%zu\n",
                (unsigned long long)c1, (unsigned long long)c2, nbytes); }
    return c1 != c2;
}

int orki_rknpu_submit_ioctl(int fd, struct rknpu_submit *sub, int domain) {
    { static int st=-1; static int seq=0; if(st<0) st = getenv("ORK_SUBTRACE") ? 1 : 0;
      if(st) fprintf(stderr,"[subtrace %d] op=%s dom=%d core=0x%x tn=%u caller=%p\n",
                     ++seq, orki_last_op?orki_last_op:"?", domain, sub->core_mask,
                     sub->task_number, __builtin_return_address(0)); }
    orki_submit_n++; orki_submit_prog += sub ? (unsigned long)sub->task_number : 0;
    /* OFFLINE GUARD. ork_npu_init_offline hands out a context with fd = -1 for CPU-side work (pack building
     * and scoring on a machine with no NPU). The paths that support it intercept BEFORE here and never
     * reach a submit; anything that arrives HERE with fd < 0 is a path that was never taught about offline.
     *
     * It must fail LOUDLY. Measured 2026-08-23: a selective-precision experiment routed six weights through
     * an int8 dispatch variant with no offline branch; the submit went nowhere, the caller saw success, and
     * the run produced a PLAUSIBLE-BUT-WRONG perplexity (+15.5% where the board says -0.86%). A silently
     * wrong number is worse than a crash — it gets believed, and it cost a full round of analysis before
     * the board contradicted it. So: one clear message naming the op, and a hard error the caller cannot
     * mistake for a completed submit. */
    if (fd < 0) {
        static int warned = 0;
        if (!warned) { warned = 1;
            fprintf(stderr,
                "[ork] ERROR: submit on an OFFLINE context (op=%s K=%d N=%d). This path has no offline\n"
                "[ork]        implementation — its result would be UNCOMPUTED, not merely slow. Offline\n"
                "[ork]        supports the CPU-side surfaces only (i4/i8 pack, run, run_chain, run_stream,\n"
                "[ork]        run_grouped, w_dump, mm_load). Run this on the board, or add an offline\n"
                "[ork]        branch to the entry point that reached here.\n",
                orki_last_op ? orki_last_op : "(unset)", orki_last_K, orki_last_N);
        }
        errno = ENODEV;
        return -1;
    }
    sub->iommu_domain_id = ork_dom(domain);  /* match the domain the weight's resident tiles live in (threaded per-call, not a global) */
    if (orki_ork_prof) { orki_prof_submits++; orki_prof_submit_progs += sub->task_number; if (sub->task_number > 1) orki_prof_submit_chained++; }
    orki_trace_submit(sub);
    /* PRE-SUBMIT fsync'd trace (ORK_PRESUBMIT_TRACE=<path>): write this submit's full context to disk AND
     * fsync it BEFORE the ioctl — so if the ioctl HARD-WEDGES the NPU/board (no self-heal, needs power-cycle),
     * the last line on disk is the exact submit that wedged. Survives the wedge (unlike stderr/page-cache logs
     * lost on the lockup). One line/submit; the tail after a wedge pins op/weight/domain/tasks/addrs. */
    if (getenv("ORK_PRESUBMIT_TRACE")) {
        static FILE *tf=NULL; static long sn=0;
        if(!tf){ tf=fopen(getenv("ORK_PRESUBMIT_TRACE"),"w"); }
        if(tf){ size_t iov=0; for(int d=0;d<ORK_IOVA_NDOM;d++) iov+=orki_iova_bytes[d];
            fprintf(tf,"#%ld op=%s K=%d N=%d wdom=%d imported=%d | submit_dom=%u tasks=%u core=0x%x | iova_total=%zuMiB dom[0]=%zu dom[1]=%zu dom[2]=%zu | crt=%ld imp=%ld dst=%ld live=%ld\n",
                    ++sn, orki_last_op, orki_last_K, orki_last_N, orki_last_wdom, orki_last_import,
                    sub->iommu_domain_id, sub->task_number, sub->core_mask, iov>>20,
                    orki_iova_bytes[0]>>20, orki_iova_bytes[1]>>20, orki_iova_bytes[2]>>20,
                    orki_bcreate_n, orki_bimport_n, orki_bdestroy_n, orki_bcreate_n+orki_bimport_n-orki_bdestroy_n);
            fflush(tf); fsync(fileno(tf)); }
    }
    int _nb = getenv("ORK_JOB_NONBLOCK")!=NULL;   /* TEST: RKNPU_JOB_NONBLOCK (1<<1=0x2) — does SUBMIT return before completion? */
    if(_nb) sub->flags |= 0x2;
    double _fd_t0 = ork_now_us();
    int rc = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, sub);
    orki_submit_stamp(sub, domain, rc);   /* ORK_KMSG_SUBMIT: interleave with the kernel watchdog in dmesg */
    { double _fd_dt = ork_now_us() - _fd_t0; orki_fd_ioctl_us += _fd_dt; orki_fd_hw_raw_last = (long long)sub->hw_elapse_time;
      orki_fd_hw_us += (double)sub->hw_elapse_time; orki_fd_n++;
      if(_nb){ int async=(rc==0 && _fd_dt<50.0);  /* returned in <50us => before the HW could run => async */
               fprintf(stderr,"[nonblock] flags=0x%x rc=%d ioctl-return=%.0fus hw_elapse=%lldus%s\n",
                   sub->flags, rc, _fd_dt, (long long)sub->hw_elapse_time,
                   async?"  <- ASYNC (RKNPU_JOB_NONBLOCK works: submit returned before HW done)":"  (blocked till done)");
               { const char*se=getenv("ORK_NONBLOCK_SLEEP_US"); unsigned su=se?(unsigned)strtoul(se,0,0):300000; if(su)usleep(su); } } }
               /* drain sleep configurable (default 300ms safety). Set SMALL (< compute) to force job OVERLAP and test
                * whether a 2nd NONBLOCK submit QUEUES (returns ~5us) or SERIALIZES (blocks ~compute) on the busy NPU. */
    if (rc < 0) {
        int e = errno;
        /* TRANSIENT SUBMIT-REJECTION RETRY (before the heavy self-heal reset). The doorbell is per-op NONBLOCK,
         * so a submit to a core whose PRIOR job has not yet retired (its completion IRQ hasn't fired) is rejected
         * EINVAL/EBUSY — a GENERAL, pre-existing in-suite race (surfaces on any doorbell user under back-to-back
         * ops: run_stream, run_i8 colsplit, fp16 bmm). The prior job retires within microseconds, so re-issue the
         * SAME regcmd/task a few times with a short backoff; a retry that lands needs NO reset (a reset clears
         * warm -> cold miscompute) and the output still lands. Only fall through to the reset if it stays wedged. */
        if (e == EINVAL || e == EBUSY) {
            for (int _r = 0; _r < 20 && rc < 0; _r++) { struct timespec _ts = {0, 50000}; nanosleep(&_ts, NULL);
                rc = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, sub); }
            if (rc >= 0) return rc;   /* retired + landed */
            errno = e;
        }
        fprintf(stderr, "[ork] WARNING: RKNPU_SUBMIT ioctl failed (rc=%d, errno=%d) | submit domain=%u task_number=%u core=0x%x | last regcmd op=%s weight[K=%d N=%d dom=%d imported=%d]. Triggering self-healing reset...\n",
                rc, e, sub->iommu_domain_id, sub->task_number, sub->core_mask,
                orki_last_op, orki_last_K, orki_last_N, orki_last_wdom, orki_last_import);
        /* live per-domain IOVA usage AT THE FAILURE POINT: shows whether the domain overflowed (size/pressure)
         * vs a non-size DMA-walk stall. orki_iova_bytes counts resident tiles + imports + transient scratch. */
        { size_t tot=0; for(int d=0; d<ORK_IOVA_NDOM; d++) if(orki_iova_bytes[d]){ fprintf(stderr,"  [iova@fail] domain %d live=%zu MiB (ceil %zu MiB)\n", d, orki_iova_bytes[d]>>20, ork_iova_ceiling()>>20); tot+=orki_iova_bytes[d]; }
          fprintf(stderr,"  [iova@fail] total live=%zu MiB\n", tot>>20); }
        if (getenv("ORK_DUMP_FAIL")) orki_dump_submit(sub);   /* full failing regcmd on demand */
        struct rknpu_action a = { .flags = RKNPU_ACT_RESET, .value = 0 };
        ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &a);
        errno = e;
    }
    return rc;
}

uint32_t ork_ppflags(void);     /* fwd (defined below) */
/* #39 WEIGHT-RESIDENT M-FOLD CHAIN (task_number=P). Each of P tasks is a width-`w` mfold tile built by the
 * validated orki_i8_synth_mfold (bit-exact standalone at w=36); the K*N weight is loaded ONCE and shared across all
 * P tasks (HW PC-chain), amortizing the weight-DMA over M=P*w rows. Chain descriptor written at words 216-219
 * EXACTLY like the PROVEN run_chain_i8/mcworker_pref_chain; terminal task clears them. subcore_task[0..2] all
 * populated (single-core), matching run_chain_i8. Apacked = P tiles of w-row C2-16 A (w*K bytes each); Bpacked =
 * shared ork_woff weight; Craw = P tiles of raw C2-4 int32 (w*N int32 each). Board only; returns 0/ok, us=avg. */
int ork_npu_mfold_chain(ork_npu *c, int P, int w, int K, int N,
                        const int8_t *Apacked, const int8_t *Bpacked, int32_t *Craw, int iters, double *us){
    int fd=c->fd; if(fd<0) return -3; if(P<1||P>64||w<1||w>64||(K%32)||(N%16)) return -2;
    int dom=c->dom_active;
    size_t tileA=(size_t)w*K, tileC=(size_t)w*N;                 /* per-tile A bytes (C2-16), C int32 elems (C2-4) */
    /* 8x guard rig (like the replay path): a malformed mfold DMA that over-reads/writes lands in MAPPED memory
     * (diagnosable) instead of hanging the AXI/IOMMU bus (errno-110). */
    size_t asz=(size_t)P*tileA*8+(1u<<20), bsz=(size_t)K*N*8+(1u<<20), csz=(size_t)P*tileC*4*8+65536;
    struct buf A =orki_bcreate(fd,asz,0x403,dom);            if(!A.cpu)  return -2;
    struct buf B =orki_bcreate(fd,bsz,0x403,dom);            if(!B.cpu) {orki_bdestroy(fd,&A);return -2;}
    struct buf Cc=orki_bcreate(fd,csz,0x403,dom);            if(!Cc.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);return -2;}
    struct buf RC=orki_bcreate(fd,(size_t)P*REGCMD_I8_N*4,0x403,dom); if(!RC.cpu){orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&Cc);return -2;}
    memset(A.cpu,0,asz); memset(B.cpu,0,bsz); memset(Cc.cpu,0,csz);
    memcpy(A.cpu,Apacked,(size_t)P*tileA); memcpy(B.cpu,Bpacked,(size_t)K*N);
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t *rcbuf=(uint32_t*)RC.cpu;
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu;   /* P-task array (c->task is 512KB / flag 0x40b) */
    for(int t=0;t<P;t++){
        uint32_t rc[REGCMD_I8_N];
        orki_i8_synth_mfold(rc, w, K, N, 0x1000000u, 0x2000000u, 0x3000000u, c->soc->cbuf_elems);
        orki_setrn(rc,REGCMD_I8_N,RK_CNA_FEATURE_DATA_ADDR,(uint32_t)(A.dma + (uint64_t)t*tileA));   /* this tile's A */
        orki_setrn(rc,REGCMD_I8_N,RK_CNA_WEIGHT_DATA_ADDR,(uint32_t)B.dma);                          /* shared weight */
        orki_setrn(rc,REGCMD_I8_N,RK_DPU_DST_BASE_ADDR,(uint32_t)(Cc.dma + (uint64_t)t*tileC*4));     /* this tile's C */
        /* chain-link at fixed words 216-219, PROVEN run_chain_i8/mcworker_pref_chain encoding. Terminal clears. */
        if(t<P-1){ uint64_t nxt = RC.dma + (uint64_t)(t+1)*REGCMD_I8_N*4;
            rc[216]=0x0010|((uint32_t)(nxt&0xffff)<<16); rc[217]=(0x0101u<<16)|((uint32_t)(nxt>>16)&0xffff);
            rc[218]=0x0014|(0x0037u<<16);                rc[219]=(0x0101u<<16)|0;
        } else { rc[216]=rc[217]=rc[218]=rc[219]=0; }
        memcpy(rcbuf + (size_t)t*REGCMD_I8_N, rc, sizeof rc);
        memset(&tk[t],0,sizeof tk[t]);
        tk[t].enable_mask=0xd; tk[t].int_mask=0x300; tk[t].int_clear=0x1ffff;
        tk[t].regcfg_amount=108; tk[t].regcmd_addr=RC.dma + (uint64_t)t*REGCMD_I8_N*4;
    }
    orki_bsync(fd,&RC,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    int ret=-1; struct rknpu_submit sub;
    #define _CSUB() do{ memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=(uint32_t)P; \
        sub.task_obj_addr=c->task.obj; sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1; sub.timeout=orki_mm_timeout_ms(); \
        sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)P}; }while(0)
    _CSUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ goto cdone; }        /* warm */
    orki_bsync(fd,&Cc,RKNPU_MEM_SYNC_FROM_DEVICE);
    if(Craw) memcpy(Craw,Cc.cpu,(size_t)P*tileC*4);
    { double t0=ork_now_us();
      for(int i=0;i<iters;i++){ _CSUB(); if(orki_rknpu_submit_ioctl(fd,&sub,dom)){ goto cdone; } }
      if(us)*us=(ork_now_us()-t0)/(iters>0?iters:1); ret=0; }
    #undef _CSUB
cdone:
    orki_bdestroy(fd,&A);orki_bdestroy(fd,&B);orki_bdestroy(fd,&Cc);orki_bdestroy(fd,&RC); return ret;
}

/* PROBE HOOKS (percore_mode_probe). ORK_NPU_TESTCORE is cached in a static, so a probe cannot alternate
 * cores within one process — and "is the precision-mode hazard per-core or device-global?" can only be
 * asked by interleaving two precisions on two cores in ONE process. Expose the core as settable state.
 * Default is unchanged: the env var, read once, clamped to 0..2. */
int orki_testcore = -1;
int orki_tc(void){ if(orki_testcore<0){ const char*e=getenv("ORK_NPU_TESTCORE"); int t=e?atoi(e):0; if(t<0||t>2)t=0; orki_testcore=t; } return orki_testcore; }
void ork_npu_set_test_core(int core){ if(core>=0&&core<=2) orki_testcore=core; }

/* Cold-buffer warmup depth. ORK_WARM_REPS overrides for A/B. See orki_submit1(). */
int orki_warm_reps(void){ static int r=-1; if(r<0){const char*e=getenv("ORK_WARM_REPS"); r=e?atoi(e):2; if(r<1)r=1;} return r; }

int orki_submit1(ork_npu *c){
    int fd=c->fd;
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.fence_fd=-1;
    sub.core_mask=1u<<orki_tc();
    sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,1};
    /* first submit on a fresh output buffer returns stale (NPU primed against wedging by the
     * RKNPU_ACT_RESET); run one throwaway warmup with a short timeout, then the real submit. */
    int reps=c->warmed?1:orki_warm_reps();
    for(int rep=0;rep<reps;rep++){ int last=(rep==reps-1); sub.timeout=orki_mm_timeout_ms();
        if(orki_rknpu_submit_ioctl(fd,&sub,c->dom_active)){ if(last){perror("SUBMIT");return -1;} continue; }
        orki_bsync(fd,&c->Cc,RKNPU_MEM_SYNC_FROM_DEVICE); }
    c->warmed=1; return 0;
}

int orki_submit1_db(ork_npu *c, size_t nout){
    int fd=c->fd;
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags()|0x2u;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.fence_fd=-1;
    sub.core_mask=1u<<orki_tc();
    sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,1};
    volatile int32_t *o=(volatile int32_t*)c->Cc.cpu; size_t li=nout-1;
    int reps=c->warmed?1:orki_warm_reps();
    for(int rep=0;rep<reps;rep++){ int last=(rep==reps-1); sub.timeout=orki_mm_timeout_ms();
        o[li]=0x7fffffff; __asm__ volatile("dc cvac,%0"::"r"(&o[li]):"memory"); __asm__ volatile("dsb ish":::"memory");   /* seed the last-word sentinel (matmul writes it last) */
        if(orki_rknpu_submit_ioctl(fd,&sub,c->dom_active)){ if(last){perror("SUBMIT"); return -1;} continue; }
        double pt=ork_now_us(), cap=(double)orki_mm_timeout_ms()*1000.0;
        for(;;){ __asm__ volatile("dc civac,%0"::"r"(&o[li]):"memory"); if(o[li]!=0x7fffffff)break; if(ork_now_us()-pt>cap)break; }   /* last-col-last writeback => last word landing = tile done */
        orki_bsync(fd,&c->Cc,RKNPU_MEM_SYNC_FROM_DEVICE); }
    c->warmed=1; return 0;
}
int orki_check_overlap(const char *name, uintptr_t a_start, uintptr_t a_end, uintptr_t c_start, uintptr_t c_end) {
    if (a_start < c_end && c_start < a_end) {
        fprintf(stderr, "[ork] ERROR [%s]: memory overlap detected! A [%p, %p) overlaps with C [%p, %p).\n",
                name, (void*)a_start, (void*)a_end, (void*)c_start, (void*)c_end);
        return 1;
    }
    return 0;
}

int orki_validate_regcmd(const char *op, ork_npu *c, const uint32_t *rc, int n, const ork_w *w, const struct buf *extra, int extra_n) {
    /* stash context so a later submit failure can name the exact weight/op/domain/import-status that faulted */
    orki_last_op = op ? op : "?";
    if (w) { orki_last_K = w->K; orki_last_N = w->N; orki_last_wdom = w->domain;
             orki_last_import = (w->own_buf_valid && w->own_buf.heap_fd > 0) ||
                             (w->own_bufs && w->n_own_bufs > 0 && w->own_bufs[0].heap_fd > 0) ||
                             (w->Bb && w->Bb[0].heap_fd > 0) || (w->Bf && w->Bf[0].heap_fd > 0); }
    for (int k = 0; k + 1 < n; k += 2) {
        uint32_t offset = rc[k] & 0xffff;
        uint32_t block_id = rc[k+1] >> 16;
        uint32_t val = (rc[k] >> 16) | ((rc[k+1] & 0xffff) << 16);
        const char *reg_name = NULL;
        if (offset == 0x1070 && block_id == 0x201) reg_name = "adma";
        else if (offset == 0x1110 && block_id == 0x201) reg_name = "bdma";
        else if (offset == 0x4020 && block_id == 0x1001) reg_name = "cdma";
        if (reg_name) {
            if (val == 0) {
                fprintf(stderr, "[ork] ERROR [%s]: regcmd sanity assertion failed! %s is NULL (0x00000000).\n", op, reg_name);
                return -1;
            }
            if ((val & 15) != 0) {
                fprintf(stderr, "[ork] ERROR [%s]: regcmd sanity assertion failed! %s address 0x%08x is not 16-byte aligned.\n", op, reg_name, val);
                return -1;
            }
            if (!orki_is_valid_dma_addr(c, val, w, extra, extra_n)) {
                fprintf(stderr, "[ork] ERROR [%s]: regcmd sanity assertion failed! %s address 0x%08x is wild/unallocated (outside all valid buffers).\n", op, reg_name, val);
                return -1;
            }
        }
    }
    return 0;
}
