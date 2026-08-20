/* npu/core/domain.c — IOMMU domain activation/priming and the IOVA wedge guard.
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

int ork_dom_default(void){ static int v=-1; if(v<0){const char*e=getenv("ORK_IOMMU_DOMAIN"); v=e?atoi(e):0;} return v; }

size_t ork_iova_ceiling(void){
    static size_t v=0;
    if(!v){ const char*e=getenv("ORK_IOVA_CEIL_MB"); long mb=e?atol(e):3900; if(mb<=0)mb=3900; v=(size_t)mb*1024u*1024u; }
    return v;
}

int ork_iova_reserve(int dom,size_t need){
    if(dom<0||dom>=ORK_IOVA_NDOM) return 1;   /* out-of-range domain: untracked, allow (rare) */
    if(orki_iova_bytes[dom]+need > ork_iova_ceiling()){
        fprintf(stderr,"[ork] IOVA guard: domain %d at %zu MiB + %zu MiB would exceed the %zu MiB cap "
                "— refusing MEM_CREATE so the caller falls back (raise ORK_IOVA_CEIL_MB or spread domains)\n",
                dom, orki_iova_bytes[dom]>>20, need>>20, ork_iova_ceiling()>>20);
        return 0;
    }
    orki_iova_bytes[dom]+=need; return 1;
}

void ork_iova_release(int dom,size_t bytes){
    if(dom<0||dom>=ORK_IOVA_NDOM) return;
    orki_iova_bytes[dom] = orki_iova_bytes[dom]>bytes ? orki_iova_bytes[dom]-bytes : 0;
}

/* Grow the per-domain arrays (native anchor + parked scratch) to hold at least `need` domains. No fixed
 * cap — the domain count is whatever the auto-sizer / ork_npu_domain_alloc drives. dom_save is allocated
 * here (so it becomes non-NULL exactly when multi-domain is first entered, preserving the single-vs-multi
 * signal the run paths key on). Called from every multi-domain entry: dom_activate, ork_dom_prime,
 * ork_npu_domain_alloc, ork_npu_set_ndomains. Returns 0 ok, -1 on OOM (caller degrades gracefully). */
int orki_dom_reserve(ork_npu *c, int need){
    if(need <= c->dom_cap) return 0;
    int nc = c->dom_cap ? c->dom_cap : 2; while(nc < need) nc *= 2;
    struct buf *na = realloc(c->dom_anchor, (size_t)nc*sizeof *na);
    struct ork_dom_scratch *ns = realloc(c->dom_save, (size_t)nc*sizeof *ns);
    if(na) c->dom_anchor = na;
    if(ns) c->dom_save = ns;
    if(!na || !ns) return -1;   /* keep whatever grew; the guarded call sites won't index past dom_cap */
    memset(c->dom_anchor + c->dom_cap, 0, (size_t)(nc-c->dom_cap)*sizeof *c->dom_anchor);
    memset(c->dom_save   + c->dom_cap, 0, (size_t)(nc-c->dom_cap)*sizeof *c->dom_save);
    c->dom_cap = nc;
    return 0;
}

void ork_dom_prime(ork_npu *c, int dom){
    int d = ork_dom(dom);
    if(d<=0) return;                                  /* domain 0 never needs an anchor */
    if(orki_dom_reserve(c, d+1)) return;                   /* grow arrays to cover domain d (OOM -> skip anchoring) */
    if(c->dom_anchor[d].cpu) return;                  /* already anchored */
    /* #54 BIMPORT-ONLY (ORK_BIMPORT_DOM): native bcreate EINVALs in non-0 domains on this boot, so the anchor
     * must be a dma-heap import too. (Quirk 1's native-first requirement can't be honored when bcreate fails;
     * the weight import that follows is import-first regardless — the probe's bit-exact verify catches any
     * aliased-IOVA corruption.) */
    { static int bimp=-1; if(bimp<0) bimp=getenv("ORK_BIMPORT_DOM")?1:0;
      if(bimp){ c->dom_anchor[d] = orki_bimport(c->fd, 65536, d); return; } }
    c->dom_anchor[d] = orki_bcreate(c->fd, 65536, 0x403, d);   /* native bcreate == establishes the domain */
}

/* #54 FIX: re-establish a domain's IOMMU page-table region before EACH imported dma-buf. The kernel sets up a
 * domain's page table lazily around the buffer that triggers it; the one up-front anchor (ork_dom_prime) covers
 * only the FIRST import — a 2nd+ imported dma-buf then lands on aliased IOVAs and the NPU reads it WRONG (probed
 * bit-exact: 1st import OK, 2nd import maxerr~2835, fixed to 0 by a fresh native bcreate before it). So drop the
 * stale anchor and bcreate a fresh native one immediately before importing each weight. Cheap (64 KiB); the
 * previous import's mapping persists after its anchor is freed (verified: 1st weight re-runs bit-exact after). */
void ork_dom_reanchor(ork_npu *c, int dom){
    int d = ork_dom(dom); if(d<=0) return;            /* domain 0 always established */
    if(orki_dom_reserve(c, d+1)) return;
    if(c->dom_anchor[d].cpu) orki_bdestroy(c->fd, &c->dom_anchor[d]);
    { static int bimp=-1; if(bimp<0) bimp=getenv("ORK_BIMPORT_DOM")?1:0;
      if(bimp){ c->dom_anchor[d] = orki_bimport(c->fd, 65536, d); return; } }   /* #54 bimport-only: bcreate EINVALs in non-0 */
    c->dom_anchor[d] = orki_bcreate(c->fd, 65536, 0x403, d);
}

void ork_dom_flush_if_dirty(ork_npu *c){
    if(!c || !c->dom_dirty) return;
    if(getenv("ORK_MC_DIAG")) fprintf(stderr,"[dom] dirty-REAP on dom=%d (wait-past-timeout + timeout_clean the stuck job before switch)\n", c->dom_active);
    /* #54 COMBINED BOUNDARY FIX. The switch waits up to 6s for iommu_domain_refcount==0, so a 6s timeout means a
     * GENUINELY STUCK job (a clean-but-retiring job just delays the switch a few ms). rknpu_job_timeout_clean
     * (the only thing that reaps a stuck nonblock job — ACT_RESET can't) reaps ONLY a job aged >= its submit
     * timeout. So: (1) WAIT past the bounded int4 doorbell timeout (i4_submit_tmo_ms) so any dropped job is
     * reapable + any last in-flight op has time to retire; then (2) per-core reap dummy triggers timeout_clean.
     * Second reap pass catches a reap-dummy that itself dropped (its own short timeout has elapsed by then).
     * Only on a dirty (real-drop) boundary — rare — so the ~1.7s is paid only when a switch would otherwise wedge. */
    { int ms = orki_i4_submit_tmo_ms() + 200; struct timespec ts = {ms/1000, (long)(ms%1000)*1000000L}; nanosleep(&ts,NULL); }
    ork_npu_reap_stuck(c, c->soc->cores);
    { struct timespec ts = {0, 400*1000000L}; nanosleep(&ts,NULL); }   /* let a dropped reap-dummy pass its 300ms timeout */
    ork_npu_reap_stuck(c, c->soc->cores);
    c->dom_dirty=0;
}

void orki_dom_activate(ork_npu *c,int dom){
    if(dom<0) dom=0;
    if(dom==c->dom_active) return;
    /* #54 RETIREMENT BARRIER: a nonblock doorbell op is "done" when its output cell lands, which can be BEFORE
     * the kernel retires the job (completion IRQ). Switching the IOMMU domain while the PRIOR domain's job is
     * still in-flight races it -> the first submit in the new domain dispatches nothing (task counter 0x0) ->
     * kernel watchdog soft reset -> corrupts the switch (the int4 multi-domain wedge; byte-diff ruled out a
     * malformed descriptor, dom0/dom1 submits are byte-identical + run fine). Let the prior domain's just-landed
     * job retire before switching. Only on a real switch (per-layer, ~tens of times); tunable/off via env. */
    { static long su=-1; if(su<0){ const char*e=getenv("ORK_DOM_SETTLE_US"); su=e?atol(e):1000; }
      if(su>0){ struct timespec ts={su/1000000,(su%1000000)*1000}; nanosleep(&ts,NULL); } }
    ork_dom_flush_if_dirty(c);   /* #54 THE multi-domain fix: if an int4 doorbell drop left a stuck job in the OUTGOING domain, REAP it now (same-domain timeout_clean, still attached to dom_active) so the switch below finds the domain idle instead of timing out -> cascade. See ork_dom_flush_if_dirty. */
    double _sw_t0 = ork_now_us();                 /* ORK_DOM_PROFILE: real-switch swap cost */
    if(orki_dom_reserve(c, (dom>c->dom_active?dom:c->dom_active)+1)) return;   /* grow arrays to cover both old + new domain (also allocates dom_save on first multi-domain use); OOM -> skip */
    struct ork_dom_scratch *old=&c->dom_save[c->dom_active], *neo=&c->dom_save[dom];
    /* park active -> old */
    old->used=1; old->regcmd=c->regcmd; old->task=c->task; old->Af=c->Af; old->Cc=c->Cc; old->ccsz=c->ccsz; old->warmed=c->warmed;
    memcpy(old->mrc,c->mrc,sizeof old->mrc); memcpy(old->mtk,c->mtk,sizeof old->mtk); memcpy(old->maf,c->maf,sizeof old->maf);
    memcpy(old->mcc,c->mcc,sizeof old->mcc); old->mtk_all=c->mtk_all; memcpy(old->mccsz,c->mccsz,sizeof old->mccsz);
    memcpy(old->mwarm,c->mwarm,sizeof old->mwarm); old->mc_alloc=c->mc_alloc;
    /* restore neo -> active (zeroed if first use → lazy alloc in this domain) */
    c->regcmd=neo->regcmd; c->task=neo->task; c->Af=neo->Af; c->Cc=neo->Cc; c->ccsz=neo->ccsz; c->warmed=neo->warmed;
    memcpy(c->mrc,neo->mrc,sizeof c->mrc); memcpy(c->mtk,neo->mtk,sizeof c->mtk); memcpy(c->maf,neo->maf,sizeof c->maf);
    memcpy(c->mcc,neo->mcc,sizeof c->mcc); c->mtk_all=neo->mtk_all; memcpy(c->mccsz,neo->mccsz,sizeof c->mccsz);
    memcpy(c->mwarm,neo->mwarm,sizeof c->mwarm); c->mc_alloc=neo->mc_alloc;
    c->dom_active=dom;
    /* first time we touch domain `dom`: it has none of the init-time scratch yet. Mirror ork_npu_init
     * EXACTLY — allocate regcmd+task+Af in THIS domain and seed the task descriptor — so a freshly-activated
     * domain carries every buffer ork_npu_init guarantees, none NULL and none a stale domain-0 IOVA. regcmd
     * and task have NO lazy size-guard (the chain path writes c->regcmd.cpu / c->task.cpu directly), so they
     * MUST be created here. Af is created here too so first-use == init: the run path's `c->Af.size<maxaf`
     * realloc already covers it, but the unguarded RE/probe entrypoints (ork_npu_probe_*) read c->Af.cpu
     * raw, and a non-NULL in-domain Af is the safe, complete invariant rather than relying on each caller.
     * Cc and the per-core mc-* scratch are NOT allocated here: their sizes are matmul-dependent, so every
     * run path lazily (re)allocates them in c->dom_active under their own .size/.cpu guards. */
    int _first = (!neo->used && !c->regcmd.cpu);
    if(_first){
        c->regcmd=orki_bscratch(c,2097152,0x403,dom); c->task=orki_bscratch(c,524288,0x40b,dom); c->Af=orki_bscratch(c,(size_t)4*32768*2,0x403,dom);
        if(c->task.cpu){ struct rknpu_task t; memset(&t,0,sizeof t); t.enable_mask=0xd;t.int_mask=0x300;t.int_clear=0x1ffff;t.regcfg_amount=108;t.regcmd_addr=c->regcmd.dma;
            memcpy(c->task.cpu,&t,sizeof t); orki_bsync(c->fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); }
    }
    { double _dt = ork_now_us() - _sw_t0;         /* ORK_DOM_PROFILE accounting: total, max, and first-touch split */
      c->dom_sw_n++; c->dom_sw_us += _dt; if(_dt > c->dom_sw_max_us) c->dom_sw_max_us = _dt;
      if(_first){ c->dom_sw_first_n++; c->dom_sw_first_us += _dt; } }
}
