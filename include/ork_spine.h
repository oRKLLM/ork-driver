/* ork_spine.h — heterogeneous CPU<->NPU execution-unit dispatcher (header-only).
 *
 * Drives an op DAG across execution UNITS — an NPU unit (the ork_dyn doorbell, on its OWN worker thread so the
 * NPU stays single-stream) plus one or more CPU worker units (pinned to spare big cores) — from a SINGLE poll
 * loop. Each op declares its dependencies (a bitmask of earlier op indices) and a placement (NPU / CPU / EITHER);
 * the scheduler dispatches every ready op to a free matching unit, so independent NPU and CPU work OVERLAPS and a
 * dependent op waits for its producer. This is general infrastructure (prefill CPU-quant || NPU-matmul, mixed
 * CPU/NPU op-graphs, bridging the IOMMU domain-swap latency, spare-core use) — NOT decode-specific.
 * Validated on RK3588 by tools/spine_sched_probe.c and examples/test_spine.c.
 *
 * CROSS-UNIT COHERENCY (hard rule): an NPU-produced dma buffer read by a CPU unit on ANOTHER thread needs a
 * `dc civac` on BOTH sides — the NPU op fn must ork_spine_civac_range() its output AFTER ork_dyn_end (producer
 * flush), and the consumer must ork_spine_civac_range() the region BEFORE reading. Read-after-drain is coherent;
 * writing INTO a device-owned buffer after ork_dyn_end is NOT (see doorbell_overlap_probe). Same-thread reads
 * are always coherent.
 *
 * SINGLE-STREAM INVARIANT: at most ONE unit may be ORK_UNIT_NPU (the doorbell cannot be driven concurrently);
 * ork_spine_run() rejects a config with more. CPU units: as many as you have spare big cores.
 *
 * DEPS: ops[i].deps is a bitmask of op indices (1<<j) that must complete before op i — so at most 32 ops per DAG.
 *
 * Charter: C11, header-only, no deps beyond libc + pthreads (as npu.c). Usable from C and C++.
 * INCLUDER MUST define _GNU_SOURCE before the first libc include (for cpu_set_t / pthread_setaffinity_np); the
 * guard below covers the case where this header is included first.
 */
#ifndef ORK_SPINE_H
#define ORK_SPINE_H
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <sched.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <stddef.h>

/* dc civac = data-cache clean+invalidate to point-of-coherency (the cross-unit handoff bsync). */
static inline void ork_spine_civac1(volatile void *p){ __asm__ volatile("dc civac,%0" :: "r"(p) : "memory"); }
static inline void ork_spine_civac_range(const void *b, size_t n){
    for (size_t o = 0; o < n; o += 64) ork_spine_civac1((char *)b + o);
    __asm__ volatile("dsb ish" ::: "memory");
}

enum { ORK_PL_NPU = 0, ORK_PL_CPU = 1, ORK_PL_EITHER = 2 };   /* op placement */
enum { ORK_UNIT_CPU = 0, ORK_UNIT_NPU = 1 };                  /* unit kind    */

typedef struct ork_spine_unit {
    pthread_t th; pthread_mutex_t mu; pthread_cond_t go, dn;
    int gen, done_gen, stop;
    long (*fn)(void *); void *arg; long ret;
    int kind;            /* ORK_UNIT_CPU / ORK_UNIT_NPU */
    int busy, op;        /* scheduler bookkeeping: which op index is in flight on this unit */
} ork_spine_unit;

typedef struct ork_spine_op {
    int deps;            /* bitmask of op indices (1<<j) that must finish before this op */
    int placement;       /* ORK_PL_NPU / ORK_PL_CPU / ORK_PL_EITHER */
    long (*fn)(void *); void *arg;
    int state;           /* 0 pend, 1 inflight, 2 done (set by the scheduler) */
    int unit, gen; long ret;
} ork_spine_op;

static void *ork__spine_unit_loop(void *p){ ork_spine_unit *u = p; pthread_mutex_lock(&u->mu);
    for (;;){ while (u->gen == u->done_gen && !u->stop) pthread_cond_wait(&u->go, &u->mu);
        if (u->stop){ pthread_mutex_unlock(&u->mu); return NULL; }
        long (*fn)(void *) = u->fn; void *arg = u->arg; int g = u->gen; pthread_mutex_unlock(&u->mu);
        long r = fn(arg); pthread_mutex_lock(&u->mu); u->ret = r; u->done_gen = g; pthread_cond_signal(&u->dn); } }

/* Start a unit as a worker thread; core>=0 pins it (the NPU unit should be pinned to a big core, and CPU units to
 * DISTINCT spare big cores so they don't fight the NPU unit or the caller's threadpool). */
static inline void ork_spine_unit_start(ork_spine_unit *u, int kind, int core){
    memset(u, 0, sizeof *u); u->kind = kind;
    pthread_mutex_init(&u->mu, 0); pthread_cond_init(&u->go, 0); pthread_cond_init(&u->dn, 0);
    pthread_create(&u->th, 0, ork__spine_unit_loop, u);
    if (core >= 0){ cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core, &cs); pthread_setaffinity_np(u->th, sizeof cs, &cs); }
}
static inline int  ork_spine_unit_dispatch(ork_spine_unit *u, long (*fn)(void *), void *arg){
    pthread_mutex_lock(&u->mu); u->fn = fn; u->arg = arg; int g = ++u->gen; pthread_cond_signal(&u->go); pthread_mutex_unlock(&u->mu); return g; }
static inline int  ork_spine_unit_poll(ork_spine_unit *u, int gen){
    pthread_mutex_lock(&u->mu); int d = u->done_gen >= gen; pthread_mutex_unlock(&u->mu); return d; }
static inline void ork_spine_unit_wait(ork_spine_unit *u, int gen){   /* sleep-poll (never busy-spin: it starves the worker's core) */
    while (!ork_spine_unit_poll(u, gen)){ struct timespec ts = {0, 20000}; nanosleep(&ts, 0); } }
static inline void ork_spine_unit_stop(ork_spine_unit *u){
    pthread_mutex_lock(&u->mu); u->stop = 1; pthread_cond_signal(&u->go); pthread_mutex_unlock(&u->mu); pthread_join(u->th, 0); }

/* Pick a FREE unit matching an op's placement: NPU op -> an NPU unit; CPU op -> a CPU unit; EITHER -> any free
 * unit (prefer a CPU unit, to keep the single NPU unit available for NPU-only ops). Returns index or -1. */
static inline int ork__spine_pick(ork_spine_unit *U, int nu, int placement){
    int npu_free = -1, cpu_free = -1;
    for (int i = 0; i < nu; i++){ if (U[i].busy) continue;
        if (U[i].kind == ORK_UNIT_NPU && npu_free < 0) npu_free = i;
        if (U[i].kind == ORK_UNIT_CPU && cpu_free < 0) cpu_free = i; }
    if (placement == ORK_PL_NPU) return npu_free;
    if (placement == ORK_PL_CPU) return cpu_free;
    return cpu_free >= 0 ? cpu_free : npu_free;   /* EITHER: prefer CPU, fall back to NPU */
}

/* Run the DAG to completion across the units in ONE poll loop. Independent ops on different-kind units overlap;
 * an op is dispatched only once its deps are done; each op's return is stored in ops[i].ret.
 * Returns 0 on success, -1 on a bad config: >1 NPU unit (single-stream), or an op needs an NPU unit and none
 * exists. The units must be started (ork_spine_unit_start) and are NOT stopped here (caller owns their lifetime). */
static inline int ork_spine_run(ork_spine_unit *U, int nu, ork_spine_op *ops, int nops){
    int nnpu = 0; for (int i = 0; i < nu; i++) if (U[i].kind == ORK_UNIT_NPU) nnpu++;
    if (nnpu > 1) return -1;                                     /* single-stream: at most one NPU unit */
    for (int i = 0; i < nops; i++){ ops[i].state = 0; ops[i].unit = -1;
        if (ops[i].placement == ORK_PL_NPU && nnpu == 0) return -1; }
    for (int i = 0; i < nu; i++) U[i].busy = 0;
    int done_mask = 0, ndone = 0;
    while (ndone < nops){
        for (int i = 0; i < nops; i++)
            if (ops[i].state == 0 && (ops[i].deps & done_mask) == ops[i].deps){
                int ui = ork__spine_pick(U, nu, ops[i].placement);
                if (ui >= 0){ ops[i].gen = ork_spine_unit_dispatch(&U[ui], ops[i].fn, ops[i].arg);
                    ops[i].state = 1; ops[i].unit = ui; U[ui].busy = 1; U[ui].op = i; } }
        for (int u = 0; u < nu; u++)
            if (U[u].busy && ork_spine_unit_poll(&U[u], ops[U[u].op].gen)){
                ops[U[u].op].ret = U[u].ret; ops[U[u].op].state = 2; done_mask |= 1 << U[u].op; ndone++; U[u].busy = 0; }
        struct timespec ts = {0, 20000}; nanosleep(&ts, 0);      /* ~20us scheduler tick */
    }
    return 0;
}
#endif /* ORK_SPINE_H */
