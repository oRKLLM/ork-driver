/* orkd — the ork-driver NPU daemon. Single process that OWNS the RK3588 NPU and serializes submits from many
 * client processes onto the single-stream hardware (the ork-driver library connects as a consumer). See
 * src/orkd_proto.h for the wire protocol and memory orkd-daemon-direction for the architecture.
 *
 * THIS FILE (first increment): the LIFECYCLE skeleton only — single-instance flock, Unix-socket accept loop,
 * subscriber ref-counting (HELLO/BYE + socket-EOF for abrupt client death), and idle-reap (exit after
 * ORKD_IDLE_MS with zero subscribers). The NPU submit path (PACK/RUN/FREE) is STUBBED (returns ORKD_ENOSYS);
 * it is wired in the next increment where the daemon calls ork_npu_init() and imports client dma-bufs. This
 * skeleton is pure POSIX (no NPU), so it builds and its lifecycle is testable on any host.
 *
 * Ownership / leak-safety: because orkd will own the IOMMU domains, a client that dies abruptly (SIGKILL) is
 * detected here as socket EOF -> the subscriber is dropped (ref--) and (once the NPU is wired) its resident
 * weights/IOVA are reclaimed. That is the design fix for the leak-on-kill that otherwise forces a reboot. */

#include "orkd_proto.h"
#include "orkd_shm.h"
#include "orkd_ring.h"
#include "ork_npu.h"

#include <stdatomic.h>

#include <errno.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define ORKD_MAX_CLIENTS 64

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int s){ (void)s; g_stop = 1; }
static uint64_t g_orkd_submits = 0;     /* always-on: total compute ops the daemon has served (socket + A-ring) */
static uint64_t g_orkd_ring_wraps = 0;  /* always-on: total A-ring wraps (a client's ring_tail crossing nslots) */

static long now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1000L + t.tv_nsec/1000000L; }

/* read exactly n bytes; 0 = clean EOF, -1 = error/partial (treat as disconnect) */
static int readn(int fd, void *buf, size_t n){
    char *p = buf; size_t got = 0;
    while (got < n){
        ssize_t r = read(fd, p+got, n-got);
        if (r == 0) return 0;                       /* EOF */
        if (r < 0){ if (errno == EINTR) continue; return -1; }
        got += (size_t)r;
    }
    return 1;
}
static int writen(int fd, const void *buf, size_t n){
    const char *p = buf; size_t put = 0;
    while (put < n){
        ssize_t r = write(fd, p+put, n-put);
        if (r < 0){ if (errno == EINTR) continue; return -1; }
        put += (size_t)r;
    }
    return 0;
}
static int send_msg(int fd, uint32_t type, uint64_t tag, const void *payload, uint32_t len){
    struct orkd_hdr h = { type, len, tag };
    if (writen(fd, &h, sizeof h)) return -1;
    if (len && writen(fd, payload, len)) return -1;
    return 0;
}
static void send_error(int fd, uint64_t tag, uint32_t code, const char *msg){
    struct orkd_error e; memset(&e, 0, sizeof e); e.code = code;
    snprintf(e.msg, sizeof e.msg, "%s", msg ? msg : "");
    send_msg(fd, ORKD_ERROR, tag, &e, sizeof e);
}

/* Per-client resident-weight table. A real model client packs one weight per matmul per layer and holds them
 * all resident (e.g. `model 12` = 7 matmuls x 12 layers = 84), and ork_w_free (no ctx) can't send a daemon-free
 * RPC so weights are reclaimed only on the client's socket-EOF — so this must comfortably exceed a model's
 * full resident set, not just a handful. */
#define ORKD_MAX_WEIGHTS 512
#define ORKD_MAX_BYTES (512u << 20)         /* sanity cap on a single weight/A transfer */

struct cweight { uint64_t id; ork_w *w; int K, N, dtype; };   /* dtype = ORKD_DT_I8 | ORKD_DT_F16 (wire dtype) */
struct ckv { uint64_t id; ork_kv_resident *kv; };   /* Tier 12f resident-KV handle (its wkt/wv are also registered in wt[]) */
struct client { int fd; int hello; uint32_t id; struct cweight wt[ORKD_MAX_WEIGHTS]; int nw; uint64_t next_wid; int domain;
                struct ckv kvt[ORKD_MAX_WEIGHTS]; int nkv;   /* resident-KV handles for ORKD_KV_APPEND */
                unsigned owned_dom;   /* bitmask of IOMMU domains this client requested via ORKD_DOM_REQ (bits 1..POOL); released on drop */
                struct orkd_ring *ring; int ring_fd; uint32_t ring_tail; };  /* A-ring: attached low-latency shm transport */

/* A-sched: PER-CLIENT IOMMU DOMAINS (opt-in, ORKD_PER_CLIENT_DOMAINS=1). Intent: pack each client's resident
 * weights into a distinct iommu domain -> isolation + a full ~4 GiB IOVA window each (rk_iommu v2 cap is
 * per-domain). The plumbing is complete (per-client domain stamped on every pack + on struct work; wk_pick
 * groups equal-priority work by domain to amortize the scratch-swap). Default OFF -> all clients share domain 0,
 * byte-identical to the proven single-domain path (validated: make test + test-orkd + multi-consumer all PASS).
 *
 * ★ VALIDATED 2026-07-20: multi-domain WORKS. The earlier "NOT YET VIABLE" note (an mc-scratch theory) was
 * wrong — the real blocker was that the standalone SDP/activation leaf helpers hardcoded iommu_domain_id=-1
 * (npu.c) so a chained SDP submitted in domain 0 against dom-1 buffers -> errno 110; fixed by threading
 * c->dom_active through them (commit 21441d0). test_orkd_2conn_seq with ORKD_PER_CLIENT_DOMAINS=1 now passes
 * bit-exact (4 clients, own domains). TWO ways to get an isolated domain: (a) AUTO — ORKD_PER_CLIENT_DOMAINS=1
 * assigns one per client on accept (dom_alloc); (b) EXPLICIT — the client requests one via ORKD_DOM_REQ
 * (dom_alloc_explicit, works regardless of the env knob) and packs into it (orkd_pack.domain). Both draw from
 * the same g_dom_inuse pool; all a client's domains (auto + requested) are released on drop. */
#define ORKD_DOMAIN_POOL 8                    /* domains 1..8 handed out; 0 = shared/default */
static int g_per_client_dom = 0;              /* ORKD_PER_CLIENT_DOMAINS: auto-assign a domain per client on accept */
static unsigned char g_dom_inuse[ORKD_DOMAIN_POOL + 1];   /* index 1..POOL; 0 unused (always-available shared) */
static int dom_alloc_explicit(void){          /* grab a free pool domain (for an explicit ORKD_DOM_REQ) -> 1..POOL, or 0 if exhausted */
    for (int d = 1; d <= ORKD_DOMAIN_POOL; d++) if (!g_dom_inuse[d]){ g_dom_inuse[d] = 1; return d; }
    return 0;
}
static int dom_alloc(void){                   /* auto per-client domain on accept (gated by the env knob); 0 = shared */
    return g_per_client_dom ? dom_alloc_explicit() : 0;
}
static void dom_release(int d){ if (d >= 1 && d <= ORKD_DOMAIN_POOL) g_dom_inuse[d] = 0; }
static int g_active_dom = 0;                   /* domain of the last-dispatched work (what the NPU is warmed to) */

/* drain n bytes into the void — keeps the stream in sync when a request can't be serviced */
static int drain(int fd, size_t n){ char b[4096]; while (n){ size_t k = n > sizeof b ? sizeof b : n; if (readn(fd, b, k) <= 0) return -1; n -= k; } return 0; }

/* ---- submit queue + scheduler ------------------------------------------------------------------------
 * A long run must NOT hog the single-stream NPU. RUN* requests are ENQUEUED (not run inline); the loop
 * dispatches ONE quantum (row-slice) of the highest-priority item per tick, re-servicing sockets between
 * quanta so a long run yields. CONTENTION-ADAPTIVE: with one queued item, run the whole thing (throughput);
 * with others waiting, slice to ORKD_QUANTUM_ROWS so they interleave (fairness). Priority from orkd_run.flags
 * (higher = sooner). Single-threaded + priority + adaptive row-slice + DOMAIN-GROUPING (equal-priority work is
 * ordered to prefer the active IOMMU domain, amortizing the scratch-swap) + opt-in PER-CLIENT DOMAINS. A
 * dedicated dispatch thread is deliberately NOT used: the row-slice quantum already bounds socket-I/O latency to
 * one quantum, and the NPU is single-stream, so a thread would add concurrency risk with no throughput gain. */
enum { ORKD_QMAX = 256, ORKD_QUANTUM_ROWS = 64 };
/* element size of A for a wire dtype (int8=1B, fp16=2B); C is 4B for both (int32 / fp32) */
static int orkd_esz_a(int wire_dt){ return wire_dt == ORKD_DT_F16 ? 2 : 1; }
struct work {
    int used, fd, type, M, K, N, m0, rc, dtype, domain;
    uint32_t prio; uint64_t tag, seq, weight_id;
    int8_t *A; int32_t *C;          /* A/C: socket-malloc'd unless the matching *_imp is set (zero-copy import) */
    void *A_imp, *C_imp;            /* imported dma-bufs to ork_dma_free (NULL => socket-malloc'd A/C) */
};
static struct work g_q[ORKD_QMAX];
static int g_qn;
static uint64_t g_wseq;

static void wk_free(ork_npu *npu, struct work *w){
    if (!w->used) return;
    if (w->A_imp) ork_dma_free(npu, w->A_imp); else free(w->A);
    if (w->C_imp) ork_dma_free(npu, w->C_imp); else free(w->C);
    memset(w, 0, sizeof *w);
    g_qn--;
}
static struct work *wk_alloc(void){
    for (int i = 0; i < ORKD_QMAX; i++) if (!g_q[i].used){ memset(&g_q[i], 0, sizeof g_q[i]); g_q[i].used = 1; g_q[i].seq = g_wseq++; g_qn++; return &g_q[i]; }
    return NULL;
}
static void wk_purge_fd(ork_npu *npu, int fd){ for (int i = 0; i < ORKD_QMAX; i++) if (g_q[i].used && g_q[i].fd == fd) wk_free(npu, &g_q[i]); }
/* Scheduler pick order: (1) PRIORITY (higher first) — a latency-sensitive client preempts across domains;
 * (2) DOMAIN AFFINITY among equal priority — prefer work already in the active domain so we don't pay a
 * scratch-swap when a same-domain item is waiting (amortizes the multi-domain cost); (3) FIFO (min seq).
 * With per-client domains OFF everything is domain 0, so (2) is a no-op and this is the proven prio+FIFO order. */
static struct work *wk_pick(void){
    struct work *best = NULL;
    for (int i = 0; i < ORKD_QMAX; i++){ struct work *w = &g_q[i]; if (!w->used) continue;
        if (!best){ best = w; continue; }
        if (w->prio != best->prio){ if (w->prio > best->prio) best = w; continue; }
        int wa = (w->domain == g_active_dom), ba = (best->domain == g_active_dom);
        if (wa != ba){ if (wa) best = w; continue; }
        if (w->seq < best->seq) best = w;
    }
    return best;
}

/* ORKD_IMPORT: the client passed a PRE-TILED weight dma-buf fd (SCM_RIGHTS). Import it into the client's
 * domain as a resident weight whose Bb/Bf tiles are VIEWS into it — no tiling, no daemon-owned buffer, the
 * bytes stay in the client's dma-buf (client manages its own IOVA). ork_mm_adopt_imported_i8 ALWAYS consumes
 * `dbuf` (stored in own_buf on success, closed on failure), so this handler never closes it after the call. */
static int handle_import(struct client *cl, ork_npu *npu, int bb_fd, int bf_fd, uint64_t tag){
    struct orkd_import im;
    if (readn(cl->fd, &im, sizeof im) <= 0){ if (bb_fd >= 0) close(bb_fd); if (bf_fd >= 0) close(bf_fd); return -1; }
    if (bb_fd < 0){ if (bf_fd >= 0) close(bf_fd); send_error(cl->fd, tag, ORKD_EPROTO, "import: missing fd"); return 0; }
    if (im.dtype != ORKD_DT_I8){ close(bb_fd); if (bf_fd >= 0) close(bf_fd); send_error(cl->fd, tag, ORKD_ENOSYS, "import int8 only"); return 0; }
    if (!im.bb_bytes || im.bb_bytes > ORKD_MAX_BYTES || im.bf_bytes > ORKD_MAX_BYTES){ close(bb_fd); if (bf_fd >= 0) close(bf_fd); send_error(cl->fd, tag, ORKD_EPROTO, "import bad size"); return 0; }
    int pdom = (int)im.domain;
    if (pdom > 0){
        if (pdom > ORKD_DOMAIN_POOL || !(cl->owned_dom & (1u << pdom))){ close(bb_fd); if (bf_fd >= 0) close(bf_fd); send_error(cl->fd, tag, ORKD_EBADH, "domain not owned by client"); return 0; }
    } else pdom = cl->domain;
    ork_npu_set_pack_domain(npu, pdom);
    ork_w *w = ork_mm_adopt_imported_i8(npu, (int)im.K, (int)im.N, bb_fd, bf_fd, (size_t)im.bb_bytes, (size_t)im.bf_bytes);
    ork_npu_set_pack_domain(npu, 0);            /* restore default */
    struct orkd_handle hh; memset(&hh, 0, sizeof hh);
    if (w && cl->nw < ORKD_MAX_WEIGHTS){ hh.id = ++cl->next_wid; hh.rc = 0; cl->wt[cl->nw++] = (struct cweight){ hh.id, w, (int)im.K, (int)im.N, (int)ORKD_DT_I8 }; }
    else { if (w) ork_mm_free(npu, w); hh.rc = -1; }   /* adopt ALWAYS consumes both fds; on w!=NULL, free reclaims the imports */
    send_msg(cl->fd, ORKD_IMPORT_OK, tag, &hh, sizeof hh);
    return 0;
}
/* ORKD_FFN: the whole SwiGLU FFN inner as ONE coalesced on-NPU chain (ork_mm_run_chain_i8_ffn, SDP-op
 * address-aliasing) against 3 resident weights. Fixed op-list: gate MM8(x,Wg) -> silu -> up MM8(x,Wu) ->
 * glu ewmul -> down MM32(glu,Wd). One socket round-trip + one submit for the entire inner; intermediates
 * never leave the NPU. Reply = the down output (M*Kd int32). ork_mm_run_chain_i8_ffn stages A/C internally
 * and dom_activates the weights' domain, so no domain setup here. Runs INLINE (single-stream serialized). */
static int handle_ffn(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_ffn f;
    if (readn(cl->fd, &f, sizeof f) <= 0) return -1;
    if (f.abytes > ORKD_MAX_BYTES){ drain(cl->fd, f.abytes); send_error(cl->fd, tag, ORKD_EPROTO, "ffn A too big"); return 0; }
    int8_t *A = malloc(f.abytes ? f.abytes : 1);
    if (!A){ drain(cl->fd, f.abytes); send_error(cl->fd, tag, ORKD_EOOM, "ffn A"); return 0; }
    if (f.abytes && readn(cl->fd, A, f.abytes) <= 0){ free(A); return -1; }
    struct cweight *cg = NULL, *cu = NULL, *cd = NULL;
    for (int j = 0; j < cl->nw; j++){ uint64_t id = cl->wt[j].id;
        if (id == f.gate_id) cg = &cl->wt[j]; if (id == f.up_id) cu = &cl->wt[j]; if (id == f.down_id) cd = &cl->wt[j]; }
    int M = (int)f.M, K = (int)f.K, Nff = (int)f.Nff, Kd = (int)f.Kd;
    if (!cg || !cu || !cd || M < 1 || K < 1 || Nff < 1 || Kd < 1 || f.abytes != (uint32_t)((size_t)M * K)){
        free(A); send_error(cl->fd, tag, (cg && cu && cd) ? ORKD_EPROTO : ORKD_EBADH, "ffn weight/dim"); return 0; }
    /* int8 intermediates ride in the low bytes of int32 slots (matches ork_mm_run_chain_i8_ffn's C usage); down is int32 */
    size_t isz = (size_t)M * Nff * 4, dsz = (size_t)M * Kd * 4;
    int32_t *Cg = malloc(isz), *Cs = malloc(isz), *Cu = malloc(isz), *Ch = malloc(isz), *Cd = malloc(dsz);
    if (!Cg || !Cs || !Cu || !Ch || !Cd){ free(A); free(Cg); free(Cs); free(Cu); free(Ch); free(Cd); send_error(cl->fd, tag, ORKD_EOOM, "ffn scratch"); return 0; }
    ork_mm_task_i8 t[5] = {
        { cg->w, M, A, Cg }, { cg->w, M, A, Cs }, { cu->w, M, A, Cu }, { cg->w, M, A, Ch }, { cd->w, M, A, Cd } };
    ork_chain_op ops[5] = {
        { 1, -1, 0, f.gate_mult, f.gate_shift },   /* gate MM8 (reads A)               */
        { 2,  0, 0, 0, 0 },                        /* silu(gate = t0)                  */
        { 1, -1, 0, f.up_mult, f.up_shift },       /* up MM8 (reads A)                 */
        { 3,  1, 2, f.glu_mult, f.glu_shift },     /* glu = silu(t1) * up(t2)          */
        { 0,  3, 0, 0, 0 } };                      /* down MM32 (reads glu = t3)       */
    int rc = ork_mm_run_chain_i8_ffn(npu, 5, t, ops, f.in_scale, f.out_scale);
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.rc = rc;
    int payload = (rc == 0);
    struct orkd_hdr rh = { ORKD_FFN_OK, (uint32_t)(sizeof hh + (payload ? dsz : 0)), tag };
    int werr = writen(cl->fd, &rh, sizeof rh) || writen(cl->fd, &hh, sizeof hh);
    if (!werr && payload) werr = writen(cl->fd, Cd, dsz);
    free(A); free(Cg); free(Cs); free(Cu); free(Ch); free(Cd);
    return werr ? -1 : 0;
}
/* ORKD_ATTN: the fused attention core [QK^T->exp->reduce,e.V] as ONE coalesced on-NPU chain (chainav pattern,
 * ork_mm_run_chain_i8_ffn_exp). Fixed op-list built daemon-side against 3 resident weights: K^T[Kp,Nk], ones[Nk,32],
 * V[Nk,dv]. Q (Nq*Kp int8) follows. Reply = Sigma(Nq*32 int32) then av(Nq*dv int32). e never leaves the NPU. */
static int handle_attn(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_attn a;
    if (readn(cl->fd, &a, sizeof a) <= 0) return -1;
    if (a.abytes > ORKD_MAX_BYTES){ drain(cl->fd, a.abytes); send_error(cl->fd, tag, ORKD_EPROTO, "attn Q too big"); return 0; }
    int8_t *Q = malloc(a.abytes ? a.abytes : 1);
    if (!Q){ drain(cl->fd, a.abytes); send_error(cl->fd, tag, ORKD_EOOM, "attn Q"); return 0; }
    if (a.abytes && readn(cl->fd, Q, a.abytes) <= 0){ free(Q); return -1; }
    struct cweight *ckt = NULL, *co = NULL, *cv = NULL;
    for (int j = 0; j < cl->nw; j++){ uint64_t id = cl->wt[j].id;
        if (id == a.wkt_id) ckt = &cl->wt[j]; if (id == a.wones_id) co = &cl->wt[j]; if (id == a.wv_id) cv = &cl->wt[j]; }
    int Nq = (int)a.Nq, Nk = (int)a.Nk, Kp = (int)a.Kp, dv = (int)a.dv;
    if (!ckt || !co || !cv || Nq < 1 || Nk < 1 || Kp < 1 || dv < 1 || a.abytes != (uint32_t)((size_t)Nq * Kp)){
        free(Q); send_error(cl->fd, tag, (ckt && co && cv) ? ORKD_EPROTO : ORKD_EBADH, "attn weight/dim"); return 0; }
    /* scb/eb hold int8 (scores, exp) in the low bytes of int32 slots (chainav C usage); ss = reduce Sigma[Nq,32]; av[Nq,dv] */
    size_t nb = (size_t)Nq * Nk * 4, sb = (size_t)Nq * 32 * 4, ab = (size_t)Nq * dv * 4;
    int32_t *scb = malloc(nb), *eb = malloc(nb), *ss = malloc(sb), *avb = malloc(ab);
    if (!scb || !eb || !ss || !avb){ free(Q); free(scb); free(eb); free(ss); free(avb); send_error(cl->fd, tag, ORKD_EOOM, "attn scratch"); return 0; }
    ork_mm_task_i8 t[4] = {
        { ckt->w, Nq, Q,            scb },   /* QK^T -> scores (reads Q)     */
        { ckt->w, Nq, (int8_t*)scb, eb  },   /* exp(t0) -> e (N-sized by wkt) */
        { co->w,  Nq, (int8_t*)eb,  ss  },   /* reduce e(t1) -> Sigma         */
        { cv->w,  Nq, (int8_t*)eb,  avb } }; /* e(t1).V -> av                 */
    ork_chain_op ops[4] = {
        { 1, -1, 0, a.r_mult, a.r_shift },   /* QK^T MM8, int32->int8 score requant */
        { 2,  0, 0, 0, 0 },                  /* exp(t0)                             */
        { 0,  1, 0, 0, 0 },                  /* reduce e(t1) -> Sigma               */
        { 0,  1, 0, 0, 0 } };                /* e(t1).V -> av                       */
    int rc = ork_mm_run_chain_i8_ffn_exp_biased(npu, 4, t, ops, a.in_scale, a.out_scale, a.max_bias);
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.rc = rc;
    int payload = (rc == 0);
    struct orkd_hdr rh = { ORKD_ATTN_OK, (uint32_t)(sizeof hh + (payload ? sb + ab : 0)), tag };
    int werr = writen(cl->fd, &rh, sizeof rh) || writen(cl->fd, &hh, sizeof hh);
    if (!werr && payload) werr = (writen(cl->fd, ss, sb) || writen(cl->fd, avb, ab));
    free(Q); free(scb); free(eb); free(ss); free(avb);
    return werr ? -1 : 0;
}
/* ORKD_ATTN_RR: N fused attention chains fanned round-robin across the NPU cores in ONE dispatch (the daemon runs
 * a DIRECT ctx, so ork_mm_run_chains_rr_biased takes the local multi-core path). Payload = nchains {wkt,wones,wv}
 * id triples then Q (nchains*Nq*Kp int8, chain-major). Reply = per chain Sigma(Nq*32) then av(Nq*dv) int32. All
 * chains share the requant + scalar-max-biased exp LUT. This is the decode attention path: one round-trip, Hkv
 * kv-head chains on separate cores concurrently, e never leaves the NPU. */
static int handle_attn_rr(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_attn_rr a;
    if (readn(cl->fd, &a, sizeof a) <= 0) return -1;
    int nch = (int)a.nchains, Nq = (int)a.Nq, Nk = (int)a.Nk, Kp = (int)a.Kp, dv = (int)a.dv;
    size_t tbytes = (nch>=1 && nch<=ORKD_ATTN_RR_MAX) ? (size_t)nch*3*8 : 0;
    /* bad nchains: still consume the triples+Q we can compute, then error. abytes was sent by the client. */
    if (nch < 1 || nch > ORKD_ATTN_RR_MAX || Nq < 1 || Nk < 1 || Kp < 1 || dv < 1 || a.abytes > ORKD_MAX_BYTES
        || a.abytes != (uint32_t)((size_t)nch * Nq * Kp)){
        drain(cl->fd, tbytes + a.abytes); send_error(cl->fd, tag, ORKD_EPROTO, "attn_rr dims"); return 0; }
    uint64_t *trip = malloc(tbytes); int8_t *Q = malloc(a.abytes ? a.abytes : 1);
    if (!trip || !Q){ free(trip); free(Q); drain(cl->fd, tbytes + a.abytes); send_error(cl->fd, tag, ORKD_EOOM, "attn_rr in"); return 0; }
    if (readn(cl->fd, trip, tbytes) <= 0 || (a.abytes && readn(cl->fd, Q, a.abytes) <= 0)){ free(trip); free(Q); return -1; }
    /* per-chain scratch + task lists; ss/avb hold the outputs to reply */
    size_t sb = (size_t)Nq*32*4, ab = (size_t)Nq*dv*4, nb = (size_t)Nq*Nk*4;
    int32_t **scb = calloc(nch,sizeof(void*)), **eb = calloc(nch,sizeof(void*)), **ss = calloc(nch,sizeof(void*)), **avb = calloc(nch,sizeof(void*));
    ork_mm_task_i8 (*tk)[4] = calloc(nch,sizeof(*tk));
    const ork_mm_task_i8 **chains = calloc(nch,sizeof(void*)); int *S = calloc(nch,sizeof(int));
    int oom = (!scb||!eb||!ss||!avb||!tk||!chains||!S), bad = 0;
    for (int n = 0; n < nch && !oom; n++){
        scb[n]=malloc(nb); eb[n]=malloc(nb); ss[n]=malloc(sb); avb[n]=malloc(ab);
        if (!scb[n]||!eb[n]||!ss[n]||!avb[n]){ oom=1; break; }
        uint64_t wkt_id=trip[n*3], wones_id=trip[n*3+1], wv_id=trip[n*3+2];
        struct cweight *ckt=NULL,*co=NULL,*cv=NULL;
        for (int j=0;j<cl->nw;j++){ uint64_t id=cl->wt[j].id;
            if (id==wkt_id) ckt=&cl->wt[j]; if (id==wones_id) co=&cl->wt[j]; if (id==wv_id) cv=&cl->wt[j]; }
        if (!ckt||!co||!cv){ bad=1; break; }
        int8_t *Qn = Q + (size_t)n*Nq*Kp;
        tk[n][0]=(ork_mm_task_i8){ ckt->w, Nq, Qn,             scb[n] };
        tk[n][1]=(ork_mm_task_i8){ ckt->w, Nq, (int8_t*)scb[n], eb[n] };
        tk[n][2]=(ork_mm_task_i8){ co->w,  Nq, (int8_t*)eb[n],  ss[n] };
        tk[n][3]=(ork_mm_task_i8){ cv->w,  Nq, (int8_t*)eb[n],  avb[n] };
        chains[n]=tk[n]; S[n]=4;
    }
    ork_chain_op ops[4] = { {1,-1,0,a.r_mult,a.r_shift}, {2,0,0,0,0}, {0,1,0,0,0}, {0,1,0,0,0} };
    int rc = (oom||bad) ? -1 : ork_mm_run_chains_rr_biased(npu, nch, chains, S, ops, a.in_scale, a.out_scale, a.max_bias);
    int payload = (rc == 0);
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.rc = rc;
    struct orkd_hdr rh = { ORKD_ATTN_RR_OK, (uint32_t)(sizeof hh + (payload ? (size_t)nch*(sb+ab) : 0)), tag };
    if (oom || bad) hh.rc = oom ? ORKD_EOOM : ORKD_EBADH;
    int werr = writen(cl->fd, &rh, sizeof rh) || writen(cl->fd, &hh, sizeof hh);
    for (int n = 0; n < nch && !werr && payload; n++) werr = (writen(cl->fd, ss[n], sb) || writen(cl->fd, avb[n], ab));
    for (int n = 0; n < nch; n++){ if(scb)free(scb[n]); if(eb)free(eb[n]); if(ss)free(ss[n]); if(avb)free(avb[n]); }
    free(scb); free(eb); free(ss); free(avb); free(tk); free(chains); free(S); free(trip); free(Q);
    return werr ? -1 : 0;
}
/* Tier 12f resident-KV: the daemon runs a DIRECT npu ctx (fd valid), so ork_kv_resident_alloc/append take the
 * LOCAL path here. Register the two resident weights in wt[] (so ORKD_CHAIN can reference them by id) and keep
 * the kv handle in kvt[] for append. Freed with the rest on client drop. */
static int handle_kv_alloc(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_kv_alloc rq;
    if (readn(cl->fd, &rq, sizeof rq) <= 0) return -1;
    struct orkd_kv_alloc_ok ok; memset(&ok, 0, sizeof ok); ok.rc = -1;
    ork_kv_resident *kv = ork_kv_resident_alloc(npu, (int)rq.HD, (int)rq.Lmax);
    if (kv && cl->nw + 2 <= ORKD_MAX_WEIGHTS && cl->nkv < ORKD_MAX_WEIGHTS){
        uint64_t wkt_id = ++cl->next_wid, wv_id = ++cl->next_wid, kv_id = ++cl->next_wid;
        cl->wt[cl->nw++] = (struct cweight){ wkt_id, kv->wkt, kv->Kp, (int)rq.Lmax, (int)ORKD_DT_I8 };
        cl->wt[cl->nw++] = (struct cweight){ wv_id,  kv->wv,  (int)rq.Lmax, (int)rq.HD, (int)ORKD_DT_I8 };
        cl->kvt[cl->nkv++] = (struct ckv){ kv_id, kv };
        ok.rc = 0; ok.kv_id = kv_id; ok.wkt_id = wkt_id; ok.wv_id = wv_id;
    } else if (kv) ork_kv_resident_free(npu, kv);
    send_msg(cl->fd, ORKD_KV_ALLOC_OK, tag, &ok, sizeof ok);
    return 0;
}
static int handle_kv_append(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_kv_append rq;
    if (readn(cl->fd, &rq, sizeof rq) <= 0) return -1;
    uint32_t HD = rq.HD; size_t nb = 2 * (size_t)HD;
    if (HD == 0 || HD > 4096){ if (nb) drain(cl->fd, nb); send_error(cl->fd, tag, ORKD_EPROTO, "bad HD"); return 0; }
    int8_t *buf = malloc(nb);
    if (!buf){ drain(cl->fd, nb); send_error(cl->fd, tag, ORKD_EOOM, "kv append alloc"); return 0; }
    if (readn(cl->fd, buf, nb) <= 0){ free(buf); return -1; }
    ork_kv_resident *kv = NULL;
    for (int i = 0; i < cl->nkv; i++) if (cl->kvt[i].id == rq.kv_id){ kv = cl->kvt[i].kv; break; }
    struct orkd_handle hh; memset(&hh, 0, sizeof hh);
    hh.rc = kv ? ork_kv_append(npu, kv, (int)rq.key, buf, buf + HD) : -2;
    free(buf);
    send_msg(cl->fd, ORKD_KV_APPEND_OK, tag, &hh, sizeof hh);
    return 0;
}
/* #2b-1 submit RPC (int8, socket-transfer; dma-buf zero-copy is #2b-2). Handlers read their own payload from
 * the fd and reply; return <0 to drop the client. The NPU op (ork_mm_run_i8) rides the interruptible doorbell,
 * so a RUN in flight never puts orkd in D-state -> orkd stays SIGTERM-clean and never blocks system shutdown. */
static int handle_pack(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_pack pk;
    if (readn(cl->fd, &pk, sizeof pk) <= 0) return -1;
    if (pk.bytes == 0 || pk.bytes > ORKD_MAX_BYTES){ drain(cl->fd, pk.bytes); send_error(cl->fd, tag, ORKD_EPROTO, "bad pack size"); return 0; }
    if (pk.dtype != ORKD_DT_I8 && pk.dtype != ORKD_DT_F16 && pk.dtype != ORKD_DT_I4){ drain(cl->fd, pk.bytes); send_error(cl->fd, tag, ORKD_ENOSYS, "int8/fp16/int4 only"); return 0; }
    int8_t *wbuf = malloc(pk.bytes);
    if (!wbuf){ drain(cl->fd, pk.bytes); send_error(cl->fd, tag, ORKD_EOOM, "pack alloc"); return 0; }
    if (readn(cl->fd, wbuf, pk.bytes) <= 0){ free(wbuf); return -1; }
    /* CLIENT-CHOSEN DOMAIN: pk.domain>0 means the client explicitly requested this domain (ORKD_DOM_REQ) and is
     * packing into it — validate it's one they own. pk.domain==0 falls back to the auto per-client domain
     * (cl->domain, 0 unless ORKD_PER_CLIENT_DOMAINS). The weight lands here; its RUNs inherit it via w->domain. */
    int pdom = (int)pk.domain;
    if (pdom > 0){
        if (pdom > ORKD_DOMAIN_POOL || !(cl->owned_dom & (1u << pdom))){ free(wbuf); send_error(cl->fd, tag, ORKD_EBADH, "domain not owned by client"); return 0; }
    } else pdom = cl->domain;
    ork_npu_set_pack_domain(npu, pdom);
    ork_w *w = (pk.dtype == ORKD_DT_F16) ? ork_mm_pack(npu, (int)pk.K, (int)pk.N, (const ork_f16 *)wbuf)
             : (pk.dtype == ORKD_DT_I4)  ? ork_mm_pack_i4(npu, (int)pk.K, (int)pk.N, wbuf)
                                         : ork_mm_pack_i8(npu, (int)pk.K, (int)pk.N, wbuf);
    ork_npu_set_pack_domain(npu, 0);            /* restore default for any non-client-scoped pack */
    free(wbuf);
    struct orkd_handle hh; memset(&hh, 0, sizeof hh);
    if (w && cl->nw < ORKD_MAX_WEIGHTS){ hh.id = ++cl->next_wid; hh.rc = 0; cl->wt[cl->nw++] = (struct cweight){ hh.id, w, (int)pk.K, (int)pk.N, (int)pk.dtype }; }
    else { if (w) ork_mm_free(npu, w); hh.rc = -1; }
    send_msg(cl->fd, ORKD_PACK_OK, tag, &hh, sizeof hh);
    return 0;
}
static int handle_run(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_run rq;
    if (readn(cl->fd, &rq, sizeof rq) <= 0) return -1;
    struct cweight *cw = NULL;
    for (int i = 0; i < cl->nw; i++) if (cl->wt[i].id == rq.weight_id){ cw = &cl->wt[i]; break; }
    if (rq.abytes > ORKD_MAX_BYTES){ drain(cl->fd, rq.abytes); send_error(cl->fd, tag, ORKD_EPROTO, "bad A size"); return 0; }
    int8_t *A = malloc(rq.abytes ? rq.abytes : 1);
    if (!A){ drain(cl->fd, rq.abytes); send_error(cl->fd, tag, ORKD_EOOM, "A alloc"); return 0; }
    if (rq.abytes && readn(cl->fd, A, rq.abytes) <= 0){ free(A); return -1; }
    if (!cw || rq.abytes != (uint32_t)((size_t)rq.M * cw->K * orkd_esz_a(cw->dtype))){ free(A); send_error(cl->fd, tag, cw ? ORKD_EPROTO : ORKD_EBADH, cw ? "A size mismatch" : "unknown weight"); return 0; }
    int32_t *C = malloc((size_t)rq.M * cw->N * 4);
    struct work *w = C ? wk_alloc() : NULL;
    if (!C || !w){ free(A); free(C); send_error(cl->fd, tag, ORKD_EOOM, "queue full"); return 0; }
    w->fd = cl->fd; w->type = ORKD_RUN; w->tag = tag; w->prio = rq.flags; w->weight_id = rq.weight_id;
    /* v2: the client stamps the op's domain (== where it packed this weight); honor it, else fall back to the
     * weight's resident domain. The scheduler dom_activates w->domain (zero-copy swap) before dispatch. */
    w->domain = rq.domain ? (int)rq.domain : ork_w_domain(cw->w);
    w->M = (int)rq.M; w->K = cw->K; w->N = cw->N; w->dtype = cw->dtype; w->A = A; w->C = C;   /* enqueued; the scheduler dispatches it */
    return 0;
}
static int handle_free(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_handle req;
    if (readn(cl->fd, &req, sizeof req) <= 0) return -1;
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.id = req.id; hh.rc = -1;
    for (int i = 0; i < cl->nw; i++) if (cl->wt[i].id == req.id){ ork_mm_free(npu, cl->wt[i].w); cl->wt[i] = cl->wt[--cl->nw]; hh.rc = 0; break; }
    send_msg(cl->fd, ORKD_PACK_OK, tag, &hh, sizeof hh);   /* PACK_OK = generic handle-op ack */
    return 0;
}
/* Stateless SDP activation op (silu/gelu/ewmul/add). Run INLINE (not queued): one-shot + tiny (M=8,N=64 geometry)
 * and the daemon is single-threaded, so it's still serialized on the single-stream NPU with any queued matmul
 * quanta. nin input payloads follow the struct (concatenated); reply = orkd_handle + the M*N*out_esz output. */
static int handle_sdp(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_sdp sp;
    if (readn(cl->fd, &sp, sizeof sp) <= 0) return -1;
    size_t half = (size_t)sp.M * sp.N * sp.in_esz, inb = sp.inbytes, outb = (size_t)sp.M * sp.N * sp.out_esz;
    if (!sp.M || !sp.N || inb > ORKD_MAX_BYTES || inb != half * (sp.nin ? sp.nin : 1)){ drain(cl->fd, sp.inbytes); send_error(cl->fd, tag, ORKD_EPROTO, "bad sdp payload"); return 0; }
    uint8_t *in = malloc(inb ? inb : 1), *out = malloc(outb ? outb : 1);
    if (!in || !out){ free(in); free(out); drain(cl->fd, sp.inbytes); send_error(cl->fd, tag, ORKD_EOOM, "sdp alloc"); return 0; }
    if (inb && readn(cl->fd, in, inb) <= 0){ free(in); free(out); return -1; }
    const uint8_t *a = in, *b = in + half;   /* binary: second operand is the second half */
    int rc;
    switch (sp.op){
        case ORKD_SDP_SILU_I8:   rc = ork_npu_silu_i8 (npu, (const signed char *)a, sp.M, sp.N, sp.in_scale, sp.out_scale, (signed char *)out, NULL); break;
        case ORKD_SDP_GELU_I8:   rc = ork_npu_gelu_i8 (npu, (const signed char *)a, sp.M, sp.N, sp.in_scale, sp.out_scale, (signed char *)out, NULL); break;
        case ORKD_SDP_EWMUL_I8:  rc = ork_npu_ewmul_i8(npu, (const int8_t *)a, (const int8_t *)b, sp.M, sp.N, sp.mult, sp.shift, (int8_t *)out, NULL); break;
        case ORKD_SDP_EWMUL_F16: rc = ork_npu_ewmul_f16(npu, (const ork_f16 *)a, (const ork_f16 *)b, sp.M, sp.N, (ork_f16 *)out, NULL); break;
        case ORKD_SDP_ADD_I8:    rc = ork_npu_add_i8  (npu, (const signed char *)a, (const signed char *)b, sp.M, sp.N, sp.a_scale, sp.b_scale, sp.out_scale, (signed char *)out, NULL); break;
        case ORKD_SDP_ADD_F16:   rc = ork_npu_add_f16 (npu, (const ork_f16 *)a, (const ork_f16 *)b, sp.M, sp.N, (ork_f16 *)out, NULL); break;
        case ORKD_SDP_RSQRT_I8:  rc = ork_npu_rsqrt_i8(npu, (const signed char *)a, sp.M, sp.N, sp.in_scale, sp.out_scale, (signed char *)out, NULL); break;
        case ORKD_SDP_EXP_I8:    rc = ork_npu_exp_i8  (npu, (const signed char *)a, sp.M, sp.N, sp.in_scale, sp.out_scale, (signed char *)out, NULL); break;
        case ORKD_SDP_SILU_I16:  rc = ork_npu_silu_i16(npu, (const int16_t *)a, sp.M, sp.N, sp.in_scale, sp.out_scale, (int16_t *)out, NULL); break;
        case ORKD_SDP_GELU_I16:  rc = ork_npu_gelu_i16(npu, (const int16_t *)a, sp.M, sp.N, sp.in_scale, sp.out_scale, (int16_t *)out, NULL); break;
        case ORKD_SDP_RSQRT_I16: rc = ork_npu_rsqrt_i16(npu, (const int16_t *)a, sp.M, sp.N, sp.in_scale, sp.out_scale, (int16_t *)out, NULL); break;
        case ORKD_SDP_EXP_I16:   rc = ork_npu_exp_i16 (npu, (const int16_t *)a, sp.M, sp.N, sp.in_scale, sp.out_scale, (int16_t *)out, NULL); break;
        case ORKD_SDP_EWMUL_I16: rc = ork_npu_ewmul_i16(npu, (const int16_t *)a, (const int16_t *)b, sp.M, sp.N, sp.mult, sp.shift, (int16_t *)out, NULL); break;
        case ORKD_SDP_ADD_I16:   rc = ork_npu_add_i16 (npu, (const int16_t *)a, (const int16_t *)b, sp.M, sp.N, sp.a_scale, sp.b_scale, sp.out_scale, (int16_t *)out, NULL); break;
        default: rc = -100;
    }
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.rc = rc;
    int payload = (rc == 0);
    struct orkd_hdr rh = { ORKD_SDP_OK, (uint32_t)(sizeof hh + (payload ? outb : 0)), tag };
    (void)!(writen(cl->fd, &rh, sizeof rh) || writen(cl->fd, &hh, sizeof hh) || (payload && writen(cl->fd, out, outb)));
    free(in); free(out);
    return 0;
}
/* Fused int8 matmul chain: S resident weights run as one PC-chained submit (ork_mm_run_chain_i8). Each task's
 * weight is resolved by id in THIS client's table; A payloads arrive concatenated (task order), C payloads are
 * returned concatenated. Run INLINE (a chain is one bounded submit; the single-threaded daemon serializes it). */
static int handle_chain(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_chain_hdr ch;
    if (readn(cl->fd, &ch, sizeof ch) <= 0) return -1;
    int S = (int)ch.S;
    size_t tsz = (size_t)(S > 0 ? S : 0) * sizeof(struct orkd_chain_task);
    if (S < 1 || S > ORKD_CHAIN_MAX || ch.abytes_total > ORKD_MAX_BYTES){ drain(cl->fd, tsz + ch.abytes_total); send_error(cl->fd, tag, ORKD_EPROTO, "bad chain"); return 0; }
    struct orkd_chain_task *ts = malloc(tsz);
    if (!ts){ drain(cl->fd, tsz + ch.abytes_total); send_error(cl->fd, tag, ORKD_EOOM, "chain tasks"); return 0; }
    if (readn(cl->fd, ts, tsz) <= 0){ free(ts); return -1; }
    uint8_t *ablob = malloc(ch.abytes_total ? ch.abytes_total : 1);
    if (!ablob){ free(ts); drain(cl->fd, ch.abytes_total); send_error(cl->fd, tag, ORKD_EOOM, "chain A"); return 0; }
    if (ch.abytes_total && readn(cl->fd, ablob, ch.abytes_total) <= 0){ free(ts); free(ablob); return -1; }
    /* build the daemon-side task array: resolve each weight, point A into ablob, allocate a C per task */
    ork_mm_task_i8 *mt = calloc((size_t)S, sizeof *mt);
    int32_t **Cs = calloc((size_t)S, sizeof *Cs);
    size_t *cb = calloc((size_t)S, sizeof *cb);   /* per-task C bytes = M*N*4 */
    size_t aoff = 0, ctot = 0; int ok = (mt && Cs && cb), rc = 0;
    for (int i = 0; ok && i < S; i++){
        struct cweight *cw = NULL;
        for (int j = 0; j < cl->nw; j++) if (cl->wt[j].id == ts[i].weight_id){ cw = &cl->wt[j]; break; }
        if (!cw || cw->dtype != ORKD_DT_I8 || ts[i].abytes != (uint32_t)((size_t)ts[i].M * cw->K) || aoff + ts[i].abytes > ch.abytes_total){ ok = 0; break; }
        cb[i] = (size_t)ts[i].M * cw->N * 4;
        Cs[i] = malloc(cb[i]);
        if (!Cs[i]){ ok = 0; break; }
        mt[i].w = cw->w; mt[i].M = (int)ts[i].M; mt[i].A = (const int8_t *)(ablob + aoff); mt[i].C = Cs[i];
        aoff += ts[i].abytes; ctot += cb[i];
    }
    if (ok) rc = ork_mm_run_chain_i8(npu, S, mt);
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.rc = ok ? rc : -1;
    int payload = (ok && rc == 0);
    struct orkd_hdr rh = { ORKD_CHAIN_OK, (uint32_t)(sizeof hh + (payload ? ctot : 0)), tag };
    int werr = writen(cl->fd, &rh, sizeof rh) || writen(cl->fd, &hh, sizeof hh);
    if (!werr && payload) for (int i = 0; i < S && !werr; i++) werr = writen(cl->fd, Cs[i], cb[i]);
    if (Cs) for (int i = 0; i < S; i++) free(Cs[i]);
    free(mt); free(Cs); free(cb); free(ts); free(ablob);
    return werr ? -1 : 0;
}
/* Heterogeneous op-sequence submit: reconstruct an ork_seq_op[] (resident weights by id + received A/B buffers
 * + allocated C) and run ork_submit_seq, which batches maximal runs of doorbell-eligible ops onto the spine,
 * breaks the chain to the SW path at each op that can't be chained, then resumes. Runs INLINE (one bounded
 * sequence; the single-threaded daemon serializes it on the single-stream NPU). C's returned concatenated. */
static int handle_seq(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_seq_hdr sh;
    if (readn(cl->fd, &sh, sizeof sh) <= 0) return -1;
    int n = (int)sh.n;
    size_t osz = (size_t)(n > 0 ? n : 0) * sizeof(struct orkd_seq_op);
    if (n < 1 || n > ORKD_SEQ_MAX || sh.in_total > ORKD_MAX_BYTES){ drain(cl->fd, osz + sh.in_total); send_error(cl->fd, tag, ORKD_EPROTO, "bad seq"); return 0; }
    struct orkd_seq_op *ops = malloc(osz);
    if (!ops){ drain(cl->fd, osz + sh.in_total); send_error(cl->fd, tag, ORKD_EOOM, "seq ops"); return 0; }
    if (readn(cl->fd, ops, osz) <= 0){ free(ops); return -1; }
    uint8_t *inblob = malloc(sh.in_total ? sh.in_total : 1);
    if (!inblob){ free(ops); drain(cl->fd, sh.in_total); send_error(cl->fd, tag, ORKD_EOOM, "seq in"); return 0; }
    if (sh.in_total && readn(cl->fd, inblob, sh.in_total) <= 0){ free(ops); free(inblob); return -1; }
    ork_seq_op *seq = calloc((size_t)n, sizeof *seq);
    void **Cs = calloc((size_t)n, sizeof *Cs);
    size_t inoff = 0, ctot = 0; int ok = (seq && Cs);
    for (int i = 0; ok && i < n; i++){
        struct orkd_seq_op *o = &ops[i];
        size_t need = (o->a_src ? 0 : o->abytes) + (o->b_src ? 0 : o->bbytes);   /* A2: referenced inputs aren't in the uploaded blob */
        if ((size_t)inoff + need > sh.in_total){ ok = 0; break; }
        seq[i].kind = (ork_seq_kind)o->kind; seq[i].M = (int)o->M; seq[i].N = (int)o->N;
        seq[i].in_scale = o->in_scale; seq[i].out_scale = o->out_scale; seq[i].b_scale = o->b_scale; seq[i].mult = o->mult; seq[i].shift = o->shift; seq[i].group = (int)o->group;
        if (o->weight_id){   /* matmul op: resolve the resident weight in this client's table */
            struct cweight *cw = NULL;
            for (int j = 0; j < cl->nw; j++) if (cl->wt[j].id == o->weight_id){ cw = &cl->wt[j]; break; }
            if (!cw){ ok = 0; break; }
            seq[i].w = cw->w;
        }
        /* A2: a_src/b_src = j+1 -> input is op j's resident output buffer (Cs[j]); else uploaded in the blob. */
        if (o->a_src){ if (o->a_src > i){ ok = 0; break; } seq[i].A = Cs[o->a_src - 1]; }
        else { seq[i].A = inblob + inoff; inoff += o->abytes; }
        if (o->b_src){ if (o->b_src > i){ ok = 0; break; } seq[i].B = Cs[o->b_src - 1]; }
        else { seq[i].B = o->bbytes ? inblob + inoff : NULL; inoff += o->bbytes; }
        Cs[i] = malloc(o->cbytes ? o->cbytes : 1);
        if (!Cs[i]){ ok = 0; break; }
        seq[i].C = Cs[i]; if (!o->c_keep) ctot += o->cbytes;   /* c_keep => stays resident, not shipped back */
    }
    int rc = ok ? ork_submit_seq(npu, seq, n) : -2;
    { const char *dbg = getenv("ORKD_SEQ_DEBUG");   /* dump each op's A-input + C-output stats to a DETERMINISTIC file
        * (ORKD_SEQ_DEBUG=path, or =1 -> /tmp/orkd-seqdbg.log). Own fd + fsync per seq so it survives a later NPU wedge
        * and never depends on stderr/orkd.log's runtime-dir location. This is the reliable observability for the
        * orkd SW-run() reduce bug (registry: resident int8 softmax seq assembly). */
      if (dbg){
        const char *path = (dbg[0] && strcmp(dbg,"1")) ? dbg : "/tmp/orkd-seqdbg.log";
        int dfd = open(path, O_CREAT|O_WRONLY|O_APPEND, 0644);
        if (dfd >= 0){ char lb[512]; int ln;
            ln = snprintf(lb, sizeof lb, "[seqdbg] n=%d ok=%d rc=%d in_total=%u\n", n, ok, rc, sh.in_total); if(ln>0) (void)!write(dfd, lb, (size_t)ln);
            for (int i = 0; ok && i < n; i++){
                struct orkd_seq_op *o = &ops[i];
                size_t asz = o->a_src ? (o->a_src>=1 && o->a_src<=i ? ops[o->a_src-1].cbytes : 0) : o->abytes;
                long a8 = 0; const int8_t *ain = (const int8_t*)seq[i].A;
                if (ain) for (size_t b = 0; b < asz; b++) a8 += ain[b];
                long c8 = 0, c32 = 0; const int8_t *c8p = (const int8_t*)Cs[i]; const int32_t *c32p = (const int32_t*)Cs[i];
                for (size_t b = 0; b < o->cbytes; b++) c8 += c8p[b];
                for (size_t q = 0; q < o->cbytes/4; q++) c32 += c32p[q];
                ln = snprintf(lb, sizeof lb, "[seqdbg] op%d kind=%u M=%d N=%d a_src=%d c_keep=%u asz=%zu cb=%u | Asum(i8)=%ld Csum(i8)=%ld Csum(i32)=%ld C[i32:0..3]=%d,%d,%d,%d\n",
                        i, o->kind, seq[i].M, seq[i].N, o->a_src, o->c_keep, asz, o->cbytes, a8, c8, c32,
                        c32p[0], o->cbytes>=8?c32p[1]:0, o->cbytes>=12?c32p[2]:0, o->cbytes>=16?c32p[3]:0);
                if(ln>0) (void)!write(dfd, lb, (size_t)ln);
            }
            fsync(dfd); close(dfd);
        }
      }
    }
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.rc = rc;
    int payload = (rc == 0);
    struct orkd_hdr rh = { ORKD_SEQ_OK, (uint32_t)(sizeof hh + (payload ? ctot : 0)), tag };
    int werr = writen(cl->fd, &rh, sizeof rh) || writen(cl->fd, &hh, sizeof hh);
    if (!werr && payload) for (int i = 0; i < n && !werr; i++) if (!ops[i].c_keep) werr = writen(cl->fd, Cs[i], ops[i].cbytes);   /* A2: skip resident intermediates */
    if (Cs) for (int i = 0; i < n; i++) free(Cs[i]);
    free(seq); free(Cs); free(ops); free(inblob);
    return werr ? -1 : 0;
}
/* #2b-2 step 3b: FULL zero-copy RUN — A read AND C written by reference. Client shares A+C dma-bufs (two fds
 * via SCM_RIGHTS); orkd imports both, runs with C written IN PLACE (ORK_ZC_OUT, set at startup → dma_find(C)
 * hit) so NO C byte-transfer either way. Forced SINGLE-CORE: output zero-copy is unsafe under concurrent
 * multi-core (the per-core coherency bsyncs don't serialize with the NPU writes — AGENTS.md ZC-OUT caveat).
 * Client invalidates C (DMA_BUF_SYNC START|READ) before reading. Reply carries no C payload. */
static int handle_run_zc2(struct client *cl, ork_npu *npu, int a_fd, int c_fd, uint64_t tag){
    struct orkd_run rq;
    if (readn(cl->fd, &rq, sizeof rq) <= 0){ if (a_fd >= 0) close(a_fd); if (c_fd >= 0) close(c_fd); return -1; }
    struct cweight *cw = NULL;
    for (int i = 0; i < cl->nw; i++) if (cl->wt[i].id == rq.weight_id){ cw = &cl->wt[i]; break; }
    if (!cw || a_fd < 0 || c_fd < 0){ if (a_fd >= 0) close(a_fd); if (c_fd >= 0) close(c_fd); send_error(cl->fd, tag, cw ? ORKD_EPROTO : ORKD_EBADH, cw ? "need A+C fds" : "unknown weight"); return 0; }
    size_t alen = (size_t)rq.M * cw->K * orkd_esz_a(cw->dtype), cn = (size_t)rq.M * cw->N;
    void *A = ork_dma_import_fd(npu, a_fd, alen);
    if (!A){ close(a_fd); close(c_fd); send_error(cl->fd, tag, ORKD_EOOM, "import A"); return 0; }
    void *C = ork_dma_import_fd(npu, c_fd, cn * 4);
    if (!C){ ork_dma_free(npu, A); close(c_fd); send_error(cl->fd, tag, ORKD_EOOM, "import C"); return 0; }
    struct work *w = wk_alloc();
    if (!w){ ork_dma_free(npu, A); ork_dma_free(npu, C); send_error(cl->fd, tag, ORKD_EOOM, "queue full"); return 0; }
    w->fd = cl->fd; w->type = ORKD_RUN_ZC2; w->tag = tag; w->prio = rq.flags; w->weight_id = rq.weight_id;
    w->domain = rq.domain ? (int)rq.domain : ork_w_domain(cw->w);   /* v2: honor the client-stamped op domain */
    w->M = (int)rq.M; w->K = cw->K; w->N = cw->N; w->dtype = cw->dtype;
    w->A = (int8_t *)A; w->A_imp = A; w->C = (int32_t *)C; w->C_imp = C;   /* dispatch: A read + C written in place, single-core */
    return 0;
}
/* #2b-2 step 3: ZERO-COPY RUN (input A by reference). The client shares A as a dma-buf fd (SCM_RIGHTS);
 * orkd PRIME-imports it into the NPU's IOMMU domain and ork_mm_run_i8 reads A IN PLACE (dma_find hit =
 * validated input zero-copy) — no A byte-transfer over the socket. C is still returned over the socket here
 * (output zero-copy is the next sub-step: needs ORK_ZC_OUT + a cross-process invalidate). */
static int handle_run_zc(struct client *cl, ork_npu *npu, int a_fd, uint64_t tag){
    struct orkd_run rq;
    if (readn(cl->fd, &rq, sizeof rq) <= 0){ if (a_fd >= 0) close(a_fd); return -1; }
    struct cweight *cw = NULL;
    for (int i = 0; i < cl->nw; i++) if (cl->wt[i].id == rq.weight_id){ cw = &cl->wt[i]; break; }
    if (!cw || a_fd < 0){ if (a_fd >= 0) close(a_fd); send_error(cl->fd, tag, cw ? ORKD_EPROTO : ORKD_EBADH, cw ? "no A fd" : "unknown weight"); return 0; }
    size_t alen = (size_t)rq.M * cw->K * orkd_esz_a(cw->dtype);
    void *A = ork_dma_import_fd(npu, a_fd, alen);    /* import A into the NPU domain (takes a_fd ownership) */
    if (!A){ close(a_fd); send_error(cl->fd, tag, ORKD_EOOM, "import A"); return 0; }
    int32_t *C = malloc((size_t)rq.M * cw->N * 4);
    struct work *w = C ? wk_alloc() : NULL;
    if (!C || !w){ ork_dma_free(npu, A); free(C); send_error(cl->fd, tag, ORKD_EOOM, "queue full"); return 0; }
    w->fd = cl->fd; w->type = ORKD_RUN_ZC; w->tag = tag; w->prio = rq.flags; w->weight_id = rq.weight_id;
    w->domain = rq.domain ? (int)rq.domain : ork_w_domain(cw->w);   /* v2: honor the client-stamped op domain */
    w->M = (int)rq.M; w->K = cw->K; w->N = cw->N; w->dtype = cw->dtype; w->A = (int8_t *)A; w->A_imp = A; w->C = C;   /* A zero-copy; C over socket */
    return 0;
}
/* #2b-2 step 1: prove cross-process dma-buf sharing. The client sent a dma-heap fd (SCM_RIGHTS); mmap it here
 * and confirm orkd sees the same bytes (fnv match). This validates the fd-passing + shared-memory plumbing;
 * PRIME-import into the NPU's IOMMU domain (zero-copy submit) is step 2 (needs a library fd hook). */
static int handle_dmabuf(struct client *cl, ork_npu *npu, int dfd, uint64_t tag){
    struct orkd_dmabuf db;
    if (readn(cl->fd, &db, sizeof db) <= 0){ if (dfd >= 0) close(dfd); return -1; }
    struct orkd_dmabuf out; memset(&out, 0, sizeof out); out.size = db.size; out.rc = -1; out.prime_ok = 0;
    if (dfd >= 0 && db.size && db.size <= ORKD_MAX_BYTES){
        void *p = ork_dma_import_fd(npu, dfd, (size_t)db.size);   /* PRIME-import into the NPU's IOMMU domain */
        if (p){
            out.prime_ok = 1;                                    /* NPU-addressable: the real zero-copy path */
            out.checksum = orkd_fnv(p, (size_t)db.size);
            out.rc = (out.checksum == db.checksum) ? 0 : -1;
            ork_dma_free(npu, p);                                /* closes dfd (import took ownership) */
            dfd = -1;
        } else {                                                 /* import failed -> plain shared mmap still proves fd passing */
            void *m = mmap(NULL, db.size, PROT_READ, MAP_SHARED, dfd, 0);
            if (m != MAP_FAILED){ out.checksum = orkd_fnv(m, db.size); out.rc = (out.checksum == db.checksum) ? 0 : -1; munmap(m, db.size); }
        }
    }
    if (dfd >= 0) close(dfd);
    send_msg(cl->fd, ORKD_DMABUF_OK, tag, &out, sizeof out);
    return 0;
}
/* CLIENT-MANAGED IOMMU DOMAINS: hand out / return pool domains so a client can pack its weights into its own
 * isolated ~4 GiB IOVA window (rk_iommu v2's cap is per-domain). orkd coordinates the pool (g_dom_inuse) so no
 * two clients collide; the client tracks the ids it holds and packs into them via orkd_pack.domain. All of a
 * client's domains are released automatically on drop (see the drop path in the main loop). */
static int handle_dom_req(struct client *cl, uint64_t tag){
    int d = dom_alloc_explicit();
    struct orkd_handle hh; memset(&hh, 0, sizeof hh);
    if (d > 0){ cl->owned_dom |= (1u << d); hh.id = (uint64_t)d; hh.rc = 0; }
    else hh.rc = -1;                                     /* pool exhausted */
    send_msg(cl->fd, ORKD_DOM_OK, tag, &hh, sizeof hh);
    return 0;
}
static int handle_dom_rel(struct client *cl, uint64_t tag){
    struct orkd_handle rq;
    if (readn(cl->fd, &rq, sizeof rq) <= 0) return -1;
    int d = (int)rq.id;
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.id = rq.id;
    if (d >= 1 && d <= ORKD_DOMAIN_POOL && (cl->owned_dom & (1u << d))){ cl->owned_dom &= ~(1u << d); dom_release(d); hh.rc = 0; }
    else hh.rc = -1;                                     /* not a domain this client owns */
    send_msg(cl->fd, ORKD_DOM_OK, tag, &hh, sizeof hh);
    return 0;
}
/* A-ring: attach a client's shared ring. The region fd arrived via SCM_RIGHTS on the ORKD_RING_SETUP header;
 * mmap it, validate the layout (magic + geometry the client and daemon must agree on), and keep the mapping.
 * From here ring_service() busy-polls this ring for requests, bypassing the socket for the hot submit path. */
static int handle_ring_setup(struct client *cl, int rfd, uint64_t tag){
    struct orkd_ring_setup rs;
    if (readn(cl->fd, &rs, sizeof rs) <= 0){ if (rfd >= 0) close(rfd); return -1; }
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.rc = -1;
    if (rfd >= 0 && rs.bytes == ORKD_RING_BYTES && !cl->ring){
        void *m = mmap(NULL, ORKD_RING_BYTES, PROT_READ|PROT_WRITE, MAP_SHARED, rfd, 0);
        if (m != MAP_FAILED){
            struct orkd_ring *r = (struct orkd_ring *)m;
            if (r->magic == ORKD_RING_MAGIC && r->nslots == ORKD_RING_SLOTS && r->slot_data == ORKD_RING_SLOT_DATA){
                cl->ring = r; cl->ring_fd = rfd; cl->ring_tail = 0; rfd = -1; hh.rc = 0;
            } else munmap(m, ORKD_RING_BYTES);
        }
    }
    if (rfd >= 0) close(rfd);
    send_msg(cl->fd, ORKD_RING_OK, tag, &hh, sizeof hh);
    return 0;
}
static int32_t g_ring_c[ORKD_RING_SLOT_DATA / 4];   /* daemon-side C scratch (single-threaded; one reused buffer) */
/* Consume ready requests from every attached ring (SPSC: this daemon is the sole consumer; ring_tail is its
 * cursor). Bounded to nslots per ring per call so a saturated ring can't starve the socket/other clients.
 * Returns 1 if any work ran (keeps the loop spinning hot), 0 if all rings were idle. */
static int ring_service(ork_npu *npu, struct client *cl, int nc){
    int did = 0;
    for (int ci = 0; ci < nc; ci++){
        struct orkd_ring *r = cl[ci].ring; if (!r) continue;
        for (int b = 0; b < (int)r->nslots; b++){
            struct orkd_ring_slot *s = &r->slot[cl[ci].ring_tail % r->nslots];
            if (atomic_load_explicit(&s->state, memory_order_acquire) != ORKD_SLOT_REQ) break;   /* acquire: see A */
            ork_w *w = NULL;
            for (int j = 0; j < cl[ci].nw; j++) if (cl[ci].wt[j].id == s->weight_id){ w = cl[ci].wt[j].w; break; }
            int M = (int)s->M, N = (int)s->N; size_t cbytes = (size_t)M * N * 4;
            int rc;
            if (!w || cbytes > sizeof g_ring_c || s->abytes > r->slot_data) rc = -1;
            else {
                int8_t *A = (int8_t *)s->data;   /* A read in place; C into the reused scratch, then copied back */
                rc = (s->dtype == ORKD_DT_F16) ? ork_mm_run(npu, w, M, (const ork_f16 *)A, (float *)g_ring_c)
                   : (s->dtype == ORKD_DT_I4)  ? ork_mm_run_i4(npu, w, M, A, g_ring_c)
                                               : ork_mm_run_i8(npu, w, M, A, g_ring_c);
                if (rc == 0) memcpy(s->data, g_ring_c, cbytes);
            }
            s->rc = rc; s->cbytes = (uint32_t)cbytes;
            atomic_store_explicit(&s->state, ORKD_SLOT_RESP, memory_order_release);   /* publish result: data-then-flag */
            cl[ci].ring_tail++; did = 1;
            g_orkd_submits++; if (cl[ci].ring_tail % r->nslots == 0) g_orkd_ring_wraps++;
        }
    }
    return did;
}
/* free all of a client's resident weights on BYE / socket-EOF — the leak-safe reclaim (a crashed client can't leak) */
static void client_reclaim(struct client *cl, ork_npu *npu){
    for (int i = 0; i < cl->nw; i++) ork_mm_free(npu, cl->wt[i].w);
    cl->nw = 0;
    /* Tier 12f: the kv handles' wkt/wv were registered in wt[] and freed above — free just the kv structs here. */
    for (int i = 0; i < cl->nkv; i++) free(cl->kvt[i].kv);
    cl->nkv = 0;
    if (cl->ring){ munmap(cl->ring, ORKD_RING_BYTES); cl->ring = NULL; }
    if (cl->ring_fd > 0){ close(cl->ring_fd); cl->ring_fd = -1; }
}

/* dispatch ONE quantum of the highest-priority queued item; reply + free when its last rows land.
 * (struct work + wk_* are defined above, before the handlers, since the handlers enqueue.) */
static void dispatch_one(ork_npu *npu, struct client *cl, int nc){
    struct work *w = wk_pick(); if (!w) return;
    ork_w *ow = NULL; int alive = 0;                 /* re-resolve the weight (client/weight may have gone since enqueue) */
    for (int i = 0; i < nc; i++) if (cl[i].fd == w->fd){ alive = 1; for (int j = 0; j < cl[i].nw; j++) if (cl[i].wt[j].id == w->weight_id){ ow = cl[i].wt[j].w; break; } break; }
    if (!alive){ wk_free(npu, w); return; }          /* client gone: drop silently */
    if (!ow){ send_error(w->fd, w->tag, ORKD_EBADH, "weight freed"); wk_free(npu, w); return; }
    g_active_dom = w->domain;                        /* the run below auto-swaps the NPU to this weight's domain */
    int remaining = w->M - w->m0;
    int q = (g_qn == 1) ? remaining : (remaining < ORKD_QUANTUM_ROWS ? remaining : ORKD_QUANTUM_ROWS);   /* adaptive */
    int zc2 = (w->type == ORKD_RUN_ZC2);
    if (zc2) ork_npu_set_core_budget(npu, 1);        /* output zero-copy is single-core-safe only */
    int esz = orkd_esz_a(w->dtype);
    const void *Aoff = (const char *)w->A + (size_t)w->m0 * w->K * esz;   /* A byte-offset (int8=1B, fp16=2B/elem) */
    int32_t *Coff = w->C + (size_t)w->m0 * w->N;                          /* C elem-offset (int32/fp32 both 4B) */
    int r = (w->dtype == ORKD_DT_F16) ? ork_mm_run(npu, ow, q, (const ork_f16 *)Aoff, (float *)Coff)
          : (w->dtype == ORKD_DT_I4)  ? ork_mm_run_i4(npu, ow, q, (const int8_t *)Aoff, Coff)
                                      : ork_mm_run_i8(npu, ow, q, (const int8_t *)Aoff, Coff);
    if (zc2) ork_npu_set_core_budget(npu, 0);
    if (r && !w->rc) w->rc = r;
    w->m0 += q;
    if (w->m0 < w->M) return;                        /* more rows -> stays queued, re-scheduled next tick */
    size_t cn = (size_t)w->M * w->N;
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.id = w->weight_id; hh.rc = w->rc;
    int payload_c = (w->type != ORKD_RUN_ZC2 && w->rc == 0);   /* ZC2's C is the client's shared buffer */
    struct orkd_hdr rh = { ORKD_RUN_OK, (uint32_t)(sizeof hh + (payload_c ? cn * 4 : 0)), w->tag };
    (void)!(writen(w->fd, &rh, sizeof rh) || writen(w->fd, &hh, sizeof hh) || (payload_c && writen(w->fd, w->C, cn * 4)));
    wk_free(npu, w);
}

/* Detach so an auto-spawned orkd outlives the client that fork+exec'd it: double-fork + setsid (reparent to
 * init, no controlling terminal), stdio -> /dev/null + stderr -> <runtime>/orkd.log. The intermediate
 * processes exit fast so the spawning client's waitpid reaps them immediately. ORKD_FOREGROUND skips this
 * (tests / debugging). Called BEFORE the flock so the surviving daemon grandchild owns the lock. */
static void daemonize(void){
    if (getenv("ORKD_FOREGROUND")) return;
    pid_t p = fork(); if (p < 0) return; if (p > 0) _exit(0);
    setsid();
    p = fork(); if (p > 0) _exit(0);              /* not a session leader -> can't reacquire a tty */
    if (chdir("/") < 0) {}
    char logp[256]; snprintf(logp, sizeof logp, "%s/orkd.log", orkd_runtime_dir());
    int nul = open("/dev/null", O_RDWR);
    int lg  = open(logp, O_CREAT|O_WRONLY|O_APPEND, 0600);
    if (nul >= 0){ dup2(nul, 0); dup2(nul, 1); if (nul > 2) close(nul); }
    if (lg  >= 0){ dup2(lg, 2);  if (lg  > 2) close(lg); }
    /* stderr -> a regular file is FULLY buffered by default, so log lines are lost if the daemon is SIGKILL'd
     * (e.g. a wedge post-mortem — exactly when the log matters). Line-buffer it so each line hits the file. */
    setvbuf(stderr, NULL, _IOLBF, 0);
}

/* Prime the NPU cold paths before serving so a client's first op is never cold. Notably the int16
 * activation-LUT calibration (silu_calibrate_idx16) is lazy and can wedge if its first run is the first op
 * AFTER a multi-core matmul (OPS_REGISTRY): here we run a tiny exp_i16 FIRST (clean calibration, no prior
 * matmul) then a tiny fp16 matmul (SW-stream path). Uses ork_submit_seq — the exact serving dispatch. Both
 * results discarded; best-effort (non-fatal). Disable with ORKD_NO_WARMUP. */
static void orkd_warmup(ork_npu *npu){
    if (!npu || getenv("ORKD_NO_WARMUP")) return;
    enum { WN = 32 };
    int16_t ein[WN], eout[WN]; for (int i = 0; i < WN; i++) ein[i] = (int16_t)(500 + i);
    ork_f16 bf[WN*WN]; for (int i = 0; i < WN*WN; i++) bf[i] = (ork_f16)1.0f;
    ork_f16 af[WN];    for (int i = 0; i < WN; i++)    af[i] = (ork_f16)1.0f;
    float cf[WN];
    ork_w *wf = ork_mm_pack(npu, WN, WN, bf);
    ork_seq_op ops[2] = {
        { .kind = ORK_OP_EXP_I16, .M = 1, .N = WN, .A = ein, .C = eout, .in_scale = 1.0/1024.0, .out_scale = 1.0/32000.0 },
        { .kind = ORK_OP_MM_F16,  .w = wf, .M = 1, .A = af, .C = cf },
    };
    int rc = ork_submit_seq(npu, ops, wf ? 2 : 1);
    if (wf) ork_mm_free(npu, wf);
    fprintf(stderr, "[orkd] warmup rc=%d (int16-LUT calibration + fp16 matmul primed)\n", rc);
    /* Bring up ALL cores (ORKD_ATTN_RR precondition: a cold core's first submit wedges). A single-task int8
     * matmul with M>1,N=512 dispatches run_multicore across cores 0..nc-1, priming each — the same warm the
     * direct chainrr probes do before ork_mm_run_chains_rr[_biased]. Without this the first RR dispatch fans a
     * chain onto a never-online core and wedges. */
    { int WK=512, WN=512, WM=32;
      int8_t *wb=malloc((size_t)WK*WN), *wa=malloc((size_t)WM*WK); int32_t *wc=malloc((size_t)WM*WN*4);
      if (wb && wa && wc){ memset(wb,1,(size_t)WK*WN); memset(wa,1,(size_t)WM*WK);
          ork_w *wi=ork_mm_pack_i8(npu,WK,WN,wb);
          if (wi){ ork_mm_task_i8 wt={wi,WM,wa,wc}; int mrc=ork_mm_run_chain_i8(npu,1,&wt); ork_mm_free(npu,wi);
              fprintf(stderr, "[orkd] warmup multicore-i8 rc=%d (all cores primed for RR)\n", mrc); } }
      free(wb); free(wa); free(wc); }
}

int main(void){
    char sockpath[256], pidpath[256];
    orkd_sock_path(sockpath, sizeof sockpath);
    orkd_pidfile_path(pidpath, sizeof pidpath);

    daemonize();                                   /* detach first; the surviving grandchild takes the flock */

    /* Single-instance: hold an exclusive flock on the pidfile. A losing racer (two clients both auto-spawned an
     * orkd) exits cleanly here and its client then connects to the winner's socket. */
    int pidfd = open(pidpath, O_CREAT|O_RDWR, 0600);
    if (pidfd < 0){ perror("orkd: open pidfile"); return 1; }
    if (flock(pidfd, LOCK_EX|LOCK_NB) < 0){
        if (errno == EWOULDBLOCK){ /* another orkd already owns the NPU */ return 0; }
        perror("orkd: flock"); return 1;
    }
    { char b[32]; int n = snprintf(b, sizeof b, "%d\n", (int)getpid());
      if (ftruncate(pidfd, 0) < 0) {} if (write(pidfd, b, (size_t)n) < 0) {} }

    signal(SIGPIPE, SIG_IGN);                 /* a client dying mid-write must not kill the daemon */
    signal(SIGTERM, on_signal);
    signal(SIGINT,  on_signal);

    int lfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (lfd < 0){ perror("orkd: socket"); return 1; }
    struct sockaddr_un sa; memset(&sa, 0, sizeof sa); sa.sun_family = AF_UNIX;
    snprintf(sa.sun_path, sizeof sa.sun_path, "%s", sockpath);
    unlink(sockpath);                          /* clear a stale socket from a prior crash */
    if (bind(lfd, (struct sockaddr*)&sa, sizeof sa) < 0){ perror("orkd: bind"); return 1; }
    if (listen(lfd, 16) < 0){ perror("orkd: listen"); return 1; }

    setenv("ORKD_IS_DAEMON", "1", 1);          /* Path B: orkd's OWN ork_npu_init takes the DIRECT NPU path, not the client route (no self-loop) */
    setenv("ORK_ZC_OUT", "1", 1);              /* enable output zero-copy: a run whose C is an imported dma-buf
                                                * (dma_find hit) writes it in place; malloc'd C (socket runs) is
                                                * a dma_find miss -> unaffected. RUN_ZC2 forces single-core (safe). */
    ork_npu *npu = ork_npu_init();             /* orkd OWNS the NPU for its whole lifetime (#2a) */
    int cores = npu ? ork_npu_cores(npu) : 0;
    unsigned idle_ms = orkd_idle_ms();
    g_per_client_dom = getenv("ORKD_PER_CLIENT_DOMAINS") != NULL;   /* A-sched: opt-in per-client IOMMU domains */
    fprintf(stderr, "[orkd] up pid=%d sock=%s idle=%ums npu=%s cores=%d per_client_dom=%d\n",
            (int)getpid(), sockpath, idle_ms, npu ? "ok" : "FAILED", cores, g_per_client_dom);
    orkd_warmup(npu);                          /* prime cold paths (int16-LUT calibration + matmul) before serving */

    struct client cl[ORKD_MAX_CLIENTS]; int nc = 0;
    uint32_t next_id = 1, refs = 0;
    long idle_since = now_ms();                /* refs==0 since this time */

    /* A-ring IDLE-BACKOFF: an attached ring is polled from shared memory (no fd to block on), so servicing it
     * means busy-spinning (timeout=0) — which pegs a core even when no submits are coming. Instead SPIN only for
     * a short hot window after the ring's last activity (covers a decode burst's inter-op gaps at ~µs latency),
     * then BACK OFF to a small poll timeout so an idle ring costs ~nothing (wakes every backoff_ms to re-check).
     * Cold-start latency after idle is bounded by backoff_ms; the first op re-enters the hot window. Tunable. */
    long ring_hot = now_ms(), ring_spin_ms = 3, ring_backoff_ms = 2;
    { const char *e = getenv("ORKD_RING_SPIN_MS"); if (e && *e){ long v = atol(e); if (v >= 0) ring_spin_ms = v; }
      e = getenv("ORKD_RING_BACKOFF_MS"); if (e && *e){ long v = atol(e); if (v > 0) ring_backoff_ms = v; } }

    while (!g_stop){
        struct pollfd pfd[ORKD_MAX_CLIENTS + 1];
        pfd[0].fd = lfd; pfd[0].events = POLLIN;
        for (int i = 0; i < nc; i++){ pfd[i+1].fd = cl[i].fd; pfd[i+1].events = POLLIN; }

        int nring = 0;                         /* A-ring: # of clients with an attached ring (busy-poll if any) */
        for (int i = 0; i < nc; i++) if (cl[i].ring) nring++;

        int timeout = -1;                      /* block indefinitely while we have subscribers */
        if (g_qn > 0) timeout = 0;             /* queued work: poll non-blocking, then dispatch a quantum */
        else if (nring > 0){                   /* attached ring: spin the hot window, then back off (idle-backoff) */
            long ring_idle = now_ms() - ring_hot;
            timeout = (ring_idle < ring_spin_ms) ? 0 : (int)ring_backoff_ms;
        }
        else if (refs == 0){
            long left = (long)idle_ms - (now_ms() - idle_since);
            timeout = left > 0 ? (int)left : 0;
        }
        int pr = poll(pfd, (nfds_t)(nc+1), timeout);
        if (pr < 0){ if (errno == EINTR) continue; perror("orkd: poll"); break; }
        if (pr == 0 && g_qn == 0 && nring == 0){   /* pure idle timeout: reap if no subscribers */
            if (refs == 0 && now_ms() - idle_since >= (long)idle_ms){
                fprintf(stderr, "[orkd] idle %ums, no subscribers -> reap (submits=%llu ring_wraps=%llu)\n", idle_ms,
                        (unsigned long long)g_orkd_submits, (unsigned long long)g_orkd_ring_wraps);
                break;
            }
            continue;
        }

        int npoll = nc;                        /* # of client fds poll() examined THIS tick (before accept grows nc) */

        if (pfd[0].revents & POLLIN){          /* new connection */
            int cfd = accept(lfd, NULL, NULL);
            if (cfd >= 0){
                if (nc < ORKD_MAX_CLIENTS){ cl[nc].fd = cfd; cl[nc].hello = 0; cl[nc].id = 0; cl[nc].nw = 0; cl[nc].next_wid = 0; cl[nc].domain = dom_alloc(); cl[nc].owned_dom = 0; cl[nc].ring = NULL; cl[nc].ring_fd = 0; cl[nc].ring_tail = 0; nc++; }
                else { send_error(cfd, 0, ORKD_EOOM, "too many clients"); close(cfd); }
            }
        }
        /* Service ONLY the client fds poll() examined (i < npoll). A connection accepted just above sits at
         * index >= npoll and its pollfd slot was never filled by poll() — reading pfd[i+1].revents there is
         * uninitialized stack garbage, and a spurious POLLIN would make the daemon recvmsg on a client that
         * hasn't spoken yet and BLOCK the whole single-threaded loop. New clients are serviced next tick (poll
         * includes them once nc has grown). THIS was the >=3-consumer hang: more accepts -> more chances to read
         * a stale revents and deadlock. On a drop we compact but must NOT re-examine slot i (its pollfd is now
         * stale for the moved-in client) — it too is serviced next tick. */
        for (int i = 0; i < npoll && i < nc; i++){
            if (!(pfd[i+1].revents & (POLLIN|POLLHUP|POLLERR))) continue;
            int drop = 0, recvd_fds[2] = { -1, -1 }, recvd_nfd = 0;
            struct orkd_hdr h;
            int rr = orkd_recv_hdr_fds(cl[i].fd, &h, recvd_fds, 2, &recvd_nfd);   /* recvmsg: captures up to 2 SCM_RIGHTS fds (A,C) */
            if (rr <= 0){ drop = 1; }          /* EOF (incl. abrupt client death) or error */
            else switch (h.type){
                case ORKD_HELLO: {
                    struct orkd_hello he;
                    if (readn(cl[i].fd, &he, sizeof he) <= 0){ drop = 1; break; }
                    struct orkd_welcome w; memset(&w, 0, sizeof w);
                    w.proto = ORKD_PROTO_VERSION; w.client_id = cl[i].id = next_id++;
                    w.soc_cores = (uint32_t)cores;
                    if (!cl[i].hello){ cl[i].hello = 1; refs++; }
                    send_msg(cl[i].fd, ORKD_WELCOME, h.tag, &w, sizeof w);
                    break;
                }
                case ORKD_PING: send_msg(cl[i].fd, ORKD_PONG, h.tag, NULL, 0); break;
                case ORKD_BYE:  drop = 1; break;
                case ORKD_PACK: if (handle_pack(&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_KV_ALLOC:  if (handle_kv_alloc (&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_KV_APPEND: g_orkd_submits++; if (handle_kv_append(&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_IMPORT: if (handle_import(&cl[i], npu, recvd_fds[0], recvd_fds[1], h.tag) < 0) drop = 1; recvd_fds[0] = recvd_fds[1] = -1; break;
                case ORKD_FFN: g_orkd_submits++; if (handle_ffn(&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_ATTN: g_orkd_submits++; if (handle_attn(&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_ATTN_RR: g_orkd_submits++; if (handle_attn_rr(&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_RUN:  g_orkd_submits++; if (handle_run (&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_FREE: if (handle_free(&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_SDP:  g_orkd_submits++; if (handle_sdp (&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_CHAIN: g_orkd_submits++; if (handle_chain(&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_SEQ:  g_orkd_submits++; if (handle_seq (&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_RUN_ZC: g_orkd_submits++; if (handle_run_zc(&cl[i], npu, recvd_fds[0], h.tag) < 0) drop = 1; recvd_fds[0] = -1; break;
                case ORKD_RUN_ZC2: g_orkd_submits++; if (handle_run_zc2(&cl[i], npu, recvd_fds[0], recvd_fds[1], h.tag) < 0) drop = 1; recvd_fds[0] = recvd_fds[1] = -1; break;
                case ORKD_DMABUF_PROBE: if (handle_dmabuf(&cl[i], npu, recvd_fds[0], h.tag) < 0) drop = 1; recvd_fds[0] = -1; break;
                case ORKD_RING_SETUP: if (handle_ring_setup(&cl[i], recvd_fds[0], h.tag) < 0) drop = 1; recvd_fds[0] = -1; break;
                case ORKD_DOM_REQ: if (handle_dom_req(&cl[i], h.tag) < 0) drop = 1; break;
                case ORKD_DOM_REL: if (handle_dom_rel(&cl[i], h.tag) < 0) drop = 1; break;
                default: send_error(cl[i].fd, h.tag, ORKD_EPROTO, "unknown message"); break;
            }
            for (int f = 0; f < 2; f++) if (recvd_fds[f] >= 0) close(recvd_fds[f]);   /* stray fds not consumed by a handler */
            if (drop){
                if (cl[i].hello && refs) refs--;
                wk_purge_fd(npu, cl[i].fd);    /* drop this client's queued work (prevents fd-reuse aliasing) */
                client_reclaim(&cl[i], npu);   /* free the client's resident weights (leak-safe on BYE/EOF) */
                dom_release(cl[i].domain);     /* return the client's auto IOMMU domain to the pool */
                for (int d = 1; d <= ORKD_DOMAIN_POOL; d++) if (cl[i].owned_dom & (1u << d)) dom_release(d);   /* + any it explicitly requested (leak-safe) */
                close(cl[i].fd);
                cl[i] = cl[nc-1]; nc--;        /* compact; NO i-- (slot i's pollfd is now stale — service next tick) */
                if (refs == 0) idle_since = now_ms();
            }
        }
        if (g_qn > 0) dispatch_one(npu, cl, nc);   /* one quantum of the highest-priority queued run per tick */
        if (nring > 0 && ring_service(npu, cl, nc)) ring_hot = now_ms();   /* A-ring: drain ready requests; refresh the hot window on any activity (idle-backoff) */
    }

    if (npu) ork_npu_free(npu);                /* release NPU/IOMMU cleanly */
    close(lfd); unlink(sockpath);
    flock(pidfd, LOCK_UN); close(pidfd); unlink(pidpath);
    fprintf(stderr, "[orkd] down (submits=%llu ring_wraps=%llu)\n",
            (unsigned long long)g_orkd_submits, (unsigned long long)g_orkd_ring_wraps);
    return 0;
}
