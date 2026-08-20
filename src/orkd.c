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
#include "orkd_internal.h"
#include "spine_kernels.h"   /* ORKD_LAYER: the CPU glue kernels the daemon runs on the resident activation */

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
int orkd_readn(int fd, void *buf, size_t n){
    char *p = buf; size_t got = 0;
    while (got < n){
        ssize_t r = read(fd, p+got, n-got);
        if (r == 0) return 0;                       /* EOF */
        if (r < 0){ if (errno == EINTR) continue; return -1; }
        got += (size_t)r;
    }
    return 1;
}
int orkd_writen(int fd, const void *buf, size_t n){
    const char *p = buf; size_t put = 0;
    while (put < n){
        ssize_t r = write(fd, p+put, n-put);
        if (r < 0){ if (errno == EINTR) continue; return -1; }
        put += (size_t)r;
    }
    return 0;
}
int orkd_send_msg(int fd, uint32_t type, uint64_t tag, const void *payload, uint32_t len){
    struct orkd_hdr h = { type, len, tag };
    if (orkd_writen(fd, &h, sizeof h)) return -1;
    if (len && orkd_writen(fd, payload, len)) return -1;
    return 0;
}
void orkd_send_error(int fd, uint64_t tag, uint32_t code, const char *msg){
    struct orkd_error e; memset(&e, 0, sizeof e); e.code = code;
    snprintf(e.msg, sizeof e.msg, "%s", msg ? msg : "");
    orkd_send_msg(fd, ORKD_ERROR, tag, &e, sizeof e);
}

/* Per-client resident-weight table. A real model client packs one weight per matmul per layer and holds them
 * all resident (e.g. `model 12` = 7 matmuls x 12 layers = 84), and ork_w_free (no ctx) can't send a daemon-free
 * RPC so weights are reclaimed only on the client's socket-EOF — so this must comfortably exceed a model's
 * full resident set, not just a handful. */


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
 * (orkd_dom_alloc_explicit, works regardless of the env knob) and packs into it (orkd_pack.domain). Both draw from
 * the same g_dom_inuse pool; all a client's domains (auto + requested) are released on drop. */
/* No fixed domain-pool cap: the daemon hands out as many domains as the client's auto-sizer requests.
 * The only ceiling is the per-client owned_dom bitmask width (uint64 -> domains 1..63, ~155 GiB of IOVA —
 * far beyond any RK3588 config), so the auto-sizer's count is never clamped in practice. */
static int g_per_client_dom = 0;              /* ORKD_PER_CLIENT_DOMAINS: auto-assign a domain per client on accept */
static unsigned char g_dom_inuse[ORKD_NDOM];  /* index 1..63; 0 unused (always-available shared) */
int orkd_dom_alloc_explicit(void){          /* grab a free pool domain (for an explicit ORKD_DOM_REQ) -> 1..63, or 0 if exhausted */
    for (int d = 1; d < ORKD_NDOM; d++) if (!g_dom_inuse[d]){ g_dom_inuse[d] = 1; return d; }
    return 0;
}
static int dom_alloc(void){                   /* auto per-client domain on accept (gated by the env knob); 0 = shared */
    return g_per_client_dom ? orkd_dom_alloc_explicit() : 0;
}
void orkd_dom_release(int d){ if (d >= 1 && d < ORKD_NDOM) g_dom_inuse[d] = 0; }
static int g_active_dom = 0;                   /* domain of the last-dispatched work (what the NPU is warmed to) */

/* drain n bytes into the void — keeps the stream in sync when a request can't be serviced */
int orkd_drain(int fd, size_t n){ char b[4096]; while (n){ size_t k = n > sizeof b ? sizeof b : n; if (orkd_readn(fd, b, k) <= 0) return -1; n -= k; } return 0; }

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
int orkd_esz_a(int wire_dt){ return wire_dt == ORKD_DT_F16 ? 2 : 1; }
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
struct work *orkd_wk_alloc(void){
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
    if (!ow){ orkd_send_error(w->fd, w->tag, ORKD_EBADH, "weight freed"); wk_free(npu, w); return; }
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
    (void)!(orkd_writen(w->fd, &rh, sizeof rh) || orkd_writen(w->fd, &hh, sizeof hh) || (payload_c && orkd_writen(w->fd, w->C, cn * 4)));
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
                else { orkd_send_error(cfd, 0, ORKD_EOOM, "too many clients"); close(cfd); }
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
                    if (orkd_readn(cl[i].fd, &he, sizeof he) <= 0){ drop = 1; break; }
                    struct orkd_welcome w; memset(&w, 0, sizeof w);
                    w.proto = ORKD_PROTO_VERSION; w.client_id = cl[i].id = next_id++;
                    w.soc_cores = (uint32_t)cores;
                    if (!cl[i].hello){ cl[i].hello = 1; refs++; }
                    orkd_send_msg(cl[i].fd, ORKD_WELCOME, h.tag, &w, sizeof w);
                    break;
                }
                case ORKD_PING: orkd_send_msg(cl[i].fd, ORKD_PONG, h.tag, NULL, 0); break;
                case ORKD_BYE:  drop = 1; break;
                case ORKD_PACK: if (orkd_handle_pack(&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_KV_ALLOC:  if (orkd_handle_kv_alloc (&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_KV_APPEND: g_orkd_submits++; if (orkd_handle_kv_append(&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_IMPORT: if (orkd_handle_import(&cl[i], npu, recvd_fds[0], recvd_fds[1], h.tag) < 0) drop = 1; recvd_fds[0] = recvd_fds[1] = -1; break;
                case ORKD_FFN: g_orkd_submits++; if (orkd_handle_ffn(&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_ATTN: g_orkd_submits++; if (orkd_handle_attn(&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_ATTN_RR: g_orkd_submits++; if (orkd_handle_attn_rr(&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_LAYER: g_orkd_submits++; if (orkd_handle_layer(&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_RUN:  g_orkd_submits++; if (orkd_handle_run (&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_FREE: if (orkd_handle_free(&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_SDP:  g_orkd_submits++; if (orkd_handle_sdp (&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_CHAIN: g_orkd_submits++; if (orkd_handle_chain(&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_SEQ:  g_orkd_submits++; if (orkd_handle_seq (&cl[i], npu, h.tag) < 0) drop = 1; break;
                case ORKD_RUN_ZC: g_orkd_submits++; if (orkd_handle_run_zc(&cl[i], npu, recvd_fds[0], h.tag) < 0) drop = 1; recvd_fds[0] = -1; break;
                case ORKD_RUN_ZC2: g_orkd_submits++; if (orkd_handle_run_zc2(&cl[i], npu, recvd_fds[0], recvd_fds[1], h.tag) < 0) drop = 1; recvd_fds[0] = recvd_fds[1] = -1; break;
                case ORKD_DMABUF_PROBE: if (orkd_handle_dmabuf(&cl[i], npu, recvd_fds[0], h.tag) < 0) drop = 1; recvd_fds[0] = -1; break;
                case ORKD_RING_SETUP: if (orkd_handle_ring_setup(&cl[i], recvd_fds[0], h.tag) < 0) drop = 1; recvd_fds[0] = -1; break;
                case ORKD_DOM_REQ: if (orkd_handle_dom_req(&cl[i], h.tag) < 0) drop = 1; break;
                case ORKD_DOM_REL: if (orkd_handle_dom_rel(&cl[i], h.tag) < 0) drop = 1; break;
                default: orkd_send_error(cl[i].fd, h.tag, ORKD_EPROTO, "unknown message"); break;
            }
            for (int f = 0; f < 2; f++) if (recvd_fds[f] >= 0) close(recvd_fds[f]);   /* stray fds not consumed by a handler */
            if (drop){
                if (cl[i].hello && refs) refs--;
                wk_purge_fd(npu, cl[i].fd);    /* drop this client's queued work (prevents fd-reuse aliasing) */
                client_reclaim(&cl[i], npu);   /* free the client's resident weights (leak-safe on BYE/EOF) */
                orkd_dom_release(cl[i].domain);     /* return the client's auto IOMMU domain to the pool */
                for (int d = 1; d < ORKD_NDOM; d++) if (cl[i].owned_dom & (1ull << d)) orkd_dom_release(d);   /* + any it explicitly requested (leak-safe) */
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
