/* orkd_client.c — connect/auto-spawn/ping/disconnect against the orkd daemon. See orkd_client.h. */
#include "orkd_client.h"
#include "orkd_client_internal.h"
#include "orkd_proto.h"
#include "orkd_shm.h"
#include "orkd_ring.h"

#include <errno.h>
#include <stdatomic.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define ORKD_BIN_ENV "ORKD_BIN"
#define ORKD_CONNECT_TRIES 200          /* * 15ms ~= 3s to let an auto-spawned daemon bind */
#define ORKD_CONNECT_BACKOFF_NS (15*1000*1000L)


/* live connections, for the atexit clean-BYE (small fixed set; a process rarely holds many) */
static orkd_conn *g_live[16];
static int g_nlive = 0, g_atexit_armed = 0;


static void orkd_atexit(void){
    for (int i = 0; i < g_nlive; i++){
        orkd_conn *c = g_live[i];
        if (c && c->fd >= 0){ struct orkd_hdr h = { ORKD_BYE, 0, 0 }; (void)wn(c->fd, &h, sizeof h); close(c->fd); c->fd = -1; }
    }
}

/* fork+exec orkd; it double-forks + daemonizes, so the intermediate we exec exits fast -> waitpid reaps it. */
static void orkd_spawn(void){
    const char *bin = getenv(ORKD_BIN_ENV);
    pid_t p = fork();
    if (p == 0){
        if (bin && *bin) execl(bin, bin, (char*)0);
        execlp("orkd", "orkd", (char*)0);
        _exit(127);                       /* exec failed */
    } else if (p > 0){
        int st; while (waitpid(p, &st, 0) < 0 && errno == EINTR){}
    }
}

static int try_connect(const char *sp){
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_un sa; memset(&sa, 0, sizeof sa); sa.sun_family = AF_UNIX;
    snprintf(sa.sun_path, sizeof sa.sun_path, "%s", sp);
    if (connect(fd, (struct sockaddr*)&sa, sizeof sa) == 0) return fd;
    close(fd); return -1;
}

orkd_conn *orkd_connect(void){
    char sp[256]; orkd_sock_path(sp, sizeof sp);
    int spawned = 0, fd = -1;
    for (int a = 0; a < ORKD_CONNECT_TRIES && fd < 0; a++){
        fd = try_connect(sp);
        if (fd < 0){
            if (!spawned){ orkd_spawn(); spawned = 1; }     /* first miss: bring one up */
            struct timespec ts = { 0, ORKD_CONNECT_BACKOFF_NS }; nanosleep(&ts, 0);
        }
    }
    if (fd < 0) return 0;

    struct orkd_hello he = { ORKD_PROTO_VERSION, (uint32_t)getpid() };
    struct orkd_hdr h = { ORKD_HELLO, (uint32_t)sizeof he, 1 };
    if (wn(fd, &h, sizeof h) || wn(fd, &he, sizeof he)){ close(fd); return 0; }
    struct orkd_welcome w;
    if (rn(fd, &h, sizeof h) <= 0 || h.type != ORKD_WELCOME || rn(fd, &w, sizeof w) <= 0 || w.proto != ORKD_PROTO_VERSION){
        close(fd); return 0;
    }
    orkd_conn *c = calloc(1, sizeof *c);
    if (!c){ close(fd); return 0; }
    c->fd = fd; c->client_id = w.client_id; c->soc_cores = w.soc_cores;
    if (!g_atexit_armed){ atexit(orkd_atexit); g_atexit_armed = 1; }
    if (g_nlive < (int)(sizeof g_live / sizeof g_live[0])) g_live[g_nlive++] = c;
    return c;
}

int orkd_ping(orkd_conn *c){
    if (!c || c->fd < 0) return -1;
    struct orkd_hdr h = { ORKD_PING, 0, 2 };
    if (wn(c->fd, &h, sizeof h)) return -1;
    if (rn(c->fd, &h, sizeof h) <= 0 || h.type != ORKD_PONG) return -1;
    return 0;
}

void orkd_set_priority(orkd_conn *c, unsigned prio){ if (c) c->prio = prio; }

/* Client-managed IOMMU domains. orkd_domain_alloc asks the daemon for an isolated pool domain (coordinated so
 * clients don't collide); orkd_set_pack_domain then makes subsequent packs (orkd_pack_*) land the weight there.
 * orkd_domain_free returns it. All held domains are also reclaimed automatically when the connection drops. */
void orkd_set_pack_domain(orkd_conn *c, uint32_t domain){ if (c) c->pack_domain = domain; }
/* v2: set the domain the daemon should activate for the NEXT op-submit. The library calls this from each routed
 * run/chain/seq wrapper with the weight's domain, so every op/chain submitted carries its domain id. */
void orkd_set_op_domain(orkd_conn *c, uint32_t domain){ if (c) c->op_domain = domain; }

int orkd_domain_alloc(orkd_conn *c){
    if (!c || c->fd < 0) return -1;
    struct orkd_hdr h = { ORKD_DOM_REQ, 0, 7 };
    if (wn(c->fd, &h, sizeof h)) return -1;
    struct orkd_hdr rh;
    if (rn(c->fd, &rh, sizeof rh) <= 0) return -1;
    if (rh.type != ORKD_DOM_OK){ if (rh.len) cdrain(c->fd, rh.len); return -1; }
    struct orkd_handle hh;
    if (rn(c->fd, &hh, sizeof hh) <= 0) return -1;
    if (rh.len > sizeof hh) cdrain(c->fd, rh.len - sizeof hh);
    return hh.rc == 0 ? (int)hh.id : -1;
}
int orkd_domain_free(orkd_conn *c, int domain){
    if (!c || c->fd < 0 || domain <= 0) return -1;
    struct orkd_handle rq; memset(&rq, 0, sizeof rq); rq.id = (uint64_t)domain;
    struct orkd_hdr h = { ORKD_DOM_REL, (uint32_t)sizeof rq, 7 };
    if (wn(c->fd, &h, sizeof h) || wn(c->fd, &rq, sizeof rq)) return -1;
    if (c->pack_domain == (uint32_t)domain) c->pack_domain = 0;   /* stop packing into a freed domain */
    struct orkd_hdr rh;
    if (rn(c->fd, &rh, sizeof rh) <= 0) return -1;
    if (rh.type != ORKD_DOM_OK){ if (rh.len) cdrain(c->fd, rh.len); return -1; }
    struct orkd_handle hh;
    if (rn(c->fd, &hh, sizeof hh) <= 0) return -1;
    if (rh.len > sizeof hh) cdrain(c->fd, rh.len - sizeof hh);
    return hh.rc;
}

/* A-ring: attach a shared low-latency ring. Allocates the region (memfd), maps it, passes the fd to the daemon
 * over the control socket (SCM_RIGHTS), and awaits RING_OK. Returns 0 iff both sides mapped it. After this,
 * orkd_run_i8_ring bypasses the socket for the hot submit path. Idempotent-ish: a second call is a no-op (0). */
int orkd_ring_setup(orkd_conn *c){
    if (!c || c->fd < 0) return -1;
    if (c->ring) return 0;
    int mfd = (int)syscall(SYS_memfd_create, "orkd_ring", 0u);
    if (mfd < 0) return -1;
    if (ftruncate(mfd, (off_t)ORKD_RING_BYTES) < 0){ close(mfd); return -1; }
    void *m = mmap(NULL, ORKD_RING_BYTES, PROT_READ|PROT_WRITE, MAP_SHARED, mfd, 0);
    if (m == MAP_FAILED){ close(mfd); return -1; }
    struct orkd_ring *r = (struct orkd_ring *)m;
    memset(r, 0, sizeof *r);
    r->magic = ORKD_RING_MAGIC; r->nslots = ORKD_RING_SLOTS; r->slot_data = ORKD_RING_SLOT_DATA;
    struct orkd_ring_setup rs; memset(&rs, 0, sizeof rs); rs.bytes = ORKD_RING_BYTES;
    struct orkd_hdr h = { ORKD_RING_SETUP, (uint32_t)sizeof rs, 6 };
    if (orkd_send_hdr_fd(c->fd, &h, mfd) || wn(c->fd, &rs, sizeof rs)){ munmap(m, ORKD_RING_BYTES); close(mfd); return -1; }
    struct orkd_hdr rh;
    if (rn(c->fd, &rh, sizeof rh) <= 0 || rh.type != ORKD_RING_OK){ if (rh.len) cdrain(c->fd, rh.len); munmap(m, ORKD_RING_BYTES); close(mfd); return -1; }
    struct orkd_handle hh;
    if (rn(c->fd, &hh, sizeof hh) <= 0 || hh.rc != 0){ munmap(m, ORKD_RING_BYTES); close(mfd); return -1; }
    c->ring = r; c->ring_fd = mfd; c->ring_next = 0;
    return 0;
}

/* Hot path: submit one int8/fp16/int4 matmul through the ring (no socket). A copied in, C copied out; the daemon
 * busy-polls the slot, runs the matmul, writes C in place. 0 ok / <0 error (incl. -2 = too big for a slot ->
 * caller should fall back to the socket path). dtype is a wire ORKD_DT_*. */
int orkd_has_ring(orkd_conn *c){ return c && c->ring ? 1 : 0; }

/* A is int8/int4 (1B/elem) or fp16 (2B/elem); C is always 4B/elem (int32/fp32). */
static size_t ring_esz_a(uint32_t dtype){ return dtype == ORKD_DT_F16 ? 2 : 1; }

/* ASYNC, precision-agnostic. submit enqueues one op (any dtype) into a free slot WITHOUT blocking and returns a
 * ticket (the slot index) to collect later; -2 = won't fit a slot or the ring is full (caller waits / falls back
 * to the socket). collect blocks (spins) until that ticket's result lands, copies C, frees the slot. Up to
 * ORKD_RING_SLOTS ops can be in flight — that is the pipeline depth: while the daemon runs op N, the client fills
 * op N+1's slot, so the per-op transport (memcpy + handshake) overlaps NPU compute instead of serializing. */
int orkd_ring_submit(orkd_conn *c, uint64_t weight_id, int M, int K, int N, uint32_t dtype, const void *A){
    if (!c || !c->ring || M <= 0 || K <= 0 || N <= 0 || !A) return -1;
    struct orkd_ring *r = c->ring;
    size_t abytes = (size_t)M * K * ring_esz_a(dtype), cbytes = (size_t)M * N * 4;
    if (abytes > r->slot_data || cbytes > r->slot_data) return -2;   /* op too big for a slot -> use the socket */
    uint32_t idx = c->ring_next % r->nslots;
    struct orkd_ring_slot *s = &r->slot[idx];
    if (atomic_load_explicit(&s->state, memory_order_acquire) != ORKD_SLOT_EMPTY) return -2;  /* ring full: collect first */
    s->weight_id = weight_id; s->M = (uint32_t)M; s->K = (uint32_t)K; s->N = (uint32_t)N; s->dtype = dtype;
    s->abytes = (uint32_t)abytes; s->cbytes = (uint32_t)cbytes; s->rc = 0; s->domain = c->op_domain;
    memcpy(s->data, A, abytes);
    atomic_store_explicit(&s->state, ORKD_SLOT_REQ, memory_order_release);   /* publish: data-then-flag */
    c->ring_next++;
    return (int)idx;
}
int orkd_ring_collect(orkd_conn *c, int ticket, void *C){
    if (!c || !c->ring || ticket < 0 || ticket >= (int)c->ring->nslots || !C) return -1;
    struct orkd_ring_slot *s = &c->ring->slot[ticket];
    while (atomic_load_explicit(&s->state, memory_order_acquire) != ORKD_SLOT_RESP){
        if (atomic_load_explicit(&c->ring->stop, memory_order_acquire)) return -1;
    }
    int rc = s->rc;
    if (rc == 0 && s->cbytes) memcpy(C, s->data, s->cbytes);   /* cbytes is daemon-authoritative (== M*N*4) */
    atomic_store_explicit(&s->state, ORKD_SLOT_EMPTY, memory_order_release);  /* release the slot */
    return rc;
}
/* Synchronous convenience (submit + immediately collect) — precision-agnostic; kept for the probe/back-compat. */
int orkd_ring_run(orkd_conn *c, uint64_t weight_id, int M, int K, int N, uint32_t dtype, const void *A, void *C){
    int t = orkd_ring_submit(c, weight_id, M, K, N, dtype, A);
    if (t < 0) return t;
    return orkd_ring_collect(c, t, C);
}
int orkd_run_i8_ring(orkd_conn *c, uint64_t weight_id, int M, int K, int N, const int8_t *A, int32_t *C){
    return orkd_ring_run(c, weight_id, M, K, N, ORKD_DT_I8, A, C);
}

void orkd_disconnect(orkd_conn *c){
    if (!c) return;
    if (c->ring){ atomic_store_explicit(&c->ring->stop, 1u, memory_order_release); munmap(c->ring, ORKD_RING_BYTES); c->ring = NULL; }
    if (c->ring_fd > 0){ close(c->ring_fd); c->ring_fd = -1; }
    if (c->fd >= 0){ struct orkd_hdr h = { ORKD_BYE, 0, 0 }; (void)wn(c->fd, &h, sizeof h); close(c->fd); c->fd = -1; }
    for (int i = 0; i < g_nlive; i++) if (g_live[i] == c){ g_live[i] = g_live[--g_nlive]; break; }
    free(c);
}

uint32_t orkd_client_id(orkd_conn *c){ return c ? c->client_id : 0; }
uint32_t orkd_soc_cores(orkd_conn *c){ return c ? c->soc_cores : 0; }

/* ---- submit RPC (#2b-1) ---- */
