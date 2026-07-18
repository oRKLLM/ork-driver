/* orkd_client.c — connect/auto-spawn/ping/disconnect against the orkd daemon. See orkd_client.h. */
#include "orkd_client.h"
#include "orkd_proto.h"
#include "orkd_shm.h"

#include <errno.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define ORKD_BIN_ENV "ORKD_BIN"
#define ORKD_CONNECT_TRIES 200          /* * 15ms ~= 3s to let an auto-spawned daemon bind */
#define ORKD_CONNECT_BACKOFF_NS (15*1000*1000L)

struct orkd_conn { int fd; uint32_t client_id; uint32_t soc_cores; };

/* live connections, for the atexit clean-BYE (small fixed set; a process rarely holds many) */
static orkd_conn *g_live[16];
static int g_nlive = 0, g_atexit_armed = 0;

static int wn(int fd, const void *b, size_t n){
    const char *p = b; size_t k = 0;
    while (k < n){ ssize_t r = write(fd, p+k, n-k); if (r < 0){ if (errno==EINTR) continue; return -1; } k += (size_t)r; }
    return 0;
}
static int rn(int fd, void *b, size_t n){
    char *p = b; size_t k = 0;
    while (k < n){ ssize_t r = read(fd, p+k, n-k); if (r == 0) return 0; if (r < 0){ if (errno==EINTR) continue; return -1; } k += (size_t)r; }
    return 1;
}

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

void orkd_disconnect(orkd_conn *c){
    if (!c) return;
    if (c->fd >= 0){ struct orkd_hdr h = { ORKD_BYE, 0, 0 }; (void)wn(c->fd, &h, sizeof h); close(c->fd); c->fd = -1; }
    for (int i = 0; i < g_nlive; i++) if (g_live[i] == c){ g_live[i] = g_live[--g_nlive]; break; }
    free(c);
}

uint32_t orkd_client_id(orkd_conn *c){ return c ? c->client_id : 0; }
uint32_t orkd_soc_cores(orkd_conn *c){ return c ? c->soc_cores : 0; }

/* ---- submit RPC (#2b-1) ---- */
static void cdrain(int fd, size_t n){ char b[4096]; while (n){ size_t k = n > sizeof b ? sizeof b : n; if (rn(fd, b, k) <= 0) return; n -= k; } }

uint64_t orkd_pack_i8(orkd_conn *c, int K, int N, const int8_t *B){
    if (!c || c->fd < 0 || K <= 0 || N <= 0 || !B) return 0;
    uint32_t bytes = (uint32_t)((size_t)K * N);
    struct orkd_pack pk; memset(&pk, 0, sizeof pk); pk.K = (uint32_t)K; pk.N = (uint32_t)N; pk.dtype = ORKD_DT_I8; pk.bytes = bytes;
    struct orkd_hdr h = { ORKD_PACK, (uint32_t)(sizeof pk + bytes), 3 };
    if (wn(c->fd, &h, sizeof h) || wn(c->fd, &pk, sizeof pk) || wn(c->fd, B, bytes)) return 0;
    struct orkd_hdr rh;
    if (rn(c->fd, &rh, sizeof rh) <= 0) return 0;
    if (rh.type != ORKD_PACK_OK){ if (rh.len) cdrain(c->fd, rh.len); return 0; }
    struct orkd_handle hh;
    if (rn(c->fd, &hh, sizeof hh) <= 0) return 0;
    if (rh.len > sizeof hh) cdrain(c->fd, rh.len - sizeof hh);
    return hh.rc == 0 ? hh.id : 0;
}

int orkd_run_i8(orkd_conn *c, uint64_t weight_id, int M, int K, int N, const int8_t *A, int32_t *C){
    if (!c || c->fd < 0 || M <= 0 || K <= 0 || N <= 0 || !A || !C) return -1;
    uint32_t abytes = (uint32_t)((size_t)M * K);
    struct orkd_run rq; memset(&rq, 0, sizeof rq); rq.weight_id = weight_id; rq.M = (uint32_t)M; rq.abytes = abytes;
    struct orkd_hdr h = { ORKD_RUN, (uint32_t)(sizeof rq + abytes), 4 };
    if (wn(c->fd, &h, sizeof h) || wn(c->fd, &rq, sizeof rq) || wn(c->fd, A, abytes)) return -1;
    struct orkd_hdr rh;
    if (rn(c->fd, &rh, sizeof rh) <= 0) return -1;
    if (rh.type != ORKD_RUN_OK){ if (rh.len) cdrain(c->fd, rh.len); return -1; }
    struct orkd_handle hh;
    if (rn(c->fd, &hh, sizeof hh) <= 0) return -1;
    size_t consumed = sizeof hh, cbytes = (size_t)M * N * 4;
    if (hh.rc == 0){
        if (rh.len < consumed + cbytes){ if (rh.len > consumed) cdrain(c->fd, rh.len - consumed); return -1; }
        if (rn(c->fd, C, cbytes) <= 0) return -1;
        consumed += cbytes;
    }
    if (rh.len > consumed) cdrain(c->fd, rh.len - consumed);
    return hh.rc;
}

int orkd_free_weight(orkd_conn *c, uint64_t weight_id){
    if (!c || c->fd < 0) return -1;
    struct orkd_handle req; memset(&req, 0, sizeof req); req.id = weight_id;
    struct orkd_hdr h = { ORKD_FREE, (uint32_t)sizeof req, 5 };
    if (wn(c->fd, &h, sizeof h) || wn(c->fd, &req, sizeof req)) return -1;
    struct orkd_hdr rh;
    if (rn(c->fd, &rh, sizeof rh) <= 0) return -1;
    if (rh.type != ORKD_PACK_OK){ if (rh.len) cdrain(c->fd, rh.len); return -1; }
    struct orkd_handle hh;
    if (rn(c->fd, &hh, sizeof hh) <= 0) return -1;
    if (rh.len > sizeof hh) cdrain(c->fd, rh.len - sizeof hh);
    return hh.rc;
}

/* #2b-2 step 1: allocate a dma-heap buffer, fill+hash it, pass the fd to orkd (SCM_RIGHTS), and confirm orkd
 * reads the same bytes through the shared mapping. 0 = shared (fnv matched); <0 = failure (-2 = no dma-heap). */
int orkd_dmabuf_probe(orkd_conn *c, size_t size){
#ifndef __linux__
    (void)c; (void)size; return -2;                 /* dma-heap is Linux-only */
#else
    if (!c || c->fd < 0 || !size) return -1;
    int dfd = orkd_dmaheap_alloc(size);
    if (dfd < 0) return -2;
    void *m = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, dfd, 0);
    if (m == MAP_FAILED){ close(dfd); return -3; }
    uint8_t *b = (uint8_t *)m;
    for (size_t i = 0; i < size; i++) b[i] = (uint8_t)(i * 131u + 7u);
    struct orkd_dmabuf db; memset(&db, 0, sizeof db); db.size = size; db.checksum = orkd_fnv(m, size);
    struct orkd_hdr h = { ORKD_DMABUF_PROBE, (uint32_t)sizeof db, 6 };
    int rc = -4;
    if (orkd_send_hdr_fd(c->fd, &h, dfd) == 0 && wn(c->fd, &db, sizeof db) == 0){
        struct orkd_hdr rh;
        if (rn(c->fd, &rh, sizeof rh) > 0 && rh.type == ORKD_DMABUF_OK){
            struct orkd_dmabuf out;
            if (rn(c->fd, &out, sizeof out) > 0){ if (rh.len > sizeof out) cdrain(c->fd, rh.len - sizeof out); rc = (out.rc == 0) ? (out.prime_ok ? 0 : 1) : -5; }
        }
    }
    munmap(m, size); close(dfd);
    return rc;
#endif
}
