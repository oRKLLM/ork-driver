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

/* fp16 pack/run (Path B increment 2). Same PACK/RUN wire as int8 — dtype=ORKD_DT_F16 in the pack tells orkd to
 * ork_mm_pack (fp16 weight) and run ork_mm_run (fp16 A -> fp32 C). B is K*N fp16; A is M*K fp16; C is M*N fp32.
 * void* operands keep this shim free of the ork_npu.h fp16 typedef (the library casts ork_f16* -> void*). */
uint64_t orkd_pack_f16(orkd_conn *c, int K, int N, const void *B){
    if (!c || c->fd < 0 || K <= 0 || N <= 0 || !B) return 0;
    uint32_t bytes = (uint32_t)((size_t)K * N * 2);
    struct orkd_pack pk; memset(&pk, 0, sizeof pk); pk.K = (uint32_t)K; pk.N = (uint32_t)N; pk.dtype = ORKD_DT_F16; pk.bytes = bytes;
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

int orkd_run_f16(orkd_conn *c, uint64_t weight_id, int M, int K, int N, const void *A, float *C){
    if (!c || c->fd < 0 || M <= 0 || K <= 0 || N <= 0 || !A || !C) return -1;
    uint32_t abytes = (uint32_t)((size_t)M * K * 2);
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

/* int4 pack/run (Path B increment 3). int4 values live byte-per-value in int8 ([-8,7]) so the wire is IDENTICAL
 * to int8 (A=M*K bytes, C=M*N*4); only dtype=ORKD_DT_I4 differs, telling orkd to ork_mm_pack_i4 / ork_mm_run_i4. */
uint64_t orkd_pack_i4(orkd_conn *c, int K, int N, const int8_t *B){
    if (!c || c->fd < 0 || K <= 0 || N <= 0 || !B) return 0;
    uint32_t bytes = (uint32_t)((size_t)K * N);
    struct orkd_pack pk; memset(&pk, 0, sizeof pk); pk.K = (uint32_t)K; pk.N = (uint32_t)N; pk.dtype = ORKD_DT_I4; pk.bytes = bytes;
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

int orkd_run_i4(orkd_conn *c, uint64_t weight_id, int M, int K, int N, const int8_t *A, int32_t *C){
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

/* SDP ops (Path B increment 4). One generic wire call (ORKD_SDP) + typed wrappers so the library routes
 * ork_npu_silu_i8 / gelu_i8 / ewmul_i8 / ewmul_f16 / add_i8 / add_f16 without exposing the op enum. */
static int orkd_sdp_call(orkd_conn *c, uint32_t op, int M, int N, int nin, int in_esz, int out_esz,
                         int mult, int shift, double in_scale, double out_scale, double a_scale, double b_scale,
                         const void *a, const void *b, void *out){
    if (!c || c->fd < 0 || M <= 0 || N <= 0 || !a || !out || (nin == 2 && !b)) return -1;
    size_t half = (size_t)M * N * in_esz, outb = (size_t)M * N * out_esz;
    struct orkd_sdp sp; memset(&sp, 0, sizeof sp);
    sp.op = op; sp.M = (uint32_t)M; sp.N = (uint32_t)N; sp.nin = (uint32_t)nin;
    sp.in_esz = (uint32_t)in_esz; sp.out_esz = (uint32_t)out_esz;
    sp.mult = mult; sp.shift = shift; sp.in_scale = in_scale; sp.out_scale = out_scale; sp.a_scale = a_scale; sp.b_scale = b_scale;
    sp.inbytes = (uint32_t)(half * nin);
    struct orkd_hdr h = { ORKD_SDP, (uint32_t)(sizeof sp + sp.inbytes), 9 };
    if (wn(c->fd, &h, sizeof h) || wn(c->fd, &sp, sizeof sp) || wn(c->fd, a, half)) return -1;
    if (nin == 2 && wn(c->fd, b, half)) return -1;
    struct orkd_hdr rh;
    if (rn(c->fd, &rh, sizeof rh) <= 0) return -1;
    if (rh.type != ORKD_SDP_OK){ if (rh.len) cdrain(c->fd, rh.len); return -1; }
    struct orkd_handle hh;
    if (rn(c->fd, &hh, sizeof hh) <= 0) return -1;
    size_t consumed = sizeof hh;
    if (hh.rc == 0){
        if (rh.len < consumed + outb){ if (rh.len > consumed) cdrain(c->fd, rh.len - consumed); return -1; }
        if (rn(c->fd, out, outb) <= 0) return -1;
        consumed += outb;
    }
    if (rh.len > consumed) cdrain(c->fd, rh.len - consumed);
    return hh.rc;
}
int orkd_silu_i8 (orkd_conn *c, const int8_t *in, int M, int N, double is, double os, int8_t *out){
    return orkd_sdp_call(c, ORKD_SDP_SILU_I8, M, N, 1, 1, 1, 0, 0, is, os, 0, 0, in, NULL, out); }
int orkd_gelu_i8 (orkd_conn *c, const int8_t *in, int M, int N, double is, double os, int8_t *out){
    return orkd_sdp_call(c, ORKD_SDP_GELU_I8, M, N, 1, 1, 1, 0, 0, is, os, 0, 0, in, NULL, out); }
int orkd_ewmul_i8(orkd_conn *c, const int8_t *a, const int8_t *b, int M, int N, int mult, int shift, int8_t *out){
    return orkd_sdp_call(c, ORKD_SDP_EWMUL_I8, M, N, 2, 1, 1, mult, shift, 0, 0, 0, 0, a, b, out); }
int orkd_ewmul_f16(orkd_conn *c, const void *a, const void *b, int M, int N, void *out){
    return orkd_sdp_call(c, ORKD_SDP_EWMUL_F16, M, N, 2, 2, 2, 0, 0, 0, 0, 0, 0, a, b, out); }
int orkd_add_i8  (orkd_conn *c, const int8_t *a, const int8_t *b, int M, int N, double as, double bs, double os, int8_t *out){
    return orkd_sdp_call(c, ORKD_SDP_ADD_I8, M, N, 2, 1, 1, 0, 0, 0, os, as, bs, a, b, out); }
int orkd_add_f16 (orkd_conn *c, const void *a, const void *b, int M, int N, void *out){
    return orkd_sdp_call(c, ORKD_SDP_ADD_F16, M, N, 2, 2, 2, 0, 0, 0, 0, 0, 0, a, b, out); }

int orkd_run_chain_i8(orkd_conn *c, int S, const orkd_chain_task_c *t){
    if (!c || c->fd < 0 || S < 1 || S > ORKD_CHAIN_MAX || !t) return -1;
    uint64_t atot = 0, ctot = 0;
    for (int i = 0; i < S; i++){ if (t[i].M <= 0 || t[i].K <= 0 || t[i].N <= 0 || !t[i].A || !t[i].C) return -1;
        atot += (size_t)t[i].M * t[i].K; ctot += (size_t)t[i].M * t[i].N * 4; }
    if (atot > 0xffffffffu) return -1;
    struct orkd_chain_hdr ch; memset(&ch, 0, sizeof ch); ch.S = (uint32_t)S; ch.abytes_total = (uint32_t)atot;
    uint32_t plen = (uint32_t)(sizeof ch + (size_t)S * sizeof(struct orkd_chain_task) + atot);
    struct orkd_hdr h = { ORKD_CHAIN, plen, 10 };
    if (wn(c->fd, &h, sizeof h) || wn(c->fd, &ch, sizeof ch)) return -1;
    for (int i = 0; i < S; i++){ struct orkd_chain_task ct; memset(&ct, 0, sizeof ct);
        ct.weight_id = t[i].weight_id; ct.M = (uint32_t)t[i].M; ct.abytes = (uint32_t)((size_t)t[i].M * t[i].K);
        if (wn(c->fd, &ct, sizeof ct)) return -1; }
    for (int i = 0; i < S; i++) if (wn(c->fd, t[i].A, (size_t)t[i].M * t[i].K)) return -1;
    struct orkd_hdr rh;
    if (rn(c->fd, &rh, sizeof rh) <= 0) return -1;
    if (rh.type != ORKD_CHAIN_OK){ if (rh.len) cdrain(c->fd, rh.len); return -1; }
    struct orkd_handle hh;
    if (rn(c->fd, &hh, sizeof hh) <= 0) return -1;
    size_t consumed = sizeof hh, cbytes = (size_t)ctot;
    if (hh.rc == 0){
        if (rh.len < consumed + cbytes){ if (rh.len > consumed) cdrain(c->fd, rh.len - consumed); return -1; }
        for (int i = 0; i < S; i++) if (rn(c->fd, t[i].C, (size_t)t[i].M * t[i].N * 4) <= 0) return -1;
        consumed += cbytes;
    }
    if (rh.len > consumed) cdrain(c->fd, rh.len - consumed);
    return hh.rc;
}

int orkd_submit_seq(orkd_conn *c, int n, const orkd_seq_op_c *o){
    if (!c || c->fd < 0 || n < 1 || n > ORKD_SEQ_MAX || !o) return -1;
    uint64_t in_total = 0, c_total = 0;
    for (int i = 0; i < n; i++){ if (!o[i].A || !o[i].C || (o[i].bbytes && !o[i].B)) return -1;
        in_total += (uint64_t)o[i].abytes + o[i].bbytes; c_total += o[i].cbytes; }
    if (in_total > 0xffffffffu) return -1;
    struct orkd_seq_hdr sh; memset(&sh, 0, sizeof sh); sh.n = (uint32_t)n; sh.in_total = (uint32_t)in_total;
    uint32_t plen = (uint32_t)(sizeof sh + (size_t)n * sizeof(struct orkd_seq_op) + in_total);
    struct orkd_hdr h = { ORKD_SEQ, plen, 11 };
    if (wn(c->fd, &h, sizeof h) || wn(c->fd, &sh, sizeof sh)) return -1;
    for (int i = 0; i < n; i++){ struct orkd_seq_op w; memset(&w, 0, sizeof w);
        w.kind = o[i].kind; w.M = (uint32_t)o[i].M; w.N = (uint32_t)o[i].N; w.weight_id = o[i].weight_id;
        w.abytes = o[i].abytes; w.bbytes = o[i].bbytes; w.cbytes = o[i].cbytes;
        w.mult = o[i].mult; w.shift = o[i].shift; w.group = o[i].group; w.in_scale = o[i].in_scale; w.out_scale = o[i].out_scale; w.b_scale = o[i].b_scale;
        if (wn(c->fd, &w, sizeof w)) return -1; }
    for (int i = 0; i < n; i++){ if (wn(c->fd, o[i].A, o[i].abytes)) return -1; if (o[i].bbytes && wn(c->fd, o[i].B, o[i].bbytes)) return -1; }
    struct orkd_hdr rh;
    if (rn(c->fd, &rh, sizeof rh) <= 0) return -1;
    if (rh.type != ORKD_SEQ_OK){ if (rh.len) cdrain(c->fd, rh.len); return -1; }
    struct orkd_handle hh;
    if (rn(c->fd, &hh, sizeof hh) <= 0) return -1;
    size_t consumed = sizeof hh, cb = (size_t)c_total;
    if (hh.rc == 0){
        if (rh.len < consumed + cb){ if (rh.len > consumed) cdrain(c->fd, rh.len - consumed); return -1; }
        for (int i = 0; i < n; i++) if (o[i].cbytes && rn(c->fd, o[i].C, o[i].cbytes) <= 0) return -1;
        consumed += cb;
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

/* #2b-2 step 3: run int8 with A passed BY REFERENCE (zero-copy) — A is placed in a shared dma-buf, its fd
 * handed to orkd (SCM_RIGHTS), and the NPU reads it in place; C comes back over the socket. (A plain-buffer
 * A is memcpy'd into the dma-buf here for convenience; a dma-buf-native caller fills the shared buffer
 * directly for true no-copy.) 0 = ok, <0 = error (-2 = no dma-heap). */
int orkd_run_i8_zc(orkd_conn *c, uint64_t weight_id, int M, int K, int N, const int8_t *A, int32_t *C){
#ifndef __linux__
    (void)c;(void)weight_id;(void)M;(void)K;(void)N;(void)A;(void)C; return -2;
#else
    if (!c || c->fd < 0 || M <= 0 || K <= 0 || N <= 0 || !A || !C) return -1;
    size_t abytes = (size_t)M * K;
    int afd = orkd_dmaheap_alloc(abytes);
    if (afd < 0) return -2;
    void *am = mmap(NULL, abytes, PROT_READ | PROT_WRITE, MAP_SHARED, afd, 0);
    if (am == MAP_FAILED){ close(afd); return -3; }
    memcpy(am, A, abytes);
    orkd_dmabuf_clean(afd);                          /* flush A to device before the NPU reads it */
    struct orkd_run rq; memset(&rq, 0, sizeof rq); rq.weight_id = weight_id; rq.M = (uint32_t)M; rq.abytes = 0;
    struct orkd_hdr h = { ORKD_RUN_ZC, (uint32_t)sizeof rq, 7 };
    if (orkd_send_hdr_fd(c->fd, &h, afd) || wn(c->fd, &rq, sizeof rq)){ munmap(am, abytes); close(afd); return -1; }
    struct orkd_hdr rh;
    if (rn(c->fd, &rh, sizeof rh) <= 0){ munmap(am, abytes); close(afd); return -1; }
    if (rh.type != ORKD_RUN_OK){ if (rh.len) cdrain(c->fd, rh.len); munmap(am, abytes); close(afd); return -1; }
    struct orkd_handle hh;
    if (rn(c->fd, &hh, sizeof hh) <= 0){ munmap(am, abytes); close(afd); return -1; }
    size_t consumed = sizeof hh, cbytes = (size_t)M * N * 4;
    if (hh.rc == 0){
        if (rh.len < consumed + cbytes){ if (rh.len > consumed) cdrain(c->fd, rh.len - consumed); munmap(am, abytes); close(afd); return -1; }
        if (rn(c->fd, C, cbytes) <= 0){ munmap(am, abytes); close(afd); return -1; }
        consumed += cbytes;
    }
    if (rh.len > consumed) cdrain(c->fd, rh.len - consumed);
    munmap(am, abytes); close(afd);
    return hh.rc;
#endif
}

/* #2b-2 step 3b: full zero-copy — A read AND C written by reference (both shared dma-bufs, fds via SCM_RIGHTS).
 * No A/C bytes cross the socket. (Plain-buffer convenience: A is memcpy'd in, C memcpy'd out; a dma-buf-native
 * caller works the shared buffers directly for true no-copy.) 0 = ok, <0 = error (-2 = no dma-heap). */
int orkd_run_i8_zc2(orkd_conn *c, uint64_t weight_id, int M, int K, int N, const int8_t *A, int32_t *C){
#ifndef __linux__
    (void)c;(void)weight_id;(void)M;(void)K;(void)N;(void)A;(void)C; return -2;
#else
    if (!c || c->fd < 0 || M <= 0 || K <= 0 || N <= 0 || !A || !C) return -1;
    size_t abytes = (size_t)M * K, cbytes = (size_t)M * N * 4;
    int afd = orkd_dmaheap_alloc(abytes), cfd = orkd_dmaheap_alloc(cbytes);
    if (afd < 0 || cfd < 0){ if (afd >= 0) close(afd); if (cfd >= 0) close(cfd); return -2; }
    void *am = mmap(NULL, abytes, PROT_READ | PROT_WRITE, MAP_SHARED, afd, 0);
    void *cm = mmap(NULL, cbytes, PROT_READ | PROT_WRITE, MAP_SHARED, cfd, 0);
    if (am == MAP_FAILED || cm == MAP_FAILED){ if (am != MAP_FAILED) munmap(am, abytes); if (cm != MAP_FAILED) munmap(cm, cbytes); close(afd); close(cfd); return -3; }
    memcpy(am, A, abytes); orkd_dmabuf_clean(afd);
    struct orkd_run rq; memset(&rq, 0, sizeof rq); rq.weight_id = weight_id; rq.M = (uint32_t)M; rq.abytes = 0;
    struct orkd_hdr h = { ORKD_RUN_ZC2, (uint32_t)sizeof rq, 8 };
    int fds[2] = { afd, cfd };
    int rc = -1;
    if (!orkd_send_hdr_fds(c->fd, &h, fds, 2) && !wn(c->fd, &rq, sizeof rq)){
        struct orkd_hdr rh;
        if (rn(c->fd, &rh, sizeof rh) > 0){
            if (rh.type == ORKD_RUN_OK){
                struct orkd_handle hh;
                if (rn(c->fd, &hh, sizeof hh) > 0){
                    if (rh.len > sizeof hh) cdrain(c->fd, rh.len - sizeof hh);
                    rc = hh.rc;
                    if (rc == 0){ orkd_dmabuf_invalidate(cfd); memcpy(C, cm, cbytes); }   /* invalidate before reading C the NPU wrote */
                }
            } else if (rh.len) cdrain(c->fd, rh.len);
        }
    }
    munmap(am, abytes); munmap(cm, cbytes); close(afd); close(cfd);
    return rc;
#endif
}
