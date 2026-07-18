/* orkd_client.c — connect/auto-spawn/ping/disconnect against the orkd daemon. See orkd_client.h. */
#include "orkd_client.h"
#include "orkd_proto.h"

#include <errno.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define ORKD_BIN_ENV "ORKD_BIN"
#define ORKD_CONNECT_TRIES 200          /* * 15ms ~= 3s to let an auto-spawned daemon bind */
#define ORKD_CONNECT_BACKOFF_NS (15*1000*1000L)

struct orkd_conn { int fd; uint32_t client_id; };

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
    c->fd = fd; c->client_id = w.client_id;
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
