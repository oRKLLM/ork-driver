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

#include <errno.h>
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

struct client { int fd; int hello; uint32_t id; };

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

    /* TODO(next increment): ork_npu *npu = ork_npu_init(); own the NPU here; report soc cores in WELCOME. */
    unsigned idle_ms = orkd_idle_ms();
    fprintf(stderr, "[orkd] up pid=%d sock=%s idle=%ums (lifecycle skeleton; NPU stubbed)\n",
            (int)getpid(), sockpath, idle_ms);

    struct client cl[ORKD_MAX_CLIENTS]; int nc = 0;
    uint32_t next_id = 1, refs = 0;
    long idle_since = now_ms();                /* refs==0 since this time */

    while (!g_stop){
        struct pollfd pfd[ORKD_MAX_CLIENTS + 1];
        pfd[0].fd = lfd; pfd[0].events = POLLIN;
        for (int i = 0; i < nc; i++){ pfd[i+1].fd = cl[i].fd; pfd[i+1].events = POLLIN; }

        int timeout = -1;                      /* block indefinitely while we have subscribers */
        if (refs == 0){
            long left = (long)idle_ms - (now_ms() - idle_since);
            timeout = left > 0 ? (int)left : 0;
        }
        int pr = poll(pfd, (nfds_t)(nc+1), timeout);
        if (pr < 0){ if (errno == EINTR) continue; perror("orkd: poll"); break; }
        if (pr == 0){                          /* timeout: idle-reap if still empty */
            if (refs == 0 && now_ms() - idle_since >= (long)idle_ms){
                fprintf(stderr, "[orkd] idle %ums, no subscribers -> reap\n", idle_ms);
                break;
            }
            continue;
        }

        if (pfd[0].revents & POLLIN){          /* new connection */
            int cfd = accept(lfd, NULL, NULL);
            if (cfd >= 0){
                if (nc < ORKD_MAX_CLIENTS){ cl[nc].fd = cfd; cl[nc].hello = 0; cl[nc].id = 0; nc++; }
                else { send_error(cfd, 0, ORKD_EOOM, "too many clients"); close(cfd); }
            }
        }
        for (int i = 0; i < nc; i++){
            if (!(pfd[i+1].revents & (POLLIN|POLLHUP|POLLERR))) continue;
            int drop = 0;
            struct orkd_hdr h;
            int rr = readn(cl[i].fd, &h, sizeof h);
            if (rr <= 0){ drop = 1; }          /* EOF (incl. abrupt client death) or error */
            else {
                char pay[512]; uint32_t plen = h.len > sizeof pay ? (uint32_t)sizeof pay : h.len;
                if (h.len && readn(cl[i].fd, pay, plen) <= 0){ drop = 1; }
                else switch (h.type){
                    case ORKD_HELLO: {
                        struct orkd_welcome w; memset(&w, 0, sizeof w);
                        w.proto = ORKD_PROTO_VERSION; w.client_id = cl[i].id = next_id++;
                        w.soc_cores = 0;           /* TODO: npu->soc->cores once wired */
                        if (!cl[i].hello){ cl[i].hello = 1; refs++; }
                        send_msg(cl[i].fd, ORKD_WELCOME, h.tag, &w, sizeof w);
                        break;
                    }
                    case ORKD_PING: send_msg(cl[i].fd, ORKD_PONG, h.tag, NULL, 0); break;
                    case ORKD_BYE:  drop = 1; break;
                    case ORKD_PACK: case ORKD_RUN: case ORKD_FREE:
                        send_error(cl[i].fd, h.tag, ORKD_ENOSYS, "NPU path not wired yet");
                        break;
                    default:
                        send_error(cl[i].fd, h.tag, ORKD_EPROTO, "unknown message");
                        break;
                }
            }
            if (drop){
                if (cl[i].hello && refs) refs--;
                close(cl[i].fd);
                cl[i] = cl[nc-1]; nc--; i--;   /* compact */
                if (refs == 0) idle_since = now_ms();
            }
        }
    }

    /* TODO(next increment): ork_npu_free(npu) — release NPU/IOMMU cleanly. */
    close(lfd); unlink(sockpath);
    flock(pidfd, LOCK_UN); close(pidfd); unlink(pidpath);
    fprintf(stderr, "[orkd] down\n");
    return 0;
}
