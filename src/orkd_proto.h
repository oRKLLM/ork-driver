/* orkd_proto.h — wire protocol between the ork-driver client library and the orkd daemon.
 *
 * orkd is a single standalone process that OWNS the RK3588 NPU (opens /dev/dri/cardN once, holds the IOMMU
 * domains + resident weights + the NONBLOCK doorbell submit loop) and serializes submits from MULTIPLE client
 * processes onto the single-stream NPU. The ork-driver library becomes a thin CLIENT: it auto-spawns orkd if
 * none is running, connects as a consumer, and deregisters on shutdown/SIGTERM. orkd idle-reaps itself once it
 * has no subscribers for a threshold. See memory orkd-daemon-direction / STREAMLINE_ARCH_WIP.md.
 *
 * Transport: a Unix SOCK_STREAM socket (control plane — these messages). The DATA plane (weights/activations)
 * is shared out-of-band by passing dma-buf fds over SCM_RIGHTS ancillary data, which orkd imports into its
 * IOMMU domain (PRIME_FD_TO_HANDLE). Every message is a fixed orkd_hdr followed by `len` payload bytes; a
 * request carries a `tag` the matching reply echoes (so a client can pipeline).
 *
 * C11, libc only. This header is shared by src/orkd.c (daemon) and the client shim in the library. */
#ifndef ORKD_PROTO_H
#define ORKD_PROTO_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define ORKD_PROTO_VERSION 1u

/* Socket path: $XDG_RUNTIME_DIR/orkd.sock if set, else /tmp/orkd-<uid>.sock. orkd_sock_path() (below) resolves
 * it identically for daemon (bind) and client (connect) so both agree without configuration. */
#define ORKD_SOCK_ENV   "ORKD_SOCK"          /* explicit override (tests, alt instances) */
#define ORKD_SOCK_NAME  "orkd.sock"
#define ORKD_PIDFILE_NAME "orkd.pid"         /* flock target that arbitrates the auto-spawn race */

/* Idle-reap: orkd exits after this many ms with zero subscribers (ORKD_IDLE_MS env overrides). Must exceed the
 * inter-process gap of a serial client burst (e.g. make test's ~28 back-to-back binaries) so the daemon
 * persists across the whole run, then reaps. */
#define ORKD_IDLE_MS_DEFAULT 5000

enum orkd_msg_type {
    ORKD_HELLO    = 1,   /* client->daemon: {orkd_hello} register a subscriber (ref++)                         */
    ORKD_WELCOME  = 2,   /* daemon->client: {orkd_welcome} ack + assigned client id + daemon caps              */
    ORKD_BYE      = 3,   /* client->daemon: graceful deregister (ref--); daemon also treats socket EOF as BYE  */
    ORKD_PING     = 4,   /* client->daemon: liveness / keepalive                                               */
    ORKD_PONG     = 5,   /* daemon->client: PING reply                                                         */
    /* ---- data plane (stubbed in the first skeleton; return ORKD_ERROR/ORKD_ENOSYS until implemented) ---- */
    ORKD_PACK     = 16,  /* client->daemon: {orkd_pack} + weight dma-buf fd (SCM_RIGHTS) -> resident weight    */
    ORKD_PACK_OK  = 17,  /* daemon->client: {orkd_handle} weight id                                            */
    ORKD_RUN      = 18,  /* client->daemon: {orkd_run} weight id + M + A/C dma-buf fds -> queued submit        */
    ORKD_RUN_OK   = 19,  /* daemon->client: {orkd_handle} result ready (C written to the shared buf)           */
    ORKD_FREE     = 20,  /* client->daemon: {orkd_handle} free a resident weight                               */
    /* ---- #2b-2 dma-buf sharing (SCM_RIGHTS fd passing) ---- */
    ORKD_DMABUF_PROBE = 32, /* client->daemon: {orkd_dmabuf} + a dma-buf fd (SCM_RIGHTS): does orkd see the same bytes? */
    ORKD_DMABUF_OK    = 33, /* daemon->client: {orkd_dmabuf} echo — rc=0 + checksum orkd computed from the shared buffer */
    ORKD_ERROR    = 255, /* daemon->client: {orkd_error} code + message                                        */
};

/* dtype for orkd_pack.dtype (wire-stable; decoupled from the library's internal enum). #2b-1 = int8 only. */
enum orkd_dtype {
    ORKD_DT_I8  = 1,   /* int8 A·B -> int32 C (ork_mm_pack_i8 / ork_mm_run_i8) */
    ORKD_DT_F16 = 2,   /* fp16 -> fp32 (not yet wired) */
};

/* error codes carried in orkd_error.code */
enum orkd_err {
    ORKD_EOK     = 0,
    ORKD_ENOSYS  = 1,    /* not implemented yet (skeleton)          */
    ORKD_EPROTO  = 2,    /* malformed message / version mismatch    */
    ORKD_EBADH   = 3,    /* unknown weight/handle id                */
    ORKD_ENPU    = 4,    /* NPU submit failed                       */
    ORKD_EOOM    = 5,    /* alloc / IOVA exhausted                  */
};

#pragma pack(push, 1)
struct orkd_hdr {
    uint32_t type;       /* enum orkd_msg_type      */
    uint32_t len;        /* payload bytes following */
    uint64_t tag;        /* request id; reply echoes it (0 for unsolicited) */
};
struct orkd_hello {
    uint32_t proto;      /* ORKD_PROTO_VERSION */
    uint32_t pid;        /* client pid (diagnostics) */
};
struct orkd_welcome {
    uint32_t proto;
    uint32_t client_id;  /* daemon-assigned */
    uint32_t soc_cores;  /* daemon's NPU core count (0 until NPU wired) */
    uint32_t flags;      /* reserved */
};
struct orkd_pack {
    uint32_t K, N;       /* weight dims */
    uint32_t dtype;      /* enum-matching ork_w dtype */
    uint32_t bytes;      /* weight blob size in the passed dma-buf */
};
struct orkd_run {
    uint64_t weight_id;
    uint32_t M;
    uint32_t flags;      /* reserved (chain kind, etc.) */
    uint32_t abytes;     /* size of the A payload that follows (= M*K int8); lets orkd drain even on a bad id */
    uint32_t pad;
};
struct orkd_handle {
    uint64_t id;         /* weight id / result marker */
    int32_t  rc;         /* 0 ok, <0 error */
    uint32_t pad;
};
struct orkd_error {
    uint32_t code;       /* enum orkd_err */
    uint32_t pad;
    char     msg[64];    /* NUL-terminated */
};
struct orkd_dmabuf {     /* #2b-2 probe: the fd rides as SCM_RIGHTS ancillary data, out of band */
    uint64_t size;       /* bytes to check in the shared buffer */
    uint64_t checksum;   /* PROBE: client's fnv of the buffer; OK: what orkd computed (must match) */
    int32_t  rc;         /* OK: 0 = imported+mapped+matched, <0 = failure */
    uint32_t prime_ok;   /* OK: 1 if DRM PRIME_FD_TO_HANDLE also succeeded (NPU-importable) */
};
#pragma pack(pop)

/* Resolve the daemon's runtime dir: $XDG_RUNTIME_DIR if set, else /tmp. Both daemon and client call these so
 * they agree without configuration. ORKD_SOCK overrides the socket path outright (alt instances / tests). */
static inline const char *orkd_runtime_dir(void){
    const char *rt = getenv("XDG_RUNTIME_DIR");
    return (rt && *rt) ? rt : "/tmp";
}
static inline void orkd_sock_path(char *buf, size_t n){
    const char *env = getenv(ORKD_SOCK_ENV);
    if (env && *env) { snprintf(buf, n, "%s", env); return; }
    snprintf(buf, n, "%s/orkd-%u.sock", orkd_runtime_dir(), (unsigned)getuid());
}
static inline void orkd_pidfile_path(char *buf, size_t n){
    snprintf(buf, n, "%s/orkd-%u.pid", orkd_runtime_dir(), (unsigned)getuid());
}
static inline unsigned orkd_idle_ms(void){
    const char *e = getenv("ORKD_IDLE_MS");
    if (e && *e) { long v = atol(e); if (v > 0) return (unsigned)v; }
    return ORKD_IDLE_MS_DEFAULT;
}

#endif /* ORKD_PROTO_H */
