/* orkd_client.h — ork-driver's client side of the orkd daemon connection.
 *
 * The library uses this to become a CONSUMER of orkd: orkd_connect() connects to a running daemon or
 * AUTO-SPAWNS one if none exists (flock in orkd arbitrates the spawn race), then registers as a subscriber
 * (HELLO/WELCOME). orkd_disconnect() deregisters gracefully (BYE); if the process instead dies abruptly, orkd
 * sees the socket EOF and deregisters anyway (the universal safety net) — so no signal handler is hijacked.
 * A registered atexit hook sends BYE on normal exit(). See src/orkd_proto.h and memory orkd-daemon-direction.
 *
 * Increment #1: lifecycle only (connect/spawn/ping/disconnect). The submit RPC (pack/run over the connection)
 * is the next increment. */
#ifndef ORKD_CLIENT_H
#define ORKD_CLIENT_H

#include <stddef.h>
#include <stdint.h>

typedef struct orkd_conn orkd_conn;

/* Connect to orkd, auto-spawning it if none is running (ORKD_BIN overrides the binary path; else PATH "orkd").
 * Sends HELLO and awaits WELCOME. Returns a connection or NULL on failure (spawn/connect/handshake). */
orkd_conn *orkd_connect(void);

/* Liveness round-trip (PING/PONG). 0 = ok, <0 = error/dead. */
int orkd_ping(orkd_conn *c);

/* Graceful deregister (BYE) + close + free. Safe on NULL. */
void orkd_disconnect(orkd_conn *c);

/* Daemon-assigned subscriber id (diagnostics); 0 if NULL. */
uint32_t orkd_client_id(orkd_conn *c);

/* NPU core count reported by the daemon in WELCOME (RK3588 = 3; 0 if the daemon's NPU init failed or NULL). */
uint32_t orkd_soc_cores(orkd_conn *c);

/* ---- submit RPC (#2b-1: int8, socket-transfer; dma-buf zero-copy is #2b-2) --------------------------------
 * Synchronous request/reply over the connection; the daemon serializes all clients' runs onto the one NPU. */

/* Pack an int8 weight B[K,N] resident in the daemon. Returns a weight id (>0) or 0 on failure. */
uint64_t orkd_pack_i8(orkd_conn *c, int K, int N, const int8_t *B);

/* Run int8 A[M,K] x weight -> C[M,N] (int32), computed on the daemon's NPU. 0 = ok, <0 = error. */
int orkd_run_i8(orkd_conn *c, uint64_t weight_id, int M, int K, int N, const int8_t *A, int32_t *C);

/* fp16 counterparts (Path B increment 2): B/A are fp16 (passed as void*), C is fp32. Pack returns id>0 / 0. */
uint64_t orkd_pack_f16(orkd_conn *c, int K, int N, const void *B);
int orkd_run_f16(orkd_conn *c, uint64_t weight_id, int M, int K, int N, const void *A, float *C);

/* int4 counterparts (Path B increment 3): int4 in int8 [-8,7], C is int32 (wire = int8). */
uint64_t orkd_pack_i4(orkd_conn *c, int K, int N, const int8_t *B);
int orkd_run_i4(orkd_conn *c, uint64_t weight_id, int M, int K, int N, const int8_t *A, int32_t *C);

/* Like orkd_run_i8 but A is passed BY REFERENCE: placed in a shared dma-buf and read zero-copy by the NPU
 * (no A byte-transfer over the socket). C still returns over the socket. 0 = ok, <0 = error (-2 = no dma-heap). */
int orkd_run_i8_zc(orkd_conn *c, uint64_t weight_id, int M, int K, int N, const int8_t *A, int32_t *C);

/* Full zero-copy: A read AND C written by reference (both shared dma-bufs; no A/C bytes over the socket).
 * Single-core on the daemon (output zero-copy is not multi-core-safe). 0 = ok, <0 = error (-2 = no dma-heap). */
int orkd_run_i8_zc2(orkd_conn *c, uint64_t weight_id, int M, int K, int N, const int8_t *A, int32_t *C);

/* Free a resident weight (also freed automatically when the client disconnects). 0 = ok, <0 = error. */
int orkd_free_weight(orkd_conn *c, uint64_t weight_id);

/* #2b-2 step 1 probe: allocate a dma-heap buffer, share its fd to orkd (SCM_RIGHTS), confirm orkd sees the
 * same bytes through the shared mapping. 0 = shared OK; <0 = failure (-2 = no dma-heap on this host). */
int orkd_dmabuf_probe(orkd_conn *c, size_t size);

#endif /* ORKD_CLIENT_H */
