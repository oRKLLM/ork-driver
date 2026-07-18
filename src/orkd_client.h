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

#endif /* ORKD_CLIENT_H */
