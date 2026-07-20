/* orkd_ring.h — A-ring: a shared-memory request/response ring for LOW-LATENCY decode submits.
 *
 * A socket round-trip per op (write hdr+req+A, blocking read hdr+rc+C) adds syscall + context-switch latency on
 * top of the ~167us NPU submit floor — fine for prefill (big, few submits), but decode is thousands of tiny
 * M=1 submits where that transport cost is a visible fraction. This ring removes the per-op syscall: the client
 * writes a request into a shared slot and busy-polls the response; the daemon busy-polls the ring (between its
 * socket polls) and writes the response in place. Establishment is still over the control socket (ORKD_RING_SETUP
 * + an SCM_RIGHTS fd for the shared region); only the hot submit path moves to the ring.
 *
 * TRANSPORT ONLY — the NPU path is unchanged (the daemon still runs the same ork_mm_run_*), so this carries no
 * new wedge risk; the risk surface is IPC correctness (SPSC ring sync + teardown).
 *
 * SPSC: the client is the sole producer, the daemon the sole consumer. Per-slot `state` is the handshake, with
 * acquire/release ordering so the data writes preceding a release-store are visible after the matching
 * acquire-load in the other process (the region is MAP_SHARED cacheable memory; ARM caches are coherent across
 * cores, so plain C11 atomics on it work cross-process). A is copied into the slot on request; C is copied back
 * into the same slot area on response (bounded by ORKD_RING_SLOT_DATA — larger ops fall back to the socket). */
#ifndef ORKD_RING_H
#define ORKD_RING_H

#include <stdint.h>
#include <stdatomic.h>

#define ORKD_RING_MAGIC     0x4f524b52u          /* "ORKR" */
#define ORKD_RING_SLOTS     8                     /* ring depth (SPSC; a synchronous client uses one at a time) */
#define ORKD_RING_SLOT_DATA (64u * 1024u)         /* per-slot payload: A on request, C on response (bounded) */

enum { ORKD_SLOT_EMPTY = 0, ORKD_SLOT_REQ = 1, ORKD_SLOT_RESP = 2 };

/* dtype codes reuse the wire values from orkd_proto.h (ORKD_DT_I8/F16/I4). */
struct orkd_ring_slot {
    _Atomic uint32_t state;                        /* EMPTY -> (client) REQ -> (daemon) RESP -> (client) EMPTY */
    uint64_t weight_id;
    uint32_t M, K, N, dtype;
    uint32_t abytes, cbytes;
    int32_t  rc;
    uint32_t _pad;
    uint8_t  data[ORKD_RING_SLOT_DATA];            /* A (abytes) on request; C (cbytes) on response */
};

struct orkd_ring {
    uint32_t magic, nslots, slot_data, _pad0;
    _Atomic uint32_t stop;                         /* set on teardown; both sides bail out of their spins */
    uint32_t _pad1[11];                            /* keep slots off the header cache line */
    struct orkd_ring_slot slot[ORKD_RING_SLOTS];
};

#define ORKD_RING_BYTES (sizeof(struct orkd_ring))

#endif /* ORKD_RING_H */
