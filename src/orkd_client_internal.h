/* orkd_client_internal.h — PRIVATE to the orkd client (orkd_client.c + orkd_client_ops.c).
 *
 * The three socket helpers both halves need. They are `static inline` rather than extern on purpose:
 * each is 1-5 lines around a single read()/write() loop, so every TU gets its own INTERNAL-LINKAGE copy
 * and the split adds ZERO new external symbols. (Contrast round 8: orkd.c's 19 handlers could not be
 * done this way — bodies that large do not belong in a header, so they became extern with an orkd_
 * prefix. Small wrapper -> static inline; real function -> extern. See MODULARIZE_PLAN.md round 9.) */
#ifndef ORKD_CLIENT_INTERNAL_H
#define ORKD_CLIENT_INTERNAL_H

#include <errno.h>
#include <stddef.h>
#include <unistd.h>
#include <stdint.h>
#include "orkd_client.h"
#include "orkd_ring.h"

/* The connection object. Defined HERE, not in orkd_client.c, because BOTH halves dereference it:
 * the transport half owns fd/ring, and every RPC wrapper in orkd_client_ops.c reads c->fd. The public
 * orkd_client.h keeps `typedef struct orkd_conn orkd_conn;` opaque, so callers still cannot see inside.
 *
 * Missing this cost a build: the interface analysis looked for the string "struct orkd_conn" and the
 * ops file only ever writes the TYPEDEF ALIAS, `orkd_conn *c`. A typedef'd struct crosses a TU
 * boundary under a name the struct keyword never appears in. */
struct orkd_conn { int fd; uint32_t client_id; uint32_t soc_cores; uint32_t prio; uint32_t pack_domain;
                   uint32_t op_domain;   /* v2: IOMMU domain stamped on the NEXT op-submit (run/ring/chain/seq); set by orkd_set_op_domain
                                          * right before a submit (the library sources it from the weight's domain). 0 = weight's pack-time domain. */
                   struct orkd_ring *ring; int ring_fd; uint32_t ring_next; };  /* A-ring: low-latency shm transport */

static inline int wn(int fd, const void *b, size_t n){
    const char *p = b; size_t k = 0;
    while (k < n){ ssize_t r = write(fd, p+k, n-k); if (r < 0){ if (errno==EINTR) continue; return -1; } k += (size_t)r; }
    return 0;
}

static inline int rn(int fd, void *b, size_t n){
    char *p = b; size_t k = 0;
    while (k < n){ ssize_t r = read(fd, p+k, n-k); if (r == 0) return 0; if (r < 0){ if (errno==EINTR) continue; return -1; } k += (size_t)r; }
    return 1;
}

static inline void cdrain(int fd, size_t n){ char b[4096]; while (n){ size_t k = n > sizeof b ? sizeof b : n; if (rn(fd, b, k) <= 0) return; n -= k; } }

#endif /* ORKD_CLIENT_INTERNAL_H */
