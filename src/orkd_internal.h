/* orkd_internal.h — PRIVATE interface between the daemon core (orkd.c) and its request handlers
 * (orkd_handlers.c). Not a public API: nothing outside those two files may include this.
 *
 * orkd.c was 1,003 lines. Splitting it costs exactly these 28 symbols their internal linkage, because
 * every handler calls the shared I/O helpers and dispatch_one calls every handler — the file has no
 * cheaper seam (measured across five cut points; see MODULARIZE_PLAN.md round 8). They carry the orkd_
 * prefix because AGENTS requires one once a symbol crosses a translation unit.
 *
 * The two file-scope globals do NOT cross: g_dom_inuse is reached only through its accessors, and the
 * cut is placed just above g_ring_c so it stays with ring_service, the only code that touches it. */
#ifndef ORKD_INTERNAL_H
#define ORKD_INTERNAL_H

#include "orkd_proto.h"
#include "ork_npu.h"

/* --- shared limits and per-client state: defined HERE, not in orkd.c, because the handlers need the
 * full types (they touch cl->fd, the weight table, the KV table) and the dispatch loop needs them too.
 * A function-prototype-only header was not enough: splitting a TU shares TYPES and MACROS as well. */
#define ORKD_MAX_WEIGHTS 512

#define ORKD_MAX_BYTES (512u << 20)         /* sanity cap on a single weight/A transfer */

struct cweight { uint64_t id; ork_w *w; int K, N, dtype; };   /* dtype = ORKD_DT_I8 | ORKD_DT_F16 (wire dtype) */

struct ckv { uint64_t id; ork_kv_resident *kv; };   /* Tier 12f resident-KV handle (its wkt/wv are also registered in wt[]) */

struct client { int fd; int hello; uint32_t id; struct cweight wt[ORKD_MAX_WEIGHTS]; int nw; uint64_t next_wid; int domain;
                struct ckv kvt[ORKD_MAX_WEIGHTS]; int nkv;   /* resident-KV handles for ORKD_KV_APPEND */
                uint64_t owned_dom;   /* bitmask of IOMMU domains this client requested via ORKD_DOM_REQ (bits 1..63); released on drop */
                struct orkd_ring *ring; int ring_fd; uint32_t ring_tail;   /* A-ring: attached low-latency shm transport */
                uint64_t layer_warm[7]; int layer_warmed; };   /* ORKD_LAYER: cache the last-warmed 7-weight-id set so the doorbell warm (per weight shape) fires ONCE per weight set, not every layer/token (steady-state: same 7 weights each token) */

#define ORKD_NDOM 64                          /* owned_dom bitmask width (domain ids 1..63; 0 = shared/default) */

struct work {
    int used, fd, type, M, K, N, m0, rc, dtype, domain;
    uint32_t prio; uint64_t tag, seq, weight_id;
    int8_t *A; int32_t *C;          /* A/C: socket-malloc'd unless the matching *_imp is set (zero-copy import) */
    void *A_imp, *C_imp;            /* imported dma-bufs to ork_dma_free (NULL => socket-malloc'd A/C) */
};


/* --- daemon core (orkd.c), used by the handlers --- */
int orkd_dom_alloc_explicit(void);
void orkd_dom_release(int d);
int orkd_drain(int fd, size_t n);
int orkd_esz_a(int wire_dt);
int orkd_readn(int fd, void *buf, size_t n);
void orkd_send_error(int fd, uint64_t tag, uint32_t code, const char *msg);
int orkd_send_msg(int fd, uint32_t type, uint64_t tag, const void *payload, uint32_t len);
struct work *orkd_wk_alloc(void);
int orkd_writen(int fd, const void *buf, size_t n);

/* --- request handlers (orkd_handlers.c), dispatched by orkd.c --- */
int orkd_handle_import(struct client *cl, ork_npu *npu, int bb_fd, int bf_fd, uint64_t tag);
int orkd_handle_ffn(struct client *cl, ork_npu *npu, uint64_t tag);
int orkd_handle_attn(struct client *cl, ork_npu *npu, uint64_t tag);
int orkd_handle_attn_rr(struct client *cl, ork_npu *npu, uint64_t tag);
int orkd_handle_layer(struct client *cl, ork_npu *npu, uint64_t tag);
int orkd_handle_kv_alloc(struct client *cl, ork_npu *npu, uint64_t tag);
int orkd_handle_kv_append(struct client *cl, ork_npu *npu, uint64_t tag);
int orkd_handle_pack(struct client *cl, ork_npu *npu, uint64_t tag);
int orkd_handle_run(struct client *cl, ork_npu *npu, uint64_t tag);
int orkd_handle_free(struct client *cl, ork_npu *npu, uint64_t tag);
int orkd_handle_sdp(struct client *cl, ork_npu *npu, uint64_t tag);
int orkd_handle_chain(struct client *cl, ork_npu *npu, uint64_t tag);
int orkd_handle_seq(struct client *cl, ork_npu *npu, uint64_t tag);
int orkd_handle_run_zc2(struct client *cl, ork_npu *npu, int a_fd, int c_fd, uint64_t tag);
int orkd_handle_run_zc(struct client *cl, ork_npu *npu, int a_fd, uint64_t tag);
int orkd_handle_dmabuf(struct client *cl, ork_npu *npu, int dfd, uint64_t tag);
int orkd_handle_dom_req(struct client *cl, uint64_t tag);
int orkd_handle_dom_rel(struct client *cl, uint64_t tag);
int orkd_handle_ring_setup(struct client *cl, int rfd, uint64_t tag);

#endif /* ORKD_INTERNAL_H */
