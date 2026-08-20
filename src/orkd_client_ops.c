/* orkd_client_ops.c — the orkd client's RPC op wrappers: one function per wire opcode.
 *
 * Split verbatim out of orkd_client.c (MODULARIZE_PLAN.md round 9) as a CONTIGUOUS line range. The
 * connection/spawn/atexit machinery, the A-ring fast path and the domain calls stay in orkd_client.c;
 * the three shared socket helpers are static inline in orkd_client_internal.h, so this split adds no
 * external symbols at all. */
#include "orkd_client.h"
#include "orkd_client_internal.h"
#include "orkd_proto.h"
#include "orkd_shm.h"
#include "orkd_ring.h"
#include <errno.h>
#include <stdatomic.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include "orkd_client_internal.h"

uint64_t orkd_pack_i8(orkd_conn *c, int K, int N, const int8_t *B){
    if (!c || c->fd < 0 || K <= 0 || N <= 0 || !B) return 0;
    uint32_t bytes = (uint32_t)((size_t)K * N);
    struct orkd_pack pk; memset(&pk, 0, sizeof pk); pk.K = (uint32_t)K; pk.N = (uint32_t)N; pk.dtype = ORKD_DT_I8; pk.bytes = bytes; pk.domain = c->pack_domain;
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

/* Tier 12f resident-KV alloc: daemon allocs K^T[512,Lmax]+V[Lmax,HD]; returns kv handle + the two weight ids. */
uint64_t orkd_kv_alloc(orkd_conn *c, int HD, int Lmax, uint64_t *wkt_id, uint64_t *wv_id){
    if (!c || c->fd < 0 || HD <= 0 || Lmax <= 0) return 0;
    struct orkd_kv_alloc rq; memset(&rq, 0, sizeof rq); rq.HD = (uint32_t)HD; rq.Lmax = (uint32_t)Lmax;
    struct orkd_hdr h = { ORKD_KV_ALLOC, (uint32_t)sizeof rq, 3 };
    if (wn(c->fd, &h, sizeof h) || wn(c->fd, &rq, sizeof rq)) return 0;
    struct orkd_hdr rh;
    if (rn(c->fd, &rh, sizeof rh) <= 0) return 0;
    if (rh.type != ORKD_KV_ALLOC_OK){ if (rh.len) cdrain(c->fd, rh.len); return 0; }
    struct orkd_kv_alloc_ok ok;
    if (rn(c->fd, &ok, sizeof ok) <= 0) return 0;
    if (rh.len > sizeof ok) cdrain(c->fd, rh.len - sizeof ok);
    if (ok.rc) return 0;
    if (wkt_id) *wkt_id = ok.wkt_id; if (wv_id) *wv_id = ok.wv_id;
    return ok.kv_id;
}
/* Append key `key`: kcol[HD] (K vector) + vrow[HD] (V vector), int8. 0/ok, <0 err. */
int orkd_kv_append(orkd_conn *c, uint64_t kv_id, int key, int HD, const int8_t *kcol, const int8_t *vrow){
    if (!c || c->fd < 0 || HD <= 0 || key < 0 || !kcol || !vrow) return -2;
    struct orkd_kv_append rq; memset(&rq, 0, sizeof rq); rq.kv_id = kv_id; rq.key = (uint32_t)key; rq.HD = (uint32_t)HD;
    struct orkd_hdr h = { ORKD_KV_APPEND, (uint32_t)(sizeof rq + 2*(size_t)HD), 3 };
    if (wn(c->fd, &h, sizeof h) || wn(c->fd, &rq, sizeof rq) || wn(c->fd, kcol, (size_t)HD) || wn(c->fd, vrow, (size_t)HD)) return -1;
    struct orkd_hdr rh;
    if (rn(c->fd, &rh, sizeof rh) <= 0) return -1;
    if (rh.type != ORKD_KV_APPEND_OK){ if (rh.len) cdrain(c->fd, rh.len); return -1; }
    struct orkd_handle hh;
    if (rn(c->fd, &hh, sizeof hh) <= 0) return -1;
    if (rh.len > sizeof hh) cdrain(c->fd, rh.len - sizeof hh);
    return hh.rc;
}

/* Import client-owned, PRE-TILED weight dma-buf(s) into the daemon (ORKD_IMPORT). fds ride SCM_RIGHTS on the
 * header: bb_fd (Bb tiles) always, bf_fd (full-K Bf, its own buffer) when bf_bytes>0. The daemon maps them
 * into this client's pack_domain (Bf into its OWN obj) and returns a weight id. 0 on failure. The caller keeps
 * ownership of the fds (may close them after this returns — the daemon dup'd via SCM_RIGHTS). */
uint64_t orkd_import_i8(orkd_conn *c, int K, int N, int bb_fd, int bf_fd, uint64_t bb_bytes, uint64_t bf_bytes){
    if (!c || c->fd < 0 || K <= 0 || N <= 0 || bb_fd < 0) return 0;
    struct orkd_import im; memset(&im, 0, sizeof im);
    im.K = (uint32_t)K; im.N = (uint32_t)N; im.dtype = ORKD_DT_I8; im.domain = c->pack_domain; im.bb_bytes = bb_bytes; im.bf_bytes = bf_bytes;
    int fds[2]; int nfd = 0; fds[nfd++] = bb_fd; if (bf_fd >= 0 && bf_bytes) fds[nfd++] = bf_fd;
    struct orkd_hdr h = { ORKD_IMPORT, (uint32_t)sizeof im, 47 };
    if (orkd_send_hdr_fds(c->fd, &h, fds, nfd) || wn(c->fd, &im, sizeof im)) return 0;
    struct orkd_hdr rh;
    if (rn(c->fd, &rh, sizeof rh) <= 0) return 0;
    if (rh.type != ORKD_IMPORT_OK){ if (rh.len) cdrain(c->fd, rh.len); return 0; }
    struct orkd_handle hh;
    if (rn(c->fd, &hh, sizeof hh) <= 0) return 0;
    if (rh.len > sizeof hh) cdrain(c->fd, rh.len - sizeof hh);
    return hh.rc == 0 ? hh.id : 0;
}

/* Coalesced FFN inner (ORKD_FFN): the whole SwiGLU [gate->silu->up->glu->down] as ONE daemon-side chain
 * submit against 3 resident weights. Sends A (M*K int8), receives the down output (M*Kd int32). 0/ok, <0 err. */
int orkd_ffn_i8(orkd_conn *c, uint64_t gate_id, uint64_t up_id, uint64_t down_id,
                int M, int K, int Nff, int Kd,
                int gate_mult, int gate_shift, int up_mult, int up_shift, int glu_mult, int glu_shift,
                double in_scale, double out_scale, const int8_t *A, int32_t *out){
    if (!c || c->fd < 0 || M <= 0 || K <= 0 || Nff <= 0 || Kd <= 0 || !A || !out) return -1;
    struct orkd_ffn f; memset(&f, 0, sizeof f);
    f.gate_id = gate_id; f.up_id = up_id; f.down_id = down_id;
    f.M = (uint32_t)M; f.K = (uint32_t)K; f.Nff = (uint32_t)Nff; f.Kd = (uint32_t)Kd; f.domain = c->op_domain;
    f.gate_mult = gate_mult; f.gate_shift = gate_shift; f.up_mult = up_mult; f.up_shift = up_shift; f.glu_mult = glu_mult; f.glu_shift = glu_shift;
    f.in_scale = in_scale; f.out_scale = out_scale;
    uint32_t abytes = (uint32_t)((size_t)M * K); f.abytes = abytes;
    struct orkd_hdr h = { ORKD_FFN, (uint32_t)(sizeof f + abytes), 49 };
    if (wn(c->fd, &h, sizeof h) || wn(c->fd, &f, sizeof f) || wn(c->fd, A, abytes)) return -1;
    struct orkd_hdr rh;
    if (rn(c->fd, &rh, sizeof rh) <= 0) return -1;
    if (rh.type != ORKD_FFN_OK){ if (rh.len) cdrain(c->fd, rh.len); return -1; }
    struct orkd_handle hh;
    if (rn(c->fd, &hh, sizeof hh) <= 0) return -1;
    size_t consumed = sizeof hh, dbytes = (size_t)M * Kd * 4;
    if (hh.rc == 0){
        if (rh.len < consumed + dbytes){ if (rh.len > consumed) cdrain(c->fd, rh.len - consumed); return -1; }
        if (rn(c->fd, out, dbytes) <= 0) return -1;
        consumed += dbytes;
    }
    if (rh.len > consumed) cdrain(c->fd, rh.len - consumed);
    return hh.rc;
}

/* Fused attention core (ORKD_ATTN, chainav): [QK^T->exp->reduce,e.V] in ONE daemon-side submit against 3
 * resident weights (K^T[Kp,Nk], ones[Nk,32], V[Nk,dv]). Sends Q (Nq*Kp int8); receives Sigma (Nq*32 int32)
 * then av (Nq*dv int32). Caller normalizes attn = av/Sigma. 0/ok, <0 err. */
int orkd_attn_i8(orkd_conn *c, uint64_t wkt_id, uint64_t wones_id, uint64_t wv_id,
                 int Nq, int Nk, int Kp, int dv, int r_mult, int r_shift,
                 double in_scale, double out_scale, double max_bias, const int8_t *Q, int32_t *Sigma, int32_t *av){
    if (!c || c->fd < 0 || Nq <= 0 || Nk <= 0 || Kp <= 0 || dv <= 0 || !Q || !Sigma || !av) return -1;
    struct orkd_attn a; memset(&a, 0, sizeof a);
    a.wkt_id = wkt_id; a.wones_id = wones_id; a.wv_id = wv_id;
    a.Nq = (uint32_t)Nq; a.Nk = (uint32_t)Nk; a.Kp = (uint32_t)Kp; a.dv = (uint32_t)dv; a.domain = c->op_domain;
    a.r_mult = r_mult; a.r_shift = r_shift; a.in_scale = in_scale; a.out_scale = out_scale; a.max_bias = max_bias;
    uint32_t abytes = (uint32_t)((size_t)Nq * Kp); a.abytes = abytes;
    struct orkd_hdr h = { ORKD_ATTN, (uint32_t)(sizeof a + abytes), 55 };
    if (wn(c->fd, &h, sizeof h) || wn(c->fd, &a, sizeof a) || wn(c->fd, Q, abytes)) return -1;
    struct orkd_hdr rh;
    if (rn(c->fd, &rh, sizeof rh) <= 0) return -1;
    if (rh.type != ORKD_ATTN_OK){ if (rh.len) cdrain(c->fd, rh.len); return -1; }
    struct orkd_handle hh;
    if (rn(c->fd, &hh, sizeof hh) <= 0) return -1;
    size_t consumed = sizeof hh, sbytes = (size_t)Nq * 32 * 4, avbytes = (size_t)Nq * dv * 4;
    if (hh.rc == 0){
        if (rh.len < consumed + sbytes + avbytes){ if (rh.len > consumed) cdrain(c->fd, rh.len - consumed); return -1; }
        if (rn(c->fd, Sigma, sbytes) <= 0) return -1; consumed += sbytes;
        if (rn(c->fd, av,    avbytes) <= 0) return -1; consumed += avbytes;
    }
    if (rh.len > consumed) cdrain(c->fd, rh.len - consumed);
    return hh.rc;
}

/* ORKD_ATTN_RR: N fused attention chains dispatched round-robin across the daemon's NPU cores in ONE round-trip.
 * wkt_ids/wones_ids/wv_ids = per-chain resident weight ids (length nchains). Q = nchains*Nq*Kp int8 (chain-major).
 * Sigma = nchains*Nq*32, av = nchains*Nq*dv int32 out (attn_c = av_c/Sigma_c). All chains share the requant + exp
 * LUT (in/out_scale,max_bias). Returns the dispatch rc (0 ok, <0 err). */
int orkd_attn_rr_i8(orkd_conn *c, int nchains, const uint64_t *wkt_ids, const uint64_t *wones_ids, const uint64_t *wv_ids,
                    int Nq, int Nk, int Kp, int dv, int r_mult, int r_shift,
                    double in_scale, double out_scale, double max_bias, const int8_t *Q, int32_t *Sigma, int32_t *av){
    if (!c || c->fd < 0 || nchains < 1 || nchains > ORKD_ATTN_RR_MAX || Nq <= 0 || Nk <= 0 || Kp <= 0 || dv <= 0) return -1;
    if (!wkt_ids || !wones_ids || !wv_ids || !Q || !Sigma || !av) return -1;
    struct orkd_attn_rr a; memset(&a, 0, sizeof a);
    a.nchains = (uint32_t)nchains; a.Nq = (uint32_t)Nq; a.Nk = (uint32_t)Nk; a.Kp = (uint32_t)Kp; a.dv = (uint32_t)dv;
    a.domain = c->op_domain; a.r_mult = r_mult; a.r_shift = r_shift; a.in_scale = in_scale; a.out_scale = out_scale; a.max_bias = max_bias;
    uint32_t abytes = (uint32_t)((size_t)nchains * Nq * Kp); a.abytes = abytes;
    uint64_t *trip = malloc((size_t)nchains * 3 * 8); if (!trip) return -1;
    for (int n = 0; n < nchains; n++){ trip[n*3]=wkt_ids[n]; trip[n*3+1]=wones_ids[n]; trip[n*3+2]=wv_ids[n]; }
    uint32_t tbytes = (uint32_t)((size_t)nchains * 3 * 8);
    struct orkd_hdr h = { ORKD_ATTN_RR, (uint32_t)(sizeof a + tbytes + abytes), 57 };
    int werr = wn(c->fd, &h, sizeof h) || wn(c->fd, &a, sizeof a) || wn(c->fd, trip, tbytes) || wn(c->fd, Q, abytes);
    free(trip);
    if (werr) return -1;
    struct orkd_hdr rh;
    if (rn(c->fd, &rh, sizeof rh) <= 0) return -1;
    if (rh.type != ORKD_ATTN_RR_OK){ if (rh.len) cdrain(c->fd, rh.len); return -1; }
    struct orkd_handle hh;
    if (rn(c->fd, &hh, sizeof hh) <= 0) return -1;
    size_t consumed = sizeof hh, sperc = (size_t)Nq * 32 * 4, avperc = (size_t)Nq * dv * 4;
    size_t want = (size_t)nchains * (sperc + avperc);
    if (hh.rc == 0){
        if (rh.len < consumed + want){ if (rh.len > consumed) cdrain(c->fd, rh.len - consumed); return -1; }
        for (int n = 0; n < nchains; n++){
            if (rn(c->fd, Sigma + (size_t)n*Nq*32, sperc) <= 0) return -1; consumed += sperc;
            if (rn(c->fd, av    + (size_t)n*Nq*dv, avperc) <= 0) return -1; consumed += avperc;
        }
    }
    if (rh.len > consumed) cdrain(c->fd, rh.len - consumed);
    return hh.rc;
}

/* ORKD_LAYER: the daemon runs a whole decode layer on the spine in ONE round-trip. `h` carries the dims/ids/
 * scales (caller fills wq..wd, D..nkv, pos, attn_scale, rope_base). Payload arrays are fp32; x_out=[D] fp32.
 * Returns the daemon rc (0 ok). */
int orkd_layer_i8(orkd_conn *c, struct orkd_layer *h,
                  const float *attn_norm, const float *q_norm, const float *ffn_norm,
                  const float *x, const float *Kc, const float *Vc, float *x_out){
    if (!c || c->fd < 0 || !h || !attn_norm || !q_norm || !ffn_norm || !x || !Kc || !Vc || !x_out) return -1;
    int D=(int)h->D, dk=(int)h->dk, dv=(int)h->dv, Hkv=(int)h->Hkv, nkv=(int)h->nkv;
    size_t an=(size_t)D*4, qn=(size_t)dk*4, fn=(size_t)D*4, xb=(size_t)D*4;
    size_t kc=(size_t)Hkv*nkv*dk*4, vc=(size_t)Hkv*nkv*dv*4;
    h->domain = c->op_domain; h->pbytes = (uint32_t)(an+qn+fn+xb+kc+vc);
    struct orkd_hdr hh = { ORKD_LAYER, (uint32_t)(sizeof *h + h->pbytes), 59 };
    if (wn(c->fd,&hh,sizeof hh) || wn(c->fd,h,sizeof *h) || wn(c->fd,attn_norm,an) || wn(c->fd,q_norm,qn) ||
        wn(c->fd,ffn_norm,fn) || wn(c->fd,x,xb) || wn(c->fd,Kc,kc) || wn(c->fd,Vc,vc)) return -1;
    struct orkd_hdr rh;
    if (rn(c->fd,&rh,sizeof rh) <= 0) return -1;
    if (rh.type != ORKD_LAYER_OK){ if (rh.len) cdrain(c->fd, rh.len); return -1; }
    struct orkd_handle rhh;
    if (rn(c->fd,&rhh,sizeof rhh) <= 0) return -1;
    size_t consumed=sizeof rhh, ob=(size_t)D*4;
    if (rhh.rc == 0){ if (rh.len < consumed+ob){ if(rh.len>consumed) cdrain(c->fd,rh.len-consumed); return -1; }
        if (rn(c->fd,x_out,ob) <= 0) return -1; consumed+=ob; }
    if (rh.len > consumed) cdrain(c->fd, rh.len - consumed);
    return rhh.rc;
}

int orkd_run_i8(orkd_conn *c, uint64_t weight_id, int M, int K, int N, const int8_t *A, int32_t *C){
    if (!c || c->fd < 0 || M <= 0 || K <= 0 || N <= 0 || !A || !C) return -1;
    uint32_t abytes = (uint32_t)((size_t)M * K);
    struct orkd_run rq; memset(&rq, 0, sizeof rq); rq.weight_id = weight_id; rq.M = (uint32_t)M; rq.abytes = abytes; rq.flags = c->prio; rq.domain = c->op_domain;
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
    struct orkd_pack pk; memset(&pk, 0, sizeof pk); pk.K = (uint32_t)K; pk.N = (uint32_t)N; pk.dtype = ORKD_DT_F16; pk.bytes = bytes; pk.domain = c->pack_domain;
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
    struct orkd_run rq; memset(&rq, 0, sizeof rq); rq.weight_id = weight_id; rq.M = (uint32_t)M; rq.abytes = abytes; rq.flags = c->prio; rq.domain = c->op_domain;
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
    struct orkd_pack pk; memset(&pk, 0, sizeof pk); pk.K = (uint32_t)K; pk.N = (uint32_t)N; pk.dtype = ORKD_DT_I4; pk.bytes = bytes; pk.domain = c->pack_domain;
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
    struct orkd_run rq; memset(&rq, 0, sizeof rq); rq.weight_id = weight_id; rq.M = (uint32_t)M; rq.abytes = abytes; rq.flags = c->prio; rq.domain = c->op_domain;
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
/* remaining i8 activations + the int16 activation/ewmul/add family (in/out int16 as void*) */
int orkd_rsqrt_i8(orkd_conn *c, const int8_t *in, int M, int N, double is, double os, int8_t *out){
    return orkd_sdp_call(c, ORKD_SDP_RSQRT_I8, M, N, 1, 1, 1, 0, 0, is, os, 0, 0, in, NULL, out); }
int orkd_exp_i8  (orkd_conn *c, const int8_t *in, int M, int N, double is, double os, int8_t *out){
    return orkd_sdp_call(c, ORKD_SDP_EXP_I8, M, N, 1, 1, 1, 0, 0, is, os, 0, 0, in, NULL, out); }
int orkd_silu_i16(orkd_conn *c, const void *in, int M, int N, double is, double os, void *out){
    return orkd_sdp_call(c, ORKD_SDP_SILU_I16, M, N, 1, 2, 2, 0, 0, is, os, 0, 0, in, NULL, out); }
int orkd_gelu_i16(orkd_conn *c, const void *in, int M, int N, double is, double os, void *out){
    return orkd_sdp_call(c, ORKD_SDP_GELU_I16, M, N, 1, 2, 2, 0, 0, is, os, 0, 0, in, NULL, out); }
int orkd_rsqrt_i16(orkd_conn *c, const void *in, int M, int N, double is, double os, void *out){
    return orkd_sdp_call(c, ORKD_SDP_RSQRT_I16, M, N, 1, 2, 2, 0, 0, is, os, 0, 0, in, NULL, out); }
int orkd_exp_i16 (orkd_conn *c, const void *in, int M, int N, double is, double os, void *out){
    return orkd_sdp_call(c, ORKD_SDP_EXP_I16, M, N, 1, 2, 2, 0, 0, is, os, 0, 0, in, NULL, out); }
int orkd_ewmul_i16(orkd_conn *c, const void *a, const void *b, int M, int N, int mult, int shift, void *out){
    return orkd_sdp_call(c, ORKD_SDP_EWMUL_I16, M, N, 2, 2, 2, mult, shift, 0, 0, 0, 0, a, b, out); }
int orkd_add_i16 (orkd_conn *c, const void *a, const void *b, int M, int N, double as, double bs, double os, void *out){
    return orkd_sdp_call(c, ORKD_SDP_ADD_I16, M, N, 2, 2, 2, 0, 0, 0, os, as, bs, a, b, out); }

int orkd_run_chain_i8(orkd_conn *c, int S, const orkd_chain_task_c *t){
    if (!c || c->fd < 0 || S < 1 || S > ORKD_CHAIN_MAX || !t) return -1;
    uint64_t atot = 0, ctot = 0;
    for (int i = 0; i < S; i++){ if (t[i].M <= 0 || t[i].K <= 0 || t[i].N <= 0 || !t[i].A || !t[i].C) return -1;
        atot += (size_t)t[i].M * t[i].K; ctot += (size_t)t[i].M * t[i].N * 4; }
    if (atot > 0xffffffffu) return -1;
    struct orkd_chain_hdr ch; memset(&ch, 0, sizeof ch); ch.S = (uint32_t)S; ch.abytes_total = (uint32_t)atot; ch.domain = c->op_domain;
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
    for (int i = 0; i < n; i++){                       /* A2: a_src/b_src>0 => input is a resident prior output (not uploaded); c_keep => output stays resident (not returned) */
        int up_a = (o[i].a_src == 0), up_b = (o[i].b_src == 0 && o[i].bbytes != 0);
        if ((up_a && !o[i].A) || (up_b && !o[i].B) || (!o[i].c_keep && !o[i].C)) return -1;
        if (up_a) in_total += o[i].abytes;
        if (up_b) in_total += o[i].bbytes;
        if (!o[i].c_keep) c_total += o[i].cbytes; }
    if (in_total > 0xffffffffu) return -1;
    struct orkd_seq_hdr sh; memset(&sh, 0, sizeof sh); sh.n = (uint32_t)n; sh.in_total = (uint32_t)in_total; sh.domain = c->op_domain;
    uint32_t plen = (uint32_t)(sizeof sh + (size_t)n * sizeof(struct orkd_seq_op) + in_total);
    struct orkd_hdr h = { ORKD_SEQ, plen, 11 };
    if (wn(c->fd, &h, sizeof h) || wn(c->fd, &sh, sizeof sh)) return -1;
    for (int i = 0; i < n; i++){ struct orkd_seq_op w; memset(&w, 0, sizeof w);
        w.kind = o[i].kind; w.M = (uint32_t)o[i].M; w.N = (uint32_t)o[i].N; w.weight_id = o[i].weight_id;
        w.abytes = o[i].abytes; w.bbytes = o[i].bbytes; w.cbytes = o[i].cbytes;
        w.mult = o[i].mult; w.shift = o[i].shift; w.group = o[i].group; w.in_scale = o[i].in_scale; w.out_scale = o[i].out_scale; w.b_scale = o[i].b_scale;
        w.a_src = o[i].a_src; w.b_src = o[i].b_src; w.c_keep = (uint32_t)o[i].c_keep;
        if (wn(c->fd, &w, sizeof w)) return -1; }
    for (int i = 0; i < n; i++){ if (o[i].a_src == 0 && wn(c->fd, o[i].A, o[i].abytes)) return -1;
        if (o[i].b_src == 0 && o[i].bbytes && wn(c->fd, o[i].B, o[i].bbytes)) return -1; }
    struct orkd_hdr rh;
    if (rn(c->fd, &rh, sizeof rh) <= 0) return -1;
    if (rh.type != ORKD_SEQ_OK){ if (rh.len) cdrain(c->fd, rh.len); return -1; }
    struct orkd_handle hh;
    if (rn(c->fd, &hh, sizeof hh) <= 0) return -1;
    size_t consumed = sizeof hh, cb = (size_t)c_total;
    if (hh.rc == 0){
        if (rh.len < consumed + cb){ if (rh.len > consumed) cdrain(c->fd, rh.len - consumed); return -1; }
        for (int i = 0; i < n; i++) if (!o[i].c_keep && o[i].cbytes && rn(c->fd, o[i].C, o[i].cbytes) <= 0) return -1;
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
    struct orkd_run rq; memset(&rq, 0, sizeof rq); rq.weight_id = weight_id; rq.M = (uint32_t)M; rq.abytes = 0; rq.flags = c->prio; rq.domain = c->op_domain;
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
    struct orkd_run rq; memset(&rq, 0, sizeof rq); rq.weight_id = weight_id; rq.M = (uint32_t)M; rq.abytes = 0; rq.flags = c->prio; rq.domain = c->op_domain;
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
