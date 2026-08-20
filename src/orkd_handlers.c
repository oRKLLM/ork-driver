/* orkd_handlers.c — the orkd daemon's request handlers, one per wire opcode.
 *
 * Split verbatim out of orkd.c (MODULARIZE_PLAN.md round 8) as a CONTIGUOUS line range. The shared
 * helpers and the handler prototypes are in orkd_internal.h; the dispatch loop, socket accept, ring
 * service and main() stay in orkd.c. */
#include "orkd_proto.h"
#include "orkd_shm.h"
#include "orkd_ring.h"
#include "ork_npu.h"
#include "spine_kernels.h"   /* ORKD_LAYER: the CPU glue kernels the daemon runs on the resident activation */
#include <stdatomic.h>
#include <errno.h>
#include <sys/mman.h>
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
#include "orkd_internal.h"

int orkd_handle_import(struct client *cl, ork_npu *npu, int bb_fd, int bf_fd, uint64_t tag){
    struct orkd_import im;
    if (orkd_readn(cl->fd, &im, sizeof im) <= 0){ if (bb_fd >= 0) close(bb_fd); if (bf_fd >= 0) close(bf_fd); return -1; }
    if (bb_fd < 0){ if (bf_fd >= 0) close(bf_fd); orkd_send_error(cl->fd, tag, ORKD_EPROTO, "import: missing fd"); return 0; }
    if (im.dtype != ORKD_DT_I8){ close(bb_fd); if (bf_fd >= 0) close(bf_fd); orkd_send_error(cl->fd, tag, ORKD_ENOSYS, "import int8 only"); return 0; }
    if (!im.bb_bytes || im.bb_bytes > ORKD_MAX_BYTES || im.bf_bytes > ORKD_MAX_BYTES){ close(bb_fd); if (bf_fd >= 0) close(bf_fd); orkd_send_error(cl->fd, tag, ORKD_EPROTO, "import bad size"); return 0; }
    int pdom = (int)im.domain;
    if (pdom > 0){
        if (pdom >= ORKD_NDOM || !(cl->owned_dom & (1ull << pdom))){ close(bb_fd); if (bf_fd >= 0) close(bf_fd); orkd_send_error(cl->fd, tag, ORKD_EBADH, "domain not owned by client"); return 0; }
    } else pdom = cl->domain;
    ork_npu_set_pack_domain(npu, pdom);
    ork_w *w = ork_i8_mm_adopt_imported(npu, (int)im.K, (int)im.N, bb_fd, bf_fd, (size_t)im.bb_bytes, (size_t)im.bf_bytes);
    ork_npu_set_pack_domain(npu, 0);            /* restore default */
    struct orkd_handle hh; memset(&hh, 0, sizeof hh);
    if (w && cl->nw < ORKD_MAX_WEIGHTS){ hh.id = ++cl->next_wid; hh.rc = 0; cl->wt[cl->nw++] = (struct cweight){ hh.id, w, (int)im.K, (int)im.N, (int)ORKD_DT_I8 }; }
    else { if (w) ork_mm_free(npu, w); hh.rc = -1; }   /* adopt ALWAYS consumes both fds; on w!=NULL, free reclaims the imports */
    orkd_send_msg(cl->fd, ORKD_IMPORT_OK, tag, &hh, sizeof hh);
    return 0;
}
/* ORKD_FFN: the whole SwiGLU FFN inner as ONE coalesced on-NPU chain (ork_i8_mm_run_chain_ffn, SDP-op
 * address-aliasing) against 3 resident weights. Fixed op-list: gate MM8(x,Wg) -> silu -> up MM8(x,Wu) ->
 * glu ewmul -> down MM32(glu,Wd). One socket round-trip + one submit for the entire inner; intermediates
 * never leave the NPU. Reply = the down output (M*Kd int32). ork_i8_mm_run_chain_ffn stages A/C internally
 * and dom_activates the weights' domain, so no domain setup here. Runs INLINE (single-stream serialized). */
int orkd_handle_ffn(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_ffn f;
    if (orkd_readn(cl->fd, &f, sizeof f) <= 0) return -1;
    if (f.abytes > ORKD_MAX_BYTES){ orkd_drain(cl->fd, f.abytes); orkd_send_error(cl->fd, tag, ORKD_EPROTO, "ffn A too big"); return 0; }
    int8_t *A = malloc(f.abytes ? f.abytes : 1);
    if (!A){ orkd_drain(cl->fd, f.abytes); orkd_send_error(cl->fd, tag, ORKD_EOOM, "ffn A"); return 0; }
    if (f.abytes && orkd_readn(cl->fd, A, f.abytes) <= 0){ free(A); return -1; }
    struct cweight *cg = NULL, *cu = NULL, *cd = NULL;
    for (int j = 0; j < cl->nw; j++){ uint64_t id = cl->wt[j].id;
        if (id == f.gate_id) cg = &cl->wt[j]; if (id == f.up_id) cu = &cl->wt[j]; if (id == f.down_id) cd = &cl->wt[j]; }
    int M = (int)f.M, K = (int)f.K, Nff = (int)f.Nff, Kd = (int)f.Kd;
    if (!cg || !cu || !cd || M < 1 || K < 1 || Nff < 1 || Kd < 1 || f.abytes != (uint32_t)((size_t)M * K)){
        free(A); orkd_send_error(cl->fd, tag, (cg && cu && cd) ? ORKD_EPROTO : ORKD_EBADH, "ffn weight/dim"); return 0; }
    /* int8 intermediates ride in the low bytes of int32 slots (matches ork_i8_mm_run_chain_ffn's C usage); down is int32 */
    size_t isz = (size_t)M * Nff * 4, dsz = (size_t)M * Kd * 4;
    int32_t *Cg = malloc(isz), *Cs = malloc(isz), *Cu = malloc(isz), *Ch = malloc(isz), *Cd = malloc(dsz);
    if (!Cg || !Cs || !Cu || !Ch || !Cd){ free(A); free(Cg); free(Cs); free(Cu); free(Ch); free(Cd); orkd_send_error(cl->fd, tag, ORKD_EOOM, "ffn scratch"); return 0; }
    ork_mm_task_i8 t[5] = {
        { cg->w, M, A, Cg }, { cg->w, M, A, Cs }, { cu->w, M, A, Cu }, { cg->w, M, A, Ch }, { cd->w, M, A, Cd } };
    ork_chain_op ops[5] = {
        { 1, -1, 0, f.gate_mult, f.gate_shift },   /* gate MM8 (reads A)               */
        { 2,  0, 0, 0, 0 },                        /* silu(gate = t0)                  */
        { 1, -1, 0, f.up_mult, f.up_shift },       /* up MM8 (reads A)                 */
        { 3,  1, 2, f.glu_mult, f.glu_shift },     /* glu = silu(t1) * up(t2)          */
        { 0,  3, 0, 0, 0 } };                      /* down MM32 (reads glu = t3)       */
    int rc = ork_i8_mm_run_chain_ffn(npu, 5, t, ops, f.in_scale, f.out_scale);
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.rc = rc;
    int payload = (rc == 0);
    struct orkd_hdr rh = { ORKD_FFN_OK, (uint32_t)(sizeof hh + (payload ? dsz : 0)), tag };
    int werr = orkd_writen(cl->fd, &rh, sizeof rh) || orkd_writen(cl->fd, &hh, sizeof hh);
    if (!werr && payload) werr = orkd_writen(cl->fd, Cd, dsz);
    free(A); free(Cg); free(Cs); free(Cu); free(Ch); free(Cd);
    return werr ? -1 : 0;
}
/* ORKD_ATTN: the fused attention core [QK^T->exp->reduce,e.V] as ONE coalesced on-NPU chain (chainav pattern,
 * ork_i8_mm_run_chain_ffn_exp). Fixed op-list built daemon-side against 3 resident weights: K^T[Kp,Nk], ones[Nk,32],
 * V[Nk,dv]. Q (Nq*Kp int8) follows. Reply = Sigma(Nq*32 int32) then av(Nq*dv int32). e never leaves the NPU. */
int orkd_handle_attn(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_attn a;
    if (orkd_readn(cl->fd, &a, sizeof a) <= 0) return -1;
    if (a.abytes > ORKD_MAX_BYTES){ orkd_drain(cl->fd, a.abytes); orkd_send_error(cl->fd, tag, ORKD_EPROTO, "attn Q too big"); return 0; }
    int8_t *Q = malloc(a.abytes ? a.abytes : 1);
    if (!Q){ orkd_drain(cl->fd, a.abytes); orkd_send_error(cl->fd, tag, ORKD_EOOM, "attn Q"); return 0; }
    if (a.abytes && orkd_readn(cl->fd, Q, a.abytes) <= 0){ free(Q); return -1; }
    struct cweight *ckt = NULL, *co = NULL, *cv = NULL;
    for (int j = 0; j < cl->nw; j++){ uint64_t id = cl->wt[j].id;
        if (id == a.wkt_id) ckt = &cl->wt[j]; if (id == a.wones_id) co = &cl->wt[j]; if (id == a.wv_id) cv = &cl->wt[j]; }
    int Nq = (int)a.Nq, Nk = (int)a.Nk, Kp = (int)a.Kp, dv = (int)a.dv;
    if (!ckt || !co || !cv || Nq < 1 || Nk < 1 || Kp < 1 || dv < 1 || a.abytes != (uint32_t)((size_t)Nq * Kp)){
        free(Q); orkd_send_error(cl->fd, tag, (ckt && co && cv) ? ORKD_EPROTO : ORKD_EBADH, "attn weight/dim"); return 0; }
    /* scb/eb hold int8 (scores, exp) in the low bytes of int32 slots (chainav C usage); ss = reduce Sigma[Nq,32]; av[Nq,dv] */
    size_t nb = (size_t)Nq * Nk * 4, sb = (size_t)Nq * 32 * 4, ab = (size_t)Nq * dv * 4;
    int32_t *scb = malloc(nb), *eb = malloc(nb), *ss = malloc(sb), *avb = malloc(ab);
    if (!scb || !eb || !ss || !avb){ free(Q); free(scb); free(eb); free(ss); free(avb); orkd_send_error(cl->fd, tag, ORKD_EOOM, "attn scratch"); return 0; }
    ork_mm_task_i8 t[4] = {
        { ckt->w, Nq, Q,            scb },   /* QK^T -> scores (reads Q)     */
        { ckt->w, Nq, (int8_t*)scb, eb  },   /* exp(t0) -> e (N-sized by wkt) */
        { co->w,  Nq, (int8_t*)eb,  ss  },   /* reduce e(t1) -> Sigma         */
        { cv->w,  Nq, (int8_t*)eb,  avb } }; /* e(t1).V -> av                 */
    ork_chain_op ops[4] = {
        { 1, -1, 0, a.r_mult, a.r_shift },   /* QK^T MM8, int32->int8 score requant */
        { 2,  0, 0, 0, 0 },                  /* exp(t0)                             */
        { 0,  1, 0, 0, 0 },                  /* reduce e(t1) -> Sigma               */
        { 0,  1, 0, 0, 0 } };                /* e(t1).V -> av                       */
    int rc = ork_i8_mm_run_chain_ffn_exp_biased(npu, 4, t, ops, a.in_scale, a.out_scale, a.max_bias);
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.rc = rc;
    int payload = (rc == 0);
    struct orkd_hdr rh = { ORKD_ATTN_OK, (uint32_t)(sizeof hh + (payload ? sb + ab : 0)), tag };
    int werr = orkd_writen(cl->fd, &rh, sizeof rh) || orkd_writen(cl->fd, &hh, sizeof hh);
    if (!werr && payload) werr = (orkd_writen(cl->fd, ss, sb) || orkd_writen(cl->fd, avb, ab));
    free(Q); free(scb); free(eb); free(ss); free(avb);
    return werr ? -1 : 0;
}
/* ORKD_ATTN_RR: N fused attention chains fanned round-robin across the NPU cores in ONE dispatch (the daemon runs
 * a DIRECT ctx, so ork_mm_run_chains_rr_biased takes the local multi-core path). Payload = nchains {wkt,wones,wv}
 * id triples then Q (nchains*Nq*Kp int8, chain-major). Reply = per chain Sigma(Nq*32) then av(Nq*dv) int32. All
 * chains share the requant + scalar-max-biased exp LUT. This is the decode attention path: one round-trip, Hkv
 * kv-head chains on separate cores concurrently, e never leaves the NPU. */
int orkd_handle_attn_rr(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_attn_rr a;
    if (orkd_readn(cl->fd, &a, sizeof a) <= 0) return -1;
    int nch = (int)a.nchains, Nq = (int)a.Nq, Nk = (int)a.Nk, Kp = (int)a.Kp, dv = (int)a.dv;
    size_t tbytes = (nch>=1 && nch<=ORKD_ATTN_RR_MAX) ? (size_t)nch*3*8 : 0;
    /* bad nchains: still consume the triples+Q we can compute, then error. abytes was sent by the client. */
    if (nch < 1 || nch > ORKD_ATTN_RR_MAX || Nq < 1 || Nk < 1 || Kp < 1 || dv < 1 || a.abytes > ORKD_MAX_BYTES
        || a.abytes != (uint32_t)((size_t)nch * Nq * Kp)){
        orkd_drain(cl->fd, tbytes + a.abytes); orkd_send_error(cl->fd, tag, ORKD_EPROTO, "attn_rr dims"); return 0; }
    uint64_t *trip = malloc(tbytes); int8_t *Q = malloc(a.abytes ? a.abytes : 1);
    if (!trip || !Q){ free(trip); free(Q); orkd_drain(cl->fd, tbytes + a.abytes); orkd_send_error(cl->fd, tag, ORKD_EOOM, "attn_rr in"); return 0; }
    if (orkd_readn(cl->fd, trip, tbytes) <= 0 || (a.abytes && orkd_readn(cl->fd, Q, a.abytes) <= 0)){ free(trip); free(Q); return -1; }
    /* per-chain scratch + task lists; ss/avb hold the outputs to reply */
    size_t sb = (size_t)Nq*32*4, ab = (size_t)Nq*dv*4, nb = (size_t)Nq*Nk*4;
    int32_t **scb = calloc(nch,sizeof(void*)), **eb = calloc(nch,sizeof(void*)), **ss = calloc(nch,sizeof(void*)), **avb = calloc(nch,sizeof(void*));
    ork_mm_task_i8 (*tk)[4] = calloc(nch,sizeof(*tk));
    const ork_mm_task_i8 **chains = calloc(nch,sizeof(void*)); int *S = calloc(nch,sizeof(int));
    int oom = (!scb||!eb||!ss||!avb||!tk||!chains||!S), bad = 0;
    for (int n = 0; n < nch && !oom; n++){
        scb[n]=malloc(nb); eb[n]=malloc(nb); ss[n]=malloc(sb); avb[n]=malloc(ab);
        if (!scb[n]||!eb[n]||!ss[n]||!avb[n]){ oom=1; break; }
        uint64_t wkt_id=trip[n*3], wones_id=trip[n*3+1], wv_id=trip[n*3+2];
        struct cweight *ckt=NULL,*co=NULL,*cv=NULL;
        for (int j=0;j<cl->nw;j++){ uint64_t id=cl->wt[j].id;
            if (id==wkt_id) ckt=&cl->wt[j]; if (id==wones_id) co=&cl->wt[j]; if (id==wv_id) cv=&cl->wt[j]; }
        if (!ckt||!co||!cv){ bad=1; break; }
        int8_t *Qn = Q + (size_t)n*Nq*Kp;
        tk[n][0]=(ork_mm_task_i8){ ckt->w, Nq, Qn,             scb[n] };
        tk[n][1]=(ork_mm_task_i8){ ckt->w, Nq, (int8_t*)scb[n], eb[n] };
        tk[n][2]=(ork_mm_task_i8){ co->w,  Nq, (int8_t*)eb[n],  ss[n] };
        tk[n][3]=(ork_mm_task_i8){ cv->w,  Nq, (int8_t*)eb[n],  avb[n] };
        chains[n]=tk[n]; S[n]=4;
    }
    ork_chain_op ops[4] = { {1,-1,0,a.r_mult,a.r_shift}, {2,0,0,0,0}, {0,1,0,0,0}, {0,1,0,0,0} };
    int rc = (oom||bad) ? -1 : ork_mm_run_chains_rr_biased(npu, nch, chains, S, ops, a.in_scale, a.out_scale, a.max_bias);
    int payload = (rc == 0);
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.rc = rc;
    struct orkd_hdr rh = { ORKD_ATTN_RR_OK, (uint32_t)(sizeof hh + (payload ? (size_t)nch*(sb+ab) : 0)), tag };
    if (oom || bad) hh.rc = oom ? ORKD_EOOM : ORKD_EBADH;
    int werr = orkd_writen(cl->fd, &rh, sizeof rh) || orkd_writen(cl->fd, &hh, sizeof hh);
    for (int n = 0; n < nch && !werr && payload; n++) werr = (orkd_writen(cl->fd, ss[n], sb) || orkd_writen(cl->fd, avb[n], ab));
    for (int n = 0; n < nch; n++){ if(scb)free(scb[n]); if(eb)free(eb[n]); if(ss)free(ss[n]); if(avb)free(avb[n]); }
    free(scb); free(eb); free(ss); free(avb); free(tk); free(chains); free(S); free(trip); free(Q);
    return werr ? -1 : 0;
}
/* ORKD_LAYER: the daemon runs a WHOLE decode layer in ONE round-trip. This handler is now a THIN transport
 * shim: deserialize the request + payload, resolve weight ids -> resident ork_w*, then run the layer via the
 * shared lib core ork_i8_mm_layer (npu->daemon is NULL on the daemon's own ctx, so it takes the LOCAL path —
 * the exact same compute a direct-NPU client runs; that is the lib<->orkd parity guarantee). Reply = x_out[D].
 * All compute/warm/coherence lives in ork_i8_mm_layer (src/npu.c). NOTE: decode-on-NPU is a measured perf loss;
 * this path exists for parity/correctness, not speed. */
static ork_w *layer_find_w(struct client *cl, uint64_t id){ for (int j=0;j<cl->nw;j++) if (cl->wt[j].id==id) return cl->wt[j].w; return NULL; }
int orkd_handle_layer(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_layer a;
    if (orkd_readn(cl->fd, &a, sizeof a) <= 0) return -1;
    int D=(int)a.D;
    size_t an=(size_t)a.D*4,qn=(size_t)a.dk*4,fn=(size_t)a.D*4,xb=(size_t)a.D*4,
           kc=(size_t)a.Hkv*a.nkv*a.dk*4,vc=(size_t)a.Hkv*a.nkv*a.dv*4;
    if (a.D<1||a.H<1||a.Hkv<1||a.dk<1||a.dv<1||a.Nff<1||a.nkv<1 || a.pbytes!=(uint32_t)(an+qn+fn+xb+kc+vc) || a.pbytes>ORKD_MAX_BYTES){
        orkd_drain(cl->fd, a.pbytes); orkd_send_error(cl->fd, tag, ORKD_EPROTO, "layer dims"); return 0; }
    char *pl = malloc(a.pbytes); if (!pl){ orkd_drain(cl->fd,a.pbytes); orkd_send_error(cl->fd,tag,ORKD_EOOM,"layer pl"); return 0; }
    if (orkd_readn(cl->fd, pl, a.pbytes) <= 0){ free(pl); return -1; }
    const float *attn_norm=(const float*)pl, *q_norm=(const float*)(pl+an), *ffn_norm=(const float*)(pl+an+qn),
          *x=(const float*)(pl+an+qn+fn), *Kc=(const float*)(pl+an+qn+fn+xb), *Vc=(const float*)(pl+an+qn+fn+xb+kc);
    ork_w *pWq=layer_find_w(cl,a.wq),*pWk=layer_find_w(cl,a.wk),*pWv=layer_find_w(cl,a.wv),*pWo=layer_find_w(cl,a.wo),
          *pWg=layer_find_w(cl,a.wg),*pWu=layer_find_w(cl,a.wu),*pWd=layer_find_w(cl,a.wd);
    if (!pWq||!pWk||!pWv||!pWo||!pWg||!pWu||!pWd){ free(pl); orkd_send_error(cl->fd,tag,ORKD_EBADH,"layer weight id"); return 0; }
    float *xo=malloc((size_t)D*4);
    struct ork_layer_dims dd={ a.D,a.H,a.Hkv,a.dk,a.dv,a.Nff,a.nkv,a.pos, a.attn_scale,a.rope_base };
    int rc = xo ? ork_i8_mm_layer(npu,&dd,pWq,pWk,pWv,pWo,pWg,pWu,pWd,attn_norm,q_norm,ffn_norm,x,Kc,Vc,xo) : -1;
    struct orkd_handle hh; memset(&hh,0,sizeof hh); hh.rc=rc;
    struct orkd_hdr rh = { ORKD_LAYER_OK, (uint32_t)(sizeof hh + (rc==0 ? (size_t)D*4 : 0)), tag };
    int werr = orkd_writen(cl->fd,&rh,sizeof rh) || orkd_writen(cl->fd,&hh,sizeof hh);
    if (!werr && rc==0) werr = orkd_writen(cl->fd, xo, (size_t)D*4);
    free(xo); free(pl);
    return werr ? -1 : 0;
}
/* Tier 12f resident-KV: the daemon runs a DIRECT npu ctx (fd valid), so ork_kv_resident_alloc/append take the
 * LOCAL path here. Register the two resident weights in wt[] (so ORKD_CHAIN can reference them by id) and keep
 * the kv handle in kvt[] for append. Freed with the rest on client drop. */
int orkd_handle_kv_alloc(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_kv_alloc rq;
    if (orkd_readn(cl->fd, &rq, sizeof rq) <= 0) return -1;
    struct orkd_kv_alloc_ok ok; memset(&ok, 0, sizeof ok); ok.rc = -1;
    ork_kv_resident *kv = ork_kv_resident_alloc(npu, (int)rq.HD, (int)rq.Lmax);
    if (kv && cl->nw + 2 <= ORKD_MAX_WEIGHTS && cl->nkv < ORKD_MAX_WEIGHTS){
        uint64_t wkt_id = ++cl->next_wid, wv_id = ++cl->next_wid, kv_id = ++cl->next_wid;
        cl->wt[cl->nw++] = (struct cweight){ wkt_id, kv->wkt, kv->Kp, (int)rq.Lmax, (int)ORKD_DT_I8 };
        cl->wt[cl->nw++] = (struct cweight){ wv_id,  kv->wv,  (int)rq.Lmax, (int)rq.HD, (int)ORKD_DT_I8 };
        cl->kvt[cl->nkv++] = (struct ckv){ kv_id, kv };
        ok.rc = 0; ok.kv_id = kv_id; ok.wkt_id = wkt_id; ok.wv_id = wv_id;
    } else if (kv) ork_kv_resident_free(npu, kv);
    orkd_send_msg(cl->fd, ORKD_KV_ALLOC_OK, tag, &ok, sizeof ok);
    return 0;
}
int orkd_handle_kv_append(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_kv_append rq;
    if (orkd_readn(cl->fd, &rq, sizeof rq) <= 0) return -1;
    uint32_t HD = rq.HD; size_t nb = 2 * (size_t)HD;
    if (HD == 0 || HD > 4096){ if (nb) orkd_drain(cl->fd, nb); orkd_send_error(cl->fd, tag, ORKD_EPROTO, "bad HD"); return 0; }
    int8_t *buf = malloc(nb);
    if (!buf){ orkd_drain(cl->fd, nb); orkd_send_error(cl->fd, tag, ORKD_EOOM, "kv append alloc"); return 0; }
    if (orkd_readn(cl->fd, buf, nb) <= 0){ free(buf); return -1; }
    ork_kv_resident *kv = NULL;
    for (int i = 0; i < cl->nkv; i++) if (cl->kvt[i].id == rq.kv_id){ kv = cl->kvt[i].kv; break; }
    struct orkd_handle hh; memset(&hh, 0, sizeof hh);
    hh.rc = kv ? ork_kv_append(npu, kv, (int)rq.key, buf, buf + HD) : -2;
    free(buf);
    orkd_send_msg(cl->fd, ORKD_KV_APPEND_OK, tag, &hh, sizeof hh);
    return 0;
}
/* #2b-1 submit RPC (int8, socket-transfer; dma-buf zero-copy is #2b-2). Handlers read their own payload from
 * the fd and reply; return <0 to drop the client. The NPU op (ork_i8_mm_run) rides the interruptible doorbell,
 * so a RUN in flight never puts orkd in D-state -> orkd stays SIGTERM-clean and never blocks system shutdown. */
int orkd_handle_pack(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_pack pk;
    if (orkd_readn(cl->fd, &pk, sizeof pk) <= 0) return -1;
    if (pk.bytes == 0 || pk.bytes > ORKD_MAX_BYTES){ orkd_drain(cl->fd, pk.bytes); orkd_send_error(cl->fd, tag, ORKD_EPROTO, "bad pack size"); return 0; }
    if (pk.dtype != ORKD_DT_I8 && pk.dtype != ORKD_DT_F16 && pk.dtype != ORKD_DT_I4){ orkd_drain(cl->fd, pk.bytes); orkd_send_error(cl->fd, tag, ORKD_ENOSYS, "int8/fp16/int4 only"); return 0; }
    int8_t *wbuf = malloc(pk.bytes);
    if (!wbuf){ orkd_drain(cl->fd, pk.bytes); orkd_send_error(cl->fd, tag, ORKD_EOOM, "pack alloc"); return 0; }
    if (orkd_readn(cl->fd, wbuf, pk.bytes) <= 0){ free(wbuf); return -1; }
    /* CLIENT-CHOSEN DOMAIN: pk.domain>0 means the client explicitly requested this domain (ORKD_DOM_REQ) and is
     * packing into it — validate it's one they own. pk.domain==0 falls back to the auto per-client domain
     * (cl->domain, 0 unless ORKD_PER_CLIENT_DOMAINS). The weight lands here; its RUNs inherit it via w->domain. */
    int pdom = (int)pk.domain;
    if (pdom > 0){
        if (pdom >= ORKD_NDOM || !(cl->owned_dom & (1ull << pdom))){ free(wbuf); orkd_send_error(cl->fd, tag, ORKD_EBADH, "domain not owned by client"); return 0; }
    } else pdom = cl->domain;
    ork_npu_set_pack_domain(npu, pdom);
    ork_w *w = (pk.dtype == ORKD_DT_F16) ? ork_f16_mm_pack(npu, (int)pk.K, (int)pk.N, (const ork_f16 *)wbuf)
             : (pk.dtype == ORKD_DT_I4)  ? ork_i4_mm_pack(npu, (int)pk.K, (int)pk.N, wbuf)
                                         : ork_i8_mm_pack(npu, (int)pk.K, (int)pk.N, wbuf);
    ork_npu_set_pack_domain(npu, 0);            /* restore default for any non-client-scoped pack */
    free(wbuf);
    struct orkd_handle hh; memset(&hh, 0, sizeof hh);
    if (w && cl->nw < ORKD_MAX_WEIGHTS){ hh.id = ++cl->next_wid; hh.rc = 0; cl->wt[cl->nw++] = (struct cweight){ hh.id, w, (int)pk.K, (int)pk.N, (int)pk.dtype }; }
    else { if (w) ork_mm_free(npu, w); hh.rc = -1; }
    orkd_send_msg(cl->fd, ORKD_PACK_OK, tag, &hh, sizeof hh);
    return 0;
}
int orkd_handle_run(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_run rq;
    if (orkd_readn(cl->fd, &rq, sizeof rq) <= 0) return -1;
    struct cweight *cw = NULL;
    for (int i = 0; i < cl->nw; i++) if (cl->wt[i].id == rq.weight_id){ cw = &cl->wt[i]; break; }
    if (rq.abytes > ORKD_MAX_BYTES){ orkd_drain(cl->fd, rq.abytes); orkd_send_error(cl->fd, tag, ORKD_EPROTO, "bad A size"); return 0; }
    int8_t *A = malloc(rq.abytes ? rq.abytes : 1);
    if (!A){ orkd_drain(cl->fd, rq.abytes); orkd_send_error(cl->fd, tag, ORKD_EOOM, "A alloc"); return 0; }
    if (rq.abytes && orkd_readn(cl->fd, A, rq.abytes) <= 0){ free(A); return -1; }
    if (!cw || rq.abytes != (uint32_t)((size_t)rq.M * cw->K * orkd_esz_a(cw->dtype))){ free(A); orkd_send_error(cl->fd, tag, cw ? ORKD_EPROTO : ORKD_EBADH, cw ? "A size mismatch" : "unknown weight"); return 0; }
    int32_t *C = malloc((size_t)rq.M * cw->N * 4);
    struct work *w = C ? orkd_wk_alloc() : NULL;
    if (!C || !w){ free(A); free(C); orkd_send_error(cl->fd, tag, ORKD_EOOM, "queue full"); return 0; }
    w->fd = cl->fd; w->type = ORKD_RUN; w->tag = tag; w->prio = rq.flags; w->weight_id = rq.weight_id;
    /* v2: the client stamps the op's domain (== where it packed this weight); honor it, else fall back to the
     * weight's resident domain. The scheduler dom_activates w->domain (zero-copy swap) before dispatch. */
    w->domain = rq.domain ? (int)rq.domain : ork_w_domain(cw->w);
    w->M = (int)rq.M; w->K = cw->K; w->N = cw->N; w->dtype = cw->dtype; w->A = A; w->C = C;   /* enqueued; the scheduler dispatches it */
    return 0;
}
int orkd_handle_free(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_handle req;
    if (orkd_readn(cl->fd, &req, sizeof req) <= 0) return -1;
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.id = req.id; hh.rc = -1;
    for (int i = 0; i < cl->nw; i++) if (cl->wt[i].id == req.id){ ork_mm_free(npu, cl->wt[i].w); cl->wt[i] = cl->wt[--cl->nw]; hh.rc = 0; break; }
    orkd_send_msg(cl->fd, ORKD_PACK_OK, tag, &hh, sizeof hh);   /* PACK_OK = generic handle-op ack */
    return 0;
}
/* Stateless SDP activation op (silu/gelu/ewmul/add). Run INLINE (not queued): one-shot + tiny (M=8,N=64 geometry)
 * and the daemon is single-threaded, so it's still serialized on the single-stream NPU with any queued matmul
 * quanta. nin input payloads follow the struct (concatenated); reply = orkd_handle + the M*N*out_esz output. */
int orkd_handle_sdp(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_sdp sp;
    if (orkd_readn(cl->fd, &sp, sizeof sp) <= 0) return -1;
    size_t half = (size_t)sp.M * sp.N * sp.in_esz, inb = sp.inbytes, outb = (size_t)sp.M * sp.N * sp.out_esz;
    if (!sp.M || !sp.N || inb > ORKD_MAX_BYTES || inb != half * (sp.nin ? sp.nin : 1)){ orkd_drain(cl->fd, sp.inbytes); orkd_send_error(cl->fd, tag, ORKD_EPROTO, "bad sdp payload"); return 0; }
    uint8_t *in = malloc(inb ? inb : 1), *out = malloc(outb ? outb : 1);
    if (!in || !out){ free(in); free(out); orkd_drain(cl->fd, sp.inbytes); orkd_send_error(cl->fd, tag, ORKD_EOOM, "sdp alloc"); return 0; }
    if (inb && orkd_readn(cl->fd, in, inb) <= 0){ free(in); free(out); return -1; }
    const uint8_t *a = in, *b = in + half;   /* binary: second operand is the second half */
    int rc;
    switch (sp.op){
        case ORKD_SDP_SILU_I8:   rc = ork_i8_npu_silu (npu, (const signed char *)a, sp.M, sp.N, sp.in_scale, sp.out_scale, (signed char *)out, NULL); break;
        case ORKD_SDP_GELU_I8:   rc = ork_i8_npu_gelu (npu, (const signed char *)a, sp.M, sp.N, sp.in_scale, sp.out_scale, (signed char *)out, NULL); break;
        case ORKD_SDP_EWMUL_I8:  rc = ork_i8_npu_ewmul(npu, (const int8_t *)a, (const int8_t *)b, sp.M, sp.N, sp.mult, sp.shift, (int8_t *)out, NULL); break;
        case ORKD_SDP_EWMUL_F16: rc = ork_f16_npu_ewmul(npu, (const ork_f16 *)a, (const ork_f16 *)b, sp.M, sp.N, (ork_f16 *)out, NULL); break;
        case ORKD_SDP_ADD_I8:    rc = ork_i8_npu_add  (npu, (const signed char *)a, (const signed char *)b, sp.M, sp.N, sp.a_scale, sp.b_scale, sp.out_scale, (signed char *)out, NULL); break;
        case ORKD_SDP_ADD_F16:   rc = ork_f16_npu_add (npu, (const ork_f16 *)a, (const ork_f16 *)b, sp.M, sp.N, (ork_f16 *)out, NULL); break;
        case ORKD_SDP_RSQRT_I8:  rc = ork_i8_npu_rsqrt(npu, (const signed char *)a, sp.M, sp.N, sp.in_scale, sp.out_scale, (signed char *)out, NULL); break;
        case ORKD_SDP_EXP_I8:    rc = ork_i8_npu_exp  (npu, (const signed char *)a, sp.M, sp.N, sp.in_scale, sp.out_scale, (signed char *)out, NULL); break;
        case ORKD_SDP_SILU_I16:  rc = ork_i16_npu_silu(npu, (const int16_t *)a, sp.M, sp.N, sp.in_scale, sp.out_scale, (int16_t *)out, NULL); break;
        case ORKD_SDP_GELU_I16:  rc = ork_i16_npu_gelu(npu, (const int16_t *)a, sp.M, sp.N, sp.in_scale, sp.out_scale, (int16_t *)out, NULL); break;
        case ORKD_SDP_RSQRT_I16: rc = ork_i16_npu_rsqrt(npu, (const int16_t *)a, sp.M, sp.N, sp.in_scale, sp.out_scale, (int16_t *)out, NULL); break;
        case ORKD_SDP_EXP_I16:   rc = ork_i16_npu_exp (npu, (const int16_t *)a, sp.M, sp.N, sp.in_scale, sp.out_scale, (int16_t *)out, NULL); break;
        case ORKD_SDP_EWMUL_I16: rc = ork_i16_npu_ewmul(npu, (const int16_t *)a, (const int16_t *)b, sp.M, sp.N, sp.mult, sp.shift, (int16_t *)out, NULL); break;
        case ORKD_SDP_ADD_I16:   rc = ork_i16_npu_add (npu, (const int16_t *)a, (const int16_t *)b, sp.M, sp.N, sp.a_scale, sp.b_scale, sp.out_scale, (int16_t *)out, NULL); break;
        default: rc = -100;
    }
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.rc = rc;
    int payload = (rc == 0);
    struct orkd_hdr rh = { ORKD_SDP_OK, (uint32_t)(sizeof hh + (payload ? outb : 0)), tag };
    (void)!(orkd_writen(cl->fd, &rh, sizeof rh) || orkd_writen(cl->fd, &hh, sizeof hh) || (payload && orkd_writen(cl->fd, out, outb)));
    free(in); free(out);
    return 0;
}
/* Fused int8 matmul chain: S resident weights run as one PC-chained submit (ork_i8_mm_run_chain). Each task's
 * weight is resolved by id in THIS client's table; A payloads arrive concatenated (task order), C payloads are
 * returned concatenated. Run INLINE (a chain is one bounded submit; the single-threaded daemon serializes it). */
int orkd_handle_chain(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_chain_hdr ch;
    if (orkd_readn(cl->fd, &ch, sizeof ch) <= 0) return -1;
    int S = (int)ch.S;
    size_t tsz = (size_t)(S > 0 ? S : 0) * sizeof(struct orkd_chain_task);
    if (S < 1 || S > ORKD_CHAIN_MAX || ch.abytes_total > ORKD_MAX_BYTES){ orkd_drain(cl->fd, tsz + ch.abytes_total); orkd_send_error(cl->fd, tag, ORKD_EPROTO, "bad chain"); return 0; }
    struct orkd_chain_task *ts = malloc(tsz);
    if (!ts){ orkd_drain(cl->fd, tsz + ch.abytes_total); orkd_send_error(cl->fd, tag, ORKD_EOOM, "chain tasks"); return 0; }
    if (orkd_readn(cl->fd, ts, tsz) <= 0){ free(ts); return -1; }
    uint8_t *ablob = malloc(ch.abytes_total ? ch.abytes_total : 1);
    if (!ablob){ free(ts); orkd_drain(cl->fd, ch.abytes_total); orkd_send_error(cl->fd, tag, ORKD_EOOM, "chain A"); return 0; }
    if (ch.abytes_total && orkd_readn(cl->fd, ablob, ch.abytes_total) <= 0){ free(ts); free(ablob); return -1; }
    /* build the daemon-side task array: resolve each weight, point A into ablob, allocate a C per task */
    ork_mm_task_i8 *mt = calloc((size_t)S, sizeof *mt);
    int32_t **Cs = calloc((size_t)S, sizeof *Cs);
    size_t *cb = calloc((size_t)S, sizeof *cb);   /* per-task C bytes = M*N*4 */
    size_t aoff = 0, ctot = 0; int ok = (mt && Cs && cb), rc = 0;
    for (int i = 0; ok && i < S; i++){
        struct cweight *cw = NULL;
        for (int j = 0; j < cl->nw; j++) if (cl->wt[j].id == ts[i].weight_id){ cw = &cl->wt[j]; break; }
        if (!cw || cw->dtype != ORKD_DT_I8 || ts[i].abytes != (uint32_t)((size_t)ts[i].M * cw->K) || aoff + ts[i].abytes > ch.abytes_total){ ok = 0; break; }
        cb[i] = (size_t)ts[i].M * cw->N * 4;
        Cs[i] = malloc(cb[i]);
        if (!Cs[i]){ ok = 0; break; }
        mt[i].w = cw->w; mt[i].M = (int)ts[i].M; mt[i].A = (const int8_t *)(ablob + aoff); mt[i].C = Cs[i];
        aoff += ts[i].abytes; ctot += cb[i];
    }
    if (ok) rc = ork_i8_mm_run_chain(npu, S, mt);
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.rc = ok ? rc : -1;
    int payload = (ok && rc == 0);
    struct orkd_hdr rh = { ORKD_CHAIN_OK, (uint32_t)(sizeof hh + (payload ? ctot : 0)), tag };
    int werr = orkd_writen(cl->fd, &rh, sizeof rh) || orkd_writen(cl->fd, &hh, sizeof hh);
    if (!werr && payload) for (int i = 0; i < S && !werr; i++) werr = orkd_writen(cl->fd, Cs[i], cb[i]);
    if (Cs) for (int i = 0; i < S; i++) free(Cs[i]);
    free(mt); free(Cs); free(cb); free(ts); free(ablob);
    return werr ? -1 : 0;
}
/* Heterogeneous op-sequence submit: reconstruct an ork_seq_op[] (resident weights by id + received A/B buffers
 * + allocated C) and run ork_submit_seq, which batches maximal runs of doorbell-eligible ops onto the spine,
 * breaks the chain to the SW path at each op that can't be chained, then resumes. Runs INLINE (one bounded
 * sequence; the single-threaded daemon serializes it on the single-stream NPU). C's returned concatenated. */
int orkd_handle_seq(struct client *cl, ork_npu *npu, uint64_t tag){
    struct orkd_seq_hdr sh;
    if (orkd_readn(cl->fd, &sh, sizeof sh) <= 0) return -1;
    int n = (int)sh.n;
    size_t osz = (size_t)(n > 0 ? n : 0) * sizeof(struct orkd_seq_op);
    if (n < 1 || n > ORKD_SEQ_MAX || sh.in_total > ORKD_MAX_BYTES){ orkd_drain(cl->fd, osz + sh.in_total); orkd_send_error(cl->fd, tag, ORKD_EPROTO, "bad seq"); return 0; }
    struct orkd_seq_op *ops = malloc(osz);
    if (!ops){ orkd_drain(cl->fd, osz + sh.in_total); orkd_send_error(cl->fd, tag, ORKD_EOOM, "seq ops"); return 0; }
    if (orkd_readn(cl->fd, ops, osz) <= 0){ free(ops); return -1; }
    uint8_t *inblob = malloc(sh.in_total ? sh.in_total : 1);
    if (!inblob){ free(ops); orkd_drain(cl->fd, sh.in_total); orkd_send_error(cl->fd, tag, ORKD_EOOM, "seq in"); return 0; }
    if (sh.in_total && orkd_readn(cl->fd, inblob, sh.in_total) <= 0){ free(ops); free(inblob); return -1; }
    ork_seq_op *seq = calloc((size_t)n, sizeof *seq);
    void **Cs = calloc((size_t)n, sizeof *Cs);
    size_t inoff = 0, ctot = 0; int ok = (seq && Cs);
    for (int i = 0; ok && i < n; i++){
        struct orkd_seq_op *o = &ops[i];
        size_t need = (o->a_src ? 0 : o->abytes) + (o->b_src ? 0 : o->bbytes);   /* A2: referenced inputs aren't in the uploaded blob */
        if ((size_t)inoff + need > sh.in_total){ ok = 0; break; }
        seq[i].kind = (ork_seq_kind)o->kind; seq[i].M = (int)o->M; seq[i].N = (int)o->N;
        seq[i].in_scale = o->in_scale; seq[i].out_scale = o->out_scale; seq[i].b_scale = o->b_scale; seq[i].mult = o->mult; seq[i].shift = o->shift; seq[i].group = (int)o->group;
        if (o->weight_id){   /* matmul op: resolve the resident weight in this client's table */
            struct cweight *cw = NULL;
            for (int j = 0; j < cl->nw; j++) if (cl->wt[j].id == o->weight_id){ cw = &cl->wt[j]; break; }
            if (!cw){ ok = 0; break; }
            seq[i].w = cw->w;
        }
        /* A2: a_src/b_src = j+1 -> input is op j's resident output buffer (Cs[j]); else uploaded in the blob. */
        if (o->a_src){ if (o->a_src > i){ ok = 0; break; } seq[i].A = Cs[o->a_src - 1]; }
        else { seq[i].A = inblob + inoff; inoff += o->abytes; }
        if (o->b_src){ if (o->b_src > i){ ok = 0; break; } seq[i].B = Cs[o->b_src - 1]; }
        else { seq[i].B = o->bbytes ? inblob + inoff : NULL; inoff += o->bbytes; }
        Cs[i] = malloc(o->cbytes ? o->cbytes : 1);
        if (!Cs[i]){ ok = 0; break; }
        seq[i].C = Cs[i]; if (!o->c_keep) ctot += o->cbytes;   /* c_keep => stays resident, not shipped back */
    }
    int rc = ok ? ork_submit_seq(npu, seq, n) : -2;
    { const char *dbg = getenv("ORKD_SEQ_DEBUG");   /* dump each op's A-input + C-output stats to a DETERMINISTIC file
        * (ORKD_SEQ_DEBUG=path, or =1 -> /tmp/orkd-seqdbg.log). Own fd + fsync per seq so it survives a later NPU wedge
        * and never depends on stderr/orkd.log's runtime-dir location. This is the reliable observability for the
        * orkd SW-run() reduce bug (registry: resident int8 softmax seq assembly). */
      if (dbg){
        const char *path = (dbg[0] && strcmp(dbg,"1")) ? dbg : "/tmp/orkd-seqdbg.log";
        int dfd = open(path, O_CREAT|O_WRONLY|O_APPEND, 0644);
        if (dfd >= 0){ char lb[512]; int ln;
            ln = snprintf(lb, sizeof lb, "[seqdbg] n=%d ok=%d rc=%d in_total=%u\n", n, ok, rc, sh.in_total); if(ln>0) (void)!write(dfd, lb, (size_t)ln);
            for (int i = 0; ok && i < n; i++){
                struct orkd_seq_op *o = &ops[i];
                size_t asz = o->a_src ? (o->a_src>=1 && o->a_src<=i ? ops[o->a_src-1].cbytes : 0) : o->abytes;
                long a8 = 0; const int8_t *ain = (const int8_t*)seq[i].A;
                if (ain) for (size_t b = 0; b < asz; b++) a8 += ain[b];
                long c8 = 0, c32 = 0; const int8_t *c8p = (const int8_t*)Cs[i]; const int32_t *c32p = (const int32_t*)Cs[i];
                for (size_t b = 0; b < o->cbytes; b++) c8 += c8p[b];
                for (size_t q = 0; q < o->cbytes/4; q++) c32 += c32p[q];
                ln = snprintf(lb, sizeof lb, "[seqdbg] op%d kind=%u M=%d N=%d a_src=%d c_keep=%u asz=%zu cb=%u | Asum(i8)=%ld Csum(i8)=%ld Csum(i32)=%ld C[i32:0..3]=%d,%d,%d,%d\n",
                        i, o->kind, seq[i].M, seq[i].N, o->a_src, o->c_keep, asz, o->cbytes, a8, c8, c32,
                        c32p[0], o->cbytes>=8?c32p[1]:0, o->cbytes>=12?c32p[2]:0, o->cbytes>=16?c32p[3]:0);
                if(ln>0) (void)!write(dfd, lb, (size_t)ln);
            }
            fsync(dfd); close(dfd);
        }
      }
    }
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.rc = rc;
    int payload = (rc == 0);
    struct orkd_hdr rh = { ORKD_SEQ_OK, (uint32_t)(sizeof hh + (payload ? ctot : 0)), tag };
    int werr = orkd_writen(cl->fd, &rh, sizeof rh) || orkd_writen(cl->fd, &hh, sizeof hh);
    if (!werr && payload) for (int i = 0; i < n && !werr; i++) if (!ops[i].c_keep) werr = orkd_writen(cl->fd, Cs[i], ops[i].cbytes);   /* A2: skip resident intermediates */
    if (Cs) for (int i = 0; i < n; i++) free(Cs[i]);
    free(seq); free(Cs); free(ops); free(inblob);
    return werr ? -1 : 0;
}
/* #2b-2 step 3b: FULL zero-copy RUN — A read AND C written by reference. Client shares A+C dma-bufs (two fds
 * via SCM_RIGHTS); orkd imports both, runs with C written IN PLACE (ORK_ZC_OUT, set at startup → dma_find(C)
 * hit) so NO C byte-transfer either way. Forced SINGLE-CORE: output zero-copy is unsafe under concurrent
 * multi-core (the per-core coherency bsyncs don't serialize with the NPU writes — AGENTS.md ZC-OUT caveat).
 * Client invalidates C (DMA_BUF_SYNC START|READ) before reading. Reply carries no C payload. */
int orkd_handle_run_zc2(struct client *cl, ork_npu *npu, int a_fd, int c_fd, uint64_t tag){
    struct orkd_run rq;
    if (orkd_readn(cl->fd, &rq, sizeof rq) <= 0){ if (a_fd >= 0) close(a_fd); if (c_fd >= 0) close(c_fd); return -1; }
    struct cweight *cw = NULL;
    for (int i = 0; i < cl->nw; i++) if (cl->wt[i].id == rq.weight_id){ cw = &cl->wt[i]; break; }
    if (!cw || a_fd < 0 || c_fd < 0){ if (a_fd >= 0) close(a_fd); if (c_fd >= 0) close(c_fd); orkd_send_error(cl->fd, tag, cw ? ORKD_EPROTO : ORKD_EBADH, cw ? "need A+C fds" : "unknown weight"); return 0; }
    size_t alen = (size_t)rq.M * cw->K * orkd_esz_a(cw->dtype), cn = (size_t)rq.M * cw->N;
    void *A = ork_dma_import_fd(npu, a_fd, alen);
    if (!A){ close(a_fd); close(c_fd); orkd_send_error(cl->fd, tag, ORKD_EOOM, "import A"); return 0; }
    void *C = ork_dma_import_fd(npu, c_fd, cn * 4);
    if (!C){ ork_dma_free(npu, A); close(c_fd); orkd_send_error(cl->fd, tag, ORKD_EOOM, "import C"); return 0; }
    struct work *w = orkd_wk_alloc();
    if (!w){ ork_dma_free(npu, A); ork_dma_free(npu, C); orkd_send_error(cl->fd, tag, ORKD_EOOM, "queue full"); return 0; }
    w->fd = cl->fd; w->type = ORKD_RUN_ZC2; w->tag = tag; w->prio = rq.flags; w->weight_id = rq.weight_id;
    w->domain = rq.domain ? (int)rq.domain : ork_w_domain(cw->w);   /* v2: honor the client-stamped op domain */
    w->M = (int)rq.M; w->K = cw->K; w->N = cw->N; w->dtype = cw->dtype;
    w->A = (int8_t *)A; w->A_imp = A; w->C = (int32_t *)C; w->C_imp = C;   /* dispatch: A read + C written in place, single-core */
    return 0;
}
/* #2b-2 step 3: ZERO-COPY RUN (input A by reference). The client shares A as a dma-buf fd (SCM_RIGHTS);
 * orkd PRIME-imports it into the NPU's IOMMU domain and ork_i8_mm_run reads A IN PLACE (dma_find hit =
 * validated input zero-copy) — no A byte-transfer over the socket. C is still returned over the socket here
 * (output zero-copy is the next sub-step: needs ORK_ZC_OUT + a cross-process invalidate). */
int orkd_handle_run_zc(struct client *cl, ork_npu *npu, int a_fd, uint64_t tag){
    struct orkd_run rq;
    if (orkd_readn(cl->fd, &rq, sizeof rq) <= 0){ if (a_fd >= 0) close(a_fd); return -1; }
    struct cweight *cw = NULL;
    for (int i = 0; i < cl->nw; i++) if (cl->wt[i].id == rq.weight_id){ cw = &cl->wt[i]; break; }
    if (!cw || a_fd < 0){ if (a_fd >= 0) close(a_fd); orkd_send_error(cl->fd, tag, cw ? ORKD_EPROTO : ORKD_EBADH, cw ? "no A fd" : "unknown weight"); return 0; }
    size_t alen = (size_t)rq.M * cw->K * orkd_esz_a(cw->dtype);
    void *A = ork_dma_import_fd(npu, a_fd, alen);    /* import A into the NPU domain (takes a_fd ownership) */
    if (!A){ close(a_fd); orkd_send_error(cl->fd, tag, ORKD_EOOM, "import A"); return 0; }
    int32_t *C = malloc((size_t)rq.M * cw->N * 4);
    struct work *w = C ? orkd_wk_alloc() : NULL;
    if (!C || !w){ ork_dma_free(npu, A); free(C); orkd_send_error(cl->fd, tag, ORKD_EOOM, "queue full"); return 0; }
    w->fd = cl->fd; w->type = ORKD_RUN_ZC; w->tag = tag; w->prio = rq.flags; w->weight_id = rq.weight_id;
    w->domain = rq.domain ? (int)rq.domain : ork_w_domain(cw->w);   /* v2: honor the client-stamped op domain */
    w->M = (int)rq.M; w->K = cw->K; w->N = cw->N; w->dtype = cw->dtype; w->A = (int8_t *)A; w->A_imp = A; w->C = C;   /* A zero-copy; C over socket */
    return 0;
}
/* #2b-2 step 1: prove cross-process dma-buf sharing. The client sent a dma-heap fd (SCM_RIGHTS); mmap it here
 * and confirm orkd sees the same bytes (fnv match). This validates the fd-passing + shared-memory plumbing;
 * PRIME-import into the NPU's IOMMU domain (zero-copy submit) is step 2 (needs a library fd hook). */
int orkd_handle_dmabuf(struct client *cl, ork_npu *npu, int dfd, uint64_t tag){
    struct orkd_dmabuf db;
    if (orkd_readn(cl->fd, &db, sizeof db) <= 0){ if (dfd >= 0) close(dfd); return -1; }
    struct orkd_dmabuf out; memset(&out, 0, sizeof out); out.size = db.size; out.rc = -1; out.prime_ok = 0;
    if (dfd >= 0 && db.size && db.size <= ORKD_MAX_BYTES){
        void *p = ork_dma_import_fd(npu, dfd, (size_t)db.size);   /* PRIME-import into the NPU's IOMMU domain */
        if (p){
            out.prime_ok = 1;                                    /* NPU-addressable: the real zero-copy path */
            out.checksum = orkd_fnv(p, (size_t)db.size);
            out.rc = (out.checksum == db.checksum) ? 0 : -1;
            ork_dma_free(npu, p);                                /* closes dfd (import took ownership) */
            dfd = -1;
        } else {                                                 /* import failed -> plain shared mmap still proves fd passing */
            void *m = mmap(NULL, db.size, PROT_READ, MAP_SHARED, dfd, 0);
            if (m != MAP_FAILED){ out.checksum = orkd_fnv(m, db.size); out.rc = (out.checksum == db.checksum) ? 0 : -1; munmap(m, db.size); }
        }
    }
    if (dfd >= 0) close(dfd);
    orkd_send_msg(cl->fd, ORKD_DMABUF_OK, tag, &out, sizeof out);
    return 0;
}
/* CLIENT-MANAGED IOMMU DOMAINS: hand out / return pool domains so a client can pack its weights into its own
 * isolated ~4 GiB IOVA window (rk_iommu v2's cap is per-domain). orkd coordinates the pool (g_dom_inuse) so no
 * two clients collide; the client tracks the ids it holds and packs into them via orkd_pack.domain. All of a
 * client's domains are released automatically on drop (see the drop path in the main loop). */
int orkd_handle_dom_req(struct client *cl, uint64_t tag){
    int d = orkd_dom_alloc_explicit();
    struct orkd_handle hh; memset(&hh, 0, sizeof hh);
    if (d > 0){ cl->owned_dom |= (1ull << d); hh.id = (uint64_t)d; hh.rc = 0; }
    else hh.rc = -1;                                     /* pool exhausted */
    orkd_send_msg(cl->fd, ORKD_DOM_OK, tag, &hh, sizeof hh);
    return 0;
}
int orkd_handle_dom_rel(struct client *cl, uint64_t tag){
    struct orkd_handle rq;
    if (orkd_readn(cl->fd, &rq, sizeof rq) <= 0) return -1;
    int d = (int)rq.id;
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.id = rq.id;
    if (d >= 1 && d < ORKD_NDOM && (cl->owned_dom & (1ull << d))){ cl->owned_dom &= ~(1ull << d); orkd_dom_release(d); hh.rc = 0; }
    else hh.rc = -1;                                     /* not a domain this client owns */
    orkd_send_msg(cl->fd, ORKD_DOM_OK, tag, &hh, sizeof hh);
    return 0;
}
/* A-ring: attach a client's shared ring. The region fd arrived via SCM_RIGHTS on the ORKD_RING_SETUP header;
 * mmap it, validate the layout (magic + geometry the client and daemon must agree on), and keep the mapping.
 * From here ring_service() busy-polls this ring for requests, bypassing the socket for the hot submit path. */
int orkd_handle_ring_setup(struct client *cl, int rfd, uint64_t tag){
    struct orkd_ring_setup rs;
    if (orkd_readn(cl->fd, &rs, sizeof rs) <= 0){ if (rfd >= 0) close(rfd); return -1; }
    struct orkd_handle hh; memset(&hh, 0, sizeof hh); hh.rc = -1;
    if (rfd >= 0 && rs.bytes == ORKD_RING_BYTES && !cl->ring){
        void *m = mmap(NULL, ORKD_RING_BYTES, PROT_READ|PROT_WRITE, MAP_SHARED, rfd, 0);
        if (m != MAP_FAILED){
            struct orkd_ring *r = (struct orkd_ring *)m;
            if (r->magic == ORKD_RING_MAGIC && r->nslots == ORKD_RING_SLOTS && r->slot_data == ORKD_RING_SLOT_DATA){
                cl->ring = r; cl->ring_fd = rfd; cl->ring_tail = 0; rfd = -1; hh.rc = 0;
            } else munmap(m, ORKD_RING_BYTES);
        }
    }
    if (rfd >= 0) close(rfd);
    orkd_send_msg(cl->fd, ORKD_RING_OK, tag, &hh, sizeof hh);
    return 0;
}
