/* npu/internal.h — the PRIVATE ABI shared by src/npu.c and the src/npu/<mod>.c precision modules.
 *
 * Round 1 of the modularization (MODULARIZE_PLAN.md) splits the npu.c monolith along the precision axis
 * into src/npu/{sdp,f16,i16,i4,ssm}.c + src/npu/i8/<mod>.c. Everything those translation units need in common
 * — the context/weight/buffer types, the dtype markers, and the hot helpers that must stay inlinable
 * across the split — lives here. NOT a public header: include/ork_npu.h is the API; this is internals,
 * and nothing outside src/ may include it.
 *
 * Include it the same way from every depth (-Isrc makes this resolve from src/npu.c, src/npu/i8/ and
 * src/npu/i8/pack.c alike):   #include "npu/internal.h"
 */
#ifndef ORK_NPU_INTERNAL_H
#define ORK_NPU_INTERNAL_H

#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include "rknpu_ioctl.h"
#include "orkd_client.h"   /* orkd_conn — struct ork_npu carries one in client mode */
#include "ork_npu.h"
#include "ork_slice.h"     /* ork_slice_caps / ork_tile — the sliced-doorbell primitives */
#include "soc.h"

typedef ork_f16 f16;
enum { DT_F16=0, DT_I8=1, DT_I4=2 };

#define ORK_MAXCORE 4   /* RK3576=2, RK3588=3; headroom for future parts. Actual = soc->cores. */

struct buf { uint32_t handle; uint64_t dma, obj; void *cpu; size_t size; int heap_fd; int domain; };  /* heap_fd: for zero-copy IMPORTED bufs (ork_dma_import / bimport) the dma-buf fd to close on destroy; 0 for ordinary MEM_CREATE-allocated bufs. domain: the iommu domain this buffer's IOVA was reserved in (for the IOVA wedge-guard accounting; set by bcreate/bimport, released by bdestroy). */
struct ork_pw { struct ork_npu *c; int id; };   /* persistent NPU-pool worker arg */
/* Parked per-domain copy of the submit-touched scratch (see ork_npu.dom_save). Mirrors exactly the
 * ork_npu fields that name DMA buffers a submit references + their warm/size bookkeeping. cres is host
 * RAM (domain-agnostic) so it is NOT parked here. */
struct ork_dom_scratch {
    int used;
    struct buf regcmd, task, Af, Cc; size_t ccsz; int warmed;
    struct buf mrc[ORK_MAXCORE], mtk[ORK_MAXCORE], maf[ORK_MAXCORE], mcc[ORK_MAXCORE], mtk_all;
    size_t mccsz[ORK_MAXCORE]; int mwarm[ORK_MAXCORE]; int mc_alloc;
};
struct ork_npu { int fd; const struct ork_soc *soc; struct buf regcmd, task, Af, Cc; size_t ccsz; void *cres; size_t cressz; int warmed, last_dt; int scratch_import; /* 1 once a native-int4 weight is bimported: route run scratch via orki_bimport (bcreate EINVALs in a bimport-filled domain). Set by ork_mm_load_i4_import; int8/fp16 never set it. */ int last_chain; int core_budget; int layer_warmed; uint64_t layer_warm[7];  /* ork_mm_layer_i8 per-NPU doorbell warm cache (keyed on the 7 weight ptrs) */ orkd_conn *daemon;  /* Path B: non-NULL => client mode, ork_mm_* route through orkd instead of a local NPU (fd=-1) */
    /* fused-chain PER-CORE scratch (increment 2: concurrent round-robin — one independent buffer set per core so
     * chains dispatched to different cores never share DRAM). chain_rc = P-program regcmd, chain_tk = P-task array,
     * chain_lrc = LUT-load regcmd (DRAM source), chain_lsc = LUT SDP scratch. Lazily allocated per core on first use. */
    struct buf chain_rc[ORK_MAXCORE], chain_tk[ORK_MAXCORE], chain_lrc[ORK_MAXCORE], chain_lsc[ORK_MAXCORE];
    const int16_t *chain_lut_p[ORK_MAXCORE];   /* per-core: which LUT is patched into chain_lrc[core] (rebuild that core's lrc when it changes) */
    int chain_lut_devloaded[ORK_MAXCORE];      /* per-core: chain_lut_devloaded[core]=1 => that core's physically-per-core SDP LUT SRAM still holds its LUT (invalidated for a core on reset). ORK_CHAIN_LUT_STICKY skips the reload when still resident */
    int chain_core;     /* per-dispatch target core (rr scheduler sets it); default 0. Public setter clamps to 0 until concurrent dispatch is wired; per-core caches make any core cache-correct */
    int chain_task_P[ORK_MAXCORE];      /* per-core task-config cache: shape (P) whose task array currently sits in chain_tk[core] */
    int chain_task_built[ORK_MAXCORE];  /* per-core: 1 => chain_tk[core] holds the current P-program task array (P=chain_task_P[core]); a LUT-load on that core memsets chain_tk[core] and clears this */
    /* multi-core (ORK_NPU_MC): per-core regcmd/task/feature/output so cores submit concurrently */
    struct buf mrc[ORK_MAXCORE], mtk[ORK_MAXCORE], maf[ORK_MAXCORE], mcc[ORK_MAXCORE], mtk_all;
    size_t mccsz[ORK_MAXCORE]; int mwarm[ORK_MAXCORE]; int mc_alloc;
    /* persistent SDP element-wise/activation-op scratch (a/b/out), REUSED across ewmul/add calls to avoid
     * per-op MEM_CREATE/MEM_DESTROY churn (that churn dominates standalone-op latency AND fragments the
     * IOVA window -> wedge). Grows monotonically to the largest cube seen; freed at teardown. */
    struct buf ppu_a, ppu_b, ppu_o; size_t ppu_sz;
    /* PER-DOMAIN SCRATCH (multi-domain residence): a submit runs in ONE iommu_domain_id, so EVERY buffer
     * it touches (regcmd, task, activation Af, output Cc, and the per-core multi-core scratch) must live in
     * the same domain as the weight. The fields above are the ACTIVE working set; dom_active is which
     * domain they currently belong to. dom_save[d] parks a domain's working set when switching away, so
     * each domain keeps its own (cheap, MB-scale) scratch resident — no realloc on every weight. */
    int dom_active; int dom_cap;   /* dom_cap = allocated length of dom_anchor[] / dom_save[]; grown on demand by dom_reserve, NO fixed cap (domain count = whatever the auto-sizer / ork_npu_domain_alloc drives) */
    int dom_dirty;   /* #54: a genuine doorbell MISS left an unreaped (dropped) job in dom_active — its completion IRQ never fired, so the kernel's per-job interrupt_count for this domain stays >0 forever. A stuck job makes the NEXT iommu-domain switch time out ("switch iommu domain time out") -> cascade. dom_activate flushes it (ACT_RESET, which aborts+clears stuck jobs, while STILL attached to this domain) before switching away. Set by the int4 doorbell reset-free-resubmit path; single-domain never switches, so it never acts on the flag (matches streaming's safety). */
    /* ORK_DOM_PROFILE: dom_activate cost telemetry (the "domain-swap window"). Steady = pure scratch
     * pointer-swap memcpy; first = one-time per-domain first-touch orki_bcreate(regcmd/task/Af). */
    uint64_t dom_sw_n, dom_sw_first_n; double dom_sw_us, dom_sw_first_us, dom_sw_max_us;
    int dom_next;   /* direct-mode domain allocator counter (ork_npu_domain_alloc hands out 1,2,3,…); Path B asks the daemon instead */
    /* Per-domain native "anchor": one small NATIVE bcreate per non-0 domain, allocated BEFORE any dma-buf
     * import is mapped into that domain. The kernel rknpu driver sets up a domain's IOVA allocator / page
     * table lazily on its FIRST buffer, and that path misbehaves when the first buffer is an IMPORTED
     * dma-buf (SG-list) — the import's pages land on wrong/aliased IOVAs and the NPU reads garbage for some
     * weight tiles (non-deterministic dropped-K corruption, affects single- AND multi-core; NATIVE weights
     * are immune). A native alloc first establishes the domain correctly. Kept alive for the ctx lifetime
     * so the domain stays anchored; freed at teardown. See ork_dom_prime(). */
    struct buf *dom_anchor;             /* [dom_cap]; per-domain native anchor, grown by dom_reserve */
    struct ork_dom_scratch *dom_save;   /* [dom_cap]; NULL until multi-domain is first used — its NULL-ness is the single-vs-multi-domain signal the run paths key on */
    /* persistent worker pool: spawned once, signalled per matmul (cuts per-matmul create/join) */
    pthread_t pth[ORK_MAXCORE]; struct ork_pw pwa[ORK_MAXCORE]; int pool_n;
    pthread_mutex_t pmu; pthread_cond_t pgo, pdn; void *pjob; int pjob_nc, pgen, pdone, pstop;
    void *(*pjob_fn)(void *); size_t pjob_stride;   /* generalized pool dispatch: per-core worker + arg stride */
    pthread_barrier_t b_ioctl; int mc_submit_rc; int mc_error;
    int f16_force_blocking;   /* colsplit worker: force BLOCKING submit even under ORK_F16_SENTINEL (nonblock-detect + blocking-heal hybrid: attempt 0 nonblock/fast-detect, retries blocking so the kernel watchdog REAPS+clears the sticky slice-1 drop) */
    int last_async_cpu;   /* sched_getcpu() of the most recent async worker at entry (diagnostic/test: -1 = none) */
    /* zero-copy registry: caller-allocated NPU-coherent DMA buffers (ork_dma_alloc). When a matmul's
     * A/C live in one of these, the regcmd points at them directly — no host gather/writeout memcpy. */
    struct buf dma_tab[64]; int dma_n;
    /* global weight arena: a POOL of large DMA chunks (each under the ~4GB single-allocation cap), bump-
     * allocated across ALL packed weights. One weight's tiles always land contiguously in a single chunk
     * (flushed in one bsync_off). Collapses thousands of per-tile bcreates to a handful of chunks => fast
     * warmup, no IOVA-handle OOM. Also the on-disk form for persisted (pre-packed) weights. */
    struct buf wchunk[64]; int wchunk_n; size_t wchunk_off;
    /* int4 EXPERT IMPORT ARENA (#54): a per-domain pool of LARGE bimport'd dma-buf chunks, bump-allocated
     * across MANY experts so each expert's tiles are base+offset VIEWS into a shared chunk. One dma-buf import
     * PER EXPERT (~9k for the 35B MoE) crams ~2340 IOMMU mappings into a domain -> the NPU wedges mid-prefill;
     * a few large chunks/domain (same anchor+bimport+bimport-scratch alloc pattern, just coarser granularity)
     * keeps mappings tiny. Chunks persist for the ctx (MoE experts are resident, never evicted); freed once at
     * teardown. Views (heap_fd=0, own_bufs=NULL, owns=0) are skipped by ork_mm_free. [64] == ORK_IOVA_NDOM. */
    struct buf *i4arena; int i4arena_n, i4arena_cap;              /* every chunk ever allocated (teardown bdestroys each) */
    struct buf i4arena_cur[64]; size_t i4arena_off[64]; int i4arena_curi[64];   /* per-domain open chunk (working copy), write offset, and its index in i4arena[] (for fd-sealing on chunk switch) */
    /* PACK DOMAIN: per-ctx default domain for the NEXT pack/load (set by ork_npu_set_pack_domain for the
     * ggml-ork caller). Read once at each pack's entry to stamp w->domain; from there the weight carries
     * its own domain. Not a process-global — concurrent ctxs don't clobber each other. -1 => default. */
    int pack_domain;
    /* Standalone-activation LUT-op index-map cache: idx(in) for the SDP activation op depends only on the
     * (constant) index params, NOT on in/out scale, so calibrate ONCE per ctx (a ramp-LUT submit) and reuse
     * for any scale. silu_idx_ok=1 once filled; silu_idx[256] maps (uint8)input -> LUT index (-1 if saturated). */
    int silu_idx_ok; short silu_idx[256];
    /* int16 variant: with the gain-1 index params (0x4068 low16=0x1000) idx = in + 512 (integer, no LUT
     * interpolation -> bit-exact), usable for |in| < 512. silu_idx16[in+512] = measured LUT index (-1 if none). */
    int silu_idx16_ok; short silu_idx16[4096];
    /* B2 terminal-SDP: a tiny resident int8 witness weight (K=512,N=16, zeros). A seq group ending in an SDP op
     * gets this witness matmul appended as its terminal so the group HW-chains (the SDP rides the chain; the
     * witness's int32 sentinel gates completion). Lazy-packed once; freed at teardown. */
    ork_w *seq_witness;
    /* persistent SSM-scan scratch pool (ork_ssm_scan_f32): the 4*nh fp16 scratch weights + all CPU staging
     * buffers, REUSED across calls at the same (nc,nr,nh) shape to avoid per-call MEM_CREATE/MEM_DESTROY churn
     * — that churn (4*nh scratch alloc/free per call) dominated the in-model scan (~10x the warm cost). CS is
     * fixed (64) so buffer sizes depend only on nc/nr/nh. Realloc only when the shape changes; freed at teardown. */
    int ssm_nc, ssm_nr, ssm_nh, ssm_csz, ssm_nb; ork_w **ssm_pS,**ssm_pD,**ssm_pC,**ssm_pO;
    ork_f16 *ssm_aS,*ssm_bS,*ssm_aD,*ssm_bD,*ssm_aC,*ssm_bC,*ssm_aO,*ssm_bO;
    float *ssm_G,*ssm_Yd,*ssm_cs,*ssm_tmp,*ssm_state,*ssm_stp;
    double *ssm_Acs,*ssm_xbar,*ssm_Acl,*ssm_Aclp; ork_mm_task_f16 *ssm_tk;
    /* per-stage int8 path (ORK_SSM_I8_MASK): int8 scratch weights + quant/dequant temps, allocated only when
     * the mask selects at least one int8 stage. Sized generously (nh*CS*nc) to cover any single stage. */
    int ssm_i8; ork_w **ssm_pSi8; int8_t *ssm_ai8; int32_t *ssm_ci32; float *ssm_ascale,*ssm_bscale,*ssm_f32t; ork_mm_task_i8 *ssm_tki8;
    /* persistent little-core (A55) marshalling helper (ORK_SSM_PIPELINE): spawned once, condvar-signalled
     * per chunk to build the G-independent operands on the idle little cluster while the pS matmul runs on
     * the big cores. Kills the per-chunk pthread-spawn cost of the naive version. */
    pthread_t ssm_hth; pthread_mutex_t ssm_hmu; pthread_cond_t ssm_hgo, ssm_hdn; int ssm_hspawn, ssm_hgen, ssm_hdone, ssm_hstop; void *ssm_hjob;
    /* #39 resident mfold scratch. fold_A = the SHARED nc16 input buffer (keyed M,domain) — marshaled ONCE per
     * input and reused across all same-input weights (QKV / gate+up shared-input batch). fold_scr[] = a small
     * cache of per-(M,N,domain) C/RC/TK+tiles (q's N=3584 and k/v's N=512 coexist without thrash). */
    struct buf fold_A; int fold_A_M, fold_A_dom;
    struct fold_scratch *fold_scr[8]; int fold_scr_n; };
struct ork_w   { int K, N, Sk, Sn, dtype, gsize; int is_orkd; uint64_t orkd_id; struct buf *Bb; struct buf *Bf; int owns; uint8_t *Bi4; size_t Bi4_bytes; uint8_t quant_kind; float *bscale; int domain; struct buf own_buf; int own_buf_valid; struct buf *own_bufs; int n_own_bufs; uint32_t *pcrc; uint32_t *pcrc_meta; int pcrc_slots; int16_t *fa_lut; double fa_osc; struct buf *Bfold; int fold_ns; /* #39 mfold: resident fold_woff-layout weight (nslice bufs, K==FOLD_REF_K); NULL unless orkpack carries it */
    struct buf Bbc; int Bbc_valid; /* (A) fp16 CONTIGUOUS weight: all Sk K-slice Bb[ks] concatenated into ONE buffer (built lazily on first ORK_F16_CONTIG colsplit) so the HW chain can walk slice->slice WITHOUT crossing a dma-buf boundary (the cross-buffer CDMA-wild) — enables one chained submit/core like int8. Sn==1 only. */
    struct buf *Bbc_ns; int Bbc_ns_valid; /* (A-wideN) fp16 Sn>1 PER-N-SLICE CONTIGUOUS weights: Sn buffers, Bbc_ns[ns] = that slice's Sk K-slice tiles (Bb[ns*Sk+ks]) concatenated. Each slice is served as a standalone Sn==1 CONTIG colsplit (no cross-buffer wild); built once, resident (reclaimed at ctx teardown like Bbc). */
    struct buf Bgap[3]; int Bgap_valid; /* (B') identity mul_perchan_f16 DRAIN-GAP dummy buffers [in,out,scale] — a chained no-op SDP inserted between K-slices (ORK_F16_GAP) to idle the weight-CDMA so the prior fp16 fetch drains before the next slice's base latches. */
    struct ork_w_sliced *sliced; /* #33 slice-and-dice rescue: pre-built c_base doorbell tiles for a refuse-prone shape (built at pack time; run at the refuse site instead of ORK_RC_WEDGE_PRONE). NULL for well-behaved weights. Carries its own dtype. */ };
/* Tier 12f resident-KV handle. MUST match the typedef in include/ork_npu.h. The standalone Makefile build does
 * NOT pull ork_npu.h into this TU, so npu.c defines it; the CMake (ggml-ork) build DOES include the header here,
 * so the guard makes this local copy defer to it (identical shape either way — no conflicting-types). */
#ifndef ORK_KV_RESIDENT_T
#define ORK_KV_RESIDENT_T
typedef struct { ork_w *wkt, *wv; int HD, Lmax, Kp; uint64_t orkd_kv; } ork_kv_resident;
#endif
/* slice-and-dice rescue (#33): forward-declared here (full def near ork_mm_pack_sliced, far below) so
 * orki_pack() can BUILD w->sliced (raw B is only in scope at pack time), ork_mm_free can free it, and
 * run_multicore's refuse sites can RUN it. Signatures match include/ork_npu.h (repeat typedef/decls are
 * legal C11 and are compatible whether or not the header is also included in this TU). */
typedef struct ork_w_sliced { int K, N, Kpad, dtype; ork_slice_caps cap; int nks, nnt, ks, ns; ork_w **sub; } ork_w_sliced;
ork_w_sliced *ork_mm_pack_sliced(ork_npu *c, int K, int N, const void *B, int dtype);
int           ork_mm_run_sliced (ork_npu *c, ork_w_sliced *w, int M, const void *A, void *C, int nc);
void          ork_mm_free_sliced(ork_npu *c, ork_w_sliced *w);

/* ---- hot helpers: static inline so the split does not cost the cross-function inlining the monolith had
 * (ork_now_us alone has ~190 call sites, many on the per-submit path). See MODULARIZE_PLAN.md risk 1. */
#include <time.h>
static inline double ork_now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec*1e-3; }

/* hot, tiny, and on the per-submit path (orki_setr 517 call sites, orki_bsync 604) — keep them
 * header-inline so the split does not cost the inlining the monolith had. */
#include <string.h>
#include <sys/ioctl.h>
static inline void orki_setr(uint32_t*rc,int n,uint32_t b,uint32_t o,uint32_t v){for(int k=0;k+1<n;k+=2)if((rc[k]&0xffff)==o&&(rc[k+1]>>16)==b){rc[k]=(o)|((v&0xffff)<<16);rc[k+1]=(b<<16)|((v>>16)&0xffff);}}
static inline void orki_bsync(int fd,struct buf*b,uint32_t f){struct rknpu_mem_sync s;memset(&s,0,sizeof s);s.obj_addr=b->obj;s.size=b->size;s.flags=f;ioctl(fd,DRM_IOCTL_RKNPU_MEM_SYNC,&s);}
static inline void orki_bsync_off(int fd,uint64_t obj,uint64_t off,size_t size,uint32_t f){struct rknpu_mem_sync s;memset(&s,0,sizeof s);s.obj_addr=obj;s.offset=off;s.size=size;s.flags=f;ioctl(fd,DRM_IOCTL_RKNPU_MEM_SYNC,&s);}

/* ---- cross-module internals (extern; defined in npu.c or a src/npu/<mod>.c module) ---- */
void orki_pin_little_core(int id);          /* npu.c  — pin the caller to an idle A55 */
void orki_ssm_pool_free(ork_npu *c);        /* npu/ssm.c — release the persistent SSM scan pool (ork_npu_free) */

#include <signal.h>

/* ---- shared enums/markers the precision modules need (moved out of npu.c at the i4 lift) ---- */
enum { XP_MC_MM, XP_SC_MM, XP_CHAIN_NT, XP_STREAM_I8, XP_STREAM_F16,
       XP_I4_MC, XP_I4_MWARM, XP_I4_INCR, XP_I4CHAIN, XP_I4_STREAM, XP_SDP, XP_NPROFILE };
enum ork_chain_kind { OCK_NONE=0, OCK_SW, OCK_HW, OCK_FUSED };
#define ORK_DYN_SENT16 ((int16_t)0x7fff)   /* int4 (int16 output) sentinel — no valid W4A4 accumulator equals it */

/* the dynamic steered-submission chain handle — i4.c builds these directly */
struct ork_dyn_chain {
    ork_npu *c; int S, P, N, reserve, mc, spin_end; unsigned dom;   /* reserve = submitted task_number (fixed budget; can't grow); mc = multi-core; spin_end = reserve if the tail is a persistent spin (forward-chained), else 0 */
    struct buf *outbuf[1024];   /* per-op output DMA buffer (writeback + doorbell) */
    int32_t   *outptr[1024];    /* per-op output cpu ptr; doorbell = outptr[i][nout[i]-1] (last written word) */
    int        nout[1024];      /* per-op output element count = M*N (M>1 support; doorbell polls the last element) */
    int        oM[1024];        /* per-op M (rows); end() copies M*N int32 back to dst for the copy-back path */
    int        oSk[1024];       /* per-op K-split count: >1 => the op's output is oSk partial [M,N] blocks in scratch
                                 * that end() must SUM into dst[M,N] (the NPU has no on-device C+= mode). 0/1 = no K-split. */
    int        ostride[1024];   /* per-op copy-back dst row stride (elements): >0 => end() writes [M, nout/M] scratch
                                 * to dst at this row stride (a column-slice of a wider C, colsplit M>1). 0 = contiguous. */
    int        ocol0[1024];     /* colsplit balanced wide-N: this core's first C column (c0). With oscat, end()
                                 * recomputes the within-slice segment widths from (ocol0, nout/oM, N, nmax). */
    int8_t     oscat[1024];     /* 1 => balanced wide-N BOUNDARY-SCATTER copy-back: scratch is segment-major
                                 * [M,segw] blocks (each program contiguous, no notch); end() scatters each segment
                                 * to C[c0+coff .. ) at row-stride N, segment widths cut at nmax slice boundaries. */
    int32_t   *dst[1024];       /* mc: caller's C to copy the in-domain mcc output back to (end); NULL = write-in-place */
    struct buf ascr[1024]; int nascr;   /* scratch A copies (freed in end); zero-copy A miscomputes at M=1 */
    int        esz;             /* output element size in bytes: 4 = int8/fp16 (int32/fp32, NPU writes C directly),
                                 * 2 = int4 (W4A4 writes an int16 accumulator to scratch, end() widens to int32).
                                 * 0 (calloc default) is treated as 4 — only the int4 doorbell sets 2. */
    int        mc_nc;           /* DIAG/RECOVER (ork_dyn_begin_mc only; 0 elsewhere): core count for this round */
    int        mc_rc[8];        /* DIAG: per-core submit-ioctl return code (0 = accepted) */
    uint64_t   dma_rw0;         /* NPU cumulative dma_rw BEFORE the round; delta = HW work (0 => never dispatched) */
    /* RECOVER context: ork_dyn_end resubmits the round on a not-dispatched miss (the ~1/4000 concurrent
     * NONBLOCK dispatch race). c->maf/mrc/mtk[i] still hold the round's data (not reused until end), so the
     * stashed submits replay it. int8 only (mc_dt); fp16 drains in-submit. */
    struct rknpu_submit mc_subs[ORK_MAXCORE];
    int        mc_Pc[ORK_MAXCORE];
    unsigned   mc_dom; int mc_seed_all; int mc_dt;
    /* #54 BCHAIN-ON-THE-SHARED-DRAIN (i4batch): the M-batched int4 BCHAIN (run_i4_bchain_db) now builds its
     * per-core programs then rides ork_dyn_end for poll + recover (the PROVEN shared drain int8 colsplit uses),
     * instead of a hand-rolled per-worker poll/recover. Its output is 2D-tiled (not per-row/dense), so the poll
     * (ork_dyn_done_i), the recover re-seed (orki_mc_recover_resubmit), and the de-tile (ork_dyn_end writeback) all
     * delegate to bch_db_cells with this stored geometry — preserving the mode-1 gate, mode-4 collision-tolerance
     * (SENT16=0x7fff reachable), and mode-2 int16->int32 de-tile. */
    int        i4batch;                 /* 1 = BCHAIN programs drained by ork_dyn_end via bch_db_cells */
    int        b_H, b_Wb, b_Wmax, b_NG, b_M, b_N;   /* BCHAIN geometry (shared across cores) */
    int        b_c0[ORK_MAXCORE], b_c1[ORK_MAXCORE], b_NT[ORK_MAXCORE];   /* per-core N-chunk range + program count */
    int32_t   *b_C;                    /* caller's int32 C (de-tile destination) */
    int        f16_contig;  /* (A) 1 = fp16 colsplit built ONE chained submit/core over the contiguous Bbc weight (no
                             * cross-buffer boundary) — the worker takes the single-submit path, NOT the per-slice SW-chain. */
    int        prepolled;   /* 1 = the per-core parallel colsplit workers already submitted AND drained every core
                             * (blocking or per-core poll) — ork_dyn_end skips its (redundant, ~500ms-stalling) poll. */
    /* GROUPED int4 (ork_dyn_begin_mc_i4_grouped, drained by ork_dyn_grouped_end): per-row Sk int16 partial
     * blocks scaled + FLOAT-accumulated — C[m][n] = sum_g aS[m*Sk+g]*bS[g*N+n]*partial_g[n]. */
    int          i4g;              /* 1 = grouped-int4 float drain */
    const float *i4g_aS, *i4g_bS;  /* activation scale (M*Sk), weight scale (Sk*N) — caller host arrays */
    float       *i4g_Cf;           /* float output C[M,N] */
    int          i4g_N, i4g_Sk;    /* N + group count */
    /* SEQ chain (ork_dyn_begin_seq_i8 — heterogeneous single-group int8 chain, drained by ork_dyn_seq_end):
     * all ops share ONE output scratch; per-op layout/esz differ (matmul int32 dense, SDP int8 EWCUBE). */
    int        seq;             /* 1 = built by begin_seq_i8 */
    int        seq_term;        /* (single-core) op index whose terminal int32 last-col sentinel gates completion */
    int        seq_nc;          /* seq cores in this round (1 = single-core; >1 = groups spread across cores) */
    int        seq_term_c[ORK_MAXCORE];   /* per-core terminal op index (its last program, a matmul = sentinel) */
    struct buf seq_out;         /* shared output scratch for all seq ops (freed in seq_end) */
    uint8_t    oesz8[1024];     /* per-op output element bytes: 4=int32 matmul, 1=int8 SDP, 2=int16 SDP (silu) */
    uint8_t    ocube[1024];     /* per-op output layout: 1=EWCUBE-i8 (int8 SDP), 2=EWCUBEH-i16 (int16 SDP), 0=dense [M,N] (matmul) */
    size_t     ooff[1024];      /* per-op byte offset into seq_out */
    struct buf silu_lrc, silu_lsc;   /* int16-SiLU HW-chain: the LUT-load regcmd + SRAM buffers, resident across the chain; freed in seq_end */
    int        silu_lut;        /* 1 = a silu LUT-load prologue ran (Lrc/Lsc valid) */
};

/* ---- scaffold internals the precision modules call (de-static'd at the lift that needed them) ---- */
int      orki_check_overlap(const char *name, uintptr_t a_start, uintptr_t a_end, uintptr_t c_start, uintptr_t c_end);
unsigned orki_mm_timeout_ms(void);
int      ork_i4_batch(void);
void     orki_bdestroy(int fd, struct buf *b);
int      orki_budget(ork_npu *c, int M);
void     orki_synth_i4(uint32_t *rc, int mc, int K, int N, uint32_t aA, uint32_t aB, uint32_t aC);
int      orki_validate_regcmd(const char *op, ork_npu *c, const uint32_t *rc, int n, const ork_w *w, const struct buf *extra, int extra_n);
void     orki_tile_i4_Aslice(uint8_t *dst, const int8_t *Arow, int k0, int Kp);
extern const char *orki_last_op; extern int orki_last_K, orki_last_N, orki_last_wdom, orki_last_import;
extern volatile sig_atomic_t orki_ork_term, orki_in_doorbell;

/* ---- provided BY src/npu/i4/ to the scaffold ---- */
int  orki_bch_db_cells(ork_npu *c,int i,int c0,int c1,int Wb,int N,int NG,int M,int H,int Wmax,int32_t *C,int mode,int only_tk);
int  orki_i4_submit_tmo_ms(void);
int  orki_run_i4_bchain_db(ork_npu *c, ork_w *w, int M, const int8_t *A, int32_t *C, int nc);
int  orki_run_i4_experts_bchain_db(ork_npu *c, const ork_mm_task_i4 *ex, int ntask, int nc);

struct ork_xspec { uint8_t kwp, rst, wtg, wc, stg, sc, setdt; };

/* mode-transition policy selectors (XSPEC rows live in core/mode.c) */
enum { KWP_NONE, KWP_MC, KWP_SC, KWP_NTI, KWP_NTL, KWP_F16 };
enum { RC_NEVER, RC_NOTKW, RC_I8ENTRY, RC_NOTLIVE, RC_NOTLIVE_NOTKW, RC_ALWAYS, RC_SDPKW };
enum { TG_NONE=0, TG_SCALAR=1, TG_PERCORE=2, TG_BOTH=3 };
enum { WC_NONE, WC_NOTKW, WC_NOTLIVE_NOTKW, WC_ALWAYS, WC_NT, WC_NT_NOTKW };

/* ---- dtype predicates ---- */
#define ORK_I8_LIVE(dt) ((dt)==DT_I8 || (dt)==3)
#define ORK_INT_DT(dt) ((dt)==DT_I8 || (dt)==DT_I4 || (dt)==3)
#define ORK_KW_DT(dt) (ORK_I8_LIVE(dt) || (dt)==DT_F16)

/* ---- env-knob accessors: one-line cached getenv, static inline so every module can read them
 * without a cross-TU call (each TU keeps its own idempotent cache) ---- */
static inline int ork_nothrash(void){ static int v=-1; if(v<0){const char*e=getenv("ORK_MIXED_NOTHRASH"); v=(e&&atoi(e))?1:0;} return v; }
static inline int ork_f16warm(void){ static int v=-1; if(v<0){const char*e=getenv("ORK_SSM_KEEPWARM"); v=e?(atoi(e)?1:0):1;} return v; }
static inline int ork_sdp_noreset(void){ static int v=-1; if(v<0){const char*e=getenv("ORK_SDP_NORESET"); v=e?(atoi(e)?1:0):1;} return v; }
static inline int ork_precomp(void){ static int v=-1; if(v<0){const char*e=getenv("ORK_PRECOMP_RC"); v=(e&&atoi(e))?1:0;} return v; }
static inline int ork_norm_npu_enabled(void){ static int v=-1; if(v<0) v=getenv("ORK_NORM_NPU")?1:0; return v; }
static inline int ork_softmax_npu_enabled(void){ static int v=-1; if(v<0) v=getenv("ORK_SOFTMAX_NPU")?1:0; return v; }

#define REGCMD_I8_EW_N (REGCMD_I8_N + REGCMD_EW_LANE_N)   /* 224 + 36 = 260 words = 126 reg entries + trailer */

/* ---- scaffold <-> f16 module ---- */
double orki_silu_f(double x);
int ork_f16_colsplit(void);
int ork_mm_run_f16_f16out(ork_npu *c, ork_w *w, int M, const ork_f16 *A, ork_f16 *out);
int ork_norm_reduce_npu(ork_npu *c,int M,int n,const f16 *x,float *ss_out);
int ork_norm_rsqrt_npu(ork_npu *c,int M,int nf,double eps,const float *ss,float *scale);
int orki_bmm_c_dense(const ork_bmm_strides *s,int N);
ork_bmm_strides orki_bmm_natural(int M,int K,int N);
unsigned orki_ew_timeout_ms(void);
void orki_bmm_scatter_i32(int32_t *dst, const int32_t *src, int rows, int cols, long sr, long sc);
void orki_set_f16_out(uint32_t*rc,int N,int stride);
void orki_set_f16_out_fp16in(uint32_t*rc,int M,int N);
void orki_set_mul_geom(uint32_t *rc,int n,int M,int N);
void orki_splice_ew_lane(uint32_t*rc,const uint32_t*base);
void orki_synth(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC,int sched,int cbuf);

void orki_bmm_gather_i8(int8_t *dst, const int8_t *src, int rows, int cols, long sr, long sc);
void orki_bmm_gather_f16(f16 *dst, const f16 *src, int rows, int cols, long sr, long sc);

/* ---- shared op/SDP constants (used by more than one precision module) ---- */
#define ORK_DYN_HEADROOM 2
#define ORK_DYN_SENT 0x7fffffff
#define ORK_SEQCUBE(m,n,MM) (((n)/16)*((MM)*16) + (m)*16 + ((n)%16))   /* NVDLA atom-16 SDP cube */
#define ORK_SILU16_C4064  0xff43770au
#define ORK_SILU16_C4068  0x7eae1100u
#define ORK_SILU16_IDXOFF 0xffffc000u
#define ORK_SILU_C4064  0xffff7dc8u
#define ORK_SILU_C4068  0x411c0800u
#define ORK_SILU_IDXOFF 0xffffc000u
#define ORK_SUBMIT_FLOOR_US 167   /* measured RK3588 single-submit floor; linger past this isn't floor-bound */
#define SILU16_NS         4096        /* dense samples across [-32768,32767], step 16 */
#define SILU16_QSTEP      16

/* ---- scaffold helpers the precision modules call ---- */
double orki_exp_f(double x);
extern struct sigaction orki_prev_sig[2];
int orki_build_act_lut16(ork_npu *c,double(*f)(double),double in_scale,double out_scale,int16_t *lut);
int orki_layer_mm(ork_npu *npu, ork_w *W, const int8_t *A, int K, int N, int32_t *C);
int ork_dyn_grouped_end(ork_dyn_chain *h);
int orki_int8_ks(ork_npu *c);
int orki_mtile_cap(int Kred);
int orki_seq_op_ok(const ork_seq_op *o, unsigned *dom, int *have_dom);
ork_async *ork_async_launch(struct ork_async tmpl);
ork_dyn_chain *ork_dyn_begin_colsplit(ork_npu *c, const ork_mm_task_i8 *t, int ncreq);
ork_dyn_chain *ork_dyn_begin_mc_i4(ork_npu *c, int S, const ork_mm_task_i8 *tasks, int nc);
void orki_mc_recover_resubmit(ork_dyn_chain *h);
void ork_install_term(void);
void orki_set_i16_out(uint32_t*rc,int N,int stride,int mult,int shift);
void orki_seq_build_op(ork_dyn_chain *h, const ork_seq_op *o, int gi, struct buf *RC, struct buf *AF, struct rknpu_task *tks, size_t *astage, size_t *coff, int pp, int nx_pp, int nx_kind, int CBUF);



enum ork_async_kind { OAK_F16=0, OAK_I8, OAK_I4, OAK_CHAIN_I8, OAK_CHAIN_I4, OAK_STREAM_I8, OAK_STREAM_I4 };

/* ---- i8 subtree symbols the scaffold calls ---- */
int ork_dyn_done_i(ork_dyn_chain *h, int i);
int orki_chain_fullk_mcap_i8(ork_npu *c, int K);
struct chain_silu_spec { const ork_chain_op *ops; int task; int sdp_task; int r_mult, r_shift; int gate_mult, gate_shift; uint32_t out_bias, idx_off, cfg4064, cfg4068; const int16_t *lut; int nlut; };
int orki_run_chain_i8_impl(ork_npu *c, int S, const ork_mm_task_i8 *tasks, const struct chain_silu_spec *ss, int force_core);
int orki_run_i4_mc_db(ork_npu *c, ork_w *w, int M, const int8_t *A, int32_t *C, int nc);
int orki_silu_calibrate_idx(ork_npu *c);
int orki_silu_calibrate_idx16(ork_npu *c);
int orki_slice_rescue_or_refuse(ork_npu *c,ork_w *w,int M,const void *A,void *C,int nc);
int orki_slice_run_i8(ork_npu *c, ork_w_sliced *w, int M, const int8_t *A, int32_t *C, int nc);
ork_w_sliced *orki_slice_pack_i8(ork_npu *c, int K, int N, const int8_t *B);
void orki_inflate_chan_nf4_i8(const uint8_t *nib, int K, const int8_t lut[16], int8_t *i8);
void orki_set_i8_out8(uint32_t*rc,int N,int stride,int mult,int shift);
void orki_set_i8_silu(uint32_t*rc,int N,int stride,int r_mult,int r_shift, uint32_t out_bias,uint32_t idx_off,uint32_t cfg4068);
void orki_silu_build_curve(ork_npu *c,double(*f)(double),double in_scale,double out_scale,int16_t *lut);
void orki_silu_build_curve_biased(ork_npu *c,double(*f)(double),double in_scale,double out_scale,double bias,int16_t *lut);
void orki_tile_f32_i8(ork_npu *c, ork_w *w, int K, int N, const float *f32, const float *inv);
void orki_tile_i8_range(int lo,int hi,void *a);
void orki_tile_i8_to_import_tiles(ork_npu *c, ork_w *w, int K, int N, const int8_t *i8);
void orki_tile_i8_to_tiles(ork_npu *c, ork_w *w, int K, int N, const int8_t *i8);

enum { OP_MM32=0, OP_MM8=1, OP_SILU=2, OP_EWMUL=3 };   /* ork_chain_op.kind values (struct is in ork_npu.h) */

#define ORK_RC_F16_SC (-502)   /* run_multicore->orki_run signal: retry the single-core fp16 reference */

struct ork_async { pthread_t th; int started; int rc; enum ork_async_kind kind;
    ork_npu *c; ork_w *w; int M;
    const void *A; void *C;                 /* single-matmul A/C (typed per kind) */
    int S; const void *tasks; };            /* chain/stream task array (typed per kind) */
struct ork_dyn_queue { ork_npu *c; int chunk_max, ncore, linger_us; ork_mm_task_i8 *tasks; int n, cap, submitted; ork_dyn_chain *h; double last_push_us; };

struct ork_pc_chain {
    ork_npu *c; int S, N, warmed; unsigned dom;
    struct buf pool;                 /* S precompiled programs, contiguous in one buffer */
    struct buf ascr[512];            /* per-program fixed A scratch (address baked into the program) */
    const void *asrc[512]; int Ksz[512];   /* caller's A source + K, re-read each run */
    struct buf *outbuf[512]; int32_t *outptr[512];
};

void orki_set_i8_ewmul(uint32_t*rc,int M,int N,int stride,int mult,int shift,uint32_t aG);
void orki_apply_ork_geom(uint32_t*rc,int n,int mc,int K,int N,int cbuf);
void *ork_fbc_thread(void *vp);

struct ork_fbc_arg { int fd, dom, core, P, rc; struct buf *tk; };

struct fold_scratch { int M, N, nslice, P, domain; int roff[64]; uint32_t *tmpl; struct buf *Cc, *RC, *TK; };

struct ork_stage {
    int K, N, Sk, Sn;
    struct buf *Bb;            /* Sk*Sn tile dma-bufs (bare: cpu/heap_fd set; dma/handle 0 until mapped) */
    struct buf *Bf;            /* Sn full-K dma-bufs (NULL if outside the Bf envelope) */
    int8_t *i8scratch;         /* reused N*K linear-int8 inflate scratch */
    ork_w view;                /* ork_w that points Bb/Bf at this slot's bufs once mapped (run target) */
    int mapped;
};

struct ork_stream_entry { struct ork_stage *stg; int K, N; int mapped; uint64_t last_use; };

struct ork_stream_pool  { ork_npu *c; struct ork_stream_entry **e; int n, cap; uint64_t clock; };

struct slc_acc { const ork_mm_task_i8 *tasks; int32_t *C; int nks, nnt, ns, N, M, c0, c1; };

struct streamw_i8sk { ork_npu *c; int core; int S; const ork_mm_task_i8 *tasks; int *ctr; int rc; };

struct tile_i8_arg { int8_t *bb; const int8_t *Bi; int KT, k0, n0, N; };

ork_w *orki_pack(ork_npu *c,int K,int N,const void *B,int dt);
extern long orki_imp_wn;
void orki_chan_scales_f32(const float *f32, int K, int N, float *inv, float *bscale);
extern const float ORKI_NF4_LEVELS[16];
void orki_expand_chan_i4_i8(const uint8_t *nib, int K, int8_t *i8);

/* named so a module-owned regcmd fuzz-override table can be extern-declared across the split */
struct ork_regovr { uint32_t blk, reg, val; };
extern struct ork_regovr orki_i8_fovr[16]; extern int orki_i8_fovr_n;

extern _Thread_local int orki_in_slice_pack;
struct ork_stream_entry *orki_pool_new_entry(struct ork_stream_pool*p,int K,int N);

int orki_run(ork_npu *c,ork_w *w,int M,const void *A,void *C);
void *orki_slice_acc_worker(void *p);
int orki_fused_mtile(int K,int M);

void orki_set_i8_silu32(uint32_t*rc,int N,int r_mult,int r_shift,uint32_t out_bias,uint32_t idx_off,uint32_t cfg4068);
int orki_chain_build_lut_fn(ork_npu*c, double(*fn)(double), double in_scale, double out_scale, int r_mult, int r_shift, uint32_t cfg4068, int16_t *lut);
double orki_gelu_f(double x);
double orki_rsqrt_f(double x);
int orki_act_lut_i16(ork_npu *c,double(*f)(double),const int16_t *in,int M,int N,double in_scale,double out_scale,int16_t *out,double *us);
void *orki_stream_worker_i8sk(void *vp);
int orki_act_lut_i8(ork_npu *c,double(*f)(double),const int8_t *in,int M,int N,double in_scale,double out_scale,int8_t *out,double *us);

#define ORK_I4A8_MAGIC  0x4F344E31u           /* 'O','4','N','1' */
#define ORK_I4A8_VER    ork_pack_format_version()  /* int4 blob compat = library MAJOR (see ork_npu.h) */
#define ORK_I4_KS 10752       /* int4 single-submit K ceiling (validated == int8's) */

/* xorshift PRNG for stochastic int4 rounding — tiny and hot, header-inline so every module gets it */
static inline uint32_t ork_xs32(uint32_t *s){ uint32_t x=*s; x^=x<<13; x^=x>>17; x^=x<<5; *s=x; return x; }
ork_dyn_chain *ork_dyn_begin_mc_i4_grouped(ork_npu *c, int M, ork_w *w, const int8_t *A, const float *aScale, const float *bScale, float *Cf, int nc);

extern struct ork_regovr orki_i4_fovr[16]; extern int orki_i4_fovr_n;

ork_w_sliced *orki_slice_pack_i4(ork_npu *c, int K, int N, const int8_t *B);
int orki_slice_run_i4(ork_npu *c, ork_w_sliced *w, int M, const int8_t *A, int32_t *C, int nc);

struct ork_i4a8_hdr { uint32_t magic, version; int32_t K, N; uint32_t quant_kind; };

void ork_stage_fill(ork_npu *c, struct ork_stage *s, const ork_w *src);

extern double orki_f16_slice_us;

#endif /* ORK_NPU_INTERNAL_H */
