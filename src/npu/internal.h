/* npu/internal.h — the PRIVATE ABI shared by src/npu.c and the src/npu/*.c precision modules.
 *
 * Round 1 of the modularization (MODULARIZE_PLAN.md) splits the npu.c monolith along the precision axis
 * into src/npu/{sdp,f16,i16,i4,ssm}.c + src/npu/i8/*.c. Everything those translation units need in common
 * — the context/weight/buffer types, the dtype markers, and the hot helpers that must stay inlinable
 * across the split — lives here. NOT a public header: include/ork_npu.h is the API; this is internals,
 * and nothing outside src/ may include it.
 *
 * Include it the same way from every depth (-Isrc makes this resolve from src/npu.c, src/npu/i8.c and
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
typedef struct ork_w_sliced ork_w_sliced;
ork_w_sliced *ork_mm_pack_sliced(ork_npu *c, int K, int N, const void *B, int dtype);
int           ork_mm_run_sliced (ork_npu *c, ork_w_sliced *w, int M, const void *A, void *C, int nc);
void          ork_mm_free_sliced(ork_npu *c, ork_w_sliced *w);

#endif /* ORK_NPU_INTERNAL_H */
