/* npu.c — core regcmd matmul engine for the Rockchip NPU (see include/ork_npu.h).
 *
 * Raw DRM submission (no librknnrt): synthesizes a register-command program per matmul tile
 * and submits it to the `rknpu` kernel driver. fp16 A x fp16 B -> fp32 C. Tiling:
 *   - K split into <= soc.ks slices, partials accumulated (host-side, fp32);
 *   - N split into <= soc.nmax output-column slices (the NPU caps output width);
 *   - each slice M-tiled: clean power-of-2 Kp uses the single-submit internal M-scheduler,
 *     odd remainder Kp falls back to one internal M-tile per submit (correct for any Kp).
 * One reused feature buffer (the NPU caches feature state by address) + a one-time
 * cold-start warmup per fresh output buffer. SoC-specific numbers come from soc.h.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/mman.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <math.h>
#include <sys/prctl.h>   /* PR_SET_TIMERSLACK — trim the default 50µs nanosleep slack so the doorbell backoffs are precise */
#include <errno.h>
#include <dlfcn.h>
#include <signal.h>
#include "rknpu_ioctl.h"
#include "ork_regs.h"
#include "orkd_client.h"   /* Path B: transparent orkd client routing (gated by ORK_USE_ORKD) */
#include "orkd_proto.h"    /* ORKD_DT_* wire dtypes for the transparent ring transport */
#include "spine_kernels.h" /* CPU glue (rmsnorm/rope/attn/silu/quant/civac) for the whole-layer core ork_mm_layer_i8 */
#include "regcmd_array_4x32x16.h"
#include "regcmd_i8.h"
#include "regcmd_i4.h"
#include "regcmd_silu.h"
#include "regcmd_ewmul.h"
#include "regcmd_softmax_f16.h"   /* RE: captured vendor forward-softmax 9-task graph (replay) */
#include "regcmd_softmax_wt.h"    /* RE: its matmul weight blobs (verbatim) */
#include "regcmd_reshape.h"       /* RE: vendor fp16 contiguous->atom-8 reshape base (task4) — WIP */
#include "ork_npu.h"
#include "ork_slice.h"           /* slice-and-dice decomposer (ork_slice_matmul / caps) for the sliced-doorbell primitive */
#include "soc.h"
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include "npu/internal.h"   /* private ABI: ork_npu/ork_w/buf types, dtype markers */
#include "npu/core.h"       /* substrate API: buffers, domains, submit, device, sched, mode, prof */
/* last_dt SCHEDULE markers that all run int8 regcmd PROGRAMS on the NPU: single-core (DT_I8=1) and the
 * chain/stream path (3, "DT_I8_CHAIN"). Switching AMONG these is not a hardware MODE change, so it needs
 * no RKNPU_ACT_RESET — the reset is only for ENTERING int8 from fp16/int4/cold (the first-int8-submit
 * wedge). Decoupling the reset from the marker lets decode interleave run_i8 singletons with run_stream
 * (QKV/gate-up) groups without a ~107ms NPU soft-reset at every matmul boundary. The cold 2-pass warmup
 * (fresh-output-buffer priming) is kept independently — see the reset sites. */
/* Every int-datapath mode (int8 / int4 / chain=3). ORK_MIXED_NOTHRASH: when set, an int↔int dtype
 * transition (e.g. the per-tensor mixed dispatch alternating W4A4 q4-tensors and W8A8 q6-tensors within a
 * decode token) does NOT ACT_RESET + re-warm + realloc the per-core buffers. MEASURED: mixed W4A4 decode
 * without this = 0.16 t/s (99% run_multicore SETUP = the per-switch reset/rewarm/realloc thrash); pure
 * int4 (no switching) = 6.43 t/s. The reset is cold-entry wedge-protection (fp16→int, or first int); it is
 * NOT needed BETWEEN already-warm int modes — exactly as int8↔chain already interleaves reset-free
 * (ORK_I8_LIVE). Off by default (validate on silicon before promoting). */
/* ORK_SSM_KEEPWARM: same lever as ORK_MIXED_NOTHRASH but for the int8-matmul <-> fp16-scan pair (Mamba-2
 * on-NPU SSM). An int8<->fp16 transition per layer otherwise ACT_RESETs + re-warms + reallocs the per-core
 * buffers (measured: matmul run 99% SETUP, 28 soft-resets/prefill). NVDLA/rocket resets only at init and
 * carries precision per-task (0x4010 PROC_PRECISION), so the reset is a conservative software choice, not a
 * HW law — and regcmd is re-synthesized every call regardless, so skipping the reset/rewarm/realloc is
 * correctness-safe (the program is always rebuilt). Buffers are grow-only so they stabilize to the fp16
 * (2-byte activation) size and are reused by int8. Off by default; validate on silicon (errno=110) before
 * promoting. The families that may interleave reset-free: int8-live (DT_I8/chain) and DT_F16. */
/* DEFAULT ON (2026-07-13): validated general, coherent, bit-exact-safe win — skips the int8<->fp16 ACT_RESET
 * churn for any fp16-op interleaved with int8 matmuls (SSM scan etc.). ORK_SSM_KEEPWARM=0 to disable. */
/* ORK_SDP_NORESET: skip the entry ACT_RESET in the element-wise fp16/int16 SDP ops (ewmul_f16/i16,
 * add_f16/i16). Their int8 twins (ewmul_i8/add_i8) already skip it; the reset was drift, not required.
 * VALIDATED (2026-07-14): coherent (test_ewmul_f16/i16, test_add pass), wedge-free after an int8/fp16
 * matmul (mode_probe SAFE); removes a ~107ms/op reset — test_ewmul_f16 3436→53ms, ewmul_i16 4292→44ms,
 * add 1419→50ms. DEFAULT ON (skip the reset). Set ORK_SDP_NORESET=0 to restore the old reset. */
/* ORK_PRECOMP_RC: reuse a weight's precompiled M=1 decode regcmd (skip per-submit synth+validate). Opt-in. */
/* ork_i4_batch() — STRATEGY A: int4 stride-2 IN-TASK batch (Exp-2026-06-19). One submit computes a whole
 * M-tile with resident weights (mc_phys=2*H, 0x405c=0, stride-2 output → physical row 2m carries logical
 * row m; NEON int16→int32 de-tile physrow=4j+4H*b), instead of the per-row PC-chain that re-streams the
 * weight every row (the W4A4 submit-bound the int8 0x1040 M-scheduler avoids). Default ON (bit-exact
 * validated ./i4; per-row fallback where the batch doesn't fit). Implemented in synth_i4 mc>1 + orki_run_i4_mc_db (per-row doorbell)
 * + stream_worker_i4; NVDLA D_BATCH_NUMBER/D_*_STRIDE analogy.
 *   PRESERVED as a distinct, named strategy — the multi-task-submit batch (many 1-row tasks per submit, the
 *   vendor's int4 approach; task_number=rows) is a SEPARATE path and must NOT overwrite/conflate with this.
 *   Env var kept as ORK_I4_MSCHED for back-compat (0=off, 1=on). */
/* ORK_I4_NSUB: N-subslice the batch path so wide-N (Ncore*K > 131072 weight budget) still batches — split N
 * into ≤131072/K-column chunks, one batch submit each, instead of falling back to per-row. Default OFF until
 * board-validated; when off, msched keeps the weight-fit guard (whole-Ncore batch or per-row). */

/* ORK_PROFILE: per-matmul host-side timing, printed on free. Lets us see how much of decode's
 * per-token wall time is spent inside ork-driver's matmul calls vs the ggml/CPU path around them. */
int    orki_ork_prof = 0;
/* ORK_LOAD_PROF: per-phase breakdown of the .orkpack import path — where load time actually goes.
 * Accumulated across every orki_bimport()/load; dumped (and reset) at teardown by ork_load_prof_dump(). */
int    orki_load_prof = 0;   /* set from ORK_LOAD_PROF in ork_npu_init, like orki_ork_prof */
double orki_lp_alloc=0, orki_lp_mmap=0, orki_lp_prime=0, orki_lp_create=0, orki_lp_memcpy=0, orki_lp_bf=0;
long   orki_lp_nchunk=0; size_t orki_lp_bytes=0;
long   orki_prof_i8_calls = 0, orki_prof_i4_calls = 0;
double orki_prof_i8_us = 0,    orki_prof_i4_us = 0;
/* RKNPU_SUBMIT ioctl counter (ORK_PROFILE). orki_prof_submits = total ioctls; the run path tags each
 * via g_prof_submit_class so we can split within-matmul tiling (K/N/M sub-submits) from chained
 * (one ioctl covering >1 program). Printed on free. Pure diagnostic — no effect when prof off. */
long   orki_prof_submits = 0;       /* total RKNPU_SUBMIT ioctls */
long   orki_prof_submit_progs = 0;  /* total PC-chained programs across all submits (>= submits) */
long   orki_prof_submit_chained = 0;/* submits that carried >1 program (chained) */
/* FLOOR-DECOMP instrumentation (always-on, ~40ns/submit): decompose the per-submit floor.
 * orki_fd_ioctl_us = wall time inside the SUBMIT ioctl (kernel job setup + NPU + completion-wait, blocking).
 * orki_fd_hw_us    = SUM of the kernel-reported sub->hw_elapse_time (the HW's own NPU-busy view) — ork never
 * read this field before. Comparing the two splits the ioctl wall into real-NPU vs driver-wait/dispatch.
 * Read via ork_npu_floor_timing(); reset via ork_npu_floor_reset(). Raw units of hw_elapse are captured
 * separately (g_fd_hw_raw) so the probe can infer ns-vs-us. */
double orki_fd_ioctl_us = 0;   /* sum wall-us inside the SUBMIT ioctl */
double orki_fd_hw_us = 0;      /* sum of sub->hw_elapse_time (as reported) */
long long orki_fd_hw_raw_last = 0; /* last raw hw_elapse_time value (for unit inference) */
long   orki_fd_n = 0;          /* number of SUBMIT ioctls timed */
/* #33 reentrancy guard: orki_slice_pack_i8 packs each sub-tile via ork_mm_pack_i8 -> orki_pack(), which would ITSELF
 * try to build a w->sliced for that sub-tile (harmless for the natural gate — sub-tiles are Sn==1, never
 * refuse-prone — but ORK_SLICE_ALL forces it and RECURSES until IOVA fills). Set while packing sub-tiles so
 * the nested orki_pack() skips its slice-build. Thread-local: concurrent packs on different threads don't clash. */
_Thread_local int orki_in_slice_pack;
/* pcrc: PRECOMPILED regcmd cache (ORK_PRECOMP_RC) — the M=1 decode regcmd for this weight is FIXED across
 * tokens (same weight tiles + reused per-core AF/CC scratch => same K/N/addresses), so synth it ONCE and
 * reuse the bytes, skipping the ~20 per-submit setr scans + validate_regcmd. pcrc holds pcrc_slots×REGCMD_N
 * words (slot = core*Sn+ns); pcrc_meta holds 6 words/slot {valid,K,Nc,aA,aB,aC} — the reuse is address-
 * validated (a buffer realloc / dtype thrash changes an addr => miss => re-synth), so it is safe even
 * without ORK_MIXED_NOTHRASH. Allocated lazily on first decode; freed in ork_w_free. rkllm's static-regcmd lever. */  /* owns=1: per-tile bcreate, reclaimable by ork_mm_free; owns=0: arena views (freed at teardown). own_buf: a single dedicated DMA buffer backing ALL of this weight's tiles as base+offset VIEWS (grouped-i4) — reclaimed as one bdestroy by ork_mm_free (own_buf_valid=1), tiles are non-owning views so they are NOT individually destroyed. own_bufs/n_own_bufs: the SIZE-BOUNDED variant (chunked consolidated import) — a weight's tiles are packed into a handful of moderate (ORK_IMPORT_CHUNK_MB, ~16MB) imported dma-buf chunks instead of one giant per-weight buffer (which hangs the DMA_HEAP_ALLOC) or one bimport per tile (which faults the chain-walk with too many foreign mappings); tiles are base+offset views into their chunk; ork_mm_free bdestroys every chunk. Bi4: optional host-side int4-packed (nibble) weight store for pack_i4a8 — the memory-compact form (K*N/2 B) for .orkpack/streaming dump; NPU-side runs int8 (DT_I8). quant_kind: ORK_QK_* — how the nibbles in Bi4 inflate (UNIFORM sign-extend now; CODEBOOK_NF4 LUT reserved). bscale: optional per-output-channel dequant scale (length N) retained alongside Bi4 so the compact int4 form (pack_i4a8 / load_i4a8) can be dumped + reloaded self-contained. domain: this weight's NPU IOMMU domain id (0 = default); its resident tiles live there and its submits run against it — multi-domain residence lets >4 GiB of weights stay resident across domains (the per-domain 32-bit IOVA cap). */
int orki_check_overlap(const char *name, uintptr_t a_start, uintptr_t a_end, uintptr_t c_start, uintptr_t c_end) {
    if (a_start < c_end && c_start < a_end) {
        fprintf(stderr, "[ork] ERROR [%s]: memory overlap detected! A [%p, %p) overlaps with C [%p, %p).\n",
                name, (void*)a_start, (void*)a_end, (void*)c_start, (void*)c_end);
        return 1;
    }
    return 0;
}
static int is_valid_dma_addr(ork_npu *c, uint32_t addr, const ork_w *w, const struct buf *extra, int extra_n) {
    if (addr == 0) return 0;
    if (c->regcmd.cpu && addr >= c->regcmd.dma && addr < c->regcmd.dma + c->regcmd.size) return 1;
    if (c->task.cpu && addr >= c->task.dma && addr < c->task.dma + c->task.size) return 1;
    if (c->Af.cpu && addr >= c->Af.dma && addr < c->Af.dma + c->Af.size) return 1;
    if (c->Cc.cpu && addr >= c->Cc.dma && addr < c->Cc.dma + c->Cc.size) return 1;
    for (int i = 0; i < ORK_MAXCORE; i++) {
        if (c->mrc[i].cpu && addr >= c->mrc[i].dma && addr < c->mrc[i].dma + c->mrc[i].size) return 1;
        if (c->mtk[i].cpu && addr >= c->mtk[i].dma && addr < c->mtk[i].dma + c->mtk[i].size) return 1;
        if (c->maf[i].cpu && addr >= c->maf[i].dma && addr < c->maf[i].dma + c->maf[i].size) return 1;
        if (c->mcc[i].cpu && addr >= c->mcc[i].dma && addr < c->mcc[i].dma + c->mcc[i].size) return 1;
    }
    if (c->mtk_all.cpu && addr >= c->mtk_all.dma && addr < c->mtk_all.dma + c->mtk_all.size) return 1;
    for (int i = 0; i < c->dma_n; i++) {
        if (c->dma_tab[i].cpu && addr >= c->dma_tab[i].dma && addr < c->dma_tab[i].dma + c->dma_tab[i].size) return 1;
    }
    if (w) {
        if (w->Bb) {
            int num_weights = w->Sn * w->Sk;
            for (int i = 0; i < num_weights; i++) {
                if (w->Bb[i].cpu && addr >= w->Bb[i].dma && addr < w->Bb[i].dma + w->Bb[i].size) return 1;
            }
        }
        if (w->Bf) {
            for (int i = 0; i < w->Sn; i++) {
                if (w->Bf[i].cpu && addr >= w->Bf[i].dma && addr < w->Bf[i].dma + w->Bf[i].size) return 1;
            }
        }
        /* fp16 CONTIG (Task #50): the contiguous concatenated weight (all K-slices in ONE buffer). Without this
         * clause a valid Bbc.dma+offset weight base was FALSE-flagged "wild/unallocated" -> validate_regcmd failed
         * -> CONTIG refused -> fell back to the concurrent per-slice path that wedges. Bbc.cpu==0 when unused. */
        if (w->Bbc.cpu && addr >= w->Bbc.dma && addr < w->Bbc.dma + w->Bbc.size) return 1;
        for (int i = 0; i < 3; i++) if (w->Bgap[i].cpu && addr >= w->Bgap[i].dma && addr < w->Bgap[i].dma + w->Bgap[i].size) return 1;   /* CONTIG GAP-stagger filler buffers */
    }
    if (extra && extra_n > 0) {
        for (int i = 0; i < extra_n; i++) {
            if (extra[i].cpu && addr >= extra[i].dma && addr < extra[i].dma + extra[i].size) return 1;
        }
    }
    return 0;
}
/* Last regcmd context (set by validate_regcmd) — dumped on a submit failure to PIN which weight/op/domain/
 * import-status faulted (e.g. the multi-domain import scale-fault: errno=22). */
const char *orki_last_op = "?"; int orki_last_K=0, orki_last_N=0, orki_last_wdom=-1, orki_last_import=0;
int orki_validate_regcmd(const char *op, ork_npu *c, const uint32_t *rc, int n, const ork_w *w, const struct buf *extra, int extra_n) {
    /* stash context so a later submit failure can name the exact weight/op/domain/import-status that faulted */
    orki_last_op = op ? op : "?";
    if (w) { orki_last_K = w->K; orki_last_N = w->N; orki_last_wdom = w->domain;
             orki_last_import = (w->own_buf_valid && w->own_buf.heap_fd > 0) ||
                             (w->own_bufs && w->n_own_bufs > 0 && w->own_bufs[0].heap_fd > 0) ||
                             (w->Bb && w->Bb[0].heap_fd > 0) || (w->Bf && w->Bf[0].heap_fd > 0); }
    for (int k = 0; k + 1 < n; k += 2) {
        uint32_t offset = rc[k] & 0xffff;
        uint32_t block_id = rc[k+1] >> 16;
        uint32_t val = (rc[k] >> 16) | ((rc[k+1] & 0xffff) << 16);
        const char *reg_name = NULL;
        if (offset == 0x1070 && block_id == 0x201) reg_name = "adma";
        else if (offset == 0x1110 && block_id == 0x201) reg_name = "bdma";
        else if (offset == 0x4020 && block_id == 0x1001) reg_name = "cdma";
        if (reg_name) {
            if (val == 0) {
                fprintf(stderr, "[ork] ERROR [%s]: regcmd sanity assertion failed! %s is NULL (0x00000000).\n", op, reg_name);
                return -1;
            }
            if ((val & 15) != 0) {
                fprintf(stderr, "[ork] ERROR [%s]: regcmd sanity assertion failed! %s address 0x%08x is not 16-byte aligned.\n", op, reg_name, val);
                return -1;
            }
            if (!is_valid_dma_addr(c, val, w, extra, extra_n)) {
                fprintf(stderr, "[ork] ERROR [%s]: regcmd sanity assertion failed! %s address 0x%08x is wild/unallocated (outside all valid buffers).\n", op, reg_name, val);
                return -1;
            }
        }
    }
    return 0;
}
/* PHASE-B K-TILE knob (ORK_KTILE, default OFF). The int8 weight K-axis is sliced into KS-element
 * slices (Bb[ns*Sk+ks]) and the slice partials accumulated host-side; the single-task M-tile is capped
 * by the 0x1040 K-reduction schedule (mg_max*64), which GROWS as Kp shrinks. Default KS=1024. Setting
 * ORK_KTILE=kt (a multiple of 32, < K) forces KS=kt at BOTH pack and run, so a smaller slice yields a
 * larger M-tile -> more per-task M-amortization (weighed against the host-accumulate cost of more slices). To make a K<=10752 weight actually TAKE the K-split
 * path (it would otherwise run full-K via Bf), Bf is suppressed when KTILE is active and < K. The run
 * path is the existing, bit-exact-validated per-slice/chain-ksplit accumulation — no new regcmd; every
 * program still goes through validate_regcmd, and accumulation == full-K == the CPU ref. WEDGE-SAFE:
 * it reuses the proven wide-K accumulate; an out-of-range value is ignored (falls back to KS=1024). */
int orki_int8_ks(ork_npu *c){ (void)c;
    static int kt=-2;
    if(kt==-2){ const char*e=getenv("ORK_KTILE"); kt=e?atoi(e):0;
        if(kt && (kt<32 || kt%32)) kt=0; }   /* must be a multiple of 32; else ignore */
    return kt>0?kt:1024;
}
/* Bb[ns*Sk+ks] = K-split x N-split (always). Bf[ns] = optional full-K per N-slice (ORK_FULLK_DEC,
 * int8 K<=10752): lets the multi-core DECODE path do ONE submit/core instead of ~K/1024 K-slices.
 * ~2x weight memory (dual layout) — fits IOVA for int8 ~1.7B; can overflow for larger/fp16. */
/* Auto-tuner policy. Multi-core + full-K decode are now the library's DEFAULT per-matmul choice
 * (no env needed); the engine sets a core budget via ork_npu_set_core_budget.
 * The driver automatically selects the core count by N-tile count, and uses full-K single-submits
 * when M is small, K<=10752, precision is int8, and it fits within IOVA. */
int orki_budget(ork_npu*c, int M){
    /* M=1 is single-core by default here; the int8 DECODE path in orki_run() overrides to the multi-core budget
     * (it calls orki_budget(c,2)) — splitting N across cores parallelizes the cold per-token weight-DMA, measured
     * +40% end-to-end decode (Qwen3-1.7B) and MONOTONIC in the in-model N-sweep: every int8 shape benefits,
     * so there is no per-shape threshold and no env knob (the old ORK_DECODE_MC gate is gone). mc_prof's warm
     * loop had mis-measured a crossover that doesn't exist in real cold decode. fp16/int4 M==1 stay single-core
     * (fp16 large-tile M-scheduler unvalidated; int4 not the decode path). The NN<nc*2 shrink at the call site
     * still keeps truly tiny int8 N single-core. */
    if (M == 1) return 1;
    int b=c->core_budget;
    const char *env_mc = getenv("ORK_NPU_MC");
    if (env_mc) {
        int env_val = atoi(env_mc);
        if (env_val >= 1 && env_val <= c->soc->cores) {
            b = env_val;
        }
    }
    if(b>c->soc->cores)b=c->soc->cores;
    if(b<1)b=1;
    return b;
} /* effective max cores */


/* PER-WEIGHT IOMMU DOMAIN. The rk_iommu v2 32-bit IOVA cap (~4 GiB) is per iommu_domain_id, not per
 * device, so spreading weights over multiple domains keeps >4 GiB resident at once. ORK_IOMMU_DOMAIN
 * is the process-wide DEFAULT domain (env, default 0). Each ork_w then carries its own `domain`:
 *   - bcreate/bimport take the domain as a PARAMETER — pass w->domain when allocating that weight's
 *     resident tiles so they land in the chosen domain.
 *   - rknpu_submit_ioctl takes the domain as a PARAMETER and stamps sub->iommu_domain_id — pass
 *     w->domain so the submit runs against the same domain the weight lives in. The per-submit struct
 *     is per-call/per-stack, so concurrent multi-core workers each carry their own domain (no global).
 * Activation/output/scratch buffers are allocated under the active domain (dom_activate); resident
 * weights and their submits dominate the IOVA budget, so per-weight placement is what matters. */
/* THREAD-SAFETY: the IOMMU domain is threaded through call parameters and the per-submit
 * rknpu_submit struct, NOT a process-global — two concurrent submits / packs must not race a
 * shared mutable domain. bcreate/bimport take an explicit `domain`; rknpu_submit_ioctl sets
 * sub->iommu_domain_id from a parameter. The pack-path default for the ggml-ork caller lives on
 * the ork_npu ctx (c->pack_domain), read once per pack to stamp w->domain. dom<0 => default. */
int ork_dom(int dom){ return dom>=0 ? dom : ork_dom_default(); }
/* ---- IOVA WEDGE GUARD -----------------------------------------------------------------------
 * The rk_iommu v2 IOVA window is 32-bit (~4 GiB) PER iommu_domain_id, and the kernel rknpu driver
 * FAULTS inside MEM_CREATE (rknpu_iommu_dma_map_sg -> rknpu_gem_object_create) when that window is
 * exhausted/fragmented, rather than returning -ENOMEM — an in-syscall OOPS that hard-wedges the NPU
 * (recoverable only by reboot; a power-cut mid-wedge risks SPI-bootloader corruption). A userspace
 * return-value check on MEM_CREATE therefore cannot prevent it (the kernel dies before returning).
 * So we account mapped bytes PER DOMAIN and REFUSE the allocation in userspace before issuing
 * MEM_CREATE once a per-domain safe ceiling would be exceeded — the allocating call then returns
 * {0} (cpu==NULL) and its caller falls back (CPU) cleanly. Ceiling default 3900 MiB (headroom under
 * the 4 GiB cap for fragmentation + kernel overhead); tune with ORK_IOVA_CEIL_MB. This guards EVERY
 * DMA-allocating path (weights, scratch, PPU-op buffers, zero-copy imports), not just one op. */
size_t orki_iova_bytes[ORK_IOVA_NDOM];   /* MEM_CREATE-mapped bytes currently live, per iommu domain */
long orki_bcreate_n, orki_bimport_n, orki_bdestroy_n;   /* cumulative alloc/free counts (ORK_PRESUBMIT_TRACE leak diag) */
/* ORK_IMPORT_TRACE: flushed per-phase trace of every dma-buf IMPORT (bimport) + the weight-level import
 * entrypoints. Each line is fflush'd so if an ioctl HANGS (D-state), the LAST printed line names the exact
 * stuck phase (DMA_HEAP_ALLOC / mmap / PRIME_FD / MEM_CREATE), domain, size, and cumulative counts. Off by
 * default (0 cost). Used to root-cause the weight-import D-state hang without guessing. */
int orki_imp_trace(void){ static int t=-1; if(t<0){ const char*e=getenv("ORK_IMPORT_TRACE"); t=e?atoi(e):0; } return t; }
long orki_imp_wn;   /* weight-import call counter (which weight is being imported when a hang hits) */
/* NPU on-chip SRAM total (bytes), queried once at init. 0 => the kernel/DTB did NOT allocate SRAM to the
 * NPU (stock config, no CONFIG_ROCKCHIP_RKNPU_SRAM / no rkvdec0_sram reassignment). bcreate uses this to
 * fail a TRY_ALLOC_SRAM request over to DRAM so the submit path is portable across kernels/DTBs. */
uint64_t orki_sram_total = 0;
/* reserve `need` bytes in domain `dom`; 1 = ok (accounted), 0 = would exceed cap (caller must not alloc). */
/* ---- graceful-teardown live-buffer registry (SIGTERM/SIGINT) --------------------------------------
 * A killed process (e.g. `timeout`) that skips MEM_DESTROY leaks its IOMMU domain/IOVA mappings — benign
 * for a single domain, but a MULTI-DOMAIN run strands whole 4 GiB domains until reboot (the kernel's
 * implicit fd-close cleanup doesn't reclaim the domain allocator state). So track every live handle
 * (bcreate + bimport) and, on SIGTERM/SIGINT, MEM_DESTROY them all + ACT_RESET before re-raising the
 * default disposition — the same reclaim a clean exit does. Disable with ORK_NO_SIGCLEAN=1. */
struct ork_live_ent { uint32_t handle; uint64_t obj; };
struct ork_live_ent *orki_live=NULL; int orki_live_n=0, orki_live_cap=0; int orki_live_fd=-1;
pthread_mutex_t orki_live_mu=PTHREAD_MUTEX_INITIALIZER;
void orki_live_add(int fd, uint32_t h, uint64_t o){ pthread_mutex_lock(&orki_live_mu); orki_live_fd=fd;
    if(orki_live_n==orki_live_cap){ int nc=orki_live_cap?orki_live_cap*2:256; void*p=realloc(orki_live,(size_t)nc*sizeof*orki_live); if(p){orki_live=p;orki_live_cap=nc;} }
    if(orki_live_n<orki_live_cap) orki_live[orki_live_n++]=(struct ork_live_ent){h,o};
    pthread_mutex_unlock(&orki_live_mu); }
void orki_live_del(uint32_t h){ pthread_mutex_lock(&orki_live_mu);
    for(int i=0;i<orki_live_n;i++) if(orki_live[i].handle==h){ orki_live[i]=orki_live[--orki_live_n]; break; }
    pthread_mutex_unlock(&orki_live_mu); }
/* IMPORT REGISTRY (fd-reap recovery, task #47). A parallel registry holding POINTERS to every dma-buf-IMPORTED
 * buf (bimport / bimport_fd) — the ones whose backing pages PERSIST across a DRM-fd close (the client/heap holds
 * the dma-buf fd). ork_ctx_fd_reap() walks this to RE-IMPORT each buffer into the reopened fd and rewrite its
 * handle/IOVA/obj/cpu IN PLACE, so resident weights survive a close+reopen with NO re-orki_pack (only bcreate'd buffers
 * — scratch — are lost, and those lazily re-create). Pointers are stable (Bb[]/own_bufs[] are calloc'd arrays);
 * bdestroy unregisters. Values-registry orki_live is for SIGTERM MEM_DESTROY; this one is for reap re-import. */
struct buf **orki_imp=NULL; int orki_imp_n=0, orki_imp_cap=0;
static void orki_imp_reg(struct buf*b){ pthread_mutex_lock(&orki_live_mu);
    for(int i=0;i<orki_imp_n;i++) if(orki_imp[i]==b){ pthread_mutex_unlock(&orki_live_mu); return; }   /* dedupe */
    if(orki_imp_n==orki_imp_cap){ int nc=orki_imp_cap?orki_imp_cap*2:128; void*p=realloc(orki_imp,(size_t)nc*sizeof*orki_imp); if(p){orki_imp=p;orki_imp_cap=nc;} }
    if(orki_imp_n<orki_imp_cap) orki_imp[orki_imp_n++]=b;
    pthread_mutex_unlock(&orki_live_mu); }
static void orki_imp_unreg(struct buf*b){ pthread_mutex_lock(&orki_live_mu);
    for(int i=0;i<orki_imp_n;i++) if(orki_imp[i]==b){ orki_imp[i]=orki_imp[--orki_imp_n]; break; }
    pthread_mutex_unlock(&orki_live_mu); }
volatile sig_atomic_t orki_sig_busy=0;
void ork_sig_teardown(int sig){
    if(!orki_sig_busy){ orki_sig_busy=1; int fd=orki_live_fd;   /* best-effort: process is terminating, no lock (races benign) */
        if(fd>=0){ int n=orki_live_n;
            for(int i=0;i<n;i++){ struct rknpu_mem_destroy d; memset(&d,0,sizeof d); d.handle=orki_live[i].handle; d.obj_addr=orki_live[i].obj; ioctl(fd,DRM_IOCTL_RKNPU_MEM_DESTROY,&d); }
            struct rknpu_action a; memset(&a,0,sizeof a); a.flags=RKNPU_ACT_RESET; ioctl(fd,DRM_IOCTL_RKNPU_ACTION,&a); } }
    signal(sig,SIG_DFL); raise(sig);
}
void orki_bdestroy(int fd,struct buf*b){ if(!b->cpu)return; munmap(b->cpu,b->size);
    struct rknpu_mem_destroy d; memset(&d,0,sizeof d); d.handle=b->handle; d.obj_addr=b->obj; ioctl(fd,DRM_IOCTL_RKNPU_MEM_DESTROY,&d);
    orki_live_del(b->handle); orki_imp_unreg(b);   /* drop the (now-dangling) buf* from the fd-reap import registry */
    orki_bdestroy_n++;
    ork_iova_release(b->domain,b->size);
    if(b->heap_fd>0){ close(b->heap_fd); b->heap_fd=0; } b->cpu=0; }
/* Ensure the persistent SDP-op scratch (a/b/out) is each >= sz bytes; (re)allocate only when it must grow.
 * Reused across every ewmul/add call so the per-op MEM_CREATE/MEM_DESTROY churn (which dominates the
 * standalone-op cost and fragments the IOVA window) is paid ONCE, not per call. 0 on success, -1 on alloc
 * failure (caller returns an error -> its caller falls back to CPU). Freed in ork_npu_free. */

/* Zero-copy IMPORT (no page alloc, no copy): allocate a dma-buf from /dev/dma_heap/system, mmap it,
 * import it into the NPU's IOMMU domain via PRIME_FD_TO_HANDLE -> MEM_CREATE(handle, flags=0, size=0).
 * The kernel maps the EXISTING dma-buf pages and returns the IOVA (dma_addr) to put in the regcmd.
 * Caller fills *cpu with the (pre-tiled) bytes, then issues a MEM_SYNC clean (bsync TO_DEVICE) before
 * the first submit. Returns a buf whose heap_fd holds the dma-buf fd (closed by bdestroy). On any
 * failure returns {0} (cpu==NULL). orki_dmaheap_fd is the cached /dev/dma_heap/system fd (-1 = unopened,
 * -2 = open failed; import then unavailable and callers fall back to bcreate). */
int orki_dmaheap_fd = -1;
/* dma-buf CPU-access cache sync on the dma-buf fd (flushes CPU caches for an imported cacheable
 * buffer — the rknpu MEM_SYNC does not cover foreign imports). Bracket the CPU fill: START|WRITE
 * before, END|WRITE after. No-op if the buffer wasn't imported (heap_fd<=0). */
/* Run-SCRATCH allocator (task/regcmd/output buffers). int4-RESIDENT runs keep every domain's WEIGHTS bimported
 * (PRIME_FD); a fresh MEM_CREATE (bcreate) then EINVALs in that domain (MEM_CREATE GEM alloc can't coexist with
 * imported memory in the same iommu domain), which starved the run scratch (mc_ensure mtk_all -> decode -3). So
 * for int4 (c->last_dt==DT_I4) route the small scratch through the SAME import path (bimport) as the weights, so
 * it coexists in-domain. int8/fp16 are UNCHANGED (bcreate). `flags` is subsumed under import (bimport's MEM_CREATE
 * uses flags=0). NOTE: at ork_npu_init/first domain-0 touch last_dt is COLD (not DT_I4) so domain 0's init scratch
 * still bcreate's into the empty domain (fine); the int4 switch only applies once an int4 run is active. */
/* Like bimport but imports an ALREADY-EXISTING dma-buf fd (e.g. one received over SCM_RIGHTS from another
 * process) instead of allocating from the heap. Takes ownership of `dbuf` (bdestroy closes it via heap_fd).
 * This is the cross-process zero-copy primitive behind ork_dma_import_fd (the orkd daemon's data plane). */
/* ESTABLISH a non-0 IOMMU domain with a small NATIVE allocation before any dma-buf import is mapped into
 * it. The kernel rknpu driver lazily sets up a domain's IOVA allocator / page table on its FIRST buffer;
 * if that first buffer is an IMPORTED dma-buf, the import's SG-list pages get wrong/aliased IOVAs and the
 * NPU reads garbage for some weight tiles (non-deterministic dropped-K corruption — reproduced with
 * tools/mc_import_probe.c: import-first FAULTS, native-alloc-first is bit-exact; single- AND multi-core).
 * One tiny native anchor per domain, kept resident for the ctx lifetime (freed at teardown). No-op for
 * domain 0 (always established) and when already anchored. Call BEFORE the first bimport into `dom`. */
/* Grow the per-domain arrays (native anchor + parked scratch) to hold at least `need` domains. No fixed
 * cap — the domain count is whatever the auto-sizer / ork_npu_domain_alloc drives. dom_save is allocated
 * here (so it becomes non-NULL exactly when multi-domain is first entered, preserving the single-vs-multi
 * signal the run paths key on). Called from every multi-domain entry: dom_activate, ork_dom_prime,
 * ork_npu_domain_alloc, ork_npu_set_ndomains. Returns 0 ok, -1 on OOM (caller degrades gracefully). */
/* Public: pre-size the ctx for `n` IOMMU domains — the backend calls this once with the auto-sizer's
 * n_domains so no per-weight grow happens mid-load. Only meaningful for n>1 (n<=1 = single-domain, leaves
 * dom_save NULL). Safe to call repeatedly; only ever grows. */
/* #54 FIX: re-establish a domain's IOMMU page-table region before EACH imported dma-buf. The kernel sets up a
 * domain's page table lazily around the buffer that triggers it; the one up-front anchor (ork_dom_prime) covers
 * only the FIRST import — a 2nd+ imported dma-buf then lands on aliased IOVAs and the NPU reads it WRONG (probed
 * bit-exact: 1st import OK, 2nd import maxerr~2835, fixed to 0 by a fresh native bcreate before it). So drop the
 * stale anchor and bcreate a fresh native one immediately before importing each weight. Cheap (64 KiB); the
 * previous import's mapping persists after its anchor is freed (verified: 1st weight re-runs bit-exact after). */
/* #54 REAP-AT-DOMAIN-BOUNDARY. A genuine int4 doorbell DROP leaves a stuck job in c->dom_active whose completion
 * IRQ never fired, so the kernel's iommu_domain_refcount for that domain stays >0. The reap that clears it —
 * rknpu_job_timeout_clean — only fires at the TOP of the NEXT submit ON THAT CORE, and (crucially) that next
 * submit is normally in the NEXT domain, AFTER the switch: so get_and_switch(D+1) waits on D's refcount>0 and
 * TIMES OUT ("switch iommu domain time out, id: N") — the reap can never happen across the boundary. FIX: before
 * switching away from a dirty domain, issue a SAME-DOMAIN per-core dummy (ork_npu_reap_stuck) whose submit runs
 * timeout_clean(core i) -> clean gem_object_put of the stuck job -> refcount returns to 0, so the switch lands.
 * (ACT_RESET does NOT do this — source-confirmed rknpu_soft_reset is HW-only, leaves the job list; only
 * timeout_clean reaps. That's why the earlier ACT_RESET version failed.) No-op unless a real drop set dom_dirty
 * (rare); clean runs + single-domain pay nothing. Between-ops / pre-teardown only (no live pool workers). */
/* RE fuzzer hook for fp16 (batch-mode mapping): overrides applied at the END of orki_synth(). Inert by default. */
/* sched=1: single-submit internal M-scheduler (clean power-of-2 Kp); sched=0: one M-tile. */
/* synth_i16 — int16 matmul regcmd (emulated W16A16 coherence layer). Same 2-BYTE geometry as the fp16 synth
 * (int16 tiles are byte-identical layout to fp16), but the CNA precision (REG_CNA_CONV_CON1 @ 0x100c) is
 * flipped FP16(proc=in=2, 0x20000120) -> INT16(proc=in=1, 0x20000090). CONFIRMED on-board: proc=1 runs on the
 * 2-byte geometry (hangs on int8's 1-byte). Integer datapath => NOTHRASH-stable with int8 layers, rides the
 * fp16 matmul scaffolding. Output stage: start with the fp16 template's (empirical), refine to integer if the
 * accumulate/convert needs it. ORK_I16_CON1 overrides 0x100c for on-board RE of the exact int16 encoding. */
/* RE fuzzer hook for int8 (tools; batch-mode RE): (block,reg,val) overrides applied at the END of synth_i8.
 * Inert by default (n=0) — production unaffected. Mirrors the int4 orki_i4_fovr hooks. */
/* int8/w8a8: A,B int8 -> C int32. Differs from fp16: weight amount/stride (no x2), K-passes
 * ceil(K/64), 0x107c=K/16, rows-budget 2x (int8 packs 2x rows/CBUF), and the 0x1040 schedule
 * uses effective K = K/2. cbuf is the fp16 budget; int8 rows = 2*cbuf/K. */
/* #39 PORT (RE): dump ork's synth_i8 regcmd words for (mc,K,N) with sentinel addresses, so a tool can diff
 * against rkllm's captured regcmd and read off the exact field delta (schedule + feature-layout) the M-fold
 * variant must apply. Returns word count (REGCMD_I8_N), or <0 on error. */
/* #39 M-FOLD variant — REGISTER-LEVEL CLONE of rkllm's CAPTURED mfold regcmd (M=36 K=3584 N=1216), decoded
 * with tools/re/regcmd_decode against ork_regs.h. Folds M into the CNA WIDTH dim; output int32 NC1HWC2.
 * MEASURED 1.66x ork's own kernel (rkllm regcmd 433 vs ork 719 us/matmul). Built as a DELTA on synth_i8.
 * Corrections from the real capture (supersede earlier RE guesses):
 *   - 0x100c = OKV_CONV1_GROUP_LINE (GROUP_LINE_OFF bit 29 SET — the fold's feature-read mode, NOT an int32-out
 *     select; precision is CONV_CON1[9:4], int8=0). The fold SETS it; the old "clear-or-STALL" belief
 *     was an artifact of a broken standalone submit.
 *   - 0x1080 = REAL surface stride 2160/M (60@M36, 108@M20), NOT the 0x0fffffe8 sentinel — that value is
 *     rkllm's M=8 chain-TAIL "inherit CBUF layout" marker and HARD-WEDGES a standalone submit (OOB DMA).
 *   - 0x4010 = int32 accumulate-out; plus the extra CNA regs rkllm sets (0x1014/104c/1050-5c/1078).
 * OPEN (validated at M=36 ONLY): the CBUF/DMA schedule regs 0x1040/0x107c/0x4024/0x40c0 are rkllm's M=36
 * LITERALS; their per-M formula still needs more captures. And BIT-EXACT additionally requires the caller's
 * A-pack + C de-tile to match rkllm's fold layout (rkllm's stride-60 A is NOT plain contiguous NC1HWC2).
 * The "2160" is capture-specific (K=3584,N=1216) pending a K/N-general derivation. GATED: only synthesized on
 * the explicit m-fold path; the default matmul is untouched. */
unsigned orki_mm_timeout_ms(void);   /* fwd (defined below) */
/* #39 WEIGHT-RESIDENT M-FOLD CHAIN using a CAPTURED bit-exact tile regcmd (task #39: "chain the captured m8
 * tile"). Identical machinery to ork_npu_mfold_chain, but each of P tasks is a memcpy of `tile_rc` (rkllm's
 * captured width-`w` mfold regcmd, e.g. mm_regcmd_m8.txt at w=8 — validated 0/9728 bit-exact by validate_layout)
 * with only the 3 address regs re-based per tile. This sidesteps orki_synth_i8_mfold's schedule (which reproduces
 * rkllm only via the per-M recipe); the captured tile carries the real planner schedule verbatim. Words 0..215
 * are the tile (108 regs); trn may be >216 (the capture's next-task bleed) — only the tile words are copied, and
 * the chain descriptor is written fresh at 216-219. 0x40c0 (=0x400 config, NOT an IOVA) is preserved untouched. */
/* #39 TIMING PROBE: replay P DIFFERENT captured tiles (tiles = P*rn words, e.g. rkllm's real chain tasks IN
 * ORDER) as one task_number=P weight-resident chain, rebasing each tile's A/W/C onto shared buffers (uniform
 * max-M stride; ZEROED operands — this measures re-DMA vs reuse, not correctness). Shared weight buffer. Returns
 * 0/ok, us=avg submit. The timing SLOPE across P answers: does each task re-stream the weight (~linear) or reuse
 * the resident copy (sublinear once a loader task has populated CBUF)? w = max tile M (for buffer sizing). */
/* #39 UNIFIED per-tile fold chain with REAL operands + Craw return — the tool for Path-1 (state-setter + big-M)
 * and Path-2 (weight-reuse). P tiles, per-tile width ws[t], per-tile captured regcmd tiles[t*rn..]. A packed as
 * concatenated per-tile nc16 (tile t = ws[t] rows, offset = sum of prior ws*K bytes); shared woff weight Bpacked;
 * Craw = concatenated per-tile c4 (ws[t]*N int32 each). wreuse: OR WEIGHT_REUSE(0x1040 bit13) into tiles t>0.
 * Rebases A/W/C per tile onto shared guard buffers preserving the concat layout. Returns 0/ok, us=avg submit. */
/* #39 Path-1 TOKEN-TILER executor: run P fold sub-tiles of one M_total-token batch as ONE multi-task submit over a
 * SHARED batch cube. Unlike ork_npu_mfold_chain_v (concatenated per-tile buffers), all tiles read/write the SAME
 * M_total x K input and M_total x N output; tile t handles rows [row_off[t], row_off[t]+m) at feature/output byte
 * offset row_off[t]*16 (nc16 in / c4 out both stride rows by 16 bytes). The caller prepares each tile's regcmd
 * (per-size skeleton with the 4 M_total regs patched — 0x4024=16*M_total, 0x107c=M_total, 0x1080=M_total-m,
 * 0x40c0=128*M_total — plus the output-stage regs + doorbell). Shared weight (0x1110). This is rkllm's real fold:
 * a batch amortized over few big-M tiles, one weight stream per tile. Apacked = M_total x K nc16 (width M_total);
 * Bpacked = K x N woff; Craw = M_total x N c4 (width M_total). Returns 0/ok, us=avg submit. */
/* #39 per-core fold submit: one core's task-group, core_mask=1u<<core, own task buffer (tasks from index 0),
 * subcore_task[*]={0,P} — EXACTLY the proven mcworker per-core pattern. Run one per thread => concurrent 3-core. */

void *ork_fbc_thread(void *vp){
    struct ork_fbc_arg *a=vp; struct rknpu_submit sub; memset(&sub,0,sizeof sub);
    sub.flags=ork_ppflags(); sub.task_number=(uint32_t)a->P; sub.task_obj_addr=a->tk->obj; sub.fence_fd=-1;
    sub.core_mask=1u<<a->core;
    sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)a->P};
    sub.timeout=orki_mm_timeout_ms();
    a->rc = orki_rknpu_submit_ioctl(a->fd,&sub,a->dom);
    return NULL;
}
/* fire the nc per-core submits CONCURRENTLY (one thread each) and wait; 0 ok, -1 if any core errored */
#include "regcmd_fold_refs.h"
/* #39 fold run-path helpers: build a size-m sub-tile from the baked template, patch its 4 M_total regs. */
/* #39 FOLD MATMUL RUN-PATH: C[M,N] int32 = A[M,K] int8 x W[K,N] int8 via rkllm's token-fold — M(<=128) tiled into
 * <=36-row sub-tiles, ALL run in ONE shared-cube multi-task submit (amortizes ioctl over the whole batch). Uses the
 * baked per-size templates (regcmd_fold_refs.h), so only K=FOLD_REF_K,N=FOLD_REF_N,M<=128 is supported — returns
 * -1 otherwise so the caller keeps the standard synth_i8 path. A row-major [M,K], W row-major [K,N], Cout row-major
 * [M,N]. Single core (core_mask CORE0). Returns 0/ok, us=avg submit; -1 unsupported shape, -2 alloc/build, -3 no fd. */
/* #39 FULL-OP fold: C[M,N] int32 = A[M,K] x W[K,N] int8 via N-SPLIT ACROSS CORES — each core single-core-folds its
 * own <=1216-wide N-slice (own weight columns + output columns, shared A input) CONCURRENTLY. This is rkllm's real
 * wide-projection scheme, and the honest end-to-end test: does the fold's compute-hiding beat its extra weight
 * re-streams (36-row tiles) under 3-core DRAM-bandwidth contention (3 cores streaming 3 different slice weights)?
 * K=FOLD_REF_K, N<=3*1216 (<=3 slices, one core each), M<=128. A/Cout row-major. 0/ok, -1 unsupported, -2 alloc. */
/* #39 RESIDENT mfold SCRATCH. Split into (a) the SHARED nc16 input buffer c->fold_A (keyed M,domain) marshaled
 * ONCE per input and reused by every same-input weight (the shared-input fold batch), and (b) a small cache
 * c->fold_scr[] of per-(M,N,domain) C(c4)/RC(regcmd)/TK(task)+tiles so q's N=3584 and k/v's N=512 coexist without
 * thrash. Building these per call (~8ms, ~10 MB-bufs) is what made the naive run-path fold 8x slower; here it's
 * paid ONCE per shape. Only the weight/A/C addrs in RC are patched per call. */

/* Ensure the shared nc16 input buffer exists for (M,domain); rebuild on key change. Then fold_fill_A marshals
 * the row-major Araw into it (once per input — the batch reuses it across weights). */
/* Get (or lazily build + cache) the per-(M,N,domain) C/RC/TK scratch. TK built once (references only RC offsets). */
/* patch RC (weight/A/C addrs) for `w` into scratch `fs`, submit the fold (rounds of <=3 cores), detile into Cout
 * (NULL to skip, e.g. timing). Uses the SHARED c->fold_A (already filled by the caller). 0 ok, -2 submit fail. */
/* #39 mfold RUN-PATH (single weight): marshal A once, run the fold from the resident w->Bfold. 0 ok, -1/-2/-3. */
/* #39 SHARED-INPUT fold BATCH: nw weights (all K=FOLD_REF_K, resident Bfold, same domain) that consume the SAME
 * input A[M,K] — QKV, or gate+up. The nc16 input marshal is done ONCE and reused across all nw folds (that marshal
 * was ~half the per-op tax; amortizing it is the point). Couts[i] row-major (NULL entries skipped). 0 ok, -1/-2/-3. */
/* #39 Path-1 CANONICAL OUTPUT-STAGE STATE-SETTER. The full-prefill sweep (tools/re full_sdp.py + full_regmap.py
 * over pf.dump, 120,923+ tiles) proved the output stage is ONE invariant config for every int8 fold matmul,
 * spread across TWO blocks: the DPU/SDP block 0x1001 AND the PDP/aux output-dims mirror block 0x801. In both,
 * every functional register holds a single value across ALL tiles; only the geometry registers vary with (M,N).
 * sdp_canon() returns that first-principles value for any 0x1001/0x801 register — no captured blob.
 * ork_npu_sdp_stamp() rewrites the value of EVERY 0x1001/0x801 register present in a REGCMD_I8_N regcmd to its
 * canonical value (leaving DST_BASE_ADDR 0x4020 for the caller's C IOVA — the only output-stage address), so a
 * proven-runnable fold skeleton whose 0x1001+0x801 blocks are zeroed gets its ENTIRE output stage rebuilt from
 * understood values. This is the "state-setter" a delta-encoded, register-inheriting big-M tile depends on (NVDLA
 * register-file persistence). surfadd = 0x40c0 SURFACE_ADD (128*M for matched small-M tiles; the burst-regime
 * value for big-M, e.g. 0x3000 at M=36 — see full_sdp.py's 0x40c0-by-M histogram). */

/* ── On-NPU (PPU) fused output stage — additive, GATED, CPU/NEON path stays the default ──────────
 *
 * The synth_i8 template above emits INT8_MM_INT8_TO_INT32: the matmul writes a full-precision int32
 * accumulator to C, and the caller does requantize + activation (SiLU) + the SwiGLU elementwise-mul
 * on the CPU/NEON (src/neon_activations.c). The PPU can instead do those in the matmul's OUTPUT STAGE
 * on-chip (RE'd 2026-07-03, see PPU_FUSED_ACTIVATION_WIP.md): int8-output requantize (this helper),
 * a fused activation LUT (enable=0x18 SiLU), and a dual-input elementwise-multiply. Fusing them turns
 * the FFN inner into an on-NPU chain and removes the per-op CPU round-trip (the M=1 decode bubble).
 *
 * WHY THE CPU/NEON PATH IS KEPT (not replaced): the fused output stage is HARDWARE-SPECIFIC to the
 * RK3588-family PPU register layout decoded here. The CPU/NEON requant+SiLU+mul path is retained
 * DELIBERATELY as the default fallback so that:
 *   (1) PORTABILITY — a future/different NPU (or an RK35xx whose PPU output stage differs) that we
 *       have not yet reverse-engineered can still run w8a8 correctly via the CPU intermediary; the
 *       fused path is an optimization layered on top, never a hard dependency.
 *   (2) BRING-UP — while the fused regcmd path is being validated bit-exact on silicon, the CPU path
 *       is the known-good route; the gate lets us ship the matmul without blocking on PPU fusion.
 *   (3) VALIDATION — it is the bit-exact reference the fused-output tests diff against.
 * Hence the gate (ork_ppu_fuse_enabled): callers/higher layers opt IN to the fused path; when it is
 * off (the default), or the SoC is not a validated PPU target, ork-driver emits the int32 output and
 * the caller's CPU/NEON stage runs — identical numerics, no behavior change for existing callers.
 *
 * set_i8_out8 rewrites a synth_i8'd regcmd's output stage from int32 to int8-requantized. From the
 * matmul INT8_TO_INT32 vs INT8_TO_INT8 capture diff (mm_out_probe): exactly 6 regs change (the rest —
 * addresses, M/N counts — are shared). The hardware requant is
 *   out_i8 = clamp_i8( round_half_to_even(acc_i32 * mult / 2^shift) )
 * — convergent (banker's) rounding, verified bit-exact on silicon; identity = mult=0x4000, shift=14.
 * stride = C row stride in ELEMENTS (defaults to N). NOTE: the int8-output stage rides on synth_i8's
 * compute regcmd, whose 0x1040 K-reduction schedule is validated for K>=512 (the prefill domain); at
 * tiny K (<=128) that schedule's small-K branch miscomputes and the requant reads garbage. FFN/attn
 * matmuls are all K>=512, so this is not a practical limit. */
/* PHASE 1 (#35 chained FFN): INT16 matmul output stage — requant int32 acc -> int16 (scale mult/2^shift),
 * 2-byte elements. Between int8 (0x40c0=0x20,0x4050=0x0124) and int32 (0x40c0=0x80,0x4050=0x07fc); int16 is
 * the 2x-denser-than-int32 midpoint. THESE ARE BEST-GUESS and env-overridable (ORK_I16OUT_*) so the encoding
 * can be SWEPT on-board (like i16_matmul_test did for the fp16/int16 output regs). This is the RE crux: the
 * on-NPU int32->int16 requant that lets the matmul feed the int16 silu inside one chain. */

/* PHASE 1 (#35 chained FFN) — the SHIM the user asked for: instead of the CLOSED int16-matmul-output,
 * graft the fp16 REGCMD template's fp16 OUT_CVT (0x4010=0xa8000002, 0x40c0=0x40=2-byte, 0x4050=0x36e,
 * gain 0x4084=1/shift 0) onto an INT8 matmul's output stage — INT8_TO_FP16. If the int32 accumulator
 * survives the fp16 CVT this yields a fast int8 matmul writing a COHERENT fp16 intermediate (no
 * per-tensor requant floor) for the fp16 silu. WEDGE-PRONE (proc-precision mismatch, per set_f16_silu
 * warning); env-overridable for the sweep. N-derived geometry, constants from regcmd_array_4x32x16.h. */

/* fp16-IN fp16-OUT DPU output stage, reconstructed from the VENDOR conv task[0] (conv_mul.rknn, decoded against
 * rocket_registers.h) — the config that actually emits fp16 to memory AND hands off cleanly to a chained fp16 SDP.
 * orki_set_f16_out (int8-tuned) hangs the fp16 matmul: it leaves the BS/BN/EW ALU stages active and — critically —
 * writes 0x4084=1 WITHOUT DPU_OUT_CVT_SCALE.FP32TOFP16_EN (bit16), so the fp16 CVT is never enabled. Here we take
 * the vendor's mode/bypass/CVT registers verbatim and keep only the matmul-shaped output GEOMETRY (N channels). */

/* Runtime gate for the PPU fused-output path. Gated on the DETECTED SoC: the fused output stage (int8
 * requantize, SiLU LUT, dual-input EW-mul) is reverse-engineered and validated against the RK3588 PPU
 * register layout; on any other chip the layout may differ, so we fall back to the portable, known-good
 * CPU/NEON requant+activation path (see the block above). No env var — the chipset is detected once at
 * startup (ork_soc_detect via device-tree) and this returns 1 only on rk3588. As the fused path is
 * validated on further SoCs, extend this check. */
int ork_ppu_fuse_enabled(ork_npu *c){
    return c && c->soc && c->soc->id && strcmp(c->soc->id, "rk3588") == 0;
}

/* set_i8_silu — fused SiLU output stage: the matmul applies SiLU on-chip via a PWL LUT in its output
 * stage (enable=0x1d on the task). Decoded + validated against silicon 2026-07-03 (staircase/const-LUT
 * probes + a composite model that reproduces the captured curve to ~2 int8; see PPU_FUSED_ACTIVATION_WIP.md).
 *
 * VALIDATED PIPELINE (the FIXED silu LUT is streamed separately by the REGCMD_SILU_LUT prologue into PPU
 * LUT SRAM; this program reads it):
 *     ii        = acc_i32 * R                      ; R = r_mult / 2^r_shift  (Q6 fixed-point index-input)
 *     idx       = ii >> 6                           ; 6 low bits = PWL interpolation weight
 *     V16       = LUT[idx]*(1-frac) + LUT[idx+1]*frac
 *     out_i8    = clamp_i8( R * V16 + out_bias )     ; SAME R gain; out_bias is the (asymmetric) zero-point
 *   R is a SINGLE scale knob (reg 0x4084 mantissa / 0x4088 shift): it sets BOTH the acc->index step AND
 *   the LUT->output gain. C0 (the silu-zero index ~512) rides on reg 0x4110 (idx_off); out_bias = reg 0x4080.
 *   The LUT is SCALE-INDEPENDENT (one fixed silu*S curve, 2 banks: neg half idx 0..511, pos half 512..1023).
 *
 * REGISTER ROLES (corrected — earlier labels were wrong): 0x4084/0x4088 = the unified scale R (NOT a
 * separate "requant"); 0x4060/0x4070/0x4108/0x410c/0x411c/0x4128/0x412c = FIXED config (scale-independent,
 * confirmed constant across a 6-model scale grid); 0x4010 high byte 0x44 + 0x4004/0x5004=0x30 = activation
 * enable; 0x4068 = a per-scale field with NO observed effect on the output (set to a replay-safe value).
 *
 * SCALE-DEPENDENCE (for a generator; not yet bit-exact-calibrated): R ~= 1/(S*out_scale), out_bias and C0
 * track out_scale. Until gen(in_scale,out_scale) is calibrated bit-exact, callers pass the register values
 * directly (a captured/known-good set). Params: r_mult,r_shift -> R (0x4084/0x4088); out_bias -> 0x4080;
 * idx_off -> 0x4110; cfg4068 -> 0x4068 (unobserved). */

/* ── Fused EW-mul (SwiGLU dual-input) output stage ───────────────────────────────────────────────
 * The SwiGLU inner is out = silu(gate(x)) ⊙ up(x). The SiLU half fuses into gate's output stage
 * (set_i8_silu). The elementwise-multiply by silu(gate) fuses into UP's output stage as a DUAL-INPUT
 * op: the PPU reads a SECOND input through a second DPU lane (regs 0x50xx, lane 0x2001) and multiplies
 * it into the requantized up_acc. Decoded from the RKNN SwiGLU capture (sw_up.reg EW-mul compute op):
 *   0x4070 = 0x904002c4      main-lane EW-mul enable/mode (plain=0x0302, silu=0x0383)
 *   0x50xx (18 regs)          second-input read surface; 0x5018 = its base address (= silu(gate) buffer)
 *   0x4100..0x412c = 0        NO LUT (pure multiply, not an activation)
 *
 * ork's REGCMD_I8 has NO 0x50xx lane (108 reg entries). synth_i8_ew SPLICES the captured
 * REGCMD_EW_LANE (18 entries) into a synth_i8'd regcmd, before its 8-word trailer -> 126 reg entries
 * (submit with regcfg_amount=126). set_i8_ewmul then patches the main-lane output stage (int8 requant +
 * EW-mul enable) and the 2nd-input address. The 0x50xx stride/partner-address fields are seeded from the
 * capture and refined by board matched-diff (see EWMUL_WIP.md) — first-run status is DIAGNOSTIC. */

/* EW-mul RE submit timeout (ms). ORK_EW_TIMEOUT lets the wedge-search fail fast (~1-2s) instead of 60s —
 * a working op completes in ~100us, so a short guard is safe; the kernel soft-resets on timeout either way. */
unsigned orki_ew_timeout_ms(void){ static int t=-1; if(t<0){const char*e=getenv("ORK_EW_TIMEOUT"); t=e?atoi(e):60000;} return (unsigned)(t>0?t:60000); }
/* Matmul-path submit timeout (ms). ORK_MM_TIMEOUT lets a mode-transition wedge-search fail fast (~1-2s)
 * instead of the 60s job timeout — the kernel soft-resets on timeout either way, so a short guard is safe
 * for RE. Default 60000 (unchanged production behavior). Mirrors orki_ew_timeout_ms for the run/chain paths. */
unsigned orki_mm_timeout_ms(void){ static int t=-1; if(t<0){const char*e=getenv("ORK_MM_TIMEOUT"); t=e?atoi(e):60000;} return (unsigned)(t>0?t:60000); }

/* Generalize the standalone element-wise MUL op (REGCMD_MUL{,_F16,_I16}) from the captured M=8,N=64 geometry
 * to arbitrary (M tokens = RDMA width, N channels). Derived by capturing the op at N=128 and M=16 and diffing:
 * M-dependent regs = M-1 (0x500c/0x4030/0x405c) and M*16 (strides 0x5040/0x4024/0x40c0 = EW_SURF_STRIDE /
 * output stride / SURFACE_ADD); N-dependent = N-1 (0x5014/0x4058) and (N-1)|(N-1)<<16 (0x403c). The channel
 * atom (16 for int8, 8 for the 2-byte fp16/int16) sets the cube; both give surf_stride = M*16 bytes. */
void orki_set_mul_geom(uint32_t *rc,int n,int M,int N){
    uint32_t sstride=(uint32_t)(M*16);
    orki_setrn(rc,n,RK_SDP_500C,(uint32_t)(M-1));          /* RDMA_DATA_CUBE_WIDTH  = M-1 */
    orki_setrn(rc,n,RK_SDP_5010,0);                        /* RDMA_DATA_CUBE_HEIGHT = 0 (H=1) */
    orki_setrn(rc,n,RK_SDP_5014,(uint32_t)(N-1));          /* RDMA_DATA_CUBE_CHANNEL= N-1 */
    orki_setrn(rc,n,RK_SDP_5040,sstride);                  /* RDMA_EW_SURF_STRIDE = M*16 */
    orki_setrn(rc,n,RK_DPU_DST_SURF_STRIDE,sstride);                  /* output surface stride */
    orki_setrn(rc,n,RK_DPU_DATA_CUBE_WIDTH,(uint32_t)(M-1));
    orki_setrn(rc,n,RK_DPU_DST_N_DIMS,(uint32_t)(((N-1)<<16)|(N-1)));
    orki_setrn(rc,n,RK_DPU_DST_N2,(uint32_t)(N-1));
    orki_setrn(rc,n,RK_DPU_WDMA_SIZE_1,(uint32_t)(M-1));
    orki_setrn(rc,n,RK_DPU_SURFACE_ADD,sstride);                  /* SURFACE_ADD = M*16 */
}

/* Apply ork's synth_i8 matmul GEOMETRY (same formulas as synth_i8, sched=1) onto an arbitrary regcmd `rc`
 * of length `n`. Used to inject ork's geometry into RKNN's EW-mul TEMPLATE (REGCMD_EWMUL_LIN) — which keeps
 * RKNN's register ORDER + EW output-stage/lane (so it executes) while making the conv engine read ork's own
 * [Nt][Kt][32][32] A/B tile layout (so acc is correct). Addresses (0x1070/0x1110/0x4020) patched by caller. */
void orki_apply_ork_geom(uint32_t*rc,int n,int mc,int K,int N,int cbuf){
    orki_setrn(rc,n,RK_CNA_DATA_SIZE1,((K-1)<<16)|K);orki_setrn(rc,n,RK_CNA_WEIGHT_SIZE0,K*N);orki_setrn(rc,n,RK_CNA_WEIGHT_SIZE1,K);
    orki_setrn(rc,n,RK_CNA_CBUF_CON1,(K+63)/64);orki_setrn(rc,n,RK_CNA_FC_DATA_SIZE1,K);orki_setrn(rc,n,RK_CNA_DMA_CON1,K/16);
    orki_setrn(rc,n,RK_CNA_DATA_SIZE0,0x10000|mc);orki_setrn(rc,n,RK_CNA_DATA_SIZE0_MIR,0x10000|mc);orki_setrn(rc,n,RK_CNA_DATA_SIZE3,mc);
    orki_setrn(rc,n,RK_DPU_DATA_CUBE_HEIGHT,mc-1);orki_setrn(rc,n,RK_DPU_WDMA_SIZE_1,(mc-1)<<16);orki_setrn(rc,n,RK_PDP_OUT_M,(mc-1)<<16);
    orki_setrn(rc,n,RK_DPU_DST_N_DIMS,((N-1)<<16)|(N-1));orki_setrn(rc,n,RK_DPU_DST_N2,N-1);orki_setrn(rc,n,RK_DPU_DATA_CUBE_NOTCH,(((N/4)-1)<<16)|((N/4)-1));
    orki_setrn(rc,n,RK_CNA_WEIGHT_SIZE2,0x1010000|N);orki_setrn(rc,n,RK_PDP_OUT_N,N-1);
    int R=(2*cbuf)/K; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
    int rows=(mc+1<R)?(mc+1):R; orki_setrn(rc,n,RK_CNA_CONV_CON2,16*rows);
    double scale=(double)K/512.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale),mg=(mc+63)/64; if(mg<1)mg=1;
    int v=base-slope*(mg-1); if(v<0x1b)v=0x1b; orki_setrn(rc,n,RK_CNA_CBUF_CON0,v);
}

/* Splice the 0x50xx second-DPU lane into a synth_i8'd matmul regcmd. base[] is a full REGCMD_I8_N buffer
 * already filled by orki_synth_i8 (108 reg entries in words 0..215, then the 8-word trailer). Output rc[] gets:
 * [108 reg entries] [REGCMD_EW_LANE 18 entries] [8-word trailer]. */
void orki_splice_ew_lane(uint32_t*rc,const uint32_t*base){
    memcpy(rc,               base,             216*4);                 /* 108 register entries (0x10xx/0x30xx/0x40xx) */
    memcpy(rc+216,           REGCMD_EW_LANE,   REGCMD_EW_LANE_N*4);    /* 18 second-lane entries (0x50xx) */
    memcpy(rc+216+REGCMD_EW_LANE_N, base+216,  8*4);                   /* the original end-of-regcmd trailer, now last */
}

/* set_i8_ewmul — NVDLA SDP element-wise MULTIPLY grafted onto ork's WORKING conv+int8-out program.
 * Recipe from the mesa "rocket" driver source (NVDLA-derived): ork's synth_i8+set_i8_out8 does the conv ->
 * SDP pipeline -> int8-out (OUT_CVT gain in 0x4084/0x4088). We ADD the SDP element-wise path: EW_CFG for a
 * multiply with the operand from the element-wise RDMA, and program the DPU_RDMA (0x50xx) to fetch the 2nd
 * input (silu(gate)) from aG at RDMA_EW_BASE_ADDR (0x5038). The DPU_RDMA block is APPENDED after 0x40xx
 * (rocket emits DPU then a contiguous DPU_RDMA block — append is valid). aG = silu(gate) dma base.
 * NVDLA fields (registers.xml): EW_CFG 0x4070 bits: EW_BYPASS(0) EW_OP_BYPASS(1) EW_OP_TYPE(2,1=mul)
 * EW_OP_SRC(6,1=rdma) EW_LUT_BYPASS(7) EW_OP_CVT_BYPASS(8) EW_RELU_BYPASS(9). EW_CVT 0x4074 offset /
 * 0x4078 {scale[0-15],shift[16-21]} scales the operand. RDMA: 0x5034 ERDMA_CFG(en+mode), 0x5038 EW_BASE,
 * 0x5040 EW_SURF_STRIDE, 0x500c/5010/5014 = W-1/H-1/C-1, 0x5068 {E,N,B,M}_WEIGHT. Submit enable_mask=0x1d. */

/* W4A4 (int4 A x int4 B -> int16 C) — uses the CAPTURED librknnrt regcmd verbatim (REGCMD_I4) as
 * the base (the real hardware program, not a guess), overriding only the K/N/address-dependent regs.
 * The precision regs (0x100c=0x360, 0x1080, 0x3010=0x601, 0x4010) stay as captured; K, N (≤nmax),
 * and the A/B/C addresses are parameterized. The captured program is M=1 (each task of the closed
 * runtime's M-tiling), so callers M-tile by looping rows. See ROADMAP. */
/* RE fuzzer hook (tools/i4_multim_fuzz.c): up to 16 (block,reg,val) overrides applied at the very END of
 * orki_synth_i4 (win over the K/N/mc-derived regs). Inert by default (n_on=0) — production is unaffected. Only
 * the fuzzer flips these on, so it can sweep the int4 regcmd space to crack the multi-M K-schedule wall. */
struct ork_regovr orki_i4_fovr[16]; int orki_i4_fovr_n=0;

/* Read-only sanity check: the benchmark methodology requires the DDR (dmc) governor at 'performance'
 * — a parked governor ~halves decode. We only WARN (never write; that needs root), so any caller
 * (llama-bench, the examples) notices a misconfigured box. Silence with ORK_NO_GOV_WARN=1. */

/* Free NPU on-chip SRAM bytes right now (0 if none). Confirms ORK_WEIGHT_SRAM actually placed a tile in SRAM
 * (free drops) for the CPU/NPU partition experiment. */
size_t ork_npu_sram_free(ork_npu *c){ if(!c) return 0; struct rknpu_action a; memset(&a,0,sizeof a); a.flags=RKNPU_GET_FREE_SRAM_SIZE;
    return ioctl(c->fd,DRM_IOCTL_RKNPU_ACTION,&a) ? 0 : a.value; }
/* Cumulative NPU DMA read/write byte counter (RKNPU_GET_TOTAL_RW_AMOUNT). Sample before/after a submit; a
 * ~0 delta means the HW did NO work (job never dispatched); a nonzero delta means it ran (did DMA). The key
 * signal for "did a failed round even reach the NPU?". 0 if unavailable. */
uint64_t ork_npu_dma_rw(ork_npu *c){ if(!c) return 0; struct rknpu_action a; memset(&a,0,sizeof a);
    a.flags=RKNPU_GET_TOTAL_RW_AMOUNT; return ioctl(c->fd,DRM_IOCTL_RKNPU_ACTION,&a) ? 0 : a.value; }
/* Snapshot the queryable NPU state (freq/volt/iommu/free-SRAM + cumulative DMA counters DT_rd/DT_wr/WT_rd/
 * total_rw) to stderr with a label. Call it when something goes awry (a round fails to land, a submit errors)
 * to capture state before a wedge/reboot destroys it. task_counter is a submit-time out field (0 for NONBLOCK)
 * so it is not queryable post-hoc; the DMA-amount deltas + the per-op output doorbells (ork_dyn_progress) are
 * the post-mortem signals. */
/* Emit a diagnostic line to /dev/kmsg so it rides netconsole OFF-BOX and survives a hard wedge that stdout/files can't
 * (a fatal wedge loses anything still on the board; kmsg->netconsole->Mac is already gone by then). Rare wedge-path only.
 * <4>=KERN_WARNING passes the default console loglevel. Best-effort: needs root (the tests run as root); silent if not. */
/* Soft-reset the NPU (RKNPU_ACT_RESET) and force a re-warm (clear c->warmed). Intended as the recovery step
 * AFTER ork_npu_dump_state when a round goes awry: it clears a stuck/faulted job so the bad state does not
 * accumulate into a hard wedge across repeated submits. Returns the ioctl result (0 ok). */
int ork_npu_soft_reset(ork_npu *c){ if(!c) return -1; struct rknpu_action a; memset(&a,0,sizeof a);
    a.flags=RKNPU_ACT_RESET; int r=ioctl(c->fd,DRM_IOCTL_RKNPU_ACTION,&a); c->warmed=0;
    for(int i=0;i<c->soc->cores;i++) c->mwarm[i]=0; return r; }
/* FD-REAP recovery (task #47): the ONLY nonblock-compatible CLEAN reap of a poisoned drop. close(fd) => drm_release
 * cancels ALL stuck jobs + tears down the whole IOMMU domain (device-global; the driver has no userspace job-abort,
 * and RKNPU_ACT_RESET does NOT release a dropped nonblock job's dangling mapping). We then reopen + re-init the
 * device and RE-IMPORT every registered dma-buf weight IN PLACE: its pages persist (client/heap holds the dma-buf fd)
 * and its CPU mmap is of the dma-buf fd (NOT the drm fd) so it SURVIVES the close — only the IOMMU mapping (drm-fd
 * MEM_CREATE) died, so re-import is just PRIME_FD_TO_HANDLE+MEM_CREATE (new IOVA/handle), no re-pack, no data copy.
 * bcreate'd scratch (regcmd/task/per-core mcc/maf/mrc/mtk/...) is LOST but lazily re-created (we zero it + its size
 * caches so the next mc_ensure/alloc rebuilds it on the fresh fd; warmed=0 forces the fp16 2-pass re-warm). Returns
 * 0 ok, <0 on reopen/re-import failure (context is then unusable). Caller must have quiesced all submits first. */
int orki_reap_n=0;
/* Recovery probe (the "dummy op") — NONBLOCK + HOST-BOUNDED, so it can NEVER hang on an already-wedged NPU.
 * A blocking submit on a wedged job enters the kernel's `continue wait` and re-waits PAST its own timeout in an
 * uninterruptible D-state (observed 61s->122s, unkillable) — so the recovery probe MUST NOT block. This issues
 * a tiny int8 matmul (1x512x16, A=B=1 => C==512) with flags PC|NONBLOCK: the ioctl returns immediately (never
 * enters the kernel wait), then we poll the output doorbell HOST-side with a hard 300ms cap. Doorbell lands
 * with the right value => NPU is dispatching again (recovered, 1); host-timeout => still wedged (0 => fault).
 * (A comprehensive fp16/SDP/PPU self-test is fine for HEALTHY validation but is the wrong tool here — it blocks.) */
/* REAP stuck async jobs the CLEAN way (task #47 Bug#2): the vendor driver reaps a stuck/timed-out async job via
 * rknpu_job_timeout_clean, which runs at the TOP of EVERY nonblock submit — if that core's job is past its timeout it
 * soft-resets + schedule_work(cleanup_work) => a CLEAN rknpu_gem_object_put of the job's task_obj. (Contrast close(fd),
 * which frees the task buffer UNDER the still-referencing stuck job => the deferred cleanup_work then double-puts a
 * freed obj => refcount-underflow UAF — the fd-reap wedge.) So: one tiny FP16 nonblock op per core (core_mask=1<<i)
 * triggers timeout_clean for that core, reaping the fp16 colsplit's dropped slice job. FP16 (matches the warm mode) so
 * the dummy itself LANDS (doesn't become a new stuck job); its output is irrelevant — timeout_clean fires before it
 * runs. The dropped job is already past its ~recov_tmo timeout by detect time (800ms poll > 500ms), so it IS reaped. */
void ork_npu_reap_stuck(ork_npu *c, int nc){
    int fd=c->fd, K=512, N=16, CBUF=c->soc->cbuf_elems;  unsigned dom=c->dom_active;
    if(nc<1) nc=1; if(nc>c->soc->cores) nc=c->soc->cores;
    struct buf A=orki_bcreate(fd,(size_t)K*2,0x403,dom), B=orki_bcreate(fd,(size_t)K*N*2,0x403,dom), Cc=orki_bcreate(fd,(size_t)N*2,0x403,dom);
    if(!A.cpu||!B.cpu||!Cc.cpu){ if(A.cpu)orki_bdestroy(fd,&A); if(B.cpu)orki_bdestroy(fd,&B); if(Cc.cpu)orki_bdestroy(fd,&Cc); return; }
    memset(A.cpu,0,(size_t)K*2); memset(B.cpu,0,(size_t)K*N*2);
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t rc[REGCMD_N]; int sched=((K&(K-1))==0 && K>=128 && K<2048);
    orki_synth(rc,1,K,N,(uint32_t)A.dma,(uint32_t)B.dma,(uint32_t)Cc.dma,sched,CBUF); orki_set_f16_out_fp16in(rc,1,N);
    memcpy(c->regcmd.cpu,rc,(size_t)REGCMD_N*4); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
    t[0].enable_mask=0xd; t[0].int_mask=0x300; t[0].int_clear=0x1ffff; t[0].regcfg_amount=108; t[0].regcmd_addr=(uint32_t)c->regcmd.dma;
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    for(int i=0;i<nc;i++){
        struct rknpu_submit s; memset(&s,0,sizeof s);
        s.flags=0x1|0x2u; s.task_number=1; s.task_obj_addr=c->task.obj; s.core_mask=1u<<i; s.fence_fd=-1; s.timeout=300;
        s.subcore_task[0]=s.subcore_task[1]=s.subcore_task[2]=(struct rknpu_subcore_task){0,1};
        orki_rknpu_submit_ioctl(fd,&s,dom);   /* triggers rknpu_job_timeout_clean(core i) -> clean reap of a timed-out stuck job */
        ork_kmsg("reap-stuck: fp16 nonblock dummy core=%d (trigger timeout_clean)", i);
        struct timespec ds={0,3000000}; nanosleep(&ds,NULL);   /* let this core's dummy land + the scheduled cleanup_work run */
    }
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&B); orki_bdestroy(fd,&Cc);
}
/* Self-healing recovery: detect -> DUMP everything -> soft RESET -> DUMMY-op probe. Returns 1 if the dummy op
 * PASSES (NPU recovered — caller keeps going), 0 if it FAILS (NPU still broken — caller should throw a fault
 * and stop, rather than spiral into a hard wedge). This is the recovery contract the caller runs on an anomaly
 * (a round that fails to land / a submit error). */
/* Deliberately force a RELIABLE NPU fault (to exercise dump/recover): a tiny int8 matmul whose WEIGHT address
 * is BOGUS (0x1000 — an unmapped low page) so the NPU DMA-reads from unmapped and faults every time. NONBLOCK +
 * a 1s bounded host poll so the caller never blocks. Returns 1 if the output doorbell landed (no fault), 0 if it
 * did NOT land in the window (the expected fault). May soft-reset or IOMMU-fault the NPU — that's the point. */

/* orkd CLIENT context — the EXPLICIT orkd entry point. Connect (auto-spawn) the orkd daemon and route
 * ork_mm_* through it: the daemon owns the single-stream NPU and serializes every submit, the safe way to
 * share it across concurrent processes. NO local NPU open (the daemon owns it); the ops check c->daemon.
 * Returns NULL if the daemon can't be reached (NO silent fallback to direct — the caller decides). The daemon
 * process itself must not call this (it sets ORKD_IS_DAEMON). Callers pick transport by CHOOSING the entry
 * point: ork_npu_init() = direct (default), ork_npu_init_orkd() = orkd client. */

/* DIRECT (in-process) NPU context — the DEFAULT entry point: opens the DRM card and owns the single-stream
 * NPU directly (do not run concurrent direct-NPU processes; they wedge the IOMMU). For back-compat, the legacy
 * ORK_USE_ORKD=1 env still redirects this to the orkd client (ork_npu_init_orkd) — but new callers should
 * select the transport by calling the desired entry point rather than relying on the env. */
/* ORK_LOAD_PROF: print (and reset) the per-phase import breakdown. Called at teardown. No-op unless set. */
void ork_ssm_prof_dump(void);            /* fwd: ORK_SSM_PROF per-section accounting */
void ork_ssm_helper_stop(ork_npu *c);    /* fwd: stop the little-core marshalling helper */
void ork_npu_xprof_dump(void);

/* ---- zero-copy DMA buffers (NPU-coherent, CPU-mapped). A matmul whose A and/or C live in one of
 * these has the regcmd point at it directly — no host gather/writeout memcpy. ork_mm_run_i8 detects
 * residency automatically (no API change); the caller just allocates A/C here. ---- */
/* Zero-copy IMPORT (no alloc, no copy) — see header. Registered in dma_tab like ork_dma_alloc so
 * ork_mm_run zero-copy detection + dma_find work; freed by ork_dma_import_free (or ork_dma_free). */
/* Import an EXTERNAL dma-buf fd (e.g. received over SCM_RIGHTS from another process) into the NPU's IOMMU
 * domain and register it for zero-copy: the returned CPU pointer maps the shared buffer, and passing a ptr
 * into it as A/C to ork_mm_run* makes the NPU read/write that buffer in place (dma_find resolves the IOVA).
 * Takes ownership of `dmabuf_fd` (closed by ork_dma_free/ork_dma_import_free). NULL on failure. This is the
 * orkd daemon's cross-process zero-copy hook (client shares a buffer; orkd runs against it, no copy). */
/* the registered DMA buffer containing host ptr p, or NULL if p isn't zero-copy-resident */
/* Clean CPU writes -> device for an imported (or ork_dma_alloc) buffer; the bsync the weight fill
 * issues once before the first submit (write-once-read-many weights). size 0 = whole buffer. */
/* Diagnostic only (tools/disk_stream_bench.c): flush `size` bytes of an ork_dma_alloc buffer to the
 * device after a host write (the bsync the streaming fill would issue). Not in the public header. */
/* Diagnostic only (tools/dmabuf_fill_probe.c): allocate a registered DMA buffer with a caller-chosen
 * rknpu mem-create flag set, so the probe can A/B the write-combine (0x401) vs cacheable (0x403) fill
 * bandwidth + NPU-read correctness WITHOUT changing the default ork_dma_alloc behavior. Additive; not
 * in the public header. The buffer is registered in dma_tab so ork_mm_run_i8 zero-copy + dma_find work. */
/* ork_dma_alloc that requests on-chip SRAM residence (fails over to DRAM if the NPU has no SRAM / it is full).
 * For validating the precompiled/doorbell submit against an SRAM-resident output the CPU polls via dc civac. */
/* Diagnostic only: clean-only flush (TO_DEVICE) of a sub-range — push dirty CPU cache lines out to DRAM
 * so the NPU reads correct data, WITHOUT the FROM_DEVICE invalidate. This is the bsync a cacheable
 * weight buffer needs before submit (the "clean cost" the probe measures separately). */
/* policy: cap the cores the auto-tuner may use for a matmul (n<=0 → all soc cores). The library
 * still picks per-matmul ≤ this (small-N matmuls use fewer). ORK_NPU_MC env overrides if set. */
/* Pin the fused chain (run_chain_i8_*) to one NPU core. Default 0 (== prior behavior). Foundation for
 * round-robin: the scheduler places independent prefill chains on different cores, each with its OWN per-core
 * scratch (chain_rc/tk/lrc/lsc[core]) and per-core caches (chain_lut_devloaded[], chain_task_P[]/
 * chain_task_built[]). Cross-core is fixed + validated sequentially (chainrr_probe, all cores coherent) via the
 * subcore_task[] fix + per-core buffers. Kept CLAMPED to 0 for the PUBLIC setter until concurrent dispatch is
 * wired and because a cold target core still needs a prior matmul bring-up; cross-core callers use the
 * scheduler path / ork_npu_set_chain_core_unsafe (which the bring-up-aware caller drives). */
void ork_npu_set_chain_core(ork_npu *c,int core){ if(!c)return;
    if(core!=0) fprintf(stderr,"[ork] ork_npu_set_chain_core(%d) IGNORED: cross-core chain needs per-core buffer routing (round-robin increment 1); clamped to 0\n",core);
    c->chain_core=0; }
/* TEST-ONLY (round-robin increment 1 bring-up experiment): pin the chain core WITHOUT the clamp. UNSAFE unless
 * the caller has already brought core `core` online (e.g. a multi-core matmul that warms cores 0..nc-1); a bare
 * submit to an un-brought-up core wedges the IOMMU. Used by chainrr_probe to validate the per-core cache on
 * each core sequentially. Not for production callers — those go through the clamped setter + the scheduler. */
void ork_npu_set_chain_core_unsafe(ork_npu *c,int core){ if(!c)return;
    c->chain_core = (core>=0 && core<c->soc->cores) ? core : 0; }
/* orkd scheduler priority for this client's subsequent submits (higher = dispatched sooner among queued work).
 * Only meaningful in client mode (c->daemon set); a no-op on a direct-NPU context (nothing to schedule against). */

/* PER-WEIGHT IOMMU DOMAIN PLACEMENT. The rk_iommu 32-bit IOVA cap (~4 GiB) is per iommu_domain_id, so a
 * model larger than 4 GiB stays fully resident (no streaming) by spreading its weights over domains.
 * Set the domain BEFORE packing/loading a weight: every subsequent ork_mm_pack_i8 / ork_mm_load_i8 (and
 * the fp16/int4 variants) places its resident tiles in `domain` and stamps it on the returned ork_w; at
 * run time ork_mm_run* submits that weight's matmuls against the same domain automatically. Activation/
 * output scratch follows the most-recently-set pack domain. domain<0 reverts to the process default
 * (env ORK_IOMMU_DOMAIN, else 0). Domains are created lazily by the kernel on first use. */
void ork_npu_set_pack_domain(ork_npu *c,int domain){ if(!c) return; c->pack_domain = domain<0 ? -1 : domain;
    if(c->daemon) orkd_set_pack_domain(c->daemon, domain<0 ? 0u : (uint32_t)domain); }   /* Path B: route the domain to the daemon so orkd_pack lands the weight there */

/* Allocate an IOMMU domain to pack weights into (isolation + a full ~4 GiB IOVA window each). Path B: request
 * one from orkd's coordinated pool (returns id>0, or <0 if exhausted). Direct: hand out a local id (1,2,…).
 * Make it the pack target with ork_npu_set_pack_domain; return it with ork_npu_domain_free. */
/* Currently ACTIVE iommu domain (the one dom_activate last swapped in), i.e. the domain the NEXT submit
 * would run in if its weight already lives there. Pure getter, no state change. The point: a caller that
 * allocates a TRANSIENT/scratch weight (an attention or GDN bmm's dynamic operand) can place it in the
 * domain that is already active, so running it needs NO dom_activate switch — the switch is what a stuck
 * (unreaped, IRQ-never-fired) job turns into a 60 s "switch iommu domain" stall on the NEXT submit. See
 * ork_dom_flush_if_dirty / dom_dirty. Co-domain scratch sidesteps the whole boundary. */
/* Make `domain` the ACTIVE iommu domain (parks/restores per-domain scratch, establishes it if fresh). A
 * DMA buffer created for a non-0 domain must be allocated while that domain is active, else it maps in the
 * currently-active domain and a submit against `domain` can't see it. Call before ork_dma_alloc-in-domain. */
int  ork_w_domain(const ork_w *w){ return w?w->domain:0; }

/* pack B[K,N] (row-major) into resident NPU tiles. dt: DT_F16 (B fp16, tile [Nt][Kt][16][32],
 * N%16) or DT_I8 (B int8, tile [Nt][Kt][32][32], N%32). K-split (KS) x N-split (NMAX). */
/* Parallel-for over [0,n): split across ALL online cores (dynamic core count from the OS). For the
 * CPU-bound weight tiling during pack — a batch job with no live inference to protect, so it uses every
 * core (big+little), no knob. Spawn-per-region: the tiled memcpy dwarfs the pthread spawn cost. */
/* All online CPUs (big+little) as an affinity mask. The pack pool runs UN-pinned across every
 * core: the calling thread may be pinned to the big cluster (an inference worker often is), but
 * the one-time pack is a CPU-bound batch job with no live inference to protect, so it should use
 * the whole machine. Returns the online-core count, or 0 if it couldn't be determined. */
/* Un-pin the calling thread: allow it to run on ALL online cores. Public so the ggml-ork dequant/
 * quant workers (std::thread, no attr) can un-pin themselves. */

/* PERSISTENT worker pool for ork_parallel_for. Spawn the workers ONCE and reuse them across every
 * call, amortizing the pthread_create/join that dominated fine-grained per-weight tiling — a fresh
 * pool per weight left the cores mostly idle in spawn/join overhead (measured: per-weight CPU tiling
 * capped ~20%). Workers are un-pinned (all cores) and sleep on a condvar between jobs; lazy-init on
 * first use, live for the process. One job at a time (the callers dispatch serially). */

/* Tile a contiguous [nt] range of int8 weight columns into the NPU 32x32 block layout. Shared by the Bb
 * K-slice tiles and the full-K Bf rebuild (same structure; differ only in KT and the k0 offset). Each nt
 * range is disjoint in bb, so this is bit-identical to the serial loop. */

/* Inflate a contiguous [nt] range of int8 weight columns straight into the fp16 [Nt][Kt][16][32] tile
 * layout, scaled per-output-channel: wf16 = (f16)((float)i8 * bscale[n]). Same mapping orki_pack() uses for
 * DT_F16, but the source element is a dequantized int8 code instead of a stored fp16 — so the resulting
 * tile bytes are BIT-IDENTICAL to ork_mm_pack of the row-major dequantized weight. Emulated W8A16. */
ork_w *orki_pack(ork_npu *c,int K,int N,const void *B,int dt){
    int nmod=dt?32:16; if(K%32||N%nmod) return NULL;
    int KS=dt ? orki_int8_ks(c) : c->soc->ks, NMAX=c->soc->nmax, nt_sz=dt?32:16, esz=dt?1:2;
    int Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=dt; w->owns=1; w->domain=ork_dom(c->pack_domain); w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    /* FIX 2 (gated, ORK_CONSOLIDATE_I8): consolidate all int8 Bb tiles into ONE per-weight DMA buffer,
     * tiles being page-aligned base+offset VIEWS — cuts thousands of GEM objects / MEM_CREATE / page-pad
     * to one alloc (matches rkllm's one-buffer-per-domain). owns flips to 0 + own_buf_valid so ork_mm_free
     * reclaims the single buffer. Each tile's regcmd bdma is own_buf.dma+off (validate_regcmd + the run
     * path read it exactly like a per-tile dma — same as the validated grouped-i4 own_buf path). Off by
     * default: this touches the regcmd ADDRESS MATH that wedged the wide-N path before; opt-in to de-risk.
     * Falls back to per-tile owning orki_bcreate (below) on any alloc failure. */
    int consolidate = (dt==DT_I8) && getenv("ORK_CONSOLIDATE_I8");
    if(consolidate){
        size_t wtotal=0;
        for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
          for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;(void)n0;(void)k0;
            wtotal += orki_pgup((size_t)Kp*Nc*esz);}}
        struct buf own=orki_bcreate(c->fd,wtotal,0x403,w->domain);
        if(own.cpu){
            w->own_buf=own; w->own_buf_valid=1; w->owns=0;   /* tiles are views; reclaim own_buf as one */
            size_t off=0;
            for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/nt_sz;
              for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32; size_t ts=orki_pgup((size_t)Kp*Nc*esz);
                struct buf*b=&w->Bb[(size_t)ns*Sk+ks];
                /* size = PAGE-PADDED tile (== per-tile bcreate's b->size) so ork_w_dump byte-matches the
                 * non-consolidated layout and round-trips through ork_mm_load_i8. */
                b->handle=own.handle; b->obj=own.obj; b->dma=own.dma+off; b->cpu=(char*)own.cpu+off; b->size=ts;
                int8_t*bb=b->cpu; const int8_t*Bi=B;
                for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
                    bb[nt*KT*32*32+kt*32*32+nl*32+kk]=Bi[(size_t)(k0+kt*32+kk)*N+(n0+nt*32+nl)];
                off += ts;}}
            orki_bsync_off(c->fd,own.obj,0,wtotal,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
            orki_bsync_off(c->fd,own.obj,0,wtotal,RKNPU_MEM_SYNC_TO_DEVICE);
        } else { consolidate=0; }   /* alloc failed → per-tile fallback below */
    }
    if(!consolidate)
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/nt_sz;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
        /* ORK_WEIGHT_SRAM: request on-chip SRAM for the int8 weight tile (bcreate fails a too-big tile / no-SRAM
         * board over to DRAM). For the CPU/NPU decode PARTITION experiment: NPU reads SRAM ‖ CPU reads DRAM. */
        const uint32_t wflags = getenv("ORK_WEIGHT_SRAM") ? (0x403u|RKNPU_MEM_TRY_ALLOC_SRAM) : 0x403u;
        /* ORK_F16_IMPORT_W (task #47 fd-reap): back the fp16 weight with a dma-heap IMPORT (bimport) instead of a
         * plain bcreate, so its pages survive a DRM-fd close+reopen (ork_ctx_fd_reap re-imports it) — the recovery
         * needs resident weights to persist across the reap. Registered in the import registry for in-place remap. */
        int f16imp = (dt==DT_F16) && getenv("ORK_F16_IMPORT_W");
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks];
        *b = f16imp ? orki_bimport(c->fd,(size_t)Kp*Nc*esz,w->domain) : orki_bcreate(c->fd,(size_t)Kp*Nc*esz,wflags,w->domain);
        if(!b->cpu){
            fprintf(stderr,"[ork] ERROR: %s failed to allocate weight buffer Bb[%zu] in orki_pack (size=%zu)\n",f16imp?"bimport":"bcreate",(size_t)ns*Sk+ks,(size_t)Kp*Nc*esz);
            for(int i=0;i<ns*Sk+ks;i++) orki_bdestroy(c->fd,&w->Bb[i]);
            free(w->Bb); free(w); return NULL;
        }
        if(f16imp) orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);   /* imports: bracket the CPU fill (rknpu MEM_SYNC doesn't cover foreign imports) */
        if(dt==DT_F16){ f16*bb=b->cpu; const f16*Bf=B;
            for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
                bb[nt*KT*16*32+kt*16*32+nl*32+kk]=Bf[(size_t)(k0+kt*32+kk)*N+(n0+nt*16+nl)];
        } else { int8_t*bb=b->cpu; const int8_t*Bi=B;
            struct tile_i8_arg ta={bb,Bi,KT,k0,n0,N}; ork_parallel_for(NN,orki_tile_i8_range,&ta);   // all-core tiling
        }
        if(f16imp) orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);
        orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);
        if(f16imp) orki_imp_reg(b);}}   /* register for fd-reap in-place re-import */
    /* AUTO full-K decode layout (int8, K<=10752): lets the multi-core decode do
     * one full-K submit/core instead of ~K/1024 K-slices. ~2x weight memory — IOVA-FITS GUARD: if
     * any bcreate fails (IOMMU full on a big model), abandon Bf entirely → decode falls back to the
     * K-split path (correct, just slower). No crash, no ceiling guess. */
    if(dt==DT_I8 && K<=10752 && !getenv("ORK_NO_BF") && !(orki_int8_ks(c)<K && getenv("ORK_KTILE"))){ int KTf=K/32; w->Bf=calloc(Sn,sizeof(struct buf)); int ok=1;
        for(int ns=0;ns<Sn && ok;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/nt_sz;
            struct buf*b=&w->Bf[ns]; *b=orki_bcreate(c->fd,(size_t)K*Nc*esz,0x403,w->domain);
            if(!b->cpu){ ok=0; break; }                 /* IOVA full → give up on Bf */
            int8_t*bb=b->cpu; const int8_t*Bi=B;
            struct tile_i8_arg ta={bb,Bi,KTf,0,n0,N}; ork_parallel_for(NN,orki_tile_i8_range,&ta);   // all-core full-K rebuild
            orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}
        if(!ok){ for(int ns=0;ns<Sn;ns++) orki_bdestroy(c->fd,&w->Bf[ns]); free(w->Bf); w->Bf=NULL; } }
    /* SLICE-AND-DICE RESCUE (#33): pre-build doorbell tiles for a REFUSE-PRONE int8 shape so the run
     * path can RUN it instead of returning ORK_RC_WEDGE_PRONE. The raw B is only in scope HERE (the
     * packed w keeps no raw weight), so the tiles MUST be built at pack time. Gated to the one shape
     * class run_multicore actually refuses — wide-N with no single-submit base (Sn>1 && (K>4096 || !Bf),
     * the slice-by-slice path that refuses on Sk>128 / a failed slice) — so well-behaved weights (Bf
     * present, or Sn==1 which r_base/r_wideK cover) build NOTHING and pay no extra IOVA. A NULL build
     * (unaligned to c_base, or IOVA full) just leaves sliced=NULL and the site still refuses (no
     * regression). ORK_NO_SLICE_RESCUE opts out for A/B. */
    if(dt==DT_I8 && !orki_in_slice_pack && !getenv("ORK_NO_SLICE_RESCUE") &&
       ((Sn>1 && (K>4096 || !w->Bf))       /* the shape run_multicore actually refuses */
        || getenv("ORK_SLICE_ALL")))        /* TEST hook: build tiles for ANY int8 shape (small-shape rescue validation) */
        w->sliced = ork_mm_pack_sliced(c, K, N, B, DT_I8);
    /* fp16 gets NO pack-time slice build: fp16's slice-and-dice FIT is the CONTIG column-split (the fp16
     * colsplit path run_multicore already routes all wide fp16 to — wide-K via its Sk-slice loop + lockstep,
     * wide-N via Sn slices), NOT distinct weight tiles. The distinct-tile fp16 twin was the WRONG transform:
     * distinct fp16 weight buffers serialize across cores (measured 6-17x slower than the single-core ref;
     * int8 tiles run concurrently -> parity, fp16 can't). So fp16 stays on colsplit + the single-core-ref
     * backstop; the tiled ork_mm_*_sliced(DT_F16) surface exists only as a documented wrong-fit (see that
     * dispatch). No fp16 w->sliced is ever built here. */
    return w;
}
ork_w *ork_mm_pack   (ork_npu *c,int K,int N,const f16    *B){
    if(c && c->daemon){ uint64_t id=orkd_pack_f16(c->daemon,K,N,B); if(!id) return NULL; ork_w *w=calloc(1,sizeof *w); if(!w) return NULL; w->is_orkd=1; w->orkd_id=id; w->K=K; w->N=N; w->dtype=DT_F16; w->domain=ork_dom(c->pack_domain); return w; }   /* Path B: fp16 pack in the daemon (remember the domain so runs carry it) */
    return orki_pack(c,K,N,B,DT_F16); }

/* ---- Tier 12f: RESIDENT K/V with per-key APPEND (decode attention) -----------------------------------------
 * A decode step appends ONE key to the KV cache, then attends over all keys. Repacking K^T/V from scratch each
 * step is packing-bound (measured ~15x slower than CPU); instead keep the two packed int8 weights RESIDENT and
 * write only the new key's tile bytes each step (+ a per-tile bsync). ork_mm_pack_i8's tile layout is
 *   bb[nt*KT*1024 + kt*1024 + nl*32 + kk]   (nt=n/32, kt=k/32, nl=n%32, kk=k%32; KT = this tile's K/32)
 * K^T weight is [Kp=512, Lmax]: K=Kp fixed so KT=16 and a new key is a new N-COLUMN (n=key) — a clean append
 * into the single N-tile (Lmax<=nmax). V weight is [Lmax, HD]: the key is the K (contraction) index, so it lands
 * in K-tile ks_idx=key/KS at local kt=(key%KS)/32 — multi-tile when Lmax>KS. Both weights are alloc'd zeroed for
 * the full Lmax, so keys beyond the current length contribute 0 (Q·0=0 score, 0 weight) and the caller just runs
 * the matmuls at K=Lmax / N=Lmax with a host softmax over the first `len` keys. Quant scales are the caller's
 * (per-key ks for K via host dequant; a single vs for V). Local NPU only. */
/* Append key `key` (0..Lmax-1): kcol[HD] = this key's K vector (int8), vrow[HD] = its V vector (int8). Writes the
 * tile bytes for the new K^T column and V row, then bsyncs the touched tile(s). Returns 0/ok, <0 bad-arg. */

/* ---- int8 JIT-inflate to fp16 (emulated W8A16 for IOVA headroom) ----
 * A gmax-selected "fp16" layer wants UNQUANTIZED activations (fp16 A, no act-quant error) but does NOT
 * need fp16 WEIGHTS — int8 weights are usually accurate enough, and residing fp16 weights doubles IOVA so
 * only ~5 layers fit a 4GiB domain. Instead: keep the weight host-side as compact int8 + per-channel
 * bscale, and inflate it into ONE REUSED fp16 scratch buffer per matmul. Resident IOVA cost = a single
 * fp16 scratch (reused across all such layers), so the number of fp16-path layers is decoupled from the
 * IOVA cap and gmax becomes a pure coherence<->speed dial. The fp16 MAC then runs int8-precision weights
 * against fp16 activations = emulated W8A16 (RK3588 has no native W8A16 datapath). */

/* Allocate a REUSABLE fp16 scratch weight: fp16 tile layout ([Nt][Kt][16][32], KS=soc->ks) sized for
 * (K,N), buffers init-synced but carrying no data. Fill per-forward with ork_mm_inflate_i8_to_f16 and run
 * via ork_mm_run / ork_mm_run_f16_silu. Reclaim with ork_mm_free like any packed weight. K%32, N%16.
 * Returns NULL on bad dims / alloc failure. */
/* Fill an fp16 scratch (from ork_mm_f16_scratch, same K,N) with wf16[k,n]=(f16)((float)i8[k*N+n]*bscale[n]).
 * i8 is row-major [K,N]; bscale is per-output-channel [N] (NULL => scale 1). Re-tiles in place (no alloc);
 * single TO_DEVICE sync per tile (buffers already inited by ork_mm_f16_scratch, like ork_mm_repack_i8).
 * The tiled bytes are bit-identical to ork_mm_pack of the row-major dequantized weight. 0/ok, <0 on bad args. */

/* PERSIST. Serialize a packed weight's resident tile bytes (Bb only; Bf is a regenerable decode-only
 * optimization) into `out` in tile order — the on-disk form for pre-packed (.orkpack) weights. Each
 * tile is its page-padded buffer size, so it round-trips through ork_mm_load_i8. Pass out=NULL to size. */
size_t ork_w_dump(const ork_w *w, void *out, size_t cap){
    if(!w || !w->Bb) return 0;
    size_t off=0, nb=(size_t)w->Sk*w->Sn;
    for(size_t i=0;i<nb;i++){ const struct buf *b=&w->Bb[i]; if(!b->cpu) continue;
        if(out){ if(off+b->size>cap) return 0; memcpy((char*)out+off,b->cpu,b->size); }
        off+=b->size; }
    return off;
}
/* CPU-ONLY int8 dump: produce the SAME bytes as ork_mm_pack_i8() + ork_w_dump(), but tile straight
 * into a caller DRAM buffer — no NPU. There is NO reason to allocate an IOMMU/IOVA DMA buffer, tile
 * into it, cache-flush it TO the device, and read it back just to write a .orkpack file: that whole
 * bcreate+bsync round-trip is the serial single-stream consumer that bottlenecks conversion. Here the
 * tiling (same orki_tile_i8_range, page-padded per tile, same Sk×Sn order as ork_w_dump) runs pure-CPU and
 * parallel across all cores; the NPU is touched only at LOAD time (ork_mm_load_i8_import). Pass out=NULL
 * to size. K%32, N%32. Byte-identical to the pack+dump path (fresh DMA bufs are zeroed; we zero-pad). */
/* Produce the Bf (full-K re-tiled) blob for weight B[K,N] straight from raw row-major int8, PURE-CPU — the
 * on-disk companion to ork_w_dump_i8_cpu so a .orkpack can carry Bf and the loader skips the runtime rebuild.
 * Layout: Sn page-padded tiles; tile ns = orki_pgup(K*Nc), holding [nt][ktf][32][32] over the FULL K (KTf=K/32).
 * Byte-identical to the load-time Bf rebuild / ork_mm_repack_i8's Bf tiling. Only the Bf run envelope
 * (K%512==0 && K<=4096) has a Bf; returns 0 otherwise. Pass out=NULL to size. */
/* #39 mfold: fold-layout (rkllm M-fold, fold_woff) weight blob for B[K,N] straight from raw row-major int8,
 * PURE-CPU — the on-disk companion (orkpack v5 "Bfold") so the run path skips the per-call fold_woff repack
 * that otherwise kills mfold. Layout: nslice tiles (NS=FOLD_REF_N wide); tile s = orki_pgup(K*sliceW) holding
 * fold_woff(n,k,K) for that slice's columns. Only K==FOLD_REF_K && N<=3*FOLD_REF_N (the baked-ref fold
 * envelope); returns 0 otherwise. Byte-identical to ork_npu_fold_op_i8's per-slice W pack. out=NULL to size. */
/* NPU-availability gate for the hybrid pack scheduler: return 1 if the NPU appears IN USE (any core
 * loaded above a small threshold), 0 if idle. Reads the kernel's rolling per-core load counter. A
 * hybrid conversion routes a weight to the NPU tile/pack path ONLY when this is 0, so a background
 * .orkpack build never steals cycles from live inference on another process — the CPU path (tile from
 * pagecache into DRAM, zero-copy import) handles everything while the NPU serves. Best-effort: on any
 * read failure it returns 0 (assume idle) so the caller keeps the NPU option. Cheap enough to poll per
 * weight. Threshold >5% treats warm-up/idle noise as free but any real submit stream as busy. */
/* Reload pre-tiled int8 weight bytes (from ork_w_dump / a .orkpack) straight into NPU DMA — bcreate +
 * memcpy + bsync, with NO dequant / quant / tiling. The fast path for streaming persisted weights: a
 * re-pack becomes a plain DMA copy. `blob`/`n` must be this exact (K,N) int8 weight's Bb dump, in pack
 * order. Returns NULL on shape/size mismatch. Mirrors orki_pack()'s int8 geometry (KS=1024).
 * Also rebuilds the full-K Bf buffer (K<=10752) — Bf is not dumped (it's a regenerable re-tiling of the
 * SAME bytes), but the decode fast path AND run_chain_i8 (Sk>1 experts) need it, so a loaded weight must
 * carry it to be a first-class drop-in for a packed one. Bf is reconstructed from Bb (un-tile → B[K][N] →
 * re-tile full-K); on IOVA exhaustion it's abandoned (Bf=NULL) → decode/run_i8 K-split still works. */

/* #39 mfold: load the fold-layout weight blob (ork_w_dump_fold_i8_cpu / orkpack v5 "Bfold") into a resident
 * ork_w carrying only w->Bfold (nslice bufs). Run via ork_npu_fold_run_w. Shape/size-checked; NULL on mismatch. */
/* #39 attach a fold-layout weight blob (ork_w_dump_fold_i8_cpu / orkpack v5 "Bfold") to an EXISTING loaded/packed
 * ork_w so ork_mm_run_i8 auto-routes small-M through the fold. Bfold bufs land in w->domain. 0 ok, <0 error.
 * No-op (0) if already attached. Caller stores the fold blob only for the winning shapes (K=3584, wide q/o N). */
/* Zero-copy IMPORT variant of ork_mm_load_i8: each resident tile is a dma-buf the NPU reads in place
 * (PRIME import) instead of a MEM_CREATE-alloc'd buffer the blob is memcpy'd into. The bytes still get
 * written once (into the imported mmap) + synced once; the saving is the kernel page allocation, not
 * the host fill (load is from a disk/RAM blob either way). Same blob format / round-trip as load_i8.
 * Falls through to NULL (caller uses ork_mm_load_i8) if import is unavailable. */

/* ---- ORKD_IMPORT: client-owned resident weight (client allocs the dma-buf, daemon only maps it) ----
 * The client (which has NO NPU fd under orkd) allocates a plain dma-heap buffer, fills it with the PRE-TILED
 * .orkpack bytes, and hands the fd to the daemon; the daemon PRIME-imports it into the client's domain and
 * lays the resident tiles as base+offset VIEWS. This keeps ALL weight residency client-side (client manages
 * its IOVA domains) — the daemon never tiles, never allocs a weight buffer, never owns the bytes. */

/* CLIENT: allocate a dma-heap buffer of `size` bytes, mmap it R/W, and (via DMA_BUF_SYNC_START) ready it for
 * CPU fill. Returns the dma-buf fd (>=0) and sets *ptr to the mapping; -1 on failure. No NPU touched. The
 * caller fills *ptr then calls ork_dmabuf_seal(fd) to flush before passing the fd to the daemon. */
int ork_dmabuf_alloc(size_t size, void **ptr){
    int hf=orki_dmaheap_open(); if(hf<0) return -1;
    size_t sz=orki_pgup(size);
    struct dma_heap_allocation_data a; memset(&a,0,sizeof a); a.len=sz; a.fd_flags=O_RDWR|O_CLOEXEC;
    if(ioctl(hf,DMA_HEAP_IOCTL_ALLOC,&a)){ perror("DMA_HEAP_ALLOC(client)"); return -1; }
    int dbuf=(int)a.fd;
    void*p=mmap(NULL,sz,PROT_READ|PROT_WRITE,MAP_SHARED,dbuf,0);
    if(p==MAP_FAILED){ perror("mmap(client dmabuf)"); close(dbuf); return -1; }
    orki_dmabuf_sync(dbuf,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
    if(ptr) *ptr=p;
    return dbuf;
}
/* CLIENT: flush the CPU-written bytes so the NPU (via the daemon's IOMMU import) sees them. */
void ork_dmabuf_seal(int dbuf){ if(dbuf>=0) orki_dmabuf_sync(dbuf,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE); }

/* DAEMON: adopt client-passed pre-tiled int8 weight dma-buf(s) as a resident weight WITHOUT tiling or
 * allocating weight bytes — PRIME-import into the current pack_domain and lay tiles as base+offset VIEWS.
 * `bb_fd` holds the Bb tiles; `bf_fd` (>=0) holds the full-K Bf region in ITS OWN import so Bf has its own
 * IOMMU obj — matching orki_pack()/load_i8, NOT a view into Bb (a shared-obj Bf wedges the base-matmul doorbell:
 * soft-reset recovery loop). Each import is one own_bufs[] chunk (freed once by ork_mm_free); the tile views
 * carry heap_fd=-1 so the Bf free-loop skips them. Returns the ork_w (owns=0) or NULL. ALWAYS consumes both
 * fds. The client already CPU-flushed the bytes (ork_dmabuf_seal), so no sync here. */

/* CLIENT orchestrator (orkd path): allocate a dma-buf, copy the PRE-TILED blob (Bb tiles, optionally followed
 * by the full-K Bf region at bf_off) into it, seal it, and hand the fd to the daemon (ORKD_IMPORT) which
 * imports it into this client's pack_domain as a resident weight (views into the client's buffer). Returns an
 * is_orkd ork_w handle (orkd_id) or NULL. `n` = blob bytes; bf_off = Bf region offset (0 = no Bf). The daemon
 * dup's the fd via SCM_RIGHTS, so the client's local mapping/fd are released here. */

/* Re-tile int8 B[K,N] into an EXISTING ork_w's resident buffers (same K,N), reusing the DMA
 * allocations — NO bcreate/bdestroy. For pooling reused weights (e.g. MoE experts) so the NPU IOMMU
 * isn't churned/fragmented by per-weight alloc+free. Returns 0 ok, -1 bad arg, -2 shape mismatch. */
/* Re-tile fp16 B[K,N] (row-major) into an EXISTING fp16 ork_w (from ork_mm_f16_scratch/ork_mm_pack, same
 * K,N) — no bcreate/bdestroy. The fp16 twin of ork_mm_repack_i8: lets a caller keep a persistent weight
 * POOL and refresh its data per chunk (kills the per-matmul IOMMU alloc/free churn). fp16 tile layout
 * [Nt][Kt][16][32] (KS=soc->ks). Bb only (the scan is single-slice small-K; no full-K Bf). 0/ok,<0. */
/* ---- Diagnostic only (tools/dmabuf_fill_probe.c): a load_i8 variant whose resident Bb tiles are
 * allocated with a CALLER-CHOSEN rknpu mem flag (0x401 WC vs 0x403 cacheable), so the probe can A/B
 * the weight-fill bandwidth AND the NPU read correctness for each flag. Allocates + leaves the blob
 * copied in once (a valid initial state); the probe then times steady-state re-fills via the accessors
 * below. Additive; not in the public header; does NOT change pack/run or the default load_i8. ---- */
/* Diagnostic accessors over an ork_w's resident Bb tiles (for the fill probe): number of tiles, and
 * the cpu ptr + byte size of tile i. The probe memcpys blob bytes into these to time steady-state fill. */
int    ork_w_ntiles(const ork_w *w){ return (w&&w->Bb)?w->Sk*w->Sn:0; }
void  *ork_w_tile_cpu(const ork_w *w,int i){ return (w&&w->Bb&&i>=0&&i<w->Sk*w->Sn)?w->Bb[i].cpu:NULL; }
size_t ork_w_tile_size(const ork_w *w,int i){ return (w&&w->Bb&&i>=0&&i<w->Sk*w->Sn)?w->Bb[i].size:0; }
/* clean-only flush (TO_DEVICE) of tile i — the bsync a cacheable weight buffer needs before submit. */
void   ork_w_tile_clean(ork_npu *c,const ork_w *w,int i){
    if(!w||!w->Bb||i<0||i>=w->Sk*w->Sn||!w->Bb[i].cpu) return;
    orki_bsync(c->fd,&w->Bb[i],RKNPU_MEM_SYNC_TO_DEVICE);
}
/* the full TO|FROM then TO orki_bsync (the current ork_dma_bsync_to_device pattern), per tile. */
void   ork_w_tile_bsync_full(ork_npu *c,const ork_w *w,int i){
    if(!w||!w->Bb||i<0||i>=w->Sk*w->Sn||!w->Bb[i].cpu) return;
    orki_bsync(c->fd,&w->Bb[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    orki_bsync(c->fd,&w->Bb[i],RKNPU_MEM_SYNC_TO_DEVICE);
}

/* ---- NEON SIMD pack/repack DIRECTLY from f32[N][K] (n-major, as ggml's to_float produces) ----
 * Fuses per-channel f32->int8 quant INTO the tile loop: for a fixed channel n the 32 K-values are
 * contiguous in f32[n][:], so NEON-load 32 f32, mul by the channel inverse scale, round/clamp to
 * [-127,127], narrow to 32 CONTIGUOUS int8 — no transposed scratch (the old transpose store was ~69%
 * of the MoE repack). Computes per-channel bscale[] for the caller. */
void orki_chan_scales_f32(const float *f32, int K, int N, float *inv, float *bscale) {
    for (int n = 0; n < N; n++) {
        const float *fr = f32 + (size_t)n * K; float mx = 1e-9f; int k = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        float32x4_t vmx = vdupq_n_f32(1e-9f);
        for (; k <= K - 4; k += 4) vmx = vmaxq_f32(vmx, vabsq_f32(vld1q_f32(fr + k)));
        float m[4]; vst1q_f32(m, vmx); float a = m[0] > m[1] ? m[0] : m[1], b = m[2] > m[3] ? m[2] : m[3]; mx = a > b ? a : b;
#endif
        for (; k < K; k++) { float v = fabsf(fr[k]); if (v > mx) mx = v; }
        inv[n] = 127.0f / mx; if (bscale) bscale[n] = mx / 127.0f;
    }
}
/* tile f32[N][K] -> int8 NPU layout (Bb K-split + Bf full-K) via precomputed per-channel inv[]. Each
 * buffer gets the full init sync (TO|FROM then TO) that orki_pack() uses — fresh buffers need it (a single TO
 * leaves the device side uninitialized -> the NPU submit wedges/times out). */
/* ---- "effective w4a8": int4-PRECISION weights, int8 compute, int4 STORAGE ----
 * RK3588's NPU MACs are int8-only — there is no native int4->int8 datapath. So we synthesize w4a8:
 * quantize each weight to int4 precision (per-channel scale = max|w|/7, range [-7,7]), keep the
 * compact nibble-packed form on the ork_w (w->Bi4, K*N/2 bytes — the memory win + the on-disk form
 * for .orkpack/streaming), then NEON-expand the nibbles back to int8 [-7,7] and DMA-tile that through
 * the existing int8 path so the result runs via ork_mm_run_i8 unchanged. bscale_out[n] carries the
 * dequant scale (C_real[m][n] = aScale[m]*bscale[n]*Ci[m][n], same convention as pack_i8_f32). */

/* quantize one output channel's K f32 weights -> int4 q in [-7,7], nibble-pack into `nib` (K/2 bytes),
 * and write the dequantized int8 q-value (== the int4 code) as f32 into `qf32[K]` for the int8 tiler.
 * sr!=0: stochastic rounding (q=floor(w/scale + u), u in [0,1) from xorshift) — SR removes the
 * quantization BIAS so dot-product error grows ~sqrt(K) instead of O(K). seed advanced per element. */
/* NEON-expand one channel's nibble-packed int4 codes -> int8 [-7,7] sign-extended floats in qf32[K].
 * (The "NEON where the NPU can't": the hardware has no int4->int8 expansion datapath; we do it in
 * software, then feed the int8 tiler.) Bulk path does 16 codes (8 bytes) per iteration. */
/* ---- NF4 codebook (non-uniform int4): the 16 fixed bitsandbytes NF4 normalized levels, index 0..15 ----
 * Unlike UNIFORM (a symmetric int4 grid), the 4-bit value indexes this per-tensor codebook of levels tuned
 * for ~N(0,1) weights. Per-channel scale = max|w_n| (so bscale[n]=absmax/127 and the int8 LUT = round(level*127)
 * reconstructs level*absmax). Better accuracy than uniform int4 for Gaussian-ish weights. */
const float ORKI_NF4_LEVELS[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f };
/* Quantize one output channel's K f32 weights to NF4: per-element find the nearest NF4 level -> 4-bit index
 * (0..15), nibble-pack into `nib` (K/2 bytes). The NF4 levels are monotonically increasing, so we find the
 * bracketing pair [lo,hi] and pick the nearer. sr!=0: stochastic-round between the two bracketing levels —
 * pick lo with probability proportional to (w_norm - level[lo])/(level[hi]-level[lo]) toward hi. `absmax`
 * is the per-channel max|w| (>0); w_norm=w/absmax in [-1,1]. Indices written to qidx[K] for the int8 inflate. */
/* Inflate one channel's NF4 indices (0..15) -> int8 codes via the fixed 16-entry LUT (round(level*127)),
 * writing f32 for the int8 tiler. NEON path uses vqtbl1q_u8 (16-byte table lookup, 16 idx/iter). The LUT
 * is the SAME for every channel (per-tensor codebook); bscale[n]=absmax/127 carries the per-channel scale. */
/* ---- DIRECT int4 -> int8-tiled inflate (no f32 intermediate, no re-quant) ----
 * The f32 path inflates nibble -> f32 code -> orki_tile_f32_i8, which re-quantizes via
 * lrintf(code*1.0) clamped to [-127,127]. But the codes are ALWAYS exact small ints
 * (UNIFORM in [-7,7]; NF4 LUT = round(level*127) in [-127,127]) so that quant is the
 * identity: the int8 byte placed in the tile equals the int4 code. So we can inflate
 * straight to int8 and rearrange bytes into the tile layout with NO float round-trip.
 * Output is bit-identical to the f32 path (proven by the matmul/memcmp gate). */
/* UNIFORM: expand one channel's nibble-packed int4 codes -> LINEAR int8 [-7,7] in i8[K]. */
/* NF4: inflate one channel's indices (stored in the nibble) -> LINEAR int8 codes via the LUT.
 * The nibble store keeps the 0..15 index (low/high nibble per k); LUT[idx] = round(level*127). */
/* Rearrange LINEAR int8 codes i8[N][K] -> the NPU tiled int8 layout, copying bytes (NO quant, NO float).
 * Byte-for-byte the same destination math as orki_tile_f32_i8 (per (ns,ks) buffer: element of channel
 * n=n0+nt*32+nl at k-pos k0+kt*32+ki lands at nt*KT*32*32 + kt*32*32 + nl*32 + ki) but feeding the int8
 * code directly, since orki_tile_f32_i8 with inv=1 maps code -> clamp(lrintf(code),-127,127) = code (identity).
 * Same per-buffer init bsync sequence as orki_tile_f32_i8 (fresh buffers need TO|FROM then TO). */
/* DIRECT int4 -> int8-tiled fill: inflate w's nibble store straight into its resident DMA tiles, no f32.
 * (kind selects UNIFORM sign-extend vs NF4 LUT.) Uses a per-channel linear-int8 scratch i8scratch[N*K]
 * (1 byte/elem vs the f32 path's 4) reused across channels. Produces bit-identical tiled bytes to the
 * f32 path. Caller provides scratch (size N*K) so the streaming consumer can reuse one allocation. */
/* ---- imatrix (importance-matrix) weighted per-channel scale selection ----
 * The quant scale is per-OUTPUT-channel; the imatrix is per-INPUT-channel (length K, importance[k] =
 * <activation_k^2>). They are orthogonal, so the imatrix can't re-weight a nearest-level pick. Instead we
 * pick the per-channel scale (clip ratio): for a grid of r in (0,1], the candidate scale is r*absmax;
 * quantizing at a smaller scale clips outliers but gives the bulk more resolution. We keep the r whose
 * dequant minimizes Sum_k imatrix[k]*(w[k]-dequant[k])^2. O(grid*K) per channel (pack is one-time). */
/* Quantize one channel at the given per-channel absmax (uniform: scale=absmax/7; NF4: scale=absmax/127)
 * into a reused dq[K] scratch (the dequantized weight, in original f32 units), and return the
 * importance-weighted reconstruction error Sum_k im[k]*(w[k]-dq[k])^2 (im NULL => unit weights).
 * Does NOT touch the nibble store — used to score a clip candidate; the winner is re-committed below. */
/* Search the clip grid for one channel and return the absmax (= r*rawabsmax) that minimizes the
 * imatrix-weighted error. dq is reused scratch[K]. With im==NULL this would just return rawabsmax. */
/* ---- COMPACT int4 PERSIST (.orkpack streaming form) ----
 * Unlike ork_w_dump (which serializes the EXPANDED int8 tile bytes, ~K*N), this dumps the COMPACT int4
 * nibble store (~K*N/2) + per-channel scales — about half the size — and ork_mm_load_i4a8 re-inflates the
 * nibbles -> int8 and re-tiles on load (the tail of the pack path, but from stored nibbles, not f32). The
 * blob is self-contained: the NF4 LUT is NOT stored (it's derived from quant_kind). */

/* Serialize the compact int4 form: header + bscale[N] (f32) + Bi4 (K*N/2 bytes). out=NULL -> required
 * size. Returns 0 if `w` is not an int4-packed weight (no Bi4/bscale) or on cap overflow. */
/* Reload the compact int4 form straight into NPU DMA: parse+validate header, read bscale + Bi4, inflate
 * each channel's nibbles -> int8 (UNIFORM sign-extend / NF4 LUT per quant_kind) and orki_tile_f32_i8 into a
 * fresh DMA buffer — the tail of the pack path, from stored nibbles instead of re-quantized f32. Retains
 * a copy of Bi4 + bscale so the loaded weight can be re-dumped byte-identically. NULL on malformed blob. */
/* Rearrange linear int8 codes i8[N][K] into IMPORTED (dma-buf) tiles, using the dma-buf's OWN cache sync
 * (the rknpu MEM_SYNC does NOT cover foreign imports). Same byte math as orki_tile_i8_to_tiles. */
/* Zero-copy IMPORT variant of ork_mm_load_i4a8: resident tiles are dma-bufs the NPU reads in place (PRIME
 * import), and the int4 nibbles inflate -> int8 directly into them (no f32 round-trip). Bit-identical to
 * ork_mm_load_i4a8 (same blob, same tiled bytes). Falls through to NULL (caller uses ork_mm_load_i4a8) if
 * import is unavailable. Retains Bi4 + bscale so the loaded weight re-dumps byte-identically. */
/* ---- DIAGNOSTIC ONLY (tools/prefetch_headroom.c): isolate the STEADY-STATE per-slice streaming prep.
 * These re-run the TAIL of the int4 pack path (inflate stored nibbles -> int8 codes; tile into the
 * ALREADY-ALLOCATED resident DMA buffers) on an int4-packed weight, with NO bcreate/alloc — exactly the
 * work a streaming double-buffer would do per cycled slice. They do not alter pack/run behavior. */
/* inflate w->Bi4 (all N channels) -> int8 codes as f32 in caller scratch qf32[N*K] (UNIFORM sign-extend
 * / NF4 LUT per quant_kind). Mirrors the inflate loop in pack_i4a8 / load_i4a8. */
/* force the inflate KIND (lets the bench time UNIFORM and NF4 on the same nibble store; the inflate
 * cost is data-independent, so it's a valid per-path microbench either way). */
/* tile inflated codes qf32[N*K] into w's existing resident DMA buffers (inv=1; codes are exact). Reuses
 * the production orki_tile_f32_i8 — same memcpy/quant + orki_bsync(TO_DEVICE) the steady-state stream would issue. */
/* DIRECT path microbench: inflate w's nibbles STRAIGHT to int8-tiled (no f32, no re-quant) into the
 * resident DMA tiles. i8scratch is caller-provided (size N*K); kind forces UNIFORM/NF4. Bit-identical
 * to ork_slice_inflate_i4a8_kind + ork_slice_tile_i8, but in one pass with no float round-trip. */
/* DIRECT inflate ONLY (nibble -> linear int8 i8[N*K]); the rearrange/bsync is the separate tile step.
 * Lets the bench split direct inflate cost from the tile+bsync cost. */

/* ---- DIAGNOSTIC ONLY (tools/stream_prefetch_probe.c): a "staging slot" that splits the int4-streaming
 * swap into its three phases so a probe can time each AND run a real double-buffered loop:
 *   (a) FILL  = int4->int8 inflate + tile into a BARE (mmap'd, NOT yet IOMMU-mapped) dma-buf + dma-buf
 *               cache clean  -> the prefetchable CPU work (ork_stage_fill).
 *   (b) MAP   = PRIME_FD_TO_HANDLE + MEM_CREATE(handle) on each bare dma-buf -> IOVA; build an ork_w view
 *               over them  -> the swap-time zero-copy import (ork_stage_map).
 *   (c) RUN   = ork_mm_run_i8 against the mapped view (ork_stage_run) -> the NPU submit.
 * ork_stage_unmap MEM_DESTROYs the maps (keeps the bare dma-buf+mmap for recycle); ork_stage_free closes.
 * This is exactly the int4 prefetch-inflate staging ring the design proposes, exposed for measurement
 * before promoting it into the library. Not in the public header. */

/* bare DMA-heap dma-buf: alloc + mmap, NO PRIME/MEM_CREATE (no IOVA yet). heap_fd = dma-buf fd. */
/* IOMMU-map an already-allocated bare dma-buf (sets dma/obj/handle). 0 ok / -1 fail. */
/* MEM_DESTROY the map (keep the dma-buf + mmap alive for recycle): clears dma/obj/handle only. */

/* tile shape mirrors ork_mm_load_i4a8: KS=1024 K-split, NMAX N-split; Bf full-K when K%512==0 && K<=4096
 * (same envelope as load_i8_import). Returns NULL if dma-heap absent / alloc fails. */
struct ork_stage *ork_stage_create(ork_npu *c, int K, int N){
    if(K%32 || N%32 || orki_dmaheap_open()<0) return NULL;
    int KS=1024, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    struct ork_stage *s=calloc(1,sizeof *s); if(!s) return NULL;
    s->K=K; s->N=N; s->Sk=Sk; s->Sn=Sn;
    s->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    s->i8scratch=malloc((size_t)N*K);
    if(!s->Bb || !s->i8scratch){ free(s->Bb); free(s->i8scratch); free(s); return NULL; }
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS;(void)n0;
        struct buf*b=&s->Bb[(size_t)ns*Sk+ks]; *b=orki_bstage_alloc((size_t)Kp*Nc);
        if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) orki_bstage_free(&s->Bb[i]); free(s->Bb); free(s->i8scratch); free(s); return NULL; }}}
    if(K%512==0 && K<=4096){ s->Bf=calloc(Sn,sizeof(struct buf)); int ok=1;
        for(int ns=0;ns<Sn && ok;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
            struct buf*b=&s->Bf[ns]; *b=orki_bstage_alloc((size_t)K*Nc); if(!b->cpu) ok=0; }
        if(!ok){ for(int ns=0;ns<Sn;ns++) orki_bstage_free(&s->Bf[ns]); free(s->Bf); s->Bf=NULL; } }
    return s;
}
/* FILL: inflate src's int4 nibble store -> int8, tile into this slot's BARE dma-bufs, clean caches.
 * src must be an int4-packed weight (ork_mm_pack_i4a8) with the same K,N. No IOVA needed (bare bufs).
 * This is the prefetchable CPU work — safe to call on a background thread (touches only this slot). */
void ork_stage_fill(ork_npu *c, struct ork_stage *s, const ork_w *src){
    if(!s || !src || !src->Bi4) return;
    int K=s->K, N=s->N, KS=1024, NMAX=c->soc->nmax, Sk=s->Sk, Sn=s->Sn, kind=src->quant_kind;
    int8_t *i8=s->i8scratch;
    if(kind==ORK_QK_CODEBOOK_NF4){ int8_t lut[16]; for(int i=0;i<16;i++) lut[i]=(int8_t)lrintf(ORKI_NF4_LEVELS[i]*127.0f);
        for(int n=0;n<N;n++) orki_inflate_chan_nf4_i8(src->Bi4+(size_t)n*(K/2),K,lut,i8+(size_t)n*K);
    } else for(int n=0;n<N;n++) orki_expand_chan_i4_i8(src->Bi4+(size_t)n*(K/2),K,i8+(size_t)n*K);
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
        struct buf*b=&s->Bb[(size_t)ns*Sk+ks]; int8_t*bb=b->cpu;
        orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
        for(int nt=0;nt<NN;nt++)for(int nl=0;nl<32;nl++){ int n=n0+nt*32+nl; const int8_t*sp=i8+(size_t)n*K+k0;
            for(int kt=0;kt<KT;kt++) memcpy(bb+((size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32),sp+kt*32,32); }
        orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
    if(s->Bf){ int KTf=K/32;
        for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
            struct buf*b=&s->Bf[ns]; int8_t*bb=b->cpu;
            orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
            for(int nt=0;nt<NN;nt++)for(int nl=0;nl<32;nl++){ int n=n0+nt*32+nl; const int8_t*sp=i8+(size_t)n*K;
                for(int kt=0;kt<KTf;kt++) memcpy(bb+((size_t)nt*KTf*32*32+(size_t)kt*32*32+nl*32),sp+kt*32,32); }
            orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
}
/* FILL from raw untiled int8 B[K][N] (row-major, K-major — as ggml-ork's per-channel quant produces):
 * tile directly into this slot's BARE dma-bufs (RAM-backed, NO IOVA / NO bcreate). The int8 counterpart
 * of ork_stage_fill (which inflates int4 first). Lets a caller add a freshly-quantized weight to the
 * stream pool WITHOUT the transient IOVA pack that would compete with the pool's mapped hot set. Tiling
 * is parallelized across all cores (ork_parallel_for + orki_tile_i8_range) — same layout as orki_pack()/load_i8. */
/* MAP: IOMMU-map every bare dma-buf in the slot and point the slot's ork_w view at them. 0 ok / -1. */
int ork_stage_map(ork_npu *c, struct ork_stage *s){
    if(!s || s->mapped) return s?0:-1;
    int ok=1;
    for(int i=0;i<s->Sk*s->Sn && ok;i++) if(orki_bstage_map(c->fd,&s->Bb[i])) ok=0;
    if(ok && s->Bf) for(int ns=0;ns<s->Sn && ok;ns++) if(orki_bstage_map(c->fd,&s->Bf[ns])) ok=0;
    if(!ok){   /* IOVA full mid-map — roll back the partial maps so a retry (after eviction) starts clean */
        for(int i=0;i<s->Sk*s->Sn;i++) orki_bstage_unmap(c->fd,&s->Bb[i]);
        if(s->Bf) for(int ns=0;ns<s->Sn;ns++) orki_bstage_unmap(c->fd,&s->Bf[ns]);
        return -1;
    }
    memset(&s->view,0,sizeof s->view);
    s->view.K=s->K; s->view.N=s->N; s->view.Sk=s->Sk; s->view.Sn=s->Sn; s->view.dtype=DT_I8; s->view.owns=0;
    s->view.Bb=s->Bb; s->view.Bf=s->Bf;
    s->mapped=1; return 0;
}
/* RUN the slot's matmul (must be mapped). Same as ork_mm_run_i8 on the slot's view. */
int ork_stage_run(ork_npu *c, struct ork_stage *s, int M, const int8_t *A, int32_t *C){
    if(!s || !s->mapped) return -1;
    return ork_mm_run_i8(c, &s->view, M, A, C);
}
/* UNMAP: MEM_DESTROY the maps; keep the bare dma-buf+mmap for the next fill (recycle the slot). */
void ork_stage_unmap(ork_npu *c, struct ork_stage *s){
    if(!s || !s->mapped) return;
    for(int i=0;i<s->Sk*s->Sn;i++) orki_bstage_unmap(c->fd,&s->Bb[i]);
    if(s->Bf) for(int ns=0;ns<s->Sn;ns++) orki_bstage_unmap(c->fd,&s->Bf[ns]);
    s->mapped=0;
}
void ork_stage_free(ork_npu *c, struct ork_stage *s){
    if(!s) return; ork_stage_unmap(c,s);
    for(int i=0;i<s->Sk*s->Sn;i++) orki_bstage_free(&s->Bb[i]);
    if(s->Bf){ for(int ns=0;ns<s->Sn;ns++) orki_bstage_free(&s->Bf[ns]); free(s->Bf); }
    free(s->Bb); free(s->i8scratch); free(s);
}
static size_t stage_bb_bytes(struct ork_stage*s){ size_t t=0; for(int i=0;i<s->Sk*s->Sn;i++) t+=s->Bb[i].size; if(s->Bf) for(int ns=0;ns<s->Sn;ns++) t+=s->Bf[ns].size; return t; }

/* ============================================================================================
 * ork_stream_pool — RAM-resident inflated-int8 weight cache with cheap map/unmap (public API).
 *
 * Phase-1 found: the EXPENSIVE per-swap costs are the int4->int8 inflate (fill) and the MEM_DESTROY
 * (unmap); the MEM_CREATE import (map) is CHEAP (170us@4MB / 654us@16MB). The win is therefore to pay
 * the inflate ONCE per entry (held resident in CPU RAM, budgeted by the caller — RAM >> the 4 GiB IOVA
 * window), and on a cache HIT pay only the cheap map; unmap only on eviction (amortized over the hits).
 *
 * The pool just provides the lifecycle (hold-in-RAM, cheap map/unmap, free); the LRU/eviction POLICY and
 * RAM budget live in the CALLER (e.g. ggml-ork's wcache). Both stores covered: int8 (fill = copy the
 * stored tile bytes) and int4 (fill = inflate the nibbles). A transient prefetch double-buffer is just a
 * small pool the caller fills ahead on a thread — no separate API. Each entry is backed by an ork_stage
 * (bare RAM-resident dma-buf, filled once, persists across unmap; map = bare MEM_CREATE import).
 * ============================================================================================ */



struct ork_stream_pool *ork_stream_pool_create(ork_npu *c){
    if(!c || orki_dmaheap_open()<0) return NULL;          /* import path unavailable -> caller falls back */
    struct ork_stream_pool *p=calloc(1,sizeof *p); if(!p) return NULL;
    p->c=c; p->cap=16; p->e=calloc(p->cap,sizeof*p->e); if(!p->e){ free(p); return NULL; }
    return p;
}
struct ork_stream_entry *orki_pool_new_entry(struct ork_stream_pool*p,int K,int N){
    struct ork_stage *stg=ork_stage_create(p->c,K,N); if(!stg) return NULL;
    struct ork_stream_entry *e=calloc(1,sizeof *e); if(!e){ ork_stage_free(p->c,stg); return NULL; }
    e->stg=stg; e->K=K; e->N=N;
    if(p->n>=p->cap){ int nc=p->cap*2; void*r=realloc(p->e,nc*sizeof*p->e); if(!r){ ork_stage_free(p->c,stg); free(e); return NULL; } p->e=r; p->cap=nc; }
    p->e[p->n++]=e; return e;
}
/* int4-stored: fill = inflate nibbles -> int8 + tile (the .orkpack i4a8 blob, ork_w_dump_i4a8). The fill
 * happens ONCE here (the expensive op, cached in RAM). NULL on import-unavailable / malformed blob. */
/* int8-stored: fill = copy the stored tile bytes (ork_w_dump blob) into the staging dma-bufs. Same blob
 * layout/validation as ork_mm_load_i8. NULL on import-unavailable / size mismatch. */
/* int8-RAW: tile a freshly-quantized UNTILED int8 B[K][N] straight into RAM staging (NO IOVA / NO
 * bcreate) — the zero-transient-pack add. The caller already quantized the weight; we tile it directly
 * into the pool instead of packing to IOVA first (which would compete with the pool's mapped hot set).
 * Map it later with ork_stream_pool_map. NULL on bad dims / import-unavailable. */
/* CHEAP map: bare MEM_CREATE import of the already-filled RAM buffer -> IOVA. The per-submit op on a hit.
 * 0 ok / -1. Idempotent (already-mapped = 0). */
int  ork_stream_pool_map  (struct ork_stream_pool *p, struct ork_stream_entry *e){
    if(!p||!e) return -1; if(e->mapped) return 0;
    if(ork_stage_map(p->c,e->stg)) return -1; e->mapped=1; return 0;
}
/* Unmap the least-recently-used MAPPED entry other than `keep` (frees IOVA, keeps it RAM-resident so the
 * next touch is a cheap remap). Returns 1 if one was unmapped, 0 if none left to evict. */
static int pool_unmap_lru(struct ork_stream_pool *p, struct ork_stream_entry *keep){
    struct ork_stream_entry *lru=NULL;
    for(int i=0;i<p->n;i++){ struct ork_stream_entry *e=p->e[i];
        if(e->mapped && e!=keep && (!lru || e->last_use<lru->last_use)) lru=e; }
    if(!lru) return 0;
    ork_stage_unmap(p->c,lru->stg); lru->mapped=0; return 1;
}
/* RUN a pooled weight, SELF-MANAGING the IOVA sliding window: if the entry isn't currently mapped, map
 * it — and if that fails because the 4 GiB IOVA window is full, evict (unmap) the LRU mapped entry and
 * retry until it fits (self-calibrating; no budget guess). The LRU MECHANISM lives here in ork-driver;
 * the caller only sets the RAM orki_budget (how many entries are held resident). 0 ok / -1 (OOM / bad args). */
int  ork_stream_pool_run  (struct ork_stream_pool *p, struct ork_stream_entry *e, int M, const int8_t *A, int32_t *C){
    if(!p||!e) return -1;
    if(!e->mapped){
        while(ork_stage_map(p->c,e->stg)!=0){          /* IOVA full → evict LRU mapped entry, retry */
            if(!pool_unmap_lru(p,e)) return -1;         /* nothing left to evict → genuine OOM */
        }
        e->mapped=1;
    }
    e->last_use=++p->clock;
    return ork_stage_run(p->c,e->stg,M,A,C);
}
/* UNMAP: MEM_DESTROY the IOVA mapping; entry STAYS filled in RAM (next map is cheap, no re-inflate). */
void ork_stream_pool_unmap(struct ork_stream_pool *p, struct ork_stream_entry *e){
    if(!p||!e||!e->mapped) return; ork_stage_unmap(p->c,e->stg); e->mapped=0;
}
size_t ork_stream_entry_bytes(const struct ork_stream_entry *e){ return e?stage_bb_bytes(e->stg):0; }  /* RAM held (for the caller's budget) */
int    ork_stream_entry_mapped(const struct ork_stream_entry *e){ return e?e->mapped:0; }
/* REMOVE: free the entry's RAM dma-buf (the caller's evict). Unmaps first if mapped. */
void ork_stream_pool_remove(struct ork_stream_pool *p, struct ork_stream_entry *e){
    if(!p||!e) return;
    for(int i=0;i<p->n;i++) if(p->e[i]==e){ ork_stage_free(p->c,e->stg); free(e); p->e[i]=p->e[--p->n]; return; }
}
void ork_stream_pool_free(struct ork_stream_pool *p){
    if(!p) return;
    for(int i=0;i<p->n;i++){ ork_stage_free(p->c,p->e[i]->stg); free(p->e[i]); }
    free(p->e); free(p);
}

/* ---- FUSED dequant->int8 pack/repack (callback per channel; NO full f32[N][K] buffer) ----
 * Materialize one channel at a time into a reused K-float scratch (stays in cache), then NEON quant+tile
 * it — avoids the DRAM round-trip of writing then re-reading a full f32[N][K], which dominates a Q4_K MoE
 * repack. Same int8/bscale result as feeding the equivalent f32 to orki_tile_f32_i8. */
void ork_w_free(ork_w *w){ if(!w)return; free(w->Bb); free(w->Bf); free(w->Bi4); free(w->bscale); free(w->pcrc); free(w->pcrc_meta); free(w->Bbc_ns); free(w); }   /* device buffers freed at ctx teardown */
/* Free a packed weight AND reclaim its NPU DMA/IOVA. Required for layer-streaming: evicted weights must
 * return their IOVA to the 4 GiB window (rk_iommu is 32-bit — see the wiki / npu-iova cap). Only weights
 * that OWN their buffers (per-tile bcreate: pack / pack_i4 / pack_i8) are reclaimed; weights whose tiles
 * are VIEWS into a single dedicated buffer (grouped-i4, own_buf_valid=1) reclaim that one buffer. */
void ork_mm_free(ork_npu *c, ork_w *w){
    if(!w) return;
    if(w->is_orkd){ if(c && c->daemon) orkd_free_weight(c->daemon, w->orkd_id); free(w->fa_lut); free(w); return; }   /* Path B: free the daemon-resident weight */
    if(c) ork_dom_flush_if_dirty(c);   /* #54: clear any stuck job before a per-domain bdestroy switches domains ("failed to destroy memory" + switch-timeout cascade) */
    if(c && w->owns){
        size_t nb=(size_t)w->Sk*w->Sn;
        if(w->Bb) for(size_t i=0;i<nb;i++) if(w->Bb[i].cpu) orki_bdestroy(c->fd,&w->Bb[i]);
    }
    /* Bf is normally its own per-N-slice bcreate/orki_bimport (never a view), even when Bb is consolidated into
     * own_buf — so reclaim it whenever present, independent of owns. EXCEPTION: ork_mm_adopt_imported_i8 lays
     * Bf as base+offset VIEWS into a dedicated Bf import (own_bufs[1]); those carry heap_fd=-1 and must NOT be
     * individually destroyed (the backing import is reclaimed once via own_bufs below). Native Bf has heap_fd>=0. */
    if(c && w->Bf) for(int i=0;i<w->Sn;i++)
        if(w->Bf[i].cpu && w->Bf[i].heap_fd>=0) orki_bdestroy(c->fd,&w->Bf[i]);
    if(c && w->Bfold) for(int i=0;i<w->fold_ns;i++) if(w->Bfold[i].cpu) orki_bdestroy(c->fd,&w->Bfold[i]);   /* #39 mfold resident weight */
    /* dedicated single-buffer weights (grouped-i4, or consolidated int8): Bb[] entries are VIEWS (share
     * own_buf's handle/obj) — destroy the one backing buffer ONLY, never the views (double-free / munmap
     * of a sub-pointer). Reclaims IOVA. */
    if(c && w->own_buf_valid) orki_bdestroy(c->fd,&w->own_buf);
    /* size-bounded consolidated import: Bb[] entries are views into own_bufs[] chunks — destroy each chunk. */
    if(c && w->own_bufs) for(int i=0;i<w->n_own_bufs;i++) if(w->own_bufs[i].cpu) orki_bdestroy(c->fd,&w->own_bufs[i]);
    free(w->own_bufs);
    if(c && w->sliced) ork_mm_free_sliced(c, w->sliced);   /* #33 reclaim the rescue tiles' sub-weights + IOVA */
    free(w->Bb); free(w->Bf); free(w->Bfold); free(w->Bi4); free(w->bscale); free(w->fa_lut); free(w);
}
/* Resident NPU bytes a packed weight occupies (Bb tiles + optional full-K Bf) — for a streaming cache
 * to budget the 4 GiB IOVA window and decide when to evict. */
size_t ork_w_bytes(const ork_w *w){
    if(!w) return 0; size_t t=0;
    if(w->own_bufs) for(int i=0;i<w->n_own_bufs;i++) t+=w->own_bufs[i].size;   /* chunked import: real chunk allocs */
    else if(w->Bb) for(size_t i=0;i<(size_t)w->Sk*w->Sn;i++) t+=w->Bb[i].size;
    if(w->Bf) for(int i=0;i<w->Sn;i++) t+=w->Bf[i].size;
    return t;
}
int ork_w_quant_kind(const ork_w *w){ return w ? (int)w->quant_kind : -1; }   /* ORK_QK_* (int4-store codebook) */
const float *ork_w_bscale(const ork_w *w){ return w ? w->bscale : NULL; }     /* per-channel dequant scale (int4 store), NULL if none */

/* ---- W4A4 public API (int4 A x int4 B -> int32 C), built on the validated synth_i4/regcmd_i4. ----
 * Tiling: N split into 64-wide tiles (the captured regcmd's N width), K split at the 10752 single-
 * submit ceiling (same as int8) with host-side int32 accumulate, M done one row per submit (the
 * captured program's M-tiling). C is int32 (holds the K-accumulated int sum; caller applies scales:
 * C_real[m][n] = aScale[m]*bScale[n]*C[m][n]). DOCUMENTED native layouts (RK3588/3576). */
/* an Nc-wide x Kp-row slice of B[K][N] at (k0,n0) -> native (Nc/64,Kp/32,64,32) int4 (2/byte).
 * Nc%64; validated single-submit up to N=8192 (SoC nmax). */
/* a Kp-slice of one A row -> native (Kp/32,1,32) int4 */
/* a Kp-slice of M activation rows -> native (Kp/32,M,32) interleaved int4 */
/* Reload pre-tiled NATIVE-W4A4 weight bytes (from ork_w_dump of a DT_I4 weight / a .orkpack native-W4A4
 * tier) straight into NPU DMA — NO dequant / FWHT-rotate / int4-quant / tile. The cold-pack fix for the
 * native W4A4 path (mul_mat_i4 / _hadamard / group_i4): the rotated+int4-tiled bytes are persisted once at
 * convert and reloaded as a plain DMA copy. `blob`/`n` = this exact (K,N) DT_I4 weight's Bb dump, pack
 * order (Kp*Nc/2 int4 bytes/tile, pgup'd). The per-channel bscale is persisted SEPARATELY by the caller and
 * re-attached (ork_w_bscale). Returns NULL on shape/size mismatch (caller falls back to packing). K%32, N%64. */
/* Native-int4 IMPORT twin of ork_mm_load_i4: identical DT_I4 tile layout (Kp*Nc/2 nibble bytes, KS=ORK_I4_KS,
 * no Bf) but allocated via orki_bimport (dma-heap + PRIME_FD into the IOMMU) instead of orki_bcreate (MEM_CREATE). MEM_CREATE
 * faults/EINVALs across non-0 domains AND at scale (a >4GiB resident int4 set — e.g. a resident MoE — hits the
 * per-domain window edge, the in-kernel rknpu_gem_object_create fault), so ork_mm_load_i4 cannot bring a big int4
 * weight set resident. This mirrors the PROVEN multi-domain-safe consolidated-chunk import from ork_mm_load_i8_import:
 * a handful of moderate (~ORK_IMPORT_CHUNK_MB) dma-buf chunks, tiles are page-aligned base+offset VIEWS; ork_mm_free
 * bdestroys the chunks (own_bufs). Falls back to per-tile bimport on chunk-alloc failure. */
/* #54 CONSOLIDATED int4 expert load — MIRRORS ork_mm_load_i8_import EXACTLY, extended to share chunks ACROSS
 * experts. Same proven mechanism: bimport into ~ORK_IMPORT_CHUNK_MB (16MB) dma-buf chunks, tiles are page-aligned
 * base+offset VIEWS, fds sealed once a chunk is full (GEM handle keeps it alive for NPU reads). The ONLY change
 * vs the per-weight ork_mm_load_i4_import is that the chunk pool is PERSISTENT per-domain, so MANY experts share
 * a chunk instead of one dma-buf per expert (~9k imports -> ~2340 mappings/domain -> wedge; 16MB chunks pack
 * ~32 experts each -> a few hundred total, ~tens/domain — the count the int8 1.7B proves safe). Critically, like
 * int8 it does NOT set scratch_import: run scratch stays bcreate and coexists with the 16MB import chunks (the
 * PROVEN int8 model — bimport scratch of ANY kind wedges). Weight owns nothing (owns=0, own_bufs=NULL ->
 * ork_mm_free skips it); the shared chunks persist for the ctx, freed once in ork_npu_free. int4-only; resident
 * (no per-expert eviction — the whole MoE design goal). */
/* grouped pack: K split into groups of G (each its own resident slice) for per-group scales. G%32,
 * K%G, G<=10752. Sk = K/G groups; run_i4_grouped scales each group's partial before accumulating. */

int orki_run_i4_bchain_db(ork_npu *c,ork_w *w,int M,const int8_t *A,int32_t *C,int nc);  /* #52: BCHAIN batch on the nonblock doorbell */
ork_dyn_chain *ork_dyn_begin_mc_i4(ork_npu *c, int S, const ork_mm_task_i8 *tasks, int nc);  /* int4 M=1 doorbell (defined below) */
int orki_i4_submit_tmo_ms(void);   /* #54 bounded int4 doorbell submit timeout (TCLEAN reap precondition); defined near the int4 workers */
ork_dyn_chain *ork_dyn_begin_mc_i4_grouped(ork_npu *c, int M, ork_w *w, const int8_t *A, const float *aScale, const float *bScale, float *Cf, int nc);  /* B: grouped-int4 doorbell */
/* #54 COALESCE: run MANY int4 experts (each M>=1 rows) through ONE nonblock doorbell. Decompose every expert's
 * M rows into M=1 tasks and hand the WHOLE set to ork_dyn_begin_mc_i4 — the doorbell distributes+chains them
 * across the cores in one submit-set per core (the HW chaining is the doorbell's job; we don't hand-wire it).
 * Collapses the per-expert submit storm (2059 matmuls x 3 cores) to ~nc submits per _exps tensor. All experts
 * MUST share one iommu domain (the doorbell = one submit = one domain); the caller streams a layer's experts
 * into a single domain. Returns 0 ok, -4 refuse (chain/buffer too big -> caller falls back), -1 error. */
int orki_run_i4_experts_bchain_db(ork_npu *c, const ork_mm_task_i4 *ex, int ntask, int nc);   /* multi-expert BCHAIN (defined below) */
/* Async pipelined submit (precision-agnostic) — orkd+ring mode only. Enqueue one matmul for w WITHOUT blocking
 * and get a ticket; ork_mm_collect(ticket) reads C later. Returns <0 if unavailable (no ring, or the op is too
 * big for a ring slot — use the synchronous ork_mm_run* instead). Keeping several ops in flight lets each op's
 * transport (memcpy + handshake) overlap the NPU compute of the ones ahead of it — the decode pipeline. */
int ork_mm_submit(ork_npu *c, ork_w *w, int M, const void *A){
    if(!c || !c->daemon || !w || !w->is_orkd || !orkd_has_ring(c->daemon)) return -1;
    uint32_t dt = w->dtype==DT_F16 ? ORKD_DT_F16 : w->dtype==DT_I4 ? ORKD_DT_I4 : ORKD_DT_I8;
    return orkd_ring_submit(c->daemon, w->orkd_id, M, w->K, w->N, dt, A);
}
int ork_mm_collect(ork_npu *c, int ticket, void *C){
    if(!c || !c->daemon) return -1;
    return orkd_ring_collect(c->daemon, ticket, C);
}

/* C[M,N] = A[M,K] x packed weights. dt-keyed: fp16 A -> fp32 C, or int8 A -> int32 C.
 * int8 uses 2x the rows budget, K-slice 1024, and effective-K/2 schedule (see synth_i8). */
/* one matmul submit with cold-start warmup; regcmd must already be staged in c->regcmd.
 * core_mask=1<<core selects a single NPU core (0x1/0x2/0x4 = core 0/1/2 — exactly what librkllmrt
 * round-robins). ALL THREE subcore_task[] must be populated even for a single core: leaving the
 * non-target entries zero NULL-derefs rknpu_job_subcore_commit (the earlier kernel Oops). */
/* Submit flags: 0x5 = RKNPU_JOB_PC | RKNPU_JOB_PINGPONG (default). ORK_NO_PINGPONG=1 -> 0x1 (PC only, ping-pong
 * OFF) — ping-pong swaps register banks the instant a task's register config is done, which can race a task's
 * SRAM/side-effect commit and STALL (errno 110), esp. in mixed/chained programs. RE knob to test whether a
 * chain stall is the ping-pong race (vs IOVA). Default unchanged (0x5). */
uint32_t ork_ppflags(void){ static int v=-1; if(v<0){const char*e=getenv("ORK_NO_PINGPONG"); v=(e&&atoi(e))?1:5;} return (uint32_t)v; }
/* P3 #7: doorbell variant of submit1 for the single-core matmul path (orki_run()'s run_fullk_dec / run_loop) —
 * NONBLOCK submit + host-bounded poll on c->Cc instead of the kernel-blocking submit. The plain matmul output
 * is int32 (int8 A·B) or fp32 (fp16), so 0x7fffffff is a safe sentinel (a real int32 accumulator, or a finite
 * fp32 result, won't equal that exact NaN bit pattern). nout = mc*Nc elements written this tile. Poll = last-
 * word gate then a full-surface verify (matmul writeback is last-col-last, so the gate is sound; the verify
 * covers any residual lag). The int8-OUTPUT submit1 callers (fused SiLU/out8) have no safe sentinel and keep
 * the blocking submit1; the ZC-OUT (cbuf) case writes the user buffer, not c->Cc, and also keeps submit1. */
/* ---- multi-core (ORK_NPU_MC=<n>): use n cores (capped at soc->cores). Split each N-slice's
 * output tiles across the cores, run concurrently on per-core buffers, accumulate into disjoint
 * columns of cres (no lock). n is a *request* — the engine can pass any count up to soc->cores,
 * so this is dynamic, not hardwired to a chip's core total. ---- */
int orki_mc_ensure(ork_npu *c,int nc){
    int fd=c->fd;
    if(!c->mtk_all.cpu) {
        c->mtk_all=orki_bscratch(c, sizeof(struct rknpu_task) * ORK_MAXCORE, 0x40b, c->dom_active);
        if(!c->mtk_all.cpu) {
            fprintf(stderr, "[ork] ERROR: mc_ensure failed to allocate mtk_all task buffer (IOMMU full?)\n");
            return -1;
        }
    }
    for(int i=0;i<nc;i++){
        if(c->mrc[i].cpu) continue;        /* alloc once, per core, up to the max ever requested */
        c->mrc[i]=orki_bscratch(c,65536,0x403,c->dom_active); c->mtk[i]=orki_bscratch(c,65536,0x40b,c->dom_active); c->maf[i]=orki_bscratch(c,(size_t)4*32768*2,0x403,c->dom_active);
        if(!c->mrc[i].cpu||!c->mtk[i].cpu||!c->maf[i].cpu) {
            fprintf(stderr, "[ork] ERROR: mc_ensure failed to allocate multi-core buffers for core %d (IOMMU full?)\n", i);
            return -1;
        }
        struct rknpu_task t;memset(&t,0,sizeof t);t.enable_mask=0xd;t.int_mask=0x300;t.int_clear=0x1ffff;t.regcfg_amount=108;t.regcmd_addr=c->mrc[i].dma;
        memcpy(c->mtk[i].cpu,&t,sizeof t); orki_bsync(fd,&c->mtk[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        struct rknpu_task *tall = (struct rknpu_task*)c->mtk_all.cpu;
        tall[i] = t;
    }
    int reg_amt = (c->last_dt == DT_I4) ? 116 : 108;
    struct rknpu_task *tall = (struct rknpu_task*)c->mtk_all.cpu;
    for(int i=0;i<nc;i++){
        struct rknpu_task *t = (struct rknpu_task*)c->mtk[i].cpu;
        if (t->regcfg_amount != reg_amt) {
            t->regcfg_amount = reg_amt;
            orki_bsync(fd,&c->mtk[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        }
        if (tall[i].regcfg_amount != reg_amt) {
            tall[i].regcfg_amount = reg_amt;
        }
    }
    orki_bsync(fd,&c->mtk_all,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    return 0;
}
/* ORK_MCPROF diagnostic: per-core phase timing (copy / submit / acc). Populated by the single-core
 * orki_run() path (the multi-core matmul now runs on the doorbell colsplit, which reports via its own
 * poll/backoff timers, not g_mc_*). Read via ork_npu_mc_timing. */
double orki_mc_copy[MCPROF_MAX], orki_mc_sub[MCPROF_MAX], orki_mc_acc[MCPROF_MAX]; long orki_mc_n[MCPROF_MAX];
double orki_mc_synth[MCPROF_MAX];   /* host regcmd-synth+bsync portion of orki_mc_sub (the OVERLAPPABLE part; ioctl/NPU = sub-synth) */

/* Pin the calling thread to a LITTLE core (low-numbered: A55 0-3 on RK3588). For off-critical-path /
 * IO-bound / memory-bound work (e.g. the SSM double-buffer marshalling helper) that should run on the
 * idle little cluster WHILE the big cores are saturated by ggml's threadpool + the NPU pool. The A55 is
 * ~2x slower but it's free time overlapped with the NPU submit. Honors ORK_NO_AFFINITY. */
/* run_multicore phase timing (ORK_RT): setup (checks+mc_ensure+cres memset), submit (pool dispatch
 * + workers + NPU), copy (cres->C). Pin where the integration's per-matmul time goes vs the kernel. */
double orki_rt_setup=0, orki_rt_submit=0, orki_rt_copy=0; long orki_rt_n=0;
/* FLOOR-DECOMP accessors: ioctl_us = total wall inside SUBMIT ioctls; hw_us = total sub->hw_elapse_time as
 * reported by the kernel (raw units — see hw_raw_last); n = count. See globals near orki_fd_ioctl_us. */
/* MODE-TRANSITION RE hooks (mode_probe.c). The standalone SDP ops (ork_npu_ewmul_*, orki_act_lut_i16 →
 * exp/silu/…) reprogram the pipeline (their own ACT_RESET + SDP regcmd) but leave c->last_dt / c->warmed
 * untouched, so a following SAME-dtype matmul sees dt==last_dt and SKIPS its reset/re-warm — running a
 * matmul regcmd on an SDP-configured pipeline. These expose the two candidate fixes so the probe can
 * measure which is sufficient:
 *   _invalidate: clear the cached mode state ONLY (last_dt=-1, warmed=0, per-core mwarm=0) — the next
 *                matmul then takes its own reset/re-warm path (fp16 entry = warmed=0 re-warm, int8 entry =
 *                ACT_RESET). No explicit HW reset here. Tests whether re-warm alone clears the wedge.
 *   _reset:      an explicit HW ACT_RESET AND invalidate — the heavyweight, always-safe reinit. */

/* ============================ MODE-TRANSITION LAYER (ork_npu_enter) ============================
 * SINGLE owner of "what does moving the NPU's stateful regcmd datapath from mode X to mode Y
 * require" — the ACT_RESET / re-warm (warmed, mwarm[]) / buffer-realloc (ccsz, mccsz[]) policy that
 * was previously copy-pasted (and quietly drifted) inline into every run/stream/chain/int4 entry.
 *
 * Each run path calls ork_npu_enter(c, target_marker, profile, chain) FIRST; the per-profile row of XSPEC
 * below IS the policy for that path. A profile is a faithful, byte-for-byte transcription of the site
 * it replaced (verified `make test` byte-identical across all dtypes and both keep-warm knobs), so
 * Phase-1 behavior was UNCHANGED — the consolidation was behavior-preserving. The drift is visible AS
 * DATA, and a policy change is a one-row edit — e.g. PHASE 2 (2026-07-14) converged the →I8_CHAIN
 * profiles: XP_CHAIN_NT used to ignore ORK_SSM_KEEPWARM (KWP_NTL + RC_NOTLIVE), so a chain entered
 * from an fp16 op ate a full ~105ms ACT_RESET where the stream profiles kept warm; switching it to
 * KWP_MC + RC_NOTLIVE_NOTKW eliminated that (chain_xition_probe: reset-cost 53538us→~0, coherent), and
 * the two stream-int8 profiles collapsed into one (XP_STREAM_I8). See the wiki "Exp-2026-07-14 Mode-
 * Transition Layer" for the full Phase-2 record and AGENTS.md §"Mode-transition layer" for how to add/change.
 *
 * EXHAUSTIVE (from -> to) permutation space — modes = { COLD(-1), F16(0), I8(1), I4(2), I8_CHAIN(3),
 * I4_CHAIN(4), I4_STREAM(5) }, plus SDP = a TRANSIENT activation/ewmul reset with NO stored marker.
 * `from` (= c->last_dt) enters ONLY through the
 * ORK_I8_LIVE / ORK_INT_DT / ORK_KW_DT predicates, so a row is keyed by (target, caller-scope), not by
 * an enumerated `from` — that collapses the NxN matrix to one row per historical site:
 *   ->F16/I8 matmul : reset only ENTERING int8 from a non-int8-live mode (first-int8-submit wedge);
 *                     fp16 never resets. Keep-warm across int8<->fp16 (ORK_SSM_KEEPWARM, default on).
 *   ->I4           : reset entering int4 from a non-int mode; keep-warm int<->int (ORK_MIXED_NOTHRASH).
 *   ->I8_CHAIN(3)  : DT_I8<->DT_I8_CHAIN is NOT a hw mode change (ORK_I8_LIVE) -> no reset.
 *   ->I4_CHAIN(4)  : unconditional reset on entry (single-core int4 M=1 chain).
 *   ->I4_STREAM(5) : unconditional reset on entry.
 *   ->SDP          : activation/ewmul reprogram the pipeline but correctly LEAVE last_dt untouched
 *                    (setdt=0), so the NEXT matmul keeps warm (no ~105us re-warm). The historical
 *                    "SDP->matmul wedge" was NOT a last_dt issue — it was the c->task LUT-descriptor
 *                    poisoning (nuance #1), fixed independently in 98c00b1 (Exp-2026-07-12). Board
 *                    mode_probe (2026-07-14) confirms EVERY SDP->matmul is SAFE with NO reset, and
 *                    that FORCING one costs ~105us/transition for zero correctness gain. XP_SDP is
 *                    therefore KEEP-WARM-AWARE: rst=RC_SDPKW (reset iff !ork_sdp_noreset(), i.e. only
 *                    when the ORK_SDP_NORESET skip is OFF), setdt=0 (no marker, leaves last_dt). This is
 *                    the op-local SDP reset expressed AS DATA — byte-identical to the historical inline
 *                    `if(!ork_sdp_noreset()) orki_act(RESET)`, default-SKIP so it does NOT re-introduce the
 *                    churn ORK_SSM_KEEPWARM removes. NEVER set XP_SDP to RC_ALWAYS (that forces the reset).
 *                    Wired via ork_npu_enter(c, c->last_dt, XP_SDP, OCK_NONE); SDP ops still not yet
 *                    converted keep the inline form (identical behavior) pending a Phase-2 sweep.
 * NUANCE #1 (kept SEPARATE, per Exp-2026-07-12): the c->task LUT-descriptor poisoning is a DISTINCT
 * axis from precision-mode and ACT_RESET does NOT fix it — the layer owns only the precision reset;
 * the c->task save/restore stays an op responsibility (no clr_task cell is wired in Phase 1). */

/* CHAINING MECHANISM in effect for a transition — passed as explicit state to ork_npu_enter so the
 * policy can branch on it for the few handoffs where the mechanism genuinely matters, and ignore it
 * (the common case) otherwise. OCK_NONE = plain per-matmul run / run_multicore / int4 batch;
 * OCK_SW = run_stream_* round-robin (multi-submit, per-core); OCK_HW = run_chain_i8 / chain_progs
 * PC-chain (one submit, task_number>1); OCK_FUSED = run_chain_i8_ffn static regcmd graph (carries
 * in-chain SDP/LUT ops — ping-pong/LUT-commit rules differ; that specialness lives in the chain body). */
/* keep-warm predicate selector (from = c->last_dt, to = target marker) */
/* reset condition selector */
/* warm/size-clear condition selector (shared enum for both the warm and the size clear) */
/* clear target: bit0 = scalar (warmed / ccsz), bit1 = per-core (mwarm[] / mccsz[]) */
/* transition profiles — one row per distinct historical site (line refs = pre-refactor src/npu.c) */
/* Enter mode `to` via `profile`. Returns 1 if a real transition fired (last_dt changed, or an SDP
 * entry reset was issued), 0 if this was a no-op (from==to). The return lets the two sites that gate a
 * caller-LOCAL warmup flag off the transition (XP_I4_INCR's `warm`, XP_I4_STREAM's `cold`) stay
 * byte-identical. See the profile table above; each `case` is a literal transcription of its site. */
/* ORK_XPROF diagnostic: count how often each profile is entered from each predecessor mode, to
 * empirically test transition reachability (e.g. is XP_CHAIN_NT ever entered from F16?). Dumped by
 * ork_npu_xprof_dump() at teardown. Zero cost when off (one cached getenv). */
long orki_xcount[XP_NPROFILE][8]; int orki_xprof=-1;
const char *orki_XPNAME[XP_NPROFILE]={"MC_MM","SC_MM","CHAIN_NT","STREAM_I8","STREAM_F16",
    "I4_MC","I4_MWARM","I4_INCR","I4CHAIN","I4_STREAM","SDP"};
const char *orki_XFROM[8]={"COLD","F16","I8","I4","CHAIN","SDP?","I4STRM","?"};
/* fp16 multicore matmul (Sn==1) doorbell colsplit — DEFAULT ON (2026-08-05). Bit-exact and 1.04-1.23x faster than
 * mcworker (A/B, governors-verified). The K-split (Sk>1) drop that used to WEDGE was root-caused as a concurrent
 * CROSS-BUFFER weight-fetch wild (HW prefetches Bb[ks+1] while Bb[ks] drains) and is now PREVENTED by CONTIG (one
 * contiguous weight buffer -> no dma-buf boundary -> no wild -> no drop; validated 1000-iter 0-drop + make test).
 * CONTIG is default-on for Sn==1 inside ork_dyn_begin_colsplit; it is the ONLY fp16 multicore path (#45) — the
 * legacy mcworker fallback has been removed. See NPU-Quirks "fp16 3-core colsplit drop" + Exp-2026-08-05-fp16-Colsplit-CONTIG. */
static int run_multicore(ork_npu *c,ork_w *w,int M,const void *A,void *C,int nc){
    int dt=w->dtype, fd=c->fd;
    /* never exceed the hardware (or the buffer-array bound) — a bad ORK_NPU_MC can't over-index */
    if(nc>c->soc->cores) nc=c->soc->cores;
    if(nc>ORK_MAXCORE)  nc=ORK_MAXCORE;
    if(nc<1) nc=1;
    /* #33 TEST / A-B hook: force a shape that HAS pre-built tiles (w->sliced) onto the slice rescue even
     * when its native path would work — to validate the rescue is bit-exact and to A/B its throughput
     * against the native path. Off by default (only fires with tiles present AND the env set). */
    if(dt==DT_I8 && w->sliced && getenv("ORK_FORCE_SLICE_RESCUE"))
        return orki_slice_rescue_or_refuse(c,w,M,A,C,nc);
    /* P3 MIGRATION: int8 matmul (Sn==1, K<=4096 with full-K Bf, nc>1) runs on the NONBLOCK doorbell via
     * ork_dyn_begin_colsplit — sub-nmax N-column split across cores (matching mcworker's t0=i*NN/nc bit-exact),
     * M-tiled within each core. Decode (M=1) is dispatch-bound => faster; prefill (M>1) is compute-bound =>
     * parity, once the ork_dyn_end heavy-job poll backoff removed the busy-poll civac contention with the NPU
     * writeback (M=256 was 2.8x slower on a pure spin; now at parity). Consolidates decode AND prefill submits
     * onto the spine (dump/self-heal coverage). No legacy fallback — git is the recovery. */
    /* COMPLETE MIGRATION: int8 multi-core matmul runs ONLY on the NONBLOCK doorbell — the blocking mcworker
     * fallback is a wedge-prone footgun (a doorbell miss self-heals; a blocking-submit miss hard-wedges the NPU
     * and needs a reboot). The doorbell's PRECISE verified bound is <=64 rows/program: base (Sn==1,K<=4096) M-tiles
     * internally to mg_max*64; wide-N (Sn>1, N-strided) and wide-K (K>4096, K-split) are verified to M<=64/program,
     * so we ADAPT prefill by M-tiling into <=64-row doorbell submits (rows are independent -> bit-exact). A shape
     * outside the envelope returns an ERROR — we never silently fall to the blocking path. */
    { int i8 = (dt==DT_I8 && (w->N/32)>=2 && nc>1);
      /* COLSPLIT IS THE DEFAULT (any M) for base, wide-N and wide-K. The old M==1 gate on wide-N/wide-K (behind
       * ORK_COLSPLIT_MGT1) existed because 23af039's first M>1 colsplit regressed prefill ~15x (per-K-slice
       * partials + a SERIAL host accumulate). That is FIXED: balanced boundary-split (no notch, no load-
       * imbalance) + PER-CORE PARALLEL ks-outer accumulate + gather-A-once now make colsplit BEAT the mcworker
       * chain on the 7B (75 vs 73 t/s, bit-exact) AND it is self-healing (a blocking mcworker miss hard-wedges
       * the NPU). The MGT1 gate is removed; the legacy blocking mcworker fall-through has been removed (#45). */
      int r_base  = i8 && w->Sn==1 && w->K<=4096 && w->Bf;   /* any M (internal mg_max*64 M-tiling) */
      int r_wideN = i8 && w->Sn>1 && w->K<=4096 && w->Bf;    /* wide-N ffn gate/up: colsplit, any M */
      int r_wideK = i8 && w->Sn==1 && (w->K>4096 || !w->Bf);  /* wide-K ffn down (K>4096) OR no-Bf K<=4096 (ORK_NO_BF FFN-chain): both ride the Bf-FREE Bb K-split colsplit, any M — removes the int8 no-Bf Sn==1 mcworker fall-through (#48) */
      if(r_base || r_wideN || r_wideK){
        /* ONE doorbell submit: ork_dyn_begin_mc -> ork_dyn_begin_colsplit auto-decomposes base (M-tiled),
         * wide-N (Sn>1 N-sliced, M-tiled) and wide-K (K>4096 K-split, M-tiled) across cores, all any-M. */
        ork_mm_task_i8 t = { .w=w, .M=M, .A=(const int8_t*)A, .C=(int32_t*)C };
        ork_dyn_chain *h = ork_dyn_begin_mc(c, 1, &t, nc);
        if(!h) return orki_slice_rescue_or_refuse(c,w,M,A,C,nc);  /* #33: run pre-built doorbell tiles if any, else refuse (never wedge-fallback) */
        return ork_dyn_end(h) < 0 ? -1 : 0;
      }
      if(i8 && w->Sn>1 && (w->K>4096 || !w->Bf)){
        /* int8 WIDE-N with no single-submit base (no Bf, or K>4096): serve each N-slice as a standalone Sn==1
         * K-split colsplit (Bf-free — the validated wide-K path, per slice). cstride=N writes each slice's sub-N
         * result into the wider C at the full row stride. With r_base/r_wideN/r_wideK covering everything else,
         * this closes the LAST int8 M>1 gap (no-Bf Sn>1 and Sn>1&K>4096) — int8 no longer falls to mcworker (#48). */
        if(w->Sk>128) return orki_slice_rescue_or_refuse(c,w,M,A,C,nc);       /* #33: slice-view array bound — run pre-built tiles if any, else refuse */
        int NMAXn=c->soc->nmax, ok=1;
        for(int ns=0; ns<w->Sn && ok; ns++){
            int c0=ns*NMAXn, sw=(w->N-c0<NMAXn)?(w->N-c0):NMAXn;
            struct buf vbb[128];
            for(int ks=0; ks<w->Sk; ks++) vbb[ks]=w->Bb[(size_t)ns*w->Sk+ks];   /* this slice's Sk K-slice tiles */
            ork_w wv=*w; wv.N=sw; wv.Sn=1; wv.Bb=vbb; wv.Bf=NULL;               /* Sn==1 no-Bf view -> forces the Bf-free K-split */
            ork_mm_task_i8 tf={ .w=&wv, .M=M, .A=(const int8_t*)A, .C=(int32_t*)((char*)C+(size_t)c0*4), .cstride=w->N };
            ork_dyn_chain *hs=ork_dyn_begin_mc(c,1,&tf,nc);
            if(!hs || ork_dyn_end(hs)<0) ok=0;
        }
        if(ok) return 0;
        return orki_slice_rescue_or_refuse(c,w,M,A,C,nc);   /* #33: a slice ineligible/failed -> run pre-built tiles if any, else refuse */
      }
      if(i8 && M==1){   /* DECODE i8 with no doorbell envelope: reject (no safe decode fallback). With the wide-N slice path above + r_*, this is now a backstop (unreachable for standard shapes). */
        if(w->sliced){ int rs=ork_mm_run_sliced(c,w->sliced,M,A,C,nc); if(rs>=0) return rs; }   /* #33: rescue on pre-built tiles before the scary refuse */
        fprintf(stderr, "[ork] int8 decode M=%d K=%d N=%d Sn=%d Bf=%d has no verified doorbell path; refusing the "
                "blocking fallback (would risk an unrecoverable NPU wedge) — rescue-eligible (ORK_RC_WEDGE_PRONE)\n", M, w->K, w->N, w->Sn, w->Bf?1:0);
        return ORK_RC_WEDGE_PRONE;
      }
      /* i8 M>1 wide-N/wide-K are fully covered by r_wideN/r_wideK/slice above; nothing falls through (the mcworker CHAIN path is removed #45). */ }
    if (dt == DT_F16 && ork_f16_colsplit() && nc > 1 && w->Sn > 1 && (w->N/16) >= 2 && w->Sk <= 64
        && !getenv("ORK_COLSPLIT_SERIAL") && !getenv("ORK_F16_NO_WIDEN")) {
        /* fp16 WIDE-N (Sn>1): per-N-slice CONTIG colsplit. Each N-slice is served as a standalone Sn==1 CONTIG
         * problem — its Sk K-slice tiles (Bb[ns*Sk+ks]) concatenated into ONE resident buffer (Bbc_ns[ns]) so the
         * per-core K-chain never crosses a dma-buf boundary (the cross-buffer prefetch WILD that CONTIG prevents).
         * Cores split THAT slice's columns; end() writes the sub-N result into the wider C at the full row-stride N
         * via task.cstride. Sn sequential begin/end. Any ineligible/wedged slice abandons colsplit for this matmul
         * and falls through to orki_run()'s single-core fp16 reference via ORK_RC_F16_SC (correctness). This removes fp16 wide-N's mcworker dependency
         * (task #45) using the validated Sn==1 no-drop path per slice. */
        int NMAXn = c->soc->nmax, KSn = c->soc->ks;
        if (!w->Bbc_ns_valid) {   /* build the per-N-slice CONTIG weights ONCE (resident; reclaimed at teardown like Bbc) */
            w->Bbc_ns = calloc((size_t)w->Sn, sizeof(struct buf));
            int build_ok = (w->Bbc_ns != NULL);
            for (int ns = 0; ns < w->Sn && build_ok; ns++) {
                int c0 = ns*NMAXn, sw = (w->N - c0 < NMAXn) ? (w->N - c0) : NMAXn;
                size_t tot = 0;
                for (int ks = 0; ks < w->Sk; ks++) { int k0 = ks*KSn, Kp = (w->K-k0<KSn)?(w->K-k0):KSn; tot += (size_t)Kp*sw*2; }
                w->Bbc_ns[ns] = orki_bcreate(fd, tot, 0x403, w->domain);
                if (!w->Bbc_ns[ns].cpu) { build_ok = 0; break; }
                size_t off = 0;
                for (int ks = 0; ks < w->Sk; ks++) { int k0 = ks*KSn, Kp = (w->K-k0<KSn)?(w->K-k0):KSn; size_t sz = (size_t)Kp*sw*2;
                    memcpy((char*)w->Bbc_ns[ns].cpu + off, w->Bb[(size_t)ns*w->Sk + ks].cpu, sz); off += sz; }
                orki_bsync(fd, &w->Bbc_ns[ns], RKNPU_MEM_SYNC_TO_DEVICE);
            }
            if (build_ok) w->Bbc_ns_valid = 1;   /* partial/failed build -> stays invalid; the slice loop bails to the single-core fp16 reference */
        }
        if (w->Bbc_ns_valid) {
            ork_install_term();
            int wideN_ok = 1;
            for (int ns = 0; ns < w->Sn && wideN_ok; ns++) {
                int c0 = ns*NMAXn, sw = (w->N - c0 < NMAXn) ? (w->N - c0) : NMAXn;
                struct buf vbb[64];   /* slice-view K-slice tiles (w->Sk <= 64 gated above) */
                for (int ks = 0; ks < w->Sk; ks++) vbb[ks] = w->Bb[(size_t)ns*w->Sk + ks];
                ork_w wv = *w; wv.N = sw; wv.Sn = 1; wv.Bb = vbb;
                wv.Bbc = w->Bbc_ns[ns]; wv.Bbc_valid = 1;   /* use the cached per-slice CONTIG (colsplit skips its own Sn==1 build) */
                wv.Bgap_valid = 0; memset(wv.Bgap, 0, sizeof wv.Bgap);
                ork_mm_task_i8 tf = { .w = &wv, .M = M, .A = (const int8_t*)A,
                                      .C = (int32_t*)((char*)C + (size_t)c0*4), .cstride = w->N };
                c->mc_error = 0;
                ork_dyn_chain *hs = ork_dyn_begin_colsplit(c, &tf, nc);
                if (!hs) wideN_ok = 0;
                else if (ork_dyn_end(hs) < 0 || c->mc_error) wideN_ok = 0;
            }
            if (wideN_ok) return 0;
            fprintf(stderr, "[ork] fp16 wide-N colsplit ineligible/wedge (K=%d N=%d Sn=%d) — single-core fp16 reference backstop\n", w->K, w->N, w->Sn);
            c->mc_error = 0;   /* fall through to orki_run()'s single-core fp16 reference (ORK_RC_F16_SC) for the whole matmul */
        }
    }
    if (dt == DT_F16 && ork_f16_colsplit() && nc > 1 && w->Sn == 1 && (w->N/32) >= 2 && !getenv("ORK_COLSPLIT_SERIAL")) {   /* fp16 SW-chain needs the parallel per-core worker (per-K-slice submits); serial inline path can't run the boundary-broken chain -> single-core fp16 reference */
        /* Stage 1: fp16 Sn==1 rides the doorbell colsplit (bit-exact f32 K-slice accumulate). Call colsplit
         * DIRECTLY (not ork_dyn_begin_mc — that entry also serves SSM stream/pool fp16 callers we must not
         * touch). h==NULL (ineligible / buffers too small) FALLS BACK to orki_run()'s single-core fp16 reference (ORK_RC_F16_SC). */
        ork_install_term();
        ork_mm_task_i8 tf = { .w = w, .M = M, .A = (const int8_t*)A, .C = (int32_t*)C };
        int ncf = nc; if (ncf > c->soc->cores) ncf = c->soc->cores;
        /* fp16 STALL DETECT + RECOVER (copies int8's orki_mc_recover_resubmit skeleton — RKNPU_ACT_RESET -> 1ms quiesce
         * -> resubmit — as a SINGLE-THREADED run-level coordinator, resubmitting the whole colsplit. int8 replays one
         * stashed submit/core; fp16 can't (its per-slice submits have no single stashed submit + a nonblock re-plumb
         * would force the 5-9x-slower serial ork_dyn_end accumulate), so the resubmit unit is the whole colsplit —
         * same detect->reset->quiesce->resubmit-same pattern, fast per-core accumulate preserved. The wild is
         * intermittent (~1/10-25) like int8's dispatch drop, so a clean-reset resubmit lands with high probability.
         * Single-threaded (NOT 3 concurrent per-core retries — that hammered a mid-reset NPU into a HARD wedge). */
        /* DEFAULT 0 = on wedge go STRAIGHT to nc=1 (no resubmit-same): nc=1 is the bit-exact reference so it GUARANTEES
         * correct output (fixes the ~1.7% wrong-answers the resubmit-same path accepted), and it never hammers a
         * mid-reset NPU (the resubmit-same x6 was the HARD-wedge risk). ORK_F16_RESUB=<n> re-enables n resubmit-same
         * attempts for A/B. Blocking stays (nonblock re-wedges fp16). */
        int fp16_sentinel_r = getenv("ORK_F16_SENTINEL") != NULL;   /* nonblock sentinel path (fast detect) */
        /* Task #50: ork_dyn_begin_colsplit now heals a dropped fp16 K-slice IN PLACE (reset+reseed+re-launch the workers
         * over the same buffers, up to ORK_F16_RESUB tries) — the int8 orki_mc_recover_resubmit model. So the run-level loop
         * must NOT also rebuild-via-begin (the Bug#2 poison path): recov_max=0 => one begin (which heals internally), then
         * residual mc_error -> nc=1 bit-exact backstop. ORK_F16_REBUILD re-enables the old rebuild-retry for A/B only. */
        int fp16_recov_max = getenv("ORK_F16_REBUILD") ? (getenv("ORK_F16_RESUB") ? atoi(getenv("ORK_F16_RESUB")) : 6) : 0, fp16_healed = 0, fp16_ineligible = 0;
        int fp16_block_heal = getenv("ORK_F16_BLOCK_HEAL") != NULL;   /* HYBRID: attempt 0 nonblock (fast detect), retries BLOCKING so the kernel watchdog REAPS the sticky slice-1 drop (userspace ACT_RESET can't) */
        for (int attempt = 0; attempt <= fp16_recov_max; attempt++) {
            c->mc_error = 0;   /* clear BEFORE begin (the parallel workers run inside begin + set it on a wedged submit) */
            c->f16_force_blocking = (fp16_block_heal && attempt > 0) ? 1 : 0;   /* nonblock-detect + blocking-heal */
            ork_dyn_chain *h = ork_dyn_begin_colsplit(c, &tf, ncf);
            if (!h) { fp16_ineligible = 1; break; }   /* ineligible / buffers too small -> single-core fp16 reference */
            int rc = ork_dyn_end(h);
            if (rc >= 0 && !c->mc_error) { fp16_healed = 1; break; }   /* landed clean (kernel-transparent resets included) */
            /* WEDGE POST-MORTEM: capture the HW fault signature (int_status/hw_elapse/iommu/freeSRAM) at the moment of
             * detection, BEFORE any reset/resubmit destroys it (per ork_npu_dump_state's "fire before a wedge loses it"
             * contract). fflush so it streams out even if the box hard-hangs next. */
            fprintf(stderr, "[F16-WEDGE run-level] attempt %d/%d rc=%d mc_error=%d — pre-recovery HW state:\n", attempt, fp16_recov_max, rc, c->mc_error);
            ork_kmsg("F16-WEDGE attempt %d/%d rc=%d mc_error=%d (K=%d N=%d M=%d)", attempt, fp16_recov_max, rc, c->mc_error, w->K, w->N, M);
            ork_npu_dump_state(c, "fp16-wedge PRE-recovery"); fflush(stderr);
            if (getenv("ORK_F16_FDCLOSE")) {   /* PROBE (ORK_F16_FDCLOSE): does closing the DRM fd REAP the poisoned mapping
                * CLEANLY (drm_release cancels the stuck job + tears down its IOMMU maps), unlike our orki_bsync (which NULL-derefs
                * in rknpu_gem_sync_ioctl)? Isolate: close -> reopen -> _exit(0) so the answer isn't masked by stale-buffer
                * teardown. netconsole shows whether a gem_sync/still-mapped fires between 'begin' and 'reopened'. */
                int oldfd = c->fd;
                ork_kmsg("FDCLOSE-REAP begin: close(fd=%d) to force drm_release job-cancel + IOMMU unmap", oldfd);
                close(oldfd);
                ork_kmsg("FDCLOSE-REAP: close() returned WITHOUT synchronous crash — reopening %s", c->soc->card);
                int nf = open(c->soc->card, O_RDWR);
                if (nf >= 0) { orki_act(nf, RKNPU_POWER_ON, 0); }
                ork_kmsg("FDCLOSE-REAP: reopened fd=%d (errno=%d) — close+reopen SURVIVED; _exit(0) (skip stale teardown)", nf, nf<0?errno:0);
                fflush(NULL);
                _exit(0);
            }
            if (attempt < fp16_recov_max && getenv("ORK_F16_FDREAP")) {   /* FD-REAP recovery (task #47): the CLEAN nonblock-safe
                * reap. close(fd)+reopen tears down the poisoned mapping via drm_release (RKNPU_ACT_RESET can't); the dma-buf
                * weights (ORK_F16_IMPORT_W) survive + are re-imported in place, scratch lazily rebuilds. Then DE-ESCALATE
                * this one op to nc=1 (single-core, no concurrent fetch) — do NOT retry the concurrent colsplit, which can
                * RE-DROP -> another reap -> rapid close/reopen CHURN that faults the kernel IOMMU. Reap cleans the poison;
                * nc=1 guarantees the retry lands first try (the bit-exact reference). Reap-mechanism proven by REAP_TEST. */
                ork_kmsg("F16-WEDGE attempt %d/%d -> FD-REAP + de-escalate to nc=1", attempt+1, fp16_recov_max);
                if (ork_ctx_fd_reap(c) < 0) fprintf(stderr, "[ork] fp16 FD-REAP failed (context unusable)\n");
                break;   /* fall through to the nc=1 de-escalation below (single-core fp16 reference on the reaped+re-imported ctx) */
            }
            if (attempt < fp16_recov_max && getenv("ORK_F16_TCLEAN")) {   /* TIMEOUT-CLEAN recovery (task #47, the driver's
                * intended path): NO reset, NO fd-close. Just retry the colsplit — its nonblock submits call
                * rknpu_job_timeout_clean at the top, which CLEANLY reaps THIS attempt's dropped (now timed-out) slice job
                * (soft-reset + schedule cleanup_work -> proper task_obj put) before dispatching. Buffers stay alive so the
                * deferred cleanup never UAFs (that was the fd-close bug). Mirrors int8 orki_mc_recover_resubmit. */
                ork_kmsg("F16-WEDGE attempt %d/%d -> TCLEAN retry (resubmit triggers timeout_clean reap)", attempt+1, fp16_recov_max);
                continue;
            }
            if (attempt < fp16_recov_max) {   /* wedge (a core's submit dropped): N resets (drain between) + resubmit-same.
                * The fp16 slice-1 doorbell drop is STICKY across a SINGLE RKNPU_ACT_RESET (proven: 6/6 resubmits re-drop);
                * ORK_F16_RESET_N>1 tests whether reset->drain->reset clears the persistent CDMA/IOMMU state a single reset
                * leaves mid-abort. Each reset is followed by ORK_F16_COOLDOWN_MS drain. */
                const char *rne = getenv("ORK_F16_RESET_N"); int reset_n = rne ? atoi(rne) : 1; if (reset_n < 0) reset_n = 0;   /* 0 = NO reset (test whether a blocking-heal resubmit reaps on its own); 2 = reset->drain->reset */
                const char *cde = getenv("ORK_F16_COOLDOWN_MS"); long cd_ms = cde ? atol(cde) : 50;
                fprintf(stderr, "[ork] fp16 colsplit wedge (K=%d N=%d M=%d) attempt %d/%d — %d× RKNPU_ACT_RESET (%ldms drain each) + resubmit\n", w->K, w->N, M, attempt+1, fp16_recov_max, reset_n, cd_ms);
                for (int rr = 0; rr < reset_n; rr++) {
                    struct rknpu_action ra; memset(&ra, 0, sizeof ra); ra.flags = RKNPU_ACT_RESET; ioctl(c->fd, DRM_IOCTL_RKNPU_ACTION, &ra);
                    ork_kmsg("attempt %d reset %d/%d + %ldms drain", attempt+1, rr+1, reset_n, cd_ms);
                    struct timespec qs = {(time_t)(cd_ms/1000), (long)(cd_ms%1000)*1000000L}; nanosleep(&qs, NULL);
                }
                c->warmed = 0; for (int z = 0; z < ORK_MAXCORE; z++) c->mwarm[z] = 0;   /* reset cleared the NPU's warm/regcmd state -> force cold re-warm */
            }
        }
        c->f16_force_blocking = 0;   /* clear the hybrid override so it never leaks into a later matmul */
        if (fp16_healed) return 0;
        if (!fp16_ineligible) {   /* recovery exhausted (repeated wilds survive resets = a genuinely stuck NPU): FINAL
            * BACKSTOP = de-escalate to single-core (nc=1: no concurrent fetch -> the bit-exact reference, never wedges).
            * Slow but correct + safe. Falls through to orki_run()'s single-core fp16 reference (ORK_RC_F16_SC) with a cold re-warm. */
            fprintf(stderr, "[ork] fp16 colsplit wedge (K=%d N=%d M=%d) — de-escalating to single-core\n", w->K, w->N, M);
            if (getenv("ORK_F16_TCLEAN")) {   /* TCLEAN: the last colsplit attempt left dropped/stuck jobs on ALL cores; nc=1
                * below only submits to core 0, so cores 1..n would keep a stuck job that UAFs at process teardown (close(fd)
                * frees the buffer under it). Reap ALL cores' stuck jobs the CLEAN way (per-core nonblock dummy -> timeout_clean)
                * so nothing lingers. This is what makes the nonblock recovery teardown-safe WITHOUT fd-close. */
                ork_npu_reap_stuck(c, ncf);
            } else if (fp16_sentinel_r) {   /* legacy sentinel path: single reset (does NOT cleanly reap — see Bug#2) */
                struct rknpu_action ra; memset(&ra, 0, sizeof ra); ra.flags = RKNPU_ACT_RESET; ioctl(c->fd, DRM_IOCTL_RKNPU_ACTION, &ra);
                struct timespec qs = {0, 1000000}; nanosleep(&qs, NULL); }
            c->mc_error = 0; nc = 1;
            c->warmed = 0; for (int z = 0; z < ORK_MAXCORE; z++) c->mwarm[z] = 0;
        }
        /* fp16_ineligible: fall through to orki_run()'s single-core fp16 reference (ORK_RC_F16_SC) at the original nc. */
    }
    /* fp16 NEVER falls to the blocking mcworker: every fp16 fallback (colsplit ineligible, wedge de-escalation,
     * Sn>1 slice-fail) routes to orki_run()'s SINGLE-CORE fp16 reference (bit-exact, no concurrent fetch -> no drop)
     * via ORK_RC_F16_SC. Removes the last fp16 mcworker dependency (#45). int8 is already fully covered/refused
     * above — the blocking mcworker path that used to sit below has been removed (#45). */
    if(dt==DT_F16) return ORK_RC_F16_SC;   /* fp16 out-of-colsplit fall-through: orki_run()'s single-core reference (correct; the tiled fp16 slice was a wrong-fit — fp16's fit is colsplit, above) */
    /* mcworker path deleted (#45). Nothing reaches here: fp16 returned ORK_RC_F16_SC above;
     * every int8 M>1 (nc>1, N>=64) returned in the i8 colsplit/refuse block; int8 N<64 never
     * reaches run_multicore (orki_run() shrinks nc->1). Refuse defensively, never a blocking fallback. */
    return orki_slice_rescue_or_refuse(c,w,M,A,C,nc);   /* #33: run pre-built doorbell tiles if any, else refuse */
}


int orki_run(ork_npu *c,ork_w *w,int M,const void *A,void *C){
    /* multi-domain residence: swap in this domain's scratch (regcmd/task/Af/Cc/mc-*) so the submit's
     * buffers all live in the weight's domain (c->dom_active), and make any lazy scratch bcreate below
     * land there too. Submits stamp their own iommu_domain_id from w->domain (per-submit, no global).
     * No-op for the common single-domain case (w->domain==0, dom_active==0). */
    if(w->domain!=c->dom_active || (w->domain!=0 && !c->dom_save)) orki_dom_activate(c,w->domain);
    /* auto-tuner: pick cores ≤ budget, capped so each gets ≥2 N-tiles (tiny matmuls don't pay the
     * multi-core spawn). budget defaults to all soc cores; ORK_NPU_MC / set_core_budget cap it. */
    /* int8 M=1 decode: use the multi-core orki_budget (split N across cores) — validated +40% end-to-end, every
     * shape benefits (in-model sweep monotonic). orki_budget(c,2) skips the M==1 single-core default + honors
     * ORK_NPU_MC. fp16/int4 M==1 keep single-core (orki_budget(c,1)==1). NN<nc*2 shrink below guards tiny int8 N. */
    int b=(M==1 && w->dtype==DT_I8) ? orki_budget(c,2) : orki_budget(c, M), cores=c->soc->cores, NN=w->N/(w->dtype?32:16);
    int nc=b<cores?b:cores; if(nc>NN)nc=NN; while(nc>1 && NN<nc*2)nc--;
    /* NOTE: imported weights on a non-0 IOMMU domain run MULTI-CORE safely — the per-domain native anchor
     * (ork_dom_prime, called at import time) establishes the domain so the spawned cores read correct IOVAs.
     * Before the anchor, multi-core imports non-deterministically corrupted output (C[last] wrong, dropped
     * K-slices) — that was a fresh-domain establishment race, NOT a core issue (single-core corrupted too).
     * No single-core gate is needed; see the >4GiB-import notes (wiki Tier 10 / NPU-Quirks). */
    if(nc>1){ int rmc=run_multicore(c,w,M,A,C,nc); if(rmc!=ORK_RC_F16_SC) return rmc; }   /* ORK_RC_F16_SC: fp16 fallback -> fall through to the single-core fp16 reference below (no mcworker) */
    orki_pin_big_core(0);                                   /* single-core path also runs on the calling thread */
    int fd=c->fd,K=w->K,N=w->N, dt=w->dtype, NMAX=c->soc->nmax, CBUF=c->soc->cbuf_elems;
    if(dt==DT_F16 && CBUF>32768) CBUF=32768;   /* int8-only cbuf raise; fp16 keeps its validated 32768 tiling (see the fp16 colsplit path) */
    int KS=dt ? orki_int8_ks(c) : c->soc->ks, RB=dt?2*CBUF:CBUF;     /* rows budget: int8 packs 2x rows/CBUF */
    /* entering int8 mode wedges the first submit unless the NPU is reset first (fp16 never
     * wedges — it cold-starts stale, which the warmup handles). Reset only when switching INTO
     * int8 — keeps fp16-only contexts free of any reset/log. Then re-warm on a fresh buffer. */
    /* ORK_MIXED_NOTHRASH extended to fp16: don't ccsz=0 (which forces a Cc REALLOC) on a dtype switch — under
     * near-full-domain IOVA pressure that per-layer realloc bcreate-FAILS -> orki_run() returns -1 (single-core) or
     * races the reset -> WEDGE (multi-core). Reusing the (per-domain, dom_activate-swapped) Cc when it still
     * fits avoids both. warmed=0 still re-warms (handles the stale-first-output). The realloc guard below still
     * reallocs on a genuine size grow. */
    ork_npu_enter(c,dt,XP_SC_MM,OCK_NONE);
    size_t need=(size_t)M*N*4;                         /* output is fp32 or int32 (both 4 bytes) */
    if(c->cressz<need){c->cres=realloc(c->cres,need);c->cressz=need;}
    memset(c->cres,0,need);
    size_t maxout=0, maxaf=0;
    for(int k0=0;k0<K;k0+=KS){
        int Kp=(K-k0<KS)?(K-k0):KS;
        int sd=dt?(Kp==1024||Kp==512):((Kp&(Kp-1))==0 && Kp>=128 && Kp<(getenv("ORK_F16_HISCHED")?4096:2048));
        int R=RB/Kp; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
        double scale=(double)Kp/(dt?512.0:256.0); int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale), mg_max = base>=0x1b ? (base-0x1b)/slope+1 : 0;
        int chunk = mg_max * 64; if(!sd) chunk = (RB/2)/Kp; if(chunk < 4*R) chunk = sd ? 4*R : ((RB/2)/Kp); if(chunk > M) chunk = M; if(chunk < 1) chunk = 1;
        int rows=chunk<M?chunk:M;
        int nc=N<NMAX?N:NMAX;
        size_t o=(size_t)rows*nc*4; if(o>maxout)maxout=o;
        size_t sz=(size_t)rows*Kp*(dt?1:2); if(sz>maxaf)maxaf=sz;
    }
    if(dt==DT_I8 && M>1 && w->Bf && (K%512)==0 && K<=4096){
        int Kp=K, R=RB/Kp; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
        double scale=(double)Kp/512.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale), mg_max = base>=0x1b ? (base-0x1b)/slope+1 : 0;
        int chunk = mg_max * 64; if(chunk < 1) chunk = 1; if(chunk > M) chunk = M;   /* M-tile = the 0x1040 schedule's validated max rows (mg_max*64). NOT R-1: R=pow2_floor(2*cbuf/K) was a FALSE "CBUF-resident rows" cap (~31) that re-streamed the K*N weight from DRAM ~2-4x too often (single-core is weight-DMA-bound). mg_max*64 is the exact bit-exact ceiling (mc+1 miscomputes). See AGENTS.md "weight-DMA amortization". */
        int rows=chunk<M?chunk:M;
        size_t sz=(size_t)rows*Kp*1;
        if(sz>maxaf)maxaf=sz;
    }
    if(c->ccsz<maxout){orki_bdestroy(fd,&c->Cc);c->Cc=orki_bcreate(fd,maxout,0x403,c->dom_active);c->ccsz=maxout;c->warmed=0; if(!c->Cc.cpu){fprintf(stderr, "[ork] ERROR: failed to allocate single-core/pre-core output buffer Cc (size=%zu, IOMMU full?)\n", maxout);return -1;}}
    if(c->Af.size<maxaf){
        orki_bdestroy(fd,&c->Af);
        c->Af=orki_bcreate(fd,maxaf,0x403,c->dom_active);
        if(!c->Af.cpu){
            fprintf(stderr, "[ork] ERROR: failed to allocate activation buffer Af (size=%zu, IOMMU full?)\n", maxaf);
            return -1;
        }
    }
    /* Tier 1c-ii: full-K prefill — one submit per M-tile over the FULL K (Bf layout), M-scheduler on,
     * result written directly (no K-split, no host accumulate). Saves the second K-slice's accumulate +
     * result cache-sync. Gated (M-scheduler at Kp=K unvalidated) — verify with examples/quant. */
    if(dt==DT_I8 && M>1 && w->Bf && (K%512)==0 && K<=4096){   /* the M-scheduler is now enabled for non-pow2 Kp (e.g. K=6144) via continuous scheduling */
        int Kp=K, sched=1, R=RB/Kp; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
        double scale=(double)Kp/512.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale), mg_max = base>=0x1b ? (base-0x1b)/slope+1 : 0;
        int chunk = mg_max * 64; if(chunk < 1) chunk = 1; if(chunk > M) chunk = M;   /* M-tile = the 0x1040 schedule's validated max rows (mg_max*64). NOT R-1: R=pow2_floor(2*cbuf/K) was a FALSE "CBUF-resident rows" cap (~31) that re-streamed the K*N weight from DRAM ~2-4x too often (single-core is weight-DMA-bound). mg_max*64 is the exact bit-exact ceiling (mc+1 miscomputes). See AGENTS.md "weight-DMA amortization". */
        /* DEBUG knob (ORK_MCAP, default off): force the M-tile (rows/submit) size. The default above is
         * now the validated optimum (mg_max*64), so this only lowers it (diagnostics) or — if set above
         * mg_max*64 — miscomputes. Was used to discover the weight-DMA-amortization lever; kept for RE. */
        { static int mcap=-2; if(mcap==-2){const char*e=getenv("ORK_MCAP"); mcap=e?atoi(e):-1;}
          if(mcap>0){ chunk=mcap; if(chunk>M)chunk=M; if(chunk<1)chunk=1; } }
        /* zero-copy: if A / C live in ork_dma_alloc buffers, the regcmd reads/writes them in place
         * (no gather/writeout memcpy). Output zero-copy needs a single N-slice (Nc==N, contiguous).
         * Opt-in (ORK_ZC_OUT) + off by default. */
        struct buf *abuf=orki_dma_find(c,A);   /* INPUT zero-copy: validated correct, default on */
        struct buf *cbuf=(getenv("ORK_ZC_OUT"))?orki_dma_find(c,C):NULL;
        if(abuf) orki_bsync(fd,abuf,RKNPU_MEM_SYNC_TO_DEVICE);   /* flush the producer's CPU writes once */
        if(cbuf) {
            orki_bsync(fd,cbuf,RKNPU_MEM_SYNC_TO_DEVICE);   /* clean dirty CPU cache lines so they don't evict over NPU output */
            c->warmed=0;             /* re-warm the fresh output buffer (necessary but not sufficient) */
        }
        for(int ns=0;ns<w->Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
            uint64_t wbase=w->Bf[ns].dma;                  /* full-K weight, whole N-slice (single core) */
            for(int m0=0;m0<M;m0+=chunk){int mc=(M-m0<chunk)?(M-m0):chunk; if(mc<=0)continue;
                double _tc0=ork_now_us();
                uint32_t adma;
                if(abuf){ adma=(uint32_t)(abuf->dma + ((const char*)A-(const char*)abuf->cpu) + (size_t)m0*K); }
                else { int8_t*ad=c->Af.cpu; const int8_t*Ai=A; for(int r=0;r<mc;r++)for(int j=0;j<K;j++) ad[(size_t)r*K+j]=Ai[(size_t)(m0+r)*K+j];
                       orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE); adma=(uint32_t)c->Af.dma; }
                double _ts0=ork_now_us(); orki_mc_copy[0]+=_ts0-_tc0;
                uint32_t cdma=cbuf?(uint32_t)(cbuf->dma + ((const char*)C-(const char*)cbuf->cpu) + (size_t)m0*N*4):(uint32_t)c->Cc.dma;
                uint32_t rc[REGCMD_N]; orki_synth_i8(rc,mc,Kp,Nc,adma,(uint32_t)wbase,cdma,sched,CBUF,cbuf?N:Nc);
                if (orki_validate_regcmd("run_fullk_dec", c, rc, REGCMD_N, w, NULL, 0)) return -1;
                memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
                if(cbuf){ if(orki_submit1(c)) return -1; }                       /* ZC-OUT writes user C, not c->Cc -> keep blocking */
                else    { if(orki_submit1_db(c,(size_t)mc*Nc)) return -1; }      /* P3 #7: c->Cc int32 output rides the doorbell */
                double _ta0=ork_now_us(); orki_mc_sub[0]+=_ta0-_ts0;
                
                /* For output zero copy, the NPU writes directly to the user-provided C buffer.
                 * We MUST invalidate the CPU cache here so the host reads the fresh NPU output instead of stale cache lines. */
                if(cbuf) orki_bsync(fd,cbuf,RKNPU_MEM_SYNC_FROM_DEVICE);
                
                if(!cbuf){ int32_t*cc=c->Cc.cpu,*cr=c->cres; for(int r=0;r<mc;r++)for(int n=0;n<Nc;n++) cr[(size_t)(m0+r)*N+(n0+n)]=cc[(size_t)r*Nc+n]; }
                orki_mc_acc[0]+=ork_now_us()-_ta0; orki_mc_n[0]++;
            }
        }
        if(cbuf){
            orki_bsync(fd,cbuf,RKNPU_MEM_SYNC_FROM_DEVICE);
            if(w->Sn>1){
                int32_t *C_ptr = (int32_t*)C;
                for (int r = M - 1; r >= 1; r--) {
                    for (int ns = w->Sn - 1; ns >= 0; ns--) {
                        int n0 = ns * NMAX;
                        int Nc = (N - n0 < NMAX) ? (N - n0) : NMAX;
                        int start_idx = n0 + Nc + (r - 1) * N;
                        int dest_idx = r * N + n0;
                        memmove(C_ptr + dest_idx, C_ptr + start_idx, (size_t)Nc * 4);
                    }
                }
            }
            return 0;
        }   /* result already in C's DMA buffer */
        memcpy(C,c->cres,need); return 0;
    }
    /* Establish the MATMUL task for submit1_db explicitly (int8). This path only refreshes c->regcmd contents
     * and had RELIED on c->task persisting as a matmul from a prior submit — but a preceding SDP/LUT op (e.g.
     * exp_i8 in the resident softmax chain) overwrites c->task with enable_mask=0x1d + its own regcmd, so the
     * following matmul submit ran the SDP task instead -> stale/garbage output (reproduced direct:
     * swreduce_probe T2 = MM_I8 after exp_i8). Set it here so the matmul never inherits an SDP task. */
    if(dt==DT_I8){ struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
        t->enable_mask=0xd; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=108; t->regcmd_addr=(uint32_t)c->regcmd.dma;
        orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); }
    for(int ns=0;ns<w->Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<w->Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;
        int sched=dt?(Kp==1024||Kp==512):((Kp&(Kp-1))==0 && Kp>=128 && Kp<(getenv("ORK_F16_HISCHED")?4096:2048)), R=RB/Kp; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; } int chunk=sched?4*R:((RB/2)/Kp); if(chunk<1)chunk=1;
        /* sched=0 uses the DEFAULT 0x1040 template, which computes correctly only while the activation tile
         * fits its budget: mc*Kp <= 32768 elements. (RB/2)/Kp overshoots (e.g. int8 K=256 -> chunk=224, but
         * rows past 32768/256=128 in one submit are GARBAGE — isolated via shape_probe). The sched=1 path
         * caps to mg_max*64; sched=0 must cap to 32768/Kp. Without this, int8 K%512!=0 at M>32768/Kp miscomputes
         * (make test never hit it: test_matmul K=2048/3584 are %512 -> run_fullk_dec). */
        if(!sched){ int cap=32768/Kp; if(cap<1)cap=1; if(chunk>cap)chunk=cap; }
        /* fp16 M-scheduler (sched=1) has a VALID Kp WINDOW [128,2048): the 0x1040 K-reduction schedule (scale=Kp/256)
 * extrapolates too HIGH for small Kp (K=64->0x1040=188, K=32->190) and miscomputes (constant-garbage output),
 * and at Kp>=2048 it miscomputes >8 rows (mc<=8 OK / mc>=9 garbage). Outside the window, sched=0 (the general
 * path, no 0x1040 override) is correct — e.g. non-pow2 K=96 already took sched=0. Gates: Kp>=128 (low, fixes
 * fp16 attention at head_dim=32/64) and Kp<2048 (high). Validated in test_bmm (K-sweep 32..256). */
        struct buf*Bb=&w->Bb[(size_t)ns*w->Sk+ks];
        for(int m0=0;m0<M;m0+=chunk){int mc=(M-m0<chunk)?(M-m0):chunk; if(mc<=0)continue;
            { static int dbg=-1; if(dbg<0)dbg=getenv("ORK_RL_DBG")?1:0; if(dbg) fprintf(stderr,"[run_loop] ns=%d ks=%d Kp=%d sched=%d chunk=%d m0=%d mc=%d M=%d Nc=%d Afsz=%zu Ccsz=%zu\n",ns,ks,Kp,sched,chunk,m0,mc,M,Nc,c->Af.size,c->Cc.size); }
            double _tc0=ork_now_us();
            if(dt==DT_F16){ f16*ad=c->Af.cpu; const f16*Af=A; for(int r=0;r<mc;r++)for(int j=0;j<Kp;j++) ad[(size_t)r*Kp+j]=Af[(size_t)(m0+r)*K+k0+j]; }
            else { int8_t*ad=c->Af.cpu; const int8_t*Ai=A; for(int r=0;r<mc;r++)for(int j=0;j<Kp;j++) ad[(size_t)r*Kp+j]=Ai[(size_t)(m0+r)*K+k0+j]; }
            orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
            double _ts0=ork_now_us(); orki_mc_copy[0]+=_ts0-_tc0;
            uint32_t rc[REGCMD_N];   /* REGCMD_N == REGCMD_I8_N == 224 */
            if(dt==DT_F16) orki_synth   (rc,mc,Kp,Nc,(uint32_t)c->Af.dma,(uint32_t)Bb->dma,(uint32_t)c->Cc.dma,sched,CBUF);
            else           orki_synth_i8(rc,mc,Kp,Nc,(uint32_t)c->Af.dma,(uint32_t)Bb->dma,(uint32_t)c->Cc.dma,sched,CBUF,Nc);
            if (orki_validate_regcmd("run_loop", c, rc, REGCMD_N, w, NULL, 0)) return -1;
            memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
            if(orki_submit1_db(c,(size_t)mc*Nc)) return -1;   /* P3 #7: single-core matmul (int32/fp32 c->Cc) rides the doorbell */
            double _ta0=ork_now_us(); orki_mc_sub[0]+=_ta0-_ts0;
            if(dt==DT_F16){ float  *cc=c->Cc.cpu,*cr=c->cres; for(int r=0;r<mc;r++)for(int n=0;n<Nc;n++) cr[(size_t)(m0+r)*N+(n0+n)]+=cc[(size_t)r*Nc+n]; }
            else { int32_t*cc=c->Cc.cpu,*cr=c->cres; for(int r=0;r<mc;r++)for(int n=0;n<Nc;n++) cr[(size_t)(m0+r)*N+(n0+n)]+=cc[(size_t)r*Nc+n]; }
            orki_mc_acc[0]+=ork_now_us()-_ta0; orki_mc_n[0]++;
        }
      }
    }
    memcpy(C,c->cres,need); return 0;
}
int ork_mm_run   (ork_npu *c,ork_w *w,int M,const f16    *A,float   *C){
    if(w && w->is_orkd){   /* Path B: fp16 run on the daemon — ring transport if attached (any precision), else socket */
        orkd_set_op_domain(c->daemon, (uint32_t)w->domain);   /* v2: carry this weight's domain with the op */
        if(c && c->daemon && orkd_has_ring(c->daemon)){ int r=orkd_ring_run(c->daemon,w->orkd_id,M,w->K,w->N,ORKD_DT_F16,A,C); if(r!=-2) return r; }
        return orkd_run_f16(c->daemon, w->orkd_id, M, w->K, w->N, A, C); }
    if(w->dtype!=DT_F16)return -1;
    if(orki_check_overlap("ork_mm_run", (uintptr_t)A, (uintptr_t)A + (size_t)M * w->K * 2, (uintptr_t)C, (uintptr_t)C + (size_t)M * w->N * 4)) return -1;
    return orki_run(c,w,M,A,C);
}

/* ---- SLICE-AND-DICE: a matmul packed as c_base doorbell tiles (K-slice + N-tile, ork_slice.h). Pack once,
 * run many. Lets a wide-K / wide-N op run ENTIRELY on the doorbell (one chained submit, K-slice int32-accumulate
 * + N-tile scatter), bit-exact vs the reference matmul — the foundation for the doorbell owning every submit
 * (SLICE_AND_DICE_PLAN.md). PRECISION-GENERAL: the handle carries its dtype and pack/run dispatch to the
 * per-precision doorbell envelope; the decomposer (tile geometry) is dtype-agnostic. ONLY DT_I8 is live today
 * (q8_0 compute path + only precision the multi-core doorbell accepts as tiles); DT_F16/DT_I4 pack returns NULL
 * until their doorbell tile path is built. sub[ki*nnt+ni] holds one c_base-sized packed weight per tile. */

void ork_mm_free_sliced(ork_npu *c, ork_w_sliced *w) {                   /* dtype-agnostic: ork_mm_free frees any sub-weight */
    if (!w) return;
    if (w->sub) { for (int i = 0; i < w->nks * w->nnt; i++) if (w->sub[i]) ork_mm_free(c, w->sub[i]); free(w->sub); }
    free(w);
}

/* int8 sub-weight packer: decompose K/N and pack each c_base tile from B[K,N]. */

/* int4 (W4A4) sub-weight packer (#33): twin of orki_slice_pack_i8, but the tile envelope is BCHAIN's
 * (run_i4_bchain_db, the per-tile executor): each sub-tile must be Sk==1, Sn==1, N%64==0, and K<=8192 so
 * BCHAIN's H=16384/K>=2. So K-slice at ks=8192 (K padded to 32 — pack_i4 needs K%32; pad rows are zeroed ->
 * contribute 0), N-tile at ns=8192 (Sn==1; BCHAIN N-tiles further by bank-width internally). B is the int8
 * nibble-container [-8,7] (pack_i4's input); a native pack_i4 weight keeps no raw nibbles, so sub-tiles are
 * re-packed from the caller's B here (as orki_slice_pack_i8 does with ork_mm_pack_i8). */

/* PRECISION DISPATCH. int8 + int4 have refuse sites and thus a sliced rescue. fp16 has NO refused shapes
 * (out-of-envelope fp16 -> ORK_RC_F16_SC -> single-core reference, a working path; its multicore fit is the
 * CONTIG colsplit, not tiles) -> no fp16 tiled path. */
ork_w_sliced *ork_mm_pack_sliced(ork_npu *c, int K, int N, const void *B, int dtype) {
    if (!c || !B || K <= 0 || N <= 0) return NULL;
    switch (dtype) {
        case DT_I8:  return orki_slice_pack_i8 (c, K, N, (const int8_t *) B);
        case DT_I4:  return orki_slice_pack_i4 (c, K, N, (const int8_t *) B);
        default:     fprintf(stderr, "[ork] slice-and-dice: int8/int4 only (fp16 has no refused shapes — uses colsplit); dtype %d unsupported\n", dtype);
                     return NULL;
    }
}

/* #33 STAGE 3: per-core PARALLEL ks-outer NEON accumulate for the sliced doorbell run — the proven
 * colsplit pattern (ork_csub_worker's WIDE-K PARALLEL ACCUMULATE). Each pool core owns a balanced,
 * DISJOINT output-column range [c0,c1); for every N-tile overlapping it, it sums that tile's nks K-slice
 * partials into C (ks-outer, so the C column block stays hot across the K-slice passes; NEON int32 add is
 * associative -> bit-exact). Global-column split parallelizes BOTH wide-N (tiles fan across cores) and
 * wide-K (a single tile's columns split across cores). Replaces the single-threaded ni/ki/m/n sum that
 * measured 1.15-2.31x slower than native (worst for wide-N). */

void *orki_slice_acc_worker(void *p){
    struct slc_acc *a = p; int nnt=a->nnt, nks=a->nks, ns=a->ns, N=a->N, M=a->M, c0=a->c0, c1=a->c1;
    for(int ni=0; ni<nnt; ni++){ int n0=ni*ns, Nw=(N-n0<ns)?(N-n0):ns;
        int lo=(n0>c0)?n0:c0, hi=((n0+Nw)<c1)?(n0+Nw):c1; if(lo>=hi) continue;   /* this tile's overlap with the core's columns */
        int woff=lo-n0, wlen=hi-lo;
        for(int ki=0; ki<nks; ki++){ const int32_t *src=(const int32_t*)a->tasks[(size_t)ki*nnt+ni].C;
            for(int m=0; m<M; m++){ int32_t *cr=a->C+(size_t)m*N+lo; const int32_t *pr=src+(size_t)m*Nw+woff; int n=0;
                if(ki==0){ for(; n<wlen; n++) cr[n]=pr[n]; }                       /* first K-slice seeds */
                else { for(; n+4<=wlen; n+=4) vst1q_s32(cr+n, vaddq_s32(vld1q_s32(cr+n), vld1q_s32(pr+n)));
                       for(; n<wlen; n++) cr[n]+=pr[n]; } } } }                    /* rest accumulate (NEON) */
    return NULL;
}

/* int4 sliced orki_run (#33): decompose a refused int4 shape into BCHAIN-legal sub-tiles (Sk==1, Sn==1, N%64,
 * K<=8192) and run EACH via orki_run_i4_bchain_db (M>=2) or the per-row doorbell (M==1) — reusing #52's
 * self-healing / pool / de-tile machinery as a black box — then int32-accumulate the K-slices + scatter N
 * (int4 C is int32 after BCHAIN's de-tile, so the int8 orki_slice_acc_worker applies verbatim). Tiles run
 * sequentially (each internally multi-core); a tile that refuses/fails fails the whole rescue -> the caller
 * refuses (never a blocking fall-back, per #45/#52). */

/* PRECISION DISPATCH. int8 + int4 (fp16 has no refused shapes -> uses colsplit, not the tiled surface). */
int ork_mm_run_sliced(ork_npu *c, ork_w_sliced *w, int M, const void *A, void *C, int nc) {
    if (!w) return -1;
    switch (w->dtype) {
        case DT_I8:  return orki_slice_run_i8 (c, w, M, (const int8_t *) A, (int32_t *) C, nc);
        case DT_I4:  return orki_slice_run_i4 (c, w, M, (const int8_t *) A, (int32_t *) C, nc);
        default:     return -3;   /* fp16: colsplit (not tiles) */
    }
}

/* ---- FUSED FFN: matmul with an on-NPU output-stage activation (SwiGLU fusion) --------------------
 * These run a RESIDENT-weight int8 matmul and apply the activation / element-wise-multiply IN the
 * matmul's SDP output stage — so the activation rides the matmul's own submit (no extra submit, no
 * mode-switch re-warm, no host round-trip). This is the ONLY regime where an on-NPU activation beats
 * inline NEON (standalone SDP ops lose ~8x to interleaving; see the RE-roadmap M4.6).
 * The SDP OUT_CVT scale R (r_mult/2^r_shift) + out_bias are SCALAR (per-tensor), so the caller must
 * quantize the activation PER-TENSOR (one scale for the whole A tile), NOT per-row. Full-K resident
 * weight (w->Bf, K%512==0, K<=4096), N-tiled, single-core, M-tile<=64/submit. The fixed silu*S PWL LUT
 * is streamed into PPU SRAM ONCE (submit-1) then every matmul+silu submit reads it. rk3588-gated.
 * out C = silu(requant(A·W)) as int8 [M*N]. 0/ok, -1 wedge, -2 shape, -3 SoC. */

/* Fused-path M-tile cap = the 0x1040 K-reduction schedule's validated max rows (mg_max*64), the SAME
 * bit-exact ceiling the plain int8 matmul uses (synth_i8 sets 0x1040 from mc, so the fused output stage
 * inherits the exact schedule). Was hardcoded 64 (conservative) — that DOUBLED submits at prefill
 * (mc=128@K2048). K<=4096 here so Kp==K. See AGENTS.md "weight-DMA amortization". */
int orki_fused_mtile(int K,int M){
    double scale=(double)K/512.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale);
    int mg_max = base>=0x1b ? (base-0x1b)/slope+1 : 0; int chunk = mg_max*64;
    const char*e=getenv("ORK_FUSED_MTILE"); if(e){ int v=atoi(e); if(v>0)chunk=v; }  /* A/B override (validation) */
    if(chunk<1)chunk=1; if(chunk>M)chunk=M; return chunk;
}
/* fp16 twin of fused_mtile: the fp16 0x1040 K-reduction schedule (orki_synth() uses scale=K/256, vs int8's K/512
 * since int8 packs 2 rows per CBUF slot) gives the SAME bit-exact M-tile ceiling mg_max*64. The old
 * ork_mm_run_f16_silu chunk=16 was a stale over-conservative cap far below this (64 @K2048, 320 @K512) —
 * bit-exact validated (tools/silu_f16_check: M-tile 16==32==64 @K2048, 16==320 @K512, 384>ceil DIFFERS).
 * ORK_F16_MTILE overrides (validation / probing above the ceiling). */

/* ⚠ CLOSED — the matmul-fused activation output is INT8-ONLY (2026-07-05 sweep). The output-precision is a
 * PREC field in reg 0x4010 (int8=bits00, int16=01, fp16=10; bit31=int32 which BYPASSES the CVT the LUT needs).
 * SWEPT PREC=1 (int16) across 3 output-stride configs (0x40c0/0x4050/0x4038) — ALL soft-reset the NPU + garbage.
 * int32 (0x800044e0) also wedges (CVT bypass vs LUT conflict). So the MATMUL+LUT program only supports int8
 * output; int16/fp16 output exists only in the STANDALONE silu program (0x50xx lane: int16=0x24004401,
 * fp16=0xa8000002 — different regcfg). No RKNN capture possible (RKNN never fuses activation into non-int8
 * matmul). CONCLUSION: higher-precision fused silu is not achievable in the matmul program. The ablation
 * (ORK_GATE_ABLATE) proved int8 silu OUTPUT is the whole FFN-chain PPL gap, so the remaining route to parity
 * is UN-FUSED: int32 matmul -> standalone int16/fp16 silu op (loses the silu-free-on-NPU fusion, adds a submit).
 * Below is the sweep harness (env-configurable format regs), kept for the record; NOT called by the chain. ── */
/* orki_set_i8_silu32 — fused SiLU output stage with INT32 output (silu value NOT quantized to int8). Keeps
 * synth_i8's default int32 output format (does NOT apply set_i8_out8's int8 override) and enables the SiLU
 * LUT with the int32-output bit (0x8000) set in 0x4010. out_i32 = R*V16[idx(acc)] + out_bias, unclamped —
 * with a fine-scale LUT that maps silu across ~±8000 (int16 V16 * R), that's ~13-14 bit silu instead of int8.
 * The ablation (ORK_GATE_ABLATE) showed the int8 silu OUTPUT is the ENTIRE FFN-chain quality gap (fp32 silu
 * = baseline PPL); this recovers it while keeping silu free on-NPU. Same LUT/config regs as set_i8_silu. */

/* ork_mm_run_i8_silu32 — resident full-K int8 matmul + fused SiLU with INT32 output (C is int32 [M*N]).
 * Same path as ork_mm_run_i8_silu but the silu value is emitted at int32 precision (dequant with the fine
 * out_scale the LUT was built for). K%512, K<=4096, N%32. 0/ok, -1 wedge, -2 shape, -3 SoC. */

/* set_f16_silu — graft the SiLU LUT output stage onto the fp16 matmul (REGCMD) program, KEEPING its native
 * fp16 output CVT (0x4010=0xa8000002, 0x40c0=0x40, 0x4050=0x36e, 0x4084 gain — all from the REGCMD template)
 * so the silu value is emitted at fp16→fp32 precision, NOT quantized to int8. Same flying-mode LUT-stage regs
 * as orki_set_i8_silu (the activation sub-module is shared; only the output precision differs — kept fp16 here, vs
 * set_i8_silu's set_i8_out8 override to int8). This is the "end-goal" higher-precision fused gate — currently
 * a measured net-loss (fp16 matmul ~3.3x int8, tools/f16_gate_bench) so gated OFF, kept for a future int8-win
 * pipeline. WIP: the acc->index map / LUT calibration for the fp16 gain is approximate. */

/* ork_mm_run_f16_silu — fp16 gate matmul + fused SiLU with fp16→fp32 output (no int8 activation quant). The
 * "end-goal" precise on-NPU gate: recovers the full PPL gap the int8 silu output loses (ablation), at the cost
 * of the fp16 matmul (~3.3x int8, tools/f16_gate_bench) — a measured net-loss TODAY, so gated OFF, built out
 * for a future pipeline where it pays off. w = fp16 weight (ork_mm_pack), A = fp16 [M,K], C = fp32 [M,N] silu.
 * K%32, N<=nmax. fp16 M-tile = orki_f16_mtile(K) = the 0x1040 schedule's bit-exact ceiling mg_max*64 (was a stale
 * chunk=16; the real "latent bug" is only ABOVE that ceiling — bit-exact validated, see f16_mtile / silu_f16_check).
 * 0/ok, -1 wedge, -2 shape, -3 SoC. STATUS (2026-07-05): RUNS on-NPU (no wedge) AND now CALIBRATED accurate —
 * tools/silu_f16_calib cracked it to mean|err|~0.08 / max 0.75 over silu[-8,8] (~1%, on par with int8 silu).
 * CALIBRATION RECIPE (the fp16 program's index only spreads for NEGATIVE acc, so): (1) NEGATE the gate — pack
 * the gate weight *(-S)* so a positive real gate becomes a negative accumulator that spreads across the LUT;
 * negatives then clamp to ~0 (== silu(neg)). (2) scale S~24 (spreads [-8,8] over the ~513-entry negative half;
 * larger S over-spreads/clamps). (3) idx_off=0xffffc000, cfg4068=0x56391100, DEFAULT 0x4064 (0xffff7dc8
 * collapses it). (4) measure idx(acc) via a ramp LUT, build curve LUT[idx]=silu(-acc/S)/(R*out_scale). R
 * (output gain) = 0x4084; 0x4044/za has NO index effect. TODO to enable: a build_f16_silu_lut() baking (1)-(4)
 * + the negate/scale into the fp16 gate weight pack, then wire the gate. Kept gated OFF (fp16 mm ~3.3x int8). */


/* Resident-weight int8 matmul with int8-REQUANTIZED output (no activation): C = clamp_i8(round((A·W)*
 * mult/2^shift)) as int8 [M*N]. For keeping int8 intermediates on the NPU across a chained FFN inner
 * (the "up" projection feeding the EW-mul). Same resident full-K path + submit1 as the fused runs. */

/* int8 matmul with INT16 requant output (set_i16_out): C = clamp_i16(round((A·W)*mult/2^shift)) [M*N] int16,
 * COMPACT-LINEAR (m*N+n) — which is exactly the CONTIGUOUS host layout the int16 SDP seq adapters read, so a
 * seq [MM_I8_OUT16 -> exp_i16/silu_i16] carries int16 forward RESIDENT (A2) with NO hardware PC-chain and NO
 * layout bridge (sidesteps the set_i16_out chaining fragility). K%512 (int8 requant small-K limit). Twin of
 * ork_mm_run_i8_out8. 0/ok, -1 wedged, -2 dims, -3 SoC. */

/* RE/calibration: run ONE M=1 full-K int8 submit (no K-split) at (K,N) to probe this SoC's
 * single-submit K-tile ceiling (`0x1044`). Allocates its own buffers — does not touch resident
 * weights. Returns 0 if the submit completed (C[N] int32 valid), -1 if it wedged (K over the
 * per-op K-tile cap; recoverable — the next call's RKNPU_ACT_RESET clears it), -2 on bad dims. */
/* RE/calibration: run ONE full-K int8 submit at (M,K,N) in either ork's current synth_i8 M-tile
 * mode (mode=0) or the rkllm-captured M-tile mode (mode=1), and return the full C[M*N] int32 plus
 * the warm-submit time (us). rkllm-mode mirrors fused2.log: 0x1010=0x20 const (NOT 16*min(M+1,R)),
 * 0x1044=(K/64)*M, 0x107c=4*M, 0x1040=0xb1-0xf*(ceil(M/8)-1). Lets us test whether rkllm's larger
 * M-per-submit (M up to 36) is bit-exact on ork and whether it lifts effective TOPS. A[M*K] B[K*N]
 * row-major int8; C[M*N] int32. Returns 0/ok (C valid), -1 wedged, -2 bad dims. ISOLATED buffers. */


/* RE/validation for the PPU fused-output stage (step 1): run ONE full-K int8 matmul at (M,K,N) with
 * the int8-REQUANTIZED output stage (set_i8_out8) instead of int32, and return C[M*N] as int8. This
 * is the isolated bit-exact test bed for the fused path — the caller compares against the CPU model
 * out_i8 = clamp_i8((acc_i32 * mult) >> shift). Isolated buffers, does not touch resident weights.
 * A[M*K] B[K*N] row-major int8; C[M*N] int8 out. mult/shift = fixed-point requant (identity =
 * 0x4000,14). 0/ok (C valid), -1 wedged, -2 bad dims. See ork_ppu_fuse_enabled + set_i8_out8. */

/* PHASE 1 (#35 chained FFN) — RE crux: run one int8 matmul with the INT16-requantized output stage
 * (set_i16_out) and return C[M*N] as int16. Clone of probe_i8_out8 (weight [NT][KT][32][32], A->Af,
 * one submit) with a 2-byte output surface. Isolates whether the int16 output encoding produces the
 * correct VALUES (out_i16 = clamp_i16(round(acc_i32 * mult / 2^shift))); the caller compares vs the
 * CPU model. C is written as raw M*N*2 bytes (layout TBD — a varying-value pass follows to map it).
 * 0/ok, -1 wedged, -2 dims. */

/* RE (int8 batch-mode A/B): run one int8 matmul via synth_i8 and return the RAW int32 output (no
 * requantize, no de-tile). Mirrors probe_i8_out8 (weight [NT][KT][32][32], A raw-copied to Af). Any
 * ork_i8_fuzz_add overrides apply inside synth_i8, so a caller can flip int8 stream->batch (0x405c=0 etc.)
 * and observe the resulting output layout. Output buffer is 2*M*N int32 (room for a stride-2 batch layout).
 * 0/ok, -1 wedged, -2 dims. Submit timeout honors ORK_I4_PROBE_TO_MS (fast-fail for wedgy fuzz values). */

/* DOORBELL PIPELINE PROFILER (Tier 11): measure the wall of `iters` SERIAL int8 matmuls, BLOCKING vs
 * NONBLOCK + DRAM-doorbell busy-poll. Blocking = submit waits (~130µs floor + compute)/op. Nonblock = submit
 * returns ~5µs, CPU busy-polls the output SENTINEL (DC CIVAC invalidate + read) until the NPU overwrites it =
 * op done, no sleep/wake. Same op (all-ones int8 -> every output = K) both ways; validates output == K. Calls
 * the raw ioctl directly with explicit flags (bypasses the env/sleep wrapper). 0/ok, <0 err. */
int ork_npu_doorbell_prof(ork_npu *c,int M,int K,int N,int iters,double *block_us,double *nb_us,int *ok_block,int *ok_nb){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    memset(W.cpu,1,(size_t)K*N);                                  /* int8 weight all-1 (layout-agnostic) */
    orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)M*N*4,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}   /* int32 out */
    int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=1; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);  /* act all-1 -> out=K */
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I8_N]; orki_synth_i8(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task*t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0xd; t->int_mask=0x300; t->int_clear=0x1ffff;
    t->regcfg_amount=108; t->regcmd_addr=(uint32_t)c->regcmd.dma; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub; memset(&sub,0,sizeof sub); sub.task_number=1; sub.task_obj_addr=c->task.obj;
    sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1; sub.timeout=4000; sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    volatile int32_t *dbell=(volatile int32_t*)((char*)O.cpu + (size_t)(M*N-1)*4);  /* last output int32 = doorbell */
    const int32_t SENT=0x7ffffff;                                  /* matmul (all-1, K) can't produce this */
    volatile int32_t *o0=(volatile int32_t*)O.cpu, *ol=(volatile int32_t*)dbell;  /* check endpoints */
    /* ---- BLOCKING: flags=0x5 via the proper submit path (domain/bookkeeping) ---- */
    *o0=SENT; *ol=SENT; __asm__ volatile("dc cvac,%0"::"r"(o0):"memory"); __asm__ volatile("dc cvac,%0"::"r"(ol):"memory"); __asm__ volatile("dsb ish":::"memory"); /* seed endpoints (like the doorbell) so CIVAC can read fresh */
    sub.flags=0x5; orki_rknpu_submit_ioctl(fd,&sub,-1); orki_rknpu_submit_ioctl(fd,&sub,-1);  /* warm (mode + first-cold) */
    double t0=ork_now_us();
    for(int i=0;i<iters;i++){ sub.flags=0x5; if(orki_rknpu_submit_ioctl(fd,&sub,-1)){*ok_block=0;} }
    if(block_us)*block_us=(ork_now_us()-t0)/iters;
    { for(long s=0;s<2000000L && (*o0==SENT||*ol==SENT);s++){ __asm__ volatile("dc civac,%0"::"r"(o0):"memory"); __asm__ volatile("dc civac,%0"::"r"(ol):"memory"); __asm__ volatile("dsb ish":::"memory"); }
      *ok_block=(*o0==K && *ol==K);
      if(getenv("ORK_DBELL_DBG"))fprintf(stderr,"[dbg-block] o[0]=%d o[last]=%d K=%d\n",*o0,*ol,K); }
    /* ---- NONBLOCK + doorbell busy-poll: flags=0x7 (adds NONBLOCK 0x2), submit returns ~5µs, spin on the sentinel ---- */
    t0=ork_now_us(); int polled_ok=1;
    for(int i=0;i<iters;i++){
        *dbell=SENT; __asm__ volatile("dc cvac, %0"::"r"(dbell):"memory"); __asm__ volatile("dsb ish":::"memory"); /* seed sentinel in DRAM */
        sub.flags=0x7;                                             /* PC|PINGPONG|NONBLOCK */
        if(orki_rknpu_submit_ioctl(fd,&sub,-1)){ polled_ok=0; break; }
        long s=0; for(;s<20000000L;s++){ __asm__ volatile("dc civac, %0"::"r"(dbell):"memory"); __asm__ volatile("dsb ish":::"memory");
            if(*dbell!=SENT) break; }                              /* busy-poll: NPU overwrote the doorbell = done */
        if(s>=20000000L){ polled_ok=0; break; }                   /* poll timed out */
    }
    if(nb_us)*nb_us=(ork_now_us()-t0)/iters;
    orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE);
    { int32_t*o=O.cpu; *ok_nb=(polled_ok && o[0]==K && o[M*N-1]==K);
      if(getenv("ORK_DBELL_DBG"))fprintf(stderr,"[dbg-nb] o[0]=%d o[last]=%d K=%d polled_ok=%d\n",o[0],o[M*N-1],K,polled_ok); }
    orki_bdestroy(fd,&W); orki_bdestroy(fd,&O);
    return 0;
}

/* overlap_prof — Tier 11: does REAL CPU work in the shadow of an async NPU op stay FREE (overlap wall ~= max),
 * or does shared LPDDR4X bandwidth contention stretch it (wall -> npu+cpu)? This is the "zero-time router"
 * thesis (speculative/batched MoE) in one number. cpu_reps = # of 512x512 fp32 GEMVs run on the CPU between
 * the NONBLOCK submit and the doorbell poll (a stand-in for the routing math; 1MB matrix -> real DRAM traffic).
 * Fills npu_solo (submit+poll, no CPU work), cpu_solo (the GEMVs alone), overlap_wall (submit+CPU-work+poll). */
int ork_npu_overlap_prof(ork_npu *c,int M,int K,int N,int cpu_reps,int iters,
                         double *npu_solo,double *cpu_solo,double *overlap_wall,int *ok){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    memset(W.cpu,1,(size_t)K*N); orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)M*N*4,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}
    int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=1; orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I8_N]; orki_synth_i8(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task*t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0xd; t->int_mask=0x300; t->int_clear=0x1ffff;
    t->regcfg_amount=108; t->regcmd_addr=(uint32_t)c->regcmd.dma; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub; memset(&sub,0,sizeof sub); sub.task_number=1; sub.task_obj_addr=c->task.obj;
    sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1; sub.timeout=4000; sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    volatile int32_t *dbell=(volatile int32_t*)((char*)O.cpu + (size_t)(M*N-1)*4);
    const int32_t SENT=0x7ffffff;
    const int RN=512; float*Wc=malloc((size_t)RN*RN*4),*xc=malloc(RN*4),*yc=malloc(RN*4);  /* CPU "router" state */
    if(!Wc||!xc||!yc){free(Wc);free(xc);free(yc);orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);return -2;}
    for(int i=0;i<RN*RN;i++)Wc[i]=(float)(((unsigned)i*2654435761u)>>28)*0.01f; for(int i=0;i<RN;i++)xc[i]=1.0f;
#define CPU_ROUTER() do{ for(int r=0;r<cpu_reps;r++){ for(int a=0;a<RN;a++){ float acc=0; const float*wr=Wc+(size_t)a*RN; \
        for(int b=0;b<RN;b++)acc+=wr[b]*xc[b]; yc[a]=acc; } xc[0]=yc[RN-1]*1e-9f; } }while(0)  /* cross-rep dep -> no DCE */
    sub.flags=0x5; orki_rknpu_submit_ioctl(fd,&sub,-1); orki_rknpu_submit_ioctl(fd,&sub,-1);  /* warm */
    int okk=1;
    /* (1) NPU solo: nonblock submit + doorbell poll, NO cpu work */
    double t0=ork_now_us();
    for(int i=0;i<iters;i++){ *dbell=SENT; __asm__ volatile("dc cvac,%0"::"r"(dbell):"memory"); __asm__ volatile("dsb ish":::"memory");
        sub.flags=0x7; if(orki_rknpu_submit_ioctl(fd,&sub,-1)){okk=0;break;}
        long s=0; for(;s<20000000L && *dbell==SENT;s++){ __asm__ volatile("dc civac,%0"::"r"(dbell):"memory"); __asm__ volatile("dsb ish":::"memory"); }
        if(s>=20000000L){okk=0;break;} }
    if(npu_solo)*npu_solo=(ork_now_us()-t0)/iters;
    /* (2) CPU solo: the router GEMVs alone, no NPU */
    t0=ork_now_us(); for(int i=0;i<iters;i++){ CPU_ROUTER(); } if(cpu_solo)*cpu_solo=(ork_now_us()-t0)/iters;
    volatile float sink=xc[0]; (void)sink;
    /* (3) OVERLAP: nonblock submit, run the router in the shadow, THEN poll for NPU completion */
    t0=ork_now_us();
    for(int i=0;i<iters;i++){ *dbell=SENT; __asm__ volatile("dc cvac,%0"::"r"(dbell):"memory"); __asm__ volatile("dsb ish":::"memory");
        sub.flags=0x7; if(orki_rknpu_submit_ioctl(fd,&sub,-1)){okk=0;break;}
        CPU_ROUTER();                                              /* CPU works while the NPU crunches */
        long s=0; for(;s<20000000L && *dbell==SENT;s++){ __asm__ volatile("dc civac,%0"::"r"(dbell):"memory"); __asm__ volatile("dsb ish":::"memory"); }
        if(s>=20000000L){okk=0;break;} }
    if(overlap_wall)*overlap_wall=(ork_now_us()-t0)/iters;
    *ok=okk;
    free(Wc);free(xc);free(yc); orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);
#undef CPU_ROUTER
    return 0;
}

/* RE (fp16 batch-mode mapping): raw fp32 output of one fp16 matmul via orki_synth(). Weight tile [NT][KT][16][32]
 * (N-tile=16), A raw-copied [M][K] fp16, output fp32 (2*M*N floats, room for a batch layout). A/B are fp16
 * bit patterns (uint16). ork_f16_fuzz overrides apply inside orki_synth(). 0/ok -1 wedged -2 dims. */

/* RE: STANDALONE fp16 matmul with FP16 output (set_f16_out, 0x4010=0x48000002 fp16-in) in the EWCUBEH atom-8
 * layout the chained SDP consumes — isolates "can the fp16 matmul emit fp16 G correctly" from the chain handoff.
 * A[M*K],B[K*N] fp16 bit patterns; out[M*N] fp16 read via EWCUBEH. task_number=1. 0/ok,-1 wedged,-2 dims. */

/* A1 (task #20): fp16 matmul with CONTIGUOUS fp16 output, consuming a PACKED resident ork_w (w->Bb) via the
 * PROVEN vendor fp16-out stage (orki_set_f16_out_fp16in, default contiguous). The point: a matmul's output is fp16
 * [M,N] directly — so it feeds an fp16 SDP op with NO f32->f16 host narrow between them, making the pair
 * ADJACENT (A2 can then keep the intermediate resident). Single-tile (M<=64, single-slice), single-core; it
 * runs its OWN submit (not the f32-out doorbell), so its SEQ_CLASS row is hw=0 (per-op SW dispatch). This is
 * the twin of ork_npu_probe_f16_mm_f16out but weight = w->Bb (resident) instead of a raw-B rebuild.
 * 0/ok, -1 submit-fail, -2 dims. */

/* ZERO-COPY STRIDED activation fp16 matmul — the densify-drop primitive for attention. A is logical [M][K] but
 * stored at row pitch `apitch` (apitch>=K, apitch%8==0) in a DEVICE (DMA) buffer; the NPU reads it DIRECTLY via
 * CNA LINE_STRIDE (0x107c=apitch/8), with NO host->Af gather. This is what a permuted-Q / KV-cache-view feeds:
 * the fork stages A once in a DMA buffer (KV-cache) and the matmul reads the strided view in place — no CPU
 * densify. Here the DMA buffer is allocated + filled internally to VALIDATE the zero-copy strided read against a
 * contiguous reference (out = A[:, :K]·B, contiguous fp16). A[M*apitch],B[K*N] fp16 patterns. 0/ok,-1,-2. */

/* SINGLE-SUBMIT fp16 matmul with per-channel scale FUSED into the output stage (BS-fold family): one task,
 * no separate SDP, no matmul-out<->SDP-in layout bridge. The EW-mul lane (0x50xx, spliced via orki_splice_ew_lane)
 * multiplies the ON-CHIP accumulator by a per-CHANNEL operand scale[N] (ERDMA per-channel 0x5034=0x08, operand
 * CONTIGUOUS [N]) before the fp16 writeout, so the output keeps the matmul's native CONTIGUOUS [M][N] layout
 * (proven 512/512). out[m][n] = (Σ_k A[m][k]B[k][n]) * scale[n]. A/B/scale/out fp16 bit patterns. Env-tunable
 * EW regs (ORK_F16EW_*) for on-board RE. 0/ok,-1 wedged,-2 dims,-3 SoC. */

/* RE/validation for the FUSED EW-mul output stage (step 3, SwiGLU dual-input): run a full-K int8 matmul
 * whose output stage int8-requantizes the accumulator AND multiplies it by a SECOND input G (= silu(gate)),
 * returning C[M*N] int8. This splices the 0x50xx second-DPU lane into the regcmd (synth_i8_ew) and submits
 * with regcfg_amount=126 (108 matmul + 18 second-lane). A[M*K] B[K*N] G[M*N] row-major int8.
 * FIRST-RUN CONTRACT: the 0x50xx dims/strides are the CAPTURED values (M=8,N=32) — validate at that shape
 * first; the exact multiply/scale semantics and the shape-dependent 0x50xx fields are resolved by board
 * matched-diff (see EWMUL_WIP.md). 0/ok (path executed, C valid), -1 wedged, -2 bad dims. */

/* PATH (b) bring-up: submit RKNN's captured EW-mul op (REGCMD_EWMUL) VERBATIM with ork's buffers, only
 * repointing the 6 buffer addresses. Its geometry is RKNN's own (unlike the synth_i8 overlay the HW
 * rejected), so this tests whether the templatized op EXECUTES on ork's submit path. Buffers mirror the
 * captured handle layout (input/weight/silu/output); contents are caller-provided (execution does not
 * depend on them). Returns 0/ok, -1 wedged. On ok, out[] gets the raw output buffer (Osz bytes).
 * in[Isz] wt[Wsz] gl[Gsz] are copied into the input/weight/silu buffers at their captured offsets. */

/* Path (b) MATMUL replay: submit the captured matmul-shaped EW-mul op (REGCMD_EWMUL_LIN, K=512/N=64/M=8)
 * with ork's buffers + ork's standard tile packing (B as [Nt][Kt][32][32], A linear). out = requant(A*B) ⊙ G,
 * G = silu(gate) int8 2nd-input. Fixed shape (the captured one) for now — used to read the multiply semantics
 * and validate vs CPU; generalization to arbitrary M/K/N is the next step. A[M*K] B[K*N] G[M*N] int8; C[M*N].
 * ORK_EW_S20=v writes uint32 v into the 0x5020 param region (2nd-input scale probe). 0/ok,-1 wedged,-2 dims. */

/* Standalone SDP element-wise MULTIPLY: submit REGCMD_MUL (captured pure-Mul op, no conv) with a,b,out patched.
 * out[i] = clamp_i8(round(a[i]*b[i]*gain) + bias), gain/bias baked from the captured op (0x4084/88/80).
 * a,b,out int8[n] (n<=4096). Reads a via SRDMA(0x5018), b via ERDMA(0x5038), writes out(0x4020). enable=0x18,
 * regcfg=69. This is the CLEAN on-NPU element-wise path (NVDLA standalone SDP layer, both operands from memory)
 * — sidesteps the conv-geometry coupling that blocked the fused-into-matmul approach. 0/ok,-1 wedged. */

/* Public EW-mul: out[m*N+n] = clamp_i8(round(up[m][n]*silu[m][n] * mult/2^shift)) computed ON THE NPU via the
 * standalone SDP element-wise op. Marshals up/silu (logical [M][N]) into the NVDLA feature cube (atom-16),
 * submits REGCMD_MUL with symmetric zero-points (za=zb=zo=0), de-marshals. GENERALIZED to arbitrary M,N via
 * orki_set_mul_geom (M,N reprogrammed from the captured M=8/N=64 op). N must be a multiple of 16 (channel atom).
 * mult must be 0..0x7fff (OUT_CVT_SCALE is SIGNED 16-bit). 0/ok, -1 wedged, -2 bad shape, -3 non-rk3588. */

/* fp16 element-wise MULTIPLY: out[m][n] = up[m][n] * silu[m][n] in fp16 on the NPU (standalone SDP fp16 op,
 * no requant). Marshals into the NVDLA fp16 feature cube (atom=8, 2-byte channels) internally. GENERALIZED to
 * arbitrary M,N (N a multiple of 8); rk3588-gated. up/silu/out are fp16 bit-patterns (ork_f16).
 * 0/ok,-1 wedged,-2 shape,-3 SoC. */

/* NEOX RoPE on the NPU (Qwen3 rope type 2). x[nrow][hd] fp16 row-major — nrow = (heads*tokens) flattened, each
 * row is one (head,token)'s head_dim vector; pos[nrow] = that row's token position. NEOX rotates pairs
 * (i, i+hd/2): out[i]=x[i]cosθ - x[i+hd/2]sinθ, out[i+hd/2]=x[i]sinθ + x[i+hd/2]cosθ, θ_i=pos*freq_base^(-2i/hd).
 * Composed on-NPU as x⊙COS + rot_half(x)⊙SIN — 2 ewmul + 1 add (COS/SIN tables + the half-swap built on CPU;
 * the transcendental-free EW math runs on the NPU). hd even, hd multiple of 8. 0/ok, <0 on primitive failure. */

/* int16 element-wise MULTIPLY: out[m][n] = clamp_i16(round(up[m][n]*silu[m][n] * mult/2^shift)) on the NPU
 * (standalone SDP int16 op). The w4a4 path's EW precision (ork's int4 matmul outputs int16). 2-byte operands,
 * NVDLA cube atom=8 (same layout as fp16). Symmetric zero-points. mult in 0..0x7fff. GENERALIZED to arbitrary
 * M,N (N a multiple of 8); rk3588-gated. 0/ok,-1 wedged,-2 shape,-3 SoC. */

/* Standalone int8 element-wise ADD — RE probe (settable scale regs to decode the add structure). out[m][n] from
 * the 2-input SDP op with ALU=add (REGCMD_ADD), reprogrammed to (M,N) by orki_set_mul_geom. Caller sets za(0x4044),
 * zb(0x4074), zo(0x4080), out scale mult(0x4084)/shift(0x4088), and the b-operand scale bscale(0x4078); the
 * ALU-mode regs (0x4040/0x4048/0x4070) stay from the template. a/b/out int8 [M*N], N%16==0. 0/ok,-1,-2,-3. */

/* On-NPU per-row MAX-REDUCE (int8): out[m] = max_n a[m*N+n], N%16, N<=8192. Batched pairwise-max TREE
 * using the SDP EW ALU in MAX mode (EW_ALU_ALGO=0 @ reg 0x4070, rocket_registers.h; NVDLA map_alu_op
 * MAX=0). Each level maxes channels [0,h) vs [h,2h) across ALL M rows in ONE submit: channel h sits at
 * cube offset (h/16)*M*16, so operand b = a + (h/16)*M*16 (valid since h stays %16 down the tree). Reduces
 * N->16 on-NPU (log2(N/16) submits, ping-pong buffers), then a 16-wide CPU tail. Reusable: softmax
 * stability, max-pool, top-k, clamp. 0/ok, <0 err. */

/* PUBLIC per-channel-scaled fp16 matmul, on NPU: out[m][n] = (Σ_k A[m][k]B[k][n]) * scale[n]. Composes the two
 * bit-exact primitives — the fp16 matmul with fp16 CONTIGUOUS output (ork_npu_probe_f16_mm_f16out, 512/512) and
 * the atom-8 per-channel EW-mul SDP (ork_npu_mul_perchan_f16, which takes a CONTIGUOUS input and repacks to
 * atom-8 internally). This is the vendor's own structure (plain fp16 matmul → separate fp16 per-channel SDP);
 * the contiguous↔atom-8 reshape is the SDP's internal O(M·N) repack (on CPU; the vendor does it as small on-NPU
 * CNA copies — a follow-on optimization, see ATTN_REDERIVE_WIP.md). A/B/scale/out fp16 bit patterns. 0/ok, <0. */

/* PURE-NPU per-channel-scaled fp16 matmul via a DIAGONAL second matmul (no SDP, no reshape, no CPU repack).
 * out = (A·B) · diag(scale): matmul1 A[M,K]·B[K,N] -> G[M,N] (contiguous fp16), then matmul2 G[M,N]·D[N,N] where
 * D=diag(scale) -> out[m][n]=Σ_k G[m][k]·D[k][n]=G[m][n]·scale[n]. The 2nd matmul reads G CONTIGUOUS as its
 * activation (synth's native activation layout) — the CNA reads it directly (the vendor-reshape discovery:
 * the CNA has a real feature LINE_STRIDE, unlike the DPU-RDMA). Cost: the diagonal adds O(M·N²) MACs (N×N
 * weight, mostly zero) — pure-NPU at the price of compute. A/B/scale/out fp16 bit patterns. 0/ok,<0. */

/* On-NPU PER-CHANNEL scale (fp16): out[m][n] = a[m][n] * b[n]. b[N] per-channel vector broadcast across all
 * M rows via ERDMA_DATA_MODE=0 (0x5034=0x08 = per-channel + DATA_SIZE TWO_BYTE for fp16). fp16 EW MUL,
 * quant-free (no gain/zero-points). N%8, N<=8192. b laid CONTIGUOUS [N] (fp16). The transposed-softmax
 * normalize (b=1/Σ per-query). 0/ok, <0 err. */

/* On-NPU PER-CHANNEL scale (int16): out[m][n] = clamp_i16(a[m][n]*b[n]*mult>>shift). b[N] per-channel
 * broadcast (ERDMA_DATA_MODE=0, 0x5034=0x08 for 2-byte). int16 chain-intermediate variant of the per-channel
 * scale (the M4 attention chain uses int16 between matmul and SDP, like the mm->silu chain). N%8. 0/ok, <0. */
/* ork_npu_mul_perchan_f16_contig — per-channel fp16 MUL that reads a CONTIGUOUS [M][N] fp16 input (the native
 * fp16 matmul output layout), via the vendor task13 config (FLYING_MODE + NOTCH addressing, EW_CFG=0x20800384,
 * ERDMA=0x8000000a). This is the SDP that matches the fp16 matmul's contiguous output — closing the chain / a
 * pure-NPU 2-submit without the CPU atom-8 repack. a=[M][N] contiguous fp16, b=[N] scale, out=[M][N]. Captured
 * at M=8,N=64; notch is verbatim for N=64. ORK_MULC_* env for on-board geometry RE. 0/ok,-1,-2,-3. */


/* Wall-#2 probe: a background thread that, mid-submit, hot-patches ONE descriptor far ahead in the treadmill
 * ring (outside the NPU's prefetch horizon) to a MARKER task, then flushes. If the NPU honors the live patch
 * (re-reads the descriptor from DRAM when it arrives), the marker's side effect appears. */

/* RE (WIP, RESHAPE_WIP.md): FULL-CHAIN REPLAY of the vendor gemm+reshape (task0-10) to validate the reshape
 * IN CONTEXT (task4 is a mid-chain op needing task1-3's pipeline state — standalone it saturates). Loads the
 * captured vendor IOVA image (gemm_mul_image.bin, 77824B, IB=0xfffed000), blanket single-delta rebases every
 * in-image reference (data/weights/chain-0x0010) to our buffer, replays task0..task10 as ONE hardware chain,
 * then hands back the GEMM output (@0xffff0000, contiguous [M][N]) and the RESHAPE output (@0xffff0a00, atom-8)
 * so the caller can verify reshape_out == atom8(gemm_out) IN-PLACE — no weight extraction / CPU ref needed.
 * M8/N64 captured geometry. Env ORK_RESHAPE_IMG=path. 0/ok, <0 err. */

/* RE PROBE (WIP, RESHAPE_WIP.md): submit the vendor fp16 contiguous->atom-8 RESHAPE base op (task4,
 * REGCMD_RESHAPE_F16) with a CONSTRUCTED permutation weight (64 all-1.0 entries = the captured N=64 pattern)
 * + our own contiguous [M][N] fp16 input, standalone (task_number=1, enable=0xd). Reads the output buffer RAW
 * so the caller can compare it to the atom-8 layout PCH16(m,n)=(n/8)*(M*16)+m*16+(n%8)*2. N=64 only (the
 * weight pattern + geometry are the captured M=8/N=64 op). Empirical step: learn what ONE reshape op does
 * (task4 is one M-tile of the vendor's M-tiled reshape) before chaining. 0/ok, <0 err. */

/* ── LOOPBACK requantizer (RE HARNESS — the loopback is NOT VIABLE; kept for the finding). Standalone SDP
 * reads INT32 (matmul accumulator) from DRAM, per-channel scale, requant -> int16.  The "loopback" idea was
 * to ROUTE AROUND the broken CNA->DPU requant-WDMA (narrow-output matmul does not self-complete) by doing
 * the 32->16 requant in a SEPARATE SDP pass (enable=0x18, DPU-RDMA — a different physical block): Pass-1 =
 * int8 matmul OUT_PRECISION=int32 (self-completes); Pass-2 = THIS op.  RESULT (2026-07-15, RK3588): the SDP
 * main feature RDMA (MRDMA) clamps to a 2-byte read: with the DPU in int32 mode it fetches HALF the bytes
 * (2E vs the 4E consumed), starving the pipeline mid-job -> WDMA terminal-count never reached -> errno=110.
 * Shape-INVARIANT at the default config (all 8 IN_PRECISION enums, M=1..32).
 * ★ OVER-FETCH FIX (2026-07-15, RK3588): the RDMA input dims (0x500c width / 0x5014 channel) are DECOUPLED
 * from the DPU output dims. INFLATING the RDMA element count (ORK_RQ_5014=2N-1) makes it fetch 4E bytes so
 * both terminal counters hit together -> the int32-input SDP SELF-COMPLETES (rc=0, ~30-80us). So int32 CAN be
 * read to completion — the "terminal silicon" wall was a fetch/consume byte-count DISAGREEMENT, fixable in SW.
 * ★ BUT the DPU still processes it as TWO int16 LANES, not a true int32: out[even 2k]=low16(a[k]),
 * out[odd 2k+1]=high16(a[k]) (spread across 2 output channels; out[n]<-a[n/2], odd=0 for a<2^16). So a true
 * int32-VALUE requant still isn't a single-datapath op (the low/high halves land in separate channels the
 * per-channel EW-mul can't recombine) — but for int16-range accumulators the even-channel (low-half) read is
 * usable. Full int32 needs recombining the lanes (open). Env: ORK_RQ_4010/MSTRIDE/5034/5044/5014/500C/DUMP;
 * argv "M N". See wiki Exp-2026-07-14 (narrow-output section). */
int ork_npu_requant_perchan_i32(ork_npu *c,const int32_t *a,const int16_t *b,int M,int N,int mult,int shift,int16_t *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;
    if(mult<0||mult>0x7fff||shift<0||shift>31) return -2;
    #define PC32(m,n) (((n)/8)*(M*32) + (m)*32 + ((n)%8)*4)          /* 4-byte atom=8 cube (int32 in) */
    #define PC16(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)          /* 2-byte atom=8 cube (int16 out) */
    size_t sza=(size_t)M*N*4; if(sza<4096)sza=4096;
    size_t szo=(size_t)M*N*2; if(szo<4096)szo=4096;
    struct buf A=orki_bcreate(fd,sza,0x403,-1), O=orki_bcreate(fd,szo,0x403,-1), B=orki_bcreate(fd,4096,0x403,-1);
    if(!A.cpu||!O.cpu||!B.cpu){ orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&B); return -2; }
    memset(A.cpu,0,sza); memset(O.cpu,0,szo); memset(B.cpu,0,4096);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++) *(int32_t*)((char*)A.cpu+PC32(m,n))=a[(size_t)m*N+n];
    for(int n=0;n<N;n++) ((int16_t*)B.cpu)[n]=b[n];                  /* per-channel vector CONTIGUOUS [N] int16 */
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (keep-warm-aware) */
    uint32_t rc[REGCMD_MUL_I16_N]; memcpy(rc,REGCMD_MUL_I16,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_MUL_I16_N,M,N);
    #define RQENV(nm,def) (getenv(nm)?(uint32_t)strtoul(getenv(nm),0,0):(uint32_t)(def))
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_OUT_PRECISION,RQENV("ORK_RQ_4010",0x30000001)); /* OUT int16 | IN int32 | PROC int16 */
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_5040,RQENV("ORK_RQ_MSTRIDE",(uint32_t)(M*32))); /* main int32 surf stride */
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_5018,(uint32_t)A.dma);        /* main input = int32 G */
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_5038,(uint32_t)B.dma);        /* per-channel scale vector */
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_5034,RQENV("ORK_RQ_5034",0x08)); /* operand per-channel, DATA_SIZE=2 (int16 b) */
    { const char*e=getenv("ORK_RQ_5044"); if(e) orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_5044,(uint32_t)strtoul(e,0,0)); } /* main-RDMA FEATURE_MODE: IN_PRECISION[17:15] */
    /* OVER-FETCH hack: RDMA input dims (0x500c width / 0x5014 channel) are DECOUPLED from the DPU output dims
     * (0x4058/0x405c). If the RDMA is stuck 2-byte fetching 2E but the DPU consumes 4E (int32) -> 50% starve,
     * INFLATE the RDMA element count so it fetches 4E bytes -> both terminal counts hit together -> clean. */
    { const char*e;
      if((e=getenv("ORK_RQ_5014"))) orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_5014,(uint32_t)strtoul(e,0,0)); /* RDMA cube CHANNEL */
      if((e=getenv("ORK_RQ_500C"))) orki_setrn(rc,REGCMD_MUL_I16_N,RK_SDP_500C,(uint32_t)strtoul(e,0,0)); /* RDMA cube WIDTH  */ }
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)mult); orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)shift);
    orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_OUT_CVT_OFFSET,0); orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_BS_ALU_CFG,0); orki_setrn(rc,REGCMD_MUL_I16_N,RK_DPU_EW_CVT_OFFSET,0);
    #undef RQENV
    if(getenv("ORK_RQ_DUMP")){ for(int k=0;k+1<REGCMD_MUL_I16_N;k+=2){ unsigned rg=rc[k]&0xffff; uint32_t v=((rc[k]>>16)&0xffff)|((rc[k+1]&0xffff)<<16);
        if(rg==0x4010||rg==0x4020||rg==0x4024||rg==0x40c0||rg==0x5018||rg==0x5034||rg==0x5038||rg==0x5040||rg==0x5044||rg==0x4084||rg==0x4088) fprintf(stderr,"  [rq] reg=%04x val=%08x\n",rg,v);} }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub; memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=1;
    sub.task_obj_addr=c->task.obj; sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1;
    sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; sub.timeout=orki_ew_timeout_ms();
    int ok=-1; double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; if(us)*us=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=*(int16_t*)((char*)O.cpu+PC16(m,n)); }
    else if(getenv("ORK_RQ_DUMP")){ int nz=0; int16_t*oc=O.cpu; for(size_t i=0;i<(size_t)M*N;i++) if(oc[i])nz++; fprintf(stderr,"  [rq] submit FAILED (errno path); O nonzero=%d/%d\n",nz,M*N); }
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&O); orki_bdestroy(fd,&B);
    #undef PC32
    #undef PC16
    return ok;
}

/* On-NPU PER-CHANNEL scale (int8): out[m][n] = clamp_i8( a[m][n] * b[n] * mult >> (shift-14) ). b is a
 * length-N per-channel vector broadcast across all M rows via the EW operand-b per-channel mode
 * (ERDMA_CFG 0x5034: ERDMA_DATA_MODE bits[31:30]=0, DATA_SIZE bits[3:2]=1 => reg 0x00000004; confirmed
 * on RK3588 — DATA_MODE=1 is per-element, =0 is per-channel broadcast). EW MUL unit; N%16, N<=8192.
 * Reusable: softmax normalize (b=1/Σ), LayerNorm/RMSNorm affine, per-channel requant. 0/ok, <0 err. */

/* RE PROBE: on-NPU PER-CHANNEL scale via the SDP BS (bias/scale) stage reading a per-channel vector from
 * memory. out[m][n] = a[m][n] * scale[n]  (scale broadcast across rows = NATIVE per-channel, no tiling).
 * Regs (rocket_registers.h — RK3588 offsets, NOT vanilla NVDLA): 0x501c RDMA_BRDMA_CFG (BRDMA_DATA_USE
 * bits[4:1]), 0x5020 RDMA_BS_BASE_ADDR (scale vector src); 0x4040 BS_CFG (BS_BYPASS b0 / BS_ALU_BYPASS b1
 * / BS_MUL_BYPASS b4 / BS_RELU_BYPASS b6), 0x4048 BS_MUL_CFG (BS_MUL_SRC b0 =MEM, BS_MUL_SHIFT_VALUE b[13:8]).
 * Config regs env-overridable (ORK_BS_R40/R48/BRDMA) to pin the DATA_USE/enable values on-board. int8. 0/ok. */
int ork_npu_probe_bs_scale(ork_npu *c,const int8_t *a,const int8_t *scale,int M,int N,int8_t *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<16||N>8192||(N&15)) return -2;
    #define BSCUBE(m,n) (((n)/16)*(M*16) + (m)*16 + ((n)%16))
    size_t sz=(size_t)M*N; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,-1), O=orki_bcreate(fd,sz,0x403,-1), S=orki_bcreate(fd,4096,0x403,-1);
    if(!A.cpu||!O.cpu||!S.cpu){ orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&S); return -2; }
    memset(A.cpu,0,sz); memset(O.cpu,0,sz); memset(S.cpu,0,4096);
    { int8_t*ac=A.cpu; for(int m=0;m<M;m++)for(int n=0;n<N;n++) ac[BSCUBE(m,n)]=a[(size_t)m*N+n]; }
    /* per-channel scale vector b[N]: try the EW-operand cube layout for a single width row (width=1). */
    { int8_t*sc=S.cpu; for(int n=0;n<N;n++) sc[(n/16)*16 + (n%16)]=scale[n]; }
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&S,RKNPU_MEM_SYNC_TO_DEVICE);
    ork_npu_enter(c,c->last_dt,XP_SDP,OCK_NONE);   /* transient SDP entry via the layer (XP_SDP=RC_SDPKW, keep-warm-aware; == the old inline reset) */
    /* EW MUL out=a*b, b (ERDMA 0x5038) as a PER-CHANNEL vector via ERDMA_DATA_MODE (0x5034 bits[31:30]). */
    uint32_t rc[REGCMD_MUL_N]; memcpy(rc,REGCMD_MUL,sizeof rc);
    orki_set_mul_geom(rc,REGCMD_MUL_N,M,N);
    orki_setrn(rc,REGCMD_MUL_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma);            /* output */
    orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5018,(uint32_t)A.dma);            /* input a (SRDMA, per-element) */
    orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5038,(uint32_t)S.dma);            /* scale b (ERDMA / EW_BASE) */
    #define ENV32(nm,def) (getenv(nm)?(uint32_t)strtoul(getenv(nm),0,0):(uint32_t)(def))
    orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5034,ENV32("ORK_ERDMA",0x40000000)); /* ERDMA_CFG: ERDMA_DATA_MODE bits[31:30] (sweep per-channel) */
    orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_SCALE,ENV32("ORK_BS_GAIN",0x00004000)); orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_SHIFT,28); /* out gain */
    #undef ENV32
    if(getenv("ORK_BS_DUMP")){ for(int k=0;k+1<REGCMD_MUL_N;k+=2){ unsigned rg=rc[k]&0xffff,ln=rc[k+1]>>16; uint32_t v=((rc[k]>>16)&0xffff)|((rc[k+1]&0xffff)<<16); if(rg==0x4020||rg==0x4070||rg==0x5018||rg==0x5034||rg==0x5038) printf("  reg=%04x lane=%04x val=%08x\n",rg,ln,v);} }
    memcpy(c->regcmd.cpu,rc,sizeof rc); orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub; memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=1;
    sub.task_obj_addr=c->task.obj; sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1;
    sub.subcore_task[0]=(struct rknpu_subcore_task){0,1}; sub.timeout=orki_ew_timeout_ms();
    int ok=-1; double t0=ork_now_us();
    if(!orki_rknpu_submit_ioctl(fd,&sub,-1)){ orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; if(us)*us=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ int8_t*oc=O.cpu; for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[(size_t)m*N+n]=oc[BSCUBE(m,n)]; }
    orki_bdestroy(fd,&A); orki_bdestroy(fd,&O); orki_bdestroy(fd,&S);
    #undef BSCUBE
    return ok;
}

/* Public on-NPU element-wise ADD (int8): out[m*N+n] = clamp_i8(round( (a*a_scale + b*b_scale)/out_scale ))
 * via the 2-input SDP op with ALU=add. Decoded structure: out = clamp((a*mult_a + b*mult_b) >> (0x4088-14) + zo),
 * mult_a=0x4084, mult_b=0x4078 are Q-format scale ratios (a_scale/out_scale, b_scale/out_scale). Symmetric quant
 * (zero-points 0). EXACT for RESIDUAL add (a_scale==b_scale==out_scale => out=clamp_i8(a+b)) and power-of-2
 * scale ratios; ARBITRARY unequal scales are approximate (the wide b-scale field 0x4078 isn't fully decoded).
 * Residual connections use equal scales, so the exact case is the intended one.
 * in/out int8 [M*N], N%16==0; rk3588-gated. 0/ok,-1,-2,-3. */

/* On-NPU fp16 element-wise ADD (residual): out[m][n] = a[m][n] + b[m][n] in fp16 via the 2-input SDP ALU=add op
 * (REGCMD_ADD_F16, gain 1). NVDLA fp16 cube (atom-8, 2-byte). in/out fp16 [M*N], N%8==0. 0/ok,-1,-2,-3. */

/* On-NPU int16 element-wise ADD: out = clamp_i16(round((a*a_scale + b*b_scale)/out_scale)) via the 2-input SDP
 * ALU=add op (REGCMD_ADD_I16). Same requant structure as int8. Residual (equal scales) => clamp_i16(a+b), exact;
 * arbitrary unequal scales approximate. in/out int16 [M*N], N%8==0. 0/ok,-1,-2,-3. */

/* ── Standalone on-NPU SiLU (activation-LUT SDP op) — RE probe ─────────────────────────────────────
 * Applies the PPU activation LUT to a SINGLE int8 memory input [M][N] via the standalone 69-reg/enable=0x18
 * SDP op (REGCMD_SILU_STD), reprogrammed to (M,N) by orki_set_mul_geom. Two submits on the single-stream NPU:
 * (1) LUT-load (REGCMD_SILU_LUT, streams the int16 curve into PPU SRAM); (2) the standalone op reads it.
 * SDP math: idx=(in*R)>>6 + C0(idx_off); out=clamp_i8(R*LUT-interp(idx) + out_bias); R=r_mult/2^r_shift.
 * The caller supplies the scale regs (r_mult,r_shift,out_bias,idx_off,cfg4064,cfg4068) and the LUT; lut==NULL
 * keeps the captured curve. This is the RE/calibration entry (measure idx(in) with a ramp LUT, then build the
 * silu curve at those indices — same 2-pass scheme as ork_mm_silu_build_lut but through THIS op, not a matmul).
 * in/out int8 [M*N] row-major; N%16==0. 0/ok, -1 wedged, -2 bad shape, -3 non-rk3588. */

/* fp16 standalone activation-LUT op — RE probe. Applies the PPU LUT to a SINGLE fp16 memory input [M][N] via
 * REGCMD_SILU_STD_F16 (single-input, fp16 gain 0x00010001). Two submits: REGCMD_SILU_LUT (load) + this op.
 * The LUT data words are streamed via 0x4104 verbatim (encoding TBD by calibration — pass lut as the raw 16-bit
 * words the op consumes). in/out fp16 [M*N], N%8==0; rk3588-gated. 0/ok,-1 wedged,-2 shape,-3 SoC. */

/* Faithful fp16 replay: run RKNN's fp16 LUT-LOAD program (loader/ln, the LE-table exponential-mode loader with
 * RKNN's curve baked in) verbatim + the fp16 compute op (REGCMD_SILU_STD_F16) verbatim — patching only the I/O
 * addresses + M/N. Unlike ork_npu_probe_silu_std_f16, this uses the fp16 loader (NOT the int8 LO-table loader)
 * and keeps the compute's baked index params. in/out fp16 [M*N], N%8==0. 0/ok,-1,-2,-3. */

/* ---- FORWARD-SOFTMAX REPLAY (RE) --------------------------------------------------------------------
 * Replay the captured vendor forward-softmax 9-task PC-chained graph VERBATIM at its capture geometry
 * (reduction N=64, 256 rows). Proves the activation->matmul asymmetry (NPU-Quirks) is resolved by
 * PC-chaining all 9 tasks in ONE submit: max -> broadcast -> x-max -> exp(LUT) -> Sum(=exp.ones, a MATMUL
 * right after the exp SDP op) -> 1/Sum -> exp*(1/Sum). Every register VALUE is vendor-validated; we only
 * rebase DATA addresses (0x4020/0x5018/0x5038/0x1070/0x1110) + the 0101:0x0010 chain to our buffers, so
 * hard-wedge risk is low (a bad address soft-times-out). `in`/`out` are raw 32768-byte buffer images (the
 * captured h4/h5); fill `in` uniformly and expect softmax=1/64 everywhere (layout-agnostic). Buffers:
 *   IN  [0xfffb6000,+0x8000)  OUT [0xfffae000,+0x8000)  SCR [0xfffbe000,+0x3a000)  WT [0xffffa280,+..)
 *   LUT scratch [0xffffab00,..) = task4's throwaway output (the exp LUT itself loads to DPU SRAM).
 * 0/ok, -1 wedge/submit-fail, -2 alloc, -3 SoC. */

/* int16 (w16a16i) standalone activation-LUT op — RE probe. Same requant-LUT math as the int8 op
 * (out=clamp_i16(R*LUT-interp(idx)+out_bias)) but int16 I/O (atom-8 2-byte cube) via REGCMD_SILU_STD_I16.
 * Two submits (LUT-load + op). Caller supplies scale regs + LUT. in/out int16 [M*N], N%8==0. 0/ok,-1,-2,-3. */

/* Replay an ASSEMBLED int16 LUT-op (from tools/re/assemble_op.c): stream `lut` via the LUT-loader, then run the
 * given `regcmd` VERBATIM (RKNN's matched index/scale params baked in) — patching only addresses + M/N geometry.
 * Bit-exact to RKNN, no index decode. in/out int16 [M*N], N%8==0. 0/ok,-1 wedged,-2 shape,-3 SoC. */
/* #38 RE: replay a CAPTURED int8 matmul regcmd (e.g. rkllm's, from tools/re/regcmd_capture) on ork's own
 * submit path, to TIME the foreign schedule on the same silicon. Data is garbage (memset) — TIMING ONLY,
 * output not checked. Allocates A[M*K]/B[K*N]/C[M*N*4] in the active domain, patches the A/B/C address regs
 * (0x1070/0x1110/0x4020 — the only addresses a plain INT8_MM_INT8_TO_INT32 regcmd references) to them,
 * submits single-core task_number=1 (chain-links ignored at n=1), warms once, times `iters`. rkllm's task
 * descriptor is identical to ork's (enable=0xd,int_mask=0x300,regcfg_amount=108 — verified from capture).
 * SAFETY: orki_mm_timeout_ms() so a bad regcmd self-terminates (soft, reboot-recoverable) rather than hanging. */
/* #39 A-layout solver: submit a FIXED (captured) regcmd for `nvar` A-variants, reusing ONE buffer set so the
 * IOVA is stable across submits. The fresh-alloc-per-call path (ork_npu_replay_i8 called N times) intermittently
 * wedges on the fold; buffer reuse is the safe pattern (cf. replay_mm_i8's iters loop, which never wedged).
 * Avar = nvar A-images, each `astride` bytes; Bdata = shared weight; Couts = nvar contiguous M*N int32 results.
 * A warm submit (variant 0) precedes the measured loop. Returns 0/ok, -2 bad shape, <0 on submit error. */
/* #39 A-layout MAPPER (Route A): recover the fold's A-read map — for each weight one-hot position Bpos[i]
 * (= ork_woff byte for logical (n0, k0_i)), which A-buffer BYTE OFFSET does the fold pull for each output
 * position? With a one-hot weight at (k0,n0), C[*][n0] = A_read(*,k0). We run 4 submits/k0 on ONE buffer set
 * (stable IOVA, wedge-safe): (P) presence A=1 -> the raw-C entries that are 1 mark this k0's M output slots;
 * (0,1,2) A[j]=(j>>7p)&0x7f base-128 digits -> offset = d0 | d1<<7 | d2<<14 at each slot. Fills, per k0 i:
 * rpos[i*M+t] = raw int32 index of the t-th output slot, aoff[i*M+t] = the A byte offset it read (t<cnt[i]).
 * Returns 0/ok. Offsets up to 2MB (fits the oversized A buffer). */

/* RE/validation for the FUSED SiLU output stage (step 2): run a full-K int8 matmul with SiLU applied
 * on-chip, returning C[M*N] as int8. TWO submits on the single-stream NPU: (1) the LUT-load program
 * (REGCMD_SILU_LUT, enable=0x18) streams the int16 silu curve into PPU LUT SRAM; (2) the matmul compute
 * (synth_i8 + set_i8_silu, enable=0x1d) reads that LUT in its output stage. The LUT persists in SRAM
 * between the two sequential submits. Isolated buffers. A[M*K] B[K*N] row-major int8; C[M*N] int8.
 * 0/ok (path executed, C valid), -1 wedged, -2 bad dims. FIRST-RUN status: replays the mm_silu capture
 * scale/LUT verbatim — proves the path EXECUTES + LUT-persists; per-scale generation is WIP. */
/* Fused-SiLU probe with the decoded knobs exposed (see set_i8_silu): r_mult/r_shift = the unified scale R
 * (0x4084/0x4088), out_bias = 0x4080, idx_off = 0x4110, cfg4068 = 0x4068. lut != NULL overrides the LUT
 * contents (streams the first `nlut` int16 values into the LUT-load's 0x4104 writes) — for the staircase/
 * const-LUT calibration harness; lut==NULL keeps the captured fixed silu curve. */


/* ork_mm_silu_build_lut — generate ork's OWN silu LUT for the fused-output path (ork-NATIVE: no RKNN
 * dependence, works on ork's 108-reg matmul program). Since ork controls both the LUT and the output-stage
 * registers, correct fused SiLU is a 2-step construction (see tools/silu_native.c, validated ~1 int8):
 *   (1) MEASURE ork's index(acc) for (r_mult,r_shift,cfg4068) via one ramp-LUT calibration submit;
 *   (2) BUILD lut[idx(acc)] = clamp_int16( silu(acc*in_scale)/out_scale / R ), R=r_mult/2^r_shift; interp gaps.
 * The caller then runs the matmul via ork_npu_probe_i8_silu_cfg(..,r_mult,r_shift,0,0xffffc000,cfg4068,lut,1030,..)
 * — do the build ONCE per (registers) and reuse the lut across matmuls of the same scale. Pick r_mult/r_shift
 * so R ~= 660*in_scale (the matmul's acc range then spans silu's transition band). out_bias MUST be 0 (the
 * validated config; the ramp readback assumes it). Fills lut[1030]. 0/ok, -1 fail. */
/* Fused EXP LUT for the coalesced chain output stage (softmax): lut[idx(acc)] = clamp(exp(acc*in_scale)/out_scale/R).
 * Same 2-pass calibration as silu; scores must be <=0 (post-max softmax domain) so exp in (0,1] fits int8 (acc>0
 * entries clamp). Lets run_chain_i8_gsilu HW-chain exp onto the score matmul in ONE submit. 0/ok,-1 fail. */
int ork_mm_chain_build_exp_lut(ork_npu*c, double in_scale, double out_scale,
                               int r_mult, int r_shift, uint32_t cfg4068, int16_t *lut){
    return orki_chain_build_lut_fn(c, orki_exp_f, in_scale, out_scale, r_mult, r_shift, cfg4068, lut);
}

/* Constant SDP index params for the standalone activation-LUT op (from the RKNN SiLU capture). The op's
 * index math idx(in) depends ONLY on these (not on in/out scale), so one calibration serves every scale. */

/* Calibrate idx(in) for the standalone activation op ONCE per ctx: run a ramp LUT (LUT[i]=i-512) at R=0.5
 * (out=clamp_i8(0.5*(idx-512)) -> idx=2*out+512), sweeping all 256 int8 inputs. R=0.5 keeps out unclamped
 * across the full input range. Fills c->silu_idx[(uint8)in] (=-1 where saturated). 0/ok, -1 wedged. */

/* build a LUT curve at the calibrated indices for f(v*in_scale)/out_scale (R=1); interp gaps, hold ends */
/* build the SDP activation LUT for f((vv-bias)*in_scale)/out_scale over int8 index vv. bias is a scalar shift
 * in int8-input units (0 for the plain curve). Used by softmax's scalar GLOBAL-max subtract: exp((x-max)*sc)
 * keeps the argument <=0 (output in (0,1]) so int8 exp never overflows — a scalar bias is bakeable into the
 * LUT (unlike a per-row max, which would need the dead per-channel-add), and the constant cancels in P=e/Sum. */

/* Generic int8 pointwise activation via the standalone SDP LUT op: out = clamp_i8(round( f((in-bias)*in_scale)/out_scale )).
 * The op is activation-agnostic (the LUT contents define f); calibrate idx once, build f's curve, run. bias is a
 * scalar shift in int8-input units (0 for plain). */

/* Public on-NPU SiLU (int8): out[m*N+n] = clamp_i8(round( silu(in[m][n]*in_scale) / out_scale )) via the
 * standalone SDP activation-LUT op. Lazily calibrates idx(in) once per ctx, builds the silu curve for the
 * requested (in_scale,out_scale), loads it, and runs the op (2 submits). in/out int8 [M*N], N%16==0.
 * rk3588-gated. 0/ok, -1 wedged, -2 bad shape, -3 non-rk3588. */
/* On-NPU GELU (int8): same standalone activation-LUT op, GELU curve. Bit-exact-class like SiLU. */
/* On-NPU rsqrt (int8) — RMSNorm building block: out = clamp_i8(round( rsqrt(in*in_scale)/out_scale )). */
/* On-NPU exp (int8) — softmax building block: out = clamp_i8(round( exp(in*in_scale)/out_scale )). */
/* On-NPU exp with a scalar GLOBAL-max subtract baked into the LUT: out = clamp_i8(round( exp((in-max)*in_scale)/out_scale )).
 * The softmax numerator with numerically-stable max-subtract, WITHOUT a per-row op: max is the scalar global max
 * (>= every row max, so every argument (in-max)<=0 => exp in (0,1], no int8 overflow), and the constant cancels
 * in P=e/Sum. `max` is in int8-input units (the int8 score value). Direct only for now (Path B TODO). */

/* int16 index params = RKNN's CAPTURED int16 SiLU index params (from REGCMD_SILU_STD_I16). Their gain (~0.008)
 * maps the FULL int16 input range onto a ~525-wide LUT-index band centred at ~510 — the same regime RKNN uses,
 * so the idx is densely + smoothly sampled and place-and-interpolate reproduces silu well (unlike int8's params
 * which give gain~2 and only span |in|<256). We keep these index params and only override R->1 / out_bias->0,
 * then build our own curve for the caller's (in_scale,out_scale) — mirroring the working int8 path. */

/* Calibrate the int16 idx map once per ctx at RKNN's index params, AT R=1 (== the run R, so the measured idx
 * matches the run exactly — no R-dependent idx shift). Ramp LUT (interpolated exactly) -> out=idx-512 ->
 * idx=out+512 (integer). Dense sampling (step 16) resolves the idx transitions for accurate inversion.
 * silu_idx16[s] = measured integer LUT index for q_in=-32768+s*16; INT16_MIN if saturated. */

/* PHASE 0 (#35 chained-FFN): a HETEROGENEOUS 2-task PC-chain in ONE submit — task0 = a tiny int8 matmul
 * (enable 0xd), task1 = the int16 silu SDP LUT-op (enable 0x18) — to prove the hardware walks the
 * CNA/DPU(matmul)->pure-SDP(silu) transition WITHOUT a per-op host round-trip/reset (the FFN-chain-critical
 * transition). Data handoff is NOT wired yet: the matmul writes a scratch (verifies task0 ran via Cd), the
 * silu reads a real int16 `in` (verifies task1 ran via `out`). Both correct + rc=0 => the heterogeneous
 * chain walked. Descriptor format mirrors the mcworker PC-chain (rc[216..219], 0x0101 next-addr/regamt).
 * in/out int16 [M*N], N%8==0. 0/ok, -1 wedge/timeout, -2 shape/alloc, -3 non-rk3588. */

/* (b) MIXED-PRECISION CHAIN probe: chain a FP16 matmul task (synth) + an INT16 silu-SDP task in ONE submit
 * and validate BOTH — the exact fp16-matmul + int16-elementwise mix the fused SSD scan needs. Confirms that
 * fp16 and int16 tasks coexist in one PC-chain (precision is per-task regcmd on the shared 2-byte datapath;
 * no inter-task reset). Mirrors chain_mm_silu_i16 but task0 is fp16 (all-1s 32x32 -> C=32.0) instead of int8.
 * Sets *mm_ok (fp16 matmul C≈32 in-chain), *silu_ok (int16 silu vs CPU). 0/ok,-1 fail,-2 bad,-3 SoC. */
int ork_ssd_probe_mixchain(ork_npu *c,int *mm_ok,int *silu_ok,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems, dom=c->dom_active;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    const int M=8,N=64; const double in_scale=8.0/32000.0, out_scale=1.0/32000.0;
    if(orki_silu_calibrate_idx16(c)) return -1;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)
    static double qsum[1030]; static int qn[1030];
    for(int k=0;k<1030;k++){ qsum[k]=0; qn[k]=0; }
    for(int s=0;s<SILU16_NS;s++){ int k=c->silu_idx16[s]; if(k<0||k>1029)continue; qsum[k]+=-32768.0+s*SILU16_QSTEP; qn[k]++; }
    int16_t lut[1030]; int lo=-1,hi=-1;
    for(int k=0;k<1030;k++){ if(qn[k]){ if(lo<0)lo=k; hi=k; double q_in=qsum[k]/qn[k]; double val=orki_silu_f(q_in*in_scale)/out_scale;
        long q=lround(val); if(q>32767)q=32767; if(q<-32768)q=-32768; lut[k]=(int16_t)q; } else lut[k]=0; }
    if(lo<0) return -1;
    for(int k=0;k<lo;k++)lut[k]=lut[lo]; for(int k=hi+1;k<1030;k++)lut[k]=lut[hi];
    for(int k=lo;k<=hi;k++){ if(qn[k])continue; int a=k,b=k; while(a>lo&&!qn[a])a--; while(b<hi&&!qn[b])b++;
        lut[k]=(int16_t)(lut[a]+(lut[b]-lut[a])*(k-a)/(b-a)); }
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;
    struct buf A=orki_bcreate(fd,sz,0x403,dom), O=orki_bcreate(fd,sz,0x403,dom);
    struct buf Lrc=orki_bcreate(fd,(size_t)REGCMD_SILU_LUT_N*4,0x403,dom), Lsc=orki_bcreate(fd,4096,0x403,dom);
    struct buf Wd=orki_bcreate(fd,32*32*2,0x403,dom), Ad=orki_bcreate(fd,32*2,0x403,dom), Cd=orki_bcreate(fd,32*4,0x403,dom); /* fp16 A/W, fp32 C */
    int ret=-1; int16_t *inb=malloc((size_t)M*N*2);
    if(!A.cpu||!O.cpu||!Lrc.cpu||!Lsc.cpu||!Wd.cpu||!Ad.cpu||!Cd.cpu||!inb){ goto mfail; }
    memset(A.cpu,0,sz); memset(O.cpu,0,sz); memset(Cd.cpu,0,32*4);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int16_t v=(int16_t)((m*N+n)%20000-8000); inb[m*N+n]=v; *(int16_t*)((char*)A.cpu+EWCUBEH(m,n))=v; }
    { uint16_t*wd=Wd.cpu,*ad=Ad.cpu; for(int i=0;i<32*32;i++)wd[i]=0x3c00; for(int i=0;i<32;i++)ad[i]=0x3c00; } /* fp16 1.0 -> C=32 */
    orki_bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&Wd,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&Ad,RKNPU_MEM_SYNC_TO_DEVICE);
    memcpy(Lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
    orki_setrn((uint32_t*)Lrc.cpu,REGCMD_SILU_LUT_N,RK_DPU_DST_BASE_ADDR,(uint32_t)Lsc.dma);
    { uint32_t*lr=(uint32_t*)Lrc.cpu; int j=0; for(int k=0;k+1<REGCMD_SILU_LUT_N;k+=2){ if((lr[k]&0xffff)==0x4104){ int32_t v=(j<1030)?(int32_t)lut[j]:0; j++;
        lr[k]=0x4104|((uint32_t)(v&0xffff)<<16); lr[k+1]=(0x1001u<<16)|(((uint32_t)v>>16)&0xffff); } } }
    orki_bsync(fd,&Lrc,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task*t=c->task.cpu; memset(t,0,sizeof *t); t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=1097; t->regcmd_addr=Lrc.dma;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit s;memset(&s,0,sizeof s);s.flags=0x1;s.task_number=1;s.task_obj_addr=c->task.obj;s.core_mask=RKNPU_CORE0_MASK;s.fence_fd=-1;s.timeout=orki_ew_timeout_ms();s.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      if(orki_rknpu_submit_ioctl(fd,&s,dom)) goto mfail; }
    /* chain: [0]=FP16 matmul (synth) -> [1]=int16 silu */
    { uint32_t *mm=(uint32_t*)c->regcmd.cpu, *si=(uint32_t*)((char*)c->regcmd.cpu+(size_t)REGCMD_I8_N*4);
      memset(mm,0,REGCMD_I8_N*4);
      orki_synth(mm,1,32,32,(uint32_t)Ad.dma,(uint32_t)Wd.dma,(uint32_t)Cd.dma,0,CBUF);   /* FP16 matmul task0 (sched=0: K=32<96 small-K 0x1040 fix) */
      uint64_t nx=c->regcmd.dma+(size_t)REGCMD_I8_N*4;
      mm[216]=0x0010|((nx&0xffff)<<16); mm[217]=(0x0101u<<16)|((nx>>16)&0xffff);
      mm[218]=0x0014|(((69+3)/2)<<16);  mm[219]=(0x0101u<<16)|0;
      memcpy(si,REGCMD_SILU_STD_I16,(size_t)REGCMD_SILU_STD_I16_N*4);
      orki_set_mul_geom(si,REGCMD_SILU_STD_I16_N,M,N);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_SDP_5040,0); orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_SDP_5038,0);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O.dma); orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_SDP_5018,(uint32_t)A.dma);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_SCALE,0x4000u); orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_SHIFT,14u); orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_OFFSET,0u);
      orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_R4110,ORK_SILU16_IDXOFF); orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_BN_ALU_CFG,ORK_SILU16_C4064); orki_setrn(si,REGCMD_SILU_STD_I16_N,RK_DPU_BN_MUL_CFG,ORK_SILU16_C4068);
      orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
      struct rknpu_task*tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,2*sizeof *tk);
      tk[0].enable_mask=0xd;  tk[0].int_mask=0x300; tk[0].int_clear=0x1ffff; tk[0].regcfg_amount=108; tk[0].regcmd_addr=c->regcmd.dma;
      tk[1].enable_mask=0x18; tk[1].int_mask=0x300; tk[1].int_clear=0x1ffff; tk[1].regcfg_amount=69;  tk[1].regcmd_addr=nx;
      orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit s;memset(&s,0,sizeof s);s.flags=0x1;s.task_number=2;s.task_obj_addr=c->task.obj;s.core_mask=RKNPU_CORE0_MASK;s.fence_fd=-1;s.timeout=orki_ew_timeout_ms();s.subcore_task[0]=(struct rknpu_subcore_task){0,2};
      double t0=ork_now_us();
      for(int rep=0;rep<2;rep++){ if(orki_rknpu_submit_ioctl(fd,&s,dom)) goto mfail;   /* rep0 primes fresh buffers */
          orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd,&Cd,RKNPU_MEM_SYNC_FROM_DEVICE); }
      if(us)*us=ork_now_us()-t0; }
    { float *cd=Cd.cpu; int ok=1; for(int i=0;i<32;i++) if(fabs(cd[i]-32.0)>0.5) ok=0; if(mm_ok)*mm_ok=ok; }   /* fp16 matmul: C=32 */
    { int ok=1,bad=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double ref=orki_silu_f(inb[m*N+n]*in_scale)/out_scale;
        double got=(double)*(int16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(fabs(got-ref)>0.03*fabs(ref)+3) bad++; }
      ok=(bad<=(M*N)/20); if(silu_ok)*silu_ok=ok; }
    ret=0;
mfail:
    free(inb); orki_bdestroy(fd,&A);orki_bdestroy(fd,&O);orki_bdestroy(fd,&Lrc);orki_bdestroy(fd,&Lsc);orki_bdestroy(fd,&Wd);orki_bdestroy(fd,&Ad);orki_bdestroy(fd,&Cd);
    #undef EWCUBEH
    return ret;
}

/* CHAIN ASSEMBLER increment-1: DATA-CONNECTED int8-matmul(int16-out) -> int16-silu in ONE PC-chain via the
 * general ork_npu_chain_progs core. Unlike ork_npu_chain_mm_silu_i16 (which only proves the chain WALKS --
 * its matmul output and silu input are SEPARATE buffers), here the gate matmul's int16 output buffer G IS
 * the silu's input, so it validates the INTERMEDIATE-BUFFER BRIDGE (the crux of the FFN chain): does the
 * matmul set_i16_out layout match the silu's 0x5018 EWCUBEH cube input layout. Computes
 *   out = clamp_i16( silu( gate_i16 * in_scale ) / out_scale ),  gate_i16 = requant_i16(A[M,K] x B[K,N]).
 * A int8 [M*K] row-major, B int8 [K*N] row-major (de-tiled here); mult/shift = the int32->int16 requant.
 * gate_out (nullable) returns G read back via EWCUBEH so a caller can localize a mismatch to the matmul
 * stage vs the silu stage. Small-shape validated regime (probe_i16_out N range). 0/ok,-1 wedge,-2 dims,-3 SoC. */

/* CHAIN (M4 building block): int8-matmul(int16-out) -> PER-CHANNEL-scale(int16) in ONE PC-chain via
 * chain_progs — the A·V->normalize pattern. The matmul's int16 output G IS the per-channel op's input
 * (0x5018), validating the intermediate-buffer bridge (like ork_npu_chain_gatesilu_i16 but SDP=per-channel
 * scale instead of silu). out = clamp_i16( requant_i16(A[M,K]xB[K,N], m1,s1) * scale[n] * m2 >> s2 ).
 * A int8[M*K], B int8[K*N] row-major, scale int16[N] per-channel. K%32,N%32,N<=nmax,M<=64. 0/ok,<0. */

/* fp16-IN chained matmul -> per-channel SDP (the attention A·V normalize path, all-fp16 like the vendor conv->mul).
 * Closes the single-submit chain with CORRECT values: fp16 matmul (synth, weight tile [N/16][K/32][16][32], fp16
 * activation) -> fp16-out G (set_f16_out with 0x4010=0x48000002 fp16-in) -> the vendor REGCMD_MUL_F16_CHAIN 2-input
 * SDP scales G per-channel by scale[N]. A[M*K],B[K*N],scale[N],out[M*N] are fp16 bit patterns (uint16). 0/ok,-1,-2,-3. */

/* (B') fp16 cross-slice DRAIN-GAP probe. Chains TWO wide fp16 matmuls with DISTINCT weight buffers (W0,W1) —
 * reproducing the cross-buffer base-latch condition — into ONE PC-chain submit, optionally with an identity
 * mul_perchan_f16 GAP between them (REGCMD_MUL_F16_CHAIN, enable 0x18, 69 regs, middle desc_slot=138 per the
 * regcfg*2 convention). The gap is a pure time-filler on a DUMMY fp16 scratch (identity scale) — its only job is
 * to idle the weight-CDMA so W0's fetch drains before W1's base latches. Both matmuls use the fp16-out stage
 * (orki_set_f16_out_fp16in — the config with the PROVEN mm->perchan HW edge). Returns the submit rc (nonzero=wedge);
 * *nz0/*nz1 = count of nonzero output words per matmul (coarse "did it compute" sanity). rk3588-gated. */

/* PER-CORE-FD concurrency probe. Runs a 3-core fp16 N-column-split matmul where EACH core submits on its OWN
 * fresh DRM fd (a separate open of the card), not the shared c->fd — to test whether per-core-fd isolation
 * changes the concurrent-fetch "wedge". Core i owns columns [n0,n0+Ncol), n0=i*Ncol, Ncol=N/cores; it computes
 * ALL M rows of its column slice as an independent fp16 matmul (fp16-out, converted to fp32 into Cout).
 *   mode 0: each core gets its OWN weight copy (bcreate on its own fd), tiled from B's column slice.
 *   mode 1: ONE shared dma-heap weight (full K*N, tiled once) imported into every core's fd (bimport +
 *           bimport_fd); each core points its weight addr at the byte offset of its column-tile block.
 * Constraint set (kept simple): Sk=1 (single K, no host accumulate), Sn=1 (N<=nmax), M<=64, K%32==0, N%16==0,
 * and N%(cores*16)==0 so each core's Ncol is a multiple of 16 (a clean col-tile boundary). *us = wall time
 * around the concurrent submit region. Returns 0 on a completed run — INCLUDING a wedge, in which case Cout
 * shows the dropped/garbage columns (that IS the signal); <0 only on genuine setup failure (-2 bad shape,
 * -1 open/alloc). See tools/percore_fd_probe.c. */

/* Public on-NPU SiLU (int16 / w16a16i): out = clamp_i16(round( silu(in*in_scale)/out_scale )) via the standalone
 * int16 activation-LUT op, using RKNN's captured index params (full-range coverage). Builds the LUT the way RKNN
 * does — for each integer LUT index k it finds the q_in whose idx==k (from the dense R=1 calibration transitions)
 * and samples silu there, so the op's interpolation lands on silu at the exact grid. RKNN-class accuracy.
 * in/out int16 [M*N], N%8==0. rk3588-gated. 0/ok,-1 wedged,-2 bad shape,-3 SoC. */
/* Build the int16 activation LUT curve for `f` at (in_scale,out_scale) into lut[1030]. Calibrates the idx map
 * once per ctx (a STANDALONE probe — must run outside any chain). 0/ok. Shared by orki_act_lut_i16 (standalone) and
 * the HW-chained silu prologue in ork_dyn_begin_seq_i8_mc. */
/* On-NPU GELU (int16 / w16a16i): same activation-LUT op, GELU curve. RKNN-class accuracy like SiLU int16. */
/* On-NPU rsqrt (int16) — RMSNorm building block. RKNN-class accuracy. */
/* On-NPU exp (int16) — softmax building block. RKNN-class accuracy. */

/* RE: does batching tasks per ioctl amortize the RKNPU_SUBMIT round-trip floor? Runs `ntask`
 * identical small int8 matmuls (single core) as (a) ntask separate task_number=1 ioctls vs (b) ONE
 * ioctl with task_number=ntask. Returns 0/ok, -1 wedge, -2 bad dims (K%32, N%32, 1<=ntask<=32).
 * FINDING (2026-06-13): the batched path (b) TIMES OUT (`task counter: 0x0` — NPU dispatches no tasks;
 * kernel soft-resets + recovers). The naive task[]/subcore config doesn't drive multi-task execution.
 * AND it's moot for cross-matmul batching: the closed runtime's captured 12-task submit is ONE
 * matmul's program (4 sub-tasks × 3 subcores), NOT multiple matmuls batched — so librkllmrt also does
 * ~1 submit/matmul and pays the same per-matmul submit floor (~11 tok/s on 1.7B, which ork-driver
 * matched). The floor is inherent; cross-matmul task-batching is not the reference's mechanism nor
 * the lever. See tools/batch_probe.c. */
int ork_npu_probe_batch(ork_npu*c,int ntask,int K,int N,double*us_unbatched,double*us_batched){
    int fd=c->fd,CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||ntask<1||ntask>32) return -2;
    struct buf W=orki_bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    memset(W.cpu,1,(size_t)K*N); orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=orki_bcreate(fd,(size_t)N*4,0x403,-1); if(!O.cpu){orki_bdestroy(fd,&W);return -2;}
    int8_t*ad=c->Af.cpu; memset(ad,1,K); orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I8_N]; orki_synth_i8(rc,1,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    orki_setrn(rc,REGCMD_I8_N,RK_CNA_CBUF_CON0,0xb1);
    struct buf extra[2] = {W, O};
    if (orki_validate_regcmd("probe_batch", c, rc, REGCMD_I8_N, NULL, extra, 2)) { orki_bdestroy(fd,&W); orki_bdestroy(fd,&O); return -1; }
    for (int i = 0; i < ntask; i++) {
        memcpy((char*)c->regcmd.cpu + i * sizeof(rc), rc, sizeof(rc));
    }
    orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task*t=c->task.cpu;                 /* task[] array: ntask tasks, separate regcmd spaces */
    for(int i=0;i<ntask;i++){memset(&t[i],0,sizeof t[i]);t[i].flags=0;t[i].op_idx=i;t[i].enable_mask=0xd;t[i].int_mask=0x300;t[i].int_clear=0x1ffff;t[i].regcfg_amount=108;t[i].regcmd_addr=c->regcmd.dma + i * sizeof(rc);}
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    /* single-core: set all subcore_task entries to avoid kernel UAPI timeout/Oops */
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=ork_ppflags();sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.timeout=orki_mm_timeout_ms();
    sub.task_number=1; sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,1};
    if(orki_rknpu_submit_ioctl(fd,&sub,-1)){orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);return -1;} orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); /* warm */
    double t0=ork_now_us();                          /* (a) ntask separate ioctls */
    for(int i=0;i<ntask;i++){ sub.task_number=1; sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,1};
        if(orki_rknpu_submit_ioctl(fd,&sub,-1)){orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);return -1;} orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); }
    *us_unbatched=ork_now_us()-t0;
    sub.task_number=ntask; sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)ntask};
    t0=ork_now_us();                                 /* (b) one ioctl, ntask tasks */
    if(orki_rknpu_submit_ioctl(fd,&sub,-1)){perror("batched SUBMIT");orki_bdestroy(fd,&W);orki_bdestroy(fd,&O);return -1;} orki_bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE);
    *us_batched=ork_now_us()-t0;
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&O); return 0;
}

/* RE: probe in-place K-slicing of a FULL-K weight buffer (for a single-layout decode+prefill).
 * Packs B[Kfull,N] fp16 in full-K tile layout, then runs ONE M=1 submit over k in [0,Kp) reading
 * from that buffer — i.e. the op processes Kp passes but the weights are laid out for Kfull. With
 * no override the per-N-tile stride is Kp's (N-tile 0 correct, 1+ wrong); pass reg/val overrides
 * (e.g. 0x1044, 0x1034, 0x1030 set to their full-K values) to hunt the stride register that makes
 * all N-tiles correct. C[N] = sum_{k<Kp} A[k]*B[k][n] if slicing is right. nov<=4. Returns 0/ok. */

/* RE: probe W4A4 (int4 A x int4 B -> int16 C) using the captured REGCMD_I4 (M=4, the capture's M).
 * A[M*K], B[K*N] hold int4 values as int8 in [-8,7]. `blayout`/`alayout` select candidate native
 * tile packings (2 int4/byte); the regcmd is correct (captured) so a layout combo that matches the
 * CPU reference reveals the native tile order. C[M*N] int16 = sum_k A[m][k]*B[k][n]. `nov`/`ovr_*`
 * patch extra CNA regs from the tool. Returns 0 ok, -1 wedge/abort, -2 bad dims. Single task[0]
 * submit — if the 12-task W4A4 program needs the other tasks (A-quant/reorder), this is wrong and a
 * multi-task path is needed (see ROADMAP). Layouts: 0=K-contig lo/hi, 1=K-contig hi/lo, 2=N-lane. */
#define ORK_I4_M 1   /* the captured W4A4 program M-tiles: each task is one M=1 GEMM (task[0] here) */
/* RK3588/3576 int4 native layouts (DOCUMENTED in rknn_matmul_api.h, not guessed):
 *   A: (K/32, M, 32)        elem[kt][m][kk] = A[m][kt*32+kk]
 *   B: (N/64, K/32, 64, 32) elem[nt][kt][nl][kk] = B[kt*32+kk][nt*64+nl]   (B row-major [K][N])
 * 2 int4 packed per byte; `nib` toggles which of the two consecutive elements is the high nibble. */

/* RE/calibration: ONE multi-M int4 submit (mc=M) — the M-scheduler experiment for Tier 4b. A is laid
 * (K/32,M,32) native; B native (N/64,K/32,64,32); the regcmd's M-count regs are set to M (synth_i4
 * mc=M). Copies the RAW int16 output buffer (M*N int16, NO de-tile) to raw so the caller can deduce
 * the multi-M C layout against a CPU reference. Returns 0 if the submit completed, -1 wedged, -2 dims.
 * Single N-slice (N<=nmax), single core. See tools/i4_multim_probe.c. */


int ork_npu_benchmark_chain(ork_npu *c, int S, int K, int N, int iters) {
    int fd = c->fd, CBUF = c->soc->cbuf_elems;
    if (K % 32 || N % 32 || N > c->soc->nmax || S < 1 || S > 64) return -2;
    
    struct buf W = orki_bcreate(fd, (size_t)K * N, 0x403,-1);
    struct buf A = orki_bcreate(fd, (size_t)S * K, 0x403,-1);
    struct buf O = orki_bcreate(fd, (size_t)S * 4096, 0x403,-1);
    
    struct buf regs_chain = orki_bcreate(fd, (size_t)S * REGCMD_I8_N * 4, 0x403,-1);
    struct buf regs_sep = orki_bcreate(fd, (size_t)S * REGCMD_I8_N * 4, 0x403,-1);
    
    struct buf task_chain = orki_bcreate(fd, (size_t)S * sizeof(struct rknpu_task), 0x40b,-1);
    struct buf task_sep = orki_bcreate(fd, (size_t)S * sizeof(struct rknpu_task), 0x40b,-1);
    
    if (!W.cpu || !A.cpu || !O.cpu || !regs_chain.cpu || !regs_sep.cpu || !task_chain.cpu || !task_sep.cpu) {
        fprintf(stderr, "[ork] ERROR: failed to allocate benchmark_chain buffers (IOMMU full?)\n");
        orki_bdestroy(fd, &W); orki_bdestroy(fd, &A); orki_bdestroy(fd, &O);
        orki_bdestroy(fd, &regs_chain); orki_bdestroy(fd, &regs_sep);
        orki_bdestroy(fd, &task_chain); orki_bdestroy(fd, &task_sep);
        return -2;
    }
    
    memset(W.cpu, 1, (size_t)K * N);
    memset(A.cpu, 1, (size_t)S * K);
    orki_bsync(fd, &W, RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd, &A, RKNPU_MEM_SYNC_TO_DEVICE);
    
    uint32_t rc[REGCMD_I8_N];
    for (int i = 0; i < S; i++) {
        uint32_t act_dma = (uint32_t)(A.dma + i * K);
        uint32_t out_dma = (uint32_t)(O.dma + i * 4096);
        orki_synth_i8(rc, 1, K, N, act_dma, (uint32_t)W.dma, out_dma, 1, CBUF, 0);
        orki_setrn(rc, REGCMD_I8_N,RK_CNA_CBUF_CON0, 0xb1);
        
        if (i < S - 1) {
            uint64_t next_dma = regs_chain.dma + (i + 1) * REGCMD_I8_N * 4;
            rc[216] = 0x0010 | ((next_dma & 0xffff) << 16);
            rc[217] = (0x0101 << 16) | ((next_dma >> 16) & 0xffff);
            rc[218] = 0x0014 | (0x0037 << 16);
            rc[219] = (0x0101 << 16) | (0);
        } else {
            rc[216] = 0; rc[217] = 0; rc[218] = 0x00000014; rc[219] = 0x01010000;
        }
        memcpy((char*)regs_chain.cpu + i * REGCMD_I8_N * 4, rc, sizeof(rc));
    }
    orki_bsync(fd, &regs_chain, RKNPU_MEM_SYNC_TO_DEVICE);
    
    for (int i = 0; i < S; i++) {
        uint32_t act_dma = (uint32_t)(A.dma + i * K);
        uint32_t out_dma = (uint32_t)(O.dma + i * 4096);
        orki_synth_i8(rc, 1, K, N, act_dma, (uint32_t)W.dma, out_dma, 1, CBUF, 0);
        orki_setrn(rc, REGCMD_I8_N,RK_CNA_CBUF_CON0, 0xb1);
        rc[216] = 0; rc[217] = 0; rc[218] = 0x00000014; rc[219] = 0x01010000;
        memcpy((char*)regs_sep.cpu + i * REGCMD_I8_N * 4, rc, sizeof(rc));
    }
    orki_bsync(fd, &regs_sep, RKNPU_MEM_SYNC_TO_DEVICE);
    
    struct rknpu_task *tk_chain = task_chain.cpu;
    memset(tk_chain, 0, S * sizeof(struct rknpu_task));
    for (int i = 0; i < S; i++) {
        tk_chain[i].enable_mask = 0xd;
        tk_chain[i].int_mask = 0x300;
        tk_chain[i].int_clear = 0x1ffff;
        tk_chain[i].regcfg_amount = 108;
        tk_chain[i].regcmd_addr = regs_chain.dma + i * REGCMD_I8_N * 4;
    }
    orki_bsync(fd, &task_chain, RKNPU_MEM_SYNC_TO_DEVICE);
    
    struct rknpu_task *tk_sep = task_sep.cpu;
    memset(tk_sep, 0, S * sizeof(struct rknpu_task));
    for (int i = 0; i < S; i++) {
        tk_sep[i].enable_mask = 0xd;
        tk_sep[i].int_mask = 0x300;
        tk_sep[i].int_clear = 0x1ffff;
        tk_sep[i].regcfg_amount = 108;
        tk_sep[i].regcmd_addr = regs_sep.dma + i * REGCMD_I8_N * 4;
    }
    orki_bsync(fd, &task_sep, RKNPU_MEM_SYNC_TO_DEVICE);
    
    orki_act(fd, RKNPU_ACT_RESET, 0);
    struct rknpu_submit sub; memset(&sub, 0, sizeof(sub));
    sub.flags = ork_ppflags();
    sub.task_number = S;
    sub.task_obj_addr = task_chain.obj;
    sub.core_mask = RKNPU_CORE0_MASK;
    sub.fence_fd = -1;
    sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)S};
    sub.timeout = orki_mm_timeout_ms();
    if (orki_rknpu_submit_ioctl(fd, &sub, -1)) {
        perror("Warmup failed");
    }
    orki_bsync(fd, &O, RKNPU_MEM_SYNC_FROM_DEVICE);
    
    double t_sep_start = ork_now_us();
    for (int it = 0; it < iters; it++) {
        for (int s = 0; s < S; s++) {
            struct rknpu_task *tk_dest = c->task.cpu;
            memcpy(tk_dest, &tk_sep[s], sizeof(struct rknpu_task));
            orki_bsync(fd, &c->task, RKNPU_MEM_SYNC_TO_DEVICE);
            
            struct rknpu_submit sub_s; memset(&sub_s, 0, sizeof(sub_s));
            sub_s.flags = 0x5;
            sub_s.task_number = 1;
            sub_s.task_obj_addr = c->task.obj;
            sub_s.core_mask = RKNPU_CORE0_MASK;
            sub_s.fence_fd = -1;
            sub_s.subcore_task[0] = sub_s.subcore_task[1] = sub_s.subcore_task[2] = (struct rknpu_subcore_task){0, 1};
            sub_s.timeout = 60000;
            if (orki_rknpu_submit_ioctl(fd, &sub_s, -1)) {
                perror("Separate submit failed");
                break;
            }
            orki_bsync(fd, &O, RKNPU_MEM_SYNC_FROM_DEVICE);
        }
    }
    double t_sep = ork_now_us() - t_sep_start;
    
    double t_chain_start = ork_now_us();
    for (int it = 0; it < iters; it++) {
        struct rknpu_submit sub_c; memset(&sub_c, 0, sizeof(sub_c));
        sub_c.flags = 0x5;
        sub_c.task_number = S;
        sub_c.task_obj_addr = task_chain.obj;
        sub_c.core_mask = RKNPU_CORE0_MASK;
        sub_c.fence_fd = -1;
        sub_c.subcore_task[0] = sub_c.subcore_task[1] = sub_c.subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)S};
        sub_c.timeout = 60000;
        if (orki_rknpu_submit_ioctl(fd, &sub_c, -1)) {
            perror("Chained submit failed");
            break;
        }
        orki_bsync(fd, &O, RKNPU_MEM_SYNC_FROM_DEVICE);
    }
    double t_chain = ork_now_us() - t_chain_start;
    
    double avg_sep = t_sep / iters;
    double avg_chain = t_chain / iters;
    
    printf("  %d separate submits: total time = %.0f us (avg per submit loop = %.1f us, per matmul = %.1f us)\n",
           S, t_sep, avg_sep, avg_sep / S);
    printf("  1 chained submit (x%d): total time = %.0f us (avg per submit loop = %.1f us, per matmul = %.1f us)\n",
           S, t_chain, avg_chain, avg_chain / S);
    printf("  Speedup: %.2fx\n", avg_sep / avg_chain);
    
    orki_bdestroy(fd, &W); orki_bdestroy(fd, &A); orki_bdestroy(fd, &O);
    orki_bdestroy(fd, &regs_chain); orki_bdestroy(fd, &regs_sep);
    orki_bdestroy(fd, &task_chain); orki_bdestroy(fd, &task_sep);
    return 0;
}

/* .orkpack compat token = library MAJOR version (atoi stops at the first '.'): a format-breaking change
 * requires a major bump, while minor/patch stay backward-compatible. See ork_npu.h. */

/* Max M rows a single full-K int8 submit handles at this K (mirrors orki_run()'s M>1 Bf tiling, npu.c
 * "Tier 1c-ii"). Each chain link is ONE full-K submit, so a task's M must not exceed this — else the
 * caller must split the task into M-tiles. Guards against wedging the (shared) NPU on an oversized mc. */

/* Optional fused-SiLU output stage for ONE task in a chain (the gate): that task's regcmd gets set_i8_silu
 * (int8 silu(gate) output instead of int32), and the silu LUT is streamed to SDP SRAM once before the chain.
 * Reuses run_chain_i8's proven warm/buffer/submit machinery. NULL = plain int32 chain (original behavior). */
/* Per-task op kind for a heterogeneous chain (the general OPTION-B path). SDP tasks (SILU/EWMUL) read prior
 * tasks' OUTPUT buffers by index (in0/in1), aliased -- the vendor matmul->SDP pattern. Matmul tasks emit int32
 * (MM32) or plain int8 (MM8, set_i8_out8 requant mult/shift) -- use MM8 when the output feeds an SDP task. */

/* ops     = per-task op array (the general path; NULL -> legacy task/sdp_task below).
 * task     = fused gate*silu task (set_i8_silu on the matmul output) -- KNOWN to wedge in a chain, kept for A/B.
 * sdp_task = single standalone int8 silu-SDP op reading the PREVIOUS task's output (legacy 2-task path). -1=off. */


/* ORKD coalesced SwiGLU FFN against 3 resident (is_orkd) weights — one ORKD_FFN round-trip, one on-NPU
 * chain submit (gate->silu->up->glu->down), intermediates never leave the NPU. Thin wrapper over the
 * orkd_ffn_i8 client; the caller owns the per-stage requant (mult/shift) + silu (in_scale,out_scale). */
int ork_mm_ffn_orkd(ork_npu *c, ork_w *wg, ork_w *wu, ork_w *wd,
                    int M, int K, int Nff, int Kd,
                    int gate_mult, int gate_shift, int up_mult, int up_shift, int glu_mult, int glu_shift,
                    double in_scale, double out_scale, const int8_t *A, int32_t *out){
    if (!c || !c->daemon || !wg || !wu || !wd) return -3;
    if (!wg->is_orkd || !wu->is_orkd || !wd->is_orkd) return -3;   /* weights must be daemon-resident */
    return orkd_ffn_i8(c->daemon, wg->orkd_id, wu->orkd_id, wd->orkd_id, M, K, Nff, Kd,
                       gate_mult, gate_shift, up_mult, up_shift, glu_mult, glu_shift, in_scale, out_scale, A, out);
}

/* ORKD fused attention core against 3 resident (is_orkd) weights (K^T[Kp,Nk], ones[Nk,32], V[Nk,dv]) — one
 * ORKD_ATTN round-trip, one on-NPU chain (QK^T->exp->reduce,e.V), e never leaves the NPU. Sigma[Nq*32] +
 * av[Nq*dv] returned; caller normalizes attn=av/Sigma. Thin wrapper over the orkd_attn_i8 client. */
int ork_mm_attn_orkd(ork_npu *c, ork_w *wkt, ork_w *wones, ork_w *wv,
                     int Nq, int Nk, int Kp, int dv, int r_mult, int r_shift,
                     double in_scale, double out_scale, double max_bias, const int8_t *Q, int32_t *Sigma, int32_t *av){
    if (!c || !c->daemon || !wkt || !wones || !wv) return -3;
    if (!wkt->is_orkd || !wones->is_orkd || !wv->is_orkd) return -3;   /* weights must be daemon-resident */
    return orkd_attn_i8(c->daemon, wkt->orkd_id, wones->orkd_id, wv->orkd_id, Nq, Nk, Kp, dv,
                        r_mult, r_shift, in_scale, out_scale, max_bias, Q, Sigma, av);
}
/* RR variant: nchains fused-attn chains (per-chain wkt/wv, shared wones) fanned across the daemon's cores in ONE
 * round-trip (ORKD_ATTN_RR / ork_mm_run_chains_rr_biased). Q = nchains*Nq*Kp int8 (chain-major); Sigma =
 * nchains*Nq*32, av = nchains*Nq*dv int32 (attn_c = av_c/Sigma_c). All weights must be daemon-resident. -3 if not. */
int ork_mm_attn_rr_orkd(ork_npu *c, int nchains, ork_w *const *wkt, ork_w *wones, ork_w *const *wv,
                        int Nq, int Nk, int Kp, int dv, int r_mult, int r_shift,
                        double in_scale, double out_scale, double max_bias, const int8_t *Q, int32_t *Sigma, int32_t *av){
    if (!c || !c->daemon || nchains < 1 || nchains > ORKD_ATTN_RR_MAX || !wkt || !wones || !wv) return -3;
    if (!wones->is_orkd) return -3;
    uint64_t *kt = malloc((size_t)nchains*8), *on = malloc((size_t)nchains*8), *v = malloc((size_t)nchains*8);
    if (!kt || !on || !v){ free(kt); free(on); free(v); return -3; }
    int ok = 1;
    for (int n = 0; n < nchains; n++){
        if (!wkt[n] || !wv[n] || !wkt[n]->is_orkd || !wv[n]->is_orkd){ ok = 0; break; }
        kt[n] = wkt[n]->orkd_id; on[n] = wones->orkd_id; v[n] = wv[n]->orkd_id;
    }
    int rc = ok ? orkd_attn_rr_i8(c->daemon, nchains, kt, on, v, Nq, Nk, Kp, dv,
                                  r_mult, r_shift, in_scale, out_scale, max_bias, Q, Sigma, av) : -3;
    free(kt); free(on); free(v);
    return rc;
}

/* --- ORKD whole-decode-layer core: the SINGLE home for the layer compute (lib<->orkd parity) ----------------
 * orki_layer_mm: one M=1 matmul against a resident weight — doorbell if K fits its envelope (K%512==0 && K<=4096),
 * else wide-K run_i8; warm-retry. (Moved here from orkd.c handle_layer so direct and daemon share one impl.) */
int orki_layer_mm(ork_npu *npu, ork_w *W, const int8_t *A, int K, int N, int32_t *C){
    /* DEFAULT run_i8: the whole-layer op is a parity/correctness path, not a perf path (decode-on-NPU is a
     * measured loss), and the M=1 doorbell MISSES at layer dims on RK3588 (a pre-existing doorbell-fragility
     * issue — see tasks #13/#21 — that returns garbage, rel=1.0). run_i8 is bit-exact + robust here, so it is
     * the default; ORK_LAYER_DOORBELL=1 opts back into the doorbell for doorbell-miss debugging only. */
    static int runi8 = -1; if (runi8 < 0) runi8 = getenv("ORK_LAYER_DOORBELL") ? 0 : 1;
    if (!runi8 && K % 512 == 0 && K <= 4096) {
        for (int t=0;t<4;t++){ ork_mm_task_i8 tk={W,1,(int8_t*)A,C}; ork_dyn_chain *h=ork_dyn_begin(npu,1,&tk);
            if (!h) break; if (ork_dyn_end(h)==0){ spine_civac_range(C,(size_t)N*4); return 0; } }
    }
    if (ork_mm_run_i8(npu,W,1,A,C)==0){ spine_civac_range(C,(size_t)N*4); return 0; }   /* wide-K (down proj K>4096) or doorbell-failed fallback */
    return -1; }
/* Transport-transparent whole-layer op. c->daemon set => forward as one ORKD_LAYER round-trip; else run locally
 * (the orkd daemon's handle_layer lands here on its own direct ctx). See the header for the compute contract. */

/* Chain [gate*silu -> up -> ...] in ONE submit: task[gate_task] gets a FUSED int8 SiLU output stage; its C
 * receives int8 silu(gate) (M*N bytes). Other tasks are plain int32 matmuls. lut/params as ork_mm_run_i8_silu
 * (build via ork_mm_silu_build_lut). Single M-tile per task for now (M<=chain mcap). 0/ok,-1 wedge,-2 dims. */

/* OPTION B: chain [... -> gate matmul(int8-out) -> silu-SDP -> ...] where task[sdp_task] is a STANDALONE int8
 * silu-SDP op reading task[sdp_task-1]'s (gate) output via aliased buffers (the vendor's matmul->SDP pattern),
 * NOT a fused matmul output stage. The gate task (sdp_task-1) gets orki_set_i8_out8 (int8 output, requant
 * gate_mult/gate_shift). The silu LUT for (in_scale,out_scale) is built internally (same as ork_npu_silu_i8).
 * tasks[sdp_task].C receives int8 silu (M*N bytes). Single M-tile per task. 0/ok,-1 wedge,-2 dims,-3 SoC. */

/* GENERAL heterogeneous FFN chain: per-task ops[] (OP_MM32/OP_MM8/OP_SILU/OP_EWMUL); SDP tasks read prior
 * outputs by index (ops[i].in0/in1), aliased. The silu LUT for (in_scale,out_scale) is built internally.
 * Chains e.g. [gate(MM8) -> silu(SDP,in0=gate) -> up(MM8) -> glu(EWMUL,in0=silu,in1=up) -> down(MM32)] in ONE
 * submit. tasks[i].C receives that op's output (int8 for MM8/SDP, int32 for MM32). Single M-tile per task.
 * 0/ok,-1 wedge,-2 dims,-3 SoC. */
/* Same heterogeneous chain, but the SDP activation task (kind 2) applies EXP instead of SiLU — the HW-chained
 * softmax numerator: [QK^T(1) -> exp(2, in0=0) -> reduce(0, reads exp)] as ONE submit, e kept on-chip. Scores
 * must be <=0 (post-max domain) so exp in (0,1] fits int8. Only differs from _ffn by the curve fn. 0/ok,-1,-2,-3. */
/* Fused-exp chain, but the exp bakes in a SCALAR max-subtract: e = exp((score-max_bias)*in_scale)/out_scale.
 * max_bias=0 => plain exp (scores must already be <=0). A max_bias >= every real score keeps all args <=0 so the
 * int8 exp never saturates, and the constant cancels in the softmax normalize av/Sigma (registry: exp_biased_probe,
 * scalar global-max-biased exp_i8). This is what makes the fused attention core correct on REAL (positive) QK^T
 * scores without a live per-query max. See ork_npu_exp_i8_biased / orki_silu_build_curve_biased.
 * (Defined before the plain wrapper below: the standalone Makefile build does not pull ork_npu.h into this TU.) */


/* ============ CONCURRENT ROUND-ROBIN CHAIN DISPATCH (ork_mm_run_chains_rr) — increment 2 ============
 * Prefill throughput: N independent fused exp-softmax-numerator chains dispatched across ALL NPU cores at once
 * (chain -> core via atomic work-stealing), each on its OWN per-core scratch (chain_rc/tk/lrc/lsc[core]) so there
 * is NO cross-core DRAM sharing. Targets ~3x over single-core for a deep prefill queue. Shared mode state
 * (ork_npu_enter) + the domain are established ONCE here single-threaded; workers pass force_core>=0 so
 * orki_run_chain_i8_impl skips the re-enter (racing the mode reset would wedge a sibling core). Chains are homogeneous:
 * same op graph (ops[]) + scales + domain; each carries its own S-task array (chains[i]). */
struct chainrr_w { ork_npu *c; int core; int nchains; const ork_mm_task_i8 *const *chains; const int *S; const struct chain_silu_spec *ss; int *ctr; int rc; };
static void *chainrr_worker(void *vp){
    struct chainrr_w *a=vp; orki_pin_big_core(a->core); a->rc=0; int k;
    while((k=__atomic_fetch_add(a->ctr,1,__ATOMIC_SEQ_CST))<a->nchains){
        int r=orki_run_chain_i8_impl(a->c, a->S[k], a->chains[k], a->ss, a->core);   /* force_core=this core; skips ork_npu_enter */
        if(r) a->rc=r;
    }
    return NULL;
}
/* Run `nchains` fused exp chains round-robin across the cores (concurrent). chains[i] = that chain's S[i]-task
 * array; ops = the shared op graph; (in_scale,out_scale) the shared exp requant. Returns 0/ok, <0 err (first
 * failing chain's code). PRECONDITION: cores must be WARM (a prior matmul on each) — a cold core's first submit
 * wedges; a chain-only caller should warm via a multi-core matmul first (see chainrr_conc_probe). Local NPU only. */
int ork_mm_run_chains_rr(ork_npu *c, int nchains, const ork_mm_task_i8 *const *chains, const int *S,
                         const ork_chain_op *ops, double in_scale, double out_scale){
    if(!c || nchains<1 || !chains || !S || !ops) return -2;
    if(c->daemon) return -3;                     /* local NPU only (the daemon owns its own scheduler) */
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(orki_silu_calibrate_idx(c)) return -1;
    for(int i=0;i<nchains;i++){ if(S[i]<1 || !chains[i]) return -2; }
    static int16_t lut[1030]; static double c_is=-1, c_os=-1;   /* stable contents -> pointer-keyed per-core LUT cache stays valid */
    if(in_scale!=c_is || out_scale!=c_os){ orki_silu_build_curve(c, orki_exp_f, in_scale, out_scale, lut); c_is=in_scale; c_os=out_scale; }
    struct chain_silu_spec ss = { ops, -1, -1, 0x4000, 14, 0, 0, 0, ORK_SILU_IDXOFF, ORK_SILU_C4064, ORK_SILU_C4068, lut, 1030 };
    int nc=c->soc->cores; if(nc>ORK_MAXCORE)nc=ORK_MAXCORE; if(nc>nchains)nc=nchains; if(nc<1)nc=1;
    /* establish SHARED state ONCE single-threaded (workers skip via force_core>=0): domain + int8-chain mode */
    if(chains[0][0].w && (chains[0][0].w->domain!=c->dom_active || (chains[0][0].w->domain!=0 && !c->dom_save)))
        orki_dom_activate(c, chains[0][0].w->domain);
    ork_npu_enter(c, 3 /* DT_I8_CHAIN */, XP_CHAIN_NT, OCK_FUSED);
    orki_npu_pool_ensure(c);
    struct chainrr_w w[ORK_MAXCORE]; int ctr=0;
    for(int i=0;i<nc;i++) w[i]=(struct chainrr_w){c,i,nchains,chains,S,&ss,&ctr,0};
    pthread_mutex_lock(&c->pmu);
    c->pjob=w; c->pjob_nc=nc; c->pjob_fn=chainrr_worker; c->pjob_stride=sizeof(struct chainrr_w);
    c->pdone=0; c->pgen++; pthread_cond_broadcast(&c->pgo);
    pthread_mutex_unlock(&c->pmu);
    chainrr_worker(&w[0]);                        /* core 0 on the calling thread */
    pthread_mutex_lock(&c->pmu); while(c->pdone<nc-1) pthread_cond_wait(&c->pdn,&c->pmu); pthread_mutex_unlock(&c->pmu);
    int rc=0; for(int i=0;i<nc;i++) if(w[i].rc) rc=w[i].rc;
    return rc;
}
/* BIASED round-robin: same concurrent dispatch as ork_mm_run_chains_rr, but the fused exp bakes in a scalar
 * max-subtract (e=exp((score-max_bias)*in_scale)/out_scale) so the chains are correct on REAL (positive) scores
 * without a live per-query max (registry: scalar global-max-biased exp; bias cancels in av/Sigma). This lets N
 * independent attention cores' [QK^T->exp->reduce,e.V] chains fan across the NPU cores from a single dispatch.
 * The exp LUT contents change IN PLACE at one static address, so orki_run_chain_i8_impl's POINTER-keyed per-core
 * device-LUT cache (chain_lut_p[cc]) would go stale. We invalidate all cores ONCE here, single-threaded, BEFORE
 * the workers start — each worker's orki_run_chain_i8_impl then rebuilds+reuploads THIS core's per-core SDP-SRAM LUT
 * on its first chain (the "LUT-cache-update op at the front of the chain, injected per core"). Per-core buffers
 * make that reload concurrency-safe. Returns 0/ok, <0 err. Local NPU only (the daemon calls it on its own ctx). */
int ork_mm_run_chains_rr_biased(ork_npu *c, int nchains, const ork_mm_task_i8 *const *chains, const int *S,
                                const ork_chain_op *ops, double in_scale, double out_scale, double max_bias){
    if(!c || nchains<1 || !chains || !S || !ops) return -2;
    if(c->daemon) return -3;                     /* local NPU only (the daemon owns its own scheduler) */
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(orki_silu_calibrate_idx(c)) return -1;
    for(int i=0;i<nchains;i++){ if(S[i]<1 || !chains[i]) return -2; }
    static int16_t lut[1030]; static double c_is=-1, c_os=-1, c_bias=-1e300;
    if(in_scale!=c_is || out_scale!=c_os || max_bias!=c_bias){
        orki_silu_build_curve_biased(c, orki_exp_f, in_scale, out_scale, max_bias, lut); c_is=in_scale; c_os=out_scale; c_bias=max_bias;
        for(int i=0;i<ORK_MAXCORE;i++) c->chain_lut_p[i]=NULL;   /* in-place LUT rebuild -> force each core to reload */
    }
    struct chain_silu_spec ss = { ops, -1, -1, 0x4000, 14, 0, 0, 0, ORK_SILU_IDXOFF, ORK_SILU_C4064, ORK_SILU_C4068, lut, 1030 };
    int nc=c->soc->cores; if(nc>ORK_MAXCORE)nc=ORK_MAXCORE; if(nc>nchains)nc=nchains; if(nc<1)nc=1;
    if(chains[0][0].w && (chains[0][0].w->domain!=c->dom_active || (chains[0][0].w->domain!=0 && !c->dom_save)))
        orki_dom_activate(c, chains[0][0].w->domain);
    ork_npu_enter(c, 3 /* DT_I8_CHAIN */, XP_CHAIN_NT, OCK_FUSED);
    orki_npu_pool_ensure(c);
    struct chainrr_w w[ORK_MAXCORE]; int ctr=0;
    for(int i=0;i<nc;i++) w[i]=(struct chainrr_w){c,i,nchains,chains,S,&ss,&ctr,0};
    pthread_mutex_lock(&c->pmu);
    c->pjob=w; c->pjob_nc=nc; c->pjob_fn=chainrr_worker; c->pjob_stride=sizeof(struct chainrr_w);
    c->pdone=0; c->pgen++; pthread_cond_broadcast(&c->pgo);
    pthread_mutex_unlock(&c->pmu);
    chainrr_worker(&w[0]);                        /* core 0 on the calling thread */
    pthread_mutex_lock(&c->pmu); while(c->pdone<nc-1) pthread_cond_wait(&c->pdn,&c->pmu); pthread_mutex_unlock(&c->pmu);
    int rc=0; for(int i=0;i<nc;i++) if(w[i].rc) rc=w[i].rc;
    return rc;
}

/* ================= DYNAMIC STEERED SUBMISSION API (validated by tools/steer_probe + doorbell_id_probe) =====
 * A run_chain_i8 chain, but submitted NONBLOCK so the host can (a) watch per-op progress via each op's output
 * doorbell (dc civac poll), (b) HALT it mid-flight to free the NPU early (write 0x0014=0 into a future op's
 * live regcmd descriptor — the sequencer reads it from DRAM at exec-time), (c) later, redirect the next-pointer
 * for runtime routing. v1 constraints: M=1 per task, single-slice conforming K (K%512==0, K<=4096), and A/C
 * resident in ork_dma_alloc buffers (so we hold DMA addrs + poll outputs coherently). One program per task
 * (P==S). Steering must lead the sequencer by ~1-2 ops (time it off ork_dyn_progress). */
/* Graceful SIGTERM/SIGINT for the doorbell. The async poll (ork_dyn_end) would otherwise spin uninterruptibly:
 * a `kill -TERM` during an NPU submit-wait was IGNORED -> orphaned process, forced board reboot (and a kill -9
 * mid-submit risks an IOMMU/NPU wedge). A chained handler sets a flag; the poll breaks on it and DRAINS
 * (bsync + writeback + free) before the process terminates. If the signal arrives while NOT in a doorbell
 * (idle), the original disposition fires immediately — the handler never swallows SIGTERM. */
volatile sig_atomic_t orki_ork_term = 0, orki_in_doorbell = 0;
struct sigaction orki_prev_sig[2];   /* [0]=SIGTERM, [1]=SIGINT */
static void ork_term_handler(int sig) {
    orki_ork_term = 1;
    if (!orki_in_doorbell) { int k = (sig == SIGINT) ? 1 : 0; sigaction(sig, &orki_prev_sig[k], NULL); raise(sig); }   /* idle: honor original disposition now */
    /* in a doorbell poll: just flag — the poll breaks, drains, then re-raises with the original disposition */
}
void ork_install_term(void) {
    static int done = 0; if (done) return; done = 1;
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_handler = ork_term_handler; sigemptyset(&sa.sa_mask);
    sigaction(SIGTERM, &sa, &orki_prev_sig[0]); sigaction(SIGINT, &sa, &orki_prev_sig[1]);
}
/* int8 M-tile row cap for a K-reduction width Kred: the 0x1040 schedule holds at most mg_max*64 rows in one
 * regcmd (a bigger tile spills the K-partition and miscomputes). = 64 @ K<=4096, larger as K shrinks. Used to
 * M-tile M>64 into chained programs and as the K-split per-slice defensive cap. */
int orki_mtile_cap(int Kred){ double scale=(double)Kred/512.0; int base=(int)(177.0-15.0*(scale-1.0)), slope=(int)(15.0*scale);
    int mg = base >= 0x1b ? (base-0x1b)/slope + 1 : 0; int cap = mg * 64; return cap < 1 ? 1 : cap; }
/* Per-row completion: a task is done only when EVERY row's last column has been overwritten. The M-tile
 * scheduler does NOT write the global last element (row M-1, col N-1) strictly last for M>1 — so a single-
 * word doorbell RACES (end() returns before earlier rows land; M=1 is safe since within a row last-col IS
 * last). Poll all M rows' last cols (M<=64, cheap). Sentinel 0x7fffffff = INT_MAX / fp32 NaN — no valid
 * matmul output equals it, so this is precision-agnostic (int8->int32, fp16->fp32). */
int orki_bch_db_cells(ork_npu *c,int i,int c0,int c1,int Wb,int N,int NG,int M,int H,int Wmax,int32_t *C,int mode,int only_tk);   /* #54 fwd: BCHAIN tile seed/gate/verify/de-tile (defined below) */
/* MULTI-CORE NONBLOCK variant: partition the S tasks across `nc` NPU cores (nc<=0 => all), build each core's
 * sub-chain into its per-core buffers (c->mrc[i]/mtk[i]/maf[i], like mcworker), and submit each NONBLOCK on
 * core_mask=1<<i. The cores run CONCURRENTLY — core-partitioned submits are the proven-safe pattern
 * (run_stream_i8/mcworker already do it); NONBLOCK just lets the host poll doorbells instead of threads
 * blocking. Beats the std::thread+blocking-stream decode overlap: thread-free, 3-core, doorbell rendezvous.
 * ork_dyn_progress/end poll every task's C[N-1] (works across cores); halt/append are single-core only.
 * v1: M=1, DT_I8, K%512==0 && K<=4096. MULTI-DOMAIN-SAFE: outputs go to the per-core IN-DOMAIN scratch
 * c->mcc[i] (dom_activate-swapped, so always in the submit's domain), then end copies them back to the
 * caller's C (a host memcpy) — so the caller's C need NOT be resident or in the submit's domain. A likewise
 * staged into the per-core AF (zero-copy A is M=1-wrong). All S tasks must share one domain (a MoE node's
 * experts do: layer-based residence); begin_mc submits + allocs its scratch in tasks[0]'s domain. */
/* int4 (W4A4) multi-core NONBLOCK doorbell — the DT_I4 sibling of ork_dyn_begin_mc. int4 is structurally
 * different from int8/fp16 at the hardware level: the datapath writes an int16 (2-byte) accumulator (widened
 * to int32 on the host, esz=2) and its HW chain is M=1 only — so it CANNOT share the 4-byte-C body. This keeps
 * that body byte-identical and specialises the int4 divergences: tile_i4_Aslice host-A staging (0.5 B/elem),
 * synth_i4 / REGCMD_I4_N stride / regcfg_amount=116, an int16 per-core output scratch (ALWAYS copy-back — the
 * NPU never writes the caller's int32 C in place), and a FULL-SURFACE int16 sentinel seed that doubles as the
 * clean-before-write the fresh/reused scratch needs (int4's int16 write-order over N is not last-col-last, so
 * both the seed and the completion poll cover the whole row). Proven bit-exact single-core by ork_dyn_i4_probe;
 * this is the productionised multi-core form the scheduler dispatches. Host (malloc) A only, as for int8/fp16.
 * end() drains via the esz-aware ork_dyn_done_i and widens int16->int32 into the caller's C. */
ork_dyn_chain *ork_dyn_begin_mc_i4(ork_npu *c, int S, const ork_mm_task_i8 *tasks, int nc) {
    if (nc < 1 || nc > c->soc->cores) nc = c->soc->cores; if (nc > S) nc = S;
    for (int i = 0; i < S; i++) { ork_w *w = tasks[i].w;
        if (!w || w->dtype != DT_I4 || tasks[i].M != 1 || w->Sk > 16) return NULL;   /* int4 HW chain: M=1; A1: Sn>1 N-tiled; A2: Sk>1 K-split (int16 partials summed in end) — Sk bounded (per-row A-slice array) */
        if (w->domain != tasks[0].w->domain) return NULL; }   /* one submit => one domain */
    if (tasks[0].w->domain != c->dom_active || (tasks[0].w->domain && !c->dom_save)) orki_dom_activate(c, tasks[0].w->domain);
    ork_npu_enter(c, 4 /*DT_I4_CHAIN*/, XP_I4CHAIN, OCK_HW);
    if (orki_mc_ensure(c, nc)) return NULL;
    int fd = c->fd;
    ork_dyn_chain *h = calloc(1, sizeof *h); if (!h) return NULL;
    h->c = c; h->S = S; h->P = S; h->N = tasks[0].w->N; h->dom = tasks[0].w->domain; h->reserve = S; h->mc = 1; h->esz = 2;
    unsigned dom = tasks[0].w->domain;
    /* int4 ALWAYS copy-back: per-core in-domain int16 scratch (M=1 => N int16/op), widened in end() */
    for (int i = 0; i < nc; i++) { int lo=(int)((long)i*S/nc), hi=(int)((long)(i+1)*S/nc), P=hi-lo; if (P<1) continue;
        size_t osz = 0; for (int p = lo; p < hi; p++) osz += (size_t)tasks[p].w->N * tasks[p].w->Sk * 2;   /* Sk int16 partial blocks/row (A2 K-split; Sk==1 => N int16) */
        if (c->mccsz[i] < osz) { orki_bdestroy(fd, &c->mcc[i]); c->mcc[i] = orki_bcreate(fd, osz, 0x403, c->dom_active);
            if (!c->mcc[i].cpu) { free(h); return NULL; } c->mccsz[i] = osz; c->mwarm[i] = 0; } }
    uint32_t rc[REGCMD_I4_N];
    int NMAX = c->soc->nmax, KS = ORK_I4_KS;
    struct rknpu_submit subs[ORK_MAXCORE]; int Pc[ORK_MAXCORE]; memset(Pc, 0, sizeof Pc);
    for (int i = 0; i < nc; i++) {
        int lo = (int)((long)i * S / nc), hi = (int)((long)(i+1) * S / nc), P = hi - lo;
        if (P < 1) { Pc[i] = 0; continue; }
        /* PROGRAM count decoupled from row-task count: a row emits Sn*Sk programs (N-slices x K-slices). */
        int Pcore = 0; for (int p = lo; p < hi; p++) Pcore += tasks[p].w->Sn * tasks[p].w->Sk;
        Pc[i] = Pcore;
        if ((size_t)Pcore * REGCMD_I4_N * 4 > c->mrc[i].size || (size_t)Pcore * sizeof(struct rknpu_task) > c->mtk[i].size) { free(h); return NULL; }
        struct buf *RC = &c->mrc[i], *AF = &c->maf[i], *CC = &c->mcc[i]; struct rknpu_task *tk = (struct rknpu_task*)c->mtk[i].cpu;
        size_t astage = 0, coff = 0; int pp = 0;
        for (int p = lo; p < hi; p++) {
            const ork_mm_task_i8 *t = &tasks[p]; ork_w *w = t->w; int K = w->K, N = w->N, Sn = w->Sn, Sk = w->Sk;
            /* A2 K-SPLIT: stage this row's Sk activation K-slices (each Kp nibbles, 0.5 B/elem), shared by the
             * row's N-slices. Sk<=16 (guarded above); each slice reads A[:, ks*KS : ks*KS+Kp]. */
            uint32_t aslice[16];
            for (int ks = 0; ks < Sk; ks++) { int k0 = ks * KS, Kp = (K - k0 < KS) ? (K - k0) : KS; size_t asz = (size_t)Kp / 2;
                if (astage + asz > AF->size) { free(h); return NULL; }
                orki_tile_i4_Aslice((uint8_t*)AF->cpu + astage, (const int8_t*)t->A, k0, Kp);
                aslice[ks] = (uint32_t)(AF->dma + astage); astage += asz; }
            /* A1 N-tile x A2 K-split: one program per (N-slice ns, K-slice ks). The row's output is Sk blocks
             * of [N] int16 (block ks = the K-slice-ks partial; column-slices write their [Nc] within it); the
             * drain SUMS the Sk blocks per column -> int32 C (oSk). Sk==1 => one block = A1's plain widen. */
            for (int ns = 0; ns < Sn; ns++) {
                int n0 = ns * NMAX, Nc = (N - n0 < NMAX) ? (N - n0) : NMAX;
                for (int ks = 0; ks < Sk; ks++) {
                    int k0 = ks * KS, Kp = (K - k0 < KS) ? (K - k0) : KS;
                    uint32_t cdma = (uint32_t)(CC->dma + coff + (size_t)ks * N * 2 + (size_t)n0 * 2);   /* block ks, columns [n0,n0+Nc) */
                    struct buf *WT = &w->Bb[(size_t)ns * Sk + ks];
                    uint32_t bdma = (uint32_t)WT->dma;                                                   /* weight N-slice ns, K-slice ks */
                    if (getenv("ORK_I4_DIAG")) { unsigned char *bc=(unsigned char*)WT->cpu;
                        fprintf(stderr,"[i4diag] mc_i4 wdom=%d dom_active=%d imported=%d ns=%d ks=%d Kp=%d Nc=%d | bdma=0x%llx obj=0x%llx size=%zu cpu=%p bytes[0..7]=",
                            w->domain, c->dom_active, (w->own_bufs&&w->n_own_bufs>0)||w->own_buf_valid, ns, ks, Kp, Nc,
                            (unsigned long long)WT->dma, (unsigned long long)WT->obj, WT->size, WT->cpu);
                        if(bc) for(int z=0;z<8;z++) fprintf(stderr,"%02x ",bc[z]); else fprintf(stderr,"(null)");
                        fprintf(stderr,"| aslice=0x%x cdma=0x%x\n", aslice[ks], cdma); fflush(stderr); }
                    memset(rc, 0, sizeof rc);
                    orki_synth_i4(rc, 1, Kp, Nc, aslice[ks], bdma, cdma);
                    if (orki_validate_regcmd("ork_dyn_mc_i4", c, rc, REGCMD_I4_N, w, NULL, 0)) { free(h); return NULL; }
                    if (pp < Pcore - 1) { uint64_t nx = RC->dma + (size_t)(pp+1) * REGCMD_I4_N * 4;
                        rc[216] = 0x0010 | ((nx & 0xffff) << 16); rc[217] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
                        rc[218] = 0x0014 | (0x0037u << 16);       rc[219] = (0x0101 << 16); }
                    memcpy((char*)RC->cpu + (size_t)pp * REGCMD_I4_N * 4, rc, REGCMD_I4_N * 4);
                    struct rknpu_task tt; memset(&tt, 0, sizeof tt); tt.enable_mask = 0xd; tt.int_mask = 0x300;
                    tt.int_clear = 0x1ffff; tt.regcfg_amount = 116; tt.regcmd_addr = RC->dma + (size_t)pp * REGCMD_I4_N * 4;   /* int4 = 116 regs */
                    tk[pp] = tt; pp++;
                }
            }
            int gi = p;
            h->outbuf[gi] = CC; h->outptr[gi] = (int32_t*)((char*)CC->cpu + coff); h->dst[gi] = (int32_t*)t->C;
            h->nout[gi] = Sk * N; h->oM[gi] = 1; h->oSk[gi] = Sk;   /* Sk int16 partial blocks of [N]; end() sums -> int32 C */
            coff += (size_t)Sk * N * 2;
        }
        memset(&subs[i], 0, sizeof subs[i]);
        subs[i].flags = ork_ppflags() | 0x2u; subs[i].task_number = pp; subs[i].task_obj_addr = c->mtk[i].obj;
        subs[i].core_mask = 1u << i; subs[i].fence_fd = -1;
        subs[i].subcore_task[0] = subs[i].subcore_task[1] = subs[i].subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)pp};
    }
    /* seed the FULL int16 output surface (clean-before-write for fresh/reused scratch; = the probe's fix) */
    for (int x = 0; x < S; x++) { int N = h->nout[x]; volatile int16_t *o = (volatile int16_t*)h->outptr[x];
        for (int col = 0; col < N; col++){ o[col] = ORK_DYN_SENT16; __asm__ volatile("dc cvac,%0"::"r"(&o[col]):"memory"); } }
    __asm__ volatile("dsb ish":::"memory");
    for (int i = 0; i < nc; i++) if (Pc[i]) {
        orki_bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd, &c->mrc[i], RKNPU_MEM_SYNC_TO_DEVICE);
        orki_bsync(fd, &c->mtk[i], RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
        subs[i].timeout = orki_i4_submit_tmo_ms(); orki_rknpu_submit_ioctl(fd, &subs[i], dom); }   /* #54 bounded (int4 doorbell): a dropped submit must be PAST its timeout by the poll window so ork_dyn_end's recover resubmit reaps it via rknpu_job_timeout_clean. With the 8s mm_timeout_ms a dom-0 drop's stuck job stayed unreaped -> iommu_domain_refcount>0 -> the switch to dom 1 TIMED OUT at scale (the 35B wedge; the small probe never dropped). */
    for (int i = 0; i < nc; i++) c->mwarm[i] = 1;
    /* TASK #4: stash context so ork_dyn_end recovers a dropped int4 round (same ~1/2000 doorbell-drop; the
     * esz==2 branch of orki_mc_recover_resubmit re-seeds the full int16 surface). */
    h->mc_nc = nc; h->mc_dt = DT_I4; h->mc_dom = dom;
    for (int i = 0; i < nc && i < ORK_MAXCORE; i++) { h->mc_subs[i] = subs[i]; h->mc_Pc[i] = Pc[i]; }
    return h;   /* async: end() drains via the esz==2 full-surface int16 poll, then widens int16->int32 into C */
}

/* ============ B: GROUPED int4 (W4A4 per-K-group scales) on the NONBLOCK doorbell ============
 * ork_mm_run_i4_grouped's doorbell path. Row-decomposed (M=1 tasks across cores); each row emits Sn*Sk programs
 * (K-slice = gsize G, Sk = K/G groups). Output = Sk int16 partial blocks of [N] per row. Unlike A2's int-sum,
 * the drain (ork_dyn_grouped_end) FLOAT scale-accumulates: C[m][n] = sum_g aS[m*Sk+g]*bS[g*N+n]*partial_g[n]
 * (matches the grouped drain). NULL if ineligible (chain/scratch too big) -> caller refuses (ORK_RC_WEDGE_PRONE; the blocking i4_mcworker_g path is removed #45). */
ork_dyn_chain *ork_dyn_begin_mc_i4_grouped(ork_npu *c, int M, ork_w *w, const int8_t *A,
                                                  const float *aScale, const float *bScale, float *Cf, int nc) {
    if (!w || w->dtype != DT_I4 || !w->gsize || M < 1 || M > 1024) return NULL;
    int G = w->gsize, K = w->K, N = w->N, Sn = w->Sn, Sk = w->Sk;   /* grouped: Sk = K/G groups */
    (void)K;
    if (Sk > 256) return NULL;                                       /* bounds the per-row aslice[]/program count */
    if (nc < 1 || nc > c->soc->cores) nc = c->soc->cores; if (nc > M) nc = M;
    if (getenv("ORK_GRP_DEBUG")) { fprintf(stderr, "[grp] M=%d K=%d N=%d G=%d Sk=%d Sn=%d nc=%d progs/core~%d\n",
        M, w->K, N, G, Sk, Sn, nc, (M+nc-1)/nc * Sn * Sk); fflush(stderr); }
    if (w->domain != c->dom_active || (w->domain && !c->dom_save)) orki_dom_activate(c, w->domain);
    ork_npu_enter(c, 4 /*DT_I4_CHAIN*/, XP_I4CHAIN, OCK_HW);
    if (orki_mc_ensure(c, nc)) return NULL;
    int fd = c->fd, NMAX = c->soc->nmax;
    ork_dyn_chain *h = calloc(1, sizeof *h); if (!h) return NULL;
    h->c = c; h->S = M; h->P = M; h->N = N; h->dom = w->domain; h->reserve = M; h->mc = 1; h->esz = 2;
    h->i4g = 1; h->i4g_aS = aScale; h->i4g_bS = bScale; h->i4g_Cf = Cf; h->i4g_N = N; h->i4g_Sk = Sk;
    unsigned dom = w->domain;
    for (int i = 0; i < nc; i++) { int lo=(int)((long)i*M/nc), hi=(int)((long)(i+1)*M/nc), P=hi-lo; if (P<1) continue;
        size_t osz = (size_t)P * Sk * N * 2;                         /* rows-on-core x Sk int16 blocks of [N] */
        if (c->mccsz[i] < osz) { orki_bdestroy(fd, &c->mcc[i]); c->mcc[i] = orki_bcreate(fd, osz, 0x403, c->dom_active);
            if (!c->mcc[i].cpu) { free(h); return NULL; } c->mccsz[i] = osz; c->mwarm[i] = 0; } }
    uint32_t rc[REGCMD_I4_N];
    struct rknpu_submit subs[ORK_MAXCORE]; int Pc[ORK_MAXCORE]; memset(Pc, 0, sizeof Pc);
    for (int i = 0; i < nc; i++) { int lo=(int)((long)i*M/nc), hi=(int)((long)(i+1)*M/nc), P=hi-lo; if (P<1) { Pc[i]=0; continue; }
        int Pcore = P * Sn * Sk; Pc[i] = Pcore;
        if ((size_t)Pcore * REGCMD_I4_N * 4 > c->mrc[i].size || (size_t)Pcore * sizeof(struct rknpu_task) > c->mtk[i].size) { free(h); return NULL; }
        struct buf *RC = &c->mrc[i], *AF = &c->maf[i], *CC = &c->mcc[i]; struct rknpu_task *tk = (struct rknpu_task*)c->mtk[i].cpu;
        size_t astage = 0, coff = 0; int pp = 0;
        for (int m = lo; m < hi; m++) { const int8_t *Arow = A + (size_t)m * w->K;
            uint32_t aslice[256];                                    /* this row's Sk group A-slices (each G nibbles) */
            for (int g = 0; g < Sk; g++) { size_t asz = (size_t)G / 2;
                if (astage + asz > AF->size) { free(h); return NULL; }
                orki_tile_i4_Aslice((uint8_t*)AF->cpu + astage, Arow, g * G, G);
                aslice[g] = (uint32_t)(AF->dma + astage); astage += asz; }
            for (int ns = 0; ns < Sn; ns++) { int n0 = ns * NMAX, Nc = (N - n0 < NMAX) ? (N - n0) : NMAX;
                for (int g = 0; g < Sk; g++) {
                    uint32_t cdma = (uint32_t)(CC->dma + coff + (size_t)g * N * 2 + (size_t)n0 * 2);   /* block g, cols [n0,n0+Nc) */
                    uint32_t bdma = (uint32_t)w->Bb[(size_t)ns * Sk + g].dma;
                    memset(rc, 0, sizeof rc);
                    orki_synth_i4(rc, 1, G, Nc, aslice[g], bdma, cdma);
                    if (orki_validate_regcmd("ork_dyn_mc_i4g", c, rc, REGCMD_I4_N, w, NULL, 0)) { free(h); return NULL; }
                    if (pp < Pcore - 1) { uint64_t nx = RC->dma + (size_t)(pp+1) * REGCMD_I4_N * 4;
                        rc[216] = 0x0010 | ((nx & 0xffff) << 16); rc[217] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
                        rc[218] = 0x0014 | (0x0037u << 16);       rc[219] = (0x0101 << 16); }
                    memcpy((char*)RC->cpu + (size_t)pp * REGCMD_I4_N * 4, rc, REGCMD_I4_N * 4);
                    struct rknpu_task tt; memset(&tt, 0, sizeof tt); tt.enable_mask = 0xd; tt.int_mask = 0x300;
                    tt.int_clear = 0x1ffff; tt.regcfg_amount = 116; tt.regcmd_addr = RC->dma + (size_t)pp * REGCMD_I4_N * 4;
                    tk[pp] = tt; pp++;
                } }
            int gi = m; h->outbuf[gi] = CC; h->outptr[gi] = (int32_t*)((char*)CC->cpu + coff); h->dst[gi] = NULL;
            h->nout[gi] = Sk * N; h->oM[gi] = 1;                     /* Sk int16 partial blocks; grouped_end float-accumulates */
            coff += (size_t)Sk * N * 2;
        }
        memset(&subs[i], 0, sizeof subs[i]);
        subs[i].flags = ork_ppflags() | 0x2u; subs[i].task_number = pp; subs[i].task_obj_addr = c->mtk[i].obj; subs[i].core_mask = 1u << i; subs[i].fence_fd = -1;
        subs[i].subcore_task[0] = subs[i].subcore_task[1] = subs[i].subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)pp};
    }
    for (int x = 0; x < M; x++) { int no = h->nout[x]; volatile int16_t *o = (volatile int16_t*)h->outptr[x];
        for (int e = 0; e < no; e++) { o[e] = ORK_DYN_SENT16; __asm__ volatile("dc cvac,%0"::"r"(&o[e]):"memory"); } }
    __asm__ volatile("dsb ish":::"memory");
    for (int i = 0; i < nc; i++) if (Pc[i]) {
        orki_bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd, &c->mrc[i], RKNPU_MEM_SYNC_TO_DEVICE);
        orki_bsync(fd, &c->mtk[i], RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
        subs[i].timeout = orki_i4_submit_tmo_ms(); orki_rknpu_submit_ioctl(fd, &subs[i], dom); }   /* #54 bounded (int4 doorbell): a dropped submit must be PAST its timeout by the poll window so ork_dyn_end's recover resubmit reaps it via rknpu_job_timeout_clean. With the 8s mm_timeout_ms a dom-0 drop's stuck job stayed unreaped -> iommu_domain_refcount>0 -> the switch to dom 1 TIMED OUT at scale (the 35B wedge; the small probe never dropped). */
    for (int i = 0; i < nc; i++) c->mwarm[i] = 1;
    ork_install_term();
    return h;
}
/* Drain the grouped-int4 doorbell: poll all rows' Sk*N int16 partials, then FLOAT scale-accumulate into C[M,N]. */
void orki_mc_recover_resubmit(ork_dyn_chain *h);   /* shared doorbell recover (defined below); grouped drain rides it */
int ork_dyn_grouped_end(ork_dyn_chain *h) {
    if (!h || !h->i4g) return -2;
    ork_npu *c = h->c; int fd = c->fd, rc = 0;
    /* Drain on the SHARED doorbell recover loop (mirrors ork_dyn_end@12140): poll all rows; on a dropped round
     * (mc int4 output never landed) orki_mc_recover_resubmit (RESET + re-seed SENT16 via its esz==2 branch + resubmit
     * each core) and re-poll, up to recov_max; auto-dump only a TRUE stall (recover exhausted). The grouped begin
     * already stashed mc_nc/mc_dt=DT_I4/mc_dom/mc_subs/mc_Pc, so this self-heals exactly like the int8 mc path. */
    orki_in_doorbell = 1;
    int recov_max = (h->mc_nc > 0 && h->mc_dt == DT_I4) ? 6 : 0;
    int landed = 0, edone[1024];
    for (int recov = 0; ; recov++) {
        double t0 = ork_now_us();
        for (int i = 0; i < h->S && i < 1024; i++) edone[i] = 0;
        double miss_to = (recov < recov_max) ? 300000.0 : 3e6;   /* fast miss-detect while retries remain, else full completion wait */
        for (;;) { int n = 0; for (int x = 0; x < h->S; x++) { if (!edone[x]) edone[x] = ork_dyn_done_i(h, x); n += edone[x]; }
            if (n >= h->S) { landed = 1; break; }
            if (orki_ork_term) break;
            double el = ork_now_us() - t0; if (el > miss_to) break;
            if (el > 1000.0) { struct timespec ts = {0, 50000}; nanosleep(&ts, NULL); } }
        if (landed || orki_ork_term) break;
        if (recov < recov_max) { if (getenv("ORK_MC_DIAG")) fprintf(stderr, "[MC-RECOVER grp] int4 grouped round never landed (attempt %d) — reset+resubmit\n", recov);
            h->c->dom_dirty = 1;   /* #54: int4 drop -> reap-at-boundary (see ork_dom_flush_if_dirty) */
            orki_mc_recover_resubmit(h); continue; }
        break;
    }
    orki_in_doorbell = 0;
    if (!landed) { rc = -1; ork_dyn_dump(h, "grouped-i4 doorbell miss (recover exhausted)"); }
    struct buf *done[1024]; int nd = 0;
    for (int i = 0; i < h->S; i++) { struct buf *b = h->outbuf[i]; int seen = 0;
        for (int j = 0; j < nd; j++) if (done[j] == b) seen = 1;
        if (!seen && b) { orki_bsync(fd, b, RKNPU_MEM_SYNC_FROM_DEVICE); if (nd < 1024) done[nd++] = b; } }
    int N = h->i4g_N, Sk = h->i4g_Sk; const float *aS = h->i4g_aS, *bS = h->i4g_bS; float *Cf = h->i4g_Cf;
    for (int m = 0; m < h->S; m++) { const int16_t *blk = (const int16_t*)h->outptr[m]; float *cr = Cf + (size_t)m * N;
        for (int n = 0; n < N; n++) { float acc = 0;
            for (int g = 0; g < Sk; g++) acc += aS[(size_t)m*Sk+g] * bS[(size_t)g*N+n] * (float)blk[(size_t)g*N+n];
            cr[n] = acc; } }
    __asm__ volatile("dsb ish":::"memory");
    free(h);
    if (orki_ork_term) { sigaction(SIGTERM, &orki_prev_sig[0], NULL); raise(SIGTERM); }
    return rc;
}

/* P3: sub-nmax N-COLUMN tiling across cores on the NONBLOCK doorbell (int8, Sn==1, K<=4096 Bf, M 1..64).
 * A single matmul C[M,N] is split by N-columns across nc cores exactly as run_multicore does (t0=i*NN/nc);
 * each core computes its [M,Ncore] column range into per-core scratch (M-tiled into orki_mtile_cap(K)-row chained
 * programs when M>cap), NONBLOCK, and end() copies each range back to C's columns [c0,c0+Ncore) at row-stride
 * N (strided for M>1). This gives the doorbell run_multicore's TILE-parallel multi-core for one matmul (the
 * op-partition would otherwise pin a single op to one core). Scratch+copy-back (not direct output): multi-core
 * direct output to a shared resident C is the unsafe ZC-OUT-multicore case; the copy-back after poll is
 * single-threaded => coherent. */
/* DOORBELL per-core parallel dispatch (DEFAULT; ORK_COLSPLIT_SERIAL forces the legacy inline path): each core
 * submits AND polls/accumulates its OWN core on its own pool thread
 * (barrier-synced fire), mirroring the mcworker/stream_worker. CRITICAL: the per-core poll is O(1) — just the
 * last output word — NOT ork_dyn_done_i's O(no) full-surface civac scan; 3 threads full-scanning ~190K words
 * each thrash the memory bus against the NPU writeback and serialize it (that was the earlier failure). The
 * one-time full-surface VERIFY still happens in ork_dyn_end after all cores land (recovery lives there too). */
static double g_f16_slice_us;   /* running max of a SUCCESSFULLY-landed fp16 K-slice completion (us). Drives the AUTO sentinel detect timeout = 1.5x this (adaptive per shape, vs a flat 800ms). Benign cross-core race — a heuristic, not correctness. */
struct ork_csub { ork_npu *c; int i; struct rknpu_submit *subs; ork_w *w; ork_dyn_chain *h; int hardened; int active; int ksbar; };
static void *ork_csub_worker(void *vp){ struct ork_csub *a = vp; ork_npu *c = a->c; int i = a->i, fd = c->fd;
    int cold = !c->mwarm[i];   /* capture BEFORE the bsync section sets mwarm=1 — fp16 uses it for the cold-buffer warmup */
    if (a->active) {
        if (a->h->oSk[i] <= 1) orki_bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE);   /* wide-K (oSk>1) shares the gathered A in maf[0], already flushed by the build gather — skip the redundant per-core maf orki_bsync (unused for i>0, double for i=0) */
        orki_bsync(fd, &c->mrc[i], RKNPU_MEM_SYNC_TO_DEVICE);
        orki_bsync(fd, &c->mtk[i], RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
        if (a->hardened || !c->mwarm[i]) orki_bsync(fd, &c->mcc[i], RKNPU_MEM_SYNC_TO_DEVICE);
        c->mwarm[i] = 1;
    }
    /* NO barrier + BLOCKING submit — EXACTLY the mcworker: 3 pool threads, each does a blocking submit on its
     * core and the kernel-waits (no userspace poll). ORK_COLSPLIT_NB flips to nonblock+poll for comparison. */
    if (a->active) {
      if (a->h->mc_dt == DT_F16 && a->h->oSk[i] > 1 && !a->h->f16_contig) {   /* CONTIG builds ONE chained submit -> take the single-submit else-path (like int8), NOT the per-slice SW-chain */
        /* AUTO SW-CHAIN: fp16 K-split cannot HW-chain across distinct Bb[ks] weight buffers (the next task's base
         * latches while the prior 2x-long fp16 weight-DMA is still draining -> cross-boundary prefetch -> cdma-wild
         * -> NPU soft-reset/wedge). So submit each K-slice's [kstart,np_ks) range SEPARATELY and BLOCKING — the
         * blocking return fully drains that slice's weight-DMA before the next slice's base is programmed (the
         * inter-op barrier the HW chain lacks). Ping-pong stays ON within each slice's same-buffer M-tile chain
         * (safe). The build terminated the regcmd chain at each slice boundary (kb[]); the oSk f32 accumulate below
         * then sums the Sk partials. Tiling recomputed identically to the build (deterministic). */
        ork_w *w = a->w; int M = a->h->oM[i], Sk = a->h->oSk[i], K = w->K;
        int KS = c->soc->ks, CBUFf = (c->soc->cbuf_elems > 32768) ? 32768 : c->soc->cbuf_elems, RBf = CBUFf, kstart = 0;
        const char *sge = getenv("ORK_F16_STAGGER"); int stag_us = sge ? atoi(sge) : 0;   /* variant A: per-core-index fetch stagger (µs) */
        /* fp16 SELF-HEAL — OPT-IN (ORK_F16_RECOV), default OFF. NAIVE retry-same-slice is HARMFUL and DISPROVEN:
         * resubmitting the identical 3-core concurrent fp16 slice re-triggers the SAME concurrent-fetch CDMA wild,
         * and hammering submits at a mid-soft-reset NPU ESCALATES a recoverable soft-reset into a HARD WEDGE
         * (board 10.3.0.236 hard-wedged 2026-08-04; power-cycle recovery, SPI survived). The correct self-heal must
         * de-escalate on fault (serial/single-core recompute — no concurrent wild — after the reset settles), NOT
         * resubmit the same concurrent slice. Kept gated for that future design; DO NOT default-on the naive retry. */
        int f16_recov = getenv("ORK_F16_RECOV") != NULL;
        const char *dte = getenv("ORK_F16_DETECT_MS"); int detect_ms = dte ? atoi(dte) : 500;   /* GROUNDED fast detect: a legit fp16 slice is ms-scale (~15ms max), so the blocking miss-timeout is the ANALOG of int8's ~300ms sentinel miss — tighten from the arbitrary 8s to ~500ms (30x the max legit slice, 16x faster detect). */
        /* REAP PRECONDITION (Task #50): the nonblock submit timeout MUST be < the poll-detect window, else a dropped
         * (benign-miss) job is NOT yet past its timeout when we go to reap it — rknpu_job_timeout_clean only reaps a
         * job whose age >= its timeout — so it lingers as subcore_data->job and the nc=1 backstop queues behind it
         * 60-180s -> rknpu_job_abort. Detect is ~1.5x the measured slice (g_f16_slice_us); set the submit timeout to
         * ~1x the slice (still >> a legit slice, which lands via IRQ before any timeout_clean regardless). Bootstrap
         * 60ms (< the 800ms bootstrap detect) until the first slice sets the baseline; never exceed detect_ms. */
        int recov_tmo = g_f16_slice_us > 0 ? (int)(g_f16_slice_us/1000) + 2 : 60;
        if (recov_tmo > detect_ms) recov_tmo = detect_ms; if (recov_tmo < 3) recov_tmo = 3;
        int f16_sentinel = (getenv("ORK_F16_SENTINEL") != NULL) && !c->f16_force_blocking;   /* force_blocking overrides -> blocking (heal). SENTINEL RECOVERY: submit NONBLOCK + CPU poll-drain each slice
            * (last-word gate + full-slice civac verify), tight timeout. Fast wedge-detect (~poll timeout, not the 8s blocking
            * timeout) AND the CPU never blocks in-kernel (no D-state hard-wedge); on a stuck sentinel -> mc_error -> run-level
            * recovery. Keeps the barrier's per-slice drain semantics (poll drains before advancing). int8-decode-path style. */
        const char *spte = getenv("ORK_F16_SENTINEL_TMO_US"); double f16_poll_ovr = spte ? atof(spte) : 0.0;   /* explicit detect-timeout override (us); 0 => AUTO = 1.5x the measured slice-completion time (g_f16_slice_us) */
        for (int ks = 0; ks < Sk; ks++) {
            int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS;
            int sched = ((Kp&(Kp-1))==0 && Kp>=128 && Kp<2048), R = RBf/Kp; if (R<1) R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
            double scale=(double)Kp/256.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale), mg=base>=0x1b?(base-0x1b)/slope+1:0;
            int kcap=mg*64; if(!sched)kcap=(RBf/2)/Kp; if(kcap<4*R)kcap=sched?4*R:((RBf/2)/Kp); if(kcap<1)kcap=1;
            int np_ks = (M + kcap - 1) / kcap;
            struct rknpu_submit s; memset(&s, 0, sizeof s);
            s.flags = ork_ppflags() | (f16_sentinel ? 0x2u : 0u);   /* SENTINEL: NONBLOCK (0x2) — CPU polls, never blocks in-kernel. Else BLOCKING (ping-pong safe within one buffer). */
            s.timeout = recov_tmo;
            s.task_start = (uint32_t)kstart; s.task_number = (uint32_t)np_ks;
            s.task_obj_addr = c->mtk[i].obj; s.core_mask = 1u << i; s.fence_fd = -1;
            s.subcore_task[0] = s.subcore_task[1] = s.subcore_task[2] = (struct rknpu_subcore_task){(uint32_t)kstart, (uint32_t)np_ks};
            if (i == 0) orki_trace_submit(&s);   /* ORK_TRACE: dump the doorbell fp16 per-slice regcmd sequence (core 0) */
            if (stag_us && i > 0) { struct timespec ts = {0, (long)i * stag_us * 1000}; nanosleep(&ts, NULL); }   /* variant A: offset core i so the 3 cores don't hit the CDMA with fp16 weight fetches at the same instant */
            int src = orki_rknpu_submit_ioctl(fd, &s, w->domain);
            if (f16_sentinel) {   /* SENTINEL RECOVERY: NONBLOCK submit returned immediately — now CPU poll-DRAIN this slice's
                * partial (last-word gate, then full-slice civac verify), with a TIGHT timeout. Fast wedge-detect + CPU never
                * stuck in-kernel. The full surface was SENT-seeded (hardened_w=1 for fp16). Timeout/reject -> mc_error -> run-level recovery. */
                if (src) { c->mc_error = 1;
                    fprintf(stderr, "[F16-WEDGE-DETECT] core=%d Kslice=%d submit_rc=%d (REJECTED at submit) mtk.int_status=0x%x\n",
                            i, ks, src, ((struct rknpu_task*)c->mtk[i].cpu)[0].int_status); fflush(stderr);
                    ork_kmsg("WEDGE-DETECT core=%d Kslice=%d submit_rc=%d (REJECTED)", i, ks, src); }
                else { volatile int32_t *o = (volatile int32_t*)a->h->outptr[i];
                    int Sk_ = a->h->oSk[i] ? a->h->oSk[i] : 1, per = a->h->nout[i] / Sk_;   /* M*Ncore words per K-slice partial */
                    /* AUTO detect timeout = 150% of the measured slice-completion time (g_f16_slice_us, updated on every
                     * successful land) — adaptive per shape instead of a flat 800ms: once the warm/first slice sets the
                     * baseline (~ms), detect drops to ~20ms, cutting the detect-to-reset VULNERABILITY WINDOW ~40x (the
                     * period the wedged NPU sits un-halted, able to keep wild-writing). 20ms floor so a tiny first
                     * measurement + jitter never false-positive; 800ms bootstrap until the first success; env overrides. */
                    double f16_poll_tmo = f16_poll_ovr > 0 ? f16_poll_ovr : (g_f16_slice_us > 0 ? 1.5 * g_f16_slice_us : 800000.0);
                    if (f16_poll_ovr <= 0 && f16_poll_tmo < 20000.0) f16_poll_tmo = 20000.0;
                    size_t last = (size_t)ks * per + per - 1; double pt = ork_now_us(); int wedged = 0;
                    for (;;) { __asm__ volatile("dc civac,%0"::"r"(&o[last]):"memory");   /* O(1) LAST-WORD gate: cheap wedge-detect (a full-slice scan per poll iter was ~7x slower). Coherency for the accumulate is the FROM_DEVICE bsync below; a real wedge faults early so the last word stays SENT. */
                        if (o[last] != ORK_DYN_SENT) { double el = ork_now_us() - pt; if (el > g_f16_slice_us) g_f16_slice_us = el; break; }   /* landed: update the slice-time baseline that drives the 1.5x auto timeout */
                        double el = ork_now_us() - pt; if (el > f16_poll_tmo) { wedged = 1; break; }
                        if (el > 500.0) { struct timespec ts = {0, 50000}; nanosleep(&ts, NULL); } }
                    if (wedged) { c->mc_error = 1;   /* IMMEDIATE located dump — earliest possible, before the barrier/join, so it
                        * streams out BEFORE any hard-hang loses it. Names the faulting (core, K-slice) + poll elapsed + the
                        * last-word value that stayed SENT (the doorbell that never landed). */
                        __asm__ volatile("dc civac,%0"::"r"(&o[last]):"memory");
                        fprintf(stderr, "[F16-WEDGE-DETECT] core=%d Kslice=%d POLL-MISS after %.0fus (tmo=%.0f) last-word[%zu]=0x%08x (SENT=stuck) mtk.int_status=0x%x\n",
                                i, ks, ork_now_us()-pt, f16_poll_tmo, last, (unsigned)o[last], ((struct rknpu_task*)c->mtk[i].cpu)[0].int_status); fflush(stderr);
                        ork_kmsg("WEDGE-DETECT core=%d Kslice=%d POLL-MISS %.0fus last=0x%08x", i, ks, ork_now_us()-pt, (unsigned)o[last]); } }
            } else {   /* BLOCKING (default): the kernel drains this slice; ORK_F16_RECOV (opt-in, DISPROVEN) resubmits on fault */
                for (int rt = 0; f16_recov && src && rt < 4; rt++) {
                    struct timespec rs = {0, 3000000 + (long)i * 2000000}; nanosleep(&rs, NULL);
                    src = orki_rknpu_submit_ioctl(fd, &s, w->domain);
                }
                if (src) c->mc_error = 1;
            }
            kstart += np_ks;
            /* LOCKSTEP: wait for every core to finish this K-slice before any core programs the next slice's
             * base — keeps all cores fetching the SAME Bb[ks] concurrently (never two distinct fp16 weight
             * buffers on the CDMA at once = the concurrent-cross-buffer wild). The blocking submit above has
             * already drained this core's slice, so the barrier only gates the CPU-side loop advance. */
            if (a->ksbar) pthread_barrier_wait(&c->b_ioctl);
        }
        if (i == 0) { const char *fwe = getenv("ORK_F16_FORCE_WEDGE");   /* TEST-ONLY fault injection (no real NPU fault; submits
            * above all succeeded): mark a simulated wedge for the FIRST N colsplit attempts (countdown), then succeed. N small
            * (e.g. 2) exercises the reset+resubmit HEAL path (must heal by attempt N+1, bit-exact); N huge (e.g. 99) exhausts
            * recovery and exercises the nc=1 de-escalation BACKSTOP. Post-barrier, core 0 only. */
            if (fwe) { static int fw_remain = -1; if (fw_remain < 0) fw_remain = atoi(fwe); if (fw_remain > 0) { c->mc_error = 1; fw_remain--; } } }
      } else {
        int nb = getenv("ORK_COLSPLIT_NB") != NULL;
        if (!nb) a->subs[i].flags &= ~0x2u;   /* blocking (default here) */
        a->subs[i].timeout = (a->h->mc_dt == DT_F16 && orki_mm_timeout_ms() > 8000) ? 8000 : orki_mm_timeout_ms();   /* fp16 (incl. CONTIG): cap wedge-detect at 8s so a CDMA-wild fault falls to the run-level reset+resubmit fast, not a 60s stall */
        int _csrc = orki_rknpu_submit_ioctl(fd, &a->subs[i], a->w->domain);
        if (_csrc && a->h->mc_dt == DT_F16) c->mc_error = 1;   /* CONTIG fp16 wedge: signal the run-level recovery (reset + resubmit / nc=1 backstop) */
        if (i == 0 && a->h->f16_contig) { const char *fwe = getenv("ORK_F16_FORCE_WEDGE");   /* TEST-ONLY fault injection for the
            * CONTIG in-place heal loop (no real NPU fault; the submit above succeeded): flag a simulated drop for the first N
            * begin passes (countdown), then succeed. N small (e.g. 2) must heal by pass N+1 bit-exact; N huge exhausts the heal
            * and exercises the nc=1 backstop. Core 0 only, after this pass's submit. */
            if (fwe) { static int fw_remain = -1; if (fw_remain < 0) fw_remain = atoi(fwe); if (fw_remain > 0) { c->mc_error = 1; fw_remain--; } } }
        if (nb) { volatile int32_t *o = (volatile int32_t*)a->h->outptr[i]; int no = a->h->nout[i];
            double pt = ork_now_us();
            for (;;) { __asm__ volatile("dc civac,%0"::"r"(&o[no-1]):"memory");
                if (o[no-1] != ORK_DYN_SENT) {   /* O(1) gate: last word landed -> full-surface VERIFY once (the
                    * K-split partial surface's write-order isn't last-word-last, so confirm EVERY word before
                    * declaring done — else the host-accumulate reads a premature partial = wrong output). Cheap:
                    * only scans once the gate passes (near-done), and each thread scans ONLY its own core. */
                    int all = 1; for (int e = 0; e < no; e++) { __asm__ volatile("dc civac,%0"::"r"(&o[e]):"memory"); if (o[e] == ORK_DYN_SENT) { all = 0; break; } }
                    if (all) break;
                }
                double el = ork_now_us() - pt; if (el > 3e6) break;
                if (el > 1000.0) { struct timespec ts = {0, 50000}; nanosleep(&ts, NULL); } } }
      }
        orki_bsync(fd, &c->mcc[i], RKNPU_MEM_SYNC_FROM_DEVICE);
        /* WIDE-K PARALLEL ACCUMULATE: sum this core's Sk [M,Ncore] partials into its C columns HERE, in this
         * pool thread — matching mcworker's chain-ksplit (each core accumulates its own partials in parallel)
         * instead of the SERIAL sum in ork_dyn_end (measured ~31ms serial vs ~13ms NPU submit on ffn_down =
         * the whole doorbell-vs-mcworker wide-K gap). The blocking submit guarantees every partial has landed;
         * the 3 cores write DISJOINT C column ranges (dst=C+c0), so no cross-core race. NEON int32 (bit-exact:
         * integer add is associative). dst[i]=NULL => ork_dyn_end's copy-back skips this core. */
        if (a->h->oSk[i] > 1 && a->h->dst[i]) {   /* PER-CORE PARALLEL accumulate (int8 AND fp16) — each core sums its
            * own Sk [M,Ncore] partials in this pool thread (per-core parallel accumulate). The blocking per-slice
            * submits + the FROM_DEVICE bsync above guarantee this core's partials have landed in DRAM; the lockstep
            * barrier (csub_barrier) guarantees no core crossed into a later Bb[ks] while another was still fetching
            * this one (the concurrent-cross-buffer CDMA wild that produced the wrong-answers). So fp16 no longer needs
            * the single-threaded full-surface verify+accumulate in ork_dyn_end (5-9x slower — it serialized all cores'
            * partials through one thread); that path stays as a dormant fallback (unreached: this sets dst[i]=NULL). */
            int Me = a->h->oM[i] ? a->h->oM[i] : 1, Sk = a->h->oSk[i], no = a->h->nout[i], Nn = no/(Sk*Me);
            size_t ds = a->h->ostride[i] > 0 ? (size_t)a->h->ostride[i] : (size_t)Nn, kstride = (size_t)Me*Nn;
            /* ks-OUTER (cache-friendly, like mcworker's chain-ksplit accumulate): read each K-slice partial
             * CONTIGUOUSLY and accumulate into the small C column block (stays hot in cache across the Sk
             * passes). The prior (m,n)-outer/ks-inner order scattered every element's reads across Sk
             * partials 1.2MB apart -> cache/TLB thrash (measured ~8-10ms vs ~2ms). Bit-exact (int add assoc;
             * fp16 f32-add order matches mcworker's ks-ascending sum lane-for-lane). */
            if (a->h->mc_dt == DT_F16) {   /* fp16: f32 partials + f32 accumulate (Stage 1) */
                const float *src = (const float*)a->h->outptr[i]; float *d = (float*)a->h->dst[i];
                for (int m = 0; m < Me; m++) { const float *bs = src + (size_t)m*Nn; float *dr = d + (size_t)m*ds;
                    for (int n = 0; n < Nn; n++) dr[n] = bs[n]; }   /* ks=0 init */
                for (int ks = 1; ks < Sk; ks++) { const float *sk = src + (size_t)ks*kstride;
                    for (int m = 0; m < Me; m++) { const float *bs = sk + (size_t)m*Nn; float *dr = d + (size_t)m*ds; int n = 0;
                        for (; n+4 <= Nn; n += 4) vst1q_f32(dr+n, vaddq_f32(vld1q_f32(dr+n), vld1q_f32(bs+n)));
                        for (; n < Nn; n++) dr[n] += bs[n]; } }
            } else {
                const int32_t *src = (const int32_t*)a->h->outptr[i]; int32_t *d = a->h->dst[i];
                for (int m = 0; m < Me; m++) { const int32_t *bs = src + (size_t)m*Nn; int32_t *dr = d + (size_t)m*ds;
                    for (int n = 0; n < Nn; n++) dr[n] = bs[n]; }   /* ks=0 init */
                for (int ks = 1; ks < Sk; ks++) { const int32_t *sk = src + (size_t)ks*kstride;
                    for (int m = 0; m < Me; m++) { const int32_t *bs = sk + (size_t)m*Nn; int32_t *dr = d + (size_t)m*ds; int n = 0;
                        for (; n+4 <= Nn; n += 4) vst1q_s32(dr+n, vaddq_s32(vld1q_s32(dr+n), vld1q_s32(bs+n)));
                        for (; n < Nn; n++) dr[n] += bs[n]; } }
            }
            a->h->dst[i] = NULL;   /* accumulated per-core; ork_dyn_end copy-back skips this i */
        }
    }
    return NULL;
}
ork_dyn_chain *ork_dyn_begin_colsplit(ork_npu *c, const ork_mm_task_i8 *t, int ncreq) {
    ork_w *w = t->w; int K = w->K, N = w->N, M = t->M, fd = c->fd, CBUF = c->soc->cbuf_elems;
    int dt = w->dtype;   /* DT_I8 today; fp16/int4 branches keyed on this (Stage 0: dt==DT_I8 == byte-identical) */
    int nt_sz = (dt == DT_F16) ? 16 : 32, NN = N / nt_sz, mcap = orki_mtile_cap(K), NMAX_C = c->soc->nmax;   /* col-tile width: int8 32, fp16 16 (each Kp*32 BYTES: int8 32x1, fp16 16x2); mcap rows/program (int8); NMAX_C = N-slice width */
    int nc = ncreq; if (nc > NN) nc = NN; if (nc > c->soc->cores) nc = c->soc->cores; if (nc < 1) nc = 1;
    if (dt == DT_F16 && w->Sk > 1 && nc > 2 && getenv("ORK_F16_2CORE")) nc = 2;   /* variant B: fewer concurrent fp16 fetchers (reliability/speed trade) */
    if (w->domain != c->dom_active || (w->domain && !c->dom_save)) orki_dom_activate(c, w->domain);
    /* Enter the correct PRECISION mode. fp16 must enter the fp16 mode EXACTLY as mcworker does
     * (DT_F16, XP_MC_MM) — entering the int8-chain mode (DT_I8_CHAIN) for fp16 leaves the stateful
     * regcmd datapath mismatched, which intermittently DMA-wilds / soft-resets the NPU on the fp16
     * submits. int8 keeps the HW-chain mode. This is the doorbell-vs-mcworker difference. */
    if (dt == DT_F16) ork_npu_enter(c, DT_F16, XP_MC_MM, OCK_NONE);
    else              ork_npu_enter(c, 3 /*DT_I8_CHAIN*/, XP_CHAIN_NT, OCK_HW);
    if (orki_mc_ensure(c, nc)) return NULL;
    ork_dyn_chain *h = calloc(1, sizeof *h); if (!h) return NULL;
    h->c = c; h->S = nc; h->P = nc; h->N = N; h->dom = w->domain; h->reserve = nc; h->mc = 1;
    h->mc_dt = dt;   /* set EARLY: ork_csub_worker (runs before the tail below) reads it for the accumulate dtype */
    struct rknpu_submit subs[ORK_MAXCORE]; int Pc[ORK_MAXCORE]; memset(Pc, 0, sizeof Pc);
    uint32_t rc[REGCMD_I8_N + 4];
    for (int i = 0; i < nc; i++) {
        int t0 = (int)((long)i * NN / nc), t1 = (int)((long)(i+1) * NN / nc), Ncore = (t1 - t0) * nt_sz, c0 = t0 * nt_sz;
        /* BALANCED wide-N (Sn>1): each core owns the even ~N/nc contiguous column range [c0,c1) (t0=i*NN/nc,
         * bit-exact to mcworker) — this keeps the per-core weight-DMA volume balanced (the wall-clock lever;
         * a whole-slice-per-core split imbalanced it 8192/8192/2560 and cost 1.3x). The range can CROSS an
         * nmax slice boundary; the base emission below cuts it into within-slice segments, each written to a
         * SEGMENT-MAJOR contiguous scratch block (synth stride=0 => no notch), and end() BOUNDARY-SCATTERS the
         * blocks to C[c0..c1) at row-stride N (h->oscat). Balanced AND notch-free — the two levers the
         * descriptor-array dump pinned as the mcworker-vs-doorbell gap (2026-08-03). */
        if (Ncore <= 0) { Pc[i] = 0; continue; }
        struct buf *RC = &c->mrc[i], *AF = &c->maf[i]; struct rknpu_task *tk = (struct rknpu_task*)c->mtk[i].cpu;
        size_t aesz = (dt == DT_F16) ? 2 : 1;   /* A element bytes: fp16 2, int8 1 */
        if ((size_t)M * K * aesz > AF->size) { orki_bdestroy(fd, &c->maf[i]); c->maf[i] = orki_bcreate(fd, (size_t)M*K*aesz, 0x403, c->dom_active); if (!c->maf[i].cpu) { free(h); return NULL; } AF = &c->maf[i]; }
        if (dt == DT_F16) {   /* fp16 colsplit (Stage 1): K-sliced Bb + host f32 accumulate; Sn==1 (gated). Mirrors the
            * int8 WIDE-K branch with orki_synth()/f32/fp16-chunk. base (Sk==1) => single partial (accumulate is a copy).
            * Weight offset t0*Kp*32 and the 108-reg task are IDENTICAL to int8/mcworker (only orki_synth()+Bb+dtype differ). */
            int CBUFf = (CBUF > 32768) ? 32768 : CBUF;   /* fp16 M-scheduler is validated only to the 32768-tile; a larger cbuf miscomputes mc>~cap (mcworker applies the same cap) */
            int KS = c->soc->ks, RBf = CBUFf;   /* fp16: RB = cbuf (int8 doubles it) */
            struct rknpu_task *tkf = (struct rknpu_task*)c->mtk[i].cpu;
            size_t ksz = (size_t)w->Sk * M * Ncore * 4;   /* Sk f32 partials [ks][M][Ncore] */
            if (c->mccsz[i] < ksz) { orki_bdestroy(fd, &c->mcc[i]); c->mcc[i] = orki_bcreate(fd, ksz, 0x403, c->dom_active);
                if (!c->mcc[i].cpu) { free(h); return NULL; } c->mccsz[i] = ksz; c->mwarm[i] = 0; }
            struct buf *CC = &c->mcc[i];
            /* pre-grow mrc/mtk for this core: fp16 chunks are small (<=8 @ K>=2048) => Sk*ceil(M/chunk) programs.
             * bound generously (512); tkf re-fetched after any grow. */
            size_t needrc = (size_t)512 * REGCMD_N * 4, needtk = (size_t)512 * sizeof(struct rknpu_task);
            if (RC->size < needrc) { orki_bdestroy(fd, &c->mrc[i]); c->mrc[i] = orki_bcreate(fd, needrc, 0x403, c->dom_active);
                if (!c->mrc[i].cpu) { free(h); return NULL; } RC = &c->mrc[i]; c->mwarm[i] = 0; }
            if (c->mtk[i].size < needtk) { orki_bdestroy(fd, &c->mtk[i]); c->mtk[i] = orki_bcreate(fd, needtk, 0x40b, c->dom_active);
                if (!c->mtk[i].cpu) { free(h); return NULL; } tkf = (struct rknpu_task*)c->mtk[i].cpu; }
            struct buf *AFS = &c->maf[0];   /* gather A ONCE (shared, read-only across cores): fp16 [Sk][M][Kp] */
            if (i == 0) { f16 *afg = (f16*)AFS->cpu; const f16 *Af = (const f16*)t->A; size_t goff = 0;
              for (int ks = 0; ks < w->Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS;
                  for (int m = 0; m < M; m++) memcpy(afg + goff + (size_t)m*Kp, Af + (size_t)m*K + k0, (size_t)Kp*2);   /* per-row memcpy (== int8 wide-K gather); Sk==1 => contiguous. Scalar j-loop was a big fixed cost on low-M shapes. */
                  goff += (size_t)M*Kp; }
              orki_bsync(fd, AFS, RKNPU_MEM_SYNC_TO_DEVICE); }
            uint32_t a_base = (uint32_t)AFS->dma;
            int contig = (w->Sn == 1) && !getenv("ORK_F16_NO_CONTIG");   /* (A) DEFAULT-ON for Sn==1: ONE chained submit/core over a CONTIGUOUS weight (Bbc) — no cross-buffer boundary => no HW cross-boundary prefetch => no CDMA wild => no drop (validated 500-iter 0-drop + shape-suite bit-exact, 2.6x). Sn>1 (multi-N-slice) keeps the per-slice path + the recovery net. ORK_F16_NO_CONTIG opts out for A/B. */
            if (contig && i == 0 && !w->Bbc_valid) {   /* build the contiguous weight ONCE (single-threaded build): concat all Sk K-slice tiles into one buffer */
                size_t tot = 0; for (int ks = 0; ks < w->Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS; tot += (size_t)Kp*N*2; }
                w->Bbc = orki_bcreate(fd, tot, 0x403, w->domain);
                if (w->Bbc.cpu) { size_t off = 0;
                    for (int ks = 0; ks < w->Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS; size_t sz = (size_t)Kp*N*2;
                        memcpy((char*)w->Bbc.cpu + off, w->Bb[ks].cpu, sz); off += sz; }
                    orki_bsync(fd, &w->Bbc, RKNPU_MEM_SYNC_TO_DEVICE); w->Bbc_valid = 1; }
            }
            if (contig && !w->Bbc_valid) contig = 0;   /* alloc failed -> fall back to the per-slice SW-chain */
            h->f16_contig = contig;
            int gapbase = (contig && getenv("ORK_F16_GAP")) ? atoi(getenv("ORK_F16_GAP")) : 0;   /* (B'') per-core DIFFERENTIAL spin-stagger base: core i gets gapbase*(nc-1-i) identity-perchan filler tasks BETWEEN slices, so core (nc-1) launches ks+1 first and core 0 last — OFFSETS the 3 cores' fp16 fetches in time to de-conflict the concurrent-fetch wild (not a synchronized drain — a staggered restart) */
            int gap = gapbase > 0;
            const int GAPM = 8, GAPN = 64;   /* tiny perchan geometry — pure time-filler; GAPM<=64 (perchan cap), GAPN%32==0 */
            uint32_t gap_pc[REGCMD_MUL_F16_CHAIN_N];
            if (gap) {
                if (i == 0 && !w->Bgap_valid) {   /* build the dummy gap buffers ONCE (shared, single-threaded here) */
                    w->Bgap[0]=orki_bcreate(fd,4096,0x403,w->domain); w->Bgap[1]=orki_bcreate(fd,4096,0x403,w->domain); w->Bgap[2]=orki_bcreate(fd,4096,0x403,w->domain);
                    if (w->Bgap[0].cpu && w->Bgap[1].cpu && w->Bgap[2].cpu) {
                        memset(w->Bgap[0].cpu,0,4096); memset(w->Bgap[1].cpu,0,4096);
                        { uint16_t *sb=(uint16_t*)w->Bgap[2].cpu; for (int e=0;e<GAPN;e++) sb[e]=0x3c00; }   /* identity per-channel scale (fp16 1.0) */
                        orki_bsync(fd,&w->Bgap[0],RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&w->Bgap[1],RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&w->Bgap[2],RKNPU_MEM_SYNC_TO_DEVICE);
                        w->Bgap_valid = 1;
                    }
                }
                if (!w->Bgap_valid) gap = 0;
                else { memcpy(gap_pc, REGCMD_MUL_F16_CHAIN, sizeof gap_pc); orki_set_mul_geom(gap_pc, REGCMD_MUL_F16_CHAIN_N, GAPM, GAPN);
                    orki_setrn(gap_pc, REGCMD_MUL_F16_CHAIN_N, RK_DPU_DST_BASE_ADDR, (uint32_t)w->Bgap[1].dma);
                    orki_setrn(gap_pc, REGCMD_MUL_F16_CHAIN_N, RK_SDP_5018, (uint32_t)w->Bgap[0].dma);
                    orki_setrn(gap_pc, REGCMD_MUL_F16_CHAIN_N, RK_SDP_5038, (uint32_t)w->Bgap[2].dma);
                    orki_setrn(gap_pc, REGCMD_MUL_F16_CHAIN_N, RK_SDP_5034, 0x00000008); }
            }
            int np2 = 0; size_t goff = 0, sloff = 0; char kb[512] = {0}; unsigned char pcp[600] = {0};   /* pcp[p]=1 => program p is a perchan drain-gap (not a matmul) */
            for (int ks = 0; ks < w->Sk; ks++) {
                int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS;
                int sched = ((Kp&(Kp-1))==0 && Kp>=128 && Kp<2048);   /* fp16 sched window (NO HISCHED here) */
                int R = RBf/Kp; if (R<1) R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
                double scale=(double)Kp/256.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale), mg_max = base>=0x1b ? (base-0x1b)/slope+1 : 0;
                int kcap = mg_max*64; if(!sched) kcap=(RBf/2)/Kp; if(kcap<4*R) kcap=sched?4*R:((RBf/2)/Kp); if(kcap<1) kcap=1;   /* fp16 M-tile cap — NEVER raise: >cap miscomputes (npu.c ~4770) */
                uint32_t wbase = (uint32_t)((contig ? (w->Bbc.dma + sloff) : w->Bb[ks].dma) + (uint64_t)t0 * Kp * 32);   /* CONTIG: slice base = Bbc + cumulative slice offset (one buffer); else per-slice Bb[ks]. N-tile stride Kp*32 (== mcworker) */
                for (int m0 = 0; m0 < M; m0 += kcap) { int mc = (M-m0<kcap)?(M-m0):kcap;
                    if ((size_t)(np2+1) * REGCMD_N * 4 > RC->size) { free(h); return NULL; }
                    memset(rc, 0, REGCMD_N * 4);
                    orki_synth(rc, mc, Kp, Ncore, (uint32_t)(a_base + (goff + (size_t)m0*Kp)*2), wbase,
                          (uint32_t)(CC->dma + ((size_t)ks*M + m0)*Ncore*4), sched, CBUFf);
                    if (orki_validate_regcmd("ork_dyn_colsplit_f16", c, rc, REGCMD_N, w, NULL, 0)) { free(h); return NULL; }
                    memcpy((char*)RC->cpu + (size_t)np2*REGCMD_N*4, rc, REGCMD_N*4);
                    np2++;
                }
                if (!contig && np2 > 0 && np2 <= 512) kb[np2-1] = 1;   /* end of this K-slice's sub-chain (per-slice submit); CONTIG leaves it linked */
                if (gap && ks < w->Sk-1) {   /* (B'') DIFFERENTIAL spin-stagger: core i gets gapbase*(nc-1-i) filler perchans between slices (padded into REGCMD_N slots; HW reads each one's 69 regs) — offsets this core's ks+1 launch vs the others */
                    int gapK = gapbase * (nc - 1 - i);
                    for (int g = 0; g < gapK && np2 < 500; g++) {
                        if ((size_t)(np2+1) * REGCMD_N * 4 > RC->size) break;
                        memcpy((char*)RC->cpu + (size_t)np2*REGCMD_N*4, gap_pc, REGCMD_MUL_F16_CHAIN_N*4);
                        pcp[np2] = 1; np2++;
                    }
                }
                goff += (size_t)M*Kp; sloff += (size_t)Kp*N*2;
            }
            for (int p = 0; p < np2; p++) { uint32_t *pr = (uint32_t*)((char*)RC->cpu + (size_t)p*REGCMD_N*4);
                if (p < np2-1 && (contig || !kb[p])) { uint64_t nx = RC->dma + (size_t)(p+1)*REGCMD_N*4;   /* CONTIG: link ALL (one chain across slices + gaps); SW-chain: link only WITHIN a K-slice */
                    int slot = pcp[p] ? 138 : 216;                    /* perchan links at its 138 slot (regcfg*2), matmul at 216 */
                    int nreg = pcp[p+1] ? ((69+3)/2) : ((108+3)/2);   /* next-amount = the NEXT program's register count */
                    pr[slot]   = 0x0010 | ((nx & 0xffff) << 16); pr[slot+1] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
                    pr[slot+2] = 0x0014 | ((uint32_t)nreg << 16); pr[slot+3] = (0x0101 << 16); }
                struct rknpu_task tt; memset(&tt, 0, sizeof tt); tt.int_mask = 0x300; tt.int_clear = 0x1ffff;
                tt.enable_mask = pcp[p] ? 0x18 : 0xd; tt.regcfg_amount = pcp[p] ? 69 : 108;   /* perchan: enable 0x18, 69 regs; matmul: 0xd, 108 */
                tt.regcmd_addr = RC->dma + (size_t)p*REGCMD_N*4; tkf[p] = tt; }
            h->outbuf[i] = CC; h->outptr[i] = (int32_t*)CC->cpu; h->nout[i] = w->Sk * M * Ncore; h->oM[i] = M; h->oSk[i] = w->Sk;
            h->dst[i] = (int32_t*)((char*)t->C + (size_t)c0 * 4); h->ostride[i] = t->cstride ? t->cstride : N;   /* f32 accumulate/copy-back -> C columns at row-stride N (cstride override: fp16 wide-N per-slice writes a sub-N result into the wider C at full stride) */
            Pc[i] = np2;
            memset(&subs[i], 0, sizeof subs[i]);
            subs[i].flags = (gap ? 0x1u : ork_ppflags()) | 0x2u; subs[i].task_number = np2; subs[i].task_obj_addr = c->mtk[i].obj;   /* gap chain carries an SDP (perchan) task -> ping-pong OFF (0x1); worker clears 0x2 -> blocking */
            subs[i].core_mask = 1u << i; subs[i].fence_fd = -1;
            subs[i].subcore_task[0] = subs[i].subcore_task[1] = subs[i].subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)np2};
            continue;
        }
        uint32_t adma = (uint32_t)AF->dma;
        if (K <= 4096) memcpy(AF->cpu, t->A, (size_t)M * K);   /* full A[M,K] for base/wide-N. WIDE-K (K>4096) re-gathers A into AF as [Sk][M][Kp] below, so this full copy would be pure waste there — skip it. */
        size_t osz = (size_t)M * Ncore * 4;
        if (c->mccsz[i] < osz) { orki_bdestroy(fd, &c->mcc[i]); c->mcc[i] = orki_bcreate(fd, osz, 0x403, c->dom_active);
            if (!c->mcc[i].cpu) { free(h); return NULL; } c->mccsz[i] = osz; c->mwarm[i] = 0; }
        struct buf *CC = &c->mcc[i];
        if (K > 4096 || !w->Bf) {   /* WIDE-K (K>4096) or NO-Bf (K<=4096, ORK_NO_BF) colsplit (Sn==1, ANY M): the Bf-FREE per-core K-split — Sk*(M-tile) partial
            * [mc,Ncore] programs over this core's column range; end() SUMS the Sk [M,Ncore] partials into
            * C[c0:c1) at row-stride N (ork_dyn_end oSk accumulate, [ks][m][n] layout). M>1: A's K-slice rows are
            * strided by the FULL K, but synth reads A contiguous [mc,Kp], so gather A[M,K] -> AF as a per-slice-
            * contiguous [Sk][M][Kp] layout first. Each K-slice's Kp<=KS gives an mg_max*64 cap >=320, so M<=256
            * prefill fits one program/slice; M-tiled defensively for larger M. */
            int KS = orki_int8_ks(c);
            size_t ksz = (size_t)w->Sk * M * Ncore * 4;
            if (c->mccsz[i] < ksz) { orki_bdestroy(fd, &c->mcc[i]); c->mcc[i] = orki_bcreate(fd, ksz, 0x403, c->dom_active);
                if (!c->mcc[i].cpu) { free(h); return NULL; } c->mccsz[i] = ksz; c->mwarm[i] = 0; CC = &c->mcc[i]; }
            /* A[M,K]->[Sk][M][Kp] is IDENTICAL for every core (A is not core-dependent — only the weight column
             * range differs per core). Gather it ONCE into maf[0] (shared, read-only) and point every core's
             * regcmd at it, instead of the redundant per-core gather that dominated the serial build (~1.4ms x
             * nc). maf[0] is sized M*K by the i==0 realloc above; the 3 cores only READ it (no write race). */
            struct buf *AFS = &c->maf[0];
            if (i == 0) { int8_t *afg = (int8_t*)AFS->cpu; size_t goff = 0;
              for (int ks = 0; ks < w->Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS;
                  for (int m = 0; m < M; m++) memcpy(afg + goff + (size_t)m*Kp, (const int8_t*)t->A + (size_t)m*K + k0, (size_t)Kp);
                  goff += (size_t)M*Kp; }
              orki_bsync(fd, AFS, RKNPU_MEM_SYNC_TO_DEVICE); }
            uint32_t a_base = (uint32_t)AFS->dma;   /* shared gathered A for all cores */
            int np2 = 0; size_t goff = 0;
            for (int ks = 0; ks < w->Sk; ks++) {
                int k0 = ks * KS, Kp = (K - k0 < KS) ? (K - k0) : KS; int sched = (Kp == 1024 || Kp == 512);
                int kcap = orki_mtile_cap(Kp); if (kcap < 1) kcap = 1;   /* rows/program for this K-slice */
                uint32_t wbase = (uint32_t)(w->Bb[ks].dma + (uint64_t)(c0 / nt_sz) * Kp * nt_sz);   /* K-slice ks weight, column sub-range */
                for (int m0 = 0; m0 < M; m0 += kcap) { int mc = (M - m0 < kcap) ? (M - m0) : kcap;
                    if ((size_t)(np2+1) * REGCMD_I8_N * 4 > RC->size) { free(h); return NULL; }
                    memset(rc, 0, sizeof rc);
                    orki_synth_i8(rc, mc, Kp, Ncore, (uint32_t)(a_base + goff + (size_t)m0*Kp), wbase,
                             (uint32_t)(CC->dma + ((size_t)ks * M + m0) * Ncore * 4), sched, CBUF, 0);   /* [mc,Ncore] rows [m0,+mc) of partial ks */
                    if (orki_validate_regcmd("ork_dyn_colsplit_ks", c, rc, REGCMD_I8_N, w, NULL, 0)) { free(h); return NULL; }
                    memcpy((char*)RC->cpu + (size_t)np2 * REGCMD_I8_N * 4, rc, REGCMD_I8_N * 4);
                    np2++;
                }
                goff += (size_t)M*Kp;
            }
            for (int p = 0; p < np2; p++) { uint32_t *pr = (uint32_t*)((char*)RC->cpu + (size_t)p * REGCMD_I8_N * 4);
                if (p < np2 - 1) { uint64_t nx = RC->dma + (size_t)(p+1) * REGCMD_I8_N * 4;
                    pr[216] = 0x0010 | ((nx & 0xffff) << 16); pr[217] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
                    pr[218] = 0x0014 | (0x0037u << 16);       pr[219] = (0x0101 << 16); }
                struct rknpu_task tt; memset(&tt, 0, sizeof tt); tt.enable_mask = 0xd; tt.int_mask = 0x300;
                tt.int_clear = 0x1ffff; tt.regcfg_amount = 108; tt.regcmd_addr = RC->dma + (size_t)p * REGCMD_I8_N * 4; tk[p] = tt; }
            h->outbuf[i] = CC; h->outptr[i] = (int32_t*)CC->cpu; h->nout[i] = w->Sk * M * Ncore; h->oM[i] = M; h->oSk[i] = w->Sk;
            h->dst[i] = (int32_t*)((char*)t->C + (size_t)c0 * 4); h->ostride[i] = t->cstride ? t->cstride : N;   /* accumulate -> C columns at row-stride N (cstride override: int8 no-Bf/wide-K wide-N per-N-slice writes a sub-N result into the wider C at full stride) */
            Pc[i] = np2;
            memset(&subs[i], 0, sizeof subs[i]);
            subs[i].flags = ork_ppflags() | 0x2u; subs[i].task_number = np2; subs[i].task_obj_addr = c->mtk[i].obj;
            subs[i].core_mask = 1u << i; subs[i].fence_fd = -1;
            subs[i].subcore_task[0] = subs[i].subcore_task[1] = subs[i].subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)np2};
            continue;
        }
        int np = 0, nseg = 0;   /* programs for this core: (within-slice SEGMENT x M-tile), segment-major scratch */
        /* This core owns the CONTIGUOUS column range [c0,c1) of C, which may CROSS nmax slice boundaries. Cut it
         * into within-slice segments; lay the scratch SEGMENT-MAJOR ([M,segw] blocks back-to-back) and write each
         * program contiguously (synth stride=0 => Ncol==row-stride, NO notch). Balanced (even [c0,c1)) + notch-free.
         * end() boundary-scatters each block to its C column sub-range at row-stride N (h->oscat, using h->ocol0). */
        { int c1e = c0 + Ncore; size_t segbase = 0; int cur = c0;
          while (cur < c1e) {
              int ns = cur / NMAX_C, sl1 = (ns + 1) * NMAX_C; if (sl1 > N) sl1 = N;
              int segend = (c1e < sl1) ? c1e : sl1, segw = segend - cur, is0 = cur - ns * NMAX_C;
              uint32_t wbase = (uint32_t)(w->Bf[ns].dma + (uint64_t)(is0 / nt_sz) * K * nt_sz);   /* slice ns, in-slice col offset */
              for (int m0 = 0; m0 < M; m0 += mcap) { int mc = (M - m0 < mcap) ? (M - m0) : mcap;
                  if ((size_t)(np+1) * REGCMD_I8_N * 4 > RC->size) { free(h); return NULL; }
                  memset(rc, 0, sizeof rc);
                  orki_synth_i8(rc, mc, K, segw, adma + (uint32_t)((size_t)m0 * K), wbase,
                           (uint32_t)(CC->dma + (segbase + (size_t)m0 * segw) * 4), 1, CBUF, 0);   /* [mc,segw] contiguous block */
                  if (orki_validate_regcmd("ork_dyn_colsplit", c, rc, REGCMD_I8_N, w, NULL, 0)) { free(h); return NULL; }
                  memcpy((char*)RC->cpu + (size_t)np * REGCMD_I8_N * 4, rc, REGCMD_I8_N * 4);
                  np++;
              }
              segbase += (size_t)M * segw; cur = segend; nseg++;
          }
        }
        for (int p = 0; p < np; p++) { uint32_t *pr = (uint32_t*)((char*)RC->cpu + (size_t)p * REGCMD_I8_N * 4);
            if (p < np - 1) { uint64_t nx = RC->dma + (size_t)(p+1) * REGCMD_I8_N * 4;
                pr[216] = 0x0010 | ((nx & 0xffff) << 16); pr[217] = (0x0101 << 16) | ((nx >> 16) & 0xffff);
                pr[218] = 0x0014 | (0x0037u << 16);       pr[219] = (0x0101 << 16); }
            struct rknpu_task tt; memset(&tt, 0, sizeof tt); tt.enable_mask = 0xd; tt.int_mask = 0x300;
            tt.int_clear = 0x1ffff; tt.regcfg_amount = 108; tt.regcmd_addr = RC->dma + (size_t)p * REGCMD_I8_N * 4; tk[p] = tt; }
        h->outbuf[i] = CC; h->outptr[i] = (int32_t*)CC->cpu; h->nout[i] = M * Ncore; h->oM[i] = M; h->oSk[i] = 0;
        h->dst[i] = (int32_t*)((char*)t->C + (size_t)c0 * 4); h->ostride[i] = N; h->ocol0[i] = c0;
        h->oscat[i] = (nseg > 1);   /* boundary-scatter ONLY when the balanced range crosses a slice boundary (>=2 segments);
            * a single-segment core (all base Sn==1, incl. attention) is a plain contiguous [M,Ncore] block -> keep the
            * cheap per-row-last-col done + ostride copy-back (oscat=0). Setting oscat unconditionally forced the
            * pathological full-surface poll on the base path and tanked native attention 73->13. */
        Pc[i] = np;
        memset(&subs[i], 0, sizeof subs[i]);
        subs[i].flags = ork_ppflags() | 0x2u; subs[i].task_number = np; subs[i].task_obj_addr = c->mtk[i].obj;
        subs[i].core_mask = 1u << i; subs[i].fence_fd = -1;
        subs[i].subcore_task[0] = subs[i].subcore_task[1] = subs[i].subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)np};
    }
    /* HARDENED (decode/stream, M<=64): this per-core scratch is reused across doorbell ops interleaved in the
     * decode loop, which can leave it dirty so a stale CPU line evicts over the NPU write and resurrects a
     * mid-row SENT. Harden with a FULL-surface SENT seed + the ALWAYS-clean bsync below (flushes it to DRAM so
     * nothing evicts over the NPU write); done_i then trusts each row's last col (row-major, last-col-last —
     * same as begin_mc's plain-int8 path). Cheap at small M. PREFILL (M>64, large output) is not interleaved,
     * so keep the original last-col seed + cold-only clean — the per-op full-buffer flush tripped test_speed's
     * latency floor at M=512. Bit-exact either way; this only changes the completion barrier, not the math. */
    /* FULL-surface SENT seed + clean-before when the done-check polls the full surface (not per-row last-col):
     * (a) M<=64 interleaved decode/stream; (b) WIDE-K (K>4096) — oSk partial [Sk][M][Ncore] surface, done only
     * when every partial word lands; (c) WIDE-N (Sn>1, M>1) — Sn scatter blocks whose write-order isn't
     * last-col-last. Without this the M>64 last-col seed mismatches the full-surface poll -> completion misses ->
     * 3s cap -> recover-resubmit stall (measured ~10 t/s on ffn_down). The per-op full flush is small vs the
     * wide K-split/scatter compute. Plain base (Sn==1,K<=4096,M>64) keeps the cheap last-col seed. */
    int hardened = (M <= 64) || (w->K > 4096) || (w->Sn > 1) || (dt == DT_F16 && w->Sk > 1);   /* fp16 K-split partials: write-order not last-col-last => full-surface seed (NONBLOCK path only) */
    /* STAGE 2: the PARALLEL path submits BLOCKING (ork_csub_worker clears the nonblock bit) + sets prepolled, so
     * completion = the blocking ioctl return + bsync FROM_DEVICE — the full-surface SENT seed is NEVER polled
     * there (pure waste: O(M*Ncore) CPU writes/core, the doorbell-specific wide-N host tax native doesn't pay).
     * Skip it and use cold-only clean-before (hardened_w=0). The NONBLOCK path (no PARALLEL, or COLSPLIT_NB)
     * still seeds + polls sentinels, so it keeps the full-surface seed + hardened clean-before. */
    int parallel_blocking = (nc > 1 && !getenv("ORK_COLSPLIT_SERIAL") && !getenv("ORK_COLSPLIT_NB"));
    /* fp16 K-split: the blocking submit's completion can precede the f32 writeback drain, so ork_csub_worker
     * runs a full-surface civac VERIFY even on the parallel-blocking path. That verify needs the SENT seed
     * flushed to DRAM first -> seed the surface AND force the clean-before (hardened_w=1) for fp16 here. */
    int fp16_hard = (dt == DT_F16 && w->Sk > 1);
    int hardened_w = parallel_blocking ? (fp16_hard ? 1 : 0) : hardened;
    /* fp16 K-split lockstep barrier: the residual wedge (during-submit "cdma address wild" + occasional
     * plausible-wrong partial) is a CONCURRENT CROSS-BUFFER FETCH — cores loop their Sk slices independently,
     * so core A can be fetching Bb[ks+1] while core B is still on Bb[ks] = two distinct fp16 weight buffers on
     * the CDMA at once (fp16's 2-byte weights double the fetch bytes; int8 tolerates it, fp16 wilds). A
     * per-slice pthread barrier marches all cores in lockstep: every core finishes slice ks before ANY starts
     * ks+1, so at any instant all cores fetch the SAME Bb[ks] (same-buffer concurrency is benign). Pure CPU
     * sync, no NPU risk. Requires every dispatched core active (else the barrier count nc deadlocks) — for the
     * fp16 colsplit nc<=NN so all cores get columns, but guard anyway and fall back to independent loops. */
    int csub_barrier = 0;
    if (fp16_hard && nc > 1 && parallel_blocking && !getenv("ORK_F16_NOBAR")) { csub_barrier = 1; for (int i = 0; i < nc; i++) if (!Pc[i]) csub_barrier = 0; }
    /* fp16 DROP RECOVERY (Task #50). campaign4 (clean board, sentinel, live-traced) PROVED int8's in-place resubmit
     * model does NOT transfer to fp16: the concurrent-fetch drop is STICKY — reset+resubmit of the identical 3-core
     * fp16 slice re-drops EVERY attempt (observed Kslice=1, cores alternating, tries 1/2/3 all POLL-MISS), and by the
     * 3rd reset a re-launched worker HUNG in an uninterruptible D-state (the kernel `continue wait`, see ork_dummy_probe's
     * header), forcing a power-cycle. So we do NOT resubmit the fp16 slice. On a worker-detected drop: (1) CLEAN-REAP the
     * stuck slice job via ork_npu_reap_stuck (per-core nonblock dummy -> rknpu_job_timeout_clean; avoids the D-state +
     * the close(fd) refcount-UAF), then (2) HEALTH-GATE with ork_dummy_probe — a tiny INT8 nonblock host-bounded op on
     * the same fd (can NEVER hang: int8's drop is a transient dispatch race, not fp16's sticky wild). c->mc_error stays
     * set so the run-level de-escalates to the nc=1 bit-exact backstop (single-core fp16 = no concurrent fetch = no drop).
     * Fast + SAFE: bounded nonblock recovery + a guaranteed-correct single-core recompute, never the resubmit thrash. */
    if (csub_barrier) pthread_barrier_init(&c->b_ioctl, NULL, nc);
    if (!parallel_blocking || fp16_hard)
    for (int i = 0; i < nc; i++) if (Pc[i]) {
        if (hardened) { int no = h->nout[i]; volatile int32_t *o = h->outptr[i]; for (int e = 0; e < no; e++) o[e] = ORK_DYN_SENT; }
        else { int Mx = h->oM[i], Nx = h->nout[i]/Mx; for (int m = 0; m < Mx; m++) {
            volatile int32_t *db = h->outptr[i] + (size_t)m*Nx + (Nx-1); *db = ORK_DYN_SENT; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); } } }
    __asm__ volatile("dsb ish":::"memory");
    if (nc > 1 && !getenv("ORK_COLSPLIT_SERIAL")) {   /* DEFAULT: per-core submit+accumulate on pool threads (ork_csub_worker); ORK_COLSPLIT_SERIAL forces the legacy inline path below */
        struct ork_csub cs[ORK_MAXCORE];
        for (int i = 0; i < nc; i++) cs[i] = (struct ork_csub){ c, i, subs, w, h, hardened_w, Pc[i] != 0, csub_barrier };
        orki_npu_pool_ensure(c);
        pthread_mutex_lock(&c->pmu); c->pjob = cs; c->pjob_nc = nc; c->pjob_fn = ork_csub_worker;
        c->pjob_stride = sizeof(struct ork_csub); c->pdone = 0; c->pgen++; pthread_cond_broadcast(&c->pgo);
        pthread_mutex_unlock(&c->pmu);
        ork_csub_worker(&cs[0]);   /* core 0 on this thread; cores 1..nc-1 on pool threads */
        pthread_mutex_lock(&c->pmu); while (c->pdone < nc - 1) pthread_cond_wait(&c->pdn, &c->pmu); pthread_mutex_unlock(&c->pmu);
        if (csub_barrier) pthread_barrier_destroy(&c->b_ioctl);
        h->prepolled = 1;   /* workers already submitted + drained every core; ork_dyn_end skips its poll */
    } else
    for (int i = 0; i < nc; i++) if (Pc[i]) {
        orki_bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd, &c->mrc[i], RKNPU_MEM_SYNC_TO_DEVICE);
        orki_bsync(fd, &c->mtk[i], RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
        if (hardened || !c->mwarm[i]) orki_bsync(fd, &c->mcc[i], RKNPU_MEM_SYNC_TO_DEVICE);   /* clean-before: ALWAYS for the
            * interleaved decode/stream regime (M<=64 — a shared-scratch dirty line would evict over the NPU write and
            * resurrect a mid-row SENT); cold-only for prefill (M>64, not interleaved — avoids the per-op full flush). */
        c->mwarm[i] = 1;
        subs[i].timeout = orki_mm_timeout_ms(); orki_rknpu_submit_ioctl(fd, &subs[i], w->domain);
    }
    if (fp16_hard && c->mc_error) {   /* dropped fp16 K-slice — DON'T resubmit (sticky re-drop + D-state hang, campaign4).
        * Clean-reap the stuck job + int8 health-gate, then leave mc_error set for the run-level nc=1 bit-exact backstop. */
        /* RESET-FIRST (campaign5 kernel-source root-cause): the vendor rknpu_job_abort() path (hit whenever a submit
         * errors/times out) does rknpu_iommu_domain_put() -> msleep(100) -> rknpu_soft_reset() — a 100ms window with the
         * IOMMU torn down but the NPU CDMA NOT yet halted, so a wild in-flight fp16 fetch escapes into kernel RAM ->
         * slab corruption -> panic (netconsole: Oops in an unrelated proc's kmem_cache_alloc). rknpu_job_timeout_clean()
         * is safe ONLY because it soft-resets BEFORE teardown. So HALT the DMA ourselves FIRST — RKNPU_ACT_RESET
         * (-> rknpu_soft_reset) as the very first recovery action + a settle — so that if any later recovery submit
         * errors into abort, its domain_put window has NO live DMA to escape. Closes the kernel-ordering hole. */
        { struct rknpu_action ra; memset(&ra, 0, sizeof ra); ra.flags = RKNPU_ACT_RESET; ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &ra);
          struct timespec ts = {0, 5000000}; nanosleep(&ts, NULL); }   /* 5ms settle: let the CDMA fully quiesce before any further submit */
        ork_kmsg("F16 drop (K=%d N=%d M=%d) -> RESET-FIRST (halt DMA) + reap_stuck + int8 dummy health-gate + de-escalate to nc=1 (NO fp16 resubmit)", w->K, w->N, M);
        ork_npu_reap_stuck(c, nc);
        int ok = ork_dummy_probe(c);
        ork_kmsg("F16 drop recovery: int8 dummy-probe %s (mc_error stays set -> nc=1 recompute)", ok ? "PASS (NPU dispatching)" : "FAIL (still wedged)");
        for (int z = 0; z < ORK_MAXCORE; z++) c->mwarm[z] = 0;   /* reap/probe reset+reused the scratch -> force a cold re-warm on the nc=1 backstop */
    }
    /* Stash the round context so ork_dyn_end recovers a dropped colsplit round (the M=1 int8 decode path also
     * hits the ~1/2000 doorbell-drop: one core's N-column slice never lands, leaving its re-seeded sentinel
     * column = SENT). colsplit is int8-only; hardened (M<=64) = full-surface seed, matching orki_mc_recover_resubmit. */
    h->mc_nc = nc; h->mc_dt = dt; h->mc_dom = w->domain; h->mc_seed_all = hardened;   /* mc_dt: I8 recover; fp16 (Stage 1) => recov_max 0 (drains in-submit) */
    for (int i = 0; i < nc && i < ORK_MAXCORE; i++) { h->mc_subs[i] = subs[i]; h->mc_Pc[i] = Pc[i]; }
    return h;
}

/* ================= HETEROGENEOUS SINGLE-CORE NONBLOCK CHAIN (ork_dyn_begin_seq_i8) =================
 * Run ONE group of int8 ops [matmul + int8 SDP ...] as one core's PC-chain on begin_mc's recipe (mc_ensure
 * mrc/maf + a chain-owned warmed output scratch, clean-before, 64B-aligned program slots, per-op forward
 * descriptor), NONBLOCK, ping-pong OFF (an SDP task is present). The TERMINAL op MUST be a matmul — its int32
 * 0x7fffffff last-col sentinel gates completion (int8 SDP output has no free poison). ork_dyn_seq_end() polls
 * the terminal + does per-op copy-back (matmul int32 dense; SDP int8 EWCUBE de-marshalled). Returns NULL if
 * ineligible (caller then runs the ops via the SW break). Stage 2: MM_I8 + EWMUL_I8; ADD/SILU/GELU follow.
 * SINGLE group / single core here; the scheduler slices a sequence into groups and (Stage 3) spreads them. */
/* per-op eligibility (Stage 2/3 envelope); records dom of the first matmul weight into *dom/*have_dom */
int orki_seq_op_ok(const ork_seq_op *o, unsigned *dom, int *have_dom){
    if(o->kind==ORK_OP_MM_I8){ ork_w *w=o->w;
        if(!w||w->dtype!=DT_I8||w->Sn!=1||w->Sk!=1||w->K%512||w->K>4096) return 0;
        if(o->M<1||o->M>64) return 0;
        if(!*have_dom){ *dom=w->domain; *have_dom=1; } else if((unsigned)w->domain!=*dom) return 0;
        return 1; }
    if(o->kind==ORK_OP_EWMUL_I8){
        if(o->M<1||o->M>64||o->N<16||(o->N&15)) return 0;
        if(o->mult<0||o->mult>0x7fff||o->shift<0||o->shift>31) return 0;
        return 1; }
    if(o->kind==ORK_OP_SILU_I16){                                  /* int16 SiLU: HW-chained activation-LUT SDP task */
        if(o->M<1||o->M>64||o->N<8||(o->N&7)) return 0;            /* atom-8 int16 cube (N%8) */
        return 1; }
    if(o->kind==ORK_OP_SILU_I8){                                   /* int8 SiLU: HW-chained activation-LUT SDP task */
        if(o->M<1||o->M>64||o->N<16||(o->N&15)) return 0;         /* atom-16 int8 cube (N%16) */
        return 1; }
    return 0; }
/* build op `gi` as program `pp` (per-core slot in RC) writing its output at seq_out byte-offset *coff; chain
 * it forward to the core's NEXT program (nx_pp>=0) or terminate (nx_pp<0). Records h's per-op output tracking. */
void orki_seq_build_op(ork_dyn_chain *h, const ork_seq_op *o, int gi, struct buf *RC, struct buf *AF,
                         struct rknpu_task *tks, size_t *astage, size_t *coff, int pp, int nx_pp, int nx_kind, int CBUF){
    uint32_t rc[REGCMD_I8_N]; memset(rc,0,sizeof rc); int rcw, dslot, regcfg, enable;
    if(o->kind==ORK_OP_MM_I8){ ork_w *w=o->w; int K=w->K,N=w->N,M=o->M;
        memcpy((char*)AF->cpu+*astage, o->A, (size_t)M*K); uint32_t adma=(uint32_t)(AF->dma+*astage); *astage+=(size_t)M*K;
        uint32_t wdma = w->Bf ? (uint32_t)w->Bf[0].dma : (uint32_t)w->Bb[0].dma;
        orki_synth_i8(rc,M,K,N,adma,wdma,(uint32_t)(h->seq_out.dma+*coff),1,CBUF,0);
        rcw=REGCMD_I8_N; dslot=216; regcfg=108; enable=0xd;
        h->outptr[gi]=(int32_t*)((char*)h->seq_out.cpu+*coff); h->oM[gi]=M; h->nout[gi]=M*N; h->oesz8[gi]=4; h->ocube[gi]=0;
        h->ooff[gi]=*coff; h->dst[gi]=(int32_t*)o->C; *coff+=(size_t)M*N*4;
    } else if(o->kind==ORK_OP_SILU_I16){ int M=o->M,N=o->N;   /* int16 SiLU activation-LUT SDP task (reads the resident LUT) */
        #define EWCUBEH_S(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)   /* int16 atom-8, 2-byte cube (== M*N*2 bytes for N%8) */
        char *a=(char*)AF->cpu+*astage; const int16_t *ha=(const int16_t*)o->A;
        for(int m=0;m<M;m++)for(int nn=0;nn<N;nn++) *(int16_t*)(a+EWCUBEH_S(m,nn))=ha[m*N+nn];
        uint32_t adma=(uint32_t)(AF->dma+*astage); *astage+=(size_t)M*N*2;
        memcpy(rc,REGCMD_SILU_STD_I16,REGCMD_SILU_STD_I16_N*4); orki_set_mul_geom(rc,REGCMD_SILU_STD_I16_N,M,N);
        orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_SDP_5040,0); orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_SDP_5038,0);
        orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_DST_BASE_ADDR,(uint32_t)(h->seq_out.dma+*coff)); orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_SDP_5018,adma);
        orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_SCALE,0x4000); orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_SHIFT,14); orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_OUT_CVT_OFFSET,0);
        orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_R4110,ORK_SILU16_IDXOFF); orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_BN_ALU_CFG,ORK_SILU16_C4064); orki_setrn(rc,REGCMD_SILU_STD_I16_N,RK_DPU_BN_MUL_CFG,ORK_SILU16_C4068);
        rcw=REGCMD_SILU_STD_I16_N; dslot=138; regcfg=69; enable=0x18;
        h->outptr[gi]=(int32_t*)((char*)h->seq_out.cpu+*coff); h->oM[gi]=M; h->nout[gi]=M*N; h->oesz8[gi]=2; h->ocube[gi]=2;
        h->ooff[gi]=*coff; h->dst[gi]=(int32_t*)o->C; *coff+=(size_t)M*N*2;
        #undef EWCUBEH_S
    } else if(o->kind==ORK_OP_SILU_I8){ int M=o->M,N=o->N;   /* int8 SiLU activation-LUT SDP task (atom-16 cube, reads resident LUT) */
        int8_t *a=(int8_t*)((char*)AF->cpu+*astage); const int8_t *ha=o->A;
        for(int m=0;m<M;m++)for(int nn=0;nn<N;nn++) a[ORK_SEQCUBE(m,nn,M)]=ha[m*N+nn];
        uint32_t adma=(uint32_t)(AF->dma+*astage); *astage+=(size_t)M*N;
        memcpy(rc,REGCMD_SILU_STD,REGCMD_SILU_STD_N*4); orki_set_mul_geom(rc,REGCMD_SILU_STD_N,M,N);
        orki_setrn(rc,REGCMD_SILU_STD_N,RK_SDP_5040,0); orki_setrn(rc,REGCMD_SILU_STD_N,RK_SDP_5038,0);
        orki_setrn(rc,REGCMD_SILU_STD_N,RK_DPU_DST_BASE_ADDR,(uint32_t)(h->seq_out.dma+*coff)); orki_setrn(rc,REGCMD_SILU_STD_N,RK_SDP_5018,adma);
        orki_setrn(rc,REGCMD_SILU_STD_N,RK_DPU_OUT_CVT_SCALE,0x4000); orki_setrn(rc,REGCMD_SILU_STD_N,RK_DPU_OUT_CVT_SHIFT,14); orki_setrn(rc,REGCMD_SILU_STD_N,RK_DPU_OUT_CVT_OFFSET,0);
        orki_setrn(rc,REGCMD_SILU_STD_N,RK_DPU_R4110,ORK_SILU_IDXOFF); orki_setrn(rc,REGCMD_SILU_STD_N,RK_DPU_BN_ALU_CFG,ORK_SILU_C4064); orki_setrn(rc,REGCMD_SILU_STD_N,RK_DPU_BN_MUL_CFG,ORK_SILU_C4068);
        rcw=REGCMD_SILU_STD_N; dslot=138; regcfg=69; enable=0x18;
        h->outptr[gi]=(int32_t*)((char*)h->seq_out.cpu+*coff); h->oM[gi]=M; h->nout[gi]=M*N; h->oesz8[gi]=1; h->ocube[gi]=1;
        h->ooff[gi]=*coff; h->dst[gi]=(int32_t*)o->C; *coff+=(size_t)M*N;
    } else { int M=o->M,N=o->N;
        int8_t *a=(int8_t*)((char*)AF->cpu+*astage), *b=a+(size_t)M*N; const int8_t *ha=o->A,*hb=o->B;
        for(int m=0;m<M;m++)for(int nn=0;nn<N;nn++){ a[ORK_SEQCUBE(m,nn,M)]=ha[m*N+nn]; b[ORK_SEQCUBE(m,nn,M)]=hb[m*N+nn]; }
        uint32_t adma=(uint32_t)(AF->dma+*astage), bdma=(uint32_t)(AF->dma+*astage+(size_t)M*N); *astage+=(size_t)2*M*N;
        memcpy(rc,REGCMD_MUL,REGCMD_MUL_N*4); orki_set_mul_geom(rc,REGCMD_MUL_N,M,N);
        orki_setrn(rc,REGCMD_MUL_N,RK_DPU_DST_BASE_ADDR,(uint32_t)(h->seq_out.dma+*coff)); orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5018,adma); orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5038,bdma);
        orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)o->mult); orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)o->shift);
        orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_OFFSET,0); orki_setrn(rc,REGCMD_MUL_N,RK_DPU_BS_ALU_CFG,0); orki_setrn(rc,REGCMD_MUL_N,RK_DPU_EW_CVT_OFFSET,0);
        rcw=REGCMD_MUL_N; dslot=138; regcfg=69; enable=0x18;
        h->outptr[gi]=(int32_t*)((char*)h->seq_out.cpu+*coff); h->oM[gi]=M; h->nout[gi]=M*N; h->oesz8[gi]=1; h->ocube[gi]=1;
        h->ooff[gi]=*coff; h->dst[gi]=o->C; *coff+=(size_t)M*N;
    }
    if(nx_pp>=0){ int amt=((nx_kind==ORK_OP_MM_I8?108:69)+3)/2; uint64_t nx=RC->dma + (size_t)nx_pp*REGCMD_I8_N*4;
        rc[dslot]  =0x0010 | ((uint32_t)(nx&0xffff)<<16); rc[dslot+1]=(0x0101u<<16)|(uint32_t)((nx>>16)&0xffff);
        rc[dslot+2]=0x0014 | ((uint32_t)amt<<16);         rc[dslot+3]=(0x0101u<<16); }
    memcpy((char*)RC->cpu+(size_t)pp*REGCMD_I8_N*4, rc, (size_t)rcw*4);
    tks[pp].enable_mask=enable; tks[pp].int_mask=0x300; tks[pp].int_clear=0x1ffff; tks[pp].regcfg_amount=regcfg;
    tks[pp].regcmd_addr=(uint32_t)(RC->dma+(size_t)pp*REGCMD_I8_N*4);
}
/* MULTI-CORE grouped heterogeneous chain. groups = [gstart[g],gstart[g+1]) contiguous slices of ops; each group
 * is a dependent sub-chain (sequential) whose TERMINAL op is a matmul. Groups are INDEPENDENT — assigned whole
 * to cores (greedy least-loaded, per-CHAIN not per-op) so a core runs its groups back-to-back as one PC-chain
 * while other cores run their groups in parallel. Each core's LAST program (a matmul) carries the completion
 * sentinel. Returns NULL if ineligible. */
/* Single-group / single-core convenience wrapper (Stage 2 API): one dependent chain on one core. */
/* Drain a seq chain: poll the terminal matmul sentinel, then per-op copy-back (matmul int32 dense; SDP int8
 * EWCUBE de-marshalled to row-major). Frees the chain. 0/ok, -1 timeout (partial), -2 bad-arg. */
/* SPIN-KEEP-ALIVE PROBE — validates the persistent-job mechanism. Program 0 is a CIRCULAR spin (its next
 * descriptor points back to itself) writing a dedicated spin slot, keeping the job alive on one core WITHOUT
 * completing. Programs 1..S are the real chain (terminate at S). Submit NONBLOCK once; the sequencer loops on
 * program 0. After spin_us, redirect program 0's next-pointer into program 1 — the sequencer flows into the
 * real chain and runs 1..S. A lost redirect race just re-loops program 0 (NO abort — the safety win vs a
 * terminator frontier). Returns #real outputs completed (0..S); *spin_alive = spin ran but real outputs
 * stayed untouched during the spin window (the loop parked as intended). */
/* Highest op index whose output has landed in DRAM (doorbell), or -1 if none yet. Non-blocking. */
/* Chain-aware anomaly dump. Uses the doorbell DETECTOR (ork_dyn_progress) to name the STUCK descriptor — the
 * first op that did NOT land = progress+1 — and extracts THAT op's context: its regcmd slot DMA address, baked
 * output C address + current doorbell value, and the in-regcmd next-descriptor words 216..219 (feed to
 * tools/re/decode_reg for a register post-mortem). Plus the per-op doorbell map and the context-level
 * ork_npu_dump_state (freq/volt/hw_elapse/int_status). Fire on an anomaly BEFORE a wedge/reboot loses it.
 * Single-core chains (mc uses per-core regcmd buffers — only the map + context are shown there). */
/* ---- Budget accounting: a submit runs a FIXED step count and the NPU STOPS at the end (program P-1's
 * terminator). Work longer than one chain must be split into successive begin() calls; these let the caller
 * size chunks (max_steps) and know how close the running chain is to its end (remaining) so it can plan the
 * next submit without an unintended break. (Single-stream NPU can't overlap submits, so the wrap costs one
 * submit floor ~167us per chain — negligible at realistic step counts.) */
/* WRAP primitive — EXPERIMENTAL / NOT PRODUCTION-SAFE. Extends a RUNNING chain in-flight by filling `task`
 * into the next reserved slot and rewriting the current terminator into a continue-descriptor. The sequencer
 * reads descriptors from DRAM at exec-time, so IF the rewrite lands before the sequencer reaches the frontier
 * it walks on. BUT this is a hard race against the kernel's fixed-task_number job model: if the rewrite lands
 * AFTER the sequencer passed the frontier, the job aborts (ret -22, ~10s timeout, NPU soft-reset) — MEASURED.
 * So this can WEDGE the NPU on a lost race. The robust wrap is chunk + resubmit (each chunk a clean job;
 * inter-chunk bubble ~one submit floor, <0.1% at large chunks). Kept for research; gate behind a reserved
 * orki_budget (ORK_DYN_RESERVE) and keep the fill frontier well ahead of ork_dyn_progress. Returns 1 (too late),
 * 0 (ok), <0 (error). Same v1 op constraints as begin (M=1, DT_I8, K%512==0 && K<=4096, N matches, C resident). */
/* Halt the chain AFTER program `at` (frees the NPU early) by zeroing its next-amount in the live regcmd DRAM.
 * Must lead the sequencer (at > current progress by ~1-2). Returns 0/ok, -1 bad arg. */
/* Resubmit an mc int8 round whose outputs never landed. The concurrent per-core NONBLOCK dispatch occasionally
 * DROPS the whole round (every submit returns rc=0, yet no output sentinel ever clears — the ~1/2000-4000
 * intermittent race; note dma_rw/int_status read 0-always on this kernel so "not dispatched" can't be proven,
 * only "never landed"). RESET clears the lost dispatch/job state, then re-clean (cold coherency) + re-seed +
 * resubmit from the stashed per-core submits (c->maf/mrc/mtk[i] still hold this round's program — the chain owns
 * them until end()). int8 (esz=4, int32 SENT) AND int4 (esz=2, full-surface int16 SENT16); validated by mc_miss_repro. */
int orki_i4_submit_tmo_ms(void);   /* #54 fwd decl: bounded int4 doorbell submit timeout (TCLEAN reap precondition); defined near the int4 workers */
void orki_mc_recover_resubmit(ork_dyn_chain *h){
    ork_npu *c = h->c; int fd = c->fd;
    if(getenv("ORK_MC_DIAG")) fprintf(stderr,"[mc-recover] doorbell MISS -> ACT_RESET + resubmit | dom=%d S=%d nc=%d esz=%d\n", h->mc_dom, h->S, h->mc_nc, h->esz);
    struct rknpu_action a; memset(&a, 0, sizeof a); a.flags = RKNPU_ACT_RESET; ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &a);
    { struct timespec ts = {0, 1000000}; nanosleep(&ts, NULL); }   /* let the reset fully settle before resubmit — a resubmit into a not-yet-quiesced NPU re-drops (sticky miss) */
    /* #54 MULTI-DOMAIN: the ACT_RESET above DROPS the NPU's IOMMU domain state. Resubmitting into a non-0
     * mc_dom then triggers a domain switch that TIMES OUT ("switch iommu domain time out, id: N") and poisons
     * all subsequent switches (dmesg-confirmed cascade). Re-establish mc_dom's page table with a fresh native
     * anchor BEFORE the resubmit so the switch lands cleanly. No-op for domain 0 (always established). */
    if(h->mc_dom > 0) ork_dom_reanchor(c, h->mc_dom);
    struct buf *cl[1024]; int ncl = 0;                                    /* re-clean output surfaces to DRAM */
    for (int x = 0; x < h->S; x++) { struct buf *b = h->outbuf[x]; int seen = 0;
        for (int j = 0; j < ncl; j++) if (cl[j] == b) seen = 1;
        if (!seen && b) { orki_bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE); if (ncl < 1024) cl[ncl++] = b; } }
    if (h->esz == 2) {   /* int4: int16 output, full-surface SENT16 (write-order not last-col-last) */
        for (int x = 0; x < h->S; x++) { int no = h->nout[x]; volatile int16_t *o = (volatile int16_t*)h->outptr[x];
            for (int e = 0; e < no; e++){ o[e] = ORK_DYN_SENT16; __asm__ volatile("dc cvac,%0"::"r"(&o[e]):"memory"); } }
    } else for (int x = 0; x < h->S; x++) { int Mx = h->oM[x]?h->oM[x]:1, Nx = h->nout[x]?h->nout[x]/Mx:h->N;   /* re-seed sentinels */
        if (h->mc_seed_all) for (int m=0;m<Mx;m++) for (int n=0;n<Nx;n++){ volatile int32_t *db=(volatile int32_t*)(h->outptr[x]+(size_t)m*Nx+n); *db=ORK_DYN_SENT; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); }
        else for (int m=0;m<Mx;m++){ volatile int32_t *db=(volatile int32_t*)(h->outptr[x]+(size_t)m*Nx+(Nx-1)); *db=ORK_DYN_SENT; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); } }
    __asm__ volatile("dsb ish":::"memory");
    for (int i = 0; i < h->mc_nc && i < ORK_MAXCORE; i++) if (h->mc_Pc[i]) {   /* resubmit each core */
        orki_bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd, &c->mrc[i], RKNPU_MEM_SYNC_TO_DEVICE);
        orki_bsync(fd, &c->mtk[i], RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
        h->mc_subs[i].timeout = (h->esz==2) ? orki_i4_submit_tmo_ms() : orki_mm_timeout_ms(); orki_rknpu_submit_ioctl(fd, &h->mc_subs[i], h->mc_dom); }   /* #54 int4 (esz==2): bounded timeout so a re-dropped recover job stays reapable (TCLEAN) */
}
/* Drain (until complete or a stall => halted), write outputs back from DMA, free. Returns highest op done. */

/* ============ SUBMIT QUEUE: chunk-pipeline over the dynamic API (the robust wrap) ==================
 * Accumulate matmul tasks, submit them as clean task_number-bounded chunks, and let the NPU run a chunk
 * NONBLOCK while the caller does other work (CPU int4 bulk in the decode split). This is the production
 * wrap: each chunk is a complete job (no in-flight extension), work > chunk_max splits into successive
 * chunks (inter-chunk bubble ~one submit floor, <0.1% at big chunks). Usage:
 *   q = ork_dyn_queue_create(c, chunk_max);
 *   for (...) ork_dyn_queue_push(q, &task);   // accumulate the NPU's share
 *   ork_dyn_queue_flush(q);                    // NPU starts (NONBLOCK)
 *   ... CPU does its bulk in parallel ...
 *   n = ork_dyn_queue_drain(q);                // rendezvous + writeback
 *   ork_dyn_queue_destroy(q); */
/* ncore<=1 => single-core chain (begin); ncore>1 (or ORK_DYN_MC) => multi-core NONBLOCK stream (begin_mc). */
/* submit the next pending chunk NONBLOCK (NPU runs while the caller works); no-op if one is already flying */
/* Idle-transition halt (the linger wiring): once the producer has drained the queue AND the linger window has
 * elapsed since the last push, null-terminate the flying chain just ahead of the sequencer (0x0014=0 via the
 * validated ork_dyn_halt) so a chain with unspent reserve/spin ahead of the frontier stops early and the NPU
 * goes idle instead of running out its reserved budget. linger_us is the grace window before giving up on more
 * work arriving. No-op (returns 0) if nothing is flying, work is still pending, we are within the linger window,
 * the chain is multi-core (halt is single-buffer only — mc self-terminates per-core), or the frontier is already
 * at the terminator. Returns 1 iff it halted. (Visible effect only for a reserved/persistent chain: a plain
 * self-terminating chunk already stops at its own frontier; this is a no-op for it, by design.) */
/* drain: finish the flying chunk + submit/finish any remaining chunks, writeback; returns total ops completed */

/* ========= PRECOMPILED-PROGRAM CACHE (regime A: fixed chain, pinned buffers) =====================
 * The per-token host-build (~95us of the submit floor) is synth_i8 + validate + memcpy of ~108 regs PER
 * program. For a FIXED decode chain the program is identical every token (same weight/shapes/addresses) —
 * only the activation *contents* change. So COMPILE the chain ONCE into a program pool (like precompiling a
 * static regcmd graph), and RUN it every token with just an A-refresh + submit — no synth, no validate. The
 * task list references the pool (the "program domain" indirection); programs need not be contiguous.
 * v1: M=1, DT_I8, K%512==0 && K<=4096, C resident. A is staged into a per-program FIXED scratch (zero-copy A
 * miscomputes at M=1); ork_pc_run refreshes that scratch from the caller's A source each token. Single-core. */

/* Re-run the precompiled chain: refresh A contents from the caller's (fixed-address) source, submit NONBLOCK,
 * poll doorbells, writeback. No synth/validate. Returns highest completed op, -1 on submit error. */

/* CHAIN ASSEMBLER CORE: submit N pre-built HETEROGENEOUS programs as ONE PC-chain (task_number=N, one ioctl).
 * Packs the programs contiguously into c->regcmd (content-driven stride, per AGENTS.md); for each non-last
 * program it WRITES that program's PC next-descriptor at its designated desc_slot (word index, e.g. matmul
 * =216 like run_chain_i8) — reg 0x0010/0x0101 = next program's regcmd dma + the 0x0014 next register-amount
 * ((next regcfg_amount+3)/2). The slot is created by the chaining code, not a pre-existing template pattern.
 * One rknpu_task per program (its own enable_mask +
 * regcfg_amount), ping-pong OFF (flags=0x1, LUT-safe). Generalizes run_chain_i8 (matmul-only, task-strided)
 * and the Phase-0 matmul->silu chain into an arbitrary op sequence — the core the FFN / attention-block
 * static chains build on. Caller pre-builds each program's regcmd (with its own buffer addresses) and any
 * LUT-loads are submitted separately first. Returns 0/ok, -2 bad args or a middle program lacks a descriptor
 * slot (can't be chained as non-last), -1 submit wedge. dom = submit IOMMU domain. */
int ork_npu_chain_progs(ork_npu *c, int n, const ork_chain_prog *progs, int dom){
    if(!c||!progs||n<1||n>1024) return -2;
    int fd=c->fd;
    /* warm-state management (mirror run_chain_i8): entering a chain from a non-int8-live mode resets +
     * unwarms so the WARM-UP reps below fire; DT_I8<->DT_I8_CHAIN is not a real mode change (keepwarm). */
    ork_npu_enter(c, 3 /* DT_I8_CHAIN */, XP_CHAIN_NT, OCK_HW);
    size_t off[1024], total=0;
    /* Each task's regcmd MUST start on a 64-byte (16-word) boundary — the HW chain-walk hangs "entering" a
     * misaligned successor task (vendor RE: tight packing landed a task at +80 mod 128 and hung; the vendor
     * lays every task 64B-aligned). Matmul-only chains never tripped this (REGCMD_I8_N=224w=896B is 64B-aligned,
     * so contiguous packing stays aligned), but a 146-word SDP task (584B) knocks every following task off the
     * boundary. Round each task's start up to 16 words. */
    for(int i=0;i<n;i++){ if(!progs[i].rc||progs[i].nwords<2) return -2; total=(total+15)&~(size_t)15; off[i]=total; total+=(size_t)progs[i].nwords; }
    if(total*4 > c->regcmd.size || (size_t)n*sizeof(struct rknpu_task) > c->task.size) return -2;
    uint32_t *base=(uint32_t*)c->regcmd.cpu;
    for(int i=0;i<n;i++){
        memcpy(base+off[i], progs[i].rc, (size_t)progs[i].nwords*4);
        if(i<n-1){
            /* WRITE this program's PC next-descriptor at its designated slot (like run_chain_i8's word 216).
             * The slot is not a pre-existing pattern in the template -- the chaining code creates it. */
            int slot=progs[i].desc_slot;
            if(slot<0 || slot+3>=progs[i].nwords) return -2;   /* this op can't be a MIDDLE program */
            uint32_t *rc=base+off[i];
            uint64_t nx=(uint64_t)c->regcmd.dma + off[i+1]*4; int nreg=(progs[i+1].regcfg_amount+3)/2;
            rc[slot]  =0x0010 | ((uint32_t)(nx&0xffff)<<16);
            rc[slot+1]=(0x0101u<<16) | (uint32_t)((nx>>16)&0xffff);
            rc[slot+2]=0x0014 | ((uint32_t)nreg<<16);
            rc[slot+3]=(0x0101u<<16);
        }
    }
    orki_bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *t=c->task.cpu; memset(t,0,(size_t)n*sizeof(struct rknpu_task));
    for(int i=0;i<n;i++){ t[i].enable_mask=progs[i].enable_mask; t[i].int_mask=0x300; t[i].int_clear=0x1ffff;
        t[i].regcfg_amount=progs[i].regcfg_amount; t[i].regcmd_addr=(uint32_t)((uint64_t)c->regcmd.dma + off[i]*4); }
    orki_bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    /* ping-pong (ork_ppflags, typically 0x5) is safe ONLY for register-config-only chains (all int8 matmul
     * tasks, like run_chain_i8); ANY SDP/LUT task (enable != 0xd) needs ping-pong OFF (0x1) so a bank swap
     * doesn't race a LUT SRAM commit (AGENTS.md "ping-pong OFF for LUT chains"). */
    int has_sdp=0; for(int i=0;i<n;i++) if(progs[i].enable_mask!=0xd) has_sdp=1;
    struct rknpu_submit s; memset(&s,0,sizeof s);
    s.flags = has_sdp ? 0x1 : ork_ppflags(); s.task_number=(uint32_t)n; s.task_obj_addr=c->task.obj;
    s.core_mask=RKNPU_CORE0_MASK; s.fence_fd=-1; s.timeout=orki_ew_timeout_ms();
    s.subcore_task[0]=s.subcore_task[1]=s.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)n};
    /* WARM-UP: a COLD int8-matmul chain submit yields EMPTY output (run_chain_i8 reps=2 cold). Submit reps
     * times; the last produces the result. c->warmed is managed by the DT_I8_CHAIN block at entry. */
    int reps = c->warmed ? 1 : 2, rr=0;
    if(getenv("ORK_CHAIN_DBG")) fprintf(stderr,"[chain_progs] n=%d dom=%d flags=0x%x warmed(pre)=%d reps=%d regcmd.dma=0x%llx task.obj=0x%llx | "
        "t0{en=0x%x rcfg=%d addr=0x%x} t%d{en=0x%x rcfg=%d addr=0x%x}\n", n,dom,s.flags,reps,reps,(unsigned long long)c->regcmd.dma,(unsigned long long)c->task.obj,
        t[0].enable_mask,t[0].regcfg_amount,t[0].regcmd_addr, n-1,t[n-1].enable_mask,t[n-1].regcfg_amount,t[n-1].regcmd_addr);
    for(int rep=0; rep<reps; rep++){ int e=orki_rknpu_submit_ioctl(fd,&s,dom); rr = e?-1:0;
        if(getenv("ORK_CHAIN_DBG")) fprintf(stderr,"[chain_progs] submit rep %d -> %d (errno=%d)\n",rep,e,errno); }
    c->warmed = 1;
    return rr;
}

/* STAGE 1 PROBE — heterogeneous NONBLOCK chain on the begin_mc RECIPE (not chain_progs). Builds [matmul ->
 * ewmul(int8 SDP, middle) -> matmul] as ONE core's PC-chain in mrc[0]/maf[0] + a warmed OUTPUT scratch (fresh
 * bcreate + clean-before, exactly like begin_mc's cold mcc — the 79f809c coherency fix that chain_progs never
 * got), NONBLOCK submit (ping-pong OFF for the SDP), completion via the TERMINAL matmul's int32 sentinel poll.
 * Proves the SDP-doorbell mechanism produces bit-exact matmul AND ewmul output (the thing the chain_progs-based
 * nb probe could not — its matmuls were empty on the superseded fresh-buffer path). *ok = all three bit-exact. */
int ork_npu_probe_seq_hetero(ork_npu *c, int *ok){
    if(ok)*ok=0;
    if(!c||!ork_ppu_fuse_enabled(c)) return -3;
    int fd=c->fd, M=8, K=512, N=64, mult=0x4000, shift=14, CBUF=c->soc->cbuf_elems;
    #define SEWC(m,n) (((n)/16)*(M*16) + (m)*16 + ((n)%16))
    if(orki_mc_ensure(c,1)) return -1;
    ork_npu_enter(c, 3 /*DT_I8_CHAIN*/, XP_CHAIN_NT, OCK_HW);
    /* pack an all-ones int8 weight [K,N] -> C = K everywhere */
    int8_t *wb=malloc((size_t)K*N); if(!wb) return -2; for(int i=0;i<K*N;i++) wb[i]=1;
    ork_w *w=ork_mm_pack_i8(c,K,N,wb); free(wb); if(!w) return -2;
    uint32_t wdma = w->Bf ? (uint32_t)w->Bf[0].dma : (uint32_t)w->Bb[0].dma;
    /* ewmul inputs + CPU ref (int8) */
    int8_t r1[512],s1[512],ref1[512]; uint32_t g=555;
    for(int i=0;i<M*N;i++){ r1[i]=(int8_t)(((g=g*1103515245u+12345u)>>20&0x7))-3; s1[i]=(int8_t)(((g=g*1103515245u+12345u)>>20&0x7))-3; }
    for(int i=0;i<M*N;i++){ long v=lround((long)r1[i]*s1[i]*mult/(double)(1<<shift)); ref1[i]=(int8_t)(v>127?127:v<-128?-128:v); }
    /* stage into maf[0]: matmul A [M,K] all-ones @0 (shared by both matmuls); ewmul A/B cube-laid after it */
    struct buf *AF=&c->maf[0], *RC=&c->mrc[0];
    size_t offA=0, offEwA=(size_t)M*K, offEwB=offEwA+(size_t)M*N;
    if(offEwB+(size_t)M*N > AF->size) { return -2; }
    memset(AF->cpu,0,offEwB+(size_t)M*N);
    { int8_t*a=(int8_t*)AF->cpu; for(int i=0;i<M*K;i++) a[offA+i]=1;
      for(int m=0;m<M;m++)for(int n=0;n<N;n++){ a[offEwA+SEWC(m,n)]=r1[m*N+n]; a[offEwB+SEWC(m,n)]=s1[m*N+n]; } }
    orki_bsync(fd,AF,RKNPU_MEM_SYNC_TO_DEVICE);
    /* output scratch: matmul0 [M,N]i32 @0, ewmul [M,N]i8 @2048, matmul2 [M,N]i32 @2560 */
    size_t oMM0=0, oEW=(size_t)M*N*4, oMM2=oEW+(size_t)M*N;
    struct buf OUT=orki_bcreate(fd,8192,0x403,c->dom_active); if(!OUT.cpu) return -2;
    memset(OUT.cpu,0,8192);
    uint32_t o0=(uint32_t)(OUT.dma+oMM0), oe=(uint32_t)(OUT.dma+oEW), o2=(uint32_t)(OUT.dma+oMM2);
    /* build 3 programs at 224-word (64B-aligned) slots in mrc[0] */
    uint32_t *base=(uint32_t*)RC->cpu; memset(base,0,3*(size_t)REGCMD_I8_N*4);
    uint32_t am=(uint32_t)(AF->dma+offA);
    { uint32_t rc[REGCMD_I8_N]; memset(rc,0,sizeof rc);
      orki_synth_i8(rc,M,K,N,am,wdma,o0,1,CBUF,0);                                  /* prog0 matmul -> o0 */
      uint64_t nx=RC->dma + (size_t)1*REGCMD_I8_N*4; int amt=(69+3)/2;         /* -> prog1 (SDP regcfg 69) */
      rc[216]=0x0010|((uint32_t)(nx&0xffff)<<16); rc[217]=(0x0101u<<16)|(uint32_t)((nx>>16)&0xffff);
      rc[218]=0x0014|((uint32_t)amt<<16);         rc[219]=(0x0101u<<16);
      memcpy(base+0*REGCMD_I8_N, rc, REGCMD_I8_N*4); }
    { uint32_t rc[REGCMD_MUL_N]; memcpy(rc,REGCMD_MUL,sizeof rc); orki_set_mul_geom(rc,REGCMD_MUL_N,M,N);
      orki_setrn(rc,REGCMD_MUL_N,RK_DPU_DST_BASE_ADDR,oe); orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5018,(uint32_t)(AF->dma+offEwA)); orki_setrn(rc,REGCMD_MUL_N,RK_SDP_5038,(uint32_t)(AF->dma+offEwB));
      orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)mult); orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)shift);
      orki_setrn(rc,REGCMD_MUL_N,RK_DPU_OUT_CVT_OFFSET,0); orki_setrn(rc,REGCMD_MUL_N,RK_DPU_BS_ALU_CFG,0); orki_setrn(rc,REGCMD_MUL_N,RK_DPU_EW_CVT_OFFSET,0);
      uint64_t nx=RC->dma + (size_t)2*REGCMD_I8_N*4; int amt=(108+3)/2;        /* -> prog2 (matmul regcfg 108) */
      rc[138]=0x0010|((uint32_t)(nx&0xffff)<<16); rc[139]=(0x0101u<<16)|(uint32_t)((nx>>16)&0xffff);
      rc[140]=0x0014|((uint32_t)amt<<16);         rc[141]=(0x0101u<<16);
      memcpy(base+1*REGCMD_I8_N, rc, REGCMD_MUL_N*4); }
    { uint32_t rc[REGCMD_I8_N]; memset(rc,0,sizeof rc);
      orki_synth_i8(rc,M,K,N,am,wdma,o2,1,CBUF,0);                                  /* prog2 matmul -> o2 (TERMINAL) */
      memcpy(base+2*REGCMD_I8_N, rc, REGCMD_I8_N*4); }
    orki_bsync(fd,RC,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->mtk[0].cpu; memset(tk,0,3*sizeof *tk);
    tk[0].enable_mask=0xd;  tk[0].int_mask=0x300; tk[0].int_clear=0x1ffff; tk[0].regcfg_amount=108; tk[0].regcmd_addr=(uint32_t)(RC->dma+0*REGCMD_I8_N*4);
    tk[1].enable_mask=0x18; tk[1].int_mask=0x300; tk[1].int_clear=0x1ffff; tk[1].regcfg_amount=69;  tk[1].regcmd_addr=(uint32_t)(RC->dma+1*REGCMD_I8_N*4);
    tk[2].enable_mask=0xd;  tk[2].int_mask=0x300; tk[2].int_clear=0x1ffff; tk[2].regcfg_amount=108; tk[2].regcmd_addr=(uint32_t)(RC->dma+2*REGCMD_I8_N*4);
    orki_bsync(fd,&c->mtk[0],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    /* clean-before: whole OUT to DRAM (no dirty CPU line evicts over the NPU writes — begin_mc's cold recipe) */
    orki_bsync(fd,&OUT,RKNPU_MEM_SYNC_TO_DEVICE);
    /* seed the TERMINAL matmul (prog2 @ o2) last-col-per-row int32 sentinel */
    volatile int32_t *t2=(volatile int32_t*)((char*)OUT.cpu+oMM2);
    for(int m=0;m<M;m++){ volatile int32_t*db=&t2[(size_t)m*N+(N-1)]; *db=ORK_DYN_SENT; __asm__ volatile("dc cvac,%0"::"r"(db):"memory"); }
    __asm__ volatile("dsb ish":::"memory");
    struct rknpu_submit s; memset(&s,0,sizeof s);
    s.flags=0x1u|0x2u;   /* PC | NONBLOCK; ping-pong OFF (SDP present) */
    s.task_number=3; s.task_obj_addr=c->mtk[0].obj; s.core_mask=RKNPU_CORE0_MASK; s.fence_fd=-1; s.timeout=orki_mm_timeout_ms();
    s.subcore_task[0]=(struct rknpu_subcore_task){0,3};
    int e=orki_rknpu_submit_ioctl(fd,&s,c->dom_active);
    int okall=0;
    if(e==0){ double t0=ork_now_us(); int landed=0;
        for(;;){ int done=1; for(int m=0;m<M;m++){ volatile int32_t*db=&t2[(size_t)m*N+(N-1)]; __asm__ volatile("dc civac,%0"::"r"(db):"memory"); if(*db==ORK_DYN_SENT){done=0;break;} } if(done){landed=1;break;} if(ork_now_us()-t0>3e6)break; }
        if(landed){ orki_bsync(fd,&OUT,RKNPU_MEM_SYNC_FROM_DEVICE);
            int32_t*c0=(int32_t*)((char*)OUT.cpu+oMM0), *c2=(int32_t*)((char*)OUT.cpu+oMM2); int8_t*ew=(int8_t*)((char*)OUT.cpu+oEW);
            int n0=0,n2=0,ne=0;
            for(int i=0;i<M*N;i++){ if(c0[i]==K)n0++; if(c2[i]==K)n2++; }
            for(int m=0;m<M;m++)for(int n=0;n<N;n++) if(ew[SEWC(m,n)]==ref1[m*N+n]) ne++;
            if(getenv("ORK_SEQ_DBG")) fprintf(stderr,"[seq-hetero] matmul0 #==K=%d/%d  ewmul #match=%d/%d  matmul2 #==K=%d/%d\n",n0,M*N,ne,M*N,n2,M*N);
            okall = (n0==M*N) && (ne==M*N) && (n2==M*N); } }
    if(ok)*ok=okall;
    orki_bdestroy(fd,&OUT);
    #undef SEWC
    return e?-1:0;
}

/* PROBE (int8 SDP on the HW-chain): can a standalone SDP op (ewmul, enable=0x18, regcfg=69) be a MIDDLE program
 * in a PC-chain, walking FORWARD through its next-descriptor? Decode of the vendor's working SDP chain
 * (regcmd_softmax_f16.h SM_TASK0) proved REGCMD_MUL is ALREADY chain-native: its tail (words 138..145) is
 * byte-identical in STRUCTURE to SM_TASK0 — a terminal descriptor at word 138 (=2*regcfg; next-addr 0 / amt 0)
 * followed by the 0x0041/0x0018/0x0081 op-enable trailer. So the port is NOT a template change; it is feeding
 * SDP progs (desc_slot=138) through the PROVEN ork_npu_chain_progs (which already handles SDP: per-prog
 * desc_slot + has_sdp ping-pong-off + reps=2 cold warm-up). Chains [ewmul0(desc_slot=138) -> ewmul1(last)] and
 * verifies BOTH outputs vs the CPU ref: *t0_ok = the middle SDP op computed correctly carrying a forward
 * descriptor; *t1_ok = the chain WALKED forward through the SDP op's slot. Both ok => int8 SDP HW-chains. */
int ork_npu_probe_sdp_chain_fwd(ork_npu *c, int *t0_ok, int *t1_ok){
    if(t0_ok)*t0_ok=0; if(t1_ok)*t1_ok=0;
    if(!c||!ork_ppu_fuse_enabled(c)) return -3;
    int fd=c->fd, M=8, N=64, mult=0x4000, shift=14;
    #define EWC(m,n) (((n)/16)*(M*16) + (m)*16 + ((n)%16))
    size_t sz=(size_t)M*N; if(sz<4096)sz=4096;
    struct buf A0=orki_bcreate(fd,sz,0x403,-1),B0=orki_bcreate(fd,sz,0x403,-1),O0=orki_bcreate(fd,sz,0x403,-1);
    struct buf A1=orki_bcreate(fd,sz,0x403,-1),B1=orki_bcreate(fd,sz,0x403,-1),O1=orki_bcreate(fd,sz,0x403,-1);
    if(!A0.cpu||!B0.cpu||!O0.cpu||!A1.cpu||!B1.cpu||!O1.cpu){ orki_bdestroy(fd,&A0);orki_bdestroy(fd,&B0);orki_bdestroy(fd,&O0);orki_bdestroy(fd,&A1);orki_bdestroy(fd,&B1);orki_bdestroy(fd,&O1); return -2; }
    int8_t r0[512],s0[512],r1[512],s1[512],ref0[512],ref1[512]; uint32_t g=12345;
    for(int i=0;i<M*N;i++){ r0[i]=(int8_t)(((g=g*1103515245u+12345u)>>20&0x7))-3; s0[i]=(int8_t)(((g=g*1103515245u+12345u)>>20&0x7))-3;
                            r1[i]=(int8_t)(((g=g*1103515245u+12345u)>>20&0x7))-3; s1[i]=(int8_t)(((g=g*1103515245u+12345u)>>20&0x7))-3; }
    for(int i=0;i<M*N;i++){ long v0=lround((long)r0[i]*s0[i]*mult/(double)(1<<shift)),v1=lround((long)r1[i]*s1[i]*mult/(double)(1<<shift));
                            ref0[i]=(int8_t)(v0>127?127:v0<-128?-128:v0); ref1[i]=(int8_t)(v1>127?127:v1<-128?-128:v1); }
    memset(A0.cpu,0,sz);memset(B0.cpu,0,sz);memset(O0.cpu,0,sz);memset(A1.cpu,0,sz);memset(B1.cpu,0,sz);memset(O1.cpu,0,sz);
    int8_t*a0=A0.cpu,*b0=B0.cpu,*a1=A1.cpu,*b1=B1.cpu;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ a0[EWC(m,n)]=r0[m*N+n]; b0[EWC(m,n)]=s0[m*N+n]; a1[EWC(m,n)]=r1[m*N+n]; b1[EWC(m,n)]=s1[m*N+n]; }
    orki_bsync(fd,&A0,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&B0,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O0,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&A1,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&B1,RKNPU_MEM_SYNC_TO_DEVICE);orki_bsync(fd,&O1,RKNPU_MEM_SYNC_TO_DEVICE);
    /* build the two ewmul regcmds exactly like the standalone ork_npu_ewmul_i8 (geom + addrs + scale) */
    uint32_t rc0[REGCMD_MUL_N],rc1[REGCMD_MUL_N];
    memcpy(rc0,REGCMD_MUL,sizeof rc0); orki_set_mul_geom(rc0,REGCMD_MUL_N,M,N);
    orki_setrn(rc0,REGCMD_MUL_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O0.dma); orki_setrn(rc0,REGCMD_MUL_N,RK_SDP_5018,(uint32_t)A0.dma); orki_setrn(rc0,REGCMD_MUL_N,RK_SDP_5038,(uint32_t)B0.dma);
    orki_setrn(rc0,REGCMD_MUL_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)mult); orki_setrn(rc0,REGCMD_MUL_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)shift);
    orki_setrn(rc0,REGCMD_MUL_N,RK_DPU_OUT_CVT_OFFSET,0); orki_setrn(rc0,REGCMD_MUL_N,RK_DPU_BS_ALU_CFG,0); orki_setrn(rc0,REGCMD_MUL_N,RK_DPU_EW_CVT_OFFSET,0);
    memcpy(rc1,REGCMD_MUL,sizeof rc1); orki_set_mul_geom(rc1,REGCMD_MUL_N,M,N);
    orki_setrn(rc1,REGCMD_MUL_N,RK_DPU_DST_BASE_ADDR,(uint32_t)O1.dma); orki_setrn(rc1,REGCMD_MUL_N,RK_SDP_5018,(uint32_t)A1.dma); orki_setrn(rc1,REGCMD_MUL_N,RK_SDP_5038,(uint32_t)B1.dma);
    orki_setrn(rc1,REGCMD_MUL_N,RK_DPU_OUT_CVT_SCALE,(uint32_t)mult); orki_setrn(rc1,REGCMD_MUL_N,RK_DPU_OUT_CVT_SHIFT,(uint32_t)shift);
    orki_setrn(rc1,REGCMD_MUL_N,RK_DPU_OUT_CVT_OFFSET,0); orki_setrn(rc1,REGCMD_MUL_N,RK_DPU_BS_ALU_CFG,0); orki_setrn(rc1,REGCMD_MUL_N,RK_DPU_EW_CVT_OFFSET,0);
    /* THE PORT: SDP progs through the proven chainer. ewmul0 is a MIDDLE program (desc_slot=138 = the chain-native
     * descriptor slot decoded from SM_TASK0); ewmul1 is the terminal (desc_slot=-1). chain_progs writes ewmul0's
     * forward descriptor at 138, detects enable!=0xd -> ping-pong OFF, and does the reps=2 cold warm-up. */
    /* HYPOTHESIS: a HW chain must BEGIN with a matmul (enable=0xd) task — the kernel programs the PC from task0,
     * and every working chain (FFN, chain_mm_silu) starts with a matmul; an SDP task0 (this probe's earlier form,
     * and the vendor's hardware-chained softmax) HANGS. So prepend an all-ones matmul task0; ewmul0 becomes a true
     * MIDDLE SDP task (carries the fwd descriptor at 138 like the FFN chain's silu). ORK_SDP_NOMM=1 = old SDP-first
     * form (control). Matmul: M=8,K=64,N=64, A=c->Af all-ones, W all-ones -> C=K=64 (sanity, not read by ewmul). */
    int nomm=!!getenv("ORK_SDP_NOMM"); int CBUF=c->soc->cbuf_elems, MK=8*64, KN=64*64;
    struct buf W=orki_bcreate(fd,(size_t)KN,0x403,-1), C=orki_bcreate(fd,(size_t)8*64*4,0x403,-1);
    static uint32_t rmm[REGCMD_I8_N];
    if(!nomm){
        if(!W.cpu||!C.cpu){ orki_bdestroy(fd,&W);orki_bdestroy(fd,&C);orki_bdestroy(fd,&A0);orki_bdestroy(fd,&B0);orki_bdestroy(fd,&O0);orki_bdestroy(fd,&A1);orki_bdestroy(fd,&B1);orki_bdestroy(fd,&O1); return -2; }
        { int8_t*wb=W.cpu; for(int i=0;i<KN;i++)wb[i]=1; int8_t*ad=c->Af.cpu; for(int i=0;i<MK;i++)ad[i]=1; }
        memset(C.cpu,0,(size_t)8*64*4);
        orki_bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&C,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
        orki_act(fd,RKNPU_ACT_RESET,0);
        orki_synth_i8(rmm,8,64,64,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)C.dma,1,CBUF,0);
    }
    ork_chain_prog progs[3]={ {rmm,REGCMD_I8_N,0xd,108,216}, {rc0,REGCMD_MUL_N,0x18,69,138}, {rc1,REGCMD_MUL_N,0x18,69,-1} };
    /* dom=-1 (default domain) to MATCH the orki_bcreate(...,-1) buffers + c->regcmd/c->task (a domain-0 submit
     * mismatches them, see ork_npu_chain_selftest). nomm control: SDP-first 2-task chain (expected to hang). */
    int rc = nomm ? ork_npu_chain_progs(c,2,progs+1,-1) : ork_npu_chain_progs(c,3,progs,-1);
    if(rc==0){ orki_bsync(fd,&O0,RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd,&O1,RKNPU_MEM_SYNC_FROM_DEVICE);
        int ok0=1,ok1=1;
        for(int m=0;m<M;m++)for(int n=0;n<N;n++){ if(*(int8_t*)((char*)O0.cpu+EWC(m,n))!=ref0[m*N+n])ok0=0; if(*(int8_t*)((char*)O1.cpu+EWC(m,n))!=ref1[m*N+n])ok1=0; }
        if(t0_ok)*t0_ok=ok0; if(t1_ok)*t1_ok=ok1; }
    orki_bdestroy(fd,&W);orki_bdestroy(fd,&C);
    orki_bdestroy(fd,&A0);orki_bdestroy(fd,&B0);orki_bdestroy(fd,&O0);orki_bdestroy(fd,&A1);orki_bdestroy(fd,&B1);orki_bdestroy(fd,&O1);
    #undef EWC
    return rc;
}

/* CHAIN ASSEMBLER self-test: chain TWO plain int8 matmuls via ork_npu_chain_progs and verify BOTH tasks
 * EXECUTE + produce output -- the exact thing Phase-0 could not (an early task0 that actually runs). Uses
 * WORKING primitives only (synth_i8 native int32 output; no set_i16_out). Layout-agnostic: A=all-1, W0=all-1,
 * W1=all-2 -> every C0 element == K, every C1 element == 2K, regardless of output tiling. Fills *t0_cnt/
 * *t1_cnt = count of the M*N int32 slots matching K / 2K (near M*N => that task ran; 0 => it didn't).
 * 0/ok, -1 wedge, -2 dims/alloc. rk3588. */
int ork_npu_chain_selftest(ork_npu *c, int *t0_cnt, int *t1_cnt){
    /* dom=-1 (default domain) to match the WORKING raw-submit paths (probe_i8_mm, softmax_replay); the init
     * buffers c->Af/c->regcmd/c->task live in the default domain, so a domain-0 submit mismatches them. */
    int fd=c->fd, CBUF=c->soc->cbuf_elems, dom=-1;
    const int M=8, K=64, N=64;
    struct buf W0=orki_bcreate(fd,(size_t)K*N,0x403,dom), W1=orki_bcreate(fd,(size_t)K*N,0x403,dom);
    struct buf C0=orki_bcreate(fd,(size_t)M*N*4,0x403,dom), C1=orki_bcreate(fd,(size_t)M*N*4,0x403,dom);
    if(!W0.cpu||!W1.cpu||!C0.cpu||!C1.cpu){ orki_bdestroy(fd,&W0);orki_bdestroy(fd,&W1);orki_bdestroy(fd,&C0);orki_bdestroy(fd,&C1); return -2; }
    { int8_t*b0=W0.cpu,*b1=W1.cpu; for(size_t i=0;i<(size_t)K*N;i++){ b0[i]=1; b1[i]=2; } }   /* uniform -> tile layout irrelevant */
    memset(C0.cpu,0,(size_t)M*N*4); memset(C1.cpu,0,(size_t)M*N*4);
    { int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=1; }
    orki_bsync(fd,&W0,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&W1,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_bsync(fd,&C0,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&C1,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    orki_act(fd,RKNPU_ACT_RESET,0);
    static uint32_t r0[REGCMD_I8_N], r1[REGCMD_I8_N];
    orki_synth_i8(r0,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W0.dma,(uint32_t)C0.dma,1,CBUF,0);
    orki_synth_i8(r1,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W1.dma,(uint32_t)C1.dma,1,CBUF,0);
    ork_chain_prog progs[2]={ {r0,REGCMD_I8_N,0xd,108,216}, {r1,REGCMD_I8_N,0xd,108,-1} };
    int nprog = getenv("ORK_GS_N1") ? 1 : 2;   /* ORK_GS_N1: single-task chain_progs (isolate chaining from the matmul itself) */
    int crc=ork_npu_chain_progs(c,nprog,progs,dom);
    if(crc){ orki_bdestroy(fd,&W0);orki_bdestroy(fd,&W1);orki_bdestroy(fd,&C0);orki_bdestroy(fd,&C1); return crc==-1?-1:-2; }
    orki_bsync(fd,&C0,RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd,&C1,RKNPU_MEM_SYNC_FROM_DEVICE);
    int n0=0,n1=0; int32_t*c0=C0.cpu,*c1=C1.cpu;
    int32_t mx0=0,mx1=0; for(int i=0;i<M*N;i++){ if(c0[i]==K)n0++; if(c1[i]==2*K)n1++;
        if(c0[i]>mx0)mx0=c0[i]; if(c1[i]>mx1)mx1=c1[i]; }
    fprintf(stderr,"[selftest] K=%d 2K=%d | C0 max=%d first=[%d %d %d %d] | C1 max=%d first=[%d %d %d %d]\n",
            K,2*K,mx0,c0[0],c0[1],c0[2],c0[3],mx1,c1[0],c1[1],c1[2],c1[3]);
    if(t0_cnt)*t0_cnt=n0; if(t1_cnt)*t1_cnt=n1;
    orki_bdestroy(fd,&W0);orki_bdestroy(fd,&W1);orki_bdestroy(fd,&C0);orki_bdestroy(fd,&C1);
    return 0;
}

/* FUSED SSD-SCAN MATMUL BENCH: chain ALL grouped-scan matmuls of ONE Mamba-2/SSD layer into a SINGLE
 * PC-chained submit with RESIDENT all-ones operands (no per-batch repack), vs the SAME matmuls as N
 * separate submits (each paying the ~48µs per-submit floor). Isolates the floor-amortization the fused
 * on-NPU scan graph would get. All-ones int8 => every output element == its own K, so a fused-chain
 * output that differs signals a wedge/miscompute. Group-batched shapes (L-factorization; Y_diag grouped):
 *   scores  [CS, Nst]x[Nst, CS]      cstate  [HG*P, CS]x[CS, Nst]
 *   Ydiag_g [CS, CS]x[CS, HG*P]      Y_off   [HG*P, Nst]x[Nst, CS]
 * each x (G*NC) batches. M<=HG*P fits ONE M-tile at these K (mcap(K=64)=10496). Returns 0/ok, <0 on
 * error; fills *fused_us/*persub_us (per-iter wall) and *ok_out (1 = fused chain bit-correct). Board only. */
int ork_ssd_fused_scan_bench(ork_npu *c,int H,int P,int Nst,int G,int CS,int NC,int iters,int dtype,int perhead,
                             double *fused_us,double *persub_us,int *ok_out){
    if(!c) return -1; if(iters<1) iters=1;
    int HG=H/G; if(HG<1)HG=1;
    int fd=c->fd, CBUF=c->soc->cbuf_elems, dom=-1;
    int f16 = (dtype==DT_F16);
    int esz = f16 ? 2 : 1;                 /* A/B element size (fp16=2B, int8=1B); C is 4B either way */
    int gb=G*NC;
    /* per-stage (nb, M, K, N). scores/cstate/Y_off group-batched (fp16-stable). Y_diag: perhead=1 uses
     * the fp16-STABLE bounded per-head form (nb=H*NC, [CS,CS]x[CS,P]); perhead=0 uses the fast group-
     * batched L-factored form (nb=G*NC, [CS,CS]x[CS,HG*P]) — faster but OVERFLOWS fp16 at large chunk decay. */
    int snb[4]={gb, gb, gb, perhead?H*NC:gb};
    int sM[4] ={CS, HG*P, HG*P, CS};
    int sK[4] ={Nst, CS,  Nst,  CS};
    int sN[4] ={CS, Nst,  CS,   perhead?P:HG*P};
    /* TILE cap: a single synth program mis-writes some dims at width==1024 (power-of-2 ISA quirk in the raw
     * synth path — run_i8 avoids it via its own tiling). Cap per-program M and N at TILE, M/N-tiled programs. */
    int TILE=512;
    int np=0; for(int s=0;s<4;s++){ int nm=(sM[s]+TILE-1)/TILE, nn=(sN[s]+TILE-1)/TILE; np+=nm*nn*snb[s]; }
    if(np<1||np>1024) return -2;
    int *tM=malloc(np*sizeof(int)),*tK=malloc(np*sizeof(int)),*tN=malloc(np*sizeof(int));
    size_t *cOff=malloc(np*sizeof(size_t));
    if(!tM||!tK||!tN||!cOff){ free(tM);free(tK);free(tN);free(cOff); return -3; }
    size_t maxA=0,maxB=0,totC=0; int t=0;
    for(int s=0;s<4;s++) for(int b=0;b<snb[s];b++){
        int fullM=sM[s], fullN=sN[s];
        for(int m0=0;m0<fullM;m0+=TILE) for(int n0=0;n0<fullN;n0+=TILE){
            int Mc=(fullM-m0<TILE)?(fullM-m0):TILE, Nc=(fullN-n0<TILE)?(fullN-n0):TILE;
            tM[t]=Mc; tK[t]=sK[s]; tN[t]=Nc;
            cOff[t]=totC; totC+=(size_t)Mc*Nc;                 /* each tile -> its own dense [Mc,Nc] region */
            size_t a=(size_t)Mc*sK[s], bb=(size_t)sK[s]*Nc;
            if(a>maxA)maxA=a; if(bb>maxB)maxB=bb; t++;
        }
    }
    struct buf Ab=orki_bcreate(fd,maxA*esz,0x403,dom), Bb=orki_bcreate(fd,maxB*esz,0x403,dom), Cb=orki_bcreate(fd,totC*4,0x403,dom);
    uint32_t *rcs=malloc((size_t)np*REGCMD_I8_N*4);
    ork_chain_prog *progs=malloc(np*sizeof(ork_chain_prog));
    int ret=0;
    if(!Ab.cpu||!Bb.cpu||!Cb.cpu||!rcs||!progs){ ret=-3; goto done; }
    if(f16){ uint16_t*pa=Ab.cpu,*pb=Bb.cpu; for(size_t i=0;i<maxA;i++)pa[i]=0x3c00; for(size_t i=0;i<maxB;i++)pb[i]=0x3c00; }  /* fp16 1.0 */
    else   { memset(Ab.cpu,1,maxA); memset(Bb.cpu,1,maxB); }
    memset(Cb.cpu,0,totC*4);
    orki_bsync(fd,&Ab,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Bb,RKNPU_MEM_SYNC_TO_DEVICE); orki_bsync(fd,&Cb,RKNPU_MEM_SYNC_TO_DEVICE);
    for(int i=0;i<np;i++){
        uint32_t *rc=rcs+(size_t)i*REGCMD_I8_N;
        uint32_t aC=(uint32_t)(Cb.dma+cOff[i]*4);
        if(f16) orki_synth   (rc,tM[i],tK[i],tN[i],(uint32_t)Ab.dma,(uint32_t)Bb.dma,aC,1,CBUF);      /* fp16, dense [M,Nc] out */
        else    orki_synth_i8(rc,tM[i],tK[i],tN[i],(uint32_t)Ab.dma,(uint32_t)Bb.dma,aC,1,CBUF,0);    /* int8, dense [M,Nc] out */
        progs[i]=(ork_chain_prog){rc,REGCMD_I8_N,0xd,108,216};
    }
    /* fp16 needs the NPU in fp16 mode: force a reset on entry (the int8-oriented chain assembler keeps
     * warm across ORK_I8_LIVE markers, so an int8->fp16 switch would otherwise skip the reset). */
    if(f16){ orki_act(fd,RKNPU_ACT_RESET,0); c->warmed=0; c->last_dt=DT_F16; }
    /* FUSED: one chained submit of all np matmul programs */
    int rc1=ork_npu_chain_progs(c,np,progs,dom);   /* warm + wedge-check */
    if(rc1){ ret=rc1; goto done; }
    { double f0=ork_now_us(); for(int it=0;it<iters;it++) ork_npu_chain_progs(c,np,progs,dom); *fused_us=(ork_now_us()-f0)/iters; }
    orki_bsync(fd,&Cb,RKNPU_MEM_SYNC_FROM_DEVICE);
    { int okc=1; int32_t*Ci=(int32_t*)Cb.cpu; float*Cf=(float*)Cb.cpu;
      for(int i=0;i<np&&okc;i++){ size_t mn=(size_t)tM[i]*tN[i];
        for(size_t e=0;e<mn;e++){ double got = f16 ? (double)Cf[cOff[i]+e] : (double)Ci[cOff[i]+e];
          if(got!=(double)tK[i]){ if(getenv("ORK_SSD_DBG")) fprintf(stderr,"[ssd_fused] mismatch prog %d/%d M=%d K=%d N=%d elem %zu: got %g exp %d\n",i,np,tM[i],tK[i],tN[i],e,got,tK[i]); okc=0;break; } } }
      if(ok_out)*ok_out=okc; }
    /* PER-SUBMIT: the SAME programs as np separate single-task submits (each pays the floor) */
    { double p0=ork_now_us(); for(int it=0;it<iters;it++) for(int i=0;i<np;i++) ork_npu_chain_progs(c,1,&progs[i],dom); *persub_us=(ork_now_us()-p0)/iters; }
done:
    orki_bdestroy(fd,&Ab);orki_bdestroy(fd,&Bb);orki_bdestroy(fd,&Cb);
    free(rcs);free(progs);free(tM);free(tK);free(tN);free(cOff);
    return ret;
}

/* (b) LAYOUT PROBE: run ONE fp16 matmul through the exact fused-chain mechanism (raw synth + a single
 * ork_npu_chain_progs task) with ROW-MAJOR resident operands A[M,K],B[K,N] → C[M,N] fp32. Decides
 * whether a real-operand fused SSD scan can stage row-major operands directly, or must replicate the
 * ork_mm_pack 32x32-block tiling. K%32,N%16. 0/ok,<0 err. rk3588. Diagnostic — not a production path. */

/* (b) FUSED-MM probe: one fp16 matmul via the fused-chain synth mechanism, but B is PACKED with
 * ork_mm_pack (→ the tiled Bb layout synth actually reads, same as orki_run()) and A is staged ROW-MAJOR,
 * C read DENSE. Determines whether the real-operand fused SSD chain can reuse ork_mm_pack for B + a
 * row-major A (vs needing to hand-tile A). Single-slice only (K<=ks, N<=nmax). C[M,N] fp32. 0/ok,<0. */

/* FUSED batched fp16 GEMM: like ork_bmm_fp16 (nbatch matmuls C[b]=A[b]*B[b], both operands dynamic), but
 * chains ALL nbatch matmuls into ONE PC-chained submit instead of one submit per batch — amortizing the
 * ~48us/submit NPU floor across the batch (the SSD scan's per-stage H-batch = the target). Each B is packed
 * via ork_mm_pack (its tiled Bb is what synth reads); A staged row-major; C dense. A[nb*M*K], B[nb*K*N],
 * C[nb*M*N] fp32. Single-slice only (K<=ks, N<=nmax) — the scan's small shapes qualify; nb<=64. 0/ok,<0.
 * Numerically identical to ork_bmm_fp16 (same synth matmul, same fp16 operands) — just one ioctl. */

/* ---- ASYNC ROUND-ROBIN STREAM (ork_mm_run_stream_i8) ----
 * Mirrors how the closed runtime keeps the 3 cores busy (see wiki Exp-2026-06-24-RKLLM-Multicore-Capture):
 * instead of barrier-splitting ONE chain across cores, dispatch a STREAM of independent matmuls to a pool
 * of per-core workers that PULL the next task dynamically (atomic counter) and run it as a single-core
 * submit on their own core — no barrier, no static partition. A core that finishes early grabs the next
 * task immediately, so the cores pipeline and never idle on a sync point. Each worker owns its per-core
 * buffers (mrc/mtk/maf/mcc); tasks' weights are read-only-shared, outputs are disjoint. */
struct streamw { ork_npu *c; int core; int S; const ork_mm_task_i8 *tasks; int *ctr; int rc; };
static void *stream_worker(void *vp) {
    struct streamw *a = vp; ork_npu *c = a->c; int fd = c->fd, i = a->core, CBUF = c->soc->cbuf_elems;
    orki_pin_big_core(i);
    int k;
    a->rc = 0;
    uint32_t rc[REGCMD_I8_N + 4];
    while ((k = __atomic_fetch_add(a->ctr, 1, __ATOMIC_SEQ_CST)) < a->S) {
        const ork_mm_task_i8 *t = &a->tasks[k];
        ork_w *w = t->w; int M = t->M, K = w->K, N = w->N, mcap = orki_chain_fullk_mcap_i8(c, K);
        uint32_t bdma = w->Bf ? (uint32_t)w->Bf[0].dma : (uint32_t)w->Bb[0].dma;
        memcpy(c->maf[i].cpu, t->A, (size_t)M * K);
        orki_bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE);
        int ntiles = (M + mcap - 1) / mcap, p = 0;
        for (int m0 = 0; m0 < M; m0 += mcap, p++) {
            int mc = (M - m0 < mcap) ? (M - m0) : mcap;
            memset(rc, 0, sizeof rc);
            orki_synth_i8(rc, mc, K, N, (uint32_t)(c->maf[i].dma + (size_t)m0 * K), bdma,
                     (uint32_t)(c->mcc[i].dma + (size_t)m0 * N * 4), 1, CBUF, 0);
            if (p < ntiles - 1) {
                uint64_t nd = c->mrc[i].dma + (size_t)(p + 1) * REGCMD_I8_N * 4;
                rc[216] = 0x0010 | ((nd & 0xffff) << 16); rc[217] = (0x0101 << 16) | ((nd >> 16) & 0xffff);
                rc[218] = 0x0014 | (0x0037 << 16); rc[219] = (0x0101 << 16) | (0);
            }
            memcpy((char *)c->mrc[i].cpu + (size_t)p * REGCMD_I8_N * 4, rc, REGCMD_I8_N * 4);
        }
        orki_bsync(fd, &c->mrc[i], RKNPU_MEM_SYNC_TO_DEVICE);
        struct rknpu_task *mt = c->mtk[i].cpu; memset(mt, 0, (size_t)ntiles * sizeof *mt);
        for (int q = 0; q < ntiles; q++) {
            mt[q].enable_mask = 0xd; mt[q].int_mask = 0x300; mt[q].int_clear = 0x1ffff;
            mt[q].regcfg_amount = 108; mt[q].regcmd_addr = c->mrc[i].dma + (size_t)q * REGCMD_I8_N * 4;
        }
        orki_bsync(fd, &c->mtk[i], RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
        struct rknpu_submit sub; memset(&sub, 0, sizeof sub);
        sub.flags = ork_ppflags(); sub.task_number = ntiles; sub.task_obj_addr = c->mtk[i].obj; sub.core_mask = 1u << i; sub.fence_fd = -1;
        sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)ntiles};
        sub.timeout = orki_mm_timeout_ms();
        /* A freshly-allocated NPU output buffer returns stale on its FIRST write, so prime THIS core's
         * output buffer with a throwaway submit on its first use (mirror mcworker's reps=c->mwarm[i]?1:2).
         * Per-core + deterministic: the old coarse whole-stream 2-pass could miss a core whose buffer the
         * dynamic task counter left idle on the warmup pass, yielding a flaky stale (zero) result. */
        int reps = c->mwarm[i] ? 1 : 2;
        for (int rep = 0; rep < reps; rep++) {
            if (orki_rknpu_submit_ioctl(fd, &sub, w->domain)) { if (rep == reps - 1) a->rc = -1; continue; }
            orki_bsync(fd, &c->mcc[i], RKNPU_MEM_SYNC_FROM_DEVICE);
        }
        c->mwarm[i] = 1;   /* this core's buffer index is disjoint per worker — no cross-thread race */
        memcpy(t->C, c->mcc[i].cpu, (size_t)M * N * 4);
    }
    return NULL;
}

/* Run S independent int8 matmuls as an async round-robin stream across the NPU cores. Each task's weight
 * must be single-slice (Sk==1 or Bf full-K) and single N-slice; A/C are plain host buffers (copied via the
 * per-core staging buffers — no zero-copy DMA here). Returns 0/ok, -1 submit fail, -2 bad arg. */

/* ---- fp16 ROUND-ROBIN STREAM (ork_mm_run_stream_f16) — fp16 twin of the int8 stream above ----
 * Dynamic·dynamic (both operands activations): weight is pre-packed per task (ork_w, fp16 Bb tiled), A/C
 * copied via per-core staging. Each worker pulls the next task and runs a SINGLE-CORE submit on its own
 * core (core_mask=1<<i) — so nbatch independent matmuls spread across all cores. Single M-tile (the SSD
 * scan is M<=64 <= one tile); K<96 uses sched=0 (the small-K 0x1040 fix). */
/* ---- SMALL-K int8 ROUND-ROBIN STREAM (ork_mm_run_stream_i8_sk) — int8 twin of run_stream_f16 ----
 * run_stream_i8 is full-K only (K%512, 0x1040 schedule); this handles the scan's small single-slice int8
 * matmuls (K%32,N%16) with the SAME 3-core round-robin batching the fp16 scan uses. Per-core: memcpy int8
 * A -> maf, synth_i8, submit on core i, copy int32 C out. Caller quantizes A + dequants int32. */

void *orki_stream_worker_i8sk(void *vp){
    struct streamw_i8sk *a=vp; ork_npu *c=a->c; int fd=c->fd, i=a->core, CBUF=c->soc->cbuf_elems;
    orki_pin_big_core(i);
    int k; a->rc=0;
    uint32_t rc[REGCMD_I8_N];
    while((k=__atomic_fetch_add(a->ctr,1,__ATOMIC_SEQ_CST))<a->S){
        const ork_mm_task_i8 *t=&a->tasks[k]; ork_w *w=t->w; int M=t->M, K=w->K, N=w->N;
        int sched=(K&(K-1))==0 && K>=256 && K<2048;   /* int8 0x1040 sched zeros output for K<256 -> off at small K */
        memcpy(c->maf[i].cpu, t->A, (size_t)M*K); orki_bsync(fd,&c->maf[i],RKNPU_MEM_SYNC_TO_DEVICE);
        memset(rc,0,REGCMD_I8_N*4);
        orki_synth_i8(rc, M, K, N, (uint32_t)c->maf[i].dma, (uint32_t)w->Bb[0].dma, (uint32_t)c->mcc[i].dma, sched, CBUF, N);
        memcpy(c->mrc[i].cpu, rc, REGCMD_I8_N*4); orki_bsync(fd,&c->mrc[i],RKNPU_MEM_SYNC_TO_DEVICE);
        struct rknpu_task *mt=c->mtk[i].cpu; memset(mt,0,sizeof *mt);
        mt[0].enable_mask=0xd; mt[0].int_mask=0x300; mt[0].int_clear=0x1ffff; mt[0].regcfg_amount=108; mt[0].regcmd_addr=c->mrc[i].dma;
        orki_bsync(fd,&c->mtk[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        struct rknpu_submit sub; memset(&sub,0,sizeof sub);
        sub.flags=ork_ppflags(); sub.task_number=1; sub.task_obj_addr=c->mtk[i].obj; sub.core_mask=1u<<i; sub.fence_fd=-1;
        sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,1};
        sub.timeout=orki_mm_timeout_ms();
        int reps=c->mwarm[i]?1:2;
        for(int rep=0;rep<reps;rep++){ if(orki_rknpu_submit_ioctl(fd,&sub,w->domain)){ if(rep==reps-1)a->rc=-1; continue; } orki_bsync(fd,&c->mcc[i],RKNPU_MEM_SYNC_FROM_DEVICE); }
        c->mwarm[i]=1;
        memcpy(t->C, c->mcc[i].cpu, (size_t)M*N*4);   /* int32 output */
    }
    return NULL;
}
/* ---- CHAINED-MULTICORE fp16 stream (ork_mm_run_stream_f16_chain) ----
 * Combines the two half-wins: PC-chaining (task_number>1, one submit amortizes the ~48us submit floor over
 * many programs — like run_chain_i8) AND 3-core parallelism (like run_stream_f16). Static strided partition:
 * core i owns tasks {i, i+nc, ...}; it synths all of them into ITS mrc[i] (each program's PC next-descriptor
 * at word 216 -> 0x0010/0x0014 links to the next), builds a cnt-entry task-descriptor array in mtk[i], and
 * issues ONE task_number=cnt submit on core i. This is the fused graph the scan wanted: N submits -> nc.
 * Matmul-only chain (register-config, no LUT) -> ping-pong safe. Escapes the per-matmul submit floor. */
/* STREAMED batched fp16 GEMM — pack each B (fp16), dispatch the nb matmuls round-robin across cores. */



/* Big-core SET mask (high-numbered cluster, matching pin_big_core's "high = big" assumption). Used to
 * launch the async worker bound to the WHOLE big cluster rather than one core: a single-core pin would
 * trap a freshly-created worker behind the caller (it inherits the caller's mask and can't run to
 * migrate); a set lets the scheduler place it on any FREE big core -> real CPU‖NPU overlap, never an
 * A55. (The run_multicore pool deliberately pins DISTINCT single cores instead — that's a simultaneous
 * barrier where distinct cores avoid contention; the async worker is a single overlapping thread.) */
static void *ork_async_worker(void *p){
    struct ork_async *h = (struct ork_async *)p;
#if defined(__linux__)
    h->c->last_async_cpu = sched_getcpu();   /* record placement (attr-pinned big-core set) for diagnostics/tests */
#endif
    switch (h->kind) {
        case OAK_F16:      h->rc = ork_mm_run        (h->c, h->w, h->M, (const f16*)h->A, (float*)h->C); break;
        case OAK_I8:       h->rc = ork_mm_run_i8     (h->c, h->w, h->M, (const int8_t*)h->A, (int32_t*)h->C); break;
        case OAK_I4:       h->rc = ork_mm_run_i4     (h->c, h->w, h->M, (const int8_t*)h->A, (int32_t*)h->C); break;
        case OAK_CHAIN_I8: h->rc = ork_mm_run_chain_i8 (h->c, h->S, (const ork_mm_task_i8*)h->tasks); break;
        case OAK_CHAIN_I4: h->rc = ork_mm_run_chain_i4 (h->c, h->S, (const ork_mm_task_i4*)h->tasks); break;
        case OAK_STREAM_I8:h->rc = ork_mm_run_stream_i8(h->c, h->S, (const ork_mm_task_i8*)h->tasks); break;
        case OAK_STREAM_I4:h->rc = ork_mm_run_stream_i4(h->c, h->S, (const ork_mm_task_i4*)h->tasks); break;
    }
    return NULL;
}

ork_async *ork_async_launch(struct ork_async tmpl){
    struct ork_async *h = calloc(1, sizeof *h);
    if (!h) return NULL;
    *h = tmpl; h->rc = -1; h->started = 0;
    /* Bind the worker to the big-core SET (not a single core) AT CREATION: a single-core child would
     * inherit the caller's narrow mask and stay trapped on the caller's core; setting the mask *after*
     * pthread_create races (the child can run the whole submit on the caller's core before Linux
     * migrates a sleeping thread). Setting it via pthread_attr means the child is *born* with the
     * big-cluster mask, so the scheduler places it on an idle big core from the first instruction →
     * real CPU‖NPU overlap, never an A55. ORK_NO_AFFINITY (in ork_big_core_set) leaves it unpinned. */
    pthread_attr_t at; pthread_attr_init(&at);
#if defined(__linux__)
    { cpu_set_t s; if (ork_big_core_set(&s)) pthread_attr_setaffinity_np(&at, sizeof s, &s); }
#endif
    int rc = pthread_create(&h->th, &at, ork_async_worker, h);
    pthread_attr_destroy(&at);
    if (rc != 0) { free(h); return NULL; }
    h->started = 1;
    return h;
}

/* Per-path async launchers. Each returns a handle immediately (NULL on bad args -> caller falls back
 * to the matching synchronous run). The returned handle MUST be passed to ork_async_wait exactly once
 * (joins the thread, frees the handle). Numerics are identical to the synchronous orki_run (reused verbatim). */
ork_async *ork_mm_run_async    (ork_npu *c, ork_w *w, int M, const ork_f16 *A, float   *C){
    if (!c || !w || M < 1) return NULL;
    return ork_async_launch((struct ork_async){ .kind=OAK_F16, .c=c, .w=w, .M=M, .A=A, .C=C }); }
ork_async *ork_mm_run_i4_async (ork_npu *c, ork_w *w, int M, const int8_t  *A, int32_t *C){
    if (!c || !w || M < 1) return NULL;
    return ork_async_launch((struct ork_async){ .kind=OAK_I4, .c=c, .w=w, .M=M, .A=A, .C=C }); }
ork_async *ork_mm_run_chain_i4_async (ork_npu *c, int S, const ork_mm_task_i4 *tasks){
    if (!c || S < 1 || !tasks) return NULL;
    return ork_async_launch((struct ork_async){ .kind=OAK_CHAIN_I4, .c=c, .S=S, .tasks=tasks }); }
ork_async *ork_mm_run_stream_i4_async(ork_npu *c, int S, const ork_mm_task_i4 *tasks){
    if (!c || S < 1 || !tasks) return NULL;
    return ork_async_launch((struct ork_async){ .kind=OAK_STREAM_I4, .c=c, .S=S, .tasks=tasks }); }

/* Join the async submit's worker thread, return its result (0/ok, <0 err), and free the handle.
 * NULL handle returns -1 (lets the caller treat "couldn't launch async" as a hard error and fall
 * back). After this returns, C holds the matmul output exactly as the synchronous path would. */
int ork_async_wait(ork_async *h){
    if (!h) return -1;
    int rc = -1;
    if (h->started) { pthread_join(h->th, NULL); rc = h->rc; }
    free(h);
    return rc;
}

/* CPU the most recent async worker was placed on at entry (sched_getcpu), or -1 if none has run /
 * not Linux. Lets a test assert the worker landed on a big core (not the caller's core, not an A55). */
int ork_npu_last_async_cpu(ork_npu *c){ return c ? c->last_async_cpu : -1; }

/* ================= HETEROGENEOUS OP-SEQUENCE SCHEDULER (ork_submit_seq) ============================
 * See the header for the design. The classification is a TABLE (SEQ_CLASS[], one row per ork_seq_kind)
 * in the same "data, not branches" spirit as XSPEC — the scheduler LOOP never switches on op-kind, it
 * only reads the row. A row is { hw, marker, profile, chain, fn }:
 *   hw     : 1 => the op-KIND can ride the HW-chain doorbell (int8 or fp16 mm) IF the runtime predicate
 *            seq_hw_ok() also holds (conforming K, M<=64, single-slice). Otherwise it takes the SW break below.
 *   marker : the ork_npu_enter target mode for the SW break path (SEQ_KEEPDT => keep c->last_dt, for SDP).
 *   profile: the XSPEC profile (XP_*) driving that transition's reset/rewarm policy.
 *   chain  : ork_chain_kind recorded as transition state (OCK_SW for every break here).
 *   fn     : the SW dispatch — the existing reliable per-op function. For an hw=1 row, fn is ALSO the
 *            fallback used when the op is that kind but fails seq_hw_ok() (non-conforming int8).
 * HW segments do NOT need a scheduler enter() — ork_dyn_begin_mc issues its own enter(DT_I8_CHAIN,
 * XP_CHAIN_NT, OCK_HW) internally, which handles the (fp16/SW)->i8-chain transition at the segment start.
 * SW breaks call enter() here (the dispatch fn re-enters idempotently; enter is a no-op on from==to and a
 * clear/reset is always conservative, so the redundancy can only add safety, never miscompute). */
enum { SEQ_KEEPDT = -1000 };   /* marker sentinel: pass c->last_dt (transient SDP: no mode marker change) */
typedef int (*ork_seq_disp)(ork_npu*, const ork_seq_op*);
struct ork_seq_class { uint8_t hw; int marker, profile, chain; ork_seq_disp fn; };
/* ===================== ASYNC SUBMIT (CPU‖NPU overlap foundation) =====================
 * The RKNPU SUBMIT ioctl is synchronous: the kernel blocks the calling thread until the NPU job
 * completes (flags=0x5 PC|PINGPONG, fence_fd=-1). The NPU is single-stream (one submit queue), so
 * "async" here does NOT mean two concurrent NPU jobs — it means the BLOCKING submit is issued on a
 * worker thread so the CALLING thread can do independent CPU work while the one NPU job runs, then
 * join at the dependency. This is DISPATCH-level and PATH-AGNOSTIC: it wraps the proven blocking run
 * functions verbatim, so it works identically for fp16 (ork_mm_run), int8 (ork_mm_run_i8), int4
 * (ork_mm_run_i4), and the chain/stream variants — nothing here is precision-specific (the int4
 * i4a8-inflated-to-int8 path is just a DT_I8 weight, so it rides the i8 entry). No kernel-fence
 * dependency that could wedge.
 *
 * Thread-safety: the caller MUST keep at most ONE async job in flight and MUST NOT touch A/C or
 * issue any other ork_mm_* on the same ctx between launch and wait (only independent CPU work) — the
 * NPU is single-stream, so there is never a concurrent submit racing the submit domain / ctx scratch. */
/* --- SW dispatch shims: adapt the generic ork_seq_op to each reliable driver function's signature --- */
static int seq_disp_i8_mm  (ork_npu *c,const ork_seq_op *o){ return ork_mm_run_i8(c,o->w,o->M,(const int8_t*)o->A,(int32_t*)o->C); }
static int seq_disp_f16_mm (ork_npu *c,const ork_seq_op *o){ ork_mm_task_f16 t={o->w,o->M,(const f16*)o->A,(float*)o->C}; return ork_mm_run_stream_f16(c,1,&t); }
static int seq_disp_i4_mm  (ork_npu *c,const ork_seq_op *o){ return ork_mm_run_i4(c,o->w,o->M,(const int8_t*)o->A,(int32_t*)o->C); }   /* multi-M rides the #4 doorbell route inside run_i4 */
static int seq_disp_ewmul_f16(ork_npu *c,const ork_seq_op *o){ double us; return ork_npu_ewmul_f16(c,(const f16*)o->A,(const f16*)o->B,o->M,o->N,(f16*)o->C,&us); }
static int seq_disp_silu_i8 (ork_npu *c,const ork_seq_op *o){ double us; return ork_npu_silu_i8(c,(const int8_t*)o->A,o->M,o->N,o->in_scale,o->out_scale,(int8_t*)o->C,&us); }
static int seq_disp_silu_i16(ork_npu *c,const ork_seq_op *o){ double us; return ork_npu_silu_i16(c,(const int16_t*)o->A,o->M,o->N,o->in_scale,o->out_scale,(int16_t*)o->C,&us); }
static int seq_disp_gelu_i8 (ork_npu *c,const ork_seq_op *o){ double us; return ork_npu_gelu_i8(c,(const int8_t*)o->A,o->M,o->N,o->in_scale,o->out_scale,(int8_t*)o->C,&us); }
static int seq_disp_ewmul_i8(ork_npu *c,const ork_seq_op *o){ double us; return ork_npu_ewmul_i8(c,(const int8_t*)o->A,(const int8_t*)o->B,o->M,o->N,o->mult,o->shift,(int8_t*)o->C,&us); }
static int seq_disp_add_i8  (ork_npu *c,const ork_seq_op *o){ double us; return ork_npu_add_i8(c,(const int8_t*)o->A,(const int8_t*)o->B,o->M,o->N,o->in_scale,o->b_scale,o->out_scale,(int8_t*)o->C,&us); }
static int seq_disp_add_f16 (ork_npu *c,const ork_seq_op *o){ double us; return ork_npu_add_f16(c,(const f16*)o->A,(const f16*)o->B,o->M,o->N,(f16*)o->C,&us); }
/* Uniform single-input SDP activations beyond the original seq subset (same (in,M,N,in_scale,out_scale,out) shape). */
static int seq_disp_gelu_i16(ork_npu *c,const ork_seq_op *o){ double us; return ork_npu_gelu_i16(c,(const int16_t*)o->A,o->M,o->N,o->in_scale,o->out_scale,(int16_t*)o->C,&us); }
static int seq_disp_rsqrt_i8(ork_npu *c,const ork_seq_op *o){ double us; return ork_npu_rsqrt_i8 (c,(const int8_t*) o->A,o->M,o->N,o->in_scale,o->out_scale,(int8_t*) o->C,&us); }
static int seq_disp_exp_i8  (ork_npu *c,const ork_seq_op *o){ double us; return ork_npu_exp_i8_biased(c,(const int8_t*) o->A,o->M,o->N,o->in_scale,o->out_scale,o->b_scale,(int8_t*) o->C,&us); }  /* b_scale = scalar max-bias (0 = plain exp) */
static int seq_disp_exp_i16 (ork_npu *c,const ork_seq_op *o){ double us; return ork_npu_exp_i16  (c,(const int16_t*)o->A,o->M,o->N,o->in_scale,o->out_scale,(int16_t*)o->C,&us); }
/* Softmax / RMSNorm normalize primitives (task #20 attention & full-layer chain). row-max: softmax max-shift
 * (reduce N->1). mul_perchan: the normalize A*b with b=1/Σ per query (also A.V per-channel scale, RMSNorm affine)
 * — 2-input, takes the per-channel vector via o->B. rsqrt_i16: 1/Σ (softmax, via rsqrt(Σ²)) and 1/√mean² (RMSNorm),
 * the accurate int16 LUT variant mirroring seq_disp_rsqrt_i8. */
static int seq_disp_reducemax_i8   (ork_npu *c,const ork_seq_op *o){ double us; return ork_npu_row_max_i8   (c,(const int8_t*) o->A,o->M,o->N,(int8_t*) o->C,&us); }
static int seq_disp_mul_perchan_f16(ork_npu *c,const ork_seq_op *o){ double us; return ork_npu_mul_perchan_f16(c,(const f16*)    o->A,(const f16*)    o->B,o->M,o->N,(f16*)    o->C,&us); }
static int seq_disp_mul_perchan_i8 (ork_npu *c,const ork_seq_op *o){ double us; return ork_npu_mul_perchan_i8 (c,(const int8_t*) o->A,(const int8_t*) o->B,o->M,o->N,o->mult,o->shift,(int8_t*) o->C,&us); }
static int seq_disp_rsqrt_i16      (ork_npu *c,const ork_seq_op *o){ double us; return ork_npu_rsqrt_i16     (c,(const int16_t*)o->A,o->M,o->N,o->in_scale,o->out_scale,(int16_t*)o->C,&us); }
/* NEOX RoPE (task #20 attention chain): x[nrow,hd] -> out. Weightless; internally ewmul_f16+ewmul_f16+add_f16
 * (self-contained SW-break). Field overload: pos[] (per-row positions) via o->B, freq_base via o->in_scale. */
static int seq_disp_rope_neox_f16  (ork_npu *c,const ork_seq_op *o){ return ork_npu_rope_neox_f16 (c,(const f16*)o->A,o->N,o->M,(const int*)o->B,o->in_scale,(f16*)o->C); }
/* RMSNorm (task #20 layer chain): x[M,n] -> out, per-channel gain via o->B, eps via o->in_scale. Weightless
 * matmul-wise (the gain is an SDP operand, not a packed ork_w); self-contained SW-break. */
static int seq_disp_rmsnorm_f16    (ork_npu *c,const ork_seq_op *o){ return ork_npu_rmsnorm_f16   (c,o->M,o->N,(const f16*)o->A,(const f16*)o->B,(float)o->in_scale,(f16*)o->C); }
/* A1: fp16 matmul with contiguous fp16 output (packed w) — SW-dispatch (own submit, not the f32-out doorbell). */
static int seq_disp_f16_mm_f16out  (ork_npu *c,const ork_seq_op *o){ return ork_mm_run_f16_f16out (c,o->w,o->M,(const f16*)o->A,(f16*)o->C); }
/* A1 int8: matmul with int8 requant output (int8 in, int8 out) — the all-int8 softmax island's int8 x-max
 * broadcast (max_i8 . -ones -> -max_bc int8) + any int8 matmul->SDP feed. mult/shift via o->mult/o->shift. */
static int seq_disp_matmul_requant_i8(ork_npu *c,const ork_seq_op *o){ return ork_mm_run_i8_out8(c,o->w,o->M,(const int8_t*)o->A,(int8_t*)o->C,o->mult,o->shift); }
/* int8 matmul -> int16 compact-linear out (set_i16_out); feeds an int16 SDP op resident, no PC-chain bridge. */
static int seq_disp_matmul_i16out_i8(ork_npu *c,const ork_seq_op *o){ return ork_mm_run_i8_out16(c,o->w,o->M,(const int8_t*)o->A,(short*)o->C,o->mult,o->shift); }
static const struct ork_seq_class SEQ_CLASS[ORK_OP_NKIND] = {
  /* ORK_OP_MM_I8   */ { 1, DT_I8,      XP_MC_MM,      OCK_SW, seq_disp_i8_mm    },
  /* ORK_OP_MM_F16  */ { 1, DT_F16,     XP_STREAM_F16, OCK_HW, seq_disp_f16_mm   },
  /* ORK_OP_MM_I4   */ { 1, 5/*I4_STRM*/,XP_I4_STREAM, OCK_SW, seq_disp_i4_mm    },
  /* ORK_OP_SILU_F16*/ { 0, SEQ_KEEPDT, XP_SDP,        OCK_SW, NULL /*TODO: fp16 SiLU needs a per-(in,out)-scale LUT plumbed through ork_seq_op*/ },
  /* ORK_OP_EWMUL_F16*/{ 0, SEQ_KEEPDT, XP_SDP,        OCK_SW, seq_disp_ewmul_f16 },
  /* ORK_OP_SILU_I8 */ { 0, SEQ_KEEPDT, XP_SDP,        OCK_SW, seq_disp_silu_i8  },
  /* ORK_OP_GELU_I8 */ { 0, SEQ_KEEPDT, XP_SDP,        OCK_SW, seq_disp_gelu_i8  },
  /* ORK_OP_EWMUL_I8*/ { 0, SEQ_KEEPDT, XP_SDP,        OCK_SW, seq_disp_ewmul_i8 },
  /* ORK_OP_ADD_I8  */ { 0, SEQ_KEEPDT, XP_SDP,        OCK_SW, seq_disp_add_i8   },
  /* ORK_OP_ADD_F16 */ { 0, SEQ_KEEPDT, XP_SDP,        OCK_SW, seq_disp_add_f16  },
  /* ORK_OP_SILU_I16*/ { 0, SEQ_KEEPDT, XP_SDP,        OCK_SW, seq_disp_silu_i16 },
  /* --- ops beyond the seq subset with a uniform SDP dispatch (designated; unlisted indices keep fn=NULL,
   *     i.e. reachable only via their typed ork_npu_* entry until the full impl registry lands) --- */
  [ORK_OP_GELU_I16] = { 0, SEQ_KEEPDT, XP_SDP, OCK_SW, seq_disp_gelu_i16 },
  [ORK_OP_RSQRT_I8] = { 0, SEQ_KEEPDT, XP_SDP, OCK_SW, seq_disp_rsqrt_i8 },
  [ORK_OP_EXP_I8]   = { 0, SEQ_KEEPDT, XP_SDP, OCK_SW, seq_disp_exp_i8   },
  [ORK_OP_EXP_I16]  = { 0, SEQ_KEEPDT, XP_SDP, OCK_SW, seq_disp_exp_i16  },
  /* softmax / RMSNorm normalize seq subset (task #20) — each self-contained SDP op; SW-break dispatch. */
  [ORK_OP_REDUCEMAX_I8]       = { 0, SEQ_KEEPDT, XP_SDP, OCK_SW, seq_disp_reducemax_i8    },
  [ORK_OP_MUL_PERCHANNEL_F16] = { 0, SEQ_KEEPDT, XP_SDP, OCK_SW, seq_disp_mul_perchan_f16 },
  [ORK_OP_MUL_PERCHANNEL_I8]  = { 0, SEQ_KEEPDT, XP_SDP, OCK_SW, seq_disp_mul_perchan_i8  },
  [ORK_OP_RSQRT_I16]          = { 0, SEQ_KEEPDT, XP_SDP, OCK_SW, seq_disp_rsqrt_i16       },
  [ORK_OP_ROPE_NEOX_F16]      = { 0, SEQ_KEEPDT, XP_SDP, OCK_SW, seq_disp_rope_neox_f16   },
  [ORK_OP_RMSNORM_F16]        = { 0, SEQ_KEEPDT, XP_SDP, OCK_SW, seq_disp_rmsnorm_f16     },
  [ORK_OP_MM_F16_F16OUT]      = { 0, SEQ_KEEPDT, XP_STREAM_F16, OCK_SW, seq_disp_f16_mm_f16out }, /* matmul w/ fp16 out; hw=0 (own submit, not the f32 doorbell) */
  [ORK_OP_MATMUL_REQUANT_I8]  = { 0, SEQ_KEEPDT, XP_MC_MM,      OCK_SW, seq_disp_matmul_requant_i8 }, /* int8 matmul -> int8 requant out; hw=0 (own submit) */
  [ORK_OP_MATMUL_I16OUT_I8]   = { 0, SEQ_KEEPDT, XP_MC_MM,      OCK_SW, seq_disp_matmul_i16out_i8 }, /* int8 matmul -> int16 compact-linear out; hw=0 */
};
/* HW-doorbell eligibility: the exact acceptance ork_dyn_begin_mc enforces per task. int8/fp16 = single-slice,
 * conforming K%512 && K<=4096, M<=64, Sn==1 (fp16 adds M*K<=32768). int4 = M==1, single K/N-slice (its HW
 * chain is M=1-only and writes int16). An op of an hw=1 KIND that fails this downgrades to the SW break path
 * (its SEQ_CLASS fn). Kept in lockstep with begin_mc / begin_mc_i4's per-task guards — change both together. */
static int seq_hw_ok(const ork_seq_op *o){
    if(o->kind!=ORK_OP_MM_I8 && o->kind!=ORK_OP_MM_F16 && o->kind!=ORK_OP_MM_I4) return 0;
    ork_w *w=o->w; if(!w||w->Sn!=1) return 0;
    if(o->kind==ORK_OP_MM_I4){                 /* int4: M=1 (begin_mc_i4; A1 Sn>1 + A2 Sk>1 supported, Sk<=16) */
        if(w->dtype!=DT_I4||o->M!=1||w->Sk>16) return 0;
        return 1;
    }
    if(o->M<1||o->M>64) return 0;
    if(w->K%512||w->K>4096) return 0;
    if(w->Sk!=1 && !w->Bf) return 0;
    if(o->kind==ORK_OP_MM_I8){ if(w->dtype!=DT_I8) return 0; }
    else { if(w->dtype!=DT_F16) return 0; if((size_t)o->M*w->K>32768) return 0; }  /* fp16 tile cap */
    return 1;
}
#define ORK_SEQ_HWBATCH 256   /* max ops per doorbell submit (well under ork_dyn_begin_mc's 1024 cap) */
/* B2 terminal-SDP: lazily pack a tiny int8 witness weight (K=512,N=16, zeros). Appended as a group's terminal
 * matmul so an SDP-ending group HW-chains (its int32 sentinel gates completion). NULL if pack fails (-> caller
 * SW-breaks, as before). Resident on the ctx; freed at teardown. */
static ork_w *seq_ensure_witness(ork_npu *c){
    if(c->seq_witness) return c->seq_witness;
    static const int8_t wz[512*16] = {0};   /* all-zero weight (read-only, shared) */
    c->seq_witness = ork_mm_pack_i8(c, 512, 16, wz);
    return c->seq_witness;
}
int ork_submit_seq(ork_npu *c, const ork_seq_op *ops, int n){
    if(!c||n<0||(n>0&&!ops)) return -2;
    if(c->daemon){   /* Path B: run the whole sequence on the daemon — it does the batch/break/resume on its NPU */
        if(n==0) return 0;
        orkd_seq_op_c *so=calloc((size_t)n,sizeof *so); if(!so) return -2;
        int ok=1;
        for(int i=0;i<n && ok;i++){ const ork_seq_op *o=&ops[i]; ork_w *w=o->w;
            so[i].kind=(uint32_t)o->kind; so[i].M=o->M; so[i].in_scale=o->in_scale; so[i].out_scale=o->out_scale;
            so[i].b_scale=o->b_scale; so[i].mult=o->mult; so[i].shift=o->shift; so[i].group=o->group;
            so[i].A=o->A; so[i].B=o->B; so[i].C=o->C;
            switch(o->kind){
              case ORK_OP_MM_I8:  if(!w||!w->is_orkd){ok=0;break;} so[i].weight_id=w->orkd_id; so[i].N=w->N; so[i].abytes=(uint32_t)((size_t)o->M*w->K);   so[i].cbytes=(uint32_t)((size_t)o->M*w->N*4); break;
              case ORK_OP_MM_F16: if(!w||!w->is_orkd){ok=0;break;} so[i].weight_id=w->orkd_id; so[i].N=w->N; so[i].abytes=(uint32_t)((size_t)o->M*w->K*2); so[i].cbytes=(uint32_t)((size_t)o->M*w->N*4); break;
              case ORK_OP_MM_F16_F16OUT: if(!w||!w->is_orkd){ok=0;break;} so[i].weight_id=w->orkd_id; so[i].N=w->N; so[i].abytes=(uint32_t)((size_t)o->M*w->K*2); so[i].cbytes=(uint32_t)((size_t)o->M*w->N*2); break;   /* fp16 output */
              case ORK_OP_MATMUL_REQUANT_I8: if(!w||!w->is_orkd){ok=0;break;} so[i].weight_id=w->orkd_id; so[i].N=w->N; so[i].abytes=(uint32_t)((size_t)o->M*w->K); so[i].cbytes=(uint32_t)((size_t)o->M*w->N); break;   /* int8 in, int8 out */
              case ORK_OP_MATMUL_I16OUT_I8: if(!w||!w->is_orkd){ok=0;break;} so[i].weight_id=w->orkd_id; so[i].N=w->N; so[i].abytes=(uint32_t)((size_t)o->M*w->K); so[i].cbytes=(uint32_t)((size_t)o->M*w->N*2); break;   /* int8 in, int16 out */
              case ORK_OP_MM_I4:  if(!w||!w->is_orkd){ok=0;break;} so[i].weight_id=w->orkd_id; so[i].N=w->N; so[i].abytes=(uint32_t)((size_t)o->M*w->K);   so[i].cbytes=(uint32_t)((size_t)o->M*w->N*4); break;
              case ORK_OP_EWMUL_F16: case ORK_OP_ADD_F16: so[i].N=o->N; so[i].abytes=(uint32_t)((size_t)o->M*o->N*2); so[i].bbytes=(uint32_t)((size_t)o->M*o->N*2); so[i].cbytes=(uint32_t)((size_t)o->M*o->N*2); break;   /* fp16 binary SDP */
              case ORK_OP_SILU_I8: case ORK_OP_GELU_I8: case ORK_OP_EXP_I8: so[i].N=o->N; so[i].abytes=(uint32_t)((size_t)o->M*o->N); so[i].cbytes=(uint32_t)((size_t)o->M*o->N); break;   /* int8 unary SDP */
              case ORK_OP_SILU_I16: so[i].N=o->N; so[i].abytes=(uint32_t)((size_t)o->M*o->N*2); so[i].cbytes=(uint32_t)((size_t)o->M*o->N*2); break;   /* int16 unary SDP */
              case ORK_OP_EWMUL_I8: case ORK_OP_ADD_I8: so[i].N=o->N; so[i].abytes=(uint32_t)((size_t)o->M*o->N); so[i].bbytes=(uint32_t)((size_t)o->M*o->N); so[i].cbytes=(uint32_t)((size_t)o->M*o->N); break;   /* int8 binary SDP */
              /* --- attention seq ops (task #20): weightless SDP; daemon reconstructs generically -> Path A adapters.
               * B carries: mul_perchan/rmsnorm per-channel vector[N]; rope pos[M] (int). --- */
              case ORK_OP_REDUCEMAX_I8:       so[i].N=o->N; so[i].abytes=(uint32_t)((size_t)o->M*o->N);   so[i].cbytes=(uint32_t)o->M; break;                                                          /* int8 [M,N] -> row-max [M] */
              case ORK_OP_EXP_I16: case ORK_OP_RSQRT_I16: so[i].N=o->N; so[i].abytes=(uint32_t)((size_t)o->M*o->N*2); so[i].cbytes=(uint32_t)((size_t)o->M*o->N*2); break;                             /* int16 unary SDP */
              case ORK_OP_MUL_PERCHANNEL_F16: so[i].N=o->N; so[i].abytes=(uint32_t)((size_t)o->M*o->N*2); so[i].bbytes=(uint32_t)((size_t)o->N*2); so[i].cbytes=(uint32_t)((size_t)o->M*o->N*2); break; /* f16 A[M,N] * b[N] */
              case ORK_OP_MUL_PERCHANNEL_I8:  so[i].N=o->N; so[i].abytes=(uint32_t)((size_t)o->M*o->N);   so[i].bbytes=(uint32_t)((size_t)o->N);   so[i].cbytes=(uint32_t)((size_t)o->M*o->N); break;   /* i8  A[M,N] * b[N] */
              case ORK_OP_ROPE_NEOX_F16:      so[i].N=o->N; so[i].abytes=(uint32_t)((size_t)o->M*o->N*2); so[i].bbytes=(uint32_t)((size_t)o->M*sizeof(int)); so[i].cbytes=(uint32_t)((size_t)o->M*o->N*2); break; /* f16 x[M,hd], pos[M] via B */
              case ORK_OP_RMSNORM_F16:        so[i].N=o->N; so[i].abytes=(uint32_t)((size_t)o->M*o->N*2); so[i].bbytes=(uint32_t)((size_t)o->N*2); so[i].cbytes=(uint32_t)((size_t)o->M*o->N*2); break;   /* f16 x[M,n], gain[n] via B */
              default: ok=0; break;   /* SILU_F16: not yet routable (mirrors the direct -3 TODO row) */
            }
        }
        /* A2 intermediate residency: an op's A/B pointing at a PRIOR op's C (same buffer) means "consume that
         * op's output on-device" — skip the re-upload (a_src/b_src=j+1) and keep the referenced output resident
         * (c_keep, not shipped back). Callers express a resident dependent chain by aliasing buffers (which is
         * already required for the direct path); un-aliased (final) outputs are still returned. n==1 seqs and
         * non-aliasing callers get a_src/b_src/c_keep=0 => byte-identical to pre-v3. */
        if(ok) for(int i=0;i<n;i++) for(int j=0;j<i;j++){
            if(ops[i].A && ops[i].A==ops[j].C){ so[i].a_src=j+1; so[j].c_keep=1; }
            if(ops[i].B && ops[i].B==ops[j].C){ so[i].b_src=j+1; so[j].c_keep=1; } }
        if(ok){ int sdom=0; for(int i=0;i<n;i++) if(ops[i].w && ops[i].w->is_orkd){ sdom=ops[i].w->domain; break; }   /* v2: carry the sequence's domain (its matmul weights share one) */
                orkd_set_op_domain(c->daemon, (uint32_t)sdom); }
        int rc = ok ? orkd_submit_seq(c->daemon, n, so) : -3;
        free(so);
        return rc;
    }
    ork_mm_task_i8 batch[ORK_SEQ_HWBATCH]; int nb=0, bdom=0, bdt=0;
    int ret=0;
    /* Flush the accumulated HW run as ONE doorbell submit (begin_mc owns its own mode enter). A run is one
     * dtype AND one domain (begin_mc requires both), so bdt/bdom key the batch; a dtype/domain change breaks
     * it. The SW fallback (begin_mc returned NULL — shouldn't happen since seq_hw_ok mirrors its guard, but
     * defensive) is dtype-aware: fp16 -> run_stream_f16, int4 -> run_stream_i4, int8 -> run_i8. */
    #define SEQ_FLUSH_HW() do{ if(nb){ ork_dyn_chain *h=ork_dyn_begin_mc(c,nb,batch,0); \
        if(getenv("ORK_SEQ_DEBUG")) fprintf(stderr,"[seq] HW flush dt=%d n=%d -> %s\n", bdt, nb, h?"doorbell":"SW-fallback"); \
        if(h){ ork_dyn_end(h); } \
        else { /* ineligible/rejected: fall back to SW per-op (still correct, just no doorbell) */ \
            for(int _q=0;_q<nb && !ret;_q++){ \
                if(bdt==DT_F16){ ork_npu_enter(c,DT_F16,XP_STREAM_F16,OCK_SW); \
                    ork_mm_task_f16 _t={batch[_q].w,batch[_q].M,(const f16*)batch[_q].A,(float*)batch[_q].C}; \
                    if(ork_mm_run_stream_f16(c,1,&_t)) ret=-1; } \
                else if(bdt==DT_I4){ ork_npu_enter(c,5/*I4_STRM*/,XP_I4_STREAM,OCK_SW); \
                    ork_mm_task_i4 _t={batch[_q].w,batch[_q].M,batch[_q].A,batch[_q].C}; \
                    if(ork_mm_run_stream_i4(c,1,&_t)) ret=-1; } \
                else { ork_npu_enter(c,DT_I8,XP_MC_MM,OCK_SW); \
                    if(ork_mm_run_i8(c,batch[_q].w,batch[_q].M,batch[_q].A,batch[_q].C)) ret=-1; } } } \
        nb=0; } }while(0)
    for(int i=0;i<n && !ret;i++){
        const ork_seq_op *o=&ops[i];
        if((int)o->kind<0||(int)o->kind>=ORK_OP_NKIND){ ret=-2; break; }
        /* GROUPED RUN (Stage 4): a maximal contiguous run of group>0 ops rides ork_dyn_begin_seq_i8_mc — a
         * group-id change delimits INDEPENDENT chains (spread across cores). group==0 (default) never enters
         * here, so the legacy per-op path below is byte-identical for existing callers. Ineligible orki_run (engine
         * returns NULL: non-int8 / M>64 / kind not yet supported / terminal-not-matmul) => SW-run each op. */
        if(o->group>0){
            SEQ_FLUSH_HW(); if(ret) break;
            int j=i; while(j<n && ops[j].group>0) j++;              /* run [i,j) */
            int gs[ORK_SEQ_HWBATCH+1]; int ng=0; gs[0]=0;
            for(int p=i+1;p<j;p++) if(ops[p].group!=ops[p-1].group) gs[++ng]=p-i;
            gs[++ng]=j-i;
            /* B2 terminal-SDP: a group ending in a non-matmul (SDP) op can't provide the terminal int32 sentinel.
             * If any group in this run ends in SDP, splice a tiny witness matmul as that group's terminal so the
             * SDP rides the HW chain (the witness's sentinel gates completion; its output is discarded). Yields the
             * proven mm->...->SDP->mm chain shape (no new hardware surface). Falls back to SW if the witness can't
             * pack or the augmented run exceeds the batch cap. */
            int need_w=0; for(int g=0;g<ng;g++) if(ops[i+gs[g+1]-1].kind!=ORK_OP_MM_I8) need_w=1;
            ork_dyn_chain *h=NULL;
            if(!need_w){
                h = (j-i<=ORK_SEQ_HWBATCH) ? ork_dyn_begin_seq_i8_mc(c, j-i, &ops[i], ng, gs, 0) : NULL;
            } else {
                ork_w *ww = seq_ensure_witness(c);
                if(ww){
                    static ork_seq_op ao[ORK_SEQ_HWBATCH]; int ags[ORK_SEQ_HWBATCH+1]; int an=0, ang=0; ags[0]=0;
                    static const int8_t wa[512]={0}; static int32_t wc[16];   /* witness A (zeros) / C (scratch, discarded) */
                    int of=0;
                    for(int g=0;g<ng && !of;g++){
                        for(int p=gs[g];p<gs[g+1];p++){ if(an>=ORK_SEQ_HWBATCH){of=1;break;} ao[an++]=ops[i+p]; }
                        if(!of && ops[i+gs[g+1]-1].kind!=ORK_OP_MM_I8){                       /* append the witness terminal */
                            if(an>=ORK_SEQ_HWBATCH){of=1;break;}
                            memset(&ao[an],0,sizeof ao[an]); ao[an].kind=ORK_OP_MM_I8; ao[an].w=ww; ao[an].M=1;
                            ao[an].A=wa; ao[an].C=wc; an++;
                        }
                        ags[++ang]=an;
                    }
                    if(!of) h = ork_dyn_begin_seq_i8_mc(c, an, ao, ang, ags, 0);
                }
            }
            if(getenv("ORK_SEQ_DEBUG")) fprintf(stderr,"[seq] grouped run [%d,%d) ng=%d -> %s\n", i,j,ng, h?"seq-chain":"SW-fallback");
            if(h){ if(ork_dyn_seq_end(h)) ret=-1; }
            else { for(int p=i;p<j && !ret;p++){ const ork_seq_op *op=&ops[p]; const struct ork_seq_class *pcl=&SEQ_CLASS[op->kind];
                    if(pcl->hw && seq_hw_ok(op)){ ork_mm_task_i8 t1={op->w,op->M,(const int8_t*)op->A,(int32_t*)op->C};
                        ork_dyn_chain *hh=ork_dyn_begin_mc(c,1,&t1,0); if(hh){ if(ork_dyn_end(hh)<0){} } else { ork_npu_enter(c,DT_I8,XP_MC_MM,OCK_SW); if(ork_mm_run_i8(c,op->w,op->M,op->A,op->C))ret=-1; } }
                    else { int mk=(pcl->marker==SEQ_KEEPDT)?c->last_dt:pcl->marker; ork_npu_enter(c,mk,pcl->profile,pcl->chain); if(!pcl->fn){ret=-3;break;} if(pcl->fn(c,op))ret=-1; } } }
            i=j-1; continue;                                        /* for-loop i++ lands at j */
        }
        const struct ork_seq_class *cl=&SEQ_CLASS[o->kind];
        int dom = o->w ? o->w->domain : 0;
        if(cl->hw && seq_hw_ok(o)){
            /* accumulate a maximal run of consecutive HW-chainable ops. begin_mc = ONE dtype + ONE domain,
             * so break the run at a dtype change (i8<->f16) too — each dtype gets its own doorbell; the
             * i8<->f16 mode transition fires inside begin_mc's ork_npu_enter at the boundary. */
            int dt = o->w->dtype;
            if(nb && (dom!=bdom || dt!=bdt || nb>=ORK_SEQ_HWBATCH)) SEQ_FLUSH_HW();
            if(!nb){ bdom=dom; bdt=dt; }
            batch[nb].w=o->w; batch[nb].M=o->M; batch[nb].A=(const int8_t*)o->A; batch[nb].C=(int32_t*)o->C; nb++;
            continue;
        }
        /* SW break: close any open HW run, transition via the layer, dispatch on the reliable fn */
        SEQ_FLUSH_HW();
        if(ret) break;
        if(getenv("ORK_SEQ_DEBUG")) fprintf(stderr,"[seq] SW break kind=%d\n", (int)o->kind);
        int mk = (cl->marker==SEQ_KEEPDT) ? c->last_dt : cl->marker;
        ork_npu_enter(c, mk, cl->profile, cl->chain);
        if(!cl->fn){ ret=-3; break; }         /* op-kind dispatch not yet wired (documented TODO row) */
        if(cl->fn(c,o)) ret=-1;
    }
    SEQ_FLUSH_HW();
    #undef SEQ_FLUSH_HW
    return ret;
}

/* Generic enum-driven submit (header: ork_submit). Single op -> its SEQ_CLASS standalone dispatch, mirroring
 * the per-op path in ork_submit_seq (enter() then the reliable SW dispatch fn). Ops without a seq dispatch fn
 * (the >=11 SDP/perchan/replay ops) return -3 here; they are reached via their typed ork_npu_* entry points
 * until the (op x mode) -> regcmd impl registry is wired. mode is advisory for a single op — HW/SW chaining is
 * decided per-transition by ork_submit_chain via the ork_chain_lookup table. */
int ork_submit(ork_npu *c, ork_op op, ork_impl_mode mode, const ork_seq_op *args){
    if((int)op<0 || (int)op>=ORK_OP_NKIND || !args) return -2;
    (void)mode;
    ork_seq_op o=*args; o.kind=op;
    const struct ork_seq_class *cl=&SEQ_CLASS[op];
    if(!cl->fn) return -3;                                  /* no generic dispatch — use the typed ork_npu_* entry */
    int mk=(cl->marker==SEQ_KEEPDT)?c->last_dt:cl->marker;
    ork_npu_enter(c,mk,cl->profile,cl->chain);
    return cl->fn(c,&o) ? -1 : 0;
}

/* Deterministic chain submit: the table decides, and there is NO failing check. A DISALLOW transition is a
 * hard PARTITION boundary — the two ops go into separate runs (independent submits), never chained. So a
 * wedge-prone pair is split, not rejected: the routing always produces correct output and cannot fail on a
 * transition. (orkd + SDK ship together and the op set is fixed, so a runtime "reject" would only re-check
 * what the table + the fixed composite asserts already guarantee.) Each maximal run goes to ork_submit_seq,
 * which HW-batches / SW-breaks within it. Returns the first real submit error, if any. */
int ork_submit_chain(ork_npu *c, const ork_seq_op *ops, int n){
    if(!ops || n<0) return -2;
    int start=0, ret=0;
    for(int i=0;i<n && !ret;i++){
        int boundary = (i+1==n) || (ork_chain_lookup(ops[i].kind, ops[i+1].kind)==ORK_CHAIN_DISALLOW);
        if(boundary){ ret = ork_submit_seq(c, ops+start, i-start+1); start = i+1; }   /* run [start,i] */
    }
    return ret;
}

/* ---- BATCHED DYNAMIC GEMM (attention / GDN-chunk primitive) --------------------------------------
 * C[b] = A[b][M,K] * B[b][K,N] for each of nbatch batches. Both operands are dynamic activations, so
 * B[b] is packed fresh each batch (unlike the resident-weight ork_mm_run* paths). Correctness-first:
 * one submit per batch. i8 reuses a single packed buffer via repack (no per-batch alloc); i4/fp16 have
 * no repack path yet, so they pack+free per batch. Chaining (fewer submits) is a follow-up. */
/* Gather a strided 2-D operand src[r,c] = src + r*sr + c*sc (element strides) into contiguous
 * row-major dst[r*cols+c]. The stride-1 inner dim (the common attention case: contraction dim
 * contiguous) hits a memcpy fast path; a strided inner dim falls back to an element loop. */
#define ORK_BMM_GATHER(NAME,T) \
void NAME(T *dst, const T *src, int rows, int cols, long sr, long sc){ \
    for(int r=0;r<rows;r++){ const T *p=src+(long)r*sr; T *d=dst+(size_t)r*cols; \
        if(sc==1) memcpy(d,p,(size_t)cols*sizeof(T)); \
        else for(int cc=0;cc<cols;cc++) d[cc]=p[(long)cc*sc]; } }
ORK_BMM_GATHER(orki_bmm_gather_i8, int8_t)
ORK_BMM_GATHER(orki_bmm_gather_f16, f16)
/* scatter contiguous src[r*cols+c] -> strided dst[r,c] = dst + r*sr + c*sc (C output; int32/float share size) */
void orki_bmm_scatter_i32(int32_t *dst, const int32_t *src, int rows, int cols, long sr, long sc){
    for(int r=0;r<rows;r++){ int32_t *d=dst+(long)r*sr; const int32_t *s=src+(size_t)r*cols;
        if(sc==1) memcpy(d,s,(size_t)cols*sizeof(int32_t)); else for(int cc=0;cc<cols;cc++) d[(long)cc*sc]=s[cc]; } }

/* natural (dense row-major) strides for the M×K/K×N/M×N batched operands */
ork_bmm_strides orki_bmm_natural(int M,int K,int N){
    return (ork_bmm_strides){ .as_m=K,.as_k=1,.bs_k=N,.bs_n=1,.cs_m=N,.cs_n=1,
                              .abs=(long)M*K,.bbs=(long)K*N,.cbs=(long)M*N }; }
/* C output is contiguous per batch iff the natural row-major layout — then run writes into C directly. */
int orki_bmm_c_dense(const ork_bmm_strides *s,int N){ return s->cs_n==1 && s->cs_m==N; }



/* ---- On-NPU normalization primitives (GATED behind ORK_NORM_NPU; default = CPU) ------------------
 * rmsnorm/l2norm per row of [M][n]. The CDP fixed-function LRN can't do a full-channel reduction (window
 * HW-capped at n<=9), BUT the norm decomposes into NPU-fittable pieces — the key one being the full
 * reduction sum(x^2), which is a MATMUL contraction (the NPU contracts K=n unboundedly via K-split):
 *     sq = x .* x                         (square, CPU — memory-bound elementwise)
 *     ss[M,16] = sq[M,n] . ones[n,16]      (NPU fp16 matmul; ss[m,0] = sum_j x[m,j]^2)   <-- on NPU
 *     scale = 1/sqrt(ss/n + eps)          (rsqrt, CPU; SDP-LUT rsqrt is a follow-up)
 *     out = x * scale * w                 (scale, CPU)
 * This is SLOWER than the fused CPU/NEON pass (an extra square pass + an N=16 reduce-matmul submit) and
 * is off by default — but it puts the reduction on the NPU as requested. The reduce-matmul + rsqrt-LUT +
 * scale can be CHAINED into ONE submit (run_chain) or fused into an adjacent matmul's output stage; that
 * amortization is the follow-up. On any NPU-path failure it falls back to the CPU result. */
/* cached ones[n,16] fp16 reduce-weight (single-slot; norm calls are single-threaded in the graph) */
static ork_w *g_ones_w=NULL; static int g_ones_n=0; static ork_npu *g_ones_ctx=NULL;
static ork_w *norm_reduce_w(ork_npu *c,int n){
    if(g_ones_w && g_ones_n==n && g_ones_ctx==c) return g_ones_w;
    if(g_ones_w){ ork_mm_free(g_ones_ctx,g_ones_w); g_ones_w=NULL; }
    f16 *ones=malloc((size_t)n*16*sizeof(f16)); if(!ones) return NULL;
    for(size_t i=0;i<(size_t)n*16;i++) ones[i]=(f16)1.0f;
    g_ones_w=ork_mm_pack(c,n,16,ones); free(ones);
    if(g_ones_w){ g_ones_n=n; g_ones_ctx=c; }
    return g_ones_w;
}
/* NPU reduction: ss[m] = sum_j x[m,j]^2 via the (x.*x)*ones matmul. Returns 0/ok, <0 to signal CPU fallback. */
int ork_norm_reduce_npu(ork_npu *c,int M,int n,const f16 *x,float *ss_out){
    if(!ork_norm_npu_enabled() || n%32) return -1;      /* K%32 for the matmul */
    ork_w *ow=norm_reduce_w(c,n); if(!ow) return -1;
    f16 *sq=malloc((size_t)M*n*sizeof(f16)); float *ss16=malloc((size_t)M*16*sizeof(float));
    int rc=-1;
    if(sq&&ss16){
        for(size_t i=0;i<(size_t)M*n;i++){ float v=(float)x[i]; sq[i]=(f16)(v*v); }
        if(ork_mm_run(c,ow,M,sq,ss16)==0){ for(int m=0;m<M;m++) ss_out[m]=ss16[(size_t)m*16]; rc=0; }
    }
    free(sq); free(ss16); return rc;
}
/* Cached fp16 rsqrt LUT (BUILD-ONCE) + a K=512 op mapping ss[M] -> scale[M]=1/sqrt(ss/nf+eps) ON the NPU,
 * DECOUPLED from the reduce so it works for ANY feature dim n (reduce K-splits; this op is K=512 = the LUT
 * builder's known-good geometry — K=32 is DEGENERATE for the fp16 fused-silu tiling, gives acc~0). The scalar
 * ss is fed DENSE + NORMALIZED: A[m,k]=ss[m]/G for all Kd cols, weight=-S*G/Kd -> acc = -S*ss (A stays ~O(1),
 * weight small — exactly the probe regime). G = the calibration upper bound g_rs.hi. nf: rmsnorm=n, l2norm=1.
 * Single-slot cache (rebuilds when ctx/nf/eps change or ss drifts outside [lo,hi]); norm calls single-threaded. */
#define ORK_RSQRT_KD 512
static struct { ork_npu *c; int nf; double eps, lo, hi, osc; ork_w *wS; int16_t lut[1030]; int valid; } g_rs = {0};
static int rsqrt_lut_ensure(ork_npu *c,int nf,double eps,double ss_lo,double ss_hi){
    if(g_rs.valid && g_rs.c==c && g_rs.nf==nf && g_rs.eps==eps && ss_lo>=g_rs.lo && ss_hi<=g_rs.hi) return 0;
    /* TIGHT padding: the 64-point PWL LUT must keep resolution IN the actual ss band (a wide range spreads the
     * probe points thin -> big rsqrt error). ~1.3x span keeps ~0.1% accuracy; drift outside triggers a rebuild. */
    double lo=ss_lo*0.9, hi=ss_hi*1.15; if(hi<=0) return -1;
    double fl=(double)nf*eps; if(lo<fl) lo=fl; if(lo>=hi) lo=hi*0.5;
    double S,R,osc; int16_t lut[1030];
    if(ork_mm_build_f16_rsqrt_lut(c,nf,eps,lo,hi,lut,&S,&R,&osc)) return -1;
    f16 *B=malloc((size_t)ORK_RSQRT_KD*16*sizeof(f16)); if(!B) return -1;
    for(int i=0;i<ORK_RSQRT_KD*16;i++) B[i]=(f16)(-S*hi/(double)ORK_RSQRT_KD);   /* acc = sum_k (ss/hi)*(-S*hi/Kd) = -S*ss */
    ork_w *wS=ork_mm_pack(c,ORK_RSQRT_KD,16,B); free(B); if(!wS) return -1;
    if(g_rs.wS) ork_mm_free(g_rs.c,g_rs.wS);
    memcpy(g_rs.lut,lut,sizeof lut); g_rs.wS=wS; g_rs.c=c; g_rs.nf=nf; g_rs.eps=eps; g_rs.lo=lo; g_rs.hi=hi; g_rs.osc=osc; g_rs.valid=1;
    return 0;
}
/* scale[m] = 1/sqrt(ss[m]/nf + eps) on the NPU (K=512 fused rsqrt). 0/ok, <0 -> caller uses CPU rsqrt. */
int ork_norm_rsqrt_npu(ork_npu *c,int M,int nf,double eps,const float *ss,float *scale){
    double lo=1e30,hi=0; for(int m=0;m<M;m++){ if(ss[m]<lo)lo=ss[m]; if(ss[m]>hi)hi=ss[m]; }
    if(hi<=0 || rsqrt_lut_ensure(c,nf,eps,lo,hi)) return -1;
    double G=g_rs.hi;
    f16 *A=malloc((size_t)M*ORK_RSQRT_KD*sizeof(f16)); float *C=malloc((size_t)M*16*sizeof(float));
    int rc=-1;
    if(A&&C){ for(int m=0;m<M;m++){ f16 v=(f16)(ss[m]/G); for(int k=0;k<ORK_RSQRT_KD;k++) A[(size_t)m*ORK_RSQRT_KD+k]=v; } /* dense ss/G */
        if(ork_mm_run_f16_silu(c,g_rs.wS,M,A,C,0,0xffffc000u,0x56391100u,g_rs.lut,1030)==0){
            for(int m=0;m<M;m++) scale[m]=(float)((double)C[(size_t)m*16]*g_rs.osc); rc=0; } }
    free(A); free(C); return rc;
}
/* rmsnorm/l2norm: reduction sum(x^2) on the NPU (ork_norm_reduce_npu, any n via K-split) when ORK_NORM_NPU
 * is set; rsqrt + scale on CPU. Validated 0.0005 vs the CPU ref. (The fully-fused single-submit reduce+rsqrt
 * — ork_mm_build_f16_rsqrt_lut, validated 0.0012 standalone for n<=2048 in test_bmm's rsqrt test — is the
 * building block for a one-submit on-NPU norm; wiring it into the general path needs LUT pre-calibration, so
 * the shipped norm keeps rsqrt on CPU for robustness across all n.) The ork_norm_rsqrt_npu helper above is
 * the decoupled K=32 rsqrt op, kept for that follow-up. */

/* On-NPU composed softmax over each row of [M][n]: y = exp(x-max)/Σexp(x-max). Gated ORK_SOFTMAX_NPU.
 * The per-row max and the final normalize (÷Σ) are CPU (cheap per-row scalars); the heavy parts run on the
 * NPU: exp via ork_npu_exp_i16 (int16 SDP LUT; x-max quantized to a shared in_scale so exp maps in*in_scale
 * -> exp(x-max)), Σ via the reduction-as-matmul (e·ones[n,16], reusing the norm reduce weight). Like the
 * norm this is submit-floor-bound standalone (gated off; the win is fusing into the attention chain). Any
 * NPU-path failure (PPU-fuse off, n%32!=0, exp/reduce error) falls back to the full CPU softmax. 0/ok,-2. */

/* Fast Walsh-Hadamard Transform (FWHT) - Exposed utility function for caller-driven quantization */
void ork_fwht_norm(float *v, int n){
    for(int len=1; len<n; len<<=1)
        for(int i=0;i<n;i+=len<<1)
            for(int j=i;j<i+len;j++){ float a=v[j], b=v[j+len]; v[j]=a+b; v[j+len]=a-b; }
    float s=1.0f/sqrtf((float)n);
    for(int i=0;i<n;i++) v[i]*=s;
}

/* ==== CPU-side pack/dump helpers linked by the ggml-ork backend (no internal callers) ==== */
/* SINGLE-THREADED int8 CPU dump — identical bytes to ork_w_dump_i8_cpu, but tiles inline on the calling
 * thread (NO internal pool). For callers that ALREADY parallelize at a coarser grain (the .orkpack expert
 * convert runs one whole expert per core); using the shared pthread pool there would nest/oversubscribe. */

/* Pack int8 B[K,N] (row-major [k*N+n]) DIRECTLY into IMPORTED dma-buf chunks, tiled into the NPU layout —
 * ork_mm_pack_i8's tiling into ork_mm_load_i8_import's chunked-import storage, with NO native bcreate, NO
 * blob round-trip, NO free/churn. Purpose: a fused per-tensor weight (fc.wg) allocates as uniform ~16MB
 * import chunks like every other weight, instead of a native-bcreate outlier that fragments the 32-bit
 * domain's IOVA (the PRIME-ENOMEM-at-low-fill that blocked per-domain fusion). Mirrors load_i8_import
 * (anchor + chunked own_bufs + full-K Bf rebuild); returns NULL if import is unavailable/fails. */

/* CPU-ONLY int4 pack straight to the compact .orkpack blob (header + bscale[N] + Bi4[K*N/2]) — byte-
 * identical to ork_mm_pack_i4a8_im() + ork_w_dump_i4a8(), but with NO bcreate/IOMMU/tiling. The per-tile
 * bcreate in the NPU int4 packer is the serial single-stream consumer that bottlenecks .orkpack conversion;
 * a WRITE only needs the compact nibbles + scales on disk (ork_mm_load_i4a8 re-tiles at load). This
 * replicates the EXACT per-channel quant of ork_mm_pack_i4a8_im (absmax/7 uniform or NF4 codebook, optional
 * imatrix clip-grid, SR with a per-call seed) so the bytes match. Single-threaded (caller parallelizes over
 * experts). out=NULL → required size. K%32,N%32. */
