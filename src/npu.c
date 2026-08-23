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
#include "spine_kernels.h" /* CPU glue (rmsnorm/rope/attn/silu/quant/civac) for the whole-layer core ork_i8_mm_layer */
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
/* #33 reentrancy guard: orki_i8_slice_pack packs each sub-tile via ork_i8_mm_pack -> orki_pack(), which would ITSELF
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
/* Last regcmd context (set by validate_regcmd) — dumped on a submit failure to PIN which weight/op/domain/
 * import-status faulted (e.g. the multi-domain import scale-fault: errno=22). */
const char *orki_last_op = "?"; int orki_last_K=0, orki_last_N=0, orki_last_wdom=-1, orki_last_import=0;
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
/* Bb[ns*Sk+ks] = K-split x N-split (always). Bf[ns] = optional full-K per N-slice (the ORK_FULLK_DEC gate is removed; now unconditional,
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
     * so there is no per-shape threshold and no env knob (the old ORK_DECODE_MC gate is removed). mc_prof's warm
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
/* ESTABLISH a non-0 IOMMU domain with a small NATIVE allocation before any dma-buf import is mapped into
 * it. The kernel rknpu driver lazily sets up a domain's IOVA allocator / page table on its FIRST buffer;
 * if that first buffer is an IMPORTED dma-buf, the import's SG-list pages get wrong/aliased IOVAs and the
 * NPU reads garbage for some weight tiles (non-deterministic dropped-K corruption — reproduced with
 * tools/mc_import_probe.c: import-first FAULTS, native-alloc-first is bit-exact; single- AND multi-core).
 * One tiny native anchor per domain, kept resident for the ctx lifetime (freed at teardown). No-op for
 * domain 0 (always established) and when already anchored. Call BEFORE the first bimport into `dom`. */
/* Public: pre-size the ctx for `n` IOMMU domains — the backend calls this once with the auto-sizer's
 * n_domains so no per-weight grow happens mid-load. Only meaningful for n>1 (n<=1 = single-domain, leaves
 * dom_save NULL). Safe to call repeatedly; only ever grows. */
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
/* #39 per-core fold submit: one core's task-group, core_mask=1u<<core, own task buffer (tasks from index 0),
 * subcore_task[*]={0,P} — EXACTLY the proven mcworker per-core pattern. Run one per thread => concurrent 3-core. */

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

/* Apply ork's synth_i8 matmul GEOMETRY (same formulas as synth_i8, sched=1) onto an arbitrary regcmd `rc`
 * of length `n`. Used to inject ork's geometry into RKNN's EW-mul TEMPLATE (REGCMD_EWMUL_LIN) — which keeps
 * RKNN's register ORDER + EW output-stage/lane (so it executes) while making the conv engine read ork's own
 * [Nt][Kt][32][32] A/B tile layout (so acc is correct). Addresses (0x1070/0x1110/0x4020) patched by caller. */


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
 * orki_i4_synth (win over the K/N/mc-derived regs). Inert by default (n_on=0) — production is unaffected. Only
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
/* Self-healing recovery: detect -> DUMP everything -> soft RESET -> DUMMY-op probe. Returns 1 if the dummy op
 * PASSES (NPU recovered — caller keeps going), 0 if it FAILS (NPU still broken — caller should throw a fault
 * and stop, rather than spiral into a hard wedge). This is the recovery contract the caller runs on an anomaly
 * (a round that fails to land / a submit error). */
/* Deliberately force a RELIABLE NPU fault (to exercise dump/recover): a tiny int8 matmul whose WEIGHT address
 * is BOGUS (0x1000 — an unmapped low page) so the NPU DMA-reads from unmapped and faults every time. NONBLOCK +
 * a 1s bounded host poll so the caller never blocks. Returns 1 if the output doorbell landed (no fault), 0 if it
 * did NOT land in the window (the expected fault). May soft-reset or IOMMU-fault the NPU — that's the point. */


/* ORK_LOAD_PROF: print (and reset) the per-phase import breakdown. Called at teardown. No-op unless set. */
void ork_ssm_prof_dump(void);            /* fwd: ORK_SSM_PROF per-section accounting */
void ork_ssm_helper_stop(ork_npu *c);    /* fwd: stop the little-core marshalling helper */
void ork_npu_xprof_dump(void);

/* Zero-copy IMPORT (no alloc, no copy) — see header. Registered in dma_tab like ork_dma_alloc so
 * ork_f16_mm_run zero-copy detection + dma_find work; freed by ork_dma_import_free (or ork_dma_free). */
/* Import an EXTERNAL dma-buf fd (e.g. received over SCM_RIGHTS from another process) into the NPU's IOMMU
 * domain and register it for zero-copy: the returned CPU pointer maps the shared buffer, and passing a ptr
 * into it as A/C to ork_f16_mm_run* makes the NPU read/write that buffer in place (dma_find resolves the IOVA).
 * Takes ownership of `dmabuf_fd` (closed by ork_dma_free/ork_dma_import_free). NULL on failure. This is the
 * orkd daemon's cross-process zero-copy hook (client shares a buffer; orkd runs against it, no copy). */
/* the registered DMA buffer containing host ptr p, or NULL if p isn't zero-copy-resident */
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
 * Set the domain BEFORE packing/loading a weight: every subsequent ork_i8_mm_pack / ork_i8_mm_load (and
 * the fp16/int4 variants) places its resident tiles in `domain` and stamps it on the returned ork_w; at
 * run time ork_f16_mm_run* submits that weight's matmuls against the same domain automatically. Activation/
 * output scratch follows the most-recently-set pack domain. domain<0 reverts to the process default
 * (env ORK_IOMMU_DOMAIN, else 0). Domains are created lazily by the kernel on first use. */
void ork_npu_set_pack_domain(ork_npu *c,int domain){ if(!c) return; c->pack_domain = domain<0 ? -1 : domain;
    if(c->daemon) orkd_set_pack_domain(c->daemon, domain<0 ? 0u : (uint32_t)domain); }   /* Path B: route the domain to the daemon so orkd_pack lands the weight there */

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


/* Tile a contiguous [nt] range of int8 weight columns into the NPU 32x32 block layout. Shared by the Bb
 * K-slice tiles and the full-K Bf rebuild (same structure; differ only in KT and the k0 offset). Each nt
 * range is disjoint in bb, so this is bit-identical to the serial loop. */

/* Inflate a contiguous [nt] range of int8 weight columns straight into the fp16 [Nt][Kt][16][32] tile
 * layout, scaled per-output-channel: wf16 = (f16)((float)i8 * bscale[n]). Same mapping orki_pack() uses for
 * DT_F16, but the source element is a dequantized int8 code instead of a stored fp16 — so the resulting
 * tile bytes are BIT-IDENTICAL to ork_f16_mm_pack of the row-major dequantized weight. Emulated W8A16. */
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
                 * non-consolidated layout and round-trips through ork_i8_mm_load. */
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
            struct tile_i8_arg ta={bb,Bi,KT,k0,n0,N}; ork_parallel_for(NN,orki_i8_tile_range,&ta);   // all-core tiling
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
            struct tile_i8_arg ta={bb,Bi,KTf,0,n0,N}; ork_parallel_for(NN,orki_i8_tile_range,&ta);   // all-core full-K rebuild
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
ork_w *ork_f16_mm_pack   (ork_npu *c,int K,int N,const f16    *B){
    if(c && c->daemon){ uint64_t id=orkd_pack_f16(c->daemon,K,N,B); if(!id) return NULL; ork_w *w=calloc(1,sizeof *w); if(!w) return NULL; w->is_orkd=1; w->orkd_id=id; w->K=K; w->N=N; w->dtype=DT_F16; w->domain=ork_dom(c->pack_domain); return w; }   /* Path B: fp16 pack in the daemon (remember the domain so runs carry it) */
    return orki_pack(c,K,N,B,DT_F16); }

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
 * (K,N), buffers init-synced but carrying no data. Fill per-forward with ork_i8_mm_inflate_to_f16 and run
 * via ork_f16_mm_run / ork_f16_mm_run_silu. Reclaim with ork_mm_free like any packed weight. K%32, N%16.
 * Returns NULL on bad dims / alloc failure. */
/* Fill an fp16 scratch (from ork_f16_mm_scratch, same K,N) with wf16[k,n]=(f16)((float)i8[k*N+n]*bscale[n]).
 * i8 is row-major [K,N]; bscale is per-output-channel [N] (NULL => scale 1). Re-tiles in place (no alloc);
 * single TO_DEVICE sync per tile (buffers already inited by ork_f16_mm_scratch, like ork_i8_mm_repack).
 * The tiled bytes are bit-identical to ork_f16_mm_pack of the row-major dequantized weight. 0/ok, <0 on bad args. */

/* PERSIST. Serialize a packed weight's resident tile bytes (Bb only; Bf is a regenerable decode-only
 * optimization) into `out` in tile order — the on-disk form for pre-packed (.orkpack) weights. Each
 * tile is its page-padded buffer size, so it round-trips through ork_i8_mm_load. Pass out=NULL to size. */
size_t ork_w_dump(const ork_w *w, void *out, size_t cap){
    if(!w) return 0;
    /* OFFLINE weight: no Bb was ever allocated. Tile the raw codes with the CPU twin, which is asserted
     * byte-identical to pack+dump (test_i4_dump_cpu) — so a caller persisting these bytes gets the same
     * .orkpack it would have got from the NPU. */
    if(w->cpu_codes)
        return (w->dtype==DT_I4) ? ork_i4_w_dump_cpu(w->off_ctx, w->K, w->N, w->cpu_codes, out, cap)
                                 : ork_i8_w_dump_cpu(w->off_ctx, w->K, w->N, w->cpu_codes, out, cap);
    if(!w->Bb) return 0;
    size_t off=0, nb=(size_t)w->Sk*w->Sn;
    for(size_t i=0;i<nb;i++){ const struct buf *b=&w->Bb[i]; if(!b->cpu) continue;
        if(out){ if(off+b->size>cap) return 0; memcpy((char*)out+off,b->cpu,b->size); }
        off+=b->size; }
    return off;
}
/* CPU-ONLY int8 dump: produce the SAME bytes as ork_i8_mm_pack() + ork_w_dump(), but tile straight
 * into a caller DRAM buffer — no NPU. There is NO reason to allocate an IOMMU/IOVA DMA buffer, tile
 * into it, cache-flush it TO the device, and read it back just to write a .orkpack file: that whole
 * bcreate+bsync round-trip is the serial single-stream consumer that bottlenecks conversion. Here the
 * tiling (same orki_i8_tile_range, page-padded per tile, same Sk×Sn order as ork_w_dump) runs pure-CPU and
 * parallel across all cores; the NPU is touched only at LOAD time (ork_i8_mm_load_import). Pass out=NULL
 * to size. K%32, N%32. Byte-identical to the pack+dump path (fresh DMA bufs are zeroed; we zero-pad). */
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


/* ---- ORKD_IMPORT: client-owned resident weight (client allocs the dma-buf, daemon only maps it) ----
 * The client (which has NO NPU fd under orkd) allocates a plain dma-heap buffer, fills it with the PRE-TILED
 * .orkpack bytes, and hands the fd to the daemon; the daemon PRIME-imports it into the client's domain and
 * lays the resident tiles as base+offset VIEWS. This keeps ALL weight residency client-side (client manages
 * its IOVA domains) — the daemon never tiles, never allocs a weight buffer, never owns the bytes. */

/* CLIENT: flush the CPU-written bytes so the NPU (via the daemon's IOMMU import) sees them. */

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
/* Re-tile fp16 B[K,N] (row-major) into an EXISTING fp16 ork_w (from ork_f16_mm_scratch/ork_f16_mm_pack, same
 * K,N) — no bcreate/bdestroy. The fp16 twin of ork_i8_mm_repack: lets a caller keep a persistent weight
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
 * the existing int8 path so the result runs via ork_i8_mm_run unchanged. bscale_out[n] carries the
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
 * The f32 path inflates nibble -> f32 code -> orki_i8_tile_f32, which re-quantizes via
 * lrintf(code*1.0) clamped to [-127,127]. But the codes are ALWAYS exact small ints
 * (UNIFORM in [-7,7]; NF4 LUT = round(level*127) in [-127,127]) so that quant is the
 * identity: the int8 byte placed in the tile equals the int4 code. So we can inflate
 * straight to int8 and rearrange bytes into the tile layout with NO float round-trip.
 * Output is bit-identical to the f32 path (proven by the matmul/memcmp gate). */
/* UNIFORM: expand one channel's nibble-packed int4 codes -> LINEAR int8 [-7,7] in i8[K]. */
/* NF4: inflate one channel's indices (stored in the nibble) -> LINEAR int8 codes via the LUT.
 * The nibble store keeps the 0..15 index (low/high nibble per k); LUT[idx] = round(level*127). */
/* Rearrange LINEAR int8 codes i8[N][K] -> the NPU tiled int8 layout, copying bytes (NO quant, NO float).
 * Byte-for-byte the same destination math as orki_i8_tile_f32 (per (ns,ks) buffer: element of channel
 * n=n0+nt*32+nl at k-pos k0+kt*32+ki lands at nt*KT*32*32 + kt*32*32 + nl*32 + ki) but feeding the int8
 * code directly, since orki_i8_tile_f32 with inv=1 maps code -> clamp(lrintf(code),-127,127) = code (identity).
 * Same per-buffer init bsync sequence as orki_i8_tile_f32 (fresh buffers need TO|FROM then TO). */
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
 * nibble store (~K*N/2) + per-channel scales — about half the size — and ork_i4a8_mm_load re-inflates the
 * nibbles -> int8 and re-tiles on load (the tail of the pack path, but from stored nibbles, not f32). The
 * blob is self-contained: the NF4 LUT is NOT stored (it's derived from quant_kind). */

/* Serialize the compact int4 form: header + bscale[N] (f32) + Bi4 (K*N/2 bytes). out=NULL -> required
 * size. Returns 0 if `w` is not an int4-packed weight (no Bi4/bscale) or on cap overflow. */
/* Rearrange linear int8 codes i8[N][K] into IMPORTED (dma-buf) tiles, using the dma-buf's OWN cache sync
 * (the rknpu MEM_SYNC does NOT cover foreign imports). Same byte math as orki_i8_tile_to_tiles. */
/* ---- DIAGNOSTIC ONLY (tools/prefetch_headroom.c): isolate the STEADY-STATE per-slice streaming prep.
 * These re-run the TAIL of the int4 pack path (inflate stored nibbles -> int8 codes; tile into the
 * ALREADY-ALLOCATED resident DMA buffers) on an int4-packed weight, with NO bcreate/alloc — exactly the
 * work a streaming double-buffer would do per cycled slice. They do not alter pack/run behavior. */
/* inflate w->Bi4 (all N channels) -> int8 codes as f32 in caller scratch qf32[N*K] (UNIFORM sign-extend
 * / NF4 LUT per quant_kind). Mirrors the inflate loop in pack_i4a8 / load_i4a8. */
/* force the inflate KIND (lets the bench time UNIFORM and NF4 on the same nibble store; the inflate
 * cost is data-independent, so it's a valid per-path microbench either way). */
/* tile inflated codes qf32[N*K] into w's existing resident DMA buffers (inv=1; codes are exact). Reuses
 * the production orki_i8_tile_f32 — same memcpy/quant + orki_bsync(TO_DEVICE) the steady-state stream would issue. */
/* DIRECT inflate ONLY (nibble -> linear int8 i8[N*K]); the rearrange/bsync is the separate tile step.
 * Lets the bench split direct inflate cost from the tile+bsync cost. */

/* ---- DIAGNOSTIC ONLY (tools/stream_prefetch_probe.c): a "staging slot" that splits the int4-streaming
 * swap into its three phases so a probe can time each AND run a real double-buffered loop:
 *   (a) FILL  = int4->int8 inflate + tile into a BARE (mmap'd, NOT yet IOMMU-mapped) dma-buf + dma-buf
 *               cache clean  -> the prefetchable CPU work (ork_stage_fill).
 *   (b) MAP   = PRIME_FD_TO_HANDLE + MEM_CREATE(handle) on each bare dma-buf -> IOVA; build an ork_w view
 *               over them  -> the swap-time zero-copy import (ork_stage_map).
 *   (c) RUN   = ork_i8_mm_run against the mapped view (ork_stage_run) -> the NPU submit.
 * ork_stage_unmap MEM_DESTROYs the maps (keeps the bare dma-buf+mmap for recycle); ork_stage_free closes.
 * This is exactly the int4 prefetch-inflate staging ring the design proposes, exposed for measurement
 * before promoting it into the library. Not in the public header. */

/* bare DMA-heap dma-buf: alloc + mmap, NO PRIME/MEM_CREATE (no IOVA yet). heap_fd = dma-buf fd. */
/* IOMMU-map an already-allocated bare dma-buf (sets dma/obj/handle). 0 ok / -1 fail. */
/* MEM_DESTROY the map (keep the dma-buf + mmap alive for recycle): clears dma/obj/handle only. */

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
 * src must be an int4-packed weight (ork_i4a8_mm_pack) with the same K,N. No IOVA needed (bare bufs).
 * This is the prefetchable CPU work — safe to call on a background thread (touches only this slot). */
void ork_stage_fill(ork_npu *c, struct ork_stage *s, const ork_w *src){
    if(!s || !src || !src->Bi4) return;
    int K=s->K, N=s->N, KS=1024, NMAX=c->soc->nmax, Sk=s->Sk, Sn=s->Sn, kind=src->quant_kind;
    int8_t *i8=s->i8scratch;
    if(kind==ORK_QK_CODEBOOK_NF4){ int8_t lut[16]; for(int i=0;i<16;i++) lut[i]=(int8_t)lrintf(ORKI_NF4_LEVELS[i]*127.0f);
        for(int n=0;n<N;n++) orki_nf4_inflate_chan_to_i8(src->Bi4+(size_t)n*(K/2),K,lut,i8+(size_t)n*K);
    } else for(int n=0;n<N;n++) orki_i4_expand_chan_to_i8(src->Bi4+(size_t)n*(K/2),K,i8+(size_t)n*K);
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
 * is parallelized across all cores (ork_parallel_for + orki_i8_tile_range) — same layout as orki_pack()/load_i8. */
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
int ork_stage_run(ork_npu *c, struct ork_stage *s, int M, const int8_t *A, int32_t *C){
    if(!s || !s->mapped) return -1;
    return ork_i8_mm_run(c, &s->view, M, A, C);
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
/* int8-stored: fill = copy the stored tile bytes (ork_w_dump blob) into the staging dma-bufs. Same blob
 * layout/validation as ork_i8_mm_load. NULL on import-unavailable / size mismatch. */
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
 * repack. Same int8/bscale result as feeding the equivalent f32 to orki_i8_tile_f32. */
void ork_w_free(ork_w *w){ if(!w)return; free(w->Bb); free(w->Bf); free(w->Bi4); free(w->bscale); free(w->pcrc); free(w->pcrc_meta); free(w->Bbc_ns); free(w); }   /* device buffers freed at ctx teardown */
/* Free a packed weight AND reclaim its NPU DMA/IOVA. Required for layer-streaming: evicted weights must
 * return their IOVA to the 4 GiB window (rk_iommu is 32-bit — see the wiki / npu-iova cap). Only weights
 * that OWN their buffers (per-tile bcreate: pack / pack_i4 / pack_i8) are reclaimed; weights whose tiles
 * are VIEWS into a single dedicated buffer (grouped-i4, own_buf_valid=1) reclaim that one buffer. */
void ork_mm_free(ork_npu *c, ork_w *w){
    if(!w) return;
    if(w->is_orkd){ if(c && c->daemon) orkd_free_weight(c->daemon, w->orkd_id); free(w->fa_lut); free(w); return; }   /* Path B: free the daemon-resident weight */
    if(w->cpu_codes){ free(w->cpu_codes); free(w->bscale); free(w->fa_lut); free(w); return; }   /* OFFLINE weight: plain heap, no device buffers */
    if(c) ork_dom_flush_if_dirty(c);   /* #54: clear any stuck job before a per-domain bdestroy switches domains ("failed to destroy memory" + switch-timeout cascade) */
    if(c && w->owns){
        size_t nb=(size_t)w->Sk*w->Sn;
        if(w->Bb) for(size_t i=0;i<nb;i++) if(w->Bb[i].cpu) orki_bdestroy(c->fd,&w->Bb[i]);
    }
    /* Bf is normally its own per-N-slice bcreate/orki_bimport (never a view), even when Bb is consolidated into
     * own_buf — so reclaim it whenever present, independent of owns. EXCEPTION: ork_i8_mm_adopt_imported lays
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
    if(w->cpu_codes) return (size_t)w->K*w->N;   /* OFFLINE: heap codes, no DMA residency to account for */
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
/* Native-int4 IMPORT twin of ork_i4_mm_load: identical DT_I4 tile layout (Kp*Nc/2 nibble bytes, KS=ORK_I4_KS,
 * no Bf) but allocated via orki_bimport (dma-heap + PRIME_FD into the IOMMU) instead of orki_bcreate (MEM_CREATE). MEM_CREATE
 * faults/EINVALs across non-0 domains AND at scale (a >4GiB resident int4 set — e.g. a resident MoE — hits the
 * per-domain window edge, the in-kernel rknpu_gem_object_create fault), so ork_i4_mm_load cannot bring a big int4
 * weight set resident. This mirrors the PROVEN multi-domain-safe consolidated-chunk import from ork_i8_mm_load_import:
 * a handful of moderate (~ORK_IMPORT_CHUNK_MB) dma-buf chunks, tiles are page-aligned base+offset VIEWS; ork_mm_free
 * bdestroys the chunks (own_bufs). Falls back to per-tile bimport on chunk-alloc failure. */
/* #54 CONSOLIDATED int4 expert load — MIRRORS ork_i8_mm_load_import EXACTLY, extended to share chunks ACROSS
 * experts. Same proven mechanism: bimport into ~ORK_IMPORT_CHUNK_MB (16MB) dma-buf chunks, tiles are page-aligned
 * base+offset VIEWS, fds sealed once a chunk is full (GEM handle keeps it alive for NPU reads). The ONLY change
 * vs the per-weight ork_i4_mm_load_import is that the chunk pool is PERSISTENT per-domain, so MANY experts share
 * a chunk instead of one dma-buf per expert (~9k imports -> ~2340 mappings/domain -> wedge; 16MB chunks pack
 * ~32 experts each -> a few hundred total, ~tens/domain — the count the int8 1.7B proves safe). Critically, like
 * int8 it does NOT set scratch_import: run scratch stays bcreate and coexists with the 16MB import chunks (the
 * PROVEN int8 model — bimport scratch of ANY kind wedges). Weight owns nothing (owns=0, own_bufs=NULL ->
 * ork_mm_free skips it); the shared chunks persist for the ctx, freed once in ork_npu_free. int4-only; resident
 * (no per-expert eviction — the whole MoE design goal). */
/* grouped pack: K split into groups of G (each its own resident slice) for per-group scales. G%32,
 * K%G, G<=10752. Sk = K/G groups; run_i4_grouped scales each group's partial before accumulating. */

int orki_i4_run_bchain_db(ork_npu *c,ork_w *w,int M,const int8_t *A,int32_t *C,int nc);  /* #52: BCHAIN batch on the nonblock doorbell */
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
/* Per-core phase timing (copy / submit / acc), read via ork_npu_mc_timing; the ORK_MCPROF env gate is removed (copy / submit / acc). Populated by the single-core
 * orki_run() path (the multi-core matmul now runs on the doorbell colsplit, which reports via its own
 * poll/backoff timers, not g_mc_*). Read via ork_npu_mc_timing. */
double orki_mc_copy[MCPROF_MAX], orki_mc_sub[MCPROF_MAX], orki_mc_acc[MCPROF_MAX]; long orki_mc_n[MCPROF_MAX];
double orki_mc_synth[MCPROF_MAX];   /* host regcmd-synth+bsync portion of orki_mc_sub (the OVERLAPPABLE part; ioctl/NPU = sub-synth) */

/* Pin the calling thread to a LITTLE core (low-numbered: A55 0-3 on RK3588). For off-critical-path /
 * IO-bound / memory-bound work (e.g. the SSM double-buffer marshalling helper) that should run on the
 * idle little cluster WHILE the big cores are saturated by ggml's threadpool + the NPU pool. The A55 is
 * ~2x slower but it's free time overlapped with the NPU submit. Honors ORK_NO_AFFINITY. */
double orki_rt_setup=0, orki_rt_submit=0, orki_rt_copy=0; long orki_rt_n=0;
/* FLOOR-DECOMP accessors: ioctl_us = total wall inside SUBMIT ioctls; hw_us = total sub->hw_elapse_time as
 * reported by the kernel (raw units — see hw_raw_last); n = count. See globals near orki_fd_ioctl_us. */


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
       * ORK_COLSPLIT_MGT1 — removed) existed because 23af039's first M>1 colsplit regressed prefill ~15x (per-K-slice
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
                uint32_t rc[REGCMD_N]; orki_i8_synth(rc,mc,Kp,Nc,adma,(uint32_t)wbase,cdma,sched,CBUF,cbuf?N:Nc);
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
        int sched=dt?(Kp==1024||Kp==512):orki_f16_sched(Kp), R=RB/Kp; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; } int chunk=sched?4*R:((RB/2)/Kp); if(chunk<1)chunk=1;
        /* fp16: the M-tile is the MEASURED envelope, not 4*R / (RB/2)/Kp. Those were derived from
         * int8 (32768 ELEMENTS) and are 2x too loose for fp16 (2 B/elem => 16384 elems = 1 CBUF
         * bank), and the sched=1 form overshot badly (4*R = 1024 @K=128 vs a real ceiling of 256,
         * which is why ork_f16_mm_run silently miscomputed there for any M in [257,1472]).
         * int8 keeps its own path untouched — it is measured-correct at mg_max*64. */
        if(!dt) chunk=orki_f16_mcap(Kp,sched);
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
            if(dt==DT_F16) orki_f16_synth   (rc,mc,Kp,Nc,(uint32_t)c->Af.dma,(uint32_t)Bb->dma,(uint32_t)c->Cc.dma,sched,CBUF);
            else           orki_i8_synth(rc,mc,Kp,Nc,(uint32_t)c->Af.dma,(uint32_t)Bb->dma,(uint32_t)c->Cc.dma,sched,CBUF,Nc);
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
int ork_f16_mm_run   (ork_npu *c,ork_w *w,int M,const f16    *A,float   *C){
    if(w && w->is_orkd){   /* Path B: fp16 run on the daemon — ring transport if attached (any precision), else socket */
        orkd_set_op_domain(c->daemon, (uint32_t)w->domain);   /* v2: carry this weight's domain with the op */
        if(c && c->daemon && orkd_has_ring(c->daemon)){ int r=orkd_ring_run(c->daemon,w->orkd_id,M,w->K,w->N,ORKD_DT_F16,A,C); if(r!=-2) return r; }
        return orkd_run_f16(c->daemon, w->orkd_id, M, w->K, w->N, A, C); }
    if(w->dtype!=DT_F16)return -1;
    if(orki_check_overlap("ork_f16_mm_run", (uintptr_t)A, (uintptr_t)A + (size_t)M * w->K * 2, (uintptr_t)C, (uintptr_t)C + (size_t)M * w->N * 4)) return -1;
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


/* PRECISION DISPATCH. int8 + int4 have refuse sites and thus a sliced rescue. fp16 has NO refused shapes
 * (out-of-envelope fp16 -> ORK_RC_F16_SC -> single-core reference, a working path; its multicore fit is the
 * CONTIG colsplit, not tiles) -> no fp16 tiled path. */
ork_w_sliced *ork_mm_pack_sliced(ork_npu *c, int K, int N, const void *B, int dtype) {
    if (!c || !B || K <= 0 || N <= 0) return NULL;
    switch (dtype) {
        case DT_I8:  return orki_i8_slice_pack (c, K, N, (const int8_t *) B);
        case DT_I4:  return orki_i4_slice_pack (c, K, N, (const int8_t *) B);
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
 * K<=8192) and run EACH via orki_i4_run_bchain_db (M>=2) or the per-row doorbell (M==1) — reusing #52's
 * self-healing / pool / de-tile machinery as a black box — then int32-accumulate the K-slices + scatter N
 * (int4 C is int32 after BCHAIN's de-tile, so the int8 orki_slice_acc_worker applies verbatim). Tiles run
 * sequentially (each internally multi-core); a tile that refuses/fails fails the whole rescue -> the caller
 * refuses (never a blocking fall-back, per #45/#52). */

/* PRECISION DISPATCH. int8 + int4 (fp16 has no refused shapes -> uses colsplit, not the tiled surface). */
int ork_mm_run_sliced(ork_npu *c, ork_w_sliced *w, int M, const void *A, void *C, int nc) {
    if (!w) return -1;
    switch (w->dtype) {
        case DT_I8:  return orki_i8_slice_run (c, w, M, (const int8_t *) A, (int32_t *) C, nc);
        case DT_I4:  return orki_i4_slice_run (c, w, M, (const int8_t *) A, (int32_t *) C, nc);
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

/* ⚠ CLOSED — the matmul-fused activation output is INT8-ONLY (2026-07-05 sweep). The output-precision is a
 * PREC field in reg 0x4010 (int8=bits00, int16=01, fp16=10; bit31=int32 which BYPASSES the CVT the LUT needs).
 * SWEPT PREC=1 (int16) across 3 output-stride configs (0x40c0/0x4050/0x4038) — ALL soft-reset the NPU + garbage.
 * int32 (0x800044e0) also wedges (CVT bypass vs LUT conflict). So the MATMUL+LUT program only supports int8
 * output; int16/fp16 output exists only in the STANDALONE silu program (0x50xx lane: int16=0x24004401,
 * fp16=0xa8000002 — different regcfg). No RKNN capture possible (RKNN never fuses activation into non-int8
 * matmul). CONCLUSION: higher-precision fused silu is not achievable in the matmul program. The ablation
 * (ORK_GATE_ABLATE — historical) proved int8 silu OUTPUT is the whole FFN-chain PPL gap, so the remaining route to parity
 * is UN-FUSED: int32 matmul -> standalone int16/fp16 silu op (loses the silu-free-on-NPU fusion, adds a submit).
 * Below is the sweep harness (env-configurable format regs), kept for the record; NOT called by the chain. ── */



/* ork_f16_mm_run_silu — fp16 gate matmul + fused SiLU with fp16→fp32 output (no int8 activation quant). The
 * "end-goal" precise on-NPU gate: recovers the full PPL gap the int8 silu output loses (ablation), at the cost
 * of the fp16 matmul (~3.3x int8, tools/f16_gate_bench) — a measured net-loss TODAY, so gated OFF, built out
 * for a future pipeline where it pays off. w = fp16 weight (ork_f16_mm_pack), A = fp16 [M,K], C = fp32 [M,N] silu.
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


/* DOORBELL PIPELINE PROFILER (Tier 11): measure the wall of `iters` SERIAL int8 matmuls, BLOCKING vs
 * NONBLOCK + DRAM-doorbell busy-poll. Blocking = submit waits (~130µs floor + compute)/op. Nonblock = submit
 * returns ~5µs, CPU busy-polls the output SENTINEL (DC CIVAC invalidate + read) until the NPU overwrites it =
 * op done, no sleep/wake. Same op (all-ones int8 -> every output = K) both ways; validates output == K. Calls
 * the raw ioctl directly with explicit flags (bypasses the env/sleep wrapper). 0/ok, <0 err. */

/* overlap_prof — Tier 11: does REAL CPU work in the shadow of an async NPU op stay FREE (overlap wall ~= max),
 * or does shared LPDDR4X bandwidth contention stretch it (wall -> npu+cpu)? This is the "zero-time router"
 * thesis (speculative/batched MoE) in one number. cpu_reps = # of 512x512 fp32 GEMVs run on the CPU between
 * the NONBLOCK submit and the doorbell poll (a stand-in for the routing math; 1MB matrix -> real DRAM traffic).
 * Fills npu_solo (submit+poll, no CPU work), cpu_solo (the GEMVs alone), overlap_wall (submit+CPU-work+poll). */


/* RE: STANDALONE fp16 matmul with FP16 output (set_f16_out, 0x4010=0x48000002 fp16-in) in the EWCUBEH atom-8
 * layout the chained SDP consumes — isolates "can the fp16 matmul emit fp16 G correctly" from the chain handoff.
 * A[M*K],B[K*N] fp16 bit patterns; out[M*N] fp16 read via EWCUBEH. task_number=1. 0/ok,-1 wedged,-2 dims. */

/* A1 (task #20): fp16 matmul with CONTIGUOUS fp16 output, consuming a PACKED resident ork_w (w->Bb) via the
 * PROVEN vendor fp16-out stage (orki_f16_set_out_fp16in, default contiguous). The point: a matmul's output is fp16
 * [M,N] directly — so it feeds an fp16 SDP op with NO f32->f16 host narrow between them, making the pair
 * ADJACENT (A2 can then keep the intermediate resident). Single-tile (M<=64, single-slice), single-core; it
 * runs its OWN submit (not the f32-out doorbell), so its SEQ_CLASS row is hw=0 (per-op SW dispatch). This is
 * the twin of ork_f16_npu_probe_mm_f16out but weight = w->Bb (resident) instead of a raw-B rebuild.
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


/* On-NPU per-row MAX-REDUCE (int8): out[m] = max_n a[m*N+n], N%16, N<=8192. Batched pairwise-max TREE
 * using the SDP EW ALU in MAX mode (EW_ALU_ALGO=0 @ reg 0x4070, rocket_registers.h; NVDLA map_alu_op
 * MAX=0). Each level maxes channels [0,h) vs [h,2h) across ALL M rows in ONE submit: channel h sits at
 * cube offset (h/16)*M*16, so operand b = a + (h/16)*M*16 (valid since h stays %16 down the tree). Reduces
 * N->16 on-NPU (log2(N/16) submits, ping-pong buffers), then a 16-wide CPU tail. Reusable: softmax
 * stability, max-pool, top-k, clamp. 0/ok, <0 err. */


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
 * Config regs were env-overridable (ORK_BS_R40/R48/BRDMA — removed) to pin DATA_USE/enable on-board. int8. 0/ok. */

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


/* fp16 standalone activation-LUT op — RE probe. Applies the PPU LUT to a SINGLE fp16 memory input [M][N] via
 * REGCMD_SILU_STD_F16 (single-input, fp16 gain 0x00010001). Two submits: REGCMD_SILU_LUT (load) + this op.
 * The LUT data words are streamed via 0x4104 verbatim (encoding TBD by calibration — pass lut as the raw 16-bit
 * words the op consumes). in/out fp16 [M*N], N%8==0; rk3588-gated. 0/ok,-1 wedged,-2 shape,-3 SoC. */


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



/* fp16-IN chained matmul -> per-channel SDP (the attention A·V normalize path, all-fp16 like the vendor conv->mul).
 * Closes the single-submit chain with CORRECT values: fp16 matmul (synth, weight tile [N/16][K/32][16][32], fp16
 * activation) -> fp16-out G (set_f16_out with 0x4010=0x48000002 fp16-in) -> the vendor REGCMD_MUL_F16_CHAIN 2-input
 * SDP scales G per-channel by scale[N]. A[M*K],B[K*N],scale[N],out[M*N] are fp16 bit patterns (uint16). 0/ok,-1,-2,-3. */

/* (B') fp16 cross-slice DRAIN-GAP probe. Chains TWO wide fp16 matmuls with DISTINCT weight buffers (W0,W1) —
 * reproducing the cross-buffer base-latch condition — into ONE PC-chain submit, optionally with an identity
 * mul_perchan_f16 GAP between them (REGCMD_MUL_F16_CHAIN, enable 0x18, 69 regs, middle desc_slot=138 per the
 * regcfg*2 convention). The gap is a pure time-filler on a DUMMY fp16 scratch (identity scale) — its only job is
 * to idle the weight-CDMA so W0's fetch drains before W1's base latches. Both matmuls use the fp16-out stage
 * (orki_f16_set_out_fp16in — the config with the PROVEN mm->perchan HW edge). Returns the submit rc (nonzero=wedge);
 * *nz0 / *nz1 = count of nonzero output words per matmul (coarse "did it compute" sanity). rk3588-gated. */

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

/* ORKD fused attention core against 3 resident (is_orkd) weights (K^T[Kp,Nk], ones[Nk,32], V[Nk,dv]) — one
 * ORKD_ATTN round-trip, one on-NPU chain (QK^T->exp->reduce,e.V), e never leaves the NPU. Sigma[Nq*32] +
 * av[Nq*dv] returned; caller normalizes attn=av/Sigma. Thin wrapper over the orkd_attn_i8 client. */

/* Transport-transparent whole-layer op. c->daemon set => forward as one ORKD_LAYER round-trip; else run locally
 * (the orkd daemon's handle_layer lands here on its own direct ctx). See the header for the compute contract. */



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
 * scores without a live per-query max. See ork_i8_npu_exp_biased / orki_silu_build_curve_biased.
 * (Defined before the plain wrapper below: the standalone Makefile build does not pull ork_npu.h into this TU.) */


/* Run `nchains` fused exp chains round-robin across the cores (concurrent). chains[i] = that chain's S[i]-task
 * array; ops = the shared op graph; (in_scale,out_scale) the shared exp requant. Returns 0/ok, <0 err (first
 * failing chain's code). PRECONDITION: cores must be WARM (a prior matmul on each) — a cold core's first submit
 * wedges; a chain-only caller should warm via a multi-core matmul first (see chainrr_conc_probe). Local NPU only. */

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

/* Drain the grouped-int4 doorbell: poll all rows' Sk*N int16 partials, then FLOAT scale-accumulate into C[M,N]. */

/* per-op eligibility (Stage 2/3 envelope); records dom of the first matmul weight into *dom / *have_dom */
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
        orki_i8_synth(rc,M,K,N,adma,wdma,(uint32_t)(h->seq_out.dma+*coff),1,CBUF,0);
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
/* Drain (until complete or a stall => halted), write outputs back from DMA, free. Returns highest op done. */

/* ncore<=1 => single-core chain (begin); ncore>1 => multi-core (the ORK_DYN_MC override is removed) NONBLOCK stream (begin_mc). */
/* submit the next pending chunk NONBLOCK (NPU runs while the caller works); no-op if one is already flying */
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

/* STAGE 1 PROBE — heterogeneous NONBLOCK chain on the begin_mc RECIPE (not chain_progs). Builds [matmul ->
 * ewmul(int8 SDP, middle) -> matmul] as ONE core's PC-chain in mrc[0]/maf[0] + a warmed OUTPUT scratch (fresh
 * bcreate + clean-before, exactly like begin_mc's cold mcc — the 79f809c coherency fix that chain_progs never
 * got), NONBLOCK submit (ping-pong OFF for the SDP), completion via the TERMINAL matmul's int32 sentinel poll.
 * Proves the SDP-doorbell mechanism produces bit-exact matmul AND ewmul output (the thing the chain_progs-based
 * nb probe could not — its matmuls were empty on the superseded fresh-buffer path). *ok = all three bit-exact. */



/* FUSED SSD-SCAN MATMUL BENCH: chain ALL grouped-scan matmuls of ONE Mamba-2/SSD layer into a SINGLE
 * PC-chained submit with RESIDENT all-ones operands (no per-batch repack), vs the SAME matmuls as N
 * separate submits (each paying the ~48µs per-submit floor). Isolates the floor-amortization the fused
 * on-NPU scan graph would get. All-ones int8 => every output element == its own K, so a fused-chain
 * output that differs signals a wedge/miscompute. Group-batched shapes (L-factorization; Y_diag grouped):
 *   scores  [CS, Nst]x[Nst, CS]      cstate  [HG*P, CS]x[CS, Nst]
 *   Ydiag_g [CS, CS]x[CS, HG*P]      Y_off   [HG*P, Nst]x[Nst, CS]
 * each x (G*NC) batches. M<=HG*P fits ONE M-tile at these K (mcap(K=64)=10496). Returns 0/ok, <0 on
 * error; fills *fused_us / *persub_us (per-iter wall) and *ok_out (1 = fused chain bit-correct). Board only. */

/* (b) LAYOUT PROBE: run ONE fp16 matmul through the exact fused-chain mechanism (raw synth + a single
 * ork_npu_chain_progs task) with ROW-MAJOR resident operands A[M,K],B[K,N] → C[M,N] fp32. Decides
 * whether a real-operand fused SSD scan can stage row-major operands directly, or must replicate the
 * ork_f16_mm_pack 32x32-block tiling. K%32,N%16. 0/ok,<0 err. rk3588. Diagnostic — not a production path. */

/* (b) FUSED-MM probe: one fp16 matmul via the fused-chain synth mechanism, but B is PACKED with
 * ork_f16_mm_pack (→ the tiled Bb layout synth actually reads, same as orki_run()) and A is staged ROW-MAJOR,
 * C read DENSE. Determines whether the real-operand fused SSD chain can reuse ork_f16_mm_pack for B + a
 * row-major A (vs needing to hand-tile A). Single-slice only (K<=ks, N<=nmax). C[M,N] fp32. 0/ok,<0. */

/* FUSED batched fp16 GEMM: like ork_bmm_fp16 (nbatch matmuls C[b]=A[b]*B[b], both operands dynamic), but
 * chains ALL nbatch matmuls into ONE PC-chained submit instead of one submit per batch — amortizing the
 * ~48us/submit NPU floor across the batch (the SSD scan's per-stage H-batch = the target). Each B is packed
 * via ork_f16_mm_pack (its tiled Bb is what synth reads); A staged row-major; C dense. A[nb*M*K], B[nb*K*N],
 * C[nb*M*N] fp32. Single-slice only (K<=ks, N<=nmax) — the scan's small shapes qualify; nb<=64. 0/ok,<0.
 * Numerically identical to ork_bmm_fp16 (same synth matmul, same fp16 operands) — just one ioctl. */


/* Run S independent int8 matmuls as an async round-robin stream across the NPU cores. Each task's weight
 * must be single-slice (Sk==1 or Bf full-K) and single N-slice; A/C are plain host buffers (copied via the
 * per-core staging buffers — no zero-copy DMA here). Returns 0/ok, -1 submit fail, -2 bad arg. */


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
        case OAK_F16:      h->rc = ork_f16_mm_run        (h->c, h->w, h->M, (const f16*)h->A, (float*)h->C); break;
        case OAK_I8:       h->rc = ork_i8_mm_run     (h->c, h->w, h->M, (const int8_t*)h->A, (int32_t*)h->C); break;
        case OAK_I4:       h->rc = ork_i4_mm_run     (h->c, h->w, h->M, (const int8_t*)h->A, (int32_t*)h->C); break;
        case OAK_CHAIN_I8: h->rc = ork_i8_mm_run_chain (h->c, h->S, (const ork_mm_task_i8*)h->tasks); break;
        case OAK_CHAIN_I4: h->rc = ork_i4_mm_run_chain (h->c, h->S, (const ork_mm_task_i4*)h->tasks); break;
        case OAK_STREAM_I8:h->rc = ork_i8_mm_run_stream(h->c, h->S, (const ork_mm_task_i8*)h->tasks); break;
        case OAK_STREAM_I4:h->rc = ork_i4_mm_run_stream(h->c, h->S, (const ork_mm_task_i4*)h->tasks); break;
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
ork_async *ork_i4_mm_run_async (ork_npu *c, ork_w *w, int M, const int8_t  *A, int32_t *C){
    if (!c || !w || M < 1) return NULL;
    return ork_async_launch((struct ork_async){ .kind=OAK_I4, .c=c, .w=w, .M=M, .A=A, .C=C }); }
ork_async *ork_i4_mm_run_chain_async (ork_npu *c, int S, const ork_mm_task_i4 *tasks){
    if (!c || S < 1 || !tasks) return NULL;
    return ork_async_launch((struct ork_async){ .kind=OAK_CHAIN_I4, .c=c, .S=S, .tasks=tasks }); }
ork_async *ork_i4_mm_run_stream_async(ork_npu *c, int S, const ork_mm_task_i4 *tasks){
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
 * functions verbatim, so it works identically for fp16 (ork_f16_mm_run), int8 (ork_i8_mm_run), int4
 * (ork_i4_mm_run), and the chain/stream variants — nothing here is precision-specific (the int4
 * i4a8-inflated-to-int8 path is just a DT_I8 weight, so it rides the i8 entry). No kernel-fence
 * dependency that could wedge.
 *
 * Thread-safety: the caller MUST keep at most ONE async job in flight and MUST NOT touch A/C or
 * issue any other ork_mm_* on the same ctx between launch and wait (only independent CPU work) — the
 * NPU is single-stream, so there is never a concurrent submit racing the submit domain / ctx scratch. */
/* --- SW dispatch shims: adapt the generic ork_seq_op to each reliable driver function's signature --- */
static int seq_disp_i8_mm  (ork_npu *c,const ork_seq_op *o){ return ork_i8_mm_run(c,o->w,o->M,(const int8_t*)o->A,(int32_t*)o->C); }
static int seq_disp_f16_mm (ork_npu *c,const ork_seq_op *o){ ork_mm_task_f16 t={o->w,o->M,(const f16*)o->A,(float*)o->C}; return ork_f16_mm_run_stream(c,1,&t); }
static int seq_disp_i4_mm  (ork_npu *c,const ork_seq_op *o){ return ork_i4_mm_run(c,o->w,o->M,(const int8_t*)o->A,(int32_t*)o->C); }   /* multi-M rides the #4 doorbell route inside run_i4 */
static int seq_disp_ewmul_f16(ork_npu *c,const ork_seq_op *o){ double us; return ork_f16_npu_ewmul(c,(const f16*)o->A,(const f16*)o->B,o->M,o->N,(f16*)o->C,&us); }
static int seq_disp_silu_i8 (ork_npu *c,const ork_seq_op *o){ double us; return ork_i8_npu_silu(c,(const int8_t*)o->A,o->M,o->N,o->in_scale,o->out_scale,(int8_t*)o->C,&us); }
static int seq_disp_silu_i16(ork_npu *c,const ork_seq_op *o){ double us; return ork_i16_npu_silu(c,(const int16_t*)o->A,o->M,o->N,o->in_scale,o->out_scale,(int16_t*)o->C,&us); }
static int seq_disp_gelu_i8 (ork_npu *c,const ork_seq_op *o){ double us; return ork_i8_npu_gelu(c,(const int8_t*)o->A,o->M,o->N,o->in_scale,o->out_scale,(int8_t*)o->C,&us); }
static int seq_disp_ewmul_i8(ork_npu *c,const ork_seq_op *o){ double us; return ork_i8_npu_ewmul(c,(const int8_t*)o->A,(const int8_t*)o->B,o->M,o->N,o->mult,o->shift,(int8_t*)o->C,&us); }
static int seq_disp_add_i8  (ork_npu *c,const ork_seq_op *o){ double us; return ork_i8_npu_add(c,(const int8_t*)o->A,(const int8_t*)o->B,o->M,o->N,o->in_scale,o->b_scale,o->out_scale,(int8_t*)o->C,&us); }
static int seq_disp_add_f16 (ork_npu *c,const ork_seq_op *o){ double us; return ork_f16_npu_add(c,(const f16*)o->A,(const f16*)o->B,o->M,o->N,(f16*)o->C,&us); }
/* Uniform single-input SDP activations beyond the original seq subset (same (in,M,N,in_scale,out_scale,out) shape). */
static int seq_disp_gelu_i16(ork_npu *c,const ork_seq_op *o){ double us; return ork_i16_npu_gelu(c,(const int16_t*)o->A,o->M,o->N,o->in_scale,o->out_scale,(int16_t*)o->C,&us); }
static int seq_disp_rsqrt_i8(ork_npu *c,const ork_seq_op *o){ double us; return ork_i8_npu_rsqrt (c,(const int8_t*) o->A,o->M,o->N,o->in_scale,o->out_scale,(int8_t*) o->C,&us); }
static int seq_disp_exp_i8  (ork_npu *c,const ork_seq_op *o){ double us; return ork_i8_npu_exp_biased(c,(const int8_t*) o->A,o->M,o->N,o->in_scale,o->out_scale,o->b_scale,(int8_t*) o->C,&us); }  /* b_scale = scalar max-bias (0 = plain exp) */
static int seq_disp_exp_i16 (ork_npu *c,const ork_seq_op *o){ double us; return ork_i16_npu_exp  (c,(const int16_t*)o->A,o->M,o->N,o->in_scale,o->out_scale,(int16_t*)o->C,&us); }
/* Softmax / RMSNorm normalize primitives (task #20 attention & full-layer chain). row-max: softmax max-shift
 * (reduce N->1). mul_perchan: the normalize A*b with b=1/Σ per query (also A.V per-channel scale, RMSNorm affine)
 * — 2-input, takes the per-channel vector via o->B. rsqrt_i16: 1/Σ (softmax, via rsqrt(Σ²)) and 1/√mean² (RMSNorm),
 * the accurate int16 LUT variant mirroring seq_disp_rsqrt_i8. */
static int seq_disp_reducemax_i8   (ork_npu *c,const ork_seq_op *o){ double us; return ork_i8_npu_row_max   (c,(const int8_t*) o->A,o->M,o->N,(int8_t*) o->C,&us); }
static int seq_disp_mul_perchan_f16(ork_npu *c,const ork_seq_op *o){ double us; return ork_f16_npu_mul_perchan(c,(const f16*)    o->A,(const f16*)    o->B,o->M,o->N,(f16*)    o->C,&us); }
static int seq_disp_mul_perchan_i8 (ork_npu *c,const ork_seq_op *o){ double us; return ork_i8_npu_mul_perchan (c,(const int8_t*) o->A,(const int8_t*) o->B,o->M,o->N,o->mult,o->shift,(int8_t*) o->C,&us); }
static int seq_disp_rsqrt_i16      (ork_npu *c,const ork_seq_op *o){ double us; return ork_i16_npu_rsqrt     (c,(const int16_t*)o->A,o->M,o->N,o->in_scale,o->out_scale,(int16_t*)o->C,&us); }
/* NEOX RoPE (task #20 attention chain): x[nrow,hd] -> out. Weightless; internally ewmul_f16+ewmul_f16+add_f16
 * (self-contained SW-break). Field overload: pos[] (per-row positions) via o->B, freq_base via o->in_scale. */
static int seq_disp_rope_neox_f16  (ork_npu *c,const ork_seq_op *o){ return ork_f16_npu_rope_neox (c,(const f16*)o->A,o->N,o->M,(const int*)o->B,o->in_scale,(f16*)o->C); }
/* RMSNorm (task #20 layer chain): x[M,n] -> out, per-channel gain via o->B, eps via o->in_scale. Weightless
 * matmul-wise (the gain is an SDP operand, not a packed ork_w); self-contained SW-break. */
static int seq_disp_rmsnorm_f16    (ork_npu *c,const ork_seq_op *o){ return ork_f16_npu_rmsnorm   (c,o->M,o->N,(const f16*)o->A,(const f16*)o->B,(float)o->in_scale,(f16*)o->C); }
/* A1: fp16 matmul with contiguous fp16 output (packed w) — SW-dispatch (own submit, not the f32-out doorbell). */
static int seq_disp_f16_mm_f16out  (ork_npu *c,const ork_seq_op *o){ return ork_f16_mm_run_f16out (c,o->w,o->M,(const f16*)o->A,(f16*)o->C); }
/* A1 int8: matmul with int8 requant output (int8 in, int8 out) — the all-int8 softmax island's int8 x-max
 * broadcast (max_i8 . -ones -> -max_bc int8) + any int8 matmul->SDP feed. mult/shift via o->mult/o->shift. */
static int seq_disp_matmul_requant_i8(ork_npu *c,const ork_seq_op *o){ return ork_i8_mm_run_out8(c,o->w,o->M,(const int8_t*)o->A,(int8_t*)o->C,o->mult,o->shift); }
/* int8 matmul -> int16 compact-linear out (set_i16_out); feeds an int16 SDP op resident, no PC-chain bridge. */
static int seq_disp_matmul_i16out_i8(ork_npu *c,const ork_seq_op *o){ return ork_i8_mm_run_out16(c,o->w,o->M,(const int8_t*)o->A,(short*)o->C,o->mult,o->shift); }
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
    c->seq_witness = ork_i8_mm_pack(c, 512, 16, wz);
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
                    if(ork_f16_mm_run_stream(c,1,&_t)) ret=-1; } \
                else if(bdt==DT_I4){ ork_npu_enter(c,5/*I4_STRM*/,XP_I4_STREAM,OCK_SW); \
                    ork_mm_task_i4 _t={batch[_q].w,batch[_q].M,batch[_q].A,batch[_q].C}; \
                    if(ork_i4_mm_run_stream(c,1,&_t)) ret=-1; } \
                else { ork_npu_enter(c,DT_I8,XP_MC_MM,OCK_SW); \
                    if(ork_i8_mm_run(c,batch[_q].w,batch[_q].M,batch[_q].A,batch[_q].C)) ret=-1; } } } \
        nb=0; } }while(0)
    for(int i=0;i<n && !ret;i++){
        const ork_seq_op *o=&ops[i];
        if((int)o->kind<0||(int)o->kind>=ORK_OP_NKIND){ ret=-2; break; }
        /* GROUPED RUN (Stage 4): a maximal contiguous run of group>0 ops rides ork_i8_dyn_begin_seq_mc — a
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
                h = (j-i<=ORK_SEQ_HWBATCH) ? ork_i8_dyn_begin_seq_mc(c, j-i, &ops[i], ng, gs, 0) : NULL;
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
                    if(!of) h = ork_i8_dyn_begin_seq_mc(c, an, ao, ang, ags, 0);
                }
            }
            if(getenv("ORK_SEQ_DEBUG")) fprintf(stderr,"[seq] grouped run [%d,%d) ng=%d -> %s\n", i,j,ng, h?"seq-chain":"SW-fallback");
            if(h){ if(ork_dyn_seq_end(h)) ret=-1; }
            else { for(int p=i;p<j && !ret;p++){ const ork_seq_op *op=&ops[p]; const struct ork_seq_class *pcl=&SEQ_CLASS[op->kind];
                    if(pcl->hw && seq_hw_ok(op)){ ork_mm_task_i8 t1={op->w,op->M,(const int8_t*)op->A,(int32_t*)op->C};
                        ork_dyn_chain *hh=ork_dyn_begin_mc(c,1,&t1,0); if(hh){ if(ork_dyn_end(hh)<0){} } else { ork_npu_enter(c,DT_I8,XP_MC_MM,OCK_SW); if(ork_i8_mm_run(c,op->w,op->M,op->A,op->C))ret=-1; } }
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
 * B[b] is packed fresh each batch (unlike the resident-weight ork_f16_mm_run* paths). Correctness-first:
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
ORK_BMM_GATHER(orki_i8_bmm_gather, int8_t)
ORK_BMM_GATHER(orki_f16_bmm_gather, f16)
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
/* NPU reduction: ss[m] = sum_j x[m,j]^2 via the (x.*x)*ones matmul. Returns 0/ok, <0 to signal CPU fallback. */
/* Cached fp16 rsqrt LUT (BUILD-ONCE) + a K=512 op mapping ss[M] -> scale[M]=1/sqrt(ss/nf+eps) ON the NPU,
 * DECOUPLED from the reduce so it works for ANY feature dim n (reduce K-splits; this op is K=512 = the LUT
 * builder's known-good geometry — K=32 is DEGENERATE for the fp16 fused-silu tiling, gives acc~0). The scalar
 * ss is fed DENSE + NORMALIZED: A[m,k]=ss[m]/G for all Kd cols, weight=-S*G/Kd -> acc = -S*ss (A stays ~O(1),
 * weight small — exactly the probe regime). G = the calibration upper bound orki_rs.hi. nf: rmsnorm=n, l2norm=1.
 * Single-slot cache (rebuilds when ctx/nf/eps change or ss drifts outside [lo,hi]); norm calls single-threaded. */
/* scale[m] = 1/sqrt(ss[m]/nf + eps) on the NPU (K=512 fused rsqrt). 0/ok, <0 -> caller uses CPU rsqrt. */


/* Fast Walsh-Hadamard Transform (FWHT) - Exposed utility function for caller-driven quantization */

/* ==== CPU-side pack/dump helpers linked by the ggml-ork backend (no internal callers) ==== */
