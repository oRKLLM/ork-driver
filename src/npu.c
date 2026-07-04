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
#include <unistd.h>
#include <sys/mman.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <math.h>
#include <errno.h>
#include "rknpu_ioctl.h"
#include "regcmd_array_4x32x16.h"
#include "regcmd_i8.h"
#include "regcmd_i4.h"
#include "regcmd_silu.h"
#include "regcmd_ewmul.h"
#include "ork_npu.h"
#include "soc.h"
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
typedef ork_f16 f16;
enum { DT_F16=0, DT_I8=1, DT_I4=2 };
/* last_dt SCHEDULE markers that all run int8 regcmd PROGRAMS on the NPU: single-core (DT_I8=1) and the
 * chain/stream path (3, "DT_I8_CHAIN"). Switching AMONG these is not a hardware MODE change, so it needs
 * no RKNPU_ACT_RESET — the reset is only for ENTERING int8 from fp16/int4/cold (the first-int8-submit
 * wedge). Decoupling the reset from the marker lets decode interleave run_i8 singletons with run_stream
 * (QKV/gate-up) groups without a ~107ms NPU soft-reset at every matmul boundary. The cold 2-pass warmup
 * (fresh-output-buffer priming) is kept independently — see the reset sites. */
#define ORK_I8_LIVE(dt) ((dt)==DT_I8 || (dt)==3)

/* ORK_PROFILE: per-matmul host-side timing, printed on free. Lets us see how much of decode's
 * per-token wall time is spent inside ork-driver's matmul calls vs the ggml/CPU path around them. */
static double ork_now_us(void);
static int    g_ork_prof = 0;
static long   g_prof_i8_calls = 0, g_prof_i4_calls = 0;
static double g_prof_i8_us = 0,    g_prof_i4_us = 0;
/* RKNPU_SUBMIT ioctl counter (ORK_PROFILE). g_prof_submits = total ioctls; the run path tags each
 * via g_prof_submit_class so we can split within-matmul tiling (K/N/M sub-submits) from chained
 * (one ioctl covering >1 program). Printed on free. Pure diagnostic — no effect when prof off. */
static long   g_prof_submits = 0;       /* total RKNPU_SUBMIT ioctls */
static long   g_prof_submit_progs = 0;  /* total PC-chained programs across all submits (>= submits) */
static long   g_prof_submit_chained = 0;/* submits that carried >1 program (chained) */
#define ORK_MAXCORE 4   /* RK3576=2, RK3588=3; headroom for future parts. Actual = soc->cores. */

struct buf { uint32_t handle; uint64_t dma, obj; void *cpu; size_t size; int heap_fd; };  /* heap_fd: for zero-copy IMPORTED bufs (ork_dma_import / bimport) the dma-buf fd to close on destroy; 0 for ordinary MEM_CREATE-allocated bufs. */
struct ork_pw { struct ork_npu *c; int id; };   /* persistent NPU-pool worker arg */
#define ORK_MAXDOM 16
/* Parked per-domain copy of the submit-touched scratch (see ork_npu.dom_save). Mirrors exactly the
 * ork_npu fields that name DMA buffers a submit references + their warm/size bookkeeping. cres is host
 * RAM (domain-agnostic) so it is NOT parked here. */
struct ork_dom_scratch {
    int used;
    struct buf regcmd, task, Af, Cc; size_t ccsz; int warmed;
    struct buf mrc[ORK_MAXCORE], mtk[ORK_MAXCORE], maf[ORK_MAXCORE], mcc[ORK_MAXCORE], mtk_all;
    size_t mccsz[ORK_MAXCORE]; int mwarm[ORK_MAXCORE]; int mc_alloc;
};
struct ork_npu { int fd; const struct ork_soc *soc; struct buf regcmd, task, Af, Cc; size_t ccsz; void *cres; size_t cressz; int warmed, last_dt; int core_budget;
    /* multi-core (ORK_NPU_MC): per-core regcmd/task/feature/output so cores submit concurrently */
    struct buf mrc[ORK_MAXCORE], mtk[ORK_MAXCORE], maf[ORK_MAXCORE], mcc[ORK_MAXCORE], mtk_all;
    size_t mccsz[ORK_MAXCORE]; int mwarm[ORK_MAXCORE]; int mc_alloc;
    /* PER-DOMAIN SCRATCH (multi-domain residence): a submit runs in ONE iommu_domain_id, so EVERY buffer
     * it touches (regcmd, task, activation Af, output Cc, and the per-core multi-core scratch) must live in
     * the same domain as the weight. The fields above are the ACTIVE working set; dom_active is which
     * domain they currently belong to. dom_save[d] parks a domain's working set when switching away, so
     * each domain keeps its own (cheap, MB-scale) scratch resident — no realloc on every weight. */
    int dom_active; int dom_seen[ORK_MAXDOM];
    struct ork_dom_scratch *dom_save;   /* [ORK_MAXDOM]; lazily allocated when multi-domain is first used */
    /* persistent worker pool: spawned once, signalled per matmul (cuts per-matmul create/join) */
    pthread_t pth[ORK_MAXCORE]; struct ork_pw pwa[ORK_MAXCORE]; int pool_n;
    pthread_mutex_t pmu; pthread_cond_t pgo, pdn; void *pjob; int pjob_nc, pgen, pdone, pstop;
    void *(*pjob_fn)(void *); size_t pjob_stride;   /* generalized pool dispatch: per-core worker + arg stride */
    pthread_barrier_t b_ioctl; int mc_submit_rc; int mc_error;
    int last_async_cpu;   /* sched_getcpu() of the most recent async worker at entry (diagnostic/test: -1 = none) */
    /* zero-copy registry: caller-allocated NPU-coherent DMA buffers (ork_dma_alloc). When a matmul's
     * A/C live in one of these, the regcmd points at them directly — no host gather/writeout memcpy. */
    struct buf dma_tab[64]; int dma_n;
    /* global weight arena: a POOL of large DMA chunks (each under the ~4GB single-allocation cap), bump-
     * allocated across ALL packed weights. One weight's tiles always land contiguously in a single chunk
     * (flushed in one bsync_off). Collapses thousands of per-tile bcreates to a handful of chunks => fast
     * warmup, no IOVA-handle OOM. Also the on-disk form for persisted (pre-packed) weights. */
    struct buf wchunk[64]; int wchunk_n; size_t wchunk_off;
    /* PACK DOMAIN: per-ctx default domain for the NEXT pack/load (set by ork_npu_set_pack_domain for the
     * ggml-ork caller). Read once at each pack's entry to stamp w->domain; from there the weight carries
     * its own domain. Not a process-global — concurrent ctxs don't clobber each other. -1 => default. */
    int pack_domain; };
struct ork_w   { int K, N, Sk, Sn, dtype, gsize; struct buf *Bb; struct buf *Bf; int owns; uint8_t *Bi4; size_t Bi4_bytes; uint8_t quant_kind; float *bscale; int domain; struct buf own_buf; int own_buf_valid; };  /* owns=1: per-tile bcreate, reclaimable by ork_mm_free; owns=0: arena views (freed at teardown). own_buf: a single dedicated DMA buffer backing ALL of this weight's tiles as base+offset VIEWS (grouped-i4) — reclaimed as one bdestroy by ork_mm_free (own_buf_valid=1), tiles are non-owning views so they are NOT individually destroyed. Bi4: optional host-side int4-packed (nibble) weight store for pack_i4a8 — the memory-compact form (K*N/2 B) for .orkpack/streaming dump; NPU-side runs int8 (DT_I8). quant_kind: ORK_QK_* — how the nibbles in Bi4 inflate (UNIFORM sign-extend now; CODEBOOK_NF4 LUT reserved). bscale: optional per-output-channel dequant scale (length N) retained alongside Bi4 so the compact int4 form (pack_i4a8 / load_i4a8) can be dumped + reloaded self-contained. domain: this weight's NPU IOMMU domain id (0 = default); its resident tiles live there and its submits run against it — multi-domain residence lets >4 GiB of weights stay resident across domains (the per-domain 32-bit IOVA cap). */
static int check_overlap(const char *name, uintptr_t a_start, uintptr_t a_end, uintptr_t c_start, uintptr_t c_end) {
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
    }
    if (extra && extra_n > 0) {
        for (int i = 0; i < extra_n; i++) {
            if (extra[i].cpu && addr >= extra[i].dma && addr < extra[i].dma + extra[i].size) return 1;
        }
    }
    return 0;
}
static int validate_regcmd(const char *op, ork_npu *c, const uint32_t *rc, int n, const ork_w *w, const struct buf *extra, int extra_n) {
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
static int int8_ks(ork_npu *c){ (void)c;
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
static int budget(ork_npu*c, int M){
    /* M=1 is single-core by default here; the int8 DECODE path in run() overrides to the multi-core budget
     * (it calls budget(c,2)) — splitting N across cores parallelizes the cold per-token weight-DMA, measured
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


static size_t pgup(size_t s){return (s+4095)&~((size_t)4095);}
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
static int ork_dom_default(void){ static int v=-1; if(v<0){const char*e=getenv("ORK_IOMMU_DOMAIN"); v=e?atoi(e):0;} return v; }
/* THREAD-SAFETY: the IOMMU domain is threaded through call parameters and the per-submit
 * rknpu_submit struct, NOT a process-global — two concurrent submits / packs must not race a
 * shared mutable domain. bcreate/bimport take an explicit `domain`; rknpu_submit_ioctl sets
 * sub->iommu_domain_id from a parameter. The pack-path default for the ggml-ork caller lives on
 * the ork_npu ctx (c->pack_domain), read once per pack to stamp w->domain. dom<0 => default. */
static int ork_dom(int dom){ return dom>=0 ? dom : ork_dom_default(); }
static struct buf bcreate(int fd,size_t size,uint32_t flags,int domain){
    struct rknpu_mem_create c; memset(&c,0,sizeof c); c.size=pgup(size); c.flags=flags; c.core_mask=RKNPU_CORE0_MASK; c.iommu_domain_id=ork_dom(domain);
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_CREATE,&c)){perror("CREATE");return (struct buf){0};}
    struct rknpu_mem_map m; memset(&m,0,sizeof m); m.handle=c.handle;
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_MAP,&m)){perror("MAP");return (struct buf){0};}
    void*p=mmap(NULL,c.size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,m.offset);
    if(p==MAP_FAILED){perror("mmap");return (struct buf){0};}
    return (struct buf){c.handle,c.dma_addr,c.obj_addr,p,c.size};
}
static void bdestroy(int fd,struct buf*b){ if(!b->cpu)return; munmap(b->cpu,b->size);
    struct rknpu_mem_destroy d; memset(&d,0,sizeof d); d.handle=b->handle; d.obj_addr=b->obj; ioctl(fd,DRM_IOCTL_RKNPU_MEM_DESTROY,&d);
    if(b->heap_fd>0){ close(b->heap_fd); b->heap_fd=0; } b->cpu=0; }

/* Zero-copy IMPORT (no page alloc, no copy): allocate a dma-buf from /dev/dma_heap/system, mmap it,
 * import it into the NPU's IOMMU domain via PRIME_FD_TO_HANDLE -> MEM_CREATE(handle, flags=0, size=0).
 * The kernel maps the EXISTING dma-buf pages and returns the IOVA (dma_addr) to put in the regcmd.
 * Caller fills *cpu with the (pre-tiled) bytes, then issues a MEM_SYNC clean (bsync TO_DEVICE) before
 * the first submit. Returns a buf whose heap_fd holds the dma-buf fd (closed by bdestroy). On any
 * failure returns {0} (cpu==NULL). g_dmaheap_fd is the cached /dev/dma_heap/system fd (-1 = unopened,
 * -2 = open failed; import then unavailable and callers fall back to bcreate). */
static int g_dmaheap_fd = -1;
static int dmaheap_open(void){
    if(g_dmaheap_fd==-1){
        const char *h=getenv("ORK_DMA_HEAP"); char path[64];
        snprintf(path,sizeof path,"/dev/dma_heap/%s", h&&*h?h:"system");
        g_dmaheap_fd=open(path,O_RDWR|O_CLOEXEC); if(g_dmaheap_fd<0) g_dmaheap_fd=-2;
    }
    return g_dmaheap_fd>=0 ? g_dmaheap_fd : -1;
}
/* dma-buf CPU-access cache sync on the dma-buf fd (flushes CPU caches for an imported cacheable
 * buffer — the rknpu MEM_SYNC does not cover foreign imports). Bracket the CPU fill: START|WRITE
 * before, END|WRITE after. No-op if the buffer wasn't imported (heap_fd<=0). */
static void dmabuf_sync(int heap_fd,uint64_t flags){
    if(heap_fd<=0) return; struct dma_buf_sync s={.flags=flags}; ioctl(heap_fd,DMA_BUF_IOCTL_SYNC,&s);
}
static struct buf bimport(int fd,size_t size,int domain){
    int hf=dmaheap_open(); if(hf<0) return (struct buf){0};
    size_t sz=pgup(size);
    struct dma_heap_allocation_data a; memset(&a,0,sizeof a); a.len=sz; a.fd_flags=O_RDWR|O_CLOEXEC;
    if(ioctl(hf,DMA_HEAP_IOCTL_ALLOC,&a)){ perror("DMA_HEAP_ALLOC"); return (struct buf){0}; }
    int dbuf=(int)a.fd;
    void*p=mmap(NULL,sz,PROT_READ|PROT_WRITE,MAP_SHARED,dbuf,0);
    if(p==MAP_FAILED){ perror("mmap(dmabuf)"); close(dbuf); return (struct buf){0}; }
    struct drm_prime_handle ph; memset(&ph,0,sizeof ph); ph.fd=dbuf; ph.flags=0;
    if(ioctl(fd,DRM_IOCTL_PRIME_FD_TO_HANDLE,&ph)){ perror("PRIME_FD_TO_HANDLE"); munmap(p,sz); close(dbuf); return (struct buf){0}; }
    struct rknpu_mem_create mc; memset(&mc,0,sizeof mc); mc.handle=ph.handle; mc.flags=0; mc.size=0; mc.core_mask=RKNPU_CORE0_MASK; mc.iommu_domain_id=ork_dom(domain);
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_CREATE,&mc)){ perror("MEM_CREATE(import)"); munmap(p,sz); close(dbuf); return (struct buf){0}; }
    struct buf b; memset(&b,0,sizeof b);
    b.handle=mc.handle; b.dma=mc.dma_addr; b.obj=mc.obj_addr; b.cpu=p; b.size=sz; b.heap_fd=dbuf;
    return b;
}
static void bsync(int fd,struct buf*b,uint32_t f){struct rknpu_mem_sync s;memset(&s,0,sizeof s);s.obj_addr=b->obj;s.size=b->size;s.flags=f;ioctl(fd,DRM_IOCTL_RKNPU_MEM_SYNC,&s);}
/* sync a sub-range of a buffer object (for arena views, which share one obj at varying offsets) */
static void bsync_off(int fd,uint64_t obj,uint64_t off,size_t size,uint32_t f){struct rknpu_mem_sync s;memset(&s,0,sizeof s);s.obj_addr=obj;s.offset=off;s.size=size;s.flags=f;ioctl(fd,DRM_IOCTL_RKNPU_MEM_SYNC,&s);}
/* Reserve `need` contiguous bytes of resident weight storage from the arena pool; returns the backing chunk
 * (and sets *base = byte offset within it) or NULL if a fresh chunk can't be allocated (caller then falls
 * back to per-tile bcreate). Chunk size = ORK_WARENA_CHUNK_MB (default 1 GiB), kept under the single-alloc
 * cap; a weight larger than the default chunk gets its own exact-size chunk. */
static struct buf *warena_reserve(ork_npu *c,size_t need,size_t *base){
    size_t a=(need+4095u)&~(size_t)4095u;
    if(c->wchunk_n==0 || c->wchunk_off+a > c->wchunk[c->wchunk_n-1].size){
        if(c->wchunk_n >= (int)(sizeof c->wchunk/sizeof c->wchunk[0])) return NULL;
        const char *e=getenv("ORK_WARENA_CHUNK_MB"); long mb=e?atol(e):1024; if(mb<=0) return NULL;
        size_t csz=(size_t)mb*1024u*1024u; if(a>csz) csz=a;
        struct buf b=bcreate(c->fd,csz,0x403,-1);
        if(!b.cpu){ fprintf(stderr,"[ork] WARNING: weight arena chunk %zuMB alloc failed — falling back to per-buffer\n",csz/1024u/1024u); return NULL; }
        c->wchunk[c->wchunk_n++]=b; c->wchunk_off=0;
    }
    struct buf *ch=&c->wchunk[c->wchunk_n-1];
    *base=c->wchunk_off; c->wchunk_off+=a;
    return ch;
}
static void act(int fd,uint32_t f,uint32_t v){struct rknpu_action a={.flags=f,.value=v};ioctl(fd,DRM_IOCTL_RKNPU_ACTION,&a);}

/* MULTI-DOMAIN SCRATCH SWAP. A submit runs in ONE iommu_domain_id, so the regcmd/task/activation/output
 * scratch a submit references must live in the same domain as the weight. dom_activate parks the current
 * active scratch into dom_save[old] and restores domain `dom`'s parked scratch (zero-initialized on first
 * use, so the run path's lazy bcreate allocates it in `dom` via c->dom_active). No-op when
 * single-domain (dom==dom_active and only domain 0 ever used). */
static void dom_activate(ork_npu *c,int dom){
    if(dom<0||dom>=ORK_MAXDOM) dom=0;
    if(dom==c->dom_active) return;
    if(!c->dom_save){ c->dom_save=calloc(ORK_MAXDOM,sizeof *c->dom_save); if(!c->dom_save){ return; } }
    struct ork_dom_scratch *old=&c->dom_save[c->dom_active], *neo=&c->dom_save[dom];
    /* park active -> old */
    old->used=1; old->regcmd=c->regcmd; old->task=c->task; old->Af=c->Af; old->Cc=c->Cc; old->ccsz=c->ccsz; old->warmed=c->warmed;
    memcpy(old->mrc,c->mrc,sizeof old->mrc); memcpy(old->mtk,c->mtk,sizeof old->mtk); memcpy(old->maf,c->maf,sizeof old->maf);
    memcpy(old->mcc,c->mcc,sizeof old->mcc); old->mtk_all=c->mtk_all; memcpy(old->mccsz,c->mccsz,sizeof old->mccsz);
    memcpy(old->mwarm,c->mwarm,sizeof old->mwarm); old->mc_alloc=c->mc_alloc;
    /* restore neo -> active (zeroed if first use → lazy alloc in this domain) */
    c->regcmd=neo->regcmd; c->task=neo->task; c->Af=neo->Af; c->Cc=neo->Cc; c->ccsz=neo->ccsz; c->warmed=neo->warmed;
    memcpy(c->mrc,neo->mrc,sizeof c->mrc); memcpy(c->mtk,neo->mtk,sizeof c->mtk); memcpy(c->maf,neo->maf,sizeof c->maf);
    memcpy(c->mcc,neo->mcc,sizeof c->mcc); c->mtk_all=neo->mtk_all; memcpy(c->mccsz,neo->mccsz,sizeof c->mccsz);
    memcpy(c->mwarm,neo->mwarm,sizeof c->mwarm); c->mc_alloc=neo->mc_alloc;
    c->dom_active=dom;
    /* first time we touch domain `dom`: it has none of the init-time scratch yet. Mirror ork_npu_init
     * EXACTLY — allocate regcmd+task+Af in THIS domain and seed the task descriptor — so a freshly-activated
     * domain carries every buffer ork_npu_init guarantees, none NULL and none a stale domain-0 IOVA. regcmd
     * and task have NO lazy size-guard (the chain path writes c->regcmd.cpu / c->task.cpu directly), so they
     * MUST be created here. Af is created here too so first-use == init: the run path's `c->Af.size<maxaf`
     * realloc already covers it, but the unguarded RE/probe entrypoints (ork_npu_probe_*) read c->Af.cpu
     * raw, and a non-NULL in-domain Af is the safe, complete invariant rather than relying on each caller.
     * Cc and the per-core mc-* scratch are NOT allocated here: their sizes are matmul-dependent, so every
     * run path lazily (re)allocates them in c->dom_active under their own .size/.cpu guards. */
    if(!neo->used && !c->regcmd.cpu){
        c->regcmd=bcreate(c->fd,2097152,0x403,dom); c->task=bcreate(c->fd,524288,0x40b,dom); c->Af=bcreate(c->fd,(size_t)4*32768*2,0x403,dom);
        if(c->task.cpu){ struct rknpu_task t; memset(&t,0,sizeof t); t.enable_mask=0xd;t.int_mask=0x300;t.int_clear=0x1ffff;t.regcfg_amount=108;t.regcmd_addr=c->regcmd.dma;
            memcpy(c->task.cpu,&t,sizeof t); bsync(c->fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); }
    }
}

static struct ork_npu *g_npu_ctx = NULL;

static void trace_submit(struct rknpu_submit *sub) {
    if (!getenv("ORK_TRACE")) return;
    fprintf(stderr, "[ork-trace] === SUBMIT flags=0x%x timeout=%u task_number=%u core=0x%x ===\n",
            sub->flags, sub->timeout, sub->task_number, sub->core_mask);
    
    if (!g_npu_ctx) return;
    
    void *task_cpu = NULL;
    if (g_npu_ctx->task.obj == sub->task_obj_addr) {
        task_cpu = g_npu_ctx->task.cpu;
    } else if (g_npu_ctx->mtk_all.obj == sub->task_obj_addr) {
        task_cpu = g_npu_ctx->mtk_all.cpu;
    } else {
        for (int i = 0; i < ORK_MAXCORE; i++) {
            if (g_npu_ctx->mtk[i].obj == sub->task_obj_addr) {
                task_cpu = g_npu_ctx->mtk[i].cpu;
                break;
            }
        }
    }
    
    if (!task_cpu) {
        fprintf(stderr, "  (task buffer not found/mapped)\n");
        return;
    }
    
    struct rknpu_task *tasks = (struct rknpu_task *)task_cpu;
    for (uint32_t i = 0; i < sub->task_number; i++) {
        struct rknpu_task *t = &tasks[i];
        fprintf(stderr, "  task[%u]: flags=0x%x op_idx=%u enable=0x%x int_mask=0x%x regcfg_amount=%u regcmd_addr=0x%llx\n",
                i, t->flags, t->op_idx, t->enable_mask, t->int_mask, t->regcfg_amount, (unsigned long long)t->regcmd_addr);
        
        void *regcmd_cpu = NULL;
        uint64_t dma_base = 0;
        if (t->regcmd_addr >= g_npu_ctx->regcmd.dma && t->regcmd_addr < g_npu_ctx->regcmd.dma + g_npu_ctx->regcmd.size) {
            regcmd_cpu = g_npu_ctx->regcmd.cpu;
            dma_base = g_npu_ctx->regcmd.dma;
        } else {
            for (int j = 0; j < ORK_MAXCORE; j++) {
                if (t->regcmd_addr >= g_npu_ctx->mrc[j].dma && t->regcmd_addr < g_npu_ctx->mrc[j].dma + g_npu_ctx->mrc[j].size) {
                    regcmd_cpu = g_npu_ctx->mrc[j].cpu;
                    dma_base = g_npu_ctx->mrc[j].dma;
                    break;
                }
            }
        }
        
        if (regcmd_cpu) {
            uint64_t offset = t->regcmd_addr - dma_base;
            uint32_t *rc = (uint32_t *)((char *)regcmd_cpu + offset);
            int n_words = (int)t->regcfg_amount * 2 + 16;
            fprintf(stderr, "  --- regcmd (%d u32 words) ---\n", n_words);
            for (int k = 0; k < n_words; k += 4) {
                fprintf(stderr, "  [%03d] %08x %08x %08x %08x\n", k,
                        rc[k], (k+1 < n_words)?rc[k+1]:0, (k+2 < n_words)?rc[k+2]:0, (k+3 < n_words)?rc[k+3]:0);
            }
        } else {
            fprintf(stderr, "  (regcmd buffer not found for dma=0x%llx)\n", (unsigned long long)t->regcmd_addr);
        }
    }
}

static int rknpu_submit_ioctl(int fd, struct rknpu_submit *sub, int domain) {
    sub->iommu_domain_id = ork_dom(domain);  /* match the domain the weight's resident tiles live in (threaded per-call, not a global) */
    if (g_ork_prof) { g_prof_submits++; g_prof_submit_progs += sub->task_number; if (sub->task_number > 1) g_prof_submit_chained++; }
    trace_submit(sub);
    int rc = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, sub);
    if (rc < 0) {
        fprintf(stderr, "[ork] WARNING: RKNPU_SUBMIT ioctl failed (rc=%d, errno=%d). Triggering self-healing reset...\n", rc, errno);
        struct rknpu_action a = { .flags = RKNPU_ACT_RESET, .value = 0 };
        ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &a);
    }
    return rc;
}
/* replace ALL matching regcmd entries — the template repeats some regs (e.g. 0x1040) and
 * the NPU uses a later copy, so a first-match-only patch leaves stale values. */
static void setr(uint32_t*rc,int n,uint32_t b,uint32_t o,uint32_t v){for(int k=0;k+1<n;k+=2)if((rc[k]&0xffff)==o&&(rc[k+1]>>16)==b){rc[k]=(o)|((v&0xffff)<<16);rc[k+1]=(b<<16)|((v>>16)&0xffff);}}
/* sched=1: single-submit internal M-scheduler (clean power-of-2 Kp); sched=0: one M-tile. */
static void synth(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC,int sched,int cbuf){
    memcpy(rc,REGCMD,REGCMD_N*4);
    setr(rc,REGCMD_N,0x201,0x1024,((K-1)<<16)|K);setr(rc,REGCMD_N,0x201,0x1030,K*N*2);setr(rc,REGCMD_N,0x201,0x1034,K*2);
    setr(rc,REGCMD_N,0x201,0x1044,K/32);setr(rc,REGCMD_N,0x201,0x1088,K);setr(rc,REGCMD_N,0x201,0x107c,K/8);
    setr(rc,REGCMD_N,0x201,0x1020,0x10000|mc);setr(rc,REGCMD_N,0x201,0x1084,0x10000|mc);setr(rc,REGCMD_N,0x201,0x102c,mc);
    setr(rc,REGCMD_N,0x1001,0x4034,mc-1);setr(rc,REGCMD_N,0x1001,0x405c,(mc-1)<<16);setr(rc,REGCMD_N,0x801,0x3014,(mc-1)<<16);
    setr(rc,REGCMD_N,0x1001,0x403c,((N-1)<<16)|(N-1));setr(rc,REGCMD_N,0x1001,0x4058,N-1);setr(rc,REGCMD_N,0x1001,0x4038,(((N/4)-1)<<16)|((N/4)-1));
    setr(rc,REGCMD_N,0x201,0x1038,0x1010000|N);setr(rc,REGCMD_N,0x801,0x3018,N-1);
    if(sched){
        int R=cbuf/K; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; } int rows=(mc+1<R)?(mc+1):R; setr(rc,REGCMD_N,0x201,0x1010,16*rows);
        double scale=(double)K/256.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale),mg=(mc+63)/64; if(mg<1)mg=1;  /* ceil: see synth_i8 (65..127-row tile needs the next K-schedule) */
        int v=base-slope*(mg-1); if(v<0x1b)v=0x1b; setr(rc,REGCMD_N,0x201,0x1040,v);
    } else { setr(rc,REGCMD_N,0x201,0x1010,16*(mc+1)); }
    setr(rc,REGCMD_N,0x201,0x1070,aA);setr(rc,REGCMD_N,0x201,0x1110,aB);setr(rc,REGCMD_N,0x1001,0x4020,aC);
}
/* int8/w8a8: A,B int8 -> C int32. Differs from fp16: weight amount/stride (no x2), K-passes
 * ceil(K/64), 0x107c=K/16, rows-budget 2x (int8 packs 2x rows/CBUF), and the 0x1040 schedule
 * uses effective K = K/2. cbuf is the fp16 budget; int8 rows = 2*cbuf/K. */
static void synth_i8(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC,int sched,int cbuf,int stride){
    memcpy(rc,REGCMD_I8,REGCMD_I8_N*4);
    setr(rc,REGCMD_I8_N,0x201,0x1024,((K-1)<<16)|K);setr(rc,REGCMD_I8_N,0x201,0x1030,K*N);setr(rc,REGCMD_I8_N,0x201,0x1034,K);
    setr(rc,REGCMD_I8_N,0x201,0x1044,(K+63)/64);setr(rc,REGCMD_I8_N,0x201,0x1088,K);setr(rc,REGCMD_I8_N,0x201,0x107c,K/16);
    setr(rc,REGCMD_I8_N,0x201,0x1020,0x10000|mc);setr(rc,REGCMD_I8_N,0x201,0x1084,0x10000|mc);setr(rc,REGCMD_I8_N,0x201,0x102c,mc);
    setr(rc,REGCMD_I8_N,0x1001,0x4034,mc-1);setr(rc,REGCMD_I8_N,0x1001,0x405c,(mc-1)<<16);setr(rc,REGCMD_I8_N,0x801,0x3014,(mc-1)<<16);
    int s=stride>0?stride:N;
    setr(rc,REGCMD_I8_N,0x1001,0x403c,((s-1)<<16)|(N-1));setr(rc,REGCMD_I8_N,0x1001,0x4058,N-1);setr(rc,REGCMD_I8_N,0x1001,0x4038,(((s/4)-1)<<16)|((N/4)-1));
    setr(rc,REGCMD_I8_N,0x201,0x1038,0x1010000|N);setr(rc,REGCMD_I8_N,0x801,0x3018,N-1);
    if(sched){
        int R=(2*cbuf)/K; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
        /* DEBUG knob (ORK_RCAP, default off): override the 0x1010 row-count hint. MEASURED NEUTRAL —
         * 0x1010 is only a hint; it does NOT change correctness or speed (the real single-core lever is
         * the M-tile size mg_max*64, set by the caller — see AGENTS.md "weight-DMA amortization"). Kept
         * only for RE/diagnostics. */
        static int rcap=-2; if(rcap==-2){const char*e=getenv("ORK_RCAP"); rcap=e?atoi(e):-1;}
        if(rcap>0) R=rcap;
        int rows=(mc+1<R)?(mc+1):R; setr(rc,REGCMD_I8_N,0x201,0x1010,16*rows);
        /* 0x1040 = K-reduction schedule, selected per 64-row group. MUST be ceil(mc/64): a tile of
         * 65..127 rows spills past the first 64-row group, so it needs the NEXT schedule (the one a
         * full 128-row tile uses), not the <=64 schedule. floor(mc/64) gave 65..127-row tiles the
         * <=64 schedule -> rows 64..mc-1 computed against the wrong K-partition (the prefill bug). */
        double scale=(double)K/512.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale),mg=(mc+63)/64; if(mg<1)mg=1;
        int v=base-slope*(mg-1); if(v<0x1b)v=0x1b;
        static int r1040=-2; if(r1040==-2){const char*e=getenv("ORK_R1040"); r1040=e?atoi(e):-1;}
        if(r1040>0) v=r1040;   /* EXPERIMENTAL: override the 0x1040 schedule (rknn's captured 0x75 for the 30-row tile) */
        setr(rc,REGCMD_I8_N,0x201,0x1040,v);
    } else { setr(rc,REGCMD_I8_N,0x201,0x1010,16*(mc+1)); }
    setr(rc,REGCMD_I8_N,0x201,0x1070,aA);setr(rc,REGCMD_I8_N,0x201,0x1110,aB);setr(rc,REGCMD_I8_N,0x1001,0x4020,aC);
}

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
static void set_i8_out8(uint32_t*rc,int N,int stride,int mult,int shift){
    int s=stride>0?stride:N;
    setr(rc,REGCMD_I8_N,0x1001,0x4010,0);                                   /* clear the 0x8000 int32-output bit -> int8 out */
    setr(rc,REGCMD_I8_N,0x1001,0x4038,(((s/16)-1)<<16)|((N/16)-1));         /* output group stride: int8 packs 4x denser than int32 (N/16 vs N/4) */
    setr(rc,REGCMD_I8_N,0x1001,0x4050,0x0124);                             /* int8 output row byte-stride config (const; int32=0x07fc) */
    setr(rc,REGCMD_I8_N,0x1001,0x40c0,0x0020);                             /* output element size = 1 byte (int32=0x0080) */
    setr(rc,REGCMD_I8_N,0x1001,0x4084,mult);                              /* requant multiplier */
    setr(rc,REGCMD_I8_N,0x1001,0x4088,shift);                             /* requant shift (>>) */
}

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
static void set_i8_silu(uint32_t*rc,int N,int stride,int r_mult,int r_shift,
                        uint32_t out_bias,uint32_t idx_off,uint32_t cfg4068){
    set_i8_out8(rc,N,stride,r_mult,r_shift);       /* int8-output byte layout + the unified scale R (0x4084/0x4088) */
    setr(rc,REGCMD_I8_N,0x1001,0x4004,0x0030); setr(rc,REGCMD_I8_N,0x2001,0x5004,0x0030); /* activation mode on */
    setr(rc,REGCMD_I8_N,0x1001,0x4010,0x44e0);     /* LUT/activation enable (output-stage high byte 0x44) */
    setr(rc,REGCMD_I8_N,0x1001,0x4060,0x00020040); /* activation mode bit 0x0002 (fixed) */
    setr(rc,REGCMD_I8_N,0x1001,0x4068,cfg4068);    /* per-scale field, no observed output effect (replay) */
    setr(rc,REGCMD_I8_N,0x1001,0x4070,0x00000302); /* fixed */
    setr(rc,REGCMD_I8_N,0x1001,0x4080,out_bias);   /* output bias / asymmetric zero-point (silu(0) -> out_bias) */
    setr(rc,REGCMD_I8_N,0x1001,0x4108,0x00000068); /* LUT config (fixed) */
    setr(rc,REGCMD_I8_N,0x1001,0x410c,0x00050500); /* LUT config (fixed) */
    setr(rc,REGCMD_I8_N,0x1001,0x4110,idx_off);    /* index offset -> C0 (silu-zero index ~512) */
    setr(rc,REGCMD_I8_N,0x1001,0x411c,0x00004000); /* fixed config (scale-independent) */
    setr(rc,REGCMD_I8_N,0x1001,0x4128,0x40320000); /* fixed config */
    setr(rc,REGCMD_I8_N,0x1001,0x412c,0x000001a0); /* fixed config */
}

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
#define REGCMD_I8_EW_N (REGCMD_I8_N + REGCMD_EW_LANE_N)   /* 224 + 36 = 260 words = 126 reg entries + trailer */

/* EW-mul RE submit timeout (ms). ORK_EW_TIMEOUT lets the wedge-search fail fast (~1-2s) instead of 60s —
 * a working op completes in ~100us, so a short guard is safe; the kernel soft-resets on timeout either way. */
static unsigned ew_timeout_ms(void){ static int t=-1; if(t<0){const char*e=getenv("ORK_EW_TIMEOUT"); t=e?atoi(e):60000;} return (unsigned)(t>0?t:60000); }

/* Generalize the standalone element-wise MUL op (REGCMD_MUL{,_F16,_I16}) from the captured M=8,N=64 geometry
 * to arbitrary (M tokens = RDMA width, N channels). Derived by capturing the op at N=128 and M=16 and diffing:
 * M-dependent regs = M-1 (0x500c/0x4030/0x405c) and M*16 (strides 0x5040/0x4024/0x40c0 = EW_SURF_STRIDE /
 * output stride / SURFACE_ADD); N-dependent = N-1 (0x5014/0x4058) and (N-1)|(N-1)<<16 (0x403c). The channel
 * atom (16 for int8, 8 for the 2-byte fp16/int16) sets the cube; both give surf_stride = M*16 bytes. */
static void set_mul_geom(uint32_t *rc,int n,int M,int N){
    uint32_t sstride=(uint32_t)(M*16);
    setr(rc,n,0x2001,0x500c,(uint32_t)(M-1));          /* RDMA_DATA_CUBE_WIDTH  = M-1 */
    setr(rc,n,0x2001,0x5010,0);                        /* RDMA_DATA_CUBE_HEIGHT = 0 (H=1) */
    setr(rc,n,0x2001,0x5014,(uint32_t)(N-1));          /* RDMA_DATA_CUBE_CHANNEL= N-1 */
    setr(rc,n,0x2001,0x5040,sstride);                  /* RDMA_EW_SURF_STRIDE = M*16 */
    setr(rc,n,0x1001,0x4024,sstride);                  /* output surface stride */
    setr(rc,n,0x1001,0x4030,(uint32_t)(M-1));
    setr(rc,n,0x1001,0x403c,(uint32_t)(((N-1)<<16)|(N-1)));
    setr(rc,n,0x1001,0x4058,(uint32_t)(N-1));
    setr(rc,n,0x1001,0x405c,(uint32_t)(M-1));
    setr(rc,n,0x1001,0x40c0,sstride);                  /* SURFACE_ADD = M*16 */
}

/* Apply ork's synth_i8 matmul GEOMETRY (same formulas as synth_i8, sched=1) onto an arbitrary regcmd `rc`
 * of length `n`. Used to inject ork's geometry into RKNN's EW-mul TEMPLATE (REGCMD_EWMUL_LIN) — which keeps
 * RKNN's register ORDER + EW output-stage/lane (so it executes) while making the conv engine read ork's own
 * [Nt][Kt][32][32] A/B tile layout (so acc is correct). Addresses (0x1070/0x1110/0x4020) patched by caller. */
static void apply_ork_geom(uint32_t*rc,int n,int mc,int K,int N,int cbuf){
    setr(rc,n,0x201,0x1024,((K-1)<<16)|K);setr(rc,n,0x201,0x1030,K*N);setr(rc,n,0x201,0x1034,K);
    setr(rc,n,0x201,0x1044,(K+63)/64);setr(rc,n,0x201,0x1088,K);setr(rc,n,0x201,0x107c,K/16);
    setr(rc,n,0x201,0x1020,0x10000|mc);setr(rc,n,0x201,0x1084,0x10000|mc);setr(rc,n,0x201,0x102c,mc);
    setr(rc,n,0x1001,0x4034,mc-1);setr(rc,n,0x1001,0x405c,(mc-1)<<16);setr(rc,n,0x801,0x3014,(mc-1)<<16);
    setr(rc,n,0x1001,0x403c,((N-1)<<16)|(N-1));setr(rc,n,0x1001,0x4058,N-1);setr(rc,n,0x1001,0x4038,(((N/4)-1)<<16)|((N/4)-1));
    setr(rc,n,0x201,0x1038,0x1010000|N);setr(rc,n,0x801,0x3018,N-1);
    int R=(2*cbuf)/K; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
    int rows=(mc+1<R)?(mc+1):R; setr(rc,n,0x201,0x1010,16*rows);
    double scale=(double)K/512.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale),mg=(mc+63)/64; if(mg<1)mg=1;
    int v=base-slope*(mg-1); if(v<0x1b)v=0x1b; setr(rc,n,0x201,0x1040,v);
}

/* Splice the 0x50xx second-DPU lane into a synth_i8'd matmul regcmd. base[] is a full REGCMD_I8_N buffer
 * already filled by synth_i8 (108 reg entries in words 0..215, then the 8-word trailer). Output rc[] gets:
 * [108 reg entries] [REGCMD_EW_LANE 18 entries] [8-word trailer]. */
static void splice_ew_lane(uint32_t*rc,const uint32_t*base){
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
static void set_i8_ewmul(uint32_t*rc,int M,int N,int stride,int mult,int shift,uint32_t aG){
    set_i8_out8(rc,N,stride,mult,shift);                         /* ork conv + int8-out; OUT_CVT gain=0x4084/88 */
    int s=stride>0?stride:N;
    if(getenv("ORK_EW_REGOP")){
        /* DECISIVE ISOLATION: EW multiply with a REGISTER-constant operand (EW_OP_SRC=0, bit6=0), NO RDMA.
         * out = up_acc * EW_OP_VALUE_0. If this works, the EW stage is fine on ork geometry & the RDMA is the
         * wedge; if it wedges, the EW stage itself is incompatible with ork's dense conv geometry. */
        setr(rc,REGCMD_I8_N,0x1001,0x4070,0x90400284);          /* EW_CFG mul, OP_SRC=0 (register operand) */
        setr(rc,REGCMD_I8_N,0x1001,0x4090,(uint32_t)strtoul(getenv("ORK_EW_REGOP"),0,0)); /* EW_OP_VALUE_0 */
        setr(rc,REGCMD_I8_N,0x1001,0x4050,0x00000125);
        setr(rc,REGCMD_I8_N,0x1001,0x4010,0x000000e0);
    } else if(!getenv("ORK_EW_NOMUL")){
        /* EW_CFG: multiply(bit2), operand from RDMA(bit6), operand-CVT active(bit8=0), LUT+ReLU bypass(7,9).
         * ORK_EW_CFG / ORK_EW_ERDMA override EW_CFG / ERDMA_CFG for int8-vs-int16 data-size tuning. */
        uint32_t ewcfg=0x904002c4; { const char*e=getenv("ORK_EW_CFG"); if(e) ewcfg=(uint32_t)strtoul(e,0,0); }
        setr(rc,REGCMD_I8_N,0x1001,0x4070,ewcfg);
        setr(rc,REGCMD_I8_N,0x1001,0x4074,0x00000000);          /* EW_CVT_OFFSET_VALUE = 0 (silu zero-point 0) */
        setr(rc,REGCMD_I8_N,0x1001,0x4078,0x00000001);          /* EW_CVT_SCALE=1, SHIFT=0 (unity operand cvt) */
        /* output-stage EW-active bits (required so the DPU expects the element-wise/RDMA stage; mode bits,
         * geometry-independent). ORK_EW_NO50=skip individual ones during bisection. */
        setr(rc,REGCMD_I8_N,0x1001,0x4050,0x00000125);          /* out row cfg + EW-enable bit0 (out8=0x124) */
        setr(rc,REGCMD_I8_N,0x1001,0x4010,0x000000e0);          /* DATA_FORMAT EW bits (out8=0) */
        { const char*eo=getenv("ORK_EW_COFF"); if(eo) setr(rc,REGCMD_I8_N,0x1001,0x4074,(uint32_t)strtoul(eo,0,0)); }
        { const char*es=getenv("ORK_EW_CSCL"); if(es) setr(rc,REGCMD_I8_N,0x1001,0x4078,(uint32_t)strtoul(es,0,0)); }
        /* DPU_RDMA (0x50xx): fetch silu(gate) as the element-wise operand at EW_BASE (0x5038).
         * ORK_EW_SPTR overrides RDMA_S_POINTER (0x5004): 0xe = PP mode (producer/consumer ping-pong, needs a
         * partner to advance the pointer — deadlocks in a standalone shot); try 0/1 for single-shot no-PP. */
        { uint32_t sp=0x0000000e; const char*e=getenv("ORK_EW_SPTR"); if(e) sp=(uint32_t)strtoul(e,0,0);
          setr(rc,REGCMD_I8_EW_N,0x2001,0x5004,sp); }
        setr(rc,REGCMD_I8_EW_N,0x2001,0x5008,0x00000001);       /* RDMA_OPERATION_ENABLE (only in EW-memory path) */
        setr(rc,REGCMD_I8_EW_N,0x2001,0x500c,M-1);              /* RDMA_DATA_CUBE_WIDTH  = M-1 */
        setr(rc,REGCMD_I8_EW_N,0x2001,0x5010,0x00000000);       /* RDMA_DATA_CUBE_HEIGHT = 0 (H=1) */
        setr(rc,REGCMD_I8_EW_N,0x2001,0x5014,N-1);              /* RDMA_DATA_CUBE_CHANNEL= N-1 */
        uint32_t erdma=0x40000004; { const char*e=getenv("ORK_EW_ERDMA"); if(e) erdma=(uint32_t)strtoul(e,0,0); }
        setr(rc,REGCMD_I8_EW_N,0x2001,0x5034,erdma);            /* RDMA_ERDMA_CFG: enable(bit0=0)+data_size/mode */
        setr(rc,REGCMD_I8_EW_N,0x2001,0x5038,aG);               /* RDMA_EW_BASE_ADDR = silu(gate) */
        setr(rc,REGCMD_I8_EW_N,0x2001,0x5018,aG);               /* RDMA_SRC_BASE_ADDR (valid) */
        /* rocket rkt_regcmd.c element-wise: the operand is read via the BRDMA channel from SRC_BASE (0x5018),
         * with BRDMA_DATA_USE=1 (0x501c bits1-4 => value 0x2). (I earlier wrongly DISABLED BRDMA -> the RDMA
         * never delivered the operand -> DPU hung.) NRDMA off; BS_BASE valid (not the stale RKNN addr). */
        setr(rc,REGCMD_I8_EW_N,0x2001,0x501c,0x00000002);       /* RDMA_BRDMA_CFG: BRDMA_DATA_USE=1 (on) */
        setr(rc,REGCMD_I8_EW_N,0x2001,0x5028,0x00000000);       /* RDMA_NRDMA_CFG: NRDMA_DATA_USE=0 (off) */
        setr(rc,REGCMD_I8_EW_N,0x2001,0x5020,aG);               /* RDMA_BS_BASE_ADDR: valid (not stale RKNN) */
        setr(rc,REGCMD_I8_EW_N,0x2001,0x5040,s);                /* RDMA_EW_SURF_STRIDE (dense = N) */
        setr(rc,REGCMD_I8_EW_N,0x2001,0x504c,s);
        setr(rc,REGCMD_I8_EW_N,0x2001,0x506c,s);
        /* 0x5044 (FEATURE_MODE_CFG int8) + 0x5068 (RDMA_WEIGHT=0x01010101) come from REGCMD_EW_LANE as-is. */
        /* ORK_EW_STRIDE overrides EW_SURF_STRIDE (cube atom-16 layout probe: M*16) */
        { const char*e=getenv("ORK_EW_STRIDE"); if(e){ uint32_t v=(uint32_t)strtoul(e,0,0);
            setr(rc,REGCMD_I8_EW_N,0x2001,0x5040,v); setr(rc,REGCMD_I8_EW_N,0x2001,0x506c,v); setr(rc,REGCMD_I8_EW_N,0x2001,0x504c,v); } }
        /* PC_OPERATION_ENABLE (regcmd trailer, reg 0x8 lane 0x81): ork's REGCMD_I8 trailer has 0x0d
         * (CNA|CORE|DPU) — the DPU_RDMA block (bit 0x10) is NOT enabled, so the RDMA never runs and the DPU
         * waits forever. RKNN's EW op has 0x1d here. Enable the RDMA block in the regcmd's own op-enable. */
        setr(rc,REGCMD_I8_EW_N,0x0081,0x0008,0x0000001d);
    }
    /* ORK_EW_BIAS overrides 0x4080 (output offset/bias) */
    { const char*e=getenv("ORK_EW_BIAS"); if(e) setr(rc,REGCMD_I8_N,0x1001,0x4080,(uint32_t)strtoul(e,0,0)); }
}

/* W4A4 (int4 A x int4 B -> int16 C) — uses the CAPTURED librknnrt regcmd verbatim (REGCMD_I4) as
 * the base (the real hardware program, not a guess), overriding only the K/N/address-dependent regs.
 * The precision regs (0x100c=0x360, 0x1080, 0x3010=0x601, 0x4010) stay as captured; K, N (≤nmax),
 * and the A/B/C addresses are parameterized. The captured program is M=1 (each task of the closed
 * runtime's M-tiling), so callers M-tile by looping rows. See ROADMAP. */
static void synth_i4(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC){
    memcpy(rc,REGCMD_I4,REGCMD_I4_N*4);
    setr(rc,REGCMD_I4_N,0x201,0x1024,((K-1)<<16)|K);       /* K range (element count) */
    setr(rc,REGCMD_I4_N,0x201,0x1030,(K*N)/2);             /* weight bytes: int4 = 0.5 B/elem */
    setr(rc,REGCMD_I4_N,0x201,0x1034,K/2);                 /* weight row bytes */
    setr(rc,REGCMD_I4_N,0x201,0x1044,(K+127)/128);        /* K-passes: ceil(K/128) (captured scaling) */
    setr(rc,REGCMD_I4_N,0x201,0x1088,K);
    setr(rc,REGCMD_I4_N,0x201,0x1038,0x1010000|N);setr(rc,REGCMD_I4_N,0x801,0x3018,N-1);
    /* N-output-stride regs, parameterized for wide-N single-submit (verified vs N=64 & N=128
     * captures: 0x403c=(N-1)dup, 0x4058=N-1, 0x3018=N-1 above). 0x40c0/0x4050 are CONSTANT across N
     * (0x80/0x7fe — left at REGCMD_I4). */
    setr(rc,REGCMD_I4_N,0x1001,0x403c,((N-1)<<16)|(N-1));
    setr(rc,REGCMD_I4_N,0x1001,0x4058,N-1);
    /* Multi-M scheduler (mc>1) — PROVEN INERT for int4 (kept runnable for re-test on future
     * kernels/SoCs; see tools/i4_multim_probe.c + ROADMAP Tier 4b). Applying int8's full multi-M
     * register set (M-count = mc, output M-stride = mc-1, CNA row-count 0x1010, schedule 0x1040,
     * 0x4038 = (N/4-1)) to the int4 program does NOT make it compute rows >0: with per-row A, row 0 is
     * bit-exact and rows 1..M-1 are computed NOWHERE in the output. The int4 datapath is structurally
     * single-row on this NPU (the closed runtime tiled int4 M across 12 tasks, and task_number>1
     * kernel-hangs this board). So int4 is 1-submit/row; this block stays for the record but is unused
     * by production (all callers pass mc=1). */
    if(mc>1){
        int mc_phys = 2 * mc;
        setr(rc,REGCMD_I4_N,0x201,0x1020,0x10000|mc_phys);setr(rc,REGCMD_I4_N,0x201,0x1084,0x10000|mc_phys);setr(rc,REGCMD_I4_N,0x201,0x102c,mc_phys);
        setr(rc,REGCMD_I4_N,0x1001,0x4034,mc_phys-1);setr(rc,REGCMD_I4_N,0x1001,0x405c,0);setr(rc,REGCMD_I4_N,0x801,0x3014,(mc_phys-1)<<16);
        setr(rc,REGCMD_I4_N,0x1001,0x4038,(((N/4)-1)<<16)|((N/4)-1));
        setr(rc,REGCMD_I4_N,0x201,0x1010,16*(mc_phys+1));
        double scale=(double)K/256.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale),mg=mc_phys/64; if(mg<1)mg=1;
        int v=base-slope*(mg-1); if(v<0x1b)v=0x1b; setr(rc,REGCMD_I4_N,0x201,0x1040,v);
    }
    setr(rc,REGCMD_I4_N,0x201,0x1070,aA);setr(rc,REGCMD_I4_N,0x201,0x1110,aB);setr(rc,REGCMD_I4_N,0x1001,0x4020,aC);
}

/* Read-only sanity check: the benchmark methodology requires the DDR (dmc) governor at 'performance'
 * — a parked governor ~halves decode. We only WARN (never write; that needs root), so any caller
 * (llama-bench, the examples) notices a misconfigured box. Silence with ORK_NO_GOV_WARN=1. */
static void warn_if_governor_parked(void){
    if(getenv("ORK_NO_GOV_WARN")) return;
    FILE*f=fopen("/sys/class/devfreq/dmc/governor","r"); if(!f) return;
    char g[64]={0};
    if(fgets(g,sizeof g,f)){ g[strcspn(g,"\n")]=0;
        if(strcmp(g,"performance")!=0)
            fprintf(stderr,"[ork] WARNING: DDR (dmc) governor is '%s', not 'performance' — decode may be ~half speed. "
                           "Pin it: echo performance | sudo tee /sys/class/devfreq/dmc/governor  (ORK_NO_GOV_WARN=1 to silence)\n",g);
    }
    fclose(f);
}

ork_npu *ork_npu_init(void){
    const struct ork_soc *soc=ork_soc_detect();
    if(!soc){fprintf(stderr,"[ork] ERROR: unknown SoC (no device-tree match) — cannot select NPU params\n");return NULL;}
    if(!soc->validated) fprintf(stderr,"[ork] WARNING: %s params are inherited/untested — validate with the regression suite\n",soc->id);
    warn_if_governor_parked();
    g_ork_prof = getenv("ORK_PROFILE") ? 1 : 0;
    const char*card=getenv("ORK_NPU_CARD"); if(!card)card=soc->card;
    int fd=open(card,O_RDWR); if(fd<0){perror("open NPU card");return NULL;}
    act(fd,RKNPU_GET_DRV_VERSION,0);act(fd,RKNPU_POWER_ON,0);act(fd,RKNPU_SET_PROC_NICE,(uint32_t)-19);
    ork_npu *c=calloc(1,sizeof *c); c->fd=fd; c->soc=soc; c->last_dt=-1; c->core_budget=soc->cores; c->pack_domain=-1; c->last_async_cpu=-1;
    pthread_mutex_init(&c->pmu,NULL); pthread_cond_init(&c->pgo,NULL); pthread_cond_init(&c->pdn,NULL);
    c->regcmd=bcreate(fd,2097152,0x403,-1); c->task=bcreate(fd,524288,0x40b,-1); c->Af=bcreate(fd,(size_t)4*32768*2,0x403,-1);
    struct rknpu_task t; memset(&t,0,sizeof t); t.enable_mask=0xd;t.int_mask=0x300;t.int_clear=0x1ffff;t.regcfg_amount=108;t.regcmd_addr=c->regcmd.dma;
    memcpy(c->task.cpu,&t,sizeof t); bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    if(!c->regcmd.cpu||!c->task.cpu||!c->Af.cpu){ork_npu_free(c);return NULL;}
    g_npu_ctx = c;
    return c;
}
void ork_npu_free(ork_npu *c){ if(!c)return; int fd=c->fd;
    if(g_ork_prof){
        if(g_prof_i8_calls) fprintf(stderr,"[ork PROFILE] run_i8: %ld calls, %.1f ms total, %.0f us/call\n",
                                    g_prof_i8_calls, g_prof_i8_us/1e3, g_prof_i8_us/g_prof_i8_calls);
        if(g_prof_i4_calls) fprintf(stderr,"[ork PROFILE] run_i4: %ld calls, %.1f ms total, %.0f us/call\n",
                                    g_prof_i4_calls, g_prof_i4_us/1e3, g_prof_i4_us/g_prof_i4_calls);
        if(g_prof_submits) fprintf(stderr,"[ork PROFILE] submits: %ld ioctls, %ld programs (%.2f prog/ioctl), %ld chained(>1prog). per-i8-call: %.2f ioctls\n",
                                   g_prof_submits, g_prof_submit_progs, (double)g_prof_submit_progs/g_prof_submits, g_prof_submit_chained,
                                   g_prof_i8_calls?(double)g_prof_submits/g_prof_i8_calls:0.0);
    }
    if (g_npu_ctx == c) g_npu_ctx = NULL;
    if(c->pool_n){ pthread_mutex_lock(&c->pmu); c->pstop=1; pthread_cond_broadcast(&c->pgo); pthread_mutex_unlock(&c->pmu);
        for(int i=1;i<c->pool_n;i++) pthread_join(c->pth[i],NULL); }
    bdestroy(fd,&c->regcmd);bdestroy(fd,&c->task);bdestroy(fd,&c->Af);bdestroy(fd,&c->Cc);bdestroy(fd,&c->mtk_all);
    for(int i=0;i<ORK_MAXCORE;i++){bdestroy(fd,&c->mrc[i]);bdestroy(fd,&c->mtk[i]);bdestroy(fd,&c->maf[i]);bdestroy(fd,&c->mcc[i]);}
    /* free PARKED per-domain scratch (the active set above is whichever domain was last run) */
    if(c->dom_save){ for(int d=0;d<ORK_MAXDOM;d++){ if(d==c->dom_active||!c->dom_save[d].used) continue;
        struct ork_dom_scratch *s=&c->dom_save[d];
        bdestroy(fd,&s->regcmd);bdestroy(fd,&s->task);bdestroy(fd,&s->Af);bdestroy(fd,&s->Cc);bdestroy(fd,&s->mtk_all);
        for(int i=0;i<ORK_MAXCORE;i++){bdestroy(fd,&s->mrc[i]);bdestroy(fd,&s->mtk[i]);bdestroy(fd,&s->maf[i]);bdestroy(fd,&s->mcc[i]);} }
        free(c->dom_save); }
    for(int i=0;i<c->dma_n;i++) bdestroy(fd,&c->dma_tab[i]);
    free(c->cres); if(fd>=0)close(fd); free(c); }

/* ---- zero-copy DMA buffers (NPU-coherent, CPU-mapped). A matmul whose A and/or C live in one of
 * these has the regcmd point at it directly — no host gather/writeout memcpy. ork_mm_run_i8 detects
 * residency automatically (no API change); the caller just allocates A/C here. ---- */
void *ork_dma_alloc(ork_npu *c, size_t size){
    if(!c || c->dma_n >= (int)(sizeof c->dma_tab/sizeof c->dma_tab[0])) return NULL;
    struct buf b=bcreate(c->fd,size,0x401,c->pack_domain); if(!b.cpu) return NULL;
    c->dma_tab[c->dma_n++]=b; return b.cpu;
}
void ork_dma_free(ork_npu *c, void *ptr){
    if(!c||!ptr) return;
    for(int i=0;i<c->dma_n;i++) if(c->dma_tab[i].cpu==ptr){ bdestroy(c->fd,&c->dma_tab[i]); c->dma_tab[i]=c->dma_tab[--c->dma_n]; memset(&c->dma_tab[c->dma_n], 0, sizeof(struct buf)); return; }
}
/* Zero-copy IMPORT (no alloc, no copy) — see header. Registered in dma_tab like ork_dma_alloc so
 * ork_mm_run zero-copy detection + dma_find work; freed by ork_dma_import_free (or ork_dma_free). */
void *ork_dma_import(ork_npu *c, size_t size){
    if(!c || c->dma_n >= (int)(sizeof c->dma_tab/sizeof c->dma_tab[0])) return NULL;
    struct buf b=bimport(c->fd,size,c->pack_domain); if(!b.cpu) return NULL;
    c->dma_tab[c->dma_n++]=b; return b.cpu;
}
void ork_dma_import_free(ork_npu *c, void *ptr){ ork_dma_free(c,ptr); }
/* the registered DMA buffer containing host ptr p, or NULL if p isn't zero-copy-resident */
static struct buf *dma_find(ork_npu *c, const void *p){
    for(int i=0;i<c->dma_n;i++){ char*base=c->dma_tab[i].cpu;
        if((const char*)p>=base && (const char*)p<base+c->dma_tab[i].size) return &c->dma_tab[i]; }
    return NULL;
}
/* Clean CPU writes -> device for an imported (or ork_dma_alloc) buffer; the bsync the weight fill
 * issues once before the first submit (write-once-read-many weights). size 0 = whole buffer. */
void ork_dma_import_sync(ork_npu *c, void *ptr, size_t size){
    (void)size; struct buf *b=dma_find(c,ptr); if(!b) return;
    /* For imported dma-bufs the dma-buf's own SYNC ioctl flushes the CPU caches (the rknpu MEM_SYNC
     * does not cover foreign imports). END|WRITE = "CPU done writing" -> clean to device. */
    dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);
}
/* Diagnostic only (tools/disk_stream_bench.c): flush `size` bytes of an ork_dma_alloc buffer to the
 * device after a host write (the bsync the streaming fill would issue). Not in the public header. */
void ork_dma_bsync_to_device(ork_npu *c, void *ptr, size_t size){
    struct buf *b=dma_find(c,ptr); if(!b) return;
    struct rknpu_mem_sync s; memset(&s,0,sizeof s);
    s.obj_addr=b->obj; s.offset=(uint64_t)((char*)ptr-(char*)b->cpu); s.size=size?size:b->size;
    s.flags=RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE; ioctl(c->fd,DRM_IOCTL_RKNPU_MEM_SYNC,&s);
    s.flags=RKNPU_MEM_SYNC_TO_DEVICE; ioctl(c->fd,DRM_IOCTL_RKNPU_MEM_SYNC,&s);
}
/* Diagnostic only (tools/dmabuf_fill_probe.c): allocate a registered DMA buffer with a caller-chosen
 * rknpu mem-create flag set, so the probe can A/B the write-combine (0x401) vs cacheable (0x403) fill
 * bandwidth + NPU-read correctness WITHOUT changing the default ork_dma_alloc behavior. Additive; not
 * in the public header. The buffer is registered in dma_tab so ork_mm_run_i8 zero-copy + dma_find work. */
void *ork_dma_alloc_flags(ork_npu *c, size_t size, unsigned flags){
    if(!c || c->dma_n >= (int)(sizeof c->dma_tab/sizeof c->dma_tab[0])) return NULL;
    struct buf b=bcreate(c->fd,size,flags,c->pack_domain); if(!b.cpu) return NULL;
    c->dma_tab[c->dma_n++]=b; return b.cpu;
}
/* Diagnostic only: clean-only flush (TO_DEVICE) of a sub-range — push dirty CPU cache lines out to DRAM
 * so the NPU reads correct data, WITHOUT the FROM_DEVICE invalidate. This is the bsync a cacheable
 * weight buffer needs before submit (the "clean cost" the probe measures separately). */
void ork_dma_clean_to_device(ork_npu *c, void *ptr, size_t size){
    struct buf *b=dma_find(c,ptr); if(!b) return;
    struct rknpu_mem_sync s; memset(&s,0,sizeof s);
    s.obj_addr=b->obj; s.offset=(uint64_t)((char*)ptr-(char*)b->cpu); s.size=size?size:b->size;
    s.flags=RKNPU_MEM_SYNC_TO_DEVICE; ioctl(c->fd,DRM_IOCTL_RKNPU_MEM_SYNC,&s);
}
const char *ork_npu_soc(const ork_npu *c){return c->soc->id;}
int ork_npu_cores(const ork_npu *c){return c->soc->cores;}
int ork_npu_validated(const ork_npu *c){return c->soc->validated;}
/* policy: cap the cores the auto-tuner may use for a matmul (n<=0 → all soc cores). The library
 * still picks per-matmul ≤ this (small-N matmuls use fewer). ORK_NPU_MC env overrides if set. */
void ork_npu_set_core_budget(ork_npu *c,int n){ if(!c)return; c->core_budget=(n>0&&n<=c->soc->cores)?n:c->soc->cores; }

/* PER-WEIGHT IOMMU DOMAIN PLACEMENT. The rk_iommu 32-bit IOVA cap (~4 GiB) is per iommu_domain_id, so a
 * model larger than 4 GiB stays fully resident (no streaming) by spreading its weights over domains.
 * Set the domain BEFORE packing/loading a weight: every subsequent ork_mm_pack_i8 / ork_mm_load_i8 (and
 * the fp16/int4 variants) places its resident tiles in `domain` and stamps it on the returned ork_w; at
 * run time ork_mm_run* submits that weight's matmuls against the same domain automatically. Activation/
 * output scratch follows the most-recently-set pack domain. domain<0 reverts to the process default
 * (env ORK_IOMMU_DOMAIN, else 0). Domains are created lazily by the kernel on first use. */
void ork_npu_set_pack_domain(ork_npu *c,int domain){ if(c) c->pack_domain = domain<0 ? -1 : domain; }
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
static int ork_all_cores_mask(cpu_set_t *s){
    long n=sysconf(_SC_NPROCESSORS_ONLN); if(n<1) return 0;
    CPU_ZERO(s); for(long i=0;i<n && i<CPU_SETSIZE;i++) CPU_SET((int)i,s);
    return (int)n;
}
/* Un-pin the calling thread: allow it to run on ALL online cores. Public so the ggml-ork dequant/
 * quant workers (std::thread, no attr) can un-pin themselves. */
void ork_unpin_current_thread(void){
    cpu_set_t all; if(ork_all_cores_mask(&all)) pthread_setaffinity_np(pthread_self(), sizeof all, &all);
}

/* PERSISTENT worker pool for ork_parallel_for. Spawn the workers ONCE and reuse them across every
 * call, amortizing the pthread_create/join that dominated fine-grained per-weight tiling — a fresh
 * pool per weight left the cores mostly idle in spawn/join overhead (measured: per-weight CPU tiling
 * capped ~20%). Workers are un-pinned (all cores) and sleep on a condvar between jobs; lazy-init on
 * first use, live for the process. One job at a time (the callers dispatch serially). */
#define ORK_POOL_MAX 64
static struct {
    int inited, n, quit;
    pthread_t th[ORK_POOL_MAX];
    pthread_mutex_t mu; pthread_cond_t go, done;
    void (*fn)(int,int,void*); void *ctx;
    int lo[ORK_POOL_MAX], hi[ORK_POOL_MAX];
    int gen, running;
} g_pool = { .mu = PTHREAD_MUTEX_INITIALIZER, .go = PTHREAD_COND_INITIALIZER, .done = PTHREAD_COND_INITIALIZER };

static void *ork_pool_worker(void *a){
    int id = (int)(intptr_t)a;
    ork_unpin_current_thread();
    int seen = 0;
    pthread_mutex_lock(&g_pool.mu);
    for(;;){
        while(g_pool.gen == seen && !g_pool.quit) pthread_cond_wait(&g_pool.go, &g_pool.mu);
        if(g_pool.quit){ pthread_mutex_unlock(&g_pool.mu); return NULL; }
        seen = g_pool.gen;
        void (*fn)(int,int,void*) = g_pool.fn; void *ctx = g_pool.ctx;
        int lo = g_pool.lo[id], hi = g_pool.hi[id];
        pthread_mutex_unlock(&g_pool.mu);
        if(fn && hi > lo) fn(lo, hi, ctx);
        pthread_mutex_lock(&g_pool.mu);
        if(--g_pool.running == 0) pthread_cond_signal(&g_pool.done);
    }
}
static void ork_pool_init(void){
    if(g_pool.inited) return;
    int cores=(int)sysconf(_SC_NPROCESSORS_ONLN); if(cores<1)cores=1;
    /* ORK_POOL_MULT oversubscribes the cores (e.g. =2 → 2 threads/core) to hide memory-latency stalls
     * in the tiling/dequant (bandwidth is not saturated, so extra threads can fill the stall bubbles). */
    int mult = getenv("ORK_POOL_MULT") ? atoi(getenv("ORK_POOL_MULT")) : 1; if(mult<1)mult=1; if(mult>8)mult=8;
    int n = cores*mult; if(n>ORK_POOL_MAX)n=ORK_POOL_MAX;
    g_pool.n = n; g_pool.inited = 1;
    for(int i=1;i<n;i++)   /* worker 0 == the caller; helpers 1..n-1 */
        if(pthread_create(&g_pool.th[i], NULL, ork_pool_worker, (void*)(intptr_t)i)!=0){ g_pool.n = i; break; }
}
static void ork_parallel_for(int n, void (*fn)(int,int,void*), void *ctx){
    if(n<=0) return;
    if(n==1){ fn(0,1,ctx); return; }
    static pthread_mutex_t dispatch = PTHREAD_MUTEX_INITIALIZER;   /* pool is a shared singleton — one job at a time */
    pthread_mutex_lock(&dispatch);
    ork_pool_init();
    int nthr = g_pool.n; if(nthr>n) nthr=n; if(nthr<1) nthr=1;   /* use the whole pool (incl. oversubscription) */
    if(nthr<=1){ pthread_mutex_unlock(&dispatch); fn(0,n,ctx); return; }
    int chunk = (n + nthr - 1) / nthr;
    pthread_mutex_lock(&g_pool.mu);
    g_pool.fn = fn; g_pool.ctx = ctx;
    for(int t=0;t<g_pool.n;t++){ int a=t*chunk; if(a>n)a=n; int b=a+chunk; if(b>n)b=n; g_pool.lo[t]=a; g_pool.hi[t]=b; }
    g_pool.running = g_pool.n - 1;   /* every helper wakes + decrements (idle ones just have empty ranges) */
    g_pool.gen++;
    pthread_cond_broadcast(&g_pool.go);
    pthread_mutex_unlock(&g_pool.mu);
    if(g_pool.hi[0] > g_pool.lo[0]) fn(g_pool.lo[0], g_pool.hi[0], ctx);   /* caller runs chunk 0 */
    pthread_mutex_lock(&g_pool.mu);
    while(g_pool.running > 0) pthread_cond_wait(&g_pool.done, &g_pool.mu);
    pthread_mutex_unlock(&g_pool.mu);
    pthread_mutex_unlock(&dispatch);
}
/* Tile a contiguous [nt] range of int8 weight columns into the NPU 32x32 block layout. Shared by the Bb
 * K-slice tiles and the full-K Bf rebuild (same structure; differ only in KT and the k0 offset). Each nt
 * range is disjoint in bb, so this is bit-identical to the serial loop. */
struct tile_i8_arg { int8_t *bb; const int8_t *Bi; int KT, k0, n0, N; };
static void tile_i8_range(int lo,int hi,void *a){
    struct tile_i8_arg *t=a;
    for(int nt=lo;nt<hi;nt++)for(int kt=0;kt<t->KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        t->bb[(size_t)nt*t->KT*1024+(size_t)kt*1024+(size_t)nl*32+kk]=
            t->Bi[(size_t)(t->k0+kt*32+kk)*t->N+(t->n0+nt*32+nl)];
}
static ork_w *pack(ork_npu *c,int K,int N,const void *B,int dt){
    int nmod=dt?32:16; if(K%32||N%nmod) return NULL;
    int KS=dt ? int8_ks(c) : c->soc->ks, NMAX=c->soc->nmax, nt_sz=dt?32:16, esz=dt?1:2;
    int Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=dt; w->owns=1; w->domain=ork_dom(c->pack_domain); w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    /* FIX 2 (gated, ORK_CONSOLIDATE_I8): consolidate all int8 Bb tiles into ONE per-weight DMA buffer,
     * tiles being page-aligned base+offset VIEWS — cuts thousands of GEM objects / MEM_CREATE / page-pad
     * to one alloc (matches rkllm's one-buffer-per-domain). owns flips to 0 + own_buf_valid so ork_mm_free
     * reclaims the single buffer. Each tile's regcmd bdma is own_buf.dma+off (validate_regcmd + the run
     * path read it exactly like a per-tile dma — same as the validated grouped-i4 own_buf path). Off by
     * default: this touches the regcmd ADDRESS MATH that wedged the wide-N path before; opt-in to de-risk.
     * Falls back to per-tile owning bcreate (below) on any alloc failure. */
    int consolidate = (dt==DT_I8) && getenv("ORK_CONSOLIDATE_I8");
    if(consolidate){
        size_t wtotal=0;
        for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
          for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;(void)n0;(void)k0;
            wtotal += pgup((size_t)Kp*Nc*esz);}}
        struct buf own=bcreate(c->fd,wtotal,0x403,w->domain);
        if(own.cpu){
            w->own_buf=own; w->own_buf_valid=1; w->owns=0;   /* tiles are views; reclaim own_buf as one */
            size_t off=0;
            for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/nt_sz;
              for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32; size_t ts=pgup((size_t)Kp*Nc*esz);
                struct buf*b=&w->Bb[(size_t)ns*Sk+ks];
                /* size = PAGE-PADDED tile (== per-tile bcreate's b->size) so ork_w_dump byte-matches the
                 * non-consolidated layout and round-trips through ork_mm_load_i8. */
                b->handle=own.handle; b->obj=own.obj; b->dma=own.dma+off; b->cpu=(char*)own.cpu+off; b->size=ts;
                int8_t*bb=b->cpu; const int8_t*Bi=B;
                for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
                    bb[nt*KT*32*32+kt*32*32+nl*32+kk]=Bi[(size_t)(k0+kt*32+kk)*N+(n0+nt*32+nl)];
                off += ts;}}
            bsync_off(c->fd,own.obj,0,wtotal,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
            bsync_off(c->fd,own.obj,0,wtotal,RKNPU_MEM_SYNC_TO_DEVICE);
        } else { consolidate=0; }   /* alloc failed → per-tile fallback below */
    }
    if(!consolidate)
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/nt_sz;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=bcreate(c->fd,(size_t)Kp*Nc*esz,0x403,w->domain);
        if(!b->cpu){
            fprintf(stderr,"[ork] ERROR: bcreate failed to allocate weight buffer Bb[%zu] in pack (size=%zu)\n",(size_t)ns*Sk+ks,(size_t)Kp*Nc*esz);
            for(int i=0;i<ns*Sk+ks;i++) bdestroy(c->fd,&w->Bb[i]);
            free(w->Bb); free(w); return NULL;
        }
        if(dt==DT_F16){ f16*bb=b->cpu; const f16*Bf=B;
            for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
                bb[nt*KT*16*32+kt*16*32+nl*32+kk]=Bf[(size_t)(k0+kt*32+kk)*N+(n0+nt*16+nl)];
        } else { int8_t*bb=b->cpu; const int8_t*Bi=B;
            struct tile_i8_arg ta={bb,Bi,KT,k0,n0,N}; ork_parallel_for(NN,tile_i8_range,&ta);   // all-core tiling
        }
        bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}}
    /* AUTO full-K decode layout (int8, K<=10752): lets the multi-core decode do
     * one full-K submit/core instead of ~K/1024 K-slices. ~2x weight memory — IOVA-FITS GUARD: if
     * any bcreate fails (IOMMU full on a big model), abandon Bf entirely → decode falls back to the
     * K-split path (correct, just slower). No crash, no ceiling guess. */
    if(dt==DT_I8 && K<=10752 && !(int8_ks(c)<K && getenv("ORK_KTILE"))){ int KTf=K/32; w->Bf=calloc(Sn,sizeof(struct buf)); int ok=1;
        for(int ns=0;ns<Sn && ok;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/nt_sz;
            struct buf*b=&w->Bf[ns]; *b=bcreate(c->fd,(size_t)K*Nc*esz,0x403,w->domain);
            if(!b->cpu){ ok=0; break; }                 /* IOVA full → give up on Bf */
            int8_t*bb=b->cpu; const int8_t*Bi=B;
            struct tile_i8_arg ta={bb,Bi,KTf,0,n0,N}; ork_parallel_for(NN,tile_i8_range,&ta);   // all-core full-K rebuild
            bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}
        if(!ok){ for(int ns=0;ns<Sn;ns++) bdestroy(c->fd,&w->Bf[ns]); free(w->Bf); w->Bf=NULL; } }
    return w;
}
ork_w *ork_mm_pack   (ork_npu *c,int K,int N,const f16    *B){ return pack(c,K,N,B,DT_F16); }
ork_w *ork_mm_pack_i8(ork_npu *c,int K,int N,const int8_t *B){ return pack(c,K,N,B,DT_I8);  }

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
 * tiling (same tile_i8_range, page-padded per tile, same Sk×Sn order as ork_w_dump) runs pure-CPU and
 * parallel across all cores; the NPU is touched only at LOAD time (ork_mm_load_i8_import). Pass out=NULL
 * to size. K%32, N%32. Byte-identical to the pack+dump path (fresh DMA bufs are zeroed; we zero-pad). */
size_t ork_w_dump_i8_cpu(ork_npu *c, int K, int N, const int8_t *B, void *out, size_t cap){
    if(!c || !B || (K%32) || (N%32)) return 0;
    int KS=int8_ks(c), NMAX=c->soc->nmax;
    int Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t off=0;
    for(int ns=0;ns<Sn;ns++){ int n0=ns*NMAX, Nc=(N-n0<NMAX)?(N-n0):NMAX, NN=Nc/32;
      for(int ks=0;ks<Sk;ks++){ int k0=ks*KS, Kp=(K-k0<KS)?(K-k0):KS, KT=Kp/32; size_t tsz=pgup((size_t)Kp*Nc);
        if(out){ if(off+tsz>cap) return 0;
            int8_t *bb=(int8_t*)out+off; memset(bb,0,tsz);   /* zero the page-pad (matches a fresh dma-buf) */
            struct tile_i8_arg ta={bb,B,KT,k0,n0,N}; ork_parallel_for(NN,tile_i8_range,&ta); }
        off+=tsz; }}
    return off;
}
/* NPU-availability gate for the hybrid pack scheduler: return 1 if the NPU appears IN USE (any core
 * loaded above a small threshold), 0 if idle. Reads the kernel's rolling per-core load counter. A
 * hybrid conversion routes a weight to the NPU tile/pack path ONLY when this is 0, so a background
 * .orkpack build never steals cycles from live inference on another process — the CPU path (tile from
 * pagecache into DRAM, zero-copy import) handles everything while the NPU serves. Best-effort: on any
 * read failure it returns 0 (assume idle) so the caller keeps the NPU option. Cheap enough to poll per
 * weight. Threshold >5% treats warm-up/idle noise as free but any real submit stream as busy. */
int ork_npu_busy(ork_npu *ctx){
    (void)ctx;
    FILE *f=fopen("/sys/kernel/debug/rknpu/load","r");
    if(!f) return 0;
    char buf[256]; size_t n=fread(buf,1,sizeof buf-1,f); fclose(f);
    if(!n) return 0; buf[n]=0;
    /* format: "NPU load:  Core0:  0%, Core1:  0%, Core2:  0%," — busy if any core % exceeds threshold */
    for(const char *p=buf; (p=strchr(p,'%')); p++){
        const char *q=p; while(q>buf && (q[-1]==' ' || (q[-1]>='0'&&q[-1]<='9'))) q--;
        if(atoi(q)>5) return 1;
    }
    return 0;
}
/* Reload pre-tiled int8 weight bytes (from ork_w_dump / a .orkpack) straight into NPU DMA — bcreate +
 * memcpy + bsync, with NO dequant / quant / tiling. The fast path for streaming persisted weights: a
 * re-pack becomes a plain DMA copy. `blob`/`n` must be this exact (K,N) int8 weight's Bb dump, in pack
 * order. Returns NULL on shape/size mismatch. Mirrors pack()'s int8 geometry (KS=1024).
 * Also rebuilds the full-K Bf buffer (K<=10752) — Bf is not dumped (it's a regenerable re-tiling of the
 * SAME bytes), but the decode fast path AND run_chain_i8 (Sk>1 experts) need it, so a loaded weight must
 * carry it to be a first-class drop-in for a packed one. Bf is reconstructed from Bb (un-tile → B[K][N] →
 * re-tile full-K); on IOVA exhaustion it's abandoned (Bf=NULL) → decode/run_i8 K-split still works. */
ork_w *ork_mm_load_i8(ork_npu *c,int K,int N,const void *blob,size_t n){
    if(K%32 || N%32) return NULL;
    int KS=1024, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t need=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS; need+=pgup((size_t)Kp*Nc);}}
    if(n!=need) return NULL;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I8; w->owns=1; w->domain=ork_dom(c->pack_domain);
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    size_t off=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=bcreate(c->fd,(size_t)Kp*Nc,0x403,w->domain);
        if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; }
        memcpy(b->cpu,(const char*)blob+off,b->size); off+=b->size;
        bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}}
    /* Rebuild Bf (full-K layout) so loaded weights chain + decode like packed ones. The Bb tiles are
     * 32x32 blocks: Bb[ns][ks] holds B[k0+kt*32+kk][n0+nt*32+nl] at [nt][kt][nl][kk]. Bf[ns] re-tiles the
     * full K dimension (KTf=K/32) of the SAME logical B for that N-slice.
     * ONLY build Bf where a full-K submit is actually valid (K%512==0 && K<=4096) — the same envelope
     * run() / run_chain_i8 use. Outside it (e.g. K=1792 ffn_down experts) Bf would never be read and just
     * doubles resident NPU bytes, exhausting the 4 GiB IOMMU window when many experts are loaded. Those
     * weights run via the K-split Bb path (run_i8), which doesn't need Bf. */
    if(K%512==0 && K<=4096){ int KTf=K/32; w->Bf=calloc(Sn,sizeof(struct buf)); int ok=1;
        for(int ns=0;ns<Sn && ok;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
            struct buf*bf=&w->Bf[ns]; *bf=bcreate(c->fd,(size_t)K*Nc,0x403,w->domain);
            if(!bf->cpu){ ok=0; break; }                /* IOVA full → give up on Bf */
            int8_t*fb=bf->cpu;
            for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
                const int8_t*sb=(const int8_t*)w->Bb[(size_t)ns*Sk+ks].cpu;
                for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++){
                    int ktf=(k0/32)+kt;   /* full-K tile index */
                    fb[(size_t)nt*KTf*32*32+(size_t)ktf*32*32+nl*32+kk]=
                        sb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]; }}
            bsync(c->fd,bf,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(c->fd,bf,RKNPU_MEM_SYNC_TO_DEVICE);}
        if(!ok){ for(int ns=0;ns<Sn;ns++) bdestroy(c->fd,&w->Bf[ns]); free(w->Bf); w->Bf=NULL; } }
    return w;
}

/* Zero-copy IMPORT variant of ork_mm_load_i8: each resident tile is a dma-buf the NPU reads in place
 * (PRIME import) instead of a MEM_CREATE-alloc'd buffer the blob is memcpy'd into. The bytes still get
 * written once (into the imported mmap) + synced once; the saving is the kernel page allocation, not
 * the host fill (load is from a disk/RAM blob either way). Same blob format / round-trip as load_i8.
 * Falls through to NULL (caller uses ork_mm_load_i8) if import is unavailable. */
ork_w *ork_mm_load_i8_import(ork_npu *c,int K,int N,const void *blob,size_t n){
    if(K%32 || N%32) return NULL;
    if(dmaheap_open()<0) return NULL;
    int KS=1024, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t need=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS;(void)n0; need+=pgup((size_t)Kp*Nc);}}
    if(n!=need) return NULL;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I8; w->owns=1; w->domain=ork_dom(c->pack_domain);
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    size_t off=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS;(void)n0;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=bimport(c->fd,(size_t)Kp*Nc,w->domain);
        if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; }
        dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
        memcpy(b->cpu,(const char*)blob+off,(size_t)Kp*Nc); off+=b->size;
        dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
    /* Bf full-K rebuild (same envelope as ork_mm_load_i8): imported too, abandoned on failure. */
    if(K%512==0 && K<=4096){ int KTf=K/32; w->Bf=calloc(Sn,sizeof(struct buf)); int ok=1;
        for(int ns=0;ns<Sn && ok;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
            struct buf*bf=&w->Bf[ns]; *bf=bimport(c->fd,(size_t)K*Nc,w->domain);
            if(!bf->cpu){ ok=0; break; }
            int8_t*fb=bf->cpu;
            dmabuf_sync(bf->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
            for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
                const int8_t*sb=(const int8_t*)w->Bb[(size_t)ns*Sk+ks].cpu;
                for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++){
                    int ktf=(k0/32)+kt;
                    fb[(size_t)nt*KTf*32*32+(size_t)ktf*32*32+nl*32+kk]=
                        sb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]; }}
            dmabuf_sync(bf->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}
        if(!ok){ for(int ns=0;ns<Sn;ns++) bdestroy(c->fd,&w->Bf[ns]); free(w->Bf); w->Bf=NULL; } }
    return w;
}

/* Re-tile int8 B[K,N] into an EXISTING ork_w's resident buffers (same K,N), reusing the DMA
 * allocations — NO bcreate/bdestroy. For pooling reused weights (e.g. MoE experts) so the NPU IOMMU
 * isn't churned/fragmented by per-weight alloc+free. Returns 0 ok, -1 bad arg, -2 shape mismatch. */
int ork_mm_repack_i8(ork_npu *c,ork_w *w,int K,int N,const int8_t *B){
    if(!w || w->dtype!=DT_I8 || !w->Bb) return -1;
    if(w->K!=K || w->N!=N) return -2;                  /* must match the slot's allocated shape */
    int KS=1024, NMAX=c->soc->nmax, Sk=w->Sk, Sn=w->Sn;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; if(!b->cpu) return -1; int8_t*bb=b->cpu;
        for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
            bb[nt*KT*32*32+kt*32*32+nl*32+kk]=B[(size_t)(k0+kt*32+kk)*N+(n0+nt*32+nl)];
        bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}}
    if(w->Bf && K<=10752){ int KTf=K/32;
        for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
            struct buf*b=&w->Bf[ns]; if(!b->cpu) continue; int8_t*bb=b->cpu;
            for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KTf;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
                bb[(size_t)nt*KTf*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(n0+nt*32+nl)];
            bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}}
    return 0;
}
/* ---- Diagnostic only (tools/dmabuf_fill_probe.c): a load_i8 variant whose resident Bb tiles are
 * allocated with a CALLER-CHOSEN rknpu mem flag (0x401 WC vs 0x403 cacheable), so the probe can A/B
 * the weight-fill bandwidth AND the NPU read correctness for each flag. Allocates + leaves the blob
 * copied in once (a valid initial state); the probe then times steady-state re-fills via the accessors
 * below. Additive; not in the public header; does NOT change pack/run or the default load_i8. ---- */
ork_w *ork_mm_load_i8_flags(ork_npu *c,int K,int N,const void *blob,size_t n,unsigned flags){
    if(K%32 || N%32) return NULL;
    int KS=1024, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t need=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS;(void)n0; need+=pgup((size_t)Kp*Nc);}}
    if(n!=need) return NULL;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I8; w->owns=1; w->domain=ork_dom(c->pack_domain);
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    size_t off=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS;(void)n0;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=bcreate(c->fd,(size_t)Kp*Nc,flags,w->domain);
        if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; }
        memcpy(b->cpu,(const char*)blob+off,b->size); off+=b->size;
        bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}}
    return w;
}
/* Diagnostic accessors over an ork_w's resident Bb tiles (for the fill probe): number of tiles, and
 * the cpu ptr + byte size of tile i. The probe memcpys blob bytes into these to time steady-state fill. */
int    ork_w_ntiles(const ork_w *w){ return (w&&w->Bb)?w->Sk*w->Sn:0; }
void  *ork_w_tile_cpu(const ork_w *w,int i){ return (w&&w->Bb&&i>=0&&i<w->Sk*w->Sn)?w->Bb[i].cpu:NULL; }
size_t ork_w_tile_size(const ork_w *w,int i){ return (w&&w->Bb&&i>=0&&i<w->Sk*w->Sn)?w->Bb[i].size:0; }
/* clean-only flush (TO_DEVICE) of tile i — the bsync a cacheable weight buffer needs before submit. */
void   ork_w_tile_clean(ork_npu *c,const ork_w *w,int i){
    if(!w||!w->Bb||i<0||i>=w->Sk*w->Sn||!w->Bb[i].cpu) return;
    bsync(c->fd,&w->Bb[i],RKNPU_MEM_SYNC_TO_DEVICE);
}
/* the full TO|FROM then TO bsync (the current ork_dma_bsync_to_device pattern), per tile. */
void   ork_w_tile_bsync_full(ork_npu *c,const ork_w *w,int i){
    if(!w||!w->Bb||i<0||i>=w->Sk*w->Sn||!w->Bb[i].cpu) return;
    bsync(c->fd,&w->Bb[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    bsync(c->fd,&w->Bb[i],RKNPU_MEM_SYNC_TO_DEVICE);
}

/* ---- NEON SIMD pack/repack DIRECTLY from f32[N][K] (n-major, as ggml's to_float produces) ----
 * Fuses per-channel f32->int8 quant INTO the tile loop: for a fixed channel n the 32 K-values are
 * contiguous in f32[n][:], so NEON-load 32 f32, mul by the channel inverse scale, round/clamp to
 * [-127,127], narrow to 32 CONTIGUOUS int8 — no transposed scratch (the old transpose store was ~69%
 * of the MoE repack). Computes per-channel bscale[] for the caller. */
static void chan_scales_f32(const float *f32, int K, int N, float *inv, float *bscale) {
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
static inline void quant32_f32_i8(int8_t *dst, const float *fr, float inv) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t vinv = vdupq_n_f32(inv); int32x4_t lo = vdupq_n_s32(-127);
    for (int kk = 0; kk < 32; kk += 8) {
        int32x4_t i0 = vmaxq_s32(vcvtnq_s32_f32(vmulq_f32(vld1q_f32(fr + kk),     vinv)), lo);
        int32x4_t i1 = vmaxq_s32(vcvtnq_s32_f32(vmulq_f32(vld1q_f32(fr + kk + 4), vinv)), lo);
        vst1_s8(dst + kk, vqmovn_s16(vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1))));
    }
#else
    for (int kk = 0; kk < 32; kk++) { int q = (int)lrintf(fr[kk] * inv); dst[kk] = (int8_t)(q > 127 ? 127 : q < -127 ? -127 : q); }
#endif
}
/* tile f32[N][K] -> int8 NPU layout (Bb K-split + Bf full-K) via precomputed per-channel inv[]. Each
 * buffer gets the full init sync (TO|FROM then TO) that pack() uses — fresh buffers need it (a single TO
 * leaves the device side uninitialized -> the NPU submit wedges/times out). */
static void tile_f32_i8(ork_npu *c, ork_w *w, int K, int N, const float *f32, const float *inv) {
    int KS = 1024, NMAX = c->soc->nmax, Sk = w->Sk, Sn = w->Sn, fd = c->fd;
    for (int ns = 0; ns < Sn; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX, NN = Nc/32;
      for (int ks = 0; ks < Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS, KT = Kp/32;
        struct buf *b = &w->Bb[(size_t)ns*Sk+ks]; if (!b->cpu) continue; int8_t *bb = b->cpu;
        for (int nt = 0; nt < NN; nt++) for (int nl = 0; nl < 32; nl++) {
            int n = n0+nt*32+nl; const float *frn = f32 + (size_t)n*K + k0; float iv = inv[n];
            for (int kt = 0; kt < KT; kt++) quant32_f32_i8(bb + ((size_t)nt*KT*32*32 + (size_t)kt*32*32 + nl*32), frn + kt*32, iv);
        }
        bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE); } }
    if (w->Bf && K <= 10752) { int KTf = K/32;
        for (int ns = 0; ns < Sn; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX, NN = Nc/32;
            struct buf *b = &w->Bf[ns]; if (!b->cpu) continue; int8_t *bb = b->cpu;
            for (int nt = 0; nt < NN; nt++) for (int nl = 0; nl < 32; nl++) {
                int n = n0+nt*32+nl; const float *frn = f32 + (size_t)n*K; float iv = inv[n];
                for (int kt = 0; kt < KTf; kt++) quant32_f32_i8(bb + ((size_t)nt*KTf*32*32 + (size_t)kt*32*32 + nl*32), frn + kt*32, iv);
            }
            bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE); } }
}
ork_w *ork_mm_pack_i8_f32(ork_npu *c, int K, int N, const float *f32, float *bscale_out) {
    if (K % 32 || N % 32) return NULL;
    int KS = 1024, NMAX = c->soc->nmax, Sk = (K+KS-1)/KS, Sn = (N+NMAX-1)/NMAX;
    ork_w *w = calloc(1, sizeof *w); if (!w) return NULL;
    w->K = K; w->N = N; w->Sk = Sk; w->Sn = Sn; w->dtype = DT_I8; w->Bb = calloc((size_t)Sk*Sn, sizeof(struct buf));
    if (!w->Bb) { free(w); return NULL; }
    for (int ns = 0; ns < Sn; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX;
      for (int ks = 0; ks < Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS;
        struct buf *b = &w->Bb[(size_t)ns*Sk+ks]; *b = bcreate(c->fd, (size_t)Kp*Nc, 0x403, w->domain);
        if (!b->cpu) { for (int i = 0; i < ns*Sk+ks; i++) bdestroy(c->fd, &w->Bb[i]); free(w->Bb); free(w); return NULL; } } }
    if (K <= 10752) { w->Bf = calloc(Sn, sizeof(struct buf)); int ok = 1;
        for (int ns = 0; ns < Sn && ok; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX;
            struct buf *b = &w->Bf[ns]; *b = bcreate(c->fd, (size_t)K*Nc, 0x403, w->domain); if (!b->cpu) ok = 0; }
        if (!ok) { for (int ns = 0; ns < Sn; ns++) bdestroy(c->fd, &w->Bf[ns]); free(w->Bf); w->Bf = NULL; } }
    float *inv = malloc((size_t)N * sizeof(float)); if (!inv) { ork_w_free(w); return NULL; }
    chan_scales_f32(f32, K, N, inv, bscale_out);
    tile_f32_i8(c, w, K, N, f32, inv);
    free(inv);
    return w;
}
/* ---- "effective w4a8": int4-PRECISION weights, int8 compute, int4 STORAGE ----
 * RK3588's NPU MACs are int8-only — there is no native int4->int8 datapath. So we synthesize w4a8:
 * quantize each weight to int4 precision (per-channel scale = max|w|/7, range [-7,7]), keep the
 * compact nibble-packed form on the ork_w (w->Bi4, K*N/2 bytes — the memory win + the on-disk form
 * for .orkpack/streaming), then NEON-expand the nibbles back to int8 [-7,7] and DMA-tile that through
 * the existing int8 path so the result runs via ork_mm_run_i8 unchanged. bscale_out[n] carries the
 * dequant scale (C_real[m][n] = aScale[m]*bscale[n]*Ci[m][n], same convention as pack_i8_f32). */
static inline uint32_t ork_xs32(uint32_t *s){ uint32_t x=*s; x^=x<<13; x^=x>>17; x^=x<<5; *s=x; return x; }
/* quantize one output channel's K f32 weights -> int4 q in [-7,7], nibble-pack into `nib` (K/2 bytes),
 * and write the dequantized int8 q-value (== the int4 code) as f32 into `qf32[K]` for the int8 tiler.
 * sr!=0: stochastic rounding (q=floor(w/scale + u), u in [0,1) from xorshift) — SR removes the
 * quantization BIAS so dot-product error grows ~sqrt(K) instead of O(K). seed advanced per element. */
static void quant_chan_i4(const float *fr, int K, float scale, int sr, uint32_t *seed, uint8_t *nib, float *qf32) {
    float inv = scale > 0 ? 1.0f/scale : 0.0f;
    for (int k = 0; k < K; k++) {
        int q;
        if (sr) { float u = (float)(ork_xs32(seed) >> 8) * (1.0f/16777216.0f);  /* u in [0,1) */
                  q = (int)floorf(fr[k]*inv + u); }
        else      q = (int)lrintf(fr[k]*inv);
        if (q > 7) q = 7; else if (q < -7) q = -7;
        qf32[k] = (float)q;
        uint8_t nb = (uint8_t)(q & 0xf);              /* low nibble holds the signed 4-bit code */
        if (k & 1) nib[k>>1] |= (uint8_t)(nb << 4); else nib[k>>1] = nb;
    }
}
/* NEON-expand one channel's nibble-packed int4 codes -> int8 [-7,7] sign-extended floats in qf32[K].
 * (The "NEON where the NPU can't": the hardware has no int4->int8 expansion datapath; we do it in
 * software, then feed the int8 tiler.) Bulk path does 16 codes (8 bytes) per iteration. */
static void expand_chan_i4_f32(const uint8_t *nib, int K, float *qf32) {
    int k = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    uint8x8_t vlo = vdup_n_u8(0x0f);
    for (; k <= K - 16; k += 16) {
        uint8x8_t pk = vld1_u8(nib + (k>>1));                 /* 8 bytes = 16 nibbles */
        int8x8_t even = vreinterpret_s8_u8(vand_u8(pk, vlo)); /* low nibbles  (codes k,k+2,...) */
        int8x8_t odd  = vreinterpret_s8_u8(vshr_n_u8(pk, 4)); /* high nibbles (codes k+1,...) */
        /* sign-extend 4-bit: shift the nibble into the top of an int8, then arithmetic-shift back */
        even = vshr_n_s8(vshl_n_s8(even, 4), 4);
        odd  = vshr_n_s8(vshl_n_s8(odd,  4), 4);
        int8x8x2_t zip = vzip_s8(even, odd);                  /* interleave -> code order */
        int8x16_t codes = vcombine_s8(zip.val[0], zip.val[1]);
        int16x8_t lo16 = vmovl_s8(vget_low_s8(codes));
        int16x8_t hi16 = vmovl_s8(vget_high_s8(codes));
        vst1q_f32(qf32 + k,      vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))));
        vst1q_f32(qf32 + k + 4,  vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))));
        vst1q_f32(qf32 + k + 8,  vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))));
        vst1q_f32(qf32 + k + 12, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))));
    }
#endif
    for (; k < K; k++) {
        uint8_t nb = (k & 1) ? (nib[k>>1] >> 4) : (nib[k>>1] & 0xf);
        int8_t c = (int8_t)(nb << 4) >> 4;                    /* sign-extend 4-bit */
        qf32[k] = (float)c;
    }
}
/* ---- NF4 codebook (non-uniform int4): the 16 fixed bitsandbytes NF4 normalized levels, index 0..15 ----
 * Unlike UNIFORM (a symmetric int4 grid), the 4-bit value indexes this per-tensor codebook of levels tuned
 * for ~N(0,1) weights. Per-channel scale = max|w_n| (so bscale[n]=absmax/127 and the int8 LUT = round(level*127)
 * reconstructs level*absmax). Better accuracy than uniform int4 for Gaussian-ish weights. */
static const float ORK_NF4_LEVELS[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f };
/* Quantize one output channel's K f32 weights to NF4: per-element find the nearest NF4 level -> 4-bit index
 * (0..15), nibble-pack into `nib` (K/2 bytes). The NF4 levels are monotonically increasing, so we find the
 * bracketing pair [lo,hi] and pick the nearer. sr!=0: stochastic-round between the two bracketing levels —
 * pick lo with probability proportional to (w_norm - level[lo])/(level[hi]-level[lo]) toward hi. `absmax`
 * is the per-channel max|w| (>0); w_norm=w/absmax in [-1,1]. Indices written to qidx[K] for the int8 inflate. */
static void quant_chan_nf4(const float *fr, int K, float absmax, int sr, uint32_t *seed, uint8_t *nib, uint8_t *qidx) {
    float inv = absmax > 0 ? 1.0f/absmax : 0.0f;
    for (int k = 0; k < K; k++) {
        float wn = fr[k]*inv; if (wn > 1.0f) wn = 1.0f; else if (wn < -1.0f) wn = -1.0f;
        /* find bracketing pair: hi = first level >= wn */
        int hi = 0; while (hi < 15 && ORK_NF4_LEVELS[hi] < wn) hi++;
        int lo = hi > 0 ? hi-1 : 0;
        int idx;
        if (lo == hi) idx = hi;
        else {
            float dlo = ORK_NF4_LEVELS[hi]-ORK_NF4_LEVELS[lo];     /* span > 0 */
            float t = (wn - ORK_NF4_LEVELS[lo]) / dlo;             /* fractional pos in [0,1] toward hi */
            if (sr) { float u = (float)(ork_xs32(seed) >> 8) * (1.0f/16777216.0f); /* u in [0,1) */
                      idx = (t > u) ? hi : lo; }                   /* P(hi) = t */
            else      idx = (t >= 0.5f) ? hi : lo;                 /* nearest level */
        }
        qidx[k] = (uint8_t)idx;
        uint8_t nb = (uint8_t)(idx & 0xf);
        if (k & 1) nib[k>>1] |= (uint8_t)(nb << 4); else nib[k>>1] = nb;
    }
}
/* Inflate one channel's NF4 indices (0..15) -> int8 codes via the fixed 16-entry LUT (round(level*127)),
 * writing f32 for the int8 tiler. NEON path uses vqtbl1q_u8 (16-byte table lookup, 16 idx/iter). The LUT
 * is the SAME for every channel (per-tensor codebook); bscale[n]=absmax/127 carries the per-channel scale. */
static void inflate_chan_nf4_f32(const uint8_t *qidx, int K, const int8_t lut[16], float *qf32) {
    int k = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int8x16_t vlut = vld1q_s8(lut);
    for (; k <= K - 16; k += 16) {
        uint8x16_t vi = vld1q_u8(qidx + k);
        int8x16_t codes = vqtbl1q_s8(vlut, vi);               /* table lookup: code = lut[idx] */
        int16x8_t lo16 = vmovl_s8(vget_low_s8(codes));
        int16x8_t hi16 = vmovl_s8(vget_high_s8(codes));
        vst1q_f32(qf32 + k,      vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))));
        vst1q_f32(qf32 + k + 4,  vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))));
        vst1q_f32(qf32 + k + 8,  vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))));
        vst1q_f32(qf32 + k + 12, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))));
    }
#endif
    for (; k < K; k++) qf32[k] = (float)lut[qidx[k]];
}
/* ---- DIRECT int4 -> int8-tiled inflate (no f32 intermediate, no re-quant) ----
 * The f32 path inflates nibble -> f32 code -> tile_f32_i8, which re-quantizes via
 * lrintf(code*1.0) clamped to [-127,127]. But the codes are ALWAYS exact small ints
 * (UNIFORM in [-7,7]; NF4 LUT = round(level*127) in [-127,127]) so that quant is the
 * identity: the int8 byte placed in the tile equals the int4 code. So we can inflate
 * straight to int8 and rearrange bytes into the tile layout with NO float round-trip.
 * Output is bit-identical to the f32 path (proven by the matmul/memcmp gate). */
/* UNIFORM: expand one channel's nibble-packed int4 codes -> LINEAR int8 [-7,7] in i8[K]. */
static void expand_chan_i4_i8(const uint8_t *nib, int K, int8_t *i8) {
    int k = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    uint8x8_t vlo = vdup_n_u8(0x0f);
    for (; k <= K - 16; k += 16) {
        uint8x8_t pk = vld1_u8(nib + (k>>1));                 /* 8 bytes = 16 nibbles */
        int8x8_t even = vreinterpret_s8_u8(vand_u8(pk, vlo)); /* low nibbles  (codes k,k+2,...) */
        int8x8_t odd  = vreinterpret_s8_u8(vshr_n_u8(pk, 4)); /* high nibbles (codes k+1,...) */
        even = vshr_n_s8(vshl_n_s8(even, 4), 4);              /* sign-extend 4-bit */
        odd  = vshr_n_s8(vshl_n_s8(odd,  4), 4);
        int8x8x2_t zip = vzip_s8(even, odd);                  /* interleave -> code order */
        vst1q_s8(i8 + k, vcombine_s8(zip.val[0], zip.val[1]));
    }
#endif
    for (; k < K; k++) {
        uint8_t nb = (k & 1) ? (nib[k>>1] >> 4) : (nib[k>>1] & 0xf);
        i8[k] = (int8_t)(nb << 4) >> 4;                       /* sign-extend 4-bit */
    }
}
/* NF4: inflate one channel's indices (stored in the nibble) -> LINEAR int8 codes via the LUT.
 * The nibble store keeps the 0..15 index (low/high nibble per k); LUT[idx] = round(level*127). */
static void inflate_chan_nf4_i8(const uint8_t *nib, int K, const int8_t lut[16], int8_t *i8) {
    int k = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int8x16_t vlut = vld1q_s8(lut); uint8x8_t vlo = vdup_n_u8(0x0f);
    for (; k <= K - 16; k += 16) {
        uint8x8_t pk = vld1_u8(nib + (k>>1));                 /* 8 bytes = 16 indices */
        uint8x8_t even = vand_u8(pk, vlo);                    /* low nibbles (idx k,k+2,...) */
        uint8x8_t odd  = vshr_n_u8(pk, 4);                    /* high nibbles (idx k+1,...) */
        uint8x8x2_t zip = vzip_u8(even, odd);                 /* interleave -> index order */
        uint8x16_t vi = vcombine_u8(zip.val[0], zip.val[1]);
        vst1q_s8(i8 + k, vqtbl1q_s8(vlut, vi));               /* code = lut[idx] */
    }
#endif
    for (; k < K; k++) { uint8_t idx = (k & 1) ? (nib[k>>1] >> 4) : (nib[k>>1] & 0xf); i8[k] = lut[idx]; }
}
/* Rearrange LINEAR int8 codes i8[N][K] -> the NPU tiled int8 layout, copying bytes (NO quant, NO float).
 * Byte-for-byte the same destination math as tile_f32_i8 (per (ns,ks) buffer: element of channel
 * n=n0+nt*32+nl at k-pos k0+kt*32+ki lands at nt*KT*32*32 + kt*32*32 + nl*32 + ki) but feeding the int8
 * code directly, since tile_f32_i8 with inv=1 maps code -> clamp(lrintf(code),-127,127) = code (identity).
 * Same per-buffer init bsync sequence as tile_f32_i8 (fresh buffers need TO|FROM then TO). */
static void tile_i8_to_tiles(ork_npu *c, ork_w *w, int K, int N, const int8_t *i8) {
    int KS = 1024, NMAX = c->soc->nmax, Sk = w->Sk, Sn = w->Sn, fd = c->fd;
    for (int ns = 0; ns < Sn; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX, NN = Nc/32;
      for (int ks = 0; ks < Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS, KT = Kp/32;
        struct buf *b = &w->Bb[(size_t)ns*Sk+ks]; if (!b->cpu) continue; int8_t *bb = b->cpu;
        for (int nt = 0; nt < NN; nt++) for (int nl = 0; nl < 32; nl++) {
            int n = n0+nt*32+nl; const int8_t *src = i8 + (size_t)n*K + k0;
            for (int kt = 0; kt < KT; kt++)
                memcpy(bb + ((size_t)nt*KT*32*32 + (size_t)kt*32*32 + nl*32), src + kt*32, 32);
        }
        bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE); } }
    if (w->Bf && K <= 10752) { int KTf = K/32;
        for (int ns = 0; ns < Sn; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX, NN = Nc/32;
            struct buf *b = &w->Bf[ns]; if (!b->cpu) continue; int8_t *bb = b->cpu;
            for (int nt = 0; nt < NN; nt++) for (int nl = 0; nl < 32; nl++) {
                int n = n0+nt*32+nl; const int8_t *src = i8 + (size_t)n*K;
                for (int kt = 0; kt < KTf; kt++)
                    memcpy(bb + ((size_t)nt*KTf*32*32 + (size_t)kt*32*32 + nl*32), src + kt*32, 32);
            }
            bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE); } }
}
/* DIRECT int4 -> int8-tiled fill: inflate w's nibble store straight into its resident DMA tiles, no f32.
 * (kind selects UNIFORM sign-extend vs NF4 LUT.) Uses a per-channel linear-int8 scratch i8scratch[N*K]
 * (1 byte/elem vs the f32 path's 4) reused across channels. Produces bit-identical tiled bytes to the
 * f32 path. Caller provides scratch (size N*K) so the streaming consumer can reuse one allocation. */
static void tile_direct_i4_i8(ork_npu *c, ork_w *w, int K, int N, int kind, int8_t *i8scratch) {
    if (kind == ORK_QK_CODEBOOK_NF4) {
        int8_t lut[16]; for (int i = 0; i < 16; i++) lut[i] = (int8_t)lrintf(ORK_NF4_LEVELS[i]*127.0f);
        for (int n = 0; n < N; n++) inflate_chan_nf4_i8(w->Bi4 + (size_t)n*(K/2), K, lut, i8scratch + (size_t)n*K);
    } else {
        for (int n = 0; n < N; n++) expand_chan_i4_i8(w->Bi4 + (size_t)n*(K/2), K, i8scratch + (size_t)n*K);
    }
    tile_i8_to_tiles(c, w, K, N, i8scratch);
}
/* ---- imatrix (importance-matrix) weighted per-channel scale selection ----
 * The quant scale is per-OUTPUT-channel; the imatrix is per-INPUT-channel (length K, importance[k] =
 * <activation_k^2>). They are orthogonal, so the imatrix can't re-weight a nearest-level pick. Instead we
 * pick the per-channel scale (clip ratio): for a grid of r in (0,1], the candidate scale is r*absmax;
 * quantizing at a smaller scale clips outliers but gives the bulk more resolution. We keep the r whose
 * dequant minimizes Sum_k imatrix[k]*(w[k]-dequant[k])^2. O(grid*K) per channel (pack is one-time). */
static const float ORK_IM_CLIP_GRID[] = { 1.0f, 0.92f, 0.85f, 0.78f, 0.70f, 0.62f, 0.55f };
#define ORK_IM_CLIP_N ((int)(sizeof(ORK_IM_CLIP_GRID)/sizeof(ORK_IM_CLIP_GRID[0])))
/* Quantize one channel at the given per-channel absmax (uniform: scale=absmax/7; NF4: scale=absmax/127)
 * into a reused dq[K] scratch (the dequantized weight, in original f32 units), and return the
 * importance-weighted reconstruction error Sum_k im[k]*(w[k]-dq[k])^2 (im NULL => unit weights).
 * Does NOT touch the nibble store — used to score a clip candidate; the winner is re-committed below. */
static float wq_err_chan(const float *fr, int K, float absmax, int nf4, const float *im, float *dq) {
    if (nf4) {
        float sc = absmax / 127.0f, inv = absmax > 0 ? 1.0f/absmax : 0.0f;
        for (int k = 0; k < K; k++) {
            float wn = fr[k]*inv; if (wn > 1.0f) wn = 1.0f; else if (wn < -1.0f) wn = -1.0f;
            int hi = 0; while (hi < 15 && ORK_NF4_LEVELS[hi] < wn) hi++;
            int lo = hi > 0 ? hi-1 : 0, idx;
            if (lo == hi) idx = hi;
            else { float t = (wn-ORK_NF4_LEVELS[lo])/(ORK_NF4_LEVELS[hi]-ORK_NF4_LEVELS[lo]); idx = (t >= 0.5f) ? hi : lo; }
            dq[k] = (float)((int8_t)lrintf(ORK_NF4_LEVELS[idx]*127.0f)) * sc;  /* match the int8 LUT path */
        }
    } else {
        float scale = absmax / 7.0f, inv = scale > 0 ? 1.0f/scale : 0.0f;
        for (int k = 0; k < K; k++) {
            int q = (int)lrintf(fr[k]*inv); if (q > 7) q = 7; else if (q < -7) q = -7;
            dq[k] = (float)q * scale;
        }
    }
    float e = 0.0f;
    for (int k = 0; k < K; k++) { float d = fr[k]-dq[k]; e += (im ? im[k] : 1.0f) * d*d; }
    return e;
}
/* Search the clip grid for one channel and return the absmax (= r*rawabsmax) that minimizes the
 * imatrix-weighted error. dq is reused scratch[K]. With im==NULL this would just return rawabsmax. */
static float wq_best_absmax(const float *fr, int K, float rawabsmax, int nf4, const float *im, float *dq) {
    float best_abs = rawabsmax, best_e = wq_err_chan(fr, K, rawabsmax, nf4, im, dq);
    for (int g = 1; g < ORK_IM_CLIP_N; g++) {
        float cand = rawabsmax * ORK_IM_CLIP_GRID[g];
        float e = wq_err_chan(fr, K, cand, nf4, im, dq);
        if (e < best_e) { best_e = e; best_abs = cand; }
    }
    return best_abs;
}
ork_w *ork_mm_pack_i4a8(ork_npu *c, int K, int N, const float *f32, float *bscale_out) {
    return ork_mm_pack_i4a8_im(c, K, N, f32, NULL, bscale_out);
}
ork_w *ork_mm_pack_i4a8_im(ork_npu *c, int K, int N, const float *f32, const float *imatrix, float *bscale_out) {
    if (K % 32 || N % 32) return NULL;
    int sr = getenv("ORK_SR") != NULL; uint32_t seed = 0x2545F491u;   /* SR PRNG: fixed seed => deterministic/testable */
    int nf4 = getenv("ORK_NF4") != NULL;   /* ORK_NF4: non-uniform NF4 codebook instead of the uniform int4 grid */
    int KS = 1024, NMAX = c->soc->nmax, Sk = (K+KS-1)/KS, Sn = (N+NMAX-1)/NMAX;
    ork_w *w = calloc(1, sizeof *w); if (!w) return NULL;
    w->K = K; w->N = N; w->Sk = Sk; w->Sn = Sn; w->dtype = DT_I8; w->owns = 1; w->domain=ork_dom(c->pack_domain); w->quant_kind = nf4 ? ORK_QK_CODEBOOK_NF4 : ORK_QK_UNIFORM;
    w->Bb = calloc((size_t)Sk*Sn, sizeof(struct buf));
    if (!w->Bb) { free(w); return NULL; }
    for (int ns = 0; ns < Sn; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX;
      for (int ks = 0; ks < Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS;
        struct buf *b = &w->Bb[(size_t)ns*Sk+ks]; *b = bcreate(c->fd, (size_t)Kp*Nc, 0x403, w->domain);
        if (!b->cpu) { for (int i = 0; i < ns*Sk+ks; i++) bdestroy(c->fd, &w->Bb[i]); free(w->Bb); free(w); return NULL; } } }
    if (K <= 10752) { w->Bf = calloc(Sn, sizeof(struct buf)); int ok = 1;
        for (int ns = 0; ns < Sn && ok; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX;
            struct buf *b = &w->Bf[ns]; *b = bcreate(c->fd, (size_t)K*Nc, 0x403, w->domain); if (!b->cpu) ok = 0; }
        if (!ok) { for (int ns = 0; ns < Sn; ns++) bdestroy(c->fd, &w->Bf[ns]); free(w->Bf); w->Bf = NULL; } }
    /* compact int4 nibble store (n-major, K contiguous): the memory-win form, kept on the ork_w */
    w->Bi4_bytes = (size_t)N * (K/2);
    w->Bi4 = malloc(w->Bi4_bytes);
    /* retain per-channel dequant scale on the ork_w so the compact int4 form can be dumped self-contained */
    w->bscale = malloc((size_t)N * sizeof(float));
    /* int8 expansion scratch (f32 codes) + per-channel inv for the int8 tiler (codes are exact, inv=1) */
    float *qf32 = malloc((size_t)N * K * sizeof(float));
    float *inv  = malloc((size_t)N * sizeof(float));
    /* NF4: a per-tensor int8 LUT = round(level*127), and an index scratch (0..15) to inflate through it */
    int8_t nf4_lut[16]; uint8_t *qidx = NULL;
    if (nf4) { for (int i = 0; i < 16; i++) nf4_lut[i] = (int8_t)lrintf(ORK_NF4_LEVELS[i]*127.0f);
               qidx = malloc((size_t)N * K); }
    /* imatrix path: reused per-channel dequant scratch[K] for the clip-grid search (NULL imatrix => unused) */
    float *imdq = imatrix ? malloc((size_t)K * sizeof(float)) : NULL;
    if (!w->Bi4 || !w->bscale || !qf32 || !inv || (nf4 && !qidx) || (imatrix && !imdq)) {
        free(qf32); free(inv); free(qidx); free(imdq); ork_w_free(w); return NULL; }
    for (int n = 0; n < N; n++) {
        const float *fr = f32 + (size_t)n*K; float mx = 1e-9f; int k = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        float32x4_t vmx = vdupq_n_f32(1e-9f);
        for (; k <= K-4; k += 4) vmx = vmaxq_f32(vmx, vabsq_f32(vld1q_f32(fr + k)));
        float m[4]; vst1q_f32(m, vmx); float a=m[0]>m[1]?m[0]:m[1], bb=m[2]>m[3]?m[2]:m[3]; mx=a>bb?a:bb;
#endif
        for (; k < K; k++) { float v = fabsf(fr[k]); if (v > mx) mx = v; }
        if (imatrix) mx = wq_best_absmax(fr, K, mx, nf4, imatrix, imdq);  /* clip-grid scale selection */
        uint8_t *nib = w->Bi4 + (size_t)n*(K/2);
        if (nf4) { w->bscale[n] = mx / 127.0f;         /* int8 LUT range +-127 */
                   quant_chan_nf4(fr, K, mx, sr, &seed, nib, qidx + (size_t)n*K); }
        else     { float scale = mx / 7.0f;            /* int4 range +-7 (NOT 127) */
                   w->bscale[n] = scale;
                   quant_chan_i4(fr, K, scale, sr, &seed, nib, qf32 + (size_t)n*K); }
        if (bscale_out) bscale_out[n] = w->bscale[n];  /* back-compat: caller's out array (optional; w->bscale is canonical) */
        inv[n] = 1.0f;                                 /* qf32 holds exact codes; no rescale */
    }
    /* inflate the compact nibble store -> int8 f32 codes (validates the pack/inflate round-trip is what we tile) */
    if (nf4) for (int n = 0; n < N; n++) inflate_chan_nf4_f32(qidx + (size_t)n*K, K, nf4_lut, qf32 + (size_t)n*K);
    else     for (int n = 0; n < N; n++) expand_chan_i4_f32(w->Bi4 + (size_t)n*(K/2), K, qf32 + (size_t)n*K);
    tile_f32_i8(c, w, K, N, qf32, inv);                /* REUSE the int8 DMA/tiling path (no dup) */
    free(qf32); free(inv); free(qidx); free(imdq);
    return w;
}
/* ---- COMPACT int4 PERSIST (.orkpack streaming form) ----
 * Unlike ork_w_dump (which serializes the EXPANDED int8 tile bytes, ~K*N), this dumps the COMPACT int4
 * nibble store (~K*N/2) + per-channel scales — about half the size — and ork_mm_load_i4a8 re-inflates the
 * nibbles -> int8 and re-tiles on load (the tail of the pack path, but from stored nibbles, not f32). The
 * blob is self-contained: the NF4 LUT is NOT stored (it's derived from quant_kind). */
#define ORK_I4A8_MAGIC  0x4F344E31u           /* 'O','4','N','1' */
#define ORK_I4A8_VER    ork_pack_format_version()  /* int4 blob compat = library MAJOR (see ork_npu.h) */
struct ork_i4a8_hdr { uint32_t magic, version; int32_t K, N; uint32_t quant_kind; };
/* Serialize the compact int4 form: header + bscale[N] (f32) + Bi4 (K*N/2 bytes). out=NULL -> required
 * size. Returns 0 if `w` is not an int4-packed weight (no Bi4/bscale) or on cap overflow. */
size_t ork_w_dump_i4a8(const ork_w *w, void *out, size_t cap){
    if(!w || !w->Bi4 || !w->bscale) return 0;
    size_t hdr=sizeof(struct ork_i4a8_hdr), sc=(size_t)w->N*sizeof(float), nib=(size_t)w->K*w->N/2;
    size_t need=hdr+sc+nib;
    if(!out) return need;
    if(cap<need) return 0;
    struct ork_i4a8_hdr h={ORK_I4A8_MAGIC, ORK_I4A8_VER, w->K, w->N, w->quant_kind};
    char *p=out;
    memcpy(p,&h,hdr); p+=hdr;
    memcpy(p,w->bscale,sc); p+=sc;
    memcpy(p,w->Bi4,nib);
    return need;
}
/* Reload the compact int4 form straight into NPU DMA: parse+validate header, read bscale + Bi4, inflate
 * each channel's nibbles -> int8 (UNIFORM sign-extend / NF4 LUT per quant_kind) and tile_f32_i8 into a
 * fresh DMA buffer — the tail of the pack path, from stored nibbles instead of re-quantized f32. Retains
 * a copy of Bi4 + bscale so the loaded weight can be re-dumped byte-identically. NULL on malformed blob. */
ork_w *ork_mm_load_i4a8(ork_npu *c, int K, int N, const void *blob, size_t n){
    if(K%32 || N%32) return NULL;
    size_t hdr=sizeof(struct ork_i4a8_hdr), sc=(size_t)N*sizeof(float), nib=(size_t)K*N/2;
    if(n != hdr+sc+nib) return NULL;
    const char *p=blob;
    struct ork_i4a8_hdr h; memcpy(&h,p,hdr); p+=hdr;
    if(h.magic!=ORK_I4A8_MAGIC || h.version!=ORK_I4A8_VER || h.K!=K || h.N!=N) return NULL;
    if(h.quant_kind!=ORK_QK_UNIFORM && h.quant_kind!=ORK_QK_CODEBOOK_NF4) return NULL;
    int KS=1024, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    ork_w *w=calloc(1,sizeof *w); if(!w) return NULL;
    w->K=K; w->N=N; w->Sk=Sk; w->Sn=Sn; w->dtype=DT_I8; w->owns=1; w->domain=ork_dom(c->pack_domain); w->quant_kind=(uint8_t)h.quant_kind;
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    if(!w->Bb){ free(w); return NULL; }
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;
        struct buf *b=&w->Bb[(size_t)ns*Sk+ks]; *b=bcreate(c->fd,(size_t)Kp*Nc,0x403,w->domain);
        if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; } } }
    if(K<=10752){ w->Bf=calloc(Sn,sizeof(struct buf)); int ok=1;
        for(int ns=0;ns<Sn && ok;ns++){ int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
            struct buf *b=&w->Bf[ns]; *b=bcreate(c->fd,(size_t)K*Nc,0x403,w->domain); if(!b->cpu) ok=0; }
        if(!ok){ for(int ns=0;ns<Sn;ns++) bdestroy(c->fd,&w->Bf[ns]); free(w->Bf); w->Bf=NULL; } }
    /* retain the compact store + scales so the loaded weight re-dumps byte-identically */
    w->Bi4_bytes=nib; w->Bi4=malloc(nib); w->bscale=malloc(sc);
    if(!w->Bi4 || !w->bscale){ ork_mm_free(c,w); return NULL; }
    memcpy(w->bscale,p,sc); p+=sc;
    memcpy(w->Bi4,p,nib);
    /* ORK_DIRECT_I4: inflate nibbles STRAIGHT to int8-tiled (1 byte/elem scratch, no f32 round-trip,
     * no re-quant) — bit-identical to the f32 path. Default off; preserves the f32 path for review. */
    if(getenv("ORK_DIRECT_I4")){
        int8_t *i8=malloc((size_t)N*K);
        if(!i8){ ork_mm_free(c,w); return NULL; }
        tile_direct_i4_i8(c, w, K, N, w->quant_kind, i8);
        free(i8);
        return w;
    }
    float *qf32=malloc((size_t)N*K*sizeof(float)), *inv=malloc((size_t)N*sizeof(float));
    if(!qf32 || !inv){ free(qf32); free(inv); ork_mm_free(c,w); return NULL; }
    int8_t nf4_lut[16];
    if(w->quant_kind==ORK_QK_CODEBOOK_NF4) for(int i=0;i<16;i++) nf4_lut[i]=(int8_t)lrintf(ORK_NF4_LEVELS[i]*127.0f);
    for(int nn=0;nn<N;nn++){
        if(w->quant_kind==ORK_QK_CODEBOOK_NF4){
            /* NF4 store keeps the 0..15 index in the nibble; inflate via the int8 LUT */
            const uint8_t *nibp=w->Bi4+(size_t)nn*(K/2); float *qf=qf32+(size_t)nn*K;
            for(int k=0;k<K;k++){ uint8_t idx=(k&1)?(nibp[k>>1]>>4):(nibp[k>>1]&0xf); qf[k]=(float)nf4_lut[idx]; }
        } else expand_chan_i4_f32(w->Bi4+(size_t)nn*(K/2), K, qf32+(size_t)nn*K);
        inv[nn]=1.0f;                                  /* qf32 holds exact codes; no rescale */
    }
    tile_f32_i8(c, w, K, N, qf32, inv);                /* REUSE the int8 DMA/tiling path (no dup) */
    free(qf32); free(inv);
    return w;
}
/* Rearrange linear int8 codes i8[N][K] into IMPORTED (dma-buf) tiles, using the dma-buf's OWN cache sync
 * (the rknpu MEM_SYNC does NOT cover foreign imports). Same byte math as tile_i8_to_tiles. */
static void tile_i8_to_import_tiles(ork_npu *c, ork_w *w, int K, int N, const int8_t *i8){
    int KS=1024, NMAX=c->soc->nmax, Sk=w->Sk, Sn=w->Sn;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; if(!b->cpu)continue; int8_t*bb=b->cpu;
        dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
        for(int nt=0;nt<NN;nt++)for(int nl=0;nl<32;nl++){ int n=n0+nt*32+nl; const int8_t*src=i8+(size_t)n*K+k0;
            for(int kt=0;kt<KT;kt++) memcpy(bb+((size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32),src+kt*32,32); }
        dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
    if(w->Bf && K<=10752){ int KTf=K/32;
        for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
            struct buf*b=&w->Bf[ns]; if(!b->cpu)continue; int8_t*bb=b->cpu;
            dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
            for(int nt=0;nt<NN;nt++)for(int nl=0;nl<32;nl++){ int n=n0+nt*32+nl; const int8_t*src=i8+(size_t)n*K;
                for(int kt=0;kt<KTf;kt++) memcpy(bb+((size_t)nt*KTf*32*32+(size_t)kt*32*32+nl*32),src+kt*32,32); }
            dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
}
/* Zero-copy IMPORT variant of ork_mm_load_i4a8: resident tiles are dma-bufs the NPU reads in place (PRIME
 * import), and the int4 nibbles inflate -> int8 directly into them (no f32 round-trip). Bit-identical to
 * ork_mm_load_i4a8 (same blob, same tiled bytes). Falls through to NULL (caller uses ork_mm_load_i4a8) if
 * import is unavailable. Retains Bi4 + bscale so the loaded weight re-dumps byte-identically. */
ork_w *ork_mm_load_i4a8_import(ork_npu *c, int K, int N, const void *blob, size_t n){
    if(K%32 || N%32 || dmaheap_open()<0) return NULL;
    size_t hdr=sizeof(struct ork_i4a8_hdr), sc=(size_t)N*sizeof(float), nib=(size_t)K*N/2;
    if(n != hdr+sc+nib) return NULL;
    const char *p=blob;
    struct ork_i4a8_hdr h; memcpy(&h,p,hdr); p+=hdr;
    if(h.magic!=ORK_I4A8_MAGIC || h.version!=ORK_I4A8_VER || h.K!=K || h.N!=N) return NULL;
    if(h.quant_kind!=ORK_QK_UNIFORM && h.quant_kind!=ORK_QK_CODEBOOK_NF4) return NULL;
    int KS=1024, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    ork_w *w=calloc(1,sizeof *w); if(!w) return NULL;
    w->K=K; w->N=N; w->Sk=Sk; w->Sn=Sn; w->dtype=DT_I8; w->owns=1; w->domain=ork_dom(c->pack_domain); w->quant_kind=(uint8_t)h.quant_kind;
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf)); if(!w->Bb){ free(w); return NULL; }
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;(void)n0;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=bimport(c->fd,(size_t)Kp*Nc,w->domain);
        if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; } } }
    if(K%512==0 && K<=4096){ w->Bf=calloc(Sn,sizeof(struct buf)); int ok=1;
        for(int ns=0;ns<Sn && ok;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
            struct buf*b=&w->Bf[ns]; *b=bimport(c->fd,(size_t)K*Nc,w->domain); if(!b->cpu) ok=0; }
        if(!ok){ for(int ns=0;ns<Sn;ns++) bdestroy(c->fd,&w->Bf[ns]); free(w->Bf); w->Bf=NULL; } }
    w->Bi4_bytes=nib; w->Bi4=malloc(nib); w->bscale=malloc(sc);
    if(!w->Bi4 || !w->bscale){ ork_mm_free(c,w); return NULL; }
    memcpy(w->bscale,p,sc); p+=sc; memcpy(w->Bi4,p,nib);
    int8_t *i8=malloc((size_t)N*K); if(!i8){ ork_mm_free(c,w); return NULL; }
    if(w->quant_kind==ORK_QK_CODEBOOK_NF4){ int8_t lut[16]; for(int i=0;i<16;i++) lut[i]=(int8_t)lrintf(ORK_NF4_LEVELS[i]*127.0f);
        for(int nn=0;nn<N;nn++) inflate_chan_nf4_i8(w->Bi4+(size_t)nn*(K/2),K,lut,i8+(size_t)nn*K);
    } else for(int nn=0;nn<N;nn++) expand_chan_i4_i8(w->Bi4+(size_t)nn*(K/2),K,i8+(size_t)nn*K);
    tile_i8_to_import_tiles(c,w,K,N,i8);
    free(i8);
    return w;
}
/* ---- DIAGNOSTIC ONLY (tools/prefetch_headroom.c): isolate the STEADY-STATE per-slice streaming prep.
 * These re-run the TAIL of the int4 pack path (inflate stored nibbles -> int8 codes; tile into the
 * ALREADY-ALLOCATED resident DMA buffers) on an int4-packed weight, with NO bcreate/alloc — exactly the
 * work a streaming double-buffer would do per cycled slice. They do not alter pack/run behavior. */
/* inflate w->Bi4 (all N channels) -> int8 codes as f32 in caller scratch qf32[N*K] (UNIFORM sign-extend
 * / NF4 LUT per quant_kind). Mirrors the inflate loop in pack_i4a8 / load_i4a8. */
/* force the inflate KIND (lets the bench time UNIFORM and NF4 on the same nibble store; the inflate
 * cost is data-independent, so it's a valid per-path microbench either way). */
void ork_slice_inflate_i4a8_kind(const ork_w *w, float *qf32, int kind) {
    if (!w || !w->Bi4) return;
    int K = w->K, N = w->N;
    if (kind == ORK_QK_CODEBOOK_NF4) {
        int8_t lut[16]; for (int i = 0; i < 16; i++) lut[i] = (int8_t)lrintf(ORK_NF4_LEVELS[i]*127.0f);
        for (int nn = 0; nn < N; nn++) {
            const uint8_t *nibp = w->Bi4 + (size_t)nn*(K/2); float *qf = qf32 + (size_t)nn*K;
            for (int k = 0; k < K; k++) { uint8_t idx = (k&1) ? (nibp[k>>1]>>4) : (nibp[k>>1]&0xf); qf[k] = (float)lut[idx]; }
        }
    } else {
        for (int nn = 0; nn < N; nn++) expand_chan_i4_f32(w->Bi4 + (size_t)nn*(K/2), K, qf32 + (size_t)nn*K);
    }
}
void ork_slice_inflate_i4a8(const ork_w *w, float *qf32) { ork_slice_inflate_i4a8_kind(w, qf32, w ? w->quant_kind : 0); }
/* tile inflated codes qf32[N*K] into w's existing resident DMA buffers (inv=1; codes are exact). Reuses
 * the production tile_f32_i8 — same memcpy/quant + bsync(TO_DEVICE) the steady-state stream would issue. */
void ork_slice_tile_i8(ork_npu *c, ork_w *w, const float *qf32, float *inv1) {
    if (!w) return;
    tile_f32_i8(c, w, w->K, w->N, qf32, inv1);
}
/* DIRECT path microbench: inflate w's nibbles STRAIGHT to int8-tiled (no f32, no re-quant) into the
 * resident DMA tiles. i8scratch is caller-provided (size N*K); kind forces UNIFORM/NF4. Bit-identical
 * to ork_slice_inflate_i4a8_kind + ork_slice_tile_i8, but in one pass with no float round-trip. */
void ork_slice_direct_i4a8_kind(ork_npu *c, ork_w *w, int8_t *i8scratch, int kind) {
    if (!w || !w->Bi4) return;
    tile_direct_i4_i8(c, w, w->K, w->N, kind, i8scratch);
}
/* DIRECT inflate ONLY (nibble -> linear int8 i8[N*K]); the rearrange/bsync is the separate tile step.
 * Lets the bench split direct inflate cost from the tile+bsync cost. */
void ork_slice_direct_inflate_i8(const ork_w *w, int8_t *i8, int kind) {
    if (!w || !w->Bi4) return;
    int K = w->K, N = w->N;
    if (kind == ORK_QK_CODEBOOK_NF4) {
        int8_t lut[16]; for (int i = 0; i < 16; i++) lut[i] = (int8_t)lrintf(ORK_NF4_LEVELS[i]*127.0f);
        for (int n = 0; n < N; n++) inflate_chan_nf4_i8(w->Bi4 + (size_t)n*(K/2), K, lut, i8 + (size_t)n*K);
    } else {
        for (int n = 0; n < N; n++) expand_chan_i4_i8(w->Bi4 + (size_t)n*(K/2), K, i8 + (size_t)n*K);
    }
}

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
struct ork_stage {
    int K, N, Sk, Sn;
    struct buf *Bb;            /* Sk*Sn tile dma-bufs (bare: cpu/heap_fd set; dma/handle 0 until mapped) */
    struct buf *Bf;            /* Sn full-K dma-bufs (NULL if outside the Bf envelope) */
    int8_t *i8scratch;         /* reused N*K linear-int8 inflate scratch */
    ork_w view;                /* ork_w that points Bb/Bf at this slot's bufs once mapped (run target) */
    int mapped;
};
/* bare DMA-heap dma-buf: alloc + mmap, NO PRIME/MEM_CREATE (no IOVA yet). heap_fd = dma-buf fd. */
static struct buf bstage_alloc(size_t size){
    int hf=dmaheap_open(); if(hf<0) return (struct buf){0};
    size_t sz=pgup(size);
    struct dma_heap_allocation_data a; memset(&a,0,sizeof a); a.len=sz; a.fd_flags=O_RDWR|O_CLOEXEC;
    if(ioctl(hf,DMA_HEAP_IOCTL_ALLOC,&a)){ perror("DMA_HEAP_ALLOC(stage)"); return (struct buf){0}; }
    int dbuf=(int)a.fd;
    void*p=mmap(NULL,sz,PROT_READ|PROT_WRITE,MAP_SHARED,dbuf,0);
    if(p==MAP_FAILED){ perror("mmap(stage)"); close(dbuf); return (struct buf){0}; }
    struct buf b; memset(&b,0,sizeof b); b.cpu=p; b.size=sz; b.heap_fd=dbuf; return b;
}
/* IOMMU-map an already-allocated bare dma-buf (sets dma/obj/handle). 0 ok / -1 fail. */
static int bstage_map(int fd, struct buf*b){
    struct drm_prime_handle ph; memset(&ph,0,sizeof ph); ph.fd=b->heap_fd; ph.flags=0;
    if(ioctl(fd,DRM_IOCTL_PRIME_FD_TO_HANDLE,&ph)){ perror("PRIME(stage)"); return -1; }
    struct rknpu_mem_create mc; memset(&mc,0,sizeof mc); mc.handle=ph.handle; mc.flags=0; mc.size=0; mc.core_mask=RKNPU_CORE0_MASK;
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_CREATE,&mc)){ perror("MEM_CREATE(stage)"); return -1; }
    b->handle=mc.handle; b->dma=mc.dma_addr; b->obj=mc.obj_addr; return 0;
}
/* MEM_DESTROY the map (keep the dma-buf + mmap alive for recycle): clears dma/obj/handle only. */
static void bstage_unmap(int fd, struct buf*b){
    if(!b->obj && !b->handle) return;
    struct rknpu_mem_destroy d; memset(&d,0,sizeof d); d.handle=b->handle; d.obj_addr=b->obj; ioctl(fd,DRM_IOCTL_RKNPU_MEM_DESTROY,&d);
    b->dma=0; b->obj=0; b->handle=0;
}
static void bstage_free(struct buf*b){ if(!b->cpu) return; munmap(b->cpu,b->size); if(b->heap_fd>0) close(b->heap_fd); memset(b,0,sizeof *b); }

/* tile shape mirrors ork_mm_load_i4a8: KS=1024 K-split, NMAX N-split; Bf full-K when K%512==0 && K<=4096
 * (same envelope as load_i8_import). Returns NULL if dma-heap absent / alloc fails. */
struct ork_stage *ork_stage_create(ork_npu *c, int K, int N){
    if(K%32 || N%32 || dmaheap_open()<0) return NULL;
    int KS=1024, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    struct ork_stage *s=calloc(1,sizeof *s); if(!s) return NULL;
    s->K=K; s->N=N; s->Sk=Sk; s->Sn=Sn;
    s->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    s->i8scratch=malloc((size_t)N*K);
    if(!s->Bb || !s->i8scratch){ free(s->Bb); free(s->i8scratch); free(s); return NULL; }
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS;(void)n0;
        struct buf*b=&s->Bb[(size_t)ns*Sk+ks]; *b=bstage_alloc((size_t)Kp*Nc);
        if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) bstage_free(&s->Bb[i]); free(s->Bb); free(s->i8scratch); free(s); return NULL; }}}
    if(K%512==0 && K<=4096){ s->Bf=calloc(Sn,sizeof(struct buf)); int ok=1;
        for(int ns=0;ns<Sn && ok;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
            struct buf*b=&s->Bf[ns]; *b=bstage_alloc((size_t)K*Nc); if(!b->cpu) ok=0; }
        if(!ok){ for(int ns=0;ns<Sn;ns++) bstage_free(&s->Bf[ns]); free(s->Bf); s->Bf=NULL; } }
    return s;
}
/* FILL: inflate src's int4 nibble store -> int8, tile into this slot's BARE dma-bufs, clean caches.
 * src must be an int4-packed weight (ork_mm_pack_i4a8) with the same K,N. No IOVA needed (bare bufs).
 * This is the prefetchable CPU work — safe to call on a background thread (touches only this slot). */
void ork_stage_fill(ork_npu *c, struct ork_stage *s, const ork_w *src){
    if(!s || !src || !src->Bi4) return;
    int K=s->K, N=s->N, KS=1024, NMAX=c->soc->nmax, Sk=s->Sk, Sn=s->Sn, kind=src->quant_kind;
    int8_t *i8=s->i8scratch;
    if(kind==ORK_QK_CODEBOOK_NF4){ int8_t lut[16]; for(int i=0;i<16;i++) lut[i]=(int8_t)lrintf(ORK_NF4_LEVELS[i]*127.0f);
        for(int n=0;n<N;n++) inflate_chan_nf4_i8(src->Bi4+(size_t)n*(K/2),K,lut,i8+(size_t)n*K);
    } else for(int n=0;n<N;n++) expand_chan_i4_i8(src->Bi4+(size_t)n*(K/2),K,i8+(size_t)n*K);
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
        struct buf*b=&s->Bb[(size_t)ns*Sk+ks]; int8_t*bb=b->cpu;
        dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
        for(int nt=0;nt<NN;nt++)for(int nl=0;nl<32;nl++){ int n=n0+nt*32+nl; const int8_t*sp=i8+(size_t)n*K+k0;
            for(int kt=0;kt<KT;kt++) memcpy(bb+((size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32),sp+kt*32,32); }
        dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
    if(s->Bf){ int KTf=K/32;
        for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
            struct buf*b=&s->Bf[ns]; int8_t*bb=b->cpu;
            dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
            for(int nt=0;nt<NN;nt++)for(int nl=0;nl<32;nl++){ int n=n0+nt*32+nl; const int8_t*sp=i8+(size_t)n*K;
                for(int kt=0;kt<KTf;kt++) memcpy(bb+((size_t)nt*KTf*32*32+(size_t)kt*32*32+nl*32),sp+kt*32,32); }
            dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
}
/* FILL from raw untiled int8 B[K][N] (row-major, K-major — as ggml-ork's per-channel quant produces):
 * tile directly into this slot's BARE dma-bufs (RAM-backed, NO IOVA / NO bcreate). The int8 counterpart
 * of ork_stage_fill (which inflates int4 first). Lets a caller add a freshly-quantized weight to the
 * stream pool WITHOUT the transient IOVA pack that would compete with the pool's mapped hot set. Tiling
 * is parallelized across all cores (ork_parallel_for + tile_i8_range) — same layout as pack()/load_i8. */
void ork_stage_fill_i8(ork_npu *c, struct ork_stage *s, const int8_t *B){
    if(!s || !B) return;
    int K=s->K, N=s->N, KS=1024, NMAX=c->soc->nmax, Sk=s->Sk, Sn=s->Sn;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
        struct buf*b=&s->Bb[(size_t)ns*Sk+ks];
        dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
        struct tile_i8_arg ta={(int8_t*)b->cpu,B,KT,k0,n0,N}; ork_parallel_for(NN,tile_i8_range,&ta);
        dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
    if(s->Bf){ int KTf=K/32;
        for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
            struct buf*b=&s->Bf[ns];
            dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
            struct tile_i8_arg ta={(int8_t*)b->cpu,B,KTf,0,n0,N}; ork_parallel_for(NN,tile_i8_range,&ta);
            dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
}
/* MAP: IOMMU-map every bare dma-buf in the slot and point the slot's ork_w view at them. 0 ok / -1. */
int ork_stage_map(ork_npu *c, struct ork_stage *s){
    if(!s || s->mapped) return s?0:-1;
    int ok=1;
    for(int i=0;i<s->Sk*s->Sn && ok;i++) if(bstage_map(c->fd,&s->Bb[i])) ok=0;
    if(ok && s->Bf) for(int ns=0;ns<s->Sn && ok;ns++) if(bstage_map(c->fd,&s->Bf[ns])) ok=0;
    if(!ok){   /* IOVA full mid-map — roll back the partial maps so a retry (after eviction) starts clean */
        for(int i=0;i<s->Sk*s->Sn;i++) bstage_unmap(c->fd,&s->Bb[i]);
        if(s->Bf) for(int ns=0;ns<s->Sn;ns++) bstage_unmap(c->fd,&s->Bf[ns]);
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
    for(int i=0;i<s->Sk*s->Sn;i++) bstage_unmap(c->fd,&s->Bb[i]);
    if(s->Bf) for(int ns=0;ns<s->Sn;ns++) bstage_unmap(c->fd,&s->Bf[ns]);
    s->mapped=0;
}
void ork_stage_free(ork_npu *c, struct ork_stage *s){
    if(!s) return; ork_stage_unmap(c,s);
    for(int i=0;i<s->Sk*s->Sn;i++) bstage_free(&s->Bb[i]);
    if(s->Bf){ for(int ns=0;ns<s->Sn;ns++) bstage_free(&s->Bf[ns]); free(s->Bf); }
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
struct ork_stream_entry { struct ork_stage *stg; int K, N; int mapped; uint64_t last_use; };
struct ork_stream_pool  { ork_npu *c; struct ork_stream_entry **e; int n, cap; uint64_t clock; };

struct ork_stream_pool *ork_stream_pool_create(ork_npu *c){
    if(!c || dmaheap_open()<0) return NULL;          /* import path unavailable -> caller falls back */
    struct ork_stream_pool *p=calloc(1,sizeof *p); if(!p) return NULL;
    p->c=c; p->cap=16; p->e=calloc(p->cap,sizeof*p->e); if(!p->e){ free(p); return NULL; }
    return p;
}
static struct ork_stream_entry *pool_new_entry(struct ork_stream_pool*p,int K,int N){
    struct ork_stage *stg=ork_stage_create(p->c,K,N); if(!stg) return NULL;
    struct ork_stream_entry *e=calloc(1,sizeof *e); if(!e){ ork_stage_free(p->c,stg); return NULL; }
    e->stg=stg; e->K=K; e->N=N;
    if(p->n>=p->cap){ int nc=p->cap*2; void*r=realloc(p->e,nc*sizeof*p->e); if(!r){ ork_stage_free(p->c,stg); free(e); return NULL; } p->e=r; p->cap=nc; }
    p->e[p->n++]=e; return e;
}
/* int4-stored: fill = inflate nibbles -> int8 + tile (the .orkpack i4a8 blob, ork_w_dump_i4a8). The fill
 * happens ONCE here (the expensive op, cached in RAM). NULL on import-unavailable / malformed blob. */
struct ork_stream_entry *ork_stream_pool_add_i4a8(struct ork_stream_pool *p, int K, int N, const void *blob, size_t n){
    if(!p) return NULL;
    /* validate + materialize an int4 source ork_w from the blob (host-side Bi4+bscale), inflate into the
     * entry's staging dma-buf, then drop the temporary source (we only needed its nibble store to fill). */
    ork_w *src=ork_mm_load_i4a8(p->c,K,N,blob,n);  /* allocates resident DMA too — temporary; freed below */
    if(!src) return NULL;
    struct ork_stream_entry *e=pool_new_entry(p,K,N);
    if(!e){ ork_mm_free(p->c,src); return NULL; }
    ork_stage_fill(p->c,e->stg,src);               /* the ONE-TIME inflate into RAM-resident staging */
    ork_mm_free(p->c,src);
    return e;
}
/* int8-stored: fill = copy the stored tile bytes (ork_w_dump blob) into the staging dma-bufs. Same blob
 * layout/validation as ork_mm_load_i8. NULL on import-unavailable / size mismatch. */
struct ork_stream_entry *ork_stream_pool_add_i8(struct ork_stream_pool *p, int K, int N, const void *blob, size_t n){
    if(!p || K%32 || N%32) return NULL;
    int KS=1024, NMAX=p->c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t need=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS;(void)n0; need+=pgup((size_t)Kp*Nc);}}
    if(n!=need) return NULL;
    struct ork_stream_entry *e=pool_new_entry(p,K,N); if(!e) return NULL;
    struct ork_stage *s=e->stg; size_t off=0;
    for(int i=0;i<s->Sk*s->Sn;i++){ struct buf*b=&s->Bb[i];
        dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
        memcpy(b->cpu,(const char*)blob+off,b->size); off+=b->size;
        dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE); }
    /* Bf full-K rebuild from the just-filled Bb tiles (same envelope as load_i8_import) */
    if(s->Bf){ int KTf=K/32;
        for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
            struct buf*bf=&s->Bf[ns]; int8_t*fb=bf->cpu;
            dmabuf_sync(bf->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
            for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
                const int8_t*sb=(const int8_t*)s->Bb[(size_t)ns*Sk+ks].cpu;
                for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++){
                    int ktf=(k0/32)+kt;
                    fb[(size_t)nt*KTf*32*32+(size_t)ktf*32*32+nl*32+kk]=
                        sb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]; }}
            dmabuf_sync(bf->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
    return e;
}
/* int8-RAW: tile a freshly-quantized UNTILED int8 B[K][N] straight into RAM staging (NO IOVA / NO
 * bcreate) — the zero-transient-pack add. The caller already quantized the weight; we tile it directly
 * into the pool instead of packing to IOVA first (which would compete with the pool's mapped hot set).
 * Map it later with ork_stream_pool_map. NULL on bad dims / import-unavailable. */
struct ork_stream_entry *ork_stream_pool_add_i8_raw(struct ork_stream_pool *p, int K, int N, const int8_t *B){
    if(!p || !B || K%32 || N%32) return NULL;
    struct ork_stream_entry *e=pool_new_entry(p,K,N); if(!e) return NULL;
    ork_stage_fill_i8(p->c,e->stg,B);
    return e;
}
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
 * the caller only sets the RAM budget (how many entries are held resident). 0 ok / -1 (OOM / bad args). */
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

int ork_mm_repack_i8_f32(ork_npu *c, ork_w *w, int K, int N, const float *f32, float *bscale_out) {
    if (!w || w->dtype != DT_I8 || !w->Bb) return -1;
    if (w->K != K || w->N != N) return -2;
    float *inv = malloc((size_t)N * sizeof(float)); if (!inv) return -1;
    chan_scales_f32(f32, K, N, inv, bscale_out);
    tile_f32_i8(c, w, K, N, f32, inv);
    free(inv);
    return 0;
}
/* ---- FUSED dequant->int8 pack/repack (callback per channel; NO full f32[N][K] buffer) ----
 * Materialize one channel at a time into a reused K-float scratch (stays in cache), then NEON quant+tile
 * it — avoids the DRAM round-trip of writing then re-reading a full f32[N][K], which dominates a Q4_K MoE
 * repack. Same int8/bscale result as feeding the equivalent f32 to tile_f32_i8. */
static int tile_dequant_i8(ork_npu *c, ork_w *w, int K, int N, ork_dequant_row_fn fn, void *dctx, float *bscale) {
    int KS = 1024, NMAX = c->soc->nmax, Sk = w->Sk, Sn = w->Sn, fd = c->fd, KTf = K/32;
    float *sc = malloc((size_t)K * sizeof(float)); if (!sc) return -1;
    for (int n = 0; n < N; n++) {
        fn(dctx, n, sc, K);                                   /* dequant channel n -> reused scratch[K] */
        float mx = 1e-9f; int k = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        float32x4_t vmx = vdupq_n_f32(1e-9f);
        for (; k <= K-4; k += 4) vmx = vmaxq_f32(vmx, vabsq_f32(vld1q_f32(sc + k)));
        float m[4]; vst1q_f32(m, vmx); float a=m[0]>m[1]?m[0]:m[1], b=m[2]>m[3]?m[2]:m[3]; mx=a>b?a:b;
#endif
        for (; k < K; k++) { float v = fabsf(sc[k]); if (v > mx) mx = v; }
        float iv = 127.0f/mx; bscale[n] = mx/127.0f;
        int ns = n/NMAX, nloc = n - ns*NMAX, nt = nloc/32, nl = nloc%32;
        for (int ks = 0; ks < Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS, KT = Kp/32;
            struct buf *b = &w->Bb[(size_t)ns*Sk+ks]; if (!b->cpu) continue; int8_t *bb = b->cpu;
            for (int kt = 0; kt < KT; kt++) quant32_f32_i8(bb + ((size_t)nt*KT*32*32 + (size_t)kt*32*32 + nl*32), sc + k0 + kt*32, iv);
        }
        if (w->Bf && K <= 10752) { struct buf *b = &w->Bf[ns]; if (b->cpu) { int8_t *bb = b->cpu;
            for (int kt = 0; kt < KTf; kt++) quant32_f32_i8(bb + ((size_t)nt*KTf*32*32 + (size_t)kt*32*32 + nl*32), sc + kt*32, iv); } }
    }
    free(sc);
    for (int i = 0; i < Sk*Sn; i++) { struct buf *b = &w->Bb[i]; if (b->cpu) { bsync(fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); bsync(fd,b,RKNPU_MEM_SYNC_TO_DEVICE); } }
    if (w->Bf) for (int ns = 0; ns < Sn; ns++) { struct buf *b = &w->Bf[ns]; if (b->cpu) { bsync(fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); bsync(fd,b,RKNPU_MEM_SYNC_TO_DEVICE); } }
    return 0;
}
ork_w *ork_mm_pack_i8_dequant(ork_npu *c, int K, int N, ork_dequant_row_fn fn, void *dctx, float *bscale_out) {
    if (K % 32 || N % 32) return NULL;
    int KS = 1024, NMAX = c->soc->nmax, Sk = (K+KS-1)/KS, Sn = (N+NMAX-1)/NMAX;
    ork_w *w = calloc(1, sizeof *w); if (!w) return NULL;
    w->K = K; w->N = N; w->Sk = Sk; w->Sn = Sn; w->dtype = DT_I8; w->Bb = calloc((size_t)Sk*Sn, sizeof(struct buf));
    if (!w->Bb) { free(w); return NULL; }
    for (int ns = 0; ns < Sn; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX;
      for (int ks = 0; ks < Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS;
        struct buf *b = &w->Bb[(size_t)ns*Sk+ks]; *b = bcreate(c->fd, (size_t)Kp*Nc, 0x403, w->domain);
        if (!b->cpu) { for (int i = 0; i < ns*Sk+ks; i++) bdestroy(c->fd, &w->Bb[i]); free(w->Bb); free(w); return NULL; } } }
    if (K <= 10752) { w->Bf = calloc(Sn, sizeof(struct buf)); int ok = 1;
        for (int ns = 0; ns < Sn && ok; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX;
            struct buf *b = &w->Bf[ns]; *b = bcreate(c->fd, (size_t)K*Nc, 0x403, w->domain); if (!b->cpu) ok = 0; }
        if (!ok) { for (int ns = 0; ns < Sn; ns++) bdestroy(c->fd, &w->Bf[ns]); free(w->Bf); w->Bf = NULL; } }
    if (tile_dequant_i8(c, w, K, N, fn, dctx, bscale_out) != 0) { ork_w_free(w); return NULL; }
    return w;
}
int ork_mm_repack_i8_dequant(ork_npu *c, ork_w *w, int K, int N, ork_dequant_row_fn fn, void *dctx, float *bscale_out) {
    if (!w || w->dtype != DT_I8 || !w->Bb) return -1;
    if (w->K != K || w->N != N) return -2;
    return tile_dequant_i8(c, w, K, N, fn, dctx, bscale_out);
}
void ork_w_free(ork_w *w){ if(!w)return; free(w->Bb); free(w->Bf); free(w->Bi4); free(w->bscale); free(w); }   /* device buffers freed at ctx teardown */
/* Free a packed weight AND reclaim its NPU DMA/IOVA. Required for layer-streaming: evicted weights must
 * return their IOVA to the 4 GiB window (rk_iommu is 32-bit — see the wiki / npu-iova cap). Only weights
 * that OWN their buffers (per-tile bcreate: pack / pack_i4 / pack_i8) are reclaimed; weights whose tiles
 * are VIEWS into a single dedicated buffer (grouped-i4, own_buf_valid=1) reclaim that one buffer. */
void ork_mm_free(ork_npu *c, ork_w *w){
    if(!w) return;
    if(c && w->owns){
        size_t nb=(size_t)w->Sk*w->Sn;
        if(w->Bb) for(size_t i=0;i<nb;i++) if(w->Bb[i].cpu) bdestroy(c->fd,&w->Bb[i]);
    }
    /* Bf is ALWAYS its own per-N-slice bcreate/bimport (never a view), even when Bb is consolidated into
     * own_buf — so reclaim it whenever present, independent of owns. */
    if(c && w->Bf) for(int i=0;i<w->Sn;i++) if(w->Bf[i].cpu) bdestroy(c->fd,&w->Bf[i]);
    /* dedicated single-buffer weights (grouped-i4, or consolidated int8): Bb[] entries are VIEWS (share
     * own_buf's handle/obj) — destroy the one backing buffer ONLY, never the views (double-free / munmap
     * of a sub-pointer). Reclaims IOVA. */
    if(c && w->own_buf_valid) bdestroy(c->fd,&w->own_buf);
    free(w->Bb); free(w->Bf); free(w->Bi4); free(w->bscale); free(w);
}
/* Resident NPU bytes a packed weight occupies (Bb tiles + optional full-K Bf) — for a streaming cache
 * to budget the 4 GiB IOVA window and decide when to evict. */
size_t ork_w_bytes(const ork_w *w){
    if(!w) return 0; size_t t=0;
    if(w->Bb) for(size_t i=0;i<(size_t)w->Sk*w->Sn;i++) t+=w->Bb[i].size;
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
#define ORK_I4_KS 10752       /* int4 single-submit K ceiling (validated == int8's) */
/* an Nc-wide x Kp-row slice of B[K][N] at (k0,n0) -> native (Nc/64,Kp/32,64,32) int4 (2/byte).
 * Nc%64; validated single-submit up to N=8192 (SoC nmax). */
static void tile_i4_Bslice(uint8_t*dst,const int8_t*B,int K,int N,int k0,int Kp,int n0,int Nc){
    int KT=Kp/32, NB=Nc/64; memset(dst,0,(size_t)Kp*Nc/2);
    for(int nb=0;nb<NB;nb++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<64;nl++)for(int kk=0;kk<32;kk++){
        size_t idx=(((size_t)nb*KT+kt)*64+nl)*32+kk;
        dst[idx/2]|= (uint8_t)(B[(size_t)(k0+kt*32+kk)*N+(n0+nb*64+nl)]&0xf) << ((idx&1)?4:0);
    }
}
/* a Kp-slice of one A row -> native (Kp/32,1,32) int4 */
static void tile_i4_Aslice(uint8_t*dst,const int8_t*Arow,int k0,int Kp){
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int col = 0;
    int8x16_t vmask = vdupq_n_s8(0x0f);
    for (; col <= Kp - 16; col += 16) {
        int8x16_t v = vld1q_s8(&Arow[k0 + col]);
        int8x16_t vmasked = vandq_s8(v, vmask);
        int8x16x2_t tuzp = vuzpq_s8(vmasked, vmasked);
        uint8x8_t veven_low = vreinterpret_u8_s8(vget_low_s8(tuzp.val[0]));
        uint8x8_t vodd_low  = vreinterpret_u8_s8(vget_low_s8(tuzp.val[1]));
        uint8x8_t vodd_shifted = vshl_n_u8(vodd_low, 4);
        uint8x8_t vcombined = vorr_u8(veven_low, vodd_shifted);
        vst1_u8(&dst[col / 2], vcombined);
    }
#else
    int KT=Kp/32; memset(dst,0,(size_t)Kp/2);
    for(int kt=0;kt<KT;kt++)for(int kk=0;kk<32;kk++){
        size_t idx=(size_t)kt*32+kk;
        dst[idx/2]|= (uint8_t)(Arow[k0+kt*32+kk]&0xf) << ((idx&1)?4:0);
    }
#endif
}
/* a Kp-slice of M activation rows -> native (Kp/32,M,32) interleaved int4 */
static void tile_i4_Aslice_mm(uint8_t*dst,const int8_t*A,int M,int K,int k0,int Kp){
    int KT=Kp/32; memset(dst,0,(size_t)M*Kp/2);
    for(int kt=0;kt<KT;kt++)for(int m=0;m<M;m++)for(int kk=0;kk<32;kk++){
        size_t idx=((size_t)kt*M+m)*32+kk; uint8_t v=(uint8_t)(A[(size_t)m*K+k0+kt*32+kk]&0xf);
        dst[idx/2]|= (idx&1)?(v<<4):v;
    }
}
ork_w *ork_mm_pack_i4(ork_npu *c,int K,int N,const int8_t *B){
    if(K%32||N%64) return NULL;
    int KS=ORK_I4_KS, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;  /* wide N-slices ≤ nmax */
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I4; w->owns=1; w->domain=ork_dom(c->pack_domain);
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    for(int ns=0;ns<Sn;ns++)for(int ks=0;ks<Sk;ks++){
        int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=bcreate(c->fd,(size_t)Kp*Nc/2,0x403,w->domain);
        if(!b->cpu){
            fprintf(stderr,"[ork] ERROR: bcreate failed to allocate weight buffer Bb[%zu] in pack_i4 (size=%zu)\n",(size_t)ns*Sk+ks,(size_t)Kp*Nc/2);
            for(int i=0;i<ns*Sk+ks;i++) bdestroy(c->fd,&w->Bb[i]);
            ork_w_free(w); return NULL;
        }
        tile_i4_Bslice(b->cpu,B,K,N,k0,Kp,n0,Nc);
        bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);
    }
    return w;
}
/* grouped pack: K split into groups of G (each its own resident slice) for per-group scales. G%32,
 * K%G, G<=10752. Sk = K/G groups; run_i4_grouped scales each group's partial before accumulating. */
ork_w *ork_mm_pack_i4_grouped(ork_npu *c,int K,int N,const int8_t *B,int G){
    if(K%32||N%64||G%32||K%G||G>ORK_I4_KS) return NULL;
    int NMAX=c->soc->nmax, Sk=K/G, Sn=(N+NMAX-1)/NMAX;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I4;w->gsize=G; w->domain=ork_dom(c->pack_domain);
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    /* Reserve the whole weight as ONE dedicated DMA buffer (own_buf); each group-tile is a 4KB-aligned
     * VIEW into it (shared obj, dma=own_buf.dma+off). Collapses the per-group bcreate storm to a single
     * allocation => fast warmup, no IOVA-handle OOM, and — crucially — RECLAIMABLE: ork_mm_free destroys
     * own_buf (returning its IOVA to the 4 GiB window), so drop/reload of a grouped weight does NOT leak
     * (streaming / MoE-swap). The whole region is flushed to device in a single bsync. Falls back to
     * per-tile owning bcreate (also reclaimable) if the dedicated alloc fails. */
    size_t wtotal=0;
    for(int ns=0;ns<Sn;ns++)for(int g=0;g<Sk;g++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
        wtotal += (((size_t)G*Nc/2)+4095u)&~(size_t)4095u; }
    struct buf own=bcreate(c->fd,wtotal,0x403,w->domain);
    if(own.cpu){
        w->own_buf=own; w->own_buf_valid=1;
        size_t off=0;
        for(int ns=0;ns<Sn;ns++)for(int g=0;g<Sk;g++){
            int k0=g*G,n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX; size_t ts=(size_t)G*Nc/2;
            struct buf*b=&w->Bb[(size_t)ns*Sk+g];
            b->handle=own.handle; b->obj=own.obj; b->dma=own.dma+off; b->cpu=(char*)own.cpu+off; b->size=ts;
            tile_i4_Bslice(b->cpu,B,K,N,k0,G,n0,Nc);
            off += (ts+4095u)&~(size_t)4095u;
        }
        bsync_off(c->fd,own.obj,0,wtotal,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        bsync_off(c->fd,own.obj,0,wtotal,RKNPU_MEM_SYNC_TO_DEVICE);
    } else {
        w->owns=1;   /* per-tile owning bcreate: reclaimable by ork_mm_free */
        for(int ns=0;ns<Sn;ns++)for(int g=0;g<Sk;g++){
            int k0=g*G,n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
            struct buf*b=&w->Bb[(size_t)ns*Sk+g]; *b=bcreate(c->fd,(size_t)G*Nc/2,0x403,w->domain);
            if(!b->cpu){ fprintf(stderr,"[ork] ERROR: weight alloc failed (G=%d Nc=%d) in pack_i4_grouped\n",G,Nc);
                for(int i=0;i<ns*Sk+g;i++) bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; }
            tile_i4_Bslice(b->cpu,B,K,N,k0,G,n0,Nc);
            bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);
        }
    }
    return w;
}

ork_w *ork_mm_pack_i4_to_i8(ork_npu *c, int K, int N, const int8_t *B) {
    /* The core fallback: it takes int4-range values ([-8,7]) unpacked in int8_t containers,
     * but physically packs them into the highly optimized int8 resident weight layout.
     * This simply defers to ork_mm_pack_i8, yielding maximum int8 silicon execution speed
     * while the caller (e.g. ggml-ork) maintains the 50% footprint reduction on disk. */
    return ork_mm_pack_i8(c, K, N, B);
}
static int run_i4_mc(ork_npu *c,ork_w *w,int M,const int8_t *A,int32_t *C,int nc);  /* defined below */
int ork_mm_run_i4(ork_npu *c,ork_w *w,int M,const int8_t *A,int32_t *C){
    if(!w||w->dtype!=DT_I4) return -1;
    if(check_overlap("ork_mm_run_i4", (uintptr_t)A, (uintptr_t)A + (size_t)M * w->K, (uintptr_t)C, (uintptr_t)C + (size_t)M * w->N * 4)) return -1;
    int NB=w->N/64;                            /* total 64-wide N-blocks (column-split granularity) */
    int nc=budget(c, M); if(nc>NB)nc=NB; if(nc<1)nc=1;   /* ≥1 N-block/core; nc==1 = serial */
    if(!g_ork_prof) return run_i4_mc(c,w,M,A,C,nc);
    double t0=ork_now_us(); int r=run_i4_mc(c,w,M,A,C,nc); g_prof_i4_us+=ork_now_us()-t0; g_prof_i4_calls++; return r;
}

/* C[M,N] = A[M,K] x packed weights. dt-keyed: fp16 A -> fp32 C, or int8 A -> int32 C.
 * int8 uses 2x the rows budget, K-slice 1024, and effective-K/2 schedule (see synth_i8). */
/* one matmul submit with cold-start warmup; regcmd must already be staged in c->regcmd.
 * core_mask=1<<core selects a single NPU core (0x1/0x2/0x4 = core 0/1/2 — exactly what librkllmrt
 * round-robins). ALL THREE subcore_task[] must be populated even for a single core: leaving the
 * non-target entries zero NULL-derefs rknpu_job_subcore_commit (the earlier kernel Oops). */
static int submit1(ork_npu *c){
    int fd=c->fd;
    static int tc=-2; if(tc==-2){const char*e=getenv("ORK_NPU_TESTCORE"); tc=e?atoi(e):0; if(tc<0||tc>2)tc=0;}
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.fence_fd=-1;
    sub.core_mask=1u<<tc;
    sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,1};
    /* first submit on a fresh output buffer returns stale (NPU primed against wedging by the
     * RKNPU_ACT_RESET); run one throwaway warmup with a short timeout, then the real submit. */
    int reps=c->warmed?1:2;
    for(int rep=0;rep<reps;rep++){ int last=(rep==reps-1); sub.timeout=60000;
        if(rknpu_submit_ioctl(fd,&sub,c->dom_active)){ if(last){perror("SUBMIT");return -1;} continue; }
        bsync(fd,&c->Cc,RKNPU_MEM_SYNC_FROM_DEVICE); }
    c->warmed=1; return 0;
}
/* ---- multi-core (ORK_NPU_MC=<n>): use n cores (capped at soc->cores). Split each N-slice's
 * output tiles across the cores, run concurrently on per-core buffers, accumulate into disjoint
 * columns of cres (no lock). n is a *request* — the engine can pass any count up to soc->cores,
 * so this is dynamic, not hardwired to a chip's core total. ---- */
static int mc_ensure(ork_npu *c,int nc){
    int fd=c->fd;
    if(!c->mtk_all.cpu) {
        c->mtk_all=bcreate(fd, sizeof(struct rknpu_task) * ORK_MAXCORE, 0x40b, c->dom_active);
        if(!c->mtk_all.cpu) {
            fprintf(stderr, "[ork] ERROR: mc_ensure failed to allocate mtk_all task buffer (IOMMU full?)\n");
            return -1;
        }
    }
    for(int i=0;i<nc;i++){
        if(c->mrc[i].cpu) continue;        /* alloc once, per core, up to the max ever requested */
        c->mrc[i]=bcreate(fd,65536,0x403,c->dom_active); c->mtk[i]=bcreate(fd,65536,0x40b,c->dom_active); c->maf[i]=bcreate(fd,(size_t)4*32768*2,0x403,c->dom_active);
        if(!c->mrc[i].cpu||!c->mtk[i].cpu||!c->maf[i].cpu) {
            fprintf(stderr, "[ork] ERROR: mc_ensure failed to allocate multi-core buffers for core %d (IOMMU full?)\n", i);
            return -1;
        }
        struct rknpu_task t;memset(&t,0,sizeof t);t.enable_mask=0xd;t.int_mask=0x300;t.int_clear=0x1ffff;t.regcfg_amount=108;t.regcmd_addr=c->mrc[i].dma;
        memcpy(c->mtk[i].cpu,&t,sizeof t); bsync(fd,&c->mtk[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        struct rknpu_task *tall = (struct rknpu_task*)c->mtk_all.cpu;
        tall[i] = t;
    }
    int reg_amt = (c->last_dt == DT_I4) ? 116 : 108;
    struct rknpu_task *tall = (struct rknpu_task*)c->mtk_all.cpu;
    for(int i=0;i<nc;i++){
        struct rknpu_task *t = (struct rknpu_task*)c->mtk[i].cpu;
        if (t->regcfg_amount != reg_amt) {
            t->regcfg_amount = reg_amt;
            bsync(fd,&c->mtk[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        }
        if (tall[i].regcfg_amount != reg_amt) {
            tall[i].regcfg_amount = reg_amt;
        }
    }
    bsync(fd,&c->mtk_all,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    return 0;
}
static double ork_now_us(void);   /* fwd (defined below) */
/* ORK_MCPROF diagnostic: per-core phase timing inside mcworker's prefill (M>1) path —
 * copy (activation tile host-copy + bsync), submit (regcmd + ioctl + result bsync), acc
 * (host accumulate). Pins why large-M multi-core barely scales. Read via ork_npu_mc_timing. */
#define MCPROF_MAX 8
static double g_mc_copy[MCPROF_MAX], g_mc_sub[MCPROF_MAX], g_mc_acc[MCPROF_MAX]; static long g_mc_n[MCPROF_MAX];
static double g_mc_synth[MCPROF_MAX];   /* host regcmd-synth+bsync portion of g_mc_sub (the OVERLAPPABLE part; ioctl/NPU = sub-synth) */
void ork_npu_mc_reset(void){ for(int i=0;i<MCPROF_MAX;i++){g_mc_copy[i]=g_mc_sub[i]=g_mc_acc[i]=g_mc_synth[i]=0;g_mc_n[i]=0;} }
void ork_npu_mc_timing(int core,double*copy,double*sub,double*acc,long*n){
    if(copy)*copy=g_mc_copy[core]; if(sub)*sub=g_mc_sub[core]; if(acc)*acc=g_mc_acc[core]; if(n)*n=g_mc_n[core]; }
double ork_npu_mc_synth(int core){ return (core>=0&&core<MCPROF_MAX)?g_mc_synth[core]:0; }

struct mcw { ork_npu *c; int core, nc, dt, M; const void *A; ork_w *w; void *cres; int rc; int reps; size_t maxout; int chain_pref; int chain_ksplit; };

static void unified_ioctl(struct mcw *a, int i, int nc) {
    ork_npu *c = a->c; int fd = c->fd; int reps = a->reps; struct buf *CC = &c->mcc[i];
    if (c->mc_error) {
        a->rc = -1;
        return;
    }
    for(int rep=0;rep<reps;rep++){
        int last=(rep==reps-1);
        struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;
        sub.task_obj_addr=c->mtk[i].obj;sub.fence_fd=-1;sub.core_mask=1u<<i;
        sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,1};
        sub.timeout=60000;
        if(rknpu_submit_ioctl(fd,&sub,a->w->domain)){ if(last){ a->rc=-1; return; } }
        bsync(fd,CC,RKNPU_MEM_SYNC_FROM_DEVICE);
    }
    c->mwarm[i]=1;
}

static void *mcworker(void *vp){
    struct mcw *a=vp; ork_npu *c=a->c; int i=a->core, nc=a->nc, dt=a->dt, M=a->M, fd=c->fd;
    int K=a->w->K, N=a->w->N, NMAX=c->soc->nmax, CBUF=c->soc->cbuf_elems;
    if(dt==DT_F16 && CBUF>32768) CBUF=32768;   /* cbuf raise (57344, int8 R=32 @K3584) is INT8-ONLY: the fp16 M-scheduler is validated only to the 32768-tile and miscomputes larger fp16 M-tiles (latent bug) */
    int KS=dt ? int8_ks(c) : c->soc->ks, RB=dt?2*CBUF:CBUF, nt_sz=dt?32:16;
    ork_w *w=a->w; const void *A=a->A; struct buf *RC=&c->mrc[i],*AF=&c->maf[i],*CC=&c->mcc[i];
    size_t maxout = a->maxout;
    if(c->mccsz[i]<maxout){
        fprintf(stderr, "[ork] ERROR: mccsz[%d]=%zu < maxout=%zu, buffer not pre-allocated!\n", i, c->mccsz[i], maxout);
        a->rc=-1;
        c->mc_error = 1;
    }
    if(M==1 && w->Bf){   /* int8 DECODE fast path: ONE full-K submit per N-slice (no K-split) */
        if (!c->mc_error) {
            int8_t*ad=AF->cpu; const int8_t*Ai=A; for(int j=0;j<K;j++)ad[j]=Ai[j]; bsync(fd,AF,RKNPU_MEM_SYNC_TO_DEVICE);
        }
        for(int ns=0;ns<w->Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/nt_sz;
            int t0=(int)((long)i*NN/nc),t1=(int)((long)(i+1)*NN/nc);
            int active = (t1 > t0);
            if(!active){
                int Ncore = nt_sz;
                uint32_t rc[REGCMD_N]; synth_i8(rc,1,K,Ncore,(uint32_t)AF->dma,(uint32_t)w->Bf[ns].dma,(uint32_t)CC->dma,1,CBUF,0);
                setr(rc,REGCMD_N,0x201,0x1040,0xb1);
                if (validate_regcmd("mcworker_dec_inactive", c, rc, REGCMD_N, w, NULL, 0)) {
                    a->rc = -1; c->mc_error = 1;
                }
                if (!c->mc_error) {
                    memcpy(RC->cpu,rc,sizeof rc); bsync(fd,RC,RKNPU_MEM_SYNC_TO_DEVICE);
                }
                unified_ioctl(a, i, nc); if(a->rc == -1) return NULL;
            } else {
                int Ncore=(t1-t0)*nt_sz, coff=t0*nt_sz; uint64_t wbase=w->Bf[ns].dma+(uint64_t)t0*K*32;
                double _tp0=ork_now_us();   /* Tier 2a teardown: copy=regcmd-prep, submit=ioctl+result-sync, acc=writeout */
                uint32_t rc[REGCMD_N]; synth_i8(rc,1,K,Ncore,(uint32_t)AF->dma,(uint32_t)wbase,(uint32_t)CC->dma,1,CBUF,0);
                setr(rc,REGCMD_N,0x201,0x1040,0xb1);                       /* M=1 single-tile schedule */
                if (validate_regcmd("mcworker_dec_active", c, rc, REGCMD_N, w, NULL, 0)) {
                    a->rc = -1; c->mc_error = 1;
                }
                if (!c->mc_error) {
                    memcpy(RC->cpu,rc,sizeof rc); bsync(fd,RC,RKNPU_MEM_SYNC_TO_DEVICE);
                    double _ti0=ork_now_us(); g_mc_copy[i]+=_ti0-_tp0;
                    unified_ioctl(a, i, nc); if(a->rc == -1) return NULL;
                    double _tw0=ork_now_us(); g_mc_sub[i]+=_tw0-_ti0;
                    int32_t*cc=CC->cpu,*cr=a->cres; for(int col=0;col<Ncore;col++)cr[n0+coff+col]=cc[col];
                    g_mc_acc[i]+=ork_now_us()-_tw0; g_mc_n[i]++;
                } else {
                    unified_ioctl(a, i, nc); if(a->rc == -1) return NULL;
                }
            }
        }
        return NULL;
    }
    if(a->chain_pref && dt==DT_I8 && M>1 && w->Bf && (K%512)==0 && K<=4096){
        /* CHAIN-PREFILL: PC-chain this core's M-tiles into chained submits (instead of one ioctl per
         * M-tile). M-tiles have disjoint output rows -> no data dependency. AF holds all M rows (staged
         * once); each chained program writes its own disjoint rows of CC; readback CC once per submit.
         * SAME weight => SAME domain & core. Only the ACTIVE N-range is chained; an inactive core (no
         * N-tiles — rare, auto-tuner avoids it) falls through to the per-tile path below.
         * PER-N-SLICE (LEVER #1): we emit ONE submit per N-slice so each submit's chained programs all
         * reference the SINGLE weight buffer Bf[ns] (the cross-N-slice chain referencing multiple
         * distinct Bf[ns] buffers in one submit is what wedged the kernel CDMA walker — errno 110 /
         * "cdma address wild"). Wide-N (Sn>1) now chains within each slice instead of falling through
         * to per-tile; Sn==1 is unchanged (one slice == one submit, as before). */
        int Kp=K, R=RB/Kp; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
        double scale=(double)Kp/512.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale), mg_max = base>=0x1b ? (base-0x1b)/slope+1 : 0;
        int chunk = mg_max * 64; if(chunk < 1) chunk = 1; if(chunk > M) chunk = M;   /* M-tile = the 0x1040 schedule's validated max rows (mg_max*64). NOT R-1: R=pow2_floor(2*cbuf/K) was a FALSE "CBUF-resident rows" cap (~31) that re-streamed the K*N weight from DRAM ~2-4x too often (single-core is weight-DMA-bound). mg_max*64 is the exact bit-exact ceiling (mc+1 miscomputes). See AGENTS.md "weight-DMA amortization". */
        /* SMALL-TILE PACKING (ORK_SMALLTILE, default OFF): mirror rkllm — pack many SMALL CBUF-resident
         * tiles back-to-back in the one chained submit instead of a few LARGE per-core tiles. With it set:
         * M-tile = ORK_SMALLTILE_M (default 32) and each core's Ncore columns are sub-tiled into
         * ORK_SMALLTILE_N-wide column blocks (default 1216, snapped to a multiple of nt_sz). Goal: each
         * task's working set stays in CBUF/SRAM so the MAC stays fed across tasks. Same validated regcmd
         * /K-handling (no 0x107c/0x1044 K-grouping). validate_regcmd + fall back to the per-tile path.
         *
         * M-TILE CEILING (corrected 2026-06-30): the real upper bound is the 0x1040 K-reduction
         * schedule, == mg_max*64 (mc+1 miscomputes; bit-exact-validated at every K: 704@K512, 320@K1024,
         * 128@K2048, 64@K3584/4096). The old "R-1 / CBUF-resident rows" RE finding was WRONG — it claimed
         * R=pow2_floor(2*CBUF/K) (~31) was a hard cap, but activations STREAM (they need not be CBUF-
         * resident) and reg 0x1010 is only a perf hint (correctness is identical regardless). That false
         * cap throttled the M-tile ~2-4x below mg_max*64 and made the kernel re-stream the K*N weight from
         * DRAM far too often (single-core is weight-DMA-bound) — see AGENTS.md "weight-DMA amortization".
         * SMALLTILE deliberately wants SMALL tiles, so it still clamps to R-1 below as its own ceiling
         * (not a correctness limit). The N axis is the free packing lever (sub-tile Ncore). */
        static int st_on=-1, st_m=0, st_n=0;
        if(st_on<0){ const char*e=getenv("ORK_SMALLTILE"); st_on=e?atoi(e):0;
            const char*em=getenv("ORK_SMALLTILE_M"); st_m=em?atoi(em):32;
            const char*en=getenv("ORK_SMALLTILE_N"); st_n=en?atoi(en):1216;
            if(st_m<1)st_m=32; if(st_n<nt_sz)st_n=nt_sz; }
        if(st_on){ int cap=R-1; if(cap<1)cap=1; chunk=st_m; if(chunk>cap)chunk=cap; if(chunk>M)chunk=M; if(chunk<1)chunk=1; }
        int nsub_w = st_on ? ((st_n+nt_sz-1)/nt_sz)*nt_sz : 0;   /* N-subtile width (mult of nt_sz); 0=whole Ncore */
        /* require the active N-range across ALL N-slices; verify all are active (Sn is 1 for the
         * 7B prefill matmuls — nmax=8192 >= N). If any N-slice is inactive for this core, fall back. */
        int all_active=1;
        for(int ns=0;ns<w->Sn;ns++){int Nc=(N-ns*NMAX<NMAX)?(N-ns*NMAX):NMAX,NN=Nc/nt_sz;
            int t0=(int)((long)i*NN/nc),t1=(int)((long)(i+1)*NN/nc); if(t1<=t0)all_active=0;}
        if(all_active){
            /* PER-N-SLICE CHAINING (LEVER #1): emit ONE submit per N-slice — chain only that slice's
             * M-tiles x N-subtiles. Every chained program in a submit references the SINGLE weight
             * buffer Bf[ns] (column subtiles are byte offsets WITHIN Bf[ns], not separate allocations),
             * so this never reintroduces the cross-N-slice / cross-buffer chain that wedges the kernel
             * CDMA walker (errno 110). For Sn==1 this is identical to the old single-chain behaviour
             * (one slice -> one submit); for Sn>1 it replaces (Sn x nmt) per-tile ioctls with (Sn)
             * chained submits. RC/tk/CC are reused from offset 0 for each slice (sequential submits on
             * this core's NPU core), so they only need to hold ONE slice's worth of programs. */
            int nmt=(M+chunk-1)/chunk;
            /* worst-case programs in a SINGLE N-slice (bounds pd[] / RC / tk usage) */
            int Pmax=0;
            for(int ns=0;ns<w->Sn;ns++){int Nc=(N-ns*NMAX<NMAX)?(N-ns*NMAX):NMAX,NN=Nc/nt_sz;
                int t0=(int)((long)i*NN/nc),t1=(int)((long)(i+1)*NN/nc); int Ncore=(t1-t0)*nt_sz;
                int nsw = nsub_w ? nsub_w : Ncore; int nnsub=(Ncore+nsw-1)/nsw;
                int Pns=nnsub*nmt; if(Pns>Pmax)Pmax=Pns; }
            size_t needrc=(size_t)Pmax*REGCMD_I8_N*4, needtk=(size_t)Pmax*sizeof(struct rknpu_task);
            /* per-program output descriptor (column-subtiled): each program writes a contiguous
             * [mco, Nsub] int32 block; scatter accounts for its (m0, n0+coff+nc0) offset. */
            int maxpd = Pmax; if(maxpd<1)maxpd=1;
            struct { size_t cc_off; int m0,mco,nc0,Nsub,n0,coff; } *pd = malloc((size_t)maxpd*sizeof *pd);
            if(needrc<=RC->size && needtk<=c->mtk[i].size && pd && !c->mc_error){
                /* stage all M rows of A into AF (contiguous [M,K]) once; reused by every N-slice */
                double _tc0=ork_now_us();
                memcpy(AF->cpu, A, (size_t)M*K); bsync(fd,AF,RKNPU_MEM_SYNC_TO_DEVICE);
                double _ts0=ork_now_us(); g_mc_copy[i]+=_ts0-_tc0;
                uint32_t rc[REGCMD_I8_N];
                struct rknpu_task *tk=(struct rknpu_task*)c->mtk[i].cpu;
                int bad=0;
                for(int ns=0;ns<w->Sn && !bad;ns++){ double _tsy0=ork_now_us(); int Nc=(N-ns*NMAX<NMAX)?(N-ns*NMAX):NMAX,NN=Nc/nt_sz;
                    int t0=(int)((long)i*NN/nc),t1=(int)((long)(i+1)*NN/nc);
                    int Ncore=(t1-t0)*nt_sz, coff=t0*nt_sz; uint64_t wbase=w->Bf[ns].dma+(uint64_t)t0*K*32;
                    int n0=ns*NMAX;
                    int nsw = nsub_w ? nsub_w : Ncore;
                    /* build THIS N-slice's chain into RC/tk/CC from offset 0; every program references
                     * Bf[ns] + a column byte-offset within it (single-buffer => no errno-110 hazard). */
                    int p=0; size_t cc_off=0;
                    for(int nc0=0;nc0<Ncore && !bad;nc0+=nsw){int Nsub=(Ncore-nc0<nsw)?(Ncore-nc0):nsw;
                        /* weight for this column subtile: Bf tile layout is [Ntile][Ktile][32][32];
                         * advancing by one nt_sz-column tile = K*32 bytes. */
                        uint64_t wsub=wbase+(uint64_t)nc0*K;
                        for(int m0=0;m0<M;m0+=chunk){int mco=(M-m0<chunk)?(M-m0):chunk; if(mco<=0)continue;
                            memset(rc,0,sizeof rc);
                            synth_i8(rc,mco,Kp,Nsub,
                                     (uint32_t)(AF->dma+(uint64_t)m0*K), (uint32_t)wsub,
                                     (uint32_t)(CC->dma+cc_off), 1, CBUF, 0);
                            if(validate_regcmd("mcworker_pref_chain", c, rc, REGCMD_I8_N, w, NULL, 0)){ bad=1; break; }
                            /* PC-chain to next program (same words as run_chain_i8). The final program
                             * of the slice gets its chain words cleared below so the chain terminates
                             * WITHIN this single-buffer slice (no cross-slice link). */
                            uint64_t nx=RC->dma+(size_t)(p+1)*REGCMD_I8_N*4;
                            rc[216]=0x0010|((nx&0xffff)<<16); rc[217]=(0x0101<<16)|((nx>>16)&0xffff);
                            rc[218]=0x0014|(0x0037<<16);      rc[219]=(0x0101<<16)|0;
                            memcpy((char*)RC->cpu+(size_t)p*REGCMD_I8_N*4, rc, REGCMD_I8_N*4);
                            struct rknpu_task t; memset(&t,0,sizeof t);
                            t.enable_mask=0xd; t.int_mask=0x300; t.int_clear=0x1ffff; t.regcfg_amount=108;
                            t.regcmd_addr=RC->dma+(size_t)p*REGCMD_I8_N*4;
                            tk[p]=t;
                            pd[p].cc_off=cc_off; pd[p].m0=m0; pd[p].mco=mco; pd[p].nc0=nc0;
                            pd[p].Nsub=Nsub; pd[p].n0=n0; pd[p].coff=coff;
                            p++;
                            cc_off += (size_t)mco*Nsub*4;
                            if(cc_off>CC->size){ bad=1; break; }
                            /* p is bounded by this slice's tile count <= Pmax (= maxpd, the malloc'd
                             * pd[] / sized RC/tk). The loop math can't exceed it; no guard needed. */
                        }
                    }
                    if(bad||p<1) break;
                    int P=p;   /* programs in THIS N-slice's chain */
                    /* terminate the chain: the last program must not link past itself */
                    { uint32_t *l=(uint32_t*)((char*)RC->cpu+(size_t)(P-1)*REGCMD_I8_N*4);
                      l[216]=l[217]=l[218]=l[219]=0; }
                    bsync(fd,RC,RKNPU_MEM_SYNC_TO_DEVICE);
                    bsync(fd,&c->mtk[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
                    g_mc_synth[i]+=ork_now_us()-_tsy0;   /* host synth+bsync (overlappable); ioctl/NPU = g_mc_sub - this */
                    int reps=c->mwarm[i]?1:2;
                    for(int rep=0;rep<reps && !c->mc_error;rep++){ int last=(rep==reps-1);
                        struct rknpu_submit sub; memset(&sub,0,sizeof sub);
                        sub.flags=0x5; sub.task_number=P; sub.task_obj_addr=c->mtk[i].obj; sub.fence_fd=-1;
                        sub.core_mask=1u<<i;
                        sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)P};
                        sub.timeout=60000;
                        if(rknpu_submit_ioctl(fd,&sub,a->w->domain)){ if(last){a->rc=-1;c->mc_error=1;bad=1;break;} continue; }
                        bsync(fd,CC,RKNPU_MEM_SYNC_FROM_DEVICE);
                    }
                    c->mwarm[i]=1;
                    double _ta0=ork_now_us(); g_mc_sub[i]+=_ta0-_ts0; _ts0=_ta0;
                    if(bad||a->rc==-1) break;
                    /* scatter THIS slice's [mco,Nsub] blocks -> cres (disjoint output cols/rows) */
                    int32_t*cr=a->cres;
                    for(int q=0;q<P;q++){
                        int32_t*cc=(int32_t*)((char*)CC->cpu+pd[q].cc_off);
                        int mco=pd[q].mco,Nsub=pd[q].Nsub,m0=pd[q].m0;
                        int col0=pd[q].n0+pd[q].coff+pd[q].nc0;
                        for(int r=0;r<mco;r++)for(int n=0;n<Nsub;n++)
                            cr[(size_t)(m0+r)*N+(col0+n)]=cc[(size_t)r*Nsub+n];
                    }
                    g_mc_acc[i]+=ork_now_us()-_ta0; g_mc_n[i]++;
                }
                if(!bad && a->rc!=-1){ free(pd); return NULL; }
                /* on bad/validate/submit failure, fall through to the per-tile path. It rewrites ALL
                 * of this core's output columns (= over the same disjoint rows/cols this core owns),
                 * so an aborted partial-slice scatter above is harmless. */
            }
            free(pd);
        }
        /* fall through to original per-tile prefill below */
    }
    if(dt==DT_I8 && M>1 && w->Bf && (K%512)==0 && K<=4096){   /* Tier 1c-ii: full-K PREFILL — one submit/M-tile over full K, no K-split accumulate */
        int Kp=K, R=RB/Kp; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
        double scale=(double)Kp/512.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale), mg_max = base>=0x1b ? (base-0x1b)/slope+1 : 0;
        int chunk = mg_max * 64; if(chunk < 1) chunk = 1; if(chunk > M) chunk = M;   /* M-tile = the 0x1040 schedule's validated max rows (mg_max*64). NOT R-1: R=pow2_floor(2*cbuf/K) was a FALSE "CBUF-resident rows" cap (~31) that re-streamed the K*N weight from DRAM ~2-4x too often (single-core is weight-DMA-bound). mg_max*64 is the exact bit-exact ceiling (mc+1 miscomputes). See AGENTS.md "weight-DMA amortization". */
        for(int ns=0;ns<w->Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/nt_sz;
            int t0=(int)((long)i*NN/nc),t1=(int)((long)(i+1)*NN/nc);
            int active = (t1 > t0);
            for(int m0=0;m0<M;m0+=chunk){int mco=(M-m0<chunk)?(M-m0):chunk; if(mco<=0)continue;
                if(!active){
                    int Ncore = nt_sz;
                    uint32_t rc[REGCMD_N]; synth_i8(rc,mco,Kp,Ncore,(uint32_t)AF->dma,(uint32_t)w->Bf[ns].dma,(uint32_t)CC->dma,1,CBUF,0);
                    if (validate_regcmd("mcworker_pref_inactive", c, rc, REGCMD_N, w, NULL, 0)) {
                        a->rc = -1; c->mc_error = 1;
                    }
                    if (!c->mc_error) {
                        memcpy(RC->cpu,rc,sizeof rc); bsync(fd,RC,RKNPU_MEM_SYNC_TO_DEVICE);
                    }
                    unified_ioctl(a, i, nc); if(a->rc == -1) return NULL;
                } else {
                    int Ncore=(t1-t0)*nt_sz, coff=t0*nt_sz; uint64_t wbase=w->Bf[ns].dma+(uint64_t)t0*K*32;
                    double _tc0=ork_now_us();
                    if (!c->mc_error) {
                        int8_t*ad=AF->cpu; const int8_t*Ai=A; for(int r=0;r<mco;r++)for(int j=0;j<K;j++)ad[(size_t)r*K+j]=Ai[(size_t)(m0+r)*K+j];
                        bsync(fd,AF,RKNPU_MEM_SYNC_TO_DEVICE);
                    }
                    double _ts0=ork_now_us(); g_mc_copy[i]+=_ts0-_tc0;
                    uint32_t rc[REGCMD_N]; synth_i8(rc,mco,Kp,Ncore,(uint32_t)AF->dma,(uint32_t)wbase,(uint32_t)CC->dma,1,CBUF,0);
                    if (validate_regcmd("mcworker_pref_active", c, rc, REGCMD_N, w, NULL, 0)) {
                        a->rc = -1; c->mc_error = 1;
                    }
                    if (!c->mc_error) {
                        memcpy(RC->cpu,rc,sizeof rc); bsync(fd,RC,RKNPU_MEM_SYNC_TO_DEVICE);
                        unified_ioctl(a, i, nc); if(a->rc == -1) return NULL;
                        double _ta0=ork_now_us(); g_mc_sub[i]+=_ta0-_ts0;
                        int32_t*cc=CC->cpu,*cr=a->cres; for(int r=0;r<mco;r++)for(int n=0;n<Ncore;n++)cr[(size_t)(m0+r)*N+(n0+coff+n)]=cc[(size_t)r*Ncore+n];
                        g_mc_acc[i]+=ork_now_us()-_ta0; g_mc_n[i]++;
                    } else {
                        unified_ioctl(a, i, nc); if(a->rc == -1) return NULL;
                    }
                }
            }
        }
        return NULL;
    }
    if(a->chain_ksplit && dt==DT_I8 && M>1 && (K%512)==0 && K>4096 && w->Sn==1){
        /* CHAIN-KSPLIT: wide-K (K>4096, Bf=NULL → no full-K weight) normally K-splits into
         * ceil(K/KS) separate ioctls/call (run_i8's dominant prefill submit cost). Instead PC-chain
         * the K-slice (and M-tile) programs into ONE submit per core, then host-accumulate the
         * K-slice partials. The NPU has no on-device C+= mode in our regcmd, so each K-slice must
         * write its own partial slot; we cap the simultaneous partials per submit to a CC byte
         * budget and accumulate between batches (still far fewer ioctls than per-slice). Only the
         * ACTIVE N-range is chained; a core with any inactive N-slice falls through. Gated
         * ORK_CHAIN_KSPLIT (default on); the address math is per-slice exact.
         * Sn==1 ONLY: like chain-prefill, a chained submit spanning >1 N-slice references multiple
         * distinct Bb[ns*Sk+ks] weight buffers, the cross-buffer reference the kernel CDMA walker
         * rejects (errno 110 / "cdma address wild"). ffn_down (the only wide-K matmul, K=18944 N=3584)
         * is Sn=1 so this is a no-op there; a hypothetical wide-K+wide-N weight (N>nmax) falls through
         * to the proven per-slice K-split accumulate below. */
        int all_active=1;
        for(int ns=0;ns<w->Sn;ns++){int Nc=(N-ns*NMAX<NMAX)?(N-ns*NMAX):NMAX,NN=Nc/nt_sz;
            int t0=(int)((long)i*NN/nc),t1=(int)((long)(i+1)*NN/nc); if(t1<=t0)all_active=0;}
        if(all_active && !c->mc_error){
            /* per-core CC partial budget (bytes). ffn_down (K=18944→19 slices, Ncore~1.2k, M=256)
             * needs ~24 MB; default 64 MB covers it in one batch. ORK_CHAIN_KSPLIT_MB overrides. */
            static long ccbudget=-1; if(ccbudget<0){const char*e=getenv("ORK_CHAIN_KSPLIT_MB"); ccbudget=(long)(e?atoi(e):64)*1024*1024;}
            int bad=0;
            /* Walk K-slices, batching into PC-chains that fit the CC budget. Each program in a batch
             * writes a disjoint CC region (its own K-slice partial for its N-slice/M-tile). After a
             * batch submit, accumulate every partial into cres, then reuse CC for the next batch. */
            int ks=0;
            while(ks<w->Sk && !bad && !c->mc_error){
                /* plan this batch: greedily add K-slices while CC partials fit the budget */
                int ks_end=ks; size_t cc_total=0; int P=0;
                /* record per (program) descriptor for accumulate; bounded by REGCMD/task buffer caps */
                struct { int ns,ks,m0,mco,Ncore,coff,n0; size_t cc_off, af_off; } pd[256];
                size_t cc_off=0, af_off=0;
                for(int kk=ks; kk<w->Sk; kk++){
                    int k0=kk*KS, Kp=(K-k0<KS)?(K-k0):KS;
                    int sched=(Kp==1024||Kp==512),R=RB/Kp; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
                    double scale=(double)Kp/512.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale), mg_max = base>=0x1b ? (base-0x1b)/slope+1 : 0;
                    int chunk = mg_max * 64; if(!sched) chunk = (RB/2)/Kp; if(chunk < 4*R) chunk = sched ? 4*R : ((RB/2)/Kp); if(chunk > M) chunk = M; if(chunk < 1) chunk = 1;
                    /* tentative bytes for this K-slice (all N-slices, all M-tiles) */
                    size_t kk_bytes=0; int kk_progs=0;
                    for(int ns=0;ns<w->Sn;ns++){int Nc=(N-ns*NMAX<NMAX)?(N-ns*NMAX):NMAX,NN=Nc/nt_sz;
                        int t0=(int)((long)i*NN/nc),t1=(int)((long)(i+1)*NN/nc); int Ncore=(t1-t0)*nt_sz;
                        for(int m0=0;m0<M;m0+=chunk){int mco=(M-m0<chunk)?(M-m0):chunk; if(mco<=0)continue;
                            kk_bytes += (size_t)mco*Ncore*4; kk_progs++; }
                    }
                    /* always take at least one K-slice even if it alone exceeds budget. A K-slice must
                     * be placed WHOLE (all its N/M programs) or not at all — partial placement drops
                     * part of the K-reduction → wrong result. If even the first slice can't fit pd[],
                     * bail to the per-tile fall-through (bad). */
                    if(kk>ks && (cc_total+kk_bytes>(size_t)ccbudget || P+kk_progs>(int)(sizeof(pd)/sizeof(pd[0])))) break;
                    if(P+kk_progs>(int)(sizeof(pd)/sizeof(pd[0]))){ bad=1; break; }   /* first slice too big for pd[] */
                    /* lay out program descriptors for this K-slice */
                    int placed=0;
                    for(int ns=0;ns<w->Sn && placed<kk_progs;ns++){int Nc=(N-ns*NMAX<NMAX)?(N-ns*NMAX):NMAX,NN=Nc/nt_sz;
                        int t0=(int)((long)i*NN/nc),t1=(int)((long)(i+1)*NN/nc); int Ncore=(t1-t0)*nt_sz, coff=t0*nt_sz;
                        for(int m0=0;m0<M && placed<kk_progs;m0+=chunk){int mco=(M-m0<chunk)?(M-m0):chunk; if(mco<=0)continue;
                            pd[P].ns=ns; pd[P].ks=kk; pd[P].m0=m0; pd[P].mco=mco; pd[P].Ncore=Ncore; pd[P].coff=coff; pd[P].n0=ns*NMAX;
                            pd[P].cc_off=cc_off; cc_off+=(size_t)mco*Ncore*4;
                            pd[P].af_off=af_off; af_off+=(size_t)mco*Kp;   /* gathered A tile [mco][Kp] */
                            P++; placed++;
                        }
                    }
                    cc_total=cc_off; ks_end=kk+1;
                    if(P>=(int)(sizeof(pd)/sizeof(pd[0]))) break;
                }
                if(bad) break;   /* planning could not represent a K-slice → per-tile fall-through */
                size_t needrc=(size_t)P*REGCMD_I8_N*4, needtk=(size_t)P*sizeof(struct rknpu_task);
                if(needrc>RC->size || needtk>c->mtk[i].size || cc_total>CC->size || af_off>AF->size){ bad=1; break; }
                /* gather each program's A tile [mco][Kp] into AF (regcmd reads A with row-stride Kp,
                 * so the full-K-strided host A must be re-tiled per K-slice). */
                double _tc0=ork_now_us();
                for(int p=0;p<P;p++){
                    int kk=pd[p].ks, k0=kk*KS, Kp=(K-k0<KS)?(K-k0):KS;
                    int8_t*ad=(int8_t*)AF->cpu+pd[p].af_off; const int8_t*Ai=A;
                    for(int r=0;r<pd[p].mco;r++)for(int j=0;j<Kp;j++)ad[(size_t)r*Kp+j]=Ai[(size_t)(pd[p].m0+r)*K+k0+j];
                }
                bsync(fd,AF,RKNPU_MEM_SYNC_TO_DEVICE);
                double _ts0=ork_now_us(); g_mc_copy[i]+=_ts0-_tc0;
                /* synth + PC-chain every program in this batch */
                uint32_t rc[REGCMD_I8_N];
                struct rknpu_task *tk=(struct rknpu_task*)c->mtk[i].cpu;
                for(int p=0;p<P && !bad;p++){
                    int kk=pd[p].ks, k0=kk*KS, Kp=(K-k0<KS)?(K-k0):KS;
                    int sched=(Kp==1024||Kp==512);
                    struct buf*Bb=&w->Bb[(size_t)pd[p].ns*w->Sk+kk];
                    uint64_t wbase=Bb->dma+(uint64_t)(pd[p].coff/nt_sz)*Kp*32;   /* coff/nt_sz = t0 tile index */
                    memset(rc,0,sizeof rc);
                    synth_i8(rc, pd[p].mco, Kp, pd[p].Ncore,
                             (uint32_t)(AF->dma+pd[p].af_off),   /* this program's gathered [mco][Kp] tile */
                             (uint32_t)wbase,
                             (uint32_t)(CC->dma+pd[p].cc_off), sched, CBUF, 0);
                    if(validate_regcmd("mcworker_pref_ksplit", c, rc, REGCMD_I8_N, w, NULL, 0)){ bad=1; break; }
                    if(p<P-1){   /* PC-chain to next program */
                        uint64_t nx=RC->dma+(size_t)(p+1)*REGCMD_I8_N*4;
                        rc[216]=0x0010|((nx&0xffff)<<16); rc[217]=(0x0101<<16)|((nx>>16)&0xffff);
                        rc[218]=0x0014|(0x0037<<16);      rc[219]=(0x0101<<16)|0;
                    }
                    memcpy((char*)RC->cpu+(size_t)p*REGCMD_I8_N*4, rc, REGCMD_I8_N*4);
                    struct rknpu_task t; memset(&t,0,sizeof t);
                    t.enable_mask=0xd; t.int_mask=0x300; t.int_clear=0x1ffff; t.regcfg_amount=108;
                    t.regcmd_addr=RC->dma+(size_t)p*REGCMD_I8_N*4;
                    tk[p]=t;
                }
                if(bad) break;
                bsync(fd,RC,RKNPU_MEM_SYNC_TO_DEVICE);
                bsync(fd,&c->mtk[i],RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
                int reps=c->mwarm[i]?1:2;
                double _tsub0=ork_now_us();
                for(int rep=0;rep<reps && !c->mc_error;rep++){ int last=(rep==reps-1);
                    struct rknpu_submit sub; memset(&sub,0,sizeof sub);
                    sub.flags=0x5; sub.task_number=P; sub.task_obj_addr=c->mtk[i].obj; sub.fence_fd=-1;
                    sub.core_mask=1u<<i;
                    sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)P};
                    sub.timeout=60000;
                    if(rknpu_submit_ioctl(fd,&sub,a->w->domain)){ if(last){a->rc=-1;c->mc_error=1;bad=1;break;} continue; }
                    bsync(fd,CC,RKNPU_MEM_SYNC_FROM_DEVICE);
                }
                c->mwarm[i]=1;
                g_mc_sub[i]+=ork_now_us()-_tsub0; g_mc_n[i]++;
                if(bad||a->rc==-1) break;
                /* accumulate this batch's K-slice partials into cres (cres pre-zeroed by run_multicore) */
                double _ta0=ork_now_us();
                int32_t*cr=a->cres;
                for(int p=0;p<P;p++){
                    int32_t*cc=(int32_t*)((char*)CC->cpu+pd[p].cc_off);
                    int Ncore=pd[p].Ncore, base_n=pd[p].n0+pd[p].coff;
                    for(int r=0;r<pd[p].mco;r++){
                        size_t cr_off=(size_t)(pd[p].m0+r)*N+base_n;
                        size_t cc_row=(size_t)r*Ncore; int col=0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                        for(;col<=Ncore-16;col+=16){
                            int32x4_t a0=vld1q_s32(&cc[cc_row+col]),    a1=vld1q_s32(&cc[cc_row+col+4]);
                            int32x4_t a2=vld1q_s32(&cc[cc_row+col+8]),  a3=vld1q_s32(&cc[cc_row+col+12]);
                            int32x4_t b0=vld1q_s32(&cr[cr_off+col]),    b1=vld1q_s32(&cr[cr_off+col+4]);
                            int32x4_t b2=vld1q_s32(&cr[cr_off+col+8]),  b3=vld1q_s32(&cr[cr_off+col+12]);
                            vst1q_s32(&cr[cr_off+col],   vaddq_s32(b0,a0)); vst1q_s32(&cr[cr_off+col+4], vaddq_s32(b1,a1));
                            vst1q_s32(&cr[cr_off+col+8], vaddq_s32(b2,a2)); vst1q_s32(&cr[cr_off+col+12],vaddq_s32(b3,a3));
                        }
#endif
                        for(;col<Ncore;col++) cr[cr_off+col]+=cc[cc_row+col];
                    }
                }
                g_mc_acc[i]+=ork_now_us()-_ta0;
                ks=ks_end;
            }
            if(!bad && a->rc!=-1) return NULL;
            /* on any failure fall through to the robust per-tile K-split path below. Zero THIS core's
             * output columns first (we may have partially accumulated) so the per-tile path's +=
             * starts clean; other cores own disjoint columns and are untouched. */
            c->mc_error=0; a->rc=0;
            int32_t*cr=a->cres;
            for(int ns=0;ns<w->Sn;ns++){int Nc=(N-ns*NMAX<NMAX)?(N-ns*NMAX):NMAX,NN=Nc/nt_sz;
                int t0=(int)((long)i*NN/nc),t1=(int)((long)(i+1)*NN/nc); if(t1<=t0)continue;
                int Ncore=(t1-t0)*nt_sz, base_n=ns*NMAX+t0*nt_sz;
                for(int r=0;r<M;r++) memset(&cr[(size_t)r*N+base_n],0,(size_t)Ncore*4);
            }
        }
    }
    for(int ns=0;ns<w->Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/nt_sz;
        int t0=(int)((long)i*NN/nc),t1=(int)((long)(i+1)*NN/nc);
        int active = (t1 > t0);
        int Ncore = active ? (t1-t0)*nt_sz : nt_sz;
        int coff = active ? t0*nt_sz : 0;
        for(int ks=0;ks<w->Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;
            int sched=dt?(Kp==1024||Kp==512):((Kp&(Kp-1))==0 && Kp<2048),R=RB/Kp;if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
            double scale=(double)Kp/(dt?512.0:256.0); int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale), mg_max = base>=0x1b ? (base-0x1b)/slope+1 : 0;
            int chunk = mg_max * 64; if(!sched) chunk = (RB/2)/Kp; if(chunk < 4*R) chunk = sched ? 4*R : ((RB/2)/Kp); if(chunk > M) chunk = M; if(chunk < 1) chunk = 1;
            struct buf*Bb=&w->Bb[(size_t)ns*w->Sk+ks]; uint64_t wbase=Bb->dma+(uint64_t)(active?t0:0)*Kp*32;  /* Kp*32 B/N-tile (both dtypes) */
            for(int m0=0;m0<M;m0+=chunk){int mco=(M-m0<chunk)?(M-m0):chunk; if(mco<=0)continue;
                if(!active){
                    uint32_t rc[REGCMD_N];
                    if(dt==DT_F16)synth   (rc,mco,Kp,Ncore,(uint32_t)AF->dma,(uint32_t)wbase,(uint32_t)CC->dma,sched,CBUF);
                    else          synth_i8(rc,mco,Kp,Ncore,(uint32_t)AF->dma,(uint32_t)wbase,(uint32_t)CC->dma,sched,CBUF,0);
                    if (validate_regcmd("mcworker_loop_inactive", c, rc, REGCMD_N, w, NULL, 0)) {
                        a->rc = -1; c->mc_error = 1;
                    }
                    if (!c->mc_error) {
                        memcpy(RC->cpu,rc,sizeof rc); bsync(fd,RC,RKNPU_MEM_SYNC_TO_DEVICE);
                    }
                    unified_ioctl(a, i, nc); if(a->rc == -1) return NULL;
                } else {
                    double _tc0=ork_now_us();
                    if (!c->mc_error) {
                        if(dt==DT_F16){f16*ad=AF->cpu;const f16*Af=A;for(int r=0;r<mco;r++)for(int j=0;j<Kp;j++)ad[(size_t)r*Kp+j]=Af[(size_t)(m0+r)*K+k0+j];}
                        else{int8_t*ad=AF->cpu;const int8_t*Ai=A;for(int r=0;r<mco;r++)for(int j=0;j<Kp;j++)ad[(size_t)r*Kp+j]=Ai[(size_t)(m0+r)*K+k0+j];}
                        bsync(fd,AF,RKNPU_MEM_SYNC_TO_DEVICE);
                    }
                    double _ts0=ork_now_us(); g_mc_copy[i]+=_ts0-_tc0;
                    uint32_t rc[REGCMD_N];
                    if(dt==DT_F16)synth   (rc,mco,Kp,Ncore,(uint32_t)AF->dma,(uint32_t)wbase,(uint32_t)CC->dma,sched,CBUF);
                    else          synth_i8(rc,mco,Kp,Ncore,(uint32_t)AF->dma,(uint32_t)wbase,(uint32_t)CC->dma,sched,CBUF,0);
                    if (validate_regcmd("mcworker_loop_active", c, rc, REGCMD_N, w, NULL, 0)) {
                        a->rc = -1; c->mc_error = 1;
                    }
                    if (!c->mc_error) {
                        memcpy(RC->cpu,rc,sizeof rc); bsync(fd,RC,RKNPU_MEM_SYNC_TO_DEVICE);
                        unified_ioctl(a, i, nc); if(a->rc == -1) return NULL;
                        double _ta0=ork_now_us(); g_mc_sub[i]+=_ta0-_ts0;
                        if(dt==DT_F16){
                            float  *cc=CC->cpu,*cr=a->cres;
                            for(int r=0;r<mco;r++) {
                                size_t cr_row_offset = (size_t)(m0+r)*N + (n0+coff);
                                size_t cc_row_offset = (size_t)r*Ncore;
                                int col = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                                for (; col <= Ncore - 16; col += 16) {
                                    float32x4_t vcc0 = vld1q_f32(&cc[cc_row_offset + col]);
                                    float32x4_t vcc1 = vld1q_f32(&cc[cc_row_offset + col + 4]);
                                    float32x4_t vcc2 = vld1q_f32(&cc[cc_row_offset + col + 8]);
                                    float32x4_t vcc3 = vld1q_f32(&cc[cc_row_offset + col + 12]);

                                    float32x4_t vcr0 = vld1q_f32(&cr[cr_row_offset + col]);
                                    float32x4_t vcr1 = vld1q_f32(&cr[cr_row_offset + col + 4]);
                                    float32x4_t vcr2 = vld1q_f32(&cr[cr_row_offset + col + 8]);
                                    float32x4_t vcr3 = vld1q_f32(&cr[cr_row_offset + col + 12]);

                                    vcr0 = vaddq_f32(vcr0, vcc0);
                                    vcr1 = vaddq_f32(vcr1, vcc1);
                                    vcr2 = vaddq_f32(vcr2, vcc2);
                                    vcr3 = vaddq_f32(vcr3, vcc3);

                                    vst1q_f32(&cr[cr_row_offset + col], vcr0);
                                    vst1q_f32(&cr[cr_row_offset + col + 4], vcr1);
                                    vst1q_f32(&cr[cr_row_offset + col + 8], vcr2);
                                    vst1q_f32(&cr[cr_row_offset + col + 12], vcr3);
                                }
#endif
                                for (; col < Ncore; col++) {
                                    cr[cr_row_offset + col] += cc[cc_row_offset + col];
                                }
                            }
                        }
                        else{
                            int32_t*cc=CC->cpu,*cr=a->cres;
                            for(int r=0;r<mco;r++) {
                                size_t cr_row_offset = (size_t)(m0+r)*N + (n0+coff);
                                size_t cc_row_offset = (size_t)r*Ncore;
                                int col = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                                for (; col <= Ncore - 16; col += 16) {
                                    int32x4_t vcc0 = vld1q_s32(&cc[cc_row_offset + col]);
                                    int32x4_t vcc1 = vld1q_s32(&cc[cc_row_offset + col + 4]);
                                    int32x4_t vcc2 = vld1q_s32(&cc[cc_row_offset + col + 8]);
                                    int32x4_t vcc3 = vld1q_s32(&cc[cc_row_offset + col + 12]);

                                    int32x4_t vcr0 = vld1q_s32(&cr[cr_row_offset + col]);
                                    int32x4_t vcr1 = vld1q_s32(&cr[cr_row_offset + col + 4]);
                                    int32x4_t vcr2 = vld1q_s32(&cr[cr_row_offset + col + 8]);
                                    int32x4_t vcr3 = vld1q_s32(&cr[cr_row_offset + col + 12]);

                                    vcr0 = vaddq_s32(vcr0, vcc0);
                                    vcr1 = vaddq_s32(vcr1, vcc1);
                                    vcr2 = vaddq_s32(vcr2, vcc2);
                                    vcr3 = vaddq_s32(vcr3, vcc3);

                                    vst1q_s32(&cr[cr_row_offset + col], vcr0);
                                    vst1q_s32(&cr[cr_row_offset + col + 4], vcr1);
                                    vst1q_s32(&cr[cr_row_offset + col + 8], vcr2);
                                    vst1q_s32(&cr[cr_row_offset + col + 12], vcr3);
                                }
#endif
                                for (; col < Ncore; col++) {
                                    cr[cr_row_offset + col] += cc[cc_row_offset + col];
                                }
                            }
                        }
                        g_mc_acc[i]+=ork_now_us()-_ta0; g_mc_n[i]++;
                    } else {
                        unified_ioctl(a, i, nc); if(a->rc == -1) return NULL;
                    }
                }
            }
        }
    }
    return NULL;
}
/* persistent worker pool: spawned once, each pinned to driving NPU core `id`. Signalled per matmul
 * (gen bump) — workers with id<nc run mcworker for that job, the rest sleep. Replaces per-matmul
 * pthread_create/join (the spawn cost matters at ~200 matmuls/decode-token). */
/* Pin the calling thread to a big CPU core. On RK3576 (4×A72+4×A53) and RK3588 (4×A76+4×A55)
 * the big cluster is the HIGH-numbered CPUs, so map NPU-driver thread `id` -> CPU (ncpu-1-id):
 * distinct big cores, no contention. Without this the scheduler parks the pool workers on the
 * little cores, making them ~2x slower than the (lucky big-core) calling thread and collapsing
 * multi-core prefill scaling to ~1.1x. ORK_NO_AFFINITY=1 disables (e.g. odd topologies). */
static void pin_big_core(int id){
    static int off=-1; if(off<0) off=getenv("ORK_NO_AFFINITY")?1:0;   /* cached: hot for i4 per-call */
    if(off) return;
#if defined(__linux__)
    long ncpu=sysconf(_SC_NPROCESSORS_ONLN); if(ncpu<2) return;
    int cpu=(int)ncpu-1-id; if(cpu<0) cpu=0;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu,&s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
#endif
    /* NOTE: this pins only the dedicated NPU-driver threads to their own big core. We deliberately do
     * NOT restrict the whole process / CPU threadpool to the big cluster: doing so at init was measured
     * to oversubscribe and CRATER decode at -t 8 (9.3 -> 2.3 tok/s) while not helping -t 4. The big-core
     * win for the CPU side comes from running with -t = big-core-count (e.g. -t 4 on RK3588), which is a
     * user/serving choice, not something to force here. See the Thread-Count wiki experiment. */
}
static void *npu_pool_worker(void *vp){
    struct ork_pw *pw=vp; ork_npu *c=pw->c; int id=pw->id, mygen=0;
    pin_big_core(id);                          /* keep this worker off the little cores */
    for(;;){
        pthread_mutex_lock(&c->pmu);
        while(c->pgen==mygen && !c->pstop) pthread_cond_wait(&c->pgo,&c->pmu);
        if(c->pstop){ pthread_mutex_unlock(&c->pmu); return NULL; }
        mygen=c->pgen; int nc=c->pjob_nc; void *args=c->pjob; void *(*fn)(void*)=c->pjob_fn; size_t st=c->pjob_stride; pthread_mutex_unlock(&c->pmu);
        if(id<nc){ fn((char*)args + (size_t)id*st);   /* mcworker (run_multicore) or chain_core_worker */
            pthread_mutex_lock(&c->pmu); if(++c->pdone==nc-1) pthread_cond_signal(&c->pdn); pthread_mutex_unlock(&c->pmu); }
    }
}
static void npu_pool_ensure(ork_npu *c){
    if(c->pool_n) return;
    pin_big_core(0);                           /* calling thread drives NPU core 0 — keep it big too */
    c->pool_n=c->soc->cores>ORK_MAXCORE?ORK_MAXCORE:c->soc->cores;
    for(int i=1;i<c->pool_n;i++){ c->pwa[i]=(struct ork_pw){c,i}; pthread_create(&c->pth[i],NULL,npu_pool_worker,&c->pwa[i]); }
}
static double ork_now_us(void);   /* defined below */
/* run_multicore phase timing (ORK_RT): setup (checks+mc_ensure+cres memset), submit (pool dispatch
 * + workers + NPU), copy (cres->C). Pin where the integration's per-matmul time goes vs the kernel. */
static double g_rt_setup=0, g_rt_submit=0, g_rt_copy=0; static long g_rt_n=0;
void ork_npu_run_timing(double*setup,double*submit,double*copy,long*n){ if(setup)*setup=g_rt_setup; if(submit)*submit=g_rt_submit; if(copy)*copy=g_rt_copy; if(n)*n=g_rt_n; }
static int run_multicore(ork_npu *c,ork_w *w,int M,const void *A,void *C,int nc){
    int dt=w->dtype, fd=c->fd;
    const double ts=ork_now_us();
    /* never exceed the hardware (or the buffer-array bound) — a bad ORK_NPU_MC can't over-index */
    if(nc>c->soc->cores) nc=c->soc->cores;
    if(nc>ORK_MAXCORE)  nc=ORK_MAXCORE;
    if(nc<1) nc=1;
    if(dt!=c->last_dt){ if(dt==DT_I8 && !ORK_I8_LIVE(c->last_dt)) act(fd,RKNPU_ACT_RESET,0); for(int i=0;i<ORK_MAXCORE;i++){c->mwarm[i]=0;c->mccsz[i]=0;} c->last_dt=dt; }
    if(mc_ensure(c,nc)) return -1;

    /* Pre-allocate multi-core buffers on the single calling thread to eliminate concurrent allocations / race conditions */
    int N=w->N, K=w->K, NMAX=c->soc->nmax, CBUF=c->soc->cbuf_elems;
    if(dt==DT_F16 && CBUF>32768) CBUF=32768;   /* int8-only cbuf raise; fp16 keeps its validated 32768 tiling (see mcworker) */
    int KS=dt ? int8_ks(c) : c->soc->ks, RB=dt?2*CBUF:CBUF, nt_sz=dt?32:16;
    /* CHAIN-PREFILL (ORK_CHAIN_PREFILL, default ON): in the int8 M>1 full-K prefill path each core
     * normally issues one ioctl per M-tile (serial ~134us floor each — the dominant prefill submit
     * source, ~18/matmul/core). When set, the core instead PC-chains ALL its M-tiles into ONE submit
     * (task_number=P). Independent disjoint-row outputs -> no data dep, same weight/domain/core. This
     * needs the core's AF to hold ALL M rows (M*K) and CC the full M*Ncore output (each tile writes
     * disjoint rows) rather than one-tile scratch. Set ORK_CHAIN_PREFILL=0 to revert per-tile submits.
     * PER-N-SLICE (LEVER #1): the chain is emitted ONE submit per N-slice, so every chained submit
     * references a SINGLE weight buffer Bf[ns]. A chain spanning >1 N-slice would reference several
     * distinct Bf[ns] buffers and the kernel's regcmd CDMA walker rejects that (RKNPU_SUBMIT errno
     * 110 / "cdma address wild" → 60s job timeout → NPU soft-reset) — exactly what the old Sn==1 gate
     * avoided. Now wide-N (e.g. ffn_gate/up N=18944 → Sn=3 when nmax<N) still chains, just per slice:
     * (Sn) chained submits instead of (Sn × M-tiles) per-tile ioctls. */
    static int chain_pref=-1; if(chain_pref<0){const char*e=getenv("ORK_CHAIN_PREFILL"); chain_pref=e?atoi(e):1;}
    int use_chain_pref = chain_pref && dt==DT_I8 && M>1 && w->Bf && (K%512)==0 && K<=4096;
    /* CHAIN-KSPLIT (ORK_CHAIN_KSPLIT, default ON): wide-K int8 prefill (K>4096, no Bf) PC-chains its
     * K-slice submits into one ioctl/core (see mcworker). Needs maf to hold all M*K of A and mcc to
     * hold a budget's worth of K-slice partials + mrc/mtk to hold the chained programs. */
    static int chain_ks=-1; if(chain_ks<0){const char*e=getenv("ORK_CHAIN_KSPLIT"); chain_ks=e?atoi(e):1;}
    static long ks_ccbudget=-1; if(ks_ccbudget<0){const char*e=getenv("ORK_CHAIN_KSPLIT_MB"); ks_ccbudget=(long)(e?atoi(e):64)*1024*1024;}
    int use_chain_ksplit = chain_ks && dt==DT_I8 && M>1 && (K%512)==0 && K>4096 && w->Sn==1;   /* Sn>1 (wide-K+wide-N): per-slice fall-through (cross-Bb-buffer chain wedges; see mcworker) */
    size_t core_maxout[ORK_MAXCORE] = {0};
    for(int i=0;i<nc;i++){
        size_t maxout=0, maxaf=0;
        for(int ns=0;ns<w->Sn;ns++){int Nc=(N-ns*NMAX<NMAX)?(N-ns*NMAX):NMAX,NN=Nc/nt_sz;
            int t0=(int)((long)i*NN/nc),t1=(int)((long)(i+1)*NN/nc),cols=(t1-t0)*nt_sz;
            int active = (cols > 0);
            int eff_cols = active ? cols : nt_sz;
            for(int k0=0;k0<K;k0+=KS){
                int Kp=(K-k0<KS)?(K-k0):KS;
                int sd=dt?(Kp==1024||Kp==512):((Kp&(Kp-1))==0 && Kp<2048);
                int R=RB/Kp; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
                double scale=(double)Kp/(dt?512.0:256.0); int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale), mg_max = base>=0x1b ? (base-0x1b)/slope+1 : 0;
                int chunk = mg_max * 64; if(!sd) chunk = (RB/2)/Kp; if(chunk < 4*R) chunk = sd ? 4*R : ((RB/2)/Kp); if(chunk > M) chunk = M; if(chunk < 1) chunk = 1;
                int rows=chunk<M?chunk:M;
                size_t o=(size_t)rows*eff_cols*4; if(o>maxout)maxout=o;
                size_t sz=(size_t)rows*Kp*(dt?1:2); if(sz>maxaf)maxaf=sz;
            }
        }
        if(dt==DT_I8 && M>1 && w->Bf && (K%512)==0 && K<=4096){
            int Kp=K, R=RB/Kp; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
            double scale=(double)Kp/512.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale), mg_max = base>=0x1b ? (base-0x1b)/slope+1 : 0;
            int chunk = mg_max * 64; if(chunk < 1) chunk = 1; if(chunk > M) chunk = M;   /* M-tile = the 0x1040 schedule's validated max rows (mg_max*64). NOT R-1: R=pow2_floor(2*cbuf/K) was a FALSE "CBUF-resident rows" cap (~31) that re-streamed the K*N weight from DRAM ~2-4x too often (single-core is weight-DMA-bound). mg_max*64 is the exact bit-exact ceiling (mc+1 miscomputes). See AGENTS.md "weight-DMA amortization". */
            int rows=chunk<M?chunk:M;
            size_t sz=(size_t)rows*Kp*1;
            if(sz>maxaf)maxaf=sz;
            /* CHAIN-PREFILL: AF holds ALL M rows (staged once), CC holds full M*eff_cols output
             * (each chained tile writes its own disjoint block; readback once after the submit).
             * The chained programs are laid out CONTIGUOUSLY in CC (running offset across N-slices
             * and, under ORK_SMALLTILE, column subtiles), so CC must hold the SUM over N-slices
             * of M*Ncore*4 — not the max. (Sn==1 for the 7B prefill matmuls, so this == the old
             * max there.) Also grow mrc/mtk to hold the (possibly large) chained program count. */
            if(use_chain_pref){
                size_t afull=(size_t)M*Kp*1; if(afull>maxaf)maxaf=afull;
                static int st_on2=-1,st_m2=0,st_n2=0;
                if(st_on2<0){const char*e=getenv("ORK_SMALLTILE");st_on2=e?atoi(e):0;
                    const char*em=getenv("ORK_SMALLTILE_M");st_m2=em?atoi(em):32;
                    const char*en=getenv("ORK_SMALLTILE_N");st_n2=en?atoi(en):1216;
                    if(st_m2<1)st_m2=32; if(st_n2<nt_sz)st_n2=nt_sz;}
                int cap2=R-1; if(cap2<1)cap2=1; int ch=st_on2?(st_m2>cap2?cap2:st_m2):chunk; if(ch>M)ch=M; if(ch<1)ch=1;
                int nsw=st_on2?((st_n2+nt_sz-1)/nt_sz)*nt_sz:0;
                int nmt=(M+ch-1)/ch; int P=0; size_t ccsum=0;
                for(int ns=0;ns<w->Sn;ns++){int Nc=(N-ns*NMAX<NMAX)?(N-ns*NMAX):NMAX,NN=Nc/nt_sz;
                    int t0=(int)((long)i*NN/nc),t1=(int)((long)(i+1)*NN/nc),cols=(t1-t0)*nt_sz;
                    int eff_cols = (cols>0)?cols:nt_sz;
                    int nsw2=nsw?nsw:eff_cols; int nnsub=(eff_cols+nsw2-1)/nsw2;
                    P += nnsub*nmt;
                    ccsum += (size_t)M*eff_cols*4;
                }
                if(ccsum>maxout)maxout=ccsum;
                size_t needrc=(size_t)P*REGCMD_I8_N*4, needtk=(size_t)P*sizeof(struct rknpu_task);
                if(c->mrc[i].size<needrc){ bdestroy(fd,&c->mrc[i]); c->mrc[i]=bcreate(fd,needrc,0x403,c->dom_active);
                    if(!c->mrc[i].cpu){ fprintf(stderr,"[ork] ERROR: pref mrc[%d] alloc failed (%zu)\n",i,needrc); return -1; } c->mwarm[i]=0; }
                if(c->mtk[i].size<needtk){ bdestroy(fd,&c->mtk[i]); c->mtk[i]=bcreate(fd,needtk,0x40b,c->dom_active);
                    if(!c->mtk[i].cpu){ fprintf(stderr,"[ork] ERROR: pref mtk[%d] alloc failed (%zu)\n",i,needtk); return -1; } }
            }
        }
        if(use_chain_ksplit){
            /* AF holds all M*K of A. CC holds a batch of K-slice partials: cap to the SMALLER of the
             * budget and the total partials this core would produce (so small matmuls don't over-alloc).
             * Programs/batch bounded by pd[] (256) and the mrc/mtk capacity (grown below). */
            /* AF holds per-program gathered A tiles [mco][Kp]; a full batch can re-gather A per
             * N-slice, so the worst case is Sn*M*K (sum of Kp over a batch == K when whole-K). */
            size_t afull=(size_t)w->Sn*M*K; if(afull>maxaf)maxaf=afull;
            size_t total_part=0;
            for(int ns=0;ns<w->Sn;ns++){int Nc=(N-ns*NMAX<NMAX)?(N-ns*NMAX):NMAX,NN=Nc/nt_sz;
                int t0=(int)((long)i*NN/nc),t1=(int)((long)(i+1)*NN/nc),cols=(t1-t0)*nt_sz; if(cols<=0)continue;
                total_part+=(size_t)w->Sk*M*cols*4;   /* Sk slices x M rows x this core's cols */
            }
            size_t want=(size_t)ks_ccbudget; if(total_part<want)want=total_part;
            if(want>maxout)maxout=want;
            /* grow regcmd/task buffers to hold up to 256 chained programs */
            size_t needrc=(size_t)256*REGCMD_I8_N*4, needtk=(size_t)256*sizeof(struct rknpu_task);
            if(c->mrc[i].size<needrc){ bdestroy(fd,&c->mrc[i]); c->mrc[i]=bcreate(fd,needrc,0x403,c->dom_active);
                if(!c->mrc[i].cpu){ fprintf(stderr,"[ork] ERROR: ksplit mrc[%d] alloc failed (%zu)\n",i,needrc); return -1; } c->mwarm[i]=0; }
            if(c->mtk[i].size<needtk){ bdestroy(fd,&c->mtk[i]); c->mtk[i]=bcreate(fd,needtk,0x40b,c->dom_active);
                if(!c->mtk[i].cpu){ fprintf(stderr,"[ork] ERROR: ksplit mtk[%d] alloc failed (%zu)\n",i,needtk); return -1; } }
        }
        core_maxout[i] = maxout;
        if(c->mccsz[i]<maxout){
            bdestroy(fd,&c->mcc[i]);
            c->mcc[i]=bcreate(fd,maxout,0x403,c->dom_active);
            c->mccsz[i]=maxout;
            c->mwarm[i]=0;
            c->mwarm[0]=0;
            if(!c->mcc[i].cpu){
                fprintf(stderr, "[ork] ERROR: failed to allocate multi-core output buffer (size=%zu) for core %d\n", maxout, i);
                return -1;
            }
        }
        if(c->maf[i].size<maxaf){
            bdestroy(fd,&c->maf[i]);
            c->maf[i]=bcreate(fd,maxaf,0x403,c->dom_active);
            if(!c->maf[i].cpu){
                fprintf(stderr, "[ork] ERROR: failed to allocate multi-core activation buffer maf[%d] (size=%zu, IOMMU full?)\n", i, maxaf);
                return -1;
            }
        }
    }

    int reps = c->mwarm[0] ? 1 : 2;

    size_t need=(size_t)M*w->N*4;
    if(c->cressz<need){c->cres=realloc(c->cres,need);c->cressz=need;} memset(c->cres,0,need);
    struct mcw args[ORK_MAXCORE]; int rc=0;
    for(int i=0;i<nc;i++) args[i]=(struct mcw){c,i,nc,dt,M,A,w,c->cres,0,reps,core_maxout[i],use_chain_pref,use_chain_ksplit};
    npu_pool_ensure(c);
    c->mc_error = 0;
    if(nc>1) pthread_barrier_init(&c->b_ioctl, NULL, nc);
    const double t1=ork_now_us();
    pthread_mutex_lock(&c->pmu); c->pjob=args; c->pjob_nc=nc; c->pjob_fn=mcworker; c->pjob_stride=sizeof(struct mcw); c->pdone=0; c->pgen++; pthread_cond_broadcast(&c->pgo); pthread_mutex_unlock(&c->pmu);
    mcworker(&args[0]);                                   /* core 0 on the calling thread */
    pthread_mutex_lock(&c->pmu); while(c->pdone<nc-1) pthread_cond_wait(&c->pdn,&c->pmu); pthread_mutex_unlock(&c->pmu);
    if(nc>1) pthread_barrier_destroy(&c->b_ioctl);
    for(int i=0;i<nc;i++){ if(args[i].rc) rc=-1; }
    if(rc) return -1;
    const double t2=ork_now_us();
    memcpy(C,c->cres,need);
    const double t3=ork_now_us();
    g_rt_setup+=t1-ts; g_rt_submit+=t2-t1; g_rt_copy+=t3-t2; g_rt_n++;
    return 0;
}

/* ---- int4 (W4A4) multi-core: WIDE submits with COLUMN-split. Each core owns a contiguous range
 * of 64-wide N-blocks within each N-slice and computes them in ONE wide submit per K-slice (not one
 * per 64-tile) — so a decode matmul is ~nc·Sk·Sn submits, not Sn·64-tiles. Per-core buffers,
 * core_mask=1<<i, all subcore_task[] populated, NO per-submit RESET (the dtype-switch RESET is done
 * once in run_i4_mc; concurrent RESET / a submit-storm is the documented board-hang). nc==1 = serial
 * (one core, whole width). Writes disjoint columns of C, no lock. ---- */
struct i4mcw { ork_npu *c; int core, nc, M; ork_w *w; const int8_t *A; int32_t *C; int rc; };
static void *i4_mcworker(void *vp){
    struct i4mcw *a=vp; ork_npu *c=a->c; int i=a->core, nc=a->nc, M=a->M, fd=c->fd;
    pin_big_core(i);                           /* core 0 = calling thread, 1.. = spawned workers */
    ork_w *w=a->w; int K=w->K, N=w->N, KS=ORK_I4_KS, NMAX=c->soc->nmax;
    struct buf *RC=&c->mrc[i], *AF=&c->maf[i], *O=&c->mcc[i]; a->rc=0;
    int32_t *acc=malloc((size_t)M*NMAX*4);
    if(!acc){
        a->rc=-1; c->mc_error=1;
    }
    for(int ns=0;ns<w->Sn;ns++){
        int n0=ns*NMAX, Nc=(N-n0<NMAX)?(N-n0):NMAX, NB=Nc/64;
        int b0=(int)((long)i*NB/nc), b1=(int)((long)(i+1)*NB/nc);
        int active = (b1 > b0);
        int ci0 = active ? b0*64 : 0, Ncore = active ? (b1-b0)*64 : 64;
        if(acc) memset(acc, 0, (size_t)M*Ncore*4);
        for(int ks=0;ks<w->Sk;ks++){
            int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;
            int chunk_M = 16;
            for (int m0 = 0; m0 < M; m0 += chunk_M) {
                int cur_chunk = (M - m0 < chunk_M) ? (M - m0) : chunk_M;
                if (!c->mc_error) {
                    if (cur_chunk > 1) {
                        for (int m = 0; m < cur_chunk; m++) {
                            tile_i4_Aslice((uint8_t*)AF->cpu + (size_t)m * (Kp / 2), a->A + (size_t)(m0 + m) * K, k0, Kp);
                        }
                    } else {
                        tile_i4_Aslice(AF->cpu, a->A + (size_t)m0 * K, k0, Kp);
                    }
                    bsync(fd,AF,RKNPU_MEM_SYNC_TO_DEVICE);
                }
                uint64_t wbase=w->Bb[(size_t)ns*w->Sk+ks].dma + (uint64_t)(active?b0:0)*Kp*32;  /* Kp*32 B per N-block */
                struct rknpu_task *tk_arr = c->mtk[i].cpu;
                memset(tk_arr, 0, cur_chunk * sizeof(struct rknpu_task));
                for(int m=0; m<cur_chunk; m++) {
                    uint32_t rc[REGCMD_I4_N];
                    uint32_t aA = (uint32_t)AF->dma + m * (Kp / 2);
                    uint32_t aC = (uint32_t)O->dma + m * (Ncore * 2);
                    synth_i4(rc, 1, Kp, Ncore, aA, (uint32_t)wbase, aC);
                    if (validate_regcmd("i4_mcworker", c, rc, REGCMD_I4_N, w, NULL, 0)) {
                        a->rc = -1; c->mc_error = 1;
                    }
                    if (m < cur_chunk - 1) {
                        uint64_t next_dma = c->mrc[i].dma + (m + 1) * REGCMD_I4_N * 4;
                        rc[216] = 0x0010 | ((next_dma & 0xffff) << 16);
                        rc[217] = (0x0101 << 16) | ((next_dma >> 16) & 0xffff);
                        rc[218] = 0x0014 | (0x0037 << 16);
                        rc[219] = (0x0101 << 16) | (0);
                    } else {
                        rc[216] = 0; rc[217] = 0; rc[218] = 0x00000014; rc[219] = 0x01010000;
                    }
                    memcpy((char*)RC->cpu + m * REGCMD_I4_N * 4, rc, sizeof(rc));
                    tk_arr[m].enable_mask = 0xd;
                    tk_arr[m].int_mask = 0x300;
                    tk_arr[m].int_clear = 0x1ffff;
                    tk_arr[m].regcfg_amount = 116;
                    tk_arr[m].regcmd_addr = c->mrc[i].dma + m * REGCMD_I4_N * 4;
                }
                if (!c->mc_error) {
                    bsync(fd, RC, RKNPU_MEM_SYNC_TO_DEVICE);
                    bsync(fd, &c->mtk[i], RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
                }
                int reps=c->mwarm[i]?1:2;
                if (c->mc_error) {
                    a->rc = -1;
                    free(acc);
                    return NULL;
                }
                struct rknpu_submit sub;
                memset(&sub, 0, sizeof sub);
                sub.flags = 0x5;
                sub.task_number = cur_chunk;
                sub.task_obj_addr = c->mtk[i].obj;
                sub.fence_fd = -1;
                sub.core_mask = 1u << i;
                sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, cur_chunk};
                for (int rep = 0; rep < reps; rep++) {
                    int last = (rep == reps - 1);
                    sub.timeout = 60000;
                    if (rknpu_submit_ioctl(fd, &sub, w->domain)) {
                        if (last) {
                            a->rc = -1;
                            free(acc);
                            return NULL;
                        }
                        continue;
                    }
                    bsync(fd, O, RKNPU_MEM_SYNC_FROM_DEVICE);
                }
                c->mwarm[i]=1;
                if (active && acc) {
                    int16_t*o=O->cpu;
                    if(cur_chunk>1) {
                        for(int m=0;m<cur_chunk;m++){
                            for(int col=0;col<Ncore;col++){
                                size_t o_idx = (size_t)m * Ncore + col;
                                acc[(m0 + m)*Ncore + col] += o[o_idx];
                            }
                        }
                    } else {
                        int32_t*acc_ptr = &acc[m0 * Ncore];
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                        int col = 0;
                        for (; col <= Ncore - 16; col += 16) {
                            int32x4_t vacc0 = vld1q_s32(&acc_ptr[col]);
                            int32x4_t vacc1 = vld1q_s32(&acc_ptr[col + 4]);
                            int32x4_t vacc2 = vld1q_s32(&acc_ptr[col + 8]);
                            int32x4_t vacc3 = vld1q_s32(&acc_ptr[col + 12]);

                            int16x8_t vo16_0 = vld1q_s16(&o[col]);
                            int16x8_t vo16_1 = vld1q_s16(&o[col + 8]);

                            vacc0 = vaddw_s16(vacc0, vget_low_s16(vo16_0));
                            vacc1 = vaddw_high_s16(vacc1, vo16_0);
                            vacc2 = vaddw_s16(vacc2, vget_low_s16(vo16_1));
                            vacc3 = vaddw_high_s16(vacc3, vo16_1);

                            vst1q_s32(&acc_ptr[col], vacc0);
                            vst1q_s32(&acc_ptr[col + 4], vacc1);
                            vst1q_s32(&acc_ptr[col + 8], vacc2);
                            vst1q_s32(&acc_ptr[col + 12], vacc3);
                        }
                        for (; col < Ncore; col++) {
                            acc_ptr[col] += o[col];
                        }
#else
                        for(int nt=0;nt<Ncore/8;nt++){
                            for(int nl=0;nl<8;nl++){
                                acc_ptr[nt*8+nl] += o[nt*8+nl];
                            }
                        }
#endif
                    }
                }
            }
        }
        if (active && acc) {
            for(int m=0;m<M;m++){
                for(int z=0;z<Ncore;z++){
                    a->C[(size_t)m*N + n0+ci0+z] = acc[m*Ncore + z];
                }
            }
        }
    }
    free(acc); return NULL;
}
static int run_i4_mc(ork_npu *c,ork_w *w,int M,const int8_t *A,int32_t *C,int nc){
    int fd=c->fd;
    if(nc>c->soc->cores)nc=c->soc->cores;
    if(nc>ORK_MAXCORE)nc=ORK_MAXCORE;
    if(nc<1)nc=1;
    if(c->last_dt!=DT_I4){ act(fd,RKNPU_ACT_RESET,0); for(int i=0;i<ORK_MAXCORE;i++){c->mwarm[i]=0;c->mccsz[i]=0;} c->last_dt=DT_I4; }
    if(mc_ensure(c,nc)) return -1;
    size_t osz=(size_t)c->soc->nmax*(M > 1 ? 2 * M : 1)*2;        /* per-core output: up to a full N-slice of int16 */
    for(int i=0;i<nc;i++){ if(c->mccsz[i]<osz){ bdestroy(fd,&c->mcc[i]); c->mcc[i]=bcreate(fd,osz,0x403,c->dom_active); c->mccsz[i]=osz; c->mwarm[i]=0; if(!c->mcc[i].cpu){fprintf(stderr, "[ork] ERROR: failed to allocate multi-core output mcc[%d] (size=%zu)\n", i, osz);return -2;} } }
    size_t asz=(size_t)M*ORK_I4_KS/2;
    if(asz < (size_t)4*32768*2) asz=(size_t)4*32768*2;
    for(int i=0;i<nc;i++){ if(c->maf[i].size<asz){ bdestroy(fd,&c->maf[i]); c->maf[i]=bcreate(fd,asz,0x403,c->dom_active); if(!c->maf[i].cpu){fprintf(stderr, "[ork] ERROR: failed to allocate multi-core activation maf[%d] (size=%zu)\n", i, asz);return -2;} } }
    /* Zero-copy chaining (the portable half of the int8 zero-copy design — perf-neutral, correctness
     * for DMA pipelines). int4 can't read/write the caller's A/C *directly* (A needs the nibble re-tile,
     * the int16 hardware output needs widening to the int32 C), so the internal AF/O scratch is
     * mandatory. But when A/C are ork_dma_alloc buffers, int4 must still observe coherency so it
     * composes with up/downstream NPU ops: invalidate a DMA-resident A once before the CPU re-tiles it
     * (see NPU-produced input), and flush a DMA-resident C once after the CPU writes it (downstream NPU
     * sees the output). dirty-line eviction otherwise corrupts a chained op — the same hazard the int8
     * output zero-copy fix addressed. */
    struct buf *abuf=dma_find(c,A), *cbuf=dma_find(c,C);
    if(abuf) bsync(fd,abuf,RKNPU_MEM_SYNC_FROM_DEVICE);   /* CPU re-tile must see an NPU-produced A */
    struct i4mcw args[ORK_MAXCORE]; pthread_t th[ORK_MAXCORE];
    for(int i=0;i<nc;i++) args[i]=(struct i4mcw){c,i,nc,M,w,A,C,0};
    c->mc_error = 0;
    for(int i=1;i<nc;i++) pthread_create(&th[i],NULL,i4_mcworker,&args[i]);
    i4_mcworker(&args[0]);                                /* core 0 on the calling thread */
    for(int i=1;i<nc;i++) pthread_join(th[i],NULL);
    for(int i=0;i<nc;i++) if(args[i].rc) return -1;
    if(cbuf) bsync(fd,cbuf,RKNPU_MEM_SYNC_TO_DEVICE);     /* flush host-written C for a downstream NPU op */
    return 0;
}

struct i4gw { ork_npu *c; int core, nc, M; ork_w *w; const int8_t *A; const float *aS,*bS; float *Cf; int rc; };
static void *i4_mcworker_g(void *vp){
    struct i4gw *a=vp; ork_npu *c=a->c; int i=a->core, nc=a->nc, M=a->M, fd=c->fd;
    pin_big_core(i);                           /* core 0 = calling thread, 1.. = spawned workers */
    ork_w *w=a->w; int K=w->K,N=w->N,G=w->gsize,NMAX=c->soc->nmax,Sk=w->Sk;
    struct buf *RC=&c->mrc[i],*AF=&c->maf[i],*O=&c->mcc[i]; a->rc=0;
    float *acc=malloc((size_t)NMAX*4);
    if(!acc){
        a->rc=-1; c->mc_error=1;
    }
    for(int ns=0;ns<w->Sn;ns++){
        int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NB=Nc/64;
        int b0=(int)((long)i*NB/nc),b1=(int)((long)(i+1)*NB/nc);
        int active=(b1>b0);
        int ci0=active?b0*64:0,Ncore=active?(b1-b0)*64:64;
        for(int m=0;m<M;m++){ const int8_t*Arow=a->A+(size_t)m*K;
            if (active && acc) {
                for(int z=0;z<Ncore;z++)acc[z]=0;
            }
            for(int g=0;g<Sk;g++){
                if (!c->mc_error) {
                    tile_i4_Aslice(AF->cpu,Arow,g*G,G); bsync(fd,AF,RKNPU_MEM_SYNC_TO_DEVICE);
                }
                uint64_t wbase=w->Bb[(size_t)ns*Sk+g].dma+(uint64_t)(active?b0:0)*G*32;
                uint32_t rc[REGCMD_I4_N]; synth_i4(rc,1,G,Ncore,(uint32_t)AF->dma,(uint32_t)wbase,(uint32_t)O->dma);
                if (validate_regcmd("i4_mcworker_g", c, rc, REGCMD_I4_N, w, NULL, 0)) {
                    a->rc = -1; c->mc_error = 1;
                }
                if (!c->mc_error) {
                    memcpy(RC->cpu,rc,sizeof rc); bsync(fd,RC,RKNPU_MEM_SYNC_TO_DEVICE);
                }
                int reps=c->mwarm[i]?1:2;
                if (c->mc_error) {
                    a->rc = -1;
                    free(acc);
                    return NULL;
                }
                struct rknpu_submit sub;
                memset(&sub, 0, sizeof sub);
                sub.flags = 0x5;
                sub.task_number = 1;
                sub.task_obj_addr = c->mtk[i].obj;
                sub.fence_fd = -1;
                sub.core_mask = 1u << i;
                sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, 1};
                for (int rep = 0; rep < reps; rep++) {
                    int last = (rep == reps - 1);
                    sub.timeout = 60000;
                    if (rknpu_submit_ioctl(fd, &sub, w->domain)) {
                        if (last) {
                            a->rc = -1;
                            free(acc);
                            return NULL;
                        }
                        continue;
                    }
                    bsync(fd, O, RKNPU_MEM_SYNC_FROM_DEVICE);
                }
                c->mwarm[i]=1;
                if (active && acc) {
                    int16_t*o=O->cpu; float as=a->aS[(size_t)m*Sk+g];
                    const float *bS_ptr = a->bS + (size_t)g*N + n0+ci0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                    float32x4_t vas = vdupq_n_f32(as);
                    int col = 0;
                    for (; col <= Ncore - 16; col += 16) {
                        int16x8_t vo16_0 = vld1q_s16(&o[col]);
                        int16x8_t vo16_1 = vld1q_s16(&o[col + 8]);

                        int32x4_t vo32_0_low  = vmovl_s16(vget_low_s16(vo16_0));
                        int32x4_t vo32_0_high = vmovl_s16(vget_high_s16(vo16_0));
                        float32x4_t vo_f_0_low  = vcvtq_f32_s32(vo32_0_low);
                        float32x4_t vo_f_0_high = vcvtq_f32_s32(vo32_0_high);

                        int32x4_t vo32_1_low  = vmovl_s16(vget_low_s16(vo16_1));
                        int32x4_t vo32_1_high = vmovl_s16(vget_high_s16(vo16_1));
                        float32x4_t vo_f_1_low  = vcvtq_f32_s32(vo32_1_low);
                        float32x4_t vo_f_1_high = vcvtq_f32_s32(vo32_1_high);

                        float32x4_t vbS_0_low  = vld1q_f32(&bS_ptr[col]);
                        float32x4_t vbS_0_high = vld1q_f32(&bS_ptr[col + 4]);
                        float32x4_t vbS_1_low  = vld1q_f32(&bS_ptr[col + 8]);
                        float32x4_t vbS_1_high = vld1q_f32(&bS_ptr[col + 12]);

                        float32x4_t vacc_0_low  = vld1q_f32(&acc[col]);
                        float32x4_t vacc_0_high = vld1q_f32(&acc[col + 4]);
                        float32x4_t vacc_1_low  = vld1q_f32(&acc[col + 8]);
                        float32x4_t vacc_1_high = vld1q_f32(&acc[col + 12]);

                        float32x4_t vprod_0_low  = vmulq_f32(vbS_0_low,  vo_f_0_low);
                        float32x4_t vprod_0_high = vmulq_f32(vbS_0_high, vo_f_0_high);
                        float32x4_t vprod_1_low  = vmulq_f32(vbS_1_low,  vo_f_1_low);
                        float32x4_t vprod_1_high = vmulq_f32(vbS_1_high, vo_f_1_high);

                        vacc_0_low  = vmlaq_f32(vacc_0_low,  vprod_0_low,  vas);
                        vacc_0_high = vmlaq_f32(vacc_0_high, vprod_0_high, vas);
                        vacc_1_low  = vmlaq_f32(vacc_1_low,  vprod_1_low,  vas);
                        vacc_1_high = vmlaq_f32(vacc_1_high, vprod_1_high, vas);

                        vst1q_f32(&acc[col],      vacc_0_low);
                        vst1q_f32(&acc[col + 4],  vacc_0_high);
                        vst1q_f32(&acc[col + 8],  vacc_1_low);
                        vst1q_f32(&acc[col + 12], vacc_1_high);
                    }
                    for (; col < Ncore; col++) {
                        acc[col] += as * bS_ptr[col] * (float)o[col];
                    }
#else
                    for(int col=0;col<Ncore;col++) acc[col]+= as * bS_ptr[col] * (float)o[col];
#endif
                }
            }
            if (active && acc) {
                for(int z=0;z<Ncore;z++) a->Cf[(size_t)m*N + n0+ci0+z]=acc[z];
            }
        }
    }
    free(acc); return NULL;
}
int ork_mm_run_i4_grouped(ork_npu *c,ork_w *w,int M,const int8_t *A,const float *aScale,const float *bScale,float *C){
    if(!w||w->dtype!=DT_I4||!w->gsize) return -1;
    if(check_overlap("ork_mm_run_i4_grouped", (uintptr_t)A, (uintptr_t)A + (size_t)M * w->K, (uintptr_t)C, (uintptr_t)C + (size_t)M * w->N * 4)) return -1;
    int fd=c->fd, NB=w->N/64, nc=budget(c, M);
    if(nc>NB)nc=NB;
    if(nc>c->soc->cores)nc=c->soc->cores;
    if(nc>ORK_MAXCORE)nc=ORK_MAXCORE;
    if(nc<1)nc=1;
    static int logged = 0;
    if (!logged) {
        fprintf(stderr, "[ork] ork_mm_run_i4_grouped: M=%d, N=%d, K=%d, nc=%d, NB=%d\n", M, w->N, w->K, nc, NB);
        logged = 1;
    }
    if(c->last_dt!=DT_I4){ act(fd,RKNPU_ACT_RESET,0); for(int i=0;i<ORK_MAXCORE;i++){c->mwarm[i]=0;c->mccsz[i]=0;} c->last_dt=DT_I4; }
    if(mc_ensure(c,nc)) return -1;
    size_t osz=(size_t)c->soc->nmax*2;
    for(int i=0;i<nc;i++){ if(c->mccsz[i]<osz){ bdestroy(fd,&c->mcc[i]); c->mcc[i]=bcreate(fd,osz,0x403,c->dom_active); c->mccsz[i]=osz; c->mwarm[i]=0; if(!c->mcc[i].cpu){fprintf(stderr, "[ork] ERROR: failed to allocate grouped multi-core output mcc[%d] (size=%zu)\n", i, osz);return -2;} } }
    struct i4gw args[ORK_MAXCORE]; pthread_t th[ORK_MAXCORE];
    for(int i=0;i<nc;i++) args[i]=(struct i4gw){c,i,nc,M,w,A,aScale,bScale,C,0};
    c->mc_error = 0;
    for(int i=1;i<nc;i++) pthread_create(&th[i],NULL,i4_mcworker_g,&args[i]);
    i4_mcworker_g(&args[0]);
    for(int i=1;i<nc;i++) pthread_join(th[i],NULL);
    for(int i=0;i<nc;i++) if(args[i].rc) return -1;
    return 0;
}

static int run(ork_npu *c,ork_w *w,int M,const void *A,void *C){
    /* multi-domain residence: swap in this domain's scratch (regcmd/task/Af/Cc/mc-*) so the submit's
     * buffers all live in the weight's domain (c->dom_active), and make any lazy scratch bcreate below
     * land there too. Submits stamp their own iommu_domain_id from w->domain (per-submit, no global).
     * No-op for the common single-domain case (w->domain==0, dom_active==0). */
    if(w->domain!=c->dom_active || (w->domain!=0 && !c->dom_save)) dom_activate(c,w->domain);
    /* auto-tuner: pick cores ≤ budget, capped so each gets ≥2 N-tiles (tiny matmuls don't pay the
     * multi-core spawn). budget defaults to all soc cores; ORK_NPU_MC / set_core_budget cap it. */
    /* int8 M=1 decode: use the multi-core budget (split N across cores) — validated +40% end-to-end, every
     * shape benefits (in-model sweep monotonic). budget(c,2) skips the M==1 single-core default + honors
     * ORK_NPU_MC. fp16/int4 M==1 keep single-core (budget(c,1)==1). NN<nc*2 shrink below guards tiny int8 N. */
    int b=(M==1 && w->dtype==DT_I8) ? budget(c,2) : budget(c, M), cores=c->soc->cores, NN=w->N/(w->dtype?32:16);
    int nc=b<cores?b:cores; if(nc>NN)nc=NN; while(nc>1 && NN<nc*2)nc--;
    /* ORK_MC1=1: route single-core (nc==1) through run_multicore so it uses the CHAINED prefill path
     * (M-tiles PC-chained into ~1 submit) instead of the per-tile single-core path (~19 submits). For
     * measuring chained-ork-1core vs rknn-1core apples-to-apples (rknn chains its M-tiles in 1 submit). */
    if(nc>1) return run_multicore(c,w,M,A,C,nc);
    { static int mc1=-1; if(mc1<0){const char*e=getenv("ORK_MC1"); mc1=e?atoi(e):0;}
      if(mc1 && M>1 && c->soc->cores>=1) return run_multicore(c,w,M,A,C,1); }
    pin_big_core(0);                                   /* single-core path also runs on the calling thread */
    int fd=c->fd,K=w->K,N=w->N, dt=w->dtype, NMAX=c->soc->nmax, CBUF=c->soc->cbuf_elems;
    if(dt==DT_F16 && CBUF>32768) CBUF=32768;   /* int8-only cbuf raise; fp16 keeps its validated 32768 tiling (see mcworker) */
    int KS=dt ? int8_ks(c) : c->soc->ks, RB=dt?2*CBUF:CBUF;     /* rows budget: int8 packs 2x rows/CBUF */
    /* entering int8 mode wedges the first submit unless the NPU is reset first (fp16 never
     * wedges — it cold-starts stale, which the warmup handles). Reset only when switching INTO
     * int8 — keeps fp16-only contexts free of any reset/log. Then re-warm on a fresh buffer. */
    if(dt!=c->last_dt){ if(dt==DT_I8 && !ORK_I8_LIVE(c->last_dt)) act(fd,RKNPU_ACT_RESET,0); c->warmed=0; c->ccsz=0; c->last_dt=dt; }
    size_t need=(size_t)M*N*4;                         /* output is fp32 or int32 (both 4 bytes) */
    if(c->cressz<need){c->cres=realloc(c->cres,need);c->cressz=need;}
    memset(c->cres,0,need);
    size_t maxout=0, maxaf=0;
    for(int k0=0;k0<K;k0+=KS){
        int Kp=(K-k0<KS)?(K-k0):KS;
        int sd=dt?(Kp==1024||Kp==512):((Kp&(Kp-1))==0 && Kp<2048);
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
    if(c->ccsz<maxout){bdestroy(fd,&c->Cc);c->Cc=bcreate(fd,maxout,0x403,c->dom_active);c->ccsz=maxout;c->warmed=0; if(!c->Cc.cpu){fprintf(stderr, "[ork] ERROR: failed to allocate single-core/pre-core output buffer Cc (size=%zu, IOMMU full?)\n", maxout);return -1;}}
    if(c->Af.size<maxaf){
        bdestroy(fd,&c->Af);
        c->Af=bcreate(fd,maxaf,0x403,c->dom_active);
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
        struct buf *abuf=dma_find(c,A);   /* INPUT zero-copy: validated correct, default on */
        struct buf *cbuf=(getenv("ORK_ZC_OUT"))?dma_find(c,C):NULL;
        if(abuf) bsync(fd,abuf,RKNPU_MEM_SYNC_TO_DEVICE);   /* flush the producer's CPU writes once */
        if(cbuf) {
            bsync(fd,cbuf,RKNPU_MEM_SYNC_TO_DEVICE);   /* clean dirty CPU cache lines so they don't evict over NPU output */
            c->warmed=0;             /* re-warm the fresh output buffer (necessary but not sufficient) */
        }
        for(int ns=0;ns<w->Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
            uint64_t wbase=w->Bf[ns].dma;                  /* full-K weight, whole N-slice (single core) */
            for(int m0=0;m0<M;m0+=chunk){int mc=(M-m0<chunk)?(M-m0):chunk; if(mc<=0)continue;
                double _tc0=ork_now_us();
                uint32_t adma;
                if(abuf){ adma=(uint32_t)(abuf->dma + ((const char*)A-(const char*)abuf->cpu) + (size_t)m0*K); }
                else { int8_t*ad=c->Af.cpu; const int8_t*Ai=A; for(int r=0;r<mc;r++)for(int j=0;j<K;j++) ad[(size_t)r*K+j]=Ai[(size_t)(m0+r)*K+j];
                       bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE); adma=(uint32_t)c->Af.dma; }
                double _ts0=ork_now_us(); g_mc_copy[0]+=_ts0-_tc0;
                uint32_t cdma=cbuf?(uint32_t)(cbuf->dma + ((const char*)C-(const char*)cbuf->cpu) + (size_t)m0*N*4):(uint32_t)c->Cc.dma;
                uint32_t rc[REGCMD_N]; synth_i8(rc,mc,Kp,Nc,adma,(uint32_t)wbase,cdma,sched,CBUF,cbuf?N:Nc);
                if (validate_regcmd("run_fullk_dec", c, rc, REGCMD_N, w, NULL, 0)) return -1;
                memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
                if(submit1(c)) return -1;
                double _ta0=ork_now_us(); g_mc_sub[0]+=_ta0-_ts0;
                
                /* For output zero copy, the NPU writes directly to the user-provided C buffer.
                 * We MUST invalidate the CPU cache here so the host reads the fresh NPU output instead of stale cache lines. */
                if(cbuf) bsync(fd,cbuf,RKNPU_MEM_SYNC_FROM_DEVICE);
                
                if(!cbuf){ int32_t*cc=c->Cc.cpu,*cr=c->cres; for(int r=0;r<mc;r++)for(int n=0;n<Nc;n++) cr[(size_t)(m0+r)*N+(n0+n)]=cc[(size_t)r*Nc+n]; }
                g_mc_acc[0]+=ork_now_us()-_ta0; g_mc_n[0]++;
            }
        }
        if(cbuf){
            bsync(fd,cbuf,RKNPU_MEM_SYNC_FROM_DEVICE);
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
    for(int ns=0;ns<w->Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<w->Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;
        int sched=dt?(Kp==1024||Kp==512):((Kp&(Kp-1))==0 && Kp<2048), R=RB/Kp; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; } int chunk=sched?4*R:((RB/2)/Kp); if(chunk<1)chunk=1;   /* fp16 M-scheduler (sched=1) miscomputes >8 rows at Kp>=2048 (validated mc<=8 OK / mc>=9 garbage) — gate it OFF for Kp>=2048 (mc=(RB/2)/Kp=8, correct), matching mcworker. Was missing the Kp<2048 guard -> fp16 single-core K>=2048 produced garbage for M>8. */
        struct buf*Bb=&w->Bb[(size_t)ns*w->Sk+ks];
        for(int m0=0;m0<M;m0+=chunk){int mc=(M-m0<chunk)?(M-m0):chunk; if(mc<=0)continue;
            double _tc0=ork_now_us();
            if(dt==DT_F16){ f16*ad=c->Af.cpu; const f16*Af=A; for(int r=0;r<mc;r++)for(int j=0;j<Kp;j++) ad[(size_t)r*Kp+j]=Af[(size_t)(m0+r)*K+k0+j]; }
            else { int8_t*ad=c->Af.cpu; const int8_t*Ai=A; for(int r=0;r<mc;r++)for(int j=0;j<Kp;j++) ad[(size_t)r*Kp+j]=Ai[(size_t)(m0+r)*K+k0+j]; }
            bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
            double _ts0=ork_now_us(); g_mc_copy[0]+=_ts0-_tc0;
            uint32_t rc[REGCMD_N];   /* REGCMD_N == REGCMD_I8_N == 224 */
            if(dt==DT_F16) synth   (rc,mc,Kp,Nc,(uint32_t)c->Af.dma,(uint32_t)Bb->dma,(uint32_t)c->Cc.dma,sched,CBUF);
            else           synth_i8(rc,mc,Kp,Nc,(uint32_t)c->Af.dma,(uint32_t)Bb->dma,(uint32_t)c->Cc.dma,sched,CBUF,Nc);
            if (validate_regcmd("run_loop", c, rc, REGCMD_N, w, NULL, 0)) return -1;
            memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
            if(submit1(c)) return -1;
            double _ta0=ork_now_us(); g_mc_sub[0]+=_ta0-_ts0;
            if(dt==DT_F16){ float  *cc=c->Cc.cpu,*cr=c->cres; for(int r=0;r<mc;r++)for(int n=0;n<Nc;n++) cr[(size_t)(m0+r)*N+(n0+n)]+=cc[(size_t)r*Nc+n]; }
            else { int32_t*cc=c->Cc.cpu,*cr=c->cres; for(int r=0;r<mc;r++)for(int n=0;n<Nc;n++) cr[(size_t)(m0+r)*N+(n0+n)]+=cc[(size_t)r*Nc+n]; }
            g_mc_acc[0]+=ork_now_us()-_ta0; g_mc_n[0]++;
        }
      }
    }
    memcpy(C,c->cres,need); return 0;
}
int ork_mm_run   (ork_npu *c,ork_w *w,int M,const f16    *A,float   *C){
    if(w->dtype!=DT_F16)return -1;
    if(check_overlap("ork_mm_run", (uintptr_t)A, (uintptr_t)A + (size_t)M * w->K * 2, (uintptr_t)C, (uintptr_t)C + (size_t)M * w->N * 4)) return -1;
    return run(c,w,M,A,C);
}
int ork_mm_run_i8(ork_npu *c,ork_w *w,int M,const int8_t *A,int32_t *C){
    if(w->dtype!=DT_I8) return -1;
    if(check_overlap("ork_mm_run_i8", (uintptr_t)A, (uintptr_t)A + (size_t)M * w->K, (uintptr_t)C, (uintptr_t)C + (size_t)M * w->N * 4)) return -1;
    if(!g_ork_prof) return run(c,w,M,A,C);
    double t0=ork_now_us(); int r=run(c,w,M,A,C); g_prof_i8_us+=ork_now_us()-t0; g_prof_i8_calls++; return r;
}

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
static double ork_now_us(void);
int ork_npu_probe_mtile_i8(ork_npu *c,int M,int K,int N,int mode,
                           const int8_t *A,const int8_t *B,int32_t *C,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64) return -2;
    struct buf W=bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    int NN=N/32,KT=K/32; int8_t*bb=W.cpu;     /* int8 tile layout [Ntile][Ktile][32][32], full K */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)];
    bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=bcreate(fd,(size_t)M*N*4,0x403,-1); if(!O.cpu){bdestroy(fd,&W);return -2;}
    int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I8_N];
    synth_i8(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    if(mode==1){   /* override with the rkllm-captured M-tile program */
        setr(rc,REGCMD_I8_N,0x201,0x1010,0x20);
        setr(rc,REGCMD_I8_N,0x201,0x1044,(K/64)*M);
        setr(rc,REGCMD_I8_N,0x201,0x107c,4*M);
        int mg=(M+7)/8; int v=0xb1-0x0f*(mg-1); if(v<0x1b)v=0x1b;
        setr(rc,REGCMD_I8_N,0x201,0x1040,v);
        setr(rc,REGCMD_I8_N,0x201,0x1020,(M<<16)|1);
        setr(rc,REGCMD_I8_N,0x201,0x1084,(M<<16)|1);
    }
    struct buf extra[2] = {W, O};
    if (validate_regcmd("probe_mtile_i8", c, rc, REGCMD_I8_N, NULL, extra, 2)) { bdestroy(fd,&W); bdestroy(fd,&O); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0;
    for(int rep=0;rep<3;rep++){ sub.timeout=60000;   /* rep0/1 warmup, rep2 timed */
        double t0=ork_now_us();
        if(rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; break; }
        bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    if(ok==0){ memcpy(C,O.cpu,(size_t)M*N*4); if(us)*us=t1; }
    bdestroy(fd,&W);bdestroy(fd,&O);
    return ok;
}

int ork_npu_probe_single_i8(ork_npu *c,int K,int N,const int8_t *A,const int8_t *B,int32_t *C){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax) return -2;
    struct buf W=bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    int NN=N/32,KT=K/32; int8_t*bb=W.cpu;     /* int8 tile layout [Ntile][Ktile][32][32], full K */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)];
    bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=bcreate(fd,(size_t)N*4,0x403,-1); if(!O.cpu){bdestroy(fd,&W);return -2;}
    int8_t*ad=c->Af.cpu; for(int j=0;j<K;j++)ad[j]=A[j]; bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    act(fd,RKNPU_ACT_RESET,0);                 /* prime for int8 / clear any prior wedge */
    uint32_t rc[REGCMD_I8_N];
    synth_i8(rc,1,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    setr(rc,REGCMD_I8_N,0x201,0x1040,0xb1);
    struct buf extra[2] = {W, O};
    if (validate_regcmd("probe_single_i8", c, rc, REGCMD_I8_N, NULL, extra, 2)) { bdestroy(fd,&W); bdestroy(fd,&O); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1;
    for(int rep=0;rep<2;rep++){ sub.timeout=60000;   /* rep0 warmup (cold buffer stale), rep1 real */
        if(rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; continue; }
        bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(C,O.cpu,(size_t)N*4); ok=0; }
    bdestroy(fd,&W);bdestroy(fd,&O);
    return ok;
}

/* RE/validation for the PPU fused-output stage (step 1): run ONE full-K int8 matmul at (M,K,N) with
 * the int8-REQUANTIZED output stage (set_i8_out8) instead of int32, and return C[M*N] as int8. This
 * is the isolated bit-exact test bed for the fused path — the caller compares against the CPU model
 * out_i8 = clamp_i8((acc_i32 * mult) >> shift). Isolated buffers, does not touch resident weights.
 * A[M*K] B[K*N] row-major int8; C[M*N] int8 out. mult/shift = fixed-point requant (identity =
 * 0x4000,14). 0/ok (C valid), -1 wedged, -2 bad dims. See ork_ppu_fuse_enabled + set_i8_out8. */
int ork_npu_probe_i8_out8(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,
                          int mult,int shift,int8_t *C,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64) return -2;
    struct buf W=bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    int NN=N/32,KT=K/32; int8_t*bb=W.cpu;     /* int8 tile layout [Ntile][Ktile][32][32], full K */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)];
    bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=bcreate(fd,(size_t)M*N,0x403,-1); if(!O.cpu){bdestroy(fd,&W);return -2;}  /* int8 output: M*N bytes */
    int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I8_N];
    synth_i8(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    set_i8_out8(rc,N,0,mult,shift);           /* rewrite output stage: int32 -> int8 requantize */
    struct buf extra[2] = {W, O};
    if (validate_regcmd("probe_i8_out8", c, rc, REGCMD_I8_N, NULL, extra, 2)) { bdestroy(fd,&W); bdestroy(fd,&O); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0;
    for(int rep=0;rep<3;rep++){ sub.timeout=60000;   /* rep0/1 warmup, rep2 timed */
        double t0=ork_now_us();
        if(rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; break; }
        bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    if(ok==0){ memcpy(C,O.cpu,(size_t)M*N); if(us)*us=t1; }
    bdestroy(fd,&W);bdestroy(fd,&O);
    return ok;
}

/* RE/validation for the FUSED EW-mul output stage (step 3, SwiGLU dual-input): run a full-K int8 matmul
 * whose output stage int8-requantizes the accumulator AND multiplies it by a SECOND input G (= silu(gate)),
 * returning C[M*N] int8. This splices the 0x50xx second-DPU lane into the regcmd (synth_i8_ew) and submits
 * with regcfg_amount=126 (108 matmul + 18 second-lane). A[M*K] B[K*N] G[M*N] row-major int8.
 * FIRST-RUN CONTRACT: the 0x50xx dims/strides are the CAPTURED values (M=8,N=32) — validate at that shape
 * first; the exact multiply/scale semantics and the shape-dependent 0x50xx fields are resolved by board
 * matched-diff (see EWMUL_WIP.md). 0/ok (path executed, C valid), -1 wedged, -2 bad dims. */
int ork_npu_probe_i8_ewmul(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,const int8_t *G,
                           int mult,int shift,int8_t *C,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(!ork_ppu_fuse_enabled(c)) return -3;   /* PPU EW-mul RE'd against the rk3588 PPU layout only */
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64) return -2;
    struct buf W=bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    int NN=N/32,KT=K/32; int8_t*bb=W.cpu;     /* int8 tile layout [Ntile][Ktile][32][32], full K */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)];
    bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=bcreate(fd,(size_t)M*N,0x403,-1); if(!O.cpu){bdestroy(fd,&W);return -2;}         /* int8 output */
    /* over-allocate the 2nd-input buffer: the captured 0x5020/0x5038 partner offsets (+0x4080/+0x400 from
     * base) must land IN-BOUNDS or the IOMMU faults (submit timeout). 64 KiB covers them for these shapes. */
    size_t gsz=(size_t)M*N; if(gsz<0x10000)gsz=0x10000;
    struct buf Gb=bcreate(fd,gsz,0x403,-1); if(!Gb.cpu){bdestroy(fd,&W);bdestroy(fd,&O);return -2;} /* 2nd input */
    memset(Gb.cpu,0,gsz); memcpy(Gb.cpu,G,(size_t)M*N); bsync(fd,&Gb,RKNPU_MEM_SYNC_TO_DEVICE);
    int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    act(fd,RKNPU_ACT_RESET,0);
    uint32_t base[REGCMD_I8_N], rc[REGCMD_I8_EW_N];
    synth_i8(base,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    splice_ew_lane(rc,base);                                    /* insert the 0x50xx second-input lane */
    set_i8_ewmul(rc,M,N,0,mult,shift,(uint32_t)Gb.dma);           /* int8 requant + EW-mul + 2nd-input addr */
    if(getenv("ORK_EW_DUMP")){                                  /* inspect the assembled regcmd, no submit */
        printf("# assembled EW-mul regcmd (%d entries) aG=0x%x aC=0x%x\n",REGCMD_I8_EW_N/2,(uint32_t)Gb.dma,(uint32_t)O.dma);
        for(int k=0;k+1<REGCMD_I8_EW_N;k+=2){uint32_t w0=rc[k],w1=rc[k+1];
            printf("  [%3d] reg=%04x lane=%04x val=%08x\n",k/2,w0&0xffff,w1>>16,((w0>>16)&0xffff)|((w1&0xffff)<<16));}
        bdestroy(fd,&W);bdestroy(fd,&O);bdestroy(fd,&Gb); return 0;
    }
    struct buf extra[3] = {W, O, Gb};
    if (validate_regcmd("probe_i8_ewmul", c, rc, REGCMD_I8_EW_N, NULL, extra, 3)) { bdestroy(fd,&W);bdestroy(fd,&O);bdestroy(fd,&Gb); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    /* bump the task's register-config count 108 -> 126 AND enable_mask 0xd -> 0x1d (the 0x10 bit enables the
     * PPU / second DPU lane; the captured EW-mul op runs at enable=0x1d, same as the SiLU compute op) for
     * this submit, then restore (c->task is shared). */
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu;
    uint32_t saved=tk->regcfg_amount, saved_en=tk->enable_mask;
    tk->regcfg_amount=REGCMD_I8_EW_N/2; tk->enable_mask=0x1d; bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0;
    for(int rep=0;rep<3;rep++){ sub.timeout=ew_timeout_ms();
        double t0=ork_now_us();
        if(rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; break; }
        bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=saved; tk->enable_mask=saved_en; bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);  /* restore shared task */
    if(ok==0){ memcpy(C,O.cpu,(size_t)M*N); if(us)*us=t1; }
    bdestroy(fd,&W);bdestroy(fd,&O);bdestroy(fd,&Gb);
    return ok;
}

/* PATH (b) bring-up: submit RKNN's captured EW-mul op (REGCMD_EWMUL) VERBATIM with ork's buffers, only
 * repointing the 6 buffer addresses. Its geometry is RKNN's own (unlike the synth_i8 overlay the HW
 * rejected), so this tests whether the templatized op EXECUTES on ork's submit path. Buffers mirror the
 * captured handle layout (input/weight/silu/output); contents are caller-provided (execution does not
 * depend on them). Returns 0/ok, -1 wedged. On ok, out[] gets the raw output buffer (Osz bytes).
 * in[Isz] wt[Wsz] gl[Gsz] are copied into the input/weight/silu buffers at their captured offsets. */
int ork_npu_probe_i8_ewmul_tmpl(ork_npu *c,const void*in,int Isz,const void*wt,int Wsz,
                                const void*gl,int Gsz,void*out,int Osz,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;   /* rk3588 PPU only */
    struct buf I=bcreate(fd,4096,0x403,-1); if(!I.cpu) return -2;
    struct buf Wt=bcreate(fd,32768,0x403,-1); if(!Wt.cpu){bdestroy(fd,&I);return -2;}
    struct buf Gl=bcreate(fd,8192,0x403,-1); if(!Gl.cpu){bdestroy(fd,&I);bdestroy(fd,&Wt);return -2;}
    struct buf O=bcreate(fd,4096,0x403,-1); if(!O.cpu){bdestroy(fd,&I);bdestroy(fd,&Wt);bdestroy(fd,&Gl);return -2;}
    memset(I.cpu,0,4096); memset(Wt.cpu,0,32768); memset(Gl.cpu,0,8192); memset(O.cpu,0,4096);
    if(in) memcpy(I.cpu, in, Isz<4096?Isz:4096);
    if(wt) memcpy((char*)Wt.cpu+0x2300, wt, Wsz<(32768-0x2300)?Wsz:(32768-0x2300)); /* weights at captured +0x2300 */
    if(gl) memcpy((char*)Gl.cpu+0x400, gl, Gsz<(8192-0x400)?Gsz:(8192-0x400));      /* silu(gate) at captured +0x400 */
    bsync(fd,&I,RKNPU_MEM_SYNC_TO_DEVICE);bsync(fd,&Wt,RKNPU_MEM_SYNC_TO_DEVICE);bsync(fd,&Gl,RKNPU_MEM_SYNC_TO_DEVICE);
    act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_EWMUL_N]; memcpy(rc,REGCMD_EWMUL,sizeof rc);
    /* ORK_EW_MULT/ORK_EW_SHIFT: override the captured requant (0x4084/0x4088, captured >>29 kills small acc)
     * so the output is interpretable when driving the verbatim op with our own uniform data. */
    { const char*em=getenv("ORK_EW_MULT"),*es=getenv("ORK_EW_SHIFT");
      if(em) setr(rc,REGCMD_EWMUL_N,0x1001,0x4084,(uint32_t)strtoul(em,0,0));
      if(es) setr(rc,REGCMD_EWMUL_N,0x1001,0x4088,(uint32_t)strtoul(es,0,0)); }
    setr(rc,REGCMD_EWMUL_N,0x0201,0x1070,(uint32_t)I.dma);          /* input x */
    setr(rc,REGCMD_EWMUL_N,0x0201,0x1110,(uint32_t)Wt.dma+0x2300);  /* weights */
    setr(rc,REGCMD_EWMUL_N,0x1001,0x4020,(uint32_t)O.dma);          /* output */
    setr(rc,REGCMD_EWMUL_N,0x2001,0x5018,(uint32_t)Gl.dma+0x400);   /* 2nd-input silu(gate) */
    setr(rc,REGCMD_EWMUL_N,0x2001,0x5038,(uint32_t)Gl.dma+0x800);   /* 2nd-input partner */
    setr(rc,REGCMD_EWMUL_N,0x2001,0x5020,(uint32_t)Wt.dma+0x2480);  /* 2nd-input param (in weight buf) */
    if(getenv("ORK_EW_DUMP")){ printf("# verbatim EW-mul regcmd, in=0x%x wt=0x%x gl=0x%x out=0x%x\n",
        (uint32_t)I.dma,(uint32_t)Wt.dma,(uint32_t)Gl.dma,(uint32_t)O.dma);
        for(int k=0;k+1<REGCMD_EWMUL_N;k+=2) printf("  [%3d] reg=%04x lane=%04x val=%08x\n",k/2,rc[k]&0xffff,rc[k+1]>>16,((rc[k]>>16)&0xffff)|((rc[k+1]&0xffff)<<16));
        bdestroy(fd,&I);bdestroy(fd,&Wt);bdestroy(fd,&Gl);bdestroy(fd,&O); return 0; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t sa=tk->regcfg_amount,se=tk->enable_mask;
    tk->regcfg_amount=126; tk->enable_mask=0x1d; bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=ew_timeout_ms();
    double t0=ork_now_us();
    if(!rknpu_submit_ioctl(fd,&sub,-1)){ bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=sa; tk->enable_mask=se; bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ if(out)memcpy(out,O.cpu,Osz<4096?Osz:4096); if(us)*us=t1; }
    bdestroy(fd,&I);bdestroy(fd,&Wt);bdestroy(fd,&Gl);bdestroy(fd,&O);
    return ok;
}

/* Path (b) MATMUL replay: submit the captured matmul-shaped EW-mul op (REGCMD_EWMUL_LIN, K=512/N=64/M=8)
 * with ork's buffers + ork's standard tile packing (B as [Nt][Kt][32][32], A linear). out = requant(A*B) ⊙ G,
 * G = silu(gate) int8 2nd-input. Fixed shape (the captured one) for now — used to read the multiply semantics
 * and validate vs CPU; generalization to arbitrary M/K/N is the next step. A[M*K] B[K*N] G[M*N] int8; C[M*N].
 * ORK_EW_S20=v writes uint32 v into the 0x5020 param region (2nd-input scale probe). 0/ok,-1 wedged,-2 dims. */
int ork_npu_probe_i8_ewmul_lin(ork_npu *c,const int8_t *A,const int8_t *B,const int8_t *G,int8_t *C,double *us){
    const int M=8,K=512,N=64; int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;   /* rk3588 PPU only */
    struct buf Wt=bcreate(fd,0x18000,0x403,-1); if(!Wt.cpu) return -2;   /* weights + 0x5020 param region */
    int NN=N/32,KT=K/32; int8_t*bb=Wt.cpu;
    /* LAYOUT-INVARIANT probe trick: fill the WHOLE weight buffer with B[0] so the matmul reads the same
     * value no matter how RKNN's op tiles it (acc = K*A*B[0] for every output, independent of layout).
     * Then overlay ork's tile packing (a no-op when B is uniform). Lets P1 run before RKNN's A/B layout
     * is reversed. For non-uniform B this region still holds B[0] outside the tile — only valid for the
     * uniform-input probes. */
    memset(bb,(unsigned char)B[0],0x18000);
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)];
    { const char*e=getenv("ORK_EW_S20"); if(e){ uint32_t v=(uint32_t)strtoul(e,0,0); for(int i=0;i<N;i++)((uint32_t*)((char*)Wt.cpu+0x8080))[i]=v; } }
    bsync(fd,&Wt,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(fd,&Wt,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=bcreate(fd,4096,0x403,-1); if(!O.cpu){bdestroy(fd,&Wt);return -2;}
    struct buf Gb=bcreate(fd,4096,0x403,-1); if(!Gb.cpu){bdestroy(fd,&Wt);bdestroy(fd,&O);return -2;}
    memset(O.cpu,0,4096); memset(Gb.cpu,0,4096); memcpy(Gb.cpu,G,(size_t)M*N); bsync(fd,&Gb,RKNPU_MEM_SYNC_TO_DEVICE);
    int8_t*ad=c->Af.cpu; memset(ad,(unsigned char)A[0],0x8000); for(int j=0;j<M*K;j++)ad[j]=A[j];  /* fill for layout-invariance */
    bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_EWMUL_LIN_N]; memcpy(rc,REGCMD_EWMUL_LIN,sizeof rc);
    /* EMITTER: inject ork's matmul geometry into RKNN's EW template (unless ORK_EW_VERBATIM) so the conv
     * engine reads ork's [Nt][Kt][32][32] A/B layout while keeping RKNN's register order + EW output stage. */
    if(!getenv("ORK_EW_VERBATIM")){
        apply_ork_geom(rc,REGCMD_EWMUL_LIN_N,M,K,N,c->soc->cbuf_elems);
        /* Also inject ork's DENSE int8-out output-stage byte config (set_i8_out8) so the output writes dense
         * [M][N] where ork reads it — the template's RKNN output-byte config produced bias-only garbage.
         * Keep the SDP element-wise MULTIPLY enable (0x4070=0x904002c4) + 0x4050 bit0 (2nd-input enable). */
        setr(rc,REGCMD_EWMUL_LIN_N,0x1001,0x4010,0x00000000);    /* int8-out (clear int32 bit) */
        setr(rc,REGCMD_EWMUL_LIN_N,0x1001,0x4038,(((N/16)-1)<<16)|((N/16)-1)); /* dense output group stride */
        setr(rc,REGCMD_EWMUL_LIN_N,0x1001,0x4050,0x00000125);    /* int8 row config + 2nd-input enable (bit0) */
        setr(rc,REGCMD_EWMUL_LIN_N,0x1001,0x40c0,0x00000020);    /* output element size = 1 byte (dense) */
        setr(rc,REGCMD_EWMUL_LIN_N,0x1001,0x4080,0x00000000);    /* zero bias for the probe */
        setr(rc,REGCMD_EWMUL_LIN_N,0x2001,0x500c,M-1);           /* 2nd-lane geometry -> ork's (M,N) */
        setr(rc,REGCMD_EWMUL_LIN_N,0x2001,0x5010,M-1);
        setr(rc,REGCMD_EWMUL_LIN_N,0x2001,0x5014,N-1);
        setr(rc,REGCMD_EWMUL_LIN_N,0x2001,0x5040,N);
        setr(rc,REGCMD_EWMUL_LIN_N,0x2001,0x504c,N);
        setr(rc,REGCMD_EWMUL_LIN_N,0x2001,0x506c,N);
    }
    /* ORK_EW_MULT/SHIFT override the captured requant so small test accumulators requant to a readable value. */
    { const char*em=getenv("ORK_EW_MULT"),*es=getenv("ORK_EW_SHIFT");
      if(em) setr(rc,REGCMD_EWMUL_LIN_N,0x1001,0x4084,(uint32_t)strtoul(em,0,0));
      if(es) setr(rc,REGCMD_EWMUL_LIN_N,0x1001,0x4088,(uint32_t)strtoul(es,0,0)); }
    uint32_t gstride = getenv("ORK_EW_VERBATIM") ? 0x80 : (uint32_t)N;  /* 2nd-input row stride */
    setr(rc,REGCMD_EWMUL_LIN_N,0x0201,0x1070,(uint32_t)c->Af.dma);        /* input A */
    setr(rc,REGCMD_EWMUL_LIN_N,0x0201,0x1110,(uint32_t)Wt.dma);           /* up-weights */
    setr(rc,REGCMD_EWMUL_LIN_N,0x1001,0x4020,(uint32_t)O.dma);            /* output */
    setr(rc,REGCMD_EWMUL_LIN_N,0x2001,0x5018,(uint32_t)Gb.dma);           /* silu(gate) 2nd input */
    setr(rc,REGCMD_EWMUL_LIN_N,0x2001,0x5038,(uint32_t)Gb.dma+gstride);   /* 2nd-input partner = base+stride */
    setr(rc,REGCMD_EWMUL_LIN_N,0x2001,0x5020,(uint32_t)Wt.dma+0x8080);    /* 2nd-input param (in weight buf) */
    /* ORK_EW_NOMUL: turn OFF the EW multiply (0x4070 -> plain 0x0302) to read R=requant(acc) alone —
     * isolates "does the matmul+requant work" from "does the multiply work". */
    if(getenv("ORK_EW_NOMUL")) setr(rc,REGCMD_EWMUL_LIN_N,0x1001,0x4070,0x00000302);
    if(getenv("ORK_EW_DUMP")){ for(int k=0;k+1<REGCMD_EWMUL_LIN_N;k+=2) printf("  [%3d] reg=%04x lane=%04x val=%08x\n",k/2,rc[k]&0xffff,rc[k+1]>>16,((rc[k]>>16)&0xffff)|((rc[k+1]&0xffff)<<16));
        bdestroy(fd,&Wt);bdestroy(fd,&O);bdestroy(fd,&Gb); return 0; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t sa=tk->regcfg_amount,se=tk->enable_mask;
    tk->regcfg_amount=126; tk->enable_mask=0x1d; bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=ew_timeout_ms(); double t0=ork_now_us();
    if(!rknpu_submit_ioctl(fd,&sub,-1)){ bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=sa; tk->enable_mask=se; bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ memcpy(C,O.cpu,(size_t)M*N); if(us)*us=t1; }
    if(ok==0 && getenv("ORK_EW_SCAN")){ int nz=0,first=-1; signed char*o=O.cpu;
        for(int i=0;i<4096;i++){ if(o[i]){ nz++; if(first<0)first=i; } }
        printf("  [scan] output-buffer nonzero bytes=%d first@0x%x  bytes@first: ",nz,first);
        for(int i=(first<0?0:first);i<(first<0?16:first+16);i++)printf("%d ",o[i]); printf("\n"); }
    bdestroy(fd,&Wt);bdestroy(fd,&O);bdestroy(fd,&Gb);
    return ok;
}

/* Standalone SDP element-wise MULTIPLY: submit REGCMD_MUL (captured pure-Mul op, no conv) with a,b,out patched.
 * out[i] = clamp_i8(round(a[i]*b[i]*gain) + bias), gain/bias baked from the captured op (0x4084/88/80).
 * a,b,out int8[n] (n<=4096). Reads a via SRDMA(0x5018), b via ERDMA(0x5038), writes out(0x4020). enable=0x18,
 * regcfg=69. This is the CLEAN on-NPU element-wise path (NVDLA standalone SDP layer, both operands from memory)
 * — sidesteps the conv-geometry coupling that blocked the fused-into-matmul approach. 0/ok,-1 wedged. */
int ork_npu_probe_i8_mul(ork_npu *c,const int8_t *a,const int8_t *b,int n,int8_t *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(n<1||n>4096) return -2;
    struct buf A=bcreate(fd,4096,0x403,-1); if(!A.cpu)return -2;
    struct buf B=bcreate(fd,4096,0x403,-1); if(!B.cpu){bdestroy(fd,&A);return -2;}
    struct buf O=bcreate(fd,4096,0x403,-1); if(!O.cpu){bdestroy(fd,&A);bdestroy(fd,&B);return -2;}
    memset(A.cpu,0,4096);memset(B.cpu,0,4096);memset(O.cpu,0,4096);
    memcpy(A.cpu,a,n);memcpy(B.cpu,b,n);
    bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);
    bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);   /* clear device-side output (op writes only part -> avoid stale) */
    act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_MUL_N]; memcpy(rc,REGCMD_MUL,sizeof rc);
    setr(rc,REGCMD_MUL_N,0x1001,0x4020,(uint32_t)O.dma);        /* output */
    setr(rc,REGCMD_MUL_N,0x2001,0x5018,(uint32_t)A.dma);        /* operand a (SRDMA) */
    setr(rc,REGCMD_MUL_N,0x2001,0x5038,(uint32_t)B.dma);        /* operand b (ERDMA element-wise) */
    { const char*em=getenv("ORK_EW_MULT"),*es=getenv("ORK_EW_SHIFT"),*eb=getenv("ORK_EW_BIAS");
      const char*co=getenv("ORK_EW_COFF"),*cs=getenv("ORK_EW_CSCL");
      if(em) setr(rc,REGCMD_MUL_N,0x1001,0x4084,(uint32_t)strtoul(em,0,0));
      if(es) setr(rc,REGCMD_MUL_N,0x1001,0x4088,(uint32_t)strtoul(es,0,0));
      if(eb) setr(rc,REGCMD_MUL_N,0x1001,0x4080,(uint32_t)strtoul(eb,0,0));
      if(co) setr(rc,REGCMD_MUL_N,0x1001,0x4074,(uint32_t)strtoul(co,0,0));   /* EW_CVT_OFFSET = zb (operand b zero-pt) */
      if(cs) setr(rc,REGCMD_MUL_N,0x1001,0x4078,(uint32_t)strtoul(cs,0,0));   /* EW_CVT_SCALE (operand b) */
      const char*ao=getenv("ORK_EW_AOFF");
      if(ao) setr(rc,REGCMD_MUL_N,0x1001,0x4044,(uint32_t)strtoul(ao,0,0)); } /* BS_ALU_OPERAND = za (operand a zero-pt) */
    if(getenv("ORK_EW_DUMP")){ for(int k=0;k+1<REGCMD_MUL_N;k+=2) printf("  [%3d] reg=%04x lane=%04x val=%08x\n",k/2,rc[k]&0xffff,rc[k+1]>>16,((rc[k]>>16)&0xffff)|((rc[k+1]&0xffff)<<16));
        bdestroy(fd,&A);bdestroy(fd,&B);bdestroy(fd,&O); return 0; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t sa=tk->regcfg_amount,se=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=ew_timeout_ms(); double t0=ork_now_us();
    if(!rknpu_submit_ioctl(fd,&sub,-1)){ bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=sa; tk->enable_mask=se; bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ memcpy(out,O.cpu,n); if(us)*us=t1; }
    if(ok==0 && getenv("ORK_EW_SCAN")){ signed char*o=O.cpu; int nz=0,first=-1,last=-1;
        for(int i=0;i<4096;i++){ if(o[i]){ nz++; if(first<0)first=i; last=i; } }
        printf("  [scan4k] nonzero=%d span[0x%x..0x%x]  vals@first: ",nz,first,last);
        for(int i=(first<0?0:first);i<(first<0?16:first+24)&&i<4096;i++)printf("%d ",o[i]); printf("\n"); }
    bdestroy(fd,&A);bdestroy(fd,&B);bdestroy(fd,&O);
    return ok;
}

/* Public EW-mul: out[m*N+n] = clamp_i8(round(up[m][n]*silu[m][n] * mult/2^shift)) computed ON THE NPU via the
 * standalone SDP element-wise op. Marshals up/silu (logical [M][N]) into the NVDLA feature cube (atom-16),
 * submits REGCMD_MUL with symmetric zero-points (za=zb=zo=0), de-marshals. GENERALIZED to arbitrary M,N via
 * set_mul_geom (M,N reprogrammed from the captured M=8/N=64 op). N must be a multiple of 16 (channel atom).
 * mult must be 0..0x7fff (OUT_CVT_SCALE is SIGNED 16-bit). 0/ok, -1 wedged, -2 bad shape, -3 non-rk3588. */
int ork_npu_ewmul_i8(ork_npu *c,const int8_t *up,const int8_t *silu,int M,int N,int mult,int shift,int8_t *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<16||N>8192||(N&15)) return -2;             /* N multiple of the int8 atom (16) */
    if(mult<0||mult>0x7fff||shift<0||shift>31) return -2;
    #define EWCUBE(m,n) (((n)/16)*(M*16) + (m)*16 + ((n)%16))    /* NVDLA cube, atom=16, surf_stride=M*16 */
    size_t sz=(size_t)M*N; if(sz<4096)sz=4096;                    /* int8 cube = M*N bytes */
    struct buf A=bcreate(fd,sz,0x403,-1); if(!A.cpu)return -2;
    struct buf B=bcreate(fd,sz,0x403,-1); if(!B.cpu){bdestroy(fd,&A);return -2;}
    struct buf O=bcreate(fd,sz,0x403,-1); if(!O.cpu){bdestroy(fd,&A);bdestroy(fd,&B);return -2;}
    memset(A.cpu,0,sz);memset(B.cpu,0,sz);memset(O.cpu,0,sz);
    int8_t*ac=A.cpu,*bc=B.cpu;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int p=EWCUBE(m,n); ac[p]=up[m*N+n]; bc[p]=silu[m*N+n]; }
    bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_MUL_N]; memcpy(rc,REGCMD_MUL,sizeof rc);
    set_mul_geom(rc,REGCMD_MUL_N,M,N);
    setr(rc,REGCMD_MUL_N,0x1001,0x4020,(uint32_t)O.dma);        /* output */
    setr(rc,REGCMD_MUL_N,0x2001,0x5018,(uint32_t)A.dma);        /* up  (SRDMA)  */
    setr(rc,REGCMD_MUL_N,0x2001,0x5038,(uint32_t)B.dma);        /* silu (ERDMA) */
    setr(rc,REGCMD_MUL_N,0x1001,0x4084,(uint32_t)mult);         /* OUT_CVT_SCALE = gain mantissa */
    setr(rc,REGCMD_MUL_N,0x1001,0x4088,(uint32_t)shift);        /* OUT_CVT_SHIFT */
    setr(rc,REGCMD_MUL_N,0x1001,0x4080,0);                      /* zo = 0 (OUT_CVT_OFFSET) */
    setr(rc,REGCMD_MUL_N,0x1001,0x4044,0);                      /* za = 0 (BS_ALU_OPERAND) */
    setr(rc,REGCMD_MUL_N,0x1001,0x4074,0);                      /* zb = 0 (EW_CVT_OFFSET) */
    memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=ew_timeout_ms(); double t0=ork_now_us();
    if(!rknpu_submit_ioctl(fd,&sub,-1)){ bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ int8_t*oc=O.cpu; for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[m*N+n]=oc[EWCUBE(m,n)]; if(us)*us=t1; }
    bdestroy(fd,&A);bdestroy(fd,&B);bdestroy(fd,&O);
    #undef EWCUBE
    return ok;
}

/* fp16 element-wise MULTIPLY: out[m][n] = up[m][n] * silu[m][n] in fp16 on the NPU (standalone SDP fp16 op,
 * no requant). Marshals into the NVDLA fp16 feature cube (atom=8, 2-byte channels) internally. GENERALIZED to
 * arbitrary M,N (N a multiple of 8); rk3588-gated. up/silu/out are fp16 bit-patterns (ork_f16).
 * 0/ok,-1 wedged,-2 shape,-3 SoC. */
int ork_npu_ewmul_f16(ork_npu *c,const ork_f16 *up,const ork_f16 *silu,int M,int N,ork_f16 *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;               /* N multiple of the fp16 atom (8) */
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)   /* BYTE offset, fp16 atom=8, surf_stride=M*16 */
    const uint16_t *u16=(const uint16_t*)up,*s16=(const uint16_t*)silu; uint16_t *o16=(uint16_t*)out;
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;                  /* fp16 cube = M*N*2 bytes */
    struct buf A=bcreate(fd,sz,0x403,-1); if(!A.cpu)return -2;
    struct buf B=bcreate(fd,sz,0x403,-1); if(!B.cpu){bdestroy(fd,&A);return -2;}
    struct buf O=bcreate(fd,sz,0x403,-1); if(!O.cpu){bdestroy(fd,&A);bdestroy(fd,&B);return -2;}
    memset(A.cpu,0,sz);memset(B.cpu,0,sz);memset(O.cpu,0,sz);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int p=EWCUBEH(m,n);
        *(uint16_t*)((char*)A.cpu+p)=u16[m*N+n]; *(uint16_t*)((char*)B.cpu+p)=s16[m*N+n]; }
    bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_MUL_F16_N]; memcpy(rc,REGCMD_MUL_F16,sizeof rc);
    set_mul_geom(rc,REGCMD_MUL_F16_N,M,N);
    setr(rc,REGCMD_MUL_F16_N,0x1001,0x4020,(uint32_t)O.dma);
    setr(rc,REGCMD_MUL_F16_N,0x2001,0x5018,(uint32_t)A.dma);
    setr(rc,REGCMD_MUL_F16_N,0x2001,0x5038,(uint32_t)B.dma);
    memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=ew_timeout_ms(); double t0=ork_now_us();
    if(!rknpu_submit_ioctl(fd,&sub,-1)){ bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) o16[m*N+n]=*(uint16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(us)*us=t1; }
    bdestroy(fd,&A);bdestroy(fd,&B);bdestroy(fd,&O);
    #undef EWCUBEH
    return ok;
}

/* int16 element-wise MULTIPLY: out[m][n] = clamp_i16(round(up[m][n]*silu[m][n] * mult/2^shift)) on the NPU
 * (standalone SDP int16 op). The w4a4 path's EW precision (ork's int4 matmul outputs int16). 2-byte operands,
 * NVDLA cube atom=8 (same layout as fp16). Symmetric zero-points. mult in 0..0x7fff. GENERALIZED to arbitrary
 * M,N (N a multiple of 8); rk3588-gated. 0/ok,-1 wedged,-2 shape,-3 SoC. */
int ork_npu_ewmul_i16(ork_npu *c,const int16_t *up,const int16_t *silu,int M,int N,int mult,int shift,int16_t *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<8||N>8192||(N&7)) return -2;               /* N multiple of the int16 atom (8) */
    if(mult<0||mult>0x7fff||shift<0||shift>31) return -2;
    #define EWCUBEH(m,n) (((n)/8)*(M*16) + (m)*16 + ((n)%8)*2)   /* 2-byte atom=8 cube (fp16/int16), surf_stride=M*16 */
    size_t sz=(size_t)M*N*2; if(sz<4096)sz=4096;                  /* int16 cube = M*N*2 bytes */
    struct buf A=bcreate(fd,sz,0x403,-1); if(!A.cpu)return -2;
    struct buf B=bcreate(fd,sz,0x403,-1); if(!B.cpu){bdestroy(fd,&A);return -2;}
    struct buf O=bcreate(fd,sz,0x403,-1); if(!O.cpu){bdestroy(fd,&A);bdestroy(fd,&B);return -2;}
    memset(A.cpu,0,sz);memset(B.cpu,0,sz);memset(O.cpu,0,sz);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int p=EWCUBEH(m,n);
        *(int16_t*)((char*)A.cpu+p)=up[m*N+n]; *(int16_t*)((char*)B.cpu+p)=silu[m*N+n]; }
    bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE);bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_MUL_I16_N]; memcpy(rc,REGCMD_MUL_I16,sizeof rc);
    set_mul_geom(rc,REGCMD_MUL_I16_N,M,N);
    setr(rc,REGCMD_MUL_I16_N,0x1001,0x4020,(uint32_t)O.dma);
    setr(rc,REGCMD_MUL_I16_N,0x2001,0x5018,(uint32_t)A.dma);
    setr(rc,REGCMD_MUL_I16_N,0x2001,0x5038,(uint32_t)B.dma);
    setr(rc,REGCMD_MUL_I16_N,0x1001,0x4084,(uint32_t)mult);     /* OUT_CVT_SCALE (gain mantissa) */
    setr(rc,REGCMD_MUL_I16_N,0x1001,0x4088,(uint32_t)shift);    /* OUT_CVT_SHIFT */
    setr(rc,REGCMD_MUL_I16_N,0x1001,0x4080,0);                  /* zo=0 */
    setr(rc,REGCMD_MUL_I16_N,0x1001,0x4044,0);                  /* za=0 */
    setr(rc,REGCMD_MUL_I16_N,0x1001,0x4074,0);                  /* zb=0 */
    memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; uint32_t saa=tk->regcfg_amount,see=tk->enable_mask;
    tk->regcfg_amount=69; tk->enable_mask=0x18; bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=ew_timeout_ms(); double t0=ork_now_us();
    if(!rknpu_submit_ioctl(fd,&sub,-1)){ bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    tk->regcfg_amount=saa; tk->enable_mask=see; bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE);
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[m*N+n]=*(int16_t*)((char*)O.cpu+EWCUBEH(m,n)); if(us)*us=t1; }
    bdestroy(fd,&A);bdestroy(fd,&B);bdestroy(fd,&O);
    #undef EWCUBEH
    return ok;
}

/* ── Standalone on-NPU SiLU (activation-LUT SDP op) — RE probe ─────────────────────────────────────
 * Applies the PPU activation LUT to a SINGLE int8 memory input [M][N] via the standalone 69-reg/enable=0x18
 * SDP op (REGCMD_SILU_STD), reprogrammed to (M,N) by set_mul_geom. Two submits on the single-stream NPU:
 * (1) LUT-load (REGCMD_SILU_LUT, streams the int16 curve into PPU SRAM); (2) the standalone op reads it.
 * SDP math: idx=(in*R)>>6 + C0(idx_off); out=clamp_i8(R*LUT-interp(idx) + out_bias); R=r_mult/2^r_shift.
 * The caller supplies the scale regs (r_mult,r_shift,out_bias,idx_off,cfg4064,cfg4068) and the LUT; lut==NULL
 * keeps the captured curve. This is the RE/calibration entry (measure idx(in) with a ramp LUT, then build the
 * silu curve at those indices — same 2-pass scheme as ork_mm_silu_build_lut but through THIS op, not a matmul).
 * in/out int8 [M*N] row-major; N%16==0. 0/ok, -1 wedged, -2 bad shape, -3 non-rk3588. */
int ork_npu_probe_silu_std(ork_npu *c,const int8_t *in,int M,int N,
                           int r_mult,int r_shift,uint32_t out_bias,uint32_t idx_off,
                           uint32_t cfg4064,uint32_t cfg4068,const int16_t *lut,int nlut,
                           int8_t *out,double *us){
    int fd=c->fd;
    if(!ork_ppu_fuse_enabled(c)) return -3;
    if(M<1||M>8192||N<16||N>8192||(N&15)) return -2;
    if(r_mult<0||r_mult>0x7fff||r_shift<0||r_shift>31) return -2;
    #define EWCUBE(m,n) (((n)/16)*(M*16) + (m)*16 + ((n)%16))    /* int8 atom-16 cube, surf_stride=M*16 */
    size_t sz=(size_t)M*N; if(sz<4096)sz=4096;
    struct buf A=bcreate(fd,sz,0x403,-1); if(!A.cpu)return -2;
    struct buf O=bcreate(fd,sz,0x403,-1); if(!O.cpu){bdestroy(fd,&A);return -2;}
    struct buf Lrc=bcreate(fd,(size_t)REGCMD_SILU_LUT_N*4,0x403,-1); if(!Lrc.cpu){bdestroy(fd,&A);bdestroy(fd,&O);return -2;}
    struct buf Lsc=bcreate(fd,4096,0x403,-1); if(!Lsc.cpu){bdestroy(fd,&A);bdestroy(fd,&O);bdestroy(fd,&Lrc);return -2;}
    memset(A.cpu,0,sz);memset(O.cpu,0,sz);
    int8_t*ac=A.cpu; for(int m=0;m<M;m++)for(int n=0;n<N;n++) ac[EWCUBE(m,n)]=in[m*N+n];
    bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE);bsync(fd,&O,RKNPU_MEM_SYNC_TO_DEVICE);
    act(fd,RKNPU_ACT_RESET,0);

    /* ---- submit 1: LUT-load (enable=0x18, regcfg=1097) ---- */
    memcpy(Lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
    setr((uint32_t*)Lrc.cpu,REGCMD_SILU_LUT_N,0x1001,0x4020,(uint32_t)Lsc.dma);
    if(lut){ uint32_t*lr=(uint32_t*)Lrc.cpu; int j=0;
        for(int k=0;k+1<REGCMD_SILU_LUT_N;k+=2){ if((lr[k]&0xffff)==0x4104){ int32_t v=(j<nlut)?(int32_t)lut[j]:0; j++;
            lr[k]=0x4104|((uint32_t)(v&0xffff)<<16); lr[k+1]=(0x1001u<<16)|(((uint32_t)v>>16)&0xffff); } } }
    bsync(fd,&Lrc,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
      t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=1097; t->regcmd_addr=Lrc.dma;
      bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.timeout=ew_timeout_ms();sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      if(rknpu_submit_ioctl(fd,&sub,-1)){ if(getenv("ORK_SILU_DBG"))fprintf(stderr,"[silu_std] submit1 (LUT-load) WEDGED\n"); bdestroy(fd,&A);bdestroy(fd,&O);bdestroy(fd,&Lrc);bdestroy(fd,&Lsc); return -1; }
      if(getenv("ORK_SILU_DBG"))fprintf(stderr,"[silu_std] submit1 (LUT-load) ok\n");
    }

    /* ---- submit 2: standalone SiLU op (enable=0x18, regcfg=69) reading the resident LUT ---- */
    uint32_t rc[REGCMD_SILU_STD_N]; memcpy(rc,REGCMD_SILU_STD,sizeof rc);
    set_mul_geom(rc,REGCMD_SILU_STD_N,M,N);
    setr(rc,REGCMD_SILU_STD_N,0x2001,0x5040,0);                 /* single-input: no ERDMA 2nd operand */
    setr(rc,REGCMD_SILU_STD_N,0x2001,0x5038,0);                 /* (set_mul_geom is for the 2-input EW-mul) */
    setr(rc,REGCMD_SILU_STD_N,0x1001,0x4020,(uint32_t)O.dma);   /* output */
    setr(rc,REGCMD_SILU_STD_N,0x2001,0x5018,(uint32_t)A.dma);   /* input (SRDMA) */
    setr(rc,REGCMD_SILU_STD_N,0x1001,0x4084,(uint32_t)r_mult);  /* R mantissa */
    setr(rc,REGCMD_SILU_STD_N,0x1001,0x4088,(uint32_t)r_shift); /* R shift */
    setr(rc,REGCMD_SILU_STD_N,0x1001,0x4080,out_bias);          /* out_bias */
    setr(rc,REGCMD_SILU_STD_N,0x1001,0x4110,idx_off);           /* C0 index offset */
    setr(rc,REGCMD_SILU_STD_N,0x1001,0x4064,cfg4064);           /* index param */
    setr(rc,REGCMD_SILU_STD_N,0x1001,0x4068,cfg4068);           /* index param */
    memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    /* full task setup — submit-1 repointed regcmd_addr at the LUT-load buffer, so re-point it here */
    struct rknpu_task *tk=(struct rknpu_task*)c->task.cpu; memset(tk,0,sizeof *tk);
    tk->enable_mask=0x18; tk->int_mask=0x300; tk->int_clear=0x1ffff; tk->regcfg_amount=69; tk->regcmd_addr=c->regcmd.dma;
    bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1; double t1=0; sub.timeout=ew_timeout_ms(); double t0=ork_now_us();
    if(!rknpu_submit_ioctl(fd,&sub,-1)){ bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
    if(ok==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++) out[m*N+n]=*(int8_t*)((char*)O.cpu+EWCUBE(m,n)); if(us)*us=t1; }
    bdestroy(fd,&A);bdestroy(fd,&O);bdestroy(fd,&Lrc);bdestroy(fd,&Lsc);
    #undef EWCUBE
    return ok;
}

/* RE/validation for the FUSED SiLU output stage (step 2): run a full-K int8 matmul with SiLU applied
 * on-chip, returning C[M*N] as int8. TWO submits on the single-stream NPU: (1) the LUT-load program
 * (REGCMD_SILU_LUT, enable=0x18) streams the int16 silu curve into PPU LUT SRAM; (2) the matmul compute
 * (synth_i8 + set_i8_silu, enable=0x1d) reads that LUT in its output stage. The LUT persists in SRAM
 * between the two sequential submits. Isolated buffers. A[M*K] B[K*N] row-major int8; C[M*N] int8.
 * 0/ok (path executed, C valid), -1 wedged, -2 bad dims. FIRST-RUN status: replays the mm_silu capture
 * scale/LUT verbatim — proves the path EXECUTES + LUT-persists; per-scale generation is WIP. */
int ork_npu_probe_i8_silu(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,int8_t *C,double *us){
    /* g2 captured register set (a known-good replay): R=0x51aa/2^0x14, bias=-97, idx_off, 0x4068 field. */
    return ork_npu_probe_i8_silu_cfg(c,M,K,N,A,B,0x51aa,0x14,0xffffff9fu,0xffffc000u,0x56391100u,NULL,0,C,us);
}
/* Fused-SiLU probe with the decoded knobs exposed (see set_i8_silu): r_mult/r_shift = the unified scale R
 * (0x4084/0x4088), out_bias = 0x4080, idx_off = 0x4110, cfg4068 = 0x4068. lut != NULL overrides the LUT
 * contents (streams the first `nlut` int16 values into the LUT-load's 0x4104 writes) — for the staircase/
 * const-LUT calibration harness; lut==NULL keeps the captured fixed silu curve. */
int ork_npu_probe_i8_silu_cfg(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,
                              int r_mult,int r_shift,uint32_t out_bias,uint32_t idx_off,uint32_t cfg4068,
                              const int16_t *lut,int nlut,int8_t *C,double *us){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax||M<1||M>64) return -2;
    struct buf W=bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    int NN=N/32,KT=K/32; int8_t*bb=W.cpu;
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)];
    bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=bcreate(fd,(size_t)M*N,0x403,-1); if(!O.cpu){bdestroy(fd,&W);return -2;}
    struct buf Lrc=bcreate(fd,(size_t)REGCMD_SILU_LUT_N*4,0x403,-1); if(!Lrc.cpu){bdestroy(fd,&W);bdestroy(fd,&O);return -2;} /* LUT-load regcmd */
    struct buf Lsc=bcreate(fd,4096,0x403,-1); if(!Lsc.cpu){bdestroy(fd,&W);bdestroy(fd,&O);bdestroy(fd,&Lrc);return -2;} /* LUT-load scratch (reg 0x4020) */
    int8_t*ad=c->Af.cpu; for(int j=0;j<M*K;j++)ad[j]=A[j]; bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    act(fd,RKNPU_ACT_RESET,0);

    /* ---- submit 1: LUT-load (enable=0x18, regcfg=1097) — streams the silu LUT into PPU SRAM ---- */
    memcpy(Lrc.cpu,REGCMD_SILU_LUT,REGCMD_SILU_LUT_N*4);
    setr((uint32_t*)Lrc.cpu,REGCMD_SILU_LUT_N,0x1001,0x4020,(uint32_t)Lsc.dma); /* patch the one output addr */
    if(lut){ uint32_t*lr=(uint32_t*)Lrc.cpu; int j=0;   /* override LUT data: stream lut[] into the 0x4104 writes in order */
        for(int k=0;k+1<REGCMD_SILU_LUT_N;k+=2){ if((lr[k]&0xffff)==0x4104){ int32_t v=(j<nlut)?(int32_t)lut[j]:0; j++;
            lr[k]=0x4104|((uint32_t)(v&0xffff)<<16); lr[k+1]=(0x1001u<<16)|(((uint32_t)v>>16)&0xffff); } } }
    bsync(fd,&Lrc,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
      t->enable_mask=0x18; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=1097; t->regcmd_addr=Lrc.dma;
      bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.timeout=60000;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      if(rknpu_submit_ioctl(fd,&sub,-1)){ bdestroy(fd,&W);bdestroy(fd,&O);bdestroy(fd,&Lrc);bdestroy(fd,&Lsc); return -1; }
    }

    /* ---- submit 2: matmul compute (enable=0x1d, regcfg=108) reading the resident LUT ---- */
    uint32_t rc[REGCMD_I8_N];
    synth_i8(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    set_i8_silu(rc,N,0,r_mult,r_shift,out_bias,idx_off,cfg4068);
    struct buf extra[2]={W,O};
    if(validate_regcmd("probe_i8_silu",c,rc,REGCMD_I8_N,NULL,extra,2)){ bdestroy(fd,&W);bdestroy(fd,&O);bdestroy(fd,&Lrc);bdestroy(fd,&Lsc); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    { struct rknpu_task *t=c->task.cpu; memset(t,0,sizeof *t);
      t->enable_mask=0x1d; t->int_mask=0x300; t->int_clear=0x1ffff; t->regcfg_amount=108; t->regcmd_addr=c->regcmd.dma;
      bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
      struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
      int ok=-1; double t1=0;
      for(int rep=0;rep<3;rep++){ sub.timeout=60000; double t0=ork_now_us();
          if(rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; break; }
          bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; t1=ork_now_us()-t0; }
      if(ok==0){ memcpy(C,O.cpu,(size_t)M*N); if(us)*us=t1; }
      bdestroy(fd,&W);bdestroy(fd,&O);bdestroy(fd,&Lrc);bdestroy(fd,&Lsc);
      return ok;
    }
}

static double ork_now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec*1e-3; }

static double silu_f(double x){ return x/(1.0+exp(-x)); }
/* ork_mm_silu_build_lut — generate ork's OWN silu LUT for the fused-output path (ork-NATIVE: no RKNN
 * dependence, works on ork's 108-reg matmul program). Since ork controls both the LUT and the output-stage
 * registers, correct fused SiLU is a 2-step construction (see tools/silu_native.c, validated ~1 int8):
 *   (1) MEASURE ork's index(acc) for (r_mult,r_shift,cfg4068) via one ramp-LUT calibration submit;
 *   (2) BUILD lut[idx(acc)] = clamp_int16( silu(acc*in_scale)/out_scale / R ), R=r_mult/2^r_shift; interp gaps.
 * The caller then runs the matmul via ork_npu_probe_i8_silu_cfg(..,r_mult,r_shift,0,0xffffc000,cfg4068,lut,1030,..)
 * — do the build ONCE per (registers) and reuse the lut across matmuls of the same scale. Pick r_mult/r_shift
 * so R ~= 660*in_scale (the matmul's acc range then spans silu's transition band). out_bias MUST be 0 (the
 * validated config; the ramp readback assumes it). Fills lut[1030]. 0/ok, -1 fail. */
int ork_mm_silu_build_lut(ork_npu*c, double in_scale, double out_scale,
                          int r_mult, int r_shift, uint32_t cfg4068, int16_t *lut){
    const int K=512, N=64;
    signed char *A=malloc(K), *B=calloc(1,(size_t)K*N); int8_t *C=malloc(N);
    int16_t *ramp=malloc(1030*2); int *acc=malloc(N*sizeof(int)), *idx=malloc(N*sizeof(int));
    if(!A||!B||!C||!ramp||!acc||!idx){ free(A);free(B);free(C);free(ramp);free(acc);free(idx); return -1; }
    double R = (double)r_mult / (double)(1u<<r_shift);
    for(int k=0;k<K;k++)A[k]=1;
    int accmax=(int)(8.0/in_scale); int step=(2*accmax)/(N-1); if(step<1)step=1;
    for(int n=0;n<N;n++){ int T=-accmax+n*step; int b=T/K; for(int k=0;k<K;k++)B[k*N+n]=(signed char)(b+(k<(T-b*K)?1:0)); }
    for(int n=0;n<N;n++){ int a=0; for(int k=0;k<K;k++)a+=A[k]*B[k*N+n]; acc[n]=a; }
    /* pass 1: ramp LUT[i]=(i-512)*8 -> out = R*LUT[idx] -> idx = round(out/(R*8)) + 512 */
    for(int i=0;i<1030;i++){ int v=(i-512)*8; if(v>32767)v=32767; if(v<-32768)v=-32768; ramp[i]=(int16_t)v; }
    if(ork_npu_probe_i8_silu_cfg(c,1,K,N,A,B,r_mult,r_shift,0u,0xffffc000u,cfg4068,ramp,1030,C,0)){
        free(A);free(B);free(C);free(ramp);free(acc);free(idx); return -1; }
    for(int n=0;n<N;n++){ int i=(int)lround(C[n]/(R*8.0))+512; idx[n]=i; }
    /* pass 2: build ork's silu LUT at the measured indices; interp gaps, hold at ends */
    int *set=calloc(1030,sizeof(int)); for(int i=0;i<1030;i++)lut[i]=0;
    for(int n=0;n<N;n++){ int i=idx[n]; if(i<0||i>1029)continue;
        double v=silu_f(acc[n]*in_scale)/out_scale/R; long q=lround(v); if(q>32767)q=32767; if(q<-32768)q=-32768;
        lut[i]=(int16_t)q; set[i]=1; }
    int lo=-1,hi=-1; for(int i=0;i<1030;i++)if(set[i]){lo=i;break;} for(int i=1029;i>=0;i--)if(set[i]){hi=i;break;}
    if(lo<0){ free(A);free(B);free(C);free(ramp);free(acc);free(idx);free(set); return -1; }
    for(int i=0;i<lo;i++)lut[i]=lut[lo]; for(int i=hi+1;i<1030;i++)lut[i]=lut[hi];
    for(int i=lo;i<=hi;i++){ if(set[i])continue; int a=i,b=i; while(a>lo&&!set[a])a--; while(b<hi&&!set[b])b++;
        lut[i]=(int16_t)(lut[a]+(lut[b]-lut[a])*(i-a)/(b-a)); }
    free(A);free(B);free(C);free(ramp);free(acc);free(idx);free(set);
    return 0;
}

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
    struct buf W=bcreate(fd,(size_t)K*N,0x403,-1); if(!W.cpu) return -2;
    memset(W.cpu,1,(size_t)K*N); bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=bcreate(fd,(size_t)N*4,0x403,-1); if(!O.cpu){bdestroy(fd,&W);return -2;}
    int8_t*ad=c->Af.cpu; memset(ad,1,K); bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I8_N]; synth_i8(rc,1,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF,0);
    setr(rc,REGCMD_I8_N,0x201,0x1040,0xb1);
    struct buf extra[2] = {W, O};
    if (validate_regcmd("probe_batch", c, rc, REGCMD_I8_N, NULL, extra, 2)) { bdestroy(fd,&W); bdestroy(fd,&O); return -1; }
    for (int i = 0; i < ntask; i++) {
        memcpy((char*)c->regcmd.cpu + i * sizeof(rc), rc, sizeof(rc));
    }
    bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task*t=c->task.cpu;                 /* task[] array: ntask tasks, separate regcmd spaces */
    for(int i=0;i<ntask;i++){memset(&t[i],0,sizeof t[i]);t[i].flags=0;t[i].op_idx=i;t[i].enable_mask=0xd;t[i].int_mask=0x300;t[i].int_clear=0x1ffff;t[i].regcfg_amount=108;t[i].regcmd_addr=c->regcmd.dma + i * sizeof(rc);}
    bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    /* single-core: set all subcore_task entries to avoid kernel UAPI timeout/Oops */
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.timeout=60000;
    sub.task_number=1; sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,1};
    if(rknpu_submit_ioctl(fd,&sub,-1)){bdestroy(fd,&W);bdestroy(fd,&O);return -1;} bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); /* warm */
    double t0=ork_now_us();                          /* (a) ntask separate ioctls */
    for(int i=0;i<ntask;i++){ sub.task_number=1; sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,1};
        if(rknpu_submit_ioctl(fd,&sub,-1)){bdestroy(fd,&W);bdestroy(fd,&O);return -1;} bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); }
    *us_unbatched=ork_now_us()-t0;
    sub.task_number=ntask; sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)ntask};
    t0=ork_now_us();                                 /* (b) one ioctl, ntask tasks */
    if(rknpu_submit_ioctl(fd,&sub,-1)){perror("batched SUBMIT");bdestroy(fd,&W);bdestroy(fd,&O);return -1;} bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE);
    *us_batched=ork_now_us()-t0;
    bdestroy(fd,&W);bdestroy(fd,&O); return 0;
}

/* RE: probe in-place K-slicing of a FULL-K weight buffer (for a single-layout decode+prefill).
 * Packs B[Kfull,N] fp16 in full-K tile layout, then runs ONE M=1 submit over k in [0,Kp) reading
 * from that buffer — i.e. the op processes Kp passes but the weights are laid out for Kfull. With
 * no override the per-N-tile stride is Kp's (N-tile 0 correct, 1+ wrong); pass reg/val overrides
 * (e.g. 0x1044, 0x1034, 0x1030 set to their full-K values) to hunt the stride register that makes
 * all N-tiles correct. C[N] = sum_{k<Kp} A[k]*B[k][n] if slicing is right. nov<=4. Returns 0/ok. */
int ork_npu_probe_slice_f16(ork_npu *c,int Kfull,int N,int Kp,int nov,
                            const uint32_t *ovr_reg,const uint32_t *ovr_val,
                            const f16 *A,const f16 *B,float *C){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(Kfull%32||Kp%32||N%16||N>c->soc->nmax||Kp>Kfull) return -2;
    struct buf W=bcreate(fd,(size_t)Kfull*N*2,0x403,-1); if(!W.cpu) return -2;
    int NN=N/16,KTf=Kfull/32; f16*bb=W.cpu;     /* full-K fp16 layout [Ntile][KTfull][16][32] */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KTf;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KTf*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*16+nl)];
    bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=bcreate(fd,(size_t)N*4,0x403,-1); if(!O.cpu){bdestroy(fd,&W);return -2;}
    f16*ad=c->Af.cpu; for(int j=0;j<Kp;j++)ad[j]=A[j]; bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t rc[REGCMD_N];
    synth(rc,1,Kp,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma,1,CBUF);
    setr(rc,REGCMD_N,0x201,0x1040,0xb1);
    for(int i=0;i<nov && i<4;i++) setr(rc,REGCMD_N,0x201,ovr_reg[i],ovr_val[i]);
    struct buf extra[2] = {W, O};
    if (validate_regcmd("probe_slice_f16", c, rc, REGCMD_N, NULL, extra, 2)) { bdestroy(fd,&W); bdestroy(fd,&O); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1;
    for(int rep=0;rep<2;rep++){ sub.timeout=60000;
        if(rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; continue; }
        bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(C,O.cpu,(size_t)N*4); ok=0; }
    bdestroy(fd,&W);bdestroy(fd,&O);
    return ok;
}

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
static void tile_i4_A(uint8_t*dst,const int8_t*A,int M,int K,int nib){
    int KT=K/32; memset(dst,0,(size_t)M*K/2);
    for(int kt=0;kt<KT;kt++)for(int m=0;m<M;m++)for(int kk=0;kk<32;kk++){
        size_t idx=((size_t)kt*M+m)*32+kk; uint8_t v=(uint8_t)(A[(size_t)m*K+kt*32+kk]&0xf);
        dst[idx/2]|= ((idx&1)^nib)?(v<<4):v;
    }
}
static void tile_i4_B(uint8_t*dst,const int8_t*B,int K,int N,int nib){
    int KT=K/32,NT=N/64; memset(dst,0,(size_t)K*N/2);
    for(int nt=0;nt<NT;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<64;nl++)for(int kk=0;kk<32;kk++){
        size_t idx=(((size_t)nt*KT+kt)*64+nl)*32+kk;
        uint8_t v=(uint8_t)(B[(size_t)(kt*32+kk)*N + (nt*64+nl)]&0xf);
        dst[idx/2]|= ((idx&1)^nib)?(v<<4):v;
    }
}
int ork_npu_probe_i4(ork_npu *c,int M,int K,int N,int nibB,int nibA,int nov,
                     const uint32_t *ovr_reg,const uint32_t *ovr_val,
                     const int8_t *A,const int8_t *B,int16_t *C){
    int fd=c->fd;
    if(K%32||N%64||N>c->soc->nmax) return -2;
    struct buf W=bcreate(fd,(size_t)K*N/2,0x403,-1); if(!W.cpu) return -2;        /* B int4: half bytes */
    tile_i4_B(W.cpu,B,K,N,nibB);
    bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=bcreate(fd,(size_t)M*N*2,0x403,-1); if(!O.cpu){bdestroy(fd,&W);return -2;}  /* int16 C, M rows */
    /* M-tiling: the captured W4A4 program runs M=1 per task; we replicate it per row. Each row's A is
     * its own native (K/32,1,32) block (contiguous K/2 bytes); each row's C is (N/8,1,8) = N int16. */
    uint8_t*ad=c->Af.cpu;
    for(int m=0;m<M;m++) tile_i4_A(ad+(size_t)m*(K/2), A+(size_t)m*K, 1, K, nibA);
    bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=0;
    for(int m=0;m<M && ok==0;m++){
        act(fd,RKNPU_ACT_RESET,0);
        uint32_t rc[REGCMD_I4_N];
        synth_i4(rc,1,K,N,(uint32_t)(c->Af.dma+(size_t)m*(K/2)),(uint32_t)W.dma,(uint32_t)(O.dma+(size_t)m*N*2));
        for(int i=0;i<nov && i<4;i++) setr(rc,REGCMD_I4_N,0x201,ovr_reg[i],ovr_val[i]);
        struct buf extra[2] = {W, O};
        if (validate_regcmd("probe_i4", c, rc, REGCMD_I4_N, NULL, extra, 2)) { bdestroy(fd,&W); bdestroy(fd,&O); return -1; }
        memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
        sub.timeout=60000; ok=-1;
        for(int rep=0;rep<2;rep++){ if(rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; continue; }
            bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; }
        if(ok==0){ int16_t*cr=(int16_t*)((char*)O.cpu+(size_t)m*N*2);   /* row m: native (N/8,1,8) */
            for(int nt=0;nt<N/8;nt++)for(int nl=0;nl<8;nl++) C[(size_t)m*N + nt*8+nl] = cr[nt*8+nl]; }
    }
    bdestroy(fd,&W);bdestroy(fd,&O);
    return ok;
}

/* RE/calibration: ONE multi-M int4 submit (mc=M) — the M-scheduler experiment for Tier 4b. A is laid
 * (K/32,M,32) native; B native (N/64,K/32,64,32); the regcmd's M-count regs are set to M (synth_i4
 * mc=M). Copies the RAW int16 output buffer (M*N int16, NO de-tile) to raw so the caller can deduce
 * the multi-M C layout against a CPU reference. Returns 0 if the submit completed, -1 wedged, -2 dims.
 * Single N-slice (N<=nmax), single core. See tools/i4_multim_probe.c. */
int ork_npu_probe_i4_mm(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,int16_t *raw){
    int fd=c->fd;
    if(K%32||N%64||N>c->soc->nmax||M<1) return -2;
    struct buf W=bcreate(fd,(size_t)K*N/2,0x403,-1); if(!W.cpu) return -2;
    tile_i4_B(W.cpu,B,K,N,0);
    bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=bcreate(fd,(size_t)M*N*2,0x403,-1); if(!O.cpu){bdestroy(fd,&W);return -2;}
    /* A layout selector via ORK_I4_ALAY: 0=(K/32,M,32) interleaved, 1=per-row contiguous (K/32,1,32)
     * x M (what the captured M=1 program reads). Lets the probe tell whether the program is single-row. */
    { int alay=getenv("ORK_I4_ALAY")?atoi(getenv("ORK_I4_ALAY")):0;
      if(alay) for(int m=0;m<M;m++) tile_i4_A((uint8_t*)c->Af.cpu+(size_t)m*(K/2),A+(size_t)m*K,1,K,0);
      else tile_i4_A(c->Af.cpu,A,M,K,0); }
    bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
    act(fd,RKNPU_ACT_RESET,0);
    uint32_t rc[REGCMD_I4_N];
    synth_i4(rc,M,K,N,(uint32_t)c->Af.dma,(uint32_t)W.dma,(uint32_t)O.dma);
    struct buf extra[2] = {W, O};
    if (validate_regcmd("probe_i4_mm", c, rc, REGCMD_I4_N, NULL, extra, 2)) { bdestroy(fd,&W); bdestroy(fd,&O); return -1; }
    memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
    int ok=-1;
    for(int rep=0;rep<2;rep++){ sub.timeout=60000; if(rknpu_submit_ioctl(fd,&sub,-1)){ ok=-1; continue; }
        bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; }
    if(ok==0) memcpy(raw,O.cpu,(size_t)M*N*2);
    bdestroy(fd,&W);bdestroy(fd,&O);
    return ok;
}

int ork_npu_probe_chain_i8(ork_npu *c, int S, int K, int N, const int8_t *A, const int8_t *B, int32_t *C) {
    int fd = c->fd, CBUF = c->soc->cbuf_elems;
    if (K % 32 || N % 32 || N > c->soc->nmax || S < 1 || S > 32) return -2;
    struct buf W = bcreate(fd, (size_t)K * N, 0x403,-1); if (!W.cpu) return -2;
    int NN = N / 32, KT = K / 32; int8_t *bb = W.cpu;
    for (int nt = 0; nt < NN; nt++) for (int kt = 0; kt < KT; kt++) for (int nl = 0; nl < 32; nl++) for (int kk = 0; kk < 32; kk++)
        bb[(size_t)nt * KT * 32 * 32 + (size_t)kt * 32 * 32 + nl * 32 + kk] = B[(size_t)(kt * 32 + kk) * N + (nt * 32 + nl)];
    bsync(fd, &W, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE); bsync(fd, &W, RKNPU_MEM_SYNC_TO_DEVICE);
    
    struct buf O = bcreate(fd, (size_t)S * 4096, 0x403,-1); if (!O.cpu) { bdestroy(fd, &W); return -2; }
    
    int8_t *ad = c->Af.cpu;
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < K; j++) ad[i * K + j] = A[i * K + j];
    }
    bsync(fd, &c->Af, RKNPU_MEM_SYNC_TO_DEVICE);
    
    act(fd, RKNPU_ACT_RESET, 0);
    
    uint32_t rc[REGCMD_I8_N];
    for (int i = 0; i < S; i++) {
        uint32_t act_dma = (uint32_t)(c->Af.dma + i * K);
        uint32_t out_dma = (uint32_t)(O.dma + i * 4096);
        synth_i8(rc, 1, K, N, act_dma, (uint32_t)W.dma, out_dma, 1, CBUF, 0);
        setr(rc, REGCMD_I8_N, 0x201, 0x1040, 0xb1);
        struct buf extra[2] = {W, O};
        if (validate_regcmd("probe_chain_i8", c, rc, REGCMD_I8_N, NULL, extra, 2)) { bdestroy(fd,&W); bdestroy(fd,&O); return -1; }
        
        if (i < S - 1) {
            uint64_t next_dma = c->regcmd.dma + (i + 1) * REGCMD_I8_N * 4;
            rc[216] = 0x0010 | ((next_dma & 0xffff) << 16);
            rc[217] = (0x0101 << 16) | ((next_dma >> 16) & 0xffff);
            rc[218] = 0x0014 | (0x0037 << 16);
            rc[219] = (0x0101 << 16) | (0);
        } else {
            rc[216] = 0;
            rc[217] = 0;
            rc[218] = 0x00000014;
            rc[219] = 0x01010000;
        }
        memcpy((char*)c->regcmd.cpu + i * sizeof(rc), rc, sizeof(rc));
    }
    bsync(fd, &c->regcmd, RKNPU_MEM_SYNC_TO_DEVICE);
    
    struct rknpu_task *t = c->task.cpu;
    memset(t, 0, S * sizeof(struct rknpu_task));
    for (int i = 0; i < S; i++) {
        t[i].enable_mask = 0xd;
        t[i].int_mask = 0x300;
        t[i].int_clear = 0x1ffff;
        t[i].regcfg_amount = 108;
        t[i].regcmd_addr = c->regcmd.dma + i * REGCMD_I8_N * 4;
    }
    bsync(fd, &c->task, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
    
    struct rknpu_submit sub; memset(&sub, 0, sizeof(sub));
    sub.task_start = 0;
    sub.task_number = 1; // HIDE the chained tasks from the kernel! The kernel rejects task_number > 1.
    sub.task_counter = 0;
    sub.priority = 0;
    sub.task_obj_addr = c->task.obj;
    sub.core_mask = RKNPU_CORE_AUTO_MASK;
    sub.subcore_task[0].task_start = 0;
    sub.subcore_task[0].task_number = 1; // Hide from subcore logic too!
    
    int ok = -1;
    for (int rep = 0; rep < 2; rep++) {
        sub.timeout = 60000;
        if (rknpu_submit_ioctl(fd, &sub, -1)) { ok = -1; continue; }
        bsync(fd, &O, RKNPU_MEM_SYNC_FROM_DEVICE);
        for (int i = 0; i < S; i++) {
            memcpy(C + i * N, (char*)O.cpu + i * 4096, (size_t)N * 4);
        }
        ok = 0;
    }
    
    bdestroy(fd, &W); bdestroy(fd, &O);
    return ok;
}

int ork_npu_benchmark_chain(ork_npu *c, int S, int K, int N, int iters) {
    int fd = c->fd, CBUF = c->soc->cbuf_elems;
    if (K % 32 || N % 32 || N > c->soc->nmax || S < 1 || S > 64) return -2;
    
    struct buf W = bcreate(fd, (size_t)K * N, 0x403,-1);
    struct buf A = bcreate(fd, (size_t)S * K, 0x403,-1);
    struct buf O = bcreate(fd, (size_t)S * 4096, 0x403,-1);
    
    struct buf regs_chain = bcreate(fd, (size_t)S * REGCMD_I8_N * 4, 0x403,-1);
    struct buf regs_sep = bcreate(fd, (size_t)S * REGCMD_I8_N * 4, 0x403,-1);
    
    struct buf task_chain = bcreate(fd, (size_t)S * sizeof(struct rknpu_task), 0x40b,-1);
    struct buf task_sep = bcreate(fd, (size_t)S * sizeof(struct rknpu_task), 0x40b,-1);
    
    if (!W.cpu || !A.cpu || !O.cpu || !regs_chain.cpu || !regs_sep.cpu || !task_chain.cpu || !task_sep.cpu) {
        fprintf(stderr, "[ork] ERROR: failed to allocate benchmark_chain buffers (IOMMU full?)\n");
        bdestroy(fd, &W); bdestroy(fd, &A); bdestroy(fd, &O);
        bdestroy(fd, &regs_chain); bdestroy(fd, &regs_sep);
        bdestroy(fd, &task_chain); bdestroy(fd, &task_sep);
        return -2;
    }
    
    memset(W.cpu, 1, (size_t)K * N);
    memset(A.cpu, 1, (size_t)S * K);
    bsync(fd, &W, RKNPU_MEM_SYNC_TO_DEVICE);
    bsync(fd, &A, RKNPU_MEM_SYNC_TO_DEVICE);
    
    uint32_t rc[REGCMD_I8_N];
    for (int i = 0; i < S; i++) {
        uint32_t act_dma = (uint32_t)(A.dma + i * K);
        uint32_t out_dma = (uint32_t)(O.dma + i * 4096);
        synth_i8(rc, 1, K, N, act_dma, (uint32_t)W.dma, out_dma, 1, CBUF, 0);
        setr(rc, REGCMD_I8_N, 0x201, 0x1040, 0xb1);
        
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
    bsync(fd, &regs_chain, RKNPU_MEM_SYNC_TO_DEVICE);
    
    for (int i = 0; i < S; i++) {
        uint32_t act_dma = (uint32_t)(A.dma + i * K);
        uint32_t out_dma = (uint32_t)(O.dma + i * 4096);
        synth_i8(rc, 1, K, N, act_dma, (uint32_t)W.dma, out_dma, 1, CBUF, 0);
        setr(rc, REGCMD_I8_N, 0x201, 0x1040, 0xb1);
        rc[216] = 0; rc[217] = 0; rc[218] = 0x00000014; rc[219] = 0x01010000;
        memcpy((char*)regs_sep.cpu + i * REGCMD_I8_N * 4, rc, sizeof(rc));
    }
    bsync(fd, &regs_sep, RKNPU_MEM_SYNC_TO_DEVICE);
    
    struct rknpu_task *tk_chain = task_chain.cpu;
    memset(tk_chain, 0, S * sizeof(struct rknpu_task));
    for (int i = 0; i < S; i++) {
        tk_chain[i].enable_mask = 0xd;
        tk_chain[i].int_mask = 0x300;
        tk_chain[i].int_clear = 0x1ffff;
        tk_chain[i].regcfg_amount = 108;
        tk_chain[i].regcmd_addr = regs_chain.dma + i * REGCMD_I8_N * 4;
    }
    bsync(fd, &task_chain, RKNPU_MEM_SYNC_TO_DEVICE);
    
    struct rknpu_task *tk_sep = task_sep.cpu;
    memset(tk_sep, 0, S * sizeof(struct rknpu_task));
    for (int i = 0; i < S; i++) {
        tk_sep[i].enable_mask = 0xd;
        tk_sep[i].int_mask = 0x300;
        tk_sep[i].int_clear = 0x1ffff;
        tk_sep[i].regcfg_amount = 108;
        tk_sep[i].regcmd_addr = regs_sep.dma + i * REGCMD_I8_N * 4;
    }
    bsync(fd, &task_sep, RKNPU_MEM_SYNC_TO_DEVICE);
    
    act(fd, RKNPU_ACT_RESET, 0);
    struct rknpu_submit sub; memset(&sub, 0, sizeof(sub));
    sub.flags = 0x5;
    sub.task_number = S;
    sub.task_obj_addr = task_chain.obj;
    sub.core_mask = RKNPU_CORE0_MASK;
    sub.fence_fd = -1;
    sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)S};
    sub.timeout = 60000;
    if (rknpu_submit_ioctl(fd, &sub, -1)) {
        perror("Warmup failed");
    }
    bsync(fd, &O, RKNPU_MEM_SYNC_FROM_DEVICE);
    
    double t_sep_start = ork_now_us();
    for (int it = 0; it < iters; it++) {
        for (int s = 0; s < S; s++) {
            struct rknpu_task *tk_dest = c->task.cpu;
            memcpy(tk_dest, &tk_sep[s], sizeof(struct rknpu_task));
            bsync(fd, &c->task, RKNPU_MEM_SYNC_TO_DEVICE);
            
            struct rknpu_submit sub_s; memset(&sub_s, 0, sizeof(sub_s));
            sub_s.flags = 0x5;
            sub_s.task_number = 1;
            sub_s.task_obj_addr = c->task.obj;
            sub_s.core_mask = RKNPU_CORE0_MASK;
            sub_s.fence_fd = -1;
            sub_s.subcore_task[0] = sub_s.subcore_task[1] = sub_s.subcore_task[2] = (struct rknpu_subcore_task){0, 1};
            sub_s.timeout = 60000;
            if (rknpu_submit_ioctl(fd, &sub_s, -1)) {
                perror("Separate submit failed");
                break;
            }
            bsync(fd, &O, RKNPU_MEM_SYNC_FROM_DEVICE);
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
        if (rknpu_submit_ioctl(fd, &sub_c, -1)) {
            perror("Chained submit failed");
            break;
        }
        bsync(fd, &O, RKNPU_MEM_SYNC_FROM_DEVICE);
    }
    double t_chain = ork_now_us() - t_chain_start;
    
    double avg_sep = t_sep / iters;
    double avg_chain = t_chain / iters;
    
    printf("  %d separate submits: total time = %.0f us (avg per submit loop = %.1f us, per matmul = %.1f us)\n",
           S, t_sep, avg_sep, avg_sep / S);
    printf("  1 chained submit (x%d): total time = %.0f us (avg per submit loop = %.1f us, per matmul = %.1f us)\n",
           S, t_chain, avg_chain, avg_chain / S);
    printf("  Speedup: %.2fx\n", avg_sep / avg_chain);
    
    bdestroy(fd, &W); bdestroy(fd, &A); bdestroy(fd, &O);
    bdestroy(fd, &regs_chain); bdestroy(fd, &regs_sep);
    bdestroy(fd, &task_chain); bdestroy(fd, &task_sep);
    return 0;
}
const char *ork_npu_version(void){
#ifdef ORK_GIT_HASH
    static char v[64]; snprintf(v, sizeof v, "%s+g%s", ORK_NPU_VERSION, ORK_GIT_HASH); return v;
#else
    return ORK_NPU_VERSION;   /* no git at build time (e.g. native board build) → semver only */
#endif
}

/* .orkpack compat token = library MAJOR version (atoi stops at the first '.'): a format-breaking change
 * requires a major bump, while minor/patch stay backward-compatible. See ork_npu.h. */
uint32_t ork_pack_format_version(void){ return (uint32_t)atoi(ORK_NPU_VERSION); }

/* Max M rows a single full-K int8 submit handles at this K (mirrors run()'s M>1 Bf tiling, npu.c
 * "Tier 1c-ii"). Each chain link is ONE full-K submit, so a task's M must not exceed this — else the
 * caller must split the task into M-tiles. Guards against wedging the (shared) NPU on an oversized mc. */
static int chain_fullk_mcap_i8(ork_npu *c, int K) {
    int RB = 2 * c->soc->cbuf_elems, R = RB / K; if (R < 1) R = 1;
    { int rp2 = 1; while (rp2 * 2 <= R) rp2 *= 2; R = rp2; }
    double scale = (double)K / 512.0; int base = (int)(177.0 - 15.0 * (scale - 1.0)), slope = (int)(15.0 * scale);
    int mg_max = base >= 0x1b ? (base - 0x1b) / slope + 1 : 0;
    int chunk = mg_max * 64; if (chunk < 1) chunk = 1;   /* schedule-valid max rows (mg_max*64), not R-1 — see "weight-DMA amortization" in AGENTS.md */
    return chunk;
}

int ork_mm_run_chain_i8(ork_npu *c, int S, const ork_mm_task_i8 *tasks) {
    if (!c) return -1;
    if (S < 1 || S > 1024) return -2;
    if (!tasks) return -2;
    if (tasks[0].w) {  /* chained weights share one submit => one domain; swap in that domain's scratch */
        if (tasks[0].w->domain != c->dom_active || (tasks[0].w->domain!=0 && !c->dom_save)) dom_activate(c, tasks[0].w->domain);
    }

    /* A single matmul has nothing to chain — dispatch to the optimized run_i8 path (multi-core
     * N-split / full-K single-submit decode via the auto-tuner). The chain path is single-core and
     * allocs per-call scratch, so it must only be used to batch S>1 independent matmuls. */
    if (S == 1) return ork_mm_run_i8(c, tasks[0].w, tasks[0].M, tasks[0].A, tasks[0].C);

    int fd = c->fd, CBUF = c->soc->cbuf_elems;
    
    // 1. Validate all tasks
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        if (!w) return -2;
        if (w->dtype != DT_I8) return -2;
        if (tasks[i].M <= 0) return -2;
        if (w->K % 32 || w->N % 32) return -2;
        // Each chain link is ONE full-K submit. Need a single-slice weight: either Sk==1 (Bb[0] holds
        // the whole K) or a Bf full-K buffer (K<=10752; built by pack/repack for the MoE experts).
        // K=2048 experts pack Sk=2 but carry Bf — use it so they can chain. N must be a single slice.
        if (w->Sn != 1) return -2;
        if (w->Sk != 1 && !w->Bf) return -2;
        // The full-K Bf submit uses synth_i8(sched=1), whose 0x1040 K-reduction schedule is only valid for
        // K%512==0 && K<=4096 (same envelope as run()'s M>1 Bf path; 512/1024 are covered, 1536-4096 too).
        // For other K (e.g. 768 down_proj, or K>4096) a full-K single submit is WRONG — reject so the caller
        // falls back to per-task run_i8 (which K-splits correctly). -3 distinguishes this from bad-arg -2.
        if (w->K % 512 != 0 || w->K > 4096) return -3;
        // M>mcap is fine — the synth loop M-tiles it into multiple chained programs (Step B below).
        if (check_overlap("ork_mm_run_chain_i8", (uintptr_t)tasks[i].A, (uintptr_t)tasks[i].A + (size_t)tasks[i].M * w->K, (uintptr_t)tasks[i].C, (uintptr_t)tasks[i].C + (size_t)tasks[i].M * w->N * 4)) return -1;
    }

    // 2. State transition reset for int8 mode
    // We must ALWAYS reset the NPU before chained execution, because previous
    // single-submit (non-chained) int8 runs leave the NPU in a state that corrupts
    // the first chained task's output if reps=1.
    if (c->last_dt != 3 /* DT_I8_CHAIN */) {
        if (!ORK_I8_LIVE(c->last_dt)) act(fd, RKNPU_ACT_RESET, 0);   /* reset only entering int8; warmup handles int8->chain */
        c->warmed = 0;
        c->last_dt = 3; // DT_I8_CHAIN
        c->ccsz = 0; // invalidate Cc size
    }

    // 3. Resolve buffers and cache coherency
    struct buf tmp_A[1024];
    struct buf tmp_C[1024];
    memset(tmp_A, 0, sizeof(tmp_A));
    memset(tmp_C, 0, sizeof(tmp_C));

    uint32_t act_dma[1024];
    uint32_t out_dma[1024];
    struct buf *cbufs[1024];
    memset(cbufs, 0, sizeof(cbufs));

    int ok = 0;
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        int M = tasks[i].M;
        int K = w->K;
        int N = w->N;

        // Resolve input activations buffer
        struct buf *abuf = dma_find(c, tasks[i].A);
        if (abuf) {
            bsync(fd, abuf, RKNPU_MEM_SYNC_TO_DEVICE);
            act_dma[i] = (uint32_t)(abuf->dma + ((const char*)tasks[i].A - (const char*)abuf->cpu));
        } else {
            tmp_A[i] = bcreate(fd, (size_t)M * K, 0x403, c->dom_active);
            if (!tmp_A[i].cpu) { ok = -1; goto cleanup; }
            memcpy(tmp_A[i].cpu, tasks[i].A, (size_t)M * K);
            bsync(fd, &tmp_A[i], RKNPU_MEM_SYNC_TO_DEVICE);
            act_dma[i] = (uint32_t)tmp_A[i].dma;
        }

        // Resolve output buffer
        struct buf *cbuf = dma_find(c, tasks[i].C);
        if (cbuf) {
            bsync(fd, cbuf, RKNPU_MEM_SYNC_TO_DEVICE);
            out_dma[i] = (uint32_t)(cbuf->dma + ((const char*)tasks[i].C - (const char*)cbuf->cpu));
            cbufs[i] = cbuf;
        } else {
            tmp_C[i] = bcreate(fd, (size_t)M * N * 4, 0x403, c->dom_active);
            if (!tmp_C[i].cpu) { ok = -1; goto cleanup; }
            bsync(fd, &tmp_C[i], RKNPU_MEM_SYNC_TO_DEVICE);
            out_dma[i] = (uint32_t)tmp_C[i].dma;
        }
    }

    // 4. Synthesize REGCMD blocks and link the chain
    struct buf extra[2048];
    int extra_n = 0;
    for (int j = 0; j < S; j++) {
        if (tmp_A[j].cpu) extra[extra_n++] = tmp_A[j];
        if (tmp_C[j].cpu) extra[extra_n++] = tmp_C[j];
    }
    // Each task is one full-K matmul of M rows; a single submit handles <= mcap rows, so a task with
    // M>mcap expands into ceil(M/mcap) M-tile programs (offsetting into its A/C buffers). ALL programs
    // across ALL tasks are PC-chained into one submit. Count total programs P first (must fit buffers).
    int prog_off[1025];   // prog_off[i] = first program index of task i (S<=1024)
    int P = 0;
    for (int i = 0; i < S; i++) {
        int mcap = chain_fullk_mcap_i8(c, tasks[i].w->K);
        prog_off[i] = P;
        P += (tasks[i].M + mcap - 1) / mcap;
    }
    if (P > 1024) { ok = -2; goto cleanup; }   // too many M-tiles for the chain regcmd/task buffers

    // run_chain_i8 is SINGLE-CORE: it PC-chains all P programs into ONE submit (low latency, one ioctl
    // for S matmuls — e.g. decode QKV/gate-up sharing an input). Cross-core throughput is now served by
    // ork_mm_run_stream_i8 (async round-robin, ~3x); the old barrier fan-out here was superseded (~1.3x).
    uint32_t rc[REGCMD_I8_N + 4];
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        int M = tasks[i].M, K = w->K, N = w->N, mcap = chain_fullk_mcap_i8(c, K);
        // full-K single submit: Bf[0] (the K<=10752 full-K layout, e.g. Sk=2 experts) if present,
        // else Bb[0] (which holds the whole K only when Sk==1). Both are the synth_i8 tile layout.
        uint32_t bdma = w->Bf ? (uint32_t)w->Bf[0].dma : (uint32_t)w->Bb[0].dma;
        int p = prog_off[i];
        for (int m0 = 0; m0 < M; m0 += mcap, p++) {
            int mc = (M - m0 < mcap) ? (M - m0) : mcap;
            memset(rc, 0, sizeof(rc));
            // Let synth_i8(sched=1) set the 0x1040 K-reduction schedule from mc (= ceil(mc/64) group).
            // Do NOT hardcode it (the old 0xb1 was an M=1 value; for mc>16 it computes rows past the
            // first 64-group against the wrong K-partition — same class as the full-K prefill bug).
            synth_i8(rc, mc, K, N, act_dma[i] + (uint32_t)((size_t)m0 * K),
                     bdma, out_dma[i] + (uint32_t)((size_t)m0 * N * 4), 1, CBUF, 0);
            if (validate_regcmd("run_chain_i8", c, rc, REGCMD_I8_N, w, extra, extra_n)) { ok = -1; goto cleanup; }
            if (p < P - 1) {   // PC-chain: this program jumps to the next; the last keeps the template's raise-interrupt tail
                uint64_t next_dma = c->regcmd.dma + (size_t)(p + 1) * REGCMD_I8_N * 4;
                rc[216] = 0x0010 | ((next_dma & 0xffff) << 16);
                rc[217] = (0x0101 << 16) | ((next_dma >> 16) & 0xffff);
                rc[218] = 0x0014 | (0x0037 << 16);
                rc[219] = (0x0101 << 16) | (0);
            }
            memcpy((char*)c->regcmd.cpu + (size_t)p * REGCMD_I8_N * 4, rc, REGCMD_I8_N * 4);
        }
    }
    bsync(fd, &c->regcmd, RKNPU_MEM_SYNC_TO_DEVICE);

    // One rknpu_task per program (P total, PC-chained), one single-core submit.
    int submit_ok = 0;
    struct rknpu_task *t = c->task.cpu;
    memset(t, 0, (size_t)P * sizeof(struct rknpu_task));
    for (int p = 0; p < P; p++) {
        t[p].enable_mask = 0xd; t[p].int_mask = 0x300; t[p].int_clear = 0x1ffff;
        t[p].regcfg_amount = 108;
        t[p].regcmd_addr = c->regcmd.dma + (size_t)p * REGCMD_I8_N * 4;
    }
    bsync(fd, &c->task, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
    static int tc = -2;
    if (tc == -2) { const char *e = getenv("ORK_NPU_TESTCORE"); tc = e ? atoi(e) : 0; if (tc < 0 || tc > 2) tc = 0; }
    struct rknpu_submit sub; memset(&sub, 0, sizeof(sub));
    sub.flags = 0x5; sub.task_number = P; sub.task_obj_addr = c->task.obj;
    sub.core_mask = 1u << tc; sub.fence_fd = -1;
    sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)P};
    int reps = c->warmed ? 1 : 2;
    for (int rep = 0; rep < reps; rep++) {
        int last = (rep == reps - 1);
        sub.timeout = 60000;
        if (rknpu_submit_ioctl(fd, &sub, tasks[0].w->domain)) { if (last) { perror("SUBMIT chained"); submit_ok = -1; } continue; }
        submit_ok = 0;
    }
    c->warmed = 1;

    // 7. Sync memory back and copy results
    for (int i = 0; i < S; i++) {
        if (cbufs[i]) {
            bsync(fd, cbufs[i], RKNPU_MEM_SYNC_FROM_DEVICE);
        } else {
            bsync(fd, &tmp_C[i], RKNPU_MEM_SYNC_FROM_DEVICE);
            if (submit_ok == 0) {
                memcpy(tasks[i].C, tmp_C[i].cpu, (size_t)tasks[i].M * tasks[i].w->N * 4);
            }
        }
    }
    ok = submit_ok;

cleanup:
    for (int i = 0; i < S; i++) {
        bdestroy(fd, &tmp_A[i]);
        bdestroy(fd, &tmp_C[i]);
    }
    return ok;
}

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
    pin_big_core(i);
    int k;
    a->rc = 0;
    uint32_t rc[REGCMD_I8_N + 4];
    while ((k = __atomic_fetch_add(a->ctr, 1, __ATOMIC_SEQ_CST)) < a->S) {
        const ork_mm_task_i8 *t = &a->tasks[k];
        ork_w *w = t->w; int M = t->M, K = w->K, N = w->N, mcap = chain_fullk_mcap_i8(c, K);
        uint32_t bdma = w->Bf ? (uint32_t)w->Bf[0].dma : (uint32_t)w->Bb[0].dma;
        memcpy(c->maf[i].cpu, t->A, (size_t)M * K);
        bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE);
        int ntiles = (M + mcap - 1) / mcap, p = 0;
        for (int m0 = 0; m0 < M; m0 += mcap, p++) {
            int mc = (M - m0 < mcap) ? (M - m0) : mcap;
            memset(rc, 0, sizeof rc);
            synth_i8(rc, mc, K, N, (uint32_t)(c->maf[i].dma + (size_t)m0 * K), bdma,
                     (uint32_t)(c->mcc[i].dma + (size_t)m0 * N * 4), 1, CBUF, 0);
            if (p < ntiles - 1) {
                uint64_t nd = c->mrc[i].dma + (size_t)(p + 1) * REGCMD_I8_N * 4;
                rc[216] = 0x0010 | ((nd & 0xffff) << 16); rc[217] = (0x0101 << 16) | ((nd >> 16) & 0xffff);
                rc[218] = 0x0014 | (0x0037 << 16); rc[219] = (0x0101 << 16) | (0);
            }
            memcpy((char *)c->mrc[i].cpu + (size_t)p * REGCMD_I8_N * 4, rc, REGCMD_I8_N * 4);
        }
        bsync(fd, &c->mrc[i], RKNPU_MEM_SYNC_TO_DEVICE);
        struct rknpu_task *mt = c->mtk[i].cpu; memset(mt, 0, (size_t)ntiles * sizeof *mt);
        for (int q = 0; q < ntiles; q++) {
            mt[q].enable_mask = 0xd; mt[q].int_mask = 0x300; mt[q].int_clear = 0x1ffff;
            mt[q].regcfg_amount = 108; mt[q].regcmd_addr = c->mrc[i].dma + (size_t)q * REGCMD_I8_N * 4;
        }
        bsync(fd, &c->mtk[i], RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
        struct rknpu_submit sub; memset(&sub, 0, sizeof sub);
        sub.flags = 0x5; sub.task_number = ntiles; sub.task_obj_addr = c->mtk[i].obj; sub.core_mask = 1u << i; sub.fence_fd = -1;
        sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)ntiles};
        sub.timeout = 60000;
        /* A freshly-allocated NPU output buffer returns stale on its FIRST write, so prime THIS core's
         * output buffer with a throwaway submit on its first use (mirror mcworker's reps=c->mwarm[i]?1:2).
         * Per-core + deterministic: the old coarse whole-stream 2-pass could miss a core whose buffer the
         * dynamic task counter left idle on the warmup pass, yielding a flaky stale (zero) result. */
        int reps = c->mwarm[i] ? 1 : 2;
        for (int rep = 0; rep < reps; rep++) {
            if (rknpu_submit_ioctl(fd, &sub, w->domain)) { if (rep == reps - 1) a->rc = -1; continue; }
            bsync(fd, &c->mcc[i], RKNPU_MEM_SYNC_FROM_DEVICE);
        }
        c->mwarm[i] = 1;   /* this core's buffer index is disjoint per worker — no cross-thread race */
        memcpy(t->C, c->mcc[i].cpu, (size_t)M * N * 4);
    }
    return NULL;
}

/* Run S independent int8 matmuls as an async round-robin stream across the NPU cores. Each task's weight
 * must be single-slice (Sk==1 or Bf full-K) and single N-slice; A/C are plain host buffers (copied via the
 * per-core staging buffers — no zero-copy DMA here). Returns 0/ok, -1 submit fail, -2 bad arg. */
int ork_mm_run_stream_i8(ork_npu *c, int S, const ork_mm_task_i8 *tasks) {
    if (!c || S < 1 || !tasks) return -2;
    /* per-core scratch lives in the active domain; stream tasks share one domain (tasks[0].w) */
    if (tasks[0].w && (tasks[0].w->domain != c->dom_active || (tasks[0].w->domain!=0 && !c->dom_save))) dom_activate(c, tasks[0].w->domain);
    const int mrc_cap = 65536 / (REGCMD_I8_N * 4);
    size_t maxMK = 0, maxMN4 = 0;
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        if (!w || w->dtype != DT_I8 || tasks[i].M <= 0) return -2;
        if (w->Sn != 1 || !w->Bf) return -2;
        // The full-K Bf single-submit is only schedule-valid for K%512==0 && K<=4096 (same envelope as
        // run()'s M>1 Bf path; the 0x1040 K-reduction schedule breaks outside it). Caller must fall back
        // to per-task run_i8 (which K-splits) for other K. Return -3 so it's distinguishable.
        if (w->K % 512 != 0 || w->K > 4096) return -3;
        if ((tasks[i].M + chain_fullk_mcap_i8(c, w->K) - 1) / chain_fullk_mcap_i8(c, w->K) > mrc_cap) return -2;
        size_t mk = (size_t)tasks[i].M * w->K, mn = (size_t)tasks[i].M * w->N * 4;
        if (mk > maxMK) maxMK = mk; if (mn > maxMN4) maxMN4 = mn;
    }
    int fd = c->fd;
    // RESET is only for ENTERING int8 from fp16/int4/cold — switching among int8 markers (single-core
    // DT_I8 <-> chain/stream 3) needs none (see ORK_I8_LIVE), so decode can interleave run_i8 singletons
    // with run_stream groups without a ~107ms soft-reset per matmul. Freshly-allocated per-core output
    // buffers are primed deterministically by stream_worker's reps=2-on-first-use (mwarm[i]); a reset
    // here clears every core's mwarm so they re-prime.
    if (c->last_dt != 3) { if (!ORK_I8_LIVE(c->last_dt)) { act(fd, RKNPU_ACT_RESET, 0); c->warmed = 0; for (int i = 0; i < ORK_MAXCORE; i++) c->mwarm[i] = 0; } c->last_dt = 3; }
    // Core count is caller-configurable up to the SoC max: budget() honors ork_npu_set_core_budget()
    // and the ORK_NPU_MC env (both capped to soc->cores). Capped to S (no more cores than tasks).
    int nc = budget(c, 2); if (nc > ORK_MAXCORE) nc = ORK_MAXCORE; if (nc > S) nc = S; if (nc < 1) nc = 1;
    if (mc_ensure(c, nc)) return -1;
    for (int i = 0; i < nc; i++) {   /* size per-core staging (A) + output (C) buffers to the largest task */
        if (c->maf[i].size < maxMK) { bdestroy(fd, &c->maf[i]); c->maf[i] = bcreate(fd, maxMK, 0x403, c->dom_active); if (!c->maf[i].cpu) return -1; }
        if (c->mccsz[i] < maxMN4) { bdestroy(fd, &c->mcc[i]); c->mcc[i] = bcreate(fd, maxMN4, 0x403, c->dom_active); c->mccsz[i] = maxMN4; if (!c->mcc[i].cpu) return -1; c->mwarm[i] = 0; /* fresh output buffer -> re-prime */ }
    }
    int rc = 0;
    npu_pool_ensure(c);
    struct streamw sw[ORK_MAXCORE];
    int ctr = 0;   /* single pass: stream_worker primes each fresh per-core buffer via reps=2-on-first-use */
    for (int i = 0; i < nc; i++) sw[i] = (struct streamw){c, i, S, tasks, &ctr, 0};
    pthread_mutex_lock(&c->pmu);
    c->pjob = sw; c->pjob_nc = nc; c->pjob_fn = stream_worker; c->pjob_stride = sizeof(struct streamw);
    c->pdone = 0; c->pgen++; pthread_cond_broadcast(&c->pgo);
    pthread_mutex_unlock(&c->pmu);
    stream_worker(&sw[0]);                            /* core 0 on the calling thread */
    pthread_mutex_lock(&c->pmu); while (c->pdone < nc - 1) pthread_cond_wait(&c->pdn, &c->pmu); pthread_mutex_unlock(&c->pmu);
    for (int i = 0; i < nc; i++) if (sw[i].rc) rc = -1;
    c->warmed = 1;
    return rc;
}

int ork_mm_run_chain_i4(ork_npu *c, int S, const ork_mm_task_i4 *tasks) {
    if (!c) return -1;
    if (S < 1 || S > 1024) return -2;
    if (!tasks) return -2;

    /* Single matmul: use the optimized run_i4 path (multi-core column-split) rather than the
     * single-core chain path. Chaining only pays off when batching S>1 independent matmuls. */
    if (S == 1) return ork_mm_run_i4(c, tasks[0].w, tasks[0].M, tasks[0].A, tasks[0].C);

    /* chained weights share one submit => one domain; swap in that domain's scratch */
    if (tasks[0].w && (tasks[0].w->domain != c->dom_active || (tasks[0].w->domain!=0 && !c->dom_save))) dom_activate(c, tasks[0].w->domain);
    int fd = c->fd;
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        if (!w || w->dtype != DT_I4) return -2;
        if (tasks[i].M != 1) return -2;
        if (w->Sn != 1 || w->Sk != 1) return -2;
        if (check_overlap("ork_mm_run_chain_i4", (uintptr_t)tasks[i].A, (uintptr_t)tasks[i].A + (size_t)tasks[i].M * w->K, (uintptr_t)tasks[i].C, (uintptr_t)tasks[i].C + (size_t)tasks[i].M * w->N * 4)) return -1;
    }

    if (c->last_dt != 4 /* DT_I4_CHAIN */) {
        act(fd, RKNPU_ACT_RESET, 0);
        c->warmed = 0;
        c->last_dt = 4;
        c->ccsz = 0;
    }

    int ok = 0;
    int max_K = 0, max_N = 0;
    for (int i = 0; i < S; i++) {
        if (tasks[i].w->K > max_K) max_K = tasks[i].w->K;
        if (tasks[i].w->N > max_N) max_N = tasks[i].w->N;
        struct buf *abuf = dma_find(c, tasks[i].A);
        if (abuf) bsync(fd, abuf, RKNPU_MEM_SYNC_FROM_DEVICE);
    }

    struct buf chain_A = bcreate(fd, (size_t)S * max_K, 0x403, c->dom_active);
    struct buf chain_C = bcreate(fd, (size_t)S * max_N * 2, 0x403, c->dom_active);
    if (!chain_A.cpu || !chain_C.cpu) {
        if (chain_A.cpu) bdestroy(fd, &chain_A);
        if (chain_C.cpu) bdestroy(fd, &chain_C);
        return -1;
    }

    uint32_t act_dma[1024];
    uint32_t out_dma[1024];

    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        uint8_t *A_dst = (uint8_t*)chain_A.cpu + (size_t)i * max_K;
        tile_i4_Aslice(A_dst, tasks[i].A, 0, w->K);
        act_dma[i] = (uint32_t)(chain_A.dma + (size_t)i * max_K);
        out_dma[i] = (uint32_t)(chain_C.dma + (size_t)i * max_N * 2);
    }
    bsync(fd, &chain_A, RKNPU_MEM_SYNC_TO_DEVICE);

    struct buf extra[2] = {chain_A, chain_C};
    uint32_t rc[REGCMD_I4_N];

    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        synth_i4(rc, 1, w->K, w->N, act_dma[i], (uint32_t)w->Bb[0].dma, out_dma[i]);
        if (validate_regcmd("run_chain_i4", c, rc, REGCMD_I4_N, w, extra, 2)) { ok = -1; goto cleanup; }

        if (i < S - 1) {
            uint64_t next_dma = c->regcmd.dma + (i + 1) * REGCMD_I4_N * 4;
            rc[216] = 0x0010 | ((next_dma & 0xffff) << 16);
            rc[217] = (0x0101 << 16) | ((next_dma >> 16) & 0xffff);
            rc[218] = 0x0014 | (0x0037 << 16);
            rc[219] = (0x0101 << 16) | (0);
        } else {
            rc[216] = 0;
            rc[217] = 0;
            rc[218] = 0x00000014;
            rc[219] = 0x01010000;
        }
        memcpy((char*)c->regcmd.cpu + i * REGCMD_I4_N * 4, rc, sizeof(rc));
    }
    bsync(fd, &c->regcmd, RKNPU_MEM_SYNC_TO_DEVICE);

    /* Mirror the validated i4_mcworker multi-task path exactly: one rknpu_task per chained regcmd,
     * task_number=S, subcore={0,S}, and the same reps/submit discipline (rknpu_submit_ioctl with a
     * cold-buffer warmup rep + bsync(C) between reps). The kernel programs first_task+last_task and
     * the HW PC-chain (rc[216..219]) walks the middle; it waits until the HW task counter reaches S. */
    struct rknpu_task *t = c->task.cpu;
    memset(t, 0, (size_t)S * sizeof(struct rknpu_task));
    for (int i = 0; i < S; i++) {
        t[i].enable_mask = 0xd;
        t[i].int_mask = 0x300;
        t[i].int_clear = 0x1ffff;
        t[i].regcfg_amount = 116;
        t[i].regcmd_addr = c->regcmd.dma + (size_t)i * REGCMD_I4_N * 4;
    }
    bsync(fd, &c->task, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);

    static int tc = -2;
    if (tc == -2) { const char* e = getenv("ORK_NPU_TESTCORE"); tc = e ? atoi(e) : 0; if (tc < 0 || tc > 2) tc = 0; }

    struct rknpu_submit sub; memset(&sub, 0, sizeof sub);
    sub.flags = 0x5; sub.task_number = S; sub.task_obj_addr = c->task.obj; sub.fence_fd = -1;
    sub.core_mask = 1u << tc;
    sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, S};

    int reps = c->warmed ? 1 : 2;
    for (int rep = 0; rep < reps; rep++) {
        int last = (rep == reps - 1);
        sub.timeout = 60000;
        if (rknpu_submit_ioctl(fd, &sub, tasks[0].w->domain)) { if (last) { ok = -1; goto cleanup; } continue; }
        bsync(fd, &chain_C, RKNPU_MEM_SYNC_FROM_DEVICE);
    }
    c->warmed = 1;

    bsync(fd, &chain_C, RKNPU_MEM_SYNC_FROM_DEVICE);
    for (int i = 0; i < S; i++) {
        int16_t *o = (int16_t*)((uint8_t*)chain_C.cpu + (size_t)i * max_N * 2);
        int32_t *C = tasks[i].C;
        int N = tasks[i].w->N;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        int col = 0;
        for (; col <= N - 16; col += 16) {
            int16x8_t vo16_0 = vld1q_s16(&o[col]);
            int16x8_t vo16_1 = vld1q_s16(&o[col + 8]);
            vst1q_s32(&C[col], vmovl_s16(vget_low_s16(vo16_0)));
            vst1q_s32(&C[col + 4], vmovl_s16(vget_high_s16(vo16_0)));
            vst1q_s32(&C[col + 8], vmovl_s16(vget_low_s16(vo16_1)));
            vst1q_s32(&C[col + 12], vmovl_s16(vget_high_s16(vo16_1)));
        }
        for (; col < N; col++) {
            C[col] = o[col];
        }
#else
        for (int col = 0; col < N; col++) {
            C[col] = o[col];
        }
#endif

        struct buf *cbuf = dma_find(c, tasks[i].C);
        if (cbuf) bsync(fd, cbuf, RKNPU_MEM_SYNC_TO_DEVICE);
    }

cleanup:
    bdestroy(fd, &chain_A);
    bdestroy(fd, &chain_C);
    return ok;
}

/* ---- ASYNC ROUND-ROBIN STREAM, int4 (ork_mm_run_stream_i4) ----
 * int4 analog of ork_mm_run_stream_i8: dispatch S independent W4A4 matmuls to per-core pool workers that
 * PULL the next task dynamically (atomic counter, no barrier), each running it as a single-core submit on
 * its own core. int4 is single-row per regcmd (synth_i4 mc=1), so a task's M rows become M single-row
 * regcmds PC-chained on one core (mirror run_chain_i4's chaining + the int16->int32 de-tile), submitted
 * task_number=M on core_mask=1<<i. Weights must be single-slice (Sn==1 && Sk==1). A/C are plain host
 * buffers staged via the per-core mrc/mtk/maf/mcc buffers (no zero-copy). 0/ok, -1 submit, -2 bad arg. */
struct streamw4 { ork_npu *c; int core; int S; const ork_mm_task_i4 *tasks; int *ctr; int rc; };
static void *stream_worker_i4(void *vp) {
    struct streamw4 *a = vp; ork_npu *c = a->c; int fd = c->fd, i = a->core;
    pin_big_core(i);
    int k; a->rc = 0;
    uint32_t rc[REGCMD_I4_N];
    while ((k = __atomic_fetch_add(a->ctr, 1, __ATOMIC_SEQ_CST)) < a->S) {
        const ork_mm_task_i4 *t = &a->tasks[k];
        ork_w *w = t->w; int M = t->M, K = w->K, N = w->N;
        uint32_t bdma = (uint32_t)w->Bb[0].dma;
        uint8_t *abase = c->maf[i].cpu;                       /* per-row stride K bytes (nibble-pack uses K/2) */
        for (int m = 0; m < M; m++) tile_i4_Aslice(abase + (size_t)m * K, t->A + (size_t)m * K, 0, K);
        bsync(fd, &c->maf[i], RKNPU_MEM_SYNC_TO_DEVICE);
        for (int m = 0; m < M; m++) {                         /* one single-row regcmd per row, PC-chained */
            memset(rc, 0, sizeof rc);
            synth_i4(rc, 1, K, N, (uint32_t)(c->maf[i].dma + (size_t)m * K), bdma,
                     (uint32_t)(c->mcc[i].dma + (size_t)m * N * 2));
            if (m < M - 1) {
                uint64_t nd = c->mrc[i].dma + (size_t)(m + 1) * REGCMD_I4_N * 4;
                rc[216] = 0x0010 | ((nd & 0xffff) << 16); rc[217] = (0x0101 << 16) | ((nd >> 16) & 0xffff);
                rc[218] = 0x0014 | (0x0037 << 16); rc[219] = (0x0101 << 16) | (0);
            } else { rc[216] = 0; rc[217] = 0; rc[218] = 0x00000014; rc[219] = 0x01010000; }
            memcpy((char *)c->mrc[i].cpu + (size_t)m * REGCMD_I4_N * 4, rc, REGCMD_I4_N * 4);
        }
        bsync(fd, &c->mrc[i], RKNPU_MEM_SYNC_TO_DEVICE);
        struct rknpu_task *mt = c->mtk[i].cpu; memset(mt, 0, (size_t)M * sizeof *mt);
        for (int q = 0; q < M; q++) {
            mt[q].enable_mask = 0xd; mt[q].int_mask = 0x300; mt[q].int_clear = 0x1ffff;
            mt[q].regcfg_amount = 116; mt[q].regcmd_addr = c->mrc[i].dma + (size_t)q * REGCMD_I4_N * 4;
        }
        bsync(fd, &c->mtk[i], RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);
        struct rknpu_submit sub; memset(&sub, 0, sizeof sub);
        sub.flags = 0x5; sub.task_number = M; sub.task_obj_addr = c->mtk[i].obj; sub.core_mask = 1u << i; sub.fence_fd = -1;
        sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, (uint32_t)M};
        /* Prime THIS core's buffers on its first use (mwarm[i]): a freshly-allocated NPU output buffer
         * returns stale on its first write, so the first task to land on a core does a throwaway warmup
         * rep then the real rep. Per-core (not an outer double-pass) because tasks are pulled round-robin
         * — a core that idles in pass 0 would otherwise stay unprimed and zero-fill the next task it grabs.
         * Same idiom as the run_multicore / chain workers. */
        int reps = c->mwarm[i] ? 1 : 2;
        for (int rep = 0; rep < reps; rep++) {
            int last = (rep == reps - 1);
            sub.timeout = 60000;
            if (rknpu_submit_ioctl(fd, &sub, w->domain)) { if (last) a->rc = -1; continue; }
            bsync(fd, &c->mcc[i], RKNPU_MEM_SYNC_FROM_DEVICE);
        }
        c->mwarm[i] = 1;
        int16_t *o = c->mcc[i].cpu; int32_t *C = t->C;        /* widen int16 NPU output -> int32 caller C */
        for (int row = 0; row < M; row++) {
            int16_t *orow = o + (size_t)row * N; int32_t *crow = C + (size_t)row * N;
            for (int col = 0; col < N; col++) crow[col] = orow[col];
        }
    }
    return NULL;
}
int ork_mm_run_stream_i4(ork_npu *c, int S, const ork_mm_task_i4 *tasks) {
    if (!c || S < 1 || !tasks) return -2;
    /* per-core scratch lives in the active domain; stream tasks share one domain (tasks[0].w) */
    if (tasks[0].w && (tasks[0].w->domain != c->dom_active || (tasks[0].w->domain!=0 && !c->dom_save))) dom_activate(c, tasks[0].w->domain);
    const int mrc_cap = 65536 / (REGCMD_I4_N * 4);
    size_t maxMK = 0, maxMN2 = 0;
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        if (!w || w->dtype != DT_I4 || tasks[i].M <= 0) return -2;
        if (w->Sn != 1 || w->Sk != 1) return -2;              /* single-slice weight only (no K/N split) */
        if (tasks[i].M > mrc_cap) return -2;                  /* M single-row regcmds must fit one mrc buffer */
        size_t mk = (size_t)tasks[i].M * w->K, mn = (size_t)tasks[i].M * w->N * 2;
        if (mk > maxMK) maxMK = mk; if (mn > maxMN2) maxMN2 = mn;
    }
    int fd = c->fd;
    int cold = 0;   /* warmup pass needed on a fresh stream-i4 mode OR freshly-allocated per-core buffer */
    if (c->last_dt != 5 /* DT_I4_STREAM */) { act(fd, RKNPU_ACT_RESET, 0); c->last_dt = 5; c->warmed = 0; for (int i = 0; i < ORK_MAXCORE; i++) c->mwarm[i] = 0; cold = 1; }
    int nc = budget(c, 2); if (nc > ORK_MAXCORE) nc = ORK_MAXCORE; if (nc > S) nc = S; if (nc < 1) nc = 1;
    if (mc_ensure(c, nc)) return -1;
    for (int i = 0; i < nc; i++) {
        if (c->maf[i].size < maxMK) { bdestroy(fd, &c->maf[i]); c->maf[i] = bcreate(fd, maxMK, 0x403, c->dom_active); if (!c->maf[i].cpu) return -1; cold = 1; }
        if (c->mccsz[i] < maxMN2) { bdestroy(fd, &c->mcc[i]); c->mcc[i] = bcreate(fd, maxMN2, 0x403, c->dom_active); c->mccsz[i] = maxMN2; if (!c->mcc[i].cpu) return -1; cold = 1; }
    }
    int rc = 0;
    if (cold) for (int i = 0; i < nc; i++) c->mwarm[i] = 0;   /* fresh mode/buffer => each core re-primes (per-core warmup in the worker) */
    npu_pool_ensure(c);
    struct streamw4 sw[ORK_MAXCORE];
    /* Single dispatch: priming is per-core inside the worker (mwarm[i]), so the result is correct
     * regardless of how the round-robin atomic counter assigns tasks to cores — no outer double-pass
     * (which left a core that idled in the first pass unprimed, zero-filling whatever task it then grabbed). */
    int ctr = 0;
    for (int i = 0; i < nc; i++) sw[i] = (struct streamw4){c, i, S, tasks, &ctr, 0};
    pthread_mutex_lock(&c->pmu);
    c->pjob = sw; c->pjob_nc = nc; c->pjob_fn = stream_worker_i4; c->pjob_stride = sizeof(struct streamw4);
    c->pdone = 0; c->pgen++; pthread_cond_broadcast(&c->pgo);
    pthread_mutex_unlock(&c->pmu);
    stream_worker_i4(&sw[0]);                             /* core 0 on the calling thread */
    pthread_mutex_lock(&c->pmu); while (c->pdone < nc - 1) pthread_cond_wait(&c->pdn, &c->pmu); pthread_mutex_unlock(&c->pmu);
    for (int i = 0; i < nc; i++) if (sw[i].rc) rc = -1;
    c->warmed = 1;
    return rc;
}

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
enum ork_async_kind { OAK_F16=0, OAK_I8, OAK_I4, OAK_CHAIN_I8, OAK_CHAIN_I4, OAK_STREAM_I8, OAK_STREAM_I4 };
struct ork_async { pthread_t th; int started; int rc; enum ork_async_kind kind;
    ork_npu *c; ork_w *w; int M;
    const void *A; void *C;                 /* single-matmul A/C (typed per kind) */
    int S; const void *tasks; };            /* chain/stream task array (typed per kind) */

/* Big-core SET mask (high-numbered cluster, matching pin_big_core's "high = big" assumption). Used to
 * launch the async worker bound to the WHOLE big cluster rather than one core: a single-core pin would
 * trap a freshly-created worker behind the caller (it inherits the caller's mask and can't run to
 * migrate); a set lets the scheduler place it on any FREE big core -> real CPU‖NPU overlap, never an
 * A55. (The run_multicore pool deliberately pins DISTINCT single cores instead — that's a simultaneous
 * barrier where distinct cores avoid contention; the async worker is a single overlapping thread.) */
static int ork_big_core_set(cpu_set_t *s){
#if defined(__linux__)
    static int off=-1; if(off<0) off=getenv("ORK_NO_AFFINITY")?1:0;
    if(off) return 0;
    long ncpu=sysconf(_SC_NPROCESSORS_ONLN); if(ncpu<2) return 0;
    CPU_ZERO(s); for(int k=(int)(ncpu/2);k<ncpu;k++) CPU_SET(k,s);  /* top half = big cluster on RK35xx */
    return 1;
#else
    (void)s; return 0;
#endif
}
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

static ork_async *ork_async_launch(struct ork_async tmpl){
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
 * (joins the thread, frees the handle). Numerics are identical to the synchronous run (reused verbatim). */
ork_async *ork_mm_run_async    (ork_npu *c, ork_w *w, int M, const ork_f16 *A, float   *C){
    if (!c || !w || M < 1) return NULL;
    return ork_async_launch((struct ork_async){ .kind=OAK_F16, .c=c, .w=w, .M=M, .A=A, .C=C }); }
ork_async *ork_mm_run_i8_async (ork_npu *c, ork_w *w, int M, const int8_t  *A, int32_t *C){
    if (!c || !w || w->dtype != DT_I8 || M < 1) return NULL;
    return ork_async_launch((struct ork_async){ .kind=OAK_I8, .c=c, .w=w, .M=M, .A=A, .C=C }); }
ork_async *ork_mm_run_i4_async (ork_npu *c, ork_w *w, int M, const int8_t  *A, int32_t *C){
    if (!c || !w || M < 1) return NULL;
    return ork_async_launch((struct ork_async){ .kind=OAK_I4, .c=c, .w=w, .M=M, .A=A, .C=C }); }
ork_async *ork_mm_run_chain_i8_async (ork_npu *c, int S, const ork_mm_task_i8 *tasks){
    if (!c || S < 1 || !tasks) return NULL;
    return ork_async_launch((struct ork_async){ .kind=OAK_CHAIN_I8, .c=c, .S=S, .tasks=tasks }); }
ork_async *ork_mm_run_chain_i4_async (ork_npu *c, int S, const ork_mm_task_i4 *tasks){
    if (!c || S < 1 || !tasks) return NULL;
    return ork_async_launch((struct ork_async){ .kind=OAK_CHAIN_I4, .c=c, .S=S, .tasks=tasks }); }
ork_async *ork_mm_run_stream_i8_async(ork_npu *c, int S, const ork_mm_task_i8 *tasks){
    if (!c || S < 1 || !tasks) return NULL;
    return ork_async_launch((struct ork_async){ .kind=OAK_STREAM_I8, .c=c, .S=S, .tasks=tasks }); }
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

/* Fast Walsh-Hadamard Transform (FWHT) - Exposed utility function for caller-driven quantization */
void ork_fwht_norm(float *v, int n){
    for(int len=1; len<n; len<<=1)
        for(int i=0;i<n;i+=len<<1)
            for(int j=i;j<i+len;j++){ float a=v[j], b=v[j+len]; v[j]=a+b; v[j+len]=a-b; }
    float s=1.0f/sqrtf((float)n);
    for(int i=0;i<n;i++) v[i]*=s;
}
