/* npu/core.h — the SUBSTRATE interface: everything in the driver whose contract has no dtype in it.
 *
 * Buffers, IOMMU domains, the DRM submit path, device lifecycle, the worker pool, the mode-transition
 * layer, profiling. Implemented in src/npu/core/*.c; used by npu.c and by every precision module.
 *
 * WHY THIS EXISTS, declared up front instead of grown: measured on the pre-split tree, ~90% of each
 * precision module's inbound boundary is this substrate (i4: 35 of 40 symbols, f16: 36 of 38, i16: 17 of
 * 19, i8: 62 of 69). Genuine cross-PRECISION coupling is tiny by comparison — i4 needs 5 int8 symbols,
 * f16 needs 2. Discovering the substrate one lift at a time made internal.h accrete (165 -> 274 lines in
 * two lifts) and made every lift pay to rediscover the same declarations. Declaring the whole interface
 * ONCE turns each remaining precision lift into a pure move against a stable header.
 *
 * internal.h holds the TYPES (what things are) and the hot static inlines; this holds the API (what you
 * can call). Include as "npu/core.h" from any depth (-Isrc). */
#ifndef ORK_NPU_CORE_H
#define ORK_NPU_CORE_H
#include "npu/internal.h"

/* ---- core/prof.c — profiling + timing accessors ---- */
void ork_load_prof_dump(void);
void ork_ssm_prof_dump(void); /* fwd: ORK_SSM_PROF per-section accounting */ void ork_ssm_helper_stop(ork_npu *c); /* fwd: stop the little-core marshalling helper */ static void ork_npu_xprof_dump(void); void ork_npu_free(ork_npu *c);
void ork_npu_mc_reset(void);
void ork_npu_mc_timing(int core,double*copy,double*sub,double*acc,long*n);
double ork_npu_mc_synth(int core);
void ork_npu_run_timing(double*setup,double*submit,double*copy,long*n);
void ork_npu_floor_timing(double*ioctl_us,double*hw_us,long long*hw_raw_last,long*n);
void ork_npu_floor_reset(void);
void ork_npu_xprof_dump(void);
int ork_npu_doorbell_prof(ork_npu *c,int M,int K,int N,int iters,double *block_us,double *nb_us,int *ok_block,int *ok_nb);
int ork_npu_overlap_prof(ork_npu *c,int M,int K,int N,int cpu_reps,int iters, double *npu_solo,double *cpu_solo,double *overlap_wall,int *ok);

/* ---- core/buf.c — DMA buffer lifecycle, dma-heap import, the weight arena, zero-copy ---- */
size_t orki_pgup(size_t s);
struct buf orki_bcreate(int fd,size_t size,uint32_t flags,int domain);
int orki_ppu_scratch3(ork_npu *c,size_t sz);
int orki_dmaheap_open(void);
void orki_dmabuf_sync(int heap_fd,uint64_t flags);
struct buf orki_bimport_f(int fd,size_t size,int domain,uint32_t memflags);
struct buf orki_bimport(int fd,size_t size,int domain);
struct buf orki_bscratch(ork_npu *c,size_t size,int flags,int dom);
struct buf orki_bimport_fd(int fd,int dbuf,size_t size,int domain);
struct buf *orki_warena_reserve(ork_npu *c,size_t need,size_t *base);
void *ork_dma_alloc(ork_npu *c, size_t size);
void ork_dma_free(ork_npu *c, void *ptr);
void *ork_dma_import(ork_npu *c, size_t size);
void ork_dma_import_free(ork_npu *c, void *ptr);
void *ork_dma_import_fd(ork_npu *c, int dmabuf_fd, size_t size);
struct buf *orki_dma_find(ork_npu *c, const void *p);
void ork_dma_import_sync(ork_npu *c, void *ptr, size_t size);
void ork_dma_bsync_to_device(ork_npu *c, void *ptr, size_t size);
void *ork_dma_alloc_flags(ork_npu *c, size_t size, unsigned flags);
void *ork_dma_alloc_sram(ork_npu *c, size_t size);
void ork_dma_clean_to_device(ork_npu *c, void *ptr, size_t size);
struct buf orki_bstage_alloc(size_t size);
int orki_bstage_map(int fd, struct buf*b);
void orki_bstage_unmap(int fd, struct buf*b);
void orki_bstage_free(struct buf*b);

/* ---- core/domain.c — IOMMU domains + the IOVA wedge guard ---- */
int ork_dom_default(void);
size_t ork_iova_ceiling(void);
int ork_iova_reserve(int dom,size_t need);
void ork_iova_release(int dom,size_t bytes);
int orki_dom_reserve(ork_npu *c, int need);
void ork_dom_prime(ork_npu *c, int dom);
void ork_dom_reanchor(ork_npu *c, int dom);
void ork_dom_flush_if_dirty(ork_npu *c);
void orki_dom_activate(ork_npu *c,int dom);

/* ---- core/submit.c — the raw DRM submit path ---- */
void orki_act(int fd,uint32_t f,uint32_t v);
void orki_dump_submit(struct rknpu_submit *sub);
void orki_trace_submit(struct rknpu_submit *sub);
int orki_rknpu_submit_ioctl(int fd, struct rknpu_submit *sub, int domain);
uint32_t ork_ppflags(void); /* fwd (defined below) */ /* #39 WEIGHT-RESIDENT M-FOLD CHAIN (task_number=P). Each of P tasks is a width-`w` mfold tile built by the * validated synth_i8_mfold (bit-exact standalone at w=36); the K*N weight is loaded ONCE and shared across all * P tasks (HW PC-chain), amortizing the weight-DMA over M=P*w rows. Chain descriptor written at words 216-219 * EXACTLY like the PROVEN run_chain_i8/mcworker_pref_chain; terminal task clears them. subcore_task[0..2] all * populated (single-core), matching run_chain_i8. Apacked = P tiles of w-row C2-16 A (w*K bytes each); Bpacked = * shared ork_woff weight; Craw = P tiles of raw C2-4 int32 (w*N int32 each). Board only; returns 0/ok, us=avg. */ int ork_npu_mfold_chain(ork_npu *c, int P, int w, int K, int N, const int8_t *Apacked, const int8_t *Bpacked, int32_t *Craw, int iters, double *us);
int orki_submit1(ork_npu *c);
int orki_submit1_db(ork_npu *c, size_t nout);

/* ---- core/device.c — context lifecycle, reset/recovery, SoC introspection, setters ---- */
void ork_npu_set_ndomains(ork_npu *c, int n);
void ork_npu_reap_stuck(ork_npu *c, int nc); /* fwd: per-core timeout_clean reap (defined below) */ int orki_i4_submit_tmo_ms(void); /* fwd: bounded int4 doorbell submit timeout (defined near the int4 workers) */ void ork_dom_flush_if_dirty(ork_npu *c);
void orki_warn_if_governor_parked(void);
size_t ork_npu_sram_total(ork_npu *c);
void ork_kmsg(const char *fmt, ...);
void ork_npu_dump_state(ork_npu *c, const char *label);
int orki_reimport_inplace(int fd, struct buf *b);
int ork_ctx_fd_reap(ork_npu *c);
int ork_dummy_probe(ork_npu *c);
int ork_npu_recover(ork_npu *c, const char *label);
int ork_npu_force_fault(ork_npu *c);
ork_npu *ork_npu_init_orkd(void);
ork_npu *ork_npu_init(void);
void ork_npu_free(ork_npu *c);
const char *ork_npu_soc(const ork_npu *c);
int ork_npu_cores(const ork_npu *c);
int ork_npu_validated(const ork_npu *c);
void ork_npu_set_core_budget(ork_npu *c,int n);
void ork_npu_set_priority(ork_npu *c,unsigned prio);
int ork_npu_domain_alloc(ork_npu *c);
int ork_npu_domain_free(ork_npu *c,int domain);
int ork_npu_pack_domain(const ork_npu *c);
int ork_npu_active_domain(const ork_npu *c);
void ork_npu_activate_domain(ork_npu *c, int domain);
int ork_npu_uses_orkd(const ork_npu *c);
int ork_npu_busy(ork_npu *ctx);
void ork_npu_mode_invalidate(ork_npu *c);
void ork_npu_mode_reset(ork_npu *c);
const char *ork_npu_version(void);
uint32_t ork_pack_format_version(void);

/* ---- core/sched.c — worker pool, core pinning, per-core scratch ---- */
int orki_mc_ensure(ork_npu *c,int nc); /* fwd: #54 pre-alloc domain-0 run scratch at init (while empty) */ ork_npu *ork_npu_init(void);
int ork_all_cores_mask(cpu_set_t *s);
void ork_unpin_current_thread(void);
void *ork_pool_worker(void *a);
void ork_pool_init(void);
void ork_parallel_for(int n, void (*fn)(int,int,void*), void *ctx);
void orki_pin_big_core(int id);
void orki_pin_little_core(int id);
void *orki_npu_pool_worker(void *vp);
void orki_npu_pool_ensure(ork_npu *c);
int ork_big_core_set(cpu_set_t *s);

/* ---- core/mode.c — the mode-transition layer (XSPEC / ork_npu_enter) ---- */
int ork_npu_enter(ork_npu *c, int to, int profile, int chain);

#endif /* ORK_NPU_CORE_H */
