/* ork/dynamic.h — MoE / chained matmuls, the NONBLOCK doorbell API, submit queue, precompiled chains
 *
 * The asynchronous submission machinery: dynamic steered chains with per-op doorbell
 * progress and mid-flight halt, the chunk-pipeline queue over it, and the precompiled
 * program cache.
 *
 * Part of the public API. Do NOT include this directly — include <ork_npu.h>, which is the
 * umbrella and the only supported entry point; these parts are a readability split of it
 * (ork_npu.h was 1519 lines) and their boundaries may move. Types live in ork_npu.h above
 * the includes, so this header is not self-contained by design. */
#ifndef ORK_DYNAMIC_H
#define ORK_DYNAMIC_H
/* Mixture of Experts (MoE) / Chained matmuls API */
typedef struct {
    ork_w *w;
    int M;
    const int8_t *A;
    int32_t *C;
    int cstride;   /* optional output row-stride (elements) override for C; 0 = use w->N. Lets a colsplit write
                    * a sub-N-width result into a WIDER C at the full row stride (fp16 wide-N per-N-slice path). */
} ork_mm_task_i8;

typedef struct {
    ork_w *w;
    int M;
    const int8_t *A;
    int32_t *C;
} ork_mm_task_i4;

int          ork_i8_mm_run_chain(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks);
/* #54: run MANY int4 experts (each M>=1) COALESCED through one nonblock doorbell (rows chained across cores,
 * ~nc submits total). All experts must share one iommu domain. 0 ok / -4 refuse (too big) / <0 err. */
int          ork_i4_mm_run_experts(ork_npu *ctx, const ork_mm_task_i4 *ex, int ntask, int nc);

/* ---- Dynamic steered submission (NONBLOCK chain + per-op doorbell progress + mid-flight halt) ----
 * Submit an S-task int8 OR fp16 chain NONBLOCK, then watch/steer it from the host. v1: M=1/task (mc: M<=64),
 * single-slice conforming K (K%512==0, K<=4096). C is a resident ork_dma_alloc buffer (zero-copy direct
 * output) or host memory (copy-back); A is HOST (malloc) memory — begin_mc STAGES A via memcpy and a
 * zero-copy DMA-A source's CPU writes are not coherently readable by that staged read (partial-K sums).
 * Enables early-exit-to-free-the-NPU and runtime observability without a kernel round-trip per chain.
 * (See tools/ork_dyn_test.c: D=int8, E=fp16.) */
typedef struct ork_dyn_chain ork_dyn_chain;
size_t        ork_npu_sram_total(ork_npu *ctx);   /* NPU on-chip SRAM bytes (0 = none: stock kernel/DTB) */
size_t        ork_npu_sram_free (ork_npu *ctx);   /* free NPU SRAM bytes now (confirms ORK_WEIGHT_SRAM placement) */
uint64_t      ork_npu_dma_rw    (ork_npu *ctx);   /* cumulative NPU DMA rw bytes; delta across a submit = HW did work (0 => never dispatched) */
void          ork_npu_dump_state(ork_npu *ctx, const char *label);   /* snapshot NPU state (freq/volt/DMA counters) to stderr on anomaly, before a wedge destroys it */
int           ork_npu_soft_reset(ork_npu *ctx);   /* RKNPU_ACT_RESET + force re-warm; recovery step after a dump so a stuck job doesn't accumulate into a hard wedge */
int           ork_ctx_fd_reap(ork_npu *ctx);   /* task #47: close+reopen the DRM fd (drm_release reaps ALL stuck jobs+IOMMU cleanly — the only nonblock-safe reap) then re-import dma-buf weights in place + reset scratch; caller must quiesce submits first. 0 ok, <0 fail */
int           ork_npu_recover   (ork_npu *ctx, const char *label);   /* self-heal: dump + soft-reset + dummy-op probe; 1 = recovered (continue), 0 = still broken (throw fault) */
int           ork_npu_force_fault(ork_npu *ctx);   /* DIAGNOSTIC: deliberately force a reliable NPU fault (bogus weight addr -> DMA fault); 0 = faulted as intended */
ork_dyn_chain *ork_dyn_begin(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks);   /* NONBLOCK-submit (single-core chain); NULL on bad args */
ork_dyn_chain *ork_dyn_begin_mc(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks, int nc); /* NONBLOCK across nc cores (nc<=0=all); halt/append N/A */
int          ork_dyn_progress(ork_dyn_chain *h);                                  /* highest completed op idx, -1 none */
void         ork_dyn_dump(ork_dyn_chain *h, const char *label);                   /* chain-aware anomaly dump: names the STUCK descriptor (progress+1) + its regcmd/addr/doorbell + hw_elapse */
int          ork_dyn_halt(ork_dyn_chain *h, int at);                              /* halt after op `at` (free NPU early) */
int          ork_dyn_end(ork_dyn_chain *h);                                       /* drain + writeback + free; ret highest done */
int          ork_dyn_max_steps(void);                                             /* per-chain step cap (split longer work across chains) */
int          ork_dyn_steps(ork_dyn_chain *h);                                     /* total steps submitted in this chain */
int          ork_dyn_remaining(ork_dyn_chain *h);                                 /* steps not yet completed (budget left before the chain ends) */
int          ork_dyn_append(ork_dyn_chain *h, const ork_mm_task_i8 *task);        /* extend a running chain in-flight (wrap); 1=too late, 0=ok, <0=err */
int          ork_dyn_spin_probe(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks, int spin_us, int *spin_alive); /* circular-spin keep-alive + redirect probe */
/* EXPERIMENTAL int4 NONBLOCK-doorbell probe: mirrors ork_i4_mm_run_chain (M=1 int4 PC-chain, host A) but
 * flips the submit to NONBLOCK (0x2) + polls an int16 output-sentinel to completion, then de-tiles int16->
 * int32 into each task->C. Answers the load-bearing question of whether the int4 (int16-output) datapath
 * survives the doorbell's non-blocking sentinel poll. Returns 0/ok, <0 err. (Not a production path.) */
int          ork_i4_dyn_probe(ork_npu *ctx, int S, const ork_mm_task_i4 *tasks);
/* Submit QUEUE: chunk-pipeline over the dynamic API — accumulate tasks, run a chunk NONBLOCK while the
 * caller does other work (CPU‖NPU decode split), auto-split work > chunk_max into successive clean chunks. */
typedef struct ork_dyn_queue ork_dyn_queue;
ork_dyn_queue *ork_dyn_queue_create(ork_npu *ctx, int chunk_max, int ncore);      /* chunk_max<=0=>max_steps; ncore<=1 single-core, >1 multi-core NONBLOCK */
void         ork_dyn_queue_set_linger(ork_dyn_queue *q, int us);                  /* coalesce window; default = one submit floor */
int          ork_dyn_queue_linger_us(ork_dyn_queue *q);
int          ork_dyn_queue_push(ork_dyn_queue *q, const ork_mm_task_i8 *task);    /* enqueue a matmul */
int          ork_dyn_queue_flush(ork_dyn_queue *q);                               /* submit next chunk NONBLOCK (NPU starts) */
int          ork_dyn_queue_pending(ork_dyn_queue *q);                             /* tasks not yet submitted */
int          ork_dyn_queue_idle(ork_dyn_queue *q);                                /* on idle+linger-elapsed, halt a flying reserved/spin chain early (null-terminate); 1 if halted */
int          ork_dyn_queue_drain(ork_dyn_queue *q);                               /* finish all chunks + writeback; ret total ops */
void         ork_dyn_queue_destroy(ork_dyn_queue *q);
/* Precompiled-program cache (regime A: fixed chain, pinned buffers). Compile the chain ONCE, then re-run it
 * every token with only the activation contents refreshed — no per-token synth/validate. */
typedef struct ork_pc_chain ork_pc_chain;
ork_pc_chain *ork_pc_compile(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks);   /* build the program pool once; NULL on bad args */
int          ork_pc_run(ork_pc_chain *pc);                                        /* refresh A + NONBLOCK submit + drain; ret highest op */
void         ork_pc_free(ork_pc_chain *pc);

#endif /* ORK_DYNAMIC_H */
