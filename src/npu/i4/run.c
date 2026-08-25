/* npu/i4/run.c — int4 execution entrypoints.
 *
 * Part of the i4 datapath; shared declarations in npu/i4/i4.h. Split out of npu/i4.c for the
 * same reason i8 is a folder: one datapath, sized for reading. */
#define _GNU_SOURCE   /* CPU_SET/pthread_setaffinity_np, as npu.c does */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include <math.h>
#include "ork_regs.h"
#include "regcmd_i8.h"
#include "orkd_proto.h"
#include "npu/internal.h"
#include "npu/core.h"
#include "regcmd_i4.h"
#include "npu/i4/i4.h"

int orki_i4_submit_tmo_ms(void){
    static int t=-1;
    if(t<0){ const char*e=getenv("ORK_I4_SUBMIT_TMO_MS");
        if(e) t=atoi(e);
        else { const char*p=getenv("ORK_I4_POLL_MS"); double pm=p?atof(p):2000.0; t=(int)(pm*0.75); }
        if(t<10) t=10; }
    return t;
}

/* Kernel-facing submit timeout. SEPARATE from orki_i4_submit_tmo_ms() because that value is ALSO used as a
 * host-side sleep in ork_dom_flush_if_dirty (`orki_i4_submit_tmo_ms() + 200` ms), so scaling it to probe the
 * kernel's behaviour turns every dirty boundary into a multi-second stall and invalidates the measurement.
 * ORK_I4_KTMO_MUL scales ONLY the number handed to the driver.
 * WHY IT EXISTS: rknpu_job_timeout_clean compares `ktime_us_delta(now, job->timestamp) >= args->timeout`,
 * i.e. MICROseconds against a value every other driver site treats as milliseconds (msecs_to_jiffies,
 * `args->timeout * 1000`). If that units bug is live, our 1500 "ms" is really a 1.5 ms reap threshold and
 * any job still running when the next submit lands on that core gets soft-reset mid-flight. MUL=1000
 * restores the intended deadline without touching the kernel. */
int orki_i4_ktmo_ms(void){
    static int m=-1; if(m<0){ const char*e=getenv("ORK_I4_KTMO_MUL"); m=e?atoi(e):1; if(m<1) m=1; }
    return orki_i4_submit_tmo_ms()*m;
}

/* ork_i4_batch() — STRATEGY A: int4 stride-2 IN-TASK batch (Exp-2026-06-19). One submit computes a whole
 * M-tile with resident weights (mc_phys=2*H, 0x405c=0, stride-2 output → physical row 2m carries logical
 * row m; NEON int16→int32 de-tile physrow=4j+4H*b), instead of the per-row PC-chain that re-streams the
 * weight every row (the W4A4 submit-bound the int8 0x1040 M-scheduler avoids). Default ON (bit-exact
 * validated ./i4; per-row fallback where the batch doesn't fit). Implemented in synth_i4 mc>1 + orki_i4_run_mc_db (per-row doorbell)
 * + stream_worker_i4; NVDLA D_BATCH_NUMBER/D_*_STRIDE analogy.
 *   PRESERVED as a distinct, named strategy — the multi-task-submit batch (many 1-row tasks per submit, the
 *   vendor's int4 approach; task_number=rows) is a SEPARATE path and must NOT overwrite/conflate with this.
 *   Env var kept as ORK_I4_MSCHED for back-compat (0=off, 1=on). */
int ork_i4_batch(void){ static int v=-1; if(v<0){const char*e=getenv("ORK_I4_MSCHED"); v=e?(atoi(e)?1:0):1;} return v; }

static int ork_i4_nsub(void){ static int v=-1; if(v<0){const char*e=getenv("ORK_I4_NSUB"); v=(e&&atoi(e))?1:0;} return v; }  /* default OFF: under pinned-DDR benchmarking N-subslice is ~neutral (submit overhead cancels the weight-DMA saving); the "1.1-1.2x" seen earlier was an unpinned-governor artifact. Kept opt-in. */

void orki_i4_synth(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC){
    memcpy(rc,REGCMD_I4,REGCMD_I4_N*4);
    orki_setrn(rc,REGCMD_I4_N,RK_CNA_DATA_SIZE1,((K-1)<<16)|K);       /* K range (element count) */
    orki_setrn(rc,REGCMD_I4_N,RK_CNA_WEIGHT_SIZE0,(K*N)/2);             /* weight bytes: int4 = 0.5 B/elem */
    orki_setrn(rc,REGCMD_I4_N,RK_CNA_WEIGHT_SIZE1,K/2);                 /* weight row bytes */
    orki_setrn(rc,REGCMD_I4_N,RK_CNA_CBUF_CON1,(K+127)/128);        /* K-passes: ceil(K/128) (captured scaling) */
    orki_setrn(rc,REGCMD_I4_N,RK_CNA_FC_DATA_SIZE1,K);
    orki_setrn(rc,REGCMD_I4_N,RK_CNA_WEIGHT_SIZE2,0x1010000|N);orki_setrn(rc,REGCMD_I4_N,RK_PDP_OUT_N,N-1);
    /* N-output-stride regs, parameterized for wide-N single-submit (verified vs N=64 & N=128
     * captures: 0x403c=(N-1)dup, 0x4058=N-1, 0x3018=N-1 above). 0x40c0/0x4050 are CONSTANT across N
     * (0x80/0x7fe — left at REGCMD_I4). */
    orki_setrn(rc,REGCMD_I4_N,RK_DPU_DST_N_DIMS,((N-1)<<16)|(N-1));
    orki_setrn(rc,REGCMD_I4_N,RK_DPU_DST_N2,N-1);
    /* Multi-M scheduler (mc>1) — native batch mode (NVDLA D_BATCH_NUMBER analog): one submit computes H
     * rows with the weight streamed ONCE, output at stride-2 (logical row m -> physical row 2m; int16 result
     * in an int32-stepped DMA). Reached only via ORK_I4_MSCHED (gated OFF; production callers pass mc=1 ->
     * the proven per-row PC-chain in i4_mcworker). Exp-2026-07-07 (tools/i4_multim_probe.c) established:
     *   - 0x1040 (K-reduction schedule) is DECISIVE, not the earlier-claimed inert set: the int8 FORMULA
     *     for it corrupts int4 (188@K64/72@K2048); leaving it at the captured base 177 works but only for
     *     the capture's K. -> ORK_I4_1040/ORK_I4_1010 overrides + ORK_I4_MREGS bitmask expose it for RE.
     *   - With mregs=0x1f (all int8-ported regs EXCEPT 0x1040) the batch mode is BIT-EXACT at K=64 (the
     *     capture K): all H rows land at stride-2, single 64-wide N-block. Verified M=2,4 @ K=64.
     *   - REMAINING WALL: row count is capture-K-tuned (K=32->1 row, K=64->4, K=128->2, K=256->1, K>=512->1)
     *     and multi-N-block output is a 2D surface (offsets (b*H+m)*2N at N=128), so at production K (2048)
     *     only row 0 computes. Cracking it needs a systematic 0x1040 x K x mc x CNA/CBUF sweep (the batch
     *     activation-cube budget reg) — see the wiki Exp-2026-07-07 log. Until then W4A4 stays 1-submit/row. */
    if(mc>1){
        int mc_phys = 2 * mc;
        /* ORK_I4_MREGS bitmask — which regs unlock native multi-M. Default 0x1f = the full int8-ported set
         * MINUS 0x1040. Fuzzing (Exp-2026-07-07, tools/i4_multim_probe.c) proved 0x1040 (the int8 K-reduction
         * schedule) is the POISON PILL: every config that sets it corrupts row 0 and rows>0; every config
         * without it computes bit-exact stride-2 (logical row m -> physical row 2m). int4's K-reduction uses
         * the captured base 0x1040 (K-independent, unlike int8) — which the M=1 path already never overrides.
         *   0x01 M-count 0x1020/0x1084/0x102c   0x02 0x4034(PPU rows)   0x04 0x3014(DPU)
         *   0x08 0x4038(out width/4)            0x10 0x1010(CNA hint)   0x20 0x1040(K-schedule=POISON) */
        static int mregs=-1; if(mregs<0){const char*e=getenv("ORK_I4_MREGS"); mregs=e?(int)strtoul(e,0,0):0x5f;}
        orki_setrn(rc,REGCMD_I4_N,RK_DPU_WDMA_SIZE_1,0);                                   /* the trigger (always) */
        /* 0x107c = K/16 : the batch activation-cube-size reg (Exp-2026-07-07 fuzz). Captured as 4 (tuned to
         * the M=4/K=64 capture); setting it to K/16 restores the native 4-ROW batch at ANY K — bit-exact at
         * K=512/1024/2048 (0x20/0x40/0x80). This lifts multi-M from 1 row to 4 rows/submit at production K
         * (4x fewer weight streams). Narrow: only K/16 gives 4 (neighbors give 2); 4 is the cap for this reg. */
        if(mregs&0x40) orki_setrn(rc,REGCMD_I4_N,RK_CNA_DMA_CON1,(uint32_t)(K/16));
        if(mregs&0x01){ orki_setrn(rc,REGCMD_I4_N,RK_CNA_DATA_SIZE0,0x10000|mc_phys);orki_setrn(rc,REGCMD_I4_N,RK_CNA_DATA_SIZE0_MIR,0x10000|mc_phys);orki_setrn(rc,REGCMD_I4_N,RK_CNA_DATA_SIZE3,mc_phys); }
        if(mregs&0x02) orki_setrn(rc,REGCMD_I4_N,RK_DPU_DATA_CUBE_HEIGHT,mc_phys-1);
        if(mregs&0x04) orki_setrn(rc,REGCMD_I4_N,RK_PDP_OUT_M,(mc_phys-1)<<16);
        if(mregs&0x08) orki_setrn(rc,REGCMD_I4_N,RK_DPU_DATA_CUBE_NOTCH,(((N/4)-1)<<16)|((N/4)-1));
        if(mregs&0x10) orki_setrn(rc,REGCMD_I4_N,RK_CNA_CONV_CON2,16*(mc_phys+1));
        if(mregs&0x20){ double scale=(double)K/256.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale),mg=mc_phys/64; if(mg<1)mg=1;
            int v=base-slope*(mg-1); if(v<0x1b)v=0x1b; orki_setrn(rc,REGCMD_I4_N,RK_CNA_CBUF_CON0,v); }
        /* ORK_I4_1040: direct override of the K-reduction schedule reg (RE: find the int4 multi-row value —
         * the int8 formula corrupts, omitting it leaves only row0 for K>64). Applied last, wins over mregs&0x20. */
        /* ORK_I4_DBNK=n: set the CBUF split to n DATA banks / (12-n) WEIGHT banks, i.e.
         * 0x1040 = ((12-n)<<4)|n. Measured (tools/re/i4_bank_sweep.c): the activation ceiling is
         * H_max = DBNK*16384/K, so raising DBNK raises rows-per-weight-stream proportionally. Paired
         * with the matching H in i4/chain.c so ONE knob keeps the split and the tiling consistent —
         * ORK_I4_H alone cannot, since the right H is per-K and a model has many K. */
        { const char*d=getenv("ORK_I4_DBNK"); if(d){ int n=atoi(d); if(n>=1&&n<=11)
              orki_setrn(rc,REGCMD_I4_N,RK_CNA_CBUF_CON0,(uint32_t)(((12-n)<<4)|n)); } }
        { const char*e=getenv("ORK_I4_1040"); if(e) orki_setrn(rc,REGCMD_I4_N,RK_CNA_CBUF_CON0,(uint32_t)strtoul(e,0,0)); }
        /* ORK_I4_1010: override CNA row/activation-cube reg (RE: multi-M computes rows_computed*K=256 elems —
         * a fixed activation-cube budget; find the reg that enlarges it so K=2048 gets >1 row). */
        { const char*e=getenv("ORK_I4_1010"); if(e) orki_setrn(rc,REGCMD_I4_N,RK_CNA_CONV_CON2,(uint32_t)strtoul(e,0,0)); }
    }
    orki_setrn(rc,REGCMD_I4_N,RK_CNA_FEATURE_DATA_ADDR,aA);orki_setrn(rc,REGCMD_I4_N,RK_CNA_WEIGHT_DATA_ADDR,aB);orki_setrn(rc,REGCMD_I4_N,RK_DPU_DST_BASE_ADDR,aC);
    for(int i=0;i<orki_i4_fovr_n;i++) orki_setr(rc,REGCMD_I4_N,orki_i4_fovr[i].blk,orki_i4_fovr[i].reg,orki_i4_fovr[i].val);  /* RE fuzzer overrides (win over all) */
}

void orki_i4_tile_direct_to_i8(ork_npu *c, ork_w *w, int K, int N, int kind, int8_t *i8scratch) {
    if (kind == ORK_QK_CODEBOOK_NF4) {
        int8_t lut[16]; for (int i = 0; i < 16; i++) lut[i] = (int8_t)lrintf(ORKI_NF4_LEVELS[i]*127.0f);
        for (int n = 0; n < N; n++) orki_nf4_inflate_chan_to_i8(w->Bi4 + (size_t)n*(K/2), K, lut, i8scratch + (size_t)n*K);
    } else {
        for (int n = 0; n < N; n++) orki_i4_expand_chan_to_i8(w->Bi4 + (size_t)n*(K/2), K, i8scratch + (size_t)n*K);
    }
    orki_i8_tile_to_tiles(c, w, K, N, i8scratch);
}

void ork_i4a8_slice_inflate_kind(const ork_w *w, float *qf32, int kind) {
    if (!w || !w->Bi4) return;
    int K = w->K, N = w->N;
    if (kind == ORK_QK_CODEBOOK_NF4) {
        int8_t lut[16]; for (int i = 0; i < 16; i++) lut[i] = (int8_t)lrintf(ORKI_NF4_LEVELS[i]*127.0f);
        for (int nn = 0; nn < N; nn++) {
            const uint8_t *nibp = w->Bi4 + (size_t)nn*(K/2); float *qf = qf32 + (size_t)nn*K;
            for (int k = 0; k < K; k++) { uint8_t idx = (k&1) ? (nibp[k>>1]>>4) : (nibp[k>>1]&0xf); qf[k] = (float)lut[idx]; }
        }
    } else {
        for (int nn = 0; nn < N; nn++) orki_i4_expand_chan_f32(w->Bi4 + (size_t)nn*(K/2), K, qf32 + (size_t)nn*K);
    }
}

void ork_i4a8_slice_inflate(const ork_w *w, float *qf32) { ork_i4a8_slice_inflate_kind(w, qf32, w ? w->quant_kind : 0); }

/* DIRECT path microbench: inflate w's nibbles STRAIGHT to int8-tiled (no f32, no re-quant) into the
 * resident DMA tiles. i8scratch is caller-provided (size N*K); kind forces UNIFORM/NF4. Bit-identical
 * to ork_i4a8_slice_inflate_kind + ork_i8_slice_tile, but in one pass with no float round-trip. */
void ork_i4a8_slice_direct_kind(ork_npu *c, ork_w *w, int8_t *i8scratch, int kind) {
    if (!w || !w->Bi4) return;
    orki_i4_tile_direct_to_i8(c, w, w->K, w->N, kind, i8scratch);
}

int ork_i4_mm_run(ork_npu *c,ork_w *w,int M,const int8_t *A,int32_t *C){
    if(c && c->fd<0 && w && w->cpu_codes){ orki_cpu_gemm_i32(M,w->K,w->N,A,w->cpu_codes,C); return 0; }   /* OFFLINE: exact CPU MAC */
    if(w && w->is_orkd){   /* Path B: int4 run on the daemon — ring transport if attached, else socket */
        orkd_set_op_domain(c->daemon, (uint32_t)w->domain);   /* v2: carry this weight's domain with the op */
        if(c && c->daemon && orkd_has_ring(c->daemon)){ int r=orkd_ring_run(c->daemon,w->orkd_id,M,w->K,w->N,ORKD_DT_I4,A,C); if(r!=-2) return r; }
        return orkd_run_i4(c->daemon, w->orkd_id, M, w->K, w->N, A, C); }
    if(!w||w->dtype!=DT_I4) return -1;
    /* Multi-domain: the submit's regcmd/task/scratch AND the weight must live in the SAME iommu domain. Activate
     * this weight's domain before the int4 submit — mirror the int8 run paths. Without it a resident int4 weight
     * in domain N submits against the stale dom_active (e.g. 0) -> RKNPU_SUBMIT EINVAL(22) -> self-heal reset ->
     * retry (correctness held via the reset, but every cross-domain expert submit thrashed -> very slow). */
    if(w->domain!=c->dom_active || (w->domain!=0 && !c->dom_save)) orki_dom_activate(c,w->domain);
    if(orki_check_overlap("ork_i4_mm_run", (uintptr_t)A, (uintptr_t)A + (size_t)M * w->K, (uintptr_t)C, (uintptr_t)C + (size_t)M * w->N * 4)) return -1;
    int NB=w->N/64;                            /* total 64-wide N-blocks (column-split granularity) */
    int nc=orki_budget(c, M); if(nc>NB)nc=NB; if(nc<1)nc=1;   /* ≥1 N-block/core; nc==1 = serial */
    /* DEFAULT int4 M>1 prefill: BCHAIN batch-chain on the NONBLOCK doorbell (run_i4_bchain_db) — H-row native
     * batches (synth_i4 mc>1) + bank-width Wb=131072/K N-tiling + weight-loaded-once chaining, self-healing on
     * the doorbell spine. Bit-exact, ~18-25x over the per-row doorbell, and serves the large-M shapes the
     * per-row path refuses (#52). Falls through (-4) to the per-row doorbell for decode (M=1)/non-qualifying. */
    /* #33 TEST/A-B hook: force a tile-bearing int4 shape onto the slice rescue (bit-exact validation). */
    if(w->sliced && getenv("ORK_FORCE_SLICE_RESCUE")){ int rs=ork_mm_run_sliced(c,w->sliced,M,A,C,nc); if(rs>=0) return rs; }
    if(getenv("ORK_I4_DIAG")) fprintf(stderr,"[i4diag] run_i4 K=%d N=%d M=%d Sk=%d Sn=%d dom=%d imported=%d -> %s\n",
        w->K,w->N,M,w->Sk,w->Sn,w->domain,(w->own_bufs&&w->n_own_bufs>0)||w->own_buf_valid,
        (M>=2 && w->Sk==1 && w->Sn==1 && (w->N%64)==0)?"BCHAIN":
        (M>=2 && w->Sk==1 && w->Sn>1 && (w->N%64)==0)?"BCHAIN-NSLICE":
        (M>=2 && w->Sn==1 && w->Sk>1 && (w->N%64)==0)?"BCHAIN-KSLICE":(w->sliced&&M>=2)?"SLICE":"mc_i4(per-row)");
    /* #54: DEFAULT int4 M>1 prefill = BCHAIN (M-batched, the perf path) — now PORTED to ride the SHARED ork_dyn_end
     * drain (poll + orki_mc_recover_resubmit + reap-at-boundary), so it is multi-domain-safe like int8 colsplit AND
     * keeps its M-batching. bch_db_worker builds+submits only; ork_dyn_end owns the drain (i4batch hooks). Falls
     * through (-4) to the per-row doorbell (orki_i4_run_mc_db) for decode (M=1) / non-qualifying shapes. */
    if(M>=2 && w->Sk==1 && w->Sn==1 && (w->N%64)==0){ int r=orki_i4_run_bchain_db(c,w,M,A,C,nc); if(r!=-4) return r; }
    /* Sn>1 PREFILL: run BCHAIN PER N-SLICE over the tiles the weight ALREADY holds.
     *
     * BCHAIN above requires Sn==1, so every wide-N shape fell to the per-row doorbell below. On
     * Qwen3.6-27B that is 192 of 400 matmuls -- and they are the big ones: 128x K=5120 N=17408 (Sn=3,
     * ffn up/gate) and 64x K=5120 N=10240 (Sn=2, attn qkv). MEASURED consequence: 667 NPU programs per
     * matmul on average and 205 us per program, i.e. ~6% of the hardware.
     *
     * The existing #33 slice rescue does solve this, but only for weights built by ork_i4_mm_pack -- it
     * needs the raw nibbles, and a pack-LOADED weight keeps none, so w->sliced is NULL for every weight
     * that came from an .orkpack. Building sub-weights at load instead would repack each tile and roughly
     * DOUBLE resident IOVA (11.3 -> ~22 GiB on this model), which does not fit.
     *
     * With Sk==1 the N-slices are independent, and Bb[ns*Sk+ks] means slice ns IS Bb[ns] -- so a shallow
     * view (Sn=1, N=Nc, Bb/Bf/bscale rebased) is a legal BCHAIN weight with no new packing and no new
     * IOVA. Each slice writes a contiguous [M][Nc] block that is scattered into C at the full row stride.
     * Bit-exactness is unchanged: same tiles, same kernel, just batched instead of per-row. */
    if(M>=2 && w->Sk==1 && w->Sn>1 && (w->N%64)==0 && !getenv("ORK_I4_NO_NSLICE_BCHAIN")){
        const int NMAXd = c->soc->nmax;
        int ok = 1;
        int32_t *Ct = (int32_t*)malloc((size_t)M*(size_t)NMAXd*sizeof(int32_t));
        if(Ct){
            for(int ns=0; ns<w->Sn && ok; ns++){
                const int n0 = ns*NMAXd;
                int Nc = w->N - n0; if(Nc > NMAXd) Nc = NMAXd;
                if(Nc <= 0 || (Nc%64)){ ok = 0; break; }        /* BCHAIN needs 64-wide N-blocks */
                struct ork_w v = *w;
                v.N = Nc; v.Sn = 1; v.Sk = 1;
                v.Bb = &w->Bb[ns];
                v.Bf = w->Bf ? &w->Bf[ns] : NULL;
                v.bscale = w->bscale ? w->bscale + n0 : NULL;
                v.sliced = NULL;                                 /* a view never recurses into the rescue */
                v.own_buf_valid = 0; v.own_bufs = NULL; v.n_own_bufs = 0;   /* and owns nothing to free */
                v.Bbc_valid = 0; v.Bbc_ns_valid = 0; v.Bbc_ns = NULL;
                int nbv = Nc/64, ncv = nc; if(ncv > nbv) ncv = nbv; if(ncv < 1) ncv = 1;
                if(orki_i4_run_bchain_db(c, &v, M, A, Ct, ncv) != 0){ ok = 0; break; }
                for(int m=0; m<M; m++)
                    memcpy(C + (size_t)m*w->N + n0, Ct + (size_t)m*Nc, (size_t)Nc*sizeof(int32_t));
            }
            free(Ct);
            if(ok) return 0;
            /* a slice refused: C may be partially written, but every path below rewrites all of it */
        }
    }
    /* Sk>1 PREFILL (Sn==1): run BCHAIN PER K-SLICE and accumulate the int32 partials.
     *
     * The N-slice path above leaves one family behind: Sk>1 shapes, which on Qwen3.6-27B is 64x
     * K=17408 N=5120 -- ffn_down, the largest tensor in the model. They were still going per-row.
     *
     * Splitting K is a partial-sum decomposition, not an independent one: each slice contributes part of
     * the K-reduction, so the int32 partials SUM. That is arithmetically exact here because the scales
     * live outside this call -- bscale is per output channel (unaffected by a K split) and the activation
     * scale is per row, applied by the caller -- so slicing K only selects columns of an already-quantised
     * A. Bit-exactness is preserved for the same reason the N-slice path preserves it: same tiles, same
     * kernel, different grouping.
     *
     * With Sn==1 the tile index ns*Sk+ks is just ks, so slice ks IS Bb[ks] -- again a shallow view with no
     * repack and no new IOVA. A is gathered per slice because BCHAIN wants [M][Kc] contiguous. */
    if(M>=2 && w->Sn==1 && w->Sk>1 && (w->N%64)==0 && !getenv("ORK_I4_NO_KSLICE_BCHAIN")){
        const int KS = ORK_I4_KS;
        int ok = 1;
        int8_t  *At = (int8_t*) malloc((size_t)M*(size_t)KS);
        int32_t *Ct = (int32_t*)malloc((size_t)M*(size_t)w->N*sizeof(int32_t));
        if(At && Ct){
            for(int ks=0; ks<w->Sk && ok; ks++){
                const int k0 = ks*KS;
                int Kc = w->K - k0; if(Kc > KS) Kc = KS;
                if(Kc <= 0 || (Kc%32)){ ok = 0; break; }         /* the tiler's K granularity */
                for(int m=0; m<M; m++) memcpy(At + (size_t)m*Kc, A + (size_t)m*w->K + k0, (size_t)Kc);
                struct ork_w v = *w;
                v.K = Kc; v.Sk = 1; v.Sn = 1;
                v.Bb = &w->Bb[ks];
                v.Bf = NULL;                                     /* full-K companion is meaningless per slice */
                v.sliced = NULL;
                v.own_buf_valid = 0; v.own_bufs = NULL; v.n_own_bufs = 0;
                v.Bbc_valid = 0; v.Bbc_ns_valid = 0; v.Bbc_ns = NULL;
                if(orki_i4_run_bchain_db(c, &v, M, At, Ct, nc) != 0){ ok = 0; break; }
                const size_t ne = (size_t)M*(size_t)w->N;
                if(ks == 0) memcpy(C, Ct, ne*sizeof(int32_t));
                else for(size_t e=0; e<ne; e++) C[e] += Ct[e];    /* K-split partials SUM */
            }
        } else ok = 0;
        free(At); free(Ct);
        if(ok) return 0;
    }
    /* Wide refuse-prone int4 PREFILL (Sn>1 or K>8192 — the shapes pack built w->sliced for): the per-row
     * orki_i4_run_mc_db below CAN run these but only per-row (~6x slower); route M>=2 straight to the BCHAIN-tiled
     * rescue (measured 663ms -> 107ms at M=128 N=16384). Decode (M==1) stays on the per-row path (cheap). */
    if(M>=2 && w->sliced){ int rs=ork_mm_run_sliced(c,w->sliced,M,A,C,nc); if(rs>=0) return rs; }
    /* decode (M=1) + non-batch shapes ride the per-row doorbell chain (ork_i4_dyn_begin_mc): Sk>1/Sn>1 via
     * chained column/K-slice programs. -4 (over-large chain / unsupported int4 shape) => refuse (rescue-eligible).
     * All blocking int4 paths (i4_mcworker / INCR / CBATCH / blocking BCHAIN) are removed (#45/#52). */
    if(!orki_ork_prof){ int r=orki_i4_run_mc_db(c,w,M,A,C,nc); if(r!=-4) return r;
        if(w->sliced){ int rs=ork_mm_run_sliced(c,w->sliced,M,A,C,nc); if(rs>=0) return rs; }   /* #33: rescue the refused shape via BCHAIN sub-tiles, else refuse */
        return ORK_RC_WEDGE_PRONE; }
    double t0=ork_now_us(); int r=orki_i4_run_mc_db(c,w,M,A,C,nc); orki_prof_i4_us+=ork_now_us()-t0; orki_prof_i4_calls++;
    if(r!=-4) return r;
    if(w->sliced){ int rs=ork_mm_run_sliced(c,w->sliced,M,A,C,nc); if(rs>=0) return rs; }   /* #33: rescue */
    return ORK_RC_WEDGE_PRONE;
}

static inline float *Cf_out_row(float *C,int m,int N){ return C + (size_t)m*N; }
int ork_i4_mm_run_grouped(ork_npu *c,ork_w *w,int M,const int8_t *A,const float *aScale,const float *bScale,float *C){
    /* OFFLINE: the exact per-group accumulate the doorbell drain performs, on the CPU. Per-group scales
     * cannot factor out of the K-sum, so each group's int32 partial is scaled as it is produced and
     * summed in fp32 — same arithmetic, no device. aScale[m*Sk+g], bScale[g*N+n] (the shipped layouts). */
    if(c && c->fd<0 && w && w->cpu_codes && w->gsize>0)
        return orki_cpu_gemm_grouped(M,w->K,w->N,w->gsize,A,w->cpu_codes,aScale,bScale,C);
    if(!w||w->dtype!=DT_I4||!w->gsize) return -1;
    if(orki_check_overlap("ork_i4_mm_run_grouped", (uintptr_t)A, (uintptr_t)A + (size_t)M * w->K, (uintptr_t)C, (uintptr_t)C + (size_t)M * w->N * 4)) return -1;
    /* BCHAIN FAST PATH (M>=2). The row-decomposed doorbell below predates the int4 BCHAIN work and never
     * adopted it: it emits M*Sn*Sk single-row programs, overflows the per-core regcmd budget, and falls into
     * recursive M-chunking — measured 286x a per-channel matmul at M=128,K=3584,G=128.
     *
     * But a K-GROUP is structurally an EXPERT: a [G x N] int4 weight with Sk=1, Sn=1, which is exactly what
     * ork_i4_mm_run's BCHAIN gate wants. Only the FUSED grouped weight (Sk=K/G) fails that gate. So build
     * zero-copy VIEWS over the per-group tiles this weight already holds (w->Bb[g]) and hand the whole set
     * to the multi-expert BCHAIN doorbell — one submit, H-row native batch. Measured 23.6x, a 12x win over
     * the path below, with no new kernels.
     *
     * Falls through to the original path on any refusal, so this can only be faster, never a new failure
     * mode. ORK_I4_GRP_NOBCHAIN=1 forces the old path for A/B. Sn>1 is left to the old path: the views would
     * need one per (ns,g) and the drain would have to stitch N-tiles, which is not what this shape needs. */
    if (M >= 2 && w->Sn == 1 && !getenv("ORK_I4_GRP_NOBCHAIN")) {
        const int G=w->gsize, Sk=w->K/G, N=w->N, K=w->K;
        ork_w  *views = calloc((size_t)Sk, sizeof *views);
        int8_t *Aslice = malloc((size_t)M*K);                       /* NG contiguous [M x G] A-slices */
        int32_t*P      = malloc((size_t)Sk*M*N*sizeof *P);           /* one int32 partial per group */
        ork_mm_task_i4 *tk = calloc((size_t)Sk, sizeof *tk);
        if (views && Aslice && P && tk) {
            for (int g=0; g<Sk; g++) {
                views[g].K=G; views[g].N=N; views[g].Sk=1; views[g].Sn=1; views[g].dtype=DT_I4;
                views[g].owns=0;                                     /* VIEW: ork_mm_free must not free Bb */
                views[g].domain=w->domain; views[g].Bb=&w->Bb[g];
                int8_t *Ag = Aslice + (size_t)g*M*G;
                for (int m=0;m<M;m++) memcpy(Ag+(size_t)m*G, A+(size_t)m*K+(size_t)g*G, (size_t)G);
                tk[g].w=&views[g]; tk[g].M=M; tk[g].A=Ag; tk[g].C=P+(size_t)g*M*N;
            }
            if (orki_i4_run_experts_bchain_db(c, tk, Sk, 0) == 0) {
                #pragma omp parallel for schedule(static) if(M>1)
                for (int m=0;m<M;m++){
                    float *cr=Cf_out_row(C,m,N);
                    for (int n=0;n<N;n++) cr[n]=0.0f;
                    for (int g=0; g<Sk; g++) {
                        const int32_t *pg=P+(size_t)g*M*N+(size_t)m*N;
                        const float as=aScale[(size_t)m*Sk+g]; const float *bs=bScale+(size_t)g*N;
                        for (int n=0;n<N;n++) cr[n]+=(float)pg[n]*as*bs[n];
                    }
                }
                free(tk); free(P); free(Aslice); free(views);
                return 0;
            }
        }
        free(tk); free(P); free(Aslice); free(views);               /* refused -> original path below */
    }

    /* Grouped int4 runs on the NONBLOCK doorbell (row-decomposed Sn*Sk chain + float scale-accumulate
     * drain). NULL (chain/scratch too big / ineligible) => refuse (rescue-eligible); the blocking
     * i4_mcworker_g path is removed (#45). */
    ork_dyn_chain *hg=ork_i4_dyn_begin_mc_grouped(c,M,w,A,aScale,bScale,C,0);
    if(hg) return ork_dyn_grouped_end(hg)?-1:0;
    /* #33 GROUPED RESCUE (M-chunk): the doorbell refused because the per-core program count
     * (rows/core)*Sn*Sk exceeds the regcmd-buffer cap (~70). Rows are INDEPENDENT, so M-CHUNK the rows —
     * each chunk is a full grouped matmul (all groups + all N, so the per-group float drain stays intact)
     * writing its own CONTIGUOUS C rows: no weight repack, no scale slicing, no scatter, just a recursive
     * call with fewer rows (which no longer refuses once rows/core*Sn*Sk <= cap). Only possible when a SINGLE
     * row fits (Sn*Sk <= cap); wide-N (Sn>1) or fine-group large-K (Sn*Sk > cap even at 1 row) still refuses —
     * needs N-tile / K-group-slice (a follow-on). The M>1 gate bounds the recursion (bottoms out at M==1). */
    int G=w->gsize, Sk=w->K/G, per_row=w->Sn*Sk;
    if(M>1 && per_row>0 && per_row<=64){                                 /* 64 = margin under the ~70-program/core cap */
        int nc=orki_budget(c,M); if(nc<1)nc=1; int rpc=64/per_row; if(rpc<1)rpc=1;
        int Msub=rpc*nc; if(Msub>=M) Msub=M-1; if(Msub<1) Msub=1;
        int ok=1;
        for(int m0=0;m0<M && ok;m0+=Msub){ int mm=(M-m0<Msub)?(M-m0):Msub;
            if(ork_i4_mm_run_grouped(c,w,mm, A+(size_t)m0*w->K, aScale+(size_t)m0*Sk, bScale, C+(size_t)m0*w->N)) ok=0; }
        if(ok) return 0;
    }
    return ORK_RC_WEDGE_PRONE;
}

int orki_i4_slice_run(ork_npu *c, ork_w_sliced *w, int M, const int8_t *A, int32_t *C, int nc) {
    if (!c || !w || !A || !C || M < 1) return -1;
    int ks = w->ks, ns = w->ns, nks = w->nks, nnt = w->nnt, S = nks * nnt, K = w->K, N = w->N, Kpad = w->Kpad;
    int8_t  *Aslc = malloc((size_t) M * Kpad);
    int32_t *part = malloc((size_t) nks * M * N * sizeof(int32_t));
    ork_mm_task_i8 *tasks = malloc((size_t) S * sizeof *tasks);
    if (!Aslc || !part || !tasks) { free(Aslc); free(part); free(tasks); return -1; }
    size_t aoff = 0, poff = 0; int rc = 0;
    for (int ki = 0; ki < nks && !rc; ki++) { int k0 = ki*ks, k1 = k0+ks < Kpad ? k0+ks : Kpad, Ks = k1-k0;
        int8_t *aptr = Aslc + aoff;
        int real = K - k0; if (real > Ks) real = Ks; if (real < 0) real = 0;   /* real A cols this slice; PAD tail -> 0 */
        for (int m = 0; m < M; m++) { memcpy(aptr + (size_t) m*Ks, A + (size_t) m*K + k0, real);
                                      if (real < Ks) memset(aptr + (size_t) m*Ks + real, 0, Ks - real); }
        aoff += (size_t) M * Ks;
        for (int ni = 0; ni < nnt && !rc; ni++) { int n0 = ni*ns, n1 = n0+ns < N ? n0+ns : N, Nw = n1-n0;
            int32_t *ptile = part + poff; poff += (size_t) M * Nw;
            tasks[ki*nnt + ni] = (ork_mm_task_i8){ w->sub[ki*nnt + ni], M, aptr, ptile };
            int nct = nc>0 ? nc : c->soc->cores; int nb = Nw/64; if (nct > nb) nct = nb; if (nct < 1) nct = 1;
            int r = (M >= 2) ? orki_i4_run_bchain_db(c, w->sub[ki*nnt + ni], M, aptr, ptile, nct)   /* BCHAIN batch on the doorbell */
                             : orki_i4_run_mc_db    (c, w->sub[ki*nnt + ni], M, aptr, ptile, nct);  /* M==1 per-row doorbell */
            if (r < 0) rc = -1; } }
    if (!rc) {   /* per-core PARALLEL ks-outer int32 accumulate + N scatter (same worker as int8) */
        int anc = nc>0 ? nc : c->soc->cores; if(anc>ORK_MAXCORE) anc=ORK_MAXCORE; if(anc<1) anc=1;
        struct slc_acc acc[ORK_MAXCORE];
        for(int i=0;i<anc;i++){ int cc0=(int)((long)i*N/anc), cc1=(int)((long)(i+1)*N/anc);
            acc[i]=(struct slc_acc){ tasks, C, nks, nnt, ns, N, M, cc0, cc1 }; }
        if(anc==1){ orki_slice_acc_worker(&acc[0]); }
        else {
            orki_npu_pool_ensure(c);
            pthread_mutex_lock(&c->pmu);
            c->pjob=acc; c->pjob_nc=anc; c->pjob_fn=orki_slice_acc_worker; c->pjob_stride=sizeof(struct slc_acc);
            c->pdone=0; c->pgen++; pthread_cond_broadcast(&c->pgo);
            pthread_mutex_unlock(&c->pmu);
            orki_slice_acc_worker(&acc[0]);
            pthread_mutex_lock(&c->pmu); while(c->pdone < anc-1) pthread_cond_wait(&c->pdn,&c->pmu); pthread_mutex_unlock(&c->pmu);
        }
    }
    free(Aslc); free(part); free(tasks); return rc;
}

int ork_i4_bmm_strided(ork_npu *c, int nbatch, int M, int K, int N,
                       const int8_t *A, const int8_t *B, int32_t *C, const ork_bmm_strides *s){
    if(!c||!A||!B||!C||!s) return -1;
    if(nbatch<1||M<1||K<1||N<1) return -2;
    if(K%32||N%64) return -2;
    int8_t *Ac=malloc((size_t)M*K), *Bc=malloc((size_t)K*N);
    int cdense=orki_bmm_c_dense(s,N); int32_t *Cc = cdense?NULL:malloc((size_t)M*N*sizeof(int32_t));
    if(!Ac||!Bc||(!cdense&&!Cc)){ free(Ac);free(Bc);free(Cc); return -3; }
    int rc=0;
    for(int b=0;b<nbatch;b++){
        orki_i8_bmm_gather(Bc,B+(long)b*s->bbs,K,N,s->bs_k,s->bs_n);
        orki_i8_bmm_gather(Ac,A+(long)b*s->abs,M,K,s->as_m,s->as_k);
        ork_w *w=ork_i4_mm_pack(c,K,N,Bc); if(!w){ rc=-3; break; }
        int32_t *Cout = cdense ? C+(long)b*s->cbs : Cc;
        int r=ork_i4_mm_run(c,w,M,Ac,Cout);
        ork_mm_free(c,w);
        if(r){ rc=-5; break; }
        if(!cdense) orki_bmm_scatter_i32(C+(long)b*s->cbs,Cc,M,N,s->cs_m,s->cs_n);
    }
    free(Ac);free(Bc);free(Cc);
    return rc;
}

int ork_i4_bmm(ork_npu *c, int nbatch, int M, int K, int N,
               const int8_t *A, const int8_t *B, int32_t *C){
    ork_bmm_strides s=orki_bmm_natural(M,K,N); return ork_i4_bmm_strided(c,nbatch,M,K,N,A,B,C,&s);
}
int orki_i4_run_mc_db(ork_npu *c, ork_w *w, int M, const int8_t *A, int32_t *C, int nc){
    ork_mm_task_i8 *tk = malloc((size_t)M * sizeof *tk); if(!tk) return -4;
    for(int m=0;m<M;m++) tk[m]=(ork_mm_task_i8){ w, 1, A + (size_t)m*w->K, C + (size_t)m*w->N };
    ork_dyn_chain *h = ork_i4_dyn_begin_mc(c, M, tk, nc);
    free(tk);
    if(!h) return -4;   /* shape/buffer limit -> caller refuses (ORK_RC_WEDGE_PRONE) */
    int d = ork_dyn_end(h);
    return (d == M-1) ? 0 : -1;
}
