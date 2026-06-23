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
#include "ork_npu.h"
#include "soc.h"
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
typedef ork_f16 f16;
enum { DT_F16=0, DT_I8=1, DT_I4=2 };
#define ORK_MAXCORE 4   /* RK3576=2, RK3588=3; headroom for future parts. Actual = soc->cores. */

struct buf { uint32_t handle; uint64_t dma, obj; void *cpu; size_t size; };
struct ork_pw { struct ork_npu *c; int id; };   /* persistent NPU-pool worker arg */
struct ork_npu { int fd; const struct ork_soc *soc; struct buf regcmd, task, Af, Cc; size_t ccsz; void *cres; size_t cressz; int warmed, last_dt; int core_budget;
    /* multi-core (ORK_NPU_MC): per-core regcmd/task/feature/output so cores submit concurrently */
    struct buf mrc[ORK_MAXCORE], mtk[ORK_MAXCORE], maf[ORK_MAXCORE], mcc[ORK_MAXCORE], mtk_all;
    size_t mccsz[ORK_MAXCORE]; int mwarm[ORK_MAXCORE]; int mc_alloc;
    /* persistent worker pool: spawned once, signalled per matmul (cuts per-matmul create/join) */
    pthread_t pth[ORK_MAXCORE]; struct ork_pw pwa[ORK_MAXCORE]; int pool_n;
    pthread_mutex_t pmu; pthread_cond_t pgo, pdn; void *pjob; int pjob_nc, pgen, pdone, pstop;
    pthread_barrier_t b_ioctl; int mc_submit_rc; int mc_error;
    /* zero-copy registry: caller-allocated NPU-coherent DMA buffers (ork_dma_alloc). When a matmul's
     * A/C live in one of these, the regcmd points at them directly — no host gather/writeout memcpy. */
    struct buf dma_tab[64]; int dma_n; };
struct ork_w   { int K, N, Sk, Sn, dtype, gsize; struct buf *Bb; struct buf *Bf; };
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
/* Bb[ns*Sk+ks] = K-split x N-split (always). Bf[ns] = optional full-K per N-slice (ORK_FULLK_DEC,
 * int8 K<=10752): lets the multi-core DECODE path do ONE submit/core instead of ~K/1024 K-slices.
 * ~2x weight memory (dual layout) — fits IOVA for int8 ~1.7B; can overflow for larger/fp16. */
/* Auto-tuner policy. Multi-core + full-K decode are now the library's DEFAULT per-matmul choice
 * (no env needed); the engine sets a core budget via ork_npu_set_core_budget.
 * The driver automatically selects the core count by N-tile count, and uses full-K single-submits
 * when M is small, K<=10752, precision is int8, and it fits within IOVA. */
static int budget(ork_npu*c, int M){
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
static struct buf bcreate(int fd,size_t size,uint32_t flags){
    struct rknpu_mem_create c; memset(&c,0,sizeof c); c.size=pgup(size); c.flags=flags; c.core_mask=RKNPU_CORE0_MASK;
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_CREATE,&c)){perror("CREATE");return (struct buf){0};}
    struct rknpu_mem_map m; memset(&m,0,sizeof m); m.handle=c.handle;
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_MAP,&m)){perror("MAP");return (struct buf){0};}
    void*p=mmap(NULL,c.size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,m.offset);
    if(p==MAP_FAILED){perror("mmap");return (struct buf){0};}
    return (struct buf){c.handle,c.dma_addr,c.obj_addr,p,c.size};
}
static void bdestroy(int fd,struct buf*b){ if(!b->cpu)return; munmap(b->cpu,b->size);
    struct rknpu_mem_destroy d; memset(&d,0,sizeof d); d.handle=b->handle; d.obj_addr=b->obj; ioctl(fd,DRM_IOCTL_RKNPU_MEM_DESTROY,&d); b->cpu=0; }
static void bsync(int fd,struct buf*b,uint32_t f){struct rknpu_mem_sync s;memset(&s,0,sizeof s);s.obj_addr=b->obj;s.size=b->size;s.flags=f;ioctl(fd,DRM_IOCTL_RKNPU_MEM_SYNC,&s);}
static void act(int fd,uint32_t f,uint32_t v){struct rknpu_action a={.flags=f,.value=v};ioctl(fd,DRM_IOCTL_RKNPU_ACTION,&a);}

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

static int rknpu_submit_ioctl(int fd, struct rknpu_submit *sub) {
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
        int R=(2*cbuf)/K; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; } int rows=(mc+1<R)?(mc+1):R; setr(rc,REGCMD_I8_N,0x201,0x1010,16*rows);
        /* 0x1040 = K-reduction schedule, selected per 64-row group. MUST be ceil(mc/64): a tile of
         * 65..127 rows spills past the first 64-row group, so it needs the NEXT schedule (the one a
         * full 128-row tile uses), not the <=64 schedule. floor(mc/64) gave 65..127-row tiles the
         * <=64 schedule -> rows 64..mc-1 computed against the wrong K-partition (the prefill bug). */
        double scale=(double)K/512.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale),mg=(mc+63)/64; if(mg<1)mg=1;
        int v=base-slope*(mg-1); if(v<0x1b)v=0x1b; setr(rc,REGCMD_I8_N,0x201,0x1040,v);
    } else { setr(rc,REGCMD_I8_N,0x201,0x1010,16*(mc+1)); }
    setr(rc,REGCMD_I8_N,0x201,0x1070,aA);setr(rc,REGCMD_I8_N,0x201,0x1110,aB);setr(rc,REGCMD_I8_N,0x1001,0x4020,aC);
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

ork_npu *ork_npu_init(void){
    const struct ork_soc *soc=ork_soc_detect();
    if(!soc){fprintf(stderr,"[ork] unknown SoC (no device-tree match) — cannot select NPU params\n");return NULL;}
    if(!soc->validated) fprintf(stderr,"[ork] WARNING: %s params are inherited/untested — validate with the regression suite\n",soc->id);
    const char*card=getenv("ORK_NPU_CARD"); if(!card)card=soc->card;
    int fd=open(card,O_RDWR); if(fd<0){perror("open NPU card");return NULL;}
    act(fd,RKNPU_GET_DRV_VERSION,0);act(fd,RKNPU_POWER_ON,0);act(fd,RKNPU_SET_PROC_NICE,(uint32_t)-19);
    ork_npu *c=calloc(1,sizeof *c); c->fd=fd; c->soc=soc; c->last_dt=-1; c->core_budget=soc->cores;
    pthread_mutex_init(&c->pmu,NULL); pthread_cond_init(&c->pgo,NULL); pthread_cond_init(&c->pdn,NULL);
    c->regcmd=bcreate(fd,2097152,0x403); c->task=bcreate(fd,524288,0x40b); c->Af=bcreate(fd,(size_t)4*32768*2,0x403);
    struct rknpu_task t; memset(&t,0,sizeof t); t.enable_mask=0xd;t.int_mask=0x300;t.int_clear=0x1ffff;t.regcfg_amount=108;t.regcmd_addr=c->regcmd.dma;
    memcpy(c->task.cpu,&t,sizeof t); bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    if(!c->regcmd.cpu||!c->task.cpu||!c->Af.cpu){ork_npu_free(c);return NULL;}
    g_npu_ctx = c;
    return c;
}
void ork_npu_free(ork_npu *c){ if(!c)return; int fd=c->fd;
    if (g_npu_ctx == c) g_npu_ctx = NULL;
    if(c->pool_n){ pthread_mutex_lock(&c->pmu); c->pstop=1; pthread_cond_broadcast(&c->pgo); pthread_mutex_unlock(&c->pmu);
        for(int i=1;i<c->pool_n;i++) pthread_join(c->pth[i],NULL); }
    bdestroy(fd,&c->regcmd);bdestroy(fd,&c->task);bdestroy(fd,&c->Af);bdestroy(fd,&c->Cc);
    for(int i=0;i<ORK_MAXCORE;i++){bdestroy(fd,&c->mrc[i]);bdestroy(fd,&c->mtk[i]);bdestroy(fd,&c->maf[i]);bdestroy(fd,&c->mcc[i]);}
    for(int i=0;i<c->dma_n;i++) bdestroy(fd,&c->dma_tab[i]);
    free(c->cres); if(fd>=0)close(fd); free(c); }

/* ---- zero-copy DMA buffers (NPU-coherent, CPU-mapped). A matmul whose A and/or C live in one of
 * these has the regcmd point at it directly — no host gather/writeout memcpy. ork_mm_run_i8 detects
 * residency automatically (no API change); the caller just allocates A/C here. ---- */
void *ork_dma_alloc(ork_npu *c, size_t size){
    if(!c || c->dma_n >= (int)(sizeof c->dma_tab/sizeof c->dma_tab[0])) return NULL;
    struct buf b=bcreate(c->fd,size,0x401); if(!b.cpu) return NULL;
    c->dma_tab[c->dma_n++]=b; return b.cpu;
}
void ork_dma_free(ork_npu *c, void *ptr){
    if(!c||!ptr) return;
    for(int i=0;i<c->dma_n;i++) if(c->dma_tab[i].cpu==ptr){ bdestroy(c->fd,&c->dma_tab[i]); c->dma_tab[i]=c->dma_tab[--c->dma_n]; memset(&c->dma_tab[c->dma_n], 0, sizeof(struct buf)); return; }
}
/* the registered DMA buffer containing host ptr p, or NULL if p isn't zero-copy-resident */
static struct buf *dma_find(ork_npu *c, const void *p){
    for(int i=0;i<c->dma_n;i++){ char*base=c->dma_tab[i].cpu;
        if((const char*)p>=base && (const char*)p<base+c->dma_tab[i].size) return &c->dma_tab[i]; }
    return NULL;
}
const char *ork_npu_soc(const ork_npu *c){return c->soc->id;}
int ork_npu_cores(const ork_npu *c){return c->soc->cores;}
int ork_npu_validated(const ork_npu *c){return c->soc->validated;}
/* policy: cap the cores the auto-tuner may use for a matmul (n<=0 → all soc cores). The library
 * still picks per-matmul ≤ this (small-N matmuls use fewer). ORK_NPU_MC env overrides if set. */
void ork_npu_set_core_budget(ork_npu *c,int n){ if(!c)return; c->core_budget=(n>0&&n<=c->soc->cores)?n:c->soc->cores; }

/* pack B[K,N] (row-major) into resident NPU tiles. dt: DT_F16 (B fp16, tile [Nt][Kt][16][32],
 * N%16) or DT_I8 (B int8, tile [Nt][Kt][32][32], N%32). K-split (KS) x N-split (NMAX). */
static ork_w *pack(ork_npu *c,int K,int N,const void *B,int dt){
    int nmod=dt?32:16; if(K%32||N%nmod) return NULL;
    int KS=dt ? 1024 : c->soc->ks, NMAX=c->soc->nmax, nt_sz=dt?32:16, esz=dt?1:2;
    int Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=dt; w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/nt_sz;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=bcreate(c->fd,(size_t)Kp*Nc*esz,0x403);
        if(!b->cpu){
            fprintf(stderr,"[ork] ERROR: bcreate failed to allocate weight buffer Bb[%zu] in pack (size=%zu)\n",(size_t)ns*Sk+ks,(size_t)Kp*Nc*esz);
            for(int i=0;i<ns*Sk+ks;i++) bdestroy(c->fd,&w->Bb[i]);
            free(w->Bb); free(w); return NULL;
        }
        if(dt==DT_F16){ f16*bb=b->cpu; const f16*Bf=B;
            for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
                bb[nt*KT*16*32+kt*16*32+nl*32+kk]=Bf[(size_t)(k0+kt*32+kk)*N+(n0+nt*16+nl)];
        } else { int8_t*bb=b->cpu; const int8_t*Bi=B;
            for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
                bb[nt*KT*32*32+kt*32*32+nl*32+kk]=Bi[(size_t)(k0+kt*32+kk)*N+(n0+nt*32+nl)];
        }
        bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}}
    /* AUTO full-K decode layout (int8, K<=10752): lets the multi-core decode do
     * one full-K submit/core instead of ~K/1024 K-slices. ~2x weight memory — IOVA-FITS GUARD: if
     * any bcreate fails (IOMMU full on a big model), abandon Bf entirely → decode falls back to the
     * K-split path (correct, just slower). No crash, no ceiling guess. */
    if(dt==DT_I8 && K<=10752){ int KTf=K/32; w->Bf=calloc(Sn,sizeof(struct buf)); int ok=1;
        for(int ns=0;ns<Sn && ok;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/nt_sz;
            struct buf*b=&w->Bf[ns]; *b=bcreate(c->fd,(size_t)K*Nc*esz,0x403);
            if(!b->cpu){ ok=0; break; }                 /* IOVA full → give up on Bf */
            int8_t*bb=b->cpu; const int8_t*Bi=B;
            for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KTf;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
                bb[(size_t)nt*KTf*32*32+(size_t)kt*32*32+nl*32+kk]=Bi[(size_t)(kt*32+kk)*N+(n0+nt*32+nl)];
            bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}
        if(!ok){ for(int ns=0;ns<Sn;ns++) bdestroy(c->fd,&w->Bf[ns]); free(w->Bf); w->Bf=NULL; } }
    return w;
}
ork_w *ork_mm_pack   (ork_npu *c,int K,int N,const f16    *B){ return pack(c,K,N,B,DT_F16); }
ork_w *ork_mm_pack_i8(ork_npu *c,int K,int N,const int8_t *B){ return pack(c,K,N,B,DT_I8);  }
void ork_w_free(ork_w *w){ if(!w)return; free(w->Bb); free(w->Bf); free(w); }   /* device buffers freed at ctx teardown */

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
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I4;
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    for(int ns=0;ns<Sn;ns++)for(int ks=0;ks<Sk;ks++){
        int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=bcreate(c->fd,(size_t)Kp*Nc/2,0x403);
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
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I4;w->gsize=G;
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    for(int ns=0;ns<Sn;ns++)for(int g=0;g<Sk;g++){
        int k0=g*G,n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
        struct buf*b=&w->Bb[(size_t)ns*Sk+g]; *b=bcreate(c->fd,(size_t)G*Nc/2,0x403);
        if(!b->cpu){
            fprintf(stderr,"[ork] ERROR: bcreate failed to allocate weight buffer Bb[%zu] in pack_i4_grouped (size=%zu)\n",(size_t)ns*Sk+g,(size_t)G*Nc/2);
            for(int i=0;i<ns*Sk+g;i++) bdestroy(c->fd,&w->Bb[i]);
            ork_w_free(w); return NULL;
        }
        tile_i4_Bslice(b->cpu,B,K,N,k0,G,n0,Nc);
        bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);
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
    return run_i4_mc(c,w,M,A,C,nc);
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
        if(rknpu_submit_ioctl(fd,&sub)){ if(last){perror("SUBMIT");return -1;} continue; }
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
        c->mtk_all=bcreate(fd, ORK_MAXCORE * sizeof(struct rknpu_task), 0x40b);
        if(!c->mtk_all.cpu) {
            fprintf(stderr, "[ork] ERROR: mc_ensure failed to allocate mtk_all task buffer (IOMMU full?)\n");
            return -1;
        }
    }
    for(int i=0;i<nc;i++){
        if(c->mrc[i].cpu) continue;        /* alloc once, per core, up to the max ever requested */
        c->mrc[i]=bcreate(fd,65536,0x403); c->mtk[i]=bcreate(fd,65536,0x40b); c->maf[i]=bcreate(fd,(size_t)4*32768*2,0x403);
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
void ork_npu_mc_reset(void){ for(int i=0;i<MCPROF_MAX;i++){g_mc_copy[i]=g_mc_sub[i]=g_mc_acc[i]=0;g_mc_n[i]=0;} }
void ork_npu_mc_timing(int core,double*copy,double*sub,double*acc,long*n){
    if(copy)*copy=g_mc_copy[core]; if(sub)*sub=g_mc_sub[core]; if(acc)*acc=g_mc_acc[core]; if(n)*n=g_mc_n[core]; }

struct mcw { ork_npu *c; int core, nc, dt, M; const void *A; ork_w *w; void *cres; int rc; int reps; size_t maxout; };

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
        if(rknpu_submit_ioctl(fd,&sub)){ if(last){ a->rc=-1; return; } }
        bsync(fd,CC,RKNPU_MEM_SYNC_FROM_DEVICE);
    }
    c->mwarm[i]=1;
}

static void *mcworker(void *vp){
    struct mcw *a=vp; ork_npu *c=a->c; int i=a->core, nc=a->nc, dt=a->dt, M=a->M, fd=c->fd;
    int K=a->w->K, N=a->w->N, NMAX=c->soc->nmax, CBUF=c->soc->cbuf_elems;
    int KS=dt ? 1024 : c->soc->ks, RB=dt?2*CBUF:CBUF, nt_sz=dt?32:16;
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
    if(dt==DT_I8 && M>1 && w->Bf && (K%512)==0 && K<=4096){   /* Tier 1c-ii: full-K PREFILL — one submit/M-tile over full K, no K-split accumulate */
        int Kp=K, R=RB/Kp; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; }
        double scale=(double)Kp/512.0; int base=(int)(177.0-15.0*(scale-1.0)),slope=(int)(15.0*scale), mg_max = base>=0x1b ? (base-0x1b)/slope+1 : 0;
        int chunk = mg_max * 64; if(chunk > R - 1) chunk = R - 1; if(chunk < 1) chunk = 1; if(chunk > M) chunk = M;
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
        mygen=c->pgen; int nc=c->pjob_nc; struct mcw *args=c->pjob; pthread_mutex_unlock(&c->pmu);
        if(id<nc){ mcworker(&args[id]);
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
    if(dt!=c->last_dt){ if(dt==DT_I8) act(fd,RKNPU_ACT_RESET,0); for(int i=0;i<ORK_MAXCORE;i++){c->mwarm[i]=0;c->mccsz[i]=0;} c->last_dt=dt; }
    if(mc_ensure(c,nc)) return -1;

    /* Pre-allocate multi-core buffers on the single calling thread to eliminate concurrent allocations / race conditions */
    int N=w->N, K=w->K, NMAX=c->soc->nmax, CBUF=c->soc->cbuf_elems;
    int KS=dt ? 1024 : c->soc->ks, RB=dt?2*CBUF:CBUF, nt_sz=dt?32:16;
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
            int chunk = mg_max * 64; if(chunk > R - 1) chunk = R - 1; if(chunk < 1) chunk = 1; if(chunk > M) chunk = M;
            int rows=chunk<M?chunk:M;
            size_t sz=(size_t)rows*Kp*1;
            if(sz>maxaf)maxaf=sz;
        }
        core_maxout[i] = maxout;
        if(c->mccsz[i]<maxout){
            bdestroy(fd,&c->mcc[i]);
            c->mcc[i]=bcreate(fd,maxout,0x403);
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
            c->maf[i]=bcreate(fd,maxaf,0x403);
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
    for(int i=0;i<nc;i++) args[i]=(struct mcw){c,i,nc,dt,M,A,w,c->cres,0,reps,core_maxout[i]};
    npu_pool_ensure(c);
    c->mc_error = 0;
    if(nc>1) pthread_barrier_init(&c->b_ioctl, NULL, nc);
    const double t1=ork_now_us();
    pthread_mutex_lock(&c->pmu); c->pjob=args; c->pjob_nc=nc; c->pdone=0; c->pgen++; pthread_cond_broadcast(&c->pgo); pthread_mutex_unlock(&c->pmu);
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
                    rc[216] = 0; rc[217] = 0; rc[218] = 0x00000014; rc[219] = 0x01010000;
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
                    if (rknpu_submit_ioctl(fd, &sub)) {
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
    for(int i=0;i<nc;i++){ if(c->mccsz[i]<osz){ bdestroy(fd,&c->mcc[i]); c->mcc[i]=bcreate(fd,osz,0x403); c->mccsz[i]=osz; c->mwarm[i]=0; if(!c->mcc[i].cpu){fprintf(stderr, "[ork] ERROR: failed to allocate multi-core output mcc[%d] (size=%zu)\n", i, osz);return -2;} } }
    size_t asz=(size_t)M*ORK_I4_KS/2;
    if(asz < (size_t)4*32768*2) asz=(size_t)4*32768*2;
    for(int i=0;i<nc;i++){ if(c->maf[i].size<asz){ bdestroy(fd,&c->maf[i]); c->maf[i]=bcreate(fd,asz,0x403); if(!c->maf[i].cpu){fprintf(stderr, "[ork] ERROR: failed to allocate multi-core activation maf[%d] (size=%zu)\n", i, asz);return -2;} } }
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
                    if (rknpu_submit_ioctl(fd, &sub)) {
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
    for(int i=0;i<nc;i++){ if(c->mccsz[i]<osz){ bdestroy(fd,&c->mcc[i]); c->mcc[i]=bcreate(fd,osz,0x403); c->mccsz[i]=osz; c->mwarm[i]=0; if(!c->mcc[i].cpu){fprintf(stderr, "[ork] ERROR: failed to allocate grouped multi-core output mcc[%d] (size=%zu)\n", i, osz);return -2;} } }
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
    /* auto-tuner: pick cores ≤ budget, capped so each gets ≥2 N-tiles (tiny matmuls don't pay the
     * multi-core spawn). budget defaults to all soc cores; ORK_NPU_MC / set_core_budget cap it. */
    int b=budget(c, M), cores=c->soc->cores, NN=w->N/(w->dtype?32:16);
    int nc=b<cores?b:cores; if(nc>NN)nc=NN; while(nc>1 && NN<nc*2)nc--;
    if(nc>1) return run_multicore(c,w,M,A,C,nc);
    pin_big_core(0);                                   /* single-core path also runs on the calling thread */
    int fd=c->fd,K=w->K,N=w->N, dt=w->dtype, NMAX=c->soc->nmax, CBUF=c->soc->cbuf_elems;
    int KS=dt ? 1024 : c->soc->ks, RB=dt?2*CBUF:CBUF;     /* rows budget: int8 packs 2x rows/CBUF */
    /* entering int8 mode wedges the first submit unless the NPU is reset first (fp16 never
     * wedges — it cold-starts stale, which the warmup handles). Reset only when switching INTO
     * int8 — keeps fp16-only contexts free of any reset/log. Then re-warm on a fresh buffer. */
    if(dt!=c->last_dt){ if(dt==DT_I8) act(fd,RKNPU_ACT_RESET,0); c->warmed=0; c->ccsz=0; c->last_dt=dt; }
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
        int chunk = mg_max * 64; if(chunk > R - 1) chunk = R - 1; if(chunk < 1) chunk = 1; if(chunk > M) chunk = M;
        int rows=chunk<M?chunk:M;
        size_t sz=(size_t)rows*Kp*1;
        if(sz>maxaf)maxaf=sz;
    }
    if(c->ccsz<maxout){bdestroy(fd,&c->Cc);c->Cc=bcreate(fd,maxout,0x403);c->ccsz=maxout;c->warmed=0; if(!c->Cc.cpu){fprintf(stderr, "[ork] ERROR: failed to allocate single-core/pre-core output buffer Cc (size=%zu, IOMMU full?)\n", maxout);return -1;}}
    if(c->Af.size<maxaf){
        bdestroy(fd,&c->Af);
        c->Af=bcreate(fd,maxaf,0x403);
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
        int chunk = mg_max * 64; if(chunk > R - 1) chunk = R - 1; if(chunk < 1) chunk = 1; if(chunk > M) chunk = M;
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
        int sched=dt?(Kp==1024||Kp==512):((Kp&(Kp-1))==0), R=RB/Kp; if(R<1)R=1; { int rp2=1; while(rp2*2<=R)rp2*=2; R=rp2; } int chunk=sched?4*R:((RB/2)/Kp); if(chunk<1)chunk=1;
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
    return run(c,w,M,A,C);
}

/* RE/calibration: run ONE M=1 full-K int8 submit (no K-split) at (K,N) to probe this SoC's
 * single-submit K-tile ceiling (`0x1044`). Allocates its own buffers — does not touch resident
 * weights. Returns 0 if the submit completed (C[N] int32 valid), -1 if it wedged (K over the
 * per-op K-tile cap; recoverable — the next call's RKNPU_ACT_RESET clears it), -2 on bad dims. */
int ork_npu_probe_single_i8(ork_npu *c,int K,int N,const int8_t *A,const int8_t *B,int32_t *C){
    int fd=c->fd, CBUF=c->soc->cbuf_elems;
    if(K%32||N%32||N>c->soc->nmax) return -2;
    struct buf W=bcreate(fd,(size_t)K*N,0x403); if(!W.cpu) return -2;
    int NN=N/32,KT=K/32; int8_t*bb=W.cpu;     /* int8 tile layout [Ntile][Ktile][32][32], full K */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*32+nl)];
    bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=bcreate(fd,(size_t)N*4,0x403); if(!O.cpu){bdestroy(fd,&W);return -2;}
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
        if(rknpu_submit_ioctl(fd,&sub)){ ok=-1; continue; }
        bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); memcpy(C,O.cpu,(size_t)N*4); ok=0; }
    bdestroy(fd,&W);bdestroy(fd,&O);
    return ok;
}

static double ork_now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec*1e-3; }
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
    struct buf W=bcreate(fd,(size_t)K*N,0x403); if(!W.cpu) return -2;
    memset(W.cpu,1,(size_t)K*N); bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=bcreate(fd,(size_t)N*4,0x403); if(!O.cpu){bdestroy(fd,&W);return -2;}
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
    if(rknpu_submit_ioctl(fd,&sub)){bdestroy(fd,&W);bdestroy(fd,&O);return -1;} bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); /* warm */
    double t0=ork_now_us();                          /* (a) ntask separate ioctls */
    for(int i=0;i<ntask;i++){ sub.task_number=1; sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,1};
        if(rknpu_submit_ioctl(fd,&sub)){bdestroy(fd,&W);bdestroy(fd,&O);return -1;} bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); }
    *us_unbatched=ork_now_us()-t0;
    sub.task_number=ntask; sub.subcore_task[0]=sub.subcore_task[1]=sub.subcore_task[2]=(struct rknpu_subcore_task){0,(uint32_t)ntask};
    t0=ork_now_us();                                 /* (b) one ioctl, ntask tasks */
    if(rknpu_submit_ioctl(fd,&sub)){perror("batched SUBMIT");bdestroy(fd,&W);bdestroy(fd,&O);return -1;} bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE);
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
    struct buf W=bcreate(fd,(size_t)Kfull*N*2,0x403); if(!W.cpu) return -2;
    int NN=N/16,KTf=Kfull/32; f16*bb=W.cpu;     /* full-K fp16 layout [Ntile][KTfull][16][32] */
    for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KTf;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
        bb[(size_t)nt*KTf*16*32+(size_t)kt*16*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(nt*16+nl)];
    bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=bcreate(fd,(size_t)N*4,0x403); if(!O.cpu){bdestroy(fd,&W);return -2;}
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
        if(rknpu_submit_ioctl(fd,&sub)){ ok=-1; continue; }
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
    struct buf W=bcreate(fd,(size_t)K*N/2,0x403); if(!W.cpu) return -2;        /* B int4: half bytes */
    tile_i4_B(W.cpu,B,K,N,nibB);
    bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=bcreate(fd,(size_t)M*N*2,0x403); if(!O.cpu){bdestroy(fd,&W);return -2;}  /* int16 C, M rows */
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
        for(int rep=0;rep<2;rep++){ if(rknpu_submit_ioctl(fd,&sub)){ ok=-1; continue; }
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
    struct buf W=bcreate(fd,(size_t)K*N/2,0x403); if(!W.cpu) return -2;
    tile_i4_B(W.cpu,B,K,N,0);
    bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(fd,&W,RKNPU_MEM_SYNC_TO_DEVICE);
    struct buf O=bcreate(fd,(size_t)M*N*2,0x403); if(!O.cpu){bdestroy(fd,&W);return -2;}
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
    for(int rep=0;rep<2;rep++){ sub.timeout=60000; if(rknpu_submit_ioctl(fd,&sub)){ ok=-1; continue; }
        bsync(fd,&O,RKNPU_MEM_SYNC_FROM_DEVICE); ok=0; }
    if(ok==0) memcpy(raw,O.cpu,(size_t)M*N*2);
    bdestroy(fd,&W);bdestroy(fd,&O);
    return ok;
}

int ork_npu_probe_chain_i8(ork_npu *c, int S, int K, int N, const int8_t *A, const int8_t *B, int32_t *C) {
    int fd = c->fd, CBUF = c->soc->cbuf_elems;
    if (K % 32 || N % 32 || N > c->soc->nmax || S < 1 || S > 32) return -2;
    struct buf W = bcreate(fd, (size_t)K * N, 0x403); if (!W.cpu) return -2;
    int NN = N / 32, KT = K / 32; int8_t *bb = W.cpu;
    for (int nt = 0; nt < NN; nt++) for (int kt = 0; kt < KT; kt++) for (int nl = 0; nl < 32; nl++) for (int kk = 0; kk < 32; kk++)
        bb[(size_t)nt * KT * 32 * 32 + (size_t)kt * 32 * 32 + nl * 32 + kk] = B[(size_t)(kt * 32 + kk) * N + (nt * 32 + nl)];
    bsync(fd, &W, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE); bsync(fd, &W, RKNPU_MEM_SYNC_TO_DEVICE);
    
    struct buf O = bcreate(fd, (size_t)S * 4096, 0x403); if (!O.cpu) { bdestroy(fd, &W); return -2; }
    
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
        if (rknpu_submit_ioctl(fd, &sub)) { ok = -1; continue; }
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
    
    struct buf W = bcreate(fd, (size_t)K * N, 0x403);
    struct buf A = bcreate(fd, (size_t)S * K, 0x403);
    struct buf O = bcreate(fd, (size_t)S * 4096, 0x403);
    
    struct buf regs_chain = bcreate(fd, (size_t)S * REGCMD_I8_N * 4, 0x403);
    struct buf regs_sep = bcreate(fd, (size_t)S * REGCMD_I8_N * 4, 0x403);
    
    struct buf task_chain = bcreate(fd, (size_t)S * sizeof(struct rknpu_task), 0x40b);
    struct buf task_sep = bcreate(fd, (size_t)S * sizeof(struct rknpu_task), 0x40b);
    
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
    if (rknpu_submit_ioctl(fd, &sub)) {
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
            if (rknpu_submit_ioctl(fd, &sub_s)) {
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
        if (rknpu_submit_ioctl(fd, &sub_c)) {
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
const char *ork_npu_version(void){ return ORK_NPU_VERSION; }

int ork_mm_run_chain_i8(ork_npu *ctx, int S, const ork_mm_task_i8 *tasks) {
    if (S < 1 || S > 64) return -2;
    int fd = ctx->fd;
    ork_npu *c = ctx;
    int CBUF = c->soc->cbuf_elems;

    if (!c->chain_i8) {
        c->chain_i8 = calloc(1, sizeof(struct ork_npu));
        c->chain_i8->regcmd = bcreate(fd, 64 * REGCMD_I8_N * 4, 0x403);
        c->chain_i8->task = bcreate(fd, 64 * sizeof(struct rknpu_task), 0x40b);
    }
    struct ork_npu *ch = c->chain_i8;
    
    int max_K = 0, max_N = 0;
    for (int i = 0; i < S; i++) {
        if (tasks[i].w->K > max_K) max_K = tasks[i].w->K;
        if (tasks[i].w->N > max_N) max_N = tasks[i].w->N;
    }
    
    struct buf chain_A = bcreate(fd, (size_t)S * max_K, 0x403);
    struct buf chain_C = bcreate(fd, (size_t)S * max_N * 4, 0x403);

    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        uint8_t *A_dst = (uint8_t*)chain_A.cpu + (size_t)i * max_K;
        for (int k = 0; k < w->K; k++) A_dst[k] = tasks[i].A[k];
    }
    bsync(fd, &chain_A, RKNPU_MEM_SYNC_TO_DEVICE);

    uint32_t rc[REGCMD_I8_N];
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        uint32_t aA = chain_A.dma + i * max_K;
        uint32_t aC = chain_C.dma + i * max_N * 4;
        synth_i8(rc, 1, w->K, w->N, aA, (uint32_t)w->Bf[0].dma, aC, 1, CBUF, 0);
        
        if (i < S - 1) {
            uint64_t next_dma = ch->regcmd.dma + (i + 1) * REGCMD_I8_N * 4;
            rc[208] = 0x0010 | ((next_dma & 0xffff) << 16);
            rc[209] = (0x0101 << 16) | ((next_dma >> 16) & 0xffff);
            rc[210] = 0x0014 | (0x0037 << 16);
            rc[211] = (0x0101 << 16) | (0);
        } else {
            rc[208] = 0xf3800010;
            rc[209] = 0x0101ffff;
            rc[210] = 0x00370014;
            rc[211] = 0x01010000;
        }
        memcpy((char*)ch->regcmd.cpu + i * REGCMD_I8_N * 4, rc, sizeof(rc));
    }
    bsync(fd, &ch->regcmd, RKNPU_MEM_SYNC_TO_DEVICE);

    struct rknpu_task *t = ch->task.cpu;
    memset(t, 0, S * sizeof(struct rknpu_task));
    for (int i = 0; i < S; i++) {
        t[i].flags = 0;
        t[i].op_idx = 0;
        t[i].enable_mask = 0xd;
        t[i].int_mask = 0x300;
        t[i].int_clear = 0x1ffff;
        t[i].regcfg_amount = 108;
        t[i].regcmd_addr = ch->regcmd.dma + i * REGCMD_I8_N * 4;
    }
    bsync(fd, &ch->task, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);

    struct rknpu_submit sub; memset(&sub, 0, sizeof sub);
    sub.flags = 0x5; sub.task_number = S; sub.task_obj_addr = ch->task.obj; sub.fence_fd = -1;
    sub.core_mask = RKNPU_CORE0_MASK;
    sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, S};

    int ok = 0;
    
    printf("DEBUG: sub.flags=0x%x, task_number=%d, task_obj_addr=0x%lx, core_mask=0x%x\n", sub.flags, sub.task_number, (unsigned long)sub.task_obj_addr, sub.core_mask);
    printf("DEBUG: t[0].regcfg_amount=%d, regcmd_addr=0x%lx\n", t[0].regcfg_amount, (unsigned long)t[0].regcmd_addr); fflush(stdout);
    if (ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub)) {
        printf("ioctl i8 fail errno=%d\n", errno);
        ok = -1;
    }

    if (ok == 0) {
        bsync(fd, &chain_C, RKNPU_MEM_SYNC_FROM_DEVICE);
        for (int i = 0; i < S; i++) {
            memcpy(tasks[i].C, (char*)chain_C.cpu + i * max_N * 4, (size_t)tasks[i].w->N * 4);
        }
    }
    bdestroy(fd, &chain_A); bdestroy(fd, &chain_C);
    return ok;
}

int ork_mm_run_chain_i4(ork_npu *ctx, int S, const ork_mm_task_i4 *tasks) {
    if (S < 1 || S > 64) return -2;
    int fd = ctx->fd;
    ork_npu *c = ctx;

    if (!c->chain_i4) {
        c->chain_i4 = calloc(1, sizeof(struct ork_npu));
        c->chain_i4->regcmd = bcreate(fd, 64 * REGCMD_I4_N * 4, 0x403);
        c->chain_i4->task = bcreate(fd, 64 * sizeof(struct rknpu_task), 0x40b);
    }
    struct ork_npu *ch = c->chain_i4;
    int max_K = 0, max_N = 0;
    for (int i = 0; i < S; i++) {
        if (tasks[i].w->K > max_K) max_K = tasks[i].w->K;
        if (tasks[i].w->N > max_N) max_N = tasks[i].w->N;
    }
    
    struct buf chain_A = bcreate(fd, (size_t)S * max_K, 0x403);
    struct buf chain_C = bcreate(fd, (size_t)S * max_N * 2, 0x403);

    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        uint8_t *A_dst = (uint8_t*)chain_A.cpu + (size_t)i * max_K;
        tile_i4_Aslice(A_dst, tasks[i].A, 0, w->K);
    }
    bsync(fd, &chain_A, RKNPU_MEM_SYNC_TO_DEVICE);

    uint32_t rc[REGCMD_I4_N];
    for (int i = 0; i < S; i++) {
        ork_w *w = tasks[i].w;
        uint32_t aA = chain_A.dma + i * max_K;
        uint32_t aC = chain_C.dma + i * max_N * 2;
        synth_i4(rc, 1, w->K, w->N, aA, (uint32_t)w->Bb[0].dma, aC);
        
        if (i < S - 1) {
            uint64_t next_dma = ch->regcmd.dma + (i + 1) * REGCMD_I4_N * 4;
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
        memcpy((char*)ch->regcmd.cpu + i * REGCMD_I4_N * 4, rc, sizeof(rc));
    }
    bsync(fd, &ch->regcmd, RKNPU_MEM_SYNC_TO_DEVICE);

    struct rknpu_task *t = ch->task.cpu;
    memset(t, 0, S * sizeof(struct rknpu_task));
    for (int i = 0; i < S; i++) {
        t[i].flags = 0;
        t[i].op_idx = 0;
        t[i].enable_mask = 0xd;
        t[i].int_mask = 0x300;
        t[i].int_clear = 0x1ffff;
        t[i].regcfg_amount = 116;
        t[i].regcmd_addr = ch->regcmd.dma + i * REGCMD_I4_N * 4;
    }
    bsync(fd, &ch->task, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE);

    struct rknpu_submit sub; memset(&sub, 0, sizeof sub);
    sub.flags = 0x5; sub.task_number = S; sub.task_obj_addr = ch->task.obj; sub.fence_fd = -1;
    sub.core_mask = RKNPU_CORE0_MASK;
    sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, S};

    int ok = 0;
    
    printf("DEBUG: sub.flags=0x%x, task_number=%d, task_obj_addr=0x%lx, core_mask=0x%x\n", sub.flags, sub.task_number, (unsigned long)sub.task_obj_addr, sub.core_mask);
    printf("DEBUG: t[0].regcfg_amount=%d, regcmd_addr=0x%lx\n", t[0].regcfg_amount, (unsigned long)t[0].regcmd_addr); fflush(stdout);
    if (ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub)) {
        printf("ioctl fail errno=%d\n", errno);
        ok = -1;
    }

    if (ok == 0) {
        bsync(fd, &chain_C, RKNPU_MEM_SYNC_FROM_DEVICE);
        for (int i = 0; i < S; i++) {
            int16_t *o = (int16_t*)((uint8_t*)chain_C.cpu + (size_t)i * max_N * 2);
            int N = tasks[i].w->N;
            for(int nt=0;nt<N/8;nt++)for(int nl=0;nl<8;nl++) tasks[i].C[nt*8+nl] = o[nt*8+nl];
        }
    }
    bdestroy(fd, &chain_A); bdestroy(fd, &chain_C);
    return ok;
}

/* Fast Walsh-Hadamard Transform (FWHT) - Exposed utility function for caller-driven quantization */
void ork_fwht_norm(float *v, int n){
    for(int len=1; len<n; len<<=1)
        for(int i=0;i<n;i+=len<<1)
            for(int j=i;j<i+len;j++){ float a=v[j], b=v[j+len]; v[j]=a+b; v[j+len]=a-b; }
    float s=1.0f/sqrtf((float)n);
    for(int i=0;i<n;i++) v[i]*=s;
}
