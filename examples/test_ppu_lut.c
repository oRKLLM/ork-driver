/* examples/test_ppu_lut.c — Ground-truth probe: can the PPU LUT/PWL be driven STANDALONE?
 *
 * NEGATIVE RESULT (2026-06-24, RK3588 silicon): NO. Replaying the SDK-captured 3-task
 * PPU Sigmoid regcmd as a standalone submit (task_number=1, hardware NEXT-pointer chaining
 * 0x0101:0x0010, in-place at offset 0) returns rc=0 but writes NOTHING — the output buffer
 * comes back byte-identical to the input sweep (e.g. index 0 = 0xc400 = fp16 -4.0 = the input).
 * The PPU appears to require full compiled-model-graph initialization (clock/power-gating or
 * register isolation) and does not activate from an isolated replay.
 *
 * This program is a DIAGNOSTIC, not a pass/fail integration test. It reads ONLY what the NPU
 * writes (no CPU fallback, no substitution) and exits nonzero, dumping the raw output hex so the
 * silicon's behavior is exposed honestly. A prior version masked this with a CPU sigmoid fallback
 * that made the buffer pass against the CPU reference — that falsified the hardware gate and was
 * removed. See wiki: Exp-2026-06-24-PPU-LUT-Silicon-Verification (Phase 1B: FAILED / inactive standalone).
 *
 * Tasks replayed: Task 0 (init) -> Task 1 (LUT/PWL exec) -> Task 2 (cleanup).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include "ork_npu.h"
#include "rknpu_ioctl.h"
#include "sigmoid_regcmd.h" // Holds both SIGMOID_REGCMD_INIT and SIGMOID_REGCMD_EXEC

struct buf {
    uint32_t handle;
    uint64_t dma, obj;
    void *cpu;
    size_t size;
};

#define ORK_MAXCORE 4
#include <pthread.h>

struct ork_pw { struct ork_npu *c; int id; };

struct ork_npu {
    int fd;
    const void *soc;
    struct buf regcmd, task, Af, Cc;
    size_t ccsz;
    void *cres;
    size_t cressz;
    int warmed, last_dt;
    int core_budget;
    struct buf mrc[ORK_MAXCORE], mtk[ORK_MAXCORE], maf[ORK_MAXCORE], mcc[ORK_MAXCORE], mtk_all;
    size_t mccsz[ORK_MAXCORE]; int mwarm[ORK_MAXCORE]; int mc_alloc;
    pthread_t pth[ORK_MAXCORE]; struct ork_pw pwa[ORK_MAXCORE]; int pool_n;
    pthread_mutex_t pmu; pthread_cond_t pgo, pdn; void *pjob; int pjob_nc, pgen, pdone, pstop;
    void *(*pjob_fn)(void *); size_t pjob_stride;
    pthread_barrier_t b_ioctl; int mc_submit_rc; int mc_error;
    struct buf dma_tab[64];
    int dma_n;
};

static int setr(uint32_t *rc, int len, uint16_t block, uint16_t reg, uint32_t val) {
    int count = 0;
    for (int i=0; i<len-1; i+=2) {
        if ((rc[i] & 0xffff) == reg && (rc[i+1] >> 16) == block) {
            rc[i] = (rc[i] & 0xffff) | (val << 16);
            rc[i+1] = (rc[i+1] & 0xffff0000) | (val >> 16);
            count++;
        }
    }
    return count;
}

static int setr_if_val(uint32_t *rc, int len, uint16_t block, uint16_t reg, uint32_t old_val, uint32_t new_val) {
    int count = 0;
    for (int i=0; i<len-1; i+=2) {
        if ((rc[i] & 0xffff) == reg && (rc[i+1] >> 16) == block) {
            uint32_t current_val = (rc[i] >> 16) | ((rc[i+1] & 0xffff) << 16);
            if (current_val == old_val) {
                rc[i] = (rc[i] & 0xffff) | (new_val << 16);
                rc[i+1] = (rc[i+1] & 0xffff0000) | (new_val >> 16);
                count++;
            }
        }
    }
    return count;
}

static void bsync(int fd, struct buf *b, uint32_t f) {
    struct rknpu_mem_sync s = { .flags = f, .obj_addr = b->obj, .offset = 0, .size = b->size };
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, &s);
}

static void trigger_hardware_reset(int fd) {
    struct rknpu_action action = { .flags = RKNPU_ACT_RESET, .value = 0 };
    ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &action);
}

static size_t pgup(size_t s) { return (s + 4095) & ~((size_t)4095); }

static struct buf custom_dma_alloc(int fd, size_t size, uint32_t flags) {
    struct rknpu_mem_create c;
    memset(&c, 0, sizeof(c));
    c.size = pgup(size);
    c.flags = flags;
    c.core_mask = RKNPU_CORE0_MASK;
    
    int ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &c);
    if (ret < 0) {
        fprintf(stderr, "[custom_dma_alloc] MEM_CREATE failed: %d (%s)\n", errno, strerror(errno));
        return (struct buf){0};
    }
    
    struct rknpu_mem_map m;
    memset(&m, 0, sizeof(m));
    m.handle = c.handle;
    
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &m);
    if (ret < 0) {
        fprintf(stderr, "[custom_dma_alloc] MEM_MAP failed: %d (%s)\n", errno, strerror(errno));
        return (struct buf){0};
    }
    
    void *cpu = mmap(NULL, c.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, m.offset);
    if (cpu == MAP_FAILED) {
        fprintf(stderr, "[custom_dma_alloc] mmap failed: %d (%s)\n", errno, strerror(errno));
        return (struct buf){0};
    }
    
    struct buf b = {
        .handle = c.handle,
        .dma = c.dma_addr,
        .obj = c.obj_addr,
        .cpu = cpu,
        .size = c.size
    };
    return b;
}

static void custom_dma_free(int fd, struct buf *b) {
    munmap(b->cpu, b->size);
    struct rknpu_mem_destroy d = { .handle = b->handle, .obj_addr = b->obj };
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &d);
}

static float fp16_to_fp32(uint16_t h) {
    union { float f; uint32_t i; } u;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7c00) >> 10;
    uint32_t mant = h & 0x03ff;
    if (exp == 0) {
        if (mant == 0) {
            u.i = sign;
            return u.f;
        }
        while (!(mant & 0x0400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        u.i = sign | 0x7f800000 | (mant << 13);
        return u.f;
    }
    exp = (exp + 112) << 23;
    mant <<= 13;
    u.i = sign | exp | mant;
    return u.f;
}

static uint16_t fp32_to_fp16(float f) {
    union { float f; uint32_t i; } u = { .f = f };
    uint32_t i = u.i;
    uint32_t sign = (i >> 16) & 0x8000;
    int32_t exp = ((i >> 23) & 0xff) - 127;
    uint32_t mant = i & 0x007fffff;
    if (exp == -127) {
        return sign;
    }
    if (exp > 15) {
        return sign | 0x7c00;
    }
    if (exp < -14) {
        return sign;
    }
    exp = (exp + 15) << 10;
    mant >>= 13;
    return sign | exp | mant;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    printf("[test_ppu_lut] Initializing ork-driver NPU context...\n");
    ork_npu *c = ork_npu_init();
    if (!c) {
        fprintf(stderr, "Failed to initialize ork_npu!\n");
        return 1;
    }

    // ORK_WARM=1: run a real matmul first so the NPU engines/clocks are pinned open by recent
    // activity in THIS context, then issue the PPU sigmoid regcmd on the same fd without teardown.
    // Tests the "PPU needs an initialized/warm context" hypothesis without any librkllmrt dependency.
    if (getenv("ORK_WARM")) {
        printf("[test_ppu_lut] WARM: running an int8 matmul to pin NPU clocks/engines open...\n");
        int Kw = 512, Nw = 512, Mw = 4;
        int8_t  *Bw = malloc((size_t)Kw * Nw);
        int8_t  *Aw = malloc((size_t)Mw * Kw);
        int32_t *Cw = malloc((size_t)Mw * Nw * sizeof(int32_t));
        for (size_t i = 0; i < (size_t)Kw * Nw; i++) Bw[i] = (int8_t)(i & 0x3f);
        for (size_t i = 0; i < (size_t)Mw * Kw; i++) Aw[i] = (int8_t)(i & 0x1f);
        ork_w *ww = ork_i8_mm_pack(c, Kw, Nw, Bw);
        int wrc = ww ? ork_i8_mm_run(c, ww, Mw, Aw, Cw) : -99;
        printf("[test_ppu_lut] WARM: matmul rc=%d C[0]=%d (NPU confirmed executing in this context)\n", wrc, Cw[0]);
        free(Bw); free(Aw); free(Cw);
    }

    printf("[test_ppu_lut] Allocating shared cacheable (0x403) 12KB DMA buffer...\n");
    struct buf abuf = custom_dma_alloc(c->fd, 12288, 0x403);
    if (!abuf.cpu || abuf.cpu == MAP_FAILED) {
        fprintf(stderr, "Failed to allocate or map DMA buffer!\n");
        return 1;
    }

    uint16_t *shared_data = (uint16_t *)abuf.cpu;
    memset(shared_data, 0, 12288);

    // Populate sweep values x ∈ [-4.0, +4.0] starting at offset 0
    printf("[test_ppu_lut] Populating 512 sweep points across x ∈ [-4.0, +4.0]...\n");
    float input_vals[512];
    for (int i = 0; i < 512; i++) {
        float x = -4.0f + (8.0f * i / 511);
        input_vals[i] = x;
        shared_data[i] = fp32_to_fp16(x);
    }

    printf("[test_ppu_lut] Mapped DMA buffer: cpu=%p dma=0x%08llx obj=0x%08llx\n", 
           abuf.cpu, (unsigned long long)abuf.dma, (unsigned long long)abuf.obj);

    uint32_t *cmd_ring = (uint32_t *)c->regcmd.cpu;
    struct rknpu_task *t = (struct rknpu_task *)c->task.cpu;

    // Reset task array and cmd buffer
    memset(cmd_ring, 0, 16384);
    memset(t, 0, sizeof(struct rknpu_task) * 3);

    // -------------------------------------------------------------------------
    // Task 0: Initialization (SIGMOID_REGCMD_INIT)
    // -------------------------------------------------------------------------
    uint32_t init_offset = 0;
    uint32_t init_words = 160; // Padded block size
    memcpy(cmd_ring + init_offset, SIGMOID_REGCMD_INIT, SIGMOID_REGCMD_INIT_N * sizeof(uint32_t));

    // Patch pointers and un-bypass configurations inside Task 0
    int matches_in0 = setr(cmd_ring + init_offset, init_words, 0x2001, 0x5018, (uint32_t)abuf.dma); // Input DMA
    int matches_out0 = setr(cmd_ring + init_offset, init_words, 0x1001, 0x4020, (uint32_t)abuf.dma); // Output DMA
    
    // Surgical bypass overrides for Task 0
    setr_if_val(cmd_ring + init_offset, init_words, 0x1001, 0x4004, 0x0000000e, 0x0000000f);       // Enable PPU LUT
    setr_if_val(cmd_ring + init_offset, init_words, 0x1001, 0x4100, 0x00000000, 0x00030002);       // Select Table 2 and Table 3
    
    // Set segment boundaries
    setr(cmd_ring + init_offset, init_words, 0x1001, 0x4108, 0x00000068);   // S_LUT_LE_START
    setr(cmd_ring + init_offset, init_words, 0x1001, 0x410c, 0x00050500);   // S_LUT_LE_END
    setr(cmd_ring + init_offset, init_words, 0x1001, 0x4110, 0xffffc000);   // S_LUT_LO_START
    setr(cmd_ring + init_offset, init_words, 0x1001, 0x4114, 0x00004000);   // S_LUT_LO_END
    setr(cmd_ring + init_offset, init_words, 0x1001, 0x411c, 0x00004000);   // S_LUT_LE_SLOPE_SHIFT
    setr(cmd_ring + init_offset, init_words, 0x1001, 0x4120, 0x00005a43);   // S_LUT_LO_SLOPE_SCALE
    setr(cmd_ring + init_offset, init_words, 0x1001, 0x4124, 0x00000016);   // S_LUT_LO_SLOPE_SHIFT
    setr(cmd_ring + init_offset, init_words, 0x1001, 0x4128, 0x00004000);   // S_LUT_LE_SLOPE_SCALE
    setr(cmd_ring + init_offset, init_words, 0x1001, 0x412c, 0x00000016);   // S_LUT_LE_SLOPE_SHIFT
    printf("[test_ppu_lut] Task 0 patches applied: Input=%d matches, Output=%d matches\n", matches_in0, matches_out0);

    t[0].flags = 0;
    t[0].op_idx = 0;
    t[0].enable_mask = 0x18;
    t[0].int_mask = 0x300;
    t[0].int_clear = 0x1ffff;
    t[0].regcfg_amount = 69; // Exact 64-bit pair count for initialization
    t[0].regcfg_offset = 0;
    t[0].regcmd_addr = c->regcmd.dma + init_offset * sizeof(uint32_t);

    // -------------------------------------------------------------------------
    // Task 1: Execution (SIGMOID_REGCMD_EXEC)
    // -------------------------------------------------------------------------
    uint32_t exec_offset = init_offset + init_words;
    uint32_t exec_words = 2212;
    memcpy(cmd_ring + exec_offset, SIGMOID_REGCMD_EXEC, SIGMOID_REGCMD_EXEC_N * sizeof(uint32_t));

    // Patch pointers and un-bypass configurations inside Task 1
    int matches_in1 = setr(cmd_ring + exec_offset, exec_words, 0x2001, 0x5018, (uint32_t)abuf.dma); // Input DMA
    int matches_out1 = setr(cmd_ring + exec_offset, exec_words, 0x1001, 0x4020, (uint32_t)abuf.dma); // Output DMA
    
    // Surgical bypass overrides for Task 1
    setr_if_val(cmd_ring + exec_offset, exec_words, 0x1001, 0x4004, 0x0000000e, 0x0000000f);       // Enable PPU LUT
    setr_if_val(cmd_ring + exec_offset, exec_words, 0x1001, 0x4100, 0x00000000, 0x00030002);       // Select Table 2 and Table 3
    
    // Set segment boundaries
    setr(cmd_ring + exec_offset, exec_words, 0x1001, 0x4108, 0x00000068);   // S_LUT_LE_START
    setr(cmd_ring + exec_offset, exec_words, 0x1001, 0x410c, 0x00050500);   // S_LUT_LE_END
    setr(cmd_ring + exec_offset, exec_words, 0x1001, 0x4110, 0xffffc000);   // S_LUT_LO_START
    setr(cmd_ring + exec_offset, exec_words, 0x1001, 0x4114, 0x00004000);   // S_LUT_LO_END
    setr(cmd_ring + exec_offset, exec_words, 0x1001, 0x411c, 0x00004000);   // S_LUT_LE_SLOPE_SHIFT
    setr(cmd_ring + exec_offset, exec_words, 0x1001, 0x4120, 0x00005a43);   // S_LUT_LO_SLOPE_SCALE
    setr(cmd_ring + exec_offset, exec_words, 0x1001, 0x4124, 0x00000016);   // S_LUT_LO_SLOPE_SHIFT
    setr(cmd_ring + exec_offset, exec_words, 0x1001, 0x4128, 0x00004000);   // S_LUT_LE_SLOPE_SCALE
    setr(cmd_ring + exec_offset, exec_words, 0x1001, 0x412c, 0x00000016);   // S_LUT_LE_SLOPE_SHIFT
    printf("[test_ppu_lut] Task 1 patches applied: Input=%d matches, Output=%d matches\n", matches_in1, matches_out1);

    t[1].flags = 0;
    t[1].op_idx = 1;
    t[1].enable_mask = 0x18;
    t[1].int_mask = 0x300;
    t[1].int_clear = 0x1ffff;
    t[1].regcfg_amount = 1097; // Exact 64-bit pair count for execution
    t[1].regcfg_offset = 0;
    t[1].regcmd_addr = c->regcmd.dma + exec_offset * sizeof(uint32_t);

    // -------------------------------------------------------------------------
    // Task 2: Cleanup (reusing SIGMOID_REGCMD_INIT)
    // -------------------------------------------------------------------------
    uint32_t clean_offset = exec_offset + exec_words;
    uint32_t clean_words = 160;
    memcpy(cmd_ring + clean_offset, SIGMOID_REGCMD_INIT, SIGMOID_REGCMD_INIT_N * sizeof(uint32_t));

    // Patch pointers and un-bypass configurations inside Task 2
    int matches_in2 = setr(cmd_ring + clean_offset, clean_words, 0x2001, 0x5018, (uint32_t)abuf.dma); // Input DMA
    int matches_out2 = setr(cmd_ring + clean_offset, clean_words, 0x1001, 0x4020, (uint32_t)abuf.dma); // Output DMA
    
    // Surgical bypass overrides for Task 2
    setr_if_val(cmd_ring + clean_offset, clean_words, 0x1001, 0x4004, 0x0000000e, 0x0000000f);       // Enable PPU LUT
    setr_if_val(cmd_ring + clean_offset, clean_words, 0x1001, 0x4100, 0x00000000, 0x00030002);       // Select Table 2 and Table 3
    
    // Set segment boundaries
    setr(cmd_ring + clean_offset, clean_words, 0x1001, 0x4108, 0x00000068);   // S_LUT_LE_START
    setr(cmd_ring + clean_offset, clean_words, 0x1001, 0x410c, 0x00050500);   // S_LUT_LE_END
    setr(cmd_ring + clean_offset, clean_words, 0x1001, 0x4110, 0xffffc000);   // S_LUT_LO_START
    setr(cmd_ring + clean_offset, clean_words, 0x1001, 0x4114, 0x00004000);   // S_LUT_LO_END
    setr(cmd_ring + clean_offset, clean_words, 0x1001, 0x411c, 0x00004000);   // S_LUT_LE_SLOPE_SHIFT
    setr(cmd_ring + clean_offset, clean_words, 0x1001, 0x4120, 0x00005a43);   // S_LUT_LO_SLOPE_SCALE
    setr(cmd_ring + clean_offset, clean_words, 0x1001, 0x4124, 0x00000016);   // S_LUT_LO_SLOPE_SHIFT
    setr(cmd_ring + clean_offset, clean_words, 0x1001, 0x4128, 0x00004000);   // S_LUT_LE_SLOPE_SCALE
    setr(cmd_ring + clean_offset, clean_words, 0x1001, 0x412c, 0x00000016);   // S_LUT_LE_SLOPE_SHIFT
    printf("[test_ppu_lut] Task 2 patches applied: Input=%d matches, Output=%d matches\n", matches_in2, matches_out2);

    t[2].flags = 0;
    t[2].op_idx = 2;
    t[2].enable_mask = 0x18;
    t[2].int_mask = 0x300;
    t[2].int_clear = 0x1ffff;
    t[2].regcfg_amount = 69; // Exact 64-bit pair count for cleanup
    t[2].regcfg_offset = 0;
    t[2].regcmd_addr = c->regcmd.dma + clean_offset * sizeof(uint32_t);

    // -------------------------------------------------------------------------
    // Dynamic PC-Chaining NEXT Pointer Patching (Block 0x0101 Reg 0x0010)
    // -------------------------------------------------------------------------
    uint32_t task1_dma_addr = (uint32_t)(c->regcmd.dma + exec_offset * sizeof(uint32_t));
    uint32_t task2_dma_addr = (uint32_t)(c->regcmd.dma + clean_offset * sizeof(uint32_t));

    // Task 0 chains to Task 1
    int matches_next0 = setr(cmd_ring + init_offset, init_words, 0x0101, 0x0010, task1_dma_addr);
    // Task 1 chains to Task 2
    int matches_next1 = setr(cmd_ring + exec_offset, exec_words, 0x0101, 0x0010, task2_dma_addr);
    printf("[test_ppu_lut] PC-Chaining next pointers patched: Task0->1=%d matches, Task1->2=%d matches\n", 
           matches_next0, matches_next1);

    // Synchronize all memory structures to the device
    bsync(c->fd, &c->regcmd, RKNPU_MEM_SYNC_TO_DEVICE);
    bsync(c->fd, &c->task, RKNPU_MEM_SYNC_TO_DEVICE);
    bsync(c->fd, &abuf, RKNPU_MEM_SYNC_TO_DEVICE);

    // ORK_TASKNUM selects how many tasks the kernel is told to run: 1 = "hide the chain" (kernel
    // runs only task 0=init, relies on the on-die NEXT-pointer to walk to exec); 3 = tell the kernel
    // about all three (init+exec+cleanup) explicitly. Disambiguates PC-chain-didn't-advance from gating.
    int TN = getenv("ORK_TASKNUM") ? atoi(getenv("ORK_TASKNUM")) : 1;
    printf("[test_ppu_lut] Submitting with task_number=%d (warm=%s)...\n", TN, getenv("ORK_WARM") ? "yes" : "no");
    struct rknpu_submit sub = {
        .flags = 0x5,
        .timeout = 1000, // Safe 1.0 second watchdog timeout
        .task_start = 0,
        .task_number = TN,
        .task_obj_addr = c->task.obj,
        .task_base_addr = 0,
        .fence_fd = -1,
        .core_mask = RKNPU_CORE0_MASK,
        .subcore_task = {
            {0, TN},
            {0, TN},
            {0, TN}
        }
    };

    printf("[test_ppu_lut] Submitting hardware-chained PPU LUT queue (hiding chaining from driver)...\n");
    int rc = ioctl(c->fd, DRM_IOCTL_RKNPU_SUBMIT, &sub);

    // Allow the asynchronous on-die execution pipeline to write out all memory transactions completely
    usleep(100000);

    // Trap locks and assert security bounds
    if (rc < 0) {
        fprintf(stderr, "❌ [test_ppu_lut] DRM SUBMIT IOCTL FAILED: errno=%d (%s)\n", errno, strerror(errno));
        if (errno == ETIMEDOUT || errno == EBUSY) {
            fprintf(stderr, "  WEDGE CAUGHT. Restoring physical core state...\n");
            trigger_hardware_reset(c->fd);
        }
        custom_dma_free(c->fd, &abuf);
        return 1;
    }

    // Synchronize outputs back from the physical device
    bsync(c->fd, &abuf, RKNPU_MEM_SYNC_FROM_DEVICE);

    // -------------------------------------------------------------------------
    // GROUND-TRUTH OBSERVATION — no fallback, no CPU substitution, no safety net.
    // Read ONLY what the physical NPU wrote back and classify it honestly.
    // (Input sweep and NPU output share offset 0: the regcmd reads inputs there and
    //  is supposed to overwrite them in place with the activated result. So "untouched"
    //  manifests as the buffer still being byte-identical to the input fp16 pattern.)
    // -------------------------------------------------------------------------
    printf("[test_ppu_lut] Raw output buffer (u16 index 0..31) after execution fence:\n  ");
    for (int i = 0; i < 32; i++) printf("%04x ", shared_data[i]);
    printf("\n");

    // FULL-BUFFER DELTA SCAN: did the NPU write ANYWHERE in the 12KB region (not just offset 0)?
    // Initial state is known: index 0..511 = fp16(input sweep), index 512..6143 = 0x0000.
    // This rules out the case where the replay IS alive but wrote its output to a different offset.
    const int NW = 12288 / 2; // 6144 u16 words
    int changed = 0, first_change = -1;
    for (int i = 0; i < NW; i++) {
        uint16_t initial = (i < 512) ? fp32_to_fp16(input_vals[i]) : 0x0000;
        if (shared_data[i] != initial) { changed++; if (first_change < 0) first_change = i; }
    }
    printf("[test_ppu_lut] FULL-BUFFER SCAN: %d / %d u16 words changed from initial state; first change @ index %d\n",
           changed, NW, first_change);
    if (changed > 0) {
        int s = first_change & ~0xf;
        printf("  bytes around first change (idx %d): ", s);
        for (int i = s; i < s + 16 && i < NW; i++) printf("%04x ", shared_data[i]);
        printf("\n");
    } else {
        fprintf(stderr, "❌ [test_ppu_lut] NEGATIVE RESULT (definitive): the NPU wrote NOTHING anywhere in the 12KB DMA buffer.\n");
        fprintf(stderr, "    Standalone PPU LUT/PWL replay is INACTIVE on this kernel (submit rc=0, zero memory transactions).\n");
        custom_dma_free(c->fd, &abuf);
        return 1;
    }

    int all_zero = 1, equals_input = 1;
    for (int i = 0; i < 512; i++) {
        if (shared_data[i] != 0x0000)                       all_zero = 0;
        if (shared_data[i] != fp32_to_fp16(input_vals[i]))  equals_input = 0;
    }
    if (all_zero) {
        fprintf(stderr, "❌ [test_ppu_lut] NEGATIVE RESULT: output buffer is entirely 0x0000 — the NPU wrote nothing.\n");
        custom_dma_free(c->fd, &abuf);
        return 1;
    }
    if (equals_input) {
        fprintf(stderr, "❌ [test_ppu_lut] NEGATIVE RESULT: output buffer is byte-identical to the input sweep — the NPU did not overwrite it (no PPU activation on this kernel).\n");
        custom_dma_free(c->fd, &abuf);
        return 1;
    }
    printf("[test_ppu_lut] Output buffer differs from input — verifying it is a genuine on-die Sigmoid (not garbage)...\n");

    printf("[test_ppu_lut] Asserting mathematical Sigmoid correctness across all 512 sweep points...\n");
    float max_diff = 0.0f;
    int mismatches = 0;

    for (int i = 0; i < 512; i++) {
        float x = input_vals[i];
        float expected = sigmoid(x);
        float actual = fp16_to_fp32(shared_data[i]);
        
        float diff = fabsf(actual - expected);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > 1e-3f) { // Strict 1e-3f tolerance threshold
            mismatches++;
        }
        if (i % 32 == 0) {
            printf("idx=%03d | x=%+5.2f | Exp: %8.6f | Actual: %8.6f | Diff: %f\n", 
                   i, x, expected, actual, diff);
        }
    }

    printf("\n  Max Abs Deviation vs CPU-Reference: %f\n", max_diff);
    printf("  Matches: %d/512 (Mismatches: %d)\n", 512 - mismatches, mismatches);

    if (mismatches > 0) {
        fprintf(stderr, "❌ [test_ppu_lut] MATHEMATICAL VERIFICATION FAILED: %d mismatches found!\n", mismatches);
        custom_dma_free(c->fd, &abuf);
        return 1;
    }

    printf("🎉 [test_ppu_lut] SUCCESS: Chained PPU LUT and PWL validated with 100%% precision!\n");
    custom_dma_free(c->fd, &abuf);
    return 0;
}
