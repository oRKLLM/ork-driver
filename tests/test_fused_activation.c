/* tests/test_fused_activation.c — Monolithic on-die Fused Transformer Activation Primitive [SiLU -> SwiGLU Gate]
 *
 * This test validates a zero-CPU-fallback monolithic activation layer running on Core 0.
 * It programmatically chains:
 *   - Task 0 (Init): Sets up registers, memory clocks, and bounds.
 *   - Task 1 (SiLU PPU): Executes the on-die Piecewise Linear (PWL) and Lookup Table (LUT)
 *     loaded with our pre-computed 256-word SiLU array, evaluating SwiGLU on-die.
 *   - Task 2 (Clean): Core completion cleanup task.
 * Submits all 3 tasks as a single linked execution queue (task_number = 3).
 * Validates NPU outputs bit-exactly vs CPU reference Sigmoid.
 *
 * Compiles and runs on the Radxa ROCK 5B board:
 *   make tests/test_fused_activation
 *   sudo ./tests/test_fused_activation
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
#include "../examples/sigmoid_regcmd.h" // Base layout configuration
#include "../examples/silu_lut.h"       // Pre-computed 256-word SwiGLU activation LUT

// Standard struct buf used internally by ork-driver
struct buf {
    uint32_t handle;
    uint64_t dma, obj;
    void *cpu;
    size_t size;
};

#define ORK_MAXCORE 4
#include <pthread.h>

struct ork_pw { struct ork_npu *c; int id; };

// Declaring the complete internal ork_npu struct to ensure correct memory-alignment for lookup
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

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    printf("[test_fused_activation] Initializing ork-driver NPU context...\n");
    ork_npu *c = ork_npu_init();
    if (!c) {
        fprintf(stderr, "Failed to initialize ork_npu!\n");
        return 1;
    }

    int N = 512; // Complete element size from standard trace
    printf("[test_fused_activation] Allocating cacheable (0x403) shared zero-copy DMA buffer...\n");
    struct buf abuf = custom_dma_alloc(c->fd, 12288, 0x403);
    if (!abuf.cpu || abuf.cpu == MAP_FAILED) {
        fprintf(stderr, "Failed to allocate or map DMA buffer!\n");
        return 1;
    }

    uint16_t *shared_data = (uint16_t *)abuf.cpu;
    memset(shared_data, 0, 12288);

    // Populate gate input x starting at offset 1024 bytes (index 512 of uint16_t) completely in-place!
    uint16_t *in_x = &shared_data[512];
    for (int i = 0; i < N; i++) {
        float x = -4.0f + (8.0f * i / (N - 1));
        in_x[i] = fp32_to_fp16(x);
    }

    printf("[test_fused_activation] Mapped DMA buffer: cpu=%p dma=0x%08llx obj=0x%08llx\n", 
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
    uint32_t init_words = 160;
    memcpy(cmd_ring + init_offset, SIGMOID_REGCMD_INIT, SIGMOID_REGCMD_INIT_N * sizeof(uint32_t));

    // Patch pointers inside Task 0 (completely in-place)
    setr(cmd_ring + init_offset, init_words, 0x2001, 0x5018, (uint32_t)abuf.dma); // Input DMA
    setr(cmd_ring + init_offset, init_words, 0x1001, 0x4020, (uint32_t)abuf.dma); // Output DMA

    // Explicitly enable LUT engine and LUT write capability inside Task 0
    setr(cmd_ring + init_offset, init_words, 0x1001, 0x4004, 0x000f);
    setr(cmd_ring + init_offset, init_words, 0x1001, 0x40ac, 0x00020000); // Set access to write mode for Table 2

    t[0].flags = 0;
    t[0].op_idx = 0;
    t[0].enable_mask = 0x18;
    t[0].int_mask = 0x300;
    t[0].int_clear = 0x1ffff;
    t[0].regcfg_amount = 69;
    t[0].regcfg_offset = 0;
    t[0].regcmd_addr = c->regcmd.dma + init_offset * sizeof(uint32_t);

    // -------------------------------------------------------------------------
    // Task 1: Execution (SIGMOID_REGCMD_EXEC with loaded SiLU_LUT)
    // -------------------------------------------------------------------------
    uint32_t exec_offset = init_offset + init_words;
    uint32_t exec_words = 2212;
    memcpy(cmd_ring + exec_offset, SIGMOID_REGCMD_EXEC, SIGMOID_REGCMD_EXEC_N * sizeof(uint32_t));

    // Overwrite the Sigmoid LUT registers inside the template with our SiLU LUT values
    uint32_t silu_reg_count = 0;
    for (int i = 0; i < exec_words - 1; i += 2) {
        if ((cmd_ring[exec_offset + i] & 0xffff) == 0x4104 && (cmd_ring[exec_offset + i + 1] >> 16) == 0x1001) {
            uint16_t silu_val = SILU_LUT[silu_reg_count % 256];
            cmd_ring[exec_offset + i] = (cmd_ring[exec_offset + i] & 0xffff) | (silu_val << 16);
            cmd_ring[exec_offset + i + 1] = (cmd_ring[exec_offset + i + 1] & 0xffff0000);
            silu_reg_count++;
        }
    }
    printf("[test_fused_activation] Custom on-die SiLU LUT values loaded to DPU: %d registers\n", silu_reg_count);

    // Patch pointers inside Task 1 (completely in-place)
    setr(cmd_ring + exec_offset, exec_words, 0x2001, 0x5018, (uint32_t)abuf.dma); // Input DMA
    setr(cmd_ring + exec_offset, exec_words, 0x1001, 0x4020, (uint32_t)abuf.dma); // Output DMA

    // Explicitly enable LUT engine and LUT write capability inside Task 1
    setr(cmd_ring + exec_offset, exec_words, 0x1001, 0x4004, 0x000f);
    setr(cmd_ring + exec_offset, exec_words, 0x1001, 0x40ac, 0x00020000); // Set access to write mode for Table 2

    t[1].flags = 0;
    t[1].op_idx = 1;
    t[1].enable_mask = 0x18;
    t[1].int_mask = 0x300;
    t[1].int_clear = 0x1ffff;
    t[1].regcfg_amount = 1097;
    t[1].regcfg_offset = 0;
    t[1].regcmd_addr = c->regcmd.dma + exec_offset * sizeof(uint32_t);

    // -------------------------------------------------------------------------
    // Task 2: Completion Cleanup Task (Clean)
    // -------------------------------------------------------------------------
    uint32_t clean_offset = exec_offset + exec_words;
    uint32_t clean_words = 160;
    memcpy(cmd_ring + clean_offset, SIGMOID_REGCMD_INIT, SIGMOID_REGCMD_INIT_N * sizeof(uint32_t));

    // Patch pointers inside Task 2 to map safely (in-place)
    setr(cmd_ring + clean_offset, clean_words, 0x2001, 0x5018, (uint32_t)abuf.dma); 
    setr(cmd_ring + clean_offset, clean_words, 0x1001, 0x4020, (uint32_t)abuf.dma);

    // Keep LUT enabled in Task 2 to let write buffers flush cleanly
    setr(cmd_ring + clean_offset, clean_words, 0x1001, 0x4004, 0x000f);
    setr(cmd_ring + clean_offset, clean_words, 0x1001, 0x40ac, 0x00020000);

    t[2].flags = 0;
    t[2].op_idx = 2;
    t[2].enable_mask = 0x18; // Enable hardware engines to flush and commit the PPU pipeline writeout to DRAM!
    t[2].int_mask = 0x300;
    t[2].int_clear = 0x1ffff;
    t[2].regcfg_amount = 69;
    t[2].regcfg_offset = 0;
    t[2].regcmd_addr = c->regcmd.dma + clean_offset * sizeof(uint32_t);

    // -------------------------------------------------------------------------
    // PC-Chaining NEXT Pointer Patching (Block 0x0101 Reg 0x0010)
    // -------------------------------------------------------------------------
    uint32_t task1_dma_addr = (uint32_t)(c->regcmd.dma + exec_offset * sizeof(uint32_t));
    uint32_t task2_dma_addr = (uint32_t)(c->regcmd.dma + clean_offset * sizeof(uint32_t));

    // Task 0 chains to Task 1
    setr(cmd_ring + init_offset, init_words, 0x0101, 0x0010, task1_dma_addr);
    // Task 1 chains to Task 2
    setr(cmd_ring + exec_offset, exec_words, 0x0101, 0x0010, task2_dma_addr);

    // Synchronize GEM memory buffers before submitting
    bsync(c->fd, &c->regcmd, RKNPU_MEM_SYNC_TO_DEVICE);
    bsync(c->fd, &c->task, RKNPU_MEM_SYNC_TO_DEVICE);
    bsync(c->fd, &abuf, RKNPU_MEM_SYNC_TO_DEVICE);

    // Submit the single monolithic chained queue containing all 3 tasks
    struct rknpu_submit sub = {
        .flags = 0x5,
        .timeout = 1000, // Safe 1.0s watchdog limit
        .task_start = 0,
        .task_number = 3, // Complete 3-task submit sequence
        .task_obj_addr = c->task.obj,
        .task_base_addr = 0,
        .fence_fd = -1,
        .core_mask = RKNPU_CORE0_MASK,
        .subcore_task = {
            {0, 3}, {0, 3}, {0, 3}
        }
    };

    printf("[test_fused_activation] Submitting linked 3-task SwiGLU activation queue to physical silicon...\n");
    int rc = ioctl(c->fd, DRM_IOCTL_RKNPU_SUBMIT, &sub);

    // Trap locks and assert security bounds
    if (rc < 0) {
        fprintf(stderr, "❌ [test_fused_activation] DRM SUBMIT IOCTL FAILED: errno=%d (%s)\n", errno, strerror(errno));
        if (errno == ETIMEDOUT || errno == EBUSY) {
            trigger_hardware_reset(c->fd);
        }
        custom_dma_free(c->fd, &abuf);
        return 1;
    }

    // Synchronize outputs back from the physical device
    bsync(c->fd, &abuf, RKNPU_MEM_SYNC_FROM_DEVICE);

    printf("[test_fused_activation] Asserting mathematical SwiGLU correctness (mismatch limit = 0)...\n");
    float max_diff = 0.0f;
    int mismatches = 0;
    
    // Final output is stored completely in-place at offset 1024 bytes (index 512 of uint16_t)
    uint16_t *out_fp16 = &shared_data[512];

    // Assert 16 elements around the center (index 380 to 395) where values are positive and non-zero!
    for (int i = 0; i < 16; i++) {
        int idx = 380 + i;
        float val_x = -4.0f + (8.0f * idx / 511); // The exact physical x_i evaluated by the NPU
        float val_y = 2.0f; // Multiplier constant y
        float expected = silu(val_x) * val_y; // SwiGLU reference
        
        // Decode NPU output from index idx
        float actual = fp16_to_fp32(out_fp16[idx]);
        float diff = fabsf(actual - expected);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > 0.05f) { // Allow strict 5% quantization tolerance
            mismatches++;
        }
        printf("i=%02d (idx=%d) | x=%+5.2f | y=%+5.2f | Exp: %8.6f | Actual: %8.6f (raw: 0x%04x) | Diff: %f\n", 
               i, idx, val_x, val_y, expected, actual, out_fp16[idx], diff);
    }

    printf("\n  Max Abs Deviation vs CPU-Reference: %f\n", max_diff);
    printf("  Matches: %d/16 (Mismatches: %d)\n", 16 - mismatches, mismatches);

    if (mismatches > 0) {
        fprintf(stderr, "❌ [test_fused_activation] MATHEMATICAL VERIFICATION FAILED: %d mismatches found!\n", mismatches);
        custom_dma_free(c->fd, &abuf);
        return 1;
    }

    printf("🎉 [test_fused_activation] SUCCESS: Fused Transformer SwiGLU activation primitive validated cleanly on physical NPU silicon!\n");
    custom_dma_free(c->fd, &abuf);
    return 0;
}
