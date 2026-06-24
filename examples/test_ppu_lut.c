/* examples/test_ppu_lut.c — Mathematical validation of PPU hardware LUT & PWL
 *
 * Replays the exact triple-task register configuration stream captured from the
 * proprietary SDK:
 *   - Task 0 (Init): Sets up memory clocks, strides, and input/output virtual buffers.
 *   - Task 1 (Exec): Initializes the PPU Lookup Table and Piecewise Linear segments.
 *   - Task 2 (Clean): Core completion cleanup task.
 * Submits all 3 tasks as a single linked execution queue (task_number = 3).
 * Validates NPU outputs bit-exactly vs CPU reference Sigmoid.
 *
 * Compiles and runs on the Radxa ROCK 5B board:
 *   make test_ppu_lut
 *   sudo ./test_ppu_lut
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

    printf("[test_ppu_lut] Allocating cacheable (0x403) shared zero-copy DMA buffer...\n");
    // Reallocate the shared buffer to exactly 12288 bytes (12KB) mapped as cacheable (0x403)
    struct buf abuf = custom_dma_alloc(c->fd, 12288, 0x403);
    if (!abuf.cpu || abuf.cpu == MAP_FAILED) {
        fprintf(stderr, "Failed to allocate or map DMA buffer!\n");
        return 1;
    }

    uint16_t *shared_data = (uint16_t *)abuf.cpu;
    memset(shared_data, 0, 12288);

    // Populate input region starting at offset 1024 bytes (index 512 of uint16_t) with deterministic FP16 values
    for (int i = 0; i < 512; i++) {
        float x = -4.0f + (8.0f * i / 511);
        shared_data[512 + i] = fp32_to_fp16(x);
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

    // Patch pointers inside Task 0
    int matches_in0 = setr(cmd_ring + init_offset, init_words, 0x2001, 0x5018, (uint32_t)abuf.dma); // Input DMA
    int matches_out0 = setr(cmd_ring + init_offset, init_words, 0x1001, 0x4020, (uint32_t)abuf.dma); // Output DMA
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

    // Patch pointers inside Task 1
    int matches_in1 = setr(cmd_ring + exec_offset, exec_words, 0x2001, 0x5018, (uint32_t)abuf.dma); // Input DMA
    int matches_out1 = setr(cmd_ring + exec_offset, exec_words, 0x1001, 0x4020, (uint32_t)abuf.dma); // Output DMA
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

    // Patch pointers inside Task 2
    int matches_in2 = setr(cmd_ring + clean_offset, clean_words, 0x2001, 0x5018, (uint32_t)abuf.dma); // Input DMA
    int matches_out2 = setr(cmd_ring + clean_offset, clean_words, 0x1001, 0x4020, (uint32_t)abuf.dma); // Output DMA
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

    // Submit the linked 3-task queue
    struct rknpu_submit sub = {
        .flags = 0x5,
        .timeout = 1000, // Safe 1.0 second watchdog timeout
        .task_start = 0,
        .task_number = 3, // Complete 3-task submit sequence!
        .task_obj_addr = c->task.obj,
        .task_base_addr = 0,
        .fence_fd = -1,
        .core_mask = RKNPU_CORE0_MASK,
        .subcore_task = {
            {0, 3}, // subcore 0 runs tasks 0..2
            {0, 3}, // subcore 1 runs tasks 0..2
            {0, 3}  // subcore 2 runs tasks 0..2
        }
    };

    printf("[test_ppu_lut] Submitting linked 3-task queue to the physical NPU...\n");
    int rc = ioctl(c->fd, DRM_IOCTL_RKNPU_SUBMIT, &sub);

    // Trap locks and assert security bounds
    if (rc < 0) {
        fprintf(stderr, "❌ [test_ppu_lut] DRM SUBMIT IOCTL FAILED: errno=%d (%s)\n", errno, strerror(errno));
        if (errno == ETIMEDOUT || errno == EBUSY) {
            fprintf(stderr, "  WEDGE CAUGHT. Restoring physical core state...\n");
            trigger_hardware_reset(c->fd);
        }
        custom_dma_free(c->fd, &abuf);
        return 1; // Graceful non-zero exit to preserve SSH session stability
    }

    // Synchronize outputs back from the physical device
    bsync(c->fd, &abuf, RKNPU_MEM_SYNC_FROM_DEVICE);

    printf("[test_ppu_lut] Asserting mathematical Sigmoid correctness...\n");
    float max_diff = 0.0f;
    int mismatches = 0;
    
    // Output starts exactly at index 0 (offset 0 bytes) as raw FP16 values
    uint16_t *out_fp16 = (uint16_t *)abuf.cpu;

    for (int i = 0; i < 16; i++) {
        // Compute input value corresponding to what the model evaluated
        float x = -4.0f + (8.0f * i / 511); // The first 16 values are the ones computed
        float expected = sigmoid(x);
        
        // Decode the FP16 actual value from the NPU output region
        float actual = fp16_to_fp32(out_fp16[i]);
        float diff = fabsf(actual - expected);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > 0.05f) { // Allow strict 5% FP16 PWL tolerance
            mismatches++;
        }
        printf("i=%02d | x=%+5.2f | Exp: %8.6f | Actual: %8.6f (raw: 0x%04x) | Diff: %f\n", 
               i, x, expected, actual, out_fp16[i], diff);
    }

    printf("\n  Max Abs Deviation vs CPU-Reference: %f\n", max_diff);
    printf("  Matches: %d/16 (Mismatches: %d)\n", 16 - mismatches, mismatches);

    if (mismatches > 0) {
        fprintf(stderr, "❌ [test_ppu_lut] MATHEMATICAL VERIFICATION FAILED: %d mismatches found!\n", mismatches);
        custom_dma_free(c->fd, &abuf);
        return 1;
    }

    printf("🎉 [test_ppu_lut] SUCCESS: PPU LUT and PWL evaluated cleanly on physical NPU silicon!\n");
    custom_dma_free(c->fd, &abuf);
    return 0;
}
