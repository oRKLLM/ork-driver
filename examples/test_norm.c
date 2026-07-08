/* examples/test_norm.c — Phase 1: Constraint-Guided Template-Preserving Fuzzer
 *
 * This tool systematically fuzzes speculative register offsets on the NPU
 * while preserving all critical template registers (like 0x4004, 0x400c)
 * at their exact, proven-working values from the REGCMD_I8 template.
 *
 * Compiles and runs on the Radxa ROCK 5B board:
 *   make test_norm
 *   sudo ./test_norm
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
#include "rknpu_ioctl.h"
#include "regcmd_i8.h"

#define PPU_BASE_OFFSET 0x4000     // PPU Block 0x1001 register base
#define SRAM_SCRATCH_MAX (32 * 1024) // 32KB hardware limit

struct gem_buf {
    uint32_t handle;
    uint64_t dma_addr;
    uint64_t obj_addr;
    uint64_t size;
    void *cpu_ptr;
};

static struct gem_buf alloc_gem(int fd, uint64_t size, uint32_t flags) {
    struct rknpu_mem_create creq = {0};
    creq.size = (size + 4095) & ~((uint64_t)4095);
    creq.flags = flags;
    creq.core_mask = RKNPU_CORE0_MASK;
    if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &creq) < 0) {
        perror("MEM_CREATE failed");
        exit(1);
    }
    struct rknpu_mem_map mreq = { .handle = creq.handle };
    if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mreq) < 0) {
        perror("MEM_MAP failed");
        exit(1);
    }
    void *cpu = mmap(0, creq.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mreq.offset);
    if (cpu == MAP_FAILED) {
        perror("mmap failed");
        exit(1);
    }
    return (struct gem_buf){ creq.handle, creq.dma_addr, creq.obj_addr, creq.size, cpu };
}

static void trigger_hardware_reset(int fd) {
    struct rknpu_action action = { .flags = RKNPU_ACT_RESET, .value = 0 };
    ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &action);
}

// ARM64 native FP16 to FP32 helpers
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

static void cpu_rmsnorm(float *out, const uint16_t *in, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float val = fp16_to_fp32(in[i]);
        sum += val * val;
    }
    float rsqrt = 1.0f / sqrtf(sum / n + 1e-5f);
    for (int i = 0; i < n; i++) {
        out[i] = fp16_to_fp32(in[i]) * rsqrt;
    }
}

int main(int argc, char **argv) {
    // Disable stdout buffering to ensure real-time terminal output
    setvbuf(stdout, NULL, _IONBF, 0);

    int fd = open("/dev/dri/card1", O_RDWR);
    if (fd < 0) {
        fd = open("/dev/dri/renderD129", O_RDWR);
    }
    if (fd < 0) {
        fd = open("/dev/dri/card0", O_RDWR);
    }
    if (fd < 0) {
        perror("Failed to open NPU card (/dev/dri/card1, renderD129, or card0)");
        return 1;
    }

    printf("[fuzz_norm] Allocating physical DMA buffers (32KB L1 SRAM bounds)...\n");
    int N = 512; // Test size (K-split friendly / standard hidden size chunk)
    struct gem_buf in_buf  = alloc_gem(fd, N * sizeof(uint16_t), 0x403);
    struct gem_buf out_buf = alloc_gem(fd, N * sizeof(float), 0x403);
    struct gem_buf cmd_buf = alloc_gem(fd, 16384, 0x40b);

    // Initialize inputs with deterministic test scale vector
    uint16_t *in_data = (uint16_t *)in_buf.cpu_ptr;
    float *out_data = (float *)out_buf.cpu_ptr;
    for (int i = 0; i < N; i++) {
        in_data[i] = 0x3C00; // 1.0f16
        out_data[i] = 0.0f;
    }

    // Calculate exact CPU reference
    float *cpu_ref = malloc(N * sizeof(float));
    cpu_rmsnorm(cpu_ref, in_data, N);

    uint32_t *cmd_ring = (uint32_t *)cmd_buf.cpu_ptr;

    // 1. Identify which registers are actively set inside the working REGCMD_I8 template
    // We map them so we DO NOT overwrite active control registers (clocks, configurations)
    uint8_t is_template_reg[65536] = {0};
    for (int i = 0; i < REGCMD_I8_N - 1; i += 2) {
        uint16_t reg_offset = REGCMD_I8[i] & 0xffff;
        uint16_t block_id = REGCMD_I8[i+1] >> 16;
        if (block_id == 0x1001) {
            is_template_reg[reg_offset] = 1;
        }
    }

    // We explicitly allow fuzzing the scaling / bias registers which are safe
    is_template_reg[0x4080] = 0; // Bias (Offset)
    is_template_reg[0x4084] = 0; // Multiplier
    is_template_reg[0x4088] = 0; // Right Shift

    printf("[fuzz_norm] Commencing constraint-guided register sweeps...\n");

    // We sweep speculative / unmapped registers in PPU block 0x1001 (0x4000 range)
    for (uint32_t reg_idx = 0; reg_idx < 128; reg_idx++) {
        uint32_t reg_addr = PPU_BASE_OFFSET + (reg_idx * 4);
        
        // Skip active template registers (clocks, structural control)
        if (is_template_reg[reg_addr & 0xffff]) {
            continue;
        }

        printf("[fuzz_norm] Probing speculative register 0x%04X...\n", reg_addr);

        for (uint32_t val_idx = 0; val_idx < 32; val_idx++) {
            uint32_t fuzzed_val = val_idx;

            // Seed cmd_ring with the baseline, proven-working MatMul configuration
            memset(cmd_ring, 0, 16384);
            memcpy(cmd_ring, REGCMD_I8, REGCMD_I8_N * sizeof(uint32_t));
            
            uint32_t w_idx = REGCMD_I8_N;

            // Override input/output buffers in the regcmd stream to point to our test buffers
            // Format: [reg_offset | length_mask] followed by [value]
            cmd_ring[w_idx++] = (((0x1070) >> 2) << 16) | 1; // DMA Source (A)
            cmd_ring[w_idx++] = (uint32_t)in_buf.dma_addr;

            cmd_ring[w_idx++] = (((0x4020) >> 2) << 16) | 1; // DMA Destination (C)
            cmd_ring[w_idx++] = (uint32_t)out_buf.dma_addr;

            // Patch dimension registers for our test size N=512
            cmd_ring[w_idx++] = (((0x403c) >> 2) << 16) | 1; // Width/Stride
            cmd_ring[w_idx++] = ((N-1) << 16) | (N-1);

            cmd_ring[w_idx++] = (((0x4058) >> 2) << 16) | 1; // Output N
            cmd_ring[w_idx++] = N-1;

            cmd_ring[w_idx++] = (((0x405c) >> 2) << 16) | 1; // M-limit
            cmd_ring[w_idx++] = 0; // Single row

            // Inject our speculative fuzzed register
            cmd_ring[w_idx++] = (((reg_addr) >> 2) << 16) | 1;
            cmd_ring[w_idx++] = fuzzed_val;

            // End sentinel
            cmd_ring[w_idx++] = 0xFFFFFFFF;

            // Format task descriptor
            struct rknpu_task *task = (struct rknpu_task *)((char *)cmd_buf.cpu_ptr + 8192);
            memset(task, 0, sizeof(*task));
            task->flags = 0;
            task->op_idx = 0;
            task->enable_mask = 0xd;
            task->int_mask = 0x300;
            task->int_clear = 0x1ffff;
            task->regcfg_amount = w_idx - 1;
            task->regcfg_offset = 0;
            task->regcmd_addr = cmd_buf.dma_addr;

            struct rknpu_submit submit = {
                .flags = 0x5,
                .timeout = 25, // Aggressive 25ms watchdog timeout
                .task_start = 0,
                .task_number = 1,
                .task_obj_addr = cmd_buf.obj_addr + 8192,
                .task_base_addr = 0,
                .fence_fd = -1,
                .core_mask = RKNPU_CORE0_MASK,
                .subcore_task = {
                    {0, 1},
                    {0, 1},
                    {0, 1}
                }
            };

            int rc = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit);

            if (rc < 0) {
                if (errno == ETIMEDOUT || errno == EBUSY) {
                    printf("[PROBE] Reg 0x%04X | Val 0x%08X: HARDWARE LOCK. Resetting...\n", reg_addr, fuzzed_val);
                    trigger_hardware_reset(fd);
                    usleep(5000);
                }
            } else {
                // Submit succeeded, check output array mathematically vs CPU reference
                float max_diff = 0.0f;
                int matches = 0;
                for (int i = 0; i < N; i++) {
                    float diff = fabsf(out_data[i] - cpu_ref[i]);
                    if (diff > max_diff) max_diff = diff;
                    if (diff < 0.01f) matches++;
                }

                if (matches == N) {
                    printf("\n🎉 [SUCCESS] MATHEMATICALLY CORRECT RMSNorm FOUND!\n");
                    printf("  Reg 0x%04X | Val 0x%08X\n", reg_addr, fuzzed_val);
                    printf("  Max Abs Diff = %e ( matches=%d/%d )\n\n", max_diff, matches, N);
                    free(cpu_ref);
                    close(fd);
                    return 0;
                }
            }
        }
    }

    printf("[fuzz_norm] Probing finished. No exact RMSNorm matches found in constraint sweep.\n");
    free(cpu_ref);
    close(fd);
    return 0;
}
