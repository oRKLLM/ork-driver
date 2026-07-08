/* tools/vec_fuzz.c — Bare-metal register-command space fuzzer with auto-reset
 *
 * Compiles on the board:
 *   gcc -O2 -Iinclude -Isrc -o tools/vec_fuzz tools/vec_fuzz.c
 *   sudo ./tools/vec_fuzz
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include "rknpu_ioctl.h"
#include "regcmd_i8.h" // Provides REGCMD_I8 array of size REGCMD_I8_N

#define PPU_BASE_OFFSET 0x3000  // NVDLA PDP / Rockchip PPU base register block
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

int main(int argc, char **argv) {
    // Disable stdout buffering to ensure real-time terminal output over SSH pipes
    setvbuf(stdout, NULL, _IONBF, 0);

    // On RK3588, /dev/dri/card1 is the NPU node (card0 is display). Prioritize card1 and renderD129.
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

    printf("[fuzz] Initializing memory mappings (32KB L1 SRAM limits)...\n");
    struct gem_buf in_buf  = alloc_gem(fd, SRAM_SCRATCH_MAX, 0x403);
    struct gem_buf out_buf = alloc_gem(fd, SRAM_SCRATCH_MAX, 0x403);
    struct gem_buf cmd_buf = alloc_gem(fd, 16384, 0x40b); // 16KB command ring

    // Initialize inputs with deterministic floating-point scale vectors (e.g. 1.0f16 = 0x3C00)
    uint16_t *in_data = (uint16_t *)in_buf.cpu_ptr;
    uint16_t *out_data = (uint16_t *)out_buf.cpu_ptr;
    for (int i = 0; i < SRAM_SCRATCH_MAX / 2; i++) {
        in_data[i] = 0x3C00;
        out_data[i] = 0x0000;
    }

    uint32_t *cmd_ring = (uint32_t *)cmd_buf.cpu_ptr;

    printf("[fuzz] Commencing register-probing loop over PPU register block...\n");

    // We sweep 128 register offsets inside the PPU block, testing bit configuration transitions
    for (uint32_t reg_idx = 0; reg_idx < 128; reg_idx++) {
        uint32_t reg_addr = PPU_BASE_OFFSET + (reg_idx * 4);
        printf("[fuzz] Probing register 0x%04X...\n", reg_addr);

        for (uint32_t bit_pos = 0; bit_pos < 32; bit_pos++) {
            uint32_t fuzzed_val = (1U << bit_pos);

            // Clean cmd_ring buffer and seed it with baseline MatMul configuration template
            memset(cmd_ring, 0, 16384);
            memcpy(cmd_ring, REGCMD_I8, REGCMD_I8_N * sizeof(uint32_t));
            
            uint32_t w_idx = REGCMD_I8_N;
            
            // Adjust the DMA source/dest to our custom GEM buffers within the regcmd stream
            // In ork-driver regcmd, we can append target overrides or modify them in place.
            // Let's append overrides to write to our newly fuzzed offsets.
            // Format: [reg_offset | length_mask] followed by [value]
            cmd_ring[w_idx++] = (((0x1070) >> 2) << 16) | 1; // DMA Source (Activation)
            cmd_ring[w_idx++] = (uint32_t)in_buf.dma_addr;

            cmd_ring[w_idx++] = (((0x4020) >> 2) << 16) | 1; // DMA Destination (Output)
            cmd_ring[w_idx++] = (uint32_t)out_buf.dma_addr;

            // Inject the fuzzed vector configuration register in the PDP register window
            cmd_ring[w_idx++] = (((reg_addr) >> 2) << 16) | 1;
            cmd_ring[w_idx++] = fuzzed_val;

            // Finalize regcmd end-of-packet sentinel
            cmd_ring[w_idx++] = 0xFFFFFFFF; 

            // Format rknpu_task descriptor in the second half of command buffer
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
                .flags = 0x5, // Crucial flags for RKNPU driver
                .timeout = 25, // Aggressive 25ms timeout: trap hangs instantly
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
                // Check if the kernel driver blocked due to a hardware pipeline jam (REGTASK Overflow / Wedge)
                if (errno == ETIMEDOUT || errno == EBUSY) {
                    printf("[PROBE] Reg 0x%04X | Bit %02d: HARDWARE LOCK (0xe010 / Bus Stall). Healing...\n", 
                           reg_addr, bit_pos);
                    trigger_hardware_reset(fd);
                    usleep(5000); // 5ms settling delay
                } else {
                    // Print rejected error numbers to see if some register writes are blacklisted/invalid by driver
                    printf("[PROBE] Reg 0x%04X | Bit %02d: Rejected (rc=%d, errno=%d)\n", reg_addr, bit_pos, rc, errno);
                }
            } else {
                // If it ran cleanly, inspect our output array for vector transformations
                if (out_data[0] != 0x0000) {
                    printf("[PROBE] Reg 0x%04X | Bit %02d: SUCCESS. Output Mutation Detected: 0x%04X\n", 
                           reg_addr, bit_pos, out_data[0]);
                }
            }
        }
    }

    printf("[fuzz] Done. Tearing down...\n");
    munmap(in_buf.cpu_ptr, SRAM_SCRATCH_MAX);
    munmap(out_buf.cpu_ptr, SRAM_SCRATCH_MAX);
    munmap(cmd_buf.cpu_ptr, 16384);
    close(fd);
    return 0;
}
