/* rknpu_ioctl.h — userspace UAPI for the Rockchip RKNPU DRM driver.
 *
 * Transcribed from the Rockchip BSP kernel
 *   drivers/rknpu/include/rknpu_ioctl.h  (branch develop-6.1)
 * verified against the board's kernel 6.1.115-vendor-rk35xx (RKNPU driver v0.9.8,
 * RK3588). The NPU is a DRM device exposed as render node /dev/dri/renderD129
 * (DRIVER=RKNPU, of_node=npu). Render nodes need no DRM auth/master.
 *
 * This header is self-contained: it defines the minimal DRM ioctl encoding so it
 * builds without libdrm-dev. See the oRKLLM wiki "ggml-backend-rknpu" page.
 */
#ifndef RKNPU_IOCTL_H
#define RKNPU_IOCTL_H

#include <stdint.h>
#include <sys/ioctl.h>

/* Self-contained kernel types (avoid a hard dep on linux/types.h). */
#ifndef __rknpu_kernel_types
#define __rknpu_kernel_types
typedef uint32_t __u32;
typedef int32_t  __s32;
typedef uint64_t __u64;
typedef int64_t  __s64;
#endif

/* ---- Minimal DRM ioctl encoding (from <drm/drm.h>) ---- */
#ifndef DRM_IOCTL_BASE
#define DRM_IOCTL_BASE     'd'
#endif
#ifndef DRM_COMMAND_BASE
#define DRM_COMMAND_BASE   0x40
#endif
#define DRM_IOWR(nr, type) _IOWR(DRM_IOCTL_BASE, nr, type)

/* ---- RKNPU command numbers (offset from DRM_COMMAND_BASE) ---- */
#define RKNPU_ACTION       0x00
#define RKNPU_SUBMIT       0x01
#define RKNPU_MEM_CREATE   0x02
#define RKNPU_MEM_MAP      0x03
#define RKNPU_MEM_DESTROY  0x04
#define RKNPU_MEM_SYNC     0x05

/* ---- enum e_rknpu_action (RKNPU_ACTION ioctl selector, in .flags) ---- */
enum e_rknpu_action {
	RKNPU_GET_HW_VERSION = 0,
	RKNPU_GET_DRV_VERSION = 1,
	RKNPU_GET_FREQ = 2,
	RKNPU_SET_FREQ = 3,
	RKNPU_GET_VOLT = 4,
	RKNPU_SET_VOLT = 5,
	RKNPU_ACT_RESET = 6,
	RKNPU_GET_BW_PRIORITY = 7,
	RKNPU_SET_BW_PRIORITY = 8,
	RKNPU_GET_BW_EXPECT = 9,
	RKNPU_SET_BW_EXPECT = 10,
	RKNPU_GET_BW_TW = 11,
	RKNPU_SET_BW_TW = 12,
	RKNPU_ACT_CLR_TOTAL_RW_AMOUNT = 13,
	RKNPU_GET_DT_WR_AMOUNT = 14,
	RKNPU_GET_DT_RD_AMOUNT = 15,
	RKNPU_GET_WT_RD_AMOUNT = 16,
	RKNPU_GET_TOTAL_RW_AMOUNT = 17,
	RKNPU_GET_IOMMU_EN = 18,
	RKNPU_SET_PROC_NICE = 19,
	RKNPU_POWER_ON = 20,
	RKNPU_POWER_OFF = 21,
	RKNPU_GET_TOTAL_SRAM_SIZE = 22,
	RKNPU_GET_FREE_SRAM_SIZE = 23,
	RKNPU_GET_IOMMU_DOMAIN_ID = 24,
	RKNPU_SET_IOMMU_DOMAIN_ID = 25,
};

/* ---- enum e_rknpu_mem_type (rknpu_mem_create.flags) ---- */
enum e_rknpu_mem_type {
	RKNPU_MEM_CONTIGUOUS = 0 << 0,
	RKNPU_MEM_NON_CONTIGUOUS = 1 << 0,
	RKNPU_MEM_NON_CACHEABLE = 0 << 1,
	RKNPU_MEM_CACHEABLE = 1 << 1,
	RKNPU_MEM_WRITE_COMBINE = 1 << 2,
	RKNPU_MEM_KERNEL_MAPPING = 1 << 3,
	RKNPU_MEM_IOMMU = 1 << 4,
	RKNPU_MEM_ZEROING = 1 << 5,
	RKNPU_MEM_SECURE = 1 << 6,
	RKNPU_MEM_DMA32 = 1 << 7,
	RKNPU_MEM_TRY_ALLOC_SRAM = 1 << 8,
	RKNPU_MEM_TRY_ALLOC_NBUF = 1 << 9,
	RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT = 1 << 10,
};

/* ---- core mask (rknpu_submit.core_mask / rknpu_mem_create.core_mask) ---- */
#define RKNPU_CORE_AUTO_MASK 0x00
#define RKNPU_CORE0_MASK     0x01
#define RKNPU_CORE1_MASK     0x02
#define RKNPU_CORE2_MASK     0x04

/* ---- mem_sync flags (rknpu_mem_sync.flags) ---- */
#define RKNPU_MEM_SYNC_TO_DEVICE   (1 << 0)
#define RKNPU_MEM_SYNC_FROM_DEVICE (1 << 1)

/* ---- structs ---- */
struct rknpu_action {
	__u32 flags;   /* enum e_rknpu_action */
	__u32 value;   /* in/out value for the action */
};

struct rknpu_mem_create {
	__u32 handle;          /* out: GEM handle */
	__u32 flags;           /* enum e_rknpu_mem_type */
	__u64 size;            /* in: bytes */
	__u64 obj_addr;        /* out: kernel object addr (opaque cookie) */
	__u64 dma_addr;        /* out: device (IOVA) address */
	__u64 sram_size;
	__s32 iommu_domain_id;
	__u32 core_mask;
};

struct rknpu_mem_map {
	__u32 handle;          /* in */
	__u32 reserved;
	__u64 offset;          /* out: mmap offset to pass to mmap() */
};

struct rknpu_mem_destroy {
	__u32 handle;          /* in */
	__u32 reserved;
	__u64 obj_addr;        /* in: cookie from mem_create */
};

struct rknpu_mem_sync {
	__u32 flags;           /* RKNPU_MEM_SYNC_* */
	__u32 reserved;
	__u64 obj_addr;        /* in: cookie from mem_create */
	__u64 offset;
	__u64 size;
};

struct rknpu_task {
	__u32 flags;
	__u32 op_idx;
	__u32 enable_mask;
	__u32 int_mask;
	__u32 int_clear;
	__u32 int_status;
	__u32 regcfg_amount;   /* number of regcmd words */
	__u32 regcfg_offset;
	__u64 regcmd_addr;     /* device addr of the register-command buffer */
} __attribute__((packed));

struct rknpu_subcore_task {
	__u32 task_start;
	__u32 task_number;
};

struct rknpu_submit {
	__u32 flags;
	__u32 timeout;
	__u32 task_start;
	__u32 task_number;
	__u32 task_counter;
	__s32 priority;
	__u64 task_obj_addr;   /* device addr of the rknpu_task[] array */
	__u32 iommu_domain_id;
	__u32 reserved;
	__u64 task_base_addr;
	__s64 hw_elapse_time;  /* out */
	__u32 core_mask;
	__s32 fence_fd;
	struct rknpu_subcore_task subcore_task[5];
};

/* ---- DRM ioctl macros ---- */
#define DRM_IOCTL_RKNPU_ACTION \
	DRM_IOWR(DRM_COMMAND_BASE + RKNPU_ACTION, struct rknpu_action)
#define DRM_IOCTL_RKNPU_SUBMIT \
	DRM_IOWR(DRM_COMMAND_BASE + RKNPU_SUBMIT, struct rknpu_submit)
#define DRM_IOCTL_RKNPU_MEM_CREATE \
	DRM_IOWR(DRM_COMMAND_BASE + RKNPU_MEM_CREATE, struct rknpu_mem_create)
#define DRM_IOCTL_RKNPU_MEM_MAP \
	DRM_IOWR(DRM_COMMAND_BASE + RKNPU_MEM_MAP, struct rknpu_mem_map)
#define DRM_IOCTL_RKNPU_MEM_DESTROY \
	DRM_IOWR(DRM_COMMAND_BASE + RKNPU_MEM_DESTROY, struct rknpu_mem_destroy)
#define DRM_IOCTL_RKNPU_MEM_SYNC \
	DRM_IOWR(DRM_COMMAND_BASE + RKNPU_MEM_SYNC, struct rknpu_mem_sync)

#endif /* RKNPU_IOCTL_H */
