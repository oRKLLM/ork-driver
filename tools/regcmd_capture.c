/* rknpu_dump.c — M1.2: capture the regcmd CONTENTS for a known matmul.
 *
 * Extends rknpu_trace.c: tracks handle -> {dma_addr, obj_addr, size, mmap_offset,
 * cpu_ptr} across MEM_CREATE / MEM_MAP / mmap, then on SUBMIT parses the task
 * descriptor (rknpu_task[]) to find each task's regcmd_addr + regcfg_amount,
 * locates the matching buffer by dma_addr, and hex-dumps:
 *   - the regcmd stream (as u32 words)
 *   - the rknpu_task descriptors (decoded)
 *   - every tracked buffer's first words (A / B / C / scratch)
 *
 *   gcc -shared -fPIC -O2 -I. -o rknpu_dump.so rknpu_dump.c -ldl
 *   sudo env LD_PRELOAD=$PWD/rknpu_dump.so LD_LIBRARY_PATH=. ./mmtest
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include "rknpu_ioctl.h"

#define MAXB 64
struct ent { uint32_t handle; uint64_t dma, obj, size, off; void *cpu; };
static struct ent tab[MAXB];
static int nent = 0;

static int (*real_ioctl)(int, unsigned long, ...) = NULL;
static void *(*real_mmap)(void *, size_t, int, int, int, off_t) = NULL;

__attribute__((constructor))
static void init(void) {
    real_ioctl = (int (*)(int, unsigned long, ...))dlsym(RTLD_NEXT, "ioctl");
    real_mmap  = (void *(*)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap");
    fprintf(stderr, "[rknpu_dump] loaded\n");
}

static struct ent *by_handle(uint32_t h) { for (int i=0;i<nent;i++) if (tab[i].handle==h) return &tab[i]; return NULL; }
static struct ent *by_off(uint64_t o)    { for (int i=0;i<nent;i++) if (tab[i].off==o)    return &tab[i]; return NULL; }
static struct ent *by_dma(uint64_t d)    { for (int i=0;i<nent;i++) if (tab[i].dma==d)    return &tab[i]; return NULL; }

static void record_map(off_t off, void *p) {
    struct ent *e = by_off((uint64_t)off);
    if (e && p != MAP_FAILED) { e->cpu = p; fprintf(stderr, "[dump] mmap handle=%u off=0x%llx -> %p\n", e->handle, (unsigned long long)off, p); }
    else if (p != MAP_FAILED && (uint64_t)off != 0) fprintf(stderr, "[dump] mmap UNTRACKED off=0x%llx -> %p\n", (unsigned long long)off, p);
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    if (!real_mmap) real_mmap = (void *(*)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap");
    void *p = real_mmap(addr, len, prot, flags, fd, off);
    record_map(off, p);
    return p;
}

static void *(*real_mmap64)(void *, size_t, int, int, int, off_t) = NULL;
void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    if (!real_mmap64) real_mmap64 = (void *(*)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap64");
    void *p = real_mmap64(addr, len, prot, flags, fd, off);
    record_map(off, p);
    return p;
}

static void hexwords(const char *tag, const uint32_t *w, int n) {
    fprintf(stderr, "  --- %s (%d u32 words) ---\n", tag, n);
    for (int i=0;i<n;i+=4) {
        fprintf(stderr, "  [%03d] %08x %08x %08x %08x\n", i,
            w[i], (i+1<n)?w[i+1]:0, (i+2<n)?w[i+2]:0, (i+3<n)?w[i+3]:0);
    }
}

int ioctl(int fd, unsigned long request, ...) {
    va_list ap; va_start(ap, request); void *arg = va_arg(ap, void *); va_end(ap);
    if (!real_ioctl) real_ioctl = (int (*)(int, unsigned long, ...))dlsym(RTLD_NEXT, "ioctl");
    int ret = real_ioctl(fd, request, arg);

    /* log ANY ioctl we don't specifically decode, to catch missing calls */
    switch (request) {
    case DRM_IOCTL_RKNPU_MEM_CREATE: case DRM_IOCTL_RKNPU_MEM_MAP:
    case DRM_IOCTL_RKNPU_MEM_DESTROY: case DRM_IOCTL_RKNPU_MEM_SYNC:
    case DRM_IOCTL_RKNPU_SUBMIT: case DRM_IOCTL_RKNPU_ACTION: break;
    default: fprintf(stderr, "[dump] OTHER ioctl fd=%d req=0x%lx -> %d\n", fd, request, ret); break;
    }

    if (request == DRM_IOCTL_RKNPU_MEM_CREATE) {
        struct rknpu_mem_create *m = arg;
        if (nent < MAXB) { tab[nent] = (struct ent){ m->handle, m->dma_addr, m->obj_addr, m->size, 0, NULL }; nent++; }
        fprintf(stderr, "[dump] MEM_CREATE handle=%u size=%llu dma=0x%llx obj=0x%llx flags=0x%x\n",
            m->handle, (unsigned long long)m->size, (unsigned long long)m->dma_addr, (unsigned long long)m->obj_addr, m->flags);
    } else if (request == DRM_IOCTL_RKNPU_MEM_MAP) {
        struct rknpu_mem_map *m = arg;
        struct ent *e = by_handle(m->handle);
        if (e) e->off = m->offset;
    } else if (request == DRM_IOCTL_RKNPU_SUBMIT) {
        struct rknpu_submit *s = arg;
        hexwords("submit-struct-raw", (uint32_t *)s, (int)(sizeof(struct rknpu_submit)/4));
        fprintf(stderr, "[dump] === SUBMIT flags=0x%x timeout=%u task_start=%u task_number=%u counter=%u prio=%d task_obj=0x%llx domain=%d base=0x%llx core=0x%x fence=%d ===\n",
            s->flags, s->timeout, s->task_start, s->task_number, s->task_counter, s->priority,
            (unsigned long long)s->task_obj_addr, s->iommu_domain_id,
            (unsigned long long)s->task_base_addr, s->core_mask, s->fence_fd);
        for (int i=0;i<5;i++)
            if (s->subcore_task[i].task_number)
                fprintf(stderr, "  subcore[%d]: task_start=%u task_number=%u\n",
                    i, s->subcore_task[i].task_start, s->subcore_task[i].task_number);
        /* raw task buffer: dump task_number * (40 bytes = 10 u32) */
        for (int i=0;i<nent;i++) if (tab[i].obj == s->task_obj_addr && tab[i].cpu) {
            int tw = (int)s->task_number * 10; if (tw > 64) tw = 64;
            hexwords("task-buffer-raw", (uint32_t *)tab[i].cpu, tw);
        }
        /* locate task descriptor buffer by obj_addr */
        struct ent *te = NULL;
        for (int i=0;i<nent;i++) if (tab[i].obj == s->task_obj_addr) { te = &tab[i]; break; }
        if (te && te->cpu) {
            struct rknpu_task *t = (struct rknpu_task *)te->cpu;
            for (uint32_t i=0;i<s->task_number;i++) {
                fprintf(stderr, "  task[%u]: flags=0x%x op_idx=%u enable=0x%x int_mask=0x%x regcfg_amount=%u regcfg_offset=%u regcmd_addr=0x%llx\n",
                    i, t[i].flags, t[i].op_idx, t[i].enable_mask, t[i].int_mask,
                    t[i].regcfg_amount, t[i].regcfg_offset, (unsigned long long)t[i].regcmd_addr);
                struct ent *re = by_dma(t[i].regcmd_addr);
                /* regcfg_amount = number of 64-bit (value,target) register writes */
                int rcwords = (int)t[i].regcfg_amount * 2 + 16;
                if (re && re->cpu) hexwords("regcmd", (uint32_t *)re->cpu, rcwords);
                else fprintf(stderr, "  (regcmd buffer for dma=0x%llx not mapped)\n", (unsigned long long)t[i].regcmd_addr);
            }
        } else {
            fprintf(stderr, "  (task descriptor not mapped; dumping all tracked buffers)\n");
        }
        /* dump every tracked buffer's head for correlation (A/B/C/scratch) */
        for (int i=0;i<nent;i++) {
            if (!tab[i].cpu) continue;
            char tag[64]; snprintf(tag, sizeof tag, "handle %u (dma=0x%llx size=%llu)", tab[i].handle, (unsigned long long)tab[i].dma, (unsigned long long)tab[i].size);
            int words = (int)(tab[i].size/4); if (words > 1024) words = 1024;
            hexwords(tag, (uint32_t *)tab[i].cpu, words);
        }
    }
    return ret;
}
