/* tools/submit_introspect.c — LD_PRELOAD shim: print rknpu SUBMIT task counts.
 * Intercepts ioctl; on DRM_IOCTL_RKNPU_SUBMIT (type 0x64, nr 0x41) decodes struct rknpu_submit and
 * prints task_number / task_counter / per-subcore task_number — to tell a MONOLITHIC per-group kernel
 * (task_number=1) from K/G PC-CHAINED tasks in one ioctl (task_number=K/G). API lead #2 mechanism.
 *   gcc -shared -fPIC -O2 -o submit_introspect.so tools/submit_introspect.c -ldl
 *   sudo LD_PRELOAD=./submit_introspect.so LD_LIBRARY_PATH=~/rknn_sdk ./pgquant_capture ...
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <dlfcn.h>
#include <stdarg.h>

struct subcore { uint32_t task_start, task_number; };
struct rknpu_submit {
    uint32_t flags, timeout, task_start, task_number, task_counter;
    int32_t  priority;
    uint64_t task_obj_addr;
    uint32_t iommu_domain_id, reserved;
    uint64_t task_base_addr;
    int64_t  hw_elapse_time;
    uint32_t core_mask;
    int32_t  fence_fd;
    struct subcore subcore_task[5];
};

static int (*real_ioctl)(int, unsigned long, ...) = 0;

int ioctl(int fd, unsigned long req, ...){
    va_list ap; va_start(ap, req); void *arg = va_arg(ap, void*); va_end(ap);
    if(!real_ioctl) real_ioctl = (int(*)(int,unsigned long,...))dlsym(RTLD_NEXT, "ioctl");
    if(((req>>8)&0xff)==0x64 && (req&0xff)==0x41){
        struct rknpu_submit *s = (struct rknpu_submit*)arg;
        fprintf(stderr, "[SUBMIT] task_number=%u task_counter=%u core_mask=0x%x dom=%u subcore_tasks=[%u,%u,%u,%u,%u]\n",
            s->task_number, s->task_counter, s->core_mask, s->iommu_domain_id,
            s->subcore_task[0].task_number, s->subcore_task[1].task_number, s->subcore_task[2].task_number,
            s->subcore_task[3].task_number, s->subcore_task[4].task_number);
    }
    return real_ioctl(fd, req, arg);
}
