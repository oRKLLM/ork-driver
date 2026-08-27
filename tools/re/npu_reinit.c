/* npu_reinit — clear all RKNPU IOMMU domain state without rebooting.
 *
 * Issues RKNPU_ACT_REINIT (0x101) on the render node. The driver aborts any in-flight jobs, tears
 * down and rebuilds every IOMMU domain, and zeroes the domain refcount and diagnostic counters.
 *
 * WHY THIS EXISTS. `rknpu_job_abort()` releases the IOMMU domain reference unconditionally, so with
 * several cores busy it can drive `iommu_domain_refcount` to zero while other cores are still
 * executing. `rknpu_iommu_domain_get_and_switch()` treats a zero count as "safe to switch", so it
 * then switches domains underneath live work. That accumulates across runs until every switch fails
 * ("mismatch domain get from iommu_get_domain_for_dev") and every allocation fails
 * ("rknpu_gem_get_pages: dma map ... fail"). Filed upstream as rockchip-linux/kernel#387.
 *
 * Until that is fixed, a REBOOT was the only way to clear it — ~90 s, and it destroys any in-RAM
 * diagnostics. This is the shortcut.
 *
 * REQUIRES A PATCHED KERNEL. RKNPU_ACT_REINIT is ours, not upstream (see the wiki's
 * Kernel-Modifications page); on a stock kernel this returns -EINVAL, which is harmless.
 *
 * CALLER CONTRACT. The reinit changes IOVAs, so every existing buffer AND every regcmd built against
 * those IOVAs is invalid afterwards. Run it between processes, not inside one.
 *
 *   cc -O2 -Isrc -o npu_reinit tools/re/npu_reinit.c && sudo ./npu_reinit [/dev/dri/renderD129]
 */
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include "rknpu_ioctl.h"

#define RKNPU_ACT_REINIT_FLAG 0x101

int main(int argc, char **argv)
{
    const char *node = argc > 1 ? argv[1] : "/dev/dri/renderD129";
    struct rknpu_action a;
    int fd = open(node, O_RDWR);

    if (fd < 0) {
        fprintf(stderr, "npu_reinit: open %s: %s\n", node, strerror(errno));
        return 2;
    }
    memset(&a, 0, sizeof a);
    a.flags = RKNPU_ACT_REINIT_FLAG;
    if (ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &a) < 0) {
        fprintf(stderr, "npu_reinit: REINIT failed: %s (errno=%d)%s\n",
                strerror(errno), errno,
                errno == EINVAL ? " — stock kernel? this action is a local patch" :
                errno == EBUSY  ? " — jobs survived the reap; device is wedged beyond this" : "");
        close(fd);
        return 1;
    }
    printf("npu_reinit: ok — iommu domain state cleared\n");
    close(fd);
    return 0;
}
