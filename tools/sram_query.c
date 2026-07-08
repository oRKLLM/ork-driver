/* tools/sram_query.c — read-only query of the rknpu DRM: SRAM size, IOMMU, freq, versions.
 * No submits, no allocations -> cannot wedge the NPU. Answers "is the NPU SRAM big enough to matter?"
 *   make sram_query && sudo ./sram_query
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include "rknpu_ioctl.h"

static long q(int fd, uint32_t action){
    struct rknpu_action a; memset(&a,0,sizeof a); a.flags=action; a.value=0;
    if(ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &a)) return -1;
    return (long)a.value;
}

int main(void){
    int fd=-1;
    for(int n=0;n<4;n++){ char up[64]; snprintf(up,sizeof up,"/sys/class/drm/card%d/device/uevent",n);
        FILE*uf=fopen(up,"r"); if(!uf) continue; char line[256]; int is_rknpu=0;
        while(fgets(line,sizeof line,uf)) if(strstr(line,"RKNPU")) is_rknpu=1; fclose(uf);
        if(!is_rknpu) continue;
        char p[32]; snprintf(p,sizeof p,"/dev/dri/card%d",n); fd=open(p,O_RDWR);
        if(fd>=0){ printf("RKNPU on /dev/dri/card%d\n",n); break; } }
    if(fd<0){ printf("no RKNPU card found\n"); return 1; }
    printf("HW version:        %ld\n", q(fd,RKNPU_GET_HW_VERSION));
    printf("DRV version:       %ld\n", q(fd,RKNPU_GET_DRV_VERSION));
    printf("FREQ:              %ld\n", q(fd,RKNPU_GET_FREQ));
    printf("IOMMU enabled:     %ld\n", q(fd,RKNPU_GET_IOMMU_EN));
    printf("IOMMU domain id:   %ld\n", q(fd,RKNPU_GET_IOMMU_DOMAIN_ID));
    printf("TOTAL SRAM bytes:  %ld  (%.1f KiB)\n", q(fd,RKNPU_GET_TOTAL_SRAM_SIZE), q(fd,RKNPU_GET_TOTAL_SRAM_SIZE)/1024.0);
    printf("FREE  SRAM bytes:  %ld  (%.1f KiB)\n", q(fd,RKNPU_GET_FREE_SRAM_SIZE), q(fd,RKNPU_GET_FREE_SRAM_SIZE)/1024.0);
    close(fd); return 0;
}
