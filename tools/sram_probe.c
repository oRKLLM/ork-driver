/* sram_probe — query the RK NPU on-chip SRAM/NBUF size (the "second memory interface").
 * Read-only: opens the DRM card, calls RKNPU_GET_TOTAL_SRAM_SIZE / FREE via RKNPU_ACTION. No submit.
 * Determines whether the weight-in-SRAM lever (RKNPU_MEM_TRY_ALLOC_SRAM) is viable for LLM weight tiles.
 *   cc -O2 -Isrc -o sram_probe tools/sram_probe.c && sudo ./sram_probe
 */
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include "rknpu_ioctl.h"

static unsigned q(int fd, unsigned action) {
    struct rknpu_action a; memset(&a, 0, sizeof a); a.flags = action; a.value = 0;
    if (ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &a) != 0) return 0xffffffffu;
    return a.value;
}

int main(void) {
    const char *nodes[] = {"/dev/dri/card1", "/dev/dri/card0", "/dev/dri/card2"};
    for (unsigned i = 0; i < sizeof nodes / sizeof *nodes; i++) {
        int fd = open(nodes[i], O_RDWR);
        if (fd < 0) continue;
        unsigned tot = q(fd, RKNPU_GET_TOTAL_SRAM_SIZE), fre = q(fd, RKNPU_GET_FREE_SRAM_SIZE);
        if (tot != 0xffffffffu) {
            printf("%s: NPU SRAM total=%u bytes (%.1f KB), free=%u bytes (%.1f KB)\n",
                   nodes[i], tot, tot / 1024.0, fre, fre / 1024.0);
            /* frame the lever: how big a weight tile fits? int4 K*N/2, int8 K*N */
            printf("  fits (int8 KxN weight): e.g. K=2048 -> N<=%u ; K=512 -> N<=%u\n",
                   tot ? tot / 2048 : 0, tot ? tot / 512 : 0);
            close(fd);
            return 0;
        }
        close(fd);
    }
    fprintf(stderr, "no DRM card answered RKNPU_GET_TOTAL_SRAM_SIZE\n");
    return 1;
}
