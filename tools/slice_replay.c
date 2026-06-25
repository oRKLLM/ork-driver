/* tools/slice_replay.c — Full 3-submit replay of the captured RKNN Sigmoid PPU op.
 *
 * Phase 1B follow-up. The standalone single-buffer replay (examples/test_ppu_lut.c) and a
 * single-submit replay both wrote no data. The SDK actually runs a persistent 7-task array
 * (obj[0..6] = two rounds of init/exec/clean with an op_idx=3 task) across THREE submits:
 *   submit 0: flags=0x5 task_start=0 num=3 (obj[0..2])   — confirmed clean (int_status fuzzes to 0x300)
 *   submit 1: flags=0x1 task_start=1 num=6 (obj[1..6])   — the operational compute push
 *   submit 2: flags=0x5 task_start=3 num=3 (obj[3..5])
 * This tool reconstructs the full 5-buffer topology + 7-task array from examples/sigmoid_slice.h
 * (extracted from ~/sigmoid_trace.log), replays all three submits in order with the SDK's exact
 * params, and delta-scans every buffer to see whether the PPU performs a destination write.
 *
 *   make slice_replay && sudo ./slice_replay
 *
 * RE diagnostic only — NOT in `make test`. main keeps activations on CPU/NEON.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include "ork_npu.h"
#include "rknpu_ioctl.h"
#include "sigmoid_slice.h"

struct buf { uint32_t handle; uint64_t dma, obj; void *cpu; size_t size; };
struct ork_npu_hdr { int fd; };

static size_t pgup(size_t s){ return (s + 4095) & ~((size_t)4095); }

static struct buf dma_alloc(int fd, size_t size, uint32_t flags) {
    struct rknpu_mem_create c; memset(&c,0,sizeof c);
    c.size = pgup(size); c.flags = flags; c.core_mask = RKNPU_CORE0_MASK;
    if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &c) < 0) { perror("MEM_CREATE"); return (struct buf){0}; }
    struct rknpu_mem_map m; memset(&m,0,sizeof m); m.handle = c.handle;
    if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &m) < 0) { perror("MEM_MAP"); return (struct buf){0}; }
    void *cpu = mmap(NULL, c.size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, m.offset);
    if (cpu == MAP_FAILED) { perror("mmap"); return (struct buf){0}; }
    return (struct buf){ c.handle, c.dma_addr, c.obj_addr, cpu, c.size };
}
static void bsync(int fd, struct buf *b, uint32_t f){
    struct rknpu_mem_sync s = { .flags=f, .obj_addr=b->obj, .offset=0, .size=b->size };
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, &s);
}
static void hw_reset(int fd){ struct rknpu_action a={.flags=RKNPU_ACT_RESET,.value=0}; ioctl(fd,DRM_IOCTL_RKNPU_ACTION,&a); }

/* SDK->live address remap in REGISTER-ENCODED form (word[k] high16=V_lo, word[k+1] low16=V_hi). */
static int patch_addr(uint32_t *w, size_t n, uint32_t oldv, uint32_t newv) {
    uint16_t olo=oldv&0xffff, ohi=oldv>>16, nlo=newv&0xffff, nhi=newv>>16; int hits=0;
    for (size_t k=0;k+1<n;k+=2)
        if ((w[k]>>16)==olo && (w[k+1]&0xffff)==ohi) {
            w[k]=(w[k]&0x0000ffff)|((uint32_t)nlo<<16); w[k+1]=(w[k+1]&0xffff0000)|nhi; hits++;
        }
    return hits;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    ork_npu *ctx = ork_npu_init();
    if (!ctx) { fprintf(stderr, "ork_npu_init failed\n"); return 1; }
    int fd = ((struct ork_npu_hdr*)ctx)->fd;

    /* 1. Allocate the full 5-buffer set with the SDK's sizes/flags. */
    struct buf B[6] = {0};
    for (int i=0;i<5;i++){
        B[SLICE_BUFS[i].handle] = dma_alloc(fd, SLICE_BUFS[i].size, SLICE_BUFS[i].flags);
        if (!B[SLICE_BUFS[i].handle].cpu){ fprintf(stderr,"alloc h%u failed\n",SLICE_BUFS[i].handle); return 1; }
        printf("[slice] h%u: %5u B flags=0x%-4x sdk_dma=0x%llx -> live 0x%llx obj=0x%llx\n",
               SLICE_BUFS[i].handle, SLICE_BUFS[i].size, SLICE_BUFS[i].flags,
               (unsigned long long)SLICE_BUFS[i].sdk_dma,(unsigned long long)B[SLICE_BUFS[i].handle].dma,
               (unsigned long long)B[SLICE_BUFS[i].handle].obj);
    }
    uint32_t oldv[5], newv[5];
    for (int i=0;i<5;i++){ oldv[i]=(uint32_t)SLICE_BUFS[i].sdk_dma; newv[i]=(uint32_t)B[SLICE_BUFS[i].handle].dma; }

    /* 2. Populate buffers with captured contents. handle 2 = regcmd buffer: config @0, 4 regcmd blocks. */
    memset(B[2].cpu,0,B[2].size);
    memcpy(B[2].cpu, HANDLE2, sizeof HANDLE2);
    memcpy((uint8_t*)B[2].cpu+0x2300, RC_xffffc300, sizeof RC_xffffc300);
    memcpy((uint8_t*)B[2].cpu+0x2580, RC_xffffc580, sizeof RC_xffffc580);
    memcpy((uint8_t*)B[2].cpu+0x4800, RC_xffffe800, sizeof RC_xffffe800);
    memcpy((uint8_t*)B[2].cpu+0x4a80, RC_xffffea80, sizeof RC_xffffea80);
    memset(B[3].cpu,0,B[3].size); memcpy(B[3].cpu, HANDLE3, sizeof HANDLE3);   // SDK input
    memset(B[4].cpu,0,B[4].size); memcpy(B[4].cpu, HANDLE4, sizeof HANDLE4);   // LUT/PWL config
    memset(B[5].cpu,0,B[5].size); memcpy(B[5].cpu, HANDLE5, sizeof HANDLE5);

    /* 3. Re-patch SDK dma addrs -> live (register-encoded) across regcmd + config buffers. */
    int ph=0;
    for (int m=0;m<5;m++){
        ph += patch_addr((uint32_t*)B[2].cpu, B[2].size/4, oldv[m], newv[m]);
        ph += patch_addr((uint32_t*)B[4].cpu, B[4].size/4, oldv[m], newv[m]);
        ph += patch_addr((uint32_t*)B[5].cpu, B[5].size/4, oldv[m], newv[m]);
    }
    printf("[slice] IOVA patch (reg-encoded): %d address pairs remapped\n", ph);

    /* 4. Build the persistent 7-task array (obj[0..6]) directly with live regcmd addresses. */
    struct rknpu_task *T = (struct rknpu_task *)B[1].cpu;
    memset(T,0,B[1].size);
    const struct { uint32_t op, rcfg, off; } TK[7] = {
        {1,69,0x2300},{2,1097,0x2580},{2,69,0x4800},{3,69,0x4a80},{1,69,0x2300},{2,1097,0x2580},{2,69,0x4800}
    };
    for (int i=0;i<7;i++){
        T[i].op_idx=TK[i].op; T[i].enable_mask=0x18; T[i].int_mask=0x300; T[i].int_clear=0x1ffff;
        T[i].regcfg_amount=TK[i].rcfg; T[i].regcfg_offset=0; T[i].regcmd_addr=B[2].dma+TK[i].off;
    }

    /* 5. Snapshot DATA buffers (3,4,5) to detect any genuine PPU compute write. */
    void *snap[6]={0}; for (int h=3;h<=5;h++){ snap[h]=malloc(B[h].size); memcpy(snap[h],B[h].cpu,B[h].size); }

    /* 6. Replay the 3 submits in order with the SDK's exact params. */
    const struct { uint32_t flags, tstart, tnum, tcnt, scs, scn; } SUB[3] = {
        {0x5,0,3,3, 0,1},  // subcore {0,1}x3
        {0x1,1,6,6, 1,2},  // subcore {1,2}x3
        {0x5,3,3,3, 3,1},  // subcore {3,1}x3
    };
    for (int s=0;s<3;s++){
        for (int h=1;h<=5;h++) bsync(fd,&B[h],RKNPU_MEM_SYNC_TO_DEVICE);
        struct rknpu_submit sub; memset(&sub,0,sizeof sub);
        sub.flags=SUB[s].flags; sub.timeout=2000; sub.task_start=SUB[s].tstart;
        sub.task_number=SUB[s].tnum; sub.task_counter=SUB[s].tcnt;
        sub.task_obj_addr=B[1].obj; sub.fence_fd=-1; sub.core_mask=RKNPU_CORE0_MASK;
        for (int k=0;k<3;k++) sub.subcore_task[k]=(struct rknpu_subcore_task){SUB[s].scs,SUB[s].scn};
        printf("[slice] submit %d: flags=0x%x task_start=%u num=%u ...", s, SUB[s].flags, SUB[s].tstart, SUB[s].tnum);
        int rc = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub);
        usleep(80000);
        if (rc < 0) {
            printf(" FAILED errno=%d (%s)\n", errno, strerror(errno));
            if (errno==ETIMEDOUT||errno==EBUSY){ fprintf(stderr,"  wedge -> RKNPU_ACT_RESET\n"); hw_reset(fd); }
            return 1;
        }
        for (int h=1;h<=5;h++) bsync(fd,&B[h],RKNPU_MEM_SYNC_FROM_DEVICE);
        // int_status of the tasks this submit ran (read back into the descriptors)
        printf(" rc=0  int_status[%u..%u]=", SUB[s].tstart, SUB[s].tstart+SUB[s].tnum-1);
        for (uint32_t i=SUB[s].tstart; i<SUB[s].tstart+SUB[s].tnum && i<7; i++) printf("0x%x ", T[i].int_status);
        printf("\n");
    }

    /* 7. Final delta-scan of the DATA buffers: did the PPU write a result anywhere? */
    printf("[slice] === final delta scan (data buffers) ===\n");
    int data_changed = 0;
    for (int h=3;h<=5;h++){
        uint16_t *now=(uint16_t*)B[h].cpu, *was=(uint16_t*)snap[h]; size_t nw=B[h].size/2;
        int changed=0, first=-1; for (size_t i=0;i<nw;i++){ if(now[i]!=was[i]){changed++; if(first<0)first=i;} }
        printf("  handle %d: %d / %zu u16 changed", h, changed, nw);
        if (changed){ data_changed=1; int st=first&~7; printf("  first@%d: ",first); for(int i=st;i<st+8;i++) printf("%04x ",now[i]); }
        printf("\n");
    }
    if (!data_changed){
        printf("\n❌ [slice] NEGATIVE: full 3-submit sequence (incl. submit 1 compute) wrote NO data.\n");
        return 1;
    }
    printf("\n✅ [slice] A DATA buffer was written — the full slice activated the PPU. Decode vs sigmoid next.\n");
    return 0;
}
