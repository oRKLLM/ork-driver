/* tools/stride_test.c — Probe to test output striding for INT8 (W8A8 -> INT32 output)
 * by verifying if 0x403c and 0x4038 encode [stride, slice_width] respectively.
 *
 *   make stride_test && sudo ./stride_test
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>
#include "ork_npu.h"
#include "rknpu_ioctl.h"
#include "regcmd_i8.h"

struct buf { uint32_t handle; uint64_t dma, obj; void *cpu; size_t size; };
struct ork_npu { int fd; const void *soc; struct buf regcmd, task, Af, Cc; size_t ccsz; void *cres; size_t cressz; int warmed, last_dt; int core_budget; };

static unsigned sd=99; static int8_t r4(void){sd=sd*1103515245+12345;return (int8_t)((int)((sd>>10)%15)-7);} /* [-7,7] */

static size_t pgup(size_t s){return (s+4095)&~((size_t)4095);}
static struct buf bcreate(int fd,size_t size,uint32_t flags){
    struct rknpu_mem_create c; memset(&c,0,sizeof c); c.size=pgup(size); c.flags=flags; c.core_mask=1;
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_CREATE,&c)){perror("CREATE");return (struct buf){0};}
    struct rknpu_mem_map m; memset(&m,0,sizeof m); m.handle=c.handle;
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_MAP,&m)){perror("MAP");return (struct buf){0};}
    void*p=mmap(NULL,c.size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,m.offset);
    if(p==MAP_FAILED){perror("mmap");return (struct buf){0};}
    return (struct buf){c.handle,c.dma_addr,c.obj_addr,p,c.size};
}
static void bdestroy(int fd,struct buf*b){ if(!b->cpu)return; munmap(b->cpu,b->size);
    struct rknpu_mem_destroy d; memset(&d,0,sizeof d); d.handle=b->handle; d.obj_addr=b->obj; ioctl(fd,DRM_IOCTL_RKNPU_MEM_DESTROY,&d); b->cpu=0; }
static void bsync(int fd,struct buf*b,uint32_t f){struct rknpu_mem_sync s;memset(&s,0,sizeof s);s.obj_addr=b->obj;s.size=b->size;s.flags=f;ioctl(fd,DRM_IOCTL_RKNPU_MEM_SYNC,&s);}
static void act(int fd,uint32_t f,uint32_t v){struct rknpu_action a={.flags=f,.value=v};ioctl(fd,DRM_IOCTL_RKNPU_ACTION,&a);}

static void setr(uint32_t*rc,int n,uint32_t b,uint32_t o,uint32_t v){
    for(int k=0;k+1<n;k+=2) {
        if((rc[k]&0xffff)==o && (rc[k+1]>>16)==b){
            rc[k]=(o)|((v&0xffff)<<16);
            rc[k+1]=(b<<16)|((v>>16)&0xffff);
        }
    }
}

static void synth_i8_strided(uint32_t*rc, int mc, int K, int Nc, int N, uint32_t aA, uint32_t aB, uint32_t aC) {
    memcpy(rc, REGCMD_I8, REGCMD_I8_N * 4);
    setr(rc, REGCMD_I8_N, 0x201, 0x1024, ((K-1)<<16)|K);
    setr(rc, REGCMD_I8_N, 0x201, 0x1030, K*Nc); // Weight size for the active Nc slice
    setr(rc, REGCMD_I8_N, 0x201, 0x1034, K);
    setr(rc, REGCMD_I8_N, 0x201, 0x1044, (K+63)/64);
    setr(rc, REGCMD_I8_N, 0x201, 0x1088, K);
    setr(rc, REGCMD_I8_N, 0x201, 0x107c, K/16);
    
    setr(rc, REGCMD_I8_N, 0x201, 0x1020, 0x10000|mc);
    setr(rc, REGCMD_I8_N, 0x201, 0x1084, 0x10000|mc);
    setr(rc, REGCMD_I8_N, 0x201, 0x102c, mc);
    
    setr(rc, REGCMD_I8_N, 0x1001, 0x4034, mc-1);
    setr(rc, REGCMD_I8_N, 0x1001, 0x405c, (mc-1)<<16);
    setr(rc, REGCMD_I8_N, 0x801, 0x3014, (mc-1)<<16);
    
    // CNA/DPU and PPU registers setting:
    // If our stride hypothesis is correct, the upper 16-bits encodes stride N-1
    // and lower 16-bits encodes slice Nc-1.
    setr(rc, REGCMD_I8_N, 0x1001, 0x403c, ((N-1)<<16)|(Nc-1));
    setr(rc, REGCMD_I8_N, 0x1001, 0x4058, Nc-1);
    setr(rc, REGCMD_I8_N, 0x1001, 0x4038, (((N/4)-1)<<16)|((Nc/4)-1));
    
    setr(rc, REGCMD_I8_N, 0x201, 0x1038, 0x1010000|Nc);
    setr(rc, REGCMD_I8_N, 0x801, 0x3018, Nc-1); // DP N-dim
    
    setr(rc, REGCMD_I8_N, 0x201, 0x1010, 16*(mc+1));
    
    setr(rc, REGCMD_I8_N, 0x201, 0x1070, aA);
    setr(rc, REGCMD_I8_N, 0x201, 0x1110, aB);
    setr(rc, REGCMD_I8_N, 0x1001, 0x4020, aC);
}

int main(int argc, char **argv) {
    int opt_403c = -1;
    int opt_4038 = -1;
    int opt_4058 = -1;
    int opt_3018 = -1;
    int opt_1038 = -1;
    int opt_40c0 = -1;
    int opt_405c = -1;
    if (argc > 1) opt_403c = (int)strtol(argv[1], NULL, 0);
    if (argc > 2) opt_4038 = (int)strtol(argv[2], NULL, 0);
    if (argc > 3) opt_4058 = (int)strtol(argv[3], NULL, 0);
    if (argc > 4) opt_3018 = (int)strtol(argv[4], NULL, 0);
    if (argc > 5) opt_1038 = (int)strtol(argv[5], NULL, 0);
    if (argc > 6) opt_40c0 = (int)strtol(argv[6], NULL, 0);
    if (argc > 7) opt_405c = (int)strtol(argv[7], NULL, 0);

    printf("Overrides: 0x403c=%08x 0x4038=%08x 0x4058=%08x 0x3018=%08x 0x1038=%08x 0x40c0=%08x 0x405c=%08x\n",
           opt_403c, opt_4038, opt_4058, opt_3018, opt_1038, opt_40c0, opt_405c);

    ork_npu *ctx = ork_npu_init();
    if (!ctx) { printf("init failed\n"); return 1; }
    
    int fd = ctx->fd;
    int M = 4, K = 64, Nc = 32, N = 64;
    
    int8_t *A = malloc((size_t)M * K);
    int8_t *B_slice = malloc((size_t)K * Nc);
    int32_t *ref = calloc((size_t)M * N, 4); // Strided C buffer of width N
    
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = r4();
    for (size_t i = 0; i < (size_t)K * Nc; i++) B_slice[i] = r4();
    
    // Compute exact CPU reference with stride N but width Nc
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < Nc; n++) {
            int s = 0;
            for (int k = 0; k < K; k++) {
                s += A[(size_t)m * K + k] * B_slice[(size_t)k * Nc + n];
            }
            ref[(size_t)m * N + n] = s;
        }
    }
    
    struct buf W = bcreate(fd, (size_t)K * Nc, 0x403); // int8 weight buffer
    int NN = Nc / 32, KT = K / 32; int8_t *bb = W.cpu;
    for (int nt = 0; nt < NN; nt++) {
        for (int kt = 0; kt < KT; kt++) {
            for (int nl = 0; nl < 32; nl++) {
                for (int kk = 0; kk < 32; kk++) {
                    bb[(size_t)nt * KT * 32 * 32 + (size_t)kt * 32 * 32 + nl * 32 + kk] = 
                        B_slice[(size_t)(kt * 32 + kk) * Nc + (nt * 32 + nl)];
                }
            }
        }
    }
    bsync(fd, &W, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE); bsync(fd, &W, RKNPU_MEM_SYNC_TO_DEVICE);
    
    struct buf O = bcreate(fd, (size_t)M * N * 4, 0x403); // output buffer C (int32_t) of width N
    int32_t *raw = (int32_t*)O.cpu;
    for (size_t i = 0; i < (size_t)M * N; i++) raw[i] = 0x7aaaaaaa;
    bsync(fd, &O, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE); bsync(fd, &O, RKNPU_MEM_SYNC_TO_DEVICE);
    
    int8_t *ad = ctx->Af.cpu;
    for (size_t i = 0; i < (size_t)M * K; i++) ad[i] = A[i];
    bsync(fd, &ctx->Af, RKNPU_MEM_SYNC_TO_DEVICE);
    
    uint32_t rc[REGCMD_I8_N];
    synth_i8_strided(rc, M, K, Nc, N, (uint32_t)ctx->Af.dma, (uint32_t)W.dma, (uint32_t)O.dma);
    
    if (opt_403c != -1) setr(rc, REGCMD_I8_N, 0x1001, 0x403c, opt_403c);
    if (opt_4038 != -1) setr(rc, REGCMD_I8_N, 0x1001, 0x4038, opt_4038);
    if (opt_4058 != -1) setr(rc, REGCMD_I8_N, 0x1001, 0x4058, opt_4058);
    if (opt_3018 != -1) setr(rc, REGCMD_I8_N, 0x801, 0x3018, opt_3018);
    if (opt_1038 != -1) setr(rc, REGCMD_I8_N, 0x201, 0x1038, opt_1038);
    if (opt_40c0 != -1) setr(rc, REGCMD_I8_N, 0x1001, 0x40c0, opt_40c0);
    if (opt_405c != -1) setr(rc, REGCMD_I8_N, 0x1001, 0x405c, opt_405c);

    act(fd, RKNPU_ACT_RESET, 0);
    memcpy(ctx->regcmd.cpu, rc, sizeof rc);
    bsync(fd, &ctx->regcmd, RKNPU_MEM_SYNC_TO_DEVICE);
    
    struct rknpu_submit sub;
    memset(&sub, 0, sizeof sub);
    sub.flags = 0x5;
    sub.task_number = 1;
    sub.task_obj_addr = ctx->task.obj;
    sub.core_mask = RKNPU_CORE0_MASK;
    sub.fence_fd = -1;
    sub.subcore_task[0] = sub.subcore_task[1] = sub.subcore_task[2] = (struct rknpu_subcore_task){0, 1};
    sub.timeout = 2000;
    
    int ok = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub);
    printf("Submit rc=%d\n", ok);
    
    if (ok >= 0) {
        bsync(fd, &O, RKNPU_MEM_SYNC_FROM_DEVICE);
        printf("Non-default values in output buffer:\n");
        for (int s = 0; s < M * N; s++) {
            if (raw[s] != 0x7aaaaaaa) {
                printf("  raw[%3d] = %d\n", s, raw[s]);
            }
        }
        int matched = 1;
        int landing_stride = -1;
        
        // Find where row 1 output actually landed
        for (int s = 0; s < M * N; s++) {
            if (raw[s] == ref[N]) { // Row 1 Col 0 landing spot
                landing_stride = s;
                break;
            }
        }
        
        printf("Row 1 Col 0 target value: %d\n", ref[N]);
        if (landing_stride != -1) {
            printf("Row 1 Col 0 actually landed at index: %d (offset: %d elements, %d bytes)\n", 
                   landing_stride, landing_stride, landing_stride * 4);
        } else {
            printf("Row 1 Col 0 value not found in output buffer!\n");
        }
        
        for (int m = 0; m < M; m++) {
            int start_idx = (m == 0) ? 0 : (Nc + (m - 1) * N);
            printf("  Row%d NPU (at %d): ", m, start_idx);
            for (int i = 0; i < 8 && i < Nc; i++) printf("%6d ", raw[start_idx + i]);
            printf(" | CPU Ref (at %d): ", m * N);
            for (int i = 0; i < 8 && i < Nc; i++) printf("%6d ", ref[m * N + i]);
            printf("\n");
            
            for (int n = 0; n < Nc; n++) {
                if (raw[start_idx + n] != ref[m * N + n]) matched = 0;
            }
        }
        
        if (matched) {
            printf("STRIDE TEST PASSED BIT-EXACTLY WITH OFFSET STRIDE!\n");
        } else {
            printf("STRIDE TEST FAILED! Matches offset-stride: no\n");
        }
    } else {
        printf("Submit failed with errno=%d\n", errno);
    }
    
    bdestroy(fd, &W);
    bdestroy(fd, &O);
    free(A);
    free(B_slice);
    free(ref);
    ork_npu_free(ctx);
    return 0;
}
