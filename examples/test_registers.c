/* examples/test_registers.c — Dissect which registers in mc>1 cause output modifications or break Row 0.
 * Testing with CONSECUTIVE rows layout for A (ORK_I4_ALAY=1 style).
 *
 *   make test_registers && sudo ./test_registers
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
#include "regcmd_i4.h"

#define ORK_MAXCORE 4
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

static void tile_i4_A(uint8_t*dst,const int8_t*A,int M,int K,int nib){
    int KT=K/32; memset(dst,0,(size_t)M*K/2);
    for(int kt=0;kt<KT;kt++)for(int m=0;m<M;m++)for(int kk=0;kk<32;kk++){
        size_t idx=((size_t)kt*M+m)*32+kk; uint8_t v=(uint8_t)(A[(size_t)m*K+kt*32+kk]&0xf);
        dst[idx/2]|= ((idx&1)^nib)?(v<<4):v;
    }
}
static void tile_i4_B(uint8_t*dst,const int8_t*B,int K,int N,int nib){
    int KT=K/32,NT=N/64; memset(dst,0,(size_t)K*N/2);
    for(int nt=0;nt<NT;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<64;nl++)for(int kk=0;kk<32;kk++){
        size_t idx=(((size_t)nt*KT+kt)*64+nl)*32+kk;
        uint8_t v=(uint8_t)(B[(size_t)(kt*32+kk)*N + (nt*64+nl)]&0xf);
        dst[idx/2]|= ((idx&1)^nib)?(v<<4):v;
    }
}

static void setr(uint32_t*rc,int n,uint32_t b,uint32_t o,uint32_t v){
    for(int k=0;k+1<n;k+=2) {
        if((rc[k]&0xffff)==o && (rc[k+1]>>16)==b){
            rc[k]=(o)|((v&0xffff)<<16);
            rc[k+1]=(b<<16)|((v>>16)&0xffff);
        }
    }
}

static void synth_i4_custom(uint32_t*rc, int mc, int K, int N, uint32_t aA, uint32_t aB, uint32_t aC, uint32_t mask) {
    memcpy(rc, REGCMD_I4, REGCMD_I4_N * 4);
    setr(rc, REGCMD_I4_N, 0x201, 0x1024, ((K-1)<<16)|K);
    setr(rc, REGCMD_I4_N, 0x201, 0x1030, (K*N)/2);
    setr(rc, REGCMD_I4_N, 0x201, 0x1034, K/2);
    setr(rc, REGCMD_I4_N, 0x201, 0x1044, (K+127)/128);
    setr(rc, REGCMD_I4_N, 0x201, 0x1088, K);
    setr(rc, REGCMD_I4_N, 0x201, 0x1038, 0x1010000|N);
    setr(rc, REGCMD_I4_N, 0x801, 0x3018, N-1);
    setr(rc, REGCMD_I4_N, 0x1001, 0x403c, ((N-1)<<16)|(N-1));
    setr(rc, REGCMD_I4_N, 0x1001, 0x4058, N-1);
    
    if (mc > 1) {
        if (mask & (1 << 0)) setr(rc, REGCMD_I4_N, 0x201, 0x1020, 0x10000|mc);
        if (mask & (1 << 1)) setr(rc, REGCMD_I4_N, 0x201, 0x1084, 0x10000|mc);
        if (mask & (1 << 2)) setr(rc, REGCMD_I4_N, 0x201, 0x102c, mc);
        if (mask & (1 << 3)) setr(rc, REGCMD_I4_N, 0x1001, 0x4034, mc-1);
        if (mask & (1 << 4)) setr(rc, REGCMD_I4_N, 0x1001, 0x405c, (mc-1)<<16);
        if (mask & (1 << 5)) setr(rc, REGCMD_I4_N, 0x801, 0x3014, (mc-1)<<16);
        if (mask & (1 << 6)) setr(rc, REGCMD_I4_N, 0x1001, 0x4038, (((N/4)-1)<<16)|((N/4)-1));
        if (mask & (1 << 7)) setr(rc, REGCMD_I4_N, 0x201, 0x1010, 16*(mc+1));
        if (mask & (1 << 8)) {
            int kk=K/256, lg=0; while(kk>1){kk>>=1; lg++;}
            int base=0xb1-15*((1<<lg)-1), slope=15*(1<<lg), mg=mc/64; if(mg<1)mg=1;
            int v=base-slope*(mg-1); if(v<0x1b)v=0x1b;
            setr(rc, REGCMD_I4_N, 0x201, 0x1040, v);
        }
    }
    setr(rc, REGCMD_I4_N, 0x201, 0x1070, aA);
    setr(rc, REGCMD_I4_N, 0x201, 0x1110, aB);
    setr(rc, REGCMD_I4_N, 0x1001, 0x4020, aC);
}

int main(void) {
    ork_npu *ctx = ork_npu_init();
    if (!ctx) { printf("init failed\n"); return 1; }
    
    int fd = ctx->fd;
    int M = 4, K = 64, N = 64;
    int8_t *A = malloc((size_t)M * K);
    int8_t *B = malloc((size_t)K * N);
    int32_t *ref = malloc((size_t)M * N * 4);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = r4();
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = r4();
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int s = 0; for (int k = 0; k < K; k++) s += A[(size_t)m * K + k] * B[(size_t)k * N + n];
            ref[(size_t)m * N + n] = s;
        }
    }
    
    struct buf W = bcreate(fd, (size_t)K * N / 2, 0x403);
    tile_i4_B(W.cpu, B, K, N, 0);
    bsync(fd, &W, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE); bsync(fd, &W, RKNPU_MEM_SYNC_TO_DEVICE);
    
    struct buf O = bcreate(fd, (size_t)M * N * 2, 0x403);
    
    // CONSECUTIVE layout style for A:
    for (int m = 0; m < M; m++) {
        tile_i4_A((uint8_t*)ctx->Af.cpu + (size_t)m * (K / 2), A + (size_t)m * K, 1, K, 0);
    }
    bsync(fd, &ctx->Af, RKNPU_MEM_SYNC_TO_DEVICE);
    
    const char *reg_names[] = {
        "0x1020", "0x1084", "0x102c", "0x4034", "0x405c", "0x3014", "0x4038", "0x1010", "0x1040"
    };
    
    // We only need to run Test 5 (Disabled 0x405c, mask=0x1ef)
    // and check if NPU matches CPU Ref!
    uint32_t mask = 0x1ef; // Disabled 0x405c
    printf("\n--- Running Test 5 (Disabled 0x405c) with CONSECUTIVE A ---\n");
    
    int16_t *raw = (int16_t*)O.cpu;
    for (size_t i = 0; i < (size_t)M * N; i++) raw[i] = 0x7aaa;
    bsync(fd, &O, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE); bsync(fd, &O, RKNPU_MEM_SYNC_TO_DEVICE);
    
    uint32_t rc[REGCMD_I4_N];
    synth_i4_custom(rc, M, K, N, (uint32_t)ctx->Af.dma, (uint32_t)W.dma, (uint32_t)O.dma, mask);
    
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
    if (ok < 0) {
        printf("Submit failed/timed out, rc=%d, errno=%d\n", ok, errno);
    } else {
        bsync(fd, &O, RKNPU_MEM_SYNC_FROM_DEVICE);
        for (int m = 0; m < M; m++) {
            printf("  Row%d NPU: ", m);
            for (int i = 0; i < 8; i++) printf("%6d ", raw[m * N + i]);
            printf(" | CPU Ref: ");
            for (int i = 0; i < 8; i++) printf("%6d ", ref[m * N + i]);
            printf("\n");
        }
    }
    
    bdestroy(fd, &W);
    bdestroy(fd, &O);
    free(A);
    free(B);
    free(ref);
    ork_npu_free(ctx);
    return 0;
}
