/* tools/i4_multim_fuzz.c — Tier 4b RE: can the int4 regcmd do MULTI-M in one submit, and what is the
 * register to trigger it? Sweeps empty/unmapped registers across CNA, DPU, and PPU blocks.
 *
 *   make i4_multim_fuzz && sudo ./i4_multim_fuzz
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

static unsigned sd=99; static int8_t r4(void){sd=sd*1103515245+12345;return (int8_t)((int)((sd>>10)%15)-7);} /* [-7,7] */

#define ORK_MAXCORE 4
struct buf { uint32_t handle; uint64_t dma, obj; void *cpu; size_t size; };
struct ork_npu { int fd; const void *soc; struct buf regcmd, task, Af, Cc; size_t ccsz; void *cres; size_t cressz; int warmed, last_dt; int core_budget; };

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

static void setr_or_inject(uint32_t *rc, int n, uint32_t b, uint32_t o, uint32_t v) {
    for (int k = 0; k + 1 < n; k += 2) {
        if ((rc[k] & 0xffff) == o && (rc[k+1] >> 16) == b) {
            rc[k] = (o) | ((v & 0xffff) << 16);
            rc[k+1] = (b << 16) | ((v >> 16) & 0xffff);
            return;
        }
    }
    for (int k = 0; k + 1 < n; k += 2) {
        uint32_t offset = rc[k] & 0xffff;
        uint32_t block_id = rc[k+1] >> 16;
        uint32_t val = (rc[k] >> 16) | ((rc[k+1] & 0xffff) << 16);
        if (val == 0 && block_id == b) {
            if ((b == 0x201 && offset >= 0x1140) || 
                (b == 0x1001 && offset >= 0x4100) || 
                (b == 0x801 && (offset == 0x3014 || offset == 0x301c || offset == 0x3030))) {
                rc[k] = (o) | ((v & 0xffff) << 16);
                rc[k+1] = (b << 16) | ((v >> 16) & 0xffff);
                return;
            }
        }
    }
    for (int k = 0; k + 1 < n; k += 2) {
        uint32_t offset = rc[k] & 0xffff;
        uint32_t block_id = rc[k+1] >> 16;
        uint32_t val = (rc[k] >> 16) | ((rc[k+1] & 0xffff) << 16);
        if (val == 0) {
            if ((block_id == 0x201 && offset >= 0x1140) || 
                (block_id == 0x1001 && offset >= 0x4100) || 
                (block_id == 0x801 && (offset == 0x3014 || offset == 0x301c || offset == 0x3030))) {
                rc[k] = (o) | ((v & 0xffff) << 16);
                rc[k+1] = (b << 16) | ((v >> 16) & 0xffff);
                return;
            }
        }
    }
    fprintf(stderr, "[fuzzer] WARNING: could not inject register %04x block %04x\n", o, b);
}

static void synth_i4(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC){
    memcpy(rc,REGCMD_I4,REGCMD_I4_N*4);
    setr(rc,REGCMD_I4_N,0x201,0x1024,((K-1)<<16)|K);
    setr(rc,REGCMD_I4_N,0x201,0x1030,(K*N)/2);
    setr(rc,REGCMD_I4_N,0x201,0x1034,K/2);
    setr(rc,REGCMD_I4_N,0x201,0x1044,(K+127)/128);
    setr(rc,REGCMD_I4_N,0x201,0x1088,K);
    setr(rc,REGCMD_I4_N,0x201,0x1038,0x1010000|N);
    setr(rc,REGCMD_I4_N,0x801,0x3018,N-1);
    setr(rc,REGCMD_I4_N,0x1001,0x403c,((N-1)<<16)|(N-1));
    setr(rc,REGCMD_I4_N,0x1001,0x4058,N-1);
    if(mc>1){
        setr(rc,REGCMD_I4_N,0x201,0x1020,0x10000|mc);
        setr(rc,REGCMD_I4_N,0x201,0x1084,0x10000|mc);
        setr(rc,REGCMD_I4_N,0x201,0x102c,mc);
        setr(rc,REGCMD_I4_N,0x1001,0x4034,mc-1);
        setr(rc,REGCMD_I4_N,0x1001,0x405c,(mc-1)<<16);
        setr(rc,REGCMD_I4_N,0x801,0x3014,(mc-1)<<16);
        setr(rc,REGCMD_I4_N,0x1001,0x4038,(((N/4)-1)<<16)|((N/4)-1));
        setr(rc,REGCMD_I4_N,0x201,0x1010,16*(mc+1));
        int kk=K/256,lg=0; while(kk>1){kk>>=1;lg++;} int base=0xb1-15*((1<<lg)-1),slope=15*(1<<lg),mg=mc/64; if(mg<1)mg=1;
        int v=base-slope*(mg-1); if(v<0x1b)v=0x1b; setr(rc,REGCMD_I4_N,0x201,0x1040,v);
    }
    setr(rc,REGCMD_I4_N,0x201,0x1070,aA);
    setr(rc,REGCMD_I4_N,0x201,0x1110,aB);
    setr(rc,REGCMD_I4_N,0x1001,0x4020,aC);
}

static int is_essential_reg(uint32_t b, uint32_t o) {
    if (b == 0x201) {
        if (o == 0x100c || o == 0x1024 || o == 0x1030 || o == 0x1034 || o == 0x1038 ||
            o == 0x1044 || o == 0x1070 || o == 0x1080 || o == 0x1088 || o == 0x1110)
            return 1;
        if (o == 0x1020 || o == 0x1084 || o == 0x102c || o == 0x1010 || o == 0x1040)
            return 1;
    } else if (b == 0x1001) {
        if (o == 0x4004 || o == 0x400c || o == 0x4010 || o == 0x4020 || o == 0x4024 ||
            o == 0x403c || o == 0x4050 || o == 0x4058 || o == 0x4070)
            return 1;
        if (o == 0x4034 || o == 0x4038 || o == 0x405c)
            return 1;
    } else if (b == 0x801) {
        if (o == 0x3010 || o == 0x3018)
            return 1;
        if (o == 0x3014)
            return 1;
    }
    return 0;
}

#define BLACKLIST_FILE "fuzz_blacklist.txt"
#define PROGRESS_FILE "fuzz_progress.txt"

static int in_blacklist(uint32_t b, uint32_t o) {
    FILE *f = fopen(BLACKLIST_FILE, "r");
    if (!f) return 0;
    uint32_t fb, fo;
    int found = 0;
    while (fscanf(f, "%x %x\n", &fb, &fo) == 2) {
        if (fb == b && fo == o) {
            found = 1;
            break;
        }
    }
    fclose(f);
    return found;
}

static void add_to_blacklist(uint32_t b, uint32_t o) {
    if (in_blacklist(b, o)) return;
    FILE *f = fopen(BLACKLIST_FILE, "a");
    if (f) {
        fprintf(f, "%x %x\n", b, o);
        fclose(f);
        printf("[fuzzer] Blacklisted register block=%x offset=%x\n", b, o);
    }
}

static void save_progress(uint32_t b, uint32_t o, int val_idx) {
    FILE *f = fopen(PROGRESS_FILE, "w");
    if (f) {
        fprintf(f, "%x %x %d\n", b, o, val_idx);
        fclose(f);
    }
}

static int load_progress(uint32_t *pb, uint32_t *po, int *pval_idx) {
    FILE *f = fopen(PROGRESS_FILE, "r");
    if (!f) return 0;
    int ok = fscanf(f, "%x %x %d\n", pb, po, pval_idx) == 3;
    fclose(f);
    return ok;
}

int main(void) {
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    int fd = ctx->fd;
    int M=4, K=64, N=64;
    
    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t*ref=malloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=r4();
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=r4();
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){
        int s=0;for(int k=0;k<K;k++)s+=A[(size_t)m*K+k]*B[(size_t)k*N+n];
        ref[(size_t)m*N+n]=s;
    }
    
    struct buf W = bcreate(fd, (size_t)K*N/2, 0x403);
    if (!W.cpu) { printf("W bcreate failed\n"); return 1; }
    tile_i4_B(W.cpu, B, K, N, 0);
    bsync(fd, &W, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE); bsync(fd, &W, RKNPU_MEM_SYNC_TO_DEVICE);
    
    struct buf O = bcreate(fd, (size_t)M*N*2, 0x403);
    if (!O.cpu) { printf("O bcreate failed\n"); bdestroy(fd, &W); return 1; }
    
    for(int m=0;m<M;m++) tile_i4_A((uint8_t*)ctx->Af.cpu+(size_t)m*(K/2),A+(size_t)m*K,1,K,0);
    bsync(fd,&ctx->Af,RKNPU_MEM_SYNC_TO_DEVICE);

    int16_t *baseline_out = malloc((size_t)M * N * 2);
    if (!baseline_out) { printf("baseline_out malloc failed\n"); return 1; }
    
    int16_t *out_cpu_base = (int16_t*)O.cpu;
    for (size_t i = 0; i < (size_t)M*N; i++) out_cpu_base[i] = 0x7aaa;
    bsync(fd, &O, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE); bsync(fd, &O, RKNPU_MEM_SYNC_TO_DEVICE);
    
    uint32_t rc_base[REGCMD_I4_N];
    synth_i4(rc_base, M, K, N, (uint32_t)ctx->Af.dma, (uint32_t)W.dma, (uint32_t)O.dma);
    
    act(fd, RKNPU_ACT_RESET, 0);
    memcpy(ctx->regcmd.cpu, rc_base, sizeof rc_base);
    bsync(fd, &ctx->regcmd, RKNPU_MEM_SYNC_TO_DEVICE);
    
    struct rknpu_submit sub_base;
    memset(&sub_base, 0, sizeof sub_base);
    sub_base.flags = 0x5;
    sub_base.task_number = 1;
    sub_base.task_obj_addr = ctx->task.obj;
    sub_base.core_mask = RKNPU_CORE0_MASK;
    sub_base.fence_fd = -1;
    sub_base.subcore_task[0] = (struct rknpu_subcore_task){0, 1};
    sub_base.timeout = 2000;
    
    int ok_base = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub_base);
    if (ok_base < 0) {
        printf("Baseline run failed! rc=%d, errno=%d\n", ok_base, errno);
        free(baseline_out);
        bdestroy(fd, &W); bdestroy(fd, &O);
        free(A); free(B); free(ref);
        ork_npu_free(ctx);
        return 1;
    }
    bsync(fd, &O, RKNPU_MEM_SYNC_FROM_DEVICE);
    memcpy(baseline_out, out_cpu_base, (size_t)M * N * 2);
    
    printf("[fuzzer] Baseline run completed successfully. First 8 elements of each row:\n");
    for (int m = 0; m < M; m++) {
        printf("  Row%d: ", m);
        for (int i = 0; i < 8; i++) printf("%04x ", (uint16_t)baseline_out[m*N + i]);
        printf("\n");
    }
    
    uint32_t fvals[] = { 3, 4, 0x10003, 0x00030003 };
    int n_fvals = sizeof(fvals)/sizeof(fvals[0]);
    
    uint32_t start_b = 0x201, start_o = 0x1000;
    int start_val_idx = 0;
    if (load_progress(&start_b, &start_o, &start_val_idx)) {
        printf("[fuzzer] Resuming progress at block=%x offset=%x val_idx=%d\n", start_b, start_o, start_val_idx);
    } else {
        printf("[fuzzer] Starting fresh fuzzing sweep...\n");
    }
    
    struct { uint32_t block; uint32_t min_off, max_off; const char *name; } spaces[] = {
        { 0x201,  0x1000, 0x11c0, "CNA" },
        { 0x801,  0x3000, 0x30c0, "DPU" },
        { 0x1001, 0x4000, 0x4140, "PPU" }
    };
    int n_spaces = sizeof(spaces)/sizeof(spaces[0]);
    
    int total_runs = 0, hit_count = 0;
    
    for (int s_idx = 0; s_idx < n_spaces; s_idx++) {
        uint32_t b = spaces[s_idx].block;
        if (b < start_b) continue;
        
        for (uint32_t o = spaces[s_idx].min_off; o <= spaces[s_idx].max_off; o += 4) {
            if (b == start_b && o < start_o) continue;
            if (is_essential_reg(b, o)) continue;
            if (in_blacklist(b, o)) {
                printf("[fuzzer] Skipping blacklisted block=%x offset=%04x\n", b, o);
                continue;
            }
            
            for (int v_idx = 0; v_idx < n_fvals; v_idx++) {
                if (b == start_b && o == start_o && v_idx < start_val_idx) continue;
                
                uint32_t val = fvals[v_idx];
                
                int16_t *out_cpu = (int16_t*)O.cpu;
                for (size_t i = 0; i < (size_t)M*N; i++) out_cpu[i] = 0x7aaa;
                bsync(fd, &O, RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE); bsync(fd, &O, RKNPU_MEM_SYNC_TO_DEVICE);
                
                uint32_t rc[REGCMD_I4_N];
                synth_i4(rc, M, K, N, (uint32_t)ctx->Af.dma, (uint32_t)W.dma, (uint32_t)O.dma);
                
                setr_or_inject(rc, REGCMD_I4_N, b, o, val);
                
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
                sub.subcore_task[0] = (struct rknpu_subcore_task){0, 1};
                sub.timeout = 2000;
                
                printf("[fuzzer] Testing block=%s(%04x) offset=%04x val=%08x... ", spaces[s_idx].name, b, o, val);
                fflush(stdout);
                
                int next_v = v_idx + 1;
                uint32_t next_o = o;
                uint32_t next_b = b;
                if (next_v >= n_fvals) {
                    next_v = 0;
                    next_o += 4;
                    if (next_o > spaces[s_idx].max_off) {
                        if (s_idx + 1 < n_spaces) {
                            next_b = spaces[s_idx + 1].block;
                            next_o = spaces[s_idx + 1].min_off;
                        } else {
                            next_b = 0;
                            next_o = 0;
                        }
                    }
                }
                save_progress(next_b, next_o, next_v);
                
                int ok = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub);
                total_runs++;
                
                if (ok < 0) {
                    printf("TIMEOUT/FAIL (rc=%d, errno=%d)\n", ok, errno);
                    struct rknpu_action a = { .flags = RKNPU_ACT_RESET, .value = 0 };
                    ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &a);
                    add_to_blacklist(b, o);
                } else {
                    bsync(fd, &O, RKNPU_MEM_SYNC_FROM_DEVICE);
                    
                    int output_changed = 0;
                    for (int i = 0; i < M*N; i++) {
                        if (out_cpu[i] != baseline_out[i]) {
                            output_changed = 1;
                            break;
                        }
                    }
                    
                    if (output_changed) {
                        hit_count++;
                        printf("HIT! Output changed compared to baseline!\n");
                        printf("   [HIT] Register block=%x offset=%x val=%x caused output modification!\n", b, o, val);
                        for (int m = 0; m < M; m++) {
                            printf("   Row%d: ", m);
                            for (int i = 0; i < 8; i++) printf("%04x ", (uint16_t)out_cpu[m*N + i]);
                            printf("\n");
                        }
                        FILE *hf = fopen("fuzz_hits.txt", "a");
                        if (hf) {
                            fprintf(hf, "HIT: block=%x offset=%x val=%x\n", b, o, val);
                            for (int m = 0; m < M; m++) {
                                fprintf(hf, "   Row%d: ", m);
                                for (int i = 0; i < N; i++) fprintf(hf, "%04x ", (uint16_t)out_cpu[m*N + i]);
                                fprintf(hf, "\n");
                            }
                            fclose(hf);
                        }
                    } else {
                        printf("OK (inert)\n");
                    }
                }
            }
        }
        start_o = 0;
        start_val_idx = 0;
    }
    
    unlink(PROGRESS_FILE);
    
    printf("[fuzzer] Finished! Total runs: %d, Hits: %d\n", total_runs, hit_count);
    
    free(baseline_out);
    bdestroy(fd, &W); bdestroy(fd, &O);
    free(A); free(B); free(ref);
    ork_npu_free(ctx);
    return 0;
}
