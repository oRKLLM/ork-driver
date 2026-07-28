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
#include <stdlib.h>
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
/* ORK_SUBMIT_ONLY=1: print only SUBMIT/subcore/task headers (the multi-core pattern), skip the verbose
 * hexdumps — so capturing a full model inference stays small. Set in init(). */
static int g_submit_only = 0;

static int (*real_ioctl)(int, unsigned long, ...) = NULL;
static void *(*real_mmap)(void *, size_t, int, int, int, off_t) = NULL;

__attribute__((constructor))
static void init(void) {
    real_ioctl = (int (*)(int, unsigned long, ...))dlsym(RTLD_NEXT, "ioctl");
    real_mmap  = (void *(*)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap");
    g_submit_only = getenv("ORK_SUBMIT_ONLY") != NULL;
    fprintf(stderr, "[rknpu_dump] loaded%s\n", g_submit_only ? " (submit-only)" : "");
}

static struct ent *by_handle(uint32_t h) { for (int i=0;i<nent;i++) if (tab[i].handle==h) return &tab[i]; return NULL; }
static struct ent *by_off(uint64_t o)    { for (int i=0;i<nent;i++) if (tab[i].off==o)    return &tab[i]; return NULL; }
static struct ent *by_dma(uint64_t d)    { for (int i=0;i<nent;i++) if (tab[i].dma==d)    return &tab[i]; return NULL; }
/* range lookup: find the buffer CONTAINING d (task regcmds live at offsets into one regcmd buffer,
 * so exact-base by_dma misses task[1..3]); returns the entry and sets *off to the byte offset. */
static struct ent *by_dma_range(uint64_t d, uint64_t *off) {
    uint32_t d32 = (uint32_t)d;
    for (int i=0;i<nent;i++) {
        uint32_t dma32 = (uint32_t)tab[i].dma;
        if (d32 >= dma32 && d32 < dma32 + tab[i].size) {
            *off = d32 - dma32;
            return &tab[i];
        }
    }
    return NULL;
}

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
    if (g_submit_only) return;
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
                uint64_t roff=0; struct ent *re = by_dma_range(t[i].regcmd_addr, &roff);
                /* regcfg_amount = number of 64-bit (value,target) register writes */
                int rcwords = (int)t[i].regcfg_amount * 2 + 16;
                if (re && re->cpu) hexwords("regcmd", (uint32_t *)((char*)re->cpu + roff), rcwords);
                else fprintf(stderr, "  (regcmd buffer for dma=0x%llx not mapped)\n", (unsigned long long)t[i].regcmd_addr);
                /* RKDUMP_MM: dump the FIRST matmul matching K/N — regcmd + weight(B) + A + real output(C)
                 * buffer regions (real_ioctl already ran, so C holds the true result) → enables a bit-exact
                 * + timed replay on ork AND reverse of rkllm's weight TILING (known logical W vs dumped bytes). */
                if (re && re->cpu && getenv("RKDUMP_MM")) {
                    static uint32_t best_M = 0;   /* keep the LARGEST-M matching program (overwrite files) */
                    {
                        uint32_t *rw = (uint32_t *)((char*)re->cpu + roff);
                        uint32_t K=0,N=0,M=0,wadr=0,aadr=0,cadr=0,wbytes=0,wstride=0;
                        for (int k=0;k+1<rcwords;k+=2){ uint32_t o=rw[k]&0xffff, v=(rw[k]>>16)|((rw[k+1]&0xffff)<<16);
                            if(o==0x1024)K=v&0xffff; else if(o==0x1038)N=v&0xffff; else if(o==0x102c)M=v&0xffff;
                            else if(o==0x1110)wadr=v; else if(o==0x1070)aadr=v; else if(o==0x1030)wbytes=v;
                            else if(o==0x1034)wstride=v; else if(o==0x4020)cadr=v; }
                        /* weight is STRIDED: 0x1034=row stride (full N), tile is N-wide -> memory SPAN =
                         * (K-1)*stride + N (int8 bytes), far larger than 0x1030=K*N. Dump the whole span so
                         * a replay's strided fetch stays in-bounds (this was the errno-110 wedge). */
                        uint32_t wspan = (wstride>=N && K>0) ? (K-1)*wstride + N : wbytes;
                        wbytes = wspan;
                        int wantK=getenv("RKDUMP_MM_K")?atoi(getenv("RKDUMP_MM_K")):3584;
                        int wantN=getenv("RKDUMP_MM_N")?atoi(getenv("RKDUMP_MM_N")):1216;
                        if ((int)K==wantK && (int)N==wantN && M>best_M) { best_M=M;
                            fprintf(stderr,"[RKDUMP_MM] MATCH M=%u K=%u N=%u wadr=0x%x aadr=0x%x cadr=0x%x wbytes=%u\n",M,K,N,wadr,aadr,cadr,wbytes);
                            FILE*f=fopen("/tmp/mm_regcmd.txt","w"); if(f){ for(int k=0;k<rcwords;k++)fprintf(f,"%08x ",rw[k]); fprintf(f,"\n"); fclose(f);}
                            { FILE*mf=fopen("/tmp/mm_meta.txt","w"); if(mf){ fprintf(mf,"M %u\nK %u\nN %u\nwbytes %u\n",M,K,N,wbytes); fclose(mf);} }
                            uint64_t off; struct ent*wb=by_dma_range(wadr,&off);
                            if(wb&&wb->cpu&&wbytes){ FILE*g=fopen("/tmp/mm_weight.bin","wb"); if(g){fwrite((char*)wb->cpu+off,1,wbytes,g);fclose(g);} fprintf(stderr,"  weight %u B @+0x%llx of handle dma=0x%llx\n",wbytes,(unsigned long long)off,(unsigned long long)wb->dma);}
                            else fprintf(stderr,"  weight buffer 0x%x NOT mapped\n",wadr);
                            struct ent*ab=by_dma_range(aadr,&off);
                            if(ab&&ab->cpu){ FILE*g=fopen("/tmp/mm_A.bin","wb"); if(g){fwrite((char*)ab->cpu+off,1,(size_t)M*K,g);fclose(g);} }
                            struct ent*cb=by_dma_range(cadr,&off);
                            if(cb&&cb->cpu){ FILE*g=fopen("/tmp/mm_C.bin","wb"); if(g){fwrite((char*)cb->cpu+off,1,(size_t)M*N*4,g);fclose(g);} }
                            fprintf(stderr,"  dumped regcmd+weight+A+C to /tmp/mm_*.{txt,bin} (best M=%u)\n",M);
                            /* CHAIN WALK: follow the in-regcmd PC-chain (0x0101:0x0010 next-addr / 0x0014 amount)
                             * to capture EVERY chained task's regcmd + per-task meta — the full M-fold K-slice
                             * accumulate chain, not just task0. Each linked regcmd is another buffer we've mapped. */
                            { FILE*cm=fopen("/tmp/mm_chain_meta.txt","w");
                              uint32_t *tw=rw; int twords=rcwords; int ti=0;
                              while(ti<32){
                                uint32_t naddr=0,namt=0,tK=0,tM=0,tca=0,taa=0,tde=0,tcb=0,tksl=0;
                                for(int k=0;k+1<twords;k+=2){ uint32_t o=tw[k]&0xffff, blk=(tw[k+1]>>16)&0xffff, v=(tw[k]>>16)|((tw[k+1]&0xffff)<<16);
                                    if(blk==0x101&&o==0x0010)naddr=v; else if(blk==0x101&&o==0x0014)namt=v;
                                    else if(blk==0x201&&o==0x1024){tK=v&0xffff; tksl=v&0xffff;} else if(blk==0x201&&o==0x102c)tM=v;
                                    else if(blk==0x1001&&o==0x4020)tca=v; else if(blk==0x201&&o==0x1070)taa=v;
                                    else if(blk==0x201&&o==0x1044)tde=v; else if(blk==0x201&&o==0x1040)tcb=v; }
                                char fn[64]; snprintf(fn,sizeof fn,"/tmp/mm_chain_%d.txt",ti);
                                FILE*cf=fopen(fn,"w"); if(cf){ int nw=twords<224?twords:224; for(int k=0;k<nw;k++)fprintf(cf,"%08x ",tw[k]); fclose(cf);}
                                if(cm)fprintf(cm,"task %d: Kfield=%u M=%u Aadr=0x%x Cadr=0x%x DATA_ENTRIES=%u CBUF_CON0=0x%x next=0x%x namt=%u\n",ti,tK,tM,taa,tca,tde,tcb,naddr,namt);
                                if(!naddr) break;
                                uint64_t noff; struct ent*nb=by_dma_range(naddr,&noff);
                                if(!(nb&&nb->cpu)){ if(cm)fprintf(cm,"  next 0x%x NOT mapped — chain truncated\n",naddr); break; }
                                size_t avail=(nb->size>noff)?(nb->size-noff)/4:0; twords=avail<512?(int)avail:512;
                                tw=(uint32_t*)((char*)nb->cpu+noff); ti++;
                              }
                              if(cm)fclose(cm);
                              fprintf(stderr,"  CHAIN: walked %d task(s) -> /tmp/mm_chain_*.txt + mm_chain_meta.txt\n",ti+1);
                            }
                        }
                    }
                }
            }
        } else {
            fprintf(stderr, "  (task descriptor not mapped; dumping all tracked buffers)\n");
        }
        /* dump every tracked buffer's head for correlation (A/B/C/scratch) */
        for (int i=0;i<nent;i++) {
            if (!tab[i].cpu) continue;
            char tag[64]; snprintf(tag, sizeof tag, "handle %u (dma=0x%llx size=%llu)", tab[i].handle, (unsigned long long)tab[i].dma, (unsigned long long)tab[i].size);
            static int cap=-1; if(cap<0){const char*e=getenv("RKDUMP_WORDS"); cap=e?atoi(e):1024;}
            int words = (int)(tab[i].size/4); if (words > cap) words = cap;
            hexwords(tag, (uint32_t *)tab[i].cpu, words);
        }
    }
    return ret;
}
