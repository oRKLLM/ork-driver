/* npu/core/buf.c — DMA buffer lifecycle: MEM_CREATE/destroy, dma-heap import, the weight arena, zero-copy.
 * Part of the dtype-agnostic substrate; interface in npu/core.h. Lifted verbatim from npu.c by the
 * precision split (MODULARIZE_PLAN.md round 1). */
#define _GNU_SOURCE   /* CPU_SET/pthread_setaffinity_np, as npu.c does */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <stdarg.h>
#include <sys/prctl.h>
#include "ork_regs.h"
#include "regcmd_i8.h"
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include "npu/internal.h"
#include "npu/core.h"
#include "npu/core/core.h"

size_t orki_pgup(size_t s){return (s+4095)&~((size_t)4095);}

/* #sram: how many bytes MEM_CREATE actually placed in on-chip SRAM. TRY_ALLOC_SRAM is a *try* —
 * the kernel silently falls back to DRAM when SRAM is exhausted (956 KB total on RK3588), so an
 * experiment that assumes placement is measuring nothing. MEM_CREATE returns the achieved size in
 * args->sram_size; accumulate it so a test can VERIFY rather than assume. */
size_t orki_sram_got;

struct buf orki_bcreate(int fd,size_t size,uint32_t flags,int domain){
    int dom=ork_dom(domain); size_t need=orki_pgup(size);
    /* SRAM failover: if the caller asked for on-chip SRAM (TRY_ALLOC_SRAM) but the NPU has none (orki_sram_total
     * ==0, stock kernel/DTB), drop to DRAM up front — don't even try. If SRAM exists but the alloc faults
     * (SRAM full/contended), retry once in DRAM below. Keeps the async submit path portable. */
    if((flags & RKNPU_MEM_TRY_ALLOC_SRAM) && orki_sram_total==0) flags &= ~RKNPU_MEM_TRY_ALLOC_SRAM;
    /* ORK_RC_UNCACHED=1 (DIAGNOSTIC): drop RKNPU_MEM_CACHEABLE from scratch allocations. The regcmd and
     * task buffers the PC fetches are allocated CACHEABLE (0x403 / 0x40b), so a stale CPU line over them
     * means the NPU's descriptor fetch reads bytes the CPU never flushed -- which would look exactly like
     * the doorbell dispatch failure: mapping valid (zero IOMMU faults measured), PC registers byte-identical
     * to a healthy commit, and the job simply never starts. Uncached makes that impossible; if the stall
     * rate collapses, stale regcmd is the cause. Diagnostic only -- the real fix would be a correct flush,
     * not making every scratch buffer uncached. */
    { static int unc = -1; if(unc < 0) unc = getenv("ORK_RC_UNCACHED") ? 1 : 0;
      if(unc) flags &= ~RKNPU_MEM_CACHEABLE; }
    if(!ork_iova_reserve(dom,need)) return (struct buf){0};   /* proactive: avoid the in-kernel MEM_CREATE fault */
    struct rknpu_mem_create c; memset(&c,0,sizeof c); c.size=need; c.flags=flags; c.core_mask=RKNPU_CORE0_MASK; c.iommu_domain_id=dom;
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_CREATE,&c)){
        if(flags & RKNPU_MEM_TRY_ALLOC_SRAM){   /* SRAM path faulted -> DRAM failover (retry once, same IOVA reservation) */
            memset(&c,0,sizeof c); c.size=need; c.flags=flags & ~RKNPU_MEM_TRY_ALLOC_SRAM; c.core_mask=RKNPU_CORE0_MASK; c.iommu_domain_id=dom;
            if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_CREATE,&c)){perror("CREATE");ork_iova_release(dom,need);return (struct buf){0};}
        } else {
            fprintf(stderr,"CREATE FAIL: errno=%d(%s) size=%zuKB dom=%d flags=0x%x bcreate#%ld dom_iova=%zuMB ceil=%zuMB\n",
                    errno, strerror(errno), need>>10, dom, flags, orki_bcreate_n,
                    (dom>=0&&dom<ORK_IOVA_NDOM?orki_iova_bytes[dom]:0)>>20, ork_iova_ceiling()>>20);
            ork_iova_release(dom,need);return (struct buf){0};}
    }
    struct rknpu_mem_map m; memset(&m,0,sizeof m); m.handle=c.handle;
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_MAP,&m)){perror("MAP");ork_iova_release(dom,need);return (struct buf){0};}
    void*p=mmap(NULL,c.size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,m.offset);
    if(p==MAP_FAILED){perror("mmap");ork_iova_release(dom,need);return (struct buf){0};}
    struct buf b; memset(&b,0,sizeof b); b.handle=c.handle; b.dma=c.dma_addr; b.obj=c.obj_addr; b.cpu=p; b.size=c.size; b.domain=dom;
    orki_live_add(fd,b.handle,b.obj);
    orki_sram_got += (size_t)c.sram_size;   /* #sram: achieved, not requested */
    orki_bcreate_n++;
    return b;
}

/* Ensure the persistent SDP-op scratch (a/b/out) is each >= sz bytes; (re)allocate only when it must grow.
 * Reused across every ewmul/add call so the per-op MEM_CREATE/MEM_DESTROY churn (which dominates the
 * standalone-op cost and fragments the IOVA window) is paid ONCE, not per call. 0 on success, -1 on alloc
 * failure (caller returns an error -> its caller falls back to CPU). Freed in ork_npu_free. */
int orki_ppu_scratch3(ork_npu *c,size_t sz){
    if(sz<4096) sz=4096;
    if(c->ppu_sz>=sz && c->ppu_a.cpu && c->ppu_b.cpu && c->ppu_o.cpu && c->ppu_a.domain==c->dom_active) return 0;   /* realloc if the active domain changed (persistent scratch must live in c->dom_active) */
    orki_bdestroy(c->fd,&c->ppu_a); orki_bdestroy(c->fd,&c->ppu_b); orki_bdestroy(c->fd,&c->ppu_o);
    c->ppu_a=orki_bcreate(c->fd,sz,0x403,c->dom_active); if(!c->ppu_a.cpu){c->ppu_sz=0;return -1;}
    c->ppu_b=orki_bcreate(c->fd,sz,0x403,c->dom_active); if(!c->ppu_b.cpu){orki_bdestroy(c->fd,&c->ppu_a);c->ppu_sz=0;return -1;}
    c->ppu_o=orki_bcreate(c->fd,sz,0x403,c->dom_active); if(!c->ppu_o.cpu){orki_bdestroy(c->fd,&c->ppu_a);orki_bdestroy(c->fd,&c->ppu_b);c->ppu_sz=0;return -1;}
    c->ppu_sz=sz; return 0;
}

int orki_dmaheap_open(void){
    if(orki_dmaheap_fd==-1){
        const char *h=getenv("ORK_DMA_HEAP"); char path[64];
        snprintf(path,sizeof path,"/dev/dma_heap/%s", h&&*h?h:"system");
        orki_dmaheap_fd=open(path,O_RDWR|O_CLOEXEC); if(orki_dmaheap_fd<0) orki_dmaheap_fd=-2;
    }
    return orki_dmaheap_fd>=0 ? orki_dmaheap_fd : -1;
}

void orki_dmabuf_sync(int heap_fd,uint64_t flags){
    if(heap_fd<=0) return; struct dma_buf_sync s={.flags=flags}; ioctl(heap_fd,DMA_BUF_IOCTL_SYNC,&s);
}

struct buf orki_bimport_f(int fd,size_t size,int domain,uint32_t memflags){
    int tr=orki_imp_trace();
    int hf=orki_dmaheap_open(); if(hf<0) return (struct buf){0};
    size_t sz=orki_pgup(size);
    int dtr=ork_dom(domain);   /* domain id for the trace (final `dom` computed post-mmap, same value) */
    if(tr){ fprintf(stderr,"[IMP] bimport dom=%d sz=%zuKB (imp#%ld, dom_bytes=%zuMB): DMA_HEAP_ALLOC...\n",dtr,sz>>10,orki_bimport_n,orki_iova_bytes[dtr>=0&&dtr<ORK_IOVA_NDOM?dtr:0]>>20); fflush(stderr); }
    double _t = orki_load_prof ? ork_now_us() : 0;   /* ORK_LOAD_PROF: per-phase import timing */
    struct dma_heap_allocation_data a; memset(&a,0,sizeof a); a.len=sz; a.fd_flags=O_RDWR|O_CLOEXEC;
    if(ioctl(hf,DMA_HEAP_IOCTL_ALLOC,&a)){ perror("DMA_HEAP_ALLOC"); return (struct buf){0}; }
    if(orki_load_prof){ orki_lp_alloc += ork_now_us()-_t; _t=ork_now_us(); }
    int dbuf=(int)a.fd;
    if(tr){ fprintf(stderr,"[IMP]   alloc ok (fd=%d) -> mmap...\n",dbuf); fflush(stderr); }
    void*p=mmap(NULL,sz,PROT_READ|PROT_WRITE,MAP_SHARED,dbuf,0);
    if(p==MAP_FAILED){ perror("mmap(dmabuf)"); close(dbuf); return (struct buf){0}; }
    if(orki_load_prof){ orki_lp_mmap += ork_now_us()-_t; }
    int dom=ork_dom(domain);
    if(!ork_iova_reserve(dom,sz)){ munmap(p,sz); close(dbuf); return (struct buf){0}; }   /* IOVA wedge guard */
    if(orki_load_prof) _t=ork_now_us();
    if(tr){ fprintf(stderr,"[IMP]   mmap ok -> PRIME_FD_TO_HANDLE...\n"); fflush(stderr); }
    /* MAKE THE TARGET DOMAIN LIVE FIRST. PRIME_FD_TO_HANDLE maps the sg into whatever domain is live
     * at that instant — the ioctl carries no domain — and the MEM_CREATE below names `dom` too LATE,
     * because an already-imported handle is not re-mapped. Without this a weight can be mapped in one
     * domain and used in another: the submit is committed and never completes, with no error
     * (measured: dom_scale_probe imported=1 failed at cycle 0; native packs are unaffected because
     * their MEM_CREATE does the switch and the mapping together). */
    { struct rknpu_action da; memset(&da,0,sizeof da); da.flags=RKNPU_ACT_SET_DOMAIN; da.value=(uint32_t)dom;
      ioctl(fd,DRM_IOCTL_RKNPU_ACTION,&da); }
    struct drm_prime_handle ph; memset(&ph,0,sizeof ph); ph.fd=dbuf; ph.flags=0;
    if(ioctl(fd,DRM_IOCTL_PRIME_FD_TO_HANDLE,&ph)){ perror("PRIME_FD_TO_HANDLE"); ork_iova_release(dom,sz); munmap(p,sz); close(dbuf); return (struct buf){0}; }
    if(orki_load_prof){ orki_lp_prime += ork_now_us()-_t; _t=ork_now_us(); }
    if(tr){ fprintf(stderr,"[IMP]   prime ok (handle=%u) -> MEM_CREATE...\n",ph.handle); fflush(stderr); }
    /* memflags: normally 0 (weights are NPU-DMA'd, need no kernel vmap). SCRATCH task/descriptor buffers pass
     * RKNPU_MEM_KERNEL_MAPPING so the kernel can READ the rknpu_task structs — dropping it (the old flags=0)
     * gave a malformed task and WEDGED the NPU. Only the kernel-read task buffers need it; data buffers pass 0. */
    struct rknpu_mem_create mc; memset(&mc,0,sizeof mc); mc.handle=ph.handle; mc.flags=memflags; mc.size=0; mc.core_mask=RKNPU_CORE0_MASK; mc.iommu_domain_id=dom;
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_CREATE,&mc)){ perror("MEM_CREATE(import)"); ork_iova_release(dom,sz); munmap(p,sz); close(dbuf); return (struct buf){0}; }
    if(orki_load_prof){ orki_lp_create += ork_now_us()-_t; orki_lp_nchunk++; orki_lp_bytes+=sz; }
    struct buf b; memset(&b,0,sizeof b);
    b.handle=mc.handle; b.dma=mc.dma_addr; b.obj=mc.obj_addr; b.cpu=p; b.size=sz; b.heap_fd=dbuf; b.domain=dom;
    orki_live_add(fd,b.handle,b.obj);
    orki_bimport_n++;
    if(tr){ fprintf(stderr,"[IMP]   MEM_CREATE ok dma=0x%llx -> bimport DONE (imp#%ld dom_bytes=%zuMB)\n",(unsigned long long)b.dma,orki_bimport_n,orki_iova_bytes[dom>=0&&dom<ORK_IOVA_NDOM?dom:0]>>20); fflush(stderr); }
    return b;
}

struct buf orki_bimport(int fd,size_t size,int domain){ return orki_bimport_f(fd,size,domain,0); }   /* weights: NPU-DMA'd, no kernel vmap */

/* Run-SCRATCH allocator (task/regcmd/output buffers). int4-RESIDENT runs keep every domain's WEIGHTS bimported
 * (PRIME_FD); a fresh MEM_CREATE (bcreate) then EINVALs in that domain (MEM_CREATE GEM alloc can't coexist with
 * imported memory in the same iommu domain), which starved the run scratch (mc_ensure mtk_all -> decode -3). So
 * for int4 (c->last_dt==DT_I4) route the small scratch through the SAME import path (bimport) as the weights, so
 * it coexists in-domain. int8/fp16 are UNCHANGED (bcreate). `flags` is subsumed under import (bimport's MEM_CREATE
 * uses flags=0). NOTE: at ork_npu_init/first domain-0 touch last_dt is COLD (not DT_I4) so domain 0's init scratch
 * still bcreate's into the empty domain (fine); the int4 switch only applies once an int4 run is active. */
struct buf orki_bscratch(ork_npu *c,size_t size,int flags,int dom){
    /* int8/fp16 (scratch_import unset): bcreate — the proven path, UNCHANGED. int4 RESIDENT (scratch_import set
     * once weights are bimported): a fresh bcreate GEM can't be placed amid ~GiB of imported SG mappings in the
     * same domain (kernel EINVAL once the domain is heavy — mc_ensure mtk_all -> decode -3). Route scratch through
     * the SAME import path so it coexists. The earlier bimport-scratch WEDGE was because bimport forced MEM_CREATE
     * flags=0, DROPPING RKNPU_MEM_KERNEL_MAPPING (0x8) that the kernel needs to READ task/descriptor buffers ->
     * malformed task. Fixed: carry KERNEL_MAPPING for the kernel-read buffers (requested flags & 0x8); NPU-DMA'd
     * data buffers pass 0. */
    /* #54 int4 RESIDENT (scratch_import set once native-int4 weights are bimported into their domains): a fresh
     * bcreate GEM draws from the limited CONTIGUOUS/CMA pool, which FRAGMENTS under the resident dma-heap weights
     * + the GGUF/orkpack mmap pressure and INTERMITTENTLY EINVALs (mc_ensure mtk_all, at a variable domain 3-8 on
     * fresh boots — nondeterministic = memory pressure, not a clean limit). Route run scratch through the SAME
     * dma-heap import path as the weights (system memory, NO contiguous requirement) so it never competes for the
     * contiguous pool. KERNEL-READ task/descriptor buffers (flags & KERNEL_MAPPING) import WITH that flag so the
     * kernel still gets a vmap to read the rknpu_task structs (the earlier "bimport scratch wedges" was flags=0
     * DROPPING KERNEL_MAPPING; carrying it fixes the malformed-task wedge — bimport_f already supports this).
     * Coherency: bsync auto-routes imported buffers (heap_fd set) to orki_dmabuf_sync (rknpu MEM_SYNC doesn't cover
     * foreign imports). int8/fp16 (scratch_import UNSET) keep bcreate — the proven path, UNCHANGED. */
    /* Run scratch is ALWAYS bcreate. Importing scratch is a PROVEN DEAD END: kernel-read task buffers (0x40b)
     * HARD-WEDGE the board (foreign dma-buf has no kernel vmap -> malformed task; 2026-08-08 + 2026-08-09
     * unreachable/power-cycle), AND importing even the NPU-DMA'd (0x403) scratch corrupts subsequent MEM_CREATE
     * (errno 14 EFAULT on the next bcreate; 2026-08-09). The contiguous-pool fragmentation under mmap pressure is
     * instead addressed by REDUCING the footprint (lean/metadata-only GGUF — run from the int4 orkpack directly),
     * not by importing scratch. int8/fp16/int4 all bcreate here. */
    (void)c;
    /* #54 BIMPORT-ONLY MULTI-DOMAIN (ORK_BIMPORT_DOM, opt-in). On the current board boot, native bcreate
     * (MEM_CREATE GEM alloc) into a NON-0 iommu domain fails outright with EINVAL (test_i4_domains: bcreate#20
     * dom=1, dom_iova=2MB — not exhaustion), so non-0 domains can't be established at all and every multi-domain
     * run wedges downstream. dma-heap bimport into a non-0 domain uses a DIFFERENT MEM_CREATE flavor (imported
     * handle) that may still work where the fresh GEM alloc EINVALs. Route non-0-domain scratch through bimport,
     * carrying KERNEL_MAPPING for the kernel-read task/descriptor buffers (flags&0x8 -> 0x40b) — dropping it was
     * the old import-scratch wedge; bimport_f supports it. Domain 0 + int8/fp16 UNCHANGED (bcreate). */
    { static int bimp=-1; if(bimp<0) bimp=getenv("ORK_BIMPORT_DOM")?1:0;
      if(bimp && ork_dom(dom)>0){ return orki_bimport_f(c->fd,size,dom,(flags & RKNPU_MEM_KERNEL_MAPPING)); } }
    return orki_bcreate(c->fd,size,flags,dom);
}

/* Like bimport but imports an ALREADY-EXISTING dma-buf fd (e.g. one received over SCM_RIGHTS from another
 * process) instead of allocating from the heap. Takes ownership of `dbuf` (bdestroy closes it via heap_fd).
 * This is the cross-process zero-copy primitive behind ork_dma_import_fd (the orkd daemon's data plane). */
struct buf orki_bimport_fd(int fd,int dbuf,size_t size,int domain){
    if(dbuf<0) return (struct buf){0};
    size_t sz=orki_pgup(size);
    void*p=mmap(NULL,sz,PROT_READ|PROT_WRITE,MAP_SHARED,dbuf,0);
    if(p==MAP_FAILED){ perror("mmap(import_fd)"); return (struct buf){0}; }
    int dom=ork_dom(domain);
    if(!ork_iova_reserve(dom,sz)){ munmap(p,sz); return (struct buf){0}; }
    /* MAKE THE TARGET DOMAIN LIVE FIRST. PRIME_FD_TO_HANDLE maps the sg into whatever domain is live
     * at that instant — the ioctl carries no domain — and the MEM_CREATE below names `dom` too LATE,
     * because an already-imported handle is not re-mapped. Without this a weight can be mapped in one
     * domain and used in another: the submit is committed and never completes, with no error
     * (measured: dom_scale_probe imported=1 failed at cycle 0; native packs are unaffected because
     * their MEM_CREATE does the switch and the mapping together). */
    { struct rknpu_action da; memset(&da,0,sizeof da); da.flags=RKNPU_ACT_SET_DOMAIN; da.value=(uint32_t)dom;
      ioctl(fd,DRM_IOCTL_RKNPU_ACTION,&da); }
    struct drm_prime_handle ph; memset(&ph,0,sizeof ph); ph.fd=dbuf; ph.flags=0;
    if(ioctl(fd,DRM_IOCTL_PRIME_FD_TO_HANDLE,&ph)){ perror("PRIME_FD_TO_HANDLE(import_fd)"); ork_iova_release(dom,sz); munmap(p,sz); return (struct buf){0}; }
    struct rknpu_mem_create mc; memset(&mc,0,sizeof mc); mc.handle=ph.handle; mc.flags=0; mc.size=0; mc.core_mask=RKNPU_CORE0_MASK; mc.iommu_domain_id=dom;
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_CREATE,&mc)){ perror("MEM_CREATE(import_fd)"); ork_iova_release(dom,sz); munmap(p,sz); return (struct buf){0}; }
    struct buf b; memset(&b,0,sizeof b);
    b.handle=mc.handle; b.dma=mc.dma_addr; b.obj=mc.obj_addr; b.cpu=p; b.size=sz; b.heap_fd=dbuf; b.domain=dom;
    orki_live_add(fd,b.handle,b.obj);
    orki_bimport_n++;
    return b;
}

struct buf *orki_warena_reserve(ork_npu *c,size_t need,size_t *base){
    size_t a=(need+4095u)&~(size_t)4095u;
    if(c->wchunk_n==0 || c->wchunk_off+a > c->wchunk[c->wchunk_n-1].size){
        if(c->wchunk_n >= (int)(sizeof c->wchunk/sizeof c->wchunk[0])) return NULL;
        const char *e=getenv("ORK_WARENA_CHUNK_MB"); long mb=e?atol(e):1024; if(mb<=0) return NULL;
        size_t csz=(size_t)mb*1024u*1024u; if(a>csz) csz=a;
        struct buf b=orki_bcreate(c->fd,csz,0x403,-1);
        if(!b.cpu){ fprintf(stderr,"[ork] WARNING: weight arena chunk %zuMB alloc failed — falling back to per-buffer\n",csz/1024u/1024u); return NULL; }
        c->wchunk[c->wchunk_n++]=b; c->wchunk_off=0;
    }
    struct buf *ch=&c->wchunk[c->wchunk_n-1];
    *base=c->wchunk_off; c->wchunk_off+=a;
    return ch;
}

/* ---- zero-copy DMA buffers (NPU-coherent, CPU-mapped). A matmul whose A and/or C live in one of
 * these has the regcmd point at it directly — no host gather/writeout memcpy. ork_i8_mm_run detects
 * residency automatically (no API change); the caller just allocates A/C here. ---- */
/* Clean CPU writes -> device for an imported (or ork_dma_alloc) buffer; the bsync the weight fill
 * issues once before the first submit (write-once-read-many weights). size 0 = whole buffer. */
/* Diagnostic only (tools/disk_stream_bench.c): flush `size` bytes of an ork_dma_alloc buffer to the
 * device after a host write (the bsync the streaming fill would issue). Not in the public header. */
/* ork_dma_alloc that requests on-chip SRAM residence (fails over to DRAM if the NPU has no SRAM / it is full).
 * For validating the precompiled/doorbell submit against an SRAM-resident output the CPU polls via dc civac. */
void *ork_dma_alloc(ork_npu *c, size_t size){
    if(!c || c->dma_n >= (int)(sizeof c->dma_tab/sizeof c->dma_tab[0])) return NULL;
    struct buf b=orki_bcreate(c->fd,size,0x401,c->pack_domain); if(!b.cpu) return NULL;
    c->dma_tab[c->dma_n++]=b; return b.cpu;
}

void ork_dma_free(ork_npu *c, void *ptr){
    if(!c||!ptr) return;
    for(int i=0;i<c->dma_n;i++) if(c->dma_tab[i].cpu==ptr){ orki_bdestroy(c->fd,&c->dma_tab[i]); c->dma_tab[i]=c->dma_tab[--c->dma_n]; memset(&c->dma_tab[c->dma_n], 0, sizeof(struct buf)); return; }
}

void *ork_dma_import(ork_npu *c, size_t size){
    if(!c || c->dma_n >= (int)(sizeof c->dma_tab/sizeof c->dma_tab[0])) return NULL;
    ork_dom_prime(c, c->pack_domain);   /* establish a non-0 domain before importing into it (see ork_dom_prime) */
    struct buf b=orki_bimport(c->fd,size,c->pack_domain); if(!b.cpu) return NULL;
    c->dma_tab[c->dma_n++]=b; return b.cpu;
}

void ork_dma_import_free(ork_npu *c, void *ptr){ ork_dma_free(c,ptr); }

void *ork_dma_import_fd(ork_npu *c, int dmabuf_fd, size_t size){
    if(!c || dmabuf_fd<0 || c->dma_n >= (int)(sizeof c->dma_tab/sizeof c->dma_tab[0])) return NULL;
    ork_dom_prime(c, c->pack_domain);
    struct buf b=orki_bimport_fd(c->fd, dmabuf_fd, size, c->pack_domain); if(!b.cpu) return NULL;
    c->dma_tab[c->dma_n++]=b; return b.cpu;
}

struct buf *orki_dma_find(ork_npu *c, const void *p){
    for(int i=0;i<c->dma_n;i++){ char*base=c->dma_tab[i].cpu;
        if((const char*)p>=base && (const char*)p<base+c->dma_tab[i].size) return &c->dma_tab[i]; }
    return NULL;
}

void ork_dma_import_sync(ork_npu *c, void *ptr, size_t size){
    (void)size; struct buf *b=orki_dma_find(c,ptr); if(!b) return;
    /* For imported dma-bufs the dma-buf's own SYNC ioctl flushes the CPU caches (the rknpu MEM_SYNC
     * does not cover foreign imports). END|WRITE = "CPU done writing" -> clean to device. */
    orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);
}

/* the full TO|FROM then TO orki_bsync (the current ork_dma_bsync_to_device pattern), per tile. */
void ork_dma_bsync_to_device(ork_npu *c, void *ptr, size_t size){
    struct buf *b=orki_dma_find(c,ptr); if(!b) return;
    struct rknpu_mem_sync s; memset(&s,0,sizeof s);
    s.obj_addr=b->obj; s.offset=(uint64_t)((char*)ptr-(char*)b->cpu); s.size=size?size:b->size;
    s.flags=RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE; orki_io_ok(c->fd,DRM_IOCTL_RKNPU_MEM_SYNC,&s,"MEM_SYNC(bidi)",0);
    s.flags=RKNPU_MEM_SYNC_TO_DEVICE; orki_io_ok(c->fd,DRM_IOCTL_RKNPU_MEM_SYNC,&s,"MEM_SYNC(to-dev)",0);
}

/* Diagnostic only (tools/dmabuf_fill_probe.c): allocate a registered DMA buffer with a caller-chosen
 * rknpu mem-create flag set, so the probe can A/B the write-combine (0x401) vs cacheable (0x403) fill
 * bandwidth + NPU-read correctness WITHOUT changing the default ork_dma_alloc behavior. Additive; not
 * in the public header. The buffer is registered in dma_tab so ork_i8_mm_run zero-copy + dma_find work. */
void *ork_dma_alloc_flags(ork_npu *c, size_t size, unsigned flags){
    if(!c || c->dma_n >= (int)(sizeof c->dma_tab/sizeof c->dma_tab[0])) return NULL;
    struct buf b=orki_bcreate(c->fd,size,flags,c->pack_domain); if(!b.cpu) return NULL;
    c->dma_tab[c->dma_n++]=b; return b.cpu;
}

void *ork_dma_alloc_sram(ork_npu *c, size_t size){ return ork_dma_alloc_flags(c, size, 0x401 | RKNPU_MEM_TRY_ALLOC_SRAM); }

void ork_dma_clean_to_device(ork_npu *c, void *ptr, size_t size){
    struct buf *b=orki_dma_find(c,ptr); if(!b) return;
    struct rknpu_mem_sync s; memset(&s,0,sizeof s);
    s.obj_addr=b->obj; s.offset=(uint64_t)((char*)ptr-(char*)b->cpu); s.size=size?size:b->size;
    s.flags=RKNPU_MEM_SYNC_TO_DEVICE; orki_io_ok(c->fd,DRM_IOCTL_RKNPU_MEM_SYNC,&s,"MEM_SYNC(to-dev)",0);
}

struct buf orki_bstage_alloc(size_t size){
    int hf=orki_dmaheap_open(); if(hf<0) return (struct buf){0};
    size_t sz=orki_pgup(size);
    struct dma_heap_allocation_data a; memset(&a,0,sizeof a); a.len=sz; a.fd_flags=O_RDWR|O_CLOEXEC;
    if(ioctl(hf,DMA_HEAP_IOCTL_ALLOC,&a)){ perror("DMA_HEAP_ALLOC(stage)"); return (struct buf){0}; }
    int dbuf=(int)a.fd;
    void*p=mmap(NULL,sz,PROT_READ|PROT_WRITE,MAP_SHARED,dbuf,0);
    if(p==MAP_FAILED){ perror("mmap(stage)"); close(dbuf); return (struct buf){0}; }
    struct buf b; memset(&b,0,sizeof b); b.cpu=p; b.size=sz; b.heap_fd=dbuf; return b;
}

int orki_bstage_map(int fd, struct buf*b){
    struct drm_prime_handle ph; memset(&ph,0,sizeof ph); ph.fd=b->heap_fd; ph.flags=0;
    if(ioctl(fd,DRM_IOCTL_PRIME_FD_TO_HANDLE,&ph)){ perror("PRIME(stage)"); return -1; }
    struct rknpu_mem_create mc; memset(&mc,0,sizeof mc); mc.handle=ph.handle; mc.flags=0; mc.size=0; mc.core_mask=RKNPU_CORE0_MASK;
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_CREATE,&mc)){ perror("MEM_CREATE(stage)"); return -1; }
    b->handle=mc.handle; b->dma=mc.dma_addr; b->obj=mc.obj_addr; return 0;
}

void orki_bstage_unmap(int fd, struct buf*b){
    if(!b->obj && !b->handle) return;
    struct rknpu_mem_destroy d; memset(&d,0,sizeof d); d.handle=b->handle; d.obj_addr=b->obj;
    orki_io_ok(fd,DRM_IOCTL_RKNPU_MEM_DESTROY,&d,"MEM_DESTROY",0);   /* a failure here LEAKS IOVA -- count it */
    b->dma=0; b->obj=0; b->handle=0;
}

void orki_bstage_free(struct buf*b){ if(!b->cpu) return; munmap(b->cpu,b->size); if(b->heap_fd>0) close(b->heap_fd); memset(b,0,sizeof *b); }
int orki_is_valid_dma_addr(ork_npu *c, uint32_t addr, const ork_w *w, const struct buf *extra, int extra_n) {
    if (addr == 0) return 0;
    if (c->regcmd.cpu && addr >= c->regcmd.dma && addr < c->regcmd.dma + c->regcmd.size) return 1;
    if (c->task.cpu && addr >= c->task.dma && addr < c->task.dma + c->task.size) return 1;
    if (c->Af.cpu && addr >= c->Af.dma && addr < c->Af.dma + c->Af.size) return 1;
    if (c->Cc.cpu && addr >= c->Cc.dma && addr < c->Cc.dma + c->Cc.size) return 1;
    for (int i = 0; i < ORK_MAXCORE; i++) {
        if (c->mrc[i].cpu && addr >= c->mrc[i].dma && addr < c->mrc[i].dma + c->mrc[i].size) return 1;
        if (c->mtk[i].cpu && addr >= c->mtk[i].dma && addr < c->mtk[i].dma + c->mtk[i].size) return 1;
        if (c->maf[i].cpu && addr >= c->maf[i].dma && addr < c->maf[i].dma + c->maf[i].size) return 1;
        if (c->mcc[i].cpu && addr >= c->mcc[i].dma && addr < c->mcc[i].dma + c->mcc[i].size) return 1;
    }
    if (c->mtk_all.cpu && addr >= c->mtk_all.dma && addr < c->mtk_all.dma + c->mtk_all.size) return 1;
    for (int i = 0; i < c->dma_n; i++) {
        if (c->dma_tab[i].cpu && addr >= c->dma_tab[i].dma && addr < c->dma_tab[i].dma + c->dma_tab[i].size) return 1;
    }
    if (w) {
        if (w->Bb) {
            int num_weights = w->Sn * w->Sk;
            for (int i = 0; i < num_weights; i++) {
                if (w->Bb[i].cpu && addr >= w->Bb[i].dma && addr < w->Bb[i].dma + w->Bb[i].size) return 1;
            }
        }
        if (w->Bf) {
            for (int i = 0; i < w->Sn; i++) {
                if (w->Bf[i].cpu && addr >= w->Bf[i].dma && addr < w->Bf[i].dma + w->Bf[i].size) return 1;
            }
        }
        /* fp16 CONTIG (Task #50): the contiguous concatenated weight (all K-slices in ONE buffer). Without this
         * clause a valid Bbc.dma+offset weight base was FALSE-flagged "wild/unallocated" -> validate_regcmd failed
         * -> CONTIG refused -> fell back to the concurrent per-slice path that wedges. Bbc.cpu==0 when unused. */
        if (w->Bbc.cpu && addr >= w->Bbc.dma && addr < w->Bbc.dma + w->Bbc.size) return 1;
        for (int i = 0; i < 3; i++) if (w->Bgap[i].cpu && addr >= w->Bgap[i].dma && addr < w->Bgap[i].dma + w->Bgap[i].size) return 1;   /* CONTIG GAP-stagger filler buffers */
    }
    if (extra && extra_n > 0) {
        for (int i = 0; i < extra_n; i++) {
            if (extra[i].cpu && addr >= extra[i].dma && addr < extra[i].dma + extra[i].size) return 1;
        }
    }
    return 0;
}

/* CLIENT: allocate a dma-heap buffer of `size` bytes, mmap it R/W, and (via DMA_BUF_SYNC_START) ready it for
 * CPU fill. Returns the dma-buf fd (>=0) and sets *ptr to the mapping; -1 on failure. No NPU touched. The
 * caller fills *ptr then calls ork_dmabuf_seal(fd) to flush before passing the fd to the daemon. */
int ork_dmabuf_alloc(size_t size, void **ptr){
    int hf=orki_dmaheap_open(); if(hf<0) return -1;
    size_t sz=orki_pgup(size);
    struct dma_heap_allocation_data a; memset(&a,0,sizeof a); a.len=sz; a.fd_flags=O_RDWR|O_CLOEXEC;
    if(ioctl(hf,DMA_HEAP_IOCTL_ALLOC,&a)){ perror("DMA_HEAP_ALLOC(client)"); return -1; }
    int dbuf=(int)a.fd;
    void*p=mmap(NULL,sz,PROT_READ|PROT_WRITE,MAP_SHARED,dbuf,0);
    if(p==MAP_FAILED){ perror("mmap(client dmabuf)"); close(dbuf); return -1; }
    orki_dmabuf_sync(dbuf,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
    if(ptr) *ptr=p;
    return dbuf;
}

void ork_dmabuf_seal(int dbuf){ if(dbuf>=0) orki_dmabuf_sync(dbuf,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE); }

/* See the declaration in npu/internal.h. B is [K][N] row-major raw codes (int4 values live in int8 slots,
 * so one loop serves both dtypes). Parallel over M; each row writes a disjoint slice of C. */
void orki_cpu_gemm_i32(int M,int K,int N,const int8_t *A,const int8_t *B,int32_t *C){
    /* k-OUTER on purpose: B is then streamed contiguously (row k is N contiguous bytes) while the C row —
     * N int32, 4 KiB at N=1024 — stays resident in L1 across the whole k loop. The n-blocked alternative
     * keeps accumulators in registers but re-reads all of B once per n-block, which is far worse.
     *
     * The inner loop is an AXPY: crow[n] += a * wrow[n]. Products fit int16 (127*127 = 16129), so widening
     * multiply-accumulate (vmlal_n_s16) goes straight from int8 lanes to the int32 accumulator with no
     * intermediate rounding — this stays BIT-EXACT with the scalar version and with the NPU's own int32
     * MAC, which is the whole reason the offline path can be trusted. k is unrolled by 4 so one
     * load/store of the accumulators serves four rows of B. */
    #pragma omp parallel for schedule(static) if(M>1)
    for(int m=0;m<M;m++){
        const int8_t *arow=A+(size_t)m*K;
        int32_t *crow=C+(size_t)m*N;
        memset(crow,0,(size_t)N*sizeof *crow);
        int k=0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        for(;k+4<=K;k+=4){
            const int16_t a0=arow[k],a1=arow[k+1],a2=arow[k+2],a3=arow[k+3];
            if(!(a0|a1|a2|a3)) continue;
            const int8_t *w0=B+(size_t)k*N,*w1=w0+N,*w2=w1+N,*w3=w2+N;
            int n=0;
            for(;n+16<=N;n+=16){
                int32x4_t c0=vld1q_s32(crow+n),   c1=vld1q_s32(crow+n+4);
                int32x4_t c2=vld1q_s32(crow+n+8), c3=vld1q_s32(crow+n+12);
                #define ORK_AXPY(WP,AV) do{ \
                    const int8x16_t v=vld1q_s8((WP)+n); \
                    const int16x8_t lo=vmovl_s8(vget_low_s8(v)), hi=vmovl_s8(vget_high_s8(v)); \
                    c0=vmlal_n_s16(c0,vget_low_s16(lo),(AV)); c1=vmlal_n_s16(c1,vget_high_s16(lo),(AV)); \
                    c2=vmlal_n_s16(c2,vget_low_s16(hi),(AV)); c3=vmlal_n_s16(c3,vget_high_s16(hi),(AV)); \
                }while(0)
                if(a0) ORK_AXPY(w0,a0);
                if(a1) ORK_AXPY(w1,a1);
                if(a2) ORK_AXPY(w2,a2);
                if(a3) ORK_AXPY(w3,a3);
                #undef ORK_AXPY
                vst1q_s32(crow+n,c0);    vst1q_s32(crow+n+4,c1);
                vst1q_s32(crow+n+8,c2);  vst1q_s32(crow+n+12,c3);
            }
            for(;n<N;n++) crow[n]+=a0*(int32_t)w0[n]+a1*(int32_t)w1[n]+a2*(int32_t)w2[n]+a3*(int32_t)w3[n];
        }
#endif
        for(;k<K;k++){                                  /* tail (and the whole loop without NEON) */
            const int32_t a=arow[k];
            if(!a) continue;
            const int8_t *wrow=B+(size_t)k*N;
            for(int n=0;n<N;n++) crow[n]+=a*(int32_t)wrow[n];
        }
    }
}

/* OFFLINE chain/stream: the chain and stream entry points exist to amortize SUBMITS across several
 * matmuls. With no device there is nothing to amortize, so run each task exactly on the CPU. Factored
 * here rather than repeated at each entry point — three copies of the same loop is how the precision
 * modules drifted apart in the first place. Returns -1 if any task is not an offline weight. */
int orki_cpu_chain_i4(int S, const ork_mm_task_i4 *t){
    for(int i=0;i<S;i++){ const ork_w *w=t[i].w; if(!w||!w->cpu_codes) return -1;
        orki_cpu_gemm_i32(t[i].M,w->K,w->N,t[i].A,w->cpu_codes,t[i].C); }
    return 0;
}
/* Per-group accumulate on the CPU, shared by the int4 and int8 grouped run paths.
 *
 * Per-group scales cannot factor out of the K-sum, so each group's int32 partial is scaled as it is
 * produced and summed in fp32 — the same arithmetic the doorbell drain performs, with no device. The MAC
 * stays int32 WITHIN a group (exactly as the hardware does) and takes ONE fp32 scale per (row, group,
 * channel); a per-element float multiply would be both slower and a different rounding.
 *
 * DTYPE-AGNOSTIC by construction, and that is the point: the codes are int8 either way. An int4 weight and
 * an int4 weight inflated into int8 containers hold the SAME values, so grouping is a property of the SCALE
 * LAYOUT, not of the weight width. Sharing one kernel is what lets W4A8 be measured against W4A4 without a
 * second copy of this loop drifting from the first.
 *
 * aScale[m*Sk+g], bScale[g*N+n] — the shipped layouts. */
int orki_cpu_gemm_grouped(int M,int K,int N,int G,const int8_t *A,const int8_t *codes,
                          const float *aScale,const float *bScale,float *C){
    if(M<1||K<1||N<1||G<1||(K%G)||!A||!codes||!aScale||!bScale||!C) return -1;
    const int Sk=K/G;
    #pragma omp parallel for schedule(static) if(M>1)
    for(int m=0;m<M;m++){
        float *crow=C+(size_t)m*N;
        for(int n=0;n<N;n++) crow[n]=0.0f;
        int32_t *acc=malloc((size_t)N*sizeof *acc);
        if(!acc) continue;
        for(int g=0;g<Sk;g++){
            const float as=aScale[(size_t)m*Sk+g];
            const float *bs=bScale+(size_t)g*N;
            orki_cpu_gemm_i32(1,G,N,A+(size_t)m*K+(size_t)g*G,codes+(size_t)g*G*N,acc);
            for(int n=0;n<N;n++) crow[n]+=(float)acc[n]*as*bs[n];
        }
        free(acc);
    }
    return 0;
}

/* OFFLINE codes, read-only, or NULL for a device-resident weight. Exists so a test can assert that an
 * offline load recovered EXACTLY what was packed — the un-tilers are index arithmetic, and index arithmetic
 * that is subtly wrong still yields plausible weights (see test_offline_load). Not a production accessor. */
const int8_t *ork_w_codes(const ork_w *w){ return w ? w->cpu_codes : NULL; }

/* RESIDENT IOVA bytes a weight of (K,N) will occupy once loaded at resident width `wbits` — 4 for native
 * int4 nibbles, 8 for int8 containers (which is what an int4-ON-DISK i4a8 / rot_a8 weight INFLATES to at
 * load). Includes the full-K Bf companion exactly when the loaders build one, and the per-tile page padding
 * they actually allocate.
 *
 * SINGLE SOURCE OF TRUTH, and it exists because the alternative failed. The consumer (a domain sizer) used
 * to re-derive this from its own table, which meant two models of the same allocation in two repos with no
 * gate to catch drift -- and they drifted twice at once: it sized every weight in a MIXED pack at one global
 * precision (missing the 2x on inflated entries, ~3 GiB on a 27B), and it gated Bf at K<=4096 where the
 * loaders use K<=10752 (missing a full-K companion on every 4096<K<=10752 weight, which a 27B is full of).
 * An under-count does not merely mis-plan: the overflowing domain's IOVA allocation fails, and a failed
 * allocation leaks a mapping the kernel cannot reclaim, degrading the board until reboot.
 *
 * So callers must ASK rather than model. want_bf: -1 = follow ORK_NO_BF (the normal case, so a caller
 * cannot disagree with the loader about it), 0/1 = force, which a sizer needs to compare a with-Bf
 * footprint against a RAM budget before deciding whether Bf is affordable at all. */
size_t ork_w_resident_bytes(ork_npu *c, int K, int N, int wbits, int want_bf){
    if(K<=0 || N<=0) return 0;
    const int NMAX = (c && c->soc) ? c->soc->nmax : 8192;
    const int KS   = (wbits==8) ? 1024 : ORK_I4_KS;      /* the loaders' K-slice for each resident form */
    const int Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t tot=0;
    for(int ns=0;ns<Sn;ns++){ int n0=ns*NMAX, Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){ int k0=ks*KS, Kp=(K-k0<KS)?(K-k0):KS;
        tot += orki_pgup((size_t)Kp*Nc/(wbits==8?1:2)); } }
    /* Bf: a full-K rebuild per N-slice, built ONLY on the int8-resident loaders and only for K<=10752. */
    const int bf = (want_bf < 0) ? (getenv("ORK_NO_BF") ? 0 : 1) : (want_bf ? 1 : 0);
    if(wbits==8 && K<=10752 && bf)
        for(int ns=0;ns<Sn;ns++){ int n0=ns*NMAX, Nc=(N-n0<NMAX)?(N-n0):NMAX;
            tot += orki_pgup((size_t)K*Nc); }
    return tot;
}

int orki_cpu_chain_i8(int S, const ork_mm_task_i8 *t){
    for(int i=0;i<S;i++){ const ork_w *w=t[i].w; if(!w||!w->cpu_codes) return -1;
        orki_cpu_gemm_i32(t[i].M,w->K,w->N,t[i].A,w->cpu_codes,t[i].C); }
    return 0;
}

/* Resident GEOMETRY of a packed weight. The wcache in ggml-ork is keyed by the source tensor's data
 * pointer, and a FUSED group weight is stored under its first member's pointer -- so one key can be asked
 * for two different shapes (per-tensor at decode, concatenated at prefill). Without a way to ask what a
 * cached weight actually IS, a consumer silently used the wrong one and the matmul emitted nothing
 * (ork-driver#4). This is that question, and it is dtype-agnostic on purpose: every tier has the collision. */
void ork_w_dims(const ork_w *w, int *K, int *N){
    if(K) *K = w ? w->K : 0;
    if(N) *N = w ? w->N : 0;
}

/* Public entry point for the internal CPU int8 GEMM (see include/ork/run.h). A forward rather than a
 * rename so the ~4 internal call sites stay put and the two cannot drift. */
void ork_i8_mm_run_cpu(int M,int K,int N,const int8_t *A,const int8_t *B,int32_t *C){
    orki_cpu_gemm_i32(M,K,N,A,B,C);
}
