/* npu/i8/pack.c — int8 weight packing, loading, import/adopt, persist/dump, resident KV, and the streaming stage/pool.
 *
 * Part of the int8 (W8A8) datapath, the driver's production path. Lifted verbatim from npu.c by the
 * precision split (MODULARIZE_PLAN.md round 1); interface types in npu/internal.h, substrate in npu/core.h. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include "ork_regs.h"
#include "regcmd_i8.h"
#include "regcmd_i4.h"
#include "regcmd_silu.h"
#include "regcmd_ewmul.h"
#include "regcmd_fold_refs.h"
#include "npu/internal.h"
#include "npu/core.h"
#include "npu/i8/i8.h"
#include "spine_kernels.h"

int ork_npu_synth_i8_dump(ork_npu *c, int mc, int K, int N, unsigned *out, int outn){
    if(!c || outn < REGCMD_I8_N) return -2;
    if(getenv("ORK_MFOLD")){ orki_synth_i8_mfold((uint32_t*)out, mc, K, N, 0x1000000u, 0x2000000u, 0x3000000u, c->soc->cbuf_elems); return REGCMD_I8_N; }
    int sched = (K==1024 || K==512);
    orki_synth_i8((uint32_t*)out, mc, K, N, 0x1000000u, 0x2000000u, 0x3000000u, sched, c->soc->cbuf_elems, 0);
    return REGCMD_I8_N;
}

void orki_tile_i8_range(int lo,int hi,void *a){
    struct tile_i8_arg *t=a;
    for(int nt=lo;nt<hi;nt++)for(int kt=0;kt<t->KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
        t->bb[(size_t)nt*t->KT*1024+(size_t)kt*1024+(size_t)nl*32+kk]=
            t->Bi[(size_t)(t->k0+kt*32+kk)*t->N+(t->n0+nt*32+nl)];
}

ork_w *ork_mm_pack_i8(ork_npu *c,int K,int N,const int8_t *B){
    if(c && c->daemon){ uint64_t id=orkd_pack_i8(c->daemon,K,N,B); if(!id) return NULL; ork_w *w=calloc(1,sizeof *w); if(!w) return NULL; w->is_orkd=1; w->orkd_id=id; w->K=K; w->N=N; w->dtype=DT_I8; w->domain=ork_dom(c->pack_domain); return w; }   /* Path B: pack resident in the daemon (remember the domain so runs carry it) */
    return orki_pack(c,K,N,B,DT_I8);  }

ork_kv_resident *ork_kv_resident_alloc(ork_npu *c, int HD, int Lmax){
    if(!c || HD%32 || Lmax%32 || Lmax<32 || Lmax>c->soc->nmax) return NULL;  /* v1: K^T single N-tile */
    int Kp=512;
    if(c->daemon){   /* orkd: the daemon allocs the resident K^T/V; we mirror the two resident weight ids (for
                      * ORKD_CHAIN) + keep the kv id for ork_kv_append. */
        uint64_t wkt_id=0, wv_id=0, kvid=orkd_kv_alloc(c->daemon, HD, Lmax, &wkt_id, &wv_id);
        if(!kvid) return NULL;
        ork_kv_resident *kv=calloc(1,sizeof *kv); ork_w *wkt=calloc(1,sizeof *wkt), *wv=calloc(1,sizeof *wv);
        if(!kv||!wkt||!wv){ free(kv); free(wkt); free(wv); return NULL; }
        wkt->is_orkd=1; wkt->orkd_id=wkt_id; wkt->K=Kp;   wkt->N=Lmax; wkt->dtype=DT_I8; wkt->domain=ork_dom(c->pack_domain);
        wv->is_orkd=1;  wv->orkd_id=wv_id;  wv->K=Lmax; wv->N=HD;   wv->dtype=DT_I8; wv->domain=ork_dom(c->pack_domain);
        kv->wkt=wkt; kv->wv=wv; kv->HD=HD; kv->Lmax=Lmax; kv->Kp=Kp; kv->orkd_kv=kvid;
        return kv;
    }
    int8_t *zk=calloc((size_t)Kp*Lmax,1), *zv=calloc((size_t)Lmax*HD,1);
    if(!zk||!zv){ free(zk); free(zv); return NULL; }
    ork_w *wkt=ork_mm_pack_i8(c,Kp,Lmax,zk), *wv=ork_mm_pack_i8(c,Lmax,HD,zv);
    free(zk); free(zv);
    if(!wkt||!wv){ if(wkt)ork_w_free(wkt); if(wv)ork_w_free(wv); return NULL; }
    ork_kv_resident *kv=calloc(1,sizeof *kv);
    if(!kv){ ork_w_free(wkt); ork_w_free(wv); return NULL; }
    kv->wkt=wkt; kv->wv=wv; kv->HD=HD; kv->Lmax=Lmax; kv->Kp=Kp;
    return kv;
}

int ork_kv_append(ork_npu *c, ork_kv_resident *kv, int key, const int8_t *kcol, const int8_t *vrow){
    if(!c || !kv || key<0 || key>=kv->Lmax || !kcol || !vrow) return -2;
    if(c->daemon) return orkd_kv_append(c->daemon, kv->orkd_kv, key, kv->HD, kcol, vrow);   /* orkd: daemon writes the tile bytes */
    int HD=kv->HD, Kp=kv->Kp, Lmax=kv->Lmax, KS=orki_int8_ks(c);
    /* The M=1 matmul reads the FULL-K blob Bf when present (bdma = Bf?Bf:Bb), so Bf is authoritative; Bb is the
     * K-sliced fallback. Update BOTH. Bf is a single full-K tile per N-slice (Sn==1 here): KTf=K/32, layout
     * bb[nt*KTf*1024 + kt*1024 + nl*32 + kk] (nt=n/32,kt=k/32,nl=n%32,kk=k%32). */
    /* --- K^T [Kp,Lmax]: element [k][n=key], k<HD (k>=HD stays 0). Bf single tile KTf=Kp/32; Bb Sk==1 same. --- */
    { int KTf=Kp/32, nt=key/32, nl=key%32;
      if(kv->wkt->Bf){ int8_t *bf=(int8_t*)kv->wkt->Bf[0].cpu;
        for(int k=0;k<HD;k++) bf[(size_t)nt*KTf*1024 + (size_t)(k/32)*1024 + (size_t)nl*32 + (k%32)] = kcol[k];
        orki_bsync(c->fd, &kv->wkt->Bf[0], RKNPU_MEM_SYNC_TO_DEVICE); }
      int8_t *bb=(int8_t*)kv->wkt->Bb[0].cpu;
      for(int k=0;k<HD;k++) bb[(size_t)nt*KTf*1024 + (size_t)(k/32)*1024 + (size_t)nl*32 + (k%32)] = kcol[k];
      orki_bsync(c->fd, &kv->wkt->Bb[0], RKNPU_MEM_SYNC_TO_DEVICE); }
    /* --- V [Lmax,HD]: element [k=key][n=e]. Bf single full-K tile (KTf=Lmax/32); Bb K-sliced tile ks_idx=key/KS. --- */
    { if(kv->wv->Bf){ int KTf=Lmax/32; int8_t *bf=(int8_t*)kv->wv->Bf[0].cpu;
        for(int e=0;e<HD;e++) bf[(size_t)(e/32)*KTf*1024 + (size_t)(key/32)*1024 + (size_t)(e%32)*32 + (key%32)] = vrow[e];
        orki_bsync(c->fd, &kv->wv->Bf[0], RKNPU_MEM_SYNC_TO_DEVICE); }
      int ks_idx=key/KS, k0=ks_idx*KS, Kp_t=(Lmax-k0<KS)?(Lmax-k0):KS, KTt=Kp_t/32, lk=key-k0, kt=lk/32, kk=lk%32;
      int8_t *bb=(int8_t*)kv->wv->Bb[ks_idx].cpu;
      for(int e=0;e<HD;e++) bb[(size_t)(e/32)*KTt*1024 + (size_t)kt*1024 + (size_t)(e%32)*32 + kk] = vrow[e];
      orki_bsync(c->fd, &kv->wv->Bb[ks_idx], RKNPU_MEM_SYNC_TO_DEVICE); }
    return 0;
}

void ork_kv_resident_free(ork_npu *c, ork_kv_resident *kv){ if(!kv)return;
    if(c && c->daemon){ free(kv->wkt); free(kv->wv); free(kv); return; }   /* orkd: mirrors are id-holders; daemon-side reclaimed on disconnect (TODO ORKD_KV_FREE for mid-session release) */
    if(kv->wkt)ork_w_free(kv->wkt); if(kv->wv)ork_w_free(kv->wv); free(kv); }

/* SINGLE-THREADED int8 CPU dump — identical bytes to ork_w_dump_i8_cpu, but tiles inline on the calling
 * thread (NO internal pool). For callers that ALREADY parallelize at a coarser grain (the .orkpack expert
 * convert runs one whole expert per core); using the shared pthread pool there would nest/oversubscribe. */
size_t ork_w_dump_i8_cpu(ork_npu *c, int K, int N, const int8_t *B, void *out, size_t cap){
    if(!c || !B || (K%32) || (N%32)) return 0;
    int KS=orki_int8_ks(c), NMAX=c->soc->nmax;
    int Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t off=0;
    for(int ns=0;ns<Sn;ns++){ int n0=ns*NMAX, Nc=(N-n0<NMAX)?(N-n0):NMAX, NN=Nc/32;
      for(int ks=0;ks<Sk;ks++){ int k0=ks*KS, Kp=(K-k0<KS)?(K-k0):KS, KT=Kp/32; size_t tsz=orki_pgup((size_t)Kp*Nc);
        if(out){ if(off+tsz>cap) return 0;
            int8_t *bb=(int8_t*)out+off; memset(bb,0,tsz);   /* zero the page-pad (matches a fresh dma-buf) */
            struct tile_i8_arg ta={bb,B,KT,k0,n0,N}; ork_parallel_for(NN,orki_tile_i8_range,&ta); }
        off+=tsz; }}
    return off;
}

size_t ork_w_dump_bf_i8_cpu(ork_npu *c, int K, int N, const int8_t *B, void *out, size_t cap){
    if(!c || !B || (K%512) || K>4096 || (N%32)) return 0;
    int NMAX=c->soc->nmax, Sn=(N+NMAX-1)/NMAX, KTf=K/32;
    size_t off=0;
    for(int ns=0;ns<Sn;ns++){ int n0=ns*NMAX, Nc=(N-n0<NMAX)?(N-n0):NMAX, NN=Nc/32; size_t tsz=orki_pgup((size_t)K*Nc);
        if(out){ if(off+tsz>cap) return 0;
            int8_t *bb=(int8_t*)out+off; memset(bb,0,tsz);   /* zero the page-pad (matches a fresh dma-buf) */
            for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KTf;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
                bb[(size_t)nt*KTf*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(n0+nt*32+nl)]; }
        off+=tsz; }
    return off;
}

/* Zero-copy IMPORT variant of ork_mm_load_i8: each resident tile is a dma-buf the NPU reads in place
 * (PRIME import) instead of a MEM_CREATE-alloc'd buffer the blob is memcpy'd into. The bytes still get
 * written once (into the imported mmap) + synced once; the saving is the kernel page allocation, not
 * the host fill (load is from a disk/RAM blob either way). Same blob format / round-trip as load_i8.
 * Falls through to NULL (caller uses ork_mm_load_i8) if import is unavailable. */
ork_w *ork_mm_load_i8(ork_npu *c,int K,int N,const void *blob,size_t n){
    if(K%32 || N%32) return NULL;
    int KS=1024, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t need=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS; need+=orki_pgup((size_t)Kp*Nc);}}
    if(n!=need) return NULL;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I8; w->owns=1; w->domain=ork_dom(c->pack_domain);
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    size_t off=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=orki_bcreate(c->fd,(size_t)Kp*Nc,0x403,w->domain);
        if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) orki_bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; }
        memcpy(b->cpu,(const char*)blob+off,b->size); off+=b->size;
        orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}}
    /* Rebuild Bf (full-K layout) so loaded weights chain + decode like packed ones. The Bb tiles are
     * 32x32 blocks: Bb[ns][ks] holds B[k0+kt*32+kk][n0+nt*32+nl] at [nt][kt][nl][kk]. Bf[ns] re-tiles the
     * full K dimension (KTf=K/32) of the SAME logical B for that N-slice.
     * ONLY build Bf where a full-K submit is actually valid (K%512==0 && K<=4096) — the same envelope
     * orki_run() / run_chain_i8 use. Outside it (e.g. K=1792 ffn_down experts) Bf would never be read and just
     * doubles resident NPU bytes, exhausting the 4 GiB IOMMU window when many experts are loaded. Those
     * weights run via the K-split Bb path (run_i8), which doesn't need Bf. */
    if(K%512==0 && K<=4096 && !getenv("ORK_NO_BF")){ int KTf=K/32; w->Bf=calloc(Sn,sizeof(struct buf)); int ok=1;
        for(int ns=0;ns<Sn && ok;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
            struct buf*bf=&w->Bf[ns]; *bf=orki_bcreate(c->fd,(size_t)K*Nc,0x403,w->domain);
            if(!bf->cpu){ ok=0; break; }                /* IOVA full → give up on Bf */
            int8_t*fb=bf->cpu;
            for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
                const int8_t*sb=(const int8_t*)w->Bb[(size_t)ns*Sk+ks].cpu;
                for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++){
                    int ktf=(k0/32)+kt;   /* full-K tile index */
                    fb[(size_t)nt*KTf*32*32+(size_t)ktf*32*32+nl*32+kk]=
                        sb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]; }}
            orki_bsync(c->fd,bf,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(c->fd,bf,RKNPU_MEM_SYNC_TO_DEVICE);}
        if(!ok){ for(int ns=0;ns<Sn;ns++) orki_bdestroy(c->fd,&w->Bf[ns]); free(w->Bf); w->Bf=NULL; } }
    return w;
}

ork_w *ork_mm_load_i8_import(ork_npu *c,int K,int N,const void *blob,size_t n){
    if(K%32 || N%32) return NULL;
    if(orki_dmaheap_open()<0) return NULL;
    int KS=1024, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t need=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS;(void)n0; need+=orki_pgup((size_t)Kp*Nc);}}
    if(n!=need) return NULL;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I8; w->owns=1; w->domain=ork_dom(c->pack_domain);
    orki_imp_wn++;
    if(orki_imp_trace()){ fprintf(stderr,"[IMP] ===== weight #%ld load_i8_import K=%d N=%d Sk=%d Sn=%d tiles=%d need=%zuMB dom=%d =====\n",orki_imp_wn,K,N,Sk,Sn,Sk*Sn,need>>20,w->domain); fflush(stderr); }
    ork_dom_prime(c, w->domain);   /* establish a non-0 domain with a native anchor BEFORE importing into it */
    if(orki_imp_trace()){ fprintf(stderr,"[IMP]   dom_prime done, entering %s import\n", getenv("ORK_NO_CONSOLIDATE_IMPORT")?"PER-TILE":"CONSOLIDATED-CHUNK"); fflush(stderr); }
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    /* SIZE-BOUNDED CONSOLIDATED IMPORT (default on; ORK_NO_CONSOLIDATE_IMPORT disables): pack this weight's
     * tiles into a HANDFUL of moderate (~ORK_IMPORT_CHUNK_MB, default 16MB) imported dma-buf CHUNKS; each tile
     * is a page-aligned base+offset VIEW (chunk.dma+off) into its chunk, NOT one bimport per tile. This is the
     * middle ground between the two extremes that both fail: (a) one bimport PER TILE — a PC-chained K-split
     * submit (ffn_down, Sk up to 19) then walks Sk separate imported IOMMU mappings and the CDMA chain-walker
     * TIMES OUT (errno=110) in a non-0 domain; (b) one GIANT per-weight orki_bimport (~68MB) — 1 mapping (no chain
     * fault) but the big DMA_HEAP_ALLOC HANGS. Bounded chunks give few mappings per chain (down-proj 68MB / 16MB
     * -> ~3-5 chunks, within the 1.7B's proven-safe ~6) AND proven-safe alloc sizes (per-tile <=8MB never hung).
     * Chunks are bump-filled in tile order (ns outer, ks inner) so a submit's K-slices land in adjacent chunks.
     * ork_mm_free bdestroys every chunk (own_bufs[]). Falls back to per-tile bimport on any alloc failure. */
    int consolidate = !getenv("ORK_NO_CONSOLIDATE_IMPORT");
    if(consolidate){
        size_t chunk_mb = 16; const char*cm=getenv("ORK_IMPORT_CHUNK_MB"); if(cm){ long v=atol(cm); if(v>0) chunk_mb=(size_t)v; }
        size_t chunk_cap = chunk_mb<<20;
        int ntiles=Sk*Sn, cap_chunks=ntiles+1;             /* worst case: one chunk per tile */
        w->own_bufs=calloc(cap_chunks,sizeof(struct buf)); w->n_own_bufs=0;
        struct buf cur; cur.cpu=NULL; size_t coff=0, csz=0;
        int ns,ks; size_t boff=0;                            /* offset into blob (matches need layout) */
        for(ns=0;ns<Sn && consolidate;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;(void)n0;
          for(ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS; size_t ts=orki_pgup((size_t)Kp*Nc);
            if(!cur.cpu || coff+ts>csz){                    /* need a new chunk */
                if(cur.cpu) orki_dmabuf_sync(cur.heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);
                csz = ts>chunk_cap ? ts : chunk_cap;        /* a single tile never exceeds cap in practice */
                cur = orki_bimport(c->fd,csz,w->domain);
                if(!cur.cpu){ consolidate=0; break; }
                orki_dmabuf_sync(cur.heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
                w->own_bufs[w->n_own_bufs++]=cur; coff=0;
            }
            struct buf*b=&w->Bb[(size_t)ns*Sk+ks];
            b->handle=cur.handle; b->obj=cur.obj; b->dma=cur.dma+coff; b->cpu=(char*)cur.cpu+coff; b->size=ts;
            double _m=orki_load_prof?ork_now_us():0; memcpy(b->cpu,(const char*)blob+boff,(size_t)Kp*Nc); if(orki_load_prof) orki_lp_memcpy+=ork_now_us()-_m; coff+=ts; boff+=ts;}}
        if(consolidate){ if(cur.cpu) orki_dmabuf_sync(cur.heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE); w->owns=0; }
        else { /* alloc failed mid-way: tear down the chunks we grabbed, fall back to per-tile below */
            for(int i=0;i<w->n_own_bufs;i++) orki_bdestroy(c->fd,&w->own_bufs[i]);
            free(w->own_bufs); w->own_bufs=NULL; w->n_own_bufs=0;
            memset(w->Bb,0,(size_t)Sk*Sn*sizeof(struct buf)); w->owns=1;
        }
    }
    if(!consolidate){
        size_t off=0;
        for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
          for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS;(void)n0;
            struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=orki_bimport(c->fd,(size_t)Kp*Nc,w->domain);
            if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) orki_bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; }
            orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
            memcpy(b->cpu,(const char*)blob+off,(size_t)Kp*Nc); off+=b->size;
            orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
    }
    /* Bf full-K rebuild (same envelope as ork_mm_load_i8): imported too, abandoned on failure. */
    if(K%512==0 && K<=4096 && !getenv("ORK_NO_BF")){ int KTf=K/32; w->Bf=calloc(Sn,sizeof(struct buf)); int ok=1;
        for(int ns=0;ns<Sn && ok;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
            struct buf*bf=&w->Bf[ns]; *bf=orki_bimport(c->fd,(size_t)K*Nc,w->domain);
            if(!bf->cpu){ ok=0; break; }
            int8_t*fb=bf->cpu; double _bf=orki_load_prof?ork_now_us():0;
            orki_dmabuf_sync(bf->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
            for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32,kf0=k0/32;
                const int8_t*sb=(const int8_t*)w->Bb[(size_t)ns*Sk+ks].cpu;
                /* per (nt): this K-slice's KT k-tiles are CONTIGUOUS in both src (nt*KT*1024 + kt*1024) and
                 * dst (nt*KTf*1024 + (kf0+kt)*1024) — so the whole [KT][32][32] run copies as ONE memcpy,
                 * replacing the old 4-deep per-BYTE loop (~1M scalar stores/weight → NN vectorized memcpys). */
                for(int nt=0;nt<NN;nt++)
                    memcpy(fb + ((size_t)nt*KTf + kf0)*32*32,
                           sb + (size_t)nt*KT*32*32,
                           (size_t)KT*32*32);}
            orki_dmabuf_sync(bf->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE); if(orki_load_prof) orki_lp_bf+=ork_now_us()-_bf;}
        if(!ok){ for(int ns=0;ns<Sn;ns++) orki_bdestroy(c->fd,&w->Bf[ns]); free(w->Bf); w->Bf=NULL; } }
    return w;
}

ork_w *ork_mm_adopt_imported_i8(ork_npu *c,int K,int N,int bb_fd,int bf_fd,size_t bb_bytes,size_t bf_bytes){
    if(K%32 || N%32 || bb_fd<0 || !c){ if(bb_fd>=0) close(bb_fd); if(bf_fd>=0) close(bf_fd); return NULL; }
    /* DEFAULT = NATIVE MATERIALIZE. Cross-process PRIME-imported (bimport_fd) weight buffers WEDGE the CHAINED
     * prefill submit (CDMA chain-walker + imported buffer = kernel watchdog reset / D-state hard wedge — HW/kernel
     * limit, confirmed: even a single imported buffer chained over M-tiles wedges; decode's single submit is fine).
     * So we materialize the client's pre-tiled bytes into NATIVE residence (ork_mm_load_i8: mmap the shared fd +
     * bcreate + memcpy + native per-N-slice Bf rebuild) — the proven-chaining layout. Still meets the goal: orkpack
     * loads, NO live-convert / NO tiling / NO socket transfer (fd-passed, one memcpy from the shared page cache),
     * client still chooses the domain. NOT zero-copy. True zero-copy needs a non-chained/coalesced submit structure
     * (the on-NPU coalesced-chain RE) — set ORK_ADOPT_ZC to use the (wedge-prone) zero-copy VIEW path for that work. */
    if(!getenv("ORK_ADOPT_ZC")){
        size_t sz=orki_pgup(bb_bytes); void *p=mmap(NULL,sz,PROT_READ,MAP_SHARED,bb_fd,0);
        close(bb_fd); if(bf_fd>=0) close(bf_fd);            /* bf ignored — ork_mm_load_i8 rebuilds Bf natively */
        if(p==MAP_FAILED) return NULL;
        ork_w *w=ork_mm_load_i8(c,K,N,p,bb_bytes);         /* native bcreate + copy + native Bf (uses c->pack_domain) */
        munmap(p,sz);
        return w;
    }
    /* ---- ORK_ADOPT_ZC: pure zero-copy VIEW path (page-cache-backed, client-owned) — WEDGES chained prefill;
     * kept for the coalesced-chain zero-copy work (task #20). ---- */
    int KS=1024, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t bbneed=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS;(void)n0; bbneed+=orki_pgup((size_t)Kp*Nc);}}
    if(bb_bytes<bbneed){ close(bb_fd); if(bf_fd>=0) close(bf_fd); return NULL; }
    int dom=ork_dom(c->pack_domain);
    ork_dom_prime(c,dom);                            /* native anchor BEFORE importing into a non-0 domain */
    struct buf bbimp=orki_bimport_fd(c->fd,bb_fd,bb_bytes,dom);
    if(!bbimp.cpu){ close(bb_fd); if(bf_fd>=0) close(bf_fd); return NULL; }   /* bimport_fd doesn't close on failure */
    ork_w *w=calloc(1,sizeof *w); if(!w){ orki_bdestroy(c->fd,&bbimp); if(bf_fd>=0) close(bf_fd); return NULL; }
    w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I8; w->owns=0; w->domain=dom;
    w->own_bufs=calloc(2,sizeof(struct buf)); w->n_own_bufs=0;   /* [0]=Bb import [1]=Bf import; ork_mm_free bdestroys each */
    if(!w->own_bufs){ orki_bdestroy(c->fd,&bbimp); if(bf_fd>=0) close(bf_fd); free(w); return NULL; }
    w->own_bufs[w->n_own_bufs++]=bbimp;
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    if(!w->Bb){ ork_mm_free(c,w); if(bf_fd>=0) close(bf_fd); return NULL; }
    size_t off=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS;(void)n0; size_t ts=orki_pgup((size_t)Kp*Nc);
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks];
        b->handle=bbimp.handle; b->obj=bbimp.obj; b->dma=bbimp.dma+off; b->cpu=(char*)bbimp.cpu+off; b->size=ts; b->domain=dom; b->heap_fd=-1;
        off+=ts;}}
    /* Bf: its OWN import (separate obj). Views into it carry heap_fd=-1 so ork_mm_free skips them and reclaims
     * the one Bf import via own_bufs[1]. On any Bf failure, Bf=NULL (base matmuls then 501-error, surfacing it)
     * but the weight itself stays valid (Bb resident). */
    if(bf_fd>=0 && bf_bytes){
        size_t bfneed=0; for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX; bfneed+=orki_pgup((size_t)K*Nc);}
        if(bf_bytes>=bfneed){
            struct buf bfimp=orki_bimport_fd(c->fd,bf_fd,bf_bytes,dom);
            if(bfimp.cpu){
                w->own_bufs[w->n_own_bufs++]=bfimp;
                w->Bf=calloc(Sn,sizeof(struct buf));
                if(w->Bf){ size_t fo=0;
                    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX; size_t fs=orki_pgup((size_t)K*Nc);
                        struct buf*bf=&w->Bf[ns];
                        bf->handle=bfimp.handle; bf->obj=bfimp.obj; bf->dma=bfimp.dma+fo; bf->cpu=(char*)bfimp.cpu+fo; bf->size=fs; bf->domain=dom; bf->heap_fd=-1;
                        fo+=fs;} }
            } else close(bf_fd);   /* Bf import failed: drop Bf, keep the weight (Bb resident) */
        } else close(bf_fd);
    } else if(bf_fd>=0) close(bf_fd);
    return w;
}

ork_w *ork_mm_import_i8(ork_npu *c,int K,int N,const void *blob,size_t n,size_t bf_off){
    if(!c || !c->daemon || !blob || K<=0 || N<=0 || !n) return NULL;
    size_t bb_bytes = bf_off ? bf_off : n;          /* Bb = blob[0..bf_off); Bf = blob[bf_off..n) */
    size_t bf_bytes = bf_off ? (n - bf_off) : 0;
    /* Bb dma-buf */
    void *pbb=NULL; int bb_fd=ork_dmabuf_alloc(bb_bytes,&pbb);
    if(bb_fd<0) return NULL;
    memcpy(pbb,blob,bb_bytes); ork_dmabuf_seal(bb_fd);
    /* Bf dma-buf (separate buffer so the daemon imports it into its OWN obj) */
    void *pbf=NULL; int bf_fd=-1;
    if(bf_bytes){ bf_fd=ork_dmabuf_alloc(bf_bytes,&pbf);
        if(bf_fd<0){ munmap(pbb,orki_pgup(bb_bytes)); close(bb_fd); return NULL; }
        memcpy(pbf,(const char*)blob+bf_off,bf_bytes); ork_dmabuf_seal(bf_fd); }
    uint64_t id=orkd_import_i8(c->daemon,K,N,bb_fd,bf_fd,(uint64_t)bb_bytes,(uint64_t)bf_bytes);
    munmap(pbb,orki_pgup(bb_bytes)); close(bb_fd);        /* daemon holds its own SCM_RIGHTS dups */
    if(bf_fd>=0){ munmap(pbf,orki_pgup(bf_bytes)); close(bf_fd); }
    if(!id) return NULL;
    ork_w *w=calloc(1,sizeof *w); if(!w) return NULL;
    w->is_orkd=1; w->orkd_id=id; w->K=K; w->N=N; w->dtype=DT_I8; w->domain=ork_dom(c->pack_domain);
    return w;
}

int ork_mm_repack_i8(ork_npu *c,ork_w *w,int K,int N,const int8_t *B){
    if(!w || w->dtype!=DT_I8 || !w->Bb) return -1;
    if(w->K!=K || w->N!=N) return -2;                  /* must match the slot's allocated shape */
    int KS=1024, NMAX=c->soc->nmax, Sk=w->Sk, Sn=w->Sn;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; if(!b->cpu) return -1; int8_t*bb=b->cpu;
        for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
            bb[nt*KT*32*32+kt*32*32+nl*32+kk]=B[(size_t)(k0+kt*32+kk)*N+(n0+nt*32+nl)];
        orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}}
    if(w->Bf && K<=10752){ int KTf=K/32;
        for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
            struct buf*b=&w->Bf[ns]; if(!b->cpu) continue; int8_t*bb=b->cpu;
            for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KTf;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
                bb[(size_t)nt*KTf*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(kt*32+kk)*N+(n0+nt*32+nl)];
            orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}}
    return 0;
}

ork_w *ork_mm_load_i8_flags(ork_npu *c,int K,int N,const void *blob,size_t n,unsigned flags){
    if(K%32 || N%32) return NULL;
    int KS=1024, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t need=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS;(void)n0; need+=orki_pgup((size_t)Kp*Nc);}}
    if(n!=need) return NULL;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I8; w->owns=1; w->domain=ork_dom(c->pack_domain);
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    size_t off=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS;(void)n0;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=orki_bcreate(c->fd,(size_t)Kp*Nc,flags,w->domain);
        if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) orki_bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; }
        memcpy(b->cpu,(const char*)blob+off,b->size); off+=b->size;
        orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}}
    return w;
}

static inline void quant32_f32_i8(int8_t *dst, const float *fr, float inv) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t vinv = vdupq_n_f32(inv); int32x4_t lo = vdupq_n_s32(-127);
    for (int kk = 0; kk < 32; kk += 8) {
        int32x4_t i0 = vmaxq_s32(vcvtnq_s32_f32(vmulq_f32(vld1q_f32(fr + kk),     vinv)), lo);
        int32x4_t i1 = vmaxq_s32(vcvtnq_s32_f32(vmulq_f32(vld1q_f32(fr + kk + 4), vinv)), lo);
        vst1_s8(dst + kk, vqmovn_s16(vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1))));
    }
#else
    for (int kk = 0; kk < 32; kk++) { int q = (int)lrintf(fr[kk] * inv); dst[kk] = (int8_t)(q > 127 ? 127 : q < -127 ? -127 : q); }
#endif
}

void orki_tile_f32_i8(ork_npu *c, ork_w *w, int K, int N, const float *f32, const float *inv) {
    int KS = 1024, NMAX = c->soc->nmax, Sk = w->Sk, Sn = w->Sn, fd = c->fd;
    for (int ns = 0; ns < Sn; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX, NN = Nc/32;
      for (int ks = 0; ks < Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS, KT = Kp/32;
        struct buf *b = &w->Bb[(size_t)ns*Sk+ks]; if (!b->cpu) continue; int8_t *bb = b->cpu;
        for (int nt = 0; nt < NN; nt++) for (int nl = 0; nl < 32; nl++) {
            int n = n0+nt*32+nl; const float *frn = f32 + (size_t)n*K + k0; float iv = inv[n];
            for (int kt = 0; kt < KT; kt++) quant32_f32_i8(bb + ((size_t)nt*KT*32*32 + (size_t)kt*32*32 + nl*32), frn + kt*32, iv);
        }
        orki_bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE); } }
    if (w->Bf && K <= 10752) { int KTf = K/32;
        for (int ns = 0; ns < Sn; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX, NN = Nc/32;
            struct buf *b = &w->Bf[ns]; if (!b->cpu) continue; int8_t *bb = b->cpu;
            for (int nt = 0; nt < NN; nt++) for (int nl = 0; nl < 32; nl++) {
                int n = n0+nt*32+nl; const float *frn = f32 + (size_t)n*K; float iv = inv[n];
                for (int kt = 0; kt < KTf; kt++) quant32_f32_i8(bb + ((size_t)nt*KTf*32*32 + (size_t)kt*32*32 + nl*32), frn + kt*32, iv);
            }
            orki_bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE); } }
}

ork_w *ork_mm_pack_i8_f32(ork_npu *c, int K, int N, const float *f32, float *bscale_out) {
    if (K % 32 || N % 32) return NULL;
    int KS = 1024, NMAX = c->soc->nmax, Sk = (K+KS-1)/KS, Sn = (N+NMAX-1)/NMAX;
    ork_w *w = calloc(1, sizeof *w); if (!w) return NULL;
    w->K = K; w->N = N; w->Sk = Sk; w->Sn = Sn; w->dtype = DT_I8; w->Bb = calloc((size_t)Sk*Sn, sizeof(struct buf));
    if (!w->Bb) { free(w); return NULL; }
    for (int ns = 0; ns < Sn; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX;
      for (int ks = 0; ks < Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS;
        struct buf *b = &w->Bb[(size_t)ns*Sk+ks]; *b = orki_bcreate(c->fd, (size_t)Kp*Nc, 0x403, w->domain);
        if (!b->cpu) { for (int i = 0; i < ns*Sk+ks; i++) orki_bdestroy(c->fd, &w->Bb[i]); free(w->Bb); free(w); return NULL; } } }
    if (K <= 10752 && !getenv("ORK_NO_BF")) { w->Bf = calloc(Sn, sizeof(struct buf)); int ok = 1;
        for (int ns = 0; ns < Sn && ok; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX;
            struct buf *b = &w->Bf[ns]; *b = orki_bcreate(c->fd, (size_t)K*Nc, 0x403, w->domain); if (!b->cpu) ok = 0; }
        if (!ok) { for (int ns = 0; ns < Sn; ns++) orki_bdestroy(c->fd, &w->Bf[ns]); free(w->Bf); w->Bf = NULL; } }
    float *inv = malloc((size_t)N * sizeof(float)); if (!inv) { ork_w_free(w); return NULL; }
    orki_chan_scales_f32(f32, K, N, inv, bscale_out);
    orki_tile_f32_i8(c, w, K, N, f32, inv);
    free(inv);
    return w;
}

void orki_inflate_chan_nf4_i8(const uint8_t *nib, int K, const int8_t lut[16], int8_t *i8) {
    int k = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int8x16_t vlut = vld1q_s8(lut); uint8x8_t vlo = vdup_n_u8(0x0f);
    for (; k <= K - 16; k += 16) {
        uint8x8_t pk = vld1_u8(nib + (k>>1));                 /* 8 bytes = 16 indices */
        uint8x8_t even = vand_u8(pk, vlo);                    /* low nibbles (idx k,k+2,...) */
        uint8x8_t odd  = vshr_n_u8(pk, 4);                    /* high nibbles (idx k+1,...) */
        uint8x8x2_t zip = vzip_u8(even, odd);                 /* interleave -> index order */
        uint8x16_t vi = vcombine_u8(zip.val[0], zip.val[1]);
        vst1q_s8(i8 + k, vqtbl1q_s8(vlut, vi));               /* code = lut[idx] */
    }
#endif
    for (; k < K; k++) { uint8_t idx = (k & 1) ? (nib[k>>1] >> 4) : (nib[k>>1] & 0xf); i8[k] = lut[idx]; }
}

void orki_tile_i8_to_tiles(ork_npu *c, ork_w *w, int K, int N, const int8_t *i8) {
    int KS = 1024, NMAX = c->soc->nmax, Sk = w->Sk, Sn = w->Sn, fd = c->fd;
    for (int ns = 0; ns < Sn; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX, NN = Nc/32;
      for (int ks = 0; ks < Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS, KT = Kp/32;
        struct buf *b = &w->Bb[(size_t)ns*Sk+ks]; if (!b->cpu) continue; int8_t *bb = b->cpu;
        for (int nt = 0; nt < NN; nt++) for (int nl = 0; nl < 32; nl++) {
            int n = n0+nt*32+nl; const int8_t *src = i8 + (size_t)n*K + k0;
            for (int kt = 0; kt < KT; kt++)
                memcpy(bb + ((size_t)nt*KT*32*32 + (size_t)kt*32*32 + nl*32), src + kt*32, 32);
        }
        orki_bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE); } }
    if (w->Bf && K <= 10752) { int KTf = K/32;
        for (int ns = 0; ns < Sn; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX, NN = Nc/32;
            struct buf *b = &w->Bf[ns]; if (!b->cpu) continue; int8_t *bb = b->cpu;
            for (int nt = 0; nt < NN; nt++) for (int nl = 0; nl < 32; nl++) {
                int n = n0+nt*32+nl; const int8_t *src = i8 + (size_t)n*K;
                for (int kt = 0; kt < KTf; kt++)
                    memcpy(bb + ((size_t)nt*KTf*32*32 + (size_t)kt*32*32 + nl*32), src + kt*32, 32);
            }
            orki_bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd, b, RKNPU_MEM_SYNC_TO_DEVICE); } }
}

void orki_tile_i8_to_import_tiles(ork_npu *c, ork_w *w, int K, int N, const int8_t *i8){
    int KS=1024, NMAX=c->soc->nmax, Sk=w->Sk, Sn=w->Sn;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; if(!b->cpu)continue; int8_t*bb=b->cpu;
        orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
        for(int nt=0;nt<NN;nt++)for(int nl=0;nl<32;nl++){ int n=n0+nt*32+nl; const int8_t*src=i8+(size_t)n*K+k0;
            for(int kt=0;kt<KT;kt++) memcpy(bb+((size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32),src+kt*32,32); }
        orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
    if(w->Bf && K<=10752){ int KTf=K/32;
        for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
            struct buf*b=&w->Bf[ns]; if(!b->cpu)continue; int8_t*bb=b->cpu;
            orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
            for(int nt=0;nt<NN;nt++)for(int nl=0;nl<32;nl++){ int n=n0+nt*32+nl; const int8_t*src=i8+(size_t)n*K;
                for(int kt=0;kt<KTf;kt++) memcpy(bb+((size_t)nt*KTf*32*32+(size_t)kt*32*32+nl*32),src+kt*32,32); }
            orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
}

void ork_slice_tile_i8(ork_npu *c, ork_w *w, const float *qf32, float *inv1) {
    if (!w) return;
    orki_tile_f32_i8(c, w, w->K, w->N, qf32, inv1);
}

void ork_slice_direct_inflate_i8(const ork_w *w, int8_t *i8, int kind) {
    if (!w || !w->Bi4) return;
    int K = w->K, N = w->N;
    if (kind == ORK_QK_CODEBOOK_NF4) {
        int8_t lut[16]; for (int i = 0; i < 16; i++) lut[i] = (int8_t)lrintf(ORKI_NF4_LEVELS[i]*127.0f);
        for (int n = 0; n < N; n++) orki_inflate_chan_nf4_i8(w->Bi4 + (size_t)n*(K/2), K, lut, i8 + (size_t)n*K);
    } else {
        for (int n = 0; n < N; n++) orki_expand_chan_i4_i8(w->Bi4 + (size_t)n*(K/2), K, i8 + (size_t)n*K);
    }
}

void ork_stage_fill_i8(ork_npu *c, struct ork_stage *s, const int8_t *B){
    if(!s || !B) return;
    int K=s->K, N=s->N, KS=1024, NMAX=c->soc->nmax, Sk=s->Sk, Sn=s->Sn;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
        struct buf*b=&s->Bb[(size_t)ns*Sk+ks];
        orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
        struct tile_i8_arg ta={(int8_t*)b->cpu,B,KT,k0,n0,N}; ork_parallel_for(NN,orki_tile_i8_range,&ta);
        orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
    if(s->Bf){ int KTf=K/32;
        for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
            struct buf*b=&s->Bf[ns];
            orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
            struct tile_i8_arg ta={(int8_t*)b->cpu,B,KTf,0,n0,N}; ork_parallel_for(NN,orki_tile_i8_range,&ta);
            orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
}

struct ork_stream_entry *ork_stream_pool_add_i8(struct ork_stream_pool *p, int K, int N, const void *blob, size_t n){
    if(!p || K%32 || N%32) return NULL;
    int KS=1024, NMAX=p->c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t need=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int Kp=(K-ks*KS<KS)?(K-ks*KS):KS;(void)n0; need+=orki_pgup((size_t)Kp*Nc);}}
    if(n!=need) return NULL;
    struct ork_stream_entry *e=orki_pool_new_entry(p,K,N); if(!e) return NULL;
    struct ork_stage *s=e->stg; size_t off=0;
    for(int i=0;i<s->Sk*s->Sn;i++){ struct buf*b=&s->Bb[i];
        orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
        memcpy(b->cpu,(const char*)blob+off,b->size); off+=b->size;
        orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE); }
    /* Bf full-K rebuild from the just-filled Bb tiles (same envelope as load_i8_import) */
    if(s->Bf){ int KTf=K/32;
        for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
            struct buf*bf=&s->Bf[ns]; int8_t*fb=bf->cpu;
            orki_dmabuf_sync(bf->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
            for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
                const int8_t*sb=(const int8_t*)s->Bb[(size_t)ns*Sk+ks].cpu;
                for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++){
                    int ktf=(k0/32)+kt;
                    fb[(size_t)nt*KTf*32*32+(size_t)ktf*32*32+nl*32+kk]=
                        sb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]; }}
            orki_dmabuf_sync(bf->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
    return e;
}

struct ork_stream_entry *ork_stream_pool_add_i8_raw(struct ork_stream_pool *p, int K, int N, const int8_t *B){
    if(!p || !B || K%32 || N%32) return NULL;
    struct ork_stream_entry *e=orki_pool_new_entry(p,K,N); if(!e) return NULL;
    ork_stage_fill_i8(p->c,e->stg,B);
    return e;
}

int ork_mm_repack_i8_f32(ork_npu *c, ork_w *w, int K, int N, const float *f32, float *bscale_out) {
    if (!w || w->dtype != DT_I8 || !w->Bb) return -1;
    if (w->K != K || w->N != N) return -2;
    float *inv = malloc((size_t)N * sizeof(float)); if (!inv) return -1;
    orki_chan_scales_f32(f32, K, N, inv, bscale_out);
    orki_tile_f32_i8(c, w, K, N, f32, inv);
    free(inv);
    return 0;
}

static int tile_dequant_i8(ork_npu *c, ork_w *w, int K, int N, ork_dequant_row_fn fn, void *dctx, float *bscale) {
    int KS = 1024, NMAX = c->soc->nmax, Sk = w->Sk, Sn = w->Sn, fd = c->fd, KTf = K/32;
    float *sc = malloc((size_t)K * sizeof(float)); if (!sc) return -1;
    for (int n = 0; n < N; n++) {
        fn(dctx, n, sc, K);                                   /* dequant channel n -> reused scratch[K] */
        float mx = 1e-9f; int k = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        float32x4_t vmx = vdupq_n_f32(1e-9f);
        for (; k <= K-4; k += 4) vmx = vmaxq_f32(vmx, vabsq_f32(vld1q_f32(sc + k)));
        float m[4]; vst1q_f32(m, vmx); float a=m[0]>m[1]?m[0]:m[1], b=m[2]>m[3]?m[2]:m[3]; mx=a>b?a:b;
#endif
        for (; k < K; k++) { float v = fabsf(sc[k]); if (v > mx) mx = v; }
        float iv = 127.0f/mx; bscale[n] = mx/127.0f;
        int ns = n/NMAX, nloc = n - ns*NMAX, nt = nloc/32, nl = nloc%32;
        for (int ks = 0; ks < Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS, KT = Kp/32;
            struct buf *b = &w->Bb[(size_t)ns*Sk+ks]; if (!b->cpu) continue; int8_t *bb = b->cpu;
            for (int kt = 0; kt < KT; kt++) quant32_f32_i8(bb + ((size_t)nt*KT*32*32 + (size_t)kt*32*32 + nl*32), sc + k0 + kt*32, iv);
        }
        if (w->Bf && K <= 10752) { struct buf *b = &w->Bf[ns]; if (b->cpu) { int8_t *bb = b->cpu;
            for (int kt = 0; kt < KTf; kt++) quant32_f32_i8(bb + ((size_t)nt*KTf*32*32 + (size_t)kt*32*32 + nl*32), sc + kt*32, iv); } }
    }
    free(sc);
    for (int i = 0; i < Sk*Sn; i++) { struct buf *b = &w->Bb[i]; if (b->cpu) { orki_bsync(fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd,b,RKNPU_MEM_SYNC_TO_DEVICE); } }
    if (w->Bf) for (int ns = 0; ns < Sn; ns++) { struct buf *b = &w->Bf[ns]; if (b->cpu) { orki_bsync(fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(fd,b,RKNPU_MEM_SYNC_TO_DEVICE); } }
    return 0;
}

ork_w *ork_mm_pack_i8_dequant(ork_npu *c, int K, int N, ork_dequant_row_fn fn, void *dctx, float *bscale_out) {
    if (K % 32 || N % 32) return NULL;
    int KS = 1024, NMAX = c->soc->nmax, Sk = (K+KS-1)/KS, Sn = (N+NMAX-1)/NMAX;
    ork_w *w = calloc(1, sizeof *w); if (!w) return NULL;
    w->K = K; w->N = N; w->Sk = Sk; w->Sn = Sn; w->dtype = DT_I8; w->Bb = calloc((size_t)Sk*Sn, sizeof(struct buf));
    if (!w->Bb) { free(w); return NULL; }
    for (int ns = 0; ns < Sn; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX;
      for (int ks = 0; ks < Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS;
        struct buf *b = &w->Bb[(size_t)ns*Sk+ks]; *b = orki_bcreate(c->fd, (size_t)Kp*Nc, 0x403, w->domain);
        if (!b->cpu) { for (int i = 0; i < ns*Sk+ks; i++) orki_bdestroy(c->fd, &w->Bb[i]); free(w->Bb); free(w); return NULL; } } }
    if (K <= 10752 && !getenv("ORK_NO_BF")) { w->Bf = calloc(Sn, sizeof(struct buf)); int ok = 1;
        for (int ns = 0; ns < Sn && ok; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX;
            struct buf *b = &w->Bf[ns]; *b = orki_bcreate(c->fd, (size_t)K*Nc, 0x403, w->domain); if (!b->cpu) ok = 0; }
        if (!ok) { for (int ns = 0; ns < Sn; ns++) orki_bdestroy(c->fd, &w->Bf[ns]); free(w->Bf); w->Bf = NULL; } }
    if (tile_dequant_i8(c, w, K, N, fn, dctx, bscale_out) != 0) { ork_w_free(w); return NULL; }
    return w;
}

int ork_mm_repack_i8_dequant(ork_npu *c, ork_w *w, int K, int N, ork_dequant_row_fn fn, void *dctx, float *bscale_out) {
    if (!w || w->dtype != DT_I8 || !w->Bb) return -1;
    if (w->K != K || w->N != N) return -2;
    return tile_dequant_i8(c, w, K, N, fn, dctx, bscale_out);
}

ork_w_sliced *orki_slice_pack_i8(ork_npu *c, int K, int N, const int8_t *B) {
    ork_slice_caps cap = ork_slice_caps_rk3588();
    if (N % cap.nmul) return NULL;                                       /* N always %32 for a real int8 weight (pack N%32) -> %nmul; defensive */
    /* PAD (adapter transform): an unaligned K (K%kmul!=0 — rare, odd-dim models / no-Bf) is padded UP to the
     * next kmul so every K-slice tile is legal. The pad rows are zero-filled (below) and A is zero-padded to
     * Kpad at run — zeros contribute nothing to the int32 sum, so the result is BIT-EXACT for the real K. N is
     * never padded (a real int8 weight is already N%32==0 -> N%nmul==0). Kpad==K for the aligned common case
     * (no behavior change). */
    int Kpad = ((K + cap.kmul - 1) / cap.kmul) * cap.kmul;
    int ks = (cap.kmax / cap.kmul) * cap.kmul, cap_ns = (cap.nmax / cap.nmul) * cap.nmul;
    int nks = (Kpad + ks - 1) / ks;
    /* BALANCED N-TILING (#33 stage 3 — native colsplit's balanced boundary-split): split N into EQUAL-width
     * tiles (nmul-aligned, <= nmax) using at least `cores` of them, so the doorbell submit hands each core an
     * even column load. The old nmax+remainder tiling put e.g. 8192+512 on 2 cores (3rd idle) = a 2.3x loss;
     * equal tiles >= cores mirror ork_dyn_begin_colsplit's t0=i*N/nc balance. */
    int nnt = (N + cap_ns - 1) / cap_ns; if (nnt < c->soc->cores) nnt = c->soc->cores;
    int nalign = 32;                                                     /* int8 orki_pack() needs each tile width %32==0 (fp16 %16); N%32==0 holds for any packed int8 weight, so all tiles stay valid */
    int ns = ((N + nnt - 1) / nnt + nalign - 1) / nalign * nalign;       /* equal width, 32-aligned */
    if (ns > cap_ns) ns = cap_ns; if (ns < nalign) ns = nalign;
    nnt = (N + ns - 1) / ns;                                             /* actual tiles after alignment */
    struct ork_w_sliced *w = calloc(1, sizeof *w); if (!w) return NULL;
    w->K = K; w->N = N; w->Kpad = Kpad; w->dtype = DT_I8; w->cap = cap; w->nks = nks; w->nnt = nnt; w->ks = ks; w->ns = ns;
    w->sub = calloc((size_t) nks * nnt, sizeof(ork_w *));
    int8_t *blk = malloc((size_t) ks * ns);
    if (!w->sub || !blk) { free(blk); ork_mm_free_sliced(c, w); return NULL; }
    orki_in_slice_pack = 1;   /* #33: the sub-tile packs below must NOT re-trigger the pack-time slice-build (no recursion) */
    for (int ki = 0; ki < nks; ki++) { int k0 = ki*ks, k1 = k0+ks < Kpad ? k0+ks : Kpad, Ks = k1-k0;   /* K extent over Kpad */
        for (int ni = 0; ni < nnt; ni++) { int n0 = ni*ns, n1 = n0+ns < N ? n0+ns : N, Nw = n1-n0;
            for (int k = 0; k < Ks; k++) { if (k0+k < K) memcpy(blk + (size_t) k*Nw, B + (size_t)(k0+k)*N + n0, Nw);   /* real weight row */
                                           else          memset(blk + (size_t) k*Nw, 0, Nw); }                        /* PAD row -> zero (contributes 0 to the sum) */
            ork_w *sw = ork_mm_pack_i8(c, Ks, Nw, blk);
            if (!sw) { orki_in_slice_pack = 0; free(blk); ork_mm_free_sliced(c, w); return NULL; }
            w->sub[ki*nnt + ni] = sw; } }
    orki_in_slice_pack = 0;
    free(blk); return w;
}

size_t ork_w_dump_i8_cpu_st(ork_npu *c, int K, int N, const int8_t *B, void *out, size_t cap){
    if(!c || !B || (K%32) || (N%32)) return 0;
    int KS=orki_int8_ks(c), NMAX=c->soc->nmax;
    int Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t off=0;
    for(int ns=0;ns<Sn;ns++){ int n0=ns*NMAX, Nc=(N-n0<NMAX)?(N-n0):NMAX, NN=Nc/32;
      for(int ks=0;ks<Sk;ks++){ int k0=ks*KS, Kp=(K-k0<KS)?(K-k0):KS, KT=Kp/32; size_t tsz=orki_pgup((size_t)Kp*Nc);
        if(out){ if(off+tsz>cap) return 0;
            int8_t *bb=(int8_t*)out+off; memset(bb,0,tsz);
            struct tile_i8_arg ta={bb,B,KT,k0,n0,N}; orki_tile_i8_range(0,NN,&ta); }   /* inline: no pool */
        off+=tsz; }}
    return off;
}

ork_w *ork_mm_pack_i8_import(ork_npu *c,int K,int N,const int8_t *B){
    if(K%32 || N%32) return NULL;
    if(orki_dmaheap_open()<0) return NULL;
    int KS=1024, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I8; w->owns=1; w->domain=ork_dom(c->pack_domain);
    ork_dom_prime(c, w->domain);
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    int consolidate = !getenv("ORK_NO_CONSOLIDATE_IMPORT");
    if(consolidate){
        size_t chunk_mb = 16; const char*cm=getenv("ORK_IMPORT_CHUNK_MB"); if(cm){ long v=atol(cm); if(v>0) chunk_mb=(size_t)v; }
        size_t chunk_cap = chunk_mb<<20;
        int cap_chunks=Sk*Sn+1;
        w->own_bufs=calloc(cap_chunks,sizeof(struct buf)); w->n_own_bufs=0;
        struct buf cur; cur.cpu=NULL; size_t coff=0, csz=0;
        int ns,ks;
        for(ns=0;ns<Sn && consolidate;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
          for(ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32; size_t ts=orki_pgup((size_t)Kp*Nc);
            if(!cur.cpu || coff+ts>csz){
                if(cur.cpu) orki_dmabuf_sync(cur.heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);
                csz = ts>chunk_cap ? ts : chunk_cap;
                cur = orki_bimport(c->fd,csz,w->domain);
                if(!cur.cpu){ consolidate=0; break; }
                orki_dmabuf_sync(cur.heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
                w->own_bufs[w->n_own_bufs++]=cur; coff=0;
            }
            struct buf*b=&w->Bb[(size_t)ns*Sk+ks];
            b->handle=cur.handle; b->obj=cur.obj; b->dma=cur.dma+coff; b->cpu=(char*)cur.cpu+coff; b->size=ts;
            int8_t*bb=b->cpu;
            for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
                bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(k0+kt*32+kk)*N+(n0+nt*32+nl)];
            coff+=ts;}}
        if(consolidate){ if(cur.cpu) orki_dmabuf_sync(cur.heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE); w->owns=0; }
        else { for(int i=0;i<w->n_own_bufs;i++) orki_bdestroy(c->fd,&w->own_bufs[i]); free(w->own_bufs); w->own_bufs=NULL; w->n_own_bufs=0; memset(w->Bb,0,(size_t)Sk*Sn*sizeof(struct buf)); w->owns=1; }
    }
    if(!consolidate){
        for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;
          for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
            struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=orki_bimport(c->fd,(size_t)Kp*Nc,w->domain);
            if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) orki_bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; }
            orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
            int8_t*bb=b->cpu;
            for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<32;nl++)for(int kk=0;kk<32;kk++)
                bb[(size_t)nt*KT*32*32+(size_t)kt*32*32+nl*32+kk]=B[(size_t)(k0+kt*32+kk)*N+(n0+nt*32+nl)];
            orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
    }
    if(K%512==0 && K<=4096 && !getenv("ORK_NO_BF")){ int KTf=K/32; w->Bf=calloc(Sn,sizeof(struct buf)); int ok=1;
        for(int ns=0;ns<Sn && ok;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/32;(void)n0;
            struct buf*bf=&w->Bf[ns]; *bf=orki_bimport(c->fd,(size_t)K*Nc,w->domain);
            if(!bf->cpu){ ok=0; break; }
            int8_t*fb=bf->cpu;
            orki_dmabuf_sync(bf->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
            for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32,kf0=k0/32;
                const int8_t*sb=(const int8_t*)w->Bb[(size_t)ns*Sk+ks].cpu;
                for(int nt=0;nt<NN;nt++)
                    memcpy(fb + ((size_t)nt*KTf + kf0)*32*32, sb + (size_t)nt*KT*32*32, (size_t)KT*32*32);}
            orki_dmabuf_sync(bf->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}
        if(!ok){ for(int ns=0;ns<Sn;ns++) orki_bdestroy(c->fd,&w->Bf[ns]); free(w->Bf); w->Bf=NULL; } }
    return w;
}
