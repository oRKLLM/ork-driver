/* npu/i4/pack.c — int4 pack, load, persist and tile layouts.
 *
 * Part of the i4 datapath; shared declarations in npu/i4/i4.h. Split out of npu/i4.c for the
 * same reason i8 is a folder: one datapath, sized for reading. */
#define _GNU_SOURCE   /* CPU_SET/pthread_setaffinity_np, as npu.c does */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include <math.h>
#include "ork_regs.h"
#include "regcmd_i8.h"
#include "orkd_proto.h"
#include "npu/internal.h"
#include "npu/core.h"
#include "regcmd_i4.h"
#include "npu/i4/i4.h"

ork_w *ork_i4a8_mm_pack(ork_npu *c, int K, int N, const float *f32, float *bscale_out) {
    return ork_i4a8_mm_pack_im(c, K, N, f32, NULL, bscale_out);
}

ork_w *ork_i4a8_mm_pack_im(ork_npu *c, int K, int N, const float *f32, const float *imatrix, float *bscale_out) {
    if (K % 32 || N % 32) return NULL;
    int sr = getenv("ORK_SR") != NULL; uint32_t seed = 0x2545F491u;   /* SR PRNG: fixed seed => deterministic/testable */
    int nf4 = getenv("ORK_NF4") != NULL;   /* ORK_NF4: non-uniform NF4 codebook instead of the uniform int4 grid */
    int KS = 1024, NMAX = c->soc->nmax, Sk = (K+KS-1)/KS, Sn = (N+NMAX-1)/NMAX;
    ork_w *w = calloc(1, sizeof *w); if (!w) return NULL;
    w->K = K; w->N = N; w->Sk = Sk; w->Sn = Sn; w->dtype = DT_I8; w->owns = 1; w->domain=ork_dom(c->pack_domain); w->quant_kind = nf4 ? ORK_QK_CODEBOOK_NF4 : ORK_QK_UNIFORM;
    w->Bb = calloc((size_t)Sk*Sn, sizeof(struct buf));
    if (!w->Bb) { free(w); return NULL; }
    for (int ns = 0; ns < Sn; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX;
      for (int ks = 0; ks < Sk; ks++) { int k0 = ks*KS, Kp = (K-k0<KS)?(K-k0):KS;
        struct buf *b = &w->Bb[(size_t)ns*Sk+ks]; *b = orki_bcreate(c->fd, (size_t)Kp*Nc, 0x403, w->domain);
        if (!b->cpu) { for (int i = 0; i < ns*Sk+ks; i++) orki_bdestroy(c->fd, &w->Bb[i]); free(w->Bb); free(w); return NULL; } } }
    if (K <= 10752 && !getenv("ORK_NO_BF")) { w->Bf = calloc(Sn, sizeof(struct buf)); int ok = 1;
        for (int ns = 0; ns < Sn && ok; ns++) { int n0 = ns*NMAX, Nc = (N-n0<NMAX)?(N-n0):NMAX;
            struct buf *b = &w->Bf[ns]; *b = orki_bcreate(c->fd, (size_t)K*Nc, 0x403, w->domain); if (!b->cpu) ok = 0; }
        if (!ok) { for (int ns = 0; ns < Sn; ns++) orki_bdestroy(c->fd, &w->Bf[ns]); free(w->Bf); w->Bf = NULL; } }
    /* compact int4 nibble store (n-major, K contiguous): the memory-win form, kept on the ork_w */
    w->Bi4_bytes = (size_t)N * (K/2);
    w->Bi4 = malloc(w->Bi4_bytes);
    /* retain per-channel dequant scale on the ork_w so the compact int4 form can be dumped self-contained */
    w->bscale = malloc((size_t)N * sizeof(float));
    /* int8 expansion scratch (f32 codes) + per-channel inv for the int8 tiler (codes are exact, inv=1) */
    float *qf32 = malloc((size_t)N * K * sizeof(float));
    float *inv  = malloc((size_t)N * sizeof(float));
    /* NF4: a per-tensor int8 LUT = round(level*127), and an index scratch (0..15) to inflate through it */
    int8_t nf4_lut[16]; uint8_t *qidx = NULL;
    if (nf4) { for (int i = 0; i < 16; i++) nf4_lut[i] = (int8_t)lrintf(ORKI_NF4_LEVELS[i]*127.0f);
               qidx = malloc((size_t)N * K); }
    /* imatrix path: reused per-channel dequant scratch[K] for the clip-grid search (NULL imatrix => unused) */
    float *imdq = imatrix ? malloc((size_t)K * sizeof(float)) : NULL;
    if (!w->Bi4 || !w->bscale || !qf32 || !inv || (nf4 && !qidx) || (imatrix && !imdq)) {
        free(qf32); free(inv); free(qidx); free(imdq); ork_w_free(w); return NULL; }
    for (int n = 0; n < N; n++) {
        const float *fr = f32 + (size_t)n*K; float mx = 1e-9f; int k = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        float32x4_t vmx = vdupq_n_f32(1e-9f);
        for (; k <= K-4; k += 4) vmx = vmaxq_f32(vmx, vabsq_f32(vld1q_f32(fr + k)));
        float m[4]; vst1q_f32(m, vmx); float a=m[0]>m[1]?m[0]:m[1], bb=m[2]>m[3]?m[2]:m[3]; mx=a>bb?a:bb;
#endif
        for (; k < K; k++) { float v = fabsf(fr[k]); if (v > mx) mx = v; }
        if (imatrix) mx = orki_wq_best_absmax(fr, K, mx, nf4, imatrix, imdq);  /* clip-grid scale selection */
        uint8_t *nib = w->Bi4 + (size_t)n*(K/2);
        if (nf4) { w->bscale[n] = mx / 127.0f;         /* int8 LUT range +-127 */
                   orki_nf4_quant_chan(fr, K, mx, sr, &seed, nib, qidx + (size_t)n*K); }
        else     { float scale = mx / 7.0f;            /* int4 range +-7 (NOT 127) */
                   w->bscale[n] = scale;
                   orki_i4_quant_chan(fr, K, scale, sr, &seed, nib, qf32 + (size_t)n*K); }
        if (bscale_out) bscale_out[n] = w->bscale[n];  /* back-compat: caller's out array (optional; w->bscale is canonical) */
        inv[n] = 1.0f;                                 /* qf32 holds exact codes; no rescale */
    }
    /* inflate the compact nibble store -> int8 f32 codes (validates the pack/inflate round-trip is what we tile) */
    if (nf4) for (int n = 0; n < N; n++) orki_nf4_inflate_chan_f32(qidx + (size_t)n*K, K, nf4_lut, qf32 + (size_t)n*K);
    else     for (int n = 0; n < N; n++) orki_i4_expand_chan_f32(w->Bi4 + (size_t)n*(K/2), K, qf32 + (size_t)n*K);
    orki_i8_tile_f32(c, w, K, N, qf32, inv);                /* REUSE the int8 DMA/tiling path (no dup) */
    free(qf32); free(inv); free(qidx); free(imdq);
    return w;
}

/* int4-stored: fill = inflate nibbles -> int8 + tile (the .orkpack i4a8 blob, ork_i4a8_w_dump). The fill
 * happens ONCE here (the expensive op, cached in RAM). NULL on import-unavailable / malformed blob. */
size_t ork_i4a8_w_dump(const ork_w *w, void *out, size_t cap){
    if(!w || !w->Bi4 || !w->bscale) return 0;
    size_t hdr=sizeof(struct ork_i4a8_hdr), sc=(size_t)w->N*sizeof(float), nib=(size_t)w->K*w->N/2;
    size_t need=hdr+sc+nib;
    if(!out) return need;
    if(cap<need) return 0;
    struct ork_i4a8_hdr h={ORK_I4A8_MAGIC, ORK_I4A8_VER, w->K, w->N, w->quant_kind};
    char *p=out;
    memcpy(p,&h,hdr); p+=hdr;
    memcpy(p,w->bscale,sc); p+=sc;
    memcpy(p,w->Bi4,nib);
    return need;
}

/* Reload the compact int4 form straight into NPU DMA: parse+validate header, read bscale + Bi4, inflate
 * each channel's nibbles -> int8 (UNIFORM sign-extend / NF4 LUT per quant_kind) and orki_i8_tile_f32 into a
 * fresh DMA buffer — the tail of the pack path, from stored nibbles instead of re-quantized f32. Retains
 * a copy of Bi4 + bscale so the loaded weight can be re-dumped byte-identically. NULL on malformed blob. */
/* Zero-copy IMPORT variant of ork_i4a8_mm_load: resident tiles are dma-bufs the NPU reads in place (PRIME
 * import), and the int4 nibbles inflate -> int8 directly into them (no f32 round-trip). Bit-identical to
 * ork_i4a8_mm_load (same blob, same tiled bytes). Falls through to NULL (caller uses ork_i4a8_mm_load) if
 * import is unavailable. Retains Bi4 + bscale so the loaded weight re-dumps byte-identically. */
/* tile shape mirrors ork_i4a8_mm_load: KS=1024 K-split, NMAX N-split; Bf full-K when K%512==0 && K<=4096
 * (same envelope as load_i8_import). Returns NULL if dma-heap absent / alloc fails. */
ork_w *ork_i4a8_mm_load(ork_npu *c, int K, int N, const void *blob, size_t n){
    if(K%32 || N%32) return NULL;
    size_t hdr=sizeof(struct ork_i4a8_hdr), sc=(size_t)N*sizeof(float), nib=(size_t)K*N/2;
    if(n != hdr+sc+nib) return NULL;
    const char *p=blob;
    struct ork_i4a8_hdr h; memcpy(&h,p,hdr); p+=hdr;
    if(h.magic!=ORK_I4A8_MAGIC || h.version!=ORK_I4A8_VER || h.K!=K || h.N!=N) return NULL;
    if(h.quant_kind!=ORK_QK_UNIFORM && h.quant_kind!=ORK_QK_CODEBOOK_NF4) return NULL;
    int KS=1024, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    ork_w *w=calloc(1,sizeof *w); if(!w) return NULL;
    w->K=K; w->N=N; w->Sk=Sk; w->Sn=Sn; w->dtype=DT_I8; w->owns=1; w->domain=ork_dom(c->pack_domain); w->quant_kind=(uint8_t)h.quant_kind;
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    if(!w->Bb){ free(w); return NULL; }
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;
        struct buf *b=&w->Bb[(size_t)ns*Sk+ks]; *b=orki_bcreate(c->fd,(size_t)Kp*Nc,0x403,w->domain);
        if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) orki_bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; } } }
    if(K<=10752 && !getenv("ORK_NO_BF")){ w->Bf=calloc(Sn,sizeof(struct buf)); int ok=1;
        for(int ns=0;ns<Sn && ok;ns++){ int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
            struct buf *b=&w->Bf[ns]; *b=orki_bcreate(c->fd,(size_t)K*Nc,0x403,w->domain); if(!b->cpu) ok=0; }
        if(!ok){ for(int ns=0;ns<Sn;ns++) orki_bdestroy(c->fd,&w->Bf[ns]); free(w->Bf); w->Bf=NULL; } }
    /* retain the compact store + scales so the loaded weight re-dumps byte-identically */
    w->Bi4_bytes=nib; w->Bi4=malloc(nib); w->bscale=malloc(sc);
    if(!w->Bi4 || !w->bscale){ ork_mm_free(c,w); return NULL; }
    memcpy(w->bscale,p,sc); p+=sc;
    memcpy(w->Bi4,p,nib);
    /* ORK_DIRECT_I4: inflate nibbles STRAIGHT to int8-tiled (1 byte/elem scratch, no f32 round-trip,
     * no re-quant) — bit-identical to the f32 path. Default off; preserves the f32 path for review. */
    if(getenv("ORK_DIRECT_I4")){
        int8_t *i8=malloc((size_t)N*K);
        if(!i8){ ork_mm_free(c,w); return NULL; }
        orki_i4_tile_direct_to_i8(c, w, K, N, w->quant_kind, i8);
        free(i8);
        return w;
    }
    float *qf32=malloc((size_t)N*K*sizeof(float)), *inv=malloc((size_t)N*sizeof(float));
    if(!qf32 || !inv){ free(qf32); free(inv); ork_mm_free(c,w); return NULL; }
    int8_t nf4_lut[16];
    if(w->quant_kind==ORK_QK_CODEBOOK_NF4) for(int i=0;i<16;i++) nf4_lut[i]=(int8_t)lrintf(ORKI_NF4_LEVELS[i]*127.0f);
    for(int nn=0;nn<N;nn++){
        if(w->quant_kind==ORK_QK_CODEBOOK_NF4){
            /* NF4 store keeps the 0..15 index in the nibble; inflate via the int8 LUT */
            const uint8_t *nibp=w->Bi4+(size_t)nn*(K/2); float *qf=qf32+(size_t)nn*K;
            for(int k=0;k<K;k++){ uint8_t idx=(k&1)?(nibp[k>>1]>>4):(nibp[k>>1]&0xf); qf[k]=(float)nf4_lut[idx]; }
        } else orki_i4_expand_chan_f32(w->Bi4+(size_t)nn*(K/2), K, qf32+(size_t)nn*K);
        inv[nn]=1.0f;                                  /* qf32 holds exact codes; no rescale */
    }
    orki_i8_tile_f32(c, w, K, N, qf32, inv);                /* REUSE the int8 DMA/tiling path (no dup) */
    free(qf32); free(inv);
    return w;
}

ork_w *ork_i4a8_mm_load_import(ork_npu *c, int K, int N, const void *blob, size_t n){
    if(K%32 || N%32 || orki_dmaheap_open()<0) return NULL;
    size_t hdr=sizeof(struct ork_i4a8_hdr), sc=(size_t)N*sizeof(float), nib=(size_t)K*N/2;
    if(n != hdr+sc+nib) return NULL;
    const char *p=blob;
    struct ork_i4a8_hdr h; memcpy(&h,p,hdr); p+=hdr;
    if(h.magic!=ORK_I4A8_MAGIC || h.version!=ORK_I4A8_VER || h.K!=K || h.N!=N) return NULL;
    if(h.quant_kind!=ORK_QK_UNIFORM && h.quant_kind!=ORK_QK_CODEBOOK_NF4) return NULL;
    int KS=1024, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    ork_w *w=calloc(1,sizeof *w); if(!w) return NULL;
    w->K=K; w->N=N; w->Sk=Sk; w->Sn=Sn; w->dtype=DT_I8; w->owns=1; w->domain=ork_dom(c->pack_domain); w->quant_kind=(uint8_t)h.quant_kind;
    ork_dom_prime(c, w->domain);   /* establish a non-0 domain with a native anchor BEFORE importing (same quirk as i8) */
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf)); if(!w->Bb){ free(w); return NULL; }
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;(void)n0;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=orki_bimport(c->fd,(size_t)Kp*Nc,w->domain);
        if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) orki_bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; } } }
    if(K%512==0 && K<=4096 && !getenv("ORK_NO_BF")){ w->Bf=calloc(Sn,sizeof(struct buf)); int ok=1;
        for(int ns=0;ns<Sn && ok;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
            struct buf*b=&w->Bf[ns]; *b=orki_bimport(c->fd,(size_t)K*Nc,w->domain); if(!b->cpu) ok=0; }
        if(!ok){ for(int ns=0;ns<Sn;ns++) orki_bdestroy(c->fd,&w->Bf[ns]); free(w->Bf); w->Bf=NULL; } }
    w->Bi4_bytes=nib; w->Bi4=malloc(nib); w->bscale=malloc(sc);
    if(!w->Bi4 || !w->bscale){ ork_mm_free(c,w); return NULL; }
    memcpy(w->bscale,p,sc); p+=sc; memcpy(w->Bi4,p,nib);
    int8_t *i8=malloc((size_t)N*K); if(!i8){ ork_mm_free(c,w); return NULL; }
    if(w->quant_kind==ORK_QK_CODEBOOK_NF4){ int8_t lut[16]; for(int i=0;i<16;i++) lut[i]=(int8_t)lrintf(ORKI_NF4_LEVELS[i]*127.0f);
        for(int nn=0;nn<N;nn++) orki_nf4_inflate_chan_to_i8(w->Bi4+(size_t)nn*(K/2),K,lut,i8+(size_t)nn*K);
    } else for(int nn=0;nn<N;nn++) orki_i4_expand_chan_to_i8(w->Bi4+(size_t)nn*(K/2),K,i8+(size_t)nn*K);
    orki_i8_tile_to_import_tiles(c,w,K,N,i8);
    free(i8);
    return w;
}

struct ork_stream_entry *ork_i4a8_stream_pool_add(struct ork_stream_pool *p, int K, int N, const void *blob, size_t n){
    if(!p) return NULL;
    /* validate + materialize an int4 source ork_w from the blob (host-side Bi4+bscale), inflate into the
     * entry's staging dma-buf, then drop the temporary source (we only needed its nibble store to fill). */
    ork_w *src=ork_i4a8_mm_load(p->c,K,N,blob,n);  /* allocates resident DMA too — temporary; freed below */
    if(!src) return NULL;
    struct ork_stream_entry *e=orki_pool_new_entry(p,K,N);
    if(!e){ ork_mm_free(p->c,src); return NULL; }
    ork_stage_fill(p->c,e->stg,src);               /* the ONE-TIME inflate into RAM-resident staging */
    ork_mm_free(p->c,src);
    return e;
}

static void tile_i4_Bslice(uint8_t*dst,const int8_t*B,int K,int N,int k0,int Kp,int n0,int Nc){
    int KT=Kp/32, NB=Nc/64; memset(dst,0,(size_t)Kp*Nc/2);
    for(int nb=0;nb<NB;nb++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<64;nl++)for(int kk=0;kk<32;kk++){
        size_t idx=(((size_t)nb*KT+kt)*64+nl)*32+kk;
        dst[idx/2]|= (uint8_t)(B[(size_t)(k0+kt*32+kk)*N+(n0+nb*64+nl)]&0xf) << ((idx&1)?4:0);
    }
}

void orki_i4_tile_Aslice(uint8_t*dst,const int8_t*Arow,int k0,int Kp){
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int col = 0;
    int8x16_t vmask = vdupq_n_s8(0x0f);
    for (; col <= Kp - 16; col += 16) {
        int8x16_t v = vld1q_s8(&Arow[k0 + col]);
        int8x16_t vmasked = vandq_s8(v, vmask);
        int8x16x2_t tuzp = vuzpq_s8(vmasked, vmasked);
        uint8x8_t veven_low = vreinterpret_u8_s8(vget_low_s8(tuzp.val[0]));
        uint8x8_t vodd_low  = vreinterpret_u8_s8(vget_low_s8(tuzp.val[1]));
        uint8x8_t vodd_shifted = vshl_n_u8(vodd_low, 4);
        uint8x8_t vcombined = vorr_u8(veven_low, vodd_shifted);
        vst1_u8(&dst[col / 2], vcombined);
    }
#else
    int KT=Kp/32; memset(dst,0,(size_t)Kp/2);
    for(int kt=0;kt<KT;kt++)for(int kk=0;kk<32;kk++){
        size_t idx=(size_t)kt*32+kk;
        dst[idx/2]|= (uint8_t)(Arow[k0+kt*32+kk]&0xf) << ((idx&1)?4:0);
    }
#endif
}

static void tile_i4_Aslice_mm(uint8_t*dst,const int8_t*A,int M,int K,int k0,int Kp){
    int KT=Kp/32; memset(dst,0,(size_t)M*Kp/2);
    for(int kt=0;kt<KT;kt++)for(int m=0;m<M;m++)for(int kk=0;kk<32;kk++){
        size_t idx=((size_t)kt*M+m)*32+kk; uint8_t v=(uint8_t)(A[(size_t)m*K+k0+kt*32+kk]&0xf);
        dst[idx/2]|= (idx&1)?(v<<4):v;
    }
}

ork_w *ork_i4_mm_pack(ork_npu *c,int K,int N,const int8_t *B){
    if(c && c->daemon){ if(K%32||N%64) return NULL; uint64_t id=orkd_pack_i4(c->daemon,K,N,B); if(!id) return NULL; ork_w *w=calloc(1,sizeof *w); if(!w) return NULL; w->is_orkd=1; w->orkd_id=id; w->K=K; w->N=N; w->dtype=DT_I4; return w; }   /* Path B: int4 pack in the daemon */
    if(K%32||N%64) return NULL;
    int KS=ORK_I4_KS, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;  /* wide N-slices ≤ nmax */
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I4; w->owns=1; w->domain=ork_dom(c->pack_domain);
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    for(int ns=0;ns<Sn;ns++)for(int ks=0;ks<Sk;ks++){
        int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=orki_bcreate(c->fd,(size_t)Kp*Nc/2,0x403,w->domain);
        if(!b->cpu){
            fprintf(stderr,"[ork] ERROR: bcreate failed to allocate weight buffer Bb[%zu] in pack_i4 (size=%zu)\n",(size_t)ns*Sk+ks,(size_t)Kp*Nc/2);
            for(int i=0;i<ns*Sk+ks;i++) orki_bdestroy(c->fd,&w->Bb[i]);
            ork_w_free(w); return NULL;
        }
        tile_i4_Bslice(b->cpu,B,K,N,k0,Kp,n0,Nc);
        orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);
    }
    /* SLICE-AND-DICE RESCUE (#33): pre-build doorbell tiles for a REFUSE-PRONE int4 shape (Sn>1 => N>nmax,
     * or K>8192 => BCHAIN H<2) so ork_i4_mm_run's refuse (ORK_RC_WEDGE_PRONE) instead RUNS the shape by
     * decomposing it into BCHAIN-legal sub-tiles (raw nibble B only in scope here — pack_i4 keeps none). The
     * reachable trigger is fused/wide-N int4 prefill (Sn>1, per-core program count > cap). Gated so well-behaved
     * int4 (Sn==1, K<=8192) builds nothing. !orki_in_slice_pack: the sub-tiles below don't recurse. */
    if(!orki_in_slice_pack && !getenv("ORK_NO_SLICE_RESCUE") && ((Sn>1 || K>8192) || getenv("ORK_SLICE_ALL")))
        w->sliced = ork_mm_pack_sliced(c, K, N, B, DT_I4);
    return w;
}

/* CPU-ONLY native-W4A4 dump: produce the SAME bytes as ork_i4_mm_pack() + ork_w_dump(), tiling straight
 * into caller DRAM — no NPU, no IOVA buffer, no DMA round-trip. The int8 twin (ork_i8_w_dump_cpu) already
 * did this; native int4 was the one tier still forced through the NPU just to write a .orkpack, which is
 * what pinned int4 pack-building to the board.
 *
 * It "emulates the NPU" only in the sense of reproducing the CNA's weight TILE LAYOUT — the compute is not
 * involved. And the layout is not re-derived here: it reuses tile_i4_Bslice, the very function the NPU pack
 * calls, so the two cannot drift. What remains is the envelope around it: the same Sn-major/Sk-minor order
 * ork_w_dump walks Bb in, and the same page-padded per-tile stride ork_i4_mm_load expects (a fresh dma-buf
 * is zeroed by the kernel, so the pad must be zeroed here to match byte-for-byte).
 *
 * Correctness has an exact oracle — examples/test_i4_dump_cpu.c packs on the NPU, ork_w_dump's it, and
 * memcmp's against this. Byte-identical or the test fails; the layout cannot be subtly wrong and pass.
 * out=NULL -> return the byte size. K%32, N%64. */
size_t ork_i4_w_dump_cpu(ork_npu *c, int K, int N, const int8_t *B, void *out, size_t cap){
    if(!c || !B || (K%32) || (N%64)) return 0;
    int KS=ORK_I4_KS, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t off=0;
    for(int ns=0;ns<Sn;ns++){ int n0=ns*NMAX, Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){ int k0=ks*KS, Kp=(K-k0<KS)?(K-k0):KS;
        size_t tsz=orki_pgup((size_t)Kp*Nc/2);                 /* nibbles: half a byte per weight */
        if(out){ if(off+tsz>cap) return 0;
            uint8_t *bb=(uint8_t*)out+off; memset(bb,0,tsz);   /* zero the page-pad (matches a fresh dma-buf) */
            tile_i4_Bslice(bb,B,K,N,k0,Kp,n0,Nc); }
        off+=tsz; }}
    return off;
}

ork_w *ork_i4_mm_load(ork_npu *c,int K,int N,const void *blob,size_t n){
    if(K%32||N%64) return NULL;
    int KS=ORK_I4_KS, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t need=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS; need+=orki_pgup((size_t)Kp*Nc/2);}}
    if(n!=need) return NULL;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I4; w->owns=1; w->domain=ork_dom(c->pack_domain);
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    size_t off=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=orki_bcreate(c->fd,(size_t)Kp*Nc/2,0x403,w->domain);
        if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) orki_bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; }
        memcpy(b->cpu,(const char*)blob+off,b->size); off+=b->size;
        orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}}
    return w;
}

ork_w *ork_i4_mm_load_arena(ork_npu *c,int K,int N,const void *blob,size_t n){
    if(K%32||N%64) return NULL;
    if(orki_dmaheap_open()<0) return NULL;
    int KS=ORK_I4_KS, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t need=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;(void)n0;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS; need+=orki_pgup((size_t)Kp*Nc/2);}}
    if(n!=need) return NULL;
    int dom=ork_dom(c->pack_domain); if(dom<0||dom>=64) return NULL;   /* arena tracks [64] domains */
    ork_w *w=calloc(1,sizeof *w); if(!w) return NULL;
    w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I4; w->owns=0; w->domain=dom;   /* shared-chunk views: owns nothing */
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf)); if(!w->Bb){ free(w); return NULL; }
    ork_dom_prime(c, dom);   /* native anchor establishes a non-0 domain BEFORE importing into it (mirror int8) */
    size_t chunk_mb=16; const char*cm=getenv("ORK_IMPORT_CHUNK_MB"); if(cm){ long v=atol(cm); if(v>0) chunk_mb=(size_t)v; }
    size_t chunk_cap=chunk_mb<<20;
    struct buf *cur=&c->i4arena_cur[dom];
    if(!cur->cpu || c->i4arena_off[dom]+need > cur->size){   /* switch chunk: keep this whole weight in ONE chunk */
        size_t csz = need>chunk_cap ? need : chunk_cap;      /* a single expert weight never exceeds 16MB in practice */
        struct buf nb=orki_bimport(c->fd,csz,dom);
        if(!nb.cpu){ free(w->Bb); free(w); return NULL; }
        if(c->i4arena_n>=c->i4arena_cap){ int nc2=c->i4arena_cap?c->i4arena_cap*2:64;
            struct buf*na=realloc(c->i4arena,(size_t)nc2*sizeof*na);
            if(!na){ orki_bdestroy(c->fd,&nb); free(w->Bb); free(w); return NULL; }
            c->i4arena=na; c->i4arena_cap=nc2; }
        /* fds stay OPEN until teardown (bdestroy closes them). 16MB chunks => ~320 fds for the 35B, under ulimit
         * (the EMFILE that forced per-weight fd-sealing was ~9k PER-EXPERT imports; consolidation removes it).
         * Sealing a chunk mid-load — while later chunks in the same domain are still being written — corrupted
         * the next chunk's reads (weight-32 miscompute); int8 only closes fds after a weight's whole load. */
        c->i4arena_curi[dom]=c->i4arena_n; c->i4arena[c->i4arena_n++]=nb; *cur=nb; c->i4arena_off[dom]=0;
    }
    orki_dmabuf_sync(cur->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);   /* bracket the CPU fill (mirror int8) */
    size_t off=c->i4arena_off[dom], boff=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;(void)n0;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS; size_t raw=(size_t)Kp*Nc/2, ts=orki_pgup(raw);
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks];
        b->handle=cur->handle; b->obj=cur->obj; b->dma=cur->dma+off; b->cpu=(char*)cur->cpu+off; b->size=ts; b->heap_fd=0; b->domain=dom;
        memcpy(b->cpu,(const char*)blob+boff,raw); off+=ts; boff+=ts; }}
    orki_dmabuf_sync(cur->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);
    c->i4arena_off[dom]=off;
    return w;
}

ork_w *ork_i4_mm_load_import(ork_npu *c,int K,int N,const void *blob,size_t n){
    if(K%32||N%64) return NULL;
    if(orki_dmaheap_open()<0) return NULL;
    int KS=ORK_I4_KS, NMAX=c->soc->nmax, Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    size_t need=0;
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;(void)n0;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS; need+=orki_pgup((size_t)Kp*Nc/2);}}
    if(n!=need) return NULL;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I4; w->owns=1; w->domain=ork_dom(c->pack_domain);
    c->scratch_import=1;   /* weights are now bimported into their domains -> the run scratch must import too (bcreate EINVALs alongside imports); see bscratch */
    ork_dom_prime(c, w->domain);   /* #54: establish the non-0 domain with ONE native anchor (Quirk 1, NPU-Quirks.md) — MATCH ork_i8_mm_load_import exactly. (Was ork_dom_reanchor/bdestroy+recreate-per-import: destroying the buffer that established the domain is suspected of breaking its lazy-init state at scale; prime-once is the documented proven guard.) */
    /* #54: pre-allocate THIS domain's run scratch NOW, while it is still LIGHT (only the anchor). mc_ensure's
     * mtk_all + per-core mrc/mtk/maf are kernel-mapped and MUST be bcreate — allocated later (at the first run,
     * after this domain fills with imports) a fresh bcreate EINVALs amid the imports (the mc_ensure mtk_all
     * failure). dom_activate makes w->domain the active set (parking the prior domain's); mc_ensure + the mcc
     * ensure below alloc once per domain (idempotent: skip if already sized). Generous mcc (>= the expert BCHAIN
     * need_o ~688 KiB) so BCHAIN never re-grows (=re-bcreates) it in the now-heavy domain. */
    if(w->domain>0){
        orki_dom_activate(c, w->domain);
        orki_mc_ensure(c, c->soc->cores);
        size_t mcc_need = (size_t)2*1024*1024;   /* covers expert BCHAIN need_o with margin */
        for(int i=0;i<c->soc->cores;i++)
            if(c->mccsz[i] < mcc_need){ orki_bdestroy(c->fd,&c->mcc[i]); c->mcc[i]=orki_bscratch(c,mcc_need,0x403,c->dom_active); if(c->mcc[i].cpu){ c->mccsz[i]=mcc_need; c->mwarm[i]=0; } }
    }
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    int consolidate = !getenv("ORK_NO_CONSOLIDATE_IMPORT");
    if(consolidate){
        size_t chunk_mb = 16; const char*cm=getenv("ORK_IMPORT_CHUNK_MB"); if(cm){ long v=atol(cm); if(v>0) chunk_mb=(size_t)v; }
        size_t chunk_cap = chunk_mb<<20;
        int ntiles=Sk*Sn, cap_chunks=ntiles+1;
        w->own_bufs=calloc(cap_chunks,sizeof(struct buf)); w->n_own_bufs=0;
        struct buf cur; cur.cpu=NULL; size_t coff=0, csz=0;
        int ns,ks; size_t boff=0;
        for(ns=0;ns<Sn && consolidate;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;(void)n0;
          for(ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS; size_t ts=orki_pgup((size_t)Kp*Nc/2);
            if(!cur.cpu || coff+ts>csz){
                if(cur.cpu) orki_dmabuf_sync(cur.heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);
                size_t rem = need - boff;                       /* cap the chunk to THIS weight's remaining need: a small per-expert weight (~0.5 MiB) must NOT grab a full chunk_cap (16 MiB) chunk — that burned ~16 MiB IOVA PER expert (~15k experts) and blew the domains. */
                csz = rem < chunk_cap ? rem : chunk_cap; if(csz < ts) csz = ts;
                cur = orki_bimport(c->fd,csz,w->domain);
                if(!cur.cpu){ consolidate=0; break; }
                orki_dmabuf_sync(cur.heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
                w->own_bufs[w->n_own_bufs++]=cur; coff=0;
            }
            struct buf*b=&w->Bb[(size_t)ns*Sk+ks];
            b->handle=cur.handle; b->obj=cur.obj; b->dma=cur.dma+coff; b->cpu=(char*)cur.cpu+coff; b->size=ts;
            memcpy(b->cpu,(const char*)blob+boff,(size_t)Kp*Nc/2); coff+=ts; boff+=ts;}}
        if(consolidate){ if(cur.cpu) orki_dmabuf_sync(cur.heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE); w->owns=0; }
        else { for(int i=0;i<w->n_own_bufs;i++) orki_bdestroy(c->fd,&w->own_bufs[i]);
            free(w->own_bufs); w->own_bufs=NULL; w->n_own_bufs=0;
            memset(w->Bb,0,(size_t)Sk*Sn*sizeof(struct buf)); w->owns=1; }
    }
    if(!consolidate){
        size_t off=0;
        for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;(void)n0;
          for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;
            struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=orki_bimport(c->fd,(size_t)Kp*Nc/2,w->domain);
            if(!b->cpu){ for(int i=0;i<ns*Sk+ks;i++) orki_bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; }
            orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE);
            memcpy(b->cpu,(const char*)blob+off,(size_t)Kp*Nc/2); off+=b->size;
            orki_dmabuf_sync(b->heap_fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE);}}
    }
    /* SEAL the dma-buf fds now that the CPU fill + coherency sync are done. The GEM handle (from MEM_CREATE) and
     * the mmap keep the buffer alive for NPU reads (via IOMMU b->dma); a resident int4 weight is READ-ONLY after
     * load, so no later dmabuf_sync is needed (verified: dmabuf_sync is called only in the load/import path, never
     * per-run). This drops the held-fd count from ~1 per weight (~16k for the 35B MoE -> blew the fd ulimit) to ~0:
     * load_i4_import runs per-weight, so each weight's fds close before the next weight imports. heap_fd=0 tells
     * bdestroy the fd is already closed (it still MEM_DESTROYs via the handle). Consolidated tiles are VIEWS
     * (heap_fd=0 already, never bdestroy'd individually); only the chunks (own_bufs) and per-tile bufs hold fds. */
    if(!getenv("ORK_NO_SEAL")){   /* TEST: does closing the dma-buf fd (int4-only; int8 keeps them open) alias the next import? */
    for(int i=0;i<w->n_own_bufs;i++){ if(w->own_bufs[i].heap_fd>0){ close(w->own_bufs[i].heap_fd); w->own_bufs[i].heap_fd=0; } }
    for(int i=0;i<Sk*Sn;i++){ if(w->Bb[i].heap_fd>0){ close(w->Bb[i].heap_fd); w->Bb[i].heap_fd=0; } }
    }
    return w;
}

ork_w *ork_i4_mm_pack_grouped(ork_npu *c,int K,int N,const int8_t *B,int G){
    if(K%32||N%64||G%32||K%G||G>ORK_I4_KS) return NULL;
    int NMAX=c->soc->nmax, Sk=K/G, Sn=(N+NMAX-1)/NMAX;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn;w->dtype=DT_I4;w->gsize=G; w->domain=ork_dom(c->pack_domain);
    w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    /* Reserve the whole weight as ONE dedicated DMA buffer (own_buf); each group-tile is a 4KB-aligned
     * VIEW into it (shared obj, dma=own_buf.dma+off). Collapses the per-group bcreate storm to a single
     * allocation => fast warmup, no IOVA-handle OOM, and — crucially — RECLAIMABLE: ork_mm_free destroys
     * own_buf (returning its IOVA to the 4 GiB window), so drop/reload of a grouped weight does NOT leak
     * (streaming / MoE-swap). The whole region is flushed to device in a single bsync. Falls back to
     * per-tile owning orki_bcreate (also reclaimable) if the dedicated alloc fails. */
    size_t wtotal=0;
    for(int ns=0;ns<Sn;ns++)for(int g=0;g<Sk;g++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
        wtotal += (((size_t)G*Nc/2)+4095u)&~(size_t)4095u; }
    struct buf own=orki_bcreate(c->fd,wtotal,0x403,w->domain);
    if(own.cpu){
        w->own_buf=own; w->own_buf_valid=1;
        size_t off=0;
        for(int ns=0;ns<Sn;ns++)for(int g=0;g<Sk;g++){
            int k0=g*G,n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX; size_t ts=(size_t)G*Nc/2;
            struct buf*b=&w->Bb[(size_t)ns*Sk+g];
            b->handle=own.handle; b->obj=own.obj; b->dma=own.dma+off; b->cpu=(char*)own.cpu+off; b->size=ts;
            tile_i4_Bslice(b->cpu,B,K,N,k0,G,n0,Nc);
            off += (ts+4095u)&~(size_t)4095u;
        }
        orki_bsync_off(c->fd,own.obj,0,wtotal,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
        orki_bsync_off(c->fd,own.obj,0,wtotal,RKNPU_MEM_SYNC_TO_DEVICE);
    } else {
        w->owns=1;   /* per-tile owning bcreate: reclaimable by ork_mm_free */
        for(int ns=0;ns<Sn;ns++)for(int g=0;g<Sk;g++){
            int k0=g*G,n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
            struct buf*b=&w->Bb[(size_t)ns*Sk+g]; *b=orki_bcreate(c->fd,(size_t)G*Nc/2,0x403,w->domain);
            if(!b->cpu){ fprintf(stderr,"[ork] ERROR: weight alloc failed (G=%d Nc=%d) in pack_i4_grouped\n",G,Nc);
                for(int i=0;i<ns*Sk+g;i++) orki_bdestroy(c->fd,&w->Bb[i]); free(w->Bb); free(w); return NULL; }
            tile_i4_Bslice(b->cpu,B,K,N,k0,G,n0,Nc);
            orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE); orki_bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);
        }
    }
    return w;
}

ork_w *ork_i4_mm_pack_to_i8(ork_npu *c, int K, int N, const int8_t *B) {
    /* The core fallback: it takes int4-range values ([-8,7]) unpacked in int8_t containers,
     * but physically packs them into the highly optimized int8 resident weight layout.
     * This simply defers to ork_i8_mm_pack, yielding maximum int8 silicon execution speed
     * while the caller (e.g. ggml-ork) maintains the 50% footprint reduction on disk. */
    return ork_i8_mm_pack(c, K, N, B);
}

/* int4 (W4A4) sub-weight packer (#33): twin of orki_i8_slice_pack, but the tile envelope is BCHAIN's
 * (run_i4_bchain_db, the per-tile executor): each sub-tile must be Sk==1, Sn==1, N%64==0, and K<=8192 so
 * BCHAIN's H=16384/K>=2. So K-slice at ks=8192 (K padded to 32 — pack_i4 needs K%32; pad rows are zeroed ->
 * contribute 0), N-tile at ns=8192 (Sn==1; BCHAIN N-tiles further by bank-width internally). B is the int8
 * nibble-container [-8,7] (pack_i4's input); a native pack_i4 weight keeps no raw nibbles, so sub-tiles are
 * re-packed from the caller's B here (as orki_i8_slice_pack does with ork_i8_mm_pack). */
ork_w_sliced *orki_i4_slice_pack(ork_npu *c, int K, int N, const int8_t *B) {
    if (N % 64) return NULL;                                             /* pack_i4 requires N%64 (a real int4 weight satisfies it); N is not padded */
    int Kpad = ((K + 31) / 32) * 32;                                    /* pad K to 32 (pack_i4 K%32); zero rows contribute 0 -> bit-exact */
    int ks = 8192, ns = 8192;                                           /* K-slice <=8192 (BCHAIN H>=2); N-tile <=8192 (Sn==1). both %64 & %32 */
    int nks = (Kpad + ks - 1) / ks, nnt = (N + ns - 1) / ns;
    struct ork_w_sliced *w = calloc(1, sizeof *w); if (!w) return NULL;
    w->K = K; w->N = N; w->Kpad = Kpad; w->dtype = DT_I4; w->cap = ork_slice_caps_rk3588(); w->nks = nks; w->nnt = nnt; w->ks = ks; w->ns = ns;
    w->sub = calloc((size_t) nks * nnt, sizeof(ork_w *));
    int8_t *blk = malloc((size_t) ks * ns);
    if (!w->sub || !blk) { free(blk); ork_mm_free_sliced(c, w); return NULL; }
    orki_in_slice_pack = 1;
    for (int ki = 0; ki < nks; ki++) { int k0 = ki*ks, k1 = k0+ks < Kpad ? k0+ks : Kpad, Ks = k1-k0;
        for (int ni = 0; ni < nnt; ni++) { int n0 = ni*ns, n1 = n0+ns < N ? n0+ns : N, Nw = n1-n0;
            for (int k = 0; k < Ks; k++) { if (k0+k < K) memcpy(blk + (size_t) k*Nw, B + (size_t)(k0+k)*N + n0, Nw);   /* real nibble row */
                                           else          memset(blk + (size_t) k*Nw, 0, Nw); }                        /* PAD row -> zero */
            ork_w *sw = ork_i4_mm_pack(c, Ks, Nw, blk);                 /* Sk==1, Sn==1, N%64 tile -> BCHAIN-eligible */
            if (!sw) { orki_in_slice_pack = 0; free(blk); ork_mm_free_sliced(c, w); return NULL; }
            w->sub[ki*nnt + ni] = sw; } }
    orki_in_slice_pack = 0; free(blk); return w;
}

void orki_i4_tile_A(uint8_t*dst,const int8_t*A,int M,int K,int nib){
    int KT=K/32; memset(dst,0,(size_t)M*K/2);
    for(int kt=0;kt<KT;kt++)for(int m=0;m<M;m++)for(int kk=0;kk<32;kk++){
        size_t idx=((size_t)kt*M+m)*32+kk; uint8_t v=(uint8_t)(A[(size_t)m*K+kt*32+kk]&0xf);
        dst[idx/2]|= ((idx&1)^nib)?(v<<4):v;
    }
}

void orki_i4_tile_B(uint8_t*dst,const int8_t*B,int K,int N,int nib){
    int KT=K/32,NT=N/64; memset(dst,0,(size_t)K*N/2);
    for(int nt=0;nt<NT;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<64;nl++)for(int kk=0;kk<32;kk++){
        size_t idx=(((size_t)nt*KT+kt)*64+nl)*32+kk;
        uint8_t v=(uint8_t)(B[(size_t)(kt*32+kk)*N + (nt*64+nl)]&0xf);
        dst[idx/2]|= ((idx&1)^nib)?(v<<4):v;
    }
}

/* CPU-ONLY int4 pack straight to the compact .orkpack blob (header + bscale[N] + Bi4[K*N/2]) — byte-
 * identical to ork_i4a8_mm_pack_im() + ork_i4a8_w_dump(), but with NO bcreate/IOMMU/tiling. The per-tile
 * bcreate in the NPU int4 packer is the serial single-stream consumer that bottlenecks .orkpack conversion;
 * a WRITE only needs the compact nibbles + scales on disk (ork_i4a8_mm_load re-tiles at load). This
 * replicates the EXACT per-channel quant of ork_i4a8_mm_pack_im (absmax/7 uniform or NF4 codebook, optional
 * imatrix clip-grid, SR with a per-call seed) so the bytes match. Single-threaded (caller parallelizes over
 * experts). out=NULL → required size. K%32,N%32. */
size_t ork_i4a8_pack_cpu_blob(ork_npu *c, int K, int N, const float *f32, const float *imatrix, int nf4, void *out, size_t cap){
    (void)c;
    if(K%32 || N%32 || !f32) return 0;
    size_t hdr=sizeof(struct ork_i4a8_hdr), sc=(size_t)N*sizeof(float), nibsz=(size_t)K*N/2, need=hdr+sc+nibsz;
    if(!out) return need;
    if(cap<need) return 0;
    nf4 = nf4 ? 1 : 0;   /* codebook routed by the caller (source-based), not an env flag */
    int sr  = getenv("ORK_SR")!=NULL; uint32_t seed=0x2545F491u;   /* per-call seed matches ork_i4a8_mm_pack_im */
    struct ork_i4a8_hdr h={ORK_I4A8_MAGIC, ORK_I4A8_VER, K, N, (uint32_t)(nf4?ORK_QK_CODEBOOK_NF4:ORK_QK_UNIFORM)};
    char    *p=(char*)out;
    float   *bscale=(float*)(p+hdr);
    uint8_t *Bi4   =(uint8_t*)(p+hdr+sc);
    float   *qf32=malloc((size_t)K*sizeof(float));       /* orki_i4_quant_chan code byproduct; reused per channel */
    uint8_t *qidx=nf4?malloc((size_t)K):NULL;
    float   *imdq=imatrix?malloc((size_t)K*sizeof(float)):NULL;
    if(!qf32 || (nf4&&!qidx) || (imatrix&&!imdq)){ free(qf32); free(qidx); free(imdq); return 0; }
    for(int n=0;n<N;n++){
        const float *fr=f32+(size_t)n*K; float mx=1e-9f; int k=0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        float32x4_t vmx=vdupq_n_f32(1e-9f);
        for(;k<=K-4;k+=4) vmx=vmaxq_f32(vmx,vabsq_f32(vld1q_f32(fr+k)));
        float m[4]; vst1q_f32(m,vmx); float a=m[0]>m[1]?m[0]:m[1], bb=m[2]>m[3]?m[2]:m[3]; mx=a>bb?a:bb;
#endif
        for(;k<K;k++){ float v=fabsf(fr[k]); if(v>mx) mx=v; }
        if(imatrix) mx=orki_wq_best_absmax(fr,K,mx,nf4,imatrix,imdq);
        uint8_t *nib=Bi4+(size_t)n*(K/2);
        if(nf4){ bscale[n]=mx/127.0f; orki_nf4_quant_chan(fr,K,mx,sr,&seed,nib,qidx); }
        else   { float scale=mx/7.0f; bscale[n]=scale; orki_i4_quant_chan(fr,K,scale,sr,&seed,nib,qf32); }
    }
    memcpy(p,&h,hdr);   /* header last: bscale/Bi4 already in place */
    free(qf32); free(qidx); free(imdq);
    return need;
}
