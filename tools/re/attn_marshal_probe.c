/* attn_marshal_probe — where does the ATTENTION-on-NPU wall actually go?
 *
 * ggml-ork's batched-attention handler (ggml_backend_ork_bmm_fp16) spends, per head:
 *   B gather   : K*N elements, strided scalar read of the KV-cache view -> row-major fp16 B[K,N]
 *   A gather   : M*K elements, f32 -> fp16
 *   repack     : ork_f16_mm_repack, row-major B[K,N] -> the NPU tile [NT][KT][16][32]
 *   run        : ork_f16_mm_run (the only part that is NPU work)
 *   C scatter  : M*N floats out
 * The FA row of OPS_REGISTRY says 71% of that wall is the non-run part. This measures the split at
 * the REAL attention shapes, for QK^T and A.V SEPARATELY (their shapes are transposes of each other,
 * so their marshalling profiles are not the same), and prices the two obvious repairs:
 *   FASTGATHER  — the gather exploits that the contraction dim is stride-1 in BOTH KV views, so the
 *                 "transpose" is really a gather of 32-element contiguous runs, not element-wise.
 *   DIRECT TILE — skip the row-major intermediate entirely and write the NPU tile from the source.
 * Board-only (needs /dev/dri). Diagnostic; not in `all`/`test`. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "ork_npu.h"

/* diagnostic tile accessors (src/npu.c) — the probe writes the NPU tile bytes directly */
int    ork_w_ntiles(const ork_w *w);
void  *ork_w_tile_cpu(const ork_w *w, int i);
size_t ork_w_tile_size(const ork_w *w, int i);
void   ork_w_tile_clean(ork_npu *c, const ork_w *w, int i);

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec*1e-3; }

typedef unsigned short u16;
static inline u16 f2h(float f){ _Float16 h=(_Float16)f; u16 o; memcpy(&o,&h,2); return o; }
static inline float h2f(u16 h){ _Float16 x; memcpy(&x,&h,2); return (float)x; }

/* ---- the handler's own loops, transcribed ----------------------------------------------------
 * src0 is the [K,N] "weight" side, element (i0,i1) at base + i0*nb0 + i1*nb1 (BYTES).
 * B[k*N+n] = src0(k,n).  A[m*K+k] = src1(k,m). */
static void b_gather_scalar(u16 *B,const char *s0,int K,int N,long nb0,long nb1,int f16src){
    for(long j=0;j<(long)K*N;j++){
        const char *p = s0 + (j/N)*nb0 + (j%N)*nb1;
        B[j] = f2h(f16src ? h2f(*(const u16*)p) : *(const float*)p);
    }
}
/* FASTGATHER: nb0 is the stride along the CONTRACTION dim. In both KV views it is elem-size (the
 * contraction dim is stride-1), so column n of B is a CONTIGUOUS run of K source elements. Walk n
 * outermost and scatter into B with stride N -> one source cache line serves 32 writes. */
static void b_gather_fast(u16 *B,const char *s0,int K,int N,long nb0,long nb1,int f16src){
    if(nb0!=(f16src?2:4)){ b_gather_scalar(B,s0,K,N,nb0,nb1,f16src); return; }
    for(int n=0;n<N;n++){
        const char *src=s0+(long)n*nb1; u16 *d=B+n;
        if(f16src){ const u16 *s=(const u16*)src; for(int k=0;k<K;k++) d[(long)k*N]=s[k]; }
        else      { const float *s=(const float*)src; for(int k=0;k<K;k++) d[(long)k*N]=f2h(s[k]); }
    }
}
static void a_gather_scalar(u16 *A,const char *s1,int M,int K,long nb0,long nb1,int f16src){
    for(long j=0;j<(long)M*K;j++){
        const char *p = s1 + (j%K)*nb0 + (j/K)*nb1;
        A[j] = f2h(f16src ? h2f(*(const u16*)p) : *(const float*)p);
    }
}
static void a_gather_fast(u16 *A,const char *s1,int M,int K,long nb0,long nb1,int f16src){
    if(nb0!=(f16src?2:4)){ a_gather_scalar(A,s1,M,K,nb0,nb1,f16src); return; }
    for(int m=0;m<M;m++){ const char *src=s1+(long)m*nb1; u16 *d=A+(long)m*K;
        if(f16src) memcpy(d,src,(size_t)K*2);
        else { const float *s=(const float*)src; int k=0;
               for(;k<K;k++) d[k]=f2h(s[k]); } }
}

/* ---- DIRECT TILE: source -> the NPU weight tile, no row-major intermediate ---------------------
 * tile(ns,ks) holds [NN][KT][16][32] with element (kt*32+kk, nt*16+nl) at [nt][kt][nl][kk].
 * With the contraction dim stride-1 in the source, [nl][kk] for fixed (nt,kt) is 16 runs of 32
 * CONTIGUOUS source elements -> 16 memcpy-sized copies per 1 KB tile block. */
static void direct_tile(ork_npu *c,ork_w *w,const char *s0,int K,int N,long nb0,long nb1,int f16src,int KS,int NMAX){
    int Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    for(int ns=0;ns<Sn;ns++){ int n0=ns*NMAX, Nc=(N-n0<NMAX)?(N-n0):NMAX, NN=Nc/16;
      for(int ks=0;ks<Sk;ks++){ int k0=ks*KS, Kp=(K-k0<KS)?(K-k0):KS, KT=Kp/32;
        u16 *bb=ork_w_tile_cpu(w,ns*Sk+ks); if(!bb) continue;
        for(int nt=0;nt<NN;nt++) for(int nl=0;nl<16;nl++){
            const char *src = s0 + (long)(n0+nt*16+nl)*nb1 + (long)k0*nb0;
            for(int kt=0;kt<KT;kt++){
                u16 *d = bb + (size_t)nt*KT*16*32 + (size_t)kt*16*32 + (size_t)nl*32;
                if(f16src) memcpy(d, src+(long)kt*32*2, 64);
                else { const float *s=(const float*)(src+(long)kt*32*4); for(int kk=0;kk<32;kk++) d[kk]=f2h(s[kk]); }
            } }
        ork_w_tile_clean(c,w,ns*Sk+ks);
      } }
}

struct res { double bs,bf,bd,bd2,as,af,rp,rn,sc,io,hw,nsub; };

static void one(ork_npu *c,const char *tag,int M,int K,int N,long nb1_b,int f16b,long nb1_a,int f16a,int iters){
    size_t nB=(size_t)K*N, nA=(size_t)M*K, nC=(size_t)M*N;
    long eb = f16b?2:4, ea = f16a?4:4;   /* A source is always f32 here (Q after RoPE / softmax out) */
    (void)ea;
    char *s0 = malloc((size_t)N*nb1_b + 64);          /* [N][K] source: contraction dim stride-1 */
    char *s1 = malloc((size_t)M*nb1_a + 64);
    u16  *B  = malloc(nB*2), *Bref = malloc(nB*2), *A = malloc(nA*2), *Aref = malloc(nA*2);
    float *C = malloc(nC*4), *D = malloc(nC*4);
    if(!s0||!s1||!B||!Bref||!A||!Aref||!C||!D){ printf("  %s: OOM\n",tag); return; }
    for(size_t i=0;i<(size_t)N*nb1_b/ (size_t)eb; i++){ float v=(float)((i*2654435761u)%1000)/1000.0f-0.5f;
        if(f16b) ((u16*)s0)[i]=f2h(v); else ((float*)s0)[i]=v; }
    for(size_t i=0;i<(size_t)M*nb1_a/4;i++) ((float*)s1)[i]=(float)((i*40503u)%997)/997.0f-0.5f;

    ork_w *w = ork_f16_mm_scratch(c,K,N);
    if(!w){ printf("  %s: scratch K=%d N=%d FAILED\n",tag,K,N); return; }

    struct res r; memset(&r,0,sizeof r);
    double t;
    t=now_us(); for(int i=0;i<iters;i++) b_gather_scalar(Bref,s0,K,N,eb,nb1_b,f16b); r.bs=(now_us()-t)/iters;
    t=now_us(); for(int i=0;i<iters;i++) b_gather_fast  (B,   s0,K,N,eb,nb1_b,f16b); r.bf=(now_us()-t)/iters;
    int bok = memcmp(B,Bref,nB*2)==0;
    t=now_us(); for(int i=0;i<iters;i++) a_gather_scalar(Aref,s1,M,K,4,nb1_a,0);     r.as=(now_us()-t)/iters;
    t=now_us(); for(int i=0;i<iters;i++) a_gather_fast  (A,   s1,M,K,4,nb1_a,0);     r.af=(now_us()-t)/iters;
    int aok = memcmp(A,Aref,nA*2)==0;
    t=now_us(); for(int i=0;i<iters;i++) ork_f16_mm_repack(c,w,K,N,(const ork_f16*)B);               r.rp=(now_us()-t)/iters;
    ork_f16_mm_run(c,w,M,(const ork_f16*)A,C);   /* warm: first fp16 run does a cold 2-pass re-warm */
    ork_npu_floor_reset();
    t=now_us(); for(int i=0;i<iters;i++) ork_f16_mm_run(c,w,M,(const ork_f16*)A,C);                  r.rn=(now_us()-t)/iters;
    { double io,hw; long long raw; long nsub; ork_npu_floor_timing(&io,&hw,&raw,&nsub);
      r.io=io/iters; r.hw=hw/iters; r.nsub=(double)nsub/iters; }
    { double su,sb,cp; long nn; ork_npu_run_timing(&su,&sb,&cp,&nn);
      printf("      run_multicore phases (cumulative, %ld calls): setup %8.1f  submit %8.1f  copy %8.1f us\n",nn,su,sb,cp);
      for(int cc=0;cc<3;cc++){ double a,b2,d2; long n2; ork_npu_mc_timing(cc,&a,&b2,&d2,&n2);
        if(n2) printf("        core%d: copy %8.1f  submit %8.1f  acc %8.1f us over %ld progs\n",cc,a,b2,d2,n2); } }
    t=now_us(); for(int i=0;i<iters;i++) memcpy(D,C,nC*4);                           r.sc=(now_us()-t)/iters;

    /* direct tile: must reproduce the repack bytes exactly */
    int KS=1024, NMAX=0; { int pr=0; ork_f16_mm_plan(c,M,K,N,&pr); }
    NMAX = N;                                   /* probe shapes are single-N-slice (N<=2048) */
    ork_f16_mm_repack(c,w,K,N,(const ork_f16*)Bref);
    int nt_=ork_w_ntiles(w); size_t tb=0; for(int i=0;i<nt_;i++) tb+=ork_w_tile_size(w,i);
    unsigned char *save=malloc(tb); size_t off=0;
    for(int i=0;i<nt_;i++){ memcpy(save+off,ork_w_tile_cpu(w,i),ork_w_tile_size(w,i)); off+=ork_w_tile_size(w,i); }
    /* the DRIVER's strided repack (ork_f16_mm_repack_strided) — pooled, this is the landable path.
     * r.bd2 is the single-threaded prototype, kept so the pool's contribution is visible. */
    t=now_us(); for(int i=0;i<iters;i++) direct_tile(c,w,s0,K,N,eb,nb1_b,f16b,KS,NMAX); r.bd2=(now_us()-t)/iters;
    t=now_us(); for(int i=0;i<iters;i++) ork_f16_mm_repack_strided(c,w,K,N,s0,1,nb1_b/eb,!f16b); r.bd=(now_us()-t)/iters;
    off=0; int dok=1;
    for(int i=0;i<nt_;i++){ size_t sz=ork_w_tile_size(w,i);
        /* compare only the live prefix: the tile buffer is page-padded */
        if(memcmp(save+off,ork_w_tile_cpu(w,i),sz)) dok=0; off+=sz; }
    free(save);

    double mac=(double)M*K*N;
    double tot_s = r.bs+r.as+r.rp+r.rn+r.sc;
    double tot_f = r.bd+r.af+r.rn+r.sc;
    printf("  %-6s M=%-5d K=%-5d N=%-5d  %6.1f MMAC | tiles=%d\n", tag,M,K,N,mac/1e6,nt_);
    printf("      CURRENT  Bgat %8.1f  Agat %8.1f  repack %8.1f  RUN %8.1f  scat %7.1f us  = %8.1f us  %6.1f GMAC/s (run alone %6.1f)\n",
           r.bs,r.as,r.rp,r.rn,r.sc,tot_s,mac/tot_s/1e3,mac/r.rn/1e3);
    printf("      RUN split: %5.1f submits  ioctl %8.1f us (%4.1f%% of run)  NPU-hw %8.1f us (%4.1f%%)  -> hw-only %6.1f GMAC/s\n",
           r.nsub, r.io, 100.0*r.io/r.rn, r.hw, 100.0*r.hw/r.rn, r.hw>0?mac/r.hw/1e3:0.0);
    printf("      FAST     Bdir %8.1f  Agat %8.1f  (repack folded)   RUN %8.1f  scat %7.1f us  = %8.1f us  %6.1f GMAC/s   [Bfast %.1f]\n",
           r.bd,r.af,r.rn,r.sc,tot_f,mac/tot_f/1e3,r.bf);
    printf("      (strided repack: driver/pooled %.1f us  vs single-thread prototype %.1f us)\n", r.bd, r.bd2);
    printf("      marshal share: current %4.1f%%  fast %4.1f%%   speedup %.2fx   exact: Bfast=%s Afast=%s Bdirect=%s\n",
           100.0*(tot_s-r.rn)/tot_s, 100.0*(tot_f-r.rn)/tot_f, tot_s/tot_f,
           bok?"ok":"MISMATCH", aok?"ok":"MISMATCH", dok?"ok":"MISMATCH");
    ork_mm_free(c,w);
    free(s0);free(s1);free(B);free(Bref);free(A);free(Aref);free(C);free(D);
}

int main(int argc,char**argv){
    int M   = argc>1?atoi(argv[1]):512;    /* ubatch tokens */
    int HD  = argc>2?atoi(argv[2]):128;    /* head_dim */
    int it  = argc>3?atoi(argv[3]):3;
    int kvs[8]; int nkv=0;
    if(argc>4){ for(int i=4;i<argc&&nkv<8;i++) kvs[nkv++]=atoi(argv[i]); }
    else { kvs[0]=512; kvs[1]=1024; kvs[2]=2048; nkv=3; }

    ork_npu *c=ork_npu_init(); if(!c){ fprintf(stderr,"init failed\n"); return 1; }
    printf("attn_marshal_probe  M=%d head_dim=%d iters=%d   (ork %s)\n",M,HD,it,ork_npu_version());
    for(int i=0;i<nkv;i++){
        int KV=kvs[i];
        printf("\n== n_kv = %d ==\n",KV);
        /* QK^T : src0 = K-cache view [head_dim, n_kv], head_dim stride-1, row stride n_embd_k_gqa(=HD*8)
         *        src1 = Q [head_dim, n_tokens] f32, head_dim stride-1 */
        one(c,"QK^T",M,HD,KV,(long)HD*8*2,1,(long)HD*4,0,it);
        /* A.V  : src0 = V^T view [n_kv, head_dim], n_kv stride-1, row stride n_ctx(=4096) f16
         *        src1 = softmax scores [n_kv, n_tokens] f32 contiguous */
        one(c,"A.V", M,KV,HD,(long)4096*2,1,(long)KV*4,0,it);
    }
    ork_npu_free(c);
    return 0;
}
