/* direct_i4_test — verify + bench the DIRECT int4->int8-tiled inflate (ORK_DIRECT_I4) against the
 * f32 round-trip path. Board-only (needs /dev/dri + rknpu). NOT in `all`/`test`.
 *
 *   Correctness: pack a weight (UNIFORM or NF4) -> dump i4a8 blob -> load twice (f32 path / direct path)
 *   -> memcmp the tiled DMA bytes (Bb + Bf) of both ork_w. Must be byte-identical (memcmp==0). Then run
 *   ork_mm_run_i8 on each and on a CPU int4 reference; the int32 outputs must match exactly.
 *
 *   Bench: time inflate+tile only (into the already-resident DMA tiles) — old f32 path
 *   (ork_slice_inflate_i4a8_kind + ork_slice_tile_i8) vs new direct (ork_slice_direct_i4a8_kind),
 *   for UNIFORM and NF4, over a set of FFN shapes. median of >=20 reps.
 *
 *   build:  make direct_i4_test     run:  sudo taskset -c 4-7 ./direct_i4_test
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "ork_npu.h"

/* mirror of the internal struct prefixes in src/npu.c (stable layout) so the test can memcmp tiled bytes */
struct buf { uint32_t handle; uint64_t dma, obj; void *cpu; size_t size; };
struct ork_w_pub { int K, N, Sk, Sn, dtype, gsize; struct buf *Bb; struct buf *Bf; int owns;
                   uint8_t *Bi4; size_t Bi4_bytes; uint8_t quant_kind; float *bscale; };

/* internal diagnostics exported from src/npu.c */
void ork_slice_inflate_i4a8_kind(const ork_w *w, float *qf32, int kind);
void ork_slice_tile_i8(ork_npu *c, ork_w *w, const float *qf32, float *inv1);
void ork_slice_direct_i4a8_kind(ork_npu *c, ork_w *w, int8_t *i8scratch, int kind);
void ork_slice_direct_inflate_i8(const ork_w *w, int8_t *i8, int kind);

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y?1:0; }
static double median(double*s,int n){ qsort(s,n,sizeof(double),cmp_d); return n&1?s[n/2]:0.5*(s[n/2-1]+s[n/2]); }

/* total resident tiled bytes (Bb + Bf), used to bound a memcmp over all sub-buffers */
static int tiles_equal(struct ork_w_pub *a, struct ork_w_pub *b){
    if(a->K!=b->K||a->N!=b->N||a->Sk!=b->Sk||a->Sn!=b->Sn) return 0;
    for(int i=0;i<a->Sk*a->Sn;i++){
        struct buf *ba=&a->Bb[i], *bb=&b->Bb[i];
        if((!!ba->cpu)!=(!!bb->cpu)) return 0;
        if(ba->cpu && (ba->size!=bb->size || memcmp(ba->cpu,bb->cpu,ba->size))) return 0;
    }
    if((!!a->Bf)!=(!!b->Bf)) return 0;
    if(a->Bf) for(int ns=0;ns<a->Sn;ns++){
        struct buf *ba=&a->Bf[ns], *bb=&b->Bf[ns];
        if((!!ba->cpu)!=(!!bb->cpu)) return 0;
        if(ba->cpu && (ba->size!=bb->size || memcmp(ba->cpu,bb->cpu,ba->size))) return 0;
    }
    return 1;
}

/* CPU int4 reference: C[m][n] = sum_k A_i8[m][k] * code[n][k], code from the nibble store (same inflate
 * the lib does). Validates the matmul against an independent dequant of the SAME nibbles. */
static void cpu_ref_i4(struct ork_w_pub *w, int M, const int8_t *A, int32_t *C){
    int K=w->K, N=w->N;
    static const float NF4[16]={-1.0f,-0.6961928009986877f,-0.5250730514526367f,-0.39491748809814453f,
        -0.28444138169288635f,-0.18477343022823334f,-0.09105003625154495f,0.0f,0.07958029955625534f,
        0.16093020141124725f,0.24611230194568634f,0.33791524171829224f,0.44070982933044434f,
        0.5626170039176941f,0.7229568362236023f,1.0f};
    int8_t lut[16]; for(int i=0;i<16;i++) lut[i]=(int8_t)lrintf(NF4[i]*127.0f);
    int8_t *code=malloc((size_t)N*K);
    for(int n=0;n<N;n++){ const uint8_t *nib=w->Bi4+(size_t)n*(K/2);
        for(int k=0;k<K;k++){ uint8_t nb=(k&1)?(nib[k>>1]>>4):(nib[k>>1]&0xf);
            code[(size_t)n*K+k] = (w->quant_kind==ORK_QK_CODEBOOK_NF4)? lut[nb] : (int8_t)((int8_t)(nb<<4)>>4); } }
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ long acc=0; const int8_t *a=A+(size_t)m*K, *cd=code+(size_t)n*K;
        for(int k=0;k<K;k++) acc += (int)a[k]*(int)cd[k]; C[(size_t)m*N+n]=(int32_t)acc; }
    free(code);
}

static ork_w *load_path(ork_npu *c,int K,int N,const void*blob,size_t bn,int direct){
    if(direct) setenv("ORK_DIRECT_I4","1",1); else unsetenv("ORK_DIRECT_I4");
    ork_w *w = ork_mm_load_i4a8(c,K,N,blob,bn);
    unsetenv("ORK_DIRECT_I4");
    return w;
}

static int verify(ork_npu *c,int K,int N,int nf4){
    const char*kn=nf4?"NF4":"UNIFORM"; int rc=0;
    float *f32=malloc((size_t)N*K*sizeof(float));
    uint32_t s=0x12345u; for(size_t i=0;i<(size_t)N*K;i++){ s^=s<<13;s^=s>>17;s^=s<<5; f32[i]=((int)(s&0xffff)-32768)/9000.0f; }
    if(nf4) setenv("ORK_NF4","1",1); else unsetenv("ORK_NF4");
    ork_w *wp = ork_mm_pack_i4a8(c,K,N,f32,NULL);
    unsetenv("ORK_NF4");
    if(!wp){ printf("  [%s K%d N%d] pack FAIL\n",kn,K,N); free(f32); return 1; }
    size_t bn = ork_w_dump_i4a8(wp,NULL,0); void*blob=malloc(bn);
    if(ork_w_dump_i4a8(wp,blob,bn)!=bn){ printf("  [%s] dump FAIL\n",kn); rc=1; goto out; }

    ork_w *wf = load_path(c,K,N,blob,bn,0);
    ork_w *wd = load_path(c,K,N,blob,bn,1);
    if(!wf||!wd){ printf("  [%s] load FAIL\n",kn); rc=1; goto out; }

    int eq = tiles_equal((struct ork_w_pub*)wf,(struct ork_w_pub*)wd);
    printf("  [%s K%d N%d] tiled DMA bytes memcmp: %s\n",kn,K,N, eq?"BIT-IDENTICAL (PASS)":"DIFFER (FAIL)");
    if(!eq) rc=1;

    /* matmul equality: f32-path weight vs direct-path weight vs CPU int4 ref */
    int M=8; int8_t *A=malloc((size_t)M*K);
    for(size_t i=0;i<(size_t)M*K;i++){ s^=s<<13;s^=s>>17;s^=s<<5; A[i]=(int8_t)((int)(s&0xff)-128); }
    int32_t *Cf=malloc((size_t)M*N*4), *Cd=malloc((size_t)M*N*4), *Cr=malloc((size_t)M*N*4);
    if(ork_mm_run_i8(c,wf,M,A,Cf)||ork_mm_run_i8(c,wd,M,A,Cd)){ printf("  [%s] run FAIL\n",kn); rc=1; }
    else {
        cpu_ref_i4((struct ork_w_pub*)wd,M,A,Cr);
        int dfd=memcmp(Cf,Cd,(size_t)M*N*4); int dfr=memcmp(Cd,Cr,(size_t)M*N*4);
        printf("  [%s] matmul: f32-path vs direct: %s ; direct vs CPU-ref: %s\n",
            kn, dfd?"DIFFER (FAIL)":"EXACT (PASS)", dfr?"DIFFER (FAIL)":"EXACT (PASS)");
        if(dfd||dfr) rc=1;
    }
    free(A);free(Cf);free(Cd);free(Cr);
    ork_mm_free(c,wf); ork_mm_free(c,wd);
out:
    free(blob); ork_mm_free(c,wp); free(f32);
    return rc;
}

static void bench(ork_npu *c,int K,int N){
    enum{WARM=5,REPS=25};
    /* a packed weight gives resident DMA tiles + a nibble store to inflate from repeatedly */
    float *f32=malloc((size_t)N*K*sizeof(float));
    uint32_t s=0x9a3u; for(size_t i=0;i<(size_t)N*K;i++){ s^=s<<13;s^=s>>17;s^=s<<5; f32[i]=((int)(s&0xffff)-32768)/9000.0f; }
    ork_w *w = ork_mm_pack_i4a8(c,K,N,f32,NULL); free(f32);
    if(!w){ printf("  K%d N%d pack FAIL\n",K,N); return; }
    float *qf32=malloc((size_t)N*K*sizeof(float)); float *inv1=malloc((size_t)N*sizeof(float));
    for(int i=0;i<N;i++) inv1[i]=1.0f;
    int8_t *i8=malloc((size_t)N*K);
    double sm[REPS]; double GB=(double)N*K/1e9;  /* effective: int8 bytes written into the tile */
    for(int kind=0;kind<2;kind++){ const char*kn=kind?"NF4":"UNIFORM";
        /* OLD: inflate(f32) + tile_f32_i8 */
        for(int i=0;i<WARM;i++){ ork_slice_inflate_i4a8_kind(w,qf32,kind); ork_slice_tile_i8(c,w,qf32,inv1); }
        for(int i=0;i<REPS;i++){ double t=now_us(); ork_slice_inflate_i4a8_kind(w,qf32,kind); ork_slice_tile_i8(c,w,qf32,inv1); sm[i]=now_us()-t; }
        double oldm=median(sm,REPS);
        /* NEW: direct inflate->int8-tiled (full) + the inflate-only split */
        for(int i=0;i<WARM;i++) ork_slice_direct_i4a8_kind(c,w,i8,kind);
        for(int i=0;i<REPS;i++){ double t=now_us(); ork_slice_direct_i4a8_kind(c,w,i8,kind); sm[i]=now_us()-t; }
        double newm=median(sm,REPS);
        double inf[REPS];
        for(int i=0;i<WARM;i++) ork_slice_direct_inflate_i8(w,i8,kind);
        for(int i=0;i<REPS;i++){ double t=now_us(); ork_slice_direct_inflate_i8(w,i8,kind); inf[i]=now_us()-t; }
        double infm=median(inf,REPS);
        printf("  K%-5d N%-5d %-7s  old %8.1f us (%5.2f GB/s)   direct %8.1f us (%5.2f GB/s)   %.2fx   [direct inflate %.1f / tile+bsync %.1f]\n",
            K,N,kn, oldm, GB/(oldm*1e-6), newm, GB/(newm*1e-6), oldm/newm, infm, newm-infm);
    }
    free(qf32);free(inv1);free(i8); ork_mm_free(c,w);
}

int main(void){
    ork_npu *c = ork_npu_init();
    if(!c){ fprintf(stderr,"init FAIL (need /dev/dri + rknpu)\n"); return 1; }
    int rc=0;
    printf("=== CORRECTNESS (tiled bytes bit-identical + matmul exact) ===\n");
    int vs[][2]={{2048,256},{2048,512},{1024,128},{4096,512}};
    for(int i=0;i<(int)(sizeof(vs)/sizeof(vs[0]));i++){ rc|=verify(c,vs[i][0],vs[i][1],0); rc|=verify(c,vs[i][0],vs[i][1],1); }
    printf("\n=== BENCH (inflate+tile into resident DMA tiles; median of 25) ===\n");
    int bs[][2]={{2048,6144},{5120,13824},{13824,5120}};
    for(int i=0;i<(int)(sizeof(bs)/sizeof(bs[0]));i++) bench(c,bs[i][0],bs[i][1]);
    printf("\n%s\n", rc?"*** OVERALL: FAIL ***":"*** OVERALL: PASS ***");
    ork_npu_free(c);
    return rc;
}
