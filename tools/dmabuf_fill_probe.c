/* tools/dmabuf_fill_probe.c — can we break the ~1.06 GB/s NPU weight dma_buf FILL wall?
 *
 * THE QUESTION: the streaming weight fill (host memcpy of pre-tiled int8 bytes into the resident NPU
 * dma_buf, then bsync) is bandwidth-bound. tools/disk_stream_bench.c fills an ork_dma_alloc() buffer,
 * which is allocated NON-CACHEABLE (rknpu flag 0x401) — the CPU write can't use the cache, so the fill
 * tops out around ~1 GB/s. The production weight path (pack_i8 / tile_f32_i8 / load_i8) already allocates
 * its Bb tiles CACHEABLE (0x403). This probe A/Bs the fill bandwidth AND the NPU-read correctness of the
 * relevant rknpu mem flags, so we can say whether (and by how much) a cacheable weight buffer wins:
 *
 *   (A) WC baseline   : Bb tiles allocated 0x401 (non-cacheable, what ork_dma_alloc uses) + plain memcpy
 *                       + full bsync (TO|FROM then TO, the ork_dma_bsync_to_device pattern).  ~1.06 GB/s?
 *   (B) cacheable     : Bb tiles allocated 0x403 (CACHEABLE) + plain memcpy (at cache speed) + a
 *                       CLEAN-only bsync (TO_DEVICE — flush dirty lines to DRAM) so the NPU reads correct
 *                       data. memcpy time and clean time reported SEPARATELY and COMBINED. (Does
 *                       cached_write + flush beat the WC memcpy? The flush of N MiB dirty lines is a real
 *                       cost — measured, not assumed.)
 *   (C) WC + NEON NT  : Bb tiles 0x401 + NEON non-temporal streaming stores (stnp, 64 B/iter, unrolled)
 *                       instead of memcpy + full bsync — does a better store pattern lift WC bandwidth?
 *
 * Each variant's NPU matmul output is checked against a CPU int8 reference (C[m,n]=sum_k A[m,k]*B[k,n]).
 * A faster fill that corrupts the NPU read is a FAIL — correctness is reported per variant.
 *
 * The resident weight is allocated ONCE per variant and the fill loop reuses it (steady-state streaming).
 *
 *   make dmabuf_fill_probe
 *   sudo systemctl stop orkllm
 *   sudo taskset -c 4-7 ./dmabuf_fill_probe
 *   sudo systemctl start orkllm
 *
 * Governors MUST be at max (the probe checks + warns): DDR dmc=performance(2112MHz), CPU cpu4..7=
 * performance(2.4GHz). A parked DDR governor invalidates all bandwidth numbers.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include "ork_npu.h"

/* diagnostic helpers exported from npu.c (not in the public header) */
ork_w *ork_mm_load_i8_flags(ork_npu *c,int K,int N,const void *blob,size_t n,unsigned flags);
int    ork_w_ntiles(const ork_w *w);
void  *ork_w_tile_cpu(const ork_w *w,int i);
size_t ork_w_tile_size(const ork_w *w,int i);
void   ork_w_tile_clean(ork_npu *c,const ork_w *w,int i);
void   ork_w_tile_bsync_full(ork_npu *c,const ork_w *w,int i);

#define FLAG_WC        0x401u   /* NON_CONTIGUOUS | IOMMU_LIMIT_IOVA_ALIGN  (NON-CACHEABLE) */
#define FLAG_CACHEABLE 0x403u   /* NON_CONTIGUOUS | CACHEABLE | IOMMU_LIMIT_IOVA_ALIGN */

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y?1:0; }

#define WARM 5
#define REPS 25

typedef struct { double med, p10, p90; } statd;
static statd summarize(double *v, int n){
    qsort(v,n,sizeof(double),cmp_d);
    statd s; s.med=v[n/2]; s.p10=v[(int)(0.1*n)]; s.p90=v[(int)(0.9*n)]; return s;
}
static double gbps(size_t bytes, double us){ return us>0 ? (double)bytes/1e9/(us/1e6) : 0; } /* GB/s, 1e9 base */

/* ---- governor verification ---- */
static int read_line(const char*path,char*buf,int cap){ FILE*f=fopen(path,"r"); if(!f) return -1;
    if(!fgets(buf,cap,f)){ fclose(f); return -1; } fclose(f); char*nl=strchr(buf,'\n'); if(nl)*nl=0; return 0; }
static long read_long(const char*path){ char b[64]; if(read_line(path,b,sizeof b)) return -1; return atol(b); }
static int g_gov_ok = 1;
static void check_governors(void){
    char gov[64]; int warned=0;
    printf("=== governor state at probe time ===\n");
    if(!read_line("/sys/class/devfreq/dmc/governor",gov,sizeof gov)){
        long f=read_long("/sys/class/devfreq/dmc/cur_freq");
        printf("  DDR dmc governor = %-12s cur_freq = %.0f MHz\n", gov, f>0?f/1e6:0);
        if(strcmp(gov,"performance")){ printf("  *** WARNING: DDR governor NOT performance — bandwidth numbers SUSPECT.\n"); warned=1; }
    }
    for(int cpu=4;cpu<=7;cpu++){
        char p[128]; snprintf(p,sizeof p,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor",cpu);
        if(!read_line(p,gov,sizeof gov)){
            if(strcmp(gov,"performance")){ printf("  *** WARNING: cpu%d governor NOT performance.\n",cpu); warned=1; }
        }
    }
    if(!warned) printf("  OK: DDR + A76 big cores all at performance.\n");
    else g_gov_ok = 0;
    printf("\n");
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
/* NEON non-temporal streaming copy: stnp bypasses the cache on the store side (the right pattern for a
 * write-combine destination). 64 B/iter, unrolled x2 (128 B). dst/src 16-aligned (page-aligned tiles). */
static void neon_stream_copy(void *dst, const void *src, size_t n){
    uint8_t *d = (uint8_t*)dst; const uint8_t *s = (const uint8_t*)src;
    size_t i = 0;
    for(; i + 128 <= n; i += 128){
        uint8x16_t a0=vld1q_u8(s+i),     a1=vld1q_u8(s+i+16),  a2=vld1q_u8(s+i+32),  a3=vld1q_u8(s+i+48);
        uint8x16_t b0=vld1q_u8(s+i+64),  b1=vld1q_u8(s+i+80),  b2=vld1q_u8(s+i+96),  b3=vld1q_u8(s+i+112);
        __asm__ volatile("stnp %q0,%q1,[%4]\n stnp %q2,%q3,[%4,#32]\n"
                         :: "w"(a0),"w"(a1),"w"(a2),"w"(a3),"r"(d+i) : "memory");
        __asm__ volatile("stnp %q0,%q1,[%4]\n stnp %q2,%q3,[%4,#32]\n"
                         :: "w"(b0),"w"(b1),"w"(b2),"w"(b3),"r"(d+i+64) : "memory");
    }
    for(; i < n; i++) d[i] = s[i];
}
#else
static void neon_stream_copy(void *dst, const void *src, size_t n){ memcpy(dst,src,n); }
#endif

/* fill all Bb tiles of `w` from `blob` (laid out in tile order by ork_w_dump), copy via `use_neon`,
 * bsync via `clean_only` (cacheable clean) or full (WC). Returns total bytes filled. */
static size_t fill_tiles(ork_npu*c, ork_w*w, const uint8_t*blob, int use_neon, int clean_only){
    int nt = ork_w_ntiles(w); size_t off=0;
    for(int i=0;i<nt;i++){ void*p=ork_w_tile_cpu(w,i); size_t sz=ork_w_tile_size(w,i); if(!p) continue;
        if(use_neon) neon_stream_copy(p, blob+off, sz); else memcpy(p, blob+off, sz);
        if(clean_only) ork_w_tile_clean(c,w,i); else ork_w_tile_bsync_full(c,w,i);
        off+=sz; }
    return off;
}
static size_t fill_copy_only(ork_w*w, const uint8_t*blob, int use_neon){
    int nt = ork_w_ntiles(w); size_t off=0;
    for(int i=0;i<nt;i++){ void*p=ork_w_tile_cpu(w,i); size_t sz=ork_w_tile_size(w,i); if(!p) continue;
        if(use_neon) neon_stream_copy(p, blob+off, sz); else memcpy(p, blob+off, sz); off+=sz; }
    return off;
}
static void bsync_all(ork_npu*c, ork_w*w, int clean_only){
    int nt = ork_w_ntiles(w);
    for(int i=0;i<nt;i++){ if(!ork_w_tile_cpu(w,i)) continue;
        if(clean_only) ork_w_tile_clean(c,w,i); else ork_w_tile_bsync_full(c,w,i); }
}

/* CPU int8 reference for a single output (m,n): sum_k A[m,k]*B[k,n], B row-major [K][N]. */
static int correctness_ok(ork_npu*c, ork_w*w, const int8_t*Bi8, int K, int N, int M){
    int8_t  *A = malloc((size_t)M*K);
    int32_t *Cn = malloc((size_t)M*N*4);
    if(!A||!Cn){ free(A); free(Cn); return -1; }
    unsigned s=0xC0FFEEu;
    for(size_t i=0;i<(size_t)M*K;i++){ s=s*1664525u+1013904223u; A[i]=(int8_t)((int)(s>>24)%9 - 4); } /* small int8 */
    if(ork_mm_run_i8(c,w,M,A,Cn)!=0){ free(A); free(Cn); return -2; }
    /* spot-check a sample of (m,n) against the CPU reference (full K accumulation) */
    int bad=0, checked=0;
    unsigned r=0x1234u;
    for(int t=0;t<64;t++){
        r=r*1103515245u+12345u; int m=(int)((r>>16)%M);
        r=r*1103515245u+12345u; int n=(int)((r>>16)%N);
        long acc=0; for(int k=0;k<K;k++) acc += (long)A[(size_t)m*K+k]*(long)Bi8[(size_t)k*N+n];
        int32_t got = Cn[(size_t)m*N+n]; checked++;
        if((long)got != acc){ if(bad<3) fprintf(stderr,"    mismatch (m=%d n=%d): got=%d want=%ld\n",m,n,got,acc); bad++; }
    }
    free(A); free(Cn);
    return bad==0 ? 0 : 1;
}

static void run_variant(ork_npu*c, int K, int N, const uint8_t*blob, size_t blob_sz,
                        const int8_t*Bi8, unsigned flag, int use_neon, int clean_only,
                        const char*name, double baseline_med){
    ork_w *w = ork_mm_load_i8_flags(c,K,N,blob,blob_sz,flag);
    if(!w){ printf("    %-22s | load_i8_flags(0x%x) FAILED\n", name, flag); return; }
    size_t bytes = 0; for(int i=0;i<ork_w_ntiles(w);i++) bytes += ork_w_tile_size(w,i);

    double samp[REPS], scp[REPS], sbs[REPS];
    /* combined fill (copy + bsync) */
    for(int i=0;i<WARM;i++) fill_tiles(c,w,blob,use_neon,clean_only);
    for(int i=0;i<REPS;i++){ double t=now_us(); fill_tiles(c,w,blob,use_neon,clean_only); samp[i]=now_us()-t; }
    statd F = summarize(samp,REPS);
    /* attribution: copy-only and bsync-only (flush a quiesced buffer) */
    for(int i=0;i<WARM;i++) fill_copy_only(w,blob,use_neon);
    for(int i=0;i<REPS;i++){ double t=now_us(); fill_copy_only(w,blob,use_neon); scp[i]=now_us()-t; }
    statd CP = summarize(scp,REPS);
    bsync_all(c,w,clean_only);
    for(int i=0;i<WARM;i++) bsync_all(c,w,clean_only);
    for(int i=0;i<REPS;i++){ double t=now_us(); bsync_all(c,w,clean_only); sbs[i]=now_us()-t; }
    statd BS = summarize(sbs,REPS);

    /* re-fill with valid bytes, then correctness-check the NPU read at M=1 and M=32 */
    fill_tiles(c,w,blob,use_neon,clean_only);
    int ok1 = correctness_ok(c,w,Bi8,K,N,1);
    int ok32= correctness_ok(c,w,Bi8,K,N,32);
    const char* verdict = (ok1==0 && ok32==0) ? "PASS" :
                          (ok1<0||ok32<0) ? "ERR(setup)" : "FAIL";

    double speed = baseline_med>0 ? baseline_med/F.med : 0;
    printf("    %-22s | %8.0f | [%5.0f/%5.0f] | %6.2f | copy %8.0f (%5.2f GB/s) | bsync %7.0f | %5.2fx | %s\n",
           name, F.med, F.p10, F.p90, gbps(bytes,F.med), CP.med, gbps(bytes,CP.med), BS.med, speed, verdict);
    ork_mm_free(c,w);
}

static void run_shape(ork_npu*c, int K, int N, const char*label){
    /* random f32 weights -> per-channel symmetric int8 B[K,N] row-major (like the real quant) */
    int8_t *Bi8 = malloc((size_t)K*N);
    if(!Bi8){ printf("  [%s] OOM\n",label); return; }
    unsigned s=0x1234u+(unsigned)(K*131+N);
    for(int n=0;n<N;n++){
        for(int k=0;k<K;k++){ s=s*1664525u+1013904223u; int v=(int)((s>>24)%255)-127; Bi8[(size_t)k*N+n]=(int8_t)v; } }

    ork_w *w8 = ork_mm_pack_i8(c,K,N,Bi8);
    if(!w8){ printf("  [%s] pack_i8 failed\n",label); free(Bi8); return; }
    size_t blob_sz = ork_w_dump(w8, NULL, 0);
    uint8_t *blob = malloc(blob_sz);
    if(!blob || ork_w_dump(w8, blob, blob_sz)!=blob_sz){ printf("  [%s] dump failed\n",label); ork_mm_free(c,w8); free(Bi8); return; }
    ork_mm_free(c,w8);

    printf("  [%s] K=%d N=%d   tiled int8 = %.2f MiB\n", label, K, N, (double)blob_sz/1048576.0);
    printf("    %-22s | fill us  | [p10/p90]    |  GB/s  | copy-only us (GB/s)   | bsync us | speedup | correct\n", "variant");

    /* (A) WC baseline establishes the reference fill time the speedups are relative to */
    ork_w *wA = ork_mm_load_i8_flags(c,K,N,blob,blob_sz,FLAG_WC);
    double baseA = 0;
    if(wA){
        double samp[REPS];
        for(int i=0;i<WARM;i++) fill_tiles(c,wA,blob,0,0);
        for(int i=0;i<REPS;i++){ double t=now_us(); fill_tiles(c,wA,blob,0,0); samp[i]=now_us()-t; }
        baseA = summarize(samp,REPS).med;
        ork_mm_free(c,wA);
    }
    run_variant(c,K,N,blob,blob_sz,Bi8, FLAG_WC,        0, 0, "(A) WC 0x401 memcpy",   baseA);
    run_variant(c,K,N,blob,blob_sz,Bi8, FLAG_CACHEABLE, 0, 1, "(B) cacheable 0x403",   baseA);
    run_variant(c,K,N,blob,blob_sz,Bi8, FLAG_WC,        1, 0, "(C) WC 0x401 NEON-stnp", baseA);
    printf("\n");
    free(blob); free(Bi8);
}

int main(void){
    check_governors();
    ork_npu *c = ork_npu_init();
    if(!c){ printf("ork_npu_init failed (need /dev/dri + rknpu; run as root / stop orkllm)\n"); return 1; }
    printf("SoC=%s cores=%d validated=%d  ork-driver %s\n", ork_npu_soc(c), ork_npu_cores(c), ork_npu_validated(c), ork_npu_version());
    printf("dma_buf weight FILL wall probe: (A) WC 0x401 memcpy+full-bsync = the ~1.06 GB/s baseline;\n");
    printf("(B) cacheable 0x403 memcpy+clean-only-bsync (clean cost broken out as bsync us);\n");
    printf("(C) WC 0x401 NEON stnp streaming stores. Each verified vs a CPU int8 matmul reference.\n");
    printf("median of %d reps (%d warm). GB/s = bytes/1e9 / s.\n\n", REPS, WARM);
    if(!g_gov_ok) printf("*** governors not all performance — numbers SUSPECT ***\n\n");

    run_shape(c, 2048, 6144, "1.7B FFN up ");
    run_shape(c, 5120, 13824, "14B FFN up  ");

    ork_npu_free(c);
    printf("done.\n");
    return 0;
}
