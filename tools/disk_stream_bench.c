/* tools/disk_stream_bench.c — DECISIVE micro-bench for w8a8-FROM-DISK streaming (roadmap P5.3 follow-on).
 *
 * Sibling of tools/prefetch_headroom.c. That tool answered "can a prefetch thread hide the int4
 * inflate+tile behind the NPU?". THIS tool answers the disk-source questions for the PRE-TILED int8
 * fast path (ork_w_dump produces tiled int8 bytes; ork_mm_load_i8 reloads them with NO inflate/retile —
 * the streaming fill is just: copy those bytes into the resident NPU dma_buf + bsync flush):
 *
 *   (A) RAM source          memcpy(RAM_buf -> dma_buf) + bsync.                      [baseline]
 *   (B) mmap WARM           file on NVMe, mmap'd, page-cache populated; memcpy(mmap -> dma_buf)+bsync.
 *   (C) mmap COLD           drop_caches, then SINGLE first-touch memcpy(mmap -> dma_buf)+bsync.
 *   (D) pread direct COLD   drop_caches, then pread(fd, dma_buf, size)+bsync  (read straight into dma).
 *
 * Plus, for the SAME shapes (so it's all one comparable table):
 *   t_npu   = NPU int8 matmul time (ork_mm_run_i8) at a few M.
 *   t_tile  = int4 inflate+tile, the RAM path (ork_slice_inflate_i4a8_kind + ork_slice_tile_i8) — the
 *             cost the pre-tiled int8 fill SKIPS. The headline lever question: (int4 inflate+tile) vs
 *             (pre-tiled int8 fill A) = how much does persisting the tiled bytes buy.
 *
 * Steady-state streaming model: the resident dma_buf is allocated ONCE and reused across reps (the fill
 * is the only per-slice cost — no per-slice bcreate). The disk file is written ONCE before timing.
 *
 *   make disk_stream_bench
 *   sudo systemctl stop orkllm
 *   sudo taskset -c 4-7 ./disk_stream_bench [/nvme/scratch/dir]   # default /dev/nvme0n1p1 mount
 *   sudo systemctl start orkllm
 *
 * Governors MUST be at max (the bench checks + warns): DDR dmc=performance(2112MHz), CPU cpu4..7=
 * performance(2.4GHz). A parked DDR governor invalidates the bandwidth numbers — the whole point.
 * Cold reps need `echo 3 > /proc/sys/vm/drop_caches` (needs root; the bench does it via system()).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "ork_npu.h"

/* internal diagnostic helpers exported from npu.c (not in the public header) */
void ork_slice_inflate_i4a8_kind(const ork_w *w, float *qf32, int kind);
void ork_slice_tile_i8(ork_npu *c, ork_w *w, const float *qf32, float *inv1);
void ork_dma_bsync_to_device(ork_npu *c, void *ptr, size_t size);

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y?1:0; }

#define WARM 5
#define REPS 24
#define COLD_REPS 4   /* cold single-shots are averaged over a few drop_caches cycles for stability */

typedef struct { double med, p10, p90; } statd;
static statd summarize(double *v, int n){
    qsort(v,n,sizeof(double),cmp_d);
    statd s; s.med=v[n/2]; s.p10=v[(int)(0.1*n)]; s.p90=v[(int)(0.9*n)]; return s;
}
static double gbps(size_t bytes, double us){ return us>0 ? (double)bytes/1024.0/1024.0/1024.0/(us/1e6) : 0; }

/* ---- governor verification (warn loudly; a parked DDR governor invalidates bandwidth numbers) ---- */
static int read_line(const char*path,char*buf,int cap){ FILE*f=fopen(path,"r"); if(!f) return -1;
    if(!fgets(buf,cap,f)){ fclose(f); return -1; } fclose(f); char*nl=strchr(buf,'\n'); if(nl)*nl=0; return 0; }
static long read_long(const char*path){ char b[64]; if(read_line(path,b,sizeof b)) return -1; return atol(b); }

static int g_gov_ok = 1;
static void check_governors(void){
    char gov[64]; int warned=0;
    printf("=== governor state at bench time ===\n");
    if(!read_line("/sys/class/devfreq/dmc/governor",gov,sizeof gov)){
        long f=read_long("/sys/class/devfreq/dmc/cur_freq");
        printf("  DDR dmc governor = %-12s cur_freq = %ld Hz (%.0f MHz)\n", gov, f, f>0?f/1e6:0);
        if(strcmp(gov,"performance")){ printf("  *** WARNING: DDR governor is NOT 'performance' — ALL bandwidth numbers below are SUSPECT. Set: echo performance | sudo tee /sys/class/devfreq/dmc/governor\n"); warned=1; }
    } else printf("  DDR dmc governor: (could not read /sys/class/devfreq/dmc/governor)\n");
    for(int cpu=4;cpu<=7;cpu++){
        char p[128]; snprintf(p,sizeof p,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor",cpu);
        if(!read_line(p,gov,sizeof gov)){
            snprintf(p,sizeof p,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq",cpu);
            long f=read_long(p);
            printf("  cpu%d governor = %-12s cur_freq = %ld kHz (%.2f GHz)\n", cpu, gov, f, f>0?f/1e6:0);
            if(strcmp(gov,"performance")){ printf("  *** WARNING: cpu%d governor is NOT 'performance' — A76 fill timing is SUSPECT.\n",cpu); warned=1; }
        }
    }
    if(!warned) printf("  OK: DDR + A76 big cores all at 'performance'.\n");
    else g_gov_ok = 0;
    printf("\n");
}

static int drop_caches(void){
    /* sync first so dirty pages are written back, then evict page cache so the next read is truly cold */
    int rc = system("sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1");
    return rc;
}

/* madvise(DONTNEED) the mmap so its pages drop too (belt-and-braces with drop_caches for the file) */
static void evict_mmap(void *p, size_t n){ if(p && p!=MAP_FAILED) madvise(p,n,MADV_DONTNEED); }

static const char *g_scratch = "/dev/nvme0n1p1";   /* default; arg overrides */

/* one shape: pack int8 (resident) -> dump tiled bytes (the pre-tiled blob) -> bench A/B/C/D fills into a
 * reused resident dma_buf; also pack int4 for t_tile (inflate+tile RAM path) and run t_npu. */
static void run_shape(ork_npu*c, int K, int N, const int*Ms, int nM, const char*label){
    /* synthesize Gaussian-ish weights f32[N][K] (n-major) for a realistic pack */
    float *f32 = malloc((size_t)N*K*sizeof(float));
    if(!f32){ printf("  [%s] OOM weights\n",label); return; }
    unsigned s=0x1234u+(unsigned)(K*131+N);
    for(size_t i=0;i<(size_t)N*K;i++){ s=s*1664525u+1013904223u; float u=(s>>8)*(1.0f/16777216.0f); f32[i]=(u-0.5f)*0.2f; }

    /* derive int8 B[K,N] (row-major) from f32 (per-channel symmetric, like the real quant) for pack_i8 */
    int8_t *Bi8 = malloc((size_t)K*N);
    if(!Bi8){ printf("  [%s] OOM Bi8\n",label); free(f32); return; }
    for(int n=0;n<N;n++){
        float mx=0; for(int k=0;k<K;k++){ float a=fabsf(f32[(size_t)n*K+k]); if(a>mx)mx=a; }
        float sc = mx>0 ? 127.0f/mx : 0;
        for(int k=0;k<K;k++){ int v=(int)lrintf(f32[(size_t)n*K+k]*sc);
            if(v>127)v=127; if(v<-127)v=-127; Bi8[(size_t)k*N+n]=(int8_t)v; } }

    ork_w *w8 = ork_mm_pack_i8(c,K,N,Bi8);   /* pre-tiled int8 resident weight */
    if(!w8){ printf("  [%s] pack_i8 failed (K=%d N=%d)\n",label,K,N); free(f32); free(Bi8); return; }

    /* the pre-tiled int8 blob (what would live on disk in an .orkpack) */
    size_t blob_sz = ork_w_dump(w8, NULL, 0);
    void *ram = malloc(blob_sz);
    if(!ram || ork_w_dump(w8, ram, blob_sz)!=blob_sz){ printf("  [%s] dump failed\n",label); goto done8; }

    /* resident NPU dma_buf, allocated ONCE, reused across all reps (steady-state streaming fill target) */
    void *dma = ork_dma_alloc(c, blob_sz);
    if(!dma){ printf("  [%s] ork_dma_alloc(%zu) failed\n",label,blob_sz); goto done8; }

    /* write the blob to an NVMe file ONCE (before timing) */
    char path[512]; snprintf(path,sizeof path,"%s/ork_stream_%dx%d.bin",g_scratch,K,N);
    { int fd=open(path,O_WRONLY|O_CREAT|O_TRUNC,0644);
      if(fd<0){ printf("  [%s] cannot create %s (need a writable NVMe mount; pass dir as arg)\n",label,path); goto done_dma; }
      size_t off=0; while(off<blob_sz){ ssize_t w=write(fd,(char*)ram+off,blob_sz-off); if(w<=0){ printf("  [%s] write failed\n",label); close(fd); goto done_dma; } off+=w; }
      fsync(fd); close(fd); }

    double samp[REPS];

    /* ---------- (A) RAM source: memcpy(ram -> dma) + bsync ---------- */
    for(int i=0;i<WARM;i++){ memcpy(dma,ram,blob_sz); ork_dma_bsync_to_device(c,dma,blob_sz); }
    for(int i=0;i<REPS;i++){ double t=now_us(); memcpy(dma,ram,blob_sz); ork_dma_bsync_to_device(c,dma,blob_sz); samp[i]=now_us()-t; }
    statd A = summarize(samp,REPS);

    /* attribution: memcpy-only (no flush) and bsync-only (flush a quiesced buffer), so the reader can see
     * how much of the "fill" is the cache-flush-to-device vs the host copy itself. */
    for(int i=0;i<WARM;i++) memcpy(dma,ram,blob_sz);
    for(int i=0;i<REPS;i++){ double t=now_us(); memcpy(dma,ram,blob_sz); samp[i]=now_us()-t; }
    statd A_cp = summarize(samp,REPS);
    ork_dma_bsync_to_device(c,dma,blob_sz);   /* quiesce */
    for(int i=0;i<WARM;i++) ork_dma_bsync_to_device(c,dma,blob_sz);
    for(int i=0;i<REPS;i++){ double t=now_us(); ork_dma_bsync_to_device(c,dma,blob_sz); samp[i]=now_us()-t; }
    statd A_bs = summarize(samp,REPS);

    /* ---------- (B) mmap WARM: file mmap'd, page-cache populated, memcpy(mmap -> dma) + bsync ---------- */
    statd B = {0,0,0};
    { int fd=open(path,O_RDONLY); if(fd<0){ printf("  [%s] reopen for mmap failed\n",label); goto done_dma; }
      void *mp=mmap(NULL,blob_sz,PROT_READ,MAP_PRIVATE,fd,0);
      if(mp==MAP_FAILED){ printf("  [%s] mmap failed\n",label); close(fd); goto done_dma; }
      /* touch once to populate the page cache (warm) */
      volatile unsigned long acc=0; for(size_t o=0;o<blob_sz;o+=4096) acc+=((unsigned char*)mp)[o]; (void)acc;
      for(int i=0;i<WARM;i++){ memcpy(dma,mp,blob_sz); ork_dma_bsync_to_device(c,dma,blob_sz); }
      for(int i=0;i<REPS;i++){ double t=now_us(); memcpy(dma,mp,blob_sz); ork_dma_bsync_to_device(c,dma,blob_sz); samp[i]=now_us()-t; }
      B = summarize(samp,REPS);
      munmap(mp,blob_sz); close(fd); }

    /* ---------- (C) mmap COLD: drop_caches, SINGLE first-touch memcpy(mmap -> dma) + bsync ---------- */
    double cold_c[COLD_REPS];
    for(int r=0;r<COLD_REPS;r++){
        drop_caches();
        int fd=open(path,O_RDONLY); if(fd<0){ cold_c[r]=0; continue; }
        posix_fadvise(fd,0,blob_sz,POSIX_FADV_DONTNEED);
        void *mp=mmap(NULL,blob_sz,PROT_READ,MAP_PRIVATE,fd,0);
        if(mp==MAP_FAILED){ close(fd); cold_c[r]=0; continue; }
        evict_mmap(mp,blob_sz);
        double t=now_us(); memcpy(dma,mp,blob_sz); ork_dma_bsync_to_device(c,dma,blob_sz); cold_c[r]=now_us()-t;
        munmap(mp,blob_sz); close(fd);
    }
    qsort(cold_c,COLD_REPS,sizeof(double),cmp_d); double C_cold=cold_c[COLD_REPS/2];

    /* ---------- (D) pread direct COLD: drop_caches, pread(fd -> dma) + bsync ---------- */
    double cold_d[COLD_REPS];
    for(int r=0;r<COLD_REPS;r++){
        drop_caches();
        int fd=open(path,O_RDONLY); if(fd<0){ cold_d[r]=0; continue; }
        posix_fadvise(fd,0,blob_sz,POSIX_FADV_DONTNEED);
        double t=now_us();
        size_t off=0; ssize_t rc=1;
        while(off<blob_sz && rc>0){ rc=pread(fd,(char*)dma+off,blob_sz-off,off); if(rc>0) off+=rc; }
        ork_dma_bsync_to_device(c,dma,blob_sz);
        cold_d[r] = (off==blob_sz) ? now_us()-t : 0;
        close(fd);
    }
    qsort(cold_d,COLD_REPS,sizeof(double),cmp_d); double D_cold=cold_d[COLD_REPS/2];

    /* ---------- int4 inflate+tile (RAM path) on the SAME shape (the cost pre-tiling skips) ---------- */
    statd T = {0,0,0}; int have_t = 0;
    { float *bscale = malloc((size_t)N*sizeof(float));
      ork_w *w4 = ork_mm_pack_i4a8(c,K,N,f32,bscale);
      if(w4){
        float *qf32 = malloc((size_t)N*K*sizeof(float));
        float *inv1 = malloc((size_t)N*sizeof(float));
        if(qf32 && inv1){
          for(int n=0;n<N;n++) inv1[n]=1.0f;
          ork_slice_inflate_i4a8_kind(w4,qf32,ORK_QK_UNIFORM);   /* prime codes */
          /* t_tile here is inflate(UNIFORM)+tile combined — the full int4->resident prep */
          for(int i=0;i<WARM;i++){ ork_slice_inflate_i4a8_kind(w4,qf32,ORK_QK_UNIFORM); ork_slice_tile_i8(c,w4,qf32,inv1); }
          for(int i=0;i<REPS;i++){ double t=now_us(); ork_slice_inflate_i4a8_kind(w4,qf32,ORK_QK_UNIFORM); ork_slice_tile_i8(c,w4,qf32,inv1); samp[i]=now_us()-t; }
          T = summarize(samp,REPS); have_t=1;
        }
        free(qf32); free(inv1);
        ork_mm_free(c,w4);
      }
      free(bscale);
    }

    /* ---------- report ---------- */
    printf("  [%s] K=%d N=%d   bytes/slice = %.2f MiB (int8 tiled)   int4 store = %.2f MiB\n",
           label, K, N, (double)blob_sz/1048576.0, (double)((size_t)K*N/2)/1048576.0);
    printf("    fill path           |    med us | [p10/p90 us] |  GB/s | factor vs RAM\n");
    printf("    (A) RAM memcpy+bsync | %9.0f | [%5.0f/%5.0f]| %5.2f |  1.00x   (memcpy-only %.0f us / bsync-only %.0f us)\n",
           A.med, A.p10, A.p90, gbps(blob_sz,A.med), A_cp.med, A_bs.med);
    printf("    (B) mmap WARM        | %9.0f | [%5.0f/%5.0f]| %5.2f |  %.2fx\n", B.med, B.p10, B.p90, gbps(blob_sz,B.med), A.med>0?B.med/A.med:0);
    printf("    (C) mmap COLD (1shot)| %9.0f |   (median %d) | %5.2f |  %.2fx\n", C_cold, COLD_REPS, gbps(blob_sz,C_cold), A.med>0?C_cold/A.med:0);
    printf("    (D) pread COLD(1shot)| %9.0f |   (median %d) | %5.2f |  %.2fx\n", D_cold, COLD_REPS, gbps(blob_sz,D_cold), A.med>0?D_cold/A.med:0);
    if(have_t) printf("    int4 inflate+tile RAM| %9.0f | [%5.0f/%5.0f]| %5.2f |  %.2fx  <- the cost pre-tiling SKIPS\n",
                      T.med, T.p10, T.p90, gbps(blob_sz,T.med), A.med>0?T.med/A.med:0);
    else       printf("    int4 inflate+tile RAM| (pack_i4a8 unavailable)\n");

    /* t_npu per M + the "hidden behind compute" verdict for the pre-tiled int8 fill (A) */
    printf("    %-4s | %9s | %12s | %s\n","M","t_npu(us)","t_fill_A(us)","pre-tiled fill vs NPU compute");
    for(int mi=0;mi<nM;mi++){
        int M = Ms[mi];
        int8_t  *Aa = malloc((size_t)M*K);
        int32_t *Cc = malloc((size_t)M*N*4);
        if(!Aa||!Cc){ printf("    M=%d OOM act\n",M); free(Aa); free(Cc); continue; }
        memset(Aa,1,(size_t)M*K);
        if(ork_mm_run_i8(c,w8,M,Aa,Cc)!=0){ printf("    M=%d ork_mm_run_i8 FAILED\n",M); free(Aa); free(Cc); continue; }
        for(int i=0;i<WARM;i++) ork_mm_run_i8(c,w8,M,Aa,Cc);
        for(int i=0;i<REPS;i++){ double t=now_us(); ork_mm_run_i8(c,w8,M,Aa,Cc); samp[i]=now_us()-t; }
        statd npu = summarize(samp,REPS);
        double ratio = npu.med>0 ? A.med/npu.med : 0;
        char verdict[160];
        if(A.med <= npu.med) snprintf(verdict,sizeof verdict,"FULLY HIDDEN (fill %.0f%% of compute)", ratio*100.0);
        else                 snprintf(verdict,sizeof verdict,"residual %.0f us/slice (fill %.2fx compute)", A.med-npu.med, ratio);
        printf("    %-4d | %9.0f | %12.0f | %s\n", M, npu.med, A.med, verdict);
        free(Aa); free(Cc);
    }
    unlink(path);
    printf("\n");

done_dma:
    ork_dma_free(c, dma);
done8:
    free(ram);
    ork_mm_free(c, w8);
    free(f32); free(Bi8);
}

int main(int argc, char**argv){
    if(argc>1) g_scratch = argv[1];
    check_governors();

    ork_npu *c = ork_npu_init();
    if(!c){ printf("ork_npu_init failed (need /dev/dri + rknpu; run as root / stop orkllm)\n"); return 1; }
    printf("SoC=%s cores=%d validated=%d  ork-driver %s\n", ork_npu_soc(c), ork_npu_cores(c), ork_npu_validated(c), ork_npu_version());
    printf("Disk-stream fill bench: pre-tiled int8 fill into a REUSED resident dma_buf, from RAM (A) /\n");
    printf("mmap-warm (B) / mmap-cold (C) / pread-cold (D). int4 inflate+tile = the prep pre-tiling skips.\n");
    printf("median of %d warm reps (%d warmup), cold = median of %d drop_caches single-shots. Scratch dir = %s\n\n",
           REPS, WARM, COLD_REPS, g_scratch);
    if(!g_gov_ok) printf("*** governors not all at performance — see warnings above; bandwidth numbers SUSPECT ***\n\n");

    int Ms[] = {1, 32, 128};
    int nM = 3;

    /* 1.7B FFN up */
    run_shape(c, 2048, 6144, Ms, nM, "1.7B FFN up   ");
    /* 14B FFN (the streaming regime) */
    run_shape(c, 5120, 13824, Ms, nM, "14B FFN up    ");
    run_shape(c, 13824, 5120, Ms, nM, "14B FFN down  ");

    ork_npu_free(c);
    printf("done.\n");
    return 0;
}
