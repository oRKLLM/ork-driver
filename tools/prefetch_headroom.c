/* tools/prefetch_headroom.c — DECISIVE micro-bench for the streaming-prefetch design (roadmap P5.3).
 *
 * THE QUESTION: when a weight is cycled through the 4 GiB NPU window as int4, expanded to int8, and run,
 * can a CPU prefetch thread that prepares slice i+1 hide its cost behind the NPU computing slice i?
 *
 *   t_npu  = NPU int8 matmul time for the slice (ork_i8_mm_run).
 *   t_prep = CPU per-slice STEADY-STATE prep = NEON int4->int8 inflate + tile into an ALREADY-ALLOCATED,
 *            reused NPU DMA buffer. NO bcreate/alloc per slice (that's a one-time cost). Split into:
 *              t_inflate = NEON int4->int8 expand (UNIFORM and NF4 measured separately)
 *              t_tile    = tile_f32_i8 copy/quant into the resident DMA buffer + bsync(TO_DEVICE)
 *            t_prep = t_inflate + t_tile.
 *
 * Double-buffer prefetch makes per-slice wall-clock = max(t_npu, t_prep) instead of t_npu + t_prep.
 * Headroom = t_npu / t_prep: ratio = t_prep / t_npu. ratio <= 1 => prefetch FULLY hides prep (streaming
 * ~= resident speed). ratio > 1 => prep-bound; residual = (t_prep - t_npu) per slice. The inflate/tile
 * split says whether a CPU prefetch thread (overlaps inflate, a CPU op) suffices, or the wall is the
 * tile/copy transfer (bandwidth) which a thread can't speed up.
 *
 * To isolate STEADY-STATE prep with NO per-slice alloc: pack the int4 weight ONCE (ork_i4a8_mm_pack ->
 * resident DMA buffers + the compact nibble store), then in the timed loop re-run ONLY the inflate + tile
 * tail (internal ork_i4a8_slice_inflate_kind / ork_i8_slice_tile — they reuse the production tile path and
 * the already-allocated buffers; they do not alloc or change pack/run behavior).
 *
 *   make prefetch_headroom
 *   sudo systemctl stop orkllm          # NPU must be exclusive for a raw micro-bench
 *   sudo taskset -c 4-7 ./prefetch_headroom   # pin CPU-side prep to the A76 big cores
 *   sudo systemctl start orkllm
 *
 * Governors MUST be at max (the bench checks + warns): DDR dmc=performance(2112MHz), CPU cpu4..7=
 * performance(2.4GHz). A parked DDR governor invalidates t_tile/t_prep (bandwidth-sensitive).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "ork_npu.h"

/* internal diagnostic helpers exported from npu.c (not in the public header) */
void ork_i4a8_slice_inflate_kind(const ork_w *w, float *qf32, int kind);
void ork_i8_slice_tile(ork_npu *c, ork_w *w, const float *qf32, float *inv1);

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y?1:0; }

#define WARM 5
#define REPS 24

/* median + p10/p90 spread over n samples (sorted in place) */
typedef struct { double med, p10, p90; } stat;
static stat summarize(double *v, int n){
    qsort(v,n,sizeof(double),cmp_d);
    stat s; s.med=v[n/2]; s.p10=v[(int)(0.1*n)]; s.p90=v[(int)(0.9*n)]; return s;
}

/* ---- governor verification (warn loudly; a parked DDR governor invalidates bandwidth numbers) ---- */
static int read_line(const char*path,char*buf,int cap){ FILE*f=fopen(path,"r"); if(!f) return -1;
    if(!fgets(buf,cap,f)){ fclose(f); return -1; } fclose(f); char*nl=strchr(buf,'\n'); if(nl)*nl=0; return 0; }
static long read_long(const char*path){ char b[64]; if(read_line(path,b,sizeof b)) return -1; return atol(b); }

static void check_governors(void){
    char gov[64]; int warned=0;
    printf("=== governor state at bench time ===\n");
    if(!read_line("/sys/class/devfreq/dmc/governor",gov,sizeof gov)){
        long f=read_long("/sys/class/devfreq/dmc/cur_freq");
        printf("  DDR dmc governor = %-12s cur_freq = %ld Hz (%.0f MHz)\n", gov, f, f>0?f/1e6:0);
        if(strcmp(gov,"performance")){ printf("  *** WARNING: DDR governor is NOT 'performance' — bandwidth-sensitive numbers (t_tile/t_prep) are SUSPECT. Set: echo performance | sudo tee /sys/class/devfreq/dmc/governor\n"); warned=1; }
    } else printf("  DDR dmc governor: (could not read /sys/class/devfreq/dmc/governor)\n");
    for(int cpu=4;cpu<=7;cpu++){
        char p[128]; snprintf(p,sizeof p,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor",cpu);
        if(!read_line(p,gov,sizeof gov)){
            snprintf(p,sizeof p,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq",cpu);
            long f=read_long(p);
            printf("  cpu%d governor = %-12s cur_freq = %ld kHz (%.2f GHz)\n", cpu, gov, f, f>0?f/1e6:0);
            if(strcmp(gov,"performance")){ printf("  *** WARNING: cpu%d governor is NOT 'performance' — A76 prep timing is SUSPECT.\n",cpu); warned=1; }
        }
    }
    if(!warned) printf("  OK: DDR + A76 big cores all at 'performance'.\n");
    printf("\n");
}

/* one shape: pack int4 once (resident buffers + nibble store), then time t_npu (per M), inflate (uniform
 * + NF4 forced on the same nibbles), and tile into the reused resident buffers. */
static void run_shape(ork_npu*c, int K, int N, const int*Ms, int nM, const char*label){
    /* synthesize Gaussian-ish weights f32[N][K] (n-major, as ggml to_float produces) for a realistic pack */
    float *f32 = malloc((size_t)N*K*sizeof(float));
    if(!f32){ printf("  [%s] OOM weights\n",label); return; }
    unsigned s=0x1234u+(unsigned)(K*131+N);
    for(size_t i=0;i<(size_t)N*K;i++){ s=s*1664525u+1013904223u; float u=(s>>8)*(1.0f/16777216.0f); f32[i]=(u-0.5f)*0.2f; }

    float *bscale = malloc((size_t)N*sizeof(float));
    ork_w *w = ork_i4a8_mm_pack(c,K,N,f32,bscale);   /* UNIFORM int4 (resident DMA + compact nibbles) */
    if(!w){ printf("  [%s] pack_i4a8 failed (K=%d N=%d)\n",label,K,N); free(f32); free(bscale); return; }

    /* prep scratch: inflated codes f32[N*K] + inv[N]=1 (codes are exact; the tiler rescales by inv) */
    float *qf32 = malloc((size_t)N*K*sizeof(float));
    float *inv1 = malloc((size_t)N*sizeof(float));
    for(int n=0;n<N;n++) inv1[n]=1.0f;
    if(!qf32||!inv1){ printf("  [%s] OOM scratch\n",label); goto done; }

    double samp[REPS];

    /* ---- t_inflate: UNIFORM (sign-extend) ---- */
    for(int i=0;i<WARM;i++) ork_i4a8_slice_inflate_kind(w,qf32,ORK_QK_UNIFORM);
    for(int i=0;i<REPS;i++){ double t=now_us(); ork_i4a8_slice_inflate_kind(w,qf32,ORK_QK_UNIFORM); samp[i]=now_us()-t; }
    stat inf_u = summarize(samp,REPS);

    /* ---- t_inflate: NF4 (LUT vqtbl) — forced on the same nibble store (cost is data-independent) ---- */
    for(int i=0;i<WARM;i++) ork_i4a8_slice_inflate_kind(w,qf32,ORK_QK_CODEBOOK_NF4);
    for(int i=0;i<REPS;i++){ double t=now_us(); ork_i4a8_slice_inflate_kind(w,qf32,ORK_QK_CODEBOOK_NF4); samp[i]=now_us()-t; }
    stat inf_n = summarize(samp,REPS);

    /* ---- t_tile: tile inflated codes into the EXISTING resident DMA buffers + bsync(TO_DEVICE) ---- */
    ork_i4a8_slice_inflate_kind(w,qf32,ORK_QK_UNIFORM);          /* ensure qf32 valid */
    for(int i=0;i<WARM;i++) ork_i8_slice_tile(c,w,qf32,inv1);
    for(int i=0;i<REPS;i++){ double t=now_us(); ork_i8_slice_tile(c,w,qf32,inv1); samp[i]=now_us()-t; }
    stat tile = summarize(samp,REPS);

    /* the slice is re-tiled with valid codes; safe to run NPU matmuls against it for each M */
    printf("  [%s] K=%d N=%d  (resident bytes=%.1f MiB int8 / int4 store=%.1f MiB)\n",
           label, K, N, (double)ork_w_bytes(w)/1048576.0, (double)((size_t)K*N/2)/1048576.0);
    printf("    inflate(uniform) med=%.0f us [p10 %.0f / p90 %.0f]   inflate(NF4) med=%.0f us [p10 %.0f / p90 %.0f]   tile med=%.0f us [p10 %.0f / p90 %.0f]\n",
           inf_u.med, inf_u.p10, inf_u.p90, inf_n.med, inf_n.p10, inf_n.p90, tile.med, tile.p10, tile.p90);
    printf("    %-4s | %9s | %9s | %9s | %9s | %8s | %s\n","M","t_npu(us)","t_infl(us)","t_tile(us)","t_prep(us)","ratio","verdict (prep vs npu)");

    for(int mi=0;mi<nM;mi++){
        int M = Ms[mi];
        int8_t  *A = malloc((size_t)M*K);
        int32_t *C = malloc((size_t)M*N*4);
        if(!A||!C){ printf("    M=%d OOM act\n",M); free(A); free(C); continue; }
        memset(A,1,(size_t)M*K);

        if(ork_i8_mm_run(c,w,M,A,C)!=0){ printf("    M=%d ork_i8_mm_run FAILED\n",M); free(A); free(C); continue; }
        for(int i=0;i<WARM;i++) ork_i8_mm_run(c,w,M,A,C);
        for(int i=0;i<REPS;i++){ double t=now_us(); ork_i8_mm_run(c,w,M,A,C); samp[i]=now_us()-t; }
        stat npu = summarize(samp,REPS);

        /* report against the UNIFORM inflate (the production default path); t_prep = inflate + tile */
        double t_npu = npu.med, t_inf = inf_u.med, t_tile = tile.med, t_prep = t_inf + t_tile;
        double ratio = t_prep / t_npu;
        char verdict[160];
        if(ratio <= 1.0)
            snprintf(verdict,sizeof verdict,"FULLY HIDDEN (%.0f%% headroom spare)", (1.0-ratio)*100.0);
        else {
            const char* bound = (t_tile > t_inf) ? "tile/transfer-bound (bandwidth; a CPU thread can't help)"
                                                 : "inflate-bound (CPU; a prefetch thread overlaps it)";
            snprintf(verdict,sizeof verdict,"residual %.0f us/slice — %s", t_prep - t_npu, bound);
        }
        printf("    %-4d | %9.0f | %9.0f | %9.0f | %9.0f | %8.2f | %s\n",
               M, t_npu, t_inf, t_tile, t_prep, ratio, verdict);
        free(A); free(C);
    }
    printf("\n");
done:
    free(qf32); free(inv1);
    ork_mm_free(c,w);
    free(f32); free(bscale);
}

int main(void){
    check_governors();

    ork_npu *c = ork_npu_init();
    if(!c){ printf("ork_npu_init failed (need /dev/dri + rknpu; run as root / stop orkllm)\n"); return 1; }
    printf("SoC=%s cores=%d validated=%d  ork-driver %s\n\n",
           ork_npu_soc(c), ork_npu_cores(c), ork_npu_validated(c), ork_npu_version());
    printf("Streaming-prefetch headroom (P5.3): per-slice prep = int4->int8 inflate + tile into a REUSED\n");
    printf("resident DMA buffer (NO per-slice alloc). Prefetch wall = max(t_npu, t_prep); ratio = t_prep/t_npu.\n");
    printf("ratio<=1 => fully hidden (streaming ~= resident). >1 => residual (t_prep-t_npu)/slice.\n");
    printf("median of %d reps (%d warmup discarded), CPU-side pinned via taskset.\n\n", REPS, WARM);

    int Ms[] = {1, 32, 128};
    int nM = 3;

    /* ~1.7B-class FFN */
    run_shape(c, 2048, 6144, Ms, nM, "1.7B FFN up   ");
    run_shape(c, 6144, 2048, Ms, nM, "1.7B FFN down ");
    /* ~14B-class FFN — the streaming regime */
    run_shape(c, 5120, 13824, Ms, nM, "14B FFN up    ");
    run_shape(c, 13824, 5120, Ms, nM, "14B FFN down  ");
    /* one attention shape */
    run_shape(c, 5120, 5120, Ms, nM, "attn 5120x5120");

    ork_npu_free(c);
    printf("done.\n");
    return 0;
}
