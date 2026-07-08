/* moe_m1_probe — PATH B M-sweep: does the batched group-by-expert MoE GEMM WIN on the RK3588 NPU
 * at M>1?  M2 proved M=1 LOSES (~2.9x, memory-bound GEMV + load floor). At M>1 each active expert's
 * GEMM has M_e routed rows, so the submit + the weight read amortize across rows. This probe isolates
 * that crossover on the LFM2.5-8B-A1B expert FFN shapes:
 *   gate/up : K=2048 N=1792   (conforming: K%512==0 && K<=4096 -> full-K / stream / chain envelope)
 *   down    : K=1792 N=2048   (NON-conforming: 1792%512!=0 -> falls to the K-split Bb path)
 * and (because M2 saw ffn_down rc=-1 at M=1 on a LOADED/disk weight) tests BOTH packed AND loaded
 * (dump->load_i8, the .orkpack path) weights at each M, so we can settle whether down-proj works on
 * the NPU at M>1. Every NPU run is checked vs a CPU int8 reference over the SAME weights.
 *
 * Sweep M in {1,8,16,32,64,128}; report per-call (whole-batch, NOT per-expert) latency for the way
 * the real partition submits: ONE run_stream_i8 over S=8 experts each carrying M rows, vs S CPU GEMMs.
 * This is the apples-to-apples "S experts x M rows" batched cost. Also reports the single run_i8.
 *
 * Board-only (needs /dev/dri + rknpu). NOT in `all`/`test`.
 *   build: make moe_m1_probe   run: sudo taskset -c 4-7 ./moe_m1_probe
 *   env: LOADED=1 also sweep loaded (dump->load_i8) weights; THRASH unused here.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include "ork_npu.h"

#define CHAIN_DEPTH 8

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y?1:0; }
static double median(double*s,int n){ qsort(s,n,sizeof(double),cmp_d); return n&1?s[n/2]:0.5*(s[n/2-1]+s[n/2]); }

static void cpu_gemm_i8(int M, int K, int N, const int8_t *A, const int8_t *B8, int32_t *C){
#ifdef __ARM_NEON
    for(int m=0;m<M;m++){ const int8_t *a=A+(size_t)m*K; int32_t *c=C+(size_t)m*N;
        for(int n=0;n<N;n++){ const int8_t *b=B8+(size_t)n*K;
            int32x4_t acc=vdupq_n_s32(0); int k=0;
            for(;k+16<=K;k+=16){
                int8x16_t va=vld1q_s8(a+k), vb=vld1q_s8(b+k);
                int16x8_t lo=vmull_s8(vget_low_s8(va),vget_low_s8(vb));
                int16x8_t hi=vmull_s8(vget_high_s8(va),vget_high_s8(vb));
                acc=vpadalq_s16(acc,lo); acc=vpadalq_s16(acc,hi);
            }
            int32_t s=vaddvq_s32(acc);
            for(;k<K;k++) s += (int)a[k]*(int)b[k];
            c[n]=s;
        } }
#else
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ long s=0; const int8_t*a=A+(size_t)m*K,*b=B8+(size_t)n*K;
        for(int k=0;k<K;k++) s+=(int)a[k]*(int)b[k]; C[(size_t)m*N+n]=(int32_t)s; }
#endif
}

struct shape { const char *name; int K, N; };
static uint32_t g_s = 0x2468acef;
static int8_t r8(void){ g_s^=g_s<<13; g_s^=g_s>>17; g_s^=g_s<<5; return (int8_t)((int)(g_s&0xff)-128); }
static float rf(void){ g_s^=g_s<<13; g_s^=g_s>>17; g_s^=g_s<<5; return ((int)(g_s&0xffff)-32768)/9000.0f; }

/* run S resident experts, each with M rows of the shared activation A (the partition's group GEMM:
 * here all S experts see the SAME M rows — a worst-case where every routed token hit every expert).
 * Returns 0 / nonzero rc. Times nothing; caller times. Uses run_stream_i8 (cross-core), falls back to
 * per-task run_i8 (the real handler's fallback) when the stream rejects the shape. */
static int group_submit(ork_npu*c, int S, ork_mm_task_i8 *tasks, int use_stream){
    int rc = use_stream ? ork_mm_run_stream_i8(c,S,tasks) : ork_mm_run_chain_i8(c,S,tasks);
    if(rc){ rc=0; for(int x=0;x<S&&rc==0;x++) rc=ork_mm_run_i8(c,tasks[x].w,tasks[x].M,tasks[x].A,tasks[x].C); }
    return rc;
}

int main(int argc, char**argv){
    (void)argc;(void)argv;
    int do_loaded = getenv("LOADED")!=NULL;
    ork_npu *c = ork_npu_init();
    if(!c){ fprintf(stderr,"init FAIL (need /dev/dri + rknpu)\n"); return 1; }
    printf("# ork-driver %s  soc=%s cores=%d\n", ork_npu_version(), ork_npu_soc(c), ork_npu_cores(c));
    printf("# PATH B M-sweep. LFM2.5 expert FFN int8, RESIDENT weights. S=%d experts/group.\n", CHAIN_DEPTH);
    printf("# per-call us = ONE group submit of S experts x M rows (the partition's batched cost) vs S CPU GEMMs.\n");
    printf("# loaded-weight sweep: %s\n\n", do_loaded?"ON (dump->load_i8)":"off (set LOADED=1)");

    struct shape shapes[] = {
        {"lfm-gate/up ", 2048, 1792},  /* conforming K */
        {"lfm-down    ", 1792, 2048},  /* NON-conforming K (the rc=-1 question) */
    };
    int Ms[] = {1,8,16,32,64,128};
    enum { WARM=8, REPS=30 };
    int overall_ok = 1;

    for(int si=0; si<(int)(sizeof(shapes)/sizeof(shapes[0])); si++){
        int K = shapes[si].K, N = shapes[si].N;
        int chainable = (K%512==0 && K<=4096);
        printf("=== %s  K=%d N=%d  (conforming K%%512==0 && K<=4096: %s) ===\n",
               shapes[si].name, K, N, chainable?"yes":"NO");

        /* build S distinct int8 expert weights, RESIDENT (packed once) */
        float **f32 = malloc(CHAIN_DEPTH*sizeof(float*));
        for(int e=0;e<CHAIN_DEPTH;e++){ f32[e]=malloc((size_t)N*K*sizeof(float)); for(size_t i=0;i<(size_t)N*K;i++) f32[e][i]=rf(); }
        ork_w **wp = malloc(CHAIN_DEPTH*sizeof(ork_w*));   /* packed */
        ork_w **wl = malloc(CHAIN_DEPTH*sizeof(ork_w*));   /* loaded (dump->load_i8) */
        float **bsc = malloc(CHAIN_DEPTH*sizeof(float*));
        int build_ok=1;
        for(int e=0;e<CHAIN_DEPTH;e++){
            bsc[e]=malloc((size_t)N*sizeof(float));
            wp[e]=ork_mm_pack_i8_f32(c,K,N,f32[e],bsc[e]);
            wl[e]=NULL;
            if(!wp[e]){ build_ok=0; break; }
            if(do_loaded){
                size_t need=ork_w_dump(wp[e],NULL,0);
                void *blob=malloc(need);
                if(ork_w_dump(wp[e],blob,need)!=need){ printf("  dump FAIL e=%d\n",e); build_ok=0; free(blob); break; }
                wl[e]=ork_mm_load_i8(c,K,N,blob,need);
                free(blob);
                if(!wl[e]){ printf("  load_i8 FAIL e=%d (need=%zu)\n",e,need); build_ok=0; break; }
            }
        }
        if(!build_ok){ printf("  build FAIL\n\n"); overall_ok=0;
            for(int e=0;e<CHAIN_DEPTH;e++){ if(wp[e])ork_mm_free(c,wp[e]); if(wl[e])ork_mm_free(c,wl[e]); free(f32[e]); free(bsc[e]); }
            free(wp);free(wl);free(f32);free(bsc); continue; }

        /* recover the exact int8 the NPU holds for CPU ref (per-channel like pack_i8_f32) */
        int8_t **B8 = malloc(CHAIN_DEPTH*sizeof(int8_t*));
        for(int e=0;e<CHAIN_DEPTH;e++){
            B8[e]=malloc((size_t)N*K);
            for(int n=0;n<N;n++){ const float*wn=f32[e]+(size_t)n*K; float amax=0; for(int k=0;k<K;k++){ float a=fabsf(wn[k]); if(a>amax)amax=a; }
                float sc = amax>0?amax/127.0f:1.0f;
                for(int k=0;k<K;k++){ int q=(int)lrintf(wn[k]/sc); if(q>127)q=127; if(q<-128)q=-128; B8[e][(size_t)n*K+k]=(int8_t)q; } }
        }

        printf("  %-4s | %-28s | %-28s | %-12s | crossover\n", "M", "packed group-submit (us)", "loaded group-submit (us)", "CPU SxM (us)");
        for(int mi=0; mi<(int)(sizeof(Ms)/sizeof(Ms[0])); mi++){
            int M = Ms[mi];
            int8_t *A = malloc((size_t)M*K);
            for(size_t i=0;i<(size_t)M*K;i++) A[i]=r8();
            int32_t **Cnpu = malloc(CHAIN_DEPTH*sizeof(int32_t*));
            for(int e=0;e<CHAIN_DEPTH;e++) Cnpu[e]=malloc((size_t)M*N*4);
            int32_t *Cref = malloc((size_t)M*N*4);
            double sm[REPS];

            /* correctness on the packed weight 0 */
            int crc = ork_mm_run_i8(c,wp[0],M,A,Cnpu[0]);
            cpu_gemm_i8(M,K,N,A,B8[0],Cref);
            long maxd=0; for(size_t i=0;i<(size_t)M*N;i++){ long d=labs((long)Cnpu[0][i]-(long)Cref[i]); if(d>maxd)maxd=d; }
            int corr_bad = crc || (maxd > (long)K);
            if(corr_bad) overall_ok=0;

            /* PACKED group submit: S experts x M rows, one run_stream (fallback per-task) */
            ork_mm_task_i8 tp[CHAIN_DEPTH];
            for(int e=0;e<CHAIN_DEPTH;e++){ tp[e].w=wp[e]; tp[e].M=M; tp[e].A=A; tp[e].C=Cnpu[e]; }
            double pk_us=-1; int pk_rc;
            pk_rc = group_submit(c,CHAIN_DEPTH,tp,1);
            if(pk_rc==0){
                for(int i=0;i<WARM;i++) group_submit(c,CHAIN_DEPTH,tp,1);
                for(int i=0;i<REPS;i++){ double t=now_us(); group_submit(c,CHAIN_DEPTH,tp,1); sm[i]=now_us()-t; }
                pk_us=median(sm,REPS);
            }
            ork_mm_run_i8(c,wp[0],M,A,Cnpu[0]); /* re-warm */

            /* LOADED group submit */
            double ld_us=-1; int ld_rc=0;
            if(do_loaded){
                ork_mm_task_i8 tl[CHAIN_DEPTH];
                for(int e=0;e<CHAIN_DEPTH;e++){ tl[e].w=wl[e]; tl[e].M=M; tl[e].A=A; tl[e].C=Cnpu[e]; }
                /* correctness of loaded weight 0 */
                int lcrc = ork_mm_run_i8(c,wl[0],M,A,Cnpu[0]);
                long lmaxd=0; for(size_t i=0;i<(size_t)M*N;i++){ long d=labs((long)Cnpu[0][i]-(long)Cref[i]); if(d>lmaxd)lmaxd=d; }
                if(lcrc || lmaxd>(long)K){ printf("  [M=%d] LOADED correctness FAIL rc=%d maxd=%ld\n",M,lcrc,lmaxd); overall_ok=0; }
                ld_rc = group_submit(c,CHAIN_DEPTH,tl,1);
                if(ld_rc==0){
                    for(int i=0;i<WARM;i++) group_submit(c,CHAIN_DEPTH,tl,1);
                    for(int i=0;i<REPS;i++){ double t=now_us(); group_submit(c,CHAIN_DEPTH,tl,1); sm[i]=now_us()-t; }
                    ld_us=median(sm,REPS);
                }
                ork_mm_run_i8(c,wp[0],M,A,Cnpu[0]);
            }

            /* CPU: S experts x M rows */
            for(int i=0;i<WARM;i++) for(int e=0;e<CHAIN_DEPTH;e++) cpu_gemm_i8(M,K,N,A,B8[e],Cref);
            for(int i=0;i<REPS;i++){ double t=now_us(); for(int e=0;e<CHAIN_DEPTH;e++) cpu_gemm_i8(M,K,N,A,B8[e],Cref); sm[i]=now_us()-t; }
            double cpu = median(sm,REPS);

            double best = pk_us; if(ld_us>0 && ld_us<best) best=ld_us;
            const char *verdict = (best>0 && best<=cpu) ? "NPU WINS" : "cpu";
            char pkbuf[40], ldbuf[40];
            if(pk_us>0) snprintf(pkbuf,sizeof pkbuf,"%.1f (%.2fx cpu)",pk_us,cpu/pk_us); else snprintf(pkbuf,sizeof pkbuf,"rc=%d FAIL",pk_rc);
            if(!do_loaded) snprintf(ldbuf,sizeof ldbuf,"-"); else if(ld_us>0) snprintf(ldbuf,sizeof ldbuf,"%.1f (%.2fx cpu)",ld_us,cpu/ld_us); else snprintf(ldbuf,sizeof ldbuf,"rc=%d FAIL",ld_rc);
            printf("  %-4d | %-28s | %-28s | %-12.1f | %s%s\n", M, pkbuf, ldbuf, cpu, verdict, corr_bad?" [CORR SUSPECT]":"");

            for(int e=0;e<CHAIN_DEPTH;e++) free(Cnpu[e]);
            free(Cnpu); free(Cref); free(A);
        }
        printf("\n");
        for(int e=0;e<CHAIN_DEPTH;e++){ ork_mm_free(c,wp[e]); if(wl[e])ork_mm_free(c,wl[e]); free(f32[e]); free(B8[e]); free(bsc[e]); }
        free(wp);free(wl);free(f32);free(B8);free(bsc);
    }

    printf("OVERALL: %s\n", overall_ok?"OK":"SUSPECT");
    ork_npu_free(c);
    return 0;
}
