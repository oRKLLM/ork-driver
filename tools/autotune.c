/* tools/autotune.c — NPU matmul tiling AUTOTUNER for ork-driver.
 *
 * Given the unique prefill matmul shapes (M,K,N) of a model, systematically search the tiling
 * configs ork parameterizes and find the per-shape config that maximizes NPU utilization
 * (effective GOPS), instead of hand-iterating. Each measured config is BIT-EXACT-GATED against a
 * CPU int8 reference computed ONCE per shape (not per config) — a config whose NPU output diverges
 * is reported WRONG and never chosen.
 *
 * Search space (knobs ork already exposes):
 *   - core count   : swept 1..soc.cores via ork_npu_set_core_budget() (in-process, per-call)
 *   - M-tile / N-subtile : ORK_SMALLTILE_M / ORK_SMALLTILE_N (per-process env; see driver below)
 *   - K-tile       : ORK_KTILE (Phase B; per-process env) — overrides the int8 K-slice so a
 *                    smaller K_tile yields a bigger R=pow2_floor(2*cbuf/K_tile) -> larger M-tile.
 *
 * Because run_i8 caches its env knobs at first call (static), the M/N/K-tile knobs are swept by a
 * DRIVER that re-execs this binary once per env-combo (the shell loop in run_autotune.sh). This
 * process: packs each shape once, times ork_mm_run_i8 warm (rep0/1 warmup, median of rep2..r),
 * bit-exact-gates, and prints one parseable line per (shape, core) it ran under the ambient env.
 *
 *   make autotune && sudo ./autotune [M] [reps]
 * Output lines:  [AUTOTUNE] shape=NAME M=.. K=.. N=.. cores=.. STM=.. STN=.. KT=.. \
 *                us=.. gops=.. status=OK|WRONG(bad/tot)|WEDGE
 * Safe: bit-exact-gated; a WRONG config is reported not used; a wedge is recoverable (soft reset).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static unsigned sd=2463534242u;
static int8_t r8(void){ sd^=sd<<13; sd^=sd>>17; sd^=sd<<5; return (int8_t)((int)(sd%7)-3); }

/* Qwen2.5-7B-Instruct prefill matmul shapes (hidden 3584, inter 18944, 28 q heads / 4 kv heads,
 * head_dim 128). K%32==0, N%32==0 required by run_i8 int8. */
struct shape { const char *name; int K, N; };
static struct shape SHAPES[] = {
    {"q_proj",   3584,  3584},
    {"kv_proj",  3584,   512},   /* k_proj and v_proj are identical (4*128) */
    {"o_proj",   3584,  3584},
    {"gate_up",  3584, 18944},   /* gate_proj and up_proj identical */
    {"down",    18944,  3584},
};
#define NSHAPE ((int)(sizeof(SHAPES)/sizeof(SHAPES[0])))

int main(int argc,char**argv){
    int M    = argc>1?atoi(argv[1]):256;
    int reps = argc>2?atoi(argv[2]):5;          /* rep0,rep1 warmup; median of the rest timed */
    if(reps<3)reps=3;

    const char*e;
    int STM = (e=getenv("ORK_SMALLTILE_M")) ? atoi(e) : 0;
    int STN = (e=getenv("ORK_SMALLTILE_N")) ? atoi(e) : 0;
    int KT  = (e=getenv("ORK_KTILE"))        ? atoi(e) : 0;
    int only = (e=getenv("AT_SHAPE")) ? atoi(e) : -1;     /* run a single shape index, or all */
    int max_cores = (e=getenv("AT_MAXCORES")) ? atoi(e) : 0;

    ork_npu *c = ork_npu_init();
    if(!c){ printf("init failed (NPU?)\n"); return 1; }
    int socc = 3;   /* rk3588 default; clamp to AT_MAXCORES if set */
    if(max_cores>0 && max_cores<socc) socc=max_cores;
    fprintf(stderr,"[autotune] SoC %s  M=%d reps=%d  env STM=%d STN=%d KT=%d\n",
            ork_npu_soc(c),M,reps,STM,STN,KT);

    int32_t *C   = NULL; size_t Csz=0;
    int32_t *ref = NULL; size_t refsz=0;
    int8_t  *A   = NULL; size_t Asz=0;
    int8_t  *B   = NULL; size_t Bsz=0;
    double *rt = malloc(sizeof(double)*reps);

    for(int s=0;s<NSHAPE;s++){
        if(only>=0 && s!=only) continue;
        int K=SHAPES[s].K, N=SHAPES[s].N;
        size_t need;
        need=(size_t)M*K;       if(Asz<need){A=realloc(A,need);Asz=need;}
        need=(size_t)K*N;       if(Bsz<need){B=realloc(B,need);Bsz=need;}
        need=(size_t)M*N*4;     if(Csz<need){C=realloc(C,need);Csz=need;}
        if(refsz<(size_t)M*N*4){ref=realloc(ref,(size_t)M*N*4);refsz=(size_t)M*N*4;}
        sd=0x1234u+s*7;
        for(size_t j=0;j<(size_t)M*K;j++)A[j]=r8();
        for(size_t j=0;j<(size_t)K*N;j++)B[j]=r8();

        /* CPU int8 reference — computed ONCE per shape, shared across all configs of this shape. */
        for(int m=0;m<M;m++){
            for(int n=0;n<N;n++){
                int32_t acc=0; const int8_t*ar=&A[(size_t)m*K];
                for(int k=0;k<K;k++) acc += (int)ar[k]*(int)B[(size_t)k*N+n];
                ref[(size_t)m*N+n]=acc;
            }
        }

        ork_w *w = ork_mm_pack_i8(c,K,N,B);
        if(!w){ printf("[AUTOTUNE] shape=%s M=%d K=%d N=%d status=PACKFAIL\n",SHAPES[s].name,M,K,N); continue; }

        for(int cores=1;cores<=socc;cores++){
            ork_npu_set_core_budget(c,cores);
            int rc=-99;
            for(int r=0;r<reps;r++){
                double t0=now_us();
                rc=ork_mm_run_i8(c,w,M,A,C);
                double dt=now_us()-t0;
                rt[r]=dt;
                if(rc!=0) break;
            }
            const char*status; char sbuf[48]; double us=0,gops=0;
            if(rc!=0){ snprintf(sbuf,sizeof sbuf,"WEDGE(rc=%d)",rc); status=sbuf; }
            else {
                /* bit-exact gate */
                long bad=0; for(size_t j=0;j<(size_t)M*N;j++) if(C[j]!=ref[j]) bad++;
                /* timed = median of rep2..reps-1 */
                int nt=reps-2; double *tt=&rt[2];
                for(int i=0;i<nt;i++)for(int j=i+1;j<nt;j++) if(tt[j]<tt[i]){double x=tt[i];tt[i]=tt[j];tt[j]=x;}
                us=tt[nt/2];
                gops = us>0 ? (2.0*M*K*N)/(us*1e3) : 0;
                if(bad){ snprintf(sbuf,sizeof sbuf,"WRONG(%ld/%d)",bad,M*N); status=sbuf; }
                else status="OK";
            }
            printf("[AUTOTUNE] shape=%s M=%d K=%d N=%d cores=%d STM=%d STN=%d KT=%d us=%.0f gops=%.0f status=%s\n",
                   SHAPES[s].name,M,K,N,cores,STM,STN,KT,us,gops,status);
            fflush(stdout);
        }
        ork_mm_free(c,w);
    }
    free(A);free(B);free(C);free(ref);free(rt);
    ork_npu_free(c);
    return 0;
}
