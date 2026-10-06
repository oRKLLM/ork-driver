/* train_scope_bench — the cheapest decisive measurement for "can we TRAIN on the RK3588 NPU?".
 *
 * Training differs from inference in one structural way. ork-driver's resident-weight design packs a
 * weight ONCE into the tiled layout ([Ntile][Ktile][16][32]) and runs it many times; a training step
 * changes every weight, and both backward matmuls need an operand that is fresh each step:
 *     dX = dY · Wᵀ   → needs Wᵀ packed (W changes once per step)
 *     dW = Xᵀ · dY   → needs dY packed (dY changes every step)
 * So the gating question is: what does a pack cost relative to the matmul it enables? If pack ≳ run,
 * per-step re-packing dominates and the design is inference-shaped, not training-shaped.
 *
 * PACK COST IS INDEPENDENT OF M (it is K×N work), which is what makes this the cheapest decisive test:
 * it needs no large M and is unaffected by the fp16 M-cap mess below.
 *
 * M-cap context, confirmed in source: orki_f16_sched(K) returns true only for K a POWER OF TWO in
 * [128,2048), and only then does orki_f16_mcap use the 11-bank bound 180224/K ("measured exact" for
 * K∈{256,512,1024}). Every other K — including K=3584 — falls to the 1-bank 16384/K. So at M=4096:
 *     K=1024 → mcap 176 →   24 M-tiles   (measured envelope)
 *     K=3584 → mcap   4 → 1024 M-tiles   (CONSERVATIVE software bound, never measured)
 * At ~1.3–13 µs per task, 1024 tasks is 1.3–13 ms of pure dispatch: timing that would measure the
 * scheduler, not the matmul. ORK_F16_MTILE overrides the cap, so arm 2 tests whether K=3584 is actually
 * correct at a larger tile — if it is, the conservative bound leaves ~12.5× on the table.
 *
 * Self-validating: every run arm is checked against an fp64 CPU reference on sampled outputs and reports
 * max relative error, so a silent miscompute from an over-ridden cap shows up as error, not as a speedup.
 */
#include <ork_npu.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

/* ork_f16 is _Float16 (include/ork_npu.h:19) — native half, so conversion is a plain cast. An earlier
 * draft hand-rolled bit-manipulation helpers assuming it was unsigned short; the compiler caught that.
 * Worth recording: the same wrong assumption in a training harness would silently corrupt gradient
 * buffers rather than fail to compile, because the bit pattern of a short and a half are both 16 bits. */
#define f2h(f) ((ork_f16)(f))
#define h2f(h) ((float)(h))

static unsigned s_=22222;
static float rnd(void){ s_=s_*1103515245u+12345u; return (float)((int)((s_>>16)&0x3ff)-512)/512.0f; }

/* fp64 reference for ONE output element — the only honest way to price fp16 accumulation error */
static double ref_elem(const ork_f16*A,const ork_f16*B,int K,int N,int m,int n){
    double acc=0;
    for(int k=0;k<K;k++) acc += (double)h2f(A[(size_t)m*K+k]) * (double)h2f(B[(size_t)k*N+n]);
    return acc;
}

int main(int argc,char**argv){
    int M    = argc>1?atoi(argv[1]):4096;
    int reps = argc>2?atoi(argv[2]):3;
    int dorun= argc>3?atoi(argv[3]):1;      /* 0 = pack costs only (fast, M-independent, no NPU submits) */
    ork_npu *c = ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }

    struct { int K,N; const char*what; } sh[] = {
        {1024, 3584, "fwd gate/up   [M,1024]x[1024,3584]"},
        {3584, 1024, "fwd down      [M,3584]x[3584,1024]"},
        {1024, 1024, "dX shape      [M,1024]x[1024,1024]"},
    };
    const char *ov = getenv("ORK_F16_MTILE");
    printf("M=%d reps=%d run=%d  ORK_F16_MTILE=%s\n\n", M, reps, dorun, ov?ov:"unset");
    printf("%-34s %9s %9s %9s %9s %10s %11s\n",
           "shape","pack_us","repack_us","run_us","GFLOP/s","pack/run","max_relerr");

    for(size_t q=0;q<sizeof sh/sizeof sh[0];q++){
        int K=sh[q].K,N=sh[q].N;
        ork_f16 *B=malloc((size_t)K*N*2);
        if(!B){ printf("%-34s  B alloc failed\n",sh[q].what); continue; }
        for(size_t i=0;i<(size_t)K*N;i++) B[i]=f2h(rnd());

        /* ---- PACK: the decisive number. K*N work, independent of M. ---- */
        double t0=now_us(); ork_w *w=NULL;
        for(int r=0;r<reps;r++){ if(w) ork_mm_free(c,w); w=ork_f16_mm_pack(c,K,N,B); }
        double pack_us=(now_us()-t0)/reps;
        if(!w){ printf("%-34s  PACK FAILED (K%%32=%d N%%16=%d)\n",sh[q].what,K%32,N%16); free(B); continue; }

        /* ---- REPACK: same tiling into the EXISTING ork_w — no bcreate/bfree, no IOMMU churn ---- */
        t0=now_us();
        int rerr=0; for(int r=0;r<reps;r++) if(ork_f16_mm_repack(c,w,K,N,B)<0) rerr=1;
        double repack_us=(now_us()-t0)/reps;

        double run_us=0,gflops=0,maxrel=-1;
        if(dorun){
            ork_f16 *A=malloc((size_t)M*K*2); float *C=malloc((size_t)M*N*4);
            if(A&&C){
                for(size_t i=0;i<(size_t)M*K;i++) A[i]=f2h(rnd());
                if(ork_f16_mm_run(c,w,M,A,C)==0){           /* warm */
                    t0=now_us(); for(int r=0;r<reps;r++) ork_f16_mm_run(c,w,M,A,C);
                    run_us=(now_us()-t0)/reps;
                    gflops = 2.0*M*K*N/(run_us*1e3);
                    maxrel=0;
                    for(int s=0;s<12;s++){
                        int m=(s*M)/12, n=(s*N)/12+(s%3); if(n>=N)n=N-1;
                        double rr=ref_elem(A,B,K,N,m,n), got=C[(size_t)m*N+n];
                        double den=fabs(rr)>1e-6?fabs(rr):1e-6, rel=fabs(got-rr)/den;
                        if(rel>maxrel) maxrel=rel;
                    }
                } else printf("  (run failed for %s)\n", sh[q].what);
            }
            free(A); free(C);
        }
        char ratio[16], err[16];
        if(run_us>0) snprintf(ratio,sizeof ratio,"%.2f",pack_us/run_us); else snprintf(ratio,sizeof ratio,"-");
        if(maxrel>=0) snprintf(err,sizeof err,"%.2e",maxrel);            else snprintf(err,sizeof err,"-");
        printf("%-34s %9.0f %9.0f %9.0f %9.1f %10s %11s\n", sh[q].what,
               pack_us, rerr?-1.0:repack_us, run_us, gflops, ratio, err);
        ork_mm_free(c,w); free(B);
    }
    printf("\npack/run >= 1 : a per-step re-pack costs more than the matmul it enables.\n");
    printf("repack << pack: allocate-once/refresh-per-step avoids the IOMMU alloc+free, which is the\n"
           "                difference between a viable training inner loop and an unviable one.\n");
    ork_npu_free(c); return 0;
}
