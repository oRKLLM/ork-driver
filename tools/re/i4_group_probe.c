/* i4_group_probe — what does a PER-GROUP int4 scale actually cost at run time?
 *
 * Per-group scales along K cannot be factored out of the accumulation: C = sum_g (A_g . B_g) * s_g. So one
 * matmul becomes K/G matmuls plus a weighted reduction over K/G int32 partials. The naive objection is
 * "K/G submits", but on the NONBLOCK doorbell spine dispatch is ~5 us and run_chain_i4 puts all of them in
 * ONE submit, so submits are the wrong thing to fear. This separates the three costs that remain:
 *
 *   1. NPU compute      — K/G narrow matmuls vs one wide one (same MACs, worse shape)
 *   2. dispatch         — chained (one submit, N tasks) vs separate (N submits)
 *   3. the REDUCTION    — K/G partials of [M x N] int32 scaled and summed. THIS is the one the doorbell
 *                         cannot help: at M=128,N=1024,G=128,K=3584 it is 28 x 0.5 MB of partials that
 *                         must be read back and combined, on a part with ~21.9 GB/s.
 *
 * Reported so the decision is made on the dominant term, not the loudest one. Board only.
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

/* internal: multi-expert BCHAIN on the nonblock doorbell — many int4 weights, each with M>=1 rows, in ONE
 * submit. A K-group is structurally an "expert", so this is the mechanism the grouped path should be using
 * and is not (a grouped weight has Sk=K/G, which fails ork_i4_mm_run's BCHAIN gate of Sk==1). */
int orki_i4_run_experts_bchain_db(ork_npu *c, const ork_mm_task_i4 *ex, int ntask, int nc);

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + 1e-9*t.tv_nsec; }
static uint32_t g_ = 22222u;
static int8_t r4(void){ g_ ^= g_<<13; g_ ^= g_>>17; g_ ^= g_<<5; return (int8_t)(((int)(g_ & 0xf)) - 8); }

int main(int argc, char **argv){
    const int M = argc>1 ? atoi(argv[1]) : 128;
    const int K = argc>2 ? atoi(argv[2]) : 3584;
    const int N = argc>3 ? atoi(argv[3]) : 1024;
    const int G = argc>4 ? atoi(argv[4]) : 128;
    const int it= argc>5 ? atoi(argv[5]) : 5;
    const int NG = K / G;
    if (K % G) { printf("K%%G must be 0\n"); return 2; }

    ork_npu *c = ork_npu_init();
    if (!c) { printf("i4_group_probe: no NPU — skipping\n"); return 0; }
    printf("i4_group_probe: M=%d K=%d N=%d G=%d -> %d groups, %d iters (SoC=%s)\n", M,K,N,G,NG,it,ork_npu_soc(c));

    int8_t *B = malloc((size_t)K*N), *A = malloc((size_t)M*K);
    for (size_t i=0;i<(size_t)K*N;i++) B[i]=r4();
    for (size_t i=0;i<(size_t)M*K;i++) A[i]=r4();

    /* ---- baseline: ONE full-K matmul, per-channel scale (what we ship today) ---- */
    ork_w *w1 = ork_i4_mm_pack(c, K, N, B);
    if (!w1) { printf("pack full-K FAILED\n"); return 1; }
    int32_t *C1 = malloc((size_t)M*N*4);
    ork_i4_mm_run(c, w1, M, A, C1);                       /* warm */
    double t0 = now(); for (int i=0;i<it;i++) ork_i4_mm_run(c, w1, M, A, C1); double t_base = (now()-t0)/it;

    /* ---- per-group: NG narrow weights, NG contiguous A slices, NG int32 partials ---- */
    ork_w  **wg = calloc(NG, sizeof *wg);
    int8_t **Ag = calloc(NG, sizeof *Ag);
    int32_t**Cg = calloc(NG, sizeof *Cg);
    int8_t  *Bs = malloc((size_t)G*N);
    for (int g=0; g<NG; g++){
        for (int k=0;k<G;k++) memcpy(Bs+(size_t)k*N, B+(size_t)(g*G+k)*N, N);   /* K-slice of the weight */
        wg[g] = ork_i4_mm_pack(c, G, N, Bs);
        if (!wg[g]) { printf("pack group %d FAILED (G=%d N=%d)\n", g, G, N); return 1; }
        Ag[g] = malloc((size_t)M*G);
        for (int m=0;m<M;m++) memcpy(Ag[g]+(size_t)m*G, A+(size_t)m*K + (size_t)g*G, G);  /* A's K-slice */
        Cg[g] = malloc((size_t)M*N*4);
    }
    free(Bs);

    /* 2a. SEPARATE submits */
    ork_mm_task_i4 *tk = calloc(NG, sizeof *tk);
    for (int g=0;g<NG;g++){ tk[g].w=wg[g]; tk[g].M=M; tk[g].A=Ag[g]; tk[g].C=Cg[g]; }
    for (int g=0;g<NG;g++) ork_i4_mm_run(c, wg[g], M, Ag[g], Cg[g]);            /* warm */
    t0 = now();
    for (int i=0;i<it;i++) for (int g=0;g<NG;g++) ork_i4_mm_run(c, wg[g], M, Ag[g], Cg[g]);
    double t_sep = (now()-t0)/it;

    /* 2b. CHAINED: all NG tasks in ONE submit */
    int chain_ok = (ork_i4_mm_run_chain(c, NG, tk) == 0);
    double t_chain = -1;
    if (chain_ok) {
        t0 = now(); for (int i=0;i<it;i++) ork_i4_mm_run_chain(c, NG, tk); t_chain = (now()-t0)/it;
    }

    /* 2c. THE REAL PATH: ork_i4_mm_run_grouped — one grouped weight, the multi-core doorbell, and the
     * per-group float scale-accumulate in its drain. This is what production would actually pay; the
     * separate/chained arms above are decompositions of the same work for attribution. */
    ork_w *wgrp = ork_i4_mm_pack_grouped(c, K, N, B, G);
    double t_grp = -1; int grp_ok = 0;
    if (wgrp) {
        float *aS = malloc((size_t)M*NG*sizeof *aS), *bS = malloc((size_t)NG*N*sizeof *bS);
        float *Cf = malloc((size_t)M*N*sizeof *Cf);
        for (size_t i=0;i<(size_t)M*NG;i++) aS[i]=1.0f/64.0f;
        for (size_t i=0;i<(size_t)NG*N;i++) bS[i]=1.0f/64.0f;
        grp_ok = (ork_i4_mm_run_grouped(c, wgrp, M, A, aS, bS, Cf) == 0);      /* warm */
        if (grp_ok) { t0 = now(); for (int i=0;i<it;i++) ork_i4_mm_run_grouped(c, wgrp, M, A, aS, bS, Cf); t_grp = (now()-t0)/it; }
        free(aS); free(bS); free(Cf);
    }

    /* 2d. THE MECHANISM THAT SHOULD APPLY: multi-expert BCHAIN doorbell — all NG group weights, each
     * carrying all M rows (H-row native batch), in ONE submit. Each group weight has Sk=1/Sn=1 so it
     * satisfies the BCHAIN gate individually; only the FUSED grouped weight (Sk=K/G) does not. */
    double t_bch = -1; int bch_ok = 0;
    {
        int r = orki_i4_run_experts_bchain_db(c, tk, NG, 0);        /* warm */
        bch_ok = (r == 0);
        if (bch_ok) { t0 = now(); for (int i=0;i<it;i++) orki_i4_run_experts_bchain_db(c, tk, NG, 0); t_bch = (now()-t0)/it; }
        else printf("  (multi-expert BCHAIN declined: rc=%d)\n", r);
    }

    /* 2e. CORRECTNESS: the new BCHAIN fast path inside ork_i4_mm_run_grouped must agree with the original
     * row-decomposed path it bypasses. A faster wrong answer is worse than a slow right one, and the two
     * differ only in DISPATCH — same int4 MACs, same per-group scales — so they should agree to fp rounding
     * of the drain's accumulation order, not merely approximately. */
    if (wgrp) {
        float *aS = malloc((size_t)M*NG*sizeof *aS), *bS = malloc((size_t)NG*N*sizeof *bS);
        float *Cnew = malloc((size_t)M*N*sizeof *Cnew), *Cold = malloc((size_t)M*N*sizeof *Cold);
        for (size_t i=0;i<(size_t)M*NG;i++) aS[i]=1.0f/64.0f + 0.001f*(float)(i%7);
        for (size_t i=0;i<(size_t)NG*N;i++) bS[i]=1.0f/64.0f + 0.001f*(float)(i%5);
        int rn = ork_i4_mm_run_grouped(c, wgrp, M, A, aS, bS, Cnew);
        setenv("ORK_I4_GRP_NOBCHAIN","1",1);
        int ro = ork_i4_mm_run_grouped(c, wgrp, M, A, aS, bS, Cold);
        unsetenv("ORK_I4_GRP_NOBCHAIN");
        /* EXACT reference: int32 within each group (what the MAC does), one fp32 scale per (row,group,
         * channel) after — so neither NPU path is being compared only to the other. */
        float *Cref = malloc((size_t)M*N*sizeof *Cref);
        #pragma omp parallel for schedule(static)
        for (int m=0;m<M;m++){
            float *cr=Cref+(size_t)m*N;
            for (int n=0;n<N;n++) cr[n]=0.0f;
            for (int g=0; g<NG; g++){
                const float as=aS[(size_t)m*NG+g]; const float *bs=bS+(size_t)g*N;
                for (int n=0;n<N;n++){
                    int32_t acc=0;
                    for (int k=0;k<G;k++) acc += (int32_t)A[(size_t)m*K+(size_t)g*G+k] * (int32_t)B[(size_t)(g*G+k)*N+n];
                    cr[n]+=(float)acc*as*bs[n];
                }
            }
        }
        { double wn=0, wo=0; size_t bn=0, bo=0;
          for (size_t i=0;i<(size_t)M*N;i++){
            double ref=(double)Cref[i], den=fabs(ref)+1e-6;
            double rn2=fabs((double)Cnew[i]-ref)/den, ro2=fabs((double)Cold[i]-ref)/den;
            if(rn2>wn) wn=rn2; if(ro2>wo) wo=ro2;
            if(rn2>1e-5) bn++; if(ro2>1e-5) bo++; }
          printf("  vs EXACT CPU REF: bchain worst %.3g (%zu bad)   row-decomposed worst %.3g (%zu bad)\n",
                 wn, bn, wo, bo); }
        free(Cref);
        if (rn || ro) printf("  CORRECTNESS: run failed (new=%d old=%d)\n", rn, ro);
        else {
            double worst = 0; size_t bad = 0;
            for (size_t i=0;i<(size_t)M*N;i++){
                double d = fabs((double)Cnew[i]-(double)Cold[i]);
                double r = d / (fabs((double)Cold[i]) + 1e-6);
                if (r > worst) worst = r;
                if (r > 1e-5) bad++;
            }
            printf("  CORRECTNESS: bchain vs row-decomposed  worst rel %.3g, %zu/%zu over 1e-5  %s\n",
                   worst, bad, (size_t)M*N, bad ? "FAIL" : "OK");
        }
        free(aS); free(bS); free(Cnew); free(Cold);
    }

    /* 3. the REDUCTION: scale each partial by its group scale and sum in fp32 */
    float *sg = malloc((size_t)NG*sizeof *sg); for (int g=0;g<NG;g++) sg[g] = 1.0f/(float)(g+2);
    float *out = malloc((size_t)M*N*sizeof *out);
    t0 = now();
    for (int i=0;i<it;i++){
        #pragma omp parallel for schedule(static)
        for (int m=0;m<M;m++){
            float *o = out + (size_t)m*N;
            memset(o, 0, (size_t)N*sizeof *o);
            for (int g=0;g<NG;g++){
                const int32_t *p = Cg[g] + (size_t)m*N; const float s = sg[g];
                for (int n=0;n<N;n++) o[n] += (float)p[n] * s;
            }
        }
    }
    double t_red = (now()-t0)/it;
    const double part_mb = (double)NG*M*N*4/1048576.0;

    printf("\n  baseline  1 x full-K submit          %8.3f ms\n", t_base*1e3);
    printf("  grouped   %d separate submits         %8.3f ms  (%.2fx baseline)\n", NG, t_sep*1e3, t_sep/t_base);
    if (chain_ok) printf("  grouped   %d tasks, ONE chained submit %8.3f ms  (%.2fx baseline)\n", NG, t_chain*1e3, t_chain/t_base);
    else          printf("  grouped   chained: DECLINED by run_chain_i4\n");
    printf("  reduction %d x [%d x %d] int32 -> f32  %8.3f ms  (%.1f MiB of partials, %.1f GB/s)\n",
           NG, M, N, t_red*1e3, part_mb, part_mb/1024.0/t_red);
    if (bch_ok) printf("  BCHAIN-db %d groups, ONE doorbell submit %8.3f ms  (%.2fx baseline)\n", NG, t_bch*1e3, t_bch/t_base);
    if (grp_ok) printf("  REAL      ork_i4_mm_run_grouped         %8.3f ms  (%.2fx baseline, incl. its own drain)\n",
                       t_grp*1e3, t_grp/t_base);
    else        printf("  REAL      ork_i4_mm_run_grouped: %s\n", wgrp ? "REFUSED at this shape" : "pack_grouped FAILED");
    double best = chain_ok && t_chain < t_sep ? t_chain : t_sep;
    printf("\n  TOTAL per-group (best dispatch + reduction) %.3f ms = %.2fx the per-channel baseline\n",
           (best+t_red)*1e3, (best+t_red)/t_base);
    printf("  of which reduction is %.0f%% — %s\n", 100.0*t_red/(best+t_red),
           t_red > best ? "the REDUCTION dominates: a faster dispatcher cannot fix this"
                        : "DISPATCH dominates: doorbell/heterogeneous overlap is the lever");

    for (int g=0;g<NG;g++){ ork_mm_free(c,wg[g]); free(Ag[g]); free(Cg[g]); }
    if (wgrp) ork_mm_free(c,wgrp);
    ork_mm_free(c,w1);
    free(wg);free(Ag);free(Cg);free(tk);free(sg);free(out);free(C1);free(A);free(B);
    ork_npu_free(c);
    return 0;
}
