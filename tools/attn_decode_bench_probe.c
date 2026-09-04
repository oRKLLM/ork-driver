/* attn_decode_bench_probe — DECODE attention, HOST-SPLIT NPU (Tier 12e) vs CPU, ARBITRARY scores, and the
 * Tier 12f RESIDENT-K/V perf model. One decode step: NH heads, single query row (Nq=1) attending to L cached
 * keys. Per head the two heavy matmuls (QK^T, e.V) run on the NPU; the [1,L] softmax runs on the HOST in fp
 * with a REAL per-head max-subtraction — correct for arbitrary (signed) scores (mirrors decode.c:attn_decode_npu).
 *
 * Two NPU timings, to isolate the 12f blocker:
 *   - "per-step pack": packs K^T/V EVERY token (what attn_decode_npu does today) — packing-bound, perf-negative.
 *   - "resident K/V":  packs K^T/V ONCE, per-step does only the matmuls (models Tier 12f: pack-once + append the
 *                      one new key/value each step, an O(HD) tile-write that's ~free vs the matmuls).
 * The gap between them IS the packing overhead 12f removes.
 *   sudo env ORK_MM_TIMEOUT=3000 ./attn_decode_bench_probe [NH] [L] [HD] [reps]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3 + t.tv_nsec/1e6; }

int main(int argc,char**argv){
    int NH=argc>1?atoi(argv[1]):32, L=argc>2?atoi(argv[2]):512, HD=argc>3?atoi(argv[3]):128, REP=argc>4?atoi(argv[4]):20;
    int Kp=512;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("attn_decode_bench_probe: NH=%d L=%d HD=%d — host-split NPU decode attn (arbitrary scores)\n",NH,L,HD);
    float scale=1.0f/sqrtf((float)HD);
    /* per-head ARBITRARY (signed) Q/K/V fp + their int8 quant */
    float **Qf=calloc(NH,sizeof*Qf),**Kf=calloc(NH,sizeof*Kf),**Vf=calloc(NH,sizeof*Vf);
    int8_t **Q8=calloc(NH,sizeof*Q8); float *qsv=calloc(NH,sizeof(float)),*ksv=calloc(NH,sizeof(float)),*vsv=calloc(NH,sizeof(float));
    ork_w **wkt=calloc(NH,sizeof*wkt),**wv=calloc(NH,sizeof*wv);
    int8_t **ktq=calloc(NH,sizeof*ktq),**vq=calloc(NH,sizeof*vq);   /* kept for the per-step repack timing */
    uint32_t g=0x77;
    for(int h=0;h<NH;h++){ Qf[h]=malloc((size_t)HD*4); Kf[h]=malloc((size_t)L*HD*4); Vf[h]=malloc((size_t)L*HD*4);
        for(int e=0;e<HD;e++){ g=g*1664525u+1013904223u; Qf[h][e]=((int)((g>>26)%17)-8)*0.1f; }
        for(size_t i=0;i<(size_t)L*HD;i++){ g=g*1664525u+1013904223u; Kf[h][i]=((int)((g>>26)%17)-8)*0.1f; }
        for(size_t i=0;i<(size_t)L*HD;i++){ g=g*1664525u+1013904223u; Vf[h][i]=((int)((g>>26)%17)-8)*0.1f; }
        float qmax=1e-6f,kmax=1e-6f,vmax=1e-6f;
        for(int e=0;e<HD;e++){ float a=fabsf(Qf[h][e]); if(a>qmax)qmax=a; }
        for(size_t i=0;i<(size_t)L*HD;i++){ float ka=fabsf(Kf[h][i]),va=fabsf(Vf[h][i]); if(ka>kmax)kmax=ka; if(va>vmax)vmax=va; }
        qsv[h]=127.0f/qmax; ksv[h]=127.0f/kmax; vsv[h]=127.0f/vmax;
        Q8[h]=malloc((size_t)Kp); memset(Q8[h],0,Kp); for(int e=0;e<HD;e++) Q8[h][e]=(int8_t)lrintf(Qf[h][e]*qsv[h]);
        /* RESIDENT pack (once): K^T[Kp,L] and V[L,HD] int8 */
        int8_t *KTp=calloc((size_t)Kp*L,1),*Vp=malloc((size_t)L*HD);
        for(int e=0;e<HD;e++)for(int j=0;j<L;j++) KTp[(size_t)e*L+j]=(int8_t)lrintf(Kf[h][(size_t)j*HD+e]*ksv[h]);
        for(size_t i=0;i<(size_t)L*HD;i++) Vp[i]=(int8_t)lrintf(Vf[h][i]*vsv[h]);
        wkt[h]=ork_i8_mm_pack(c,Kp,L,KTp); wv[h]=ork_i8_mm_pack(c,L,HD,Vp); ktq[h]=KTp; vq[h]=Vp;
        if(!wkt[h]||!wv[h]){ printf("pack fail h=%d\n",h); return 2; }
    }
    int8_t *w8=malloc((size_t)L); int32_t *scores=malloc((size_t)L*4),*attv=malloc((size_t)HD*4);
    double *sc=malloc((size_t)L*sizeof(double)); float *att=malloc((size_t)HD*4),*catt=malloc((size_t)HD*4);

    /* one head's attention using the RESIDENT packed weights (softmax on host) */
    #define RUN_HEAD(h) do{ \
        ork_mm_task_i8 t1={ wkt[h],1,Q8[h],scores }; if(ork_i8_mm_run_chain(c,1,&t1)) return 1; \
        double mx=-1e300; for(int j=0;j<L;j++){ sc[j]=(double)scores[j]/((double)qsv[h]*ksv[h])*scale; if(sc[j]>mx)mx=sc[j]; } \
        double Z=0; for(int j=0;j<L;j++){ sc[j]=exp(sc[j]-mx); Z+=sc[j]; } if(Z<=0)Z=1; \
        double wmax=0; for(int j=0;j<L;j++){ sc[j]/=Z; if(sc[j]>wmax)wmax=sc[j]; } double ws=127.0/(wmax>1e-9?wmax:1.0); \
        for(int j=0;j<L;j++){ int wi=(int)lrint(sc[j]*ws); w8[j]=(int8_t)(wi>127?127:(wi<0?0:wi)); } \
        ork_mm_task_i8 t2={ wv[h],1,w8,attv }; if(ork_i8_mm_run_chain(c,1,&t2)) return 1; \
        for(int e=0;e<HD;e++) att[e]=(float)((double)attv[e]/(ws*vsv[h])); }while(0)

    /* coherence (resident path) vs pure-fp CPU */
    double worst=0;
    for(int h=0;h<NH;h++){ RUN_HEAD(h);
        double mx=-1e300; for(int j=0;j<L;j++){ double s=0; for(int e=0;e<HD;e++)s+=Qf[h][e]*Kf[h][(size_t)j*HD+e]; sc[j]=s*scale; if(sc[j]>mx)mx=sc[j]; }
        double Z=0; for(int j=0;j<L;j++){ sc[j]=exp(sc[j]-mx); Z+=sc[j]; }
        for(int e=0;e<HD;e++){ double a=0; for(int j=0;j<L;j++)a+=sc[j]*Vf[h][(size_t)j*HD+e]; catt[e]=(float)(a/Z); }
        double rm=0,er=0; for(int e=0;e<HD;e++){ if(fabsf(catt[e])>rm)rm=fabsf(catt[e]); double d=fabs(att[e]-catt[e]); if(d>er)er=d; }
        double rel=er/(rm>1e-6?rm:1); if(rel>worst)worst=rel; }
    printf("  coherence NPU(host-split) vs CPU: worst rel-err=%.4f %s\n", worst, worst<0.08?"COHERENT":"CHECK");

    /* timed: RESIDENT K/V (pack-once; per-step = matmuls only — the Tier 12f target) */
    double t0=now_ms();
    for(int r=0;r<REP;r++) for(int h=0;h<NH;h++) RUN_HEAD(h);
    double res_ms=(now_ms()-t0)/REP;
    /* timed: PER-STEP PACK — repack K^T/V every token, which is what decode must do while the KV cache
     * grows. This is the number that decides whether the chained decode shape is reachable at all: the
     * chain needs K^T and V as packed WEIGHT cubes, and the cache gains a row per token. */
    t0=now_ms();
    for(int r=0;r<REP;r++) for(int h=0;h<NH;h++){
        ork_w *pk=ork_i8_mm_pack(c,Kp,L,ktq[h]), *pv=ork_i8_mm_pack(c,L,HD,vq[h]);
        if(!pk||!pv){ printf("  per-step pack: FAILED to pack\n"); return 1; }
        ork_w_free(pk); ork_w_free(pv);
    }
    double pack_ms=(now_ms()-t0)/REP;

    /* timed: RESIDENT-KV APPEND — the real per-token cost of keeping K/V packed (Tier 12f). The resident
     * timing above assumes the append is free; this measures it. One key column + one value row per head. */
    double app_ms = -1.0;
    if (L % 32 == 0 && HD % 32 == 0) {
        ork_kv_resident **kv = calloc(NH, sizeof *kv);
        int ok = 1;
        for (int h = 0; h < NH && ok; h++) { kv[h] = ork_kv_resident_alloc(c, HD, L); if (!kv[h]) ok = 0; }
        if (ok) {
            int8_t *kcol = malloc((size_t)HD), *vrow = malloc((size_t)HD);
            for (int e = 0; e < HD; e++) { kcol[e] = (int8_t)(e & 31); vrow[e] = (int8_t)(e & 15); }
            /* warm: fill each cache once, then time appending ONE key per head (a decode step) */
            for (int h = 0; h < NH; h++) for (int k = 0; k < L; k++) ork_kv_append(c, kv[h], k, kcol, vrow);
            t0 = now_ms();
            for (int r = 0; r < REP; r++) for (int h = 0; h < NH; h++)
                if (ork_kv_append(c, kv[h], (L - 1), kcol, vrow)) { printf("  append failed\n"); ok = 0; }
            app_ms = (now_ms() - t0) / REP;
            free(kcol); free(vrow);
        } else printf("  resident alloc refused (Lmax>nmax or align) — append not timed\n");
        for (int h = 0; h < NH; h++) if (kv[h]) ork_kv_resident_free(c, kv[h]);
        free(kv);
    }

    /* timed: FUSED CHAIN — [QK^T -> exp -> reduce, e.V] as ONE submit per head, all heads fanned
     * round-robin across the 3 cores in one dispatch. This is the shape under discussion. It removes the
     * host softmax round-trip the resident path above still pays (read [1,L] scores back, softmax in fp,
     * requantise, submit A.V) — 2 submits + host work per head becomes 1 chained submit per head.
     * Calibration mirrors chainrr_biased_probe: r_mult targets int8 ~110 over the global raw range and
     * max_bias is the global score max, so exp args stay <=0 and the bias cancels in av/Sigma. */
    double fused_ms = -1.0, fused_err = -1.0;
    {
        /* raw scores per head, to calibrate r_mult / max_bias the way a real layer would */
        long maxabs = 1;
        int32_t **raw = calloc(NH, sizeof *raw);
        for (int h = 0; h < NH; h++) { raw[h] = calloc((size_t)L, 4);
            for (int j = 0; j < L; j++) { long a = 0;
                for (int e = 0; e < HD; e++) a += (long)Q8[h][e] * ktq[h][(size_t)e*L + j];
                raw[h][j] = (int32_t)a; if (labs(a) > maxabs) maxabs = labs(a); } }
        int r_shift = 16, r_mult = (int)(((long)110 << r_shift)/maxabs); if (r_mult < 1) r_mult = 1;
        long smax = -1000000;
        for (int h = 0; h < NH; h++) for (int j = 0; j < L; j++) {
            long sq = ((long)raw[h][j]*r_mult) >> r_shift; if (sq > 127) sq = 127; if (sq < -128) sq = -128;
            if (sq > smax) smax = sq; }
        double in_scale = 0.03125, out_scale = 1.0/127.0, max_bias = (double)smax;

        int8_t *ones = malloc((size_t)L*32); memset(ones, 1, (size_t)L*32);
        ork_w *w_ones = ork_i8_mm_pack(c, L, 32, ones); free(ones);
        ork_chain_op ops[4] = { {1,-1,0,r_mult,r_shift}, {2,0,0,0,0}, {0,1,0,0,0}, {0,1,0,0,0} };
        ork_mm_task_i8 **chains = calloc(NH, sizeof *chains); int *Sv = calloc(NH, sizeof *Sv);
        int32_t **scb = calloc(NH,sizeof*scb), **eb = calloc(NH,sizeof*eb),
                **ssb = calloc(NH,sizeof*ssb), **avb = calloc(NH,sizeof*avb);
        for (int h = 0; h < NH; h++) {
            scb[h]=calloc((size_t)L,4); eb[h]=calloc((size_t)L,4);
            ssb[h]=calloc(32,4);        avb[h]=calloc((size_t)HD,4);
            chains[h] = malloc(4*sizeof(ork_mm_task_i8)); Sv[h] = 4;
            chains[h][0] = (ork_mm_task_i8){ wkt[h], 1, Q8[h],            scb[h] };
            chains[h][1] = (ork_mm_task_i8){ wkt[h], 1, (int8_t*)scb[h],  eb[h]  };
            chains[h][2] = (ork_mm_task_i8){ w_ones, 1, (int8_t*)eb[h],   ssb[h] };
            chains[h][3] = (ork_mm_task_i8){ wv[h],  1, (int8_t*)eb[h],   avb[h] };
        }
        if (w_ones) {
            int rc = ork_mm_run_chains_rr_biased(c, NH, (const ork_mm_task_i8*const*)chains, Sv, ops,
                                                 in_scale, out_scale, max_bias);   /* warm */
            if (rc) printf("  fused chain rr rc=%d (not timed)\n", rc);
            else {
                t0 = now_ms();
                for (int r = 0; r < REP; r++)
                    if (ork_mm_run_chains_rr_biased(c, NH, (const ork_mm_task_i8*const*)chains, Sv, ops,
                                                    in_scale, out_scale, max_bias)) { rc = -9; break; }
                if (!rc) {
                    fused_ms = (now_ms()-t0)/REP;
                    /* NO coherence check here on purpose. An earlier version of this probe
                     * reported rel-err 0.33-0.55 for this path, which was WRONG -- its r_mult/in_scale
                     * calibration omitted the 1/sqrt(HD) score scaling, so it compared against a
                     * softmax at the wrong temperature. The chain itself is coherent at M=1:
                     * chainrr_biased_probe (which calibrates properly) gives max|err| 0.0092 at
                     * Nk=512, 0.0061 at 1024, 0.0053 at 2048 -- BETTER than its Nq=32 case. Use that
                     * probe for accuracy; this one is for TIMING only. */
                    fused_err = -1.0;
                }
            }
        } else printf("  fused chain: w_ones pack failed\n");
        for (int h = 0; h < NH; h++) { free(scb[h]); free(eb[h]); free(ssb[h]); free(avb[h]); free(chains[h]); free(raw[h]); }
        free(scb); free(eb); free(ssb); free(avb); free(chains); free(Sv); free(raw);
        if (w_ones) ork_mm_free(c, w_ones);
    }

    /* timed: CPU fp32 SIMD-shaped attention — the HONEST baseline. The double-precision scalar loop below
     * is not what ggml runs; comparing against it flatters the NPU by several times. Same math in float,
     * restructured so the compiler can vectorise both halves: QK^T as HD-length dots, A.V as an AXPY over
     * the HD dimension (contiguous in V) rather than a strided inner sum. */
    float *acc = malloc((size_t)HD*4), *scf = malloc((size_t)L*4);
    t0 = now_ms();
    for (int r = 0; r < REP; r++) for (int h = 0; h < NH; h++) {
        const float *q = Qf[h], *Kh = Kf[h], *Vh = Vf[h];
        float mx = -3e38f;
        for (int j = 0; j < L; j++) {
            const float *kj = Kh + (size_t)j*HD; float sum = 0.f;
            for (int e = 0; e < HD; e++) sum += q[e]*kj[e];
            scf[j] = sum*scale; if (scf[j] > mx) mx = scf[j];
        }
        float Z = 0.f;
        for (int j = 0; j < L; j++) { scf[j] = expf(scf[j]-mx); Z += scf[j]; }
        for (int e = 0; e < HD; e++) acc[e] = 0.f;
        for (int j = 0; j < L; j++) { const float *vj = Vh + (size_t)j*HD, w = scf[j];
            for (int e = 0; e < HD; e++) acc[e] += w*vj[e]; }
        float inv = 1.f/Z; for (int e = 0; e < HD; e++) catt[e] = acc[e]*inv;
    }
    double cpu32_ms = (now_ms()-t0)/REP;
    free(acc); free(scf);

    /* timed: CPU fp softmax attention */
    t0=now_ms();
    for(int r=0;r<REP;r++) for(int h=0;h<NH;h++){ double mx=-1e300;
        for(int j=0;j<L;j++){ double s=0; for(int e=0;e<HD;e++)s+=Qf[h][e]*Kf[h][(size_t)j*HD+e]; sc[j]=s*scale; if(sc[j]>mx)mx=sc[j]; }
        double Z=0; for(int j=0;j<L;j++){ sc[j]=exp(sc[j]-mx); Z+=sc[j]; }
        for(int e=0;e<HD;e++){ double a=0; for(int j=0;j<L;j++)a+=sc[j]*Vf[h][(size_t)j*HD+e]; catt[e]=(float)(a/Z); } }
    double cpu_ms=(now_ms()-t0)/REP;

    printf("  CPU naive f64 scalar    : %8.3f ms/token  (%d heads) -- NOT a fair baseline\n", cpu_ms, NH);
    printf("  CPU fp32 vectorisable   : %8.3f ms/token  (%d heads) <== the honest baseline\n", cpu32_ms, NH);
    printf("  NPU resident-K/V (12f)  : %8.3f ms/token  (pack-once; matmuls only)\n", res_ms);
    printf("  NPU per-step repack ONLY: %8.3f ms/token  (K^T+V repack, no matmuls)\n", pack_ms);
    printf("  NPU per-step total      : %8.3f ms/token  (repack + matmuls)\n", pack_ms+res_ms);
    printf("  speedup resident vs CPU : %.2fx  %s\n", cpu_ms/res_ms, res_ms<cpu_ms?"(NPU wins)":"(CPU wins)");
    printf("  speedup per-step vs CPU : %.2fx  %s\n", cpu_ms/(pack_ms+res_ms), (pack_ms+res_ms)<cpu_ms?"(NPU wins)":"(CPU wins)");
    printf("  repack is %.0f%% of the per-step cost\n", 100.0*pack_ms/(pack_ms+res_ms));
    if (app_ms >= 0) {
        printf("  --- Tier 12f: resident + APPEND (the reachable path) ---\n");
        printf("  vs fp32 CPU baseline    : %.2fx  %s\n", cpu32_ms/(app_ms+res_ms),
               (app_ms+res_ms) < cpu32_ms ? "(NPU wins)" : "(CPU wins)");
        printf("  NPU append ONLY         : %8.3f ms/token  (%d heads, 1 key each)\n", app_ms, NH);
        printf("  NPU resident+append     : %8.3f ms/token  (append + matmuls)\n", app_ms + res_ms);
        printf("  speedup vs CPU          : %.2fx  %s\n", cpu_ms/(app_ms+res_ms),
               (app_ms+res_ms) < cpu_ms ? "(NPU wins)" : "(CPU wins)");
        if (fused_ms > 0) {
            printf("  --- FUSED CHAIN [QK^T->exp->reduce,e.V], all heads RR in one dispatch ---\n");
            printf("  NPU fused chain         : %8.3f ms/token  (timing only; accuracy: see chainrr_biased_probe)\n",
                   fused_ms);
            printf("  fused+append            : %8.3f ms/token\n", fused_ms + app_ms);
            printf("  fused vs fp32 CPU       : %.2fx  %s\n", cpu32_ms/(fused_ms+app_ms),
                   (fused_ms+app_ms) < cpu32_ms ? "(NPU wins)" : "(CPU wins)");
            printf("  fused vs host-split     : %.2fx\n", (res_ms+app_ms)/(fused_ms+app_ms));
        }
        printf("  append is %.1f%% of resident cost; repack was %.0fx the append\n",
               100.0*app_ms/(app_ms+res_ms), app_ms > 1e-9 ? pack_ms/app_ms : 0.0);
    }
    printf("%s\n", worst<0.08?"PASS — host-split decode attn coherent (arbitrary scores); resident-K/V perf model":"FAIL");
    for(int h=0;h<NH;h++){ ork_w_free(wkt[h]); ork_w_free(wv[h]); }
    ork_npu_free(c);
    return worst<0.08?0:1;
}
