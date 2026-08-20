/* test_moe_dispatch.c — Stage-0 pipeline-tax probe (make-or-break for the grouped/block-sparse MoE plan).
 *
 * The NPU per-expert dispatch floor is ~18us/program (regcmd decode + MAC config + SRAM fill/drain). The
 * whole "dense-ify the MoE" direction hinges on ONE question: does that per-program tax AMORTIZE when many
 * experts are submitted as a chain / a grouped call / one big program — or is it paid N times no matter what?
 *
 *   DECODE regime  (M=1, NE=8 = top-8):  N separate ork_i4_mm_run  vs  ork_i4_mm_run_chain (HW PC-chain)
 *                                        vs  ork_i4_dyn_probe (nonblock doorbell chain).
 *   PREFILL regime (M=16, NE=32):        N separate  vs  ork_i4_mm_run_experts nc=1 / nc=3 (grouped/coalesce).
 *   CEILING:                             ONE ork_i4_mm_run of M=NE*Me rows vs a single weight — the
 *                                        amortization lower-bound (1 dispatch, all the rows).
 *
 * READING IT: if a chained/grouped call ~= N * (separate per-op), the tax is paid per program and does NOT
 * amortize with the existing mechanisms => a grouped GEMM needs a genuine single-MAC-config reformulation
 * (block-diagonal weights) — hard; reassess. If the CEILING (one big program) is << N*separate, the tax IS
 * amortizable in principle; whichever of chain/grouped approaches the ceiling is the mechanism to build on.
 * DECODE is the highest-value target (native ~6 t/s): run_chain_i4/dyn beating N-separate at M=1 is the
 * green light for the "concatenate top-8 -> one grouped submit" decode win.
 *
 *   make test_moe_dispatch && sudo ./test_moe_dispatch
 * No model, no NPU-output golden — pure dispatch timing + a bit-exact spot check. Runs in ~1s.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static void fill_i4(int8_t*p,size_t n,unsigned s){ for(size_t i=0;i<n;i++){ s=s*1103515245u+12345u; p[i]=(int8_t)((int)((s>>17)%15)-7); } }
static long verify(const int8_t*A,const int8_t*B,const int32_t*C,int M,int K,int N){ long e=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[(size_t)m*K+k]*B[(size_t)k*N+n];
        long d=C[(size_t)m*N+n]-s; if(d<0)d=-d; if(d>e)e=d; } return e; }

/* best-of-R wall for a thunk over `reps` inner iters */
#define BEST(dst, reps, body) do{ double b=1e18; for(int r=0;r<5;r++){ double t0=now_us(); \
    for(int it=0;it<(reps);it++){ body; } double u=(now_us()-t0)/(reps); if(u<b)b=u; } dst=b; }while(0)

int main(void){
    const int K=2048, N=512;                  /* qwen3.6 gate/up expert shape */
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }

    /* ---- pack a pool of distinct expert weights (max NE across regimes) ---- */
    const int NEMAX=32;
    ork_w   **W = calloc(NEMAX,sizeof*W);
    int8_t  **B = calloc(NEMAX,sizeof*B);
    for(int e=0;e<NEMAX;e++){ B[e]=malloc((size_t)K*N); fill_i4(B[e],(size_t)K*N,23+e);
        W[e]=ork_i4_mm_pack(c,K,N,B[e]); if(!W[e]){ printf("pack %d fail\n",e); return 1; } }

    /* ================= DECODE regime: M=1, NE=8 (top-8) ================= */
    {
        const int NE=8, M=1;
        int8_t **A=calloc(NE,sizeof*A); int32_t **C=calloc(NE,sizeof*C);
        ork_mm_task_i4 *tk=calloc(NE,sizeof*tk);
        for(int e=0;e<NE;e++){ A[e]=malloc((size_t)M*K); C[e]=malloc((size_t)M*N*4);
            fill_i4(A[e],(size_t)M*K,7+e); tk[e]=(ork_mm_task_i4){W[e],M,A[e],C[e]}; }

        double sep=0, chain=0, dyn=0;
        for(int e=0;e<NE;e++) ork_i4_mm_run(c,W[e],M,A[e],C[e]);                 /* warm */
        BEST(sep,   8, { for(int e=0;e<NE;e++) ork_i4_mm_run(c,W[e],M,A[e],C[e]); });
        long e_sep = verify(A[0],B[0],C[0],M,K,N);
        int rc_chain = ork_i4_mm_run_chain(c,NE,tk);
        if(rc_chain==0){ BEST(chain, 8, { ork_i4_mm_run_chain(c,NE,tk); }); }
        int rc_dyn = ork_i4_dyn_probe(c,NE,tk);
        if(rc_dyn==0){ BEST(dyn, 8, { ork_i4_dyn_probe(c,NE,tk); }); }

        printf("\n[DECODE  M=1  NE=8  K=%d N=%d]\n",K,N);
        printf("  separate   : %8.1f us  (%.1f us/expert)   %s\n", sep, sep/NE, e_sep==0?"bit-exact":"MISCOMPUTE");
        if(rc_chain==0) printf("  run_chain_i4: %8.1f us  (%.1f us/expert)   %.2fx vs separate  %s\n",
            chain, chain/NE, sep/chain, sep/chain>=1.5?"<-- AMORTIZES":"(flat)");
        else            printf("  run_chain_i4: unavailable (rc=%d)\n", rc_chain);
        if(rc_dyn==0)   printf("  dyn_i4_probe: %8.1f us  (%.1f us/expert)   %.2fx vs separate  %s\n",
            dyn, dyn/NE, sep/dyn, sep/dyn>=1.5?"<-- AMORTIZES":"(flat)");
        else            printf("  dyn_i4_probe: unavailable (rc=%d)\n", rc_dyn);
    }

    /* ================= PREFILL regime: M=16, NE=32 ================= */
    {
        const int NE=32, M=16;                 /* M_e ~= ubatch*topk/n_expert = 512*8/256 = 16 at pp512 */
        int8_t **A=calloc(NE,sizeof*A); int32_t **C=calloc(NE,sizeof*C);
        ork_mm_task_i4 *ex=calloc(NE,sizeof*ex);
        for(int e=0;e<NE;e++){ A[e]=malloc((size_t)M*K); C[e]=malloc((size_t)M*N*4);
            fill_i4(A[e],(size_t)M*K,7+e); ex[e]=(ork_mm_task_i4){W[e],M,A[e],C[e]}; }

        double sep=0, g1=0, g3=0;
        for(int e=0;e<NE;e++) ork_i4_mm_run(c,W[e],M,A[e],C[e]);                 /* warm */
        BEST(sep, 4, { for(int e=0;e<NE;e++) ork_i4_mm_run(c,W[e],M,A[e],C[e]); });
        long e_sep = verify(A[0],B[0],C[0],M,K,N);
        int r1 = ork_i4_mm_run_experts(c,ex,NE,1); if(r1==0) BEST(g1,4,{ ork_i4_mm_run_experts(c,ex,NE,1); });
        int r3 = ork_i4_mm_run_experts(c,ex,NE,3); if(r3==0) BEST(g3,4,{ ork_i4_mm_run_experts(c,ex,NE,3); });
        long e_grp = verify(A[0],B[0],ex[0].C,M,K,N);

        printf("\n[PREFILL M=16 NE=32 K=%d N=%d]\n",K,N);
        printf("  separate       : %8.1f us  (%.1f us/expert)   %s\n", sep, sep/NE, e_sep==0?"bit-exact":"MISCOMPUTE");
        if(r1==0) printf("  experts nc=1   : %8.1f us  (%.1f us/expert)   %.2fx vs separate\n", g1, g1/NE, sep/g1);
        if(r3==0) printf("  experts nc=3   : %8.1f us  (%.1f us/expert)   %.2fx vs separate  %s\n",
            g3, g3/NE, sep/g3, sep/g3>=1.5?"<-- AMORTIZES/PARALLELIZES":"(flat)");
        printf("  grouped output : %s\n", e_grp==0?"bit-exact":"MISCOMPUTE");

        /* CEILING: one program, M=NE*M rows, single weight — the amortization lower-bound */
        int Mbig=NE*M; int8_t *Ab=malloc((size_t)Mbig*K); int32_t *Cb=malloc((size_t)Mbig*N*4);
        fill_i4(Ab,(size_t)Mbig*K,99);
        ork_i4_mm_run(c,W[0],Mbig,Ab,Cb);                                        /* warm */
        double big=0; BEST(big, 4, { ork_i4_mm_run(c,W[0],Mbig,Ab,Cb); });
        printf("  CEILING 1x(M=%d): %8.1f us  (one dispatch, all %d rows) — floor if the tax fully amortized\n",
            Mbig, big, Mbig);
        printf("  => per-PROGRAM tax ~= (separate - ceiling*NE-scaled); if grouped -> ceiling, build it; if grouped ~ separate, need block-diagonal.\n");
        free(Ab); free(Cb);
    }

    printf("\n---- Stage-0 verdict: compare each mechanism's us/expert to 'separate'. >=1.5x => the per-program\n");
    printf("     dispatch tax amortizes with that mechanism (green light). ~1x everywhere => tax is per-program\n");
    printf("     inherent; a grouped MoE GEMM needs a single-MAC-config (block-diagonal) reformulation. ----\n");
    ork_npu_free(c);
    return 0;
}
