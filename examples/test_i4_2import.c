/* test_i4_2import.c — MINIMAL, ZERO accumulated state: does the 2nd IMPORTED int4 weight in a NON-0 domain
 * run bit-exact (BCHAIN, M>=2 prefill)? This is the ONE open int4-MoE bug (int8 reads many imports/domain
 * bit-exact; int4's run reads the 2nd imported buffer in a non-0 domain wrong). No other tests, no domain
 * churn — just: init -> alloc ONE non-0 domain -> import TWO experts -> BCHAIN-run each -> verify.
 *
 *   make test_i4_2import && sudo env ORK_MM_TIMEOUT=2500 timeout 120 ./test_i4_2import
 *
 * PASS => the 2nd imported buffer in a non-0 domain is fine (bug is elsewhere / arena-specific).
 * FAIL on expert 2 (expert 1 OK) => reproduced: the 2nd imported buffer in a non-0 domain reads wrong.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ork_npu.h"

static void fill_i4(int8_t *p, size_t n, unsigned seed){
    unsigned sd = seed;
    for (size_t i = 0; i < n; i++){ sd = sd*1103515245u + 12345u; p[i] = (int8_t)((int)((sd>>17)%15) - 7); }
}
static long verify(const int8_t *A, const int8_t *B, const int32_t *C, int M, int K, int N){
    long maxe = 0;
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++){
        long s = 0; for (int k = 0; k < K; k++) s += (long)A[(size_t)m*K+k]*B[(size_t)k*N+n];
        long e = C[(size_t)m*N+n] - s; if (e < 0) e = -e; if (e > maxe) maxe = e;
    }
    return maxe;
}
/* import one expert into domain `dom`: pack in dom0 -> dump native int4 -> per-expert import into `dom`. */
static ork_w *imp_one(ork_npu *c, int dom, int K, int N, int8_t *B, unsigned seed){
    fill_i4(B, (size_t)K*N, seed);
    ork_npu_set_pack_domain(c, 0);
    ork_w *w0 = ork_mm_pack_i4(c, K, N, B); if (!w0) return NULL;
    size_t tb = ork_w_dump(w0, NULL, 0); char *blob = malloc(tb); ork_w_dump(w0, blob, tb); ork_mm_free(c, w0);
    ork_npu_set_pack_domain(c, dom);
    ork_w *w = ork_mm_load_i4_import(c, K, N, blob, tb); free(blob); return w;
}

int main(void){
    ork_npu *c = ork_npu_init(); if (!c){ printf("init failed (NPU?)\n"); return 1; }
    const int K = 512, N = 2048, M = 96;                 /* the 35B ffn_down prefill expert shape */
    int d = ork_npu_domain_alloc(c);
    if (d <= 0){ printf("domain_alloc=%d — cannot test a non-0 domain\n", d); ork_npu_free(c); return 1; }
    ork_npu_activate_domain(c, d);                        /* establish the domain (native anchor) */
    printf("== 2nd-import-in-non-0-domain test: domain %d, K=%d N=%d M=%d ==\n", d, K, N, M);

    int8_t *A  = malloc((size_t)M*K); fill_i4(A, (size_t)M*K, 5);
    int8_t *B1 = malloc((size_t)K*N), *B2 = malloc((size_t)K*N);
    int32_t *C = malloc((size_t)M*N*4);
    ork_w *w1 = imp_one(c, d, K, N, B1, 101);            /* 1st import into domain d */
    if (!w1){ printf("import 1 FAILED (NPU/IOMMU wedged? reboot needed)\n"); ork_npu_free(c); return 1; }
    ork_w *w2 = imp_one(c, d, K, N, B2, 202);            /* 2nd import into the SAME domain d */
    if (!w2){ printf("import 2 FAILED\n"); ork_mm_free(c, w1); ork_npu_free(c); return 1; }

    int  r1 = ork_mm_run_i4(c, w1, M, A, C); long e1 = r1 ? -1 : verify(A, B1, C, M, K, N);
    printf("  expert 1 (1st import): rc=%d maxerr=%-5ld %s\n", r1, e1, (r1==0 && e1==0) ? "OK" : "FAIL");
    int  r2 = ork_mm_run_i4(c, w2, M, A, C); long e2 = r2 ? -1 : verify(A, B2, C, M, K, N);
    printf("  expert 2 (2nd import): rc=%d maxerr=%-5ld %s\n", r2, e2, (r2==0 && e2==0) ? "OK" : "FAIL");
    /* re-run expert 1 AFTER expert 2 — did the 2nd import corrupt the 1st? */
    int  r3 = ork_mm_run_i4(c, w1, M, A, C); long e3 = r3 ? -1 : verify(A, B1, C, M, K, N);
    printf("  expert 1 (re-run after 2): rc=%d maxerr=%-5ld %s\n", r3, e3, (r3==0 && e3==0) ? "OK" : "FAIL");

    /* #54 COALESCED doorbell: run BOTH experts through ONE ork_mm_run_i4_experts call (rows chained across
     * cores in one submit-set) — both must be bit-exact. */
    int32_t *Cc1 = malloc((size_t)M*N*4), *Cc2 = malloc((size_t)M*N*4);
    ork_mm_task_i4 ex[2] = { { w1, M, A, Cc1 }, { w2, M, A, Cc2 } };
    int rc = ork_mm_run_i4_experts(c, ex, 2, 3);
    long ec1 = rc ? -1 : verify(A, B1, Cc1, M, K, N);
    long ec2 = rc ? -1 : verify(A, B2, Cc2, M, K, N);
    printf("  COALESCED 2-expert doorbell: rc=%d maxerr1=%-5ld maxerr2=%-5ld %s\n", rc, ec1, ec2,
           (rc==0 && ec1==0 && ec2==0) ? "OK" : "FAIL");

    /* #54 COALESCED with MORE experts than cores (5 experts, 3 cores -> 2+2+1/core): exercises multiple experts
     * chained on ONE core (cumulative tk + per-expert de-tile). */
    int bad_many = 0;
    { const int NE = 5;
      ork_w *we[NE]; int8_t *Be[NE]; int32_t *Ce[NE]; ork_mm_task_i4 exm[NE];
      for (int q = 0; q < NE; q++){ Be[q] = malloc((size_t)K*N); we[q] = imp_one(c, d, K, N, Be[q], 300+q);
          Ce[q] = malloc((size_t)M*N*4); exm[q] = (ork_mm_task_i4){ we[q], M, A, Ce[q] }; }
      int rcm = 1; for (int q=0;q<NE;q++) if(!we[q]){ printf("  [coal-many] import %d FAILED\n", q); goto coal_done; }
      rcm = ork_mm_run_i4_experts(c, exm, NE, 3);
      long worst = 0; for (int q = 0; q < NE; q++){ long e = rcm ? -1 : verify(A, Be[q], Ce[q], M, K, N); if (e>worst) worst = e; }
      printf("  COALESCED %d-expert (2+/core): rc=%d worst_maxerr=%ld %s\n", NE, rcm, worst, (rcm==0&&worst==0)?"OK":"FAIL");
      bad_many = (rcm != 0 || worst != 0);
      coal_done: for (int q=0;q<NE;q++){ if(we[q]) ork_mm_free(c,we[q]); free(Be[q]); free(Ce[q]); } }

    /* #54 MULTI-DOMAIN RESIDENT: reproduce the 35B regime — experts packed RESIDENT across SEVERAL domains,
     * a coalesced run PER domain in ascending order. The 35B hung (D-state) at the 3rd domain (dom=2); dom0/dom1
     * ran fine. This walks dom = d, d+1, d+2, ... allocating + importing + coalesced-running in each, ALL kept
     * resident (no free), to catch the transition/accumulation hang cheaply (seconds) instead of on the 35B. */
    int bad_md = 0;
    { const int NDOM = 5, NPE = 4;                     /* 5 domains x 4 experts each, all resident */
      ork_w   *wd[NDOM][NPE]; int8_t *Bd[NDOM][NPE]; int32_t *Cd[NDOM][NPE];
      int      dom_id[NDOM]; ork_mm_task_i4 exd[NPE];
      dom_id[0] = d;                                   /* reuse the already-allocated domain d as the 1st */
      for (int q = 0; q < NPE; q++){ Bd[0][q]=malloc((size_t)K*N); wd[0][q]=imp_one(c,d,K,N,Bd[0][q],700+q);
          Cd[0][q]=malloc((size_t)M*N*4); }
      for (int dd = 1; dd < NDOM; dd++){               /* allocate additional domains */
          dom_id[dd] = ork_npu_domain_alloc(c);
          if (dom_id[dd] <= 0){ printf("  [multidom] domain_alloc #%d FAILED (only %d domains) — stopping\n", dd, dd);
              for (int k=dd;k<NDOM;k++) dom_id[k]=-1; break; }
          ork_npu_activate_domain(c, dom_id[dd]);
          for (int q = 0; q < NPE; q++){ Bd[dd][q]=malloc((size_t)K*N); wd[dd][q]=imp_one(c,dom_id[dd],K,N,Bd[dd][q],700+dd*10+q);
              Cd[dd][q]=malloc((size_t)M*N*4); }
      }
      for (int dd = 0; dd < NDOM && dom_id[dd] > 0; dd++){   /* coalesced run in EACH domain, ascending */
          int okd = 1; for (int q=0;q<NPE;q++){ if(!wd[dd][q]){ okd=0; break; } exd[q]=(ork_mm_task_i4){wd[dd][q],M,A,Cd[dd][q]}; }
          if (!okd){ printf("  [multidom] dom_slot %d import FAILED\n", dd); bad_md=1; break; }
          int rcd = ork_mm_run_i4_experts(c, exd, NPE, 3);
          long worst = 0; for (int q=0;q<NPE;q++){ long e=rcd?-1:verify(A,Bd[dd][q],Cd[dd][q],M,K,N); if(e>worst)worst=e; }
          printf("  [multidom] domain %d (slot %d): rc=%d worst_maxerr=%ld %s\n", dom_id[dd], dd, rcd, worst, (rcd==0&&worst==0)?"OK":"FAIL");
          if (rcd!=0 || worst!=0) bad_md=1;
      }
      for (int dd=0; dd<NDOM; dd++) if(dom_id[dd]>0) for (int q=0;q<NPE;q++){ if(wd[dd][q]) ork_mm_free(c,wd[dd][q]); free(Bd[dd][q]); free(Cd[dd][q]); }
    }

    int bad = (r1||e1||r2||e2||r3||e3||rc||ec1||ec2||bad_many||bad_md);
    printf("%s\n", bad ? "FAIL: imported int4 weights in a non-0 domain are NOT all bit-exact"
                       : "PASS: 1st AND 2nd imported int4 weights in a non-0 domain are bit-exact");
    ork_mm_free(c, w1); ork_mm_free(c, w2); free(A); free(B1); free(B2); free(C);
    ork_npu_free(c); return bad ? 1 : 0;
}
