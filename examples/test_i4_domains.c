/* examples/test_i4_domains.c — reproduce (and gate the fix for) the int4 doorbell submit in a NON-0 IOMMU domain.
 *
 * The 35B MoE W4A4 run stalls: "run_i4_bchain_db incomplete (recover exhausted), hw_elapse=0 (0=>no work),
 * STUCK op#0". Root theory: resident int4 weights live in NON-0 domains (multi-domain residence >4 GiB), and
 * the int4 doorbell submit (run_i4_bchain_db / run_i4_mc_db) does not land there — it submits against the stale/
 * wrong iommu_domain_id. This isolates it WITHOUT the 35B (seconds, not 30-min cycles): pack/import a small int4
 * weight into domain 0 (baseline, must pass) vs an alloc'd non-0 domain (the suspect), run both the M>1 BCHAIN
 * path and the M=1 per-row path, and verify bit-exact vs a CPU int4 reference.
 *
 * PASS = every domain (0 and non-0), both pack + import, both M, matches the CPU ref bit-exact.
 * FAIL/STUCK in a non-0 domain (but OK in domain 0) == reproduced the bug; when the doorbell fix lands, this
 * flips to all-OK and becomes the regression gate. Run under timeout — a stuck doorbell exhausts its recover
 * loop and returns nonzero, but bound it anyway:  make test_i4_domains && sudo timeout 180 ./test_i4_domains
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ork_npu.h"

static void fill_i4(int8_t *p, size_t n, unsigned seed){
    unsigned sd = seed;
    for (size_t i = 0; i < n; i++){ sd = sd*1103515245u + 12345u; p[i] = (int8_t)((int)((sd>>17)%15) - 7); }
}
/* verify C == A(int4) x B(int4), return maxabs error (0 == bit-exact). */
static long verify(const int8_t *A, const int8_t *B, const int32_t *C, int M, int K, int N){
    long maxe = 0;
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++){
        long s = 0; for (int k = 0; k < K; k++) s += (long)A[(size_t)m*K+k]*B[(size_t)k*N+n];
        long e = C[(size_t)m*N+n] - s; if (e < 0) e = -e; if (e > maxe) maxe = e;
    }
    return maxe;
}
/* Diagnostic: report the DISTRIBUTION of wrong columns for M=1 (first-bad col, bad-count, contiguous-tail?).
 * If bad cols cluster at high N -> weight/output TAIL unmapped (SG boundary / HW prefetch). If scattered ->
 * wholesale wrong address. Prints a compact map. */
static void verify_map(const int8_t *A, const int8_t *B, const int32_t *C, int K, int N){
    int first=-1,last=-1,bad=0; long maxe=0;
    for (int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[k]*B[(size_t)k*N+n];
        long e=C[n]-s; if(e<0)e=-e; if(e){ bad++; if(first<0)first=n; last=n; if(e>maxe)maxe=e; } }
    printf("      DIAG: N=%d bad_cols=%d first=%d last=%d maxe=%ld  %s\n", N, bad, first, last, maxe,
           (first>=0 && bad==(last-first+1)) ? "(CONTIGUOUS run)" : (bad?"(scattered)":"(clean)"));
    /* per-64-block bad count (64 = the N-slice/column-split granularity) */
    printf("      per-64-block bad: ");
    for (int b=0;b*64<N;b++){ int bb=0; for(int n=b*64;n<(b+1)*64&&n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[k]*B[(size_t)k*N+n];
        if(C[n]!=s)bb++; } printf("%d ", bb); }
    printf("\n");
}

/* PACK (bcreate) path: pack B into `dom`, run, verify. */
static int run_pack(ork_npu *c, int dom, int M, int K, int N){
    int8_t *A = malloc((size_t)M*K), *B = malloc((size_t)K*N); int32_t *C = malloc((size_t)M*N*4);
    fill_i4(A, (size_t)M*K, 11+dom); fill_i4(B, (size_t)K*N, 23+dom);
    ork_npu_set_pack_domain(c, dom);
    ork_w *w = ork_mm_pack_i4(c, K, N, B);
    if (!w){ printf("  [pack]   dom=%d M=%-3d: PACK FAILED\n", dom, M); free(A);free(B);free(C); return 1; }
    int rc = ork_mm_run_i4(c, w, M, A, C);
    if (rc){ printf("  [pack]   dom=%d M=%-3d: RUN rc=%d (stuck/refused in domain)\n", dom, M, rc); ork_mm_free(c,w); free(A);free(B);free(C); return 1; }
    long maxe = verify(A, B, C, M, K, N);
    printf("  [pack]   dom=%d M=%-3d K=%d N=%d: maxerr=%-4ld %s\n", dom, M, K, N, maxe, maxe==0?"OK":"FAIL");
    ork_mm_free(c,w); free(A);free(B);free(C); return maxe != 0;
}

/* IMPORT path (what resident MoE experts use): pack in dom0 -> dump -> import into `dom` -> run -> verify. */
static int run_import(ork_npu *c, int dom, int M, int K, int N){
    int8_t *A = malloc((size_t)M*K), *B = malloc((size_t)K*N); int32_t *C = malloc((size_t)M*N*4);
    fill_i4(A, (size_t)M*K, 41+dom); fill_i4(B, (size_t)K*N, 53+dom);
    ork_npu_set_pack_domain(c, 0);
    ork_w *w0 = ork_mm_pack_i4(c, K, N, B);
    if (!w0){ printf("  [import] dom=%d M=%-3d: PACK(dump) FAILED\n", dom, M); free(A);free(B);free(C); return 1; }
    size_t tb = ork_w_dump(w0, NULL, 0);
    char *blob = malloc(tb); ork_w_dump(w0, blob, tb); ork_mm_free(c,w0);
    ork_npu_set_pack_domain(c, dom);
    ork_w *w = ork_mm_load_i4_import(c, K, N, blob, tb);
    if (!w){ printf("  [import] dom=%d M=%-3d: IMPORT FAILED (n=%zu)\n", dom, M, tb); free(blob);free(A);free(B);free(C); return 1; }
    int rc = ork_mm_run_i4(c, w, M, A, C);
    if (rc){ printf("  [import] dom=%d M=%-3d: RUN rc=%d (stuck/refused in domain)\n", dom, M, rc); ork_mm_free(c,w); free(blob);free(A);free(B);free(C); return 1; }
    long maxe = verify(A, B, C, M, K, N);
    printf("  [import] dom=%d M=%-3d K=%d N=%d: maxerr=%-4ld %s\n", dom, M, K, N, maxe, maxe==0?"OK":"FAIL");
    if (maxe && M==1) verify_map(A, B, C, K, N);
    ork_mm_free(c,w); free(blob);free(A);free(B);free(C); return maxe != 0;
}

/* ALTERNATE path (the 35B forward pattern): two weights resident in two DIFFERENT established domains; submit to
 * one then IMMEDIATELY the other, many times. Each ork_mm_run_i4 dom_activate()s the weight's domain at its top —
 * so the domain switch lands right after the prior op's NONBLOCK doorbell output merely LANDED (compute done), but
 * possibly BEFORE its kernel job RETIRED. Rapid A->B->A->B tightens that window to nothing. The existing per-domain
 * tests switch domains only ~3× with lots of work between (they pass); this is the pattern that stalls the 35B. */
static int run_alternate(ork_npu *c, int dA, int dB, int M, int iters){
    const int K = 2048, N = 512;
    int8_t *A = malloc((size_t)M*K), *Ba = malloc((size_t)K*N), *Bb = malloc((size_t)K*N);
    int32_t *Ca = malloc((size_t)M*N*4), *Cb = malloc((size_t)M*N*4);
    fill_i4(A, (size_t)M*K, 61); fill_i4(Ba, (size_t)K*N, 71); fill_i4(Bb, (size_t)K*N, 83);
    ork_npu_set_pack_domain(c, dA); ork_w *wa = ork_mm_pack_i4(c, K, N, Ba);
    ork_npu_set_pack_domain(c, dB); ork_w *wb = ork_mm_pack_i4(c, K, N, Bb);
    if (!wa || !wb){ printf("  [alt]    dA=%d dB=%d M=%-3d: PACK FAILED\n", dA, dB, M); goto done; }
    long mea = 0, meb = 0;
    for (int it = 0; it < iters; it++){
        int ra = ork_mm_run_i4(c, wa, M, A, Ca);
        if (ra){ printf("  [alt]    dA=%d M=%-3d it=%d: RUN(A) rc=%d  <-- STALLED/refused across swap\n", dA, M, it, ra); goto fail; }
        long e = verify(A, Ba, Ca, M, K, N); if (e > mea) mea = e;
        int rb = ork_mm_run_i4(c, wb, M, A, Cb);
        if (rb){ printf("  [alt]    dB=%d M=%-3d it=%d: RUN(B) rc=%d  <-- STALLED/refused across swap\n", dB, M, it, rb); goto fail; }
        e = verify(A, Bb, Cb, M, K, N); if (e > meb) meb = e;
    }
    printf("  [alt]    dA=%d dB=%d M=%-3d x%d: maxerrA=%-4ld maxerrB=%-4ld %s\n", dA, dB, M, iters, mea, meb, (mea==0&&meb==0)?"OK":"FAIL");
    { int r = (mea!=0||meb!=0); ork_mm_free(c,wa); ork_mm_free(c,wb); free(A);free(Ba);free(Bb);free(Ca);free(Cb); return r; }
fail:
    ork_mm_free(c,wa); ork_mm_free(c,wb);
done:
    free(A);free(Ba);free(Bb);free(Ca);free(Cb); return 1;
}

/* #54 ARENA path: load MANY distinct int4 weights via ork_mm_load_i4_arena into ONE domain (they must share
 * a few big chunks, not one dma-buf each) and verify every one computes bit-exact. Validates the consolidation
 * mechanism (shared-chunk base+offset views) cheaply, well below the ~2340-mappings/domain wedge threshold. */
static int run_arena(ork_npu *c, int dom, int count, int M, int K, int N){
    int bad = 0;
    int8_t *A = malloc((size_t)M*K); fill_i4(A, (size_t)M*K, 7);
    ork_w **ws = calloc(count, sizeof *ws);
    int8_t **Bs = calloc(count, sizeof *Bs);
    for (int i = 0; i < count; i++){
        int8_t *B = malloc((size_t)K*N); fill_i4(B, (size_t)K*N, 1000+i); Bs[i] = B;
        ork_npu_set_pack_domain(c, 0);
        ork_w *w0 = ork_mm_pack_i4(c, K, N, B);
        if (!w0){ printf("  [arena]  pack(dump) %d FAILED\n", i); bad = 1; break; }
        size_t tb = ork_w_dump(w0, NULL, 0); char *blob = malloc(tb); ork_w_dump(w0, blob, tb); ork_mm_free(c, w0);
        ork_npu_set_pack_domain(c, dom);
        ws[i] = ork_mm_load_i4_arena(c, K, N, blob, tb); free(blob);
        if (!ws[i]){ printf("  [arena]  arena-load %d FAILED\n", i); bad = 1; break; }
    }
    int32_t *C = malloc((size_t)M*N*4); long worst = 0;
    for (int i = 0; i < count && !bad; i++){
        if (ork_mm_run_i4(c, ws[i], M, A, C)){ printf("  [arena]  run %d rc!=0\n", i); bad = 1; break; }
        long e = verify(A, Bs[i], C, M, K, N); if (e > worst) worst = e;
        if (e){ printf("  [arena]  weight %d MISCOMPUTE maxe=%ld\n", i, e); bad = 1; }
    }
    printf("  [arena]  %d weights dom=%d M=%d K=%d N=%d: worst_maxerr=%ld %s\n", count, dom, M, K, N, worst, bad?"FAIL":"OK");
    for (int i = 0; i < count; i++){ if (ws[i]) ork_mm_free(c, ws[i]); free(Bs[i]); }
    free(ws); free(Bs); free(A); free(C);
    return bad;
}

/* CONTIG check: load MANY distinct experts as PER-EXPERT contiguous buffers (ork_mm_load_i4_import — one
 * consolidated chunk each, no shared-chunk boundary) into ONE domain, verify each bit-exact. If this passes
 * where run_arena (shared chunks) fails, the shared-chunk dma-buf boundary is the bug and per-expert CONTIG
 * is the fix (the int8 model: one contiguous buffer per weight). */
static int run_import_many(ork_npu *c, int dom, int count, int M, int K, int N){
    int bad = 0;
    int8_t *A = malloc((size_t)M*K); fill_i4(A, (size_t)M*K, 9);
    ork_w **ws = calloc(count, sizeof *ws);
    int8_t **Bs = calloc(count, sizeof *Bs);
    for (int i = 0; i < count; i++){
        int8_t *B = malloc((size_t)K*N); fill_i4(B, (size_t)K*N, 2000+i); Bs[i] = B;
        ork_npu_set_pack_domain(c, 0);
        ork_w *w0 = ork_mm_pack_i4(c, K, N, B);
        if (!w0){ printf("  [imp-many] pack %d FAILED\n", i); bad = 1; break; }
        size_t tb = ork_w_dump(w0, NULL, 0); char *blob = malloc(tb); ork_w_dump(w0, blob, tb); ork_mm_free(c, w0);
        ork_npu_set_pack_domain(c, dom);
        ws[i] = ork_mm_load_i4_import(c, K, N, blob, tb); free(blob);   /* per-expert contiguous buffer */
        if (!ws[i]){ printf("  [imp-many] import %d FAILED\n", i); bad = 1; break; }
    }
    int32_t *C = malloc((size_t)M*N*4); long worst = 0;
    for (int i = 0; i < count && !bad; i++){
        if (ork_mm_run_i4(c, ws[i], M, A, C)){ printf("  [imp-many] run %d rc!=0\n", i); bad = 1; break; }
        long e = verify(A, Bs[i], C, M, K, N); if (e > worst) worst = e;
        if (e){ printf("  [imp-many] weight %d MISCOMPUTE maxe=%ld\n", i, e); bad = 1; }
    }
    printf("  [imp-many] %d per-expert CONTIG imports dom=%d M=%d K=%d N=%d: worst=%ld %s\n", count, dom, M, K, N, worst, bad?"FAIL":"OK");
    for (int i = 0; i < count; i++){ if (ws[i]) ork_mm_free(c, ws[i]); free(Bs[i]); }
    free(ws); free(Bs); free(A); free(C);
    return bad;
}

int main(void){
    ork_npu *c = ork_npu_init(); if (!c){ printf("init failed (NPU?)\n"); return 1; }
    const int K = 2048, N = 512;      /* the failing 35B expert shape (gate/up); small so the CPU ref is cheap */
    int fail = 0;

    printf("== int4 doorbell in DOMAIN 0 (baseline — must pass) ==\n");
    fail |= run_pack  (c, 0, 128, K, N);   /* BCHAIN (M>1 prefill) — the path that stuck on the 35B */
    fail |= run_pack  (c, 0, 1,   K, N);   /* per-row doorbell (decode) */
    fail |= run_import(c, 0, 128, K, N);
    fail |= run_import(c, 0, 1,   K, N);

    printf("== int4 doorbell in NON-0 domains (the suspect) ==\n");
    for (int t = 0; t < 3; t++){
        int d = ork_npu_domain_alloc(c);
        if (d <= 0){ printf("  domain_alloc(#%d) returned %d — cannot test non-0 domains\n", t, d); fail = 1; break; }
        ork_npu_activate_domain(c, d);   /* ESTABLISH the domain (native anchor) BEFORE packing/switching into it —
                                          * the working int8/import path does this via ork_dom_prime; without it the
                                          * kernel IOMMU switch to an unestablished domain times out (probe bug, not HW). */
        printf("  -- allocated + established domain %d --\n", d);
        fail |= run_pack  (c, d, 128, K, N);   /* BCHAIN in a non-0 domain — the exact 35B failure */
        fail |= run_pack  (c, d, 1,   K, N);
        fail |= run_import(c, d, 128, K, N);   /* imported (resident-expert path) in a non-0 domain */
        fail |= run_import(c, d, 1,   K, N);
    }

    printf("== int4 EXACT 35B ffn_down shape K=512 N=2048 M=96 (prefill expert bucket) in non-0 domains ==\n");
    {
        int d = ork_npu_domain_alloc(c);
        if (d > 0) { ork_npu_activate_domain(c, d);
            printf("  -- ffn_down K=512 N=2048 in domain %d --\n", d);
            fail |= run_pack  (c, d, 96, 512, 2048);   /* packed */
            fail |= run_import(c, d, 96, 512, 2048);   /* imported (resident-expert path) — the 35B case */
        } else printf("  domain_alloc returned %d — skip\n", d);
    }

    printf("== CONTIG CHECK: 60 per-expert contiguous imports (int8-style) in ONE non-0 domain ==\n");
    { int d = ork_npu_domain_alloc(c);
      if (d > 0){ ork_npu_activate_domain(c, d); fail |= run_import_many(c, d, 60, 96, 512, 2048); }
      else printf("  domain_alloc=%d skip\n", d); }

    printf("== #54 ARENA in DOMAIN 0 (imports always work here — isolates non-0-domain 2nd-import bug) ==\n");
    fail |= run_arena(c, 0, 60, 96, 512, 2048);

    printf("== #54 CONSOLIDATED ARENA: 60 distinct experts (K=512 N=2048) into ONE domain, shared chunks ==\n");
    {
        int d = ork_npu_domain_alloc(c);
        if (d > 0) { ork_npu_activate_domain(c, d);
            fail |= run_arena(c, d, 60, 96, 512, 2048);   /* 60 experts must share a few 256MB chunks + all bit-exact */
        } else printf("  domain_alloc returned %d — skip\n", d);
    }

    printf("== int4 doorbell RAPID ALTERNATION across two domains (the 35B forward pattern — the suspect) ==\n");
    {
        int d1 = ork_npu_domain_alloc(c), d2 = ork_npu_domain_alloc(c);
        if (d1 <= 0 || d2 <= 0){ printf("  domain_alloc returned %d/%d — cannot test alternation\n", d1, d2); fail = 1; }
        else {
            ork_npu_activate_domain(c, d1); ork_npu_activate_domain(c, d2);   /* establish BOTH before alternating */
            printf("  -- alternating between established domains %d and %d --\n", d1, d2);
            fail |= run_alternate(c, d1, d2, 128, 8);   /* BCHAIN (M>1) — the exact 35B expert path */
            fail |= run_alternate(c, d1, d2, 1,   8);   /* per-row (decode) */
        }
    }

    printf("%s\n", fail ? "DOMAIN PROBE: FAIL — int4 doorbell does NOT work across domains (bug reproduced)"
                        : "DOMAIN PROBE: ALL OK — int4 doorbell works in every domain");
    ork_npu_free(c); return fail;
}
