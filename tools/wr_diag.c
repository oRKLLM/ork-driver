/* wr_diag — HOW is WEIGHT_REUSE's output wrong? Deterministic? Structured? Recoverable?
 *
 * The sweep established that reuse corrupts whenever it fires, but "corrupted" is a black box. This asks:
 *   1. DETERMINISM  -- same shape twice with reuse on: identical, or varying run to run?
 *   2. LOCALITY     -- which rows? M-tile bands are [0,mcap), [mcap,2mcap)... If only m0>0 bands are wrong,
 *                      the damage is confined to the tiles that carry the bit. test_matmul's ref-bad=1830 in
 *                      ROW 0 suggests otherwise -- row 0 is in the LOADER tile, which never gets the bit.
 *   3. COLUMN LOCALITY -- colsplit gives each core a column segment; a whole bad segment means one core's
 *                      weight was wrong, which is a different failure from scattered bit errors.
 *   4. STRUCTURE    -- are wrong values a shifted/scaled/stale version of the right ones, or unrelated?
 *                      A stale-weight read would produce a plausible dot product; a bit-level fault would not.
 *
 *   make wr_diag && sudo tools/util/npu_guard.sh -- ./wr_diag [M] [K] [N]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern int orki_i8_chain_fullk_mcap(ork_npu *c, int K);
static uint32_t sd = 12345;
/* ORK_WR_FULLRAND=1 widens this from 2 bits to the full int8 range. It is not cosmetic: bit k of an LCG
 * mod 2^32 has period 2^(k+1), so bits 16-17 REPEAT EVERY 262144 VALUES. Any shape whose per-M-tile
 * activation size (mcap*K) is a multiple of that period gets consecutive tiles holding BIT-IDENTICAL
 * rows -- and then reusing the previous tile's data is correct BY COINCIDENCE, with no fetch having been
 * declined. K=2048 (128*2048) and K=4096 (64*4096) are both exactly 262144. Both were recorded "CLEAN". */
static int rnd(void){ sd = sd*1103515245u + 12345u;
    static int full = -1; if(full < 0) full = getenv("ORK_WR_FULLRAND") ? 1 : 0;
    return full ? (int)((sd>>16)&0xff) - 128 : (int)((sd>>16)&3); }

static int run(ork_npu*c, int M,int K,int N, const int8_t*A, const int8_t*B, int32_t*C, int wr){
    if(wr) setenv("ORK_MTILE_WR","1",1); else unsetenv("ORK_MTILE_WR");
    ork_w*w=ork_i8_mm_pack(c,K,N,B); if(!w) return -1;
    int rc=ork_i8_mm_run(c,w,M,A,C);
    ork_mm_free(c,w); unsetenv("ORK_MTILE_WR"); return rc;
}

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):256, K=argc>2?atoi(argv[2]):2048, N=argc>3?atoi(argv[3]):2048;
    setvbuf(stdout,0,_IONBF,0);
    setenv("ORK_MTILE_WR_ALL","1",1);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    int mcap=orki_i8_chain_fullk_mcap(c,K), tiles=(M+mcap-1)/mcap;
    printf("wr_diag: M=%d K=%d N=%d  mcap=%d tiles=%d  (3 cores => ~%d columns per core segment)\n\n",
           M,K,N,mcap,tiles,N/3);

    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N);
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)(rnd()-1);
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)(rnd()-1);
    int32_t *Cref=malloc((size_t)M*N*4), *Cb1=malloc((size_t)M*N*4), *Cb2=malloc((size_t)M*N*4);
    if(run(c,M,K,N,A,B,Cref,0)){ printf("ref run failed\n"); return 2; }
    /* ORK_WR_PRE_K/N: inject a DIFFERENT shape between the reference pass and the reuse pass. Hypothesis:
     * reuse reads whatever the banks currently hold, so it is correct only when the IMMEDIATELY PRECEDING
     * op used the same weight -- which this harness accidentally arranges (ref then reuse, same weight), as
     * does a 20-iteration timing loop. Real workloads run a different matmul each time, which is the
     * wr_sweep / test_matmul case (9/9 corrupt). If so, priming flips this shape from 0.0%% to corrupt. */
    if(getenv("ORK_WR_PRE_K")){
        int pk=atoi(getenv("ORK_WR_PRE_K")), pn=getenv("ORK_WR_PRE_N")?atoi(getenv("ORK_WR_PRE_N")):pk;
        int8_t*pa=malloc((size_t)M*pk),*pb=malloc((size_t)pk*pn); int32_t*pc2=malloc((size_t)M*pn*4);
        for(size_t j=0;j<(size_t)M*pk;j++) pa[j]=(int8_t)(rnd()-1);
        for(size_t j=0;j<(size_t)pk*pn;j++) pb[j]=(int8_t)(rnd()-1);
        int prc=run(c,M,pk,pn,pa,pb,pc2,0);
        printf("  [prime] ran a DIFFERENT shape K=%d N=%d between ref and reuse (rc=%d)\n",pk,pn,prc);
        free(pa); free(pb); free(pc2);
    }
    if(run(c,M,K,N,A,B,Cb1,1)){ printf("reuse run 1 failed\n"); return 2; }
    if(run(c,M,K,N,A,B,Cb2,1)){ printf("reuse run 2 failed\n"); return 2; }

    /* 1. determinism */
    long d12=0; for(size_t j=0;j<(size_t)M*N;j++) if(Cb1[j]!=Cb2[j]) d12++;
    printf("  [1] DETERMINISM : reuse run1 vs run2 -> %ld/%zu differ  => %s\n",
           d12,(size_t)M*N, d12? "NON-deterministic" : "deterministic");

    /* 2. row locality, by M-tile band */
    printf("  [2] ROW LOCALITY (wrong elements per M-tile band):\n");
    for(int t=0;t<tiles;t++){
        int r0=t*mcap, r1=(t+1)*mcap; if(r1>M) r1=M;
        long bad=0; for(int i=r0;i<r1;i++) for(int n=0;n<N;n++) if(Cb1[(size_t)i*N+n]!=Cref[(size_t)i*N+n]) bad++;
        printf("        tile %d rows[%4d,%4d)  %s : %8ld / %8ld wrong (%.1f%%)\n", t, r0, r1,
               t? "REUSE" : "loader", bad, (long)(r1-r0)*N, 100.0*bad/((r1-r0)*(double)N));
    }

    /* 3. column locality, by core segment */
    printf("  [3] COLUMN LOCALITY (wrong elements per third of N):\n");
    for(int s=0;s<3;s++){
        int c0=s*(N/3), c1=(s==2)?N:(s+1)*(N/3);
        long bad=0; for(int i=0;i<M;i++) for(int n=c0;n<c1;n++) if(Cb1[(size_t)i*N+n]!=Cref[(size_t)i*N+n]) bad++;
        printf("        seg %d cols[%4d,%4d) : %8ld / %8ld wrong (%.1f%%)\n", s, c0, c1,
               bad, (long)M*(c1-c0), 100.0*bad/(M*(double)(c1-c0)));
    }

    /* 4. structure of the error */
    printf("  [4] ERROR STRUCTURE (first 6 mismatches):\n");
    int shown=0;
    for(size_t j=0;j<(size_t)M*N && shown<6;j++){
        if(Cb1[j]!=Cref[j]){
            int i=(int)(j/N), n=(int)(j%N);
            printf("        (%4d,%4d) ref=%9d got=%9d  diff=%9d  ratio=%.4f\n",
                   i,n,Cref[j],Cb1[j],Cb1[j]-Cref[j], Cref[j]? (double)Cb1[j]/Cref[j] : 0.0);
            shown++;
        }
    }
    long zero=0,tot=0; for(size_t j=0;j<(size_t)M*N;j++) if(Cb1[j]!=Cref[j]){ tot++; if(Cb1[j]==0) zero++; }
    printf("        total wrong=%ld, of which exactly ZERO=%ld (%.1f%%)\n", tot, zero, tot?100.0*zero/tot:0.0);

    /* [4a] ERROR MAGNITUDE. C[m,n] = sum_k A[m,k]*B[k,n], so the two operands corrupt ASYMMETRICALLY: a
     * stale WEIGHT column ruins only that column (columns are independent), but a stale ACTIVATION at ANY k
     * ruins the WHOLE ROW for every n, because every output in the row sums over all k. Partial data
     * staleness therefore CASCADES to ~100% of outputs, and the error COUNT saturates -- it cannot tell
     * "one bank stale" from "everything stale". The MAGNITUDE can: if a fraction f of the k-terms are
     * stale, each output should be off by roughly f, not wildly. */
    { double sum_rel = 0; long n_rel = 0; int32_t maxabs = 0; double refmag = 0;
      for (size_t j = 0; j < (size_t)M*N; j++) {
          int32_t r = Cref[j] < 0 ? -Cref[j] : Cref[j]; refmag += r;
          if (Cb1[j] == Cref[j]) continue;
          int32_t d = Cb1[j] - Cref[j]; if (d < 0) d = -d;
          if (d > maxabs) maxabs = d;
          if (r > 0) { sum_rel += (double)d / r; n_rel++; }
      }
      printf("  [4a] ERROR MAGNITUDE over wrong elements: mean |diff|/|ref| = %.3f, max |diff| = %d, mean |ref| = %.0f\n",
             n_rel ? sum_rel/n_rel : 0.0, maxabs, refmag/((double)M*N));
      printf("        <<1  => only a FRACTION of the k-terms are stale (partial residency, cascaded by the K-sum)\n"
             "        ~1+  => the reduction is essentially unrelated to the reference (whole operand wrong)\n"); }

    /* [5] IS THE CORRECT PART CONTIGUOUS IN COLUMNS?  The pack layout is column-major in 32-wide groups
     * (woff = ((n/32)*KT + k/32)*1024 + ...), so a weight bank holding the last-streamed bytes would make a
     * CONTIGUOUS RUN of 32-column groups correct. If instead the correct columns are scattered, residency is
     * interleaved and no N-tiling could fix it. This is the test of "are the weights uploaded in a layout
     * that reuse can work with". */
    /* [4b] ROW distribution within the first reuse tile. WEIGHT_REUSE corrupted in COLUMNS (the correct
     * ones a contiguous run = the resident weight slice). DATA_REUSE's operand is the ACTIVATION, so any
     * partial residency should localise in ROWS instead. A contiguous run of correct rows would be the
     * direct analogue and would give a data-side capacity law; scattered rows would not. */
    printf("  [4b] ROW DISTRIBUTION of CORRECT elements inside the first reuse tile:\n        ");
    { int r0 = mcap, r1 = (2*mcap < M) ? 2*mcap : M, shown = 0, runs = 0, prev = -1;
      for (int r = r0; r < r1 && shown < 64; r++, shown++) {
          int ok = 1;
          for (int n2 = 0; n2 < N; n2++) if (Cb1[(size_t)r*N+n2] != Cref[(size_t)r*N+n2]) { ok = 0; break; }
          printf("%c", ok ? '#' : '.');
          if (ok != prev) { runs++; prev = ok; }
      }
      printf("   ('#' = row fully correct; %d transitions)\n", runs);
      printf("        %s\n", runs <= 3 ? "CONTIGUOUS -> row-wise residency, a data-side capacity law is plausible"
                                        : "SCATTERED  -> not simple row residency"); }
    printf("  [5] COLUMN DISTRIBUTION of CORRECT elements within core segment 0 (32-col groups):\n");
    { int segw = N/3, ngrp = segw/32, row = mcap;   /* first reuse row */
      int shown=0, runs=0, prev=-1;
      printf("        ");
      for(int g=0; g<ngrp && shown<64; g++){
          int c0=g*32, ok=1;
          for(int n2=c0;n2<c0+32;n2++) if(Cb1[(size_t)row*N+n2]!=Cref[(size_t)row*N+n2]){ ok=0; break; }
          printf("%c", ok?'#':'.');
          if(ok!=prev){ runs++; prev=ok; }
          shown++;
      }
      printf("   ('#' = all 32 cols correct, '.' = has errors; %d transitions)\n", runs);
      printf("        contiguity: %s\n", runs<=3 ? "CONTIGUOUS -> residency is a contiguous slice; N-tiling could work"
                                                  : "SCATTERED -> residency is interleaved; N-tiling cannot fix it"); }
    ork_npu_free(c); return 0;
}
