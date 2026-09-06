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
static int rnd(void){ sd = sd*1103515245u + 12345u; return (int)((sd>>16)&3); }

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

    /* [5] IS THE CORRECT PART CONTIGUOUS IN COLUMNS?  The pack layout is column-major in 32-wide groups
     * (woff = ((n/32)*KT + k/32)*1024 + ...), so a weight bank holding the last-streamed bytes would make a
     * CONTIGUOUS RUN of 32-column groups correct. If instead the correct columns are scattered, residency is
     * interleaved and no N-tiling could fix it. This is the test of "are the weights uploaded in a layout
     * that reuse can work with". */
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
