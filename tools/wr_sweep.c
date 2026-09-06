/* wr_sweep — a VALIDATED sweep of where WEIGHT_REUSE is correct and where it corrupts.
 *
 * WHY A NEW HARNESS. wreuse_probe reports "bit-exact" for shapes that test_matmul proves are broken
 * (M=256 K=2048 N=2048: probe bit-exact, test_matmul 1830 wrong outputs, bit firing in BOTH). A probe that
 * cannot detect a known failure carries no information when it reports success, so every "correct" cell it
 * produced is void. This harness therefore copies test_matmul's structure exactly -- the same fixed-seed
 * PRNG fill, ONE context shared across shapes, pack once, run ONCE via ork_i8_mm_run, checksum the result --
 * and is VALIDATED against the known-bad shape before any sweep result is believed.
 *
 * Detection method is test_matmul's: run the whole sweep twice, once with reuse off and once on, and diff
 * the per-shape checksums. The reuse-off pass is the reference (make test passes without the knob), so any
 * checksum that moves is reuse corrupting that shape. No CPU reference needed, and it cannot be fooled by a
 * sampling stride.
 *
 *   make wr_sweep && sudo tools/util/npu_guard.sh -- ./wr_sweep            # reference pass
 *   make wr_sweep && sudo tools/util/npu_guard.sh -- env ORK_MTILE_WR=1 ORK_MTILE_WR_ALL=1 ./wr_sweep
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern int orki_i8_chain_fullk_mcap(ork_npu *c, int K);

/* test_matmul's fixed-seed PRNG, verbatim, so the inputs match the harness this is validated against */
static uint32_t sd = 12345;
static int rnd(void){ sd = sd*1103515245u + 12345u; return (int)((sd>>16)&3); }
static uint64_t fnv64(const void*p,size_t n){ const uint8_t*b=p; uint64_t h=1469598103934665603ULL;
    for(size_t i=0;i<n;i++){ h^=b[i]; h*=1099511628211ULL; } return h; }

int main(void){
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    int on = getenv("ORK_MTILE_WR") != NULL;
    printf("wr_sweep: reuse=%s   (run twice and diff the checksums; the reuse=off pass is the reference)\n\n",
           on?"ON":"off");
    printf("  %-22s %-6s %-7s %-6s %s\n","shape M,K,N","mcap","tiles","fires","checksum");

    /* {256,2048,2048} and {256,3584,3584} are the VALIDATION shapes -- test_matmul proves reuse breaks them.
     * If this harness does not move their checksums, it is not a detector and the rest of the table is void. */
    int shapes[][3] = {
        {256,2048,2048}, {256,3584,3584},                      /* known-bad (validation) */
        {128,2048,2048}, {200,1024,2048}, {128,512,1536},      /* 1-tile cases: reuse cannot fire */
        {256,1024,2048}, {256,1536,2048}, {256,2560,2048},
        {256,3072,2048}, {256,4096,2048},
        {384,2048,1024}, {512,2048,2048}, {256,2048,4096},
    };
    int ns = (int)(sizeof shapes/sizeof shapes[0]);
    for(int s=0;s<ns;s++){
        int M=shapes[s][0],K=shapes[s][1],N=shapes[s][2];
        int mcap=orki_i8_chain_fullk_mcap(c,K), tiles=(M+mcap-1)/mcap;
        int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t*C=malloc((size_t)M*N*4);
        if(!A||!B||!C){ printf("  alloc fail\n"); free(A); free(B); free(C); continue; }
        for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)(rnd()-1);
        for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)(rnd()-1);
        ork_w*w=ork_i8_mm_pack(c,K,N,B);
        if(!w){ printf("  %-22s pack failed\n","-"); free(A); free(B); free(C); continue; }
        int rc=ork_i8_mm_run(c,w,M,A,C);       /* ONCE -- exactly as test_matmul's ChainPrefill does */
        char sh[32]; snprintf(sh,sizeof sh,"%d,%d,%d",M,K,N);
        if(rc){ printf("  %-22s %-6d %-7d %-6s run rc=%d\n",sh,mcap,tiles,"-",rc); }
        else   { printf("  %-22s %-6d %-7d %-6s 0x%016llx\n",sh,mcap,tiles,
                        tiles>1?(on?"yes":"n/a"):"no", (unsigned long long)fnv64(C,(size_t)M*N*4)); }
        ork_mm_free(c,w); free(A); free(B); free(C);
    }
    printf("\n  'fires' = reuse can act (tiles>1) with the knob on. A checksum that MOVES between the two\n"
           "  passes is reuse corrupting that shape; one that holds is genuinely unaffected.\n");
    ork_npu_free(c); return 0;
}
