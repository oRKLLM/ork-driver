/* dr_time — is DATA_REUSE's correctness the SAME VARIABLE as its speed?
 *
 * NVDLA's programming guide documents the reuse bit as a REQUEST the hardware honours only when the
 * retained CBUF slices cover the new layer's need ("if CDMA_HEIGHT_N <= (CSC_HEIGHT - D_RELEASE)_{N-1},
 * the Nth CDMA fetch will be skipped"), and says the skip may be FULL or PARTIAL. That predicts a
 * correlation no register sweep could see: a K whose output stays CORRECT under DATA_REUSE skipped
 * nothing, so it must show NO speedup; a K whose output is CORRUPT skipped a fetch, so it must show one.
 *
 * POSITIVE CONTROL (this probe voids itself without it): K=1536/3072/3584 are known-corrupt under
 * DATA_REUSE. If their checksums do NOT move, the knob did not fire and every number here is noise --
 * which is exactly how the first version of this probe reported six CLEAN rows and a 1.03x "speedup"
 * while emitting the baseline program in both arms (ORK_MTILE_WR_ALL only lifts the shape GATE; the
 * feature itself is ORK_MTILE_WR).  K=2560 is excluded: it submit-times-out (errno=110) under reuse.
 */
#include <ork_npu.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static unsigned s_=12345; static int rnd(void){ s_=s_*1103515245u+12345u; return (int)((s_>>16)&0xff)-128; }
int main(void){
    ork_npu *c = ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    const int M=256, N=2048;
    /* expect_corrupt mirrors the measured wr_diag table; it is the control, not an assumption under test */
    const int Ks[]  = {2048, 4096, 1536, 3072, 3584};
    const int badK[]= {   0,    0,    1,    1,    1};
    const int nk = 5, iters = 4;
    int ctrl_seen = 0, ctrl_tot = 0;
    printf("K      expect    observed   base_us   reuse_us   speedup\n");
    for(int q=0;q<nk;q++){
        int K=Ks[q];
        int8_t *A=malloc((size_t)M*K), *B=malloc((size_t)K*N);
        int32_t *C=malloc((size_t)M*N*4);
        s_=12345; for(size_t i=0;i<(size_t)M*K;i++)A[i]=rnd(); for(size_t i=0;i<(size_t)K*N;i++)B[i]=rnd();
        ork_w *w = ork_i8_mm_pack(c,K,N,B); if(!w){ printf("%-6d pack failed\n",K); free(A);free(B);free(C); continue; }
        unsigned long long ck[2]={0,0}; double us[2]={0,0}; int ok=1;
        for(int r=0;r<2;r++){
            if(r){ setenv("ORK_MTILE_WR","1",1); setenv("ORK_MTILE_DR","1",1); }
            else { unsetenv("ORK_MTILE_WR"); unsetenv("ORK_MTILE_DR"); }
            /* CORRECTNESS comes from ONE run, before any repeat. A timing loop re-submits the SAME
             * program, so the last iteration's output is a WARM one whose CBUF was primed by the
             * identical run before it -- that is the exact defect that made wreuse_probe a broken
             * detector, and it is why the checksum is taken here and the timing separately below. */
            if(ork_i8_mm_run(c,w,M,A,C)){ printf("%-6d run failed (arm %d)\n",K,r); ok=0; break; }
            unsigned long long h=1469598103934665603ULL;
            for(size_t i=0;i<(size_t)M*N;i++){ h^=(unsigned)C[i]; h*=1099511628211ULL; }
            ck[r]=h;
            double t0=now(); for(int it=0;it<iters;it++) ork_i8_mm_run(c,w,M,A,C); us[r]=(now()-t0)/iters;
        }
        if(ok){
            int moved = (ck[0]!=ck[1]);
            if(badK[q]){ ctrl_tot++; if(moved) ctrl_seen++; }
            printf("%-6d %-9s %-10s %-9.0f %-10.0f %.3f\n", K,
                   badK[q]?"corrupt":"CLEAN", moved?"corrupt":"CLEAN", us[0], us[1], us[0]/us[1]);
        }
        unsetenv("ORK_MTILE_WR"); unsetenv("ORK_MTILE_DR");
        ork_mm_free(c,w); free(A); free(B); free(C);
    }
    printf("\npositive control: %d/%d known-corrupt shapes actually corrupted\n", ctrl_seen, ctrl_tot);
    if(ctrl_seen < ctrl_tot){
        printf("VOID — the knob did not fire; these timings are two copies of the baseline, not a comparison.\n");
        ork_npu_free(c); return 3;
    }
    printf("VALID — reuse was applied. If CLEAN rows show ~1.00x and corrupt rows >1.00x, correctness\n"
           "and speed are the same variable: the bit is a request the hardware declines when it cannot\n"
           "cover the fetch, and honours (partially) when it can.\n");
    ork_npu_free(c); return 0;
}
