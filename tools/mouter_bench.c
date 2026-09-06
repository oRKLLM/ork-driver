/* mouter_bench — does transposing the colsplit loop make DATA_REUSE valid?
 *
 * Default order is segment-outer: consecutive programs on a core share the WEIGHT and differ in the
 * ACTIVATION, so bit 13 is the licensed bit and bit 12 corrupts at every shape (measured). Transposed to
 * M-tile-outer, consecutive programs share the ACTIVATION and differ in the weight, so bit 12 should
 * become the valid one. That is the direct test of "invalid by construction" -- if the same bit that
 * corrupts 100% of outputs in one order is bit-exact in the other, the explanation holds.
 *
 * Three arms, because the reorder and the reuse bit must be separable:
 *   base          default order, no reuse          <- reference checksum
 *   MOUTER=2      transposed, NO reuse bit         <- isolates the reordering itself
 *   MOUTER=1      transposed, DATA_REUSE on        <- the hypothesis
 * A mismatch in arm 2 means the transpose is broken; a mismatch only in arm 3 means bit 12 is still wrong.
 */
#include <ork_npu.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static unsigned s_=12345; static int rnd(void){ s_=s_*1103515245u+12345u; return (int)((s_>>16)&0xff)-128; }
static unsigned long long ck(const int32_t*C,size_t n){ unsigned long long h=1469598103934665603ULL;
    for(size_t i=0;i<n;i++){ h^=(unsigned)C[i]; h*=1099511628211ULL; } return h; }
int main(int argc,char**argv){
    const char *segw = argc>1?argv[1]:"64"; int iters = argc>2?atoi(argv[2]):4;
    ork_npu *c = ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    /* real prefill projections: FFN gate/up (N=18944) and o/qkv-class widths, all >> nmax so they are
     * already multi-segment; plus a narrow one as the control where nothing is segmented. */
    struct { int K,N; } sh[] = {{3584,18944},{3584,4608},{2048,8192},{2048,2048}};
    int Ms[] = {64,256};
    printf("segment width forced to %s columns\n", segw);
    printf("%-6s %-6s %-5s %-10s %-12s %-10s %-12s %s\n",
           "K","N","M","base_us","reorder","reorder_us","reuse(b12)","reuse_us / x");
    for(size_t q=0;q<sizeof sh/sizeof sh[0];q++){
        int K=sh[q].K,N=sh[q].N;
        int8_t *B=malloc((size_t)K*N); s_=999; for(size_t i=0;i<(size_t)K*N;i++)B[i]=rnd();
        ork_w *w=ork_i8_mm_pack(c,K,N,B); if(!w){ printf("%-6d %-6d pack failed\n",K,N); free(B); continue; }
        for(size_t r=0;r<sizeof Ms/sizeof Ms[0];r++){
            int M=Ms[r]; int8_t *A=malloc((size_t)M*K); int32_t *C=malloc((size_t)M*N*4);
            s_=4242; for(size_t i=0;i<(size_t)M*K;i++)A[i]=rnd();
            unsigned long long h[3]; double us[3]; int ok=1;
            for(int arm=0;arm<3;arm++){
                if(arm==0){ unsetenv("ORK_WR_MOUTER"); unsetenv("ORK_WR_SEGW"); }
                else { /* segw "0" = do NOT force segmentation: use whatever nmax already imposes. That is the
                        * only case that matters in production -- a wide-N shape is ALREADY cut into segments,
                        * so its activation is ALREADY re-fetched per segment and reuse costs nothing to add. */
                       if (segw[0] != '0') setenv("ORK_WR_SEGW",segw,1); else unsetenv("ORK_WR_SEGW");
                       setenv("ORK_WR_MOUTER", arm==1?"2":"1", 1); }
                if(ork_i8_mm_run(c,w,M,A,C)){ ok=0; break; }
                h[arm]=ck(C,(size_t)M*N);
                double t0=now(); for(int it=0;it<iters;it++) ork_i8_mm_run(c,w,M,A,C); us[arm]=(now()-t0)/iters;
            }
            unsetenv("ORK_WR_MOUTER"); unsetenv("ORK_WR_SEGW");
            if(!ok) printf("%-6d %-6d %-5d run failed\n",K,N,M);
            else printf("%-6d %-6d %-5d %-10.0f %-12s %-10.0f %-12s %.0f / %.3fx\n", K,N,M, us[0],
                        h[1]==h[0]?"bit-exact":"MISMATCH", us[1],
                        h[2]==h[0]?"bit-exact":"MISMATCH", us[2], us[0]/us[2]);
            free(A); free(C);
        }
        ork_mm_free(c,w); free(B);
    }
    ork_npu_free(c); return 0;
}
