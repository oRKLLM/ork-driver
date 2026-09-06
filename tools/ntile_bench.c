/* ntile_bench — does WEIGHT_REUSE N-tiling (ORK_WR_NTILE) pay?
 *
 * The trade is explicit. Capping a colsplit segment at segcap = WEIGHT_BANK*32KB/K columns lets every
 * M-tile after the first reuse the segment's weight instead of re-streaming it, so the weight is read
 * once per SEGMENT rather than once per (segment x M-tile). But it also multiplies the program count by
 * ceil(Ncore/segcap), and each extra program costs ~1.3us of per-task hardware floor. Whether that nets
 * out is a measurement, not an argument: the saving grows with M (more tiles to amortise over) while the
 * program-count penalty is fixed, so the M sweep is the point of this probe.
 *
 * CORRECTNESS IS THE GATE, not a footnote: N-tiling changes the segmentation that end()'s boundary
 * scatter re-derives, so a mismatched cut mis-places columns silently. Every row therefore checks the
 * N-tiled checksum against the untiled one and prints MISMATCH rather than a speedup if they differ.
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
    ork_npu *c = ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    struct { int K,N; } sh[] = {{2048,2048},{3072,2048},{3584,3584},{4096,4096}};
    int Ms[] = {256,512,1024}, iters = argc>1?atoi(argv[1]):3;
    printf("%-6s %-6s %-5s %-9s %-9s %-8s %s\n","K","N","M","off_us","ntile_us","speedup","correct");
    for(size_t q=0;q<sizeof sh/sizeof sh[0];q++){
        int K=sh[q].K, N=sh[q].N;
        int8_t *B=malloc((size_t)K*N); s_=999; for(size_t i=0;i<(size_t)K*N;i++)B[i]=rnd();
        ork_w *w=ork_i8_mm_pack(c,K,N,B); if(!w){ printf("%-6d %-6d pack failed\n",K,N); free(B); continue; }
        for(size_t r=0;r<sizeof Ms/sizeof Ms[0];r++){
            int M=Ms[r];
            int8_t *A=malloc((size_t)M*K); int32_t *C=malloc((size_t)M*N*4);
            s_=4242; for(size_t i=0;i<(size_t)M*K;i++)A[i]=rnd();
            unsigned long long h[2]; double us[2]; int ok=1;
            for(int arm=0;arm<2;arm++){
                if(arm) setenv("ORK_WR_NTILE","1",1); else unsetenv("ORK_WR_NTILE");
                if(ork_i8_mm_run(c,w,M,A,C)){ ok=0; break; }
                h[arm]=ck(C,(size_t)M*N);
                double t0=now(); for(int it=0;it<iters;it++) ork_i8_mm_run(c,w,M,A,C); us[arm]=(now()-t0)/iters;
            }
            unsetenv("ORK_WR_NTILE");
            if(!ok) printf("%-6d %-6d %-5d run failed\n",K,N,M);
            else printf("%-6d %-6d %-5d %-9.0f %-9.0f %-8.3f %s\n",K,N,M,us[0],us[1],us[0]/us[1],
                        h[0]==h[1]?"bit-exact":"*** MISMATCH ***");
            free(A); free(C);
        }
        ork_mm_free(c,w); free(B);
    }
    ork_npu_free(c); return 0;
}
