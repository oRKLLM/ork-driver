/* law_bench — does the LAW-GATED weight reuse win where the weight fits NATURALLY?
 *
 * The gate is now the measured law (K*segw <= WEIGHT_BANK*32KB) evaluated on the geometry we were already
 * going to emit -- not a shape whitelist, which can only be wrong about shapes nobody has run yet, and not
 * the forcing function (ORK_WR_NTILE narrows segments to MAKE the weight fit and measured 0.39-0.42x).
 * Natural fit needs segw <= ~64, i.e. N/nc <= 64, so this probes NARROW-N shapes: MoE expert projections,
 * GQA K/V, adapters, draft models.
 *
 * A win also requires M > mcap, or there is no second M-tile and so no re-stream to avoid. Both arms are
 * checked bit-exact against each other; the reuse arm must be identical, not merely close.
 */
#include <ork_npu.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static unsigned s_=12345; static int rnd(void){ s_=s_*1103515245u+12345u; return (int)((s_>>16)&0xff)-128; }
static unsigned long long ck(const int32_t*C,size_t n){ unsigned long long h=1469598103934665603ULL;
    for(size_t i=0;i<n;i++){ h^=(unsigned)C[i]; h*=1099511628211ULL; } return h; }
extern int orki_i8_chain_fullk_mcap(ork_npu *c, int K);
int main(int argc,char**argv){
    int iters = argc>1?atoi(argv[1]):10;
    ork_npu *c = ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    struct { int K,N; } sh[] = {{2048,192},{2048,128},{2048,384},{1024,192},{3072,192},{2048,2048}};
    int Ms[] = {256,512,1024};
    printf("%-6s %-5s %-5s %-5s %-6s %-10s %-10s %-8s %s\n",
           "K","N","M","mcap","tiles","reuse_us","noreuse_us","speedup","correct");
    for(size_t q=0;q<sizeof sh/sizeof sh[0];q++){
        int K=sh[q].K,N=sh[q].N, mcap=orki_i8_chain_fullk_mcap(c,K);
        int8_t *B=malloc((size_t)K*N); s_=999; for(size_t i=0;i<(size_t)K*N;i++)B[i]=rnd();
        ork_w *w=ork_i8_mm_pack(c,K,N,B); if(!w){ printf("%-6d %-5d pack failed\n",K,N); free(B); continue; }
        for(size_t r=0;r<sizeof Ms/sizeof Ms[0];r++){
            int M=Ms[r]; int8_t *A=malloc((size_t)M*K); int32_t *C=malloc((size_t)M*N*4);
            s_=4242; for(size_t i=0;i<(size_t)M*K;i++)A[i]=rnd();
            unsigned long long h[2]; double us[2]; int ok=1;
            for(int arm=0;arm<2;arm++){          /* arm0 = law gate active (default), arm1 = forced off */
                if(arm) setenv("ORK_NO_MTILE_WR","1",1); else unsetenv("ORK_NO_MTILE_WR");
                if(ork_i8_mm_run(c,w,M,A,C)){ ok=0; break; }
                h[arm]=ck(C,(size_t)M*N);
                double t0=now(); for(int it=0;it<iters;it++) ork_i8_mm_run(c,w,M,A,C); us[arm]=(now()-t0)/iters;
            }
            unsetenv("ORK_NO_MTILE_WR");
            if(!ok) printf("%-6d %-5d %-5d run failed\n",K,N,M);
            else printf("%-6d %-5d %-5d %-5d %-6d %-10.1f %-10.1f %-8.3f %s\n", K,N,M,mcap,
                        (M+mcap-1)/mcap, us[0], us[1], us[1]/us[0],
                        h[0]==h[1]?"bit-exact":"*** MISMATCH ***");
            free(A); free(C);
        }
        ork_mm_free(c,w); free(B);
    }
    ork_npu_free(c); return 0;
}
