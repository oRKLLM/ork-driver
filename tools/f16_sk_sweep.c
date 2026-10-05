/* f16_sk_sweep — walk the fp16 K-slice count upward at a fixed shape.
 *
 * Written to localise a suspected Sk=4 bug. IT FOUND NO SUCH BUG: every slice count from 1 to 7 passes
 * standalone, including K=3584 N=256 — the exact shape that fails inside `make test` under ORK_F16_KS=1024.
 * The failure is therefore state-dependent, not slice-count-dependent, and the most likely cause is the
 * DOCUMENTED fp16 colsplit concurrent-fetch CDMA wild (Exp-2026-08-05, prevented within a buffer by CONTIG):
 * halving the K-slice size doubles the slice boundaries and so doubles exposure to a known intermittent
 * fault. Measured intermittency at M=4096: attempt 1 ran (451.3 GFLOP/s), attempt 2 produced nothing.
 *
 * Kept because the negative result is the useful part — it rules out slice count, which was the obvious
 * hypothesis and was wrong. Also note the K-slice size itself is NOT a throughput lever on this shape:
 * ks=2048 measures 453.4 GFLOP/s and ks=1024 measures 451.3, i.e. identical within noise. An earlier
 * "6.04x from ks=1024" was an artifact of comparing against a baseline taken before a reboot, with the
 * DDR governor unpinned — re-baseline after any reboot before comparing anything.
 */
#include <ork_npu.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static unsigned s_=7777; static float rnd(void){ s_=s_*1103515245u+12345u; return (float)((int)((s_>>13)&0x7fff)-16384)/16384.0f; }

int main(int argc,char**argv){
    int M = argc>1?atoi(argv[1]):256;
    int N = argc>2?atoi(argv[2]):256;
    int ks_built = argc>3?atoi(argv[3]):0;      /* for the banner only; 0 = unknown */
    int Ks[] = {1024, 2048, 3072, 3584, 4096, 5120, 7168};
    ork_npu *c = ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    printf("M=%d N=%d   (library built with soc->ks=%s)\n", M, N, ks_built?argv[3]:"?");
    printf("%-7s %-5s %-10s %-12s %s\n","K","Sk","us","result","max relerr");
    for(size_t q=0;q<sizeof Ks/sizeof Ks[0];q++){
        int K=Ks[q];
        int sk = ks_built>0 ? (K+ks_built-1)/ks_built : 0;
        ork_f16 *A=malloc((size_t)M*K*2), *B=malloc((size_t)K*N*2);
        float *C=malloc((size_t)M*N*4);
        if(!A||!B||!C){ printf("%-7d alloc failed\n",K); free(A);free(B);free(C); continue; }
        for(size_t i=0;i<(size_t)M*K;i++) A[i]=(ork_f16)rnd();
        for(size_t i=0;i<(size_t)K*N;i++) B[i]=(ork_f16)rnd();
        ork_w *w=ork_f16_mm_pack(c,K,N,B);
        if(!w){ printf("%-7d %-5d %-10s %-12s\n",K,sk,"-","PACK FAIL"); free(A);free(B);free(C); continue; }
        double t0=now_us(); int rc=ork_f16_mm_run(c,w,M,A,C); double us=now_us()-t0;
        if(rc){ printf("%-7d %-5d %-10.0f %-12s rc=%d  <-- HANG / submit failure\n",K,sk,us,"RUN FAIL",rc);
                ork_mm_free(c,w); free(A);free(B);free(C); continue; }
        double mx=0;
        for(int s=0;s<8;s++){
            int m=(s*M)/8, n=(s*N)/8+(s%3); if(n>=N)n=N-1;
            double r=0; for(int k=0;k<K;k++) r += (double)(float)A[(size_t)m*K+k]*(double)(float)B[(size_t)k*N+n];
            double d=fabs(r)>1e-6?fabs(r):1e-6, e=fabs(C[(size_t)m*N+n]-r)/d; if(e>mx)mx=e;
        }
        printf("%-7d %-5d %-10.0f %-12s %.2e%s\n",K,sk,us, mx<1e-2?"ok":"MISCOMPUTE", mx, mx<1e-2?"":"  <-- wrong answer");
        ork_mm_free(c,w); free(A); free(B); free(C);
    }
    ork_npu_free(c); return 0;
}
