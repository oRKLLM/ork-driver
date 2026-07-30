/* fold_resident_probe — measure the orkpack-format proposition: mfold run from a PRE-PACKED resident fold
 * weight (ork_w_dump_fold_i8_cpu -> ork_mm_load_fold_i8 -> ork_npu_fold_run_w, NO per-call fold_woff repack)
 * vs the normal ork_mm_run_i8 path, at the 7B q/k/v/o shapes (K=3584). A/B time + BIT-EXACT check.
 *
 * This is the go/no-go for the orkpack v5 "Bfold" format change: if resident-fold beats normal at M<=~72
 * (bit-exact), the format bump + doorbell-tiler + 7B regen are worth it; if not, we skip that integration.
 *
 * BOARD: sudo env ORK_MM_TIMEOUT=3000 timeout 300 ./fold_resident_probe [iters]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static uint32_t g=0x1234567u; static int8_t r8(void){ g=g*1664525u+1013904223u; return (int8_t)((g>>24)&0x3f)-32; }
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

int main(int argc,char**argv){
    int iters=argc>1?atoi(argv[1]):100;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    int K=3584; int Ns[]={512,3584}; int Ms[]={8,16,32,64,128};
    printf("fold_resident_probe: K=%d iters=%d  (resident mfold vs normal ork_mm_run_i8, q/k/v/o shapes)\n",K,iters);
    for(size_t ni=0; ni<sizeof Ns/sizeof*Ns; ni++){
        int N=Ns[ni];
        int8_t *W=malloc((size_t)K*N), *A=malloc((size_t)128*K);
        for(size_t i=0;i<(size_t)K*N;i++) W[i]=r8();
        for(size_t i=0;i<(size_t)128*K;i++) A[i]=r8();
        /* reference weight (normal path) */
        ork_w *wn=ork_mm_pack_i8(c,K,N,W);
        /* fold weight: dump -> load resident */
        size_t fb=ork_w_dump_fold_i8_cpu(c,K,N,W,NULL,0);
        void *blob=malloc(fb); ork_w_dump_fold_i8_cpu(c,K,N,W,blob,fb);
        ork_w *wf=ork_mm_load_fold_i8(c,K,N,blob,fb);
        if(!wn||!wf){ printf("  N=%d pack/load fail (wn=%p wf=%p fb=%zu)\n",N,(void*)wn,(void*)wf,fb); free(W);free(A);free(blob); continue; }
        printf("\n== N=%d (fold blob %.1f MB) ==\n", N, fb/1e6);
        for(size_t mi=0; mi<sizeof Ms/sizeof*Ms; mi++){
            int M=Ms[mi];
            int32_t *Cn=calloc((size_t)M*N,4), *Cf=calloc((size_t)M*N,4);
            if(ork_mm_run_i8(c,wn,M,A,Cn)){ printf("  M=%-3d normal run FAIL\n",M); free(Cn);free(Cf); continue; }
            double us_f=0; int rc=ork_npu_fold_run_w(c,wf,M,A,Cf,iters,&us_f);
            if(rc){ printf("  M=%-3d fold run rc=%d\n",M,rc); free(Cn);free(Cf); continue; }
            /* bit-exact */
            long mism=0; for(size_t e=0;e<(size_t)M*N;e++) if(Cn[e]!=Cf[e]){ mism++; }
            /* time normal */
            double t0=now_us(); for(int i=0;i<iters;i++) ork_mm_run_i8(c,wn,M,A,Cn); double us_n=(now_us()-t0)/iters;
            printf("  M=%-3d  normal %8.1f us | fold %8.1f us | %.2fx  mism=%ld%s\n",
                   M, us_n, us_f, us_n/us_f, mism, mism?"  <-- MISMATCH":"");
            free(Cn); free(Cf);
        }
        ork_mm_free(c,wn); ork_mm_free(c,wf); free(W); free(A); free(blob);
    }
    printf("\nDONE\n");
    return 0;
}
