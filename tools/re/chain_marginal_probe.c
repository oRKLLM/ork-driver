/* chain_marginal_probe — measure the TRUE single-ioctl HW-chain marginal cost per task
 * (task_number=S, one submit, HW walks the 0x0010/0x0014 descriptor), NOT the doorbell spine.
 *
 * Correction to ws_xition_probe: that probe's tiny M=1 tasks routed through ork_dyn_begin_mc (the
 * doorbell spine), which rebuilds the chain every call + does per-task host setup/writeback -> a 7us/task
 * SOFTWARE cost, not the HW chain. run_chain_i8_impl only takes its single-submit HW-chain path when
 * routable=0, i.e. a task with M>64 && !Bf. pack_i8(K<=4096) is Sk=1/Bb-only (no Bf), so M=128 forces it.
 *
 * We sweep S at M=128 (HW chain) and report per-task marginal + a per-CHAIN-START intercept, to separate:
 *   - chain START / prep  (intercept, once per chain)
 *   - per-task steady-state (slope: expected ~= the tile's own DMA/compute, overlapped; NOT a fixed adder)
 * vs the doorbell spine's ~7us/task and the ~96us individual-submit floor. All-ones => C==K, self-checking.
 *
 * BOARD: sudo env ORK_MM_TIMEOUT=3000 timeout 180 ./chain_marginal_probe [iters]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }

int main(int argc, char**argv){
    int iters = argc>1?atoi(argv[1]):200;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu *c = ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    int K=512, N=64;   /* tiny weight (32KB, ~3us DMA) so the slope is chain overhead, not bandwidth */
    int8_t *B = malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    ork_w *w = ork_mm_pack_i8(c,K,N,B); free(B);
    if(!w){ printf("pack fail\n"); return 2; }
    printf("chain_marginal_probe: K=%d N=%d iters=%d  (M=128 forces the single-ioctl HW-chain path)\n",K,N,iters);

    (void)K;(void)w;
    /* Isolate the per-task chain cost from weight re-stream: single-submit HW chain, S=32, M=1 (minimal
     * compute/output), sweep weight size via N (K fixed 512). per_task(N) = FLOOR + weight_DMA(K*N).
     * The linear-fit INTERCEPT (N->0) is the irreducible per-task cost: chain transition + host, with the
     * weight re-stream removed. That floor is what a WEIGHT_REUSE chain (which skips the weight DMA) would
     * pay per tile — the number that decides weight-stationary. */
    int S=32, M=1;
    int Ns[]={32,64,128,256,512,1024}; int nN=sizeof Ns/sizeof*Ns;
    printf("\n[single-submit HW chain, S=%d M=%d]  sweep weight size (K*N):\n", S, M);
    printf("   N   wKB   us/task   (K*N bytes)\n");
    double xs[8],ys[8]; int np=0;
    for(int j=0;j<nN;j++){
        int Nj=Ns[j];
        int8_t *Bj=malloc((size_t)K*Nj); memset(Bj,1,(size_t)K*Nj);
        ork_w *wj=ork_mm_pack_i8(c,K,Nj,Bj); free(Bj);
        if(!wj){ printf("  %4d  pack fail\n",Nj); continue; }
        int8_t *A=malloc((size_t)M*K); memset(A,1,(size_t)M*K);
        int32_t *Cs[64]; ork_mm_task_i8 tk[64];
        for(int i=0;i<S;i++){ Cs[i]=calloc((size_t)M*Nj,4); tk[i]=(ork_mm_task_i8){wj,M,A,Cs[i]}; }
        int rc=ork_mm_run_chain_i8(c,S,tk);
        if(rc){ printf("  %4d  rc=%d\n",Nj,rc); for(int i=0;i<S;i++){ free(Cs[i]); } free(A); continue; }
        double t0=now_us(); for(int i=0;i<iters;i++){ ork_mm_run_chain_i8(c,S,tk); } double per=(now_us()-t0)/iters/S;
        int bad=0; for(int i=0;i<S;i++){ for(size_t e=0;e<(size_t)M*Nj;e++){ if(Cs[i][e]!=K){bad++;break;} } }
        printf("  %4d  %4d   %7.2f     (%d)%s\n", Nj, K*Nj/1024, per, K*Nj, bad?"  C!=K":"");
        xs[np]=(double)K*Nj; ys[np]=per; np++;
        for(int i=0;i<S;i++){ free(Cs[i]); } free(A);
    }
    if(np>=2){
        double sx=0,sy=0,sxx=0,sxy=0;
        for(int i=0;i<np;i++){ sx+=xs[i]; sy+=ys[i]; sxx+=xs[i]*xs[i]; sxy+=xs[i]*ys[i]; }
        double b=(np*sxy-sx*sy)/(np*sxx-sx*sx), a=(sy-b*sx)/np;
        printf("  --> per_task = %.2f us floor + weight_DMA(%.1f GB/s).  FLOOR (reuse chain per-tile) ~= %.2f us\n",
               a, 1.0/b/1e3, a);
    }
    printf("\nDONE\n");
    return 0;
}
