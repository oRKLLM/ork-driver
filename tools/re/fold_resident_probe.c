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
    int K=3584; int Ns[]={512,3584,18944}; int Ms[]={8,16,32,64,128};   /* k/v, q/o, gate/up */
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
    /* ---- AUTO-ROUTE: a normally-packed q/o weight with a fold blob attached; ork_mm_run_i8 must self-select
     * the fold at M<=64 and the normal path at M>64, both bit-exact vs a pure-normal reference weight. ---- */
    { int N=3584; int8_t *W=malloc((size_t)K*N), *A=malloc((size_t)128*K);
      for(size_t i=0;i<(size_t)K*N;i++) W[i]=r8(); for(size_t i=0;i<(size_t)128*K;i++) A[i]=r8();
      ork_w *wref=ork_mm_pack_i8(c,K,N,W), *wrt=ork_mm_pack_i8(c,K,N,W);
      size_t fb=ork_w_dump_fold_i8_cpu(c,K,N,W,NULL,0); void *blob=malloc(fb); ork_w_dump_fold_i8_cpu(c,K,N,W,blob,fb);
      int at=ork_w_attach_fold_i8(c,wrt,blob,fb);
      printf("\n== AUTO-ROUTE (N=3584): ork_mm_run_i8 self-select (attach rc=%d) ==\n", at);
      int RM[]={32,128};
      for(size_t j=0;j<sizeof RM/sizeof*RM;j++){ int M=RM[j];
        int32_t *Cr=calloc((size_t)M*N,4), *Ce=calloc((size_t)M*N,4);
        ork_mm_run_i8(c,wref,M,A,Ce); ork_mm_run_i8(c,wrt,M,A,Cr);
        long mism=0; for(size_t e=0;e<(size_t)M*N;e++) if(Cr[e]!=Ce[e]){ mism++; }
        double t0=now_us(); for(int i=0;i<iters;i++) ork_mm_run_i8(c,wrt,M,A,Cr);  double us=(now_us()-t0)/iters;
        double t1=now_us(); for(int i=0;i<iters;i++) ork_mm_run_i8(c,wref,M,A,Ce); double un=(now_us()-t1)/iters;
        printf("  M=%-3d routed %8.1f us | normal %8.1f us | %.2fx  mism=%ld  (%s)\n",
               M, us, un, un/us, mism, M<=64?"fold expected":"normal expected");
        free(Cr); free(Ce);
      }
      ork_mm_free(c,wref); ork_mm_free(c,wrt); free(W); free(A); free(blob);
    }
    /* ---- SHARED-INPUT BATCH: QKV (q N=3584, k/v N=512) share input A. Batch fold (one A-marshal) vs 3 separate
     * normal ork_mm_run_i8. bit-exact + timing. This is where amortizing the nc16 marshal should pay off. ---- */
    { int K3=3584; int QN[3]={3584,512,512}; const char*nm[3]={"q","k","v"};
      int8_t *Wq[3], *A=malloc((size_t)128*K3);
      ork_w *wn[3], *wfd[3]; int ok=1;
      for(size_t i=0;i<(size_t)128*K3;i++) A[i]=r8();
      for(int j=0;j<3;j++){ Wq[j]=malloc((size_t)K3*QN[j]); for(size_t i=0;i<(size_t)K3*QN[j];i++) Wq[j][i]=r8();
        wn[j]=ork_mm_pack_i8(c,K3,QN[j],Wq[j]);
        size_t fb=ork_w_dump_fold_i8_cpu(c,K3,QN[j],Wq[j],NULL,0); void*bl=malloc(fb); ork_w_dump_fold_i8_cpu(c,K3,QN[j],Wq[j],bl,fb);
        wfd[j]=ork_mm_load_fold_i8(c,K3,QN[j],bl,fb); free(bl);
        if(!wn[j]||!wfd[j]) ok=0; }
      /* FUSED baseline = the SHIPPING ggml-ork group fusion: concat q|k|v along N (=4608), quantize A once,
       * ONE normal matmul. This is what the fold-batch must actually beat (not 3x separate). */
      int Ncat=QN[0]+QN[1]+QN[2]; int8_t*Wcat=malloc((size_t)K3*Ncat); ork_w*wcat=NULL;
      if(ok){ for(int k=0;k<K3;k++){ int o2=0; for(int j=0;j<3;j++){ for(int n=0;n<QN[j];n++) Wcat[(size_t)k*Ncat+o2+n]=Wq[j][(size_t)k*QN[j]+n]; o2+=QN[j]; } }
        wcat=ork_mm_pack_i8(c,K3,Ncat,Wcat); if(!wcat) ok=0; }
      printf("\n== SHARED-INPUT BATCH (QKV) vs 3x-separate AND vs FUSED-wide(N=%d, shipping) ==\n", Ncat);
      int BM[]={8,16,32,64};
      for(size_t mi=0; ok && mi<sizeof BM/sizeof*BM; mi++){ int M=BM[mi];
        int32_t *Cf[3],*Cn[3]; for(int j=0;j<3;j++){ Cf[j]=calloc((size_t)M*QN[j],4); Cn[j]=calloc((size_t)M*QN[j],4); }
        int32_t *Ccat=calloc((size_t)M*Ncat,4);
        double us_b=0; ork_w*ws[3]={wfd[0],wfd[1],wfd[2]}; int32_t*Co[3]={Cf[0],Cf[1],Cf[2]};
        int rc=ork_npu_fold_batch_w(c,3,ws,M,A,Co,60,&us_b);
        for(int j=0;j<3;j++) ork_mm_run_i8(c,wn[j],M,A,Cn[j]);
        long mism=0; for(int j=0;j<3;j++) for(size_t e=0;e<(size_t)M*QN[j];e++) if(Cf[j][e]!=Cn[j][e]){ mism++; break; }
        double t0=now_us(); for(int i=0;i<60;i++) for(int j=0;j<3;j++) ork_mm_run_i8(c,wn[j],M,A,Cn[j]); double us_3=(now_us()-t0)/60;
        double t1=now_us(); for(int i=0;i<60;i++) ork_mm_run_i8(c,wcat,M,A,Ccat); double us_c=(now_us()-t1)/60;
        printf("  M=%-3d  batch-fold %7.1f | 3x-sep %7.1f (%.2fx) | fused-wide %7.1f (%.2fx)  mism=%ld rc=%d\n",
               M, us_b, us_3, us_3/us_b, us_c, us_c/us_b, mism, rc);
        for(int j=0;j<3;j++){ free(Cf[j]); free(Cn[j]); } free(Ccat);
      }
      if(wcat) ork_mm_free(c,wcat); free(Wcat);
      for(int j=0;j<3;j++){ if(wn[j])ork_mm_free(c,wn[j]); if(wfd[j])ork_mm_free(c,wfd[j]); free(Wq[j]); } free(A);
    }
    /* ---- gate+up batch (FFN, both N=18944, share input) — the FLOP bulk. Expected ~wash (DRAM-BW-bound). ---- */
    { int K3=3584, GN=18944; int8_t *Wg[2], *A=malloc((size_t)128*K3); ork_w *wn[2],*wfd[2]; int ok=1;
      for(size_t i=0;i<(size_t)128*K3;i++) A[i]=r8();
      for(int j=0;j<2;j++){ Wg[j]=malloc((size_t)K3*GN); for(size_t i=0;i<(size_t)K3*GN;i++) Wg[j][i]=r8();
        wn[j]=ork_mm_pack_i8(c,K3,GN,Wg[j]);
        size_t fb=ork_w_dump_fold_i8_cpu(c,K3,GN,Wg[j],NULL,0); void*bl=malloc(fb); ork_w_dump_fold_i8_cpu(c,K3,GN,Wg[j],bl,fb);
        wfd[j]=ork_mm_load_fold_i8(c,K3,GN,bl,fb); free(bl); if(!wn[j]||!wfd[j]) ok=0; }
      printf("\n== SHARED-INPUT BATCH (gate+up: 2x N=18944) ==\n");
      int BM[]={8,32};
      for(size_t mi=0; ok && mi<sizeof BM/sizeof*BM; mi++){ int M=BM[mi];
        int32_t *Cf[2],*Cn[2]; for(int j=0;j<2;j++){ Cf[j]=calloc((size_t)M*GN,4); Cn[j]=calloc((size_t)M*GN,4); }
        double us_b=0; ork_w*ws[2]={wfd[0],wfd[1]}; int32_t*Co[2]={Cf[0],Cf[1]};
        int rc=ork_npu_fold_batch_w(c,2,ws,M,A,Co,20,&us_b);
        for(int j=0;j<2;j++) ork_mm_run_i8(c,wn[j],M,A,Cn[j]);
        long mism=0; for(int j=0;j<2;j++) for(size_t e=0;e<(size_t)M*GN;e++) if(Cf[j][e]!=Cn[j][e]){ mism++; break; }
        double t0=now_us(); for(int i=0;i<20;i++) for(int j=0;j<2;j++) ork_mm_run_i8(c,wn[j],M,A,Cn[j]); double us_n=(now_us()-t0)/20;
        printf("  M=%-3d  batch-fold %8.1f us | 2x normal %8.1f us | %.2fx  rc=%d mism=%ld\n", M, us_b, us_n, us_n/us_b, rc, mism);
        for(int j=0;j<2;j++){ free(Cf[j]); free(Cn[j]); }
      }
      for(int j=0;j<2;j++){ if(wn[j])ork_mm_free(c,wn[j]); if(wfd[j])ork_mm_free(c,wfd[j]); free(Wg[j]); } free(A);
    }
    printf("\nDONE\n");
    return 0;
}
