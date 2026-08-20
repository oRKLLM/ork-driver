/* test_rsqrt_lut — validate the fused on-NPU rsqrt: a reduce-matmul (sq·(-S)) with a rsqrt fused-output LUT
 * emits scale = 1/sqrt(ss/n+eps) directly (ss = sum x^2). Compares the NPU-emitted scale to the CPU rsqrt
 * reference. This is the last piece for a FULLY-on-NPU chained rmsnorm (reduce+rsqrt in one submit).
 * Board-only. Build: cc -O2 -Iinclude -Isrc -pthread -o test_rsqrt_lut tools/re/test_rsqrt_lut.c src/*.c ... -lm */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

int main(void){
    srand(7);
    ork_npu *npu = ork_npu_init();
    if(!npu){ fprintf(stderr,"init failed\n"); return 2; }
    const int M=8, n=2048; const double eps=1e-6;

    float *x = malloc((size_t)M*n*4);
    for(size_t i=0;i<(size_t)M*n;i++) x[i]=(rand()/(float)RAND_MAX)*2.f-1.f;  /* ~U(-1,1) */

    /* CPU reference: ss[m]=Σx², scale_ref[m]=1/sqrt(ss/n+eps); + the ss range for LUT calibration */
    double *ss=malloc((size_t)M*8), ss_min=1e30, ss_max=0; float *sref=malloc((size_t)M*4);
    for(int m=0;m<M;m++){ double s=0; for(int i=0;i<n;i++){ double v=x[(size_t)m*n+i]; s+=v*v; } ss[m]=s;
        if(s<ss_min)ss_min=s; if(s>ss_max)ss_max=s; sref[m]=(float)(1.0/sqrt(s/n+eps)); }
    fprintf(stderr,"ss range [%.1f, %.1f], scale_ref ~ %.4f\n", ss_min, ss_max, sref[0]);

    /* build the rsqrt LUT calibrated to the ss range (+10%% margin) */
    int16_t lut[1030]; double S=0,R=0,osc=0;
    int brc = ork_f16_mm_build_rsqrt_lut(npu, n, eps, ss_min*0.9, ss_max*1.1, lut, &S, &R, &osc);
    if(brc){ fprintf(stderr,"build_rsqrt_lut rc=%d (ppu-fuse off? -> SKIP)\n", brc); ork_npu_free(npu); return brc==-2?77:1; }
    fprintf(stderr,"LUT built: S=%.5f R=%.3f out_scale=%.6g\n", S, R, osc);

    /* runtime fused reduce+rsqrt: weight[n,16] = -S (acc=-S*Σsq=-S*ss); A=sq=x^2; C=R*LUT[idx] -> scale=C*osc */
    ork_f16 *B=malloc((size_t)n*16*2), *sq=malloc((size_t)M*n*2); float *C=malloc((size_t)M*16*4);
    for(size_t i=0;i<(size_t)n*16;i++) B[i]=(ork_f16)(-S);
    for(size_t i=0;i<(size_t)M*n;i++){ float v=x[i]; sq[i]=(ork_f16)(v*v); }
    ork_w *w = ork_f16_mm_pack(npu, n, 16, B);
    if(!w){ fprintf(stderr,"pack failed\n"); return 1; }
    int rrc = ork_f16_mm_run_silu(npu, w, M, sq, C, 0, 0xffffc000u, 0x56391100u, lut, 1030);
    if(rrc){ fprintf(stderr,"run_f16_silu rc=%d\n", rrc); ork_mm_free(npu,w); return 1; }

    double maxrel=0; int worst=0;
    for(int m=0;m<M;m++){ double snpu = (double)C[(size_t)m*16] * osc; double r=fabs(snpu-sref[m])/(fabs(sref[m])+1e-6);
        if(r>maxrel){maxrel=r;worst=m;}
        fprintf(stderr,"  m=%d ss=%.1f scale_npu=%.5f scale_ref=%.5f rel=%.4f\n", m, ss[m], snpu, sref[m], r); }
    ork_mm_free(npu,w);
    fprintf(stderr,"FUSED (K=n) maxrel=%.4f\n\n", maxrel);

    /* DECOUPLED K=32 path (same LUT/S/osc): feed ss directly as A[m,0], weight[32,16] row0=-S -> acc=-S*ss.
     * This is the n>2048 path; compare per-row to isolate the ~10%% bug vs the fused path above. */
    /* DECOUPLED via K=Kd=512 (the builder's known-good geometry; K=32 was degenerate -> acc~0 -> constant).
     * DENSE A = ss/ss_max ~ O(1) in every col, weight = -S*ss_max/Kd -> acc = sum_{k} (ss/ss_max)*(-S*ss_max/Kd)
     * = -S*ss. Normal-range A + small weight, exactly like the probe. */
    const int Kd=512; double G=ss_max;
    ork_f16 *Bd=malloc((size_t)Kd*16*2), *Ad=malloc((size_t)M*Kd*2); float *Cd=malloc((size_t)M*16*4);
    for(int i=0;i<Kd*16;i++) Bd[i]=(ork_f16)(-S*G/(double)Kd);
    for(int m=0;m<M;m++) for(int k=0;k<Kd;k++) Ad[(size_t)m*Kd+k]=(ork_f16)(ss[m]/G);
    ork_w *wd=ork_f16_mm_pack(npu,Kd,16,Bd);
    double dmax=0;
    if(wd && ork_f16_mm_run_silu(npu,wd,M,Ad,Cd,0,0xffffc000u,0x56391100u,lut,1030)==0){
        for(int m=0;m<M;m++){ double snpu=(double)Cd[(size_t)m*16]*osc; double r=fabs(snpu-sref[m])/(fabs(sref[m])+1e-6);
            if(r>dmax)dmax=r; fprintf(stderr,"  [K32] m=%d ss=%.1f scale_npu=%.5f ref=%.5f rel=%.4f\n",m,ss[m],snpu,sref[m],r); }
    } else fprintf(stderr,"[K32] run failed\n");
    fprintf(stderr,"DECOUPLED (K=32) maxrel=%.4f\n", dmax);
    if(wd) ork_mm_free(npu,wd);
    ork_npu_free(npu);
    int ok = maxrel < 0.05 && dmax < 0.05;
    fprintf(stderr, "\nRSQRT_LUT fused=%.4f decoupled=%.4f -> %s\n", maxrel, dmax, ok?"PASS":"FAIL");
    return ok?0:1;
}
