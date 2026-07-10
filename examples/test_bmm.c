/* test_bmm — self-validating test for the batched-dynamic-GEMM + normalization primitives, part of the
 * `make test` suite (the examples ARE the tests: each self-validates vs a CPU reference and exits 0/nonzero).
 *
 * Covers:
 *   ork_bmm_i8 / ork_bmm_i4 / ork_bmm_fp16  — batched dynamic GEMM (the attention / GDN-chunk primitive:
 *                                             C[b] = A[b][M,K]·B[b][K,N], BOTH operands dynamic). Per dtype
 *                                             it builds random per-batch A,B, computes a CPU reference, and
 *                                             asserts: i8/i4 EXACT (integer matmul), fp16 within tolerance.
 *   ork_npu_rmsnorm_f16 / ork_npu_l2norm_f16 — gated NPU norm entrypoints (CPU-correct today; the NPU-native
 *                                             SDP path is gated off until the LRN-template RE lands). Checked
 *                                             vs a double-precision reference.
 * Exits 0 on all-pass, 1 on any mismatch, 2 if the NPU is unavailable. */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ork_f16 is a native _Float16 — convert with plain casts */
static ork_f16 f2h(float f){ return (ork_f16)f; }
static float   h2f(ork_f16 h){ return (float)h; }

/* CPU/NEON norm primitive (neon_activations.h, linked via CORE) — exercised directly here */
extern void ork_l2norm_f32(float *o, const float *x, int n, float eps);

static int rint8(int lo,int hi){ return lo + rand()%(hi-lo+1); }

static int test_int(ork_npu *npu, int is_i4, int nbatch, int M, int K, int N){
    const char *nm = is_i4 ? "i4" : "i8";
    int8_t *A = malloc((size_t)nbatch*M*K), *B = malloc((size_t)nbatch*K*N);
    int32_t *C = calloc((size_t)nbatch*M*N, 4), *Cref = calloc((size_t)nbatch*M*N, 4);
    int lo = is_i4 ? -8 : -127, hi = is_i4 ? 7 : 127;
    for(size_t i=0;i<(size_t)nbatch*M*K;i++) A[i]=(int8_t)rint8(lo,hi);
    for(size_t i=0;i<(size_t)nbatch*K*N;i++) B[i]=(int8_t)rint8(lo,hi);
    for(int b=0;b<nbatch;b++)
        for(int m=0;m<M;m++) for(int n=0;n<N;n++){
            int32_t acc=0; for(int k=0;k<K;k++) acc += (int32_t)A[((size_t)b*M+m)*K+k]*(int32_t)B[((size_t)b*K+k)*N+n];
            Cref[((size_t)b*M+m)*N+n]=acc;
        }
    int rc = is_i4 ? ork_bmm_i4(npu,nbatch,M,K,N,A,B,C) : ork_bmm_i8(npu,nbatch,M,K,N,A,B,C);
    int fail = 0;
    if(rc){ fprintf(stderr,"[%s] ork_bmm rc=%d (nbatch=%d M=%d K=%d N=%d)\n",nm,rc,nbatch,M,K,N); fail=1; }
    else { size_t bad=0, first=(size_t)-1;
        for(size_t i=0;i<(size_t)nbatch*M*N;i++) if(C[i]!=Cref[i]){ if(first==(size_t)-1)first=i; bad++; }
        if(bad){ fprintf(stderr,"[%s] MISMATCH %zu/%zu (first idx %zu: got %d want %d)\n",nm,bad,(size_t)nbatch*M*N,first,C[first],Cref[first]); fail=1; }
        else fprintf(stderr,"[%s] OK nbatch=%d M=%d K=%d N=%d (exact)\n",nm,nbatch,M,K,N);
    }
    free(A);free(B);free(C);free(Cref); return fail;
}

static int test_fp16(ork_npu *npu, int nbatch, int M, int K, int N){
    ork_f16 *A = malloc((size_t)nbatch*M*K*2), *B = malloc((size_t)nbatch*K*N*2);
    float *C = calloc((size_t)nbatch*M*N,4), *Cref = calloc((size_t)nbatch*M*N,4);
    for(size_t i=0;i<(size_t)nbatch*M*K;i++) A[i]=f2h(((rand()/(float)RAND_MAX)*2.f-1.f));
    for(size_t i=0;i<(size_t)nbatch*K*N;i++) B[i]=f2h(((rand()/(float)RAND_MAX)*2.f-1.f));
    for(int b=0;b<nbatch;b++)
        for(int m=0;m<M;m++) for(int n=0;n<N;n++){
            float acc=0; for(int k=0;k<K;k++) acc += h2f(A[((size_t)b*M+m)*K+k])*h2f(B[((size_t)b*K+k)*N+n]);
            Cref[((size_t)b*M+m)*N+n]=acc;
        }
    int rc = ork_bmm_fp16(npu,nbatch,M,K,N,A,B,C);
    int fail = 0;
    if(rc){ fprintf(stderr,"[f16] ork_bmm_fp16 rc=%d\n",rc); fail=1; }
    else { double maxrel=0; size_t worst=0;
        for(size_t i=0;i<(size_t)nbatch*M*N;i++){ double d=fabs((double)C[i]-Cref[i]); double r=d/(fabs(Cref[i])+1e-3); if(r>maxrel){maxrel=r;worst=i;} }
        if(maxrel>0.03){ fprintf(stderr,"[f16] MISMATCH maxrel=%.4f at %zu (got %.4f want %.4f)\n",maxrel,worst,C[worst],Cref[worst]); fail=1; }
        else fprintf(stderr,"[f16] OK nbatch=%d M=%d K=%d N=%d (maxrel=%.4f)\n",nbatch,M,K,N,maxrel);
    }
    free(A);free(B);free(C);free(Cref); return fail;
}

static int test_norm(ork_npu *npu, int M, int n){
    ork_f16 *x=malloc((size_t)M*n*2), *w=malloc((size_t)n*2), *o=malloc((size_t)M*n*2);
    for(size_t i=0;i<(size_t)M*n;i++) x[i]=f2h((rand()/(float)RAND_MAX)*2.f-1.f);
    for(int i=0;i<n;i++) w[i]=f2h(0.5f+(rand()/(float)RAND_MAX));
    int fail=0;
    /* rmsnorm */
    if(ork_npu_rmsnorm_f16(npu,M,n,x,w,1e-6f,o)){ fprintf(stderr,"[rmsnorm] rc!=0\n"); fail=1; }
    else { double maxrel=0;
        for(int m=0;m<M;m++){ double ss=0; for(int i=0;i<n;i++){double v=h2f(x[m*n+i]);ss+=v*v;} double s=1.0/sqrt(ss/n+1e-6);
            for(int i=0;i<n;i++){ double ref=h2f(x[m*n+i])*s*h2f(w[i]); double r=fabs(h2f(o[m*n+i])-ref)/(fabs(ref)+1e-3); if(r>maxrel)maxrel=r; } }
        if(maxrel>0.03){ fprintf(stderr,"[rmsnorm] MISMATCH maxrel=%.4f\n",maxrel); fail=1; } else fprintf(stderr,"[rmsnorm] OK M=%d n=%d (maxrel=%.4f)\n",M,n,maxrel); }
    /* l2norm */
    if(ork_npu_l2norm_f16(npu,M,n,x,1e-6f,o)){ fprintf(stderr,"[l2norm] rc!=0\n"); fail=1; }
    else { double maxrel=0;
        for(int m=0;m<M;m++){ double ss=0; for(int i=0;i<n;i++){double v=h2f(x[m*n+i]);ss+=v*v;} double s=1.0/sqrt(ss+1e-6);
            for(int i=0;i<n;i++){ double ref=h2f(x[m*n+i])*s; double r=fabs(h2f(o[m*n+i])-ref)/(fabs(ref)+1e-3); if(r>maxrel)maxrel=r; } }
        if(maxrel>0.03){ fprintf(stderr,"[l2norm] MISMATCH maxrel=%.4f\n",maxrel); fail=1; } else fprintf(stderr,"[l2norm] OK M=%d n=%d (maxrel=%.4f)\n",M,n,maxrel); }
    free(x);free(w);free(o); return fail;
}

/* ork_l2norm_f32 (NEON CPU): o = x/sqrt(sum(x^2)+eps). Validate vs a double-precision scalar reference. */
static int test_l2norm_f32(int n){
    float *x=malloc((size_t)n*4), *o=malloc((size_t)n*4);
    for(int i=0;i<n;i++) x[i]=(rand()/(float)RAND_MAX)*2.f-1.f;
    ork_l2norm_f32(o,x,n,1e-6f);
    double ss=0; for(int i=0;i<n;i++) ss+=(double)x[i]*x[i]; double s=1.0/sqrt(ss+1e-6);
    double maxrel=0; for(int i=0;i<n;i++){ double ref=x[i]*s; double r=fabs(o[i]-ref)/(fabs(ref)+1e-6); if(r>maxrel)maxrel=r; }
    free(x);free(o);
    if(maxrel>1e-4){ fprintf(stderr,"[l2norm_f32] MISMATCH maxrel=%.6f\n",maxrel); return 1; }
    fprintf(stderr,"[l2norm_f32] OK n=%d (maxrel=%.6f)\n",n,maxrel); return 0;
}

/* Fused on-NPU reduce+rsqrt (ork_mm_build_f16_rsqrt_lut): a reduce-matmul emits scale=1/sqrt(ss/n+eps)
 * directly. SKIPs (returns 0) if the PPU fused-output path is unavailable (non-rk3588) — keeps make test
 * portable. n<=2048 (fp16 single-tile). Compares NPU-emitted scale to CPU 1/sqrt. */
static int test_rsqrt(ork_npu *npu, int M, int n){
    const double eps=1e-6;
    float *x=malloc((size_t)M*n*4); for(size_t i=0;i<(size_t)M*n;i++) x[i]=(rand()/(float)RAND_MAX)*2.f-1.f;
    double ssmin=1e30,ssmax=0; float *sref=malloc((size_t)M*4); double *ss=malloc((size_t)M*8);
    for(int m=0;m<M;m++){ double s=0; for(int i=0;i<n;i++){double v=x[(size_t)m*n+i];s+=v*v;} ss[m]=s;
        if(s<ssmin)ssmin=s; if(s>ssmax)ssmax=s; sref[m]=(float)(1.0/sqrt(s/n+eps)); }
    int16_t lut[1030]; double S=0,R=0,osc=0;
    int brc=ork_mm_build_f16_rsqrt_lut(npu,n,eps,ssmin*0.95,ssmax*1.05,lut,&S,&R,&osc);
    if(brc==-2){ fprintf(stderr,"[rsqrt-lut] SKIP (PPU fused-output unavailable)\n"); free(x);free(sref);free(ss); return 0; }
    int fail=0;
    if(brc){ fprintf(stderr,"[rsqrt-lut] build rc=%d\n",brc); fail=1; }
    else {
        ork_f16 *B=malloc((size_t)n*16*2), *sq=malloc((size_t)M*n*2); float *C=malloc((size_t)M*16*4);
        for(size_t i=0;i<(size_t)n*16;i++) B[i]=(ork_f16)(-S);
        for(size_t i=0;i<(size_t)M*n;i++){ float v=x[i]; sq[i]=(ork_f16)(v*v); }
        ork_w *w=ork_mm_pack(npu,n,16,B);
        if(!w || ork_mm_run_f16_silu(npu,w,M,sq,C,0,0xffffc000u,0x56391100u,lut,1030)){ fprintf(stderr,"[rsqrt-lut] run failed\n"); fail=1; }
        else { double maxrel=0; for(int m=0;m<M;m++){ double snpu=(double)C[(size_t)m*16]*osc; double r=fabs(snpu-sref[m])/(fabs(sref[m])+1e-6); if(r>maxrel)maxrel=r; }
            if(maxrel>0.03){ fprintf(stderr,"[rsqrt-lut fused] MISMATCH maxrel=%.4f\n",maxrel); fail=1; } else fprintf(stderr,"[rsqrt-lut fused] OK M=%d n=%d (maxrel=%.4f)\n",M,n,maxrel); }
        if(w) ork_mm_free(npu,w); free(B);free(sq);free(C);
        /* DECOUPLED K=512 rsqrt (the any-n path): feed ss dense/normalized, weight -S*G/Kd -> acc=-S*ss */
        const int Kd=512; double Gd=ssmax;
        ork_f16 *Bd=malloc((size_t)Kd*16*2), *Ad=malloc((size_t)M*Kd*2); float *Cd=malloc((size_t)M*16*4);
        for(int i=0;i<Kd*16;i++) Bd[i]=(ork_f16)(-S*Gd/(double)Kd);
        for(int m=0;m<M;m++){ ork_f16 v=(ork_f16)(ss[m]/Gd); for(int k=0;k<Kd;k++) Ad[(size_t)m*Kd+k]=v; }
        ork_w *wd=ork_mm_pack(npu,Kd,16,Bd);
        if(!wd || ork_mm_run_f16_silu(npu,wd,M,Ad,Cd,0,0xffffc000u,0x56391100u,lut,1030)){ fprintf(stderr,"[rsqrt-lut decoupled] run failed\n"); fail=1; }
        else { double dmax=0; for(int m=0;m<M;m++){ double snpu=(double)Cd[(size_t)m*16]*osc; double r=fabs(snpu-sref[m])/(fabs(sref[m])+1e-6); if(r>dmax)dmax=r; }
            if(dmax>0.03){ fprintf(stderr,"[rsqrt-lut decoupled] MISMATCH maxrel=%.4f\n",dmax); fail=1; } else fprintf(stderr,"[rsqrt-lut decoupled] OK K=512 (maxrel=%.4f)\n",dmax); }
        if(wd) ork_mm_free(npu,wd); free(Bd);free(Ad);free(Cd);
    }
    free(x);free(sref);free(ss); return fail;
}

int main(void){
    srand(1234);
    ork_npu *npu = ork_npu_init();
    if(!npu){ fprintf(stderr,"ork_npu_init failed (no NPU / no perms)\n"); return 2; }
    int fail = 0;
    /* attention-shaped: nbatch = heads, M = query len, K = head_dim, N = kv len */
    fail |= test_int (npu, 0, 8, 16, 128, 64);   /* i8  */
    fail |= test_int (npu, 0, 4, 32, 256, 32);   /* i8  another shape */
    fail |= test_int (npu, 1, 8, 16, 128, 64);   /* i4  (N%64) */
    fail |= test_fp16(npu,    8, 16, 128, 64);   /* fp16 */
    fail |= test_norm(npu, 8, 128);              /* rmsnorm + l2norm (reduce on NPU when ORK_NORM_NPU; else CPU) */
    fail |= test_norm(npu, 4, 4096);
    fail |= test_l2norm_f32(4096);               /* CPU/NEON l2norm primitive */
    fail |= test_rsqrt(npu, 8, 2048);            /* fused on-NPU reduce+rsqrt (rsqrt-LUT); skips if no PPU-fuse */
    ork_npu_free(npu);
    fprintf(stderr, fail ? "\nTEST_BMM: FAIL\n" : "\nTEST_BMM: PASS\n");
    return fail ? 1 : 0;
}
