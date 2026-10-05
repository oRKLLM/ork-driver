/* train_dw_bench — the gradient matmul dW = Xᵀ·dY, which is the one a training step cannot avoid and
 * the one inference never performs.
 *
 * Mapping onto ork-driver's primitive. ork_f16_mm_run computes C[M',N'] = A[M',K'] · B[K',N'] with B
 * packed. For dW[k,n] = Σ_m X[m,k]·dY[m,n] that is M'=K, K'=M, N'=N — so A is Xᵀ (must be materialised,
 * CPU) and B is dY (must be packed, every step, because dY is fresh each step).
 *
 * STRUCTURAL CONSEQUENCE, visible before any timing: the contraction dim of dW is the TOKEN count M.
 * orki_f16_sched(K) requires K to be a power of two AND < 2048, so with M=4096 tokens the dW matmul is
 * permanently in the sched=0 regime (mcap = 16384/4096 = 4 rows/tile) no matter what the layer's K and N
 * are. dW cannot reach the 11-bank envelope at realistic batch×seq. That is independent of, and in
 * addition to, the K=3584 cap problem in the forward pass.
 *
 * ERROR METHODOLOGY. "fp16 error" conflates two different effects, and for dW they differ by orders of
 * magnitude, so this probe separates them against an fp64 reference:
 *   acc_err   = NPU  vs  fp64 over the SAME fp16-cast operands  -> accumulation error alone
 *   total_err = NPU  vs  fp64 over the ORIGINAL fp32 operands   -> includes operand quantisation to fp16
 * The second is what training actually experiences. Gradients span decades and fp16's smallest normal is
 * 6.1e-5, so dY elements below that are flushed to subnormal or zero BEFORE the MAC ever runs — an error
 * no accumulator width can recover. An earlier version of this harness drew operands as multiples of
 * 2^-9, which fp32 accumulates exactly, and duly reported 0.00e+00 error; that measured the generator,
 * not the hardware. Operands here are log-uniform over a configurable decade span, signed.
 */
#include <ork_npu.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static unsigned s_=99991;
static double u01(void){ s_=s_*1103515245u+12345u; return (double)((s_>>8)&0xffffff)/16777216.0; }
/* log-uniform magnitude over [10^-lo_dec, 10^hi_dec], random sign — a realistic gradient distribution */
static float logu(double lo_dec,double hi_dec){
    double e = lo_dec + u01()*(hi_dec-lo_dec);
    double m = pow(10.0,e);
    return (float)(u01()<0.5 ? -m : m);
}

int main(int argc,char**argv){
    int M = argc>1?atoi(argv[1]):4096;     /* tokens = contraction dim of dW */
    int K = argc>2?atoi(argv[2]):1024;     /* dW rows */
    int N = argc>3?atoi(argv[3]):1024;     /* dW cols */
    double glo = argc>4?atof(argv[4]):-6.0, ghi = argc>5?atof(argv[5]):-2.0;  /* gradient decades */
    ork_npu *c = ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }

    printf("dW = X^T.dY   M(tokens)=%d K=%d N=%d   grad magnitudes 1e%.0f..1e%.0f\n", M,K,N,glo,ghi);
    printf("contraction dim K'=%d -> sched=%s, mcap=%d  => %d M-tiles for M'=%d\n\n",
           M, ((M&(M-1))==0 && M>=128 && M<2048)?"1":"0",
           ((M&(M-1))==0 && M>=128 && M<2048)?180224/M:16384/M,
           (K + (((M&(M-1))==0&&M>=128&&M<2048)?180224/M:16384/M) - 1) /
               (((M&(M-1))==0&&M>=128&&M<2048)?180224/M:16384/M), K);

    float *Xf=malloc((size_t)M*K*4), *Yf=malloc((size_t)M*N*4);
    ork_f16 *Xt=malloc((size_t)K*M*2), *Yh=malloc((size_t)M*N*2);
    float *C=malloc((size_t)K*N*4);
    if(!Xf||!Yf||!Xt||!Yh||!C){ printf("alloc failed\n"); return 1; }

    for(size_t i=0;i<(size_t)M*K;i++) Xf[i]=(float)(u01()*2.0-1.0);     /* activations ~ unit scale */
    for(size_t i=0;i<(size_t)M*N;i++) Yf[i]=logu(glo,ghi);              /* gradients span decades    */

    /* A = X^T, cast to fp16. This transpose is CPU work a training step pays every layer, every step. */
    double t0=now_us();
    for(int m=0;m<M;m++) for(int k=0;k<K;k++) Xt[(size_t)k*M+m]=(ork_f16)Xf[(size_t)m*K+k];
    double transpose_us=now_us()-t0;
    for(size_t i=0;i<(size_t)M*N;i++) Yh[i]=(ork_f16)Yf[i];

    t0=now_us(); ork_w *w=ork_f16_mm_pack(c,M,N,Yh); double pack_us=now_us()-t0;
    if(!w){ printf("pack failed (M%%32=%d N%%16=%d)\n",M%32,N%16); return 1; }
    t0=now_us(); int rc=ork_f16_mm_run(c,w,K,Xt,C); double run_us=now_us()-t0;
    if(rc){ printf("run failed rc=%d\n",rc); return 1; }

    /* fp64 references on a sample: one over the fp16 operands the NPU saw, one over the fp32 originals */
    double acc_max=0, tot_max=0, flushed=0;
    for(size_t i=0;i<(size_t)M*N;i++) if(Yf[i]!=0.0f && (float)(ork_f16)Yf[i]==0.0f) flushed++;
    for(int s=0;s<16;s++){
        int k=(s*K)/16, n=(s*N)/16+(s%5); if(n>=N)n=N-1;
        double r16=0, r32=0;
        for(int m=0;m<M;m++){
            r16 += (double)(float)Xt[(size_t)k*M+m] * (double)(float)Yh[(size_t)m*N+n];
            r32 += (double)Xf[(size_t)m*K+k]        * (double)Yf[(size_t)m*N+n];
        }
        double got=C[(size_t)k*N+n];
        double d16=fabs(r16)>1e-30?fabs(r16):1e-30, d32=fabs(r32)>1e-30?fabs(r32):1e-30;
        double a=fabs(got-r16)/d16, t=fabs(got-r32)/d32;
        if(a>acc_max) acc_max=a; if(t>tot_max) tot_max=t;
    }
    double gflops = 2.0*(double)M*K*N/(run_us*1e3);
    printf("%-26s %10.0f us\n","CPU transpose of X",transpose_us);
    printf("%-26s %10.0f us   (dY is fresh every step — unavoidable)\n","pack dY",pack_us);
    printf("%-26s %10.0f us   %8.1f GFLOP/s\n","NPU dW matmul",run_us,gflops);
    printf("%-26s %10.0f us\n","TOTAL per dW",transpose_us+pack_us+run_us);
    printf("\n%-26s %10.2e   (NPU vs fp64 over the SAME fp16 operands)\n","accumulation error",acc_max);
    printf("%-26s %10.2e   (NPU vs fp64 over the ORIGINAL fp32)\n","total error",tot_max);
    printf("%-26s %9.2f%%     (|dY| below fp16's 6.1e-5 normal minimum)\n","gradients flushed to 0",
           100.0*flushed/((double)M*N));
    ork_mm_free(c,w); free(Xf);free(Yf);free(Xt);free(Yh);free(C);
    ork_npu_free(c); return 0;
}
