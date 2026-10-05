/* train_step_proof — does a REAL training step work on this NPU? Correctness only; speed is irrelevant.
 *
 * This answers a different question from the rest of the scoping work. Throughput said no-go; the goal
 * here is a first CORRECT training loop even if it is 25x slower than ideal. Under that frame most of the
 * difficulty evaporates: dX and dW are ordinary matmuls, which ork-driver already computes bit-exactly in
 * its validated envelope, and everything else — loss, optimizer, master weights — runs on the CPU in fp32
 * where correctness is not in question.
 *
 * The loop, one linear layer, which is the irreducible unit of a training step:
 *     forward   Y  = X · W          (NPU, fp16 operands, fp32 output)
 *     loss      L  = ½‖Y‖²          (CPU fp32; minimised at W = 0, so convergence is checkable)
 *     grad out  dY = Y              (exact for this loss)
 *     backward  dW = Xᵀ · dY        (NPU — the matmul a training step cannot avoid)
 *     update    W32 -= lr · dW      (CPU fp32 MASTER weights, cast to fp16 per step)
 *
 * fp32 masters are not optional: an SGD increment at a sane lr is far below fp16 resolution, so updating
 * fp16 weights in place would silently round every step to zero and the loss would sit flat while looking
 * like it ran. The master/cast/repack cycle is what makes the step real.
 *
 * THREE CHECKS, because "the loss went down" alone is weak evidence:
 *   1. GRADIENT vs an fp64 CPU reference over the same fp16 operands — is dW the right tensor at all?
 *   2. GRADIENT vs CENTRAL FINITE DIFFERENCES of the actual loss — does it match the loss we minimise?
 *      (independent of 1: a consistent sign or transpose error passes 1 and fails this.)
 *   3. CONVERGENCE over N steps — loss must decrease monotonically toward 0.
 * A transposed or sign-flipped dW passes nothing here; a merely imprecise one passes all three.
 */
#include <ork_npu.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned s_=24680; static float rnd(void){ s_=s_*1103515245u+12345u; return (float)((int)((s_>>13)&0x7fff)-16384)/16384.0f; }

static int M,K,N;
static ork_f16 *Xh, *Wh, *Xt, *dYh;
static float *W32, *Y, *dW;
static ork_npu *ctx; static ork_w *wW, *wG;

/* forward on the NPU: Y = X·W, and the fp32 loss ½‖Y‖² */
static double forward(void){
    if (ork_f16_mm_repack(ctx,wW,K,N,Wh) < 0) { printf("repack W failed\n"); exit(1); }
    if (ork_f16_mm_run(ctx,wW,M,Xh,Y))        { printf("forward failed\n");  exit(1); }
    double L=0; for(size_t i=0;i<(size_t)M*N;i++) L += (double)Y[i]*Y[i];
    return 0.5*L;
}

/* backward on the NPU: dW = Xᵀ·dY, with dY = Y. Xᵀ is materialised on the CPU (no NPU transpose). */
static void backward(void){
    for(size_t i=0;i<(size_t)M*N;i++) dYh[i]=(ork_f16)Y[i];          /* dY = Y for L = ½‖Y‖² */
    if (ork_f16_mm_repack(ctx,wG,M,N,dYh) < 0) { printf("repack dY failed\n"); exit(1); }
    if (ork_f16_mm_run(ctx,wG,K,Xt,dW))        { printf("backward failed\n");  exit(1); }
}

int main(int argc,char**argv){
    M = argc>1?atoi(argv[1]):256;
    K = argc>2?atoi(argv[2]):1024;
    N = argc>3?atoi(argv[3]):1024;
    int steps = argc>4?atoi(argv[4]):12;
    double lr = argc>5?atof(argv[5]):2e-4;
    ctx = ork_npu_init(); if(!ctx){ printf("init failed\n"); return 1; }

    Xh=malloc((size_t)M*K*2); Wh=malloc((size_t)K*N*2); Xt=malloc((size_t)K*M*2); dYh=malloc((size_t)M*N*2);
    W32=malloc((size_t)K*N*4); Y=malloc((size_t)M*N*4); dW=malloc((size_t)K*N*4);
    if(!Xh||!Wh||!Xt||!dYh||!W32||!Y||!dW){ printf("alloc failed\n"); return 1; }

    for(size_t i=0;i<(size_t)M*K;i++) Xh[i]=(ork_f16)(rnd()*0.5f);
    for(size_t i=0;i<(size_t)K*N;i++){ W32[i]=rnd()*0.5f; Wh[i]=(ork_f16)W32[i]; }
    for(int m=0;m<M;m++) for(int k=0;k<K;k++) Xt[(size_t)k*M+m]=Xh[(size_t)m*K+k];   /* Xᵀ, once — X is fixed */

    wW = ork_f16_mm_pack(ctx,K,N,Wh);   /* resident; refreshed each step via repack */
    wG = ork_f16_mm_pack(ctx,M,N,dYh);  /* resident; dY is fresh each step */
    if(!wW||!wG){ printf("pack failed\n"); return 1; }

    printf("one linear layer  M=%d K=%d N=%d  lr=%g  steps=%d\n\n", M,K,N,lr,steps);
    double L0 = forward(); backward();

    /* --- check 1: dW vs fp64 over the same fp16 operands --- */
    double e1=0;
    for(int s=0;s<8;s++){
        int k=(s*K)/8, n=(s*N)/8+(s%3); if(n>=N)n=N-1;
        double r=0; for(int m=0;m<M;m++) r += (double)(float)Xh[(size_t)m*K+k]*(double)(float)dYh[(size_t)m*N+n];
        double d=fabs(r)>1e-9?fabs(r):1e-9, e=fabs(dW[(size_t)k*N+n]-r)/d; if(e>e1)e1=e;
    }
    printf("check 1  dW vs fp64 (same fp16 operands)      max relerr %.3e   %s\n", e1, e1<1e-2?"PASS":"FAIL");

    /* --- check 2: dW vs CENTRAL FINITE DIFFERENCES of the real loss --- */
    double e2=0; const double eps=0.05;   /* large: the probe differences an fp16-rounded forward */
    for(int s=0;s<5;s++){
        int k=(s*K)/5, n=(s*N)/5+(s%2); if(n>=N)n=N-1; size_t idx=(size_t)k*N+n;
        float save=W32[idx];
        W32[idx]=save+(float)eps; Wh[idx]=(ork_f16)W32[idx]; double Lp=forward();
        W32[idx]=save-(float)eps; Wh[idx]=(ork_f16)W32[idx]; double Lm=forward();
        W32[idx]=save;            Wh[idx]=(ork_f16)save;
        double fd=(Lp-Lm)/(2*eps), an=dW[idx];
        double den=fabs(fd)>1e-6?fabs(fd):1e-6, e=fabs(an-fd)/den; if(e>e2)e2=e;
    }
    printf("check 2  dW vs central finite differences     max relerr %.3e   %s\n", e2, e2<0.15?"PASS":"FAIL");

    /* --- check 3: does it actually TRAIN? --- */
    printf("\nstep        loss\n  0  %14.4f  (initial)\n", L0);
    double prev=L0; int mono=1;
    for(int t=1;t<=steps;t++){
        forward(); backward();
        for(size_t i=0;i<(size_t)K*N;i++){ W32[i] -= (float)(lr*dW[i]); Wh[i]=(ork_f16)W32[i]; }
        double L=forward();
        printf("%3d  %14.4f  %s\n", t, L, L<prev?"":"  <-- NOT decreasing");
        if(L>=prev) mono=0;
        prev=L;
    }
    printf("\ncheck 3  loss decreased every step            %s   (%.4f -> %.4f, %.1f%% of initial)\n",
           mono?"PASS":"FAIL", L0, prev, 100.0*prev/L0);
    printf("\n%s\n", (e1<1e-2 && e2<0.15 && mono)
        ? "ALL THREE PASS — a real training step runs correctly on this NPU."
        : "NOT all checks passed — see above.");
    ork_mm_free(ctx,wW); ork_mm_free(ctx,wG);
    ork_npu_free(ctx); return 0;
}
