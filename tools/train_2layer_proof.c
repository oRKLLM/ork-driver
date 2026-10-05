/* train_2layer_proof — backprop THROUGH a network on the RK3588 NPU, not one layer in isolation.
 *
 * train_step_proof showed a single linear layer trains correctly. That does not exercise the thing a
 * real network needs: the gradient flowing BACKWARD ACROSS a layer boundary. Here:
 *
 *     X ──W1──► H ──W2──► Y ,   L = ½‖Y‖²
 *
 *     dY  = Y                       (exact for this loss)
 *     dW2 = Hᵀ · dY                 NPU   (contraction over M)
 *     dH  = dY · W2ᵀ                NPU   ◄── THE CHAINING STEP (contraction over N)
 *     dW1 = Xᵀ · dH                 NPU   (contraction over M)
 *
 * Why the finite-difference check is on W1 specifically: perturbing W1 can only change the loss THROUGH
 * dH. If the chaining matmul is transposed, mis-strided, or stale, dW1 will still look like a plausible
 * matmul (check 1 passes) and the loss may even fall — but it will not match the numerical derivative of
 * the real loss. So check 2 on W1 is the actual proof of chaining; everything else is corroboration.
 *
 * This step also exercises the two transposes a real training loop pays every step and that a single-layer
 * test does not: Hᵀ (an ACTIVATION that changes every step) and W2ᵀ (a WEIGHT that changes every step).
 * Only Xᵀ is loop-invariant. All five NPU operands are resident weights refreshed with
 * ork_f16_mm_repack — no IOMMU alloc/free in the inner loop.
 *
 * fp32 master weights again mandatory: an SGD increment at a sane lr is below fp16 resolution, so an
 * in-place fp16 update rounds to zero and the loss sits flat while appearing to run.
 */
#include <ork_npu.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned s_=13579; static float rnd(void){ s_=s_*1103515245u+12345u; return (float)((int)((s_>>13)&0x7fff)-16384)/16384.0f; }

static int M,K1,K2,N;
static ork_f16 *Xh,*Xt,*W1h,*W2h,*W2t,*Hh,*Ht,*dYh,*dHh;
static float *W1_32,*W2_32,*H,*Y,*dW1,*dW2,*dH;
static ork_npu *c;
static ork_w *wW1,*wW2,*wW2t,*wdY,*wdH;

static void die(const char*m){ printf("%s\n",m); exit(1); }

static double forward(void){
    if(ork_f16_mm_repack(c,wW1,K1,K2,W1h)<0) die("repack W1");
    if(ork_f16_mm_run(c,wW1,M,Xh,H))         die("fwd L1");
    for(size_t i=0;i<(size_t)M*K2;i++) Hh[i]=(ork_f16)H[i];           /* H is the next layer's input */
    if(ork_f16_mm_repack(c,wW2,K2,N,W2h)<0)  die("repack W2");
    if(ork_f16_mm_run(c,wW2,M,Hh,Y))         die("fwd L2");
    double L=0; for(size_t i=0;i<(size_t)M*N;i++) L+=(double)Y[i]*Y[i];
    return 0.5*L;
}

static void backward(void){
    for(size_t i=0;i<(size_t)M*N;i++) dYh[i]=(ork_f16)Y[i];           /* dY = Y */

    for(int m=0;m<M;m++) for(int k=0;k<K2;k++) Ht[(size_t)k*M+m]=Hh[(size_t)m*K2+k];   /* Hᵀ: activation, per step */
    if(ork_f16_mm_repack(c,wdY,M,N,dYh)<0) die("repack dY");
    if(ork_f16_mm_run(c,wdY,K2,Ht,dW2))    die("dW2");                /* dW2 = Hᵀ·dY */

    for(int k=0;k<K2;k++) for(int n=0;n<N;n++) W2t[(size_t)n*K2+k]=W2h[(size_t)k*N+n]; /* W2ᵀ: weight, per step */
    if(ork_f16_mm_repack(c,wW2t,N,K2,W2t)<0) die("repack W2t");
    if(ork_f16_mm_run(c,wW2t,M,dYh,dH))      die("dH");               /* dH = dY·W2ᵀ  ◄── CHAINING */

    for(size_t i=0;i<(size_t)M*K2;i++) dHh[i]=(ork_f16)dH[i];
    if(ork_f16_mm_repack(c,wdH,M,K2,dHh)<0) die("repack dH");
    if(ork_f16_mm_run(c,wdH,K1,Xt,dW1))     die("dW1");               /* dW1 = Xᵀ·dH */
}

int main(int argc,char**argv){
    M  = argc>1?atoi(argv[1]):128;
    K1 = argc>2?atoi(argv[2]):512;
    K2 = argc>3?atoi(argv[3]):512;
    N  = argc>4?atoi(argv[4]):512;
    int steps = argc>5?atoi(argv[5]):12;
    double lr = argc>6?atof(argv[6]):1e-4;
    c = ork_npu_init(); if(!c) die("init failed");

    Xh=malloc((size_t)M*K1*2);  Xt=malloc((size_t)K1*M*2);
    W1h=malloc((size_t)K1*K2*2);W2h=malloc((size_t)K2*N*2); W2t=malloc((size_t)N*K2*2);
    Hh=malloc((size_t)M*K2*2);  Ht=malloc((size_t)K2*M*2);
    dYh=malloc((size_t)M*N*2);  dHh=malloc((size_t)M*K2*2);
    W1_32=malloc((size_t)K1*K2*4); W2_32=malloc((size_t)K2*N*4);
    H=malloc((size_t)M*K2*4); Y=malloc((size_t)M*N*4);
    dW1=malloc((size_t)K1*K2*4); dW2=malloc((size_t)K2*N*4); dH=malloc((size_t)M*K2*4);
    if(!Xh||!Xt||!W1h||!W2h||!W2t||!Hh||!Ht||!dYh||!dHh||!W1_32||!W2_32||!H||!Y||!dW1||!dW2||!dH) die("alloc");

    for(size_t i=0;i<(size_t)M*K1;i++) Xh[i]=(ork_f16)(rnd()*0.5f);
    for(size_t i=0;i<(size_t)K1*K2;i++){ W1_32[i]=rnd()*0.3f; W1h[i]=(ork_f16)W1_32[i]; }
    for(size_t i=0;i<(size_t)K2*N;i++){ W2_32[i]=rnd()*0.3f; W2h[i]=(ork_f16)W2_32[i]; }
    for(int m=0;m<M;m++) for(int k=0;k<K1;k++) Xt[(size_t)k*M+m]=Xh[(size_t)m*K1+k];   /* Xᵀ once: X is fixed */
    memset(dYh,0,(size_t)M*N*2); memset(dHh,0,(size_t)M*K2*2); memset(W2t,0,(size_t)N*K2*2);

    wW1=ork_f16_mm_pack(c,K1,K2,W1h); wW2=ork_f16_mm_pack(c,K2,N,W2h);
    wW2t=ork_f16_mm_pack(c,N,K2,W2t); wdY=ork_f16_mm_pack(c,M,N,dYh); wdH=ork_f16_mm_pack(c,M,K2,dHh);
    if(!wW1||!wW2||!wW2t||!wdY||!wdH) die("pack failed");

    printf("two layers  X[%d,%d] -W1-> H[%d,%d] -W2-> Y[%d,%d]   lr=%g steps=%d\n\n",M,K1,M,K2,M,N,lr,steps);
    double L0=forward(); backward();

    /* check 1: both gradients are the right matmul of their operands */
    double e1=0;
    for(int s=0;s<6;s++){
        int k=(s*K1)/6, n=(s*K2)/6+(s%2); if(n>=K2)n=K2-1;
        double r=0; for(int m=0;m<M;m++) r+=(double)(float)Xh[(size_t)m*K1+k]*(double)(float)dHh[(size_t)m*K2+n];
        double d=fabs(r)>1e-9?fabs(r):1e-9,e=fabs(dW1[(size_t)k*K2+n]-r)/d; if(e>e1)e1=e;
    }
    printf("check 1  dW1 vs fp64 (same fp16 operands)       max relerr %.3e   %s\n",e1,e1<1e-2?"PASS":"FAIL");

    /* check 2: THE CHAINING PROOF — W1 reaches the loss only through dH */
    double e2=0; const double eps=0.05;
    for(int s=0;s<5;s++){
        int k=(s*K1)/5,n=(s*K2)/5+(s%2); if(n>=K2)n=K2-1; size_t idx=(size_t)k*K2+n;
        float sv=W1_32[idx];
        W1_32[idx]=sv+(float)eps; W1h[idx]=(ork_f16)W1_32[idx]; double Lp=forward();
        W1_32[idx]=sv-(float)eps; W1h[idx]=(ork_f16)W1_32[idx]; double Lm=forward();
        W1_32[idx]=sv;            W1h[idx]=(ork_f16)sv;
        double fd=(Lp-Lm)/(2*eps),an=dW1[idx];
        double d=fabs(fd)>1e-6?fabs(fd):1e-6,e=fabs(an-fd)/d; if(e>e2)e2=e;
    }
    printf("check 2  dW1 vs finite differences (CHAINED)    max relerr %.3e   %s\n",e2,e2<0.20?"PASS":"FAIL");

    /* check 3: does the two-layer stack train? */
    forward(); backward();
    printf("\nstep        loss\n  0  %14.4f  (initial)\n",L0);
    double prev=L0; int mono=1;
    for(int t=1;t<=steps;t++){
        forward(); backward();
        for(size_t i=0;i<(size_t)K1*K2;i++){ W1_32[i]-=(float)(lr*dW1[i]); W1h[i]=(ork_f16)W1_32[i]; }
        for(size_t i=0;i<(size_t)K2*N;i++){  W2_32[i]-=(float)(lr*dW2[i]); W2h[i]=(ork_f16)W2_32[i]; }
        double L=forward();
        printf("%3d  %14.4f  %s\n",t,L,L<prev?"":"  <-- NOT decreasing");
        if(L>=prev) mono=0; prev=L;
    }
    printf("\ncheck 3  loss decreased every step              %s   (%.1f -> %.1f, %.1f%% of initial)\n",
           mono?"PASS":"FAIL",L0,prev,100.0*prev/L0);
    printf("\n%s\n",(e1<1e-2&&e2<0.20&&mono)
        ? "ALL THREE PASS — gradients flow correctly ACROSS a layer boundary on this NPU."
        : "NOT all checks passed — see above.");
    ork_mm_free(c,wW1); ork_mm_free(c,wW2); ork_mm_free(c,wW2t); ork_mm_free(c,wdY); ork_mm_free(c,wdH);
    ork_npu_free(c); return 0;
}
