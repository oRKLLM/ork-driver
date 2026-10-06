/* attn_bwd_fp64_ref — the known-good reference I should have written before any of the checks.
 *
 * The NPU attention-backward probe reports dV correct but dQ/dK failing, inconsistently (4.7e-01 at one
 * init, 1.4e-01 and 3.8e+01 at another). Three different things could cause that and the existing test
 * cannot tell them apart:
 *     (a) the FORMULA for dQ/dK is wrong;
 *     (b) the formula is right but the NPU path computes it wrongly;
 *     (c) both are right and the FINITE-DIFFERENCE CHECK is unreliable — plausible, because it
 *         differentiates an fp16-rounded forward and divides by max(|fd|,1e-4), which inflates the
 *         relative error without bound wherever the true derivative is near zero.
 *
 * This tool implements the WHOLE forward and backward in fp64 on the CPU and runs two checks that are
 * independent of the NPU and of fp16 entirely:
 *     check A   fp64 analytic gradient  vs  fp64 central differences of the fp64 loss
 *               -> isolates (a). If this fails, the formula is wrong and nothing else matters.
 *     check B   the same, but with the relative-error FLOOR removed and |fd| reported alongside
 *               -> isolates (c). If A passes only because of the floor, this exposes it.
 *
 * CPU-only, no NPU, no board required. Once A passes, this becomes the oracle the NPU path is compared
 * against, which is a far stronger test than differencing a rounded forward.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int T,D; static double scale;
static double *Q,*K,*V,*S,*P,*O,*dP,*dS,*dQ,*dK,*dV;
static unsigned s_=31415;
static double rnd(void){ s_=s_*1103515245u+12345u; return (double)((int)((s_>>13)&0x7fff)-16384)/16384.0; }

/* full fp64 forward; returns L = ½‖O‖² */
static double fwd(void){
    for(int i=0;i<T;i++){
        double mx=-1e300;
        for(int j=0;j<T;j++){ double a=0; for(int k=0;k<D;k++) a+=Q[(size_t)i*D+k]*K[(size_t)j*D+k];
            a*=scale; S[(size_t)i*T+j]=a; if(a>mx)mx=a; }
        double sum=0;
        for(int j=0;j<T;j++){ double e=exp(S[(size_t)i*T+j]-mx); P[(size_t)i*T+j]=e; sum+=e; }
        for(int j=0;j<T;j++) P[(size_t)i*T+j]/=sum;
    }
    for(int i=0;i<T;i++) for(int k=0;k<D;k++){ double a=0;
        for(int j=0;j<T;j++) a+=P[(size_t)i*T+j]*V[(size_t)j*D+k]; O[(size_t)i*D+k]=a; }
    double L=0; for(size_t i=0;i<(size_t)T*D;i++) L+=O[i]*O[i];
    return 0.5*L;
}

/* full fp64 backward from dO = O */
static void bwd(void){
    for(int j=0;j<T;j++) for(int k=0;k<D;k++){ double a=0;
        for(int i=0;i<T;i++) a+=P[(size_t)i*T+j]*O[(size_t)i*D+k]; dV[(size_t)j*D+k]=a; }        /* dV = Pᵀ·dO */
    for(int i=0;i<T;i++) for(int j=0;j<T;j++){ double a=0;
        for(int k=0;k<D;k++) a+=O[(size_t)i*D+k]*V[(size_t)j*D+k]; dP[(size_t)i*T+j]=a; }        /* dP = dO·Vᵀ */
    for(int i=0;i<T;i++){ double r=0;
        for(int j=0;j<T;j++) r+=dP[(size_t)i*T+j]*P[(size_t)i*T+j];
        for(int j=0;j<T;j++) dS[(size_t)i*T+j]=P[(size_t)i*T+j]*(dP[(size_t)i*T+j]-r)*scale; }   /* softmax bwd */
    for(int i=0;i<T;i++) for(int k=0;k<D;k++){ double a=0;
        for(int j=0;j<T;j++) a+=dS[(size_t)i*T+j]*K[(size_t)j*D+k]; dQ[(size_t)i*D+k]=a; }       /* dQ = dS·K  */
    for(int j=0;j<T;j++) for(int k=0;k<D;k++){ double a=0;
        for(int i=0;i<T;i++) a+=dS[(size_t)i*T+j]*Q[(size_t)i*D+k]; dK[(size_t)j*D+k]=a; }       /* dK = dSᵀ·Q */
}

/* central differences of the fp64 loss. Reports the relative error BOTH with the usual floor and
 * unfloored, plus the raw magnitudes — so a "huge relative error" caused by a near-zero derivative is
 * visibly distinguishable from a genuinely wrong gradient. */
static void check(double *master,const double *grad,const char *name,double eps){
    double worst_f=0, worst_u=0; int n=T*D;
    printf("  d%-2s  %-13s %-13s %-11s %-11s\n", name, "analytic", "finite diff", "relerr", "relerr(floored)");
    for(int s=0;s<6;s++){
        int idx=(s*n)/6+s; if(idx>=n) idx=n-1;
        double sv=master[idx];
        master[idx]=sv+eps; double Lp=fwd();
        master[idx]=sv-eps; double Lm=fwd();
        master[idx]=sv;     fwd();
        double fd=(Lp-Lm)/(2*eps), an=grad[idx];
        double ru = fabs(fd)>0 ? fabs(an-fd)/fabs(fd) : (fabs(an)>0?1e9:0);
        double rf = fabs(an-fd)/(fabs(fd)>1e-4?fabs(fd):1e-4);
        if(rf>worst_f) worst_f=rf; if(ru>worst_u) worst_u=ru;
        printf("       %-13.3e %-13.3e %-11.3e %-11.3e\n", an, fd, ru, rf);
    }
    printf("       worst unfloored %.3e   worst floored %.3e   %s\n\n",
           worst_u, worst_f, worst_u<1e-4?"FORMULA CORRECT":"FORMULA SUSPECT");
}

int main(int argc,char**argv){
    T = argc>1?atoi(argv[1]):128;
    D = argc>2?atoi(argv[2]):64;
    double init = argc>3?atof(argv[3]):2.0;
    double eps  = argc>4?atof(argv[4]):1e-5;      /* fp64 allows a MUCH smaller eps than fp16 did */
    scale = 1.0/sqrt((double)D);
    size_t td=(size_t)T*D, tt=(size_t)T*T;
    Q=malloc(td*8); K=malloc(td*8); V=malloc(td*8); O=malloc(td*8);
    dQ=malloc(td*8); dK=malloc(td*8); dV=malloc(td*8);
    S=malloc(tt*8); P=malloc(tt*8); dP=malloc(tt*8); dS=malloc(tt*8);
    if(!Q||!dS){ printf("alloc failed\n"); return 1; }
    for(size_t i=0;i<td;i++){ Q[i]=rnd()*init; K[i]=rnd()*init; V[i]=rnd()*0.4; }

    printf("fp64 reference attention backward   T=%d D=%d init=%.2f eps=%g   (CPU only, no NPU)\n\n",T,D,init,eps);
    double L=fwd(); bwd();
    printf("loss %.9f\n\n", L);
    check(V,dV,"V",eps);
    check(Q,dQ,"Q",eps);
    check(K,dK,"K",eps);
    printf("If all three say FORMULA CORRECT, the backward maths is right and the NPU probe's dQ/dK\n"
           "failures are either the fp16 path or the fp16-forward finite-difference check — and the\n"
           "unfloored-vs-floored columns above show whether the relative-error floor was inflating them.\n");
    return 0;
}
