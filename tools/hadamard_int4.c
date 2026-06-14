/* tools/hadamard_int4.c — does a Hadamard rotation make int4 (W4A4) accurate? (Tier 4a)
 *
 * Pure CPU math, NO NPU — measures the QUANTIZATION error of an int4 matmul vs the fp32 reference,
 * with and without a Hadamard rotation. The rotation Q = H/sqrt(K) (H the K×K Walsh-Hadamard matrix)
 * is orthonormal and symmetric, so A·B = (A·Q)·(Q·B) exactly — but the rotated A',B' have their
 * outliers spread across channels, so per-group int4 quant of A',B' is far more accurate. This is the
 * QuaRot/SpinQuant idea; here we just prove it cuts our int4 error before doing the model-pipeline work.
 *
 *   cc -O2 -o hadamard_int4 tools/hadamard_int4.c -lm && ./hadamard_int4 [M] [K] [N] [G]   (K power of 2)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static float frand(void){ return ((float)rand()/(float)RAND_MAX)*2.0f - 1.0f; }
/* standard normal (Box-Muller) — real weights/activations are ~Gaussian with heavy-tailed outliers */
static float grand(void){ float u1=((float)rand()+1)/((float)RAND_MAX+2), u2=((float)rand()+1)/((float)RAND_MAX+2);
    return sqrtf(-2*logf(u1))*cosf(6.2831853f*u2); }

/* in-place fast Walsh-Hadamard transform on a length-n vector (n a power of 2), then *1/sqrt(n)
 * so the operator is the orthonormal Q (Q*Q = I). */
static void fwht_norm(float *v, int n){
    for(int len=1; len<n; len<<=1)
        for(int i=0;i<n;i+=len<<1)
            for(int j=i;j<i+len;j++){ float a=v[j], b=v[j+len]; v[j]=a+b; v[j+len]=a-b; }
    float s=1.0f/sqrtf((float)n);
    for(int i=0;i<n;i++) v[i]*=s;
}

/* per-group symmetric int quant+dequant of a length-K vector (group size G, `bits`-bit), returns the
 * dequantized approximation in `out`. e.g. bits=4 -> [-7,7], bits=8 -> [-127,127]. */
static void q_dequant_vec(const float *x, float *out, int K, int G, int bits){
    int lim=(1<<(bits-1))-1;   /* 7 for int4, 127 for int8 */
    for(int g=0; g<K; g+=G){
        float mx=1e-9f; for(int k=0;k<G;k++){ float a=fabsf(x[g+k]); if(a>mx)mx=a; }
        float s=mx/lim, inv=lim/mx;
        for(int k=0;k<G;k++){ int q=(int)lrintf(x[g+k]*inv); if(q>lim)q=lim; if(q<-lim)q=-lim; out[g+k]=q*s; }
    }
}

/* RMS relative error of a quantized matmul C=Aq·Bq vs the fp32 reference A·B. `abits`/`bbits` set the
 * activation/weight bit-width. If `rotate`, apply the Hadamard rotation to A (rows) and B (columns)
 * along K before quantizing (the product is identical in fp32, but outliers are spread out). */
static double matmul_q_err(const float *A, const float *B, const float *Cref, int M, int K, int N, int G, int abits, int bbits, int rotate){
    float *Aq=malloc((size_t)M*K*4), *Bq=malloc((size_t)K*N*4);
    float *ra=malloc((size_t)K*4), *rb=malloc((size_t)K*4);
    for(int m=0;m<M;m++){ const float *row=A+(size_t)m*K;
        if(rotate){ memcpy(ra,row,(size_t)K*4); fwht_norm(ra,K); q_dequant_vec(ra,Aq+(size_t)m*K,K,G,abits); }
        else q_dequant_vec(row,Aq+(size_t)m*K,K,G,abits); }
    for(int n=0;n<N;n++){
        for(int k=0;k<K;k++) rb[k]=B[(size_t)k*N+n];           /* gather column n */
        if(rotate) fwht_norm(rb,K);
        float *col=malloc((size_t)K*4); q_dequant_vec(rb,col,K,G,bbits);
        for(int k=0;k<K;k++) Bq[(size_t)k*N+n]=col[k]; free(col);
    }
    double num=0,den=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){
        double c=0; for(int k=0;k<K;k++) c+=(double)Aq[(size_t)m*K+k]*Bq[(size_t)k*N+n];
        double r=Cref[(size_t)m*N+n], e=c-r; num+=e*e; den+=r*r; }
    free(Aq);free(Bq);free(ra);free(rb);
    return den>0? sqrt(num/den):0;
}

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):16, K=argc>2?atoi(argv[2]):2048, N=argc>3?atoi(argv[3]):512, G=argc>4?atoi(argv[4]):128;
    if(K&(K-1)){ printf("K must be a power of 2 for the plain Hadamard (got %d)\n",K); return 1; }
    srand(1234);
    float *A=malloc((size_t)M*K*4), *B=malloc((size_t)K*N*4), *Cref=malloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=grand();
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=grand();
    /* inject ACTIVATION outliers (the thing Hadamard is supposed to tame): a few K-channels 20x larger */
    for(int k=0;k<K;k+=37) for(int m=0;m<M;m++) A[(size_t)m*K+k]*=20.0f;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double s=0; for(int k=0;k<K;k++) s+=(double)A[(size_t)m*K+k]*B[(size_t)k*N+n]; Cref[(size_t)m*N+n]=s; }
    printf("matmul quant RMS rel err, M=%d K=%d N=%d G=%d (Gaussian + injected activation outliers):\n",M,K,N,G);
    printf("  W8A8 (int8/int8)            sanity : %.3f%%\n", 100*matmul_q_err(A,B,Cref,M,K,N,G,8,8,0));
    printf("  W4A4 (int4/int4)  plain            : %.3f%%\n", 100*matmul_q_err(A,B,Cref,M,K,N,G,4,4,0));
    printf("  W4A4 (int4/int4)  + Hadamard       : %.3f%%\n", 100*matmul_q_err(A,B,Cref,M,K,N,G,4,4,1));
    printf("  W4A8 (act8/wt4)   plain  [not HW]  : %.3f%%\n", 100*matmul_q_err(A,B,Cref,M,K,N,G,8,4,0));
    printf("  W4A8 (act8/wt4)   + Hadamard [n/HW]: %.3f%%\n", 100*matmul_q_err(A,B,Cref,M,K,N,G,8,4,1));
    free(A);free(B);free(Cref); return 0;
}
