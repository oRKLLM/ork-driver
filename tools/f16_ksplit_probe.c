/* f16_ksplit_probe — recover the fp16 K=3584 penalty with NO reverse engineering.
 *
 * Four attempts at poking the bank split directly got 2.4x but never bit-exactness, and two of them hung
 * the board. The residual proved bank-split-independent (~0.8% wrong at both 0x48 and 0x57, including the
 * value int8 itself emits at this K), so at least one more register is involved and nobody knows which.
 *
 * K-split sidesteps the whole question. int8 stays inside the shared affine formula's valid range by
 * K-splitting at 4096; fp16 has no K-split, runs past scale=12.8, and a `K<2048` gate was added to route
 * around the breakage — discarding the bank allocation. So instead of guessing what sched=0 fails to
 * program, DECOMPOSE THE MATMUL INTO K VALUES THE FAST PATH ALREADY VALIDATES:
 *
 *     3584 = 1024 + 1024 + 1024 + 512        (orki_f16_sched: power of two, 128 <= K < 2048)
 *     mcap: 180224/1024 = 176                 vs 16384/3584 = 4 for the monolithic form
 *
 * Every chunk runs a configuration that is already measured-exact, so correctness is inherited rather
 * than hoped for. Cost: one host-side strided copy per chunk to make A's K-slice contiguous, plus an
 * fp32 host accumulate of the partials — both measured here, not hand-waved.
 *
 * Numerics note: this will NOT be bit-identical to the monolithic run, and should not be. Splitting the
 * contraction changes the summation order, so partials differ in the last ulp. Both forms are therefore
 * compared against an fp64 reference, and the honest question is which is CLOSER to fp64 — splitting a
 * long accumulation into partial fp32 sums usually IMPROVES accuracy rather than degrading it.
 */
#include <ork_npu.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static unsigned s_=1337; static float rnd(void){ s_=s_*1103515245u+12345u; return (float)((int)((s_>>13)&0x7fff)-16384)/16384.0f; }

int main(int argc,char**argv){
    int M = argc>1?atoi(argv[1]):1024;
    int K = argc>2?atoi(argv[2]):3584;
    int N = argc>3?atoi(argv[3]):1024;
    ork_npu *c = ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }

    /* greedy decomposition into fast-path K values, largest first */
    int chunk[16], nch=0, rem=K;
    while(rem>0 && nch<16){
        int t = rem>=1024?1024 : rem>=512?512 : rem>=256?256 : rem>=128?128 : rem;
        chunk[nch++]=t; rem-=t;
    }
    printf("K=%d N=%d M=%d\n  monolithic: sched=%s mcap=%d -> %d M-tiles\n  split into %d chunks:",
           K,N,M, ((K&(K-1))==0&&K>=128&&K<2048)?"1":"0", 16384/K, (M+16384/K-1)/(16384/K), nch);
    for(int i=0;i<nch;i++) printf(" %d",chunk[i]);
    printf("  (each sched=1, mcap %d)\n\n", 180224/1024);

    ork_f16 *A=malloc((size_t)M*K*2), *B=malloc((size_t)K*N*2), *Ac=malloc((size_t)M*1024*2);
    float *Cm=malloc((size_t)M*N*4), *Cs=malloc((size_t)M*N*4), *Cp=malloc((size_t)M*N*4);
    if(!A||!B||!Ac||!Cm||!Cs||!Cp){ printf("alloc failed\n"); return 1; }
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(ork_f16)rnd();
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=(ork_f16)rnd();

    /* ---- monolithic: the current shipped behaviour ---- */
    ork_w *w=ork_f16_mm_pack(c,K,N,B); if(!w){ printf("pack failed\n"); return 1; }
    if(ork_f16_mm_run(c,w,M,A,Cm)){ printf("monolithic run failed\n"); return 1; }
    double t0=now_us(); ork_f16_mm_run(c,w,M,A,Cm); double t_mono=now_us()-t0;
    ork_mm_free(c,w);

    /* ---- K-split: every chunk on the validated fast path ---- */
    double t_copy=0,t_pack=0,t_run=0,t_acc=0;
    double tAll=now_us();
    memset(Cs,0,(size_t)M*N*4);
    int k0=0;
    for(int i=0;i<nch;i++){
        int Kc=chunk[i];
        double a=now_us();                                   /* A's K-slice must be contiguous */
        for(int m=0;m<M;m++) memcpy(Ac+(size_t)m*Kc, A+(size_t)m*K+k0, (size_t)Kc*2);
        t_copy+=now_us()-a;
        a=now_us(); ork_w *wc=ork_f16_mm_pack(c,Kc,N,B+(size_t)k0*N); t_pack+=now_us()-a;
        if(!wc){ printf("chunk pack failed Kc=%d\n",Kc); return 1; }
        a=now_us(); int rc=ork_f16_mm_run(c,wc,M,Ac,Cp); t_run+=now_us()-a;
        if(rc){ printf("chunk run failed Kc=%d rc=%d\n",Kc,rc); return 1; }
        a=now_us(); for(size_t j=0;j<(size_t)M*N;j++) Cs[j]+=Cp[j]; t_acc+=now_us()-a;
        ork_mm_free(c,wc); k0+=Kc;
    }
    double t_split=now_us()-tAll;

    /* ---- fp64 reference on sampled outputs: which form is closer to the truth? ---- */
    double em=0,es=0;
    for(int s=0;s<16;s++){
        int m=(s*M)/16, n=(s*N)/16+(s%7); if(n>=N)n=N-1;
        double r=0; for(int k=0;k<K;k++) r += (double)(float)A[(size_t)m*K+k]*(double)(float)B[(size_t)k*N+n];
        double d=fabs(r)>1e-9?fabs(r):1e-9;
        double a=fabs(Cm[(size_t)m*N+n]-r)/d, b=fabs(Cs[(size_t)m*N+n]-r)/d;
        if(a>em)em=a; if(b>es)es=b;
    }
    long diff=0; for(size_t j=0;j<(size_t)M*N;j++) if(Cm[j]!=Cs[j]) diff++;
    double g=2.0*M*K*N;
    printf("%-34s %10s %12s\n","form","us","GFLOP/s");
    printf("%-34s %10.0f %12.1f\n","monolithic (shipped, mcap 4)",t_mono,g/(t_mono*1e3));
    printf("%-34s %10.0f %12.1f\n","K-split total",t_split,g/(t_split*1e3));
    printf("    of which  A K-slice copies  %10.0f us\n",t_copy);
    printf("    of which  packs (%d)         %10.0f us\n",nch,t_pack);
    printf("    of which  NPU runs           %10.0f us\n",t_run);
    printf("    of which  host fp32 accum    %10.0f us\n",t_acc);
    printf("\nspeedup %.2fx   (NPU-only portion %.2fx)\n", t_mono/t_split, t_mono/t_run);
    printf("elements differing monolithic vs split: %ld/%zu  (expected nonzero — summation order differs)\n",
           diff,(size_t)M*N);
    printf("max relerr vs fp64:  monolithic %.3e   K-split %.3e   -> %s\n", em, es,
           es<=em?"K-split is as accurate or BETTER":"K-split is less accurate");
    ork_npu_free(c); return 0;
}
