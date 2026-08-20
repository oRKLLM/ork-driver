/* f16_shape_probe — map the fp16 bare-synth() correct-vs-miscompute (M,K,N) boundary and bisect the
 * register cause via the ork_f16_fuzz_* overrides. Board only.
 *
 *   make f16_shape_probe && sudo ORK_MM_TIMEOUT=2000 ./f16_shape_probe [mode]
 *
 * mode (default 0):
 *   0  sweep K x N x M, classify each shape (default synth sched=1 — the raw-synth path)
 *   1  K=64,N=64,M=64: default vs fuzz 0x201:0x1040=0xB1 (template default) — the causation test
 *   2  K sweep at N=64,M=64: dump the sched=1 0x1040 formula value + result, vs 0x1040=0xB1 override
 *
 * ork_f16_npu_probe_mm runs ONE fp16 matmul through bare synth() (hardcoded sched=1); ork_f16_fuzz_add
 * injects regcmd overrides applied at the END of synth() (so they win over the sched logic). Classifies
 * the raw fp32 output vs a CPU (fp16-operand, f64-accum) reference.
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef unsigned short u16;

/* fp16 helpers via the compiler _Float16 (ork_f16) */
static u16 f2h(double x){ ork_f16 h=(ork_f16)x; u16 b; memcpy(&b,&h,2); return b; }
static double h2f(u16 b){ ork_f16 h; memcpy(&h,&b,2); return (double)h; }

/* classify one shape. returns rel-L2; fills *ncorr_col, *ncorr_row, *iszero. */
static double run_shape(ork_npu*c,int M,int K,int N,u16*A,u16*B,float*raw,
                        int*ncorr_col,int*ncorr_row,int*iszero,int*rc_out){
    /* CPU ref: fp16 operands, f64 accumulate */
    static double *ref=NULL; static size_t refsz=0;
    if(refsz<(size_t)M*N){ ref=realloc(ref,(size_t)M*N*sizeof(double)); refsz=(size_t)M*N; }
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double a=0;
        for(int k=0;k<K;k++) a+=h2f(A[(size_t)m*K+k])*h2f(B[(size_t)k*N+n]);
        ref[(size_t)m*N+n]=a; }
    int rc=ork_f16_npu_probe_mm(c,M,K,N,A,B,raw);
    *rc_out=rc;
    if(rc){ *ncorr_col=*ncorr_row=0; *iszero=0; return -1; }
    /* overall rel-L2 */
    double num=0,den=0,amax=0;
    for(int i=0;i<M*N;i++){ double e=raw[i]-ref[i]; num+=e*e; den+=ref[i]*ref[i]; if(fabs(raw[i])>amax)amax=fabs(raw[i]); }
    double rl2=den>0?sqrt(num/den):0;
    *iszero=(amax<1e-6);
    /* per-column correctness (partial-N detector) */
    int cc=0; for(int n=0;n<N;n++){ double cn=0,cd=0; for(int m=0;m<M;m++){ double e=raw[(size_t)m*N+n]-ref[(size_t)m*N+n]; cn+=e*e; cd+=ref[(size_t)m*N+n]*ref[(size_t)m*N+n]; } double r=cd>0?sqrt(cn/cd):0; if(r<0.05)cc++; }
    *ncorr_col=cc;
    /* per-row correctness (wrong-rows / partial-K detector) */
    int cr=0; for(int m=0;m<M;m++){ double rn=0,rd=0; for(int n=0;n<N;n++){ double e=raw[(size_t)m*N+n]-ref[(size_t)m*N+n]; rn+=e*e; rd+=ref[(size_t)m*N+n]*ref[(size_t)m*N+n]; } double r=rd>0?sqrt(rn/rd):0; if(r<0.05)cr++; }
    *ncorr_row=cr;
    return rl2;
}

static const char* classify(double rl2,int ncol,int N,int nrow,int M,int iszero,int rc){
    if(rc==-1) return "WEDGE/err";
    if(iszero) return "ZERO-OUT";
    if(rl2<3e-2) return "correct";
    if(ncol>0 && ncol<N) return "partial-N";
    if(nrow>0 && nrow<M) return "partial-rows";
    return "wrong";
}

int main(int argc,char**argv){
    int mode=argc>1?atoi(argv[1]):0;
    ork_npu*c=ork_npu_init(); if(!c){ fprintf(stderr,"no NPU — skip\n"); return 0; }
    int MAXM=64,MAXK=1024,MAXN=512;
    u16 *A=malloc((size_t)MAXM*MAXK*2), *B=malloc((size_t)MAXK*MAXN*2);
    float *raw=malloc((size_t)2*MAXM*MAXN*4);
    srand(20260712);

    if(mode==1){
        int M=64,K=64,N=64;
        for(int i=0;i<M*K;i++)A[i]=f2h(((double)rand()/RAND_MAX)*2-1);
        for(int i=0;i<K*N;i++)B[i]=f2h(((double)rand()/RAND_MAX)*2-1);
        int ncol,nrow,zero,rc;
        ork_f16_fuzz_clear();
        double r0=run_shape(c,M,K,N,A,B,raw,&ncol,&nrow,&zero,&rc);
        fprintf(stderr,"[K=64] DEFAULT (sched=1, 0x1040=0xBC): rel-L2=%.3e cols=%d/%d rows=%d/%d %s\n",
                r0,ncol,N,nrow,M,classify(r0,ncol,N,nrow,M,zero,rc));
        ork_f16_fuzz_clear(); ork_f16_fuzz_add(0x201,0x1040,0xB1);   /* template default */
        double r1=run_shape(c,M,K,N,A,B,raw,&ncol,&nrow,&zero,&rc);
        fprintf(stderr,"[K=64] FUZZ 0x201:0x1040=0xB1 (template): rel-L2=%.3e cols=%d/%d rows=%d/%d %s\n",
                r1,ncol,N,nrow,M,classify(r1,ncol,N,nrow,M,zero,rc));
        ork_f16_fuzz_clear();
        int pass=(r0>0.5 && r1<3e-2);
        fprintf(stderr,"\nCAUSATION: %s (default zeros, 0x1040=0xB1 fixes it)\n", pass?"CONFIRMED":"NOT confirmed");
        free(A);free(B);free(raw); ork_npu_free(c);
        return pass?0:1;
    }

    if(mode==2){
        int Ks[]={32,48,64,96,128,160,192,224,256,384,512,0};
        int M=64,N=64;
        fprintf(stderr,"K sweep (M=%d N=%d): default sched=1 vs 0x1040=0xB1 override\n",M,N);
        for(int ki=0;Ks[ki];ki++){ int K=Ks[ki]; if(K%32)continue;
            for(int i=0;i<M*K;i++)A[i]=f2h(((double)rand()/RAND_MAX)*2-1);
            for(int i=0;i<K*N;i++)B[i]=f2h(((double)rand()/RAND_MAX)*2-1);
            int ncol,nrow,zero,rc;
            ork_f16_fuzz_clear();
            double r0=run_shape(c,M,K,N,A,B,raw,&ncol,&nrow,&zero,&rc);
            const char*c0=classify(r0,ncol,N,nrow,M,zero,rc);
            ork_f16_fuzz_clear(); ork_f16_fuzz_add(0x201,0x1040,0xB1);
            double r1=run_shape(c,M,K,N,A,B,raw,&ncol,&nrow,&zero,&rc);
            const char*c1=classify(r1,ncol,N,nrow,M,zero,rc);
            ork_f16_fuzz_clear();
            fprintf(stderr,"  K=%-4d sched1 rel-L2=%.2e %-12s | 0x1040=0xB1 rel-L2=%.2e %-12s\n",K,r0,c0,r1,c1);
        }
        free(A);free(B);free(raw); ork_npu_free(c);
        return 0;
    }

    if(mode==3 || mode==4){
        /* int8 path: ork_i8_npu_probe_mm runs one int8 matmul via bare synth_i8() (hardcoded sched=1).
         * mode 3: K sweep (M=64,N=64). mode 4: N sweep incl N=1024 (M=64,K=64) — the FLOOR_DECOMP case. */
        signed char *Ai=malloc((size_t)MAXM*MAXK), *Bi=malloc((size_t)MAXK*MAXN);
        int *iraw=malloc((size_t)2*MAXM*MAXN*sizeof(int));
        static double *ref=NULL; static size_t refsz=0;
        int Ks[]={32,64,96,128,256,512,1024,0};
        int Ns[]={32,64,128,256,512,1024,2048,0};
        int M=64;
        if(mode==3){
            fprintf(stderr,"int8 bare-synth_i8() (sched=1) K sweep (M=%d N=64):\n",M);
            for(int ki=0;Ks[ki];ki++){ int K=Ks[ki],N=64; if(K%32||N%32)continue;
                for(int i=0;i<M*K;i++)Ai[i]=(signed char)((rand()%15)-7);
                for(int i=0;i<K*N;i++)Bi[i]=(signed char)((rand()%15)-7);
                if(refsz<(size_t)M*N){ref=realloc(ref,(size_t)M*N*sizeof(double));refsz=(size_t)M*N;}
                for(int m=0;m<M;m++)for(int n=0;n<N;n++){double a=0;for(int k=0;k<K;k++)a+=(double)Ai[(size_t)m*K+k]*(double)Bi[(size_t)k*N+n];ref[(size_t)m*N+n]=a;}
                ork_i8_fuzz_clear();
                int rc=ork_i8_npu_probe_mm(c,M,K,N,Ai,Bi,iraw);
                double num=0,den=0,amax=0; int cc=0;
                if(!rc){ for(int i=0;i<M*N;i++){double e=iraw[i]-ref[i];num+=e*e;den+=ref[i]*ref[i];if(fabs((double)iraw[i])>amax)amax=fabs((double)iraw[i]);}
                    for(int n=0;n<N;n++){double cn=0,cd=0;for(int m2=0;m2<M;m2++){double e=iraw[(size_t)m2*N+n]-ref[(size_t)m2*N+n];cn+=e*e;cd+=ref[(size_t)m2*N+n]*ref[(size_t)m2*N+n];}if(cd==0||sqrt(cn/cd)<0.02)cc++;} }
                double rl2=den>0?sqrt(num/den):0;
                fprintf(stderr,"  K=%-4d rel-L2=%.2e cols=%d/%d %s\n",K,rl2,cc,N, rc?"WEDGE/err":(amax<1e-9?"ZERO-OUT":(rl2<2e-2?"correct":(cc<N&&cc>0?"partial-N":"wrong"))));
            }
        } else {
            int K=64;
            fprintf(stderr,"int8 bare-synth_i8() (sched=1) N sweep (M=%d K=%d):\n",M,K);
            for(int ni=0;Ns[ni];ni++){ int N=Ns[ni]; if(N%32||N>MAXN)continue;
                for(int i=0;i<M*K;i++)Ai[i]=(signed char)((rand()%15)-7);
                for(int i=0;i<K*N;i++)Bi[i]=(signed char)((rand()%15)-7);
                if(refsz<(size_t)M*N){ref=realloc(ref,(size_t)M*N*sizeof(double));refsz=(size_t)M*N;}
                for(int m=0;m<M;m++)for(int n=0;n<N;n++){double a=0;for(int k=0;k<K;k++)a+=(double)Ai[(size_t)m*K+k]*(double)Bi[(size_t)k*N+n];ref[(size_t)m*N+n]=a;}
                ork_i8_fuzz_clear();
                int rc=ork_i8_npu_probe_mm(c,M,K,N,Ai,Bi,iraw);
                double num=0,den=0,amax=0; int cc=0;
                if(!rc){ for(int i=0;i<M*N;i++){double e=iraw[i]-ref[i];num+=e*e;den+=ref[i]*ref[i];if(fabs((double)iraw[i])>amax)amax=fabs((double)iraw[i]);}
                    for(int n=0;n<N;n++){double cn=0,cd=0;for(int m2=0;m2<M;m2++){double e=iraw[(size_t)m2*N+n]-ref[(size_t)m2*N+n];cn+=e*e;cd+=ref[(size_t)m2*N+n]*ref[(size_t)m2*N+n];}if(cd==0||sqrt(cn/cd)<0.02)cc++;} }
                double rl2=den>0?sqrt(num/den):0;
                fprintf(stderr,"  N=%-4d rel-L2=%.2e cols=%d/%d %s\n",N,rl2,cc,N, rc?"WEDGE/err":(amax<1e-9?"ZERO-OUT":(rl2<2e-2?"correct":(cc<N&&cc>0?"partial-N":"wrong"))));
            }
        }
        free(Ai);free(Bi);free(iraw); free(A);free(B);free(raw); ork_npu_free(c);
        return 0;
    }

    /* mode 0: full sweep, default synth (sched=1) */
    int Ks[]={32,64,96,128,256,512,0};
    int Ns[]={16,32,64,128,256,0};
    int Ms[]={1,8,64,0};
    fprintf(stderr,"fp16 bare-synth() (sched=1 hardcoded) shape sweep — classify vs CPU ref\n");
    fprintf(stderr,"%-4s %-4s %-4s %-10s %-8s %s\n","M","K","N","rel-L2","zero?","class");
    for(int mi=0;Ms[mi];mi++)for(int ki=0;Ks[ki];ki++)for(int ni=0;Ns[ni];ni++){
        int M=Ms[mi],K=Ks[ki],N=Ns[ni];
        if(N>MAXN||K>MAXK||M>MAXM)continue;
        for(int i=0;i<M*K;i++)A[i]=f2h(((double)rand()/RAND_MAX)*2-1);
        for(int i=0;i<K*N;i++)B[i]=f2h(((double)rand()/RAND_MAX)*2-1);
        int ncol,nrow,zero,rc;
        ork_f16_fuzz_clear();
        double rl2=run_shape(c,M,K,N,A,B,raw,&ncol,&nrow,&zero,&rc);
        fprintf(stderr,"%-4d %-4d %-4d %-10.3e %-8s %s (cols %d/%d rows %d/%d)\n",
                M,K,N,rl2,zero?"YES":"no",classify(rl2,ncol,N,nrow,M,zero,rc),ncol,N,nrow,M);
    }
    free(A);free(B);free(raw); ork_npu_free(c);
    return 0;
}
