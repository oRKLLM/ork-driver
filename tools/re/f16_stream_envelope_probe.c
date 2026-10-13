/* f16_stream_envelope_probe — the (K, N, M) validity envelope of the two fp16 stream entrypoints.
 *
 * orki_f16_mcap_n is the guard both stream paths consult. It models K only (plus a narrow-N rescue),
 * so a shape it admits can still miscompute. attn_headchain_probe showed that at K=128 both stream
 * entrypoints return wrong output while ork_f16_mm_run stays correct. This probe maps WHERE.
 *
 * Grading is a CPU fp32 reference, not arm-vs-arm: the reference for Mmax rows is computed once and
 * every M is graded against its first M rows (A is a single [Mmax,K] buffer, so that is exact).
 * Each cell is REPS independent calls and reports a pass RATE — a single rep cannot tell 1/6 from 0/n.
 * Scanned UPWARD at every M; never bisected (the fp16 envelope is non-monotonic at K=128).
 *
 * usage: f16_stream_envelope_probe [K] [N] [S] [reps] [Mmax] [armmask: 1=stream 2=chain 3=both]
 * Board only. Diagnostic; not in `all`/`test`. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec*1e-3; }

/* The 0x1040 bank split orki_f16_synth would emit for this (K, M-tile) — host-side arithmetic, a copy
 * of the formula in orki_f16_synth, printed so a measured cell can be read against the encoding. */
static void split_of(int K,int mc,int*dbnk,int*wbnk,int*raw){
    double scale=(double)K/256.0;
    int base=(int)(177.0-15.0*(scale-1.0)), slope=(int)(15.0*scale), mg=(mc+63)/64; if(mg<1)mg=1;
    int v=base-slope*(mg-1); if(v<0x1b)v=0x1b;
    *raw=v; *dbnk=v&0xf; *wbnk=(v>>4)&0xf;
}

int main(int argc,char**argv){
    int K    = argc>1?atoi(argv[1]):128;
    int N    = argc>2?atoi(argv[2]):1024;
    int S    = argc>3?atoi(argv[3]):4;
    int reps = argc>4?atoi(argv[4]):6;
    int Mmax = argc>5?atoi(argv[5]):512;
    int arms = argc>6?atoi(argv[6]):3;
    int Ms[]={16,32,48,64,80,96,112,128,144,160,192,224,256,288,320,384,448,512,640,768,1024};
    int nM=(int)(sizeof Ms/sizeof*Ms);

    setvbuf(stdout,NULL,_IOLBF,0);
    ork_npu *c=ork_npu_init(); if(!c){ fprintf(stderr,"init failed\n"); return 1; }
    int gcap=ork_f16_mm_mmax(c,K,N);
    printf("f16_stream_envelope_probe  K=%d N=%d S=%d reps=%d Mmax=%d  (ork %s)\n",K,N,S,reps,Mmax,ork_npu_version());
    printf("  guard says: ork_f16_mm_mmax(K=%d,N=%d) = %d   weight %ld B = %.2f CBUF banks\n",
           K,N,gcap,(long)K*N*2,(double)K*N*2/32768.0);

    ork_f16 *A=malloc((size_t)Mmax*K*2), *B=malloc((size_t)K*N*2);
    float *R=malloc((size_t)Mmax*N*4);
    if(!A||!B||!R){ fprintf(stderr,"OOM\n"); return 1; }
    for(size_t i=0;i<(size_t)Mmax*K;i++) A[i]=(ork_f16)(((i*2654435761u)%1000)/1000.0f-0.5f);
    for(size_t i=0;i<(size_t)K*N;i++)    B[i]=(ork_f16)(((i*40503u)%997)/997.0f-0.5f);
    for(int m=0;m<Mmax;m++) for(int n=0;n<N;n++){ float acc=0;
        for(int k=0;k<K;k++) acc+=(float)A[(size_t)m*K+k]*(float)B[(size_t)k*N+n];
        R[(size_t)m*N+n]=acc; }
    double band=0; for(size_t i=0;i<(size_t)Mmax*N;i++){ double v=fabs(R[i]); if(v>band)band=v; }

    ork_w **w=calloc(S,sizeof *w);
    for(int s=0;s<S;s++){ w[s]=ork_f16_mm_pack(c,K,N,B); if(!w[s]){ fprintf(stderr,"pack failed\n"); return 1; } }
    ork_mm_task_f16 *tk=malloc((size_t)S*sizeof *tk);
    float *C=malloc((size_t)S*Mmax*N*4); if(!C){ fprintf(stderr,"OOM\n"); return 1; }

    /* tol: fp16 inputs accumulated in fp32; the reference uses the same fp16 bit patterns, so a correct
     * arm lands within a few ulps of band. 1e-3*band is three orders above the observed 8e-7 and three
     * below the observed failures (~2-5 against band ~1.5). */
    double tol=1e-3*band+1e-5;
    printf("  |C|max %.3f   tol %.3g\n",band,tol);
    printf("  %-6s %-18s | %-28s | %-28s\n","M","0x1040 (D/W)","RUN_STREAM","RUN_STREAM_CHAIN");

    for(int mi=0;mi<nM;mi++){ int M=Ms[mi]; if(M>Mmax) break;
        int raw,d,wb; split_of(K,(M<gcap?M:gcap),&d,&wb,&raw);
        int okS=0,okC=0,rcS=0,rcC=0; double tS=0,tC=0,wS=0,wC=0;
        for(int r=0;r<reps;r++){
            /* alternate arm order per rep: arm position is worth up to 1.126x on this board */
            for(int turn=0;turn<2;turn++){
                int chain=((r&1)?(turn==0):(turn==1));
                if(!(arms&(chain?2:1))) continue;
                for(int s=0;s<S;s++){ memset(C+(size_t)s*Mmax*N,0,(size_t)M*N*4);
                    tk[s]=(ork_mm_task_f16){w[s],M,A,C+(size_t)s*Mmax*N}; }
                double t0=now_us();
                int rc= chain?ork_f16_mm_run_stream_chain(c,S,tk):ork_f16_mm_run_stream(c,S,tk);
                double el=now_us()-t0;
                double worst=0; int bad=0;
                if(rc) bad=1;
                else for(int s=0;s<S;s++){ const float*o=C+(size_t)s*Mmax*N;
                    for(size_t i=0;i<(size_t)M*N;i++){ double df=fabs((double)o[i]-R[i]);
                        if(!(df<=tol)) bad=1;
                        if(df>worst) worst=df; } }
                if(chain){ rcC=rc; tC+=el; if(!bad)okC++; if(worst>wC)wC=worst; }
                else     { rcS=rc; tS+=el; if(!bad)okS++; if(worst>wS)wS=worst; }
            }
        }
        printf("  %-6d %#06x (D%2d/W%2d) | %-5s %d/%d %9.0fus d%-9.3g | %-5s %d/%d %9.0fus d%-9.3g\n",
               M,raw,d,wb,
               !(arms&1)?"-":rcS?"RC":(okS==reps?"PASS":(okS?"FLAKY":"FAIL")),okS,reps,tS/reps,wS,
               !(arms&2)?"-":rcC?"RC":(okC==reps?"PASS":(okC?"FLAKY":"FAIL")),okC,reps,tC/reps,wC);
    }
    for(int s=0;s<S;s++) ork_mm_free(c,w[s]);
    free(w);free(tk);free(A);free(B);free(R);free(C);
    ork_npu_free(c);
    return 0;
}
