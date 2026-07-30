/* fold_bench — WIRE-UP + BENCHMARK of the RE'd fold (capture-replay) vs ork's normal int8 matmul.
 * A full M-row matmul C=A*W (int8) is done as ceil(M/8) width-8 fold tiles swept over a RESIDENT woff weight
 * via ork_npu_replay_i8_sweep (weight loaded once, A/C per tile — rkllm's weight-resident row-tile idea).
 * Verifies bit-exact vs CPU, then times fold vs ork_mm_run_i8 at the same (M,K,N).
 *   sudo env ORK_MM_TIMEOUT=6 ./fold_bench [M=228] [K=3584] [N=1216] [iters=5]   (needs /tmp/mm_regcmd.txt, M=8)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "ork_npu.h"
extern int ork_npu_replay_i8_sweep(ork_npu*,const unsigned*,int,int,int,int,const signed char*,int,int,const signed char*,int,int*);
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static uint32_t rng=0x13572468; static int r7(void){ rng=rng*1664525u+1013904223u; return (int)((rng>>24)%15)-7; }
static size_t nc16(int m,int k,int W){ return (size_t)(k/16)*((size_t)W*16)+(size_t)m*16+(k%16); }
static size_t woff(int k,int n,int K){ int KT=K/32; return ((size_t)(n/32)*KT+(k/32))*1024+(size_t)(n%32)*32+(k%32); }
static size_t c4(int m,int n,int W){ return (size_t)(n/4)*((size_t)W*4)+(size_t)m*4+(n%4); }

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int M=argc>1?atoi(argv[1]):228, K=argc>2?atoi(argv[2]):3584, N=argc>3?atoi(argv[3]):1216, iters=argc>4?atoi(argv[4]):5;
    const int TW=8;                                        /* captured tile width */
    int nt=(M+TW-1)/TW;                                    /* number of width-8 tiles */
    printf("fold_bench M=%d K=%d N=%d tiles=%d(x w=%d) iters=%d\n",M,K,N,nt,TW,iters);
    uint32_t rc[512]; FILE*f=fopen("/tmp/mm_regcmd.txt","r"); int rn=0;
    if(!f){perror("mm_regcmd.txt");return 2;} while(rn<512&&fscanf(f,"%x",&rc[rn])==1)rn++; fclose(f);

    int8_t*Al=malloc((size_t)M*K),*Wl=malloc((size_t)K*N);
    for(size_t i=0;i<(size_t)M*K;i++)Al[i]=(int8_t)r7(); for(size_t i=0;i<(size_t)K*N;i++)Wl[i]=(int8_t)r7();
    /* pack weight woff (resident, shared across tiles) */
    signed char*Wp=calloc((size_t)K*N,1); for(int k=0;k<K;k++)for(int n=0;n<N;n++)Wp[woff(k,n,K)]=Wl[(size_t)k*N+n];
    /* pack A as nt width-8 tiles (nc16); last tile row-padded with zeros */
    size_t astride=(size_t)((K+15)/16)*TW*16;
    signed char*Av=calloc((size_t)nt*astride,1);
    for(int t=0;t<nt;t++)for(int m=0;m<TW;m++){ int gm=t*TW+m; if(gm>=M)continue;
        for(int k=0;k<K;k++) Av[(size_t)t*astride + nc16(m,k,TW)] = Al[(size_t)gm*K+k]; }
    int32_t*Couts=calloc((size_t)nt*TW*N,4);

    ork_npu*c=ork_npu_init(); if(!c){printf("init (board only)\n");return 77;}
    ork_npu_set_core_budget(c,1);
    { int8_t*wd=calloc((size_t)K*N,1);int32_t*td=calloc((size_t)8*N,4);int8_t*ad=calloc((size_t)8*K,1);   /* warm */
      ork_w*w=ork_mm_pack_i8(c,K,N,wd); if(w){ork_mm_run_i8(c,w,8,ad,td);ork_mm_free(c,w);} free(wd);free(td);free(ad); }

    /* ---- FOLD: sweep nt width-8 tiles over resident woff weight ---- */
    double tf=now_us(); int rr=0;
    for(int it=0;it<iters;it++) rr|=ork_npu_replay_i8_sweep(c,rc,rn,TW,K,N,Av,nt,(int)astride,Wp,(int)((size_t)K*N),Couts);
    double fold_us=(now_us()-tf)/iters;
    if(rr){ printf("fold sweep rc=%d (err/stall)\n",rr); ork_npu_free(c); return 1; }
    /* verify bit-exact vs CPU on a sample of outputs */
    long bad=0; int checked=0;
    for(int gm=0;gm<M && checked<64; gm+=37) for(int n=0;n<N && checked<64; n+=101){
        int t=gm/TW, m=gm%TW; long s=0; for(int k=0;k<K;k++) s+=(long)Al[(size_t)gm*K+k]*Wl[(size_t)k*N+n];
        int32_t got=Couts[(size_t)t*TW*N + c4(m,n,TW)]; if(got!=(int32_t)s)bad++; checked++; }
    printf("fold: %.0f us/matmul (M=%d)  verify %s (%d checked)\n",fold_us,M, bad?"MISMATCH":"bit-exact",checked);

    /* ---- NORMAL: ork_mm_run_i8 at M ---- */
    int32_t*Cn=calloc((size_t)M*N,4); double norm_us=0;
    { ork_w*w=ork_mm_pack_i8(c,K,N,Wl); if(w){ ork_mm_run_i8(c,w,M,Al,Cn); /* warm */
        double tn=now_us(); for(int it=0;it<iters;it++) ork_mm_run_i8(c,w,M,Al,Cn); norm_us=(now_us()-tn)/iters; ork_mm_free(c,w);} }
    printf("normal ork_mm_run_i8: %.0f us/matmul (M=%d)\n",norm_us,M);
    printf("=> fold %.2fx %s normal  (fold %.1f us/row, normal %.1f us/row)\n",
           norm_us/ (fold_us>0?fold_us:1), (fold_us<norm_us)?"FASTER than":"slower than", fold_us/M, norm_us/M);
    ork_npu_free(c); return bad?1:0;
}
