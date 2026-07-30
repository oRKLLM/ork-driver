/* fold_chain_bench — the WEIGHT-RESIDENT fold chain: P width-8 fold tiles in ONE task_number=P submit over a
 * shared woff weight, via ork_npu_replay_i8_chain. Verifies bit-exact, then times vs ork_mm_run_i8 at M=P*8.
 * The key question: does chain time grow SLOWLY with P (weight amortized -> fold wins) or ~linearly (no win)?
 *   sudo env ORK_MM_TIMEOUT=8 ./fold_chain_bench [P=28] [K=3584] [N=1216] [iters=3]   (needs /tmp/mm_regcmd.txt, M=8)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "ork_npu.h"
extern int ork_npu_replay_i8_chain(ork_npu*,const unsigned*,int,int,int,int,const signed char*,int,const signed char*,int,int,int*,int,double*);
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static uint32_t rng=0x0bad1dea; static int r7(void){ rng=rng*1664525u+1013904223u; return (int)((rng>>24)%15)-7; }
static size_t nc16(int m,int k,int W){ return (size_t)(k/16)*((size_t)W*16)+(size_t)m*16+(k%16); }
static size_t woff(int k,int n,int K){ int KT=K/32; return ((size_t)(n/32)*KT+(k/32))*1024+(size_t)(n%32)*32+(k%32); }
static size_t c4(int m,int n,int W){ return (size_t)(n/4)*((size_t)W*4)+(size_t)m*4+(n%4); }

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int P=argc>1?atoi(argv[1]):28, K=argc>2?atoi(argv[2]):3584, N=argc>3?atoi(argv[3]):1216, iters=argc>4?atoi(argv[4]):3;
    const int TW=8; int M=P*TW;
    printf("fold_chain_bench P=%d (w=%d, M=%d) K=%d N=%d iters=%d\n",P,TW,M,K,N,iters);
    uint32_t rc[512]; FILE*f=fopen("/tmp/mm_regcmd.txt","r"); int rn=0;
    if(!f){perror("mm_regcmd.txt");return 2;} while(rn<512&&fscanf(f,"%x",&rc[rn])==1)rn++; fclose(f);

    int8_t*Al=malloc((size_t)M*K),*Wl=malloc((size_t)K*N);
    for(size_t i=0;i<(size_t)M*K;i++)Al[i]=(int8_t)r7(); for(size_t i=0;i<(size_t)K*N;i++)Wl[i]=(int8_t)r7();
    signed char*Wp=calloc((size_t)K*N,1); for(int k=0;k<K;k++)for(int n=0;n<N;n++)Wp[woff(k,n,K)]=Wl[(size_t)k*N+n];
    size_t tileA=(size_t)((K+15)/16)*TW*16;
    signed char*Ap=calloc((size_t)P*tileA,1);
    for(int t=0;t<P;t++)for(int m=0;m<TW;m++)for(int k=0;k<K;k++) Ap[(size_t)t*tileA+nc16(m,k,TW)]=Al[(size_t)(t*TW+m)*K+k];
    int32_t*Craw=calloc((size_t)P*TW*N,4);

    ork_npu*c=ork_npu_init(); if(!c){printf("init (board only)\n");return 77;}
    ork_npu_set_core_budget(c,1);
    { int8_t*wd=calloc((size_t)K*N,1);int32_t*td=calloc((size_t)8*N,4);int8_t*ad=calloc((size_t)8*K,1);
      ork_w*w=ork_mm_pack_i8(c,K,N,wd); if(w){ork_mm_run_i8(c,w,8,ad,td);ork_mm_free(c,w);} free(wd);free(td);free(ad); }

    double us=0; int r=ork_npu_replay_i8_chain(c,rc,rn,TW,K,N,Ap,(int)tileA,Wp,(int)((size_t)K*N),P,Craw,iters,&us);
    if(r){ printf("chain rc=%d (err/stall)\n",r); ork_npu_free(c); return 1; }
    long bad=0; int checked=0;
    for(int gm=0;gm<M&&checked<48;gm+=13)for(int n=0;n<N&&checked<48;n+=97){
        int t=gm/TW,m=gm%TW; long s=0; for(int k=0;k<K;k++) s+=(long)Al[(size_t)gm*K+k]*Wl[(size_t)k*N+n];
        if(Craw[(size_t)t*TW*N+c4(m,n,TW)]!=(int32_t)s)bad++; checked++; }
    printf("FOLD-CHAIN: %.0f us/matmul (task_number=%d, M=%d)  verify %s  => %.1f us/row, %.1f us/tile\n",
           us,P,M, bad?"MISMATCH":"BIT-EXACT", us/M, us/P);

    int32_t*Cn=calloc((size_t)M*N,4); double norm_us=0;
    { ork_w*w=ork_mm_pack_i8(c,K,N,Wl); if(w){ ork_mm_run_i8(c,w,M,Al,Cn);
        double tn=now_us(); for(int it=0;it<iters;it++) ork_mm_run_i8(c,w,M,Al,Cn); norm_us=(now_us()-tn)/iters; ork_mm_free(c,w);} }
    printf("NORMAL ork_mm_run_i8: %.0f us/matmul (M=%d) => %.1f us/row\n",norm_us,M,norm_us/M);
    printf("=> fold-chain is %.2fx %s normal\n", norm_us/(us>0?us:1), (us<norm_us)?"FASTER than":"slower than");
    ork_npu_free(c); return bad?1:0;
}
