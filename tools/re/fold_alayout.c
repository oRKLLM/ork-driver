/* fold_alayout.c — #39: SOLVE rkllm's M-fold A-layout (+ C de-tile) empirically & SAFELY.
 * Replays rkllm's CAPTURED fold regcmd (/tmp/mm_regcmd.txt, the ground-truth non-wedging config) with RANDOM
 * A[M][K], W[K][N]; weight packed in ork_woff (independently CONFIRMED == mtx512 weight_int8). For each
 * A-packing hypothesis we pack A that way, submit once, capture the raw C buffer, then in software test every
 * C-de-tile hypothesis against the CPU reference C=A*W. The (A-pack, de-tile) pair that is bit-exact IS the
 * fold's layout. Since the regcmd is fixed to rkllm's proven config, this cannot wedge (t0 replay + iters=50
 * both ran clean). Board only. Run:  sudo env ORK_MM_TIMEOUT=6 ./fold_alayout
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"

static uint32_t rng=0x1234567u;
static int r7(void){ rng=rng*1664525u+1013904223u; return (int)((rng>>25)%7)-3; }   /* -3..3 */
static size_t ork_woff(int n,int k,int K){ int KT=(K+31)/32; return ((size_t)(n/32)*KT+(k/32))*1024+(size_t)(n%32)*32+(k%32); }

/* A-pack hypotheses: logical (m,k) -> byte offset in the A buffer. */
static const char* ANAME[]={"nc16","rowmajor[M][K]","nc64","nc32","colmajor[K][M]","nc16_Mheight"};
static size_t apack(int h,int m,int k,int M,int K){
    switch(h){
      case 0: return (size_t)(k/16)*((size_t)M*16)+(size_t)m*16+(k%16);          /* NC1HWC2 C2=16, M-in-width */
      case 1: return (size_t)m*K+k;                                              /* row-major [M][K] */
      case 2: return (size_t)(k/64)*((size_t)M*64)+(size_t)m*64+(k%64);          /* NC1HWC2 C2=64 */
      case 3: return (size_t)(k/32)*((size_t)M*32)+(size_t)m*32+(k%32);          /* NC1HWC2 C2=32 */
      case 4: return (size_t)k*M+m;                                              /* col-major [K][M] */
      case 5: return (size_t)(k/16)*((size_t)M*16)+(size_t)(k%16)*M+m;           /* NC1HWC2 C2=16, M innermost */
    }
    return 0;
}
enum { NAH=6 };
/* C-de-tile hypotheses: logical (m,n) -> int32 index in the raw C buffer. */
static const char* CNAME[]={"nc4","rowmajor[M][N]","nc16","colmajor[N][M]","nc8"};
static size_t cdet(int h,int m,int n,int M,int N){
    switch(h){
      case 0: return (size_t)(n/4)*((size_t)M*4)+(size_t)m*4+(n%4);              /* NC1HWC2 C2=4 int32 */
      case 1: return (size_t)m*N+n;                                             /* row-major [M][N] */
      case 2: return (size_t)(n/16)*((size_t)M*16)+(size_t)m*16+(n%16);          /* NC1HWC2 C2=16 */
      case 3: return (size_t)n*M+m;                                            /* col-major [N][M] */
      case 4: return (size_t)(n/8)*((size_t)M*8)+(size_t)m*8+(n%8);              /* NC1HWC2 C2=8 */
    }
    return 0;
}
enum { NCH=5 };

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    int M=0,K=0,N=0; { FILE*f=fopen("/tmp/mm_meta.txt","r"); if(!f){fprintf(stderr,"no /tmp/mm_meta.txt (capture rkllm first)\n");return 1;}
        char kk[32];int v; while(fscanf(f,"%31s %d",kk,&v)==2){ if(!strcmp(kk,"M"))M=v;else if(!strcmp(kk,"K"))K=v;else if(!strcmp(kk,"N"))N=v;} fclose(f);}
    unsigned rc[2048]; int rn=0; { FILE*f=fopen("/tmp/mm_regcmd.txt","r"); if(!f){fprintf(stderr,"no /tmp/mm_regcmd.txt\n");return 1;} while(rn<2048&&fscanf(f,"%x",&rc[rn])==1)rn++; fclose(f);}
    printf("fold_alayout: M=%d K=%d N=%d rn=%d (replaying rkllm's captured fold regcmd)\n",M,K,N,rn);
    if(M<1||K<32||N<16){fprintf(stderr,"bad dims\n");return 1;}

    int8_t *A=malloc((size_t)M*K),*W=malloc((size_t)K*N); int32_t*Cref=calloc((size_t)M*N,4);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(int8_t)r7();
    for(size_t i=0;i<(size_t)K*N;i++)W[i]=(int8_t)r7();
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++)s+=(long)A[(size_t)m*K+k]*W[(size_t)k*N+n]; Cref[(size_t)m*N+n]=(int32_t)s; }
    int8_t*Wok=calloc((size_t)K*N,1); for(int k=0;k<K;k++)for(int n=0;n<N;n++)Wok[ork_woff(n,k,K)]=W[(size_t)k*N+n];

    ork_npu*c=ork_npu_init(); if(!c){printf("init (board only)\n");return 77;}
    ork_npu_set_core_budget(c,1);
    /* int8 warm on the proven path */
    { ork_w*w=ork_mm_pack_i8(c,K,N,W); if(w){int32_t*t=calloc((size_t)M*N,4);ork_mm_run_i8(c,w,M,A,t);free(t);ork_mm_free(c,w);} }

    size_t astride=(size_t)M*K;                     /* all A hypotheses fit exactly in M*K (K%64==0) */
    int8_t*Avar=calloc(astride*NAH,1);              /* NAH packed A-variants, one contiguous block */
    for(int ha=0;ha<NAH;ha++){ int8_t*Ab=Avar+astride*ha;
        for(int m=0;m<M;m++)for(int k=0;k<K;k++){ size_t o=apack(ha,m,k,M,K); if(o<astride)Ab[o]=A[(size_t)m*K+k]; } }
    int32_t*Couts=calloc(astride?(size_t)M*N*NAH:1,4);
    printf("submitting %d A-variants via ONE buffer set (stable IOVA, wedge-safe)...\n",NAH);
    int r=ork_npu_replay_i8_sweep(c,rc,rn,M,K,N,Avar,NAH,(int)astride,Wok,(int)((size_t)K*N),Couts);
    if(r){ printf("sweep rc=%d\n",r); ork_npu_free(c); return 1; }
    int best_a=-1,best_c=-1;
    for(int ha=0;ha<NAH;ha++){ int32_t*Cout=Couts+(size_t)M*N*ha;
        int bestbad=M*N+1,bc=-1;
        for(int hc=0;hc<NCH;hc++){ int bad=0; for(int m=0;m<M&&bad<bestbad;m++)for(int n=0;n<N;n++){ if(Cout[cdet(hc,m,n,M,N)]!=Cref[(size_t)m*N+n])bad++; }
            if(bad<bestbad){bestbad=bad;bc=hc;} }
        printf("  A=%-16s -> best de-tile=%-14s mism=%d/%d\n",ANAME[ha],bc>=0?CNAME[bc]:"?",bestbad,M*N);
        if(bestbad==0){ best_a=ha;best_c=bc; printf("  *** BIT-EXACT: A-pack=%s  C-detile=%s ***\n",ANAME[ha],CNAME[bc]); }
    }
    if(best_a<0) printf("no exact combo among %d A x %d C hypotheses; extend the tables\n",NAH,NCH);
    ork_npu_free(c); return best_a<0;
}
