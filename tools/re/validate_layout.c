/* validate_layout — DECISIVE test of the recovered fold layouts (self-consistent, no captured operands).
 * Generates ork's OWN random logical A[M][K], W[K][N]; packs A in nc16 (C2-16) and W in woff (RK3588 matmul
 * native (N/32,K/32,32,32)); replays rkllm's CAPTURED regcmd (/tmp/mm_regcmd.txt) via ork_npu_replay_i8;
 * de-tiles the C2-4 output (c4) and compares to a CPU reference C=A*W. If BIT-EXACT => the layouts
 * (nc16 in / woff weight / c4 out) are correct and capture-replay WORKS (the earlier offline mismatch was a
 * captured-operand artifact, not a layout error).
 *   sudo env ORK_MM_TIMEOUT=6 ./validate_layout [M=8] [K=3584] [N=1216]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"
extern int ork_npu_replay_i8(ork_npu*,const unsigned*,int,int,int,int,const signed char*,int,const signed char*,int,int*,int,double*);

static uint32_t rng=0x2468acef;
static int r7(void){ rng=rng*1664525u+1013904223u; return (int)((rng>>24)%15)-7; }   /* logical values in [-7,7] */
static size_t nc16(int m,int k,int M){ return (size_t)(k/16)*((size_t)M*16)+(size_t)m*16+(k%16); }
static size_t woff(int k,int n,int K){ int KT=K/32; return ((size_t)(n/32)*KT+(k/32))*1024+(size_t)(n%32)*32+(k%32); }
static size_t c4(int m,int n,int M){ return (size_t)(n/4)*((size_t)M*4)+(size_t)m*4+(n%4); }

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int M=argc>1?atoi(argv[1]):8, K=argc>2?atoi(argv[2]):3584, N=argc>3?atoi(argv[3]):1216;
    printf("validate_layout M=%d K=%d N=%d\n",M,K,N);
    uint32_t rc[512]; FILE*f=fopen("/tmp/mm_regcmd.txt","r"); int rn=0;
    if(!f){perror("mm_regcmd.txt");return 2;} while(rn<512 && fscanf(f,"%x",&rc[rn])==1) rn++; fclose(f);

    int8_t*Al=malloc((size_t)M*K), *Wl=malloc((size_t)K*N);          /* logical */
    for(size_t i=0;i<(size_t)M*K;i++) Al[i]=(int8_t)r7();
    for(size_t i=0;i<(size_t)K*N;i++) Wl[i]=(int8_t)r7();
    int32_t*Cref=calloc((size_t)M*N,4);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)Al[(size_t)m*K+k]*Wl[(size_t)k*N+n]; Cref[(size_t)m*N+n]=(int32_t)s; }

    size_t asz=(size_t)((K+15)/16)*M*16, wsz=(size_t)K*N;
    signed char*Ap=calloc(asz,1), *Wp=calloc(wsz,1);
    for(int m=0;m<M;m++)for(int k=0;k<K;k++) Ap[nc16(m,k,M)]=Al[(size_t)m*K+k];
    for(int k=0;k<K;k++)for(int n=0;n<N;n++) Wp[woff(k,n,K)]=Wl[(size_t)k*N+n];

    ork_npu*c=ork_npu_init(); if(!c){printf("init (board only)\n");return 77;}
    ork_npu_set_core_budget(c,1);
    { int8_t*wd=calloc(wsz,1); int32_t*td=calloc((size_t)8*N,4); int8_t*ad=calloc((size_t)8*K,1);   /* warm */
      ork_w*w=ork_mm_pack_i8(c,K,N,wd); if(w){ ork_mm_run_i8(c,w,8,ad,td); ork_mm_free(c,w);} free(wd);free(td);free(ad); }

    int32_t*Craw=calloc((size_t)M*N,4); double us=0;
    int r=ork_npu_replay_i8(c,rc,rn,M,K,N,Ap,(int)asz,Wp,(int)wsz,Craw,1,&us);
    if(r){ printf("replay rc=%d\n",r); return 1; }
    long mm=0,mx=0; int first=-1;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){
        int32_t got=Craw[c4(m,n,M)], ref=Cref[(size_t)m*N+n];
        long e=labs((long)got-ref); if(e){mm++; if(first<0)first=m*N+n;} if(e>mx)mx=e; }
    printf("RESULT: %ld/%d mismatch  maxerr=%ld  %.0f us  %s\n",mm,M*N,mx,us,
           mm?"MISMATCH":"*** BIT-EXACT — nc16/woff/c4 layouts CONFIRMED, capture-replay works! ***");
    if(mm&&first>=0) printf("  first mismatch m=%d n=%d (got %d ref %d)\n",first/N,first%N,Craw[c4(first/N,first%N,M)],Cref[first]);
    return mm?1:0;
}
