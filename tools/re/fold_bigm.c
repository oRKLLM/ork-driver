/* fold_bigm — #39 Path-1: run a big-M fold tile via a [state-setter loader][big-M] chain and verify bit-exact.
 * A big-M tile omits ~20 DPU output-stage regs (0x4098-0x412c) and inherits them; it can't be self-contained
 * (>108 regs). So tile 0 is a self-contained loader (mm_regcmd_m8.txt) whose 0x40c0 is patched to the big-M
 * value (ORK_LOADER_40C0, default 0x3000); it establishes the DPU state the big-M tile (tile 1) inherits.
 * Verifies tile 1's wb rows against a CPU int8 ref.
 *   sudo env ORK_MM_TIMEOUT=500 ./fold_bigm [wb] [K] [N] [loader.txt] [body.txt]   (default 36 3584 1216 m8 m36)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"
extern int ork_npu_mfold_chain_v(ork_npu*,int,const int*,int,int,const unsigned*,int,const int8_t*,const int8_t*,int32_t*,int,int,double*);
static uint32_t rng=0x9e37u; static int r7(void){ rng=rng*1664525u+1013904223u; return (int)((rng>>25)%7)-3; }
static size_t nc16(int m,int c,int w){ return (size_t)(c/16)*((size_t)w*16)+(size_t)m*16+(c%16); }
static size_t woff(int n,int k,int K){ int KT=(K+31)/32; return ((size_t)(n/32)*KT+(k/32))*1024+(size_t)(n%32)*32+(k%32); }
static size_t c4(int m,int n,int w){ return (size_t)(n/4)*((size_t)w*4)+(size_t)m*4+(n%4); }
static int loadtile(const char*p, uint32_t*rc){ FILE*f=fopen(p,"r"); if(!f){perror(p);return 0;} int n=0; while(n<232&&fscanf(f,"%x",&rc[n])==1)n++; fclose(f); return n; }
int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int wb=argc>1?atoi(argv[1]):36, K=argc>2?atoi(argv[2]):3584, N=argc>3?atoi(argv[3]):1216;
    const char*lf=argc>4?argv[4]:"/tmp/mm_regcmd_m8.txt", *bf=argc>5?argv[5]:"/tmp/mm_regcmd_m36.txt";
    const int wl=8;                                   /* loader is the self-contained M=8 tile */
    static uint32_t tiles[2*232]; if(loadtile(lf,tiles)!=232||loadtile(bf,tiles+232)!=232){printf("load fail\n");return 2;}
    /* patch loader's 0x40c0 (DPU_SURFACE_ADD) to the big-M tile's expected value so the big-M tile inherits it */
    uint32_t want40c0 = getenv("ORK_LOADER_40C0")?(uint32_t)strtoul(getenv("ORK_LOADER_40C0"),0,0):0x3000;
    for(int k=0;k+1<232;k+=2) if((tiles[k]&0xffff)==0x40c0 && (tiles[k+1]>>16)==0x1001){ tiles[k]=(want40c0<<16)|0x40c0; break; }
    printf("fold_bigm: [loader M=%d 0x40c0=%#x][body M=%d]  K=%d N=%d\n",wl,want40c0,wb,K,N);
    int ws[2]={wl,wb}; int Mtot=wl+wb;
    int8_t *A=malloc((size_t)Mtot*K),*W=malloc((size_t)K*N); int32_t*ref=calloc((size_t)wb*N,4);
    for(size_t i=0;i<(size_t)Mtot*K;i++) A[i]=(int8_t)r7();
    for(size_t i=0;i<(size_t)K*N;i++) W[i]=(int8_t)r7();
    /* ref = body rows (A[wl..wl+wb)) x W */
    for(int m=0;m<wb;m++)for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[(size_t)(wl+m)*K+k]*W[(size_t)k*N+n]; ref[(size_t)m*N+n]=(int32_t)s; }
    /* pack A concat: tile0 = rows[0,wl) width wl; tile1 = rows[wl,wl+wb) width wb */
    int8_t *Ap=calloc((size_t)Mtot*K,1);
    for(int m=0;m<wl;m++)for(int k=0;k<K;k++) Ap[nc16(m,k,wl)]=A[(size_t)m*K+k];
    size_t a1=(size_t)wl*K; for(int m=0;m<wb;m++)for(int k=0;k<K;k++) Ap[a1+nc16(m,k,wb)]=A[(size_t)(wl+m)*K+k];
    int8_t *Wp=calloc((size_t)K*N,1); for(int k=0;k<K;k++)for(int n=0;n<N;n++) Wp[woff(n,k,K)]=W[(size_t)k*N+n];
    int32_t *Craw=calloc((size_t)Mtot*N,4);
    ork_npu*c=ork_npu_init(); if(!c){printf("init (board only)\n");return 77;}
    ork_npu_set_core_budget(c,1);
    { ork_w*ww=ork_mm_pack_i8(c,K,N,W); if(ww){int32_t*t=calloc((size_t)8*N,4);int8_t*a8=calloc((size_t)8*K,1);ork_mm_run_i8(c,ww,8,a8,t);free(t);free(a8);ork_mm_free(c,ww);} }
    double us=0; int r=ork_npu_mfold_chain_v(c,2,ws,K,N,tiles,232,Ap,Wp,Craw,0,3,&us);
    if(r){ printf("chain_v rc=%d (STALL/err)\n",r); ork_npu_free(c); return 1; }
    int32_t *Cb=Craw+(size_t)wl*N;   /* tile 1 (body) output */
    long mm=0,mx=0; for(int m=0;m<wb;m++)for(int n=0;n<N;n++){ int32_t got=Cb[c4(m,n,wb)], rf=ref[(size_t)m*N+n]; long e=labs((long)got-rf); if(e)mm++; if(e>mx)mx=e; }
    printf("RESULT body(M=%d): %ld/%d mismatch maxerr=%ld  %.1f us/submit  %s\n", wb, mm, wb*N, mx, us, mm?"MISMATCH":"*** BIT-EXACT — big-M tile works via state-setter! ***");
    ork_npu_free(c); return mm?1:0;
}
