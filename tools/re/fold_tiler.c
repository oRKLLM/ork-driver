/* fold_tiler — #39 Path-1 TOKEN-TILER: process an M_total-token batch as one multi-task fold submit, end to end.
 * Decomposes M_total into fold sub-tiles of M<=36 (rkllm's size set, greedy largest-first), builds each sub-tile
 * from a captured per-size reference regcmd (/tmp/fold_ref_<M>.txt) by patching the 4 M_total-dependent regs
 * (0x4024=16*M_total, 0x107c=min(M_total,128), 0x1080=M_total-M, 0x40c0=128*M_total), forcing GROUP_LINE +
 * the operation-enable doorbell + any missing output-stage regs, assigns row offsets, and runs them over ONE
 * shared M_total x N output cube (ork_npu_fold_batch). Verifies the full batch A[M_total x K] x W[K x N] bit-exact.
 *   sudo env ORK_MM_TIMEOUT=800 ./fold_tiler [M_total] [K] [N]     (default 72 3584 1216)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"
extern int ork_npu_fold_batch(ork_npu*,int,int,int,int,const int*,const unsigned*,int,const int8_t*,const int8_t*,int32_t*,int,int,double*);
static uint32_t rng=0x9e37u; static int r7(void){ rng=rng*1664525u+1013904223u; return (int)((rng>>25)%7)-3; }
static size_t nc16(int m,int c,int w){ return (size_t)(c/16)*((size_t)w*16)+(size_t)m*16+(c%16); }
static size_t woff(int n,int k,int K){ int KT=(K+31)/32; return ((size_t)(n/32)*KT+(k/32))*1024+(size_t)(n%32)*32+(k%32); }
static size_t c4(int m,int n,int w){ return (size_t)(n/4)*((size_t)w*4)+(size_t)m*4+(n%4); }
static int loadtile(const char*p,uint32_t*rc){ FILE*f=fopen(p,"r"); if(!f){perror(p);return 0;} int n=0; while(n<232&&fscanf(f,"%x",&rc[n])==1)n++; fclose(f); return n; }

/* working reg-pair list */
static int pidx(const uint32_t*pr,int np,unsigned blk,unsigned reg){ for(int j=0;j<np;j++) if((pr[2*j+1]>>16)==blk && (pr[2*j]&0xffff)==reg) return j; return -1; }
static void setv(uint32_t*pr,int*np,unsigned blk,unsigned reg,uint32_t v){
    int j=pidx(pr,*np,blk,reg);
    if(j<0){ j=*np; (*np)++; }
    pr[2*j]=(reg&0xffff)|((v&0xffff)<<16); pr[2*j+1]=(blk<<16)|((v>>16)&0xffff);
}
static const unsigned DPU_OUT[20]={0x4098,0x409c,0x40a0,0x40a4,0x40a8,0x40ac,0x40c0,0x40c4,0x4100,0x4104,0x4108,0x410c,0x4110,0x4114,0x4118,0x411c,0x4120,0x4124,0x4128,0x412c};

/* build one sub-tile regcmd (232 words) for size m at this batch's M_total */
static int build_tile(int m,int Mtot,uint32_t*out232){
    char path[64]; snprintf(path,sizeof path,"/tmp/fold_ref_%d.txt",m);
    static uint32_t cap[232]; if(loadtile(path,cap)!=232){ return -1; }
    static uint32_t pr[2*130]; int np=0;
    for(int k=0;k+1<216;k+=2){ uint32_t w0=cap[k],w1=cap[k+1]; unsigned reg=w0&0xffff,blk=w1>>16;
        if(w0==0&&w1==0) continue;
        if(blk==0x101 && (reg==0x0010||reg==0x0014)) continue;
        int j=pidx(pr,np,blk,reg); if(j<0){ pr[2*np]=w0; pr[2*np+1]=w1; np++; } else { pr[2*j]=w0; pr[2*j+1]=w1; } }
    for(int i=0;i<20;i++) if(pidx(pr,np,0x1001,DPU_OUT[i])<0) setv(pr,&np,0x1001,DPU_OUT[i],0);  /* ensure output-stage present */
    setv(pr,&np,0x81,0x0008,0x000d);                       /* PC_OPERATION_ENABLE doorbell */
    setv(pr,&np,0x41,0x0000,0);
    setv(pr,&np,0x201,0x100c,0x20000000);                  /* GROUP_LINE (batch sub-tile) */
    /* the 4 M_total-dependent regs */
    setv(pr,&np,0x1001,0x4024,(uint32_t)(16*Mtot));        /* DST_SURF_STRIDE = 16*M_total */
    setv(pr,&np,0x201,0x107c,(uint32_t)(Mtot<128?Mtot:128));/* DMA burst = M_total (cap 128) */
    setv(pr,&np,0x201,0x1080,(uint32_t)(Mtot-m));          /* DMA stride = M_total - M */
    setv(pr,&np,0x1001,0x40c0,(uint32_t)(128*Mtot));       /* SURFACE_ADD = 128*M_total */
    if(np>108){ fprintf(stderr,"tile m=%d: %d pairs > 108\n",m,np); return -2; }
    memset(out232,0,232*4); for(int j=0;j<np;j++){ out232[2*j]=pr[2*j]; out232[2*j+1]=pr[2*j+1]; }
    return 0;
}

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int Mtot=argc>1?atoi(argv[1]):72, K=argc>2?atoi(argv[2]):3584, N=argc>3?atoi(argv[3]):1216;
    if(Mtot<1||Mtot>128){ printf("M_total must be 1..128\n"); return 2; }
    /* greedy decompose into fold sizes */
    static const int SZ[]={36,32,28,24,20,16,14,12,10,8,6,4,2,1};
    int m[64],roff[64],P=0,rem=Mtot,off=0;
    while(rem>0){ for(unsigned s=0;s<sizeof SZ/sizeof*SZ;s++) if(SZ[s]<=rem){ m[P]=SZ[s]; roff[P]=off; off+=SZ[s]; rem-=SZ[s]; P++; break; } }
    printf("M_total=%d K=%d N=%d -> %d sub-tiles:",Mtot,K,N,P);
    for(int t=0;t<P;t++) printf(" %d@%d",m[t],roff[t]); printf("\n");

    static uint32_t tiles[64*232];
    for(int t=0;t<P;t++){ int r=build_tile(m[t],Mtot,tiles+(size_t)t*232);
        if(r){ printf("build_tile m=%d failed rc=%d (need /tmp/fold_ref_%d.txt)\n",m[t],r,m[t]); return 3; } }

    /* full-batch operands */
    int8_t *A=malloc((size_t)Mtot*K),*W=malloc((size_t)K*N); int32_t*ref=calloc((size_t)Mtot*N,4);
    for(size_t i=0;i<(size_t)Mtot*K;i++) A[i]=(int8_t)r7();
    for(size_t i=0;i<(size_t)K*N;i++) W[i]=(int8_t)r7();
    for(int i=0;i<Mtot;i++)for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[(size_t)i*K+k]*W[(size_t)k*N+n]; ref[(size_t)i*N+n]=(int32_t)s; }
    int8_t *Ap=calloc((size_t)Mtot*K,1); for(int i=0;i<Mtot;i++)for(int k=0;k<K;k++) Ap[nc16(i,k,Mtot)]=A[(size_t)i*K+k];
    int8_t *Wp=calloc((size_t)K*N,1); for(int k=0;k<K;k++)for(int n=0;n<N;n++) Wp[woff(n,k,K)]=W[(size_t)k*N+n];
    int32_t *Craw=calloc((size_t)Mtot*N,4);

    ork_npu*c=ork_npu_init(); if(!c){ printf("init (board only)\n"); return 77; }
    ork_npu_set_core_budget(c,1);
    { ork_w*ww=ork_i8_mm_pack(c,K,N,W); if(ww){int32_t*t=calloc((size_t)8*N,4);int8_t*a8=calloc((size_t)8*K,1);ork_i8_mm_run(c,ww,8,a8,t);free(t);free(a8);ork_mm_free(c,ww);} }
    double us=0;
    int ncore=getenv("ORK_FOLD_NCORE")?atoi(getenv("ORK_FOLD_NCORE")):1;
    int r=ork_npu_fold_batch(c,Mtot,K,N,P,roff,tiles,232,Ap,Wp,Craw,ncore,3,&us);
    if(r){ printf("RESULT: fold_batch rc=%d (STALL/err)\n",r); ork_npu_free(c); return 1; }
    long mm=0,mx=0; for(int i=0;i<Mtot;i++)for(int n=0;n<N;n++){ int32_t got=Craw[c4(i,n,Mtot)],rf=ref[(size_t)i*N+n]; long e=labs((long)got-rf); if(e)mm++; if(e>mx)mx=e; }
    printf("RESULT M_total=%d: %ld/%d mismatch maxerr=%ld  %.1f us/submit (%d tiles, %.1f us/tile)  %s\n",
           Mtot, mm, Mtot*N, mx, us, P, us/P, mm?"MISMATCH":"*** BIT-EXACT — token-tiler works! ***");
    free(A);free(W);free(ref);free(Ap);free(Wp);free(Craw); ork_npu_free(c); return mm?1:0;
}
