/* bigm_run — #39 Path-1 CORRECTED single-tile test: run a real M=36 fold sub-tile faithfully, WITHOUT clobbering
 * its captured geometry. The decomposition (decompose.py) showed rkllm never runs a standalone M=36 — every M=36
 * tile is a sub-tile of an M_total-token batch with surfstr(0x4024)=16*M_total and its 36 output rows scattered
 * into the M_total x N output cube. Our earlier bigm_selfcontain HARD-WEDGED because sdp_stamp overwrote the
 * captured surfstr with 16*36=576 (the "presume self-contained" bug), making an inconsistent tile -> OOB DMA.
 *
 * This test takes a REAL captured tile (default the plain M=36, surfstr=640 -> M_total=40), COMPACTS it to unique
 * regs (last-write-wins; the dups are idempotent), APPENDS only the genuinely-omitted regs it needs to self-run
 * (any missing output-stage 0x1001 reg -> 0, except 0x40c0 -> its M value; PC_OPERATION_ENABLE 0x81:0x0008=0xd;
 * BLK41 0x41:0x0000=0), and LEAVES all captured geometry (surfstr, cube dims, bank, burst, stride) intact. It
 * sizes the output buffer for M_total (=surfstr/16), packs the 36 input rows nc16 width 36 (the tile's own M),
 * runs P=1, and de-tiles the output c4 width M_total. Bit-exact => the M=36 sub-tile reproduces; runs-but-wrong
 * => not a HW wall (de-tile model detail); wedge => deeper.
 *   sudo env ORK_MM_TIMEOUT=500 ./bigm_run [K] [N] [tile.txt]     (default 3584 1216 mm_regcmd_m36_plain.txt)
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
static int findpair(const uint32_t*w,int np,unsigned blk,unsigned reg){ for(int j=0;j<np;j++) if((w[2*j+1]>>16)==blk && (w[2*j]&0xffff)==reg) return j; return -1; }
static int v16(const uint32_t*w,int np,unsigned blk,unsigned reg){ int j=findpair(w,np,blk,reg); return j<0?-1:(int)(w[2*j]>>16); }

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int K=argc>1?atoi(argv[1]):3584, N=argc>2?atoi(argv[2]):1216;
    const char*tf=argc>3?argv[3]:"/tmp/mm_regcmd_m36_plain.txt";
    static uint32_t cap[232]; if(loadtile(tf,cap)!=232){ printf("load %s fail\n",tf); return 2; }

    /* compact to unique (blk,off), last-write-wins; drop padding + chain descriptor */
    static uint32_t sc[232]; memset(sc,0,sizeof sc); int np=0;
    for(int k=0;k+1<216;k+=2){ uint32_t w0=cap[k],w1=cap[k+1]; unsigned reg=w0&0xffff, blk=w1>>16;
        if(w0==0&&w1==0) continue;
        if(blk==0x101 && (reg==0x0010||reg==0x0014)) continue;
        int f=findpair(sc,np,blk,reg);
        if(f<0){ sc[2*np]=w0; sc[2*np+1]=w1; np++; } else { sc[2*f]=w0; sc[2*f+1]=w1; } }
    int M=v16(sc,np,0x201,0x102c), surf=v16(sc,np,0x1001,0x4024);
    if(M<0||surf<0){ printf("tile missing M(0x102c=%d) or surfstr(0x4024=%d)\n",M,surf); return 2; }
    int Mtot=surf/16;
    /* append only genuinely-missing regs (keep ALL captured geometry untouched) */
    static const unsigned OUT1001[22]={0x4098,0x409c,0x40a0,0x40a4,0x40a8,0x40ac,0x40c0,0x40c4,
        0x4100,0x4104,0x4108,0x410c,0x4110,0x4114,0x4118,0x411c,0x4120,0x4124,0x4128,0x412c,0,0};
    int appended=0;
    for(int i=0;i<20;i++){ unsigned reg=OUT1001[i];
        if(findpair(sc,np,0x1001,reg)<0){ unsigned val=(reg==0x40c0)?(unsigned)(M==36?0x3000:128*M):0;
            sc[2*np]=(val<<16)|reg; sc[2*np+1]=(0x1001u<<16); np++; appended++; } }
    if(findpair(sc,np,0x81,0x0008)<0){ sc[2*np]=(0x000du<<16)|0x0008; sc[2*np+1]=(0x0081u<<16); np++; appended++; }  /* doorbell */
    if(findpair(sc,np,0x41,0x0000)<0){ sc[2*np]=0x0000; sc[2*np+1]=(0x0041u<<16); np++; appended++; }
    printf("tile: M=%d surfstr=%d -> M_total=%d ; %d unique regs + %d appended = %d pairs (%d words)\n",
           M, surf, Mtot, np-appended, appended, np, 2*np);
    if(2*np>216){ printf("ERROR %d words > 216\n",2*np); return 3; }

    /* operands: 36 input rows; output cube is M_total-wide */
    int8_t *A=malloc((size_t)M*K),*W=malloc((size_t)K*N); int32_t*ref=calloc((size_t)M*N,4);
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)r7();
    for(size_t i=0;i<(size_t)K*N;i++) W[i]=(int8_t)r7();
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[(size_t)m*K+k]*W[(size_t)k*N+n]; ref[(size_t)m*N+n]=(int32_t)s; }
    int8_t *Ap=calloc((size_t)Mtot*K,1);                          /* input cube is M_total-wide; fill rows 0..M-1 nc16 width M_total */
    for(int m=0;m<M;m++)for(int k=0;k<K;k++) Ap[nc16(m,k,Mtot)]=A[(size_t)m*K+k];
    int8_t *Wp=calloc((size_t)K*N,1); for(int k=0;k<K;k++)for(int n=0;n<N;n++) Wp[woff(n,k,K)]=W[(size_t)k*N+n];
    int32_t *Craw=calloc((size_t)Mtot*N,4);

    ork_npu*c=ork_npu_init(); if(!c){ printf("init (board only)\n"); return 77; }
    ork_npu_set_core_budget(c,1);
    { ork_w*ww=ork_i8_mm_pack(c,K,N,W); if(ww){int32_t*t=calloc((size_t)8*N,4);int8_t*a8=calloc((size_t)8*K,1);ork_i8_mm_run(c,ww,8,a8,t);free(t);free(a8);ork_mm_free(c,ww);} }
    int ws[1]={Mtot}; double us=0;                                /* size buffers for M_total (surfstr in-bounds) */
    int r=ork_npu_mfold_chain_v(c,1,ws,K,N,sc,232,Ap,Wp,Craw,0,3,&us);
    if(r){ printf("RESULT: chain_v rc=%d (STALL/errno-110) — tile did not complete\n",r); ork_npu_free(c); return 1; }
    /* de-tile: try output width M_total (expected) and M (fallback) */
    for(int wtry=0; wtry<2; wtry++){ int W_out = wtry? M : Mtot;
        long mm=0,mx=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int32_t got=Craw[c4(m,n,W_out)],rf=ref[(size_t)m*N+n]; long e=labs((long)got-rf); if(e)mm++; if(e>mx)mx=e; }
        printf("  de-tile width=%d: %ld/%d mismatch maxerr=%ld  %s\n", W_out, mm, M*N, mx, mm?"":"*** BIT-EXACT ***"); }
    printf("RESULT: tile COMPLETED (no wedge, %.1f us) — a consistent M=36 sub-tile RUNS standalone.\n", us);
    free(A);free(W);free(ref);free(Ap);free(Wp);free(Craw); ork_npu_free(c); return 0;
}
