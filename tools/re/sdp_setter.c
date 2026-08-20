/* sdp_setter — #39 Path-1: prove the CANONICAL SDP/PPU state-setter reconstructs the output stage from first
 * principles. The full-prefill scan (full_sdp.py over pf.dump) showed block 0x1001 (DPU/SDP) is one invariant
 * config: every functional reg constant across 134k tiles, only geometry varies with (M,N). This harness:
 *   (1) CONTROL: run the captured M=8 tile as-is (chain_v P=1), assert bit-exact vs a CPU int8 ref.
 *   (2) PROOF: ZERO every 0x1001 value in a copy of that tile (keep reg-ids), then ork_npu_sdp_stamp() rebuilds
 *       the whole output stage from sdp_canon(); run again, assert bit-exact. Bit-exact => the SDP/PPU state is
 *       synthesized from understood values, not the captured blob.
 *   (3) BONUS (ORK_SDP_BIGM=1): use the stamped M=8 tile (surfadd patched to the big-M value) as a state-setter
 *       LOADER before the captured big-M body ([loader][M36], chain_v P=2); report whether big-M inherits + runs.
 *   sudo env ORK_MM_TIMEOUT=500 ./sdp_setter [N] [K] [m8.txt] [m36.txt]      (default 1216 3584 m8 m36)
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

/* run a single width-w tile (already prepared) via chain_v P=1; ref[w*N] = A(rows)xW; returns mismatch count */
static long run1(ork_npu*c,const uint32_t*tile,int w,int K,int N,const int8_t*A,const int8_t*W,double*us,long*maxerr){
    int8_t *Ap=calloc((size_t)w*K,1), *Wp=calloc((size_t)K*N,1); int32_t*Craw=calloc((size_t)w*N,4);
    for(int m=0;m<w;m++)for(int k=0;k<K;k++) Ap[nc16(m,k,w)]=A[(size_t)m*K+k];
    for(int k=0;k<K;k++)for(int n=0;n<N;n++) Wp[woff(n,k,K)]=W[(size_t)k*N+n];
    int ws[1]={w};
    int r=ork_npu_mfold_chain_v(c,1,ws,K,N,tile,232,Ap,Wp,Craw,0,3,us);
    long mm=-1,mx=0;
    if(!r){ mm=0; for(int m=0;m<w;m++)for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[(size_t)m*K+k]*W[(size_t)k*N+n];
                int32_t got=Craw[c4(m,n,w)]; long e=labs((long)got-s); if(e)mm++; if(e>mx)mx=e; } }
    else mm=-1;
    if(maxerr)*maxerr=mx; free(Ap);free(Wp);free(Craw); return r?-1:mm;
}

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int N=argc>1?atoi(argv[1]):1216, K=argc>2?atoi(argv[2]):3584;
    const char*lf=argc>3?argv[3]:"/tmp/mm_regcmd_m8.txt", *bf=argc>4?argv[4]:"/tmp/mm_regcmd_m36.txt";
    const int wl=8;
    static uint32_t cap[232], syn[232];
    if(loadtile(lf,cap)!=232){ printf("load %s fail\n",lf); return 2; }
    /* operands (M=8 rows) */
    int8_t *A=malloc((size_t)wl*K), *W=malloc((size_t)K*N);
    for(size_t i=0;i<(size_t)wl*K;i++) A[i]=(int8_t)r7();
    for(size_t i=0;i<(size_t)K*N;i++) W[i]=(int8_t)r7();

    ork_npu*c=ork_npu_init(); if(!c){ printf("init (board only)\n"); return 77; }
    ork_npu_set_core_budget(c,1);
    /* pre-warm the shared weight pack path (matches fold_bigm; avoids first-touch alloc in the timed run) */
    { ork_w*ww=ork_i8_mm_pack(c,K,N,W); if(ww){int32_t*t=calloc((size_t)8*N,4);int8_t*a8=calloc((size_t)8*K,1);ork_i8_mm_run(c,ww,8,a8,t);free(t);free(a8);ork_mm_free(c,ww);} }

    /* (1) CONTROL — captured tile as-is */
    double us0=0; long mx0=0; long mm0=run1(c,cap,wl,K,N,A,W,&us0,&mx0);
    if(mm0<0){ printf("CONTROL chain_v STALL/err (rc)\n"); ork_npu_free(c); return 1; }
    printf("CONTROL  (captured m8):  %ld/%d mismatch maxerr=%ld  %.1f us  %s\n",
           mm0,wl*N,mx0,us0, mm0?"MISMATCH":"bit-exact");

    /* (2) PROOF — zero the whole 0x1001 block, then synthesize it back with the canonical state-setter */
    memcpy(syn,cap,sizeof cap);
    int zeroed=0;
    for(int k=0;k+1<232;k+=2){ unsigned blk=(syn[k+1]>>16)&0xffff;
        if(blk==0x1001||blk==0x801){ syn[k]&=0xffff; syn[k+1]&=0xffff0000u; zeroed++; } }  /* both output blocks; keep reg-id+blk, wipe value */
    int nset=ork_npu_sdp_stamp(syn,232,wl,N,0x400);                              /* 0x400 = 128*8 (matched M=8) */
    printf("stamp: zeroed %d output-stage pairs (0x1001 DPU/SDP + 0x801 PDP/aux), sdp_stamp rewrote %d regs\n", zeroed, nset);
    double us1=0; long mx1=0; long mm1=run1(c,syn,wl,K,N,A,W,&us1,&mx1);
    if(mm1<0){ printf("PROOF chain_v STALL/err (rc)\n"); ork_npu_free(c); return 1; }
    printf("SYNTH SDP (canonical):   %ld/%d mismatch maxerr=%ld  %.1f us  %s\n",
           mm1,wl*N,mx1,us1, mm1?"MISMATCH":"*** BIT-EXACT — SDP/PPU state-setter reconstructed from first principles ***");

    int rc = (mm0==0 && mm1==0) ? 0 : 1;

    /* (3) BONUS — canonical state-setter as the loader before the captured big-M body */
    if(getenv("ORK_SDP_BIGM")){
        static uint32_t body[232];
        if(loadtile(bf,body)==232){
            int wb=36; uint32_t s36=0x3000;                                     /* full_sdp.py: 0x40c0 dominant @ M=36 */
            uint32_t ldr[232]; memcpy(ldr,cap,sizeof ldr);
            for(int k=0;k+1<232;k+=2){ unsigned blk=(ldr[k+1]>>16)&0xffff; if(blk==0x1001){ ldr[k]&=0xffff; ldr[k+1]&=0xffff0000u; } }
            ork_npu_sdp_stamp(ldr,232,wl,N,s36);                                 /* loader establishes big-M's inherited SURFACE_ADD */
            uint32_t tiles[2*232]; memcpy(tiles,ldr,sizeof ldr); memcpy(tiles+232,body,sizeof body);
            int Mtot=wl+wb, ws[2]={wl,wb};
            int8_t *A2=malloc((size_t)Mtot*K); int32_t*ref=calloc((size_t)wb*N,4);
            for(size_t i=0;i<(size_t)Mtot*K;i++) A2[i]=(int8_t)r7();
            for(int m=0;m<wb;m++)for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)A2[(size_t)(wl+m)*K+k]*W[(size_t)k*N+n]; ref[(size_t)m*N+n]=(int32_t)s; }
            int8_t *Ap=calloc((size_t)Mtot*K,1);
            for(int m=0;m<wl;m++)for(int k=0;k<K;k++) Ap[nc16(m,k,wl)]=A2[(size_t)m*K+k];
            size_t a1=(size_t)wl*K; for(int m=0;m<wb;m++)for(int k=0;k<K;k++) Ap[a1+nc16(m,k,wb)]=A2[(size_t)(wl+m)*K+k];
            int8_t *Wp=calloc((size_t)K*N,1); for(int k=0;k<K;k++)for(int n=0;n<N;n++) Wp[woff(n,k,K)]=W[(size_t)k*N+n];
            int32_t *Craw=calloc((size_t)Mtot*N,4); double us2=0;
            int r=ork_npu_mfold_chain_v(c,2,ws,K,N,tiles,232,Ap,Wp,Craw,0,3,&us2);
            if(r) printf("BIGM [synth-loader M=8][M=36]: chain_v rc=%d (STALL/err) — big-M still gated (CBUF capacity)\n",r);
            else { int32_t*Cb=Craw+(size_t)wl*N; long mm=0,mx=0;
                for(int m=0;m<wb;m++)for(int n=0;n<N;n++){ int32_t got=Cb[c4(m,n,wb)],rf=ref[(size_t)m*N+n]; long e=labs((long)got-rf); if(e)mm++; if(e>mx)mx=e; }
                printf("BIGM body(M=36): %ld/%d mismatch maxerr=%ld  %.1f us  %s\n",mm,wb*N,mx,us2, mm?"MISMATCH":"*** BIT-EXACT — big-M runs off the synth state-setter! ***"); }
            free(A2);free(ref);free(Ap);free(Wp);free(Craw);
        } else printf("BIGM: load %s fail (skip)\n",bf);
    }
    free(A);free(W); ork_npu_free(c); return rc;
}
