/* reshape_probe — derive the EXACT contiguous->atom-8 permutation with DISTINCT input.
 * Run the reshape (T0=2, start at the transpose task2) with ORK_RESHAPE_GINJ (fills the reshape input
 * 0xffff0000 with fp16(i+1), all distinct), read reshape_out @0xffff0680. Each output value v uniquely
 * identifies its source input index = round(v)-1, so reshape_out[p] <- input[map(p)] is exactly derivable.
 * BOARD: sudo env ORK_RESHAPE_T0=2 ORK_RESHAPE_GINJ=1 ORK_EW_TIMEOUT=2000 ./reshape_probe   (needs image bin) */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static float h2f(unsigned short u){ __fp16 h; memcpy(&h,&u,2); return (float)h; }
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    if(!ork_ppu_fuse_enabled(c)){printf("PPU fuse not enabled — SKIP\n");ork_npu_free(c);return 0;}
    int M=8,N=64;
    unsigned short *g=calloc(M*N,2), *r=calloc(4096,2); double us=0;
    int rc=ork_npu_replay_reshape_f16(c,g,M*N,r,4096,&us);
    printf("replay_reshape rc=%d  %.0fus\n",rc,us);
    int ginj=getenv("ORK_RESHAPE_GINJ")!=0;
    if(!ginj){ printf("  (set ORK_RESHAPE_T0=2 ORK_RESHAPE_GINJ=1 to derive the permutation)\n"); free(g);free(r);ork_npu_free(c);return rc?1:0; }
    /* input[i] = fp16(i+1). For each reshape-out position p, recover the source input index. */
    /* Hypothesis A: PCH16 atom-8  a8(m,n)=(n/8)*(M*8)+m*8+(n%8).  Hypothesis B: TRANSPOSE t(m,n)=n*M+m. */
    int okA=0,okB=0,mapped=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){
        int i=m*N+n;                                  /* input index (contiguous [M][N]) */
        /* find where fp16(i+1) landed */
        int found=-1; for(int p=0;p<M*N;p++){ int idx=(int)lroundf(h2f(r[p]))-1; if(idx==i){found=p;break;} }
        if(found<0) continue; mapped++;
        int a8=(n/8)*(M*8)+m*8+(n%8);
        int tr=n*M+m;
        if(found==a8)okA++;
        if(found==tr)okB++;
        if(mapped<=16) printf("   in(m=%d,n=%d)=%d -> r[%d]   PCH16=%d transpose=%d\n",m,n,i,found,a8,tr);
    }
    printf("  mapped=%d/%d   PCH16(atom8) match=%d   TRANSPOSE(n*M+m) match=%d\n",mapped,M*N,okA,okB);
    free(g);free(r); ork_npu_free(c);
    return rc?1:0;
}
