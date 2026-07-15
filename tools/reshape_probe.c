/* reshape_probe — WIP: FULL-CHAIN REPLAY validation of the vendor fp16 contiguous->atom-8 reshape.
 * Replays task0-10 (input-convert + GEMM + reshape) from the captured vendor image, then verifies IN-PLACE
 * that the reshape output (atom-8 @0xffff0a00) is a correct rearrangement of the GEMM output (contiguous
 * @0xffff0000). No weight extraction / CPU ref needed: reshape_out[atom8(m,n)] must equal gemm_out[m][n].
 * BOARD: sudo env ORK_EW_TIMEOUT=2000 ./reshape_probe   (needs gemm_mul_image.bin in cwd) */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    if(!ork_ppu_fuse_enabled(c)){printf("PPU fuse not enabled — SKIP\n");ork_npu_free(c);return 0;}
    int M=8,N=64;
    unsigned short *g=calloc(M*N,2), *r=calloc(4096,2); double us=0;
    int rc=ork_npu_replay_reshape_f16(c,g,M*N,r,4096,&us);
    printf("replay_reshape rc=%d  %.0fus\n",rc,us);
    int gnz=0,rnz=0; for(int i=0;i<M*N;i++)if(g[i])gnz++; for(int i=0;i<4096;i++)if(r[i])rnz++;
    printf("  gemm_out nonzero=%d/%d   reshape_out nonzero=%d\n",gnz,M*N,rnz);
    /* verify reshape_out is atom-8 of gemm_out, trying the PCH16 formula */
    int ok=0,tot=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int a8=(n/8)*(M*8)+m*8+(n%8); tot++;
        if(r[a8]==g[m*N+n]) ok++; }
    printf("  atom-8 (PCH16) match: %d/%d\n",ok,tot);
    /* if PCH16 wrong, derive the empirical permutation: for each gemm elem, where does its value land? */
    if(ok<tot){
        printf("  deriving layout (gemm[m][n] -> reshape pos), first 24:\n");
        int shown=0;
        for(int m=0;m<M && shown<24;m++)for(int n=0;n<N && shown<24;n++){
            unsigned short v=g[m*N+n]; if(!v)continue;
            for(int p=0;p<4096;p++) if(r[p]==v){ printf("    g[%d][%d]=0x%04x -> r[%d]  (PCH16=%d)\n",m,n,v,p,(n/8)*(M*8)+m*8+(n%8)); shown++; break; }
        }
    }
    free(g);free(r); ork_npu_free(c);
    return (rc==0&&ok==tot)?0:1;
}
