/* tools/fused_mtile_check.c — validate the raised fused-SiLU M-tile cap (mg_max*64) by SELF-CONSISTENCY:
 * run ork_i8_mm_run_silu at M>64 with the big tile (mc=mg_max*64, one submit) vs forced mc=64 (2+ submits),
 * on identical inputs. The mc=64 path is already bit-exact vs the probe (tools/fused_silu_test), and the plain
 * matmul is bit-exact at mg_max*64 (AGENTS.md), so big-tile == 64-tile => the big tile is correct.
 *   make fused_mtile_check && sudo ./fused_mtile_check [M] [K] [N]   (board only)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):128, K=argc>2?atoi(argv[2]):2048, N=argc>3?atoi(argv[3]):64;
    ork_npu*c=ork_npu_init(); if(!c){printf("no board\n");return 0;}
    int8_t*B=malloc((size_t)K*N), *A=malloc((size_t)M*K);
    int8_t*o64=malloc((size_t)M*N), *oBig=malloc((size_t)M*N);
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=(int8_t)((i*7)%13-6);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(int8_t)((i*11)%17-8);
    ork_w*w=ork_i8_mm_pack(c,K,N,B); if(!w){printf("pack failed\n");return 2;}
    const int RM=0x51aa,RS=0x14; const unsigned OB=0xffffff9fu,IO=0xffffc000u,C4=0x56391100u;

    setenv("ORK_FUSED_MTILE","64",1);
    int r1=ork_i8_mm_run_silu(c,w,M,A,o64,RM,RS,OB,IO,C4,NULL,0);
    unsetenv("ORK_FUSED_MTILE");                     /* default = mg_max*64 (one big tile) */
    int r2=ork_i8_mm_run_silu(c,w,M,A,oBig,RM,RS,OB,IO,C4,NULL,0);
    if(r1||r2){ printf("run failed r64=%d rBig=%d\n",r1,r2); return 1; }

    int mism=0,maxe=0; for(int i=0;i<M*N;i++){int e=abs(o64[i]-oBig[i]); if(e){mism++; if(e>maxe)maxe=e;}}
    printf("M=%d K=%d N=%d  mc=64 vs mc=mg_max*64:  mism=%d/%d  max|e|=%d  %s\n",
           M,K,N,mism,M*N,maxe, mism?"FAIL (big tile miscomputes!)":"OK (big tile bit-exact == 64-tile)");
    ork_mm_free(c,w); ork_npu_free(c);
    return mism?1:0;
}
