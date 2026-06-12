/* tools/ksubmit_probe.c — confirm the single-submit K-tile ceiling (int8) on this SoC.
 *
 * The M=1 full-K single submit (decode fast path) works up to a per-op K-tile cap. fp16
 * (0x1044 = K/32) tops out at K=8192 (256 tiles); int8 (0x1044 = ceil(K/64)) packs 2x denser,
 * so the same 256-tile cap predicts an int8 ceiling at K=16384. This sweeps K and validates
 * each single-submit result against a CPU reference: it should be correct while ceil(K/64) <=
 * 256, and WEDGE beyond. Confirms the limit is a hardware tile count, not raw K.
 *   make ksubmit_probe && sudo ./ksubmit_probe
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ork_npu.h"
static unsigned sd=999; static int rnd(void){sd=sd*1103515245+12345;return (int)((sd>>16)%5)-2;}
int main(int argc,char**argv){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed (NPU?)\n");return 1;}
    int dflt[]={6144,8192,11008,12288,16384,16448,18944};   /* tiles: 96,128,172,192,256,257,296 */
    int Ka[32],*Ks=dflt, KMAX=18944;
    int N=64, NK=sizeof(dflt)/sizeof(*dflt);
    const char*ne=getenv("ORK_PROBE_N"); if(ne)N=atoi(ne);     /* vary N to test K-ceiling vs N */
    if(argc>1){ NK=argc-1<32?argc-1:32; KMAX=0; for(int i=0;i<NK;i++){Ka[i]=atoi(argv[i+1]);if(Ka[i]>KMAX)KMAX=Ka[i];} Ks=Ka; }
    printf("(N=%d)\n",N);
    printf("SoC %s — single-submit int8 K ceiling (0x1044 = ceil(K/64) tiles; predicted cap 256 -> K<=16384)\n",
           ork_npu_soc(c));
    int8_t*A=malloc(KMAX),*B=malloc((size_t)KMAX*N); int32_t*C=malloc((size_t)N*4);
    int fail=0;
    for(int i=0;i<NK;i++){int K=Ks[i],tiles=(K+63)/64;
        for(int j=0;j<K;j++)A[j]=(int8_t)rnd();
        for(size_t j=0;j<(size_t)K*N;j++)B[j]=(int8_t)rnd();
        int rc=ork_npu_probe_single_i8(c,K,N,A,B,C);
        if(rc==0){ int bad=0;
            for(int n=0;n<N;n++){int32_t ref=0;for(int k=0;k<K;k++)ref+=(int)A[k]*(int)B[(size_t)k*N+n];if(C[n]!=ref)bad++;}
            printf("  K=%-6d %3d tiles : %s\n",K,tiles,bad?"COMPLETED BUT WRONG":"OK (single submit, correct)");
            if(bad)fail=1;
        } else if(rc==-1){ printf("  K=%-6d %3d tiles : WEDGED (submit timeout — exceeds tile cap)\n",K,tiles); }
        else printf("  K=%-6d %3d tiles : bad dims\n",K,tiles);
    }
    free(A);free(B);free(C); ork_npu_free(c);
    printf("%s\n",fail?"FAIL (a correct-range K was wrong)":"done");
    return fail;
}
