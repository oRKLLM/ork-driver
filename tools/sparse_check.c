/* sparse_check.c — verify NPU int8 matmul correctness at a shape whose FULL O(M*N*K) ref is too slow.
 * Runs ork_mm_run_i8(M,K,N) on `cores` cores, then checks a SPREAD of sampled (row,col) cells with a
 * full-K dot product (cheap: ~samples*K MACs). Catches M-tile/K-slice boundary miscompute fast.
 *   sparse_check M K N [cores]   (env ORK_TMM_CORES ignored; pass cores arg) */
#include <stdio.h>
#include <stdlib.h>
#include "ork_npu.h"
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):256, K=argc>2?atoi(argv[2]):18944, N=argc>3?atoi(argv[3]):3584, cores=argc>4?atoi(argv[4]):3;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    ork_npu_set_core_budget(c,cores);
    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t*C=malloc((size_t)M*N*4);
    unsigned s=12345; for(size_t i=0;i<(size_t)M*K;i++){s=s*1103515245u+12345u;A[i]=(int8_t)(((s>>16)%7)-3);}
    for(size_t i=0;i<(size_t)K*N;i++){s=s*1103515245u+12345u;B[i]=(int8_t)(((s>>16)%5)-2);}
    ork_w*w=ork_mm_pack_i8(c,K,N,B); if(!w){printf("pack failed\n");return 1;}
    if(ork_mm_run_i8(c,w,M,A,C)){printf("run failed\n");return 1;}
    /* sample rows spread across M (incl. the M-tile boundaries 63/64/127/128) and cols across N */
    int rows[]={0,1,63,64,65,127,128,129,200,255, M-1}; int nr=sizeof(rows)/sizeof(rows[0]);
    int cols[]={0,1,63,64,N/2,N-65,N-64,N-1}; int ncl=sizeof(cols)/sizeof(cols[0]);
    int mism=0; long maxe=0;
    for(int ri=0;ri<nr;ri++){int i=rows[ri]; if(i>=M)continue;
        for(int ci=0;ci<ncl;ci++){int n=cols[ci]; if(n>=N)continue;
            long ref=0; for(int k=0;k<K;k++) ref+=(long)A[(size_t)i*K+k]*B[(size_t)k*N+n];
            long got=C[(size_t)i*N+n], d=got-ref; if(d<0)d=-d; if(d>maxe)maxe=d;
            if(got!=ref){mism++; if(mism<=4)printf("  MISM (%d,%d): got %ld ref %ld\n",i,n,got,ref);}
        }}
    printf("sparse M=%d K=%d N=%d cores=%d: samples=%d mism=%d maxerr=%ld %s\n",
           M,K,N,cores,nr*ncl,mism,maxe,mism?"FAIL":"OK");
    ork_w_free(w); ork_npu_free(c); return mism?1:0;
}
