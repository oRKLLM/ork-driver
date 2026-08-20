/* tools/st_validate.c — bit-exact check for the ORK_SMALLTILE prefill packing.
 * For each chain-prefill shape (int8, M>1, K%512==0, K<=4096) it runs run_i8 and compares the
 * NPU output to a CPU int32 reference (exact integer product). The harness is meant to be run
 * TWICE by the caller: once with ORK_SMALLTILE unset (per-core large tiles, the trusted path)
 * and once with ORK_SMALLTILE=1 (small uniform tiles) — both MUST report ok and the SAME output.
 * It also writes the C matrix to a file (argv[1]) so the two runs can be diffed byte-for-byte.
 *   make st_validate && sudo ./st_validate out.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ork_npu.h"

static unsigned long s=12345; static int rnd(void){ s=s*1103515245ul+12345ul; return (int)((s>>16)&0xff); }

int main(int argc,char**argv){
    const char*outpath = argc>1?argv[1]:NULL;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    ork_npu_set_core_budget(c,3);
    /* chain-prefill (K<=4096) shapes from real 7B prefill matmuls + a few stressors */
    int shapes[][3] = { {256,3584,3584}, {256,2048,2048}, {128,512,1536}, {200,1024,2048}, {256,3584,512} };
    int ns=(int)(sizeof(shapes)/sizeof(shapes[0]));
    int fail=0;
    FILE*f = outpath?fopen(outpath,"wb"):NULL;
    for(int si=0;si<ns;si++){
        int M=shapes[si][0],K=shapes[si][1],N=shapes[si][2];
        int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t*C=malloc((size_t)M*N*4);
        for(size_t i=0;i<(size_t)M*K;i++)A[i]=(int8_t)(rnd()-128);
        for(size_t i=0;i<(size_t)K*N;i++)B[i]=(int8_t)(rnd()-128);
        ork_w*w=ork_i8_mm_pack(c,K,N,B);
        if(!w){printf("pack failed %d,%d,%d\n",M,K,N);return 1;}
        if(ork_i8_mm_run(c,w,M,A,C)){printf("run failed %d,%d,%d\n",M,K,N);return 1;}
        /* CPU ref: only spot-check a stride of rows/cols to stay fast but cover all M-tiles/N-subtiles */
        long bad=0;
        int rstep = M>32?(M/32):1, cstep = N>64?(N/64):1;
        for(int i=0;i<M && bad<5;i+=rstep)for(int n=0;n<N;n+=cstep){
            long ref=0; for(int k=0;k<K;k++)ref+=(long)A[(size_t)i*K+k]*(long)B[(size_t)k*N+n];
            if(C[(size_t)i*N+n]!=(int32_t)ref){bad++; if(bad<=3)printf("  mism @ (%d,%d): got %d ref %ld\n",i,n,C[(size_t)i*N+n],ref);}
        }
        printf("  %s M=%d K=%d N=%d\n", bad?"WRONG":"ok  ", M,K,N); fflush(stdout);
        if(bad)fail=1;
        if(f) fwrite(C,4,(size_t)M*N,f);
        ork_w_free(w);free(A);free(B);free(C);
    }
    if(f)fclose(f);
    printf("%s\n", fail?"FAIL":"ALL OK");
    ork_npu_free(c);
    return fail?1:0;
}
