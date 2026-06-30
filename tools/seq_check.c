/* tools/seq_check.c — reproduce the mixed-K layer/model failure minimally: run a SEQUENCE of int8
 * matmuls (the layer's NPU ops: Q/K/V/O @ K=512, gate/up @ K=512, down @ K=2048) on ONE context,
 * multi-core, and CPU-reference-check EACH. Pinpoints which matmul miscomputes in-sequence (vs
 * bit-exact in isolation) → the cross-matmul state interaction that cbuf=57344 triggers.
 *   make seq_check && sudo ./seq_check [M] [reps]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ork_npu.h"

typedef struct { const char*tag; int K,N; int8_t*B; int8_t*A; ork_w*w; } mm;

static int run_check(ork_npu*c, mm*m, int M){
    int K=m->K,N=m->N; int32_t*C=malloc((size_t)M*N*4);
    ork_mm_run_i8(c,m->w,M,m->A,C);
    int mism=0; long maxe=0;
    for(int i=0;i<M;i++)for(int j=0;j<N;j++){
        long acc=0; for(int k=0;k<K;k++) acc+=(long)m->A[(size_t)i*K+k]*m->B[(size_t)k*N+j];
        long d=(long)C[(size_t)i*N+j]-acc; if(d<0)d=-d; if(d>maxe)maxe=d; if(C[(size_t)i*N+j]!=acc)mism++;
    }
    printf("  %-6s K=%-4d N=%-4d : mism=%d maxerr=%ld %s\n", m->tag,K,N,mism,maxe, mism?"FAIL":"OK");
    free(C); return mism?1:0;
}

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):512, reps=argc>2?atoi(argv[2]):2;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    int cores=ork_npu_cores(c); ork_npu_set_core_budget(c,cores);
    printf("INIT soc=%s cores=%d M=%d (cbuf-dependent: K=2048 multi-core tile)\n", ork_npu_soc(c),cores,M);

    /* the layer's NPU matmul sequence (K,N) */
    int KN[][2]={ {512,512},{512,512},{512,512},{512,512},{512,2048},{512,2048},{2048,512} };
    const char*tags[]={"Q","K","V","O","gate","up","down"};
    int ns=7; mm seq[7];
    for(int s=0;s<ns;s++){
        int K=KN[s][0],N=KN[s][1];
        seq[s].tag=tags[s]; seq[s].K=K; seq[s].N=N;
        seq[s].B=malloc((size_t)K*N); for(size_t i=0;i<(size_t)K*N;i++) seq[s].B[i]=(int8_t)((i%5)-2);
        seq[s].A=malloc((size_t)M*K); for(size_t i=0;i<(size_t)M*K;i++) seq[s].A[i]=(int8_t)(((i+s)%7)-3);
        ork_npu_set_pack_domain(c,0);
        seq[s].w=ork_mm_pack_i8(c,K,N,seq[s].B); if(!seq[s].w){printf("pack %s failed\n",tags[s]);return 1;}
    }
    int fail=0;
    for(int r=0;r<reps;r++){
        printf("--- sequence pass %d ---\n", r);
        for(int s=0;s<ns;s++) fail |= run_check(c, &seq[s], M);
    }
    printf("%s\n", fail?"SEQUENCE FAIL":"SEQUENCE OK");
    ork_npu_free(c); return fail;
}
