/* percore_fd_probe — does per-core-fd isolation change the concurrent-fetch "wedge"? Runs a 3-core fp16
 * N-column-split matmul where EACH core submits on its OWN DRM fd (a fresh open of the card). mode 0 = each
 * core gets its own weight copy; mode 1 = one shared dma-heap weight imported into every core's fd. Fills the
 * full C[M,N], compares against a CPU fp16 reference (accumulate in float), and prints maxerr/maxrel + OK vs
 * WEDGE/WRONG. BOARD: sudo env ORK_MM_TIMEOUT=3000 ./percore_fd_probe [M K N [mode]] */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static unsigned sd=12345; static int rnd(void){ sd=sd*1103515245u+12345u; return (int)((sd>>16)%4); }

int main(int argc,char**argv){
    int M   = argc>1?atoi(argv[1]):16;
    int K   = argc>2?atoi(argv[2]):512;
    int N   = argc>3?atoi(argv[3]):96;
    int mode= argc>4?atoi(argv[4]):0;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    ork_f16 *A=malloc((size_t)M*K*sizeof *A), *B=malloc((size_t)K*N*sizeof *B);
    float   *C=malloc((size_t)M*N*sizeof *C);
    if(!A||!B||!C){printf("oom\n");return 1;}
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(ork_f16)(((int)rnd()-1)*0.25);   /* small fp16 values {-0.25,0,0.25,0.5} */
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=(ork_f16)(((int)rnd()-1)*0.25);
    for(size_t i=0;i<(size_t)M*N;i++) C[i]=0.0f;

    double us=0;
    int rc=ork_npu_f16_percore_probe(c,M,K,N,A,B,C,&us,mode);
    if(rc<0){ printf("PCFD M=%d K=%d N=%d mode=%d : rc=%d (setup error)\n",M,K,N,mode,rc); ork_npu_free(c); return 1; }

    double maxerr=0, maxrel=0;                                    /* CPU fp16 reference: accumulate in float */
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){
        double acc=0; for(int k=0;k<K;k++) acc+=(double)(float)A[(size_t)m*K+k]*(double)(float)B[(size_t)k*N+n];
        double got=C[(size_t)m*N+n], err=fabs(got-acc);
        if(err>maxerr) maxerr=err;
        double den=fabs(acc); if(den>1e-3){ double rel=err/den; if(rel>maxrel) maxrel=rel; }
    }
    int ok = (maxrel<0.02) || (maxerr<0.05);                      /* fp16 tolerance */
    printf("PCFD M=%d K=%d N=%d mode=%d : us=%.1f maxerr=%.4f maxrel=%.4f %s\n",
           M,K,N,mode,us,maxerr,maxrel, ok?"OK":"WEDGE/WRONG");
    free(A); free(B); free(C); ork_npu_free(c);
    return ok?0:1;
}
