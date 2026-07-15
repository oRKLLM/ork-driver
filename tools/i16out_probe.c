/* discriminating test: does a STANDALONE int8-matmul with int16 output (probe_i16_out, validate_regcmd path,
 * dom=-1) work? If yes, the int16-out matmul is fine standalone and the i16-seq bug is my hand-rolled submit. */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char**argv){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    int M=8,K=32,N=64;
    if(argc>=4){ M=atoi(argv[1]); K=atoi(argv[2]); N=atoi(argv[3]); }
    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); short*out=malloc((size_t)M*N*2);
    unsigned s=5;
    for(int i=0;i<M*K;i++){s=s*1103515245+12345;A[i]=(int8_t)((s>>16)&1);}
    for(int i=0;i<K*N;i++){s=s*1103515245+12345;B[i]=(int8_t)((s>>16)&1);}
    double us=0;
    int rc=ork_npu_probe_i16_out(c,M,K,N,A,B,0x4000,14,out,&us);   /* unit-gain int16 out */
    int bad=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int acc=0; for(int k=0;k<K;k++) acc+=(int)A[(size_t)m*K+k]*(int)B[(size_t)k*N+n];
        if(out[(size_t)m*N+n]!=acc){ if(bad<4)printf("  [%d][%d] NPU=%d ref=%d\n",m,n,out[(size_t)m*N+n],acc); bad++; } }
    printf("standalone int16-out matmul: rc=%d  %d/%d exact  %.0f us  %s\n",rc,M*N-bad,M*N,us,(rc==0&&!bad)?"OK — standalone int16-out WORKS":"CHECK");
    ork_npu_free(c); return (rc==0&&!bad)?0:1;
}
