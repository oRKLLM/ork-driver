/* crack_tiling.c — #38: reverse rkllm's int8 weight TILING by controlled one-hot probes.
 * Runs rkllm's captured regcmd (mm_regcmd.txt) on ork's silicon with a weight buffer that is ZERO except
 * a single 1 at offset p, and an A that ENCODES k per row:
 *   A[0][k]=1  A[1][k]=k&0x7f  A[2][k]=(k>>7)&0x7f
 * With a one-hot weight at logical (k*,n*): C[m][n*] = A[m][k*], else 0. So the nonzero column = n*, and
 * k* = C[2][n*]*128 + C[1][n*]. => offset p maps to logical (k*,n*). Probe a strategic set of p to deduce
 * the tiling. int8-warm first (raw foreign submit needs int8 mode or it wedges).
 *   ./crack_tiling            (built-in probe set)
 *   ./crack_tiling p0 p1 ...  (explicit offsets)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int M=0,K=0,N=0; { FILE*f=fopen("/tmp/mm_meta.txt","r"); if(!f){fprintf(stderr,"no meta\n");return 1;}
        char k[32]; int v; while(fscanf(f,"%31s %d",k,&v)==2){ if(!strcmp(k,"M"))M=v; else if(!strcmp(k,"K"))K=v; else if(!strcmp(k,"N"))N=v;} fclose(f); }
    unsigned *rc=malloc(4096*sizeof(unsigned)); int rn=0;
    { FILE*f=fopen("/tmp/mm_regcmd.txt","r"); if(!f){fprintf(stderr,"no regcmd\n");return 1;} while(rn<4096&&fscanf(f,"%x",&rc[rn])==1)rn++; fclose(f); }
    if(M<3){ fprintf(stderr,"need M>=3 (have %d) to encode k; recapture a larger-M program\n",M); return 1; }
    size_t WB=(size_t)K*N;                       /* weight amount = 0x1030 = K*N (tiles exactly, K%32=N%32=0) */
    printf("crack: M=%d K=%d N=%d rn=%d WB=%zu\n",M,K,N,rn,WB);

    ork_npu*c=ork_npu_init(); if(!c){printf("init (board only)\n");return 77;}
    /* int8 warm */
    ork_npu_set_core_budget(c,1);
    { int8_t*a=malloc((size_t)M*K),*b=malloc((size_t)K*N); int32_t*cc=malloc((size_t)M*N*4);
      memset(a,1,(size_t)M*K); memset(b,1,(size_t)K*N); ork_w*w=ork_i8_mm_pack(c,K,N,b);
      if(w){ ork_i8_mm_run(c,w,M,a,cc); ork_mm_free(c,w);} free(a);free(b);free(cc); }
    printf("int8 warmed\n");

    int8_t*A=calloc((size_t)M*K,1);
    for(int k=0;k<K;k++){ A[0*K+k]=1; A[1*K+k]=(int8_t)(k&0x7f); A[2*K+k]=(int8_t)((k>>7)&0x7f); }
    int8_t*B=calloc(WB,1);
    int32_t*C=calloc((size_t)M*N,4);

    /* range mode: ./crack_tiling range START COUNT [STRIDE]  -> compact "p k n c0" lines for offline analysis */
    long rstart=-1,rcount=0,rstride=1;
    if(argc>=4 && !strcmp(argv[1],"range")){ rstart=atol(argv[2]); rcount=atol(argv[3]); if(argc>4)rstride=atol(argv[4]); }
    int builtin[]={0,1,2,3,4,5,6,7, 30,31,32,33,34, 62,63,64,65, 126,127,128,129,
                   255,256,257, 1023,1024, 1215,1216,1217, 2431,2432, 3583,3584,
                   (int)WB-1};
    int np = (rstart>=0) ? (int)rcount : (argc>1 ? argc-1 : (int)(sizeof(builtin)/sizeof(int)));
    if(rstart<0) printf("probe: p -> (k,n)  [C0 should be 1]\n");
    for(int i=0;i<np;i++){
        long p = (rstart>=0) ? (rstart + (long)i*rstride) : (argc>1 ? atol(argv[i+1]) : builtin[i]);
        if(p<0||(size_t)p>=WB) continue;
        memset(B,0,WB); B[p]=1;
        memset(C,0,(size_t)M*N*4);
        double us; int r=ork_i8_npu_replay(c,rc,rn,M,K,N,A,(int)((size_t)M*K),B,(int)WB,C,1,&us);
        if(r){ printf("  p=%ld replay rc=%d\n",p,r); continue; }
        int ns=-1; for(int n=0;n<N;n++) if(C[0*N+n]!=0){ ns=n; break; }
        if(ns<0){ if(rstart>=0) printf("%ld -1 -1 0\n",p); else printf("  p=%-8ld -> (no output lit; C0 all zero)\n",p); continue; }
        int ks = C[2*N+ns]*128 + (C[1*N+ns]&0x7f);
        if(rstart>=0) printf("%ld %d %d %d\n",p,ks,ns,C[0*N+ns]);
        else printf("  p=%-8ld -> k=%-5d n=%-5d   [C0=%d C1=%d C2=%d]\n",p,ks,ns,C[0*N+ns],C[1*N+ns],C[2*N+ns]);
    }
    ork_npu_free(c); return 0;
}
