/* swreduce_probe — reproduce the orkd reduce bug DIRECT (no daemon), to escape the orkd-over-ssh wall.
 * The daemon's reduce is a SW-path MM_I8 (K=64, K%512!=0 -> seq_hw_ok false -> run()) that runs AFTER an
 * exp_i8 (SDP-LUT) in the resident chain, and returns rc=0 with a STALE output (its submit never wrote Cc).
 * If a DIRECT [exp_i8 -> MM_I8(K=64)] reproduces the garbage, the bug is matmul-after-SDP at the SW path and
 * is debuggable without orkd. Tests, same ork_submit_seq dispatch the daemon uses:
 *   T1 (control): MM_I8 K=64 reduce alone (fresh context)         -> expect CORRECT (minimal_i8mm proved this)
 *   T2 (after SDP): exp_i8 (own seq) THEN MM_I8 K=64 reduce (own seq, same ctx) -> does it go stale/garbage?
 *   sudo env ORK_MM_TIMEOUT=3000 timeout 60 ./swreduce_probe [M] [K] [N]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static uint32_t g=0x2f9a1u; static int r8(void){ g=g*1664525u+1013904223u; return (int)((g>>26)%50); }  /* [0,49] like exp out */
static int reduce_and_check(ork_npu*c,ork_w*w,int M,int K,int N,const char*tag){
    int8_t *A=malloc((size_t)M*K); for(size_t i=0;i<(size_t)M*K;i++)A[i]=(int8_t)r8();
    int32_t *C=malloc((size_t)M*N*4); for(size_t i=0;i<(size_t)M*N;i++)C[i]=0x7f7f7f7f;   /* poison to detect no-write */
    ork_seq_op op={ .kind=ORK_OP_MM_I8, .w=w, .M=M, .A=A, .C=C };
    int rc=ork_submit_seq(c,&op,1);
    int bad=0; long me=0;
    for(int m=0;m<M;m++){ long S=0; for(int k=0;k<K;k++)S+=A[(size_t)m*K+k];
        for(int n=0;n<N;n++){ long er=labs((long)C[(size_t)m*N+n]-S); if(er>me)me=er; if(er>0)bad++; } }
    printf("  [%s] MM_I8 reduce K=%d rc=%d max|err|=%ld %s (%d/%d)  C[0]=%d want=%ld\n",
           tag,K,rc,me,(rc||bad)?"MISMATCH":"OK",bad,M*N,C[0],(long)({long S=0;for(int k=0;k<K;k++)S+=A[k];S;}));
    free(A); free(C); return rc||bad;
}
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):64, K=argc>2?atoi(argv[2]):64, N=argc>3?atoi(argv[3]):32;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("swreduce_probe: M=%d K=%d N=%d (DIRECT; reduce alone vs reduce-after-exp_i8)\n",M,K,N);
    int8_t *ones=malloc((size_t)K*N); memset(ones,1,(size_t)K*N);
    ork_w *w=ork_i8_mm_pack(c,K,N,ones); if(!w){printf("pack fail\n");return 2;}
    int fail=0;

    /* T1: reduce alone (fresh context) */
    fail |= reduce_and_check(c,w,M,K,N,"T1 alone");

    /* T2: exp_i8 (SDP-LUT) first, then the reduce (separate seqs, same context) */
    { int Ne=64; int8_t *xi=malloc((size_t)M*Ne), *eo=malloc((size_t)M*Ne);
      for(size_t i=0;i<(size_t)M*Ne;i++) xi[i]=(int8_t)-(int)(r8()%40);   /* <=0 for exp */
      ork_seq_op e={ .kind=ORK_OP_EXP_I8, .M=M, .N=Ne, .A=xi, .C=eo, .in_scale=0.05, .out_scale=1.0/127.0 };
      int erc=ork_submit_seq(c,&e,1);
      printf("  [exp_i8] rc=%d eo[0]=%d (SDP-LUT mode set)\n",erc,eo[0]);
      free(xi); free(eo); }
    fail |= reduce_and_check(c,w,M,K,N,"T2 after-exp");

    printf("%s\n", fail?"REPRODUCED or FAIL — see which test":"BOTH OK (bug not reproduced direct -> needs orkd context)");
    ork_mm_free(c,w); ork_npu_free(c);
    return fail;
}
