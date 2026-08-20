/* softmax_i8_split_probe — isolate the resident-int8-softmax reduce bug by SPLITTING the chain into two
 * separate orkd seqs so the intermediate e is returnable (not A2-resident):
 *   Stage A: seq [MATMUL_REQUANT_I8(K-pad) -> EXP_I8(biased)]  -> e RETURNED   (tests exp-after-matmul in-chain)
 *   Stage B: seq [MM_I8 reduce] on the returned e              -> Sigma        (tests the reduce alone)
 * If A's e matches CPU but B's Sigma is wrong -> the reduce (under the 3-op resident chain) is the bug.
 * If A's e is wrong -> exp-after-requant-matmul is the bug. Run through orkd (wedge-free).
 *   sudo env ORK_USE_ORKD=1 ORKD_BIN=./orkd ORK_MM_TIMEOUT=3000 timeout 100 ./softmax_i8_split_probe [N] [d]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static uint32_t g=0x50f7a5u; static int r8lo(void){ g=g*1664525u+1013904223u; return (int)((g>>26)%9)-4; }
int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):64, d=argc>2?atoi(argv[2]):128, Kp=512;
    setvbuf(stdout,0,_IONBF,0);
    if(!getenv("ORK_USE_ORKD")){ printf("run with ORK_USE_ORKD=1 (direct wedges)\n"); return 2; }
    if(d>Kp||N%32){ printf("need d<=512, N%%32\n"); return 2; }
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("softmax_i8_split_probe: N=%d d=%d (ORKD; isolate exp vs reduce)\n",N,d);
    int mult=0x4000, shift=18; double in_scale=0.05, out_scale=1.0/127.0;
    int8_t *Q=malloc((size_t)N*d), *Kk=malloc((size_t)N*d);
    for(size_t i=0;i<(size_t)N*d;i++){ Q[i]=(int8_t)r8lo(); Kk[i]=(int8_t)r8lo(); }
    int8_t *Qp=calloc((size_t)N*Kp,1), *KTp=calloc((size_t)Kp*N,1);
    for(int i=0;i<N;i++)for(int k=0;k<d;k++) Qp[(size_t)i*Kp+k]=Q[(size_t)i*d+k];
    for(int k=0;k<d;k++)for(int j=0;j<N;j++) KTp[(size_t)k*N+j]=Kk[(size_t)j*d+k];
    int8_t *csc=malloc((size_t)N*N); int gmax=-128;
    for(int m=0;m<N;m++)for(int j=0;j<N;j++){ long acc=0; for(int k=0;k<d;k++)acc+=(long)Q[(size_t)m*d+k]*Kk[(size_t)j*d+k];
        long q=(acc*mult)>>shift; if(q>127)q=127; if(q<-128)q=-128; csc[(size_t)m*N+j]=(int8_t)q; if(q>gmax)gmax=(int)q; }
    int8_t *ecpu=malloc((size_t)N*N);
    for(size_t i=0;i<(size_t)N*N;i++){ double v=exp(((double)csc[i]-gmax)*in_scale)/out_scale; long q=lround(v); if(q>127)q=127; if(q<-128)q=-128; ecpu[i]=(int8_t)q; }
    printf("  gmax=%d\n",gmax);

    { int8_t wi[32],wo[32]; for(int i=0;i<32;i++) wi[i]=(int8_t)(i-16); ork_i8_npu_exp(c,wi,1,32,in_scale,out_scale,wo,NULL); }
    ork_w *w_kt=ork_i8_mm_pack(c,Kp,N,KTp); if(!w_kt){printf("pack KT fail\n");return 2;}
    int8_t *ones=malloc((size_t)N*32); memset(ones,1,(size_t)N*32);
    ork_w *w_ones=ork_i8_mm_pack(c,N,32,ones); if(!w_ones){printf("pack ones fail\n");return 2;}
    int8_t *scores=malloc((size_t)N*N), *e=malloc((size_t)N*N); int32_t *ss=malloc((size_t)N*32*4);
    for(size_t i=0;i<(size_t)N*N;i++){ scores[i]=-128; e[i]=-128; }
    int fail=0;

    /* STAGE A: [requant -> exp], e returned (op1 is terminal, NOT aliased -> shipped back) */
    { ork_seq_op a[2]={ { .kind=ORK_OP_MATMUL_REQUANT_I8, .w=w_kt, .M=N, .A=Qp, .C=scores, .mult=mult, .shift=shift },
                        { .kind=ORK_OP_EXP_I8, .M=N, .N=N, .A=scores, .C=e, .in_scale=in_scale, .out_scale=out_scale, .b_scale=(double)gmax } };
      int rc=ork_submit_seq(c,a,2);
      int bad=0; double me=0; if(rc){ printf("  [A] rc=%d\n",rc); fail=1; }
      else { for(size_t i=0;i<(size_t)N*N;i++){ double er=fabs((double)e[i]-ecpu[i]); if(er>me)me=er; if(er>4)bad++; } }
      printf("  [A] [requant->exp] e vs CPU: rc=%d max|err|=%.0f LSB %s (%d/%d)  e[0]=%d(want %d)\n",rc,me,bad?"MISMATCH":"OK",bad,N*N,(int)e[0],(int)ecpu[0]);
      if(bad)fail=1; }

    /* STAGE B: [reduce] on the NPU e (fresh seq, e uploaded as normal input) */
    { ork_seq_op b={ .kind=ORK_OP_MM_I8, .w=w_ones, .M=N, .A=e, .C=ss };
      int rc=ork_submit_seq(c,&b,1);
      int bad=0; double me=0; if(rc){ printf("  [B] rc=%d\n",rc); fail=1; }
      else for(int m=0;m<N;m++){ long Sc=0; for(int j=0;j<N;j++) Sc+=e[(size_t)m*N+j]; double er=fabs((double)ss[(size_t)m*32]-Sc); double rel=er/(Sc>0?Sc:1); if(rel>me)me=rel; if(rel>0.001)bad++; }
      printf("  [B] [reduce] Sigma vs CPU sum(e_npu): rc=%d max rel-err=%.3e %s (%d/%d)  ss[0]=%d\n",rc,me,bad?"MISMATCH":"OK",bad,N,ss[0]);
      if(bad)fail=1; }

    printf("%s\n", fail?"DIAG: see which stage failed above":"BOTH STAGES OK (bug is specific to the 3-op resident chain)");
    ork_mm_free(c,w_kt); ork_mm_free(c,w_ones); ork_npu_free(c);
    return fail;
}
