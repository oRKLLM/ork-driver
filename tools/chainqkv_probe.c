/* chainqkv_probe — task #20 HW CHAIN, attention FRONT: Q,K,V = xn.{Wq,Wk,Wv} as ONE coalesced submit
 * (run_chain_i8 batches the 3 independent projections). This is the front's first HW-chain piece; the full
 * front chain would prepend RMSNorm (x^2->reduce->rsqrt->scale, a rsqrt-curve chain). NOTE: this front is a
 * SEPARATE chain from the attention core (chainav_probe) — QK^T needs K transposed+packed as a static weight
 * (dynamic K can't be a chain weight) and rope isn't a chain kind, so rope + K^T-pack are a host bridge
 * between the front chain and the core chain.  sudo env ORK_MM_TIMEOUT=3000 ./chainqkv_probe [M] [d] [dh]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static uint32_t g=0x71a3u; static int r8(void){ g=g*1664525u+1013904223u; return (int)((g>>26)%7)-3; }  /* [-3,3] */
static int check(const char*tag,const int32_t*C,const int8_t*A,const int8_t*W,int M,int K,int N){
    int bad=0; long me=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long acc=0; for(int k=0;k<K;k++)acc+=(long)A[(size_t)m*K+k]*W[(size_t)k*N+n];
        long er=labs((long)C[(size_t)m*N+n]-acc); if(er>me)me=er; if(er>0)bad++; }
    printf("  [%s] int32 out vs CPU: max|err|=%ld %s (%d/%d)\n",tag,me,bad?"MISMATCH":"OK",bad,M*N);
    return bad;
}
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):32, d=argc>2?atoi(argv[2]):512, dh=argc>3?atoi(argv[3]):128;   /* d=hidden (K%512), dh=head dim */
    setvbuf(stdout,0,_IONBF,0);
    if(d%512){ printf("d must be %%512 (K constraint)\n"); return 2; }
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("chainqkv_probe: M=%d d=%d dh=%d (HW chain FRONT: Q,K,V = xn.W in ONE submit)\n",M,d,dh);
    int8_t *xn=malloc((size_t)M*d), *Wq=malloc((size_t)d*dh), *Wk=malloc((size_t)d*dh), *Wv=malloc((size_t)d*dh);
    for(size_t i=0;i<(size_t)M*d;i++)xn[i]=(int8_t)r8();
    for(size_t i=0;i<(size_t)d*dh;i++){ Wq[i]=(int8_t)r8(); Wk[i]=(int8_t)r8(); Wv[i]=(int8_t)r8(); }
    ork_w *wq=ork_mm_pack_i8(c,d,dh,Wq), *wk=ork_mm_pack_i8(c,d,dh,Wk), *wv=ork_mm_pack_i8(c,d,dh,Wv);
    if(!wq||!wk||!wv){ printf("pack fail\n"); return 2; }
    int32_t *Q=calloc((size_t)M*dh,4), *K=calloc((size_t)M*dh,4), *V=calloc((size_t)M*dh,4);
    ork_mm_task_i8 tasks[3] = { { wq, M, xn, Q }, { wk, M, xn, K }, { wv, M, xn, V } };   /* 3 independent projections */
    int rc=ork_mm_run_chain_i8(c,3,tasks);   /* ONE coalesced submit */
    printf("  run_chain_i8([Q,K,V]) rc=%d  Q[0]=%d K[0]=%d V[0]=%d\n", rc, Q[0], K[0], V[0]);
    if(rc){ printf("FAIL rc=%d\n",rc); return 1; }
    int fail=0;
    fail |= check("Q=xn.Wq",Q,xn,Wq,M,d,dh);
    fail |= check("K=xn.Wk",K,xn,Wk,M,d,dh);
    fail |= check("V=xn.Wv",V,xn,Wv,M,d,dh);
    printf("%s\n", fail?"FAIL":"PASS — QKV front (Q,K,V) as ONE HW-chain submit, bit-exact (rope + K^T-pack bridge to the core chain)");
    ork_mm_free(c,wq); ork_mm_free(c,wk); ork_mm_free(c,wv); ork_npu_free(c);
    return fail;
}
