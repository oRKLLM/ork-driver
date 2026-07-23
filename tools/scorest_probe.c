/* scorest_probe — method (b), realized as operand-swap: compute scores^T = K . Q^T instead of Q . K^T, so K
 * (the big KV-cache) is the ACTIVATION (read linearly, NO transpose/pack) and only Q^T is the packed weight.
 * For decode (Nq=1) Q^T == Q in memory -> a trivial [d,1] pack; the large K never gets the host K^T densify+pack.
 * Standard int8 matmul (no special HW mode): A=K_pad[Nk,512], W=pack(Q^T_pad[512,Nq]) -> scores^T[Nk,Nq].
 * Validates scores^T[j,i] == QK^T[i,j].  sudo env ORK_MM_TIMEOUT=3000 ./scorest_probe [Nq] [Nk] [d]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static uint32_t g=0x33b1u; static int q2(void){ g=g*1664525u+1013904223u; return (int)((g>>27)%5)-2; }
int main(int argc,char**argv){
    int Nq=argc>1?atoi(argv[1]):32, Nk=argc>2?atoi(argv[2]):512, d=argc>3?atoi(argv[3]):128, Kp=512;
    setvbuf(stdout,0,_IONBF,0);
    if(Nq%32){ printf("Nq must be %%32 (weight N)\n"); return 2; }
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("scorest_probe: Nq=%d Nk=%d d=%d (scores^T = K.Q^T; K is the ACTIVATION, no K^T pack)\n",Nq,Nk,d);
    int8_t *Q=malloc((size_t)Nq*d), *K=malloc((size_t)Nk*d);
    for(size_t i=0;i<(size_t)Nq*d;i++)Q[i]=(int8_t)q2();
    for(size_t i=0;i<(size_t)Nk*d;i++)K[i]=(int8_t)q2();
    /* A = K padded on the contraction dim d->512 (activation; linear, cheap zero-pad — NOT a transpose/pack) */
    int8_t *Ka=calloc((size_t)Nk*Kp,1);
    for(int j=0;j<Nk;j++)for(int k=0;k<d;k++) Ka[(size_t)j*Kp+k]=K[(size_t)j*d+k];
    /* W = Q^T padded: [512, Nq] with rows k<d = Q[i][k], i.e. QTp[k][i]=Q[i][k]; k>=d zero */
    int8_t *QTp=calloc((size_t)512*Nq,1);
    for(int k=0;k<d;k++)for(int i=0;i<Nq;i++) QTp[(size_t)k*Nq+i]=Q[(size_t)i*d+k];
    ork_w *w_qt=ork_mm_pack_i8(c,512,Nq,QTp); if(!w_qt){ printf("pack Q^T fail\n"); return 2; }
    int32_t *st=calloc((size_t)Nk*Nq,4);
    ork_mm_task_i8 t={ w_qt, Nk, Ka, st };   /* M=Nk (keys), K=512, N=Nq -> scores^T[Nk,Nq] */
    int rc=ork_mm_run_chain_i8(c,1,&t);      /* single int8 matmul (multi-core auto-tuner) */
    printf("  matmul(A=K_pad, W=Q^T) rc=%d  st[0]=%d\n", rc, st[0]);
    if(rc){ printf("FAIL rc=%d\n",rc); return 1; }
    int bad=0; long me=0;
    for(int j=0;j<Nk;j++)for(int i=0;i<Nq;i++){ long acc=0; for(int k=0;k<d;k++)acc+=(long)Q[(size_t)i*d+k]*K[(size_t)j*d+k];
        long er=labs((long)st[(size_t)j*Nq+i]-acc); if(er>me)me=er; if(er>0)bad++; }
    printf("  scores^T[j,i] vs CPU QK^T[i,j]: max|err|=%ld %s (%d/%d)\n", me, bad?"MISMATCH":"OK", bad, Nk*Nq);
    printf("%s\n", bad?"FAIL":"PASS — scores^T=K.Q^T bit-exact; K is the activation (no K^T pack) — decode: only trivial Q^T pack");
    ork_mm_free(c,w_qt); ork_npu_free(c);
    return bad;
}
