/* two_stream_f16_probe — reproduce the attention "two-consecutive stream_f16_chain (N differs)"
 * pattern that produced garbage pre-refactor, and check coherence on v0.7.0 (ork_npu_enter).
 *
 * Attention per head-group: QK^T (weight K^T[HD,L2] -> S[M,L2], N=L2) IMMEDIATELY followed by
 * A.V (weight V[L2,HD] -> O[M,HD], N=HD). L2 != HD, so two back-to-back stream_f16_chain calls
 * see a DIFFERENT N. All-ones operands => C[m][n] = sum_k 1 = K (=HD for QK^T, =L2 for A.V).
 * Self-validating: nonzero exit on any mismatch. BOARD: sudo ./two_stream_f16_probe */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { S = 3, M = 64, HD = 128, L2 = 256 };   /* HD != L2 : the trigger */

static int check(const char*tag,float*C,int rows,int n,int expect){
    int bad=0; for(int i=0;i<rows*n;i++){ int v=(int)(C[i]+0.5f); if(v!=expect){ if(bad<3) printf("  %s mismatch @%d: got %g expect %d\n",tag,i,C[i],expect); bad++; } }
    printf("  %-6s rows=%d n=%d : expect %d  bad=%d/%d  %s\n",tag,rows,n,expect,bad,rows*n,bad?"FAIL":"ok");
    return bad;
}

int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    ork_f16 *KT=malloc((size_t)HD*L2*2), *V=malloc((size_t)L2*HD*2);
    ork_f16 *Q=malloc((size_t)M*HD*2),  *P=malloc((size_t)M*L2*2);
    for(size_t i=0;i<(size_t)HD*L2;i++) KT[i]=(ork_f16)1.0f;
    for(size_t i=0;i<(size_t)L2*HD;i++) V[i]=(ork_f16)1.0f;
    for(size_t i=0;i<(size_t)M*HD;i++)  Q[i]=(ork_f16)1.0f;
    for(size_t i=0;i<(size_t)M*L2;i++)  P[i]=(ork_f16)1.0f;
    /* QK^T weights: pack K^T as [HD, L2] (K=HD, N=L2); A.V weights: pack V as [L2, HD] (K=L2, N=HD) */
    ork_w *wK[S], *wV[S];
    for(int i=0;i<S;i++){ wK[i]=ork_f16_mm_pack(c,HD,L2,KT); wV[i]=ork_f16_mm_pack(c,L2,HD,V);
        if(!wK[i]||!wV[i]){printf("pack failed\n");return 2;} }
    float *Sc[S], *Ov[S];
    for(int i=0;i<S;i++){ Sc[i]=malloc((size_t)M*L2*4); Ov[i]=malloc((size_t)M*HD*4); }

    int bad=0;
    for(int iter=0; iter<4; iter++){
        ork_mm_task_f16 qk[S], av[S];
        for(int i=0;i<S;i++){ memset(Sc[i],0,(size_t)M*L2*4); memset(Ov[i],0,(size_t)M*HD*4);
            qk[i]=(ork_mm_task_f16){wK[i],M,Q,Sc[i]}; av[i]=(ork_mm_task_f16){wV[i],M,P,Ov[i]}; }
        int r1=ork_f16_mm_run_stream_chain(c,S,qk);   /* QK^T : N=L2 */
        int r2=ork_f16_mm_run_stream_chain(c,S,av);   /* A.V  : N=HD (back-to-back, N changed) */
        printf("iter %d (rc %d,%d):\n",iter,r1,r2);
        bad += check("QK^T",Sc[0],M,L2,HD);   /* sum over HD ones = HD */
        bad += check("A.V", Ov[0],M,HD,L2);   /* sum over L2 ones = L2 */
    }
    ork_npu_free(c);
    printf(bad?"\nRESULT: FAIL (%d mismatches) — two-stream bug still present\n":"\nRESULT: OK — two-consecutive stream_f16_chain coherent\n",bad);
    return bad?1:0;
}
