/* test_orkd_2conn_seq — MULTI-CONSUMER orkd with SEQ workloads: ONE process, TWO connections, each submitting
 * a heterogeneous GROUPED op-sequence through the daemon, interleaved.
 *
 * Extends test_orkd_2conn from plain matmul to ork_submit_seq: each consumer submits a 3-op grouped chain
 * [MM_I8 -> EWMUL_I8(SDP) -> MM_I8] (group=1 => rides ork_dyn_begin_seq_i8_mc on the daemon). Interleaving the
 * two consumers' ork_submit_seq calls exercises the daemon's SEQ handler (handle_seq) serializing two live
 * consumers, the grouped-chain routing, and the Path-B group wire-forwarding — all under contention on the
 * single NPU. Distinct per-consumer+per-iter data; every op verified bit-exact vs the CPU reference.
 *
 *   make test_orkd_2conn_seq && sudo env ORK_USE_ORKD=1 ORKD_BIN=$PWD/orkd ./test_orkd_2conn_seq [iters]
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TM 8
#define TK 512
#define TN 64

/* submit one grouped seq [mm(A0)->ewmul(ua,va)->mm(Ac)] on connection c (weight w = B); verify all three ops. */
static int seq_verify(ork_npu *c, ork_w *w, const int8_t *B,
                      const int8_t *A0, const int8_t *Ac, const int8_t *ua, const int8_t *va,
                      char tag, int it){
    const int mult=0x4000, shift=14;
    int32_t C0[TM*TN], C2[TM*TN], R0[TM*TN], R2[TM*TN]; int8_t O1[TM*TN], Eref[TM*TN];
    for(int m=0;m<TM;m++)for(int n=0;n<TN;n++){ long a=0,b=0; for(int k=0;k<TK;k++){ a+=(long)A0[m*TK+k]*B[k*TN+n]; b+=(long)Ac[m*TK+k]*B[k*TN+n]; } R0[m*TN+n]=(int)a; R2[m*TN+n]=(int)b; }
    for(int i=0;i<TM*TN;i++){ long v=lround((long)ua[i]*va[i]*mult/(double)(1<<shift)); Eref[i]=(int8_t)(v>127?127:v<-128?-128:v); }
    memset(C0,0,sizeof C0); memset(C2,0,sizeof C2); memset(O1,0,sizeof O1);
    ork_seq_op ops[3]; memset(ops,0,sizeof ops);
    ops[0].kind=ORK_OP_MM_I8;    ops[0].w=w; ops[0].M=TM; ops[0].N=TN; ops[0].A=A0; ops[0].C=C0; ops[0].group=1;
    ops[1].kind=ORK_OP_EWMUL_I8;             ops[1].M=TM; ops[1].N=TN; ops[1].A=ua; ops[1].B=va; ops[1].C=O1; ops[1].mult=mult; ops[1].shift=shift; ops[1].group=1;
    ops[2].kind=ORK_OP_MM_I8;    ops[2].w=w; ops[2].M=TM; ops[2].N=TN; ops[2].A=Ac; ops[2].C=C2; ops[2].group=1;
    if(ork_submit_seq(c, ops, 3)){ fprintf(stderr,"[%c] it=%d submit_seq FAIL\n", tag, it); return 1; }
    int bad=0;
    for(int i=0;i<TM*TN;i++) if(C0[i]!=R0[i]){ if(!bad)fprintf(stderr,"[%c] it=%d mm0 [%d] %d!=%d\n",tag,it,i,C0[i],R0[i]); bad=1; }
    for(int i=0;i<TM*TN;i++) if(O1[i]!=Eref[i]){ if(!bad)fprintf(stderr,"[%c] it=%d ewmul [%d] %d!=%d\n",tag,it,i,O1[i],Eref[i]); bad=1; }
    for(int i=0;i<TM*TN;i++) if(C2[i]!=R2[i]){ if(!bad)fprintf(stderr,"[%c] it=%d mm2 [%d] %d!=%d\n",tag,it,i,C2[i],R2[i]); bad=1; }
    return bad;
}

int main(int argc, char **argv){
    int iters = argc>1?atoi(argv[1]):6; if(iters<1)iters=1;
    setenv("ORK_USE_ORKD","1",1);
    fprintf(stderr,"[2conn-seq] TWO connections, each submits grouped [mm->ewmul->mm] SEQ, interleaved (%d iters)\n", iters);
    ork_npu *cA = ork_npu_init(); if(!cA){ fprintf(stderr,"conn A init failed\n"); return 2; }
    ork_npu *cB = ork_npu_init(); if(!cB){ fprintf(stderr,"conn B init failed\n"); ork_npu_free(cA); return 2; }
    static int8_t BA[TK*TN], BB[TK*TN], A0[TM*TK], Ac[TM*TK], ua[TM*TN], va[TM*TN];
    unsigned g=0xC0FFEE;
    #define R8() ((int8_t)(((g=g*1103515245u+12345u)>>18&0x1f)-16))
    #define RSDP() ((int8_t)(((g=g*1103515245u+12345u)>>20&7)-3))
    for(int i=0;i<TK*TN;i++) BA[i]=R8();
    for(int i=0;i<TK*TN;i++) BB[i]=R8();
    ork_w *wA = ork_mm_pack_i8(cA, TK, TN, BA);
    ork_w *wB = ork_mm_pack_i8(cB, TK, TN, BB);
    int bad=0;
    for(int it=0; it<iters && !bad; it++){
        for(int i=0;i<TM*TK;i++) A0[i]=R8(); for(int i=0;i<TM*TK;i++) Ac[i]=R8();
        for(int i=0;i<TM*TN;i++) ua[i]=RSDP(); for(int i=0;i<TM*TN;i++) va[i]=RSDP();
        if(seq_verify(cA, wA, BA, A0, Ac, ua, va, 'A', it)) bad|=1;
        for(int i=0;i<TM*TK;i++) A0[i]=R8(); for(int i=0;i<TM*TK;i++) Ac[i]=R8();
        for(int i=0;i<TM*TN;i++) ua[i]=RSDP(); for(int i=0;i<TM*TN;i++) va[i]=RSDP();
        if(seq_verify(cB, wB, BB, A0, Ac, ua, va, 'B', it)) bad|=2;
    }
    if(wA) ork_mm_free(cA, wA);
    if(wB) ork_mm_free(cB, wB);
    ork_npu_free(cA);
    ork_npu_free(cB);
    #undef R8
    #undef RSDP
    printf("MULTI_CONSUMER_2CONN_SEQ: %s — two connections, interleaved grouped [mm->ewmul->mm] seq, %d iters%s\n",
           bad?"FAIL":"PASS", iters, bad?"":", both consumers bit-exact");
    return bad?1:0;
}
