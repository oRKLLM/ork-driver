/* test_orkd_2conn_seq — MULTI-CONSUMER orkd with SEQ workloads: ONE process, N connections (argv[1], default 2),
 * each submitting a heterogeneous GROUPED op-sequence through the daemon, round-robin interleaved.
 *
 * ★ FINDING (2026-07-19): N=2 is solid (bit-exact, any iter count). N>=3 HANGS (exit 124, no NPU wedge) after a
 * few submits — and it hangs with ORK_2CONN_G0=1 (ungrouped/SW-break) TOO, so it's the DAEMON's >=2-client
 * poll/dispatch (queue v1 was only validated at 2 clients), NOT the grouped-seq doorbell or the op path. The
 * fix belongs to A-sched (task #17, the queue/scheduler rework). Until then, orkd multi-consumer is proven at 2.
 * ORK_2CONN_G0=1 forces ungrouped ops (isolates the daemon from the doorbell).
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
    int grp = getenv("ORK_2CONN_G0") ? 0 : 1;   /* G0=1 => ungrouped (SW-break path) — isolates the grouped-seq doorbell from the daemon's N-client handling */
    ops[0].kind=ORK_OP_MM_I8;    ops[0].w=w; ops[0].M=TM; ops[0].N=TN; ops[0].A=A0; ops[0].C=C0; ops[0].group=grp;
    ops[1].kind=ORK_OP_EWMUL_I8;             ops[1].M=TM; ops[1].N=TN; ops[1].A=ua; ops[1].B=va; ops[1].C=O1; ops[1].mult=mult; ops[1].shift=shift; ops[1].group=grp;
    ops[2].kind=ORK_OP_MM_I8;    ops[2].w=w; ops[2].M=TM; ops[2].N=TN; ops[2].A=Ac; ops[2].C=C2; ops[2].group=grp;
    if(ork_submit_seq(c, ops, 3)){ fprintf(stderr,"[%c] it=%d submit_seq FAIL\n", tag, it); return 1; }
    int bad=0;
    for(int i=0;i<TM*TN;i++) if(C0[i]!=R0[i]){ if(!bad)fprintf(stderr,"[%c] it=%d mm0 [%d] %d!=%d\n",tag,it,i,C0[i],R0[i]); bad=1; }
    for(int i=0;i<TM*TN;i++) if(O1[i]!=Eref[i]){ if(!bad)fprintf(stderr,"[%c] it=%d ewmul [%d] %d!=%d\n",tag,it,i,O1[i],Eref[i]); bad=1; }
    for(int i=0;i<TM*TN;i++) if(C2[i]!=R2[i]){ if(!bad)fprintf(stderr,"[%c] it=%d mm2 [%d] %d!=%d\n",tag,it,i,C2[i],R2[i]); bad=1; }
    return bad;
}

int main(int argc, char **argv){
    int NC = argc>1?atoi(argv[1]):2; if(NC<1)NC=1; if(NC>8)NC=8;   /* number of concurrent consumers (connections) */
    int iters = argc>2?atoi(argv[2]):6; if(iters<1)iters=1;
    setenv("ORK_USE_ORKD","1",1);
    fprintf(stderr,"[Nconn-seq] %d connections, each submits grouped [mm->ewmul->mm] SEQ, round-robin interleaved (%d iters)\n", NC, iters);
    ork_npu *ctx[8]; static int8_t B[8][TK*TN]; ork_w *w[8]; static int8_t A0[TM*TK], Ac[TM*TK], ua[TM*TN], va[TM*TN];
    unsigned g=0xC0FFEE;
    #define R8() ((int8_t)(((g=g*1103515245u+12345u)>>18&0x1f)-16))
    #define RSDP() ((int8_t)(((g=g*1103515245u+12345u)>>20&7)-3))
    for(int i=0;i<NC;i++){ ctx[i]=ork_npu_init();                    /* sequential init: conn 0 auto-spawns orkd, rest connect */
        if(!ctx[i]){ fprintf(stderr,"conn %d init failed\n",i); for(int j=0;j<i;j++) ork_npu_free(ctx[j]); return 2; } }
    for(int i=0;i<NC;i++){ for(int k=0;k<TK*TN;k++) B[i][k]=R8(); w[i]=ork_mm_pack_i8(ctx[i], TK, TN, B[i]); }   /* distinct resident weight per consumer */
    int bad=0;
    for(int it=0; it<iters && !bad; it++){
        for(int i=0;i<NC && !bad;i++){                              /* round-robin the NC consumers each iter */
            for(int k=0;k<TM*TK;k++) A0[k]=R8(); for(int k=0;k<TM*TK;k++) Ac[k]=R8();
            for(int k=0;k<TM*TN;k++) ua[k]=RSDP(); for(int k=0;k<TM*TN;k++) va[k]=RSDP();
            if(seq_verify(ctx[i], w[i], B[i], A0, Ac, ua, va, (char)('0'+i), it)) bad=1;
        }
    }
    for(int i=0;i<NC;i++){ if(w[i]) ork_mm_free(ctx[i], w[i]); ork_npu_free(ctx[i]); }
    #undef R8
    #undef RSDP
    printf("MULTI_CONSUMER_NCONN_SEQ: %s — %d connections, round-robin grouped [mm->ewmul->mm] seq, %d iters%s\n",
           bad?"FAIL":"PASS", NC, iters, bad?"":", all consumers bit-exact");
    return bad?1:0;
}
