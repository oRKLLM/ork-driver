/* sdp_chain_probe — int8 SDP on the HW chain.
 *  [fwd]        : ewmul as a MIDDLE task walks forward (ork_npu_chain_progs, desc_slot=138).
 *  [seq-hetero] : Stage 1 mechanism — [matmul->ewmul->matmul] NONBLOCK, terminal-matmul sentinel (hand-built).
 *  [seq-api]    : Stage 2 — the SAME chain via the reusable ork_dyn_begin_seq_i8 / ork_dyn_seq_end API
 *                 (ork_seq_op[], packed weight, per-op copy-back). Board only. */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int seq_api_test(ork_npu *c){
    const int M=8, K=512, N=64, mult=0x4000, shift=14;
    int8_t *W=malloc((size_t)K*N); for(int i=0;i<K*N;i++) W[i]=1;               /* all-ones -> matmul C=K */
    ork_w *w=ork_mm_pack_i8(c,K,N,W); free(W); if(!w){ printf("[seq-api] pack failed\n"); return 1; }
    int8_t *A0=malloc((size_t)M*K); for(int i=0;i<M*K;i++) A0[i]=1;              /* matmul A all-ones */
    int8_t A1[512],B1[512],ref[512]; unsigned g=333;
    for(int i=0;i<M*N;i++){ A1[i]=(int8_t)(((g=g*1103515245u+12345u)>>20&7))-3; B1[i]=(int8_t)(((g=g*1103515245u+12345u)>>20&7))-3; }
    for(int i=0;i<M*N;i++){ long v=lround((long)A1[i]*B1[i]*mult/(double)(1<<shift)); ref[i]=(int8_t)(v>127?127:v<-128?-128:v); }
    int32_t *C0=calloc(M*N,4), *C2=calloc(M*N,4); int8_t O1[512];
    memset(O1,0,sizeof O1);
    ork_seq_op ops[3];
    memset(ops,0,sizeof ops);
    ops[0]=(ork_seq_op){.kind=ORK_OP_MM_I8,   .w=w, .M=M, .N=N, .A=A0, .C=C0};
    ops[1]=(ork_seq_op){.kind=ORK_OP_EWMUL_I8,.w=NULL,.M=M,.N=N, .A=A1, .B=B1, .C=O1, .mult=mult, .shift=shift};
    ops[2]=(ork_seq_op){.kind=ORK_OP_MM_I8,   .w=w, .M=M, .N=N, .A=A0, .C=C2};
    ork_dyn_chain *h=ork_dyn_begin_seq_i8(c,3,ops);
    if(!h){ printf("[seq-api] begin_seq_i8 returned NULL (ineligible)\n"); free(A0);free(C0);free(C2); return 1; }
    int rc=ork_dyn_seq_end(h);
    int n0=0,n2=0,ne=0;
    for(int i=0;i<M*N;i++){ if(C0[i]==K)n0++; if(C2[i]==K)n2++; }
    for(int i=0;i<M*N;i++) if(O1[i]==ref[i]) ne++;
    int ok = (rc==0)&&(n0==M*N)&&(ne==M*N)&&(n2==M*N);
    printf("[seq-api]    rc=%d  matmul0 %d/%d  ewmul %d/%d  matmul2 %d/%d  -> %s\n",
           rc, n0,M*N, ne,M*N, n2,M*N, ok?"OK":"BAD");
    free(A0);free(C0);free(C2);
    return ok?0:1;
}

/* Stage 3: G independent [matmul->ewmul->matmul] groups spread across nc cores; all bit-exact. */
static int seq_mc_test(ork_npu *c, int G, int nc){
    const int M=8, K=512, N=64, mult=0x4000, shift=14;
    int8_t *W=malloc((size_t)K*N); for(int i=0;i<K*N;i++) W[i]=1;
    ork_w *w=ork_mm_pack_i8(c,K,N,W); free(W); if(!w){ printf("[seq-mc] pack failed\n"); return 1; }
    int8_t *A0=malloc((size_t)M*K); for(int i=0;i<M*K;i++) A0[i]=1;
    int nops=3*G;
    ork_seq_op *ops=calloc(nops,sizeof *ops);
    int *gstart=malloc((G+1)*sizeof(int));
    int8_t (*A1)[512]=malloc((size_t)G*512), (*B1)[512]=malloc((size_t)G*512), (*ref)[512]=malloc((size_t)G*512);
    int32_t (*C0)[512]=malloc((size_t)G*512*4), (*C2)[512]=malloc((size_t)G*512*4); int8_t (*O)[512]=malloc((size_t)G*512);
    for(int g=0; g<G; g++){ unsigned s=1000u+g*7u;
        for(int i=0;i<M*N;i++){ A1[g][i]=(int8_t)(((s=s*1103515245u+12345u)>>20&7))-3; B1[g][i]=(int8_t)(((s=s*1103515245u+12345u)>>20&7))-3; }
        for(int i=0;i<M*N;i++){ long v=lround((long)A1[g][i]*B1[g][i]*mult/(double)(1<<shift)); ref[g][i]=(int8_t)(v>127?127:v<-128?-128:v); }
        gstart[g]=3*g;
        ops[3*g+0]=(ork_seq_op){.kind=ORK_OP_MM_I8,   .w=w, .M=M, .N=N, .A=A0, .C=C0[g]};
        ops[3*g+1]=(ork_seq_op){.kind=ORK_OP_EWMUL_I8,.w=NULL,.M=M,.N=N, .A=A1[g], .B=B1[g], .C=O[g], .mult=mult, .shift=shift};
        ops[3*g+2]=(ork_seq_op){.kind=ORK_OP_MM_I8,   .w=w, .M=M, .N=N, .A=A0, .C=C2[g]};
    }
    gstart[G]=nops;
    ork_dyn_chain *h=ork_dyn_begin_seq_i8_mc(c,nops,ops,G,gstart,nc);
    int ok=0, rc=-1;
    if(!h){ printf("[seq-mc] begin_seq_i8_mc NULL (ineligible)\n"); }
    else { rc=ork_dyn_seq_end(h); ok=(rc==0);
        for(int g=0; g<G; g++){ int n0=0,n2=0,ne=0;
            for(int i=0;i<M*N;i++){ if(C0[g][i]==K)n0++; if(C2[g][i]==K)n2++; if(O[g][i]==ref[g][i])ne++; }
            if(!(n0==M*N&&n2==M*N&&ne==M*N)){ ok=0; printf("[seq-mc]   group %d BAD: mm0=%d ew=%d mm2=%d /%d\n",g,n0,ne,n2,M*N); } }
        printf("[seq-mc]     rc=%d  %d groups x [mm->ewmul->mm] across %d cores -> %s\n", rc, G, nc, ok?"OK":"BAD"); }
    free(A0);free(ops);free(gstart);free(A1);free(B1);free(ref);free(C0);free(C2);free(O);
    return ok?0:1;
}

int main(void){
    ork_npu *c=ork_npu_init();
    if(!c){ fprintf(stderr,"init failed\n"); return 2; }
    int t0=-1,t1=-1;
    int rc=ork_npu_probe_sdp_chain_fwd(c,&t0,&t1);
    printf("[fwd]        rc=%d  ewmul0(middle)=%s  ewmul1(walked)=%s\n", rc, t0?"OK":"BAD", t1?"OK":"BAD");
    int okh=-1;
    int rc2=ork_npu_probe_seq_hetero(c,&okh);
    printf("[seq-hetero] rc=%d  all bit-exact=%s\n", rc2, okh?"OK":"BAD");
    int api=seq_api_test(c);
    int mc=seq_mc_test(c,3,3);
    ork_npu_free(c);
    int pass = (rc==0&&t0&&t1&&rc2==0&&okh&&api==0&&mc==0);
    printf("VERDICT: %s\n", pass?"int8 SDP chains ride the NONBLOCK doorbell, single + multi-core (Stage 3). PASS":"FAIL");
    return pass?0:1;
}
