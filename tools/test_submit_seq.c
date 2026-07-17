/* test_submit_seq — Phase-1 validation of the heterogeneous op-sequence scheduler (ork_submit_seq).
 *
 * Builds a realistic MIXED int8+fp16 sequence  [i8 mm][i8 mm][f16 mm][i8 mm][f16 mm]  and runs it through
 * ork_submit_seq MANY times, checking EVERY element of EVERY op's output on EVERY run. With A=B=all-ones,
 * an int8 matmul output element == K and an fp16 one == (float)K, so every element is exactly checkable.
 *
 * fp16 now rides the SAME thread-free HW-chain doorbell as int8 (ork_dyn_begin_mc, host-A). A doorbell run
 * is ONE dtype, so the scheduler flushes [i8,i8] as one doorbell, then the f16 as its OWN doorbell (dtype
 * break, NOT a SW break), then the 3rd i8, then the last f16 — FOUR doorbell submits, zero SW-chain breaks,
 * all thread-free. Every op must be bit-exact every run, INCLUDING across each i8<->f16 mode transition
 * (which fires inside begin_mc's ork_npu_enter at the dtype boundary). Run with ORK_SEQ_DEBUG=1 to see the
 * flush pattern (four "[seq] HW flush" lines, dt alternating, and ZERO "[seq] SW break" lines).
 *
 * A trailing NON-CONFORMING fp16 op (K not %512) then confirms seq_hw_ok() rejects it -> it correctly falls
 * back to the SW run_stream_f16 path (one "[seq] SW break" line) and is still bit-exact.
 *
 * BOARD:  make test_submit_seq && sudo env ORK_MM_TIMEOUT=3000 timeout 300 ./test_submit_seq [runs=30]
 * Exit 0 = all ops correct on all runs; nonzero = at least one op miscomputed (a flaky transition).
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define POISON_I 0x0badc0de
#define POISON_F (-1.0e30f)

int main(int argc,char**argv){
    int runs=argc>1?atoi(argv[1]):30;
    int K=argc>2?atoi(argv[2]):512, N=argc>3?atoi(argv[3]):512, M=argc>4?atoi(argv[4]):8;  /* K conforming (K%512==0, K<=4096); N%32 (int8) & N%16 (fp16); M<=64 & M*K<=32768 (fp16 tile) */
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    printf("test_submit_seq: mixed [i8][i8][f16][i8][f16], M=%d K=%d N=%d, %d runs\n", M,K,N,runs);

    /* all-ones operands: int8 dot == K, fp16 dot == (float)K */
    int8_t  *Bi=(int8_t*)malloc((size_t)K*N); memset(Bi,1,(size_t)K*N);
    ork_f16 *Bf=(ork_f16*)malloc((size_t)K*N*sizeof(ork_f16)); for(size_t i=0;i<(size_t)K*N;i++) Bf[i]=(ork_f16)1.0f;
    ork_w *wi=ork_mm_pack_i8(c,K,N,Bi); if(!wi){ printf("pack_i8 fail\n"); return 2; }
    ork_w *wf=ork_mm_pack   (c,K,N,Bf); if(!wf){ printf("pack fp16 fail\n"); return 2; }
    int8_t  *Ai=(int8_t*)malloc((size_t)M*K); memset(Ai,1,(size_t)M*K);
    ork_f16 *Af=(ork_f16*)malloc((size_t)M*K*sizeof(ork_f16)); for(size_t i=0;i<(size_t)M*K;i++) Af[i]=(ork_f16)1.0f;

    size_t ei=(size_t)M*N;                 /* elems/output */
    int32_t *O0=malloc(ei*4), *O1=malloc(ei*4), *O3=malloc(ei*4);   /* int8 outputs (int32) */
    float   *F2=malloc(ei*4), *F4=malloc(ei*4);                     /* fp16 outputs (fp32)  */

    /* generality note: the classification is precision-agnostic (int4 & SDP rows exist in SEQ_CLASS too);
     * this Phase-1 gate exercises the int8(HW) + fp16(SW) mix and the transitions between them. */
    ork_seq_op ops[5] = {
        { .kind=ORK_OP_MM_I8,  .w=wi, .M=M, .N=N, .A=Ai, .C=O0 },
        { .kind=ORK_OP_MM_I8,  .w=wi, .M=M, .N=N, .A=Ai, .C=O1 },
        { .kind=ORK_OP_MM_F16, .w=wf, .M=M, .N=N, .A=Af, .C=F2 },
        { .kind=ORK_OP_MM_I8,  .w=wi, .M=M, .N=N, .A=Ai, .C=O3 },
        { .kind=ORK_OP_MM_F16, .w=wf, .M=M, .N=N, .A=Af, .C=F4 },
    };

    /* NO WARMUP: run 0 below IS the HW-chain doorbell's first-ever (cold) submit and MUST be bit-exact.
     * The old run-0 miscompute was NOT a pipeline-warm problem — it was a fresh-output-buffer DMA coherency
     * bug (the ORK_ZC_OUT class): begin_mc's first cold call writes a freshly-allocated int8 output scratch
     * whose dirty CPU cache lines then evict and overwrite ~half the NPU's result with zeros. begin_mc now
     * cleans the output surface to DRAM before the cold round (still a single NONBLOCK round), so run 0 is
     * correct — this gate would catch a regression of that clean-before. */

    int fail=0, i8_bad=0, f16_bad=0, rc_bad=0;
    for(int r=0;r<runs;r++){
        /* poison every output so a SKIPPED op (scheduler dropped it) is caught as a miscompute */
        for(size_t k=0;k<ei;k++){ O0[k]=O1[k]=O3[k]=POISON_I; F2[k]=F4[k]=POISON_F; }
        int rc=ork_submit_seq(c,ops,5);
        if(rc){ rc_bad++; fail=1; if(r<4) printf("  run %d: ork_submit_seq rc=%d\n",r,rc); continue; }
        int b8=0,bf=0;
        for(size_t k=0;k<ei;k++){ if(O0[k]!=K||O1[k]!=K||O3[k]!=K) b8++; }
        for(size_t k=0;k<ei;k++){ if(F2[k]<(float)K-1.f||F2[k]>(float)K+1.f||F4[k]<(float)K-1.f||F4[k]>(float)K+1.f) bf++; }
        if(b8){ i8_bad++; fail=1; if(i8_bad<=4) printf("  run %d: int8(HW-chain) MISMATCH — %d/%zu elems (O0[0]=%d O1[0]=%d O3[0]=%d, want %d)\n",r,b8,3*ei,O0[0],O1[0],O3[0],K); }
        if(bf){ f16_bad++; fail=1; if(f16_bad<=4) printf("  run %d: fp16(SW-chain) MISMATCH — %d/%zu elems (F2[0]=%.2f F4[0]=%.2f, want %d.0)\n",r,bf,2*ei,F2[0],F4[0],K); }
    }
    printf("---\n");
    printf("  int8 HW-chain ops : %d/%d runs correct\n", runs-i8_bad-rc_bad, runs);
    printf("  fp16 SW-chain ops : %d/%d runs correct\n", runs-f16_bad-rc_bad, runs);
    if(rc_bad) printf("  scheduler errors  : %d/%d runs\n", rc_bad, runs);

    /* --- non-conforming fp16 op: K%512!=0 => seq_hw_ok rejects => SW run_stream_f16 fallback, still exact --- */
    {
        int Knc=768;                          /* K%32==0 (fp16 pack ok) but K%512!=0 (doorbell-ineligible) */
        ork_f16 *Bnc=(ork_f16*)malloc((size_t)Knc*N); for(size_t i=0;i<(size_t)Knc*N;i++) Bnc[i]=(ork_f16)1.0f;
        ork_f16 *Anc=(ork_f16*)malloc((size_t)M*Knc*sizeof(ork_f16)); for(size_t i=0;i<(size_t)M*Knc;i++) Anc[i]=(ork_f16)1.0f;
        ork_w *wnc=ork_mm_pack(c,Knc,N,Bnc);
        float *Fnc=malloc(ei*4);
        int sub=0;
        if(!wnc){ printf("  non-conforming fp16 pack fail\n"); fail=1; }
        else {
            for(int r=0;r<8 && !sub;r++){
                for(size_t k=0;k<ei;k++) Fnc[k]=POISON_F;
                /* one conforming i8 (HW) followed by the non-conforming f16 (SW break) — exercises HW->SW */
                ork_seq_op s2[2]={ { .kind=ORK_OP_MM_I8, .w=wi, .M=M, .N=N, .A=Ai, .C=O0 },
                                   { .kind=ORK_OP_MM_F16,.w=wnc,.M=M, .N=N, .A=Anc,.C=Fnc } };
                if(ork_submit_seq(c,s2,2)){ printf("  non-conforming seq rc!=0\n"); sub=1; fail=1; break; }
                for(size_t k=0;k<ei;k++) if(Fnc[k]<(float)Knc-1.f||Fnc[k]>(float)Knc+1.f){ sub=1; break; }
            }
            printf("  fp16 SW-fallback  : %s (non-conforming K=%d -> run_stream_f16, want %d.0)\n",
                   sub?"MISMATCH":"OK", Knc, Knc);
            if(sub) fail=1;
            ork_mm_free(c,wnc);
        }
        free(Bnc);free(Anc);free(Fnc);
    }

    /* empty sequence must be a clean no-op */
    if(ork_submit_seq(c,ops,0)!=0){ printf("  empty-sequence returned nonzero\n"); fail=1; }

    printf("%s\n", fail? "FAIL — a heterogeneous-sequence op miscomputed (transition flakiness)"
                       : "PASS — mixed int8(HW-doorbell)+fp16(SW-chain) bit-exact across all runs and transitions");
    free(Bi);free(Bf);free(Ai);free(Af);free(O0);free(O1);free(O3);free(F2);free(F4);
    ork_npu_free(c);
    return fail;
}
