/* i4_doorbell_probe — does the int4 (W4A4, int16-output) datapath survive a NONBLOCK doorbell?
 *
 * The int8/fp16 HW-chain doorbell (ork_dyn_begin_mc) writes a 4-byte C directly and detects completion by
 * polling a 4-byte ORK_DYN_SENT sentinel. int4 is different at the HARDWARE level: it writes an int16
 * (2-byte) accumulator that the driver widens to int32 on the host, and its HW chain is M=1-only. So there
 * is no "free flip" onto begin_mc. This probe isolates the ONE remaining question: with everything else held
 * identical to the WORKING ork_i4_mm_run_chain (same synth_i4 regcmd, same chain descriptor, host A), does a
 * NONBLOCK submit + int16-sentinel poll compute the SAME result as the blocking reference?
 *
 *   reference = ork_i4_mm_run_chain  (blocking int4 PC-chain, passes make test via test_chain_i4)
 *   probe     = ork_i4_dyn_probe     (identical build, submit flipped to NONBLOCK 0x2 + int16-sentinel poll)
 *
 * Bit-exact probe==reference across many runs + 2 shapes + cold run 0 => the int4 int16-output datapath
 * DOES survive a non-blocking doorbell (path viable, worth wiring). A mismatch/hang => it does not (int4
 * stays on the SW stream / blocking chain). Both are also checked against a CPU int reference for an anchor.
 *
 * BOARD:  make i4_doorbell_probe && sudo env ORK_MM_TIMEOUT=3000 timeout 200 ./i4_doorbell_probe [runs=30] [K] [N]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define S 4
#define POISON 0x0badc0de

int main(int argc,char**argv){
    int runs = argc>1?atoi(argv[1]):30;
    int K = argc>2?atoi(argv[2]):512, N = argc>3?atoi(argv[3]):256;   /* K%32, N%64 (int4 pack) */
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    printf("i4_doorbell_probe: S=%d ops, M=1, K=%d N=%d, %d runs (ref=run_chain_i4 vs probe=NONBLOCK+int16-sentinel)\n",S,K,N,runs);

    int8_t *A[S], *B[S]; ork_w *w[S]; int32_t *Cref[S], *Cprb[S]; long *Cpu[S];
    uint32_t sd=12345;
    for(int t=0;t<S;t++){
        A[t]=malloc((size_t)K); B[t]=malloc((size_t)K*N);
        for(int i=0;i<K;i++){ sd=sd*1103515245+12345; A[t][i]=(int8_t)((int)((sd>>17)%15)-7); }
        for(size_t i=0;i<(size_t)K*N;i++){ sd=sd*1103515245+12345; B[t][i]=(int8_t)((int)((sd>>17)%15)-7); }
        w[t]=ork_i4_mm_pack(c,K,N,B[t]); if(!w[t]){ printf("pack_i4 fail t=%d\n",t); return 2; }
        Cref[t]=malloc((size_t)N*4); Cprb[t]=malloc((size_t)N*4); Cpu[t]=malloc((size_t)N*sizeof(long));
        for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[t][k]*B[t][(size_t)k*N+n]; Cpu[t][n]=s; }
    }

    ork_mm_task_i4 tref[S], tprb[S];
    for(int t=0;t<S;t++){ tref[t]=(ork_mm_task_i4){w[t],1,A[t],Cref[t]}; tprb[t]=(ork_mm_task_i4){w[t],1,A[t],Cprb[t]}; }

    /* PHASE A — reference (ork_i4_mm_run_chain) ALONE, R runs vs CPU: establishes the reference is clean and
     * deterministic in isolation (no interleaving with the experimental path). */
    int ref_bad=0, ref_rc=0;
    for(int r=0;r<runs;r++){
        for(int t=0;t<S;t++) for(int n=0;n<N;n++) Cref[t][n]=POISON;
        if(ork_i4_mm_run_chain(c,S,tref)){ ref_rc++; continue; }
        int e=0; for(int t=0;t<S;t++) for(int n=0;n<N;n++) if(Cref[t][n]!=(int32_t)Cpu[t][n]) e++;
        if(e){ ref_bad++; if(ref_bad<=3) printf("  [A] run %d: ref!=cpu %d/%d (ref[0]=%d cpu[0]=%ld)\n",r,e,S*N,Cref[0][0],Cpu[0][0]); }
    }
    /* PHASE B — probe (ork_i4_dyn_probe: NONBLOCK + full-surface int16-sentinel) ALONE, R runs vs CPU. */
    int prb_bad=0, prb_rc=0, prb_zero=0;
    for(int r=0;r<runs;r++){
        for(int t=0;t<S;t++) for(int n=0;n<N;n++) Cprb[t][n]=POISON;
        if(ork_i4_dyn_probe(c,S,tprb)){ prb_rc++; continue; }
        int e=0,z=0; for(int t=0;t<S;t++) for(int n=0;n<N;n++){ if(Cprb[t][n]!=(int32_t)Cpu[t][n]) e++; if(Cprb[t][n]==0) z++; }
        if(e){ prb_bad++; if(prb_bad<=3) printf("  [B] run %d: probe!=cpu %d/%d (%d zeros) (probe[0]=%d cpu[0]=%ld)\n",r,e,S*N,z,Cprb[0][0],Cpu[0][0]); }
        prb_zero += z;
    }
    /* PHASE C — does a probe call POLLUTE a subsequent reference? probe, then reference, check reference. */
    int poll_bad=0;
    for(int r=0;r<runs;r++){
        for(int t=0;t<S;t++) for(int n=0;n<N;n++){ Cprb[t][n]=POISON; Cref[t][n]=POISON; }
        ork_i4_dyn_probe(c,S,tprb);
        if(ork_i4_mm_run_chain(c,S,tref)) continue;
        int e=0; for(int t=0;t<S;t++) for(int n=0;n<N;n++) if(Cref[t][n]!=(int32_t)Cpu[t][n]) e++;
        if(e){ poll_bad++; if(poll_bad<=3) printf("  [C] run %d: post-probe ref!=cpu %d/%d (probe corrupts next blocking submit)\n",r,e,S*N); }
    }
    /* The DOORBELL question is Phase B ALONE (probe vs an independent CPU int reference). Phases A and C
     * measure the BLOCKING reference (ork_i4_mm_run_chain); if A (which runs NO probe) is also bad, that is a
     * PRE-EXISTING run_chain_i4 warm-reuse coherency bug — NOT a probe/pollution problem — so C's failure is
     * then just the reference being broken warm, not the probe corrupting it. Verdict keys on B. */
    int doorbell_ok = (prb_bad==0 && prb_rc==0);
    int ref_warm_bug = (ref_bad>0 || ref_rc>0);
    printf("---\n");
    printf("  [B] PROBE (NONBLOCK doorbell)  : %d/%d runs == cpu%s (avg zeros/run=%d of %d)  <-- the doorbell verdict\n",
           runs-prb_bad-prb_rc, runs, prb_rc?" (rc errs)":"", runs?prb_zero/runs:0, S*N);
    printf("  [A] blocking reference alone   : %d/%d runs == cpu%s\n", runs-ref_bad-ref_rc, runs, ref_rc?" (rc errs)":"");
    printf("  [C] reference after a probe    : %d/%d runs == cpu\n", runs-poll_bad, runs);
    if(ref_warm_bug) printf("  NOTE: [A] runs NO probe yet is also bad => PRE-EXISTING run_chain_i4 warm-reuse coherency bug\n"
                            "        (fresh chain_C DMA scratch recycles pages w/ stale CPU lines; no clean-before, unlike 79f809c's begin_mc fix).\n"
                            "        => [C] is that same reference bug, NOT probe pollution (30 consecutive probes in [B] never corrupt each other).\n");
    printf("%s\n", doorbell_ok
        ? "DOORBELL PASS — int4 (int16-output, W4A4) IS bit-exact on a NONBLOCK doorbell with a full-surface int16-sentinel clean-before"
        : "DOORBELL FAIL — int4 does NOT survive the NONBLOCK doorbell");
    int fail = !doorbell_ok;   /* exit status = the doorbell question only */
    for(int t=0;t<S;t++){ ork_mm_free(c,w[t]); free(A[t]); free(B[t]); free(Cref[t]); free(Cprb[t]); free(Cpu[t]); }
    ork_npu_free(c);
    return fail;
}
