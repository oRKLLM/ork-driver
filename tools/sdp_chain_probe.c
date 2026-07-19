/* sdp_chain_probe — int8 SDP on the HW chain.
 *  [fwd]        : ewmul as a MIDDLE task walks forward (via ork_npu_chain_progs, desc_slot=138).
 *  [seq-hetero] : STAGE 1 — [matmul->ewmul(SDP middle)->matmul] NONBLOCK on the begin_mc recipe
 *                 (warmed scratch + clean-before), completion via the terminal matmul sentinel; all bit-exact.
 * Board only. */
#include "ork_npu.h"
#include <stdio.h>

int main(void){
    ork_npu *c=ork_npu_init();
    if(!c){ fprintf(stderr,"init failed\n"); return 2; }
    int t0=-1,t1=-1;
    int rc=ork_npu_probe_sdp_chain_fwd(c,&t0,&t1);
    printf("[fwd]        rc=%d  ewmul0(middle)=%s  ewmul1(walked)=%s\n", rc, t0?"OK":"BAD", t1?"OK":"BAD");
    int ok=-1;
    int rc2=ork_npu_probe_seq_hetero(c,&ok);
    printf("[seq-hetero] rc=%d  matmul+ewmul(SDP)+matmul NONBLOCK, terminal sentinel, all bit-exact=%s\n", rc2, ok?"OK":"BAD");
    ork_npu_free(c);
    if(rc==0 && t0 && t1 && rc2==0 && ok){
        printf("VERDICT: int8 SDP rides a NONBLOCK matmul chain (Stage 1 mechanism). PASS\n"); return 0; }
    printf("VERDICT: FAIL (fwd rc=%d t0=%d t1=%d | seq rc=%d ok=%d)\n", rc,t0,t1,rc2,ok);
    return 1;
}
