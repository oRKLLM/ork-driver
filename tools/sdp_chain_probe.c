/* sdp_chain_probe — does int8 SDP (ewmul) HW-chain forward via its chain-native descriptor (word 138)?
 * Ports the standalone ewmul into the proven ork_npu_chain_progs (desc_slot=138). Board only. */
#include "ork_npu.h"
#include <stdio.h>

int main(void){
    ork_npu *c=ork_npu_init();
    if(!c){ fprintf(stderr,"init failed\n"); return 2; }
    int t0=-1,t1=-1;
    int rc=ork_npu_probe_sdp_chain_fwd(c,&t0,&t1);
    printf("rc=%d  ewmul0(middle,carries-fwd-desc)=%s  ewmul1(chain-walked-fwd)=%s\n",
           rc, t0?"OK":"BAD", t1?"OK":"BAD");
    ork_npu_free(c);
    if(rc!=0){ printf("VERDICT: submit failed rc=%d (hang/wedge => int8 SDP does NOT HW-chain)\n",rc); return 1; }
    if(t0&&t1){ printf("VERDICT: int8 SDP HW-CHAINS forward — a middle SDP op walks to the next task. PASS\n"); return 0; }
    printf("VERDICT: submit ok but output wrong (t0=%d t1=%d) — chained but miscomputed\n",t0,t1);
    return 1;
}
