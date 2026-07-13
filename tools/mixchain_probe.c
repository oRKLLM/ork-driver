/* mixchain_probe — gate for the fused SSD-scan graph: does a FP16 matmul task and an INT16 silu-SDP task
 * coexist correctly in ONE PC-chained submit? (precision is per-task regcmd on the shared 2-byte datapath;
 * the descriptor-poisoning wedge is fixed, so this is a correctness confirmation.) Board only.
 *   make mixchain_probe && sudo ./mixchain_probe
 */
#include "ork_npu.h"
#include <stdio.h>
int main(void){
    ork_npu *c=ork_npu_init(); if(!c){ fprintf(stderr,"no NPU — skip\n"); return 0; }
    int mm_ok=0, silu_ok=0; double us=0;
    int rc=ork_ssd_probe_mixchain(c,&mm_ok,&silu_ok,&us);
    fprintf(stderr,"[mixchain] rc=%d  fp16-matmul-in-chain=%s  int16-silu-in-chain=%s  %.0fus\n",
            rc, mm_ok?"OK":"BAD", silu_ok?"OK":"BAD", us);
    int fail = (rc!=0)||!mm_ok||!silu_ok;
    fprintf(stderr, fail?"\nMIXCHAIN_PROBE: FAIL (fp16+int16 do NOT coexist in one chain)\n"
                        :"\nMIXCHAIN_PROBE: PASS (fp16 matmul + int16 SDP coexist in one submit)\n");
    ork_npu_free(c);
    return fail?1:0;
}
