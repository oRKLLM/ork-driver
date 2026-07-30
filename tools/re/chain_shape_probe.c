/* chain_shape_probe — map the run_chain_i8 shape envelope (the rc=-3 limit found by ws_xition_probe).
 * Code says run_chain_i8_impl rejects a chain link with (w->K % 512 != 0 || w->K > 4096) -> -3, because
 * the per-link full-K single submit uses synth_i8(sched=1) whose 0x1040 K-reduction schedule is only
 * valid there; other K must fall back to per-task run_i8 (which K-splits). This confirms the boundary
 * on-board and checks that the CHAINABLE shapes still compute correctly (all-ones => C==K).
 *
 * BOARD: sudo env ORK_MM_TIMEOUT=3000 timeout 180 ./chain_shape_probe
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int main(void){
    setvbuf(stdout,0,_IONBF,0);
    ork_npu *c = ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    int Ks[] = {128,256,384,512,768,1024,1536,2048,3072,3584,4096,4608,8192};
    int N=64, M=8, S=4;
    printf("chain_shape_probe: N=%d M=%d S=%d  (expect chain OK iff K%%512==0 && K<=4096)\n", N,M,S);
    printf("   K     pack  run_chain_i8 rc   verdict\n");
    for(size_t j=0;j<sizeof Ks/sizeof*Ks;j++){
        int K=Ks[j];
        int8_t *B = malloc((size_t)K*N); memset(B,1,(size_t)K*N);
        ork_w *w = ork_mm_pack_i8(c,K,N,B); free(B);
        if(!w){ printf("  %5d  FAIL  (pack rejected)\n", K); continue; }
        int8_t *A = malloc((size_t)M*K); memset(A,1,(size_t)M*K);
        int32_t *Cs[8];
        ork_mm_task_i8 tk[8];
        for(int i=0;i<S;i++){ Cs[i]=calloc((size_t)M*N,4); tk[i]=(ork_mm_task_i8){w,M,A,Cs[i]}; }
        int rc = ork_mm_run_chain_i8(c,S,tk);
        int bad=0;
        if(rc==0){ for(int i=0;i<S;i++){ for(size_t e=0;e<(size_t)M*N;e++){ if(Cs[i][e]!=K){bad++;break;} } } }
        const char *v;
        if(rc==0)       v = bad ? "CHAINED but WRONG C!" : "chained OK, C==K";
        else if(rc==-3) v = "rc=-3 (K-schedule guard -> run_i8 fallback)";
        else if(rc==-2) v = "rc=-2 (bad arg)";
        else            v = "rc<0 (other)";
        int expect_ok = (K%512==0 && K<=4096);
        printf("  %5d   ok   %14d   %s%s\n", K, rc, v,
               (rc==0)==expect_ok ? "" : "   <-- MISMATCH vs predicted envelope");
        for(int i=0;i<S;i++) free(Cs[i]); free(A);
    }
    printf("DONE\n");
    return 0;
}
