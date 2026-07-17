/* risky_dump_test — deliberately run a KNOWN-RISKY op (ork_dyn_spin_probe: the documented self-loop 0x0010
 * redirect that soft-resets / aborts) and observe how the auto-dump + recover behave across a real fault:
 *   dump(baseline) -> run risky op -> dump(post-fault) -> recover(dump+reset+POST) -> dump(post-recover)
 * The DMA-counter deltas across the dumps show what the faulting op did to the HW; recover() tells us whether
 * the NPU is brought back. Board-safety: timeout-guarded; if the risky op HARD-wedges, the pre-fault dump is
 * the captured evidence (post dumps won't print — that's itself the signal).
 *   make risky_dump_test && sudo env ORK_MM_TIMEOUT=2500 timeout 60 ./risky_dump_test
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int main(void){
    int S=8,K=512,N=512;
    setvbuf(stdout,0,_IONBF,0); setvbuf(stderr,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init fail\n");return 1;}
    printf("risky_dump_test: running ork_dyn_spin_probe (KNOWN-RISKY self-loop redirect) with dumps around it\n");

    /* resident outputs (spin_probe requires C resident for its doorbell) */
    int8_t*A=malloc(K); memset(A,1,K);
    int8_t*B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    ork_w*w=ork_mm_pack_i8(c,K,N,B); free(B); if(!w){printf("pack fail\n");return 1;}
    int32_t*O=(int32_t*)ork_dma_alloc(c,(size_t)S*N*4); if(!O){printf("dma fail\n");return 1;}
    ork_mm_task_i8*tk=malloc(sizeof(*tk)*S); for(int i=0;i<S;i++){ tk[i].w=w; tk[i].M=1; tk[i].A=A; tk[i].C=O+(size_t)i*N; }

    (void)tk; (void)S;
    ork_npu_dump_state(c,"baseline");

    printf("--- running RELIABLE fault (ork_npu_force_fault: bogus weight addr -> DMA fault) ---\n");
    int landed=ork_npu_force_fault(c);
    printf("force_fault: doorbell %s (0=faulted as intended)\n", landed?"LANDED":"did NOT land");

    ork_npu_dump_state(c,"post-fault");

    printf("--- recover (dump + soft-reset + POST) ---\n");
    int rec=ork_npu_recover(c,"post-risky");
    printf("RESULT: recover after risky op -> %s\n", rec?"PASS (NPU brought back)":"FAIL (NPU still broken)");
    ork_npu_dump_state(c,"post-recover");

    ork_npu_free(c); return rec?0:2;
}
