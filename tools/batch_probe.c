/* tools/batch_probe.c — does batching tasks per RKNPU_SUBMIT cut the per-matmul submit floor?
 * The llama.cpp integration is submit-latency-bound (~134us irreducible floor/matmul, ~168
 * matmuls/token). This times `ntask` identical small int8 matmuls as ntask separate ioctls vs ONE
 * ioctl with task_number=ntask. If batched << unbatched, task-batching is the lever.
 *   make batch_probe && sudo ./batch_probe [ntask] [K] [N]
 */
#include <stdio.h>
#include <stdlib.h>
#include "ork_npu.h"
int main(int argc,char**argv){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed (NPU?)\n");return 1;}
    int K=argc>2?atoi(argv[2]):896, N=argc>3?atoi(argv[3]):896;   /* ~a 0.5B projection */
    printf("submit-batching probe (single core, int8 %dx%d):\n",K,N);
    for(int nt=1;nt<=16;nt*=2){
        if(argc>1 && nt!=atoi(argv[1])) continue;
        double ub=0,b=0; int rc=ork_npu_probe_batch(c,nt,K,N,&ub,&b);
        if(rc){printf("  ntask=%-2d rc=%d (wedge/dims)\n",nt,rc);continue;}
        printf("  ntask=%-2d: %d separate ioctls = %.0f us (%.1f us/task) | 1 ioctl x%d = %.0f us (%.1f us/task)  -> %.2fx\n",
               nt, nt, ub, ub/nt, nt, b, b/nt, ub/b);
    }
    ork_npu_free(c); return 0;
}
