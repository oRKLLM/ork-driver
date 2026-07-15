/* doorbell_prof — Tier 11: measure a REAL N-op pipeline wall, BLOCKING vs NONBLOCK+DRAM-doorbell busy-poll.
 * Each op is an int8 matmul (all-ones -> every output = K). Blocking submit waits (~130us floor + compute)/op;
 * nonblock submit returns ~5us then the CPU busy-polls the output sentinel until the NPU overwrites it (spin,
 * no sleep/wake). Reports per-op us + speedup + correctness for a sweep of shapes. BOARD:
 *   sudo env ORK_MM_TIMEOUT=4000 ./doorbell_prof [iters]   (default 30) */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char**argv){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    int iters=argc>1?atoi(argv[1]):30;
    int sh[][3]={{1,512,512},{1,2048,2048},{64,512,512},{64,2048,2048},{64,4096,4096}}; int ns=5;
    printf("N-op pipeline wall (%d serial ops): BLOCKING vs NONBLOCK+doorbell-poll\n",iters);
    printf("   M    K     N   | block us/op | nonblk us/op | speedup | ok\n");
    int fail=0;
    for(int s=0;s<ns;s++){ int M=sh[s][0],K=sh[s][1],N=sh[s][2];
        double bu=0,nu=0; int okb=0,okn=0;
        int rc=ork_npu_doorbell_prof(c,M,K,N,iters,&bu,&nu,&okb,&okn);
        if(rc){ printf("   %d %5d %5d | rc=%d (unsupported/err)\n",M,K,N,rc); continue; }
        printf("   %2d %5d %5d | %9.1f | %10.1f | %6.2fx | %s%s\n",
               M,K,N,bu,nu, nu>0?bu/nu:0.0, okb?"blk-ok":"blk-BAD", okn?" nb-ok":" nb-BAD");
        if(!okb||!okn)fail=1;
    }
    printf("%s\n", fail?"CHECK — a correctness flag failed (poll returned early / async race?)":"ALL correct (async doorbell pipeline bit-valid)");
    ork_npu_free(c); return fail;
}
