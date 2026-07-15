/* overlap_prof — Tier 11: the "zero-time router" thesis in one number. Runs a real CPU GEMV (stand-in for MoE
 * routing math) in the SHADOW of an async NPU op (nonblock submit + doorbell poll) and asks: is the CPU work
 * swallowed for free (overlap wall ~= max(npu,cpu)), or does shared LPDDR4X bandwidth contention serialize it
 * (wall -> npu+cpu)? Sweeps CPU work from router-sized up past the NPU op. BOARD:
 *   sudo env ORK_MM_TIMEOUT=4000 ./overlap_prof [iters] */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char**argv){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    int iters=argc>1?atoi(argv[1]):10;
    int M=64,K=4096,N=4096;                        /* heavy NPU op (~2750us) — stands in for an attention layer */
    int reps[]={1,4,16,64}; int nr=4;
    printf("CPU-router-in-NPU-shadow overlap (%d iters, NPU op M=%d K=%d N=%d):\n",iters,M,K,N);
    printf(" cpu_reps | npu_solo us | cpu_solo us | overlap us |  sum us | hidden%% | ok\n");
    int fail=0;
    for(int r=0;r<nr;r++){ double ns=0,cs=0,ow=0; int ok=0;
        int rc=ork_npu_overlap_prof(c,M,K,N,reps[r],iters,&ns,&cs,&ow,&ok);
        if(rc){printf(" %8d | rc=%d\n",reps[r],rc); continue;}
        double sum=ns+cs, mn=ns<cs?ns:cs, hidden=100.0*(sum-ow)/(mn>0?mn:1);
        if(hidden>100)hidden=100; if(hidden<0)hidden=0;
        printf(" %8d | %11.1f | %11.1f | %10.1f | %7.1f | %6.1f%% | %s\n",reps[r],ns,cs,ow,sum,hidden,ok?"ok":"BAD");
        if(!ok)fail=1;
    }
    printf("hidden%%: 100 = CPU work fully swallowed by NPU compute (free overlap); 0 = fully serialized (bandwidth contention)\n");
    ork_npu_free(c); return fail;
}
