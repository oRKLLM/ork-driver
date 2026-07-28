/* synth_dump.c — #39: dump ork's synth_i8 regcmd for (mc,K,N) to diff vs rkllm's captured regcmd. No submit. */
#include <stdio.h>
#include <stdlib.h>
#include "ork_npu.h"
int main(int argc,char**argv){
    if(argc<4){fprintf(stderr,"usage: %s mc K N\n",argv[0]);return 2;}
    int mc=atoi(argv[1]),K=atoi(argv[2]),N=atoi(argv[3]);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 77;}
    unsigned out[512]; int n=ork_npu_synth_i8_dump(c,mc,K,N,out,512);
    if(n<0){printf("dump rc=%d\n",n);return 1;}
    for(int i=0;i<n;i++) printf("%08x ",out[i]);
    printf("\n");
    ork_npu_free(c); return 0;
}
