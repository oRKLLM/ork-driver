/* fold_chain_multi — #39 weight-reuse timing probe. Replays rkllm's real chain tasks (in order, from a file of
 * P lines x 232 hex words) as one task_number=Pn weight-resident chain and reports us/submit, sweeping Pn.
 * The timing SLOPE answers: does each task re-DMA the K*N weight (~linear, ~400us/task) or reuse the resident
 * copy once a loader has populated CBUF (sublinear)? Zeroed operands (timing, not correctness).
 *   sudo env ORK_MM_TIMEOUT=500 ./fold_chain_multi [file] [K] [N] [wmax]   (default /tmp/mm_chain_all.txt 3584 1216 36)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"
extern int ork_npu_mfold_chain_multi(ork_npu*,int,int,int,int,const unsigned*,int,int,double*);
int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    const char*rf=argc>1?argv[1]:"/tmp/mm_chain_all.txt";
    int K=argc>2?atoi(argv[2]):3584, N=argc>3?atoi(argv[3]):1216, wmax=argc>4?atoi(argv[4]):36;
    static uint32_t rc[64*232]; int P=0;
    FILE*f=fopen(rf,"r"); if(!f){perror(rf);return 2;}
    char*line=NULL; size_t lc=0;
    while(P<64 && getline(&line,&lc,f)>0){
        int nn=0; char*p=line; while(nn<232){ char*e; long v=strtol(p,&e,16); if(e==p)break; rc[P*232+nn]=(uint32_t)v; nn++; p=e; }
        if(nn==232) P++;
    }
    fclose(f);
    printf("fold_chain_multi: loaded %d tasks (rn=232) K=%d N=%d wmax=%d\n",P,K,N,wmax);
    if(P<1) return 2;
    ork_npu*c=ork_npu_init(); if(!c){printf("init (board only)\n");return 77;}
    ork_npu_set_core_budget(c,1);
    { int8_t*wd=calloc((size_t)K*N,1);int32_t*td=calloc((size_t)8*N,4);int8_t*ad=calloc((size_t)8*K,1);  /* warm the int8 mode */
      ork_w*ww=ork_mm_pack_i8(c,K,N,wd); if(ww){ork_mm_run_i8(c,ww,8,ad,td);ork_mm_free(c,ww);} free(wd);free(td);free(ad); }
    printf(" Pn |  us/submit | us/task | note\n");
    int steps[]={1,2,4,8,12,16,20,21}; double prev=0;
    for(int si=0; si<(int)(sizeof steps/sizeof steps[0]); si++){
        int Pn=steps[si]; if(Pn>P) break;
        double us=0; int r=ork_npu_mfold_chain_multi(c,Pn,wmax,K,N,rc,232,3,&us);
        if(r){ printf(" %2d |  STALL/err rc=%d\n",Pn,r); break; }
        printf(" %2d | %9.1f | %7.1f | marginal/task since prev = %.1f\n", Pn, us, us/Pn, prev>0?(us-prev)/(Pn - (si>0?steps[si-1]:0)):us/Pn);
        prev=us;
    }
    ork_npu_free(c); return 0;
}
