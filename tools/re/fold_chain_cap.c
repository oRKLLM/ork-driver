/* fold_chain_cap — #39: chain the CAPTURED bit-exact M=8 mfold tile as P weight-resident tasks and VERIFY.
 * Loads rkllm's captured width-8 fold regcmd (default /tmp/mm_regcmd_m8.txt, the one validate_layout proved
 * 0/9728 bit-exact), packs A as P tiles of 8 rows (C2-16), W as ork_woff (shared/resident), runs one
 * task_number=P submit via ork_npu_mfold_chain_cap, de-tiles C (C2-4) per tile, compares CPU int8 ref.
 * The decisive test for "at M>8 you chain additional M=8 tasks".
 *   sudo env ORK_MM_TIMEOUT=500 ./fold_chain_cap [P] [K] [N] [regcmd]     (default P=2 K=3584 N=1216, M=8*P)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"
extern int ork_npu_mfold_chain_cap(ork_npu*,int,int,int,int,const unsigned*,int,const int8_t*,const int8_t*,int32_t*,int,double*);
static uint32_t rng=0x1234567u; static int r7(void){ rng=rng*1664525u+1013904223u; return (int)((rng>>25)%7)-3; }
static size_t nc16(int m,int c,int w){ return (size_t)(c/16)*((size_t)w*16)+(size_t)m*16+(c%16); }    /* C2-16 input, width w */
static size_t woff(int n,int k,int K){ int KT=(K+31)/32; return ((size_t)(n/32)*KT+(k/32))*1024+(size_t)(n%32)*32+(k%32); }
static size_t c4(int m,int n,int w){ return (size_t)(n/4)*((size_t)w*4)+(size_t)m*4+(n%4); }           /* C2-4 output, width w */

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int P=argc>1?atoi(argv[1]):2, w=argc>2?atoi(argv[2]):8, K=argc>3?atoi(argv[3]):3584, N=argc>4?atoi(argv[4]):1216;
    const char*rf=argc>5?argv[5]:"/tmp/mm_regcmd_m8.txt";
    const int M=P*w;
    /* load captured tile regcmd */
    static uint32_t rc[512]; int rn=0;
    FILE*f=fopen(rf,"r"); if(!f){ perror(rf); return 2; }
    while(rn<512 && fscanf(f,"%x",&rc[rn])==1) rn++;
    fclose(f);
    printf("fold_chain_cap: P=%d w=%d (M=%d) K=%d N=%d  captured rn=%d (%s)\n",P,w,M,K,N,rn,rf);
    if(rn<108){ printf("regcmd too short\n"); return 2; }

    int8_t *A=malloc((size_t)M*K),*W=malloc((size_t)K*N); int32_t*Cref=calloc((size_t)M*N,4);
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)r7();
    for(size_t i=0;i<(size_t)K*N;i++) W[i]=(int8_t)r7();
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[(size_t)m*K+k]*W[(size_t)k*N+n]; Cref[(size_t)m*N+n]=(int32_t)s; }
    /* pack A: P tiles, tile t = rows [t*8, t*8+8) as width-8 C2-16 */
    int8_t *Ap=calloc((size_t)P*w*K,1);
    for(int t=0;t<P;t++)for(int m=0;m<w;m++)for(int k=0;k<K;k++) Ap[(size_t)t*w*K + nc16(m,k,w)] = A[(size_t)(t*w+m)*K+k];
    int8_t *Wp=calloc((size_t)K*N,1); for(int k=0;k<K;k++)for(int n=0;n<N;n++) Wp[woff(n,k,K)]=W[(size_t)k*N+n];
    int32_t *Craw=calloc((size_t)P*w*N,4);

    ork_npu*c=ork_npu_init(); if(!c){printf("init (board only)\n");return 77;}
    ork_npu_set_core_budget(c,1);
    { ork_w*ww=ork_mm_pack_i8(c,K,N,W); if(ww){int32_t*t=calloc((size_t)8*N,4);int8_t*a8=calloc((size_t)8*K,1);ork_mm_run_i8(c,ww,8,a8,t);free(t);free(a8);ork_mm_free(c,ww);} }  /* warm proven path */

    double us=0; int r=ork_npu_mfold_chain_cap(c,P,w,K,N,rc,rn,Ap,Wp,Craw,5,&us);
    if(r){ printf("chain_cap rc=%d (STALL/err)\n",r); ork_npu_free(c); return 1; }
    long mm=0,mx=0; int first=-1;
    for(int t=0;t<P;t++)for(int m=0;m<w;m++)for(int n=0;n<N;n++){
        int32_t got=Craw[(size_t)t*w*N + c4(m,n,w)], ref=Cref[(size_t)(t*w+m)*N+n];
        long e=labs((long)got-ref); if(e){mm++; if(first<0)first=(t*w+m)*N+n;} if(e>mx)mx=e; }
    printf("RESULT cap-chain: %ld/%d mismatch  maxerr=%ld  %.1f us/submit (%d tasks, M=%d)  %s\n",
           mm,M*N,mx,us,P,M, mm?"MISMATCH":"*** BIT-EXACT — captured tile chains, weight-resident at M>8! ***");
    ork_npu_free(c); return mm?1:0;
}
