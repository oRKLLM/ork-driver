/* replay_tile — (A) MVP: replay rkllm's CAPTURED fold regcmd for ONE full-K/full-N tile, bit-exact.
 * Feeds rkllm's EXACT regcmd (/tmp/mm_regcmd.txt) + rkllm's EXACT operand bytes (/tmp/mm_A.bin C2-16,
 * /tmp/mm_weight.bin ork_woff) through ork_i8_npu_replay (patches A/B/C addrs, submits task_number=1),
 * then de-tiles the C2-4 output and compares to a CPU reference computed from the SAME input bytes
 * (de-tiled via the confirmed fold layouts). If bit-exact + no wedge => ork can execute rkllm's proven
 * schedule verbatim (capture-replay viable); if it wedges => even rkllm's exact regcmd won't run rebased.
 *   sudo env ORK_MM_TIMEOUT=6 ./replay_tile [M=36] [K=3584] [N=1216]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"
extern int ork_i8_npu_replay(ork_npu*,const unsigned*,int,int,int,int,const signed char*,int,const signed char*,int,int*,int,double*);

/* confirmed fold layouts (width w = M) */
static size_t nc16(int m,int k,int w){ return (size_t)(k/16)*((size_t)w*16)+(size_t)m*16+(k%16); }        /* C2-16 input  */
static size_t woff(int n,int k,int K){ int KT=(K+31)/32; return ((size_t)(n/32)*KT+(k/32))*1024+(size_t)(n%32)*32+(k%32); } /* ork weight */
static size_t c4  (int m,int n,int w){ return (size_t)(n/4)*((size_t)w*4)+(size_t)m*4+(n%4); }            /* C2-4 output  */

static int read_regcmd(const char*path,uint32_t*rc,int max){
    FILE*f=fopen(path,"r"); if(!f){perror(path);return -1;}
    int n=0; while(n<max && fscanf(f,"%x",&rc[n])==1) n++; fclose(f); return n;
}
static signed char* read_bin(const char*path,size_t cap,size_t*got){
    FILE*f=fopen(path,"rb"); if(!f){perror(path);return NULL;}
    fseek(f,0,SEEK_END); long fsz=ftell(f); fseek(f,0,SEEK_SET);
    size_t want=(size_t)fsz; if(cap && want>cap) want=cap;
    signed char*b=malloc(want); size_t rd=fread(b,1,want,f); fclose(f);
    if(got)*got=rd; return b;
}

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int M=argc>1?atoi(argv[1]):36, K=argc>2?atoi(argv[2]):3584, N=argc>3?atoi(argv[3]):1216;
    printf("replay_tile: rkllm captured regcmd, M=%d K=%d N=%d\n",M,K,N);

    uint32_t rc[512]; int rn=read_regcmd("/tmp/mm_regcmd.txt",rc,512);
    if(rn<8){ printf("bad regcmd (rn=%d)\n",rn); return 2; }
    printf("regcmd words=%d\n",rn);

    size_t asz=0,wsz=0;
    signed char*A=read_bin("/tmp/mm_A.bin",0,&asz);          if(!A) return 2;        /* full A span (incl fold padding) */
    signed char*W=read_bin("/tmp/mm_weight.bin",0,&wsz); if(!W){free(A);return 2;}   /* FULL weight span (fold may read strided beyond K*N) */
    printf("fed A=%zu bytes, W=%zu bytes\n",asz,wsz);

    /* CPU reference from the SAME bytes, de-tiled via the confirmed layouts */
    int32_t*Cref=calloc((size_t)M*N,4);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long s=0;
        for(int k=0;k<K;k++) s += (long)A[nc16(m,k,M)] * (long)W[woff(n,k,K)];
        Cref[(size_t)m*N+n]=(int32_t)s; }

    ork_npu*c=ork_npu_init(); if(!c){printf("init (board only)\n");return 77;}
    ork_npu_set_core_budget(c,1);
    /* warm the proven int8 path first (like validate_mfold): pack a dummy weight + run M=8 */
    { int8_t*wd=calloc(wsz,1); int32_t*td=calloc((size_t)8*N,4); int8_t*ad=calloc((size_t)8*K,1);
      ork_w*w=ork_i8_mm_pack(c,K,N,wd); if(w){ ork_i8_mm_run(c,w,8,ad,td); ork_mm_free(c,w); }
      free(wd);free(td);free(ad); }

    int32_t*Craw=calloc((size_t)M*N,4); double us=0;
    int r=ork_i8_npu_replay(c,rc,rn,M,K,N,A,(int)asz,W,(int)wsz,Craw,5,&us);
    if(r){ printf("replay rc=%d (STALL/err — even rkllm's exact regcmd won't run rebased)\n",r); ork_npu_free(c); return 1; }

    /* GROUND TRUTH: ork's raw output buffer vs rkllm's raw output buffer (mm_C.bin). Same regcmd + same
     * input bytes => byte-identical output, with ZERO layout assumptions. This is the decisive replay test. */
    { FILE*f=fopen("/tmp/mm_C.bin","rb");
      if(f){ int32_t*rkC=malloc((size_t)M*N*4); size_t rd=fread(rkC,1,(size_t)M*N*4,f); fclose(f);
        size_t nel=rd/4; long rmm=0,rmx=0; int rf=-1;
        for(size_t i=0;i<nel;i++){ long e=labs((long)Craw[i]-(long)rkC[i]); if(e){rmm++; if(rf<0)rf=(int)i;} if(e>rmx)rmx=e; }
        printf("RAW vs rkllm C (%zu int32 captured): %ld mismatch  maxerr=%ld  %s%s\n",
               nel,rmm,rmx, rmm?"":"*** BYTE-IDENTICAL — ork replays rkllm's fold verbatim! ***",
               (rf>=0)?"":""); if(rmm&&rf>=0) printf("  first raw mismatch at int32 idx %d (ork=%d rkllm=%d)\n",rf,Craw[rf],rkC[rf]);
        free(rkC);
      } else printf("(no /tmp/mm_C.bin for raw compare)\n"); }

    long mm=0,mx=0; int first=-1;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){
        int32_t got=Craw[c4(m,n,M)], ref=Cref[(size_t)m*N+n];
        long e=labs((long)got-ref); if(e){mm++; if(first<0)first=m*N+n;} if(e>mx)mx=e; }
    printf("RESULT vs CPU-ref (de-tiled): %ld/%d mismatch  maxerr=%ld  %.1f us/submit  %s\n",
           mm,M*N,mx,us, mm?"(layout of my de-tile ref may be off — trust the RAW compare)":"*** BIT-EXACT ***");
    if(mm && first>=0) printf("  first mismatch at m=%d n=%d\n",first/N,first%N);
    ork_npu_free(c); free(A);free(W);free(Cref);free(Craw); return mm?1:0;
}
